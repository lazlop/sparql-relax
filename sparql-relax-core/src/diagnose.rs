//! Ablation-style error diagnosis, ported from the strategy in
//! `sparql_prune.py`: remove some of the query's BGP triples and re-run the
//! rest. If the removed triples never actually hold (jointly) between any
//! of the resulting variable bindings, they're the culprits blocking the
//! original query.
//!
//! This module only identifies *which* triple pattern(s) are responsible —
//! it does no variable binding/resolution beyond the `Yes`/`No` truth check
//! needed to confirm a combination is a genuine culprit. Resolving what a
//! culprit's variables are actually bound to (needed to search for a
//! relaxed path) is [`crate::relax`]'s job, done only for culprits that
//! diagnosis has already found — so a plain diagnosis-only call never pays
//! for binding work it won't use.
//!
//! `depth` controls how many triples are removed *together*: at depth 1,
//! every single triple is tried alone (today's baseline case); if none of
//! those unblock the query, depth 2 tries every *pair* of triples removed
//! together, then depth 3 every triple of three, and so on up to `depth` —
//! stopping as soon as some combination at the current size unblocks the
//! query, or once there are no more triples left to combine. This catches
//! queries where no single triple is broken, but two (or more) are jointly
//! responsible for the empty/wrong result.
//!
//! `FILTER` expressions get the same one-at-a-time ablation treatment
//! (remove one, re-run, see if that unblocks/grows the result set) but
//! without the extra graph-truth check triples get: there's no way to ask
//! Oxigraph "does this arbitrary expression hold" short of evaluating SPARQL
//! expressions ourselves, so a strict row-count increase after removal is
//! the signal. Filters are only ever reported, never relaxed, and `depth`
//! does not apply to them (queries rarely have enough interacting filters
//! for combining removals to matter, unlike triples).

use crate::algebra::{collect_bgp_triples, collect_filters, pattern_of, remove_filter, remove_triple, with_pattern};
use crate::error::{RelaxError, Result};
use oxigraph::model::{GraphNameRef, NamedOrBlankNode, Term};
use oxigraph::sparql::{QueryResults, QuerySolution, SparqlEvaluator};
use oxigraph::store::Store;
use spargebra::Query;
use spargebra::SparqlParser;
use spargebra::algebra::Expression;
use spargebra::term::{NamedNodePattern, TermPattern, TriplePattern};

/// One or more BGP triples whose *joint* removal unblocks the query, and
/// which never jointly hold for any binding of the rest of the query — the
/// combination most likely responsible for the original query's empty/wrong
/// result. Has one triple unless a single-triple removal wasn't enough to
/// unblock the query and a larger combination (see `depth` on [`diagnose`])
/// was needed.
pub struct Culprit {
    pub triples: Vec<TriplePattern>,
    /// The combination size at which this culprit was found (1 = a single
    /// broken triple, 2 = a pair jointly responsible, ...).
    pub depth: usize,
}

/// A `FILTER` expression whose removal strictly grew the result set, i.e. it
/// was excluding rows that the rest of the query would otherwise return.
pub struct FilterCulprit {
    pub expression: Expression,
    pub row_count_without_filter: usize,
}

pub struct Diagnosis {
    pub original_row_count: usize,
    pub culprits: Vec<Culprit>,
    pub filter_culprits: Vec<FilterCulprit>,
}

/// Default for `depth`/`ablation_depth`: single triples are tried first, and
/// it rarely takes more than a couple of jointly-broken triples to explain a
/// bad query, so this stays small to keep the (combinatorial) search cheap.
pub const DEFAULT_ABLATION_DEPTH: usize = 3;

/// Diagnoses `query_text` against `store` with [`DEFAULT_ABLATION_DEPTH`].
/// A convenient default for callers who don't need to tune the search.
pub fn diagnose_default(query_text: &str, store: &Store) -> Result<Diagnosis> {
    diagnose(query_text, store, DEFAULT_ABLATION_DEPTH)
}

/// Diagnoses `query_text` against `store`. Only `SELECT` queries are
/// supported.
///
/// `depth` bounds how many triples may be removed together while searching
/// for a culprit combination: single triples are always tried first (depth
/// 1); if none of those unblock the query, pairs are tried (depth 2), then
/// triples of three (depth 3), and so on up to `depth`. The search stops
/// escalating as soon as some combination at the current size unblocks the
/// query (there may be more than one such combination; all are returned),
/// or once the combination size would exceed the number of candidate
/// triples. `depth = 1` reproduces single-triple-only diagnosis.
pub fn diagnose(query_text: &str, store: &Store, depth: usize) -> Result<Diagnosis> {
    let query = SparqlParser::new().parse_query(query_text)?;
    if !matches!(query, Query::Select { .. }) {
        return Err(RelaxError::UnsupportedQueryForm("SELECT"));
    }
    let pattern = pattern_of(&query);

    let original_rows = run_select(query_text, store)?;

    let triple_candidates: Vec<TriplePattern> = collect_bgp_triples(pattern)
        .into_iter()
        .filter(|t| matches!(t.predicate, NamedNodePattern::NamedNode(_)))
        .collect();
    let filter_candidates = collect_filters(pattern);
    if triple_candidates.is_empty() && filter_candidates.is_empty() {
        return Err(RelaxError::NoTriples);
    }

    let mut culprits = Vec::new();
    let max_depth = depth.max(1);
    for k in 1..=max_depth {
        if k > triple_candidates.len() {
            break; // no more triples left to combine
        }

        let mut found_at_this_depth = false;
        for combo in combinations(&triple_candidates, k) {
            if !is_culprit_combo(&query, pattern, &combo, store) {
                continue;
            }
            found_at_this_depth = true;
            culprits.push(Culprit { triples: combo, depth: k });
        }

        if found_at_this_depth {
            break; // don't escalate to a larger combination size
        }
    }

    let mut filter_culprits = Vec::new();
    for expression in filter_candidates {
        let Some(reduced_pattern) = remove_filter(pattern, &expression) else { continue };
        let reduced_text = with_pattern(&query, reduced_pattern).to_string();
        let Ok(reduced_rows) = run_select(&reduced_text, store) else { continue };
        // Removing a FILTER can only ever keep or grow the result set (it's
        // a pure restriction), so a strict increase means this filter was
        // actually excluding rows.
        if reduced_rows.len() > original_rows.len() {
            filter_culprits.push(FilterCulprit { expression, row_count_without_filter: reduced_rows.len() });
        }
    }

    Ok(Diagnosis { original_row_count: original_rows.len(), culprits, filter_culprits })
}

/// Whether removing every triple in `combo` together unblocks the query,
/// and no single row satisfies every triple in `combo` jointly (i.e. the
/// combination is genuinely, jointly responsible — not just incidentally
/// relaxing enough constraints for something else to match).
fn is_culprit_combo(query: &Query, pattern: &spargebra::algebra::GraphPattern, combo: &[TriplePattern], store: &Store) -> bool {
    let Some(reduced_pattern) = combo.iter().try_fold(pattern.clone(), |p, t| remove_triple(&p, t)) else {
        return false;
    };
    let reduced_text = with_pattern(query, reduced_pattern).to_string();
    let Ok(reduced_rows) = run_select(&reduced_text, store) else { return false };
    if reduced_rows.is_empty() {
        return false; // removing this combination alone doesn't unblock anything
    }

    let jointly_holds_somewhere =
        reduced_rows.iter().any(|row| combo.iter().all(|t| triple_holds_for_row(store, t, row)));
    !jointly_holds_somewhere
}

/// All distinct size-`k` combinations of `items` (order-independent, no
/// repeats), preserving `items`' original order within each combination.
fn combinations<T: Clone>(items: &[T], k: usize) -> Vec<Vec<T>> {
    if k == 0 {
        return vec![Vec::new()];
    }
    if k > items.len() {
        return Vec::new();
    }
    let mut result = Vec::new();
    for i in 0..=(items.len() - k) {
        for mut rest in combinations(&items[i + 1..], k - 1) {
            rest.insert(0, items[i].clone());
            result.push(rest);
        }
    }
    result
}

pub(crate) fn resolve_term_pattern(term: &TermPattern, row: &QuerySolution) -> Option<Term> {
    match term {
        TermPattern::NamedNode(n) => Some(Term::NamedNode(n.clone())),
        TermPattern::BlankNode(b) => Some(Term::BlankNode(b.clone())),
        TermPattern::Literal(l) => Some(Term::Literal(l.clone())),
        TermPattern::Variable(v) => row.get(v.as_str()).cloned(),
    }
}

/// Whether `triple`'s (possibly variable) subject/object, resolved against
/// `row`, actually holds as a real triple in `store`. An unresolved
/// (unbound) side is treated as a wildcard.
fn triple_holds_for_row(store: &Store, triple: &TriplePattern, row: &QuerySolution) -> bool {
    let predicate = match &triple.predicate {
        NamedNodePattern::NamedNode(p) => p,
        NamedNodePattern::Variable(_) => return true,
    };

    let subject_term = resolve_term_pattern(&triple.subject, row);
    let subject_filter = match &subject_term {
        None => None,
        Some(Term::Literal(_)) => return false, // literals can never be subjects
        Some(Term::NamedNode(n)) => Some(NamedOrBlankNode::NamedNode(n.clone())),
        Some(Term::BlankNode(b)) => Some(NamedOrBlankNode::BlankNode(b.clone())),
    };
    let object_filter = resolve_term_pattern(&triple.object, row);

    store
        .quads_for_pattern(
            subject_filter.as_ref().map(|s| s.as_ref()),
            Some(predicate.as_ref()),
            object_filter.as_ref().map(|o| o.as_ref()),
            Some(GraphNameRef::DefaultGraph),
        )
        .flatten()
        .next()
        .is_some()
}

pub(crate) fn run_select(query_text: &str, store: &Store) -> Result<Vec<QuerySolution>> {
    let results = SparqlEvaluator::new().parse_query(query_text)?.on_store(store).execute()?;
    match results {
        QueryResults::Solutions(iter) => {
            let mut rows = Vec::new();
            for solution in iter {
                rows.push(solution?);
            }
            Ok(rows)
        }
        _ => Err(RelaxError::UnsupportedQueryForm("SELECT")),
    }
}
