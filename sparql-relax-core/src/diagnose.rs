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

use crate::algebra::{ask_query, collect_bgp_triples, collect_filters, pattern_of, remove_filter, remove_triple, with_pattern};
use crate::error::{RelaxError, Result};
use oxigraph::model::{GraphNameRef, NamedOrBlankNode, Term};
use oxigraph::sparql::{CancellationToken, QueryEvaluationError, QueryResults, QuerySolution, SparqlEvaluator};
use oxigraph::store::Store;
use rayon::prelude::*;
use spargebra::Query;
use spargebra::SparqlParser;
use spargebra::algebra::{Expression, GraphPattern};
use spargebra::term::{NamedNodePattern, TermPattern, TriplePattern};
use std::thread;
use std::time::{Duration, Instant};

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

/// Default for `timeout`: a single ablation check is normally well under
/// this; five seconds is enough headroom for that while still bounding the
/// rare reduced query that turns into a much larger join than the original
/// (see the `timeout` docs on [`diagnose`]).
pub const DEFAULT_ABLATION_TIMEOUT: Duration = Duration::from_secs(5);

/// Diagnoses `query_text` against `store` with [`DEFAULT_ABLATION_DEPTH`]
/// and [`DEFAULT_ABLATION_TIMEOUT`]. A convenient default for callers who
/// don't need to tune the search.
pub fn diagnose_default(query_text: &str, store: &Store) -> Result<Diagnosis> {
    diagnose(query_text, store, DEFAULT_ABLATION_DEPTH, Some(DEFAULT_ABLATION_TIMEOUT))
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
///
/// `timeout` bounds every SPARQL query this call runs — the original query
/// itself, each candidate combination's ablation check, and each filter's
/// ablation check — with one shared deadline rather than a fresh budget per
/// check, so the *whole* call is bounded by roughly `timeout` regardless of
/// how many candidates there are to work through. A combination or filter
/// that can't be checked in time is treated as "not a culprit" (a
/// conservative false rather than a risky claim based on partial evidence);
/// if the *original* query itself can't even be evaluated within the
/// budget, [`RelaxError::Timeout`](crate::error::RelaxError::Timeout) is
/// returned, since there's nothing meaningful to diagnose without it. Pass
/// `None` to leave it unbounded — but see the module docs: an abandoned
/// caller-side timeout (e.g. a Python `future.result(timeout=...)`) doesn't
/// stop this function from still running to completion in the background,
/// so a real, enforced `timeout` here is what actually bounds the work.
pub fn diagnose(query_text: &str, store: &Store, depth: usize, timeout: Option<Duration>) -> Result<Diagnosis> {
    let query = SparqlParser::new().parse_query(query_text)?;
    ensure_select(&query)?;
    let pattern = pattern_of(&query).clone();
    diagnose_parsed(&query, &pattern, store, depth, timeout)
}

pub(crate) fn ensure_select(query: &Query) -> Result<()> {
    if !matches!(query, Query::Select { .. }) {
        return Err(RelaxError::UnsupportedQueryForm("SELECT"));
    }
    Ok(())
}

/// Same as [`diagnose`], but takes an already-parsed `query`/`pattern`
/// instead of re-parsing query text itself.
/// [`crate::relax::diagnose_and_relax`] uses this — rather than calling
/// [`diagnose`] and then separately re-parsing the same text a second time
/// for its own use — so the culprit triples it gets back are guaranteed to
/// be the exact same values it can later find and remove from its own copy
/// of the pattern. Two independent parses of identical text are not
/// guaranteed to produce identical `TriplePattern` values in every case,
/// which would otherwise make a culprit "disappear" when relaxation tries
/// to remove it.
pub(crate) fn diagnose_parsed(
    query: &Query,
    pattern: &GraphPattern,
    store: &Store,
    depth: usize,
    timeout: Option<Duration>,
) -> Result<Diagnosis> {
    // One shared deadline for every query this call runs, rather than a
    // fresh budget per check — see the `timeout` docs on [`diagnose`].
    let deadline = timeout.map(|t| Instant::now() + t);

    let Some(original_rows) = run_select_query_with_deadline(query.clone(), store, deadline)? else {
        return Err(RelaxError::Timeout);
    };

    let triple_candidates: Vec<TriplePattern> = collect_bgp_triples(pattern)
        .into_iter()
        .filter(|t| matches!(t.predicate, NamedNodePattern::NamedNode(_)))
        .collect();
    let filter_candidates = collect_filters(pattern);
    if triple_candidates.is_empty() && filter_candidates.is_empty() {
        return Err(RelaxError::NoTriples);
    }

    // One shared cancellation token (and its one timer thread) for every
    // combination/filter checked below, rather than one per check — see
    // `SharedDeadline`'s docs on why that matters once a query has enough
    // candidate triples for `depth` 2 or 3 to mean hundreds of combinations.
    let guard = SharedDeadline::new(deadline);

    let mut culprits = Vec::new();
    let max_depth = depth.max(1);
    for k in 1..=max_depth {
        if k > triple_candidates.len() {
            break; // no more triples left to combine
        }

        // Every combination's ablation check is an independent read against
        // `store` (Oxigraph stores support concurrent reads), so check them
        // all in parallel rather than one at a time — C(n, k) grows fast
        // once a query has more than a handful of candidate triples.
        let found: Vec<Culprit> = combinations(&triple_candidates, k)
            .into_par_iter()
            .filter(|combo| is_culprit_combo(query, pattern, combo, store, original_rows.is_empty(), &guard))
            .map(|triples| Culprit { triples, depth: k })
            .collect();

        if !found.is_empty() {
            culprits.extend(found);
            break; // don't escalate to a larger combination size
        }
    }

    // Each filter's ablation check is an independent read against `store`,
    // exactly like the triple-combo checks above, so check them all in
    // parallel too rather than one at a time.
    let filter_culprits: Vec<FilterCulprit> = filter_candidates
        .into_par_iter()
        .filter_map(|expression| {
            let reduced_pattern = remove_filter(pattern, &expression)?;
            let reduced_rows = run_select_query_with_guard(with_pattern(query, reduced_pattern), store, &guard).ok().flatten()?;
            // Removing a FILTER can only ever keep or grow the result set
            // (it's a pure restriction), so a strict increase means this
            // filter was actually excluding rows. (Unlike triple combos,
            // this can't reuse the cheaper ASK-existence shortcut above
            // even when the original query is empty: the actual row count
            // is part of `FilterCulprit`, not just a yes/no.)
            (reduced_rows.len() > original_rows.len())
                .then(|| FilterCulprit { expression, row_count_without_filter: reduced_rows.len() })
        })
        .collect();

    Ok(Diagnosis { original_row_count: original_rows.len(), culprits, filter_culprits })
}

/// Whether removing every triple in `combo` together unblocks the query,
/// and no single row satisfies every triple in `combo` jointly (i.e. the
/// combination is genuinely, jointly responsible — not just incidentally
/// relaxing enough constraints for something else to match).
///
/// `original_is_empty` (`original_row_count == 0` in [`diagnose_parsed`])
/// enables a shortcut: the original pattern is exactly `reduced_pattern ∧
/// combo`, so a row of `reduced_pattern` that also satisfied `combo`
/// pointwise would itself be a full solution to the original query. If the
/// original query has zero rows, no such row can exist — the per-row
/// [`triple_holds_for_row`] check below is *guaranteed* to never match, so
/// it's skipped entirely in favor of a single cheap existence check
/// ([`pattern_has_solution`]). This only holds when the original query is
/// empty; otherwise the full per-row check below is still needed.
///
/// `guard` bounds the query work the same way as
/// [`run_select_query_with_deadline`], but is built once per [`diagnose_parsed`]
/// call and shared (via a cheap token clone) across every combination
/// checked here, rather than spawning a fresh timer thread per combination —
/// see [`SharedDeadline`]. Hitting the deadline is treated as "not a
/// culprit" (`false`) rather than a hang: claiming culprit-hood on
/// unfinished evaluation would be a false positive (see the per-row loop
/// below, which only saw *some* of the reduced query's rows before being
/// cut off — not enough to conclude none of them jointly satisfy `combo`).
fn is_culprit_combo(query: &Query, pattern: &GraphPattern, combo: &[TriplePattern], store: &Store, original_is_empty: bool, guard: &SharedDeadline) -> bool {
    let Some(reduced_pattern) = combo.iter().try_fold(pattern.clone(), |p, t| remove_triple(&p, t)) else {
        return false;
    };

    if original_is_empty {
        return pattern_has_solution(query, reduced_pattern, store, guard);
    }

    let cancellation_token = match guard.token() {
        Some(token) => token,
        None => return false, // budget already exhausted before this combo could even start
    };
    let mut evaluator = SparqlEvaluator::new();
    if let Some(token) = cancellation_token {
        evaluator = evaluator.with_cancellation_token(token);
    }

    // Streams the reduced query's solutions one at a time rather than
    // collecting them all first (unlike [`run_select_query`]): the common
    // case is a combo that *does* jointly hold somewhere (not a culprit),
    // and that can stop at the very first matching row instead of paying to
    // materialize a potentially large result set that removing constraining
    // triples tends to produce.
    let Ok(results) = evaluator.for_query(with_pattern(query, reduced_pattern)).on_store(store).execute() else {
        return false;
    };
    let QueryResults::Solutions(solutions) = results else { return false };

    let mut any_row = false;
    for solution in solutions {
        match solution {
            Ok(row) => {
                any_row = true;
                if combo.iter().all(|t| triple_holds_for_row(store, t, &row)) {
                    return false; // some binding satisfies every triple in the combo; not a genuine culprit set
                }
            }
            Err(QueryEvaluationError::Cancelled) => return false, // saw only some rows; not enough to conclude "not a culprit"
            Err(_) => break,
        }
    }
    any_row // non-empty, and no row ever satisfied the whole combo jointly
}

/// Whether `pattern` has at least one solution against `store`, evaluated as
/// a SPARQL `ASK` (short-circuits on the first matching solution) rather
/// than a `SELECT` that has to be told to stop separately or fully
/// materialized to find out. `guard` bounds the evaluation the same way as
/// [`run_select_query_with_deadline`] (see [`SharedDeadline`]); hitting it is
/// treated as "no solution found" (`false`) rather than a hang, since
/// claiming a solution exists on unfinished evaluation would be a false
/// positive.
fn pattern_has_solution(query: &Query, pattern: GraphPattern, store: &Store, guard: &SharedDeadline) -> bool {
    let cancellation_token = match guard.token() {
        Some(token) => token,
        None => return false,
    };
    let mut evaluator = SparqlEvaluator::new();
    if let Some(token) = cancellation_token {
        evaluator = evaluator.with_cancellation_token(token);
    }
    matches!(evaluator.for_query(ask_query(query, pattern)).on_store(store).execute(), Ok(QueryResults::Boolean(true)))
}

/// One shared cancellation token — and the single background timer thread
/// backing it — for every ablation combination/filter [`diagnose_parsed`]
/// checks against the same deadline, built once per call rather than once
/// per combination checked. A query with a few dozen candidate triples can
/// have hundreds of combinations to check at `depth` 2 or 3; building a
/// fresh [`CancellationToken`] (and the `thread::spawn` timer inside
/// [`deadline_token`]) for each one would mean hundreds of throwaway OS
/// threads racing to enforce what is, logically, one shared deadline.
/// [`CancellationToken`] is cheap to clone (see its own docs), so every
/// check just clones this one token instead.
pub(crate) enum SharedDeadline {
    /// No `timeout` was requested: every check proceeds unbounded.
    Unbounded,
    /// A `timeout` was requested but had already elapsed by the time this
    /// was constructed: every check should short-circuit as "not enough
    /// budget to even start" rather than kicking off a query only to have it
    /// cancelled immediately.
    Expired,
    /// A `timeout` was requested and is still live: every check shares this
    /// one token (and the one timer thread behind it).
    Active(CancellationToken),
}

impl SharedDeadline {
    fn new(deadline: Option<Instant>) -> Self {
        match deadline {
            None => SharedDeadline::Unbounded,
            Some(deadline) => match deadline_token(deadline) {
                Some(token) => SharedDeadline::Active(token),
                None => SharedDeadline::Expired,
            },
        }
    }

    /// `None` if the deadline has already passed (caller should give up
    /// without starting a query); `Some(None)` to run unbounded; `Some(Some(token))`
    /// to run with a (cloned, still shared) cancellation token.
    fn token(&self) -> Option<Option<CancellationToken>> {
        match self {
            SharedDeadline::Unbounded => Some(None),
            SharedDeadline::Expired => None,
            SharedDeadline::Active(token) => Some(Some(token.clone())),
        }
    }
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

/// Runs an already-parsed `query` (e.g. one built from a reduced pattern
/// rather than starting from SPARQL text) to completion, unbounded.
fn run_select_query(query: Query, store: &Store) -> Result<Vec<QuerySolution>> {
    let results = SparqlEvaluator::new().for_query(query).on_store(store).execute()?;
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

/// A cancellation token a background timer will cancel once `deadline`
/// passes, or `None` if `deadline` is already in the past — nothing left to
/// run, so there's no point starting a query only to cancel it immediately.
/// Shared by every deadline-aware query execution in this module so the
/// "spawn a timer, cancel on timeout" wiring only needs writing once.
///
/// Backed by Oxigraph's own `CancellationToken`, checked by the query
/// engine on every underlying quad lookup — a real mid-evaluation abort,
/// not just a wrapper that gives up waiting while the query keeps running
/// in the background (see the module docs on why that distinction matters).
fn deadline_token(deadline: Instant) -> Option<CancellationToken> {
    let remaining = deadline.saturating_duration_since(Instant::now());
    if remaining.is_zero() {
        return None;
    }
    let cancellation_token = CancellationToken::new();
    let timer_token = cancellation_token.clone();
    thread::spawn(move || {
        thread::sleep(remaining);
        timer_token.cancel();
    });
    Some(cancellation_token)
}

/// Same as [`run_select_query`], but aborts the evaluation rather than
/// running it unbounded if it doesn't finish by `deadline` (`None` means no
/// deadline — delegates straight to [`run_select_query`]). `Ok(None)`
/// specifically means the deadline was hit; it's kept distinct from a
/// genuine empty result (`Ok(Some(vec![]))`) so a caller can degrade
/// gracefully (e.g. fall back to some already-known-safe alternative)
/// instead of treating a slow query as a real error — a single expensive
/// reduced-query evaluation (e.g. removing a triple leaves the rest of the
/// query essentially unconstrained, forcing a large join) shouldn't be able
/// to hang or fail a caller that's relaxing many culprits at once.
pub(crate) fn run_select_query_with_deadline(query: Query, store: &Store, deadline: Option<Instant>) -> Result<Option<Vec<QuerySolution>>> {
    run_select_query_with_guard(query, store, &SharedDeadline::new(deadline))
}

/// Same as [`run_select_query_with_deadline`], but takes an already-built
/// [`SharedDeadline`] instead of constructing one (and its background timer
/// thread) fresh — for call sites that check many combinations/filters
/// against the same deadline and want to share one guard across all of them.
pub(crate) fn run_select_query_with_guard(query: Query, store: &Store, guard: &SharedDeadline) -> Result<Option<Vec<QuerySolution>>> {
    let Some(cancellation_token) = guard.token() else {
        return Ok(None);
    };
    let Some(cancellation_token) = cancellation_token else {
        return run_select_query(query, store).map(Some);
    };

    let results = match SparqlEvaluator::new().with_cancellation_token(cancellation_token).for_query(query).on_store(store).execute() {
        Ok(results) => results,
        Err(QueryEvaluationError::Cancelled) => return Ok(None),
        Err(e) => return Err(RelaxError::Evaluation(e)),
    };
    match results {
        QueryResults::Solutions(iter) => {
            let mut rows = Vec::new();
            for solution in iter {
                match solution {
                    Ok(row) => rows.push(row),
                    Err(QueryEvaluationError::Cancelled) => return Ok(None),
                    Err(e) => return Err(RelaxError::Evaluation(e)),
                }
            }
            Ok(Some(rows))
        }
        _ => Err(RelaxError::UnsupportedQueryForm("SELECT")),
    }
}
