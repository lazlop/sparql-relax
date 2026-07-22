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
//! connecting path) is [`crate::connect`]'s job, done only for culprits that
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
//! the signal. Filters are only ever reported, never connected, and `depth`
//! does not apply to them (queries rarely have enough interacting filters
//! for combining removals to matter, unlike triples).

use crate::algebra::{
    ask_query, collect_bgp_triples, collect_filters, collect_required_bgp_triples, pattern_of, remove_filter, remove_triple,
    variables_of_triples, widen_projection, with_pattern,
};
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

/// A combination of triples whose reduced pattern (with them removed) was
/// never evaluated against `store` at all, because doing so would force a
/// cartesian product — see `algebra::has_cartesian_join`. That's
/// exactly the shape that can make a query engine materialize a full N×M
/// cross product before yielding a single row, regardless of how tightly
/// `timeout` is set (see the module docs). This is *not* a claim about
/// whether the combination is or isn't a genuine culprit — it's surfaced
/// separately from [`Culprit`] specifically so it isn't mistaken for one of
/// the ordinary "checked, found nothing" negatives that `depth`'s search
/// naturally produces.
pub struct CartesianRiskCombo {
    pub triples: Vec<TriplePattern>,
    /// The combination size at which this was encountered (see
    /// [`Culprit::depth`]).
    pub depth: usize,
}

pub struct Diagnosis {
    pub original_row_count: usize,
    pub culprits: Vec<Culprit>,
    pub filter_culprits: Vec<FilterCulprit>,
    pub cartesian_risks: Vec<CartesianRiskCombo>,
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

/// `rdf:type`'s IRI — checked against every candidate triple's predicate to
/// prioritize the combination search below. Measured across this tool's
/// building-automation eval set, roughly two-thirds of confirmed culprits
/// involve at least one `rdf:type` triple (a variable typed as the wrong
/// class), so combinations that include one are worth checking first.
const RDF_TYPE_IRI: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";

/// Whether `triple`'s predicate is `rdf:type` — see [`RDF_TYPE_IRI`].
fn is_rdf_type_triple(triple: &TriplePattern) -> bool {
    matches!(&triple.predicate, NamedNodePattern::NamedNode(p) if p.as_str() == RDF_TYPE_IRI)
}

/// Diagnoses `query_text` against `store` with [`DEFAULT_ABLATION_DEPTH`]
/// and [`DEFAULT_ABLATION_TIMEOUT`], with the cartesian-risk guard left on
/// (see `ignore_cartesian_risk` on [`diagnose`]). A convenient default for
/// callers who don't need to tune the search.
pub fn diagnose_default(query_text: &str, store: &Store) -> Result<Diagnosis> {
    diagnose(query_text, store, DEFAULT_ABLATION_DEPTH, Some(DEFAULT_ABLATION_TIMEOUT), false)
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
/// Within one combination size, combinations touching an `rdf:type` triple
/// are checked in a first wave, ahead of every other combination of that
/// size (see [`RDF_TYPE_IRI`]) — a real fix involving a mistyped variable is
/// common enough in practice that finding one there skips checking the rest
/// of that size's (usually much larger) combination space entirely. If the
/// type-wave comes up empty, every remaining combination of that size is
/// still checked, so this only ever prunes work on the already-successful
/// path, never coverage on an unsuccessful one — the "all are returned"
/// guarantee above holds within whichever wave actually finds something.
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
///
/// A combination whose reduced pattern (with it removed) would force a
/// cartesian product is never evaluated at all, `timeout` notwithstanding —
/// see `algebra::has_cartesian_join`. That's a distinct outcome
/// from "checked, and it wasn't a culprit": it's collected separately in
/// [`Diagnosis::cartesian_risks`] rather than silently folded into a
/// negative result, unless `ignore_cartesian_risk` is set (see below).
///
/// `ignore_cartesian_risk` disables that guard entirely: every combination
/// is actually evaluated against `store`, so [`Diagnosis::cartesian_risks`]
/// always comes back empty, and a combination that would otherwise have
/// been skipped can be confirmed a genuine [`Culprit`] instead. `false` (the
/// default via [`diagnose_default`]) preserves the guard. Passing `true`
/// means opting out of the protection it applies: a disconnected BGP can
/// make the query engine materialize a full N×M cross product before
/// yielding a single row, regardless of how tightly `timeout` is set — a
/// measured case elsewhere in this project sat for over 200 seconds and
/// permanently occupied a shared worker thread until the whole process was
/// killed (see `eval/run_eval.py`'s process-level watchdog for why that
/// backstop lives at the process level, not inside this call). Only set
/// this once you've independently judged the risk worth taking for this
/// specific query/graph, ideally from a process you can afford to kill
/// outright if a check gets stuck.
pub fn diagnose(query_text: &str, store: &Store, depth: usize, timeout: Option<Duration>, ignore_cartesian_risk: bool) -> Result<Diagnosis> {
    let query = SparqlParser::new().parse_query(query_text)?;
    ensure_select(&query)?;
    let pattern = pattern_of(&query).clone();
    diagnose_parsed(&query, &pattern, store, depth, timeout, ignore_cartesian_risk)
}

pub(crate) fn ensure_select(query: &Query) -> Result<()> {
    if !matches!(query, Query::Select { .. }) {
        return Err(RelaxError::UnsupportedQueryForm("SELECT"));
    }
    Ok(())
}

/// Same as [`diagnose`], but takes an already-parsed `query`/`pattern`
/// instead of re-parsing query text itself.
/// [`crate::connect::diagnose_and_connect`] uses this — rather than calling
/// [`diagnose`] and then separately re-parsing the same text a second time
/// for its own use — so the culprit triples it gets back are guaranteed to
/// be the exact same values it can later find and remove from its own copy
/// of the pattern. Two independent parses of identical text are not
/// guaranteed to produce identical `TriplePattern` values in every case,
/// which would otherwise make a culprit "disappear" when connection tries
/// to remove it.
///
/// `ignore_cartesian_risk` is passed straight through from whichever public
/// entry point called this — see its docs on [`diagnose`] (and on
/// [`crate::connect::diagnose_and_connect`], which shares this same guard).
pub(crate) fn diagnose_parsed(
    query: &Query,
    pattern: &GraphPattern,
    store: &Store,
    depth: usize,
    timeout: Option<Duration>,
    ignore_cartesian_risk: bool,
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
    let mut cartesian_risks = Vec::new();
    let max_depth = depth.max(1);
    for k in 1..=max_depth {
        if k > triple_candidates.len() {
            break; // no more triples left to combine
        }

        // Every combination's ablation check is an independent read against
        // `store` (Oxigraph stores support concurrent reads), so check them
        // all in parallel rather than one at a time — C(n, k) grows fast
        // once a query has more than a handful of candidate triples.
        //
        // Split into two waves rather than one combined parallel batch: the
        // type-wave (see [`is_rdf_type_triple`]) is checked first, and if it
        // already unblocks the query, the rest-wave is never dispatched at
        // all. Doing this as two sequential (each internally parallel)
        // batches, instead of just sorting one combined batch, is what
        // actually saves the work — a combined batch would still run every
        // combination via `into_par_iter().collect()` before anything
        // downstream could look at the results.
        let mut found = Vec::new();
        let (type_combos, rest_combos): (Vec<_>, Vec<_>) =
            combinations(&triple_candidates, k).into_iter().partition(|combo| combo.iter().any(is_rdf_type_triple));
        for wave in [type_combos, rest_combos] {
            if wave.is_empty() || !found.is_empty() {
                continue;
            }
            let verdicts: Vec<(Vec<TriplePattern>, ComboVerdict)> = wave
                .into_par_iter()
                .map(|combo| {
                    let verdict =
                        classify_combo(query, pattern, &combo, store, original_rows.is_empty(), &guard, ignore_cartesian_risk);
                    (combo, verdict)
                })
                .collect();

            for (triples, verdict) in verdicts {
                match verdict {
                    ComboVerdict::Culprit => found.push(Culprit { triples, depth: k }),
                    ComboVerdict::CartesianRisk => cartesian_risks.push(CartesianRiskCombo { triples, depth: k }),
                    ComboVerdict::NotCulprit => {}
                }
            }
        }

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

    Ok(Diagnosis { original_row_count: original_rows.len(), culprits, filter_culprits, cartesian_risks })
}

/// The three outcomes [`classify_combo`] can reach for one candidate
/// combination, kept distinct rather than collapsing `CartesianRisk` into
/// `NotCulprit`: the latter means "checked, and it wasn't"; the former means
/// "never checked at all" (see [`CartesianRiskCombo`]) — conflating them
/// would make a query this tool declined to evaluate look identical to one
/// it actually ruled out.
enum ComboVerdict {
    NotCulprit,
    Culprit,
    CartesianRisk,
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
/// see [`SharedDeadline`]. Hitting the deadline reaches [`ComboVerdict::NotCulprit`]
/// rather than a hang: claiming culprit-hood on unfinished evaluation would
/// be a false positive (see the per-row loop below, which only saw *some*
/// of the reduced query's rows before being cut off — not enough to
/// conclude none of them jointly satisfy `combo`).
///
/// Before running anything, checks whether the reduced pattern (`combo`
/// removed) would force a cartesian product — see
/// [`crate::algebra::has_cartesian_join`] — and returns
/// [`ComboVerdict::CartesianRisk`] without evaluating it at all if so. This
/// applies uniformly to both branches below (the cheap `ASK`-based shortcut
/// included) rather than only the full per-row scan: whether a physical
/// query plan needs to fully materialize one side of a disconnected join
/// before it can answer *at all* — even a plain existence check — depends on
/// the query engine's own planner, not on which SPARQL query form asked the
/// question, so there's no branch here that's provably safe to exempt.
///
/// The connectivity check itself only looks at the *required* pattern (see
/// [`crate::algebra::collect_required_bgp_triples`]), skipping every
/// `OPTIONAL`'s own triples: a variable shared only inside an `OPTIONAL`
/// doesn't guarantee connectivity for any given solution (the optional side
/// can be entirely absent), so counting it would hide a real disconnect —
/// and cartesian risk — in the pattern every solution actually has to
/// satisfy.
fn classify_combo(
    query: &Query,
    pattern: &GraphPattern,
    combo: &[TriplePattern],
    store: &Store,
    original_is_empty: bool,
    guard: &SharedDeadline,
    ignore_cartesian_risk: bool,
) -> ComboVerdict {
    let Some(reduced_pattern) = combo.iter().try_fold(pattern.clone(), |p, t| remove_triple(&p, t)) else {
        return ComboVerdict::NotCulprit;
    };

    if !ignore_cartesian_risk && crate::algebra::has_cartesian_join(&collect_required_bgp_triples(&reduced_pattern)) {
        return ComboVerdict::CartesianRisk;
    }

    if original_is_empty {
        return if pattern_has_solution(query, reduced_pattern, store, guard) {
            ComboVerdict::Culprit
        } else {
            ComboVerdict::NotCulprit
        };
    }

    let cancellation_token = match guard.token() {
        Some(token) => token,
        None => return ComboVerdict::NotCulprit, // budget already exhausted before this combo could even start
    };
    let mut evaluator = SparqlEvaluator::new();
    if let Some(token) = cancellation_token {
        evaluator = evaluator.with_cancellation_token(token);
    }

    // Widen the reduced query's projection so `combo`'s subject/object
    // variables are visible in its rows even if they're plain WHERE-clause
    // bridge variables never listed in the original SELECT — see
    // `widen_projection`'s docs for why the original projection alone isn't
    // enough here.
    let widened_pattern = widen_projection(&reduced_pattern, &variables_of_triples(combo));

    // Streams the reduced query's solutions one at a time rather than
    // collecting them all first (unlike [`run_select_query`]): the common
    // case is a combo that *does* jointly hold somewhere (not a culprit),
    // and that can stop at the very first matching row instead of paying to
    // materialize a potentially large result set that removing constraining
    // triples tends to produce.
    let Ok(results) = evaluator.for_query(with_pattern(query, widened_pattern)).on_store(store).execute() else {
        return ComboVerdict::NotCulprit;
    };
    let QueryResults::Solutions(solutions) = results else { return ComboVerdict::NotCulprit };

    let mut any_row = false;
    for solution in solutions {
        match solution {
            Ok(row) => {
                any_row = true;
                if combo.iter().all(|t| triple_holds_for_row(store, t, &row)) {
                    return ComboVerdict::NotCulprit; // some binding satisfies every triple in the combo; not a genuine culprit set
                }
            }
            // Cancelled or any other error: only *some* rows were seen
            // before evaluation stopped — not enough to conclude no row
            // jointly satisfies `combo`, so this isn't treated as a genuine
            // culprit. Silently falling through on a non-Cancelled error
            // would otherwise report a false-positive culprit based on
            // partial evidence.
            Err(_) => return ComboVerdict::NotCulprit,
        }
    }
    if any_row { ComboVerdict::Culprit } else { ComboVerdict::NotCulprit } // non-empty, and no row ever satisfied the whole combo jointly
}

/// Matches each text in `texts` back to one of `available`'s actual
/// `TriplePattern`s by exact `Display`-string equality, one not-yet-claimed
/// occurrence per text: two distinct triples in a pattern can share
/// identical SPARQL text (e.g. a repeated triple pattern), so matching
/// greedily against the first unclaimed occurrence disambiguates without
/// assuming the whole pattern has no duplicates. Used by [`pruned_query_text`].
fn match_triples_by_text(available: &[TriplePattern], texts: &[String]) -> Result<Vec<TriplePattern>> {
    let mut used = vec![false; available.len()];
    let mut matched = Vec::with_capacity(texts.len());
    for text in texts {
        let idx = available
            .iter()
            .enumerate()
            .find(|(i, t)| !used[*i] && t.to_string() == *text)
            .map(|(i, _)| i)
            .ok_or_else(|| RelaxError::CulpritNotFound(text.clone()))?;
        used[idx] = true;
        matched.push(available[idx].clone());
    }
    Ok(matched)
}

/// The SPARQL text of `query_text` with every triple in `triples` (matched
/// by exact text, the same form [`Culprit::triples`]/
/// [`CartesianRiskCombo::triples`] already carry as strings once converted
/// at the Python boundary) removed from its basic graph pattern — no path
/// substitution, just ablation. A pure syntactic transform: no `Store`
/// involved, and nothing here is executed.
///
/// Used to score what a confirmed culprit combination's removal alone gets
/// you (e.g. via value-set F1 against ground truth), independent of whether
/// a real path fix was also found for it — the same "diagnose's own signal"
/// [`crate::connect::ConnectedCulprit::pruned_query`] already provides for
/// combinations `diagnose_and_connect` confirms, but usable here for any
/// [`Culprit`] a plain [`diagnose`] call confirmed, which never had a
/// `ConnectedCulprit` (or any query text) built for it at all.
pub fn pruned_query_text(query_text: &str, triples: &[String]) -> Result<String> {
    let query = SparqlParser::new().parse_query(query_text)?;
    ensure_select(&query)?;
    let pattern = pattern_of(&query).clone();
    let available = collect_bgp_triples(&pattern);

    let matched = match_triples_by_text(&available, triples)?;
    let reduced_pattern = matched
        .iter()
        .try_fold(pattern, |p, t| remove_triple(&p, t))
        .ok_or_else(|| RelaxError::CulpritNotFound(triples.join(", ")))?;
    Ok(with_pattern(&query, reduced_pattern).to_string())
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
/// to hang or fail a caller that's connecting many culprits at once.
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
