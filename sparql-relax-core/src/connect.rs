//! Orchestrates diagnosis and connection: for each culprit combination the
//! ablation diagnosis in [`crate::diagnose`] flags as broken (a single
//! triple at `ablation_depth` 1, or several triples jointly responsible at
//! a higher depth), resolves what its variables are actually bound to —
//! diagnosis itself does none of this binding work, so a plain diagnosis
//! call never pays for it — then searches each triple's bound endpoints for
//! a real forward/inverse path (via [`crate::bfs`]), splices every triple
//! that found one into the pattern in its place and simply drops whichever
//! didn't, and confirms the result by actually re-running the modified
//! query. A combination only goes unconnected (falling back to
//! [`ConnectedCulprit::pruned_query`]) when *none* of its triples found a
//! path — as soon as at least one does, the rest are dropped rather than
//! left broken, since a partially-connected query is still a strictly better
//! candidate than dropping the whole combination, and it's re-verified
//! against the graph exactly like a fully-connected one, so a bad partial fix
//! still scores as empty instead of being trusted blindly.
//!
//! A broken triple's other side is sometimes not bound anywhere else in the
//! query (e.g. `?sensor` in `building hasSensor ?sensor` if nothing else
//! constrains `?sensor`), so there's no specific target to search for —
//! only whichever single side did resolve. Such a triple is skipped
//! entirely rather than searched: without a fixed goal, "path search" would
//! just be an undirected exploration outward from one anchor, offering
//! suggestions with nothing to verify them against except re-running the
//! whole query, which isn't worth the cost of searching for.
//!
//! Path search itself can optionally be restricted to a caller-supplied set
//! of predicate namespaces (see [`NamespaceScope`]) — real edges whose
//! predicate falls outside every listed namespace are simply invisible to
//! the search, even if they'd otherwise connect the two endpoints.
//!
//! Connecting one culprit combination can itself be expensive: resolving
//! endpoints or verifying a candidate fix both re-run a SPARQL query that,
//! with the broken triple(s) out of the way, can turn into a much larger
//! join than the original ever was, and the path search in between can run
//! long too on a real graph with a high-fan-out hub node. Each combination
//! gets its own `timeout` budget covering all three; if it can't finish in
//! time, that combination falls back to its
//! [`ConnectedCulprit::pruned_query`] (the broken triple(s) simply dropped,
//! already known cheap and non-empty from diagnosis) rather than hanging or
//! failing the whole [`diagnose_and_connect`] call over one slow combination.
//!
//! Diagnosis has the same kind of internally-enforced budget, independent
//! of `timeout` above (see `diagnose_timeout` on [`diagnose_and_connect`] and
//! the `timeout` docs on [`crate::diagnose::diagnose`]) — it matters just as
//! much there: a caller that gives up waiting on a slow diagnosis call (a
//! Python `future.result(timeout=...)`, say) doesn't make the call actually
//! stop running unless *something* inside it is enforcing a real deadline.

use crate::algebra::{
    pattern_of, remove_triple, replace_triple_with_path, variables_of_triples, widen_projection, with_limit, with_pattern,
};
use crate::bfs::{FrontierSearch, Hop, path_holds, path_to_property_path};
use crate::diagnose::{
    CartesianRiskCombo, Culprit, DEFAULT_ABLATION_DEPTH, DEFAULT_ABLATION_TIMEOUT, DEFAULT_IGNORE_CARTESIAN_RISK, diagnose_parsed,
    ensure_select, resolve_term_pattern, run_select_query_with_deadline,
};
use crate::error::{RelaxError, Result};
use crate::fanout::FanoutIndex;
use oxigraph::model::Term;
use oxigraph::store::Store;
use rayon::prelude::*;
use spargebra::Query;
use spargebra::SparqlParser;
use spargebra::algebra::{GraphPattern, PropertyPathExpression};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

/// Default for `sample_limit`: by default (see [`DEFAULT_FIND_ALL_PATHS`]),
/// this caps how many *distinct subjects* [`search_candidates_grouped`]
/// searches, not flat pairs — and a cartesian-risk combination evaluated
/// with `ignore_cartesian_risk` cross-joins its bound endpoints, so the
/// reduced query's row order tends to exhaust every match for one subject
/// before advancing to the next. 500 is cheap relative to the search it
/// bounds (each subject's search is at most a `max_depth`-bounded BFS
/// against its own candidate set, not a full store scan) and comfortably
/// covers that skew without examining every row of a potentially large
/// reduced query the way `None` would.
pub const DEFAULT_SAMPLE_LIMIT: usize = 500;

/// Default for `find_all_paths`: `false`, meaning path search stops as soon
/// as it has found *a* connecting path (the shortest one reachable within
/// `max_depth`, across all sampled endpoints — see
/// [`search_candidates_grouped`]) rather than continuing to search every
/// sampled endpoint for every distinct path it might individually need.
/// That's the right default for most callers: a single short property path
/// is a simpler, easier-to-read fix, and it's cheaper to find. Pass `true`
/// when a broken triple's *different* bound pairs are known (or suspected)
/// to genuinely need different real paths — e.g. one entity reached via a
/// 2-hop path, another via an unrelated 1-hop path — so the fix should
/// recover all of them via [`search_candidates`] instead of just the first
/// one found (see
/// `combines_distinct_paths_from_different_bound_pairs_as_alternatives`).
pub const DEFAULT_FIND_ALL_PATHS: bool = false;

/// Default path-search depth: a culprit triple is only ever searched once
/// both its subject and object are known (see the module docs), so this is
/// a concrete point-to-point search bounded by its target.
pub const DEFAULT_PAIR_SEARCH_DEPTH: usize = 2;

/// Default for `result_limit`: a connected query (especially one whose paths
/// were combined via `|` alternation) can match far more broadly than the
/// original, so its row count is capped by default rather than left
/// unbounded.
pub const DEFAULT_RESULT_LIMIT: usize = 50_000;

/// Default for `timeout`: the query work needed to connect one culprit
/// combination (resolving endpoints, verifying a candidate fix) is normally
/// well under this; five seconds is enough headroom for that while still
/// bounding the rare combination whose reduced query turns into a much
/// larger join than the original.
pub const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

/// Namespace prefixes [`NamespaceScope::default`] restricts path search to:
/// the building-automation ontologies (Brick, ASHRAE 223P, RDFS, QUDT) this
/// tool is normally used against. Ported from the Python implementation's
/// `_RELAX_NAMESPACES`.
pub const DEFAULT_CONNECT_NAMESPACES: &[&str] = &[
    "https://brickschema.org/schema/Brick#", // brick:
    "https://brickschema.org/schema/Brick/", // ref: (covers both ref# and bare Brick/)
    "http://data.ashrae.org/standard223#",   // s223:
    "http://www.w3.org/2000/01/rdf-schema#", // rdfs:
    "http://qudt.org/schema/qudt/",          // qudt:
];

/// Which predicate namespaces path search is allowed to traverse when
/// looking for a real path between a broken triple's bound endpoints. A
/// real edge whose predicate falls outside every allowed namespace is
/// simply invisible to the search, even if it would otherwise connect the
/// two endpoints.
#[derive(Clone)]
pub enum NamespaceScope {
    /// Only follow edges whose predicate IRI starts with one of these
    /// prefixes.
    Only(Vec<String>),
    /// No restriction: any real predicate found in the store is fair game
    /// (the original behavior, before namespace scoping existed).
    Unrestricted,
}

impl Default for NamespaceScope {
    /// Restricts to [`DEFAULT_CONNECT_NAMESPACES`] — the common case for this
    /// tool's building-automation use case, where a real but out-of-domain
    /// predicate (e.g. an ad hoc `ex:` edge in a hand-authored graph) is
    /// rarely a fix anyone actually wants suggested.
    fn default() -> Self {
        NamespaceScope::Only(DEFAULT_CONNECT_NAMESPACES.iter().map(|ns| ns.to_string()).collect())
    }
}

impl NamespaceScope {
    /// The predicate-namespace filter this scope resolves to — `None` means
    /// unrestricted. Used both by path search itself and, since
    /// [`FanoutIndex::build`] needs to scan the same predicate set path
    /// search is restricted to (see its docs), by callers building a
    /// [`FanoutIndex`] to match a given `NamespaceScope`.
    pub fn as_filter(&self) -> Option<&[String]> {
        match self {
            NamespaceScope::Only(namespaces) => Some(namespaces),
            NamespaceScope::Unrestricted => None,
        }
    }
}

/// A broken triple's subject and object, both resolved for one binding of
/// the rest of the query — the concrete point-to-point pair path search
/// runs between.
type BoundEndpoint = (Term, Term);

/// One triple within a connected culprit combination, and what path search
/// found for it specifically.
pub struct ConnectedTriple {
    /// The broken triple pattern, as SPARQL text (e.g. `?s <p> ?o`).
    pub triple_text: String,
    /// Every distinct forward/inverse hop sequence found, combined into the
    /// path below via `|`. By default (`find_all_paths: false`, see
    /// [`DEFAULT_FIND_ALL_PATHS`]) this holds at most one: search stops as
    /// soon as it finds a connecting path, the shortest one reachable
    /// within `max_depth`. Different sampled bound endpoints can genuinely
    /// need different paths (e.g. one entity reached via a 2-hop path,
    /// another via an unrelated 1-hop path) — pass `find_all_paths: true`
    /// to search every sampled endpoint and keep every distinct path found,
    /// rather than only the first.
    pub hop_alternatives: Vec<Vec<Hop>>,
    /// The hop alternatives rendered as a single SPARQL property path (e.g.
    /// `<p1>/<p2>` alone, or `(<p1>/<p2>)|<p3>` when more than one distinct
    /// path was found). `None` if no connecting path was found.
    pub path_text: Option<String>,
}

pub struct ConnectedCulprit {
    /// The ablation combination size at which this culprit was found (see
    /// `ablation_depth` on [`diagnose_and_connect`]); 1 unless it was only
    /// found jointly responsible alongside other triples.
    pub found_at_depth: usize,
    /// Every triple in the culprit combination, each with its own path
    /// search result, in the same order they appear in the query.
    pub triples: Vec<ConnectedTriple>,
    /// The full query with every triple above that found a path replaced by
    /// it, and every triple that didn't simply dropped. `None` only when
    /// *none* of the combination's triples found a path — as soon as one
    /// does, the rest are dropped rather than leaving the whole combination
    /// unconnected.
    pub connected_query: Option<String>,
    /// Row count of `connected_query` when re-executed. Zero if no
    /// connection was built at all, or it still returns nothing.
    pub row_count: usize,
    /// The original query with every triple in this combination simply
    /// removed — no path substitution, so it isn't a real fix (it silently
    /// drops a constraint rather than connecting it) and its rows shouldn't
    /// be trusted as answers. Always present, and its text alone needs no
    /// store access to build, so it's there even if `timeout` cuts off
    /// everything else for this combination. Useful as a fallback when
    /// `connected_query` is `None` or still returns nothing, so the caller
    /// isn't left with no query at all.
    pub pruned_query: String,
    /// Row count of `pruned_query` when re-executed. Guaranteed non-empty
    /// under normal operation — an empty result here would mean this
    /// combination was never a genuine culprit in the first place (see the
    /// diagnosis module docs) — *except* when `timeout` cuts off the
    /// verification itself before it can complete, in which case this falls
    /// back to `0` even though `pruned_query` is still known to return rows.
    pub pruned_row_count: usize,
}

/// A `FILTER` flagged by ablation as excluding rows. Reported, not connected:
/// there's no graph-path search that applies to an arbitrary expression.
pub struct FilterReport {
    /// The filter expression, as SPARQL text (e.g. `?o > 5`).
    pub expression_text: String,
    /// Row count of the query with just this filter removed.
    pub row_count_without_filter: usize,
}

pub struct ConnectReport {
    pub original_row_count: usize,
    pub results: Vec<ConnectedCulprit>,
    pub filter_results: Vec<FilterReport>,
    /// Combinations diagnosis flagged as cartesian risks — skipped entirely
    /// (never evaluated, and so never attempted here) because their reduced
    /// pattern was disconnected. Always empty when `ignore_cartesian_risk`
    /// was set on this call, since every combination was actually evaluated
    /// instead of being skipped. Mirrors [`crate::diagnose::Diagnosis::cartesian_risks`]
    /// exactly, for the same reason: a combination this call declined to
    /// check shouldn't look identical to one it actually ruled out.
    pub cartesian_risks: Vec<CartesianRiskCombo>,
}

/// Diagnoses `query_text` against `store` with [`DEFAULT_ABLATION_DEPTH`],
/// [`DEFAULT_SAMPLE_LIMIT`], [`DEFAULT_PAIR_SEARCH_DEPTH`],
/// [`NamespaceScope::default`] (restricted to [`DEFAULT_CONNECT_NAMESPACES`]),
/// [`DEFAULT_CONNECT_TIMEOUT`],
/// [`DEFAULT_ABLATION_TIMEOUT`](crate::diagnose::DEFAULT_ABLATION_TIMEOUT),
/// [`DEFAULT_IGNORE_CARTESIAN_RISK`](crate::diagnose::DEFAULT_IGNORE_CARTESIAN_RISK),
/// and [`DEFAULT_FIND_ALL_PATHS`], and attempts to connect each broken
/// culprit combination found. A convenient default for callers who don't
/// need to tune the search.
pub fn diagnose_and_connect_default(query_text: &str, store: &Store) -> Result<ConnectReport> {
    diagnose_and_connect(
        query_text,
        store,
        DEFAULT_ABLATION_DEPTH,
        None,
        Some(DEFAULT_SAMPLE_LIMIT),
        Some(DEFAULT_RESULT_LIMIT),
        NamespaceScope::default(),
        Some(DEFAULT_CONNECT_TIMEOUT),
        Some(DEFAULT_ABLATION_TIMEOUT),
        DEFAULT_IGNORE_CARTESIAN_RISK,
        DEFAULT_FIND_ALL_PATHS,
    )
}

/// Diagnoses `query_text` against `store` and attempts to connect each broken
/// culprit combination found.
///
/// `ablation_depth` is passed through to [`crate::diagnose::diagnose`]:
/// single triples are tried first, and only if none of those unblock the
/// query does diagnosis escalate to jointly removing pairs, then triples of
/// three, and so on up to `ablation_depth`.
///
/// `max_depth` bounds the forward/inverse graph-path search itself. `None`
/// (the default, see [`diagnose_and_connect_default`]) uses
/// [`DEFAULT_PAIR_SEARCH_DEPTH`]; passing `Some(n)` overrides it.
///
/// `sample_limit` caps how many of a culprit's bound endpoints are searched
/// for a path, or `None` to search every distinct endpoint found. Filters
/// flagged by diagnosis are reported as-is, with no connection attempted.
///
/// `result_limit` caps how many rows a connected query's `LIMIT` allows (only
/// tightening one already present in the original query, never loosening
/// it); `None` leaves it unbounded. Defaults to [`DEFAULT_RESULT_LIMIT`] via
/// [`diagnose_and_connect_default`], since a connected path (especially an
/// alternation of several distinct paths) can match far more broadly than
/// the original triple did.
///
/// `namespace_scope` restricts which predicates path search is allowed to
/// traverse (see [`NamespaceScope`]); pass [`NamespaceScope::Unrestricted`]
/// to search any real predicate found in the store, with no restriction.
///
/// `timeout` bounds all the work needed to connect *each* culprit combination:
/// resolving endpoints, the path search itself (bounded by hand rather than
/// via query cancellation — see the `bfs` module docs), and verifying a
/// candidate fix — not diagnosis, which has its own separate budget (see
/// `diagnose_timeout`). A combination that can't finish within its budget
/// falls back to [`ConnectedCulprit::pruned_query`] rather than hanging or
/// failing the whole call. Defaults to [`DEFAULT_CONNECT_TIMEOUT`] via
/// [`diagnose_and_connect_default`]; pass `None` to leave it unbounded.
///
/// `diagnose_timeout` is passed straight through to
/// [`crate::diagnose::diagnose`] — see its docs for what it bounds and why
/// an internally-enforced timeout (rather than relying on the caller to
/// abandon a slow call) matters. Independent of `timeout` above: diagnosis
/// runs once, before any connection work starts, so the two phases'
/// budgets don't interact.
///
/// `ignore_cartesian_risk` disables diagnosis's [`crate::algebra::has_cartesian_join`]
/// guard for this call: a culprit combination whose reduced pattern is
/// disconnected is actually evaluated against `store` instead of being
/// skipped and reported as a [`crate::diagnose::CartesianRiskCombo`], and if
/// confirmed genuine, connection attempts to fix it exactly like any other
/// culprit. `true` (see [`crate::diagnose::DEFAULT_IGNORE_CARTESIAN_RISK`],
/// the default via [`diagnose_and_connect_default`]) opts out of the
/// protection the guard applies: a disconnected BGP can make the query
/// engine materialize a full N×M cross product before yielding a single
/// row, regardless of how tightly `timeout` is set — a measured case
/// elsewhere in this project sat for over 200 seconds and permanently
/// occupied a shared worker thread until the whole process was killed (see
/// `eval/run_eval.py`'s process-level watchdog for why that backstop lives
/// at the process level, not inside this call). That's a deliberate
/// default, not a blind one — see `DEFAULT_IGNORE_CARTESIAN_RISK`'s docs
/// for the measured tradeoff that justifies it. Pass `false` to restore the
/// guard for a caller that can't tolerate the risk (no watchdog of its own
/// to kill and restart a wedged worker).
///
/// `find_all_paths` controls how many distinct paths are searched for per
/// broken triple — see [`DEFAULT_FIND_ALL_PATHS`] and
/// [`ConnectedTriple::hop_alternatives`]. `false` (the default via
/// [`diagnose_and_connect_default`]) stops at the first (shortest)
/// connecting path found; `true` searches every sampled bound endpoint and
/// keeps every distinct path found, in case different endpoints genuinely
/// need different ones.
///
/// Builds a fresh [`FanoutIndex`] from `store` on every call — see
/// [`diagnose_and_connect_with_fanout_index`], which this delegates to, for
/// what that's for. Fine for a one-off query; a caller connecting many
/// queries against the same `store` should build the index once with
/// [`FanoutIndex::build`] and call [`diagnose_and_connect_with_fanout_index`]
/// directly instead, the same way a repeated caller should prefer building
/// one `Store` over passing raw text here repeatedly (see this module's own
/// docs and `sparql-relax-py`'s `Store` docstring).
#[allow(clippy::too_many_arguments)]
pub fn diagnose_and_connect(
    query_text: &str,
    store: &Store,
    ablation_depth: usize,
    max_depth: Option<usize>,
    sample_limit: Option<usize>,
    result_limit: Option<usize>,
    namespace_scope: NamespaceScope,
    timeout: Option<Duration>,
    diagnose_timeout: Option<Duration>,
    ignore_cartesian_risk: bool,
    find_all_paths: bool,
) -> Result<ConnectReport> {
    let fanout_index = FanoutIndex::build(store, namespace_scope.as_filter());
    diagnose_and_connect_with_fanout_index(
        query_text,
        store,
        &fanout_index,
        ablation_depth,
        max_depth,
        sample_limit,
        result_limit,
        namespace_scope,
        timeout,
        diagnose_timeout,
        ignore_cartesian_risk,
        find_all_paths,
    )
}

/// Same as [`diagnose_and_connect`], but takes an already-built
/// [`FanoutIndex`] instead of building one fresh — for a caller (like
/// `sparql-relax-py`'s `Store`) that connects many queries against the same
/// `store` and wants to build the index once, up front, rather than
/// re-scanning the whole graph on every call. The index only matters for
/// path search's candidate filtering (see [`crate::bfs::find_path`] and
/// [`crate::fanout`]'s module docs); it plays no role in diagnosis itself.
///
/// `fanout_index` should have been built with the same `allowed_namespaces`
/// as `namespace_scope` below resolves to (see [`FanoutIndex::build`]):
/// its graph-wide degree cap is only computed over predicates a namespace
/// filter admits, so a mismatched `fanout_index` (e.g. built unrestricted,
/// then called here with a narrower `namespace_scope`) would rank fan-out
/// against a different, larger population than the one path search is
/// actually restricted to. A caller that always uses the same
/// `namespace_scope` for a given `Store` (the common case) never has to
/// think about this.
#[allow(clippy::too_many_arguments)]
pub fn diagnose_and_connect_with_fanout_index(
    query_text: &str,
    store: &Store,
    fanout_index: &FanoutIndex,
    ablation_depth: usize,
    max_depth: Option<usize>,
    sample_limit: Option<usize>,
    result_limit: Option<usize>,
    namespace_scope: NamespaceScope,
    timeout: Option<Duration>,
    diagnose_timeout: Option<Duration>,
    ignore_cartesian_risk: bool,
    find_all_paths: bool,
) -> Result<ConnectReport> {
    let query = SparqlParser::new().parse_query(query_text)?;
    ensure_select(&query)?;
    let pattern = pattern_of(&query).clone();
    let diagnosis = diagnose_parsed(&query, &pattern, store, ablation_depth, diagnose_timeout, ignore_cartesian_risk)?;
    let allowed_namespaces = namespace_scope.as_filter();

    // Every culprit combination is connected independently against the same
    // read-only `store`, so search them all in parallel. Each gets its own
    // fresh `timeout` budget, computed from the moment its own connection
    // work starts rather than shared off one call-wide clock — a slow
    // combination's budget shouldn't eat into a different combination's.
    let results = diagnosis
        .culprits
        .par_iter()
        .map(|culprit| {
            connect_combo(
                &query,
                &pattern,
                culprit,
                store,
                max_depth,
                sample_limit,
                result_limit,
                allowed_namespaces,
                fanout_index,
                timeout,
                find_all_paths,
            )
        })
        .collect::<Result<Vec<_>>>()?;

    let filter_results = diagnosis
        .filter_culprits
        .into_iter()
        .map(|f| FilterReport {
            expression_text: f.expression.to_string(),
            row_count_without_filter: f.row_count_without_filter,
        })
        .collect();

    Ok(ConnectReport {
        original_row_count: diagnosis.original_row_count,
        results,
        filter_results,
        cartesian_risks: diagnosis.cartesian_risks,
    })
}

/// Every triple in a culprit combination removed together, the endpoints
/// that resolves for each, and the pruned fallback built from the same
/// reduced pattern (see [`ConnectedCulprit::pruned_query`]).
struct BoundEndpoints {
    per_triple: Vec<Vec<BoundEndpoint>>,
    pruned_query: String,
    pruned_row_count: usize,
}

/// Removes every triple in `culprit` together (mirroring how diagnosis
/// found it) and resolves each triple's subject/object against the
/// resulting rows, one endpoint list per triple in `culprit.triples`. A row
/// that only resolves one side of a triple contributes nothing — see the
/// module docs on why one-sided endpoints aren't searched.
///
/// `deadline` bounds the row materialization needed for endpoint binding —
/// the query most likely to balloon into a large join once the broken
/// triple(s) are out of the way (see the module docs). If it's hit, this
/// returns empty endpoint lists (search then finds nothing, same as if no
/// path existed) rather than propagating an error, so one slow combination
/// degrades gracefully instead of failing the whole call. `pruned_query`'s
/// text needs no store access at all, so it's still built and returned
/// either way; `pruned_row_count` falls back to `0` only in this same
/// timeout case (see its doc comment).
fn bind_endpoints(
    query: &Query,
    pattern: &GraphPattern,
    culprit: &Culprit,
    store: &Store,
    result_limit: Option<usize>,
    deadline: Option<Instant>,
) -> Result<BoundEndpoints> {
    // Removed one triple at a time (rather than `try_fold`'s single combined
    // `Option`) so a removal failure can name the specific triple that
    // wasn't found, not just whichever happened to be first in the combo.
    let mut reduced_pattern = pattern.clone();
    for triple in &culprit.triples {
        reduced_pattern = crate::algebra::remove_triple(&reduced_pattern, triple)
            .ok_or_else(|| RelaxError::CulpritNotFound(triple.to_string()))?;
    }

    let mut pruned_pattern = reduced_pattern;
    if let Some(limit) = result_limit {
        pruned_pattern = with_limit(pruned_pattern, limit);
    }
    let pruned_query = with_pattern(query, pruned_pattern.clone()).to_string();

    // Executes the *limited* pattern, not the raw `reduced_pattern` — this is
    // the query most likely to balloon once the culprit triple(s) are gone
    // (see the module docs), and endpoint sampling below only ever needs
    // `sample_limit` (a handful) distinct pairs per triple, so there's no
    // reason to let it materialize more than `result_limit` rows either.
    //
    // Widen the (limited) reduced query's projection so each culprit
    // triple's subject/object is visible in its rows even if it's a plain
    // WHERE-clause bridge variable never listed in the original SELECT —
    // see `widen_projection`'s docs for why the original projection alone
    // isn't enough for the `resolve_term_pattern` calls below.
    let widened_pattern = widen_projection(&pruned_pattern, &variables_of_triples(&culprit.triples));
    let reduced_query = with_pattern(query, widened_pattern);
    let Some(reduced_rows) = run_select_query_with_deadline(reduced_query, store, deadline)? else {
        let per_triple = culprit.triples.iter().map(|_| Vec::new()).collect();
        return Ok(BoundEndpoints { per_triple, pruned_query, pruned_row_count: 0 });
    };

    let per_triple = culprit
        .triples
        .iter()
        .map(|triple| {
            let mut endpoints: Vec<BoundEndpoint> = Vec::new();
            let mut seen: HashSet<BoundEndpoint> = HashSet::new();
            for row in &reduced_rows {
                let s = resolve_term_pattern(&triple.subject, row);
                let o = resolve_term_pattern(&triple.object, row);
                let (Some(s), Some(o)) = (s, o) else { continue };
                let endpoint = (s, o);
                if seen.insert(endpoint.clone()) {
                    endpoints.push(endpoint);
                }
            }
            endpoints
        })
        .collect();

    let pruned_row_count = reduced_rows.len();
    Ok(BoundEndpoints { per_triple, pruned_query, pruned_row_count })
}

/// Searches `sampled` for every distinct hop sequence connecting a culprit
/// triple's endpoints, sweeping depth levels one at a time — every sampled
/// endpoint is tried at 1 hop before *any* is tried at 2, then at 2 before
/// any at 3, and so on up to `max_depth`. A single [`crate::bfs::find_path`]
/// call already returns the shortest path for one endpoint, but that alone
/// doesn't stop a *different*, unrelated endpoint's longer coincidental
/// path from being discovered and accepted first, purely because it
/// happened to sort earlier in `sampled` — this sweep makes that
/// impossible: nothing at depth 2 is ever even attempted while an
/// unresolved endpoint might still have a depth-1 match waiting.
///
/// Each endpoint keeps its own [`FrontierSearch`] across the whole sweep,
/// advanced by exactly one hop per depth level, rather than a fresh
/// [`crate::bfs::find_path`] call per level — so an endpoint still
/// unresolved at depth 2 reuses the frontier/visited state its own depth-1
/// attempt already built, instead of re-fetching and re-filtering that same
/// first hop's edges over again.
///
/// Every sampled endpoint is searched to exhaustion — a longer path found
/// for one entity doesn't rule out a different, unrelated entity needing
/// its own separate path (see
/// `combines_distinct_paths_from_different_bound_pairs_as_alternatives`).
/// Used only when the caller passes `find_all_paths: true` (see
/// [`DEFAULT_FIND_ALL_PATHS`]); the default instead prefers
/// [`search_candidates_grouped`], which stops at the first connecting path
/// found rather than searching every sample for every distinct one.
fn search_candidates(
    store: &Store,
    sampled: &[&BoundEndpoint],
    max_depth: usize,
    allowed_namespaces: Option<&[String]>,
    fanout_index: &FanoutIndex,
    deadline: Option<Instant>,
) -> Vec<Vec<Hop>> {
    let mut candidates: Vec<Vec<Hop>> = Vec::new();
    let mut unresolved: Vec<(&BoundEndpoint, FrontierSearch)> =
        sampled.iter().map(|&endpoint| (endpoint, FrontierSearch::new(&endpoint.0))).collect();

    for _ in 1..=max_depth {
        let mut still_unresolved = Vec::new();
        for (endpoint, mut search) in unresolved {
            let (s, o) = endpoint;
            // Already covered by a candidate found at a shallower depth
            // this sweep (or reused from an earlier triple's search) — no
            // need to search this endpoint again at all.
            if candidates.iter().any(|hops| path_holds(store, s, o, hops)) {
                continue;
            }
            // `s == o` needs no path at all — matches `find_path`'s own
            // start-equals-goal shortcut (an empty hop sequence, which
            // contributes nothing to `candidates` and needs no further
            // searching), so this endpoint is simply dropped rather than
            // ever calling `advance`.
            if s == o {
                continue;
            }
            if search.is_exhausted() {
                continue;
            }
            let goal = HashSet::from([o.clone()]);
            match search.advance(store, &goal, allowed_namespaces, Some(fanout_index), deadline) {
                Some(hops) if !hops.is_empty() && !candidates.contains(&hops) => candidates.push(hops),
                // No path within this depth yet, and the frontier isn't
                // exhausted — still a candidate for a deeper sweep.
                None => still_unresolved.push((endpoint, search)),
                // `hops` already in `candidates`: nothing to add or retry.
                _ => {}
            }
        }
        unresolved = still_unresolved;
        if unresolved.is_empty() {
            break;
        }
    }

    candidates
}

/// Groups `endpoints` by subject, in first-seen order, keeping at most
/// `subject_limit` distinct subjects (`None` keeps every distinct subject)
/// — but, for whichever subjects are kept, *every* object they were ever
/// paired with, not just the first one encountered. Used to build
/// [`search_candidates_grouped`]'s per-subject goal sets: for a
/// cartesian-risk combo especially (whose endpoints are arbitrary
/// cross-joined pairs — see [`bind_endpoints`]), a subject's real match can
/// be anywhere among the objects it was crossed with, so truncating that
/// per-subject list the way a flat sample truncates pairs would reintroduce
/// the exact order-dependence this grouping exists to remove.
fn group_by_subject(endpoints: &[BoundEndpoint], subject_limit: Option<usize>) -> Vec<(Term, HashSet<Term>)> {
    let mut chosen: Vec<Term> = Vec::new();
    let mut chosen_set: HashSet<Term> = HashSet::new();
    for (s, _) in endpoints {
        if chosen_set.contains(s) {
            continue;
        }
        if subject_limit.is_some_and(|limit| chosen.len() >= limit) {
            continue;
        }
        chosen.push(s.clone());
        chosen_set.insert(s.clone());
    }

    let mut goals_by_subject: HashMap<Term, HashSet<Term>> = HashMap::new();
    for (s, o) in endpoints {
        if chosen_set.contains(s) {
            goals_by_subject.entry(s.clone()).or_default().insert(o.clone());
        }
    }

    chosen.into_iter().map(|s| { let goals = goals_by_subject.remove(&s).unwrap_or_default(); (s, goals) }).collect()
}

/// The default search strategy (see [`DEFAULT_FIND_ALL_PATHS`]): searches
/// by subject instead of by flat pair — up to `subject_limit` distinct
/// subjects (see [`group_by_subject`]), each checked against its *entire*
/// candidate-object set at once via [`crate::bfs::find_path_to_any`] rather
/// than one [`crate::bfs::find_path`] call per candidate object. That
/// guarantees whichever subjects get chosen find their real match if one
/// exists — no dependence on where in an arbitrarily-ordered flat sample it
/// happens to fall — and it's cheaper too: one BFS expansion per subject
/// serves every one of its candidates, instead of re-fetching and
/// re-filtering the same subject's edges once per candidate. That property
/// matters most for a cartesian-risk combo, whose bound endpoints are an
/// arbitrary cross join rather than genuinely-related rows (see
/// [`bind_endpoints`]) — a flat sample there can easily exhaust one
/// subject's mismatched pairings before ever reaching another's real match.
///
/// Sweeps depth levels the same way [`search_candidates`] does — every
/// chosen subject is tried at 1 hop before any is tried at 2 — but, unlike
/// [`search_candidates`], stops entirely as soon as *any* depth level
/// produces a candidate, since the point of this strategy is exactly to
/// return the first (shortest) connecting path found rather than every
/// distinct one every sampled subject might individually need — that
/// exhaustive alternative is what `find_all_paths: true` (→
/// [`search_candidates`]) is for.
///
/// Each subject keeps its own [`FrontierSearch`] across the whole sweep (see
/// [`search_candidates`]'s docs on why), advanced by one hop per depth level
/// rather than restarted via a fresh [`crate::bfs::find_path_to_any`] call
/// each time.
fn search_candidates_grouped(
    store: &Store,
    endpoints: &[BoundEndpoint],
    subject_limit: Option<usize>,
    max_depth: usize,
    allowed_namespaces: Option<&[String]>,
    fanout_index: &FanoutIndex,
    deadline: Option<Instant>,
) -> Vec<Vec<Hop>> {
    let groups = group_by_subject(endpoints, subject_limit);
    let mut candidates: Vec<Vec<Hop>> = Vec::new();
    let mut unresolved: Vec<(Term, HashSet<Term>, FrontierSearch)> =
        groups.into_iter().map(|(s, goals)| { let search = FrontierSearch::new(&s); (s, goals, search) }).collect();

    for _ in 1..=max_depth {
        let mut still_unresolved = Vec::new();
        for (subject, goals, mut search) in unresolved {
            // Drop any goal already covered by a candidate found at a
            // shallower depth this sweep, so a subject whose only
            // remaining candidates are already explained by an existing
            // hop sequence isn't searched again for nothing.
            let remaining_goals: HashSet<Term> =
                goals.iter().filter(|g| !candidates.iter().any(|hops| path_holds(store, &subject, g, hops))).cloned().collect();
            if remaining_goals.is_empty() {
                continue;
            }
            // A subject that already equals one of its own remaining goals
            // needs no path at all — matches `find_path_to_any`'s own
            // `goals.contains(start)` shortcut (an empty hop sequence,
            // which contributes nothing to `candidates`), so this subject
            // is simply dropped rather than ever calling `advance`.
            if remaining_goals.contains(&subject) {
                continue;
            }
            if search.is_exhausted() {
                continue;
            }
            match search.advance(store, &remaining_goals, allowed_namespaces, Some(fanout_index), deadline) {
                Some(hops) if !hops.is_empty() && !candidates.contains(&hops) => candidates.push(hops),
                None => still_unresolved.push((subject, goals, search)),
                _ => {}
            }
        }
        unresolved = still_unresolved;

        if !candidates.is_empty() {
            break;
        }
        if unresolved.is_empty() {
            break;
        }
    }

    candidates
}

#[allow(clippy::too_many_arguments)]
fn connect_combo(
    query: &Query,
    pattern: &GraphPattern,
    culprit: &Culprit,
    store: &Store,
    max_depth: Option<usize>,
    sample_limit: Option<usize>,
    result_limit: Option<usize>,
    allowed_namespaces: Option<&[String]>,
    fanout_index: &FanoutIndex,
    timeout: Option<Duration>,
    find_all_paths: bool,
) -> Result<ConnectedCulprit> {
    // One budget for all of this combination's query work — resolving
    // endpoints and (later) verifying the candidate fix both draw from it,
    // rather than each getting its own fresh `timeout`.
    let deadline = timeout.map(|t| Instant::now() + t);

    let bound = bind_endpoints(query, pattern, culprit, store, result_limit, deadline)?;
    let pruned_query = bound.pruned_query;
    let pruned_row_count = bound.pruned_row_count;

    // Every triple in the combination is searched independently, so this
    // level runs in parallel — each is just reads against the same `store`.
    // Within a single triple's sampled endpoints, the search is sequential
    // instead (see the comment below on why).
    let effective_max_depth = max_depth.unwrap_or(DEFAULT_PAIR_SEARCH_DEPTH);
    let per_triple: Vec<(ConnectedTriple, Option<PropertyPathExpression>)> = culprit
        .triples
        .par_iter()
        .zip(bound.per_triple.par_iter())
        .map(|(triple, endpoints)| {
            // Sequential rather than `par_iter` over sampled endpoints: a
            // hop sequence already found for one often generalizes to
            // another (e.g. every sensor in the same building reached via
            // the same 2-hop path), and `path_holds` can confirm that with
            // a handful of direct store lookups — far cheaper than a fresh
            // bounded BFS. Trying already-found candidates first lets later
            // endpoints skip a fresh search entirely whenever one
            // generalizes, which matters more than the parallelism given up
            // here (`sample_limit` keeps this small regardless).
            //
            // `find_all_paths` (default `false`, see
            // `DEFAULT_FIND_ALL_PATHS`) picks the search strategy: by
            // default, `search_candidates_grouped` searches by subject and
            // stops at the first (shortest) connecting path found, rather
            // than `search_candidates`'s flat per-pair sampling that
            // searches every sample for every distinct path it might need.
            let mut candidates = if find_all_paths {
                let sampled: Vec<&BoundEndpoint> = match sample_limit {
                    Some(limit) => endpoints.iter().take(limit).collect(),
                    None => endpoints.iter().collect(),
                };
                search_candidates(store, &sampled, effective_max_depth, allowed_namespaces, fanout_index, deadline)
            } else {
                search_candidates_grouped(store, endpoints, sample_limit, effective_max_depth, allowed_namespaces, fanout_index, deadline)
            };
            candidates.sort_by_key(Vec::len);

            let path_expr = combine_as_alternatives(&candidates);
            let connected_triple = ConnectedTriple {
                triple_text: triple.to_string(),
                hop_alternatives: candidates,
                path_text: path_expr.as_ref().map(PropertyPathExpression::to_string),
            };
            (connected_triple, path_expr)
        })
        .collect();

    let (connected_triples, paths): (Vec<_>, Vec<_>) = per_triple.into_iter().unzip();

    // Splice in a path for every triple that found one, and simply drop
    // (prune) whichever ones didn't — rather than requiring the whole
    // combination to have a path for every triple before building anything.
    // A pair where one triple gets a real path substitution and the other
    // just gets dropped is still a strictly better candidate than dropping
    // both, and it's re-verified against the graph below exactly like a
    // fully-connected query is, so a bad partial fix still scores as empty
    // rather than being trusted blindly. Only when *none* of the
    // combination's triples found a path is there nothing to splice in, so
    // that case alone falls back to `pruned_query` (identical to it, so
    // building a redundant `connected_query` would add nothing).
    let any_found = paths.iter().any(Option::is_some);
    if !any_found {
        return Ok(ConnectedCulprit {
            found_at_depth: culprit.depth,
            triples: connected_triples,
            connected_query: None,
            row_count: 0,
            pruned_query,
            pruned_row_count,
        });
    }

    let mut connected_pattern = pattern.clone();
    for (triple, path_expr) in culprit.triples.iter().zip(paths.into_iter()) {
        let next = match path_expr {
            Some(path_expr) => replace_triple_with_path(&connected_pattern, triple, path_expr),
            None => remove_triple(&connected_pattern, triple),
        };
        let Some(next) = next else {
            return Ok(ConnectedCulprit {
                found_at_depth: culprit.depth,
                triples: connected_triples,
                connected_query: None,
                row_count: 0,
                pruned_query,
                pruned_row_count,
            });
        };
        connected_pattern = next;
    }

    if let Some(limit) = result_limit {
        connected_pattern = with_limit(connected_pattern, limit);
    }
    let connected_query_obj = with_pattern(query, connected_pattern);
    let connected_text = connected_query_obj.to_string();

    // A candidate fix found in time but too expensive to *verify* in what's
    // left of the budget falls back the same way an unfound path does:
    // `connected_query: None`, so the caller looks at `pruned_query` instead
    // of trusting an unconfirmed `row_count`.
    match run_select_query_with_deadline(connected_query_obj, store, deadline) {
        Ok(Some(rows)) => Ok(ConnectedCulprit {
            found_at_depth: culprit.depth,
            triples: connected_triples,
            connected_query: Some(connected_text),
            row_count: rows.len(),
            pruned_query,
            pruned_row_count,
        }),
        Ok(None) => Ok(ConnectedCulprit {
            found_at_depth: culprit.depth,
            triples: connected_triples,
            connected_query: None,
            row_count: 0,
            pruned_query,
            pruned_row_count,
        }),
        Err(_) => Ok(ConnectedCulprit {
            found_at_depth: culprit.depth,
            triples: connected_triples,
            connected_query: Some(connected_text),
            row_count: 0,
            pruned_query,
            pruned_row_count,
        }),
    }
}

/// Folds every candidate hop sequence into one `PropertyPathExpression`,
/// joined with SPARQL's `|` alternation, so the connected query can match
/// through any of them rather than only the one path search happened to
/// prefer.
fn combine_as_alternatives(candidates: &[Vec<Hop>]) -> Option<PropertyPathExpression> {
    let mut iter = candidates.iter();
    let mut expr = path_to_property_path(iter.next()?)?;
    for hops in iter {
        expr = PropertyPathExpression::Alternative(Box::new(expr), Box::new(path_to_property_path(hops)?));
    }
    Some(expr)
}
