//! Orchestrates diagnosis and relaxation: for each culprit combination the
//! ablation diagnosis in [`crate::diagnose`] flags as broken (a single
//! triple at `ablation_depth` 1, or several triples jointly responsible at
//! a higher depth), resolves what its variables are actually bound to —
//! diagnosis itself does none of this binding work, so a plain diagnosis
//! call never pays for it — then searches each triple's bound endpoints for
//! a real forward/inverse path (via [`crate::bfs`]), splices *all* of the
//! combination's triples in place at once, and confirms the result by
//! actually re-running the modified query. A combination is only relaxed as
//! a whole: if any one of its triples has no discoverable path, the others
//! being fixable wouldn't produce a working query on its own (they're only
//! broken *together*), so no relaxed query is built for it.
//!
//! A broken triple's other side is sometimes not bound anywhere else in the
//! query (e.g. `?sensor` in `building hasSensor ?sensor` if nothing else
//! constrains `?sensor`), so there's no specific target to search for. When
//! only one side resolves, path search instead explores outward from that
//! single anchor with no fixed goal and returns whatever real paths it
//! finds as suggestions — not verified fixes, since there's nothing to
//! verify against except by actually re-running the whole query afterward.

use crate::algebra::{pattern_of, replace_triple_with_path, with_pattern};
use crate::bfs::{Hop, explore_from, find_path, path_to_property_path, reverse_hops};
use crate::diagnose::{Culprit, DEFAULT_ABLATION_DEPTH, diagnose, resolve_term_pattern, run_select};
use crate::error::Result;
use oxigraph::model::Term;
use oxigraph::store::Store;
use spargebra::Query;
use spargebra::SparqlParser;
use spargebra::algebra::{GraphPattern, PropertyPathExpression};

/// Default for `sample_limit`: a representative handful of bound endpoints
/// is normally enough to find a generalizable path without examining every
/// row of a potentially large reduced query.
pub const DEFAULT_SAMPLE_LIMIT: usize = 5;

/// Default path-search depth when both a culprit triple's subject and
/// object are known (a concrete point-to-point search, bounded by its
/// target — cheap enough to search a little deeper).
pub const DEFAULT_PAIR_SEARCH_DEPTH: usize = 2;

/// Default path-search depth when only one side of a culprit triple is
/// known (an undirected exploration with no fixed goal — more expensive per
/// level, so shallower by default).
pub const DEFAULT_ANCHOR_SEARCH_DEPTH: usize = 1;

/// What a broken triple's subject/object resolved to, for one binding of
/// the rest of the query. Both sides resolving gives a concrete
/// point-to-point path search; only one resolving still gives *something*
/// to search outward from, just without a specific target.
#[derive(PartialEq, Eq, Clone, Hash)]
enum BoundEndpoint {
    Pair(Term, Term),
    SubjectOnly(Term),
    ObjectOnly(Term),
}

/// One triple within a relaxed culprit combination, and what path search
/// found for it specifically.
pub struct RelaxedTriple {
    /// The broken triple pattern, as SPARQL text (e.g. `?s <p> ?o`).
    pub triple_text: String,
    /// Every distinct forward/inverse hop sequence found (one per sampled
    /// bound endpoint, deduplicated), combined into the path below via `|`.
    /// Different endpoints can genuinely need different paths (e.g. one
    /// entity reached via a 2-hop path, another via an unrelated 1-hop
    /// path) — picking only one would silently drop the others, so every
    /// distinct path found is kept and used.
    pub hop_alternatives: Vec<Vec<Hop>>,
    /// The hop alternatives rendered as a single SPARQL property path (e.g.
    /// `<p1>/<p2>` alone, or `(<p1>/<p2>)|<p3>` when more than one distinct
    /// path was found). `None` if no connecting path was found.
    pub path_text: Option<String>,
}

pub struct RelaxedCulprit {
    /// The ablation combination size at which this culprit was found (see
    /// `ablation_depth` on [`diagnose_and_relax`]); 1 unless it was only
    /// found jointly responsible alongside other triples.
    pub found_at_depth: usize,
    /// Every triple in the culprit combination, each with its own path
    /// search result, in the same order they appear in the query.
    pub triples: Vec<RelaxedTriple>,
    /// The full query with every triple above replaced by its discovered
    /// path, only present if *all* of them had one — relaxing just some of
    /// a jointly-broken combination wouldn't produce a working query, since
    /// the others are still broken on their own.
    pub relaxed_query: Option<String>,
    /// Row count of `relaxed_query` when re-executed. Zero if no combined
    /// relaxation was built, or it still returns nothing.
    pub row_count: usize,
}

/// A `FILTER` flagged by ablation as excluding rows. Reported, not relaxed:
/// there's no graph-path search that applies to an arbitrary expression.
pub struct FilterReport {
    /// The filter expression, as SPARQL text (e.g. `?o > 5`).
    pub expression_text: String,
    /// Row count of the query with just this filter removed.
    pub row_count_without_filter: usize,
}

pub struct RelaxReport {
    pub original_row_count: usize,
    pub results: Vec<RelaxedCulprit>,
    pub filter_results: Vec<FilterReport>,
}

/// Diagnoses `query_text` against `store` with [`DEFAULT_ABLATION_DEPTH`],
/// [`DEFAULT_SAMPLE_LIMIT`], and adaptive per-endpoint path-search depth
/// (see [`diagnose_and_relax`]), and attempts to relax each broken culprit
/// combination found. A convenient default for callers who don't need to
/// tune the search.
pub fn diagnose_and_relax_default(query_text: &str, store: &Store) -> Result<RelaxReport> {
    diagnose_and_relax(query_text, store, DEFAULT_ABLATION_DEPTH, None, Some(DEFAULT_SAMPLE_LIMIT))
}

/// Diagnoses `query_text` against `store` and attempts to relax each broken
/// culprit combination found.
///
/// `ablation_depth` is passed through to [`crate::diagnose::diagnose`]:
/// single triples are tried first, and only if none of those unblock the
/// query does diagnosis escalate to jointly removing pairs, then triples of
/// three, and so on up to `ablation_depth`.
///
/// `max_depth` bounds the forward/inverse graph-path search itself. `None`
/// (the default, see [`diagnose_and_relax_default`]) uses an adaptive depth
/// per endpoint instead of one fixed value: [`DEFAULT_PAIR_SEARCH_DEPTH`]
/// when a culprit triple's subject and object are both known (a concrete,
/// target-bounded search), or the shallower [`DEFAULT_ANCHOR_SEARCH_DEPTH`]
/// when only one side is known (an undirected exploration with no fixed
/// goal, so more expensive per level). Passing `Some(n)` overrides both
/// cases uniformly.
///
/// `sample_limit` caps how many of a culprit's bound endpoints are searched
/// for a path, or `None` to search every distinct endpoint found. Filters
/// flagged by diagnosis are reported as-is, with no relaxation attempted.
pub fn diagnose_and_relax(
    query_text: &str,
    store: &Store,
    ablation_depth: usize,
    max_depth: Option<usize>,
    sample_limit: Option<usize>,
) -> Result<RelaxReport> {
    let diagnosis = diagnose(query_text, store, ablation_depth)?;
    let query = SparqlParser::new().parse_query(query_text)?;
    let pattern = pattern_of(&query).clone();

    let results = diagnosis
        .culprits
        .iter()
        .map(|culprit| relax_combo(&query, &pattern, culprit, store, max_depth, sample_limit))
        .collect::<Result<Vec<_>>>()?;

    let filter_results = diagnosis
        .filter_culprits
        .into_iter()
        .map(|f| FilterReport {
            expression_text: f.expression.to_string(),
            row_count_without_filter: f.row_count_without_filter,
        })
        .collect();

    Ok(RelaxReport { original_row_count: diagnosis.original_row_count, results, filter_results })
}

/// Removes every triple in `culprit` together (mirroring how diagnosis
/// found it) and resolves each triple's subject/object against the
/// resulting rows, one endpoint list per triple in `culprit.triples`.
fn bind_endpoints(
    query: &Query,
    pattern: &GraphPattern,
    culprit: &Culprit,
    store: &Store,
) -> Result<Vec<Vec<BoundEndpoint>>> {
    let reduced_pattern = culprit
        .triples
        .iter()
        .try_fold(pattern.clone(), |p, t| crate::algebra::remove_triple(&p, t))
        .expect("culprit triples came from diagnosing this same query, so they must be present");
    let reduced_text = with_pattern(query, reduced_pattern).to_string();
    let reduced_rows = run_select(&reduced_text, store)?;

    Ok(culprit
        .triples
        .iter()
        .map(|triple| {
            let mut endpoints: Vec<BoundEndpoint> = Vec::new();
            for row in &reduced_rows {
                let s = resolve_term_pattern(&triple.subject, row);
                let o = resolve_term_pattern(&triple.object, row);
                let endpoint = match (s, o) {
                    (Some(s), Some(o)) => BoundEndpoint::Pair(s, o),
                    (Some(s), None) => BoundEndpoint::SubjectOnly(s),
                    (None, Some(o)) => BoundEndpoint::ObjectOnly(o),
                    (None, None) => continue,
                };
                if !endpoints.contains(&endpoint) {
                    endpoints.push(endpoint);
                }
            }
            endpoints
        })
        .collect())
}

/// Candidate hop sequences (subject → object) for one bound endpoint:
/// a concrete point-to-point search when both sides are known, or an
/// undirected exploration from whichever single side is known otherwise.
/// `max_depth_override` of `None` picks [`DEFAULT_PAIR_SEARCH_DEPTH`] or
/// [`DEFAULT_ANCHOR_SEARCH_DEPTH`] depending on which kind of endpoint this
/// is; `Some(n)` uses `n` for either kind.
fn candidates_for(store: &Store, endpoint: &BoundEndpoint, max_depth_override: Option<usize>) -> Vec<Vec<Hop>> {
    match endpoint {
        BoundEndpoint::Pair(s, o) => {
            let depth = max_depth_override.unwrap_or(DEFAULT_PAIR_SEARCH_DEPTH);
            find_path(store, s, o, depth).into_iter().collect()
        }
        BoundEndpoint::SubjectOnly(s) => {
            let depth = max_depth_override.unwrap_or(DEFAULT_ANCHOR_SEARCH_DEPTH);
            explore_from(store, s, depth)
        }
        BoundEndpoint::ObjectOnly(o) => {
            let depth = max_depth_override.unwrap_or(DEFAULT_ANCHOR_SEARCH_DEPTH);
            explore_from(store, o, depth).iter().map(|hops| reverse_hops(hops)).collect()
        }
    }
}

fn relax_combo(
    query: &Query,
    pattern: &GraphPattern,
    culprit: &Culprit,
    store: &Store,
    max_depth: Option<usize>,
    sample_limit: Option<usize>,
) -> Result<RelaxedCulprit> {
    let bound_endpoints = bind_endpoints(query, pattern, culprit, store)?;

    let mut relaxed_triples = Vec::new();
    let mut paths = Vec::new(); // Some(path_expr) per triple, parallel to culprit.triples

    for (triple, endpoints) in culprit.triples.iter().zip(&bound_endpoints) {
        let sampled: Vec<&BoundEndpoint> = match sample_limit {
            Some(limit) => endpoints.iter().take(limit).collect(),
            None => endpoints.iter().collect(),
        };

        let mut candidates: Vec<Vec<Hop>> = Vec::new();
        for endpoint in &sampled {
            for hops in candidates_for(store, endpoint, max_depth) {
                if !hops.is_empty() && !candidates.contains(&hops) {
                    candidates.push(hops);
                }
            }
        }
        candidates.sort_by_key(Vec::len);

        let path_expr = combine_as_alternatives(&candidates);
        relaxed_triples.push(RelaxedTriple {
            triple_text: triple.to_string(),
            hop_alternatives: candidates,
            path_text: path_expr.as_ref().map(PropertyPathExpression::to_string),
        });
        paths.push(path_expr);
    }

    // Only build a combined relaxed query if every triple in the
    // combination had a discoverable path — otherwise the ones without one
    // are still broken and the query would still fail regardless.
    let all_found = paths.iter().all(Option::is_some);
    if !all_found {
        return Ok(RelaxedCulprit { found_at_depth: culprit.depth, triples: relaxed_triples, relaxed_query: None, row_count: 0 });
    }

    let mut relaxed_pattern = pattern.clone();
    for (triple, path_expr) in culprit.triples.iter().zip(paths.into_iter().flatten()) {
        let Some(next) = replace_triple_with_path(&relaxed_pattern, triple, path_expr) else {
            return Ok(RelaxedCulprit { found_at_depth: culprit.depth, triples: relaxed_triples, relaxed_query: None, row_count: 0 });
        };
        relaxed_pattern = next;
    }

    let relaxed_text = with_pattern(query, relaxed_pattern).to_string();
    let row_count = run_select(&relaxed_text, store).map(|rows| rows.len()).unwrap_or(0);

    Ok(RelaxedCulprit {
        found_at_depth: culprit.depth,
        triples: relaxed_triples,
        relaxed_query: Some(relaxed_text),
        row_count,
    })
}

/// Folds every candidate hop sequence into one `PropertyPathExpression`,
/// joined with SPARQL's `|` alternation, so the relaxed query can match
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
