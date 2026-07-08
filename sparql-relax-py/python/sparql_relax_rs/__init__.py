"""Python bindings for sparql-relax-rs.

Diagnoses which basic-graph-pattern triple(s) or FILTER expression(s) in a
SPARQL query are likely responsible for it returning empty/wrong results
against a given RDF graph (ablation-style, like `sparql_prune.py`), then
searches for a real forward/inverse path over the graph's actual edges to
fix each broken triple (replacing the frequency-based predicate-path search
in `sparql_relax.py` with an actual BFS over Oxigraph's triples). Filters are
only ever reported, never relaxed.

Diagnosis can also look for *combinations* of jointly-broken triples, not
just single ones: pass `depth > 1` to `diagnose`/`diagnose_and_relax`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ._sparql_relax_rs import diagnose as _diagnose
from ._sparql_relax_rs import diagnose_and_relax as _diagnose_and_relax

__all__ = [
    "Culprit",
    "FilterCulprit",
    "Diagnosis",
    "RelaxedTriple",
    "RelaxedCulprit",
    "FilterReport",
    "RelaxReport",
    "diagnose",
    "diagnose_and_relax",
]


@dataclass
class Culprit:
    """One or more BGP triples whose *joint* removal unblocks the query, and which never jointly
    hold for any binding of the rest of the query.

    Has one triple in `triples` unless a single-triple removal wasn't enough to unblock the query
    and a larger combination (see `depth` on `diagnose`) was needed to find it. This only
    identifies *which* triple(s) are broken — no variable binding is done here (that only happens,
    for culprits diagnosis already found, in `diagnose_and_relax`), so a plain `diagnose` call
    doesn't pay for resolution work it won't use.
    """

    triples: List[str]
    """One SPARQL text form per triple in the combination (e.g. `?s <p> ?o`)."""

    depth: int
    """The combination size at which this culprit was found (1 = a single broken triple, 2 = a
    pair jointly responsible, ...)."""


@dataclass
class FilterCulprit:
    """A FILTER expression whose removal strictly grew the result set."""

    expression: str
    row_count_without_filter: int


@dataclass
class Diagnosis:
    original_row_count: int
    culprits: List[Culprit]
    filter_culprits: List[FilterCulprit]


@dataclass
class RelaxedTriple:
    """One triple within a relaxed culprit combination, and what path search found for it."""

    triple: str
    path_text: Optional[str]
    """The discovered path(s) rendered as a SPARQL property path, e.g. `<p1>/<p2>` or
    `(<p1>/<p2>)|<p3>` when more than one distinct connecting path was found. `None` if no path
    was found within `max_depth` hops."""


@dataclass
class RelaxedCulprit:
    found_at_depth: int
    """The combination size at which this culprit was found (see `Culprit.depth`)."""

    triples: List[RelaxedTriple]
    """Every triple in the culprit combination, each with its own path search result."""

    relaxed_query: Optional[str]
    """The full query with every triple above replaced by its discovered path. Only present if
    *all* of them had one — relaxing just some of a jointly-broken combination wouldn't produce a
    working query, since the others are still broken on their own."""

    row_count: int
    """Row count of `relaxed_query` when re-executed. Zero if no combined relaxation was built, or
    it still returns nothing."""

    @property
    def fixed(self) -> bool:
        """Whether a relaxed query was found that returns at least one row."""
        return self.row_count > 0


@dataclass
class FilterReport:
    """A FILTER flagged by diagnosis, reported as-is (never relaxed)."""

    expression: str
    row_count_without_filter: int


@dataclass
class RelaxReport:
    original_row_count: int
    results: List[RelaxedCulprit]
    filter_results: List[FilterReport]


def diagnose(data: str, query: str, format: str = "turtle", depth: int = 3) -> Diagnosis:
    """Diagnoses which BGP triple(s)/FILTER(s) in `query` are likely broken against `data`.

    `data` is RDF text in the given `format` (default Turtle); `query` must be a SPARQL SELECT.
    Applies the same remove-it-and-see-what-happens ablation logic to both basic-graph-pattern
    triples with concrete predicates and FILTER expressions (including the condition on an
    OPTIONAL). A triple is flagged when its predicate never holds for any binding of the rest of
    the query; a filter is flagged when removing it alone strictly grows the result set.

    `depth` controls how many triples may be removed *together* while searching for a culprit:
    single triples are always tried first (depth 1); if none of those unblock the query, pairs are
    tried (depth 2), then triples of three (depth 3), and so on up to `depth`. The search stops
    escalating as soon as some combination at the current size unblocks the query, or once the
    combination size would exceed the number of candidate triples. This catches queries where no
    single triple is broken, but two (or more) are jointly responsible for the empty/wrong result.
    Defaults to 3, which keeps the (combinatorial) search cheap while still catching the common
    case of two or three jointly-broken triples. `depth` does not apply to FILTER ablation (always
    tried one at a time).

    This only identifies *which* triple(s)/filter(s) are broken — it does no variable-binding
    work, so it's cheap even on large result sets. Use `diagnose_and_relax` to also resolve what a
    culprit's variables are bound to and search for a fix.
    """
    original_row_count, culprits, filter_culprits = _diagnose(data, query, format, depth)
    return Diagnosis(
        original_row_count=original_row_count,
        culprits=[Culprit(triples=triples, depth=culprit_depth) for triples, culprit_depth in culprits],
        filter_culprits=[
            FilterCulprit(expression=e, row_count_without_filter=n) for e, n in filter_culprits
        ],
    )


def diagnose_and_relax(
    data: str,
    query: str,
    format: str = "turtle",
    ablation_depth: int = 3,
    max_depth: Optional[int] = None,
    sample_limit: Optional[int] = 5,
) -> RelaxReport:
    """Diagnoses `query` and searches for a real forward/inverse graph path fixing each culprit
    combination found.

    This does the variable-binding work `diagnose` skips: for each culprit combination, it removes
    every triple in it, re-runs the rest of the query, and resolves each triple's subject/object
    against the resulting rows. For each triple, this gives a bounded breadth-first search over the
    graph's real edges between its bound endpoints, trying both a forward (`<p>`) and inverse
    (`^<p>`) step at each hop. Different bound pairs can need genuinely different real paths (one
    entity reached via a 2-hop path, another via an unrelated 1-hop path); rather than picking just
    one, every distinct path found for that triple is combined into a single SPARQL alternation
    (`|`) so the fix recovers all of them.

    Sometimes a broken triple's other side isn't bound anywhere else in the query at all (e.g.
    `?sensor` in `building hasSensor ?sensor` if nothing else constrains `?sensor`) — there's no
    specific target to search for then, only whichever single side did resolve to explore outward
    from. In that case the returned path(s) are suggestions related to that one known side, not
    verified fixes; the real correctness check is always the re-executed row count.

    A combination is only relaxed as a whole: if any one of its triples has no discoverable path,
    no combined relaxed query is built (the others being fixable wouldn't help, since they were
    only broken *together*). Filters flagged by diagnosis are included in `filter_results` as-is;
    no relaxation is attempted for them.

    `ablation_depth` is passed through to `diagnose` to control how many triples may be jointly
    removed while searching for a culprit (see `diagnose`'s `depth`; same default of 3).

    `max_depth` bounds the path search itself. Left as `None` (the default), it adapts to how much
    of a triple resolved: depth 2 when both its subject and object are bound (a concrete,
    target-bounded search), or the shallower depth 1 when only one side is bound (an undirected
    exploration with no fixed goal, so more expensive per level). Pass an explicit integer to
    override both cases uniformly.

    `sample_limit` caps how many distinct bound pairs are considered per triple; defaults to 5 (a
    representative sample rather than every row). Pass `None` to consider every distinct pair
    instead of stopping early (only worth doing for small graphs/result sets, since it means
    examining every row the reduced query returns).
    """
    original_row_count, results, filter_results = _diagnose_and_relax(
        data, query, format, ablation_depth, max_depth, sample_limit
    )
    return RelaxReport(
        original_row_count=original_row_count,
        results=[
            RelaxedCulprit(
                found_at_depth=found_at_depth,
                triples=[RelaxedTriple(triple=t, path_text=p) for t, p in triples],
                relaxed_query=relaxed_query,
                row_count=row_count,
            )
            for found_at_depth, triples, relaxed_query, row_count in results
        ],
        filter_results=[
            FilterReport(expression=e, row_count_without_filter=n) for e, n in filter_results
        ],
    )
