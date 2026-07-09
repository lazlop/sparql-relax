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
from typing import List, Optional, Sequence

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
    "DEFAULT_RELAX_NAMESPACES",
    "DEFAULT_RELAX_TIMEOUT",
]

DEFAULT_RELAX_NAMESPACES: tuple[str, ...] = (
    "https://brickschema.org/schema/Brick#",  # brick:
    "https://brickschema.org/schema/Brick/",  # ref: (covers both ref# and bare Brick/)
    "http://data.ashrae.org/standard223#",  # s223:
    "http://www.w3.org/2000/01/rdf-schema#",  # rdfs:
    "http://qudt.org/schema/qudt/",  # qudt:
)
"""Namespace prefixes `diagnose_and_relax` restricts path search to by default: the
building-automation ontologies (Brick, ASHRAE 223P, RDFS, QUDT) this tool is normally used
against. Pass a different sequence via `allowed_namespaces`, or `None` for no restriction."""

DEFAULT_RELAX_TIMEOUT: float = 5.0
"""Default `timeout` (seconds) for `diagnose_and_relax`: the SPARQL query work needed to relax
each culprit combination is normally well under this. Pass `None` to leave it unbounded."""


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

    pruned_query: str
    """The original query with every triple in this combination simply removed (no path
    substitution). Not a real fix — it silently drops a constraint rather than relaxing it, so its
    rows shouldn't be trusted as answers — but always present (its text needs no store access to
    build, so it's there even if `timeout` cuts off everything else for this combination), so it's
    a fallback when `relaxed_query` is `None` or still returns nothing."""

    pruned_row_count: int
    """Row count of `pruned_query` when re-executed. Guaranteed non-empty under normal operation —
    an empty result here would mean this combination was never a genuine culprit to begin with —
    *except* when `timeout` cuts off the verification itself before it can complete, in which case
    this falls back to `0` even though `pruned_query` is still known to return rows."""

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
    result_limit: Optional[int] = 50_000,
    allowed_namespaces: Optional[Sequence[str]] = DEFAULT_RELAX_NAMESPACES,
    timeout: Optional[float] = DEFAULT_RELAX_TIMEOUT,
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
    specific target to search for then, only whichever single side did resolve. Such a triple is
    skipped entirely rather than searched: without a fixed goal, path search would just be an
    undirected exploration with nothing to verify a suggestion against except re-running the whole
    query, which isn't worth the cost of searching for.

    A combination is only relaxed as a whole: if any one of its triples has no discoverable path,
    no combined relaxed query is built (the others being fixable wouldn't help, since they were
    only broken *together*). Filters flagged by diagnosis are included in `filter_results` as-is;
    no relaxation is attempted for them.

    Every result also carries `pruned_query`: the original query with every triple in the
    combination simply removed, no path substitution. It isn't a real fix (it silently drops a
    constraint instead of relaxing it), but it's always present and guaranteed non-empty — useful
    as a fallback when `relaxed_query` is `None` or still returns nothing, so you're never left with
    no query at all.

    `ablation_depth` is passed through to `diagnose` to control how many triples may be jointly
    removed while searching for a culprit (see `diagnose`'s `depth`; same default of 3).

    `max_depth` bounds the path search itself; defaults to 2 (`None`).

    `sample_limit` caps how many distinct bound pairs are considered per triple; defaults to 5 (a
    representative sample rather than every row). Pass `None` to consider every distinct pair
    instead of stopping early (only worth doing for small graphs/result sets, since it means
    examining every row the reduced query returns).

    `result_limit` caps how many rows a relaxed query's `LIMIT` allows; defaults to 50,000, since a
    relaxed path (especially an alternation of several distinct paths) can match far more broadly
    than the original triple did. Only ever tightens a `LIMIT` already present in the original
    query, never loosens it. Pass `None` to leave it unbounded.

    `allowed_namespaces` restricts path search to predicates whose IRI starts with one of these
    prefixes; a real edge outside every listed namespace is invisible to the search even if it
    would otherwise connect the two endpoints. Defaults to `DEFAULT_RELAX_NAMESPACES` (Brick,
    ASHRAE 223P, RDFS, QUDT) — the common case for this tool's building-automation use case, where
    a real but out-of-domain predicate is rarely a fix anyone actually wants suggested. Pass `None`
    explicitly for no restriction (any real predicate found in the graph is fair game), or your own
    sequence of namespace prefixes to restrict to something else.

    `timeout` (seconds) bounds the SPARQL query work needed to relax *each* culprit combination
    (resolving endpoints, verifying a candidate fix) — not diagnosis, and not path search itself,
    which never touches the query engine. A combination that can't finish within its budget falls
    back to `pruned_query` rather than hanging or failing the whole call. Defaults to
    `DEFAULT_RELAX_TIMEOUT` (5 seconds); pass `None` to leave it unbounded.
    """
    original_row_count, results, filter_results = _diagnose_and_relax(
        data,
        query,
        format,
        ablation_depth,
        max_depth,
        sample_limit,
        result_limit,
        allowed_namespaces=list(allowed_namespaces) if allowed_namespaces is not None else None,
        timeout=timeout,
    )
    return RelaxReport(
        original_row_count=original_row_count,
        results=[
            RelaxedCulprit(
                found_at_depth=found_at_depth,
                triples=[RelaxedTriple(triple=t, path_text=p) for t, p in triples],
                relaxed_query=relaxed_query,
                row_count=row_count,
                pruned_query=pruned_query,
                pruned_row_count=pruned_row_count,
            )
            for found_at_depth, triples, relaxed_query, row_count, pruned_query, pruned_row_count in results
        ],
        filter_results=[
            FilterReport(expression=e, row_count_without_filter=n) for e, n in filter_results
        ],
    )
