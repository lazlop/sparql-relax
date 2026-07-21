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

The module-level `diagnose`/`diagnose_and_relax` functions each parse `data`
into a fresh in-memory graph on every call. For more than one query against
the same graph, build a `Store` once and call its methods instead — see its
docstring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ._sparql_relax import Store as _Store

__all__ = [
    "Culprit",
    "FilterCulprit",
    "CartesianRiskCombo",
    "Diagnosis",
    "RelaxedTriple",
    "RelaxedCulprit",
    "FilterReport",
    "RelaxReport",
    "Term",
    "QueryResult",
    "Store",
    "diagnose",
    "diagnose_and_relax",
    "query",
    "DEFAULT_RELAX_NAMESPACES",
    "DEFAULT_RELAX_TIMEOUT",
    "DEFAULT_ABLATION_TIMEOUT",
    "DEFAULT_QUERY_TIMEOUT",
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

DEFAULT_ABLATION_TIMEOUT: float = 5.0
"""Default `timeout` for `diagnose`, and default `diagnose_timeout` for `diagnose_and_relax`
(seconds): a single ablation check is normally well under this. Pass `None` to leave it unbounded
— but see `diagnose`'s docs for why an internally-enforced timeout matters even when the caller
has its own external one."""

DEFAULT_QUERY_TIMEOUT: float = 10.0
"""Default `timeout` for `query` (seconds): more generous than `DEFAULT_ABLATION_TIMEOUT`, since a
query run through this path is normally one `diagnose`/`diagnose_and_relax` has already confirmed
works, and the caller now wants full results for rather than a cheap yes/no check. Pass `None` to
leave it unbounded."""


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
class CartesianRiskCombo:
    """A combination of triples whose reduced pattern (with them removed) was never evaluated
    against the graph at all, because doing so would force a cartesian product — some of the
    remaining triples' variables never overlap, even transitively, with the rest. That's exactly
    the shape that can make a query engine materialize a full N×M cross product before yielding a
    single row, regardless of how tightly `timeout` is set.

    This is *not* a claim that the combination is or isn't a genuine `Culprit` — it's kept
    separate specifically so a combination this call declined to check doesn't look identical to
    one it actually checked and ruled out.
    """

    triples: List[str]
    """One SPARQL text form per triple in the combination — same shape as `Culprit.triples`."""

    depth: int
    """The combination size at which this was encountered (see `Culprit.depth`)."""


@dataclass
class Diagnosis:
    original_row_count: int
    culprits: List[Culprit]
    filter_culprits: List[FilterCulprit]
    cartesian_risks: List[CartesianRiskCombo]


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


@dataclass
class Term:
    """A single RDF term, split the same way as the SPARQL 1.1 Query Results JSON Format's `type`
    field, so it's a one-to-one translation for anything already speaking that shape."""

    kind: str
    """`"uri"`, `"bnode"`, or `"literal"`."""

    value: str
    """The IRI, blank node label, or literal lexical value."""

    datatype: Optional[str] = None
    """The literal's datatype IRI. Only set when `kind` is `"literal"`."""

    language: Optional[str] = None
    """The literal's language tag, if any. Only ever set when `kind` is `"literal"`."""


@dataclass
class QueryResult:
    """The result of `query`/`Store.query`, shaped by which SPARQL query form produced it.

    Exactly one of `boolean`, `(variables, rows)`, or `triples` is populated, matching `form`
    (`"boolean"`, `"solutions"`, or `"graph"` respectively) — the others are left at their default
    of `None`.
    """

    form: str
    """`"boolean"` (from `ASK`), `"solutions"` (from `SELECT`), or `"graph"` (from `CONSTRUCT`/
    `DESCRIBE`)."""

    boolean: Optional[bool] = None
    """The `ASK` result. Only set when `form == "boolean"`."""

    variables: Optional[List[str]] = None
    """The `SELECT` column order. Only set when `form == "solutions"`."""

    rows: Optional[List[List[Optional[Term]]]] = None
    """Each row aligned to `variables` position-for-position, with `None` wherever that variable
    was left unbound in that particular row. Only set when `form == "solutions"`; see `bindings`
    for a more convenient per-row shape."""

    triples: Optional[List[Tuple[Term, Term, Term]]] = None
    """The `CONSTRUCT`/`DESCRIBE` result graph, as `(subject, predicate, object)` tuples. Only set
    when `form == "graph"`."""

    @property
    def bindings(self) -> List[Dict[str, Term]]:
        """`rows` reshaped into one `{variable: Term}` dict per row, omitting any variable left
        unbound in that row rather than carrying a `None` placeholder for it. `[]` for any form
        other than `"solutions"`."""
        if self.form != "solutions" or self.variables is None or self.rows is None:
            return []
        return [{var: term for var, term in zip(self.variables, row) if term is not None} for row in self.rows]


def _term_from_tuple(t: Tuple[str, str, Optional[str], Optional[str]]) -> Term:
    kind, value, datatype, language = t
    return Term(kind=kind, value=value, datatype=datatype, language=language)


def _query_result_from_tuple(t) -> QueryResult:
    form, boolean, variables, rows, triples = t
    return QueryResult(
        form=form,
        boolean=boolean,
        variables=variables,
        rows=[[None if term is None else _term_from_tuple(term) for term in row] for row in rows]
        if rows is not None
        else None,
        triples=[(_term_from_tuple(s), _term_from_tuple(p), _term_from_tuple(o)) for s, p, o in triples]
        if triples is not None
        else None,
    )


def _diagnosis_from_tuples(original_row_count: int, culprits, filter_culprits, cartesian_risks) -> Diagnosis:
    return Diagnosis(
        original_row_count=original_row_count,
        culprits=[Culprit(triples=triples, depth=culprit_depth) for triples, culprit_depth in culprits],
        filter_culprits=[
            FilterCulprit(expression=e, row_count_without_filter=n) for e, n in filter_culprits
        ],
        cartesian_risks=[CartesianRiskCombo(triples=triples, depth=d) for triples, d in cartesian_risks],
    )


def _relax_report_from_tuples(original_row_count: int, results, filter_results) -> RelaxReport:
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


class Store:
    """An RDF graph loaded once and held for repeated `diagnose`/`diagnose_and_relax` calls
    against it.

    The module-level `diagnose`/`diagnose_and_relax` functions each parse `data` and build a fresh
    in-memory store from scratch on every call — fine for a one-off query, but wasteful for the
    common case of running many queries against the same graph (e.g. evaluating a batch of
    generated queries against one building's data), where that parse-and-index work is identical
    and pointless to repeat. Build a `Store` once and call its methods instead:

    ```python
    store = Store(building_ttl_text)
    for query in generated_queries:
        report = store.diagnose_and_relax(query)
    ```

    Each method carries the same parameters as its module-level counterpart of the same name,
    just without `data`/`format`, which are fixed for the lifetime of the `Store` (set once here,
    in `__init__`).
    """

    def __init__(self, data: str, format: str = "turtle") -> None:
        self._store = _Store(data, format)

    def diagnose(self, query: str, depth: int = 3, timeout: Optional[float] = DEFAULT_ABLATION_TIMEOUT) -> Diagnosis:
        """See the module-level `diagnose` for what this does and what `depth`/`timeout` control."""
        original_row_count, culprits, filter_culprits, cartesian_risks = self._store.diagnose(query, depth=depth, timeout=timeout)
        return _diagnosis_from_tuples(original_row_count, culprits, filter_culprits, cartesian_risks)

    def diagnose_and_relax(
        self,
        query: str,
        ablation_depth: int = 3,
        max_depth: Optional[int] = None,
        sample_limit: Optional[int] = 5,
        result_limit: Optional[int] = 50_000,
        allowed_namespaces: Optional[Sequence[str]] = DEFAULT_RELAX_NAMESPACES,
        timeout: Optional[float] = DEFAULT_RELAX_TIMEOUT,
        diagnose_timeout: Optional[float] = DEFAULT_ABLATION_TIMEOUT,
    ) -> RelaxReport:
        """See the module-level `diagnose_and_relax` for what this does and what its parameters
        control."""
        original_row_count, results, filter_results = self._store.diagnose_and_relax(
            query,
            ablation_depth=ablation_depth,
            max_depth=max_depth,
            sample_limit=sample_limit,
            result_limit=result_limit,
            allowed_namespaces=list(allowed_namespaces) if allowed_namespaces is not None else None,
            timeout=timeout,
            diagnose_timeout=diagnose_timeout,
        )
        return _relax_report_from_tuples(original_row_count, results, filter_results)

    def query(
        self, query: str, row_limit: Optional[int] = None, timeout: Optional[float] = DEFAULT_QUERY_TIMEOUT
    ) -> QueryResult:
        """See the module-level `query` for what this does and what `row_limit`/`timeout`
        control."""
        return _query_result_from_tuple(self._store.query(query, row_limit=row_limit, timeout=timeout))

    def check_cartesian_risks(
        self,
        query: str,
        risks: Sequence[CartesianRiskCombo],
        original_is_empty: bool,
        timeout: Optional[float] = DEFAULT_ABLATION_TIMEOUT,
    ) -> List[Culprit]:
        """Re-evaluates `risks` — combinations a prior `diagnose` call on this same `query`
        flagged as `CartesianRiskCombo`s and never actually checked — against this store,
        returning every one confirmed as a genuine `Culprit`.

        `risks` should be that prior call's `Diagnosis.cartesian_risks`, unmodified;
        `original_is_empty` should be that same call's `original_row_count == 0` — there's no
        cheaper way for this method to learn it than re-running the whole original query itself,
        so it isn't done here a second time; pass it through from what you already have.

        Calling this at all means opting out of the protection `diagnose` applies for exactly this
        shape: a disconnected BGP can make the query engine materialize a full N×M cross product
        before yielding a single row, and unlike `diagnose`'s own bounded checks, nothing here can
        force a stuck native evaluation to give up if the engine doesn't check its cancellation
        token often enough — a measured case elsewhere in this project sat for over 200 seconds and
        permanently occupied a shared worker thread until the whole process was killed (see
        `eval/run_eval.py`'s process-level watchdog for why that backstop lives at the process
        level, not inside this call). Use this only after `diagnose` has already come up empty, and
        only once you've independently judged the risk worth taking for this specific query/graph —
        ideally from a process you can afford to kill outright if a check gets stuck.

        `timeout` (seconds) bounds every risk combination checked here with one shared deadline,
        exactly like `diagnose`'s own `timeout` — not a fresh budget per combination. Defaults to
        `DEFAULT_ABLATION_TIMEOUT`; pass `None` to leave it unbounded (not recommended given the
        docs above).
        """
        culprits = self._store.check_cartesian_risks(
            query, [list(risk.triples) for risk in risks], original_is_empty, timeout=timeout
        )
        return [Culprit(triples=triples, depth=d) for triples, d in culprits]


def diagnose(
    data: str,
    query: str,
    format: str = "turtle",
    depth: int = 3,
    timeout: Optional[float] = DEFAULT_ABLATION_TIMEOUT,
) -> Diagnosis:
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

    `timeout` (seconds) bounds every SPARQL query this call runs — the original query itself, and
    every triple-combo/filter ablation check, all sharing one deadline rather than a fresh budget
    per check — so the *whole* call is bounded by roughly `timeout` regardless of how many
    candidates there are to work through. A check that can't finish in time is treated as "not a
    culprit"; if the *original* query can't even be evaluated within the budget, this raises rather
    than returning a result, since there's nothing meaningful to diagnose without it. Defaults to
    `DEFAULT_ABLATION_TIMEOUT` (5 seconds); pass `None` to leave it unbounded.

    Note that abandoning this call from the caller's side (e.g. a `concurrent.futures.Future`'s
    `result(timeout=...)`) does *not* stop it from running to completion in the background —
    Python threads can't be force-cancelled. A real, internally-enforced `timeout` here is what
    actually bounds the work; without it, a batch of many rows each abandoning slow calls will
    accumulate orphaned background searches that keep consuming CPU and threads indefinitely,
    starving every subsequent row rather than just the one that was slow.

    Builds a throwaway `Store` from `data` on every call — for more than one query against the
    same graph, build a `Store` once instead and call its `diagnose` method.
    """
    return Store(data, format).diagnose(query, depth=depth, timeout=timeout)


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
    diagnose_timeout: Optional[float] = DEFAULT_ABLATION_TIMEOUT,
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

    `timeout` (seconds) bounds all the work needed to relax *each* culprit combination — resolving
    endpoints, the path search itself, and verifying a candidate fix — not diagnosis, which has its
    own separate budget (see `diagnose_timeout`). A combination that can't finish within its budget
    falls back to `pruned_query` rather than hanging or failing the whole call. Defaults to
    `DEFAULT_RELAX_TIMEOUT` (5 seconds); pass `None` to leave it unbounded.

    `diagnose_timeout` (seconds) is passed straight through to `diagnose`'s own `timeout` — see its
    docs for what it bounds and why an internally-enforced timeout matters even when the caller has
    its own external one. Independent of `timeout` above: diagnosis runs once, before any
    relaxation work starts, so the two budgets don't interact. Also defaults to
    `DEFAULT_ABLATION_TIMEOUT` (5 seconds).

    Builds a throwaway `Store` from `data` on every call — for more than one query against the
    same graph, build a `Store` once instead and call its `diagnose_and_relax` method.
    """
    return Store(data, format).diagnose_and_relax(
        query,
        ablation_depth=ablation_depth,
        max_depth=max_depth,
        sample_limit=sample_limit,
        result_limit=result_limit,
        allowed_namespaces=allowed_namespaces,
        timeout=timeout,
        diagnose_timeout=diagnose_timeout,
    )


def query(
    data: str,
    query: str,
    format: str = "turtle",
    row_limit: Optional[int] = None,
    timeout: Optional[float] = DEFAULT_QUERY_TIMEOUT,
) -> QueryResult:
    """Runs `query` (any SPARQL query form — SELECT, ASK, CONSTRUCT, DESCRIBE) against the RDF
    graph in `data` and returns its actual results.

    Unlike `diagnose`/`diagnose_and_relax`, which exist to explain and fix a query that returns
    nothing (or wrongly), this is the ordinary case of running one that already works. The
    intended workflow is: call `diagnose` (or `diagnose_and_relax`) first — it's cheap even when
    the query succeeds, and if it doesn't, it tells you exactly which triple or filter is at fault
    instead of just an empty result — then call `query` once you're getting rows back, to fetch
    the full result set rather than diagnosis's row counts and samples.

    `row_limit` caps how many rows a SELECT/CONSTRUCT/DESCRIBE result may return, applied as a
    LIMIT on the query itself before evaluation — only ever tightening a LIMIT already present in
    the query, never loosening it — so an oversized result is bounded during evaluation rather
    than computed in full and then truncated. Has no effect on ASK, which only ever returns one
    boolean. Defaults to `None` (unbounded).

    `timeout` (seconds) bounds evaluation; a query that doesn't finish in time raises rather than
    continuing to run unobserved in the background — see `diagnose`'s docs for why an
    internally-enforced timeout matters even when the caller has its own external one. Defaults to
    `DEFAULT_QUERY_TIMEOUT` (10 seconds); pass `None` to leave it unbounded.

    Returns a `QueryResult` tagged by `form`: `"boolean"` (ASK), `"solutions"` (SELECT, via
    `variables`/`rows`/`bindings`), or `"graph"` (CONSTRUCT/DESCRIBE, via `triples`).

    Builds a throwaway `Store` from `data` on every call — for more than one query against the
    same graph, build a `Store` once instead and call its `query` method.
    """
    return Store(data, format).query(query, row_limit=row_limit, timeout=timeout)
