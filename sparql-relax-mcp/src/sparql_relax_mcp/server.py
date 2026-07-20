"""MCP server exposing `sparql_relax`: SPARQL query execution and diagnosis over
in-memory RDF graphs, for AI agents.

Intended agent workflow: `load_dataset` once, then for every query, call `diagnose`
before trusting its result. Diagnosis is nearly free when the query already works (it
just confirms the row count) and, when the query returns nothing or looks wrong, it
explains *why* -- which triple or FILTER is broken, and often a corrected query found
by searching the graph's real edges -- instead of leaving the agent with just an empty
result to guess at. Only call `query`, which fetches the full result set, once
`diagnose` has confirmed the query returns rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP
from sparql_relax import QueryResult, Store, Term

mcp = FastMCP(
    name="sparql-relax",
    instructions=(
        "Tools for running and debugging SPARQL queries against RDF graphs. Load a graph with "
        "load_dataset, then ALWAYS call diagnose on a query before trusting its result -- it's "
        "cheap even when the query already works, and when it doesn't it explains exactly which "
        "triple or FILTER is broken and, where possible, suggests a corrected query. Only call "
        "query, which returns the full result set, once diagnose has confirmed rows come back."
    ),
)


@dataclass
class _Dataset:
    store: Store
    format: str
    triple_count: int


_datasets: dict[str, _Dataset] = {}


def _require_dataset(name: str) -> Store:
    dataset = _datasets.get(name)
    if dataset is None:
        available = ", ".join(sorted(_datasets)) or "(none loaded)"
        raise ValueError(f"no dataset named {name!r} is loaded. Loaded datasets: {available}. Call load_dataset first.")
    return dataset.store


def _term_to_json(term: Optional[Term]) -> Optional[dict[str, Any]]:
    if term is None:
        return None
    out: dict[str, Any] = {"type": term.kind, "value": term.value}
    if term.datatype is not None:
        out["datatype"] = term.datatype
    if term.language is not None:
        out["lang"] = term.language
    return out


@mcp.tool()
def load_dataset(name: str, data: Optional[str] = None, path: Optional[str] = None, format: str = "turtle") -> dict[str, Any]:
    """Load RDF data into memory as a named dataset for `diagnose`/`query` to run against.

    Pass exactly one of `data` (the RDF text itself) or `path` (an absolute path to a local RDF
    file to read) -- not both. `format` is one of "turtle" (default), "ntriples", "nquads",
    "rdfxml", or "trig".

    Loading a dataset under a `name` that's already loaded replaces it.
    """
    if (data is None) == (path is None):
        raise ValueError("pass exactly one of `data` or `path`, not both")
    if path is not None:
        data = Path(path).read_text()
    assert data is not None
    store = Store(data, format=format)
    count_result = store.query("SELECT (COUNT(*) AS ?c) WHERE { ?s ?p ?o }")
    triple_count = int(count_result.rows[0][0].value)  # type: ignore[union-attr,index]
    _datasets[name] = _Dataset(store=store, format=format, triple_count=triple_count)
    return {"name": name, "format": format, "triple_count": triple_count}


@mcp.tool()
def list_datasets() -> list[dict[str, Any]]:
    """List every dataset currently loaded via `load_dataset`, with its format and triple count."""
    return [{"name": name, "format": ds.format, "triple_count": ds.triple_count} for name, ds in sorted(_datasets.items())]


@mcp.tool()
def diagnose(dataset: str, query: str, relax: bool = True) -> dict[str, Any]:
    """Run a SPARQL SELECT query against `dataset` and diagnose it. ALWAYS call this before
    `query` -- even when you expect the query to succeed.

    On a working query this is nearly free: it just confirms the row count (`ok: true`). On a
    query that returns nothing, or fewer rows than expected, it explains *why* -- which BGP
    triple(s) or FILTER(s) are responsible -- and, by searching the graph's actual edges for a
    real connecting path, often a corrected query that actually returns rows (see `relaxed_query`
    on each culprit). That is strictly more useful than the bare empty result a plain query run
    would give you, at little to no extra cost when the query already works.

    `relax` (default true) does that path search for a fix. It's the expensive part of diagnosis
    -- a bounded breadth-first search over the graph's real edges per culprit, plus re-resolving
    variable bindings -- so pass `relax=false` for a cheaper check that only reports *which*
    triple(s)/filter(s) are broken, without attempting to fix them (no `relaxed_query`,
    `discovered_path`, or fallback fields on each culprit).

    Only SELECT queries can be diagnosed (ASK/CONSTRUCT/DESCRIBE aren't supported here -- use
    `query` directly for those). Once this reports `ok: true`, call `query` to fetch the full
    result set: this tool's `row_count` fields are counts, not the actual rows.

    Relaxation's path search defaults to predicates in the Brick, ASHRAE 223P, RDFS, and QUDT
    namespaces (this tool's usual building-automation domain) -- a real fix outside those
    namespaces won't be found, though the diagnosis of *which* triple is broken is unaffected.
    """
    store = _require_dataset(dataset)

    if relax:
        report = store.diagnose_and_relax(query)
        original_row_count = report.original_row_count
        culprits = [
            {
                "depth": result.found_at_depth,
                "triples": [{"triple": t.triple, "discovered_path": t.path_text} for t in result.triples],
                "fixed": result.fixed,
                "relaxed_query": result.relaxed_query,
                "row_count_with_fix": result.row_count,
                "fallback_query_with_broken_triples_removed": result.pruned_query,
                "fallback_row_count": result.pruned_row_count,
            }
            for result in report.results
        ]
        filter_issues = [
            {"expression": f.expression, "row_count_without_filter": f.row_count_without_filter} for f in report.filter_results
        ]
    else:
        diagnosis = store.diagnose(query)
        original_row_count = diagnosis.original_row_count
        culprits = [{"depth": c.depth, "triples": [{"triple": t} for t in c.triples]} for c in diagnosis.culprits]
        filter_issues = [
            {"expression": f.expression, "row_count_without_filter": f.row_count_without_filter} for f in diagnosis.filter_culprits
        ]

    ok = original_row_count > 0 and not culprits and not filter_issues
    if ok:
        message = f"Query returned {original_row_count} row(s) with no issues found. Call `query` to fetch the full results."
    elif culprits or filter_issues:
        if relax:
            message = (
                "Query is broken. See `culprits`/`filter_issues` for what's wrong, and `relaxed_query` "
                "on any culprit where a fix was found."
            )
        else:
            message = (
                "Query is broken. See `culprits`/`filter_issues` for what's wrong. Call again with "
                "`relax=true` (the default) to search for a corrected query."
            )
    else:
        message = (
            "Query returned 0 rows and no single broken triple/filter could be isolated -- the "
            "issue may be structural (e.g. two jointly-broken triples beyond the search depth, or "
            "an unbound variable) rather than one clear culprit."
        )

    return {
        "ok": ok,
        "row_count": original_row_count,
        "culprits": culprits,
        "filter_issues": filter_issues,
        "message": message,
    }


@mcp.tool()
def query(dataset: str, query: str, row_limit: Optional[int] = 1000) -> dict[str, Any]:
    """Run any SPARQL query (SELECT/ASK/CONSTRUCT/DESCRIBE) against `dataset` and return its
    actual results.

    Call `diagnose` first on any new query -- it's cheap even when the query works, and it catches
    broken queries with an actionable explanation instead of a bare empty result. Reach for this
    tool only once `diagnose` has confirmed the query returns rows (or for ASK/CONSTRUCT/DESCRIBE
    queries, which `diagnose` doesn't support).

    `row_limit` caps how many rows a SELECT/CONSTRUCT/DESCRIBE result may return (default 1000,
    to keep results a reasonable size to return to you); has no effect on ASK. Pass a higher value
    if you know you need more rows, or `null` for no limit.
    """
    store = _require_dataset(dataset)
    result: QueryResult = store.query(query, row_limit=row_limit)

    if result.form == "boolean":
        return {"form": "boolean", "result": result.boolean}
    if result.form == "solutions":
        return {
            "form": "solutions",
            "variables": result.variables,
            "rows": [{var: _term_to_json(term) for var, term in row.items()} for row in result.bindings],
        }
    return {
        "form": "graph",
        "triples": [
            {"subject": _term_to_json(s), "predicate": _term_to_json(p), "object": _term_to_json(o)}
            for s, p, o in (result.triples or [])
        ],
    }


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
