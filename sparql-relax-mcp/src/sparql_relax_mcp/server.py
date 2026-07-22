"""MCP server exposing `sparql_relax`: SPARQL query execution and diagnosis over
in-memory RDF graphs, for AI agents.

Intended agent workflow: `load_dataset` once, then for every query, call `diagnose`
before trusting its result. Diagnosis is nearly free when the query already works (it
just confirms the row count) and, when the query returns nothing or looks wrong, it
explains *why* -- which triple or FILTER is broken -- instead of leaving the agent with
just an empty result to guess at. This is the reliable, repeatable core of the tool, and
the one most agents should reach for. `diagnose`'s `relax=True` option additionally
searches the graph's real edges for a corrected query, but that search is experimental
(slower, namespace-restricted, and not guaranteed to find or verify a real fix) -- most
agents are better served by the default diagnosis and fixing the query themselves from
its explanation. Only call `query`, which fetches the full result set, once `diagnose`
has confirmed the query returns rows.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from mcp.server.fastmcp import FastMCP
from sparql_relax import QueryResult, Store, Term

mcp = FastMCP(
    name="sparql-relax",
    instructions=(
        "Tools for running and debugging SPARQL queries against RDF graphs. Load a graph with "
        "load_dataset, then ALWAYS call diagnose on a query before trusting its result -- it's "
        "cheap even when the query already works, and when it doesn't it explains exactly which "
        "triple or FILTER is broken. This default diagnosis is the reliable part of this tool; "
        "diagnose's relax=True option additionally tries to search the graph for a corrected "
        "query, but that search is experimental and its suggestions should be verified, not "
        "trusted outright -- leave relax off unless you specifically want to try it. Only call "
        "query, which returns the full result set, once diagnose has confirmed rows come back."
    ),
)


@dataclass
class _Dataset:
    store: Store
    data: str
    format: str
    triple_count: int


_datasets: dict[str, _Dataset] = {}


def _require_dataset(name: str) -> Store:
    dataset = _datasets.get(name)
    if dataset is None:
        available = ", ".join(sorted(_datasets)) or "(none loaded)"
        raise ValueError(f"no dataset named {name!r} is loaded. Loaded datasets: {available}. Call load_dataset first.")
    return dataset.store


# ==============================================================================
#  WATCHDOG
# ==============================================================================
#
# `Store.diagnose`/`diagnose_and_relax` run a Rust-side ablation search that,
# for a disconnected BGP, can make Oxigraph's query engine materialize a full
# N x M cross product without ever checking its own cancellation token --
# see sparql-relax-core/src/diagnose.rs's module docs, and eval/run_eval.py's
# own watchdog (which this mirrors) for a measured case that took over 200
# seconds. Because that stuck evaluation runs on rayon's shared global thread
# pool, it doesn't just make one call slow -- it permanently occupies a
# worker thread for the rest of this process's life, since nothing on the
# Python side can force a native thread to stop, and every subsequent
# diagnose call submits more work onto that same, increasingly saturated
# pool.
#
# Unlike eval/run_eval.py -- a batch script where killing a disposable
# per-row worker costs nothing -- this server is a single long-lived process
# holding every dataset an agent has loaded for the whole session in
# `_datasets`. Losing that on every diagnose call the way run_eval.py's
# per-row workers do would be far more disruptive than losing one row's
# result. So diagnose/diagnose_and_relax calls are routed through one
# persistent forked worker instead, only replaced -- killed and re-forked --
# when `load_dataset` changes what's loaded, or when a call times out;
# datasets are expected to change rarely within a session (often just
# once), so this stays cheap in the common case.
#
# The worker deliberately does *not* inherit the parent's already-built
# `Store` objects via fork's copy-on-write, tempting as that is to avoid
# re-parsing: by the time any worker is first forked, this process has
# already run at least one query itself (load_dataset's own triple-count
# check), which means Oxigraph/rayon's global thread pool may already be
# initialized. Forking a process that already has native background threads
# is exactly the hazard CPython's own multiprocessing docs warn about: a
# lock (or thread-pool bookkeeping) held by a thread that doesn't exist in
# the child can leave the child deadlocked the first time it's touched,
# silently and non-deterministically -- os.fork() only duplicates the
# calling thread, not whatever else was live at that instant. So the worker
# instead re-parses each dataset's raw RDF text into its own, entirely
# fresh `Store` the first time it's asked for -- the only thing crossing
# the fork boundary is inert Python text (`_Dataset.data`/`.format`), never
# an already-touched native object. This is the same reason
# eval/run_eval.py's own `BuildingCache` parses fresh inside each forked
# `RowWorker` rather than sharing a pre-built one from its parent.
#
# This deliberately does *not* also wrap the plain `query` tool: `query`
# doesn't run the automatic ablation search that's the actual mechanism
# behind the hang -- a hand-crafted disconnected query passed to `query`
# directly is comparatively rare, and adding fork overhead to the tool
# that's supposed to be the cheap, ordinary path isn't worth guarding
# against it.

DIAGNOSE_HARD_TIMEOUT_SECONDS = 30.0
"""Wall-clock cap per diagnose/diagnose_and_relax call, enforced by killing and
replacing the worker process if exceeded. Well above DEFAULT_ABLATION_TIMEOUT's/
DEFAULT_RELAX_TIMEOUT's own 5-second (soft, Rust-side) budgets -- this is only
meant to catch the rare case where that Rust-side deadline itself isn't honored
(see the module docs above), not to second-guess an ordinary, successful search."""


def _diagnose_worker_loop(conn: "mp.connection.Connection") -> None:
    """Runs in the forked worker: services one `(dataset, method_name, args,
    kwargs)` request at a time, blocking on `conn.recv()` between them. Reads
    `_datasets` directly as this module's own global for each dataset's raw
    `data`/`format` -- a real fork (not spawn) makes that exactly the
    parent's state as of the moment this worker was forked -- but builds its
    own fresh `Store` per dataset name, cached locally for the rest of this
    worker's life, rather than reusing the parent's already-built one (see
    the module docs above on why). Exits when the parent closes its end of
    the pipe or sends the `None` shutdown sentinel."""
    local_stores: dict[str, Store] = {}
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return
        if msg is None:
            return
        dataset_name, method_name, args, kwargs = msg
        try:
            if dataset_name not in local_stores:
                dataset = _datasets.get(dataset_name)
                if dataset is None:
                    available = ", ".join(sorted(_datasets)) or "(none loaded)"
                    raise ValueError(f"no dataset named {dataset_name!r} is loaded. Loaded datasets: {available}. Call load_dataset first.")
                local_stores[dataset_name] = Store(dataset.data, format=dataset.format)
            store = local_stores[dataset_name]
            result = getattr(store, method_name)(*args, **kwargs)
            conn.send(("ok", result))
        except Exception as exc:
            conn.send(("error", str(exc)))


class DiagnoseWorker:
    """A persistent forked worker plus the hard-timeout watchdog around it, for
    `diagnose`/`diagnose_and_relax` specifically -- see the module docs above
    for why. `call()` looks like a plain function call from the caller's
    side, but under the hood: send the request, wait up to `hard_timeout`
    seconds for a reply, and if that expires -- or the worker dies outright --
    kill whatever's left of it and start a replacement (which re-forks from
    whatever `_datasets` looks like *now*, so nothing loaded after the dead
    worker was spawned is lost) before reporting the call as failed.

    `worker_loop`/`hard_timeout` are only ever overridden by tests (to inject
    a fast, deterministic stand-in for a real hang rather than waiting on
    one); real callers should just use the defaults.
    """

    def __init__(
        self, worker_loop: Callable[["mp.connection.Connection"], None] = _diagnose_worker_loop, hard_timeout: float = DIAGNOSE_HARD_TIMEOUT_SECONDS
    ) -> None:
        self._worker_loop = worker_loop
        self._hard_timeout = hard_timeout
        self._ctx = mp.get_context("fork")
        self._conn: Optional["mp.connection.Connection"] = None
        self._proc: Optional[mp.process.BaseProcess] = None
        self._spawn()

    def _spawn(self) -> None:
        parent_conn, child_conn = self._ctx.Pipe()
        proc = self._ctx.Process(target=self._worker_loop, args=(child_conn,), daemon=True)
        proc.start()
        child_conn.close()
        self._conn = parent_conn
        self._proc = proc

    def _kill_and_respawn(self) -> None:
        assert self._proc is not None and self._conn is not None
        try:
            self._proc.kill()
        except Exception:
            pass
        self._proc.join(timeout=5)
        self._conn.close()
        self._spawn()

    def invalidate(self) -> None:
        """Replaces the worker with a fresh fork, so it picks up whatever
        `_datasets` looks like right now. Call this whenever `load_dataset`
        loads or replaces a dataset -- otherwise the worker would keep
        serving diagnose/diagnose_and_relax calls against its stale,
        fork-time copy."""
        self._kill_and_respawn()

    def call(self, dataset: str, method_name: str, *args: Any, **kwargs: Any) -> Any:
        assert self._conn is not None
        try:
            self._conn.send((dataset, method_name, args, kwargs))
        except (BrokenPipeError, OSError):
            self._kill_and_respawn()
            raise RuntimeError("diagnose worker died before this call could be sent; it has been restarted")

        if not self._conn.poll(self._hard_timeout):
            self._kill_and_respawn()
            raise RuntimeError(
                f"{method_name} exceeded its {self._hard_timeout:.0f}s hard timeout (likely a disconnected-BGP "
                "combination the query engine got stuck materializing) and was killed; the worker has been restarted"
            )

        try:
            status, payload = self._conn.recv()
        except (EOFError, OSError):
            self._kill_and_respawn()
            raise RuntimeError("diagnose worker died while processing this call; it has been restarted")

        if status == "error":
            raise RuntimeError(payload)
        return payload

    def shutdown(self) -> None:
        if self._conn is None or self._proc is None:
            return
        try:
            self._conn.send(None)
        except Exception:
            pass
        self._proc.join(timeout=5)
        if self._proc.is_alive():
            self._proc.kill()
            self._proc.join(timeout=5)
        self._conn.close()


_diagnose_worker: Optional[DiagnoseWorker] = None


def _get_diagnose_worker() -> DiagnoseWorker:
    global _diagnose_worker
    if _diagnose_worker is None:
        _diagnose_worker = DiagnoseWorker()
    return _diagnose_worker


def _invalidate_diagnose_worker() -> None:
    """Called whenever `load_dataset` changes what's loaded. No-op if no worker
    has been created yet (it'll fork fresh from the current `_datasets` on its
    own first use, so there's nothing stale to replace)."""
    if _diagnose_worker is not None:
        _diagnose_worker.invalidate()


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
    _datasets[name] = _Dataset(store=store, data=data, format=format, triple_count=triple_count)
    _invalidate_diagnose_worker()
    return {"name": name, "format": format, "triple_count": triple_count}


@mcp.tool()
def list_datasets() -> list[dict[str, Any]]:
    """List every dataset currently loaded via `load_dataset`, with its format and triple count."""
    return [{"name": name, "format": ds.format, "triple_count": ds.triple_count} for name, ds in sorted(_datasets.items())]


@mcp.tool()
def diagnose(dataset: str, query: str, relax: bool = False, ignore_cartesian_risk: bool = False) -> dict[str, Any]:
    """Run a SPARQL SELECT query against `dataset` and diagnose it. ALWAYS call this before
    `query` -- even when you expect the query to succeed.

    On a working query this is nearly free: it just confirms the row count (`ok: true`). On a
    query that returns nothing, or fewer rows than expected, it explains *why* -- which BGP
    triple(s) or FILTER(s) are responsible. If `relax=True`, it also searches the graph's
    actual edges for a real connecting path, often finding a corrected query that actually
    returns rows (see `relaxed_query` on each culprit).

    Note: Relaxation is experimental. For AI agents, it is often more effective to use
    diagnose with `relax=False`, then allow the agent to correct the query itself based
    on the diagnosis.

    Only SELECT queries can be diagnosed (ASK/CONSTRUCT/DESCRIBE aren't supported here -- use
    `query` directly for those). Once this reports `ok: true`, call `query` to fetch the full
    result set: this tool's `row_count` fields are counts, not the actual rows.

    When `relax=True`, path search defaults to predicates in the Brick, ASHRAE 223P, RDFS,
    and QUDT namespaces (this tool's usual building-automation domain) -- a real fix outside
    those namespaces won't be found, though the diagnosis of *which* triple is broken
    is unaffected.

    Some triple combinations are skipped rather than checked at all, because checking them would
    force the query engine to materialize a full N x M cross product before yielding a single row
    -- see `cartesian_risks_skipped` in the result. These are the plausible reason no culprit was
    isolated, not proof one way or the other. Pass `ignore_cartesian_risk=True` to force those
    combinations to actually be checked instead; only do this if the query is small enough that a
    stuck evaluation is an acceptable risk, since nothing can force a stuck check to give up early.
    """
    _require_dataset(dataset)  # fail fast with a clear error before involving the watchdog worker at all
    worker = _get_diagnose_worker()
    if relax:
        report = worker.call(dataset, "diagnose_and_relax", query, ignore_cartesian_risk=ignore_cartesian_risk)
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
        cartesian_risks = report.cartesian_risks
    else:
        report = worker.call(dataset, "diagnose", query, ignore_cartesian_risk=ignore_cartesian_risk)
        culprits = [
            {
                "depth": c.depth,
                "triples": [{"triple": t, "discovered_path": None} for t in c.triples],
                "fixed": False,
                "relaxed_query": None,
                "row_count_with_fix": None,
                "fallback_query_with_broken_triples_removed": None,
                "fallback_row_count": None,
            }
            for c in report.culprits
        ]
        filter_issues = [
            {"expression": f.expression, "row_count_without_filter": f.row_count_without_filter} for f in report.filter_culprits
        ]
        cartesian_risks = report.cartesian_risks

    cartesian_risks_skipped = [{"triples": list(r.triples), "depth": r.depth} for r in cartesian_risks]

    ok = report.original_row_count > 0 and not culprits and not filter_issues
    if ok:
        message = f"Query returned {report.original_row_count} row(s) with no issues found. Call `query` to fetch the full results."
    elif culprits or filter_issues:
        if relax:
            message = (
                "Query is broken. See `culprits`/`filter_issues` for what's wrong, and `relaxed_query` "
                "on any culprit where a fix was found."
            )
        else:
            message = (
                "Query is broken. See `culprits`/`filter_issues` for what's wrong. Call again with "
                "`relax=true` to search for a corrected query."
            )
    elif cartesian_risks_skipped:
        message = (
            "Query returned 0 rows and no broken triple/filter could be isolated, but "
            f"{len(cartesian_risks_skipped)} combination(s) were skipped rather than checked (see "
            "`cartesian_risks_skipped`) -- the real culprit may be among them. Call again with "
            "`ignore_cartesian_risk=true` to force those combinations to actually be checked."
        )
    else:
        message = (
            "Query returned 0 rows and no single broken triple/filter could be isolated -- the "
            "issue may be structural (e.g. two jointly-broken triples beyond the search depth, or "
            "an unbound variable) rather than one clear culprit."
        )

    return {
        "ok": ok,
        "row_count": report.original_row_count,
        "culprits": culprits,
        "filter_issues": filter_issues,
        "cartesian_risks_skipped": cartesian_risks_skipped,
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
    try:
        mcp.run(transport="stdio")
    finally:
        # `daemon=True` already ensures the worker (if any) dies with this
        # process even without this, but shutting it down explicitly first
        # gives it a chance to exit cleanly rather than being SIGKILL'd.
        if _diagnose_worker is not None:
            _diagnose_worker.shutdown()


if __name__ == "__main__":
    main()
