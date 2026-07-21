#!/usr/bin/env python3
"""
run_eval.py

Evaluates sparql-relax-rs's `diagnose`/`diagnose_and_relax` against the
LLM-generated SPARQL queries in Experiment_Results/*.csv that *originally*
returned zero rows (per that CSV's own `gen_num_rows`/`returns_results`
columns, recorded when those queries were first generated) — the queries
diagnosis exists for — scored against each row's ground-truth query.

For every such row, this:

  1. Assigns it a stable `uuid` — a `uuid5` hash of
     (source_csv, query_id, building, generated_sparql), so the same row
     gets the same id on every run rather than a fresh random one; that's
     what lets `--resume` and any cross-run comparison join on `uuid` alone.
  2. Records `gt_rows` (the ground-truth query's row count) and `gen_rows`
     (the generated query's row count) — the latter is always 0, exactly
     because that's this script's own selection criterion (see `load_rows`),
     read from the CSV's own recorded `gen_num_rows`/`returns_results`
     rather than re-run here.
  3. Scores the (known-empty) generated query's own value set against
     ground truth — value-set F1, GT rows covered, excess result rows, the
     same "flatten every bound value across every row/column into a set,
     then precision/recall/F1 against ground truth" metric the previous
     Python implementation's eval used, for comparability — as the
     `diagnose_*` columns. Since the generated query is guaranteed to
     return zero rows, this needs no query execution at all: an empty
     result set trivially covers nothing and produces no excess, and its F1
     is 0.0 against any non-empty ground truth (1.0 in the one edge case
     where the ground truth is *also* empty, `calculate_f1`'s convention).
  4. Runs `diagnose_and_relax` on the generated query (`ablation_depth=3` by
     default), scores every culprit combination's `relaxed_query` (if one
     was built) the same way, and records the best-scoring one as the
     `relax_*` columns.

Every zero-result row is processed; there's no F1-threshold skip, since a
query with zero original rows can't already be a perfect match (barring an
empty ground truth too, which scores 1.0 either way and isn't worth
special-casing).

Unlike the old pure-Python ablation (which brute-forced dozens of
predicate-substitution variants per triple and needed a multiprocess
worker-recycling supervisor to stay within memory), sparql-relax-rs's
Rust core does one bounded search per query, so processing itself is a
plain sequential loop, one row at a time, with no query-level concurrency.

It does still run inside a single persistent worker subprocess, though —
see `RowWorker`/`_worker_loop` below — not for throughput, but because
`diagnose_and_relax`'s Rust-side `timeout` isn't always enough on its own:
Oxigraph's query engine can go a long time between checking its
cancellation token (a cartesian-join BGP or a `*`/`+` property path can
make it materialize a large intermediate result before yielding control
back at all), and a stuck evaluation like that permanently occupies a
`rayon` worker thread for the rest of the process — no Python-side
timeout can force a native thread to stop. The parent enforces a hard
wall-clock cap per row and kills-and-restarts the worker if it's ever
exceeded, which is the only thing that can actually reclaim a wedged
thread. `BuildingCache` still lives inside that one worker for its whole
lifetime, so this costs nothing extra in the common case; only a row that
actually trips the watchdog pays for a fresh worker (and everything it
needs to reload).

`sparql_relax.diagnose`/`diagnose_and_relax` each reparse and reindex
their RDF graph from scratch on every call if you pass them raw text — on
a ~1-2MB building graph, that alone costs roughly as much as the search
itself. `BuildingCache` below avoids paying that per row: it builds one
`sparql_relax.Store` per building (see that class's docstring) and
reuses it for every row referencing that building, so the graph is parsed
exactly once no matter how many rows this script processes against it.

Usage:
  # Quick smoke run: 25 sampled zero-result rows per CSV (the default)
  python3 run_eval.py

  # Every zero-result row in every CSV (slow — see the note above)
  python3 run_eval.py --all

  # A couple of specific CSVs, a bigger sample, custom output path
  python3 run_eval.py --csv "Experiment_Results/DA-KGQA/o3-mini.csv" \\
      --sample-per-csv 100 --output my_eval.csv

  # Diagnose-only: skip relaxation (path search + candidate scoring)
  # entirely, just report which triples/filters are flagged as culprits.
  # Much cheaper per row than the full pass.
  python3 run_eval.py --diagnose-only

  # Diagnose-only, but also try combinations diagnose() flagged as cartesian
  # risks and skipped, once the safe search alone found nothing -- trusting
  # the query engine to come back quickly rather than treating them as
  # untouchable. Riskier per row (see --try-cartesian-risks' own help text),
  # so consider a larger --hard-timeout too.
  python3 run_eval.py --diagnose-only --try-cartesian-risks --hard-timeout 120
"""

from __future__ import annotations

import argparse
import csv
import glob
import multiprocessing as mp
import random
import sys
import time
import uuid as uuid_module
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import pyoxigraph

import sparql_relax

SCRIPT_DIR = Path(__file__).resolve().parent


# ==============================================================================
#  VALUE-SET F1 SCORING
# ==============================================================================


def calculate_f1(gen_values: Optional[set], gt_values: Optional[set]) -> float:
    """Computes F1 score between two sets of values."""
    if gen_values is None or gt_values is None:
        return 0.0
    if not gt_values and not gen_values:
        return 1.0
    if not gt_values or not gen_values:
        return 0.0
    tp = len(gen_values & gt_values)
    precision = tp / len(gen_values)
    recall = tp / len(gt_values)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return f1


def _get_full_query_stats(query_text: str, store: pyoxigraph.Store, limit: Optional[int]) -> tuple[Optional[set], Optional[int], Optional[int], Optional[list[frozenset]], Optional[str]]:
    """Runs a SELECT query and returns (values, row_count, col_count, row_sets, error)."""
    q = query_text
    if limit and "LIMIT" not in query_text.upper():
        q = f"{query_text}\nLIMIT {limit}"
    try:
        solutions = store.query(q)
        values = set()
        row_sets = []
        row_count = 0
        col_count = 0
        for solution in solutions:
            row_count += 1
            if row_count == 1:
                col_count = len(solution)
            row_vals = frozenset(str(term) for term in solution if term is not None)
            row_sets.append(row_vals)
            for term in solution:
                if term is not None:
                    values.add(str(term))
        return values, row_count, col_count, row_sets, None
    except Exception as exc:
        return None, None, None, None, str(exc)


# (deleted _get_row_sets)


def _row_coverage_stats(res_sets: list[frozenset], gt_sets: list[frozenset]) -> tuple[int, int]:
    """Returns (gt_rows_covered, excess_result_rows)."""
    if not gt_sets:
        return 0, len(res_sets) if res_sets else 0
    if not res_sets:
        return 0, 0

    gt_covered = sum(1 for g in gt_sets if any(r <= g for r in res_sets))
    excess = sum(1 for r in res_sets if not any(r <= g for g in gt_sets))
    return gt_covered, excess


def value_set_f1(query_text: str, gt_values: set, store: pyoxigraph.Store, limit: Optional[int]) -> tuple[float, Optional[str]]:
    """Scores `query_text`'s value set against a precomputed ground-truth
    value set. A query that errors scores 0.0."""
    gen_values, _, _, _, error = _get_full_query_stats(query_text, store, limit)
    return calculate_f1(gen_values, gt_values), error


# ==============================================================================
#  BUILDING GRAPH CACHE
# ==============================================================================


class BuildingCache:
    """Loads each building's TTL once — as a `pyoxigraph.Store` for fast scoring queries, and a
    `sparql_relax.Store` for `diagnose`/`diagnose_and_relax` — and reuses both for every row
    referencing that building.

    Building the `sparql_relax.Store` here (once) rather than passing raw text to
    `diagnose`/`diagnose_and_relax` on every row (which would each reparse and reindex the whole
    graph from scratch) is the single biggest lever available for a batch like this one, worth far
    more than any of the search-side tuning knobs below: on `b59.ttl` (46k triples, ~1.5MB), a
    throwaway per-call parse costs roughly 100ms *before* any diagnosis work even starts, and this
    script runs thousands of rows against a handful of buildings.
    """

    def __init__(self, buildings_dir: Path):
        self.buildings_dir = buildings_dir
        self._stores: dict[str, Optional[pyoxigraph.Store]] = {}
        self._relax_stores: dict[str, Optional[sparql_relax.Store]] = {}

    def _ensure_loaded(self, building: str) -> None:
        if building not in self._stores:
            self._load(building)

    def _load(self, building: str) -> None:
        path = self.buildings_dir / f"{building}.ttl"
        if not path.exists():
            print(f"  warning: building graph not found: {path}", file=sys.stderr, flush=True)
            self._stores[building] = None
            self._relax_stores[building] = None
            return
        t0 = time.monotonic()
        text = path.read_text()
        store = pyoxigraph.Store()
        store.load(text.encode("utf-8"), format=pyoxigraph.RdfFormat.TURTLE)
        relax_store = sparql_relax.Store(text)
        print(f"  loaded {path.name} ({len(store)} triples) in {round(time.monotonic() - t0, 3)}s", flush=True)
        self._stores[building] = store
        self._relax_stores[building] = relax_store

    def store(self, building: str) -> Optional[pyoxigraph.Store]:
        self._ensure_loaded(building)
        return self._stores[building]

    def relax_store(self, building: str) -> Optional[sparql_relax.Store]:
        self._ensure_loaded(building)
        return self._relax_stores[building]


# ==============================================================================
#  CSV DISCOVERY / LOADING
# ==============================================================================


def find_csvs(results_dir: Path) -> list[str]:
    return sorted(glob.glob(str(results_dir / "**" / "*.csv"), recursive=True))


def load_rows(csv_path: str) -> pd.DataFrame:
    """Loads `csv_path` and filters to rows whose *original* generated query — as recorded in the
    Experiment_Results CSV itself, not recomputed here — returned zero rows. That's what
    `gen_num_rows`/`returns_results` capture: both are written once, when the query was first
    generated, so they reflect the original run rather than anything this script computes."""
    df = pd.read_csv(csv_path)
    if "generated_sparql" not in df.columns or "ground_truth_sparql" not in df.columns:
        return pd.DataFrame()
    df = df.dropna(subset=["generated_sparql", "ground_truth_sparql", "building"]).copy()
    df = df[df["generated_sparql"].str.strip() != ""]
    df = df[df["ground_truth_sparql"].str.strip() != ""]
    if "gen_num_rows" in df.columns:
        df = df[df["gen_num_rows"].fillna(0).astype(int) == 0]
    elif "returns_results" in df.columns:
        df = df[df["returns_results"].fillna(True) == False]  # noqa: E712
    else:
        print(f"  warning: {csv_path} has neither gen_num_rows nor returns_results — "
              f"cannot filter to originally-zero-result rows, skipping file", file=sys.stderr, flush=True)
        return pd.DataFrame()
    return df


# ==============================================================================
#  OUTPUT
# ==============================================================================

OUTPUT_FIELDS = [
    "uuid", "source_csv", "query_id", "question", "building", "approach", "model_name",
    "gt_rows", "gt_col_count", "gt_unique_value_count",
    "gen_rows",
    "diagnose_culprit_found", "diagnose_value_set_f1", "diagnose_rows_covered", "diagnose_excess_rows",
    "num_bgp_culprits", "num_filter_culprits", "num_cartesian_risks", "cartesian_risk_only",
    "cartesian_risk_attempted", "num_cartesian_risks_confirmed", "cartesian_risk_culprit_triples",
    "relax_attempted", "relax_stmt_index", "relax_stmt_type", "relax_stmt_text",
    "relax_removed_statements", "relax_sparql",
    "relax_value_set_f1", "relax_rows_covered", "relax_excess_rows",
    "relax_result_row_count", "relax_result_col_count",
    "timed_out", "error", "elapsed_sec",
]


def _uuid_for_row(csv_path: str, row: dict) -> str:
    """A stable per-row id: `uuid5` of (csv_path, query_id, building, generated_sparql), so the
    same logical row gets the same `uuid` on every run rather than a fresh random one each time —
    that's what lets `--resume` (and any cross-run comparison) join on `uuid` alone."""
    key = f"{csv_path}|{row.get('query_id', '')}|{row.get('building', '')}|{row.get('generated_sparql', '')}"
    return str(uuid_module.uuid5(uuid_module.NAMESPACE_URL, key))


def _int_or_zero(value) -> int:
    """Coerces a CSV-sourced numeric field to `int`, treating `None`/`NaN`/unparseable values as
    0 rather than propagating them — `gen_num_rows` can be missing or `NaN` for some rows even
    within a CSV that has the column at all."""
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def _blank_relax_fields() -> dict:
    """Default values for the relax-stage columns when nothing was found (or relaxation wasn't
    attempted at all — see `--diagnose-only`)."""
    return {
        "relax_stmt_index": -1, "relax_stmt_type": "baseline",
        "relax_stmt_text": "", "relax_removed_statements": "[]", "relax_sparql": "",
        "relax_value_set_f1": "", "relax_rows_covered": "", "relax_excess_rows": "",
        "relax_result_row_count": "", "relax_result_col_count": "",
    }


@dataclass
class EvalConfig:
    limit: Optional[int]
    ablation_depth: int
    max_depth: Optional[int]
    sample_limit: Optional[int]
    verbose: bool
    timeout: float
    diagnose_only: bool
    try_cartesian_risks: bool


def _base_fields(row: dict, csv_path: str) -> dict:
    return {
        "source_csv": csv_path,
        "query_id": str(row.get("query_id", "")),
        "question": str(row.get("question", "")),
        "building": str(row.get("building", "")),
        "approach": str(row.get("approach", Path(csv_path).parent.name)),
        "model_name": str(row.get("model_name", "")),
    }


def _is_timeout_message(message: str) -> bool:
    """Whether an exception's text names a timeout — covers both
    `RelaxError::Timeout`/`RelaxError::QueryTimeout` (see
    sparql-relax-core/src/error.rs) and this script's own watchdog messages."""
    lowered = message.lower()
    return "timeout" in lowered or "timed out" in lowered


def process_row(row: dict, csv_path: str, cache: BuildingCache, cfg: EvalConfig) -> Optional[dict]:
    building = str(row.get("building", ""))
    store = cache.store(building)
    relax_store = cache.relax_store(building)
    if store is None or relax_store is None:
        return None

    base = _base_fields(row, csv_path)
    gen_query = str(row["generated_sparql"])
    gt_query = str(row["ground_truth_sparql"])

    t0 = time.monotonic()
    gt_values, gt_rows, gt_cols, gt_sets, gt_error = _get_full_query_stats(gt_query, store, cfg.limit)
    if gt_values is None:
        gt_values, gt_rows, gt_cols, gt_sets = set(), 0, 0, []
    t_gt = round(time.monotonic() - t0, 3)

    # The generated query is guaranteed to return zero rows — that's this
    # script's own selection criterion (see `load_rows`) — so its value set
    # is trivially empty and there's no need to actually re-run it here:
    # `calculate_f1`/`_row_coverage_stats` already handle an empty result set
    # correctly (0 rows covered, 0 excess, F1 0.0 unless the ground truth is
    # *also* empty). `gen_rows` itself comes straight from the CSV's own
    # recorded `gen_num_rows`, not recomputed.
    gen_rows = _int_or_zero(row.get("gen_num_rows"))
    diagnose_f1 = calculate_f1(set(), gt_values)
    diagnose_cov, diagnose_exc = _row_coverage_stats([], gt_sets)

    if cfg.verbose:
        print(f"    scored GT in {t_gt}s -> diagnose_f1={diagnose_f1:.3f}, cov={diagnose_cov}, exc={diagnose_exc}", flush=True)

    common = {
        **base, "uuid": _uuid_for_row(csv_path, row),
        "gt_rows": gt_rows, "gt_col_count": gt_cols, "gt_unique_value_count": len(gt_values),
        "gen_rows": gen_rows,
        "diagnose_culprit_found": False,
        "diagnose_value_set_f1": round(diagnose_f1, 6),
        "diagnose_rows_covered": diagnose_cov, "diagnose_excess_rows": diagnose_exc,
        "relax_attempted": not cfg.diagnose_only,
    }

    if cfg.diagnose_only:
        return _diagnose_only_row(relax_store, gen_query, cfg, common, base["query_id"], building, t_gt, store, gt_values, gt_sets)

    if cfg.verbose:
        print(f"  [{base['query_id']}] {building}  diagnose_f1={diagnose_f1:.3f} -> relaxing...", flush=True)

    best_f1 = -1.0
    best = None
    best_idx = -1
    best_diagnose_f1 = -1.0
    best_diagnose_culprit = None
    relax_error = ""

    # Relaxation phase. diagnose_and_relax runs diagnosis internally (so
    # there's no separate diagnose() call here) and enforces its own
    # timeout/diagnose_timeout deadlines via a real Rust-side cancellation
    # token — see sparql-relax-core/src/diagnose.rs — so it's called
    # directly rather than wrapped in a Python-side watchdog thread/future
    # here specifically. That Rust-side deadline is usually sufficient but
    # not watertight (see the module docstring); the backstop for the rare
    # case it misses is the hard-timeout process watchdog this function
    # runs under (see `RowWorker`), not a per-call wrapper here.
    t_relax_start = time.monotonic()
    try:
        report = relax_store.diagnose_and_relax(
            gen_query,
            ablation_depth=cfg.ablation_depth,
            max_depth=cfg.max_depth,
            sample_limit=cfg.sample_limit,
            timeout=cfg.timeout,
            diagnose_timeout=cfg.timeout,
        )
        for idx, culprit in enumerate(report.results):
            # Diagnose's own signal: what removing this culprit combination
            # gets you with no path substitution at all. `pruned_query` is
            # always present whenever diagnose flags a combination as a
            # genuine culprit (see `RelaxedCulprit.pruned_query`'s docs —
            # "guaranteed non-empty under normal operation"), so this is
            # scored for every combination diagnose found, independent of
            # whether relax's path search separately confirms a real
            # substitute for it. This is what `diagnose_value_set_f1` below
            # actually measures diagnosis's own contribution against — not
            # the trivially-zero original query.
            if culprit.pruned_query:
                f1, _ = value_set_f1(culprit.pruned_query, gt_values, store, cfg.limit)
                if f1 > best_diagnose_f1:
                    best_diagnose_f1 = f1
                    best_diagnose_culprit = culprit
            # Relax's own signal: a real, graph-verified path substitution.
            if culprit.relaxed_query:
                f1, _ = value_set_f1(culprit.relaxed_query, gt_values, store, cfg.limit)
                if f1 > best_f1:
                    best_f1 = f1
                    best = culprit
                    best_idx = idx
    except Exception as exc:
        best_f1 = -1.0
        relax_error = str(exc)
    t_relax_elapsed = round(time.monotonic() - t_relax_start, 3)
    total_elapsed_sec = round(t_gt + t_relax_elapsed, 3)

    if best_diagnose_culprit is not None:
        diag_values, _, _, diag_sets, _ = _get_full_query_stats(best_diagnose_culprit.pruned_query, store, cfg.limit)
        if diag_values is None:
            diag_values, diag_sets = set(), []
        diag_cov, diag_exc = _row_coverage_stats(diag_sets, gt_sets)
        common["diagnose_culprit_found"] = True
        common["diagnose_value_set_f1"] = round(best_diagnose_f1, 6)
        common["diagnose_rows_covered"] = diag_cov
        common["diagnose_excess_rows"] = diag_exc

    if best is None:
        return {
            **common,
            **_blank_relax_fields(),
            "relax_value_set_f1": round(best_f1, 6) if best_f1 >= 0 else "",
            "timed_out": _is_timeout_message(relax_error), "error": relax_error,
            "elapsed_sec": total_elapsed_sec,
        }

    rel_values, rel_rows, rel_cols, rel_sets, _ = _get_full_query_stats(best.relaxed_query, store, cfg.limit)
    if rel_values is None:
        rel_values, rel_rows, rel_cols, rel_sets = set(), 0, 0, []
    rel_cov, rel_exc = _row_coverage_stats(rel_sets, gt_sets)

    return {
        **common,
        "relax_stmt_index": best_idx,
        "relax_stmt_type": "relaxed",
        "relax_stmt_text": " | ".join(t.triple for t in best.triples),
        "relax_removed_statements": str([t.triple for t in best.triples]),
        "relax_sparql": best.relaxed_query,
        "relax_value_set_f1": round(best_f1, 6),
        "relax_rows_covered": rel_cov,
        "relax_excess_rows": rel_exc,
        "relax_result_row_count": rel_rows,
        "relax_result_col_count": rel_cols,
        "timed_out": _is_timeout_message(relax_error), "error": relax_error,
        "elapsed_sec": total_elapsed_sec,
    }


def _diagnose_only_row(
    relax_store: sparql_relax.Store,
    gen_query: str,
    cfg: EvalConfig,
    common: dict,
    query_id: str,
    building: str,
    t_gt: float,
    store: pyoxigraph.Store,
    gt_values: set,
    gt_sets: list[frozenset],
) -> dict:
    """Runs just `Store.diagnose()` — no endpoint resolution, no path search — and
    reports which triples/filters were flagged as culprits, plus how often the
    cartesian-join guard (see `algebra::has_cartesian_join`) declined to check a
    combination at all rather than confirming or ruling it out
    (`Diagnosis.cartesian_risks`). Used by `--diagnose-only`, where the relax
    phase's endpoint resolution/path search isn't wanted at all.

    Still scores every confirmed culprit combination's *pruned* query (the
    triples simply removed, no path substitution — via `sparql_relax.pruned_query`)
    against ground truth, same "diagnose's own signal" metric the full
    (non-diagnose-only) path already reports as `diagnose_value_set_f1` — this
    is what makes it possible to tell whether a confirmed culprit (safe or, see
    below, cartesian-risk-recovered) is a real fix or a technicality.

    If `cfg.try_cartesian_risks` is set and the safe search above came up
    completely empty (no BGP or FILTER culprit at any depth), this also
    re-evaluates `diagnosis.cartesian_risks` via `Store.check_cartesian_risks`
    — actually running the combos `diagnose` skipped, on the theory that
    trusting the engine to come back quickly is a risk worth taking *once
    the safe search has nothing left to offer*, not routinely. This is opting
    out of `diagnose`'s own protection for exactly this shape (see that
    method's docs on the hang risk); it's caught the same way every other
    row-level hang is here — by `RowWorker`'s hard-timeout watchdog, not by
    anything in this function itself."""
    if cfg.verbose:
        print(f"  [{query_id}] {building}  -> diagnosing...", flush=True)

    t_diag_start = time.monotonic()
    diagnose_error = ""
    num_culprits = num_filter_culprits = num_cartesian_risks = 0
    num_cartesian_risks_confirmed = 0
    cartesian_risk_attempted = False
    cartesian_risk_culprit_triples = ""
    candidates: list = []
    try:
        diagnosis = relax_store.diagnose(gen_query, depth=cfg.ablation_depth, timeout=cfg.timeout)
        num_culprits = len(diagnosis.culprits)
        num_filter_culprits = len(diagnosis.filter_culprits)
        num_cartesian_risks = len(diagnosis.cartesian_risks)
        found_safely = num_culprits > 0 or num_filter_culprits > 0
        common["diagnose_culprit_found"] = found_safely
        candidates.extend(diagnosis.culprits)

        if cfg.try_cartesian_risks and not found_safely and diagnosis.cartesian_risks:
            cartesian_risk_attempted = True
            if cfg.verbose:
                print(f"    -> safe search empty, trying {num_cartesian_risks} cartesian-risk combo(s)...", flush=True)
            # Every row this function is ever called for is originally
            # zero-result (see `load_rows`'s filter), so `original_is_empty`
            # is always true here — `check_cartesian_risks` only needs the
            # cheap ASK-existence shortcut, never the full per-row check.
            confirmed = relax_store.check_cartesian_risks(
                gen_query, diagnosis.cartesian_risks, original_is_empty=True, timeout=cfg.timeout
            )
            num_cartesian_risks_confirmed = len(confirmed)
            if confirmed:
                common["diagnose_culprit_found"] = True
                cartesian_risk_culprit_triples = " | ".join(confirmed[0].triples)
                candidates.extend(confirmed)
    except Exception as exc:
        diagnose_error = str(exc)
    t_diag_elapsed = time.monotonic() - t_diag_start

    # Score every candidate's pruned query (no path substitution) and keep the
    # best value-set F1 found -- mirrors exactly what the full (non-diagnose-
    # only) path already does for its own `diagnose_value_set_f1` (see
    # `process_row`'s `best_diagnose_culprit` loop), just without the relax
    # phase's endpoint/path-search cost.
    best_f1, best_cov, best_exc = -1.0, 0, 0
    for culprit in candidates:
        try:
            pruned = sparql_relax.pruned_query(gen_query, culprit.triples)
        except Exception:
            continue
        values, _, _, sets, _ = _get_full_query_stats(pruned, store, cfg.limit)
        if values is None:
            continue
        f1 = calculate_f1(values, gt_values)
        if f1 > best_f1:
            best_f1 = f1
            best_cov, best_exc = _row_coverage_stats(sets, gt_sets)
    if best_f1 >= 0:
        common["diagnose_value_set_f1"] = round(best_f1, 6)
        common["diagnose_rows_covered"] = best_cov
        common["diagnose_excess_rows"] = best_exc

    return {
        **common,
        **_blank_relax_fields(),
        "num_bgp_culprits": num_culprits,
        "num_filter_culprits": num_filter_culprits,
        "num_cartesian_risks": num_cartesian_risks,
        # A risk was flagged and no culprit was confirmed by the safe search
        # at any depth — the guard is the plausible reason diagnosis came
        # back empty on this row, not proof (a combo it declined to check
        # could have been a culprit, or could just as easily not have been).
        # Unaffected by whether `try_cartesian_risks` then went on to
        # actually confirm one — see `cartesian_risk_attempted`/
        # `num_cartesian_risks_confirmed` for that outcome instead.
        "cartesian_risk_only": num_cartesian_risks > 0 and num_culprits == 0,
        "cartesian_risk_attempted": cartesian_risk_attempted,
        "num_cartesian_risks_confirmed": num_cartesian_risks_confirmed,
        "cartesian_risk_culprit_triples": cartesian_risk_culprit_triples,
        "timed_out": _is_timeout_message(diagnose_error), "error": diagnose_error,
        "elapsed_sec": round(t_gt + t_diag_elapsed, 3),
    }



# ==============================================================================
#  PROCESS-LEVEL WATCHDOG
# ==============================================================================
#
# diagnose_and_relax's `timeout`/`diagnose_timeout` are real, Rust-side
# deadlines (see sparql-relax-core/src/diagnose.rs) — but they only bound the
# work *if* Oxigraph's query engine actually checks its CancellationToken
# often enough to notice. In practice it doesn't always: a BGP with
# disconnected triple patterns (a cartesian join) or a `*`/`+` property path
# can make the engine materialize a large intermediate result — a hash-join
# build side, a transitive-closure frontier — without yielding control back
# in between, so the deadline passes unnoticed until that materialization
# finishes on its own. Measured live against Experiment_Results/DA-KGQA's
# llama.csv (bldg11, query MORTAR_002's `isAssociatedWith`/`hasQuantity`
# join): a `timeout=20.0` run sat on one row for over 200s before being
# killed by hand, never once returning control to Python.
#
# Because that stuck evaluation runs inside `rayon`'s shared global thread
# pool (every `diagnose_parsed`/`relax_combo` combination is dispatched via
# `.into_par_iter()`), one such row doesn't just run long — it permanently
# occupies a worker thread for the rest of the process's life, since nothing
# on the Python side can force a native thread to stop. Each subsequent row
# submits more combinations onto that same, increasingly saturated pool, so
# a single pathological query early in a run degrades every row after it,
# which is what actually produces the "run_eval hangs" symptom on a long
# batch even though any individual diagnose_and_relax call is nominally
# bounded.
#
# The raw `pyoxigraph.Store.query()` calls this script makes directly for
# value-set-F1 scoring (`_get_full_query_stats`, used for the original/GT
# queries and every relaxed candidate) have no timeout at all — pyoxigraph's
# Python bindings don't expose a cancellation mechanism — so they're exposed
# to the exact same failure mode with no protection whatsoever.
#
# A Python-side `future.result(timeout=...)` around any of this would not
# help (see the module docstring on `diagnose_and_relax` above): abandoning a
# thread-based call doesn't stop the underlying native work, which keeps
# running and keeps holding its rayon thread hostage. Only killing the whole
# OS process actually reclaims a wedged native thread. So each row here runs
# inside a persistent worker *subprocess* (keeping the BuildingCache
# performance win for the common case — reload only happens after an actual
# watchdog kill, not on every row); the parent enforces a hard wall-clock cap
# per row and, if it's ever exceeded, kills the worker outright and starts a
# fresh one for the rows that follow.


def _worker_loop(conn: "mp.connection.Connection", buildings_dir: str) -> None:
    """Runs in the persistent worker subprocess: owns one `BuildingCache` for
    its whole lifetime (reused across every row sent to it) and processes
    rows one at a time, blocking on `conn.recv()` between them. Exits when
    the parent closes its end of the pipe or sends the `None` shutdown
    sentinel."""
    cache = BuildingCache(Path(buildings_dir))
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return
        if msg is None:
            return
        row, csv_path, cfg = msg
        try:
            result = process_row(row, csv_path, cache, cfg)
            conn.send(("ok", result))
        except Exception as exc:
            conn.send(("error", str(exc)))


class RowWorker:
    """A persistent worker subprocess plus the hard-timeout watchdog around
    it. `process()` looks like a plain function call from the caller's side,
    but under the hood: send the row, wait up to `hard_timeout` seconds for a
    reply, and if that expires — or the worker dies outright, e.g. an OOM
    kill — SIGKILL whatever's left of it and transparently start a
    replacement before reporting the row as failed. The replacement pays a
    fresh `BuildingCache` (i.e. every building gets reloaded on first use
    again), but only that one time, not on every row."""

    def __init__(self, buildings_dir: Path, hard_timeout: float):
        self._buildings_dir = buildings_dir
        self._hard_timeout = hard_timeout
        self._ctx = mp.get_context("fork")
        self._conn: Optional["mp.connection.Connection"] = None
        self._proc: Optional[mp.process.BaseProcess] = None
        self._spawn()

    def _spawn(self) -> None:
        parent_conn, child_conn = self._ctx.Pipe()
        proc = self._ctx.Process(target=_worker_loop, args=(child_conn, str(self._buildings_dir)), daemon=True)
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

    def process(self, row: dict, csv_path: str, cfg: EvalConfig) -> tuple[Optional[dict], Optional[str]]:
        """Returns `(result, error)`; exactly one is `None`. `result` may
        itself be `None` on success (the building graph was missing —
        mirrors `process_row`'s own `None` return), which is why success/
        failure is signalled separately rather than by `result is None`."""
        assert self._conn is not None
        try:
            self._conn.send((row, csv_path, cfg))
        except (BrokenPipeError, OSError):
            self._kill_and_respawn()
            return None, "worker died before this row could be sent; killed and restarted"

        if not self._conn.poll(self._hard_timeout):
            self._kill_and_respawn()
            return None, f"row exceeded hard watchdog timeout ({self._hard_timeout:.0f}s); worker killed and restarted"

        try:
            status, payload = self._conn.recv()
        except (EOFError, OSError):
            self._kill_and_respawn()
            return None, "worker died while processing this row; killed and restarted"

        if status == "error":
            return None, payload
        return payload, None

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


# ==============================================================================
#  MAIN
# ==============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=str(SCRIPT_DIR / "Experiment_Results"), metavar="DIR")
    parser.add_argument("--csv", nargs="+", metavar="PATH", help="Explicit CSV file(s); overrides --results-dir")
    parser.add_argument(
        "--buildings-dir",
        default=str(SCRIPT_DIR / "buildings"),
        metavar="DIR",
    )
    parser.add_argument("--output", default=str(SCRIPT_DIR / "eval_results.csv"), metavar="FILE")
    parser.add_argument("--limit", type=int, default=20_000, metavar="N", help="LIMIT added to queries that lack one (0 = off)")
    parser.add_argument(
        "--sample-per-csv", type=int, default=25, metavar="N",
        help="Random originally-zero-result rows per CSV (default: 25)",
    )
    parser.add_argument("--all", action="store_true", help="Process every originally-zero-result row instead of sampling")
    parser.add_argument("--max-queries", type=int, default=None, metavar="N", help="Stop after N total rows (across all CSVs)")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed, for reproducible --sample-per-csv runs")
    parser.add_argument("--skip-buildings", nargs="+", default=[], metavar="NAME")
    parser.add_argument(
        "--ablation-depth", type=int, default=3, metavar="N",
        help="Max combination size diagnosis may remove jointly while searching for a culprit "
             "(default: 3)",
    )
    parser.add_argument("--max-depth", type=int, default=None, metavar="N", help="Omit for the adaptive default (2 point-to-point / 1 anchor-only)")
    parser.add_argument("--sample-limit", type=int, default=5, metavar="N")
    parser.add_argument(
        "--diagnose-only", action="store_true",
        help="Skip relaxation entirely: run diagnose() instead of diagnose_and_relax(), reporting "
             "which triples/filters are flagged as culprits with no endpoint resolution, path "
             "search, or candidate scoring. Cheaper per row and useful for iterating on ablation "
             "diagnosis on its own. --sample-limit and --max-depth (relaxation-only knobs) are "
             "ignored in this mode.",
    )
    parser.add_argument(
        "--try-cartesian-risks", action="store_true",
        help="Only meaningful with --diagnose-only. When diagnose()'s safe search comes up "
             "completely empty (no BGP/FILTER culprit at any depth) but it did flag some "
             "combinations as cartesian risks, actually run those combos anyway via "
             "Store.check_cartesian_risks — trusting the query engine to come back quickly rather "
             "than skipping them outright. This opts out of the protection diagnose() applies for "
             "disconnected BGPs (see its docstring): a bad combo can make Oxigraph materialize a "
             "full N x M cross product without yielding, for as long as 200+ seconds in cases "
             "measured against this dataset. The row-level --hard-timeout watchdog is what actually "
             "catches that if it happens (killing and restarting the worker, at the cost of losing "
             "that row's results) -- consider raising --hard-timeout when using this flag, since a "
             "row can now attempt many more, and riskier, queries than the default budget assumes.",
    )
    parser.add_argument(
        "--timeout", type=float, default=20.0, metavar="SECONDS",
        help="Per-row cap on diagnose_and_relax (or diagnose, with --diagnose-only); a handful of "
             "queries need a genuinely expensive reduced-query evaluation that no search-ordering "
             "fix avoids, so this bounds worst-case latency instead of letting one row stall the "
             "whole batch (default: 20s)",
    )
    parser.add_argument(
        "--hard-timeout", type=float, default=None, metavar="SECONDS",
        help="Wall-clock cap per row enforced by killing and restarting the worker subprocess if "
             "exceeded — a real backstop for the rare case where --timeout's Rust-side deadline "
             "doesn't get checked in time (a cartesian-join or transitive-path query can make "
             "Oxigraph's engine block on an expensive materialization without yielding), and for "
             "the value-set-F1 scoring queries this script runs directly via pyoxigraph, which "
             "have no timeout of their own at all. Defaults to max(60, 3 * --timeout + 30).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="If --output already exists, skip rows already present in it (matched by "
             "source_csv/query_id/building) and append rather than overwrite.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    buildings_dir = Path(args.buildings_dir)
    if not buildings_dir.is_dir():
        sys.exit(f"Buildings directory not found: {buildings_dir}")

    csv_paths = args.csv if args.csv else find_csvs(results_dir)
    if not csv_paths:
        sys.exit(f"No CSV files found under {results_dir}")

    if args.try_cartesian_risks and not args.diagnose_only:
        sys.exit("--try-cartesian-risks only has an effect together with --diagnose-only")

    limit = args.limit if args.limit > 0 else None
    cfg = EvalConfig(
        limit=limit,
        ablation_depth=args.ablation_depth, max_depth=args.max_depth, sample_limit=args.sample_limit,
        verbose=args.verbose, timeout=args.timeout, diagnose_only=args.diagnose_only,
        try_cartesian_risks=args.try_cartesian_risks,
    )
    skip_buildings = set(args.skip_buildings)
    rng = random.Random(args.seed)

    hard_timeout = args.hard_timeout if args.hard_timeout is not None else max(60.0, 3 * args.timeout + 30)

    work_items: list[tuple[dict, str]] = []
    for csv_path in csv_paths:
        df = load_rows(csv_path)
        if df.empty:
            print(f"  {Path(csv_path).name}: no originally-zero-result rows – skipped", flush=True)
            continue
        df = df[~df["building"].astype(str).isin(skip_buildings)]
        rows = df.to_dict("records")
        if not args.all and len(rows) > args.sample_per_csv:
            rows = rng.sample(rows, args.sample_per_csv)
        print(f"  {csv_path}  ({len(rows)}/{len(df)} originally-zero-result rows selected)", flush=True)
        work_items.extend((row, csv_path) for row in rows)

    if args.max_queries is not None:
        work_items = work_items[: args.max_queries]

    out_path = Path(args.output)
    resuming = args.resume and out_path.exists()
    if resuming:
        with out_path.open("r", newline="", encoding="utf-8") as f:
            done_uuids = {existing.get("uuid", "") for existing in csv.DictReader(f)}
        before = len(work_items)
        work_items = [(row, csv_path) for row, csv_path in work_items if _uuid_for_row(csv_path, row) not in done_uuids]
        print(
            f"  --resume: {len(done_uuids)} rows already in {out_path}, "
            f"{before - len(work_items)} skipped, {len(work_items)} remaining",
            flush=True,
        )

    print(f"\n{'=' * 70}\n  Total rows to process: {len(work_items)}\n{'=' * 70}\n", flush=True)

    total = len(work_items)
    worker = RowWorker(buildings_dir, hard_timeout)
    file_mode = "a" if resuming else "w"
    try:
        with out_path.open(file_mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
            if not resuming:
                writer.writeheader()

            n_processed = n_improved = n_with_culprits = n_watchdog_killed = 0
            n_diagnose_found = n_diagnose_improved = 0
            n_any_cartesian_risk = n_cartesian_risk_only = 0
            n_cartesian_risk_attempted = n_cartesian_risk_recovered = 0
            relax_f1s: list[float] = []
            diagnose_f1s: list[float] = []
            cartesian_risk_f1s: list[float] = []
            t_start = time.monotonic()

            for i, (row, csv_path) in enumerate(work_items, start=1):
                out_row, watchdog_error = worker.process(row, csv_path, cfg)
                if watchdog_error is not None:
                    n_watchdog_killed += 1
                    print(f"  [{i}/{total}] WATCHDOG: {csv_path} {row.get('query_id', '')} "
                          f"{row.get('building', '')}: {watchdog_error}", file=sys.stderr, flush=True)
                    out_row = {
                        **_base_fields(row, csv_path), "uuid": _uuid_for_row(csv_path, row),
                        **_blank_relax_fields(),
                        "gen_rows": _int_or_zero(row.get("gen_num_rows")),
                        "relax_attempted": not cfg.diagnose_only,
                        "timed_out": True, "error": watchdog_error,
                        "elapsed_sec": hard_timeout,
                    }
                if out_row is None:
                    continue
                writer.writerow(out_row)
                f.flush()
                n_processed += 1
                if cfg.diagnose_only:
                    # `relax_stmt_type` never becomes "relaxed" in this mode (no relax phase
                    # ever runs — see `_blank_relax_fields`), so `diagnose_culprit_found` is
                    # the real signal here, not the relax-path field below.
                    if out_row.get("diagnose_culprit_found"):
                        n_with_culprits += 1
                    f1 = out_row.get("diagnose_value_set_f1")
                    if isinstance(f1, (int, float)):
                        diagnose_f1s.append(f1)
                        if f1 > 0:
                            n_diagnose_improved += 1
                    if out_row.get("num_cartesian_risks"):
                        n_any_cartesian_risk += 1
                    if out_row.get("cartesian_risk_only"):
                        n_cartesian_risk_only += 1
                    if out_row.get("cartesian_risk_attempted"):
                        n_cartesian_risk_attempted += 1
                        if out_row.get("num_cartesian_risks_confirmed"):
                            n_cartesian_risk_recovered += 1
                            if isinstance(f1, (int, float)):
                                cartesian_risk_f1s.append(f1)
                else:
                    if out_row["relax_stmt_type"] == "relaxed":
                        n_with_culprits += 1
                    if out_row.get("diagnose_culprit_found") is True:
                        n_diagnose_found += 1
                        if out_row["diagnose_value_set_f1"] > 0:
                            n_diagnose_improved += 1
                    if out_row["relax_value_set_f1"] != "":
                        f1 = out_row["relax_value_set_f1"]
                        relax_f1s.append(f1)
                        if f1 > 0:
                            n_improved += 1

                if i % 25 == 0 or i == total:
                    if cfg.diagnose_only:
                        print(f"  [{i}/{total}] processed={n_processed} "
                              f"with_culprits={n_with_culprits} cartesian_risk={n_any_cartesian_risk} "
                              f"cartesian_blocked_all={n_cartesian_risk_only} "
                              f"risk_recovered={n_cartesian_risk_recovered}/{n_cartesian_risk_attempted} "
                              f"watchdog_killed={n_watchdog_killed}", flush=True)
                    else:
                        print(f"  [{i}/{total}] processed={n_processed} "
                              f"diagnose_found={n_diagnose_found} diagnose_improved={n_diagnose_improved} "
                              f"relaxed={n_with_culprits} improved={n_improved} watchdog_killed={n_watchdog_killed}", flush=True)
    finally:
        worker.shutdown()

    elapsed_total = round(time.monotonic() - t_start, 1)
    print(f"\n{'=' * 70}", flush=True)
    print(f"Done in {elapsed_total}s. Results written to: {out_path}", flush=True)
    print(f"  processed (originally-zero-result rows): {n_processed}", flush=True)
    print(f"  rows killed by the hard watchdog timeout ({hard_timeout:.0f}s): {n_watchdog_killed}", flush=True)
    if cfg.diagnose_only:
        print(f"  rows with a culprit flagged (triple or filter): {n_with_culprits}", flush=True)
        print(f"    of which diagnose_value_set_f1 > 0 (pruning it also recovers real GT overlap): {n_diagnose_improved}", flush=True)
        if diagnose_f1s:
            print(f"  avg diagnose_value_set_f1 across all processed rows: {sum(diagnose_f1s) / len(diagnose_f1s):.4f}", flush=True)
        print(f"  rows where the cartesian-join guard declined to check at least one combination: {n_any_cartesian_risk}", flush=True)
        print(f"    of which no culprit was confirmed by any other combo (guard is the plausible reason nothing was found): {n_cartesian_risk_only}", flush=True)
        if cfg.try_cartesian_risks:
            print(f"  of those, actually tried via --try-cartesian-risks: {n_cartesian_risk_attempted}", flush=True)
            print(f"    of which a risky combo was confirmed as a real culprit: {n_cartesian_risk_recovered}", flush=True)
            if cartesian_risk_f1s:
                n_risk_f1_positive = sum(1 for f1 in cartesian_risk_f1s if f1 > 0)
                avg_risk_f1 = sum(cartesian_risk_f1s) / len(cartesian_risk_f1s)
                print(f"    of which diagnose_value_set_f1 > 0 (not just unblocking, real GT overlap): {n_risk_f1_positive}", flush=True)
                print(f"    avg diagnose_value_set_f1 among cartesian-risk-recovered rows: {avg_risk_f1:.4f}", flush=True)
    else:
        print(f"  diagnose found a genuine culprit (pruning it unblocks the query): {n_diagnose_found}", flush=True)
        print(f"    of which diagnose_value_set_f1 > 0 (pruning it also recovers real GT overlap): {n_diagnose_improved}", flush=True)
        print(f"  relax additionally confirmed a graph-verified path substitution: {n_with_culprits}", flush=True)
        print(f"  relax_value_set_f1 > 0 (a real, non-trivial path-substituted fix): {n_improved}", flush=True)
        if relax_f1s:
            avg_relax_f1 = sum(relax_f1s) / len(relax_f1s)
            print(f"  avg relax_value_set_f1 among path-substituted fixes: {avg_relax_f1:.4f}", flush=True)
    print(f"{'=' * 70}", flush=True)


if __name__ == "__main__":
    main()
