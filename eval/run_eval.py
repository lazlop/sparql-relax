#!/usr/bin/env python3
"""
run_eval.py

Evaluates sparql-relax-rs's `diagnose`/`diagnose_and_relax` against the
LLM-generated SPARQL queries in Experiment_Results/*.csv that *originally*
returned zero rows (per that CSV's own `gen_num_rows`/`returns_results`
columns, recorded when those queries were first generated) — the queries
diagnosis exists for — scored against each row's ground-truth query.

For every such row, this:

  1. Runs `diagnose_and_relax` on the generated query (`ablation_depth=3`
     by default), which runs diagnosis internally before attempting to
     relax any culprit it finds.
  2. Scores the original query's own value set against ground truth
     (value-set F1, GT rows covered, excess result rows — the same
     "flatten every bound value across every row/column into a set, then
     precision/recall/F1 against ground truth" metric the previous Python
     implementation's eval used, for comparability) — the diagnose-stage
     metrics.
  3. Scores every culprit combination's `relaxed_query` (if one was built)
     the same way, and records the best-scoring one — the relax-stage
     metrics, alongside diagnosis details (how many culprits, how many
     combinations diagnosis needed >1 triple for, how many filters were
     flagged).

Every zero-result row is processed; there's no F1-threshold skip, since a
query with zero original rows can't already be a perfect match (barring an
empty ground truth too, which scores 1.0 either way and isn't worth
special-casing).

Unlike the old pure-Python ablation (which brute-forced dozens of
predicate-substitution variants per triple and needed a multiprocess
worker-recycling supervisor to stay within memory), sparql-relax-rs's
Rust core does one bounded search per query, so this script is a plain
sequential loop — no process pool, no memory-based worker recycling.

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
"""

from __future__ import annotations

import argparse
import csv
import glob
import random
import sys
import time
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
    "source_csv", "query_id", "question", "building", "approach", "model_name",
    "skipped", "original_value_set_f1", "best_value_set_f1", "best_stmt_index",
    "best_stmt_type", "best_stmt_text", "removed_statements", "original_sparql",
    "best_sparql", "delta_value_set_f1", "result_row_count", "result_col_count",
    "result_unique_value_count", "gt_row_count", "gt_col_count", "gt_unique_value_count",
    "syntax_ok", "timed_out", "error", "elapsed_sec", "relax_attempted",
    "relax_value_set_f1", "relax_result_row_count", "relax_result_col_count",
    "relax_sparql", "relax_delta_value_set_f1", "relax_elapsed_sec",
    "gt_rows_covered", "excess_result_rows", "relax_gt_rows_covered", "relax_excess_result_rows"
]


def _blank_relax_fields() -> dict:
    """Default values for relaxation columns when no relaxation is performed or found."""
    return {
        "best_value_set_f1": "", "best_stmt_index": -1, "best_stmt_type": "baseline",
        "best_stmt_text": "", "removed_statements": "[]", "best_sparql": "",
        "delta_value_set_f1": "", "relax_attempted": False, "relax_value_set_f1": "",
        "relax_result_row_count": "", "relax_result_col_count": "", "relax_sparql": "",
        "relax_delta_value_set_f1": "", "relax_elapsed_sec": "",
        "relax_gt_rows_covered": "", "relax_excess_result_rows": "",
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


def process_row(row: dict, csv_path: str, cache: BuildingCache, cfg: EvalConfig) -> Optional[dict]:
    building = str(row.get("building", ""))
    store = cache.store(building)
    relax_store = cache.relax_store(building)
    if store is None or relax_store is None:
        return None

    base = {
        "source_csv": csv_path,
        "query_id": str(row.get("query_id", "")),
        "question": str(row.get("question", "")),
        "building": building,
        "approach": str(row.get("approach", Path(csv_path).parent.name)),
        "model_name": str(row.get("model_name", "")),
    }

    gen_query = str(row["generated_sparql"])
    gt_query = str(row["ground_truth_sparql"])

    t_score0 = time.monotonic()
    gt_values, gt_rows, gt_cols, gt_sets, gt_error = _get_full_query_stats(gt_query, store, cfg.limit)
    if gt_values is None:
        gt_values, gt_rows, gt_cols, gt_sets = set(), 0, 0, []

    gen_values, gen_rows, gen_cols, gen_sets, gen_error = _get_full_query_stats(gen_query, store, cfg.limit)
    if gen_values is None:
        gen_values, gen_rows, gen_cols, gen_sets = set(), 0, 0, []

    original_f1 = calculate_f1(gen_values, gt_values)
    orig_cov, orig_exc = _row_coverage_stats(gen_sets, gt_sets)
    t_score = round(time.monotonic() - t_score0, 3)

    if cfg.verbose:
        print(f"    scored original+GT in {t_score}s -> original_f1={original_f1:.3f}, cov={orig_cov}, exc={orig_exc}", flush=True)

    common = {
        **base, "skipped": False,
        "original_value_set_f1": round(original_f1, 6),
        "result_row_count": gen_rows, "result_col_count": gen_cols,
        "result_unique_value_count": len(gen_values),
        "gt_row_count": gt_rows, "gt_col_count": gt_cols,
        "gt_unique_value_count": len(gt_values),
        "syntax_ok": gen_error == "", "timed_out": "timed out" in (gen_error or "").lower(),
        "error": gen_error or "", "elapsed_sec": t_score,
        "gt_rows_covered": orig_cov, "excess_result_rows": orig_exc,
        "original_sparql": gen_query,
        "relax_attempted": True,
    }

    if cfg.diagnose_only:
        return _diagnose_only_row(relax_store, gen_query, cfg, common, base["query_id"], building)

    if cfg.verbose:
        print(f"  [{base['query_id']}] {building}  original_f1={original_f1:.3f} -> relaxing...", flush=True)

    best_f1 = -1.0
    best = None
    best_idx = -1

    # Relaxation phase. diagnose_and_relax runs diagnosis internally (so
    # there's no separate diagnose() call here) and enforces its own
    # timeout/diagnose_timeout deadlines via a real Rust-side cancellation
    # token — see sparql-relax-core/src/diagnose.rs — so it's called
    # directly rather than wrapped in a Python-side watchdog thread/future
    # that would only add overhead without bounding anything further.
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
            if not culprit.relaxed_query:
                continue
            f1, _ = value_set_f1(culprit.relaxed_query, gt_values, store, cfg.limit)
            if f1 > best_f1:
                best_f1 = f1
                best = culprit
                best_idx = idx
    except Exception:
        best_f1 = -1.0
    t_relax_elapsed = round(time.monotonic() - t_relax_start, 3)

    # Total time for baseline scoring + diagnosis + relaxation (diagnose_and_relax
    # does diagnosis internally, so t_relax_elapsed already covers both).
    total_elapsed_sec = round(t_score + t_relax_elapsed, 3)

    if best is None or best_f1 <= original_f1:
        return {
            **common,
            "best_value_set_f1": round(original_f1, 6),
            "best_stmt_index": -1, "best_stmt_type": "baseline",
            "best_stmt_text": "", "removed_statements": "[]",
            "best_sparql": gen_query,
            "delta_value_set_f1": 0.0,
            "relax_value_set_f1": round(best_f1, 6) if best_f1 >= 0 else "",
            "relax_result_row_count": "", "relax_result_col_count": "",
            "relax_sparql": "", "relax_delta_value_set_f1": "",
            "relax_elapsed_sec": t_relax_elapsed,
            "relax_gt_rows_covered": "", "relax_excess_result_rows": "",
            "elapsed_sec": total_elapsed_sec,
        }

    rel_values, rel_rows, rel_cols, rel_sets, _ = _get_full_query_stats(best.relaxed_query, store, cfg.limit)
    if rel_values is None:
        rel_values, rel_rows, rel_cols, rel_sets = set(), 0, 0, []
    rel_cov, rel_exc = _row_coverage_stats(rel_sets, gt_sets)

    return {
        **common,
        "best_value_set_f1": round(best_f1, 6),
        "best_stmt_index": best_idx,
        "best_stmt_type": "relaxed",
        "best_stmt_text": " | ".join(t.triple for t in best.triples),
        "removed_statements": str([t.triple for t in best.triples]),
        "best_sparql": best.relaxed_query,
        "delta_value_set_f1": round(best_f1 - original_f1, 6),
        "relax_value_set_f1": round(best_f1, 6),
        "relax_result_row_count": rel_rows,
        "relax_result_col_count": rel_cols,
        "relax_sparql": best.relaxed_query,
        "relax_delta_value_set_f1": round(best_f1 - original_f1, 6),
        "relax_elapsed_sec": t_relax_elapsed,
        "relax_gt_rows_covered": rel_cov,
        "relax_excess_result_rows": rel_exc,
        "elapsed_sec": total_elapsed_sec,
    }


def _diagnose_only_row(
    relax_store: sparql_relax.Store, gen_query: str, cfg: EvalConfig, common: dict, query_id: str, building: str
) -> dict:
    """Runs just `Store.diagnose()` — no endpoint resolution, no path search, no
    candidate scoring — and reports which triples/filters were flagged as
    culprits. Used by `--diagnose-only`, where the relax phase's cost isn't
    wanted at all."""
    if cfg.verbose:
        print(f"  [{query_id}] {building}  -> diagnosing...", flush=True)

    t_diag_start = time.monotonic()
    diagnose_error = ""
    try:
        relax_store.diagnose(gen_query, depth=cfg.ablation_depth, timeout=cfg.timeout)
    except Exception as exc:
        diagnose_error = str(exc)
    t_diag_elapsed = time.monotonic() - t_diag_start

    return {
        **common,
        "best_value_set_f1": common["original_value_set_f1"],
        "best_stmt_index": -1, "best_stmt_type": "baseline",
        "best_stmt_text": "", "removed_statements": "[]",
        "best_sparql": common["original_sparql"],
        "delta_value_set_f1": 0.0,
        **_blank_relax_fields(),
        "relax_attempted": True,
        "error": diagnose_error or common["error"],
        "elapsed_sec": round(common["elapsed_sec"] + t_diag_elapsed, 3),
    }



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
        "--timeout", type=float, default=20.0, metavar="SECONDS",
        help="Per-row cap on diagnose_and_relax (or diagnose, with --diagnose-only); a handful of "
             "queries need a genuinely expensive reduced-query evaluation that no search-ordering "
             "fix avoids, so this bounds worst-case latency instead of letting one row stall the "
             "whole batch (default: 20s)",
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

    limit = args.limit if args.limit > 0 else None
    cfg = EvalConfig(
        limit=limit,
        ablation_depth=args.ablation_depth, max_depth=args.max_depth, sample_limit=args.sample_limit,
        verbose=args.verbose, timeout=args.timeout, diagnose_only=args.diagnose_only,
    )
    skip_buildings = set(args.skip_buildings)
    rng = random.Random(args.seed)

    cache = BuildingCache(buildings_dir)

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
            done_keys = {
                (existing.get("source_csv", ""), existing.get("query_id", ""), existing.get("building", ""))
                for existing in csv.DictReader(f)
            }
        before = len(work_items)
        work_items = [
            (row, csv_path) for row, csv_path in work_items
            if (csv_path, str(row.get("query_id", "")), str(row.get("building", ""))) not in done_keys
        ]
        print(
            f"  --resume: {len(done_keys)} rows already in {out_path}, "
            f"{before - len(work_items)} skipped, {len(work_items)} remaining",
            flush=True,
        )

    print(f"\n{'=' * 70}\n  Total rows to process: {len(work_items)}\n{'=' * 70}\n", flush=True)

    total = len(work_items)
    file_mode = "a" if resuming else "w"
    with out_path.open(file_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        if not resuming:
            writer.writeheader()

        n_processed = n_relaxed_found = n_improved = n_with_culprits = 0
        deltas: list[float] = []
        t_start = time.monotonic()

        for i, (row, csv_path) in enumerate(work_items, start=1):
            out_row = process_row(row, csv_path, cache, cfg)
            if out_row is None:
                continue
            writer.writerow(out_row)
            f.flush()
            n_processed += 1
            if out_row["best_stmt_type"] == "relaxed":
                n_with_culprits += 1
            if out_row["relax_value_set_f1"] != "":
                n_relaxed_found += 1
                delta = out_row["delta_value_set_f1"]
                deltas.append(delta)
                if delta > 0:
                    n_improved += 1

            if i % 25 == 0 or i == total:
                if cfg.diagnose_only:
                    print(f"  [{i}/{total}] processed={n_processed} "
                          f"with_culprits={n_with_culprits}", flush=True)
                else:
                    print(f"  [{i}/{total}] processed={n_processed} "
                          f"relaxed={n_relaxed_found} improved={n_improved}", flush=True)

    elapsed_total = round(time.monotonic() - t_start, 1)
    print(f"\n{'=' * 70}", flush=True)
    print(f"Done in {elapsed_total}s. Results written to: {out_path}", flush=True)
    print(f"  processed (originally-zero-result rows): {n_processed}", flush=True)
    if cfg.diagnose_only:
        print(f"  rows with a culprit flagged (triple or filter): {n_with_culprits}", flush=True)
    else:
        print(f"  relaxation found a fix (any culprit with a path): {n_relaxed_found}", flush=True)
        print(f"  strictly improved F1:  {n_improved}", flush=True)
        if deltas:
            avg_delta = sum(deltas) / len(deltas)
            print(f"  avg delta_f1 among relaxed: {avg_delta:+.4f}", flush=True)
    print(f"{'=' * 70}", flush=True)


if __name__ == "__main__":
    main()
