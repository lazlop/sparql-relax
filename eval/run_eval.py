#!/usr/bin/env python3
"""
run_eval.py

Evaluates sparql-relax-rs's `diagnose_and_relax` against the LLM-generated
SPARQL queries in Experiment_Results/*.csv, scored against each row's
ground-truth query.

For every (query_id, building) row below --threshold on a value-set F1
metric (recomputed here at runtime against the real graph — the same
"flatten every bound value across every row/column into a set, then
precision/recall/F1 against ground truth" metric the previous Python
implementation's eval used, for comparability), this:

  1. Runs `diagnose_and_relax` on the generated query.
  2. Scores every culprit combination's `relaxed_query` (if one was built)
     the same way.
  3. Records the best relaxed score found, alongside diagnosis details
     (how many culprits, how many combinations diagnosis needed >1 triple
     for, how many filters were flagged).

Unlike the old pure-Python ablation (which brute-forced dozens of
predicate-substitution variants per triple and needed a multiprocess
worker-recycling supervisor to stay within memory), sparql-relax-rs's
Rust core does one bounded search per query, so this script is a plain
sequential loop — no process pool, no memory-based worker recycling.

The one real cost worth knowing about: `diagnose_and_relax` takes the RDF
graph as raw text and reparses it inside Rust on every call (there's no
store-caching entry point yet). For the ~1-2MB building graphs referenced
by these CSVs that's still fast (well under the query-execution time,
in practice), but it means this doesn't scale to graphs of very different
size without that API gaining a cached-store path.

Usage:
  # Quick smoke run: 25 sampled rows per CSV (the default)
  python3 run_eval.py

  # Every row in every CSV (slow — see the note above)
  python3 run_eval.py --all

  # A couple of specific CSVs, a bigger sample, custom output path
  python3 run_eval.py --csv "Experiment_Results/DA-KGQA/o3-mini.csv" \\
      --sample-per-csv 100 --output my_eval.csv
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

import sparql_relax_rs

SCRIPT_DIR = Path(__file__).resolve().parent


# ==============================================================================
#  VALUE-SET F1 SCORING
# ==============================================================================


def _value_set(query_text: str, store: pyoxigraph.Store, limit: Optional[int]) -> tuple[Optional[set], Optional[str]]:
    """Runs a SELECT query and flattens every bound term across every
    row/column into a set of string values. Returns (values, error)."""
    q = query_text
    if limit and "LIMIT" not in query_text.upper():
        q = f"{query_text}\nLIMIT {limit}"
    try:
        solutions = store.query(q)
        values = set()
        for solution in solutions:
            for term in solution:
                if term is not None:
                    values.add(str(term))
        return values, None
    except Exception as exc:  # noqa: BLE001 - want to record any query failure
        return None, str(exc)


def value_set_f1(query_text: str, gt_values: set, store: pyoxigraph.Store, limit: Optional[int]) -> tuple[float, Optional[str]]:
    """Scores `query_text`'s value set against a precomputed ground-truth
    value set. A query that errors scores 0.0 (matching the prior Python
    eval's convention) rather than being excluded."""
    gen_values, error = _value_set(query_text, store, limit)
    if gen_values is None:
        return 0.0, error
    if not gt_values and not gen_values:
        return 1.0, None
    if not gt_values or not gen_values:
        return 0.0, None
    tp = len(gen_values & gt_values)
    precision = tp / len(gen_values)
    recall = tp / len(gt_values)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return f1, None


# ==============================================================================
#  BUILDING GRAPH CACHE
# ==============================================================================


class BuildingCache:
    """Loads each building's TTL once (as both raw text, for
    `diagnose_and_relax`, and a `pyoxigraph.Store`, for fast scoring
    queries) and reuses it for every row referencing that building."""

    def __init__(self, buildings_dir: Path):
        self.buildings_dir = buildings_dir
        self._stores: dict[str, Optional[pyoxigraph.Store]] = {}
        self._texts: dict[str, Optional[str]] = {}

    def _load(self, building: str) -> None:
        path = self.buildings_dir / f"{building}.ttl"
        if not path.exists():
            print(f"  warning: building graph not found: {path}", file=sys.stderr, flush=True)
            self._stores[building] = None
            self._texts[building] = None
            return
        t0 = time.monotonic()
        text = path.read_text()
        store = pyoxigraph.Store()
        store.load(text.encode("utf-8"), format=pyoxigraph.RdfFormat.TURTLE)
        print(f"  loaded {path.name} ({len(store)} triples) in {round(time.monotonic() - t0, 3)}s", flush=True)
        self._stores[building] = store
        self._texts[building] = text

    def store(self, building: str) -> Optional[pyoxigraph.Store]:
        if building not in self._stores:
            self._load(building)
        return self._stores[building]

    def text(self, building: str) -> Optional[str]:
        if building not in self._texts:
            self._load(building)
        return self._texts[building]


# ==============================================================================
#  CSV DISCOVERY / LOADING
# ==============================================================================


def find_csvs(results_dir: Path) -> list[str]:
    return sorted(glob.glob(str(results_dir / "**" / "*.csv"), recursive=True))


def load_rows(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "generated_sparql" not in df.columns or "ground_truth_sparql" not in df.columns:
        return pd.DataFrame()
    df = df.dropna(subset=["generated_sparql", "ground_truth_sparql", "building"]).copy()
    df = df[df["generated_sparql"].str.strip() != ""]
    df = df[df["ground_truth_sparql"].str.strip() != ""]
    return df


# ==============================================================================
#  OUTPUT
# ==============================================================================

OUTPUT_FIELDS = [
    "source_csv", "query_id", "question", "building", "approach", "model_name",
    "skipped", "score_error",
    "original_f1",
    "num_bgp_culprits", "num_filter_culprits",
    "best_found_at_depth", "best_relaxed_f1", "delta_f1",
    "best_relaxed_triples", "best_relaxed_path_texts", "best_relaxed_query",
    "diagnose_error", "elapsed_sec",
]


@dataclass
class EvalConfig:
    threshold: float
    limit: Optional[int]
    ablation_depth: int
    max_depth: Optional[int]
    sample_limit: Optional[int]
    verbose: bool


def process_row(row: dict, csv_path: str, cache: BuildingCache, cfg: EvalConfig) -> Optional[dict]:
    building = str(row.get("building", ""))
    store = cache.store(building)
    text = cache.text(building)
    if store is None or text is None:
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
    gt_values, gt_error = _value_set(gt_query, store, cfg.limit)
    if gt_values is None:
        gt_values = set()  # ground truth itself failed to run; treat as empty

    original_f1, score_error = value_set_f1(gen_query, gt_values, store, cfg.limit)
    t_score = round(time.monotonic() - t_score0, 3)
    if cfg.verbose:
        print(f"    scored original+GT in {t_score}s -> original_f1={original_f1:.3f}", flush=True)

    if original_f1 >= cfg.threshold:
        return {
            **base, "skipped": True, "score_error": score_error or gt_error or "",
            "original_f1": round(original_f1, 6),
            "num_bgp_culprits": "", "num_filter_culprits": "",
            "best_found_at_depth": "", "best_relaxed_f1": "", "delta_f1": "",
            "best_relaxed_triples": "", "best_relaxed_path_texts": "", "best_relaxed_query": "",
            "diagnose_error": "", "elapsed_sec": "",
        }

    if cfg.verbose:
        print(f"  [{base['query_id']}] {building}  original_f1={original_f1:.3f} -> relaxing...", flush=True)

    t0 = time.monotonic()
    diagnose_error = ""
    best_f1 = -1.0
    best = None
    num_bgp_culprits = 0
    num_filter_culprits = 0
    try:
        t_relax0 = time.monotonic()
        report = sparql_relax_rs.diagnose_and_relax(
            text, gen_query,
            ablation_depth=cfg.ablation_depth,
            max_depth=cfg.max_depth,
            sample_limit=cfg.sample_limit,
        )
        t_relax = round(time.monotonic() - t_relax0, 3)
        num_bgp_culprits = len(report.results)
        num_filter_culprits = len(report.filter_results)
        if cfg.verbose:
            print(
                f"    diagnose_and_relax took {t_relax}s -> "
                f"{num_bgp_culprits} triple culprit(s), {num_filter_culprits} filter culprit(s)",
                flush=True,
            )
        t_candidates0 = time.monotonic()
        for culprit in report.results:
            if not culprit.relaxed_query:
                continue
            f1, _ = value_set_f1(culprit.relaxed_query, gt_values, store, cfg.limit)
            if f1 > best_f1:
                best_f1 = f1
                best = culprit
        if cfg.verbose:
            print(f"    scored {num_bgp_culprits} candidate(s) in {round(time.monotonic() - t_candidates0, 3)}s", flush=True)
    except Exception as exc:  # noqa: BLE001 - keep going across the whole batch
        diagnose_error = str(exc)
        if cfg.verbose:
            print(f"    diagnose_and_relax errored after {round(time.monotonic() - t0, 3)}s: {exc}", flush=True)

    elapsed = round(time.monotonic() - t0, 3)
    if cfg.verbose:
        print(f"    row total: {elapsed}s", flush=True)

    if best is None:
        return {
            **base, "skipped": False, "score_error": score_error or gt_error or "",
            "original_f1": round(original_f1, 6),
            "num_bgp_culprits": num_bgp_culprits, "num_filter_culprits": num_filter_culprits,
            "best_found_at_depth": "", "best_relaxed_f1": "", "delta_f1": "",
            "best_relaxed_triples": "", "best_relaxed_path_texts": "", "best_relaxed_query": "",
            "diagnose_error": diagnose_error, "elapsed_sec": elapsed,
        }

    return {
        **base, "skipped": False, "score_error": score_error or gt_error or "",
        "original_f1": round(original_f1, 6),
        "num_bgp_culprits": num_bgp_culprits, "num_filter_culprits": num_filter_culprits,
        "best_found_at_depth": best.found_at_depth,
        "best_relaxed_f1": round(best_f1, 6),
        "delta_f1": round(best_f1 - original_f1, 6),
        "best_relaxed_triples": " | ".join(t.triple for t in best.triples),
        "best_relaxed_path_texts": " | ".join(t.path_text or "" for t in best.triples),
        "best_relaxed_query": best.relaxed_query,
        "diagnose_error": diagnose_error, "elapsed_sec": elapsed,
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
    parser.add_argument("--threshold", type=float, default=1.0, metavar="F", help="Only relax rows with original_f1 < F")
    parser.add_argument("--limit", type=int, default=20_000, metavar="N", help="LIMIT added to queries that lack one (0 = off)")
    parser.add_argument("--sample-per-csv", type=int, default=25, metavar="N", help="Random rows per CSV (default: 25)")
    parser.add_argument("--all", action="store_true", help="Process every row instead of sampling")
    parser.add_argument("--max-queries", type=int, default=None, metavar="N", help="Stop after N total rows (across all CSVs)")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed, for reproducible --sample-per-csv runs")
    parser.add_argument("--skip-buildings", nargs="+", default=[], metavar="NAME")
    parser.add_argument("--ablation-depth", type=int, default=3, metavar="N")
    parser.add_argument("--max-depth", type=int, default=None, metavar="N", help="Omit for the adaptive default (2 point-to-point / 1 anchor-only)")
    parser.add_argument("--sample-limit", type=int, default=5, metavar="N")
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
        threshold=args.threshold, limit=limit,
        ablation_depth=args.ablation_depth, max_depth=args.max_depth, sample_limit=args.sample_limit,
        verbose=args.verbose,
    )
    skip_buildings = set(args.skip_buildings)
    rng = random.Random(args.seed)

    cache = BuildingCache(buildings_dir)

    work_items: list[tuple[dict, str]] = []
    for csv_path in csv_paths:
        df = load_rows(csv_path)
        if df.empty:
            print(f"  {Path(csv_path).name}: no usable SPARQL rows – skipped", flush=True)
            continue
        df = df[~df["building"].astype(str).isin(skip_buildings)]
        rows = df.to_dict("records")
        if not args.all and len(rows) > args.sample_per_csv:
            rows = rng.sample(rows, args.sample_per_csv)
        print(f"  {csv_path}  ({len(rows)}/{len(df)} rows selected)", flush=True)
        work_items.extend((row, csv_path) for row in rows)

    if args.max_queries is not None:
        work_items = work_items[: args.max_queries]

    print(f"\n{'=' * 70}\n  Total rows to process: {len(work_items)}\n{'=' * 70}\n", flush=True)

    out_path = Path(args.output)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()

        n_processed = n_skipped = n_relaxed_found = n_improved = 0
        deltas: list[float] = []
        t_start = time.monotonic()

        for i, (row, csv_path) in enumerate(work_items, start=1):
            out_row = process_row(row, csv_path, cache, cfg)
            if out_row is None:
                continue
            writer.writerow(out_row)
            f.flush()
            n_processed += 1
            if out_row["skipped"]:
                n_skipped += 1
            elif out_row["best_relaxed_f1"] != "":
                n_relaxed_found += 1
                delta = out_row["delta_f1"]
                deltas.append(delta)
                if delta > 0:
                    n_improved += 1

            if i % 25 == 0 or i == len(work_items):
                print(f"  [{i}/{len(work_items)}] processed={n_processed} skipped={n_skipped} "
                      f"relaxed={n_relaxed_found} improved={n_improved}", flush=True)

    elapsed_total = round(time.monotonic() - t_start, 1)
    print(f"\n{'=' * 70}", flush=True)
    print(f"Done in {elapsed_total}s. Results written to: {out_path}", flush=True)
    print(f"  processed:        {n_processed}", flush=True)
    print(f"  already >= threshold (skipped): {n_skipped}", flush=True)
    print(f"  relaxation attempted:  {n_processed - n_skipped}", flush=True)
    print(f"  relaxation found a fix (any culprit with a path): {n_relaxed_found}", flush=True)
    print(f"  strictly improved F1:  {n_improved}", flush=True)
    if deltas:
        avg_delta = sum(deltas) / len(deltas)
        print(f"  avg delta_f1 among relaxed: {avg_delta:+.4f}", flush=True)
    print(f"{'=' * 70}", flush=True)


if __name__ == "__main__":
    main()
