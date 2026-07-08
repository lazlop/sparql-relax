#!/usr/bin/env python3
"""
batch_ablation.py

For every experiment-results CSV, find rows where row_matching_f1 < threshold
and run the SPARQL ablation (remove one statement at a time) for each such query.
Each building's graph is loaded once and cached for reuse.

Usage:
  # All CSVs under Experiment_Results/ that have row_matching_f1:
  python3.10 batch_ablation.py

  # Specific CSV files:
  python3.10 batch_ablation.py \\
      --csv "Experiment_Results/ReAct(100)/o3-mini.csv" \\
           "Experiment_Results/DA-KGQA/o3-mini.csv"

  # Tune the timeout and threshold:
  python3.10 batch_ablation.py --timeout 120 --threshold 0.5

  # Use the ground truth queries instead of generated ones:
  python3.10 batch_ablation.py --use-gt

Options:
  --results-dir DIR    Root directory to scan for CSVs  (default: Experiment_Results)
  --csv PATH [...]     Explicit CSV file(s) to process  (overrides --results-dir)
  --buildings-dir DIR  Directory containing .ttl files  (default: eval_buildings)
  --output FILE        Output CSV path                  (default: ablation_batch_results.csv)
  --timeout SEC        Per-query ablation timeout        (default: 300)
  --threshold FLOAT    Only process rows with row_f1 <  (default: 1.0)
  --max-queries N      Stop after N queries (for testing)
  --use-gt             Ablate ground-truth queries instead of generated ones
  --verbose            Print each variant query
  --limit N             Add LIMIT N to every query that lacks one (default: 40000)
  --workers N           Number of queries to process in parallel (default: 1)
 --variant-workers N   Threads for ablation variants within a single query (default:
  --prune-mode MODE     Use sparql_prune instead of sparql_ablation.
                        Choices: single | greedy | beam
  --score-key METRIC    Metric to optimise in greedy/beam mode (default: value_set_f1)
  --beam-top-k K        Beam width once results appear in beam mode (default: 5)
"""

import argparse
import csv
import gc
import glob
import json
import multiprocessing as mp
import os
import queue
import resource
import sys
import threading
import time
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Dict, List, Optional

import pyoxigraph
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from sparql_ablation import run_ablation, run_query, run_query_value_set, _row_coverage_stats
from sparql_prune import run_pruning
from sparql_relax import run_relaxation


# ==============================================================================
#  NATIVE HEAP TRIMMER
# ==============================================================================
# pyoxigraph uses system libc (ptmalloc).  After each ablation, hundreds of
# Rust query-solution objects are freed back to ptmalloc's per-thread arena,
# but ptmalloc does not return that memory to the OS on its own — causing RSS
# to grow by GBs over time even though there is no true leak.
# malloc_trim(0) tells ptmalloc to release all free arena memory back to the
# OS.  It must be called from the SAME THREAD that made the allocations.
try:
    import ctypes as _ctypes
    _libc = _ctypes.CDLL("libc.so.6")
    def _trim_native_heap() -> None:
        gc.collect()           # free Python-level wrappers first
        _libc.malloc_trim(0)   # then flush ptmalloc free lists to OS
except (OSError, AttributeError):
    def _trim_native_heap() -> None:
        gc.collect()


def _get_rss_mb() -> float:
    """Return current process RSS in MB."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


# ==============================================================================
#  GRAPH CACHE
# ==============================================================================

class GraphCache:
    """
    Per-building cache of pyoxigraph.Store objects. Thread-safe.

    pyoxigraph.Store supports concurrent reads from multiple threads, so a
    single store per building is shared by all workers — no copies or
    checkout/return cycle needed.

    Different buildings are loaded concurrently (per-building locks with
    double-checked locking so each TTL is parsed at most once).
    """

    def __init__(self, buildings_dir: str):
        self.buildings_dir   = Path(buildings_dir)
        self._stores:         Dict[str, Optional[pyoxigraph.Store]] = {}
        self._global_lock    = threading.Lock()
        self._building_locks: Dict[str, threading.Lock] = {}

    def _ensure_store(self, building: str) -> Optional[pyoxigraph.Store]:
        with self._global_lock:
            if building in self._stores:
                return self._stores[building]
            if building not in self._building_locks:
                self._building_locks[building] = threading.Lock()
            bld_lock = self._building_locks[building]

        with bld_lock:
            with self._global_lock:
                if building in self._stores:
                    return self._stores[building]

            ttl_path = self.buildings_dir / f"{building}.ttl"
            if not ttl_path.exists():
                print(f"  ⚠  TTL not found: {ttl_path}", file=sys.stderr)
                with self._global_lock:
                    self._stores[building] = None
                return None

            print(f"  Loading {ttl_path.name} …", end=" ", flush=True)
            g = pyoxigraph.Store()
            with open(ttl_path, "rb") as _f:
                g.load(_f, format=pyoxigraph.RdfFormat.TURTLE)
            print(f"({len(g)} triples)")
            with self._global_lock:
                self._stores[building] = g
            return g

    def acquire(self, building: str) -> Optional[pyoxigraph.Store]:
        """Return the shared Store for *building* (never blocks)."""
        return self._ensure_store(building)

    def release(self, _building: str, _graph: pyoxigraph.Store) -> None:
        """No-op — Store is shared, not checked out."""



# ==============================================================================
#  GT RESULTS CACHE
# ==============================================================================

class GTCache:
    """
    Bounded LRU cache of ground-truth query results keyed by (building, gt_query).
    Thread-safe.

    Many rows share the same query_id and therefore the same ground-truth
    query (e.g. all model outputs for MORTAR_001). Without this cache each
    row re-executes the same — potentially expensive — GT query.

    max_size caps how many result sets are kept in memory at once. Oldest
    entries are evicted first. Each entry can hold up to `limit` rows, so
    peak memory is bounded by max_size × limit × avg_row_size.
    """

    def __init__(self, max_size: int = 500):
        self._cache: OrderedDict = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()

    def get(self, building: str, gt_query: str, graph, limit) -> list:
        key = (building, gt_query)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

        # Execute outside the lock so different buildings run concurrently.
        rows, _ = run_query(gt_query, graph, limit)
        if rows is None:
            rows = []

        with self._lock:
            # Another thread may have populated while we were running — keep first.
            if key not in self._cache:
                self._cache[key] = rows
                self._cache.move_to_end(key)
                if len(self._cache) > self._max_size:
                    self._cache.popitem(last=False)
            return self._cache[key]


# ==============================================================================
#  CSV DISCOVERY
# ==============================================================================

def find_csvs(results_dir: str) -> List[str]:
    pattern = str(Path(results_dir) / "**" / "*.csv")
    return sorted(glob.glob(pattern, recursive=True))


def load_rows(csv_path: str) -> pd.DataFrame:
    """
    Load a results CSV and return rows that have valid SPARQL in both columns.
    value_set_f1 is computed at runtime against the graph, so no CSV-level
    metric filter is applied here.
    """
    df = pd.read_csv(csv_path)
    if "generated_sparql" not in df.columns or "ground_truth_sparql" not in df.columns:
        return pd.DataFrame()
    df = df.dropna(subset=["generated_sparql", "ground_truth_sparql"]).copy()
    df = df[df["generated_sparql"].str.strip() != ""]
    df = df[df["ground_truth_sparql"].str.strip() != ""]
    return df


# ==============================================================================
#  OUTPUT WRITER
# ==============================================================================

OUTPUT_FIELDS = [
    "source_csv", "query_id", "question", "building", "approach", "model_name",
    "skipped",
    "original_value_set_f1",
    "best_value_set_f1",
    "best_stmt_index", "best_stmt_type", "best_stmt_text",
    "removed_statements",   # JSON list of each removed statement (multi-step for greedy/beam)
    "original_sparql",      # the query that was ablated (gen or GT depending on --use-gt)
    "best_sparql",
    "delta_value_set_f1",
    "result_row_count", "result_col_count", "result_unique_value_count",
    "gt_row_count", "gt_col_count", "gt_unique_value_count",
    "syntax_ok", "timed_out", "error", "elapsed_sec",
    "relax_attempted", "relax_value_set_f1", "relax_result_row_count", "relax_result_col_count",
    "relax_sparql", "relax_delta_value_set_f1", "relax_elapsed_sec",
    "gt_rows_covered", "excess_result_rows",
    "relax_gt_rows_covered", "relax_excess_result_rows",
]


class ResultWriter:
    def __init__(self, output_path: str, resume: bool = False):
        self.output_path = output_path
        mode = "a" if resume else "w"
        self._file = open(output_path, mode, newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=OUTPUT_FIELDS,
                                      extrasaction="ignore")
        if not resume:
            self._writer.writeheader()
        self._lock = threading.Lock()

    def write(self, row: dict):
        with self._lock:
            self._writer.writerow(row)
            self._file.flush()  # write incrementally so partial runs are not lost

    def close(self):
        self._file.close()


# ==============================================================================
#  WORKER PROCESS
# ==============================================================================

def _make_base_row(row: dict, csv_path: str, seq_num: int) -> dict:
    """Common identity fields shared by every output row."""
    return {
        "source_csv": csv_path,
        "query_id":   str(row.get("query_id", f"row_{seq_num}")),
        "question":   str(row.get("question", "")),
        "building":   str(row.get("building", "")),
        "approach":   str(row.get("approach", Path(csv_path).parent.name)),
        "model_name": str(row.get("model_name", "")),
    }


def _process_item(
    item: tuple,
    seq_num: int,
    cache: GraphCache,
    gt_cache: GTCache,
    limit: Optional[int],
    cfg: dict,
) -> Optional[dict]:
    """Process one work item. Returns a single summary row, or None if the
    building graph could not be loaded (unrecoverable — no row written)."""
    row, csv_path = item
    building = str(row.get("building", ""))
    query_id = str(row.get("query_id", f"row_{seq_num}"))
    graph = cache.acquire(building)
    if graph is None:
        print(f"  ✗ Skipping {query_id} – no graph for '{building}'", flush=True)
        return None

    base = _make_base_row(row, csv_path, seq_num)

    # Pre-check: compute value_set_f1 to decide whether to ablate.
    gen_query = str(row["ground_truth_sparql"] if cfg["use_gt"] else row["generated_sparql"])
    gt_query  = str(row["ground_truth_sparql"])
    _pre_timed_out = False
    _pre_tex = ThreadPoolExecutor(max_workers=1)
    try:
        _pre_fut = _pre_tex.submit(
            lambda: (run_query_value_set(gt_query, graph, limit),
                     run_query_value_set(gen_query, graph, limit))
        )
        try:
            gt_vals, gen_vals = _pre_fut.result(timeout=cfg["timeout"])
        except FutureTimeoutError:
            _pre_timed_out = True
            gt_vals = gen_vals = None
            print(f"    ⏰ Pre-check timed out after {cfg['timeout']}s — skipping", flush=True)
    finally:
        _pre_tex.shutdown(wait=not _pre_timed_out)

    if _pre_timed_out:
        return {**base, "skipped": True, "original_value_set_f1": "",
                "best_value_set_f1": "", "best_stmt_index": "", "best_stmt_type": "",
                "best_stmt_text": "", "removed_statements": "[]",
                "original_sparql": gen_query, "best_sparql": "",
                "delta_value_set_f1": "",
                "result_row_count": "", "result_col_count": "", "result_unique_value_count": "",
                "gt_row_count": "", "gt_col_count": "", "gt_unique_value_count": "",
                "syntax_ok": "", "timed_out": True, "error": "pre-check timeout", "elapsed_sec": "",
                "relax_attempted": False, "relax_value_set_f1": "",
                "relax_result_row_count": "", "relax_result_col_count": "",
                "relax_sparql": "", "relax_delta_value_set_f1": ""}

    if gt_vals is None or gen_vals is None:
        vsf1 = 0.0
    elif not gt_vals and not gen_vals:
        vsf1 = 1.0
    elif not gt_vals or not gen_vals:
        vsf1 = 0.0
    else:
        tp = len(gen_vals & gt_vals)
        prec = tp / len(gen_vals)
        rec  = tp / len(gt_vals)
        vsf1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
    del gt_vals, gen_vals

    if vsf1 >= cfg["threshold"]:
        print(f"\n  [{seq_num}] {query_id}  value_set_f1={vsf1:.3f} >= {cfg['threshold']} – skipped",
              flush=True)
        return {**base, "skipped": True, "original_value_set_f1": round(vsf1, 6),
                "best_value_set_f1": round(vsf1, 6), "best_stmt_index": -1,
                "best_stmt_type": "baseline", "best_stmt_text": "",
                "removed_statements": "[]",
                "original_sparql": gen_query, "best_sparql": gen_query,
                "delta_value_set_f1": 0.0,
                "result_row_count": "", "result_col_count": "", "result_unique_value_count": "",
                "gt_row_count": "", "gt_col_count": "", "gt_unique_value_count": "",
                "syntax_ok": "", "timed_out": False, "error": "", "elapsed_sec": "",
                "relax_attempted": False, "relax_value_set_f1": "",
                "relax_result_row_count": "", "relax_result_col_count": "",
                "relax_sparql": "", "relax_delta_value_set_f1": ""}

    print(f"\n  [{seq_num}] {query_id}  value_set_f1={vsf1:.3f}  …", flush=True)
    t0 = time.monotonic()

    # Run ablation with thread-based timeout (SIGALRM unsafe in non-main threads).
    query  = str(row["ground_truth_sparql"] if cfg["use_gt"] else row["generated_sparql"])
    gt     = str(row["ground_truth_sparql"])
    precomputed_gt = gt_cache.get(building, gt, graph, limit)

    timed_out = False
    ablation_results = []
    cancel_event = threading.Event()
    _tex = ThreadPoolExecutor(max_workers=1)
    try:
        if cfg["prune_mode"] is not None:
            _fut = _tex.submit(
                run_pruning, query, gt, graph,
                verbose=cfg["verbose"], limit=limit, workers=cfg["variant_workers"],
                fast=cfg["fast"], gt_rows=precomputed_gt,
                mode=cfg["prune_mode"], score_key=cfg["score_key"],
                top_k=cfg["beam_top_k"], max_frontier=cfg["beam_max_frontier"],
                max_depth=cfg["beam_max_depth"], cancel_event=cancel_event,
            )
        else:
            _fut = _tex.submit(
                run_ablation, query, gt, graph,
                verbose=cfg["verbose"], limit=limit, workers=cfg["variant_workers"],
                fast=cfg["fast"], gt_rows=precomputed_gt, cancel_event=cancel_event,
            )
        try:
            ablation_results = _fut.result(timeout=cfg["timeout"])
        except FutureTimeoutError:
            timed_out = True
            cancel_event.set()
            print(f"    ⏰ Timed out after {cfg['timeout']}s", flush=True)
        except Exception as exc:
            print(f"    ✗ Error during ablation: {exc}", flush=True)
            traceback.print_exc()
    finally:
        cancel_event.set()
        _tex.shutdown(wait=not timed_out)

    elapsed = round(time.monotonic() - t0, 3)

    if not ablation_results:
        _trim_native_heap()
        return {**base, "skipped": False, "original_value_set_f1": round(vsf1, 6),
                "best_value_set_f1": "", "best_stmt_index": "", "best_stmt_type": "",
                "best_stmt_text": "", "removed_statements": "[]",
                "original_sparql": gen_query, "best_sparql": "",
                "delta_value_set_f1": "",
                "result_row_count": "", "result_col_count": "", "result_unique_value_count": "",
                "gt_row_count": "", "gt_col_count": "", "gt_unique_value_count": "",
                "syntax_ok": "", "timed_out": timed_out, "error": "ablation failed",
                "elapsed_sec": elapsed,
                "relax_attempted": False, "relax_value_set_f1": "",
                "relax_result_row_count": "", "relax_result_col_count": "",
                "relax_sparql": "", "relax_delta_value_set_f1": ""}

    # Find best result across all variants (including baseline).
    # Pick the one with the highest value_set_f1.
    baseline_result = next((r for r in ablation_results if r["type"] == "baseline"), None)
    gt_col_count    = baseline_result.get("gt_col_count", "")         if baseline_result else ""
    gt_unique_cnt   = baseline_result.get("gt_unique_value_count", "") if baseline_result else ""
    gt_row_count_b  = baseline_result.get("gt_row_count", "")         if baseline_result else ""

    def _vsf1(r):
        v = r["scores"].get("value_set_f1", None)
        return float(v) if v is not None else -1.0

    best_r   = max(ablation_results, key=_vsf1)
    best_vsf1 = _vsf1(best_r)
    is_baseline = best_r["type"] == "baseline"

    # stmt_index: position among non-baseline results (0-based), -1 for baseline
    non_base_list = [r for r in ablation_results if r["type"] != "baseline"]
    if is_baseline:
        best_idx = -1
    else:
        best_idx = non_base_list.index(best_r) if best_r in non_base_list else -1

    delta = round(best_vsf1 - vsf1, 6) if best_vsf1 >= 0 else ""

    # Build a JSON list of each individual removed statement.
    # In greedy/beam mode removed_text is ' | '-joined; split it back out.
    _raw_removed = best_r.get("removed_text", "") if not is_baseline else ""
    removed_stmts_list = (
        [s.strip() for s in _raw_removed.split(" | ") if s.strip()]
        if _raw_removed else []
    )
    removed_statements_json = json.dumps(removed_stmts_list, ensure_ascii=False)

    print(f"      best: value_set_f1={best_vsf1:.3f} (Δ{delta:+.3f})  "
          f"← {'<baseline>' if is_baseline else _raw_removed[:60].strip()}"
          f"  [{elapsed}s]", flush=True)

    # ── Row-level coverage stats for best result ──────────────────────────────
    coverage_row: dict = {
        "gt_rows_covered": "", "excess_result_rows": "",
        "relax_gt_rows_covered": "", "relax_excess_result_rows": "",
    }
    if precomputed_gt is not None:
        best_rows, _ = run_query(best_r["query"], graph, limit)
        if best_rows is not None:
            cov = _row_coverage_stats(best_rows, precomputed_gt)
            coverage_row["gt_rows_covered"]    = cov["gt_rows_covered"]
            coverage_row["excess_result_rows"] = cov["excess_result_rows"]

    # ── Post-pruning predicate relaxation ─────────────────────────────────────
    relax_row: dict = {
        "relax_attempted":          False,
        "relax_value_set_f1":       "",
        "relax_result_row_count":   "",
        "relax_result_col_count":   "",
        "relax_sparql":             "",
        "relax_delta_value_set_f1": "",
        "relax_elapsed_sec":        "",
    }
    if cfg.get("relax_predicates", 0) > 0 and not is_baseline and removed_stmts_list:
        t_relax = time.monotonic()
        relax_cancel = threading.Event()
        relax_results = run_relaxation(
            query, gt, graph,
            verbose=cfg["verbose"], limit=limit,
            workers=cfg["variant_workers"], fast=cfg["fast"],
            gt_rows=precomputed_gt, score_key=cfg["score_key"],
            relax_predicates=cfg["relax_predicates"],
            depth=cfg.get("relax_depth", 1),
            star=cfg.get("relax_star", False),
            removed_stmt_texts=removed_stmts_list,
            cancel_event=relax_cancel,
        )
        relax_row["relax_elapsed_sec"] = round(time.monotonic() - t_relax, 3)
        relax_variants = [r for r in relax_results if r["type"] != "baseline"]
        relax_row["relax_attempted"] = True
        if relax_variants:
            best_relax = relax_variants[0]  # already sorted by score_key desc
            r_vsf1 = best_relax["scores"].get("value_set_f1", 0.0)
            relax_row["relax_value_set_f1"]        = round(r_vsf1, 6)
            relax_row["relax_result_row_count"]    = best_relax.get("row_count", "")
            relax_row["relax_result_col_count"]    = best_relax.get("col_count", "")
            relax_row["relax_sparql"]              = best_relax.get("query", "")
            relax_row["relax_delta_value_set_f1"]  = (
                round(r_vsf1 - best_vsf1, 6) if best_vsf1 >= 0 else ""
            )
            if precomputed_gt is not None:
                relax_rows, _ = run_query(best_relax["query"], graph, limit)
                if relax_rows is not None:
                    rcov = _row_coverage_stats(relax_rows, precomputed_gt)
                    coverage_row["relax_gt_rows_covered"]    = rcov["gt_rows_covered"]
                    coverage_row["relax_excess_result_rows"] = rcov["excess_result_rows"]

    _trim_native_heap()
    return {
        **base,
        "skipped":                  False,
        "original_value_set_f1":    round(vsf1, 6),
        "best_value_set_f1":        round(best_vsf1, 6) if best_vsf1 >= 0 else "",
        "best_stmt_index":          best_idx,
        "best_stmt_type":           best_r["type"],
        "best_stmt_text":           _raw_removed,
        "removed_statements":       removed_statements_json,
        "original_sparql":          gen_query,
        "best_sparql":              best_r.get("query", ""),
        "delta_value_set_f1":       delta,
        "result_row_count":         best_r.get("row_count", ""),
        "result_col_count":         best_r.get("col_count", ""),
        "result_unique_value_count": best_r.get("unique_value_count", ""),
        "gt_row_count":             gt_row_count_b,
        "gt_col_count":             gt_col_count,
        "gt_unique_value_count":    gt_unique_cnt,
        "syntax_ok":                best_r.get("syntax_ok", ""),
        "timed_out":                timed_out,
        "error":                    best_r.get("error", "") or "",
        "elapsed_sec":              elapsed,
        **relax_row,
        **coverage_row,
    }


def _worker_main(
    work_q: mp.Queue,
    result_q: mp.Queue,
    cfg: dict,
) -> None:
    """
    Worker process entry point.

    Loads its own GraphCache (fully isolated from other workers), pulls items
    from work_q, sends results to result_q. Exits when:
      - work_q is empty (gets a None sentinel)
      - queries_per_worker queries have been completed
      - RSS exceeds max_rss_mb
    Exiting drops all pyoxigraph.Store objects and returns their memory to the OS.
    """
    cache    = GraphCache(cfg["buildings_dir"])
    gt_cache = GTCache(max_size=20)
    limit    = cfg["limit"]
    queries_done = 0
    pid = os.getpid()

    while True:
        # Memory check before pulling next item
        if cfg["max_rss_mb"]:
            rss = _get_rss_mb()
            if rss > cfg["max_rss_mb"]:
                print(f"  [worker {pid}] RSS {rss:.0f} MB exceeds limit "
                      f"{cfg['max_rss_mb']} MB — exiting to free memory", flush=True)
                break

        try:
            item = work_q.get(timeout=10)
        except queue.Empty:
            break
        if item is None:  # sentinel
            break

        seq_num, work = item
        out_row = _process_item(work, seq_num, cache, gt_cache, limit, cfg)
        result_q.put(out_row)
        queries_done += 1

        if cfg["queries_per_worker"] and queries_done >= cfg["queries_per_worker"]:
            print(f"  [worker {pid}] completed {queries_done} queries — "
                  f"exiting to free memory  (RSS {_get_rss_mb():.0f} MB)", flush=True)
            break

    # Force-exit immediately. os._exit() bypasses Python's normal shutdown
    # (atexit handlers, thread joins, __del__ finalizers), which is exactly
    # what we want: any abandoned query threads from timed-out ablations would
    # otherwise block a normal exit indefinitely, preventing memory reclamation.
    # The OS reclaims all memory — Python heap, Rust allocator, ptmalloc arenas —
    # the instant the process dies, regardless of live threads.
    os._exit(0)


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch ablation: test statement removal for all low-F1 queries."
    )
    parser.add_argument("--results-dir", default="Experiment_Results", metavar="DIR",
                        help="Root directory to scan for experiment CSVs")
    parser.add_argument("--csv", nargs="+", metavar="PATH",
                        help="Explicit CSV file(s); skips --results-dir scan")
    parser.add_argument("--buildings-dir", default="eval_buildings", metavar="DIR",
                        help="Directory containing building .ttl files")
    parser.add_argument("--output", default="ablation_batch_results.csv", metavar="FILE",
                        help="Output CSV path (default: ablation_batch_results.csv)")
    parser.add_argument("--timeout", type=int, default=18000, metavar="SEC",
                        help="Per-query ablation timeout in seconds (default: 18000)")
    parser.add_argument("--threshold", type=float, default=1.0, metavar="FLOAT",
                        help="Only process rows where value_set_f1 < FLOAT (default: 1.0)")
    parser.add_argument("--max-queries", type=int, default=None, metavar="N",
                        help="Stop after processing N queries (for testing)")
    parser.add_argument("--use-gt", action="store_true",
                        help="Ablate ground-truth queries instead of generated ones")
    parser.add_argument("--verbose", action="store_true",
                        help="Print each variant query during ablation")
    parser.add_argument("--limit", type=int, default=40_000, metavar="N",
                        help="Add LIMIT N to every query that lacks one (default: 40000,"
                             " pass 0 to disable)")
    parser.add_argument("--workers", type=int, default=8, metavar="N",
                        help="Number of worker processes (default: 8)")
    parser.add_argument("--variant-workers", type=int, default=5, metavar="N",
                        help="Threads for ablation variants within a single query (default: 5)")
    parser.add_argument("--fast", action="store_true",
                        help="Skip column alignment; score only with value_set_f1 (faster)")
    parser.add_argument("--skip-buildings", nargs="+", metavar="NAME", default=[],
                        help="Building name(s) to skip entirely")
    parser.add_argument("--resume", action="store_true",
                        help="Append to --output and skip already-completed combinations")
    parser.add_argument("--prune-mode", choices=["single", "greedy", "beam"],
                        default=None, metavar="MODE",
                        help="Use sparql_prune instead of sparql_ablation.")
    parser.add_argument("--score-key", default="value_set_f1", metavar="METRIC",
                        help="Metric to optimise in greedy/beam prune modes (default: value_set_f1)")
    parser.add_argument("--beam-top-k", type=int, default=5, metavar="K",
                        help="Beam width in beam prune mode (default: 5)")
    parser.add_argument("--beam-max-frontier", type=int, default=500, metavar="N",
                        help="Max frontier size in beam phase-1 (default: 500)")
    parser.add_argument("--beam-max-depth", type=int, default=0, metavar="D",
                        help="Max search depth for greedy/beam modes (0 = unlimited, default: 0)")
    parser.add_argument("--relax-predicates", type=int, default=0, metavar="N",
                        help="After pruning, try predicate relaxations using the top-N most "
                             "frequent graph predicates per segment. "
                             "Total variants per query = N^depth × relaxable_triples. "
                             "0 = disabled (default)")
    parser.add_argument("--relax-depth", type=int, default=1, metavar="D",
                        help="Number of chained segments in each relaxation path "
                             "((^<p>)?|<p>?)/... — depth=1 gives N variants, "
                             "depth=2 gives N² variants, etc. (default: 1)")
    parser.add_argument("--relax-star", action="store_true",
                        help="Use * (zero-or-more) instead of ? (zero-or-one) in relaxation "
                             "paths. Broader but slower.")
    parser.add_argument("--queries-per-worker", type=int, default=50, metavar="N",
                        help="Restart a worker process after N completed queries to free "
                             "accumulated memory (default: 50, 0 = never restart)")
    parser.add_argument("--max-rss-mb", type=int, default=8000, metavar="MB",
                        help="Restart a worker process when its RSS exceeds MB "
                             "(default: 8000)")
    parser.add_argument("--restart-after", type=int, default=200, metavar="N",
                        help="Re-exec the supervisor process after N total completed queries "
                             "to free all accumulated memory in the parent (default: 200, "
                             "0 = disabled). Uses os.execv so the process image is fully "
                             "replaced; --resume is added automatically.")

    args = parser.parse_args()
    limit = args.limit if args.limit > 0 else None

    cfg = {
        "buildings_dir":   args.buildings_dir,
        "timeout":         args.timeout,
        "threshold":       args.threshold,
        "use_gt":          args.use_gt,
        "verbose":         args.verbose,
        "limit":           limit,
        "variant_workers": args.variant_workers,
        "fast":            args.fast,
        "prune_mode":      args.prune_mode,
        "score_key":       args.score_key,
        "beam_top_k":      args.beam_top_k,
        "beam_max_frontier": args.beam_max_frontier,
        "beam_max_depth":  args.beam_max_depth,
        "queries_per_worker": args.queries_per_worker,
        "max_rss_mb":      args.max_rss_mb,
        "restart_after":   args.restart_after,
        "relax_predicates": args.relax_predicates,
        "relax_depth":      args.relax_depth,
        "relax_star":       args.relax_star,
    }

    # ── discover CSVs ──────────────────────────────────────────────────────────
    csv_paths = args.csv if args.csv else find_csvs(args.results_dir)
    if not csv_paths:
        sys.exit(f"No CSV files found under {args.results_dir}")

    # ── build set of already-completed keys (for --resume) ────────────────────
    # Each query (whether skipped or fully ablated) produces exactly one row, so
    # (source_csv, query_id, building) uniquely identifies a completed entry.
    done_keys: set = set()
    if args.resume and Path(args.output).exists():
        existing = pd.read_csv(args.output, usecols=["source_csv", "query_id", "building"])
        done_keys = set(zip(existing["source_csv"], existing["query_id"], existing["building"]))
        print(f"Resuming: {len(done_keys)} already-completed queries will be skipped.\n")

    skip_buildings = set(args.skip_buildings)

    # ── collect all (row, csv_path) pairs to process ──────────────────────────
    work_items: List[tuple] = []
    for csv_path in csv_paths:
        df = load_rows(csv_path)
        if df.empty:
            print(f"  {Path(csv_path).name}: no SPARQL rows – skipped")
            continue
        print(f"  {csv_path}  ({len(df)} rows)")
        for _, row in df.iterrows():
            if str(row.get("building", "")) in skip_buildings:
                continue
            if done_keys:
                key = (csv_path, str(row.get("query_id", "")), str(row.get("building", "")))
                if key in done_keys:
                    continue
            work_items.append((row.to_dict(), csv_path))

    if args.max_queries is not None:
        work_items = work_items[: args.max_queries]

    print(f"\n{'='*70}")
    print(f"  Total queries to process: {len(work_items)}")
    print(f"{'='*70}\n")

    limit_str = str(limit) if limit else "off"
    print(f"workers={args.workers}  queries-per-worker={args.queries_per_worker}"
          f"  max-rss-mb={args.max_rss_mb or 'off'}"
          f"  timeout={args.timeout}s  limit={limit_str}  output={args.output}\n")

    # ── set up multiprocessing queues and result writer ───────────────────────
    work_q:   mp.Queue = mp.Queue()
    result_q: mp.Queue = mp.Queue()
    writer = ResultWriter(args.output, resume=args.resume)

    # Fill work queue
    for i, item in enumerate(work_items):
        work_q.put((i + 1, item))

    total_work     = len(work_items)
    total_done    = 0
    total_queries = 0

    # ── supervisor loop ───────────────────────────────────────────────────────
    # Maintain a pool of worker processes. When one exits (hit memory/query
    # limit), start a fresh one as long as there's still work to do.
    live: Dict[int, mp.Process] = {}  # pid → Process

    def _spawn() -> None:
        p = mp.Process(target=_worker_main, args=(work_q, result_q, cfg), daemon=True)
        p.start()
        live[p.pid] = p

    def _reexec() -> None:
        """Replace this process with a fresh Python interpreter via os.execv.

        All memory — Python heap, Rust/ptmalloc arenas buffered in the
        supervisor — is freed instantly by the kernel. The fresh process
        re-reads the output CSV via --resume to skip already-written rows.
        """
        writer.close()
        for p in live.values():
            p.terminate()
        for p in live.values():
            p.join(timeout=3)
        new_argv = sys.argv[:]
        if "--resume" not in new_argv:
            new_argv.append("--resume")
        print(f"\n  [supervisor] re-execing after {total_done} queries to free memory "
              f"(RSS {_get_rss_mb():.0f} MB) …\n", flush=True)
        os.execv(sys.executable, [sys.executable] + new_argv)

    # Start initial workers
    for _ in range(min(args.workers, total_work)):
        _spawn()

    try:
        while total_done < total_work:
            # Collect any results that are ready
            drained = False
            while True:
                try:
                    out_row = result_q.get(timeout=0.1)
                    drained = True
                except queue.Empty:
                    break
                total_done += 1
                if out_row is not None:
                    total_queries += 1
                    writer.write(out_row)

                # Re-exec the supervisor process to free accumulated memory.
                if args.restart_after and total_done % args.restart_after == 0:
                    _reexec()  # does not return

            # Reap dead workers and replace them if work remains
            for pid in list(live):
                p = live[pid]
                if not p.is_alive():
                    p.join()
                    del live[pid]
                    # Only spawn if there are items still in the queue
                    if not work_q.empty() and len(live) < args.workers:
                        _spawn()

            if not drained:
                time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nInterrupted — stopping workers …", flush=True)
        for p in live.values():
            p.terminate()
    finally:
        for p in live.values():
            p.join(timeout=5)
        writer.close()

    print(f"\n{'='*70}")
    print(f"Done.  Processed {total_queries} queries")
    print(f"Results written to: {args.output}")
    os._exit(0)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
