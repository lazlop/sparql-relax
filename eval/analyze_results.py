#!/usr/bin/env python3
"""
analyze_results.py

Reads a sparql-relax batch evaluation CSV (one row per query, as produced by
run_eval.py) and reports:
  - overall run summary (errors, timeouts, relax attempts/successes)
  - per-approach comparison table (diagnose vs. relax vs. combined F1 /
    row coverage / excess rows)
  - per-building comparison table
  - error / timeout breakdown by approach
  - bar chart comparing diagnose-only F1 against combined (relax-if-available,
    else diagnose) F1 per approach

Usage:
  python3 analyze_results.py [csv_path]
  python3 analyze_results.py current-results.csv
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_PATH = "current-results.csv"

NUMERIC_COLS = [
    "gt_rows", "gt_col_count", "gt_unique_value_count", "gen_rows",
    "diagnose_value_set_f1", "diagnose_rows_covered", "diagnose_excess_rows",
    "relax_value_set_f1", "relax_rows_covered", "relax_excess_rows",
    "relax_result_row_count", "relax_result_col_count", "elapsed_sec",
]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else CSV_PATH
    df = pd.read_csv(path)

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["timed_out"] = df["timed_out"].astype(str).str.lower().isin(("true", "1"))
    df["relax_attempted"] = df["relax_attempted"].astype(str).str.lower().isin(("true", "1"))
    df["errored"] = df["error"].notna() & (df["error"].astype(str).str.strip() != "")

    # Search (diagnose) phase: rows returned by the original query, whether or
    # not they matched ground truth.
    df["search_result_rows"] = df["diagnose_rows_covered"].fillna(0) + df["diagnose_excess_rows"].fillna(0)
    df["search_coverage_pct"] = df["diagnose_rows_covered"] / df["gt_rows"].replace(0, np.nan)
    df["relax_coverage_pct"] = df["relax_rows_covered"] / df["gt_rows"].replace(0, np.nan)

    # relax_value_set_f1 is only populated when the relax search found a fix;
    # combined = that fix's F1, else fall back to the original diagnose F1.
    relax_has_fix = df["relax_value_set_f1"].notna()
    df["combined_f1"] = df["relax_value_set_f1"].fillna(df["diagnose_value_set_f1"])
    df["combined_coverage_pct"] = df["relax_coverage_pct"].where(relax_has_fix, df["search_coverage_pct"])
    df["relax_improved"] = relax_has_fix & (
        df["relax_value_set_f1"] > df["diagnose_value_set_f1"].fillna(0)
    )

    _print_overview(df)
    _print_approach_table(df)
    _print_building_table(df)
    _print_removal_type_table(df)
    _print_error_table(df)
    _plot_approach_f1_combined(df)
    _plot_building_f1(df)


# ── summary sections ─────────────────────────────────────────────────────────


def _print_overview(df: pd.DataFrame) -> None:
    n = len(df)
    n_err = int(df["errored"].sum())
    n_timeout = int(df["timed_out"].sum())
    n_attempted = int(df["relax_attempted"].sum())
    n_fixed = int(df["relax_value_set_f1"].notna().sum())
    n_improved = int(df["relax_improved"].sum())

    print("=" * 80)
    print("OVERVIEW")
    print("=" * 80)
    print(f"Total rows:            {n}")
    print(f"  Errored:             {n_err}  ({n_err / n:.1%})")
    print(f"  Timed out:           {n_timeout}  ({n_timeout / n:.1%})")
    print(f"  Relax attempted:     {n_attempted}  ({n_attempted / n:.1%})")
    print(f"  Relax found a fix:   {n_fixed}  ({n_fixed / n:.1%})")
    print(f"  Relax improved F1:   {n_improved}  ({n_improved / n:.1%})")

    diag_f1 = df["diagnose_value_set_f1"].dropna()
    comb_f1 = df["combined_f1"].dropna()
    print(f"\n  Mean diagnose F1 (search only): {diag_f1.mean():.4f} ± {diag_f1.std():.4f}  [n={len(diag_f1)}]")
    print(f"  Mean combined F1 (search+relax): {comb_f1.mean():.4f} ± {comb_f1.std():.4f}  [n={len(comb_f1)}]")
    print()


def _print_approach_table(df: pd.DataFrame) -> None:
    def _agg(g: pd.DataFrame) -> pd.Series:
        diag = g["diagnose_value_set_f1"].dropna()
        comb = g["combined_f1"].dropna()
        cov = g["search_coverage_pct"].dropna()
        exc = g["diagnose_excess_rows"].dropna()
        return pd.Series({
            "n": len(g),
            "diagnose_f1_mean": diag.mean(),
            "diagnose_f1_std": diag.std(),
            "combined_f1_mean": comb.mean(),
            "combined_f1_std": comb.std(),
            "row_cov_mean": cov.mean(),
            "excess_rows_mean": exc.mean(),
            "n_relax_fixed": int(g["relax_value_set_f1"].notna().sum()),
            "n_relax_improved": int(g["relax_improved"].sum()),
            "errored_pct": g["errored"].mean(),
            "timed_out_pct": g["timed_out"].mean(),
        })

    table = (
        df.groupby("approach")
        .apply(_agg, include_groups=False)
        .reset_index()
        .sort_values("combined_f1_mean", ascending=False)
    )

    print("=" * 80)
    print("APPROACH COMPARISON  (diagnose = search-only, combined = relax-if-fixed else diagnose)")
    print("=" * 80)
    hdr = (f"  {'Approach':<42}  {'n':>5}  {'Diag F1':>14}  {'Comb F1':>14}  "
           f"{'RowCov%':>8}  {'ExcRows':>8}  {'Fixed':>6}  {'Improved':>9}  "
           f"{'Err%':>6}  {'TO%':>6}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for _, r in table.iterrows():
        diag_str = f"{r['diagnose_f1_mean']:.3f} ±{r['diagnose_f1_std']:.3f}" if not pd.isna(r["diagnose_f1_mean"]) else "n/a"
        comb_str = f"{r['combined_f1_mean']:.3f} ±{r['combined_f1_std']:.3f}" if not pd.isna(r["combined_f1_mean"]) else "n/a"
        cov = r["row_cov_mean"]
        exc = r["excess_rows_mean"]
        print(f"  {r['approach']:<42}  {int(r['n']):>5}  {diag_str:>14}  {comb_str:>14}  "
              f"{cov:>8.3f}  {exc:>8.2f}  {int(r['n_relax_fixed']):>6}  {int(r['n_relax_improved']):>9}  "
              f"{r['errored_pct']:>6.1%}  {r['timed_out_pct']:>6.1%}")
    print()


def _print_building_table(df: pd.DataFrame) -> None:
    if "building" not in df.columns:
        return

    def _agg(g: pd.DataFrame) -> pd.Series:
        diag = g["diagnose_value_set_f1"].dropna()
        comb = g["combined_f1"].dropna()
        cov = g["search_coverage_pct"].dropna()
        return pd.Series({
            "n": len(g),
            "diagnose_f1_mean": diag.mean(),
            "combined_f1_mean": comb.mean(),
            "row_cov_mean": cov.mean(),
            "n_relax_fixed": int(g["relax_value_set_f1"].notna().sum()),
            "errored_pct": g["errored"].mean(),
        })

    table = (
        df.groupby("building")
        .apply(_agg, include_groups=False)
        .reset_index()
        .sort_values("combined_f1_mean", ascending=False)
    )

    print("=" * 80)
    print("BUILDING COMPARISON")
    print("=" * 80)
    hdr = f"  {'Building':<24}  {'n':>5}  {'Diag F1':>9}  {'Comb F1':>9}  {'RowCov%':>8}  {'Fixed':>6}  {'Err%':>6}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for _, r in table.iterrows():
        print(f"  {r['building']:<24}  {int(r['n']):>5}  {r['diagnose_f1_mean']:>9.3f}  "
              f"{r['combined_f1_mean']:>9.3f}  {r['row_cov_mean']:>8.3f}  "
              f"{int(r['n_relax_fixed']):>6}  {r['errored_pct']:>6.1%}")
    print()


def _print_removal_type_table(df: pd.DataFrame) -> None:
    """Breakdown of which statement-type removal relax used, for rows where it found a fix."""
    if "relax_stmt_type" not in df.columns:
        return
    fixed = df[df["relax_value_set_f1"].notna()]
    if fixed.empty:
        return

    table = (
        fixed.groupby("relax_stmt_type")
        .agg(
            n_queries=("relax_value_set_f1", "count"),
            mean_relax_f1=("relax_value_set_f1", "mean"),
            mean_delta=("relax_improved", "mean"),
        )
        .reset_index()
        .sort_values("n_queries", ascending=False)
    )

    print("=" * 80)
    print("RELAX FIX TYPE BREAKDOWN  (queries where relax found a fix)")
    print("=" * 80)
    print(table.to_string(index=False))
    print()


def _print_error_table(df: pd.DataFrame) -> None:
    if not df["errored"].any():
        return

    table = (
        df[df["errored"]]
        .groupby("approach")
        .agg(n_errors=("errored", "count"), n_timeouts=("timed_out", "sum"))
        .reset_index()
        .sort_values("n_errors", ascending=False)
    )

    print("=" * 80)
    print("ERRORS BY APPROACH")
    print("=" * 80)
    print(table.to_string(index=False))
    print()


# ── plots ─────────────────────────────────────────────────────────────────────


def _plot_approach_f1_combined(df: pd.DataFrame) -> None:
    """
    One bar per approach: solid = diagnose (search-only) F1 / row coverage %,
    hatched extension on top = the extra F1 / coverage gained once relax's fix
    (where found) is folded in. Both series use the same denominator (all
    processed queries), making the comparison fair.

    Each bar also carries a second, lighter hatch capping the portion of its
    mean attributable to queries that actually returned some rows to score
    against — rather than an empty result set, which scores 0 by construction.
    For the combined series that gate is "diagnose OR relax returned rows":
    gating on relax alone would wrongly discount a query that search already
    nailed just because relax wasn't attempted or found nothing further to fix.

    Per-approach means are first averaged within each model_name, then
    averaged across models, so one high-volume model doesn't dominate an
    approach that spans several models (e.g. ReAct(100), run on both
    deepseek-r1 and llama).
    """
    processed = df.copy()
    if "model_name" not in processed.columns:
        processed["model_name"] = "unknown"

    # Distinguish "relax not attempted" (NaN) from "attempted but found
    # nothing better" (0) before filling, so combined_f1's fallback stays correct.
    relax_mask = processed["relax_value_set_f1"].notna()

    for col in ["diagnose_rows_covered", "diagnose_excess_rows", "relax_rows_covered", "relax_result_row_count"]:
        processed[col] = processed[col].fillna(0)

    processed["diagnose_result_row_count"] = processed["diagnose_rows_covered"] + processed["diagnose_excess_rows"]

    processed["gt_rows_covered_pct"] = processed["diagnose_rows_covered"] / processed["gt_rows"].replace(0, np.nan)
    combined_covered = processed["relax_rows_covered"].where(relax_mask, processed["diagnose_rows_covered"])
    processed["combined_row_cov_pct"] = combined_covered / processed["gt_rows"].replace(0, np.nan)

    # ── Global aggregation (all processed queries — same denominator) ──────────
    by_ma = (
        processed.groupby(["model_name", "approach"])
        .agg(
            n=("diagnose_value_set_f1", "count"),
            mean_res=("diagnose_value_set_f1", "mean"),
            mean_combined=("combined_f1", "mean"),
            mean_row_cov=("gt_rows_covered_pct", "mean"),
            mean_combined_row_cov=("combined_row_cov_pct", "mean"),
        )
        .reset_index()
    )

    by_a = (
        by_ma.groupby("approach")
        .agg(
            n=("n", "sum"),
            mean_res=("mean_res", "mean"),
            mean_combined=("mean_combined", "mean"),
            mean_row_cov=("mean_row_cov", "mean"),
            mean_combined_row_cov=("mean_combined_row_cov", "mean"),
        )
        .reset_index()
        .sort_values("mean_res", ascending=False)
    )

    if by_a.empty:
        return

    # ── Filtered aggregation: search — queries where search returned rows ──────
    search_hit = processed[processed["diagnose_result_row_count"] > 0]
    by_a_sf = (
        search_hit.groupby(["model_name", "approach"])
        .agg(
            mean_res_filt=("diagnose_value_set_f1", "mean"),
            mean_row_cov_filt=("gt_rows_covered_pct", "mean"),
        )
        .reset_index()
        .groupby("approach")
        .agg(mean_res_filt=("mean_res_filt", "mean"), mean_row_cov_filt=("mean_row_cov_filt", "mean"))
        .reset_index()
    )
    by_a = by_a.merge(by_a_sf, on="approach", how="left")

    # ── Filtered aggregation: combined — queries where *either* phase (search
    # or relax) returned rows to score against ──────────────────────────────
    any_hit = processed[
        (processed["diagnose_result_row_count"] > 0) | (processed["relax_result_row_count"] > 0)
    ]
    by_a_cf = (
        any_hit.groupby(["model_name", "approach"])
        .agg(
            mean_combined_filt=("combined_f1", "mean"),
            mean_combined_row_cov_filt=("combined_row_cov_pct", "mean"),
        )
        .reset_index()
        .groupby("approach")
        .agg(
            mean_combined_filt=("mean_combined_filt", "mean"),
            mean_combined_row_cov_filt=("mean_combined_row_cov_filt", "mean"),
        )
        .reset_index()
    )
    by_a = by_a.merge(by_a_cf, on="approach", how="left")

    # ── Rename approaches ──────────────────────────────────────────────────────
    rename_map = {
        "ReAct(w5000triples)_google/gemini-flash": "R5000, G",
        "ReAct(5000)": "R5000, L",
        "ReAct(w5000triples)": "R5000, O3",
        "ReAct(100)": "R100, D",
        "ReAct(w100triples)_google/gemini-flash": "R100, G",
        "ReAct(w100triples)_4o-mini": "R100, O4",
        "ReAct(w100triples)": "R100, O3",
        "ReACT(noKG)": "R, O3",
        "ReACT(noKG)_test_google/gemini-flash": "R, G",
        "ReAct(noKG)": "R, O3b",
        "dakgqa": "DA, O3",
        "DA-KGQA": "DA, L",
        "dakgqa_google/gemini-flash": "DA, G",
    }
    by_a = by_a[by_a["approach"].isin(rename_map.keys())]
    by_a["approach"] = by_a["approach"].map(lambda a: rename_map.get(a, a))

    if by_a.empty:
        return

    approach_order = by_a["approach"].tolist()
    x = np.arange(len(approach_order))

    COLORS = {
        "result":       "#4a90d9",
        "combined":     "#e07b54",
        "row_cov":      "#5bbf9e",
        "comb_row_cov": "#d94f4f",
    }

    bar_series = [
        ("Search Value Set", by_a["mean_res"].tolist(), by_a["mean_res_filt"].tolist(), COLORS["result"]),
        ("Search+Relax Value Set", by_a["mean_combined"].tolist(), by_a["mean_combined_filt"].tolist(), COLORS["combined"]),
        ("Search Row Coverage %", by_a["mean_row_cov"].tolist(), by_a["mean_row_cov_filt"].tolist(), COLORS["row_cov"]),
        ("Search+Relax Row Coverage %", by_a["mean_combined_row_cov"].tolist(), by_a["mean_combined_row_cov_filt"].tolist(), COLORS["comb_row_cov"]),
    ]

    n_series = len(bar_series)
    total_bar_width = 0.75
    w = total_bar_width / n_series
    offsets = np.linspace(-(total_bar_width - w) / 2, (total_bar_width - w) / 2, n_series)

    fig, ax = plt.subplots(figsize=(max(6, len(approach_order) * 1.1), 3.5))

    all_bars = []
    for offset, (label, global_vals, filt_vals, color) in zip(offsets, bar_series):
        bars = ax.bar(x + offset, global_vals, w, label=label, color=color, zorder=3)
        all_bars.extend(bars)

        filt_arr = np.array(filt_vals, dtype=float)
        glob_arr = np.array(global_vals, dtype=float)
        delta = np.where(np.isnan(filt_arr), 0, filt_arr - glob_arr)
        delta = np.maximum(delta, 0)

        hatched_bars = ax.bar(
            x + offset, delta, w,
            bottom=glob_arr,
            color=color, alpha=0.5,
            edgecolor=color, linewidth=1.0,
            hatch="///",
            zorder=3,
            label=f"{label} (rows>0)",
        )
        for bar_h, g, f_val in zip(hatched_bars, glob_arr, filt_arr):
            if not np.isnan(f_val) and f_val > g:
                ax.text(bar_h.get_x() + bar_h.get_width() / 2, f_val + 0.005,
                        f"{f_val:.3f}", ha="center", va="bottom",
                        fontsize=7, rotation=90, color=color, fontstyle="italic")

    for bar in all_bars:
        h = bar.get_height()
        if not np.isnan(h) and h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=6, rotation=90)

    ax.legend(fontsize=6, loc="upper right", ncol=2)

    for i, row in by_a.reset_index(drop=True).iterrows():
        ax.text(i - 0.1, -0.1, f"n={int(row['n'])}", ha="center", va="top",
                fontsize=6.5, color="gray",
                transform=ax.get_xaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels(approach_order, rotation=0, ha="right", fontsize=8)

    active_cols = [
        "mean_res", "mean_combined", "mean_row_cov", "mean_combined_row_cov",
        "mean_res_filt", "mean_combined_filt", "mean_row_cov_filt", "mean_combined_row_cov_filt",
    ]
    max_val = by_a[active_cols].max().max()
    ax.set_ylim(0, min(1.0, max_val + 0.15))
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_ylabel("Mean Score")
    ax.set_title(
        "Value-Set F1 & Row Coverage — Search vs Search+Relax  (same denominator: all processed queries)",
        fontsize=9,
    )

    fig.tight_layout()
    out = "approach_f1_combined.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to {out}")
    plt.show()


def _plot_building_f1(df: pd.DataFrame) -> None:
    """Grouped bars: diagnose vs. combined F1, one group per building."""
    if "building" not in df.columns:
        return

    by_b = (
        df.groupby("building")
        .agg(
            n=("diagnose_value_set_f1", "count"),
            diagnose_f1=("diagnose_value_set_f1", "mean"),
            combined_f1=("combined_f1", "mean"),
        )
        .reset_index()
        .sort_values("building")
    )
    if by_b.empty:
        return

    x = np.arange(len(by_b))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(5, len(by_b) * 1.3), 3.5))

    ax.bar(x - w / 2, by_b["diagnose_f1"], w, label="Diagnose F1", color="#4a90d9", zorder=3)
    ax.bar(x + w / 2, by_b["combined_f1"], w, label="Combined F1", color="#e07b54", zorder=3)

    for xi, row in zip(x, by_b.itertuples()):
        ax.text(xi - w / 2, row.diagnose_f1 + 0.005, f"{row.diagnose_f1:.3f}",
                 ha="center", va="bottom", fontsize=7, rotation=90)
        ax.text(xi + w / 2, row.combined_f1 + 0.005, f"{row.combined_f1:.3f}",
                 ha="center", va="bottom", fontsize=7, rotation=90)
        ax.text(xi, -0.06, f"n={row.n}", ha="center", va="top", fontsize=6.5,
                 color="gray", transform=ax.get_xaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels(by_b["building"], fontsize=8)
    ax.set_ylim(0, min(1.0, by_b[["diagnose_f1", "combined_f1"]].max().max() + 0.15))
    ax.set_ylabel("Mean value-set F1")
    ax.set_title("Diagnose vs. Combined F1 by Building", fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = "building_f1.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Chart saved to {out}")
    plt.show()


if __name__ == "__main__":
    main()
