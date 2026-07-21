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
    One bar per approach: solid = diagnose (search-only) F1, hatched extension
    on top = the extra F1 gained once relax's fix (where found) is folded in.
    """
    by_a = (
        df.groupby("approach")
        .agg(
            n=("diagnose_value_set_f1", "count"),
            diagnose_f1=("diagnose_value_set_f1", "mean"),
            combined_f1=("combined_f1", "mean"),
        )
        .reset_index()
        .sort_values("diagnose_f1", ascending=False)
    )
    if by_a.empty:
        return

    x = np.arange(len(by_a))
    diag = by_a["diagnose_f1"].to_numpy()
    comb = by_a["combined_f1"].to_numpy()
    delta = np.maximum(comb - diag, 0)

    fig, ax = plt.subplots(figsize=(max(6, len(by_a) * 1.1), 4))

    bars = ax.bar(x, diag, 0.6, label="Diagnose F1 (search only)", color="#4a90d9", zorder=3)
    ax.bar(x, delta, 0.6, bottom=diag, label="+ Relax fix", color="#e07b54",
           alpha=0.7, edgecolor="#e07b54", hatch="///", zorder=3)

    for xi, g, c in zip(x, diag, comb):
        ax.text(xi, g + 0.005, f"{g:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)
        if c > g:
            ax.text(xi, c + 0.005, f"{c:.3f}", ha="center", va="bottom", fontsize=7,
                     rotation=90, color="#e07b54", fontstyle="italic")

    xtick_labels = [f"{a}\n(n={n})" for a, n in zip(by_a["approach"], by_a["n"])]

    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, rotation=30, ha="right", fontsize=7.5)
    ax.set_ylim(0, min(1.0, max(comb.max(), diag.max()) + 0.15) if len(comb) else 1.0)
    ax.set_ylabel("Mean value-set F1")
    ax.set_title("Diagnose vs. Combined (search+relax) F1 by Approach", fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    out = "approach_f1_combined.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Chart saved to {out}")
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
