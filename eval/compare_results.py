"""Compares sparql-relax-rs's current-results.csv against previous-results.csv, a prior
evaluation from BuildingQA's old Python ablation (batch_ablation.py — the pure-Python
predicate-substitution search sparql-relax-rs's Rust `diagnose_and_relax` replaced).

current-results.csv (via run_eval.py) now only covers rows whose *original* generated query
returned zero results — the population diagnosis exists for. previous-results.csv predates that
filter and covers every row batch_ablation.py evaluated, most of which never had zero results to
begin with. Comparing the two files as-is would pair up sparql-relax-rs's zero-result rows against
batch_ablation.py rows that didn't start from the same problem, so both sides are filtered to
`result_row_count == 0` (the original generated query's row count, recomputed at eval time by
either script) before comparing, putting them on the same population.

`best_value_set_f1` — not `delta_value_set_f1` — is the metric compared: it's the actual final
value-set F1 each script's diagnose+relax pipeline produced (falling back to the original score
when nothing better was found), whereas delta is a difference that collapses to 0 in that fallback
case even when a relaxed candidate was still scored. See the two scripts' own diagnose/relax
functions for how each computes it — the columns share a name but not an implementation, so this
compares outcomes, not internal mechanics.
"""

import pandas as pd


def normalize_path(path):
    if not isinstance(path, str):
        return path
    return path.replace('/home/lazlo/Desktop/semantics/sparql-relax/eval/', '')


def load_zero_result_rows(path: str) -> pd.DataFrame:
    """Loads `path` and filters to rows whose original generated query returned zero results —
    `result_row_count` is recomputed at eval time by both scripts, so it means the same thing in
    either file regardless of which one produced it."""
    df = pd.read_csv(path)
    df['norm_csv'] = df['source_csv'].apply(normalize_path)
    return df[df['result_row_count'] == 0]


current_df = load_zero_result_rows('current-results.csv')
previous_df = load_zero_result_rows('previous-results.csv')

print(f"current-results.csv: {len(current_df)} originally-zero-result rows")
print(f"previous-results.csv: {len(previous_df)} originally-zero-result rows")

key = ['norm_csv', 'query_id', 'building', 'question']
merged = current_df.merge(
    previous_df[key + ['original_value_set_f1', 'best_value_set_f1', 'gt_rows_covered', 'excess_result_rows']],
    on=key,
    suffixes=('_curr', '_prev'),
)

if merged.empty:
    print("No matching rows found between current and previous results.")
else:
    unmatched_curr = len(current_df) - len(merged)
    unmatched_prev = len(previous_df) - len(merged)
    print(f"Matched {len(merged)} rows on {key} "
          f"({unmatched_curr} current / {unmatched_prev} previous rows had no counterpart).\n")

    # Sanity check: with both sides filtered to originally-zero-result rows, original_value_set_f1
    # should be ~0 on both sides (nonzero only when the ground truth also returned zero rows — see
    # calculate_f1's "both empty" convention). A mismatch here would mean the two files' filters
    # aren't actually selecting the same kind of row.
    orig_mismatch = merged[merged['original_value_set_f1_curr'] != merged['original_value_set_f1_prev']]
    if not orig_mismatch.empty:
        print(f"Warning: {len(orig_mismatch)} rows have different original_value_set_f1 between "
              f"current and previous despite both being originally-zero-result rows.\n")

    avg_curr = merged['best_value_set_f1_curr'].mean()
    avg_prev = merged['best_value_set_f1_prev'].mean()
    print(f"Average best_value_set_f1 (Current, sparql-relax-rs): {avg_curr:.4f}")
    print(f"Average best_value_set_f1 (Previous, batch_ablation.py): {avg_prev:.4f}")
    print(f"Delta: {avg_curr - avg_prev:+.4f}")

    improved = (merged['best_value_set_f1_curr'] > merged['best_value_set_f1_prev']).sum()
    regressed = (merged['best_value_set_f1_curr'] < merged['best_value_set_f1_prev']).sum()
    unchanged = (merged['best_value_set_f1_curr'] == merged['best_value_set_f1_prev']).sum()
    print(f"Improved: {improved}, Regressed: {regressed}, Unchanged: {unchanged}\n")

    avg_cov_curr = merged['gt_rows_covered_curr'].mean()
    avg_cov_prev = merged['gt_rows_covered_prev'].mean()
    avg_exc_curr = merged['excess_result_rows_curr'].mean()
    avg_exc_prev = merged['excess_result_rows_prev'].mean()
    print(f"Average gt_rows_covered: current={avg_cov_curr:.3f}  previous={avg_cov_prev:.3f}")
    print(f"Average excess_result_rows: current={avg_exc_curr:.3f}  previous={avg_exc_prev:.3f}")
