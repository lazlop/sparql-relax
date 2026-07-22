"""Compares sparql-relax-rs's current-results.csv against previous-results.csv, a prior
evaluation from BuildingQA's old Python ablation (batch_ablation.py — the pure-Python
predicate-substitution search sparql-relax-rs's Rust `diagnose_and_connect` replaced).

current-results.csv (via run_eval.py) now only covers rows whose *original* generated query
returned zero results — the population diagnosis exists for. previous-results.csv predates that
filter and covers every row batch_ablation.py evaluated, most of which never had zero results to
begin with. Comparing the two files as-is would pair up sparql-relax-rs's zero-result rows against
batch_ablation.py rows that didn't start from the same problem, so both sides are filtered to
their own file's "original generated query had zero rows" column before comparing, putting them
on the same population. That column is named differently in each file — previous-results.csv
(and batch_ablation.py) call it `result_row_count`; current-results.csv (run_eval.py, see its
`gen_rows` field) calls it `gen_rows` and is *already* zero for every row by construction (see
run_eval.py's module docstring — it only ever processes originally-zero-result rows in the first
place), so the filter is a no-op there but kept anyway to make the "same population" invariant
explicit rather than assumed.

`best_value_set_f1` — not `delta_value_set_f1` — is the metric compared: it's the actual final
value-set F1 each script's diagnose+relax pipeline produced (falling back to the original score
when nothing better was found), whereas delta is a difference that collapses to 0 in that fallback
case even when a relaxed candidate was still scored. previous-results.csv has this as a single
`best_value_set_f1` column already. current-results.csv (run_eval.py's newer, split schema) instead
has separate `diagnose_value_set_f1` (the pre-relax score, ~0 by construction) and
`relax_value_set_f1` (blank whenever relaxation was skipped or found nothing better — see
run_eval.py's `_blank_relax_fields`), so `load_current_zero_result_rows` below reconstructs the
same "final outcome, falling back to the original score" value as `best_value_set_f1` — and
likewise for `gt_rows_covered`/`excess_result_rows` from their `diagnose_*`/`relax_*` counterparts —
so both sides compare on identically-named, identically-defined columns. See the two scripts' own
diagnose/relax functions for how each computes its inputs — the reconstructed columns share a name
but not an implementation, so this compares outcomes, not internal mechanics.
"""

import pandas as pd


def normalize_path(path):
    if not isinstance(path, str):
        return path
    return path.replace('/home/lazlo/Desktop/semantics/sparql-relax/eval/', '')


def load_current_zero_result_rows(path: str) -> pd.DataFrame:
    """Loads current-results.csv (run_eval.py's schema) and derives the previous schema's
    `original_value_set_f1`/`best_value_set_f1`/`gt_rows_covered`/`excess_result_rows` columns from
    its split `diagnose_*`/`relax_*` fields, so it can be compared against previous-results.csv on
    identically-named columns."""
    df = pd.read_csv(path)
    df['norm_csv'] = df['source_csv'].apply(normalize_path)
    df = df[df['gen_rows'] == 0]

    df['original_value_set_f1'] = df['diagnose_value_set_f1']
    # relax_value_set_f1/_rows_covered/_excess_rows are blank (-> NaN once read) whenever
    # relaxation wasn't attempted or found nothing better than the original — fall back to the
    # diagnose-stage value in that case, matching previous-results.csv's `best_*` fallback semantics.
    df['best_value_set_f1'] = df['relax_value_set_f1'].fillna(df['diagnose_value_set_f1'])
    df['gt_rows_covered'] = df['relax_rows_covered'].fillna(df['diagnose_rows_covered'])
    df['excess_result_rows'] = df['relax_excess_rows'].fillna(df['diagnose_excess_rows'])
    return df


def load_previous_zero_result_rows(path: str) -> pd.DataFrame:
    """Loads previous-results.csv (batch_ablation.py's schema) and filters to rows whose original
    generated query returned zero results."""
    df = pd.read_csv(path)
    df['norm_csv'] = df['source_csv'].apply(normalize_path)
    return df[df['result_row_count'] == 0]


current_df = load_current_zero_result_rows('current-results.csv')
previous_df = load_previous_zero_result_rows('previous-results.csv')

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
