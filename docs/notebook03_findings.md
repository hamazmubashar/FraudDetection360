# Notebook 03: Feature Engineering — Findings

## M-Columns
- NaN replaced with explicit "not_applicable" category (not imputed), preserving
  the Notebook 01 finding that missingness here means "check doesn't apply to
  this ProductCD," not "value unknown."

## DeviceInfo Grouping
- 1,786 raw device strings grouped into 8 categories based on observed string
  patterns: samsung, ios, windows, macos, lg, browser_only_signature (rv:/Trident
  strings — browser engine signatures, not devices), other_android_brand, other.
- Verified iteratively: initial "other" bucket (9,414 rows) was inspected and
  found to contain an unaddressed LG pattern (2,331 rows) — added as its own
  category, shrinking "other" to a genuinely long-tail 7,016 rows (1.2% of data).

## Categorical Encoding
- card6: merged rare "charge card" (15 rows) → credit, "debit or credit" (30
  rows) → debit, per the small-sample-size reasoning from Notebook 02.
- 16 categorical columns (ProductCD, card4, card6, email domains, M1-M9,
  DeviceInfo_grouped, DeviceType) converted to pandas 'category' dtype rather
  than one-hot encoded, since LightGBM handles categorical splits natively —
  avoids exploding the dataset with hundreds of sparse dummy columns (e.g.
  P_emaildomain alone has 59 unique values).

## V-Column Redundancy Reduction
- Computed full pairwise correlation across all 339 V-columns.
- Tested three thresholds (0.90/0.95/0.99) before choosing — drop counts were
  181/124/36 respectively, showing threshold choice meaningfully affects
  feature count. Chose 0.95 as a defensible middle ground between over-pruning
  (risk of losing unique signal) and under-pruning (redundant, harder-to-explain
  features).
- Dropped 124 near-duplicate V-columns (correlation > 0.95 with another).

## D-Columns / Identity Columns
- Deliberately left untouched (still NaN where missing) — verified post-hoc
  that missingness percentages exactly match Notebook 01's original findings,
  confirming no accidental modification during other feature engineering steps.

## Leakage Check
- Confirmed no numeric feature has suspiciously high correlation with isFraud
  (highest is V257 at 0.38) — no evidence of data leakage.

## Final Dataset
- Shape: 590,540 rows × 315 columns (down from 439 after redundancy reduction)
- Saved as train_model_ready.parquet

## Next Steps (Notebook 04: Baseline Models)
- Time-based train/test split (not random) per our earlier reasoning.
- Train Logistic Regression, Random Forest, LightGBM baselines.
- Evaluate using precision/recall/ROC-AUC/PR-AUC — never accuracy.