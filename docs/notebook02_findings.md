# Notebook 02: Exploratory Data Analysis — Findings

## Transaction Amount
- Mean amount: $134.51 (legit) vs $149.24 (fraud) — modest, not dramatic difference.
- Quartile comparison shows fraud is NOT simply "bigger" or "smaller": fraud sits
  slightly lower at the 25th percentile but higher at the median/75th percentile
  than legit transactions — distributions overlap heavily.
- Outlier check (95th/99th/99.9th percentile): legitimate transactions have a much
  longer, more extreme tail (max $31,937 vs fraud's max $5,191) — consistent with
  rare large legitimate purchases (e.g. big-ticket items) that fraudsters are
  unlikely to risk on a single stolen card.
- Conclusion: TransactionAmt alone is a weak-to-moderate standalone signal.

## Time Patterns (Hour of Day)
- Derived hour_of_day and day_of_week from TransactionDT (seconds-based, no real
  calendar date/timezone attached — day_of_week labels are cyclical only, not
  confirmed to map to actual weekdays).
- Strong finding: fraud rate spikes to 10.6% at hour 7 (vs 3.5% baseline), during
  low-transaction-volume hours (6-9), then drops to ~2.3-2.9% during high-volume
  afternoon hours.
- Verified NOT a fluke: split data into first/second half by time and confirmed
  the low-volume/high-fraud pattern replicates independently in both halves.
- Bonus finding: fraud rate at hour 7 nearly doubled between the first half
  (7.6%) and second half (14.5%) of the 182-day period — direct evidence of
  fraud pattern drift within just 6 months, reinforcing why a time-based
  train/test split and periodic model retraining are necessary (not just a
  textbook assumption — observed directly in this data).

## ProductCD / Card Type
- ProductCD=C has the highest fraud rate (11.7%, large sample of 68,519 rows) —
  more than 3x the overall average. ProductCD=W (74% of all data) has the lowest
  (2.0%).
- Notable limitation: ProductCD=C is simultaneously the highest-fraud AND
  lowest-feature-coverage segment (M1/M2/M7 are ~100% missing for C), meaning
  our model will likely have the least information precisely where fraud is
  most concentrated — a genuine data limitation, not a modeling failure.
- card6: credit cards (6.7% fraud) show meaningfully higher fraud than debit
  (2.4%) — both on large samples, so trustworthy.
- card4: discover shows highest fraud rate (7.7%) but smaller sample (6,651) —
  flagged as probably real but less certain than the ProductCD/card6 findings.

## Anonymized V-Columns (V1-V339)
- Individual correlation with isFraud: top columns (V257, V246, V244, V242, etc.)
  show correlations up to 0.38 — notably strong for this dataset (has_identity,
  our earlier strong finding, only correlates ~0.1-0.15).
- Redundancy check: top correlated V-columns are highly correlated WITH EACH
  OTHER (up to 0.97), confirming many are near-duplicate signals rather than
  15 independent strong features. Feature engineering should account for this
  (e.g. correlation-based pruning or PCA on the V-block) rather than feeding
  all raw duplicates into the model.
- V-columns do NOT correlate with TransactionAmt (all near zero) — confirms
  they carry genuinely independent information, not a disguised amount proxy.

## Missingness Structure (dataset-wide)
- 435 columns collapse into only 70 distinct missingness "fingerprints" —
  confirms missingness is driven by a small number of root causes (has_identity,
  ProductCD, etc.) rather than being column-by-column random.
- 43 columns are never missing at all (core transaction fields); the largest
  shared-pattern group covers 46 columns at once.
- Implication for Notebook 03: organize NaN-handling strategy around these
  root-cause groups, not column-by-column.

## Next Steps (Notebook 03: Feature Engineering)
- Encode hour_of_day / time-drift feature; consider a "low-volume hour" flag.
- Group/prune redundant V-columns; consider correlation threshold or PCA.
- Apply NaN-as-category strategy for M-columns and DeviceInfo grouping
  (from Notebook 01), organized by the 70 missingness-fingerprint groups.
- Document ProductCD=C's high-fraud/low-coverage tradeoff as a known limitation.