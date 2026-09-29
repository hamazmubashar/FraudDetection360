# Notebook 01: Data Loading & Quality Investigation — Findings

## Dataset
- IEEE-CIS Fraud Detection (Kaggle)
- train_transaction: 590,540 rows × 394 columns
- train_identity: 144,233 rows × 41 columns
- Merged (left join on TransactionID): 590,540 rows × 435 columns (no rows lost)

## Target Variable
- isFraud: 3.50% fraud (20,663 / 590,540) — confirms severe class imbalance
- A naive "always predict not-fraud" model would score 96.5% accuracy while
  catching zero fraud — this is why accuracy is unusable as our success metric.

## Merge Strategy: Identity Data
- Only 24.4% of transactions have matching identity records (144,233 / 590,540)
- Decision: LEFT JOIN (keep all transactions) + new `has_identity` flag,
  instead of dropping unmatched rows or imputing mean/mode values.
- Evidence this was the right call: fraud rate is 2.09% for transactions
  WITHOUT identity data vs 7.85% WITH identity data — a ~4x difference across
  large sample sizes (446K vs 144K rows), proving missingness itself is a
  meaningful signal, not noise to be filled in.

## Missingness Investigation (evidence-based, not assumed)
- 214 of 435 columns are missing >50% of values; 12 columns >90% missing.
- id_ columns (id_01–id_38): missingness is tied 1:1 to `has_identity`
  (verified: 0 mismatches between id_01 nullity and has_identity).
- D columns (D1–D15): correlation matrix shows they are NOT one uniform
  family — at least 2-3 distinct clusters (e.g. D4/D6/D12 correlate at
  0.96-1.00; D3/D5/D7 correlate at 0.71-0.99), plus D9 which is uncorrelated
  with every other D column (max correlation 0.07) and has a completely
  different value range (0-0.96 vs hundreds for the others).
  → Decision: treat each D column as an independent feature; do not apply
    one uniform imputation/engineering rule across all of them.
- M columns (M1-M9): missingness is driven by ProductCD, not random.
  M1/M2/M7 are ~100% missing for ProductCD in {C,H,R,S} and only partially
  missing for ProductCD=W; M4 instead depends on ProductCD=C.
  → Conclusion: NaN in M-columns means "not applicable to this transaction
    type," not "value unknown." Plan to encode as an explicit category
    rather than impute.

## Duplicates
- 0 duplicate rows, 0 duplicate TransactionIDs — merge is clean.

## Categorical Columns
- 31 object-dtype columns identified and fully accounted for.
- card6 has two rare, likely-erroneous categories: "debit or credit" (30
  rows) and "charge card" (15 rows). Both show 0% fraud rate, but sample
  size is too small (~1 expected fraud case at baseline rate) to treat this
  as a meaningful signal — decision is to merge into nearest existing
  category based on data-entry-error judgment, not statistical evidence.

## DeviceType / DeviceInfo
- DeviceType missing for 449,730 rows; confirmed dependent on has_identity
  (0 rows have DeviceType without any identity data), but not a perfect
  1:1 match — 3,423 rows have identity data but no DeviceType, showing
  identity fields aren't captured as one indivisible block.
- DeviceInfo has 1,786 unique raw values, heavily long-tailed (device model
  strings) — will need grouping/simplification before modeling.

## Next Steps (Notebook 02)
- Deeper EDA: distribution of TransactionAmt, time-based patterns via
  TransactionDT, fraud rate breakdowns by ProductCD/card4/card6.
- Begin feature engineering plan: NaN-as-category strategy for M-columns,
  DeviceInfo grouping, decision on which D-columns to keep as-is vs
  transform.