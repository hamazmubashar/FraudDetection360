# Notebook 04: Baseline Models & Time-Based Split — Findings

## Train/Test Split
- Time-based split (NOT random): first 80% of transactions (chronologically) as
  train, last 20% as test — simulates real deployment (train on past, predict
  future) and avoids data leakage from random shuffling.
- Train: 472,432 rows (3.51% fraud) | Test: 118,108 rows (3.44% fraud)
- Overall fraud rate stayed stable between train/test despite earlier finding
  that fraud concentration shifts within specific hours over time — these are
  different phenomena (daily time-of-day pattern vs. total period average).

## Feature/Target Prep
- Dropped TransactionID (arbitrary sequential ID, no real relationship to fraud)
  and raw TransactionDT (an ever-increasing counter specific to this dataset's
  182-day window — would not generalize to future data with larger DT values).
  Kept derived hour_of_day/day_of_week instead, since these are genuinely
  repeating/cyclical signals that generalize into the future.

## Baseline 1: Logistic Regression
- Required numeric, non-null input — categoricals encoded via .cat.codes,
  NaN filled with -999 placeholder (crude, loses per-column NaN meaning
  proven important in Notebooks 01-03), features scaled via StandardScaler.
- Bug encountered: initial run missed 16 object-dtype columns (id_12, id_15...,
  DeviceInfo) not caught by the predefined categorical list — fixed by
  dynamically detecting all object/category columns instead of hardcoding.
- Result: Precision 0.12, Recall 0.70, ROC-AUC 0.82, PR-AUC 0.17
- Confusion matrix: 21,903 false positives (customers wrongly flagged) vs
  2,863 true positives caught — heavily skewed toward false alarms.
- Notably: model accuracy (0.80) is LOWER than a naive "always predict not
  fraud" baseline (0.965) despite doing real, useful fraud-catching work —
  concrete proof of why accuracy is the wrong metric for this problem.

## Baseline 2: Random Forest
- Used same encoded features as Logistic Regression (no scaling needed —
  tree-based models split on individual feature thresholds, unaffected by scale).
- Result: Precision 0.87, Recall 0.26, ROC-AUC 0.89, PR-AUC 0.53
- Opposite tradeoff from Logistic Regression: very few false alarms (153) but
  misses most fraud (3,024 false negatives) at default 0.5 threshold.

## Baseline 3: LightGBM
- Used native category dtypes and real NaN values (no -999 workaround,
  no manual encoding) — the entire point of choosing this model.
- Bug encountered: same 16 leftover object columns caused a dtype error;
  fixed by converting them to category dtype directly (not numeric codes,
  preserving native handling).
- Second issue: training warned about "too many bins" for categorical
  features — traced to the original (ungrouped) DeviceInfo column (1,639
  unique values) still being present alongside our Notebook 03
  DeviceInfo_grouped feature. Dropped the redundant original column,
  which resolved the warning.
- Result: Precision 0.21, Recall 0.74, ROC-AUC 0.90, PR-AUC 0.50 — best
  ROC-AUC and recall of the three baselines, better precision than Logistic
  Regression at a similar recall level.

## Model Comparison Summary
| Model | Precision | Recall | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.12 | 0.70 | 0.82 | 0.17 |
| Random Forest | 0.87 | 0.26 | 0.89 | 0.53 |
| LightGBM | 0.21 | 0.74 | 0.90 | 0.50 |

## Decision
LightGBM selected to carry forward into Notebook 05 (imbalance handling) and
Notebook 06 (threshold optimization) — best overall ranking ability (ROC-AUC,
PR-AUC close to Random Forest's), native handling of NaN/categoricals
consistent with our Notebook 01-03 investigation, and a more tunable
default precision/recall balance than Random Forest's extreme conservatism.

## Next Steps (Notebook 05)
- Apply SMOTE and/or class-weight variations to LightGBM specifically.
- Compare against Isolation Forest / autoencoder anomaly-detection approaches.
- Carry forward best-performing variant into Notebook 06 for threshold tuning.