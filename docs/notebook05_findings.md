# Notebook 05: Imbalance Handling & Anomaly Detection — Findings

## Approach A: Tuned Class Weighting
Tested scale_pos_weight at half (13.73), natural (27.46), and double (54.92) the
true imbalance ratio. Confirmed a clean precision/recall tradeoff (precision
0.31->0.20->0.13 as recall rises 0.62->0.75->0.82), while ROC-AUC/PR-AUC stayed
nearly flat (0.89-0.90 / 0.49-0.50) across all three — strong evidence this
parameter mainly shifts the decision threshold rather than improving the
model's underlying ranking ability.

## Approach B: SMOTE + LightGBM
Applied SMOTE at sampling_strategy=0.3 (conservative, not full 50/50 balance,
to avoid generating excessive synthetic data from only 16,599 real fraud
examples — ended up creating ~120K synthetic examples, more than the real
fraud count, a limitation worth flagging). Required converting LightGBM's
native categorical/NaN handling to numeric-encoded (-999 filled) format,
losing that advantage on both train and test.
Result: Precision 0.72, Recall 0.33, ROC-AUC 0.88, PR-AUC 0.46 — highest
precision seen, but PR-AUC is LOWER than the untouched baseline (0.50),
indicating SMOTE did not improve genuine ranking ability, only shifted the
threshold at a real pipeline-complexity cost.

## Approach C: Isolation Forest (unsupervised anomaly detection)
Tested training on (a) full training data (fraud+legit mixed) and (b)
legitimate-transactions-only. Legit-only performed marginally better
(Precision 0.108 vs 0.090, PR-AUC 0.094 vs 0.092), confirming the
theoretical expectation that isolation-based methods should learn from
"normal" data specifically.
Both versions performed dramatically worse than all supervised approaches
(ROC-AUC ~0.75 vs 0.88-0.90 for supervised models; PR-AUC ~0.09 vs 0.46-0.50).
Conclusion: unsupervised anomaly detection is disadvantaged here because we
have reliable fraud labels available for supervised learning — this approach
would be more valuable for detecting genuinely novel fraud patterns without
historical labels, not as a replacement for supervised modeling in this
context.

## Decision
None of the alternative techniques improved on simple class-weighted LightGBM
by PR-AUC. Carrying forward: LightGBM with scale_pos_weight = natural ratio
(~27.46), native categorical/NaN handling preserved, into Notebook 06 for
threshold optimization — the stage actually designed to solve the
precision/recall tradeoff these experiments confirmed is threshold-driven,
not resolvable through resampling or reweighting alone.

## Next Steps (Notebook 06)
- Systematic threshold search across the full 0-1 range.
- Define false-positive cost (customer friction) vs false-negative cost
  (fraud loss) to find a business-optimal threshold, not just a
  statistically "balanced" one.