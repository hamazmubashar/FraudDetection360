# Notebook 06: Threshold Optimization & Cost Analysis — Findings

## Cost Assumptions
- False Negative (missed fraud) cost: $149.24 — data-derived (average fraud
  TransactionAmt from Notebook 02), not an assumption.
- False Positive (wrongly blocked customer) cost: $10 — a documented business
  assumption (customer friction/goodwill cost), not derivable from this dataset.

## Threshold Search
- Tested thresholds from 0.05 to 0.90 in 0.05 increments, calculating total
  dollar cost = (false positives x $10) + (false negatives x $149.24) at each.
- Cost curve forms a clean U-shape, confirming neither extreme (too permissive
  or too strict) is optimal.
- Optimal threshold under $10 FP cost assumption: 0.55 — total cost $273,042,
  vs $971,423 at threshold 0.05 (a 72% cost reduction from a poorly-chosen
  threshold to the optimized one).
- At threshold 0.55: 9,858 false positives, 1,169 false negatives, 2,895 fraud
  cases correctly caught.

## Sensitivity Analysis
Tested FP cost assumptions from $5 to $50 to check how fragile the "optimal"
threshold is to our $10 guess:
| FP Cost | Optimal Threshold | Total Cost |
|---|---|---|
| $5 | 0.40 | $202,184 |
| $10 | 0.55 | $273,042 |
| $20 | 0.70 | $340,951 |
| $30 | 0.75 | $379,065 |
| $50 | 0.85 | $411,350 |

Finding: the optimal threshold is NOT robust to the FP cost assumption — it
ranges from 0.40 to 0.85 depending on that single input. This is an honest
limitation, not a flaw in the method: the direction of the relationship
(higher FP cost -> stricter threshold) is logically consistent, confirming
the model and framework behave correctly even though the exact "best" number
depends on business data we don't have (actual customer churn/support cost).

## Recommendation
Default deployed threshold: 0.55 (based on $10 FP cost assumption), but
implemented as a configurable parameter rather than a hardcoded constant —
in real deployment, this should be tuned using the business's actual
false-positive cost data, not fixed permanently based on this dataset's
assumption.

## Final Model
- LightGBM, scale_pos_weight=27.46 (natural class imbalance ratio)
- Native categorical/NaN handling preserved throughout
- Saved as final_fraud_model.pkl, with configuration in final_model_config.json

## Next Steps
- SQL layer: business queries on cleaned/engineered data
- Power BI dashboard: fraud trends, cost-saved-by-model visualization
- Streamlit app: real-time transaction scoring using final_fraud_model.pkl
  and the configurable threshold