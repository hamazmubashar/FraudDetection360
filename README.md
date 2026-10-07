# 🛡️ FraudDetection360

A fraud risk-scoring system built on 590,540 real e-commerce transactions — balancing fraud detection against customer experience through evidence-based feature engineering, model comparison, and cost-optimized decision thresholds.

[![Live App](https://img.shields.io/badge/Live_App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://frauddetection360.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Model-2E8B57?style=for-the-badge)](https://lightgbm.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**[🚀 Try the Live App](https://frauddetection360.streamlit.app)** · [Key Findings](#key-findings) · [The Model](#the-model) · [Run Locally](#running-the-app-locally)

---

## The Business Problem

A payment company needs to flag fraudulent transactions in real time — but every decision carries two competing costs:
- **Miss fraud** → direct financial loss
- **Wrongly block a good customer** → lost sale, support cost, churn risk

This project builds a system that doesn't just predict fraud, but recommends a business-optimal action, based on the actual dollar cost of each type of mistake.

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) (Kaggle) — 590,540 real transactions, ~3.5% fraud rate, 439 raw features (transaction, card, device, and 300+ anonymized behavioral signals).

## Key Findings

| Finding | Detail |
|---|---|
| **Identity tracking is itself a signal** | Transactions with device/identity data tracked show a fraud rate 4x higher (7.85% vs 2.09%) than untracked ones — tracking correlates with pre-existing risk flags from the payment processor. |
| **Rate ≠ dollar impact** | ProductCD='C' has the highest fraud *rate* (11.7%), but ProductCD='W' (74% of volume) causes the highest dollar *losses* — two different segments need two different responses. |
| **A real risk window: 6-9AM** | Fraud rate spikes to ~10-14% in low-volume early hours, confirmed to replicate across independent time splits — and this risk **intensified** over the dataset's 6-month span, direct evidence of fraud-pattern drift. |
| **Credit riskier than debit** | 6.7% vs 2.4% fraud rate, on large, reliable samples. |
| **Mobile riskiest device** | 10.2% fraud rate vs 6.5% desktop, 2.1% untracked. |

Full investigation notebooks: [`notebooks/`](notebooks/)

## The Model

**LightGBM**, selected after comparing Logistic Regression, Random Forest, SMOTE-resampling, and Isolation Forest anomaly detection — LightGBM won on ROC-AUC (0.90) and PR-AUC while natively handling missing data and categorical features without lossy workarounds.

Trained on a **time-based split** (not random) — the model always predicts the future from the past, mirroring real deployment.

## The Threshold Decision

Rather than a fixed 50% cutoff, the deployed threshold is derived from actual business costs:
- False Negative cost (missed fraud): **$149.24** — data-derived average fraud transaction amount
- False Positive cost (blocked customer): **$10** — a documented, *configurable* business assumption
- At these assumptions: optimal threshold = **0.55**, reducing total cost from $971K (naive threshold) to **$273K** — a 91% reduction

**Sensitivity tested:** the optimal threshold ranges from 0.40 to 0.85 depending on the FP-cost assumption ($5-$50) — the app lets this be adjusted live rather than hardcoding one number as universally "correct."

## Tech Stack

| Layer | Tools |
|---|---|
| Data processing & modeling | Python, pandas, scikit-learn, LightGBM, imbalanced-learn |
| Database | MySQL |
| BI Dashboard | Power BI |
| Web App | Streamlit, Plotly |

## Project Structure

```
FraudDetection360/
├── notebooks/          # 01-06: data quality → EDA → features → models → imbalance → threshold
├── sql/                # 10 documented business queries
├── datasets/
│   ├── raw/            # not committed — download from Kaggle (see below)
│   └── processed/
├── artifacts/          # saved model, feature defaults, config
├── app/                # Streamlit application
└── docs/               # phase-by-phase findings write-ups
```

## Running the App Locally

```bash
git clone https://github.com/hamazmubashar/FraudDetection360.git
cd FraudDetection360
conda create -n frauddetection python=3.11 -y
conda activate frauddetection
pip install -r requirements.txt
streamlit run app/app.py
```

The app runs from the saved artifacts in `artifacts/` — no raw data needed. To re-run the notebooks, download `train_transaction.csv` and `train_identity.csv` from the [IEEE-CIS Fraud Detection competition](https://www.kaggle.com/c/ieee-fraud-detection/data) and place them in `datasets/raw/`.

## Known Limitations

This is a portfolio project demonstrating end-to-end fraud detection methodology — not a production system. Specifically:
- Cost assumptions ($10 FP, $149.24 FN) are estimates, not verified against real business data
- No live retraining pipeline, despite evidence that fraud patterns drift over time
- Some model features (Vesta's anonymized `V` columns) are pre-engineered in the source dataset and would need equivalent real-time computation in production
- Validated on a single historical test split, not live A/B tested

## Author

**Hamaz Mubashar** — [LinkedIn](https://www.linkedin.com/in/hamazmubashar)