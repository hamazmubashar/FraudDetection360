<div align="center">

# FraudDetection360

**Fraud scoring for 590,540 real card transactions, with the block/allow cutoff set by what each mistake actually costs.**

[![Live App](https://img.shields.io/badge/Live_App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://frauddetection360.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Model-2E8B57?style=for-the-badge)](https://lightgbm.readthedocs.io/)
[![MySQL](https://img.shields.io/badge/MySQL-Queries-4479A1?style=for-the-badge&logo=mysql&logoColor=white)](sql/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

[**Try the live app**](https://frauddetection360.streamlit.app) · [Findings](#what-the-data-showed) · [The model](#the-model) · [The cutoff](#choosing-the-cutoff) · [Run it](#run-it-locally)

</div>

---

## The problem

Every fraud decision can go wrong in two ways. Miss a fraudulent payment and the company loses the money. Block a genuine customer and it loses the sale, pays for a support call, and may lose the customer. A model that only maximises accuracy ignores that trade-off.

So this project doesn't stop at "fraud or not". It scores each transaction and then picks the cutoff that minimises the total cost of both kinds of mistake, with the cost assumptions visible and adjustable in the app.

**Data:** [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) on Kaggle. 590,540 real e-commerce transactions, about 3.5% fraud, 439 raw columns covering the transaction, card, device and 300+ anonymised behaviour signals.

## What the data showed

| Finding | Detail |
|---|---|
| Having identity data is itself a warning sign | Transactions with device/identity records were fraud 7.85% of the time versus 2.09% without. The payment processor seems to collect this data when something already looks risky. |
| The highest fraud rate isn't the biggest loss | Product type C has the highest fraud rate (11.7%), but type W, at 74% of volume, loses the most money. They need different responses. |
| Early morning is risky, and getting riskier | Fraud runs at 10–14% between 6 and 9 AM. The pattern held across separate time periods and grew over the six months, a clear sign of drift. |
| Credit is riskier than debit | 6.7% versus 2.4%, on large samples. |
| Mobile is the riskiest device | 10.2%, versus 6.5% on desktop and 2.1% where the device isn't recorded. |

Each notebook has a matching write-up in [`docs/`](docs/).

## The model

I compared Logistic Regression, Random Forest, SMOTE oversampling and Isolation Forest against **LightGBM**, which won on both ROC-AUC (0.90) and PR-AUC. It also handles missing values and categories natively, which matters when most identity columns are empty.

Training and testing use a **time-based split**: the model always predicts later transactions from earlier ones, the way it would in production. A random split would leak future patterns into training and flatter the scores.

## Choosing the cutoff

Instead of the default 0.5, the cutoff comes from costs:

- **Missed fraud:** $149.24, the average fraudulent transaction in the data
- **Blocked genuine customer:** $10, an assumption you can change in the app

With those numbers the best cutoff is **0.55**. It catches **71% of fraud** (2,895 of 4,064 cases in the test period), with 23% of alerts being real fraud. Total modelled cost is about **$273K**: 72% lower than a lenient 0.05 cutoff ($971K) and 32% lower than a strict 0.90 cutoff ($401K).

The right cutoff depends on that $10 assumption. Across $5 to $50 it moves between 0.40 and 0.85, which is why the app lets you adjust it instead of presenting one number as correct.

## Tech stack

| Area | Tools |
|---|---|
| Data and modelling | Python, pandas, scikit-learn, LightGBM, imbalanced-learn |
| SQL | MySQL, 10 business queries in [`sql/`](sql/) |
| BI | Power BI dashboard |
| App | Streamlit, Plotly |

## Run it locally

```bash
git clone https://github.com/hamazmubashar/FraudDetection360.git
cd FraudDetection360
conda create -n frauddetection python=3.11 -y
conda activate frauddetection
pip install -r requirements.txt
streamlit run app/app.py
```

The app runs from the saved model in `artifacts/`, so you don't need the raw data. To re-run the notebooks, download `train_transaction.csv` and `train_identity.csv` from the [competition page](https://www.kaggle.com/c/ieee-fraud-detection/data) into `datasets/raw/`.

## Project structure

```
FraudDetection360/
├── notebooks/       # 6 notebooks: data quality → EDA → features → models → imbalance → cutoff
├── docs/            # written findings for each notebook
├── sql/             # business queries
├── app/             # Streamlit app
├── artifacts/       # saved model, feature defaults and config
└── datasets/
    ├── raw/         # not committed; download from Kaggle
    └── processed/
```

## Limitations

This shows the full method end to end; it isn't a production system.

- The $10 and $149.24 costs are estimates, not figures from a real business.
- Fraud patterns drift (the morning spike grew over time), but there's no retraining pipeline yet.
- Many of the strongest features are anonymised columns pre-computed by the data provider; production would need to build equivalents in real time.
- Results come from one historical test period, not a live trial.

## Author

**Hamaz Mubashar** · [LinkedIn](https://www.linkedin.com/in/hamazmubashar) · [GitHub](https://github.com/hamazmubashar)

Released under the [MIT License](LICENSE).
