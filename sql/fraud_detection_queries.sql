-- =============================================================
-- FraudDetection360 — Business Intelligence SQL Queries
-- =============================================================
-- Purpose: Analyze transaction-level fraud patterns using the
-- cleaned/engineered dataset from Notebooks 01-05 (Python/LightGBM
-- pipeline). These queries validate and extend the EDA findings
-- from Notebook 02, expressed in SQL for business reporting and
-- to demonstrate SQL proficiency independent of the Python work.
-- =============================================================

-- -------------------------------------------------------------
-- SETUP: Create and select the working database
-- -------------------------------------------------------------
CREATE DATABASE IF NOT EXISTS frauddetection360;
USE frauddetection360;

-- -------------------------------------------------------------
-- QUERY 1: Overall Fraud Rate
-- -------------------------------------------------------------
-- Business question: "What percentage of all transactions in our
-- dataset are fraudulent?"
-- This is our baseline reference number — every other query's
-- fraud rate should be compared against this overall average to
-- judge whether a segment is higher- or lower-risk than typical.
-- -------------------------------------------------------------
SELECT 
    COUNT(*) AS total_transactions,
    SUM(isFraud) AS total_fraud,
    ROUND(SUM(isFraud) / COUNT(*) * 100, 2) AS fraud_rate_pct
FROM transactions;

-- -------------------------------------------------------------
-- QUERY 2: Fraud Rate by Product Category (ProductCD)
-- -------------------------------------------------------------
-- Business question: "Which product categories are riskiest?"
-- Segments transactions by ProductCD and ranks them by fraud
-- rate (highest first), so the business can see at a glance
-- which product lines need the most scrutiny.
-- Finding (validated against Python EDA in Notebook 02):
-- ProductCD = 'C' is the highest-risk category at ~11.7% fraud,
-- more than 3x the overall average, while 'W' (74% of all
-- transactions) is the lowest at ~2.0%.
-- -------------------------------------------------------------
SELECT 
    ProductCD,
    COUNT(*) AS total_transactions,
    SUM(isFraud) AS fraud_count,
    ROUND(SUM(isFraud) / COUNT(*) * 100, 2) AS fraud_rate_pct
FROM transactions
GROUP BY ProductCD
ORDER BY fraud_rate_pct DESC;

-- -------------------------------------------------------------
-- UTILITY: Confirm time-based columns exist before querying them
-- -------------------------------------------------------------
-- hour_of_day / hour_bucket were engineered in the Python pipeline
-- (derived from TransactionDT) and carried into this table.
-- -------------------------------------------------------------
SHOW COLUMNS FROM transactions LIKE 'hour%';

-- -------------------------------------------------------------
-- QUERY 3: Fraud Rate by Hour of Day
-- -------------------------------------------------------------
-- Business question: "Does fraud risk change depending on the
-- time of day a transaction occurs?"
-- Segments transactions into 24 hourly buckets and calculates
-- the fraud rate for each — useful for staffing fraud review
-- teams or setting time-based risk rules.
-- Finding (validated against Python EDA in Notebook 02): fraud
-- rate spikes to ~10.6% around hour 7, during otherwise
-- low-transaction-volume hours, then drops to ~2.3-2.9% during
-- high-volume afternoon hours — a real, replicated pattern, not
-- a statistical fluke (confirmed via time-split testing in Python).
-- -------------------------------------------------------------
SELECT 
    hour_bucket,
    COUNT(*) AS total_transactions,
    SUM(isFraud) AS fraud_count,
    ROUND(SUM(isFraud) / COUNT(*) * 100, 2) AS fraud_rate_pct
FROM transactions
GROUP BY hour_bucket
ORDER BY hour_bucket;

-- -------------------------------------------------------------
-- QUERY 4: Fraud Dollar Impact by Product Category
-- -------------------------------------------------------------
-- Business question: "Which product category costs us the most
-- in actual fraud losses, not just fraud rate?"
-- This reframes Query 2's percentages into real dollar terms —
-- a category can have a high fraud RATE but low fraud COST if its
-- transaction amounts are small, and vice versa. This is the more
-- financially actionable version of the same question.
-- -------------------------------------------------------------
SELECT 
    ProductCD,
    COUNT(*) AS total_transactions,
    SUM(isFraud) AS fraud_count,
    ROUND(SUM(CASE WHEN isFraud = 1 THEN TransactionAmt ELSE 0 END), 2) AS total_fraud_dollar_amount,
    ROUND(AVG(CASE WHEN isFraud = 1 THEN TransactionAmt END), 2) AS avg_fraud_transaction_amount
FROM transactions
GROUP BY ProductCD
ORDER BY total_fraud_dollar_amount DESC;

-- -------------------------------------------------------------
-- QUERY 5: Fraud Rate by Card Type (Debit vs Credit)
-- -------------------------------------------------------------
-- Business question: "Are credit or debit transactions riskier?"
-- Finding (validated against Python EDA): credit cards show
-- notably higher fraud (~6.7%) than debit (~2.4%) on large,
-- trustworthy sample sizes.
-- -------------------------------------------------------------
SELECT 
    card6 AS card_type,
    COUNT(*) AS total_transactions,
    SUM(isFraud) AS fraud_count,
    ROUND(SUM(isFraud) / COUNT(*) * 100, 2) AS fraud_rate_pct
FROM transactions
WHERE card6 IN ('debit', 'credit')
GROUP BY card6
ORDER BY fraud_rate_pct DESC;

-- -------------------------------------------------------------
-- QUERY 6: Top 10 Highest-Value Fraud Transactions
-- -------------------------------------------------------------
-- Business question: "What were our biggest individual fraud
-- losses, and what do they have in common?"
-- Useful for manual case review and spotting patterns among
-- the most costly fraud incidents specifically.
-- -------------------------------------------------------------
SELECT 
    TransactionID,
    TransactionAmt,
    ProductCD,
    card4 AS card_network,
    card6 AS card_type,
    hour_bucket
FROM transactions
WHERE isFraud = 1
ORDER BY TransactionAmt DESC
LIMIT 10;

-- -------------------------------------------------------------
-- QUERY 7: Fraud Rate by Device Type
-- -------------------------------------------------------------
-- Business question: "Does the device used affect fraud risk?"
-- Cross-references DeviceType against fraud outcome. Note:
-- ~76% of transactions have NULL DeviceType (no device
-- fingerprinting captured) — see Notebook 01 findings for why
-- this missingness itself is a meaningful fraud signal.
-- -------------------------------------------------------------
SELECT 
    COALESCE(DeviceType, 'not_tracked') AS device_type,
    COUNT(*) AS total_transactions,
    SUM(isFraud) AS fraud_count,
    ROUND(SUM(isFraud) / COUNT(*) * 100, 2) AS fraud_rate_pct
FROM transactions
GROUP BY DeviceType
ORDER BY fraud_rate_pct DESC;

-- -------------------------------------------------------------
-- QUERY 8: Fraud Rate Drift — First Half vs Second Half of Data
-- -------------------------------------------------------------
-- Business question: "Is fraud behavior stable over time, or
-- does it drift — and if so, how much?"
-- Splits the ~182-day dataset into two halves by TransactionDT
-- and compares fraud rate at hour 7 specifically (our highest-
-- risk hour) across both halves.
-- Finding (validated against Python EDA in Notebook 02): fraud
-- rate at hour 7 nearly doubled between the first half (~7.6%)
-- and second half (~14.5%) of the period — direct evidence of
-- fraud pattern drift within just 6 months, motivating the need
-- for periodic model retraining in production.
-- -------------------------------------------------------------
SELECT 
    CASE 
        WHEN TransactionDT < (SELECT (MIN(TransactionDT) + MAX(TransactionDT)) / 2 FROM transactions)
        THEN 'first_half' 
        ELSE 'second_half' 
    END AS time_period,
    hour_bucket,
    COUNT(*) AS total_transactions,
    SUM(isFraud) AS fraud_count,
    ROUND(SUM(isFraud) / COUNT(*) * 100, 2) AS fraud_rate_pct
FROM transactions
WHERE hour_bucket IN (6, 7, 8, 9, 13, 14)
GROUP BY time_period, hour_bucket
ORDER BY hour_bucket, time_period;

-- -------------------------------------------------------------
-- QUERY 9: Executive Summary — Overall Fraud Metrics
-- -------------------------------------------------------------
-- Business question: "Give me the full picture in one glance."
-- Combines volume, fraud count, fraud rate, and dollar exposure
-- into a single summary row — the kind of top-line number a
-- business stakeholder or dashboard would want first.
-- -------------------------------------------------------------
SELECT 
    COUNT(*) AS total_transactions,
    SUM(isFraud) AS total_fraud_transactions,
    ROUND(SUM(isFraud) / COUNT(*) * 100, 2) AS overall_fraud_rate_pct,
    ROUND(SUM(TransactionAmt), 2) AS total_transaction_volume_dollars,
    ROUND(SUM(CASE WHEN isFraud = 1 THEN TransactionAmt ELSE 0 END), 2) AS total_fraud_exposure_dollars,
    ROUND(SUM(CASE WHEN isFraud = 1 THEN TransactionAmt ELSE 0 END) / SUM(TransactionAmt) * 100, 2) AS fraud_pct_of_total_dollar_volume
FROM transactions;

-- -------------------------------------------------------------
-- QUERY 10: Power BI Export -- Row-Level Dashboard Dataset
-- -------------------------------------------------------------
-- Business question: "Give me one clean, row-level export with
-- everything the dashboard needs, so Power BI can build its own
-- aggregations (by ProductCD, hour, card type, device type) rather
-- than us pre-aggregating every possible cut in SQL."
-- Purpose: this is the export used to bring FraudDetection360 data
-- into Power BI for the executive dashboard (KPIs, fraud-by-segment
-- charts, time-based risk pattern, model impact summary).
-- -------------------------------------------------------------
SELECT 
    TransactionID,
    isFraud,
    TransactionAmt,
    ProductCD,
    card4 AS card_network,
    card6 AS card_type,
    DeviceType,
    hour_bucket,
    CASE 
        WHEN TransactionDT < (SELECT (MIN(TransactionDT) + MAX(TransactionDT)) / 2 FROM transactions)
        THEN 'first_half' 
        ELSE 'second_half' 
    END AS time_period
FROM transactions;