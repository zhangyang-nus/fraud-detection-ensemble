# Multi-Pipeline Ensemble Fraud Detection

A card-not-present (CNP) fraud detection system built on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset (410K+ transactions, 3.5% fraud rate). Combines four complementary pipelines through a stacked decision layer to help a card issuer flag high-risk transactions for step-up authentication or manual review — without over-flagging legitimate customers.

**Result:** 36.45% fraud recall at a 1.04% false-positive rate (0.892 ROC-AUC), exceeding the project's 25%-recall target.

## Background

This was originally a 4-person team project for an NUS data science module, built and submitted as a KNIME (visual/no-code) workflow. I was technical lead: I scoped the overall system architecture, built the shared data cleaning and feature engineering pipeline, personally implemented the Isolation Forest pipeline, and designed the final stacking decision layer. Three teammates each implemented one of the other pipelines under that architecture.

**`fraud_pipeline_202607.py`** in this repo is my own post-submission reimplementation of the full workflow in Python/scikit-learn/XGBoost, written to understand the mechanics underneath the visual KNIME nodes. It mirrors the original pipeline design and is commented to explain the reasoning behind each step, not just the code.

## System Design

```
Raw transactions
      │
      ▼
Chronological split (70% train / 15% val / 15% test by TransactionDT)
      │
      ▼
Shared feature engineering
  • missing-value handling (explicit missingness flags, not blind imputation)
  • card–address / card–email relationship features (frequency, novelty, amount deviation)
      │
      ├──► Pipeline 1: Logistic Regression   (M1–M9 match-pattern features)
      ├──► Pipeline 2: XGBoost                (card–address/email relationship features)
      ├──► Pipeline 3: Rule-based scorecard   (transparent, auditable points score)
      └──► Pipeline 4: Isolation Forest       (unsupervised anomaly detection)
                    │
                    ▼
      Stacked Logistic Regression decision layer
                    │
                    ▼
      Final fraud probability → flag for review / step-up auth
```

**Why chronological splitting?** A random split lets the model "see the future" during training and inflates validation scores. Splitting by `TransactionDT` tests the realistic question: trained on the past, how well does it generalize to transactions that happen later?

**Why an ensemble instead of one model?** Each pipeline captures a different kind of signal — XGBoost supplies the core predictive power from engineered relationship features, Isolation Forest catches unusual transactions the supervised models miss, and the rule-based scorecard keeps the system auditable for the issuer even where the ML components are black boxes.

## Results

| Metric | Value |
|---|---|
| Fraud recall | 36.45% |
| False-positive rate | 1.04% |
| Precision | 55.59% |
| ROC-AUC | 0.892 |
| PR-AUC | 0.425 |

A leave-one-out ablation study (removing each pipeline in turn and re-scoring) showed XGBoost as the main predictive driver, Isolation Forest as the strongest contributor to false-positive control, the Logistic Regression pipeline adding modest incremental recall, and the rule-based scorecard adding negligible predictive lift but retained for explainability.

## Usage

```bash
python fraud_pipeline.py --data path/to/ieee_cis_joined_full.csv --target-recall 0.80
```

`--target-recall` controls the probability threshold selected on the validation set (default 0.80); lower it to trade recall for a lower false-positive rate.

**Note:** the raw dataset is not included here due to Kaggle's redistribution terms — download it directly from the [competition page](https://www.kaggle.com/c/ieee-fraud-detection).

## Tech Stack

Python, scikit-learn, XGBoost, pandas, NumPy · (original team submission: KNIME, H2O)
