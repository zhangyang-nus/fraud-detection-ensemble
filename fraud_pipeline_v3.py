"""
Fraud Detection: Multi-Pipeline Ensemble
=========================================
Python/scikit-learn equivalent of the KNIME workflow.

This is written to be READ, not just run. Each section explains the ML concept
behind it, because the point is to learn the basics.

Structure (mirrors the KNIME workflow):
    1. Load data
    2. Chronological split (train / validation / test)
    3. Missing value treatment
    4. Feature engineering
    5. Four independent pipelines:
         a. Logistic Regression
         b. XGBoost
         c. Transparent rule-based scorecard
         d. Isolation Forest (unsupervised)
    6. Decision layer: stacked meta-model combining all four
    7. Evaluation: ROC-AUC, PR-AUC, confusion matrix, feature importance
    8. Ablation study: marginal contribution of each pipeline

Run:  python fraud_pipeline_v3.py --data path/to/ieee_cis_joined_full.csv
      python fraud_pipeline_v3.py --data data.csv --max-fpr 0.01

CHANGES IN THIS VERSION (v3)
----------------------------
5. FAIR OPERATING-POINT COMPARISON. Every pipeline now selects its own
   threshold on the same validation half under the same FPR cap, so
   recall/precision/FPR are comparable across models. Previously the ensemble
   got a business-chosen threshold while the individual pipelines got an
   arbitrary 0.5 — any apparent ensemble advantage could have been the
   threshold rather than the model. Results print as a head-to-head table.
6. Validation operating point is printed, evidencing that the threshold really
   satisfies the constraint on the data it was chosen on.
7. Drift section reports validation FPR vs. realised test FPR. The cap is a
   constraint satisfied on validation, not a promise about future data; the gap
   is the argument for periodic recalibration.
8. Ablation now names the case where removing a pipeline IMPROVES the ensemble
   — meaning that pipeline dilutes the signal and should be dropped.

CHANGES IN v2
-------------
1. Scorecard's "unusually large amount" threshold is now fitted on train and
   frozen, instead of being recomputed from whatever set it scored. The old
   behaviour meant a test transaction's score depended on the other test
   transactions around it — deployable systems can't see the future batch.
2. Unseen-ProductCD fallback now uses the train global 90th percentile rather
   than the current dataframe's. (Likely never triggered on IEEE-CIS, but the
   channel existed; it now also prints a note if it ever fires.)
3. Operating threshold is chosen by capping FPR (default 5%) and maximising
   recall underneath it, rather than demanding 80% recall and accepting
   whatever FPR resulted (~22%, unusable for a card issuer).
4. Validation is split chronologically in two: first half trains the
   meta-model, second half selects the threshold. Previously one set did both,
   so the threshold was tuned on in-sample meta-model predictions.

Note on scope: none of the above was label leakage — isFraud was never used to
build features, and test was never fitted on. (2), (3) and (5) are about
deployability, objective choice and fair comparison; (1) and (4) are genuine,
if narrow, distribution/in-sample issues.
"""

import argparse
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

warnings.filterwarnings("ignore")
RANDOM_STATE = 42


# =============================================================================
# 1. LOAD DATA
# =============================================================================
def load_data(path):
    """
    KNIME equivalent: CSV Reader node.

    We sort by TransactionDT because this dataset is a time series of
    transactions. Order matters for how we split later.
    """
    df = pd.read_csv(path)
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    print(f"Fraud rate: {df['isFraud'].mean():.2%}")
    return df


# =============================================================================
# 2. CHRONOLOGICAL SPLIT
# =============================================================================
def chronological_split(df, train_frac=0.70):
    """
    KNIME equivalent: Table Partitioner nodes (FIRST_ROWS mode).

    CONCEPT — Why not a random split?
    ---------------------------------
    With a random split, a transaction from March could land in training while
    a transaction from February lands in test. The model would be "seeing the
    future" during training, which inflates your scores and gives you a model
    that looks great in testing and fails in production.

    Fraud patterns also drift over time (fraudsters adapt). A chronological
    split tests the realistic question: "trained on the past, how does it do
    on the future?"

    We produce three sets:
      - train (70%)  : fit the models
      - val   (15%)  : fit the decision layer / tune thresholds
      - test  (15%)  : final honest evaluation, touched ONCE at the very end
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * (1 - train_frac) / 2)

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    print(f"\nSplit: train={len(train):,}  val={len(val):,}  test={len(test):,}")
    for name, part in [("train", train), ("val", val), ("test", test)]:
        print(f"  {name} fraud rate: {part['isFraud'].mean():.2%}")
    return train, val, test


# =============================================================================
# 3. MISSING VALUE TREATMENT
# =============================================================================
class MissingValueHandler:
    """
    KNIME equivalent: Missing Value nodes.

    CONCEPT — Fit on train, apply everywhere (this is THE core ML discipline)
    ------------------------------------------------------------------------
    Notice `fit()` computes medians/modes and stores them, then `transform()`
    applies those STORED values to any dataset. If you instead computed the
    median of the test set and used it on the test set, information from test
    leaks into your evaluation. This is called DATA LEAKAGE and it is the most
    common way beginners accidentally fake good results.

    Every scikit-learn transformer follows this fit/transform pattern for
    exactly this reason.

    CONCEPT — Missingness can itself be a signal
    --------------------------------------------
    In this dataset, whether a field is missing is often more predictive than
    its value. A transaction with no recipient email is a structurally
    different kind of transaction. So instead of only filling gaps, we also
    create explicit "was this missing?" flag columns.
    """

    def __init__(self):
        self.medians_ = {}
        self.modes_ = {}

    def fit(self, df):
        for col in ["D4", "V34", "dist1"]:
            if col in df.columns:
                self.medians_[col] = df[col].median()
        for col in ["card4", "card6"]:
            if col in df.columns:
                mode = df[col].mode()
                self.modes_[col] = mode.iloc[0] if len(mode) else "unknown"
        return self

    def transform(self, df):
        df = df.copy()

        # --- Availability flags: create BEFORE filling, or the signal is lost
        flag_cols = {
            "has_Remail": "R_emaildomain",
            "has_dist1": "dist1",
            "has_id01": "id_01",
            "has_id05": "id_05",
            "has_devicetype": "DeviceType",
            "has_deviceinfo": "DeviceInfo",
        }
        for flag, src in flag_cols.items():
            df[flag] = df[src].notna().astype(int) if src in df.columns else 0

        # An overall "do we have any identity evidence at all?" indicator
        id_parts = [c for c in ["has_id01", "has_id05", "has_devicetype"] if c in df]
        df["has_identity_record"] = (df[id_parts].sum(axis=1) > 0).astype(int)

        # --- Group-level missingness flags
        df["identity_missing_flag"] = 1 - df["has_identity_record"]
        df["email_missing_flag"] = (
            df["P_emaildomain"].isna().astype(int)
            if "P_emaildomain" in df.columns else 0
        )
        df["distance_missing_flag"] = 1 - df["has_dist1"]
        df["address_missing_flag"] = (
            df["addr1"].isna().astype(int) if "addr1" in df.columns else 0
        )

        # --- Categorical fields: missing becomes its own explicit category
        for col in ["P_emaildomain", "R_emaildomain", "DeviceType"]:
            if col in df.columns:
                df[col] = df[col].fillna("missing")

        # --- Card metadata: mode imputation (missingness here is <0.01%, too
        #     rare to carry signal, so we just fill it)
        for col, mode in self.modes_.items():
            df[col] = df[col].fillna(mode)

        # --- Address codes: -1 sentinel.
        #     These are anonymised CATEGORY codes, not quantities, so -1 just
        #     means "unknown address bucket". Important: because they're
        #     categorical, we must not feed them to Logistic Regression as raw
        #     numbers (see build_feature_matrix).
        for col in ["addr1", "addr2"]:
            if col in df.columns:
                df[col] = df[col].fillna(-1)

        # --- Numeric fields: median imputation, but keep the missingness flag
        for col, med in self.medians_.items():
            if col in df.columns:
                df[f"{col}_was_missing"] = df[col].isna().astype(int)
                df[col] = df[col].fillna(med)

        return df

    def fit_transform(self, df):
        return self.fit(df).transform(df)


# =============================================================================
# 4. FEATURE ENGINEERING
# =============================================================================
class FeatureEngineer:
    """
    KNIME equivalent: Rule Engine / Math Formula / GroupBy / Joiner nodes.

    CONCEPT — Why engineer features at all?
    ---------------------------------------
    Raw columns often don't express the thing you actually care about. "This
    card-address pair has been seen 3 times before" is far more useful to a
    model than the raw card and address IDs, which are meaningless codes.
    Feature engineering is where domain knowledge enters the model.

    CONCEPT — Frequency encoding and leakage
    -----------------------------------------
    Frequency tables are computed on TRAIN ONLY and then looked up for
    val/test. Combinations never seen in training get frequency 0 and are
    flagged as novel. Recomputing frequencies on the test set would leak
    information about the test distribution.
    """

    def __init__(self, rare_threshold=5):
        # rare_threshold: how few occurrences counts as "rare". 5 is a common
        # starting point; tune it on validation, not on test.
        self.rare_threshold = rare_threshold
        self.card_addr_freq_ = {}
        self.card_email_freq_ = {}
        self.card_addr_amt_mean_ = {}
        self.amt_q90_by_product_ = {}
        self.global_amt_mean_ = 0.0
        # Train-fitted global fallbacks. These exist so that NOTHING is ever
        # computed from the distribution of val/test at transform time.
        self.global_amt_q90_ = 0.0
        self.global_amt_q99_ = 0.0

    @staticmethod
    def _key(df, cols):
        """Build a combination key like 'visa|150.0' from several columns."""
        return df[cols].astype(str).agg("|".join, axis=1)

    def fit(self, df):
        ca = self._key(df, ["card1", "addr1"])
        ce = self._key(df, ["card1", "P_emaildomain"])

        self.card_addr_freq_ = ca.value_counts().to_dict()
        self.card_email_freq_ = ce.value_counts().to_dict()

        # Mean transaction amount per card-address group (for deviation feature)
        self.card_addr_amt_mean_ = df.groupby(ca)["TransactionAmt"].mean().to_dict()
        self.global_amt_mean_ = df["TransactionAmt"].mean()
        self.global_amt_q90_ = df["TransactionAmt"].quantile(0.90)
        self.global_amt_q99_ = df["TransactionAmt"].quantile(0.99)

        # 90th percentile of amount WITHIN each product category.
        # Per-category is better than one global threshold: $500 is normal for
        # one product type and wildly unusual for another.
        self.amt_q90_by_product_ = (
            df.groupby("ProductCD")["TransactionAmt"].quantile(0.90).to_dict()
        )
        return self

    def transform(self, df):
        df = df.copy()
        m_cols = [c for c in df.columns if c.startswith("M") and c[1:].isdigit()]

        # --- M-field summaries -------------------------------------------------
        # EDA showed missing M-values and failed M-values carry DIFFERENT
        # signals, so we count them separately rather than lumping together.
        if m_cols:
            df["m_failure_count"] = (df[m_cols] == "F").sum(axis=1)
            df["m_missing_count"] = df[m_cols].isna().sum(axis=1)
            df["m_match_count"] = (df[m_cols] == "T").sum(axis=1)
            for c in m_cols:
                df[f"{c}_missing"] = df[c].isna().astype(int)
                df[f"{c}_failed"] = (df[c] == "F").astype(int)
            if "M5" in df.columns:
                df["M5_matched_flag"] = (df["M5"] == "T").astype(int)

        # --- Email relationship ------------------------------------------------
        # Three states, not two: matched / different / missing. EDA showed
        # "missing" behaves differently from "mismatch", so collapsing them
        # would destroy signal.
        if {"P_emaildomain", "R_emaildomain"}.issubset(df.columns):
            df["email_domain_match_status"] = np.where(
                df["R_emaildomain"] == "missing", -1,
                (df["P_emaildomain"] == df["R_emaildomain"]).astype(int),
            )

        # --- Card network x funding type --------------------------------------
        if {"card4", "card6"}.issubset(df.columns):
            df["card_network_funding"] = (
                df["card4"].astype(str) + "_" + df["card6"].astype(str)
            )

        # --- Relationship frequency / rarity / novelty ------------------------
        ca = self._key(df, ["card1", "addr1"])
        ce = self._key(df, ["card1", "P_emaildomain"])

        df["card_addr_freq"] = ca.map(self.card_addr_freq_).fillna(0)
        df["card_addr_rare"] = (df["card_addr_freq"] < self.rare_threshold).astype(int)
        df["card_addr_unseen"] = (df["card_addr_freq"] == 0).astype(int)

        df["card_email_freq"] = ce.map(self.card_email_freq_).fillna(0)
        df["card_email_rare"] = (df["card_email_freq"] < self.rare_threshold).astype(int)
        df["card_email_unseen"] = (df["card_email_freq"] == 0).astype(int)

        # --- Amount deviation from the group's typical amount -----------------
        # For unseen groups we fall back to the global mean, so the deviation
        # is still defined (rather than NaN) and means "vs. typical overall".
        group_mean = ca.map(self.card_addr_amt_mean_).fillna(self.global_amt_mean_)
        df["amt_dev_card_addr"] = df["TransactionAmt"] - group_mean

        # --- Amount unusually high for its product category -------------------
        q90 = df["ProductCD"].map(self.amt_q90_by_product_)
        # FIX: fall back to the TRAIN global 90th percentile, not this
        # dataframe's own. Previously an unseen ProductCD would have caused the
        # threshold to be computed from val/test's own distribution.
        n_unseen = int(q90.isna().sum())
        if n_unseen:
            print(f"  [note] {n_unseen:,} rows had a ProductCD unseen in train; "
                  f"using train global q90 fallback")
        q90 = q90.fillna(self.global_amt_q90_)
        df["amt_above_q90_by_product"] = (df["TransactionAmt"] > q90).astype(int)

        # --- Log transforms ---------------------------------------------------
        # CONCEPT: transaction amounts are heavily right-skewed (most small, a
        # few enormous). Logistic Regression is sensitive to this; log
        # compresses the tail. log1p(x) = log(1+x), which safely handles zeros.
        for col in ["TransactionAmt", "C1", "C9", "V130"]:
            if col in df.columns:
                df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

        return df

    def fit_transform(self, df):
        return self.fit(df).transform(df)


# =============================================================================
# 5. THE FOUR PIPELINES
# =============================================================================

# ---------------------------------------------------------------------------
# 5c. Transparent rule-based scorecard (no learning involved)
# ---------------------------------------------------------------------------
def rule_based_scorecard(df, global_amt_q99):
    """
    KNIME equivalent: Rule Engine + Column Aggregator producing total_risk_score.

    NOTE ON `global_amt_q99` (this used to be a bug)
    ------------------------------------------------
    The "unusually large amount" threshold is passed IN, having been fitted on
    the training period. Earlier this function computed
    `df["TransactionAmt"].quantile(0.99)` from whatever dataframe it was handed
    — so a test transaction's score depended on the other 88,580 test
    transactions around it.

    That is not label leakage (isFraud is never touched here), but it is
    "transductive preprocessing": it assumes you can see the whole future batch
    before scoring any single transaction in it. That contradicts how this
    system is described — a decision made at transaction time. In production a
    new transaction arrives alone; you cannot peek at tomorrow's transactions to
    decide whether today's is unusually large.

    A fixed, historically-derived threshold is both honest and deployable.

    CONCEPT — Why include a hand-built rule system alongside ML models?
    -------------------------------------------------------------------
    Because a human can read it. If a regulator, a fraud analyst, or a customer
    asks "why was I flagged?", you can answer precisely: "+2 for an unusually
    large amount, +2 for a rare card-address pair". No ML model gives you that
    for free. It also acts as a sanity floor: if your fancy model can't beat
    simple hand-written rules, something is wrong.

    The point weights below are judgement calls informed by EDA, not learned.
    That is intentional — this pipeline is meant to be auditable and adjustable
    by the business, not optimal.
    """
    amount_points = (
        df.get("amt_above_q90_by_product", 0) * 2
        + (df["TransactionAmt"] > global_amt_q99).astype(int) * 1
    )
    card_category_points = (
        df.get("card_addr_rare", 0) * 2
        + df.get("card_addr_unseen", 0) * 2
        + df.get("card_email_rare", 0) * 1
    )
    evidence_point = (
        df.get("has_identity_record", 0) * 2
        + df.get("has_Remail", 0) * 2
    )
    m_points = (
        df.get("m_failure_count", 0) * 1
        + (df.get("m_missing_count", 0) >= 5).astype(int) * 2
        + df.get("M5_matched_flag", 0) * 1
    )

    out = pd.DataFrame({
        "amount_points": amount_points,
        "card_category_points": card_category_points,
        "evidence_point": evidence_point,
        "m_points": m_points,
    })
    out["total_risk_score"] = out.sum(axis=1)
    return out


# ---------------------------------------------------------------------------
# Feature matrix construction (shared by the learned models)
# ---------------------------------------------------------------------------
def build_feature_matrix(df, categorical_cols, numeric_cols, encoders=None, fit=False):
    """
    CONCEPT — Models only eat numbers
    ----------------------------------
    "visa_credit" means nothing to a model. We must encode categories numerically.

    Two common approaches:
      - One-hot encoding: one 0/1 column per category. No false ordering, but
        explodes in width if a column has thousands of categories.
      - Ordinal/label encoding: map each category to an integer. Compact, but
        implies a false ordering (is visa=1 "less than" amex=3?). Trees don't
        care much; linear models very much do.

    Here we one-hot the low-cardinality columns (safe for both model types) and
    drop the high-cardinality raw ID codes, whose useful information we already
    extracted as frequency features above.
    """
    if encoders is None:
        encoders = {}

    frames = [df[numeric_cols].astype(float)]

    for col in categorical_cols:
        if col not in df.columns:
            continue
        if fit:
            # Keep only the most common categories; everything else -> "other".
            # This stops rare categories from creating hundreds of near-empty
            # columns, and guarantees train/test have the same shape.
            top = df[col].astype(str).value_counts().head(20).index.tolist()
            encoders[col] = top
        top = encoders.get(col, [])
        vals = df[col].astype(str).where(df[col].astype(str).isin(top), "other")
        dummies = pd.get_dummies(vals, prefix=col)
        # Force identical columns across train/val/test
        expected = [f"{col}_{c}" for c in top] + [f"{col}_other"]
        dummies = dummies.reindex(columns=expected, fill_value=0)
        frames.append(dummies.astype(float))

    X = pd.concat(frames, axis=1)
    X = X.replace([np.inf, -np.inf], 0).fillna(0)
    return X, encoders


# =============================================================================
# 6 + 7. MAIN: train pipelines, stack them, evaluate
# =============================================================================
def evaluate(y_true, y_score, threshold, label):
    """
    CONCEPT — Which metric, and why not accuracy?
    ----------------------------------------------
    With a 3.5% fraud rate, a model that predicts "never fraud" scores 96.5%
    accuracy while catching zero fraud. Accuracy is useless here. Instead:

    ROC-AUC : ranking quality across all thresholds. 0.5 = random, 1.0 = perfect.
              Can look flatteringly high on imbalanced data.
    PR-AUC  : area under precision-recall curve. Baseline = the fraud rate
              itself (~0.035), NOT 0.5. Much more honest for rare-event problems.
              This is the one to lead with for fraud.
    Recall  : of all real fraud, what fraction did we catch? ("fraud caught")
    FPR     : of all legit transactions, what fraction did we wrongly flag?
              ("friction added" — every one is an annoyed real customer)
    Precision: of everything we flagged, what fraction was really fraud?
              This drives the review team's workload.
    """
    roc = roc_auc_score(y_true, y_score)
    pr = average_precision_score(y_true, y_score)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    print(f"\n--- {label} ---")
    print(f"  ROC-AUC : {roc:.4f}")
    print(f"  PR-AUC  : {pr:.4f}   (baseline = {y_true.mean():.4f})")
    print(f"  @threshold {threshold:.2f}:")
    print(f"    Confusion matrix:  TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")
    print(f"    Recall (fraud caught)   : {recall:.2%}")
    print(f"    Precision               : {precision:.2%}")
    print(f"    FPR (friction added)    : {fpr:.2%}")
    return {"roc_auc": roc, "pr_auc": pr, "recall": recall,
            "precision": precision, "fpr": fpr,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def pick_threshold_for_fpr(y_true, y_score, max_fpr=0.05):
    """
    CONCEPT — The threshold is a BUSINESS decision, not a modelling one
    --------------------------------------------------------------------
    The model outputs a probability. Turning that into "block / allow" needs a
    cutoff, and where you put it is a trade-off between catching fraud and
    annoying real customers. There is no mathematically "correct" answer — it
    depends on how much a missed fraud costs vs. how much a false alarm costs.

    WHICH SIDE DO YOU CONSTRAIN?
    ----------------------------
    You can either (a) demand a recall level and accept whatever false-positive
    rate falls out, or (b) cap the false-positive rate and take the best recall
    available underneath it.

    (a) is what this script did originally, targeting 80% recall. The objective
    was met faithfully — and produced ~22% FPR, i.e. more than one in five
    legitimate customers flagged. For a card issuer that is unusable. The
    binding real-world constraint is almost always the review team's capacity
    and customer tolerance, not a recall aspiration.

    So we now do (b): maximise recall SUBJECT TO FPR <= max_fpr. This is the
    honest framing for a payment-risk system.

    IMPORTANT: choose this on VALIDATION data, then apply the frozen number to
    test. Choosing it on test would make your test score optimistic.
    """
    fpr, recall, thresholds = roc_curve(y_true, y_score)

    feasible = np.where(fpr <= max_fpr)[0]
    if len(feasible) == 0:
        print(f"  [warn] no threshold achieves FPR <= {max_fpr:.1%}; using 0.5")
        return 0.5

    best_recall = recall[feasible].max()
    best = feasible[recall[feasible] == best_recall]
    # Ties on recall: take the one with the lowest FPR (least friction).
    best_idx = best[np.argmin(fpr[best])]
    return float(thresholds[best_idx])


def main(data_path, max_fpr):
    # ---- 1-2. Load and split -------------------------------------------------
    df = load_data(data_path)
    train, val, test = chronological_split(df)

    # ---- 3. Missing values (fit on train only!) ------------------------------
    mv = MissingValueHandler().fit(train)
    train, val, test = mv.transform(train), mv.transform(val), mv.transform(test)

    # ---- 4. Feature engineering (fit on train only!) -------------------------
    fe = FeatureEngineer().fit(train)
    train, val, test = fe.transform(train), fe.transform(val), fe.transform(test)

    y_train, y_val, y_test = train["isFraud"], val["isFraud"], test["isFraud"]

    # ---- Choose which columns feed the learned models ------------------------
    engineered = [
        "m_failure_count", "m_missing_count", "m_match_count", "M5_matched_flag",
        "email_domain_match_status", "has_Remail", "has_dist1",
        "has_id01", "has_id05", "has_devicetype", "has_identity_record",
        "identity_missing_flag", "email_missing_flag",
        "distance_missing_flag", "address_missing_flag",
        "card_addr_freq", "card_addr_rare", "card_addr_unseen",
        "card_email_freq", "card_email_rare", "card_email_unseen",
        "amt_dev_card_addr", "amt_above_q90_by_product",
        "log_TransactionAmt", "log_C1", "log_C9", "log_V130",
        "D4", "V34", "dist1",
    ]
    numeric_cols = [c for c in engineered if c in train.columns]
    categorical_cols = [c for c in ["ProductCD", "card4", "card6", "DeviceType",
                                     "P_emaildomain", "card_network_funding"]
                        if c in train.columns]

    X_train, enc = build_feature_matrix(train, categorical_cols, numeric_cols, fit=True)
    X_val, _ = build_feature_matrix(val, categorical_cols, numeric_cols, enc)
    X_test, _ = build_feature_matrix(test, categorical_cols, numeric_cols, enc)
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    print(f"\nFeature matrix: {X_train.shape[1]} columns")

    # =========================================================================
    # PIPELINE A: Logistic Regression
    # =========================================================================
    # CONCEPT: fits a weighted sum of features, squashed through a sigmoid into
    # a 0-1 probability. Linear, fast, and the coefficients are interpretable:
    # a positive coefficient means "more of this -> more fraud".
    #
    # Scaling matters here. Features range from 0/1 flags to frequencies in the
    # thousands. Without standardising, the optimiser struggles and coefficients
    # aren't comparable to each other. Trees don't need this; linear models do.
    #
    # class_weight="balanced" tells it to care about the rare fraud class rather
    # than optimising overall accuracy by ignoring fraud entirely.
    print("\n[A] Training Logistic Regression...")
    scaler = StandardScaler().fit(X_train)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced",
                            random_state=RANDOM_STATE)
    lr.fit(scaler.transform(X_train), y_train)

    lr_val = lr.predict_proba(scaler.transform(X_val))[:, 1]
    lr_test = lr.predict_proba(scaler.transform(X_test))[:, 1]

    # =========================================================================
    # PIPELINE B: XGBoost
    # =========================================================================
    # CONCEPT: gradient boosting builds many shallow decision trees in
    # sequence, where each new tree tries to fix the errors of the ones before
    # it. It captures non-linear patterns and feature interactions that
    # Logistic Regression cannot ("rare card-address AND high amount AND no
    # identity record"), which is why it usually wins on tabular data.
    #
    # scale_pos_weight is XGBoost's version of class balancing.
    print("[B] Training XGBoost...")
    n_neg, n_pos = (y_train == 0).sum(), (y_train == 1).sum()
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,      # smaller = slower but usually better generalisation
        subsample=0.8,          # each tree sees 80% of rows -> reduces overfitting
        colsample_bytree=0.8,   # and 80% of columns
        scale_pos_weight=n_neg / n_pos,
        eval_metric="aucpr",    # optimise PR-AUC, the right metric for rare events
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train)

    xgb_val = xgb_model.predict_proba(X_val)[:, 1]
    xgb_test = xgb_model.predict_proba(X_test)[:, 1]

    # =========================================================================
    # PIPELINE C: Transparent scorecard
    # =========================================================================
    print("[C] Computing rule-based scorecard...")
    # The high-amount threshold comes from the TRAIN period and is frozen.
    rule_val = rule_based_scorecard(val, fe.global_amt_q99_)["total_risk_score"]
    rule_test = rule_based_scorecard(test, fe.global_amt_q99_)["total_risk_score"]

    # =========================================================================
    # PIPELINE D: Isolation Forest (UNSUPERVISED)
    # =========================================================================
    # CONCEPT: this one never sees the fraud labels. It learns what "normal"
    # looks like and scores how anomalous each transaction is, by measuring how
    # easily a random tree can isolate it.
    #
    # Why bother, when we have labels? Because it can flag NEW fraud patterns
    # that don't resemble anything in the historical labelled data. The
    # supervised models can only recognise fraud that looks like past fraud.
    print("[D] Training Isolation Forest (unsupervised)...")
    iso = IsolationForest(n_estimators=200, contamination=0.035,
                          random_state=RANDOM_STATE, n_jobs=-1)
    iso.fit(scaler.transform(X_train))   # note: no y_train passed
    # score_samples: lower = more anomalous. Negate so higher = more suspicious,
    # keeping all four pipelines pointing the same direction.
    iso_val = -iso.score_samples(scaler.transform(X_val))
    iso_test = -iso.score_samples(scaler.transform(X_test))

    # =========================================================================
    # 6. DECISION LAYER — stacking
    # =========================================================================
    # CONCEPT — Stacking / stacked generalisation
    # --------------------------------------------
    # Rather than averaging the four scores or hand-picking weights, we train a
    # SECOND model whose only inputs are the four pipeline scores. It learns
    # how much to trust each pipeline, and can learn things like "trust the
    # scorecard more when the anomaly score is also high".
    #
    # Crucially the meta-model is trained on VALIDATION predictions, not
    # training predictions. On training data the base models have partly
    # memorised the answers, so their scores look artificially good and the
    # meta-model would learn the wrong weights.
    #
    # We keep it a Logistic Regression with 4 inputs so the result stays
    # interpretable: you can read off exactly how much each pipeline counts.
    print("\n[DECISION LAYER] Stacking four pipelines...")

    meta_val = pd.DataFrame({
        "lr_score": lr_val, "xgb_score": xgb_val,
        "rule_score": rule_val.values, "iso_score": iso_val,
    })
    meta_test = pd.DataFrame({
        "lr_score": lr_test, "xgb_score": xgb_test,
        "rule_score": rule_test.values, "iso_score": iso_test,
    })

    # CONCEPT — Don't let one set do two jobs
    # ----------------------------------------
    # Previously the FULL validation set both trained the meta-model AND chose
    # the threshold. The threshold was therefore picked on the meta-model's own
    # in-sample predictions, which are optimistic — the model has already partly
    # fitted those rows. The chosen threshold then looks better on validation
    # than it will behave on fresh data.
    #
    # Fix: split validation in two, CHRONOLOGICALLY (same reasoning as the main
    # split — no peeking forward). First half trains the meta-model, second half
    # is fresh data used only to choose the operating threshold.
    #
    # Final ordering, strictly forward in time:
    #   base models -> meta-model -> threshold -> test
    cut = len(meta_val) // 2
    meta_tr, y_meta_tr = meta_val.iloc[:cut], y_val.iloc[:cut]
    meta_th, y_meta_th = meta_val.iloc[cut:], y_val.iloc[cut:]
    print(f"  validation split: {len(meta_tr):,} for meta-training, "
          f"{len(meta_th):,} for threshold selection")

    meta_scaler = StandardScaler().fit(meta_tr)
    meta = LogisticRegression(max_iter=1000, class_weight="balanced",
                              random_state=RANDOM_STATE)
    meta.fit(meta_scaler.transform(meta_tr), y_meta_tr)

    combined_th = meta.predict_proba(meta_scaler.transform(meta_th))[:, 1]
    combined_test = meta.predict_proba(meta_scaler.transform(meta_test))[:, 1]

    print("\n  Meta-model weights (how much each pipeline counts):")
    for name, coef in sorted(zip(meta_val.columns, meta.coef_[0]),
                             key=lambda kv: -abs(kv[1])):
        print(f"    {name:12s} {coef:+.4f}")

    # ---- Pick the operating threshold on held-out validation half ------------
    threshold = pick_threshold_for_fpr(y_meta_th, combined_th, max_fpr)
    print(f"\n  Threshold maximising recall subject to FPR <= {max_fpr:.0%} "
          f"(chosen on held-out validation half): {threshold:.4f}")

    # Show the validation operating point itself, to evidence that the
    # threshold really does satisfy the constraint on the data it was chosen on.
    # Any gap between this FPR and the test FPR below is distribution drift
    # between the two time periods — the thing you'd monitor in production.
    evaluate(y_meta_th, combined_th, threshold,
             "COMBINED — validation operating point (threshold-selection half)")

    # =========================================================================
    # 7. EVALUATION on the held-out TEST set
    # =========================================================================
    print("\n" + "=" * 70)
    print("EVALUATION — held-out test set (never used for any fitting)")
    print("=" * 70)

    # CONCEPT — Apples-to-apples operating-point comparison
    # -----------------------------------------------------
    # Comparing a model at threshold 0.5 against another at a business-chosen
    # threshold tells you nothing: the difference could be entirely the
    # threshold. To ask "which model is better AT THE OPERATING POINT WE'D
    # ACTUALLY DEPLOY", every model must get the same treatment — its own
    # threshold, chosen on the same validation half, under the same FPR cap.
    #
    # ROC-AUC / PR-AUC remain the threshold-independent comparison; this adds
    # the threshold-dependent one, done fairly.
    print(f"\nEvery model below gets its OWN threshold, chosen on the same")
    print(f"validation half under the same constraint (FPR <= {max_fpr:.0%}).")
    print("This makes recall/precision/FPR directly comparable across rows.")

    # Slice each pipeline's validation scores to the threshold-selection half,
    # so every model selects its threshold on identical rows.
    pipeline_val_scores = {
        "Logistic Regression alone": lr_val[cut:],
        "XGBoost alone": xgb_val[cut:],
        "Rule scorecard alone": rule_val.values[cut:],
        "Isolation Forest alone": iso_val[cut:],
    }
    pipeline_test_scores = {
        "Logistic Regression alone": lr_test,
        "XGBoost alone": xgb_test,
        "Rule scorecard alone": rule_test.values,
        "Isolation Forest alone": iso_test,
    }

    results = {}
    chosen_thresholds = {}
    for name in pipeline_val_scores:
        thr = pick_threshold_for_fpr(y_meta_th, pipeline_val_scores[name], max_fpr)
        chosen_thresholds[name] = thr
        results[name] = evaluate(y_test, pipeline_test_scores[name], thr, name)

    chosen_thresholds["COMBINED (decision layer)"] = threshold
    results["COMBINED (decision layer)"] = evaluate(
        y_test, combined_test, threshold, "COMBINED (decision layer)"
    )

    # ---- Head-to-head summary table -----------------------------------------
    print("\n" + "=" * 88)
    print(f"HEAD-TO-HEAD at matched constraint (validation FPR <= {max_fpr:.0%})")
    print("=" * 88)
    print(f"{'Model':<28}{'thresh':>9}{'test FPR':>10}{'recall':>9}"
          f"{'precision':>11}{'ROC-AUC':>9}{'PR-AUC':>9}")
    print("-" * 88)
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["pr_auc"]):
        print(f"{name:<28}{chosen_thresholds[name]:>9.4f}{r['fpr']:>9.2%}"
              f"{r['recall']:>9.2%}{r['precision']:>11.2%}"
              f"{r['roc_auc']:>9.4f}{r['pr_auc']:>9.4f}")
    print("\nSorted by PR-AUC. If a single pipeline matches or beats the")
    print("ensemble on BOTH PR-AUC and recall-at-this-FPR, the ensemble is not")
    print("earning its added complexity — that is a legitimate finding, not a")
    print("failure. Choosing the simpler model on evidence is good governance.")

    print("\n" + classification_report(
        y_test, (combined_test >= threshold).astype(int),
        target_names=["legit", "fraud"], digits=4))

    # ---- Top features from XGBoost ------------------------------------------
    print("\n--- Top 10 features (XGBoost gain) ---")
    imp = pd.Series(xgb_model.feature_importances_, index=X_train.columns)
    for rank, (feat, val_) in enumerate(imp.nlargest(10).items(), 1):
        print(f"  {rank:2d}. {feat:35s} {val_:.4f}")

    # =========================================================================
    # 8. ABLATION — marginal contribution of each pipeline
    # =========================================================================
    # CONCEPT — Leave-one-out ablation
    # ---------------------------------
    # "How much does each pipeline actually ADD?" A pipeline can have a large
    # coefficient yet contribute nothing, if another pipeline carries the same
    # information. The honest test is to remove it and see what breaks.
    #
    # Everything else is held constant — same split, same threshold logic — so
    # any difference is attributable to the removed pipeline.
    print("\n" + "=" * 70)
    print("ABLATION — what each pipeline adds to the ensemble")
    print("=" * 70)

    full_pr = results["COMBINED (decision layer)"]["pr_auc"]
    full_roc = results["COMBINED (decision layer)"]["roc_auc"]

    best_single = max(
        [(n, r["pr_auc"]) for n, r in results.items() if "alone" in n],
        key=lambda kv: kv[1])
    print(f"\nStrongest single pipeline: {best_single[0]} (PR-AUC {best_single[1]:.4f})")
    print(f"Full ensemble PR-AUC: {full_pr:.4f}  "
          f"(gain over best single: {full_pr - best_single[1]:+.4f})")

    print(f"\n{'Pipeline removed':<20} {'PR-AUC':>9} {'drop':>9} {'ROC-AUC':>9} {'drop':>9}")
    print("-" * 60)
    for drop in meta_val.columns:
        keep = [c for c in meta_val.columns if c != drop]
        # Same meta-training data as the full model, so the only difference
        # between these rows and the full ensemble is the dropped pipeline.
        sc = StandardScaler().fit(meta_tr[keep])
        m = LogisticRegression(max_iter=1000, class_weight="balanced",
                               random_state=RANDOM_STATE)
        m.fit(sc.transform(meta_tr[keep]), y_meta_tr)
        s = m.predict_proba(sc.transform(meta_test[keep]))[:, 1]

        pr_a = average_precision_score(y_test, s)
        roc_a = roc_auc_score(y_test, s)
        print(f"{drop:<20} {pr_a:>9.4f} {full_pr - pr_a:>+9.4f} "
              f"{roc_a:>9.4f} {full_roc - roc_a:>+9.4f}")

    print("\nA large positive drop = that pipeline adds real, non-redundant signal.")
    print("A drop near zero = it is largely redundant with the others.")
    print("A NEGATIVE drop (removal IMPROVES the ensemble) = that pipeline is")
    print("actively diluting the signal, and should be dropped from the system.")

    # =========================================================================
    # 9. DRIFT / MONITORING
    # =========================================================================
    # CONCEPT — A threshold is a promise made on old data
    # ---------------------------------------------------
    # The FPR cap was enforced on validation. Test is a LATER time period, and
    # the score distribution shifts as customer and fraud behaviour change. So
    # the realised test FPR will not exactly equal the cap — and the size of
    # that gap is precisely what a production monitoring job would alert on.
    y_pred_val = (combined_th >= threshold).astype(int)
    tn_v, fp_v, fn_v, tp_v = confusion_matrix(y_meta_th, y_pred_val).ravel()
    val_fpr = fp_v / (fp_v + tn_v) if (fp_v + tn_v) else 0.0
    val_recall = tp_v / (tp_v + fn_v) if (tp_v + fn_v) else 0.0
    test_fpr = results["COMBINED (decision layer)"]["fpr"]
    test_recall = results["COMBINED (decision layer)"]["recall"]

    print("\n" + "=" * 70)
    print("DRIFT — validation operating point vs. realised test performance")
    print("=" * 70)
    print(f"  FPR:    validation {val_fpr:.2%}  ->  test {test_fpr:.2%}  "
          f"({test_fpr - val_fpr:+.2%})")
    print(f"  Recall: validation {val_recall:.2%}  ->  test {test_recall:.2%}  "
          f"({test_recall - val_recall:+.2%})")
    print("\nThe FPR cap was a constraint satisfied on validation, not a")
    print("guarantee about future data. Drift of this size is normal and is")
    print("the argument for periodic recalibration rather than a fixed")
    print("threshold set once at deployment.")

    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="Path to ieee_cis_joined_full.csv")
    p.add_argument("--max-fpr", type=float, default=0.05,
                   help="Cap on false-positive rate when choosing the "
                        "operating threshold (default 0.05 = 5%%)")
    args = p.parse_args()
    main(args.data, args.max_fpr)
