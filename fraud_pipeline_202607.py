"""
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

Run:
    pip install pandas numpy scikit-learn xgboost
    python fraud_pipeline.py --data path/to/ieee_cis_joined_full.csv
    
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
    Raw columns often don't express the thing you actually care about. "This
    card-address pair has been seen 3 times before" is far more useful to a
    model than the raw card and address IDs, which are meaningless codes.
    Feature engineering is where domain knowledge enters the model.

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
        q90 = q90.fillna(df["TransactionAmt"].quantile(0.90))
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
def rule_based_scorecard(df):

    amount_points = (
        df.get("amt_above_q90_by_product", 0) * 2
        + (df["TransactionAmt"] > df["TransactionAmt"].quantile(0.99)).astype(int) * 1
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


def pick_threshold_for_recall(y_true, y_score, target_recall=0.80):

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns len(thresholds) = len(recall) - 1
    ok = np.where(recall[:-1] >= target_recall)[0]
    if len(ok) == 0:
        return 0.5
    return float(thresholds[ok[-1]])


def main(data_path, target_recall):
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
    rule_val = rule_based_scorecard(val)["total_risk_score"]
    rule_test = rule_based_scorecard(test)["total_risk_score"]

    # =========================================================================
    # PIPELINE D: Isolation Forest (UNSUPERVISED)
    # =========================================================================

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
    """
    Rather than averaging the four scores or hand-picking weights, we train a
    SECOND model whose only inputs are the four pipeline scores. It learns
    how much to trust each pipeline, and can learn things like "trust the
    scorecard more when the anomaly score is also high".
    
    Crucially the meta-model is trained on VALIDATION predictions, not
    training predictions. On training data the base models have partly
    memorised the answers, so their scores look artificially good and the
    meta-model would learn the wrong weights.
    
    We keep it a Logistic Regression with 4 inputs so the result stays
    interpretable: you can read off exactly how much each pipeline counts.
    print("\n[DECISION LAYER] Stacking four pipelines...")
    """
    meta_val = pd.DataFrame({
        "lr_score": lr_val, "xgb_score": xgb_val,
        "rule_score": rule_val.values, "iso_score": iso_val,
    })
    meta_test = pd.DataFrame({
        "lr_score": lr_test, "xgb_score": xgb_test,
        "rule_score": rule_test.values, "iso_score": iso_test,
    })

    meta_scaler = StandardScaler().fit(meta_val)
    meta = LogisticRegression(max_iter=1000, class_weight="balanced",
                              random_state=RANDOM_STATE)
    meta.fit(meta_scaler.transform(meta_val), y_val)

    combined_val = meta.predict_proba(meta_scaler.transform(meta_val))[:, 1]
    combined_test = meta.predict_proba(meta_scaler.transform(meta_test))[:, 1]

    print("\n  Meta-model weights (how much each pipeline counts):")
    for name, coef in sorted(zip(meta_val.columns, meta.coef_[0]),
                             key=lambda kv: -abs(kv[1])):
        print(f"    {name:12s} {coef:+.4f}")

    # ---- Pick the operating threshold on VALIDATION --------------------------
    threshold = pick_threshold_for_recall(y_val, combined_val, target_recall)
    print(f"\n  Threshold for ~{target_recall:.0%} recall (chosen on val): {threshold:.4f}")

    # =========================================================================
    # 7. EVALUATION on the held-out TEST set
    # =========================================================================
    print("\n" + "=" * 70)
    print("EVALUATION — held-out test set (never used for any fitting)")
    print("=" * 70)

    results = {}
    for name, score in [
        ("Logistic Regression alone", lr_test),
        ("XGBoost alone", xgb_test),
        ("Rule scorecard alone", rule_test),
        ("Isolation Forest alone", iso_test),
    ]:
        # Individual pipelines get threshold 0.5 / median only for the
        # confusion matrix; the AUC numbers are threshold-independent and are
        # what actually matter for comparison.
        thr = 0.5 if score.max() <= 1 else np.median(score)
        results[name] = evaluate(y_test, score, thr, name)

    results["COMBINED (decision layer)"] = evaluate(
        y_test, combined_test, threshold, "COMBINED (decision layer)"
    )

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
        sc = StandardScaler().fit(meta_val[keep])
        m = LogisticRegression(max_iter=1000, class_weight="balanced",
                               random_state=RANDOM_STATE)
        m.fit(sc.transform(meta_val[keep]), y_val)
        s = m.predict_proba(sc.transform(meta_test[keep]))[:, 1]

        pr_a = average_precision_score(y_test, s)
        roc_a = roc_auc_score(y_test, s)
        print(f"{drop:<20} {pr_a:>9.4f} {full_pr - pr_a:>+9.4f} "
              f"{roc_a:>9.4f} {full_roc - roc_a:>+9.4f}")

    print("\nA large positive drop = that pipeline adds real, non-redundant signal.")
    print("A drop near zero = it is largely redundant with the others.")

    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="Path to ieee_cis_joined_full.csv")
    p.add_argument("--target-recall", type=float, default=0.80,
                   help="Recall to target when choosing the threshold (default 0.80)")
    args = p.parse_args()
    main(args.data, args.target_recall)
