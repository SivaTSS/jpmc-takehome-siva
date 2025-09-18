import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    classification_report,
    roc_curve,
)

# Non-interactive backend for saving figures
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    matplotlib = None
    plt = None

TARGET_COL = "label"
WEIGHT_COL = "weight"


def load_data(columns_path: str, data_path: str) -> pd.DataFrame:
    with open(columns_path, "r") as f:
        cols = [line.strip() for line in f if line.strip()]
    df = pd.read_csv(data_path, header=None, names=cols)
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    y = df[TARGET_COL].map({"- 50000.": 0, "50000+.": 1}).astype(int)
    w = pd.to_numeric(df[WEIGHT_COL], errors="coerce").fillna(0.0)
    X = df.drop(columns=[TARGET_COL])
    X_tr, X_va, y_tr, y_va, w_tr, w_va = train_test_split(
        X, y, w, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_tr, X_va, y_tr, y_va, w_tr, w_va


def clean_data(df_: pd.DataFrame) -> pd.DataFrame:
    df = df_.copy()

    for c in [
        "age",
        "wage per hour",
        "capital gains",
        "capital losses",
        "dividends from stocks",
        "weeks worked in year",
        "num persons worked for employer",
        WEIGHT_COL,
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    code_map_yn = {0: "NIU", 1: "Yes", 2: "No"}
    for c in ["own business or self employed", "veterans benefits"]:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").astype("Int64")
            df[c] = s.map(code_map_yn).astype("string").fillna("NIU")

    for c in ["detailed industry recode", "detailed occupation recode"]:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").astype("Int64")
            df[c] = s.astype("string")
            df.loc[s.eq(0) | s.isna(), c] = "NIU"

    if "year" in df.columns:
        s = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df["year"] = s.map({94: "1994", 95: "1995"}).astype("string")

    return df


def add_features(df_: pd.DataFrame) -> pd.DataFrame:
    df = df_.copy()

    if "education" in df.columns:
        education_order = [
            "Children",
            "Less than 1st grade",
            "1st 2nd 3rd or 4th grade",
            "5th or 6th grade",
            "7th and 8th grade",
            "9th grade",
            "10th grade",
            "11th grade",
            "12th grade no diploma",
            "High school graduate",
            "Some college but no degree",
            "Associate degree-occupational/vocational",
            "Associate degree-academic program",
            "Bachelors degree(BA AB BS)",
            "Masters degree(MA MS MEng MEd MSW MBA)",
            "Professional school degree (MD DDS DVM LLB JD)",
            "Doctorate degree(PhD EdD)",
        ]
        edu_map = {v: i for i, v in enumerate(education_order)}
        df["education_ord"] = df["education"].map(edu_map).astype("Int64")

    if "wage per hour" in df.columns:
        df["has_wage"] = (df["wage per hour"].fillna(0) > 0)

    if {"capital gains", "capital losses", "dividends from stocks"}.issubset(df.columns):
        cg = df["capital gains"].fillna(0)
        cl = df["capital losses"].fillna(0)
        dv = df["dividends from stocks"].fillna(0)
        df["invest_income"] = dv + cg - cl
        df["has_invest_income"] = (df["invest_income"] > 0)

    if "weeks worked in year" in df.columns:
        wks = df["weeks worked in year"]
        df["worked_any_weeks"] = (wks.fillna(0) > 0)
        df["full_year_52w"] = (wks == 52)

    if "enroll in edu inst last wk" in df.columns:
        s = df["enroll in edu inst last wk"].astype("string")
        df["is_student"] = s.isin(["High school", "College or university"])

    return df


def _clip_nonneg(X: np.ndarray) -> np.ndarray:
    return np.clip(X, 0, None)


def build_preprocessor(X_train_fe: pd.DataFrame) -> Tuple[ColumnTransformer, List[str]]:
    feature_cols = [c for c in X_train_fe.columns if c != WEIGHT_COL]

    num_all = [
        c
        for c in feature_cols
        if pd.api.types.is_numeric_dtype(X_train_fe[c]) and not pd.api.types.is_bool_dtype(X_train_fe[c])
    ]
    cat_cols = [c for c in feature_cols if c not in num_all]

    skewed_num = [c for c in ["wage per hour", "capital gains", "capital losses", "dividends from stocks"] if c in num_all]
    no_scale_num = [c for c in ["weeks worked in year", "num persons worked for employer", "education_ord"] if c in num_all]
    scaled_num = [c for c in ["age"] if c in num_all]
    remaining = [c for c in num_all if c not in set(skewed_num + no_scale_num + scaled_num)]
    scaled_num += remaining

    skewed_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("clip", FunctionTransformer(_clip_nonneg, feature_names_out="one-to-one")),
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", RobustScaler()),
        ]
    )

    no_scale_pipe = Pipeline(steps=[("impute", SimpleImputer(strategy="median"))])

    scaled_num_pipe = Pipeline(steps=[("impute", SimpleImputer(strategy="median")), ("scale", RobustScaler())])

    cat_pipe = Pipeline(steps=[("impute", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", drop="if_binary"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num_skewed", skewed_pipe, skewed_num),
            ("num_noscale", no_scale_pipe, no_scale_num),
            ("num_scaled", scaled_num_pipe, scaled_num),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )
    return preprocessor, feature_cols


def compute_balanced_train_weights(y_train: pd.Series, w_train: pd.Series) -> Tuple[np.ndarray, float]:
    w_pos = float(w_train[y_train == 1].sum())
    w_neg = float(w_train[y_train == 0].sum())
    pos_factor = w_neg / max(w_pos, 1e-12)
    sw_train_bal = w_train * np.where(y_train == 1, pos_factor, 1.0)
    return sw_train_bal.astype(float).values, pos_factor


def predict_proba_safe(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-s))
    return model.predict(X).astype(float)


def eval_simple(name: str, model, X_val, y_val, w_val) -> Dict[str, float]:
    p = predict_proba_safe(model, X_val)
    roc_auc = roc_auc_score(y_val, p, sample_weight=w_val)
    pr_auc = average_precision_score(y_val, p, sample_weight=w_val)
    prec, rec, thr = precision_recall_curve(y_val, p, sample_weight=w_val)
    f1 = (2 * prec * rec) / (prec + rec + 1e-12)
    if len(thr) == 0:
        best_thr = 0.5
    else:
        best_idx = int(np.nanargmax(f1[1:]))  # align with thr
        best_thr = float(thr[best_idx])
    yhat = (p >= best_thr).astype(int)
    print(f"\n== {name} ==")
    print(f"ROC-AUC_w={roc_auc:.3f} | PR-AUC_w={pr_auc:.3f} | Thr*={best_thr:.3f}")
    print(classification_report(y_val, yhat, sample_weight=w_val, digits=3))
    return {"roc_auc_w": float(roc_auc), "pr_auc_w": float(pr_auc), "best_thr": best_thr}


def pick_threshold_for_precision(y: pd.Series, p: np.ndarray, w: Optional[pd.Series], target: float = 0.70) -> float:
    prec, rec, thr = precision_recall_curve(y, p, sample_weight=w)
    if len(thr) == 0:
        return 0.5
    prec, rec = prec[1:], rec[1:]
    ok = prec >= target
    if ok.any():
        return float(thr[ok][np.argmax(rec[ok])])
    f1 = (2 * prec * rec) / (prec + rec + 1e-12)
    return float(thr[np.nanargmax(f1)])


def save_curves(
    y_true: pd.Series,
    probs: np.ndarray,
    weights: Optional[pd.Series],
    out_dir: Path,
    tag: str,
    timestamp: str,
) -> Dict[str, str]:
    """
    Save ROC and Precision-Recall curves to PNG files. Returns dict of file paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    if plt is None:
        return paths

    # ROC
    fpr, tpr, _ = roc_curve(y_true, probs, sample_weight=weights)
    auc_val = roc_auc_score(y_true, probs, sample_weight=weights)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc_val:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC - {tag}")
    plt.legend()
    roc_path = out_dir / f"{tag}_ROC_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()
    paths["roc"] = str(roc_path)

    # PR
    prec, rec, _ = precision_recall_curve(y_true, probs, sample_weight=weights)
    ap_val = average_precision_score(y_true, probs, sample_weight=weights)
    plt.figure()
    plt.plot(rec, prec, label=f"AP={ap_val:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall - {tag}")
    plt.legend()
    pr_path = out_dir / f"{tag}_PR_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(pr_path)
    plt.close()
    paths["pr"] = str(pr_path)

    return paths


from sklearn.model_selection import train_test_split
