# src/classifier.py
import argparse
import os
import sys
from pathlib import Path
import json
from datetime import datetime

import numpy as np

from utils import (
    TARGET_COL,
    WEIGHT_COL,
    load_data,
    split_data,
    clean_data,
    add_features,
    build_preprocessor,
    compute_balanced_train_weights,
    eval_simple,
    predict_proba_safe,
    pick_threshold_for_precision,
    save_curves,  # uses a non-interactive matplotlib backend to save figs
)

# Optional libraries are handled gracefully
AVAIL_MODELS = {}


def _register_models():
    from sklearn.linear_model import SGDClassifier
    from sklearn.ensemble import RandomForestClassifier

    AVAIL_MODELS["sgd"] = lambda: (
        "SGD Logistic",
        SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-5,
            max_iter=25,
            tol=1e-3,
            n_jobs=-1,
            random_state=42,
            learning_rate="optimal",
        ),
    )

    AVAIL_MODELS["rf"] = lambda: (
        "RandomForest",
        RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42,
        ),
    )

    try:
        from xgboost import XGBClassifier

        AVAIL_MODELS["xgb"] = lambda: (
            "XGBoost",
            XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                n_jobs=-1,
                random_state=42,
                tree_method="hist",
                eval_metric="logloss",
                scale_pos_weight=1.0,
            ),
        )
    except Exception:
        pass

    try:
        from lightgbm import LGBMClassifier

        AVAIL_MODELS["lgbm"] = lambda: (
            "LightGBM",
            LGBMClassifier(
                n_estimators=700,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                scale_pos_weight=1.0,
            ),
        )
    except Exception:
        pass

    try:
        from catboost import CatBoostClassifier

        AVAIL_MODELS["cat"] = lambda: (
            "CatBoost",
            CatBoostClassifier(
                iterations=800,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=3.0,
                random_seed=42,
                loss_function="Logloss",
                eval_metric="AUC",
                verbose=False,
                thread_count=-1,
            ),
        )
    except Exception:
        pass


def _dense_if_needed(name: str, X_train, X_valid):
    from scipy import sparse

    need_dense = name in {"RandomForest", "CatBoost"}
    if need_dense and sparse.issparse(X_train):
        Xtr = X_train.astype(np.float32).toarray()
        Xva = X_valid.astype(np.float32).toarray()
        return Xtr, Xva
    return X_train, X_valid


def _save_artifacts(models_dir: Path, name: str, model, preprocessor):
    try:
        import joblib
    except Exception:
        return
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, models_dir / "preprocessor.pkl")
    joblib.dump(model, models_dir / f"model_{name}.pkl")


class _Tee:
    # Simple stdout/stderr tee to also write to a log file
    def __init__(self, *files):
        self._files = files

    def write(self, obj):
        for f in self._files:
            try:
                f.write(obj)
                f.flush()
            except Exception:
                pass

    def flush(self):
        for f in self._files:
            try:
                f.flush()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Income classification training and evaluation")
    # Defaults assume running from src/ with project folders one level up
    parser.add_argument("--data_path", type=str, default="../data/census-bureau.data")
    parser.add_argument("--columns_path", type=str, default="../data/census-bureau.columns")
    parser.add_argument("--models_dir", type=str, default="../models")
    parser.add_argument("--outputs_dir", type=str, default="../outputs")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--models", type=str, default="sgd,rf,xgb,lgbm,cat", help="comma-separated keys")
    parser.add_argument("--target_precision", type=float, default=0.70, help="used for XGB threshold demo")
    parser.add_argument("--save_models", action="store_true", help="save fitted models and preprocessor")
    parser.add_argument("--metrics_out", type=str, default="", help="override path to save metrics json")
    parser.add_argument("--run_name", type=str, default="", help="optional name prefix for the run folder")
    args = parser.parse_args()

    # Per-run folder under outputs/logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{args.run_name}_" if args.run_name else ""
    outputs_dir = Path(args.outputs_dir)
    base_logs = outputs_dir / "logs"
    run_id = f"{prefix}run_{timestamp}"
    run_dir = base_logs / run_id
    figs_dir = run_dir / "figs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    base_logs.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Default metrics path lives inside the run folder
    if not args.metrics_out:
        args.metrics_out = str(run_dir / "metrics.json")

    log_path = run_dir / "classifier.log"
    with open(log_path, "w", encoding="utf-8") as lf:
        sys.stdout = _Tee(sys.stdout, lf)
        sys.stderr = _Tee(sys.stderr, lf)

        _register_models()
        chosen = [m.strip() for m in args.models.split(",") if m.strip()]
        chosen = [m for m in chosen if m in AVAIL_MODELS]
        if not chosen:
            raise SystemExit("No valid models selected or available. Check installed libraries and --models.")

        # Small run manifest inside the run folder
        manifest = {
            "run_dir": str(run_dir),
            "timestamp": timestamp,
            "args": {
                "data_path": args.data_path,
                "columns_path": args.columns_path,
                "models_dir": args.models_dir,
                "outputs_dir": args.outputs_dir,
                "test_size": args.test_size,
                "random_state": args.random_state,
                "models": chosen,
                "target_precision": args.target_precision,
                "save_models": bool(args.save_models),
                "metrics_out": args.metrics_out,
                "run_name": args.run_name,
            },
        }
        with open(run_dir / "run.json", "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, indent=2)

        df = load_data(args.columns_path, args.data_path)
        print(f"Shape: {df.shape}")
        print("Target counts:\n", df[TARGET_COL].value_counts())

        X_train, X_valid, y_train, y_valid, w_train, w_valid = split_data(
            df, test_size=args.test_size, random_state=args.random_state
        )

        X_train_clean = clean_data(X_train)
        X_valid_clean = clean_data(X_valid)

        X_train_fe = add_features(X_train_clean)
        X_valid_fe = add_features(X_valid_clean)

        preprocessor, feature_cols = build_preprocessor(X_train_fe)
        X_train_proc = preprocessor.fit_transform(X_train_fe[feature_cols])
        X_valid_proc = preprocessor.transform(X_valid_fe[feature_cols])

        sw_train_bal, pos_factor = compute_balanced_train_weights(y_train, w_train)
        print(f"pos_factor used in training: {pos_factor:.3f}")

        results = []
        for key in chosen:
            name, mdl = AVAIL_MODELS[key]()
            Xtr, Xva = _dense_if_needed(name, X_train_proc, X_valid_proc)

            mdl.fit(Xtr, y_train, sample_weight=sw_train_bal)
            metrics = eval_simple(name, mdl, Xva, y_valid, w_valid)
            results.append({"model": name, **metrics})

            # Save ROC and PR figures per model into this run folder
            probs = predict_proba_safe(mdl, Xva)
            fig_paths = save_curves(
                y_true=y_valid,
                probs=probs,
                weights=w_valid,
                out_dir=figs_dir,
                tag=name,
                timestamp=timestamp,
            )
            if fig_paths:
                print(f"Saved figures for {name}: {fig_paths}")

            if args.save_models:
                _save_artifacts(Path(args.models_dir), name, mdl, preprocessor)

            if name == "XGBoost":
                # Threshold selection demo
                p_full = predict_proba_safe(mdl, X_valid_proc)
                thr = pick_threshold_for_precision(y_valid, p_full, w_valid, target=args.target_precision)
                yhat = (p_full >= thr).astype(int)
                print(f"\nXGBoost threshold @ precision>={args.target_precision:.2f}: {thr:.3f}")
                from sklearn.metrics import classification_report

                print(classification_report(y_valid, yhat, sample_weight=w_valid, digits=3))

        # Summary table
        if results:
            results_sorted = sorted(results, key=lambda r: r["pr_auc_w"], reverse=True)
            print("\nSummary (sorted by PR-AUC_w):")
            for r in results_sorted:
                print(
                    f"{r['model']:12s} | ROC-AUC_w={r['roc_auc_w']:.6f} | PR-AUC_w={r['pr_auc_w']:.6f} | Thr*={r['best_thr']:.3f}"
                )

        # Persist metrics inside this run folder unless overridden
        Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"\nSaved metrics to {args.metrics_out}")
        print(f"Log folder: {run_dir}")
        print(f"Figures folder: {figs_dir}")
        print(f"Log file: {log_path}")


if __name__ == "__main__":
    main()
