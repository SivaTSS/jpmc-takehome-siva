# src/segment.py
import argparse
import json
from pathlib import Path
from datetime import datetime
import sys

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, FunctionTransformer
from sklearn.impute import SimpleImputer

from utils import (
    TARGET_COL,
    WEIGHT_COL,
    load_data,
    clean_data,
)

# plotting (non-interactive backend is set in utils if used there; we import pyplot here)
try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
except Exception:
    plt = None


def _build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str], list[str]]:
    
    num_cols = [
        "age",
        "weeks worked in year",
        "wage per hour",
        "capital gains", "capital losses", "dividends from stocks",
        "num persons worked for employer",
    ]
    num_cols = [c for c in num_cols if c in X.columns]

    cat_cols = [
        "class of worker",
        "detailed industry recode",
        "detailed occupation recode",
        "marital stat",
        "major industry code",
        "major occupation code",
        "race",
        "hispanic origin",
        "sex",
        "member of a labor union",
        "reason for unemployment",
        "full or part time employment stat",
        "tax filer stat",
        "region of previous residence",
        "state of previous residence",
        "detailed household and family stat",
        "detailed household summary in household",
        "migration code-change in msa",
        "migration code-change in reg",
        "migration code-move within reg",
        "live in this house 1 year ago",
        "migration prev res in sunbelt",
        "family members under 18",
        "country of birth father",
        "country of birth mother",
        "country of birth self",
        "citizenship",
        "fill inc questionnaire for veteran's admin",
        "veterans benefits",
    ]
    cat_cols = [c for c in cat_cols if c in X.columns]

    skewed_num = [c for c in ["capital gains", "capital losses", "dividends from stocks"] if c in num_cols]
    regular_num = [c for c in num_cols if c not in skewed_num]

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse=False)

    pre = ColumnTransformer(
        transformers=[
            ("num_skewed", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("clip0", FunctionTransformer(lambda x: np.clip(x, 0, None), feature_names_out="one-to-one")),
                ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                ("scale", RobustScaler()),
            ]), skewed_num),
            ("num_regular", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler())
            ]), regular_num),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("ohe", ohe),
            ]), cat_cols),
        ],
        remainder="drop",
    )
    return pre, num_cols, cat_cols, skewed_num


def _choose_k_elbow(X_small: np.ndarray, w: np.ndarray, k_min: int, k_max: int, random_state: int):
    Ks = list(range(k_min, k_max + 1))
    inertias = []
    for k in Ks:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X_small, sample_weight=w)
        inertias.append(float(km.inertia_))
    inertias = np.array(inertias, dtype=float)
    k_suggest = Ks[1:-1][int(np.argmin(np.diff(inertias, 2)))] if len(inertias) >= 3 else Ks[0]
    print("Elbow inertias:", {k: it for k, it in zip(Ks, inertias)})
    print(f"Elbow suggested k={k_suggest}")
    return int(k_suggest), Ks, inertias


class _Tee:
    # tee stdout/stderr to a log file
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


# ---------- Plot helpers ----------

def _plot_elbow(Ks, inertias, out_path: Path):
    if plt is None:
        return
    plt.figure()
    plt.plot(Ks, inertias, "bx-")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow - compact feature set")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _cluster_order(labels: np.ndarray):
    clusters = np.unique(labels)
    try:
        return sorted(int(c) for c in clusters)
    except Exception:
        return sorted(clusters, key=lambda x: str(x))


def _plot_cluster_sizes(labels: np.ndarray, w: np.ndarray, out_path: Path, title="Cluster weighted share"):
    if plt is None:
        return
    order = _cluster_order(labels)
    shares = []
    for c in order:
        m = (labels == c)
        shares.append(w[m].sum() / w.sum() if w.sum() > 0 else 0.0)

    plt.figure(figsize=(7, 4))
    bars = plt.bar([str(c) for c in order], shares)
    if PercentFormatter is not None:
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    for b, s in zip(bars, shares):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{s*100:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.title(title)
    plt.xlabel("Cluster")
    plt.ylabel("Weighted share")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _weighted_crosstab(labels: np.ndarray, X: pd.DataFrame, w: np.ndarray, cat: str, top_n: int = 6) -> pd.DataFrame:
    order = _cluster_order(labels)
    vc_all = (
        pd.DataFrame({"v": X[cat].astype("string").fillna("NA"), "w": w})
        .groupby("v")["w"].sum().sort_values(ascending=False)
    )
    cats = vc_all.head(top_n).index.tolist()

    rows = []
    for c in order:
        m = (labels == c)
        w_c = w[m]
        denom = w_c.sum() if w_c.sum() > 0 else 1.0
        vc = (
            pd.DataFrame({"v": X.loc[m, cat].astype("string").fillna("NA"), "w": w_c})
            .groupby("v")["w"].sum()
            .reindex(cats, fill_value=0.0)
        )
        rows.append((c, (vc / denom).to_numpy()))
    data = np.vstack([r[1] for r in rows])
    return pd.DataFrame(data, index=[r[0] for r in rows], columns=cats)


def _plot_heatmap(df: pd.DataFrame, out_path: Path, title: str):
    if plt is None or df.empty:
        return
    plt.figure(figsize=(max(7, 1.3 * len(df.columns)), 0.6 * len(df.index) + 2))
    im = plt.imshow(df.values, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Share")
    plt.xticks(np.arange(df.shape[1]), df.columns, rotation=45, ha="right")
    plt.yticks(np.arange(df.shape[0]), [str(i) for i in df.index])
    plt.title(title)

    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            val = df.values[i, j]
            color = "white" if val > 0.4 else "black"
            plt.text(j, i, f"{val*100:.0f}%", ha="center", va="center", fontsize=9, color=color)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_numeric_means(labels: np.ndarray, X: pd.DataFrame, w: np.ndarray, numeric_cols: list[str], out_dir: Path):
    if plt is None:
        return
    order = _cluster_order(labels)
    for nc in numeric_cols:
        if nc not in X.columns:
            continue
        means = []
        for c in order:
            m = (labels == c)
            w_c = w[m]
            denom = w_c.sum() if w_c.sum() > 0 else 1.0
            vals = pd.to_numeric(X.loc[m, nc], errors="coerce").fillna(0).to_numpy()
            means.append((w_c * vals).sum() / denom)

        plt.figure(figsize=(7, 4))
        bars = plt.bar([str(c) for c in order], means)
        for b, v in zip(bars, means):
            plt.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        plt.title(f"{nc} by cluster (weighted mean)")
        plt.xlabel("Cluster")
        plt.ylabel(nc)
        plt.tight_layout()
        out_path = Path(out_dir) / f"numeric_means_{nc.replace(' ', '_')}.png"
        plt.savefig(out_path)
        plt.close()


def _add_bands(X: pd.DataFrame) -> pd.DataFrame:
    Z = X.copy()
    if "age" in Z.columns:
        age_vals = pd.to_numeric(Z["age"], errors="coerce")
        Z["age_band"] = pd.cut(
            age_vals,
            bins=[-np.inf, 17, 24, 34, 44, 54, 64, np.inf],
            labels=["<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
        )
    if "weeks worked in year" in Z.columns:
        wv = pd.to_numeric(Z["weeks worked in year"], errors="coerce")
        Z["weeks_band"] = pd.cut(
            wv,
            bins=[-np.inf, 0, 13, 26, 39, 47, np.inf],
            labels=["0", "1-13", "14-26", "27-39", "40-47", "48-52+"]
        )
    return Z


def _cluster_summary(labels: np.ndarray, X: pd.DataFrame, y: np.ndarray, w: np.ndarray) -> pd.DataFrame:
    rows = []
    uniq = np.unique(labels)
    w_total = float(w.sum()) if w.sum() > 0 else 1.0

    for seg in uniq:
        m = (labels == seg)
        w_c = w[m]
        share = float(w_c.sum()) / w_total if w_total > 0 else 0.0

        pos_rate = 0.0
        if y is not None:
            y_c = y[m].astype(float)
            pos_rate = float((w_c * y_c).sum() / (w_c.sum() if w_c.sum() > 0 else 1.0))

        row = {
            "cluster": int(seg),
            "weighted_share": share,
            "weighted_pos_rate": pos_rate,
        }

        for nc in ("age", "weeks worked in year", "wage per hour", "num persons worked for employer"):
            if nc in X.columns:
                vals = pd.to_numeric(X.loc[m, nc], errors="coerce").fillna(0).to_numpy(dtype=float)
                row[f"mean_{nc}"] = float((w_c * vals).sum() / (w_c.sum() if w_c.sum() > 0 else 1.0))

        rows.append(row)

    df_sum = pd.DataFrame(rows).sort_values("weighted_share", ascending=False)
    return df_sum.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="Population segmentation with KMeans")
    # Defaults assume running from src/ with project folders one level up
    ap.add_argument("--data_path", type=str, default="../data/census-bureau.data")
    ap.add_argument("--columns_path", type=str, default="../data/census-bureau.columns")
    ap.add_argument("--models_dir", type=str, default="../models")
    ap.add_argument("--outputs_dir", type=str, default="../outputs")
    ap.add_argument("--k", type=int, default=0, help="fixed k if >0; if 0 use k=6 after elbow")
    ap.add_argument("--k_min", type=int, default=3)
    ap.add_argument("--k_max", type=int, default=9)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--segments_out", type=str, default="", help="override path for segments.csv")
    ap.add_argument("--summary_out", type=str, default="", help="override path for summary.json")
    ap.add_argument("--profile_topn", type=int, default=6)
    ap.add_argument("--save_models", action="store_true")
    ap.add_argument("--run_name", type=str, default="", help="optional name prefix for the run folder")
    args = ap.parse_args()

    # per-run folder under outputs/logs
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

    # default output file locations inside the run folder unless overridden
    if not args.segments_out:
        args.segments_out = str(run_dir / "segments.csv")
    if not args.summary_out:
        args.summary_out = str(run_dir / "summary.json")

    log_path = run_dir / "segment.log"
    with open(log_path, "w", encoding="utf-8") as lf:
        sys.stdout = _Tee(sys.stdout, lf)
        sys.stderr = _Tee(sys.stderr, lf)

        # save run manifest
        manifest = {
            "run_dir": str(run_dir),
            "timestamp": timestamp,
            "args": {
                "data_path": args.data_path,
                "columns_path": args.columns_path,
                "models_dir": args.models_dir,
                "outputs_dir": args.outputs_dir,
                "k": args.k, "k_min": args.k_min, "k_max": args.k_max,
                "random_state": args.random_state,
                "segments_out": args.segments_out,
                "summary_out": args.summary_out,
                "profile_topn": args.profile_topn,
                "save_models": bool(args.save_models),
                "run_name": args.run_name,
            },
        }
        with open(run_dir / "run.json", "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, indent=2)

        # load + clean
        df = load_data(args.columns_path, args.data_path)
        y = df[TARGET_COL].map({"- 50000.": 0, "50000+.": 1}).astype(int)
        w = pd.to_numeric(df[WEIGHT_COL], errors="coerce").fillna(0.0)

        X = df.drop(columns=[TARGET_COL])
        X_clean = clean_data(X)
        X_fe = X_clean  

        pre, num_cols, cat_cols, skewed_num = _build_preprocessor(X_fe)
        feat_cols = num_cols + cat_cols
        X_small = pre.fit_transform(X_fe[feat_cols])
        print("Feature matrix shape:", getattr(X_small, "shape", None))

        # elbow, use k=6 by default when --k not given
        if args.k > 0:
            k = int(args.k)
            Ks, inertias = [], []
            print(f"Using fixed k={k}")
        else:
            k_suggest, Ks, inertias = _choose_k_elbow(X_small, w.to_numpy(dtype=float), args.k_min, args.k_max, args.random_state)
            if len(Ks) > 0 and len(inertias) > 0:
                _plot_elbow(Ks, inertias, figs_dir / "elbow.png")
            k = 6
            print(f"overriding to k={k} (Elbow suggested k={k_suggest})")

        km = KMeans(n_clusters=k, random_state=args.random_state, n_init=10)
        km.fit(X_small, sample_weight=w.to_numpy(dtype=float))
        labels = km.labels_.astype(int)
        X_fe[f"segment_kmeans_{k}"] = labels

        # segments csv in run folder
        seg_df = pd.DataFrame({f"segment_kmeans_{k}": labels})
        out_path = Path(args.segments_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        seg_df.to_csv(out_path, index_label="row_id")
        print(f"Saved segments to {out_path}")

        # summary dataframe (weighted)
        summary = _cluster_summary(labels, X_fe, y.to_numpy(dtype=float), w.to_numpy(dtype=float))
        print(f"\nKMeans segments (k={k}) - weighted")
        print(summary.to_string(index=False))

        sp = Path(args.summary_out)
        sp.parent.mkdir(parents=True, exist_ok=True)
        summary.to_json(sp, orient="records", indent=2)
        print(f"Saved summary to {sp}")

        # 1) Cluster size bars
        _plot_cluster_sizes(labels, w.to_numpy(dtype=float), figs_dir / f"cluster_share_k{k}.png", title=f"Cluster weighted share (k={k})")

        # 2) Bucket numeric fields for categorical-style profiling (age_band, weeks_band)
        X_prof = _add_bands(X_fe)

        # 3) Heatmaps for key categoricals 
        cat_vars_to_plot = [
            "age_band",
            "education", "class of worker", "marital stat",
            "full or part time employment stat", "sex",
            "major industry code", "major occupation code",
            "race", "hispanic origin",
            "tax filer stat", "citizenship",
            "detailed industry recode", "detailed occupation recode",
            "weeks_band",
        ]
        cat_vars_to_plot = [c for c in cat_vars_to_plot if c in X_prof.columns]
        for cat in cat_vars_to_plot:
            df_hm = _weighted_crosstab(labels, X_prof, w.to_numpy(dtype=float), cat, top_n=args.profile_topn)
            _plot_heatmap(df_hm, figs_dir / f"heatmap_{cat.replace(' ', '_')}.png", title=f"{cat} - top categories per cluster (weighted shares)")

        # 4) Numeric means by cluster
        numeric_cols_to_plot = [
            "age", "weeks worked in year", "wage per hour", "num persons worked for employer"
        ]
        numeric_cols_to_plot = [c for c in numeric_cols_to_plot if c in X_fe.columns]
        _plot_numeric_means(labels, X_fe, w.to_numpy(dtype=float), numeric_cols_to_plot, figs_dir)

        # Optionally persist models
        if args.save_models:
            try:
                import joblib
                mdl_dir = Path(args.models_dir)
                mdl_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(pre, mdl_dir / "seg_preprocessor.pkl")
                joblib.dump(km, mdl_dir / f"seg_kmeans_k{k}.pkl")
                print(f"Saved models to {mdl_dir}")
            except Exception as e:
                print(f"Skipping model save: {e}")

        print(f"\nLog folder: {run_dir}")
        print(f"Figures folder: {figs_dir}")
        print(f"Log file: {log_path}")


if __name__ == "__main__":
    main()
