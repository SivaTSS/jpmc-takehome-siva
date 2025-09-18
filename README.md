# Income Classification & Segmentation Take-Home Project

This repo contains a lightweight, reproducible pipeline to:
1. Train and evaluate a classifier that predicts whether income is `> $50k`
2. Build a KMeans segmentation for marketing personas, with weighted profiling and plots

The code provides clean CLI entry points (`argparse`) and organized outputs per run.

---

## Project Structure

```
project-root/
│
├─ data/
│  ├─ census-bureau.data
│  └─ census-bureau.columns
│
├─ models/                 # saved models & preprocessors (when --save_models is used)
├─ outputs/
│  └─ logs/
│     └─ run_YYYYMMDD_HHMMSS/   # one folder per run (created automatically)
│        ├─ classifier.log / segment.log
│        ├─ run.json            # manifest of args & paths used
│        ├─ metrics.json        # classification metrics (if classifier was run)
│        ├─ segments.csv        # segmentation labels (if segmenter was run)
│        ├─ summary.json        # cluster summary (if segmenter was run)
│        └─ figs/               # all saved figures for that run
│
├─ src/
│  ├─ utils.py
│  ├─ classifier.py
│  └─ segment.py
│
├─ req.txt                 # Python dependencies (core + optional)
└─ README.md
```

> **Paths:** The scripts default to being run **from `src/`** and assume `../data`, `../models`, and `../outputs` exist relative to `src/`. You can override paths via CLI flags.

---

## Setup

- **Python**: 3.9+ recommended
- **Install deps** (core + optional boosters):
  ```bash
  pip install -r requirements.txt
  ```

If you don’t want optional libraries, install only core deps:

```bash
pip install numpy pandas scikit-learn scipy joblib matplotlib
```

---

## Quickstart

From the **project root**, place the data files here:

```
data/
  census-bureau.data
  census-bureau.columns
```

Then run from **inside `src/`** (defaults assume `../data`, `../outputs`, `../models`):

```bash
# Classification (train + eval, save models and metrics)
python classifier.py --save_models

# Segmentation (KMeans, notebook-matched behavior)
python segment.py --save_models
```

Each run creates a new per-run log folder:

```
outputs/logs/run_YYYYMMDD_HHMMSS/
```

---

## Classification

**Script:** `src/classifier.py`

### What it does

* Cleans the CPS-like columns (e.g., `Yes/No/NIU` mapping), standardizes types.
* Adds engineered features (education ordinal, investment income, work attachment, student flag).
* Builds a `ColumnTransformer` with tailored pipelines for skewed, numeric, and categorical features (OHE).
* Handles imbalance using survey weights with a positive-class balancing factor.
* Trains multiple models (selectable), evaluates with **weighted ROC-AUC** & **PR-AUC**, auto-picks a best-F1 threshold for reporting.
* For XGBoost: finds a threshold to satisfy a **target precision** (marketing-friendly knob).

### Example runs

Run with all available models (only those installed will execute):

```bash
# from src/
python classifier.py --models sgd,rf,xgb,lgbm,cat --save_models
```

Run with a subset:

```bash
python classifier.py --models sgd,rf
```

Change the precision target for XGBoost thresholding:

```bash
python classifier.py --models xgb --target_precision 0.80
```

Override defaults (paths & naming):

```bash
python classifier.py   --data_path ../data/census-bureau.data   --columns_path ../data/census-bureau.columns   --models_dir ../models   --outputs_dir ../outputs   --run_name demo_run   --save_models
```

### Outputs per run

`outputs/logs/run_YYYYMMDD_HHMMSS/`

* `classifier.log` – full console log
* `run.json` – manifest of args
* `metrics.json` – list of `{model, roc_auc_w, pr_auc_w, best_thr}`
* `figs/`
  * `<ModelName>_ROC_<timestamp>.png`
  * `<ModelName>_PR_<timestamp>.png`

If `--save_models` is used:

* `models/preprocessor.pkl`
* `models/model_<ModelName>.pkl`

---

## Segmentation

**Script:** `src/segment.py`

### What it does

* Cleans data (same approach as classification).
* Builds a compact feature set (selected numeric + key categoricals with OHE) and runs KMeans.
* **Elbow plot** is computed/saved.
* **Notebook-matched behavior**: if `--k` is not provided, the code computes elbow **but uses `k=6`** for clustering to match the original notebook’s plots.
* Saves per-cluster **weighted share**, **weighted >50k rate**, heatmaps for top categories, and bar plots for numeric means.

### Example runs

```bash
# from src/
python segment.py --save_models
```

Force a specific number of clusters:

```bash
python segment.py --k 8
```

Override paths or name the run:

```bash
python segment.py   --data_path ../data/census-bureau.data   --columns_path ../data/census-bureau.columns   --outputs_dir ../outputs   --models_dir ../models   --run_name segdemo   --save_models
```

### Outputs per run

`outputs/logs/run_YYYYMMDD_HHMMSS/`

* `segment.log` – full console log
* `run.json` – manifest of args
* `segments.csv` – one column `segment_kmeans_<k>`
* `summary.json` – per-cluster `weighted_share`, `weighted_pos_rate`, and numeric means
* `figs/`
  * `elbow.png`
  * `cluster_share_k<k>.png`
  * `heatmap_<categorical>.png` (for each profiled categorical)
  * `numeric_means_<column>.png` (for each profiled numeric)

If `--save_models` is used:

* `models/seg_preprocessor.pkl`
* `models/seg_kmeans_k<k>.pkl`

---

## Reproducibility

* Default `random_state=42` across scripts.
* Run manifests (`run.json`) capture all CLI arguments and output paths.
* Preprocessors and models are persisted with `joblib` when `--save_models` is passed.

---

## CLI Reference

Common flags (both scripts):

* `--data_path` (default: `../data/census-bureau.data`)
* `--columns_path` (default: `../data/census-bureau.columns`)
* `--outputs_dir` (default: `../outputs`)
* `--models_dir` (default: `../models`)
* `--run_name` (prefix for the per-run folder name)

**Classifier-only**

* `--models` (e.g., `sgd,rf,xgb,lgbm,cat`)
* `--target_precision` (default `0.70`, used only for the XGBoost threshold demo)
* `--save_models` (persist preprocessor + model files)
* `--metrics_out` (override metrics path; otherwise saved in the run folder)

**Segment-only**

* `--k` (fix k; if omitted or 0, elbow is computed but the clustering uses `k=6` to match the original notebook)
* `--k_min`, `--k_max` (elbow range)
* `--profile_topn` (top categories shown per categorical variable)
* `--save_models` (persist preprocessor + kmeans)

---
