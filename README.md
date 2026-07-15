# Used Car Market Value Predictor

A machine learning project that estimates used car prices from vehicle features, deployed as a live [Streamlit](https://streamlit.io) web app. The primary focus is the end-to-end ML workflow: cleaning a large, messy listings dataset, engineering features, building a leakage-safe preprocessing pipeline, tuning a gradient-boosted model, and exporting an artifact small enough to actually deploy on a memory-constrained host. This is a research-focused project centred on model training and evaluation methodology rather than a production pricing service.

**Live app:** deployed on Streamlit Community Cloud — [Used Car App](https://usedcarapp.streamlit.app/)

Based on Kaggle Data — [marketcheck-automotive-data-us-canada](https://www.kaggle.com/datasets/rupeshraundal/marketcheck-automotive-data-us-canada/data?select=ca-dealers-used.csv)

## Project Overview

Used car prices are difficult to evaluate because listings vary by region, mileage, vehicle condition, trim, and market demand. Buyers may not know whether a listing is overpriced, while sellers may not know if their asking price is realistic.

This project builds a regression model that predicts used car prices from ~358,000 Canadian listings. Beyond accuracy, a core constraint shaped the work: the model has to fit in memory on a serverless host with a ~1 GB ceiling. An initial Random Forest regressor reached usable accuracy but produced a **4.7 GB** artifact, impractical to load in such a container. That constraint drove a migration to XGBoost, which trains faster and serializes to a fraction of the size at comparable accuracy — the deployed artifact is **17.3 MB**.

## Model Performance

Held-out test set (20% split), target variable `price`:

| Metric | Value |
|---|---|
| MAE | **$1,893** |
| RMSE | $4,701 |
| R² | **0.946** |

Accuracy bands (share of test listings within a given error):

| Within | Share |
|---|---|
| ±$2,500 | 79.7% |
| ±$5,000 | 93.4% |
| ±10% | 74.8% |
| ±20% | 91.3% |

The model is proportionally **weakest on cheap cars** (larger percentage error at the low end), **luxury/exotic makes** (Ferrari, Porsche, Maserati), and **cars 15+ years old**. It is strongest on mainstream 4–10 year-old vehicles. The residual gap is dominated by price variance the feature set cannot capture — condition, accident history, and seller motivation are not in the data. A further limitation is data vintage: these listings were scraped in 2022, so the model reflects that market and its absolute price levels should not be read as current.

## Machine Learning Workflow

The project is organized as three notebooks, each owning a distinct stage.

**`01_eda_modeling.ipynb`** — Data loading and inspection, dtype correction, missing-value and duplicate analysis, removal of unusable target rows, handling of impossible numerical values, treatment of suspicious zero-mileage listings as missing, and exploratory visualization of price distributions and relationships to mileage, year, and make.

**`02_model_training_evaluation.ipynb`** — Feature engineering (vehicle age, miles per year, region from postal code, model/trim combinations), column pruning, train/test split, baseline model construction, and comparison across candidate regressors. Hands off a feature-engineered, pre-split dataset as Parquet (`data/model_ready.parquet`).

**`03_error_analysis_model_export.ipynb`** — Self-contained for the production model: its own split, an inline preprocessing pipeline, staged hyperparameter tuning on GPU, error analysis, and export of the deployment bundle (pipeline, metadata, dashboard aggregates, and app form options).

## Modeling Decisions

**Leakage-safe preprocessing.** Imputation and encoding live inside a scikit-learn `Pipeline` rather than being applied to the full dataset up front. This means every cross-validation fold refits the preprocessor on only its own training rows, so imputation medians and one-hot category vocabularies are never computed from validation data.

**High-cardinality encoding.** Columns like `model_trim`, `region`, and `city` use frequency-controlled one-hot encoding (`min_frequency`), folding rare categories into an infrequent bucket. This keeps the feature matrix narrow and reduces overfitting on sparsely represented trims and locations.

**Staged hyperparameter tuning.** Rather than a single combinatorially explosive grid, tuning follows a coordinate-descent approach: fix a moderate learning rate and tree count, tune tree-complexity parameters (`max_depth`, `min_child_weight`, `gamma`, `subsample`, `colsample_bytree`), then tune regularization (`reg_lambda`, `reg_alpha`), then lower the learning rate while raising the number of estimators. Each stage is a small exhaustive `GridSearchCV` scored on cross-validated MAE.

**GPU training, CPU export.** Tuning runs on GPU (`device="cuda"`, `tree_method="hist"`). The deployed artifact is a CPU refit of the winning parameters, so inference never attempts to acquire CUDA on a CPU-only host.

## Error Analysis

The final model is evaluated beyond a single headline metric:

- Residual distribution and residuals-versus-predicted diagnostics
- Accuracy bands (share of listings within a fixed dollar or percentage error)
- Segmented error (MAE and median % error) by `make`, `province`, and vehicle-age bucket
- Inspection of the largest errors, to separate data-quality artifacts from genuine model blind spots

These segment statistics are exported to `models/dashboard_aggregates.json` so the app can show where the model is weaker without shipping the full dataset.

## Application & Deployment

The web app is a **single Streamlit process** — Streamlit is the entire backend and frontend. There is no separate API, no serverless function, and no database; the fitted pipeline carries its own encoders, and nothing is read from or written to persistent storage at inference time.

**Prediction flow.** The form collects only the raw inputs a user actually knows (year, mileage, make, body type, fuel type, drivetrain, transmission, engine, province). The app then derives the engineered features server-side — `car_age`, `miles_per_year`, `miles_was_zero` — exactly as they were computed in training, aligns the row to the model's exact column order with `reindex(columns=expected_columns)`, and calls `pipeline.predict`. It shows the point estimate plus a ±MAE range. A startup self-test predicts a known reference car and refuses to serve if the result is out of range, guarding against a silent library-version or artifact mismatch.

**Precomputed sidecars** keep the app off the 358k-row dataset entirely:

- `models/xgb_deployment_pipeline.joblib` — the fitted CPU pipeline, loaded once via `@st.cache_resource`
- `models/xgb_deployment_metadata.json` — the exact `expected_columns` order, final params, test metrics, accuracy bands, and pinned library versions
- `models/form_options.json` — dropdown options (low-cardinality lists, `make → model_trim` and `province → city` maps, numeric ranges), frequency-filtered to mirror the encoders
- `models/dashboard_aggregates.json` — precomputed price/error segment statistics for the explore and model-info views

**Deployment mechanics.** Streamlit Community Cloud watches the `main` branch: **a push to `main` is a deploy**. Dependencies are pinned in `requirements.txt` to the exact training stack because joblib pickles are version-coupled — a numpy/scikit-learn/xgboost mismatch can break loading or silently misbehave. The app must run on **Python 3.14** (selected in the Community Cloud app's advanced settings) to match the interpreter the artifact was pickled under.

## Repository Structure

```text
app.py                                   # Streamlit entry point (predict page)
pages/
  1_Explore.py                           # market charts from dashboard_aggregates.json
  2_Model_Info.py                        # metrics, accuracy bands, error-by-segment
viz.py                                   # shared chart helpers + palette
requirements.txt                         # runtime deps, pinned to the training stack
models/
  xgb_deployment_pipeline.joblib         # deployed CPU pipeline (17.3 MB)
  xgb_deployment_metadata.json           # schema, metrics, library versions
  form_options.json                      # form dropdown options
  dashboard_aggregates.json              # precomputed chart data
notebooks/
  01_eda_modeling.ipynb
  02_model_training_evaluation.ipynb
  03_error_analysis_model_export.ipynb
data/                                    # datasets (git-ignored; not needed at runtime)
```

## Running Locally

The deployment artifacts are committed, so the app runs out of the box:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Reproducing the model from scratch additionally requires the source data and the notebook dependencies; run notebooks `01 → 02 → 03` in order.

## Tech Stack

**Modeling**
- Python, Pandas / NumPy
- Scikit-learn (`Pipeline`, `ColumnTransformer`, `GridSearchCV`)
- XGBoost (GPU training via `device="cuda"`)
- Matplotlib / Seaborn, Jupyter Notebook

**Serialization**
- joblib for the fitted pipeline
- Parquet for inter-notebook dataset handoff, chosen over CSV because it preserves dtypes and null semantics

**Application & deployment**
- Streamlit (UI, server, and inference in one process)
- Streamlit Community Cloud (push-to-`main` deploys)

## Roadmap

- [x] Complete hyperparameter tuning and record final test metrics
- [x] Export a deployable pipeline, metadata, and precomputed aggregates
- [x] Ship the live prediction form with a point estimate and range
- [x] Cascading `make → model → trim`, postal-code → region, and city inputs
- [x] Explore page: price-distribution and segment charts from the precomputed aggregates
- [x] Model-info page: metrics, accuracy bands, and an honest account of the model's weak spots
- [ ] Segment-specific ranges (make / vehicle-age) in place of the flat ±MAE band
- [ ] Web-scrape newer listings through 2026 to refresh the 2022-vintage training data and retrain
