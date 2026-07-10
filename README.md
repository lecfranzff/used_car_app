# Used Car Market Value Predictor

A machine learning project that estimates used car prices from vehicle features. The primary focus is the end-to-end ML workflow: cleaning a large, messy listings dataset, engineering features, building a leakage-safe preprocessing pipeline, tuning a gradient-boosted model, and exporting an artifact small enough to actually deploy. The web app is the deployment target rather than the centerpiece.

## Project Overview

Used car prices are difficult to evaluate because listings vary by region, mileage, vehicle condition, trim, and market demand. Buyers may not know whether a listing is overpriced, while sellers may not know if their asking price is realistic.

This project builds a regression model that predicts used car prices and surfaces the major factors driving each estimate. Beyond accuracy, a core constraint shaped the work: the model has to fit in memory on a serverless backend. An initial Random Forest regressor reached usable accuracy but produced a 4.7 GB artifact, which is impractical to load in a serverless container. That constraint drove a migration to XGBoost, which trains faster and serializes to a fraction of the size at comparable accuracy.

## Machine Learning Workflow

The project is organized as three notebooks, each owning a distinct stage.

**`01_eda_modeling.ipynb`** — Data loading and inspection, dtype correction, missing-value and duplicate analysis, removal of unusable target rows, handling of impossible numerical values, treatment of suspicious zero-mileage listings as missing, and exploratory visualization of price distributions and relationships to mileage, year, and make.

**`02_model_training_evaluation.ipynb`** — Feature engineering (vehicle age, miles per year, region, model/trim combinations), column pruning, train/test split, baseline model construction, and comparison across candidate regressors. Hands off a feature-engineered, pre-split dataset as Parquet.

**`03_error_analysis_model_export.ipynb`** — Self-contained for the production model: its own split, an inline preprocessing pipeline, staged hyperparameter tuning on GPU, error analysis, feature importance, and export of the deployment artifact.

## Modeling Decisions

**Leakage-safe preprocessing.** Imputation and encoding live inside a scikit-learn `Pipeline` rather than being applied to the full dataset up front. This means every cross-validation fold refits the preprocessor on only its own training rows, so imputation medians and one-hot category vocabularies are never computed from validation data.

**High-cardinality encoding.** Columns like `model_trim`, `region`, and `city` use frequency-controlled one-hot encoding, folding rare categories into an infrequent bucket. This keeps the feature matrix narrow and reduces overfitting on sparsely represented trims and locations.

**Staged hyperparameter tuning.** Rather than a single combinatorially explosive grid, tuning follows the coordinate-descent approach recommended in the XGBoost documentation: fix a moderate learning rate and tree count, tune tree-complexity parameters (`max_depth`, `min_child_weight`, `gamma`, `subsample`, `colsample_bytree`), then tune regularization (`reg_lambda`, `reg_alpha`), then lower the learning rate while raising the number of estimators. Each stage is a small exhaustive `GridSearchCV` scored on cross-validated MAE.

**GPU training, CPU export.** Tuning runs on GPU (`device="cuda"`, `tree_method="hist"`). The deployed artifact is a CPU refit of the winning parameters, so inference never attempts to acquire CUDA in a serverless container.

## Error Analysis

The final model is evaluated beyond a single headline metric:

- Residual distribution and residuals-versus-predicted diagnostics
- Accuracy bands (share of listings within a fixed dollar or percentage error)
- Segmented error by price decile, make, vehicle age, and mileage
- Inspection of the largest absolute and percentage errors, to separate data quality problems from genuine model blind spots
- Feature importance from the fitted booster, mapped back to human-readable column names

## Tech Stack

**Modeling**
- Python
- Pandas / NumPy
- Scikit-learn (`Pipeline`, `ColumnTransformer`, `GridSearchCV`)
- XGBoost (GPU training via `device="cuda"`)
- Matplotlib / Seaborn
- Jupyter Notebook

**Serialization**
- joblib for the fitted pipeline
- Parquet for inter-notebook dataset handoff, chosen over CSV because it preserves dtypes and null semantics

**Deployment (in progress)**
- Modal — hosts the model behind a FastAPI endpoint, keeping the artifact warm in memory between requests
- Vercel — hosts the frontend
- No database. The model carries its own fitted encoders; nothing is read from or written to persistent storage at inference time. A datastore would only be introduced to log predictions, which is not required to ship.

## Target Variable

The target variable is:

```text
price
```

## Roadmap

- Complete hyperparameter tuning and record final test metrics
- Deploy the FastAPI inference endpoint on Modal
- Build the frontend form and prediction display on Vercel
- Surface a predicted price range alongside the point estimate
- Add per-prediction
