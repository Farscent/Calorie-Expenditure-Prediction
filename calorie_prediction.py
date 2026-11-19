"""Reproducible course-data regression; no holdout-guided model tuning."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
NUMERIC = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]
FEATURES = ["Gender", *NUMERIC]
TARGET = "Calories"
SCORING = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}


def validate_frame(frame: pd.DataFrame, *, labeled: bool) -> pd.DataFrame:
    """Validate supplied schema without silently dropping or modifying records."""
    expected = {"User_ID", *FEATURES, *([TARGET] if labeled else [])}
    if set(frame.columns) != expected:
        raise ValueError(f"Schema mismatch: missing={expected-set(frame.columns)}, extra={set(frame.columns)-expected}")
    if frame.empty or frame["User_ID"].isna().any() or frame["User_ID"].duplicated().any():
        raise ValueError("Provide nonempty data with unique, nonmissing User_ID values.")
    result = frame.copy(deep=True)
    for column in NUMERIC + ([TARGET] if labeled else []):
        result[column] = pd.to_numeric(result[column], errors="raise")
        values = result[column].dropna().to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} contains an infinite value.")
        if column != TARGET and (values <= 0).any():
            raise ValueError(f"{column} must be positive when present.")
    if labeled and (result[TARGET].isna().any() or (result[TARGET] < 0).any()):
        raise ValueError("Calories must contain finite, nonmissing, nonnegative labels.")
    if result["Gender"].dropna().map(lambda x: not isinstance(x, str) or not x.strip()).any():
        raise ValueError("Gender must contain nonempty strings or missing values.")
    return result


def profile_groups(frame: pd.DataFrame) -> np.ndarray:
    """Keep identical predictor profiles together, even under distinct IDs."""
    return pd.util.hash_pandas_object(frame[FEATURES], index=False).to_numpy()


def split_development_holdout(frame: pd.DataFrame, seed: int = 42):
    groups = profile_groups(frame)
    development, holdout = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(frame, groups=groups))
    return development, holdout


def make_pipeline(estimator, *, scale: bool = False) -> Pipeline:
    numeric = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        numeric.append(("scale", StandardScaler()))
    preprocess = ColumnTransformer([
        ("numeric", Pipeline(numeric), NUMERIC),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), ["Gender"]),
    ], remainder="drop")
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def regression_metrics(actual, predicted) -> dict[str, float]:
    return {"rmse": float(root_mean_squared_error(actual, predicted)),
            "mae": float(mean_absolute_error(actual, predicted)),
            "r2": float(r2_score(actual, predicted)),
            "mean_error": float(np.mean(np.asarray(predicted) - np.asarray(actual)))}


def cv_row(name: str, scores) -> dict:
    return {"model": name, **{
        f"{metric}_{stat}": float(function((-1 if metric != "r2" else 1) * scores[f"test_{metric}"]))
        for metric in SCORING
        for stat, function in [("mean", np.mean), ("std", lambda x: np.std(x, ddof=1))]
    }}


def write_json(path: Path, data) -> None:
    def compatible(value):
        if isinstance(value, dict):
            return {key: compatible(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [compatible(item) for item in value]
        if isinstance(value, np.generic):
            return compatible(value.item())
        # XGBoost's default `missing` parameter is NaN; preserve that meaning as
        # a string instead of producing a nonstandard JSON floating-point token.
        if isinstance(value, float) and not np.isfinite(value):
            return "NaN" if np.isnan(value) else str(value)
        return value
    path.write_text(json.dumps(compatible(data), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_experiment(data_dir: Path = ROOT, output_dir: Path = ROOT / "results", *,
                   iterations: int = 12, folds: int = 5, jobs: int = 4, seed: int = 42) -> dict:
    """Select on development CV, evaluate once, then refit for unlabeled submission."""
    if iterations < 1 or folds < 2 or jobs < 1:
        raise ValueError("Use positive iterations/jobs and at least two folds.")
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    train_path, test_path = data_dir / "Train.csv", data_dir / "Test.csv"
    source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in [train_path, test_path]}
    train = validate_frame(pd.read_csv(train_path, float_precision="round_trip"), labeled=True)
    test = validate_frame(pd.read_csv(test_path, float_precision="round_trip"), labeled=False)
    if set(train.User_ID) & set(test.User_ID):
        raise ValueError("Training and submission IDs overlap.")
    development, holdout = split_development_holdout(train, seed)
    X, y = train[FEATURES], train[TARGET]
    X_dev, y_dev = X.iloc[development], y.iloc[development]
    X_holdout, y_holdout = X.iloc[holdout], y.iloc[holdout]
    groups = profile_groups(train)
    cv = list(GroupKFold(n_splits=folds, shuffle=True, random_state=seed).split(X_dev, y_dev, groups[development]))
    assert not set(groups[development]) & set(groups[holdout])
    for fit, validation in cv:
        assert not set(groups[development][fit]) & set(groups[development][validation])

    candidates = {
        "Mean baseline": make_pipeline(DummyRegressor(strategy="mean")),
        "Linear regression": make_pipeline(LinearRegression(), scale=True),
        "Random forest": make_pipeline(RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=jobs)),
        "XGBoost baseline": make_pipeline(XGBRegressor(objective="reg:squarederror", tree_method="hist", n_estimators=100,
                                                       random_state=seed, n_jobs=jobs, eval_metric="rmse")),
    }
    rows, fold_rows = [], []
    for name, pipeline in candidates.items():
        print(f"Cross-validating {name}...", flush=True)
        scores = cross_validate(pipeline, X_dev, y_dev, cv=cv, scoring=SCORING, n_jobs=1, error_score="raise")
        rows.append(cv_row(name, scores))
        for fold in range(folds):
            fold_rows.append({"model": name, "fold": fold + 1, **{
                metric: float(scores[f"test_{metric}"][fold] * (-1 if metric != "r2" else 1)) for metric in SCORING}})

    # One parallelism layer: folds/search are serial; tree estimators use `jobs`.
    print(f"Searching {iterations} XGBoost candidates across {folds} development folds...", flush=True)
    search = RandomizedSearchCV(candidates["XGBoost baseline"], {
        "model__n_estimators": [500, 1000, 1500],
        "model__max_depth": [2, 3, 4],
        "model__learning_rate": [0.03, 0.06, 0.1],
        "model__subsample": [0.9, 1.0],
        "model__colsample_bytree": [0.9, 1.0],
        "model__min_child_weight": [1, 5],
        "model__reg_lambda": [1.0, 5.0],
    }, n_iter=iterations, scoring=SCORING, refit="rmse", cv=cv,
        random_state=seed, n_jobs=1, error_score="raise", return_train_score=False)
    search.fit(X_dev, y_dev)
    candidates["XGBoost tuned"] = clone(search.best_estimator_)
    best = search.best_index_
    tuned_scores = {f"test_{metric}": np.array([search.cv_results_[f"split{i}_test_{metric}"][best] for i in range(folds)]) for metric in SCORING}
    rows.append(cv_row("XGBoost tuned", tuned_scores))
    fold_rows.extend({"model": "XGBoost tuned", "fold": i+1, **{
        metric: float(tuned_scores[f"test_{metric}"][i] * (-1 if metric != "r2" else 1)) for metric in SCORING}}
        for i in range(folds))
    comparison = pd.DataFrame(rows).sort_values("rmse_mean", kind="stable").reset_index(drop=True)
    selected_name = comparison.iloc[0]["model"]
    selected = clone(candidates[selected_name]).fit(X_dev, y_dev)
    # Do not use the following labels to alter preprocessing, search, or selection.
    # Preserve exact float32 estimator outputs as float64 when exporting CSVs;
    # otherwise pandas writes shortened float32 decimals that slightly change
    # reconstructed residual statistics.
    prediction = np.asarray(selected.predict(X_holdout), dtype=np.float64)
    metrics = regression_metrics(y_holdout, prediction)
    print(f"Selected {selected_name}; holdout RMSE={metrics['rmse']:.4f}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = output_dir.parent / "artifacts"
    figures = output_dir.parent / "assets"
    artifacts.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "cv_scores.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "cv_fold_scores.csv", index=False)
    pd.DataFrame(search.cv_results_).to_csv(output_dir / "search_results.csv", index=False)
    residuals = pd.DataFrame({"User_ID": train.iloc[holdout].User_ID.to_numpy(),
                              "actual": y_holdout.to_numpy(), "predicted": prediction,
                              "residual": y_holdout.to_numpy() - prediction})
    residuals.to_csv(output_dir / "holdout_predictions.csv", index=False)
    membership = pd.DataFrame({"User_ID": train.User_ID, "split": "holdout", "validation_fold": 0})
    membership.loc[development, "split"] = "development"
    for index, (_, validation) in enumerate(cv, 1):
        membership.loc[development[validation], "validation_fold"] = index
    membership.to_csv(output_dir / "split_membership.csv", index=False)

    importance = permutation_importance(selected, X_holdout, y_holdout, scoring="neg_root_mean_squared_error",
                                        n_repeats=5, random_state=seed, n_jobs=1)
    importance_frame = pd.DataFrame({"feature": FEATURES, "rmse_increase_mean": importance.importances_mean,
                                     "rmse_increase_std": importance.importances_std}).sort_values("rmse_increase_mean", ascending=False)
    importance_frame.to_csv(output_dir / "permutation_importance.csv", index=False)
    subgroup_rows = []
    age_bands = pd.cut(X_holdout.Age, [0, 39, 59, np.inf], labels=["under 40", "40-59", "60+"])
    for variable, labels in [("Gender", X_holdout.Gender), ("Age band", age_bands)]:
        for label in labels.dropna().unique():
            mask = (labels == label).to_numpy()
            subgroup_rows.append({"variable": variable, "group": str(label), "n": int(mask.sum()),
                                  **regression_metrics(y_holdout.to_numpy()[mask], prediction[mask])})
    pd.DataFrame(subgroup_rows).to_csv(output_dir / "subgroup_metrics.csv", index=False)

    from diagnostics import save_figures
    save_figures(X_dev, y_dev, y_holdout, prediction, importance_frame, figures)
    joblib.dump(selected, artifacts / "development_model.joblib")
    # Submission model is intentionally distinct from the evaluated development model.
    submission_model = clone(candidates[selected_name]).fit(X, y)
    submission = create_submission(submission_model, test)
    submission.to_csv(artifacts / "submission.csv", index=False)
    joblib.dump(submission_model, artifacts / "submission_model.joblib")
    summary = {
        "selected_model": selected_name, "holdout_metrics": metrics,
        "selected_model_parameters": selected.named_steps["model"].get_params(),
        "search_best_parameters": search.best_params_,
        "configuration": {"seed": seed, "folds": folds, "search_iterations": iterations, "estimator_jobs": jobs},
        "data": {"training_rows": len(train), "submission_rows": len(test), "development_rows": len(development),
                 "holdout_rows": len(holdout), "duplicate_predictor_profiles": int(pd.Series(groups).duplicated().sum()),
                 "development_holdout_profile_overlap": 0,
                 "training_submission_profile_overlap": len(set(groups) & set(profile_groups(test))),
                 "source_sha256": source_hashes},
        "negative_holdout_predictions": int((prediction < 0).sum()),
        "submission": {"rows": len(submission), "model_fit": "all labeled Train.csv rows", "labels_available": False},
        "versions": {"python": platform.python_version(), **{p: version(p) for p in ["numpy", "pandas", "scikit-learn", "xgboost", "matplotlib", "joblib"]}},
    }
    write_json(output_dir / "summary.json", summary)
    # Guard against accidental in-place source modification by future workflow changes.
    assert source_hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [train_path, test_path]}
    return summary


def create_submission(model, frame: pd.DataFrame) -> pd.DataFrame:
    validated = validate_frame(frame, labeled=False)
    predictions = np.asarray(model.predict(validated[FEATURES]), dtype=np.float64)
    if len(predictions) != len(validated) or not np.isfinite(predictions).all():
        raise ValueError("Submission predictions must be finite and match the input row count.")
    return pd.DataFrame({"User_ID": validated.User_ID.to_numpy(), "Calories": predictions})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_experiment(args.data_dir, args.output_dir, iterations=args.iterations, folds=args.folds, jobs=args.jobs, seed=args.seed)


if __name__ == "__main__":
    main()
