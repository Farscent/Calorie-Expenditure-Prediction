import hashlib
import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GroupKFold

from calorie_prediction import (FEATURES, NUMERIC, create_submission, make_pipeline,
                                profile_groups, regression_metrics, run_experiment,
                                split_development_holdout, validate_frame)


@pytest.fixture
def sample():
    rng = np.random.default_rng(7)
    rows = 100
    frame = pd.DataFrame({
        "User_ID": np.arange(rows), "Gender": rng.choice(["male", "female"], rows),
        "Age": rng.integers(20, 70, rows), "Height": rng.uniform(145, 195, rows),
        "Weight": rng.uniform(45, 110, rows), "Duration": rng.integers(1, 31, rows).astype(float),
        "Heart_Rate": rng.uniform(70, 120, rows), "Body_Temp": rng.uniform(37, 41, rows),
    })
    frame["Calories"] = frame.Duration * 5 + frame.Weight * 0.1
    return frame


def test_duplicate_profiles_stay_in_one_partition(sample):
    duplicate = sample.iloc[[0]].copy()
    duplicate["User_ID"] = 1000
    frame = pd.concat([sample, duplicate], ignore_index=True)
    development, holdout = split_development_holdout(frame)
    groups = profile_groups(frame)
    assert not set(groups[development]) & set(groups[holdout])
    assert (0 in development) == (100 in development)
    for fit, validation in GroupKFold(3, shuffle=True, random_state=42).split(frame.iloc[development], groups=groups[development]):
        assert not set(groups[development][fit]) & set(groups[development][validation])
    assert np.array_equal(development, split_development_holdout(frame)[0])


def test_preprocessing_uses_training_statistics_and_ignores_ids(sample):
    sample.loc[0, "Weight"] = np.nan
    sample.loc[0, "Gender"] = np.nan
    model = make_pipeline(DummyRegressor()).fit(sample[FEATURES], sample.Calories)
    numeric = model.named_steps["preprocess"].named_transformers_["numeric"]
    statistic = numeric.named_steps["impute"].statistics_[NUMERIC.index("Weight")]
    assert statistic == sample.Weight.median()
    assert not any("User_ID" in name for name in model.named_steps["preprocess"].get_feature_names_out())
    unseen = sample[FEATURES].iloc[:2].copy()
    unseen["Gender"] = "unseen-category"
    unseen["Weight"] = [np.nan, 10000]
    assert np.isfinite(model.predict(unseen)).all()
    assert numeric.named_steps["impute"].statistics_[NUMERIC.index("Weight")] == statistic


@pytest.mark.parametrize("column,value", [("Calories", np.nan), ("Calories", -1), ("Heart_Rate", np.inf), ("Duration", 0)])
def test_invalid_measurements_fail(sample, column, value):
    sample.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_frame(sample, labeled=True)


def test_schema_duplicate_ids_and_missing_values(sample):
    with pytest.raises(ValueError, match="Schema mismatch"):
        validate_frame(sample.assign(unexpected=1), labeled=True)
    duplicate = sample.copy()
    duplicate.loc[1, "User_ID"] = duplicate.loc[0, "User_ID"]
    with pytest.raises(ValueError, match="unique"):
        validate_frame(duplicate, labeled=True)
    sample.loc[0, "Weight"] = np.nan
    assert validate_frame(sample, labeled=True).Weight.isna().sum() == 1


def test_metrics_and_submission_order(sample):
    metrics = regression_metrics([1, 2, 3], [2, 2, 2])
    assert metrics["rmse"] == pytest.approx(np.sqrt(2 / 3))
    assert metrics["mae"] == pytest.approx(2 / 3)
    assert metrics["r2"] == 0
    model = make_pipeline(DummyRegressor()).fit(sample[FEATURES], sample.Calories)
    inputs = sample.drop(columns="Calories").iloc[::-1]
    untouched = inputs.copy(deep=True)
    submission = create_submission(model, inputs)
    assert submission.User_ID.tolist() == inputs.User_ID.tolist()
    assert submission.columns.tolist() == ["User_ID", "Calories"]
    pd.testing.assert_frame_equal(inputs, untouched)
    with pytest.raises(ValueError, match="Schema mismatch"):
        create_submission(model, sample)  # Accidentally passing labels is rejected.


def test_holdout_labels_do_not_affect_selection_or_cv(sample, tmp_path):
    """An end-to-end counterfactual guards the evaluation boundary."""
    membership = split_development_holdout(sample)
    original = sample.copy()
    changed = sample.copy()
    changed.loc[membership[1], "Calories"] += 100
    summaries = []
    cv_results = []
    for name, frame in [("original", original), ("changed", changed)]:
        directory = tmp_path / name
        directory.mkdir()
        frame.to_csv(directory / "Train.csv", index=False)
        test = sample.drop(columns="Calories").iloc[:10].copy()
        test["User_ID"] += 2000
        test.to_csv(directory / "Test.csv", index=False)
        summaries.append(run_experiment(directory, directory / "results", iterations=1, folds=2, jobs=1))
        cv_results.append(pd.read_csv(directory / "results" / "cv_scores.csv"))
        assert (directory / "artifacts" / "submission_model.joblib").exists()
        assert len(pd.read_csv(directory / "artifacts" / "submission.csv")) == 10
        assert (directory / "assets" / "holdout_diagnostics.png").exists()
    assert summaries[0]["selected_model"] == summaries[1]["selected_model"]
    assert summaries[0]["search_best_parameters"] == summaries[1]["search_best_parameters"]
    pd.testing.assert_frame_equal(cv_results[0], cv_results[1])
    assert summaries[0]["holdout_metrics"]["rmse"] != summaries[1]["holdout_metrics"]["rmse"]


def test_saved_report_and_notebook_are_traceable():
    root = Path(__file__).resolve().parents[1]
    summary = json.loads((root / "results/summary.json").read_text(encoding="utf-8"))
    for filename, digest in summary["data"]["source_sha256"].items():
        assert hashlib.sha256((root / filename).read_bytes()).hexdigest() == digest
    predictions = pd.read_csv(root / "results/holdout_predictions.csv")
    actual = regression_metrics(predictions.actual, predictions.predicted)
    assert actual == pytest.approx(summary["holdout_metrics"])
    comparison = pd.read_csv(root / "results/cv_scores.csv")
    assert comparison.loc[comparison.rmse_mean.idxmin(), "model"] == summary["selected_model"]
    notebook = nbformat.read(root / "group-2-olc-final-project-bersih-v2.ipynb", as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is not None
            assert not any(output.output_type == "error" for output in cell.outputs)
