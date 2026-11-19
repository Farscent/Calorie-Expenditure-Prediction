"""Predict using a trusted, locally generated submission pipeline."""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from calorie_prediction import ROOT, create_submission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts/submission_model.joblib")
    parser.add_argument("--input", type=Path, default=ROOT / "Test.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/submission.csv")
    args = parser.parse_args()
    result = create_submission(joblib.load(args.model), pd.read_csv(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result):,} predictions to {args.output}")


if __name__ == "__main__":
    main()
