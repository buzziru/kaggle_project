from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from predicting_sales.config import DEFAULT_DATA_DIR, DEFAULT_MATRIX_PATH, DEFAULT_MODEL_PATH, DEFAULT_SUBMISSION_PATH  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the final LightGBM model and create a Kaggle submission.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX_PATH)
    parser.add_argument("--submission", type=Path, default=DEFAULT_SUBMISSION_PATH)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--num-boost-round", type=int, default=10000)
    parser.add_argument("--early-stopping-rounds", type=int, default=300)
    parser.add_argument("--log-period", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from predicting_sales.data import load_joblib, load_test
    from predicting_sales.modeling import (
        make_submission,
        predict_test,
        save_model,
        split_lgbm_data,
        train_lgbm,
        validation_rmse,
    )

    all_data = load_joblib(args.matrix)
    test = load_test(args.data_dir)

    split = split_lgbm_data(all_data)
    print(f"Train shape: {split.x_train.shape}")
    print(f"Validation shape: {split.x_val.shape}")
    print(f"Test shape: {split.x_test.shape}")

    model, _ = train_lgbm(
        split,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        log_period=args.log_period,
    )
    print(f"Validation RMSE: {validation_rmse(model, split):.6f}")

    predictions = predict_test(model, split)
    submission = make_submission(all_data, test, predictions)
    args.submission.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.submission, index=False)
    save_model(model, args.model_output)

    print(f"Saved submission to {args.submission}")
    print(f"Saved model to {args.model_output}")


if __name__ == "__main__":
    main()
