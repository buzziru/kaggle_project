from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = PACKAGE_ROOT.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_MATRIX_PATH = DEFAULT_DATA_DIR / "all_data_result.joblib"
DEFAULT_SUBMISSION_PATH = DEFAULT_DATA_DIR / "submission_lgbm.csv"
DEFAULT_MODEL_PATH = DEFAULT_DATA_DIR / "lgbm_model.joblib"

IDX_FEATURES = ["date_block_num", "shop_id", "item_id"]
TARGET = "item_cnt_month"
VALIDATION_MONTH = 33
TEST_MONTH = 34

DROP_COLS = [
    TARGET,
    "item_avg_date_sales_lag_1",
    "item_category_avg_date_sales_lag_2",
]

CATEGORICAL_FEATURES = [
    "shop_id",
    "item_category_id",
    "city",
    "type_code",
    "platform",
    "month",
    "meta",
]

LGBM_PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.0654,
    "metric": "rmse",
    "boosting_type": "gbdt",
    "force_col_wise": True,
    "random_state": 2025,
    "verbosity": -1,
    "n_jobs": -1,
    "learning_rate": 0.01,
    "feature_fraction": 0.6618,
    "bagging_fraction": 0.6289,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.5,
    "num_leaves": 225,
    "min_child_samples": 50,
    "max_depth": 12,
}

