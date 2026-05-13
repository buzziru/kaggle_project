from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm.callback import early_stopping, log_evaluation, record_evaluation
from sklearn.metrics import mean_squared_error

from .config import (
    CATEGORICAL_FEATURES,
    DROP_COLS,
    LGBM_PARAMS,
    TEST_MONTH,
    VALIDATION_MONTH,
)
from .preprocessing import fix_shop_id


@dataclass
class SplitData:
    x_train: pd.DataFrame
    y_train: pd.Series
    x_val: pd.DataFrame
    y_val: pd.Series
    x_test: pd.DataFrame


def split_lgbm_data(
    all_data: pd.DataFrame,
    val_month: int = VALIDATION_MONTH,
    drop_cols: list[str] | None = None,
) -> SplitData:
    drop_cols = drop_cols or DROP_COLS

    train_mask = all_data["date_block_num"] < val_month
    val_mask = all_data["date_block_num"] == val_month
    test_mask = all_data["date_block_num"] == TEST_MONTH

    return SplitData(
        x_train=all_data.loc[train_mask].drop(drop_cols, axis=1),
        y_train=all_data.loc[train_mask, "item_cnt_month"].clip(0, 20),
        x_val=all_data.loc[val_mask].drop(drop_cols, axis=1),
        y_val=all_data.loc[val_mask, "item_cnt_month"].clip(0, 20),
        x_test=all_data.loc[test_mask].drop(drop_cols, axis=1),
    )


def train_lgbm(
    split: SplitData,
    params: dict | None = None,
    num_boost_round: int = 10000,
    early_stopping_rounds: int = 300,
    log_period: int = 100,
) -> tuple[lgb.Booster, dict]:
    params = (params or LGBM_PARAMS).copy()
    data_params = params.copy()
    data_params["data_random_seed"] = params.get("random_state", 2025)

    dtrain = lgb.Dataset(
        split.x_train,
        split.y_train,
        categorical_feature=CATEGORICAL_FEATURES,
        params=data_params,
    )
    dval = lgb.Dataset(
        split.x_val,
        split.y_val,
        categorical_feature=CATEGORICAL_FEATURES,
        reference=dtrain,
        params=data_params,
    )

    evals_result: dict = {}
    model = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dtrain, dval],
        valid_names=["train", "eval"],
        callbacks=[
            log_evaluation(period=log_period),
            early_stopping(stopping_rounds=early_stopping_rounds),
            record_evaluation(evals_result),
        ],
    )
    return model, evals_result


def validation_rmse(model: lgb.Booster, split: SplitData) -> float:
    preds = model.predict(split.x_val, num_iteration=model.best_iteration).clip(0, 20)
    return float(np.sqrt(mean_squared_error(split.y_val, preds)))


def predict_test(model: lgb.Booster, split: SplitData) -> np.ndarray:
    return model.predict(split.x_test, num_iteration=model.best_iteration).clip(0, 20)


def make_submission(all_data: pd.DataFrame, test: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    df_pred = all_data.loc[all_data["date_block_num"] == TEST_MONTH, ["shop_id", "item_id"]].copy()
    df_pred["item_cnt_month"] = predictions
    test = fix_shop_id(test)
    submission = pd.merge(test, df_pred, on=["shop_id", "item_id"], how="left")
    return submission[["ID", "item_cnt_month"]]


def save_model(model: lgb.Booster, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

