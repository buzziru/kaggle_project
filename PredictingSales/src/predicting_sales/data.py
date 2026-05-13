from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd


@dataclass
class RawData:
    sales_train: pd.DataFrame
    items: pd.DataFrame
    item_categories: pd.DataFrame
    shops: pd.DataFrame
    test: pd.DataFrame


def load_raw_data(data_dir: str | Path) -> RawData:
    data_dir = Path(data_dir)
    return RawData(
        sales_train=pd.read_csv(data_dir / "sales_train.csv"),
        items=pd.read_csv(data_dir / "items.csv"),
        item_categories=pd.read_csv(data_dir / "item_categories.csv"),
        shops=pd.read_csv(data_dir / "shops.csv"),
        test=pd.read_csv(data_dir / "test.csv"),
    )


def save_joblib(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_joblib(path: str | Path):
    return joblib.load(Path(path))


def load_test(data_dir: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / "test.csv")

