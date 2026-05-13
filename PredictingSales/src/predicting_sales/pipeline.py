from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import load_raw_data, save_joblib
from .features import (
    add_all_features,
    build_base_matrix,
    prepare_item_categories,
    prepare_items,
    prepare_shops,
)
from .preprocessing import filter_test_shops, fix_shop_id, remove_outliers


def build_feature_matrix(data_dir: str | Path) -> pd.DataFrame:
    raw = load_raw_data(data_dir)

    sales_train = filter_test_shops(raw.sales_train, raw.test)
    sales_train = remove_outliers(sales_train)
    sales_train = fix_shop_id(sales_train)
    test = fix_shop_id(raw.test)

    shops = prepare_shops(raw.shops)
    items = prepare_items(raw.items, sales_train)
    item_categories = prepare_item_categories(raw.item_categories)

    all_data = build_base_matrix(sales_train, shops, items, item_categories, test)
    return add_all_features(all_data, sales_train)


def build_and_save_feature_matrix(data_dir: str | Path, output: str | Path) -> pd.DataFrame:
    matrix = build_feature_matrix(data_dir)
    save_joblib(matrix, output)
    return matrix

