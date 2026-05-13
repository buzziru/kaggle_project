from __future__ import annotations

import pandas as pd


def downcast(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    start_memory = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        dtype_name = df[col].dtype.name
        if dtype_name == "object":
            continue
        if dtype_name == "bool":
            df[col] = df[col].astype("int8")
        elif dtype_name.startswith("int") or (df[col] % 1 == 0).all():
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif dtype_name.startswith("float"):
            df[col] = pd.to_numeric(df[col], downcast="float")

    if verbose:
        end_memory = df.memory_usage().sum() / 1024**2
        print(f"Memory usage reduced from {start_memory:.2f} MB to {end_memory:.2f} MB")

    return df


def filter_test_shops(sales_train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    return sales_train[sales_train["shop_id"].isin(test["shop_id"].unique())].copy()


def remove_outliers(sales_train: pd.DataFrame) -> pd.DataFrame:
    sales_train = sales_train[sales_train["item_cnt_day"] < 1000]
    sales_train = sales_train[sales_train["item_price"] < 50000]
    return sales_train.copy()


def fix_shop_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.loc[df["shop_id"] == 0, "shop_id"] = 57
    df.loc[df["shop_id"] == 1, "shop_id"] = 58
    df.loc[df["shop_id"] == 10, "shop_id"] = 11
    df.loc[df["shop_id"] == 39, "shop_id"] = 40
    return df

