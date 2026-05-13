from __future__ import annotations

import gc
import re
from itertools import product

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .config import IDX_FEATURES, TEST_MONTH
from .preprocessing import downcast


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9а-яА-Я\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def create_grid(sales: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    grid = []
    for block_num in sales["date_block_num"].unique():
        cur_shops = sales.loc[sales["date_block_num"] == block_num, "shop_id"].unique()
        cur_items = sales.loc[sales["date_block_num"] == block_num, "item_id"].unique()
        grid.append(np.array(list(product(cur_shops, cur_items, [block_num])), dtype="int32"))

    train_grid = pd.DataFrame(np.vstack(grid), columns=["shop_id", "item_id", "date_block_num"], dtype=np.int32)
    test_grid = test[["shop_id", "item_id"]].copy()
    test_grid["date_block_num"] = TEST_MONTH
    return pd.concat([train_grid, test_grid], ignore_index=True, sort=False, axis=0)


def add_mean_features(
    df: pd.DataFrame,
    groupby_features: list[str],
    mean_feature_list: list[str] | None = None,
) -> pd.DataFrame:
    col_name = [col for col in groupby_features if col != "date_block_num"]
    base_name = "_".join([col.replace("_id", "") for col in col_name])
    feature_name = f"{base_name}_avg_date_sales"
    group = df.groupby(groupby_features).agg(**{feature_name: ("item_cnt_month", "mean")}).reset_index()
    df = df.merge(group, on=groupby_features, how="left")

    if mean_feature_list is not None:
        mean_feature_list.append(feature_name)
    return df


def add_lag_features(
    df: pd.DataFrame,
    key_features: list[str],
    lag_feature_cols: list[str],
    lag_period: list[int],
) -> pd.DataFrame:
    df_result = df.copy()

    for period in lag_period:
        df_lag = df[key_features + lag_feature_cols].copy()
        df_lag["date_block_num"] += period
        df_lag = df_lag.rename(columns={col: f"{col}_lag_{period}" for col in lag_feature_cols})
        df_result = pd.merge(df_result, df_lag, on=key_features, how="left")

    for col in [f"{col}_lag_{period}" for col in lag_feature_cols for period in lag_period]:
        if "cnt" in col or "sales" in col:
            df_result[col] = df_result[col].fillna(0)

    return df_result


def add_diff_features(df: pd.DataFrame, lag_cols: list[str]) -> pd.DataFrame:
    df_result = df.copy()
    base_name = lag_cols[0].replace("_lag_1", "")

    for i in range(len(lag_cols) - 1):
        df_result[f"{base_name}_diff_{i + 1}"] = df_result[lag_cols[i]] - df_result[lag_cols[i + 1]]

    return df_result


def prepare_shops(shops: pd.DataFrame) -> pd.DataFrame:
    shops = shops.copy()
    shops["shop_name_clean"] = shops["shop_name"].apply(clean_text)
    shops["city"] = shops["shop_name_clean"].str.split(" ").str[0]
    shops.loc[shops["city"].isin(["выездная", "интернет"]), "city"] = "special"

    city_corrections = {
        "спб": "санкт-петербург",
        "н": "нижнийновгород",
        "нижний": "нижнийновгород",
        "ростовнадону": "ростов-на-дону",
        "ростов": "ростов-на-дону",
    }
    shops["city"] = shops["city"].replace(city_corrections)

    moscow_satellite_cities = ["жуковский", "мытищи", "химки", "чехов", "балашиха", "сергиев"]
    shops.loc[shops["city"].isin(moscow_satellite_cities), "city"] = "москваобласть"
    shops["city"] = LabelEncoder().fit_transform(shops["city"])
    return shops[["shop_id", "city"]]


def get_platform(name: str) -> str:
    if "pc" in name or "пк" in name:
        return "PC"
    if "ps3" in name:
        return "PS3"
    if "ps4" in name:
        return "PS4"
    if "xbox" in name or "x360" in name:
        return "Xbox"
    if "psp" in name:
        return "PSP"
    if "vita" in name or "psv" in name:
        return "PSVita"
    if "wii" in name:
        return "Wii"
    if "mac" in name:
        return "Mac"
    if "android" in name:
        return "Android"
    return "Etc"


def get_meta_type(name: str) -> str:
    if "цифровая" in name or "digital" in name:
        return "Digital"
    if "версия" in name:
        return "Version"
    if "bd" in name or "blu-ray" in name:
        return "BluRay"
    if "dvd" in name:
        return "DVD"
    if "cd" in name:
        return "CD"
    if "lp" in name:
        return "Vinyl"
    if "jewel" in name:
        return "Jewel"
    if "region" in name or "регион" in name:
        return "Region"
    if "edition" in name or "издание" in name:
        return "Edition"
    if "box" in name:
        return "Box"
    if "фигурка" in name:
        return "Figure"
    if "футболка" in name:
        return "TShirt"
    if "игрушка" in name:
        return "Toy"
    if "арт" in name:
        return "Art"
    return "Normal"


def prepare_items(items: pd.DataFrame, sales_train: pd.DataFrame) -> pd.DataFrame:
    items = items.copy()
    first_sale_month = sales_train.groupby("item_id").agg({"date_block_num": "min"})["date_block_num"]
    items["first_sale_month"] = items["item_id"].map(first_sale_month).fillna(TEST_MONTH)
    items["cleaned_item_name"] = items["item_name"].apply(clean_text)
    items["platform"] = items["cleaned_item_name"].apply(get_platform)
    items["meta"] = items["cleaned_item_name"].apply(get_meta_type)

    for col in ["platform", "meta"]:
        items[col] = LabelEncoder().fit_transform(items[col])

    return items.drop(columns=["item_name", "cleaned_item_name"])


def prepare_item_categories(item_categories: pd.DataFrame) -> pd.DataFrame:
    item_categories = item_categories.copy()
    item_categories["split"] = item_categories["item_category_name"].str.split("-")
    item_categories["type"] = item_categories["split"].map(lambda x: x[0].strip())

    type_map = {
        "PC": "PC",
        "Аксессуары": "Accessories",
        "Билеты (Цифра)": "Tickets (Digital)",
        "Доставка товара": "Delivery",
        "Игровые консоли": "Consoles",
        "Игры": "Games",
        "Игры Android": "Games Android",
        "Игры MAC": "Games MAC",
        "Игры PC": "Games PC",
        "Карты оплаты": "Payment Cards",
        "Карты оплаты (Кино, Музыка, Игры)": "Payment Cards",
        "Кино": "Movies",
        "Книги": "Books",
        "Музыка": "Music",
        "Подарки": "Gifts",
        "Программы": "Software",
        "Служебные": "Service",
        "Чистые носители (шпиль)": "Blank Media",
        "Чистые носители (штучные)": "Blank Media",
        "Элементы питания": "Batteries",
    }

    item_categories["type_code"] = item_categories["type"].map(type_map).fillna("Etc")
    item_categories = item_categories[["item_category_id", "item_category_name", "type_code"]]
    item_categories["type_code"] = LabelEncoder().fit_transform(item_categories["type_code"])
    return item_categories.drop(columns="item_category_name")


def build_base_matrix(
    sales_train: pd.DataFrame,
    shops: pd.DataFrame,
    items: pd.DataFrame,
    item_categories: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    train_matrix = create_grid(sales_train, test)
    group = sales_train.groupby(IDX_FEATURES).agg(
        item_cnt_month=("item_cnt_day", "sum"),
        transaction_cnt=("item_cnt_day", "count"),
    ).reset_index()

    train_matrix = train_matrix.merge(group, on=IDX_FEATURES, how="left")
    train_matrix["item_cnt_month"] = train_matrix["item_cnt_month"].fillna(0).clip(0, 20)
    train_matrix["date_block_num"] = train_matrix["date_block_num"].astype(np.int8)
    train_matrix["shop_id"] = train_matrix["shop_id"].astype(np.int8)
    train_matrix["item_id"] = train_matrix["item_id"].astype(np.int16)
    train_matrix["item_cnt_month"] = train_matrix["item_cnt_month"].astype(np.float16)

    all_data = train_matrix.copy()
    all_data.fillna(0, inplace=True)
    all_data = all_data.merge(shops, on="shop_id", how="left")
    all_data = all_data.merge(items, on="item_id", how="left")

    date_item_avg_price = sales_train.groupby(["date_block_num", "item_id"]).agg(
        date_item_avg_price=("item_price", "mean")
    ).reset_index()
    all_data = all_data.merge(date_item_avg_price, on=["date_block_num", "item_id"], how="left")
    all_data = all_data.merge(item_categories, on="item_category_id", how="left")
    return all_data


def add_all_features(all_data: pd.DataFrame, sales_train: pd.DataFrame) -> pd.DataFrame:
    lag_1_list: list[str] = []
    lag_3_list = ["transaction_cnt", "date_item_avg_price", "item_cnt_month"]
    features_to_drop: list[str] = []

    all_data = downcast(all_data)
    all_data["month"] = all_data["date_block_num"] % 12

    mean_feature_groups = [
        ["date_block_num", "shop_id", "item_category_id"],
        ["date_block_num", "item_id"],
        ["date_block_num", "item_category_id"],
        ["date_block_num", "city", "item_id"],
    ]
    for group in mean_feature_groups:
        all_data = add_mean_features(all_data, group)

    mean_feature_list = ["shop_item_category_avg_date_sales", "city_item_avg_date_sales"]
    lag_1_list.extend(mean_feature_list)
    lag_3_list.extend(["item_avg_date_sales", "item_category_avg_date_sales"])

    all_data = downcast(all_data)
    all_data = all_data.sort_values(by=IDX_FEATURES).reset_index(drop=True)
    all_data = add_lag_features(all_data, key_features=IDX_FEATURES, lag_feature_cols=lag_1_list, lag_period=[1])
    all_data = all_data.sort_values(by=IDX_FEATURES).reset_index(drop=True)
    all_data = add_lag_features(all_data, key_features=IDX_FEATURES, lag_feature_cols=lag_3_list, lag_period=[1, 2, 3])
    all_data = downcast(all_data)

    features_to_drop.extend(lag_1_list)
    features_to_drop.extend(lag_3_list)
    features_to_drop.remove("item_cnt_month")
    features_to_drop.remove("date_item_avg_price")
    all_data = all_data.drop(columns=features_to_drop)
    features_to_drop = []
    gc.collect()

    item_price_table = all_data.groupby(["date_block_num", "item_id"])["date_item_avg_price"].first().reset_index()
    item_price_table = item_price_table.sort_values(by=["item_id", "date_block_num"])
    item_price_grp = item_price_table.groupby("item_id")["date_item_avg_price"]
    item_price_table["item_avg_price_expanding"] = item_price_grp.expanding(min_periods=1).mean().values
    item_price_table["item_avg_price_expanding"] = item_price_table.groupby("item_id")["item_avg_price_expanding"].shift(1)

    all_data = all_data.merge(
        item_price_table[["date_block_num", "item_id", "item_avg_price_expanding"]],
        on=["date_block_num", "item_id"],
        how="left",
    )
    all_data["item_avg_price_expanding"] = all_data.groupby("item_id")["item_avg_price_expanding"].ffill()
    temp_lag_1 = all_data["date_item_avg_price_lag_1"].fillna(all_data["item_avg_price_expanding"])
    all_data["delta_price_lag"] = (temp_lag_1 - all_data["item_avg_price_expanding"]) / all_data["item_avg_price_expanding"]
    all_data["delta_price_lag"] = all_data["delta_price_lag"].replace([np.inf, -np.inf], np.nan).fillna(0)
    all_data = all_data.drop(columns="item_avg_price_expanding")

    all_data["item_age"] = all_data["date_block_num"] - all_data["first_sale_month"]
    all_data.loc[all_data["item_age"] >= 12, "item_age"] = 12

    new_item_df = all_data[all_data["item_age"] == 0]
    new_item_cat_mean = new_item_df.groupby("item_category_id")["item_cnt_month"].mean().reset_index()
    new_item_cat_mean.columns = ["item_category_id", "new_item_cat_avg_cnt"]
    all_data = all_data.merge(new_item_cat_mean, on="item_category_id", how="left")
    all_data["new_item_effect"] = np.where(all_data["item_age"] == 0, all_data["new_item_cat_avg_cnt"], 0)
    all_data = all_data.drop(columns="new_item_cat_avg_cnt")

    shop_first_sale = sales_train.groupby(["shop_id", "item_id"])["date_block_num"].min()
    all_data["item_shop_first_sale"] = all_data.set_index(["shop_id", "item_id"]).index.map(shop_first_sale)
    all_data["item_shop_first_sale"] = all_data["item_shop_first_sale"].fillna(TEST_MONTH)
    all_data["item_shop_age"] = all_data["date_block_num"] - all_data["item_shop_first_sale"]
    all_data.loc[all_data["item_shop_age"] < 0, "item_shop_age"] = 0
    all_data.loc[all_data["item_shop_age"] >= 12, "item_shop_age"] = 12
    features_to_drop.extend(["first_sale_month", "item_shop_first_sale"])

    temp_df = all_data[["date_block_num", "shop_id", "item_id", "item_cnt_month"]].copy()
    temp_df = temp_df.sort_values(by=["shop_id", "item_id", "date_block_num"]).reset_index(drop=True)
    temp_df["item_shop_last_sale"] = np.nan
    temp_df.loc[temp_df["item_cnt_month"] > 0, "item_shop_last_sale"] = temp_df["date_block_num"]
    last_sale_record = temp_df.groupby(["item_id", "shop_id"])["item_shop_last_sale"].shift(1).ffill()
    temp_df["item_shop_last_sale"] = temp_df["date_block_num"] - last_sale_record
    all_data = pd.merge(
        all_data,
        temp_df[["date_block_num", "shop_id", "item_id", "item_shop_last_sale"]],
        on=["date_block_num", "shop_id", "item_id"],
        how="left",
    )
    all_data.loc[all_data["item_shop_last_sale"] >= 12, "item_shop_last_sale"] = 12
    all_data.loc[all_data["item_shop_last_sale"] < 0, "item_shop_last_sale"] = 12
    all_data["item_shop_last_sale"] = all_data["item_shop_last_sale"].fillna(12)

    all_data = all_data.drop(columns=features_to_drop)
    features_to_drop = []
    all_data = downcast(all_data)

    all_data["rolling_3m_cnt_mean"] = all_data[["item_cnt_month_lag_1", "item_cnt_month_lag_2", "item_cnt_month_lag_3"]].mean(axis=1)
    all_data["rolling_3m_cnt_std"] = all_data[["item_cnt_month_lag_1", "item_cnt_month_lag_2", "item_cnt_month_lag_3"]].std(axis=1)
    all_data["rolling_3m_item_mean"] = all_data[["item_avg_date_sales_lag_1", "item_avg_date_sales_lag_2", "item_avg_date_sales_lag_3"]].mean(axis=1)
    all_data["rolling_3m_item_std"] = all_data[["item_avg_date_sales_lag_1", "item_avg_date_sales_lag_2", "item_avg_date_sales_lag_3"]].std(axis=1)
    all_data["rolling_3m_item_cat_mean"] = all_data[
        ["item_category_avg_date_sales_lag_1", "item_category_avg_date_sales_lag_2", "item_category_avg_date_sales_lag_3"]
    ].mean(axis=1)
    all_data["rolling_3m_price_mean"] = all_data[
        ["date_item_avg_price_lag_1", "date_item_avg_price_lag_2", "date_item_avg_price_lag_3"]
    ].mean(axis=1)
    features_to_drop.append("date_item_avg_price")
    all_data = downcast(all_data)

    all_data = add_diff_features(all_data, ["item_cnt_month_lag_1", "item_cnt_month_lag_2", "item_cnt_month_lag_3"])
    all_data = add_diff_features(all_data, ["item_avg_date_sales_lag_1", "item_avg_date_sales_lag_2", "item_avg_date_sales_lag_3"])

    all_data = all_data.drop(columns=features_to_drop)
    target_cols = [col for col in all_data.columns if re.search(r"_lag_[2-3]", col)]
    features_to_drop = target_cols
    for keep_col in [
        "item_avg_date_sales_lag_2",
        "item_category_avg_date_sales_lag_2",
        "item_cnt_month_lag_2",
        "item_cnt_month_lag_3",
        "transaction_cnt_lag_2",
    ]:
        features_to_drop.remove(keep_col)
    all_data = all_data.drop(columns=features_to_drop)

    train_subset = all_data[all_data["date_block_num"] <= 32]
    group_cat = train_subset.groupby("item_category_id")["item_cnt_month"].mean()
    nov_sales = train_subset[train_subset["month"] == 10]
    group_cat_nov = nov_sales.groupby("item_category_id")["item_cnt_month"].mean()
    cat_nov_ratio = (group_cat_nov / group_cat).fillna(1.0)
    cat_ratio_df = cat_nov_ratio.reset_index()
    cat_ratio_df.columns = ["item_category_id", "category_nov_ratio"]
    all_data = pd.merge(all_data, cat_ratio_df, on="item_category_id", how="left")
    all_data["category_nov_ratio"] = all_data["category_nov_ratio"].fillna(1.0)

    for col in ["item_cnt_month", "item_cnt_month_lag_1", "item_cnt_month_lag_2", "item_cnt_month_lag_3", "rolling_3m_cnt_mean"]:
        all_data[col] = all_data[col].clip(0, 20)

    return downcast(all_data)

