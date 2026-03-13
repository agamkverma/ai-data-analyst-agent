"""data_engine.py — CSV loading, type coercion, smart column detection, filtering."""

import io
import pandas as pd
import numpy as np

DATE_KW     = ["date","time","month","year","day","period","quarter","week"]
REVENUE_KW  = ["revenue","sales","income","amount","price","total","turnover"]
PROFIT_KW   = ["profit","margin","earnings","net","gain","contribution"]
COST_KW     = ["cost","expense","cogs","spend","expenditure"]
REGION_KW   = ["region","area","territory","zone","location","country","state","city"]
CATEGORY_KW = ["category","segment","type","group","division","class","department"]
PRODUCT_KW  = ["product","item","sku","name","service","goods","brand"]
CUSTOMER_KW = ["customer","client","account","buyer","company","firm"]
QUANTITY_KW = ["units","quantity","qty","count","volume","number","orders"]


def load_dataframe(source) -> pd.DataFrame:
    if isinstance(source, str):
        df = pd.read_csv(source)
    elif isinstance(source, bytes):
        df = pd.read_csv(io.BytesIO(source))
    else:
        df = pd.read_csv(source)
    df.columns = [str(c).strip() for c in df.columns]
    return _coerce_types(df)


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object:
            num = pd.to_numeric(
                df[col].astype(str).str.replace(r"[$,€£%]", "", regex=True),
                errors="coerce"
            )
            if num.notna().mean() > 0.80:
                df[col] = num
                continue
        if df[col].dtype == object:
            col_l = col.lower()
            if any(k in col_l for k in DATE_KW):
                try:
                    df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                except Exception:
                    pass
    return df


def _find(df, keywords):
    for kw in keywords:
        for col in df.columns:
            if kw in col.lower():
                return col
    return None


def detect_columns(df: pd.DataFrame) -> dict:
    raw = {
        "date":     _find(df, DATE_KW),
        "revenue":  _find(df, REVENUE_KW),
        "profit":   _find(df, PROFIT_KW),
        "cost":     _find(df, COST_KW),
        "region":   _find(df, REGION_KW),
        "category": _find(df, CATEGORY_KW),
        "product":  _find(df, PRODUCT_KW),
        "customer": _find(df, CUSTOMER_KW),
        "quantity": _find(df, QUANTITY_KW),
    }
    return {k: v for k, v in raw.items() if v is not None}


def enrich_dataframe(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    df = df.copy()
    dc = col_map.get("date")
    if dc and pd.api.types.is_datetime64_any_dtype(df[dc]):
        df["_Year"]     = df[dc].dt.year
        df["_Month"]    = df[dc].dt.to_period("M").astype(str)
        df["_Quarter"]  = df[dc].dt.to_period("Q").astype(str)
        df["_MonthNum"] = df[dc].dt.month
        df["_MonthDt"]  = df[dc].dt.to_period("M").dt.to_timestamp()
    rc = col_map.get("revenue"); pc = col_map.get("profit")
    if rc and pc:
        df["_Margin"] = (df[pc] / df[rc].replace(0, np.nan) * 100).round(2)
    return df


def apply_filters(df: pd.DataFrame, col_map: dict, filters: dict) -> pd.DataFrame:
    df = df.copy()
    for role, val in filters.items():
        if role == "date_range" and val:
            dc = col_map.get("date")
            if dc and pd.api.types.is_datetime64_any_dtype(df[dc]):
                s, e = pd.Timestamp(val[0]), pd.Timestamp(val[1])
                df = df[(df[dc] >= s) & (df[dc] <= e)]
        else:
            col = col_map.get(role)
            if col and val:
                df = df[df[col].isin(val)]
    return df
