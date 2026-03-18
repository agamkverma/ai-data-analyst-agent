"""
dataset_profiler.py
-------------------
Handles automatic dataset profiling:
- Structure detection (date, numeric, categorical columns)
- Missing value analysis
- Summary statistics
- KPI computation (revenue, profit, margin, etc.)
"""

import pandas as pd
import numpy as np
from typing import Optional


# ── Column role keywords ────────────────────────────────────────────────────────
_DATE_KW     = ["date","time","month","year","day","period","quarter","week","timestamp","dt"]
_REVENUE_KW  = ["revenue","sales","income","amount","price","total","turnover","gmv","sale"]
_PROFIT_KW   = ["profit","margin","earnings","net","gain","contribution","ebit"]
_COST_KW     = ["cost","expense","cogs","spend","expenditure","overhead"]
_REGION_KW   = ["region","area","territory","zone","location","country","state","city","district","geo"]
_CATEGORY_KW = ["category","segment","type","group","division","class","department","vertical","sector"]
_PRODUCT_KW  = ["product","item","sku","name","service","goods","brand","model","description"]
_CUSTOMER_KW = ["customer","client","account","buyer","company","firm","partner","user"]
_QUANTITY_KW = ["units","quantity","qty","count","volume","number","orders","pieces","sold"]


def _match(cols: list[str], keywords: list[str]) -> Optional[str]:
    """Return the first column whose name contains any keyword (case-insensitive)."""
    for kw in keywords:
        for col in cols:
            if kw.lower() in col.lower():
                return col
    return None


def detect_column_roles(df: pd.DataFrame) -> dict:
    """Auto-detect semantic roles for dataset columns."""
    cols = df.columns.tolist()
    roles = {
        "date":     _match(cols, _DATE_KW),
        "revenue":  _match(cols, _REVENUE_KW),
        "profit":   _match(cols, _PROFIT_KW),
        "cost":     _match(cols, _COST_KW),
        "region":   _match(cols, _REGION_KW),
        "category": _match(cols, _CATEGORY_KW),
        "product":  _match(cols, _PRODUCT_KW),
        "customer": _match(cols, _CUSTOMER_KW),
        "quantity": _match(cols, _QUANTITY_KW),
    }
    return {k: v for k, v in roles.items() if v is not None}


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Intelligently coerce column types:
    - Strip currency symbols and try numeric conversion
    - Parse date columns
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            # Try numeric
            cleaned = df[col].astype(str).str.replace(r"[$,€£%₹]", "", regex=True).str.strip()
            num = pd.to_numeric(cleaned, errors="coerce")
            if num.notna().mean() > 0.80:
                df[col] = num
                continue
            # Try datetime on date-keyword columns
            col_l = col.lower()
            if any(k in col_l for k in _DATE_KW):
                try:
                    df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                except Exception:
                    pass
    return df


def enrich_dataframe(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    """Add derived time columns for trend analysis."""
    df = df.copy()
    dc = col_map.get("date")
    if dc and pd.api.types.is_datetime64_any_dtype(df[dc]):
        df["_Year"]     = df[dc].dt.year
        df["_Month"]    = df[dc].dt.to_period("M").astype(str)
        df["_Quarter"]  = df[dc].dt.to_period("Q").astype(str)
        df["_MonthNum"] = df[dc].dt.month
        df["_MonthDt"]  = df[dc].dt.to_period("M").dt.to_timestamp()
    rc = col_map.get("revenue")
    pc = col_map.get("profit")
    if rc and pc and rc in df.columns and pc in df.columns:
        df["_Margin%"] = (df[pc] / df[rc].replace(0, np.nan) * 100).round(2)
    return df


def compute_kpis(df: pd.DataFrame, col_map: dict) -> dict:
    """Compute all key business KPIs from the dataset."""
    k = {}
    rc  = col_map.get("revenue");  pc  = col_map.get("profit")
    qc  = col_map.get("quantity"); rgc = col_map.get("region")
    cc  = col_map.get("category"); prc = col_map.get("product")
    cuc = col_map.get("customer")

    if rc and rc in df.columns:
        k["total_revenue"]    = round(float(df[rc].sum()), 2)
        k["avg_order_value"]  = round(float(df[rc].mean()), 2)
        k["max_order_value"]  = round(float(df[rc].max()), 2)
        k["min_order_value"]  = round(float(df[rc].min()), 2)
        k["revenue_std"]      = round(float(df[rc].std()), 2)

    if pc and pc in df.columns:
        k["total_profit"] = round(float(df[pc].sum()), 2)
        k["avg_profit"]   = round(float(df[pc].mean()), 2)

    if rc and pc and rc in df.columns and pc in df.columns and df[rc].sum() > 0:
        k["profit_margin_pct"] = round(float(df[pc].sum() / df[rc].sum() * 100), 2)

    if qc and qc in df.columns:
        k["total_units"] = int(df[qc].sum())

    if k.get("total_revenue") and k.get("total_units"):
        k["revenue_per_unit"] = round(k["total_revenue"] / k["total_units"], 2)

    if rgc and rc and rgc in df.columns and rc in df.columns:
        rg = df.groupby(rgc)[rc].sum()
        k["best_region"]      = str(rg.idxmax())
        k["worst_region"]     = str(rg.idxmin())
        k["best_region_rev"]  = round(float(rg.max()), 2)
        k["region_count"]     = int(df[rgc].nunique())

    if cc and rc and cc in df.columns and rc in df.columns:
        cg = df.groupby(cc)[rc].sum()
        k["best_category"]     = str(cg.idxmax())
        k["best_category_rev"] = round(float(cg.max()), 2)
        k["category_count"]    = int(df[cc].nunique())

    if prc and rc and prc in df.columns and rc in df.columns:
        pg = df.groupby(prc)[rc].sum()
        k["best_product"]     = str(pg.idxmax())
        k["best_product_rev"] = round(float(pg.max()), 2)
        k["product_count"]    = int(df[prc].nunique())

    if cuc and cuc in df.columns:
        k["customer_count"] = int(df[cuc].nunique())

    k["total_records"] = len(df)
    return k


def generate_full_profile(df: pd.DataFrame) -> dict:
    """
    Return a complete dataset profile dictionary:
    structure, column types, missing values, statistics, and KPIs.
    """
    col_map = detect_column_roles(df)
    df_e    = enrich_dataframe(df, col_map)

    # Column classification
    num_cols   = df.select_dtypes(include="number").columns.tolist()
    dt_cols    = df.select_dtypes(include=["datetime","datetimetz"]).columns.tolist()
    cat_cols   = df.select_dtypes(include="object").columns.tolist()

    # Missing values
    missing     = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    # Numeric statistics
    num_stats = {}
    for col in [c for c in num_cols if not c.startswith("_")]:
        s = df[col].describe()
        num_stats[col] = {
            "mean":   round(float(s["mean"]), 2),
            "std":    round(float(s["std"]),  2),
            "min":    round(float(s["min"]),  2),
            "25%":    round(float(s["25%"]),  2),
            "50%":    round(float(s["50%"]),  2),
            "75%":    round(float(s["75%"]),  2),
            "max":    round(float(s["max"]),  2),
        }

    # Categorical statistics
    cat_stats = {}
    for col in cat_cols:
        cat_stats[col] = {
            "unique":  int(df[col].nunique()),
            "top_val": str(df[col].mode().iloc[0]) if len(df[col].dropna()) else "—",
            "top_cnt": int(df[col].value_counts().iloc[0]) if len(df[col].dropna()) else 0,
        }

    # Date range
    date_range = None
    dc = col_map.get("date")
    if dc and pd.api.types.is_datetime64_any_dtype(df[dc]):
        vd = df[dc].dropna()
        if len(vd):
            date_range = {
                "min":  str(vd.min().date()),
                "max":  str(vd.max().date()),
                "span": int((vd.max() - vd.min()).days),
            }

    return {
        "row_count":   len(df),
        "col_count":   len(df.columns),
        "col_map":     col_map,
        "num_cols":    [c for c in num_cols   if not c.startswith("_")],
        "dt_cols":     dt_cols,
        "cat_cols":    cat_cols,
        "missing":     {c: int(v) for c, v in missing.items() if v > 0},
        "missing_pct": {c: float(v) for c, v in missing_pct.items() if v > 0},
        "missing_total": int(missing.sum()),
        "num_stats":   num_stats,
        "cat_stats":   cat_stats,
        "date_range":  date_range,
        "kpis":        compute_kpis(df_e, col_map),
    }


def profile_to_context_string(df: pd.DataFrame, profile: dict) -> str:
    """Convert profile to a rich text string for LLM context injection."""
    col_map = profile["col_map"]
    kpis    = profile["kpis"]
    lines   = [
        f"Dataset: {profile['row_count']} rows × {profile['col_count']} columns.",
        f"Columns: {', '.join(df.columns.tolist())}",
        f"Detected roles: {col_map}",
        f"Numeric columns: {', '.join(profile['num_cols'])}",
        f"Categorical columns: {', '.join(profile['cat_cols'])}",
    ]
    if profile["date_range"]:
        dr = profile["date_range"]
        lines.append(f"Date range: {dr['min']} to {dr['max']} ({dr['span']} days)")
    if kpis.get("total_revenue"):
        lines.append(f"Total Revenue: ${kpis['total_revenue']:,.2f}")
    if kpis.get("total_profit"):
        lines.append(f"Total Profit: ${kpis['total_profit']:,.2f}")
    if kpis.get("profit_margin_pct"):
        lines.append(f"Profit Margin: {kpis['profit_margin_pct']:.1f}%")
    if kpis.get("avg_order_value"):
        lines.append(f"Avg Order Value: ${kpis['avg_order_value']:,.2f}")
    if kpis.get("best_region"):
        lines.append(f"Best Region: {kpis['best_region']} (${kpis.get('best_region_rev',0):,.2f})")
    if kpis.get("best_category"):
        lines.append(f"Best Category: {kpis['best_category']} (${kpis.get('best_category_rev',0):,.2f})")
    if kpis.get("best_product"):
        lines.append(f"Best Product: {kpis['best_product']} (${kpis.get('best_product_rev',0):,.2f})")
    if profile["missing_total"] > 0:
        lines.append(f"Missing values: {profile['missing_total']} total — {profile['missing']}")
    else:
        lines.append("Missing values: None ✓")
    return "\n".join(lines)
