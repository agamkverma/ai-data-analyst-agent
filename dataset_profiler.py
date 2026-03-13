"""dataset_profiler.py — Dataset statistics, KPI computation, context string."""

import pandas as pd
import numpy as np


def generate_profile(df: pd.DataFrame, col_map: dict) -> dict:
    p = {}
    p["row_count"] = len(df)
    p["col_count"] = len(df.columns)
    missing = df.isnull().sum()
    p["missing"] = {c: int(v) for c, v in missing.items() if v > 0}
    p["missing_total"] = int(missing.sum())
    p["dtypes_summary"] = {
        "numeric":  list(df.select_dtypes(include="number").columns),
        "datetime": list(df.select_dtypes(include=["datetime","datetimetz"]).columns),
        "text":     list(df.select_dtypes(include="object").columns),
    }
    num_cols = [c for c in df.select_dtypes(include="number").columns if not c.startswith("_")]
    p["numeric_stats"] = df[num_cols].describe().round(2).to_dict() if num_cols else {}

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    p["categorical_stats"] = {
        col: {"unique": int(df[col].nunique()), "top5": df[col].value_counts().head(5).to_dict()}
        for col in cat_cols
    }

    dc = col_map.get("date")
    if dc and pd.api.types.is_datetime64_any_dtype(df[dc]):
        vd = df[dc].dropna()
        if len(vd):
            p["date_range"] = {
                "min": str(vd.min().date()),
                "max": str(vd.max().date()),
                "span": (vd.max() - vd.min()).days,
            }
    else:
        p["date_range"] = None

    p["kpis"] = _kpis(df, col_map)
    return p


def _kpis(df, col_map):
    k = {}
    rc = col_map.get("revenue"); pc = col_map.get("profit")
    qc = col_map.get("quantity"); rgc = col_map.get("region")
    cc = col_map.get("category"); prc = col_map.get("product")

    if rc:  k["total_revenue"] = round(df[rc].sum(), 2)
    if pc:  k["total_profit"]  = round(df[pc].sum(), 2)
    if rc and pc and df[rc].sum():
        k["profit_margin_pct"] = round(df[pc].sum() / df[rc].sum() * 100, 2)
    if qc:  k["total_units"] = int(df[qc].sum())
    if rgc and rc: k["best_region"]   = df.groupby(rgc)[rc].sum().idxmax()
    if cc  and rc: k["best_category"] = df.groupby(cc)[rc].sum().idxmax()
    if prc and rc: k["best_product"]  = df.groupby(prc)[rc].sum().idxmax()
    k["total_records"] = len(df)
    return k


def profile_to_context(df: pd.DataFrame, col_map: dict, profile: dict) -> str:
    lines = [
        f"Dataset: {profile['row_count']} rows, {profile['col_count']} columns.",
        f"Columns: {', '.join(df.columns.tolist())}",
        f"Detected roles: {col_map}",
    ]
    k = profile.get("kpis", {})
    if k.get("total_revenue"): lines.append(f"Total Revenue: ${k['total_revenue']:,.2f}")
    if k.get("total_profit"):  lines.append(f"Total Profit:  ${k['total_profit']:,.2f}")
    if k.get("profit_margin_pct"): lines.append(f"Profit Margin: {k['profit_margin_pct']}%")
    if k.get("best_region"):   lines.append(f"Best Region:   {k['best_region']}")
    if k.get("best_category"): lines.append(f"Best Category: {k['best_category']}")
    dr = profile.get("date_range")
    if dr: lines.append(f"Date Range: {dr['min']} to {dr['max']}")
    return "\n".join(lines)
