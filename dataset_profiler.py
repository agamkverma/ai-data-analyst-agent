"""
dataset_profiler.py — DataLensAI v2.0
Statistical Dataset Profiler

Responsibilities:
  - Per-column statistical summaries (mean, std, percentiles, skewness, kurtosis)
  - Cardinality and frequency analysis for categorical columns
  - Missing value detection and quality scoring
  - Outlier detection (IQR + Z-score)
  - Duplicate row detection
  - Correlation matrix between numeric columns
  - KPI card data construction

Author: Agam Kumar Verma
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from data_engine import DataEngine

log = logging.getLogger(__name__)


class DatasetProfiler:
    """
    Computes comprehensive statistical profiles for every column in the dataset.

    Parameters
    ----------
    engine : DataEngine
        A fully loaded and cleaned DataEngine instance.
    """

    def __init__(self, engine: DataEngine) -> None:
        self.engine  = engine
        self.df      = engine.df
        self._cache: dict = {}  # memoize expensive computations

    # ══════════════════════════════════════════════════════════════════════
    # PUBLIC: FULL PROFILE
    # ══════════════════════════════════════════════════════════════════════
    def full_profile(self) -> dict:
        """
        Returns a complete profile dict:
          - meta
          - quality
          - columns  (per-column stats)
          - correlations
        """
        if "full" in self._cache:
            return self._cache["full"]

        profile = {
            "meta":         self._meta(),
            "quality":      self.quality_score(),
            "columns":      self._all_column_stats(),
            "correlations": self._correlation_matrix(),
        }
        self._cache["full"] = profile
        return profile

    # ══════════════════════════════════════════════════════════════════════
    # META
    # ══════════════════════════════════════════════════════════════════════
    def _meta(self) -> dict:
        e = self.engine
        return {
            "filename":          e.filename,
            "rows":              e.row_count,
            "columns":           len(e.columns),
            "numeric_columns":   len(e.numeric_columns),
            "categorical_columns": len(e.categorical_columns),
            "datetime_columns":  len(e.datetime_columns),
            "memory_mb":         round(self.df.memory_usage(deep=True).sum() / 1024 / 1024, 3),
            "detected_fields":   e.detected_fields,
        }

    # ══════════════════════════════════════════════════════════════════════
    # QUALITY SCORE
    # ══════════════════════════════════════════════════════════════════════
    def quality_score(self) -> dict:
        """
        Composite quality score (0–100):
          - Null penalty:      -40 pts proportional to null rate
          - Duplicate penalty: -30 pts proportional to duplicate rate
          - Type coverage:     +bonus for having key field types
        """
        if "quality" in self._cache:
            return self._cache["quality"]

        df    = self.df
        total = df.size  # total cells

        # Null count
        nulls = int(df.isnull().sum().sum())

        # Duplicate rows
        dups = int(df.duplicated().sum())

        # Per-column null counts
        col_nulls = {col: int(df[col].isnull().sum()) for col in df.columns}

        # Score calculation
        null_penalty = (nulls / max(total, 1)) * 40
        dup_penalty  = (dups  / max(len(df),  1)) * 30
        score = max(0, round(100 - null_penalty - dup_penalty))

        result = {
            "score":      score,
            "nulls":      nulls,
            "duplicates": dups,
            "total_cells": total,
            "null_rate":  round(nulls / max(total, 1) * 100, 2),
            "dup_rate":   round(dups  / max(len(df),  1) * 100, 2),
            "col_nulls":  col_nulls,
            "grade":      self._score_to_grade(score),
        }
        self._cache["quality"] = result
        return result

    @staticmethod
    def _score_to_grade(score: int) -> str:
        if score >= 95: return "A+"
        if score >= 90: return "A"
        if score >= 80: return "B"
        if score >= 70: return "C"
        if score >= 50: return "D"
        return "F"

    # ══════════════════════════════════════════════════════════════════════
    # PER-COLUMN STATS
    # ══════════════════════════════════════════════════════════════════════
    def _all_column_stats(self) -> list[dict]:
        return [self._column_stats(col["name"], col["type"]) for col in self.engine.columns]

    def _column_stats(self, col: str, ctype: str) -> dict:
        series = self.df[col]
        null_count = int(series.isnull().sum())
        base = {
            "name":       col,
            "type":       ctype,
            "count":      int(series.notna().sum()),
            "null_count": null_count,
            "null_pct":   round(null_count / max(len(series), 1) * 100, 2),
        }

        if ctype == "numeric":
            base.update(self._numeric_stats(col))
        elif ctype == "categorical":
            base.update(self._categorical_stats(col))
        elif ctype == "datetime":
            base.update(self._datetime_stats(col))

        return base

    # ── Numeric ───────────────────────────────────────────────────────────
    def _numeric_stats(self, col: str) -> dict:
        s = pd.to_numeric(self.df[col], errors="coerce").dropna()
        if s.empty:
            return {}

        q1, q2, q3 = s.quantile([0.25, 0.50, 0.75])
        iqr         = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        outliers    = s[(s < lower_fence) | (s > upper_fence)]

        skewness = float(scipy_stats.skew(s))
        kurt     = float(scipy_stats.kurtosis(s))

        return {
            "sum":           round(float(s.sum()), 4),
            "mean":          round(float(s.mean()), 4),
            "median":        round(float(q2), 4),
            "std":           round(float(s.std()), 4),
            "min":           round(float(s.min()), 4),
            "max":           round(float(s.max()), 4),
            "q1":            round(float(q1), 4),
            "q3":            round(float(q3), 4),
            "iqr":           round(float(iqr), 4),
            "skewness":      round(skewness, 4),
            "kurtosis":      round(kurt, 4),
            "skew_label":    self._skew_label(skewness),
            "outlier_count": int(len(outliers)),
            "outlier_pct":   round(len(outliers) / max(len(s), 1) * 100, 2),
            "lower_fence":   round(float(lower_fence), 4),
            "upper_fence":   round(float(upper_fence), 4),
            "cv":            round(float(s.std() / s.mean()) if s.mean() != 0 else 0, 4),
        }

    @staticmethod
    def _skew_label(skew: float) -> str:
        if abs(skew) < 0.5:  return "symmetric"
        if abs(skew) < 1.0:  return "moderately skewed"
        return "highly skewed"

    # ── Categorical ───────────────────────────────────────────────────────
    def _categorical_stats(self, col: str) -> dict:
        s   = self.df[col].dropna().astype(str)
        vc  = s.value_counts()
        top = vc.head(20)

        return {
            "unique":        int(s.nunique()),
            "cardinality":   self._cardinality_label(int(s.nunique()), len(s)),
            "top_values":    [(k, int(v)) for k, v in top.items()],
            "top_value":     str(vc.index[0]) if len(vc) else "",
            "top_value_pct": round(float(vc.iloc[0]) / max(len(s), 1) * 100, 2) if len(vc) else 0,
            "avg_length":    round(s.str.len().mean(), 1),
        }

    @staticmethod
    def _cardinality_label(unique: int, total: int) -> str:
        ratio = unique / max(total, 1)
        if unique <= 5:      return "binary/low"
        if ratio < 0.05:     return "low"
        if ratio < 0.50:     return "medium"
        return "high"

    # ── Datetime ─────────────────────────────────────────────────────────
    def _datetime_stats(self, col: str) -> dict:
        s = pd.to_datetime(self.df[col], errors="coerce").dropna()
        if s.empty:
            return {}
        return {
            "min_date":   str(s.min().date()),
            "max_date":   str(s.max().date()),
            "range_days": int((s.max() - s.min()).days),
            "unique":     int(s.nunique()),
        }

    # ══════════════════════════════════════════════════════════════════════
    # CORRELATION MATRIX
    # ══════════════════════════════════════════════════════════════════════
    def _correlation_matrix(self) -> dict:
        num_cols = self.engine.numeric_columns
        if len(num_cols) < 2:
            return {}
        corr = (
            self.df[num_cols]
            .apply(pd.to_numeric, errors="coerce")
            .corr()
            .round(4)
        )
        return corr.to_dict()

    # ══════════════════════════════════════════════════════════════════════
    # PUBLIC: GET SINGLE COLUMN STATS
    # ══════════════════════════════════════════════════════════════════════
    def get_column_stats(self, col: str) -> dict:
        """Return stats for a single column by name."""
        col_meta = next((c for c in self.engine.columns if c["name"] == col), None)
        if not col_meta:
            return {}
        return self._column_stats(col, col_meta["type"])

    def get_numeric_series_stats(self, col: str) -> dict:
        """Fast access to numeric stats for a specific column."""
        return self._numeric_stats(col)

    def get_top_values(self, col: str, n: int = 10) -> list[tuple[str, int]]:
        """Top-N value counts for a categorical column."""
        if col not in self.df.columns:
            return []
        vc = self.df[col].value_counts().head(n)
        return [(str(k), int(v)) for k, v in vc.items()]

    # ══════════════════════════════════════════════════════════════════════
    # KPI CARD BUILDER
    # ══════════════════════════════════════════════════════════════════════
    def build_kpis(self) -> list[dict]:
        """
        Build KPI card data from detected fields.
        Returns a list of up to 6 KPI dicts compatible with the frontend.
        """
        e    = self.engine
        df   = self.df
        kpis = []

        PAL_GREEN  = "linear-gradient(135deg,#10B981,#34D399)"
        PAL_BLUE   = "linear-gradient(135deg,#3B82F6,#38BDF8)"
        PAL_PURPLE = "linear-gradient(135deg,#7C3AED,#A78BFA)"
        PAL_AMBER  = "linear-gradient(135deg,#F59E0B,#F97316)"
        PAL_PINK   = "linear-gradient(135deg,#EC4899,#A78BFA)"

        # 1. Total Revenue
        rev = e.revenue_col
        if rev:
            stats = self._numeric_stats(rev)
            kpis.append({
                "id":   "total_revenue",
                "lbl":  "Total " + rev.replace("_", " "),
                "val":  _fmt(stats.get("sum", 0), "$"),
                "ico":  "💰",
                "kc":   PAL_GREEN,
                "kb":   "rgba(16,185,129,.14)",
                "tr":   "up",
                "tl":   "↑ Avg " + _fmt(stats.get("mean", 0), "$"),
                "sub":  "Max: " + _fmt(stats.get("max", 0), "$"),
                "raw":  stats.get("sum", 0),
            })

        # 2. Total Profit + Margin
        prf = e.profit_col
        if prf:
            pstats = self._numeric_stats(prf)
            rev_sum = self.get_numeric_series_stats(rev)["sum"] if rev else 0
            margin  = round(pstats.get("sum", 0) / rev_sum * 100, 1) if rev_sum else None
            kpis.append({
                "id":   "total_profit",
                "lbl":  "Total Profit",
                "val":  _fmt(pstats.get("sum", 0), "$"),
                "ico":  "📈",
                "kc":   PAL_BLUE,
                "kb":   "rgba(59,130,246,.14)",
                "tr":   "up",
                "tl":   "↑ " + _fmt(pstats.get("mean", 0), "$") + " avg",
                "sub":  ("Margin: " + str(margin) + "%") if margin else "—",
                "raw":  pstats.get("sum", 0),
            })

            if margin is not None:
                kpis.append({
                    "id":   "profit_margin",
                    "lbl":  "Profit Margin",
                    "val":  str(margin) + "%",
                    "ico":  "🎯",
                    "kc":   PAL_GREEN if margin >= 20 else PAL_AMBER,
                    "kb":   "rgba(16,185,129,.14)" if margin >= 20 else "rgba(245,158,11,.14)",
                    "tr":   "up" if margin >= 20 else "flat",
                    "tl":   "↑ Above 20%" if margin >= 20 else "↗ Review costs",
                    "sub":  "Industry avg ≈ 20%",
                    "raw":  margin,
                })

        # 3. Top Category
        cat = e.category_col
        if cat and rev:
            cat_data = e.group_sum(cat, rev, top_n=1)
            if cat_data:
                top_cat, top_cat_val = cat_data[0]
                rev_sum = self._numeric_stats(rev).get("sum", 0)
                share   = round(top_cat_val / rev_sum * 100, 1) if rev_sum else 0
                kpis.append({
                    "id":   "top_category",
                    "lbl":  "Top Category",
                    "val":  _trunc(top_cat, 12),
                    "ico":  "🏷",
                    "kc":   PAL_AMBER,
                    "kb":   "rgba(245,158,11,.14)",
                    "tr":   "up",
                    "tl":   f"↑ {share}% share",
                    "sub":  _fmt(top_cat_val, "$") + " revenue",
                    "raw":  top_cat_val,
                })

        # 4. Best Region
        reg = e.region_col
        if reg and rev:
            reg_data = e.group_sum(reg, rev, top_n=1)
            if reg_data:
                top_reg, top_reg_val = reg_data[0]
                rev_sum = self._numeric_stats(rev).get("sum", 0)
                share   = round(top_reg_val / rev_sum * 100, 1) if rev_sum else 0
                kpis.append({
                    "id":   "best_region",
                    "lbl":  "Best Region",
                    "val":  _trunc(top_reg, 12),
                    "ico":  "🌍",
                    "kc":   PAL_PINK,
                    "kb":   "rgba(236,72,153,.14)",
                    "tr":   "up",
                    "tl":   f"↑ {share}% share",
                    "sub":  _fmt(top_reg_val, "$") + " revenue",
                    "raw":  top_reg_val,
                })

        # 5. Data Quality
        quality = self.quality_score()
        kpis.append({
            "id":   "data_quality",
            "lbl":  "Data Quality",
            "val":  str(quality["score"]) + "%",
            "ico":  "✓" if quality["score"] >= 80 else "⚠",
            "kc":   PAL_GREEN if quality["score"] >= 80 else PAL_AMBER,
            "kb":   "rgba(16,185,129,.13)" if quality["score"] >= 80 else "rgba(245,158,11,.13)",
            "tr":   "up" if quality["score"] >= 80 else "flat",
            "tl":   ("✓ No nulls" if quality["nulls"] == 0 else f"⚠ {quality['nulls']} nulls"),
            "sub":  f"{e.row_count:,} rows · {len(e.columns)} cols",
            "raw":  quality["score"],
        })

        # 6. Total Records
        kpis.append({
            "id":   "total_records",
            "lbl":  "Total Records",
            "val":  f"{e.row_count:,}",
            "ico":  "🗄",
            "kc":   PAL_PURPLE,
            "kb":   "rgba(124,58,237,.14)",
            "tr":   "flat",
            "tl":   f"→ {len(e.columns)} columns",
            "sub":  f"{len(e.numeric_columns)} numeric · {len(e.categorical_columns)} categorical",
            "raw":  e.row_count,
        })

        return kpis[:6]


# ── Utility formatters ────────────────────────────────────────────────────────
def _fmt(n: float, prefix: str = "") -> str:
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "—"
    a = abs(n)
    if a >= 1e9: return f"{prefix}{n/1e9:.2f}B"
    if a >= 1e6: return f"{prefix}{n/1e6:.2f}M"
    if a >= 1e3: return f"{prefix}{n/1e3:.1f}K"
    return f"{prefix}{n:.2f}"


def _trunc(s: str, n: int = 14) -> str:
    return s if len(s) <= n else s[:n-1] + "…"
