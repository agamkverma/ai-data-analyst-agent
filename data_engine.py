"""
data_engine.py — DataLensAI v2.0
Core Data Pipeline Engine

Responsibilities:
  - CSV ingestion from bytes, file path, or DataFrame
  - Column type inference (numeric / categorical / datetime)
  - Smart field detection (revenue, profit, date, category, region)
  - Data cleaning (null handling, type coercion, deduplication)
  - Aggregation helpers used by all other modules
  - Export to CSV bytes

Author: Agam Kumar Verma
"""

from __future__ import annotations

import io
import logging
import re
from typing import Any, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Column name patterns for auto-field detection ───────────────────────────
_PAT = {
    "revenue":  re.compile(r"revenue|sales|amount|total|price|income|gmv|earning|turnover", re.I),
    "profit":   re.compile(r"profit|margin|net|gain|ebit|ebitda|gross_profit", re.I),
    "date":     re.compile(r"date|time|month|year|week|period|quarter|day", re.I),
    "category": re.compile(r"category|segment|product|type|class|group|brand|department|dept|item", re.I),
    "region":   re.compile(r"region|area|zone|country|city|state|location|territory|branch|division", re.I),
    "quantity": re.compile(r"quantity|qty|units|count|volume|pieces", re.I),
}

# ── Numeric value pattern (handles $, €, £, commas, percent) ────────────────
_NUM_CLEAN = re.compile(r"[$€£,\s%]")


class DataEngine:
    """
    Central data store and processing engine for a single uploaded dataset.

    Attributes
    ----------
    filename : str
        Original filename.
    df : pd.DataFrame
        Cleaned, type-coerced DataFrame.
    columns : list[ColumnMeta]
        Per-column metadata (name, inferred type).
    detected_fields : dict
        Auto-detected semantic fields (revenue, profit, date, etc.).
    row_count : int
        Number of rows after cleaning.
    """

    def __init__(self, filename: str = "dataset.csv", max_rows: int = 500_000) -> None:
        self.filename  = filename
        self.max_rows  = max_rows
        self.df        = pd.DataFrame()
        self.columns: list[dict] = []
        self.detected_fields: dict[str, Optional[str]] = {}
        self.row_count  = 0
        self._raw_count = 0   # rows before dedup / null filtering

    # ══════════════════════════════════════════════════════════════════════
    # LOAD
    # ══════════════════════════════════════════════════════════════════════
    def load_from_bytes(self, content: bytes) -> "DataEngine":
        """Parse CSV from raw bytes (uploaded file)."""
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = content.decode("latin-1")

        # Try multiple common delimiters
        for sep in [",", ";", "\t", "|"]:
            try:
                df = pd.read_csv(
                    io.StringIO(text),
                    sep=sep,
                    low_memory=False,
                    nrows=self.max_rows,
                    on_bad_lines="skip",
                )
                if len(df.columns) > 1:
                    break
            except Exception:
                continue

        if df.empty or len(df.columns) < 1:
            raise ValueError("Could not parse CSV — check delimiter and encoding.")

        self._raw_count = len(df)
        self.df         = df
        log.info(f"Loaded {self._raw_count} rows × {len(df.columns)} cols from '{self.filename}'")
        return self

    def load_from_path(self, path: str) -> "DataEngine":
        """Parse CSV from file path."""
        with open(path, "rb") as f:
            return self.load_from_bytes(f.read())

    def load_from_dataframe(self, df: pd.DataFrame, filename: str = "dataframe.csv") -> "DataEngine":
        """Load directly from an existing DataFrame."""
        self.filename   = filename
        self._raw_count = len(df)
        self.df         = df.copy()
        return self

    # ══════════════════════════════════════════════════════════════════════
    # CLEAN
    # ══════════════════════════════════════════════════════════════════════
    def clean(self) -> "DataEngine":
        """
        Full cleaning pipeline:
          1. Strip whitespace from column names
          2. Drop fully-empty rows and columns
          3. Coerce numeric columns (handle currency symbols)
          4. Parse datetime columns
          5. Infer and attach column types
          6. Detect semantic fields
          7. Compute row count
        """
        df = self.df

        # 1. Clean column names
        df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]

        # 2. Drop empty rows / columns
        df.dropna(how="all", inplace=True)
        df.dropna(axis=1, how="all", inplace=True)

        # 3. Coerce numeric columns
        df = self._coerce_numerics(df)

        # 4. Parse datetime columns
        df = self._coerce_datetimes(df)

        self.df        = df
        self.row_count = len(df)

        # 5. Build column metadata
        self._build_column_meta()

        # 6. Detect semantic fields
        self._detect_fields()

        log.info(
            f"Cleaned: {self.row_count} rows | "
            f"{len(self.columns)} cols | "
            f"fields={self.detected_fields}"
        )
        return self

    # ── PRIVATE: numeric coercion ──────────────────────────────────────────
    def _coerce_numerics(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if df[col].dtype == object:
                cleaned = df[col].astype(str).str.strip().str.replace(_NUM_CLEAN, "", regex=True)
                converted = pd.to_numeric(cleaned, errors="coerce")
                # Accept coercion if ≥ 80% of non-null values parse as numbers
                non_null = df[col].notna().sum()
                if non_null > 0 and converted.notna().sum() / non_null >= 0.80:
                    df[col] = converted
        return df

    # ── PRIVATE: datetime coercion ─────────────────────────────────────────
    def _coerce_datetimes(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if df[col].dtype == object:
                sample = df[col].dropna().head(30).astype(str)
                dt_hits = sum(
                    1 for v in sample
                    if re.search(r"\d{4}-\d{2}|\d{2}/\d{2}/\d{4}|^20\d{2}$", v)
                )
                if len(sample) > 0 and dt_hits / len(sample) >= 0.4:
                    try:
                        df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                    except Exception:
                        pass
        return df

    # ── PRIVATE: column metadata ───────────────────────────────────────────
    def _build_column_meta(self) -> None:
        self.columns = []
        for col in self.df.columns:
            dtype = self.df[col].dtype
            if pd.api.types.is_datetime64_any_dtype(dtype):
                ctype = "datetime"
            elif pd.api.types.is_numeric_dtype(dtype):
                ctype = "numeric"
            else:
                ctype = "categorical"
            self.columns.append({"name": col, "type": ctype})

    # ── PRIVATE: semantic field detection ─────────────────────────────────
    def _detect_fields(self) -> None:
        num_cols = [c["name"] for c in self.columns if c["type"] == "numeric"]
        cat_cols = [c["name"] for c in self.columns if c["type"] == "categorical"]
        dt_cols  = [c["name"] for c in self.columns if c["type"] == "datetime"]

        self.detected_fields = {k: None for k in _PAT}

        # Revenue — match by name, then pick numeric with highest sum
        rev = next((c for c in num_cols if _PAT["revenue"].search(c)), None)
        if not rev and num_cols:
            rev = max(num_cols, key=lambda c: self.df[c].sum(skipna=True))
        self.detected_fields["revenue"] = rev

        # Profit
        self.detected_fields["profit"] = next(
            (c for c in num_cols if _PAT["profit"].search(c)), None
        )

        # Quantity
        self.detected_fields["quantity"] = next(
            (c for c in num_cols if _PAT["quantity"].search(c)), None
        )

        # Date — datetime cols first, then categorical with date-like names
        self.detected_fields["date"] = (
            dt_cols[0] if dt_cols
            else next((c for c in cat_cols if _PAT["date"].search(c)), None)
        )

        # Category
        self.detected_fields["category"] = next(
            (c for c in cat_cols if _PAT["category"].search(c)), cat_cols[0] if cat_cols else None
        )

        # Region
        self.detected_fields["region"] = next(
            (c for c in cat_cols if _PAT["region"].search(c)), None
        )

    # ══════════════════════════════════════════════════════════════════════
    # AGGREGATION HELPERS
    # ══════════════════════════════════════════════════════════════════════
    def group_sum(
        self,
        group_col: str,
        value_col: str,
        top_n: int = 10,
        ascending: bool = False,
    ) -> list[tuple[str, float]]:
        """Aggregate value_col by group_col, return sorted list of (label, value)."""
        if group_col not in self.df.columns or value_col not in self.df.columns:
            return []
        result = (
            self.df.groupby(group_col, observed=True)[value_col]
            .sum()
            .sort_values(ascending=ascending)
            .head(top_n)
        )
        return [(str(k), float(v)) for k, v in result.items()]

    def monthly_trend(
        self,
        date_col: str,
        value_col: str,
        last_n: int = 18,
    ) -> list[tuple[str, float]]:
        """Aggregate value_col by month derived from date_col."""
        if date_col not in self.df.columns or value_col not in self.df.columns:
            return []
        df = self.df[[date_col, value_col]].copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        df["_month"] = df[date_col].dt.to_period("M").astype(str)
        result = (
            df.groupby("_month")[value_col]
            .sum()
            .sort_index()
            .tail(last_n)
        )
        return [(k, float(v)) for k, v in result.items()]

    def numeric_col_series(self, col: str) -> pd.Series:
        """Return a clean numeric series for a column."""
        if col not in self.df.columns:
            return pd.Series(dtype=float)
        return pd.to_numeric(self.df[col], errors="coerce").dropna()

    def value_counts_top(self, col: str, top_n: int = 20) -> list[tuple[str, int]]:
        """Top-N value counts for a categorical column."""
        if col not in self.df.columns:
            return []
        return [(str(k), int(v)) for k, v in self.df[col].value_counts().head(top_n).items()]

    def cross_tab(
        self,
        row_col: str,
        col_col: str,
        value_col: str,
    ) -> dict:
        """Pivot table: rows = row_col, cols = col_col, values = sum of value_col."""
        if any(c not in self.df.columns for c in [row_col, col_col, value_col]):
            return {}
        pivot = self.df.pivot_table(
            index=row_col,
            columns=col_col,
            values=value_col,
            aggfunc="sum",
            fill_value=0,
        )
        return pivot.to_dict()

    # ══════════════════════════════════════════════════════════════════════
    # CONVENIENCE PROPERTIES
    # ══════════════════════════════════════════════════════════════════════
    @property
    def numeric_columns(self) -> list[str]:
        return [c["name"] for c in self.columns if c["type"] == "numeric"]

    @property
    def categorical_columns(self) -> list[str]:
        return [c["name"] for c in self.columns if c["type"] == "categorical"]

    @property
    def datetime_columns(self) -> list[str]:
        return [c["name"] for c in self.columns if c["type"] == "datetime"]

    @property
    def revenue_col(self) -> Optional[str]:
        return self.detected_fields.get("revenue")

    @property
    def profit_col(self) -> Optional[str]:
        return self.detected_fields.get("profit")

    @property
    def date_col(self) -> Optional[str]:
        return self.detected_fields.get("date")

    @property
    def category_col(self) -> Optional[str]:
        return self.detected_fields.get("category")

    @property
    def region_col(self) -> Optional[str]:
        return self.detected_fields.get("region")

    # ══════════════════════════════════════════════════════════════════════
    # EXPORT
    # ══════════════════════════════════════════════════════════════════════
    def export_csv(self) -> bytes:
        """Export cleaned DataFrame as CSV bytes."""
        return self.df.to_csv(index=False).encode("utf-8")

    def sample(self, n: int = 5) -> list[dict]:
        """Return n sample rows as list of dicts."""
        return self.df.head(n).to_dict(orient="records")

    def __repr__(self) -> str:
        return (
            f"DataEngine(file='{self.filename}', "
            f"rows={self.row_count}, cols={len(self.columns)}, "
            f"fields={list(self.detected_fields.keys())})"
        )
