"""
prediction_engine.py — DataLensAI v2.0
Machine Learning Prediction & Forecasting Engine

Responsibilities:
  - Time-series revenue/sales forecasting (linear, moving average, exponential)
  - Anomaly detection via IQR + Z-score methods
  - Correlation matrix computation
  - Linear regression between any two numeric columns
  - Growth rate analysis
  - Trend direction and confidence scoring

All ML is done with scikit-learn and numpy — no external ML service required.
Designed to degrade gracefully when insufficient data exists.

Author: Agam Kumar Verma
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_absolute_error
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

from data_engine import DataEngine
from dataset_profiler import _fmt

log = logging.getLogger(__name__)

# ── Minimum data requirements ─────────────────────────────────────────────────
MIN_POINTS_FORECAST = 6   # minimum time periods for forecasting
MIN_POINTS_CORR     = 10  # minimum rows for correlation


class PredictionEngine:
    """
    ML forecasting and anomaly detection for a DataEngine session.

    Parameters
    ----------
    engine : DataEngine
        A fully cleaned DataEngine instance.
    """

    def __init__(self, engine: DataEngine) -> None:
        self.engine = engine
        self.df     = engine.df

    # ══════════════════════════════════════════════════════════════════════
    # FORECAST — main entry point
    # ══════════════════════════════════════════════════════════════════════
    def forecast(
        self,
        target_col: str,
        periods:    int = 6,
        method:     str = "linear",
    ) -> dict:
        """
        Forecast future values for target_col.

        Parameters
        ----------
        target_col : str   Column to forecast (must be numeric).
        periods    : int   Number of future periods to predict.
        method     : str   "linear" | "moving_avg" | "exponential"

        Returns
        -------
        dict with keys:
          historical    — list of {period, value} actual data points
          forecast      — list of {period, value, lower, upper} predicted points
          trend         — "up" | "down" | "flat"
          trend_pct     — % change from first to last historical
          r2            — model fit score (for linear)
          summary       — human-readable string
          method        — method used
        """
        if not SKLEARN_OK:
            return self._error("scikit-learn is not installed. Run: pip install scikit-learn")

        series = self._build_time_series(target_col)
        if series is None or len(series) < MIN_POINTS_FORECAST:
            return self._error(
                f"Not enough time-series data to forecast '{target_col}'. "
                f"Need at least {MIN_POINTS_FORECAST} time periods (found {len(series) if series else 0})."
            )

        labels = [str(s[0]) for s in series]
        values = [float(s[1]) for s in series]

        if method == "moving_avg":
            return self._moving_avg_forecast(target_col, labels, values, periods)
        if method == "exponential":
            return self._exponential_forecast(target_col, labels, values, periods)
        return self._linear_forecast(target_col, labels, values, periods)

    # ── Linear Regression Forecast ────────────────────────────────────────
    def _linear_forecast(
        self,
        col:     str,
        labels:  list[str],
        values:  list[float],
        periods: int,
    ) -> dict:
        n = len(values)
        X = np.arange(n).reshape(-1, 1)
        y = np.array(values)

        model = LinearRegression()
        model.fit(X, y)

        # Historical predictions (for R² and fit line)
        y_pred   = model.predict(X)
        r2       = float(r2_score(y, y_pred))
        mae      = float(mean_absolute_error(y, y_pred))
        residual = float(np.std(y - y_pred))

        # Future periods
        X_future  = np.arange(n, n + periods).reshape(-1, 1)
        y_future  = model.predict(X_future)
        conf_band = residual * 1.96  # 95% confidence interval (Gaussian approx)

        # Build future period labels
        future_labels = self._extend_labels(labels, periods)

        historical = [{"period": l, "value": round(v, 2)} for l, v in zip(labels, values)]
        forecast   = [
            {
                "period": future_labels[i],
                "value":  round(float(y_future[i]), 2),
                "lower":  round(float(y_future[i]) - conf_band, 2),
                "upper":  round(float(y_future[i]) + conf_band, 2),
            }
            for i in range(periods)
        ]

        trend, trend_pct = self._calc_trend(values)
        slope = float(model.coef_[0])
        summary = (
            f"Linear trend for {col.replace('_',' ')}: "
            f"slope={_fmt(slope)}/period, R²={r2:.3f}, "
            f"forecast next {periods} periods from {_fmt(float(y_future[0]),'$')} "
            f"to {_fmt(float(y_future[-1]),'$')}."
        )

        return {
            "historical":   historical,
            "forecast":     forecast,
            "trend":        trend,
            "trend_pct":    trend_pct,
            "r2":           round(r2, 4),
            "mae":          round(mae, 4),
            "slope":        round(slope, 4),
            "summary":      summary,
            "method":       "linear_regression",
            "column":       col,
            "periods":      periods,
        }

    # ── Moving Average Forecast ───────────────────────────────────────────
    def _moving_avg_forecast(
        self,
        col:     str,
        labels:  list[str],
        values:  list[float],
        periods: int,
        window:  int = 3,
    ) -> dict:
        values_arr  = np.array(values, dtype=float)
        window      = min(window, len(values) - 1)
        rolling_avg = np.convolve(values_arr, np.ones(window) / window, mode="valid")
        last_avg    = float(rolling_avg[-1])
        last_std    = float(np.std(values_arr[-window:]))

        future_labels = self._extend_labels(labels, periods)
        forecast = [
            {
                "period": future_labels[i],
                "value":  round(last_avg, 2),
                "lower":  round(last_avg - last_std * 1.96, 2),
                "upper":  round(last_avg + last_std * 1.96, 2),
            }
            for i in range(periods)
        ]

        trend, trend_pct = self._calc_trend(values)
        return {
            "historical":   [{"period": l, "value": round(v, 2)} for l, v in zip(labels, values)],
            "forecast":     forecast,
            "trend":        trend,
            "trend_pct":    trend_pct,
            "r2":           None,
            "mae":          None,
            "slope":        None,
            "summary":      f"Moving average ({window}-period) for {col}: projected value ≈ {_fmt(last_avg,'$')}/period.",
            "method":       f"moving_avg_{window}",
            "column":       col,
            "periods":      periods,
        }

    # ── Exponential Smoothing Forecast ────────────────────────────────────
    def _exponential_forecast(
        self,
        col:     str,
        labels:  list[str],
        values:  list[float],
        periods: int,
        alpha:   float = 0.3,
    ) -> dict:
        smoothed = [values[0]]
        for v in values[1:]:
            smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])

        last = float(smoothed[-1])
        std  = float(np.std(values[-4:])) if len(values) >= 4 else float(np.std(values))

        future_labels = self._extend_labels(labels, periods)
        forecast = [
            {
                "period": future_labels[i],
                "value":  round(last, 2),
                "lower":  round(last - std * 1.96, 2),
                "upper":  round(last + std * 1.96, 2),
            }
            for i in range(periods)
        ]

        trend, trend_pct = self._calc_trend(values)
        return {
            "historical":   [{"period": l, "value": round(v, 2)} for l, v in zip(labels, values)],
            "forecast":     forecast,
            "trend":        trend,
            "trend_pct":    trend_pct,
            "r2":           None,
            "mae":          None,
            "slope":        None,
            "summary":      f"Exponential smoothing (α={alpha}) for {col}: projected value ≈ {_fmt(last,'$')}/period.",
            "method":       f"exponential_alpha_{alpha}",
            "column":       col,
            "periods":      periods,
        }

    # ══════════════════════════════════════════════════════════════════════
    # ANOMALY DETECTION
    # ══════════════════════════════════════════════════════════════════════
    def detect_anomalies(self, target_col: Optional[str] = None) -> dict:
        """
        Detect outliers across numeric columns using IQR and Z-score methods.

        Parameters
        ----------
        target_col : Optional[str]
            If given, restrict to that column. Otherwise, scan all numeric cols.

        Returns
        -------
        dict with:
          anomalies — list of {column, method, count, pct, indices, sample_values}
          summary   — text summary
        """
        cols = [target_col] if target_col else self.engine.numeric_columns[:8]
        results = []

        for col in cols:
            series = pd.to_numeric(self.df[col], errors="coerce").dropna()
            if len(series) < 10:
                continue

            # IQR method
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr     = q3 - q1
            lower   = q1 - 1.5 * iqr
            upper   = q3 + 1.5 * iqr
            iqr_mask   = (series < lower) | (series > upper)
            iqr_count  = int(iqr_mask.sum())

            # Z-score method
            z_scores = np.abs(scipy_stats.zscore(series))
            z_mask   = z_scores > 3.0
            z_count  = int(z_mask.sum())

            if iqr_count > 0:
                sample_vals = series[iqr_mask].head(5).round(2).tolist()
                results.append({
                    "column":        col,
                    "method":        "IQR",
                    "count":         iqr_count,
                    "pct":           round(iqr_count / len(series) * 100, 2),
                    "lower_fence":   round(float(lower), 2),
                    "upper_fence":   round(float(upper), 2),
                    "sample_values": [float(v) for v in sample_vals],
                    "severity":      "high" if iqr_count / len(series) > 0.05 else "low",
                })

            if z_count > 0:
                results.append({
                    "column":        col,
                    "method":        "Z-score",
                    "count":         z_count,
                    "pct":           round(z_count / len(series) * 100, 2),
                    "sample_values": series[z_mask].head(3).round(2).tolist(),
                    "severity":      "high" if z_count > 5 else "low",
                })

        total_anomalies = sum(r["count"] for r in results if r["method"] == "IQR")
        summary = (
            f"Detected {total_anomalies} outliers (IQR method) across "
            f"{len(set(r['column'] for r in results))} column(s)."
            if results else
            "No significant outliers detected."
        )

        return {
            "anomalies": results,
            "total":     total_anomalies,
            "summary":   summary,
            "method":    "IQR + Z-score",
        }

    # ══════════════════════════════════════════════════════════════════════
    # CORRELATION MATRIX
    # ══════════════════════════════════════════════════════════════════════
    def correlation_matrix(self) -> dict:
        """
        Pearson correlation matrix for all numeric columns.

        Returns
        -------
        dict:
          matrix     — 2D correlation table
          top_pairs  — top 5 strongest correlations with context labels
          columns    — list of column names
        """
        num_cols = self.engine.numeric_columns
        if len(num_cols) < 2:
            return {"matrix": {}, "top_pairs": [], "columns": [], "error": "Need ≥ 2 numeric columns."}

        df_num = self.df[num_cols].apply(pd.to_numeric, errors="coerce")
        if len(df_num.dropna()) < MIN_POINTS_CORR:
            return {"matrix": {}, "top_pairs": [], "columns": num_cols, "error": "Not enough data rows."}

        corr = df_num.corr().round(4)
        matrix = corr.to_dict()

        # Top pairs
        pairs = []
        processed = set()
        for c1 in num_cols:
            for c2 in num_cols:
                if c1 == c2 or (c2, c1) in processed:
                    continue
                processed.add((c1, c2))
                val = corr.loc[c1, c2]
                if not np.isnan(val):
                    pairs.append({
                        "col1":        c1,
                        "col2":        c2,
                        "r":           round(float(val), 4),
                        "r_abs":       abs(float(val)),
                        "label":       self._corr_label(float(val)),
                        "direction":   "positive" if val > 0 else "negative",
                    })

        top_pairs = sorted(pairs, key=lambda x: -x["r_abs"])[:5]
        for p in top_pairs:
            del p["r_abs"]

        return {
            "matrix":    matrix,
            "top_pairs": top_pairs,
            "columns":   num_cols,
        }

    @staticmethod
    def _corr_label(r: float) -> str:
        ar = abs(r)
        if ar >= 0.9:  return "very strong"
        if ar >= 0.7:  return "strong"
        if ar >= 0.5:  return "moderate"
        if ar >= 0.3:  return "weak"
        return "negligible"

    # ══════════════════════════════════════════════════════════════════════
    # LINEAR REGRESSION (two columns)
    # ══════════════════════════════════════════════════════════════════════
    def regression(self, x_col: str, y_col: str) -> dict:
        """
        Fit OLS linear regression: y ~ x.

        Returns
        -------
        dict with slope, intercept, r2, p_value, points
        """
        if not SKLEARN_OK:
            return self._error("scikit-learn required.")

        df_clean = self.df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(df_clean) < 10:
            return self._error(f"Not enough data points for regression ({len(df_clean)} rows).")

        X = df_clean[[x_col]].values
        y = df_clean[y_col].values

        model = LinearRegression()
        model.fit(X, y)

        y_pred = model.predict(X)
        r2     = float(r2_score(y, y_pred))
        mae    = float(mean_absolute_error(y, y_pred))

        # p-value via scipy
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(X.flatten(), y)

        # Sample of points for scatter plot
        sample_df = df_clean.sample(min(300, len(df_clean)), random_state=42)
        points    = [{"x": round(float(r[x_col]), 4), "y": round(float(r[y_col]), 4)} for _, r in sample_df.iterrows()]

        return {
            "x_col":     x_col,
            "y_col":     y_col,
            "slope":     round(float(model.coef_[0]), 6),
            "intercept": round(float(model.intercept_), 6),
            "r2":        round(r2, 4),
            "mae":       round(mae, 4),
            "p_value":   round(float(p_value), 6),
            "significant": p_value < 0.05,
            "points":    points,
            "n":         len(df_clean),
            "summary": (
                f"For every 1-unit increase in {x_col}, {y_col} changes by "
                f"{_fmt(float(model.coef_[0]))} (R²={r2:.3f}, "
                f"{'statistically significant' if p_value < 0.05 else 'not significant'}, "
                f"p={p_value:.4f})."
            ),
        }

    # ══════════════════════════════════════════════════════════════════════
    # GROWTH RATE ANALYSIS
    # ══════════════════════════════════════════════════════════════════════
    def growth_analysis(self, col: str) -> dict:
        """Compute period-over-period growth rates from a time-series column."""
        series = self._build_time_series(col)
        if not series or len(series) < 3:
            return self._error("Not enough time periods for growth analysis.")

        labels  = [s[0] for s in series]
        values  = [s[1] for s in series]
        growth  = []

        for i in range(1, len(values)):
            prev = values[i - 1]
            curr = values[i]
            pct  = round((curr - prev) / prev * 100, 2) if prev != 0 else None
            growth.append({
                "period":      labels[i],
                "value":       round(curr, 2),
                "prev_value":  round(prev, 2),
                "growth_pct":  pct,
                "direction":   "up" if (pct or 0) > 0 else "down",
            })

        avg_growth = np.mean([g["growth_pct"] for g in growth if g["growth_pct"] is not None])
        cagr_n     = len(values) - 1
        cagr       = round(((values[-1] / values[0]) ** (1 / cagr_n) - 1) * 100, 2) if values[0] > 0 else None

        return {
            "column":     col,
            "periods":    growth,
            "avg_growth": round(float(avg_growth), 2),
            "cagr":       cagr,
            "summary":    (
                f"Average period-over-period growth: {avg_growth:.1f}%. "
                f"CAGR over {cagr_n} periods: {cagr}%." if cagr else
                f"Average growth: {avg_growth:.1f}%."
            ),
        }

    # ══════════════════════════════════════════════════════════════════════
    # PRIVATE HELPERS
    # ══════════════════════════════════════════════════════════════════════
    def _build_time_series(self, col: str) -> Optional[list[tuple]]:
        """Build a sorted (period_label, sum_value) time series from the dataset."""
        e        = self.engine
        date_col = e.date_col

        if date_col:
            return e.monthly_trend(date_col, col, last_n=36)

        # Fallback: try to use the col's index as period if it's sequential
        series = pd.to_numeric(self.df[col], errors="coerce").dropna()
        if len(series) < MIN_POINTS_FORECAST:
            return None
        # Group into equal bins of ~10 rows each
        n_bins  = min(24, len(series) // 5)
        binned  = series.groupby(pd.cut(series.index, bins=n_bins)).mean().dropna()
        return [(f"Period {i+1}", round(float(v), 2)) for i, v in enumerate(binned)]

    @staticmethod
    def _extend_labels(labels: list[str], n: int) -> list[str]:
        """
        Extend a list of period labels forward by n steps.
        Handles YYYY-MM format and plain period labels.
        """
        if not labels:
            return [f"Period +{i+1}" for i in range(n)]

        last = labels[-1]
        result = []

        # Try YYYY-MM
        try:
            period = pd.Period(last, freq="M")
            for i in range(1, n + 1):
                result.append(str(period + i))
            return result
        except Exception:
            pass

        # Try YYYY
        try:
            year = int(last[:4])
            for i in range(1, n + 1):
                result.append(str(year + i))
            return result
        except Exception:
            pass

        # Plain labels
        return [f"{last}+{i}" for i in range(1, n + 1)]

    @staticmethod
    def _calc_trend(values: list[float]) -> tuple[str, float]:
        """Return trend direction and % change from first to last value."""
        if len(values) < 2:
            return "flat", 0.0
        first, last = values[0], values[-1]
        if first == 0:
            return "flat", 0.0
        pct = round((last - first) / abs(first) * 100, 2)
        if pct > 3:   return "up",   pct
        if pct < -3:  return "down", pct
        return "flat", pct

    @staticmethod
    def _error(msg: str) -> dict:
        return {
            "historical": [],
            "forecast":   [],
            "trend":      "flat",
            "trend_pct":  0,
            "r2":         None,
            "mae":        None,
            "slope":      None,
            "summary":    msg,
            "method":     "error",
            "error":      msg,
        }
