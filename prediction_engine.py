"""prediction_engine.py — Forecasting & anomaly detection using sklearn + matplotlib."""

import io, base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

BG      = "#0F172A"
BG2     = "#1E293B"
FG      = "#E2E8F0"
FG2     = "#94A3B8"
PALETTE = ["#4F8EF7","#22D3A5","#F7874F","#A855F7","#F7C34F","#EF4444"]


def _to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _money(x, _):
    if abs(x)>=1e6: return f"${x/1e6:.1f}M"
    if abs(x)>=1e3: return f"${x/1e3:.0f}K"
    return f"${x:.0f}"


def _setup(w=11, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.tick_params(colors=FG2, labelsize=9)
    ax.xaxis.label.set_color(FG2); ax.yaxis.label.set_color(FG2)
    ax.title.set_color(FG)
    for sp in ax.spines.values(): sp.set_edgecolor(BG2)
    ax.grid(axis="y", color=BG2, linewidth=0.8, alpha=0.6); ax.set_axisbelow(True)
    return fig, ax


def generate_forecast(df: pd.DataFrame, col_map: dict, periods_ahead: int = 3,
                       title: str = "Revenue Forecast") -> str:
    dc = col_map.get("date"); rc = col_map.get("revenue") or col_map.get("profit")
    if not dc or not rc or dc not in df.columns or rc not in df.columns:
        return _err_img("Date/metric column not found")

    tmp = df[[dc, rc]].copy()
    tmp[dc] = pd.to_datetime(tmp[dc], errors="coerce")
    tmp = tmp.dropna().set_index(dc)
    mo = tmp[rc].resample("ME").sum()

    if len(mo) < 3:
        return _err_img("Need ≥ 3 months of data for forecasting")

    X = np.arange(len(mo)).reshape(-1, 1)
    y = mo.values

    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import make_pipeline
        deg   = 2 if len(mo) >= 6 else 1
        model = make_pipeline(PolynomialFeatures(deg), LinearRegression())
        model.fit(X, y)
        y_fit = model.predict(X)
        X_fut = np.arange(len(mo), len(mo)+periods_ahead).reshape(-1, 1)
        y_fut = np.maximum(model.predict(X_fut), 0)
    except Exception:
        m, b = np.polyfit(np.arange(len(mo)), y, 1)
        y_fit = m * np.arange(len(mo)) + b
        X_fut = np.arange(len(mo), len(mo)+periods_ahead)
        y_fut = np.maximum(m*X_fut + b, 0)

    fut_dates = pd.date_range(mo.index[-1]+pd.offsets.MonthEnd(1), periods=periods_ahead, freq="ME")
    std       = mo.tail(6).std() * 0.6
    upper, lower = y_fut + std, np.maximum(y_fut - std, 0)

    fig, ax = _setup()
    labels  = mo.index.strftime("%b %Y").tolist()
    fut_lbl = [d.strftime("%b %Y") for d in fut_dates]
    all_x   = range(len(labels)+periods_ahead)

    # Historical area
    ax.fill_between(range(len(labels)), y, alpha=0.10, color=PALETTE[0])
    ax.plot(range(len(labels)), y, color=PALETTE[0], lw=2.5, marker="o", markersize=5, label="Historical")

    # Trend line
    ax.plot(range(len(labels)), y_fit, color=PALETTE[1], lw=1.5, linestyle=":", label="Trend", alpha=0.8)

    # Confidence band
    fc_x = list(range(len(labels)-1, len(labels)+periods_ahead))
    conn_y     = [y[-1]] + list(y_fut)
    conn_upper = [y[-1]] + list(upper)
    conn_lower = [y[-1]] + list(lower)
    ax.fill_between(fc_x, conn_lower, conn_upper, alpha=0.18, color=PALETTE[3])

    # Forecast line
    ax.plot(fc_x, conn_y, color=PALETTE[3], lw=2.5, linestyle="--",
            marker="D", markersize=7, label="Forecast")

    # Value annotations
    for i, (xi, val) in enumerate(zip(fc_x[1:], y_fut)):
        ax.annotate(f"${val:,.0f}", (xi, val), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=8, color=PALETTE[3], fontweight="bold")

    # Divider
    ax.axvline(len(labels)-1, color=FG2, linewidth=1, linestyle="--", alpha=0.5)
    ax.text(len(labels)-0.7, ax.get_ylim()[1]*0.97, "Forecast →",
            color=FG2, fontsize=8, va="top")

    all_labels = labels + fut_lbl
    step = max(1, len(all_labels)//10)
    ax.set_xticks(range(0, len(all_labels), step))
    ax.set_xticklabels(all_labels[::step], rotation=35, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(FuncFormatter(_money))
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12, color=FG)
    ax.legend(frameon=False, labelcolor=FG2, fontsize=9)
    fig.tight_layout()
    return _to_b64(fig)


def detect_anomalies(df: pd.DataFrame, col_map: dict,
                      title: str = "Anomaly Detection — Revenue") -> str:
    dc = col_map.get("date"); rc = col_map.get("revenue") or col_map.get("profit")
    if not dc or not rc or dc not in df.columns or rc not in df.columns:
        return _err_img("Date/revenue columns required")

    tmp = df[[dc, rc]].copy()
    tmp[dc] = pd.to_datetime(tmp[dc], errors="coerce")
    tmp = tmp.dropna().set_index(dc)
    mo  = tmp[rc].resample("ME").sum().reset_index()
    mo.columns = ["Period","Value"]

    if len(mo) < 4:
        return _err_img("Need ≥ 4 months of data")

    mn = mo["Value"].mean(); sd = mo["Value"].std()
    mo["Z"] = (mo["Value"] - mn) / sd
    mo["Anomaly"] = mo["Z"].abs() > 2.0

    labels  = mo["Period"].dt.strftime("%b %Y")
    fig, ax = _setup()

    # Area + line
    ax.fill_between(range(len(mo)), mo["Value"], alpha=0.08, color=PALETTE[0])
    ax.plot(range(len(mo)), mo["Value"], color=PALETTE[0], lw=2, zorder=2)

    # Normal points
    norm = mo[~mo["Anomaly"]]
    ax.scatter(norm.index, norm["Value"], color=PALETTE[0], s=40, zorder=3, label="Normal")

    # Anomaly points
    anom = mo[mo["Anomaly"]]
    if len(anom):
        ax.scatter(anom.index, anom["Value"], color=PALETTE[5], s=130, marker="X",
                   zorder=4, linewidths=1.5, edgecolors=PALETTE[5], label=f"Anomaly ({len(anom)})")
        for i, row in anom.iterrows():
            ax.annotate(f"${row['Value']:,.0f}", (i, row["Value"]),
                        textcoords="offset points", xytext=(6, 8),
                        fontsize=8, color=PALETTE[5], fontweight="bold")

    # Mean band
    ax.axhline(mn,  color=FG2, linewidth=1, linestyle="--", alpha=0.6)
    ax.axhspan(mn-2*sd, mn+2*sd, alpha=0.06, color=PALETTE[0])
    ax.text(0.01, mn, f" μ=${mn:,.0f}", va="bottom", fontsize=8, color=FG2, transform=ax.get_yaxis_transform())

    step = max(1, len(labels)//10)
    ax.set_xticks(range(0, len(labels), step))
    ax.set_xticklabels(labels.iloc[::step], rotation=35, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(FuncFormatter(_money))
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12, color=FG)
    ax.legend(frameon=False, labelcolor=FG2, fontsize=9)
    fig.tight_layout()
    return _to_b64(fig)


def _err_img(msg: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis("off")
    ax.text(0.5, 0.5, f"⚠ {msg}", transform=ax.transAxes,
            ha="center", va="center", color="#EF4444", fontsize=12)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
