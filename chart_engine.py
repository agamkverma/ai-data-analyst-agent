"""chart_engine.py — Matplotlib-based chart generation returning base64 PNG strings."""

import io, base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter

# ── Design tokens ─────────────────────────────────────────────────────────────
BG       = "#0F172A"
BG2      = "#1E293B"
FG       = "#E2E8F0"
FG2      = "#94A3B8"
GRID     = "#1E293B"
PALETTE  = ["#4F8EF7","#22D3A5","#F7874F","#A855F7","#F7C34F","#EF4444","#06B6D4","#84CC16"]
ACCENT   = "#4F8EF7"


def _setup_fig(w=10, h=5.2):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG2, labelsize=9)
    ax.xaxis.label.set_color(FG2)
    ax.yaxis.label.set_color(FG2)
    ax.title.set_color(FG)
    for spine in ax.spines.values():
        spine.set_edgecolor(BG2)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def _to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _money_fmt(x, _):
    if abs(x) >= 1e6: return f"${x/1e6:.1f}M"
    if abs(x) >= 1e3: return f"${x/1e3:.0f}K"
    return f"${x:.0f}"


def build_chart(df: pd.DataFrame, plan: dict, col_map: dict) -> str:
    intent    = plan.get("intent","")
    chart_t   = plan.get("chart_type","bar")
    group_by  = _resolve(df, plan.get("group_by"))
    metric    = _resolve(df, plan.get("metric"))
    top_n     = plan.get("top_n")
    sort_ord  = plan.get("sort_order","desc")
    title     = plan.get("title","Analysis")
    time_col  = _resolve(df, plan.get("time_column"))
    time_grp  = plan.get("time_grouping","month")

    if metric is None:
        num = [c for c in df.select_dtypes(include="number").columns if not c.startswith("_")]
        if num: metric = num[0]

    try:
        if intent == "trend_over_time":
            return _trend(df, col_map, time_col, metric, time_grp, title)
        elif intent == "forecast":
            from prediction_engine import generate_forecast
            return generate_forecast(df, col_map)
        elif intent == "anomaly":
            from prediction_engine import detect_anomalies
            return detect_anomalies(df, col_map)
        elif chart_t == "pie" or intent == "distribution":
            return _pie(df, group_by, metric, top_n, title)
        elif chart_t == "heatmap":
            return _heatmap(df, col_map, title)
        elif chart_t == "scatter":
            return _scatter(df, col_map, title)
        elif chart_t == "table" or intent == "kpi_summary":
            return _table(df, group_by, metric, top_n, title)
        else:
            return _bar(df, group_by, metric, sort_ord, top_n, title)
    except Exception as e:
        return _error_img(str(e), title)


# ── Chart builders ────────────────────────────────────────────────────────────

def _bar(df, group_by, metric, sort_ord, top_n, title):
    if not group_by or not metric or group_by not in df.columns or metric not in df.columns:
        return _error_img("Column not found", title)

    agg = df.groupby(group_by)[metric].sum().sort_values(ascending=(sort_ord=="asc"))
    if top_n: agg = agg.tail(top_n) if sort_ord=="asc" else agg.head(top_n)

    fig, ax = _setup_fig(10, max(4.5, len(agg)*0.52))
    colors  = [PALETTE[i % len(PALETTE)] for i in range(len(agg))]
    bars    = ax.barh(agg.index.astype(str), agg.values, color=colors,
                      height=0.6, edgecolor="none")

    for bar, val in zip(bars, agg.values):
        ax.text(val + agg.values.max()*0.01, bar.get_y()+bar.get_height()/2,
                f"${val:,.0f}", va="center", ha="left", fontsize=8.5, color=FG2)

    ax.xaxis.set_major_formatter(FuncFormatter(_money_fmt))
    ax.set_xlabel(metric, fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12, color=FG)
    ax.invert_yaxis()
    fig.tight_layout()
    return _to_b64(fig)


def _pie(df, group_by, metric, top_n, title):
    if not group_by or not metric or group_by not in df.columns or metric not in df.columns:
        return _error_img("Column not found", title)

    agg = df.groupby(group_by)[metric].sum().sort_values(ascending=False)
    if top_n: agg = agg.head(top_n)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    wedges, texts, autotexts = ax.pie(
        agg.values, labels=None,
        colors=[PALETTE[i % len(PALETTE)] for i in range(len(agg))],
        autopct="%1.1f%%", startangle=140,
        pctdistance=0.78, wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=2)
    )
    for at in autotexts:
        at.set(color=BG, fontsize=8.5, fontweight="bold")

    legend = ax.legend(
        wedges, [f"{k} (${v:,.0f})" for k, v in zip(agg.index, agg.values)],
        loc="center left", bbox_to_anchor=(0.85, 0.5),
        frameon=True, framealpha=0.1, facecolor=BG2,
        edgecolor=BG2, labelcolor=FG2, fontsize=8.5
    )
    ax.set_title(title, fontsize=13, fontweight="bold", color=FG, pad=10)
    fig.tight_layout()
    return _to_b64(fig)


def _trend(df, col_map, time_col, metric, time_grp, title):
    dc = time_col or col_map.get("date")
    rc = metric   or col_map.get("revenue")
    pc = col_map.get("profit")

    if not dc or not rc or dc not in df.columns or rc not in df.columns:
        return _error_img("Date or metric column not found", title)

    tmp = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(tmp[dc]):
        tmp[dc] = pd.to_datetime(tmp[dc], errors="coerce")
    tmp = tmp.dropna(subset=[dc]).set_index(dc)

    freq = {"month":"ME","quarter":"QE","year":"YE","week":"W"}.get(time_grp,"ME")
    rev_m = tmp[rc].resample(freq).sum()

    fmt   = {"month":"%b %Y","quarter":"Q%q %Y","year":"%Y","week":"W%W %Y"}.get(time_grp,"%b %Y")
    try:    labels = rev_m.index.strftime(fmt)
    except: labels = rev_m.index.astype(str)

    fig, ax = _setup_fig(11, 5)
    ax.fill_between(range(len(rev_m)), rev_m.values, alpha=0.12, color=PALETTE[0])
    ax.plot(range(len(rev_m)), rev_m.values, color=PALETTE[0], lw=2.5, marker="o",
            markersize=5, label=rc)

    if pc and pc in df.columns:
        prof_m = tmp[pc].resample(freq).sum()
        ax.plot(range(len(prof_m)), prof_m.values, color=PALETTE[1], lw=2,
                marker="s", markersize=4, linestyle="--", label=pc)

    ax.set_xticks(range(len(rev_m)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(FuncFormatter(_money_fmt))
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12, color=FG)
    ax.legend(frameon=False, labelcolor=FG2, fontsize=9)
    fig.tight_layout()
    return _to_b64(fig)


def _heatmap(df, col_map, title):
    rc  = col_map.get("revenue")
    cc  = col_map.get("category")
    rgc = col_map.get("region")
    if not (rc and cc and rgc): return _error_img("Need category, region & revenue columns", title)
    if rc not in df.columns or cc not in df.columns or rgc not in df.columns:
        return _error_img("Columns not in dataset", title)

    pivot = df.pivot_table(index=cc, columns=rgc, values=rc, aggfunc="sum", fill_value=0)
    fig, ax = plt.subplots(figsize=(9, max(4, len(pivot)*0.7)))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("bi", ["#0F172A","#1D4ED8","#22D3A5"])
    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, color=FG2, fontsize=9)
    ax.set_yticks(range(len(pivot.index)));   ax.set_yticklabels(pivot.index,   color=FG2, fontsize=9)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            ax.text(j, i, f"${v:,.0f}", ha="center", va="center",
                    fontsize=8, color=FG if v < pivot.values.max()*0.6 else BG, fontweight="bold")

    ax.set_title(title, fontsize=13, fontweight="bold", color=FG, pad=10)
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.tight_layout()
    return _to_b64(fig)


def _scatter(df, col_map, title):
    rc  = col_map.get("revenue")
    pc  = col_map.get("profit")
    cc  = col_map.get("category") or col_map.get("region")
    if not rc or not pc or rc not in df.columns or pc not in df.columns:
        return _error_img("Revenue and Profit columns needed for scatter", title)

    fig, ax = _setup_fig(10, 5.5)
    if cc and cc in df.columns:
        groups = df[cc].unique()
        for i, g in enumerate(groups):
            sub = df[df[cc] == g]
            ax.scatter(sub[rc], sub[pc], color=PALETTE[i % len(PALETTE)],
                       alpha=0.7, s=55, edgecolors="none", label=str(g))
        ax.legend(frameon=False, labelcolor=FG2, fontsize=8.5, markerscale=1.2)
    else:
        ax.scatter(df[rc], df[pc], color=PALETTE[0], alpha=0.6, s=55, edgecolors="none")

    ax.xaxis.set_major_formatter(FuncFormatter(_money_fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(_money_fmt))
    ax.set_xlabel(rc, fontsize=9); ax.set_ylabel(pc, fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12, color=FG)
    fig.tight_layout()
    return _to_b64(fig)


def _table(df, group_by, metric, top_n, title):
    if group_by and metric and group_by in df.columns and metric in df.columns:
        agg = df.groupby(group_by)[metric].sum().sort_values(ascending=False)
        if top_n: agg = agg.head(top_n)
        data = [(str(k), f"${v:,.0f}") for k, v in agg.items()]
        cols = [group_by, metric]
    else:
        vis = [c for c in df.columns if not c.startswith("_")][:6]
        data = [tuple(str(v) for v in row) for row in df[vis].head(12).values]
        cols = vis

    n_rows = len(data)
    fig_h  = max(2.5, n_rows * 0.38 + 0.8)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis("off")

    tbl = ax.table(
        cellText=data, colLabels=cols, cellLoc="left", loc="center",
        bbox=[0, 0, 1, 1]
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(BG2)
        if r == 0:
            cell.set_facecolor("#1D3D7C"); cell.set_text_props(color=PALETTE[0], fontweight="bold")
        else:
            cell.set_facecolor("#0F172A" if r%2 else BG2)
            cell.set_text_props(color=FG2)
        cell.set_linewidth(0.5)

    ax.set_title(title, fontsize=12, fontweight="bold", color=FG, pad=8)
    fig.tight_layout()
    return _to_b64(fig)


def _error_img(msg, title):
    fig, ax = _setup_fig(8, 3)
    ax.text(0.5, 0.5, f"⚠ {title}\n{msg}", transform=ax.transAxes,
            ha="center", va="center", color="#EF4444", fontsize=12,
            bbox=dict(boxstyle="round", facecolor=BG2, edgecolor="#EF4444", alpha=0.8))
    ax.axis("off")
    return _to_b64(fig)


def _resolve(df, name):
    if name is None: return None
    if name in df.columns: return name
    for c in df.columns:
        if c.lower() == str(name).lower(): return c
    return None


# ── Standalone dashboard charts (for auto-dashboard) ─────────────────────────

def dashboard_revenue_by_region(df, col_map):
    rc  = col_map.get("revenue"); rgc = col_map.get("region")
    if not rc or not rgc: return None
    agg = df.groupby(rgc)[rc].sum().sort_values(ascending=False)
    fig, ax = _setup_fig(6, 4)
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(agg))]
    bars = ax.bar(agg.index.astype(str), agg.values, color=colors, edgecolor="none", width=0.6)
    for bar, val in zip(bars, agg.values):
        ax.text(bar.get_x()+bar.get_width()/2, val+agg.values.max()*0.02,
                f"${val/1e3:.0f}K", ha="center", va="bottom", fontsize=8.5, color=FG2)
    ax.yaxis.set_major_formatter(FuncFormatter(_money_fmt))
    ax.set_title("Revenue by Region", fontsize=12, fontweight="bold", color=FG, pad=8)
    ax.set_xlabel(""); ax.tick_params(axis="x", labelsize=9)
    fig.tight_layout()
    return _to_b64(fig)


def dashboard_category_pie(df, col_map):
    rc = col_map.get("revenue"); cc = col_map.get("category")
    if not rc or not cc: return None
    return _pie(df, cc, rc, None, "Revenue by Category")


def dashboard_monthly_trend(df, col_map):
    return _trend(df, col_map, None, None, "month", "Monthly Revenue & Profit Trend")


def dashboard_top_products(df, col_map):
    rc = col_map.get("revenue"); prc = col_map.get("product")
    if not rc or not prc: return None
    return _bar(df, prc, rc, "desc", 8, "Top 8 Products by Revenue")


def dashboard_heatmap(df, col_map):
    return _heatmap(df, col_map, "Revenue: Category × Region")
