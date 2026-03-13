"""
app.py — AI Business Intelligence Copilot (Flask)
Run:  python app.py
Then open:  http://localhost:5000
"""

import os, json
from datetime import datetime
from flask import Flask, render_template, request, session, jsonify, redirect, url_for
import pandas as pd

from data_engine      import load_dataframe, detect_columns, enrich_dataframe, apply_filters
from dataset_profiler import generate_profile, profile_to_context
from ai_query_engine  import process_query
from chart_engine     import (build_chart, dashboard_revenue_by_region,
                               dashboard_category_pie, dashboard_monthly_trend,
                               dashboard_top_products, dashboard_heatmap)
from insight_generator import generate_insight
from prediction_engine import generate_forecast, detect_anomalies

app = Flask(__name__)
app.secret_key = "bi-copilot-secret-2025"

# ── In-memory store (single user / demo) ──────────────────────────────────────
STORE = {
    "df": None, "col_map": {}, "profile": {}, "context": "",
    "chat": [], "api_key": "", "provider": "demo",
    "last_chart": None, "last_plan": None, "last_insight": "",
}

SAMPLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "data", "sample_sales_data.csv")


def fmt_c(v):
    if v >= 1_000_000: return f"${v/1_000_000:.2f}M"
    if v >= 1_000:     return f"${v/1_000:.1f}K"
    return f"${v:,.0f}"


def load_sample():
    df_raw  = load_dataframe(SAMPLE_PATH)
    col_map = detect_columns(df_raw)
    df      = enrich_dataframe(df_raw, col_map)
    profile = generate_profile(df, col_map)
    ctx     = profile_to_context(df, col_map, profile)
    STORE["df"]      = df
    STORE["col_map"] = col_map
    STORE["profile"] = profile
    STORE["context"] = ctx


# Auto-load sample on startup
load_sample()


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    df = STORE["df"]
    if df is None:
        return render_template("index.html", loaded=False)

    col_map = STORE["col_map"]
    profile = STORE["profile"]
    k       = profile.get("kpis", {})

    rc  = col_map.get("revenue"); pc = col_map.get("profit")
    qc  = col_map.get("quantity"); rgc= col_map.get("region")
    cc  = col_map.get("category")

    total_rev    = df[rc].sum()    if rc  else 0
    total_profit = df[pc].sum()    if pc  else 0
    total_units  = int(df[qc].sum()) if qc else 0
    margin_pct   = (total_profit/total_rev*100) if total_rev else 0
    best_region  = k.get("best_region","N/A")
    best_cat     = k.get("best_category","N/A")

    # Date range for filter
    dr = profile.get("date_range")
    date_min = dr["min"] if dr else ""
    date_max = dr["max"] if dr else ""

    # Filter options
    regions    = sorted(df[rgc].dropna().unique().tolist()) if rgc else []
    categories = sorted(df[cc].dropna().unique().tolist())  if cc  else []

    return render_template("index.html",
        loaded=True,
        total_rev    = fmt_c(total_rev),
        total_profit = fmt_c(total_profit),
        margin_pct   = f"{margin_pct:.1f}%",
        margin_color = "#22D3A5" if margin_pct>=30 else "#F7C34F" if margin_pct>=15 else "#EF4444",
        total_units  = f"{total_units:,}",
        best_region  = best_region,
        best_cat     = best_cat,
        row_count    = f"{len(df):,}",
        col_count    = profile.get("col_count",0),
        missing      = profile.get("missing_total",0),
        date_min     = date_min,
        date_max     = date_max,
        regions      = regions,
        categories   = categories,
        chat         = STORE["chat"][-20:],
        last_chart   = STORE["last_chart"],
        last_insight = STORE["last_insight"],
        last_plan    = STORE["last_plan"] or {},
        api_key      = STORE["api_key"],
        provider     = STORE["provider"],
    )


@app.route("/upload", methods=["POST"])
def upload():
    if "file" in request.files and request.files["file"].filename:
        f    = request.files["file"]
        data = f.read()
        df_raw  = load_dataframe(data)
        col_map = detect_columns(df_raw)
        df      = enrich_dataframe(df_raw, col_map)
        profile = generate_profile(df, col_map)
        ctx     = profile_to_context(df, col_map, profile)
        STORE.update({"df":df,"col_map":col_map,"profile":profile,"context":ctx,
                      "chat":[],"last_chart":None,"last_plan":None,"last_insight":""})
    else:
        load_sample()
    return redirect(url_for("index"))


@app.route("/settings", methods=["POST"])
def settings():
    STORE["api_key"]  = request.form.get("api_key","")
    STORE["provider"] = request.form.get("provider","demo")
    return redirect(url_for("index"))


@app.route("/query", methods=["POST"])
def query():
    data  = request.get_json()
    q     = data.get("query","").strip()
    df    = STORE["df"]
    if not q or df is None:
        return jsonify({"error":"No query or data"})

    # Apply filters if provided
    filters = data.get("filters",{})
    col_map = STORE["col_map"]
    if filters:
        df = apply_filters(df, col_map, filters)

    plan    = process_query(q, STORE["context"],
                            api_key=STORE["api_key"],
                            provider=STORE["provider"])
    chart   = build_chart(df, plan, col_map)
    insight = generate_insight(df, plan, col_map,
                               api_key=STORE["api_key"],
                               provider=STORE["provider"])

    STORE["last_chart"]   = chart
    STORE["last_plan"]    = plan
    STORE["last_insight"] = insight
    STORE["chat"].append({"role":"user",    "content":q,
                          "time":datetime.now().strftime("%H:%M")})
    STORE["chat"].append({"role":"assistant","content":insight,
                          "time":datetime.now().strftime("%H:%M")})

    return jsonify({
        "chart":   chart,
        "insight": insight,
        "plan":    plan,
        "reply":   insight,
    })


@app.route("/dashboard_charts")
def dashboard_charts():
    df = STORE["df"]
    if df is None:
        return jsonify({})
    col_map = STORE["col_map"]
    return jsonify({
        "region":   dashboard_revenue_by_region(df, col_map),
        "category": dashboard_category_pie(df, col_map),
        "trend":    dashboard_monthly_trend(df, col_map),
        "products": dashboard_top_products(df, col_map),
        "heatmap":  dashboard_heatmap(df, col_map),
    })


@app.route("/forecast_chart")
def forecast_chart():
    df = STORE["df"]
    if df is None: return jsonify({})
    col_map   = STORE["col_map"]
    periods   = int(request.args.get("periods", 3))
    metric    = request.args.get("metric","revenue")
    fmap      = {**col_map}
    pc        = col_map.get("profit")
    rc        = col_map.get("revenue")
    if metric == "profit" and pc:
        fmap["revenue"] = pc
    chart = generate_forecast(STORE["df"], fmap, periods_ahead=periods,
                               title=f"{metric.title()} Forecast")
    return jsonify({"chart": chart})


@app.route("/anomaly_chart")
def anomaly_chart():
    df = STORE["df"]
    if df is None: return jsonify({})
    chart = detect_anomalies(df, STORE["col_map"])
    return jsonify({"chart": chart})


@app.route("/export_csv")
def export_csv():
    from flask import Response
    df = STORE["df"]
    if df is None: return "No data", 404
    vis = [c for c in df.columns if not c.startswith("_")]
    csv = df[vis].to_csv(index=False)
    return Response(csv, mimetype="text/csv",
                    headers={"Content-Disposition":
                             f"attachment;filename=bi_export_{datetime.now():%Y%m%d_%H%M}.csv"})


@app.route("/export_report")
def export_report():
    from flask import Response
    lines = [
        "AI BUSINESS INTELLIGENCE COPILOT — ANALYSIS REPORT",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "="*60, "",
        "DATASET CONTEXT", "-"*40,
        STORE["context"], "",
        "ANALYSIS Q&A HISTORY", "-"*40,
    ]
    for m in STORE["chat"]:
        prefix = "Q:" if m["role"]=="user" else "A:"
        lines.append(f"{prefix} {m['content']}")
    return Response("\n".join(lines), mimetype="text/plain",
                    headers={"Content-Disposition":
                             f"attachment;filename=bi_report_{datetime.now():%Y%m%d_%H%M}.txt"})


@app.route("/clear_chat", methods=["POST"])
def clear_chat():
    STORE["chat"] = []; STORE["last_chart"]=None
    STORE["last_plan"]=None; STORE["last_insight"]=""
    return redirect(url_for("index"))


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  📊 AI Business Intelligence Copilot")
    print("  Running at:  http://localhost:5000")
    print("="*55 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
