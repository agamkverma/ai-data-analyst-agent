"""
app.py
------
AI Data Analyst Agent — Main Streamlit Application
Run: streamlit run app.py

This is the main entry point. It wires together all modules:
  dataset_profiler → analysis_engine → chart_engine → insight_generator → query_engine
"""

import os
import io
import warnings
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# Module imports
from dataset_profiler import (
    coerce_types, detect_column_roles, enrich_dataframe,
    generate_full_profile, profile_to_context_string
)
from analysis_engine import (
    revenue_by_group, revenue_and_profit_by_group, profit_margin_by_group,
    monthly_trend, quarterly_trend, category_distribution, top_n_by_metric,
    bottom_n_by_metric, region_category_heatmap, sales_rep_performance,
    revenue_forecast, correlation_analysis
)
from chart_engine import (
    bar_chart, line_chart, pie_donut_chart, grouped_bar_chart,
    area_chart, heatmap_chart, scatter_plot, combo_bar_line,
    forecast_chart, kpi_gauge, _empty_chart
)
from insight_generator import generate_insights, generate_query_insight
from query_engine import parse_query, llm_parse_query, detect_intent

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "AI Data Analyst Agent",
    page_icon   = "📊",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', system-ui, sans-serif !important;
}
.stApp {
    background: #020817;
    color: #E2E8F0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #070D1C 0%, #040A16 100%);
    border-right: 1px solid #1E293B;
}
section[data-testid="stSidebar"] .stMarkdown p { color: #94A3B8; font-size: 13px; }
section[data-testid="stSidebar"] h2 { color: #E2E8F0 !important; font-family: 'Syne', sans-serif !important; }

/* ── Headers ── */
h1 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important;
     background: linear-gradient(135deg, #4F8EF7, #22D3A5);
     -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
h2, h3 { font-family: 'Syne', sans-serif !important; color: #E2E8F0 !important; }

/* ── KPI Cards ── */
.kpi-card {
    background: linear-gradient(135deg, #0F172A, #1a2235);
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 1.2rem 1.3rem;
    position: relative;
    overflow: hidden;
    transition: transform .2s, border-color .2s;
    margin-bottom: 0.5rem;
}
.kpi-card:hover { transform: translateY(-3px); border-color: #4F8EF7; }
.kpi-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: var(--accent, linear-gradient(90deg, #4F8EF7, #22D3A5));
}
.kpi-icon  { font-size: 22px; margin-bottom: 6px; }
.kpi-label { font-size: 10px; font-weight: 700; letter-spacing: .14em;
             text-transform: uppercase; color: #64748B; margin-bottom: 5px;
             font-family: 'Syne', sans-serif; }
.kpi-value { font-family: 'JetBrains Mono', monospace; font-size: 28px;
             font-weight: 600; color: #F1F5F9; line-height: 1.1; }
.kpi-delta { font-size: 11px; color: #64748B; margin-top: 4px; }
.kpi-badge { font-size: 10px; padding: 2px 8px; border-radius: 20px;
             font-weight: 700; font-family: 'Syne', sans-serif;
             display: inline-block; margin-top: 5px; }
.badge-green  { background: rgba(34,211,165,.12); color: #22D3A5; }
.badge-blue   { background: rgba(79,142,247,.12);  color: #4F8EF7; }
.badge-orange { background: rgba(247,135,79,.12);  color: #F7874F; }
.badge-purple { background: rgba(168,85,247,.12);  color: #A855F7; }

/* ── Insight Box ── */
.insight-box {
    background: linear-gradient(135deg, #0A1020, #0c1830);
    border: 1px solid rgba(79,142,247,.2);
    border-left: 3px solid #4F8EF7;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 13px;
    line-height: 1.8;
    color: #CBD5E1;
    margin-bottom: 1rem;
}
.insight-box .insight-label {
    font-family: 'Syne', sans-serif;
    font-size: 10px; font-weight: 700; letter-spacing: .12em;
    text-transform: uppercase; color: #4F8EF7; margin-bottom: 8px;
}

/* ── Query Result ── */
.query-result {
    background: linear-gradient(135deg, #0A1020, #0c1a2e);
    border: 1px solid #1E293B;
    border-radius: 10px;
    padding: 1rem;
    margin: 0.6rem 0;
}
.query-result .intent-badge {
    font-size: 10px; padding: 3px 10px; border-radius: 20px;
    background: rgba(79,142,247,.12); color: #4F8EF7;
    font-family: 'Syne', sans-serif; font-weight: 700;
    letter-spacing: .06em; display: inline-block; margin-bottom: 8px;
}

/* ── Section Header ── */
.section-header {
    display: flex; align-items: center; gap: 10px;
    margin: 1.5rem 0 0.8rem;
    padding-bottom: 8px;
    border-bottom: 1px solid #1E293B;
}
.section-header .sh-icon { font-size: 18px; }
.section-header .sh-title { font-family: 'Syne', sans-serif; font-size: 14px;
                             font-weight: 700; color: #E2E8F0; letter-spacing: .04em; }
.section-header .sh-sub { font-size: 11px; color: #475569; margin-left: auto; }

/* ── Dataset preview ── */
.stDataFrame { border-radius: 10px !important; overflow: hidden !important; }
.dataframe { background: #0F172A !important; color: #94A3B8 !important; }

/* ── Streamlit component overrides ── */
.stTextInput > div > div > input, .stTextArea > div > div > textarea {
    background: #0F172A !important; border: 1px solid #334155 !important;
    color: #E2E8F0 !important; border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stSelectbox > div > div { background: #0F172A !important; border: 1px solid #334155 !important; }
.stButton > button {
    background: linear-gradient(135deg, #1D4ED8, #2563EB) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important; transition: all .2s !important;
}
.stButton > button:hover { transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(79,142,247,.3) !important; }
div[data-testid="metric-container"] {
    background: #0F172A; border: 1px solid #334155;
    border-radius: 12px; padding: 1rem;
}

/* ── Expander ── */
details { background: #0F172A !important; border: 1px solid #1E293B !important;
          border-radius: 10px !important; }
summary { color: #94A3B8 !important; }

/* ── Tab bar ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0A0E1A !important;
    border-bottom: 1px solid #1E293B !important;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #64748B !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 12px !important; font-weight: 600 !important;
    border-radius: 6px 6px 0 0 !important;
}
.stTabs [aria-selected="true"] {
    color: #4F8EF7 !important;
    border-bottom: 2px solid #4F8EF7 !important;
    background: rgba(79,142,247,.05) !important;
}

/* ── Divider ── */
hr { border-color: #1E293B !important; }

/* ── Success / Warning ── */
.stSuccess { background: rgba(34,211,165,.08) !important; border: 1px solid rgba(34,211,165,.2) !important; }
.stWarning { background: rgba(247,195,79,.08) !important; border: 1px solid rgba(247,195,79,.2)  !important; }
.stInfo    { background: rgba(79,142,247,.08)  !important; border: 1px solid rgba(79,142,247,.2)  !important; }
.stError   { background: rgba(239,68,68,.08)   !important; border: 1px solid rgba(239,68,68,.2)   !important; }
</style>
""", unsafe_allow_html=True)

SAMPLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "data", "sample_sales_data.csv")


# ── Utility ────────────────────────────────────────────────────────────────────

def fmt(v: float) -> str:
    if abs(v) >= 1e6: return f"${v/1e6:.2f}M"
    if abs(v) >= 1e3: return f"${v/1e3:.1f}K"
    return f"${v:,.0f}"


def kpi_card(icon, label, value, badge_text="", badge_class="badge-blue",
             delta="", accent="linear-gradient(90deg,#4F8EF7,#22D3A5)") -> str:
    return f"""
    <div class="kpi-card" style="--accent:{accent}">
      <div class="kpi-icon">{icon}</div>
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      {"<div class='kpi-delta'>" + delta + "</div>" if delta else ""}
      {"<span class='kpi-badge " + badge_class + "'>" + badge_text + "</span>" if badge_text else ""}
    </div>"""


def section_header(icon, title, sub="") -> str:
    return f"""
    <div class="section-header">
      <span class="sh-icon">{icon}</span>
      <span class="sh-title">{title}</span>
      {"<span class='sh-sub'>" + sub + "</span>" if sub else ""}
    </div>"""


@st.cache_data(show_spinner=False)
def load_and_profile(data: bytes) -> tuple[pd.DataFrame, dict]:
    df      = pd.read_csv(io.BytesIO(data))
    df      = coerce_types(df)
    col_map = detect_column_roles(df)
    df      = enrich_dataframe(df, col_map)
    profile = generate_full_profile(df)
    return df, profile


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 0.5rem">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">
        <div style="width:34px;height:34px;border-radius:9px;
                    background:linear-gradient(135deg,#4F8EF7,#22D3A5);
                    display:flex;align-items:center;justify-content:center;font-size:18px">📊</div>
        <div>
          <div style="font-family:Syne,sans-serif;font-size:16px;font-weight:800;
                      background:linear-gradient(135deg,#E2E8F0,#94A3B8);
                      -webkit-background-clip:text;-webkit-text-fill-color:transparent">
            AI Data Analyst
          </div>
          <div style="font-size:10px;color:#475569;letter-spacing:.1em;text-transform:uppercase">
            Autonomous Analytics
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # ── Dataset Upload ──
    st.markdown("**📂 Dataset**")
    upload = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    use_sample = st.button("▶ Load Sample Dataset", use_container_width=True)

    st.divider()

    # ── AI Settings ──
    st.markdown("**🤖 AI Configuration**")
    provider = st.selectbox("Provider", ["Demo (no key needed)", "OpenAI GPT-4o", "Google Gemini 1.5"],
                              label_visibility="collapsed")
    api_key = ""
    if provider != "Demo (no key needed)":
        api_key = st.text_input("API Key", type="password",
                                 placeholder="sk-... or AI...",
                                 label_visibility="collapsed")
    provider_key = "demo" if provider.startswith("Demo") else (
        "openai" if "OpenAI" in provider else "gemini"
    )

    st.divider()

    # ── Quick Queries ──
    st.markdown("**💡 Quick Queries**")
    quick_queries = [
        ("📍", "Revenue by region"),
        ("🏆", "Top 5 products by revenue"),
        ("📈", "Monthly revenue trend"),
        ("🥧", "Category breakdown"),
        ("⚠️", "Bottom performing regions"),
        ("💰", "Profit margin by category"),
        ("🔵", "Revenue vs profit scatter"),
        ("🗺️", "Heatmap category vs region"),
        ("📅", "Quarterly revenue trend"),
        ("🔮", "Revenue forecast"),
    ]
    if "quick_query" not in st.session_state:
        st.session_state.quick_query = ""
    for icon, q in quick_queries:
        if st.button(f"{icon} {q}", use_container_width=True, key=f"qb_{q}"):
            st.session_state.quick_query = q

    st.divider()
    st.markdown("""
    <div style="font-size:11px;color:#334155;line-height:1.7">
    ✅ Standalone — no API key needed<br>
    ✅ 10+ chart types<br>
    ✅ AI business insights<br>
    ✅ Natural language queries<br>
    ✅ Upload any CSV
    </div>
    """, unsafe_allow_html=True)


# ── Session State ───────────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df      = None
    st.session_state.profile = None
    st.session_state.ctx     = ""
    st.session_state.history = []  # chat history

# Load data
if use_sample and os.path.exists(SAMPLE_PATH):
    with open(SAMPLE_PATH, "rb") as f:
        raw = f.read()
    df, profile = load_and_profile(raw)
    st.session_state.df      = df
    st.session_state.profile = profile
    st.session_state.ctx     = profile_to_context_string(df, profile)
    st.success("✓ Sample dataset loaded — 120 rows · FY 2023 Sales Data")

elif upload is not None:
    raw = upload.read()
    try:
        df, profile = load_and_profile(raw)
        st.session_state.df      = df
        st.session_state.profile = profile
        st.session_state.ctx     = profile_to_context_string(df, profile)
        st.success(f"✓ {upload.name} loaded — {len(df):,} rows · {len(df.columns)} columns")
    except Exception as e:
        st.error(f"❌ Failed to load dataset: {e}")


# ── Welcome Screen ──────────────────────────────────────────────────────────────
if st.session_state.df is None:
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                min-height:70vh;text-align:center;padding:2rem">
      <div style="font-size:64px;margin-bottom:1.2rem">🤖</div>
      <h1 style="font-size:2.4rem;margin-bottom:.6rem">AI Data Analyst Agent</h1>
      <p style="color:#64748B;font-size:15px;max-width:480px;line-height:1.8;margin-bottom:1.5rem">
        Upload any CSV dataset and the AI will automatically analyze it,
        generate charts, compute KPIs, and provide business insights — 
        like a real data analyst.
      </p>
      <div style="display:flex;gap:12px;flex-wrap:wrap;justify-content:center;font-size:13px;color:#475569">
        <span style="padding:6px 14px;background:#0F172A;border:1px solid #1E293B;border-radius:20px">📊 Auto Charts</span>
        <span style="padding:6px 14px;background:#0F172A;border:1px solid #1E293B;border-radius:20px">💡 AI Insights</span>
        <span style="padding:6px 14px;background:#0F172A;border:1px solid #1E293B;border-radius:20px">💬 NL Queries</span>
        <span style="padding:6px 14px;background:#0F172A;border:1px solid #1E293B;border-radius:20px">🔮 Forecasting</span>
        <span style="padding:6px 14px;background:#0F172A;border:1px solid #1E293B;border-radius:20px">🔍 Anomalies</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Main Dashboard ──────────────────────────────────────────────────────────────
df      = st.session_state.df
profile = st.session_state.profile
col_map = profile["col_map"]
kpis    = profile["kpis"]

rc  = col_map.get("revenue");   pc  = col_map.get("profit")
cc  = col_map.get("category");  rgc = col_map.get("region")
prc = col_map.get("product");   dc  = col_map.get("date")
qc  = col_map.get("quantity");  cuc = col_map.get("customer")

# ── Page Header ────────────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("<h1 style='margin-bottom:0'>AI Data Analyst Agent</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#475569;font-size:13px;margin-top:4px'>"
                f"{kpis.get('total_records',0):,} records · "
                f"{profile['col_count']} columns · "
                f"{len(profile['num_cols'])} numeric · "
                f"{len(profile['cat_cols'])} categorical"
                f"{'  ·  ' + profile['date_range']['min'] + ' → ' + profile['date_range']['max'] if profile.get('date_range') else ''}"
                f"</p>", unsafe_allow_html=True)
with col_h2:
    st.download_button("⬇ Export CSV", df.to_csv(index=False),
                        file_name="analysis_export.csv", mime="text/csv",
                        use_container_width=True)


# ── KPI Cards Row ──────────────────────────────────────────────────────────────
st.markdown(section_header("📈", "Key Performance Indicators",
                            f"FY {profile['date_range']['min'][:4] if profile.get('date_range') else '2023'}"),
            unsafe_allow_html=True)

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    st.markdown(kpi_card("💵", "Total Revenue", fmt(kpis.get("total_revenue",0)),
                          "Primary KPI", "badge-blue",
                          accent="linear-gradient(90deg,#4F8EF7,#22D3A5)"),
                unsafe_allow_html=True)
with c2:
    st.markdown(kpi_card("📈", "Total Profit", fmt(kpis.get("total_profit",0)),
                          "Net Income", "badge-green",
                          accent="linear-gradient(90deg,#22D3A5,#06B6D4)"),
                unsafe_allow_html=True)
with c3:
    margin = kpis.get("profit_margin_pct", 0)
    m_color = "#22D3A5" if margin >= 30 else "#F7C34F" if margin >= 15 else "#EF4444"
    badge_cls = "badge-green" if margin >= 30 else "badge-orange" if margin >= 15 else "badge-purple"
    st.markdown(kpi_card("🎯", "Profit Margin", f"{margin:.1f}%",
                          "Excellent" if margin>=30 else "Good" if margin>=20 else "Monitor",
                          badge_cls, accent=f"linear-gradient(90deg,{m_color},{m_color}88)"),
                unsafe_allow_html=True)
with c4:
    st.markdown(kpi_card("📦", "Avg Order Value", fmt(kpis.get("avg_order_value",0)),
                          f"{kpis.get('total_records',0):,} orders", "badge-orange",
                          accent="linear-gradient(90deg,#F7874F,#F7C34F)"),
                unsafe_allow_html=True)
with c5:
    st.markdown(kpi_card("🌎", "Best Region", str(kpis.get("best_region","N/A")),
                          fmt(kpis.get("best_region_rev",0)), "badge-purple",
                          accent="linear-gradient(90deg,#A855F7,#EC4899)"),
                unsafe_allow_html=True)
with c6:
    st.markdown(kpi_card("⭐", "Best Category", str(kpis.get("best_category","N/A")),
                          fmt(kpis.get("best_category_rev",0)), "badge-blue",
                          accent="linear-gradient(90deg,#06B6D4,#4F8EF7)"),
                unsafe_allow_html=True)


# ── Main Tabs ───────────────────────────────────────────────────────────────────
tab_analysis, tab_dashboard, tab_query, tab_forecast, tab_profile, tab_preview = st.tabs([
    "📈 Analysis", "📊 Dashboard", "💬 AI Query", "🔮 Forecast", "📋 Profile", "🗂 Data Preview"
])


# ══════════════════════════════════════════════════════════════════════
# TAB 1: ANALYSIS — Auto-generated charts
# ══════════════════════════════════════════════════════════════════════
with tab_analysis:

    # ── AI Insights ──
    st.markdown(section_header("💡", "AI Business Insights", "Auto-generated from your dataset"),
                unsafe_allow_html=True)

    with st.spinner("🤖 Generating insights…"):
        insights = generate_insights(df, col_map, kpis, st.session_state.ctx,
                                      api_key, provider_key)
    st.markdown(f'<div class="insight-box"><div class="insight-label">💡 AI Analysis</div>{insights}</div>',
                unsafe_allow_html=True)

    st.divider()

    # ── Chart Row 1: Revenue by Region + Monthly Trend ──
    st.markdown(section_header("📊", "Revenue Analysis"), unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1.4])

    with col1:
        if rgc and rc:
            data = revenue_by_group(df, rgc, rc)
            fig  = bar_chart(data, rgc, rc,
                              title="Revenue by Region", horizontal=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Region or Revenue column not detected in this dataset.")

    with col2:
        if dc and rc:
            data = monthly_trend(df, dc, rc, pc)
            cols = [c for c in [rc, pc] if c in data.columns]
            fig  = line_chart(data, "Period", cols, title="Monthly Revenue & Profit Trend")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Date or Revenue column not detected.")

    # ── Chart Row 2: Category Pie + Top 5 Products ──
    col3, col4 = st.columns(2)

    with col3:
        if cc and rc:
            data = category_distribution(df, cc, rc)
            fig  = pie_donut_chart(data, cc, rc, title="Revenue Distribution by Category")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Category or Revenue column not detected.")

    with col4:
        if prc and rc:
            data = top_n_by_metric(df, prc, rc, n=5)
            fig  = bar_chart(data, prc, rc, title="Top 5 Products by Revenue",
                              horizontal=True, top_n=5)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Product or Revenue column not detected.")

    # ── Chart Row 3: Profit Margin + Region×Category Heatmap ──
    col5, col6 = st.columns(2)

    with col5:
        if cc and rc and pc:
            data = revenue_and_profit_by_group(df, cc, rc, pc)
            fig  = combo_bar_line(data, cc, rc, "Margin%",
                                   title="Revenue vs Profit Margin by Category")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Category, Revenue, or Profit column not detected.")

    with col6:
        if rgc and cc and rc:
            data = region_category_heatmap(df, rgc, cc, rc)
            fig  = heatmap_chart(data, title="Revenue Heatmap: Category × Region")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Region or Category column not detected.")

    # ── Chart Row 4: Revenue vs Profit Scatter + Sales Rep ──
    col7, col8 = st.columns(2)

    with col7:
        if rc and pc:
            fig = scatter_plot(df, rc, pc, color_col=cc,
                                title="Revenue vs Profit (by Category)")
            st.plotly_chart(fig, use_container_width=True)

    with col8:
        rep_col = None
        for col in df.columns:
            if any(kw in col.lower() for kw in ["rep","salesperson","agent","seller"]):
                rep_col = col; break
        if rep_col and rc:
            data = sales_rep_performance(df, rep_col, rc, pc)
            fig  = bar_chart(data, rep_col, rc, title="Sales Rep Performance",
                              horizontal=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            if qc and cc:
                data = top_n_by_metric(df, cc, qc, n=5) if qc else pd.DataFrame()
                if not data.empty:
                    fig = bar_chart(data, cc, qc, title="Units Sold by Category", horizontal=False)
                    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 2: DASHBOARD — KPI Summary Dashboard
# ══════════════════════════════════════════════════════════════════════
with tab_dashboard:
    st.markdown(section_header("📊", "Executive Dashboard", "Auto-generated business intelligence"),
                unsafe_allow_html=True)

    # Row 1: Gauges
    g1, g2, g3 = st.columns(3)
    with g1:
        if kpis.get("total_revenue"):
            fig = kpi_gauge(kpis["total_revenue"], kpis["total_revenue"]*1.5,
                             "Total Revenue", "#4F8EF7")
            st.plotly_chart(fig, use_container_width=True)
    with g2:
        if kpis.get("total_profit"):
            fig = kpi_gauge(kpis["total_profit"], kpis["total_revenue"],
                             "Total Profit", "#22D3A5")
            st.plotly_chart(fig, use_container_width=True)
    with g3:
        if kpis.get("avg_order_value"):
            fig = kpi_gauge(kpis["avg_order_value"], kpis["max_order_value"],
                             "Avg Order Value", "#A855F7")
            st.plotly_chart(fig, use_container_width=True)

    # Row 2: Top products + quarterly
    d1, d2 = st.columns([1.2, 1])
    with d1:
        if prc and rc:
            data = top_n_by_metric(df, prc, rc, n=10)
            fig  = bar_chart(data, prc, rc, title="Top 10 Products by Revenue",
                              horizontal=True)
            st.plotly_chart(fig, use_container_width=True)
    with d2:
        if dc and rc:
            data = quarterly_trend(df, dc, rc)
            if not data.empty:
                fig = area_chart(data, "Period", rc, title="Quarterly Revenue")
                st.plotly_chart(fig, use_container_width=True)

    # Row 3: Revenue + profit grouped by region
    if rgc and rc and pc:
        data = revenue_and_profit_by_group(df, rgc, rc, pc)
        fig  = grouped_bar_chart(data, rgc, [rc, pc],
                                  title="Revenue & Profit by Region — Side by Side")
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 3: QUERY — Natural Language Interface
# ══════════════════════════════════════════════════════════════════════
with tab_query:
    st.markdown(section_header("💬", "Natural Language Query Interface",
                                "Ask questions about your data in plain English"),
                unsafe_allow_html=True)

    # Query input
    qcol1, qcol2 = st.columns([5, 1])
    with qcol1:
        user_query = st.text_input(
            "Ask a question",
            value=st.session_state.quick_query,
            placeholder="e.g. Show revenue by region / Top 5 products by profit / Monthly sales trend",
            label_visibility="collapsed",
        )
        st.session_state.quick_query = ""
    with qcol2:
        run_query = st.button("🔍 Analyze", use_container_width=True)

    # Process query
    if run_query and user_query.strip():
        with st.spinner("🤖 Analyzing your question…"):
            # Try LLM parse first, fall back to rule-based
            plan = {}
            if api_key and provider_key != "demo":
                plan = llm_parse_query(user_query, st.session_state.ctx, api_key, provider_key)
            if not plan:
                plan = parse_query(user_query, col_map, df.columns.tolist())

            intent    = plan.get("intent", "kpi_summary")
            group_col = plan.get("group_by")
            metric    = plan.get("metric") or rc
            top_n     = plan.get("top_n", 5)
            ascending = plan.get("ascending", False)
            title     = plan.get("title", user_query.title())
            subtitle  = plan.get("subtitle", "")
            chart_t   = plan.get("chart", "bar")

        # Render result
        st.markdown(f'<div class="query-result">'
                    f'<span class="intent-badge">Intent: {intent}</span>'
                    f'<br><b style="color:#E2E8F0">{title}</b>'
                    f'{"<br><span style=color:#475569;font-size:11px>" + subtitle + "</span>" if subtitle else ""}'
                    f'</div>',
                    unsafe_allow_html=True)

        # ── Execute chart based on intent ──
        fig = None
        result_df = pd.DataFrame()

        if chart_t == "pie" or "pie" in intent or "distribution" in intent:
            if group_col and metric:
                result_df = category_distribution(df, group_col, metric)
                fig = pie_donut_chart(result_df, group_col, metric, title=title)

        elif chart_t == "heatmap" or "heat" in intent:
            if rgc and cc and rc:
                result_df = region_category_heatmap(df, rgc, cc, rc)
                fig = heatmap_chart(result_df, title=title)

        elif chart_t in ("line","area") or "trend" in intent or "monthly" in intent:
            if dc and metric:
                freq = "QE" if "quarter" in intent else "ME"
                result_df = monthly_trend(df, dc, metric, freq=freq)
                fig = area_chart(result_df, "Period", metric, title=title)

        elif chart_t == "scatter" or "scatter" in intent:
            if rc and pc:
                fig = scatter_plot(df, rc, pc, color_col=cc, title=title)
                result_df = df[[rc, pc]].head(20)

        elif chart_t == "forecast" or "forecast" in intent:
            if dc and rc:
                result_df = revenue_forecast(df, dc, rc, periods_ahead=3)
                fig = forecast_chart(result_df, dc, rc, title=title)

        elif chart_t == "kpi" or intent == "kpi_summary":
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Total Revenue", fmt(kpis.get("total_revenue",0)))
                st.metric("Avg Order Value", fmt(kpis.get("avg_order_value",0)))
            with col_b:
                st.metric("Total Profit", fmt(kpis.get("total_profit",0)))
                st.metric("Total Units", f"{kpis.get('total_units',0):,}")
            with col_c:
                st.metric("Profit Margin", f"{kpis.get('profit_margin_pct',0):.1f}%")
                st.metric("Best Region", str(kpis.get("best_region","N/A")))

        else:
            # Default: bar chart
            if group_col and metric and group_col in df.columns and metric in df.columns:
                result_df = revenue_by_group(df, group_col, metric, top_n=top_n, ascending=ascending)
                fig = bar_chart(result_df, group_col, metric, title=title,
                                 horizontal=True, top_n=top_n)

        if fig:
            st.plotly_chart(fig, use_container_width=True)

        # AI insight for this query
        result_summary = result_df.head(5).to_string() if not result_df.empty else ""
        insight = generate_query_insight(df, col_map, kpis, user_query,
                                          result_summary, api_key, provider_key)
        st.markdown(f'<div class="insight-box"><div class="insight-label">💡 Insight</div>{insight}</div>',
                    unsafe_allow_html=True)

        # Add to history
        st.session_state.history.append({
            "query":   user_query,
            "intent":  intent,
            "insight": insight,
        })

    # ── Chat History ──
    if st.session_state.history:
        st.divider()
        st.markdown(section_header("📜", "Query History"), unsafe_allow_html=True)
        for i, h in enumerate(reversed(st.session_state.history[-5:])):
            with st.expander(f"🔍 {h['query']}", expanded=(i == 0)):
                st.markdown(f"**Intent:** `{h['intent']}`")
                st.markdown(h["insight"])


# ══════════════════════════════════════════════════════════════════════
# TAB 4: FORECAST & ANOMALY
# ══════════════════════════════════════════════════════════════════════
with tab_forecast:
    st.markdown(section_header("🔮", "Predictive Analytics",
                                "Forecasting & anomaly detection powered by sklearn"),
                unsafe_allow_html=True)

    fc1, fc2 = st.columns([2, 1])
    with fc1:
        periods_ahead = st.slider("Forecast periods (months)", 1, 12, 3)
    with fc2:
        forecast_metric = st.selectbox(
            "Metric",
            [c for c in [rc, pc] if c],
            label_visibility="visible"
        )

    if dc and forecast_metric:
        with st.spinner("Computing forecast…"):
            fc_map = {**col_map, "revenue": forecast_metric}
            fore_df = revenue_forecast(df, dc, forecast_metric, periods_ahead)
        if not fore_df.empty:
            fig = forecast_chart(fore_df, dc, forecast_metric,
                                  title=f"{forecast_metric} Forecast — Next {periods_ahead} Months")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            <div class="insight-box">
            <div class="insight-label">ℹ️ Methodology</div>
            <b>Polynomial Regression</b> (degree 2 if ≥ 6 months of data, else linear regression)
            fitted on historical monthly aggregates. The shaded band represents ±1σ confidence range.
            Forecast values are annotated directly on the chart.
            </div>""", unsafe_allow_html=True)

    # ── Anomaly Detection ──
    st.divider()
    st.markdown(section_header("🔍", "Anomaly Detection", "Z-score method"), unsafe_allow_html=True)

    if dc and rc:
        trend = monthly_trend(df, dc, rc)
        if not trend.empty:
            mn = trend[rc].mean()
            sd = trend[rc].std()
            trend["Z"] = (trend[rc] - mn) / sd
            trend["Anomaly"] = trend["Z"].abs() > 2.0
            n_anomalies = trend["Anomaly"].sum()

            # Build chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trend["Period"].astype(str), y=trend[rc],
                mode="lines+markers", name="Monthly Revenue",
                line=dict(color="#4F8EF7", width=2.5),
                marker=dict(size=6),
                fill="tozeroy", fillcolor="rgba(79,142,247,0.06)",
            ))
            if n_anomalies > 0:
                anom = trend[trend["Anomaly"]]
                fig.add_trace(go.Scatter(
                    x=anom["Period"].astype(str), y=anom[rc],
                    mode="markers", name=f"Anomaly ({n_anomalies})",
                    marker=dict(color="#EF4444", size=14, symbol="x",
                                line=dict(width=2.5, color="#EF4444")),
                ))
            fig.add_hline(y=mn, line_dash="dash", line_color="#64748B",
                           annotation_text=f"Mean: {fmt(mn)}", annotation_font_color="#64748B")
            fig.add_hrect(y0=mn-2*sd, y1=mn+2*sd, fillcolor="rgba(79,142,247,0.05)",
                           line_width=0, annotation_text="Normal band (±2σ)")
            fig.update_layout(
                paper_bgcolor="#111827", plot_bgcolor="#0F1525",
                font=dict(color="#94A3B8"), title="Monthly Revenue — Anomaly Detection",
                title_font=dict(color="#E2E8F0", size=15),
                legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#94A3B8"),
                xaxis=dict(gridcolor="#1E293B", tickfont_color="#94A3B8"),
                yaxis=dict(gridcolor="#1E293B", tickfont_color="#94A3B8"),
                margin=dict(l=40, r=20, t=50, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.info(f"🔍 Detected **{n_anomalies}** anomalous month(s) using Z-score > ±2σ methodology.")


# ══════════════════════════════════════════════════════════════════════
# TAB 5: PROFILE — Dataset structure & statistics
# ══════════════════════════════════════════════════════════════════════
with tab_profile:
    st.markdown(section_header("📋", "Dataset Profile", "Auto-detected structure and statistics"),
                unsafe_allow_html=True)

    # Overview row
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Total Rows", f"{profile['row_count']:,}")
    p2.metric("Total Columns", profile['col_count'])
    p3.metric("Numeric Columns", len(profile['num_cols']))
    p4.metric("Missing Values", profile['missing_total'])

    st.divider()

    # Column roles + numeric stats
    pr1, pr2 = st.columns(2)

    with pr1:
        st.markdown("#### 🔍 Detected Column Roles")
        roles_data = {"Role": [], "Column": [], "Type": []}
        for role, col in col_map.items():
            roles_data["Role"].append(role.title())
            roles_data["Column"].append(col)
            roles_data["Type"].append(str(df[col].dtype))
        st.dataframe(pd.DataFrame(roles_data), use_container_width=True, hide_index=True)

    with pr2:
        st.markdown("#### 📊 Categorical Column Stats")
        cat_data = {"Column": [], "Unique Values": [], "Top Value": [], "Top Count": []}
        for col, stats in profile["cat_stats"].items():
            cat_data["Column"].append(col)
            cat_data["Unique Values"].append(stats["unique"])
            cat_data["Top Value"].append(stats["top_val"])
            cat_data["Top Count"].append(stats["top_cnt"])
        st.dataframe(pd.DataFrame(cat_data), use_container_width=True, hide_index=True)

    # Numeric stats
    st.markdown("#### 📉 Numeric Column Statistics")
    if profile["num_stats"]:
        st.dataframe(
            pd.DataFrame(profile["num_stats"]).T.reset_index().rename(columns={"index": "Column"}),
            use_container_width=True, hide_index=True
        )

    # Correlation
    if len(profile["num_cols"]) >= 2:
        st.markdown("#### 🔗 Correlation Matrix")
        corr = correlation_analysis(df, col_map)
        if not corr.empty:
            fig = heatmap_chart(corr, title="Numeric Column Correlation")
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 6: DATA PREVIEW
# ══════════════════════════════════════════════════════════════════════
with tab_preview:
    st.markdown(section_header("🗂", "Dataset Preview",
                                f"Showing first 100 rows of {profile['row_count']:,} total"),
                unsafe_allow_html=True)

    # Filters
    f1, f2, f3 = st.columns(3)
    filter_region = filter_cat = None
    with f1:
        if rgc:
            regions = ["All"] + sorted(df[rgc].dropna().unique().tolist())
            filter_region = st.selectbox("Filter by Region", regions)
    with f2:
        if cc:
            cats = ["All"] + sorted(df[cc].dropna().unique().tolist())
            filter_cat = st.selectbox("Filter by Category", cats)
    with f3:
        search = st.text_input("Search (any column)", placeholder="Type to filter…")

    filtered = df.copy()
    if filter_region and filter_region != "All" and rgc:
        filtered = filtered[filtered[rgc] == filter_region]
    if filter_cat and filter_cat != "All" and cc:
        filtered = filtered[filtered[cc] == filter_cat]
    if search:
        mask = filtered.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
        filtered = filtered[mask]

    vis_cols = [c for c in filtered.columns if not c.startswith("_")]
    st.dataframe(
        filtered[vis_cols].head(100),
        use_container_width=True,
        height=400,
    )
    st.caption(f"Showing {min(100, len(filtered)):,} of {len(filtered):,} filtered rows")

    # Download filtered
    st.download_button("⬇ Download Filtered Data",
                        filtered[vis_cols].to_csv(index=False),
                        file_name="filtered_export.csv", mime="text/csv")


# ── Footer ──────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center;font-size:11px;color:#334155;padding:0.5rem 0">
  AI Data Analyst Agent &nbsp;·&nbsp;
  Python · Pandas · Streamlit · Plotly · scikit-learn &nbsp;·&nbsp;
  Portfolio Project
</div>
""", unsafe_allow_html=True)
