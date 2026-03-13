"""insight_generator.py — Generate business insight text from analysis results."""

import pandas as pd
import numpy as np


def generate_insight(df: pd.DataFrame, plan: dict, col_map: dict,
                     api_key: str = "", provider: str = "openai") -> str:
    if api_key:
        try:
            return _llm(df, plan, col_map, api_key, provider)
        except Exception:
            pass
    return _rule(df, plan, col_map)


def _llm(df, plan, col_map, api_key, provider):
    summary = _summary(df, plan, col_map)
    prompt  = (
        "You are a senior business analyst. Write a concise 2–4 sentence professional business insight "
        "based on the data below. Highlight key findings and one actionable recommendation. "
        "No bullet points — natural business language only.\n\n"
        f"Data:\n{summary}\nContext: {plan.get('insight_request','')}"
    )
    if provider == "openai":
        import openai
        c = openai.OpenAI(api_key=api_key)
        r = c.chat.completions.create(model="gpt-4o",
            messages=[{"role":"user","content":prompt}], temperature=0.4, max_tokens=200)
        return r.choices[0].message.content.strip()
    elif provider == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-1.5-pro").generate_content(prompt).text.strip()
    return _rule(df, plan, col_map)


def _rule(df, plan, col_map) -> str:
    intent    = plan.get("intent","")
    group_by  = _rcol(df, plan.get("group_by"))
    metric    = _rcol(df, plan.get("metric"))
    top_n     = plan.get("top_n")
    rc        = col_map.get("revenue")
    pc        = col_map.get("profit")
    rgc       = col_map.get("region")
    cc        = col_map.get("category")

    parts = []

    # Top/Bottom N
    if intent in ("top_n","bottom_n") and group_by and metric and group_by in df.columns and metric in df.columns:
        asc = (intent=="bottom_n")
        agg = df.groupby(group_by)[metric].sum().sort_values(ascending=asc)
        n   = top_n or 5; items = agg.head(n)
        total = agg.sum(); pct = items.sum()/total*100 if total else 0
        qualifier = "top" if intent=="top_n" else "bottom"
        lst = ", ".join([f"{k} (${v:,.0f})" for k,v in items.items()])
        parts.append(f"The {qualifier} {n} {group_by.lower()}s by {metric.lower()}: {lst}. "
                     f"These account for {pct:.1f}% of total {metric.lower()}.")
        if intent=="bottom_n" and len(agg)>2:
            gap = agg.iloc[-1]-agg.iloc[0]
            parts.append(f"A ${gap:,.0f} gap exists between the best and worst performer. "
                         f"Closing this gap represents a significant growth opportunity.")

    # Trend
    elif intent=="trend_over_time" and metric and metric in df.columns and "_Month" in df.columns:
        mo = df.groupby("_Month")[metric].sum()
        if len(mo)>=2:
            chg = (mo.iloc[-1]-mo.iloc[0])/mo.iloc[0]*100 if mo.iloc[0] else 0
            dir = "grown" if chg>0 else "declined"
            parts.append(f"{metric} has {dir} by {abs(chg):.1f}% over the analysis period, "
                         f"from ${mo.iloc[0]:,.0f} to ${mo.iloc[-1]:,.0f}.")
            peak = mo.idxmax(); pv = mo.max()
            parts.append(f"Peak performance was in {peak} (${pv:,.0f}). "
                         f"Replicating conditions from this period could drive sustained growth.")

    # Regional
    elif rgc and metric and rgc in df.columns and metric in df.columns:
        rp  = df.groupby(rgc)[metric].sum().sort_values(ascending=False)
        bst = rp.index[0]; wst = rp.index[-1]
        pct = rp.iloc[0]/rp.sum()*100 if rp.sum() else 0
        parts.append(f"The {bst} region leads with ${rp.iloc[0]:,.0f} in {metric.lower()} "
                     f"({pct:.1f}% of total).")
        parts.append(f"The {wst} region underperforms at ${rp.iloc[-1]:,.0f}. "
                     f"A targeted intervention in {wst} could meaningfully lift overall performance.")

    # Category
    elif cc and rc and cc in df.columns and rc in df.columns:
        cp  = df.groupby(cc)[rc].sum().sort_values(ascending=False)
        pct = cp.iloc[0]/cp.sum()*100 if cp.sum() else 0
        parts.append(f"{cp.index[0]} leads all categories with ${cp.iloc[0]:,.0f} in revenue "
                     f"({pct:.1f}% of total).")
        if pc and pc in df.columns:
            margins = (df.groupby(cc)[pc].sum() / df.groupby(cc)[rc].sum() * 100).sort_values(ascending=False)
            parts.append(f"{margins.index[0]} has the highest profit margin at {margins.iloc[0]:.1f}%, "
                         f"making it the most efficient segment.")

    # KPI
    elif intent=="kpi_summary":
        if rc and rc in df.columns:
            parts.append(f"Total revenue stands at ${df[rc].sum():,.0f}.")
        if pc and pc in df.columns:
            m = df[pc].sum()/df[rc].sum()*100 if rc and df[rc].sum() else 0
            parts.append(f"Total profit is ${df[pc].sum():,.0f} with an overall margin of {m:.1f}%.")

    # Fallback
    if not parts:
        if metric and metric in df.columns:
            parts.append(f"Total {metric.lower()} is ${df[metric].sum():,.2f} with an average "
                         f"of ${df[metric].mean():,.2f} per record.")
        else:
            parts.append("Analysis complete. Review the chart for detailed patterns and trends.")

    return " ".join(parts)


def _rcol(df, name):
    if not name: return None
    if name in df.columns: return name
    for c in df.columns:
        if c.lower()==str(name).lower(): return c
    return None


def _summary(df, plan, col_map) -> str:
    g = _rcol(df, plan.get("group_by")); m = _rcol(df, plan.get("metric"))
    lines = []
    if g and m and g in df.columns and m in df.columns:
        agg = df.groupby(g)[m].sum().sort_values(ascending=False).head(8)
        lines += [f"{k}: ${v:,.0f}" for k,v in agg.items()]
    rc = col_map.get("revenue"); pc = col_map.get("profit")
    if rc and rc in df.columns: lines.append(f"Total Revenue: ${df[rc].sum():,.0f}")
    if pc and pc in df.columns: lines.append(f"Total Profit:  ${df[pc].sum():,.0f}")
    return "\n".join(lines) or "Business data available."
