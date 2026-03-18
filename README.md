
[README.md](https://github.com/user-attachments/files/26080676/README.md)
# 🤖 AI Data Analyst Agent
### Autonomous Data Analysis Platform

> Upload any CSV dataset and the AI automatically analyzes it, generates charts, computes KPIs, provides business insights, and answers your questions — like a real data analyst.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.20+-3F4F75?style=flat&logo=plotly&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=flat&logo=pandas)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?style=flat&logo=scikit-learn&logoColor=white)

---

## ✨ Features

| Feature | Description |
|---|---|
| 📂 **Smart Upload** | Upload any CSV — columns auto-detected (date, revenue, region, category…) |
| 📊 **Auto KPI Cards** | 6 dynamic KPI cards computed from your data |
| 📈 **10+ Chart Types** | Bar, line, area, pie/donut, scatter, heatmap, waterfall, forecast, anomaly |
| 💡 **AI Insights** | 8 business insights generated automatically (rule-based + LLM optional) |
| 💬 **NL Queries** | Ask questions in plain English — AI interprets and charts the answer |
| 🔮 **Forecasting** | Polynomial regression forecast with confidence interval |
| 🔍 **Anomaly Detection** | Z-score based monthly anomaly flagging |
| 📋 **Dataset Profiler** | Full statistical profile, correlation matrix, column roles |
| 🎨 **Pro Dark Theme** | Production-grade Streamlit UI with custom CSS |

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/ai-data-analyst-agent.git
cd ai-data-analyst-agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run app.py

# 4. Open browser → http://localhost:8501
```

---

## 📁 Project Structure

```
AI_Data_Analyst_Agent/
│
├── app.py                 ← Main Streamlit application (UI + routing)
├── dataset_profiler.py    ← Auto column detection, KPIs, statistics
├── analysis_engine.py     ← All data computations and aggregations
├── chart_engine.py        ← Plotly chart factory (10+ chart types)
├── insight_generator.py   ← AI insight text (rule-based + GPT-4o/Gemini)
├── query_engine.py        ← NL query parser → analysis plan
│
├── data/
│   └── sample_sales_data.csv   ← 120-row FY2023 sales dataset
│
├── requirements.txt
└── README.md
```

---

## 🧠 Architecture

```
User Upload / Question
        ↓
  dataset_profiler.py
  • Detect column roles
  • Compute KPIs
  • Generate profile
        ↓
  query_engine.py (for queries)
  • NL → structured plan
  • LLM or rule-based
        ↓
  analysis_engine.py
  • Aggregations
  • Time series
  • Forecasting
        ↓
  chart_engine.py
  • Plotly figures
  • Dark themed
        ↓
  insight_generator.py
  • Business narrative
  • Rule-based or LLM
        ↓
   app.py → Streamlit UI
```

---

## 🤖 AI Modes

| Mode | Setup | Quality |
|---|---|---|
| **Demo (default)** | No API key needed | Rule-based, covers 95% of queries |
| **OpenAI GPT-4o** | Add key in sidebar | Best quality insights + parsing |
| **Google Gemini 1.5 Pro** | Add key in sidebar | Excellent alternative to GPT-4o |

---

## 💬 Example Queries

```
Show revenue by region
Which category has the highest profit margin?
Top 10 products by revenue
Show monthly sales trend
Which region underperforms?
Revenue vs profit scatter
Forecast next 3 months
Show heatmap of category vs region
What's the average order value?
Bottom 5 products by profit
```

---

## 📊 Supported Charts

| Chart | Use Case |
|---|---|
| Horizontal Bar | Rankings, top/bottom N |
| Vertical Bar | Category comparisons |
| Grouped Bar | Multi-metric comparison |
| Area/Line | Time series trends |
| Donut Pie | Share/distribution |
| Scatter | Correlation analysis |
| Heatmap | Cross-dimensional analysis |
| Combo Bar+Line | Revenue + margin overlay |
| Forecast | Predictive analytics |
| Gauge | KPI target tracking |

---

## 🛠️ Tech Stack

| Layer | Tech |
|---|---|
| Frontend | Streamlit + custom CSS |
| Charts | Plotly (interactive) |
| Data | Pandas, NumPy |
| ML | scikit-learn (forecast), SciPy (stats) |
| AI (optional) | OpenAI GPT-4o / Google Gemini 1.5 Pro |
| Fonts | Syne · DM Sans · JetBrains Mono |

---

## 📦 Sample Dataset

The included `sample_sales_data.csv` has:
- **120 rows** of FY 2023 sales orders
- **12 columns**: Order ID, Date, Product, Category, Region, Sales Rep, Revenue, Cost, Profit, Units, Customer
- **3 categories**: Electronics, Furniture, Office Supplies
- **4 regions**: North, South, East, West

---

## 📄 License

MIT License — free to use, modify, and share.

---

*Built as a portfolio project demonstrating full-stack AI analytics: data engineering, ML, NLP, visualization, and modern UI.*
