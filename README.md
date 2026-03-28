[README.md](https://github.com/user-attachments/files/26320910/README.md)
# DataLensAI v2.0 — AI Business Intelligence Platform

> **Turn raw CSV data into executive-grade intelligence — instantly.**
> Built by **Agam Kumar Verma** · Portfolio: [agamkverma.github.io/agam-portfolio](https://agamkverma.github.io/agam-portfolio)

---

## 🧠 What is DataLensAI?

DataLensAI is a full-stack AI-powered Business Intelligence platform that combines a zero-build-step frontend (single HTML file) with a Python/FastAPI backend capable of:

- **Auto-profiling** any uploaded CSV dataset
- **Generating KPIs** dynamically from detected column types
- **Rendering smart charts** (bar, line, doughnut, scatter, mixed)
- **AI Insights** via rule-based engine or OpenAI / Google Gemini
- **Natural language querying** of your data
- **ML-based predictions** (linear regression, trend forecasting)
- **Professional PDF exports** of complete analysis reports

---

## 📁 Project Structure

```
DataLensAI/
│
├── app.py                  # FastAPI application — routes, CORS, lifecycle
├── data_engine.py          # CSV parsing, cleaning, type inference, transformations
├── dataset_profiler.py     # Statistical profiling, quality scoring, anomaly detection
├── chart_engine.py         # Plotly chart generation, chart-config builder
├── insight_generator.py    # Rule-based + AI insight engine (OpenAI / Gemini)
├── ai_query_engine.py      # NLP query parsing, intent detection, answer builder
├── prediction_engine.py    # Linear regression, forecasting, anomaly detection
│
├── DataLensAI_v2.html      # Frontend — standalone, zero dependencies
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
└── README.md               # This file
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Vanilla HTML · CSS · JavaScript · Chart.js · jsPDF |
| **Backend** | Python 3.10+ · FastAPI · Uvicorn |
| **Data** | Pandas · NumPy · SciPy |
| **ML** | Scikit-learn · StatsModels |
| **Charts** | Plotly (server-side) · Chart.js (client-side) |
| **AI** | OpenAI GPT-4o-mini · Google Gemini 1.5 Flash |
| **Export** | jsPDF (client) · Matplotlib (server) |

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/agamkverma/DataLensAI.git
cd DataLensAI

# Create virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your API keys (optional — app works without them)
```

```env
# .env
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...
AI_PROVIDER=gemini          # "openai" or "gemini"
MAX_ROWS=100000             # Row limit for upload
CORS_ORIGINS=*              # Production: set to your domain
PORT=8000
```

### 3. Run the Server

```bash
python app.py
# OR
uvicorn app:app --reload --port 8000
```

### 4. Open the App

Open `DataLensAI_v2.html` in your browser — or visit `http://localhost:8000` if serving via FastAPI.

---

## 🔌 API Endpoints

### Dataset Operations

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | Upload CSV file, returns full profile |
| `GET`  | `/api/profile/{session_id}` | Get dataset profile & stats |
| `GET`  | `/api/kpis/{session_id}` | Get KPI cards |
| `DELETE` | `/api/session/{session_id}` | Clear session data |

### Analytics

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/api/charts/{session_id}` | Get all chart configs |
| `GET`  | `/api/insights/{session_id}` | Get AI-generated insights |
| `POST` | `/api/query` | Natural language query |
| `POST` | `/api/predict` | Run prediction / forecast |

### Export

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/api/export/csv/{session_id}` | Download cleaned CSV |
| `GET`  | `/api/export/json/{session_id}` | Download analysis JSON |
| `GET`  | `/api/export/pdf/{session_id}` | Download PDF report |

---

## 📊 Sample API Usage

### Upload CSV

```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@your_data.csv" \
  -F "session_id=my-session-123"
```

**Response:**
```json
{
  "session_id": "my-session-123",
  "rows": 1240,
  "columns": 9,
  "quality_score": 94,
  "detected_fields": {
    "revenue": "Sales_Amount",
    "profit": "Profit",
    "date": "Order_Date",
    "category": "Category",
    "region": "Region"
  },
  "kpis": [...],
  "profile": {...}
}
```

### Natural Language Query

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"session_id": "my-session-123", "query": "Which region has highest revenue?"}'
```

**Response:**
```json
{
  "answer": "The West region leads with $1.42M revenue (38% of total)...",
  "intent": "region_analysis",
  "data": [["West", 1420000], ["North", 1120000]],
  "chart_config": {...}
}
```

---

## 🧩 Module Overview

### `data_engine.py`
Core data pipeline — handles CSV ingestion, type inference, cleaning, and transformations. All other modules consume the `DataEngine` output.

### `dataset_profiler.py`
Statistical deep-dive per column: mean, std, percentiles, skewness, kurtosis, cardinality, outlier detection (IQR + Z-score), correlations, and a composite quality score.

### `chart_engine.py`
Builds Plotly chart configs server-side. Auto-selects chart types based on detected column semantics. Returns both Plotly JSON (for server-rendered charts) and Chart.js-compatible configs (for the HTML frontend).

### `insight_generator.py`
Two-mode insight engine:
- **Rule-Based**: Statistical heuristics for revenue concentration, regional gaps, margin analysis, trend direction, data quality warnings.
- **AI-Powered**: Sends a structured data summary prompt to OpenAI or Gemini and parses the structured JSON response.

### `ai_query_engine.py`
Parses natural language queries with intent detection (regex + keyword matching). Dispatches to the correct data aggregation path and formats the answer. Falls back to AI when intent is ambiguous.

### `prediction_engine.py`
ML forecasting module:
- **Linear regression** on numeric columns
- **Time-series forecasting** (rolling average + linear trend)
- **Anomaly detection** (IQR + Z-score)
- **Correlation matrix** computation

---

## 🔐 Security Notes

- Sessions are in-memory (no database required for demo). For production, add Redis or PostgreSQL session storage.
- API keys are read from `.env` — never committed to version control.
- File uploads are validated for type, size, and encoding before processing.
- Rate limiting is recommended for production deployments (`slowapi` compatible).

---

## 📈 Performance

| Dataset Size | Upload + Profile | Insight Generation |
|---|---|---|
| < 10K rows | < 0.5s | < 1s |
| 10K – 100K rows | 0.5s – 2s | 1s – 3s |
| 100K – 500K rows | 2s – 8s | 3s – 8s |
| > 500K rows | Chunked streaming | Sampled (50K rows) |

---

## 🛣️ Roadmap

- [ ] PostgreSQL persistent session storage
- [ ] Multi-file join support
- [ ] Excel (.xlsx) upload support
- [ ] Scheduled report emails
- [ ] Team collaboration (shared dashboards)
- [ ] Custom KPI formula builder
- [ ] LangChain agent for complex multi-step queries

---

## 👤 Author

**Agam Kumar Verma**
Mathematics Graduate · MGKVP Varanasi
Business Data & Operations Analyst

- 🌐 Portfolio: [agamkverma.github.io/agam-portfolio](https://agamkverma.github.io/agam-portfolio)
- 🎓 Google Advanced Data Analytics Certificate
- 🎓 Google Generative AI Learning Path Specialization

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.
