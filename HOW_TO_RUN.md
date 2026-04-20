# How to Run — NFIE (Narrative Framing Intelligence Engine)

## Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Windows (tested), macOS, or Linux

---

## 1. Install Dependencies

Open a terminal in the project directory:

```powershell
pip install -r requirements.txt
```

Download the spaCy language model:

```powershell
python -m spacy download en_core_web_sm
```

**Note:** On the first pipeline run, HuggingFace models (~500MB–1.5GB total) will be
downloaded automatically. This is a one-time operation; subsequent runs use the cached models.

---

## 2. Configure Environment

The `.env` file in the project root contains the NewsAPI key and all configuration values.
Verify or update as needed:

```
NEWS_API_KEY=""
APP_ENV=development
LOG_LEVEL=DEBUG
API_HOST=0.0.0.0
API_PORT=8000
EMBEDDING_MODEL=all-MiniLM-L6-v2
CLASSIFIER_MODEL=facebook/bart-large-mnli
SENTIMENT_MODEL=cardiffnlp/twitter-roberta-base-sentiment-latest
```

---

## 3. Run the Pipeline (CLI — Quick Test)

Execute all 7 analysis phases on the built-in mock dataset:

```powershell
$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8=1; python run_pipeline.py
```

Outputs a formatted comparison report to the terminal. First run may take 2–5 minutes
due to model downloads.

---

## 4. Start the FastAPI Backend

```powershell
python -m uvicorn src.api.main:app --reload
```

Server starts at: **http://localhost:8000**

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check — shows loaded models |
| GET | `/docs` | Interactive Swagger API documentation |
| GET | `/mock-demo` | Full analysis demo using built-in dataset |
| GET | `/articles` | List all stored articles |
| POST | `/ingest` | Fetch articles from NewsAPI by query |
| POST | `/analyze-event` | Submit 2–20 articles for comparison |
| POST | `/analyze-query` | Fetch and analyze in one call |

### Testing the API

```powershell
# Health check
Invoke-RestMethod -Uri "http://localhost:8000/health"

# Run mock demo
Invoke-RestMethod -Uri "http://localhost:8000/mock-demo"

# Fetch real articles
Invoke-RestMethod -Uri "http://localhost:8000/ingest" -Method POST `
  -ContentType "application/json" `
  -Body '{"query": "US China trade tariff", "page_size": 10, "use_mock": false}'
```

Open **http://localhost:8000/docs** in a browser for interactive Swagger testing.

---

## 5. Start the Streamlit Dashboard

Open a second terminal (keep the API server running):

```powershell
$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8=1; streamlit run dashboard/app.py
```

Dashboard opens at: **http://localhost:8501**

### Using the Dashboard

1. Select a data source from the sidebar:
   - **Use Mock Dataset** — Instant analysis, no API key needed
   - **Paste Articles** — Manually input 2–8 articles for comparison
   - **NewsAPI Query** — Search and fetch real articles by keyword
2. Click **Run Analysis**
3. Wait 30–60 seconds on first run (model loading)
4. View results: divergence score, sentiment chart, bias scores, frame radar chart, entity heatmap, and per-article breakdown

---

## 6. Running All Components Together

Use three separate terminals:

**Terminal 1 — API Server:**
```powershell
python -m uvicorn src.api.main:app --reload
```

**Terminal 2 — Streamlit Dashboard:**
```powershell
$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8=1; streamlit run dashboard/app.py
```

**Terminal 3 — CLI Pipeline (optional):**
```powershell
$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8=1; python run_pipeline.py
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `UnicodeEncodeError: 'charmap' codec` | Set `$env:PYTHONIOENCODING="utf-8"` and `$env:PYTHONUTF8=1` before running |
| `ModuleNotFoundError: spacy` | Run `pip install spacy` then `python -m spacy download en_core_web_sm` |
| Pipeline hangs at Phase 2 | Normal on first run — downloading Sentence-BERT model (~80MB). Wait for completion. |
| Pipeline hangs at Phase 4 | Normal on first run — downloading BART-large-MNLI (~500MB). Wait for completion. |
| `python-dotenv could not parse` | Ensure `.env` contains only `KEY=VALUE` lines with no extra characters |
| Port already in use | Use `--port 8001` for uvicorn or `--server.port 8502` for streamlit |

---

## File Reference

```
ccnlp_miniproject/
├── run_pipeline.py           # CLI: runs all 7 analysis phases
├── src/api/main.py           # FastAPI server (Phase 8)
├── dashboard/app.py          # Streamlit dashboard (Phase 9)
├── .env                      # API keys and configuration
├── configs/settings.py       # Centralized settings
├── requirements.txt          # Python dependencies
├── PROJECT_INFO.md           # Full project documentation
└── HOW_TO_RUN.md             # This file
```
