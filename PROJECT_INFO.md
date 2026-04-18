# NFIE — Narrative Framing Intelligence Engine

## Project Overview

The **Narrative Framing Intelligence Engine (NFIE)** is an NLP-powered system that reveals how different news outlets construct reality when covering the **same event**. It collects news articles from multiple sources, runs them through a multi-phase analysis pipeline, and produces a structured comparison report quantifying differences in framing, sentiment, lexical bias, and entity emphasis.

**Core Question:** Given the exact same event, how differently do news outlets present it to their audiences — and can we measure those differences?

---

## Problem Statement

When a major event occurs (e.g., US–China trade tariffs), different outlets report on it with dramatically different:

- **Framing** — One outlet emphasizes economic impact, another emphasizes national security
- **Sentiment** — Identical facts presented with positive vs. negative emotional charge
- **Word Choice** — "Economic bullying" vs. "trade restrictions" (same action, different connotation)
- **Entity Focus** — Certain people, organizations, or countries emphasized or omitted

NFIE quantifies these differences, turning invisible editorial bias into measurable data.

---

## Architecture

```
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      NFIE PIPELINE                                    │
 │                                                                       │
 │  Phase 1           Phase 2            Phase 3           Phase 4       │
 │ ┌───────────┐   ┌──────────────┐   ┌────────────┐   ┌─────────────┐  │
 │ │   Data    │   │  Embeddings  │   │   Named    │   │   Frame     │  │
 │ │ Ingestion │──▸│ & Clustering │──▸│   Entity   │──▸│Classification│ │
 │ │ (NewsAPI) │   │   (FAISS)    │   │   Recog.   │   │ (Zero-Shot) │  │
 │ └───────────┘   └──────────────┘   └────────────┘   └─────────────┘  │
 │                                                                       │
 │  Phase 5           Phase 6            Phase 7                         │
 │ ┌───────────┐   ┌──────────────┐   ┌────────────┐                    │
 │ │ Sentiment │   │   Lexical    │   │ Comparison │                    │
 │ │  & Tone   │──▸│    Bias      │──▸│   Engine   │──▸ REPORT          │
 │ │ Analysis  │   │  Detection   │   │  (Final)   │                    │
 │ └───────────┘   └──────────────┘   └────────────┘                    │
 │                                                                       │
 │  Phase 8: FastAPI Backend          Phase 9: Streamlit Dashboard       │
 └────────────────────────────────────────────────────────────────────────┘
```

---

## Phase-by-Phase Breakdown

### Phase 1 — Data Ingestion
**Module:** `src/ingestion/fetcher.py`

- Fetches news articles from **NewsAPI** or loads a built-in mock dataset (8 articles on US–China tariffs from Reuters, Fox News, The Guardian, Breitbart, Al Jazeera, NPR, Wired, Foreign Affairs)
- Normalizes every article into a consistent schema: `id`, `title`, `content`, `source`, `url`, `date`, `word_count`
- Deduplicates using MD5 hashing of `(source + title + date)`
- **Technology:** `requests`, NewsAPI REST API

### Phase 2 — Embeddings and Clustering
**Module:** `src/clustering/embedder.py`

- Converts each article into a **384-dimensional dense vector** using **Sentence-BERT** (`all-MiniLM-L6-v2`)
- Stores vectors in a **FAISS** index (Facebook AI Similarity Search) for fast nearest-neighbor retrieval
- Groups related articles using **Agglomerative Clustering** with cosine distance
- **Why Agglomerative over KMeans:** Does not require specifying the number of clusters in advance; determines cluster count automatically via distance threshold
- **Technology:** `sentence-transformers`, `faiss-cpu`, `scikit-learn`

### Phase 3 — Named Entity Recognition
**Module:** `src/ner/extractor.py`

- Extracts named entities (PERSON, ORG, GPE, MONEY, PERCENT) using **spaCy** (`en_core_web_sm`)
- Validates clustering quality via **entity coherence score**: articles sharing many entities are likely about the same event
- Compares entity prominence across sources to detect framing by emphasis
- **Technology:** `spaCy`, Jaccard similarity

### Phase 4 — Frame Classification
**Module:** `src/classification/framer.py`

- Classifies articles into 8 frame categories: political, economic, emotional, security, nationalist, humanitarian, legal, scientific
- Uses **zero-shot classification** via `facebook/bart-large-mnli` — an NLI-trained model that classifies text into categories without task-specific fine-tuning
- Supports **multi-label classification** (an article can be simultaneously economic and political)
- Computes **frame divergence score** (0.0 = identical framing, 1.0 = completely different)
- **Technology:** `transformers` (HuggingFace), BART-large-MNLI

### Phase 5 — Sentiment and Tone Analysis
**Module:** `src/sentiment/analyzer.py`

- **ML-based sentiment:** Uses `cardiffnlp/twitter-roberta-base-sentiment-latest` (RoBERTa fine-tuned on 58 million tweets) for headline and body sentiment
- **Tone profiling:** Lexicon-based detection across four dimensions — alarm, aggression, measured, emotional
- **Sentence-level granularity:** Scores individual sentences to locate where emotionally charged language appears
- Compound score: weighted average of headline sentiment (40%) and body sentiment (60%)
- **Technology:** `transformers`, RoBERTa, custom lexicons

### Phase 6 — Lexical Bias Detection
**Module:** `src/bias/detector.py`

- **TF-IDF analysis:** Identifies terms uniquely overused by each outlet compared to the full corpus
- **Power word detection:** Flags loaded vocabulary (e.g., "aggression", "patriot", "devastate")
- **Hedging language detection:** Identifies doubt signals ("allegedly", "reportedly", "critics say")
- **Exclusive vocabulary:** Words appearing in one source's top-10 TF-IDF that no other source uses prominently
- Bias score: combines power word density, amplifier count, and hedging frequency (0 = neutral, 1 = highly loaded)
- **Technology:** Custom TF-IDF, lexicon matching

### Phase 7 — Comparison Engine
**Module:** `src/comparison/engine.py`

- Aggregates results from Phases 3–6 into a unified comparison report
- Computes **overall divergence score** with weighted components: Frame (40%) + Sentiment (35%) + Bias (25%)
- Generates human-readable key difference bullets and assigns a divergence level (Minimal / Low / Moderate / High / Extreme)
- Produces per-article summary tables for dashboard display
- Auto-generates an event summary from the most neutral article (highest word count × lowest bias)

### Phase 8 — FastAPI Backend
**Module:** `src/api/main.py`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with model status |
| `/docs` | GET | Auto-generated Swagger documentation |
| `/mock-demo` | GET | Full demo using built-in dataset |
| `/articles` | GET | List stored articles |
| `/ingest` | POST | Fetch articles from NewsAPI |
| `/analyze-event` | POST | Submit articles for full comparison |
| `/analyze-query` | POST | Fetch + analyze in one call |

- Pre-warms NER and sentiment models at startup
- Pydantic validation for all request schemas
- **Technology:** FastAPI, Uvicorn, Pydantic

### Phase 9 — Streamlit Dashboard
**Module:** `dashboard/app.py`

- Three input modes: mock data, manual article entry, or live NewsAPI search
- Visualizations: sentiment bar chart, bias score chart, frame radar chart, entity heatmap, per-article summary table
- Interactive controls for enabling/disabling individual analysis phases
- Raw JSON output for debugging
- **Technology:** Streamlit, Plotly, Pandas

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Sentence Embeddings | `all-MiniLM-L6-v2` (Sentence-BERT) | Text to 384-dim vectors |
| Vector Search | FAISS (`faiss-cpu`) | Similarity search at scale |
| Clustering | Agglomerative (`scikit-learn`) | Event-based article grouping |
| Named Entity Recognition | spaCy (`en_core_web_sm`) | Entity extraction |
| Frame Classification | BART-large-MNLI (HuggingFace) | Zero-shot frame detection |
| Sentiment Analysis | RoBERTa (`twitter-roberta-base-sentiment`) | 3-class sentiment scoring |
| Bias Detection | Custom TF-IDF + lexicons | Lexical bias quantification |
| API Backend | FastAPI + Uvicorn | REST API |
| Dashboard | Streamlit + Plotly | Interactive visualization |
| Data Source | NewsAPI | Real-time news ingestion |

---

## Key NLP Concepts

| Concept | Application in NFIE |
|---------|-------------------|
| Sentence Embeddings | Dense vector representations where semantic similarity equals geometric closeness |
| Zero-Shot Classification | Classifying into frame categories without task-specific training data |
| Named Entity Recognition | Extracting people, organizations, countries, and monetary values from text |
| TF-IDF | Surfacing words uniquely important to each outlet compared to the full corpus |
| Agglomerative Clustering | Grouping articles by event without pre-specifying the number of clusters |
| Cosine Similarity | Measuring semantic relatedness between embedding vectors |
| Multi-Label Classification | Allowing articles to carry multiple frame labels simultaneously |
| Lexicon-based Tone Analysis | Detecting alarm, aggression, measured, and emotional language via curated word lists |

---

## Project Structure

```
ccnlp_miniproject/
├── .env                            # API keys and configuration
├── configs/
│   └── settings.py                 # Centralized settings
├── src/
│   ├── ingestion/
│   │   └── fetcher.py              # Phase 1: Data ingestion
│   ├── clustering/
│   │   └── embedder.py             # Phase 2: Embeddings + FAISS + clustering
│   ├── ner/
│   │   └── extractor.py            # Phase 3: Named entity recognition
│   ├── classification/
│   │   └── framer.py               # Phase 4: Frame classification
│   ├── sentiment/
│   │   └── analyzer.py             # Phase 5: Sentiment + tone analysis
│   ├── bias/
│   │   └── detector.py             # Phase 6: Lexical bias detection
│   ├── comparison/
│   │   └── engine.py               # Phase 7: Comparison engine
│   ├── api/
│   │   └── main.py                 # Phase 8: FastAPI backend
│   └── utils/
│       ├── cache.py                # Batch processing utilities
│       └── logger.py               # Structured logging
├── dashboard/
│   └── app.py                      # Phase 9: Streamlit dashboard
├── data/
│   ├── raw/                        # Fetched articles
│   ├── processed/                  # Analysis results (JSON)
│   └── embeddings/                 # FAISS index files
├── run_pipeline.py                 # CLI end-to-end runner
├── requirements.txt                # Dependencies
├── PROJECT_INFO.md                 # This document
└── HOW_TO_RUN.md                   # Setup and run instructions
```

---

## Sample Output

Analysis of 8 articles on US–China tariffs:

| Source | Primary Frame | Sentiment | Bias Score |
|--------|--------------|-----------|------------|
| Reuters | Economic | Neutral | 0.12 (Low) |
| Fox News | Political | Negative | 0.45 (Medium) |
| The Guardian | Economic | Negative | 0.22 (Low) |
| Breitbart | Nationalist | Negative | 0.68 (High) |
| Al Jazeera | Political | Negative | 0.38 (Medium) |
| NPR | Humanitarian | Negative | 0.29 (Low) |
| Wired | Economic | Negative | 0.18 (Low) |
| Foreign Affairs | Security | Neutral | 0.15 (Low) |

**Key Findings:**
- Breitbart frames the event as nationalist self-defense; Foreign Affairs frames it as geopolitical strategy
- NPR focuses on the human cost to working families; Wired focuses on tech industry impact
- Fox News uses the most loaded vocabulary; Reuters uses the most neutral language

---

## Suggested Presentation Structure

1. Title — NFIE: Narrative Framing Intelligence Engine
2. Problem — Same event, different stories: how editorial framing shapes public perception
3. Architecture — 7-phase NLP pipeline overview
4. Phases 1–2 — Data ingestion, sentence embeddings, FAISS clustering
5. Phases 3–4 — Named entity recognition, zero-shot frame classification (include radar chart)
6. Phases 5–6 — Sentiment analysis, lexical bias detection (include bar charts)
7. Phase 7 — Comparison engine and divergence scoring
8. Demo — Streamlit dashboard walkthrough
9. Technology Stack — Summary table
10. Results — Divergence report with key findings
11. Future Work — Real-time monitoring, fine-tuned models, multilingual support
