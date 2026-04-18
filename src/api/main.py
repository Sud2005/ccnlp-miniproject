"""
src/api/main.py
────────────────
Phase 8: FastAPI Backend

ENDPOINTS:
  POST /analyze-event      — Run full NFIE pipeline on submitted articles
  POST /ingest             — Fetch articles from NewsAPI by query
  GET  /clusters           — List all stored event clusters
  GET  /cluster/{id}       — Get specific cluster with comparison report
  GET  /health             — Health check

WHY FASTAPI (not Flask):
  - Native async support (important for I/O-bound model inference)
  - Auto-generated OpenAPI docs at /docs
  - Pydantic validation = no boilerplate input checking
  - 3x faster than Flask for concurrent requests
"""

from contextlib import asynccontextmanager
from typing import Optional
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from configs.settings import settings

try:
    from src.utils.logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ── Pydantic Models (Request/Response Schemas) ─────────────────────────────────

class ArticleInput(BaseModel):
    """Input schema for a single article."""
    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=50)
    source: str = Field(..., min_length=1, max_length=100)
    url: Optional[str] = ""
    date: Optional[str] = ""
    query: Optional[str] = ""


class AnalyzeEventRequest(BaseModel):
    """
    Request body for /analyze-event.
    Send 2–20 articles about the same event from different sources.
    """
    articles: list[ArticleInput] = Field(
        ...,
        min_length=2,
        max_length=20,
        description="2–20 articles about the same event from different sources"
    )
    run_ner: bool = Field(True, description="Run Named Entity Recognition")
    run_frames: bool = Field(True, description="Run frame classification")
    run_sentiment: bool = Field(True, description="Run sentiment analysis")
    run_bias: bool = Field(True, description="Run bias detection")


class IngestRequest(BaseModel):
    """Request body for /ingest — fetch articles by query."""
    query: str = Field(..., min_length=2, description="News search query")
    page_size: int = Field(10, ge=1, le=50)
    use_mock: bool = Field(False, description="Use mock data instead of real API")


# ── App Initialization ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup/shutdown lifecycle handler.
    Pre-warms models so the first request isn't slow.
    """
    logger.info("NFIE API starting up...")

    # Pre-warm NER model (takes 2-3 seconds on first load)
    try:
        from src.ner.extractor import NERExtractor
        app.state.ner = NERExtractor()
        logger.info("NER model loaded")
    except Exception as e:
        logger.warning(f"NER not loaded: {e}")
        app.state.ner = None

    # Pre-warm sentiment model
    try:
        from src.sentiment.analyzer import SentimentAnalyzer
        app.state.sentiment = SentimentAnalyzer()
        logger.info("Sentiment model loaded")
    except Exception as e:
        logger.warning(f"Sentiment model not loaded: {e}")
        app.state.sentiment = None

    # Frame classifier (large model — skip pre-warming in dev)
    app.state.framer = None  # Loaded on demand

    logger.info("NFIE API ready")
    yield

    logger.info("NFIE API shutting down")


app = FastAPI(
    title="Narrative Framing Intelligence Engine",
    description=(
        "Analyzes how different news outlets frame the same event. "
        "Detects differences in framing, sentiment, bias, and tone."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS (allow Streamlit dashboard to call the API) ───────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Quick health check endpoint for load balancers and monitoring."""
    return {
        "status": "ok",
        "version": "1.0.0",
        "models_loaded": {
            "ner": app.state.ner is not None,
            "sentiment": app.state.sentiment is not None,
        }
    }


@app.post("/analyze-event")
async def analyze_event(request: AnalyzeEventRequest):
    """
    Core NFIE endpoint.

    Accepts 2–20 articles about the same event and returns a full
    framing comparison report including:
      - Frame analysis per source
      - Sentiment comparison
      - Bias detection
      - Entity prominence comparison
      - Overall divergence score

    Example curl:
        curl -X POST http://localhost:8000/analyze-event \\
          -H "Content-Type: application/json" \\
          -d @sample_request.json
    """
    logger.info(f"/analyze-event called with {len(request.articles)} articles")

    # Convert Pydantic models to dicts
    from src.ingestion.fetcher import normalize_article
    articles = []
    for a in request.articles:
        normalized = normalize_article(
            title=a.title,
            content=a.content,
            source=a.source,
            url=a.url or "",
            date=a.date or "",
            query=a.query or "",
        )
        articles.append(normalized)

    try:
        # Run analysis phases
        if request.run_ner and app.state.ner:
            articles = app.state.ner.extract_batch(articles)

        if request.run_frames:
            from src.classification.framer import FrameClassifier
            if app.state.framer is None:
                app.state.framer = FrameClassifier()
            articles = app.state.framer.classify_batch(articles)

        if request.run_sentiment and app.state.sentiment:
            articles = app.state.sentiment.analyze_batch(articles)

        if request.run_bias:
            from src.bias.detector import BiasDetector
            bias_detector = BiasDetector()
            articles = bias_detector.analyze_batch(articles)

        # Run comparison
        from src.comparison.engine import ComparisonEngine
        engine = ComparisonEngine()
        report = engine.compare(articles)
        report["enriched_articles"] = articles

        return JSONResponse(content=report)

    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/ingest")
async def ingest_articles(request: IngestRequest):
    """
    Fetch articles from NewsAPI (or mock data) and store them.

    Returns the fetched and stored articles.
    """
    from src.ingestion.fetcher import NewsAPIFetcher, save_articles, get_mock_articles

    logger.info(f"/ingest called: query='{request.query}' mock={request.use_mock}")

    if request.use_mock:
        articles = get_mock_articles(request.query)
    else:
        fetcher = NewsAPIFetcher()
        articles = fetcher.fetch(request.query, page_size=request.page_size)

    if not articles:
        raise HTTPException(status_code=404, detail="No articles found for query")

    save_path = save_articles(articles)

    return {
        "status": "ok",
        "articles_fetched": len(articles),
        "stored_at": str(save_path),
        "sources": list(set(a["source"] for a in articles)),
        "preview": [
            {"title": a["title"], "source": a["source"]}
            for a in articles[:5]
        ],
    }


@app.post("/analyze-query")
async def analyze_query(request: IngestRequest):
    """
    Combined endpoint: Fetch + Analyze in one call.
    Fetches articles by query and immediately runs the full analysis pipeline.
    """
    from src.ingestion.fetcher import NewsAPIFetcher, get_mock_articles
    from src.comparison.engine import run_full_pipeline

    if request.use_mock:
        articles = get_mock_articles(request.query)
    else:
        fetcher = NewsAPIFetcher()
        articles = fetcher.fetch(request.query, page_size=request.page_size)

    if len(articles) < 2:
        raise HTTPException(
            status_code=400,
            detail="Need at least 2 articles to compare. Try a different query or use mock data."
        )

    try:
        report = run_full_pipeline(articles)
        return JSONResponse(content=report)
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/articles")
async def list_articles():
    """List all stored articles."""
    from src.ingestion.fetcher import load_articles
    articles = load_articles()
    return {
        "total": len(articles),
        "articles": [
            {
                "id": a["id"],
                "title": a["title"],
                "source": a["source"],
                "date": a["date"],
                "word_count": a.get("word_count", 0),
            }
            for a in articles
        ]
    }


@app.get("/mock-demo")
async def mock_demo():
    """
    Run a full demo analysis using the built-in mock dataset.
    No API key needed. Great for testing the full pipeline.
    """
    from src.ingestion.fetcher import get_mock_articles
    from src.comparison.engine import run_full_pipeline

    logger.info("Running mock demo...")
    articles = get_mock_articles("US China tariff")

    try:
        report = run_full_pipeline(articles)
        return JSONResponse(content=report)
    except Exception as e:
        logger.error(f"Demo error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Run directly ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,   # Auto-reload on code changes (dev mode)
        log_level="debug",
    )
