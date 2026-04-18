"""
src/ingestion/fetcher.py
────────────────────────
Phase 1: Data Ingestion

WHAT IT DOES:
  - Fetches news articles from NewsAPI OR loads a mock dataset
  - Normalizes all articles into a consistent schema
  - Saves to data/raw/ as JSON

WHY THIS SCHEMA:
  Every downstream module (embedding, NER, comparison) needs the same
  fields. Enforcing structure here means no KeyErrors later.

  Required fields:
    id         — unique hash (source + title + date) for dedup
    title      — article headline
    content    — full article body (or description if content unavailable)
    source     — news outlet name (e.g., "BBC", "Fox News")
    url        — original article URL
    date       — ISO 8601 publication date
    query      — what search term found this article (tracks event grouping)
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from configs.settings import settings

try:
    from src.utils.logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ── Schema ────────────────────────────────────────────────────────────────────

def make_article_id(source: str, title: str, date: str) -> str:
    """
    Generate a stable unique ID from source+title+date.
    Using a hash (not UUID) means the same article always gets the same ID —
    critical for deduplication across multiple fetches.
    """
    raw = f"{source}|{title}|{date}"
    return hashlib.md5(raw.encode()).hexdigest()


def normalize_article(
    title: str,
    content: str,
    source: str,
    url: str,
    date: str,
    query: str = "",
    metadata: Optional[dict] = None,
) -> dict:
    """
    Returns a fully normalized article dict.
    All downstream modules receive this exact shape.
    """
    return {
        "id": make_article_id(source, title, date),
        "title": title.strip(),
        "content": content.strip(),
        "source": source.strip(),
        "url": url.strip(),
        "date": date,
        "query": query,
        "char_count": len(content),
        "word_count": len(content.split()),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
    }


# ── NewsAPI Fetcher ────────────────────────────────────────────────────────────

class NewsAPIFetcher:
    """
    Fetches articles from NewsAPI (https://newsapi.org).
    Free tier: 100 requests/day, articles from last 30 days.
    """

    BASE_URL = "https://newsapi.org/v2/everything"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.NEWS_API_KEY
        if not self.api_key:
            logger.warning("No NEWS_API_KEY found — will fall back to mock data")

    def fetch(
        self,
        query: str,
        page_size: int = 20,
        language: str = "en",
        sort_by: str = "publishedAt",
    ) -> list[dict]:
        """
        Fetch articles matching a query string.

        Args:
            query:     Search term (e.g., "US China trade war")
            page_size: How many articles to return (max 100 for free tier)
            language:  ISO 639-1 language code
            sort_by:   relevancy | popularity | publishedAt

        Returns:
            List of normalized article dicts
        """
        if not self.api_key:
            logger.info("No API key — returning mock dataset")
            return get_mock_articles(query)

        params = {
            "q": query,
            "pageSize": page_size,
            "language": language,
            "sortBy": sort_by,
            "apiKey": self.api_key,
        }

        logger.info(f"Fetching NewsAPI: query='{query}' pageSize={page_size}")

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            articles = []
            for raw in data.get("articles", []):
                # NewsAPI sometimes returns [Removed] or empty content
                content = raw.get("content") or raw.get("description") or ""
                if not content or content == "[Removed]":
                    continue

                article = normalize_article(
                    title=raw.get("title", "Untitled"),
                    content=content,
                    source=raw.get("source", {}).get("name", "Unknown"),
                    url=raw.get("url", ""),
                    date=raw.get("publishedAt", datetime.now(timezone.utc).isoformat()),
                    query=query,
                    metadata={"author": raw.get("author", "")},
                )
                articles.append(article)

            logger.info(f"Fetched {len(articles)} articles for '{query}'")
            return articles

        except requests.RequestException as e:
            logger.error(f"NewsAPI request failed: {e}")
            return get_mock_articles(query)


# ── Mock Dataset ──────────────────────────────────────────────────────────────

def get_mock_articles(query: str = "test") -> list[dict]:
    """
    Returns a realistic mock dataset of 8 articles covering the same event
    (US-China trade war) from different outlets with different framings.

    WHY MOCK DATA:
      - Lets you build and test all phases without an API key
      - Controlled data = predictable test assertions
      - Different sources already have different tones baked in
    """

    raw_articles = [
        {
            "title": "US Imposes New Tariffs on Chinese Goods, Escalating Trade War",
            "content": (
                "The Biden administration announced sweeping new tariffs on Chinese imports "
                "Monday, marking a dramatic escalation in the ongoing trade conflict between "
                "the world's two largest economies. The tariffs, ranging from 25% to 100%, "
                "target electric vehicles, solar panels, and semiconductors. White House "
                "officials framed the move as necessary to protect American workers and "
                "counter unfair Chinese trade practices. Economists warn the tariffs could "
                "raise consumer prices and slow economic growth. China's commerce ministry "
                "condemned the action as 'economic bullying' and threatened retaliatory measures."
            ),
            "source": "Reuters",
            "url": "https://reuters.com/mock/tariff-1",
            "date": "2024-05-15T10:00:00Z",
        },
        {
            "title": "Biden's China Tariffs: Bold Move to Save American Jobs or Political Stunt?",
            "content": (
                "President Biden's dramatic tariff announcement has reignited the debate over "
                "America's economic future. Supporters say the 100% tariff on Chinese EVs is "
                "exactly the kind of muscular industrial policy needed to revive manufacturing "
                "in swing states like Michigan and Pennsylvania. Critics, including free-market "
                "economists, argue the tariffs are economically illiterate protectionism that "
                "will hurt American consumers. The timing — months before a presidential "
                "election — has raised eyebrows among trade experts who see this as campaign "
                "strategy dressed up as economic policy."
            ),
            "source": "Fox News",
            "url": "https://foxnews.com/mock/tariff-2",
            "date": "2024-05-15T11:30:00Z",
        },
        {
            "title": "US Tariffs on China Could Trigger Global Recession, Experts Warn",
            "content": (
                "International economists and trade bodies issued stark warnings Tuesday "
                "following Washington's announcement of steep tariffs on Chinese goods. "
                "The IMF said the measures risk destabilizing global supply chains that "
                "have only recently recovered from pandemic disruptions. The WTO's "
                "director-general called for restraint, noting that tit-for-tat trade "
                "barriers historically harm both sides. Developing nations expressed alarm "
                "that an escalating US-China trade war could choke off export growth "
                "critical to poverty reduction. Stock markets in Asia fell sharply on the news."
            ),
            "source": "The Guardian",
            "url": "https://theguardian.com/mock/tariff-3",
            "date": "2024-05-15T12:00:00Z",
        },
        {
            "title": "Standing Up to China: Why America's New Tariffs Are Long Overdue",
            "content": (
                "For too long, China has played by its own rules — flooding global markets "
                "with subsidized products, stealing American intellectual property, and "
                "systematically dismantling US manufacturing. The new tariffs are not "
                "aggression; they are self-defense. American workers in Ohio and Michigan "
                "have watched their livelihoods erode for decades while Washington did "
                "nothing. These tariffs send an unambiguous message: the era of economic "
                "appeasement is over. China must compete fairly or face consequences. "
                "National security demands we never again be dependent on a strategic "
                "adversary for critical technologies."
            ),
            "source": "Breitbart",
            "url": "https://breitbart.com/mock/tariff-4",
            "date": "2024-05-15T13:00:00Z",
        },
        {
            "title": "China Vows Retaliation as US Trade Tensions Spiral",
            "content": (
                "Beijing issued its strongest response yet to American trade measures on "
                "Tuesday, with senior officials vowing 'firm countermeasures' against what "
                "they described as unilateral protectionism. The Chinese government accused "
                "Washington of violating WTO rules and disrupting international trade norms. "
                "State media called the tariffs an attempt to suppress China's technological "
                "development and maintain American hegemony. Analysts in Beijing warned that "
                "China has significant leverage including rare earth minerals and Treasury "
                "bond holdings. Diplomatic channels remain open but sources describe "
                "back-channel talks as 'deeply strained.'"
            ),
            "source": "Al Jazeera",
            "url": "https://aljazeera.com/mock/tariff-5",
            "date": "2024-05-15T14:00:00Z",
        },
        {
            "title": "American Consumers Face Higher Prices as Tariff War Intensifies",
            "content": (
                "The human cost of Washington's trade war with Beijing is becoming increasingly "
                "clear to ordinary American families. A solar panel installation that cost "
                "$15,000 last year could soon cost $22,000. Electric vehicles from Chinese "
                "brands that offered affordable alternatives to Tesla are effectively priced "
                "out of the US market. Working-class families who stood to benefit most from "
                "cheaper clean energy technology may now be priced out. Consumer advocacy "
                "groups say the tariffs function as a regressive tax that falls hardest on "
                "those least able to afford it."
            ),
            "source": "NPR",
            "url": "https://npr.org/mock/tariff-6",
            "date": "2024-05-15T15:00:00Z",
        },
        {
            "title": "Tech Industry Warns Chip Tariffs Could Cripple US Innovation",
            "content": (
                "Silicon Valley executives and semiconductor companies are sounding alarms "
                "about sweeping new tariffs on Chinese technology components. Major chipmakers "
                "say the measures create dangerous supply chain uncertainty at exactly the "
                "wrong moment — when the US is trying to build domestic semiconductor capacity. "
                "The Semiconductor Industry Association warned that tariffs on Chinese "
                "packaging and assembly services could add billions in costs to the US chip "
                "industry. Several tech CEOs have quietly lobbied against the tariffs while "
                "publicly supporting the administration's broader China policy goals."
            ),
            "source": "Wired",
            "url": "https://wired.com/mock/tariff-7",
            "date": "2024-05-15T16:00:00Z",
        },
        {
            "title": "Trade War or Security War? The Real Stakes of US-China Tariffs",
            "content": (
                "Beneath the economic arguments over tariffs lies a deeper strategic struggle. "
                "Pentagon officials and national security analysts increasingly view trade "
                "policy as an extension of geopolitical competition. The US cannot allow "
                "China to dominate critical technologies — batteries, semiconductors, "
                "drone components — that will define the next generation of military power. "
                "From this perspective, higher consumer prices are an acceptable cost for "
                "strategic independence. China, for its part, views American tech restrictions "
                "as an existential threat to its development model. Both sides are digging in "
                "for a decades-long competition that transcends any single tariff announcement."
            ),
            "source": "Foreign Affairs",
            "url": "https://foreignaffairs.com/mock/tariff-8",
            "date": "2024-05-15T17:00:00Z",
        },
    ]

    normalized = []
    for raw in raw_articles:
        normalized.append(
            normalize_article(
                title=raw["title"],
                content=raw["content"],
                source=raw["source"],
                url=raw["url"],
                date=raw["date"],
                query=query,
            )
        )

    return normalized


# ── Storage ────────────────────────────────────────────────────────────────────

def save_articles(articles: list[dict], path: Optional[Path] = None) -> Path:
    """
    Save articles to JSON. Deduplicates by article ID before saving.

    WHY JSON (not a DB): For prototyping speed. Phase 8 can swap to
    PostgreSQL/SQLite with zero changes to upstream modules.
    """
    path = path or settings.ARTICLE_STORE_PATH
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing articles (to deduplicate across runs)
    existing = {}
    if path.exists():
        with open(path, "r") as f:
            for a in json.load(f):
                existing[a["id"]] = a

    # Merge new articles
    new_count = 0
    for article in articles:
        if article["id"] not in existing:
            existing[article["id"]] = article
            new_count += 1

    with open(path, "w") as f:
        json.dump(list(existing.values()), f, indent=2)

    logger.info(f"Saved {new_count} new articles. Total: {len(existing)} → {path}")
    return path


def load_articles(path: Optional[Path] = None) -> list[dict]:
    """Load all stored articles."""
    path = path or settings.ARTICLE_STORE_PATH
    path = Path(path)

    if not path.exists():
        logger.warning(f"Article store not found at {path}")
        return []

    with open(path, "r") as f:
        articles = json.load(f)

    logger.info(f"Loaded {len(articles)} articles from {path}")
    return articles
