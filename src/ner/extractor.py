"""
src/ner/extractor.py
─────────────────────
Phase 3: Named Entity Recognition

WHY NER?
  Two articles might be about the same event but use different names:
    "Biden" vs "the President" vs "the White House"
    "Beijing" vs "China" vs "the Chinese government"

  NER extracts these canonical entities so we can:
  1. Validate clusters (same entities = likely same event)
  2. Build entity-level bias profiles (does outlet A always frame Entity X negatively?)
  3. Show which people/places/orgs appear prominently per source

spaCy Model: en_core_web_sm
  - Small (12MB), fast
  - Recognizes: PERSON, ORG, GPE (geopolitical), DATE, MONEY, PERCENT, etc.
  - Production alternative: en_core_web_trf (transformer-based, more accurate)
"""

from collections import Counter, defaultdict
from typing import Optional

try:
    import spacy
    from spacy.tokens import Doc
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

from configs.settings import settings

try:
    from src.utils.logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ── Entity Types We Care About ─────────────────────────────────────────────────
RELEVANT_ENTITY_TYPES = {
    "PERSON",   # Named people (Biden, Xi Jinping)
    "ORG",      # Organizations (WTO, Tesla, Pentagon)
    "GPE",      # Countries, cities, states (China, Washington, Michigan)
    "NORP",     # Nationalities/political groups (Chinese, Republicans)
    "PRODUCT",  # Products (EVs, semiconductors)
    "EVENT",    # Named events (trade war, G7 summit)
    "LAW",      # Laws/regulations (WTO rules, tariff code)
    "MONEY",    # Monetary amounts ($15,000)
    "PERCENT",  # Percentages (25%, 100%)
}


# ── NER Extractor ─────────────────────────────────────────────────────────────

class NERExtractor:
    """
    Extracts named entities from article text using spaCy.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.SPACY_MODEL
        self.nlp = self._load_model()

    def _load_model(self):
        """Load spaCy model with graceful fallback."""
        if not SPACY_AVAILABLE:
            logger.error("spaCy not installed. Run: pip install spacy")
            return None

        try:
            nlp = spacy.load(self.model_name)
            logger.info(f"Loaded spaCy model: {self.model_name}")
            return nlp
        except OSError:
            logger.error(
                f"spaCy model '{self.model_name}' not found. "
                f"Run: python -m spacy download {self.model_name}"
            )
            return None

    def extract(self, text: str) -> list[dict]:
        """
        Extract entities from a single text string.

        Returns:
            List of entity dicts:
            {
              "text":  "Biden",
              "label": "PERSON",
              "start": 4,
              "end":   9,
            }
        """
        if not self.nlp:
            return []

        # spaCy processes up to 1M chars; truncate for speed
        text = text[:10000]
        doc = self.nlp(text)

        entities = []
        for ent in doc.ents:
            if ent.label_ in RELEVANT_ENTITY_TYPES:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                })

        return entities

    def extract_from_article(self, article: dict) -> dict:
        """
        Run NER on a full article (title + content).

        Returns enriched article with 'entities' and 'entity_counts' fields.
        """
        full_text = f"{article['title']}. {article['content']}"
        entities = self.extract(full_text)

        # Count entity mentions by type
        entity_counts: dict[str, Counter] = defaultdict(Counter)
        for ent in entities:
            entity_counts[ent["label"]][ent["text"]] += 1

        # Convert to plain dict for JSON serialization
        counts_serializable = {
            label: dict(counter)
            for label, counter in entity_counts.items()
        }

        return {
            **article,
            "entities": entities,
            "entity_counts": counts_serializable,
            "entity_summary": {
                label: list(counter.most_common(5))
                for label, counter in entity_counts.items()
            },
        }

    def extract_batch(self, articles: list[dict]) -> list[dict]:
        """
        Process multiple articles efficiently using spaCy's pipe().

        WHY pipe() instead of looping:
          spaCy's nlp.pipe() batches texts internally, reusing model
          memory and running tokenization in parallel — ~4x faster.
        """
        if not self.nlp:
            return articles

        logger.info(f"Running NER on {len(articles)} articles")

        full_texts = [f"{a['title']}. {a['content'][:5000]}" for a in articles]

        enriched = []
        for article, doc in zip(articles, self.nlp.pipe(full_texts, batch_size=50)):
            entities = []
            entity_counts: dict[str, Counter] = defaultdict(Counter)

            for ent in doc.ents:
                if ent.label_ in RELEVANT_ENTITY_TYPES:
                    entities.append({
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                    })
                    entity_counts[ent.label_][ent.text] += 1

            enriched.append({
                **article,
                "entities": entities,
                "entity_counts": {
                    label: dict(counter)
                    for label, counter in entity_counts.items()
                },
                "entity_summary": {
                    label: list(counter.most_common(5))
                    for label, counter in entity_counts.items()
                },
            })

        logger.info("NER extraction complete")
        return enriched


# ── Cluster Validation ────────────────────────────────────────────────────────

def validate_cluster_with_ner(cluster: dict) -> dict:
    """
    Use entity overlap to score cluster cohesion.

    LOGIC:
      If articles in a cluster share many named entities (same people,
      places, orgs), they're likely about the same event.
      Low entity overlap = cluster might be a false positive.

    Returns cluster enriched with:
      - shared_entities: entities appearing in >50% of articles
      - entity_coherence_score: 0.0–1.0 (1.0 = perfect overlap)
    """
    articles = cluster.get("articles", [])
    if not articles:
        return cluster

    # Gather all entity texts per article
    entity_sets = []
    for article in articles:
        ec = article.get("entity_counts", {})
        article_entities = set()
        for label_entities in ec.values():
            article_entities.update(label_entities.keys())
        entity_sets.append(article_entities)

    if not any(entity_sets):
        return {**cluster, "entity_coherence_score": 0.0, "shared_entities": []}

    # Count how many articles mention each entity
    entity_mention_count: Counter = Counter()
    for es in entity_sets:
        for ent in es:
            entity_mention_count[ent] += 1

    # "Shared" = appears in >50% of articles
    threshold = max(2, len(articles) * 0.5)
    shared = [
        ent for ent, count in entity_mention_count.items()
        if count >= threshold
    ]

    # Coherence score = avg Jaccard similarity between all article pairs
    if len(entity_sets) < 2:
        coherence = 1.0
    else:
        jaccard_scores = []
        for i in range(len(entity_sets)):
            for j in range(i + 1, len(entity_sets)):
                a, b = entity_sets[i], entity_sets[j]
                if not a and not b:
                    continue
                intersection = len(a & b)
                union = len(a | b)
                jaccard_scores.append(intersection / union if union > 0 else 0)
        coherence = sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0.0

    return {
        **cluster,
        "shared_entities": shared[:20],  # Top 20
        "entity_coherence_score": round(coherence, 3),
    }


# ── Cross-Cluster Entity Comparison ──────────────────────────────────────────

def compare_entity_framing(articles: list[dict]) -> dict:
    """
    Given articles from the same cluster (same event), compare which
    entities each source mentions and how prominently.

    WHY THIS MATTERS:
      Source A might mention "workers" 5 times and "shareholders" 0 times.
      Source B might mention "shareholders" 4 times and "workers" 0 times.
      This is framing through entity salience — a real, measurable bias signal.

    Returns:
        {
          "entity": "Biden",
          "type": "PERSON",
          "mentions_by_source": {"Reuters": 3, "Fox News": 1, ...}
        }
    """
    # Aggregate entity mentions by source
    source_entity_counts: dict[str, Counter] = defaultdict(Counter)
    for article in articles:
        source = article.get("source", "Unknown")
        for label, entities in article.get("entity_counts", {}).items():
            for entity_text, count in entities.items():
                source_entity_counts[source][entity_text] += count

    # Gather all unique entities
    all_entities: set[str] = set()
    for counter in source_entity_counts.values():
        all_entities.update(counter.keys())

    # Build comparison table
    comparison = []
    for entity in all_entities:
        mentions_by_source = {
            source: counter.get(entity, 0)
            for source, counter in source_entity_counts.items()
        }
        total = sum(mentions_by_source.values())
        if total < 2:  # Skip rarely mentioned entities
            continue

        comparison.append({
            "entity": entity,
            "mentions_by_source": mentions_by_source,
            "total_mentions": total,
        })

    # Sort by total mentions
    comparison.sort(key=lambda x: x["total_mentions"], reverse=True)

    return {
        "entity_comparison": comparison[:30],
        "sources_analyzed": list(source_entity_counts.keys()),
    }
