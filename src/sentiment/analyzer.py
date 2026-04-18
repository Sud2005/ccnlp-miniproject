"""
src/sentiment/analyzer.py
──────────────────────────
Phase 5: Sentiment & Tone Analysis

MODEL: cardiffnlp/twitter-roberta-base-sentiment-latest
  - Fine-tuned RoBERTa on 58M tweets
  - 3-class output: Negative / Neutral / Positive
  - Handles informal language well (good for news too)
  - Alternative: distilbert-base-uncased-finetuned-sst-2-english (2-class)

WHAT WE MEASURE:
  1. Overall sentiment (pos/neg/neutral) per article
  2. Sentence-level sentiment (where is the negative language concentrated?)
  3. Tone indicators (urgent, alarming, measured, neutral, celebratory)
  4. Sentiment delta between sources (same event, different emotional charge)
"""

import re
from typing import Optional

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from configs.settings import settings

try:
    from src.utils.logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ── Tone Indicators ────────────────────────────────────────────────────────────
# Hand-crafted lexicons for tone detection (complement to ML sentiment)

ALARM_WORDS = {
    "crisis", "collapse", "catastrophe", "devastating", "alarming",
    "dire", "dangerous", "escalation", "spiral", "catastrophic",
    "threat", "chaos", "breakdown", "failure", "emergency", "stark",
}

AGGRESSIVE_WORDS = {
    "attack", "strike", "retaliate", "condemn", "aggressive", "hostile",
    "confront", "battle", "fight", "war", "bully", "coerce", "punish",
    "demand", "ultimatum", "refuse", "reject", "defy",
}

MEASURED_WORDS = {
    "analysis", "data", "report", "study", "according", "expert",
    "evidence", "research", "indicate", "suggest", "estimate",
    "official", "policy", "framework", "assess", "examine",
}

EMOTIONAL_WORDS = {
    "devastating", "heartbreaking", "outrage", "shocking", "fear",
    "hope", "desperate", "suffering", "victim", "tragedy", "brave",
    "struggle", "pain", "proud", "horrifying",
}


# ── Sentiment Analyzer ────────────────────────────────────────────────────────

class SentimentAnalyzer:
    """Analyzes sentiment and tone of news articles."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.SENTIMENT_MODEL
        self.sentiment_pipe = self._load_model()

    def _load_model(self):
        if not TRANSFORMERS_AVAILABLE:
            logger.error("transformers not installed")
            return None

        try:
            logger.info(f"Loading sentiment model: {self.model_name}")
            pipe = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=-1,
                truncation=True,
                max_length=512,
            )
            logger.info("Sentiment model loaded")
            return pipe
        except Exception as e:
            logger.warning(f"Could not load sentiment model: {e}. Using fallback.")
            return None

    def analyze_text(self, text: str) -> dict:
        """
        Analyze sentiment of a text string.

        Returns:
            {
              "label":       "negative",     # positive | neutral | negative
              "score":       0.92,           # confidence 0–1
              "compound":    -0.72,          # mapped to -1..+1 scale
            }
        """
        if self.sentiment_pipe:
            return self._ml_sentiment(text)
        else:
            return self._lexicon_sentiment(text)

    def _ml_sentiment(self, text: str) -> dict:
        """ML-based sentiment using transformer model."""
        try:
            text = text[:512]  # Respect token limit
            result = self.sentiment_pipe(text)[0]
            label = result["label"].lower()

            # Normalize labels (different models use different naming)
            label_map = {
                "label_0": "negative", "label_1": "neutral", "label_2": "positive",
                "neg": "negative", "neu": "neutral", "pos": "positive",
                "negative": "negative", "neutral": "neutral", "positive": "positive",
            }
            label = label_map.get(label, label)
            score = float(result["score"])

            # Compound score: negative=-1, neutral=0, positive=1
            compound_map = {"negative": -score, "neutral": 0.0, "positive": score}
            compound = compound_map.get(label, 0.0)

            return {
                "label": label,
                "score": round(score, 4),
                "compound": round(compound, 4),
            }
        except Exception as e:
            logger.warning(f"ML sentiment failed: {e}")
            return self._lexicon_sentiment(text)

    def _lexicon_sentiment(self, text: str) -> dict:
        """Simple lexicon-based sentiment fallback (no ML required)."""
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))

        pos_words = {"good", "great", "positive", "success", "growth", "benefit",
                     "opportunity", "improve", "prosper", "strong", "win", "gain"}
        neg_words = {"bad", "terrible", "negative", "fail", "loss", "damage",
                     "harm", "crisis", "decline", "weak", "threat", "danger"}

        pos_count = len(words & pos_words)
        neg_count = len(words & neg_words)
        total = pos_count + neg_count

        if total == 0:
            return {"label": "neutral", "score": 0.5, "compound": 0.0}

        if pos_count > neg_count:
            score = pos_count / total
            return {"label": "positive", "score": round(score, 4), "compound": round(score * 0.5, 4)}
        elif neg_count > pos_count:
            score = neg_count / total
            return {"label": "negative", "score": round(score, 4), "compound": round(-score * 0.5, 4)}
        else:
            return {"label": "neutral", "score": 0.5, "compound": 0.0}

    def analyze_tone(self, text: str) -> dict:
        """
        Detect tone indicators beyond simple pos/neg sentiment.

        Returns multi-dimensional tone profile:
        {
          "alarm_score":     0.3,  # How alarming/crisis-framed is the text?
          "aggression_score": 0.2, # How combative/adversarial?
          "measured_score":  0.6,  # How data-driven/analytical?
          "emotional_score": 0.4,  # How emotionally charged?
          "dominant_tone":   "measured",
        }
        """
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        total_words = max(len(words), 1)
        word_set = set(words)

        alarm_hits = len(word_set & ALARM_WORDS)
        aggressive_hits = len(word_set & AGGRESSIVE_WORDS)
        measured_hits = len(word_set & MEASURED_WORDS)
        emotional_hits = len(word_set & EMOTIONAL_WORDS)

        # Normalize by word count (per-1000-words rate)
        scale = 1000 / total_words
        alarm_score = round(min(alarm_hits * scale / 10, 1.0), 3)
        aggression_score = round(min(aggressive_hits * scale / 10, 1.0), 3)
        measured_score = round(min(measured_hits * scale / 10, 1.0), 3)
        emotional_score = round(min(emotional_hits * scale / 10, 1.0), 3)

        # Dominant tone
        tone_scores = {
            "alarm": alarm_score,
            "aggression": aggression_score,
            "measured": measured_score,
            "emotional": emotional_score,
        }
        dominant_tone = max(tone_scores, key=tone_scores.get)

        return {
            "alarm_score": alarm_score,
            "aggression_score": aggression_score,
            "measured_score": measured_score,
            "emotional_score": emotional_score,
            "dominant_tone": dominant_tone,
        }

    def analyze_article(self, article: dict) -> dict:
        """Full sentiment analysis for one article."""
        text = f"{article['title']}. {article['content']}"

        sentiment = self.analyze_text(article["title"])  # Title sentiment is often most loaded
        body_sentiment = self.analyze_text(article["content"][:1000])
        tone = self.analyze_tone(text)

        # Sentence-level analysis (first 10 sentences for speed)
        sentences = re.split(r'(?<=[.!?])\s+', article["content"])[:10]
        sentence_sentiments = []
        for sent in sentences:
            if len(sent.strip()) < 20:
                continue
            ss = self.analyze_text(sent)
            sentence_sentiments.append({
                "text": sent[:100] + "..." if len(sent) > 100 else sent,
                **ss,
            })

        return {
            **article,
            "sentiment": {
                "headline": sentiment,
                "body": body_sentiment,
                "sentences": sentence_sentiments,
                "tone": tone,
                "overall_compound": round(
                    (sentiment["compound"] * 0.4 + body_sentiment["compound"] * 0.6), 4
                ),
            },
        }

    def analyze_batch(self, articles: list[dict]) -> list[dict]:
        """Analyze sentiment for a list of articles."""
        logger.info(f"Analyzing sentiment for {len(articles)} articles")
        return [self.analyze_article(a) for a in articles]


# ── Cross-Source Sentiment Comparison ─────────────────────────────────────────

def compare_sentiment(articles: list[dict]) -> dict:
    """
    Compare sentiment across articles covering the same event.

    This is the core NFIE metric: given the SAME event, how differently
    do outlets charge it emotionally?

    Returns:
        {
          "sentiment_by_source": {
            "Reuters":   {"compound": -0.3, "label": "negative"},
            "Fox News":  {"compound": -0.7, "label": "negative"},
          },
          "most_negative_source": "Fox News",
          "most_positive_source": "Reuters",
          "sentiment_range":       0.4,    # difference between most pos and most neg
          "sentiment_divergence":  0.62,   # normalized divergence score 0–1
        }
    """
    sentiment_by_source = {}
    for article in articles:
        sent = article.get("sentiment", {})
        if not sent:
            continue
        source = article["source"]
        compound = sent.get("overall_compound", 0.0)
        sentiment_by_source[source] = {
            "compound": compound,
            "label": sent.get("headline", {}).get("label", "neutral"),
            "tone": sent.get("tone", {}).get("dominant_tone", "unknown"),
            "alarm_score": sent.get("tone", {}).get("alarm_score", 0.0),
        }

    if not sentiment_by_source:
        return {"error": "No sentiment data available"}

    compounds = [v["compound"] for v in sentiment_by_source.values()]
    sentiment_range = round(max(compounds) - min(compounds), 3)

    # Divergence = standard deviation of compound scores (normalized 0–1)
    if len(compounds) > 1:
        mean = sum(compounds) / len(compounds)
        variance = sum((c - mean) ** 2 for c in compounds) / len(compounds)
        std = variance ** 0.5
        divergence = round(min(std / 0.5, 1.0), 3)  # 0.5 = approx max std
    else:
        divergence = 0.0

    sorted_by_compound = sorted(
        sentiment_by_source.items(), key=lambda x: x[1]["compound"]
    )

    return {
        "sentiment_by_source": sentiment_by_source,
        "most_negative_source": sorted_by_compound[0][0] if sorted_by_compound else None,
        "most_positive_source": sorted_by_compound[-1][0] if sorted_by_compound else None,
        "sentiment_range": sentiment_range,
        "sentiment_divergence": divergence,
    }
