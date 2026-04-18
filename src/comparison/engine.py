"""
src/comparison/engine.py
─────────────────────────
Phase 7: Comparison Engine

The core NFIE output module. Takes a cluster of articles about the same
event and produces a structured comparison report.

WHAT IT COMBINES:
  - Frame analysis (Phase 4): How was the event framed?
  - Sentiment analysis (Phase 5): What emotional charge was applied?
  - Bias detection (Phase 6): What words reveal ideological slant?
  - Entity comparison (Phase 3): Who/what was mentioned prominently?

FINAL OUTPUT STRUCTURE:
  {
    "event_summary": "...",
    "articles_analyzed": 8,
    "sources": ["Reuters", "Fox News", ...],
    "frame_comparison": {...},
    "sentiment_comparison": {...},
    "bias_comparison": {...},
    "entity_comparison": {...},
    "overall_divergence_score": 0.74,
    "divergence_level": "High",
    "key_differences": ["Reuters focuses on economics...", ...],
  }
"""

from typing import Optional

from src.classification.framer import compare_frames
from src.sentiment.analyzer import compare_sentiment
from src.bias.detector import compare_bias
from src.ner.extractor import compare_entity_framing

try:
    from src.utils.logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ── Comparison Engine ─────────────────────────────────────────────────────────

class ComparisonEngine:
    """
    Aggregates all analysis modules into a unified comparison report.
    """

    def compare(self, articles: list[dict]) -> dict:
        """
        Run full comparison on a cluster of articles about the same event.

        Args:
            articles: List of fully-enriched article dicts (all phases run)

        Returns:
            Complete comparison report dict
        """
        if not articles:
            return {"error": "No articles provided"}

        logger.info(f"Running comparison on {len(articles)} articles from: "
                    f"{[a['source'] for a in articles]}")

        sources = [a["source"] for a in articles]

        # ── Frame Comparison ──────────────────────────────────────────────────
        frame_comparison = compare_frames(articles)

        # ── Sentiment Comparison ──────────────────────────────────────────────
        sentiment_comparison = compare_sentiment(articles)

        # ── Bias Comparison ───────────────────────────────────────────────────
        bias_comparison = compare_bias(articles)

        # ── Entity Comparison ─────────────────────────────────────────────────
        entity_comparison = compare_entity_framing(articles)

        # ── Overall Divergence Score ──────────────────────────────────────────
        overall_divergence = self._calculate_overall_divergence(
            frame_comparison,
            sentiment_comparison,
            bias_comparison,
        )

        # ── Key Differences (human-readable) ─────────────────────────────────
        key_differences = self._extract_key_differences(
            articles,
            frame_comparison,
            sentiment_comparison,
            bias_comparison,
        )

        # ── Event Summary ─────────────────────────────────────────────────────
        event_summary = self._generate_event_summary(articles)

        report = {
            "event_summary": event_summary,
            "articles_analyzed": len(articles),
            "sources": sources,
            "frame_comparison": frame_comparison,
            "sentiment_comparison": sentiment_comparison,
            "bias_comparison": bias_comparison,
            "entity_comparison": entity_comparison,
            "overall_divergence_score": overall_divergence,
            "divergence_level": self._divergence_label(overall_divergence),
            "key_differences": key_differences,
            "per_article_summary": self._per_article_summary(articles),
        }

        logger.info(f"Comparison complete. Divergence: {overall_divergence} "
                    f"({report['divergence_level']})")
        return report

    def _calculate_overall_divergence(
        self,
        frame_comparison: dict,
        sentiment_comparison: dict,
        bias_comparison: dict,
    ) -> float:
        """
        Weighted combination of individual divergence scores.

        Weights (tunable):
          - Frame divergence:     40% (framing is the primary signal)
          - Sentiment divergence: 35% (emotional charge is highly visible)
          - Bias divergence:      25% (lexical signals)
        """
        frame_div = frame_comparison.get("frame_divergence_score", 0.0)
        sentiment_div = sentiment_comparison.get("sentiment_divergence", 0.0)
        bias_div = bias_comparison.get("bias_divergence_score", 0.0)

        # Normalize bias_div (it's on a 0–1 scale already but can be > 1)
        bias_div = min(bias_div, 1.0)

        weighted = (
            frame_div * 0.40
            + sentiment_div * 0.35
            + bias_div * 0.25
        )
        return round(min(weighted, 1.0), 3)

    def _divergence_label(self, score: float) -> str:
        """Map score to human-readable divergence level."""
        if score < 0.2:
            return "Minimal"
        elif score < 0.4:
            return "Low"
        elif score < 0.6:
            return "Moderate"
        elif score < 0.8:
            return "High"
        else:
            return "Extreme"

    def _extract_key_differences(
        self,
        articles: list[dict],
        frame_comparison: dict,
        sentiment_comparison: dict,
        bias_comparison: dict,
    ) -> list[str]:
        """
        Generate human-readable bullets describing the most notable differences.
        """
        differences = []

        # Frame differences
        frames_by_source = frame_comparison.get("frames_by_source", {})
        if len(frames_by_source) >= 2:
            frame_summary = frame_comparison.get("summary", "")
            if frame_summary:
                differences.append(f"FRAMING: {frame_summary}")

        # Sentiment differences
        most_neg = sentiment_comparison.get("most_negative_source")
        most_pos = sentiment_comparison.get("most_positive_source")
        sent_range = sentiment_comparison.get("sentiment_range", 0.0)

        if most_neg and most_pos and most_neg != most_pos and sent_range > 0.1:
            differences.append(
                f"SENTIMENT: {most_neg} uses significantly more negative language "
                f"than {most_pos} (range: {sent_range:.2f})"
            )

        # Bias differences
        most_biased = bias_comparison.get("most_biased_source")
        most_neutral = bias_comparison.get("most_neutral_source")
        if most_biased and most_neutral and most_biased != most_neutral:
            differences.append(
                f"LEXICAL BIAS: {most_biased} uses more loaded/partisan vocabulary; "
                f"{most_neutral} uses more neutral language"
            )

        # Exclusive vocabulary
        excl_vocab = bias_comparison.get("exclusive_vocabulary", {})
        for source, words in excl_vocab.items():
            if words:
                differences.append(
                    f"VOCABULARY: {source} uniquely uses: {', '.join(words[:5])}"
                )

        # Alarm level comparison
        sentiment_by_source = sentiment_comparison.get("sentiment_by_source", {})
        alarm_levels = {
            s: d.get("alarm_score", 0.0)
            for s, d in sentiment_by_source.items()
            if "alarm_score" in d
        }
        if alarm_levels:
            max_alarm_source = max(alarm_levels, key=alarm_levels.get)
            if alarm_levels[max_alarm_source] > 0.3:
                differences.append(
                    f"ALARM LEVEL: {max_alarm_source} uses significantly more "
                    f"crisis/alarm language (score: {alarm_levels[max_alarm_source]:.2f})"
                )

        return differences[:8]  # Cap at 8 bullet points

    def _generate_event_summary(self, articles: list[dict]) -> str:
        """
        Generate a brief event summary from the most neutral-seeming article
        (highest word count + lowest bias score = likely most comprehensive).
        """
        if not articles:
            return ""

        # Score articles by neutrality proxy (longer + less biased)
        def neutrality_score(a):
            word_count = a.get("word_count", 0)
            bias = a.get("bias_analysis", {}).get("bias_score", 0.5)
            return word_count * (1 - bias)

        most_neutral = max(articles, key=neutrality_score)
        title = most_neutral.get("title", "")
        content = most_neutral.get("content", "")
        source = most_neutral.get("source", "")

        # Take first 2 sentences of content
        sentences = content.split(". ")[:2]
        summary = ". ".join(sentences)
        if len(summary) > 300:
            summary = summary[:300] + "..."

        return f"{title} — {summary} (Source: {source})"

    def _per_article_summary(self, articles: list[dict]) -> list[dict]:
        """
        One-line summary per article for the dashboard table view.
        """
        summaries = []
        for article in articles:
            frame = article.get("frame_analysis", {}).get("primary_frame", "unknown")
            sentiment = article.get("sentiment", {})
            compound = sentiment.get("overall_compound", 0.0)
            tone = sentiment.get("tone", {}).get("dominant_tone", "unknown")
            bias = article.get("bias_analysis", {}).get("bias_score", 0.0)

            summaries.append({
                "source": article["source"],
                "title": article["title"][:80] + ("..." if len(article["title"]) > 80 else ""),
                "primary_frame": frame,
                "sentiment_compound": compound,
                "sentiment_label": "Positive" if compound > 0.1 else ("Neutral" if compound > -0.1 else "Negative"),
                "dominant_tone": tone,
                "bias_score": bias,
                "url": article.get("url", ""),
                "date": article.get("date", ""),
            })

        return summaries


# ── Pipeline Runner ────────────────────────────────────────────────────────────

def run_full_pipeline(articles: list[dict]) -> dict:
    """
    Run all analysis phases on a list of articles, then compare.

    This is the function called by the FastAPI endpoint in Phase 8.
    It lazily imports modules to avoid circular dependencies.

    Phases executed:
      3 → NER
      4 → Frame Classification
      5 → Sentiment
      6 → Bias Detection
      7 → Comparison
    """
    from src.ner.extractor import NERExtractor
    from src.classification.framer import FrameClassifier
    from src.sentiment.analyzer import SentimentAnalyzer
    from src.bias.detector import BiasDetector

    logger.info(f"Starting full pipeline on {len(articles)} articles")

    # Phase 3: NER
    ner = NERExtractor()
    articles = ner.extract_batch(articles)

    # Phase 4: Frame Classification
    framer = FrameClassifier()
    articles = framer.classify_batch(articles)

    # Phase 5: Sentiment
    sentiment = SentimentAnalyzer()
    articles = sentiment.analyze_batch(articles)

    # Phase 6: Bias
    bias = BiasDetector()
    articles = bias.analyze_batch(articles)

    # Phase 7: Compare
    engine = ComparisonEngine()
    report = engine.compare(articles)
    report["enriched_articles"] = articles

    logger.info("Full pipeline complete")
    return report
