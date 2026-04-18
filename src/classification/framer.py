"""
src/classification/framer.py
──────────────────────────────
Phase 4: Frame Classification

WHAT IS FRAMING?
  The same event can be told through different "frames" — lenses that
  emphasize certain aspects and downplay others.

  Example: A factory closure can be framed as:
    - Economic:    "500 jobs lost, GDP impact of $2M"
    - Political:   "Government policy blamed for manufacturing decline"
    - Nationalist: "Foreign competition destroying American industry"
    - Humanitarian:"Families devastated, community torn apart"

HOW WE DETECT IT:
  Zero-shot classification using facebook/bart-large-mnli
    - A model trained on Natural Language Inference (NLI)
    - Given text T and label L, it predicts: does T entail L?
    - We rephrased: "This text is about [frame]" → score per frame
    - No fine-tuning needed — works out of the box

WHY ZERO-SHOT (not fine-tuned)?
  - We don't have labeled framing training data (expensive to create)
  - Zero-shot generalizes well with good label descriptions
  - Can add frames without retraining

PRODUCTION UPGRADE PATH:
  1. Collect 500+ human-labeled examples per frame
  2. Fine-tune RoBERTa with multi-label BCE loss
  3. Achieve ~85% F1 vs ~65% for zero-shot
"""

from typing import Optional

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from configs.settings import settings

try:
    from src.utils.logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ── Frame Descriptions ────────────────────────────────────────────────────────
# More descriptive labels = better zero-shot accuracy
# Empirically: full sentences outperform single words for MNLI classifiers

FRAME_HYPOTHESES = {
    "political": "This text discusses political power, government decisions, elections, or partisan conflicts.",
    "economic": "This text focuses on economic impacts, trade, markets, money, jobs, or financial consequences.",
    "emotional": "This text uses emotional appeals, human stories, fear, outrage, or personal suffering.",
    "security": "This text discusses national security, military, threats, surveillance, or defense.",
    "nationalist": "This text appeals to national identity, patriotism, sovereignty, or us-vs-them thinking.",
    "humanitarian": "This text focuses on human welfare, suffering, rights, or ethical concerns.",
    "legal": "This text discusses laws, regulations, courts, compliance, or legal consequences.",
    "scientific": "This text presents data, research, expert analysis, or evidence-based arguments.",
}


# ── Frame Classifier ──────────────────────────────────────────────────────────

class FrameClassifier:
    """
    Multi-label frame classifier using zero-shot NLI.

    Each article can belong to MULTIPLE frames — a Reuters article might be
    simultaneously economic AND political AND legal.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.CLASSIFIER_MODEL
        self.classifier = self._load_model()
        self.frame_labels = list(FRAME_HYPOTHESES.keys())

    def _load_model(self):
        """Load the zero-shot classification pipeline."""
        if not TRANSFORMERS_AVAILABLE:
            logger.error("transformers not installed")
            return None

        try:
            logger.info(f"Loading zero-shot classifier: {self.model_name}")
            clf = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=-1,  # CPU; change to 0 for GPU
            )
            logger.info("Zero-shot classifier loaded")
            return clf
        except Exception as e:
            logger.error(f"Failed to load classifier: {e}")
            return None

    def classify(self, text: str, top_k: int = 3) -> dict:
        """
        Classify text into frames.

        Args:
            text:   Article text (title + content excerpt recommended)
            top_k:  Return top K frames only

        Returns:
            {
              "frames": [
                {"frame": "economic", "score": 0.82},
                {"frame": "political", "score": 0.71},
              ],
              "primary_frame": "economic",
              "all_scores": {"economic": 0.82, "political": 0.71, ...}
            }
        """
        if not self.classifier:
            return self._fallback_classify(text)

        # Truncate to 512 tokens (model limit)
        text = text[:2000]

        result = self.classifier(
            text,
            candidate_labels=list(FRAME_HYPOTHESES.values()),
            multi_label=True,  # Multi-label: scores are independent, not softmax
        )

        # Map back from hypothesis → frame name
        hypothesis_to_frame = {v: k for k, v in FRAME_HYPOTHESES.items()}

        scores_by_frame = {}
        for label, score in zip(result["labels"], result["scores"]):
            frame = hypothesis_to_frame.get(label, label)
            scores_by_frame[frame] = round(float(score), 4)

        # Sort by score
        sorted_frames = sorted(
            scores_by_frame.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return {
            "frames": [
                {"frame": f, "score": s}
                for f, s in sorted_frames[:top_k]
            ],
            "primary_frame": sorted_frames[0][0] if sorted_frames else "unknown",
            "all_scores": scores_by_frame,
        }

    def classify_article(self, article: dict) -> dict:
        """Classify a single article and return enriched dict."""
        text = f"{article['title']}. {article['content'][:1500]}"
        frame_result = self.classify(text)
        return {**article, "frame_analysis": frame_result}

    def classify_batch(self, articles: list[dict]) -> list[dict]:
        """Classify multiple articles with progress logging."""
        logger.info(f"Classifying frames for {len(articles)} articles")
        results = []
        for i, article in enumerate(articles):
            classified = self.classify_article(article)
            results.append(classified)
            if (i + 1) % 10 == 0:
                logger.info(f"  Classified {i+1}/{len(articles)}")
        logger.info("Frame classification complete")
        return results

    def _fallback_classify(self, text: str) -> dict:
        """
        Keyword-based fallback when model unavailable.
        NOT production-ready — for testing pipeline without GPU/large model.
        """
        text_lower = text.lower()

        keyword_frames = {
            "political": ["government", "president", "congress", "election", "party", "political", "democrat", "republican", "white house", "senate"],
            "economic": ["economy", "trade", "tariff", "jobs", "market", "gdp", "price", "cost", "inflation", "growth", "recession"],
            "security": ["security", "military", "defense", "threat", "weapon", "intelligence", "surveillance", "war", "nato"],
            "nationalist": ["american", "patriot", "sovereignty", "border", "nation", "foreign", "domestic", "homeland"],
            "emotional": ["fear", "devastating", "heartbreak", "outrage", "shocking", "alarming", "tragedy", "crisis"],
            "humanitarian": ["families", "workers", "children", "poverty", "rights", "welfare", "suffering", "people"],
            "legal": ["law", "regulation", "court", "legal", "wto", "compliance", "legislation", "rules"],
            "scientific": ["study", "research", "data", "evidence", "analysis", "expert", "scientist", "statistics"],
        }

        scores = {}
        for frame, keywords in keyword_frames.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            scores[frame] = round(score / len(keywords), 4)

        sorted_frames = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return {
            "frames": [{"frame": f, "score": s} for f, s in sorted_frames[:3]],
            "primary_frame": sorted_frames[0][0] if sorted_frames else "unknown",
            "all_scores": scores,
            "method": "keyword_fallback",
        }


# ── Frame Comparison ──────────────────────────────────────────────────────────

def compare_frames(articles: list[dict]) -> dict:
    """
    Compare frame distributions across articles in the same cluster.

    Returns a comparison showing how differently each source framed the event.

    Example output:
    {
      "Reuters":      {"primary_frame": "economic",  "top_frames": [...]},
      "Fox News":     {"primary_frame": "political", "top_frames": [...]},
      "The Guardian": {"primary_frame": "economic",  "top_frames": [...]},
      "frame_divergence": 0.73,  # 0=identical, 1=completely different
    }
    """
    source_frames = {}
    for article in articles:
        frame_analysis = article.get("frame_analysis", {})
        if not frame_analysis:
            continue
        source_frames[article["source"]] = {
            "primary_frame": frame_analysis.get("primary_frame", "unknown"),
            "top_frames": frame_analysis.get("frames", []),
            "all_scores": frame_analysis.get("all_scores", {}),
        }

    # Calculate frame divergence across sources
    divergence = _calculate_frame_divergence(source_frames)

    return {
        "frames_by_source": source_frames,
        "frame_divergence_score": divergence,
        "summary": _build_frame_summary(source_frames),
    }


def _calculate_frame_divergence(source_frames: dict) -> float:
    """
    Measure how differently sources framed the same event.
    Returns 0.0 (all identical) to 1.0 (completely different).
    """
    if len(source_frames) < 2:
        return 0.0

    all_frames = list(FRAME_HYPOTHESES.keys())
    score_vectors = []

    for source_data in source_frames.values():
        all_scores = source_data.get("all_scores", {})
        vector = [all_scores.get(f, 0.0) for f in all_frames]
        score_vectors.append(vector)

    if not score_vectors:
        return 0.0

    # Average pairwise L1 distance between score vectors
    distances = []
    for i in range(len(score_vectors)):
        for j in range(i + 1, len(score_vectors)):
            v1, v2 = score_vectors[i], score_vectors[j]
            l1 = sum(abs(a - b) for a, b in zip(v1, v2))
            distances.append(l1)

    max_possible_l1 = 2.0 * len(all_frames)  # max distance between distributions
    avg_distance = sum(distances) / len(distances)
    return round(min(avg_distance / max_possible_l1, 1.0), 3)


def _build_frame_summary(source_frames: dict) -> str:
    """Generate a human-readable summary of framing differences."""
    if not source_frames:
        return "No frame data available."

    primary_frames = [
        f"{source}: {data['primary_frame']}"
        for source, data in source_frames.items()
    ]

    unique_primary = set(data["primary_frame"] for data in source_frames.values())
    if len(unique_primary) == 1:
        return f"All sources used the same primary frame: {list(unique_primary)[0]}."
    else:
        return f"Sources used different primary frames: {'; '.join(primary_frames)}."
