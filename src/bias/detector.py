"""
src/bias/detector.py
─────────────────────
Phase 6: Lexical Bias Detection

WHAT IS LEXICAL BIAS?
  Word choice itself encodes framing. Compare:
    "Militants attacked the convoy" vs "Freedom fighters engaged the convoy"
    "Illegal immigrants" vs "Undocumented migrants"
    "Regime" vs "Government"
    "Economic bullying" vs "Trade restrictions"

  These are the SAME events described with different lexical choices —
  each signaling a different ideological stance.

WHAT WE DO:
  1. TF-IDF per source: find words uniquely weighted by each outlet
  2. Power word detection: identify loaded, emotionally charged vocabulary
  3. Hedging language: "reportedly", "allegedly", "claimed" (signals doubt)
  4. Agency framing: who is the subject (actor) vs object (recipient)?
  5. Bias divergence score: how lexically different are the sources?

TECHNIQUE: TF-IDF (Term Frequency–Inverse Document Frequency)
  - TF: how often a word appears in THIS article
  - IDF: how rare the word is across ALL articles (same event)
  - High TF-IDF = this source uses this word far more than others
  - These high-TF-IDF words ARE the framing fingerprint
"""

import re
from collections import Counter, defaultdict
from math import log
from typing import Optional

try:
    from src.utils.logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ── Loaded/Power Word Lexicons ────────────────────────────────────────────────

POWER_WORDS = {
    # Negative/aggressive framing
    "aggression", "bully", "coerce", "exploit", "steal", "invade",
    "threaten", "attack", "retaliate", "punish", "impose", "force",
    "escalate", "spiral", "collapse", "catastrophe", "devastate",
    "condemn", "reject", "defy", "refuse", "demand", "ultimatum",
    # Positive/justified framing
    "protect", "defend", "secure", "ensure", "strengthen", "restore",
    "revive", "save", "preserve", "support", "boost", "empower",
    # Identity/nationalist framing
    "american", "patriot", "sovereignty", "homeland", "nation",
    "foreign", "adversary", "enemy", "rival", "competitor",
    # Economic alarm
    "recession", "crash", "crisis", "inflation", "surge", "plunge",
    "skyrocket", "devastation", "turmoil",
}

HEDGE_WORDS = {
    "allegedly", "reportedly", "claimed", "accused", "purportedly",
    "supposedly", "according to", "said to", "believed to",
    "critics say", "some argue", "officials claim",
}

AUTHORITY_WORDS = {
    "expert", "official", "analyst", "economist", "researcher",
    "professor", "spokesperson", "advisor", "scientist", "study",
    "data", "report", "evidence",
}

EMOTION_AMPLIFIERS = {
    "devastating", "shocking", "alarming", "outrageous", "horrifying",
    "stunning", "dramatic", "dire", "stark", "severe", "brutal",
    "radical", "extreme", "massive", "unprecedented",
}

# Common stop words to exclude from TF-IDF
STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "must",
    "that", "this", "these", "those", "it", "its", "as", "not",
    "also", "more", "new", "said", "says", "say",
}


# ── TF-IDF Engine ─────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Simple whitespace+punct tokenizer."""
    text = text.lower()
    tokens = re.findall(r'\b[a-z]{3,}\b', text)
    return [t for t in tokens if t not in STOP_WORDS]


def compute_tfidf(articles: list[dict]) -> dict[str, dict[str, float]]:
    """
    Compute TF-IDF scores for each article.

    WHY TF-IDF FOR BIAS DETECTION:
      If "Fox News" uses "invasion" 5 times while Reuters uses it 0 times,
      TF-IDF will surface "invasion" as a high-scoring term for Fox News.
      This reveals vocabulary choices unique to each source.

    Returns:
        {
          "article_id": {
            "word": tfidf_score,
            ...
          }
        }
    """
    # Step 1: Tokenize all articles
    article_tokens: dict[str, list[str]] = {}
    for article in articles:
        text = f"{article['title']} {article['content']}"
        article_tokens[article["id"]] = tokenize(text)

    num_docs = len(articles)

    # Step 2: Document frequency (how many docs contain each word)
    doc_freq: Counter = Counter()
    for tokens in article_tokens.values():
        for word in set(tokens):  # set: count each word once per doc
            doc_freq[word] += 1

    # Step 3: Compute TF-IDF per article
    tfidf_scores: dict[str, dict[str, float]] = {}

    for article in articles:
        tokens = article_tokens[article["id"]]
        term_freq = Counter(tokens)
        total_terms = max(len(tokens), 1)

        scores = {}
        for word, count in term_freq.items():
            tf = count / total_terms
            idf = log((num_docs + 1) / (doc_freq[word] + 1)) + 1  # Smoothed IDF
            scores[word] = round(tf * idf, 5)

        # Sort by score, keep top 30
        tfidf_scores[article["id"]] = dict(
            sorted(scores.items(), key=lambda x: x[1], reverse=True)[:30]
        )

    return tfidf_scores


# ── Bias Feature Extractor ────────────────────────────────────────────────────

class BiasDetector:
    """
    Extracts lexical bias signals from news articles.
    """

    def analyze_article(self, article: dict) -> dict:
        """
        Full bias analysis for a single article.

        Returns enriched article with bias signals.
        """
        text = f"{article['title']}. {article['content']}"
        text_lower = text.lower()
        words = tokenize(text)
        word_set = set(words)
        total_words = max(len(words), 1)

        # Power words used
        power_hits = [w for w in words if w in POWER_WORDS]
        power_density = round(len(power_hits) / total_words * 100, 2)

        # Hedging language
        hedge_hits = [phrase for phrase in HEDGE_WORDS if phrase in text_lower]

        # Authority appeals
        authority_hits = [w for w in words if w in AUTHORITY_WORDS]

        # Emotion amplifiers
        amplifier_hits = [w for w in words if w in EMOTION_AMPLIFIERS]

        # Passive vs active voice (simple heuristic)
        passive_patterns = re.findall(
            r'\b(?:was|were|been|being|is|are)\s+\w+ed\b', text_lower
        )
        active_ratio = round(
            1 - (len(passive_patterns) / max(len(re.findall(r'\b\w+\b', text_lower)) / 20, 1)),
            3,
        )

        # Bias score: combines power word density + amplifier count
        bias_score = round(
            min(
                (power_density * 0.5 + len(amplifier_hits) * 0.3 + len(hedge_hits) * 0.2) / 10,
                1.0,
            ),
            3,
        )

        return {
            **article,
            "bias_analysis": {
                "power_words": list(set(power_hits))[:15],
                "power_word_density": power_density,
                "hedge_phrases": hedge_hits,
                "authority_appeals": list(set(authority_hits))[:10],
                "emotion_amplifiers": list(set(amplifier_hits))[:10],
                "passive_constructions": len(passive_patterns),
                "active_ratio": active_ratio,
                "bias_score": bias_score,
            },
        }

    def analyze_batch(self, articles: list[dict]) -> list[dict]:
        """Analyze bias for all articles and add TF-IDF scores."""
        logger.info(f"Running bias detection on {len(articles)} articles")

        # First pass: individual article analysis
        analyzed = [self.analyze_article(a) for a in articles]

        # Second pass: TF-IDF (needs all articles for IDF calculation)
        tfidf_scores = compute_tfidf(articles)

        for article in analyzed:
            article["tfidf_signature"] = tfidf_scores.get(article["id"], {})

        logger.info("Bias detection complete")
        return analyzed


# ── Cross-Source Bias Comparison ──────────────────────────────────────────────

def compare_bias(articles: list[dict]) -> dict:
    """
    Compare lexical bias across articles about the same event.

    KEY INSIGHT: The same event described by different outlets will have
    different "TF-IDF fingerprints" — words that are distinctly overused
    by each outlet reveal their framing priorities.

    Returns:
        {
          "signature_by_source":  { "Reuters": {"word": score, ...}, ... },
          "exclusive_vocabulary":  { "Reuters": ["word1", ...], ... },
          "bias_scores":           { "Reuters": 0.2, "Fox News": 0.7, ...},
          "bias_divergence_score": 0.65,
          "most_biased_source":    "Fox News",
          "most_neutral_source":   "Reuters",
        }
    """
    # Group TF-IDF signatures by source
    source_signatures: dict[str, dict[str, float]] = defaultdict(dict)
    bias_scores: dict[str, float] = {}

    for article in articles:
        source = article["source"]
        sig = article.get("tfidf_signature", {})
        # Merge (keep highest score for each word per source)
        for word, score in sig.items():
            if word not in source_signatures[source] or score > source_signatures[source][word]:
                source_signatures[source][word] = score

        ba = article.get("bias_analysis", {})
        if ba:
            current = bias_scores.get(source, 0.0)
            bias_scores[source] = max(current, ba.get("bias_score", 0.0))

    # Find "exclusive vocabulary" — words in top-10 of one source but not others
    exclusive_vocab: dict[str, list[str]] = {}
    for source, sig in source_signatures.items():
        top_words = set(list(sig.keys())[:10])
        other_sources_top_words: set[str] = set()
        for other_source, other_sig in source_signatures.items():
            if other_source != source:
                other_sources_top_words.update(list(other_sig.keys())[:10])
        exclusive_vocab[source] = list(top_words - other_sources_top_words)

    # Bias divergence = how different are the bias scores?
    if len(bias_scores) > 1:
        scores = list(bias_scores.values())
        divergence = round(max(scores) - min(scores), 3)
    else:
        divergence = 0.0

    sorted_bias = sorted(bias_scores.items(), key=lambda x: x[1], reverse=True)

    return {
        "signature_by_source": {
            s: dict(list(sig.items())[:15])
            for s, sig in source_signatures.items()
        },
        "exclusive_vocabulary": exclusive_vocab,
        "bias_scores": bias_scores,
        "bias_divergence_score": divergence,
        "most_biased_source": sorted_bias[0][0] if sorted_bias else None,
        "most_neutral_source": sorted_bias[-1][0] if sorted_bias else None,
    }
