"""
tests/test_pipeline.py
────────────────────────
Unit tests for all NFIE phases.

RUN:
    cd /home/claude/nfie
    pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np

from src.ingestion.fetcher import (
    get_mock_articles, normalize_article, make_article_id,
    save_articles, load_articles
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_articles():
    """Standard mock dataset used across all tests."""
    return get_mock_articles("test query")


@pytest.fixture
def two_articles(mock_articles):
    """Minimal two-article set for comparison tests."""
    return mock_articles[:2]


# ── Phase 1: Ingestion ─────────────────────────────────────────────────────────

class TestIngestion:
    def test_mock_articles_returns_list(self, mock_articles):
        assert isinstance(mock_articles, list)
        assert len(mock_articles) >= 2

    def test_mock_articles_have_required_fields(self, mock_articles):
        required = {"id", "title", "content", "source", "url", "date", "query", "word_count"}
        for article in mock_articles:
            missing = required - set(article.keys())
            assert not missing, f"Article missing fields: {missing}"

    def test_article_id_is_deterministic(self):
        """Same inputs always produce same ID — critical for deduplication."""
        id1 = make_article_id("Reuters", "Test Title", "2024-01-01")
        id2 = make_article_id("Reuters", "Test Title", "2024-01-01")
        assert id1 == id2

    def test_article_id_differs_by_source(self):
        id1 = make_article_id("Reuters", "Test Title", "2024-01-01")
        id2 = make_article_id("Fox News", "Test Title", "2024-01-01")
        assert id1 != id2

    def test_normalize_article_strips_whitespace(self):
        a = normalize_article(
            title="  Spaced Title  ",
            content="  Some content here.  ",
            source="  Reuters  ",
            url="http://example.com",
            date="2024-01-01",
        )
        assert a["title"] == "Spaced Title"
        assert a["content"] == "Some content here."
        assert a["source"] == "Reuters"

    def test_word_count_is_nonzero(self, mock_articles):
        for a in mock_articles:
            assert a["word_count"] > 0

    def test_save_and_load_articles(self, mock_articles, tmp_path):
        path = tmp_path / "test_articles.json"
        save_articles(mock_articles, path=path)
        loaded = load_articles(path=path)
        assert len(loaded) == len(mock_articles)

    def test_save_deduplicates(self, mock_articles, tmp_path):
        path = tmp_path / "test_articles.json"
        save_articles(mock_articles, path=path)
        save_articles(mock_articles, path=path)  # Save same articles twice
        loaded = load_articles(path=path)
        assert len(loaded) == len(mock_articles)  # No duplicates


# ── Phase 2: Embeddings & Clustering ──────────────────────────────────────────

class TestEmbeddings:
    def test_embedder_produces_correct_shape(self, mock_articles):
        from src.clustering.embedder import ArticleEmbedder
        embedder = ArticleEmbedder()
        embeddings = embedder.embed_articles(mock_articles[:3])
        assert embeddings.shape == (3, embedder.embedding_dim)

    def test_embeddings_are_float32(self, mock_articles):
        from src.clustering.embedder import ArticleEmbedder
        embedder = ArticleEmbedder()
        embeddings = embedder.embed_articles(mock_articles[:2])
        assert embeddings.dtype == np.float32

    def test_embeddings_are_normalized(self, mock_articles):
        """L2-normalized embeddings have unit norm ≈ 1.0."""
        from src.clustering.embedder import ArticleEmbedder
        embedder = ArticleEmbedder()
        embeddings = embedder.embed_articles(mock_articles[:2])
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_similar_articles_have_high_cosine_similarity(self, mock_articles):
        """Two articles about the same event should be more similar than random."""
        from src.clustering.embedder import ArticleEmbedder
        embedder = ArticleEmbedder()
        # First 3 are about the same event, last would be different in real data
        embeddings = embedder.embed_articles(mock_articles[:3])
        # Cosine similarity = dot product (since normalized)
        sim_01 = float(embeddings[0] @ embeddings[1])
        sim_02 = float(embeddings[0] @ embeddings[2])
        # All mock articles are about same event — should all be fairly similar
        assert sim_01 > 0.5
        assert sim_02 > 0.5

    def test_faiss_store_add_and_search(self, mock_articles):
        from src.clustering.embedder import ArticleEmbedder, FAISSStore
        embedder = ArticleEmbedder()
        embeddings = embedder.embed_articles(mock_articles[:4])
        article_ids = [a["id"] for a in mock_articles[:4]]

        store = FAISSStore(embedding_dim=embedder.embedding_dim)
        store.add(embeddings, article_ids)
        assert store.index.ntotal == 4

        # Search with first article's embedding — should find itself as #1
        results = store.search(embeddings[0], k=3, threshold=0.0)
        assert len(results) > 0
        result_ids = [r[0] for r in results]
        assert article_ids[0] in result_ids

    def test_clustering_returns_clusters(self, mock_articles):
        from src.clustering.embedder import ArticleEmbedder, cluster_articles
        embedder = ArticleEmbedder()
        embeddings = embedder.embed_articles(mock_articles)
        clusters = cluster_articles(mock_articles, embeddings)
        assert len(clusters) >= 1
        # Total articles in all clusters should equal input
        total = sum(c["size"] for c in clusters)
        assert total == len(mock_articles)

    def test_faiss_save_and_load(self, mock_articles, tmp_path):
        from src.clustering.embedder import ArticleEmbedder, FAISSStore
        embedder = ArticleEmbedder()
        embeddings = embedder.embed_articles(mock_articles[:3])
        ids = [a["id"] for a in mock_articles[:3]]

        # Save
        store = FAISSStore(embedding_dim=embedder.embedding_dim)
        store.add(embeddings, ids)
        index_path = tmp_path / "test.bin"
        store.save(path=index_path)

        # Load
        store2 = FAISSStore(embedding_dim=embedder.embedding_dim)
        success = store2.load(path=index_path)
        assert success
        assert store2.index.ntotal == 3
        assert store2.article_ids == ids


# ── Phase 3: NER ──────────────────────────────────────────────────────────────

class TestNER:
    def test_ner_extractor_returns_entities(self, mock_articles):
        from src.ner.extractor import NERExtractor
        ner = NERExtractor()
        if ner.nlp is None:
            pytest.skip("spaCy model not installed")
        enriched = ner.extract_from_article(mock_articles[0])
        assert "entities" in enriched
        assert "entity_counts" in enriched
        assert isinstance(enriched["entities"], list)

    def test_ner_finds_expected_entities(self, mock_articles):
        from src.ner.extractor import NERExtractor
        ner = NERExtractor()
        if ner.nlp is None:
            pytest.skip("spaCy model not installed")

        # Article is about US-China trade — should find GPE entities
        article = mock_articles[0]  # Reuters article
        enriched = ner.extract_from_article(article)
        entity_types = {e["label"] for e in enriched["entities"]}
        # Should find at least one of: PERSON, ORG, GPE
        assert entity_types & {"PERSON", "ORG", "GPE"}, \
            f"Expected entity types not found. Got: {entity_types}"

    def test_cluster_validation_adds_coherence_score(self, mock_articles):
        from src.ner.extractor import NERExtractor, validate_cluster_with_ner
        ner = NERExtractor()
        if ner.nlp is None:
            pytest.skip("spaCy model not installed")
        enriched_articles = ner.extract_batch(mock_articles[:3])
        cluster = {"articles": enriched_articles, "size": 3}
        validated = validate_cluster_with_ner(cluster)
        assert "entity_coherence_score" in validated
        assert 0.0 <= validated["entity_coherence_score"] <= 1.0


# ── Phase 4: Frame Classification ─────────────────────────────────────────────

class TestFrameClassification:
    def test_fallback_classifier_works_without_model(self):
        from src.classification.framer import FrameClassifier
        framer = FrameClassifier.__new__(FrameClassifier)
        framer.classifier = None  # Force fallback
        framer.frame_labels = ["political", "economic", "security"]
        result = framer._fallback_classify(
            "The government announced new economic policy affecting trade"
        )
        assert "frames" in result
        assert "primary_frame" in result
        assert len(result["frames"]) > 0
        assert result["primary_frame"] in ["political", "economic", "security",
                                            "nationalist", "emotional", "humanitarian",
                                            "legal", "scientific"]

    def test_frame_comparison_returns_divergence(self, mock_articles):
        from src.classification.framer import FrameClassifier, compare_frames
        framer = FrameClassifier()
        enriched = framer.classify_batch(mock_articles[:3])
        comparison = compare_frames(enriched)
        assert "frame_divergence_score" in comparison
        assert 0.0 <= comparison["frame_divergence_score"] <= 1.0
        assert "frames_by_source" in comparison


# ── Phase 5: Sentiment ─────────────────────────────────────────────────────────

class TestSentiment:
    def test_lexicon_sentiment_positive(self):
        from src.sentiment.analyzer import SentimentAnalyzer
        analyzer = SentimentAnalyzer()
        result = analyzer._lexicon_sentiment("great success growth positive benefit")
        assert result["label"] == "positive"

    def test_lexicon_sentiment_negative(self):
        from src.sentiment.analyzer import SentimentAnalyzer
        analyzer = SentimentAnalyzer()
        result = analyzer._lexicon_sentiment("terrible crisis fail danger damage harm")
        assert result["label"] == "negative"

    def test_sentiment_compound_in_range(self, mock_articles):
        from src.sentiment.analyzer import SentimentAnalyzer
        analyzer = SentimentAnalyzer()
        enriched = analyzer.analyze_batch(mock_articles[:3])
        for a in enriched:
            compound = a["sentiment"]["overall_compound"]
            assert -1.0 <= compound <= 1.0, f"Compound {compound} out of range"

    def test_tone_analysis_returns_all_keys(self):
        from src.sentiment.analyzer import SentimentAnalyzer
        analyzer = SentimentAnalyzer()
        tone = analyzer.analyze_tone("The crisis is devastating and alarming")
        required_keys = {"alarm_score", "aggression_score", "measured_score",
                         "emotional_score", "dominant_tone"}
        assert required_keys.issubset(set(tone.keys()))

    def test_alarm_tone_detected(self):
        from src.sentiment.analyzer import SentimentAnalyzer
        analyzer = SentimentAnalyzer()
        tone = analyzer.analyze_tone(
            "The catastrophic crisis is devastating and alarming with dire consequences"
        )
        assert tone["alarm_score"] > 0

    def test_sentiment_comparison(self, mock_articles):
        from src.sentiment.analyzer import SentimentAnalyzer, compare_sentiment
        analyzer = SentimentAnalyzer()
        enriched = analyzer.analyze_batch(mock_articles[:4])
        comparison = compare_sentiment(enriched)
        assert "sentiment_by_source" in comparison
        assert "sentiment_divergence" in comparison
        assert 0.0 <= comparison["sentiment_divergence"] <= 1.0


# ── Phase 6: Bias Detection ────────────────────────────────────────────────────

class TestBiasDetection:
    def test_bias_detector_returns_bias_score(self, mock_articles):
        from src.bias.detector import BiasDetector
        detector = BiasDetector()
        enriched = detector.analyze_batch(mock_articles[:3])
        for a in enriched:
            assert "bias_analysis" in a
            assert "bias_score" in a["bias_analysis"]
            assert 0.0 <= a["bias_analysis"]["bias_score"] <= 1.0

    def test_high_bias_text_scores_higher(self):
        """Breitbart-style text should score higher bias than neutral Reuters text."""
        from src.bias.detector import BiasDetector
        detector = BiasDetector()

        neutral_article = {
            "id": "n1", "title": "Policy announced",
            "content": "The government announced a new trade policy affecting imports according to official data.",
            "source": "Neutral", "word_count": 15,
        }
        biased_article = {
            "id": "b1", "title": "China attacks America",
            "content": "China's devastating aggressive invasion of American markets must be stopped. "
                       "These foreign enemies are coercing and bullying patriots into submission.",
            "source": "Biased", "word_count": 25,
        }

        enriched = detector.analyze_batch([neutral_article, biased_article])
        neutral_score = enriched[0]["bias_analysis"]["bias_score"]
        biased_score = enriched[1]["bias_analysis"]["bias_score"]
        assert biased_score >= neutral_score, \
            f"Expected biased({biased_score}) >= neutral({neutral_score})"

    def test_tfidf_returns_scores(self, mock_articles):
        from src.bias.detector import compute_tfidf
        scores = compute_tfidf(mock_articles)
        assert len(scores) == len(mock_articles)
        for article_id, word_scores in scores.items():
            assert isinstance(word_scores, dict)
            assert all(isinstance(v, float) for v in word_scores.values())

    def test_bias_comparison(self, mock_articles):
        from src.bias.detector import BiasDetector, compare_bias
        detector = BiasDetector()
        enriched = detector.analyze_batch(mock_articles)
        comparison = compare_bias(enriched)
        assert "bias_scores" in comparison
        assert "most_biased_source" in comparison
        assert "bias_divergence_score" in comparison


# ── Phase 7: Comparison Engine ────────────────────────────────────────────────

class TestComparisonEngine:
    @pytest.fixture
    def fully_enriched_articles(self, mock_articles):
        """Run all phases to produce fully enriched articles."""
        from src.ner.extractor import NERExtractor
        from src.classification.framer import FrameClassifier
        from src.sentiment.analyzer import SentimentAnalyzer
        from src.bias.detector import BiasDetector

        articles = mock_articles[:4]

        ner = NERExtractor()
        if ner.nlp:
            articles = ner.extract_batch(articles)

        framer = FrameClassifier()
        articles = framer.classify_batch(articles)

        analyzer = SentimentAnalyzer()
        articles = analyzer.analyze_batch(articles)

        detector = BiasDetector()
        articles = detector.analyze_batch(articles)

        return articles

    def test_comparison_returns_required_keys(self, fully_enriched_articles):
        from src.comparison.engine import ComparisonEngine
        engine = ComparisonEngine()
        report = engine.compare(fully_enriched_articles)

        required_keys = {
            "event_summary", "articles_analyzed", "sources",
            "frame_comparison", "sentiment_comparison", "bias_comparison",
            "overall_divergence_score", "divergence_level",
            "key_differences", "per_article_summary",
        }
        missing = required_keys - set(report.keys())
        assert not missing, f"Report missing keys: {missing}"

    def test_divergence_score_in_range(self, fully_enriched_articles):
        from src.comparison.engine import ComparisonEngine
        engine = ComparisonEngine()
        report = engine.compare(fully_enriched_articles)
        score = report["overall_divergence_score"]
        assert 0.0 <= score <= 1.0

    def test_divergence_label_valid(self, fully_enriched_articles):
        from src.comparison.engine import ComparisonEngine
        engine = ComparisonEngine()
        report = engine.compare(fully_enriched_articles)
        valid_levels = {"Minimal", "Low", "Moderate", "High", "Extreme"}
        assert report["divergence_level"] in valid_levels

    def test_per_article_summary_count(self, fully_enriched_articles):
        from src.comparison.engine import ComparisonEngine
        engine = ComparisonEngine()
        report = engine.compare(fully_enriched_articles)
        assert len(report["per_article_summary"]) == len(fully_enriched_articles)


# ── Phase 10: Caching ─────────────────────────────────────────────────────────

class TestCaching:
    def test_cache_set_and_get(self, tmp_path):
        from src.utils.cache import NFIECache
        c = NFIECache(cache_dir=tmp_path / "cache")
        c.set("test_key", {"value": 42})
        result = c.get("test_key")
        assert result == {"value": 42}

    def test_cache_miss_returns_none(self, tmp_path):
        from src.utils.cache import NFIECache
        c = NFIECache(cache_dir=tmp_path / "cache")
        result = c.get("nonexistent_key")
        assert result is None

    def test_cache_delete(self, tmp_path):
        from src.utils.cache import NFIECache
        c = NFIECache(cache_dir=tmp_path / "cache")
        c.set("del_key", "value")
        c.delete("del_key")
        assert c.get("del_key") is None

    def test_batch_processor_deduplication(self, mock_articles):
        from src.utils.cache import BatchProcessor
        doubled = mock_articles + mock_articles
        unique = BatchProcessor.deduplicate_articles(doubled)
        assert len(unique) == len(mock_articles)

    def test_batch_processor_filter_short(self):
        from src.utils.cache import BatchProcessor
        articles = [
            {"id": "1", "word_count": 10, "title": "Short"},
            {"id": "2", "word_count": 100, "title": "Long"},
            {"id": "3", "word_count": 50, "title": "Medium"},
        ]
        filtered = BatchProcessor.filter_short_articles(articles, min_words=50)
        assert len(filtered) == 2
        assert all(a["word_count"] >= 50 for a in filtered)

    def test_model_registry_singleton(self):
        from src.utils.cache import ModelRegistry
        call_count = 0

        def loader():
            nonlocal call_count
            call_count += 1
            return {"model": "fake"}

        # Call twice — loader should only run once
        r1 = ModelRegistry.get_or_load("test_model_xyz", loader)
        r2 = ModelRegistry.get_or_load("test_model_xyz", loader)
        assert r1 is r2
        assert call_count == 1
        ModelRegistry._models.pop("test_model_xyz", None)  # Cleanup
