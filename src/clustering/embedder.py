"""
src/clustering/embedder.py
───────────────────────────
Phase 2: Event Clustering via Sentence Embeddings + FAISS

WHY EMBEDDINGS?
  Raw text can't be compared mathematically.
  Embeddings convert text → dense float vectors (e.g., 384 dimensions)
  where semantic similarity = geometric closeness.
  "US imposes tariffs on China" and "Washington hits Beijing with trade duties"
  will have vectors very close together in 384-dimensional space,
  even though they share zero words.

WHY FAISS?
  Facebook AI Similarity Search is the industry standard for:
    - Storing millions of vectors efficiently
    - Finding nearest neighbors in milliseconds (not seconds)
    - Running entirely in-memory (no DB needed for prototyping)
  Alternative: ChromaDB, Pinecone, Weaviate (for production scale)
"""

import json
import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

from configs.settings import settings

try:
    from src.utils.logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ── Embedder ──────────────────────────────────────────────────────────────────

class ArticleEmbedder:
    """
    Converts articles to embeddings and clusters them by topic.

    Model: all-MiniLM-L6-v2
      - 384-dimensional output
      - 80MB model size (fast to download)
      - State-of-the-art for semantic similarity tasks
      - 14,200 sentences/second on CPU
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")

    def embed_articles(self, articles: list[dict]) -> np.ndarray:
        """
        Convert a list of articles to a 2D numpy array of embeddings.

        We embed (title + content) concatenated.
        WHY: Title alone misses context; content alone misses the framing
        that's often concentrated in headlines.

        Returns:
            np.ndarray shape: (num_articles, embedding_dim)
        """
        if not articles:
            raise ValueError("No articles provided for embedding")

        texts = [
            f"{a['title']}. {a['content']}"
            for a in articles
        ]

        logger.info(f"Embedding {len(texts)} articles...")

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,  # L2 normalize → cosine sim = dot product
        )

        logger.info(f"Embeddings shape: {embeddings.shape}")
        return embeddings.astype(np.float32)


# ── FAISS Index ────────────────────────────────────────────────────────────────

class FAISSStore:
    """
    Manages the FAISS vector index.

    Index type: IndexFlatIP (Inner Product)
      - With L2-normalized vectors, inner product = cosine similarity
      - "Flat" = exact search (no approximation)
      - For >100k articles, switch to IndexIVFFlat for speed
    """

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        # IndexFlatIP: exact nearest-neighbor search using inner product
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.article_ids: list[str] = []  # Maps FAISS row index → article ID

    def add(self, embeddings: np.ndarray, article_ids: list[str]):
        """Add embeddings to the index with corresponding article IDs."""
        assert embeddings.shape[0] == len(article_ids), \
            "Number of embeddings must match number of article IDs"
        assert embeddings.dtype == np.float32, \
            "FAISS requires float32 arrays"

        self.index.add(embeddings)
        self.article_ids.extend(article_ids)
        logger.info(f"FAISS index now contains {self.index.ntotal} vectors")

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        threshold: float = 0.75,
    ) -> list[tuple[str, float]]:
        """
        Find k most similar articles to a query embedding.

        Returns:
            List of (article_id, similarity_score) tuples above threshold,
            sorted by similarity descending.
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            if score >= threshold:
                results.append((self.article_ids[idx], float(score)))

        return results

    def save(self, path: Optional[Path] = None):
        """Persist the FAISS index and ID mapping to disk."""
        path = Path(path or settings.FAISS_INDEX_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path))

        id_map_path = path.with_suffix(".ids.pkl")
        with open(id_map_path, "wb") as f:
            pickle.dump(self.article_ids, f)

        logger.info(f"FAISS index saved to {path}")
        logger.info(f"Article ID map saved to {id_map_path}")

    def load(self, path: Optional[Path] = None) -> bool:
        """Load index from disk. Returns True if successful."""
        path = Path(path or settings.FAISS_INDEX_PATH)
        id_map_path = path.with_suffix(".ids.pkl")

        if not path.exists() or not id_map_path.exists():
            logger.warning("No saved FAISS index found")
            return False

        self.index = faiss.read_index(str(path))
        with open(id_map_path, "rb") as f:
            self.article_ids = pickle.load(f)

        logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
        return True


# ── Clustering ────────────────────────────────────────────────────────────────

def cluster_articles(
    articles: list[dict],
    embeddings: np.ndarray,
    threshold: float = None,
) -> list[dict]:
    """
    Group articles about the same event into clusters.

    Algorithm: Agglomerative (Hierarchical) Clustering
      WHY not KMeans:
        - KMeans requires knowing K in advance — we don't know how many
          events are in our dataset
        - Agglomerative uses a distance threshold, so it finds K automatically
        - Better suited for variable-sized clusters (1 article vs 10)

    Distance metric: cosine (since embeddings are L2-normalized)

    Returns:
        List of cluster dicts:
        {
          "cluster_id": int,
          "articles": [article_dict, ...],
          "size": int,
          "centroid": np.ndarray (average embedding),
        }
    """
    threshold = threshold or (1.0 - settings.SIMILARITY_THRESHOLD)

    if len(articles) < 2:
        logger.warning("Need at least 2 articles to cluster")
        return [{"cluster_id": 0, "articles": articles, "size": len(articles)}]

    logger.info(f"Clustering {len(articles)} articles (threshold={threshold})")

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="cosine",
        linkage="average",  # UPGMA — balanced cluster sizes
    )
    labels = clustering.fit_predict(embeddings)

    # Group articles by cluster label
    cluster_map: dict[int, list] = {}
    for i, (article, label) in enumerate(zip(articles, labels)):
        if label not in cluster_map:
            cluster_map[label] = []
        cluster_map[label].append((i, article))

    # Build output
    clusters = []
    for cluster_id, members in sorted(cluster_map.items()):
        indices = [m[0] for m in members]
        cluster_articles = [m[1] for m in members]

        # Centroid = mean of member embeddings
        centroid = embeddings[indices].mean(axis=0)

        cluster = {
            "cluster_id": int(cluster_id),
            "articles": cluster_articles,
            "size": len(cluster_articles),
            "centroid": centroid.tolist(),
            "sources": [a["source"] for a in cluster_articles],
        }
        clusters.append(cluster)

    # Sort by cluster size (largest first)
    clusters.sort(key=lambda c: c["size"], reverse=True)

    logger.info(f"Found {len(clusters)} clusters")
    for c in clusters[:5]:  # Log top 5
        logger.info(f"  Cluster {c['cluster_id']}: {c['size']} articles from {c['sources']}")

    return clusters


# ── Pipeline Entry Point ───────────────────────────────────────────────────────

def build_index_and_clusters(
    articles: list[dict],
    save_index: bool = True,
) -> tuple[FAISSStore, np.ndarray, list[dict]]:
    """
    Full Phase 2 pipeline:
      1. Embed articles
      2. Build FAISS index
      3. Cluster into events

    Returns:
        (faiss_store, embeddings_array, clusters_list)
    """
    # Step 1: Embed
    embedder = ArticleEmbedder()
    embeddings = embedder.embed_articles(articles)

    # Step 2: FAISS
    store = FAISSStore(embedding_dim=embedder.embedding_dim)
    article_ids = [a["id"] for a in articles]
    store.add(embeddings, article_ids)

    if save_index:
        store.save()

    # Step 3: Cluster
    clusters = cluster_articles(articles, embeddings)

    return store, embeddings, clusters


def save_clusters(clusters: list[dict], path: Optional[Path] = None) -> Path:
    """Save clusters to JSON (excluding centroid arrays which aren't JSON-serializable)."""
    path = path or (settings.DATA_PROCESSED_DIR / "clusters.json")
    path = Path(path)

    # Convert centroid to list for JSON serialization
    serializable = []
    for c in clusters:
        sc = {k: v for k, v in c.items() if k != "centroid"}
        serializable.append(sc)

    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)

    logger.info(f"Clusters saved to {path}")
    return path
