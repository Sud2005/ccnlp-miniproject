import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).parent.parent

class Settings:
    APP_ENV: str = os.getenv("APP_ENV", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "DEBUG")
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", 8000))
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    CLASSIFIER_MODEL: str = os.getenv("CLASSIFIER_MODEL", "facebook/bart-large-mnli")
    SENTIMENT_MODEL: str = os.getenv("SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest")
    SPACY_MODEL: str = "en_core_web_sm"

    DATA_RAW_DIR: Path = ROOT_DIR / "data" / "raw"
    DATA_PROCESSED_DIR: Path = ROOT_DIR / "data" / "processed"
    DATA_EMBEDDINGS_DIR: Path = ROOT_DIR / "data" / "embeddings"
    MODELS_DIR: Path = ROOT_DIR / "models"
    LOGS_DIR: Path = ROOT_DIR / "logs"
    CACHE_DIR: Path = ROOT_DIR / ".cache"

    FAISS_INDEX_PATH: Path = ROOT_DIR / "data" / "embeddings" / "faiss_index.bin"
    ARTICLE_STORE_PATH: Path = ROOT_DIR / "data" / "processed" / "articles.json"

    SIMILARITY_THRESHOLD: float = 0.75
    MIN_CLUSTER_SIZE: int = 2
    MAX_CLUSTERS: int = 50

    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", 86400))

    FRAME_LABELS: list = [
        "political",
        "economic",
        "emotional",
        "security",
        "nationalist",
        "humanitarian",
        "legal",
        "scientific"
    ]

    def ensure_dirs(self):
        for d in [self.DATA_RAW_DIR, self.DATA_PROCESSED_DIR, self.DATA_EMBEDDINGS_DIR, 
                  self.MODELS_DIR, self.LOGS_DIR, self.CACHE_DIR]:
            d.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.ensure_dirs()
