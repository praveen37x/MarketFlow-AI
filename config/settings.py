"""
MarketFlow-AI Configuration Settings
Environment-based settings using standard Pydantic BaseModel.
"""
import os
from pathlib import Path
from pydantic import BaseModel, Field


class Settings(BaseModel):
    # App & Meta
    PROJECT_NAME: str = "MarketFlow-AI"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default_factory=lambda: os.getenv("ENVIRONMENT", "production"))
    DEBUG: bool = Field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")

    # Base Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"
    BRONZE_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "bronze"
    SILVER_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "silver"
    GOLD_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "gold"
    CATALOG_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "catalog"
    DUCKDB_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "catalog" / "marketflow.duckdb"

    # Ingestion Parameters
    DEFAULT_SYMBOLS: list[str] = ["NIFTY", "BANKNIFTY", "FINNIFTY", "SENSEX", "RELIANCE", "HDFCBANK"]
    STREAM_BATCH_SIZE: int = 500
    MICRO_BATCH_INTERVAL_SEC: float = 0.5

    # Lakehouse SLAs
    MAX_TIMESTAMP_SKEW_SEC: int = 86400
    DUCKDB_MEMORY_LIMIT: str = "4GB"
    DUCKDB_THREADS: int = 4

    # ML & Feature Store
    FEATURE_WINDOW_MINUTES: list[int] = [1, 5, 15, 60]
    MODEL_DIR: Path = Path(__file__).resolve().parent.parent / "models"
    MODEL_PATH: Path = Path(__file__).resolve().parent.parent / "models" / "volatility_regime_model.joblib"

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000


settings = Settings()
