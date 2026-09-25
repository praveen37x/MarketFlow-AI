"""
MarketFlow-AI: Pydantic Data Validation Schemas
"""
from typing import Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "HEALTHY"})
    version: str = Field(..., json_schema_extra={"example": "1.0.0"})
    engine: str = Field(..., json_schema_extra={"example": "DuckDB + Polars"})
    bronze_files: int
    silver_records: int
    gold_records: int
    catalog_path: str


class IngestRequest(BaseModel):
    symbol: str | None = Field(default=None, json_schema_extra={"example": "NIFTY"})
    count: int = Field(default=250, ge=10, le=5000)


class FeatureVectorResponse(BaseModel):
    symbol: str
    features: dict[str, float]


class AIPredictionResponse(BaseModel):
    symbol: str
    predicted_regime: str
    regime_id: int
    confidence: float
    anomaly_score: float
    is_anomaly: bool
    features: dict[str, float]


class AICopilotResponse(BaseModel):
    symbol: str
    current_price: float
    intraday_change_pct: float
    volatility_regime: str
    confidence: float
    anomaly_score: float
    risk_level: str
    market_outlook: str
    recommendation: str
    microstructure_signals: dict[str, Any]
    data_lineage: dict[str, Any]
