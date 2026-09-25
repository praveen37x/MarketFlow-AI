"""
MarketFlow-AI: High-Performance FastAPI Lakehouse Service
Provides institutional REST endpoints for real-time market data,
Lakehouse Medallion tier telemetry, Feature Store vectors, and AI inference.
"""
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from src.ingestion.collectors import StreamCollector
from src.transformation.cleaner import SilverCleaner
from src.lakehouse.engine import DuckDBEngine
from src.lakehouse.marts import GoldMartsBuilder
from src.lakehouse.feature_store import FeatureStore
from src.ai.model import VolatilityRegimeModel
from src.ai.sentiment_agent import MarketAnalystAgent
from src.api.schemas import (
    HealthResponse,
    IngestRequest,
    FeatureVectorResponse,
    AIPredictionResponse,
    AICopilotResponse,
)

# Application state singletons
collector: StreamCollector | None = None
cleaner: SilverCleaner | None = None
engine: DuckDBEngine | None = None
marts_builder: GoldMartsBuilder | None = None
feature_store: FeatureStore | None = None
ai_model: VolatilityRegimeModel | None = None
analyst_agent: MarketAnalystAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize lakehouse components and bootstrap historical seed data."""
    global collector, cleaner, engine, marts_builder, feature_store, ai_model, analyst_agent

    # Ensure directories
    settings.BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    settings.SILVER_DIR.mkdir(parents=True, exist_ok=True)
    settings.GOLD_DIR.mkdir(parents=True, exist_ok=True)
    settings.CATALOG_DIR.mkdir(parents=True, exist_ok=True)

    collector = StreamCollector(settings.BRONZE_DIR, batch_size=settings.STREAM_BATCH_SIZE)
    cleaner = SilverCleaner(settings.BRONZE_DIR, settings.SILVER_DIR)
    engine = DuckDBEngine(settings.DUCKDB_PATH, settings.SILVER_DIR, settings.GOLD_DIR)
    marts_builder = GoldMartsBuilder(engine, settings.GOLD_DIR)
    feature_store = FeatureStore(engine)
    ai_model = VolatilityRegimeModel()
    analyst_agent = MarketAnalystAgent(engine, feature_store, ai_model)

    # If no bronze data exists yet, bootstrap an initial batch
    bronze_files = list(settings.BRONZE_DIR.glob("*.parquet"))
    if not bronze_files:
        collector.ingest_batch(count=300)
        cleaner.process_bronze_files()
        engine.refresh_views()
        marts_builder.build_all_marts()

    yield

    if engine:
        engine.close()


app = FastAPI(
    title="MarketFlow-AI Lakehouse Service",
    description="Real-Time Market Data Lakehouse & AI Intelligence Engine powered by DuckDB, Polars, and Machine Learning.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
async def root() -> dict[str, str]:
    return {
        "service": "MarketFlow-AI",
        "description": "Real-Time Market Data Lakehouse & AI Copilot",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_check": "/api/v1/health",
    }


@app.get("/api/v1/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check() -> HealthResponse:
    """Return health metrics across Medallion lakehouse tiers."""
    bronze_count = len(list(settings.BRONZE_DIR.glob("*.parquet")))
    
    silver_records = 0
    try:
        silver_rows = engine.query("SELECT COUNT(*) AS cnt FROM silver_ticks;")
        silver_records = silver_rows[0]["cnt"] if silver_rows else 0
    except Exception:
        silver_records = 0

    gold_records = 0
    try:
        gold_rows = engine.query("SELECT COUNT(*) AS cnt FROM mart_ohlcv_1m;")
        gold_records = gold_rows[0]["cnt"] if gold_rows else 0
    except Exception:
        gold_records = 0

    return HealthResponse(
        status="HEALTHY",
        version="1.0.0",
        engine="DuckDB 1.5.5 + Polars 1.44.2 (Zero JVM)",
        bronze_files=bronze_count,
        silver_records=silver_records,
        gold_records=gold_records,
        catalog_path=str(settings.DUCKDB_PATH),
    )


@app.post("/api/v1/ingest/trigger", tags=["Ingestion"])
async def trigger_ingestion(req: IngestRequest) -> dict[str, Any]:
    """Ingest a new micro-batch into Bronze, process to Silver, and refresh Gold."""
    symbols = [req.symbol] if req.symbol else None
    bronze_path = collector.ingest_batch(count=req.count, symbols=symbols)
    silver_path, audit = cleaner.process_bronze_files()
    engine.refresh_views()
    marts_builder.build_all_marts()

    return {
        "status": "SUCCESS",
        "bronze_partition": str(bronze_path),
        "silver_partition": str(silver_path),
        "records_ingested": req.count,
        "contract_audit": audit.to_dict() if audit else None,
    }


@app.get("/api/v1/market/ticks", tags=["Market Data"])
async def get_recent_ticks(
    symbol: str = Query("NIFTY", description="Market ticker symbol"),
    limit: int = Query(50, ge=1, le=500),
) -> list[dict[str, Any]]:
    """Retrieve the latest streaming market ticks for a symbol."""
    sql = f"""
    SELECT 
        event_id,
        parsed_time AS timestamp,
        symbol,
        price,
        bid,
        ask,
        spread,
        spread_bps,
        volume,
        trade_side
    FROM silver_ticks
    WHERE symbol = '{symbol}'
    ORDER BY parsed_time DESC
    LIMIT {limit};
    """
    return engine.query(sql)


@app.get("/api/v1/analytics/ohlcv", tags=["Analytics"])
async def get_ohlcv(
    symbol: str = Query("NIFTY", description="Symbol"),
    limit: int = Query(30, ge=1, le=200),
) -> list[dict[str, Any]]:
    """Fetch 1-minute OHLCV candlestick aggregates from the Gold analytical mart."""
    sql = f"""
    SELECT 
        bar_time,
        symbol,
        open,
        high,
        low,
        close,
        total_volume,
        vwap,
        tick_count,
        avg_spread_bps
    FROM mart_ohlcv_1m
    WHERE symbol = '{symbol}'
    ORDER BY bar_time DESC
    LIMIT {limit};
    """
    return engine.query(sql)


@app.get("/api/v1/analytics/features", response_model=FeatureVectorResponse, tags=["Feature Store"])
async def get_features(symbol: str = Query("NIFTY")) -> FeatureVectorResponse:
    """Retrieve real-time feature vector for ML scoring."""
    vec = feature_store.get_realtime_feature_vector(symbol)
    return FeatureVectorResponse(symbol=symbol, features=vec)


@app.get("/api/v1/ai/predict", response_model=AIPredictionResponse, tags=["AI & Machine Learning"])
async def predict_regime(symbol: str = Query("NIFTY")) -> AIPredictionResponse:
    """Run real-time ML inference to classify market volatility regime and anomaly score."""
    vec = feature_store.get_realtime_feature_vector(symbol)
    pred = ai_model.predict(vec)
    return AIPredictionResponse(
        symbol=symbol,
        predicted_regime=pred["predicted_regime"],
        regime_id=pred["regime_id"],
        confidence=pred["confidence"],
        anomaly_score=pred["anomaly_score"],
        is_anomaly=pred["is_anomaly"],
        features=vec,
    )


@app.get("/api/v1/ai/copilot", response_model=AICopilotResponse, tags=["AI & Machine Learning"])
async def ai_copilot_analysis(symbol: str = Query("NIFTY")) -> AICopilotResponse:
    """Generate structured AI Market Copilot brief combining Lakehouse data and ML models."""
    analysis = analyst_agent.analyze_symbol(symbol)
    return AICopilotResponse(**analysis)
