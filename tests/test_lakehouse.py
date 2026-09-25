"""
Tests for Lakehouse Layer (DuckDB Engine & Feature Store)
"""
from pathlib import Path
from src.lakehouse.engine import DuckDBEngine
from src.lakehouse.feature_store import FeatureStore


def test_duckdb_engine_initialization(tmp_path: Path):
    db_path = tmp_path / "test_marketflow.duckdb"
    silver_dir = tmp_path / "silver"
    gold_dir = tmp_path / "gold"
    silver_dir.mkdir()
    gold_dir.mkdir()

    engine = DuckDBEngine(db_path, silver_dir, gold_dir)
    res = engine.query("SELECT 1 AS val;")
    assert len(res) == 1
    assert res[0]["val"] == 1

    feature_store = FeatureStore(engine)
    vec = feature_store.get_realtime_feature_vector("NIFTY")
    assert "momentum_bps" in vec
    assert "volatility_std" in vec
    assert "spread_bps" in vec
    engine.close()
