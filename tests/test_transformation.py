"""
Tests for Transformation Layer (Data Contracts & Silver Cleaner)
"""
import polars as pl
from src.transformation.contracts import DataContractEnforcer


def test_data_contract_enforcer_valid_data():
    df = pl.DataFrame({
        "event_id": ["e1", "e2"],
        "timestamp": ["2026-01-01T10:00:00Z", "2026-01-01T10:00:01Z"],
        "symbol": ["NIFTY", "NIFTY"],
        "price": [24500.0, 24505.0],
        "bid": [24499.0, 24504.0],
        "ask": [24501.0, 24506.0],
        "volume": [150, 200],
    })

    valid_df, quarantine_df, report = DataContractEnforcer.validate_and_filter(df)
    assert len(valid_df) == 2
    assert len(quarantine_df) == 0
    assert report.passed is True
    assert report.pass_rate_pct == 100.0


def test_data_contract_quarantining_corrupt_data():
    # Record 1: valid
    # Record 2: negative price (violates rule)
    # Record 3: ask < bid (violates spread rule)
    df = pl.DataFrame({
        "event_id": ["e1", "e2", "e3"],
        "timestamp": ["2026-01-01T10:00:00Z", "2026-01-01T10:00:01Z", "2026-01-01T10:00:02Z"],
        "symbol": ["NIFTY", "NIFTY", "NIFTY"],
        "price": [24500.0, -100.0, 24510.0],
        "bid": [24499.0, 100.0, 24520.0],
        "ask": [24501.0, 102.0, 24500.0],
        "volume": [150, 200, 300],
    })

    valid_df, quarantine_df, report = DataContractEnforcer.validate_and_filter(df)
    assert len(valid_df) == 1
    assert len(quarantine_df) == 2
    assert report.passed is False
    assert report.price_violations >= 1
    assert report.spread_violations >= 1
