"""
Tests for AI & Machine Learning Pipeline
"""
from src.ai.model import VolatilityRegimeModel, REGIME_LABELS


def test_volatility_regime_model_prediction():
    model = VolatilityRegimeModel()
    dummy_features = {
        "volatility_std": 12.0,
        "momentum_bps": 2.5,
        "spread_bps": 3.0,
        "volume_ratio": 1.0,
    }

    pred = model.predict(dummy_features)
    assert "predicted_regime" in pred
    assert pred["predicted_regime"] in REGIME_LABELS.values()
    assert 0.0 <= pred["confidence"] <= 1.0
    assert 0.0 <= pred["anomaly_score"] <= 1.0
    assert isinstance(pred["is_anomaly"], (bool, int))
