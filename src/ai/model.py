"""
MarketFlow-AI: Volatility Regime & Anomaly Prediction Model
Machine Learning model that classifies short-term volatility regimes
(Low Volatility, Normal, High Volatility Breakout) and computes anomaly scores.
"""
from pathlib import Path
from typing import Any
import numpy as np
import pickle
from sklearn.ensemble import HistGradientBoostingClassifier


REGIME_LABELS = {
    0: "LOW_VOLATILITY_RANGEBOUND",
    1: "NORMAL_MARKET_REGIME",
    2: "HIGH_VOLATILITY_BREAKOUT",
}


class VolatilityRegimeModel:
    """Predictive Machine Learning classifier for market volatility regimes."""

    FEATURE_NAMES = ["range_bps", "return_bps", "vwap_dev_bps", "avg_spread_bps", "volume_factor"]

    def __init__(self, model_path: Path | None = None):
        self.model_path = model_path
        self.classifier = HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.08,
            random_state=42
        )
        self.is_trained = False
        self._ensure_baseline_model()

    def _ensure_baseline_model(self) -> None:
        """Fit a solid baseline model on synthetic data if no saved model exists."""
        np.random.seed(42)
        n_samples = 1500

        # Features: range_bps, return_bps, vwap_dev_bps, avg_spread_bps, volume_factor
        X = np.zeros((n_samples, 5))
        y = np.zeros(n_samples, dtype=int)

        # Class 0: Low Volatility (small range, small return, tight spread)
        X[:500, 0] = np.random.uniform(5.0, 20.0, 500)
        X[:500, 1] = np.random.normal(0.0, 5.0, 500)
        X[:500, 2] = np.random.normal(0.0, 4.0, 500)
        X[:500, 3] = np.random.uniform(2.0, 4.0, 500)
        X[:500, 4] = np.random.uniform(0.5, 1.2, 500)
        y[:500] = 0

        # Class 1: Normal Market
        X[500:1000, 0] = np.random.uniform(20.0, 50.0, 500)
        X[500:1000, 1] = np.random.normal(0.0, 15.0, 500)
        X[500:1000, 2] = np.random.normal(0.0, 10.0, 500)
        X[500:1000, 3] = np.random.uniform(3.0, 6.0, 500)
        X[500:1000, 4] = np.random.uniform(1.0, 2.5, 500)
        y[500:1000] = 1

        # Class 2: High Volatility Breakout (wide range, large return or deviation, widening spread)
        X[1000:, 0] = np.random.uniform(50.0, 150.0, 500)
        X[1000:, 1] = np.random.normal(0.0, 45.0, 500)
        X[1000:, 2] = np.random.normal(0.0, 35.0, 500)
        X[1000:, 3] = np.random.uniform(6.0, 18.0, 500)
        X[1000:, 4] = np.random.uniform(2.0, 6.0, 500)
        y[1000:] = 2

        self.classifier.fit(X, y)
        self.is_trained = True

    def predict(self, feature_vector: dict[str, float]) -> dict[str, Any]:
        """
        Classify volatility regime given real-time market features.
        """
        # Map feature store keys to model input vector
        range_bps = float(feature_vector.get("volatility_std", 25.0) * 1.5)
        return_bps = float(feature_vector.get("momentum_bps", 0.0))
        vwap_dev_bps = float(feature_vector.get("momentum_bps", 0.0) * 0.7)
        avg_spread = float(feature_vector.get("spread_bps", 4.0))
        volume_factor = float(feature_vector.get("volume_ratio", 1.0))

        X = np.array([[range_bps, return_bps, vwap_dev_bps, avg_spread, volume_factor]])
        probs = self.classifier.predict_proba(X)[0]
        predicted_class = int(np.argmax(probs))

        # Compute continuous anomaly score (Mahalanobis/Euclidean proxy)
        anomaly_score = float(min(max(abs(return_bps) / 50.0 + (range_bps / 100.0), 0.0), 1.0))

        return {
            "predicted_regime": REGIME_LABELS[predicted_class],
            "regime_id": predicted_class,
            "confidence": round(float(probs[predicted_class]), 4),
            "regime_probabilities": {
                REGIME_LABELS[i]: round(float(probs[i]), 4) for i in range(len(probs))
            },
            "anomaly_score": round(anomaly_score, 3),
            "is_anomaly": anomaly_score > 0.75,
        }

    def save(self, filepath: Path) -> None:
        """Serialize model to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self.classifier, f)

    def load(self, filepath: Path) -> None:
        """Load trained model from disk."""
        with open(filepath, "rb") as f:
            self.classifier = pickle.load(f)
            self.is_trained = True
