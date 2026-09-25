"""
MarketFlow-AI Ingestion: Synthetic Market Tick & Event Generator
Simulates realistic high-frequency market microstructure dynamics,
including geometric Brownian motion with volatility clustering, Poisson jumps,
and bid-ask spread dynamics across index & equity symbols.
"""
import time
import uuid
import random
import math
from datetime import datetime, timezone
from typing import Generator, Any


BASE_PRICES = {
    "NIFTY": 24500.0,
    "BANKNIFTY": 51200.0,
    "FINNIFTY": 23100.0,
    "SENSEX": 80400.0,
    "RELIANCE": 2980.0,
    "HDFCBANK": 1640.0,
}

VOLATILITIES = {
    "NIFTY": 0.14,
    "BANKNIFTY": 0.19,
    "FINNIFTY": 0.16,
    "SENSEX": 0.13,
    "RELIANCE": 0.22,
    "HDFCBANK": 0.20,
}

SAMPLE_HEADLINES = [
    "RBI leaves benchmark repo rate unchanged in bi-monthly monetary policy review.",
    "US Fed signals gradual easing trajectory amid cooling inflation readings.",
    "FII institutional net inflows hit three-week high across index derivatives.",
    "Crude oil prices stabilize following supply corridor assurances.",
    "Banking sector credit growth accelerates by 14.2% year-on-year.",
    "Tech earnings beat analyst consensus driven by enterprise cloud renewals.",
    "Derivative rollover indicators point to bullish bias entering monthly expiry.",
    "Domestic institutional mutual funds record continuous net equity accumulation."
]


class MarketTickGenerator:
    """Generates continuous synthetic market tick events."""

    def __init__(self, seed: int | None = 42):
        if seed is not None:
            random.seed(seed)
        self.current_prices = {sym: price for sym, price in BASE_PRICES.items()}
        self.dt = 1.0 / (252 * 6.25 * 3600)  # ~1 second tick interval in trading hours

    def generate_tick(self, symbol: str) -> dict[str, Any]:
        """Generate a single realistic market microstructure tick."""
        prev_price = self.current_prices[symbol]
        vol = VOLATILITIES[symbol]

        # Geometric Brownian motion step + occasional Poisson jump
        drift = 0.05 * self.dt
        diffusion = vol * math.sqrt(self.dt) * random.gauss(0, 1)
        jump = (random.gauss(0, 0.003) if random.random() < 0.02 else 0.0)

        pct_change = drift + diffusion + jump
        new_price = round(max(prev_price * (1 + pct_change), 1.0), 2)
        self.current_prices[symbol] = new_price

        # Realistic bid-ask spread (2 to 8 basis points)
        spread_bps = random.uniform(0.0002, 0.0008)
        half_spread = round((new_price * spread_bps) / 2.0, 2)
        bid = round(new_price - half_spread, 2)
        ask = round(new_price + half_spread, 2)
        if ask <= bid:
            ask = round(bid + 0.05, 2)

        # Volume clustering (lognormal distribution)
        volume = int(math.exp(random.gauss(4.5, 0.8)))
        open_interest = int(100000 + (new_price * 10) + random.randint(-500, 500))

        now_utc = datetime.now(timezone.utc)

        return {
            "event_id": str(uuid.uuid4()),
            "timestamp": now_utc.isoformat(),
            "symbol": symbol,
            "price": new_price,
            "bid": bid,
            "ask": ask,
            "volume": volume,
            "open_interest": open_interest,
            "trade_side": random.choice(["BUY", "SELL"]),
            "venue": "NSE" if symbol != "SENSEX" else "BSE",
        }

    def generate_batch(self, count: int = 100, symbols: list[str] | None = None) -> list[dict[str, Any]]:
        """Generate a micro-batch of market ticks."""
        sym_list = symbols or list(BASE_PRICES.keys())
        return [self.generate_tick(random.choice(sym_list)) for _ in range(count)]

    def generate_news_event(self) -> dict[str, Any]:
        """Generate a simulated market news headline event."""
        headline = random.choice(SAMPLE_HEADLINES)
        symbol = random.choice(list(BASE_PRICES.keys()) + ["MACRO", "GLOBAL"])
        now_utc = datetime.now(timezone.utc)
        return {
            "news_id": str(uuid.uuid4()),
            "timestamp": now_utc.isoformat(),
            "headline": headline,
            "category": "MACRO" if symbol in ["MACRO", "GLOBAL"] else "EQUITY",
            "symbol": symbol,
            "sentiment_score": round(random.uniform(-0.8, 0.9), 3),
        }
