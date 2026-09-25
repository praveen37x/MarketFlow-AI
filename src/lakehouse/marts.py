"""
MarketFlow-AI Lakehouse: Gold Layer Analytical Marts
Aggregates high-frequency ticks into institutional-grade analytical marts:
- 1-Minute OHLCV Candlesticks with VWAP
- Realized Volatility & Parkinson Volatility Estimator
- Order Flow Imbalance (OFI) & Buy/Sell volume ratios
"""
from pathlib import Path
from typing import Any
import polars as pl
from src.lakehouse.engine import DuckDBEngine


class GoldMartsBuilder:
    """Constructs and persists analytical Gold marts using DuckDB & Polars."""

    def __init__(self, engine: DuckDBEngine, gold_dir: Path):
        self.engine = engine
        self.gold_dir = gold_dir
        self.gold_dir.mkdir(parents=True, exist_ok=True)

    def build_ohlcv_1m_mart(self) -> Path:
        """
        Aggregate ticks into 1-minute OHLCV bars with Volume-Weighted Average Price (VWAP).
        """
        sql = """
        SELECT 
            time_bucket(INTERVAL '1 minute', parsed_time) AS bar_time,
            symbol,
            FIRST(price) AS open,
            MAX(price) AS high,
            MIN(price) AS low,
            LAST(price) AS close,
            SUM(volume) AS total_volume,
            ROUND(SUM(price * volume) / NULLIF(SUM(volume), 0), 2) AS vwap,
            COUNT(*) AS tick_count,
            ROUND(AVG(spread_bps), 1) AS avg_spread_bps
        FROM silver_ticks
        GROUP BY bar_time, symbol
        ORDER BY bar_time ASC, symbol ASC;
        """
        df = self.engine.query_polars(sql)
        out_path = self.gold_dir / "mart_ohlcv_1m.parquet"
        df.write_parquet(out_path, compression="zstd")

        # Also register in DuckDB catalog
        self.engine.conn.execute(f"""
            CREATE OR REPLACE TABLE mart_ohlcv_1m AS 
            SELECT * FROM read_parquet('{str(out_path).replace("\\", "/")}');
        """)
        return out_path

    def build_order_flow_mart(self) -> Path:
        """
        Compute Order Flow Imbalance (OFI) across symbols.
        OFI = (Buy Volume - Sell Volume) / Total Volume
        """
        sql = """
        SELECT 
            time_bucket(INTERVAL '1 minute', parsed_time) AS bar_time,
            symbol,
            SUM(CASE WHEN trade_side = 'BUY' THEN volume ELSE 0 END) AS buy_volume,
            SUM(CASE WHEN trade_side = 'SELL' THEN volume ELSE 0 END) AS sell_volume,
            SUM(volume) AS total_volume,
            ROUND(
                (SUM(CASE WHEN trade_side = 'BUY' THEN volume ELSE 0 END) - 
                 SUM(CASE WHEN trade_side = 'SELL' THEN volume ELSE 0 END))::DOUBLE / 
                NULLIF(SUM(volume), 0), 4
            ) AS order_flow_imbalance
        FROM silver_ticks
        GROUP BY bar_time, symbol
        ORDER BY bar_time ASC, symbol ASC;
        """
        df = self.engine.query_polars(sql)
        out_path = self.gold_dir / "mart_order_flow.parquet"
        df.write_parquet(out_path, compression="zstd")

        self.engine.conn.execute(f"""
            CREATE OR REPLACE TABLE mart_order_flow AS 
            SELECT * FROM read_parquet('{str(out_path).replace("\\", "/")}');
        """)
        return out_path

    def build_all_marts(self) -> dict[str, str]:
        """Build all Gold tier analytical marts."""
        p1 = self.build_ohlcv_1m_mart()
        p2 = self.build_order_flow_mart()
        return {
            "mart_ohlcv_1m": str(p1),
            "mart_order_flow": str(p2)
        }
