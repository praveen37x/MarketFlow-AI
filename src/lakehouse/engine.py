"""
MarketFlow-AI Lakehouse: DuckDB Engine Manager
Embedded in-process OLAP engine that connects to the catalog,
registers views over Parquet lake tiers, and executes sub-millisecond queries.
"""
from pathlib import Path
import threading
from typing import Any
import duckdb
import polars as pl


class DuckDBEngine:
    """Manages the DuckDB embedded OLAP engine and view registration."""

    def __init__(self, db_path: Path, silver_dir: Path, gold_dir: Path):
        self.db_path = db_path
        self.silver_dir = silver_dir
        self.gold_dir = gold_dir
        self._lock = threading.Lock()
        self.conn: duckdb.DuckDBPyConnection | None = None
        self._initialize_database()

    def _initialize_database(self) -> None:
        """Create catalog directory and configure DuckDB connection."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path))
        self.conn.execute("PRAGMA threads=4;")
        self.conn.execute("PRAGMA memory_limit='4GB';")
        self.refresh_views()

    def refresh_views(self) -> None:
        """Register views over the Silver and Gold Parquet layers."""
        with self._lock:
            silver_pattern = str(self.silver_dir / "*.parquet").replace("\\", "/")
            silver_files = list(self.silver_dir.glob("*.parquet"))

            if silver_files:
                self.conn.execute(f"""
                    CREATE OR REPLACE VIEW silver_ticks AS 
                    SELECT * FROM read_parquet('{silver_pattern}');
                """)
            else:
                self.conn.execute("""
                    CREATE OR REPLACE VIEW silver_ticks AS 
                    SELECT 
                        '' AS event_id,
                        TIMESTAMP '2026-01-01 00:00:00' AS parsed_time,
                        '' AS symbol,
                        0.0 AS price,
                        0.0 AS bid,
                        0.0 AS ask,
                        0 AS volume,
                        0 AS open_interest,
                        '' AS trade_side,
                        '' AS venue,
                        0.0 AS spread,
                        0.0 AS mid_price,
                        0.0 AS dollar_volume,
                        0.0 AS spread_bps,
                        DATE '2026-01-01' AS trade_date
                    WHERE 1 = 0;
                """)

    def query(self, sql: str) -> list[dict[str, Any]]:
        """
        Execute arbitrary SQL and return results as Python dictionaries
        using Polars zero-copy representation (pure pandas-free).
        """
        with self._lock:
            rel = self.conn.sql(sql)
            if rel is None:
                return []
            # Use Polars zero-copy conversion to avoid pandas dependency
            pldf = rel.pl()
            return pldf.to_dicts()

    def query_polars(self, sql: str) -> pl.DataFrame:
        """Execute SQL query and return zero-copy Polars DataFrame."""
        with self._lock:
            return self.conn.sql(sql).pl()

    def close(self) -> None:
        """Close connection."""
        with self._lock:
            if self.conn:
                self.conn.close()
                self.conn = None
