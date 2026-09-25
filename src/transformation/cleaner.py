"""
MarketFlow-AI Transformation: Silver Layer Vectorized Engine
Uses Polars for blazing-fast vectorized transformations:
- Deduplication across event IDs
- Timestamp parsing & UTC normalization
- Microstructure feature engineering (spread, mid_price, dollar_volume)
- Writing partitioned Silver Parquet tables
"""
from pathlib import Path
from typing import Any
import polars as pl
from src.transformation.contracts import DataContractEnforcer, ContractAuditReport


class SilverCleaner:
    """Transforms raw Bronze batches into validated, conformed Silver datasets."""

    def __init__(self, bronze_dir: Path, silver_dir: Path):
        self.bronze_dir = bronze_dir
        self.silver_dir = silver_dir
        self.silver_dir.mkdir(parents=True, exist_ok=True)

    def process_bronze_files(self) -> tuple[Path | None, ContractAuditReport | None]:
        """Read all Bronze Parquet files, clean and persist to Silver layer."""
        bronze_files = list(self.bronze_dir.glob("*.parquet"))
        if not bronze_files:
            return None, None

        # Load all bronze files using Polars lazy execution
        lazy_df = pl.scan_parquet(str(self.bronze_dir / "*.parquet"))
        raw_df = lazy_df.collect()

        # 1. Deduplicate by unique event_id
        dedup_df = raw_df.unique(subset=["event_id"])

        # 2. Run data contracts validation
        valid_df, quarantine_df, audit_report = DataContractEnforcer.validate_and_filter(dedup_df)

        if len(valid_df) == 0:
            return None, audit_report

        # 3. Vectorized feature engineering with explicit UTC timezone
        silver_df = (
            valid_df
            .with_columns([
                pl.col("timestamp").str.to_datetime(time_zone="UTC").alias("parsed_time"),
                (pl.col("ask") - pl.col("bid")).round(2).alias("spread"),
                ((pl.col("bid") + pl.col("ask")) / 2.0).round(2).alias("mid_price"),
                (pl.col("price") * pl.col("volume")).alias("dollar_volume"),
            ])
            .with_columns([
                # Relative spread in basis points
                ((pl.col("spread") / pl.col("mid_price")) * 10000).round(1).alias("spread_bps"),
                pl.col("parsed_time").dt.date().alias("trade_date"),
            ])
            .sort("parsed_time")
        )

        # 4. Write partitioned Silver Parquet
        out_file = self.silver_dir / "silver_ticks_conformed.parquet"
        silver_df.write_parquet(out_file, compression="zstd")

        # Save quarantined records if any exist
        if len(quarantine_df) > 0:
            quarantine_path = self.silver_dir / "quarantine_records.parquet"
            quarantine_df.write_parquet(quarantine_path, compression="zstd")

        return out_file, audit_report
