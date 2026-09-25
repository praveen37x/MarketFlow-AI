"""
MarketFlow-AI Ingestion: Stream Collector
Asynchronous stream collector that coordinates the tick generator,
buffer ingestion, and periodic flushing to the Bronze lakehouse tier.
"""
import time
from pathlib import Path
from typing import Any
from src.ingestion.generator import MarketTickGenerator
from src.ingestion.buffer import MicroBatchBuffer


class StreamCollector:
    """Manages continuous ingestion into the Lakehouse Bronze layer."""

    def __init__(self, bronze_dir: Path, batch_size: int = 500):
        self.bronze_dir = bronze_dir
        self.batch_size = batch_size
        self.generator = MarketTickGenerator()
        self.buffer = MicroBatchBuffer(max_size=50000)
        self.is_running = False

    def ingest_batch(self, count: int = 500, symbols: list[str] | None = None) -> Path | None:
        """Synchronously generate a batch of ticks and flush directly to Bronze."""
        ticks = self.generator.generate_batch(count=count, symbols=symbols)
        self.buffer.add_batch(ticks)
        return self.buffer.flush_to_bronze(self.bronze_dir)

    def get_metrics(self) -> dict[str, Any]:
        """Return real-time ingestion telemetry."""
        return {
            "buffer_depth": self.buffer.size(),
            "total_received": self.buffer.total_received,
            "total_flushed": self.buffer.total_flushed,
            "bronze_directory": str(self.bronze_dir),
        }
