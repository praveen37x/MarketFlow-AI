"""
MarketFlow-AI Ingestion: Ring Buffer & Micro-Batch Manager
Thread-safe micro-batch buffer that holds in-flight streaming ticks
and periodically flushes raw events into Bronze Parquet partitions.
"""
import threading
from collections import deque
from typing import Any
import polars as pl
from pathlib import Path


class MicroBatchBuffer:
    """Thread-safe ring buffer for streaming market events."""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._buffer: deque[dict[str, Any]] = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self.total_received: int = 0
        self.total_flushed: int = 0

    def add(self, event: dict[str, Any]) -> None:
        """Add a single event to the ring buffer."""
        with self._lock:
            self._buffer.append(event)
            self.total_received += 1

    def add_batch(self, events: list[dict[str, Any]]) -> None:
        """Add multiple events to the ring buffer."""
        with self._lock:
            self._buffer.extend(events)
            self.total_received += len(events)

    def size(self) -> int:
        """Current number of buffered events."""
        with self._lock:
            return len(self._buffer)

    def drain(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Drain events from buffer for Bronze flush."""
        with self._lock:
            if not self._buffer:
                return []
            count = len(self._buffer) if limit is None else min(limit, len(self._buffer))
            items = [self._buffer.popleft() for _ in range(count)]
            self.total_flushed += len(items)
            return items

    def flush_to_bronze(self, bronze_dir: Path) -> Path | None:
        """Drain buffer and write immutable raw Parquet batch into Bronze layer."""
        events = self.drain()
        if not events:
            return None

        bronze_dir.mkdir(parents=True, exist_ok=True)
        df = pl.DataFrame(events)

        # Generate unique partition filename
        first_time = events[0]["timestamp"].replace(":", "-").replace(".", "-")
        filename = f"bronze_batch_{first_time}_{len(events)}.parquet"
        out_path = bronze_dir / filename

        df.write_parquet(out_path, compression="snappy")
        return out_path
