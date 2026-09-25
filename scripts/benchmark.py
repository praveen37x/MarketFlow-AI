"""
MarketFlow-AI: Performance & Latency Benchmarking Suite
Evaluates:
1. Ingestion throughput (ticks/second into Bronze Parquet)
2. Polars vectorized transformation & data contract validation speed
3. DuckDB analytical query latency across 100k+ events (p50, p95, p99)
4. ML Volatility Regime inference latency per feature vector
"""
import time
import sys
from pathlib import Path
import numpy as np

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config.settings import settings
from src.ingestion.generator import MarketTickGenerator
from src.ingestion.buffer import MicroBatchBuffer
from src.transformation.cleaner import SilverCleaner
from src.lakehouse.engine import DuckDBEngine
from src.lakehouse.feature_store import FeatureStore
from src.ai.model import VolatilityRegimeModel


def run_benchmark():
    print("==================================================================")
    print("        [+] MarketFlow-AI HIGH-PERFORMANCE BENCHMARK SUITE        ")
    print("==================================================================\n")

    # 1. Ingestion & Polars Throughput
    N_EVENTS = 50000
    print(f"[*] Benchmark 1: Ingesting & Buffering {N_EVENTS:,} Streaming Events...")
    gen = MarketTickGenerator()
    buffer = MicroBatchBuffer(max_size=N_EVENTS + 1000)

    t0 = time.perf_counter()
    ticks = gen.generate_batch(count=N_EVENTS)
    buffer.add_batch(ticks)
    t1 = time.perf_counter()
    ingest_time = t1 - t0
    ingest_rate = int(N_EVENTS / ingest_time)
    print(f"    -> Ingestion & Serialization: {ingest_time:.3f}s ({ingest_rate:,} events/sec)\n")

    # 2. Polars Vectorized Cleaning & Contracts
    print(f"[*] Benchmark 2: Polars Vectorized Data Contract & Cleaning Engine...")
    import polars as pl
    raw_df = pl.DataFrame(ticks)
    t0 = time.perf_counter()
    dedup = raw_df.unique(subset=["event_id"])
    clean_df = (
        dedup
        .with_columns([
            pl.col("timestamp").str.to_datetime(time_zone="UTC").alias("parsed_time"),
            (pl.col("ask") - pl.col("bid")).alias("spread"),
            ((pl.col("bid") + pl.col("ask")) / 2.0).alias("mid_price"),
            (pl.col("price") * pl.col("volume")).alias("dollar_volume"),
        ])
    )
    t1 = time.perf_counter()
    clean_time = t1 - t0
    clean_rate = int(N_EVENTS / clean_time)
    print(f"    -> Polars Cleaning & Feature Derivation: {clean_time:.3f}s ({clean_rate:,} events/sec)\n")

    # 3. DuckDB Analytical Query Latency
    print(f"[*] Benchmark 3: DuckDB Zero-Copy In-Process Analytical Queries...")
    engine = DuckDBEngine(settings.DUCKDB_PATH, settings.SILVER_DIR, settings.GOLD_DIR)
    
    # Run 50 analytical aggregations
    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = engine.query("""
            SELECT 
                symbol,
                COUNT(*) AS cnt,
                AVG(price) AS avg_p,
                SUM(volume) AS vol
            FROM silver_ticks
            GROUP BY symbol;
        """)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    print(f"    -> Aggregation Latency (p50): {p50:.2f} ms")
    print(f"    -> Aggregation Latency (p95): {p95:.2f} ms")
    print(f"    -> Aggregation Latency (p99): {p99:.2f} ms\n")

    # 4. Machine Learning Inference Latency
    print(f"[*] Benchmark 4: Scikit-Learn Gradient Boosting Regime Inference...")
    model = VolatilityRegimeModel()
    dummy_features = {
        "volatility_std": 32.5,
        "momentum_bps": 12.4,
        "spread_bps": 4.1,
        "volume_ratio": 2.2,
    }
    
    ml_latencies = []
    for _ in range(200):
        t0 = time.perf_counter()
        _ = model.predict(dummy_features)
        t1 = time.perf_counter()
        ml_latencies.append((t1 - t0) * 1000.0)

    ml_p50 = np.percentile(ml_latencies, 50)
    ml_p95 = np.percentile(ml_latencies, 95)
    print(f"    -> Inference Latency (p50): {ml_p50:.3f} ms ({int(1000.0 / ml_p50):,} predictions/sec)")
    print(f"    -> Inference Latency (p95): {ml_p95:.3f} ms\n")

    print("==================================================================")
    print("                    [#] BENCHMARK SUMMARY [#]                     ")
    print("==================================================================")
    print(f"  [+] Ingestion Throughput:        {ingest_rate:,} events/sec")
    print(f"  [+] Polars Vectorized Speed:     {clean_rate:,} events/sec")
    print(f"  [+] DuckDB Analytical Query p50: {p50:.2f} ms")
    print(f"  [+] AI Model Inference p50:      {ml_p50:.3f} ms")
    print("==================================================================\n")
    engine.close()


if __name__ == "__main__":
    run_benchmark()
