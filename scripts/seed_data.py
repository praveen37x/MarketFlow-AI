"""
MarketFlow-AI: Historical Data Seeder
Populates the Lakehouse with an initial realistic dataset across all asset symbols.
"""
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config.settings import settings
from src.ingestion.collectors import StreamCollector
from src.transformation.cleaner import SilverCleaner
from src.lakehouse.engine import DuckDBEngine
from src.lakehouse.marts import GoldMartsBuilder


def seed(count_per_symbol: int = 1000):
    print("==========================================================")
    print("  MarketFlow-AI: Lakehouse Historical Data Seeder")
    print("==========================================================")

    settings.BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    settings.SILVER_DIR.mkdir(parents=True, exist_ok=True)
    settings.GOLD_DIR.mkdir(parents=True, exist_ok=True)
    settings.CATALOG_DIR.mkdir(parents=True, exist_ok=True)

    collector = StreamCollector(settings.BRONZE_DIR)
    cleaner = SilverCleaner(settings.BRONZE_DIR, settings.SILVER_DIR)
    engine = DuckDBEngine(settings.DUCKDB_PATH, settings.SILVER_DIR, settings.GOLD_DIR)
    marts = GoldMartsBuilder(engine, settings.GOLD_DIR)

    total_ticks = 0
    for sym in settings.DEFAULT_SYMBOLS:
        print(f"[*] Ingesting {count_per_symbol} micro-batch ticks for {sym}...")
        collector.ingest_batch(count=count_per_symbol, symbols=[sym])
        total_ticks += count_per_symbol

    print(f"\n[+] Total Bronze ticks generated: {total_ticks:,}")
    print("[*] Running Polars Silver transformation & Data Contracts...")
    silver_path, audit = cleaner.process_bronze_files()
    print(f"[+] Silver Parquet written to: {silver_path}")
    if audit:
        print(f"    - Valid: {audit.valid_records:,} ({audit.to_dict()['pass_rate_pct']}%)")
        print(f"    - Quarantined: {audit.quarantined_records:,}")

    print("[*] Rebuilding DuckDB Gold Marts (1m OHLCV & Order Flow)...")
    engine.refresh_views()
    marts_res = marts.build_all_marts()
    for name, path in marts_res.items():
        print(f"[+] Gold Mart '{name}' updated: {path}")

    # Summary count
    cnt = engine.query("SELECT COUNT(*) AS total FROM silver_ticks;")[0]["total"]
    print("\n==========================================================")
    print(f"  SUCCESS: Lakehouse seeded with {cnt:,} conformed records!")
    print("==========================================================")
    engine.close()


if __name__ == "__main__":
    seed()
