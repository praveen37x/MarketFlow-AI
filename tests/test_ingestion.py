"""
Tests for Ingestion Layer (Generator & MicroBatchBuffer)
"""
from src.ingestion.generator import MarketTickGenerator
from src.ingestion.buffer import MicroBatchBuffer


def test_generator_single_tick():
    gen = MarketTickGenerator(seed=123)
    tick = gen.generate_tick("NIFTY")
    assert tick["symbol"] == "NIFTY"
    assert tick["price"] > 0
    assert tick["bid"] <= tick["ask"]
    assert tick["volume"] > 0
    assert tick["trade_side"] in ["BUY", "SELL"]
    assert "timestamp" in tick
    assert "event_id" in tick


def test_generator_batch():
    gen = MarketTickGenerator(seed=123)
    batch = gen.generate_batch(count=50, symbols=["BANKNIFTY", "RELIANCE"])
    assert len(batch) == 50
    assert all(t["symbol"] in ["BANKNIFTY", "RELIANCE"] for t in batch)


def test_micro_batch_buffer():
    buffer = MicroBatchBuffer(max_size=100)
    gen = MarketTickGenerator(seed=123)
    ticks = gen.generate_batch(count=25)

    buffer.add_batch(ticks)
    assert buffer.size() == 25
    assert buffer.total_received == 25

    drained = buffer.drain(limit=10)
    assert len(drained) == 10
    assert buffer.size() == 15
    assert buffer.total_flushed == 10
