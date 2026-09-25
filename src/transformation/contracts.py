"""
MarketFlow-AI Transformation: Data Quality Contracts & Schema Enforcement
Implements rigorous data contracts, boundary validations, and schema checks
to guarantee zero corrupt records pass into the Silver & Gold analytical tiers.
"""
from dataclasses import dataclass
from typing import Any
import polars as pl


@dataclass
class ContractAuditReport:
    total_records: int
    valid_records: int
    quarantined_records: int
    null_violations: int
    price_violations: int
    spread_violations: int
    passed: bool

    @property
    def pass_rate_pct(self) -> float:
        return round((self.valid_records / max(self.total_records, 1)) * 100.0, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "quarantined_records": self.quarantined_records,
            "null_violations": self.null_violations,
            "price_violations": self.price_violations,
            "spread_violations": self.spread_violations,
            "passed": self.passed,
            "pass_rate_pct": self.pass_rate_pct,
        }


class DataContractEnforcer:
    """Enforces data contracts and schema integrity on raw market feeds."""

    REQUIRED_COLUMNS = [
        "event_id", "timestamp", "symbol", "price", "bid", "ask", "volume"
    ]

    @classmethod
    def validate_and_filter(cls, df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, ContractAuditReport]:
        """
        Validate incoming dataframe against strict market data contracts.
        Returns:
            - valid_df: Conformed, clean rows
            - quarantine_df: Rows violating data quality rules
            - audit_report: Metric summary of data health
        """
        total = len(df)
        if total == 0:
            return df, df, ContractAuditReport(0, 0, 0, 0, 0, 0, True)

        # 1. Check required schema columns
        for col in cls.REQUIRED_COLUMNS:
            if col not in df.columns:
                raise ValueError(f"Data Contract Schema Violation: Missing required column '{col}'")

        # 2. Null Checks across critical fields
        null_mask = (
            df["event_id"].is_null() |
            df["timestamp"].is_null() |
            df["symbol"].is_null() |
            df["price"].is_null() |
            df["bid"].is_null() |
            df["ask"].is_null() |
            df["volume"].is_null()
        )
        null_violations = int(null_mask.sum())

        # 3. Price & volume boundary assertions (price > 0, volume >= 0)
        price_mask = (df["price"] <= 0) | (df["volume"] < 0) | (df["bid"] <= 0) | (df["ask"] <= 0)
        price_violations = int(price_mask.sum())

        # 4. Bid-Ask spread sanity (ask must be strictly >= bid)
        spread_mask = df["ask"] < df["bid"]
        spread_violations = int(spread_mask.sum())

        # Composite quarantine filter
        violator_mask = null_mask | price_mask | spread_mask

        valid_df = df.filter(~violator_mask)
        quarantine_df = df.filter(violator_mask)

        quarantined_count = len(quarantine_df)
        valid_count = len(valid_df)

        report = ContractAuditReport(
            total_records=total,
            valid_records=valid_count,
            quarantined_records=quarantined_count,
            null_violations=null_violations,
            price_violations=price_violations,
            spread_violations=spread_violations,
            passed=(quarantined_count == 0),
        )

        return valid_df, quarantine_df, report
