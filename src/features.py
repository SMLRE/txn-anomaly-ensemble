"""Build transaction features in DuckDB."""
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["amount_robust_z", "hour_z", "is_weekend", "new_counterparty", "txns_prev_60min", "near_threshold", "log_amount"]


def build_features() -> pd.DataFrame:
    con = duckdb.connect()
    con.execute(f"CREATE VIEW transactions AS SELECT * FROM read_parquet('{ROOT / 'data/raw/transactions.parquet'}')")
    df = con.execute((ROOT / "sql" / "01_txn_features.sql").read_text()).df()
    df[FEATURES] = df[FEATURES].fillna(0.0)
    return df
