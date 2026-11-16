"""
Synthetic corporate transaction data with injected, labelled anomalies.

    transactions.parquet  txn_id, client_id, ts, amount, counterparty_id, channel, is_anomaly, anomaly_type

Each corporate client has its own normal behaviour (typical amount, business
hours, a stable set of counterparties). About 1% of transactions are injected
anomalies of five kinds:

    LARGE_AMOUNT       far above the client's usual size
    ODD_HOUR           at night / weekend, for a client that never does that
    NEW_CPTY_LARGE     a large payment to a never-seen counterparty
    BURST              a rapid sequence of payments (velocity)
    STRUCTURING        repeated amounts just below a round reporting threshold

Labels exist only because the data is synthetic. In production, labelled fraud
is scarce, which is why unsupervised detectors are used in the first place.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)  # pandas/numpy timedelta unit notice


def generate(n_clients: int = 250, days: int = 90, seed: int = 17):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-01-05")
    rows = []
    tid = 0
    for c in range(n_clients):
        cid = f"CORP{c:04d}"
        typical = rng.lognormal(14.0, 1.0)
        per_day = rng.gamma(2.0, 2.0)
        cptys = [f"{cid}-P{k}" for k in range(rng.integers(5, 40))]
        weekend_ok = rng.random() < 0.15
        for d in range(days):
            day = start + pd.Timedelta(days=d)
            if day.dayofweek >= 5 and not weekend_ok:
                continue
            for _ in range(rng.poisson(per_day)):
                hour = float(np.clip(rng.normal(12.5, 2.3), 7, 19))
                rows.append((f"T{tid:08d}", cid, day + pd.Timedelta(hours=hour), typical * rng.lognormal(0, 0.55),
                             rng.choice(cptys), rng.choice(["BULK", "SINGLE", "API"], p=[.5, .35, .15]), 0, "NORMAL"))
                tid += 1
        # ---- injected anomalies
        for _ in range(rng.poisson(0.9)):
            kind = rng.choice(["LARGE_AMOUNT", "ODD_HOUR", "NEW_CPTY_LARGE", "BURST", "STRUCTURING"])
            day = start + pd.Timedelta(days=int(rng.integers(5, days)))
            if kind == "LARGE_AMOUNT":
                ev = [(day + pd.Timedelta(hours=rng.uniform(9, 17)), typical * rng.uniform(8, 25), rng.choice(cptys))]
            elif kind == "ODD_HOUR":
                ev = [(day + pd.Timedelta(hours=rng.choice([1.5, 2.5, 3.0, 23.5])), typical * rng.uniform(1.5, 4), rng.choice(cptys))]
            elif kind == "NEW_CPTY_LARGE":
                ev = [(day + pd.Timedelta(hours=rng.uniform(9, 17)), typical * rng.uniform(4, 10), f"NEW-{tid}")]
            elif kind == "BURST":
                t0 = day + pd.Timedelta(hours=rng.uniform(9, 16))
                ev = [(t0 + pd.Timedelta(minutes=3 * k), typical * rng.uniform(0.8, 2), f"NEW-{tid}-{k}") for k in range(int(rng.integers(6, 12)))]
            else:
                thr = 5_000_000 if typical < 3e6 else 50_000_000
                ev = [(day + pd.Timedelta(hours=rng.uniform(9, 17), days=k), thr * rng.uniform(0.95, 0.995), f"NEW-{tid}-{k}") for k in range(int(rng.integers(3, 6)))]
            for ts, amt, cp in ev:
                rows.append((f"T{tid:08d}", cid, ts, amt, cp, "SINGLE", 1, kind))
                tid += 1
    df = pd.DataFrame(rows, columns=["txn_id", "client_id", "ts", "amount", "counterparty_id", "channel", "is_anomaly", "anomaly_type"])
    df["amount"] = df.amount.round(2)
    return df.sort_values(["client_id", "ts"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=250)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "raw")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    df = generate(a.clients, seed=a.seed)
    df.to_parquet(a.out / "transactions.parquet", index=False)
    print(f"transactions {len(df):,} | anomalies {df.is_anomaly.sum():,} ({df.is_anomaly.mean():.2%})")
    print(df[df.is_anomaly == 1].anomaly_type.value_counts().to_string())


if __name__ == "__main__":
    main()
