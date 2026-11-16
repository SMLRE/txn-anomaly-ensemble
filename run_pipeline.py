"""
End-to-end anomaly detection pipeline.

    1. Generate synthetic transactions with labelled anomalies (if missing)
    2. Client-relative features in SQL (DuckDB)
    3. Tune: DBSCAN eps via the k-distance elbow; Isolation Forest contamination sweep
    4. Run the three detectors; compare OR / 2-of-3 / 3-of-3 ensembles
    5. Validation: precision, recall, F1, AUC, recall by anomaly type
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.detectors import contamination_sweep, dbscan, isolation_forest, k_distance_elbow, metrics, scale, statistical, vote
from src.features import FEATURES, build_features

ROOT = Path(__file__).parent
REPORTS, FIG = ROOT / "reports", ROOT / "reports" / "figures"
MIN_SAMPLES, CONTAMINATION = 10, 0.01
ACCENT = "#E8A54B"
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    if not (ROOT / "data/raw/transactions.parquet").exists():
        subprocess.run([sys.executable, str(ROOT / "data/generate_synthetic.py")], check=True)
    d = build_features()
    X, y = scale(d[FEATURES]), d.is_anomaly.to_numpy()
    print(f"{len(d):,} transactions | {y.sum()} labelled anomalies ({y.mean():.2%})")

    # ---- tuning
    eps, kd = k_distance_elbow(X, MIN_SAMPLES)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(kd, color="#5B8DEF"); ax.axhline(eps, color=ACCENT, ls="--", label=f"elbow eps = {eps:.2f}")
    ax.set_xlabel("points sorted by distance"); ax.set_ylabel(f"distance to {MIN_SAMPLES}th neighbour"); ax.set_yscale("log")
    ax.set_title("DBSCAN eps from the k-distance elbow"); ax.legend(); fig.tight_layout(); fig.savefig(FIG / "k_distance_elbow.png"); plt.close(fig)

    sweep = contamination_sweep(X, y)
    sweep.to_csv(REPORTS / "iforest_contamination_sweep.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for col, c in [("precision", "#5B8DEF"), ("recall", "#3C8D5A"), ("f1", ACCENT)]:
        ax.plot(sweep.contamination, sweep[col], "o-", color=c, label=col)
    ax.set_xscale("log"); ax.set_xlabel("contamination"); ax.set_title("Isolation Forest contamination sweep"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "contamination_sweep.png"); plt.close(fig)
    print(sweep.to_string(index=False))

    # ---- detectors + ensembles
    f_if, s_if = isolation_forest(X, CONTAMINATION)
    f_db = dbscan(X, eps, MIN_SAMPLES)
    f_st, s_st = statistical(d)
    flags = {"isolation_forest": f_if, "dbscan": f_db, "statistical": f_st}
    votes = np.sum(list(flags.values()), axis=0)
    res = {k: metrics(y, v) for k, v in flags.items()}
    res["ensemble_OR (original)"] = metrics(y, vote(flags, 1), votes)
    res["ensemble_2of3 (fix)"] = metrics(y, vote(flags, 2), votes)
    res["ensemble_3of3"] = metrics(y, vote(flags, 3), votes)
    table = pd.DataFrame(res).T
    table.to_csv(REPORTS / "detector_comparison.csv")
    print(table)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ens = table.loc[["ensemble_OR (original)", "ensemble_2of3 (fix)", "ensemble_3of3"]]
    ax.bar(["OR (any 1)", "2 of 3", "all 3"], ens.alerts, color=["#D9534F", ACCENT, "#8A8A8A"])
    for i, (a, p, r) in enumerate(zip(ens.alerts, ens.precision, ens.recall)):
        ax.text(i, a * 1.02, f"precision {p:.0%}\nrecall {r:.0%}", ha="center", fontsize=9)
    ax.set_ylabel("alerts raised"); ax.set_ylim(0, ens.alerts.max() * 1.3)
    ax.set_title("Same three detectors, different voting rules"); fig.tight_layout(); fig.savefig(FIG / "ensemble_alerts.png"); plt.close(fig)

    d["flag_2of3"] = vote(flags, 2)
    by_type = d[d.is_anomaly == 1].groupby("anomaly_type").flag_2of3.agg(["size", "mean"]).rename(columns={"size": "n", "mean": "recall_2of3"})
    by_type.round(3).to_csv(REPORTS / "recall_by_type.csv")
    print(by_type)

    alerts = d[d.flag_2of3 == 1].assign(votes=votes[d.flag_2of3 == 1])[["txn_id", "client_id", "ts", "amount", "votes"] + FEATURES[:-1]]
    alerts.sort_values(["votes", "amount"], ascending=False).head(300).to_csv(REPORTS / "alert_queue_sample.csv", index=False)

    orr, fix = res["ensemble_OR (original)"], res["ensemble_2of3 (fix)"]
    metrics_out = {
        "transactions": int(len(d)), "anomalies": int(y.sum()),
        "dbscan_eps_elbow": round(eps, 4), "iforest_contamination": CONTAMINATION,
        "results": res,
        "alert_reduction_or_to_2of3": round(1 - fix["alerts"] / orr["alerts"], 4),
        "precision_gain_x": round(fix["precision"] / orr["precision"], 2),
        "recall_by_type_2of3": by_type.recall_2of3.round(3).to_dict(),
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics_out, indent=2))
    return metrics_out


if __name__ == "__main__":
    main()
