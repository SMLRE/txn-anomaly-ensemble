"""
Three independent detectors and the ensemble logic.

    Isolation Forest   global, multivariate outliers
    DBSCAN             points in no dense neighbourhood (noise label -1)
    Statistical rules  robust z-score on amount, unusual hour, velocity

The original notebook combined them with OR (flag if ANY detector fires).
Each detector's false positives then add up, and analysts drown in alerts.
The fix is agreement voting: flag only when at least k of 3 detectors agree.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def scale(X: pd.DataFrame) -> np.ndarray:
    return StandardScaler().fit_transform(X)


# ----------------------------------------------------------------- detectors
def isolation_forest(Xs: np.ndarray, contamination: float, seed: int = 0):
    m = IsolationForest(n_estimators=300, contamination=contamination, random_state=seed, n_jobs=1).fit(Xs)
    score = -m.score_samples(Xs)              # higher = more anomalous
    return (m.predict(Xs) == -1).astype(int), score


def dbscan(Xs: np.ndarray, eps: float, min_samples: int):
    labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=1).fit_predict(Xs)
    return (labels == -1).astype(int)


def statistical(df: pd.DataFrame, z_amt: float = 3.5, z_hour: float = 3.0, velocity: int = 3):
    flag = (df.amount_robust_z > z_amt) | (df.hour_z > z_hour) | (df.txns_prev_60min >= velocity) | (df.near_threshold == 1)
    score = np.maximum.reduce([df.amount_robust_z / z_amt, df.hour_z / z_hour, df.txns_prev_60min / velocity, df.near_threshold * 1.0])
    return flag.astype(int).to_numpy(), np.asarray(score)


# ----------------------------------------------------------------- tuning
def k_distance_elbow(Xs: np.ndarray, k: int, sample: int = 20000, seed: int = 0) -> tuple[float, np.ndarray]:
    """eps for DBSCAN at the 'knee' of the sorted k-distance curve (max distance to the chord)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xs), min(sample, len(Xs)), replace=False)
    d, _ = NearestNeighbors(n_neighbors=k + 1).fit(Xs).kneighbors(Xs[idx])
    kd = np.sort(d[:, -1])
    x = np.linspace(0, 1, len(kd)); y = (kd - kd.min()) / (kd.max() - kd.min())
    knee = int(np.argmax(np.abs(y - x)))           # farthest point below the chord from first to last
    return float(kd[knee]), kd


def contamination_sweep(Xs: np.ndarray, y: np.ndarray, grid=(0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05)) -> pd.DataFrame:
    rows = []
    for c in grid:
        f, s = isolation_forest(Xs, c)
        rows.append({"contamination": c, **metrics(y, f, s)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- ensemble + evaluation
def vote(flags: dict[str, np.ndarray], k: int) -> np.ndarray:
    return (np.sum(list(flags.values()), axis=0) >= k).astype(int)


def metrics(y, flag, score=None) -> dict:
    out = {"alerts": int(flag.sum()), "precision": precision_score(y, flag, zero_division=0),
           "recall": recall_score(y, flag), "f1": f1_score(y, flag)}
    if score is not None:
        out["auc"] = roc_auc_score(y, score)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}
