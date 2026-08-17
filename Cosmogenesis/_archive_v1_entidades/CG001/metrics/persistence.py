"""Métricas observables CG001 — IPD, IH, IN (protocolo §56-57)."""
from __future__ import annotations

from typing import Any

import numpy as np

from CG001.core.entity import Entity
from CG001.core.environment import Environment


def _cluster_niches(positions: np.ndarray, eps: float, min_count: int) -> int:
    if len(positions) < min_count:
        return 0
    # Clustering greedy simple por distancia (auditabilidad > optimización)
    used = np.zeros(len(positions), dtype=bool)
    clusters = 0
    for i in range(len(positions)):
        if used[i]:
            continue
        dists = np.linalg.norm(positions - positions[i], axis=1)
        members = np.where(dists < eps)[0]
        if len(members) >= min_count:
            clusters += 1
            used[members] = True
    return clusters


def compute_metrics(alive: list[Entity], env: Environment, cfg: dict, *, fast: bool = False) -> dict[str, Any]:
    if not alive:
        return {
            "IPD": 1.0,
            "IH": 0.0,
            "IN": 0,
            "IPA": 0.0,
            "ICG0": 0.0,
            "S_mean": 0.0,
            "S_max": 0.0,
            "delta_mean": 0.0,
            "H_delta": 0.0,
        }

    s_vals = np.array([e.S for e in alive], dtype=np.float64)
    d_vals = np.array([e.delta_struct for e in alive], dtype=np.float64)
    h_vals = np.array([e.H for e in alive], dtype=np.float64)
    pos = np.array([e.pos for e in alive], dtype=np.float64)

    s_mean = float(s_vals.mean())
    s_max = float(s_vals.max())
    ipd = s_max / s_mean if s_mean > 1e-9 else 1.0
    ih = float(h_vals.sum()) + env.total_history()
    # IN = nichos ambientales Env (protocolo §55/§130), no clústeres de entidades
    in_clusters = int(env.niches.sum())
    in_entity_clusters = (
        0 if fast else _cluster_niches(
            pos,
            float(cfg.get("niche_cluster_eps", 3.0)),
            int(cfg.get("niche_min_count", 5)),
        )
    )
    h_delta = float(-np.sum(d_vals * np.log(d_vals + 1e-9)))

    return {
        "IPD": round(ipd, 4),
        "IH": round(ih, 4),
        "IN": in_clusters,
        "IN_cluster": in_entity_clusters,
        "IPA": round(float(env.stability.mean()), 6),
        "ICG0": round(float(env.history.max()), 6),
        "S_mean": round(s_mean, 6),
        "S_max": round(s_max, 6),
        "delta_mean": round(float(d_vals.mean()), 6),
        "H_delta": round(h_delta, 4),
    }