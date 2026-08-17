"""
CG002 — Acoplamiento originario y emergencia de orden
Protocolo: PROTOCOLO_CG002_ACOPLAMIENTO_ORIGINARIO.md v0.1c

UNA regla primitiva (§5.1 / paso dirigido con θ_CP). Sin grilla, radio, π, c, U_Cos.
Outcomes se REGISTRAN, no se codifican.
Lectura honesta: B (dirección) N/A si θ_CP=0; frustración + embedding siempre.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from cg002_observables_v02 import (
    dimension_efectiva,
    g_dirigido,
    imbalance_orientado,
    laplaciano_signado_lambda_min,
    matriz_compat_circulo,
    rango_estructural,
    tracking_score,
    vector_predictor,
)

# --- Parámetros fijos (no ajustar post-hoc salvo bump de protocolo) ---
K_FIRMA = 8
KAPPA_S = 1e-6
S0 = 1.0
ETA = 0.05
MU = 0.01
EPS_TAU = 1e-4
TAU_MIN = 3
W_MIN = 0.1 * S0
RHO_CRIT = 0.6
DS_CRIT = 2.0
K_MAX = 10_000
TAU_CAP = 500
K_MAX_MIN = 1500  # μ no extingue antes ~1400 pasos
EPS_DEG = 1e-3
EPS_TRACK = 0.1
S_MAX_DEFAULT = None  # v0.1c: colapso duro mata coop (ver mini_barrido); homeostasis en v0.1d


@dataclass(frozen=True)
class CG002Config:
    N: int = 8
    alpha: float = 1.0
    seed: int = 1
    k_max: int = K_MAX
    tau_cap: int = TAU_CAP
    k_firma: int = K_FIRMA
    theta_cp: float = 0.0
    s_max: float | None = S_MAX_DEFAULT
    log_snapshots: bool = False
    # --- Brazo de control añadido 13-ago-2026 (autorizado por el director) ---
    # barajar_por_paso: α sigue en 1 (HAY acoplamiento) pero la asignación
    # firma→nodo se re-permuta en cada micro-paso, de modo que la relación
    # concreta NO persiste. Destruye la ESTRUCTURA sin apagar el acoplamiento.
    # Con False el motor es bit a bit idéntico a v0.1c (verificado).
    # Barajar UNA sola vez al inicio sería un simple renombre de nodos
    # (isomorfo, verificado numéricamente) y NO sirve como control.
    barajar_por_paso: bool = False
    seed_baraja: int = 90000


def _vivo(s: float) -> bool:
    return s > KAPPA_S


def _arista_activa(w: float) -> bool:
    return w > W_MIN


def _construir_adyacencia(w_link: np.ndarray, s: np.ndarray) -> np.ndarray:
    n = len(s)
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        if not _vivo(s[i]):
            continue
        for j in range(i + 1, n):
            if _vivo(s[j]) and _arista_activa(w_link[i, j]):
                adj[i, j] = adj[j, i] = True
    return adj


def _diametro_grafo(adj: np.ndarray) -> int:
    n = adj.shape[0]
    vivos = [i for i in range(n) if adj[i].any() or any(adj[j, i] for j in range(n))]
    if not vivos:
        return 0
    max_d = 0
    for start in vivos:
        dist = {start: 0}
        q = [start]
        for u in q:
            for v in range(n):
                if adj[u, v] and v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        if dist:
            max_d = max(max_d, max(dist.values()))
    return max_d


def _componentes(adj: np.ndarray, s: np.ndarray) -> int:
    n = len(s)
    visitado = [False] * n
    comps = 0
    for i in range(n):
        if not _vivo(s[i]) or visitado[i]:
            continue
        comps += 1
        stack = [i]
        visitado[i] = True
        while stack:
            u = stack.pop()
            for v in range(n):
                if adj[u, v] and _vivo(s[v]) and not visitado[v]:
                    visitado[v] = True
                    stack.append(v)
    return comps


def _balance_cooperativo(f_signed: np.ndarray) -> float:
    """ρ: fracción cooperativa global. NO es dirección (v0.1 simétrico)."""
    pos = float(np.sum(np.maximum(f_signed, 0.0)))
    tot = float(np.sum(np.abs(f_signed)))
    return pos / (tot + 1e-12)


def _frustracion_subgrafo(f_signed: np.ndarray, s: np.ndarray) -> float:
    vivos = np.where(s > KAPPA_S)[0]
    if len(vivos) < 2:
        return 0.0
    sub = f_signed[np.ix_(vivos, vivos)].astype(np.float64)
    scale = float(np.max(np.abs(sub)))
    if scale < 1e-15:
        return 0.0
    return laplaciano_signado_lambda_min(sub / scale)


def _metricas_geometria(omega: np.ndarray, k: int) -> dict[str, float]:
    c = matriz_compat_circulo(omega, k)
    return {
        "rango_estructural": float(rango_estructural(c)),
        "dimension_efectiva": float(dimension_efectiva(c)),
    }


def _metricas_direccion(w_orient: np.ndarray, theta_cp: float) -> dict[str, Any]:
    if abs(theta_cp) < 1e-15:
        return {
            "criterio_B": "N/A",
            "t_hat_mod": 0.0,
            "asimetria": 0.0,
            "flujo_neto": [],
        }
    m = imbalance_orientado(w_orient)
    return {
        "criterio_B": "evaluable",
        "t_hat_mod": m["T_hat_mod"],
        "asimetria": m["asimetria"],
        "flujo_neto": m["flujo_neto"].tolist(),
    }


def _calcular_frontera(w_link: np.ndarray, f_abs_accum: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, float]:
    n = len(s)
    h = np.zeros(n, dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if not _vivo(s[i]):
            continue
        for j in range(n):
            if i != j and _arista_activa(w_link[i, j]):
                h[i] += 1.0
        if h[i] >= 1:
            b[i] = f_abs_accum[i] / (h[i] + 1e-12)
    mean_b = float(b[b > 0].mean()) if np.any(b > 0) else 1.0
    ds = b / (mean_b + 1e-12)
    ds_max = float(ds.max()) if np.any(ds > 0) else 0.0
    return ds, ds_max


def _snapshot(
    tau: int,
    k: int,
    s: np.ndarray,
    omega: np.ndarray,
    w_link: np.ndarray,
    w_orient: np.ndarray,
    f_signed: np.ndarray,
    delta_struct: float,
    phi: float,
    theta_cp: float,
) -> dict[str, Any]:
    adj = _construir_adyacencia(w_link, s)
    nodos = []
    for i in range(len(s)):
        if not _vivo(s[i]):
            continue
        nodos.append({
            "id": i,
            "S": float(s[i]),
            "omega": int(omega[i]),
            "grado": int(adj[i].sum()),
        })
    aristas = []
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            if adj[i, j]:
                fs = float(f_signed[i, j])
                aristas.append({
                    "source": i,
                    "target": j,
                    "peso": float(w_link[i, j]),
                    "sign": fs if abs(fs) > 1e-15 else 1.0,
                })
    flujo: list[float] = []
    t_hat = 0.0
    asimetria = 0.0
    if abs(theta_cp) > 1e-15:
        m = imbalance_orientado(w_orient)
        flujo = m["flujo_neto"].tolist()
        t_hat = m["T_hat_mod"]
        asimetria = m["asimetria"]
    return {
        "tau": tau,
        "k": k,
        "delta_struct": float(delta_struct),
        "phi_frustracion": float(phi),
        "theta_cp": float(theta_cp),
        "nodos": nodos,
        "aristas": aristas,
        "diametro": _diametro_grafo(adj),
        "componentes": _componentes(adj, s),
        "flujo_neto": flujo,
        "t_hat_mod": t_hat,
        "asimetria": asimetria,
    }


def _paso_acoplamiento(
    s: np.ndarray,
    omega: np.ndarray,
    alpha: float,
    k: int,
    theta_cp: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Un micro-paso. Devuelve delta_s, delta_f_signed, delta_w_orient, sum_abs_f, sum_pos_f."""
    n = len(s)
    delta_s = np.zeros(n, dtype=np.float64)
    df = np.zeros((n, n), dtype=np.float64)
    dw = np.zeros((n, n), dtype=np.float64)
    sum_abs_f = 0.0
    sum_pos_f = 0.0

    if alpha <= 0 or n < 2:
        return delta_s, df, dw, sum_abs_f, sum_pos_f

    for i in range(n):
        if not _vivo(s[i]):
            continue
        for j in range(i + 1, n):
            if not _vivo(s[j]):
                continue
            gi, gj = g_dirigido(int(omega[i]), int(omega[j]), k, theta_cp)
            mag = math.sqrt(s[i] * s[j])
            f_i = alpha * gi * mag
            f_j = alpha * gj * mag
            sum_abs_f += abs(f_i) + abs(f_j)
            sum_pos_f += max(0.0, f_i) + max(0.0, f_j)
            delta_s[i] += ETA * f_i
            delta_s[j] += ETA * f_j
            df[i, j] += f_i
            df[j, i] += f_j
            dw[i, j] += ETA * f_i
            dw[j, i] += ETA * f_j

    return delta_s, df, dw, sum_abs_f, sum_pos_f


def correr(cfg: CG002Config | None = None) -> dict[str, Any]:
    cfg = cfg or CG002Config()
    n = cfg.N
    rng = np.random.default_rng(cfg.seed)

    s = np.full(n, S0, dtype=np.float64)
    omega = rng.integers(0, cfg.k_firma, size=n)
    # RNG independiente para el brazo barajado (no toca el sorteo de ω)
    rng_baraja = np.random.default_rng(cfg.seed_baraja + cfg.seed)

    w_link = np.zeros((n, n), dtype=np.float64)
    f_signed = np.zeros((n, n), dtype=np.float64)
    w_orient = np.zeros((n, n), dtype=np.float64)
    f_abs_accum = np.zeros(n, dtype=np.float64)

    tau = 0
    delta_struct = 0.0
    k_micro = 0
    prev_edges: set[tuple[int, int]] = set()
    snapshots: list[dict[str, Any]] = []
    phi_series: list[float] = []

    eventos: dict[str, Any] = {
        "competencia_pasos": 0,
        "fusion_aristas_max": 0,
        "aniquilacion": False,
        "colapso_techo": 0,
    }
    s_prev = s.copy()

    if cfg.log_snapshots:
        snapshots.append(
            _snapshot(
                0, 0, s, omega, w_link, w_orient, f_signed, 0.0, 0.0, cfg.theta_cp
            )
        )

    while k_micro < cfg.k_max and tau < cfg.tau_cap and np.any(s > KAPPA_S):
        k_micro += 1
        s_before = s.copy()

        s = (1.0 - MU) * s

        omega_eff = rng_baraja.permutation(omega) if cfg.barajar_por_paso else omega

        delta_s, df, dw, sum_abs_f, _ = _paso_acoplamiento(
            s, omega_eff, cfg.alpha, cfg.k_firma, cfg.theta_cp
        )
        f_signed += df
        w_orient += dw

        for i in range(n):
            for j in range(i + 1, n):
                if df[i, j] != 0.0 or df[j, i] != 0.0:
                    f_abs_accum[i] += abs(df[i, j]) + abs(df[j, i])
                    f_abs_accum[j] += abs(df[i, j]) + abs(df[j, i])
                    coop = (max(0.0, df[i, j]) + max(0.0, df[j, i])) * 0.5
                    if coop > 0:
                        w_link[i, j] += coop
                        w_link[j, i] += coop

        s = s + delta_s
        s = np.maximum(s, 0.0)

        if cfg.s_max is not None and cfg.s_max > 0:
            mask_techo = s > cfg.s_max
            if np.any(mask_techo):
                eventos["colapso_techo"] += int(mask_techo.sum())
                s = np.where(mask_techo, 0.0, s)

        if cfg.alpha > 0 and np.any(df != 0):
            neg = df[(df < 0) & (np.tri(n, dtype=bool) | np.tri(n, k=-1, dtype=bool).T)]
            if neg.size and np.mean(np.abs(delta_s)) > 0:
                drops = [s_before[i] - s[i] for i in range(n) if s_before[i] > s[i]]
                if drops and np.mean(drops) > np.mean(np.abs(delta_s)) * 0.25:
                    eventos["competencia_pasos"] += 1

        aristas_act = sum(
            1 for i in range(n) for j in range(i + 1, n) if _arista_activa(w_link[i, j])
        )
        eventos["fusion_aristas_max"] = max(eventos["fusion_aristas_max"], aristas_act)

        if np.any((s_prev > KAPPA_S) & (s <= KAPPA_S)) and np.sum(s <= KAPPA_S) >= 2:
            eventos["aniquilacion"] = True

        curr_edges = {
            (min(i, j), max(i, j))
            for i in range(n)
            for j in range(i + 1, n)
            if _arista_activa(w_link[i, j]) and _vivo(s[i]) and _vivo(s[j])
        }
        delta_topo = 1.0 if curr_edges != prev_edges else 0.0
        prev_edges = curr_edges

        delta_step = sum_abs_f + float(np.sum(np.abs(delta_s))) + delta_topo
        delta_struct += delta_step

        if delta_step > EPS_TAU:
            tau += 1
            phi = _frustracion_subgrafo(f_signed, s)
            phi_series.append(phi)
            if cfg.log_snapshots:
                snapshots.append(
                    _snapshot(
                        tau,
                        k_micro,
                        s,
                        omega,
                        w_link,
                        w_orient,
                        f_signed,
                        delta_struct,
                        phi,
                        cfg.theta_cp,
                    )
                )

        s_prev = s.copy()

    adj = _construir_adyacencia(w_link, s)
    rho = _balance_cooperativo(f_signed)
    phi_final = _frustracion_subgrafo(f_signed, s)
    geom = _metricas_geometria(omega, cfg.k_firma)
    direccion = _metricas_direccion(w_orient, cfg.theta_cp)
    ds, ds_max = _calcular_frontera(w_link, f_abs_accum, s)
    n_vivos = int(np.sum(s > KAPPA_S))
    n_aristas = int(adj.sum() // 2)
    n_coop_vivos = int(sum(1 for i in range(n) if _vivo(s[i]) and adj[i].any()))

    v_pred, norm_pred = vector_predictor(omega, cfg.k_firma, cfg.theta_cp)
    fn = np.array(direccion["flujo_neto"], dtype=np.float64) if direccion["flujo_neto"] else np.zeros(n)
    score_track, degenerada = tracking_score(fn, v_pred, eps_deg=EPS_DEG)
    tracking_limpio = bool(
        score_track is not None
        and not degenerada
        and abs(cfg.theta_cp) > 1e-12
        and abs(score_track) >= EPS_TRACK
        and np.sign(score_track) == np.sign(cfg.theta_cp)
    )

    return {
        "N": n,
        "alpha": cfg.alpha,
        "seed": cfg.seed,
        "theta_cp": cfg.theta_cp,
        "s_max": cfg.s_max,
        "k_firma": cfg.k_firma,
        "k_micro": k_micro,
        "tau": tau,
        "delta_struct": float(delta_struct),
        "balance_cooperativo": rho,
        "rho": rho,
        "phi_frustracion": phi_final,
        "phi_series": phi_series,
        "rango_estructural": geom["rango_estructural"],
        "dimension_efectiva": geom["dimension_efectiva"],
        "criterio_B": direccion["criterio_B"],
        "t_hat_mod": direccion["t_hat_mod"],
        "asimetria": direccion["asimetria"],
        "flujo_neto": direccion["flujo_neto"],
        "norm_v_pred": norm_pred,
        "degenerada": degenerada,
        "tracking_score": score_track,
        "tracking_limpio": tracking_limpio,
        "n_coop_vivos": n_coop_vivos,
        "diametro": _diametro_grafo(adj),
        "n_aristas": n_aristas,
        "componentes": _componentes(adj, s),
        "n_vivos": n_vivos,
        "ds_max": ds_max,
        "cn4_positivo": bool(ds_max > DS_CRIT and n_vivos >= 1 and cfg.alpha > 0),
        "omega": omega.tolist(),
        "S_final": s.tolist(),
        "eventos": eventos,
        "snapshots": snapshots,
    }


def signo_estable(vals: list[float] | np.ndarray) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    mu = float(arr.mean())
    if abs(mu) < 1e-12:
        return mu, 0.0
    return mu, float((np.sign(arr) == np.sign(mu)).mean())


def evaluar_veredictos(resultados: list[dict[str, Any]]) -> dict[str, Any]:
    """V1–V6 + criterio B según protocolo v0.1b."""
    if not resultados:
        return {}

    by_key: dict[tuple[int, float, float], list[dict]] = {}
    for r in resultados:
        key = (r["N"], r["alpha"], r.get("theta_cp", 0.0))
        by_key.setdefault(key, []).append(r)

    veredictos: dict[str, Any] = {}

    # V1: emerge estructura relacional (aristas) con α=1, ausente con α=0
    for n in sorted({k[0] for k in by_key}):
        pos = [r for r in resultados if r["N"] == n and r["alpha"] == 1.0]
        neg = [r for r in resultados if r["N"] == n and r["alpha"] == 0.0]
        if pos and neg:
            frac_aristas_pos = sum(1 for r in pos if r["n_aristas"] > 0) / len(pos)
            frac_aristas_neg = sum(1 for r in neg if r["n_aristas"] > 0) / len(neg)
            veredictos[f"V1_N{n}"] = {
                "frac_aristas_alpha1": frac_aristas_pos,
                "frac_aristas_alpha0": frac_aristas_neg,
                "positivo": frac_aristas_pos > frac_aristas_neg,
            }

    for alpha in (0.0, 1.0):
        subset = [r for r in resultados if r["alpha"] == alpha]
        if subset:
            frac_tau = sum(1 for r in subset if r["tau"] > 0) / len(subset)
            veredictos[f"V2_alpha{int(alpha)}"] = {
                "frac_tau_pos": frac_tau,
                "positivo": (frac_tau >= 0.83 if alpha == 1.0 else frac_tau < 0.17),
            }

    cp_runs = [r for r in resultados if r["alpha"] == 1.0 and abs(r.get("theta_cp", 0)) > 1e-12]
    if cp_runs:
        n_deg = sum(1 for r in cp_runs if r.get("degenerada"))
        no_deg = [r for r in cp_runs if not r.get("degenerada") and r.get("tracking_score") is not None]
        scores_by_theta: dict[float, list[float]] = {}
        for r in no_deg:
            scores_by_theta.setdefault(r["theta_cp"], []).append(float(r["tracking_score"]))
        sign_fracs = []
        for vals in scores_by_theta.values():
            if len(vals) >= 2:
                _, sc = signo_estable(vals)
                sign_fracs.append(sc)
        frac_sign = float(np.mean(sign_fracs)) if sign_fracs else 0.0
        frac_limpio = sum(1 for r in no_deg if r.get("tracking_limpio")) / max(1, len(no_deg))
        veredictos["V3"] = {
            "theta_cp_activo": True,
            "frac_degenerada": n_deg / len(cp_runs),
            "frac_tracking": sum(1 for r in no_deg if abs(r.get("tracking_score", 0) or 0) >= EPS_TRACK) / max(1, len(cp_runs)),
            "frac_tracking_limpio": sum(1 for r in cp_runs if r.get("tracking_limpio")) / len(cp_runs),
            "frac_signo_estable": frac_sign,
            "positivo": frac_sign >= 0.83 and frac_limpio >= 0.5,
        }
    else:
        veredictos["V3"] = {"theta_cp_activo": False, "criterio": "N/A (θ_CP=0)", "positivo": None}

    comp = any(r["eventos"]["competencia_pasos"] > 0 for r in resultados if r["alpha"] == 1.0)
    fus = any(r["eventos"]["fusion_aristas_max"] > 0 for r in resultados if r["alpha"] == 1.0)
    ani = any(r["eventos"]["aniquilacion"] for r in resultados if r["alpha"] == 1.0)
    veredictos["V4"] = {
        "competencia_observada": comp,
        "fusion_observada": fus,
        "aniquilacion_observada": ani,
        "positivo": comp and fus and ani,
    }

    alpha1 = [r for r in resultados if r["alpha"] == 1.0]
    veredictos["V5"] = {
        "frac_diametro_pos": sum(1 for r in alpha1 if r["diametro"] > 0) / max(1, len(alpha1)),
        "positivo": any(r["diametro"] > 0 for r in alpha1),
    }

    v6 = [r for r in resultados if r["alpha"] == 1.0 and r["N"] >= 4]
    if v6:
        veredictos["V6"] = {
            "frac_cn4": sum(1 for r in v6 if r["cn4_positivo"]) / len(v6),
            "positivo": sum(1 for r in v6 if r["cn4_positivo"]) / len(v6) >= 0.5,
        }

    # Frustración agregada (C-N2.5.10)
    frust = [r for r in resultados if r["alpha"] == 1.0 and r["N"] >= 3]
    if frust:
        veredictos["VF_frustracion"] = {
            "frac_phi_pos": sum(1 for r in frust if r["phi_frustracion"] > 1e-6) / len(frust),
            "media_phi": float(np.mean([r["phi_frustracion"] for r in frust])),
        }

    # Criterio B
    if cp_runs:
        no_deg = [r for r in cp_runs if not r.get("degenerada")]
        abs_scores = [abs(float(r["tracking_score"])) for r in cp_runs if r.get("tracking_score") is not None]
        dist_gate = {}
        for thr in (0.1, 0.3, 0.5, 0.7):
            dist_gate[f"frac_limpio_abs_ge_{thr}"] = (
                sum(1 for a in abs_scores if a >= thr) / max(1, len(abs_scores))
            )
        veredictos["B_direccion"] = {
            "evaluable": True,
            "frac_degenerada": sum(1 for r in cp_runs if r.get("degenerada")) / len(cp_runs),
            "frac_tracking_limpio": sum(1 for r in cp_runs if r.get("tracking_limpio")) / len(cp_runs),
            "frac_T_hat_pos": sum(1 for r in no_deg if r["t_hat_mod"] > 1e-9) / max(1, len(no_deg)),
            "media_abs_tracking": float(np.mean(abs_scores)) if abs_scores else 0.0,
            "mediana_abs_tracking": float(np.median(abs_scores)) if abs_scores else 0.0,
            "distribucion_abs_cos": dist_gate,
            "epsilon_track": EPS_TRACK,
            "umbral_083_no_calibrado": True,
            "positivo": sum(1 for r in cp_runs if r.get("tracking_limpio")) / len(cp_runs) >= 0.83,
            "pass_cualificado_cc": "fuerza_moderada_ver_INFORME",
        }
    else:
        veredictos["B_direccion"] = {"evaluable": False, "criterio": "N/A (θ_CP=0, grafo simétrico)"}

    return veredictos


def mini_barrido_smax(
    *,
    seeds: list[int] | None = None,
    s_max_candidates: list[float] | None = None,
    log_dir: Path | None = None,
) -> dict[str, Any]:
    """Calibra S_max: N=4, seeds 1-7, θ_CP∈{0,0.3}."""
    seeds = seeds or list(range(1, 8))
    s_max_candidates = s_max_candidates or [10.0 * S0, 30.0 * S0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = log_dir or Path(__file__).resolve().parent / "logs" / f"cg002_smax_calib_{ts}"
    log_dir.mkdir(parents=True, exist_ok=True)

    filas: list[dict[str, Any]] = []
    for s_max in s_max_candidates:
        for seed in seeds:
            for theta in (0.0, 0.3):
                r = correr(CG002Config(N=4, alpha=1.0, seed=seed, theta_cp=theta, s_max=s_max, k_max=K_MAX_MIN))
                filas.append({
                    "s_max": s_max,
                    "seed": seed,
                    "theta_cp": theta,
                    "n_vivos": r["n_vivos"],
                    "n_coop_vivos": r["n_coop_vivos"],
                    "colapso_techo": r["eventos"]["colapso_techo"],
                    "tau": r["tau"],
                })

    agg: dict[float, dict[str, float]] = {}
    for sm in s_max_candidates:
        sub = [f for f in filas if f["s_max"] == sm]
        agg[sm] = {
            "media_coop_vivos": float(np.mean([f["n_coop_vivos"] for f in sub])),
            "media_vivos": float(np.mean([f["n_vivos"] for f in sub])),
            "media_colapso_techo": float(np.mean([f["colapso_techo"] for f in sub])),
        }

    elegido = max(agg.keys(), key=lambda sm: (agg[sm]["media_coop_vivos"], -agg[sm]["media_colapso_techo"]))
    viable = [sm for sm, a in agg.items() if a["media_coop_vivos"] >= 0.5]
    out = {
        "timestamp": ts,
        "candidatos": s_max_candidates,
        "filas": filas,
        "agregados": {str(k): v for k, v in agg.items()},
        "s_max_elegido": elegido,
        "s_max_viable": viable[0] if viable else None,
        "diagnostico": "colapso_duro_mata_coop" if not viable else "ok",
    }
    with (log_dir / "resultado.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out


def run_batch(
    *,
    N_list: list[int] | None = None,
    seeds: list[int] | None = None,
    alphas: list[float] | None = None,
    theta_cps: list[float] | None = None,
    k_firma: int = K_FIRMA,
    k_max: int = K_MAX_MIN,
    s_max: float | None = S_MAX_DEFAULT,
    log_dir: Path | None = None,
    log_snapshots: bool = False,
) -> Path:
    N_list = N_list or [3, 4, 8]
    seeds = seeds or list(range(1, 4))
    alphas = alphas or [0.0, 1.0]
    theta_cps = theta_cps if theta_cps is not None else [0.0]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = log_dir or Path(__file__).resolve().parent / "logs" / f"cg002_{ts}"
    log_dir.mkdir(parents=True, exist_ok=True)

    resultados: list[dict[str, Any]] = []
    for n in N_list:
        for alpha in alphas:
            for theta_cp in theta_cps:
                for seed in seeds:
                    cfg = CG002Config(
                        N=n,
                        alpha=alpha,
                        seed=seed,
                        k_max=k_max,
                        k_firma=k_firma,
                        theta_cp=theta_cp,
                        s_max=s_max,
                        log_snapshots=log_snapshots,
                    )
                    r = correr(cfg)
                    resultados.append(r)
                    if log_snapshots and r["snapshots"]:
                        tag = f"n{n}_a{int(alpha)}_cp{theta_cp:.2f}_s{seed}"
                        snap_path = log_dir / f"grafo_{tag}.jsonl"
                        with snap_path.open("w", encoding="utf-8") as f:
                            for snap in r["snapshots"]:
                                f.write(json.dumps(snap) + "\n")

    cols = [
        "N", "alpha", "theta_cp", "s_max", "k_firma", "seed", "k_micro", "tau", "delta_struct",
        "balance_cooperativo", "phi_frustracion", "rango_estructural", "dimension_efectiva",
        "criterio_B", "t_hat_mod", "asimetria", "degenerada", "tracking_score", "tracking_limpio",
        "frac_degenerada", "diametro", "n_aristas", "n_vivos", "n_coop_vivos",
        "ds_max", "cn4_positivo",
    ]
    with (log_dir / "series.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in resultados:
            w.writerow({k: r.get(k, "") for k in cols})

    out = {
        "protocolo": "CG002 v0.1c",
        "s_max": s_max,
        "timestamp": ts,
        "N_list": N_list,
        "seeds": seeds,
        "alphas": alphas,
        "theta_cps": theta_cps,
        "k_firma": k_firma,
        "k_max": k_max,
        "n_corridas": len(resultados),
        "veredictos": evaluar_veredictos(resultados),
        "resultados": [{k: v for k, v in r.items() if k != "snapshots"} for r in resultados],
    }
    with (log_dir / "resultado.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    return log_dir


def smoke_test() -> None:
    print("=== CG002 smoke v0.1c ===\n")

    cal = mini_barrido_smax()
    s_cap = cal.get("s_max_viable")  # None si colapso duro mata cooperación
    print(f"Mini-barrido S_max: diagnostico={cal.get('diagnostico')} viable={s_cap}")
    print(f"  agregados: {cal['agregados']}\n")

    log1 = run_batch(
        N_list=[3, 4, 8],
        seeds=[1, 2, 3],
        alphas=[0.0, 1.0],
        theta_cps=[0.0],
        k_firma=8,
        k_max=K_MAX_MIN,
        s_max=s_cap,
        log_snapshots=False,
    )
    print(f"Simétrico: {log1}")

    # N=3 K=12 (frustración triádica)
    log2 = run_batch(
        N_list=[3],
        seeds=[1, 2, 3],
        alphas=[1.0],
        theta_cps=[0.0],
        k_firma=12,
        k_max=K_MAX_MIN,
        s_max=s_cap,
        log_snapshots=False,
    )
    print(f"N=3 K=12: {log2}")

    log3 = run_batch(
        N_list=[4, 8],
        seeds=list(range(1, 8)),
        alphas=[1.0],
        theta_cps=[0.0, 0.1, 0.3, 0.5],
        k_firma=8,
        k_max=K_MAX_MIN,
        s_max=s_cap,
        log_snapshots=True,
    )
    print(f"θ_CP: {log3}")

    for label, path in [("simetrico", log1), ("K12", log2), ("theta_cp", log3)]:
        with (path / "resultado.json").open(encoding="utf-8") as f:
            res = json.load(f)
        print(f"\n--- {label} ---")
        for k, v in res["veredictos"].items():
            p = v.get("positivo")
            tag = "PASS" if p is True else ("FAIL" if p is False else "N/A")
            print(f"  {k}: {tag}  {v}")

    print("\n--- Control G (α=0) ---")
    for seed in [1, 2, 3]:
        r = correr(CG002Config(N=4, alpha=0.0, seed=seed, k_max=K_MAX_MIN))
        ok = r["tau"] == 0 and r["delta_struct"] == 0.0 and r["n_aristas"] == 0
        print(f"  seed={seed}: tau={r['tau']} Δ={r['delta_struct']:.4f} aristas={r['n_aristas']} {'OK' if ok else 'ALERTA'}")

    print("\n--- Control G' (θ_CP=0 dirección) ---")
    r0 = correr(CG002Config(N=8, alpha=1.0, seed=1, theta_cp=0.0, k_max=K_MAX_MIN))
    r5 = correr(CG002Config(N=8, alpha=1.0, seed=1, theta_cp=0.5, k_max=K_MAX_MIN))
    print(f"  θ=0: asimetria={r0['asimetria']:.6f} T_hat={r0['t_hat_mod']:.6f} B={r0['criterio_B']}")
    print(f"  θ=0.5: asimetria={r5['asimetria']:.6f} T_hat={r5['t_hat_mod']:.6f} B={r5['criterio_B']}")


if __name__ == "__main__":
    smoke_test()