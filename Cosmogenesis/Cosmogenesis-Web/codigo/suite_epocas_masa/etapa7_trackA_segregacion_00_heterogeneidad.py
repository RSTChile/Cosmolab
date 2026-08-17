#!/usr/bin/env python3
"""
ETAPA 7 / TRACK A — Paso 1: heterogeneidad intrínseca de mass_proxy
====================================================================

Pregunta: ¿los átomos estables tienen mass_proxy REALMENTE heterogéneo
(varianza significativa entre átomos de la MISMA corrida), o son casi
todos del mismo tamaño/masa? Si no hay heterogeneidad real, ningún
observable de "masa" puede discriminar nada — hay que reportarlo y
NO seguir forzando una prueba de segregación.

Este script es de SOLO MEDICIÓN (no pre-registra ninguna prueba de
PASS/FAIL causal). Copia (no importa) las piezas físicas necesarias
del motor v6 (suite_epocas_masa_v6_mass_linaje.py): generación de campo,
formación de átomos E3 estricta (components_strict), y persistencia
(match_persist). NO se copian ni usan co_member_score / n_long_co_pairs
/ fusion_events — no hacen falta para esta medición.

mass_proxy (definición idéntica a v6, línea ~189 del original):
    mass_proxy = max(sum_phi, 1e-6) * (1.0 + f_core)
donde sum_phi = suma de phi (campo de orden) en los nodos del átomo,
f_core = fracción de nodos con |Φ| por encima del umbral de núcleo.
Es un valor 100% LOCAL al átomo (campo Φ propio), sin ninguna
referencia a linaje, co-membresía, fusión, ni posición de otros átomos.

Salida: histograma + estadísticas (mean, std, CV=std/mean, rango,
percentiles) de mass_proxy entre todos los átomos ESTABLES (age >=
PERSIST_STEPS) recolectados a lo largo de la ventana E4, para varias
semillas del set estándar, en modo REAL (G_GRAV=0.20, igual que v6).
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[2] / "results" / "etapa7_trackA_segregacion"
OUT.mkdir(parents=True, exist_ok=True)

# ---- constantes E3 estricto (idénticas a v6) ----
K_MIN, K_MAX = 4, 14
F_CORE_MIN, F_CORE_MAX = 0.15, 0.75
COHESION_MIN, COHESION_MAX = 1.2, 6.5
PERSIST_STEPS = 4
VEV_POST_MIN = 0.10
PHI_CORE_THR = 0.35
CENTROID_TOL = 2.5


@dataclass
class P:
    L: int = 28
    pasos: int = 400
    H_EXP: float = 6.0
    H_TOPO: float = 0.01
    seed: int = 2025
    R0: float = 2.0
    U: float = 0.5
    TC: float = 0.55
    D_PHI: float = 0.05
    DT_PHI: float = 0.08
    SIGMA0: float = 0.10
    G_RHO: float = 0.8
    FREEZE_TNORM: float = 0.40
    RHO0: float = 1.0
    RHO_FREEZE: float = 0.05
    MIX0: float = 0.35
    MIX_FLOOR: float = 0.08
    MIX_FLOOR_RHO: float = 0.02
    ALPHA_CUT: float = 2.5
    medium_on: bool = True
    G_GRAV: float = 0.20
    GRAV_START_FRAC: float = 0.65
    grav_mode: str = "real"
    Y0: float = 0.3


def medium_norm(Phi):
    a = np.abs(Phi)
    p95 = float(np.percentile(a, 95)) + 1e-12
    return np.clip(a / p95, 0.0, 1.0)


def weighted_cut(ar, ad, Phi, nc, rng, alpha, blind=False):
    if nc <= 0:
        return
    ph = None if blind else medium_norm(Phi)
    if ar.any():
        idx = np.argwhere(ar)
        tot = int(ar.sum() + ad.sum())
        n_r = min(int(round(nc * float(ar.sum()) / max(tot, 1))), len(idx))
        if n_r > 0:
            if blind:
                p = None
            else:
                edge = 0.5 * (ph + np.roll(ph, -1, 1))
                w = np.where(ar, (1.0 - edge + 1e-3) ** alpha, 0.0)
                flat = w[tuple(idx.T)]
                p = None if flat.sum() <= 0 else flat / flat.sum()
            sel = rng.choice(len(idx), size=n_r, replace=False, p=p)
            for i in sel:
                ar[tuple(idx[i])] = False
        rem = nc - n_r
    else:
        rem = nc
    if rem > 0 and ad.any():
        idx = np.argwhere(ad)
        n_d = min(rem, len(idx))
        if n_d > 0:
            if blind:
                p = None
            else:
                edge = 0.5 * (ph + np.roll(ph, -1, 0))
                w = np.where(ad, (1.0 - edge + 1e-3) ** alpha, 0.0)
                flat = w[tuple(idx.T)]
                p = None if flat.sum() <= 0 else flat / flat.sum()
            sel = rng.choice(len(idx), size=n_d, replace=False, p=p)
            for i in sel:
                ad[tuple(idx[i])] = False


def components_strict(phi, ar, ad, Phi):
    media = float(phi.mean())
    n = phi.shape[0]
    visto = np.zeros_like(phi, dtype=bool)
    out = []
    for y in range(n):
        for x in range(n):
            if visto[y, x]:
                continue
            q = deque([(y, x)])
            visto[y, x] = True
            lado = phi[y, x] >= media
            nodes = [(y, x)]
            sum_phi = float(phi[y, x])
            sum_abs = float(abs(Phi[y, x]))
            n_core = 1 if abs(Phi[y, x]) >= PHI_CORE_THR else 0
            perim = 0
            while q:
                cy, cx = q.popleft()
                for ny, nx, bond in (
                    (cy, (cx + 1) % n, ar[cy, cx]),
                    (cy, (cx - 1) % n, ar[cy, (cx - 1) % n]),
                    ((cy + 1) % n, cx, ad[cy, cx]),
                    ((cy - 1) % n, cx, ad[(cy - 1) % n, cx]),
                ):
                    same = (phi[ny, nx] >= media) == lado
                    if not bond or not same:
                        perim += 1
                    if bond and not visto[ny, nx] and same:
                        visto[ny, nx] = True
                        q.append((ny, nx))
                        nodes.append((ny, nx))
                        sum_phi += float(phi[ny, nx])
                        sum_abs += float(abs(Phi[ny, nx]))
                        if abs(Phi[ny, nx]) >= PHI_CORE_THR:
                            n_core += 1
            k = len(nodes)
            cy = float(np.mean([p_[0] for p_ in nodes]))
            cx = float(np.mean([p_[1] for p_ in nodes]))
            f_core = n_core / k if k else 0.0
            cohes = perim / (np.sqrt(k) + 1e-12)
            is_atom = (
                K_MIN <= k <= K_MAX
                and F_CORE_MIN <= f_core <= F_CORE_MAX
                and COHESION_MIN <= cohes <= COHESION_MAX
                and n_core >= 1
                and (k - n_core) >= 1
            )
            out.append(
                {
                    "k": k,
                    "perim": perim,
                    "sum_phi": sum_phi,
                    "n_core": n_core,
                    "f_core": f_core,
                    "cohes": float(cohes),
                    "cy": cy,
                    "cx": cx,
                    "is_atom": is_atom,
                    "mass_proxy": max(sum_phi, 1e-6) * (1.0 + f_core),
                }
            )
    return out


def match_persist(atoms_now, tracks, next_id_start=1000):
    used = set()
    new_tracks = []
    max_id = next_id_start
    for tr in tracks:
        max_id = max(max_id, tr.get("id", 0))
    for at in atoms_now:
        best_i, best_d = None, 1e9
        for i, tr in enumerate(tracks):
            if i in used:
                continue
            d = np.hypot(tr["cy"] - at["cy"], tr["cx"] - at["cx"])
            if d < best_d:
                best_d, best_i = d, i
        if best_i is not None and best_d <= CENTROID_TOL:
            tr = tracks[best_i]
            used.add(best_i)
            tid = tr.get("id", max_id + 1)
            max_id = max(max_id, tid)
            new_tracks.append(
                {"cy": at["cy"], "cx": at["cx"], "age": tr["age"] + 1,
                 "mass": at["mass_proxy"], "k": at["k"], "id": tid}
            )
        else:
            max_id += 1
            new_tracks.append(
                {"cy": at["cy"], "cx": at["cx"], "age": 1,
                 "mass": at["mass_proxy"], "k": at["k"], "id": max_id}
            )
    n_stable = sum(1 for tr in new_tracks if tr["age"] >= PERSIST_STEPS)
    return new_tracks, n_stable


def simulate_measure(p: P) -> dict:
    """Corre el campo hasta el final; recolecta mass_proxy de TODOS los
    átomos estables vistos en cada paso E4 (con id único por track, se
    guarda solo una medición por id: la PRIMERA vez que se vuelve estable
    dentro de la ventana E4, para no contar el mismo átomo muchas veces)."""
    rng = np.random.default_rng(p.seed)
    L = p.L
    phi = np.ones((L, L)) + 0.2 * rng.normal(size=(L, L))
    Phi = 0.06 * rng.normal(size=(L, L))
    ar = np.ones((L, L), dtype=bool)
    ad = np.ones((L, L), dtype=bool)

    tracks = []
    grav_start = int(p.GRAV_START_FRAC * p.pasos)
    seen_ids = set()
    mass_by_id = {}
    k_by_id = {}

    for step in range(p.pasos):
        tg = step / max(p.pasos - 1, 1)
        a = float(np.exp(p.H_EXP * tg))
        Tnorm = float(np.exp(-p.H_EXP * tg))
        rho_c = p.RHO0 / (a**3)
        rho_hat_c = rho_c / p.RHO0
        frozen = (Tnorm < p.FREEZE_TNORM) or (rho_c < p.RHO_FREEZE)
        e4_window = step >= grav_start and frozen

        if not frozen:
            rho_hat = phi / (float(np.mean(phi)) + 1e-12)
            r_field = p.R0 * (Tnorm - p.TC) - p.G_RHO * (rho_hat - 1.0)
            lap = (
                np.roll(Phi, -1, 1) + np.roll(Phi, 1, 1)
                + np.roll(Phi, -1, 0) + np.roll(Phi, 1, 0) - 4 * Phi
            )
            dV = 2 * r_field * Phi + 4 * p.U * Phi**3
            D_eff = p.D_PHI * rho_hat_c
            sig = p.SIGMA0 * np.sqrt(max(Tnorm, 1e-6) * max(rho_hat_c, 1e-12))
            Phi = Phi + p.DT_PHI * (-dV + D_eff * lap) + sig * rng.normal(size=(L, L))

        ph = medium_norm(Phi)
        left = np.roll(ar, 1, 1)
        up = np.roll(ad, 1, 0)
        cnt = ar.astype(int) + left.astype(int) + ad.astype(int) + up.astype(int)
        s = (
            np.where(ar, np.roll(phi, -1, 1), 0)
            + np.where(left, np.roll(phi, 1, 1), 0)
            + np.where(ad, np.roll(phi, -1, 0), 0)
            + np.where(up, np.roll(phi, 1, 0), 0)
        )
        mean = np.divide(s, cnt, out=np.zeros_like(phi), where=cnt > 0)
        if p.medium_on:
            mix_med = p.MIX_FLOOR + (p.MIX0 - p.MIX_FLOOR) * ph
        else:
            mix_med = np.full_like(phi, p.MIX0)
        mix = mix_med * rho_hat_c + p.MIX_FLOOR_RHO * (1.0 - min(rho_hat_c, 1.0))
        phi_new = phi.copy()
        msk = cnt > 0
        phi_new[msk] = phi[msk] + mix[msk] * (mean[msk] - phi[msk])
        phi = phi_new

        H_fis = p.H_TOPO * np.sqrt(Tnorm + 1e-12)
        tot = int(ar.sum() + ad.sum())
        nc = int(round(H_fis * tot))
        if nc > 0:
            weighted_cut(ar, ad, Phi, nc, rng, p.ALPHA_CUT, blind=not p.medium_on)

        cl = components_strict(phi, ar, ad, Phi)
        vev_ok = float(np.mean(np.abs(Phi))) >= VEV_POST_MIN
        atoms = [c for c in cl if c["is_atom"]] if (frozen and vev_ok) else []
        tracks, n_stable = match_persist(atoms, tracks)
        stable_atoms = [t for t in tracks if t["age"] >= PERSIST_STEPS]

        if e4_window:
            for t in stable_atoms:
                tid = int(t["id"])
                if tid not in seen_ids:
                    seen_ids.add(tid)
                    mass_by_id[tid] = t["mass"]
                    k_by_id[tid] = t["k"]

    return {"mass_by_id": mass_by_id, "k_by_id": k_by_id}


def stats(vals):
    v = np.array(vals, dtype=float)
    if len(v) == 0:
        return {"n": 0}
    mean = float(v.mean())
    std = float(v.std())
    return {
        "n": int(len(v)),
        "mean": mean,
        "std": std,
        "cv": std / mean if mean > 0 else None,
        "min": float(v.min()),
        "max": float(v.max()),
        "p10": float(np.percentile(v, 10)),
        "p25": float(np.percentile(v, 25)),
        "p50": float(np.percentile(v, 50)),
        "p75": float(np.percentile(v, 75)),
        "p90": float(np.percentile(v, 90)),
    }


def histogram_ascii(vals, bins=12, width=40):
    v = np.array(vals, dtype=float)
    if len(v) == 0:
        return "(sin datos)"
    counts, edges = np.histogram(v, bins=bins)
    maxc = counts.max() if counts.max() > 0 else 1
    lines = []
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * int(round(width * c / maxc))
        lines.append(f"  [{lo:8.2f}, {hi:8.2f}) {c:4d} {bar}")
    return "\n".join(lines)


def main():
    seeds = (7, 42, 99, 777, 2025, 3141, 8191, 99991, 12345, 54321)
    all_masses = []
    per_seed = {}
    for s in seeds:
        r = simulate_measure(P(seed=s, G_GRAV=0.20, grav_mode="real"))
        masses = list(r["mass_by_id"].values())
        ks = list(r["k_by_id"].values())
        per_seed[s] = {"mass": stats(masses), "k": stats(ks), "n_atoms": len(masses)}
        all_masses.extend(masses)
        print(f"seed={s:5d} n_atoms_estables_E4={len(masses):3d} "
              f"mean_mass={per_seed[s]['mass'].get('mean'):.3f} "
              f"std_mass={per_seed[s]['mass'].get('std'):.3f} "
              f"cv={per_seed[s]['mass'].get('cv')}")

    pooled = stats(all_masses)
    print("\n=== POOLED (todas las semillas, todos los átomos estables en E4) ===")
    print(json.dumps(pooled, indent=2))
    print("\nHistograma (pooled, mass_proxy):")
    print(histogram_ascii(all_masses, bins=14))

    out = {
        "protocol_note": "SOLO MEDICION — no es prueba causal ni de PASS/FAIL. "
                          "mass_proxy = max(sum_phi,1e-6)*(1+f_core), 100% local al atomo, "
                          "sin co_member_score/n_long_co_pairs/fusion_events.",
        "seeds": list(seeds),
        "per_seed": {str(k): v for k, v in per_seed.items()},
        "pooled": pooled,
    }
    path = OUT / "trackA_00_heterogeneidad_mass_proxy.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nJSON -> {path}")


if __name__ == "__main__":
    main()
