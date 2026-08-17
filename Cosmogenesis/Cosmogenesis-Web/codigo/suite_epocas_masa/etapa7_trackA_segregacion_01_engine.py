#!/usr/bin/env python3
"""
ETAPA 7 / TRACK A — Segregación gravitacional por masa intrínseca
====================================================================

Pre-registro: PROTOCOLO_TRACKA_SEGREGACION_PREREGISTRO.md (escrito ANTES
de este motor de producción).

Pregunta: ¿los átomos con mass_proxy alto (intrínseco, del campo Φ propio,
SIN linaje) migran preferentemente hacia el centro de su grupo ligado
gravitacionalmente a lo largo de la ventana E4, de forma REAL≻SHUFFLE?

Este archivo es NUEVO. No importa ni edita suite_epocas_masa_v1..v6.py.
No calcula co_member_score, n_long_co_pairs, ni fusion_events — no
existen en este código. La única "masa" es mass_proxy local
(max(sum_phi,1e-6)*(1+f_core), idéntica a v6 línea ~189).

Diseño (resumen, detalle completo en el pre-registro):
  1. Campo evoluciona igual que v6 hasta el primer paso de la ventana E4
     (frozen and step>=grav_start).
  2. En ESE instante se congela la población de átomos estables (age>=4):
     mass_intrinsic fijo por átomo, posición inicial = centroide de campo.
     No se agregan átomos nuevos ni se re-miden masas después (evita
     circularidad masa<->posición y evita el bug de v3-v6 donde el N-body
     se reiniciaba cada vez que cambiaba el conteo de átomos).
  3. Se integra N-body (idéntico físicamente a v6: softening, cutoff,
     DT_NB) en modos real/off/shuffle/invert, SIN reset de posición,
     durante el resto de la ventana E4.
  4. Al último paso: grupos por proximidad (union-find, mismo
     GROUP_LINK_R que v6). Por grupo (tamaño>=3): correlación de Pearson
     dentro-de-grupo entre mass_intrinsic y distancia-al-centroide.
  5. r_seed_modo = promedio no ponderado de r_grupo sobre grupos válidos.
"""
from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[2] / "results" / "etapa7_trackA_segregacion"
OUT.mkdir(parents=True, exist_ok=True)

PROTOCOL_ID = "TRACKA_SEGREGACION_2026-07-23"

# ---- E3 estricto (idéntico a v6) ----
K_MIN, K_MAX = 4, 14
F_CORE_MIN, F_CORE_MAX = 0.15, 0.75
COHESION_MIN, COHESION_MAX = 1.2, 6.5
PERSIST_STEPS = 4
VEV_POST_MIN = 0.10
PHI_CORE_THR = 0.35
CENTROID_TOL = 2.5

# ---- E4 N-body (idéntico a v6) ----
SOFTENING = 1.2
DT_NB = 0.35
GROUP_LINK_R = 4.5
FORCE_CUTOFF = 8.0

# ---- criterio pre-registrado ----
R_REAL_MAX = -0.15       # r_REAL debe ser <= esto
MARGIN_VS_SHUFFLE = 0.10  # r_REAL debe ser <= r_SHUFFLE - este margen
RATE_PASS = 0.55
RATE_PARTIAL = 0.30
MIN_GROUP_SIZE = 3
MIN_VALID_SEEDS = 6

SEEDS_STD = (7, 42, 99, 777, 2025, 3141, 8191, 99991, 12345, 54321)
SEEDS_G_SWEEP = (2025, 42, 777, 3141)
G_SWEEP_VALUES = (0.05, 0.10, 0.20, 0.30, 0.45)


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
    grav_mode: str = "real"  # real | off | shuffle | invert


# ---------------------------------------------------------------- campo ----

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
                    "k": k, "cy": cy, "cx": cx, "is_atom": is_atom,
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
                 "mass_proxy": at["mass_proxy"], "id": tid}
            )
        else:
            max_id += 1
            new_tracks.append(
                {"cy": at["cy"], "cx": at["cx"], "age": 1,
                 "mass_proxy": at["mass_proxy"], "id": max_id}
            )
    return new_tracks


# --------------------------------------------------------------- N-body ----

def toroidal_delta(a, b, L):
    d = b - a
    if d > L / 2:
        d -= L
    if d < -L / 2:
        d += L
    return d


def nbody_step(pos, masses, G, L, mode, src_mass):
    """Un paso de integración. src_mass = masas usadas como FUENTE (shuffle
    permuta esto, receptor usa siempre `masses`)."""
    N = len(masses)
    if N == 0 or mode == "off" or G <= 0:
        return pos
    acc = np.zeros_like(pos)
    sign = -1.0 if mode == "invert" else 1.0
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            dy = toroidal_delta(pos[i, 0], pos[j, 0], L)
            dx = toroidal_delta(pos[i, 1], pos[j, 1], L)
            r = float(np.hypot(dy, dx))
            if r > FORCE_CUTOFF:
                continue
            r2 = r * r + SOFTENING**2
            r_soft = np.sqrt(r2)
            strength = G * masses[i] * src_mass[j] / r2
            acc[i, 0] += sign * strength * (dy / r_soft)
            acc[i, 1] += sign * strength * (dx / r_soft)
    pos = pos + DT_NB * acc
    pos[:, 0] = np.mod(pos[:, 0], L)
    pos[:, 1] = np.mod(pos[:, 1], L)
    return pos


def groups_union_find(pos, L, link_r=GROUP_LINK_R):
    N = len(pos)
    parent = list(range(N))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(N):
        for j in range(i + 1, N):
            dy = toroidal_delta(pos[i, 0], pos[j, 0], L)
            dx = toroidal_delta(pos[i, 1], pos[j, 1], L)
            if np.hypot(dy, dx) < link_r:
                union(i, j)

    clusters = {}
    for i in range(N):
        r = find(i)
        clusters.setdefault(r, []).append(i)
    return list(clusters.values())


def toroidal_centroid(pts, L):
    """Centroide circular (para toro): usa media angular por eje."""
    ang_y = pts[:, 0] / L * 2 * np.pi
    ang_x = pts[:, 1] / L * 2 * np.pi
    cy = (np.arctan2(np.mean(np.sin(ang_y)), np.mean(np.cos(ang_y))) / (2 * np.pi)) % 1.0 * L
    cx = (np.arctan2(np.mean(np.sin(ang_x)), np.mean(np.cos(ang_x))) / (2 * np.pi)) % 1.0 * L
    return cy, cx


def pearson_r(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return None
    sx, sy = x.std(), y.std()
    if sx <= 1e-12 or sy <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman_r(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return None

    def rank(v):
        order = np.argsort(v)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(v))
        # average ties
        uniq, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.zeros(len(uniq))
        np.add.at(sums, inv, ranks)
        avg = sums / cnt
        return avg[inv]

    return pearson_r(rank(x), rank(y))


# ------------------------------------------------------------- pipeline ----

def field_to_frozen_population(p: P):
    """Evoluciona el campo hasta el primer paso de la ventana E4 y
    devuelve la población de átomos estables en ese instante (lista de
    dicts con id, mass_proxy fijo, posición inicial)."""
    rng = np.random.default_rng(p.seed)
    L = p.L
    phi = np.ones((L, L)) + 0.2 * rng.normal(size=(L, L))
    Phi = 0.06 * rng.normal(size=(L, L))
    ar = np.ones((L, L), dtype=bool)
    ad = np.ones((L, L), dtype=bool)

    tracks = []
    grav_start = int(p.GRAV_START_FRAC * p.pasos)
    frozen_population = None
    freeze_step = None

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
        tracks = match_persist(atoms, tracks)
        stable_atoms = [t for t in tracks if t["age"] >= PERSIST_STEPS]

        if e4_window and frozen_population is None:
            # primer instante de la ventana E4: congelar población
            frozen_population = [
                {"id": t["id"], "mass_intrinsic": t["mass_proxy"],
                 "cy": t["cy"], "cx": t["cx"]}
                for t in stable_atoms
            ]
            freeze_step = step
            break  # no seguimos evolucionando el campo; el resto es N-body puro

    return frozen_population, freeze_step, grav_start, p.pasos


def run_segregation(seed: int, mode: str, G: float, pasos: int = 400) -> dict:
    """Corre campo -> congela población -> integra N-body en `mode` ->
    mide correlación mass_intrinsic vs distancia-a-centroide final."""
    p = P(seed=seed, pasos=pasos, G_GRAV=G, grav_mode=mode)
    frozen_pop, freeze_step, grav_start, total_pasos = field_to_frozen_population(p)

    if frozen_pop is None or len(frozen_pop) < MIN_GROUP_SIZE:
        return {
            "seed": seed, "mode": mode, "G": G, "valid": False,
            "reason": "poblacion_insuficiente_al_congelar",
            "n_population": 0 if frozen_pop is None else len(frozen_pop),
        }

    N = len(frozen_pop)
    pos = np.array([[a["cy"], a["cx"]] for a in frozen_pop], dtype=float)
    mass = np.array([a["mass_intrinsic"] for a in frozen_pop], dtype=float)
    ids = [a["id"] for a in frozen_pop]

    rng_shuf = np.random.default_rng(seed * 7919 + 13)  # independiente del campo
    if mode == "shuffle":
        perm = rng_shuf.permutation(N)
        src_mass = mass[perm]
    else:
        src_mass = mass

    n_steps_e4 = total_pasos - freeze_step
    L = p.L
    for _ in range(n_steps_e4):
        pos = nbody_step(pos, mass, G, L, mode, src_mass)

    # agrupar al último paso
    groups = groups_union_find(pos, L, GROUP_LINK_R)
    group_rs_pearson = []
    group_rs_spearman = []
    group_sizes = []
    for g_idx in groups:
        if len(g_idx) < MIN_GROUP_SIZE:
            continue
        pts = pos[g_idx]
        cy, cx = toroidal_centroid(pts, L)
        dists = []
        for (y, x) in pts:
            dy = toroidal_delta(cy, y, L)
            dx = toroidal_delta(cx, x, L)
            dists.append(float(np.hypot(dy, dx)))
        m_g = mass[g_idx]
        rp = pearson_r(m_g, dists)
        rs = spearman_r(m_g, dists)
        if rp is not None:
            group_rs_pearson.append(rp)
        if rs is not None:
            group_rs_spearman.append(rs)
        group_sizes.append(len(g_idx))

    if not group_rs_pearson:
        return {
            "seed": seed, "mode": mode, "G": G, "valid": False,
            "reason": "sin_grupos_validos_tamano>=3_al_final",
            "n_population": N,
        }

    return {
        "seed": seed, "mode": mode, "G": G, "valid": True,
        "n_population": N,
        "n_groups_valid": len(group_rs_pearson),
        "group_sizes": group_sizes,
        "r_mean_pearson": float(np.mean(group_rs_pearson)),
        "r_mean_spearman": float(np.mean(group_rs_spearman)) if group_rs_spearman else None,
        "r_per_group_pearson": group_rs_pearson,
        "mass_range": [float(mass.min()), float(mass.max())],
    }


def verdict_from_rate(rate, n_valid):
    if n_valid < MIN_VALID_SEEDS:
        return "INCONCLUSO"
    if rate >= RATE_PASS:
        return "PASS"
    if rate >= RATE_PARTIAL:
        return "PARTIAL"
    return "FAIL"


def main():
    t0 = time.time()
    print(f"=== ETAPA 7 / TRACK A — segregación gravitacional ({PROTOCOL_ID}) ===\n")

    # ---- 1) controles principales: 10 semillas, G=0.20, 4 modos ----
    print("--- controles principales (10 semillas, G=0.20) ---")
    ctrl_rows = []
    for s in SEEDS_STD:
        row = {"seed": s}
        for mode in ("real", "off", "shuffle", "invert"):
            row[mode] = run_segregation(s, mode, G=0.20, pasos=400)
        ctrl_rows.append(row)
        r_real = row["real"].get("r_mean_pearson")
        r_shuf = row["shuffle"].get("r_mean_pearson")
        r_off = row["off"].get("r_mean_pearson")
        r_inv = row["invert"].get("r_mean_pearson")
        print(
            f"  seed={s:5d} "
            f"r_real={r_real if r_real is None else round(r_real,3)} "
            f"r_shuf={r_shuf if r_shuf is None else round(r_shuf,3)} "
            f"r_off={r_off if r_off is None else round(r_off,3)} "
            f"r_inv={r_inv if r_inv is None else round(r_inv,3)} "
            f"n_pop={row['real'].get('n_population')}"
        )

    # ---- veredicto pre-registrado ----
    wins = []
    valid_seeds = []
    off_quiet = []
    invert_not_neg = []
    for row in ctrl_rows:
        real, shuf = row["real"], row["shuffle"]
        if not (real.get("valid") and shuf.get("valid")):
            continue
        r_real = real["r_mean_pearson"]
        r_shuf = shuf["r_mean_pearson"]
        valid_seeds.append(row["seed"])
        win = (r_real <= R_REAL_MAX) and (r_real <= r_shuf - MARGIN_VS_SHUFFLE)
        wins.append(win)
        off = row["off"]
        if off.get("valid"):
            off_quiet.append(abs(off["r_mean_pearson"]) < 0.15)
        inv = row["invert"]
        if inv.get("valid"):
            invert_not_neg.append(inv["r_mean_pearson"] >= -0.05)

    n_valid = len(valid_seeds)
    rate = (sum(wins) / n_valid) if n_valid else 0.0
    verdict = verdict_from_rate(rate, n_valid)

    print(f"\nrate seed_win (REAL segrega, R/S separado) = {rate:.2f} "
          f"sobre n_valid={n_valid} semillas")
    print(f"veredicto controles principales: {verdict}")

    # ---- 2) barrido G ----
    print("\n--- barrido G_GRAV (4 semillas) ---")
    g_rows = []
    for G in G_SWEEP_VALUES:
        sub_wins = []
        for s in SEEDS_G_SWEEP:
            real = run_segregation(s, "real", G=G, pasos=400)
            shuf = run_segregation(s, "shuffle", G=G, pasos=400)
            row = {"G": G, "seed": s, "real": real, "shuffle": shuf}
            g_rows.append(row)
            if real.get("valid") and shuf.get("valid"):
                r_real = real["r_mean_pearson"]
                r_shuf = shuf["r_mean_pearson"]
                win = (r_real <= R_REAL_MAX) and (r_real <= r_shuf - MARGIN_VS_SHUFFLE)
                sub_wins.append(win)
        rate_g = (sum(sub_wins) / len(sub_wins)) if sub_wins else None
        print(f"  G={G:.2f} rate_win={rate_g} (n_valid={len(sub_wins)})")

    elapsed = time.time() - t0

    out = {
        "protocol_id": PROTOCOL_ID,
        "preregistro": "PROTOCOLO_TRACKA_SEGREGACION_PREREGISTRO.md",
        "criterio_pass_verbatim": {
            "seed_win": "r_REAL <= -0.15 AND r_REAL <= r_SHUFFLE - 0.10",
            "rate_pass": RATE_PASS,
            "rate_partial": RATE_PARTIAL,
            "min_group_size": MIN_GROUP_SIZE,
            "min_valid_seeds": MIN_VALID_SEEDS,
        },
        "no_usa_linaje": True,
        "observables_prohibidos_no_usados": [
            "co_member_score", "n_long_co_pairs", "fusion_events"
        ],
        "seeds_std": list(SEEDS_STD),
        "seeds_g_sweep": list(SEEDS_G_SWEEP),
        "g_sweep_values": list(G_SWEEP_VALUES),
        "controles_principales": {
            "rows": [
                {
                    "seed": row["seed"],
                    "real": row["real"],
                    "shuffle": row["shuffle"],
                    "off": row["off"],
                    "invert": row["invert"],
                }
                for row in ctrl_rows
            ],
            "n_valid_seeds": n_valid,
            "valid_seed_ids": valid_seeds,
            "wins": wins,
            "rate": rate,
            "verdict": verdict,
            "rate_off_quiet": (sum(off_quiet) / len(off_quiet)) if off_quiet else None,
            "rate_invert_not_negative": (sum(invert_not_neg) / len(invert_not_neg)) if invert_not_neg else None,
        },
        "sweep_G": {"rows": g_rows},
        "elapsed_s": elapsed,
    }

    path = OUT / "trackA_01_segregacion_result.json"
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nJSON -> {path}")
    print(f"elapsed = {elapsed:.1f}s")
    print(f"\n=== VEREDICTO GLOBAL: {verdict} (rate={rate:.2f}, n_valid={n_valid}) ===")


if __name__ == "__main__":
    main()
