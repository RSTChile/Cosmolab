#!/usr/bin/env python3
"""
ETAPA 7 — TRACK D: masa dinámica por recuperación cinemática tipo Kepler.

Pre-registro: PROTOCOLO_TRACKD_KEPLER_PREREGISTRO.md (ANTES de producción).

Idea (Alexis): si dos cuerpos están en interacción ligada, se puede INFERIR su masa
combinada a partir puramente de la CINEMÁTICA (separación r, velocidad relativa v) vía
M_dyn = v²·r / G_GRAV — SIN mirar nunca cuántos pasos llevan juntos ni con quién comparten
grupo. Canal totalmente distinto del linaje (co_member_score / n_long_co_pairs /
fusion_events NO existen en este archivo — no solo no se usan, no se calculan).

Motor de campo/átomos (medium_norm, weighted_cut, components_strict, match_persist,
toroidal_delta) es una copia del motor v6 (mismas constantes E0-E3), porque ese motor no
se toca (regla dura: no editar v1-v6). El N-body (nbody_step) se extiende para devolver
TODOS los pares dentro de FORCE_CUTOFF con su r (no solo los de GROUP_LINK_R, que es lo
que v6 usaba para co-membresía/grupos — aquí no hay grupos ni co-membresía en absoluto).

Posiciones: se registran EN CADA PASO de la ventana E4 (no cada 20, que es solo el
muestreo del log `hist` legible) — necesario para diferenciar posición→velocidad.

Contrato Alexis: E0–E3 sin masa; Track D solo mide dentro de la ventana E4. Anti-Shannon.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[2] / "results" / "etapa7_trackD_kepler"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constantes E0-E3 (idénticas a v6 — mecánica de campo/átomos, no se cambia)
# ---------------------------------------------------------------------------
K_MIN, K_MAX = 4, 14
F_CORE_MIN, F_CORE_MAX = 0.15, 0.75
COHESION_MIN, COHESION_MAX = 1.2, 6.5
PERSIST_STEPS = 4
VEV_POST_MIN = 0.10
PHI_CORE_THR = 0.35
CENTROID_TOL = 2.5

# ---------------------------------------------------------------------------
# Constantes E4 N-body (idénticas a v6)
# ---------------------------------------------------------------------------
SOFTENING = 1.2
DT_NB = 0.35
FORCE_CUTOFF = 8.0

# ---------------------------------------------------------------------------
# Track D — pre-registrado en PROTOCOLO_TRACKD_KEPLER_PREREGISTRO.md
# ---------------------------------------------------------------------------
PAIR_MIN_STEPS = 5
CV_RATIO_MIN = 1.15
RHO_REAL_MIN = 0.25
RHO_GAP_MIN = 0.15
SEED_RATE_PASS = 0.55
OFF_NULL_CLEAN_FRAC = 0.90
PROTOCOL_ID = "TRACKD_KEPLER_2026-07-23"


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
    Y0: float = 0.3


# ---------------------------------------------------------------------------
# Motor de campo (copia v6, sin cambios de comportamiento)
# ---------------------------------------------------------------------------
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
                    "v_phi": sum_abs / k if k else 0.0,
                    "n_core": n_core,
                    "f_core": f_core,
                    "cohes": float(cohes),
                    "cy": cy,
                    "cx": cx,
                    "nodes": nodes,
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
                {
                    "cy": at["cy"],
                    "cx": at["cx"],
                    "age": tr["age"] + 1,
                    "mass": at["mass_proxy"],
                    "k": at["k"],
                    "id": tid,
                }
            )
        else:
            max_id += 1
            new_tracks.append(
                {
                    "cy": at["cy"],
                    "cx": at["cx"],
                    "age": 1,
                    "mass": at["mass_proxy"],
                    "k": at["k"],
                    "id": max_id,
                }
            )
    n_stable = sum(1 for tr in new_tracks if tr["age"] >= PERSIST_STEPS)
    return new_tracks, n_stable


def toroidal_delta(a, b, L):
    d = b - a
    if d > L / 2:
        d -= L
    if d < -L / 2:
        d += L
    return d


def nbody_step_ext(pos, masses, ids, G, L, mode, perm=None):
    """
    Igual física que v6 nbody_step, pero devuelve TODOS los pares en FORCE_CUTOFF
    (force_pairs: frozenset(id_i,id_j) -> r), no solo los de GROUP_LINK_R.
    No hay grupos ni co-membresía en este archivo.
    """
    N = len(masses)
    if N == 0:
        return pos, 0.0, 0.0, 0, {}
    if mode == "off" or G <= 0:
        return pos, 0.0, 0.0, 0, {}

    acc = np.zeros_like(pos)
    E_bind = 0.0
    pair_r = []
    force_pairs = {}
    sign = -1.0 if mode == "invert" else 1.0

    src_mass = masses.copy()
    if mode == "shuffle" and perm is not None:
        src_mass = masses[perm]

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
            fy = dy / r_soft
            fx = dx / r_soft
            acc[i, 0] += sign * strength * fy
            acc[i, 1] += sign * strength * fx
            if i < j:
                E_bind += -G * masses[i] * src_mass[j] / r_soft
                pair_r.append(r)
                key = frozenset((int(ids[i]), int(ids[j])))
                if len(key) == 2:
                    force_pairs[key] = r

    pos = pos + DT_NB * acc
    pos[:, 0] = np.mod(pos[:, 0], L)
    pos[:, 1] = np.mod(pos[:, 1], L)
    mean_r = float(np.mean(pair_r)) if pair_r else 0.0
    n_pairs = len(force_pairs)
    return pos, float(E_bind), mean_r, n_pairs, force_pairs


# ---------------------------------------------------------------------------
# Estadística sin scipy (Spearman = Pearson sobre rangos, empates=rango medio)
# ---------------------------------------------------------------------------
def _rankdata(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sorted_x = x[order]
    i = 0
    n = len(x)
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0  # 1-indexed average rank
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or len(y) < 2:
        return None
    rx = _rankdata(x)
    ry = _rankdata(y)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


# ---------------------------------------------------------------------------
# Simulación Track D
# ---------------------------------------------------------------------------
def simulate_trackD(p: P) -> dict:
    rng = np.random.default_rng(p.seed)
    L = p.L
    phi = np.ones((L, L)) + 0.2 * rng.normal(size=(L, L))
    Phi = 0.06 * rng.normal(size=(L, L))
    ar = np.ones((L, L), dtype=bool)
    ad = np.ones((L, L), dtype=bool)

    tracks = []
    grav_start = int(p.GRAV_START_FRAC * p.pasos)

    # --- estado kinemático puro (Track D) ---
    atom_pos: dict[int, np.ndarray] = {}
    id_massproxy_samples: dict[int, list] = defaultdict(list)
    pair_series: dict[frozenset, list] = defaultdict(list)  # M_dyn_t
    pair_prev_rel: dict[frozenset, tuple] = {}
    prev_ids_set: frozenset = frozenset()
    nb_perm = None

    e4_steps = 0
    n_atoms_e4_max = 0

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
                np.roll(Phi, -1, 1)
                + np.roll(Phi, 1, 1)
                + np.roll(Phi, -1, 0)
                + np.roll(Phi, 1, 0)
                - 4 * Phi
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

        if e4_window and len(stable_atoms) >= 2:
            e4_steps += 1
            ids_now = [int(t["id"]) for t in stable_atoms]
            n_atoms_e4_max = max(n_atoms_e4_max, len(ids_now))

            for t in stable_atoms:
                tid = int(t["id"])
                if tid not in atom_pos:
                    atom_pos[tid] = np.array([t["cy"], t["cx"]], dtype=float)
                id_massproxy_samples[tid].append(float(t["mass"]))

            pos_arr = np.array([atom_pos[tid] for tid in ids_now], dtype=float)
            masses_arr = np.array([t["mass"] for t in stable_atoms], dtype=float)
            ids_arr = np.array(ids_now, dtype=int)

            ids_set = frozenset(ids_now)
            if p.grav_mode == "shuffle":
                if ids_set != prev_ids_set or nb_perm is None or len(nb_perm) != len(ids_now):
                    nb_perm = rng.permutation(len(ids_now))
            else:
                nb_perm = None
            prev_ids_set = ids_set

            new_pos, E_bind, mean_r, n_pairs, force_pairs = nbody_step_ext(
                pos_arr, masses_arr, ids_arr, p.G_GRAV, L, p.grav_mode, perm=nb_perm
            )

            for i, tid in enumerate(ids_now):
                atom_pos[tid] = new_pos[i]
            # escribir de vuelta a tracks para que match_persist del próximo paso
            # re-asocie el mismo átomo físico (igual truco que v6)
            id_to_newpos = {tid: new_pos[i] for i, tid in enumerate(ids_now)}
            for t in stable_atoms:
                tid = int(t["id"])
                if tid in id_to_newpos:
                    t["cy"], t["cx"] = float(id_to_newpos[tid][0]), float(id_to_newpos[tid][1])

            # --- cinemática pura: r, v, M_dyn por par en FORCE_CUTOFF ---
            cur_rel = {}
            for key, _r in force_pairs.items():
                a_id, b_id = sorted(key)  # orden estable (a menor, b mayor)
                pa, pb = atom_pos[a_id], atom_pos[b_id]
                dy = toroidal_delta(pb[0], pa[0], L)
                dx = toroidal_delta(pb[1], pa[1], L)
                r_t = float(np.hypot(dy, dx))
                cur_rel[key] = (dy, dx, r_t)
                if key in pair_prev_rel and p.G_GRAV > 0:
                    dy_prev, dx_prev, r_prev = pair_prev_rel[key]
                    ddy = toroidal_delta(dy_prev, dy, L)
                    ddx = toroidal_delta(dx_prev, dx, L)
                    v_t = float(np.hypot(ddy, ddx)) / DT_NB
                    r_mid = 0.5 * (r_t + r_prev)
                    m_dyn = (v_t**2) * r_mid / p.G_GRAV
                    pair_series[key].append(m_dyn)
            pair_prev_rel = cur_rel

    # --- agregados por par ---
    qualifying = {}
    for key, series in pair_series.items():
        if len(series) < PAIR_MIN_STEPS:
            continue
        arr = np.asarray(series, dtype=float)
        mean_m = float(np.mean(arr))
        std_m = float(np.std(arr))
        cv = (std_m / mean_m) if mean_m > 1e-9 else None
        a_id, b_id = sorted(key)
        mp_a = float(np.mean(id_massproxy_samples[a_id])) if id_massproxy_samples[a_id] else 0.0
        mp_b = float(np.mean(id_massproxy_samples[b_id])) if id_massproxy_samples[b_id] else 0.0
        qualifying[key] = {
            "n_steps": len(series),
            "mean_M": mean_m,
            "std_M": std_m,
            "cv": cv,
            "mass_proxy_pair": mp_a + mp_b,
        }

    cvs = [v["cv"] for v in qualifying.values() if v["cv"] is not None]
    cv_med = float(np.median(cvs)) if len(cvs) >= 3 else None

    rho_mass = None
    if len(qualifying) >= 5:
        means = [v["mean_M"] for v in qualifying.values()]
        mps = [v["mass_proxy_pair"] for v in qualifying.values()]
        rho_mass = spearman(means, mps)

    return {
        "params": asdict(p),
        "n_qualifying_pairs": len(qualifying),
        "n_pairs_any": len(pair_series),
        "cv_med": cv_med,
        "rho_mass": rho_mass,
        "e4_steps": int(e4_steps),
        "n_atoms_e4_max": int(n_atoms_e4_max),
        "pairs": {
            "|".join(map(str, sorted(k))): v for k, v in qualifying.items()
        },
    }


def run_seed_modes(seed: int, G: float, pasos: int = 400) -> dict:
    modes = ("real", "off", "shuffle", "invert")
    outs = {}
    for m in modes:
        p = P(seed=seed, pasos=pasos, G_GRAV=(0.0 if m == "off" else G), grav_mode=m)
        outs[m] = simulate_trackD(p)

    r, o, s, inv = outs["real"], outs["off"], outs["shuffle"], outs["invert"]

    def get(d, k):
        return d[k]

    t1 = None
    if r["cv_med"] is not None and s["cv_med"] is not None and r["cv_med"] > 1e-12:
        t1 = (s["cv_med"] / r["cv_med"]) >= CV_RATIO_MIN
    t2 = None
    if r["rho_mass"] is not None:
        gap = r["rho_mass"] - (s["rho_mass"] if s["rho_mass"] is not None else 0.0)
        t2 = (r["rho_mass"] >= RHO_REAL_MIN) and (gap >= RHO_GAP_MIN)

    pass_d = bool(t1) or bool(t2)
    off_zero_pairs = o["n_qualifying_pairs"] == 0 and o["n_pairs_any"] == 0

    return {
        "seed": seed,
        "G": G,
        "pasos": pasos,
        "T1_stability": t1,
        "T2_correlation": t2,
        "PASS_D": pass_d,
        "off_zero_pairs": off_zero_pairs,
        "real": {
            "n_qualifying_pairs": r["n_qualifying_pairs"],
            "cv_med": r["cv_med"],
            "rho_mass": r["rho_mass"],
            "e4_steps": r["e4_steps"],
        },
        "shuffle": {
            "n_qualifying_pairs": s["n_qualifying_pairs"],
            "cv_med": s["cv_med"],
            "rho_mass": s["rho_mass"],
        },
        "off": {
            "n_qualifying_pairs": o["n_qualifying_pairs"],
            "n_pairs_any": o["n_pairs_any"],
        },
        "invert": {
            "n_qualifying_pairs": inv["n_qualifying_pairs"],
            "cv_med": inv["cv_med"],
            "rho_mass": inv["rho_mass"],
        },
    }


SEEDS_STD = (7, 42, 99, 777, 2025, 3141, 8191, 99991, 12345, 54321)
SEEDS_G_SWEEP = (2025, 42, 777, 3141)


def main(smoke: bool = False):
    t0 = time.time()
    print(f"=== TRACK D — Kepler cinemático ({PROTOCOL_ID}) smoke={smoke} ===\n")

    if smoke:
        seeds = (7, 42, 99, 777)
        pasos = 200
        g_sweep = (0.10, 0.20, 0.30)
        out_name = "trackD_smoke_result.json"
    else:
        seeds = SEEDS_STD
        pasos = 400
        g_sweep = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
        out_name = "trackD_production_result.json"

    print("--- control principal G=0.20 ---")
    ctrl = []
    for sd in seeds:
        row = run_seed_modes(sd, G=0.20, pasos=pasos)
        ctrl.append(row)
        print(
            f"  seed={sd:5d} PASS_D={row['PASS_D']} T1={row['T1_stability']} T2={row['T2_correlation']} "
            f"nqualR={row['real']['n_qualifying_pairs']} cvR={row['real']['cv_med']} "
            f"cvS={row['shuffle']['cv_med']} rhoR={row['real']['rho_mass']} rhoS={row['shuffle']['rho_mass']} "
            f"off_zero={row['off_zero_pairs']}"
        )

    rate_d = sum(1 for c in ctrl if c["PASS_D"]) / len(ctrl)
    frac_off_zero = sum(1 for c in ctrl if c["off_zero_pairs"]) / len(ctrl)
    n_seeds_with_real_pairs = sum(1 for c in ctrl if c["real"]["n_qualifying_pairs"] > 0)

    print("\n--- barrido G ---")
    g_rows = []
    for G in g_sweep:
        for sd in SEEDS_G_SWEEP:
            row = run_seed_modes(int(sd), G=float(G), pasos=pasos)
            g_rows.append(row)
        sub = [r for r in g_rows if abs(r["G"] - G) < 1e-9]
        rate_g = np.mean([r["PASS_D"] for r in sub])
        print(f"  G={G:.2f} PASS_D_rate={rate_g:.2f}")

    if not frac_off_zero >= OFF_NULL_CLEAN_FRAC:
        verdict = "INCONCLUSO_OFF_NULL_SUCIO"
    elif n_seeds_with_real_pairs < 5:
        verdict = "INCONCLUSO_DATO_INSUFICIENTE"
    elif rate_d >= SEED_RATE_PASS:
        verdict = "PASS"
    elif rate_d >= 0.30:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    synthesis = {
        "global_verdict": verdict,
        "protocol_id": PROTOCOL_ID,
        "smoke": smoke,
        "pasos": pasos,
        "n_seeds": len(seeds),
        "rate_D": rate_d,
        "frac_off_zero_pairs": frac_off_zero,
        "n_seeds_with_real_pairs": n_seeds_with_real_pairs,
        "thresholds": {
            "PAIR_MIN_STEPS": PAIR_MIN_STEPS,
            "CV_RATIO_MIN": CV_RATIO_MIN,
            "RHO_REAL_MIN": RHO_REAL_MIN,
            "RHO_GAP_MIN": RHO_GAP_MIN,
            "SEED_RATE_PASS": SEED_RATE_PASS,
            "OFF_NULL_CLEAN_FRAC": OFF_NULL_CLEAN_FRAC,
        },
        "elapsed_s": time.time() - t0,
        "no_lineage_ingredients": True,
        "note": "co_member_score / n_long_co_pairs / fusion_events NO se calculan en este archivo.",
    }

    out = {
        "synthesis": synthesis,
        "controls_G020": ctrl,
        "sweep_G": g_rows,
    }
    path = OUT / out_name
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nJSON -> {path}")
    print("\n=== GLOBAL ===", verdict)
    print(json.dumps(synthesis, indent=2))


if __name__ == "__main__":
    import sys

    main(smoke=("--smoke" in sys.argv))
