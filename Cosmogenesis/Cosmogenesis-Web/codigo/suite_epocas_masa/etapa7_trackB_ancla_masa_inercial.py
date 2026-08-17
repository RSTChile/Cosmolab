#!/usr/bin/env python3
"""
ETAPA 7 — TRACK B: masa inercial operacional por ancla gravitatoria externa.

Pre-registro: PROTOCOLO_TRACKB_ANCLA_PREREGISTRO.md (ANTES de producción y de smoke).

Idea (mandato de la misión): en física real la masa inercial se mide operacionalmente
aplicando una fuerza CONOCIDA e independiente y midiendo la aceleración resultante
(m = F/a). Aquí: un ANCLA gravitatoria externa de masa M_anchor fija y conocida, en
posiciones fijas del grid, que NO es un átomo — no participa de match_persist, no entra a
ningún cálculo de grupos/co-membresía/fusión, no tiene id de track, nunca se mueve.

`co_member_score`, `n_long_co_pairs`, `fusion_events` NO se calculan en este script (no
solo no se usan: no existen en el código). El único ingrediente de identidad es el id de
track de cada átomo (continuidad de objeto, no co-membresía ni linaje).

Motor de campo/átomos (medium_norm, weighted_cut, components_strict, match_persist,
toroidal_delta) es una copia literal de v6 (mecánica E0-E3, sin cambios de comportamiento)
porque ese motor no se toca (regla dura: no editar v1-v6).

La respuesta al ancla (F=ma, dividiendo por mass_proxy) es código NUEVO propio de Track B,
NO una llamada a nbody_step de v6 (que usa strength=G*m_i*m_j/r^2 directamente como
aceleración, SIN dividir por masa — convención propia de la fuerza MUTUA entre átomos que
no tocamos). Aquí SÍ se divide por mass_proxy porque el mandato es probar F=ma con una
fuerza que no dependa de la masa del receptor. Ver PROTOCOLO_TRACKB... para la advertencia
de circularidad registrada de antemano (REAL es cuasi-tautológico consigo mismo; el peso
probatorio real está en REAL vs SHUFFLE).

Contrato Alexis: E0-E3 sin masa; Track B solo mide dentro de la ventana E4, sobre un
snapshot congelado de átomos estables. Anti-Shannon.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[2] / "results" / "etapa7_trackB_ancla"
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
# Constantes de fuerza (idénticas a nbody_step de v6 — misma ley, mismo G/softening/cutoff)
# ---------------------------------------------------------------------------
SOFTENING = 1.2
FORCE_CUTOFF = 8.0

# ---------------------------------------------------------------------------
# Track B — pre-registrado en PROTOCOLO_TRACKB_ANCLA_PREREGISTRO.md
# ---------------------------------------------------------------------------
R2_MIN = 0.30
GAP_MIN = 0.15
RATE_PASS = 0.55
OFF_NULL_CLEAN_FRAC = 0.99
N_SHUFFLE_REPEATS = 8
MIN_ATOMS_SNAPSHOT = 3
PROTOCOL_ID = "TRACKB_ANCLA_2026-07-23"

ANCHOR_POSITIONS_PROD = [(14.0, 14.0), (7.0, 7.0), (7.0, 21.0), (21.0, 7.0), (21.0, 21.0)]
ANCHOR_POSITIONS_SMOKE = [(14.0, 14.0), (7.0, 7.0)]
M_ANCHOR_SWEEP_PROD = (5.0, 10.0, 20.0, 40.0, 80.0)
M_ANCHOR_SWEEP_SMOKE = (10.0, 40.0)


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
    Y0: float = 0.3


# ---------------------------------------------------------------------------
# Motor de campo (copia literal de v6, sin cambios de comportamiento)
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


# ---------------------------------------------------------------------------
# Snapshot E0->E3->E4-entry (una sola corrida de campo por semilla)
# ---------------------------------------------------------------------------
def snapshot_stable_atoms(p: P, min_atoms: int = 2):
    """Corre el motor de campo hasta el primer paso de la ventana E4 con
    >= min_atoms átomos estables. Devuelve (step, atoms, L) o (None, [], L).
    No sigue evolucionando el campo mas alla de ese punto: el ancla se aplica
    sobre este snapshot congelado (aislamiento deliberado, ver protocolo)."""
    rng = np.random.default_rng(p.seed)
    L = p.L
    phi = np.ones((L, L)) + 0.2 * rng.normal(size=(L, L))
    Phi = 0.06 * rng.normal(size=(L, L))
    ar = np.ones((L, L), dtype=bool)
    ad = np.ones((L, L), dtype=bool)
    tracks = []
    grav_start = int(p.GRAV_START_FRAC * p.pasos)

    for step in range(p.pasos):
        tg = step / max(p.pasos - 1, 1)
        Tnorm = float(np.exp(-p.H_EXP * tg))
        a = float(np.exp(p.H_EXP * tg))
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

        if e4_window and len(stable_atoms) >= min_atoms:
            return step, stable_atoms, L

    return None, [], L


# ---------------------------------------------------------------------------
# Respuesta al ancla (código NUEVO, F=ma explícito, no reusa nbody_step)
# ---------------------------------------------------------------------------
def anchor_response(atoms, L, anchor_pos, M_anchor, G_GRAV, mode, perm=None):
    """Devuelve, por átomo del snapshot: (id, r, mass_proxy, a_obs, in_range, M_anchor,
    anchor_pos). F_known(r) = G*M_anchor/(r^2+SOFTENING^2) -- NO depende de mass_proxy del
    átomo. REAL/INVERT dividen por mass_proxy propio; SHUFFLE por mass_proxy de perm[i]."""
    ay, ax = anchor_pos
    out = []
    for i, at in enumerate(atoms):
        dy = toroidal_delta(ay, at["cy"], L)
        dx = toroidal_delta(ax, at["cx"], L)
        r = float(np.hypot(dy, dx))
        in_range = r <= FORCE_CUTOFF
        base = {"id": at["id"], "r": r, "mass_proxy": at["mass"], "in_range": in_range,
                "M_anchor": M_anchor, "anchor_pos": anchor_pos}
        if not in_range or M_anchor <= 0:
            out.append({**base, "a_obs": 0.0})
            continue
        F_known = G_GRAV * M_anchor / (r * r + SOFTENING**2)
        if mode == "real":
            divisor = at["mass"]
        elif mode == "invert":
            divisor = at["mass"]
        elif mode == "shuffle":
            j = int(perm[i])
            divisor = atoms[j]["mass"]
        else:
            raise ValueError(mode)
        a = F_known / max(divisor, 1e-9)
        if mode == "invert":
            a = -a
        out.append({**base, "a_obs": a})
    return out


# ---------------------------------------------------------------------------
# Regresión OLS sin scipy
# ---------------------------------------------------------------------------
def ols_fit(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) < 1e-12:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else None
    return {"slope": float(slope), "intercept": float(intercept), "r2": r2, "n": len(x)}


def loglog_fit_mass_r(records):
    """log(a_obs) ~ beta_m*log(mass_proxy) + beta_r*log(r) + const, via lstsq. Diagnóstico."""
    rows = [r for r in records if r["a_obs"] > 1e-12 and r["mass_proxy"] > 1e-9 and r["r"] > 1e-9]
    if len(rows) < 5:
        return None
    y = np.log(np.array([r["a_obs"] for r in rows]))
    xm = np.log(np.array([r["mass_proxy"] for r in rows]))
    xr = np.log(np.array([r["r"] for r in rows]))
    A = np.column_stack([xm, xr, np.ones_like(xm)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    beta_m, beta_r, const = coef
    y_pred = A @ coef
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else None
    return {"beta_mass": float(beta_m), "beta_r": float(beta_r), "const": float(const), "r2": r2, "n": len(rows)}


# ---------------------------------------------------------------------------
# Corrida completa por semilla
# ---------------------------------------------------------------------------
def run_seed(seed: int, anchor_positions, m_anchor_sweep, n_shuffle_repeats: int, G_GRAV: float = 0.20) -> dict:
    p = P(seed=seed, G_GRAV=G_GRAV)
    step, atoms, L = snapshot_stable_atoms(p, min_atoms=2)
    if not atoms or len(atoms) < MIN_ATOMS_SNAPSHOT:
        return {
            "seed": seed,
            "insufficient_atoms": True,
            "n_atoms_snapshot": len(atoms),
            "snapshot_step": step,
        }
    rng = np.random.default_rng(seed * 7919 + 3)

    recs_real, recs_shuffle, recs_off, recs_invert = [], [], [], []
    for anchor_pos in anchor_positions:
        for M_anchor in m_anchor_sweep:
            out_real = anchor_response(atoms, L, anchor_pos, M_anchor, p.G_GRAV, "real")
            out_invert = anchor_response(atoms, L, anchor_pos, M_anchor, p.G_GRAV, "invert")
            out_off = anchor_response(atoms, L, anchor_pos, 0.0, p.G_GRAV, "real")
            recs_real.extend([r for r in out_real if r["in_range"]])
            recs_invert.extend(out_invert)
            recs_off.extend(out_off)

            shuffle_batches = []
            for _ in range(n_shuffle_repeats):
                perm = rng.permutation(len(atoms))
                out_s = anchor_response(atoms, L, anchor_pos, M_anchor, p.G_GRAV, "shuffle", perm=perm)
                shuffle_batches.append([r for r in out_s if r["in_range"]])
            # promedio de a_obs sobre las repeticiones, por atom-id, para esta (anchor_pos,M_anchor)
            by_id = defaultdict(list)
            for batch in shuffle_batches:
                for r in batch:
                    by_id[r["id"]].append(r)
            for aid, rows in by_id.items():
                mean_a = float(np.mean([r["a_obs"] for r in rows]))
                recs_shuffle.append({
                    "id": aid, "r": rows[0]["r"], "mass_proxy": rows[0]["mass_proxy"],
                    "a_obs": mean_a, "in_range": True,
                })

    off_zero = all(r["a_obs"] == 0.0 for r in recs_off)

    def build_xy(records):
        xs, ys = [], []
        for r in records:
            if r["a_obs"] > 1e-12:
                xs.append(r["mass_proxy"])
                ys.append(1.0 / r["a_obs"])
        return xs, ys

    x_real, y_real = build_xy(recs_real)
    x_shuf, y_shuf = build_xy(recs_shuffle)
    fit_real = ols_fit(x_real, y_real)
    fit_shuf = ols_fit(x_shuf, y_shuf)
    diag_real = loglog_fit_mass_r(recs_real)
    diag_shuf = loglog_fit_mass_r(recs_shuffle)

    t1 = bool(fit_real is not None and fit_real["slope"] > 0)
    t2 = None
    if fit_real is not None and fit_real["r2"] is not None:
        r2_real = fit_real["r2"]
        r2_shuf = fit_shuf["r2"] if (fit_shuf is not None and fit_shuf["r2"] is not None) else 0.0
        t2 = bool(r2_real >= R2_MIN and (r2_real - r2_shuf) >= GAP_MIN)
    pass_b = bool(t1 and t2) if t2 is not None else False

    # sanity de signo INVERT vs REAL (mismo atomo/ancla/M_anchor -> producto punto opuesto)
    invert_sign_ok = None
    if recs_real and recs_invert:
        # comparar por (id, M_anchor, anchor_pos) -- misma config física, sentido opuesto
        def key_of(r):
            return (r["id"], r["M_anchor"], r["anchor_pos"])

        real_map = {key_of(r): r["a_obs"] for r in recs_real}
        inv_map = {key_of(r): r["a_obs"] for r in recs_invert if r["in_range"]}
        signs_ok = []
        for key, a_r in list(real_map.items())[:200]:
            a_i = inv_map.get(key)
            if a_i is not None:
                signs_ok.append(abs(a_i - (-a_r)) < 1e-9 * max(abs(a_r), 1))
        invert_sign_ok = bool(np.mean(signs_ok) > 0.99) if signs_ok else None

    return {
        "seed": seed,
        "insufficient_atoms": False,
        "n_atoms_snapshot": len(atoms),
        "snapshot_step": step,
        "n_in_range_real": len(recs_real),
        "n_in_range_shuffle_avg": len(recs_shuffle),
        "off_zero": off_zero,
        "fit_real": fit_real,
        "fit_shuffle": fit_shuf,
        "diag_loglog_real": diag_real,
        "diag_loglog_shuffle": diag_shuf,
        "T1_slope_positive": t1,
        "T2_r2_and_gap": t2,
        "PASS_B": pass_b,
        "invert_sign_ok": invert_sign_ok,
    }


SEEDS_STD = (7, 42, 99, 777, 2025, 3141, 8191, 99991, 12345, 54321)


def main(smoke: bool = False):
    t0 = time.time()
    started_at = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"=== TRACK B — ancla gravitatoria externa ({PROTOCOL_ID}) smoke={smoke} inicio={started_at} ===\n")

    if smoke:
        seeds = (7, 42, 99, 2025)
        anchor_positions = ANCHOR_POSITIONS_SMOKE
        m_sweep = M_ANCHOR_SWEEP_SMOKE
        n_repeats = 4
        out_name = "trackB_smoke_result.json"
    else:
        seeds = SEEDS_STD
        anchor_positions = ANCHOR_POSITIONS_PROD
        m_sweep = M_ANCHOR_SWEEP_PROD
        n_repeats = N_SHUFFLE_REPEATS
        out_name = "trackB_production_result.json"

    rows = []
    for sd in seeds:
        row = run_seed(sd, anchor_positions, m_sweep, n_repeats, G_GRAV=0.20)
        rows.append(row)
        if row.get("insufficient_atoms"):
            print(f"  seed={sd:5d} INSUFFICIENT_ATOMS n_atoms={row['n_atoms_snapshot']}")
            continue
        fr = row["fit_real"]
        fs = row["fit_shuffle"]
        slope_str = f"{fr['slope']:.4g}" if fr else "None"
        print(
            f"  seed={sd:5d} n_atoms={row['n_atoms_snapshot']:3d} step={row['snapshot_step']} "
            f"PASS_B={row['PASS_B']} T1={row['T1_slope_positive']} T2={row['T2_r2_and_gap']} "
            f"R2_real={fr['r2'] if fr else None} R2_shuf={fs['r2'] if fs else None} "
            f"slope_real={slope_str} off_zero={row['off_zero']} "
            f"invert_ok={row['invert_sign_ok']}"
        )

    valid = [r for r in rows if not r.get("insufficient_atoms")]
    n_valid = len(valid)
    rate_b = (sum(1 for r in valid if r["PASS_B"]) / n_valid) if n_valid else 0.0
    frac_off_zero = (sum(1 for r in valid if r["off_zero"]) / n_valid) if n_valid else 0.0
    n_invert_ok = sum(1 for r in valid if r.get("invert_sign_ok"))

    if frac_off_zero < OFF_NULL_CLEAN_FRAC or n_valid < 5:
        verdict = "INCONCLUSO"
    elif rate_b >= RATE_PASS:
        verdict = "PASS"
    elif rate_b >= 0.30:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    finished_at = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    synthesis = {
        "global_verdict": verdict,
        "protocol_id": PROTOCOL_ID,
        "smoke": smoke,
        "started_at": started_at,
        "finished_at": finished_at,
        "n_seeds": len(seeds),
        "n_seeds_valid": n_valid,
        "rate_B": rate_b,
        "frac_off_zero": frac_off_zero,
        "n_invert_sign_ok": n_invert_ok,
        "thresholds": {
            "R2_MIN": R2_MIN,
            "GAP_MIN": GAP_MIN,
            "RATE_PASS": RATE_PASS,
            "OFF_NULL_CLEAN_FRAC": OFF_NULL_CLEAN_FRAC,
            "N_SHUFFLE_REPEATS": n_repeats,
            "MIN_ATOMS_SNAPSHOT": MIN_ATOMS_SNAPSHOT,
        },
        "anchor_positions": anchor_positions,
        "m_anchor_sweep": list(m_sweep),
        "elapsed_s": time.time() - t0,
        "no_lineage_ingredients": True,
        "note": "co_member_score / n_long_co_pairs / fusion_events NO se calculan en este archivo.",
    }

    out = {"synthesis": synthesis, "rows": rows}
    path = OUT / out_name
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nJSON -> {path}")
    print("\n=== GLOBAL ===", verdict)
    print(json.dumps(synthesis, indent=2, default=str))


if __name__ == "__main__":
    import sys

    main(smoke=("--smoke" in sys.argv))
