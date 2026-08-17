#!/usr/bin/env python3
"""
SUITE ÉPOCAS MASA v3 — gravedad ATÓMICA (N-body entre centroides H)

Contrato Alexis: masa solo E4, tras átomo, con gravedad sobre H densificado.
v2 hallazgo: campo suave + SHUFFLE de pozos densifica ≈ REAL → no prueba
gravedad-sobre-H. v3 responde con dinámica entre átomos.

E3: mismos criterios estrictos (núcleo+halo+cohesión+persistencia).

E4 N-body:
  - Cada átomo estricto estable = partícula en (cy, cx) con "carga" = k o sum_phi
  - Fuerza REAL: atracción par a par ∝ m_i m_j / r^2 (suave), dirección a lo largo
    del vector entre centroides (toroidal).
  - Actualiza posiciones de centroides; ρ_H se reconstruye depositando núcleos
    en la malla cerca del centroide (no un campo barajable genérico).
  - mass_obs solo en modo REAL, y solo si hay densificación de grupos + enlace.

Nulos:
  OFF:      G=0
  SHUFFLE:  se permutan las *etiquetas/identidades* en el grafo de atracción
            (quién es atraído por la masa de quién) — misma lista de masas,
            partners barajados cada paso o fijados al inicio del tramo E4
  INVERT:   fuerzas repulsivas

Métricas (no Rm/v3 como masa):
  - mass_obs: f(N_groups, dens_group, binding) solo REAL
  - R_gyr: radio de giración medio de grupos
  - E_bind: energía de enlace aproximada (suma -G m_i m_j / r para pares cercanos)
  - dens_enhance: max ρ_H / mean
  - n_atoms_stable, n_groups

Anti-Shannon: sin 1/1836; barridos G y seeds.
"""
from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[2] / "results" / "suite_epocas_masa_v3"
OUT.mkdir(parents=True, exist_ok=True)

# E3 strict
K_MIN, K_MAX = 4, 14
F_CORE_MIN, F_CORE_MAX = 0.15, 0.75
COHESION_MIN, COHESION_MAX = 1.2, 6.5
PERSIST_STEPS = 4
VEV_POST_MIN = 0.10
PHI_CORE_THR = 0.35
CENTROID_TOL = 2.5

# E4 N-body
SOFTENING = 1.2
DT_NB = 0.35
GROUP_LINK_R = 4.5          # radio para formar grupo / enlace
MASS_REAL_MIN = 0.3
BIND_REAL_MIN = 0.05        # |E_bind| mínimo en REAL
GYR_SHRINK_MAX = 0.92       # R_gyr_final / R_gyr_start < esto en REAL (compacta)
DENS_MIN = 1.2
# REAL debe superar nulos
BIND_VS_SHUFFLE_MIN = 1.25  # |E_bind|_REAL / max(|E_bind|_SHUF, eps)
RATE_PASS = 0.55


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
    report_leak: bool = True


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


def match_persist(atoms_now, tracks):
    used = set()
    new_tracks = []
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
            new_tracks.append(
                {
                    "cy": at["cy"],
                    "cx": at["cx"],
                    "age": tr["age"] + 1,
                    "mass": at["mass_proxy"],
                    "k": at["k"],
                    "id": tr.get("id", len(new_tracks)),
                }
            )
        else:
            new_tracks.append(
                {
                    "cy": at["cy"],
                    "cx": at["cx"],
                    "age": 1,
                    "mass": at["mass_proxy"],
                    "k": at["k"],
                    "id": len(tracks) + len(new_tracks) + 1000,
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


def nbody_step(pos, masses, G, L, mode, rng, perm=None):
    """
    pos: (N,2) float cy,cx
    masses: (N,)
    mode: real | off | shuffle | invert
    perm: for shuffle, maps source mass index -> which mass value pulls (fixed per E4 window)
    Returns new_pos, E_bind, mean_pair_r
    """
    N = len(masses)
    if N == 0:
        return pos, 0.0, 0.0, 0
    if mode == "off" or G <= 0:
        return pos, 0.0, 0.0, 0

    acc = np.zeros_like(pos)
    E_bind = 0.0
    pair_r = []
    sign = -1.0 if mode == "invert" else 1.0  # invert: repulsión (sign flip on force toward)

    # mass that particle j "presents" as source
    src_mass = masses.copy()
    if mode == "shuffle" and perm is not None:
        src_mass = masses[perm]

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            dy = toroidal_delta(pos[i, 0], pos[j, 0], L)
            dx = toroidal_delta(pos[i, 1], pos[j, 1], L)
            r2 = dy * dy + dx * dx + SOFTENING**2
            r = np.sqrt(r2)
            # atracción de i hacia j: proporcional a m_i * m_src_j
            # REAL: m_src_j = m_j (gravedad de ese átomo H)
            # SHUFFLE: m_src_j = m_{perm[j]} — identidad de fuente barajada
            strength = G * masses[i] * src_mass[j] / r2
            # dirección unitaria hacia j
            fy = dy / r
            fx = dx / r
            # atracción: mover i hacia j => +direction * strength
            # invert: alejar
            acc[i, 0] += sign * strength * fy
            acc[i, 1] += sign * strength * fx
            if i < j:
                E_bind += -G * masses[i] * src_mass[j] / r  # más negativo = más ligado en REAL
                pair_r.append(r)

    pos = pos + DT_NB * acc
    pos[:, 0] = np.mod(pos[:, 0], L)
    pos[:, 1] = np.mod(pos[:, 1], L)
    mean_r = float(np.mean(pair_r)) if pair_r else 0.0
    n_close = sum(1 for r in pair_r if r < GROUP_LINK_R)
    return pos, float(E_bind), mean_r, n_close


def deposit_rho(pos, masses, L):
    rho = np.zeros((L, L))
    if len(masses) == 0:
        return rho
    for (y, x), m in zip(pos, masses):
        iy, ix = int(y) % L, int(x) % L
        # depósito 3x3
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                w = m if (dy, dx) == (0, 0) else 0.25 * m
                rho[(iy + dy) % L, (ix + dx) % L] += w
    if rho.sum() > 0:
        rho *= (L * L) / rho.sum()
    return rho


def groups_and_gyr(pos, L, link_r=GROUP_LINK_R):
    """Grupos por unión de pares a distancia < link_r; R_gyr medio."""
    N = len(pos)
    if N == 0:
        return 0, 0.0, []
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
    gyrs = []
    for ids in clusters.values():
        if len(ids) < 2:
            continue
        pts = pos[ids]
        # centro toroidal approx mean
        cy, cx = float(pts[:, 0].mean()), float(pts[:, 1].mean())
        d2 = []
        for y, x in pts:
            dy = toroidal_delta(cy, y, L)
            dx = toroidal_delta(cx, x, L)
            d2.append(dy * dy + dx * dx)
        gyrs.append(np.sqrt(np.mean(d2)))
    n_groups = sum(1 for ids in clusters.values() if len(ids) >= 2)
    r_gyr = float(np.mean(gyrs)) if gyrs else float("nan")
    return n_groups, r_gyr, list(clusters.values())


def premature_leak(clusters, y0):
    m1, m3, g1, g3 = [], [], [], []
    for c in clusters:
        m = y0 * c["v_phi"] * c["sum_phi"]
        g = y0 * c["sum_phi"]
        if c["k"] == 1:
            m1.append(m)
            g1.append(g)
        if c["k"] == 3 and c["perim"] == 8:
            m3.append(m)
            g3.append(g)
    if not m1 or not m3:
        return None
    return abs(
        float(np.mean(m1) / (np.mean(m3) + 1e-30))
        - float(np.mean(g1) / (np.mean(g3) + 1e-30))
    )


def simulate(p: P) -> dict:
    rng = np.random.default_rng(p.seed)
    L = p.L
    phi = np.ones((L, L)) + 0.2 * rng.normal(size=(L, L))
    Phi = 0.06 * rng.normal(size=(L, L))
    ar = np.ones((L, L), dtype=bool)
    ad = np.ones((L, L), dtype=bool)

    tracks = []
    hist = []
    grav_start = int(p.GRAV_START_FRAC * p.pasos)

    # estado N-body
    nb_pos = None  # (N,2)
    nb_mass = None
    nb_perm = None
    r_gyr_start = None
    e_bind_acc = []

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

        # init / sync N-body from stable atoms when entering E4 or count changes
        E_bind = 0.0
        mean_r = 0.0
        n_close = 0
        n_groups = 0
        r_gyr = float("nan")
        dens_enhance = 1.0
        mass_obs = 0.0
        rho_H = np.zeros((L, L))

        if e4_window and len(stable_atoms) >= 2:
            pos = np.array([[t["cy"], t["cx"]] for t in stable_atoms], dtype=float)
            masses = np.array([t["mass"] for t in stable_atoms], dtype=float)
            if nb_pos is None or len(nb_pos) != len(pos):
                nb_pos = pos.copy()
                nb_mass = masses.copy()
                if p.grav_mode == "shuffle":
                    nb_perm = rng.permutation(len(masses))
                else:
                    nb_perm = None
                n_g0, r0, _ = groups_and_gyr(nb_pos, L)
                r_gyr_start = r0 if r0 == r0 else None  # not nan
            else:
                # re-sync masses lightly; keep evolved positions
                nb_mass = masses
                # snap count match: if same N, keep nb_pos; already handled size change above

            nb_pos, E_bind, mean_r, n_close = nbody_step(
                nb_pos,
                nb_mass,
                p.G_GRAV,
                L,
                p.grav_mode,
                rng,
                perm=nb_perm,
            )
            # write back centroids into tracks for persistence continuity
            for i, t in enumerate(stable_atoms):
                if i < len(nb_pos):
                    t["cy"], t["cx"] = float(nb_pos[i, 0]), float(nb_pos[i, 1])

            rho_H = deposit_rho(nb_pos, nb_mass, L)
            if rho_H.mean() > 0:
                dens_enhance = float(rho_H.max() / (rho_H.mean() + 1e-12))
            n_groups, r_gyr, _ = groups_and_gyr(nb_pos, L)
            e_bind_acc.append(E_bind)

            # mass_obs SOLO modo real: compactación + enlace
            if p.grav_mode == "real" and p.G_GRAV > 0:
                bind_strength = max(0.0, -E_bind)  # positivo si ligado
                gyr_factor = 1.0
                if r_gyr_start and r_gyr == r_gyr and r_gyr_start > 0:
                    gyr_factor = max(r_gyr_start / (r_gyr + 1e-6), 1.0)
                mass_obs = float(
                    bind_strength * dens_enhance * gyr_factor * max(n_groups, 0) / max(len(nb_mass), 1)
                )
            else:
                mass_obs = 0.0

        # época
        if Tnorm > p.TC:
            ep = "E0"
        elif not frozen:
            ep = "E1"
        elif n_stable < 1:
            ep = "E2"
        elif not e4_window:
            ep = "E3"
        else:
            ep = "E4" if p.grav_mode == "real" and p.G_GRAV > 0 else "E3"

        leak = premature_leak(cl, p.Y0) if p.report_leak else None

        if step % 20 == 0 or step == p.pasos - 1:
            hist.append(
                {
                    "step": step,
                    "a": a,
                    "Tnorm": Tnorm,
                    "epoch": ep,
                    "Phi_abs": float(np.mean(np.abs(Phi))),
                    "n_atoms_strict": len(atoms),
                    "n_atoms_stable": n_stable,
                    "n_groups": n_groups,
                    "E_bind": E_bind,
                    "mean_pair_r": mean_r,
                    "n_close_pairs": n_close,
                    "R_gyr": r_gyr if r_gyr == r_gyr else None,
                    "R_gyr_start": r_gyr_start,
                    "dens_enhance": dens_enhance,
                    "mass_obs": mass_obs,
                    "grav_mode": p.grav_mode,
                    "leak_sep": leak,
                }
            )

    return {
        "params": asdict(p),
        "hist": hist,
        "E_bind_mean_E4": float(np.mean(e_bind_acc)) if e_bind_acc else 0.0,
        "E_bind_min_E4": float(np.min(e_bind_acc)) if e_bind_acc else 0.0,
    }


def analyze(sim: dict) -> dict:
    hist = sim["hist"]
    p = sim["params"]
    pre = [h for h in hist if h["epoch"] in ("E0", "E1", "E2", "E3")]
    e4 = [h for h in hist if h["epoch"] == "E4"]
    late = [h for h in hist if h["step"] >= int(0.65 * p["pasos"])]

    def mx(rows, key):
        vals = [h[key] for h in rows if h.get(key) is not None]
        return float(np.max(vals)) if vals else 0.0

    def mn(rows, key):
        vals = [h[key] for h in rows if h.get(key) is not None]
        return float(np.min(vals)) if vals else 0.0

    def avg(rows, key):
        vals = [h[key] for h in rows if h.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    mass_pre = mx(pre, "mass_obs")
    mass_e4 = mx(e4, "mass_obs") if e4 else mx(late, "mass_obs")
    # E_bind más negativo = más ligado
    E_min = sim["E_bind_min_E4"]
    E_mean = sim["E_bind_mean_E4"]
    dens = mx(late, "dens_enhance")
    n_st = mx(hist, "n_atoms_stable")
    n_gr = mx(late, "n_groups")

    # R_gyr shrink
    r_start = None
    r_end = None
    for h in hist:
        if h.get("R_gyr_start") is not None:
            r_start = h["R_gyr_start"]
        if h.get("R_gyr") is not None:
            r_end = h["R_gyr"]
    gyr_ratio = (
        (r_end / r_start) if (r_start and r_end and r_start > 0) else None
    )

    e0 = [h for h in hist if h["epoch"] == "E0"]
    e0_ok = bool(e0) and (avg(e0, "Phi_abs") or 1) < 0.18
    post = [h for h in hist if h["Tnorm"] <= p["TC"]]
    e1_ok = (
        e0_ok
        and post
        and (avg(post, "Phi_abs") or 0) > VEV_POST_MIN
        and (avg(post, "Phi_abs") or 0) > 1.3 * max(avg(e0, "Phi_abs") or 0, 1e-6)
    )
    e3_ok = n_st >= 1

    mode = p["grav_mode"]
    if mode == "real":
        e4_ok = (
            mass_pre <= 1e-12
            and mass_e4 >= MASS_REAL_MIN
            and (-E_min) >= BIND_REAL_MIN
            and dens >= DENS_MIN
        )
    else:
        e4_ok = mass_e4 <= 1e-12 and mass_pre <= 1e-12

    return {
        "grav_mode": mode,
        "E0_ok": e0_ok,
        "E1_ok": e1_ok,
        "E3_ok": e3_ok,
        "E4_ok_for_mode": e4_ok,
        "mass_pre": mass_pre,
        "mass_E4": mass_e4,
        "E_bind_min": E_min,
        "E_bind_mean": E_mean,
        "bind_strength": max(0.0, -E_min),
        "dens_enhance": dens,
        "n_atoms_stable": n_st,
        "n_groups_max": n_gr,
        "gyr_ratio": gyr_ratio,
        "leak_max": mx(pre, "leak_sep"),
        "zero_mass_pre": mass_pre <= 1e-12,
    }


def run_controls(seed: int, G: float = 0.20) -> dict:
    modes = ("real", "off", "shuffle", "invert")
    outs = {}
    for m in modes:
        p = P(
            seed=seed,
            G_GRAV=(0.0 if m == "off" else G),
            grav_mode=m,
        )
        sim = simulate(p)
        outs[m] = analyze(sim)

    r, o, s, inv = outs["real"], outs["off"], outs["shuffle"], outs["invert"]
    eps = 1e-12
    bind_vs_shuf = r["bind_strength"] / max(s["bind_strength"], eps)
    bind_vs_off = r["bind_strength"] / max(o["bind_strength"], eps)
    mass_vs_shuf = r["mass_E4"] / max(s["mass_E4"], eps)
    dens_vs_shuf = r["dens_enhance"] / max(s["dens_enhance"], 1.0)

    # causalidad: masa y enlace de REAL superan nulos; mass nulos = 0
    e4_causal = (
        r["E3_ok"]
        and r["zero_mass_pre"]
        and o["mass_E4"] <= eps
        and s["mass_E4"] <= eps
        and inv["mass_E4"] <= eps
        and r["mass_E4"] >= MASS_REAL_MIN
        and r["bind_strength"] >= BIND_REAL_MIN
        and bind_vs_shuf >= BIND_VS_SHUFFLE_MIN
        and r["dens_enhance"] >= DENS_MIN
    )
    # compactación: R_gyr baja en REAL más que en SHUFFLE (si ambos definidos)
    gyr_real = r["gyr_ratio"]
    gyr_shuf = s["gyr_ratio"]
    gyr_causal = False
    if gyr_real is not None and gyr_shuf is not None:
        gyr_causal = gyr_real < gyr_shuf * 0.95  # REAL compacta más
    elif gyr_real is not None:
        gyr_causal = gyr_real < GYR_SHRINK_MAX

    return {
        "seed": seed,
        "G": G,
        "modes": outs,
        "bind_vs_shuffle": bind_vs_shuf,
        "bind_vs_off": bind_vs_off,
        "mass_vs_shuffle": mass_vs_shuf,
        "dens_vs_shuffle": dens_vs_shuf,
        "e4_causal": e4_causal,
        "gyr_causal": gyr_causal,
        "E3_ok": r["E3_ok"],
        "E0_ok": r["E0_ok"],
        "E1_ok": r["E1_ok"],
    }


def main():
    print("=== SUITE ÉPOCAS MASA v3 — N-body atómico ===\n")
    t0 = time.time()
    results = {}

    seeds = (7, 42, 99, 777, 2025, 3141, 8191, 99991, 12345, 54321)

    print("--- controles REAL/OFF/SHUFFLE/INVERT ---")
    ctrl = []
    for s in seeds:
        row = run_controls(s, G=0.20)
        ctrl.append(row)
        r = row["modes"]["real"]
        sh = row["modes"]["shuffle"]
        print(
            f"  seed={s:5d} E3={row['E3_ok']} causal={row['e4_causal']} "
            f"gyr_c={row['gyr_causal']} "
            f"mR={r['mass_E4']:.3f} mS={sh['mass_E4']:.3f} "
            f"bindR={r['bind_strength']:.3f} bindS={sh['bind_strength']:.3f} "
            f"dR={r['dens_enhance']:.2f} dS={sh['dens_enhance']:.2f} "
            f"bindR/S={row['bind_vs_shuffle']:.2f}"
        )

    results["controls"] = {
        "n": len(ctrl),
        "rate_E3": sum(c["E3_ok"] for c in ctrl) / len(ctrl),
        "rate_e4_causal": sum(c["e4_causal"] for c in ctrl) / len(ctrl),
        "rate_gyr_causal": sum(c["gyr_causal"] for c in ctrl) / len(ctrl),
        "rate_E0": sum(c["E0_ok"] for c in ctrl) / len(ctrl),
        "rate_E1": sum(c["E1_ok"] for c in ctrl) / len(ctrl),
        "mean_mass_real": float(np.mean([c["modes"]["real"]["mass_E4"] for c in ctrl])),
        "mean_mass_off": float(np.mean([c["modes"]["off"]["mass_E4"] for c in ctrl])),
        "mean_mass_shuffle": float(
            np.mean([c["modes"]["shuffle"]["mass_E4"] for c in ctrl])
        ),
        "mean_mass_invert": float(
            np.mean([c["modes"]["invert"]["mass_E4"] for c in ctrl])
        ),
        "mean_bind_real": float(
            np.mean([c["modes"]["real"]["bind_strength"] for c in ctrl])
        ),
        "mean_bind_shuffle": float(
            np.mean([c["modes"]["shuffle"]["bind_strength"] for c in ctrl])
        ),
        "mean_bind_off": float(
            np.mean([c["modes"]["off"]["bind_strength"] for c in ctrl])
        ),
        "mean_dens_real": float(
            np.mean([c["modes"]["real"]["dens_enhance"] for c in ctrl])
        ),
        "mean_dens_shuffle": float(
            np.mean([c["modes"]["shuffle"]["dens_enhance"] for c in ctrl])
        ),
        "mean_bind_vs_shuffle": float(np.mean([c["bind_vs_shuffle"] for c in ctrl])),
        "mean_leak": float(np.mean([c["modes"]["real"]["leak_max"] for c in ctrl])),
        "rows": [
            {
                "seed": c["seed"],
                "e4_causal": c["e4_causal"],
                "gyr_causal": c["gyr_causal"],
                "E3_ok": c["E3_ok"],
                "bind_vs_shuffle": c["bind_vs_shuffle"],
                "mass_real": c["modes"]["real"]["mass_E4"],
                "mass_shuffle": c["modes"]["shuffle"]["mass_E4"],
                "bind_real": c["modes"]["real"]["bind_strength"],
                "bind_shuffle": c["modes"]["shuffle"]["bind_strength"],
                "dens_real": c["modes"]["real"]["dens_enhance"],
                "dens_shuffle": c["modes"]["shuffle"]["dens_enhance"],
                "gyr_ratio_real": c["modes"]["real"]["gyr_ratio"],
                "gyr_ratio_shuffle": c["modes"]["shuffle"]["gyr_ratio"],
            }
            for c in ctrl
        ],
    }

    print("--- barrido G ---")
    g_rows = []
    for G in np.linspace(0.0, 0.45, 10):
        for s in (2025, 42, 777, 3141):
            row = run_controls(int(s), G=float(G))
            g_rows.append(
                {
                    "G": float(G),
                    "seed": int(s),
                    "e4_causal": row["e4_causal"],
                    "bind_vs_shuffle": row["bind_vs_shuffle"],
                    "mass_real": row["modes"]["real"]["mass_E4"],
                    "bind_real": row["modes"]["real"]["bind_strength"],
                    "bind_shuffle": row["modes"]["shuffle"]["bind_strength"],
                    "E3": row["E3_ok"],
                }
            )
        sub = [r for r in g_rows if abs(r["G"] - float(G)) < 1e-9]
        print(
            f"  G={G:.2f} causal={np.mean([r['e4_causal'] for r in sub]):.2f} "
            f"bindR/S={np.mean([r['bind_vs_shuffle'] for r in sub]):.2f} "
            f"mR={np.mean([r['mass_real'] for r in sub]):.3f}"
        )
    results["sweep_G"] = {
        "rows": g_rows,
        "rate_causal_Ggt0": sum(r["e4_causal"] for r in g_rows if r["G"] > 0)
        / max(sum(1 for r in g_rows if r["G"] > 0), 1),
        "rate_causal_G0": sum(r["e4_causal"] for r in g_rows if r["G"] == 0)
        / max(sum(1 for r in g_rows if r["G"] == 0), 1),
    }

    # multi-L smoke
    print("--- L smoke ---")
    l_rows = []
    for L in (24, 28, 32, 40):
        for s in (2025, 42, 777):
            row = run_controls(int(s), G=0.20)
            # re-run with L - need custom
            outs = {}
            for m in ("real", "shuffle"):
                sim = simulate(P(seed=int(s), L=int(L), G_GRAV=0.20, grav_mode=m))
                outs[m] = analyze(sim)
            bind_vs = outs["real"]["bind_strength"] / max(
                outs["shuffle"]["bind_strength"], 1e-12
            )
            causal = (
                outs["real"]["E3_ok"]
                and outs["real"]["mass_E4"] >= MASS_REAL_MIN
                and outs["shuffle"]["mass_E4"] <= 1e-12
                and outs["real"]["bind_strength"] >= BIND_REAL_MIN
                and bind_vs >= BIND_VS_SHUFFLE_MIN
            )
            l_rows.append(
                {
                    "L": L,
                    "seed": s,
                    "causal": causal,
                    "bind_vs": bind_vs,
                    "mass_real": outs["real"]["mass_E4"],
                    "E3": outs["real"]["E3_ok"],
                }
            )
        sub = [r for r in l_rows if r["L"] == L]
        print(
            f"  L={L} causal={np.mean([r['causal'] for r in sub]):.2f} "
            f"bindR/S={np.mean([r['bind_vs'] for r in sub]):.2f}"
        )
    results["sweep_L"] = {
        "rows": l_rows,
        "rate_causal": sum(r["causal"] for r in l_rows) / len(l_rows),
    }

    c = results["controls"]
    mass_nulls = (
        c["mean_mass_off"] <= 1e-12
        and c["mean_mass_shuffle"] <= 1e-12
        and c["mean_mass_invert"] <= 1e-12
    )
    e3_ok = c["rate_E3"] >= RATE_PASS
    e4_ok = c["rate_e4_causal"] >= RATE_PASS
    bind_sep = c["mean_bind_vs_shuffle"] >= BIND_VS_SHUFFLE_MIN

    if e3_ok and e4_ok and mass_nulls and bind_sep:
        verdict = "E3_OK_E4_ATOMIC_NBODY_CAUSAL_OK"
    elif e3_ok and mass_nulls and c["mean_mass_real"] > MASS_REAL_MIN and not e4_ok:
        if c["mean_bind_vs_shuffle"] > 1.05:
            verdict = "E3_OK_E4_PARTIAL_bind_sep_weak"
        else:
            verdict = "E3_OK_E4_MASS_GATE_OK_BIND_NOT_CAUSAL"
    elif e3_ok and not mass_nulls:
        verdict = "E3_OK_MASS_NULL_LEAK"
    else:
        verdict = "E3_E4_ATOMIC_FAIL"

    synthesis = {
        "global_verdict": verdict,
        "rate_E3": c["rate_E3"],
        "rate_e4_causal": c["rate_e4_causal"],
        "rate_gyr_causal": c["rate_gyr_causal"],
        "mass_nulls_clean": mass_nulls,
        "mean_mass_real": c["mean_mass_real"],
        "mean_mass_off": c["mean_mass_off"],
        "mean_mass_shuffle": c["mean_mass_shuffle"],
        "mean_mass_invert": c["mean_mass_invert"],
        "mean_bind_real": c["mean_bind_real"],
        "mean_bind_shuffle": c["mean_bind_shuffle"],
        "mean_bind_vs_shuffle": c["mean_bind_vs_shuffle"],
        "mean_dens_real": c["mean_dens_real"],
        "mean_dens_shuffle": c["mean_dens_shuffle"],
        "rate_causal_sweep_G_gt0": results["sweep_G"]["rate_causal_Ggt0"],
        "rate_causal_sweep_L": results["sweep_L"]["rate_causal"],
        "mean_leak_v3_not_mass": c["mean_leak"],
        "elapsed_s": time.time() - t0,
        "design": {
            "E4": "N-body entre centroides de átomos H estables",
            "NULL_SHUFFLE": "permutación de identidades de masa-fuente (quién atrae)",
            "mass_obs": "solo grav_mode=real; f(enlace, dens, grupos, compactación)",
        },
    }
    results["synthesis"] = synthesis

    path = OUT / "suite_epocas_masa_v3_result.json"
    # compact modes in controls for JSON size
    out_ctrl = dict(c)
    out_ctrl["rows"] = c["rows"]
    out = {
        "synthesis": synthesis,
        "controls": out_ctrl,
        "sweep_G": results["sweep_G"],
        "sweep_L": results["sweep_L"],
    }
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nJSON → {path}")

    md = []
    md.append("# SUITE ÉPOCAS MASA v3 — N-body atómico\n\n")
    md.append("**Fecha:** 2026-07-22\n\n")
    md.append(f"## Veredicto\n\n**`{verdict}`**\n\n")
    md.append("### Diseño\n\n")
    md.append(
        "- Gravedad entre **centroides de átomos H** (N-body), no campo suave barajable.\n"
        "- NULL_SHUFFLE: permuta **quién es fuente de atracción** (identidades), "
        "no un escalar de pozos genérico.\n"
        "- `mass_obs` solo en REAL.\n\n"
    )
    md.append("### Controles (10 seeds, G=0.20)\n\n")
    md.append(
        f"| modo | mean mass | mean |E_bind| | dens |\n"
        f"|------|-----------|----------------|------|\n"
        f"| REAL | **{c['mean_mass_real']:.3f}** | **{c['mean_bind_real']:.3f}** | "
        f"**{c['mean_dens_real']:.2f}** |\n"
        f"| OFF | {c['mean_mass_off']:.3f} | {c['mean_bind_off']:.3f} | — |\n"
        f"| SHUFFLE | {c['mean_mass_shuffle']:.3f} | {c['mean_bind_shuffle']:.3f} | "
        f"{c['mean_dens_shuffle']:.2f} |\n"
        f"| INVERT | {c['mean_mass_invert']:.3f} | — | — |\n\n"
    )
    md.append(
        f"- rate E3: **{c['rate_E3']:.2f}**\n"
        f"- rate E4 causal: **{c['rate_e4_causal']:.2f}**\n"
        f"- rate gyr_causal: **{c['rate_gyr_causal']:.2f}**\n"
        f"- mean bind REAL/SHUFFLE: **{c['mean_bind_vs_shuffle']:.2f}** "
        f"(umbral {BIND_VS_SHUFFLE_MIN})\n"
        f"- mass nulls clean: **{mass_nulls}**\n"
        f"- leak v3 (no masa): **{c['mean_leak']:.3f}**\n"
        f"- causal G>0: **{results['sweep_G']['rate_causal_Ggt0']:.2f}** · "
        f"causal L: **{results['sweep_L']['rate_causal']:.2f}**\n\n"
    )
    md.append(f"Tiempo: {synthesis['elapsed_s']:.1f} s\n")
    (OUT / "RESUMEN_SUITE_EPOCAS_MASA_v3.md").write_text("".join(md), encoding="utf-8")
    print(f"MD  → {OUT / 'RESUMEN_SUITE_EPOCAS_MASA_v3.md'}")
    print("\n=== GLOBAL ===", verdict)
    print(json.dumps(synthesis, indent=2))


if __name__ == "__main__":
    main()
