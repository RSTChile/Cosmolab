#!/usr/bin/env python3
"""
SUITE ÉPOCAS MASA v4 — pares mutuos estables + linaje de fusión

Hereda v3 (N-body entre centroides H) y endurece el juez causal E4.

Problema v3: E_bind total REAL ≈ SHUFFLE a veces (fuerzas all-to-all de la misma
familia de magnitudes) → rate causal 0.30. El canal de *masa* ya estaba limpio
(nulls OFF/SHUFFLE/INVERT = 0); el cuello era la *separación de enlace*.

Respuesta v4 (sin subir umbrales a mano para fabricar PASS):
  1. CUTOFF de fuerza: solo actúan pares a r < FORCE_CUTOFF (no "todo atrae a todo").
  2. PARES MUTUOS: contador de co-proximidad por IDs de átomos; E_mutual y mass_obs
     solo usan pares con edad ≥ MUTUAL_MIN_STEPS. SHUFFLE con fuentes foráneas no
     sostiene el mismo grafo de quién se queda con quién.
  3. LINAJE DE FUSIÓN: fracciones de co-membresía en grupos a lo largo de E4 +
     conteo de eventos de fusión (dos componentes se unen). REAL debe formar
     linajes más estables que SHUFFLE.

Contrato Alexis intacto:
  E0–E3 sin mass_obs; masa solo E4 con gravedad REAL sobre H densificado.
  mass_obs = 0 en OFF / SHUFFLE / INVERT.
  Rm/v3 sigue siendo leak de nombre, no masa.
  Anti-Shannon: sin 1/1836; barridos G y seeds.

Nulos:
  OFF:      G=0
  SHUFFLE:  permuta identidades de masa-fuente (quién atrae)
  INVERT:   repulsión
"""
from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[2] / "results" / "suite_epocas_masa_v4"
OUT.mkdir(parents=True, exist_ok=True)

# E3 strict
K_MIN, K_MAX = 4, 14
F_CORE_MIN, F_CORE_MAX = 0.15, 0.75
COHESION_MIN, COHESION_MAX = 1.2, 6.5
PERSIST_STEPS = 4
VEV_POST_MIN = 0.10
PHI_CORE_THR = 0.35
CENTROID_TOL = 2.5

# E4 N-body + v4 hardenings
SOFTENING = 1.2
DT_NB = 0.35
GROUP_LINK_R = 4.5
FORCE_CUTOFF = 8.0          # v4: sin fuerza más allá de ~2× radio de grupo
MUTUAL_MIN_STEPS = 5        # pares que persisten ≥5 pasos E4
MASS_REAL_MIN = 0.3
BIND_REAL_MIN = 0.05
MUTUAL_MIN = 1              # al menos 1 par mutuo en REAL
DENS_MIN = 1.2
# umbrales de separación (mismos espíritu que v3; no se inflan)
BIND_VS_SHUFFLE_MIN = 1.25
MUTUAL_VS_SHUFFLE_MIN = 1.25   # n_mutual_REAL / max(n_mutual_S, eps)
COMEM_VS_SHUFFLE_MIN = 1.15    # co-membresía media REAL / SHUFFLE
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


def nbody_step(pos, masses, ids, G, L, mode, perm=None):
    """
    Fuerza con CUTOFF. Devuelve también E_total (todos pares en cutoff) y
    mapa de pares cercanos (id_i, id_j) -> r para linaje/mutuos.
    """
    N = len(masses)
    if N == 0:
        return pos, 0.0, 0.0, 0, {}
    if mode == "off" or G <= 0:
        return pos, 0.0, 0.0, 0, {}

    acc = np.zeros_like(pos)
    E_bind = 0.0
    pair_r = []
    close_pairs = {}  # frozenset({id_i,id_j}) -> r
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
                if r < GROUP_LINK_R:
                    key = frozenset((int(ids[i]), int(ids[j])))
                    if len(key) == 2:
                        close_pairs[key] = r

    pos = pos + DT_NB * acc
    pos[:, 0] = np.mod(pos[:, 0], L)
    pos[:, 1] = np.mod(pos[:, 1], L)
    mean_r = float(np.mean(pair_r)) if pair_r else 0.0
    n_close = len(close_pairs)
    return pos, float(E_bind), mean_r, n_close, close_pairs


def deposit_rho(pos, masses, L):
    rho = np.zeros((L, L))
    if len(masses) == 0:
        return rho
    for (y, x), m in zip(pos, masses):
        iy, ix = int(y) % L, int(x) % L
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                w = m if (dy, dx) == (0, 0) else 0.25 * m
                rho[(iy + dy) % L, (ix + dx) % L] += w
    if rho.sum() > 0:
        rho *= (L * L) / rho.sum()
    return rho


def groups_from_ids(pos, ids, L, link_r=GROUP_LINK_R):
    """Union-find por proximidad; devuelve lista de sets de atom IDs y R_gyr."""
    N = len(pos)
    if N == 0:
        return [], 0.0, 0
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

    clusters_idx = {}
    for i in range(N):
        r = find(i)
        clusters_idx.setdefault(r, []).append(i)

    id_groups = []
    gyrs = []
    for idxs in clusters_idx.values():
        id_set = frozenset(int(ids[i]) for i in idxs)
        id_groups.append(id_set)
        if len(idxs) < 2:
            continue
        pts = pos[idxs]
        cy, cx = float(pts[:, 0].mean()), float(pts[:, 1].mean())
        d2 = []
        for y, x in pts:
            dy = toroidal_delta(cy, y, L)
            dx = toroidal_delta(cx, x, L)
            d2.append(dy * dy + dx * dx)
        gyrs.append(np.sqrt(np.mean(d2)))
    n_groups = sum(1 for g in id_groups if len(g) >= 2)
    r_gyr = float(np.mean(gyrs)) if gyrs else float("nan")
    return id_groups, r_gyr, n_groups


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

    nb_pos = None
    nb_mass = None
    nb_ids = None
    nb_perm = None
    r_gyr_start = None
    e_bind_acc = []
    e_mutual_acc = []
    mass_obs_max = 0.0  # max over ALL E4 steps (not only hist samples)

    # v4 trackers
    mutual_age = defaultdict(int)       # frozenset({id1,id2}) -> consecutive steps close
    mutual_age_max = defaultdict(int)   # max age seen
    pair_co_steps = defaultdict(int)    # steps as co-members of same group
    pair_possible_steps = 0             # E4 steps with ≥2 atoms
    prev_groups = []                    # list of frozensets of ids
    fusion_events = 0
    e4_steps = 0

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

        E_bind = 0.0
        E_mutual = 0.0
        mean_r = 0.0
        n_close = 0
        n_groups = 0
        r_gyr = float("nan")
        dens_enhance = 1.0
        mass_obs = 0.0
        n_mutual = 0

        if e4_window and len(stable_atoms) >= 2:
            pos = np.array([[t["cy"], t["cx"]] for t in stable_atoms], dtype=float)
            masses = np.array([t["mass"] for t in stable_atoms], dtype=float)
            ids = np.array([t["id"] for t in stable_atoms], dtype=int)

            if nb_pos is None or len(nb_pos) != len(pos):
                nb_pos = pos.copy()
                nb_mass = masses.copy()
                nb_ids = ids.copy()
                if p.grav_mode == "shuffle":
                    nb_perm = rng.permutation(len(masses))
                else:
                    nb_perm = None
                _, r0, _ = groups_from_ids(nb_pos, nb_ids, L)
                r_gyr_start = r0 if r0 == r0 else None
                # reset trackers on topology change of N
                mutual_age = defaultdict(int)
                prev_groups = []
            else:
                nb_mass = masses
                # keep evolved positions; re-bind ids by matching previous ids order
                # stable_atoms order may change: remap nb_pos by id
                id_to_pos = {int(i): nb_pos[k] for k, i in enumerate(nb_ids)}
                new_pos = []
                for tid in ids:
                    if int(tid) in id_to_pos:
                        new_pos.append(id_to_pos[int(tid)])
                    else:
                        # new id: take from current track position
                        idx = list(ids).index(tid)
                        new_pos.append(pos[idx])
                nb_pos = np.array(new_pos, dtype=float)
                nb_ids = ids.copy()

            nb_pos, E_bind, mean_r, n_close, close_pairs = nbody_step(
                nb_pos, nb_mass, nb_ids, p.G_GRAV, L, p.grav_mode, perm=nb_perm
            )

            # write back centroids
            id_to_newpos = {int(nb_ids[i]): nb_pos[i] for i in range(len(nb_ids))}
            for t in stable_atoms:
                if int(t["id"]) in id_to_newpos:
                    t["cy"], t["cx"] = (
                        float(id_to_newpos[int(t["id"])][0]),
                        float(id_to_newpos[int(t["id"])][1]),
                    )

            # --- mutual pairs (consecutive proximity by atom ID) ---
            active_keys = set(close_pairs.keys())
            for key in list(mutual_age.keys()):
                if key not in active_keys:
                    mutual_age[key] = 0
            for key in active_keys:
                mutual_age[key] += 1
                mutual_age_max[key] = max(mutual_age_max[key], mutual_age[key])

            # E_mutual: only pairs with current age ≥ MUTUAL_MIN_STEPS
            E_mutual = 0.0
            n_mutual = 0
            id_to_idx = {int(nb_ids[i]): i for i in range(len(nb_ids))}
            for key, age in mutual_age.items():
                if age < MUTUAL_MIN_STEPS:
                    continue
                pair = list(key)
                if len(pair) != 2:
                    continue
                if pair[0] not in id_to_idx or pair[1] not in id_to_idx:
                    continue
                i, j = id_to_idx[pair[0]], id_to_idx[pair[1]]
                dy = toroidal_delta(nb_pos[i, 0], nb_pos[j, 0], L)
                dx = toroidal_delta(nb_pos[i, 1], nb_pos[j, 1], L)
                r = float(np.hypot(dy, dx)) + 1e-9
                if r > FORCE_CUTOFF:
                    continue
                r_soft = np.sqrt(r * r + SOFTENING**2)
                # energy uses presented sources as in mode
                src_j = nb_mass[nb_perm[j]] if (p.grav_mode == "shuffle" and nb_perm is not None) else nb_mass[j]
                src_i = nb_mass[nb_perm[i]] if (p.grav_mode == "shuffle" and nb_perm is not None) else nb_mass[i]
                # symmetric-ish: mean of both directed contributions
                E_mutual += -0.5 * p.G_GRAV * (
                    nb_mass[i] * src_j + nb_mass[j] * src_i
                ) / r_soft
                n_mutual += 1

            # --- fusion lineage / co-membership ---
            id_groups, r_gyr, n_groups = groups_from_ids(nb_pos, nb_ids, L)
            e4_steps += 1
            if len(nb_ids) >= 2:
                pair_possible_steps += 1
                # co-membership: for every pair of current ids in same group
                for g in id_groups:
                    if len(g) < 2:
                        continue
                    gl = list(g)
                    for a in range(len(gl)):
                        for b in range(a + 1, len(gl)):
                            pair_co_steps[frozenset((gl[a], gl[b]))] += 1

                # fusion events: two previously separate multi-sets merge
                if prev_groups:
                    # map atom -> old group root label
                    old_of = {}
                    for gi, g in enumerate(prev_groups):
                        for aid in g:
                            old_of[aid] = gi
                    for g in id_groups:
                        if len(g) < 2:
                            continue
                        old_labels = {old_of[a] for a in g if a in old_of}
                        if len(old_labels) >= 2:
                            fusion_events += 1
                prev_groups = id_groups

            rho_H = deposit_rho(nb_pos, nb_mass, L)
            if rho_H.mean() > 0:
                dens_enhance = float(rho_H.max() / (rho_H.mean() + 1e-12))

            e_bind_acc.append(E_bind)
            e_mutual_acc.append(E_mutual)

            # mass_obs SOLO real: usa enlace MUTUO (no E_bind total).
            # Requiere al menos 1 par mutuo; no multiplica por n_mutual×n_groups
            # (eso anulaba mass aunque mutual_bind fuera alto — bug v4.0).
            if p.grav_mode == "real" and p.G_GRAV > 0 and n_mutual >= 1:
                bind_strength = max(0.0, -E_mutual)
                gyr_factor = 1.0
                if r_gyr_start and r_gyr == r_gyr and r_gyr_start > 0:
                    gyr_factor = max(r_gyr_start / (r_gyr + 1e-6), 1.0)
                mass_obs = float(
                    bind_strength
                    * dens_enhance
                    * gyr_factor
                    * max(n_groups, 1)
                    / max(len(nb_mass), 1)
                )
            else:
                mass_obs = 0.0
            if mass_obs > mass_obs_max:
                mass_obs_max = mass_obs

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
                    "E_mutual": E_mutual,
                    "n_mutual": n_mutual,
                    "mean_pair_r": mean_r,
                    "n_close_pairs": n_close,
                    "R_gyr": r_gyr if r_gyr == r_gyr else None,
                    "R_gyr_start": r_gyr_start,
                    "dens_enhance": dens_enhance,
                    "mass_obs": mass_obs,
                    "grav_mode": p.grav_mode,
                    "leak_sep": leak,
                    "fusion_events_so_far": fusion_events,
                }
            )

    # co-membership score: mean fraction of E4 steps that final-candidate pairs spent together
    n_mut_stable = sum(1 for a in mutual_age_max.values() if a >= MUTUAL_MIN_STEPS)
    if pair_possible_steps > 0 and pair_co_steps:
        comem_fracs = [
            v / pair_possible_steps for v in pair_co_steps.values() if v > 0
        ]
        co_member_score = float(np.mean(comem_fracs)) if comem_fracs else 0.0
        # only pairs that ever co-membered — also report high-persistence pairs
        n_long_co = sum(
            1 for v in pair_co_steps.values() if v >= MUTUAL_MIN_STEPS
        )
    else:
        co_member_score = 0.0
        n_long_co = 0

    return {
        "params": asdict(p),
        "hist": hist,
        "E_bind_mean_E4": float(np.mean(e_bind_acc)) if e_bind_acc else 0.0,
        "E_bind_min_E4": float(np.min(e_bind_acc)) if e_bind_acc else 0.0,
        "E_mutual_mean_E4": float(np.mean(e_mutual_acc)) if e_mutual_acc else 0.0,
        "E_mutual_min_E4": float(np.min(e_mutual_acc)) if e_mutual_acc else 0.0,
        "mass_obs_max": float(mass_obs_max),
        "n_mutual_stable": int(n_mut_stable),
        "n_long_co_pairs": int(n_long_co),
        "co_member_score": float(co_member_score),
        "fusion_events": int(fusion_events),
        "e4_steps": int(e4_steps),
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

    def avg(rows, key):
        vals = [h[key] for h in rows if h.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    mass_pre = mx(pre, "mass_obs")
    # mass_E4: max sobre TODOS los pasos E4 (mass_obs_max), no solo hist cada 20
    mass_e4 = float(sim.get("mass_obs_max", 0.0) or 0.0)
    if p["grav_mode"] != "real" or p.get("G_GRAV", 0) <= 0:
        mass_e4 = 0.0
    E_min = sim["E_bind_min_E4"]
    E_mean = sim["E_bind_mean_E4"]
    Em_min = sim["E_mutual_min_E4"]
    Em_mean = sim["E_mutual_mean_E4"]
    dens = mx(late, "dens_enhance")
    n_st = mx(hist, "n_atoms_stable")
    n_gr = mx(late, "n_groups")
    n_mut_hist = mx(late, "n_mutual")

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

    mutual_bind = max(0.0, -Em_min)
    mode = p["grav_mode"]
    if mode == "real":
        e4_ok = (
            mass_pre <= 1e-12
            and mass_e4 >= MASS_REAL_MIN
            and mutual_bind >= BIND_REAL_MIN
            and sim["n_mutual_stable"] >= MUTUAL_MIN
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
        "bind_strength": max(0.0, -E_min),  # total (diagnóstico)
        "mutual_bind": mutual_bind,         # v4 juez principal
        "E_mutual_mean": Em_mean,
        "n_mutual_stable": sim["n_mutual_stable"],
        "n_mutual_late": n_mut_hist,
        "n_long_co_pairs": sim["n_long_co_pairs"],
        "co_member_score": sim["co_member_score"],
        "fusion_events": sim["fusion_events"],
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
    mutual_vs_shuf = r["mutual_bind"] / max(s["mutual_bind"], eps)
    nmut_vs_shuf = r["n_mutual_stable"] / max(s["n_mutual_stable"], eps)
    comem_vs_shuf = r["co_member_score"] / max(s["co_member_score"], eps)
    fusion_vs_shuf = r["fusion_events"] / max(s["fusion_events"], eps)

    # causalidad v4: nulls limpios + masa REAL + pares mutuos + linaje
    e4_causal = (
        r["E3_ok"]
        and r["zero_mass_pre"]
        and o["mass_E4"] <= eps
        and s["mass_E4"] <= eps
        and inv["mass_E4"] <= eps
        and r["mass_E4"] >= MASS_REAL_MIN
        and r["mutual_bind"] >= BIND_REAL_MIN
        and r["n_mutual_stable"] >= MUTUAL_MIN
        and mutual_vs_shuf >= BIND_VS_SHUFFLE_MIN
        and (
            nmut_vs_shuf >= MUTUAL_VS_SHUFFLE_MIN
            or comem_vs_shuf >= COMEM_VS_SHUFFLE_MIN
        )
        and r["dens_enhance"] >= DENS_MIN
    )

    # linaje fuerte (sub-flag): co-membresía o fusiones
    lineage_causal = (
        comem_vs_shuf >= COMEM_VS_SHUFFLE_MIN
        or (r["fusion_events"] > s["fusion_events"] and fusion_vs_shuf >= 1.15)
        or nmut_vs_shuf >= MUTUAL_VS_SHUFFLE_MIN
    )

    gyr_real = r["gyr_ratio"]
    gyr_shuf = s["gyr_ratio"]
    gyr_causal = False
    if gyr_real is not None and gyr_shuf is not None:
        gyr_causal = gyr_real < gyr_shuf * 0.95
    elif gyr_real is not None:
        gyr_causal = gyr_real < 0.92

    return {
        "seed": seed,
        "G": G,
        "modes": outs,
        "bind_vs_shuffle": bind_vs_shuf,
        "mutual_vs_shuffle": mutual_vs_shuf,
        "nmut_vs_shuffle": nmut_vs_shuf,
        "comem_vs_shuffle": comem_vs_shuf,
        "fusion_vs_shuffle": fusion_vs_shuf,
        "e4_causal": e4_causal,
        "lineage_causal": lineage_causal,
        "gyr_causal": gyr_causal,
        "E3_ok": r["E3_ok"],
        "E0_ok": r["E0_ok"],
        "E1_ok": r["E1_ok"],
    }


def main():
    print("=== SUITE ÉPOCAS MASA v4 — pares mutuos + linaje de fusión ===\n")
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
            f"lin={row['lineage_causal']} "
            f"mR={r['mass_E4']:.3f} mS={sh['mass_E4']:.3f} "
            f"mutR={r['mutual_bind']:.3f} mutS={sh['mutual_bind']:.3f} "
            f"nMutR={r['n_mutual_stable']} nMutS={sh['n_mutual_stable']} "
            f"coR={r['co_member_score']:.3f} coS={sh['co_member_score']:.3f} "
            f"fusR={r['fusion_events']} fusS={sh['fusion_events']} "
            f"mutR/S={row['mutual_vs_shuffle']:.2f}"
        )

    results["controls"] = {
        "n": len(ctrl),
        "rate_E3": sum(c["E3_ok"] for c in ctrl) / len(ctrl),
        "rate_e4_causal": sum(c["e4_causal"] for c in ctrl) / len(ctrl),
        "rate_lineage_causal": sum(c["lineage_causal"] for c in ctrl) / len(ctrl),
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
        "mean_mutual_real": float(
            np.mean([c["modes"]["real"]["mutual_bind"] for c in ctrl])
        ),
        "mean_mutual_shuffle": float(
            np.mean([c["modes"]["shuffle"]["mutual_bind"] for c in ctrl])
        ),
        "mean_nmut_real": float(
            np.mean([c["modes"]["real"]["n_mutual_stable"] for c in ctrl])
        ),
        "mean_nmut_shuffle": float(
            np.mean([c["modes"]["shuffle"]["n_mutual_stable"] for c in ctrl])
        ),
        "mean_comem_real": float(
            np.mean([c["modes"]["real"]["co_member_score"] for c in ctrl])
        ),
        "mean_comem_shuffle": float(
            np.mean([c["modes"]["shuffle"]["co_member_score"] for c in ctrl])
        ),
        "mean_fusion_real": float(
            np.mean([c["modes"]["real"]["fusion_events"] for c in ctrl])
        ),
        "mean_fusion_shuffle": float(
            np.mean([c["modes"]["shuffle"]["fusion_events"] for c in ctrl])
        ),
        "mean_bind_real": float(
            np.mean([c["modes"]["real"]["bind_strength"] for c in ctrl])
        ),
        "mean_bind_shuffle": float(
            np.mean([c["modes"]["shuffle"]["bind_strength"] for c in ctrl])
        ),
        "mean_dens_real": float(
            np.mean([c["modes"]["real"]["dens_enhance"] for c in ctrl])
        ),
        "mean_dens_shuffle": float(
            np.mean([c["modes"]["shuffle"]["dens_enhance"] for c in ctrl])
        ),
        "mean_mutual_vs_shuffle": float(
            np.mean([c["mutual_vs_shuffle"] for c in ctrl])
        ),
        "mean_nmut_vs_shuffle": float(np.mean([c["nmut_vs_shuffle"] for c in ctrl])),
        "mean_comem_vs_shuffle": float(
            np.mean([c["comem_vs_shuffle"] for c in ctrl])
        ),
        "mean_leak": float(np.mean([c["modes"]["real"]["leak_max"] for c in ctrl])),
        "rows": [
            {
                "seed": c["seed"],
                "e4_causal": c["e4_causal"],
                "lineage_causal": c["lineage_causal"],
                "gyr_causal": c["gyr_causal"],
                "E3_ok": c["E3_ok"],
                "mutual_vs_shuffle": c["mutual_vs_shuffle"],
                "nmut_vs_shuffle": c["nmut_vs_shuffle"],
                "comem_vs_shuffle": c["comem_vs_shuffle"],
                "fusion_vs_shuffle": c["fusion_vs_shuffle"],
                "mass_real": c["modes"]["real"]["mass_E4"],
                "mass_shuffle": c["modes"]["shuffle"]["mass_E4"],
                "mutual_real": c["modes"]["real"]["mutual_bind"],
                "mutual_shuffle": c["modes"]["shuffle"]["mutual_bind"],
                "nmut_real": c["modes"]["real"]["n_mutual_stable"],
                "nmut_shuffle": c["modes"]["shuffle"]["n_mutual_stable"],
                "comem_real": c["modes"]["real"]["co_member_score"],
                "comem_shuffle": c["modes"]["shuffle"]["co_member_score"],
                "fusion_real": c["modes"]["real"]["fusion_events"],
                "fusion_shuffle": c["modes"]["shuffle"]["fusion_events"],
                "dens_real": c["modes"]["real"]["dens_enhance"],
                "dens_shuffle": c["modes"]["shuffle"]["dens_enhance"],
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
                    "lineage_causal": row["lineage_causal"],
                    "mutual_vs_shuffle": row["mutual_vs_shuffle"],
                    "nmut_vs_shuffle": row["nmut_vs_shuffle"],
                    "mass_real": row["modes"]["real"]["mass_E4"],
                    "mutual_real": row["modes"]["real"]["mutual_bind"],
                    "mutual_shuffle": row["modes"]["shuffle"]["mutual_bind"],
                    "E3": row["E3_ok"],
                }
            )
        sub = [r for r in g_rows if abs(r["G"] - float(G)) < 1e-9]
        print(
            f"  G={G:.2f} causal={np.mean([r['e4_causal'] for r in sub]):.2f} "
            f"lin={np.mean([r['lineage_causal'] for r in sub]):.2f} "
            f"mutR/S={np.mean([r['mutual_vs_shuffle'] for r in sub]):.2f} "
            f"mR={np.mean([r['mass_real'] for r in sub]):.3f}"
        )
    results["sweep_G"] = {
        "rows": g_rows,
        "rate_causal_Ggt0": sum(r["e4_causal"] for r in g_rows if r["G"] > 0)
        / max(sum(1 for r in g_rows if r["G"] > 0), 1),
        "rate_causal_G0": sum(r["e4_causal"] for r in g_rows if r["G"] == 0)
        / max(sum(1 for r in g_rows if r["G"] == 0), 1),
        "rate_lineage_Ggt0": sum(r["lineage_causal"] for r in g_rows if r["G"] > 0)
        / max(sum(1 for r in g_rows if r["G"] > 0), 1),
    }

    print("--- L smoke ---")
    l_rows = []
    for L in (24, 28, 32, 40):
        for s in (2025, 42, 777):
            outs = {}
            for m in ("real", "shuffle"):
                sim = simulate(P(seed=int(s), L=int(L), G_GRAV=0.20, grav_mode=m))
                outs[m] = analyze(sim)
            mut_vs = outs["real"]["mutual_bind"] / max(
                outs["shuffle"]["mutual_bind"], 1e-12
            )
            nmut_vs = outs["real"]["n_mutual_stable"] / max(
                outs["shuffle"]["n_mutual_stable"], 1e-12
            )
            comem_vs = outs["real"]["co_member_score"] / max(
                outs["shuffle"]["co_member_score"], 1e-12
            )
            causal = (
                outs["real"]["E3_ok"]
                and outs["real"]["mass_E4"] >= MASS_REAL_MIN
                and outs["shuffle"]["mass_E4"] <= 1e-12
                and outs["real"]["mutual_bind"] >= BIND_REAL_MIN
                and outs["real"]["n_mutual_stable"] >= MUTUAL_MIN
                and mut_vs >= BIND_VS_SHUFFLE_MIN
                and (
                    nmut_vs >= MUTUAL_VS_SHUFFLE_MIN
                    or comem_vs >= COMEM_VS_SHUFFLE_MIN
                )
            )
            l_rows.append(
                {
                    "L": L,
                    "seed": s,
                    "causal": causal,
                    "mut_vs": mut_vs,
                    "nmut_vs": nmut_vs,
                    "comem_vs": comem_vs,
                    "mass_real": outs["real"]["mass_E4"],
                    "E3": outs["real"]["E3_ok"],
                }
            )
        sub = [r for r in l_rows if r["L"] == L]
        print(
            f"  L={L} causal={np.mean([r['causal'] for r in sub]):.2f} "
            f"mutR/S={np.mean([r['mut_vs'] for r in sub]):.2f} "
            f"nMutR/S={np.mean([r['nmut_vs'] for r in sub]):.2f}"
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
    lin_ok = c["rate_lineage_causal"] >= RATE_PASS
    mut_sep = c["mean_mutual_vs_shuffle"] >= BIND_VS_SHUFFLE_MIN

    if e3_ok and e4_ok and mass_nulls and mut_sep and lin_ok:
        verdict = "E3_OK_E4_FUSION_LINEAGE_CAUSAL_OK"
    elif e3_ok and mass_nulls and c["mean_mass_real"] > MASS_REAL_MIN and not e4_ok:
        if c["mean_mutual_vs_shuffle"] > 1.05 or c["rate_lineage_causal"] >= 0.3:
            verdict = "E3_OK_E4_PARTIAL_lineage_weak"
        else:
            verdict = "E3_OK_E4_MASS_GATE_OK_LINEAGE_NOT_CAUSAL"
    elif e3_ok and not mass_nulls:
        verdict = "E3_OK_MASS_NULL_LEAK"
    else:
        verdict = "E3_E4_FUSION_FAIL"

    synthesis = {
        "global_verdict": verdict,
        "rate_E3": c["rate_E3"],
        "rate_e4_causal": c["rate_e4_causal"],
        "rate_lineage_causal": c["rate_lineage_causal"],
        "rate_gyr_causal": c["rate_gyr_causal"],
        "mass_nulls_clean": mass_nulls,
        "mean_mass_real": c["mean_mass_real"],
        "mean_mass_off": c["mean_mass_off"],
        "mean_mass_shuffle": c["mean_mass_shuffle"],
        "mean_mass_invert": c["mean_mass_invert"],
        "mean_mutual_real": c["mean_mutual_real"],
        "mean_mutual_shuffle": c["mean_mutual_shuffle"],
        "mean_mutual_vs_shuffle": c["mean_mutual_vs_shuffle"],
        "mean_nmut_real": c["mean_nmut_real"],
        "mean_nmut_shuffle": c["mean_nmut_shuffle"],
        "mean_nmut_vs_shuffle": c["mean_nmut_vs_shuffle"],
        "mean_comem_real": c["mean_comem_real"],
        "mean_comem_shuffle": c["mean_comem_shuffle"],
        "mean_comem_vs_shuffle": c["mean_comem_vs_shuffle"],
        "mean_fusion_real": c["mean_fusion_real"],
        "mean_fusion_shuffle": c["mean_fusion_shuffle"],
        "mean_dens_real": c["mean_dens_real"],
        "mean_dens_shuffle": c["mean_dens_shuffle"],
        "rate_causal_sweep_G_gt0": results["sweep_G"]["rate_causal_Ggt0"],
        "rate_lineage_sweep_G_gt0": results["sweep_G"]["rate_lineage_Ggt0"],
        "rate_causal_sweep_L": results["sweep_L"]["rate_causal"],
        "mean_leak_v3_not_mass": c["mean_leak"],
        "elapsed_s": time.time() - t0,
        "design": {
            "E4": "N-body + cutoff FORCE_CUTOFF + pares mutuos + linaje fusión",
            "NULL_SHUFFLE": "permutación de identidades de masa-fuente",
            "mass_obs": "solo REAL; f(E_mutual, dens, grupos, n_mutual, compactación)",
            "juez_causal": "mutual_bind R/S + (n_mutual R/S o co_member R/S)",
        },
    }
    results["synthesis"] = synthesis

    path = OUT / "suite_epocas_masa_v4_result.json"
    out = {
        "synthesis": synthesis,
        "controls": {k: v for k, v in c.items()},
        "sweep_G": results["sweep_G"],
        "sweep_L": results["sweep_L"],
    }
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nJSON → {path}")

    md = []
    md.append("# SUITE ÉPOCAS MASA v4 — pares mutuos + linaje de fusión\n\n")
    md.append("**Fecha:** 2026-07-22\n\n")
    md.append(f"## Veredicto\n\n**`{verdict}`**\n\n")
    md.append("### Diseño (endurecimiento vs v3)\n\n")
    md.append(
        "1. **FORCE_CUTOFF**: fuerza solo a r < 8 (no all-to-all).\n"
        "2. **Pares mutuos**: co-proximidad por ID ≥ 5 pasos; `E_mutual` y `mass_obs` solo con esos pares.\n"
        "3. **Linaje**: co-membresía de grupos a lo largo de E4 + eventos de fusión.\n"
        "4. NULL_SHUFFLE: permuta **quién es fuente** (igual que v3).\n"
        "5. `mass_obs` solo en REAL; OFF/SHUFFLE/INVERT → 0.\n\n"
    )
    md.append("### Controles (10 seeds, G=0.20)\n\n")
    md.append(
        f"| métrica | REAL | SHUFFLE | OFF |\n"
        f"|---------|------|---------|-----|\n"
        f"| mass | **{c['mean_mass_real']:.3f}** | {c['mean_mass_shuffle']:.3f} | {c['mean_mass_off']:.3f} |\n"
        f"| mutual_bind | **{c['mean_mutual_real']:.3f}** | {c['mean_mutual_shuffle']:.3f} | — |\n"
        f"| n_mutual_stable | **{c['mean_nmut_real']:.2f}** | {c['mean_nmut_shuffle']:.2f} | — |\n"
        f"| co_member_score | **{c['mean_comem_real']:.3f}** | {c['mean_comem_shuffle']:.3f} | — |\n"
        f"| fusion_events | **{c['mean_fusion_real']:.2f}** | {c['mean_fusion_shuffle']:.2f} | — |\n"
        f"| dens | **{c['mean_dens_real']:.2f}** | {c['mean_dens_shuffle']:.2f} | — |\n\n"
    )
    md.append(
        f"- rate E3: **{c['rate_E3']:.2f}**\n"
        f"- rate E4 causal (v4): **{c['rate_e4_causal']:.2f}**\n"
        f"- rate lineage_causal: **{c['rate_lineage_causal']:.2f}**\n"
        f"- mean mutual REAL/SHUFFLE: **{c['mean_mutual_vs_shuffle']:.2f}** "
        f"(umbral {BIND_VS_SHUFFLE_MIN})\n"
        f"- mean n_mutual R/S: **{c['mean_nmut_vs_shuffle']:.2f}**\n"
        f"- mean co_member R/S: **{c['mean_comem_vs_shuffle']:.2f}**\n"
        f"- mass nulls clean: **{mass_nulls}**\n"
        f"- leak v3 (no masa): **{c['mean_leak']:.3f}**\n"
        f"- causal G>0: **{results['sweep_G']['rate_causal_Ggt0']:.2f}** · "
        f"lineage G>0: **{results['sweep_G']['rate_lineage_Ggt0']:.2f}** · "
        f"causal L: **{results['sweep_L']['rate_causal']:.2f}**\n\n"
    )
    md.append(
        "### Lectura\n\n"
        "- Si mass nulls limpios y mutual/linaje separan REAL≻SHUFFLE → E4 causal admitido.\n"
        "- Si mass limpia pero linaje no separa → el clumping aún no es grafo de fusión "
        "específico de gravedad-H (hallazgo útil, no adorno).\n"
        "- Rm/v3 sigue **no** siendo masa.\n\n"
    )
    md.append(f"Tiempo: {synthesis['elapsed_s']:.1f} s\n")
    (OUT / "RESUMEN_SUITE_EPOCAS_MASA_v4.md").write_text("".join(md), encoding="utf-8")
    print(f"MD  → {OUT / 'RESUMEN_SUITE_EPOCAS_MASA_v4.md'}")
    print("\n=== GLOBAL ===", verdict)
    print(json.dumps(synthesis, indent=2))


if __name__ == "__main__":
    main()
