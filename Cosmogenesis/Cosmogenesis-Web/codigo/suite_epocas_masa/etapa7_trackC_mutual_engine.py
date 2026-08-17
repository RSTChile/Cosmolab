#!/usr/bin/env python3
"""
etapa7_trackC_mutual_engine.py — Track C: motor instrumentado de diagnóstico
de E_mutual (energía de ligadura de pares REAL vs SHUFFLE).

NO EDITA v1-v6 ni motor_1a7. Importa el módulo v6
(suite_epocas_masa_v6_mass_linaje.py) por archivo y reutiliza sin tocar:
  P, medium_norm, weighted_cut, components_strict, match_persist,
  toroidal_delta, deposit_rho, groups_from_ids, premature_leak.

Solo se REESCRIBE (copia local, nueva función) el bucle principal
`simulate()` y `nbody_step()`, para poder:
  (a) parametrizar FORCE_CUTOFF / SOFTENING / MUTUAL_MIN_STEPS sin tocar
      las constantes globales de v6 (H2),
  (b) barrer GRAV_START_FRAC / pasos, que ya son campos de P (H1),
  (c) inyectar un control `posrandom_e4_entry` que aleatoriza las
      posiciones de los átomos al primer instante en que entran a la
      fase N-body de E4, rompiendo la herencia espacial del campo (H3),
  (d) registrar, por paso, la energía de CADA par próximo tanto en la
      versión "gateada" (edad>=mutual_min_steps, como v6) como en la
      versión "instantánea" (sin gate de persistencia), para poder
      recalcular post-hoc min/mean, sumado/promediado-por-par (H4),
      sin tener que volver a correr la simulación.

Referencia cruzada con v6 (para auditoría, líneas del motor original):
  nbody_step():            v6 líneas 249-299
  bloque E_mutual (gate):  v6 líneas 544-570
  mutual_bind = max(0,-min_t(E_mutual)): v6 analyze() línea 757
"""
from __future__ import annotations

import sys
import importlib.util
from collections import defaultdict, deque
from dataclasses import asdict
from pathlib import Path

import numpy as np

ENGINE_PATH = Path(__file__).resolve().parent / "suite_epocas_masa_v6_mass_linaje.py"


def _load_v6():
    spec = importlib.util.spec_from_file_location("v6engine_trackC", ENGINE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v6engine_trackC"] = mod  # necesario para @dataclass (P) en exec_module
    spec.loader.exec_module(mod)
    return mod


v6 = _load_v6()  # motor v6 SIN editar, cargado por archivo

P = v6.P
PERSIST_STEPS = v6.PERSIST_STEPS
VEV_POST_MIN = v6.VEV_POST_MIN
GROUP_LINK_R = v6.GROUP_LINK_R  # 4.5: definición de "par próximo" (no se toca; H2 solo barre force_cutoff/softening de la FUERZA/energía)


def toroidal_delta(a, b, L):
    return v6.toroidal_delta(a, b, L)


def nbody_step_cfg(pos, masses, ids, G, L, mode, perm, force_cutoff, softening):
    """
    Copia paramétrica de v6.nbody_step (líneas 249-299), idéntica en
    lógica, solo con FORCE_CUTOFF/SOFTENING como argumentos en vez de
    constantes de módulo (para poder barrerlos sin tocar v6.py).
    """
    N = len(masses)
    if N == 0:
        return pos, 0.0, 0.0, 0, {}
    if mode == "off" or G <= 0:
        return pos, 0.0, 0.0, 0, {}

    acc = np.zeros_like(pos)
    E_bind = 0.0
    pair_r = []
    close_pairs = {}
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
            if r > force_cutoff:
                continue
            r2 = r * r + softening**2
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

    pos = pos + v6.DT_NB * acc
    pos[:, 0] = np.mod(pos[:, 0], L)
    pos[:, 1] = np.mod(pos[:, 1], L)
    mean_r = float(np.mean(pair_r)) if pair_r else 0.0
    n_close = len(close_pairs)
    return pos, float(E_bind), mean_r, n_close, close_pairs


def simulate_diag(
    p,
    *,
    force_cutoff: float = 8.0,
    softening: float = 1.2,
    mutual_min_steps: int = 5,
    posrandom_e4_entry: bool = False,
) -> dict:
    """
    Copia instrumentada de v6.simulate() (líneas 386-703). Fiel al
    original salvo por:
      - force_cutoff/softening/mutual_min_steps parametrizados (H2),
      - posrandom_e4_entry: al crear nb_pos por primera vez, reemplaza
        las posiciones heredadas por U(0,L)^2 (H3),
      - registra, por paso E4, la lista completa de energías de pares
        próximos (gated e instant) para poder recomputar cualquier
        estadístico post-hoc (H4), sin gastar RNG extra en el camino
        baseline (posrandom_e4_entry=False no consume rng.uniform
        adicional -> reproduce bit-a-bit el camino de v6 en E0-E3;
        la única fuente de posible divergencia con v6.simulate es la
        posición inicial de E4 cuando posrandom_e4_entry=True).
    """
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
    # H4: series temporales de sumas (gated/instant) + registro por-par
    e_mutual_gated_series = []   # sum_t (gated)  == v6 E_mutual original
    e_mutual_instant_series = []  # sum_t (instant, sin gate de persistencia)
    n_mutual_gated_series = []
    n_mutual_instant_series = []
    per_pair_energy_gated_all = []    # cada entrada: energía de 1 par en 1 paso (gated)
    per_pair_energy_instant_all = []  # ídem, instant

    mutual_age = defaultdict(int)
    mutual_age_max = defaultdict(int)
    pair_co_steps = defaultdict(int)
    pair_possible_steps = 0
    prev_groups = []
    fusion_events = 0
    e4_steps = 0
    dens_last = 1.0
    n_groups_last = 0
    r_gyr_last = float("nan")
    n_atoms_last = 0

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

        ph = v6.medium_norm(Phi)
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
            v6.weighted_cut(ar, ad, Phi, nc, rng, p.ALPHA_CUT, blind=not p.medium_on)

        cl = v6.components_strict(phi, ar, ad, Phi)
        vev_ok = float(np.mean(np.abs(Phi))) >= VEV_POST_MIN
        atoms = [c for c in cl if c["is_atom"]] if (frozen and vev_ok) else []
        tracks, n_stable = v6.match_persist(atoms, tracks)
        stable_atoms = [t for t in tracks if t["age"] >= PERSIST_STEPS]

        E_bind = 0.0
        E_mutual_gated = 0.0
        E_mutual_instant = 0.0
        mean_r = 0.0
        n_groups = 0
        r_gyr = float("nan")
        dens_enhance = 1.0
        n_mutual_gated = 0
        n_mutual_instant = 0

        if e4_window and len(stable_atoms) >= 2:
            pos = np.array([[t["cy"], t["cx"]] for t in stable_atoms], dtype=float)
            masses = np.array([t["mass"] for t in stable_atoms], dtype=float)
            ids = np.array([t["id"] for t in stable_atoms], dtype=int)

            if nb_pos is None or len(nb_pos) != len(pos):
                if posrandom_e4_entry:
                    # H3: rompe la herencia espacial del campo -> posiciones U(0,L)^2
                    nb_pos = rng.uniform(0.0, L, size=pos.shape)
                else:
                    nb_pos = pos.copy()
                nb_mass = masses.copy()
                nb_ids = ids.copy()
                if p.grav_mode == "shuffle":
                    nb_perm = rng.permutation(len(masses))
                else:
                    nb_perm = None
                _, r0, _ = v6.groups_from_ids(nb_pos, nb_ids, L)
                r_gyr_start = r0 if r0 == r0 else None
                mutual_age = defaultdict(int)
                prev_groups = []
            else:
                nb_mass = masses
                id_to_pos = {int(i): nb_pos[k] for k, i in enumerate(nb_ids)}
                new_pos = []
                for tid in ids:
                    if int(tid) in id_to_pos:
                        new_pos.append(id_to_pos[int(tid)])
                    else:
                        idx = list(ids).index(tid)
                        new_pos.append(pos[idx])
                nb_pos = np.array(new_pos, dtype=float)
                nb_ids = ids.copy()

            nb_pos, E_bind, mean_r, n_close, close_pairs = nbody_step_cfg(
                nb_pos, nb_mass, nb_ids, p.G_GRAV, L, p.grav_mode, nb_perm,
                force_cutoff, softening,
            )

            id_to_newpos = {int(nb_ids[i]): nb_pos[i] for i in range(len(nb_ids))}
            for t in stable_atoms:
                if int(t["id"]) in id_to_newpos:
                    t["cy"], t["cx"] = (
                        float(id_to_newpos[int(t["id"])][0]),
                        float(id_to_newpos[int(t["id"])][1]),
                    )

            # --- edad de pares próximos (idéntico a v6) ---
            active_keys = set(close_pairs.keys())
            for key in list(mutual_age.keys()):
                if key not in active_keys:
                    mutual_age[key] = 0
            for key in active_keys:
                mutual_age[key] += 1
                mutual_age_max[key] = max(mutual_age_max[key], mutual_age[key])

            id_to_idx = {int(nb_ids[i]): i for i in range(len(nb_ids))}

            def pair_energy(pair_key):
                pair = list(pair_key)
                if len(pair) != 2:
                    return None
                if pair[0] not in id_to_idx or pair[1] not in id_to_idx:
                    return None
                i, j = id_to_idx[pair[0]], id_to_idx[pair[1]]
                dy = toroidal_delta(nb_pos[i, 0], nb_pos[j, 0], L)
                dx = toroidal_delta(nb_pos[i, 1], nb_pos[j, 1], L)
                r = float(np.hypot(dy, dx)) + 1e-9
                if r > force_cutoff:
                    return None
                r_soft = np.sqrt(r * r + softening**2)
                src_j = nb_mass[nb_perm[j]] if (p.grav_mode == "shuffle" and nb_perm is not None) else nb_mass[j]
                src_i = nb_mass[nb_perm[i]] if (p.grav_mode == "shuffle" and nb_perm is not None) else nb_mass[i]
                e = -0.5 * p.G_GRAV * (nb_mass[i] * src_j + nb_mass[j] * src_i) / r_soft
                return e

            # instant: TODOS los pares próximos de este paso, sin gate de edad (H4a)
            for key in active_keys:
                e = pair_energy(key)
                if e is None:
                    continue
                E_mutual_instant += e
                n_mutual_instant += 1
                per_pair_energy_instant_all.append(e)

            # gated: solo pares con edad >= mutual_min_steps (idéntico a v6)
            for key, age in mutual_age.items():
                if age < mutual_min_steps:
                    continue
                e = pair_energy(key)
                if e is None:
                    continue
                E_mutual_gated += e
                n_mutual_gated += 1
                per_pair_energy_gated_all.append(e)

            # --- linaje / co-membresía / fusión (idéntico a v6; no se usa para el
            #     observable reparado, solo se registra por completitud diagnóstica) ---
            id_groups, r_gyr, n_groups = v6.groups_from_ids(nb_pos, nb_ids, L)
            e4_steps += 1
            if len(nb_ids) >= 2:
                pair_possible_steps += 1
                for g in id_groups:
                    if len(g) < 2:
                        continue
                    gl = list(g)
                    for a_ in range(len(gl)):
                        for b_ in range(a_ + 1, len(gl)):
                            pair_co_steps[frozenset((gl[a_], gl[b_]))] += 1
                if prev_groups:
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

            rho_H = v6.deposit_rho(nb_pos, nb_mass, L)
            if rho_H.mean() > 0:
                dens_enhance = float(rho_H.max() / (rho_H.mean() + 1e-12))

            e_bind_acc.append(E_bind)
            e_mutual_gated_series.append(E_mutual_gated)
            e_mutual_instant_series.append(E_mutual_instant)
            n_mutual_gated_series.append(n_mutual_gated)
            n_mutual_instant_series.append(n_mutual_instant)

            dens_last = dens_enhance
            n_groups_last = n_groups
            r_gyr_last = r_gyr
            n_atoms_last = len(nb_mass)

        # época (solo para diagnóstico, no gatea nada aquí)
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

        if step % 20 == 0 or step == p.pasos - 1:
            hist.append({"step": step, "epoch": ep, "n_atoms_stable": n_stable})

    n_mut_stable = sum(1 for a in mutual_age_max.values() if a >= mutual_min_steps)
    if pair_possible_steps > 0 and pair_co_steps:
        comem_fracs = [v / pair_possible_steps for v in pair_co_steps.values() if v > 0]
        co_member_score = float(np.mean(comem_fracs)) if comem_fracs else 0.0
        n_long_co = sum(1 for v in pair_co_steps.values() if v >= mutual_min_steps)
    else:
        co_member_score = 0.0
        n_long_co = 0

    e3_ok = bool(hist) and max(h["n_atoms_stable"] for h in hist) >= 1

    return {
        "params": asdict(p),
        "force_cutoff": force_cutoff,
        "softening": softening,
        "mutual_min_steps": mutual_min_steps,
        "posrandom_e4_entry": posrandom_e4_entry,
        "E3_ok": e3_ok,
        "e4_steps": int(e4_steps),
        "e_mutual_gated_series": e_mutual_gated_series,
        "e_mutual_instant_series": e_mutual_instant_series,
        "n_mutual_gated_series": n_mutual_gated_series,
        "n_mutual_instant_series": n_mutual_instant_series,
        "per_pair_energy_gated_all": per_pair_energy_gated_all,
        "per_pair_energy_instant_all": per_pair_energy_instant_all,
        "co_member_score": float(co_member_score),
        "n_long_co_pairs": int(n_long_co),
        "fusion_events": int(fusion_events),
        "n_mutual_stable": int(n_mut_stable),
    }


def stats_from_sim(sim: dict) -> dict:
    """
    Deriva TODOS los estadísticos H4 de una corrida instrumentada, sin
    volver a simular. `mutual_bind_min_gated` reproduce exactamente la
    definición v6 (`mutual_bind = max(0,-min_t(E_mutual))`).
    """
    def series_stats(series):
        if not series:
            return {"min": 0.0, "mean": 0.0, "max": 0.0}
        arr = np.array(series, dtype=float)
        return {"min": float(arr.min()), "mean": float(arr.mean()), "max": float(arr.max())}

    g = series_stats(sim["e_mutual_gated_series"])
    inst = series_stats(sim["e_mutual_instant_series"])

    per_pair_g = sim["per_pair_energy_gated_all"]
    per_pair_i = sim["per_pair_energy_instant_all"]
    per_pair_mean_g = float(np.mean(per_pair_g)) if per_pair_g else 0.0
    per_pair_mean_i = float(np.mean(per_pair_i)) if per_pair_i else 0.0

    return {
        "min_gated": max(0.0, -g["min"]),        # == v6 mutual_bind original
        "mean_gated": max(0.0, -g["mean"]),
        "min_instant": max(0.0, -inst["min"]),
        "mean_instant": max(0.0, -inst["mean"]),
        "per_pair_mean_gated": max(0.0, -per_pair_mean_g),
        "per_pair_mean_instant": max(0.0, -per_pair_mean_i),
        "n_pair_terms_gated": len(per_pair_g),
        "n_pair_terms_instant": len(per_pair_i),
        "E3_ok": sim["E3_ok"],
        "co_member_score": sim["co_member_score"],
        "fusion_events": sim["fusion_events"],
    }


def rs(real_val, shuf_val, eps=1e-12):
    if shuf_val <= eps:
        return float("inf") if real_val > eps else float("nan")
    return real_val / shuf_val
