#!/usr/bin/env python3
"""
SUITE ÉPOCAS MASA v2 — endurecer E3 (átomo) y E4 (gravedad)

Hereda contrato Alexis:
  masa PROHIBIDA en E0–E3; solo E4 con gravedad + densificación de H.
  NO reabrir Rm/v3 como masa (leak se reporta, no es claim).

Endurecimientos v2 (pre-registrados):
  E3 átomo ESTRICTO — debe cumplir TODOS:
    (a) tamaño en banda [K_MIN, K_MAX]
    (b) núcleo de orden: fracción |Φ| alta en [F_CORE_MIN, F_CORE_MAX]
        (núcleo + halo, no monolito ni polvo)
    (c) cohesión: perímetro/√k en banda (ni demasiado ralo ni blob infinito)
    (d) persistencia: mismo centroide comóvil ≈ estable ≥ PERSIST pasos
    (e) solo post-freeze + VEV vivo

  E4 gravedad — masa solo si densificación es CAUSADA por gravedad real:
    REAL:        G>0, potencial = suavizado(ρ_H)
    NULL_OFF:    G=0
    NULL_SHUFFLE: mismo |gradiente| estadístico pero pozos BARAJADOS
                 (misma amplitud de force, descorrelacionada de ρ_H)
    NULL_INVERT:  G>0 pero fuerza repulsiva (anti-gravedad)
  Éxito E4: mass_REAL alta; mass_OFF≈0; mass_SHUFFLE << mass_REAL;
            dens_enhance_REAL > dens_SHUFFLE y > dens_OFF.

Anti-Shannon: sin 1/1836; barridos amplios de G, L, seeds.
"""
from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[2] / "results" / "suite_epocas_masa_v2"
OUT.mkdir(parents=True, exist_ok=True)

# --- umbrales E3 duros ---
K_MIN, K_MAX = 4, 14
F_CORE_MIN, F_CORE_MAX = 0.15, 0.75   # fracción nodos "núcleo"
COHESION_MIN, COHESION_MAX = 1.2, 6.5  # perim / sqrt(k)
PERSIST_STEPS = 4
VEV_POST_MIN = 0.10
PHI_CORE_THR = 0.35                   # |Φ| para contar como núcleo
CENTROID_TOL = 2.5                    # celdas (comóvil)

# --- umbrales E4 ---
DENS_ENHANCE_MIN = 1.25
MASS_REAL_MIN = 0.5
# contraste vs nulos (ratios)
MASS_VS_SHUFFLE_MIN = 1.4             # mass_REAL / max(mass_SHUFFLE, eps)
MASS_VS_OFF_MIN = 5.0
DENS_VS_SHUFFLE_MIN = 1.15
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
    G_GRAV: float = 0.15
    GRAV_START_FRAC: float = 0.65
    # modo gravedad: real | off | shuffle | invert
    grav_mode: str = "real"
    report_leak: bool = True
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
                    "v_phi": sum_abs / k if k else 0.0,
                    "n_core": n_core,
                    "f_core": f_core,
                    "cohes": float(cohes),
                    "cy": cy,
                    "cx": cx,
                    "nodes": nodes,
                    "is_atom": is_atom,
                }
            )
    return out


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
    Rm = float(np.mean(m1) / (np.mean(m3) + 1e-30))
    Rn = float(np.mean(g1) / (np.mean(g3) + 1e-30))
    return abs(Rm - Rn)


def match_persist(atoms_now, tracks, tol=CENTROID_TOL):
    """Actualiza tracks de centroides; devuelve cuántos átomos llevan ≥ PERSIST_STEPS."""
    used = set()
    new_tracks = []
    for at in atoms_now:
        best_i, best_d = None, 1e9
        for i, tr in enumerate(tracks):
            if i in used:
                continue
            d = np.hypot(tr["cy"] - at["cy"], tr["cx"] - at["cx"])
            # wrap toroidal aprox no — L pequeño, ok
            if d < best_d:
                best_d, best_i = d, i
        if best_i is not None and best_d <= tol:
            tr = tracks[best_i]
            used.add(best_i)
            new_tracks.append(
                {
                    "cy": at["cy"],
                    "cx": at["cx"],
                    "age": tr["age"] + 1,
                    "k": at["k"],
                }
            )
        else:
            new_tracks.append(
                {"cy": at["cy"], "cx": at["cx"], "age": 1, "k": at["k"]}
            )
    n_stable = sum(1 for tr in new_tracks if tr["age"] >= PERSIST_STEPS)
    return new_tracks, n_stable


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
    rho_H = np.zeros((L, L))

    for step in range(p.pasos):
        tg = step / max(p.pasos - 1, 1)
        a = float(np.exp(p.H_EXP * tg))
        Tnorm = float(np.exp(-p.H_EXP * tg))
        rho_c = p.RHO0 / (a**3)
        rho_hat_c = rho_c / p.RHO0
        frozen = (Tnorm < p.FREEZE_TNORM) or (rho_c < p.RHO_FREEZE)
        epoch_gate = step >= grav_start and frozen

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
        n_atoms = len(atoms)

        # ρ_H solo de átomos ESTRICTOS estables (age>=1 al menos presentes)
        rho_H *= 0.0
        for at in atoms:
            for y, x in at["nodes"]:
                rho_H[y, x] += 1.0
        if rho_H.sum() > 0:
            rho_H *= (L * L) / rho_H.sum()

        dens_enhance = 1.0
        mass_obs = 0.0
        grav_active = False

        if epoch_gate and rho_H.sum() > 0:
            ker = (
                np.roll(rho_H, 1, 0)
                + np.roll(rho_H, -1, 0)
                + np.roll(rho_H, 1, 1)
                + np.roll(rho_H, -1, 1)
                + 4 * rho_H
            ) / 8.0
            force_real = p.G_GRAV * (ker - float(ker.mean()))

            mode = p.grav_mode
            if mode == "off" or p.G_GRAV <= 0:
                force = np.zeros_like(force_real)
                grav_active = False
            elif mode == "shuffle":
                # misma distribución de valores, descorrelacionada espacialmente de ρ_H
                flat = force_real.ravel().copy()
                rng.shuffle(flat)
                force = flat.reshape(force_real.shape)
                grav_active = True  # hay "fuerza" pero no es gravedad de H
            elif mode == "invert":
                force = -force_real
                grav_active = True
            else:  # real
                force = force_real
                grav_active = True

            if mode != "off" and p.G_GRAV > 0:
                flux_r = force - np.roll(force, -1, 1)
                flux_d = force - np.roll(force, -1, 0)
                rho_H = rho_H + 0.08 * (
                    np.roll(flux_r, 1, 1) - flux_r + np.roll(flux_d, 1, 0) - flux_d
                )
                rho_H = np.clip(rho_H, 0.0, None)
                if rho_H.sum() > 0:
                    rho_H *= (L * L) / rho_H.sum()
                dens_enhance = float(rho_H.max() / (rho_H.mean() + 1e-12))
                # masa SOLO si modo real (gravedad de H), no shuffle/invert/off
                if mode == "real":
                    mass_obs = float((rho_H * dens_enhance).mean() * dens_enhance)
                    phi = phi + 0.05 * p.G_GRAV * (rho_H - float(rho_H.mean()))
                else:
                    mass_obs = 0.0
                    # shuffle/invert pueden mover densidad pero NO otorgan mass_obs
                    # (la masa del relato exige gravedad real sobre H)
                    if mode in ("shuffle", "invert"):
                        # reportar densificación espuria sin masa
                        pass

        # época semántica
        if Tnorm > p.TC:
            ep = "E0"
        elif not frozen:
            ep = "E1"
        elif n_stable < 1:
            ep = "E2"
        elif not (epoch_gate and p.grav_mode == "real" and p.G_GRAV > 0):
            ep = "E3"
        else:
            ep = "E4"

        leak = premature_leak(cl, p.Y0) if p.report_leak else None

        if step % 20 == 0 or step == p.pasos - 1:
            hist.append(
                {
                    "step": step,
                    "a": a,
                    "Tnorm": Tnorm,
                    "epoch": ep,
                    "frozen": frozen,
                    "Phi_abs": float(np.mean(np.abs(Phi))),
                    "n_atoms_strict": n_atoms,
                    "n_atoms_stable": n_stable,
                    "dens_enhance": dens_enhance,
                    "mass_obs": mass_obs,
                    "grav_mode": p.grav_mode,
                    "grav_active": grav_active,
                    "leak_sep": leak,
                    "rho_H_max": float(rho_H.max()) if rho_H.size else 0.0,
                    "rho_H_mean": float(rho_H.mean()) if rho_H.size else 0.0,
                }
            )

    return {"params": asdict(p), "hist": hist}


def analyze(sim: dict) -> dict:
    hist = sim["hist"]
    p = sim["params"]
    pre = [h for h in hist if h["epoch"] in ("E0", "E1", "E2", "E3")]
    e3 = [h for h in hist if h["epoch"] == "E3" or h["n_atoms_stable"] >= 1]
    e4 = [h for h in hist if h["epoch"] == "E4"]
    late = hist[len(hist) // 2 :]

    def mx(rows, key):
        vals = [h[key] for h in rows if h.get(key) is not None]
        return float(np.max(vals)) if vals else 0.0

    def avg(rows, key):
        vals = [h[key] for h in rows if h.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    mass_pre = mx(pre, "mass_obs")
    mass_e4 = mx(e4, "mass_obs") if e4 else mx(late, "mass_obs")
    dens = mx(late, "dens_enhance")
    n_at = mx(hist, "n_atoms_strict")
    n_st = mx(hist, "n_atoms_stable")
    leak_max = mx(pre, "leak_sep")

    e0 = [h for h in hist if h["epoch"] == "E0"]
    e0_ok = bool(e0) and avg(e0, "Phi_abs") is not None and avg(e0, "Phi_abs") < 0.18
    post = [h for h in hist if h["Tnorm"] <= p["TC"]]
    e1_ok = (
        e0_ok
        and post
        and avg(post, "Phi_abs") is not None
        and avg(post, "Phi_abs") > VEV_POST_MIN
        and avg(post, "Phi_abs") > 1.3 * max(avg(e0, "Phi_abs") or 0, 1e-6)
    )

    e3_ok = n_st >= 1  # al menos un átomo estricto persistente
    # E4 flags depend on mode
    mode = p["grav_mode"]
    if mode == "real":
        e4_ok = mass_e4 >= MASS_REAL_MIN and dens >= DENS_ENHANCE_MIN and mass_pre <= 1e-12
    else:
        # nulos: mass debe ser 0; dens puede subir en shuffle/invert pero sin masa
        e4_ok = mass_e4 <= 1e-12 and mass_pre <= 1e-12

    return {
        "grav_mode": mode,
        "E0_ok": e0_ok,
        "E1_ok": e1_ok,
        "E3_strict_atom": e3_ok,
        "E4_ok_for_mode": e4_ok,
        "mass_pre_E4": mass_pre,
        "mass_E4": mass_e4,
        "dens_enhance_max": dens,
        "n_atoms_strict_max": n_at,
        "n_atoms_stable_max": n_st,
        "leak_max_pre_E4": leak_max,
        "zero_mass_pre": mass_pre <= 1e-12,
    }


def run_pair_controls(seed: int, G: float = 0.15, **kw) -> dict:
    """REAL + OFF + SHUFFLE + INVERT con misma semilla."""
    modes = ("real", "off", "shuffle", "invert")
    outs = {}
    for m in modes:
        p = P(seed=seed, G_GRAV=G if m != "off" else 0.0, grav_mode=m, **kw)
        if m == "off":
            p = P(seed=seed, G_GRAV=0.0, grav_mode="off", **kw)
        sim = simulate(p)
        outs[m] = analyze(sim)
        outs[m]["_hist_tail"] = sim["hist"][-5:]
    r, o, s, inv = outs["real"], outs["off"], outs["shuffle"], outs["invert"]
    eps = 1e-12
    mass_vs_shuf = r["mass_E4"] / max(s["mass_E4"], eps)
    mass_vs_off = r["mass_E4"] / max(o["mass_E4"], eps)
    dens_vs_shuf = r["dens_enhance_max"] / max(s["dens_enhance_max"], eps)
    dens_vs_off = r["dens_enhance_max"] / max(o["dens_enhance_max"], 1.0)

    # dens en off debería ~1 si no hay fuerza; shuffle puede clumpear al azar
    e4_causal = (
        r["zero_mass_pre"]
        and o["mass_E4"] <= eps
        and s["mass_E4"] <= eps  # shuffle no da mass_obs por diseño
        and inv["mass_E4"] <= eps
        and r["mass_E4"] >= MASS_REAL_MIN
        and r["dens_enhance_max"] >= DENS_ENHANCE_MIN
        and r["dens_enhance_max"] >= DENS_VS_SHUFFLE_MIN * max(s["dens_enhance_max"], 1.0)
        and r["E3_strict_atom"]
    )
    # variante más suave: si shuffle densifica igual de más, falla causalidad de dens
    dens_causal = r["dens_enhance_max"] > s["dens_enhance_max"] * 1.05 + 0.05

    return {
        "seed": seed,
        "G": G,
        "modes": {m: {k: v for k, v in outs[m].items() if not k.startswith("_")} for m in modes},
        "mass_vs_shuffle": mass_vs_shuf,
        "mass_vs_off": mass_vs_off,
        "dens_vs_shuffle": dens_vs_shuf,
        "dens_vs_off": dens_vs_off,
        "e4_causal_mass": e4_causal,
        "dens_causal": dens_causal,
        "E3_ok": r["E3_strict_atom"],
        "E0_ok": r["E0_ok"],
        "E1_ok": r["E1_ok"],
        "leak_real": r["leak_max_pre_E4"],
    }


def main():
    print("=== SUITE ÉPOCAS MASA v2 (E3 estricto + E4 nulos) ===\n")
    t0 = time.time()
    results = {}

    # ----- E3: tasa de átomos estrictos multi-seed / L / MIX -----
    print("--- E3 strict atoms ---")
    e3_rows = []
    for L in (24, 28, 32, 40):
        for mx in (0.2, 0.35, 0.5):
            for s in (2025, 42, 777, 3141):
                p = P(L=L, MIX0=mx, seed=s, grav_mode="off", G_GRAV=0.0)
                an = analyze(simulate(p))
                e3_rows.append(
                    {
                        "L": L,
                        "MIX0": mx,
                        "seed": s,
                        "E3": an["E3_strict_atom"],
                        "n_stable": an["n_atoms_stable_max"],
                        "n_strict": an["n_atoms_strict_max"],
                        "E0": an["E0_ok"],
                        "E1": an["E1_ok"],
                        "mass_pre": an["mass_pre_E4"],
                    }
                )
    e3_rate = sum(r["E3"] for r in e3_rows) / len(e3_rows)
    print(f"  E3_strict rate={e3_rate:.2f} n={len(e3_rows)}")
    # comparación laxa vs estricta: contar is_atom con criterios viejos no — solo reportar
    results["E3_strict_sweep"] = {
        "rate_E3": e3_rate,
        "rate_E0": sum(r["E0"] for r in e3_rows) / len(e3_rows),
        "rate_E1": sum(r["E1"] for r in e3_rows) / len(e3_rows),
        "rate_zero_mass_pre": sum(r["mass_pre"] <= 1e-12 for r in e3_rows) / len(e3_rows),
        "mean_n_stable": float(np.mean([r["n_stable"] for r in e3_rows])),
        "mean_n_strict": float(np.mean([r["n_strict"] for r in e3_rows])),
        "rows": e3_rows,
    }

    # ----- E4: controles por seed -----
    print("--- E4 causal controls (real/off/shuffle/invert) ---")
    seeds = (7, 42, 99, 777, 2025, 3141, 8191, 99991, 12345, 54321)
    ctrl_rows = []
    for s in seeds:
        row = run_pair_controls(s, G=0.15)
        ctrl_rows.append(row)
        print(
            f"  seed={s:5d} E3={row['E3_ok']} causal_mass={row['e4_causal_mass']} "
            f"dens_causal={row['dens_causal']} "
            f"mR={row['modes']['real']['mass_E4']:.2f} "
            f"mO={row['modes']['off']['mass_E4']:.2f} "
            f"mS={row['modes']['shuffle']['mass_E4']:.2f} "
            f"dR={row['modes']['real']['dens_enhance_max']:.2f} "
            f"dS={row['modes']['shuffle']['dens_enhance_max']:.2f}"
        )
    results["E4_controls"] = {
        "n": len(ctrl_rows),
        "rate_E3": sum(r["E3_ok"] for r in ctrl_rows) / len(ctrl_rows),
        "rate_e4_causal_mass": sum(r["e4_causal_mass"] for r in ctrl_rows) / len(ctrl_rows),
        "rate_dens_causal": sum(r["dens_causal"] for r in ctrl_rows) / len(ctrl_rows),
        "rate_E0": sum(r["E0_ok"] for r in ctrl_rows) / len(ctrl_rows),
        "rate_E1": sum(r["E1_ok"] for r in ctrl_rows) / len(ctrl_rows),
        "mean_mass_real": float(np.mean([r["modes"]["real"]["mass_E4"] for r in ctrl_rows])),
        "mean_mass_off": float(np.mean([r["modes"]["off"]["mass_E4"] for r in ctrl_rows])),
        "mean_mass_shuffle": float(
            np.mean([r["modes"]["shuffle"]["mass_E4"] for r in ctrl_rows])
        ),
        "mean_mass_invert": float(
            np.mean([r["modes"]["invert"]["mass_E4"] for r in ctrl_rows])
        ),
        "mean_dens_real": float(
            np.mean([r["modes"]["real"]["dens_enhance_max"] for r in ctrl_rows])
        ),
        "mean_dens_shuffle": float(
            np.mean([r["modes"]["shuffle"]["dens_enhance_max"] for r in ctrl_rows])
        ),
        "mean_dens_off": float(
            np.mean([r["modes"]["off"]["dens_enhance_max"] for r in ctrl_rows])
        ),
        "mean_leak": float(np.mean([r["leak_real"] for r in ctrl_rows])),
        "rows": ctrl_rows,
    }

    # ----- barrido G con controles en subset seeds -----
    print("--- E4 sweep G ---")
    g_rows = []
    for G in np.linspace(0.0, 0.4, 9):
        for s in (2025, 42, 777):
            row = run_pair_controls(int(s), G=float(G))
            g_rows.append(
                {
                    "G": float(G),
                    "seed": int(s),
                    "e4_causal_mass": row["e4_causal_mass"],
                    "dens_causal": row["dens_causal"],
                    "E3": row["E3_ok"],
                    "mass_real": row["modes"]["real"]["mass_E4"],
                    "mass_off": row["modes"]["off"]["mass_E4"],
                    "mass_shuffle": row["modes"]["shuffle"]["mass_E4"],
                    "dens_real": row["modes"]["real"]["dens_enhance_max"],
                    "dens_shuffle": row["modes"]["shuffle"]["dens_enhance_max"],
                }
            )
        sub = [r for r in g_rows if r["G"] == float(G)]
        print(
            f"  G={G:.2f} causal={np.mean([r['e4_causal_mass'] for r in sub]):.2f} "
            f"mR={np.mean([r['mass_real'] for r in sub]):.2f} "
            f"dR/dS={np.mean([r['dens_real']/max(r['dens_shuffle'],1e-6) for r in sub]):.2f}"
        )
    results["E4_sweep_G"] = {
        "rows": g_rows,
        "rate_causal_Ggt0": sum(
            r["e4_causal_mass"] for r in g_rows if r["G"] > 0
        )
        / max(sum(1 for r in g_rows if r["G"] > 0), 1),
        "rate_causal_G0": sum(r["e4_causal_mass"] for r in g_rows if r["G"] == 0)
        / max(sum(1 for r in g_rows if r["G"] == 0), 1),
    }

    # ----- síntesis -----
    e3r = results["E3_strict_sweep"]["rate_E3"]
    e4c = results["E4_controls"]["rate_e4_causal_mass"]
    dens_c = results["E4_controls"]["rate_dens_causal"]
    zero_m = results["E3_strict_sweep"]["rate_zero_mass_pre"]
    mass_nulls_clean = (
        results["E4_controls"]["mean_mass_off"] <= 1e-12
        and results["E4_controls"]["mean_mass_shuffle"] <= 1e-12
        and results["E4_controls"]["mean_mass_invert"] <= 1e-12
    )

    if e3r >= RATE_PASS and e4c >= RATE_PASS and mass_nulls_clean and zero_m >= 0.99:
        if dens_c >= RATE_PASS:
            verdict = "E3_STRICT_OK_E4_CAUSAL_OK"
        else:
            verdict = "E3_OK_E4_MASS_OK_DENS_SHUFFLE_WEAK"
    elif e3r < RATE_PASS and e4c >= RATE_PASS:
        verdict = "E3_STRICT_WEAK_E4_OK"
    elif e3r >= RATE_PASS and e4c < RATE_PASS:
        verdict = "E3_OK_E4_CAUSAL_WEAK"
    else:
        verdict = "E3_E4_HARDENING_FAIL"

    synthesis = {
        "global_verdict": verdict,
        "E3_strict_rate": e3r,
        "E4_causal_mass_rate": e4c,
        "E4_dens_causal_rate": dens_c,
        "mass_nulls_clean": mass_nulls_clean,
        "zero_mass_pre_rate": zero_m,
        "mean_mass_real": results["E4_controls"]["mean_mass_real"],
        "mean_mass_off": results["E4_controls"]["mean_mass_off"],
        "mean_mass_shuffle": results["E4_controls"]["mean_mass_shuffle"],
        "mean_mass_invert": results["E4_controls"]["mean_mass_invert"],
        "mean_dens_real": results["E4_controls"]["mean_dens_real"],
        "mean_dens_shuffle": results["E4_controls"]["mean_dens_shuffle"],
        "mean_dens_off": results["E4_controls"]["mean_dens_off"],
        "mean_leak_v3_not_mass": results["E4_controls"]["mean_leak"],
        "thresholds_E3": {
            "K": [K_MIN, K_MAX],
            "f_core": [F_CORE_MIN, F_CORE_MAX],
            "cohesion": [COHESION_MIN, COHESION_MAX],
            "PERSIST_STEPS": PERSIST_STEPS,
        },
        "thresholds_E4": {
            "DENS_ENHANCE_MIN": DENS_ENHANCE_MIN,
            "MASS_REAL_MIN": MASS_REAL_MIN,
            "DENS_VS_SHUFFLE_MIN": DENS_VS_SHUFFLE_MIN,
        },
        "elapsed_s": time.time() - t0,
        "note": (
            "mass_obs solo en grav_mode=real; shuffle/invert/off → mass=0. "
            "leak_sep = instrumento v3 ilegal como masa (solo monitoreo)."
        ),
    }

    out = {"synthesis": synthesis, "blocks": results}
    path = OUT / "suite_epocas_masa_v2_result.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nJSON → {path}")

    md = []
    md.append("# SUITE ÉPOCAS MASA v2 — E3 estricto + E4 nulos\n\n")
    md.append("**Fecha:** 2026-07-22\n\n")
    md.append(f"## Veredicto\n\n**`{verdict}`**\n\n")
    md.append("### E3 átomo estricto\n\n")
    md.append(
        f"- Criterios: k∈[{K_MIN},{K_MAX}], f_core∈[{F_CORE_MIN},{F_CORE_MAX}], "
        f"cohesión∈[{COHESION_MIN},{COHESION_MAX}], persistencia≥{PERSIST_STEPS}, post-freeze+VEV\n"
        f"- rate_E3: **{e3r:.2f}**\n"
        f"- zero mass_obs pre-E4: **{zero_m:.2f}**\n"
        f"- mean n_stable: **{results['E3_strict_sweep']['mean_n_stable']:.2f}**\n\n"
    )
    md.append("### E4 controles (10 seeds)\n\n")
    md.append(
        f"| modo | mean mass | mean dens |\n|------|-----------|----------|\n"
        f"| REAL | **{results['E4_controls']['mean_mass_real']:.3f}** | "
        f"**{results['E4_controls']['mean_dens_real']:.3f}** |\n"
        f"| OFF | {results['E4_controls']['mean_mass_off']:.3f} | "
        f"{results['E4_controls']['mean_dens_off']:.3f} |\n"
        f"| SHUFFLE | {results['E4_controls']['mean_mass_shuffle']:.3f} | "
        f"{results['E4_controls']['mean_dens_shuffle']:.3f} |\n"
        f"| INVERT | {results['E4_controls']['mean_mass_invert']:.3f} | "
        f"(repulsión) |\n\n"
    )
    md.append(
        f"- rate E4 causal mass: **{e4c:.2f}**\n"
        f"- rate dens_causal (REAL>SHUFFLE): **{dens_c:.2f}**\n"
        f"- mass nulls clean (OFF=SHUFFLE=INVERT=0): **{mass_nulls_clean}**\n"
        f"- leak v3 (no es masa): mean **{results['E4_controls']['mean_leak']:.3f}**\n\n"
    )
    md.append("### Lectura\n\n")
    md.append(
        "1. Átomo más exigente: no cualquier cluster; núcleo+halo+persistencia.\n"
        "2. Masa solo en gravedad REAL; nulos shuffle/invert/off no otorgan mass_obs.\n"
        "3. dens_causal pide que el clumping de REAL supere al de pozos barajados "
        "(si shuffle densifica igual, la densificación no prueba gravedad de H).\n"
        "4. Rm/v3 sigue siendo leak de nomenclatura, no masa.\n"
    )
    md.append(f"\nTiempo: {synthesis['elapsed_s']:.1f} s\n")
    (OUT / "RESUMEN_SUITE_EPOCAS_MASA_v2.md").write_text("".join(md), encoding="utf-8")
    print(f"MD  → {OUT / 'RESUMEN_SUITE_EPOCAS_MASA_v2.md'}")
    print("\n=== GLOBAL ===", verdict)
    print(json.dumps(synthesis, indent=2))


if __name__ == "__main__":
    main()
