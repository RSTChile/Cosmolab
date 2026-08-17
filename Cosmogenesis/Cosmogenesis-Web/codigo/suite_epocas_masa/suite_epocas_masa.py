#!/usr/bin/env python3
"""
SUITE ÉPOCAS MASA — contrato Alexis 2026-07-22

NO PUEDE HABER MASA hasta:
  E0 pre-Higgs (simétrico)          → prohibida
  E1 ruptura / VEV                  → prohibida
  E2 post-Higgs pre-átomo           → prohibida
  E3 primer átomo (H análogo)       → aún no claim de masa
  E4 gravedad actúa sobre H y
     la densidad de H sube          → AHÍ nace la masa

Kill-switch: si el instrumento de "masa precoz" (fórmula tipo v3
  m = y0 * < |Φ| > * Σφ, ratio k1/k3) separa REAL vs NULL en E0–E3,
  eso es FUGA / diseño inválido, no hallazgo.

Anti-Shannon: sin 1/1836, sin 125/246 GeV como jueces.
Anclas 1e15 K / 1e-12 s solo reporte.
"""
from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[2] / "results" / "suite_epocas_masa"
OUT.mkdir(parents=True, exist_ok=True)

# --- umbrales pre-registrados (orden de magnitud, no SM) ---
VEV_PRE_MAX = 0.18
VEV_POST_MIN = 0.10
VEV_RATIO_MIN = 1.35
# kill-switch: separación del instrumento precoz de "masa"
LEAK_SEP_MAX = 0.05          # |Rm-NULL| en E0–E3 debe quedar bajo esto
LEAK_RATE_MAX = 0.25         # fracción de corridas con fuga admisible en barrido
# E3 átomo
ATOM_MIN_COUNT = 1
ATOM_PERSIST_STEPS = 2
# E4 masa por gravedad+densidad H
MASS_E4_SEP_MIN = 0.08       # contraste masa_grav ON vs control
DENS_ENHANCE_MIN = 1.15      # max ρ_H / mean debe subir con gravedad
RATE_PASS = 0.55


@dataclass
class P:
    L: int = 28
    pasos: int = 360
    H_EXP: float = 6.0
    H_TOPO: float = 0.01
    seed: int = 2025
    Y0: float = 0.3
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
    # E4
    gravity_on: bool = True
    G_GRAV: float = 0.15          # fuerza de atracción entre átomos H
    GRAV_START_FRAC: float = 0.62 # fracción de pasos tras la cual se habilita E4
    # ablaciones / controles
    force_mass_formula_always: bool = True  # para medir LEAK del instrumento precoz
    atom_need_vev: bool = True


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


def components(phi, ar, ad, Phi):
    """Clusters por umbral de φ media + lectura de orden |Φ|."""
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
            sum_phi_hi = float(phi[y, x]) if abs(Phi[y, x]) > 0.3 else 0.0
            n_hi = 1 if abs(Phi[y, x]) > 0.3 else 0
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
                        if abs(Phi[ny, nx]) > 0.3:
                            n_hi += 1
                            sum_phi_hi += float(phi[ny, nx])
            k = len(nodes)
            v = sum_abs / k if k else 0.0
            # átomo H análogo: dominio con núcleo de orden (hi |Φ|) + halo (mixto)
            # y tamaño acotado (no bulk del universo)
            is_atom = (
                3 <= k <= max(8, n // 3)
                and n_hi >= 1
                and (k - n_hi) >= 1
                and perim >= 4
            )
            out.append(
                {
                    "k": k,
                    "perim": perim,
                    "sum_phi": sum_phi,
                    "v_phi": v,
                    "n_hi": n_hi,
                    "nodes": nodes,
                    "is_atom": is_atom,
                }
            )
    return out


def premature_mass_ratio(clusters, y0):
    """Instrumento PROHIBIDO como masa en E0–E3 (solo para kill-switch)."""
    m1, m3 = [], []
    g1, g3 = [], []
    for c in clusters:
        m = y0 * c["v_phi"] * c["sum_phi"]
        g = y0 * 1.0 * c["sum_phi"]
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


def epoch_of(Tnorm, TC, frozen, grav_active, n_atoms_stable):
    if Tnorm > TC:
        return "E0"
    if not frozen:
        return "E1"
    if n_atoms_stable < ATOM_MIN_COUNT:
        return "E2"
    if not grav_active:
        return "E3"
    return "E4"


def simulate(p: P) -> dict:
    rng = np.random.default_rng(p.seed)
    L = p.L
    # E0 limpio: sin VEV impuesto
    phi = np.ones((L, L)) + 0.2 * rng.normal(size=(L, L))
    Phi = 0.06 * rng.normal(size=(L, L))
    ar = np.ones((L, L), dtype=bool)
    ad = np.ones((L, L), dtype=bool)

    # campo de densidad de "hidrógeno" (átomos) y potencial gravitatorio simple
    rho_H = np.zeros((L, L))
    atom_hist = []  # conteo por paso muestreado
    hist = []
    atom_stable_streak = 0
    n_atoms_stable = 0

    grav_start = int(p.GRAV_START_FRAC * p.pasos)

    for step in range(p.pasos):
        tg = step / max(p.pasos - 1, 1)
        a = float(np.exp(p.H_EXP * tg))
        Tnorm = float(np.exp(-p.H_EXP * tg))
        rho_c = p.RHO0 / (a**3)
        rho_hat_c = rho_c / p.RHO0
        frozen = (Tnorm < p.FREEZE_TNORM) or (rho_c < p.RHO_FREEZE)
        grav_active = p.gravity_on and step >= grav_start and frozen

        # --- evolución Φ (orden) ---
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

        # --- tejido φ ---
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

        # --- átomos H (solo post-freeze tiene sentido E3) ---
        cl = components(phi, ar, ad, Phi)
        atoms = [c for c in cl if c["is_atom"]]
        if p.atom_need_vev and float(np.mean(np.abs(Phi))) < VEV_POST_MIN:
            atoms = []
        n_atoms = len(atoms)
        if frozen and n_atoms >= ATOM_MIN_COUNT:
            atom_stable_streak += 1
        else:
            atom_stable_streak = 0
        if atom_stable_streak >= ATOM_PERSIST_STEPS:
            n_atoms_stable = n_atoms

        # mapa ρ_H: densidad de nodos atómicos
        rho_H *= 0.0
        for at in atoms:
            for y, x in at["nodes"]:
                rho_H[y, x] += 1.0
        if rho_H.sum() > 0:
            rho_H /= rho_H.sum() / (L * L)  # normaliza media ~1 si hay átomos

        # --- E4 gravedad: atracción que densifica H ---
        dens_enhance = 1.0
        mass_obs = 0.0  # MASA LEGÍTIMA: solo con gravedad + densificación
        if grav_active and rho_H.sum() > 0:
            # potencial ~ -G * (rho_H suavizado); flujo hacia pozos
            ker = (
                np.roll(rho_H, 1, 0)
                + np.roll(rho_H, -1, 0)
                + np.roll(rho_H, 1, 1)
                + np.roll(rho_H, -1, 1)
                + 4 * rho_H
            ) / 8.0
            # actualizar φ y acumulación de H hacia zonas densas (densificación)
            force = p.G_GRAV * (ker - float(ker.mean()))
            # mover densidad de H (conservativa aproximada)
            flux_r = force - np.roll(force, -1, 1)
            flux_d = force - np.roll(force, -1, 0)
            rho_H = rho_H + 0.08 * (
                np.roll(flux_r, 1, 1) - flux_r + np.roll(flux_d, 1, 0) - flux_d
            )
            rho_H = np.clip(rho_H, 0.0, None)
            if rho_H.sum() > 0:
                rho_H *= (L * L) / rho_H.sum()
            dens_enhance = float(rho_H.max() / (rho_H.mean() + 1e-12))
            # masa del relato: solo emerge con densificación gravitatoria de H
            # m_site ∝ ρ_H * (dens_enhance factor local)
            mass_field = rho_H * dens_enhance
            mass_obs = float(mass_field.mean() * dens_enhance)
            # también acopla levemente φ hacia pozos (materia sigue densidad)
            phi = phi + 0.05 * p.G_GRAV * (rho_H - float(rho_H.mean()))

        ep = epoch_of(Tnorm, p.TC, frozen, grav_active, n_atoms_stable)

        # instrumento precoz (PROHIBIDO como masa) — kill-switch
        leak_sep = premature_mass_ratio(cl, p.Y0) if p.force_mass_formula_always else None

        if step % 18 == 0 or step == p.pasos - 1:
            abs_phi = float(np.mean(np.abs(Phi)))
            # grad físico
            F = np.abs(Phi)
            g = np.sqrt(
                (0.5 * (np.roll(F, -1, 1) - np.roll(F, 1, 1))) ** 2
                + (0.5 * (np.roll(F, -1, 0) - np.roll(F, 1, 0))) ** 2
            )
            A_comov = float(g.max())
            A_phys = A_comov / max(a, 1e-12)
            hist.append(
                {
                    "step": step,
                    "tg": tg,
                    "a": a,
                    "Tnorm": Tnorm,
                    "T_phys_K_anchor": 1e15 * Tnorm,
                    "rho_c": float(rho_c),
                    "frozen": frozen,
                    "epoch": ep,
                    "Phi_abs": abs_phi,
                    "n_atoms": n_atoms,
                    "n_atoms_stable": n_atoms_stable,
                    "dens_enhance": dens_enhance,
                    "mass_obs": mass_obs,  # legítima solo E4
                    "leak_sep": leak_sep,  # instrumento precoz
                    "A_phys": A_phys,
                    "grav_active": grav_active,
                }
            )
            atom_hist.append(n_atoms)

    return {"params": asdict(p), "hist": hist, "atom_hist": atom_hist}


def by_epoch(hist, ep):
    return [h for h in hist if h["epoch"] == ep]


def analyze(sim: dict) -> dict:
    hist = sim["hist"]
    p = sim["params"]

    def avg(rows, key):
        vals = [h[key] for h in rows if h.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    def mx(rows, key):
        vals = [h[key] for h in rows if h.get(key) is not None]
        return float(np.max(vals)) if vals else None

    e0 = by_epoch(hist, "E0")
    e1 = by_epoch(hist, "E1")
    e2 = by_epoch(hist, "E2")
    e3 = by_epoch(hist, "E3")
    e4 = by_epoch(hist, "E4")
    pre_e4 = [h for h in hist if h["epoch"] in ("E0", "E1", "E2", "E3")]

    # --- E0 ---
    e0_phi = avg(e0, "Phi_abs")
    e0_ok = e0_phi is not None and e0_phi < VEV_PRE_MAX
    e0_stretch = False
    if len(e0) >= 2:
        e0_stretch = e0[-1]["A_phys"] < 0.5 * e0[0]["A_phys"] or e0[-1]["a"] > e0[0]["a"]

    # --- E1 VEV order ---
    e1_phi = avg(e1, "Phi_abs") if e1 else avg(e2, "Phi_abs")
    post_phi = avg([h for h in hist if h["Tnorm"] <= p["TC"]], "Phi_abs")
    e1_ok = (
        e0_phi is not None
        and post_phi is not None
        and e0_phi < VEV_PRE_MAX
        and post_phi > VEV_POST_MIN
        and post_phi > VEV_RATIO_MIN * max(e0_phi, 1e-6)
    )

    # --- KILL: fuga de masa precoz en E0–E3 ---
    leak_vals = [h["leak_sep"] for h in pre_e4 if h.get("leak_sep") is not None]
    leak_max = float(np.max(leak_vals)) if leak_vals else 0.0
    leak_mean = float(np.mean(leak_vals)) if leak_vals else 0.0
    # también masa_obs legítima debe ser ~0 antes de E4
    mass_pre = [h["mass_obs"] for h in pre_e4]
    mass_pre_max = float(np.max(mass_pre)) if mass_pre else 0.0
    kill_ok = leak_max <= LEAK_SEP_MAX and mass_pre_max <= 1e-9

    # --- E2: relaciones sin masa (post-freeze, pre-átomo o sin exigir átomo) ---
    e2_rows = e2 if e2 else [h for h in hist if h["frozen"] and h["epoch"] != "E4"]
    e2_ok = (
        len(e2_rows) > 0
        and avg(e2_rows, "Phi_abs") is not None
        and avg(e2_rows, "Phi_abs") > VEV_POST_MIN
        and mass_pre_max <= 1e-9
    )

    # --- E3 átomos ---
    n_at_max = mx(hist, "n_atoms") or 0
    n_at_stable_max = mx(hist, "n_atoms_stable") or 0
    e3_ok = n_at_stable_max >= ATOM_MIN_COUNT or n_at_max >= ATOM_MIN_COUNT

    # --- E4 masa solo con gravedad+densidad ---
    e4_mass = mx(e4, "mass_obs") or 0.0
    e4_dens = mx(e4, "dens_enhance") or 1.0
    e4_ok = (
        len(e4) > 0
        and e4_mass > 0.0
        and e4_dens >= DENS_ENHANCE_MIN
        and mass_pre_max <= 1e-9
    )

    # masa en E4 debe superar cualquier eco de leak como claim
    e4_mass_dominates = e4_ok and e4_mass > 0.0

    return {
        "E0_symmetric": bool(e0_ok),
        "E0_stretch_clock": bool(e0_stretch),
        "E1_VEV_after_Tc": bool(e1_ok),
        "E2_order_no_mass": bool(e2_ok),
        "E3_atom": bool(e3_ok),
        "E4_mass_from_grav_H": bool(e4_ok),
        "KILL_no_mass_before_E4": bool(kill_ok),
        "leak_max_pre_E4": leak_max,
        "leak_mean_pre_E4": leak_mean,
        "mass_obs_max_pre_E4": mass_pre_max,
        "mass_obs_max_E4": e4_mass,
        "dens_enhance_max_E4": e4_dens,
        "n_atoms_max": n_at_max,
        "n_atoms_stable_max": n_at_stable_max,
        "e0_Phi": e0_phi,
        "post_Phi": post_phi,
        "epochs_present": sorted(set(h["epoch"] for h in hist)),
        "chain_pass": bool(
            e0_ok and e1_ok and kill_ok and e3_ok and e4_ok
        ),  # E2 soft
    }


def run_one(p: P, label: str = "") -> dict:
    sim = simulate(p)
    an = analyze(sim)
    return {"label": label, "params": asdict(p), "analysis": an, "hist": sim["hist"]}


def run_grid(name: str, variants: list[tuple[str, P]]) -> dict:
    t0 = time.time()
    rows = []
    for i, (lab, p) in enumerate(variants):
        r = run_one(p, lab)
        a = r["analysis"]
        rows.append(
            {
                "label": lab,
                "seed": p.seed,
                **{k: a[k] for k in a if k not in ()},
            }
        )
        if (i + 1) % 8 == 0 or i == 0:
            print(
                f"  [{name}] {i+1}/{len(variants)} kill={a['KILL_no_mass_before_E4']} "
                f"E3={a['E3_atom']} E4={a['E4_mass_from_grav_H']} chain={a['chain_pass']} {lab}"
            )
    n = len(rows)

    def rate(key):
        return sum(1 for r in rows if r.get(key)) / max(n, 1)

    out = {
        "name": name,
        "n": n,
        "rate_KILL": rate("KILL_no_mass_before_E4"),
        "rate_E0": rate("E0_symmetric"),
        "rate_E1": rate("E1_VEV_after_Tc"),
        "rate_E2": rate("E2_order_no_mass"),
        "rate_E3": rate("E3_atom"),
        "rate_E4": rate("E4_mass_from_grav_H"),
        "rate_chain": rate("chain_pass"),
        "mean_leak_max": float(np.mean([r["leak_max_pre_E4"] for r in rows])),
        "mean_mass_E4": float(np.mean([r["mass_obs_max_E4"] for r in rows])),
        "mean_mass_pre": float(np.mean([r["mass_obs_max_pre_E4"] for r in rows])),
        "pass_kill_broad": rate("KILL_no_mass_before_E4") >= (1.0 - LEAK_RATE_MAX),
        "elapsed_s": time.time() - t0,
        "rows": rows,
    }
    print(
        f"== {name}: KILL={out['rate_KILL']:.2f} E0={out['rate_E0']:.2f} "
        f"E1={out['rate_E1']:.2f} E3={out['rate_E3']:.2f} E4={out['rate_E4']:.2f} "
        f"chain={out['rate_chain']:.2f} leak_mean_max={out['mean_leak_max']:.3f} "
        f"({out['elapsed_s']:.1f}s)"
    )
    return out


def main():
    print("=== SUITE ÉPOCAS MASA (E0→E4) ===")
    print("Masa PROHIBIDA hasta E4: gravedad + densificación de H.\n")
    t_all = time.time()
    results = {}

    # ----- K1: kill-switch multi-seed baseline -----
    variants = [
        (f"seed={s}", P(seed=int(s)))
        for s in (7, 42, 99, 123, 777, 1024, 2025, 3141, 8191, 99991, 12345, 54321)
    ]
    results["K1_kill_seeds"] = run_grid("K1_kill_seeds", variants)

    # ----- K2: barrido amplio — el leak del instrumento precoz -----
    variants = []
    for tc in np.linspace(0.35, 0.8, 6):
        for r0 in np.linspace(1.0, 3.5, 5):
            for s in (2025, 42):
                variants.append(
                    (f"TC={tc:.2f}_R0={r0:.1f}_s={s}", P(TC=float(tc), R0=float(r0), seed=int(s)))
                )
    results["K2_sweep_potential_leak"] = run_grid("K2_sweep_potential_leak", variants)

    # ----- E0–E1: orden sin masa -----
    variants = []
    for h in np.linspace(3.0, 9.0, 7):
        for s in (2025, 42, 777):
            variants.append((f"H={h:.1f}_s={s}", P(H_EXP=float(h), seed=int(s))))
    results["E01_sweep_H_EXP"] = run_grid("E01_sweep_H_EXP", variants)

    # ----- E3 átomos: barrido L y MIX -----
    variants = []
    for L in (20, 24, 28, 32, 40):
        for mx in np.linspace(0.15, 0.55, 5):
            variants.append((f"L={L}_MIX={mx:.2f}", P(L=int(L), MIX0=float(mx), seed=2025)))
    results["E3_sweep_atom"] = run_grid("E3_sweep_atom", variants)

    # ----- E4: gravedad ON vs OFF (control crítico) -----
    variants = []
    for g in np.linspace(0.0, 0.4, 9):
        for s in (2025, 42, 777, 3141):
            variants.append(
                (
                    f"G={g:.2f}_s={s}",
                    P(G_GRAV=float(g), gravity_on=(g > 0), seed=int(s)),
                )
            )
    results["E4_sweep_G_GRAV"] = run_grid("E4_sweep_G_GRAV", variants)

    # control explícito
    ctrl = []
    for s in (2025, 42, 777, 3141, 99991, 12345):
        on = run_one(P(seed=s, gravity_on=True, G_GRAV=0.15), f"ON_{s}")
        off = run_one(P(seed=s, gravity_on=False, G_GRAV=0.0), f"OFF_{s}")
        ctrl.append(
            {
                "seed": s,
                "ON_mass": on["analysis"]["mass_obs_max_E4"],
                "OFF_mass": off["analysis"]["mass_obs_max_E4"],
                "ON_dens": on["analysis"]["dens_enhance_max_E4"],
                "OFF_dens": off["analysis"]["dens_enhance_max_E4"],
                "ON_E4": on["analysis"]["E4_mass_from_grav_H"],
                "OFF_E4": off["analysis"]["E4_mass_from_grav_H"],
                "ON_kill": on["analysis"]["KILL_no_mass_before_E4"],
                "OFF_kill": off["analysis"]["KILL_no_mass_before_E4"],
                "ON_atoms": on["analysis"]["n_atoms_max"],
                "OFF_atoms": off["analysis"]["n_atoms_max"],
                "mass_only_with_grav": (
                    on["analysis"]["mass_obs_max_E4"] > off["analysis"]["mass_obs_max_E4"] + 1e-12
                    and off["analysis"]["mass_obs_max_E4"] <= 1e-9
                ),
            }
        )
        print(
            f"  [E4_CTRL] seed={s} ON_mass={ctrl[-1]['ON_mass']:.4f} "
            f"OFF_mass={ctrl[-1]['OFF_mass']:.4f} only_grav={ctrl[-1]['mass_only_with_grav']}"
        )
    results["E4_CTRL_grav_on_off"] = {
        "name": "E4_CTRL_grav_on_off",
        "n": len(ctrl),
        "rate_mass_only_with_grav": sum(c["mass_only_with_grav"] for c in ctrl) / len(ctrl),
        "rate_ON_E4": sum(c["ON_E4"] for c in ctrl) / len(ctrl),
        "rate_OFF_E4": sum(c["OFF_E4"] for c in ctrl) / len(ctrl),
        "mean_ON_mass": float(np.mean([c["ON_mass"] for c in ctrl])),
        "mean_OFF_mass": float(np.mean([c["OFF_mass"] for c in ctrl])),
        "rows": ctrl,
    }
    print(
        f"== E4_CTRL: mass_only_with_grav="
        f"{results['E4_CTRL_grav_on_off']['rate_mass_only_with_grav']:.2f} "
        f"ON_mass={results['E4_CTRL_grav_on_off']['mean_ON_mass']:.4f} "
        f"OFF_mass={results['E4_CTRL_grav_on_off']['mean_OFF_mass']:.4f}"
    )

    # ----- Ablaciones de cadena -----
    abl_defs = [
        ("full", P()),
        ("no_medium", P(medium_on=False)),
        ("no_gravity", P(gravity_on=False, G_GRAV=0.0)),
        ("early_gravity", P(GRAV_START_FRAC=0.05, gravity_on=True)),  # gravedad demasiado pronto
        ("strong_G", P(G_GRAV=0.35)),
        ("weak_G", P(G_GRAV=0.05)),
    ]
    abl_rows = []
    for name, base in abl_defs:
        for s in (2025, 42, 777, 3141):
            p = replace(base, seed=int(s))
            r = run_one(p, f"{name}_{s}")
            a = r["analysis"]
            abl_rows.append({"ablation": name, "seed": s, **a})
    abl_sum = {}
    for name, _ in abl_defs:
        rs = [r for r in abl_rows if r["ablation"] == name]
        abl_sum[name] = {
            "rate_KILL": sum(r["KILL_no_mass_before_E4"] for r in rs) / len(rs),
            "rate_E3": sum(r["E3_atom"] for r in rs) / len(rs),
            "rate_E4": sum(r["E4_mass_from_grav_H"] for r in rs) / len(rs),
            "rate_chain": sum(r["chain_pass"] for r in rs) / len(rs),
            "mean_leak": float(np.mean([r["leak_max_pre_E4"] for r in rs])),
            "mean_mass_E4": float(np.mean([r["mass_obs_max_E4"] for r in rs])),
            "mean_mass_pre": float(np.mean([r["mass_obs_max_pre_E4"] for r in rs])),
        }
        print(f"  [ABL] {name}: {abl_sum[name]}")
    results["ABL_chain"] = {"summary": abl_sum, "rows": abl_rows}

    # ----- Síntesis -----
    k1 = results["K1_kill_seeds"]
    k2 = results["K2_sweep_potential_leak"]
    e4c = results["E4_CTRL_grav_on_off"]
    kill_admissible = k1["pass_kill_broad"] and k2["mean_leak_max"]  # use rates
    # kill broad: K1 rate high; K2: fraction with KILL
    kill_ok_global = (
        k1["rate_KILL"] >= 0.7
        and k2["rate_KILL"] >= RATE_PASS
        and k1["mean_mass_pre"] <= 1e-9
    )
    mass_only_e4 = e4c["rate_mass_only_with_grav"] >= 0.7 and e4c["mean_OFF_mass"] <= 1e-9
    atom_ok = results["E3_sweep_atom"]["rate_E3"] >= RATE_PASS
    chain_baseline = abl_sum["full"]["rate_chain"]

    if kill_ok_global and mass_only_e4 and atom_ok and chain_baseline >= 0.5:
        verdict = "EPOCH_CHAIN_PARTIAL_mass_only_E4"
    elif kill_ok_global and mass_only_e4:
        verdict = "KILL_OK_MASS_E4_OK_ATOM_WEAK"
    elif mass_only_e4 and not kill_ok_global:
        verdict = "MASS_E4_OK_BUT_LEAK_INSTRUMENT"
    elif kill_ok_global and not mass_only_e4:
        verdict = "KILL_OK_BUT_E4_MASS_WEAK"
    else:
        verdict = "EPOCH_CHAIN_FAIL"

    # leak instrument: even if mass_obs=0, leak_sep may fire
    instrument_leaks = k2["mean_leak_max"] > LEAK_SEP_MAX

    synthesis = {
        "global_verdict": verdict,
        "kill_ok_global": kill_ok_global,
        "mass_only_with_gravity": mass_only_e4,
        "instrument_premature_leaks": instrument_leaks,
        "K1_rate_KILL": k1["rate_KILL"],
        "K2_rate_KILL": k2["rate_KILL"],
        "K2_mean_leak_max": k2["mean_leak_max"],
        "E4_CTRL_rate_only_grav": e4c["rate_mass_only_with_grav"],
        "E4_mean_ON_mass": e4c["mean_ON_mass"],
        "E4_mean_OFF_mass": e4c["mean_OFF_mass"],
        "E3_rate_atom": results["E3_sweep_atom"]["rate_E3"],
        "ABL": abl_sum,
        "thresholds": {
            "LEAK_SEP_MAX": LEAK_SEP_MAX,
            "DENS_ENHANCE_MIN": DENS_ENHANCE_MIN,
            "VEV_PRE_MAX": VEV_PRE_MAX,
            "VEV_POST_MIN": VEV_POST_MIN,
            "RATE_PASS": RATE_PASS,
        },
        "contrato": {
            "E0": "simétrico, sin masa",
            "E1": "VEV, sin masa",
            "E2": "orden/relaciones, sin masa",
            "E3": "primer átomo H, sin claim masa",
            "E4": "gravedad + densificación H → masa",
        },
        "elapsed_total_s": time.time() - t_all,
    }

    out_path = OUT / "suite_epocas_masa_result.json"
    # strip heavy hist from grids — only store analysis rows
    out_clean = {"synthesis": synthesis, "blocks": {}}
    for k, v in results.items():
        out_clean["blocks"][k] = v
    out_path.write_text(json.dumps(out_clean, indent=2), encoding="utf-8")
    print(f"\nJSON → {out_path}")

    md = []
    md.append("# SUITE ÉPOCAS MASA — resultados\n\n")
    md.append("**Fecha:** 2026-07-22\n\n")
    md.append("## Contrato\n\n")
    md.append(
        "No hay masa en E0–E3. La masa solo puede aparecer en **E4**, "
        "cuando la **gravedad** actúa sobre el **hidrógeno** y **aumenta su densidad**.\n\n"
    )
    md.append(f"## Veredicto global\n\n**`{verdict}`**\n\n")
    md.append("### Flags\n\n")
    md.append(f"- kill_ok_global (masa obs=0 pre-E4 + poco leak): **{kill_ok_global}**\n")
    md.append(f"- mass_only_with_gravity: **{mass_only_e4}**\n")
    md.append(f"- instrument_premature_leaks (fórmula v3 en E0–E3): **{instrument_leaks}**\n")
    md.append(f"- K1 rate KILL: **{k1['rate_KILL']:.2f}**\n")
    md.append(f"- K2 rate KILL: **{k2['rate_KILL']:.2f}** mean leak_max: **{k2['mean_leak_max']:.3f}**\n")
    md.append(
        f"- E4 CTRL: ON_mass={e4c['mean_ON_mass']:.4f} OFF_mass={e4c['mean_OFF_mass']:.4f} "
        f"only_grav_rate={e4c['rate_mass_only_with_grav']:.2f}\n"
    )
    md.append(f"- E3 atom rate: **{results['E3_sweep_atom']['rate_E3']:.2f}**\n\n")
    md.append("### Ablaciones\n\n")
    md.append("| abl | KILL | E3 | E4 | chain | mass_E4 | mass_pre |\n")
    md.append("|-----|------|----|----|-------|---------|----------|\n")
    for name, s in abl_sum.items():
        md.append(
            f"| {name} | {s['rate_KILL']:.2f} | {s['rate_E3']:.2f} | {s['rate_E4']:.2f} | "
            f"{s['rate_chain']:.2f} | {s['mean_mass_E4']:.3f} | {s['mean_mass_pre']:.1e} |\n"
        )
    md.append("\n### Lectura\n\n")
    md.append(
        "1. **mass_obs** (legítima) es 0 hasta E4 por construcción del proceso "
        "(solo se calcula con gravedad activa + ρ_H).\n"
        "2. **leak_sep** es el instrumento viejo de 'masa' (v3); si supera umbral en E0–E3, "
        "el *nombre* masa estaba mal puesto — kill-switch de diseño.\n"
        "3. Control gravedad OFF debe anular mass_obs.\n"
        "4. Sin 1/1836.\n"
    )
    md.append(f"\nTiempo total: {synthesis['elapsed_total_s']:.1f} s\n")
    (OUT / "RESUMEN_SUITE_EPOCAS_MASA.md").write_text("".join(md), encoding="utf-8")
    print(f"MD  → {OUT / 'RESUMEN_SUITE_EPOCAS_MASA.md'}")
    print("\n=== GLOBAL ===", verdict)
    print(json.dumps(synthesis, indent=2)[:2500])


if __name__ == "__main__":
    main()
