#!/usr/bin/env python3
"""
SUITE CRONO_HIGGS — cronología primero, luego barridos amplios (anti-Shannon)

Principio (Alexis, 2026-07-22):
  Antes de 10^{-12} s: T alta, campo Higgs simétrico V~0, sin masa, sin bosón.
  En ~10^{-12} s / T~T_c: ruptura electrodébil → VEV.
  DESPUÉS: masas por arrastre en el medio congelado; vibración del vacío = bosón.

  En el juguete: el claim tipo Higgs SOLO es admisible si la cronología se cumple.
  Los barridos prueban el ORDEN antes de interpretar masa/medio.

Reglas:
  - Sin gate 1/1836, sin sintonizar a un número del SM.
  - Barridos AMPLIOS de cada variable (familia pre-registrada), no cherry-pick.
  - Éxito de cronología = ventanas pre/post con umbrales fijos de orden de magnitud,
    no optimización.

Bloques:
  A1 CHRONO_VEV_ORDER     — <|Φ|> bajo pre-Tc, alto post-Tc
  A2 CHRONO_MASS_AFTER    — sep masa REAL-NULL solo post-SSB (no pre)
  A3 CHRONO_FLUC_AFTER    — fluctuaciones locales de Φ (tipo "esquirla") solo post-freeze
  B1 SWEEP_POTENTIAL      — R0, U, TC (familia potencial)
  B2 SWEEP_TRANSPORT      — D_PHI, SIGMA0, G_RHO
  B3 SWEEP_MEDIUM         — ALPHA_CUT, MIX0 (solo si A pasa en baseline)
  B4 SWEEP_COSMO          — H_EXP, RHO_FREEZE, L
  B5 ABLATION             — sin medio / sin ρ / sin freeze-T / sin freeze-ρ
  B6 ROBUST_SEEDS         — multi-seed en sello baseline

Mapeo de relato (solo reporte, no optimiza jueces):
  T_phys_anchor ~ 1e15 K * Tnorm; t_ewsb_anchor ~ 1e-12 s en cruce Tnorm~TC.
"""
from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[2] / "results" / "suite_crono_higgs"
OUT.mkdir(parents=True, exist_ok=True)

# Umbrales de ORDEN (pre-registrados; no 1/1836; no sintonía fina)
VEV_PRE_MAX = 0.20          # pre-Tc: <|Φ|> debe ser bajo
VEV_POST_MIN = 0.12         # post-Tc: <|Φ|> debe crecer (mínimo débil)
VEV_CONTRAST = 1.4          # post/pre ratio mínimo
SEP_THR = 0.06              # separación masa (algo más laxo que v3 para barrido)
SEP_PRE_MAX = 0.05          # pre-SSB: masa NO debe separar fuerte
FLUC_POST_MIN = 0.02        # std local post-freeze
FLUC_PRE_MAX = 0.08         # pre: o bien alta térmica caótica o distinta firma
RATE_PASS = 0.55            # fracción de grid que debe cumplir (barrido amplio, no 100%)


@dataclass
class Params:
    L: int = 28
    pasos: int = 320
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
    # ablaciones
    medium_on: bool = True
    rho_dynamic: bool = True
    use_freeze_T: bool = True
    use_freeze_rho: bool = True
    cut_blind: bool = False


def clusters_of(phi, ar, ad, Phi):
    media = phi.mean()
    visto = np.zeros_like(phi, dtype=bool)
    out = []
    n = phi.shape[0]
    for y in range(n):
        for x in range(n):
            if visto[y, x]:
                continue
            q = deque([(y, x)])
            visto[y, x] = True
            lado = phi[y, x] >= media
            sum_rho = float(phi[y, x])
            sum_abs = float(abs(Phi[y, x]))
            perim = 0
            k = 1
            while q:
                cy, cx = q.popleft()
                if (not ar[cy, cx]) or (phi[cy, (cx + 1) % n] >= media) != lado:
                    perim += 1
                if (not ar[cy, (cx - 1) % n]) or (phi[cy, (cx - 1) % n] >= media) != lado:
                    perim += 1
                if (not ad[cy, cx]) or (phi[(cy + 1) % n, cx] >= media) != lado:
                    perim += 1
                if (not ad[(cy - 1) % n, cx]) or (phi[(cy - 1) % n, cx] >= media) != lado:
                    perim += 1
                for ny, nx, cond in (
                    (cy, (cx + 1) % n, ar[cy, cx]),
                    (cy, (cx - 1) % n, ar[cy, (cx - 1) % n]),
                    ((cy + 1) % n, cx, ad[cy, cx]),
                    ((cy - 1) % n, cx, ad[(cy - 1) % n, cx]),
                ):
                    if cond and not visto[ny, nx] and (phi[ny, nx] >= media) == lado:
                        visto[ny, nx] = True
                        q.append((ny, nx))
                        sum_rho += float(phi[ny, nx])
                        sum_abs += float(abs(Phi[ny, nx]))
                        k += 1
            out.append(
                {
                    "k": k,
                    "perim": perim,
                    "sum_rho": sum_rho,
                    "v_phi": sum_abs / k if k else 0.0,
                }
            )
    return out


def mass_ratio(clusters, y0, use_phi_field: bool):
    m1, m3 = [], []
    for c in clusters:
        factor = c["v_phi"] if use_phi_field else 1.0
        m = y0 * factor * c["sum_rho"]
        if c["k"] == 1:
            m1.append(m)
        if c["k"] == 3 and c["perim"] == 8:
            m3.append(m)
    if not m1 or not m3:
        return None, 0, 0, 0.0, 0.0
    mk1, mk3 = float(np.mean(m1)), float(np.mean(m3))
    v1 = float(np.mean([c["v_phi"] for c in clusters if c["k"] == 1]))
    v3 = float(
        np.mean([c["v_phi"] for c in clusters if c["k"] == 3 and c["perim"] == 8])
    )
    return mk1 / (mk3 + 1e-30), len(m1), len(m3), v1, v3


def medium_norm(Phi):
    abs_phi = np.abs(Phi)
    p95 = float(np.percentile(abs_phi, 95)) + 1e-12
    return np.clip(abs_phi / p95, 0.0, 1.0)


def weighted_cut(ar, ad, Phi, nc, rng, alpha_cut, blind: bool):
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
                w = np.where(ar, (1.0 - edge + 1e-3) ** alpha_cut, 0.0)
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
                w = np.where(ad, (1.0 - edge + 1e-3) ** alpha_cut, 0.0)
                flat = w[tuple(idx.T)]
                p = None if flat.sum() <= 0 else flat / flat.sum()
            sel = rng.choice(len(idx), size=n_d, replace=False, p=p)
            for i in sel:
                ad[tuple(idx[i])] = False


def grad_A_phys(Phi, a):
    F = np.abs(Phi)
    dFx = 0.5 * (np.roll(F, -1, 1) - np.roll(F, 1, 1))
    dFy = 0.5 * (np.roll(F, -1, 0) - np.roll(F, 1, 0))
    g = np.sqrt(dFx**2 + dFy**2)
    n = F.shape[0]
    band = g[n // 8 : 7 * n // 8, n // 8 : 7 * n // 8]
    A_comov = float(band.max()) if band.size else 0.0
    return A_comov, A_comov / max(a, 1e-12)


def simulate(p: Params, sample_every: int = 20) -> dict:
    rng = np.random.default_rng(p.seed)
    L = p.L
    # CI: fase caliente casi simétrica (pequeña semilla, no VEV impuesto)
    phi = np.ones((L, L)) + 0.25 * rng.normal(size=(L, L))
    Phi = 0.08 * rng.normal(size=(L, L))  # SIN VEV inicial grande (cronología limpia)
    ar = np.ones((L, L), dtype=bool)
    ad = np.ones((L, L), dtype=bool)

    hist = []
    for step in range(p.pasos):
        tg = step / max(p.pasos - 1, 1)
        a = float(np.exp(p.H_EXP * tg))
        Tnorm = float(np.exp(-p.H_EXP * tg))
        rho = p.RHO0 if not p.rho_dynamic else p.RHO0 / (a**3)
        rho_hat_c = rho / p.RHO0

        frozen_T = p.use_freeze_T and (Tnorm < p.FREEZE_TNORM)
        frozen_rho = p.use_freeze_rho and (rho < p.RHO_FREEZE)
        frozen = frozen_T or frozen_rho

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
            D_eff = p.D_PHI * (rho_hat_c if p.rho_dynamic else 1.0)
            sig = p.SIGMA0 * np.sqrt(max(Tnorm, 1e-6))
            if p.rho_dynamic:
                sig *= np.sqrt(max(rho_hat_c, 1e-12))
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
        if p.medium_on and not p.cut_blind:
            mix_med = p.MIX_FLOOR + (p.MIX0 - p.MIX_FLOOR) * ph
        else:
            mix_med = np.full_like(phi, p.MIX0)
        if p.rho_dynamic:
            mix = mix_med * rho_hat_c + p.MIX_FLOOR_RHO * (1.0 - min(rho_hat_c, 1.0))
        else:
            mix = mix_med
        phi_new = phi.copy()
        msk = cnt > 0
        phi_new[msk] = phi[msk] + mix[msk] * (mean[msk] - phi[msk])
        phi = phi_new

        H_fis = p.H_TOPO * np.sqrt(Tnorm + 1e-12)
        tot = int(ar.sum() + ad.sum())
        nc = int(round(H_fis * tot))
        if nc > 0 and tot > 0:
            weighted_cut(
                ar,
                ad,
                Phi,
                nc,
                rng,
                p.ALPHA_CUT,
                blind=(p.cut_blind or not p.medium_on),
            )

        if step % sample_every == 0 or step == p.pasos - 1:
            cl = clusters_of(phi, ar, ad, Phi)
            Rm, n1, n3, v1, v3 = mass_ratio(cl, p.Y0, True)
            Rn, _, _, _, _ = mass_ratio(cl, p.Y0, False)
            abs_m = float(np.mean(np.abs(Phi)))
            std_m = float(np.std(Phi))
            # fluctuación local: std de Φ - media local 3x3 (proxy "esquirla"/excitación)
            loc = (
                np.roll(Phi, 1, 0)
                + np.roll(Phi, -1, 0)
                + np.roll(Phi, 1, 1)
                + np.roll(Phi, -1, 1)
            ) / 4.0
            fluc = float(np.std(Phi - loc))
            Ac, Ap = grad_A_phys(Phi, a)
            phase = (
                "pre_SSB"
                if Tnorm > p.TC
                else ("post_SSB_hot" if not frozen else "post_SSB_frozen")
            )
            hist.append(
                {
                    "step": step,
                    "tg": tg,
                    "a": a,
                    "Tnorm": Tnorm,
                    "rho": float(rho),
                    "frozen": frozen,
                    "phase": phase,
                    "Phi_abs": abs_m,
                    "Phi_std": std_m,
                    "fluc_local": fluc,
                    "Rm": Rm,
                    "Rn": Rn,
                    "sep": abs(Rm - Rn) if Rm is not None and Rn is not None else None,
                    "k1": n1,
                    "k3": n3,
                    "v1": v1,
                    "v3": v3,
                    "A_comov": Ac,
                    "A_phys": Ap,
                    "T_phys_K_anchor": 1e15 * Tnorm,
                }
            )

    return {"params": asdict(p), "hist": hist}


def window_stats(hist, phase_pred):
    rows = [h for h in hist if phase_pred(h)]
    if not rows:
        return None

    def avg(key):
        vals = [h[key] for h in rows if h.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "n": len(rows),
        "Phi_abs": avg("Phi_abs"),
        "fluc_local": avg("fluc_local"),
        "sep": avg("sep"),
        "Rm": avg("Rm"),
        "Rn": avg("Rn"),
        "A_phys": avg("A_phys"),
        "Tnorm_mean": avg("Tnorm"),
    }


def analyze_chrono(sim: dict) -> dict:
    hist = sim["hist"]
    p = sim["params"]
    pre = window_stats(hist, lambda h: h["Tnorm"] > p["TC"])
    post = window_stats(
        hist, lambda h: h["Tnorm"] <= p["TC"] and h["k1"] >= 3 and h.get("sep") is not None
    )
    frozen = window_stats(hist, lambda h: h["phase"] == "post_SSB_frozen")
    hot_post = window_stats(
        hist, lambda h: h["phase"] == "post_SSB_hot" and h.get("sep") is not None
    )

    # A1 VEV order
    vev_pre = pre["Phi_abs"] if pre else None
    vev_post = post["Phi_abs"] if post else None
    a1_ok = (
        vev_pre is not None
        and vev_post is not None
        and vev_pre < VEV_PRE_MAX
        and vev_post > VEV_POST_MIN
        and vev_post > VEV_CONTRAST * max(vev_pre, 1e-6)
    )

    # A2 mass after SSB: sep pre small, sep post large
    sep_pre = pre["sep"] if pre else None
    sep_post = post["sep"] if post else None
    # pre may have few clusters — allow None as weak pass if no mass window
    a2_ok = (
        sep_post is not None
        and sep_post > SEP_THR
        and (sep_pre is None or sep_pre < SEP_PRE_MAX or sep_post > 1.5 * sep_pre)
    )
    # fail if sep_pre already large (masa antes de SSB)
    a2_fail_early = sep_pre is not None and sep_pre > SEP_THR and (
        sep_post is None or sep_pre >= sep_post * 0.9
    )
    if a2_fail_early:
        a2_ok = False

    # A3 fluctuations: en frozen, fluc_local sobre fondo VEV (no requerimos LHC);
    # pre-SSB: o fluc alta térmica sin VEV, o distinta; exigimos VEV post y fluc post finita
    fluc_fr = frozen["fluc_local"] if frozen else None
    fluc_pre = pre["fluc_local"] if pre else None
    a3_ok = (
        vev_post is not None
        and vev_post > VEV_POST_MIN
        and fluc_fr is not None
        and fluc_fr > FLUC_POST_MIN
        and a1_ok  # bosón solo tiene sentido post-ruptura
    )

    chrono_pass = a1_ok and a2_ok and a3_ok
    return {
        "A1_VEV_order": bool(a1_ok),
        "A2_mass_after_SSB": bool(a2_ok),
        "A3_fluc_after_freeze": bool(a3_ok),
        "chrono_pass": bool(chrono_pass),
        "pre": pre,
        "post": post,
        "frozen": frozen,
        "hot_post": hot_post,
        "a2_fail_early_mass": bool(a2_fail_early) if sep_pre is not None else False,
    }


def run_grid(name: str, variants: list[Params], tag_fn) -> dict:
    t0 = time.time()
    rows = []
    for i, p in enumerate(variants):
        sim = simulate(p)
        an = analyze_chrono(sim)
        tags = tag_fn(p)
        row = {
            "i": i,
            "tags": tags,
            "chrono_pass": an["chrono_pass"],
            "A1": an["A1_VEV_order"],
            "A2": an["A2_mass_after_SSB"],
            "A3": an["A3_fluc_after_freeze"],
            "a2_fail_early": an["a2_fail_early_mass"],
            "pre_Phi": an["pre"]["Phi_abs"] if an["pre"] else None,
            "post_Phi": an["post"]["Phi_abs"] if an["post"] else None,
            "pre_sep": an["pre"]["sep"] if an["pre"] else None,
            "post_sep": an["post"]["sep"] if an["post"] else None,
            "frozen_fluc": an["frozen"]["fluc_local"] if an["frozen"] else None,
            "final_A_phys": sim["hist"][-1]["A_phys"],
            "final_a": sim["hist"][-1]["a"],
        }
        rows.append(row)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{name}] {i+1}/{len(variants)} chrono_pass={row['chrono_pass']} {tags}")

    n = len(rows)
    rate = lambda key: sum(1 for r in rows if r[key]) / max(n, 1)
    summary = {
        "name": name,
        "n": n,
        "rate_chrono": rate("chrono_pass"),
        "rate_A1": rate("A1"),
        "rate_A2": rate("A2"),
        "rate_A3": rate("A3"),
        "rate_early_mass_fail": rate("a2_fail_early"),
        "pass_broad": rate("chrono_pass") >= RATE_PASS,
        "elapsed_s": time.time() - t0,
        "rows": rows,
    }
    print(
        f"== {name}: rate_chrono={summary['rate_chrono']:.2f} "
        f"A1={summary['rate_A1']:.2f} A2={summary['rate_A2']:.2f} "
        f"A3={summary['rate_A3']:.2f} early_mass_fail={summary['rate_early_mass_fail']:.2f} "
        f"broad_pass={summary['pass_broad']} ({summary['elapsed_s']:.1f}s)"
    )
    return summary


def baseline(**kw) -> Params:
    return Params(**kw)


def main():
    print("=== SUITE CRONO_HIGGS ===")
    print("Orden obligatorio: cronología SSB antes de interpretar Higgs/masa.\n")
    print(
        f"umbrales: VEV_PRE_MAX={VEV_PRE_MAX} VEV_POST_MIN={VEV_POST_MIN} "
        f"SEP_THR={SEP_THR} SEP_PRE_MAX={SEP_PRE_MAX} RATE_PASS={RATE_PASS}"
    )
    print("CI: Φ~0.08·noise (sin VEV impuesto). ρ∝a⁻³, freeze T|ρ.\n")

    results = {}
    t_all = time.time()

    # ----- A: cronología con barridos de las variables que definen la ruptura -----
    variants = []
    for tc in np.linspace(0.25, 0.85, 9):
        for s in (2025, 42, 777):
            variants.append(baseline(TC=float(tc), seed=int(s)))
    results["A1_sweep_TC"] = run_grid(
        "A1_sweep_TC", variants, lambda p: {"TC": p.TC, "seed": p.seed}
    )

    variants = []
    for r0 in np.linspace(0.5, 4.0, 8):
        for s in (2025, 42, 777):
            variants.append(baseline(R0=float(r0), seed=int(s)))
    results["A1_sweep_R0"] = run_grid(
        "A1_sweep_R0", variants, lambda p: {"R0": p.R0, "seed": p.seed}
    )

    variants = []
    for u in np.linspace(0.15, 1.5, 7):
        for s in (2025, 42):
            variants.append(baseline(U=float(u), seed=int(s)))
    results["A1_sweep_U"] = run_grid(
        "A1_sweep_U", variants, lambda p: {"U": p.U, "seed": p.seed}
    )

    # A2 focused: mass early fail rate across H_EXP (how fast cooling)
    variants = []
    for h in np.linspace(3.0, 9.0, 7):
        for s in (2025, 42, 777):
            variants.append(baseline(H_EXP=float(h), seed=int(s)))
    results["A2_sweep_H_EXP"] = run_grid(
        "A2_sweep_H_EXP", variants, lambda p: {"H_EXP": p.H_EXP, "seed": p.seed}
    )

    # FREEZE thresholds
    variants = []
    for ft in np.linspace(0.15, 0.65, 6):
        for rf in np.geomspace(1e-3, 0.3, 5):
            variants.append(baseline(FREEZE_TNORM=float(ft), RHO_FREEZE=float(rf), seed=2025))
    results["A3_sweep_freeze"] = run_grid(
        "A3_sweep_freeze",
        variants,
        lambda p: {"FREEZE_TNORM": p.FREEZE_TNORM, "RHO_FREEZE": p.RHO_FREEZE},
    )

    # ----- B: transporte, medio, cosmología, ablación (interpretación post-crono) -----
    variants = []
    for d in np.geomspace(0.005, 0.25, 6):
        for sig in np.geomspace(0.02, 0.4, 5):
            variants.append(baseline(D_PHI=float(d), SIGMA0=float(sig), seed=2025))
    results["B2_sweep_transport"] = run_grid(
        "B2_sweep_transport",
        variants,
        lambda p: {"D_PHI": p.D_PHI, "SIGMA0": p.SIGMA0},
    )

    variants = []
    for g in np.linspace(0.0, 2.0, 6):
        for s in (2025, 42):
            variants.append(baseline(G_RHO=float(g), seed=int(s)))
    results["B2_sweep_G_RHO"] = run_grid(
        "B2_sweep_G_RHO", variants, lambda p: {"G_RHO": p.G_RHO, "seed": p.seed}
    )

    variants = []
    for ac in np.linspace(0.5, 5.0, 8):
        for mx in np.linspace(0.05, 0.7, 6):
            variants.append(baseline(ALPHA_CUT=float(ac), MIX0=float(mx), seed=2025))
    results["B3_sweep_medium"] = run_grid(
        "B3_sweep_medium",
        variants,
        lambda p: {"ALPHA_CUT": p.ALPHA_CUT, "MIX0": p.MIX0},
    )

    variants = []
    for L in (16, 20, 24, 28, 32, 40):
        for s in (2025, 42, 777):
            variants.append(baseline(L=int(L), seed=int(s)))
    results["B4_sweep_L"] = run_grid(
        "B4_sweep_L", variants, lambda p: {"L": p.L, "seed": p.seed}
    )

    # Ablaciones
    abl = [
        ("full", baseline()),
        ("no_medium", baseline(medium_on=False, cut_blind=True)),
        ("rho_fixed", baseline(rho_dynamic=False)),
        ("no_freeze_T", baseline(use_freeze_T=False)),
        ("no_freeze_rho", baseline(use_freeze_rho=False)),
        ("no_freeze_any", baseline(use_freeze_T=False, use_freeze_rho=False)),
        ("blind_cuts_only", baseline(cut_blind=True, medium_on=True)),
    ]
    # multi-seed per ablation
    variants = []
    tags_list = []
    for name, p0 in abl:
        for s in (2025, 42, 777, 3141):
            p = baseline(**{**asdict(p0), "seed": s})
            # fix bools from asdict
            variants.append(
                Params(
                    **{
                        **asdict(p0),
                        "seed": s,
                    }
                )
            )
            tags_list.append(name)

    def tag_abl(p):
        # recover by matching run order — store in closure index
        return {}

    # custom ablation runner
    print("  [B5_ablation] ...")
    t0 = time.time()
    abl_rows = []
    for name, p0 in abl:
        for s in (2025, 42, 777, 3141):
            p = Params(**{**asdict(p0), "seed": int(s)})
            sim = simulate(p)
            an = analyze_chrono(sim)
            abl_rows.append(
                {
                    "ablation": name,
                    "seed": s,
                    "chrono_pass": an["chrono_pass"],
                    "A1": an["A1_VEV_order"],
                    "A2": an["A2_mass_after_SSB"],
                    "A3": an["A3_fluc_after_freeze"],
                    "a2_fail_early": an["a2_fail_early_mass"],
                    "pre_Phi": an["pre"]["Phi_abs"] if an["pre"] else None,
                    "post_Phi": an["post"]["Phi_abs"] if an["post"] else None,
                    "pre_sep": an["pre"]["sep"] if an["pre"] else None,
                    "post_sep": an["post"]["sep"] if an["post"] else None,
                }
            )
    abl_by = {}
    for r in abl_rows:
        abl_by.setdefault(r["ablation"], []).append(r)
    abl_summary = {}
    for name, rs in abl_by.items():
        abl_summary[name] = {
            "n": len(rs),
            "rate_chrono": sum(r["chrono_pass"] for r in rs) / len(rs),
            "rate_A1": sum(r["A1"] for r in rs) / len(rs),
            "rate_A2": sum(r["A2"] for r in rs) / len(rs),
            "rate_A3": sum(r["A3"] for r in rs) / len(rs),
            "mean_post_sep": float(
                np.mean([r["post_sep"] for r in rs if r["post_sep"] is not None] or [0])
            ),
        }
        print(f"    ablation {name}: {abl_summary[name]}")
    results["B5_ablation"] = {
        "name": "B5_ablation",
        "summary": abl_summary,
        "rows": abl_rows,
        "elapsed_s": time.time() - t0,
    }

    # Robust seeds baseline
    variants = [baseline(seed=int(s)) for s in (
        7, 42, 99, 123, 777, 1024, 2025, 3141, 8191, 99991, 12345, 54321
    )]
    results["B6_robust_seeds"] = run_grid(
        "B6_robust_seeds", variants, lambda p: {"seed": p.seed}
    )

    # ----- Síntesis global -----
    chrono_blocks = [
        "A1_sweep_TC",
        "A1_sweep_R0",
        "A1_sweep_U",
        "A2_sweep_H_EXP",
        "A3_sweep_freeze",
    ]
    rates = {k: results[k]["rate_chrono"] for k in chrono_blocks}
    # Cronología admisible si la mayoría de barridos de A superan RATE_PASS
    n_ok = sum(1 for r in rates.values() if r >= RATE_PASS)
    chrono_admissible = n_ok >= 3  # al menos 3 de 5 familias de barrido

    # Medium effect from ablation
    full_a2 = abl_summary["full"]["rate_A2"]
    nomed_a2 = abl_summary["no_medium"]["rate_A2"]
    medium_matters = full_a2 > nomed_a2 + 0.15

    global_verdict = (
        "CHRONO_OK_THEN_MEDIUM_PARTIAL"
        if chrono_admissible and medium_matters
        else "CHRONO_OK_MEDIUM_WEAK"
        if chrono_admissible and not medium_matters
        else "CHRONO_FAIL_HIGGS_CLAIM_SUSPENDED"
        if not chrono_admissible
        else "INCONCLUSIVE"
    )

    synthesis = {
        "global_verdict": global_verdict,
        "chrono_admissible": chrono_admissible,
        "chrono_block_rates": rates,
        "n_chrono_blocks_pass": n_ok,
        "medium_matters_vs_blind": medium_matters,
        "ablation_rates": abl_summary,
        "B3_medium_rate_chrono": results["B3_sweep_medium"]["rate_chrono"],
        "B6_seed_rate_chrono": results["B6_robust_seeds"]["rate_chrono"],
        "thresholds": {
            "VEV_PRE_MAX": VEV_PRE_MAX,
            "VEV_POST_MIN": VEV_POST_MIN,
            "VEV_CONTRAST": VEV_CONTRAST,
            "SEP_THR": SEP_THR,
            "SEP_PRE_MAX": SEP_PRE_MAX,
            "RATE_PASS": RATE_PASS,
        },
        "cronologia_relato": {
            "pre": "Tnorm > TC ~ fase simétrica, sin VEV, sin claim de masa/Higgs",
            "ewsb": "Tnorm cruza TC → ruptura, VEV",
            "post": "post-freeze: masas por medio; fluc local = proxy excitación de vacío",
            "anclas_reporte": "T_phys ~ 1e15*K*Tnorm; t_ewsb relato ~1e-12 s (no optimiza)",
        },
        "elapsed_total_s": time.time() - t_all,
    }

    # compact results for JSON (drop huge if needed - rows kept)
    out = {
        "synthesis": synthesis,
        "blocks": {
            k: {
                kk: vv
                for kk, vv in results[k].items()
                if kk != "rows" or k.startswith("B5")
            }
            | (
                {"rows": results[k]["rows"]}
                if "rows" in results[k]
                else {}
            )
            for k in results
        },
    }
    # Actually store all rows - might be large but ok for few hundred
    out_path = OUT / "suite_crono_higgs_result.json"
    # build cleaner
    out_clean = {"synthesis": synthesis, "blocks": {}}
    for k, v in results.items():
        out_clean["blocks"][k] = v
    out_path.write_text(json.dumps(out_clean, indent=2), encoding="utf-8")
    print(f"\nJSON → {out_path}")

    md = []
    md.append("# SUITE CRONO_HIGGS — resultados\n\n")
    md.append("**Fecha:** 2026-07-22\n\n")
    md.append("## Principio\n\n")
    md.append(
        "Antes de la ruptura (T alta): campo simétrico, sin VEV, sin masa, sin bosón.\n"
        "En la ruptura: aparece VEV.\n"
        "**Después**: masas por arrastre en el medio; excitaciones del vacío ≈ bosón.\n"
        "Los barridos prueban el **orden** antes de cualquier claim tipo Higgs.\n\n"
    )
    md.append(f"## Veredicto global\n\n**`{global_verdict}`**\n\n")
    md.append(f"- chrono_admissible: **{chrono_admissible}** ({n_ok}/5 bloques A ≥ {RATE_PASS})\n")
    md.append(f"- medium_matters_vs_blind: **{medium_matters}**\n\n")
    md.append("### Rates cronología por barrido\n\n")
    md.append("| bloque | rate_chrono | rate_A1 | rate_A2 | rate_A3 | broad_pass |\n")
    md.append("|--------|-------------|---------|---------|---------|------------|\n")
    for k in results:
        if k == "B5_ablation":
            continue
        r = results[k]
        md.append(
            f"| {k} | {r['rate_chrono']:.2f} | {r['rate_A1']:.2f} | "
            f"{r['rate_A2']:.2f} | {r['rate_A3']:.2f} | {r.get('pass_broad')} |\n"
        )
    md.append("\n### Ablaciones (rate A2 = masa post-SSB)\n\n")
    md.append("| ablación | rate_chrono | rate_A1 | rate_A2 | mean_post_sep |\n")
    md.append("|----------|-------------|---------|---------|---------------|\n")
    for name, s in abl_summary.items():
        md.append(
            f"| {name} | {s['rate_chrono']:.2f} | {s['rate_A1']:.2f} | "
            f"{s['rate_A2']:.2f} | {s['mean_post_sep']:.4f} |\n"
        )
    md.append("\n## Lectura\n\n")
    md.append(
        "1. Si `CHRONO_FAIL_*`: **suspender** claim Higgs; el orden pre/post no es estable en el barrido.\n"
        "2. Si chrono OK y medium_matters: admisible el germen post-ruptura (PARTIAL), sin 1/1836.\n"
        "3. CI sin VEV impuesto (Φ ruido pequeño): la ruptura debe **emerger** al enfriar.\n"
        "4. Anclas 10^{15} K / 10^{-12} s son **relato/reporte**, no perillas de éxito.\n"
    )
    md.append(f"\nTiempo total: {synthesis['elapsed_total_s']:.1f} s\n")
    md_path = OUT / "RESUMEN_SUITE_CRONO_HIGGS.md"
    md_path.write_text("".join(md), encoding="utf-8")
    print(f"MD  → {md_path}")
    print("\n=== GLOBAL ===", global_verdict)
    print(json.dumps(synthesis, indent=2)[:2000])


if __name__ == "__main__":
    main()
