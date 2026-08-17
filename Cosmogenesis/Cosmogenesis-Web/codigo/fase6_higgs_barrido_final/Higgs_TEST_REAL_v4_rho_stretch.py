#!/usr/bin/env python3
"""
Higgs_TEST_REAL_v4_rho_stretch — v3 + eslabón densidad cosmológica / estiramiento

Antecedente:
  v1: VEV, sin señal de masa
  v2: muros Φ, tejido ciego → sin herencia
  v3: tejido ∝ |Φ| → ROBUST_PARTIAL_medium_coupling (sin ρ∝a⁻³ activa)
  TEST_RHO_DISPERSION: a↑ ⇒ ρ↓ ⇒ ∇_phys se suaviza; D∝ρ congela transporte

Delta v4 (sello NUEVO, pre-registrado; NO sintonía a 1/1836):
  1) Hereda v3: hard-freeze T + cortes/mezcla condicionados por |Φ|
  2) ρ_cosmo = ρ0 / a³ ACTIVA en:
       - difusión de Φ: D_eff = D_PHI * (ρ/ρ0)
       - ruido de Φ:    σ_eff = SIGMA0 * sqrt(Tnorm) * sqrt(ρ/ρ0)  (apaga con rarefacción)
       - mezcla de φ:   mix_eff = mix_v3 * (ρ/ρ0) + MIX_FLOOR_RHO * (1 - ρ/ρ0)_clipped
         (en rarefacto el transporte de φ se apaga; piso mínimo pre-registrado)
  3) Freeze de Φ: Tnorm < FREEZE_TNORM  OR  ρ_cosmo < RHO_FREEZE
  4) Lecturas nuevas: A_comov(Φ), A_phys(Φ)=A_comov/a, w_phys de muros
  5) Brazos:
       REAL           : medio ON + ρ(a)
       NULL_RHO_FIXED : medio ON + ρ≡ρ0 (sin rarefacción)
       NULL_NO_MEDIUM : ρ(a) ON pero cortes/mezcla ciegos a Φ (como v2)

Pregunta pre-registrada:
  ¿El germen medio→tejido→masa de v3 SOBREVICE cuando el medio se enrarece
  y los gradientes se estiran (A_phys colapsado), o era artefacto de malla
  sin rarefacción?

Éxito ≠ 1/1836.
Éxito = VEV + |Rm-NULL_geo| > SEP_THR + A_phys_final/A_phys_init < STRETCH_MAX
        (+ contraste REAL vs NULL_RHO_FIXED como lectura, no gate de sintonía)
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np

# --- sello v4 (fijo antes de ver resultados) ---
L = 30
PASOS = 400
H_EXP = 6.0
H_TOPO = 0.01
SEED = 2025
Y0 = 0.3
R0 = 2.0
U = 0.5
TC = 0.55
D_PHI = 0.05
DT_PHI = 0.08
SIGMA0 = 0.10
G_RHO = 0.8
FREEZE_TNORM = 0.40
RHO0 = 1.0
RHO_FREEZE = 0.05          # freeze por rarefacción (pre-registrado)
MIX0 = 0.35
MIX_FLOOR = 0.08
MIX_FLOOR_RHO = 0.02       # piso de mezcla en rarefacto extremo
ALPHA_CUT = 2.5
VEV_THR = 0.15
SEP_THR = 0.08
NULL_GEO_TOL = 0.08
HIER_THR = 0.10
WALL_FRAC_THR = 0.05
STRETCH_MAX = 0.25         # A_phys_final / A_phys_init (colapso físico)
INHERIT_THR = 0.08
SEEDS_ROBUST = (2025, 42, 777, 3141, 99991)

OUT_DIR = (
    Path(__file__).resolve().parents[2] / "results" / "fase6_higgs_barrido_final"
)


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
            nodes = [(y, x)]
            lado = phi[y, x] >= media
            sum_rho = float(phi[y, x])
            sum_abs = float(abs(Phi[y, x]))
            perim = 0
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
                        nodes.append((ny, nx))
                        sum_rho += float(phi[ny, nx])
                        sum_abs += float(abs(Phi[ny, nx]))
            k = len(nodes)
            v = sum_abs / k if k else 0.0
            out.append(
                {"k": k, "perim": perim, "sum_rho": sum_rho, "v_phi": v, "nodes": nodes}
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


def phi_structure_diagnostics(Phi):
    abs_phi = np.abs(Phi)
    mean_abs = float(np.mean(abs_phi))
    std_abs = float(np.std(abs_phi))
    p90 = float(np.percentile(abs_phi, 90))
    thr = 0.4 * max(p90, 1e-12)
    wall_frac = float(np.mean(abs_phi < thr))
    return {
        "Phi_abs_mean": mean_abs,
        "Phi_abs_std": std_abs,
        "Phi_abs_p90": p90,
        "wall_frac": wall_frac,
        "sign_frac_pos": float(np.mean(Phi > 0)),
        "contrast_std_over_mean": std_abs / (mean_abs + 1e-12),
    }


def phi_gradient_metrics(Phi, a: float) -> dict:
    """Abruptness comóvil/física del medio Φ (análogo TEST_RHO_DISPERSION)."""
    # gradiente de |Φ| (muros = caídas de orden)
    F = np.abs(Phi)
    dFx = 0.5 * (np.roll(F, -1, 1) - np.roll(F, 1, 1))
    dFy = 0.5 * (np.roll(F, -1, 0) - np.roll(F, 1, 0))
    g = np.sqrt(dFx**2 + dFy**2)
    n = F.shape[0]
    band = slice(n // 8, 7 * n // 8)
    g_band = g[band, band]
    A_comov = float(g_band.max()) if g_band.size else 0.0
    A_phys = A_comov / max(a, 1e-12)
    # ancho efectivo ~ inverso de abruptness comóvil normalizado
    mean_g = float(g_band.mean()) + 1e-12
    w_comov = 1.0 / (mean_g + 1e-12)
    w_phys = w_comov * a
    return {
        "A_comov": A_comov,
        "A_phys": A_phys,
        "w_comov": float(w_comov),
        "w_phys": float(w_phys),
        "g_mean_comov": mean_g,
    }


def medium_norm(Phi):
    abs_phi = np.abs(Phi)
    p95 = float(np.percentile(abs_phi, 95)) + 1e-12
    return np.clip(abs_phi / p95, 0.0, 1.0)


def weighted_cut_bonds(ar, ad, Phi, nc, rng, alpha_cut, blind: bool = False):
    """Corta nc enlaces. blind=True → pesos uniformes (sin |Φ|)."""
    if nc <= 0:
        return 0
    n_cut = 0
    ph = None if blind else medium_norm(Phi)

    if ar.any():
        idx = np.argwhere(ar)
        tot_live = int(ar.sum() + ad.sum())
        n_r = int(round(nc * float(ar.sum()) / max(tot_live, 1)))
        n_r = min(n_r, len(idx))
        if n_r > 0:
            if blind:
                p = None
            else:
                edge_r = 0.5 * (ph + np.roll(ph, -1, axis=1))
                w_r = np.where(ar, (1.0 - edge_r + 1e-3) ** alpha_cut, 0.0)
                flat_w = w_r[tuple(idx.T)]
                if flat_w.sum() <= 0:
                    p = None
                else:
                    p = flat_w / flat_w.sum()
            sel = rng.choice(len(idx), size=n_r, replace=False, p=p)
            for i in sel:
                ar[tuple(idx[i])] = False
            n_cut += n_r
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
                edge_d = 0.5 * (ph + np.roll(ph, -1, axis=0))
                w_d = np.where(ad, (1.0 - edge_d + 1e-3) ** alpha_cut, 0.0)
                flat_w = w_d[tuple(idx.T)]
                if flat_w.sum() <= 0:
                    p = None
                else:
                    p = flat_w / flat_w.sum()
            sel = rng.choice(len(idx), size=n_d, replace=False, p=p)
            for i in sel:
                ad[tuple(idx[i])] = False
            n_cut += n_d
    return n_cut


def run_arm(
    mode: str,
    seed: int = SEED,
    L_run: int = L,
    verbose: bool = False,
) -> dict:
    """
    mode:
      REAL           — medio ON, ρ=ρ0/a³
      NULL_RHO_FIXED — medio ON, ρ≡ρ0
      NULL_NO_MEDIUM — ρ(a), cortes/mezcla ciegos a Φ
    """
    assert mode in ("REAL", "NULL_RHO_FIXED", "NULL_NO_MEDIUM")
    medium_on = mode != "NULL_NO_MEDIUM"
    rho_dynamic = mode != "NULL_RHO_FIXED"

    rng = np.random.default_rng(seed)
    phi = np.ones((L_run, L_run)) + 0.3 * rng.normal(size=(L_run, L_run))
    Phi = 0.6 * np.sign(rng.normal(size=(L_run, L_run))) + 0.2 * rng.normal(
        size=(L_run, L_run)
    )
    ar = np.ones((L_run, L_run), dtype=bool)
    ad = np.ones((L_run, L_run), dtype=bool)

    hist = []
    for step in range(PASOS):
        tg = step / max(PASOS - 1, 1)
        a = float(np.exp(H_EXP * tg))
        Tnorm = float(np.exp(-H_EXP * tg))  # = 1/a
        rho = RHO0 if not rho_dynamic else RHO0 / (a**3)
        rho_hat_c = rho / RHO0

        frozen_T = Tnorm < FREEZE_TNORM
        frozen_rho = rho < RHO_FREEZE
        frozen = frozen_T or frozen_rho

        if not frozen:
            rho_hat = phi / (float(np.mean(phi)) + 1e-12)
            r_field = R0 * (Tnorm - TC) - G_RHO * (rho_hat - 1.0)
            lap = (
                np.roll(Phi, -1, 1)
                + np.roll(Phi, 1, 1)
                + np.roll(Phi, -1, 0)
                + np.roll(Phi, 1, 0)
                - 4 * Phi
            )
            dV = 2 * r_field * Phi + 4 * U * Phi**3
            D_eff = D_PHI * rho_hat_c
            sig = SIGMA0 * np.sqrt(max(Tnorm, 1e-6)) * np.sqrt(max(rho_hat_c, 1e-12))
            noise = sig * rng.normal(size=(L_run, L_run))
            Phi = Phi + DT_PHI * (-dV + D_eff * lap) + noise

        ph = medium_norm(Phi)

        # mezcla de φ
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
        if medium_on:
            mix_med = MIX_FLOOR + (MIX0 - MIX_FLOOR) * ph
        else:
            # ciego: mezcla uniforme base (sin estructura de Φ)
            mix_med = np.full_like(phi, MIX0)
        # rarefacción apaga transporte de φ
        mix = mix_med * rho_hat_c + MIX_FLOOR_RHO * (1.0 - min(rho_hat_c, 1.0))
        phi_new = phi.copy()
        mask = cnt > 0
        phi_new[mask] = phi[mask] + mix[mask] * (mean[mask] - phi[mask])
        phi = phi_new

        # cortes: H_fis con rarefacción suave (más frágil al enfriar; ρ no multiplica nc
        # para no reintroducir perilla de jerarquía — solo T como v3)
        H_fis = H_TOPO * np.sqrt(Tnorm + 1e-12)
        tot = int(ar.sum() + ad.sum())
        nc = int(round(H_fis * tot))
        if nc > 0 and tot > 0:
            weighted_cut_bonds(
                ar, ad, Phi, nc, rng, ALPHA_CUT, blind=not medium_on
            )

        if step % 50 == 0 or step == PASOS - 1:
            cl = clusters_of(phi, ar, ad, Phi)
            Rm, n1, n3, v1, v3 = mass_ratio(cl, Y0, use_phi_field=True)
            Rn, _, _, _, _ = mass_ratio(cl, Y0, use_phi_field=False)
            diag = phi_structure_diagnostics(Phi)
            gmet = phi_gradient_metrics(Phi, a)
            rec = {
                "step": step,
                "tg": tg,
                "a": a,
                "Tnorm": Tnorm,
                "rho": float(rho),
                "rho_hat_c": float(rho_hat_c),
                "frozen": frozen,
                "frozen_T": frozen_T,
                "frozen_rho": frozen_rho,
                "ratio_REAL": Rm,
                "ratio_NULL": Rn,
                "v_k1": v1,
                "v_k3": v3,
                "k1": n1,
                "k3": n3,
                "bonds_live": int(ar.sum() + ad.sum()),
                **diag,
                **{f"grad_{k}": v for k, v in gmet.items()},
            }
            # flatten grad keys without double prefix confusion
            rec["A_comov"] = gmet["A_comov"]
            rec["A_phys"] = gmet["A_phys"]
            rec["w_comov"] = gmet["w_comov"]
            rec["w_phys"] = gmet["w_phys"]
            hist.append(rec)
            if verbose:
                print(
                    f"[{mode}] step {step:3d} a={a:6.2f} ρ={rho:.2e} "
                    f"frz={int(frozen)} Aphys={gmet['A_phys']:.4f} "
                    f"<|Φ|>={diag['Phi_abs_mean']:.3f} "
                    f"Rm={Rm if Rm is not None else float('nan'):.3f} "
                    f"NULL={Rn if Rn is not None else float('nan'):.3f} "
                    f"k1={n1} k3={n3}"
                )

    late = [
        h
        for h in hist
        if h["Tnorm"] < TC
        and h["k1"] >= 5
        and h["k3"] >= 3
        and h["ratio_REAL"] is not None
        and h["ratio_NULL"] is not None
    ]

    def mean_key(key, default=0.0):
        if not late:
            return default
        vals = [h[key] for h in late if h.get(key) is not None]
        return float(np.mean(vals)) if vals else default

    Phi_ssb = mean_key("Phi_abs_mean")
    std_ssb = mean_key("Phi_abs_std")
    wall_ssb = mean_key("wall_frac")
    v1_ssb = mean_key("v_k1")
    v3_ssb = mean_key("v_k3")
    Rm_ssb = mean_key("ratio_REAL", default=None) if late else None
    if late:
        Rm_ssb = float(np.mean([h["ratio_REAL"] for h in late]))
        Null_ssb = float(np.mean([h["ratio_NULL"] for h in late]))
    else:
        Rm_ssb, Null_ssb = None, None
    sep = (
        abs(Rm_ssb - Null_ssb)
        if Rm_ssb is not None and Null_ssb is not None
        else None
    )
    contrast_v = abs(v1_ssb - v3_ssb) / (0.5 * (v1_ssb + v3_ssb) + 1e-12)

    init, final = hist[0], hist[-1]
    A_phys_ratio = final["A_phys"] / max(init["A_phys"], 1e-12)
    A_comov_ratio = final["A_comov"] / max(init["A_comov"], 1e-12)
    stretch_ok = A_phys_ratio < STRETCH_MAX and final["a"] > 50.0

    vev_ok = Phi_ssb > VEV_THR
    structure_ok = (std_ssb > 0.05 * Phi_ssb) or (wall_ssb > WALL_FRAC_THR)
    inherit_ok = contrast_v > INHERIT_THR
    signal_ok = sep is not None and sep > SEP_THR and vev_ok
    hierarchy = Rm_ssb is not None and Rm_ssb < HIER_THR
    # señal con medio ya estirado en físico
    signal_stretched_ok = bool(signal_ok and stretch_ok)

    if not late:
        verdict = "TEST_FAIL_no_window"
    elif not vev_ok:
        verdict = "TEST_FAIL_no_VEV"
    elif signal_stretched_ok and hierarchy:
        verdict = "TEST_PASS_higgs_rho_stretch"
    elif signal_stretched_ok:
        verdict = "TEST_PARTIAL_medium_with_stretch"
    elif signal_ok and not stretch_ok:
        verdict = "TEST_PARTIAL_signal_without_stretch"
    elif vev_ok and structure_ok and inherit_ok and not signal_ok:
        verdict = "TEST_FAIL_inherit_but_no_mass_signal"
    elif vev_ok and structure_ok and not inherit_ok:
        verdict = "TEST_FAIL_no_inheritance"
    else:
        verdict = "TEST_FAIL_or_inconclusive"

    return {
        "mode": mode,
        "seed": seed,
        "L": L_run,
        "verdict": verdict,
        "Phi_abs_mean_SSB": Phi_ssb,
        "wall_frac_SSB": wall_ssb,
        "v_k1_SSB": v1_ssb,
        "v_k3_SSB": v3_ssb,
        "contrast_v": contrast_v,
        "Rm_mean_SSB": Rm_ssb,
        "NULL_mean_SSB": Null_ssb,
        "separation_Rm_NULL": sep,
        "A_phys_init": init["A_phys"],
        "A_phys_final": final["A_phys"],
        "A_phys_ratio": A_phys_ratio,
        "A_comov_ratio": A_comov_ratio,
        "w_phys_final": final["w_phys"],
        "a_final": final["a"],
        "rho_final": final["rho"],
        "flags": {
            "vev_ok": bool(vev_ok),
            "structure_ok": bool(structure_ok),
            "inherit_ok": bool(inherit_ok),
            "signal_ok": bool(signal_ok),
            "stretch_ok": bool(stretch_ok),
            "signal_stretched_ok": bool(signal_stretched_ok),
            "hierarchy": bool(hierarchy),
        },
        "history": hist,
    }


def aggregate_verdict(results: dict) -> dict:
    """Compara brazos + multi-seed REAL."""
    real_runs = results["REAL_seeds"]
    n = len(real_runs)
    n_sig = sum(1 for r in real_runs if r["flags"]["signal_ok"])
    n_str = sum(1 for r in real_runs if r["flags"]["stretch_ok"])
    n_both = sum(1 for r in real_runs if r["flags"]["signal_stretched_ok"])
    seps = [r["separation_Rm_NULL"] for r in real_runs if r["separation_Rm_NULL"] is not None]
    sep_med = float(np.median(seps)) if seps else 0.0
    A_ratios = [r["A_phys_ratio"] for r in real_runs]

    ref = results["arms_seed2025"]
    real = ref["REAL"]
    rho_fix = ref["NULL_RHO_FIXED"]
    no_med = ref["NULL_NO_MEDIUM"]

    # contraste: REAL debe superar NULL_NO_MEDIUM en separación
    sep_real = real["separation_Rm_NULL"] or 0.0
    sep_blind = no_med["separation_Rm_NULL"] or 0.0
    medium_beats_blind = sep_real > sep_blind + 0.5 * SEP_THR

    # ρ fija vs dinámica: lectura (no kill)
    sep_rf = rho_fix["separation_Rm_NULL"] or 0.0
    rho_changes_signal = abs(sep_real - sep_rf) > 0.03

    rate_both = n_both / max(n, 1)
    rate_sig = n_sig / max(n, 1)

    if rate_both >= 0.7 and medium_beats_blind and sep_med > SEP_THR:
        label = "ROBUST_PARTIAL_higgs_with_rho_stretch"
    elif rate_sig >= 0.7 and rate_both < 0.7:
        label = "PARTIAL_signal_stretch_unstable"
    elif medium_beats_blind and sep_real > SEP_THR and real["flags"]["stretch_ok"]:
        label = "TEST_PARTIAL_medium_with_stretch"
    elif real["flags"]["stretch_ok"] and not real["flags"]["signal_ok"]:
        label = "TEST_FAIL_stretch_but_no_mass_signal"
    elif real["flags"]["signal_ok"] and not medium_beats_blind:
        label = "TEST_FAIL_signal_not_from_medium"
    else:
        label = "TEST_FAIL_or_weak"

    return {
        "verdict": label,
        "n_seeds": n,
        "rate_signal": rate_sig,
        "rate_stretch": n_str / max(n, 1),
        "rate_signal_stretched": rate_both,
        "sep_median_REAL": sep_med,
        "A_phys_ratio_median": float(np.median(A_ratios)),
        "sep_REAL_seed2025": sep_real,
        "sep_NULL_RHO_FIXED": sep_rf,
        "sep_NULL_NO_MEDIUM": sep_blind,
        "medium_beats_blind": bool(medium_beats_blind),
        "rho_changes_signal": bool(rho_changes_signal),
        "flags_REAL_2025": real["flags"],
        "thresholds": {
            "SEP_THR": SEP_THR,
            "STRETCH_MAX": STRETCH_MAX,
            "RHO_FREEZE": RHO_FREEZE,
            "rate_need": 0.7,
        },
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Higgs_TEST_REAL_v4_rho_stretch ===")
    print(
        f"sello L={L} H_EXP={H_EXP} RHO_FREEZE={RHO_FREEZE} "
        f"D∝ρ MIX∝ρ freeze T|{RHO_FREEZE} ALPHA_CUT={ALPHA_CUT}"
    )
    print("pregunta: ¿germen v3 sobrevive a rarefacción + estiramiento?\n")

    arms_2025 = {}
    for mode in ("REAL", "NULL_RHO_FIXED", "NULL_NO_MEDIUM"):
        print(f"--- brazo {mode} seed={SEED} ---")
        arms_2025[mode] = run_arm(mode, seed=SEED, verbose=True)
        r = arms_2025[mode]
        print(
            f"  → {r['verdict']}  sep={r['separation_Rm_NULL']}  "
            f"A_phys×{r['A_phys_ratio']:.4f}  "
            f"Rm={r['Rm_mean_SSB']} NULL={r['NULL_mean_SSB']}  "
            f"flags={r['flags']}\n"
        )

    print("=== multi-seed REAL ===")
    real_seeds = []
    for s in SEEDS_ROBUST:
        r = run_arm("REAL", seed=s, verbose=False)
        real_seeds.append(r)
        print(
            f"  seed={s:5d}  {r['verdict']:40s}  "
            f"sep={r['separation_Rm_NULL'] if r['separation_Rm_NULL'] is not None else float('nan'):.4f}  "
            f"Aphys×{r['A_phys_ratio']:.4f}  "
            f"sig={int(r['flags']['signal_ok'])} str={int(r['flags']['stretch_ok'])}"
        )

    payload_arms = {
        "arms_seed2025": {
            m: {k: v for k, v in arms_2025[m].items() if k != "history"}
            | {"history": arms_2025[m]["history"]}
            for m in arms_2025
        },
        "REAL_seeds": [
            {k: v for k, v in r.items() if k != "history"} for r in real_seeds
        ],
        # histories only for seed 2025 arms (size)
    }
    # rebuild for aggregate without stripping wrong
    agg_input = {
        "arms_seed2025": arms_2025,
        "REAL_seeds": real_seeds,
    }
    summary = aggregate_verdict(agg_input)

    print("\n=== VEREDICTO GLOBAL ===")
    print(summary["verdict"])
    print(json.dumps({k: v for k, v in summary.items() if k != "flags_REAL_2025"}, indent=2))
    print("flags REAL 2025:", summary["flags_REAL_2025"])

    out = {
        "version": "v4_rho_stretch",
        "design_delta": [
            "inherit_v3_medium_conditioned_tissue",
            "rho_cosmo_eq_rho0_over_a3_active",
            "D_Phi_and_noise_scale_with_rho",
            "mix_scales_with_rho",
            "freeze_T_or_rho",
            "report_A_phys_Phi_stretch",
            "arms_REAL_RHO_FIXED_NO_MEDIUM",
            "no_1_over_1836_gate",
        ],
        "sello": {
            "L": L,
            "PASOS": PASOS,
            "H_EXP": H_EXP,
            "RHO0": RHO0,
            "RHO_FREEZE": RHO_FREEZE,
            "FREEZE_TNORM": FREEZE_TNORM,
            "D_PHI": D_PHI,
            "ALPHA_CUT": ALPHA_CUT,
            "MIX0": MIX0,
            "MIX_FLOOR": MIX_FLOOR,
            "MIX_FLOOR_RHO": MIX_FLOOR_RHO,
            "SEP_THR": SEP_THR,
            "STRETCH_MAX": STRETCH_MAX,
            "SEEDS_ROBUST": list(SEEDS_ROBUST),
        },
        "summary": summary,
        "arms_seed2025": {
            m: {k: v for k, v in arms_2025[m].items() if k != "history"}
            for m in arms_2025
        },
        "arms_seed2025_history": {m: arms_2025[m]["history"] for m in arms_2025},
        "REAL_seeds": [
            {k: v for k, v in r.items() if k != "history"} for r in real_seeds
        ],
        "lectura": {
            "si_PASS_o_PARTIAL_stretch": (
                "El germen medio→masa sobrevive a rarefacción y estiramiento físico."
            ),
            "si_signal_without_stretch": (
                "Hay separación de masa pero el medio no se estiró: claim cosmológico débil."
            ),
            "si_stretch_but_no_signal": (
                "El estiramiento/ρ funcionan pero no producen masa ≠ geometría."
            ),
            "si_blind_igual_REAL": (
                "La señal no viene del medio Φ (no es mecanismo tipo Higgs)."
            ),
        },
    }

    out_json = OUT_DIR / "Higgs_TEST_REAL_v4_rho_stretch_result.json"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nJSON → {out_json}")

    # resumen MD
    s = summary
    md = []
    md.append("# Higgs_TEST_REAL_v4_rho_stretch — resultado\n\n")
    md.append("**Fecha:** 2026-07-22\n\n")
    md.append("## Pregunta\n\n")
    md.append(
        "¿El germen v3 (medio→tejido→masa) sobrevive cuando ρ∝a⁻³ está activa "
        "y los gradientes de Φ se estiran en físico?\n\n"
    )
    md.append(f"## Veredicto global\n\n**`{s['verdict']}`**\n\n")
    md.append("### Agregados multi-seed REAL\n\n")
    md.append(f"- rate señal: **{s['rate_signal']:.2f}** ({s['n_seeds']} seeds)\n")
    md.append(f"- rate stretch: **{s['rate_stretch']:.2f}**\n")
    md.append(f"- rate señal+stretch: **{s['rate_signal_stretched']:.2f}**\n")
    md.append(f"- sep mediana: **{s['sep_median_REAL']:.4f}** (umbral {SEP_THR})\n")
    md.append(f"- A_phys ratio mediana: **{s['A_phys_ratio_median']:.4f}**\n\n")
    md.append("### Brazos seed=2025\n\n")
    md.append("| brazo | sep Rm−NULL | A_phys× | Rm | stretch | signal |\n")
    md.append("|-------|-------------|---------|----|---------|--------|\n")
    for m in ("REAL", "NULL_RHO_FIXED", "NULL_NO_MEDIUM"):
        r = arms_2025[m]
        sep = r["separation_Rm_NULL"]
        md.append(
            f"| {m} | {sep if sep is not None else float('nan'):.4f} | "
            f"{r['A_phys_ratio']:.4f} | "
            f"{r['Rm_mean_SSB'] if r['Rm_mean_SSB'] is not None else float('nan'):.4f} | "
            f"{r['flags']['stretch_ok']} | {r['flags']['signal_ok']} |\n"
        )
    md.append("\n### Lectura\n\n")
    md.append(
        f"- medium_beats_blind: **{s['medium_beats_blind']}** "
        f"(sep REAL {s['sep_REAL_seed2025']:.4f} vs blind {s['sep_NULL_NO_MEDIUM']:.4f})\n"
        f"- rho_changes_signal: **{s['rho_changes_signal']}** "
        f"(RHO_FIXED sep {s['sep_NULL_RHO_FIXED']:.4f})\n"
    )
    md.append(
        "\nNo es claim SM/1/1836. Es: orden Φ + rarefacción + estiramiento "
        "→ ¿masa ≠ geometría?\n"
    )
    md.append("\n## Artefactos\n\n")
    md.append("- `codigo/fase6_higgs_barrido_final/Higgs_TEST_REAL_v4_rho_stretch.py`\n")
    md.append("- `results/fase6_higgs_barrido_final/Higgs_TEST_REAL_v4_rho_stretch_result.json`\n")
    out_md = OUT_DIR / "RESUMEN_TEST_REAL_v4_rho_stretch.md"
    out_md.write_text("".join(md), encoding="utf-8")
    print(f"MD  → {out_md}")


if __name__ == "__main__":
    main()
