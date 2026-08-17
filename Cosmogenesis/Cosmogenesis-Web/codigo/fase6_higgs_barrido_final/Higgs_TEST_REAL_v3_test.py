"""
Higgs_TEST_REAL_v3 — tejido condicionado por el medio |Φ|

Antecedente:
  v1: VEV uniforme → sin señal de masa
  v2 hard-freeze: muros en Φ vivos, pero k1/k3 muestreat al azar → sin señal

Delta de diseño v3 (sello NUEVO, pre-registrado; no sintonía a 1/1836):
  1) Hard-freeze de Φ en frío (hereda v2).
  2) Rotura de enlaces ar/ad ponderada por |Φ| del borde:
     medio débil (muro, |Φ| bajo) → más fácil cortar;
     medio fuerte (bulk, |Φ| alto) → más cohesivo.
     Sin if k, sin gate 1/1836.
  3) Mezcla de densidad φ con tasa local ∝ fuerza del medio
     (bulk difunde más; muros aíslan más).

Pregunta pre-registrada (igual v1/v2):
  ¿VEV + fórmula única m = y0 * factor * sum_ρ produce
  separación REAL vs NULL geométrico (~1/3)?

Éxito ≠ 1/1836.
Éxito = VEV vivo + |Rm - NULL| > SEP_THR (+ opcional jerarquía).
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np

# --- sello v3 (fijo antes de ver resultados) ---
L = 30
PASOS = 400
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
ALPHA_CUT = 2.5       # peso de corte ~ (1 - |Φ|_borde)^ALPHA_CUT
MIX0 = 0.35           # tasa base de mezcla de φ
MIX_FLOOR = 0.08      # mezcla mínima en muros
VEV_THR = 0.15
SEP_THR = 0.08
NULL_GEO_TOL = 0.08
HIER_THR = 0.10
WALL_FRAC_THR = 0.05


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
    """Fórmula ÚNICA: m = y0 * factor * sum_rho.
    REAL: factor = < |Phi| >_dom
    NULL geométrico: factor = 1
    """
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
    sign_frac_pos = float(np.mean(Phi > 0))
    return {
        "Phi_abs_mean": mean_abs,
        "Phi_abs_std": std_abs,
        "Phi_abs_p90": p90,
        "wall_frac": wall_frac,
        "sign_frac_pos": sign_frac_pos,
        "contrast_std_over_mean": std_abs / (mean_abs + 1e-12),
    }


def medium_norm(Phi):
    """Fuerza del medio en [0,1] por sitio (robusto a escala)."""
    abs_phi = np.abs(Phi)
    p95 = float(np.percentile(abs_phi, 95)) + 1e-12
    return np.clip(abs_phi / p95, 0.0, 1.0)


def weighted_cut_bonds(ar, ad, Phi, nc, rng, alpha_cut):
    """Corta nc enlaces con probabilidad ∝ (1 - fuerza_borde)^alpha.
    Muros (medio débil) se rompen antes que el bulk.
    """
    if nc <= 0:
        return 0
    ph = medium_norm(Phi)
    n_cut = 0

    # ar: enlace horizontal (y,x) — (y,x+1)
    if ar.any():
        edge_r = 0.5 * (ph + np.roll(ph, -1, axis=1))
        w_r = np.where(ar, (1.0 - edge_r + 1e-3) ** alpha_cut, 0.0)
        idx = np.argwhere(ar)
        weights = w_r[ar]
        n_r = min(nc, len(idx))
        # reparto proporcional a enlaces vivos ar vs total (misma idea que v1/v2)
        tot_live = int(ar.sum() + ad.sum())
        n_r = int(round(nc * float(ar.sum()) / max(tot_live, 1)))
        n_r = min(n_r, len(idx))
        if n_r > 0 and weights.sum() > 0:
            p = weights / weights.sum()
            # choice sobre índices lineales de idx
            flat_w = w_r[tuple(idx.T)]
            p = flat_w / flat_w.sum()
            sel = rng.choice(len(idx), size=n_r, replace=False, p=p)
            for i in sel:
                ar[tuple(idx[i])] = False
            n_cut += n_r
        rem = nc - n_r
    else:
        rem = nc

    if rem > 0 and ad.any():
        edge_d = 0.5 * (ph + np.roll(ph, -1, axis=0))
        w_d = np.where(ad, (1.0 - edge_d + 1e-3) ** alpha_cut, 0.0)
        idx = np.argwhere(ad)
        n_d = min(rem, len(idx))
        if n_d > 0:
            flat_w = w_d[tuple(idx.T)]
            if flat_w.sum() > 0:
                p = flat_w / flat_w.sum()
                sel = rng.choice(len(idx), size=n_d, replace=False, p=p)
                for i in sel:
                    ad[tuple(idx[i])] = False
                n_cut += n_d
    return n_cut


def run(L_run: int | None = None, seed: int | None = None, verbose: bool = True):
    """Corre un experimento. L y seed opcionales (default = sello de módulo)."""
    L_use = int(L if L_run is None else L_run)
    seed_use = int(SEED if seed is None else seed)
    rng = np.random.default_rng(seed_use)
    phi = np.ones((L_use, L_use)) + 0.3 * rng.normal(size=(L_use, L_use))
    Phi = 0.6 * np.sign(rng.normal(size=(L_use, L_use))) + 0.2 * rng.normal(
        size=(L_use, L_use)
    )
    ar = np.ones((L_use, L_use), dtype=bool)
    ad = np.ones((L_use, L_use), dtype=bool)

    hist = []
    for step in range(PASOS):
        tg = step / PASOS
        a = float(np.exp(6 * tg))
        Tnorm = float(np.exp(-6 * tg))
        frozen = Tnorm < FREEZE_TNORM

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
            noise = SIGMA0 * np.sqrt(max(Tnorm, 1e-6)) * rng.normal(size=(L_use, L_use))
            Phi = Phi + DT_PHI * (-dV + D_PHI * lap) + noise

        ph = medium_norm(Phi)

        # mezcla de densidad con tasa local ∝ fuerza del medio
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
        mix = MIX_FLOOR + (MIX0 - MIX_FLOOR) * ph
        phi_new = phi.copy()
        mask = cnt > 0
        phi_new[mask] = phi[mask] + mix[mask] * (mean[mask] - phi[mask])
        phi = phi_new

        # rotura de enlaces condicionada por el medio
        H_fis = H_TOPO * np.sqrt(Tnorm + 1e-12)
        tot = int(ar.sum() + ad.sum())
        nc = int(round(H_fis * tot))
        if nc > 0 and tot > 0:
            weighted_cut_bonds(ar, ad, Phi, nc, rng, ALPHA_CUT)

        if step % 50 == 0 or step == PASOS - 1:
            cl = clusters_of(phi, ar, ad, Phi)
            Rm, n1, n3, v1, v3 = mass_ratio(cl, Y0, use_phi_field=True)
            Rn, _, _, _, _ = mass_ratio(cl, Y0, use_phi_field=False)
            diag = phi_structure_diagnostics(Phi)
            rec = {
                "step": step,
                "a": a,
                "Tnorm": Tnorm,
                "frozen": frozen,
                "ratio_REAL": Rm,
                "ratio_NULL": Rn,
                "v_k1": v1,
                "v_k3": v3,
                "k1": n1,
                "k3": n3,
                "bonds_live": int(ar.sum() + ad.sum()),
                **diag,
            }
            hist.append(rec)
            if verbose:
                print(
                    f"step {step:3d} a={a:6.2f} frz={int(frozen)} "
                    f"<|Phi|>={diag['Phi_abs_mean']:.4f} wall={diag['wall_frac']:.3f} "
                    f"v1={v1:.4f} v3={v3:.4f} "
                    f"Rm={Rm if Rm is not None else float('nan'):.4f} "
                    f"NULL={Rn if Rn is not None else float('nan'):.4f} "
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

    Phi_ssb = float(np.mean([h["Phi_abs_mean"] for h in late])) if late else 0.0
    std_ssb = float(np.mean([h["Phi_abs_std"] for h in late])) if late else 0.0
    wall_ssb = float(np.mean([h["wall_frac"] for h in late])) if late else 0.0
    v1_ssb = float(np.mean([h["v_k1"] for h in late])) if late else 0.0
    v3_ssb = float(np.mean([h["v_k3"] for h in late])) if late else 0.0
    Rm_ssb = float(np.mean([h["ratio_REAL"] for h in late])) if late else None
    Null_ssb = float(np.mean([h["ratio_NULL"] for h in late])) if late else None
    sep = abs(Rm_ssb - Null_ssb) if Rm_ssb is not None and Null_ssb is not None else None
    contrast_v = abs(v1_ssb - v3_ssb) / (0.5 * (v1_ssb + v3_ssb) + 1e-12)

    vev_ok = Phi_ssb > VEV_THR
    structure_ok = (std_ssb > 0.05 * Phi_ssb) or (wall_ssb > WALL_FRAC_THR)
    null_ok = Null_ssb is not None and abs(Null_ssb - 1.0 / 3.0) < NULL_GEO_TOL
    # en v3 el NULL sigue siendo factor=1 (geometría de dominios);
    # si el tejido condicionado cambia tamaños, NULL puede alejarse de 1/3
    # — eso es OK: la señal es REAL vs NULL bajo la misma topología.
    signal_ok = sep is not None and sep > SEP_THR and vev_ok
    hierarchy = Rm_ssb is not None and Rm_ssb < HIER_THR
    inherit_ok = contrast_v > 0.08  # dominios heredan contraste del medio

    if not late:
        verdict = "TEST_FAIL_no_window"
    elif not vev_ok:
        verdict = "TEST_FAIL_no_VEV"
    elif vev_ok and not structure_ok and not signal_ok:
        verdict = "TEST_FAIL_VEV_still_uniform"
    elif vev_ok and structure_ok and not inherit_ok and not signal_ok:
        verdict = "TEST_FAIL_no_inheritance"
    elif vev_ok and structure_ok and inherit_ok and not signal_ok:
        verdict = "TEST_FAIL_inherit_but_no_mass_signal"
    elif signal_ok and not hierarchy:
        verdict = "TEST_PARTIAL_medium_coupling"
    elif signal_ok and hierarchy:
        verdict = "TEST_PASS_higgs_like"
    else:
        verdict = "TEST_INCONCLUSIVE"

    notes = {
        "TEST_FAIL_no_VEV": "Sin VEV: no se testeo el medio.",
        "TEST_FAIL_VEV_still_uniform": "VEV sin contraste espacial ni señal.",
        "TEST_FAIL_no_inheritance": (
            "Hay muros en Φ pero el tejido condicionado no hace que v_k1 y v_k3 difieran: "
            "los dominios aún no heredan el contraste del medio."
        ),
        "TEST_FAIL_inherit_but_no_mass_signal": (
            "v_k1 ≠ v_k3 (herencia) pero |Rm-NULL| aún bajo el umbral: "
            "el contraste no basta para separar masas bajo la fórmula única."
        ),
        "TEST_PARTIAL_medium_coupling": (
            "VEV + separación Rm vs NULL: germen de mecanismo tipo medio; "
            "jerarquía no fuerte. Éxito parcial v3 (sin exigir 1/1836)."
        ),
        "TEST_PASS_higgs_like": (
            "VEV + señal vs NULL + Rm estructurado. Mecanismo tipo Higgs operativo en el juguete."
        ),
        "TEST_FAIL_no_window": "Sin ventana con k1 y k3 suficientes.",
    }

    return {
        "version": "v3_tissue_conditioned_by_medium",
        "design_delta": [
            "hard_freeze_Phi_from_v2",
            "bond_cutting_weighted_by_local_abs_Phi",
            "density_mixing_rate_proportional_to_medium",
            "no_1_over_1836_gate",
            "unique_mass_formula",
        ],
        "verdict": verdict,
        "note": notes.get(verdict, ""),
        "Phi_abs_mean_SSB": Phi_ssb,
        "Phi_abs_std_SSB": std_ssb,
        "wall_frac_SSB": wall_ssb,
        "v_k1_SSB": v1_ssb,
        "v_k3_SSB": v3_ssb,
        "contrast_v_k1_k3": contrast_v,
        "Rm_mean_SSB": Rm_ssb,
        "NULL_mean_SSB": Null_ssb,
        "separation_Rm_NULL": sep,
        "flags": {
            "vev_ok": vev_ok,
            "structure_ok": bool(structure_ok),
            "inherit_ok": bool(inherit_ok),
            "null_ok": bool(null_ok),
            "signal_ok": bool(signal_ok),
            "hierarchy_Rm_lt_0.1": bool(hierarchy),
        },
        "constants": {
            "L": L_use,
            "PASOS": PASOS,
            "R0": R0,
            "U": U,
            "TC": TC,
            "D_PHI": D_PHI,
            "SIGMA0": SIGMA0,
            "G_RHO": G_RHO,
            "FREEZE_TNORM": FREEZE_TNORM,
            "ALPHA_CUT": ALPHA_CUT,
            "MIX0": MIX0,
            "MIX_FLOOR": MIX_FLOOR,
            "freeze_mode": "hard_lock_Phi",
            "Y0": Y0,
            "SEED": seed_use,
            "VEV_THR": VEV_THR,
            "SEP_THR": SEP_THR,
        },
        "history": hist,
    }


def main():
    print("=== Higgs_TEST_REAL_v3 (tejido condicionado por medio) ===")
    print("Pregunta: VEV + formula unica + tejido(Φ) => masa != geometria?")
    print(
        f"sello ALPHA_CUT={ALPHA_CUT} MIX0={MIX0} G_RHO={G_RHO} "
        f"freeze_T={FREEZE_TNORM} Y0={Y0} seed={SEED}"
    )
    out = run()
    print("\n=== VEREDICTO ===")
    print(out["verdict"])
    print(out["note"])
    print(
        f"<|Phi|>={out['Phi_abs_mean_SSB']:.4f} wall={out['wall_frac_SSB']:.3f} "
        f"v1={out['v_k1_SSB']:.4f} v3={out['v_k3_SSB']:.4f} "
        f"contrast_v={out['contrast_v_k1_k3']:.3f}"
    )
    print(f"flags={out['flags']}")
    print(
        f"Rm_SSB={out['Rm_mean_SSB']}  NULL_SSB={out['NULL_mean_SSB']}  "
        f"|Rm-NULL|={out['separation_Rm_NULL']}"
    )
    out_path = (
        Path(__file__).resolve().parents[2]
        / "results"
        / "fase6_higgs_barrido_final"
        / "Higgs_TEST_REAL_v3_result.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
