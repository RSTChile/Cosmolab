#!/usr/bin/env python3
"""
CF4b_barrido_acoplamiento.py — CF-4b: "¿existe un regimen donde la
masa-ligadura domina sobre los constituyentes?" (barrido de gamma)

QUE ES ESTO (para retomar sin releer todo):
  CF-4 (Cosmogenesis-Web/codigo/CF4_ligadura/CF4_confinamiento.py) fallo
  (ratio_lig nunca cerca de 5.0) porque R0=2.0, U=0.5, D_PHI=0.05 estaban
  hardcodeados, heredados de v6, NUNCA barridos -- un sesgo estructural
  hacia el FAIL escondido en las unidades (violacion accidental de T1,
  ver INSTRUCCION_CF-4b_..._PARA_CC_y_Grok.md secc.2). CF-4b corrige eso:
  convierte la razon acoplamiento/potencial gamma = D_PHI/(R0*U) (con R0,
  U fijos = a CF-4) en el EJE del barrido, en varias decadas a ambos lados
  del valor de CF-4 (D_PHI=0.05), y verifica estabilidad numerica en cada
  punto (D_eff grande puede volver inestable la actualizacion EXPLICITA
  del campo -- eso se reporta si ocurre, no se oculta).

  El observable de masa (m1, m2_real, m2_null, ratio_lig, ratio_null) NO
  cambia respecto a CF-4. NO usa co_member_score/n_long_co_pairs/
  fusion_events/linaje (ver HALLAZGO_ABIERTO_etapa7_v6_masa_es_linaje_CS.md
  -- por que esas variables estan prohibidas en cualquier observable de
  este arco).

  Protocolo pre-registrado: PROTOCOLO_CF-4b_PREREGISTRO.md (leer primero,
  fija rango de gamma, H_TOPO elegidos y por que, semillas, umbrales
  heredados/congelados de CF-4 -- NO se tocan aqui).

REUTILIZACION (sin editar CF4_confinamiento.py):
  Se IMPORTA (no se retipea) de CF4_confinamiento.py: la dataclass P,
  medium_norm(), weighted_cut(), find_closures(), null_bind_energy(), y
  las constantes THRESH_BIG/THRESH_NULL (para garantizar que el criterio
  de PASS es EXACTAMENTE el mismo objeto, no una copia que pueda
  divergir por error de tipeo).
  El bucle simulate() se REIMPLEMENTA aqui (copiado, misma fisica letra
  por letra) SOLO para agregar instrumentacion de estabilidad numerica
  (deteccion de divergencia por overflow/NaN bajo np.errstate, y registro
  de max_abs_phi por corrida) -- ver protocolo secc.3. La fisica dentro
  del paso (evolucion de Phi, corte de enlaces, BFS de cierres, NULL de
  enlaces barajados) es identica a CF4_confinamiento.simulate().

CONVENCION DE ARCHIVOS: codigo nuevo aqui (CF4b_acoplamiento/), resultados
en Cosmogenesis-Web/results/CF4b_acoplamiento/. No edita CF4_confinamiento.py
ni ningun archivo v1-v6/motor_1a7. No toca topologia (CG001/ANIMA/VSTCosmo).
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CF4_DIR = HERE.parent  # .../codigo/CF4_ligadura
OUT = HERE.parents[2] / "results" / "CF4b_acoplamiento"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(CF4_DIR))
import CF4_confinamiento as cf4  # noqa: E402  (import tras sys.path.insert, deliberado)

# --- reutilizados por IMPORT directo de CF4_confinamiento (no retipeados) ---
P = cf4.P
weighted_cut = cf4.weighted_cut
find_closures = cf4.find_closures
null_bind_energy = cf4.null_bind_energy
THRESH_BIG = cf4.THRESH_BIG      # 5.0, heredado de CF-4, CONGELADO
THRESH_NULL = cf4.THRESH_NULL    # 1.25, heredado de CF-4, CONGELADO

PROTOCOL_ID = "CF4b_ACOPLAMIENTO_2026-07-24"

# --- parametros fijos, identicos a CF-4 (protocolo secc.0/1) ---
L_PROD = cf4.L_PROD           # 28
PASOS_PROD = cf4.PASOS_PROD   # 400
H_EXP = cf4.H_EXP
TC = cf4.TC
R0 = cf4.R0                   # 2.0 -- FIJO, no se barre en CF-4b
U = cf4.U                     # 0.5 -- FIJO, no se barre en CF-4b
DT_PHI = cf4.DT_PHI           # 0.08
SIGMA0 = cf4.SIGMA0
RHO0 = cf4.RHO0
FREEZE_TNORM = cf4.FREEZE_TNORM
RHO_FREEZE = cf4.RHO_FREEZE
ALPHA_CUT = cf4.ALPHA_CUT
MEASURE_STRIDE = cf4.MEASURE_STRIDE
NULL_REPEATS = cf4.NULL_REPEATS

D_PHI_CF4 = cf4.D_PHI  # 0.05 -- valor de referencia (el que fallo en CF-4)
GAMMA_CF4 = D_PHI_CF4 / (R0 * U)

# --- EJE NUEVO: barrido de D_PHI (=> gamma = D_PHI/(R0*U)), protocolo secc.1 ---
D_PHI_SWEEP = (
    0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
    0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0,
)

# --- H_TOPO elegidos con criterio de la tabla per_seed_H_table de CF-4,
#     protocolo secc.2: 0.04 (mayor n_joint_pop=5232, mean_k~2.85, escala
#     "pocos cuerpos") y 0.10 (mayor median_ratio_null=1.072 de CF-4, el
#     mas cercano al umbral 1.25) ---
H_TOPO_CHOSEN = (0.04, 0.10)

SEEDS_PROD = (7, 42, 99, 777)
SEEDS_SMOKE = (7, 42)
D_PHI_SMOKE = (0.005, 0.05, 5.0)

L_SMOKE = cf4.L_SMOKE         # 16
PASOS_SMOKE = cf4.PASOS_SMOKE  # 120


def simulate_gamma(p) -> tuple[list[dict], dict]:
    """
    Copia letra por letra de CF4_confinamiento.simulate(), con dos
    diferencias UNICAS (documentadas en el modulo docstring y en el
    protocolo secc.3):
      1) la aritmetica de la actualizacion de Phi corre bajo
         np.errstate(over="raise", invalid="raise") -- si diverge
         (overflow / NaN), se detecta EN EL PASO exacto, se detiene esa
         corrida ahi y se marca status="diverged" (no se sigue integrando
         sobre un campo roto, no se ocultan registros invalidos).
      2) se trackea max_abs_phi (maximo de |Phi| mientras siguio finito)
         como diagnostico cuantitativo de cercania al borde de
         estabilidad, incluso en corridas que NO divergieron.
    Todo lo demas (evolucion de Phi, corte de enlaces, BFS de cierres,
    m1/m2_real/m2_null) es identico a CF4_confinamiento.simulate().
    """
    rng = np.random.default_rng(p.seed)
    L = p.L
    Phi = 0.06 * rng.normal(size=(L, L))
    ar = np.ones((L, L), dtype=bool)
    ad = np.ones((L, L), dtype=bool)

    records: list[dict] = []
    diag = {
        "status": "ok",
        "diverged_step": None,
        "diverged_reason": None,
        "max_abs_phi": 0.0,
        "n_steps_completed": 0,
        "n_nonfinite_skipped": 0,
    }
    gamma = p.D_PHI / (p.R0 * p.U)

    for step in range(p.pasos):
        tg = step / max(p.pasos - 1, 1)
        a = float(np.exp(p.H_EXP * tg))
        Tnorm = float(np.exp(-p.H_EXP * tg))
        rho_c = p.RHO0 / (a**3)
        rho_hat_c = rho_c / p.RHO0
        frozen = (Tnorm < p.FREEZE_TNORM) or (rho_c < p.RHO_FREEZE)

        r_field = p.R0 * (Tnorm - p.TC)

        with np.errstate(over="raise", invalid="raise"):
            try:
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
                Phi_new = (
                    Phi
                    + p.DT_PHI * (-dV + D_eff * lap)
                    + sig * rng.normal(size=(L, L))
                )
            except FloatingPointError as e:
                diag["status"] = "diverged"
                diag["diverged_step"] = step
                diag["diverged_reason"] = f"FloatingPointError: {e}"
                break

        if not np.all(np.isfinite(Phi_new)):
            diag["status"] = "diverged"
            diag["diverged_step"] = step
            diag["diverged_reason"] = "Phi no finito (NaN/Inf) tras actualizacion (sin excepcion numpy)"
            break

        Phi = Phi_new
        cur_max = float(np.max(np.abs(Phi)))
        if cur_max > diag["max_abs_phi"]:
            diag["max_abs_phi"] = cur_max
        diag["n_steps_completed"] = step + 1

        # --- corte de enlaces = intensidad de confinamiento (H_TOPO), identico a CF-4 ---
        H_fis = p.H_TOPO * np.sqrt(Tnorm + 1e-12)
        tot = int(ar.sum() + ad.sum())
        nc = int(round(H_fis * tot))
        if nc > 0:
            weighted_cut(ar, ad, Phi, nc, rng, p.ALPHA_CUT)

        if (not frozen) and (step % p.MEASURE_STRIDE == 0):
            closures = find_closures(ar, ad, L)
            v_min = -(r_field**2) / (4.0 * p.U) if r_field < 0 else 0.0
            for c in closures:
                k = c["k"]
                nodes = c["nodes"]
                edges = c["edges"]
                m1 = float(
                    sum(
                        (r_field * Phi[y, x] ** 2 + p.U * Phi[y, x] ** 4) - v_min
                        for (y, x) in nodes
                    )
                )
                if k == 1:
                    m2_real = 0.0
                    m2_null = None
                else:
                    m2_real = float(
                        sum(
                            D_eff * (Phi[ny, nx] - Phi[my, mx]) ** 2
                            for edge in edges
                            for (ny, nx), (my, mx) in [tuple(edge)]
                        )
                    )
                    m2_null = null_bind_energy(
                        nodes, Phi, D_eff, len(edges), rng, p.NULL_REPEATS
                    )
                # Guardia adicional: Phi (el array) puede seguir siendo
                # finito por np.isfinite() mientras que Phi[y,x]**4 en un
                # solo nodo YA desbordo a inf/nan al calcular m1/m2 (esta
                # aritmetica corre fuera del bloque np.errstate(raise) de
                # arriba, que solo protege la actualizacion de Phi). Sin
                # este filtro, un registro con m1/m2 = inf/nan se cuela en
                # la curva silenciosamente (NaN literal en el JSON, medianas
                # contaminadas). Se descarta el registro (no se inventa un
                # valor) y se cuenta en n_nonfinite_skipped -- reportado en
                # stability_table, no oculto.
                if not (
                    np.isfinite(m1)
                    and np.isfinite(m2_real)
                    and (m2_null is None or np.isfinite(m2_null))
                ):
                    diag["n_nonfinite_skipped"] += 1
                    continue
                records.append(
                    {
                        "step": step,
                        "Tnorm": Tnorm,
                        "H_TOPO": p.H_TOPO,
                        "D_PHI": p.D_PHI,
                        "gamma": gamma,
                        "seed": p.seed,
                        "k": k,
                        "n_edges": len(edges),
                        "m1": m1,
                        "m2_real": m2_real,
                        "m2_null": m2_null,
                    }
                )
    return records, diag


def run_grid_gamma(dphi_vals, h_topo_vals, seeds, L, pasos, tag, verbose=True):
    t0 = time.time()
    all_records: list[dict] = []
    stability_table: list[dict] = []
    n_runs = 0
    n_diverged = 0
    for dphi in dphi_vals:
        for h in h_topo_vals:
            for s in seeds:
                p = P(L=L, pasos=pasos, H_TOPO=float(h), seed=int(s), D_PHI=float(dphi))
                recs, diag = simulate_gamma(p)
                all_records.extend(recs)
                n_runs += 1
                gamma = float(dphi) / (R0 * U)
                stability_table.append(
                    {
                        "D_PHI": float(dphi),
                        "gamma": gamma,
                        "H_TOPO": float(h),
                        "seed": int(s),
                        "status": diag["status"],
                        "diverged_step": diag["diverged_step"],
                        "diverged_reason": diag["diverged_reason"],
                        "max_abs_phi": diag["max_abs_phi"],
                        "n_steps_completed": diag["n_steps_completed"],
                        "n_nonfinite_skipped": diag["n_nonfinite_skipped"],
                        "n_records": len(recs),
                    }
                )
                if diag["status"] != "ok":
                    n_diverged += 1
                if verbose:
                    ks = [r["k"] for r in recs]
                    flag = "" if diag["status"] == "ok" else f"  *** {diag['status'].upper()} step={diag['diverged_step']} ***"
                    nf = f"  nonfinite_skipped={diag['n_nonfinite_skipped']}" if diag["n_nonfinite_skipped"] else ""
                    print(
                        f"  [{tag}] D_PHI={dphi:8.4f} gamma={gamma:8.4f} H_TOPO={h:.3f} seed={s:6d} "
                        f"n_meas={len(recs):5d} k=({min(ks) if ks else 0},{max(ks) if ks else 0}) "
                        f"max|Phi|={diag['max_abs_phi']:.3g} elapsed={time.time()-t0:.1f}s{flag}{nf}"
                    )
    if verbose:
        print(
            f"[{tag}] {n_runs} corridas ({n_diverged} divergieron), "
            f"{len(all_records)} instancias de cierre, {time.time()-t0:.1f}s total"
        )
    return all_records, stability_table


def _ratio_lig_list(records):
    return [r["m2_real"] / max(r["m1"], 1e-9) for r in records if r["k"] >= 2]


def _ratio_null_list(records):
    return [
        r["m2_real"] / max(r["m2_null"], 1e-9)
        for r in records
        if r["k"] >= 3 and r["m2_null"] is not None
    ]


def per_seed_gamma_table(records: list[dict]) -> list[dict]:
    """Filas crudas por (D_PHI, H_TOPO, seed): razones agregadas, para el reporte."""
    rows = []
    keys = sorted(set((r["D_PHI"], r["H_TOPO"], r["seed"]) for r in records))
    for dphi, h, s in keys:
        sub = [r for r in records if r["D_PHI"] == dphi and r["H_TOPO"] == h and r["seed"] == s]
        lig = _ratio_lig_list(sub)
        nul = _ratio_null_list(sub)
        rows.append(
            {
                "D_PHI": dphi,
                "gamma": dphi / (R0 * U),
                "H_TOPO": h,
                "seed": s,
                "n_closures_k_ge2": len(lig),
                "n_closures_k_ge3": len(nul),
                "ratio_lig_mean": float(np.mean(lig)) if lig else None,
                "ratio_null_mean": float(np.mean(nul)) if nul else None,
                "k_max": max((r["k"] for r in sub), default=0),
            }
        )
    return rows


def curve_by_gamma(records: list[dict], dphi_vals, h_topo_vals, stability_table=None) -> dict:
    """
    Curva ratio_lig(gamma) y ratio_null(gamma) por H_TOPO, con dispersion
    REAL entre semillas (no solo el agregado sobre todos los registros).

    Incluye por punto n_seeds_diverged/n_seeds_total (de stability_table) y
    un flag point_pass_both_median_RELIABLE: el criterio pre-registrado
    (point_pass_both_median) se calcula IGUAL, sin tocarlo -- pero si TODAS
    las semillas de ese punto divergieron numericamente, el "pass" mecanico
    esta contaminado por datos de pre-colapso (pocos cierres sobrevivientes,
    Phi ya casi roto) y NO cuenta como candidato de banda estable. Esto no
    cambia el umbral ni el criterio -- solo evita que un artefacto numerico
    se lea como señal fisica sin que quede dicho explicitamente.
    """
    stability_table = stability_table or []
    out = {}
    for h in h_topo_vals:
        h = float(h)
        points = []
        for dphi in dphi_vals:
            dphi = float(dphi)
            sub_all = [r for r in records if r["D_PHI"] == dphi and r["H_TOPO"] == h]
            lig_all = _ratio_lig_list(sub_all)
            nul_all = _ratio_null_list(sub_all)

            # dispersion entre semillas: un numero (media de ratio_lig/ratio_null) POR semilla
            seeds_here = sorted(set(r["seed"] for r in sub_all))
            per_seed_lig = []
            per_seed_null = []
            for s in seeds_here:
                sub_s = [r for r in sub_all if r["seed"] == s]
                l = _ratio_lig_list(sub_s)
                n = _ratio_null_list(sub_s)
                if l:
                    per_seed_lig.append(float(np.mean(l)))
                if n:
                    per_seed_null.append(float(np.mean(n)))

            joint_pop = [r for r in sub_all if r["k"] >= 3 and r["m2_null"] is not None]
            joint_pass = [
                (r["m2_real"] / max(r["m1"], 1e-9) >= THRESH_BIG)
                and (r["m2_real"] / max(r["m2_null"], 1e-9) >= THRESH_NULL)
                for r in joint_pop
            ]

            stab_here = [
                st for st in stability_table
                if st["D_PHI"] == dphi and st["H_TOPO"] == h
            ]
            n_seeds_total_stab = len(stab_here)
            n_seeds_diverged = sum(1 for st in stab_here if st["status"] != "ok")
            all_seeds_diverged = bool(stab_here) and n_seeds_diverged == n_seeds_total_stab

            pass_both_median = bool(
                (lig_all and np.median(lig_all) >= THRESH_BIG)
                and (nul_all and np.median(nul_all) >= THRESH_NULL)
            )

            points.append(
                {
                    "D_PHI": dphi,
                    "gamma": dphi / (R0 * U),
                    "n_records": len(sub_all),
                    "n_k_ge2": len(lig_all),
                    "n_k_ge3_with_null": len(nul_all),
                    "ratio_lig_median": float(np.median(lig_all)) if lig_all else None,
                    "ratio_lig_mean": float(np.mean(lig_all)) if lig_all else None,
                    "ratio_null_median": float(np.median(nul_all)) if nul_all else None,
                    "ratio_null_mean": float(np.mean(nul_all)) if nul_all else None,
                    "ratio_lig_per_seed_mean": per_seed_lig,
                    "ratio_null_per_seed_mean": per_seed_null,
                    "ratio_lig_seed_std": float(np.std(per_seed_lig)) if len(per_seed_lig) > 1 else 0.0,
                    "ratio_null_seed_std": float(np.std(per_seed_null)) if len(per_seed_null) > 1 else 0.0,
                    "rate_pass_joint": float(np.mean(joint_pass)) if joint_pass else 0.0,
                    "n_seeds_with_data": len(seeds_here),
                    "point_pass_both_median": pass_both_median,
                    "n_seeds_diverged": n_seeds_diverged,
                    "n_seeds_total_stability": n_seeds_total_stab,
                    "all_seeds_diverged": all_seeds_diverged,
                    "point_pass_both_median_RELIABLE": pass_both_median and not all_seeds_diverged,
                }
            )
        out[f"{h:.3f}"] = points
    return out


def find_stable_bands(curve_points: list[dict], min_contig=3, flag_key="point_pass_both_median") -> list[dict]:
    """
    Banda ESTABLE (protocolo secc.0 / instruccion secc.4): >=min_contig
    puntos de gamma CONTIGUOS (en el orden del barrido, ordenado por gamma)
    donde flag_key es True. Aplica el criterio congelado mecanicamente --
    no decide si "hay mecanismo", solo reporta que puntos cumplen la
    formula pre-registrada. Se llama dos veces (ver analyze_gamma): una con
    el flag crudo (point_pass_both_median) y otra con el flag RELIABLE
    (excluye puntos donde el 100% de las semillas diverigo numericamente).
    """
    pts = sorted(curve_points, key=lambda r: r["gamma"])
    bands = []
    run = []
    for pt in pts:
        if pt[flag_key]:
            run.append(pt)
        else:
            if len(run) >= min_contig:
                bands.append(run)
            run = []
    if len(run) >= min_contig:
        bands.append(run)
    return [
        {
            "gamma_min": b[0]["gamma"],
            "gamma_max": b[-1]["gamma"],
            "n_points": len(b),
            "gammas": [p["gamma"] for p in b],
        }
        for b in bands
    ]


def analyze_gamma(records: list[dict], dphi_vals, h_topo_vals, stability_table=None) -> dict:
    stability_table = stability_table or []
    k_hist: dict[int, int] = {}
    for r in records:
        k_hist[r["k"]] = k_hist.get(r["k"], 0) + 1

    curves = curve_by_gamma(records, dphi_vals, h_topo_vals, stability_table)
    # bandas con el criterio congelado tal cual (crudo, para auditoria total)
    bands_raw = {h: find_stable_bands(pts, flag_key="point_pass_both_median") for h, pts in curves.items()}
    # bandas EXCLUYENDO puntos donde el 100% de las semillas diverigo
    # numericamente (ver docstring de curve_by_gamma) -- esta es la que
    # importa para leer si "existe regimen", la otra es solo auditoria
    bands_reliable = {h: find_stable_bands(pts, flag_key="point_pass_both_median_RELIABLE") for h, pts in curves.items()}

    def _max_point(key, reliable_only):
        pts_flat = [
            (pt["gamma"], h, pt[key])
            for h, pts in curves.items()
            for pt in pts
            if pt[key] is not None and (not reliable_only or not pt["all_seeds_diverged"])
        ]
        return max(pts_flat, key=lambda t: t[2]) if pts_flat else None

    max_lig_point = _max_point("ratio_lig_median", reliable_only=False)
    max_null_point = _max_point("ratio_null_median", reliable_only=False)
    max_lig_point_reliable = _max_point("ratio_lig_median", reliable_only=True)
    max_null_point_reliable = _max_point("ratio_null_median", reliable_only=True)

    def _fmt(pt, key):
        return {"gamma": pt[0], "H_TOPO": pt[1], key: pt[2]} if pt else None

    return {
        "n_records_total": len(records),
        "k_histogram": {str(k): v for k, v in sorted(k_hist.items())},
        "curves_by_H_TOPO": curves,
        "stable_bands_by_H_TOPO_RAW_MECHANICAL": bands_raw,
        "stable_bands_by_H_TOPO_RELIABLE": bands_reliable,
        "max_ratio_lig_median_point": _fmt(max_lig_point, "ratio_lig_median"),
        "max_ratio_null_median_point": _fmt(max_null_point, "ratio_null_median"),
        "max_ratio_lig_median_point_RELIABLE_noDiverged": _fmt(max_lig_point_reliable, "ratio_lig_median"),
        "max_ratio_null_median_point_RELIABLE_noDiverged": _fmt(max_null_point_reliable, "ratio_null_median"),
        "THRESH_BIG": THRESH_BIG,
        "THRESH_NULL": THRESH_NULL,
        "GAMMA_CF4_reference": GAMMA_CF4,
    }


def peak_rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    # macOS: ru_maxrss en bytes; Linux: en KB
    val = ru.ru_maxrss
    if sys.platform == "darwin":
        return val / (1024 * 1024)
    return val / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="corre config smoke (mecanica+estabilidad), no produccion")
    args = ap.parse_args()

    t_wall0 = time.time()
    print(f"=== CF-4b barrido de acoplamiento gamma ({PROTOCOL_ID}) ===")
    print(f"gamma = D_PHI/(R0*U), R0={R0} U={U} fijos (identicos a CF-4). gamma_CF4={GAMMA_CF4}")

    if args.smoke:
        print("--- SMOKE (validacion de mecanica + deteccion de divergencia, no decide PASS) ---")
        records, stability = run_grid_gamma(
            D_PHI_SMOKE, H_TOPO_CHOSEN, SEEDS_SMOKE, L_SMOKE, PASOS_SMOKE, tag="smoke"
        )
        summary = analyze_gamma(records, D_PHI_SMOKE, H_TOPO_CHOSEN, stability)
        out = {
            "protocol_id": PROTOCOL_ID,
            "mode": "smoke",
            "params": {
                "L": L_SMOKE,
                "pasos": PASOS_SMOKE,
                "D_PHI_sweep": list(D_PHI_SMOKE),
                "H_TOPO_chosen": list(H_TOPO_CHOSEN),
                "seeds": list(SEEDS_SMOKE),
                "R0": R0,
                "U": U,
            },
            "summary": summary,
            "per_seed_gamma_H_table": per_seed_gamma_table(records),
            "stability_table": stability,
            "wall_time_sec": time.time() - t_wall0,
            "peak_rss_mb": peak_rss_mb(),
        }
        path = OUT / "CF4b_smoke_result.json"
        path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nJSON -> {path}")
        n_diverged = sum(1 for s in stability if s["status"] != "ok")
        print(f"divergencias en smoke: {n_diverged}/{len(stability)}")
        print(f"wall_time_sec={out['wall_time_sec']:.2f} peak_rss_mb={out['peak_rss_mb']:.1f}")
        return

    print("--- PRODUCCION ---")
    records, stability = run_grid_gamma(
        D_PHI_SWEEP, H_TOPO_CHOSEN, SEEDS_PROD, L_PROD, PASOS_PROD, tag="prod"
    )
    summary = analyze_gamma(records, D_PHI_SWEEP, H_TOPO_CHOSEN, stability)
    seed_table = per_seed_gamma_table(records)

    wall_time = time.time() - t_wall0
    rss = peak_rss_mb()

    out = {
        "protocol_id": PROTOCOL_ID,
        "mode": "production",
        "params": {
            "L": L_PROD,
            "pasos": PASOS_PROD,
            "D_PHI_sweep": list(D_PHI_SWEEP),
            "gamma_sweep": [d / (R0 * U) for d in D_PHI_SWEEP],
            "H_TOPO_chosen": list(H_TOPO_CHOSEN),
            "seeds": list(SEEDS_PROD),
            "R0_fixed": R0,
            "U_fixed": U,
            "MEASURE_STRIDE": MEASURE_STRIDE,
            "NULL_REPEATS": NULL_REPEATS,
            "THRESH_BIG": THRESH_BIG,
            "THRESH_NULL": THRESH_NULL,
            "GAMMA_CF4_reference": GAMMA_CF4,
        },
        "summary": summary,
        "per_seed_gamma_H_table": seed_table,
        "stability_table": stability,
        "wall_time_sec": wall_time,
        "peak_rss_mb": rss,
    }
    path = OUT / "CF4b_produccion_result.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nJSON -> {path}")

    # k histogram con m2 asociado (mismo estilo que CF-4, pedido para CF-6)
    k_m2: dict[int, list[float]] = {}
    for r in records:
        k_m2.setdefault(r["k"], []).append(r["m2_real"])
    k_m2_summary = {
        str(k): {
            "n": len(v),
            "mean_m2_real": float(np.mean(v)),
            "median_m2_real": float(np.median(v)),
        }
        for k, v in sorted(k_m2.items())
    }
    (OUT / "CF4b_histograma_k_m2.json").write_text(
        json.dumps(k_m2_summary, indent=2), encoding="utf-8"
    )
    print(f"JSON -> {OUT / 'CF4b_histograma_k_m2.json'}")

    n_diverged = sum(1 for s in stability if s["status"] != "ok")
    md = [
        "# CF-4b -- barrido de acoplamiento gamma (resultado crudo, sin adjudicar)\n\n",
        f"Protocolo: `{PROTOCOL_ID}`\n\n",
        f"- rango gamma barrido: [{min(d/(R0*U) for d in D_PHI_SWEEP):.4g}, "
        f"{max(d/(R0*U) for d in D_PHI_SWEEP):.4g}] "
        f"(referencia CF-4: gamma={summary['GAMMA_CF4_reference']:.4g})\n",
        f"- H_TOPO elegidos: {list(H_TOPO_CHOSEN)} (ver protocolo secc.2)\n",
        f"- semillas: {list(SEEDS_PROD)}\n",
        f"- n instancias de cierre medidas: **{summary['n_records_total']}**\n",
        f"- corridas totales: **{len(stability)}**, divergieron: **{n_diverged}**\n",
        f"- max ratio_lig_median (CRUDO, incluye puntos 100% divergidos): "
        f"**{summary['max_ratio_lig_median_point']}**\n",
        f"- max ratio_lig_median (RELIABLE, excluye puntos donde TODAS las semillas "
        f"divergieron numericamente): **{summary['max_ratio_lig_median_point_RELIABLE_noDiverged']}**\n",
        f"- max ratio_null_median (CRUDO): **{summary['max_ratio_null_median_point']}**\n",
        f"- max ratio_null_median (RELIABLE): "
        f"**{summary['max_ratio_null_median_point_RELIABLE_noDiverged']}**\n",
        f"- bandas estables, criterio congelado CRUDO (>=3 puntos contiguos, ambos umbrales, "
        f"incluye puntos divergidos): **{summary['stable_bands_by_H_TOPO_RAW_MECHANICAL']}**\n",
        f"- bandas estables RELIABLE (excluye puntos donde el 100% de semillas divergio): "
        f"**{summary['stable_bands_by_H_TOPO_RELIABLE']}**\n",
        f"- umbrales heredados (congelados): ratio_lig>={THRESH_BIG}, ratio_null>={THRESH_NULL}\n",
        "\nVer JSON completo para la curva punto a punto (incluye n_seeds_diverged y "
        "all_seeds_diverged por punto) y la tabla de estabilidad completa. "
        "No se adjudica aqui si existe o no el regimen -- lo decide CS con la curva a la vista.\n",
    ]
    (OUT / "CF4b_RESUMEN.md").write_text("".join(md), encoding="utf-8")
    print(f"MD  -> {OUT / 'CF4b_RESUMEN.md'}")

    print(f"\nwall_time_sec={wall_time:.2f} peak_rss_mb={rss:.1f}")
    print(f"corridas={len(stability)} divergieron={n_diverged}")
    print("\n=== stable_bands_by_H_TOPO_RELIABLE ===")
    print(json.dumps(summary["stable_bands_by_H_TOPO_RELIABLE"], indent=2))
    print("\n=== stable_bands_by_H_TOPO_RAW_MECHANICAL ===")
    print(json.dumps(summary["stable_bands_by_H_TOPO_RAW_MECHANICAL"], indent=2))


if __name__ == "__main__":
    main()
