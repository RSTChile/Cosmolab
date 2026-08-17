#!/usr/bin/env python3
"""
CF4_confinamiento.py — CF-4: "¿el 99% de la masa es energia de ligadura?"

QUE ES ESTO (para retomar sin releer todo):
  Mide la masa en la epoca FISICA correcta: durante el CONFINAMIENTO
  (analogo a la transicion QCD, quarks atrapados en un cierre), en la
  fase CALIENTE, ANTES de que existan atomos. NO reutiliza ni
  components_strict() ni los criterios K_MIN/K_MAX/F_CORE/COHESION de
  v1-v6 (esos son para la etapa POSTERIOR, de atomo). NO usa
  co_member_score/n_long_co_pairs/fusion_events/linaje de v6 (esa fue
  la circularidad documentada en HALLAZGO_ABIERTO_etapa7_v6_masa_es_linaje_CS.md).

  Reutiliza SOLO piezas fisicas de suite_epocas_masa_v6_mass_linaje.py
  (sin editar ese archivo, sin importarlo — reimplementado aqui):
    - la evolucion del campo Phi (potencial Phi^4 + difusion + ruido)
    - el corte de enlaces weighted_cut() (H_TOPO/ALPHA_CUT = intensidad
      de confinamiento: cuan fuerte resiste el enlace a cortarse)
    - el esqueleto algoritmico de components_strict() (union de nodos
      vecinos via arrays de enlace ar/ad) pero SIN el filtro de atomo.

  Observable de masa (nucleo del experimento):
    m1(cierre) = suma de V(Phi_i) de los nodos COMO SI estuvieran libres
                 (sin termino de enlace) = "masa de constituyentes libres"
    m2(cierre) = suma de D_eff*(Phi_i-Phi_j)^2 sobre los enlaces INTERNOS
                 del cierre = "energia de ligadura" = "trabajo para separarlo"
                 (exactamente el termino que D_eff*lap ya representa en
                 la dinamica -- energia real de la fisica, no inventada)
    NULL = mismo cierre, mismos nodos/tamano/grado, enlaces internos
           RE-CONECTADOS al azar entre los mismos nodos (no la topologia
           real de confinamiento) -> m2_NULL

  Protocolo pre-registrado: PROTOCOLO_CF-4_PREREGISTRO.md (leer primero,
  fija el criterio de PASS ANTES de correr, no se toca despues).

  Convencion de carpeta: CF4_* en codigo/CF4_ligadura/, resultados en
  results/CF4_ligadura/. No edita ningun archivo v1-v6 ni motor_1a7.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import asdict, dataclass
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / "CF4_ligadura"
OUT.mkdir(parents=True, exist_ok=True)

PROTOCOL_ID = "CF4_LIGADURA_2026-07-23"

# --- parametros fijos, identicos a defaults de v6 (sec. 5 del protocolo) ---
L_PROD = 28
PASOS_PROD = 400
H_EXP = 6.0
TC = 0.55
R0 = 2.0
U = 0.5
D_PHI = 0.05
DT_PHI = 0.08
SIGMA0 = 0.10
RHO0 = 1.0
FREEZE_TNORM = 0.40
RHO_FREEZE = 0.05
ALPHA_CUT = 2.5
MEASURE_STRIDE = 5
NULL_REPEATS = 5

# --- barrido de intensidad de confinamiento (sec. 2.2 del protocolo) ---
H_TOPO_SWEEP = (0.002, 0.004, 0.007, 0.01, 0.02, 0.04, 0.07, 0.10)
SEEDS = (7, 42, 99, 777, 2025, 3141, 8191, 99991, 12345, 54321)

# --- smoke (sec. 7.1 del protocolo) ---
L_SMOKE = 16
PASOS_SMOKE = 120
H_TOPO_SMOKE = (0.005, 0.02, 0.08)
SEEDS_SMOKE = (7, 42)

# --- criterio de PASS (sec. 4, congelado antes de correr) ---
THRESH_BIG = 5.0
THRESH_NULL = 1.25
RATE_PASS_REF = 0.55


@dataclass
class P:
    L: int
    pasos: int
    H_TOPO: float
    seed: int
    H_EXP: float = H_EXP
    TC: float = TC
    R0: float = R0
    U: float = U
    D_PHI: float = D_PHI
    DT_PHI: float = DT_PHI
    SIGMA0: float = SIGMA0
    RHO0: float = RHO0
    FREEZE_TNORM: float = FREEZE_TNORM
    RHO_FREEZE: float = RHO_FREEZE
    ALPHA_CUT: float = ALPHA_CUT
    MEASURE_STRIDE: int = MEASURE_STRIDE
    NULL_REPEATS: int = NULL_REPEATS


def medium_norm(Phi):
    a = np.abs(Phi)
    p95 = float(np.percentile(a, 95)) + 1e-12
    return np.clip(a / p95, 0.0, 1.0)


def weighted_cut(ar, ad, Phi, nc, rng, alpha):
    """Reimplementacion de v6 weighted_cut (sin modo 'blind', no usado en CF-4)."""
    if nc <= 0:
        return
    ph = medium_norm(Phi)
    n_r = 0
    if ar.any():
        idx = np.argwhere(ar)
        tot = int(ar.sum() + ad.sum())
        n_r = min(int(round(nc * float(ar.sum()) / max(tot, 1))), len(idx))
        if n_r > 0:
            edge = 0.5 * (ph + np.roll(ph, -1, 1))
            w = np.where(ar, (1.0 - edge + 1e-3) ** alpha, 0.0)
            flat = w[tuple(idx.T)]
            pr = None if flat.sum() <= 0 else flat / flat.sum()
            sel = rng.choice(len(idx), size=n_r, replace=False, p=pr)
            for i in sel:
                ar[tuple(idx[i])] = False
        rem = nc - n_r
    else:
        rem = nc
    if rem > 0 and ad.any():
        idx = np.argwhere(ad)
        n_d = min(rem, len(idx))
        if n_d > 0:
            edge = 0.5 * (ph + np.roll(ph, -1, 0))
            w = np.where(ad, (1.0 - edge + 1e-3) ** alpha, 0.0)
            flat = w[tuple(idx.T)]
            pr = None if flat.sum() <= 0 else flat / flat.sum()
            sel = rng.choice(len(idx), size=n_d, replace=False, p=pr)
            for i in sel:
                ad[tuple(idx[i])] = False


def find_closures(ar, ad, L):
    """
    Componentes conexos del grafo de enlaces ar/ad vivos.
    Reutiliza el ESQUELETO algoritmico de components_strict() de v6
    (union de nodos vecinos via ar/ad, mismo patron de vecinos
    left/right/up/down), pero SIN el filtro "mismo lado" (phi>=media)
    ni ningun criterio de atomo. k emerge, no se impone (T0).
    """
    visited = np.zeros((L, L), dtype=bool)
    closures = []
    for y0 in range(L):
        for x0 in range(L):
            if visited[y0, x0]:
                continue
            q = deque([(y0, x0)])
            visited[y0, x0] = True
            nodes = [(y0, x0)]
            edge_set = set()
            while q:
                cy, cx = q.popleft()
                neigh = []
                if ar[cy, cx]:
                    neigh.append((cy, (cx + 1) % L))
                if ar[cy, (cx - 1) % L]:
                    neigh.append((cy, (cx - 1) % L))
                if ad[cy, cx]:
                    neigh.append(((cy + 1) % L, cx))
                if ad[(cy - 1) % L, cx]:
                    neigh.append(((cy - 1) % L, cx))
                for ny, nx in neigh:
                    edge_set.add(frozenset(((cy, cx), (ny, nx))))
                    if not visited[ny, nx]:
                        visited[ny, nx] = True
                        nodes.append((ny, nx))
                        q.append((ny, nx))
            closures.append({"nodes": nodes, "edges": list(edge_set), "k": len(nodes)})
    return closures


def null_bind_energy(nodes, Phi, D_eff, m_edges, rng, repeats):
    """
    NULL: mismos nodos del cierre, m_edges enlaces internos RE-CONECTADOS
    al azar entre esos mismos nodos (preserva tamano k y numero de
    enlaces, no la topologia real de confinamiento). k=2 excluido
    (caso degenerado: unico par posible, ver protocolo sec.3).
    """
    k = len(nodes)
    if k < 3 or m_edges <= 0:
        return None
    vals = np.array([Phi[y, x] for (y, x) in nodes])
    all_pairs = list(itertools.combinations(range(k), 2))
    n_pairs = len(all_pairs)
    m_eff = min(m_edges, n_pairs)
    energies = []
    for _ in range(repeats):
        sel = rng.choice(n_pairs, size=m_eff, replace=False)
        e = 0.0
        for pi in sel:
            i, j = all_pairs[pi]
            e += D_eff * (vals[i] - vals[j]) ** 2
        energies.append(e)
    return float(np.mean(energies))


def simulate(p: P) -> list[dict]:
    """
    Corre la dinamica de Phi + corte de enlaces (confinamiento) y mide
    m1/m2/m2_NULL por cierre en cada paso de la fase caliente (frozen=False),
    cada MEASURE_STRIDE pasos. No hay tracking de identidad entre pasos:
    cada medicion es una instantanea independiente (evita cualquier nocion
    de linaje/persistencia, T2).
    """
    rng = np.random.default_rng(p.seed)
    L = p.L
    Phi = 0.06 * rng.normal(size=(L, L))
    ar = np.ones((L, L), dtype=bool)
    ad = np.ones((L, L), dtype=bool)

    records = []
    for step in range(p.pasos):
        tg = step / max(p.pasos - 1, 1)
        a = float(np.exp(p.H_EXP * tg))
        Tnorm = float(np.exp(-p.H_EXP * tg))
        rho_c = p.RHO0 / (a**3)
        rho_hat_c = rho_c / p.RHO0
        frozen = (Tnorm < p.FREEZE_TNORM) or (rho_c < p.RHO_FREEZE)

        # --- evolucion de Phi (identica a v6 lineas ~430-443, con
        #     simplificacion documentada: r_field solo depende de Tnorm,
        #     sin el termino G_RHO*(rho_hat-1) porque CF-4 no modela el
        #     campo phi de v6) ---
        r_field = p.R0 * (Tnorm - p.TC)
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

        # --- corte de enlaces = intensidad de confinamiento (H_TOPO) ---
        H_fis = p.H_TOPO * np.sqrt(Tnorm + 1e-12)
        tot = int(ar.sum() + ad.sum())
        nc = int(round(H_fis * tot))
        if nc > 0:
            weighted_cut(ar, ad, Phi, nc, rng, p.ALPHA_CUT)

        if (not frozen) and (step % p.MEASURE_STRIDE == 0):
            closures = find_closures(ar, ad, L)
            # m1 = V(Phi) MEDIDO RESPECTO AL VACIO (V - V_min), no V absoluto.
            # Motivo (encontrado en smoke, corregido ANTES de produccion):
            # V(Phi)=r*Phi^2+U*Phi^4 tiene un aditivo libre (invariancia de
            # norma de la fisica ante constantes en el potencial); en fase
            # rota (r_field<0) V_min=-r_field^2/(4U) es muy negativo, y sumar
            # V absoluto vuelve "m1" negativo/casi-cero sin significado fisico
            # (una masa "libre" negativa no es una cantidad honesta). La
            # correccion estandar en teoria de campos: la masa es la energia
            # de excitacion SOBRE el vacio, siempre >=0 por construccion del
            # minimo. m2 (energia de acople, depende de diferencias Phi_i-Phi_j)
            # no tiene esta ambiguedad y no se toca.
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
                records.append(
                    {
                        "step": step,
                        "Tnorm": Tnorm,
                        "H_TOPO": p.H_TOPO,
                        "seed": p.seed,
                        "k": k,
                        "n_edges": len(edges),
                        "m1": m1,
                        "m2_real": m2_real,
                        "m2_null": m2_null,
                    }
                )
    return records


def run_grid(h_topo_vals, seeds, L, pasos, tag, verbose=True):
    t0 = time.time()
    all_records = []
    n_runs = 0
    for h in h_topo_vals:
        for s in seeds:
            p = P(L=L, pasos=pasos, H_TOPO=float(h), seed=int(s))
            recs = simulate(p)
            all_records.extend(recs)
            n_runs += 1
            if verbose:
                ks = [r["k"] for r in recs]
                print(
                    f"  [{tag}] H_TOPO={h:.4f} seed={s:6d} "
                    f"n_meas_closures={len(recs)} k_range=({min(ks) if ks else 0},{max(ks) if ks else 0}) "
                    f"elapsed={time.time()-t0:.1f}s"
                )
    if verbose:
        print(f"[{tag}] {n_runs} corridas, {len(all_records)} instancias de cierre, {time.time()-t0:.1f}s total")
    return all_records


def analyze(records: list[dict]) -> dict:
    """
    Agrega ratio_lig = m2_real/m1 (k>=2) y ratio_null = m2_real/m2_null (k>=3),
    tasa de PASS por el criterio pre-registrado (sec.4), histograma de k,
    y verificacion T4 (el NULL debe caer).
    """
    k_hist: dict[int, int] = {}
    for r in records:
        k_hist[r["k"]] = k_hist.get(r["k"], 0) + 1

    # k>=2: tiene al menos un enlace interno -> m2_real definido
    big_pop = [r for r in records if r["k"] >= 2]
    ratio_lig = [r["m2_real"] / max(r["m1"], 1e-9) for r in big_pop]

    # k>=3: comparacion NULL valida (k=2 degenerado, ver protocolo)
    null_pop = [r for r in records if r["k"] >= 3 and r["m2_null"] is not None]
    ratio_null = [
        r["m2_real"] / max(r["m2_null"], 1e-9) for r in null_pop
    ]

    joint_pop = [
        r
        for r in records
        if r["k"] >= 3 and r["m2_null"] is not None
    ]
    joint_pass = [
        (r["m2_real"] / max(r["m1"], 1e-9) >= THRESH_BIG)
        and (r["m2_real"] / max(r["m2_null"], 1e-9) >= THRESH_NULL)
        for r in joint_pop
    ]
    rate_pass = float(np.mean(joint_pass)) if joint_pass else 0.0

    # por H_TOPO
    by_h = {}
    h_vals = sorted(set(r["H_TOPO"] for r in records))
    for h in h_vals:
        sub_joint = [r for r in joint_pop if abs(r["H_TOPO"] - h) < 1e-12]
        sub_pass = [
            (r["m2_real"] / max(r["m1"], 1e-9) >= THRESH_BIG)
            and (r["m2_real"] / max(r["m2_null"], 1e-9) >= THRESH_NULL)
            for r in sub_joint
        ]
        sub_lig = [
            r["m2_real"] / max(r["m1"], 1e-9)
            for r in records
            if r["k"] >= 2 and abs(r["H_TOPO"] - h) < 1e-12
        ]
        sub_null_ratio = [
            r["m2_real"] / max(r["m2_null"], 1e-9)
            for r in records
            if r["k"] >= 3 and r["m2_null"] is not None and abs(r["H_TOPO"] - h) < 1e-12
        ]
        by_h[h] = {
            "n_joint_pop": len(sub_joint),
            "rate_pass": float(np.mean(sub_pass)) if sub_pass else 0.0,
            "median_ratio_lig": float(np.median(sub_lig)) if sub_lig else None,
            "median_ratio_null": float(np.median(sub_null_ratio)) if sub_null_ratio else None,
            "mean_k": float(np.mean([r["k"] for r in records if abs(r["H_TOPO"] - h) < 1e-12])),
        }

    mean_m2_real = float(np.mean([r["m2_real"] for r in null_pop])) if null_pop else 0.0
    mean_m2_null = float(np.mean([r["m2_null"] for r in null_pop])) if null_pop else 0.0
    t4_null_falls = mean_m2_null < mean_m2_real

    return {
        "n_records_total": len(records),
        "n_k_ge2": len(big_pop),
        "n_k_ge3_with_null": len(null_pop),
        "k_histogram": {str(k): v for k, v in sorted(k_hist.items())},
        "ratio_lig_median": float(np.median(ratio_lig)) if ratio_lig else None,
        "ratio_lig_mean": float(np.mean(ratio_lig)) if ratio_lig else None,
        "ratio_lig_p10_p90": [float(np.percentile(ratio_lig, 10)), float(np.percentile(ratio_lig, 90))] if ratio_lig else None,
        "ratio_null_median": float(np.median(ratio_null)) if ratio_null else None,
        "ratio_null_mean": float(np.mean(ratio_null)) if ratio_null else None,
        "rate_pass_joint": rate_pass,
        "rate_pass_ref_055": rate_pass >= RATE_PASS_REF,
        "mean_m2_real_k3plus": mean_m2_real,
        "mean_m2_null_k3plus": mean_m2_null,
        "T4_null_falls": t4_null_falls,
        "by_H_TOPO": {f"{h:.4f}": v for h, v in by_h.items()},
        "THRESH_BIG": THRESH_BIG,
        "THRESH_NULL": THRESH_NULL,
    }


def per_seed_table(records: list[dict]) -> list[dict]:
    """Filas crudas por semilla x H_TOPO: razones agregadas (para el reporte)."""
    rows = []
    keys = sorted(set((r["H_TOPO"], r["seed"]) for r in records))
    for h, s in keys:
        sub = [r for r in records if r["H_TOPO"] == h and r["seed"] == s]
        sub2 = [r for r in sub if r["k"] >= 2]
        sub3 = [r for r in sub if r["k"] >= 3 and r["m2_null"] is not None]
        lig = [r["m2_real"] / max(r["m1"], 1e-9) for r in sub2]
        nul = [r["m2_real"] / max(r["m2_null"], 1e-9) for r in sub3]
        rows.append(
            {
                "H_TOPO": h,
                "seed": s,
                "n_closures_k_ge2": len(sub2),
                "n_closures_k_ge3": len(sub3),
                "ratio_lig_mean": float(np.mean(lig)) if lig else None,
                "ratio_null_mean": float(np.mean(nul)) if nul else None,
                "k_max": max((r["k"] for r in sub), default=0),
            }
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="corre config smoke (mecanica), no produccion")
    args = ap.parse_args()

    print(f"=== CF-4 confinamiento — energia de ligadura ({PROTOCOL_ID}) ===")
    if args.smoke:
        print("--- SMOKE (validacion de mecanica, no decide PASS) ---")
        records = run_grid(H_TOPO_SMOKE, SEEDS_SMOKE, L_SMOKE, PASOS_SMOKE, tag="smoke")
        summary = analyze(records)
        out = {
            "protocol_id": PROTOCOL_ID,
            "mode": "smoke",
            "params": {
                "L": L_SMOKE,
                "pasos": PASOS_SMOKE,
                "H_TOPO_sweep": list(H_TOPO_SMOKE),
                "seeds": list(SEEDS_SMOKE),
            },
            "summary": summary,
            "per_seed_H": per_seed_table(records),
        }
        path = OUT / "CF4_smoke_result.json"
        path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nJSON -> {path}")
        print(json.dumps(summary, indent=2)[:4000])
        return

    print("--- PRODUCCION ---")
    records = run_grid(H_TOPO_SWEEP, SEEDS, L_PROD, PASOS_PROD, tag="prod")
    summary = analyze(records)
    seed_table = per_seed_table(records)

    out = {
        "protocol_id": PROTOCOL_ID,
        "mode": "production",
        "params": {
            "L": L_PROD,
            "pasos": PASOS_PROD,
            "H_TOPO_sweep": list(H_TOPO_SWEEP),
            "seeds": list(SEEDS),
            "MEASURE_STRIDE": MEASURE_STRIDE,
            "NULL_REPEATS": NULL_REPEATS,
            "THRESH_BIG": THRESH_BIG,
            "THRESH_NULL": THRESH_NULL,
        },
        "summary": summary,
        "per_seed_H_table": seed_table,
    }
    path = OUT / "CF4_produccion_result.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nJSON -> {path}")

    # k histogram con m2 asociado (para CF-6, pedido explicito)
    k_m2 = {}
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
    (OUT / "CF4_histograma_k_m2.json").write_text(
        json.dumps(k_m2_summary, indent=2), encoding="utf-8"
    )
    print(f"JSON -> {OUT / 'CF4_histograma_k_m2.json'}")

    md = [
        "# CF-4 — energia de ligadura de confinamiento (resultado crudo)\n\n",
        f"Protocolo: `{PROTOCOL_ID}`\n\n",
        f"- n instancias de cierre medidas: **{summary['n_records_total']}**\n",
        f"- n con k>=2 (ratio_lig definido): **{summary['n_k_ge2']}**\n",
        f"- n con k>=3 y NULL (ratio_null definido): **{summary['n_k_ge3_with_null']}**\n",
        f"- ratio_lig (m2/m1) mediana: **{summary['ratio_lig_median']}**\n",
        f"- ratio_null (m2_REAL/m2_NULL) mediana: **{summary['ratio_null_median']}**\n",
        f"- rate_pass conjunto (a AND b, umbral {THRESH_BIG}x / {THRESH_NULL}x): **{summary['rate_pass_joint']:.3f}**\n",
        f"- T4 (NULL cae en promedio agregado): **{summary['T4_null_falls']}**\n",
        f"- mean m2_REAL (k>=3): **{summary['mean_m2_real_k3plus']:.6f}**, mean m2_NULL (k>=3): **{summary['mean_m2_null_k3plus']:.6f}**\n",
    ]
    (OUT / "CF4_RESUMEN.md").write_text("".join(md), encoding="utf-8")
    print(f"MD  -> {OUT / 'CF4_RESUMEN.md'}")
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
