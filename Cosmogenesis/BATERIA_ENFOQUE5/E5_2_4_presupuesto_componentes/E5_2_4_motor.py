#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E5.2-4 — Presupuesto por componentes: ¿en qué se reparte E a lo largo del barrido?
====================================================================================

Agente: CC (E5.2-4), batería Enfoque 5 (30 experimentos en paralelo, prefijo propio).
Protocolo congelado ANTES de este motor: PROTOCOLO_E5.2-4_PREREGISTRO.md (mismo directorio).

Este archivo IMPORTA cs074_rcruz.py sin modificarlo (campo_inicial, paso_difusion,
paso_expansion, medir_D, medir_pasos_lavado). Toda la física es la de la base; lo único
propio de esta pieza es la partición del presupuesto de varianza en tres componentes.

Definición (ver protocolo §3):
  E_total    = Var(phi_0)                                  (presupuesto declarado, E1)
  X          = Var_within(phi_final, activo_final)          (ANOVA, dentro de arcos vivos)
  Ligada     = Var_between(phi_final, activo_final)         (ANOVA, entre arcos aislados)
  Degradada  = E_total - Var(phi_final)                     (residuo == telescópico, T2)

Identidad exacta por construcción: X + Ligada + Degradada = E_total, en todo checkpoint.

NULL: se permuta phi_final conservando activo_final (barajado, sin re-correr la física).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE_DIR = HERE.parent.parent  # Cosmogenesis/
sys.path.insert(0, str(BASE_DIR))
import cs074_rcruz as base  # noqa: E402  (import tras ajustar sys.path)

# ---------------------------------------------------------------------------
# Parámetros congelados en el protocolo (NO se tocan tras ver resultados)
# ---------------------------------------------------------------------------
N = 200
EPS_LIST = [1e-3, 1e-2, 0.1, 0.3, 1.0]
R_LIST = list(np.logspace(-3, 3, 13))
SEMILLAS = 12
N_CHECKPOINTS = 7
CAL_EPS = 1e-3
CAL_SEMILLAS = 6
MARGEN_LAVADO = base.MARGEN_LAVADO
TOL_CONSERVACION = 1e-9  # tolerancia relativa para el chequeo T6


def componentes_conectadas(activo: np.ndarray) -> list[np.ndarray]:
    """
    Componentes conexas del anillo dado el patrón de aristas vivas `activo`
    (activo[i] = arista entre nodo i y nodo (i+1) mod N). Como el grafo base es
    un ciclo simple, cortar m aristas produce exactamente m arcos contiguos
    (o 1 componente = todo el anillo si no se cortó nada).
    """
    Nn = len(activo)
    cortes = np.where(~activo)[0]
    if len(cortes) == 0:
        return [np.arange(Nn)]
    cortes = np.sort(cortes)
    comps = []
    n_cortes = len(cortes)
    for idx in range(n_cortes):
        start = (cortes[idx] + 1) % Nn
        end = cortes[(idx + 1) % n_cortes]
        if start <= end:
            comp = np.arange(start, end + 1)
        else:
            comp = np.concatenate([np.arange(start, Nn), np.arange(0, end + 1)])
        comps.append(comp)
    return comps


def anova_particion(phi: np.ndarray, activo: np.ndarray):
    """
    Descomposición ANOVA exacta: Var_total = Var_within + Var_between (poblacional,
    ddof=0), sobre las componentes conexas definidas por `activo`.
    Devuelve (var_within, var_between, n_componentes, tam_max_componente).
    """
    Nn = len(phi)
    mu = phi.mean()
    comps = componentes_conectadas(activo)
    var_within = 0.0
    var_between = 0.0
    tam_max = 0
    for c in comps:
        vals = phi[c]
        n_c = len(c)
        tam_max = max(tam_max, n_c)
        m_c = vals.mean()
        var_within += n_c * vals.var()
        var_between += n_c * (m_c - mu) ** 2
    return var_within / Nn, var_between / Nn, len(comps), tam_max


def corrida_presupuesto(N, eps, H, pasos, seed, n_checkpoints=N_CHECKPOINTS):
    """
    Una corrida física (idéntica en mecánica a cs074_rcruz.evolucionar, pero con
    contabilidad paso a paso de la varianza perdida por difusión). Devuelve el
    estado final + checkpoints intermedios + observables auxiliares de la base.
    """
    rng = np.random.default_rng(seed)
    phi, _ = base.campo_inicial(N, eps, rng)
    activo = np.ones(N, dtype=bool)
    E_total = float(phi.var())

    checkpoint_steps = sorted(set(np.linspace(0, pasos, n_checkpoints, dtype=int).tolist()))
    if pasos not in checkpoint_steps:
        checkpoint_steps.append(pasos)
    checkpoint_set = set(checkpoint_steps)

    checkpoints = []
    cum_degradada = 0.0

    if E_total <= 0:
        # caso degenerado (no debería ocurrir: eps=0 excluido del barrido)
        return None

    if 0 in checkpoint_set:
        vw0, vb0, n_comp0, tam_max0 = anova_particion(phi, activo)
        checkpoints.append(
            {
                "paso": 0,
                "X": vw0,
                "Ligada": vb0,
                "Degradada_residuo": E_total - vw0 - vb0,
                "Degradada_telescopica": 0.0,
                "n_componentes": n_comp0,
                "tam_max_componente": tam_max0,
                "frac_exp": 0.0,
            }
        )

    for t in range(1, pasos + 1):
        var_antes = float(phi.var())
        phi = base.paso_difusion(phi, activo)
        var_despues = float(phi.var())
        cum_degradada += (var_antes - var_despues)
        activo = base.paso_expansion(activo, H, rng)
        if t in checkpoint_set:
            vw, vb, n_comp, tam_max = anova_particion(phi, activo)
            degradada_residuo = E_total - vw - vb
            checkpoints.append(
                {
                    "paso": t,
                    "X": vw,
                    "Ligada": vb,
                    "Degradada_residuo": degradada_residuo,
                    "Degradada_telescopica": cum_degradada,
                    "n_componentes": n_comp,
                    "tam_max_componente": tam_max,
                    "frac_exp": 1.0 - float(activo.mean()),
                }
            )

    phi_final = phi
    activo_final = activo

    # --- REAL ---
    X_real, Ligada_real, n_comp_final, tam_max_final = anova_particion(phi_final, activo_final)
    Degradada_real_residuo = E_total - X_real - Ligada_real
    Degradada_real_telescopica = cum_degradada
    deriva_conservacion = abs(
        (X_real + Ligada_real + Degradada_real_residuo) - E_total
    ) / E_total
    deriva_dos_metodos = abs(Degradada_real_residuo - Degradada_real_telescopica) / E_total

    # --- NULL: barajar phi_final, MISMO activo_final, sin re-correr física ---
    phi_null = rng.permutation(phi_final)
    X_null, Ligada_null, _, _ = anova_particion(phi_null, activo_final)
    Degradada_null = E_total - X_null - Ligada_null

    # --- observables auxiliares de la base (contexto, no juez) ---
    contraste0 = float(np.sqrt(E_total))
    P_real = base.persistencia(phi_final, contraste0)
    std_ratio_real = float(phi_final.std() / contraste0) if contraste0 > 0 else 0.0

    max_deriva_checkpoints = 0.0
    for cp in checkpoints:
        suma = cp["X"] + cp["Ligada"] + cp["Degradada_residuo"]
        d = abs(suma - E_total) / E_total
        max_deriva_checkpoints = max(max_deriva_checkpoints, d)
        d2 = abs(cp["Degradada_residuo"] - cp["Degradada_telescopica"]) / E_total
        max_deriva_checkpoints = max(max_deriva_checkpoints, d2)

    return {
        "E_total": E_total,
        "X_real": X_real,
        "Ligada_real": Ligada_real,
        "Degradada_real_residuo": Degradada_real_residuo,
        "Degradada_real_telescopica": Degradada_real_telescopica,
        "X_null": X_null,
        "Ligada_null": Ligada_null,
        "Degradada_null": Degradada_null,
        "n_componentes_final": n_comp_final,
        "tam_max_componente_final": tam_max_final,
        "frac_exp_final": 1.0 - float(activo_final.mean()),
        "deriva_conservacion_final": deriva_conservacion,
        "deriva_dos_metodos_final": deriva_dos_metodos,
        "max_deriva_checkpoints": max_deriva_checkpoints,
        "P_real_aux": P_real,
        "std_ratio_real_aux": std_ratio_real,
        "checkpoints": checkpoints,
    }


def main():
    t0 = time.time()

    print(f"[calibracion] midiendo pasos de lavado en N={N}, eps={CAL_EPS} ...", file=sys.stderr, flush=True)
    cal = base.medir_pasos_lavado(N, CAL_EPS, CAL_SEMILLAS)
    pasos_fijo = cal["pasos"]
    print(
        f"[calibracion] mediana_lavado={cal['mediana']} pasos_fijo={pasos_fijo} "
        f"lavo_todas={cal['lavo_todas']} tiempos={cal['tiempos']}",
        file=sys.stderr,
        flush=True,
    )

    filas = []
    n_total = len(EPS_LIST) * len(R_LIST) * SEMILLAS
    n_hecho = 0
    peor_deriva_global = 0.0

    for eps in EPS_LIST:
        D = float(np.mean([base.medir_D(N, eps, s) for s in range(SEMILLAS)]))
        for r_tgt in R_LIST:
            if D > 0:
                H = float(min(r_tgt * D, 1.0))
                r_eff = H / D
            else:
                H = 0.0
                r_eff = float("nan")

            fila_semillas = []
            for s in range(SEMILLAS):
                seed = 2_000_000 + s
                res = corrida_presupuesto(N, eps, H, pasos_fijo, seed=seed)
                if res is None:
                    continue
                fila_semillas.append(res)
                peor_deriva_global = max(peor_deriva_global, res["max_deriva_checkpoints"])
                n_hecho += 1

            def arr(key):
                return np.array([f[key] for f in fila_semillas], dtype=float)

            E_tot_arr = arr("E_total")
            Xr = arr("X_real")
            Lr = arr("Ligada_real")
            Dr = arr("Degradada_real_residuo")
            Xn = arr("X_null")
            Ln = arr("Ligada_null")
            Dn = arr("Degradada_null")

            filas.append(
                {
                    "eps": eps,
                    "D": D,
                    "r_target": r_tgt,
                    "H": H,
                    "r_efectivo": r_eff,
                    "pasos": pasos_fijo,
                    "semillas": SEMILLAS,
                    "E_total_mean": float(E_tot_arr.mean()),
                    "X_real_frac_mean": float((Xr / E_tot_arr).mean()),
                    "X_real_frac_std": float((Xr / E_tot_arr).std()),
                    "Ligada_real_frac_mean": float((Lr / E_tot_arr).mean()),
                    "Ligada_real_frac_std": float((Lr / E_tot_arr).std()),
                    "Degradada_real_frac_mean": float((Dr / E_tot_arr).mean()),
                    "Degradada_real_frac_std": float((Dr / E_tot_arr).std()),
                    "X_null_frac_mean": float((Xn / E_tot_arr).mean()),
                    "Ligada_null_frac_mean": float((Ln / E_tot_arr).mean()),
                    "Degradada_null_frac_mean": float((Dn / E_tot_arr).mean()),
                    "max_deriva_checkpoints": float(max(f["max_deriva_checkpoints"] for f in fila_semillas)) if fila_semillas else None,
                    "n_componentes_final_mean": float(np.mean([f["n_componentes_final"] for f in fila_semillas])) if fila_semillas else None,
                    "P_real_aux_mean": float(np.mean([f["P_real_aux"] for f in fila_semillas])) if fila_semillas else None,
                    "std_ratio_real_aux_mean": float(np.mean([f["std_ratio_real_aux"] for f in fila_semillas])) if fila_semillas else None,
                    "por_semilla": fila_semillas,
                }
            )
            print(
                f"[{n_hecho}/{n_total}] eps={eps:g} r_tgt={r_tgt:.4g} H={H:.4g} "
                f"X={filas[-1]['X_real_frac_mean']:.4f} Lig={filas[-1]['Ligada_real_frac_mean']:.4f} "
                f"Deg={filas[-1]['Degradada_real_frac_mean']:.4f} peor_deriva={peor_deriva_global:.2e}",
                file=sys.stderr,
                flush=True,
            )

    result = {
        "experimento": "E5.2-4 presupuesto por componentes",
        "protocolo": "PROTOCOLO_E5.2-4_PREREGISTRO.md",
        "N": N,
        "eps_list": EPS_LIST,
        "r_list": R_LIST,
        "semillas": SEMILLAS,
        "pasos_fijo": pasos_fijo,
        "calibracion_lavado": cal,
        "tol_conservacion": TOL_CONSERVACION,
        "peor_deriva_global": peor_deriva_global,
        "filas": filas,
        "elapsed_s": time.time() - t0,
    }

    out_json = HERE / "E5_2_4_resultado_crudo.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[archivo] {out_json}", file=sys.stderr)
    print(f"[peor_deriva_global] {peor_deriva_global:.3e} (tolerancia {TOL_CONSERVACION:.0e})", file=sys.stderr)
    print(f"[elapsed] {result['elapsed_s']:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
