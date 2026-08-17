#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E5.3-5 — Test de falsación externo: distancia emergente al 4.9%/31.5%, sin ajuste
====================================================================================

Motor propio del agente E5.3-5 (batería Enfoque 5, 30 experimentos en paralelo).
Pre-registro congelado ANTES de este archivo:
  PROTOCOLO_E5.3-5_PREREGISTRO.md (mismo directorio)

Reutiliza SIN EDITAR:
  - cs074_rcruz.py (campo_inicial, paso_difusion, paso_expansion, medir_D)
  - ../E5_3_2_eficiencia_vs_ligadura/E5_3_2_motor.py (segmentos_desde_activo,
    descomponer_energia, energia_total_inicial, y su función barrido() completa si hace
    falta recomputar esa parte)

Este experimento NO inventa una tercera definición de eficiencia. Implementa, en la parte
A, la definición CANÓNICA congelada en PROTOCOLO_E5.3-1_PREREGISTRO.md (motor E5.3-1 no
estaba en disco al momento de correr -> se reconstruye fielmente desde su texto, reutilizando
la función de segmentación de E5.3-2 porque son el mismo objeto topológico, ver protocolo
§2). En la parte B, ejecuta (o carga si ya existe) el motor E5.3-2 completo, sin tocarlo.

Luego SOLO agrega: para cada celda y cada corrida individual de ambos grids, calcula
  dist_49  = |eficiencia_real - 0.049|
  dist_315 = |eficiencia_real - 0.315|
y reporta la distribución completa. Ningún valor 0.049/0.315 entra en la física del motor
(campo, difusión, expansión, ruido) en ningún punto -- solo en esta etapa de comparación
final, después de tener la curva completa (regla de oro #2 del documento madre).
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE_CODE = HERE.parent.parent / "cs074_rcruz.py"
E532_DIR = HERE.parent / "E5_3_2_eficiencia_vs_ligadura"
E532_MOTOR = E532_DIR / "E5_3_2_motor.py"
E532_JSON = E532_DIR / "E5_3_2_resultado_crudo.json"


def _import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


cs074 = _import_module(BASE_CODE, "cs074_rcruz_base_e535")
campo_inicial = cs074.campo_inicial
paso_difusion = cs074.paso_difusion
paso_expansion = cs074.paso_expansion
medir_D = cs074.medir_D

e532 = _import_module(E532_MOTOR, "e532_motor_e535")
segmentos_desde_activo = e532.segmentos_desde_activo

# ---------------- Constantes declaradas en el pre-registro (T1) ----------------
BLANCO_49 = 0.049
BLANCO_315 = 0.315
UMBRAL_CERCA = 0.02  # puntos porcentuales, fijado en el pre-registro, igual para ambos blancos

# --- Parte A (definición E5.3-1: dominios aislados, energía total dentro) ---
N_A = 200
SEMILLAS_A = 20
EPS_LIST_A = [0.0] + list(np.logspace(-12, 0, 13))  # eps=0 = control, excluido del grid ppal
R_LIST_A = list(np.logspace(-3, 3, 13))
H_FLOOR = 1e-6
PASOS_MIN, PASOS_MAX = 100, 3000
NOISE_COEF = 0.01  # amplitud = NOISE_COEF * sqrt(D_eps), declarado en PROTOCOLO_E5.3-1


def pasos_de_H(H: float) -> int:
    return int(min(max(math.ceil(5.0 / max(H, H_FLOOR)), PASOS_MIN), PASOS_MAX))


def dominios_aislados(activo: np.ndarray):
    """Componentes conexas del anillo vía aristas activas == segmentos de E5.3-2.
    Devuelve lista de arrays de indices; excluye el caso de un unico segmento de tamano N
    (no aislado -> no cuenta como ligado) dejando que el llamador filtre por tamano < N.
    """
    return segmentos_desde_activo(activo)


def corrida_a(N: int, eps: float, r: float, D: float, noise_amp: float, H: float,
              pasos: int, seed: int):
    rng = np.random.default_rng(seed)
    phi0, _ = campo_inicial(N, eps, rng)
    E_total = float(np.sum((phi0 - 1.0) ** 2))

    phi = phi0.copy()
    activo = np.ones(N, dtype=bool)
    for _ in range(pasos):
        phi = paso_difusion(phi, activo)
        if noise_amp > 0.0:
            phi = phi + rng.normal(0.0, noise_amp, size=phi.shape)
        activo = paso_expansion(activo, H, rng)

    segmentos = dominios_aislados(activo)
    aislados = [s for s in segmentos if 1 <= s.size < N]
    if aislados:
        idx_lig = np.concatenate(aislados)
        e_ligada = float(np.sum((phi[idx_lig] - 1.0) ** 2))
    else:
        e_ligada = 0.0
    n_dominios_aislados = len(aislados)

    if E_total <= 0.0:
        eficiencia_real = None  # indefinida (0/0), solo eps=0
    else:
        eficiencia_real = e_ligada / E_total
        # guardia T6: E_ligada no puede exceder E_total (estructura total dentro de
        # dominios aislados es un subconjunto de la energia total del campo final,
        # pero el campo puede haber GANADO varianza local por el ruido dinamico; se
        # audita, no se fuerza)
        # (verificacion agregada fuera, no aqui, para no ocultar violaciones)

    # NULL: permutar valores finales de phi manteniendo la particion (misma rng, tirada
    # adicional), igual principio que E5.3-1 y E5.3-2
    phi_null = rng.permutation(phi)
    if aislados:
        e_ligada_n = float(np.sum((phi_null[idx_lig] - 1.0) ** 2))
    else:
        e_ligada_n = 0.0
    eficiencia_null = None if E_total <= 0.0 else e_ligada_n / E_total

    frac_exp = 1.0 - float(activo.mean())

    return {
        "E_total": E_total,
        "e_ligada": e_ligada,
        "eficiencia_real": eficiencia_real,
        "eficiencia_null": eficiencia_null,
        "n_dominios_aislados": n_dominios_aislados,
        "n_segmentos": len(segmentos),
        "frac_exp": frac_exp,
        "e_ligada_leq_E_total": bool(e_ligada <= E_total * (1.0 + 1e-6) + 1e-12),
    }


def barrido_a(eps_list=None, r_list=None, semillas=SEMILLAS_A, N=N_A, verbose=True):
    eps_list = EPS_LIST_A if eps_list is None else eps_list
    r_list = R_LIST_A if r_list is None else r_list
    t0 = time.time()
    filas = []
    control_filas = []
    for eps in eps_list:
        Ds = [medir_D(N, eps, s) for s in range(max(semillas, 4))]
        D = float(np.mean(Ds))
        noise_amp = NOISE_COEF * math.sqrt(D) if D > 0 else 0.0
        for r in r_list:
            H = float(min(r * D, 1.0)) if D > 0 else 0.0
            pasos = pasos_de_H(H)
            eff_real, eff_null, n_dom, fracs, e_leq, e_totals = [], [], [], [], [], []
            for s in range(semillas):
                res = corrida_a(N, eps, r, D, noise_amp, H, pasos, seed=20_000 + s)
                if res["eficiencia_real"] is not None:
                    eff_real.append(res["eficiencia_real"])
                    eff_null.append(res["eficiencia_null"])
                n_dom.append(res["n_dominios_aislados"])
                fracs.append(res["frac_exp"])
                e_leq.append(res["e_ligada_leq_E_total"])
                e_totals.append(res["E_total"])
            # --- Diagnostico SNR (NO altera la fisica; solo audita, agregado tras el
            # calculo, T6): compara el presupuesto inicial E_total vs. la varianza que el
            # ruido dinamico (0.01*sqrt(D_eps) por paso, N sitios) acumula en 'pasos' pasos
            # como caminata aleatoria independiente por sitio. Si E_total << ruido
            # acumulado esperado, la 'eficiencia' de esa celda esta dominada por ruido, no
            # por estructura -- se reporta el hecho, no se oculta ni se corrige el motor.
            ruido_acumulado_esperado = N * (noise_amp ** 2) * pasos
            E_total_medio = float(np.mean(e_totals)) if e_totals else 0.0
            snr = (E_total_medio / ruido_acumulado_esperado) if ruido_acumulado_esperado > 0 else float("inf")
            fila = {
                "eps": float(eps), "r": float(r), "D": D, "H": H, "pasos": pasos,
                "noise_amp": noise_amp, "N": N, "semillas": semillas,
                "indefinida": len(eff_real) == 0,
                "E_total_medio": E_total_medio,
                "ruido_acumulado_esperado": ruido_acumulado_esperado,
                "razon_senal_ruido": snr,
                "snr_saludable": bool(snr > 1.0),
            }
            if eff_real:
                rr = np.array(eff_real); nn = np.array(eff_null)
                sd = np.sqrt((rr.var() + nn.var()) / 2.0)
                sd = max(sd, 1.0 / max(len(rr), 1))
                z = float((rr.mean() - nn.mean()) / sd)
                fila.update({
                    "eficiencia_real_media": float(rr.mean()),
                    "eficiencia_real_std": float(rr.std()),
                    "eficiencia_real_por_semilla": [float(x) for x in rr],
                    "eficiencia_null_media": float(nn.mean()),
                    "eficiencia_null_std": float(nn.std()),
                    "eficiencia_null_por_semilla": [float(x) for x in nn],
                    "z": z,
                })
            fila["n_dominios_aislados_medio"] = float(np.mean(n_dom))
            fila["frac_exp_medio"] = float(np.mean(fracs))
            fila["conservacion_ok_todas"] = bool(all(e_leq))
            if eps == 0.0:
                control_filas.append(fila)
            else:
                filas.append(fila)
        if verbose:
            print(f"[E5.3-5 parte-A eps={eps:.3g}] D={D:.6g} noise_amp={noise_amp:.3g} "
                  f"({len(r_list)} r-puntos x {semillas} semillas listos) t={time.time()-t0:.1f}s",
                  file=sys.stderr, flush=True)
    return {
        "filas": filas,
        "control_eps0": control_filas,
        "elapsed_s": time.time() - t0,
        "N": N, "semillas": semillas,
        "eps_list": [float(e) for e in eps_list if e != 0.0],
        "r_list": [float(r) for r in r_list],
        "H_floor": H_FLOOR, "pasos_min": PASOS_MIN, "pasos_max": PASOS_MAX,
        "noise_coef": NOISE_COEF,
        "definicion": (
            "eficiencia = E_ligada/E_total; E_ligada = suma de (phi_final-1)^2 en dominios "
            "AISLADOS (componentes conexas de tamano<N via aristas activas remanentes); "
            "E_total = suma de (phi0-1)^2 al inicio de esa corrida. Definicion CANONICA "
            "transcrita de PROTOCOLO_E5.3-1_PREREGISTRO.md (motor E5.3-1 no estaba en disco; "
            "reconstruccion fiel desde el texto congelado, ejecutada por el agente E5.3-5)."
        ),
    }


def obtener_parte_b(forzar_recompute=False, verbose=True, grid_pequeno=False):
    """Carga E5_3_2_resultado_crudo.json si existe; si no, ejecuta e532.barrido() (motor
    ajeno sin editar) con su grid de PRODUCCION completo tal como esta escrito.

    grid_pequeno=True (solo para modo smoke de ESTE motor, nunca en produccion): sobre-
    escribe temporalmente los atributos de modulo de e532 (mismo patron de auto-prueba que
    uso el propio agente E5.3-2 para su humo) con un subconjunto pequeno, SOLO para validar
    el cableado del agregado; nunca se usa para el resultado final reportado."""
    if not forzar_recompute and E532_JSON.exists():
        if verbose:
            print(f"[E5.3-5 parte-B] usando resultado en disco: {E532_JSON}", file=sys.stderr)
        data = json.loads(E532_JSON.read_text(encoding="utf-8"))
        return data, "cargado_de_disco"
    if verbose:
        print("[E5.3-5 parte-B] E5_3_2_resultado_crudo.json no existe -> ejecutando "
              "E5_3_2_motor.barrido() (motor ajeno, sin editar)"
              + (" con grid REDUCIDO de humo" if grid_pequeno else " con su grid de produccion"),
              file=sys.stderr, flush=True)
    if grid_pequeno:
        e532.SEMILLAS = 3
        e532.EPS_LIST = [0.0, 1e-3, 1.0]
        e532.L_LIST = [1e-3, 1.0, 1e2]
    resultado = e532.barrido()
    return resultado, ("recomputado_por_E5_3_5_SMOKE" if grid_pequeno else "recomputado_por_E5_3_5")


def distancias(eficiencia):
    return {
        "dist_49": abs(eficiencia - BLANCO_49),
        "dist_315": abs(eficiencia - BLANCO_315),
    }


def agregar_distancias(filas_a, filas_b):
    """Construye la distribucion conjunta de distancias, por celda (media sobre semillas)
    y por corrida individual (semilla a semilla), separadas por metodo."""
    celdas = []
    corridas = []

    for f in filas_a:
        if f.get("indefinida"):
            continue
        d = distancias(f["eficiencia_real_media"])
        celdas.append({"metodo": "E5.3-1_dominios", "eps": f["eps"], "r": f["r"],
                        "eficiencia_media": f["eficiencia_real_media"],
                        "eficiencia_std": f["eficiencia_real_std"], "z": f["z"],
                        "snr_saludable": f.get("snr_saludable"),
                        "razon_senal_ruido": f.get("razon_senal_ruido"),
                        **d})
        for val in f["eficiencia_real_por_semilla"]:
            dd = distancias(val)
            corridas.append({"metodo": "E5.3-1_dominios", "eps": f["eps"], "r": f["r"],
                              "eficiencia": val, "snr_saludable": f.get("snr_saludable"), **dd})

    for f in filas_b:
        d = distancias(f["eficiencia_real_media"])
        celdas.append({"metodo": "E5.3-2_ANOVA", "eps": f["eps"], "L": f["L"],
                        "eficiencia_media": f["eficiencia_real_media"],
                        "eficiencia_std": f["eficiencia_real_std"], "z": f["z"],
                        **d})
        for val in f["eficiencia_real_por_semilla"]:
            dd = distancias(val)
            corridas.append({"metodo": "E5.3-2_ANOVA", "eps": f["eps"], "L": f["L"],
                              "eficiencia": val, **dd})

    return celdas, corridas


def resumen_estadistico(valores):
    if not valores:
        return None
    arr = np.array(valores, dtype=float)
    return {
        "n": int(arr.size),
        "min": float(arr.min()), "max": float(arr.max()),
        "media": float(arr.mean()), "mediana": float(np.median(arr)),
        "std": float(arr.std()),
        "p05": float(np.percentile(arr, 5)), "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)), "p95": float(np.percentile(arr, 95)),
        "frac_menor_0.02": float(np.mean(arr < 0.02)),
        "frac_menor_0.05": float(np.mean(arr < 0.05)),
        "frac_menor_0.10": float(np.mean(arr < 0.10)),
    }


def main():
    modo = sys.argv[1] if len(sys.argv) > 1 else "produccion"
    t0 = time.time()

    if modo == "smoke":
        eps_list = [0.0, 1e-6, 1e-3, 1.0]
        r_list = [1e-2, 1.0, 1e2]
        semillas_a = 3
        resA = barrido_a(eps_list=eps_list, r_list=r_list, semillas=semillas_a)
        resB, fuente_b = obtener_parte_b(grid_pequeno=True)
        filas_b = resB["filas"]
    elif modo == "produccion":
        resA = barrido_a()
        resB, fuente_b = obtener_parte_b()
        filas_b = resB["filas"]
    else:
        raise SystemExit(f"modo desconocido: {modo} (usa smoke|produccion)")

    celdas, corridas = agregar_distancias(resA["filas"], filas_b)

    out_a = HERE / f"E5_3_5_resultado_e531def_{modo}.json"
    out_a.write_text(json.dumps(resA, indent=2, ensure_ascii=False), encoding="utf-8")

    out_b = HERE / f"E5_3_5_resultado_e532_{modo}.json"
    out_b.write_text(json.dumps({"fuente": fuente_b, "resultado": resB}, indent=2, ensure_ascii=False),
                      encoding="utf-8")

    dist49_celdas = [c["dist_49"] for c in celdas]
    dist315_celdas = [c["dist_315"] for c in celdas]
    dist49_corridas = [c["dist_49"] for c in corridas]
    dist315_corridas = [c["dist_315"] for c in corridas]

    # Diagnostico SNR (solo aplica a metodo E5.3-1_dominios; E5.3-2_ANOVA liga el ruido a
    # eps por diseno propio y no muestra esta patologia -- snr_saludable=None ahi, se
    # incluye igual en el conjunto "sano" para no perder esas celdas del resumen filtrado).
    celdas_sanas = [c for c in celdas if c.get("snr_saludable") in (True, None)]
    dist49_sanas = [c["dist_49"] for c in celdas_sanas]
    dist315_sanas = [c["dist_315"] for c in celdas_sanas]
    n_contaminadas = sum(1 for c in celdas if c.get("snr_saludable") is False)

    agregado = {
        "experimento": "E5_3_5_falsacion_externa",
        "modo": modo,
        "blanco_49": BLANCO_49, "blanco_315": BLANCO_315, "umbral_cerca": UMBRAL_CERCA,
        "fuente_parte_b": fuente_b,
        "n_celdas": len(celdas), "n_corridas": len(corridas),
        "n_celdas_snr_contaminadas_E5_3_1": n_contaminadas,
        "aviso_snr": (
            "En el metodo E5.3-1_dominios, el ruido dinamico esta atado a sqrt(D_eps), NO a "
            "eps (asi lo especifica PROTOCOLO_E5.3-1_PREREGISTRO.md SS2.1). D_eps es casi "
            "constante entre epsilons (~8e-4), mientras E_total escala como eps^2*N. Para "
            "eps por debajo de un umbral medido (~0.016 en este grid), el ruido acumulado "
            "en 'pasos' pasos domina el presupuesto inicial y 'eficiencia' deja de medir "
            "estructura para medir ruido acumulado (valores hasta >1e8, viola E_ligada<=E_total). "
            "Ver snr_saludable/razon_senal_ruido por celda. Esto NO se corrigio (no es motor "
            "propio) -- se reporta a CS integro."
        ),
        "resumen_dist49_por_celda": resumen_estadistico(dist49_celdas),
        "resumen_dist315_por_celda": resumen_estadistico(dist315_celdas),
        "resumen_dist49_por_corrida": resumen_estadistico(dist49_corridas),
        "resumen_dist315_por_corrida": resumen_estadistico(dist315_corridas),
        "resumen_dist49_por_celda_SOLO_SNR_SANAS": resumen_estadistico(dist49_sanas),
        "resumen_dist315_por_celda_SOLO_SNR_SANAS": resumen_estadistico(dist315_sanas),
        "celdas_mas_cercanas_a_49": sorted(celdas, key=lambda c: c["dist_49"])[:15],
        "celdas_mas_cercanas_a_315": sorted(celdas, key=lambda c: c["dist_315"])[:15],
        "celdas_sanas_mas_cercanas_a_49": sorted(celdas_sanas, key=lambda c: c["dist_49"])[:15],
        "celdas_sanas_mas_cercanas_a_315": sorted(celdas_sanas, key=lambda c: c["dist_315"])[:15],
        "celdas": celdas,
        "elapsed_s": time.time() - t0,
    }
    out_c = HERE / f"E5_3_5_agregado_distancias_{modo}.json"
    out_c.write_text(json.dumps(agregado, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[archivo A] {out_a}", file=sys.stderr)
    print(f"[archivo B] {out_b}", file=sys.stderr)
    print(f"[archivo agregado] {out_c}", file=sys.stderr)
    print(f"[elapsed total] {agregado['elapsed_s']:.1f}s", file=sys.stderr)
    print(json.dumps({
        "n_celdas": agregado["n_celdas"], "n_corridas": agregado["n_corridas"],
        "n_celdas_snr_contaminadas_E5_3_1": agregado["n_celdas_snr_contaminadas_E5_3_1"],
        "resumen_dist49_por_celda": agregado["resumen_dist49_por_celda"],
        "resumen_dist315_por_celda": agregado["resumen_dist315_por_celda"],
        "resumen_dist49_por_celda_SOLO_SNR_SANAS": agregado["resumen_dist49_por_celda_SOLO_SNR_SANAS"],
        "resumen_dist315_por_celda_SOLO_SNR_SANAS": agregado["resumen_dist315_por_celda_SOLO_SNR_SANAS"],
        "top5_cerca_49": agregado["celdas_mas_cercanas_a_49"][:5],
        "top5_cerca_315": agregado["celdas_mas_cercanas_a_315"][:5],
        "top5_sanas_cerca_49": agregado["celdas_sanas_mas_cercanas_a_49"][:5],
        "top5_sanas_cerca_315": agregado["celdas_sanas_mas_cercanas_a_315"][:5],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
