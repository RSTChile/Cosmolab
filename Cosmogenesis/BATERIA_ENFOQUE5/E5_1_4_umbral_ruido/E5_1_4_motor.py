#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E5_1_4_motor.py — "Umbral de exergía frente al ruido dinámico, barrido de 8 décadas"
======================================================================================

Experimento E5.1-4 de BATERIA_ENFOQUE5 (Tema 1 — Persistencia de la exergía).
Protocolo pre-registrado en: PROTOCOLO_E5.1-4_PREREGISTRO.md (ESCRITO ANTES de este
motor; no se edita el protocolo tras ver resultados — T3). La ADENDA 2026-07-25 al
final de ese protocolo describe los dos arreglos aplicados aquí, ANTES de correr.

Reutiliza el código base cs074_rcruz.py TAL CUAL (importado por ruta, sin tocar ni una
línea de ese archivo) y le añade lo único que la ficha pide y que el código base NO
tiene: forzamiento estocástico dinámico EN CADA PASO (no solo en la condición inicial),
con amplitud barrida en 8 décadas.

Mecánica del ruido (ver protocolo §3):
    phi_{t+1} = paso_expansion(paso_difusion(phi_t, activo), H, rng)
    phi_{t+1} = phi_{t+1} + amplitud_por_paso * xi_t,   xi_t ~ N(0,1) iid por sitio y paso

Barrido (protocolo §4):
    amplitud_ruido: 17 puntos log-espaciados en [1e-8, 1] (8 décadas, 2 pts/década)
    r = H/D:        {0, 0.1, 0.3, 1, 3, 10, 30, 100}   (8 valores, rango extremo)
    semillas:       16 (control apareado real/NULL, misma semilla)
    eps_real:       1e-3 (fijo, pre-registrado)  |  NULL: eps=0, mismo H(r), mismo ruido
    N=200, pasos medidos vía medir_pasos_lavado(N=200, eps=1e-3) del código base.

Observables (protocolo §6, + ADENDA 2026-07-25 Arreglo 3):
    X_final  — persistencia() del código base tal cual (referencia = contraste inicial
               real). Para eps=0 es 0 por construcción (rama contraste0<=0).
    X_alt    — mismo cálculo (correlación x varianza normalizada) pero usando como
               referencia la escala del propio ruido inyectado (amplitud_ruido total
               pre-registrada), para que el NULL (eps=0) no sea trivialmente 0 y sirva
               de verificación cruzada independiente (T4, regla 4 "segundo
               observable/método").
    Xh_final — NUEVO (Arreglo 3): definición canónica homologada de la batería,
               exergia_X(phi) = (1/N)*sum((phi-1)^2), importada tal cual de
               _observables_homologadas.py. Se calcula EN PARALELO a X_final/X_alt
               (no los reemplaza), sobre el mismo phi final de cada corrida.

--- ADENDA 2026-07-25 (primera corrida real de este motor; ver PROTOCOLO §ADENDA) ---
Arreglo 2 (ruido calibrado): la corrida anterior (E5_1_4_stderr.txt, muerta a 19.1%)
sumaba `amplitud * rng.standard_normal(...)` CONSTANTE cada uno de los `pasos` pasos
fijos — un paseo aleatorio sin amortiguar cuya varianza acumulada crece sin tope
(~pasos * amplitud^2), lo que producía la explosión numérica observada en vivo
(X_final_real=27,221,303 a amplitud=1.0, r=0). Aquí la `amplitud` pre-registrada
(17 puntos, 1e-8..1, NO tocada) se reinterpreta como presupuesto TOTAL de ruido a
repartir en los `pasos` pasos, no como amplitud por paso: se usa
ruido_por_paso(NOISE_REL=amplitud, eps=1.0, pasos_fijo=pasos) = amplitud/sqrt(pasos)
como la amplitud real aplicada en cada paso (BATERIA_ENFOQUE5/_ruido_calibrado.py, ya
verificado en otros 4 experimentos hoy). Así la varianza acumulada total al final de
la corrida es ~amplitud^2, independiente de `pasos` — que es lo que el barrido de
8 décadas pretendía medir desde el principio.

Arreglo 3 (Xh canónica): se agrega Xh_final = exergia_X(phi) en paralelo a X_final y
X_alt, sin reemplazarlos (BATERIA_ENFOQUE5/_observables_homologadas.py).

Detalle de auditoría (paso 3 del encargo): se guarda, para las 4352 combinaciones
individuales (no solo el agregado por fila r×amplitud), sum(phi) y sum(phi^2) del phi
FINAL de cada corrida (suficiente para reconstruir Xh_final sin recomputar, porque
exergia_X(phi) = sum(phi^2)/N - 2*sum(phi)/N + 1) — en el JSON principal, ligero. El
array phi FINAL crudo completo (N=200 valores, redondeados a 6 decimales) de las 4352
corridas se guarda aparte en E5_1_4_phi_final_crudo.npz (comprimido) para no inflar el
JSON principal; los índices de fila coinciden 1:1 con el orden de `filas` en el JSON
(orden: r externo, amplitud_ruido interno, ambos en el orden de R_GRID/AMPLITUDES_RUIDO
tal como están declarados abajo).

NO se edita cs074_rcruz.py. NO se cambian los parámetros tras ver resultados (T3). NO
se toca AMPLITUDES_RUIDO, R_GRID, ni SEMILLAS (regla del proyecto).
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE_PATH = HERE.parent.parent / "cs074_rcruz.py"
BATERIA_DIR = HERE.parent.parent  # Cosmogenesis/  (para importar BATERIA_ENFOQUE5.*)
OUT = HERE

# --- Cargar el código base SIN editarlo (import por ruta) ---
_spec = importlib.util.spec_from_file_location("cs074_rcruz_base", str(BASE_PATH))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

# --- Arreglo 2 y Arreglo 3: módulos compartidos de la batería (aditivos, no editan cs074) ---
sys.path.insert(0, str(BATERIA_DIR))
from BATERIA_ENFOQUE5._ruido_calibrado import ruido_por_paso  # noqa: E402  (Arreglo 2)
from BATERIA_ENFOQUE5._observables_homologadas import exergia_X  # noqa: E402  (Arreglo 3)

# ---------------------------------------------------------------------------
# Parámetros pre-registrados (protocolo §4) — NO se tocan tras ver resultados
# ---------------------------------------------------------------------------
N = 200
EPS_REAL = 1e-3
SEMILLAS = 16
R_GRID = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
AMPLITUDES_RUIDO = list(np.logspace(-8, 0, 17))  # 17 pts log-espaciados, 8 décadas (protocolo §4, exacto)
D_CALIB_SEMILLAS = 8


def paso_con_ruido(phi, activo, H, amplitud_por_paso, rng):
    """Un paso: difusión (código base) + expansión (código base) + ruido dinámico (NUEVO).

    ADENDA 2026-07-25 (Arreglo 2): `amplitud_por_paso` ya viene calibrada por el
    llamador (ruido_por_paso = amplitud_total/sqrt(pasos)), NO es la amplitud
    pre-registrada cruda — ver docstring del módulo.
    """
    phi = base.paso_difusion(phi, activo)
    activo = base.paso_expansion(activo, H, rng)
    if amplitud_por_paso > 0.0:
        phi = phi + amplitud_por_paso * rng.standard_normal(phi.shape)
    return phi, activo


def corrida_con_ruido(N, eps, H, amplitud, pasos, seed):
    """
    Evoluciona el campo con forzamiento estocástico por paso.
    Devuelve X_final (persistencia estándar, ref=contraste inicial real),
    X_alt (persistencia con ref=amplitud_ruido TOTAL pre-registrada), Xh_final
    (definición canónica homologada, Arreglo 3), diagnóstico de deriva de sum(phi),
    y detalle crudo del phi final (sum, sum^2, array) para auditoría (paso 3 del encargo).

    ADENDA 2026-07-25 (Arreglo 2): `amplitud` sigue siendo el punto pre-registrado del
    barrido de 8 décadas (presupuesto TOTAL de ruido), pero la amplitud aplicada EN CADA
    PASO es amplitud/sqrt(pasos) (ruido_por_paso), no `amplitud` directamente — así la
    varianza acumulada total al final de la corrida es ~amplitud^2, sin importar `pasos`.
    """
    rng = np.random.default_rng(seed)
    phi, _ = base.campo_inicial(N, eps, rng)
    activo = np.ones(N, dtype=bool)
    contraste0 = float(phi.std())
    suma0 = float(phi.sum())

    amplitud_por_paso = ruido_por_paso(NOISE_REL=amplitud, eps=1.0, pasos_fijo=pasos)

    for _ in range(pasos):
        phi, activo = paso_con_ruido(phi, activo, H, amplitud_por_paso, rng)

    suma1 = float(phi.sum())
    suma1_sq = float(np.sum(phi ** 2))

    # X_final: persistencia() del código base, tal cual (ref = contraste inicial real)
    X_final = base.persistencia(phi, contraste0)

    # X_alt: mismo cálculo pero con referencia = escala TOTAL del ruido pre-registrado
    # (amplitud, no amplitud_por_paso — es la escala físicamente comparable entre puntos
    # del barrido), o contraste0 real si amplitud=0, para que en ese punto ambos
    # observables coincidan por continuidad.
    ref_alt = amplitud if amplitud > 0.0 else contraste0
    X_alt = base.persistencia(phi, ref_alt) if ref_alt > 0 else 0.0

    # Xh_final: definición canónica homologada (Arreglo 3), EN PARALELO, no reemplaza nada.
    Xh_final = exergia_X(phi)

    frac_exp = 1.0 - float(activo.mean())
    deriva_suma = abs(suma1 - suma0) / (abs(suma0) + 1e-30)

    return {
        "X_final": X_final,
        "X_alt": X_alt,
        "Xh_final": Xh_final,
        "frac_exp": frac_exp,
        "std_final": float(phi.std()),
        "deriva_suma_abs": suma1 - suma0,
        "deriva_suma_rel": deriva_suma,
        "amplitud_por_paso": amplitud_por_paso,
        "suma_phi_final": suma1,
        "suma_phi2_final": suma1_sq,
        "phi_final": phi,  # array crudo (N,) — se separa a .npz, no va al JSON principal
    }


def main():
    t0 = time.time()
    log_lines = []

    def log(msg):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, file=sys.stderr, flush=True)
        log_lines.append(line)

    log(f"E5.1-4 motor — inicio (re-corrida ADENDA 2026-07-25: Arreglo 2 ruido calibrado + Arreglo 3 Xh canonica). "
        f"N={N} eps_real={EPS_REAL} semillas={SEMILLAS}")
    log(f"amplitudes_ruido ({len(AMPLITUDES_RUIDO)} pts): {AMPLITUDES_RUIDO}")
    log(f"r_grid ({len(R_GRID)} pts): {R_GRID}")

    # --- calibración medida (no puesta a mano), igual metodología que el código base ---
    cal = base.medir_pasos_lavado(N, EPS_REAL, D_CALIB_SEMILLAS)
    pasos = cal["pasos"]
    log(f"calibracion pasos_lavado: mediana={cal['mediana']} pasos={pasos} lavo_todas={cal['lavo_todas']} tiempos={cal['tiempos']}")

    D = float(np.mean([base.medir_D(N, EPS_REAL, s) for s in range(D_CALIB_SEMILLAS)]))
    log(f"D medido (eps_real={EPS_REAL}, promedio {D_CALIB_SEMILLAS} semillas) = {D:.8f}")

    H_por_r = {}
    for r in R_GRID:
        H = float(min(r * D, 1.0)) if D > 0 else (0.0 if r == 0 else 1.0)
        H_por_r[r] = H
    log(f"H(r) derivado: {H_por_r}")

    amplitud_por_paso_map = {float(a): ruido_por_paso(NOISE_REL=a, eps=1.0, pasos_fijo=pasos) for a in AMPLITUDES_RUIDO}
    log(f"Arreglo 2 — amplitud_por_paso = amplitud_total/sqrt(pasos={pasos}) = 1/{np.sqrt(pasos):.4f}: {amplitud_por_paso_map}")

    total_corridas = len(AMPLITUDES_RUIDO) * len(R_GRID) * SEMILLAS * 2
    log(f"total corridas planificadas: {total_corridas}")

    # --- Buffers para el detalle crudo (paso 3 del encargo): phi final de las 4352 corridas ---
    n_filas = len(R_GRID) * len(AMPLITUDES_RUIDO)
    phi_final_real_buf = np.zeros((n_filas, SEMILLAS, N), dtype=np.float64)
    phi_final_null_buf = np.zeros((n_filas, SEMILLAS, N), dtype=np.float64)
    r_por_fila = np.zeros(n_filas, dtype=np.float64)
    amplitud_por_fila = np.zeros(n_filas, dtype=np.float64)

    filas = []
    n_hecho = 0
    fila_idx = 0
    t_ultimo_reporte = time.time()

    for r in R_GRID:
        H = H_por_r[r]
        for amplitud in AMPLITUDES_RUIDO:
            reales, nulls = [], []
            for s in range(SEMILLAS):
                seed = 20000 + s  # misma semilla real/NULL -> control apareado
                rr = corrida_con_ruido(N, EPS_REAL, H, amplitud, pasos, seed=seed)
                nn = corrida_con_ruido(N, 0.0, H, amplitud, pasos, seed=seed)
                reales.append(rr)
                nulls.append(nn)
                phi_final_real_buf[fila_idx, s, :] = rr["phi_final"]
                phi_final_null_buf[fila_idx, s, :] = nn["phi_final"]
                n_hecho += 2

            r_por_fila[fila_idx] = r
            amplitud_por_fila[fila_idx] = amplitud

            def agg(lst, key):
                arr = np.array([d[key] for d in lst], dtype=float)
                return float(arr.mean()), float(arr.std())

            Xf_r_m, Xf_r_s = agg(reales, "X_final")
            Xf_n_m, Xf_n_s = agg(nulls, "X_final")
            Xa_r_m, Xa_r_s = agg(reales, "X_alt")
            Xa_n_m, Xa_n_s = agg(nulls, "X_alt")
            Xh_r_m, Xh_r_s = agg(reales, "Xh_final")
            Xh_n_m, Xh_n_s = agg(nulls, "Xh_final")
            drift_r_m, _ = agg(reales, "deriva_suma_rel")
            drift_n_m, _ = agg(nulls, "deriva_suma_rel")

            filas.append({
                "r": r,
                "H": H,
                "D": D,
                "amplitud_ruido": amplitud,
                "amplitud_por_paso": amplitud_por_paso_map[float(amplitud)],
                "pasos": pasos,
                "semillas": SEMILLAS,
                "fila_idx_npz": fila_idx,
                "X_final_real_mean": Xf_r_m, "X_final_real_std": Xf_r_s,
                "X_final_null_mean": Xf_n_m, "X_final_null_std": Xf_n_s,
                "X_alt_real_mean": Xa_r_m, "X_alt_real_std": Xa_r_s,
                "X_alt_null_mean": Xa_n_m, "X_alt_null_std": Xa_n_s,
                "Xh_final_real_mean": Xh_r_m, "Xh_final_real_std": Xh_r_s,
                "Xh_final_null_mean": Xh_n_m, "Xh_final_null_std": Xh_n_s,
                "deriva_suma_rel_real_mean": drift_r_m,
                "deriva_suma_rel_null_mean": drift_n_m,
                "X_final_real_por_semilla": [d["X_final"] for d in reales],
                "X_final_null_por_semilla": [d["X_final"] for d in nulls],
                "X_alt_real_por_semilla": [d["X_alt"] for d in reales],
                "X_alt_null_por_semilla": [d["X_alt"] for d in nulls],
                "Xh_final_real_por_semilla": [d["Xh_final"] for d in reales],
                "Xh_final_null_por_semilla": [d["Xh_final"] for d in nulls],
                "suma_phi_final_real_por_semilla": [d["suma_phi_final"] for d in reales],
                "suma_phi2_final_real_por_semilla": [d["suma_phi2_final"] for d in reales],
                "suma_phi_final_null_por_semilla": [d["suma_phi_final"] for d in nulls],
                "suma_phi2_final_null_por_semilla": [d["suma_phi2_final"] for d in nulls],
            })

            fila_idx += 1

            if time.time() - t_ultimo_reporte > 30:
                frac = n_hecho / total_corridas
                elapsed = time.time() - t0
                eta = elapsed / max(frac, 1e-9) - elapsed
                log(f"progreso: {n_hecho}/{total_corridas} ({100*frac:.1f}%) elapsed={elapsed:.0f}s eta={eta:.0f}s | r={r} amplitud={amplitud:.3e} "
                    f"X_final_real={Xf_r_m:.4f} X_final_null={Xf_n_m:.4f} Xh_real={Xh_r_m:.4f} Xh_null={Xh_n_m:.4f}")
                t_ultimo_reporte = time.time()

    elapsed_total = time.time() - t0
    log(f"listo. {n_hecho} corridas en {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")

    # --- auditoría de la garantía de construcción: NULL X_final debe ser 0 en toda la grilla ---
    null_Xfinal_maxabs = max(abs(f["X_final_null_mean"]) for f in filas)
    log(f"auditoria: max|X_final_null_mean| en toda la grilla = {null_Xfinal_maxabs:.3e} (debe ser ~0 por construccion, contraste0=0 con eps=0)")

    # --- auditoría de que el Arreglo 2 realmente acota el ruido (no debe haber explosion) ---
    max_abs_phi = float(max(np.max(np.abs(phi_final_real_buf)), np.max(np.abs(phi_final_null_buf))))
    log(f"auditoria Arreglo 2: max|phi_final| en toda la grilla (real+null) = {max_abs_phi:.6f} "
        f"(la corrida vieja SIN arreglo llegaba a X_final~2.7e7 a amplitud=1.0,r=0 -> exploto; aqui debe quedar acotado)")

    result = {
        "experimento": "E5.1-4 — Umbral de exergia frente al ruido dinamico, barrido de 8 decadas",
        "adenda": "2026-07-25: Arreglo 2 (ruido calibrado por sqrt(pasos)) + Arreglo 3 (Xh canonica en paralelo). Ver PROTOCOLO_E5.1-4_PREREGISTRO.md, seccion ADENDA.",
        "codigo_base": str(BASE_PATH),
        "codigo_base_editado": False,
        "parametros": {
            "N": N,
            "eps_real": EPS_REAL,
            "semillas": SEMILLAS,
            "r_grid": R_GRID,
            "amplitudes_ruido": list(AMPLITUDES_RUIDO),
            "amplitud_por_paso_por_amplitud": amplitud_por_paso_map,
            "pasos": pasos,
            "D_medido": D,
            "H_por_r": H_por_r,
        },
        "calibracion_pasos_lavado": cal,
        "auditoria_null_X_final_maxabs": null_Xfinal_maxabs,
        "auditoria_arreglo2_max_abs_phi_final": max_abs_phi,
        "total_corridas": n_hecho,
        "elapsed_s": elapsed_total,
        "filas": filas,
        "log": log_lines,
        "detalle_crudo_phi_final": {
            "archivo_npz": "E5_1_4_phi_final_crudo.npz",
            "nota": "phi final (N=200, redondeado a 6 decimales) de las 4352 corridas individuales, "
                    "arrays phi_final_real[fila_idx, semilla, N] y phi_final_null[fila_idx, semilla, N]. "
                    "fila_idx coincide con 'fila_idx_npz' de cada fila del JSON (orden r externo, "
                    "amplitud_ruido interno). sum(phi) y sum(phi^2) por corrida ya estan en las listas "
                    "'suma_phi_final_*_por_semilla' / 'suma_phi2_final_*_por_semilla' de cada fila (suficiente "
                    "para reconstruir Xh_final = suma_phi2/N - 2*suma_phi/N + 1 sin recomputar).",
        },
    }

    out_json = OUT / "E5_1_4_resultado.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"escrito: {out_json}")

    out_npz = OUT / "E5_1_4_phi_final_crudo.npz"
    np.savez_compressed(
        out_npz,
        phi_final_real=np.round(phi_final_real_buf, 6),
        phi_final_null=np.round(phi_final_null_buf, 6),
        r_por_fila=r_por_fila,
        amplitud_por_fila=amplitud_por_fila,
    )
    log(f"escrito: {out_npz} ({out_npz.stat().st_size / 1e6:.1f} MB)")

    (OUT / "E5_1_4_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    print(json.dumps({"ok": True, "filas": len(filas), "elapsed_s": elapsed_total, "out": str(out_json), "out_npz": str(out_npz)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
