#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E5_3_3_motor.py — Estabilidad temporal de la eficiencia (¿congela o deriva?)
=============================================================================

Experimento E5.3-3 de BATERIA_ENFOQUE5 (Tema 3: eficiencia de conversion emergente).
Ver PROTOCOLO_E5.3-3_PREREGISTRO.md en esta misma carpeta para la definicion completa,
el criterio de congelamiento (fijado ANTES de correr) y la justificacion T0-T7.

Reutiliza tal cual (solo lectura, sin editar) la dinamica de cs074_rcruz.py:
  - campo_inicial(N, eps, rng): fondo=1 + eps*perturbacion multi-modo, std normalizado.
  - paso_difusion(phi, activo): difusion vectorizada SOLO por aristas vivas.
  - paso_expansion(activo, H, rng): corte Bernoulli de aristas vivas, prob H por paso.
  - medir_D(N, eps, seed): D medido del propio campo (fraccion de contraste que un paso
    de difusion pura borra), usado para mapear r -> H = min(r*D, 1).
  - persistencia(phi, contraste0): corr_local * var_ratio, ya usada en CS074 como medida
    de estructura sobreviviente; aqui se reinterpreta como eficiencia(t) = E_ligada(t)/E_total.

Definicion de eficiencia(t) (ver protocolo, seccion 1):
    E_total       = var(phi_0)                      (presupuesto estructural fijado por eps)
    E_ligada(t)   = corr(phi_t, roll(phi_t,1)) * var(phi_t)   (estructura coherente restante)
    eficiencia(t) = E_ligada(t) / E_total = persistencia(phi_t, contraste0)

NULL por checkpoint: barajar espacialmente phi_t (misma varianza, sin orden) antes de medir.

Congelamiento (T3, fijado antes de correr): la curva "congela" en el checkpoint k* si para
TODO k >= k*, |e_k - e_{k-1}| / max(e_{k-1}, 1e-6) < UMBRAL_CONGELA (2%), sostenido hasta el
final del rango. Si nunca se cumple, se reporta "no congela en el rango".
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent  # Cosmogenesis/
sys.path.insert(0, str(BASE))

from cs074_rcruz import campo_inicial, paso_difusion, paso_expansion, medir_D  # noqa: E402

# ------------------------------------------------------------------
# Parametros pre-registrados (PROTOCOLO_E5.3-3_PREREGISTRO.md, seccion 3)
# ------------------------------------------------------------------
N = 200
CHECKPOINTS = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
EPS_LIST = [0.0, 1e-6, 1e-4, 1e-2, 0.1, 1.0]
R_LIST = [0.0, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]
SEMILLAS = 12
UMBRAL_CONGELA = 0.02  # 2% de cambio relativo entre checkpoints consecutivos


def persistencia_de(phi, contraste0):
    """Identica a cs074_rcruz.persistencia — reimplementada aqui para poder
    aplicarla tanto al campo real como al barajado (NULL) en el mismo checkpoint
    sin duplicar estado. Logica verbatim de cs074_rcruz.py."""
    if contraste0 <= 0 or phi.std() <= 1e-12:
        return 0.0, 0.0
    c = np.corrcoef(phi, np.roll(phi, 1))[0, 1]
    if not np.isfinite(c):
        c = 0.0
    c = max(0.0, float(c))
    v = float(phi.var() / (contraste0 ** 2))
    return float(c * v), v


def trayectoria(N, eps, H, seed, checkpoints):
    """Corre una trayectoria continua hasta max(checkpoints) pasos, midiendo
    eficiencia(t) real y NULL (barajado espacial) en cada checkpoint, mas el
    segundo observable crudo var_ratio(t). No reinicia el campo entre checkpoints."""
    rng = np.random.default_rng(seed)
    phi, _ = campo_inicial(N, eps, rng)
    activo = np.ones(N, dtype=bool)
    contraste0 = float(phi.std())

    filas = []
    pasos_hechos = 0
    max_pasos = max(checkpoints)
    cp_idx = 0
    cp_set = sorted(checkpoints)

    if contraste0 <= 0:
        # eps=0: campo constante, sin estructura posible en ningun t. Se documenta
        # analiticamente (ahorra 1e5 pasos triviales x 8 r x 12 semillas = 96 corridas).
        for cp in cp_set:
            filas.append({"paso": cp, "eficiencia_real": 0.0, "eficiencia_null": 0.0,
                          "var_ratio": 0.0, "analitico_eps0": True})
        return filas, contraste0

    while pasos_hechos < max_pasos:
        phi = paso_difusion(phi, activo)
        activo = paso_expansion(activo, H, rng)
        pasos_hechos += 1
        if cp_idx < len(cp_set) and pasos_hechos == cp_set[cp_idx]:
            e_real, v = persistencia_de(phi, contraste0)
            phi_shuf = rng.permutation(phi)
            e_null, _ = persistencia_de(phi_shuf, contraste0)
            # T6: verificar conservacion (var no puede exceder var(phi_0))
            viol_T6 = bool(v > 1.0 + 1e-9)
            filas.append({
                "paso": pasos_hechos,
                "eficiencia_real": e_real,
                "eficiencia_null": e_null,
                "var_ratio": v,
                "analitico_eps0": False,
                "viol_T6": viol_T6,
            })
            cp_idx += 1
    return filas, contraste0


def detectar_congelamiento(curva_pasos, curva_eficiencia, umbral=UMBRAL_CONGELA):
    """curva_pasos, curva_eficiencia: listas paralelas ordenadas por paso creciente.
    Devuelve (congela: bool, paso_congela: int|None)."""
    n = len(curva_eficiencia)
    for k_star in range(1, n):
        ok = True
        for k in range(k_star, n):
            prev = curva_eficiencia[k - 1]
            cur = curva_eficiencia[k]
            rel = abs(cur - prev) / max(prev, 1e-6)
            if rel >= umbral:
                ok = False
                break
        if ok:
            return True, curva_pasos[k_star]
    return False, None


def main():
    t0 = time.time()
    resultados = []  # una fila por (eps, r, semilla) con su curva completa
    meta_D = {}

    for eps in EPS_LIST:
        if eps <= 0:
            meta_D[eps] = 0.0
            continue
        Ds = [medir_D(N, eps, s) for s in range(SEMILLAS)]
        meta_D[eps] = float(np.mean(Ds))

    total_combos = len(EPS_LIST) * len(R_LIST) * SEMILLAS
    hecho = 0
    for eps in EPS_LIST:
        D = meta_D[eps]
        for r_tgt in R_LIST:
            if D > 0:
                H = float(min(r_tgt * D, 1.0))
            else:
                H = 0.0 if r_tgt == 0 else 1.0
            for s in range(SEMILLAS):
                seed = 5000 + s
                filas, contraste0 = trayectoria(N, eps, H, seed, CHECKPOINTS)
                pasos = [f["paso"] for f in filas]
                efic_real = [f["eficiencia_real"] for f in filas]
                efic_null = [f["eficiencia_null"] for f in filas]
                var_ratio = [f["var_ratio"] for f in filas]
                congela, paso_congela = detectar_congelamiento(pasos, efic_real)
                congela_null, paso_congela_null = detectar_congelamiento(pasos, efic_null)
                viol_T6 = any(f.get("viol_T6", False) for f in filas)
                resultados.append({
                    "eps": eps,
                    "r_target": r_tgt,
                    "H": H,
                    "D": D,
                    "seed": seed,
                    "contraste0": contraste0,
                    "pasos": pasos,
                    "eficiencia_real": efic_real,
                    "eficiencia_null": efic_null,
                    "var_ratio": var_ratio,
                    "congela": congela,
                    "paso_congela": paso_congela,
                    "congela_null": congela_null,
                    "paso_congela_null": paso_congela_null,
                    "viol_T6": viol_T6,
                    "analitico_eps0": filas[0]["analitico_eps0"] if filas else False,
                })
                hecho += 1
        print(f"[progreso] eps={eps} listo ({hecho}/{total_combos})", file=sys.stderr, flush=True)

    elapsed = time.time() - t0

    # ---- Agregacion por (eps, r): mediana/dispersion de paso_congela entre semillas ----
    agregados = []
    for eps in EPS_LIST:
        for r_tgt in R_LIST:
            filas = [f for f in resultados if f["eps"] == eps and f["r_target"] == r_tgt]
            n_congela = sum(1 for f in filas if f["congela"])
            pasos_congela = [f["paso_congela"] for f in filas if f["congela"]]
            # curva promedio de eficiencia real y null entre semillas (mismo eje de pasos)
            pasos_eje = filas[0]["pasos"]
            efic_real_mat = np.array([f["eficiencia_real"] for f in filas])
            efic_null_mat = np.array([f["eficiencia_null"] for f in filas])
            var_ratio_mat = np.array([f["var_ratio"] for f in filas])
            agregados.append({
                "eps": eps,
                "r_target": r_tgt,
                "H": filas[0]["H"],
                "D": filas[0]["D"],
                "n_semillas": len(filas),
                "n_congela": n_congela,
                "frac_congela": n_congela / len(filas) if filas else 0.0,
                "paso_congela_mediana": float(np.median(pasos_congela)) if pasos_congela else None,
                "paso_congela_min": min(pasos_congela) if pasos_congela else None,
                "paso_congela_max": max(pasos_congela) if pasos_congela else None,
                "pasos_eje": pasos_eje,
                "eficiencia_real_media": efic_real_mat.mean(axis=0).tolist(),
                "eficiencia_real_std": efic_real_mat.std(axis=0).tolist(),
                "eficiencia_null_media": efic_null_mat.mean(axis=0).tolist(),
                "eficiencia_null_std": efic_null_mat.std(axis=0).tolist(),
                "var_ratio_media": var_ratio_mat.mean(axis=0).tolist(),
                "z_final_vs_null": float(
                    (efic_real_mat[:, -1].mean() - efic_null_mat[:, -1].mean())
                    / max(np.sqrt((efic_real_mat[:, -1].var() + efic_null_mat[:, -1].var()) / 2.0), 1e-9)
                ),
                "viol_T6_alguna": any(f["viol_T6"] for f in filas),
            })

    salida = {
        "experimento": "E5_3_3_estabilidad_temporal",
        "N": N,
        "checkpoints": CHECKPOINTS,
        "eps_list": EPS_LIST,
        "r_list": R_LIST,
        "semillas": SEMILLAS,
        "umbral_congela": UMBRAL_CONGELA,
        "meta_D": meta_D,
        "elapsed_s": elapsed,
        "agregados": agregados,
        "crudo_por_semilla": resultados,
    }

    out_json = HERE / "E5_3_3_resultado.json"
    out_json.write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[archivo] {out_json}", file=sys.stderr)
    print(f"[elapsed] {elapsed:.1f}s", file=sys.stderr)
    n_viol = sum(1 for a in agregados if a["viol_T6_alguna"])
    print(f"[T6] combinaciones con violacion de conservacion: {n_viol}/{len(agregados)}", file=sys.stderr)


if __name__ == "__main__":
    main()
