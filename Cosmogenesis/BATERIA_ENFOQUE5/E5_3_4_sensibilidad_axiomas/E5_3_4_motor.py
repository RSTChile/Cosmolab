#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E5.3-4 -- Sensibilidad de la eficiencia a los dos axiomas (E1 on/off, E2 on/off)
=================================================================================

Ver pre-registro: PROTOCOLO_E5.3-4_PREREGISTRO.md (definicion EXACTA de que significa
"apagar" E1 y E2 en esta implementacion, escrita ANTES de este motor).

Reusa el motor fisico de cs074_rcruz.py SIN EDITARLO. Las funciones de paso
(campo_inicial, paso_difusion, paso_expansion) se REIMPLEMENTAN aqui en version
BATCHED (vectorizadas simultaneamente sobre las 12 semillas, eje 0 = semilla, eje 1 =
sitio del anillo) por razones de rendimiento: la primera version de este motor (llamando
directamente a las funciones 1D de cs074_rcruz.py dentro de un loop Python por semilla)
resulto demasiado lenta para la grilla completa (proyectado >3h). Las versiones batched
son matematicamente IDENTICAS fila-por-fila a las funciones originales de
cs074_rcruz.py (verificado por equivalencia numerica en
`test_equivalencia_batch.py`-inline mas abajo, funcion `_verificar_equivalencia()`,
que se corre una vez al importar este modulo). cs074_rcruz.py en si NO se edita ni se
deja de usar: se sigue usando para medir D y calibrar pasos (medir_D, medir_pasos_lavado),
que no son el cuello de botella.

Definicion de eficiencia (heredada de E5.3-3, mismo Tema 3, la unica definicion ya
congelada en disco para esta bateria en el momento de escribir esto -- E5.3-1 no existe):

    E_total          = contraste0_sq = var(phi_0)
    eficiencia_final = max(0, corr(phi_final, roll(phi_final,1))) * var(phi_final) / E_total

Axiomas (definicion EXACTA en el protocolo, seccion 1):
  E2 ON  -> tras cada paso_expansion, se inyecta delta_iny = delta_frac_cortada_este_paso *
            E_lat(t) como amplitud extra de la desviacion de phi (o en nodos recien
            aislados si el campo esta plano). E_lat0 = mean(phi_0)**2 (medido).
  E1 ON  -> el reservorio E_lat se PAGA (E_lat -= delta_iny) y se IMPONE el tope
            var(phi_t) <= contraste0_sq cada paso (se reescala si se excede).
  E1 OFF -> el reservorio E_lat NO se paga (delta_iny gratis, se repite sin agotarse) y el
            tope NO se impone (var puede exceder contraste0_sq; se mide el exceso, no se
            oculta).
  E2 OFF -> nunca se inyecta nada; la dinamica es identica bit-a-bit (por fila) a
            cs074_rcruz.py.

NULL: barajado espacial de phi_final (igual a cs074_rcruz.py / E5.3-3).

No se edita cs074_rcruz.py. No topologia (mismo anillo N nodos del motor base). Sin commits.
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

from cs074_rcruz import (  # noqa: E402  (import tras sys.path fix, motor NO editado)
    campo_inicial,
    paso_difusion,
    paso_expansion,
    medir_D,
    medir_pasos_lavado,
    P_LAVADO,
)

# ---------------------------------------------------------------------------
# Pre-registro (congelado ANTES de correr, ver .md hermano) -- NO tocar tras revisar
# ---------------------------------------------------------------------------
N = 200
EPS_LIST = [0.0, 1e-12, 1e-9, 1e-6, 1e-4, 1e-2, 1e-1, 1.0]
R_LIST = [0.0, 1e-3, 1e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
N_SEMILLAS = 12
SEED_BASE = 7000
EPS_CALIBRACION = 1e-2  # mismo valor representativo que uso E5.2-2
AXIOMAS = [(True, True), (False, True), (True, False), (False, False)]  # (E1, E2)


# ---------------------------------------------------------------------------
# Version BATCHED (S semillas simultaneas) de las 3 funciones de paso de cs074_rcruz.py
# ---------------------------------------------------------------------------
def campo_inicial_batch(N, eps, rng, S):
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    fondo = np.ones((S, N), dtype=np.float64)
    if eps <= 0.0:
        return fondo.copy()
    pert = np.zeros((S, N), dtype=np.float64)
    for m in range(1, 6):
        fase = rng.uniform(0, 2 * np.pi, size=S)[:, None]
        pert += np.sin(2 * np.pi * m * x[None, :] + fase) / m
    pert -= pert.mean(axis=1, keepdims=True)
    std = pert.std(axis=1, keepdims=True)
    std = np.where(std > 0, std, 1.0)
    pert = pert / std
    return fondo + eps * pert


def paso_difusion_batch(phi, activo):
    left = np.roll(phi, 1, axis=1)
    right = np.roll(phi, -1, axis=1)
    e_left = np.roll(activo, 1, axis=1)
    e_right = activo
    n_nb = e_left.astype(np.float64) + e_right.astype(np.float64)
    s = e_left * left + e_right * right
    media = np.divide(s, n_nb, out=phi.copy(), where=n_nb > 0)
    nuevo = phi + 0.5 * (media - phi)
    return np.where(n_nb > 0, nuevo, phi)


def paso_expansion_batch(activo, H, rng):
    if H <= 0.0:
        return activo
    activo = activo.copy()
    if H >= 1.0:
        activo[:] = False
        return activo
    u = rng.random(activo.shape)
    cortar = activo & (u < H)
    activo[cortar] = False
    return activo


def _verificar_equivalencia():
    """Chequeo de arranque: la version batched (S=1) debe coincidir EXACTO con las
    funciones 1D originales de cs074_rcruz.py para difusion (determinista) y para
    campo_inicial (misma formula por fila). paso_expansion es estocastica (Bernoulli) asi
    que no se compara valor-a-valor, se deja documentado que el algoritmo es identico."""
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    phi_orig, _ = campo_inicial(50, 0.3, rng1)
    phi_batch = campo_inicial_batch(50, 0.3, rng2, S=1)[0]
    assert np.allclose(phi_orig, phi_batch, atol=1e-12), "campo_inicial_batch no coincide con campo_inicial"

    activo = np.random.default_rng(1).random(50) > 0.3
    phi_d_orig = paso_difusion(phi_orig.copy(), activo)
    phi_d_batch = paso_difusion_batch(phi_orig.copy()[None, :], activo[None, :])[0]
    assert np.allclose(phi_d_orig, phi_d_batch, atol=1e-12), "paso_difusion_batch no coincide con paso_difusion"


_verificar_equivalencia()


def paso_inyeccion_E2_batch(phi, activo_antes, activo_despues, E_lat, rng):
    """Axioma E2 batched. phi,activo: (S,N). E_lat: (S,). Devuelve (phi_nuevo, delta_iny(S,))."""
    S, Nn = phi.shape
    recien_cortadas = activo_antes & (~activo_despues)
    delta_frac = recien_cortadas.mean(axis=1)  # (S,)
    delta_iny = delta_frac * np.maximum(E_lat, 0.0)  # (S,)
    media = phi.mean(axis=1, keepdims=True)
    dev = phi - media
    var_actual = (dev ** 2).mean(axis=1)  # (S,)
    phi_nuevo = phi.copy()

    con_estructura = var_actual > 1e-15
    quiere_inyectar = delta_iny > 0.0

    mask_a = quiere_inyectar & con_estructura
    if mask_a.any():
        factor = np.sqrt(1.0 + delta_iny[mask_a] / var_actual[mask_a])
        phi_nuevo[mask_a] = media[mask_a, 0][:, None] + dev[mask_a] * factor[:, None]

    mask_b = quiere_inyectar & (~con_estructura)
    if mask_b.any():
        e_left = np.roll(activo_despues, 1, axis=1)
        e_right = activo_despues
        aislados = (~e_left) & (~e_right)  # (S,N)
        for i in np.where(mask_b)[0]:
            idx = np.where(aislados[i])[0]
            n_iso = idx.size
            if n_iso <= 0:
                delta_iny[i] = 0.0
                continue
            signos = rng.choice(np.array([-1.0, 1.0]), size=n_iso)
            amp = np.sqrt(delta_iny[i] / n_iso)
            phi_nuevo[i, idx] = media[i, 0] + signos * amp

    return phi_nuevo, delta_iny


def clamp_E1_batch(phi, contraste0_sq, E1_on):
    """contraste0_sq: (S,). Devuelve (phi, exceso(S,)) -- exceso SIEMPRE se mide (T6);
    el reescalado solo se aplica si E1_on."""
    media = phi.mean(axis=1, keepdims=True)
    dev = phi - media
    var_actual = (dev ** 2).mean(axis=1)  # (S,)
    exceso = np.maximum(0.0, var_actual - contraste0_sq)
    if E1_on:
        need = exceso > 0.0
        if need.any():
            factor = np.ones(phi.shape[0])
            factor[need] = np.sqrt(contraste0_sq[need] / np.maximum(var_actual[need], 1e-300))
            phi = media + dev * factor[:, None]
    return phi, exceso


def eficiencia_batch(phi_final, contraste0_sq):
    """corr_local * var / contraste0_sq por fila. Devuelve (eficiencia(S,) con NaN donde
    contraste0_sq<=0, var_final(S,))."""
    media = phi_final.mean(axis=1, keepdims=True)
    var_f = phi_final.var(axis=1)
    left = np.roll(phi_final, 1, axis=1)
    # correlacion de Pearson por fila entre phi_final y su vecino izquierdo
    a = phi_final - media
    b = left - left.mean(axis=1, keepdims=True)
    num = (a * b).mean(axis=1)
    den = np.sqrt((a ** 2).mean(axis=1) * (b ** 2).mean(axis=1))
    corr = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-15)
    corr = np.clip(corr, 0.0, None)
    corr = np.nan_to_num(corr, nan=0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        efic = np.where(contraste0_sq > 1e-15, corr * var_f / np.maximum(contraste0_sq, 1e-300), np.nan)
    return efic, var_f


def campo_inicial_batch_vareps(N, eps_arr, rng, Big):
    """Como campo_inicial_batch pero con un eps DISTINTO por fila (eps_arr shape (Big,)).
    Para eps_arr[i]<=0 el resultado de esa fila es fondo puro (equivalente exacto a
    campo_inicial con eps<=0), solo que aqui se generan las fases de todas maneras (mismo
    algoritmo, no cambia el resultado: eps_arr[i]*pert = 0)."""
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    fondo = np.ones((Big, N), dtype=np.float64)
    pert = np.zeros((Big, N), dtype=np.float64)
    for m in range(1, 6):
        fase = rng.uniform(0, 2 * np.pi, size=Big)[:, None]
        pert += np.sin(2 * np.pi * m * x[None, :] + fase) / m
    pert -= pert.mean(axis=1, keepdims=True)
    std = pert.std(axis=1, keepdims=True)
    std = np.where(std > 0, std, 1.0)
    pert = pert / std
    return fondo + np.maximum(eps_arr, 0.0)[:, None] * pert


def paso_expansion_batch_vareps(activo, H_arr, rng):
    """Como paso_expansion_batch pero con un H distinto POR FILA (H_arr shape (Big,)).
    Filas con H<=0: sin cambio. Filas con H>=1: cortan TODAS sus aristas vivas (deterministico,
    igual que paso_expansion original para H>=1). Filas 0<H<1: Bernoulli(H) por arista."""
    activo = activo.copy()
    u = rng.random(activo.shape)
    Hb = H_arr[:, None]
    cortar_bernoulli = activo & (u < Hb) & (Hb > 0.0) & (Hb < 1.0)
    activo[cortar_bernoulli] = False
    full_rows = H_arr >= 1.0
    if full_rows.any():
        activo[full_rows, :] = False
    return activo


def correr_todo_batch(N, eps_arr, H_arr, pasos, seed, E1_on, E2_on, Big):
    """Evoluciona TODAS las filas (Big = n_eps*n_r*n_semillas de UN axioma) EN PARALELO,
    un solo loop Python de `pasos` iteraciones para toda la grilla. Devuelve dict de arrays
    (Big,) con lo necesario para eficiencia + diagnostico de conservacion."""
    rng = np.random.default_rng(seed)
    phi0 = campo_inicial_batch_vareps(N, eps_arr, rng, Big)
    contraste0_sq = phi0.var(axis=1)  # (Big,)
    E_lat = (phi0.mean(axis=1)) ** 2  # (Big,) -- E_lat0 medido del propio campo
    phi = phi0.copy()
    activo = np.ones((Big, N), dtype=bool)

    exceso_max = np.zeros(Big)
    exceso_suma = np.zeros(Big)
    exceso_min = np.full(Big, np.inf)
    delta_iny_total = np.zeros(Big)

    for _ in range(pasos):
        phi = paso_difusion_batch(phi, activo)
        activo_antes = activo
        activo = paso_expansion_batch_vareps(activo, H_arr, rng)
        if E2_on:
            phi_iny, delta_iny = paso_inyeccion_E2_batch(phi, activo_antes, activo, E_lat, rng)
            phi = phi_iny
            delta_iny_total += delta_iny
            if E1_on:
                E_lat = np.maximum(0.0, E_lat - delta_iny)
            # si E1 off: E_lat no se paga (protocolo 1.2 punto 5)
        phi, exceso = clamp_E1_batch(phi, contraste0_sq, E1_on)
        exceso_max = np.maximum(exceso_max, exceso)
        exceso_min = np.minimum(exceso_min, exceso)
        exceso_suma += exceso

    exceso_media = exceso_suma / max(pasos, 1)

    return {
        "phi_final": phi,
        "contraste0_sq": contraste0_sq,
        "E_lat0": (phi0.mean(axis=1)) ** 2,
        "E_lat_final": E_lat,
        "delta_iny_total": delta_iny_total,
        "exceso_min": exceso_min,
        "exceso_media": exceso_media,
        "exceso_max": exceso_max,
    }


def main():
    t0 = time.time()

    # --- calibracion de pasos (medida, no impuesta; UNA sola vez, fija para las 4 variantes) ---
    cal = medir_pasos_lavado(N, EPS_CALIBRACION, N_SEMILLAS, P_thr=P_LAVADO)
    pasos = cal["pasos"]
    print(
        f"[calibracion] N={N} eps_cal={EPS_CALIBRACION} mediana_lavado={cal['mediana']} "
        f"pasos={pasos} lavo_todas={cal['lavo_todas']}",
        file=sys.stderr, flush=True,
    )

    # --- D medido por eps (para H = min(r*D,1)) ---
    D_por_eps = {}
    for eps in EPS_LIST:
        if eps <= 0:
            D_por_eps[eps] = 0.0
            continue
        Ds = [medir_D(N, eps, SEED_BASE + s) for s in range(N_SEMILLAS)]
        D_por_eps[eps] = float(np.mean(Ds))
    print(f"[D_medido] {D_por_eps}", file=sys.stderr, flush=True)

    def resumen(arr):
        finite = arr[np.isfinite(arr)]
        n_none = int(arr.size - finite.size)
        if finite.size == 0:
            return {"media": None, "mediana": None, "std": None, "n_validos": 0, "n_none": n_none}
        return {
            "media": float(finite.mean()),
            "mediana": float(np.median(finite)),
            "std": float(finite.std()),
            "n_validos": int(finite.size),
            "n_none": n_none,
        }

    # --- construir la grilla (eps, r) una vez: cada fila del batch grande = una celda (eps,r,semilla) ---
    celdas = []  # (eps_idx, eps, r_idx, r_tgt, H, D, r_eff) en orden -- mismo orden para las 4 variantes
    for eps_idx, eps in enumerate(EPS_LIST):
        D = D_por_eps[eps]
        for r_idx, r_tgt in enumerate(R_LIST):
            if D > 0:
                H = float(min(r_tgt * D, 1.0))
                r_eff = H / D
            else:
                H = 0.0 if r_tgt == 0 else 1.0
                r_eff = 0.0 if r_tgt == 0 else float("inf")
            celdas.append((eps_idx, eps, r_idx, r_tgt, H, D, r_eff))

    n_celdas_grid = len(celdas)  # n_eps * n_r
    Big = n_celdas_grid * N_SEMILLAS
    eps_arr = np.repeat(np.array([c[1] for c in celdas], dtype=np.float64), N_SEMILLAS)
    H_arr = np.repeat(np.array([c[4] for c in celdas], dtype=np.float64), N_SEMILLAS)

    filas = []
    for axioma_idx, (E1_on, E2_on) in enumerate(AXIOMAS):
        # MISMA semilla para las 4 variantes de axioma (comparacion controlada): phi_0 y el
        # flujo de numeros aleatorios consumido son identicos entre variantes hasta el punto
        # en que el axioma en si causa una diferencia (clamp de E1 disparando, o consumo
        # extra de rng por la inyeccion de E2) -- si se usara una semilla distinta por
        # axioma, cualquier diferencia observada estaria confundida con una realizacion
        # aleatoria distinta, no con el axioma (bug detectado y corregido ANTES de la
        # corrida de produccion, ver protocolo seccion 6bis).
        seed_axioma = SEED_BASE
        t_ax0 = time.time()
        out = correr_todo_batch(N, eps_arr, H_arr, pasos, seed_axioma, E1_on, E2_on, Big)
        efic_real, var_f = eficiencia_batch(out["phi_final"], out["contraste0_sq"])

        # NULL: barajado espacial por fila (vectorizado con permutacion independiente por fila)
        rng_n = np.random.default_rng(seed_axioma + 900_000_000)
        idx_perm = np.argsort(rng_n.random(out["phi_final"].shape), axis=1)
        phi_null = np.take_along_axis(out["phi_final"], idx_perm, axis=1)
        efic_null, _ = eficiencia_batch(phi_null, out["contraste0_sq"])

        print(
            f"[axioma {axioma_idx+1}/4] E1_on={E1_on} E2_on={E2_on} Big={Big} "
            f"pasos={pasos} tiempo={time.time()-t_ax0:.1f}s",
            file=sys.stderr, flush=True,
        )

        for eps_idx, eps, r_idx, r_tgt, H, D, r_eff in celdas:
            fila_ini = (eps_idx * len(R_LIST) + r_idx) * N_SEMILLAS
            sl = slice(fila_ini, fila_ini + N_SEMILLAS)
            filas.append({
                "E1_on": E1_on,
                "E2_on": E2_on,
                "eps": eps,
                "r_target": r_tgt,
                "H": round(H, 10),
                "D": round(D, 10),
                "r_efectivo": r_eff,
                "pasos": pasos,
                "n_semillas": N_SEMILLAS,
                "contraste0_sq_degenerado": bool(eps <= 0),
                "eficiencia_real": resumen(efic_real[sl]),
                "eficiencia_null": resumen(efic_null[sl]),
                "var_phi_final": resumen(var_f[sl]),
                "exceso_max": resumen(out["exceso_max"][sl]),
                "exceso_media": resumen(out["exceso_media"][sl]),
                "delta_iny_total": resumen(out["delta_iny_total"][sl]),
                "E_lat_final": resumen(out["E_lat_final"][sl]),
                "eficiencias_real_por_semilla": [
                    (round(float(v), 8) if np.isfinite(v) else None) for v in efic_real[sl]
                ],
            })

    # --- sensibilidad ΔE1 y ΔE2 sobre la grilla (ε,r), a partir de las medias ya calculadas ---
    idx = {(f["E1_on"], f["E2_on"], f["eps"], f["r_target"]): f for f in filas}
    sensibilidad = []
    for eps in EPS_LIST:
        for r_tgt in R_LIST:
            f_oo = idx.get((True, True, eps, r_tgt))
            f_fo = idx.get((False, True, eps, r_tgt))
            f_of = idx.get((True, False, eps, r_tgt))
            f_ff = idx.get((False, False, eps, r_tgt))

            def m(f):
                return f["eficiencia_real"]["media"] if f else None

            def delta(a, b):
                if a is None or b is None:
                    return None
                return float(a - b)

            sensibilidad.append({
                "eps": eps,
                "r_target": r_tgt,
                "eficiencia_E1on_E2on": m(f_oo),
                "eficiencia_E1off_E2on": m(f_fo),
                "eficiencia_E1on_E2off": m(f_of),
                "eficiencia_E1off_E2off": m(f_ff),
                "delta_E1_con_E2on": delta(m(f_oo), m(f_fo)),
                "delta_E1_con_E2off": delta(m(f_of), m(f_ff)),
                "delta_E2_con_E1on": delta(m(f_oo), m(f_of)),
                "delta_E2_con_E1off": delta(m(f_fo), m(f_ff)),
            })

    max_diff_E1_sin_E2 = 0.0
    for row in sensibilidad:
        d = row["delta_E1_con_E2off"]
        if d is not None:
            max_diff_E1_sin_E2 = max(max_diff_E1_sin_E2, abs(d))

    result = {
        "experimento": "E5.3-4 sensibilidad de la eficiencia a E1/E2",
        "definicion_eficiencia": (
            "eficiencia_final = max(0, corr(phi_final, roll(phi_final,1))) * "
            "var(phi_final) / var(phi_0)  [heredada de E5.3-3, unica definicion en disco "
            "para Tema 3 al momento de escribir; E5.3-1 no existe en disco]"
        ),
        "N": N,
        "eps_list": EPS_LIST,
        "r_list": R_LIST,
        "n_semillas": N_SEMILLAS,
        "pasos": pasos,
        "calibracion_lavado": cal,
        "D_por_eps": D_por_eps,
        "axiomas_barridos": [{"E1_on": a, "E2_on": b} for a, b in AXIOMAS],
        "filas": filas,
        "sensibilidad_grilla": sensibilidad,
        "chequeo_prediccion_E1_solo_con_E2": {
            "max_abs_delta_E1_con_E2_off": max_diff_E1_sin_E2,
            "prediccion": "deberia ser ~0 (menos de 1e-6): sin canal E2 no hay nada que el tope de E1 recorte",
            "se_cumple": bool(max_diff_E1_sin_E2 < 1e-6),
        },
        "elapsed_s": time.time() - t0,
    }

    out_json = HERE / "E5_3_4_resultado.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[archivo] {out_json}", file=sys.stderr)
    print(f"[elapsed] {result['elapsed_s']:.1f}s", file=sys.stderr)
    print(
        f"[chequeo E1-solo-con-E2] max|delta|={max_diff_E1_sin_E2:.3e} "
        f"se_cumple={result['chequeo_prediccion_E1_solo_con_E2']['se_cumple']}"
    )


if __name__ == "__main__":
    main()
