#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F1_2_motor.py — F1-2: "Persistencia por información mutua espacial (observable independiente)"
================================================================================================

Experimento F1-2 de la BATERÍA DE FUNDAMENTOS (Enfoque 1). Ver especificación
congelada en:
  BATERIA_FUNDAMENTOS/F1_2_info_mutua/PROTOCOLO_F1-2_PREREGISTRO.md

FÍSICA: importada SIN modificar desde `cs074_rcruz.py` (código base, archivo
ajeno, NO se toca) — `campo_inicial()`, `medir_D()`, `medir_pasos_lavado()` se
usan tal cual (mismo método de calibración que usa F1-1, para comparabilidad).

OBSERVABLE: nuevo en este archivo, `informacion_mutua_mitades_batch()` —
información mutua normalizada entre las dos mitades del dominio (ver protocolo,
sección 2). NO es la `persistencia()` de F1-1; es un estimador ortogonal.

GRID: idéntico al pre-registrado por el agente F1-1 (mismo documento madre,
misma sección "F1-1", copiado LITERAL aquí como constantes — NO se importa en
caliente el módulo `F1_1_motor.py` del agente paralelo para no acoplar procesos
ni tocar sus archivos/side-effects; los valores se transcriben y se puede
verificar que coinciden leyendo ambos protocolos).

INGENIERÍA — por qué este motor no es un calco 1:1 del patrón de F1_1_motor.py:
la grilla completa (13 eps × 34 r × 4 N × 12 semillas = 1768 combos × 24
corridas = 42.432 corridas) es, corrida corrida-por-corrida con las funciones
de un solo canal de `cs074_rcruz.py`, COMPUTACIONALMENTE INTRATABLE en esta
sesión (medido empíricamente: para N=1600 una sola corrida de los `pasos`
calibrados tarda ~46s → 42.432 corridas de ese orden = días). La causa NO es la
física (es barata) sino el overhead fijo de Python/NumPy por llamada, repetido
miles de veces sobre arreglos chicos.

Solución: SE VECTORIZA POR LOTES (batching) — se corren TODAS las combinaciones
(eps, r, semilla) de un N dado como un único arreglo (M, N) con M = 13×34×12 =
5304 "canales" independientes evolucionando EN PARALELO bajo la MISMA regla
física (difusión + expansión Bernoulli), en vez de una corrida Python por
combinación. Esto no cambia la física ni el barrido — solo la implementación.
Se valida (`_validar_kernel_batched`) que el kernel batched reproduce EXACTO
(bit a bit) al kernel de un solo canal de `cs074_rcruz.py` cuando M=1 y ambos
reciben la misma secuencia de números aleatorios; el gate de validación puede
FALLAR (T6) y si falla se aborta la corrida sin producir resultados.

Desviación declarada (no afecta la física, sí el generador aleatorio):
`cs074_rcruz.py` crea un `np.random.default_rng(seed)` nuevo por cada llamada a
`corrida()`, de modo que dos corridas con la MISMA semilla pero distinto H usan
los MISMOS números aleatorios crudos (números aleatorios comunes). Aquí, por
tratabilidad, la dinámica estocástica de TODO el lote de un N (los 5304 canales)
comparte un único generador (`rng_dyn`), que entrega un bloque (M,N) de números
por paso — cada canal sigue recibiendo una secuencia i.i.d. válida (la garantía
estadística del muestreo Monte Carlo se preserva), pero NO se reproduce la
técnica de números-comunes entre distintos r de F1-1. Se declara aquí y en el
JSON de salida; no compromete la validez de la comparación agregada (medias +
dispersión entre semillas), solo renuncia a la reducción de varianza por pareo
que sí usa F1-1.

Uso:
  python3 F1_2_motor.py smoke
  python3 F1_2_motor.py produccion --N 200
  python3 F1_2_motor.py produccion --N 400
  python3 F1_2_motor.py produccion --N 800
  python3 F1_2_motor.py produccion --N 1600
  python3 F1_2_motor.py resumen        # combina los 4 JSON de producción + evalúa PASS
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
COSMOGENESIS_DIR = HERE.parent.parent  # .../Cosmogenesis
OUT_DIR = HERE / "resultados"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Importar cs074_rcruz.py SIN MODIFICARLO (por ruta explícita, robusto a cwd) ---
_CS074_PATH = COSMOGENESIS_DIR / "cs074_rcruz.py"
if not _CS074_PATH.exists():
    raise SystemExit(f"[F1-2] ERROR: no se encuentra el código base en {_CS074_PATH}")
_spec = importlib.util.spec_from_file_location("cs074_rcruz_base_f12", str(_CS074_PATH))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)  # type: ignore

SEED_BASE = 1000

# ---------------------------------------------------------------------------
# Grid pre-registrado — IDÉNTICO al de F1-1 (PROTOCOLO_F1-1_PREREGISTRO.md,
# sección 4), transcrito literal aquí (ver nota de ingeniería en el docstring
# de arriba sobre por qué no se importa en caliente el módulo de F1-1).
# ---------------------------------------------------------------------------
def eps_grid_full():
    return [0.0] + [float(v) for v in np.logspace(-12, 0, 12)]


def r_grid_full():
    return [
        0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85,
        0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0, 3.0,
        5.0, 7.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0,
    ]


def eps_grid_smoke():
    return [0.0, 1e-9, 1e-3, 1.0]


def r_grid_smoke():
    return [0.0, 0.3, 0.9, 1.0, 1.1, 2.0, 10.0, 100.0]


N_LIST_FULL = [200, 400, 800, 1600]
SEMILLAS_FULL = 12
SEMILLAS_SMOKE = 4
K_BINS = 8          # metaparámetro congelado (sección 2 del protocolo)
N_BLOQUES = 8        # para el diagnóstico secundario de entropía de bloques


def log(msg: str, log_path: Path | None = None):
    ts = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    line = f"[{ts}] {msg}"
    print(line, file=sys.stderr, flush=True)
    if log_path is not None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# Kernel físico BATCHED — matemáticamente idéntico a paso_difusion/
# paso_expansion de cs074_rcruz.py, vectorizado sobre una dimensión extra de
# "canales" M. Validado contra el original en _validar_kernel_batched().
# ---------------------------------------------------------------------------
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


def paso_expansion_batch(activo, H_arr, rng):
    """H_arr: shape (M,). Bernoulli por arista viva, vectorizado por canal."""
    M, N = activo.shape
    if np.all(H_arr <= 0.0):
        return activo
    activo = activo.copy()
    full_cut = H_arr >= 1.0
    if np.any(full_cut):
        activo[full_cut, :] = False
    mid = (H_arr > 0.0) & (H_arr < 1.0)
    if np.any(mid):
        u = rng.random((M, N))
        cortar = activo & mid[:, None] & (u < H_arr[:, None])
        activo = activo & ~cortar
    return activo


def evolucionar_batch(phi, activo, H_arr, pasos, rng):
    contraste0 = phi.std(axis=1)
    for _ in range(pasos):
        phi = paso_difusion_batch(phi, activo)
        activo = paso_expansion_batch(activo, H_arr, rng)
    return phi, activo, contraste0


def permutar_filas(phi, rng):
    """NULL: permuta cada fila (cada canal) independientemente. Vectorizado."""
    rand_vals = rng.random(phi.shape)
    perm_idx = np.argsort(rand_vals, axis=1)
    return np.take_along_axis(phi, perm_idx, axis=1)


def _validar_kernel_batched(n_steps=40, N=64, seed=42, H=0.03, eps=0.2):
    """Gate T6: el kernel batched (M=1) debe reproducir EXACTO al kernel de un
    solo canal de cs074_rcruz.py bajo la MISMA secuencia de aleatorios. Si esto
    falla, la producción NO debe correr (se aborta)."""
    rng_init = np.random.default_rng(seed)
    phi0, _ = base.campo_inicial(N, eps, rng_init)
    activo0 = np.ones(N, dtype=bool)

    rng_a = np.random.default_rng(seed + 1)
    rng_b = np.random.default_rng(seed + 1)
    phi_a, activo_a = phi0.copy(), activo0.copy()
    phi_b, activo_b = phi0[None, :].copy(), activo0[None, :].copy()

    for i in range(n_steps):
        phi_a = base.paso_difusion(phi_a, activo_a)
        activo_a = base.paso_expansion(activo_a, H, rng_a)
        phi_b = paso_difusion_batch(phi_b, activo_b)
        activo_b = paso_expansion_batch(activo_b, np.array([H]), rng_b)
        if (not np.allclose(phi_a, phi_b[0], atol=0, rtol=0)) or (not np.array_equal(activo_a, activo_b[0])):
            return False, f"diverge en paso {i}"
    return True, f"kernel batched == kernel base en {n_steps} pasos, N={N}, H={H}"


# ---------------------------------------------------------------------------
# Observable F1-2: información mutua espacial entre mitades (sección 2 del
# protocolo). Independiente de persistencia() de F1-1.
# ---------------------------------------------------------------------------
def _nmi_row(a, b, K):
    if a.std() <= 1e-14 or b.std() <= 1e-14:
        return 0.0, 0.0, 0.0
    pooled = np.concatenate([a, b])
    edges = np.unique(np.quantile(pooled, np.linspace(0.0, 1.0, K + 1)))
    if edges.size < 3:
        return 0.0, 0.0, 0.0
    Keff = edges.size - 1
    a_bin = np.clip(np.digitize(a, edges[1:-1]), 0, Keff - 1)
    b_bin = np.clip(np.digitize(b, edges[1:-1]), 0, Keff - 1)
    joint, _, _ = np.histogram2d(
        a_bin, b_bin, bins=Keff, range=[[-0.5, Keff - 0.5], [-0.5, Keff - 0.5]]
    )
    tot = joint.sum()
    if tot <= 0:
        return 0.0, 0.0, 0.0
    p = joint / tot
    pa = p.sum(axis=1)
    pb = p.sum(axis=0)
    outer = np.outer(pa, pb)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where((p > 0) & (outer > 0), p / outer, 1.0)
        logratio = np.where(ratio > 0, np.log2(ratio, out=np.zeros_like(ratio), where=ratio > 0), 0.0)
    mi = float(np.sum(np.where(p > 0, p * logratio, 0.0)))
    mi = max(mi, 0.0)
    Ha = -float(np.sum(pa[pa > 0] * np.log2(pa[pa > 0])))
    Hb = -float(np.sum(pb[pb > 0] * np.log2(pb[pb > 0])))
    denom = Ha + Hb
    nmi = (2.0 * mi / denom) if denom > 0 else 0.0
    nmi = float(np.clip(nmi, 0.0, 1.0))
    return nmi, mi, denom / 2.0


def informacion_mutua_mitades_batch(phi, K=K_BINS):
    M, N = phi.shape
    half = N // 2
    A = phi[:, :half]
    B = phi[:, half:half * 2]
    nmi = np.zeros(M)
    mi_bits = np.zeros(M)
    h_avg = np.zeros(M)
    for i in range(M):
        nmi[i], mi_bits[i], h_avg[i] = _nmi_row(A[i], B[i], K)
    return nmi, mi_bits, h_avg


def entropia_bloques_batch(phi, n_bloques=N_BLOQUES, K=K_BINS):
    M, N = phi.shape
    nb = n_bloques if N % n_bloques == 0 else N
    bloques = phi.reshape(M, nb, N // nb).mean(axis=2)
    out = np.zeros(M)
    for i in range(M):
        b = bloques[i]
        if b.std() <= 1e-14:
            continue
        edges = np.unique(np.quantile(b, np.linspace(0.0, 1.0, K + 1)))
        if edges.size < 3:
            continue
        hist, _ = np.histogram(b, bins=edges)
        p = hist / hist.sum()
        p = p[p > 0]
        H = -np.sum(p * np.log2(p))
        out[i] = H / np.log2(edges.size - 1)
    return out


# ---------------------------------------------------------------------------
# Barrido batched para un N dado
# ---------------------------------------------------------------------------
def barrido_N_batch(N, eps_list, r_list, semillas, log_path, eps_cal=1e-3, semillas_cal=None):
    if semillas_cal is None:
        semillas_cal = max(semillas, 4)
    t0 = time.time()
    cal = base.medir_pasos_lavado(N, eps_cal, semillas_cal)
    pasos = cal["pasos"]
    log(f"[N={N}] calibracion lavado: eps_cal={eps_cal} pasos={pasos} mediana={cal['mediana']} "
        f"lavo_todas={cal['lavo_todas']}", log_path)

    D_por_eps = {}
    for eps in eps_list:
        D_por_eps[eps] = float(np.mean([base.medir_D(N, eps, s) for s in range(semillas)]))
    log(f"[N={N}] D medido por eps: " + ", ".join(f"{e:.2e}:{D_por_eps[e]:.5f}" for e in eps_list), log_path)

    combos = []  # (eps, r_target, D, H, seed_idx)
    for eps in eps_list:
        D = D_por_eps[eps]
        for r_tgt in r_list:
            if D > 0:
                H = float(min(r_tgt * D, 1.0))
            else:
                H = 0.0 if r_tgt == 0 else 1.0
            for s in range(semillas):
                combos.append((eps, r_tgt, D, H, s))
    M = len(combos)
    log(f"[N={N}] total canales (eps×r×semilla) = {M}, pasos calibrados = {pasos}", log_path)

    phi = np.empty((M, N), dtype=np.float64)
    for i, (eps, r_tgt, D, H, s) in enumerate(combos):
        seed = SEED_BASE + s
        rng_i = np.random.default_rng(seed)
        phi_i, _ = base.campo_inicial(N, eps, rng_i)
        phi[i] = phi_i
    activo = np.ones((M, N), dtype=bool)
    H_arr = np.array([c[3] for c in combos], dtype=np.float64)

    t_dyn0 = time.time()
    log(f"[N={N}] iniciando evolución batched ({pasos} pasos, arreglo {M}x{N})...", log_path)
    rng_dyn = np.random.default_rng(20260724 + N)  # ver nota de desviación declarada (docstring)
    phi_final, activo_final, contraste0 = evolucionar_batch(phi, activo, H_arr, pasos, rng_dyn)
    t_dyn1 = time.time()
    log(f"[N={N}] evolución batched terminada en {t_dyn1 - t_dyn0:.1f}s", log_path)

    nmi_real, mi_real, havg_real = informacion_mutua_mitades_batch(phi_final)
    hbloq_real = entropia_bloques_batch(phi_final)
    frac_exp = 1.0 - activo_final.mean(axis=1)

    phi_null = permutar_filas(phi_final, rng_dyn)
    nmi_null, mi_null, havg_null = informacion_mutua_mitades_batch(phi_null)
    log(f"[N={N}] observables (MI real/null) calculados sobre {M} canales", log_path)

    idx_by_er = {}
    for i, (eps, r_tgt, D, H, s) in enumerate(combos):
        idx_by_er.setdefault((eps, r_tgt), []).append(i)

    filas = []
    for (eps, r_tgt), idxs in idx_by_er.items():
        idxs = np.array(idxs)
        Preal = nmi_real[idxs]
        Pnull = nmi_null[idxs]
        D = combos[idxs[0]][2]
        H = combos[idxs[0]][3]
        r_eff = (H / D) if D > 0 else (0.0 if r_tgt == 0 else float("inf"))
        sd = float(np.sqrt((Preal.var() + Pnull.var()) / 2.0))
        sd = max(sd, 1.0 / max(len(Preal), 1))
        z = float((Preal.mean() - Pnull.mean()) / sd)
        filas.append({
            "N": N, "eps": eps, "r_target": r_tgt, "H": H, "D": D, "r_eff": r_eff,
            "pasos": pasos,
            "P_mi_real_mean": float(Preal.mean()), "P_mi_real_std": float(Preal.std()),
            "P_mi_null_mean": float(Pnull.mean()), "P_mi_null_std": float(Pnull.std()),
            "P_mi_real_seeds": [round(float(x), 6) for x in Preal],
            "P_mi_null_seeds": [round(float(x), 6) for x in Pnull],
            "mi_bits_real_mean": float(mi_real[idxs].mean()),
            "h_avg_bits_real_mean": float(havg_real[idxs].mean()),
            "H_bloques_norm_real_mean": float(hbloq_real[idxs].mean()),
            "frac_exp_mean": float(frac_exp[idxs].mean()),
            "z": round(z, 4),
        })
    filas.sort(key=lambda f: (f["eps"], f["r_target"]))
    elapsed = time.time() - t0
    return filas, cal, pasos, elapsed, M


def evaluar_criterio(filas):
    """Evaluación MECÁNICA del criterio de PASS pre-registrado (protocolo,
    sección 7). NO es el veredicto de lectura final ni la comparación con
    F1-1 — eso lo da CS con ambas curvas crudas."""
    rows_eps0 = [f for f in filas if f["eps"] == 0.0]
    frac_eps0_no_distinguible = float(np.mean([f["z"] < 3.0 for f in rows_eps0])) if rows_eps0 else None
    control_eps0_pass = (frac_eps0_no_distinguible is not None) and (frac_eps0_no_distinguible >= 0.95)

    rows_r0 = [f for f in filas if f["r_target"] == 0.0 and f["eps"] > 0]
    mean_diff_r0 = float(np.mean([f["P_mi_real_mean"] - f["P_mi_null_mean"] for f in rows_r0])) if rows_r0 else None
    control_r0_pass = (mean_diff_r0 is not None) and (mean_diff_r0 < 0.15)

    rows_null = [f for f in filas if f["r_target"] >= 10.0 and f["eps"] > 1e-6]
    frac_null_cae = float(np.mean([f["z"] >= 3.0 for f in rows_null])) if rows_null else None
    null_cae_pass = (frac_null_cae is not None) and (frac_null_cae >= 0.5)

    if null_cae_pass and control_r0_pass and control_eps0_pass:
        veredicto = "PASS_mecanico"
    else:
        veredicto = "FAIL_o_NEGATIVO_mecanico"

    return {
        "frac_eps0_no_distinguible_de_null": frac_eps0_no_distinguible,
        "control_eps0_pass": control_eps0_pass,
        "mean_diff_P_mi_real_menos_null_r0_eps_gt0": mean_diff_r0,
        "control_r0_pass": control_r0_pass,
        "frac_null_cae_z_ge_3_en_r_ge_10_eps_gt_1e-6": frac_null_cae,
        "null_cae_pass": null_cae_pass,
        "veredicto_mecanico_PASS_FAIL": veredicto,
        "nota": "Evaluación mecánica del gate interno pre-registrado de F1-2. "
                "El veredicto de comparación con F1-1 (T2, la verificación cruzada "
                "real de esta pareja) lo da CS cruzando ambos JSON crudos; CC no "
                "adjudica.",
    }


def run_validacion(log_path):
    ok, detalle = _validar_kernel_batched()
    log(f"[VALIDACION KERNEL] ok={ok} detalle={detalle}", log_path)
    if not ok:
        raise SystemExit(f"[F1-2] ABORTADO: el kernel batched NO reproduce al kernel base ({detalle}). "
                          f"No se corre producción con un motor no validado (T6).")
    return {"ok": ok, "detalle": detalle}


def run_smoke():
    log_path = OUT_DIR / "F1_2_log_ejecucion.txt"
    log("=== F1-2 SMOKE: inicio ===", log_path)
    t0 = time.time()
    val = run_validacion(log_path)
    N = 100
    eps_list = eps_grid_smoke()
    r_list = r_grid_smoke()
    semillas = SEMILLAS_SMOKE
    filas, cal, pasos, elapsed, M = barrido_N_batch(N, eps_list, r_list, semillas, log_path,
                                                      eps_cal=1e-3, semillas_cal=semillas)
    criterio = evaluar_criterio(filas)
    result = {
        "experimento": "F1-2", "modo": "smoke", "N": N,
        "eps_list": eps_list, "r_list": r_list, "semillas": semillas,
        "K_bins": K_BINS, "n_bloques": N_BLOQUES,
        "pasos": pasos, "calibracion_lavado": cal,
        "validacion_kernel_batched": val,
        "filas": filas, "criterio_pass_mecanico": criterio,
        "n_canales_batch": M,
        "elapsed_s": elapsed,
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
    }
    out = OUT_DIR / "F1_2_smoke_resultado.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"=== F1-2 SMOKE: fin. elapsed={elapsed:.1f}s -> {out} ===", log_path)
    log(f"[SMOKE] criterio: {criterio}", log_path)
    log(f"=== F1-2 SMOKE total wall time: {time.time() - t0:.1f}s ===", log_path)


def run_produccion(N):
    if N not in N_LIST_FULL:
        raise SystemExit(f"N={N} no está en el grid pre-registrado {N_LIST_FULL}")
    log_path = OUT_DIR / "F1_2_log_ejecucion.txt"
    log(f"=== F1-2 PRODUCCION N={N}: inicio ===", log_path)
    val = run_validacion(log_path)
    eps_list = eps_grid_full()
    r_list = r_grid_full()
    semillas = SEMILLAS_FULL
    filas, cal, pasos, elapsed, M = barrido_N_batch(N, eps_list, r_list, semillas, log_path,
                                                      eps_cal=1e-3, semillas_cal=semillas)
    criterio = evaluar_criterio(filas)
    result = {
        "experimento": "F1-2", "modo": "produccion", "N": N,
        "eps_list": eps_list, "r_list": r_list, "semillas": semillas,
        "seed_base": SEED_BASE, "K_bins": K_BINS, "n_bloques": N_BLOQUES,
        "pasos": pasos, "calibracion_lavado": cal,
        "validacion_kernel_batched": val,
        "n_filas": len(filas), "n_canales_batch": M,
        "filas": filas, "criterio_pass_mecanico": criterio,
        "elapsed_s": elapsed,
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
    }
    out = OUT_DIR / f"F1_2_produccion_N{N}_resultado.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"=== F1-2 PRODUCCION N={N}: fin. elapsed={elapsed:.1f}s ({elapsed / 60:.1f}min) -> {out} ===",
        log_path)
    log(f"[N={N}] criterio: {criterio}", log_path)


def run_resumen():
    log_path = OUT_DIR / "F1_2_log_ejecucion.txt"
    log("=== F1-2 RESUMEN: combinando producción de los 4 N ===", log_path)
    todas_filas = []
    por_N = {}
    faltantes = []
    for N in N_LIST_FULL:
        p = OUT_DIR / f"F1_2_produccion_N{N}_resultado.json"
        if not p.exists():
            faltantes.append(N)
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        todas_filas.extend(data["filas"])
        por_N[N] = {
            "pasos": data["pasos"],
            "elapsed_s": data["elapsed_s"],
            "criterio_pass_mecanico": data["criterio_pass_mecanico"],
            "calibracion_lavado": data["calibracion_lavado"],
            "n_canales_batch": data.get("n_canales_batch"),
        }
    if faltantes:
        log(f"[RESUMEN] AVISO: faltan resultados de N={faltantes}, resumen parcial", log_path)

    criterio_global = evaluar_criterio(todas_filas) if todas_filas else None

    r_estrella_por_N = {}
    for N in por_N:
        filas_N = [f for f in todas_filas if f["N"] == N and abs(f["eps"] - 1e-3) < 1e-15]
        filas_N.sort(key=lambda f: f["r_target"])
        r_star = None
        for f in filas_N:
            if f["z"] >= 3.0:
                r_star = f["r_target"]
                break
        r_estrella_por_N[N] = r_star

    resumen = {
        "experimento": "F1-2", "modo": "resumen",
        "N_incluidos": list(por_N.keys()), "N_faltantes": faltantes,
        "n_filas_totales": len(todas_filas), "por_N": por_N,
        "r_estrella_aprox_por_N_en_eps_1e-3": r_estrella_por_N,
        "criterio_pass_mecanico_global": criterio_global,
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
    }
    out = OUT_DIR / "F1_2_produccion_resumen.json"
    out.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"=== F1-2 RESUMEN: fin -> {out} ===", log_path)
    log(f"[RESUMEN] criterio global: {criterio_global}", log_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modo", choices=["smoke", "produccion", "resumen", "validar"])
    ap.add_argument("--N", type=int, default=None)
    args = ap.parse_args()

    if args.modo == "validar":
        log_path = OUT_DIR / "F1_2_log_ejecucion.txt"
        print(run_validacion(log_path))
    elif args.modo == "smoke":
        run_smoke()
    elif args.modo == "produccion":
        if args.N is None:
            for N in N_LIST_FULL:
                run_produccion(N)
        else:
            run_produccion(args.N)
    elif args.modo == "resumen":
        run_resumen()


if __name__ == "__main__":
    main()
