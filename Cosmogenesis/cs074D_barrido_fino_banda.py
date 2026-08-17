#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs074D_barrido_fino_banda.py — ¿La estructura vive en una banda estrecha no azarosa?
=========================================================================================

Quién soy / qué hago (código autodescriptivo):
  Implementa PROTOCOLO_cs074D_barrido_fino_banda_PREREGISTRO.md (leer primero). Llama a
  `correr_holistico_energia()` de cs074_energia_holistica.py -- NO SE MODIFICA el motor,
  solo se lo barre con un muestreo Latin Hypercube de 6 variables continuas (mucho más
  amplio que cualquier barrido anterior: incluye población de partículas, tasa de
  expansión y fracciones antiquark/electrón, nunca barridas juntas con ε y la reserva).

Si alguna vez el motor no soporta algo que este diseño pide, el script debe fallar fuerte
(assert) y detenerse -- no se improvisa una solución dentro de cs074_energia_holistica.py.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import qmc
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cs074_energia_holistica import correr_holistico_energia  # noqa: E402

OUT = HERE / "resultados_cs074D_barrido_fino"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Espacio de muestreo (protocolo §2) -- 6 dimensiones, orden fijo
# ---------------------------------------------------------------------------
DIMS = ["amp_rugosidad", "tasa_expansion", "E_reserva", "nq", "naq_frac", "ne_frac"]
RANGOS = {
    "amp_rugosidad": (1e-6, 10.0, "log"),
    "tasa_expansion": (0.001, 0.2, "log"),
    "E_reserva": (0.001, 1000.0, "log"),
    "nq": (150, 600, "lin"),
    "naq_frac": (0.5, 0.95, "lin"),
    "ne_frac": (0.15, 0.6, "lin"),
}
N_PASOS_ESTRUCTURA = 60
NPOS_RATIO = 0.7  # npos = NPOS_RATIO * ne, misma razon default 100:70 de cs074/A/B
SEMILLAS_D = list(range(12))
SEED_NULL_OFFSET = 90_000
Z_THR = 2.0
K_VECINOS = 10
N_BOOT_CONECTIVIDAD = 1000


def _transformar(muestra_unitaria):
    """muestra_unitaria: array (N,6) en [0,1]^6 (salida cruda de LHS) -> dict de arrays
    con los valores físicos reales, en la escala (log o lineal) que declara RANGOS."""
    out = {}
    for i, dim in enumerate(DIMS):
        lo, hi, escala = RANGOS[dim]
        u = muestra_unitaria[:, i]
        if escala == "log":
            out[dim] = np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))
        else:
            out[dim] = lo + u * (hi - lo)
    return out


def generar_configuraciones(n, seed=2026):
    """LHS de 6D (protocolo §2), devuelve lista de dicts con los parámetros ya derivados
    (nq/naq/ne/npos enteros) listos para pasar a correr_holistico_energia()."""
    sampler = qmc.LatinHypercube(d=len(DIMS), seed=seed)
    u = sampler.random(n=n)
    vals = _transformar(u)
    configs = []
    for i in range(n):
        nq = int(round(vals["nq"][i]))
        naq = int(round(nq * vals["naq_frac"][i]))
        ne = int(round(nq * vals["ne_frac"][i]))
        npos = int(round(NPOS_RATIO * ne))
        configs.append(dict(
            amp_rugosidad=float(vals["amp_rugosidad"][i]),
            tasa_expansion=float(vals["tasa_expansion"][i]),
            E_reserva=float(vals["E_reserva"][i]),
            nq=nq, naq=naq, ne=ne, npos=npos,
        ))
    return configs


def correr_configuracion(cfg, idx_cfg):
    """Una configuración: 12 semillas x (REAL + NULL). Devuelve resumen agregado."""
    fr, fn = [], []
    nclust_r, fmayor_r = [], []
    ok_reales, ok_nulls = 0, 0
    for s in SEMILLAS_D:
        real = correr_holistico_energia(
            nq=cfg["nq"], naq=cfg["naq"], ne=cfg["ne"], npos=cfg["npos"],
            pasos_basal=150, amp_rugosidad=cfg["amp_rugosidad"],
            tasa_expansion=cfg["tasa_expansion"], E_reserva=cfg["E_reserva"],
            n_pasos_estructura=N_PASOS_ESTRUCTURA,
            seed_layout=12345 + s, guardar_curva=False,
        )
        null = correr_holistico_energia(
            nq=cfg["nq"], naq=cfg["naq"], ne=cfg["ne"], npos=cfg["npos"],
            pasos_basal=150, amp_rugosidad=cfg["amp_rugosidad"],
            tasa_expansion=cfg["tasa_expansion"], E_reserva=cfg["E_reserva"],
            n_pasos_estructura=N_PASOS_ESTRUCTURA,
            seed_layout=12345 + s, seed_dens_null=SEED_NULL_OFFSET + s,
            guardar_curva=False,
        )
        if real.get("ok"):
            fr.append(real["frac_masa_ligada"])
            nclust_r.append(real["n_clusters_finales"])
            fmayor_r.append(real["frac_masa_en_mayor_cluster"])
            ok_reales += 1
        if null.get("ok"):
            fn.append(null["frac_masa_ligada"])
            ok_nulls += 1

    if not fr or not fn:
        return dict(idx=idx_cfg, cfg=cfg, ok=False, ok_reales=ok_reales, ok_nulls=ok_nulls)

    fr_arr, fn_arr = np.array(fr), np.array(fn)
    sd = max(np.sqrt((fr_arr.var() + fn_arr.var()) / 2.0), 1e-9)
    z = float((fr_arr.mean() - fn_arr.mean()) / sd)

    return dict(
        idx=idx_cfg, cfg=cfg, ok=True, ok_reales=ok_reales, ok_nulls=ok_nulls,
        frac_masa_ligada_real_media=float(fr_arr.mean()), frac_masa_ligada_real_std=float(fr_arr.std()),
        frac_masa_ligada_null_media=float(fn_arr.mean()), frac_masa_ligada_null_std=float(fn_arr.std()),
        z=z,
        n_clusters_finales_media=float(np.mean(nclust_r)) if nclust_r else None,
        frac_masa_en_mayor_cluster_media=float(np.mean(fmayor_r)) if fmayor_r else None,
    )


def _normalizar_espacio(configs):
    """Cada eje a [0,1], log-normalizado en los ejes log (protocolo §5) -- para la métrica
    de conectividad espacial."""
    M = np.zeros((len(configs), len(DIMS)))
    for j, dim in enumerate(DIMS):
        lo, hi, escala = RANGOS[dim]
        if dim in ("nq",):
            vals = np.array([c["nq"] for c in configs], dtype=float)
        elif dim == "naq_frac":
            vals = np.array([c["naq"] / c["nq"] for c in configs], dtype=float)
        elif dim == "ne_frac":
            vals = np.array([c["ne"] / c["nq"] for c in configs], dtype=float)
        else:
            vals = np.array([c[dim] for c in configs], dtype=float)
        if escala == "log":
            vals = np.log(vals)
            lo, hi = np.log(lo), np.log(hi)
        M[:, j] = (vals - lo) / (hi - lo)
    return M


def analizar_conectividad(filas_ok, configs_ok, log_fn=print):
    """Protocolo §5: tasa de vecinos-hit observada vs control de etiquetas barajadas."""
    M = _normalizar_espacio(configs_ok)
    n = len(configs_ok)
    z_arr = np.array([f["z"] for f in filas_ok])
    hit = z_arr > Z_THR
    n_hit = int(hit.sum())
    log_fn(f"[D] configuraciones con z>{Z_THR}: {n_hit}/{n} ({100*n_hit/n:.1f}%)")

    if n_hit == 0:
        return dict(n_hit=0, frac_hit=0.0, tasa_vecinos_hit_obs=None,
                     z_conectividad=None, lectura="sin_hits_z2")
    if n_hit / n > 0.5:
        return dict(n_hit=n_hit, frac_hit=n_hit / n, tasa_vecinos_hit_obs=None,
                     z_conectividad=None, lectura="sin_banda_generico")

    tree = cKDTree(M)
    k = min(K_VECINOS + 1, n)  # +1 porque el propio punto es su vecino más cercano (dist 0)
    _, idxs = tree.query(M, k=k)
    idxs_vecinos = idxs[:, 1:]  # excluir el propio punto

    def tasa_vecinos_hit(hit_mask):
        tasas = []
        for i in np.where(hit_mask)[0]:
            vecinos = idxs_vecinos[i]
            tasas.append(float(hit_mask[vecinos].mean()))
        return float(np.mean(tasas)) if tasas else 0.0

    obs = tasa_vecinos_hit(hit)

    rng = np.random.default_rng(7)
    boot = []
    for _ in range(N_BOOT_CONECTIVIDAD):
        hit_shuf = rng.permutation(hit)
        boot.append(tasa_vecinos_hit(hit_shuf))
    boot = np.array(boot)
    z_conn = float((obs - boot.mean()) / boot.std()) if boot.std() > 0 else None

    if z_conn is not None and z_conn > 2.0:
        lectura = "banda_estrecha"
    else:
        lectura = "disperso_sin_patron"

    return dict(n_hit=n_hit, frac_hit=n_hit / n,
                tasa_vecinos_hit_obs=obs, tasa_vecinos_hit_boot_media=float(boot.mean()),
                tasa_vecinos_hit_boot_std=float(boot.std()),
                z_conectividad=z_conn, lectura=lectura)


def correr_experimento_d(n_configs, seed_lhs=2026, log_fn=print):
    t0 = time.time()
    configs = generar_configuraciones(n_configs, seed=seed_lhs)
    log_fn(f"[D] {n_configs} configuraciones LHS generadas, {len(SEMILLAS_D)} semillas c/u, "
           f"REAL+NULL -> {n_configs*len(SEMILLAS_D)*2} corridas totales")

    filas = []
    for i, cfg in enumerate(configs):
        r = correr_configuracion(cfg, i)
        filas.append(r)
        if (i + 1) % max(1, n_configs // 20) == 0 or (i + 1) == n_configs:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_configs - i - 1)
            log_fn(f"[D] config {i+1}/{n_configs} t={elapsed:.0f}s eta={eta:.0f}s")

    elapsed = time.time() - t0
    filas_ok = [f for f in filas if f["ok"]]
    configs_ok = [f["cfg"] for f in filas_ok]
    log_fn(f"[D] TOTAL elapsed={elapsed:.0f}s ok={len(filas_ok)}/{n_configs}")

    conectividad = analizar_conectividad(filas_ok, configs_ok, log_fn=log_fn) if filas_ok else {}

    return dict(filas=filas, elapsed_s=elapsed, n_configs=n_configs,
                dims=DIMS, rangos=RANGOS, semillas=SEMILLAS_D,
                conectividad=conectividad)


def main():
    log_lines = []

    def p(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, file=sys.stderr, flush=True)
        log_lines.append(line)

    modo = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if modo == "smoke":
        p("=== SMOKE TEST: 20 configuraciones ===")
        resultado = correr_experimento_d(20, log_fn=p)
        nombre = "cs074D_result_smoke.json"
    elif modo.startswith("--full"):
        n = int(modo.split("=")[1]) if "=" in modo else 2000
        p(f"=== FULL: {n} configuraciones ===")
        resultado = correr_experimento_d(n, log_fn=p)
        nombre = "cs074D_result_FULL.json"
    else:
        p(f"modo desconocido: {modo}")
        return

    resultado["log"] = log_lines
    p(f"[D] conectividad = {resultado.get('conectividad')}")

    out_json = OUT / nombre
    out_json.write_text(json.dumps(resultado, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    p(f"[archivo] {out_json}")


if __name__ == "__main__":
    main()
