#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs076_direccion_temporal.py — ¿La regla de actualización de CG001 tiene una dirección
temporal T+/T- propia a nivel MICRO, o solo la entropía AGREGADA sube (ya confirmado en
CS009/C-N3)?

Quién soy / qué hago (código autodescriptivo):
  Nodo C-N2.5.6-10 de la Teoría Cosmosemiótica (Fase I-A del roadmap de continuación,
  5-ago-2026). Distinto de C-N3/CS009: aquello ya mostró que el DESORDEN AGREGADO del
  campo entero sólo sube. Esto pregunta algo más fino: ¿la propia regla de actualización,
  mirada en UNA celda individual paso a paso, distingue "adelante" de "atrás" en el
  tiempo, o esa distinción sólo existe cuando se suma todo el campo?

  Reusa cg001_field.py (motor YA verificado, NO se modifica) importando sus funciones
  internas (_inicializar_phi, _paso) — la dinámica que avanza el campo es exactamente la
  misma que corre correr()/demo(). Lo único que agrega este script es la EXTRACCIÓN de la
  trayectoria completa (no resumida cada log_every) de un puñado de celdas individuales,
  algo que correr() no expone.

  Tres estadísticos, sobre los incrementos Δx_t = x_{t+1} - x_t de cada celda trackeada:

  1. ASIMETRÍA DE INCREMENTOS: tercer momento estandarizado (skewness) de Δx_t.

  2. PRODUCCIÓN DE ENTROPÍA LOCAL: lam_eff(celda,t) * |a(celda,t)| acumulado en el tiempo,
     leyendo la misma cantidad que el motor ya calcula (variable "disipado" de _paso),
     evaluada en la celda en vez de sumada sobre todo el campo. Por construcción del
     modelo (lam_eff>0, |a|>=0) esto es monótono no-decreciente SIEMPRE — no es un test
     contra azar, es una réplica a nivel micro del resultado ya confirmado de C-N3/CS009.
     Se reporta como chequeo de consistencia, no como hallazgo nuevo.

  3. VIOLACIÓN DE BALANCE DETALLADO (proxy documentado de Σ_T = log(P[Γ]/P[Γ†])):
     cg001_field.py es DETERMINISTA — no expone un kernel de transición estocástico
     explícito, así que NO se puede calcular log P[Γ]/P[Γ†] sobre trayectorias exactas
     (eso requeriría un modelo estocástico explícito que este código no tiene). Lo que se
     calcula en su lugar es el estimador estándar de termodinámica estocástica para
     detectar asimetría temporal sin conocer el kernel exacto (comparar la distribución
     conjunta empírica de pares consecutivos (x_t, x_{t+1}) contra la misma distribución
     leída al revés (x_{t+1}, x_t) — si la dinámica fuese temporalmente simétrica
     (balance detallado), ambas coincidirían; el estadístico es la divergencia KL entre
     ambas, estimada con un histograma 2D con suavizado de Laplace). Esto se declara
     explícitamente como PROXY, no como medición exacta.

  CONTROL NULL (dos formas independientes, para no depender de una sola noción de "azar"):
  (a) ORDEN BARAJADO: permutar al azar el orden temporal de los mismos incrementos Δx_t
      reales (misma distribución marginal de pasos, sin la secuencia real).
  (b) PASEO ALEATORIO SIMÉTRICO: incrementos gaussianos nuevos con la MISMA varianza que
      los incrementos reales, sin ninguna estructura de la dinámica real.
  12 semillas (estándar del proyecto) para REAL y para cada brazo NULL.

  PRUEBA ANTI-SHANNON (apagar el mecanismo candidato): la memoria histórica "m" (parámetro
  gamma) es la única pieza de cg001_field.py que rompe la simetría temporal más allá de la
  simple relajación difusiva instantánea (gamma=0 -> sin memoria, lam_eff=lam constante).
  Se corre también con gamma=0 (mismo epsilon, mismas semillas) y se compara: si el
  estadístico de violación de balance detallado cae hacia el rango NULL al apagar la
  memoria, es evidencia de que la memoria es la pieza que produce la asimetría; si no
  cambia, la memoria no es la responsable y la asimetría (si existe) viene de otro lado
  (por ejemplo, de la relajación misma).

  NO se declara aquí ningún cierre ni veredicto final de "confirmado" / "refutado" — se
  reportan los números y los z-scores, y la decisión de qué hacer con ese resultado queda
  para Alexis (regla de la casa del proyecto: ningún experimento se cierra sin su
  autorización explícita, ver nota-permanente-no-cerrar-experimentos.md).
"""
from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import skew

from cg001_field import FieldConfig, _inicializar_phi  # noqa: E402  (motor reusado, no modificado)

HERE = Path(__file__).resolve().parent
OUT = HERE / "resultados_cs076_direccion_temporal"
OUT.mkdir(exist_ok=True)

N_SEMILLAS = 12
N_CELDAS_TRACKEADAS = 10  # celdas individuales seguidas paso a paso por semilla
N_BINS_KL = 12  # bins del histograma 2D para el proxy de balance detallado
PSEUDO_CONTEO = 0.5  # suavizado de Laplace (evita log(0) en la KL)


# ---------------------------------------------------------------------------
# 1) Trayectoria celda-por-celda, reusando la dinámica exacta de cg001_field
# ---------------------------------------------------------------------------
def _elegir_celdas(rng: np.random.Generator, cfg: FieldConfig, incluir_centro: bool) -> list[tuple[int, int, int]]:
    """Elige N_CELDAS_TRACKEADAS celdas al azar (más la celda del epsilon si corresponde)."""
    celdas = []
    if incluir_centro:
        c = cfg.L // 2
        celdas.append((c, c, c))
    while len(celdas) < N_CELDAS_TRACKEADAS:
        z, y, x = rng.integers(0, cfg.L, size=3)
        celdas.append((int(z), int(y), int(x)))
    return celdas


def correr_trayectoria(seed: int, con_epsilon: bool, cfg: FieldConfig) -> dict:
    """
    Avanza el campo EXACTAMENTE con la misma fórmula que _paso() de cg001_field.py
    (gaussian_filter + lam_eff con memoria + relajación), pero además guarda, paso a
    paso, el valor de phi/a/lam_eff/disipado en un conjunto fijo de celdas -- algo que
    correr() no expone porque sólo resume cada log_every pasos.
    """
    rng = np.random.default_rng(seed)
    phi = _inicializar_phi(rng, cfg, con_epsilon)
    m = np.zeros_like(phi)
    celdas = _elegir_celdas(np.random.default_rng(seed + 10_000), cfg, con_epsilon)

    series = {c: {"phi": [], "a": [], "disip_local": []} for c in celdas}

    for _t in range(cfg.pasos):
        vecindad = gaussian_filter(phi, sigma=cfg.sigma, mode="wrap")
        a = phi - vecindad
        abs_a = np.abs(a)
        m = cfg.decay * m + abs_a
        lam_eff = cfg.lam / (1.0 + cfg.gamma * m)

        for c in celdas:
            series[c]["phi"].append(float(phi[c]))
            series[c]["a"].append(float(a[c]))
            series[c]["disip_local"].append(float((lam_eff * abs_a)[c]))

        phi = phi - lam_eff * a

    return {"celdas": celdas, "series": series}


# ---------------------------------------------------------------------------
# 2) Los tres estadísticos
# ---------------------------------------------------------------------------
def incrementos_pooled(series: dict) -> np.ndarray:
    """Concatena Δx_t de todas las celdas trackeadas en una sola serie de incrementos."""
    todos = []
    for c, s in series.items():
        x = np.asarray(s["phi"], dtype=np.float64)
        todos.append(np.diff(x))
    return np.concatenate(todos)


def asimetria_incrementos(incr: np.ndarray) -> float:
    return float(skew(incr))


def produccion_entropia_local(series: dict) -> dict:
    """Suma acumulada de disip_local por celda; reporta si es monótona (debe serlo siempre)."""
    acumulados = []
    monotonas = []
    for c, s in series.items():
        d = np.asarray(s["disip_local"], dtype=np.float64)
        acumulados.append(float(d.sum()))
        monotonas.append(bool(np.all(d >= -1e-12)))  # disip_local siempre >=0 por construcción
    return {"entropia_local_media": float(np.mean(acumulados)), "todas_monotonas": all(monotonas)}


def _kl_balance_detallado(pares_x: np.ndarray, pares_y: np.ndarray) -> float:
    """
    Proxy de Sigma_T: KL( P(x_t,x_t+1) || P(x_t+1,x_t) ) via histograma 2D + Laplace.
    pares_x, pares_y son x_t y x_t+1 respectivamente (mismo largo).
    """
    rango = (float(min(pares_x.min(), pares_y.min())), float(max(pares_x.max(), pares_y.max())))
    if rango[0] == rango[1]:
        return 0.0
    hist_fwd, _, _ = np.histogram2d(pares_x, pares_y, bins=N_BINS_KL, range=(rango, rango))
    hist_bwd = hist_fwd.T  # leer al revés = swap (x_t,x_t+1) -> (x_t+1,x_t)

    p = hist_fwd + PSEUDO_CONTEO
    q = hist_bwd + PSEUDO_CONTEO
    p = p / p.sum()
    q = q / q.sum()
    kl = float(np.sum(p * np.log(p / q)))
    return kl


def violacion_balance_detallado(series: dict) -> float:
    xs, ys = [], []
    for c, s in series.items():
        x = np.asarray(s["phi"], dtype=np.float64)
        xs.append(x[:-1])
        ys.append(x[1:])
    return _kl_balance_detallado(np.concatenate(xs), np.concatenate(ys))


# ---------------------------------------------------------------------------
# 3) Controles NULL: reconstruir series "phi" sintéticas a partir de los incrementos
# ---------------------------------------------------------------------------
def null_orden_barajado(series: dict, rng: np.random.Generator) -> dict:
    out = {}
    for c, s in series.items():
        x = np.asarray(s["phi"], dtype=np.float64)
        incr = np.diff(x)
        incr_barajado = rng.permutation(incr)
        x_null = np.concatenate([[x[0]], x[0] + np.cumsum(incr_barajado)])
        out[c] = {"phi": x_null.tolist()}
    return out


def null_paseo_aleatorio(series: dict, rng: np.random.Generator) -> dict:
    out = {}
    for c, s in series.items():
        x = np.asarray(s["phi"], dtype=np.float64)
        incr = np.diff(x)
        sigma = float(incr.std())
        incr_sinteticos = rng.normal(0.0, sigma, size=incr.shape)
        x_null = np.concatenate([[x[0]], x[0] + np.cumsum(incr_sinteticos)])
        out[c] = {"phi": x_null.tolist()}
    return out


def estadisticos_de(series_solo_phi: dict) -> dict:
    incr = incrementos_pooled(series_solo_phi)
    xs, ys = [], []
    for c, s in series_solo_phi.items():
        x = np.asarray(s["phi"], dtype=np.float64)
        xs.append(x[:-1])
        ys.append(x[1:])
    kl = _kl_balance_detallado(np.concatenate(xs), np.concatenate(ys))
    return {"skew": asimetria_incrementos(incr), "kl_balance": kl}


# ---------------------------------------------------------------------------
# 4) Corrida principal: REAL (gamma normal) + anti-Shannon (gamma=0) x 12 semillas
# ---------------------------------------------------------------------------
def z_score(real_vals: list[float], null_vals: list[float]) -> float:
    mu, sd = float(np.mean(null_vals)), float(np.std(null_vals))
    if sd < 1e-12:
        return float("nan")
    return (float(np.mean(real_vals)) - mu) / sd


def correr_bloque(cfg: FieldConfig, etiqueta: str) -> dict:
    real_skew, real_kl, real_entropia = [], [], []
    null_shuf_skew, null_shuf_kl = [], []
    null_walk_skew, null_walk_kl = [], []
    monotonas_ok = []

    for seed in range(N_SEMILLAS):
        traj = correr_trayectoria(seed=seed, con_epsilon=True, cfg=cfg)
        series = traj["series"]

        incr = incrementos_pooled(series)
        real_skew.append(asimetria_incrementos(incr))
        real_kl.append(violacion_balance_detallado(series))
        ent = produccion_entropia_local(series)
        real_entropia.append(ent["entropia_local_media"])
        monotonas_ok.append(ent["todas_monotonas"])

        rng_null = np.random.default_rng(seed + 50_000)
        s_shuf = null_orden_barajado(series, rng_null)
        st = estadisticos_de(s_shuf)
        null_shuf_skew.append(st["skew"])
        null_shuf_kl.append(st["kl_balance"])

        rng_null2 = np.random.default_rng(seed + 90_000)
        s_walk = null_paseo_aleatorio(series, rng_null2)
        st2 = estadisticos_de(s_walk)
        null_walk_skew.append(st2["skew"])
        null_walk_kl.append(st2["kl_balance"])

    return {
        "etiqueta": etiqueta,
        "gamma": cfg.gamma,
        "n_semillas": N_SEMILLAS,
        "entropia_local_media": float(np.mean(real_entropia)),
        "entropia_local_siempre_monotona": all(monotonas_ok),
        "skew": {
            "real_media": float(np.mean(real_skew)),
            "null_barajado_media": float(np.mean(null_shuf_skew)),
            "null_barajado_z": z_score(real_skew, null_shuf_skew),
            "null_paseo_media": float(np.mean(null_walk_skew)),
            "null_paseo_z": z_score(real_skew, null_walk_skew),
        },
        "kl_balance_detallado": {
            "real_media": float(np.mean(real_kl)),
            "null_barajado_media": float(np.mean(null_shuf_kl)),
            "null_barajado_z": z_score(real_kl, null_shuf_kl),
            "null_paseo_media": float(np.mean(null_walk_kl)),
            "null_paseo_z": z_score(real_kl, null_walk_kl),
        },
        "raw": {
            "real_skew": real_skew, "real_kl": real_kl,
            "null_barajado_skew": null_shuf_skew, "null_barajado_kl": null_shuf_kl,
            "null_paseo_skew": null_walk_skew, "null_paseo_kl": null_walk_kl,
        },
    }


def main() -> None:
    t0 = time.time()
    cfg_normal = FieldConfig()  # gamma=8.0 (memoria activa, default del proyecto)
    cfg_sin_memoria = replace(cfg_normal, gamma=0.0)  # anti-Shannon: apaga el candidato

    print("=== CS076 — Dirección temporal T+/T- a nivel micro (C-N2.5.6-10) ===")
    print(f"L={cfg_normal.L} pasos={cfg_normal.pasos} celdas/semilla={N_CELDAS_TRACKEADAS} semillas={N_SEMILLAS}\n")

    print("Bloque A: gamma normal (memoria activa, configuración default del proyecto)...")
    bloque_normal = correr_bloque(cfg_normal, "gamma_normal")

    print("Bloque B (anti-Shannon): gamma=0 (memoria apagada)...")
    bloque_sin_memoria = correr_bloque(cfg_sin_memoria, "gamma_cero")

    resultado = {
        "experimento": "CS076_direccion_temporal",
        "fecha": time.strftime("%Y-%m-%d"),
        "duracion_seg": round(time.time() - t0, 1),
        "bloque_gamma_normal": bloque_normal,
        "bloque_gamma_cero_anti_shannon": bloque_sin_memoria,
    }

    out_path = OUT / "resultado_cs076.json"
    out_path.write_text(json.dumps(resultado, indent=2, ensure_ascii=False))

    def resumen(b: dict) -> None:
        print(f"\n--- {b['etiqueta']} (gamma={b['gamma']}) ---")
        print(f"  entropía local: media={b['entropia_local_media']:.4f} monótona_siempre={b['entropia_local_siempre_monotona']}")
        sk = b["skew"]
        print(f"  skew  REAL={sk['real_media']:+.4f}  NULL_barajado={sk['null_barajado_media']:+.4f} (z={sk['null_barajado_z']:+.2f})  NULL_paseo={sk['null_paseo_media']:+.4f} (z={sk['null_paseo_z']:+.2f})")
        kl = b["kl_balance_detallado"]
        print(f"  KL_balance REAL={kl['real_media']:.4f}  NULL_barajado={kl['null_barajado_media']:.4f} (z={kl['null_barajado_z']:+.2f})  NULL_paseo={kl['null_paseo_media']:.4f} (z={kl['null_paseo_z']:+.2f})")

    resumen(bloque_normal)
    resumen(bloque_sin_memoria)
    print(f"\nDuración total: {resultado['duracion_seg']}s. Resultado guardado en {out_path}")
    print("\nNO es un veredicto de cierre — números crudos para que Alexis decida.")


if __name__ == "__main__":
    main()
