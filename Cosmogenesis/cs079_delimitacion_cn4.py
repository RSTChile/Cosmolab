#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs079_delimitacion_cn4.py — CS079: delimitación FoF como hipótesis propia (nodo C-N4)
=======================================================================================

Quién soy / qué hago (código autodescriptivo):
  Ataca el nodo C-N4 del roadmap Fase I-C: CS073 traza sumideros/grumos de gas con un
  criterio "amigos-de-amigos" (friends-of-friends, FoF) de longitud de enlace FIJA, pero
  esa frontera nunca se puso a prueba como hipótesis propia. Este script pregunta: ¿la
  distribución de distancias entre partículas de gas (y el perfil de densidad radial
  alrededor de los picos más densos) muestra una discontinuidad/valle genuina en REAL que
  la distinga de los controles NULL, o cualquier longitud de enlace razonable "ve" un
  grupo aunque el campo no tenga estructura genuina?

Qué NO hace (a propósito, reglas de la casa de esta tarea):
  No corre ninguna simulación Phantom nueva (los volcados ya existen en
  /Users/alexis/phantom_cs073/bateria_n2000/). No toca leer_volcado_phantom.py ni
  cs072_modulos/ ni nada de cs073 — sólo IMPORTA leer_dump() de sólo lectura. No declara
  cierre/verdicto de la hipótesis C-N4 — sólo reporta números; la decisión final es de
  Alexis López Tapia (metodología anti-Shannon del proyecto: cualquier hallazgo positivo
  debe sobrevivir comparar contra NULL, y el veredicto de cierre nunca lo pone el script).

Método:
  1. Lee el volcado más evolucionado (cosmog_00500, el último paso disponible en todas
     las corridas de la batería) de ic_real y de ic_null1 .. ic_null8 con leer_dump()
     (usa sarracen bajo el capó, sin tocar el binario Fortran a mano).
  2. Para cada corrida: k-d tree (scipy.spatial.cKDTree) sobre (x,y,z) de las partículas
     de gas, distancia al k-ésimo vecino más cercano (k=8, razonable para SPH/FoF) para
     cada partícula → histograma de esas distancias.
       Lógica: si el FoF fija una longitud de enlace arbitraria pero el gas SÍ tiene una
       frontera física real (grumo denso vs. entorno difuso), el histograma de distancias
       a vecinos debería ser BIMODAL — dos poblaciones, "dentro de un grumo" (distancias
       cortas) y "en el campo difuso" (distancias largas) — con un valle real entre
       modos, que es justamente donde un FoF bien elegido debería cortar. Si el campo NO
       tiene estructura genuina (NULL), el histograma debería ser unimodal/suave, y
       cualquier longitud de enlace que "encuentre grupos" ahí sería un artefacto del
       parámetro, no de los datos.
  3. Cuantifica bimodalidad con dos métricas independientes (no se instaló diptest ni
     unidip — no están en el venv del proyecto y la tarea prohíbe tocar el entorno más
     allá de lo imprescindible; ambas métricas usadas son estándar y ya cubiertas por
     scipy/numpy):
       (a) Coeficiente de bimodalidad de Sarle/Pfister (BC), a partir de skewness y
           kurtosis muestral de las distancias k-NN. BC > 5/9 ≈ 0.555 es el umbral
           clásico de "sospecha de bimodalidad" (heurística, NO es una prueba formal
           como el dip test de Hartigan — se reporta como tal, sin sobre-interpretar).
       (b) Profundidad de valle relativa: scipy.signal.find_peaks sobre el histograma
           suavizado (kernel gaussiano simple) ubica los dos picos más prominentes y el
           mínimo entre ellos; se reporta profundidad = (altura_pico_menor -
           altura_valle) / altura_pico_menor. 0 = no hay valle (unimodal); →1 = valle
           casi total (dos poblaciones separadas por completo).
  4. Segunda métrica, independiente de la anterior: para cada corrida, toma la partícula
     de gas de mayor densidad (columna `rho`, ya calculada por leer_dump()/calc_density
     de sarracen) como "pico", calcula la distancia radial de TODAS las partículas de gas
     a ese pico, las agrupa en cascarones log-espaciados y promedia rho por cascarón →
     perfil de densidad radial. Ajusta una pendiente log-log (power-law) por mínimos
     cuadrados sobre los cascarones con datos; una pendiente negativa pronunciada = caída
     rápida de densidad = frontera nítida entre grumo y entorno; una pendiente cercana a
     cero o con R² bajo = sin frontera clara / perfil plano-ruidoso.
  5. Compara REAL contra cada NULL (null1..null8) en ambas familias de métricas y guarda
     todo (números crudos + plots) para que Alexis o cualquier CS lo pueda re-auditar.

Salida (carpeta cs079_resultados/, creada junto a este script):
  hist_knn.png          — histogramas de distancia k-NN: real vs. null1/2/3 superpuestos
  perfil_radial.png     — perfiles de densidad radial: real vs. null1/2/3
  resultados_cs079.json — tabla completa de métricas para las 9 corridas (real+8 nulls)

Cómo correrlo:
    source venv/bin/activate
    python3 cs079_delimitacion_cn4.py

No declara cierre. El veredicto final sobre si C-N4 amenaza la validez del FoF de CS073
es de Alexis López Tapia — este script sólo mide y reporta.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis

# Import de solo lectura de la herramienta ya construida/verificada — NO se modifica.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from leer_volcado_phantom import leer_dump  # noqa: E402

BASE = Path("/Users/alexis/phantom_cs073/bateria_n2000")
CORRIDAS = ["ic_real"] + [f"ic_null{i}" for i in range(1, 9)]
DUMP = "cosmog_00500"  # último paso disponible en todas las corridas (verificado con ls)
K_VECINOS = 8
OUT_DIR = Path(__file__).resolve().parent / "cs079_resultados"
OUT_DIR.mkdir(exist_ok=True)


def distancias_knn(gas, k: int = K_VECINOS) -> np.ndarray:
    """Distancia al k-ésimo vecino más cercano para cada partícula de gas."""
    xyz = gas[["x", "y", "z"]].to_numpy()
    tree = cKDTree(xyz)
    # k+1 porque el vecino "0" es la propia partícula (distancia 0)
    dist, _ = tree.query(xyz, k=k + 1)
    return dist[:, -1]


def coeficiente_bimodalidad_sarle(x: np.ndarray) -> float:
    """BC de Sarle/Pfister. BC > 5/9 ~ sospecha de bimodalidad (heurística, no prueba
    formal). n = tamaño muestral, se usa la corrección estándar de exceso de kurtosis
    para muestras finitas."""
    n = len(x)
    g = skew(x, bias=False)
    k = kurtosis(x, fisher=True, bias=False)  # excess kurtosis
    correccion = (3 * (n - 1) ** 2) / ((n - 2) * (n - 3)) if n > 3 else 3.0
    bc = (g ** 2 + 1) / (k + correccion)
    return float(bc)


def profundidad_valle(x: np.ndarray, bins: int = 40) -> dict:
    """Histograma suavizado (kernel gaussiano simple sobre los conteos) -> localiza los
    dos picos más prominentes y el valle entre ellos -> profundidad relativa."""
    counts, edges = np.histogram(x, bins=bins)
    centros = 0.5 * (edges[:-1] + edges[1:])
    # suavizado simple (kernel gaussiano de 3 bins de sigma) para no cazar ruido de bin-a-bin
    kernel = np.exp(-0.5 * (np.arange(-4, 5) / 1.5) ** 2)
    kernel /= kernel.sum()
    suave = np.convolve(counts, kernel, mode="same")

    picos, props = find_peaks(suave, prominence=suave.max() * 0.02)
    resultado = {
        "n_picos_detectados": int(len(picos)),
        "profundidad_valle": 0.0,
        "dist_entre_modos": None,
        "counts": counts.tolist(),
        "centros": centros.tolist(),
    }
    if len(picos) >= 2:
        # tomamos los DOS picos más prominentes (por altura suavizada)
        orden = picos[np.argsort(suave[picos])[::-1]][:2]
        p1, p2 = sorted(orden)
        valle_idx = p1 + int(np.argmin(suave[p1:p2 + 1]))
        altura_valle = suave[valle_idx]
        altura_pico_menor = min(suave[p1], suave[p2])
        prof = 0.0
        if altura_pico_menor > 0:
            prof = float((altura_pico_menor - altura_valle) / altura_pico_menor)
        resultado["profundidad_valle"] = prof
        resultado["dist_entre_modos"] = float(centros[p2] - centros[p1])
    return resultado


def perfil_densidad_radial(gas, n_bins: int = 15) -> dict:
    """Perfil de densidad radial alrededor de la partícula de gas más densa. Devuelve
    cascarones log-espaciados con rho promedio, y una pendiente log-log ajustada por
    mínimos cuadrados sobre los cascarones con >=3 partículas."""
    xyz = gas[["x", "y", "z"]].to_numpy()
    rho = gas["rho"].to_numpy()
    i_pico = int(np.argmax(rho))
    centro = xyz[i_pico]
    r = np.linalg.norm(xyz - centro, axis=1)
    r_no_cero = r[r > 0]
    if len(r_no_cero) < 10:
        return {"pendiente_loglog": None, "r2": None, "rho_pico": float(rho[i_pico]),
                "cascarones_r": [], "cascarones_rho": []}

    r_min, r_max = r_no_cero.min(), r_no_cero.max()
    bordes = np.logspace(np.log10(max(r_min, 1e-6)), np.log10(r_max), n_bins + 1)
    r_medios, rho_medios, n_por_cascaron = [], [], []
    for lo, hi in zip(bordes[:-1], bordes[1:]):
        mascara = (r >= lo) & (r < hi)
        if mascara.sum() >= 3:
            r_medios.append(np.sqrt(lo * hi))  # centro geométrico del cascarón log
            rho_medios.append(float(np.mean(rho[mascara])))
            n_por_cascaron.append(int(mascara.sum()))

    pendiente, r2 = None, None
    if len(r_medios) >= 4:
        logr = np.log10(r_medios)
        logrho = np.log10(rho_medios)
        A = np.vstack([logr, np.ones_like(logr)]).T
        coef, residuales, _, _ = np.linalg.lstsq(A, logrho, rcond=None)
        pendiente = float(coef[0])
        pred = A @ coef
        ss_res = float(np.sum((logrho - pred) ** 2))
        ss_tot = float(np.sum((logrho - logrho.mean()) ** 2))
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else None

    return {
        "pendiente_loglog": pendiente,
        "r2": r2,
        "rho_pico": float(rho[i_pico]),
        "n_gas": int(len(gas)),
        "cascarones_r": r_medios,
        "cascarones_rho": rho_medios,
        "cascarones_n": n_por_cascaron,
    }


def main() -> None:
    resultados = {}
    for nombre in CORRIDAS:
        path = BASE / nombre / DUMP
        if not path.exists():
            print(f"AVISO: {path} no existe, se salta {nombre}")
            continue
        print(f"Leyendo {nombre}/{DUMP} ...")
        gas, sinks = leer_dump(path)
        n_gas = len(gas)
        n_sinks = 0 if sinks is None else len(sinks)

        dist_knn = distancias_knn(gas)
        bc = coeficiente_bimodalidad_sarle(dist_knn)
        valle = profundidad_valle(dist_knn)
        perfil = perfil_densidad_radial(gas)

        resultados[nombre] = {
            "n_gas": n_gas,
            "n_sinks": n_sinks,
            "knn_dist_mediana": float(np.median(dist_knn)),
            "knn_dist_media": float(np.mean(dist_knn)),
            "knn_dist_std": float(np.std(dist_knn)),
            "coef_bimodalidad_sarle": bc,
            "bimodal_sospechoso_BC>5/9": bool(bc > 5 / 9),
            "valle": valle,
            "perfil_radial": perfil,
            "dist_knn_raw_sample": dist_knn.tolist(),  # para reproducir plots/reauditar
        }
        print(
            f"  {nombre}: n_gas={n_gas}, n_sinks={n_sinks}, "
            f"BC={bc:.3f}, n_picos_hist={valle['n_picos_detectados']}, "
            f"profundidad_valle={valle['profundidad_valle']:.3f}, "
            f"pendiente_loglog_rho={perfil['pendiente_loglog']}"
        )

    # --- tabla resumen a stdout ---
    print("\n=== RESUMEN CS079 (C-N4: delimitación FoF como hipótesis propia) ===")
    print(f"{'corrida':10s} {'n_gas':>6s} {'BC':>7s} {'valle':>7s} {'dist_modos':>11s} {'pend_loglog':>12s} {'r2':>6s}")
    for nombre, r in resultados.items():
        dm = r["valle"]["dist_entre_modos"]
        dm_str = f"{dm:.4f}" if dm is not None else "n/a"
        pend = r["perfil_radial"]["pendiente_loglog"]
        pend_str = f"{pend:.3f}" if pend is not None else "n/a"
        r2 = r["perfil_radial"]["r2"]
        r2_str = f"{r2:.3f}" if r2 is not None else "n/a"
        print(f"{nombre:10s} {r['n_gas']:6d} {r['coef_bimodalidad_sarle']:7.3f} "
              f"{r['valle']['profundidad_valle']:7.3f} {dm_str:>11s} {pend_str:>12s} {r2_str:>6s}")

    # --- guardar JSON completo (sin el array raw de distancias para no inflar tamaño de más) ---
    resultados_json = {}
    for nombre, r in resultados.items():
        r2 = dict(r)
        r2.pop("dist_knn_raw_sample", None)
        resultados_json[nombre] = r2
    with open(OUT_DIR / "resultados_cs079.json", "w") as f:
        json.dump(resultados_json, f, indent=2)
    print(f"\nJSON escrito: {OUT_DIR / 'resultados_cs079.json'}")

    # --- plots ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # histograma k-NN: real vs 3 nulls representativos
        fig, ax = plt.subplots(figsize=(8, 5))
        comparar = [c for c in ["ic_real", "ic_null1", "ic_null2", "ic_null3"] if c in resultados]
        for nombre in comparar:
            dist = np.array(resultados[nombre]["dist_knn_raw_sample"])
            estilo = "-" if nombre == "ic_real" else "--"
            ax.hist(dist, bins=40, histtype="step", density=True, label=nombre, linestyle=estilo, linewidth=2 if nombre == "ic_real" else 1.3)
        ax.set_xlabel(f"distancia al {K_VECINOS}º vecino más cercano")
        ax.set_ylabel("densidad de probabilidad")
        ax.set_title("CS079 (C-N4): histograma de distancias k-NN — REAL vs NULL")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / "hist_knn.png", dpi=140)
        plt.close(fig)

        # perfil de densidad radial: real vs 3 nulls representativos
        fig, ax = plt.subplots(figsize=(8, 5))
        for nombre in comparar:
            perfil = resultados[nombre]["perfil_radial"]
            if perfil["cascarones_r"]:
                estilo = "-o" if nombre == "ic_real" else "--x"
                ax.plot(perfil["cascarones_r"], perfil["cascarones_rho"], estilo, label=nombre,
                         linewidth=2 if nombre == "ic_real" else 1.3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("distancia radial al pico de densidad más alto")
        ax.set_ylabel("rho promedio por cascarón")
        ax.set_title("CS079 (C-N4): perfil de densidad radial — REAL vs NULL")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / "perfil_radial.png", dpi=140)
        plt.close(fig)
        print(f"PNGs escritos en: {OUT_DIR}")
    except ImportError:
        print("AVISO: matplotlib no disponible, se omiten los plots (JSON sí se escribió).")

    print("\nCS079 terminado. Este script NO declara cierre/verdicto — el veredicto sobre")
    print("si C-N4 amenaza la validez del FoF de CS073 es de Alexis López Tapia.")


if __name__ == "__main__":
    main()
