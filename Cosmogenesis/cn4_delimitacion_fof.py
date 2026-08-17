#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cn4_delimitacion_fof.py — C-N4: ¿la frontera "amigos-de-amigos" (FoF) es real o arbitraria?
=============================================================================================

Quién soy / qué hago (código autodescriptivo):
  Ejecuta el protocolo del nodo C-N4 (DISENO_EXPERIMENTOS_NODOS_ABIERTOS_desde_2.5.5_CS.md,
  sección 7), ahora desbloqueado por leer_volcado_phantom.py. Pregunta, como hipótesis
  propia nunca puesta a prueba: el criterio "amigos-de-amigos" (FoF) que el proyecto usa
  para trazar fronteras entre sumideros/grumos y el resto del gas en CS073, ¿cae donde el
  gas de verdad tiene una discontinuidad de densidad (un "vacío" real en la distribución de
  distancias entre partículas vecinas), o cualquier longitud de enlace razonable dibujaría
  "algo parecido a un grupo" igual, incluso sobre un campo sin estructura genuina?

Qué mido y por qué (nearest-neighbor, no un k-NN cualquiera):
  Para cada partícula de gas calculo la distancia a su vecino MÁS CERCANO (k=1, vía
  scipy.spatial.cKDTree, query(k=2) y descarto la partícula misma). Elijo NN=1 como
  estadística primaria porque el propio algoritmo FoF opera sobre PARES: dos partículas
  quedan "amigas" si su distancia es menor que la longitud de enlace, encadenando después
  por componentes conexas. La distribución de distancias al vecino más próximo es
  exactamente la estadística de la que depende esa decisión binaria. También calculo el
  2º y 3er vecino más cercano como chequeo de robustez: si un "vacío" sólo aparece en k=1 y
  desaparece en k=2/3, es una señal más frágil que si aparece consistente en las tres.

Corridas analizadas (sólo lectura, nada se modifica en /Users/alexis/phantom_cs073/):
  ic_real + ic_null1, ic_null2, ic_null3 de bateria_n2000/, todas en el dump cosmog_00500
  (el último disponible en las cuatro carpetas -- misma "edad" física t=0.5, con estructura
  ya formada: 7-8 sumideros nacidos por corrida según
  RESULTADO_bateria_ignicion_sumideros_N2000_CS.md). Se usan sólo las partículas de GAS (no
  las de sumidero) -- son las que el criterio FoF tendría que delimitar del resto.

Control NULL #1 (físico, ya presente en el diseño del proyecto):
  Las corridas ic_null1..3 -- mismo proceso físico que ic_real, condición inicial de la
  malla causal barajada (b_FoF=0.2 nunca ajustada por corrida, ver cs073_cierre_holistico.py
  y RESULTADO_bateria_ignicion_sumideros_N2000_CS.md). Ya sabemos por ese resultado previo
  que el NULL también forma sumideros (7-8, con menos masa) -- así que un vacío en el NULL
  no sería sorpresa, pero su AUSENCIA o presencia relativa a REAL sí es informativa.

Control NULL #2 (interno, más duro, pedido explícito del protocolo, paso 4):
  Para REAL y para ic_null1, ADEMÁS del NULL físico de arriba, construyo un control de
  "barajado de ejes": permuto INDEPENDIENTEMENTE los arrays x, y, z entre las partículas de
  ESE MISMO dump (cada eje se baraja por separado, con su propia permutación aleatoria).
  Esto preserva la distribución marginal 1D de cada coordenada (si había más partículas
  concentradas en cierto rango de x, eso se conserva) pero destruye la correlación conjunta
  (x,y,z) que forma los grumos 3D reales -- aísla si el "vacío" depende de la ESTRUCTURA
  espacial genuina o sólo de cómo se reparten las densidades a lo largo de cada eje por
  separado. Semilla fija (ver SEMILLA_BARAJADO) para reproducibilidad.

Detección de vacío/gap (automática, no a ojo, mismo criterio para las 4+2 corridas):
  KDE gaussiana sobre log10(distancia NN) (scipy.stats.gaussian_kde, bw_method='scott'),
  evaluada en una grilla fina. Busco extremos locales (scipy.signal.argrelextrema). Si hay
  al menos un mínimo local flanqueado por dos máximos locales, calculo la "profundidad del
  vacío" = min(altura_pico_izq, altura_pico_der) / altura_valle -- cuánto más alto el pico
  más bajo comparado con el valle. profundidad=1.0 significa sin dip real (curva
  monótona o un único máximo, sin mínimo local interior). Este número es un criterio de
  REPORTE, no decide el veredicto del experimento -- los histogramas y los números crudos
  quedan en cn4_resultados.json y en la figura para que el director juzgue.

Comparación con el criterio FoF vigente del proyecto:
  cs073_cierre_holistico.py (línea ~158) y cs074_energia_holistica.py (línea ~227) usan
  linking_length = 0.2 * espaciamiento (b=0.2, convención cosmológica estándar -- comentario
  explícito "b_FoF=0.2 estándar" en el docstring de cs073_cierre_holistico.py), con
  espaciamiento = lado_de_caja / N_total^(1/3) (cs073_ignicion.py línea 109-110) o, en la
  variante de cierre holístico, espaciamiento = 1.0 * a_final (unidades comóviles
  normalizadas). Acá repito ese MISMO cálculo (b=0.2 * espaciamiento) usando el espaciamiento
  medio real de cada dump de Phantom: espaciamiento = (Volumen_bounding_box / N_gas)^(1/3) --
  el análogo físico directo de "lado_de_caja / N^(1/3)" cuando no hay una caja periódica
  explícita. Comparo esa longitud de enlace vigente con la posición del vacío (si lo hay) en
  el histograma NN de cada corrida.

  Nota que este script deja documentada, no escondida: la batería N2000 de Phantom en sí
  (RESULTADO_bateria_ignicion_sumideros_N2000_CS.md) NO usa FoF para decidir qué partículas
  forman un sumidero -- usa el criterio propio de Phantom (rho_crit_cgs=1000, h_acc=0.3,
  r_crit=0.6: un umbral de densidad + radio de acreción, mecanismo distinto). El FoF con
  b=0.2 vive en el pipeline paralelo de N-body puro (cs073_cierre_holistico.py /
  cs074_energia_holistica.py), que es el único criterio FoF explícito y citable que el
  proyecto usa hoy. La comparación de este experimento es contra ESE criterio.

Restricciones respetadas: sólo lectura de /Users/alexis/phantom_cs073/ (nada se modifica ahí,
ni leer_volcado_phantom.py se toca). Este script NO declara cierre ni veredicto final de
C-N4 -- reporta números y figuras; la adjudicación es del director (ver
ADJUDICACION_CN4_delimitacion_fof_CS.md, que sí resume method + resultado pero sin cerrar
el arco).

Salidas al correr como script:
  - cn4_histogramas_nn.png  (grilla de histogramas NN con KDE, vacío marcado si lo hay, y
    la longitud de enlace vigente del proyecto superpuesta como línea vertical)
  - cn4_resultados.json     (todos los números crudos: distancias resumidas, picos/valles
    detectados, espaciamiento y linking_length por corrida)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
from scipy.signal import argrelextrema

from leer_volcado_phantom import leer_dump

BATERIA = Path("/Users/alexis/phantom_cs073/bateria_n2000")
DUMP = "cosmog_00500"  # último paso disponible en las 4 carpetas -> misma edad física t=0.5
CORRIDAS = ["ic_real", "ic_null1", "ic_null2", "ic_null3"]
SEMILLA_BARAJADO = 20260805  # fija, reproducible (fecha de la sesión como semilla, sin significado especial)

AQUI = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Carga y geometría
# ---------------------------------------------------------------------------

def cargar_posiciones_gas(carpeta: Path, dump: str) -> np.ndarray:
    """Lee un volcado con leer_dump() y devuelve SOLO las posiciones (N,3) del gas
    (no de los sumideros -- son las partículas que el criterio FoF tiene que delimitar)."""
    gas, _sinks = leer_dump(carpeta / dump)
    return gas[["x", "y", "z"]].to_numpy(dtype=float)


def distancias_kNN(pos: np.ndarray, k: int) -> np.ndarray:
    """Distancia de cada partícula a su k-ésimo vecino más cercano (k=1 -> el más próximo),
    excluyendo la partícula misma. cKDTree.query(k=k+1) porque el vecino "0" es uno mismo
    (distancia 0)."""
    tree = cKDTree(pos)
    d, _idx = tree.query(pos, k=k + 1)
    return d[:, k]


def barajar_ejes(pos: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Control interno duro (paso 4 del protocolo): permuta x, y, z INDEPENDIENTEMENTE entre
    las partículas. Preserva la distribución marginal 1D de cada coordenada, destruye la
    correlación conjunta (x,y,z) que forma estructura 3D real."""
    out = np.empty_like(pos)
    for eje in range(pos.shape[1]):
        out[:, eje] = rng.permutation(pos[:, eje])
    return out


def espaciamiento_y_enlace(pos: np.ndarray) -> tuple[float, float]:
    """Espaciamiento medio = (volumen del bounding box / N partículas) ** (1/3) -- el análogo
    físico directo de 'lado_de_caja / N_total^(1/3)' que usa cs073_ignicion.py /
    cs073_cierre_holistico.py cuando no hay caja periódica explícita. linking_length = 0.2 *
    espaciamiento, MISMA convención b=0.2 que el proyecto usa hoy (ver docstring del módulo)."""
    extents = pos.max(axis=0) - pos.min(axis=0)
    volumen = float(np.prod(extents))
    n = len(pos)
    espaciamiento = (volumen / n) ** (1.0 / 3.0)
    linking_length = 0.2 * espaciamiento
    return espaciamiento, linking_length


# ---------------------------------------------------------------------------
# Detección automática de vacío/gap en la distribución de distancias NN
# ---------------------------------------------------------------------------

def detectar_vacio(distancias: np.ndarray, n_grid: int = 600, orden_extremos: int = 5,
                    min_pico_relativo: float = 0.10) -> dict:
    """KDE gaussiana sobre log10(distancia), busca el par (máximo, mínimo, máximo) local más
    profundo. Devuelve un dict con toda la evidencia cruda -- no sólo un booleano -- para que
    el histograma y el informe puedan mostrar el número, no una etiqueta.

    min_pico_relativo descarta mínimos locales cuyos DOS picos vecinos sean ambos menores que
    esa fracción del máximo global de densidad -- sin este filtro, un "valle" trivial en la
    cola casi vacía de la distribución (dos wiggles de ruido numérico cerca de densidad~0)
    puede tener una razón pico/valle muy alta sin ser una separación real entre dos modos
    genuinos. Se detectó este artefacto exactamente en ic_null3 durante el desarrollo del
    script (el valle "ganador" sin filtro caía en un wiggle minúsculo del borde izquierdo,
    lejos del valle visualmente evidente que comparte con ic_null1/ic_null2) -- de ahí el
    filtro, documentado acá en vez de escondido."""
    log_d = np.log10(distancias[distancias > 0])
    kde = gaussian_kde(log_d, bw_method="scott")
    grilla = np.linspace(log_d.min() - 0.3, log_d.max() + 0.3, n_grid)
    densidad = kde(grilla)
    techo = float(densidad.max())

    idx_max = argrelextrema(densidad, np.greater, order=orden_extremos)[0]
    idx_min = argrelextrema(densidad, np.less, order=orden_extremos)[0]

    mejor = None
    for im in idx_min:
        izq = idx_max[idx_max < im]
        der = idx_max[idx_max > im]
        if len(izq) == 0 or len(der) == 0:
            continue
        pico_izq = izq[-1]
        pico_der = der[0]
        altura_valle = densidad[im]
        altura_pico = min(densidad[pico_izq], densidad[pico_der])
        if altura_valle <= 0:
            continue
        if altura_pico < min_pico_relativo * techo:
            continue  # ambos picos vecinos son ruido de cola, no modos genuinos
        profundidad = float(altura_pico / altura_valle)
        candidato = dict(
            profundidad=profundidad,
            valle_log10_dist=float(grilla[im]),
            valle_dist=float(10 ** grilla[im]),
            pico_izq_dist=float(10 ** grilla[pico_izq]),
            pico_der_dist=float(10 ** grilla[pico_der]),
        )
        if mejor is None or profundidad > mejor["profundidad"]:
            mejor = candidato

    return dict(
        n_maximos_locales=int(len(idx_max)),
        n_minimos_locales=int(len(idx_min)),
        hay_vacio_notable=bool(mejor is not None and mejor["profundidad"] > 1.5),
        mejor_valle=mejor,
        grilla_log10=grilla.tolist(),
        densidad_kde=densidad.tolist(),
    )


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------

def resumen_distancias(d: np.ndarray) -> dict:
    return dict(
        n=int(len(d)),
        min=float(d.min()), p25=float(np.percentile(d, 25)),
        mediana=float(np.median(d)), p75=float(np.percentile(d, 75)),
        max=float(d.max()), media=float(d.mean()), std=float(d.std()),
    )


def analizar_corrida(nombre: str, carpeta: Path, rng: np.random.Generator, con_barajado_ejes: bool) -> dict:
    pos = cargar_posiciones_gas(carpeta, DUMP)
    espaciamiento, linking_length = espaciamiento_y_enlace(pos)

    resultado = dict(corrida=nombre, n_gas=len(pos),
                      espaciamiento=espaciamiento, linking_length_proyecto=linking_length)

    for k in (1, 2, 3):
        d = distancias_kNN(pos, k)
        resultado[f"nn{k}"] = dict(resumen=resumen_distancias(d), vacio=detectar_vacio(d))

    if con_barajado_ejes:
        pos_barajada = barajar_ejes(pos, rng)
        d_bar = distancias_kNN(pos_barajada, 1)
        resultado["nn1_barajado_ejes"] = dict(resumen=resumen_distancias(d_bar), vacio=detectar_vacio(d_bar))

    return resultado


def graficar(resultados: list[dict], salida: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_filas = len(resultados)
    fig, ejes = plt.subplots(n_filas, 1, figsize=(8, 3.2 * n_filas), sharex=False)
    if n_filas == 1:
        ejes = [ejes]

    for ax, r in zip(ejes, resultados):
        nn1 = r["nn1"]
        grilla = np.array(r["nn1"]["vacio"]["grilla_log10"])
        densidad = np.array(r["nn1"]["vacio"]["densidad_kde"])
        ax.plot(10 ** grilla, densidad, color="#2b6cb0", lw=1.8, label="KDE dist. NN (k=1)")
        ax.set_xscale("log")
        ax.axvline(r["linking_length_proyecto"], color="#c0392b", ls="--", lw=1.4,
                   label=f"linking_length proyecto = {r['linking_length_proyecto']:.3g}")
        valle = nn1["vacio"]["mejor_valle"]
        if valle is not None:
            ax.axvline(valle["valle_dist"], color="#27ae60", ls=":", lw=1.6,
                       label=f"valle detectado = {valle['valle_dist']:.3g} (prof.={valle['profundidad']:.2f})")
        ax.set_title(f"{r['corrida']}  (n_gas={r['n_gas']}, dump {DUMP})")
        ax.set_ylabel("densidad KDE")
        ax.legend(fontsize=7, loc="upper right")
    ejes[-1].set_xlabel("distancia al vecino más cercano (unidades del dump)")
    fig.tight_layout()
    fig.savefig(salida, dpi=140)
    print(f"Figura guardada en {salida}")


def main() -> None:
    rng = np.random.default_rng(SEMILLA_BARAJADO)
    resultados = []
    for nombre in CORRIDAS:
        carpeta = BATERIA / nombre
        con_barajado = nombre in ("ic_real", "ic_null1")
        print(f"Procesando {nombre} ...")
        r = analizar_corrida(nombre, carpeta, rng, con_barajado_ejes=con_barajado)
        resultados.append(r)
        v1 = r["nn1"]["vacio"]
        print(f"  n_gas={r['n_gas']}  espaciamiento={r['espaciamiento']:.4g}  "
              f"linking_length_proyecto={r['linking_length_proyecto']:.4g}")
        print(f"  NN1: vacío notable={v1['hay_vacio_notable']}  "
              f"({v1['n_minimos_locales']} mínimos locales)  mejor_valle={v1['mejor_valle']}")
        if con_barajado:
            vb = r["nn1_barajado_ejes"]["vacio"]
            print(f"  NN1 barajado-ejes: vacío notable={vb['hay_vacio_notable']}  "
                  f"mejor_valle={vb['mejor_valle']}")

    salida_json = AQUI / "cn4_resultados.json"
    with open(salida_json, "w") as f:
        json.dump(resultados, f, indent=2)
    print(f"Resultados crudos guardados en {salida_json}")

    graficar(resultados, AQUI / "cn4_histogramas_nn.png")


if __name__ == "__main__":
    main()
