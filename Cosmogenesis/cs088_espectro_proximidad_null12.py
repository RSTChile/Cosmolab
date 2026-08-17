"""
cs088_espectro_proximidad_null12.py — el diagnóstico espectral de CS084/CS085, pero sobre un grafo de
PROXIMIDAD GEOMÉTRICA (k-NN, k=4) construido directo sobre posiciones de partículas, para poder incluir
por fin a NULL-1 y NULL-2 en la comparación.

POR QUÉ ESTE SCRIPT (encargo de Alexis, 9-ago-2026): CS085 aplicó el diagnóstico espectral (λ_max, λ2,
dimensión espectral vía núcleo de calor — funciones reusadas tal cual de `cs084_espectro_laplaciano.py`)
a la malla CAUSAL de REAL/NULL-3/RANDOM y encontró una escala limpia con la estructura preservada
(λ2≈0.020-0.022 REAL, 0.077-0.158 NULL-3, 0.234-0.314 RANDOM). Pero NULL-1 (mismo radio, ángulo al azar)
y NULL-2 (Zel'dovich) quedaron explícitamente AFUERA de esa comparación porque no tienen grafo de
vecindad de fondo — son construcciones puramente estadísticas sobre posiciones, nunca pasan por
`malla_causal_atomos`. Ya sabemos (por las baterías NULL1/NULL2 con Phantom real) que ambos dan 0/8
sumideros — nunca colapsan. La pregunta de este script: si les construimos un grafo NUEVO (no causal,
sino de PROXIMIDAD: cada partícula conectada a sus k=4 vecinos más cercanos en el espacio, el mismo k que
usa la malla causal, para comparar grado con grado) sobre sus posiciones iniciales, ¿el mismo instrumento
espectral separa a REAL de NULL-1/NULL-2 igual de limpio que separó REAL de NULL-3/RANDOM con la malla
causal? Si SÍ separa: la firma espectral podría estar capturando algo de la densidad/posición geométrica
en sí, no sólo de la relación causal. Si NO separa (proximidad geométrica se ve igual en los tres): la
firma espectral de CS085 sería específica de la malla CAUSAL, no un artefacto de "cualquier grafo sobre
estas posiciones".

QUÉ SE REUSA, SIN TOCAR NINGÚN ARCHIVO CONGELADO (sólo import/lectura):
  - `leer_volcado_phantom.py` (`leer_dump`): lee el volcado binario INICIAL (`cosmog_00000`, antes de que
    Phantom corra ni un paso de gravedad) de cada corrida ya existente en disco — posiciones x,y,z de
    2000 partículas de gas, condición inicial pura, no resultado ya colapsado.
  - `cs084_espectro_laplaciano.py` (vía `cs085_espectro_jerarquia_cs073.py`, que ya las importa): las
    funciones de CÁLCULO del diagnóstico — `dimension_espectral` (núcleo de calor), `unfolding_local` +
    `estadisticas_espaciado` (espaciado de niveles), `T_GRID`.
  - `cs085_espectro_jerarquia_cs073.py`: se importan directo (sin ejecutar su `main()`, que sólo corre
    bajo `if __name__ == "__main__"`) sus funciones `laplaciano_denso_desde_adj` (L=D-A denso + eigh +
    nº de componentes) y `diagnostico_fila` (arma la fila de tabla con los 3 diagnósticos) — se reusa
    la MISMA matemática y el MISMO formato de columnas para poder comparar directo con la tabla de CS085.
  - NINGÚN archivo de batería (`bateria_n2000/`, `bateria_null1_n2000/`, `bateria_null2_n2000/`,
    `bateria_real_extra_n2000/`) se toca — sólo se leen los volcados binarios `cosmog_00000` ya en disco.

QUÉ ES NUEVO ACÁ (no existía en CS084/CS085): la construcción del grafo de proximidad geométrica
(`construir_grafo_knn`) — un k-NN simple vía `scipy.spatial.cKDTree`, simetrizado por UNIÓN (si j está
entre los k vecinos más cercanos de i, o i entre los de j, hay arista i-j; la relación "vecino más
cercano" no es simétrica en general, así que el grado promedio resultante queda algo por encima de k=4,
se reporta el nº de aristas real de cada grafo, no se fuerza a que coincida). Este grafo es de naturaleza
DISTINTA a la malla causal de CS073/CS085 (que construye aristas por un criterio de "quién pudo influir
causalmente sobre quién" en dos fases, no por cercanía euclídea pura) — la comparación entre ambos es
SUGESTIVA, no una réplica exacta del mismo objeto.

SEMILLAS: NULL-1 y NULL-2 SÍ tienen múltiples corridas independientes ya en disco (8 cada una, con
distinta semilla de generación de condición inicial) — se usan 5 de cada (s1..s5 para NULL-1,
s401..s405 para NULL-2), variabilidad GENUINA (no remuestreo del mismo dump). Para REAL sólo hay UNA
corrida canónica en `bateria_n2000/ic_real`, pero `bateria_real_extra_n2000/` tiene 5 corridas REALES
adicionales con condición inicial generada con semilla distinta (`ic_real_s301..s305` — se verificó que
las posiciones NO son las mismas que la canónica, no es un remuestreo cosmético) — se usan como las 5
semillas REAL, más la canónica como sexta fila de referencia.

No se declara cierre ni veredicto — sólo se reportan números. La lectura final es de Alexis.

Codea/ejecuta: CC (Claude).
"""
from __future__ import annotations
import os, sys, time, csv
import numpy as np
from scipy.spatial import cKDTree

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)

from leer_volcado_phantom import leer_dump                                    # sólo lectura de volcados
from cs085_espectro_jerarquia_cs073 import laplaciano_denso_desde_adj, diagnostico_fila  # sólo import

RUTA_BASE = "/Users/alexis/phantom_cs073"
K_PROXIMIDAD = 4          # mismo grado nominal que usa la malla causal real (k=4), para comparar parejo
OUT_CSV = os.path.join(_HERE, "cs088_espectro_proximidad_null12.csv")

# -------------------- casos: (carpeta de la corrida, etiqueta de semilla) --------------------
CASOS_REAL = [(f"{RUTA_BASE}/bateria_n2000/ic_real", "canónica")] + [
    (f"{RUTA_BASE}/bateria_real_extra_n2000/ic_real_s{300+i}", f"s{300+i}") for i in range(1, 6)
]
CASOS_NULL1 = [(f"{RUTA_BASE}/bateria_null1_n2000/ic_null1_s{i}", f"s{i}") for i in range(1, 6)]
CASOS_NULL2 = [(f"{RUTA_BASE}/bateria_null2_n2000/ic_null2_s{400+i}", f"s{400+i}") for i in range(1, 6)]


# ============================ construcción: grafo k-NN geométrico desde posiciones ============================
def construir_grafo_knn(pos, k=K_PROXIMIDAD):
    """Grafo de proximidad: cada partícula i conectada a sus k vecinos más cercanos en 3D (posiciones
    x,y,z del volcado inicial de Phantom, ANTES de que la gravedad mueva nada). Simetrizado por UNIÓN
    (ver docstring del módulo). Devuelve (adj: dict[int,set[int]], n)."""
    n = len(pos)
    tree = cKDTree(pos)
    _dist, idx = tree.query(pos, k=k + 1)  # columna 0 es el propio punto (distancia 0), se descarta
    adj = {i: set() for i in range(n)}
    for i in range(n):
        for j in idx[i, 1:]:
            j = int(j)
            if j == i:
                continue
            adj[i].add(j)
            adj[j].add(i)
    return adj, n


def leer_posiciones_iniciales(carpeta):
    """Lee `cosmog_00000` (primer volcado, condición inicial pura, antes del primer paso de gravedad)
    de una carpeta de corrida de Phantom y devuelve las posiciones x,y,z como array (n,3)."""
    gas, _sinks = leer_dump(os.path.join(carpeta, "cosmog_00000"))
    return gas[["x", "y", "z"]].to_numpy()


def procesar_grupo(grupo, casos):
    filas = []
    for carpeta, etiqueta_semilla in casos:
        t0 = time.time()
        pos = leer_posiciones_iniciales(carpeta)
        adj, n = construir_grafo_knn(pos, k=K_PROXIMIDAD)
        fila = diagnostico_fila(grupo, f"{grupo} proximidad k={K_PROXIMIDAD} (semilla {etiqueta_semilla})",
                                 etiqueta_semilla, adj, n, extra=dict(carpeta=os.path.basename(carpeta)))
        fila.pop("_eigvals", None)
        filas.append(fila)
        print(f"  [{grupo}] semilla={etiqueta_semilla}: n={n} n_aristas={fila['n_aristas']} "
              f"lambda2={fila['lambda2']} lambda_max={fila['lambda_max']} "
              f"({time.time()-t0:.2f}s)", flush=True)
    return filas


def main():
    t_inicio = time.time()
    print("=" * 100, flush=True)
    print("CS088 — espectro del laplaciano sobre grafos de PROXIMIDAD (k-NN, k=4) — REAL / NULL-1 / NULL-2",
          flush=True)
    print("=" * 100, flush=True)

    filas = []
    print("\n[REAL] posiciones iniciales, k-NN k=4...", flush=True)
    filas += procesar_grupo("REAL", CASOS_REAL)
    print("\n[NULL-1] posiciones iniciales (radio exacto, ángulo al azar), k-NN k=4...", flush=True)
    filas += procesar_grupo("NULL-1", CASOS_NULL1)
    print("\n[NULL-2] posiciones iniciales (Zel'dovich), k-NN k=4...", flush=True)
    filas += procesar_grupo("NULL-2", CASOS_NULL2)

    campos = list(filas[0].keys())
    for f in filas[1:]:
        for k in f:
            if k not in campos:
                campos.append(k)
    with open(OUT_CSV, "w", newline="") as fcsv:
        wr = csv.DictWriter(fcsv, fieldnames=campos, restval="")
        wr.writeheader()
        for f in filas:
            wr.writerow(f)
    print(f"\nCSV -> {OUT_CSV}", flush=True)
    print(f"COMPLETO en {time.time()-t_inicio:.1f}s", flush=True)


if __name__ == "__main__":
    main()
