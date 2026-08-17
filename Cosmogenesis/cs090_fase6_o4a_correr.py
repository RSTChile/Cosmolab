#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cs090_fase6_o4a_correr.py — O4-A: réplica con un motor gravitacional INDEPENDIENTE
==================================================================================

QUÉ HACE (en simple)
--------------------
Toma 10 pares "Clase III vs Clase I" de la Fase V-B (los que siguen siendo contraste
válido después de la corrección de diámetro), agarra EXACTAMENTE las mismas condiciones
iniciales que se le dieron a Phantom (`cosmogenesis_ic.txt`, sin regenerarlas), y las
corre con el motor de gravedad pura de `cs090_fase6_o4a_nbody.py`.

Después mide un observable análogo — "cuánta masa terminó apelotonada en las regiones
más densas" — y pregunta lo único que importa acá:
    ¿el ganador de cada par es el mismo en los dos motores?

NO se busca reproducir el número de Phantom. Los dos motores hacen física distinta
(Phantom: fluido + viscosidad + sumideros que acretan; acá: puntos que sólo se atraen).
La comparación válida es de ORDEN, no de valor absoluto.

DISEÑO DE LA MUESTRA (declarado ANTES de mirar resultados)
----------------------------------------------------------
De los 37 pares válidos se eligen 10, en dos estratos de 5, por un criterio mecánico
(no por conveniencia):
  * estrato SEÑAL FUERTE: los 5 pares con mayor |Δfracción| en Phantom (0.025 a 0.034)
  * estrato EMPATE      : los 5 pares con menor |Δfracción| en Phantom (0.001 a 0.002)
La razón de partir en estratos: en los empates ningún método puede "acertar" el orden,
porque el orden de Phantom ahí es prácticamente ruido. Mezclarlos y reportar un único
porcentaje escondería eso. Separados, la predicción es clara: si el efecto es real,
el estrato fuerte debería concordar y el de empate debería salir cerca de cara-o-cruz.

OBSERVABLE ANÁLOGO (declarado antes de mirar resultados)
--------------------------------------------------------
Principal: friends-of-friends con longitud de enlace ell = 1.0 y tamaño mínimo 5
partículas, sobre el estado final (t = 0.5, el mismo tiempo final de Phantom);
suavizado gravitatorio eps = 0.6 (= r_crit de Phantom, el radio que separa "un
sumidero" de "dos"). Se reporta también una grilla de (eps, ell, n_min) como control
de robustez, y una variante por densidad local de k-ésimo vecino.

Referencia de escala: la separación media entre partículas es 97.6/2000^(1/3) ≈ 7.8,
así que enlazar a 1.0 sólo agrupa material que de verdad colapsó.

USO
---
    ./venv/bin/python cs090_fase6_o4a_correr.py convergencia   # test de paso de tiempo
    ./venv/bin/python cs090_fase6_o4a_correr.py correr [n_procesos]
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from multiprocessing import Pool

import cs090_fase6_o4a_nbody as nb

AQUI = os.path.dirname(os.path.abspath(__file__))
T_FINAL = 0.5          # mismo tmax que Phantom (cosmog.in)
DT = 2.0e-3            # paso fijo; elegido POR el test de convergencia (modo
                       # `convergencia`): el observable sale idéntico con 4e-3, 2e-3,
                       # 1e-3 y 5e-4, y la deriva de energía ya es ~1e-6 en 2e-3.
EPS_PRINCIPAL = 0.6    # = r_crit de Phantom
ELL_PRINCIPAL = 1.0
NMIN_PRINCIPAL = 5

# --- los 10 pares elegidos (rule_I, rule_III, d_frac de Phantom, estrato) -----------
PARES = [
    # estrato SEÑAL FUERTE (|d_frac| mayor)
    ("PAR_piloto_B_r1_r17",                         "A2-B0-C2-r1",          "A2-B0-C2-r17",           0.0340, "fuerte"),
    ("A2-B0-C2-batch4-r23_vs_A2-B0-C2-batch4-r10",  "A2-B0-C2-batch4-r23",  "A2-B0-C2-batch4-r10",    0.0325, "fuerte"),
    ("A2-B0-C2-batch3-r120_vs_A2-B0-C2-batch3-r111","A2-B0-C2-batch3-r120", "A2-B0-C2-batch3-r111",   0.0295, "fuerte"),
    ("A2-B0-C2-batch3-r104_vs_A2-B0-C2-batch3-r60", "A2-B0-C2-batch3-r104", "A2-B0-C2-batch3-r60",    0.0270, "fuerte"),
    ("PAR_v2_G_r12v1fix_r19",                       "A2-B0-C2-r12v1fix",    "A2-B0-C2-r19",           0.0250, "fuerte"),
    # estrato EMPATE (|d_frac| menor)
    ("A2-B0-C2-batch3-r59_vs_A2-B0-C2-batch3-r58",  "A2-B0-C2-batch3-r59",  "A2-B0-C2-batch3-r58",   -0.0010, "empate"),
    ("A2-B0-C2-batch4-r39_vs_A2-B0-C2-batch4-r26",  "A2-B0-C2-batch4-r39",  "A2-B0-C2-batch4-r26",    0.0010, "empate"),
    ("PAR_piloto_C_r6_r14",                         "A2-B0-C2-r6",          "A2-B0-C2-r14",          -0.0015, "empate"),
    ("A2-B0-C2-batch4-r38_vs_A2-B0-C2-batch4-r72",  "A2-B0-C2-batch4-r38",  "A2-B0-C2-batch4-r72",    0.0020, "empate"),
    ("PAR_v2_H_r9_r39",                             "A2-B0-C2-r9",          "A2-B0-C2-r39",          -0.0020, "empate"),
]

BASES = [
    "/Users/alexis/phantom_cs073/bateria_fase5b_a2b0c2_piloto",
    "/Users/alexis/phantom_cs073/bateria_fase5b_a2b0c2_escala_v2",
    "/Users/alexis/phantom_cs073/bateria_fase5b_a2b0c2_escala_v3",
    "/Users/alexis/phantom_cs073/bateria_fase5b_a2b0c2_escala_v4",
]


def ruta_ic(regla):
    """Ubica el `cosmogenesis_ic.txt` de una regla en las 4 tandas de Fase V-B.
    Reusar el archivo (en vez de regenerarlo) garantiza que los dos motores reciben
    byte a byte el mismo input — que es justamente lo que hace válida la comparación."""
    hits = []
    for b in BASES:
        for suf in ("_I", "_II", "_III", "_IV"):
            p = os.path.join(b, regla + suf, "cosmogenesis_ic.txt")
            if os.path.exists(p):
                hits.append(p)
    assert len(hits) == 1, f"{regla}: {len(hits)} candidatos {hits}"
    return hits[0]


GRILLA_ELL = [0.6, 1.0, 2.0]
GRILLA_NMIN = [3, 5, 10]


def medir_observables(pos, m_part):
    """Calcula el observable análogo en toda la grilla de robustez + variante kNN."""
    out = {}
    for ell in GRILLA_ELL:
        for nmin in GRILLA_NMIN:
            f, _, ngr = nb.observable_fof(pos, m_part, ell, nmin)
            out[f"fof_ell{ell}_nmin{nmin}"] = f
            out[f"ngrupos_ell{ell}_nmin{nmin}"] = ngr
    n = pos.shape[0]
    lado = 97.6
    rho_media = n * m_part / lado ** 3
    for mult in (100.0, 1000.0):
        f, _ = nb.observable_densidad_knn(pos, m_part, k=8, rho_umbral=mult * rho_media)
        out[f"knn8_frac_rho_mayor_{int(mult)}x"] = f
    return out


def correr_una(args):
    """Integra una regla y devuelve una fila con diagnóstico + observables."""
    regla, rol, dt, eps = args
    t0 = time.time()
    ruta = ruta_ic(regla)
    pos0, vel0, m_part, cab = nb.leer_ic_cosmogenesis(ruta)
    obs_ini = medir_observables(pos0, m_part)
    posf, velf, diag = nb.integrar_leapfrog(pos0, vel0, m_part, eps, T_FINAL, dt)
    obs_fin = medir_observables(posf, m_part)
    fila = dict(regla=regla, rol=rol, ruta_ic=ruta, dt=dt, eps=eps, t_final=T_FINAL,
                nsteps=diag["nsteps"],
                E0=diag["E0"], Ef=diag["Ef"], deriva_energia_rel=diag["deriva_rel"],
                Ekin0=diag["Ekin0"], Ekinf=diag["Ekinf"],
                virial_final=diag["Ekinf"] / abs(diag["Epotf"]),
                segundos=round(time.time() - t0, 1))
    for k, v in obs_ini.items():
        fila["ini_" + k] = v
    for k, v in obs_fin.items():
        fila["fin_" + k] = v
    print(f"  [ok] {regla:26s} {rol:3s} dE/E={diag['deriva_rel']:+.2e} "
          f"fof={obs_fin[f'fof_ell{ELL_PRINCIPAL}_nmin{NMIN_PRINCIPAL}']:.4f} "
          f"({fila['segundos']}s)", flush=True)
    return fila


def modo_convergencia():
    """Test de convergencia en paso de tiempo sobre 2 sistemas reales del proyecto.

    Se corre la MISMA condición inicial con dt, dt/2 y dt/4 y se compara (a) la deriva
    de energía y (b) el observable. Si el observable ya no se mueve al partir el paso,
    el paso alcanza. Es la prueba que decide DT antes de gastar cómputo en los 10 pares.
    """
    reglas = ["A2-B0-C2-r1", "A2-B0-C2-r17"]
    tareas = [(r, "?", dt, EPS_PRINCIPAL) for r in reglas for dt in (4e-3, 2e-3, 1e-3, 5e-4)]
    with Pool(min(8, len(tareas))) as p:
        filas = p.map(correr_una, tareas)
    df = pd.DataFrame(filas)
    col = f"fin_fof_ell{ELL_PRINCIPAL}_nmin{NMIN_PRINCIPAL}"
    print(df[["regla", "dt", "nsteps", "deriva_energia_rel", "virial_final", col, "segundos"]].to_string())
    df.to_csv(os.path.join(AQUI, "cs090_fase6_o4a_convergencia_dt.csv"), index=False)
    return df


def modo_correr(nproc=5):
    tareas = []
    vistos = set()
    for _, rI, rIII, _, _ in PARES:
        for r, rol in ((rI, "I"), (rIII, "III")):
            if r not in vistos:
                vistos.add(r)
                tareas.append((r, rol, DT, EPS_PRINCIPAL))
    print(f"{len(tareas)} reglas únicas, dt={DT}, eps={EPS_PRINCIPAL}, "
          f"{int(T_FINAL/DT)} pasos c/u, {nproc} procesos", flush=True)
    t0 = time.time()
    with Pool(nproc) as p:
        filas = p.map(correr_una, tareas)
    df = pd.DataFrame(filas)
    salida = os.path.join(AQUI, "cs090_fase6_o4a_corridas_nbody.csv")
    df.to_csv(salida, index=False)
    print(f"escrito {salida} en {round(time.time()-t0,1)} s")
    return df


if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "correr"
    if modo == "convergencia":
        modo_convergencia()
    else:
        modo_correr(int(sys.argv[2]) if len(sys.argv) > 2 else 5)
