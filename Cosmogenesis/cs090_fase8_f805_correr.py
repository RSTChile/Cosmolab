#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cs090_fase8_f805_correr.py — F8-05: ¿el +13.8% de F7-03 sobrevive a cambiar de motor?
=====================================================================================

QUÉ HACE (en simple)
--------------------
F7-03 encontró que, con la misma cantidad de nudos, el mismo número de alambres en cada
nudo y **exactamente el mismo número de triangulitos**, la maqueta que los tiene apilados
compartiendo varilla (`solap`) junta un 13.8% más de arena que la que los tiene sueltos
(`disj`) — en los 12 grafos, sin excepción.

Ese resultado salió de UN motor: Phantom. Esta tarea lo vuelve a preguntar con un motor
distinto, escrito desde cero y con otra física (`cs090_fase6_o4a_nbody.py`, validado en
la Fase VI): gravedad pura, suma directa, sin presión, sin viscosidad, sin sumideros.

Analogía: si dos relojes construidos por relojeros distintos, con mecanismos distintos,
dicen que la carrera la ganó el mismo corredor, la victoria es del corredor y no del
reloj. Los tiempos absolutos no van a coincidir — el orden sí debería.

TRES NÚMEROS, NO UNO
--------------------
El informe de O4-A dejó una advertencia que acá se toma en serio: el observable medido
sobre las **condiciones iniciales sin integrar nada** ya predecía el resultado de Phantom
con r = +0.98. Por eso este script mide SIEMPRE las dos cosas:

  1. el observable en t = 0     → "geometría de partida" (cero dinámica)
  2. el observable en t = 0.5   → "geometría de partida + gravedad pura"

y el análisis los compara contra el tercero:

  3. la fracción de masa en sumideros que midió Phantom (ya existente, no se re-corre).

Si (1) ya separa `solap` de `disj` con la misma fuerza que (2) y (3), entonces lo que se
está midiendo es **cómo nació apelotonada la nube**, no lo que hizo la gravedad después.

LO QUE SE REUSA TAL CUAL (nada se regenera, nada se modifica)
------------------------------------------------------------
* Las condiciones iniciales: los `cosmogenesis_ic.txt` que ya recibió Phantom, dentro de
  `/Users/alexis/phantom_cs073/bateria_fase7_f703_organizacion/`. Usar el MISMO archivo
  byte a byte es todo el punto: los dos motores tienen que partir del mismo lugar.
* El integrador: `cs090_fase6_o4a_nbody.py`, importado, jamás tocado.
* Los parámetros ya validados allá: dt = 2e-3 (250 pasos), t_final = 0.5, eps = 0.6.

USO
---
    ./venv/bin/python cs090_fase8_f805_correr.py [n_procesos]
"""

import os
import sys
import json
import time
import hashlib
import numpy as np
import pandas as pd
from multiprocessing import Pool

import cs090_fase6_o4a_nbody as nb          # SÓLO se importa; no se modifica

AQUI = os.path.dirname(os.path.abspath(__file__))
BATERIA = "/Users/alexis/phantom_cs073/bateria_fase7_f703_organizacion"

# --- parámetros heredados de O4-A, ya validados allá (§2.1 de FASE6_O4A) -------------
T_FINAL = 0.5          # el mismo tmax de Phantom en esta línea
DT = 2.0e-3            # 250 pasos; el observable no se movía ni un dígito entre 4e-3 y 5e-4
EPS_PRINCIPAL = 0.6    # = r_crit de Phantom (el radio que separa "un sumidero" de "dos")

# --- observable análogo: friends-of-friends -----------------------------------------
# Principal DECLARADO ANTES de mirar nada, idéntico al de O4-A: ell = 1.0, n_min = 5.
ELL_PRINCIPAL = 1.0
NMIN_PRINCIPAL = 5
# Grilla de robustez. Se agregan ell = 0.3 y 0.45 (no estaban en O4-A) porque estas
# condiciones iniciales nacen MÁS apelotonadas que las de Fase V-B —el FoF a ell = 1.0
# ya agrupa >50% de la masa en t=0— y un observable saturado pierde poder de resolución.
# Se declaran como CONTROL DE ROBUSTEZ, no como observable principal.
GRILLA_ELL = [0.3, 0.45, 0.6, 1.0, 2.0]
GRILLA_NMIN = [3, 5, 10]

BRAZOS = ("solap", "disj")


# ---------------------------------------------------------------------------
# 1. LOCALIZAR LAS CONDICIONES INICIALES DE F7-03 (sin regenerar ninguna)
# ---------------------------------------------------------------------------
def inventario_f703():
    """Recorre la batería de F7-03 y devuelve las 12 parejas (solap, disj).

    Cada carpeta se verifica contra su propio `meta_regla.json`: que la tarea sea la de
    F7-03, que el brazo y el (rule_id, seed) del meta coincidan con el nombre de la
    carpeta, y que el meta declare grados idénticos al grafo original. Nunca se confía
    en el nombre de la carpeta solo — es la lección del bug de colisión de nombres
    documentado en FASE6_O3B §2.1.
    """
    grafos = {}
    for nombre in sorted(os.listdir(BATERIA)):
        if "_f703_" not in nombre:
            continue
        brazo = nombre.rsplit("_f703_", 1)[1]
        if brazo not in BRAZOS:
            continue
        carpeta = os.path.join(BATERIA, nombre)
        meta = json.load(open(os.path.join(carpeta, "meta_regla.json")))
        assert meta["tarea"] == "FASE7_F703_organizacion_triangulos", nombre
        assert meta["brazo"] == brazo, f"{nombre}: meta dice brazo={meta['brazo']}"
        esperado = f"{meta['rule_id']}_s{meta['seed']}_f703_{brazo}"
        assert esperado == nombre, f"{nombre} != {esperado}"
        assert meta["grados_identicos_al_original"] is True, nombre
        assert os.path.abspath(meta["carpeta"]) == os.path.abspath(carpeta), nombre
        ic = os.path.join(carpeta, "cosmogenesis_ic.txt")
        assert os.path.exists(ic), f"falta IC en {nombre}"
        clave = (meta["rule_id"], int(meta["seed"]))
        grafos.setdefault(clave, {})[brazo] = dict(carpeta=carpeta, ic=ic, meta=meta)
    # sólo los grafos que tienen los dos brazos
    completos = {k: v for k, v in grafos.items() if set(v) == set(BRAZOS)}
    assert len(completos) == 12, f"esperaba 12 grafos, encontré {len(completos)}"
    return completos


def md5(ruta):
    h = hashlib.md5()
    with open(ruta, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 2. EL OBSERVABLE ANÁLOGO, EN TODA LA GRILLA
# ---------------------------------------------------------------------------
def medir_observables(pos, m_part):
    """Fracción de masa en regiones densas, medida de varias maneras.

    FoF = "cada partícula se da la mano con toda vecina a menos de ell; los grupos son
    las cadenas de manos; el observable es cuánta masa quedó en grupos de al menos
    n_min miembros". Más la variante por densidad local del 8.º vecino, que no depende
    de ninguna longitud de enlace.
    """
    out = {}
    for ell in GRILLA_ELL:
        for nmin in GRILLA_NMIN:
            f, _, ngr = nb.observable_fof(pos, m_part, ell, nmin)
            out[f"fof_ell{ell}_nmin{nmin}"] = f
            out[f"ngrupos_ell{ell}_nmin{nmin}"] = ngr
    n = pos.shape[0]
    lado = 97.6            # lado de la caja de estas corridas (verificado sobre las IC)
    rho_media = n * m_part / lado ** 3
    for mult in (100.0, 1000.0):
        f, _ = nb.observable_densidad_knn(pos, m_part, k=8, rho_umbral=mult * rho_media)
        out[f"knn8_frac_rho_mayor_{int(mult)}x"] = f
    return out


# ---------------------------------------------------------------------------
# 3. UNA CORRIDA
# ---------------------------------------------------------------------------
def correr_una(args):
    """Integra un brazo de un grafo y devuelve diagnóstico + observables en t=0 y t=0.5."""
    rule_id, seed, brazo, ruta_ic, dt, eps = args
    t0 = time.time()
    pos0, vel0, m_part, cab = nb.leer_ic_cosmogenesis(ruta_ic)
    assert cab["npart"] == 2000, f"{rule_id}/{brazo}: npart={cab['npart']}"
    assert abs(m_part - 9.4) < 1e-9, f"{rule_id}/{brazo}: m_part={m_part}"

    obs_ini = medir_observables(pos0, m_part)
    posf, velf, diag = nb.integrar_leapfrog(pos0, vel0, m_part, eps, T_FINAL, dt)
    obs_fin = medir_observables(posf, m_part)

    # control barato: el momento lineal se conserva exacto en gravedad aislada
    p0 = m_part * vel0.sum(0)
    pf = m_part * velf.sum(0)
    err_p = float(np.max(np.abs(pf - p0)) / max(np.max(np.abs(p0)), 1e-30))

    fila = dict(rule_id=rule_id, seed=seed, brazo=brazo, ruta_ic=ruta_ic,
                md5_ic=md5(ruta_ic), npart=cab["npart"], m_part=m_part,
                dt=dt, eps=eps, t_final=T_FINAL, nsteps=diag["nsteps"],
                E0=diag["E0"], Ef=diag["Ef"], deriva_energia_rel=diag["deriva_rel"],
                Ekin0=diag["Ekin0"], Ekinf=diag["Ekinf"],
                virial_final=diag["Ekinf"] / abs(diag["Epotf"]),
                error_rel_momento=err_p,
                segundos=round(time.time() - t0, 1))
    for k, v in obs_ini.items():
        fila["ini_" + k] = v
    for k, v in obs_fin.items():
        fila["fin_" + k] = v
    print(f"  [ok] {rule_id:24s} {brazo:6s} dE/E={diag['deriva_rel']:+.2e} "
          f"fof_ini={obs_ini[f'fof_ell{ELL_PRINCIPAL}_nmin{NMIN_PRINCIPAL}']:.4f} "
          f"fof_fin={obs_fin[f'fof_ell{ELL_PRINCIPAL}_nmin{NMIN_PRINCIPAL}']:.4f} "
          f"({fila['segundos']}s)", flush=True)
    return fila


def main(nproc=6):
    grafos = inventario_f703()
    tareas = []
    md5s = {}
    for (rule_id, seed), d in sorted(grafos.items()):
        for brazo in BRAZOS:
            ic = d[brazo]["ic"]
            md5s[(rule_id, seed, brazo)] = md5(ic)
            tareas.append((rule_id, seed, brazo, ic, DT, EPS_PRINCIPAL))
    # verificación: las 24 condiciones iniciales tienen que ser 24 archivos distintos
    assert len(set(md5s.values())) == 24, "hay condiciones iniciales repetidas"
    print(f"{len(grafos)} grafos × {len(BRAZOS)} brazos = {len(tareas)} corridas; "
          f"24 md5 distintos OK", flush=True)
    print(f"dt={DT} ({int(T_FINAL/DT)} pasos), eps={EPS_PRINCIPAL}, t_final={T_FINAL}, "
          f"{nproc} procesos", flush=True)
    t0 = time.time()
    with Pool(nproc) as p:
        filas = p.map(correr_una, tareas)
    df = pd.DataFrame(filas)
    salida = os.path.join(AQUI, "cs090_fase8_f805_corridas_nbody.csv")
    df.to_csv(salida, index=False)
    print(f"\nescrito {salida} — {len(df)} filas en {round(time.time()-t0,1)} s")
    print(f"deriva de energía: min {df.deriva_energia_rel.min():.2e}  "
          f"max {df.deriva_energia_rel.max():.2e}")
    print(f"error relativo de momento: max {df.error_rel_momento.max():.2e}")
    return df


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
