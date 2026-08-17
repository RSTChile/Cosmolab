#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cs090_fase6_o3a_geometria_ic.py — FASE VI, tarea O3-A (anexo): ¿lo que escala con la resolución es la
GEOMETRÍA INICIAL o la FÍSICA DEL COLAPSO?
=======================================================================================================

POR QUÉ EXISTE ESTE ANEXO
-------------------------
Este archivo NO estaba en el plan original de O3-A. Se agrega porque, DESPUÉS de que O3-A se lanzara,
`FASE6_O4A_solver_independiente_CS.md` midió algo que cambia cómo hay que leer cualquier resultado de
Phantom en esta línea: la fracción de masa en sumideros que devuelve Phantom está predicha con r = +0.98
por lo apelotonada que ya nacía la nube ANTES de integrar un solo paso. O sea: buena parte de "el
resultado del colapso" ya está escrito en la condición inicial.

Entonces, si O3-A encuentra que la ventaja de la Clase III sobre la Clase I CRECE al subir N, quedan dos
lecturas posibles y hay que separarlas con un número, no con una opinión:

  (a) crece la GEOMETRÍA: a más partículas, el layout de resortes de la regla Clase III nace más
      apelotonado que el de la Clase I, y Phantom simplemente refleja eso;
  (b) crece la FÍSICA: las dos nubes nacen igual de apelotonadas, y es el colapso gravitatorio el que
      amplifica más la de la Clase III cuando hay más resolución.

Este script mide (a) directamente: aplica un friends-of-friends a las CONDICIONES INICIALES (t = 0, cero
dinámica) de las mismas reglas y los mismos pares que corrió O3-A, a N = 2000 y a N = 4000, y calcula el
mismo tipo de diferencia Δ = III − I. Después se compara la serie Δ_geometría(N) con la serie
Δ_masa_final(N) de Phantom. Si las dos crecen igual, lo que escala es la geometría; si Δ_geometría se
queda quieto y Δ_masa crece, lo que escala es el colapso.

EN SIMPLE, CON ANALOGÍA
-----------------------
Phantom es un horno y las condiciones iniciales son la masa cruda. Dos panaderos (Clase III y Clase I)
amasan con la misma harina y el mismo horno. Si el pan de uno sale más compacto, puede ser porque su
horno lo hizo subir distinto (física) o porque su bollo YA estaba más apretado antes de entrar al horno
(geometría). Acá se pesa el bollo crudo, antes del horno, a dos niveles de "resolución del amasado", y se
mira si la diferencia entre los bollos crudos crece igual que la diferencia entre los panes horneados.

DECISIONES DE MEDICIÓN (y por qué)
----------------------------------
- **Longitud de enlace `ell` en unidades de la separación media entre partículas**, no en unidades
  absolutas. La caja tiene lado fijo (no escala con N) y la masa total también es fija, así que a N=4000
  las partículas están intrínsecamente más juntas que a N=2000: un `ell` absoluto fijo declararía
  "más apelotonado" a N=4000 por puro conteo. `ell = b · L / N^(1/3)` es la convención estándar de FoF
  cosmológico (b ≈ 0.2) y es la única que hace comparables dos resoluciones. Se reportan tres b
  (0.20, 0.30, 0.50) para no colgar la conclusión de un solo umbral.
- **Umbral de tamaño de grupo en MASA FÍSICA fija (47.0 = 5 partículas de N=2000)**, no en número de
  miembros: a N=4000 cada partícula pesa la mitad, así que "5 miembros" sería medio grumo. Con masa fija,
  el criterio de "grumo" es el mismo objeto físico en las dos resoluciones.
- **FoF no periódico.** Las IC tienen coordenadas en [0, 97.593] en las dos resoluciones (mismo lado
  fijo, misma escritura), y la comparación que importa es III − I DENTRO de cada N, donde cualquier
  efecto de borde afecta a los dos por igual.
- Se REUSA `fof_masa` de `cs090_fase6_o4a_observable_comun.py` (el observable de O4-A) tal cual, sin
  tocarlo: es exactamente la misma vara con la que se midió el r = +0.98, así que las dos tareas hablan
  el mismo idioma.

Uso:
    ./venv/bin/python cs090_fase6_o3a_geometria_ic.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import sys
sys.path.insert(0, "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis")
from cs090_fase6_o4a_observable_comun import fof_masa   # import puro: no se modifica

AQUI = Path("/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis")
SEL_JSON = AQUI / "cs090_fase6_o3a_pares_seleccionados.json"
CSV_SALIDA = AQUI / "cs090_fase6_o3a_geometria_ic.csv"
BASE_O3A = Path("/Users/alexis/phantom_cs073/bateria_fase6_o3a_resolucion")
LADO_ESCRITO = 97.593          # extensión real de las coordenadas escritas en cosmogenesis_ic.txt
MASA_MIN_FIJA = 47.0           # 5 partículas de N=2000; masa física, no número de miembros
BS = (0.20, 0.30, 0.50)        # longitudes de enlace, en unidades de separación media


def ruta_ic(rule_id: str, clase: str, N: int) -> Path | None:
    """La IC de N=2000 vive en las carpetas congeladas de Fase V-B; la de N>2000, en las de O3-A."""
    if N == 2000:
        c = glob.glob(f"/Users/alexis/phantom_cs073/bateria_fase5b_a2b0c2_*/{rule_id}_{clase}/cosmogenesis_ic.txt")
        return Path(c[0]) if c else None
    p = BASE_O3A / f"N{N}" / f"{rule_id}_{clase}" / "cosmogenesis_ic.txt"
    return p if p.exists() else None


def medir(rule_id: str, clase: str, N: int) -> dict | None:
    p = ruta_ic(rule_id, clase, N)
    if p is None:
        return None
    d = np.loadtxt(p, skiprows=2)
    pos = d[:, :3]
    n = len(pos)
    masas = np.full(n, 18800.0 / n)          # masa total fija = 18800 en las dos resoluciones
    sep_media = LADO_ESCRITO / n ** (1.0 / 3.0)
    fila = dict(rule_id=rule_id, clase=clase, N=N, npart=n, sep_media=sep_media)
    for b in BS:
        # fof_masa devuelve (fracción de masa en grupos grandes, nº de grupos grandes)
        frac, ngrupos = fof_masa(pos, masas, b * sep_media, MASA_MIN_FIJA)
        fila[f"fof_b{b:.2f}"] = float(frac)
        fila[f"ngrupos_b{b:.2f}"] = int(ngrupos)
    # medida sin umbral de FoF: dispersión de la densidad local (vecino k=8), otra vara de "apelotonado"
    from scipy.spatial import cKDTree
    dist, _ = cKDTree(pos).query(pos, k=9)
    r8 = dist[:, 8]
    fila["dens_k8_cv"] = float(np.std(1.0 / r8 ** 3) / np.mean(1.0 / r8 ** 3))
    return fila


def main() -> None:
    sel = json.loads(SEL_JSON.read_text())
    filas = []
    for s in sel:
        for N in (2000, 4000):
            for rid, clase in ((s["rid_I"], "I"), (s["rid_III"], "III")):
                f = medir(rid, clase, N)
                if f is not None:
                    f["par"] = s["par"]
                    filas.append(f)
    d = pd.DataFrame(filas)
    d.to_csv(CSV_SALIDA, index=False)
    print(f"[geometria] {len(d)} filas -> {CSV_SALIDA}")

    cols = [f"fof_b{b:.2f}" for b in BS] + ["dens_k8_cv"]
    print("\n[Δ geometría de la IC (III − I), por N; sólo pares con las 4 mediciones]")
    resumen = []
    for N in (2000, 4000):
        g = d[d.N == N]
        pares_ok = [p for p, gg in g.groupby("par") if set(gg.clase) == {"I", "III"}]
        fila = dict(N=N, n_pares=len(pares_ok))
        for c in cols:
            dif = [float(g[(g.par == p) & (g.clase == "III")][c].iloc[0]
                         - g[(g.par == p) & (g.clase == "I")][c].iloc[0]) for p in pares_ok]
            fila[f"d_{c}_media"] = float(np.mean(dif))
            fila[f"d_{c}_npos"] = int(sum(x > 0 for x in dif))
        resumen.append(fila)
    r = pd.DataFrame(resumen)
    pd.set_option("display.width", 220)
    print(r.to_string(index=False))

    # correlación par-a-par entre el Δ geométrico de la IC y el Δ de masa final de Phantom, a cada N
    pares_phantom = pd.read_csv(AQUI / "cs090_fase6_o3a_pares_por_N.csv")
    print("\n[correlación Δ_geometría(IC, t=0) ↔ Δ_masa_final(Phantom), por N]")
    for N in (2000, 4000):
        g = d[d.N == N]
        ph = pares_phantom[pares_phantom.N == N].set_index("par")["d_fraccion_masa"].to_dict()
        for c in cols:
            xs, ys = [], []
            for p, gg in g.groupby("par"):
                if set(gg.clase) != {"I", "III"} or p not in ph:
                    continue
                xs.append(float(gg[gg.clase == "III"][c].iloc[0] - gg[gg.clase == "I"][c].iloc[0]))
                ys.append(float(ph[p]))
            if len(xs) >= 4:
                print(f"  N={N} {c:14s} n={len(xs):2d} pearson={np.corrcoef(xs, ys)[0,1]:+.3f} "
                      f"spearman={stats.spearmanr(xs, ys).statistic:+.3f}")


if __name__ == "__main__":
    main()
