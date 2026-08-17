#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs072_posicion_en_espacio.py -- primer paso para darle posicion real a las particulas
de cs072_motor_23, usando la malla 3D que EstadoFisico (cs075_base_fisica.py) ya tiene.
Pedido del director (30-jul-2026): "usa ambos experimentos, y dale a las particulas
posicion".

QUE HACE Y QUE NO HACE, para no repetir el error de antes:
- NO toca cs072_motor_23.py, cs075_base_fisica.py ni cs072_proceso_holistico.py.
- NO cambia ninguna formula de fuerza todavia. La fisica de union (fuerte/EM/gravedad/
  debil) sigue siendo la MISMA, ya verificada, sin depender de distancia -- eso se
  decide DESPUES de ver este resultado, no antes.
- Solo AGREGA una capa: posicion determinista (sin azar, por indice -- mismo criterio
  que ya usa _catalogo() para color: color=i%3) de cada particula en la malla N=16 de
  EstadoFisico, y MIDE si los bariones/hidrogeno que el motor YA forma (sin saber nada
  de espacio) resultan de particulas que ademas estan cerca en esa malla, o dispersas
  sin ninguna relacion -- diagnostico puro, antes de decidir el proximo paso.

cuenta() de cs072_motor_23.py solo devuelve CONTEOS, no los indices de cada barion/
hidrogeno -- se re-deriva aca la MISMA logica de deteccion (color-trio ligado, proton+
electron ligado), verificado contra cuenta() para confirmar que da el mismo conteo antes
de confiar en los indices.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cs072_motor_23 import cuenta  # noqa: E402
from cs072_proceso_holistico import corre_holistico  # noqa: E402
from cs075_base_fisica import EstadoFisico  # noqa: E402


def asignar_posiciones(n_particulas, N_grid=16):
    """Posicion determinista por indice (sin azar): unravel_index sobre la malla N_grid^3,
    mismo criterio de 'invariante al indice, cero azar' que ya usa _catalogo() (color=i%3).
    Si n_particulas > N_grid^3, se envuelve (varias particulas por celda) -- se reporta,
    no se oculta."""
    capacidad = N_grid ** 3
    idx = np.arange(n_particulas) % capacidad
    pos = np.array(np.unravel_index(idx, (N_grid, N_grid, N_grid))).T  # (n_particulas, 3)
    return pos, bool(n_particulas > capacidad)


def _trios_con_indices(estado):
    """Re-deriva bariones/hidrogeno CON los indices de cada uno -- misma logica exacta
    que cuenta() en cs072_motor_23.py (verbatim, solo agrega el registro de indices)."""
    B, color, carga, es_anti, es_quark, viva, N = (
        estado["B"], estado["color"], estado["carga"], estado["es_anti"],
        estado["es_quark"], estado["viva"], estado["N"])
    b0 = max(float(B.sum(axis=1).mean()) / max(N - 1, 1), 1e-12)
    umbral = 1.5 * b0
    ligado = B > umbral

    def trios(mask):
        idxs = np.where(mask & (color >= 0) & (viva > 0.5))[0]
        usados = np.zeros(N, bool)
        out = []
        for i in idxs:
            if usados[i]:
                continue
            vec = [j for j in idxs if j != i and not usados[j] and color[j] != color[i] and ligado[i, j]]
            for j in vec:
                terc = [k for k in vec if k != j and color[k] != color[i] and color[k] != color[j]
                        and ligado[i, k] and ligado[j, k]]
                if terc:
                    k = terc[0]
                    out.append((i, j, k))
                    usados[[i, j, k]] = True
                    break
        return out

    bar = trios(~es_anti)
    protones = [t for t in bar if int(carga[t[0]]) + int(carga[t[1]]) + int(carga[t[2]]) == 3]
    elec = list(np.where((~es_anti) & (~es_quark) & (viva > 0.5))[0])
    hidrogenos = []
    for (i, j, k) in protones:
        for e in list(elec):
            if ligado[i, e] or ligado[j, e] or ligado[k, e]:
                hidrogenos.append((i, j, k, e))
                elec.remove(e)
                break
    return bar, protones, hidrogenos


def main():
    eps = 0.5  # el punto ya verificado (reproduce la configuracion de referencia)
    from cs072_proceso_holistico import construir_catalogo_desde_semilla
    nq, naq, ne, npos, diag = construir_catalogo_desde_semilla(eps)
    estado = corre_holistico(nq, naq, ne, npos, homogeneo=False, expansion=True, pasos=300)

    c_oficial = cuenta(estado)
    bar, protones, hidrogenos = _trios_con_indices(estado)

    consistente = (len(bar) == c_oficial["bariones"] and len(hidrogenos) == c_oficial["hidrogeno"])
    print(f"=== verificacion: mi re-derivacion coincide con cuenta() oficial: {consistente} ===")
    print(f"    cuenta() oficial: bariones={c_oficial['bariones']} hidrogeno={c_oficial['hidrogeno']}")
    print(f"    re-derivado:      bariones={len(bar)} hidrogeno={len(hidrogenos)}")
    if not consistente:
        print("    *** NO COINCIDE -- no confio en los indices, paro aca. ***")
        return

    N_particulas = estado["N"]
    pos, hay_wrap = asignar_posiciones(N_particulas, N_grid=16)
    print(f"\n=== posiciones asignadas: {N_particulas} particulas en malla 16^3 "
          f"(envuelve: {hay_wrap}) ===")

    def dist_max(indices):
        p = pos[list(indices)]
        d = 0.0
        for a in range(len(p)):
            for b in range(a + 1, len(p)):
                d = max(d, float(np.linalg.norm(p[a] - p[b])))
        return d

    print(f"\n=== distancia espacial REAL entre los miembros de cada barion/hidrogeno ===")
    print(f"(malla 16x16x16 -- distancia maxima posible en la diagonal: "
          f"{np.linalg.norm([15,15,15]):.2f})\n")

    distancias_bariones = [dist_max(t) for t in bar]
    distancias_hidrogeno = [dist_max(h) for h in hidrogenos]

    for i, (t, d) in enumerate(zip(bar, distancias_bariones)):
        print(f"  barion {i}: particulas {t}  posiciones {[pos[x].tolist() for x in t]}  "
              f"distancia_max={d:.2f}")
    for i, (h, d) in enumerate(zip(hidrogenos, distancias_hidrogeno)):
        print(f"  hidrogeno {i}: particulas {h}  posiciones {[pos[x].tolist() for x in h]}  "
              f"distancia_max={d:.2f}")

    # comparacion contra lo que daria una asignacion SIN relacion (control): la distancia
    # media EXACTA entre dos celdas cualquiera de la malla 16^3 -- calculada sobre TODAS
    # las celdas (16^3=4096), sin ninguna muestra al azar (cero azar, valor exacto y
    # reproducible, no una aproximacion).
    todas = np.array(np.unravel_index(np.arange(16 ** 3), (16, 16, 16))).T  # (4096, 3)
    # distancia media de la celda (0,0,0) a TODAS las demas -- por simetria de la malla
    # (traslacion no cambia la distribucion de distancias entre pares), es la misma que
    # el promedio sobre todos los pares posibles.
    d_control_todas = np.linalg.norm(todas - todas[0], axis=1)
    d_control_media = float(d_control_todas[1:].mean())  # excluye la distancia a si misma (0)

    media_bariones = float(np.mean(distancias_bariones)) if distancias_bariones else None
    media_hidrogeno = float(np.mean(distancias_hidrogeno)) if distancias_hidrogeno else None

    print(f"\n=== comparacion ===")
    print(f"  distancia media, bariones formados: {media_bariones}")
    print(f"  distancia media, hidrogeno formado: {media_hidrogeno}")
    print(f"  distancia media, control (dos celdas cualquiera de la malla): {d_control_media:.2f}")
    print(f"\n  -> los bariones/hidrogeno que el motor forma HOY estan, en el espacio, "
          f"{'cerca de lo que se esperaria al azar (o mas cerca)' if (media_bariones or 99) <= d_control_media else 'MAS LEJOS que dos celdas cualquiera al azar -- el motor los liga SIN NINGUNA relacion espacial'}.")

    resultado = dict(eps=eps, N_particulas=N_particulas, hay_wrap=hay_wrap,
                     consistente_con_cuenta=consistente,
                     bariones_indices=[list(t) for t in bar],
                     bariones_distancias=distancias_bariones,
                     hidrogeno_indices=[list(h) for h in hidrogenos],
                     hidrogeno_distancias=distancias_hidrogeno,
                     distancia_media_bariones=media_bariones,
                     distancia_media_hidrogeno=media_hidrogeno,
                     distancia_media_control_azar=d_control_media)
    out = HERE / "cs072_resultado_posicion_en_espacio.json"
    out.write_text(json.dumps(resultado, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[archivo] {out}")


if __name__ == "__main__":
    main()
