#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs072_bateria_hipotesis_espacio.py -- varias hipotesis, probadas sistematicamente, sobre
por que 3_fuerte no respondio a la distancia real (cs072_fuerte_em_con_distancia.py: sin
cambio, verificado dos veces con dos esquemas de posicion distintos -- no era el sesgo).

Cada hipotesis ataca una explicacion distinta:

  H1 corte_duro_fuerte: el confinamiento real es de CORTO alcance (no decae suave como
     1/d^2 -- se apaga del todo mas alla de una distancia). Prueba: corte duro en vez de
     atenuacion continua.
  H2 umbral_local: la razon mas probable de por que la atenuacion pareja no cambio nada
     es que el umbral de "ligado" en cuenta() es relativo al PROMEDIO GLOBAL de toda la
     matriz -- si distancia atenua parejo, el umbral se mueve con ella y la estructura
     relativa no cambia. Prueba: umbral LOCAL (promedio entre vecinos cercanos, no
     global).
  H3 distancia_de_verdad: mide directamente si los quarks que SI forman trio estan mas
     cerca, en el esquema scrambled, que un par cualquiera de quarks del mismo estatus
     -- diagnostico, no cambia nada, solo mide.
  H4 otra_configuracion: repite fuerte-con-corte-duro en eps=1.0 (mas particulas, mas
     asimetria) para ver si el hallazgo generaliza o es propio de eps=0.5.

NO TOCA ningun archivo existente. Reusa cs072_gravedad_con_distancia.py y
cs072_fuerte_em_con_distancia.py (import, no reescritura).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cs072_motor_23 import cuenta, R_STRONG, T_CONF  # noqa: E402
import cs072_proceso_holistico as ph  # noqa: E402
from cs072_proceso_holistico import Agente23, Aporte, construir_catalogo_desde_semilla  # noqa: E402
from cs072_posicion_en_espacio import _trios_con_indices  # noqa: E402


def posicion_scramble(N_particulas, N_grid=16):
    """Mismo esquema ya validado que rompe adyacencia local (idx*97 mod capacidad)."""
    capacidad = N_grid ** 3
    idx = (np.arange(N_particulas) * 97) % capacidad
    pos = np.array(np.unravel_index(idx, (N_grid, N_grid, N_grid))).T
    diffs = pos[:, None, :] - pos[None, :, :]
    D = np.linalg.norm(diffs, axis=-1)
    return pos, D


# ===========================================================================
# H1 -- corte duro (confinamiento de corto alcance real, no atenuacion continua)
# ===========================================================================
class A3_Fuerte_corte_duro(Agente23):
    """#3 fuerte con corte duro: activo SOLO si distancia <= RADIO (analogo al radio de
    confinamiento real, corto), CERO mas alla -- no atenuado, apagado. RADIO=3 (celdas
    de malla) es una eleccion declarada, no medida de ningun archivo -- se dice asi, no
    se disfraza."""
    numero, nombre, fase = 3, "3_fuerte_corte_duro", "enlace"
    RADIO = 3

    def __init__(self, D):
        super().__init__()
        self.mask_cerca = D <= self.RADIO

    def aporte(self, e, apagar):
        if "3_fuerte" not in apagar and e.T_ef < T_CONF:
            dB = R_STRONG * (e.cd & e.me & self.mask_cerca).astype(float)
            return Aporte(dB=dB)
        return Aporte()


# ===========================================================================
# H2 -- umbral LOCAL en vez de global (re-derivacion propia de cuenta(), no toca cuenta())
# ===========================================================================
def trios_con_umbral_local(estado, D, radio_vecindad=4):
    """Igual que _trios_con_indices, pero el umbral de 'ligado' es 1.5x el promedio
    de B SOLO entre pares dentro de radio_vecindad, no el promedio global de toda la
    matriz -- ataca directamente la hipotesis de por que la atenuacion pareja no
    cambio nada (el umbral global se movia junto con la atenuacion)."""
    B, color, carga, es_anti, es_quark, viva, N = (
        estado["B"], estado["color"], estado["carga"], estado["es_anti"],
        estado["es_quark"], estado["viva"], estado["N"])
    cerca = D <= radio_vecindad
    np.fill_diagonal(cerca, False)
    b0_local = np.zeros(N)
    for i in range(N):
        vecinos = B[i][cerca[i]]
        b0_local[i] = max(float(vecinos.mean()), 1e-12) if vecinos.size else 1e-12
    umbral_local = 1.5 * np.maximum(b0_local[:, None], b0_local[None, :])
    ligado = B > umbral_local

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


def correr_variante(nq, naq, ne, npos, D, piezas_reemplazadas):
    nueva_lista = [piezas_reemplazadas.get(p.nombre, p) for p in ph.PIEZAS_23]
    ph._AGENTES_CON_APORTE = [p for p in nueva_lista if p.fase == "enlace"]
    estado = ph.corre_holistico(nq, naq, ne, npos, homogeneo=False, expansion=True, pasos=300)
    ph._AGENTES_CON_APORTE = [p for p in ph.PIEZAS_23 if p.fase == "enlace"]
    return estado


def main():
    resultados = {}

    # --- H1: corte duro, eps=0.5 ---
    print("=== H1: corte duro (RADIO=3) en 3_fuerte, eps=0.5 ===")
    eps = 0.5
    nq, naq, ne, npos, diag = construir_catalogo_desde_semilla(eps)
    N = nq + naq + ne + npos
    pos, D = posicion_scramble(N)

    estado_base = correr_variante(nq, naq, ne, npos, D, {})
    c_base = cuenta(estado_base)
    bar_base, _, _ = _trios_con_indices(estado_base)

    estado_h1 = correr_variante(nq, naq, ne, npos, D, {"3_fuerte": A3_Fuerte_corte_duro(D)})
    c_h1 = cuenta(estado_h1)
    bar_h1, _, _ = _trios_con_indices(estado_h1)

    print(f"  baseline: bariones={c_base['bariones']} hidrogeno={c_base['hidrogeno']} indices={[sorted(t) for t in bar_base]}")
    print(f"  H1 corte duro: bariones={c_h1['bariones']} hidrogeno={c_h1['hidrogeno']} indices={[sorted(t) for t in bar_h1]}")
    h1_cambio = ([sorted(t) for t in bar_base] != [sorted(t) for t in bar_h1])
    print(f"  -> cambio: {h1_cambio}")
    resultados["H1_corte_duro"] = dict(baseline=c_base, con_corte=c_h1,
                                       bariones_baseline=[sorted(t) for t in bar_base],
                                       bariones_corte=[sorted(t) for t in bar_h1], cambio=h1_cambio)

    # --- H2: umbral local ---
    print("\n=== H2: umbral LOCAL (radio_vecindad=4) en vez de global, eps=0.5, sin distancia en la fuerza ===")
    bar_h2, prot_h2, hid_h2 = trios_con_umbral_local(estado_base, D, radio_vecindad=4)
    print(f"  con umbral local: bariones={len(bar_h2)} hidrogeno={len(hid_h2)} indices={[sorted(t) for t in bar_h2]}")
    h2_cambio = ([sorted(t) for t in bar_base] != [sorted(t) for t in bar_h2])
    print(f"  -> cambio respecto al umbral global: {h2_cambio}")
    resultados["H2_umbral_local"] = dict(bariones=len(bar_h2), hidrogeno=len(hid_h2),
                                         bariones_indices=[sorted(t) for t in bar_h2], cambio=h2_cambio)

    # --- H2b: umbral local + fuerza con corte duro juntos ---
    print("\n=== H2b: umbral LOCAL + corte duro en 3_fuerte, juntos ===")
    bar_h2b, prot_h2b, hid_h2b = trios_con_umbral_local(estado_h1, D, radio_vecindad=4)
    print(f"  bariones={len(bar_h2b)} hidrogeno={len(hid_h2b)} indices={[sorted(t) for t in bar_h2b]}")
    h2b_cambio = ([sorted(t) for t in bar_base] != [sorted(t) for t in bar_h2b])
    print(f"  -> cambio respecto al baseline: {h2b_cambio}")
    resultados["H2b_umbral_local_mas_corte"] = dict(bariones=len(bar_h2b), hidrogeno=len(hid_h2b),
                                                     bariones_indices=[sorted(t) for t in bar_h2b], cambio=h2b_cambio)

    # --- H3: diagnostico -- distancia real entre quarks que SI forman trio vs promedio general ---
    print("\n=== H3: diagnostico -- distancia de los quarks que forman trio vs promedio de todos los pares de quarks ===")
    quarks_idx = np.where((~estado_base["es_anti"]) & estado_base["es_quark"] & (estado_base["viva"] > 0.5))[0]
    D_quarks = D[np.ix_(quarks_idx, quarks_idx)]
    triu = np.triu_indices_from(D_quarks, k=1)
    d_media_todos = float(D_quarks[triu].mean())
    d_trio = []
    for t in bar_base:
        for a in range(3):
            for b in range(a + 1, 3):
                d_trio.append(D[t[a], t[b]])
    d_media_trio = float(np.mean(d_trio)) if d_trio else None
    print(f"  distancia media entre TODOS los pares de quarks disponibles: {d_media_todos:.2f}")
    print(f"  distancia media entre los quarks que SI forman trio: {d_media_trio}")
    resultados["H3_diagnostico_distancia"] = dict(d_media_todos_los_pares=d_media_todos,
                                                   d_media_trio_formado=d_media_trio)

    # --- H4: generalizacion -- corte duro con eps=1.0 ---
    print("\n=== H4: corte duro en 3_fuerte, eps=1.0 (generalizacion) ===")
    eps2 = 1.0
    nq2, naq2, ne2, npos2, diag2 = construir_catalogo_desde_semilla(eps2)
    N2 = nq2 + naq2 + ne2 + npos2
    pos2, D2 = posicion_scramble(N2)

    estado_base2 = correr_variante(nq2, naq2, ne2, npos2, D2, {})
    c_base2 = cuenta(estado_base2)
    bar_base2, _, _ = _trios_con_indices(estado_base2)

    estado_h4 = correr_variante(nq2, naq2, ne2, npos2, D2, {"3_fuerte": A3_Fuerte_corte_duro(D2)})
    c_h4 = cuenta(estado_h4)
    bar_h4, _, _ = _trios_con_indices(estado_h4)

    print(f"  baseline (eps=1.0): bariones={c_base2['bariones']} hidrogeno={c_base2['hidrogeno']}")
    print(f"  con corte duro:     bariones={c_h4['bariones']} hidrogeno={c_h4['hidrogeno']}")
    h4_cambio = ([sorted(t) for t in bar_base2] != [sorted(t) for t in bar_h4])
    print(f"  -> cambio: {h4_cambio}")
    resultados["H4_generalizacion_eps1"] = dict(baseline=c_base2, con_corte=c_h4, cambio=h4_cambio)

    out = HERE / "cs072_resultado_bateria_hipotesis_espacio.json"
    out.write_text(json.dumps(resultados, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[archivo] {out}")

    print("\n=== RESUMEN ===")
    for k, v in resultados.items():
        cambio = v.get("cambio", "N/A (diagnostico)")
        print(f"  {k}: cambio={cambio}")


if __name__ == "__main__":
    main()
