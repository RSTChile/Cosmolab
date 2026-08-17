#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs072_sensibilidad_ligado_frac.py -- LIGADO_FRAC=1.5 (y el 1.5 hardcodeado en cuenta(),
textualmente independiente aunque el mismo numero hoy) no tienen ninguna justificacion
encontrada en el proyecto, igual que PODA_FRAC. PODA_FRAC resulto NO importar (sacarla
no cambio el conteo de referencia). Este script prueba si LIGADO_FRAC/el umbral de
deteccion corren la misma suerte, o si el resultado de referencia SI depende de ellos.

DOS pruebas separadas:
  A. LIGADO_FRAC (constante de modulo, afecta A22_QCD->masa_ef->gravedad): barrido via
     monkeypatch del modulo (mismo patron ya usado y verificado toda la sesion).
  B. El umbral de deteccion de cuenta() (hardcodeado 1.5, DISTINTO de LIGADO_FRAC):
     re-derivacion propia (mismo patron de _trios_con_indices) sobre el MISMO B final,
     con el umbral como parametro -- no se toca cuenta() ni cs072_motor_23.py.

Configuracion base: pipeline completo de hoy (asimetria real via CF-1+CF-2, EM retrasada
al piso real 0.0158, poda desactivada).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cs072_sin_poda import desactivar_poda  # noqa: E402
desactivar_poda()

from cs072_motor_23 import cuenta  # noqa: E402
import cs072_proceso_holistico as ph  # noqa: E402
from cs072_proceso_holistico import construir_catalogo_desde_semilla  # noqa: E402
from cs072_experimento_integrado import correr_integrado  # noqa: E402


def prueba_A_LIGADO_FRAC_modulo(eps=0.5):
    """Barre LIGADO_FRAC (afecta masa_ef via A22_QCD). OJO: cs072_proceso_holistico.py
    hizo `from cs072_motor_23 import LIGADO_FRAC` -- eso copia el nombre a SU PROPIO
    namespace, desacoplado del original. Parchar cs072_motor_23.LIGADO_FRAC no alcanza;
    hay que parchar ph.LIGADO_FRAC (el que el bucle de corre_holistico() realmente lee)."""
    print("=== A: sensibilidad de LIGADO_FRAC (constante de modulo, afecta gravedad) ===")
    original = ph.LIGADO_FRAC
    filas = []
    for valor in [1.1, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0]:
        ph.LIGADO_FRAC = valor
        r = correr_integrado(eps, T_umbral_em=0.0158, radio_fuerte=999, pasos=2000)
        print(f"  LIGADO_FRAC={valor:.2f}  bariones={r['bariones']}  hidrogeno={r['hidrogeno']}  B_max={r['B_max']:.2f}")
        filas.append(dict(LIGADO_FRAC=valor, bariones=r["bariones"], hidrogeno=r["hidrogeno"]))
    ph.LIGADO_FRAC = original
    return filas


def _trios_con_umbral(estado, factor_umbral):
    """Re-derivacion de cuenta(), IDENTICA salvo que el 1.5 hardcodeado se reemplaza por
    factor_umbral (parametro). Verbatim el resto de la logica de cs072_motor_23.cuenta()."""
    B, color, carga, es_anti, es_quark, viva, N = (
        estado["B"], estado["color"], estado["carga"], estado["es_anti"],
        estado["es_quark"], estado["viva"], estado["N"])
    b0 = max(float(B.sum(axis=1).mean()) / max(N - 1, 1), 1e-12)
    umbral = factor_umbral * b0
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
    H = 0
    for (i, j, k) in protones:
        for e in list(elec):
            if ligado[i, e] or ligado[j, e] or ligado[k, e]:
                H += 1
                elec.remove(e)
                break
    return len(bar), H


def prueba_B_umbral_deteccion(eps=0.5):
    """Barre el umbral de DETECCION (el 1.5 hardcodeado en cuenta()), sobre el MISMO B
    final -- una sola corrida, muchos umbrales, para separar 'la dinamica cambio' de
    'solo cambio donde trazamos la linea de ligado'."""
    print("\n=== B: sensibilidad del umbral de deteccion (el 1.5 de cuenta(), sobre el MISMO B final) ===")
    from cs072_proceso_holistico import corre_holistico_desde_semilla
    estado = corre_holistico_desde_semilla(eps, homogeneo=False, expansion=True, pasos=2000)
    c_oficial = cuenta(estado)
    print(f"  (referencia, cuenta() oficial con 1.5 hardcodeado): bariones={c_oficial['bariones']} hidrogeno={c_oficial['hidrogeno']}")
    filas = []
    for factor in [1.1, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0]:
        bar, hid = _trios_con_umbral(estado, factor)
        print(f"  factor_umbral={factor:.2f}  bariones={bar}  hidrogeno={hid}")
        filas.append(dict(factor_umbral=factor, bariones=bar, hidrogeno=hid))
    return c_oficial, filas


def main():
    resultados = {}
    resultados["A_LIGADO_FRAC_modulo"] = prueba_A_LIGADO_FRAC_modulo(0.5)
    c_oficial, filas_b = prueba_B_umbral_deteccion(0.5)
    resultados["B_umbral_deteccion"] = dict(referencia_oficial=c_oficial, filas=filas_b)

    out = HERE / "cs072_resultado_sensibilidad_ligado_frac.json"
    out.write_text(json.dumps(resultados, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[archivo] {out}")


if __name__ == "__main__":
    main()
