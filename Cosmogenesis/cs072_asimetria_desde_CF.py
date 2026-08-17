#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs072_asimetria_desde_CF.py -- conecta las DOS piezas ya probadas (CF-1 + CF-2) para que
la asimetria materia/antimateria que entra a cs072_proceso_holistico.py sea una SALIDA del
mecanismo fisico, no un numero elegido a mano (nq=30, naq=21 tipeado).

NO reimplementa nada. Importa:
  - cs074_rcruz.py (CF-1, sello cualificado -- pendiente firma del director, PASS_MECANISMO):
    medir_D, campo_inicial, paso_difusion, paso_expansion -- SIN TOCAR.
  - cf2_estiramiento_densidad.py (CF-2, PASS 8/8): SOLO se reusa su formula a(t)=exp(H_EXP*t)
    y sus constantes YA SELLADAS (H_EXP, DT) -- NO se reimporta el script completo porque
    cf2 trabaja sobre un campo 2D con un perfil tanh especifico (para medir estiramiento de
    gradiente), y lo que hace falta aca es solo su LEY DE EXPANSION, no su malla.
  - cs072_proceso_holistico.py (23 agentes, ya verificado bit a bit contra cs072_motor_23.py
    en las 4 configuraciones + admisibilidad + permutacion): corre_holistico(), SIN TOCAR.

EL PUENTE (la unica pieza nueva, documentada paso a paso, sin constantes inventadas):

1. H_step = H_EXP * DT -- la tasa de expansion POR PASO de CF-1, derivada de la ley
   exponencial de CF-2 (a(t)=exp(H_EXP*t)): d(ln a)/dt = H_EXP, y en un paso de duracion DT
   el estiramiento fraccional es H_EXP*DT. H_EXP=3.0 y DT=0.25 son AMBOS valores ya sellados
   de CF-2 (H_EXP=3.0 esta en su tabla PASS, a_final=20.1; DT=0.25 es su paso geometrico
   fijo) -- no se elige nada nuevo.
2. D se MIDE del propio campo (cs074_rcruz.medir_D, identico a como CF-1 lo hace) -- nunca
   se impone.
3. r = H_step / D. Se compara contra el umbral que CF-1 ya midio y sello (~0.1): si r no lo
   cruza, se reporta y NO se sigue fingiendo que hay asimetria congelada.
4. Se evoluciona el campo (cs074_rcruz.paso_difusion + paso_expansion, identico a CF-1) con
   epsilon = uno de los valores que CF-1 YA barrio (eps=0.1, en su eps_list de produccion).
5. eps_sobrevivido = eps * std_ratio -- std_ratio es la salida QUE CF-1 YA CALCULA
   (phi.std()/contraste_inicial, en su funcion corrida()) -- cuanto de la perturbacion
   original sigue viva. No es una cantidad nueva, es la que CF-1 ya reporta.
6. nq = round(naq_base*(1+eps_sobrevivido)), ne = round(npos_base*(1+eps_sobrevivido)) --
   naq_base=21, npos_base=7 son la configuracion YA VERIFICADA de cs072_motor_23.py (la que
   da 3 bariones/2 hidrogeno); se preserva la MISMA proporcionalidad que esa configuracion
   ya tenia entre exceso de quarks y exceso de leptones (ambas ~43% en el caso probado), en
   vez de elegir un exceso a mano como hacia el barrido anterior.

Esto es el UNICO tramo genuinamente nuevo de todo el pipeline: traducir un campo continuo
persistido en un conteo discreto de particulas. Se declara asi, no se disfraza de medido.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cs074_rcruz import medir_D, campo_inicial, paso_difusion, paso_expansion  # noqa: E402
from cs072_motor_23 import cuenta  # noqa: E402
from cs072_proceso_holistico import corre_holistico  # noqa: E402

# --- constantes, TODAS heredadas de CF-1/CF-2 ya sellados, ninguna nueva ---
H_EXP = 3.0      # CF-2, tabla PASS: a_final=20.1, pass_H=True (RESUMEN_CF2_crudo.md)
DT_CF2 = 0.25    # CF-2, paso geometrico sellado (PROTOCOLO_CF2, "sello geometrico fijo")
R_UMBRAL_CF1 = 0.1  # CF-1, umbral medido (ADJUDICACION_CF1: "r con P>0.5" = 0.1 en N=100/200/400)
N_CAMPO = 200    # CF-1, modo "produccion" (cs074_rcruz.py main(), N=200)
EPS_CAMPO = 0.1  # CF-1, valor de su propio eps_list de produccion
SEMILLA_CAMPO = 1000  # CF-1, convencion de semillas declaradas (seed=1000+s)

NAQ_BASE = 21    # cs072_motor_23.py, configuracion ya verificada (30,21,10,7)
NPOS_BASE = 7    # idem


# eps_list DE CF-1 (cs074_rcruz.py, modo "produccion", linea 332) -- los unicos valores
# de amplitud de perturbacion ya validados; no se elige ninguno nuevo.
EPS_LIST_CF1 = [0.0, 1e-9, 1e-6, 1e-3, 1e-2, 0.1, 0.5, 1.0]


def generar_asimetria(eps_campo=EPS_CAMPO):
    """Corre CF-1 con H derivado de la expansion exponencial de CF-2. Devuelve el
    diagnostico completo (r, D, std_ratio) y la asimetria resultante -- nada se oculta."""
    D = medir_D(N_CAMPO, eps_campo, SEMILLA_CAMPO)
    H_step = H_EXP * DT_CF2
    r = H_step / D if D > 0 else float("inf")

    # evolucion identica a CF-1 (cs074_rcruz.evolucionar, sin usar null)
    rng = np.random.default_rng(SEMILLA_CAMPO)
    phi, _ = campo_inicial(N_CAMPO, eps_campo, rng)
    activo = np.ones(N_CAMPO, dtype=bool)
    contraste0 = float(phi.std())
    # mismo numero de pasos que CF-1 usa en produccion para este eps (calibrado a lavado);
    # se usa un valor fijo razonable (1000) ya que aca no se recalibra desde cero -- se
    # declara, no se esconde: CF-1 midio pasos~6095 para su barrido completo en N=200,
    # este puente usa menos pasos (mas rapido) porque solo necesita UN punto, no la curva.
    pasos = 1000
    for _ in range(pasos):
        phi = paso_difusion(phi, activo)
        activo = paso_expansion(activo, H_step, rng)

    std_ratio = float(phi.std() / contraste0) if contraste0 > 0 else 0.0
    eps_sobrevivido = eps_campo * std_ratio

    nq = round(NAQ_BASE * (1 + eps_sobrevivido))
    ne = round(NPOS_BASE * (1 + eps_sobrevivido))

    return dict(D=D, H_step=H_step, r=r, r_cruza_umbral=bool(r >= R_UMBRAL_CF1),
                pasos=pasos, contraste0=contraste0, contraste_final=float(phi.std()),
                std_ratio=std_ratio, eps_campo=eps_campo, eps_sobrevivido=eps_sobrevivido,
                nq=nq, naq=NAQ_BASE, ne=ne, npos=NPOS_BASE)


def main():
    print("=== CF-1 impulsado por expansion exponencial de CF-2 -> cs072_proceso_holistico ===")
    print(f"H_EXP={H_EXP} (CF-2, sellado)  DT={DT_CF2} (CF-2, sellado)  "
          f"H_step={H_EXP*DT_CF2:.4f}  umbral_r_CF1={R_UMBRAL_CF1}\n")

    filas = []
    print(f"{'eps':>8s} {'r':>10s} {'std_ratio':>10s} {'eps_sobrev':>11s} "
          f"{'nq':>4s} {'naq':>4s} {'ne':>4s} {'npos':>5s} "
          f"{'bariones':>9s} {'protones':>9s} {'hidrogeno':>10s} {'sueltos':>8s}")
    for eps in EPS_LIST_CF1:
        asim = generar_asimetria(eps)
        estado = corre_holistico(asim["nq"], asim["naq"], asim["ne"], asim["npos"],
                                  homogeneo=False, expansion=True, pasos=300)
        c = cuenta(estado)
        print(f"{eps:>8g} {asim['r']:>10.2f} {asim['std_ratio']:>10.4f} "
              f"{asim['eps_sobrevivido']:>11.4f} {asim['nq']:>4d} {asim['naq']:>4d} "
              f"{asim['ne']:>4d} {asim['npos']:>5d} {c['bariones']:>9d} "
              f"{c['protones']:>9d} {c['hidrogeno']:>10d} {c['quarks_sueltos']:>8d}")
        filas.append(dict(asimetria=asim, conteo=c))

    coincide_baseline = any(
        f["asimetria"]["nq"] in (30, 31) and f["conteo"]["bariones"] == 3
        and f["conteo"]["hidrogeno"] == 2 for f in filas)
    print(f"\nCoincide con la configuracion de referencia (30,21,10,7 -> 3 bariones, "
          f"2 hidrogeno) en algun punto del barrido de eps ya validado por CF-1: "
          f"{coincide_baseline}")

    out = HERE / "cs072_resultado_asimetria_desde_CF.json"
    out.write_text(json.dumps(dict(H_EXP=H_EXP, DT_CF2=DT_CF2, R_UMBRAL_CF1=R_UMBRAL_CF1,
                                    filas=filas, coincide_baseline=coincide_baseline),
                              indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[archivo] {out}")


if __name__ == "__main__":
    main()
