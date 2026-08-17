#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs072_bateria_colapso_EM.py -- por que retrasar EM (T_umbral=0.0158, el piso real)
colapsa la estructura en eps=0.8/1.0 (N grande) pero no en eps<=0.5 (N chico), aunque
el colapso NO depende del radio espacial (ya verificado: persiste con radio=999).

Hipotesis, cada una en su propio agente/funcion:
  H1 trayectoria_T: la curva de enfriamiento (piso, paso de cruce de T_CONF) depende de N?
  H2 traza_temporal: los bariones llegan a formarse en algun momento y despues se
     destruyen, o nunca llegan a formarse?
  H3 desacoplar_N_de_eps: es el conteo de particulas (N) o la asimetria (eps) lo que
     importa? Prueba cruzada.
  H4 barrido_fino_eps: donde exactamente esta el borde entre "funciona" y "colapsa"?
  H5 poda_diagnostico: poda se vuelve mas agresiva con mas particulas cuando EM esta apagada?

NO TOCA ningun archivo existente.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cs072_motor_23 import cuenta, R_STRONG, R_EM, T_CONF, _catalogo, _campo_termico, G_QCD, LIGADO_FRAC, PODA_FRAC  # noqa: E402
import cs072_proceso_holistico as ph  # noqa: E402
from cs072_proceso_holistico import EstadoCongelado, construir_catalogo_desde_semilla  # noqa: E402
from cs072_experimento_integrado import (A3_Fuerte_integrado, A4_EM_integrado,  # noqa: E402
                                          A12_LocalidadM2Memoria_acotada, posicion_comovil,
                                          correr_integrado)

resultados = {}


def H1_trayectoria_T():
    print("=== H1: la curva de enfriamiento depende de N? ===")
    filas = []
    for eps in [0.2, 0.5, 0.8, 1.0]:
        nq, naq, ne, npos, diag = construir_catalogo_desde_semilla(eps)
        N = nq + naq + ne + npos
        T = _campo_termico(N, False, mecanismo_semilla=True, fluct23=True)
        paso_conf = None
        for paso in range(3000):
            T = T * (1 - 0.02 * (T.max() - T) / (T.max() + 1e-9))
            T_ef = float(T.mean())
            if paso_conf is None and T_ef < T_CONF:
                paso_conf = paso
        piso = float(T.mean())
        print(f"  eps={eps}  N={N}  paso_T_CONF={paso_conf}  piso_final(3000 pasos)={piso:.6f}")
        filas.append(dict(eps=eps, N=N, paso_T_CONF=paso_conf, piso=piso))
    resultados["H1_trayectoria_T"] = filas


def H2_traza_temporal(eps=0.8, T_umbral_em=0.0158, pasos=2000):
    print(f"\n=== H2: traza temporal, eps={eps} -- ¿bariones se forman y se destruyen, o nunca aparecen? ===")
    nq, naq, ne, npos, diag = construir_catalogo_desde_semilla(eps)
    N = nq + naq + ne + npos
    pos, D_comovil = posicion_comovil(N)

    ag_fuerte = A3_Fuerte_integrado(D_comovil, pasos)
    ag_fuerte.RADIO = 999  # sin restriccion espacial, ya descartada como causa
    reemplazos = {
        "3_fuerte": ag_fuerte,
        "4_em": A4_EM_integrado(T_umbral_em),
        "12_localidad_M2_memoria": A12_LocalidadM2Memoria_acotada(),
    }
    nueva = [reemplazos.get(p.nombre, p) for p in ph.PIEZAS_23]
    ph._AGENTES_CON_APORTE = [p for p in nueva if p.fase == "enlace"]

    color, carga, es_anti, es_quark, masa = _catalogo(nq, naq, ne, npos, con_masa=True)
    T = _campo_termico(N, False, mecanismo_semilla=True, fluct23=True)
    B = np.zeros((N, N)); viva = np.ones(N); sabor = (carga > 0).astype(np.int8)
    cd = (color[:, None] != color[None, :]) & (color[:, None] >= 0) & (color[None, :] >= 0)
    np.fill_diagonal(cd, False)
    me = (es_anti[:, None] == es_anti[None, :])
    co = (carga[:, None] != 0) & (carga[None, :] != 0) & (np.sign(carga[:, None]) != np.sign(carga[None, :]))

    max_bariones_visto = 0
    filas = []
    for paso in range(pasos):
        T = T * (1 - 0.02 * (T.max() - T) / (T.max() + 1e-9))
        b0_pre = max(float(B.sum(axis=1).mean()) / max(N - 1, 1), 1e-12)
        ligado_qcd = (B > b0_pre * LIGADO_FRAC) & cd & me
        masa_ef = masa + G_QCD * (B * ligado_qcd).sum(axis=1)
        e = EstadoCongelado(color, carga, es_anti, es_quark, masa, masa_ef, T, B, viva, sabor, cd, me, co)
        dB_total = np.zeros((N, N)); flips = []; muertos = []
        for ag in ph._AGENTES_CON_APORTE:
            a = ag.aporte(e, frozenset())
            if a.dB is not None: dB_total = dB_total + a.dB
            if a.flip_sabor_idx is not None: flips.append(a.flip_sabor_idx)
            if a.muere_idx is not None: muertos.append(a.muere_idx)
        if flips:
            idx = np.concatenate(flips); sabor[idx] = 1 - sabor[idx]
            carga[idx] = np.where(carga[idx] > 0, -1, 2).astype(np.int8)
        if muertos:
            idx = np.unique(np.concatenate(muertos)); viva = viva.copy(); viva[idx] = 0.0; viva = np.clip(viva, 0, 1)
        B = B + dB_total * np.sqrt(np.outer(viva, viva)); np.fill_diagonal(B, 0.0)
        exceso = ph.PODA.aporte_poda(B, b0_pre, frozenset(), True)
        if exceso is not None and exceso.any():
            B[exceso, :] *= 0.5; B[:, exceso] *= 0.5

        if paso in [0, 50, 100, 119, 200, 300, 500, 800, 1000, 1200, 1500, 1800, 1999]:
            estado_tmp = dict(B=B, color=color, carga=carga, es_anti=es_anti, es_quark=es_quark,
                              masa=masa_ef, viva=viva, N=N, T=T)
            c = cuenta(estado_tmp)
            max_bariones_visto = max(max_bariones_visto, c["bariones"])
            print(f"  paso={paso:>5d}  T_ef={float(T.mean()):.6f}  bariones={c['bariones']}  "
                  f"hidrogeno={c['hidrogeno']}  B.max={B.max():.4f}")
            filas.append(dict(paso=paso, T_ef=float(T.mean()), bariones=c["bariones"],
                              hidrogeno=c["hidrogeno"], B_max=float(B.max())))
    ph._AGENTES_CON_APORTE = [p for p in ph.PIEZAS_23 if p.fase == "enlace"]
    print(f"  -> maximo de bariones visto en cualquier momento: {max_bariones_visto}")
    resultados["H2_traza_temporal"] = dict(eps=eps, filas=filas, max_bariones_visto=max_bariones_visto)


def H3_desacoplar_N_de_eps():
    print("\n=== H3: es N o es eps? construir un caso con N de eps=0.8 pero exceso de eps=0.5 ===")
    # eps=0.5 da nq=31,naq=21,ne=10,npos=7 (N=69). eps=0.8 da nq=38,naq=21,ne=13,npos=7 (N=79)
    # Caso cruzado: mismo N que eps=0.8 (agrandando naq/npos en vez de nq/ne), MISMA proporcion de exceso que eps=0.5
    nq5, naq5, ne5, npos5, _ = construir_catalogo_desde_semilla(0.5)
    nq8, naq8, ne8, npos8, _ = construir_catalogo_desde_semilla(0.8)
    print(f"  eps=0.5: nq={nq5} naq={naq5} ne={ne5} npos={npos5} N={nq5+naq5+ne5+npos5}")
    print(f"  eps=0.8: nq={nq8} naq={naq8} ne={ne8} npos={npos8} N={nq8+naq8+ne8+npos8}")

    # escalar la config de eps=0.5 al mismo N que eps=0.8 (mismo exceso relativo, mas particulas base)
    factor = (nq8 + naq8 + ne8 + npos8) / (nq5 + naq5 + ne5 + npos5)
    nq_cruz = round(nq5 * factor); naq_cruz = round(naq5 * factor)
    ne_cruz = round(ne5 * factor); npos_cruz = round(npos5 * factor)
    N_cruz = nq_cruz + naq_cruz + ne_cruz + npos_cruz
    print(f"  cruzado (N de eps=0.8, exceso relativo de eps=0.5): nq={nq_cruz} naq={naq_cruz} "
          f"ne={ne_cruz} npos={npos_cruz} N={N_cruz}")

    pos, D_comovil = posicion_comovil(N_cruz)
    ag_fuerte = A3_Fuerte_integrado(D_comovil, 2000); ag_fuerte.RADIO = 999
    reemplazos = {"3_fuerte": ag_fuerte, "4_em": A4_EM_integrado(0.0158),
                 "12_localidad_M2_memoria": A12_LocalidadM2Memoria_acotada()}
    nueva = [reemplazos.get(p.nombre, p) for p in ph.PIEZAS_23]
    ph._AGENTES_CON_APORTE = [p for p in nueva if p.fase == "enlace"]
    estado = ph.corre_holistico(nq_cruz, naq_cruz, ne_cruz, npos_cruz, homogeneo=False, expansion=True, pasos=2000)
    ph._AGENTES_CON_APORTE = [p for p in ph.PIEZAS_23 if p.fase == "enlace"]
    c = cuenta(estado)
    print(f"  resultado: bariones={c['bariones']} hidrogeno={c['hidrogeno']} B.max={estado['B'].max():.4f}")
    resultados["H3_desacoplar_N_de_eps"] = dict(nq=nq_cruz, naq=naq_cruz, ne=ne_cruz, npos=npos_cruz,
                                                N=N_cruz, bariones=c["bariones"], hidrogeno=c["hidrogeno"])


def H4_barrido_fino_eps():
    print("\n=== H4: barrido fino de eps (0.5 a 0.8) para ubicar el borde exacto ===")
    filas = []
    for eps in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
        r = correr_integrado(eps, T_umbral_em=0.0158, radio_fuerte=999, pasos=2000)
        print(f"  eps={eps:.2f}  N={r['N']}  bariones={r['bariones']}  hidrogeno={r['hidrogeno']}  B_max={r['B_max']:.4f}")
        filas.append(r)
    resultados["H4_barrido_fino_eps"] = filas


def H5_poda_diagnostico():
    print("\n=== H5: cuenta cuantas veces poda actua, eps=0.5 vs eps=0.8, EM retrasada ===")
    for eps in [0.5, 0.8]:
        nq, naq, ne, npos, diag = construir_catalogo_desde_semilla(eps)
        N = nq + naq + ne + npos
        pos, D_comovil = posicion_comovil(N)
        ag_fuerte = A3_Fuerte_integrado(D_comovil, 2000); ag_fuerte.RADIO = 999
        reemplazos = {"3_fuerte": ag_fuerte, "4_em": A4_EM_integrado(0.0158),
                     "12_localidad_M2_memoria": A12_LocalidadM2Memoria_acotada()}
        nueva = [reemplazos.get(p.nombre, p) for p in ph.PIEZAS_23]
        ph._AGENTES_CON_APORTE = [p for p in nueva if p.fase == "enlace"]

        color, carga, es_anti, es_quark, masa = _catalogo(nq, naq, ne, npos, con_masa=True)
        T = _campo_termico(N, False, mecanismo_semilla=True, fluct23=True)
        B = np.zeros((N, N)); viva = np.ones(N); sabor = (carga > 0).astype(np.int8)
        cd = (color[:, None] != color[None, :]) & (color[:, None] >= 0) & (color[None, :] >= 0)
        np.fill_diagonal(cd, False)
        me = (es_anti[:, None] == es_anti[None, :])
        co = (carga[:, None] != 0) & (carga[None, :] != 0) & (np.sign(carga[:, None]) != np.sign(carga[None, :]))

        n_podas = 0
        n_celdas_podadas_total = 0
        for paso in range(2000):
            T = T * (1 - 0.02 * (T.max() - T) / (T.max() + 1e-9))
            b0_pre = max(float(B.sum(axis=1).mean()) / max(N - 1, 1), 1e-12)
            ligado_qcd = (B > b0_pre * LIGADO_FRAC) & cd & me
            masa_ef = masa + G_QCD * (B * ligado_qcd).sum(axis=1)
            e = EstadoCongelado(color, carga, es_anti, es_quark, masa, masa_ef, T, B, viva, sabor, cd, me, co)
            dB_total = np.zeros((N, N)); flips = []; muertos = []
            for ag in ph._AGENTES_CON_APORTE:
                a = ag.aporte(e, frozenset())
                if a.dB is not None: dB_total = dB_total + a.dB
                if a.flip_sabor_idx is not None: flips.append(a.flip_sabor_idx)
                if a.muere_idx is not None: muertos.append(a.muere_idx)
            if flips:
                idx = np.concatenate(flips); sabor[idx] = 1 - sabor[idx]
                carga[idx] = np.where(carga[idx] > 0, -1, 2).astype(np.int8)
            if muertos:
                idx = np.unique(np.concatenate(muertos)); viva = viva.copy(); viva[idx] = 0.0; viva = np.clip(viva, 0, 1)
            B = B + dB_total * np.sqrt(np.outer(viva, viva)); np.fill_diagonal(B, 0.0)
            exceso = ph.PODA.aporte_poda(B, b0_pre, frozenset(), True)
            if exceso is not None and exceso.any():
                n_podas += 1
                n_celdas_podadas_total += int(exceso.sum())
                B[exceso, :] *= 0.5; B[:, exceso] *= 0.5
        ph._AGENTES_CON_APORTE = [p for p in ph.PIEZAS_23 if p.fase == "enlace"]
        c = cuenta(dict(B=B, color=color, carga=carga, es_anti=es_anti, es_quark=es_quark,
                        masa=masa_ef, viva=viva, N=N, T=T))
        print(f"  eps={eps}: N={N}  n_pasos_con_poda={n_podas}/2000  celdas_podadas_total={n_celdas_podadas_total}  "
              f"bariones_final={c['bariones']}")
        resultados.setdefault("H5_poda_diagnostico", []).append(
            dict(eps=eps, N=N, n_pasos_con_poda=n_podas, celdas_podadas_total=n_celdas_podadas_total,
                bariones_final=c["bariones"]))


def main():
    H1_trayectoria_T()
    H2_traza_temporal(eps=0.8)
    H3_desacoplar_N_de_eps()
    H4_barrido_fino_eps()
    H5_poda_diagnostico()

    out = HERE / "cs072_resultado_bateria_colapso_EM.json"
    out.write_text(json.dumps(resultados, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[archivo] {out}")


if __name__ == "__main__":
    main()
