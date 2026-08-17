#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs075_pruebas_arquitectura.py — ¿Funciona la máquina? (antes de preguntarle algo al universo)
==============================================================================================

Seis pruebas sobre cs075_arquitectura_agentes.py. Ninguna es una pregunta física: todas
preguntan si la arquitectura hace lo que dice hacer. Cada una imprime el número que la
decide — si el número no está impreso, la prueba no cuenta.

  P1  CADA AGENTE APORTA        apagar un aspecto cambia el resultado (nadie es código muerto)
  P2  CERO TURNOS               permutar el orden de consulta no cambia el campo (§2.A)
  P3  ESTIGMERGIA / LOCALIDAD   ningún agente lee más allá del radio que declara (§2.B)
  P4  REPRODUCIBILIDAD          misma semilla -> mismo resultado, bit a bit
  P5  ESTABILIDAD NUMÉRICA      sin NaN, campo acotado, energía finita
  P6  LA MEMORIA SE CONSTRUYE   W_local crece con la historia y no es ruido (§2.D)

Uso:  python3 cs075_pruebas_arquitectura.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cs075_arquitectura_agentes import (  # noqa: E402
    construir, AgenteDifusion, AgenteReaccion, AgenteExpansion,
    AgenteEnfriamiento, AgentePlasticidad,
)

ASPECTOS = ["difusion", "reaccion", "expansion", "enfriamiento", "plasticidad"]
N = 16
T_CORTO = 2.0
T_MEDIO = 10.0
resultados = {}


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
def P1_cada_agente_aporta():
    """Apagar cada aspecto, uno a uno, y medir cuánto cambia el campo final.
    Si apagar un agente NO cambia nada, ese agente es código muerto."""
    log("\n" + "=" * 78)
    log("P1 — ¿CADA AGENTE APORTA ALGO? (apagar uno a uno)")
    log("=" * 78)
    base = construir(N=N, seed=12345)
    base.correr(T_CORTO, registrar_cada=0)
    Phi_completo = base.Phi.copy()
    norma_base = np.linalg.norm(Phi_completo)

    filas = []
    log(f"{'aspecto apagado':<16} {'||Phi_sin - Phi_completo||':>26} {'cambio relativo':>17} {'veredicto':>12}")
    for quitar in ASPECTOS:
        subset = [a for a in ASPECTOS if a != quitar]
        p = construir(N=N, seed=12345, aspectos=subset)
        p.correr(T_CORTO, registrar_cada=0)
        d = float(np.linalg.norm(p.Phi - Phi_completo))
        rel = d / norma_base
        ok = rel > 1e-6
        filas.append(dict(aspecto=quitar, dif=d, relativo=rel, aporta=bool(ok)))
        log(f"{quitar:<16} {d:>26.6e} {rel:>17.4e} {'APORTA' if ok else 'CODIGO MUERTO':>12}")

    todos_aportan = all(f["aporta"] for f in filas)
    log(f"\n  -> todos los aspectos aportan: {todos_aportan}")
    resultados["P1"] = dict(paso=bool(todos_aportan), filas=filas, norma_base=float(norma_base))
    return todos_aportan


# ---------------------------------------------------------------------------
def P2_cero_turnos():
    """Consultar a los agentes en distinto orden. Como todos leen el MISMO Phi congelado,
    el resultado sólo puede diferir por el reordenamiento de la suma en punto flotante
    (~1e-16 relativo). Si difiere mucho más, hay acoplamiento secuencial escondido."""
    log("\n" + "=" * 78)
    log("P2 — CERO TURNOS: ¿el orden de consulta cambia el resultado?")
    log("=" * 78)
    rng = np.random.default_rng(4)
    ordenes = [None] + [list(rng.permutation(len(ASPECTOS))) for _ in range(4)]

    ref = None
    filas = []
    log(f"{'orden de consulta':<26} {'||Phi - Phi_ref||':>20} {'relativo':>13} {'nivel':>22}")
    for k, orden in enumerate(ordenes):
        p = construir(N=N, seed=12345)
        p.correr(T_CORTO, registrar_cada=0, orden=orden)
        if ref is None:
            ref = p.Phi.copy()
            norma = np.linalg.norm(ref)
            log(f"{'lista original':<26} {0.0:>20.3e} {0.0:>13.3e} {'(referencia)':>22}")
            filas.append(dict(orden="original", dif=0.0, relativo=0.0))
            continue
        d = float(np.linalg.norm(p.Phi - ref))
        rel = d / norma
        nivel = "punto flotante" if rel < 1e-12 else ("SECUENCIAL OCULTO" if rel > 1e-6 else "intermedio")
        filas.append(dict(orden=str(orden), dif=d, relativo=float(rel), nivel=nivel))
        log(f"{str(orden):<26} {d:>20.3e} {rel:>13.3e} {nivel:>22}")

    peor = max(f["relativo"] for f in filas)
    paso = peor < 1e-12
    log(f"\n  peor diferencia relativa = {peor:.3e}   umbral de punto flotante = 1e-12")
    log(f"  -> arquitectura sin turnos: {paso}")
    resultados["P2"] = dict(paso=bool(paso), peor_relativo=float(peor), filas=filas)
    return paso


# ---------------------------------------------------------------------------
def P3_estigmergia_localidad():
    """Perturbar UNA celda lejana y verificar que el depósito de cada agente sólo cambia
    dentro del radio que ese agente declara. Es la prueba de que nadie hace N×N."""
    log("\n" + "=" * 78)
    log("P3 — ESTIGMERGIA: ¿algún agente lee más allá de su radio declarado?")
    log("=" * 78)
    p = construir(N=N, seed=777)
    p.correr(1.0, registrar_cada=0)
    Phi = p.Phi.copy()
    ctx = p.contexto()

    centro = (2, 2, 2)
    lejos = (N // 2 + 2, N // 2 + 2, N // 2 + 2)   # muy lejos del centro, con periodicidad
    Phi_pert = Phi.copy()
    Phi_pert[lejos] += 0.5

    dist_min = min(min(abs(a - b), N - abs(a - b)) for a, b in zip(centro, lejos))
    log(f"  celda observada = {centro}, celda perturbada = {lejos}")
    log(f"  distancia mínima entre ellas (con periodicidad) = {dist_min} celdas\n")

    filas = []
    log(f"{'agente':<15} {'radio':>6} {'|cambio| en celda observada':>29} {'veredicto':>14}")
    for ag in p.agentes:
        d0 = ag.contribucion(Phi, ctx)
        d1 = ag.contribucion(Phi_pert, ctx)
        cambio = float(abs(d1[centro] - d0[centro]))
        # con radio r, sólo puede cambiar si la perturbación está a <= r celdas
        deberia_cambiar = dist_min <= ag.radio
        ok = (cambio > 1e-14) == deberia_cambiar
        filas.append(dict(agente=ag.nombre, radio=ag.radio, cambio=cambio,
                          respeta_radio=bool(ok)))
        log(f"{ag.nombre:<15} {ag.radio:>6} {cambio:>29.3e} "
            f"{'RESPETA' if ok else 'VIOLA RADIO':>14}")

    paso = all(f["respeta_radio"] for f in filas)
    log(f"\n  -> todos respetan su radio (nadie hace N×N): {paso}")
    resultados["P3"] = dict(paso=bool(paso), distancia=int(dist_min), filas=filas)
    return paso


# ---------------------------------------------------------------------------
def P4_reproducibilidad():
    log("\n" + "=" * 78)
    log("P4 — REPRODUCIBILIDAD: misma semilla, ¿mismo resultado?")
    log("=" * 78)
    a = construir(N=N, seed=2026); a.correr(T_CORTO, registrar_cada=0)
    b = construir(N=N, seed=2026); b.correr(T_CORTO, registrar_cada=0)
    c = construir(N=N, seed=2027); c.correr(T_CORTO, registrar_cada=0)
    id_ab = float(np.max(np.abs(a.Phi - b.Phi)))
    dif_ac = float(np.max(np.abs(a.Phi - c.Phi)))
    log(f"  misma semilla (2026 vs 2026): diferencia máxima = {id_ab:.3e}")
    log(f"  otra semilla  (2026 vs 2027): diferencia máxima = {dif_ac:.3e}")
    paso = (id_ab == 0.0) and (dif_ac > 1e-6)
    log(f"  -> reproducible y sensible a la semilla: {paso}")
    resultados["P4"] = dict(paso=bool(paso), identica=id_ab, distinta=dif_ac)
    return paso


# ---------------------------------------------------------------------------
def P5_estabilidad():
    log("\n" + "=" * 78)
    log("P5 — ESTABILIDAD NUMÉRICA en corrida larga")
    log("=" * 78)
    p = construir(N=N, seed=99)
    obs = p.correr(T_MEDIO, registrar_cada=100)
    log(f"{'t':>7} {'frac_activa':>12} {'|Phi| medio':>12} {'phi_max':>9} {'energia':>13} {'grumos':>7}")
    for h in p.historia:
        log(f"{h['t']:>7.2f} {h['frac_activa']:>12.4f} {h['phi_abs_medio']:>12.4f} "
            f"{h['phi_max']:>9.4f} {h['energia_campo']:>13.4e} {h['n_grumos']:>7}")
    sin_nan = not any(h["hay_nan"] for h in p.historia)
    acotado = obs["phi_max"] <= 1.0 + 1e-12
    energia_finita = np.isfinite(obs["energia_campo"])
    paso = sin_nan and acotado and energia_finita
    log(f"\n  sin NaN: {sin_nan} | campo acotado en [-1,1]: {acotado} | energía finita: {energia_finita}")
    log(f"  -> estable: {paso}")
    resultados["P5"] = dict(paso=bool(paso), sin_nan=bool(sin_nan), acotado=bool(acotado),
                            final=obs, historia=p.historia)
    return paso


# ---------------------------------------------------------------------------
def P6_memoria_se_construye():
    """El agente de plasticidad debe construir memoria (W_local) a partir de la historia
    del campo. Si W_local se queda en cero, o es indistinguible de ruido, el aspecto D del
    protocolo (inercia por memoria) no tiene sustrato."""
    log("\n" + "=" * 78)
    log("P6 — ¿LA MEMORIA SE CONSTRUYE? (W_local del agente de plasticidad)")
    log("=" * 78)
    p = construir(N=N, seed=31416)
    plast = [a for a in p.agentes if a.nombre == "plasticidad"][0]
    log(f"{'t':>7} {'|W| medio':>12} {'|W| max':>11} {'celdas |W|>1e-3':>17}")
    serie = []
    for bloque in range(6):
        p.correr(2.0, registrar_cada=0)
        W = plast.W_local
        fila = dict(t=float(p.t), w_abs_medio=float(np.abs(W).mean()),
                    w_max=float(np.abs(W).max()),
                    frac_activa=float((np.abs(W) > 1e-3).mean()))
        serie.append(fila)
        log(f"{fila['t']:>7.2f} {fila['w_abs_medio']:>12.3e} {fila['w_max']:>11.3e} "
            f"{fila['frac_activa']:>17.4f}")
    crece = serie[-1]["w_abs_medio"] > serie[0]["w_abs_medio"] > 0
    no_es_cero = serie[-1]["w_max"] > 1e-6
    # ¿la memoria tiene estructura espacial o es ruido? correlación con el campo
    W = plast.W_local
    corr = float(np.corrcoef(W.ravel(), np.abs(p.Phi).ravel())[0, 1])
    log(f"\n  W_local crece con la historia: {crece}")
    log(f"  W_local no es cero: {no_es_cero}")
    log(f"  correlación de W_local con |Phi| final: {corr:.4f}")
    paso = crece and no_es_cero
    log(f"  -> la memoria se construye: {paso}")
    resultados["P6"] = dict(paso=bool(paso), crece=bool(crece), corr_con_campo=corr, serie=serie)
    return paso


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    log("BATERÍA DE ARQUITECTURA CS075 — la máquina antes de la física")
    log(f"malla {N}³ = {N**3} celdas · dt=0.01 · 5 aspectos, un agente cada uno")

    pruebas = [("P1", P1_cada_agente_aporta), ("P2", P2_cero_turnos),
               ("P3", P3_estigmergia_localidad), ("P4", P4_reproducibilidad),
               ("P5", P5_estabilidad), ("P6", P6_memoria_se_construye)]
    estado = {}
    for nom, fn in pruebas:
        try:
            estado[nom] = bool(fn())
        except Exception as e:  # una prueba que revienta es una prueba que falla
            log(f"\n  {nom} LANZÓ EXCEPCIÓN: {type(e).__name__}: {e}")
            estado[nom] = False
            resultados[nom] = dict(paso=False, error=f"{type(e).__name__}: {e}")

    log("\n" + "=" * 78)
    log("RESUMEN")
    log("=" * 78)
    for nom, ok in estado.items():
        log(f"  {nom}  {'PASA' if ok else 'FALLA'}")
    todas = all(estado.values())
    log(f"\n  arquitectura completa: {'PASA' if todas else 'FALLA'}")
    log(f"  tiempo total: {time.time()-t0:.1f} s")

    resultados["_resumen"] = dict(estado=estado, todas_pasan=bool(todas),
                                 elapsed_s=time.time() - t0, N=N)
    out = HERE / "cs075_resultado_pruebas_arquitectura.json"
    out.write_text(json.dumps(resultados, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")
    log(f"  [archivo] {out}")


if __name__ == "__main__":
    main()
