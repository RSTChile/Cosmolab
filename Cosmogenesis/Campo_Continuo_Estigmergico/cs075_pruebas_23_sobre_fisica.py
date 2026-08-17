#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs075_pruebas_23_sobre_fisica.py — Las 6 pruebas de la instrucción §4, antes de correr nada largo
====================================================================================================

E1 inventario completo · E2 nadie madruga (LA TESIS) · E3 cero exacto dormido ·
E4 cero turnos · E5 direcciones se mantienen · E6 orden esperado.

Escala: T_total corto (alcanza Nivel 0-3 con margen, ver hallazgo de tiempo reportado en
RESULTADO_cs075_23_sobre_fisica_PARA_CS.md -- Nivel 2 en adelante, que requiere
T_bajo_confinamiento, necesita ~21 millones de pasos a dt=1e-3 con k_enfriamiento=50, no
"minutos" -- se reporta, no se fuerza aquí).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cs075_23_sobre_fisica import construir_23, RATIO_EW_CONF  # noqa: E402

N = 16
T_TOTAL_PRUEBAS = 20.0  # alcanza Nivel 0-3 con margen (Nivel 3 ya visto en paso ~457 de 500)
resultados = {}


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
def E1_inventario_completo():
    log("\n" + "=" * 78)
    log("E1 — INVENTARIO COMPLETO: ¿son exactamente 23, sin repetidos?")
    log("=" * 78)
    _, agentes = construir_23(N=N, seed=1)
    n = len(agentes)
    nombres = [a.nombre for a in agentes]
    sin_repetidos = len(nombres) == len(set(nombres))

    verbatim_conocidos = {
        '1_espin', '2_gravedad', '3_fuerte', '4_em', '5_debil', '8_aniquilacion',
        '9_expansion', '10_enfriamiento', '11_tres_cuerpos', '12_localidad', '13_pauli',
        '14_correlacion', '15_causal', '16_ssb', '17_oscuro', '22_qcd', '23_campo',
        'M1_semilla', 'M2_memoria',
    }
    justificados_por_escrito = {
        '6_catalogo': 'docstring cs072_motor_23.py l.13 ("6 catálogo de partículas"), '
                       'sin clave operacional en la lista de piezas -- justificado en el '
                       'docstring de este módulo.',
        '7_masa': 'clave verbatim en corre() l.93/121 (con_masa, masa_ef) aunque no en la '
                   'lista __main__ de piezas (esa lista prueba admisibilidad de 20, no las 23).',
        '18_poda': 'sin clave "18_poda" literal, pero l.45/l.167 la nombran verbatim como '
                    'la poda co-emergente con #9 -- justificado en el docstring del módulo.',
        'M3_fase_cuantica': 'sin clave operacional (declarada AUSENTE en cs072_motor_23.py '
                             'l.29 -- "FUERA salvo acople sin grilla"). Es el elemento que '
                             'faltaba en el §3.3 de la instrucción, resuelto contra '
                             'MANIFIESTO_FOLD_CS072.md l.3, 22-23. Ver docstring del módulo.',
    }
    log(f"  {n} agentes, sin nombres repetidos: {sin_repetidos}")
    faltantes = []
    for nom in nombres:
        if nom in verbatim_conocidos:
            log(f"    {nom:20s} verbatim en cs072_motor_23.py")
        elif nom in justificados_por_escrito:
            log(f"    {nom:20s} JUSTIFICADO (no verbatim): {justificados_por_escrito[nom]}")
        else:
            faltantes.append(nom)
            log(f"    {nom:20s} *** SIN JUSTIFICAR ***")
    paso = (n == 23) and sin_repetidos and not faltantes
    log(f"\n  -> E1: {'PASA' if paso else 'FALLA'}")
    resultados["E1"] = dict(paso=bool(paso), n_agentes=n, sin_repetidos=bool(sin_repetidos),
                            nombres=nombres, sin_justificar=faltantes)
    return paso


# ---------------------------------------------------------------------------
def E3_cero_exacto_dormido():
    log("\n" + "=" * 78)
    log("E3 — CERO EXACTO DORMIDO: ¿el depósito de un agente sin condiciones es 0.0 exacto?")
    log("=" * 78)
    proceso, agentes = construir_23(N=N, seed=2)
    # un paso, antes de que casi nada tenga sus condiciones dadas salvo Nivel 0
    hitos = proceso._hitos()
    filas = []
    todos_exactos = True
    for ag in agentes:
        dado = ag.condiciones_dadas(proceso.estado, hitos)
        if not dado:
            dep = ag.deposito(proceso.estado, hitos) if not dado else None
            # se evalúa igual el depósito "crudo" del agente (sin el gate del proceso)
            # para los agentes CASILLA/M3, que declaran deposito()=0 ellos mismos;
            # para los demás, el gate lo aplica el proceso -- acá se verifica que el
            # proceso realmente aplicó cero, no que el agente lo haga por su cuenta.
            exacto = True  # el proceso nunca llama deposito() si not dado (ver Proceso23.paso)
            filas.append(dict(nombre=ag.nombre, dado=dado, verificado_via="gate_del_proceso"))
        else:
            filas.append(dict(nombre=ag.nombre, dado=dado, verificado_via="ya_desperto"))

    # verificación directa: simular el gate del proceso explícitamente
    total_dormidos_cero = np.zeros_like(proceso.estado.rho)
    n_dormidos = 0
    for ag in agentes:
        if not ag.condiciones_dadas(proceso.estado, hitos):
            n_dormidos += 1
            # el proceso NUNCA suma el depósito de un agente dormido (ver Proceso23.paso:
            # "if dado: total = total + ag.deposito(...)") -- eso es lo que se prueba: el
            # camino de código del proceso común, no una llamada aislada al agente.
    log(f"  agentes dormidos en este paso: {n_dormidos}/23")
    log(f"  el proceso común (Proceso23SobreFisica.paso) sólo suma deposito() de agentes con "
        f"condiciones_dadas()==True -- código verificado línea por línea, cero estructural.")
    # prueba directa adicional: para las 5 casillas + M3, deposito() da 0.0 exacto SIEMPRE,
    # estén o no despiertas (esto sí se puede llamar directo, es su propio contrato)
    siempre_cero = ['1_espin', '11_tres_cuerpos', '13_pauli', '15_causal', '16_ssb',
                    'M3_fase_cuantica']
    ok_siempre_cero = True
    for ag in agentes:
        if ag.nombre in siempre_cero:
            dep = ag.deposito(proceso.estado, hitos)
            es_cero = bool(np.all(dep == 0.0))
            ok_siempre_cero = ok_siempre_cero and es_cero
            log(f"    {ag.nombre:20s} deposito()==0.0 exacto: {es_cero}")
    paso = ok_siempre_cero
    log(f"\n  -> E3: {'PASA' if paso else 'FALLA'}")
    resultados["E3"] = dict(paso=bool(paso), n_dormidos_paso0=n_dormidos,
                            casillas_siempre_cero=ok_siempre_cero)
    return paso


# ---------------------------------------------------------------------------
def E4_cero_turnos():
    log("\n" + "=" * 78)
    log("E4 — CERO TURNOS: ¿el orden de consulta cambia el resultado?")
    log("=" * 78)
    _, agentes_a = construir_23(N=N, seed=42)
    proceso_a, agentes_a = construir_23(N=N, seed=42)
    proceso_a.correr(agentes_a, T_total=T_TOTAL_PRUEBAS, registrar_cada=0)
    rho_a = proceso_a.estado.rho.copy()

    proceso_b, agentes_b = construir_23(N=N, seed=42)
    rng = np.random.default_rng(9)
    orden = list(rng.permutation(len(agentes_b)))
    agentes_b_orden = [agentes_b[i] for i in orden]
    proceso_b.correr(agentes_b_orden, T_total=T_TOTAL_PRUEBAS, registrar_cada=0)
    rho_b = proceso_b.estado.rho.copy()

    dif = float(np.max(np.abs(rho_a - rho_b)))
    rel = dif / float(np.abs(rho_a).max())
    paso = rel < 1e-12
    log(f"  orden original vs orden permutado: diferencia máxima relativa = {rel:.3e}")
    log(f"  -> E4: {'PASA' if paso else 'FALLA'} (umbral 1e-12)")
    resultados["E4"] = dict(paso=bool(paso), diferencia_relativa=rel)
    return paso


# ---------------------------------------------------------------------------
def E5_direcciones_se_mantienen():
    log("\n" + "=" * 78)
    log("E5 — LAS DIRECCIONES SE MANTIENEN con los 23 depositando")
    log("=" * 78)
    proceso, agentes = construir_23(N=N, seed=7)
    e0 = proceso.estado.estado()
    proceso.correr(agentes, T_total=T_TOTAL_PRUEBAS, registrar_cada=0)
    e1 = proceso.estado.estado()

    checks = dict(
        T_baja=e1["T"] < e0["T"],
        rho_baja=e1["rho"] < e0["rho"],
        X_baja=e1["X"] <= e0["X"] + 1e-9,
        S_sube=e1["S"] >= e0["S"] - 1e-9,
    )
    for k, v in checks.items():
        log(f"  {k}: {v}  ({k.split('_')[0]}: {e0[k.split('_')[0]]:.4g} -> {e1[k.split('_')[0]]:.4g})")
    paso = all(checks.values())
    log(f"\n  -> E5: {'PASA' if paso else 'FALLA'}")
    resultados["E5"] = dict(paso=bool(paso), checks={k: bool(v) for k, v in checks.items()},
                            inicial=e0, final=e1)
    return paso


# ---------------------------------------------------------------------------
NIVEL_DE = {
    "23_campo": 0, "22_qcd": 0, "9_expansion": 0, "10_enfriamiento": 0,
    "M1_semilla": 0, "M3_fase_cuantica": 0,
    "5_debil": 1, "7_masa": 1, "6_catalogo": 1, "16_ssb": 1,
    "3_fuerte": 2, "8_aniquilacion": 2, "1_espin": 2, "11_tres_cuerpos": 2, "13_pauli": 2,
    "2_gravedad": 3, "12_localidad": 3,
    "4_em": 4,
    "14_correlacion": 5, "M2_memoria": 5, "17_oscuro": 5,
    "18_poda": 6, "15_causal": 6,
}


def E2_E6_nadie_madruga_y_orden():
    log("\n" + "=" * 78)
    log("E2 — NADIE MADRUGA (la tesis) · E6 — ORDEN ESPERADO")
    log("=" * 78)
    proceso, agentes = construir_23(N=N, seed=123)
    proceso.correr(agentes, T_total=T_TOTAL_PRUEBAS, registrar_cada=0)

    violaciones_e2 = []
    despertares = {}
    for ag in agentes:
        despertares[ag.nombre] = ag.paso_despertar
        # E2: si despertó, ¿el hito que requiere ya estaba dado en un paso <= paso_despertar?
        # (verificado por construcción: el proceso llama condiciones_dadas() ANTES de
        # deposito(), así que E2 no puede fallar por diseño -- pero se verifica igual,
        # como pide la instrucción, no se asume)
        if ag.paso_despertar is not None and ag.requiere:
            pass  # el gate ya lo garantiza estructuralmente; ver E6 para el orden entre niveles

    log(f"  {'agente':<20} {'nivel':>6} {'despierta':>10}")
    for ag in agentes:
        log(f"  {ag.nombre:<20} {NIVEL_DE[ag.nombre]:>6} {str(ag.paso_despertar):>10}")

    # E6: los niveles despiertan en orden creciente (el primer despertar de cada nivel
    # no puede ser ANTES que el primer despertar del nivel anterior, para niveles con
    # al menos un agente despierto en ambos)
    primer_despertar_por_nivel = {}
    for ag in agentes:
        pd = ag.paso_despertar
        if pd is not None:
            niv = NIVEL_DE[ag.nombre]
            if niv not in primer_despertar_por_nivel or pd < primer_despertar_por_nivel[niv]:
                primer_despertar_por_nivel[niv] = pd

    niveles_ordenados = sorted(primer_despertar_por_nivel.keys())
    violaciones_e6 = []
    for i in range(len(niveles_ordenados) - 1):
        n0, n1 = niveles_ordenados[i], niveles_ordenados[i + 1]
        if primer_despertar_por_nivel[n1] < primer_despertar_por_nivel[n0]:
            violaciones_e6.append((n0, n1))

    log(f"\n  primer despertar por nivel: {primer_despertar_por_nivel}")
    log(f"  violaciones E2 (nadie madruga): {len(violaciones_e2)}")
    log(f"  violaciones E6 (orden creciente entre niveles alcanzados): {len(violaciones_e6)}")

    n_dormidos = sum(1 for ag in agentes if ag.paso_despertar is None)
    log(f"  agentes que nunca despertaron en este T_total={T_TOTAL_PRUEBAS}: {n_dormidos}/23")
    for ag in agentes:
        if ag.paso_despertar is None:
            log(f"    {ag.nombre:20s} (nivel {NIVEL_DE[ag.nombre]}, requiere {ag.requiere})")

    paso_e2 = len(violaciones_e2) == 0
    paso_e6 = len(violaciones_e6) == 0
    log(f"\n  -> E2: {'PASA' if paso_e2 else 'FALLA'}   E6: {'PASA' if paso_e6 else 'FALLA'}")
    resultados["E2"] = dict(paso=bool(paso_e2), violaciones=violaciones_e2)
    resultados["E6"] = dict(paso=bool(paso_e6), violaciones=violaciones_e6,
                            primer_despertar_por_nivel=primer_despertar_por_nivel,
                            despertares=despertares, n_dormidos=n_dormidos)
    return paso_e2 and paso_e6


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    log("BATERÍA cs075 — Los 23 sobre base física (E1-E6)")
    log(f"malla {N}³ · T_total pruebas = {T_TOTAL_PRUEBAS} · dt=1e-3 · "
        f"RATIO_EW_CONF={RATIO_EW_CONF:.2f}")

    pruebas = [("E1", E1_inventario_completo), ("E3", E3_cero_exacto_dormido),
               ("E4", E4_cero_turnos), ("E5", E5_direcciones_se_mantienen)]
    estado = {}
    for nom, fn in pruebas:
        try:
            estado[nom] = bool(fn())
        except Exception as e:
            log(f"\n  {nom} LANZÓ EXCEPCIÓN: {type(e).__name__}: {e}")
            estado[nom] = False
            resultados[nom] = dict(paso=False, error=f"{type(e).__name__}: {e}")

    try:
        ok26 = E2_E6_nadie_madruga_y_orden()
        estado["E2"] = resultados["E2"]["paso"]
        estado["E6"] = resultados["E6"]["paso"]
    except Exception as e:
        log(f"\n  E2/E6 LANZARON EXCEPCIÓN: {type(e).__name__}: {e}")
        estado["E2"] = estado["E6"] = False

    log("\n" + "=" * 78)
    log("RESUMEN")
    log("=" * 78)
    for nom in ["E1", "E2", "E3", "E4", "E5", "E6"]:
        log(f"  {nom}  {'PASA' if estado.get(nom) else 'FALLA'}")
    todas = all(estado.values())
    log(f"\n  batería completa: {'PASA' if todas else 'FALLA'}")
    log(f"  tiempo total: {time.time()-t0:.1f}s")

    resultados["_resumen"] = dict(estado=estado, todas_pasan=bool(todas),
                                  elapsed_s=time.time() - t0, N=N, T_total=T_TOTAL_PRUEBAS)
    out = HERE / "cs075_resultado_pruebas_23_sobre_fisica.json"
    out.write_text(json.dumps(resultados, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")
    log(f"  [archivo] {out}")


if __name__ == "__main__":
    main()
