#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs072_proceso_holistico.py -- Las 23 piezas de cs072_motor_23.py, envueltas como agentes
que actuan SIMULTANEAMENTE sobre un estado compartido, en vez de una tras otra dentro de
un bucle secuencial.

QUE ES ESTO Y QUE NO ES. Esto NO reemplaza cs072_motor_23.py -- lo importa y lo usa tal
cual (catalogo, condicion inicial). NINGUNA formula de fisica se reescribe: cada agente
copia, LINEA POR LINEA, el mismo calculo que ya existe en corre() (verbatim, citado en el
docstring de cada agente con el numero de linea de origen). Lo unico que cambia es la
ORQUESTACION: en corre(), cada pieza lee el estado TAL COMO QUEDO despues de que la pieza
anterior ya actuo este mismo paso (mutacion secuencial). Aca, todos los agentes leen el
MISMO estado congelado (como estaba al EMPEZAR el paso), calculan su aporte por separado,
y un proceso comun combina y aplica todos los aportes juntos -- mismo patron ya usado y
validado en cs075_arquitectura_agentes.py (ProcesoComun) y cs075_23_sobre_fisica.py
(Proceso23SobreFisica), aplicado ahora sobre la fisica REAL y ya probada, no sobre un
campo de densidad/temperatura inventado.

DOS FASES, no una, porque asi es la propia estructura de corre() (no es una eleccion
mia): (1) identidad+enlace -- las piezas que construyen dB, cambian sabor/carga o marcan
aniquilacion, TODAS leen el B/T/viva/carga de INICIO de paso (verificado en corre():
ninguna de esas piezas lee el resultado de OTRA pieza dentro del mismo bloque -- #3, #4,
#2, #12+M2, #5 y #8 leen todos el mismo B/T/viva de inicio de paso; ese es el sentido de
"simultaneo" aca: ninguna pieza ve el aporte de otra pieza este mismo paso). Sus 6
aportes se calculan TODOS antes de tocar nada, y recien ahi se aplican juntos. (2) poda
-- #9/#18 leen el B YA ACTUALIZADO por la fase 1 (asi es en corre(), linea 167-173: la
poda corta grado EXCESIVO, que solo existe despues de sumar los enlaces nuevos de este
paso) y aplican el recorte. Este documento no inventa una tercera opcion.

UN punto se probo con las dos variantes posibles, no se eligio a ojo: al APLICAR (no al
calcular) el aporte de enlace, corre() escala dB por `sqrt(outer(viva,viva))` usando el
`viva` YA actualizado por la aniquilacion de este mismo paso (linea 165) -- un par que
muere este paso no llega a sumar sus enlaces nuevos este mismo paso. Se probo tambien la
alternativa (escalar con el viva de INICIO de paso, antes de la aniquilacion) y NO
reproducia el resultado probado en 2 de 4 configuraciones (A y C, las que corren sin
poda): bariones coincidian pero hidrogeno no (0 en vez de 2) -- una diferencia real y
medida, no ignorada. Se adopto la version que aplica dB con el viva YA actualizado
(identica a corre()) porque es la que reproduce el numero probado en las 4
configuraciones, sin excepcion -- verificado en disco antes de escribir esta linea. Las
23 piezas siguen calculando su aporte de forma simultanea (nadie lee el resultado de
otra); lo que corre en dos sub-pasos es la APLICACION (primero identidad/aniquilacion,
recien despues el enlace escalado) -- misma estructura de dos sub-pasos que ya tiene
corre() en sus lineas 146-165.

Las piezas SIN codigo en corre() (verificado corriendo la propia admisibilidad de
cs072_motor_23.py: 1_espin, 11_tres_cuerpos, 13_pauli, 14_correlacion, 15_causal,
16_ssb, 17_oscuro dan INERTE) tienen agente-envoltorio que aporta CERO exacto -- no se
les inventa actividad que el motor probado no tiene. 10_enfriamiento, 23_campo y
M1_semilla tambien dieron INERTE en esa prueba (bajo expansion=True, homogeneo=False,
respectivamente: #10 esta subsumido por #9 cuando expansion=True -- mismo `elif` de
corre(); #23/M1 son perturbaciones de la condicion inicial demasiado chicas para mover
bar/H en 300 pasos) -- se preservan como estan, con el mismo agente que aplica la MISMA
formula, no se los fuerza a "hacer algo".
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cs072_motor_23 import _catalogo, _campo_termico, R_STRONG, R_EM, R_GRAV, T_CONF, T_EW, \
    LIGADO_FRAC, PODA_FRAC, SEED_EPS, G_QCD  # noqa: E402  -- import, no se reescribe nada


# ===========================================================================
# Estado congelado: una FOTO de todo lo que corre() muta durante un paso.
# ===========================================================================
class EstadoCongelado:
    """Solo lectura para los agentes -- ningun agente escribe sobre esto."""
    __slots__ = ("color", "carga", "es_anti", "es_quark", "masa", "masa_ef", "T", "B",
                 "viva", "sabor", "N", "cd", "me", "co", "T_ef", "b0")

    def __init__(self, color, carga, es_anti, es_quark, masa, masa_ef, T, B, viva, sabor,
                 cd, me, co):
        self.color, self.carga, self.es_anti, self.es_quark = color, carga, es_anti, es_quark
        self.masa, self.masa_ef, self.T, self.B, self.viva, self.sabor = masa, masa_ef, T, B, viva, sabor
        self.N = len(color)
        self.cd, self.me, self.co = cd, me, co
        self.T_ef = float(T.mean())
        self.b0 = max(float(B.sum(axis=1).mean()) / max(self.N - 1, 1), 1e-12)


class Aporte:
    """Lo que UN agente devuelve: un delta a B (enlace), y/o cambios de identidad
    (indices que cambian de sabor/carga), y/o indices que mueren este paso. Todo
    OPCIONAL -- la mayoria de los agentes solo tocan una cosa, varios no tocan nada
    (INERTE, ver docstring del modulo)."""
    __slots__ = ("dB", "flip_sabor_idx", "muere_idx")

    def __init__(self, dB=None, flip_sabor_idx=None, muere_idx=None):
        self.dB = dB
        self.flip_sabor_idx = flip_sabor_idx
        self.muere_idx = muere_idx


class Agente23:
    numero = None
    nombre = None
    fase = None  # "enlace" o "poda"

    def aporte(self, e: EstadoCongelado, apagar: frozenset) -> Aporte:
        raise NotImplementedError


# ===========================================================================
# FASE "enlace" -- todas leen el MISMO EstadoCongelado (inicio de paso)
# ===========================================================================
class A3_Fuerte(Agente23):
    """#3 fuerte/confinamiento -- verbatim corre() l.130-131."""
    numero, nombre, fase = 3, "3_fuerte", "enlace"

    def aporte(self, e, apagar):
        if "3_fuerte" not in apagar and e.T_ef < T_CONF:
            return Aporte(dB=R_STRONG * (e.cd & e.me).astype(float))
        return Aporte()


class A4_EM(Agente23):
    """#4 electromagnetismo -- verbatim corre() l.132-134."""
    numero, nombre, fase = 4, "4_em", "enlace"

    def aporte(self, e, apagar):
        if "4_em" not in apagar:
            return Aporte(dB=R_EM * e.co.astype(float))
        return Aporte()


class A2_Gravedad(Agente23):
    """#2 gravedad -- verbatim corre() l.135-137. Usa masa_ef (de #22 QCD), YA
    congelada en EstadoCongelado -- ver A22_QCD, calculada antes del paso."""
    numero, nombre, fase = 2, "2_gravedad", "enlace"

    def aporte(self, e, apagar):
        if "2_gravedad" not in apagar:
            dB = R_GRAV * np.outer(e.masa_ef, e.masa_ef) / max(float(e.masa_ef.mean()) ** 2, 1e-300) * 0.1
            return Aporte(dB=dB)
        return Aporte()


class A12_LocalidadM2Memoria(Agente23):
    """#12 localidad + M2 memoria -- verbatim corre() l.138-142. En corre() estan
    fusionadas (localidad SOLO actua si M2_memoria tambien esta activa) -- se preserva
    esa misma dependencia, no se separan en dos agentes independientes porque el
    codigo original no lo permite sin inventar un comportamiento nuevo."""
    numero, nombre, fase = 12, "12_localidad_M2_memoria", "enlace"

    def aporte(self, e, apagar):
        if "M2_memoria" not in apagar:
            persist = e.B > e.b0
            if "12_localidad" not in apagar:
                return Aporte(dB=0.05 * persist * e.B)
        return Aporte()


class A22_QCD(Agente23):
    """#22 QCD -- no aporta dB directamente: en corre() (l.120-125) calcula masa_ef,
    que #2 gravedad usa. Se resuelve ANTES del paso (ver construir_estado_congelado) y
    se guarda en EstadoCongelado.masa_ef -- este agente existe para que #22 sea
    togglable independientemente (admisibilidad), replicando exactamente esa logica."""
    numero, nombre, fase = 22, "22_qcd", "enlace"

    def aporte(self, e, apagar):
        return Aporte()  # su efecto ya esta aplicado en e.masa_ef, ver mas abajo


class A5_Debil(Agente23):
    """#5 debil/cambio de sabor -- verbatim corre() l.147-154. Lee B de INICIO de paso
    (en corre() tambien lee B ANTES de que dB de este paso se aplique -- mismo
    congelamiento, no es un cambio de semantica acá)."""
    numero, nombre, fase = 5, "5_debil", "enlace"

    def aporte(self, e, apagar):
        if "5_debil" not in apagar and e.T_ef > T_EW:
            s = e.B.sum(axis=1)
            sq = s[e.es_quark]
            if e.es_quark.any():
                thr = max(float(sq.mean()), 1e-12)
                inest = e.es_quark & (s < thr)
                if inest.any():
                    return Aporte(flip_sabor_idx=np.where(inest)[0])
        return Aporte()


class A8_Aniquilacion(Agente23):
    """#8 aniquilacion -- verbatim corre() l.155-162, pero devuelve los indices que
    mueren en vez de mutar `viva` directamente (asi otros agentes de esta MISMA fase
    pueden decidir con el `viva` de inicio de paso -- ver docstring del modulo, la
    eleccion declarada de simultaneidad estricta)."""
    numero, nombre, fase = 8, "8_aniquilacion", "enlace"

    def aporte(self, e, apagar):
        if "8_aniquilacion" not in apagar:
            muertos = []
            for es_q in [True, False]:
                for c in [0, 1, 2, -1]:
                    mat = np.where((~e.es_anti) & (e.es_quark == es_q) & (e.color == c) & (e.viva > 0.5))[0]
                    ant = np.where((e.es_anti) & (e.es_quark == es_q) & (e.color == c) & (e.viva > 0.5))[0]
                    k = min(len(mat), len(ant))
                    if k > 0:
                        muertos.extend(mat[:k].tolist())
                        muertos.extend(ant[:k].tolist())
            if muertos:
                return Aporte(muere_idx=np.array(muertos, dtype=int))
        return Aporte()


class A9_ExpansionEnfriamiento(Agente23):
    """#9 expansion (+ #18 poda, co-emergente) y #10 enfriamiento -- verbatim corre()
    l.112-116. T no es parte de la fase "enlace" combinable (es un escalar por nodo,
    no una matriz de enlace) -- se resuelve ANTES del paso igual que corre() lo hace
    ANTES del bloque de enlace, y el T resultante ya esta en EstadoCongelado del
    SIGUIENTE paso (ver construir_estado_congelado / paso())."""
    numero, nombre, fase = 9, "9_expansion_10_enfriamiento", None  # no produce Aporte, ver paso()


class A1_Espin(Agente23):
    """#1 espin/marco -- INERTE en corre() (verificado corriendo la admisibilidad del
    motor probado: dB=0, sin cambio de conteo). Sin codigo propio en corre() mas alla
    del comentario l.143-144. Aporte cero exacto, no se inventa."""
    numero, nombre, fase = 1, "1_espin", "enlace"

    def aporte(self, e, apagar):
        return Aporte()


class A11_TresCuerpos(Agente23):
    """#11 vertice 3-cuerpos -- INERTE (verificado); V se inicializa y nunca se
    actualiza en corre(). Aporte cero exacto."""
    numero, nombre, fase = 11, "11_tres_cuerpos", "enlace"

    def aporte(self, e, apagar):
        return Aporte()


class A13_Pauli(Agente23):
    """#13 Pauli -- INERTE (verificado). Sin codigo en corre(). Aporte cero exacto."""
    numero, nombre, fase = 13, "13_pauli", "enlace"

    def aporte(self, e, apagar):
        return Aporte()


class A14_Correlacion(Agente23):
    """#14 correlacion -- INERTE (verificado). Sin codigo propio separado de #12 en
    corre(). Aporte cero exacto."""
    numero, nombre, fase = 14, "14_correlacion", "enlace"

    def aporte(self, e, apagar):
        return Aporte()


class A15_Causal(Agente23):
    """#15 causal -- INERTE (verificado). t_causal se inicializa y nunca se actualiza
    en corre(). Aporte cero exacto."""
    numero, nombre, fase = 15, "15_causal", "enlace"

    def aporte(self, e, apagar):
        return Aporte()


class A16_SSB(Agente23):
    """#16 SSB -- INERTE (verificado). orient es composicion inicial, nunca se
    actualiza en corre(). Aporte cero exacto."""
    numero, nombre, fase = 16, "16_ssb", "enlace"

    def aporte(self, e, apagar):
        return Aporte()


class A17_Oscuro(Agente23):
    """#17 oscuro -- INERTE (verificado). Sin codigo en corre(). Aporte cero exacto."""
    numero, nombre, fase = 17, "17_oscuro", "enlace"

    def aporte(self, e, apagar):
        return Aporte()


class A23_Campo(Agente23):
    """#23 campo primordial -- DENTRO del paso, INERTE para bar/H (verificado): solo
    afecta la condicion inicial de T (l.98), sin codigo propio en el bucle. Aporte
    cero exacto aca, igual que en corre(). Pero, junto con M1_Semilla, dejo de ser
    inerte para el experimento COMPLETO: ver `construir_catalogo_desde_semilla()`
    mas abajo -- ahi es donde este agente y M1 controlan de verdad la rugosidad
    inicial (via CF-1+CF-2, cs072_asimetria_desde_CF.py), ANTES de que exista
    catalogo. El resto de las 23 piezas trabaja con lo que esa rugosidad produce
    (nq/naq/ne/npos), no al reves -- pedido explicito del director, 30-jul-2026."""
    numero, nombre, fase = 23, "23_campo", "enlace"

    def aporte(self, e, apagar):
        return Aporte()


class M1_Semilla(Agente23):
    """M1 semilla -- DENTRO del paso, INERTE para bar/H (verificado): solo afecta la
    condicion inicial de T (l.97), sin codigo propio en el bucle. Aporte cero exacto
    aca. Pero, junto con A23_Campo, es donde vive de verdad la asimetria primordial
    del experimento COMPLETO: ver `construir_catalogo_desde_semilla()` mas abajo --
    ahi CF-1 (persistencia) impulsado por la expansion exponencial de CF-2 (ambos
    sellados, sin constantes nuevas) determina nq/naq/ne/npos ANTES de construir el
    catalogo. Nombre fiel a su rol canonico ("M1 semilla eps: asimetria fria
    infinitesimal, condicion S>0") -- ahora es literal, no solo declarativo."""
    numero, nombre, fase = 101, "M1_semilla", "enlace"

    def aporte(self, e, apagar):
        return Aporte()


class M3_FaseCuantica(Agente23):
    """M3 fase cuantica -- ausencia arquitectonica declarada, igual que en
    cs075_23_sobre_fisica.py: no hay representacion de amplitud/fase en esta base
    (particulas con color/carga clasicos), cero exacto siempre."""
    numero, nombre, fase = 102, "M3_fase_cuantica", "enlace"

    def aporte(self, e, apagar):
        return Aporte()


# ===========================================================================
# 6_catalogo, 7_masa -- no actuan DENTRO del paso (son la condicion inicial /
# parametro estatico), igual que en corre(): la catalogacion pasa UNA vez, antes del
# bucle. Se registran igual, como en cs075_23_sobre_fisica.py se registro M1/M3 en
# Nivel 0 -- para que el inventario cierre en 23, no porque actuen paso a paso.
# ===========================================================================
class A6_Catalogo(Agente23):
    numero, nombre, fase = 6, "6_catalogo", None

    def aporte(self, e, apagar):
        return Aporte()


class A7_Masa(Agente23):
    """#7 masa -- actua UNA vez, en la construccion del catalogo (con_masa), no paso a
    paso. Verbatim corre() l.93."""
    numero, nombre, fase = 7, "7_masa", None

    def aporte(self, e, apagar):
        return Aporte()


# ===========================================================================
# FASE "poda" -- lee el B YA ACTUALIZADO por la fase enlace (asi es en corre(), l.167-173)
# ===========================================================================
class A9_Poda(Agente23):
    """#9/#18 poda (co-emergentes, mismo mecanismo verbatim corre() l.167-173)."""
    numero, nombre, fase = 9, "9_poda_18_inflacion", "poda"

    def aporte_poda(self, B_actualizado, b0_inicio_paso, apagar, expansion):
        if expansion and "9_expansion" not in apagar:
            grado = (B_actualizado > b0_inicio_paso * LIGADO_FRAC).sum(axis=1)
            gmean = max(float(grado.mean()), 1.0)
            exceso = grado > PODA_FRAC * gmean
            return exceso
        return None


PIEZAS_23 = [
    A1_Espin(), A2_Gravedad(), A3_Fuerte(), A4_EM(), A5_Debil(), A6_Catalogo(),
    A7_Masa(), A8_Aniquilacion(), A9_ExpansionEnfriamiento(), A11_TresCuerpos(),
    A12_LocalidadM2Memoria(), A13_Pauli(), A14_Correlacion(), A15_Causal(),
    A16_SSB(), A17_Oscuro(), A22_QCD(), A23_Campo(), M1_Semilla(), M3_FaseCuantica(),
]
# 20 objetos (M2_memoria e #18 inflacion no son agentes propios -- comparten codigo con
# #12 y #9 respectivamente en corre(), igual que en el motor probado; A9_Poda actua en
# la segunda fase, no cuenta dos veces). Verificado: coincide con las 20 piezas de la
# propia prueba de admisibilidad de cs072_motor_23.py __main__, mas 6_catalogo (siempre
# activo, no togglable ahi) mas 18 (=9) mas M3 (ausencia declarada) = 23.
assert len(PIEZAS_23) == 20
_AGENTES_CON_APORTE = [p for p in PIEZAS_23 if p.fase == "enlace"]
PODA = A9_Poda()


# ===========================================================================
# INVENTARIO_23 -- los 23 numeros canonicos, uno por uno, auditable. NO son 23
# objetos Python distintos (corre() en si mismo comparte codigo entre varios: #18=#9,
# #14 sin codigo propio separado de #12, M3 ausente) -- son 23 ENTRADAS que dicen,
# para cada numero del inventario, que objeto/mecanismo lo implementa y su estado
# verificado (activo con formula propia / comparte mecanismo con otro numero /
# inerte verificado en el motor probado). Mismo estilo que el "informe()" de
# cs075_23_sobre_fisica.py -- pero sobre la fisica REAL, no reinventada.
# ===========================================================================
INVENTARIO_23 = [
    dict(numero=1, nombre="1_espin", mecanismo="A1_Espin", estado="INERTE (verificado)"),
    dict(numero=2, nombre="2_gravedad", mecanismo="A2_Gravedad", estado="activo"),
    dict(numero=3, nombre="3_fuerte", mecanismo="A3_Fuerte", estado="activo"),
    dict(numero=4, nombre="4_em", mecanismo="A4_EM", estado="activo"),
    dict(numero=5, nombre="5_debil", mecanismo="A5_Debil", estado="activo"),
    dict(numero=6, nombre="6_catalogo", mecanismo="A6_Catalogo", estado="condicion inicial (siempre)"),
    dict(numero=7, nombre="7_masa", mecanismo="A7_Masa", estado="condicion inicial (siempre)"),
    dict(numero=8, nombre="8_aniquilacion", mecanismo="A8_Aniquilacion", estado="activo"),
    dict(numero=9, nombre="9_expansion", mecanismo="A9_ExpansionEnfriamiento + A9_Poda", estado="activo"),
    dict(numero=10, nombre="10_enfriamiento", mecanismo="A9_ExpansionEnfriamiento (rama elif)", estado="INERTE si expansion=True (verificado, co-emergente con #9)"),
    dict(numero=11, nombre="11_tres_cuerpos", mecanismo="A11_TresCuerpos", estado="INERTE (verificado)"),
    dict(numero=12, nombre="12_localidad", mecanismo="A12_LocalidadM2Memoria", estado="activo"),
    dict(numero=13, nombre="13_pauli", mecanismo="A13_Pauli", estado="INERTE (verificado)"),
    dict(numero=14, nombre="14_correlacion", mecanismo="A14_Correlacion", estado="INERTE (verificado, sin codigo propio en corre())"),
    dict(numero=15, nombre="15_causal", mecanismo="A15_Causal", estado="INERTE (verificado)"),
    dict(numero=16, nombre="16_ssb", mecanismo="A16_SSB", estado="INERTE (verificado)"),
    dict(numero=17, nombre="17_oscuro", mecanismo="A17_Oscuro", estado="INERTE (verificado)"),
    dict(numero=18, nombre="18_inflacion", mecanismo="A9_Poda (co-emergente con #9, verbatim docstring cs072_motor_23.py)", estado="activo, comparte mecanismo con #9"),
    dict(numero=22, nombre="22_qcd", mecanismo="A22_QCD (masa_ef, resuelto antes del paso)", estado="activo"),
    dict(numero=23, nombre="23_campo", mecanismo="A23_Campo", estado="INERTE para bar/H (verificado, solo afecta condicion inicial de T)"),
    dict(numero=101, nombre="M1_semilla", mecanismo="M1_Semilla", estado="INERTE para bar/H (verificado, solo afecta condicion inicial de T)"),
    dict(numero=102, nombre="M2_memoria", mecanismo="A12_LocalidadM2Memoria (gate compartido con #12)", estado="activo, comparte mecanismo con #12"),
    dict(numero=103, nombre="M3_fase_cuantica", mecanismo="M3_FaseCuantica", estado="ausencia arquitectonica declarada (no hay amplitud/fase en esta base)"),
]
assert len(INVENTARIO_23) == 23, f"el inventario canonico es 23, hay {len(INVENTARIO_23)}"
_nombres = [p["nombre"] for p in INVENTARIO_23]
assert len(_nombres) == len(set(_nombres)), "nombres repetidos en INVENTARIO_23"


# ===========================================================================
# El proceso: construye el catalogo con cs072_motor_23 (SIN reescribirlo), corre N
# pasos aplicando las 23 piezas simultaneamente por fase.
# ===========================================================================
def corre_holistico(nq, naq, ne, npos, homogeneo=False, expansion=True, pasos=300,
                     apagar=frozenset(), con_masa=True, semilla=True, fluct23=True,
                     perm=None):
    """Misma firma que corre() de cs072_motor_23.py, mismo catalogo, misma condicion
    inicial (import directo, no reimplementado). Unico cambio: dentro del paso, las
    piezas leen el MISMO estado congelado y se aplican juntas, en vez de una tras
    otra mutando el estado que la siguiente ya lee actualizado."""
    color, carga, es_anti, es_quark, masa = _catalogo(
        nq, naq, ne, npos, con_masa=("7_masa" not in apagar and con_masa))
    N = len(color)
    if perm is not None:
        color, carga, es_anti, es_quark, masa = (color[perm], carga[perm], es_anti[perm],
                                                   es_quark[perm], masa[perm])
    T = _campo_termico(N, homogeneo, mecanismo_semilla=(semilla and "M1_semilla" not in apagar),
                        fluct23=(fluct23 and "23_campo" not in apagar))
    B = np.zeros((N, N))
    viva = np.ones(N)
    sabor = (carga > 0).astype(np.int8)

    cd = (color[:, None] != color[None, :]) & (color[:, None] >= 0) & (color[None, :] >= 0)
    np.fill_diagonal(cd, False)
    me = (es_anti[:, None] == es_anti[None, :])
    co = (carga[:, None] != 0) & (carga[None, :] != 0) & (np.sign(carga[:, None]) != np.sign(carga[None, :]))

    for step in range(pasos):
        # T: identico a corre() -- se resuelve antes del paso, un solo agente escribe T
        if expansion and "9_expansion" not in apagar:
            T = T * (1 - 0.02 * (T.max() - T) / (T.max() + 1e-9))
        elif "10_enfriamiento" not in apagar:
            T = T * 0.999

        # masa_ef (#22 QCD): verbatim corre() l.120-125, resuelta antes de congelar,
        # igual que en corre() (se calcula antes del bloque de enlace, con el B de
        # inicio de paso).
        b0_pre = max(float(B.sum(axis=1).mean()) / max(N - 1, 1), 1e-12)
        T_ef_pre = float(T.mean())
        if "22_qcd" not in apagar:
            ligado_qcd = (B > b0_pre * LIGADO_FRAC) & cd & me
            masa_ef = masa + G_QCD * (B * ligado_qcd).sum(axis=1)
        else:
            masa_ef = masa

        # --- FASE 1: enlace + identidad + aniquilacion, TODOS sobre el mismo congelado ---
        e = EstadoCongelado(color, carga, es_anti, es_quark, masa, masa_ef, T, B, viva,
                             sabor, cd, me, co)
        dB_total = np.zeros((N, N))
        flips = []
        muertos = []
        for ag in _AGENTES_CON_APORTE:
            a = ag.aporte(e, apagar)
            if a.dB is not None:
                dB_total = dB_total + a.dB
            if a.flip_sabor_idx is not None:
                flips.append(a.flip_sabor_idx)
            if a.muere_idx is not None:
                muertos.append(a.muere_idx)

        # aplicar identidad (#5): igual que corre(), usa el B de INICIO de paso para
        # decidir, y las cargas resultantes no afectan el dB YA calculado este paso
        # (tampoco lo afectaban en corre(): el flip pasa DESPUES del bloque de enlace).
        if flips:
            idx = np.concatenate(flips)
            sabor[idx] = 1 - sabor[idx]
            carga[idx] = np.where(carga[idx] > 0, -1, 2).astype(np.int8)

        # aplicar aniquilacion (todas las decisiones de #8 ya se calcularon arriba,
        # sobre el mismo B/viva/color de inicio de paso que vieron las demas piezas --
        # eso es lo "simultaneo": nadie decidio en base al resultado de otra pieza).
        if muertos:
            idx = np.unique(np.concatenate(muertos))
            viva = viva.copy()
            viva[idx] = 0.0
            viva = np.clip(viva, 0, 1)

        # aplicar enlace, escalado por supervivencia -- IDENTICO a corre() l.165: usa
        # el `viva` YA actualizado por la aniquilacion de este mismo paso (arriba), no
        # el congelado. Probado y verificado (ver docstring del modulo): es la unica
        # de las dos variantes que reproduce el resultado probado en las 4
        # configuraciones sin excepcion.
        B = B + dB_total * np.sqrt(np.outer(viva, viva))
        np.fill_diagonal(B, 0.0)

        # --- FASE 2: poda, sobre el B YA actualizado (igual que corre() l.167-173) ---
        exceso = PODA.aporte_poda(B, b0_pre, apagar, expansion)
        if exceso is not None and exceso.any():
            B[exceso, :] *= 0.5
            B[:, exceso] *= 0.5

    estado = dict(B=B, color=color, carga=carga, es_anti=es_anti, es_quark=es_quark,
                  masa=masa_ef, viva=viva, N=N, T=T)
    return estado


# ===========================================================================
# M1_Semilla + A23_Campo, para el experimento COMPLETO: controlan la rugosidad
# inicial de verdad (CF-1+CF-2), en vez de ser inertes. Pedido del director,
# 30-jul-2026: "un agente controla la rugosidad inicial, el resto trabaja con lo
# que eso produce". Import de cs072_asimetria_desde_CF.py (CF-1+CF-2 ya sellados,
# sin constantes nuevas) -- no se reimplementa el puente, se reusa.
# ===========================================================================
def construir_catalogo_desde_semilla(eps, naq_base=21, npos_base=7):
    """M1_Semilla + A23_Campo, literal: corre CF-1 (persistencia) impulsado por la
    expansion exponencial de CF-2 (generar_asimetria, sin tocar) y devuelve
    (nq, naq, ne, npos) -- la asimetria de partículas es una SALIDA del mecanismo,
    no una entrada elegida a mano. naq_base/npos_base son la configuracion YA
    VERIFICADA de cs072_motor_23.py (21, 7); el unico parametro libre es eps, la
    rugosidad primordial, y solo se usan los valores que CF-1 ya barrio (eps_list
    en cs072_asimetria_desde_CF.py)."""
    from cs072_asimetria_desde_CF import generar_asimetria
    asim = generar_asimetria(eps)
    return asim["nq"], asim["naq"], asim["ne"], asim["npos"], asim


def corre_holistico_desde_semilla(eps, homogeneo=False, expansion=True, pasos=300,
                                   apagar=frozenset(), con_masa=True, semilla=True,
                                   fluct23=True, perm=None):
    """El experimento completo, de punta a punta: M1_Semilla+A23_Campo (CF-1+CF-2)
    determinan nq/naq/ne/npos; las 23 piezas (esta misma clase Agente23, sin tocar)
    trabajan con lo que eso produce. Misma firma que corre_holistico() salvo que
    nq/naq/ne/npos ya NO se pasan a mano -- se derivan de `eps`."""
    nq, naq, ne, npos, diagnostico_semilla = construir_catalogo_desde_semilla(eps)
    estado = corre_holistico(nq, naq, ne, npos, homogeneo=homogeneo, expansion=expansion,
                              pasos=pasos, apagar=apagar, con_masa=con_masa,
                              semilla=semilla, fluct23=fluct23, perm=perm)
    estado["diagnostico_semilla"] = diagnostico_semilla
    return estado
