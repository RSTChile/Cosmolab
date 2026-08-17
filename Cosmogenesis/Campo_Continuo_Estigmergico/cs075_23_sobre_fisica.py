#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs075_23_sobre_fisica.py — Los 23 del inventario canónico sobre la base física real,
con puerta de emergencia
=========================================================================================

Implementa INSTRUCCION_CS075_PARA_CC.md (Cosmogenesis/). Importa (NO edita)
`EstadoFisico` de `cs075_base_fisica.py` y sigue el patrón arquitectónico de
`cs075_arquitectura_agentes.py` (estigmergia: cada agente lee el campo común congelado
y deposita en él; el proceso común congela, pide depósitos a todos, suma, aplica una
vez) — pero con la interfaz de agente propia que pide la instrucción §3.1
(numero/nombre/requiere/es_casilla_falsacion + condiciones_dadas/deposito/consolidar),
distinta de `AgenteCampo` (que trabaja sobre el campo Φ topológico, cerrado).

=========================================================================================
RESOLUCIÓN DE LA AMBIGÜEDAD DEL §3.3 (verificada en disco, no inventada)
=========================================================================================
La instrucción admite que su propio desglose de niveles suma 22, no 23, y pide verificar
el inventario contra MANIFIESTO_FOLD_CS072.md antes de implementar. Se verificó:

  MANIFIESTO_FOLD_CS072.md, línea 3: "Frontera del TODO: 18 ELEMENTOS + 3 MECANISMOS +
  2 FLUCTUACIONES CUÁNTICAS (QCD #22 + campo #23) = 23."
  Línea 22-23: "21 = 18 ELEMENTOS DEL ARCO + 3 MECANISMOS DE ORIGEN (base...)."

Es decir: el conteo base de 21 YA incluye los TRES mecanismos (M1, M2, Y M3) — no dos.
`cs072_motor_23.py` (línea 29, docstring) declara M3 "fase cuántica (amplitud/fase;
FUERA salvo acople sin grilla; ausencia declarada)": M3 SÍ cuenta en el inventario de 23
(es un mecanismo de origen, igual que M1/M2), pero su efecto está DECLARADO AUSENTE en
cualquier motor sin representación de fase cuántica — como el de esta instrucción, que
trabaja con densidad/temperatura reales, no con amplitud/fase.

**El elemento que faltaba en el §3.3 de la instrucción es M3 (fase cuántica), no #24.**
El borrador de la instrucción (Nivel 6) puso "#24 tiempo (lector puro)" para completar
23 — pero #24 NO es parte del inventario canónico: el propio manifiesto lo fija en 18+3+2
sin dejar lugar a un #24, y otro documento del proyecto (`DISENO_experimento_holistico_
todos_los_factores_PARA_CC.md`, encabezado) ya había establecido esta misma distinción:
"las piezas canónicas del motor van del 1 al 23... El motor además tiene el tiempo como
pieza #24 -- NO ocupa un casillero" de los 23. Este archivo NO implementa #24 (no lo pide
el §3-5 de la instrucción como entregable, y contarlo violaría el cierre en 23 del
manifiesto). M3 reemplaza el hueco: Nivel 0 (mecanismo de origen, junto a M1 -- ambos
existen conceptualmente desde t=0), `deposito()` cero exacto siempre (declarado, no una
casilla de falsación en el sentido de las 5 -- es una ausencia ARQUITECTÓNICA: no hay
representación de fase cuántica en una malla de densidad clásica, no un nulo físico
medido y registrado). Ver clase `M3_FaseCuantica` abajo.

Las 5 casillas de falsación (#1, #11, #13, #15, #16) están confirmadas verbatim contra
`INFORME_CS_motor_23_piezas_construido.md` ("YA FALSADAS COMO SELECTORES en el arco
histórico... Estas están DECLARADAS como casillas de falsación").

=========================================================================================
LOS HITOS — anclaje de cada umbral (instrucción §3.2)
=========================================================================================
- `T_bajo_electrodebil`: T < T_EW. **Anclaje v2** (INSTRUCCION_CS075_v2_EJECUCION_PARA_CC.md
  §1, corrige v1): el anclaje original derivaba T_EW/T_CONF de la razón física 159 GeV/155
  MeV ≈ 1026 -- pero el proyecto YA tenía estos dos umbrales fijados y normalizados en
  `cs072_motor_23.py` l.42-43 (`T_CONF=0.6`, `T_EW=0.9`), usados tal cual sobre `T_ef=T.mean()`
  en un motor donde T arranca en ~1.0. Usar la razón GeV/MeV era inventar una escala donde el
  proyecto ya tenía la suya. Corregido: `T_EW = 0.9·T_inicial`, `T_CONF = 0.6·T_inicial`
  (T_inicial capturado en paso_n=0, antes de cualquier paso -- ver `Proceso23SobreFisica.
  __init__`). Ningún número nuevo: ambos son verbatim de `cs072_motor_23.py`.
- `T_bajo_confinamiento`: T < T_CONF (derivado arriba).
- `hay_sobredensidad`: existe alguna celda con δ=ρ/⟨ρ⟩ > LIGADO_FRAC. **Anclaje**:
  LIGADO_FRAC=1.5 es la constante VERBATIM de `cs072_motor_23.py` línea 44 ("ligado" =
  enlace > 1.5x promedio) -- el mismo criterio de "significativo" que ya usa el proyecto,
  no un umbral nuevo.
- `hay_nucleos`: celdas sobredensas que persisten ≥MIN_PERSISTENCIA pasos, contadas sólo
  DESPUÉS de `T_bajo_confinamiento` (para que #3 fuerte haya tenido oportunidad de actuar
  -- si no hay confinamiento, no hay fuerza fuerte que ligue nada, así que "núcleo" no
  tiene sentido antes de ese umbral).
- `hay_atomos` y `hay_red`: **NO SE ENCONTRÓ ANCLAJE EN NINGÚN ARCHIVO DEL PROYECTO.**
  `EstadoFisico` tiene UN solo campo escalar (densidad/temperatura acoplada) -- no hay
  carga, no hay especies, no hay forma de distinguir "neutro" de "cargado" sin agregar
  una distinción que la base física no tiene (y la instrucción prohíbe tocar esa base).
  Esto es EXACTAMENTE el tipo de desacuerdo que la instrucción pide reportar en vez de
  inventar (§7). Se implementa una interpretación DECLARADA, no anclada, y se marca así
  en el código y en el informe final: "hay_atomos" = persistencia EXTENDIDA (2×
  MIN_PERSISTENCIA) sobre las mismas celdas sobredensas -- una estructura más asentada
  que un núcleo recién formado, usando SOLO tiempo y densidad (lo único que la base
  tiene), sin inventar una segunda cantidad física. "hay_red" = ≥2 regiones conexas
  distintas de celdas con esa persistencia extendida (componentes conexas, mismo criterio
  que ya usa `contar_grumos` en `cs075_arquitectura_agentes.py`). **Esta interpretación
  se reporta como no verificada -- pará-y-reportá aplicado por escrito, no en silencio.**
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
COSMOGENESIS = HERE.parent
sys.path.insert(0, str(COSMOGENESIS))

from cs075_base_fisica import EstadoFisico, C_LUZ  # noqa: E402

# ---------------------------------------------------------------------------
# Constantes ancladas (ver docstring arriba para la fuente de cada una)
# ---------------------------------------------------------------------------
LIGADO_FRAC = 1.5                       # cs072_motor_23.py l.44, verbatim
RATIO_EW_CONF = 159_000.0 / 155.0       # 159 GeV / 155 MeV -- YA NO SE USA PARA EL UMBRAL
                                         # (ver INSTRUCCION_CS075_v2_EJECUCION_PARA_CC.md §1:
                                         # el proyecto ya tenía T_CONF/T_EW normalizados en
                                         # cs072_motor_23.py, esta razón física era una escala
                                         # inventada donde ya había una). Se conserva sólo para
                                         # no romper el campo de salida que ya la reportaba.
T_EW_FRAC = 0.9                         # cs072_motor_23.py l.43, verbatim (T_EW=0.9)
T_CONF_FRAC = 0.6                       # cs072_motor_23.py l.42, verbatim (T_CONF=0.6)
MIN_PERSISTENCIA = 5                    # pasos para contar sobredensidad como "núcleo"
FACTOR_ATOMOS = 2                       # DECLARADO, no anclado (ver docstring "hay_atomos")


def _recentrar(dep):
    """Fuerza suma espacial EXACTA cero: convierte un depósito en una REDISTRIBUCIÓN
    pura, nunca una fuente/sumidero neto de densidad. Necesario porque `EstadoFisico`
    usa una distribución lognormal (rho_inicial*exp(amp*ruido)) para la rugosidad
    primordial -- asimétrica (cola larga hacia arriba) -- así que un término como
    "(rho/media-1)^2 * signo(rho-media)", que PARECE simétrico, en realidad no suma
    cero sobre una distribución sesgada: acumula densidad neta paso a paso.
    Hallado midiendo (protocolo del proyecto: nunca razonando) -- ver E5 y el informe
    final. Se aplica a los agentes que representan procesos LOCALES/de redistribución
    (fuerte, EM, gravedad, QCD, oscuro, catálogo, débil, enfriamiento-local); NO se
    aplica a los procesos declaradamente GLOBALES con efecto neto intencional
    (#9 expansión -- dilución real; #8 aniquilación, #18 poda -- sumideros
    intencionales; M1 semilla -- fuente primordial declarada, pequeña)."""
    return dep - dep.mean()


def _vecindad_media(campo):
    return (np.roll(campo, 1, 0) + np.roll(campo, -1, 0)
            + np.roll(campo, 1, 1) + np.roll(campo, -1, 1)
            + np.roll(campo, 1, 2) + np.roll(campo, -1, 2)) / 6.0


# ===========================================================================
# Interfaz de agente (instrucción §3.1)
# ===========================================================================
class Agente23:
    numero = None
    nombre = None
    requiere = ()
    es_casilla_falsacion = False

    def __init__(self):
        self.paso_despertar = None
        self.pasos_dormido = 0
        self.pasos_despierto = 0

    def condiciones_dadas(self, estado, hitos):
        return all(hitos.get(h, False) for h in self.requiere)

    def deposito(self, estado, hitos):
        """PURA: no muta estado. Cero exacto si condiciones_dadas es False -- lo aplica
        el proceso común, no cada agente (así ningún agente puede "hacer trampa")."""
        raise NotImplementedError

    def consolidar(self, estado, hitos):
        return None

    def _registrar(self, paso, dado):
        """El proceso común llama esto UNA vez por paso, con el resultado YA calculado
        de condiciones_dadas() -- registra despertar/dormido/despierto (instrucción §3.4).
        No lo llama el propio agente: así el registro es auditable desde afuera."""
        if dado:
            if self.paso_despertar is None:
                self.paso_despertar = paso
            self.pasos_despierto += 1
        else:
            self.pasos_dormido += 1

    def informe(self):
        return dict(numero=self.numero, nombre=self.nombre, requiere=list(self.requiere),
                    es_casilla_falsacion=self.es_casilla_falsacion,
                    paso_despertar=self.paso_despertar,
                    pasos_dormido=self.pasos_dormido, pasos_despierto=self.pasos_despierto)


# ===========================================================================
# NIVEL 0 — el universo primordial (sin precondiciones)
# ===========================================================================
class A23_FluctuacionCampo(Agente23):
    """#23 fluctuación cuántica del campo -- la rugosidad primordial ya está en la
    condición inicial de EstadoFisico (amp_asimetria); acá se refuerza levemente cada
    paso como recordatorio de que la rugosidad es un proceso, no sólo una condición
    inicial que se apaga."""
    numero, nombre, requiere = 23, "23_campo", ()

    def deposito(self, estado, hitos):
        # CORREGIDO (mismo defecto que #2, ver ahí): lineal-en-exceso, sin gate, activo
        # desde el paso 0 -- retroalimentación positiva lenta (coef. 0.01) pero, dado
        # que corre TODA la simulación sin apagarse nunca, diverge igual dado suficientes
        # pasos. Saturada con el mismo patrón tanh que #2/#5/#22.
        std = estado.rho.std()
        if std == 0.0:
            return np.zeros_like(estado.rho)
        crudo = np.tanh((estado.rho - estado.rho.mean()) / std)
        return 0.01 * _recentrar(crudo) * estado.rho.mean()


class A22_FluctuacionQCD(Agente23):
    """#22 fluctuación cuántica QCD -- energía de campo del sector fuerte, previa a que
    haya hadrones (es la componente que en cs072_motor_23.py se suma a la masa efectiva,
    G_QCD=R_STRONG). Aquí: un término de auto-energía proporcional a rho, ya presente
    desde el inicio como parte del vacío."""
    numero, nombre, requiere = 22, "22_qcd", ()

    def deposito(self, estado, hitos):
        # HISTORIAL: (rho/media-1)^2 sin acotar -- divergencia tipo Riccati (ver §ADENDA
        # RESULTADO), corregida con tanh saturante recentrado -- misma forma que #2
        # gravedad (redistribución que refuerza donde YA hay exceso).
        #
        # TERCERA CORRECCIÓN (medida, ver cs075_resultado_diag_quien_sube_X.json): esa
        # forma resultó ser el mayor contribuyente, por lejos, a que X suba en vez de
        # bajar (cov_acum=46,2 -- ~9x el siguiente agente). Copié la forma de #2 para
        # frenar el overflow sin volver a leer el propio docstring de este agente: dice
        # "proporcional a rho, YA PRESENTE DESDE EL INICIO COMO PARTE DEL VACÍO" -- una
        # energía de vacío es, por definición física, UNIFORME (no compite por estructura
        # existente, no depende de si una celda ya tiene exceso sobre la media). La forma
        # de redistribución-que-refuerza era, dicho llanamente, la física equivocada para
        # lo que este agente dice representar. Corregido: fuente uniforme, mismo patrón
        # que M1_Semilla (no zero-sum -- es un mecanismo intencionalmente global, no una
        # redistribución) -- por construcción no contribuye a la varianza/exergía, como
        # corresponde a una energía de vacío verdaderamente uniforme.
        return 0.001 * estado.rho.mean() * np.ones_like(estado.rho)


class A9_Expansion(Agente23):
    """#9 expansión/despliegue -- YA está en EstadoFisico.paso() como dilución rho*a^-3;
    este agente es la parte que la instrucción pide expresar como depósito propio (no
    duplica la dilución del núcleo, aporta la componente de PODA por sobredensidad
    excesiva, verbatim PODA_FRAC=2.5 de cs072_motor_23.py l.45 -- co-emergente con #18,
    ver A18_Poda)."""
    numero, nombre, requiere = 9, "9_expansion", ()

    def deposito(self, estado, hitos):
        return -0.02 * estado.H() * estado.rho


class A10_Enfriamiento(Agente23):
    """#10 enfriamiento como proceso -- YA está en EstadoFisico.paso() (T*=factor);
    aporte adicional pequeño, monótono, sin precondición (INFORME_CS: #10 está subsumido
    en #9, "la expansión enfría" -- se mantienen separados para que cada uno pueda
    fallar por su cuenta, mismo criterio que ya usó cs075_23_agentes.py)."""
    numero, nombre, requiere = 10, "10_enfriamiento", ()

    def deposito(self, estado, hitos):
        crudo = (estado.T / estado.T.mean()) * (estado.rho - estado.rho.mean())
        return -0.01 * _recentrar(crudo)


class M1_Semilla(Agente23):
    """M1 semilla / asimetría ε -- el desbalance de partida ya está en la condición
    inicial (amp_asimetria de EstadoFisico); acá se reafirma como un sesgo determinista
    mínimo, no un RNG nuevo (manifiesto prohíbe azar en la semilla)."""
    numero, nombre, requiere = 101, "M1_semilla", ()

    def deposito(self, estado, hitos):
        # constante ABSOLUTA -- mismo problema de escala que #22/#5 a largo plazo (una
        # fuente que no se achica cuando rho se diluye termina dominando). Reescalada
        # por rho.mean() actual: sigue siendo la MISMA fracción declarada de asimetría
        # (amp_asimetria), no una cantidad nueva.
        return 0.001 * estado.amp_asimetria * estado.rho.mean() * np.ones_like(estado.rho)


class M3_FaseCuantica(Agente23):
    """M3 fase cuántica (amplitud/fase) -- el elemento que faltaba en el §3.3 de la
    instrucción (ver docstring del módulo). Cuenta en el inventario de 23
    (MANIFIESTO_FOLD_CS072.md l.22-23: "21 = 18 + 3 MECANISMOS DE ORIGEN"), Nivel 0
    (mecanismo de origen, como M1), pero `cs072_motor_23.py` declara su acople "FUERA
    salvo acople sin grilla; ausencia declarada" -- EstadoFisico es una malla de densidad
    clásica, sin representación de fase cuántica. Su depósito es CERO EXACTO siempre,
    no porque falle una condición sino porque el modelo no tiene con qué representarlo.
    Distinto de las 5 casillas de falsación (que SÍ pueden actuar y su nulo es un
    resultado físico medido) -- acá es una ausencia arquitectónica declarada."""
    numero, nombre, requiere = 103, "M3_fase_cuantica", ()

    def deposito(self, estado, hitos):
        return np.zeros_like(estado.rho)


# ===========================================================================
# NIVEL 1 — requieren T_bajo_electrodebil
# ===========================================================================
class A5_Debil(Agente23):
    """#5 débil/cambio de sabor -- la ÚNICA que se APAGA al enfriarse (cs072_motor_23.py
    l.147: `if T_ef > T_EW`). Aquí la puerta es la inversa de las demás: actúa DESDE que
    se cruza T_bajo_electrodebil (paso ~5, umbral v2 T_EW=0,9·T_inicial) hasta que se
    cruza T_bajo_confinamiento (paso ~36, umbral v2 T_CONF=0,6·T_inicial) -- se apaga sola,
    sin apagar-y-reportar aparte."""
    numero, nombre, requiere = 5, "5_debil", ("T_bajo_electrodebil",)

    def deposito(self, estado, hitos):
        if hitos.get("T_bajo_confinamiento", False):
            return np.zeros_like(estado.rho)  # ya muy frío: la débil dejó de operar
        # CORREGIDO (medido, ver cs075_resultado_diag_quien_sube_X.json): tercer mayor
        # contribuyente a que X suba (cov_acum=5,24). La forma anterior (tanh del exceso,
        # recentrada) la copió por analogía explícita de #22 ("mismo problema de escala
        # que #22") -- sin razón física propia para reforzar el exceso ya existente: un
        # cambio de sabor no tiene motivo conocido para concentrar o dispersar masa según
        # si una celda ya está por encima o por debajo de la media. Con #22 corregido a
        # fuente uniforme por la misma razón, se corrige #5 igual, por consistencia:
        # fuente uniforme (mismo patrón que M1_Semilla), no redistribución que refuerza.
        return 0.0004 * estado.rho.mean() * np.ones_like(estado.rho)


class A7_Masa(Agente23):
    """#7 masa (log-masa) -- emerge en la ruptura electrodébil, no antes
    (LINEA_TIEMPO_MASA_topologia_vs_fisica.md fila 4). Aporta inercia: resiste el cambio
    relativo de densidad local (un término que se opone al gradiente propio)."""
    numero, nombre, requiere = 7, "7_masa", ("T_bajo_electrodebil",)

    def deposito(self, estado, hitos):
        return -0.015 * (estado.rho - _vecindad_media(estado.rho))


class A6_Catalogo(Agente23):
    """#6 catálogo de partículas -- las especies existen cuando hay masa que las
    distinga (requiere #7, que a su vez requiere T_bajo_electrodebil). En campo continuo:
    empuja la densidad hacia un conjunto discreto de niveles (especies), sin fijar a
    mano cuáles -- son múltiplos de la densidad media."""
    numero, nombre, requiere = 6, "6_catalogo", ("T_bajo_electrodebil",)

    def deposito(self, estado, hitos):
        m = estado.rho.mean()
        niveles = np.round(estado.rho / m) * m
        return 0.005 * _recentrar(niveles - estado.rho)


class A16_SSB(Agente23):
    """#16 SSB multi-dimensional -- CASILLA DE FALSACIÓN (INFORME_CS: "no rompió colapso
    en el arco"). Requiere T_bajo_electrodebil: la ruptura de simetría es justamente el
    evento electrodébil."""
    numero, nombre, requiere = 16, "16_ssb", ("T_bajo_electrodebil",)
    es_casilla_falsacion = True

    def deposito(self, estado, hitos):
        return np.zeros_like(estado.rho)


# ===========================================================================
# NIVEL 2 — requieren T_bajo_confinamiento
# ===========================================================================
class A3_Fuerte(Agente23):
    """#3 fuerte/confinamiento -- actúa bajo la temperatura de confinamiento
    (cs072_motor_23.py l.130: `if T_ef < T_CONF`). Confina: tira la densidad de una
    celda hacia el promedio de su vecindad, el mecanismo de ligadura local."""
    numero, nombre, requiere = 3, "3_fuerte", ("T_bajo_confinamiento",)

    def deposito(self, estado, hitos):
        vec = _vecindad_media(estado.rho)
        return 0.15 * (vec - estado.rho)


class A8_Aniquilacion(Agente23):
    """#8 aniquilación materia-antimateria -- por descarte de poblaciones, no por tasa
    (piezas/README). En densidad pura (sin signo) se interpreta como relajación de los
    picos más extremos hacia el fondo -- el exceso que no "cierra" en estructura se
    cancela contra el fondo, igual que el exceso de quarks sueltos se descarta."""
    numero, nombre, requiere = 8, "8_aniquilacion", ("T_bajo_confinamiento",)

    def deposito(self, estado, hitos):
        exceso = estado.rho - LIGADO_FRAC * estado.rho.mean()
        return -0.03 * np.maximum(exceso, 0.0)


class A1_Espin(Agente23):
    """#1 espín/marco nemático -- CASILLA DE FALSACIÓN (INFORME_CS: "FALSADO C")."""
    numero, nombre, requiere = 1, "1_espin", ("T_bajo_confinamiento",)
    es_casilla_falsacion = True

    def deposito(self, estado, hitos):
        return np.zeros_like(estado.rho)


class A11_TresCuerpos(Agente23):
    """#11 vértice de 3 cuerpos -- CASILLA DE FALSACIÓN (INFORME_CS: "FALSADO")."""
    numero, nombre, requiere = 11, "11_tres_cuerpos", ("T_bajo_confinamiento",)
    es_casilla_falsacion = True

    def deposito(self, estado, hitos):
        return np.zeros_like(estado.rho)


class A13_Pauli(Agente23):
    """#13 exclusión de Pauli -- CASILLA DE FALSACIÓN (INFORME_CS: "FALSADO x3")."""
    numero, nombre, requiere = 13, "13_pauli", ("T_bajo_confinamiento",)
    es_casilla_falsacion = True

    def deposito(self, estado, hitos):
        return np.zeros_like(estado.rho)


# ===========================================================================
# NIVEL 3 — requieren hay_sobredensidad
# ===========================================================================
class A2_Gravedad(Agente23):
    """#2 gravedad ∝ masa -- teje la red por sobredensidad (piezas/README: pre-métrica,
    sobre un escalar). Requiere que exista sobredensidad significativa (δ>1.5) a la que
    agarrarse.

    CORREGIDO (medido, no de la instrucción): la fórmula original (`sobre` en unidades
    absolutas de rho, sin saturar) es retroalimentación positiva pura -- más exceso más
    depósito más exceso -- que diverge en punto flotante mucho antes de llegar a
    confinamiento (medido: overflow a rho~1e+289 en el paso ~50.000 con dt=1e-3, ver
    RESULTADO_..._PARA_CS.md §5). Mismo patrón de saturación + reescalado que ya usan
    #5 y #22 (tanh acotado en rho.std(), reescalado por rho.mean() actual): la
    sobredensidad sigue concentrándose (gravedad no repele, sólo lo que ya excede se
    acreedita), pero la TASA relativa de crecimiento se satura en vez de ser lineal sin
    freno."""
    numero, nombre, requiere = 2, "2_gravedad", ("hay_sobredensidad",)

    def deposito(self, estado, hitos):
        std = estado.rho.std()
        if std == 0.0:
            return np.zeros_like(estado.rho)
        crudo = np.tanh((estado.rho - estado.rho.mean()) / std)
        sobre = np.maximum(crudo, 0.0)
        return 0.06 * _recentrar(sobre) * estado.rho.mean()


class A12_Localidad(Agente23):
    """#12 localidad/geometrogénesis -- sólo lo cercano interactúa; el laplaciano, la
    única lectura de vecindad más allá de la propia celda. Requiere sobredensidad: sin
    diferencias no hay "cerca" que distinguir de "lejos"."""
    numero, nombre, requiere = 12, "12_localidad", ("hay_sobredensidad",)

    def deposito(self, estado, hitos):
        lap = (_vecindad_media(estado.rho) - estado.rho) / (estado.a ** 2)
        return 0.05 * lap


# ===========================================================================
# NIVEL 4 — requieren hay_nucleos
# ===========================================================================
class A4_EM(Agente23):
    """#4 electromagnetismo -- recombinación, liga la estructura y da la geometría
    (piezas/README: sin EM la geometría colapsa). Requiere núcleos: sin algo previamente
    ligado por la fuerte, no hay a qué ligar un electrón."""
    numero, nombre, requiere = 4, "4_em", ("hay_nucleos",)

    def deposito(self, estado, hitos):
        vec = _vecindad_media(estado.rho)
        return 0.10 * (vec - estado.rho) * (estado.rho > estado.rho.mean())


# ===========================================================================
# NIVEL 5 — requieren hay_atomos (DECLARADO, no anclado -- ver docstring del módulo)
# ===========================================================================
class A14_Correlacion(Agente23):
    """#14 distancia por correlación -- el manifiesto la anota solapada con #12
    localidad (misma memoria de enlace). Requiere átomos: correlaciona entidades, no
    plasma en flujo."""
    numero, nombre, requiere = 14, "14_correlacion", ("hay_atomos",)

    def deposito(self, estado, hitos):
        vec = _vecindad_media(estado.rho)
        return 0.02 * vec * (estado.rho - estado.rho.mean()) / max(estado.rho.mean(), 1e-9)


class M2_Memoria(Agente23):
    """M2 memoria de enlace -- lo que YA persiste se refuerza (cs072_motor_23.py l.139).
    El agente CON memoria propia (protocolo §2.D / instrucción: "sólo los agentes con
    memoria propia"). Requiere átomos: no hay historia de algo que no persiste."""
    numero, nombre, requiere = 102, "M2_memoria", ("hay_atomos",)

    def __init__(self, forma):
        super().__init__()
        self.W_local = np.zeros(forma)

    def deposito(self, estado, hitos):
        vec = _vecindad_media(estado.rho)
        return 0.1 * self.W_local * (vec - estado.rho)

    def consolidar(self, estado, hitos):
        if not self.condiciones_dadas(estado, hitos):
            return
        vec = _vecindad_media(estado.rho)
        contraste = (estado.rho - estado.rho.mean()) / max(estado.rho.std(), 1e-9)
        self.W_local = np.clip(self.W_local + (0.02 * contraste * vec - 0.005 * self.W_local)
                                * estado.dt, -1.0, 1.0)


class A17_Oscuro(Agente23):
    """#17 sector oscuro emergente -- INFORME_CS: "emerge como probabilidad... es la
    especie que NO siente EM, y eso sólo se distingue cuando el EM ya opera". Requiere
    átomos (para que #4 EM ya esté operando y la distinción tenga sentido): sigue la
    gravedad pero no participa del término de #4 EM."""
    numero, nombre, requiere = 17, "17_oscuro", ("hay_atomos",)

    def deposito(self, estado, hitos):
        # CORREGIDO -- mismo defecto exacto que #2 original (retroalimentación positiva
        # sin saturar), encontrado por auditoría de código al investigar el overflow
        # de #2 (ver RESULTADO_..._PARA_CS.md §5). No era el causante del colapso medido
        # (requiere hay_atomos, nunca alcanzado en las corridas hechas), pero divergiría
        # igual si se llegara a activar -- se corrige ahora, no se espera a medirlo.
        std = estado.rho.std()
        if std == 0.0:
            return np.zeros_like(estado.rho)
        crudo = np.tanh((estado.rho - estado.rho.mean()) / std)
        sobre = np.maximum(crudo, 0.0)
        return 0.03 * _recentrar(sobre) * estado.rho.mean()


# ===========================================================================
# NIVEL 6 — requieren hay_red (DECLARADO, no anclado -- ver docstring del módulo)
# ===========================================================================
class A18_Poda(Agente23):
    """#18 poda/dilución -- acoplada a #9 expansión, co-emergente (cs072_motor_23.py
    l.167, PODA_FRAC=2.5 verbatim l.45). Requiere una red de entidades (≥2 regiones
    persistentes): no se poda un enlace que todavía no existe."""
    numero, nombre, requiere = 18, "18_poda", ("hay_red",)

    def deposito(self, estado, hitos):
        activo = (estado.rho > LIGADO_FRAC * estado.rho.mean()).astype(float)
        grado = _vecindad_media(activo) * 6.0
        exceso = np.maximum(grado - 2.5, 0.0)
        return -0.05 * exceso * estado.rho


class A15_Causal(Agente23):
    """#15 estructura causal/cono -- CASILLA DE FALSACIÓN (INFORME_CS: "no dio eje").
    Requiere red: un cono causal necesita ≥2 entidades entre las que trazar antes/después."""
    numero, nombre, requiere = 15, "15_causal", ("hay_red",)
    es_casilla_falsacion = True

    def deposito(self, estado, hitos):
        return np.zeros_like(estado.rho)


# ===========================================================================
# El proceso común: congela, pide depósitos, suma, aplica, detecta hitos
# ===========================================================================
class Proceso23SobreFisica:
    def __init__(self, N=16, dt=1e-3, seed=12345, amp_asimetria=0.1, k_enfriamiento=50.0):
        self.estado = EstadoFisico(N=N, dt=dt, seed=seed, amp_asimetria=amp_asimetria,
                                    k_enfriamiento=k_enfriamiento)
        # CORREGIDO v2 (ver INSTRUCCION_CS075_v2_EJECUCION_PARA_CC.md §1): T_EW/T_CONF ya
        # NO se derivan de la razón física GeV/MeV -- el proyecto ya tenía estos umbrales
        # fijados y normalizados en cs072_motor_23.py l.42-43 (T_CONF=0.6, T_EW=0.9), usados
        # tal cual sobre T_ef=T.mean() que en ese motor arranca en ~1.0. Acá EstadoFisico no
        # guarda T_inicial como atributo propio, pero en paso_n=0 (antes de cualquier paso)
        # T.mean() == T_inicial exacto por construcción (T = T_inicial*(rho/rho.mean())) --
        # se captura ese valor y las puertas comparan T/T_inicial contra los mismos 0.6/0.9.
        self.T_inicial = float(self.estado.T.mean())
        self.T_EW = T_EW_FRAC * self.T_inicial
        self.T_CONF = T_CONF_FRAC * self.T_inicial
        self.contador_sobredenso = np.zeros((N, N, N), dtype=int)
        self.cronologia = []
        self.paso_n = 0

    def _hitos(self):
        e = self.estado
        T_media = e.temperatura_media()
        delta = e.rho / e.rho.mean()
        sobre = delta > LIGADO_FRAC
        hay_sobre = bool(np.any(sobre))
        T_bajo_conf = bool(T_media < self.T_CONF)

        self.contador_sobredenso = np.where(sobre, self.contador_sobredenso + 1, 0)
        nucleos_mask = (self.contador_sobredenso >= MIN_PERSISTENCIA)
        hay_nucleos = bool(T_bajo_conf and np.any(nucleos_mask))

        atomos_mask = (self.contador_sobredenso >= MIN_PERSISTENCIA * FACTOR_ATOMOS)
        hay_atomos = bool(hay_nucleos and np.any(atomos_mask))

        n_regiones = 0
        if hay_atomos:
            etiquetas, n_regiones = ndimage.label(atomos_mask)
        hay_red = bool(hay_atomos and n_regiones >= 2)

        return dict(
            expansion_supraluminica=e.es_supraluminico(),
            T_bajo_electrodebil=bool(T_media < self.T_EW),
            T_bajo_confinamiento=T_bajo_conf,
            hay_sobredensidad=hay_sobre,
            hay_nucleos=hay_nucleos,
            hay_atomos=hay_atomos,
            hay_red=hay_red,
            n_celdas_sobredensas=int(sobre.sum()),
            n_celdas_nucleo=int(nucleos_mask.sum()),
            n_celdas_atomo=int(atomos_mask.sum()),
            n_regiones_atomo=int(n_regiones),
            T_media=T_media,
        )

    def paso(self, agentes):
        hitos = self._hitos()
        total = np.zeros_like(self.estado.rho)
        for ag in agentes:
            dado = ag.condiciones_dadas(self.estado, hitos)
            ag._registrar(self.paso_n, dado)
            if dado:
                total = total + ag.deposito(self.estado, hitos)
        self.estado.paso(depositos=total)
        for ag in agentes:
            if ag.condiciones_dadas(self.estado, hitos):
                ag.consolidar(self.estado, hitos)
        self.paso_n += 1
        return hitos

    def correr(self, agentes, T_total, registrar_cada=200):
        pasos = int(round(T_total / self.estado.dt))
        for k in range(pasos):
            hitos = self.paso(agentes)
            if registrar_cada and (k % registrar_cada == 0 or k == pasos - 1):
                fila = dict(**self.estado.estado(), **hitos)
                self.cronologia.append(fila)
        return self.cronologia[-1] if self.cronologia else None


def construir_23(N=16, dt=1e-3, seed=12345, amp_asimetria=0.1, k_enfriamiento=50.0):
    proceso = Proceso23SobreFisica(N=N, dt=dt, seed=seed, amp_asimetria=amp_asimetria,
                                    k_enfriamiento=k_enfriamiento)
    agentes = [
        A23_FluctuacionCampo(), A22_FluctuacionQCD(), A9_Expansion(), A10_Enfriamiento(),
        M1_Semilla(), M3_FaseCuantica(),
        A5_Debil(), A7_Masa(), A6_Catalogo(), A16_SSB(),
        A3_Fuerte(), A8_Aniquilacion(), A1_Espin(), A11_TresCuerpos(), A13_Pauli(),
        A2_Gravedad(), A12_Localidad(),
        A4_EM(),
        A14_Correlacion(), M2_Memoria(forma=(N, N, N)), A17_Oscuro(),
        A18_Poda(), A15_Causal(),
    ]
    assert len(agentes) == 23, f"el inventario es 23, hay {len(agentes)}"
    nombres = [a.nombre for a in agentes]
    assert len(nombres) == len(set(nombres)), "nombres repetidos"
    return proceso, agentes
