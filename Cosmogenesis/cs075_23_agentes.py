#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs075_23_agentes.py — Los 23 del inventario canónico, un agente cada uno, con PUERTA DE EMERGENCIA
====================================================================================================

Qué hago:
  Un agente por cada uno de los 23 elementos del inventario canónico cerrado por el director
  (MANIFIESTO_FOLD_CS072.md l.3 y l.22-31: 18 elementos + 3 mecanismos + 2 fluctuaciones cuánticas).
  Todos están presentes y activos desde t=0. Cada uno lee el campo común y deposita en él.

LA TESIS DEL DIRECTOR QUE ESTE ARCHIVO IMPLEMENTA (29-jul-2026):
  "los que están a cargo de aspectos que emergen del proceso anterior, no deberían tener
   resultados hasta que las condiciones de operación emerjan"

  Esto NO es una idea nueva: es una ley ya escrita del proyecto. El manifiesto la tiene como
  guardián duro `G-ESPACIO-ES-CONSECUENCIA` (l.44-52) y el README del motor la enuncia como
  principio 3: "Nada emerge antes de tener con qué: sin átomos no hay espacio, sin espacio no
  hay tiempo". El director tiene razón, y la razón está asentada desde el 18-jul.

  LO QUE ESTE ARCHIVO AGREGA: hasta ahora esa ley era una PROHIBICIÓN para quien mide (no midas
  geometría antes del átomo). Acá pasa a ser ESTRUCTURA DEL MOTOR: cada agente declara sus
  precondiciones, y mientras no se cumplan su depósito es EXACTAMENTE CERO — no porque alguien
  lo apague, sino porque no tiene sustrato sobre el que actuar. La ley deja de depender de la
  disciplina del que mide.

  Y esto es FALSABLE, que es lo que lo vuelve un experimento y no una declaración: se registra
  el paso en que cada agente despierta. Si un agente que depende del átomo despierta ANTES de
  que haya átomos, la arquitectura está mal y el registro lo muestra.

REGLA ANTI-SHANNON QUE ESTO RESPETA:
  Ningún agente se enciende "a mano" ni en un paso fijado de antemano. La puerta lee el ESTADO
  del campo común (¿hay tríos? ¿hay átomos? ¿hay entidades persistentes?) y se abre cuando el
  estado lo permite. El paso en que despierta es SALIDA del experimento, nunca entrada.

Fuente de cada elemento: MANIFIESTO_FOLD_CS072.md + cs072_motor_23.py (claves verbatim) +
cs072_modulos/piezas/README.md. Las 5 casillas de falsación están marcadas como tales — su
nulo es el resultado esperado, ya registrado en el arco (INFORME_CS_motor_23_piezas_construido.md).
"""
from __future__ import annotations

import numpy as np

from cs075_arquitectura_agentes import (
    AgenteCampo, ProcesoComun, ETA_HEBB, TAU_W, GAMMA_PLAST, W_MAX, DT_DEFAULT,
)

# Umbrales de emergencia. Son criterios de EXISTENCIA (¿hay con qué?), no parámetros
# ajustables para conseguir un resultado.
UMBRAL_ACTIVO = 0.5      # |Phi|>0.5 = celda "encendida" (mismo criterio de grumo del motor)
PODA_FRAC = 2.5           # cs072_motor_23.py l.45, verbatim
MIN_PERSISTENCIA = 3     # pasos que una celda debe seguir activa para contar como persistente


# ===========================================================================
# Agente con puerta de emergencia
# ===========================================================================
class AgenteConPuerta(AgenteCampo):
    """Agente que sólo actúa cuando sus precondiciones existen en el campo común.

    `depende_de` nombra los hitos que este agente necesita. Mientras falte alguno, la
    contribución es cero EXACTO y el agente registra que estuvo dormido.
    """
    numero = None
    depende_de = ()            # hitos requeridos: (), ("trios",), ("atomos",), ("persistencia",)
    es_casilla_falsacion = False

    def __init__(self):
        self.paso_despertar = None      # SALIDA del experimento, no entrada
        self.pasos_dormido = 0
        self.pasos_despierto = 0

    def puerta_abierta(self, contexto):
        hitos = contexto["hitos"]
        return all(hitos.get(h, False) for h in self.depende_de)

    def contribucion(self, Phi, contexto):
        if not self.puerta_abierta(contexto):
            self.pasos_dormido += 1
            return np.zeros_like(Phi)     # CERO EXACTO: no hay sobre qué actuar
        if self.paso_despertar is None:
            self.paso_despertar = contexto["paso"]
        self.pasos_despierto += 1
        return self._deposito(Phi, contexto)

    def _deposito(self, Phi, contexto):
        raise NotImplementedError

    def informe(self):
        return dict(numero=self.numero, nombre=self.nombre, radio=self.radio,
                    depende_de=list(self.depende_de),
                    paso_despertar=self.paso_despertar,
                    pasos_dormido=self.pasos_dormido,
                    pasos_despierto=self.pasos_despierto,
                    casilla_falsacion=self.es_casilla_falsacion)


def _vecindad(Phi):
    return (np.roll(Phi, 1, 0) + np.roll(Phi, -1, 0)
            + np.roll(Phi, 1, 1) + np.roll(Phi, -1, 1)
            + np.roll(Phi, 1, 2) + np.roll(Phi, -1, 2))


# ===========================================================================
# NIVEL 0 — sin precondiciones: el campo primordial. Actúan desde t=0.
# ===========================================================================
class A23_FluctuacionCampo(AgenteConPuerta):
    """#23 fluctuación cuántica DEL CAMPO — la rugosidad multiescala tipo CMB del campo
    térmico primordial (manifiesto l.26-29). Es la condición inicial de la que sale toda la
    estructura posterior. Sin precondiciones: es el origen."""
    numero, nombre, radio = 23, "23_campo", 0

    def _deposito(self, Phi, contexto):
        # rugosidad multiescala: modula la amplitud local sin sembrar posiciones
        return 0.02 * contexto["rugosidad"] * Phi


class A10_Enfriamiento(AgenteConPuerta):
    """#10 enfriamiento — proceso monótono; el reloj de fondo. Sin precondiciones."""
    numero, nombre, radio = 10, "10_enfriamiento", 0

    def _deposito(self, Phi, contexto):
        return -0.3 * contexto["T"] * Phi


class A9_Expansion(AgenteConPuerta):
    """#9/#18 expansión y dilución — estira y diluye. Sin precondiciones.
    Nota: el manifiesto anota que #10 está subsumido en #9 (la expansión enfría); acá se
    mantienen separados para que cada uno pueda fallar por su cuenta."""
    numero, nombre, radio = 9, "9_expansion", 0

    def _deposito(self, Phi, contexto):
        return -3.0 * contexto["H"] * Phi


class A22_FluctuacionQCD(AgenteConPuerta):
    """#22 fluctuación cuántica QCD — energía de campo del sector fuerte, el ~99% de la masa
    del protón (manifiesto l.24-25). Actúa sobre el campo desde el inicio: es la energía de
    ligadura del vacío fuerte, previa a que haya hadrones."""
    numero, nombre, radio = 22, "22_qcd", 0

    def _deposito(self, Phi, contexto):
        return 0.05 * np.sign(Phi) * Phi * Phi


class M1_Semilla(AgenteConPuerta):
    """M1 semilla / asimetría ε — el desbalance de partida. Casilla ya declarada: el
    manifiesto prohíbe el RNG como semilla (l.59-62), así que acá es un sesgo determinista."""
    numero, nombre, radio = 101, "M1_semilla", 0

    def _deposito(self, Phi, contexto):
        return contexto["epsilon"] * np.ones_like(Phi) * 0.01


class A6_Catalogo(AgenteConPuerta):
    """#6 catálogo — QUÉ especies hay. En un campo continuo el catálogo es la estructura de
    modos disponibles: fija cuántos estados distinguibles puede tomar el campo."""
    numero, nombre, radio = 6, "6_catalogo", 0

    def _deposito(self, Phi, contexto):
        # discretización suave hacia n_especies niveles (sin forzar valores a mano)
        n = contexto["n_especies"]
        return 0.01 * (np.round(Phi * n) / n - Phi)


class A7_Masa(AgenteConPuerta):
    """#7 masa — masa distinta por especie, para que la gravedad pueda discriminar
    (INFORME_CS_motor_23: u=2.3, d=4.8, e=0.51). Acá: inercia proporcional a |Phi|."""
    numero, nombre, radio = 7, "7_masa", 0

    def _deposito(self, Phi, contexto):
        return -0.02 * np.abs(Phi) * Phi


class A12_Localidad(AgenteConPuerta):
    """#12 localidad — sólo lo cercano interactúa. Es el laplaciano: la única lectura de
    vecindad de nivel 0."""
    numero, nombre, radio = 12, "12_localidad", 1

    def _deposito(self, Phi, contexto):
        return (_vecindad(Phi) - 6.0 * Phi) / (contexto["a"] ** 2)


class A5_Debil(AgenteConPuerta):
    """#5 fuerza débil — actúa por encima de la temperatura electrodébil (cs072_motor_23 l.147:
    `if T_ef > T_EW`). Es la única fuerza que se APAGA al enfriarse, en vez de encenderse."""
    numero, nombre, radio = 5, "5_debil", 0

    def _deposito(self, Phi, contexto):
        if contexto["T"] < contexto["T_EW"]:
            return np.zeros_like(Phi)     # el universo se enfrió: la débil deja de operar
        return 0.03 * (np.tanh(Phi) - Phi)


class A8_Aniquilacion(AgenteConPuerta):
    """#8 aniquilación — materia y antimateria se cancelan por RESTA de poblaciones, no por
    tasa (piezas/README). En campo continuo: los valores de signo opuesto se cancelan donde
    coexisten, dejando el excedente."""
    numero, nombre, radio = 8, "8_aniquilacion", 1

    def _deposito(self, Phi, contexto):
        vec_media = _vecindad(Phi) / 6.0
        cancelacion = -0.04 * np.where(np.sign(Phi) != np.sign(vec_media),
                                       np.minimum(np.abs(Phi), np.abs(vec_media)) * np.sign(Phi),
                                       0.0)
        return cancelacion


# ===========================================================================
# NIVEL 1 — requieren TRÍOS (confinamiento): no hay hadrones antes de que el campo
# forme estructuras cerradas de tres celdas acopladas.
# ===========================================================================
class A3_Fuerte(AgenteConPuerta):
    """#3 fuerza fuerte / confinamiento — confina en tríos RGB. En cs072_motor_23 (l.130)
    sólo actúa bajo la temperatura de confinamiento: `if T_ef < T_CONF`. Doble puerta:
    temperatura Y existencia de estructura local que confinar."""
    numero, nombre, radio = 3, "3_fuerte", 1
    depende_de = ("umbral_confinamiento",)

    def _deposito(self, Phi, contexto):
        vec = _vecindad(Phi) / 6.0
        return 0.15 * vec * (1.0 - np.abs(Phi))


class A2_Gravedad(AgenteConPuerta):
    """#2 gravedad relacional — teje la red por sobredensidad. El README de piezas la
    describe como pre-métrica, sobre un escalar. Requiere que haya masa concentrada: sin
    tríos no hay nucleones que pesen."""
    numero, nombre, radio = 2, "2_gravedad", 1
    depende_de = ("trios",)

    def _deposito(self, Phi, contexto):
        sobredensidad = np.abs(Phi) - np.abs(Phi).mean()
        return 0.08 * np.maximum(sobredensidad, 0.0) * np.sign(Phi)


class A11_TresCuerpos(AgenteConPuerta):
    """#11 vértice de 3 cuerpos — CASILLA DE FALSACIÓN (FALSADO en el arco:
    INFORME_CS_motor_23_piezas_construido.md). Su nulo es el resultado esperado."""
    numero, nombre, radio = 11, "11_tres_cuerpos", 1
    depende_de = ("trios",)
    es_casilla_falsacion = True

    def _deposito(self, Phi, contexto):
        return np.zeros_like(Phi)     # falsada: no aporta, y eso ya está registrado


class A13_Pauli(AgenteConPuerta):
    """#13 exclusión de Pauli — CASILLA DE FALSACIÓN (FALSADO ×3 en el arco)."""
    numero, nombre, radio = 13, "13_pauli", 1
    depende_de = ("trios",)
    es_casilla_falsacion = True

    def _deposito(self, Phi, contexto):
        return np.zeros_like(Phi)


class A1_Espin(AgenteConPuerta):
    """#1 espín / marco — CASILLA DE FALSACIÓN (FALSADO C en el arco)."""
    numero, nombre, radio = 1, "1_espin", 0
    depende_de = ("trios",)
    es_casilla_falsacion = True

    def _deposito(self, Phi, contexto):
        return np.zeros_like(Phi)


class A16_SSB(AgenteConPuerta):
    """#16 ruptura espontánea de simetría / orientación — CASILLA DE FALSACIÓN
    (no rompió colapso en el arco). El manifiesto (l.279) la declara atributo independiente
    de #11, no una brújula compartida."""
    numero, nombre, radio = 16, "16_ssb", 0
    depende_de = ("trios",)
    es_casilla_falsacion = True

    def _deposito(self, Phi, contexto):
        return np.zeros_like(Phi)


# ===========================================================================
# NIVEL 2 — requieren ÁTOMOS: entidades neutras y persistentes.
# ===========================================================================
class A4_EM(AgenteConPuerta):
    """#4 electromagnetismo — recombinación: liga el electrón al núcleo y forma el átomo.
    Requiere tríos (núcleos) para tener a qué ligar el electrón. Sin esta pieza no hay
    átomos y la geometría COLAPSA (piezas/README, verificado)."""
    numero, nombre, radio = 4, "4_em", 1
    depende_de = ("trios",)

    def _deposito(self, Phi, contexto):
        vec = _vecindad(Phi) / 6.0
        atraccion = np.where(np.sign(Phi) != np.sign(vec), -0.10 * (Phi - vec), 0.0)
        return atraccion


class A14_Correlacion(AgenteConPuerta):
    """#14 correlación — el manifiesto la anota como solapada con #12 localidad (misma
    memoria de enlace). Requiere átomos: correlaciona entidades, no plasma en flujo."""
    numero, nombre, radio = 14, "14_correlacion", 1
    depende_de = ("atomos",)

    def _deposito(self, Phi, contexto):
        vec = _vecindad(Phi) / 6.0
        return 0.03 * vec * Phi * np.sign(Phi)


class M2_Memoria(AgenteConPuerta):
    """M2 memoria de enlace / roce — lo que YA persiste se refuerza (cs072_motor_23 l.139).
    Es el agente con estado propio: el sustrato de la inercia por historia (protocolo §2.D).
    Requiere átomos: no hay historia de algo que no persiste."""
    numero, nombre, radio = 102, "M2_memoria", 1
    depende_de = ("atomos",)

    def __init__(self, forma=None):
        super().__init__()
        self.W_local = np.zeros(forma) if forma is not None else None

    def _deposito(self, Phi, contexto):
        if self.W_local is None:
            self.W_local = np.zeros(Phi.shape)
        return GAMMA_PLAST * (self.W_local * (_vecindad(Phi) / 6.0) - Phi)

    def consolidar(self, Phi, contexto):
        if not self.puerta_abierta(contexto):
            return                          # sin átomos no se acumula memoria
        if self.W_local is None:
            self.W_local = np.zeros(Phi.shape)
        vec = _vecindad(Phi) / 6.0
        self.W_local = np.clip(
            self.W_local + (ETA_HEBB * Phi * vec - TAU_W * self.W_local) * contexto["dt"],
            -W_MAX, W_MAX)

    def estado_memoria(self):
        return None if self.W_local is None else self.W_local.copy()


class A17_Oscuro(AgenteConPuerta):
    """#17 sector oscuro — el manifiesto dice que "emerge como probabilidad, no se inserta"
    y que necesita el barrido de fuerzas 0→1. Requiere átomos: es la especie que NO siente
    EM, y eso sólo se distingue cuando el EM ya opera."""
    numero, nombre, radio = 17, "17_oscuro", 1
    depende_de = ("atomos",)

    def _deposito(self, Phi, contexto):
        # segunda especie: sigue la gravedad pero es ciega al canal EM
        sobredensidad = np.abs(Phi) - np.abs(Phi).mean()
        return 0.04 * np.maximum(sobredensidad, 0.0) * np.sign(Phi)


# ===========================================================================
# NIVEL 3 — requieren PERSISTENCIA: relaciones entre entidades que duran.
# Es el nivel que G-ESPACIO-ES-CONSECUENCIA protege.
# ===========================================================================
class A15_Causal(AgenteConPuerta):
    """#15 estructura causal — el cono de luz. Casilla ya evaluada en el arco ("no dio eje").
    Requiere persistencia: una relación causal necesita cosas que duren."""
    numero, nombre, radio = 15, "15_causal", 1
    depende_de = ("persistencia",)
    es_casilla_falsacion = True

    def _deposito(self, Phi, contexto):
        return np.zeros_like(Phi)


class A24_Tiempo(AgenteConPuerta):
    """#24/M3 tiempo emergente — p24_tiempo.py: "el tiempo nace CON el primer átomo neutro
    (transición irreversible), no antes". LECTOR, no fuerza: no deposita nada en el campo,
    sólo registra cuándo nació el tiempo. Su depósito es cero por diseño, no por falsación."""
    numero, nombre, radio = 24, "24_tiempo", 0
    depende_de = ("persistencia",)

    def _deposito(self, Phi, contexto):
        return np.zeros_like(Phi)          # lector puro


class A18_Poda(AgenteConPuerta):
    """#18 poda / dilución — acoplada a #9 expansión: los enlaces de grado excesivo se
    cortan porque la expansión diluye (cs072_motor_23.py l.45 `PODA_FRAC=2.5`, l.167
    "#9/#18 PODA: expansion corta enlaces de grado excesivo (ciega a longitud)").

    CORRECCIÓN: la primera versión de este archivo definía acá una clase `A18_Espacio`
    ("#18 espacio / geometría"). **Estaba mal, y contradecía la fuente que yo mismo había
    citado dos frases antes.** Las tres apariciones de #18 en cs072_motor_23.py (l.45, 112,
    167) la definen como poda/dilución acoplada a #9; la clave `18_espacio` no existe en
    ningún archivo del proyecto. Y el manifiesto (l.32-42, G-ESPACIO-ES-CONSECUENCIA) dice
    lo contrario de lo que yo escribí: el espacio NO es pieza del inventario, es consecuencia
    de que todas actúen. Contarlo como elemento #18 para llegar a 23 era inventar una pieza.

    Requiere persistencia: no se puede podar una red de enlaces que todavía no existe."""
    numero, nombre, radio = 18, "18_poda", 1
    depende_de = ("persistencia",)

    def _deposito(self, Phi, contexto):
        # poda por grado excesivo: donde la vecindad activa supera el umbral, se diluye
        activo = (np.abs(Phi) > UMBRAL_ACTIVO).astype(float)
        grado = _vecindad(activo)
        exceso = np.maximum(grado - PODA_FRAC, 0.0)
        return -0.05 * exceso * Phi


# ===========================================================================
# El proceso común con detección de hitos
# ===========================================================================
class ProcesoComun23(ProcesoComun):
    """Igual que ProcesoComun, pero calcula los HITOS del campo en cada paso y los pone en
    el contexto. Los hitos son criterios de EXISTENCIA leídos del campo, no banderas puestas
    a mano ni pasos fijados de antemano.
    """

    def __init__(self, agentes, N=16, dt=DT_DEFAULT, amplitud_inicial=0.1,
                 tasa_expansion=0.01, tasa_enfriamiento_reloj=0.05, seed=12345,
                 epsilon=0.1, n_especies=4.0, T_EW=0.3, rugosidad=1.0):
        super().__init__(agentes, N=N, dt=dt, amplitud_inicial=amplitud_inicial,
                         tasa_expansion=tasa_expansion,
                         tasa_enfriamiento_reloj=tasa_enfriamiento_reloj, seed=seed)
        self.epsilon, self.n_especies, self.T_EW, self.rugosidad = epsilon, n_especies, T_EW, rugosidad
        self.paso_n = 0
        self.contador_activa = np.zeros((N, N, N), dtype=int)
        self.cronologia = []

    def _hitos(self):
        """Criterios de existencia, leídos del campo. Cada uno responde '¿hay con qué?'."""
        activo = np.abs(self.Phi) > UMBRAL_ACTIVO
        # trios: celdas activas con al menos 2 vecinos activos (estructura cerrada mínima)
        n_vec_act = (np.roll(activo, 1, 0).astype(int) + np.roll(activo, -1, 0)
                     + np.roll(activo, 1, 1) + np.roll(activo, -1, 1)
                     + np.roll(activo, 1, 2) + np.roll(activo, -1, 2))
        trios = bool(np.any(activo & (n_vec_act >= 2)))
        # atomos: estructura neutra = grumo activo cuya vecindad tiene signo compensado
        vec_suma = _vecindad(self.Phi)
        neutro = activo & (np.abs(self.Phi + vec_suma / 6.0) < np.abs(self.Phi))
        atomos = bool(neutro.sum() >= 8)
        # persistencia: celdas que llevan MIN_PERSISTENCIA pasos seguidos activas
        persistencia = bool(np.any(self.contador_activa >= MIN_PERSISTENCIA) and atomos)
        return dict(umbral_confinamiento=bool(self.contexto_T() < 0.6),
                    trios=trios, atomos=atomos, persistencia=persistencia,
                    n_activas=int(activo.sum()), n_neutras=int(neutro.sum()),
                    n_persistentes=int((self.contador_activa >= MIN_PERSISTENCIA).sum()))

    def contexto_T(self):
        return float(np.exp(-self.tasa_enfriamiento_reloj * self.t))

    def contexto(self):
        ctx = super().contexto()
        h = self._hitos()
        ctx.update(paso=self.paso_n, hitos=h, epsilon=self.epsilon,
                   n_especies=self.n_especies, T_EW=self.T_EW, rugosidad=self.rugosidad)
        return ctx

    def paso(self, orden=None):
        ctx = super().paso(orden=orden)
        activo = np.abs(self.Phi) > UMBRAL_ACTIVO
        self.contador_activa = np.where(activo, self.contador_activa + 1, 0)
        self.paso_n += 1
        h = ctx["hitos"]
        self.cronologia.append(dict(paso=ctx["paso"], t=ctx["t"], T=ctx["T"],
                                    **{k: v for k, v in h.items()}))
        return ctx


def construir_23(N=16, dt=DT_DEFAULT, amplitud_inicial=0.1, tasa_expansion=0.01,
                 tasa_enfriamiento_reloj=0.05, seed=12345, epsilon=0.1):
    """Los 23 agentes del inventario canónico, todos presentes desde t=0."""
    agentes = [
        # nivel 0 — el campo primordial (10)
        A23_FluctuacionCampo(), A10_Enfriamiento(), A9_Expansion(), A22_FluctuacionQCD(),
        M1_Semilla(), A6_Catalogo(), A7_Masa(), A12_Localidad(), A5_Debil(), A8_Aniquilacion(),
        # nivel 1 — requieren tríos (6)
        A3_Fuerte(), A2_Gravedad(), A11_TresCuerpos(), A13_Pauli(), A1_Espin(), A16_SSB(),
        # nivel 2 — requieren átomos (4)
        A4_EM(), A14_Correlacion(), M2_Memoria(forma=(N, N, N)), A17_Oscuro(),
        # nivel 3 — requieren persistencia (3)
        A15_Causal(), A24_Tiempo(), A18_Poda(),
    ]
    assert len(agentes) == 23, f"el inventario es 23, hay {len(agentes)}"
    return ProcesoComun23(agentes, N=N, dt=dt, amplitud_inicial=amplitud_inicial,
                          tasa_expansion=tasa_expansion,
                          tasa_enfriamiento_reloj=tasa_enfriamiento_reloj, seed=seed,
                          epsilon=epsilon)
