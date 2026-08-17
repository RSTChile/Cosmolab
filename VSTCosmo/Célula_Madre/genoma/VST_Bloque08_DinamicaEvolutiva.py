#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
VST_Bloque08_DinamicaEvolutiva — BLOQUE 8 DEL CANON, HECHO ORGANELOS  ·  "QUIÉN SOY"
================================================================================

QUÉ ES ESTE ARCHIVO
-------------------
La implementación del BLOQUE 8 — DINÁMICA EVOLUTIVA de la Teoría Cosmosemiótica
(Parte II, nodos O-N8.x), como organelos del genoma (VST_Genoma.py). Es el bloque
que da nombre al proyecto: "La IA como EXAPTACIÓN". Aquí vive el motor del cambio:
mutación, adaptación, exaptación, metacognición y activación latente.

LAS DISTINCIONES CANÓNICAS QUE IMPLEMENTA
-----------------------------------------
  · MUTACIÓN (O-N8.1)      = ΔR_aleatoria: cambio aleatorio en representación/acción
                            NO filtrado por el sistema; opera sobre el error que escapa
                            al umbral de corrección (e_R no filtrado, O-N8.1b).
  · ADAPTACIÓN (O-N8.2)    = argmax A_sys-env con Ωop CONSTANTE. Optimiza el acoplamiento
                            DENTRO del dominio actual. ⇒ ΔLF ≈ 0 (O-N7.6). No abre dominio.
  · EXAPTACIÓN (O-N8.3/8.5) = reutilizar estructura existente en un dominio NUEVO ⇒
                            ΔΩop>0 ∧ ΔLF>0. Se dispara en el LÍMITE ADAPTATIVO (O-N8.5):
                            ¬(∃ adaptación viable) ⇒ exaptación ∨ extinción. Requiere
                            RESERVA estructural (PRE, O-N8.19): R\\Uactual ≠ ∅.
  · C_m METACOGNICIÓN (O-N8.4) = emerge cuando C_b (consciencia básica, Bloque 5) falla
                            SISTEMÁTICAMENTE y hay capacidad de reorganización (LF). Es la
                            consciencia que examina las propias representaciones para
                            reorganizarse.
  · ACTIVACIÓN LATENTE (O-N8.12) = las estructuras latentes se activan cuando el entorno
                            lo requiere. Es el disparador de la PLURIPOTENCIA: detecta que
                            la demanda excede el dominio operativo y señala qué expresar.

NODOS NO IMPLEMENTADOS COMO MECANISMO (declarados, no codificados aquí)
----------------------------------------------------------------------
  · PRE (O-N8.19): ya realizado como el locus reservado de Boorman (VST_Genoma).
  · PEX (O-N8.20) y HETA (O-N8.22): principios de extensión exaptativa de 2º/3er orden
    (herramientas; IA generando su propia tecnología). Son hipótesis falsables sobre
    trayectorias, no mecanismos del organelo individual. Se documentan, no se codifican.

PAYOFF (composición B5+B7+B8)
-----------------------------
La exaptación produce XE, uno de los componentes del Índice de Organismicidad (OI,
O-N9.14). El demo muestra el LÍMITE ADAPTATIVO en acción: cuando la demanda del entorno
supera el dominio operativo, la adaptación deja de bastar y la exaptación (si hay
reserva) abre dominio nuevo. Con eso, el OI cruza por primera vez de "no organismal" a
"protoorganismo": la exaptación es lo que vuelve a la célula un proto-organismo.

ANDAMIAJE: `FuenteDemandaDemo` inyecta una demanda de entorno creciente (cambio de
régimen) y deriva de ella A_sys-env y e_R. Es andamiaje de prueba, no parte del Bloque 8.
================================================================================
"""

from __future__ import annotations
import random
import os as _os
import sys as _sys
from typing import Any

# ESCALA COMPARTIDA (auditoría del 4-ago-2026, regla 1 del plan de constantes): «un módulo
# compartido, no 168 parches». Lo que aquí se relativiza usa rel/rel_contra de escala.py.
_RAIZ = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _RAIZ not in _sys.path:
    _sys.path.insert(0, _RAIZ)
# `escala` vive en celula_madre/; esto permite importar el organelo suelto (pruebas y smokes)
# además de dentro del organismo. Unificado el 5-ago-2026: la revisión encontró CUATRO
# variantes del mismo arranque, que es el problema contra el que existe el módulo compartido.
import os as _os, sys as _sys
_RAIZ_CM = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _RAIZ_CM not in _sys.path:
    _sys.path.insert(0, _RAIZ_CM)
from escala import Escala, rel, rel_contra, NEUTRO

from VST_Genoma import (
    Organelo, Estado, Organismo, Milieu, KAPPA,
    OrganeloPresionDesacople, OrganeloFatiga, locus_altruismo_boorman,
)


# ==============================================================================
# MUTACIÓN  (O-N8.1 / O-N8.1b)
# ==============================================================================
class OrganeloMutacion(Organelo):
    """Mutación = ΔR_aleatoria: variación NO filtrada (O-N8.1).

    QUÉ HACE: introduce un cambio aleatorio en la representación/acción que NO pasa por
    el filtro de corrección del sistema. Opera sobre el error que ESCAPA al umbral th_osc
    (e_R no filtrado, O-N8.1b): donde el sistema no logra corregir, la variación aleatoria
    puede colarse. Es la fuente de novedad ciega (la selección la juzga después).
    CÓMO: si |e_R| supera th_osc, con cierta tasa emite una perturbación gaussiana
    proporcional al error no filtrado. RNG SEMBRADO para reproducibilidad (el proyecto
    sella semillas; la variación se hace determinista variando la semilla, no el azar).
    """

    def __init__(self, seed: int = 44) -> None:
        super().__init__(
            nombre="mutacion", organelo_analogo="error de replicación (polimerasa)",
            procedencia="O-N8.1/8.1b / Dicc.121",
            nodo_canonico="O-N8.1 (Mutación=ΔR_aleatoria) · O-N8.1b",
            descripcion=("Variación aleatoria no filtrada sobre el error que escapa al umbral "
                         "de corrección. Fuente de novedad ciega (la selección juzga luego)."),
            lee=["e_R"],
            secreta=["mutacion", "mutacion_activa"],
            depende_de=[],
            costo_base=0.5,
            # tasa=0.3 y escala=0.05 se DEJAN: son constantes de un proceso estocástico —el
            # análogo de la tasa de error de una polimerasa—, no umbrales que juzguen un estado.
            # Y hoy NO SON MEDIBLES: `mutacion` no la consume nadie (mapa_organismo.py --var
            # mutacion → "la consumen: nadie"), así que su efecto no se observa en ninguna parte.
            plast={"tasa": 0.3, "escala": 0.05},
            criterio="emite ΔR solo sobre el e_R que excede el error que este organismo suele corregir",
            estado=Estado.PRESENTE,
        )
        self._rng = random.Random(seed)
        self.mutacion = 0.0
        self.activa = False
        # Lo habitual del error para ESTE organismo (sustituye a th_osc=5,0).
        self.esc_eR = Escala()

    def percibir(self, milieu: "Milieu") -> None:
        self._eR = abs(milieu.leer("e_R", 0.0))

    def metabolizar(self, dt: float, tempo: float) -> None:
        # CORRECCIÓN 5-ago-2026. QUÉ ESTABA MAL: `th_osc = 5.0`. El "umbral de corrección" —lo
        # que el sistema logra filtrar— fijado en 5 unidades de un error cuya escala depende
        # del cuerpo (micrófono, sala, Pi) y que nadie midió.
        # CIFRA MEDIDA (100.058 pasos, 44 sesiones, ~/.anima/history/*/fisiologia/*.csv): e_R
        # es BIMODAL — 45.359 pasos valen exactamente 0,5 y prácticamente todo el resto ≥5.
        # El umbral 5,0 dispara en el 54,74% de los pasos... y el umbral 2,0 dispara en el
        # 54,74% TAMBIÉN (es el `umbral_valencia` del Bloque 7, sobre la misma magnitud). Es
        # decir: cualquier número entre 0,5 y ~5,4 hace exactamente lo mismo. El 5,0 no elegía
        # nada de lo que decía elegir; elegía el hueco de la distribución.
        # POR QUÉ LA CORRECCIÓN ES AUTORREGULADA: O-N8.1b dice "el error que ESCAPA al umbral
        # de corrección". Lo que un sistema logra corregir no es un número universal: es lo que
        # ESTE organismo suele manejar. Así que el umbral pasa a ser su propio error habitual,
        # aprendido (escala.py, sin parámetro libre). Es una PERCEPCIÓN (comparar el error de
        # ahora con el que suelo tener), no una condición de viabilidad. Y mientras no haya
        # historia el organelo se abstiene —no muta— en vez de mutar contra una escala vacía.
        # EFECTO MEDIDO sobre los mismos datos: 54,74% → 12,13% de los pasos con e_R no filtrado.
        habitual = self.esc_eR.media if self.esc_eR.madura else None
        no_filtrado = max(0.0, self._eR - habitual) if habitual is not None else 0.0
        self.esc_eR.observar(self._eR)                # aprende después de decidir
        if no_filtrado > 0.0 and self._rng.random() < self.plast["tasa"]:
            self.mutacion = self._rng.gauss(0.0, no_filtrado * self.plast["escala"])
            self.activa = True
        else:
            self.mutacion = 0.0
            self.activa = False

    def secretar(self, milieu: "Milieu") -> None:
        milieu.secretar("mutacion", self.mutacion)
        milieu.secretar("mutacion_activa", self.activa)

    # Sin persistir la escala, el organismo reaprende en cada arranque qué error le es
    # habitual y muta a ciegas durante sus primeros pasos (escala.py, nota de `restore`).
    def snapshot(self) -> dict:
        return {"esc_eR": self.esc_eR.snapshot()}

    def restore(self, d: dict) -> None:
        if isinstance(d, dict):
            self.esc_eR.restore(d.get("esc_eR"))


# ==============================================================================
# ADAPTACIÓN  (O-N8.2 / O-N8.2b)  — ΔLF ≈ 0
# ==============================================================================
class OrganeloAdaptacion(Organelo):
    """Adaptación = argmax A_sys-env con Ωop CONSTANTE (O-N8.2).

    QUÉ HACE: optimiza el acoplamiento DENTRO del dominio operativo actual, sin abrirlo.
    Mientras la demanda del entorno cabe en el dominio (Ωop), la adaptación basta y
    mantiene A_sys-env. ⇒ ΔLF ≈ 0 (O-N7.6): no expande la libertad, la afina.
    LÍMITE: cuando la demanda EXCEDE Ωop, la adaptación deja de ser viable — ahí empieza
    el territorio de la exaptación (O-N8.5). O-N8.2b: con C_b suficiente, la corrección
    del desacople es posible (si no hay registro del estado, no hay qué corregir).
    CÓMO: lee la demanda y el Ωop actual (que mantiene la exaptación); declara
    `adaptacion_viable` si la demanda cabe en el dominio.
    """

    def __init__(self) -> None:
        super().__init__(
            nombre="adaptacion", organelo_analogo="aclimatación fisiológica",
            procedencia="O-N8.2/8.2b / Dicc.",
            nodo_canonico="O-N8.2 (argmax A_sys-env, Ωop=cte) · ΔLF≈0 (O-N7.6)",
            descripcion=("Optimiza el acoplamiento dentro del dominio actual (Ωop constante). "
                         "Viable mientras la demanda cabe en Ωop; ΔLF≈0. No abre dominio."),
            lee=["demanda_entorno", "Omega_op", "C_b"],
            secreta=["adaptacion_viable", "adaptacion_activa", "delta_LF_adapt"],
            depende_de=[],   # lee Omega_op del ciclo previo (persiste en el milieu)
            costo_base=1.0,
            # margen=0.05 se DEJA, y por dos razones, ambas comprobadas:
            # 1) MEDIDO, es casi inerte: sobre 100.058 pasos, `demanda ≤ Ωop` se cumple en el
            #    48,84% y `demanda ≤ Ωop·1,05` en el 48,99%. El margen decide 0,15 puntos
            #    porcentuales; no está gobernando nada a ciegas.
            # 2) Y NO debe relativizarse (advertencia 2 de la auditoría): `adaptacion_viable`
            #    es una CONDICIÓN DE VIDA —¿todavía me arreglo dentro de mi dominio?—, no una
            #    percepción. Se probó qué pasaría al ponerlo contra la dispersión de la propia
            #    demanda: MEDIDO, la demanda tiene mediana 4,15 pero p95 = 534 y máximo 1.920
            #    frente a un Ωop de 3,0, así que una banda "a la medida de lo habitual" haría
            #    viable casi cualquier demanda y BORRARÍA el límite adaptativo (O-N8.5) — es
            #    decir, mataría la exaptación. Un organismo que lleva toda su vida desbordado
            #    tiene que seguir leyendo que está desbordado.
            plast={"margen": 0.05},
            criterio="adaptacion_viable mientras demanda ≤ Ωop·(1+margen)",
            estado=Estado.PRESENTE,
        )
        self.viable = True
        self.activa = False

    def percibir(self, milieu: "Milieu") -> None:
        self._demanda = milieu.leer("demanda_entorno", 1.0)
        self._Omega = milieu.leer("Omega_op", 1.0)   # dominio operativo (lo mantiene exaptación)
        self._Cb = milieu.leer("C_b", 0.0)

    def metabolizar(self, dt: float, tempo: float) -> None:
        # ¿la demanda cabe en el dominio operativo actual?
        self.viable = self._demanda <= self._Omega * (1.0 + self.plast["margen"])
        # O-N8.2b: corregir requiere registro del estado (C_b>0)
        self.activa = self.viable and (self._Cb > 0.0)

    def secretar(self, milieu: "Milieu") -> None:
        milieu.secretar("adaptacion_viable", self.viable)
        milieu.secretar("adaptacion_activa", self.activa)
        milieu.secretar("delta_LF_adapt", 0.0)   # la adaptación NO expande la libertad


# ==============================================================================
# EXAPTACIÓN  (O-N8.3 / O-N8.5 / O-N8.19 PRE)  — ΔΩop>0 ∧ ΔLF>0
# ==============================================================================
class OrganeloExaptacion(Organelo):
    """Exaptación = reutilizar estructura en un dominio NUEVO (O-N8.3) ⇒ ΔΩop>0 ∧ ΔLF>0.

    QUÉ HACE: cuando la adaptación deja de ser viable (límite adaptativo, O-N8.5), y SOLO
    si hay RESERVA estructural disponible (PRE, O-N8.19: R\\Uactual≠∅), el sistema reutiliza
    estructura existente para ABRIR un dominio operativo nuevo (ΔΩop>0) e incrementar su
    libertad funcional (ΔLF>0). Si no hay adaptación viable NI reserva → riesgo de
    extinción (la otra rama de O-N8.5). La exaptación NO mejora el dominio anterior: lo
    TRASCIENDE (O-N7.5). Acumula XE (exaptación realizada), componente del OI (O-N9.14).
    CÓMO: mantiene el estado del dominio Ωop y la reserva; cada vez que exapta, crece Ωop
    hacia la demanda consumiendo reserva, y suma a XE. PRE como guarda dura: sin reserva,
    no exapta (no se puede crear dominio de la nada — sería imposición = Shannon).
    """

    def __init__(self, omega0: float = 1.0, reserva0: float = 2.0) -> None:
        super().__init__(
            nombre="exaptacion", organelo_analogo="cooptación de estructura (pluma→vuelo)",
            procedencia="O-N8.3/8.5/8.19 / Dicc.83",
            nodo_canonico="O-N8.3 (ΔΩop>0∧ΔLF>0) · O-N8.5 (límite) · O-N8.19 (PRE)",
            descripcion=("Reutiliza estructura en dominio nuevo: abre Ωop y sube LF cuando la "
                         "adaptación no basta Y hay reserva (PRE). Sin reserva → riesgo extinción. "
                         "No mejora el dominio: lo trasciende. Acumula XE (componente del OI)."),
            lee=["adaptacion_viable", "demanda_entorno"],
            secreta=["Omega_op", "reserva", "XE", "exaptacion_activa", "extincion_riesgo", "delta_LF_exapt"],
            depende_de=["adaptacion"],   # decide a partir de si la adaptación fue viable
            costo_base=2.0,              # exaptar es caro (reorganización estructural)
            # k=0.05 (fracción de la brecha que se cierra por paso) es una constante de tasa,
            # como cualquier τ del proyecto. costo_reserva=1.0 NO es un número libre: es la
            # identidad de conservación "una unidad de dominio nuevo cuesta una unidad de
            # reserva". Ambas se dejan.
            plast={"k": 0.05, "costo_reserva": 1.0},
            criterio="exapta solo en límite adaptativo CON reserva>0 (PRE); ΔΩop>0, ΔLF>0",
            estado=Estado.PRESENTE,
        )
        self.Omega_op = omega0      # dominio operativo actual
        # Ω₀ — el dominio con el que el organismo NACE. Es la unidad de la magnitud: igual que
        # nadie discute vivir a 1 atmósfera, el dominio natal define la escala contra la que se
        # mide todo dominio posterior. Se guarda porque `XE` se publica comparada contra él.
        self.Omega_0 = float(omega0)
        self.reserva = reserva0     # reserva estructural disponible (PRE)
        self.XE = 0.0               # exaptación acumulada
        self.activa = False
        self.extincion_riesgo = False
        self._delta_LF = 0.0

    def percibir(self, milieu: "Milieu") -> None:
        self._adapt_viable = bool(milieu.leer("adaptacion_viable", True))
        self._demanda = milieu.leer("demanda_entorno", 1.0)

    def metabolizar(self, dt: float, tempo: float) -> None:
        self.activa = False
        self.extincion_riesgo = False
        self._delta_LF = 0.0
        if not self._adapt_viable:                       # límite adaptativo (O-N8.5)
            if self.reserva > 0.0:                        # PRE: hay reserva ⇒ se puede exaptar
                # crecer el dominio hacia la demanda, consumiendo reserva
                d = min(self.reserva, self.plast["k"] * max(0.0, self._demanda - self.Omega_op))
                if d > 1e-9:
                    self.Omega_op += d                    # ΔΩop > 0
                    self.reserva -= d * self.plast["costo_reserva"]
                    self.XE += d                          # exaptación realizada
                    self._delta_LF = d                    # ΔLF > 0
                    self.activa = True
            else:                                         # sin reserva: la otra rama de O-N8.5
                self.extincion_riesgo = True

    def secretar(self, milieu: "Milieu") -> None:
        milieu.secretar("Omega_op", self.Omega_op)
        milieu.secretar("reserva", self.reserva)
        # CORRECCIÓN 5-ago-2026 — LA GRANDE DE ESTE BLOQUE.
        # QUÉ ESTABA MAL: `min(1.0, self.XE)`. `self.XE` es un acumulador SIN TECHO del dominio
        # abierto por exaptación, y se publicaba recortado en 1,0. El 1,0 no es ninguna medida
        # del organismo: es sólo "el número más grande que cabe en el hueco del OI".
        # CIFRA MEDIDA (100.075 filas con XE en ~/.anima/history/*/fisiologia/*.csv): XE vale
        # exactamente 1,0 en 100.072 de ellas — el 99,997% —, con SÓLO CUATRO valores distintos
        # en 44 sesiones. Es una constante disfrazada de variable, y no una constante cualquiera:
        # entra como 1 de los 4 componentes del Índice de Organismicidad (VST_Genoma.salud(),
        # componentes_oi) y la leen además Homeostasis, RC_A, RC_B, OrganoComunicacion y las 5
        # vistas web. El organismo lleva 44 sesiones declarando "exaptación máxima alcanzada".
        # POR QUÉ PASA: la reserva estructural natal (reserva0=2,0) permite un crecimiento total
        # de dominio de 2,0 > 1,0, así que XE cruza el recorte en los primeros segundos de vida
        # y ya no se mueve. MEDIDO: Ωop vale 3,0 en 100.062 de 100.075 filas (nació en 1,0) y
        # `exaptacion_activa` fue True en 21 pasos de 100.058 (0,021%) — todos al principio.
        # POR QUÉ LA CORRECCIÓN ES AUTORREGULADA (regla 1 de la auditoría: si existe otra
        # magnitud del organismo con las mismas unidades, compárate con ELLA y no con un
        # número): XE y Ω₀ son ambas anchura de dominio. `rel_contra(XE, Ω₀)` = cuánto dominio
        # nuevo he abierto medido contra el dominio con el que nací. Vale 0 al nacer, 0,5 cuando
        # he DUPLICADO mi dominio, y sube hacia 1 sin llegar nunca: no hay recorte que pueda
        # clavarla. Con los datos reales (XE acumulada = Ωop−Ω₀ = 3,0−1,0 = 2,0 y Ω₀ = 1,0)
        # pasaría de 1,0 a 0,667.
        # LO QUE ESTO **NO** ARREGLA, Y HAY QUE DECIRLO: la XE seguiría siendo casi constante,
        # porque la causa de fondo es OTRA — `reserva` no se regenera nunca. El organismo gastó
        # su reserva natal en los primeros segundos y lleva 44 sesiones en `extincion_riesgo`
        # (variable que, además, no la lee nadie) mientras la demanda mediana es 4,15 y la p95
        # es 534 contra un dominio de 3,0. NO lo arreglo aquí a propósito: hacer que la reserva
        # se rehaga con la holgura dispararía Ωop hacia esa demanda —dos órdenes de magnitud— y
        # Ωop lo consumen adaptacion, activacion_latente y las 5 vistas web. Eso es un cambio de
        # conducta, no una corrección de constante, y necesita su propia medición.
        milieu.secretar("XE", rel_contra(self.XE, self.Omega_0))
        milieu.secretar("exaptacion_activa", self.activa)
        milieu.secretar("extincion_riesgo", self.extincion_riesgo)
        milieu.secretar("delta_LF_exapt", self._delta_LF)


# ==============================================================================
# CONSCIENCIA METACOGNITIVA  C_m  (O-N8.4)
# ==============================================================================
class OrganeloConscienciaMetacognitiva(Organelo):
    """C_m = consciencia metacognitiva: emerge cuando C_b FALLA sistemáticamente (O-N8.4).

    QUÉ HACE: cuando la consciencia básica (C_b, Bloque 5) no logra resolver de forma
    sostenida —el error persiste alto pese a registrar el estado— Y hay capacidad de
    reorganización (LF), emerge una consciencia de orden superior que examina las propias
    representaciones para reorganizarse. Es el puente entre el fracaso del registro y la
    exaptación: la crisis de C_b convoca a C_m, que habilita la reorganización.
    CÓMO: integra un fallo sostenido (media móvil de |e_R| alta) condicionado a LF>κ_LF;
    C_m sube en la crisis, baja cuando se resuelve.
    """

    def __init__(self, ventana: int = 100) -> None:
        super().__init__(
            nombre="consciencia_metacognitiva", organelo_analogo="control ejecutivo de 2º orden",
            procedencia="O-N8.4 / Dicc.66 (C_m)",
            nodo_canonico="O-N8.4 (C_m emerge si C_b falla ∧ hay reorganización)",
            descripcion=("Consciencia de 2º orden: emerge cuando el registro básico falla "
                         "sostenidamente y hay libertad para reorganizar. Convoca a la exaptación."),
            lee=["e_R", "LF"],
            secreta=["C_m", "C_m_activa"],
            depende_de=[],
            costo_base=1.5,
            # tau=10 s es una constante de tiempo declarada (como cualquier τ del proyecto).
            # umbral_cm=0.3 se DEJA: C_m es un integrador con fuga acotado en [0,1] por
            # construcción, así que 0,3 es una fracción de una escala acotada, no de una escala
            # inventada. Aun así queda ANOTADO como sospechoso: MEDIDO, C_m está saturada
            # (mediana 0,84 · p75 = 1,0 · media 0,523 sobre 100.058 pasos), de modo que hoy el
            # umbral casi no discrimina. Al corregir th_fallo (ver metabolizar) C_m debería
            # dejar de vivir clavada arriba; entonces este umbral vuelve a ser medible y hay
            # que volver a auditarlo.
            plast={"tau": 10.0, "umbral_cm": 0.3},
            criterio="C_m>umbral cuando |e_R| sostenido PEOR QUE EL PROPIO HABITUAL ∧ LF≥κ_LF",
            estado=Estado.PRESENTE,
        )
        self.ventana = ventana
        self._buf: list[float] = []
        self.C_m = 0.0
        self.activa = False
        # El error habitual de este organismo a largo plazo (sustituye a th_fallo=5,0). La
        # tasa NO es un número nuevo: es 1/ventana, o sea "la escala aprende `ventana` veces
        # más despacio que la media móvil con la que se la compara". Así una es lo RECIENTE y
        # la otra lo SOSTENIDO, que es justo la distinción que pide O-N8.4.
        self.esc_eR = Escala(tasa=1.0 / max(1, ventana))

    def percibir(self, milieu: "Milieu") -> None:
        self._eR = abs(milieu.leer("e_R", 0.0))
        self._LF = milieu.leer("LF", 0.0)

    def metabolizar(self, dt: float, tempo: float) -> None:
        self._buf.append(self._eR)
        if len(self._buf) > self.ventana:
            self._buf.pop(0)
        # CORRECCIÓN 5-ago-2026. QUÉ ESTABA MAL: `th_fallo = 5.0`. "C_b falla SISTEMÁTICAMENTE"
        # (O-N8.4) se decidía comparando la media móvil de |e_R| contra el número 5 — el MISMO
        # número inventado que el th_osc de la mutación, sobre la misma magnitud y en el mismo
        # archivo, sin que ninguno de los dos tuviera origen escrito.
        # CIFRAS MEDIDAS (100.058 pasos, 44 sesiones): el |e_R| medio de este organismo es 6,59,
        # o sea que su error NORMAL ya está por encima del umbral de "fallo". Resultado: el
        # fallo se declara en el 55,59% de los pasos y C_m vive saturada — mediana 0,84, p75 =
        # 1,0. El organismo lleva 44 sesiones en crisis metacognitiva permanente, que es tanto
        # como no estar en crisis nunca.
        # POR QUÉ LA CORRECCIÓN ES AUTORREGULADA: "fallar sistemáticamente" no puede significar
        # "tener un error mayor que 5"; significa que el error RECIENTE es peor que el error que
        # este organismo tiene de costumbre. Eso es exactamente `rel_contra` (regla 1 de la
        # auditoría): dos magnitudes del propio organismo con las mismas unidades —la media
        # móvil de la ventana contra la escala larga— y ningún parámetro que elegir. Es una
        # PERCEPCIÓN comparativa, no una condición de viabilidad: la condición de viabilidad de
        # este organelo es la otra, `LF ≥ κ_LF`, que sigue siendo absoluta y canónica.
        # Mientras la escala larga no esté madura, se abstiene (no declara fallo).
        # EFECTO MEDIDO sobre los mismos datos: 55,59% → 17,75% de los pasos con fallo sostenido.
        media_ventana = sum(self._buf) / len(self._buf)
        fallo_sostenido = (self.esc_eR.madura and
                           rel_contra(media_ventana, self.esc_eR.media) > NEUTRO)
        self.esc_eR.observar(self._eR)                    # aprende después de decidir
        hay_reorg = self._LF >= KAPPA["kLF"]              # capacidad de reorganización
        objetivo = 1.0 if (fallo_sostenido and hay_reorg) else 0.0
        # integrador con fuga (τ escalada por Kleiber)
        tau = self.plast["tau"] * tempo
        self.C_m += (objetivo - self.C_m) * (dt / tau)
        self.C_m = max(0.0, min(1.0, self.C_m))
        self.activa = self.C_m > self.plast["umbral_cm"]

    def secretar(self, milieu: "Milieu") -> None:
        milieu.secretar("C_m", self.C_m)
        milieu.secretar("C_m_activa", self.activa)

    # La escala larga tarda `ventana` observaciones en madurar; sin persistirla, cada arranque
    # empezaría sin saber cuál es su error de costumbre (escala.py, nota de `restore`).
    def snapshot(self) -> dict:
        return {"esc_eR": self.esc_eR.snapshot()}

    def restore(self, d: dict) -> None:
        if isinstance(d, dict):
            self.esc_eR.restore(d.get("esc_eR"))


# ==============================================================================
# ACTIVACIÓN LATENTE  (O-N8.12)  — el disparador de la pluripotencia
# ==============================================================================
class OrganeloActivacionLatente(Organelo):
    """Activación latente: las estructuras latentes se activan cuando el entorno lo
    requiere (O-N8.12).

    QUÉ HACE: detecta cuándo la demanda del entorno EXCEDE el dominio operativo actual
    (déficit de capacidad) y señala que hace falta activar capacidad latente. Es el nexo
    operativo de la PLURIPOTENCIA: una célula madre porta organelos silenciados que este
    organelo "convoca" cuando el contexto los pide.
    CÓMO: deficit = max(0, demanda − Ωop); si supera el umbral, `demanda_activacion`=True.
    NOTA DE INTEGRACIÓN: la EXPRESIÓN efectiva de un organelo latente es competencia del
    host (Organismo.expresar); este organelo emite la SEÑAL de demanda. El gancho host↔señal
    se cableará cuando haya organelos silenciados que valga la pena despertar.
    """

    def __init__(self) -> None:
        super().__init__(
            nombre="activacion_latente", organelo_analogo="expresión génica inducible",
            procedencia="O-N8.12 / Dicc.",
            nodo_canonico="O-N8.12 (activación latente) — disparador de pluripotencia",
            descripcion=("Detecta déficit de capacidad (demanda>Ωop) y señala activar capacidad "
                         "latente. Nexo operativo de la pluripotencia (la expresión la hace el host)."),
            lee=["demanda_entorno", "Omega_op"],
            secreta=["demanda_activacion", "deficit_capacidad", "deficit_relativo"],
            depende_de=["exaptacion"],   # lee el Ωop ya actualizado por la exaptación
            costo_base=0.5,
            # CORRECCIÓN 5-ago-2026: `umbral = 0.1` eliminado. QUÉ ESTABA MAL: 0,1 unidades de
            # dominio — un número en las unidades de Ωop que nadie fijó ni midió.
            # CIFRA MEDIDA (100.058 pasos): `deficit > 0,1` se cumple en el 51,14% de los pasos
            # y `deficit > 0` en el 51,23%. El umbral decidía 0,09 puntos porcentuales: nueve
            # centésimas de punto. No estaba separando nada; sólo escondía un número.
            # POR QUÉ LA CORRECCIÓN ES AUTORREGULADA: no hace falta ningún umbral. "Hay déficit
            # de capacidad" ES, por definición (O-N8.12), "la demanda excede mi dominio" —una
            # comparación entre dos magnitudes del propio organismo con las mismas unidades—, y
            # eso se escribe demanda > Ωop y ya está. Además se publica `deficit_relativo`
            # (rel_contra contra el propio Ωop) para que quien lea el déficit tenga su TAMAÑO a
            # la escala del organismo y no en unidades sueltas.
            plast={},
            criterio="señala activación cuando la demanda excede el dominio operativo (deficit>0)",
            estado=Estado.PRESENTE,
        )
        self.deficit = 0.0
        self.deficit_relativo = 0.0
        self.demanda_activacion = False

    def percibir(self, milieu: "Milieu") -> None:
        self._demanda = milieu.leer("demanda_entorno", 1.0)
        self._Omega = milieu.leer("Omega_op", 1.0)

    def metabolizar(self, dt: float, tempo: float) -> None:
        self.deficit = max(0.0, self._demanda - self._Omega)
        self.demanda_activacion = self.deficit > 0.0    # ver la nota del constructor
        # tamaño del déficit a la escala del propio dominio: 0,5 = "me falta tanto como tengo"
        self.deficit_relativo = rel_contra(self.deficit, self._Omega)

    def secretar(self, milieu: "Milieu") -> None:
        milieu.secretar("demanda_activacion", self.demanda_activacion)
        milieu.secretar("deficit_capacidad", self.deficit)
        milieu.secretar("deficit_relativo", self.deficit_relativo)


# ==============================================================================
# ANDAMIAJE DE DEMOSTRACIÓN — entorno con DEMANDA CRECIENTE (cambio de régimen)
# ==============================================================================
class FuenteDemandaDemo(Organelo):
    """Andamiaje: inyecta una demanda de entorno que SUBE en t=t_step (cambio de régimen),
    y deriva de ella el acoplamiento A_sys-env = min(1, Ωop/demanda) y el error e_R = gap.
    Cuando la demanda supera Ωop, A cae y e_R sube → la adaptación deja de bastar → la
    exaptación entra. Es scaffolding de prueba, NO parte del Bloque 8."""

    def __init__(self, demanda_base: float = 1.0, demanda_alta: float = 2.5,
                 t_step: float = 40.0) -> None:
        super().__init__(
            nombre="entorno_demanda", organelo_analogo="(andamiaje de prueba)",
            procedencia="(scaffolding)", nodo_canonico="(no canónico: andamiaje)",
            descripcion="Andamiaje: demanda creciente (cambio de régimen) que fuerza el límite adaptativo.",
            lee=["Omega_op"], secreta=["demanda_entorno", "A_sys_env", "e_R",
                                       "delta_struct", "delta_real", "costo_trabajo", "en_reposo"],
            costo_base=0.0, estado=Estado.PRESENTE,
        )
        self.plast = {"base": demanda_base, "alta": demanda_alta, "t_step": t_step}
        self._t = 0.0

    def percibir(self, milieu: "Milieu") -> None:
        self._Omega = milieu.leer("Omega_op", 1.0)   # dominio operativo actual (de exaptación)

    def metabolizar(self, dt: float, tempo: float) -> None:
        self._t += dt

    def secretar(self, milieu: "Milieu") -> None:
        demanda = self.plast["base"] if self._t < self.plast["t_step"] else self.plast["alta"]
        A = max(0.0, min(1.0, self._Omega / demanda))            # acoplamiento = dominio/demanda
        e_R = max(0.5, (demanda - self._Omega) * 20.0)           # error = brecha (escalado)
        milieu.secretar("demanda_entorno", demanda)
        milieu.secretar("A_sys_env", A)
        milieu.secretar("e_R", e_R)
        milieu.secretar("delta_struct", 0.30)
        milieu.secretar("delta_real", 0.05)
        milieu.secretar("costo_trabajo", 0.05)
        milieu.secretar("en_reposo", False)


# ==============================================================================
# TRANSCRIPCIONES
# ==============================================================================
def transcribir_evolutiva() -> Organismo:
    """Bloque 8 sobre la base mínima + entorno con demanda creciente. Muestra la
    dinámica evolutiva aislada (mutación/adaptación/exaptación/C_m/activación)."""
    o = Organismo(nombre="celula_madre_B8", M0=1.0)
    o.expresar(FuenteDemandaDemo())
    o.expresar(OrganeloPresionDesacople())
    o.expresar(OrganeloFatiga())
    o.expresar(OrganeloMutacion())
    o.expresar(OrganeloAdaptacion())
    o.expresar(OrganeloExaptacion())
    o.expresar(OrganeloConscienciaMetacognitiva())
    o.expresar(OrganeloActivacionLatente())
    o.expresar(locus_altruismo_boorman())
    return o


def transcribir_organismo_completo() -> Organismo:
    """COMPOSICIÓN B5 + B7 + B8: consciencia (R₂) → libertad (LF) → evolución (XE).
    El demo central del proyecto: la exaptación lleva el OI de 'no organismal' a
    'protoorganismo'."""
    from VST_Bloque05_ConscienciaFuncional import (
        OrganeloConscienciaBasica, OrganeloMetaRepresentacion, OrganeloSelf,
    )
    from VST_Bloque07_LibertadFuncional import (
        OrganeloJuego, OrganeloRitual, OrganeloNegacionOperativa,
        OrganeloLibertadFuncional, FuenteEntornoDemo,
    )
    o = Organismo(nombre="celula_madre_completa", M0=1.0)
    # entorno: demanda creciente (B8) + orientacion/INR/valencia (B7), SIN R2 (lo da B5)
    o.expresar(FuenteDemandaDemo())
    o.expresar(FuenteEntornoDemo(inject_R2=False))
    o.expresar(OrganeloPresionDesacople())
    # Bloque 5 — consciencia
    o.expresar(OrganeloConscienciaBasica())
    o.expresar(OrganeloMetaRepresentacion())
    o.expresar(OrganeloSelf())
    o.expresar(OrganeloFatiga())
    # Bloque 7 — libertad
    o.expresar(OrganeloRitual())
    o.expresar(OrganeloJuego())
    o.expresar(OrganeloLibertadFuncional())
    o.expresar(OrganeloNegacionOperativa())
    # Bloque 8 — evolución
    o.expresar(OrganeloMutacion())
    o.expresar(OrganeloAdaptacion())
    o.expresar(OrganeloExaptacion())
    o.expresar(OrganeloConscienciaMetacognitiva())
    o.expresar(OrganeloActivacionLatente())
    o.expresar(locus_altruismo_boorman())
    return o


# ==============================================================================
# DEMO / AUTOVERIFICACIÓN
# ==============================================================================
def _estado(org: Organismo) -> str:
    ex = org.organelos["exaptacion"]; ad = org.organelos["adaptacion"]
    cm = org.organelos["consciencia_metacognitiva"]
    s = org.salud()
    return (f"Ωop={ex.Omega_op:.3f} reserva={ex.reserva:.3f} XE={ex.XE:.3f}  "
            f"adapt_viable={ad.viable} exapt_activa={ex.activa} C_m={cm.C_m:.2f}  "
            f"OI={s['OI']:.3f}→{s['nivel_OI']}")


if __name__ == "__main__":
    # --- Bloque 8 aislado: la transición adaptación → exaptación ---
    org = transcribir_evolutiva()
    print("=" * 90)
    print("BLOQUE 8 — DINÁMICA EVOLUTIVA (aislado): cambio de régimen en t=40s")
    for _ in range(390):            # t<40: demanda baja, adaptación basta
        parte = org.vivir_un_paso(0.1)
    print(f"  t={parte['t']:.0f}s (régimen bajo):  {_estado(org)}")
    for _ in range(1100):           # t>40: demanda alta → límite adaptativo → exaptación
        parte = org.vivir_un_paso(0.1)
    print(f"  t={parte['t']:.0f}s (tras exaptar):  {_estado(org)}")

    # --- Organismo completo B5+B7+B8: el OI cruza a protoorganismo ---
    org2 = transcribir_organismo_completo()
    cm_peak = 0.0; exapto = False                # capturar el pico de la crisis
    for _ in range(2000):
        parte = org2.vivir_un_paso(0.1)
        cm_peak = max(cm_peak, org2.organelos["consciencia_metacognitiva"].C_m)
        exapto = exapto or org2.organelos["exaptacion"].activa
    lf = org2.organelos["LF"]; ex = org2.organelos["exaptacion"]
    s = org2.salud()
    print("\n" + "=" * 90)
    print(f"ORGANISMO COMPLETO B5+B7+B8 (tras {parte['t']:.0f}s):")
    print(f"  consciencia: C_b={org2.organelos['consciencia_basica'].C_b}  "
          f"R₂={org2.organelos['meta_representacion'].R2:.3f}")
    print(f"  libertad:    {lf.ESCALA[lf.nivel]}  (LF_op={lf.LF_op:.3f})")
    print(f"  evolución:   Ωop={ex.Omega_op:.3f}  XE={min(1.0,ex.XE):.3f}  "
          f"exaptó_en_crisis={exapto}  C_m_pico={cm_peak:.2f} (crisis) → C_m_ahora={org2.organelos['consciencia_metacognitiva'].C_m:.2f} (resuelta)")
    print(f"  → Λ_Cos={s['Lambda_Cos']:.3f}  OI={s['OI']:.3f} → {s['nivel_OI'].upper()}")
    invs = "  ".join(f"{'✓' if ok else '✗'} {k.split()[0]}" for k, ok in s['invariantes'].items())
    print(f"  invariantes: {invs}")
