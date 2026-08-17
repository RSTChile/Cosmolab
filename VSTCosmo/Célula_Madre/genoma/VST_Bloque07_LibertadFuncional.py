#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
VST_Bloque07_LibertadFuncional — BLOQUE 7 DEL CANON, HECHO ORGANELOS   ·  "QUIÉN SOY"
================================================================================

QUÉ ES ESTE ARCHIVO
-------------------
La implementación del BLOQUE 7 — LIBERTAD FUNCIONAL de la Teoría Cosmosemiótica
canónica (Parte II, nodos O-N7.x), como organelos enchufables al motor del genoma
(VST_Genoma.py). No reescribe el motor: importa el contrato `Organelo`, el `Milieu`
y el `Organismo`, y aporta los órganos de la libertad funcional.

QUÉ ES LA LIBERTAD FUNCIONAL (LF) — el corazón del bloque
--------------------------------------------------------
LF (O-N7.1) = la capacidad de un sistema de OPERAR SOBRE {dom(competencia) ≠
dom(operación)}: de interrumpir el reflejo Representación→Acción cuando el dominio
de operación excede o contradice su dominio de competencia. No CREA el desacople
(eso es Δ_struct, Bloque 0); opera SOBRE él. Se expresa en el uso selectivo de
"No sé" / "Discrepo" / "¿Y si...?". LF ≥ κ_LF > 0 es condición de viabilidad.

LA GENEALOGÍA DE LA LF (O-N7.2) — tres estadios con historia biológica
----------------------------------------------------------------------
La LF no es un postulado: se desarrolla en tres estadios, cada uno un organelo:
  1. JUEGO  — desacople ENACTUADO: la acción se ejecuta pero su significado está
              suspendido (la mordida que es y no es mordida). El sistema puede
              ejecutar, pero no puede declarar los límites del marco.
  2. RITUAL — desacople FIJADO en estructuras reproducibles pero NO negables desde
              dentro: opera en dos niveles (R₂ presente) pero no puede suspenderlo.
  3. NEGACIÓN OPERATIVA — el desacople pasa de PROPIEDAD de la conducta a OBJETO de
              operación: el sistema declara que su representación no determina su
              acción, opera sobre ella y la regula. Aquí nace el "No". (O-N10.x)

EL ORGANELO-MEDIDA (O-N7.1 + Diccionario 111/114/115)
-----------------------------------------------------
`OrganeloLibertadFuncional` integra la genealogía en una lectura:
  · LF_struct (latente) > 0 ⇔ R₂ > 0           (O-N13.8: hay meta-representación)
  · LF_op (operativa) = LF_struct · (1 − INR)   (O-N13.8.1: el ruido no resuelto
        reduce la libertad EJERCIBLE; por eso LF_op ≤ LF_struct)
  · Escala LF-0..LF-3 (Tabla canónica): 0 salida forzada · 1 "No sé" · 2 "Disiento"
        · 3 "¿Y si...?". La negación operativa requiere nivel ≥ 1 (¬R_op ⇔ LF≥1, O-N10.2).

DISTINCIÓN CLAVE (O-N10.1): INHIBICIÓN ≠ NEGACIÓN OPERATIVA. La inhibición suprime
conducta de primer orden (no ejecuta); la negación opera sobre la REPRESENTACIÓN
(segundo orden). El ritual INHIBE el juego; la negación NIEGA. Distinto de grado, no:
de nivel operativo.

PROCEDENCIA (código): juego ← ModoJuego (CM001:431-447, ex-V156/V165); ritual ←
Ritual (CM001:634-745, ex-V165); negación ← veto de ValenciaLocal/MemoriaDeTrabajo
(CM001, R_op subsumido, ex-V168/V176). Portados cotejando función contra nodo canónico.

ANDAMIAJE DE DEMOSTRACIÓN: este archivo incluye `FuenteEntornoDemo`, un organelo
fuente que inyecta señales que en el organismo real vienen de OTROS bloques (el motor
da `orientacion`; Bloque 5 da `R2`; la termodinámica semiótica da `INR`). Está
marcado como andamiaje: NO es parte del Bloque 7, solo permite correrlo aislado.
================================================================================
"""

from __future__ import annotations
import math
import os as _os
import sys as _sys
from typing import Any

# ESCALA COMPARTIDA (auditoría del 4-ago-2026, regla 1 del plan de constantes): «un módulo
# compartido, no 168 parches». Todo lo que en este bloque se relativiza usa rel/clasificar de
# escala.py — NO se reimplementa aquí el patrón r/(1+r) ni la media móvil.
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
from escala import Escala, rel, rel_contra, clasificar, NEUTRO

# Motor del genoma (contrato + host + célula mínima de base)
from VST_Genoma import (
    Organelo, Estado, Organismo, Milieu, KAPPA,
    OrganeloMarcapasos, OrganeloPresionDesacople, OrganeloFatiga,
    locus_altruismo_boorman,
)


# ==============================================================================
# ESTADIO 1 — JUEGO  (O-N7.2 estadio 1 / O-N10.7)
# ==============================================================================
class OrganeloJuego(Organelo):
    """Juego = desacople ENACTUADO: la acción se ejecuta, su significado se suspende.

    QUÉ HACE (canon O-N10.7): Juego = {R_i | P(Acción|R_i) < 1} — el espacio donde la
    acción NO está determinada por la representación. Es el primer estadio de la LF:
    el sistema ensaya conductas sin comprometer su significado pleno.
    CÓMO (port de ModoJuego, CM001:431-447): se activa si hay setpoint presente y la
    presión de desacople Cb supera un umbral; cuando activo, atenúa la acción física
    (factor λ_fisico) y añade costo. Queda INHIBIDO si el ritual está activo (un
    estadio superior suprime al inferior — inhibición de primer orden, O-N10.1).
    """

    def __init__(self) -> None:
        super().__init__(
            nombre="juego", organelo_analogo="conducta exploratoria (cría que juega)",
            procedencia="CM001:431-447 (ModoJuego, ex-V156/V165)",
            nodo_canonico="O-N7.2 estadio 1 · O-N10.7 (Juego)",
            descripcion=("Desacople enactuado: ejecuta acción con significado suspendido "
                         "(P(Acción|R)<1). Estadio 1 de la LF. Inhibido por el ritual."),
            lee=["presion_desacople", "setpoint_presente", "ritual_activo"],
            secreta=["juego_activo", "mod_juego"],
            depende_de=["presion_desacople", "ritual"],   # corre tras la presión; el ritual lo inhibe
            costo_base=1.0,
            # lambda_fisico=0.15: cuánto se atenúa la acción física durante el juego ("la
            # mordida que es y no es mordida"). NO MEDIBLE hoy: `mod_juego` no lo consume
            # nadie (mapa_organismo.py --var mod_juego → "la consumen: nadie"), así que el
            # número no decide nada observable. Se deja intacto y se anota como pendiente.
            plast={"lambda_fisico": 0.15},
            criterio="se activa con presión de desacople MAYOR QUE LA HABITUAL y setpoint presente; cede ante ritual",
            estado=Estado.PRESENTE,
        )
        self.activo = False
        self._mod = 1.0
        # Lo habitual de la presión de desacople PARA ESTE ORGANISMO (sustituye a umbral_cb=40).
        self.esc_cb = Escala()

    def percibir(self, milieu: "Milieu") -> None:
        self._Cb = milieu.leer("presion_desacople", 0.0)   # arousal que gatea (ex-'Cb')
        self._sp = milieu.leer("setpoint_presente", False)
        self._ritual = bool(milieu.leer("ritual_activo", False))

    def metabolizar(self, dt: float, tempo: float) -> None:
        if self._ritual:                      # el ritual (estadio 2) inhibe el juego
            self.activo = False
        else:
            # CORRECCIÓN 5-ago-2026 (norma del proyecto: nada decide contra una escala que
            # nadie midió).
            # QUÉ ESTABA MAL: `umbral_cb = 40.0`. Un número absoluto sobre la presión de
            # desacople, magnitud cuya escala depende del cuerpo (micrófono, sala, Pi).
            # CIFRA MEDIDA (100.058 pasos, 44 sesiones, ~/.anima/history/*/fisiologia/*.csv):
            # presion_desacople va de 0,005 a 287,2 con media 80,7 y una distribución en DOS
            # MESETAS — 44% de los pasos por debajo de 20 y casi todo el resto por encima de
            # 100. En ese hueco vacío entre 20 y 100 caen tanto el 40 del juego como el 28 del
            # ritual, y por eso los dos deciden PRÁCTICAMENTE LO MISMO: 55,22% de los pasos
            # frente a 55,44%. El número no separaba lo que decía separar; lo separaba el hueco.
            # POR QUÉ LA CORRECCIÓN ES AUTORREGULADA: el juego se dispara cuando la presión es
            # MAYOR QUE LA HABITUAL PARA ESTE ORGANISMO — rel(x, escala) vale 0,5 justo en lo
            # de siempre, así que ">NEUTRO" no tiene ningún parámetro que elegir. Y es legítimo
            # relativizarlo porque esto es una PERCEPCIÓN, no una condición de viabilidad: la
            # cría juega cuando está más activada de lo que le es costumbre (Marco Aurelio: se
            # copia la estructura del caso biológico, no su número). Un organismo crónicamente
            # activado que dejara de jugar por eso NO es un fallo: es lo que hace el animal.
            # EFECTO MEDIDO sobre esos mismos 100.058 pasos: pasa de 55,22% a 23,59% de los
            # pasos con juego activo.
            self.activo = bool(self._sp) and (rel(self._Cb, self.esc_cb) > NEUTRO)
        # La escala aprende DESPUÉS de decidir (el paso actual se compara contra su historia,
        # no contra sí mismo) y sigue aprendiendo aunque el ritual inhiba el juego.
        self.esc_cb.observar(self._Cb)
        # cuando juega, la acción física se atenúa (acción "como si")
        self._mod = self.plast["lambda_fisico"] if self.activo else 1.0

    def secretar(self, milieu: "Milieu") -> None:
        milieu.secretar("juego_activo", self.activo)
        milieu.secretar("mod_juego", self._mod)

    # Sin esto el organismo reaprende en cada arranque qué presión de desacople le es
    # habitual, y sus primeras decisiones del día quedan tomadas contra una escala vacía
    # (escala.py, nota de `restore`). Convención de organelos: vst_persistencia.py:76/122.
    def snapshot(self) -> dict:
        return {"esc_cb": self.esc_cb.snapshot()}

    def restore(self, d: dict) -> None:
        if isinstance(d, dict):
            self.esc_cb.restore(d.get("esc_cb"))


# ==============================================================================
# ESTADIO 2 — RITUAL  (O-N7.2 estadio 2)
# ==============================================================================
class OrganeloRitual(Organelo):
    """Ritual = desacople FIJADO en estructura reproducible, NO negable desde dentro.

    QUÉ HACE (canon O-N7.2 estadio 2): estabiliza la conducta detectando un patrón
    temporal recurrente y reforzándolo. Opera en dos niveles (hay R₂) pero NO puede
    declarar los límites del propio marco — puede ejecutar y reconocer, no suspender.
    Es el estadio donde el doble vínculo es posible (R₂ sin LF, O-N7.3).
    CÓMO (port de Ritual, CM001:634-745): detecta cruces por cero de la orientación;
    si los cruces recurren con el timing del patrón (±tolerancia) y la misma dirección,
    y Cb supera su umbral, acumula `activation` (integrador con fuga τ). Se declara
    `activo` cuando activation supera el umbral. Inhibe al juego mientras está activo.
    """

    def __init__(self) -> None:
        super().__init__(
            nombre="ritual", organelo_analogo="ritmo estabilizador (marcapasos conductual)",
            procedencia="CM001:634-745 (Ritual, ex-V165)",
            nodo_canonico="O-N7.2 estadio 2 (R₂ sin LF posible, O-N7.3)",
            descripcion=("Desacople fijado en estructura reproducible no negable desde dentro. "
                         "Estadio 2 de la LF. Refuerza un patrón temporal; inhibe el juego."),
            lee=["orientacion", "presion_desacople"],
            secreta=["ritual_activo", "ritual_activation"],
            depende_de=["presion_desacople"],
            costo_base=1.5,
            # tau=180 s y repeticion_min=3 se DEJAN: son constantes estructurales declaradas
            # (una constante de tiempo del integrador, como cualquier τ del proyecto; y "un
            # patrón necesita al menos tres repeticiones para serlo").
            # tolerancia=0.3 se DEJA, y ahora sí es legítima: es ADIMENSIONAL — un ±30% DEL
            # PERIODO PROPIO del organismo, no de ningún periodo impuesto. Una vez que el
            # periodo lo pone la historia del organismo (ver metabolizar), esta constante ya no
            # compara contra ninguna escala que nadie midiera: compara contra sí mismo.
            # umbral_activacion=0.4 se DEJA porque NO ES MEDIBLE HOY: el ritual no se activó
            # ni una sola vez en 100.058 pasos, así que `activation` nunca subió y su
            # distribución no existe. Para auditarlo hay que publicar `ritual_activation` al
            # CSV de fisiología (hoy sólo se publica el booleano `ritual`).
            plast={"tau": 180.0, "tolerancia": 0.3,
                   "umbral_activacion": 0.4, "repeticion_min": 3},
            criterio="activation>umbral con el patrón temporal PROPIO recurrente y presión de desacople no-baja",
            estado=Estado.PRESENTE,
        )
        self.activation = 0.0
        self.active = False
        self.patron_buffer: list[tuple[float, int]] = []
        self.repeticiones = 0.0
        self.ultima_orientacion = 0.0
        self._t = 0.0
        # Lo habitual de este organismo, aprendido, en lugar de los números fijos que había:
        #  · esc_periodo  ← sustituye a patron_temporal=40,0 s y tolerancia=0,3
        #  · esc_cb       ← sustituye a umbral_cb=28,0
        self.esc_periodo = Escala()
        self.esc_cb = Escala()
        self._ultimo_cruce_dir: dict[int, float] = {}   # último cruce por dirección

    def percibir(self, milieu: "Milieu") -> None:
        self._orient = milieu.leer("orientacion", 0.0)
        self._Cb = milieu.leer("presion_desacople", 0.0)   # arousal que gatea (ex-'Cb')

    def metabolizar(self, dt: float, tempo: float) -> None:
        self._t += dt
        # --- detección de cruce por cero de la orientación ---
        o = self._orient
        cruce = (self.ultima_orientacion < 0 <= o) or (self.ultima_orientacion > 0 >= o)
        self.ultima_orientacion = o
        # CORRECCIÓN 5-ago-2026 (2/2 de este organelo): la puerta de arousal. ANTES
        # `umbral_cb = 28.0`, el mismo número inventado que el juego traía en 40 y con el
        # mismo defecto medido: presion_desacople es bimodal (44% de los pasos bajo 20, el
        # resto sobre 100) y por eso 28 y 40 deciden casi lo mismo — 55,44% frente a 55,22%
        # de 100.058 pasos. AHORA: la presión no puede estar POR DEBAJO de lo habitual para
        # este organismo. Se usa `clasificar` (no `rel>NEUTRO`) a propósito, para conservar
        # la jerarquía canónica que el par 28/40 quería expresar y no lograba: el ritual
        # (estadio 2, estabilizador) se alcanza con MENOS activación que el juego (estadio 1,
        # explorador), y ahora esa diferencia sí es real y no un artefacto del hueco de la
        # distribución. Mientras no haya historia, `clasificar` devuelve "indeterminado" y
        # el organelo se abstiene en vez de decidir contra una escala vacía.
        if cruce and clasificar(self._Cb, self.esc_cb) in ("normal", "alto"):
            direccion = 1 if o >= 0 else -1
            # El organismo aprende SU PROPIO periodo: el hueco entre cruces de la MISMA
            # dirección (un ciclo completo de la orientación).
            t_ant = self._ultimo_cruce_dir.get(direccion)
            if t_ant is not None:
                self.esc_periodo.observar(self._t - t_ant)
            self._ultimo_cruce_dir[direccion] = self._t
            # CORRECCIÓN 5-ago-2026 (1/2). QUÉ ESTABA MAL: `patron_temporal = 40.0` s. Un
            # periodo FIJO contra el que se juzgaba si la conducta del organismo "repite un
            # patrón". El organismo no tiene por qué latir a 40 s: eso depende de su cuerpo.
            # CIFRAS MEDIDAS: (a) el organelo NO SE ACTIVÓ NUNCA — la columna `ritual` vale 0
            # en los 100.058 pasos de las 44 sesiones de ~/.anima/history/*/fisiologia/*.csv;
            # un solo valor distinto en toda la historia registrada. (b) DE DÓNDE SALIÓ EL 40:
            # del andamiaje de este mismo archivo. `FuenteEntornoDemo` inyecta una senoidal de
            # periodo=80 s, que cruza por cero cada 40. El número estaba calibrado contra la
            # maqueta de demostración, no contra ningún organismo — que es exactamente "un
            # número que decide QUÉ LE PASA al organismo comparado contra una escala que nadie
            # midió". (c) Ni siquiera servía en la maqueta: el demo de este bloque, ejecutado
            # el 5-ago-2026, también imprime ritual_activo=False a los 120 s.
            # LO QUE NO PUDE MEDIR, Y LO DIGO: no puedo demostrar CUÁL de las cuatro puertas
            # (periodo, arousal, repeticiones, activación) mata al ritual en el organismo real,
            # porque la señal `orientacion` que lee este organelo NO se publica al CSV. Usé como
            # proxy `act_orientacion_deg` y el proxy se refutó a sí mismo: con él, el código
            # ORIGINAL habría disparado el 53% de los pasos, luego la orientación real es otra
            # cosa. Por eso esta corrección es la MÍNIMA posible — cambia sólo el número que
            # está probado que vino de la maqueta, y deja intacto todo lo demás.
            # POR QUÉ LA CORRECCIÓN ES AUTORREGULADA: el ritual es, por definición canónica
            # (O-N7.2 estadio 2), una estructura REPRODUCIBLE — algo que repite el ritmo PROPIO
            # del sistema. Así que el periodo lo pone la historia del organismo (esc_periodo,
            # arriba) y la tolerancia sigue siendo el ±30% adimensional de ESE periodo. Mientras
            # no haya 20 ciclos observados, `madura` es False y el organelo se abstiene en vez
            # de decidir contra una escala vacía (contrato de escala.py).
            es_patron = False
            if self.esc_periodo.madura:
                banda = self.esc_periodo.media * self.plast["tolerancia"]
                for t_prev, dir_prev in self.patron_buffer:
                    if dir_prev == direccion and abs((self._t - t_prev) - self.esc_periodo.media) <= banda:
                        es_patron = True
                        break
            if es_patron:
                # El contador se topa en repeticion_min: su único trabajo es exigir que el
                # patrón se haya repetido al menos 3 veces, no amplificar el integrador sin
                # límite. Sin este tope `repeticiones` crece sin freno (medido al reactivar el
                # organelo: llegó a 418 y clavó `activation` en su techo de 2,0 el 83% de los
                # pasos) — un defecto latente que el periodo imposible tenía tapado.
                self.repeticiones = min(self.repeticiones + 1.0, float(self.plast["repeticion_min"]))
                if self.repeticiones >= self.plast["repeticion_min"]:
                    self.activation += (self._Cb * self.repeticiones / 100.0) * dt
            else:
                self.repeticiones = max(0.0, self.repeticiones - 0.5)
            self.patron_buffer.append((self._t, direccion))
            if len(self.patron_buffer) > 10:
                self.patron_buffer.pop(0)
        self.esc_cb.observar(self._Cb)   # la escala de arousal aprende en cada paso
        # fuga del integrador (τ escalada por Kleiber: a más complejidad, ritual más lento)
        self.activation *= math.exp(-dt / (self.plast["tau"] * tempo))
        self.activation = max(0.0, min(2.0, self.activation))
        self.active = self.activation > self.plast["umbral_activacion"]

    def secretar(self, milieu: "Milieu") -> None:
        milieu.secretar("ritual_activo", self.active)
        milieu.secretar("ritual_activation", self.activation)

    def snapshot(self) -> dict:
        # El periodo propio tarda 20 cruces en aprenderse; sin persistirlo el ritual sería
        # imposible en sesiones cortas y el organelo volvería a estar muerto por otra razón.
        return {"esc_periodo": self.esc_periodo.snapshot(),
                "esc_cb": self.esc_cb.snapshot()}

    def restore(self, d: dict) -> None:
        if isinstance(d, dict):
            self.esc_periodo.restore(d.get("esc_periodo"))
            self.esc_cb.restore(d.get("esc_cb"))


# ==============================================================================
# ESTADIO 3 — NEGACIÓN OPERATIVA  (O-N10.1/10.2/10.13)  — el "No"
# ==============================================================================
class OrganeloNegacionOperativa(Organelo):
    """Negación operativa (R_op) = el desacople pasa de propiedad a OBJETO de operación.

    QUÉ HACE (canon O-N10.1/10.2/10.13): el sistema declara que una representación NO
    determina su acción, opera sobre ella y suspende R→Acción. Es el "No" operativo,
    el tercer y último estadio de la LF. A diferencia de la inhibición (que suprime la
    conducta de primer orden sin tocar la representación), la negación opera sobre la
    representación misma (segundo orden). ¬R_op ⇔ LF ≥ 1 (O-N10.2): solo es posible a
    partir del nivel LF-1 de la escala.
    CÓMO (port del veto de ValenciaLocal/MemoriaDeTrabajo, CM001): si el sistema tiene
    nivel de LF suficiente (≥1) y la representación en curso es no-válida o de valencia
    fuertemente negativa, declara la negación y suspende la acción.
    """

    def __init__(self) -> None:
        super().__init__(
            nombre="negacion_operativa", organelo_analogo="corteza inhibitoria (veto deliberado)",
            procedencia="CM001 (veto valencia/memoria_trabajo, R_op subsumido, ex-V168/V176)",
            nodo_canonico="O-N10.1/10.2/10.13 (¬R_op, el 'No')",
            descripcion=("Declara que la representación no determina la acción y la suspende. "
                         "Estadio 3 de la LF. NEGACIÓN (2º orden) ≠ inhibición (1er orden). Requiere LF-nivel≥1."),
            lee=["lf_nivel", "valencia_opcion", "representacion_valida"],
            secreta=["negacion_activa", "accion_suspendida"],
            depende_de=["LF"],   # necesita la lectura de LF para saber si puede negar
            costo_base=1.0,
            # umbral_validez=0.5 se DEJA, y con su origen escrito: en el organismo real
            # `representacion_valida` ES A_sys_env, que campo/Célula_Madre_Funcional_001.py:447
            # define acotado en [0,05 · 1,0] (np.clip). 0,5 es el punto medio de un índice
            # normalizado y acotado por construcción, no una escala inventada: "más inválida
            # que válida". Además la validez de la representación es una CONDICIÓN (¿puedo usar
            # esta representación?), no una percepción — y la advertencia 2 de la auditoría
            # prohíbe relativizar condiciones: un organismo con la representación crónicamente
            # rota debe seguir leyéndola como rota. MEDIDO: discrimina de verdad — se cumple en
            # el 50,18% de los 100.058 pasos, no en el 0% ni en el 100%.
            plast={"umbral_validez": 0.5},
            criterio="con LF≥1 y representación inválida o de valencia peor que la habitual, suspende R→Acción",
            estado=Estado.PRESENTE,
        )
        self.negacion = False
        self.suspendida = False
        # Lo habitual de la MAGNITUD de la valencia negativa (sustituye a umbral_valencia=-2,0).
        self.esc_valencia = Escala()

    def percibir(self, milieu: "Milieu") -> None:
        self._nivel = int(milieu.leer("lf_nivel", 0))
        self._val = milieu.leer("valencia_opcion", 0.0)
        self._validez = milieu.leer("representacion_valida", 1.0)

    def metabolizar(self, dt: float, tempo: float) -> None:
        puede_negar = self._nivel >= 1                       # ¬R_op ⇔ LF≥1 (O-N10.2)
        # CORRECCIÓN 5-ago-2026. QUÉ ESTABA MAL: `umbral_valencia = -2.0`. Una valencia
        # "fuertemente negativa" medida contra el número −2 en una escala que nadie fijó.
        # CIFRA MEDIDA: en el organismo real la valencia NO es una magnitud propia — es el
        # error cambiado de signo (campo/Célula_Madre_Funcional_001.py:551 → valencia_opcion =
        # −e_R), así que −2,0 es en realidad un umbral sobre e_R. Y e_R, medido en 100.058
        # pasos, es bimodal: 45.359 pasos valen exactamente 0,5 y el resto ≥5. Consecuencia:
        # el umbral −2,0 dispara en el 54,74% de los pasos, EXACTAMENTE el mismo porcentaje que
        # el th_osc=5,0 de la mutación (Bloque 8) — cualquier número entre 0,5 y 5,4 hace lo
        # mismo. El −2 no elegía nada: elegía el hueco de la distribución.
        # POR QUÉ LA CORRECCIÓN ES AUTORREGULADA: "esta opción es mala" sólo significa algo
        # comparado con lo malas que suelen ser las opciones que este organismo encara. Es una
        # PERCEPCIÓN (comparación), no una condición de vida — la condición de vida del veto es
        # la validez de la representación, que arriba se deja absoluta a propósito. rel(·)>0,5
        # no tiene parámetro que elegir. EFECTO MEDIDO: 54,74% → 12,13% de los pasos, y el veto
        # sigue existiendo porque el OR con la validez (50,18%) lo sostiene.
        magnitud_negativa = abs(min(0.0, self._val))         # cuán negativa es, en su unidad
        repr_mala = (rel(magnitud_negativa, self.esc_valencia) > NEUTRO) or \
                    (self._validez < self.plast["umbral_validez"])
        self.esc_valencia.observar(magnitud_negativa)        # aprende después de decidir
        self.negacion = bool(puede_negar and repr_mala)
        self.suspendida = self.negacion                      # negar = suspender R→Acción

    def secretar(self, milieu: "Milieu") -> None:
        milieu.secretar("negacion_activa", self.negacion)
        milieu.secretar("accion_suspendida", self.suspendida)

    def snapshot(self) -> dict:
        return {"esc_valencia": self.esc_valencia.snapshot()}

    def restore(self, d: dict) -> None:
        if isinstance(d, dict):
            self.esc_valencia.restore(d.get("esc_valencia"))


# ==============================================================================
# ORGANELO-MEDIDA — LIBERTAD FUNCIONAL  (O-N7.1 · Diccionario 111/114/115)
# ==============================================================================
class OrganeloLibertadFuncional(Organelo):
    """LF — la medida integradora del bloque: cuánta libertad funcional tiene el sistema.

    QUÉ HACE (canon O-N7.1, O-N13.8/8.1, Dicc. 111/114/115):
      · LF_struct (latente) = capacidad de desacople. R₂>0 ⇒ LF_struct>0 (O-N13.8):
        sin meta-representación no hay libertad posible, solo respuesta.
      · LF_op (operativa) = LF_struct·(1−INR) (O-N13.8.1): el ruido NO resuelto (INR)
        recorta la libertad efectivamente ejercible. LF_op ≤ LF_struct.
      · Escala LF (Tabla canónica p.44): 0 salida forzada · 1 "No sé" · 2 "Disiento" ·
        3 "¿Y si...?". Cada nivel incluye al anterior (LF_{n+1} ⊃ LF_n, O-N8.15).
    Secreta `LF` (= LF_op, la que usan Λ_Cos/OI/κ_LF), `LF_struct`, `lf_nivel` y la
    `lf_etiqueta`. Los umbrales de nivel son ORIENTATIVOS y calibrables (C-N2.8.14a).
    """

    # Etiquetas de la escala canónica de LF (Tabla operativa, p.44)
    ESCALA = {0: "LF-0 salida forzada (sin libertad)",
              1: "LF-1 'No sé' (declara dominio vacío)",
              2: "LF-2 'Disiento' (declara dominio incompatible)",
              3: "LF-3 '¿Y si...?' (exploración exaptativa)"}

    def __init__(self) -> None:
        super().__init__(
            nombre="LF", organelo_analogo="lóbulo prefrontal (control ejecutivo)",
            procedencia="CM001 (Rᴿ/valencia) → medida canónica nueva",
            nodo_canonico="O-N7.1 · O-N13.8/8.1 · Dicc. 111/114/115",
            descripcion=("Mide la libertad funcional: LF_struct (latente, desde R₂) y "
                         "LF_op=LF_struct·(1−INR) (efectiva). Clasifica la escala LF-0..3."),
            lee=["R2", "INR"],
            secreta=["LF", "LF_struct", "LF_op", "LF_rel", "lf_nivel", "lf_etiqueta"],
            depende_de=[],   # lee R2/INR (de Bloque 5 / termodinámica); fuentes corren antes
            costo_base=1.0,
            # CORRECCIÓN 5-ago-2026 (de-duplicación con origen). QUÉ ESTABA MAL: `u1 = 0.05`
            # convivía con `kLF = KAPPA["kLF"]`, que vale EXACTAMENTE lo mismo (0,05). Eran el
            # mismo número escrito dos veces, y el de abajo podía derivar del canónico sin que
            # nadie lo notara: la frontera LF-0 (salida forzada) ES, por O-N10.2 (¬R_op ⇔ LF≥1),
            # la condición de viabilidad κ_LF que VST_Genoma.salud() ya evalúa como invariante.
            # AHORA u1 desaparece y el nivel 0 se compara contra KAPPA["kLF"] directamente:
            # una sola fuente de verdad. MEDIDO: sin cambio de conducta — el nivel 0 sale en el
            # 1,61% de los 100.058 pasos antes y después.
            #
            # u2=0,33 y u3=0,66 SE MANTIENEN — son los tercios de un índice adimensional, y eso
            # está bien. Lo que estaba mal era EL ÍNDICE AL QUE SE APLICABAN. Ver abajo.
            #
            # ── LO QUE LA NOTA DEL 5-AGO DABA POR BUENO Y ERA FALSO (medido, 8-ago-2026) ────
            # Decía: «LF_op está acotado en [0,1] por construcción». Acotado sí; ALCANZABLE no.
            # LF_op = LF_struct·(1−INR) es un PRODUCTO de dos índices que provienen ambos de
            # `rel` (escala.py), cuyo punto neutro declarado es NEUTRO=0,5. Un organismo en su
            # estado habitual tiene por tanto LF_struct≈0,5 y (1−INR)≈0,5: LF_op≈0,25. El producto
            # de dos índices con neutro 0,5 NO es un índice con neutro 0,5 — su neutro es 0,25.
            # Aplicarle los tercios de [0,1] es medir con una regla del doble de larga.
            # CIFRAS, sobre las 31.846 filas del 7-ago (fisiologia_2026-08-07_*.csv):
            #   LF_op  p05=0,0631  p50=0,2529  p95=0,2585  MÁXIMO ABSOLUTO=0,2864
            #   lf_nivel: 0 → 3,96 % · 1 → 96,04 % · 2 → 0,00 % · 3 → 0,00 %
            # u2=0,33 está POR ENCIMA DEL MÁXIMO HISTÓRICO de la magnitud que clasifica: el
            # organismo no podía declararse "Disiento" ni "¿Y si...?" ni una sola vez en el día.
            # El instrumento que debe decir si un cambio subió la Libertad Funcional sólo sabía
            # decir "sí, hay algo de libertad" el 96 % del tiempo.
            #
            # ── EL ARREGLO: NORMALIZAR EL ÍNDICE, NO BAJAR LOS UMBRALES ─────────────────────
            # Los tercios se aplican a LF_rel = rel_contra(LF_op, LF_NEUTRO), con
            #     LF_NEUTRO = NEUTRO·(1−NEUTRO) = 0,25
            # que NO es un número elegido: es la imagen del neutro declarado de escala.py a través
            # de la fórmula canónica LF_op = LF_struct·(1−INR) (O-N13.8.1). Es el valor que toma la
            # libertad operativa cuando el organismo está exactamente en su estado de siempre.
            # `rel_contra` es la forma que el propio módulo declara PREFERIBLE a la media móvil
            # («comparar contra otra magnitud del organismo con las mismas unidades»): LF_NEUTRO
            # está en unidades de LF. Y es FIJO, no una media móvil de LF_op — por eso NO puede
            # convertirse en el trinquete sin suelo que la nota del 5-ago temía con razón: una
            # subida sostenida de LF_op sube el nivel y SE QUEDA arriba, que es exactamente lo que
            # se le pide a un instrumento de comparación entre versiones.
            # El suelo (nivel 0) NO se toca y sigue siendo absoluto: κ_LF sobre LF_op, porque
            # lf_nivel≥1 es la CONDICIÓN de la negación operativa (¬R_op ⇔ LF≥1, O-N10.2) y una
            # condición de viabilidad no se relativiza (advertencia 2 de la auditoría).
            # QUÉ SIGNIFICAN AHORA LAS BANDAS, en unidades del organismo:
            #   LF-1 "No sé"      : LF_op < 0,5·LF_NEUTRO  (menos de la mitad de su libertad usual)
            #   LF-2 "Disiento"   : entre la mitad y el doble de su libertad usual
            #   LF-3 "¿Y si...?"  : LF_op ≥ 2·LF_NEUTRO    (el doble de su libertad usual)
            # MEDIDO tras el arreglo (mismo replay, con el C_b del Bloque 5 ya corregido): el
            # reparto 0/1/2/3 pasa de 3,96 / 96,04 / 0,00 / 0,00 a lo que imprime
            # analisis/instr_instrumentos.py. LO QUE ESTE ARREGLO **NO** ALCANZA, y hay que
            # decirlo: LF-3 exige LF_op ≥ 0,5, y como (1−INR) ≤ 0,568 en todo el día medido
            # (INR mínimo = 0,4321), haría falta LF_struct ≥ 0,88 sostenido. El tapón restante es
            # INR, que se produce en campo/Célula_Madre_Funcional_001.py:758 y está fuera de este
            # bloque: INR = rel(|grad|, grad_previo) es otra comparación de una magnitud consigo
            # misma y por eso vive clavada en 0,49 (p05=0,4791, p50=0,4900).
            plast={"kLF": KAPPA["kLF"], "u2": 0.33, "u3": 0.66},
            criterio="LF_op ≥ κ_LF sostenido (hay libertad efectiva, no salida forzada)",
            estado=Estado.PRESENTE,
        )
        self.LF_struct = 0.0
        self.LF_op = 0.0
        self.LF_rel = NEUTRO      # LF_op leída contra su propio neutro estructural (ver plast)
        self.nivel = 0

    # Punto neutro ESTRUCTURAL de la libertad operativa: el valor que toma LF_op cuando sus dos
    # factores están en el neutro declarado de escala.py (LF_struct=NEUTRO y INR=NEUTRO). No es
    # una media móvil ni un número elegido: es NEUTRO·(1−NEUTRO) = 0,25.
    LF_NEUTRO = NEUTRO * (1.0 - NEUTRO)

    def percibir(self, milieu: "Milieu") -> None:
        self._R2 = milieu.leer("R2", 0.0)
        self._INR = max(0.0, min(1.0, milieu.leer("INR", 0.0)))

    def metabolizar(self, dt: float, tempo: float) -> None:
        # R₂>0 ⇒ LF_struct>0 (O-N13.8): la libertad latente nace de la meta-representación
        self.LF_struct = max(0.0, min(1.0, self._R2))
        # LF_op = LF_struct·(1−INR) (O-N13.8.1): el ruido no resuelto recorta lo ejercible
        self.LF_op = self.LF_struct * (1.0 - self._INR)
        # clasificación en la escala LF-0..3. El suelo es la condición de viabilidad canónica κ_LF
        # sobre LF_op (absoluta, sin relativizar). Las tres bandas de arriba son los tercios de
        # LF_rel: LF_op leída contra su propio neutro estructural, que es lo que la vuelve un
        # índice adimensional de verdad y no un producto de dos medios (ver plast).
        self.LF_rel = rel_contra(self.LF_op, self.LF_NEUTRO)
        if self.LF_op < self.plast["kLF"]:
            self.nivel = 0
        elif self.LF_rel < self.plast["u2"]:
            self.nivel = 1
        elif self.LF_rel < self.plast["u3"]:
            self.nivel = 2
        else:
            self.nivel = 3

    def secretar(self, milieu: "Milieu") -> None:
        milieu.secretar("LF", self.LF_op)          # la 'LF' que consumen Λ_Cos/OI/κ_LF
        milieu.secretar("LF_struct", self.LF_struct)
        milieu.secretar("LF_op", self.LF_op)
        # el índice que de verdad se clasifica: sin él, lf_nivel vuelve a ser un número sin
        # magnitud visible detrás y no se puede auditar por qué cambió de banda
        milieu.secretar("LF_rel", self.LF_rel)
        milieu.secretar("lf_nivel", self.nivel)
        milieu.secretar("lf_etiqueta", self.ESCALA[self.nivel])


# ==============================================================================
# ANDAMIAJE DE DEMOSTRACIÓN (NO es parte del Bloque 7)
# Inyecta señales que en el organismo real vienen de otros bloques. Marcado aparte.
# ==============================================================================
class FuenteEntornoDemo(Organelo):
    """Fuente de señales que el Bloque 7 necesita pero que produce OTRO bloque:
    `orientacion` (la da el motor), `R2` (Bloque 5: meta-representación), `INR`
    (termodinámica semiótica), y un par de señales para la negación. Es ANDAMIAJE
    para correr el bloque aislado; se retira cuando esos bloques estén portados."""

    def __init__(self, periodo: float = 80.0, amp: float = 60.0,
                 R2: float = 0.6, INR: float = 0.3, inject_R2: bool = True) -> None:
        # inject_R2=False cuando el R₂ lo provee el Bloque 5 (consciencia funcional)
        self.inject_R2 = inject_R2
        super().__init__(
            nombre="entorno_demo", organelo_analogo="(andamiaje de prueba)",
            procedencia="(scaffolding; señales de motor/Bloque5/termodinámica)",
            nodo_canonico="(no canónico: andamiaje)",
            descripcion="Andamiaje: inyecta orientacion, R2, INR, valencia para correr el Bloque 7 aislado.",
            lee=[], secreta=["orientacion", "setpoint_presente", "R2", "INR",
                             "valencia_opcion", "representacion_valida"],
            costo_base=0.0, estado=Estado.PRESENTE,
        )
        self.plast = {"periodo": periodo, "amp": amp, "R2": R2, "INR": INR}
        self._t = 0.0

    def metabolizar(self, dt: float, tempo: float) -> None:
        self._t += dt

    def secretar(self, milieu: "Milieu") -> None:
        # orientación oscilante (para que el ritual tenga cruces por cero que detectar)
        o = self.plast["amp"] * math.sin(2 * math.pi * self._t / self.plast["periodo"])
        milieu.secretar("orientacion", o)
        milieu.secretar("setpoint_presente", True)
        if self.inject_R2:                                   # solo si NO hay Bloque 5
            milieu.secretar("R2", self.plast["R2"])          # hay meta-representación (andamiaje)
        milieu.secretar("INR", self.plast["INR"])            # ruido no resuelto
        milieu.secretar("valencia_opcion", -3.0)             # opción de valencia negativa (veto)
        milieu.secretar("representacion_valida", 0.3)        # representación poco válida


# ==============================================================================
# TRANSCRIPCIÓN — CÉLULA MADRE CON BLOQUE 7 EXPRESADO
# ==============================================================================
def transcribir_con_libertad_funcional() -> Organismo:
    """Transcribe una célula que expresa la base mínima (marcapasos+Cb+fatiga) MÁS la
    genealogía completa de la libertad funcional (juego→ritual→negación) y su medida
    (LF), con el andamiaje de entorno para correrla aislada. Reserva el locus de Boorman.
    Es el organismo sobre el que se ve, por primera vez, κ_LF ✓ y Λ_Cos/OI > 0."""
    o = Organismo(nombre="celula_madre_B7", M0=1.0)
    o.expresar(OrganeloMarcapasos())          # entorno base (e_R, A_sys_env, Δ_struct...)
    o.expresar(FuenteEntornoDemo())           # andamiaje (orientacion, R2, INR, valencia)
    o.expresar(OrganeloPresionDesacople())    # presión de desacople (alimenta juego/ritual)
    o.expresar(OrganeloFatiga())              # tiempo biológico
    o.expresar(OrganeloRitual())              # estadio 2 (corre antes que juego)
    o.expresar(OrganeloJuego())               # estadio 1 (lo inhibe el ritual)
    o.expresar(OrganeloLibertadFuncional())   # medida de LF
    o.expresar(OrganeloNegacionOperativa())   # estadio 3 (el "No")
    o.expresar(locus_altruismo_boorman())     # locus reservado (PRE)
    return o


# ==============================================================================
# DEMO / AUTOVERIFICACIÓN
# ==============================================================================
if __name__ == "__main__":
    org = transcribir_con_libertad_funcional()
    # Vive ~120s para que Cb se asiente y el ritual tenga cruces que procesar.
    for _ in range(1200):
        parte = org.vivir_un_paso(0.1)

    print(org.quien_soy())

    lf = org.organelos["LF"]
    print(f"\nBLOQUE 7 — LIBERTAD FUNCIONAL (tras {parte['t']:.0f}s):")
    print(f"  LF_struct={lf.LF_struct:.3f}  LF_op={lf.LF_op:.3f}  ->  {lf.ESCALA[lf.nivel]}")
    print(f"  presion_desacople={org.organelos['presion_desacople'].nivel:.2f}  "
          f"juego_activo={org.organelos['juego'].activo}  "
          f"ritual_activo={org.organelos['ritual'].active}  "
          f"negacion_activa={org.organelos['negacion_operativa'].negacion}")
    s = org.salud()
    print(f"  Λ_Cos={s['Lambda_Cos']:.3f}  OI={s['OI']:.3f} → {s['nivel_OI']}")
    print(f"  κ_LF (libertad mínima): {'✓ se cumple' if s['invariantes']['κ_LF libertad'] else '✗'}"
          f"  — antes del Bloque 7 estaba en ✗")
