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
from typing import Any

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
            plast={"umbral_cb": 40.0, "lambda_fisico": 0.15},
            criterio="se activa con Cb alto y setpoint presente; cede ante ritual",
            estado=Estado.PRESENTE,
        )
        self.activo = False
        self._mod = 1.0

    def percibir(self, milieu: "Milieu") -> None:
        self._Cb = milieu.leer("presion_desacople", 0.0)   # arousal que gatea (ex-'Cb')
        self._sp = milieu.leer("setpoint_presente", False)
        self._ritual = bool(milieu.leer("ritual_activo", False))

    def metabolizar(self, dt: float, tempo: float) -> None:
        if self._ritual:                      # el ritual (estadio 2) inhibe el juego
            self.activo = False
        else:
            self.activo = bool(self._sp) and (self._Cb > self.plast["umbral_cb"])
        # cuando juega, la acción física se atenúa (acción "como si")
        self._mod = self.plast["lambda_fisico"] if self.activo else 1.0

    def secretar(self, milieu: "Milieu") -> None:
        milieu.secretar("juego_activo", self.activo)
        milieu.secretar("mod_juego", self._mod)


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
            plast={"tau": 180.0, "patron_temporal": 40.0, "tolerancia": 0.3,
                   "umbral_cb": 28.0, "umbral_activacion": 0.4, "repeticion_min": 3},
            criterio="activation>umbral con patrón temporal recurrente y Cb alto",
            estado=Estado.PRESENTE,
        )
        self.activation = 0.0
        self.active = False
        self.patron_buffer: list[tuple[float, int]] = []
        self.repeticiones = 0.0
        self.ultima_orientacion = 0.0
        self._t = 0.0

    def percibir(self, milieu: "Milieu") -> None:
        self._orient = milieu.leer("orientacion", 0.0)
        self._Cb = milieu.leer("presion_desacople", 0.0)   # arousal que gatea (ex-'Cb')

    def metabolizar(self, dt: float, tempo: float) -> None:
        self._t += dt
        # --- detección de cruce por cero de la orientación ---
        o = self._orient
        cruce = (self.ultima_orientacion < 0 <= o) or (self.ultima_orientacion > 0 >= o)
        self.ultima_orientacion = o
        if cruce and self._Cb > self.plast["umbral_cb"]:
            direccion = 1 if o >= 0 else -1
            # ¿el cruce repite un patrón temporal previo (mismo timing y dirección)?
            es_patron = False
            for t_prev, dir_prev in self.patron_buffer:
                dt_prev = self._t - t_prev
                timing_ok = abs(dt_prev - self.plast["patron_temporal"]) <= \
                    self.plast["patron_temporal"] * self.plast["tolerancia"]
                if timing_ok and dir_prev == direccion:
                    es_patron = True
                    break
            if es_patron:
                self.repeticiones += 1
                if self.repeticiones >= self.plast["repeticion_min"]:
                    self.activation += (self._Cb * self.repeticiones / 100.0) * dt
            else:
                self.repeticiones = max(0.0, self.repeticiones - 0.5)
            self.patron_buffer.append((self._t, direccion))
            if len(self.patron_buffer) > 10:
                self.patron_buffer.pop(0)
        # fuga del integrador (τ escalada por Kleiber: a más complejidad, ritual más lento)
        self.activation *= math.exp(-dt / (self.plast["tau"] * tempo))
        self.activation = max(0.0, min(2.0, self.activation))
        self.active = self.activation > self.plast["umbral_activacion"]

    def secretar(self, milieu: "Milieu") -> None:
        milieu.secretar("ritual_activo", self.active)
        milieu.secretar("ritual_activation", self.activation)


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
            plast={"umbral_valencia": -2.0, "umbral_validez": 0.5},
            criterio="con LF≥1 y representación inválida/negativa, suspende R→Acción",
            estado=Estado.PRESENTE,
        )
        self.negacion = False
        self.suspendida = False

    def percibir(self, milieu: "Milieu") -> None:
        self._nivel = int(milieu.leer("lf_nivel", 0))
        self._val = milieu.leer("valencia_opcion", 0.0)
        self._validez = milieu.leer("representacion_valida", 1.0)

    def metabolizar(self, dt: float, tempo: float) -> None:
        puede_negar = self._nivel >= 1                       # ¬R_op ⇔ LF≥1 (O-N10.2)
        repr_mala = (self._val < self.plast["umbral_valencia"]) or \
                    (self._validez < self.plast["umbral_validez"])
        self.negacion = bool(puede_negar and repr_mala)
        self.suspendida = self.negacion                      # negar = suspender R→Acción

    def secretar(self, milieu: "Milieu") -> None:
        milieu.secretar("negacion_activa", self.negacion)
        milieu.secretar("accion_suspendida", self.suspendida)


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
            secreta=["LF", "LF_struct", "LF_op", "lf_nivel", "lf_etiqueta"],
            depende_de=[],   # lee R2/INR (de Bloque 5 / termodinámica); fuentes corren antes
            costo_base=1.0,
            plast={"kLF": KAPPA["kLF"], "u1": 0.05, "u2": 0.33, "u3": 0.66},
            criterio="LF_op ≥ κ_LF sostenido (hay libertad efectiva, no salida forzada)",
            estado=Estado.PRESENTE,
        )
        self.LF_struct = 0.0
        self.LF_op = 0.0
        self.nivel = 0

    def percibir(self, milieu: "Milieu") -> None:
        self._R2 = milieu.leer("R2", 0.0)
        self._INR = max(0.0, min(1.0, milieu.leer("INR", 0.0)))

    def metabolizar(self, dt: float, tempo: float) -> None:
        # R₂>0 ⇒ LF_struct>0 (O-N13.8): la libertad latente nace de la meta-representación
        self.LF_struct = max(0.0, min(1.0, self._R2))
        # LF_op = LF_struct·(1−INR) (O-N13.8.1): el ruido no resuelto recorta lo ejercible
        self.LF_op = self.LF_struct * (1.0 - self._INR)
        # clasificación en la escala LF-0..3 (umbrales orientativos, calibrables)
        if self.LF_op < self.plast["u1"]:
            self.nivel = 0
        elif self.LF_op < self.plast["u2"]:
            self.nivel = 1
        elif self.LF_op < self.plast["u3"]:
            self.nivel = 2
        else:
            self.nivel = 3

    def secretar(self, milieu: "Milieu") -> None:
        milieu.secretar("LF", self.LF_op)          # la 'LF' que consumen Λ_Cos/OI/κ_LF
        milieu.secretar("LF_struct", self.LF_struct)
        milieu.secretar("LF_op", self.LF_op)
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
