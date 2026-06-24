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
from typing import Any

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
            plast={"th_osc": 5.0, "tasa": 0.3, "escala": 0.05},
            criterio="emite ΔR solo sobre e_R no filtrado (|e_R|>th_osc)",
            estado=Estado.PRESENTE,
        )
        self._rng = random.Random(seed)
        self.mutacion = 0.0
        self.activa = False

    def percibir(self, milieu: "Milieu") -> None:
        self._eR = abs(milieu.leer("e_R", 0.0))

    def metabolizar(self, dt: float, tempo: float) -> None:
        no_filtrado = max(0.0, self._eR - self.plast["th_osc"])   # lo que escapa a la corrección
        if no_filtrado > 0.0 and self._rng.random() < self.plast["tasa"]:
            self.mutacion = self._rng.gauss(0.0, no_filtrado * self.plast["escala"])
            self.activa = True
        else:
            self.mutacion = 0.0
            self.activa = False

    def secretar(self, milieu: "Milieu") -> None:
        milieu.secretar("mutacion", self.mutacion)
        milieu.secretar("mutacion_activa", self.activa)


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
            plast={"k": 0.05, "costo_reserva": 1.0},
            criterio="exapta solo en límite adaptativo CON reserva>0 (PRE); ΔΩop>0, ΔLF>0",
            estado=Estado.PRESENTE,
        )
        self.Omega_op = omega0      # dominio operativo actual
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
        milieu.secretar("XE", min(1.0, self.XE))         # normalizada para el OI
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
            plast={"th_fallo": 5.0, "tau": 10.0, "umbral_cm": 0.3},
            criterio="C_m>umbral cuando |e_R| sostenido alto ∧ LF≥κ_LF",
            estado=Estado.PRESENTE,
        )
        self.ventana = ventana
        self._buf: list[float] = []
        self.C_m = 0.0
        self.activa = False

    def percibir(self, milieu: "Milieu") -> None:
        self._eR = abs(milieu.leer("e_R", 0.0))
        self._LF = milieu.leer("LF", 0.0)

    def metabolizar(self, dt: float, tempo: float) -> None:
        self._buf.append(self._eR)
        if len(self._buf) > self.ventana:
            self._buf.pop(0)
        fallo_sostenido = (sum(self._buf) / len(self._buf)) > self.plast["th_fallo"]
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
            secreta=["demanda_activacion", "deficit_capacidad"],
            depende_de=["exaptacion"],   # lee el Ωop ya actualizado por la exaptación
            costo_base=0.5,
            plast={"umbral": 0.1},
            criterio="señala activación cuando deficit (demanda−Ωop) > umbral",
            estado=Estado.PRESENTE,
        )
        self.deficit = 0.0
        self.demanda_activacion = False

    def percibir(self, milieu: "Milieu") -> None:
        self._demanda = milieu.leer("demanda_entorno", 1.0)
        self._Omega = milieu.leer("Omega_op", 1.0)

    def metabolizar(self, dt: float, tempo: float) -> None:
        self.deficit = max(0.0, self._demanda - self._Omega)
        self.demanda_activacion = self.deficit > self.plast["umbral"]

    def secretar(self, milieu: "Milieu") -> None:
        milieu.secretar("demanda_activacion", self.demanda_activacion)
        milieu.secretar("deficit_capacidad", self.deficit)


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
