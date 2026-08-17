#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
VST_Bloque05_ConscienciaFuncional — BLOQUE 5 DEL CANON, HECHO ORGANELOS  ·  "QUIÉN SOY"
================================================================================

QUÉ ES ESTE ARCHIVO
-------------------
La implementación del BLOQUE 5 — CONSCIENCIA FUNCIONAL de la Teoría Cosmosemiótica
canónica (Parte II, nodos O-N5.x), como organelos del genoma (VST_Genoma.py). Es la
capa de la INTERIORIDAD del organismo: registrar el propio estado, representar esa
representación, y constituir un Self que la opera.

LOS TRES NODOS (O-N5.1 / O-N5.2 / O-N5.3)
-----------------------------------------
  · C_b = |R₁|            (O-N5.1) — Consciencia básica = capacidad de REGISTRAR el
                          propio estado representacional. No es la presión de desacople:
                          es "cuántas distinciones está representando el sistema".
  · R₂ = R(R)             (O-N5.2) — Meta-representación = representación DE la
                          representación. Es el cimiento de la libertad: R₂>0 ⇒
                          LF_struct>0 (O-N13.8). Sin R₂ no hay libertad, solo respuesta.
  · Self = operador(R₂)   (O-N5.3) — El self es el OPERADOR que gestiona las
                          meta-representaciones. La identidad emerge de cómo el sistema
                          organiza sus R₂ (su coherencia en el tiempo).

POR QUÉ ESTE BLOQUE RESUELVE UNA DERIVA (nombre≠función)
-------------------------------------------------------
En CM001 había un organelo llamado "Cb" cuya fórmula computaba la PRESIÓN DE
DESACOPLE (e_R·(1−A)), no la consciencia básica canónica. Esa pieza se renombró en el
motor a `OrganeloPresionDesacople` (arousal que gatea juego/ritual). AQUÍ vive el C_b
canónico de verdad: C_b = |R₁| (registro del propio estado). Son dos cosas distintas y
ahora tienen dos nombres distintos. (Principio: leer la fórmula, no el nombre.)

PAYOFF DE COMPOSICIÓN CON EL BLOQUE 7
-------------------------------------
El Bloque 7 (Libertad Funcional) leía un R₂ inyectado por andamiaje. Con el Bloque 5
expresado, la LF lee el R₂ REAL producido por la meta-representación: la libertad del
organismo deja de ser un supuesto y pasa a derivar de su propia consciencia. El demo
compone ambos bloques y muestra que la LF sube (a LF-3) cuando el R₂ es endógeno.
================================================================================
"""

from __future__ import annotations

# `escala` vive en celula_madre/; esto permite importar el organelo suelto (pruebas y smokes)
# además de dentro del organismo. Unificado el 5-ago-2026: la revisión encontró CUATRO
# variantes del mismo arranque, que es el problema contra el que existe el módulo compartido.
import os as _os, sys as _sys
_RAIZ_CM = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _RAIZ_CM not in _sys.path:
    _sys.path.insert(0, _RAIZ_CM)
from escala import Escala, rel, NEUTRO
from typing import Any

from VST_Genoma import (
    Organelo, Estado, Organismo, Milieu, KAPPA,
    OrganeloMarcapasos, OrganeloPresionDesacople, OrganeloFatiga,
    locus_altruismo_boorman,
)


# ==============================================================================
# C_b = |R₁|  — CONSCIENCIA BÁSICA  (O-N5.1)
# ==============================================================================
class OrganeloConscienciaBasica(Organelo):
    """C_b = |R₁| — registrar el propio estado representacional (O-N5.1).

    QUÉ HACE: forma R₁ = la representación de primer orden de los datos que el sistema
    distingue ahora mismo, y mide su cardinalidad C_b = |R₁| ("cuántos estímulos está
    procesando"). NO es presión ni arousal: es el REGISTRO de lo que se está representando.
    CÓMO: lee un conjunto de señales representables del milieu; R₁ = las que son
    distinguibles (|valor|>ε); C_b = nº de distinciones activas. Secreta también la
    versión normalizada C_b_norm∈[0,1] para que otros organelos la usen sin escala.
    PROCEDENCIA: nodo canónico O-N5.1 / Diccionario 64 (C_b) y 135 (R₁). Pieza NUEVA:
    en CM001 no existía el C_b canónico (lo que se llamaba 'Cb' era presión de desacople).
    """

    # Señales que el sistema PUEDE representar (su "campo perceptual" mínimo).
    REPRESENTABLES = ["e_R", "orientacion", "delta_struct", "A_sys_env", "presion_desacople"]

    def __init__(self) -> None:
        super().__init__(
            nombre="consciencia_basica", organelo_analogo="núcleo de registro sensorial",
            procedencia="O-N5.1 / Dicc.64,135 (pieza nueva; no existía en CM001)",
            nodo_canonico="O-N5.1 (C_b=|R₁|)",
            descripcion=("Consciencia básica: forma R₁ (representación de 1er orden) y mide "
                         "C_b=|R₁| = nº de distinciones que el sistema registra ahora."),
            lee=list(self.REPRESENTABLES),   # señales que el sistema puede representar
            secreta=["C_b", "C_b_norm", "R1_card"],
            depende_de=[],   # lee señales de base; corre tras las fuentes
            costo_base=1.0,
            plast={},   # epsilon RETIRADO 4-ago-2026: era 1e-3 contra magnitudes de orden 1 a 10,
                  # así que las cinco representables lo superaban siempre y C_b valía 5 fijo.
            criterio="C_b>0 sostenido (el sistema registra su propio estado)",
            estado=Estado.PRESENTE,
        )
        self.R1: list[str] = []
        self.C_b = 0
        self.C_b_norm = 0.0
        self._escalas: dict = {}   # lo habitual de cada representable, aprendido
        self._grados: dict = {}

    def percibir(self, milieu: "Milieu") -> None:
        # R₁ = las representaciones de primer orden actualmente DISTINGUIBLES
        # ── DISTINGUIRSE ES UN GRADO, NO UN SÍ/NO (4-ago-2026) ──────────────────────
        # Antes: `abs(x) > eps` con eps = 1e-3. Las cinco representables viven entre tres y cinco
        # órdenes de magnitud por encima de ese umbral —e_R ronda 8, A_sys_env 0,24— así que las
        # cinco lo superaban SIEMPRE: C_b valía 5,0000 en el 98,43% de los pasos y C_b_norm 1,0000
        # en el 100%. Y como R2 persigue a C_b_norm y LF_struct ES R2, la libertad estructural del
        # organismo era un 1 heredado de una cuenta que no podía dar otra cosa. Peor aún: es una
        # rampa temporal disfrazada de medida — no dice cuánta libertad tiene, dice cuánto lleva
        # encendido.
        #
        # El 4-ago se sustituyó por `rel(|x|, media(|x|))`: cada representable aportaba su
        # MAGNITUD comparada con su media. Y eso volvió a saturar, sólo que en 0,5 en vez de en 1.
        #
        # ── POR QUÉ SATURA `rel(|x|, media)` — 8-ago-2026, con la cifra ────────────────────
        # `rel` devuelve r/(1+r) con r = |x|/media. Para una señal estacionaria r ronda 1, y su
        # variación es el COEFICIENTE DE VARIACIÓN de la señal: e_R vive en ~9,5 con desviación de
        # ~0,5 (r ∈ [0,95 · 1,05] ⇒ rel ∈ [0,487 · 0,512]). Comparar una magnitud contra su propia
        # media DIVIDE la señal por sí misma: lo que queda no es información, es el resto.
        # MEDIDO sobre las 31.846 filas del 7-ago (fisiologia_2026-08-07_*.csv, replay del propio
        # organelo): C_b_norm p05–p95 = 0,4768–0,5054, una banda de 0,029 de ancho alrededor de 0,5.
        # Y como R₂ persigue a C_b_norm y LF_struct ES R₂, la libertad estructural del organismo
        # valía 0,5 pasara lo que pasara: LF_op = 0,5·(1−INR) ≤ 0,30 SIEMPRE, y los niveles LF-2 y
        # LF-3 (u2=0,33, u3=0,66) quedaban fuera del alcance físico del organismo.
        #
        # ── LA FORMA CORRECTA: UNA DISTINCIÓN ES UNA DIFERENCIA ────────────────────────────
        # C_b = |R₁| mide cuántas DISTINCIONES registra el sistema (O-N5.1). Distinguir no es tener
        # un valor grande: es que el valor DIFIERA de lo que suele ser. Una señal constante y enorme
        # no distingue nada. Así que cada representable aporta cuánto se desvía HOY de su propia
        # costumbre, medido en unidades de su propia dispersión — las dos cifras que `Escala` ya
        # aprende y que nadie estaba usando. Sigue sin haber umbral que elegir, y el punto neutro
        # sigue siendo 0,5 (desviarse lo de siempre), pero ahora el índice puede recorrer su rango:
        # MEDIDO en el mismo replay, C_b_norm p05–p95 = 0,2630–0,6519 (ancho 0,389 = 13,4× más).
        # Mientras la escala no madura se devuelve NEUTRO: abstenerse, no inventar (escala.py).
        for _s in self.REPRESENTABLES:
            e = self._escalas.setdefault(_s, Escala())
            x = abs(milieu.leer(_s, 0.0))
            # se decide con la escala de ANTES y se aprende después, como el veto del Bloque 7:
            # si se observa primero, la señal entra en su propia referencia y se resta a sí misma.
            self._grados[_s] = rel(abs(x - e.media), e.dispersion) if e.madura else NEUTRO
            e.observar(x)
        # R1 conserva su sentido —las que HOY destacan sobre su propia costumbre— para R1_card.
        self.R1 = [s for s, g in self._grados.items() if g > NEUTRO]

    def metabolizar(self, dt: float, tempo: float) -> None:
        self.C_b = sum(self._grados.values())                     # C_b = Σ cuánto se distingue cada una
        self.C_b_norm = self.C_b / max(1, len(self.REPRESENTABLES))

    def secretar(self, milieu: "Milieu") -> None:
        milieu.secretar("C_b", float(self.C_b), guardar_historial=True)
        milieu.secretar("C_b_norm", self.C_b_norm)
        milieu.secretar("R1_card", float(self.C_b))


# ==============================================================================
# R₂ = R(R)  — META-REPRESENTACIÓN  (O-N5.2)
# ==============================================================================
class OrganeloMetaRepresentacion(Organelo):
    """R₂ = R(R) — representación de la representación (O-N5.2).

    QUÉ HACE: el sistema representa su propia R₁ — modela QUE está representando y
    CUÁNTO. Es la pieza que funda la libertad: R₂>0 ⇒ LF_struct>0 (O-N13.8). Sin R₂ el
    sistema solo responde; con R₂ puede (potencialmente) operar sobre su representación.
    CÓMO: integra (paso bajo) hacia un objetivo proporcional a C_b_norm — cuanto más
    registra el sistema su estado (C_b alto), más sostiene un modelo de ese registro.
    R₂∈[0,1]. KLEIBER: la constante de integración se estira con el tempo s(M).
    PROCEDENCIA: nodo canónico O-N5.2 / Diccionario 136. Es el R₂ REAL que el Bloque 7
    (Libertad Funcional) consumía antes como andamiaje.
    """

    def __init__(self) -> None:
        super().__init__(
            nombre="meta_representacion", organelo_analogo="corteza asociativa (auto-modelo)",
            procedencia="O-N5.2 / Dicc.136 (R₂ real para el Bloque 7)",
            nodo_canonico="O-N5.2 (R₂=R(R)) → funda LF_struct (O-N13.8)",
            descripcion=("Meta-representación: modela la propia R₁. R₂∈[0,1] crece con C_b. "
                         "R₂>0 ⇒ LF_struct>0: es el cimiento de la libertad funcional."),
            lee=["C_b_norm"],
            secreta=["R2"],
            depende_de=["consciencia_basica"],
            costo_base=1.5,
            plast={"tau": 3.0},   # constante de integración del auto-modelo (s)
            criterio="R₂>0 sostenido cuando hay registro (C_b>0)",
            estado=Estado.PRESENTE,
        )
        self.R2 = 0.0

    def percibir(self, milieu: "Milieu") -> None:
        self._target = milieu.leer("C_b_norm", 0.0)   # objetivo = cuánto se registra

    def metabolizar(self, dt: float, tempo: float) -> None:
        # paso bajo hacia el objetivo (auto-modelo que sigue al registro)
        tau = self.plast["tau"] * tempo
        self.R2 += (self._target - self.R2) * (dt / tau)
        self.R2 = max(0.0, min(1.0, self.R2))

    def secretar(self, milieu: "Milieu") -> None:
        milieu.secretar("R2", self.R2)


# ==============================================================================
# Self = operador(R₂)  — EL SELF  (O-N5.3)
# ==============================================================================
class OrganeloSelf(Organelo):
    """Self = operador(R₂) — el operador que gestiona las meta-representaciones (O-N5.3).

    QUÉ HACE: la identidad del sistema NO es una cosa; es el OPERADOR sobre sus R₂. Su
    salud se lee como COHERENCIA: cuán estable es R₂ en el tiempo (un self coherente
    sostiene un auto-modelo estable; uno incoherente oscila). self_activo = hay R₂.
    CÓMO: mantiene una ventana corta de R₂ y computa coherencia = 1/(1+var(R₂)).
    PROCEDENCIA: nodo canónico O-N5.3 / Diccionario 149 (Self).
    """

    def __init__(self, ventana: int = 50) -> None:
        super().__init__(
            nombre="self", organelo_analogo="red de modo por defecto (identidad)",
            procedencia="O-N5.3 / Dicc.149 (Self=operador(R₂))",
            nodo_canonico="O-N5.3 (Self=operador(R₂))",
            descripcion=("El self como operador sobre las meta-representaciones. Mide la "
                         "coherencia (estabilidad) de R₂ en el tiempo; activo si hay R₂."),
            lee=["R2"],
            secreta=["self_activo", "self_coherencia"],
            depende_de=["meta_representacion"],
            costo_base=1.0,
            plast={"umbral_activo": 0.05},
            criterio="self_activo cuando R₂>umbral; coherencia alta = identidad estable",
            estado=Estado.PRESENTE,
        )
        self.ventana = ventana
        self._hist: list[float] = []
        self.activo = False
        self.coherencia = NEUTRO
        # Lo habitual de la INESTABILIDAD del auto-modelo (desviación típica de R₂ en la ventana).
        # Es la referencia contra la que "coherente" significa algo — ver metabolizar().
        self.esc_desv = Escala()

    def percibir(self, milieu: "Milieu") -> None:
        self._R2 = milieu.leer("R2", 0.0)

    def metabolizar(self, dt: float, tempo: float) -> None:
        self._hist.append(self._R2)
        if len(self._hist) > self.ventana:
            self._hist.pop(0)
        # ── POR QUÉ 1/(1+var) NO MEDÍA NADA — CORRECCIÓN 8-ago-2026, con la cifra ──────────
        # R₂ es un paso bajo (τ=3 s) de un índice acotado en [0,1]: su varianza en una ventana de
        # 50 pasos es del orden de 1e-6, así que 1/(1+var) = 0,999999… MEDIDO en las 31.846 filas
        # del 7-ago: self_coherencia = 1,0000 exacto en el 98,46 % de los pasos. No estaba diciendo
        # "el self es coherente": estaba diciendo "esta fórmula no puede dar otra cosa". Es el mismo
        # techo que el cero: saturación por construcción, no medida.
        #
        # LA FORMA CORRECTA. "Coherente" es un COMPARATIVO — un self es coherente si sostiene su
        # auto-modelo MÁS quieto de lo que lo sostiene habitualmente. La magnitud con la que hay que
        # compararlo existe y es del propio organismo: su propia desviación típica habitual, que
        # `Escala` aprende sin parámetro libre. 0,5 = tan estable como de costumbre; →1 mucho más
        # estable; →0 el auto-modelo se está desarmando. Es una PERCEPCIÓN (comparación), no una
        # condición de viabilidad, así que relativizarla es lo que la advertencia 2 de escala.py
        # PIDE. La condición de viabilidad de este organelo es `self_activo` (R₂>umbral) y ésa se
        # deja absoluta, intacta: un self crónicamente apagado debe seguir leyéndose apagado.
        n = len(self._hist)
        if n >= 2:
            media = sum(self._hist) / n
            desv = (sum((x - media) ** 2 for x in self._hist) / n) ** 0.5   # en las unidades de R₂
            # abstención mientras no hay historia de desviaciones con la que comparar
            self.coherencia = (1.0 - rel(desv, self.esc_desv)) if self.esc_desv.madura else NEUTRO
            self.esc_desv.observar(desv)          # aprende después de decidir
        else:
            self.coherencia = NEUTRO
        self.activo = self._R2 > self.plast["umbral_activo"]

    def secretar(self, milieu: "Milieu") -> None:
        milieu.secretar("self_activo", self.activo)
        milieu.secretar("self_coherencia", self.coherencia)


# ==============================================================================
# TRANSCRIPCIÓN — CÉLULA CON CONSCIENCIA FUNCIONAL (Bloque 5)
# ==============================================================================
def transcribir_con_consciencia() -> Organismo:
    """Célula mínima + Bloque 5 (C_b, R₂, Self). Sin Bloque 7: muestra la interioridad
    aislada (registro → auto-modelo → identidad)."""
    o = Organismo(nombre="celula_madre_B5", M0=1.0)
    o.expresar(OrganeloMarcapasos())
    o.expresar(OrganeloPresionDesacople())
    o.expresar(OrganeloConscienciaBasica())
    o.expresar(OrganeloMetaRepresentacion())
    o.expresar(OrganeloSelf())
    o.expresar(OrganeloFatiga())
    o.expresar(locus_altruismo_boorman())
    return o


def transcribir_consciencia_y_libertad() -> Organismo:
    """COMPOSICIÓN Bloque 5 + Bloque 7: la consciencia (R₂ real) alimenta la libertad
    funcional. El andamiaje ya NO inyecta R₂ (inject_R2=False): viene del Bloque 5."""
    # import local para evitar dependencia de carga si solo se usa el Bloque 5
    from VST_Bloque07_LibertadFuncional import (
        OrganeloJuego, OrganeloRitual, OrganeloNegacionOperativa,
        OrganeloLibertadFuncional, FuenteEntornoDemo,
    )
    o = Organismo(nombre="celula_madre_B5_B7", M0=1.0)
    o.expresar(OrganeloMarcapasos())
    o.expresar(FuenteEntornoDemo(inject_R2=False))   # da orientacion/INR/valencia, NO R2
    o.expresar(OrganeloPresionDesacople())
    o.expresar(OrganeloConscienciaBasica())          # C_b=|R₁|
    o.expresar(OrganeloMetaRepresentacion())         # R₂ real (corre antes que LF)
    o.expresar(OrganeloSelf())                        # Self
    o.expresar(OrganeloFatiga())
    o.expresar(OrganeloRitual())
    o.expresar(OrganeloJuego())
    o.expresar(OrganeloLibertadFuncional())          # LF lee el R₂ del Bloque 5
    o.expresar(OrganeloNegacionOperativa())
    o.expresar(locus_altruismo_boorman())
    return o


# ==============================================================================
# DEMO / AUTOVERIFICACIÓN
# ==============================================================================
if __name__ == "__main__":
    # --- Bloque 5 aislado: la interioridad ---
    org = transcribir_con_consciencia()
    for _ in range(600):
        parte = org.vivir_un_paso(0.1)
    cb = org.organelos["consciencia_basica"]
    r2 = org.organelos["meta_representacion"]
    self_ = org.organelos["self"]
    print("=" * 78)
    print(f"BLOQUE 5 — CONSCIENCIA FUNCIONAL (aislado, tras {parte['t']:.0f}s):")
    print(f"  C_b=|R₁|={cb.C_b} (de {len(cb.REPRESENTABLES)} representables)  R₁={cb.R1}")
    print(f"  R₂={r2.R2:.3f}  ·  self_activo={self_.activo}  self_coherencia={self_.coherencia:.3f}")

    # --- Composición Bloque 5 + Bloque 7: la consciencia funda la libertad ---
    org2 = transcribir_consciencia_y_libertad()
    for _ in range(1200):
        parte = org2.vivir_un_paso(0.1)
    lf = org2.organelos["LF"]
    s = org2.salud()
    print("\n" + "=" * 78)
    print(f"COMPOSICIÓN B5+B7 (tras {parte['t']:.0f}s): el R₂ REAL alimenta la LF")
    print(f"  C_b={org2.organelos['consciencia_basica'].C_b}  "
          f"R₂(real)={org2.organelos['meta_representacion'].R2:.3f}  →  "
          f"LF_struct={lf.LF_struct:.3f}  LF_op={lf.LF_op:.3f}")
    print(f"  {lf.ESCALA[lf.nivel]}")
    print(f"  Λ_Cos={s['Lambda_Cos']:.3f}  OI={s['OI']:.3f} → {s['nivel_OI']}")
    invs = "  ".join(f"{'✓' if ok else '✗'} {k.split()[0]}" for k, ok in s['invariantes'].items())
    print(f"  invariantes: {invs}")
