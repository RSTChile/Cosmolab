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
            plast={"epsilon": 1e-3},
            criterio="C_b>0 sostenido (el sistema registra su propio estado)",
            estado=Estado.PRESENTE,
        )
        self.R1: list[str] = []
        self.C_b = 0
        self.C_b_norm = 0.0

    def percibir(self, milieu: "Milieu") -> None:
        eps = self.plast["epsilon"]
        # R₁ = las representaciones de primer orden actualmente DISTINGUIBLES
        self.R1 = [s for s in self.REPRESENTABLES if abs(milieu.leer(s, 0.0)) > eps]

    def metabolizar(self, dt: float, tempo: float) -> None:
        self.C_b = len(self.R1)                                   # C_b = |R₁|
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
        self.coherencia = 0.0

    def percibir(self, milieu: "Milieu") -> None:
        self._R2 = milieu.leer("R2", 0.0)

    def metabolizar(self, dt: float, tempo: float) -> None:
        self._hist.append(self._R2)
        if len(self._hist) > self.ventana:
            self._hist.pop(0)
        # varianza simple de R₂ en la ventana -> coherencia = 1/(1+var)
        n = len(self._hist)
        if n >= 2:
            media = sum(self._hist) / n
            var = sum((x - media) ** 2 for x in self._hist) / n
            self.coherencia = 1.0 / (1.0 + var)
        else:
            self.coherencia = 0.0
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
