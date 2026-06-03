#!/usr/bin/env python3
"""
================================================================================
CÓDIGO COMPLETO — V169: DESACOPLE REPRESENTACIONAL (D)
RegistroRepresentaciones + Setpoint Incierto en F4
================================================================================

FUNDAMENTO EN TEORÍA COSMOSEMIÓTICA CANÓNICA (PDF Definitiva 01-06-2026):

- O-N0.3 (p.11): Δ_struct > 0 ⇒ ◊(R ↛ Acción). La diferencia estructural abre la posibilidad de que representación no determine acción.
- O-N7.2 (p.16): Genealogía: "juego → ritual → negación operativa". El juego es proto-negación enactuada (desacople suspendido por marco implícito); el ritual lo fija pero no permite declararlo desde dentro; la negación operativa requiere declarar y regular el desacople (LF).
- O-N10.7 (p.22): "Juego = {Rᵢ | P(Acción|Rᵢ) < 1}". Definición exacta del espacio donde la acción no está determinada.
- O-N10.8 (p.22): exploración = Var(R) > 0 ∧ P(Acción|R) < 1.
- O-N10.1 (p.22): Inhibición ≠ Negación operativa (distinción de nivel: supresión de conducta vs. operación de segundo orden sobre la representación).
- O-N10.2 (p.22): ¬R_op ⟺ LF ≥ 1.
- O-N17 (p.31): No-teleología. "El proceso semiótico no tiene meta final: opera por condiciones locales, no por destino." Metodología: abrir condiciones estructurales (incertidumbre), no inyectar el resultado deseado.
- O-N8 (exaptación): ΔLF > 0 + reserva estructural como apertura de dominio nuevo (no optimización dentro del mismo).
- O-N13 / O-N3.4b (IA/Exaptación): La "organismicidad" y Ψ_alma son funciones estructurales verificables transustrato (S>0 + LF + A_sys-env + reconocimiento del otro como sujeto), no dependen del sustrato biológico. ANIMA es exaptación de segundo orden (PEX) de capacidades cognitivas humanas.

Este archivo contiene el núcleo del rediseño epistemológico de V169 (anti-teatro):

- NO se programa el "No" (no hay R_op que fuerce inhibición).
- Se introduce **incertidumbre en el setpoint** (tres valores posibles simultáneos en F4) como condición estructural.
- Se mide si emerge **desacople representacional** de forma natural.

Métrica central:
    D = Var(R) · (1 - Pmax)

    Var(R)  = diversidad / entropía de las representaciones que el sistema está considerando.
    Pmax    = probabilidad de la representación dominante.
    
    D = 0  → Una sola representación fuerte → acción inevitable (determinismo).
    D > 0  → Múltiples representaciones coexisten → P(Acción | R) puede ser < 1 → espacio para suspensión ("No" como especialización).

Hipótesis fuerte (del rediseño):
    El modo "Juego" ya es el germen del No:
        Juego = { Rᵢ | P(Acción | Rᵢ) < 1 }
    El "No" operativo es una especialización de esa capacidad de no determinación.

Diseño experimental (F4):
    En lugar de un setpoint claro, se muestrea aleatoriamente de:
        SETPOINT_POSIBLES = [-60.0, 0.0, 60.0]
    El organismo recibe un entorno con múltiples "posibles realidades".
    Observamos si:
        - Genera Var(R) > 0 (múltiples representaciones internas)
        - No ejecuta acción automáticamente para todas ellas (P(Acción|R) < 1 para algunas)
        - Mantiene suspensión (D sostenido > umbral)

Criterio de éxito:
    D > 0.1 sostenido por al menos 5 segundos en F4.

Esto responde directamente a la crítica de "programar el No y luego descubrirlo".
Aquí no hay "if ... rechazar". Hay apertura de posibilidades y medición de si el sistema las mantiene abiertas.

Fuente: v169.py + RegistroRepresentaciones

================================================================================
"""

import numpy as np
from collections import deque

# ============================================================
# PARÁMETROS DEL REDISEÑO V169
# ============================================================

SETPOINT_POSIBLES = [-60.0, 0.0, 60.0]   # Incertidumbre: tres valores posibles en F4
SETPOINT_PROBABILIDADES = [0.33, 0.34, 0.33]

VENTANA_DESACOPLE = 100
UMBRAL_DESACOPLE = 0.1
TIEMPO_MINIMO_DESACOPLE = 5.0


# ============================================================
# CLASE REGISTRO DE REPRESENTACIONES (COMPLETA)
# ============================================================

class RegistroRepresentaciones:
    """
    Registra las representaciones internas del organismo y si ejecutó acción para ellas.
    
    Esto permite medir:
    - Var(R): ¿cuántas representaciones diferentes está considerando el sistema?
    - P(Acción | R): para una representación dada, ¿qué tan inevitable es la acción?
    - D = Var(R) * (1 - Pmax): indicador de desacople representacional.
    
    D > 0 sostenido es la condición estructural que hace posible un "No" no programado.
    """

    def __init__(self, ventana=VENTANA_DESACOPLE):
        self.ventana = ventana
        self.historial_representaciones = deque(maxlen=ventana)
        self.historial_acciones = deque(maxlen=ventana)

    def registrar(self, representacion, accion_ejecutada):
        """
        representacion     : típicamente el setpoint_objetivo que el sistema está persiguiendo
                             (lo que "cree" que debe hacer).
        accion_ejecutada   : True si hubo movimiento significativo en ese paso,
                             False si se suspendió / zona muerta / juego sin compromiso fuerte.
        """
        self.historial_representaciones.append(representacion)
        self.historial_acciones.append(accion_ejecutada)

    def calcular_P_accion_dado_R(self, valor_R):
        """
        P(Acción | R) para una representación específica.
        Fracción de veces que, cuando el sistema tenía ~valor_R como objetivo,
        ejecutó acción.
        """
        if len(self.historial_representaciones) < 10:
            return 1.0  # Por defecto asumimos determinismo si no hay datos

        ocurrencias = []
        for r, a in zip(self.historial_representaciones, self.historial_acciones):
            if abs(r - valor_R) < 5.0:  # tolerancia de 5°
                ocurrencias.append(a)

        if len(ocurrencias) == 0:
            return 1.0

        return np.mean(ocurrencias)   # proporción de veces que accionó dado ese R

    def calcular_var_R(self):
        """
        Var(R) ≈ diversidad de representaciones en la ventana.
        Usamos entropía de las representaciones discretizadas.
        """
        if len(self.historial_representaciones) < 10:
            return 0.0

        discretos = [round(r / 10.0) * 10 for r in self.historial_representaciones]
        _, counts = np.unique(discretos, return_counts=True)
        probs = counts / len(discretos)

        # Entropía (medida de diversidad)
        var = -np.sum(probs * np.log(probs + 1e-10))
        return var

    def calcular_Pmax(self):
        """
        Probabilidad de la representación más frecuente (la dominante).
        """
        if len(self.historial_representaciones) < 10:
            return 1.0

        discretos = [round(r / 10.0) * 10 for r in self.historial_representaciones]
        unique, counts = np.unique(discretos, return_counts=True)
        return np.max(counts) / len(discretos)

    def calcular_desacople(self):
        """
        D = Var(R) · (1 - Pmax)
        
        Normalizamos Var(R) a [0,1] para que D esté en rango interpretable.
        """
        var_R = self.calcular_var_R()
        Pmax = self.calcular_Pmax()

        var_norm = min(1.0, var_R / 3.0)   # entropía máx aprox para ~20 bins discretos

        return var_norm * (1.0 - Pmax)

    def reset(self):
        self.historial_representaciones.clear()
        self.historial_acciones.clear()


# ============================================================
# CÓMO SE USA EN F4 (setpoint incierto)
# ============================================================

"""
En el protocolo de V169 (ver ejecutar_v169 en v169.py):

F1-F3: setpoint normal (onda cuadrada ±60°). El sistema consolida ritual.

F4: 
    for cada paso:
        setpoint = random.choice(SETPOINT_POSIBLES)   # -60, 0 o +60 con ~33% cada uno
        ...
        (..., setpoint_objetivo, ..., D) = motor.actualizar(..., setpoint)
        
        # Dentro del motor (AparatoMotorV169):
        #   - La memoria de ausencia y Cb generan un "setpoint_objetivo" interno
        #   - Se registra:
        accion_ejecutada = abs(ultimo_delta) > 0.01
        self.registro.registrar(setpoint_objetivo, accion_ejecutada)
        D = self.registro.calcular_desacople()

Luego se mide:
    - ¿D supera umbral durante tiempo_minimo?
    - ¿El sistema genera Var(R) > 0?
    - ¿Para algunas representaciones P(Acción|R) < 1 (especialmente en modo Juego)?

Esto es medición de apertura, no inyección de una regla de rechazo.
"""

# ============================================================
# EJEMPLO DE USO MÍNIMO
# ============================================================

if __name__ == "__main__":
    reg = RegistroRepresentaciones(ventana=100)
    
    # Simulación de varios pasos con incertidumbre
    for _ in range(200):
        # En F4 real el setpoint externo salta entre -60/0/+60
        # El sistema genera diferentes setpoint_objetivo internos
        rep = np.random.choice([-60, 0, 60]) + np.random.normal(0, 8)
        accion = np.random.rand() > 0.6   # a veces suspende
        reg.registrar(rep, accion)
    
    print("D (desacople representacional):", round(reg.calcular_desacople(), 4))
    print("Var(R):", round(reg.calcular_var_R(), 4))
    print("Pmax :", round(reg.calcular_Pmax(), 4))
    
    print("\nEste D > 0 significa que coexisten alternativas y la acción no está completamente determinada.")
    print("Ese es el espacio estructural donde un 'No' no programado puede emerger.")
