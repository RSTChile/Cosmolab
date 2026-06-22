#!/usr/bin/env python3
"""
================================================================================
CÓDIGO COMPLETO — R_op : PRIMER "NO" OPERATIVO (Etapa 5 de ANIMA-2)
V168 (basado en V168.py)
================================================================================

Este archivo contiene el código completo de la nueva capacidad:

R_op (Primer "No" operativo)
- Usa la señal de desajuste generada por la Meta-representación (Rᴿ de V167)
- Cuando la señal supera umbral (0.7) con histéresis, INHIBE activamente el ritual
- Mantiene la inhibición por duración mínima
- Desinhibe cuando la señal cae y ha pasado tiempo suficiente
- Esto es el primer "No" del sistema a su propio comportamiento ritualizado histórico

Contexto:
- V167 validó que el ritual puede persistir ciegamente y generar desajuste detectable (corr 0.901 en la corrida reportada).
- V168 agrega la capacidad de USAR esa detección para suspender el marco (inhibición operativa).

El "No" no es programado como "si error_alto entonces no ritual".
Es consecuencia de:
  Señal_desajuste (de Rᴿ) → R_op decide inhibir → fuerza ritual_activo=False

Fuente: V168.py

Parámetros exactos de inhibición:
  R_OP_UMBRAL_INHIBICION = 0.7
  R_OP_HISTERESIS = 0.5 s (debe superar umbral por este tiempo)
  R_OP_INHIBITION_DURATION = 5.0 s (mínimo de inhibición)
  R_OP_DESINHIBICION_THRESHOLD = 0.3

Criterios de éxito Etapa 5 (del código):
  1. Inhibición activa en F4
  2. Error RMS menor tras inhibición (R_op branch mejor que control)
  3. Menor tiempo en ritual en la rama con R_op

Resultado reportado por el usuario para la corrida V168-ob.py (base V167):
  (el texto pegado muestra la validación de Etapa 4 previa y "listo para Etapa 5")

================================================================================
"""

import numpy as np
from collections import deque

# ============================================================
# PARÁMETROS R_op (exactos del script)
# ============================================================

R_OP_UMBRAL_INHIBICION = 0.7          # Mismo que umbral de desajuste en esta etapa
R_OP_HISTERESIS = 0.5                 # Necesita señal > umbral por 0.5s para inhibir
R_OP_INHIBITION_DURATION = 5.0        # Duración mínima de inhibición (segundos)
R_OP_DESINHIBICION_THRESHOLD = 0.3    # Señal debe caer bajo este umbral para desinhibir

# (Otros parámetros heredados de V167 se omiten aquí por brevedad; ver V168.py)


# ============================================================
# CLASE R_op — EL PRIMER "NO" OPERATIVO (COMPLETA)
# ============================================================

class R_op:
    """
    Primer "No" operativo: capacidad de inhibir el ritual
    cuando la señal de desajuste supera un umbral.
    
    Incluye:
    - Histéresis para evitar oscilaciones rápidas (debe superar umbral por tiempo)
    - Duración mínima de inhibición (no se desactiva inmediatamente)
    - Desinhibición cuando la señal cae por debajo del umbral de desinhibición
      y ha pasado el tiempo mínimo.
    
    Esto implementa "decir no a un marco histórico disfuncional" de forma operativa,
    usando la información de la meta-representación (Rᴿ).
    """
    
    def __init__(self, umbral_inhibicion=R_OP_UMBRAL_INHIBICION,
                 histéresis=R_OP_HISTERESIS,
                 duracion_minima=R_OP_INHIBITION_DURATION,
                 umbral_desinhibicion=R_OP_DESINHIBICION_THRESHOLD):
        self.umbral_inhibicion = umbral_inhibicion
        self.histeresis = histéresis
        self.duracion_minima = duracion_minima
        self.umbral_desinhibicion = umbral_desinhibicion
        
        self.inhibicion_activa = False
        self.tiempo_en_inhibicion = 0.0
        self.historial_inhibicion = []
        self.señal_para_historial = []
        self.tiempo_desde_ultimo_cruce = 0.0   # en realidad "tiempo por encima del umbral"
    
    def actualizar(self, señal_desajuste, dt):
        """
        Reglas explícitas de decisión de inhibición:
        
        1. Si no está inhibido:
           - Si señal > umbral_inhibicion (0.7), acumula tiempo.
           - Si el tiempo acumulado >= histéresis (0.5s), activa inhibición.
        
        2. Si está inhibido:
           - Acumula tiempo_en_inhibicion.
           - Si señal < umbral_desinhibicion (0.3) Y tiempo_en_inhibicion >= duracion_minima (5s),
             entonces desinhibe.
        
        Retorna: inhibicion_activa (bool)
        """
        self.señal_para_historial.append(señal_desajuste)
        
        if not self.inhibicion_activa:
            # Verificar si hay que inhibir
            if señal_desajuste > self.umbral_inhibicion:
                self.tiempo_desde_ultimo_cruce += dt
                if self.tiempo_desde_ultimo_cruce >= self.histeresis:
                    self.inhibicion_activa = True
                    self.tiempo_en_inhibicion = 0.0
            else:
                self.tiempo_desde_ultimo_cruce = 0.0
        else:
            # Ya inhibido: verificar si hay que desinhibir
            self.tiempo_en_inhibicion += dt
            
            if (señal_desajuste < self.umbral_desinhibicion and 
                self.tiempo_en_inhibicion >= self.duracion_minima):
                self.inhibicion_activa = False
                self.tiempo_en_inhibicion = 0.0
        
        self.historial_inhibicion.append(self.inhibicion_activa)
        return self.inhibicion_activa
    
    def reset(self):
        self.inhibicion_activa = False
        self.tiempo_en_inhibicion = 0.0
        self.historial_inhibicion = []
        self.señal_para_historial = []
        self.tiempo_desde_ultimo_cruce = 0.0


# ============================================================
# INTEGRACIÓN EN EL MOTOR (fragmento clave de AparatoMotorV168)
# ============================================================

# (Extracto del código que muestra cómo R_op actúa sobre el ritual)

"""
En AparatoMotorV168.actuar(...):

    # ETAPA 3: Ritual (se actualiza ANTES que juego)
    ritual_activo = self.ritual.actualizar(self.orientacion, Cb, t, dt)
    
    # ETAPA 4: Meta-representación observacional (Rᴿ)
    senal_desajuste, hay_desajuste = self.meta.actualizar(error, Cb, ritual_activo, dt)
    
    # ETAPA 5: R_op (Primer "No" operativo) - NUEVO
    inhibir_ritual = False
    if self.enable_rop and self.rop is not None:
        inhibir_ritual = self.rop.actualizar(senal_desajuste, dt)
    
    # Si R_op decide inhibir, fuerza el ritual a inactivo (el "No" operativo)
    if inhibir_ritual:
        ritual_activo = False
        self.ritual.active = False
    
    # Luego continúa con Juego (que se inhibe si ritual sigue activo), fatiga, etc.
    # El return incluye ..., ritual_activo, ..., senal_desajuste, ..., inhibir_ritual
"""

# ============================================================
# CÓDIGO DE USO / CRITERIOS (del análisis en V168.py)
# ============================================================

def ejemplo_uso_rop():
    rop = R_op()
    # En el loop del experimento:
    # inhibir = rop.actualizar(senal_desajuste_actual, DT)
    # if inhibir:
    #     ... forzar ritual off
    print("R_op listo. Umbral inhibición:", R_OP_UMBRAL_INHIBICION)


if __name__ == "__main__":
    ejemplo_uso_rop()
    print("\nVer V168.py para el protocolo completo de control vs experimental (CON R_op),")
    print("logging de 'inhibir_ritual', criterios de éxito y gráficos comparativos.")
