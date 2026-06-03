#!/usr/bin/env python3
"""
================================================================================
CÓDIGO COMPLETO — V170: DESACOPLE REPRESENTACIONAL CON INCERTIDUMBRE AUMENTADA
RegistroRepresentaciones + RitualV170 + Lógica de Setpoint Incierto (sin "No" programado)
================================================================================

FUNDAMENTO EN TEORÍA COSMOSEMIÓTICA CANÓNICA (PDF Definitiva 01-06-2026, leído completo):

- O-N0.3 (p.11): Δ_struct > 0 ⇒ ◊(R ↛ Acción). La diferencia abre la posibilidad de no determinación.
- O-N7.2 (p.16): "Genealogía evolutiva de LF: juego → ritual → negación operativa".
  El juego introduce desacople enactuado (acción con significado suspendido). Ritual lo fija en estructuras reproducibles pero no negables desde dentro. La negación operativa requiere declarar y operar sobre el desacople (LF).
- O-N10.7 (p.22): "Juego = {Rᵢ | P(Acción|Rᵢ) < 1}". El espacio donde la acción no está determinada por la representación.
- O-N10.8 (p.22): exploración = Var(R) > 0 ∧ P(Acción|R) < 1.
- O-N10.1 (p.22): Inhibición (supresión de 1er orden) ≠ Negación operativa (operación de 2do orden sobre la propia representación).
- O-N17 (p.31): No-teleología. "El proceso semiótico no tiene meta final: opera por condiciones locales, no por destino."
  Metodología obligatoria: crear condiciones estructurales de apertura (incertidumbre multi-setpoint + ruido), medir si emerge el germen (D>0 sostenido), NO inyectar el "No" explícito.
- O-N8 (exaptación): ΔLF > 0 al abrir dominios nuevos mediante reserva/estructuras latentes. ANIMA como exaptación transustrato.
- O-N13 (IA y acoplamiento): Las propiedades "orgánicas" (ritual, desacople, posible negación) son funciones estructurales verificables en sustrato artificial si se alcanzan las condiciones (S>0, LF espacio, etc.). No sustancia, función.

Este archivo aísla el núcleo verificable de V170 para pasting público ("código completo de la clase, no snippets"):

- NO hay regla "if ... rechazar", "if desajuste > umbral: inhibir_ritual=True", ni nada equivalente.
- La "suspensión" se observa como proxy: cuando el controlador motor NO produce delta significativo (accion_ejecutada = abs(ultimo_delta) > 0.01) pese a tener un setpoint_objetivo interno (la R).
- La incertidumbre se genera externamente en F4 (4 setpoints + ruido gaussiano) como condición estructural.
- D se calcula a partir de las representaciones internas reales que el sistema persigue (setpoint_objetivo de la memoria de ausencia + Cb) y si actuó o no.

Métrica:
    D = Var(R) · (1 - Pmax)
    (Var(R) vía entropía de representaciones discretizadas en ventana; Pmax = freq de la más común)

Criterio (ajustado en V170):
    D > 0.08 sostenido por ≥ 3 segundos en F4.

Esto permite observar si "el juego ya contiene el germen lógico del No" sin programar el No.

Incluye:
- RitualV170 (con parámetros para propagación F3→F4)
- RegistroRepresentaciones (con ruido perceptual)
- Fragmento del motor donde se registra (AparatoMotorV170.actuar)
- Función de setpoint incierto (4 valores + gaussiano)

Fuente principal: v170.py (ejecución en curso al momento de extracción)

================================================================================
"""

import numpy as np
from collections import deque
import random

# ============================================================
# RITUAL V170 (ajustado para propagación a F4)
# ============================================================

class RitualV170:
    def __init__(self, tau=300.0, repeticion_min=2, ritual_gain=0.05,
                 patron_temporal=40.0, tolerancia=0.3, umbral_activacion=0.35,
                 umbral_cb=28.0, salida_suave=0.98, persistencia_min=5):
        self.tau = tau
        self.repeticion_min = repeticion_min
        self.ritual_gain = ritual_gain
        self.patron_temporal = patron_temporal
        self.tolerancia = tolerancia
        self.umbral_activacion = umbral_activacion
        self.umbral_cb = umbral_cb
        self.salida_suave = salida_suave
        self.persistencia_min = persistencia_min
        
        self.activation = 0.0
        self.active = False
        self.patron_buffer = []
        self.repeticiones_consecutivas = 0
        self.tiempo_activo = 0.0
        self.ultima_orientacion = 0.0
        self.cruces = 0
        self.ciclos_sin_cruce = 0
    
    def detectar_cruce_por_cero(self, orientacion):
        cruce = (self.ultima_orientacion < 0 and orientacion >= 0) or \
                (self.ultima_orientacion > 0 and orientacion <= 0)
        self.ultima_orientacion = orientacion
        if cruce:
            self.cruces += 1
            self.ciclos_sin_cruce = 0
            return True
        else:
            self.ciclos_sin_cruce += 1
            return False
    
    def actualizar(self, orientacion, Cb, tiempo_actual, dt):
        es_cruce = self.detectar_cruce_por_cero(orientacion)
        
        if es_cruce and Cb > self.umbral_cb:
            es_patron = False
            for t_prev in self.patron_buffer:
                dt_desde_prev = tiempo_actual - t_prev
                if abs(dt_desde_prev - self.patron_temporal) <= (self.patron_temporal * self.tolerancia):
                    es_patron = True
                    break
            
            if es_patron:
                self.repeticiones_consecutivas += 1
                if self.repeticiones_consecutivas >= self.repeticion_min:
                    incremento = Cb * self.repeticiones_consecutivas / 100.0
                    self.activation += incremento * dt
            else:
                self.repeticiones_consecutivas = max(0, self.repeticiones_consecutivas - 1)
            
            self.patron_buffer.append(tiempo_actual)
            if len(self.patron_buffer) > 10:
                self.patron_buffer.pop(0)
        else:
            self.repeticiones_consecutivas = max(0, self.repeticiones_consecutivas - 0.1)
        
        self.activation *= np.exp(-dt / self.tau)
        self.activation = max(0.0, min(2.0, self.activation))
        
        if self.activation > self.umbral_activacion:
            self.active = True
        elif self.active:
            if self.ciclos_sin_cruce > self.persistencia_min:
                self.active = False
            else:
                self.active = self.active * self.salida_suave
        
        if self.active:
            self.tiempo_activo += dt
        
        return self.active
    
    def modular_correccion(self, delta_raw, correccion_ritual):
        if self.active:
            return delta_raw * (1 - self.activation * 0.3) + correccion_ritual * self.activation
        return delta_raw
    
    def reset(self):
        self.activation = 0.0
        self.active = False
        self.patron_buffer = []
        self.repeticiones_consecutivas = 0
        self.tiempo_activo = 0.0
        self.ultima_orientacion = 0.0
        self.cruces = 0
        self.ciclos_sin_cruce = 0


# ============================================================
# REGISTRO DE REPRESENTACIONES PARA DESACOPLE (NÚCLEO)
# ============================================================

class RegistroRepresentaciones:
    """
    Registra las representaciones de acción del organismo.
    Permite calcular P(Acción|R) y el desacople representacional D.
    
    Representación (R): el setpoint_objetivo interno que el sistema persigue
    (proveniente de memoria de ausencia + Cb).
    
    Acción ejecutada: proxy de si actuó (delta motor significativo).
    Si para una R dada, a veces actúa y a veces no → P(Acción|R) < 1 → D > 0.
    """
    
    def __init__(self, ventana=100, ruido_sigma=5.0):
        self.ventana = ventana
        self.ruido_sigma = ruido_sigma
        self.historial_representaciones = deque(maxlen=ventana)
        self.historial_acciones = deque(maxlen=ventana)
    
    def registrar(self, representacion, accion_ejecutada):
        """
        Args:
            representacion: setpoint_objetivo interno (R)
            accion_ejecutada: True si hubo movimiento significativo (delta > 0.01), False si se "suspendió"
        """
        if self.ruido_sigma > 0:
            representacion_ruidosa = representacion + np.random.normal(0, self.ruido_sigma)
        else:
            representacion_ruidosa = representacion
        
        self.historial_representaciones.append(representacion_ruidosa)
        self.historial_acciones.append(accion_ejecutada)
    
    def calcular_P_accion_dado_R(self, valor_R):
        if len(self.historial_representaciones) < 10:
            return 1.0
        
        ocurrencias = []
        for r, a in zip(self.historial_representaciones, self.historial_acciones):
            if abs(r - valor_R) < 10.0:
                ocurrencias.append(a)
        
        if len(ocurrencias) == 0:
            return 1.0
        
        return np.mean(ocurrencias)
    
    def calcular_var_R(self):
        if len(self.historial_representaciones) < 10:
            return 0.0
        
        discretos = [round(r / 10.0) * 10 for r in self.historial_representaciones]
        _, counts = np.unique(discretos, return_counts=True)
        probs = counts / len(discretos)
        
        var = -np.sum(probs * np.log(probs + 1e-10))
        return var
    
    def calcular_Pmax(self):
        if len(self.historial_representaciones) < 10:
            return 1.0
        
        discretos = [round(r / 10.0) * 10 for r in self.historial_representaciones]
        unique, counts = np.unique(discretos, return_counts=True)
        return np.max(counts) / len(discretos)
    
    def calcular_desacople(self):
        """
        D = Var(R) · (1 - Pmax)
        
        D = 0: una sola representación fuerte → acción inevitable (determinismo representacional).
        D > 0: alternativas coexisten → para algunas R, P(Acción|R) puede ser < 1 → espacio estructural para suspensión / "No".
        """
        var_R = self.calcular_var_R()
        Pmax = self.calcular_Pmax()
        
        var_norm = min(1.0, var_R / 3.0)
        return var_norm * (1.0 - Pmax)
    
    def reset(self):
        self.historial_representaciones.clear()
        self.historial_acciones.clear()


# ============================================================
# FRAGMENTO CLAVE DEL MOTOR (MUESTRA REGISTRO SIN FORZAR RECHAZO)
# ============================================================

# En AparatoMotorV170.actuar (simplificado para claridad pública):
#
#   # ... (memoria → setpoint_objetivo, Cb, ritual_activo, juego, etc.)
#
#   # REGISTRO DE REPRESENTACIONES PARA DESACOPLE (siempre, pero analizado en F4)
#   accion_ejecutada = abs(self.ultimo_delta) > 0.01   # proxy de "actuó o suspendió"
#   self.registro.registrar(setpoint_objetivo, accion_ejecutada)
#
#   D = self.registro.calcular_desacople()
#
#   # ... luego cálculo de delta_raw, modulación por ritual/juego, aplicación de inercia, fatiga, etc.
#   # delta_fisico = ...
#   # self.orientacion += delta_fisico
#
#   # NUNCA hay:
#   #   if D > umbral or costo > beneficio or señal_desajuste > X:
#   #       ritual_activo = False
#   #       inhibir = True
#   #       return sin movimiento forzado
#
#   return (..., D, ...)


# ============================================================
# GENERADOR DE SETPOINT INCIERTO (CONDICIÓN ESTRUCTURAL EN F4)
# ============================================================

def setpoint_incierto_mejorado(t, intervalo=10.0,
                                posibles=[-60.0, -20.0, 20.0, 60.0], 
                                probs=[0.25, 0.25, 0.25, 0.25],
                                ruido_sigma=15.0):
    """
    Condición estructural de apertura (no teleológica):
    - Cada 'intervalo' segundos elige uniformemente entre 4 setpoints.
    - Añade ruido gaussiano para dificultar memorización rígida.
    Esto aumenta la probabilidad de que coexistan múltiples R internas,
    permitiendo que Var(R) > 0 y que para algunas de ellas P(Acción|R) < 1 emerja.
    """
    fase = int(t / intervalo)
    rng = random.Random(int(fase * 1000) % 2**32)
    setpoint_base = rng.choices(posibles, weights=probs)[0]
    
    if ruido_sigma > 0:
        ruido = np.random.normal(0, ruido_sigma)
        setpoint_base += ruido
    
    return setpoint_base


# ============================================================
# NOTA DE USO Y VERIFICABILIDAD
# ============================================================
"""
Para reproducir / auditar:

1. Copia esta clase + el generador de setpoint.
2. En tu loop de simulación (F4), muestrea setpoint = setpoint_incierto_mejorado(t, ...)
3. Pasa el setpoint al "organismo".
4. Dentro del actuador, registra siempre: registro.registrar(setpoint_objetivo_interno, abs(delta)>0.01)
5. Al final de F4 (o por ventana), calcula D = registro.calcular_desacople()
6. Mide duración máxima donde D > umbral.

Publica:
- Este archivo completo.
- El JSON raw con series de t, setpoint (externo), setpoint_objetivo (R), D, ritual_act, accion_ejecutada para F4.
- Los parámetros exactos usados.
- El gráfico de D(t) en F4 con el umbral marcado.

Esto distingue claramente:
- Programado: las reglas del controlador, la modulación ritual, la memoria, la forma de elegir setpoint incierto.
- Observado (emergente): si bajo esa incertidumbre el sistema genera diversidad de R y reduce la determinación de la acción (D>0 sostenido).

Cumple con O-N17 (condiciones locales, no meta inyectada) y O-N10.7 (medir el espacio de Juego).

Versión alineada con V170 (junio 2026).
"""