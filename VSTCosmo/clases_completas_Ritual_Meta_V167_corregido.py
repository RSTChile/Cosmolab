#!/usr/bin/env python3
"""
================================================================================
CÓDIGO COMPLETO DE LAS CLASES CRÍTICAS — V167 CORREGIDO (ANIMA-2 Etapa 4: Rᴿ)
================================================================================

Este archivo contiene:
- Todos los parámetros relevantes usados por Ritual y Meta-representación.
- La clase RitualV167 completa (detector de cruces de cero + patrón temporal + activación).
- La clase MetaRepresentacionObservacional completa (el monitor de desajuste, SOLO observacional).
- Partes clave de AparatoMotorV167 que las integran (actualización, jerarquía ritual > juego, NO inhibición).
- La lógica de cálculo de correlación (la corrección para evitar NaN que se menciona en el encabezado del script).

Propósito: Responder directamente a la solicitud de "código completo de la clase (no solo snippets) y logs raw".

Fuente: v167-ob.py (el script "CORREGIDO" que produjo los resultados del terminal pegados abajo).

NOTA IMPORTANTE (para transparencia):
- Todo lo que el monitor "detecta" está definido por estas reglas explícitas.
- La "emergencia" / persistencia / correlación surge de la *interacción* de estas reglas + el resto de la arquitectura (memoria de ausencia, Cb, fatiga, juego, inercia, etc.) bajo el protocolo experimental (F1 baseline → F2 control prolongado → F3 con ritual → F4 desafío con setpoint invertido).
- El monitor NO modifica el comportamiento del ritual (es puramente observacional en esta etapa; la inhibición vendrá en Etapa 5 = R_op).

Logs raw / resultados de esta corrida exacta:
- v167_logs/v167_corregido_resultados_terminal_20260603.txt  (el output completo del terminal)
- Gráfico: v167_logs/v167_meta_observacional_corregido_20260603_064549.png
- Script completo que se ejecutó: v167-ob.py (en este directorio)

Métricas clave de ESTA corrida (V167 CORREGIDO):
  Tiempo ritual activo: 382.8s (23.9%)
  Ritual activo en F4: True
  Señal desajuste máx F4: 1.000
  Correlación ritual_señal (F3): 0.901
  Error RMS F4 Control: 10.45° | Ritual: 30.03°
  Criterios 1-4 cumplidos ✅

================================================================================
"""

import numpy as np
from collections import deque

# ============================================================
# PARÁMETROS (copiados exactamente del script v167-ob.py CORREGIDO)
# ============================================================

DT = 0.01

# Fatiga / motor base (afectan la dinámica que el ritual "ve")
ZONA_MUERTA_BASE = 2.0
KP_BASE = 0.002
KP_MIN = 0.0005
KP_MAX = 0.005
INERCIA = 0.95
SENSIBILIDAD_GRAD = 10.0
K_GAIN = 0.0003
K_PRECISION = 0.002
K_TEMBLOR = 0.001
TAU_RECUPERACION = 300.0

# Memoria de ausencia
TAU_BASE = 30.0
K_MEM = 0.005
SUELO_CONFIANZA = 0.2
K_HOLD = 0.001

# Consciencia básica (Cb) — el "umbral Cb>28" que menciona Grok viene de aquí + RITUAL_UMBRAL_CB
TAU_CB = 10.0
CB_MAX = 500.0

# Juego
UMBRAL_CB_JUEGO = 35.0
K_INFLUENCIA_JUEGO = 0.00035

# RITUAL (los "cruces de cero", "umbral Cb>28", "decaimiento exp(-dt/180)", "patrón temporal 40s" etc.)
RITUAL_TAU = 180.0
RITUAL_REPETICION_MIN = 3
RITUAL_GAIN = 0.05
RITUAL_PATRON_TEMPORAL = 40.0
RITUAL_TOLERANCIA = 0.3
RITUAL_UMBRAL_ACTIVACION = 0.4
RITUAL_UMBRAL_CB = 28.0          # <--- el umbral explícito que menciona Grok
RITUAL_SALIDA_SUAVE = 0.95
RITUAL_PERSISTENCIA_MIN = 3

# META-REPRESENTACIÓN OBSERVACIONAL (Rᴿ)
# "presión = error_norm·Cb_norm", "integración de desajuste", leaky integrator, etc.
META_TAU = 30.0
META_UMBRAL_DESAJUSTE = 0.5
META_VENTANA_ERROR = 200
META_K_SUAVIDAD = 0.1

SEMILLA_BASE = 44
PERIODO_ALTERNANCIA = 80.0


# ============================================================
# CLASE RITUALV167 (COMPLETA) — detector de cruces de cero + patrón + activación
# ============================================================

class RitualV167:
    def __init__(self, tau=RITUAL_TAU, repeticion_min=RITUAL_REPETICION_MIN,
                 ritual_gain=RITUAL_GAIN, patron_temporal=RITUAL_PATRON_TEMPORAL,
                 tolerancia=RITUAL_TOLERANCIA, umbral_activacion=RITUAL_UMBRAL_ACTIVACION,
                 umbral_cb=RITUAL_UMBRAL_CB, salida_suave=RITUAL_SALIDA_SUAVE,
                 persistencia_min=RITUAL_PERSISTENCIA_MIN):
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
        """Detector explícito de cruces por cero en la orientación del motor."""
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
        """
        Regla completa:
        - Si hay cruce Y Cb > umbral (28.0) → mirar buffer de tiempos de cruce.
        - Si hay patrón temporal repetido (dt ≈ 40s ± tolerancia) suficiente veces → subir activation.
        - Siempre: activation *= exp(-dt / tau)  (decaimiento exponencial explícito)
        - Si activation > 0.4 → active = True
        - Persistencia: si active, se mantiene hasta que pasen muchos ciclos_sin_cruce.
        """
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
        
        # DECAIMIENTO EXPLÍCITO (el exp(-dt/180) que menciona Grok)
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
        """Cuando el ritual está activo, modula la corrección motora (hace el comportamiento más 'rígido')."""
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
# CLASE METAREPRESENTACIONOBSERVACIONAL (COMPLETA) — el monitor Rᴿ
# ============================================================

class MetaRepresentacionObservacional:
    """
    Etapa 4: Monitor observacional de desajuste (SOLO OBSERVA, NO INHIBE).
    
    Reglas explícitas que implementan exactamente lo que Grok señaló:
    - buffer de error, Cb y ritual_activo
    - error_sostenido = mean del buffer
    - Si ritual_sostenido:
        error_norm = min(1.0, error_sostenido / 60.0)
        Cb_norm = min(1.0, Cb_sostenido / 500.0)
        presion_activa = error_norm * Cb_norm
        + caso especial "ritual ciego" (Cb baja + error alto) → presion_ciega
      presion = max(...)
    - Integrador leaky: d_desajuste = presion - self.desajuste / tau
      self.desajuste += d_desajuste * dt
    - Retorna (senal, hay_desajuste > umbral)
    
    Todo está escrito explícitamente aquí. No hay "magia" ni aprendizaje no supervisado.
    La correlación surge cuando estas condiciones se dan de forma sostenida en F3 y persisten en F4.
    """
    
    def __init__(self, tau=META_TAU, umbral_desajuste=META_UMBRAL_DESAJUSTE,
                 ventana_error=META_VENTANA_ERROR, k_suavidad=META_K_SUAVIDAD):
        self.tau = tau
        self.umbral_desajuste = umbral_desajuste
        self.ventana_error = ventana_error
        self.k_suavidad = k_suavidad
        
        self.desajuste = 0.0
        self.historial_desajuste = []
        self.buffer_error = deque(maxlen=ventana_error)
        self.buffer_Cb = deque(maxlen=ventana_error)
        self.buffer_ritual = deque(maxlen=ventana_error)
    
    def actualizar(self, error, Cb, ritual_activo, dt):
        """
        Implementación completa de "presión = error_norm·Cb_norm" + integración.
        """
        self.buffer_error.append(abs(error))
        self.buffer_Cb.append(Cb)
        self.buffer_ritual.append(ritual_activo)
        
        if len(self.buffer_error) > self.ventana_error // 2:
            error_sostenido = np.mean(self.buffer_error)
            Cb_sostenido = np.mean(self.buffer_Cb)
            ritual_sostenido = np.mean(self.buffer_ritual) > 0.5
        else:
            error_sostenido = abs(error)
            Cb_sostenido = Cb
            ritual_sostenido = ritual_activo
        
        if ritual_sostenido:
            error_norm = min(1.0, error_sostenido / 60.0)
            Cb_norm = min(1.0, Cb_sostenido / 500.0)
            
            presion_activa = error_norm * Cb_norm
            
            # Caso "ritual ciego"
            if Cb_sostenido < 50 and error_sostenido > 30:
                presion_ciega = error_norm * 0.8
            else:
                presion_ciega = 0.0
            
            presion = max(presion_activa, presion_ciega)
        else:
            presion = 0.0
        
        # Integrador leaky explícito
        d_desajuste = presion - self.desajuste / self.tau
        self.desajuste += d_desajuste * dt
        self.desajuste = max(0.0, min(1.0, self.desajuste))
        
        self.historial_desajuste.append(self.desajuste)
        
        senal_suavizada = self.desajuste
        return senal_suavizada, senal_suavizada > self.umbral_desajuste
    
    def reset(self):
        self.desajuste = 0.0
        self.historial_desajuste = []
        self.buffer_error.clear()
        self.buffer_Cb.clear()
        self.buffer_ritual.clear()


# ============================================================
# FRAGMENTO DE APARATOMOTORV167 (integración y jerarquía)
# ============================================================

# (Se incluyen solo las partes relevantes para que se vea cómo Ritual y Meta se cablean.
# El motor completo también contiene FatigaMetabolicaV167, MemoriaAusenciaV167, etc.)

class AparatoMotorV167:  # versión simplificada para el extracto — ver v167-ob.py para el resto
    def __init__(self, enable_meta=True):
        # ... (inicializaciones de fatiga, memoria, consciencia, juego, etc.)
        self.ritual = RitualV167()
        self.meta = MetaRepresentacionObservacional() if enable_meta else None
        self.enable_meta = enable_meta
        # ... resto de estado (orientacion, Kp, inercia, etc.)

    def actuar(self, gradiente, LF_activa, fuente_activa, t, setpoint_raw, dt=DT):
        # ... (cálculo de Cb, error, etc. — ver script completo)
        
        # ETAPA 3: Ritual (se actualiza primero)
        ritual_activo = self.ritual.actualizar(self.orientacion, Cb, t, dt)
        
        # ETAPA 4: Meta-representación observacional (SOLO OBSERVA)
        senal_desajuste = 0.0
        hay_desajuste = False
        if self.enable_meta and self.meta is not None:
            senal_desajuste, hay_desajuste = self.meta.actualizar(error, Cb, ritual_activo, dt)
            # IMPORTANTE: NO se inhibe el ritual aquí. Solo se registra la señal.
        
        # ETAPA 2: Juego (INHIBIDO explícitamente si ritual_activo)
        if ritual_activo:
            juego_activo = False
            # ...
        else:
            juego_activo = self.juego.actualizar(...)
        
        # ... luego se usa ritual_activo y senal_desajuste para logging y para modular_correccion
        
        # La modulación ritual (cuando active) ocurre más adelante en el cálculo de delta:
        # delta_raw = self.ritual.modular_correccion(delta_raw, correccion_ritual)
        
        # return ... incluye ritual_activo, self.ritual.activation, senal_desajuste, hay_desajuste


# ============================================================
# LÓGICA DE CORRELACIÓN (la corrección mencionada en el encabezado del script)
# ============================================================

def calcular_correlacion_ritual_senal_corregida(f3_ritual):
    """
    Versión corregida (evita NaN) usada en los resultados V167 CORREGIDO.
    Extraída de v167-ob.py líneas ~916-940.
    """
    correlacion = 0.0
    if len(f3_ritual['ritual_activo']) > 100 and len(f3_ritual['senal_desajuste']) > 100:
        inicio = len(f3_ritual['ritual_activo']) // 4
        fin = 3 * len(f3_ritual['ritual_activo']) // 4
        ritual_vals = np.array(f3_ritual['ritual_activo'][inicio:fin], dtype=float)
        senal_vals = np.array(f3_ritual['senal_desajuste'][inicio:fin], dtype=float)
        
        if np.std(ritual_vals) > 1e-6 and np.std(senal_vals) > 1e-6:
            correlacion = np.corrcoef(ritual_vals, senal_vals)[0, 1]
        else:
            correlacion = 0.0
            if len(ritual_vals) > 100:
                step = max(1, len(ritual_vals) // 200)
                ritual_down = ritual_vals[::step]
                senal_down = senal_vals[::step]
                if np.std(ritual_down) > 1e-6 and np.std(senal_down) > 1e-6:
                    correlacion = np.corrcoef(ritual_down, senal_down)[0, 1]
    return correlacion


# ============================================================
# CÓDIGO DE USO MÍNIMO (para que se pueda instanciar y ver las reglas)
# ============================================================

if __name__ == "__main__":
    print("Clases RitualV167 y MetaRepresentacionObservacional listas.")
    print("Parámetros clave:")
    print(f"  RITUAL_UMBRAL_CB = {RITUAL_UMBRAL_CB}")
    print(f"  RITUAL_PATRON_TEMPORAL = {RITUAL_PATRON_TEMPORAL}s")
    print(f"  RITUAL_TAU (decaimiento) = {RITUAL_TAU}s → exp(-dt/{RITUAL_TAU})")
    print(f"  META_TAU = {META_TAU}")
    print(f"  META_UMBRAL_DESAJUSTE = {META_UMBRAL_DESAJUSTE}")
    print(f"  META_UMBRAL_ERROR implícito en la lógica de 'ritual ciego' y error_norm/60")

    r = RitualV167()
    m = MetaRepresentacionObservacional()
    print("\nInstancias creadas exitosamente. Ver v167-ob.py para el protocolo completo de 4 etapas + logging de historiales.")
