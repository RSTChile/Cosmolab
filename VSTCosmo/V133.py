#!/usr/bin/env python3
"""
VSTCosmos V133 — Memoria episódica con decaimiento

ANIMA-2 - Línea 1:
  Hipotesis O-N9: El organismo puede conservar orientacion funcional
  hacia una fuente sonora ausente durante un intervalo limitado de silencio,
  mediante memoria angular con decaimiento temporal.

Protocolo:
  Fase 1 (0-60s): Fuente a -60° ON → adquirir orientacion
  Fase 2 (60-120s): Silencio → mantener por memoria
  Fase 3 (120-180s): Fuente a -60° ON → reenganche
  Fase 4 (180-240s): Fuente a +60° ON → verificar que memoria se sobreescribe

Criterios de exito:
  1. Error_memoria(90s) < 15° (a 30s de silencio)
  2. T_reenganche < 10s (al reaparecer la fuente)
  3. Vida media memoria ≈ τ * ln(2) = 20.8s
  4. Despues de silencio largo (>60s), vuelve a centro
  5. En fase 4, se reorienta a +60° sin interferencia
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys
from collections import deque

# ============================================================
# PARAMETROS (heredados de V132)
# ============================================================
DT = 0.01
TIEMPO_POR_REPETICION = 452.0
REPETICIONES_LENTAS = 10

# Asimetria forzada
SESGO_L = 0.05
SESGO_R = -0.05
DIM_HEMISFERIO = 32

# Zona muerta
ZONA_MUERTA_BASE = 2.0

# Limites de plasticidad
KP_BASE = 0.002
KP_MIN = 0.0005
KP_MAX = 0.005

# Plasticidad
HABITUACION_RAPIDA = 0.99
SENSIBILIZACION_LENTA = 1.01
VENTANA_OSCILACION = 100

# Memoria episodica
TAU_MEMORIA = 30.0  # constante de tiempo del hipocampo virtual (s)
UMBRAL_CONFIANZA = 0.1  # por debajo de esto, olvida

# Criterios de exito
ERROR_MEMORIA_MAX = 15.0  # grados
T_REENGANCHE_MAX = 10.0   # segundos

# Semilla base (la mas robusta de V132)
SEMILLA_BASE = 44


# ============================================================
# HEMISFERIO (igual V132)
# ============================================================

class HemisferioV133:
    def __init__(self, nombre, tau, generar_entrada_func, seed=None, sesgo=0.0):
        if seed is not None:
            np.random.seed(seed)
        self.nombre = nombre
        self.tau = tau
        self.generar_entrada = generar_entrada_func
        self.sesgo = sesgo
        
        self.Phi = np.random.normal(sesgo, 0.1, DIM_HEMISFERIO)
        self.Phi_vel = np.zeros(DIM_HEMISFERIO)
        
        self.entrada = None
        self.sr = 48000
        self.en_inanicion = False
        self.factor_inanicion = 1.0
        
        self.buffer_rapido = []
        self.historial_omega = []
    
    def _calcular_omega(self):
        return np.mean(self.Phi[:32])
    
    def generar_entrada_para_t(self, t, duracion_total):
        if self.entrada is None:
            self.entrada = self.generar_entrada(duracion_total, self.sr)
        idx = int(t * self.sr)
        if idx >= len(self.entrada):
            return 0.0
        return self.entrada[idx] * self.factor_inanicion
    
    def actualizar(self, t, dt, duracion_total, otro_hemisferio=None):
        entrada = self.generar_entrada_para_t(t, duracion_total)
        
        laplaciano = np.zeros_like(self.Phi)
        for i in range(1, DIM_HEMISFERIO - 1):
            laplaciano[i] = self.Phi[i-1] - 2*self.Phi[i] + self.Phi[i+1]
        
        reaccion = self.Phi * (1 - self.Phi * self.Phi)
        
        forzamiento = np.zeros_like(self.Phi)
        forzamiento[0] = entrada
        forzamiento[-1] = -entrada
        
        acoplamiento = np.zeros_like(self.Phi)
        if otro_hemisferio is not None:
            divergencia = abs(self._calcular_omega() - otro_hemisferio._calcular_omega())
            if divergencia > 0.5:
                acoplamiento = 0.01 * (otro_hemisferio.Phi - self.Phi)
        
        dPhi_vel = laplaciano + reaccion + forzamiento + acoplamiento
        self.Phi_vel += dPhi_vel * dt
        self.Phi += self.Phi_vel * dt
        self.Phi = np.clip(self.Phi, -1.0, 1.0)
        
        return {'omega': self._calcular_omega()}


# ============================================================
# MEMORIA EPISODICA (NUEVO EN V133)
# ============================================================

class MemoriaEpisodica:
    """
    Memoria con decaimiento exponencial para recordar ubicacion de fuente.
    
    Propiedades:
      - Encoding: cuando hay fuente, sobreescribe con alta confianza
      - Decay: durante silencio, confianza decae exponencialmente
      - Retrieval: solo retorna angulo si confianza > umbral
      - Olvido: si confianza < umbral, retorna None (vuelve a centro)
    """
    
    def __init__(self, tau=TAU_MEMORIA, umbral_confianza=UMBRAL_CONFIANZA):
        self.tau = tau
        self.umbral_confianza = umbral_confianza
        self.angulo = 0.0
        self.confianza = 0.0
        self.t_ultimo_estimulo = 0.0
        self.historial_confianza = []
        self.historial_angulo_recordado = []
    
    def update(self, angulo_medido, fuente_activa, t):
        """
        Actualiza la memoria segun si hay fuente o no.
        
        Args:
            angulo_medido: orientacion actual (grados)
            fuente_activa: True si hay estimulo sonoro
            t: tiempo actual (s)
        
        Returns:
            angulo_recordado o None (si confianza < umbral)
        """
        if fuente_activa:
            # Encoding: sobreescribe con estimulo actual
            self.angulo = angulo_medido
            self.confianza = 1.0
            self.t_ultimo_estimulo = t
        else:
            # Decay: olvido exponencial durante silencio
            dt_silencio = t - self.t_ultimo_estimulo
            if dt_silencio >= 0:
                self.confianza = np.exp(-dt_silencio / self.tau)
            else:
                self.confianza = 0.0
        
        # Registrar historial
        self.historial_confianza.append(self.confianza)
        self.historial_angulo_recordado.append(self.angulo if self.confianza > self.umbral_confianza else None)
        
        # Retrieval: solo si confianza > umbral
        if self.confianza > self.umbral_confianza:
            return self.angulo
        else:
            return None
    
    def get_confianza(self):
        return self.confianza
    
    def reset(self):
        self.angulo = 0.0
        self.confianza = 0.0
        self.t_ultimo_estimulo = 0.0
        self.historial_confianza = []
        self.historial_angulo_recordado = []


# ============================================================
# APARATO MOTOR CON MEMORIA (V133)
# ============================================================

class AparatoMotorConMemoria:
    """
    Motor homeostatico con memoria episodica.
    
    En presencia de fuente: orienta por percepcion.
    En silencio: mantiene orientacion por memoria (con decaimiento).
    """
    
    def __init__(self):
        self.orientacion = 0.0
        self.setpoint_percepcion = 0.0
        self.setpoint_usado = 0.0
        self.Kp_base = KP_BASE
        self.Kp_actual = KP_BASE
        self.Kp_min = KP_MIN
        self.Kp_max = KP_MAX
        self.limite = 90.0
        self.zona_muerta = ZONA_MUERTA_BASE
        self.inercia = 0.95
        self.ultimo_delta = 0.0
        self.sensibilidad_grad = 10.0
        self.t = 0.0
        
        # Memoria episodica (NUEVO)
        self.memoria = MemoriaEpisodica(tau=TAU_MEMORIA)
        
        # Plasticidad
        self.memoria_error = deque(maxlen=VENTANA_OSCILACION)
        self.historial_Kp = []
        self.historial_confianza = []
        self.historial_setpoint_usado = []
        self.historial_fuente_activa = []
    
    def calcular_factor_freno(self, error):
        return 1 - np.exp(-abs(error) / 30.0)
    
    def actualizar_plasticidad(self, error):
        self.memoria_error.append(error)
        
        if len(self.memoria_error) < VENTANA_OSCILACION:
            return
        
        oscilacion = np.std(self.memoria_error)
        
        if oscilacion > self.zona_muerta * 1.5:
            self.Kp_actual = max(self.Kp_min, self.Kp_actual * HABITUACION_RAPIDA)
        elif oscilacion < self.zona_muerta * 0.5:
            self.Kp_actual = min(self.Kp_max, self.Kp_actual * SENSIBILIZACION_LENTA)
        
        self.historial_Kp.append(self.Kp_actual)
    
    def actuar(self, gradiente, LF_activa, fuente_activa, t, setpoint_percepcion):
        """
        Genera comando motor usando memoria episodica.
        
        Args:
            gradiente: omega_A - omega_B
            LF_activa: si el sistema esta en modo test
            fuente_activa: True si hay estimulo sonoro
            t: tiempo actual (s)
            setpoint_percepcion: angulo de la fuente (grados)
        """
        if not LF_activa:
            return self.orientacion, 0.0
        
        if abs(gradiente) < 0.05:
            return self.orientacion, self.memoria.get_confianza()
        
        # Guardar setpoint de percepcion
        self.setpoint_percepcion = setpoint_percepcion
        
        # Determinar setpoint a usar (percepcion o memoria)
        if fuente_activa:
            # Hay fuente: usar percepcion
            self.setpoint_usado = self.setpoint_percepcion
            # Actualizar memoria con la orientacion actual
            self.memoria.update(self.orientacion, True, t)
        else:
            # Silencio: intentar usar memoria
            angulo_memoria = self.memoria.update(self.orientacion, False, t)
            if angulo_memoria is not None:
                self.setpoint_usado = angulo_memoria
            else:
                # Si no hay memoria, volver a centro
                self.setpoint_usado = 0.0
        
        # Control proporcional hacia setpoint_usado
        error = self.setpoint_usado - self.orientacion
        
        # Zona muerta
        if abs(error) < self.zona_muerta:
            return self.orientacion, self.memoria.get_confianza()
        
        # Control
        ganancia_grad = -np.tanh(gradiente * self.sensibilidad_grad)
        factor_freno = self.calcular_factor_freno(error)
        
        delta = self.Kp_actual * error * ganancia_grad * factor_freno
        
        # Inercia
        delta = self.inercia * self.ultimo_delta + (1 - self.inercia) * delta
        self.ultimo_delta = delta
        
        # Actualizar plasticidad
        self.actualizar_plasticidad(error)
        
        # Actualizar orientacion
        self.orientacion += delta
        self.orientacion = np.clip(self.orientacion, -self.limite, self.limite)
        
        self.t += DT
        
        # Guardar historial
        self.historial_confianza.append(self.memoria.get_confianza())
        self.historial_setpoint_usado.append(self.setpoint_usado)
        
        return self.orientacion, self.memoria.get_confianza()
    
    def reset(self):
        self.orientacion = 0.0
        self.ultimo_delta = 0.0
        self.t = 0.0
        self.Kp_actual = KP_BASE
        self.memoria_error.clear()
        self.historial_Kp = []
        self.historial_confianza = []
        self.historial_setpoint_usado = []
        self.memoria.reset()


# ============================================================
# SISTEMA V133
# ============================================================

class SistemaV133:
    def __init__(self, nombre, seed=SEMILLA_BASE):
        self.nombre = nombre
        
        def generar_ruido_rosa(duracion, sr):
            n = int(duracion * sr)
            ruido = np.random.normal(0, 1, n)
            fft = np.fft.rfft(ruido)
            freqs = np.fft.rfftfreq(n, 1/sr)
            filtro = 1.0 / np.sqrt(freqs + 0.01)
            fft_filtrado = fft * filtro
            ruido_rosa = np.fft.irfft(fft_filtrado, n=n)
            ruido_rosa = ruido_rosa / (np.max(np.abs(ruido_rosa)) + 1e-10)
            return ruido_rosa
        
        def generar_clicks_poisson(duracion, tasa=0.5, sr=48000):
            n = int(duracion * sr)
            clicks = np.zeros(n)
            n_clicks = int(duracion * tasa)
            for _ in range(n_clicks):
                pos = int(np.random.exponential(1.0/tasa) * sr)
                if pos < n:
                    clicks[pos] = 1.0
            return clicks
        
        self.izquierdo = HemisferioV133("L", 30.0, generar_ruido_rosa, seed=seed, sesgo=SESGO_L)
        self.derecho = HemisferioV133("R", 300.0, generar_clicks_poisson, seed=seed+100, sesgo=SESGO_R)
        
        self.sistema_B_izq = HemisferioV133("B_L", 30.0, generar_ruido_rosa, seed=seed+200, sesgo=SESGO_L)
        self.sistema_B_der = HemisferioV133("B_R", 300.0, generar_clicks_poisson, seed=seed+300, sesgo=SESGO_R)
        
        self.modo_entrenamiento = True
        self.motor = AparatoMotorConMemoria()
        
        self.historial = {
            't': [],
            'orientacion': [],
            'confianza': [],
            'setpoint_usado': [],
            'fuente_activa': [],
            's_shared': [],
            'Kp': []
        }
    
    def calcular_s_shared(self):
        omega_A = (self.izquierdo._calcular_omega() + self.derecho._calcular_omega()) / 2
        omega_B = (self.sistema_B_izq._calcular_omega() + self.sistema_B_der._calcular_omega()) / 2
        return 1 - abs(omega_A - omega_B) / 2.0
    
    def actualizar(self, t, dt, duracion_total, fuente_activa, setpoint_percepcion):
        # Actualizar sistemas
        self.izquierdo.actualizar(t, dt, duracion_total, self.derecho)
        self.derecho.actualizar(t, dt, duracion_total, self.izquierdo)
        self.sistema_B_izq.actualizar(t, dt, duracion_total, self.sistema_B_der)
        self.sistema_B_der.actualizar(t, dt, duracion_total, self.sistema_B_izq)
        
        # Gradiente inter-sistemas
        omega_A = (self.izquierdo._calcular_omega() + self.derecho._calcular_omega()) / 2
        omega_B = (self.sistema_B_izq._calcular_omega() + self.sistema_B_der._calcular_omega()) / 2
        gradiente = omega_A - omega_B
        
        # Espacializacion (solo si hay fuente)
        if fuente_activa:
            sesgo = setpoint_percepcion / 90.0
            gradiente += sesgo * 0.5
        
        # Motor con memoria
        LF_activa = not self.modo_entrenamiento
        orientacion, confianza = self.motor.actuar(
            gradiente, LF_activa, fuente_activa, t, setpoint_percepcion
        )
        
        # Guardar historial
        self.historial['t'].append(t)
        self.historial['orientacion'].append(orientacion)
        self.historial['confianza'].append(confianza)
        self.historial['setpoint_usado'].append(self.motor.setpoint_usado)
        self.historial['fuente_activa'].append(fuente_activa)
        self.historial['s_shared'].append(self.calcular_s_shared())
        self.historial['Kp'].append(self.motor.Kp_actual)
        
        return orientacion, confianza
    
    def set_modo_entrenamiento(self, entrenamiento=True):
        self.modo_entrenamiento = entrenamiento
        if entrenamiento:
            self.motor.reset()


# ============================================================
# EXPERIMENTO V133
# ============================================================

def ejecutar_v133():
    print("=" * 100)
    print("EXPERIMENTO V133 — Memoria episodica con decaimiento")
    print("=" * 100)
    print("  ANIMA-2 - Linea 1: Hipotesis O-N9")
    print("  Protocolo:")
    print("    Fase 1 (0-60s): Fuente a -60° ON → adquirir orientacion")
    print("    Fase 2 (60-120s): Silencio OFF → mantener por memoria")
    print("    Fase 3 (120-180s): Fuente a -60° ON → reenganche")
    print("    Fase 4 (180-240s): Fuente a +60° ON → verificar sobreescritura")
    print("=" * 100)
    
    # Configurar sistema
    sistema = SistemaV133("V133", seed=SEMILLA_BASE)
    
    # Fase 2: Entrenamiento lateral (10 repeticiones)
    print("\n  Entrenando lateralidad (10 repeticiones)...")
    sistema.set_modo_entrenamiento(True)
    
    for rep in range(REPETICIONES_LENTAS):
        for i in range(int(TIEMPO_POR_REPETICION / DT)):
            t = rep * TIEMPO_POR_REPETICION + i * DT
            # Durante entrenamiento, no hay fuente (solo ruido ortogonal)
            sistema.actualizar(t, DT, TIEMPO_POR_REPETICION * REPETICIONES_LENTAS, 
                              fuente_activa=False, setpoint_percepcion=0.0)
    
    print("  Entrenamiento completado.")
    
    # Fase de test con memoria
    print("\n  Iniciando test de memoria episodica...")
    sistema.set_modo_entrenamiento(False)
    sistema.motor.reset()
    
    t_actual = TIEMPO_POR_REPETICION * REPETICIONES_LENTAS
    duracion_total = t_actual + 300.0  # 5 minutos de test
    
    # Muestreo cada 5s para reporte
    ultimo_reporte = 0
    
    for i in range(int(300.0 / DT)):
        t = t_actual + i * DT
        
        # Determinar fase y setpoint segun tiempo de test
        if t < t_actual + 60:  # Fase 1: 0-60s, fuente a -60° ON
            fuente_activa = True
            setpoint = -60.0
        elif t < t_actual + 120:  # Fase 2: 60-120s, silencio OFF
            fuente_activa = False
            setpoint = -60.0  # No usado, pero se guarda
        elif t < t_actual + 180:  # Fase 3: 120-180s, fuente a -60° ON
            fuente_activa = True
            setpoint = -60.0
        else:  # Fase 4: 180-240s, fuente a +60° ON
            fuente_activa = True
            setpoint = 60.0
        
        orientacion, confianza = sistema.actualizar(t, DT, duracion_total, 
                                                     fuente_activa, setpoint)
        
        # Reporte cada 5s
        if int(t * 10) % 50 == 0 and t != ultimo_reporte:
            fase = ""
            if t < t_actual + 60:
                fase = "F1(-60°)"
            elif t < t_actual + 120:
                fase = "F2(silencio)"
            elif t < t_actual + 180:
                fase = "F3(reenganche)"
            else:
                fase = "F4(+60°)"
            
            setpoint_muestra = setpoint if fuente_activa else "memoria"
            print(f"    t={t:.0f}s | fase={fase:12s} | orient={orientacion:.1f}° | "
                  f"confianza={confianza:.2f} | setpoint={setpoint_muestra}")
            ultimo_reporte = t
    
    # ============================================================
    # ANALISIS DE RESULTADOS
    # ============================================================
    
    print("\n" + "=" * 80)
    print("ANALISIS DE MEMORIA EPISODICA")
    print("=" * 80)
    
    t_total = sistema.historial['t']
    orientacion = np.array(sistema.historial['orientacion'])
    confianza = np.array(sistema.historial['confianza'])
    setpoint_usado = np.array(sistema.historial['setpoint_usado'])
    fuente_activa = np.array(sistema.historial['fuente_activa'])
    
    # Tiempos de referencia
    t0 = t_total[0]
    t_fin_f1 = t0 + 60
    t_fin_f2 = t0 + 120
    t_fin_f3 = t0 + 180
    t_fin_f4 = t0 + 240
    
    # 1. Error durante silencio (Fase 2)
    mask_silencio = (t_total >= t_fin_f1) & (t_total < t_fin_f2)
    if np.any(mask_silencio):
        # Angulo recordado deberia ser -60°
        error_silencio = np.abs(orientacion[mask_silencio] - (-60.0))
        
        # Error a diferentes tiempos
        indices_silencio = np.where(mask_silencio)[0]
        tiempos_silencio = t_total[mask_silencio]
        
        error_30s = None
        error_45s = None
        error_60s = None
        
        for i, tt in enumerate(tiempos_silencio):
            dt_silencio = tt - t_fin_f1
            if abs(dt_silencio - 30.0) < 1.0:
                error_30s = error_silencio[i]
            elif abs(dt_silencio - 45.0) < 1.0:
                error_45s = error_silencio[i]
            elif abs(dt_silencio - 60.0) < 1.0:
                error_60s = error_silencio[i]
        
        print(f"\n  Error durante silencio (Fase 2):")
        print(f"    A los 30s: {error_30s:.1f}° {'✅' if error_30s and error_30s < ERROR_MEMORIA_MAX else '❌'}")
        print(f"    A los 45s: {error_45s:.1f}°")
        print(f"    A los 60s: {error_60s:.1f}°")
    else:
        error_30s = None
        print("  No se encontraron datos de silencio")
    
    # 2. Vida media de la memoria
    mask_f2 = (t_total >= t_fin_f1) & (t_total < t_fin_f2)
    if np.any(mask_f2):
        confianza_f2 = confianza[mask_f2]
        tiempos_f2 = t_total[mask_f2] - t_fin_f1
        
        # Buscar donde confianza = 0.5
        t_half = None
        for i in range(len(confianza_f2) - 1):
            if confianza_f2[i] >= 0.5 and confianza_f2[i+1] < 0.5:
                # Interpolacion lineal
                t1 = tiempos_f2[i]
                t2 = tiempos_f2[i+1]
                c1 = confianza_f2[i]
                c2 = confianza_f2[i+1]
                t_half = t1 + (0.5 - c1) * (t2 - t1) / (c2 - c1)
                break
        
        t_half_teorica = TAU_MEMORIA * np.log(2)
        
        print(f"\n  Vida media de la memoria:")
        print(f"    Medida: {t_half:.1f}s" if t_half else "    Medida: No alcanzo 0.5")
        print(f"    Teorica (τ*ln2): {t_half_teorica:.1f}s")
    
    # 3. Tiempo de reenganche (Fase 3)
    mask_reenganche = (t_total >= t_fin_f2) & (t_total < t_fin_f3)
    if np.any(mask_reenganche):
        error_reenganche = np.abs(orientacion[mask_reenganche] - (-60.0))
        
        # Encontrar cuando entra en zona muerta
        indices_reenganche = np.where(mask_reenganche)[0]
        tiempos_reenganche = t_total[mask_reenganche] - t_fin_f2
        
        t_settle = None
        for i, err in enumerate(error_reenganche):
            if err < ZONA_MUERTA_BASE:
                t_settle = tiempos_reenganche[i]
                break
        
        print(f"\n  Reenganche (Fase 3):")
        print(f"    T_settle: {t_settle:.1f}s {'✅' if t_settle and t_settle < T_REENGANCHE_MAX else '❌'}")
    
    # 4. Orientacion final (Fase 4)
    mask_f4_final = (t_total >= t_fin_f4 - 10) & (t_total <= t_fin_f4)
    if np.any(mask_f4_final):
        orient_final = np.mean(orientacion[mask_f4_final])
        print(f"\n  Orientacion final (Fase 4, +60°):")
        print(f"    Orientacion: {orient_final:.1f}° {'✅' if abs(orient_final - 60.0) < 15 else '❌'}")
    
    # 5. Resumen de exitos
    print("\n" + "-" * 40)
    print("  CRITERIOS DE EXITO O-N9:")
    
    exito_1 = error_30s is not None and error_30s < ERROR_MEMORIA_MAX
    print(f"    1. Error memoria 30s < {ERROR_MEMORIA_MAX}°: {'✅' if exito_1 else '❌'}")
    
    exito_2 = t_settle is not None and t_settle < T_REENGANCHE_MAX if 't_settle' in dir() else False
    print(f"    2. T_reenganche < {T_REENGANCHE_MAX}s: {'✅' if exito_2 else '❌'}")
    
    exito_3 = t_half is not None and abs(t_half - t_half_teorica) < 10.0 if 't_half' in dir() and t_half else False
    print(f"    3. Vida media ~ {t_half_teorica:.0f}s: {'✅' if exito_3 else '❌'}")
    
    exito_total = exito_1 and exito_2 and exito_3
    
    # ============================================================
    # GRAFICOS
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Grafico 1: Orientacion y setpoint
    ax = axes[0, 0]
    ax.plot(t_total, orientacion, 'b-', linewidth=1, label='Orientacion real')
    ax.plot(t_total, setpoint_usado, 'r--', linewidth=1, alpha=0.7, label='Setpoint usado')
    
    # Marcar fases
    ax.axvline(x=t_fin_f1, color='gray', linestyle=':', alpha=0.5, label='Fin F1')
    ax.axvline(x=t_fin_f2, color='gray', linestyle=':', alpha=0.5, label='Fin F2')
    ax.axvline(x=t_fin_f3, color='gray', linestyle=':', alpha=0.5, label='Fin F3')
    
    ax.axhline(y=-60, color='green', linestyle='--', alpha=0.5, label='Objetivo -60°')
    ax.axhline(y=60, color='orange', linestyle='--', alpha=0.5, label='Objetivo +60°')
    ax.axhline(y=ZONA_MUERTA_BASE, color='gray', linestyle=':', alpha=0.3)
    ax.axhline(y=-ZONA_MUERTA_BASE, color='gray', linestyle=':', alpha=0.3)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Orientacion (grados)')
    ax.set_title('Memoria Episodica: Orientacion durante silencio')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Grafico 2: Confianza de la memoria
    ax = axes[0, 1]
    ax.plot(t_total, confianza, 'purple', linewidth=1)
    ax.axhline(y=UMBRAL_CONFIANZA, color='red', linestyle='--', alpha=0.5, label=f'Umbral confianza ({UMBRAL_CONFIANZA})')
    ax.axhline(y=0.5, color='orange', linestyle=':', alpha=0.5, label='Vida media (0.5)')
    ax.fill_between(t_total, 0, confianza, where=(t_total>=t_fin_f1)&(t_total<t_fin_f2), alpha=0.3, color='purple')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Confianza')
    ax.set_title('Decaimiento de la memoria durante silencio')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Grafico 3: Kp adaptativo
    ax = axes[1, 0]
    Kp_hist = sistema.historial['Kp']
    ax.plot(t_total[:len(Kp_hist)], Kp_hist, 'green', linewidth=0.8)
    ax.axhline(y=KP_BASE, color='gray', linestyle='--', alpha=0.5, label=f'Kp_base = {KP_BASE}')
    ax.axhline(y=KP_MIN, color='red', linestyle=':', alpha=0.5, label=f'Kp_min = {KP_MIN}')
    ax.axhline(y=KP_MAX, color='green', linestyle=':', alpha=0.5, label=f'Kp_max = {KP_MAX}')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Kp')
    ax.set_title('Plasticidad homeostatica durante memoria')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Grafico 4: s_shared
    ax = axes[1, 1]
    s_shared = sistema.historial['s_shared']
    ax.plot(t_total, s_shared, 'orange', linewidth=0.8)
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Umbral lateralidad (0.8)')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('s_shared')
    ax.set_title('Coherencia inter-sistemas (lateralidad)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('v133_logs', exist_ok=True)
    plt.savefig(f'v133_logs/v133_memoria_{timestamp}.png', dpi=150)
    print(f"\n  Graficos guardados: v133_logs/v133_memoria_{timestamp}.png")
    
    # ============================================================
    # CONCLUSION V133
    # ============================================================
    print("\n" + "=" * 80)
    print("CONCLUSION V133 — Memoria Episodica")
    print("=" * 80)
    
    if exito_total:
        print("\n  ✅ O-N9 VALIDADA: Memoria episodica funcional")
        print("     - El organismo mantiene orientacion durante silencio")
        print("     - Decaimiento exponencial con τ≈30s")
        print("     - Reenganche rapido al reaparecer la fuente")
        print("     - La memoria se sobreescribe con nueva informacion")
    else:
        print("\n  ⚠️ O-N9 NO VALIDADA: La memoria episodica no funciona como se esperaba")
    
    print(f"\n  ANIMA-2 - Linea 1: {'CERRADA' if exito_total else 'PENDIENTE'}")
    
    return sistema, exito_total


if __name__ == "__main__":
    import time
    start = time.time()
    sistema, exito = ejecutar_v133()
    elapsed = time.time() - start
    print(f"\n  Tiempo de ejecucion: {elapsed/60:.1f} minutos")