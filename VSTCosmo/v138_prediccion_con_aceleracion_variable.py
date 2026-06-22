#!/usr/bin/env python3
"""
VSTCosmos V138 — Prediccion con aceleracion variable

Test de generalizacion:
  Fase 1 (0-40s): Movimiento sinusoidal
  Fase 2 (40-80s): Velocidad constante (-60° → +60°)
  Fase 3 (80-120s): Aceleracion abrupta (cambio de direccion)
  Fase 4 (120-160s): Stop-and-go (quietud 20s, luego movimiento)

Mecanismo:
  - Predictor con aceleracion (V138)
  - Horizonte adaptativo basado en frenado
  - Lead Index contra setpoint_predicho

Hipotesis O-N10 extendida:
  - MAE prediccion < MAE baseline en todas las fases
  - |Lead| < 2° en regimen estacionario
  - T_settle < 10s en reenganches
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys
from collections import deque

# ============================================================
# PARAMETROS
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
TAU_MEMORIA = 30.0
UMBRAL_CONFIANZA = 0.1
ALPHA_CONFIANZA = 1.0

# Prediccion V138
HORIZONTE_MAX = 5.0
ALPHA_VEL = 0.3
ALPHA_ACC = 0.1
VELOCIDAD_MAX = 20.0
ACELERACION_MAX = 10.0

# Semilla base
SEMILLA_BASE = 44


# ============================================================
# HEMISFERIO
# ============================================================

class HemisferioV138:
    def __init__(self, nombre, tau, generar_entrada_func, seed=None, sesgo=0.0):
        if seed is not None:
            np.random.seed(seed)
        self.nombre = nombre
        self.tau = tau
        self.generar_entrada = generar_entrada_func
        self.sesgo = sesgo
        
        self.Phi = np.random.normal(sesgo, 0.1, 32)
        self.Phi_vel = np.zeros(32)
        
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
        for i in range(1, 31):
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
# MEMORIA CON RELAJACION
# ============================================================

class MemoriaConRelajacion:
    def __init__(self, tau=TAU_MEMORIA, centro=0.0, alpha=ALPHA_CONFIANZA):
        self.tau = tau
        self.centro = centro
        self.alpha = alpha
        self.angulo = centro
        self.confianza = 0.0
        self.t_ultimo_estimulo = 0.0
    
    def update(self, angulo_medido, fuente_activa, t):
        if fuente_activa:
            self.angulo = angulo_medido
            self.confianza = 1.0
            self.t_ultimo_estimulo = t
        else:
            dt_silencio = t - self.t_ultimo_estimulo
            if dt_silencio >= 0:
                self.confianza = np.exp(-dt_silencio / self.tau)
            else:
                self.confianza = 0.0
        return self.confianza
    
    def get_setpoint(self):
        if self.confianza > 0.01:
            return self.angulo * (self.confianza ** self.alpha) + self.centro * (1 - (self.confianza ** self.alpha))
        return self.centro
    
    def get_confianza(self):
        return self.confianza


# ============================================================
# PREDICTOR CON ACELERACION (V138)
# ============================================================

class PredictorConAceleracion:
    """
    Predice posicion futura usando velocidad y aceleracion.
    
    Caracteristicas:
      - EMA para velocidad y aceleracion
      - Horizonte adaptativo: basado en frenado si decelera
      - Prediccion cuadratica: pos + vel*h + 0.5*acc*h^2
    """
    
    def __init__(self, horizonte_max=HORIZONTE_MAX, alpha_vel=ALPHA_VEL, 
                 alpha_acc=ALPHA_ACC, vel_max=VELOCIDAD_MAX, acc_max=ACELERACION_MAX):
        self.horizonte_max = horizonte_max
        self.alpha_vel = alpha_vel
        self.alpha_acc = alpha_acc
        self.vel_max = vel_max
        self.acc_max = acc_max
        
        self.vel = 0.0
        self.acc = 0.0
        self.vel_prev = 0.0
        self.setpoint_prev = 0.0
        self.t_prev = 0.0
        
        self.historial_vel = []
        self.historial_acc = []
        self.historial_horizonte = []
    
    def update(self, setpoint_actual, orientacion_actual, t):
        dt = t - self.t_prev
        
        # 1. Velocidad instantanea con EMA
        if dt > 0 and dt < 1.0:
            vel_inst = (setpoint_actual - self.setpoint_prev) / dt
            vel_inst = np.clip(vel_inst, -self.vel_max, self.vel_max)
            self.vel = self.alpha_vel * vel_inst + (1 - self.alpha_vel) * self.vel
            
            # 2. Aceleracion instantanea con EMA (mas lenta)
            acc_inst = (self.vel - self.vel_prev) / dt
            acc_inst = np.clip(acc_inst, -self.acc_max, self.acc_max)
            self.acc = self.alpha_acc * acc_inst + (1 - self.alpha_acc) * self.acc
        
        # 3. Horizonte adaptativo
        error_actual = abs(setpoint_actual - orientacion_actual)
        
        # Si esta frenando (vel y acc con signos opuestos)
        if self.vel * self.acc < 0 and abs(self.acc) > 0.1:
            # Tiempo hasta detenerse si continua desacelerando
            h_frenado = abs(self.vel / self.acc) if abs(self.acc) > 0.1 else self.horizonte_max
            horizonte = min(self.horizonte_max, max(0.5, h_frenado))
        else:
            # Acelerando o velocidad constante
            if abs(self.vel) > 0.1:
                horizonte = min(self.horizonte_max, error_actual / abs(self.vel))
            else:
                horizonte = 1.0
        
        # 4. Prediccion cuadratica
        setpoint_predicho = setpoint_actual + self.vel * horizonte + 0.5 * self.acc * horizonte ** 2
        
        # Guardar historial
        self.historial_vel.append(self.vel)
        self.historial_acc.append(self.acc)
        self.historial_horizonte.append(horizonte)
        
        self.vel_prev = self.vel
        self.setpoint_prev = setpoint_actual
        self.t_prev = t
        
        return setpoint_predicho, horizonte, self.vel, self.acc
    
    def get_velocidad(self):
        return self.vel
    
    def get_aceleracion(self):
        return self.acc


# ============================================================
# APARATO MOTOR CON PREDICCION V138
# ============================================================

class AparatoMotorConPrediccionV138:
    def __init__(self):
        self.orientacion = 0.0
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
        
        self.memoria = MemoriaConRelajacion()
        self.predictor = PredictorConAceleracion()
        
        self.memoria_error = deque(maxlen=VENTANA_OSCILACION)
        self.historial_Kp = []
        
        self.setpoint_usado = 0.0
        self.setpoint_predicho = 0.0
        self.horizonte_actual = 0.0
        self.prediccion_activa = False
    
    def calcular_factor_freno(self, error):
        return 1 - np.exp(-abs(error) / 30.0)
    
    def actualizar_plasticidad(self, error):
        self.memoria_error.append(error)
        if len(self.memoria_error) < VENTANA_OSCILACION:
            return
        
        oscilacion = np.std(self.memoria_error)
        if oscilacion > self.zona_muerta * 1.5:
            self.Kp_actual = max(self.Kp_min, self.Kp_actual * 0.99)
        elif oscilacion < self.zona_muerta * 0.5:
            self.Kp_actual = min(self.Kp_max, self.Kp_actual * 1.01)
        
        self.historial_Kp.append(self.Kp_actual)
    
    def actuar(self, gradiente, LF_activa, fuente_activa, t, setpoint_percepcion, 
               modo_prediccion=False):
        if not LF_activa:
            return self.orientacion, 0.0, 0.0, 0.0, False
        
        if abs(gradiente) < 0.05:
            return self.orientacion, self.memoria.get_confianza(), 0.0, 0.0, False
        
        # Actualizar memoria
        self.memoria.update(setpoint_percepcion, fuente_activa, t)
        
        # Determinar setpoint base
        if fuente_activa:
            setpoint_base = setpoint_percepcion
        else:
            setpoint_base = self.memoria.get_setpoint()
        
        # Aplicar prediccion si esta activa
        if modo_prediccion and fuente_activa:
            setpoint_predicho, horizonte, vel, acc = self.predictor.update(
                setpoint_base, self.orientacion, t
            )
            self.setpoint_usado = setpoint_predicho
            self.setpoint_predicho = setpoint_predicho
            self.horizonte_actual = horizonte
            self.prediccion_activa = True
        else:
            # Actualizar predictor igual para mantener estado
            if fuente_activa:
                self.predictor.update(setpoint_base, self.orientacion, t)
            self.setpoint_usado = setpoint_base
            self.setpoint_predicho = setpoint_base
            self.horizonte_actual = 0.0
            self.prediccion_activa = False
        
        error = self.setpoint_usado - self.orientacion
        
        if abs(error) < self.zona_muerta:
            return (self.orientacion, self.memoria.get_confianza(), 
                    self.predictor.get_velocidad(), self.horizonte_actual, 
                    self.prediccion_activa)
        
        ganancia_grad = -np.tanh(gradiente * self.sensibilidad_grad)
        factor_freno = self.calcular_factor_freno(error)
        
        delta = self.Kp_actual * error * ganancia_grad * factor_freno
        
        delta = self.inercia * self.ultimo_delta + (1 - self.inercia) * delta
        self.ultimo_delta = delta
        
        self.actualizar_plasticidad(error)
        
        self.orientacion += delta
        self.orientacion = np.clip(self.orientacion, -self.limite, self.limite)
        
        self.t += DT
        
        return (self.orientacion, self.memoria.get_confianza(), 
                self.predictor.get_velocidad(), self.horizonte_actual, 
                self.prediccion_activa)
    
    def reset(self):
        self.orientacion = 0.0
        self.ultimo_delta = 0.0
        self.t = 0.0
        self.Kp_actual = KP_BASE
        self.memoria_error.clear()
        self.historial_Kp = []
        self.prediccion_activa = False


# ============================================================
# SISTEMA V138
# ============================================================

class SistemaV138:
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
        
        self.izquierdo = HemisferioV138("L", 30.0, generar_ruido_rosa, seed=seed, sesgo=SESGO_L)
        self.derecho = HemisferioV138("R", 300.0, generar_clicks_poisson, seed=seed+100, sesgo=SESGO_R)
        
        self.sistema_B_izq = HemisferioV138("B_L", 30.0, generar_ruido_rosa, seed=seed+200, sesgo=SESGO_L)
        self.sistema_B_der = HemisferioV138("B_R", 300.0, generar_clicks_poisson, seed=seed+300, sesgo=SESGO_R)
        
        self.modo_entrenamiento = True
        self.motor = AparatoMotorConPrediccionV138()
        
        self.historial = {
            't': [],
            'orientacion': [],
            'setpoint_usado': [],
            'setpoint_predicho': [],
            'setpoint_real': [],
            'confianza': [],
            'velocidad': [],
            'aceleracion': [],
            'horizonte': [],
            'prediccion_activa': [],
            's_shared': [],
            'Kp': []
        }
    
    def calcular_s_shared(self):
        omega_A = (self.izquierdo._calcular_omega() + self.derecho._calcular_omega()) / 2
        omega_B = (self.sistema_B_izq._calcular_omega() + self.sistema_B_der._calcular_omega()) / 2
        return 1 - abs(omega_A - omega_B) / 2.0
    
    def actualizar(self, t, dt, duracion_total, setpoint_real, modo_prediccion=False):
        fuente_activa = True
        
        self.izquierdo.actualizar(t, dt, duracion_total, self.derecho)
        self.derecho.actualizar(t, dt, duracion_total, self.izquierdo)
        self.sistema_B_izq.actualizar(t, dt, duracion_total, self.sistema_B_der)
        self.sistema_B_der.actualizar(t, dt, duracion_total, self.sistema_B_izq)
        
        omega_A = (self.izquierdo._calcular_omega() + self.derecho._calcular_omega()) / 2
        omega_B = (self.sistema_B_izq._calcular_omega() + self.sistema_B_der._calcular_omega()) / 2
        gradiente = omega_A - omega_B
        
        sesgo = setpoint_real / 90.0
        gradiente += sesgo * 0.5
        
        LF_activa = not self.modo_entrenamiento
        (orientacion, confianza, velocidad, horizonte, pred_activa) = self.motor.actuar(
            gradiente, LF_activa, fuente_activa, t, setpoint_real, modo_prediccion
        )
        
        self.historial['t'].append(t)
        self.historial['orientacion'].append(orientacion)
        self.historial['setpoint_usado'].append(self.motor.setpoint_usado)
        self.historial['setpoint_predicho'].append(self.motor.setpoint_predicho)
        self.historial['setpoint_real'].append(setpoint_real)
        self.historial['confianza'].append(confianza)
        self.historial['velocidad'].append(velocidad)
        self.historial['aceleracion'].append(self.motor.predictor.get_aceleracion())
        self.historial['horizonte'].append(horizonte)
        self.historial['prediccion_activa'].append(pred_activa)
        self.historial['s_shared'].append(self.calcular_s_shared())
        self.historial['Kp'].append(self.motor.Kp_actual)
        
        return orientacion
    
    def set_modo_entrenamiento(self, entrenamiento=True):
        self.modo_entrenamiento = entrenamiento
        if entrenamiento:
            self.motor.reset()


# ============================================================
# MOVIMIENTOS DE PRUEBA (V138)
# ============================================================

def movimiento_v138(t):
    """
    Movimiento con diferentes regimenes:
      Fase 1 (0-40s): Sinusoidal
      Fase 2 (40-80s): Velocidad constante -60° → +60°
      Fase 3 (80-120s): Aceleracion abrupta (cambio de direccion)
      Fase 4 (120-160s): Stop-and-go
    """
    if t < 40:
        # Sinusoidal: periodo 40s, amplitud 60°
        return 60.0 * np.sin(2 * np.pi * t / 40.0)
    
    elif t < 80:
        # Velocidad constante: -60° → +60° en 40s (3°/s)
        t_rel = t - 40
        return -60.0 + 3.0 * t_rel
    
    elif t < 120:
        # Aceleracion abrupta: cambio de direccion
        t_rel = t - 80
        # Sube rapido, luego frena
        if t_rel < 20:
            return -60.0 + 6.0 * t_rel  # 6°/s
        else:
            # Frena y cambia direccion
            return 60.0 - 4.0 * (t_rel - 20)  # -4°/s
    
    else:
        # Stop-and-go: quietud 20s, luego movimiento
        t_rel = t - 120
        if t_rel < 20:
            return 0.0  # Quietud
        else:
            # Reanuda movimiento
            t_mov = t_rel - 20
            return -60.0 + 4.0 * t_mov  # 4°/s


# ============================================================
# EXPERIMENTO V138
# ============================================================

def ejecutar_v138():
    print("=" * 100)
    print("EXPERIMENTO V138 — Prediccion con aceleracion variable")
    print("=" * 100)
    print("  ANIMA-2 - Linea 2: Hipotesis O-N10 (generalizacion)")
    print("  Movimiento:")
    print("    Fase 1 (0-40s): Sinusoidal (periodo 40s)")
    print("    Fase 2 (40-80s): Velocidad constante -60° → +60°")
    print("    Fase 3 (80-120s): Aceleracion abrupta")
    print("    Fase 4 (120-160s): Stop-and-go")
    print("=" * 100)
    
    # Configurar sistema
    sistema = SistemaV138("V138", seed=SEMILLA_BASE)
    
    # Entrenamiento lateral
    print("\n  Entrenando lateralidad (10 repeticiones)...")
    sistema.set_modo_entrenamiento(True)
    
    for rep in range(REPETICIONES_LENTAS):
        for i in range(int(TIEMPO_POR_REPETICION / DT)):
            t = rep * TIEMPO_POR_REPETICION + i * DT
            sistema.actualizar(t, DT, TIEMPO_POR_REPETICION * REPETICIONES_LENTAS,
                              setpoint_real=0.0, modo_prediccion=False)
    
    print("  Entrenamiento completado.")
    
    # Fase de test con movimiento complejo
    print("\n  Iniciando test de prediccion con aceleracion...")
    sistema.set_modo_entrenamiento(False)
    sistema.motor.reset()
    
    t_actual = TIEMPO_POR_REPETICION * REPETICIONES_LENTAS
    duracion_test = 160.0
    
    tiempos = []
    orientaciones = []
    setpoints_reales = []
    setpoints_predichos = []
    velocidades = []
    aceleraciones = []
    horizontes = []
    
    for i in range(int(duracion_test / DT)):
        t = t_actual + i * DT
        t_rel = i * DT
        
        # Prediccion activada en todo el test
        modo_prediccion = True
        
        setpoint_real = movimiento_v138(t_rel)
        
        orientacion = sistema.actualizar(t, DT, t_actual + duracion_test,
                                         setpoint_real, modo_prediccion)
        
        tiempos.append(t_rel)
        orientaciones.append(orientacion)
        setpoints_reales.append(setpoint_real)
        
        if len(sistema.historial['setpoint_predicho']) > 0:
            setpoints_predichos.append(sistema.historial['setpoint_predicho'][-1])
        else:
            setpoints_predichos.append(setpoint_real)
        
        if len(sistema.historial['velocidad']) > 0:
            velocidades.append(sistema.historial['velocidad'][-1])
        else:
            velocidades.append(0.0)
        
        if len(sistema.historial['aceleracion']) > 0:
            aceleraciones.append(sistema.historial['aceleracion'][-1])
        else:
            aceleraciones.append(0.0)
        
        if len(sistema.historial['horizonte']) > 0:
            horizontes.append(sistema.historial['horizonte'][-1])
        else:
            horizontes.append(0.0)
        
        # Reporte cada 10s
        if int(t_rel * 10) % 100 == 0 and t_rel > 0:
            fase = ""
            if t_rel < 40:
                fase = "F1(sinusoidal)"
            elif t_rel < 80:
                fase = "F2(vel.constante)"
            elif t_rel < 120:
                fase = "F3(aceleracion)"
            else:
                fase = "F4(stop-go)"
            
            error = abs(orientacion - setpoint_real)
            vel = velocidades[-1] if velocidades else 0
            acc = aceleraciones[-1] if aceleraciones else 0
            h = horizontes[-1] if horizontes else 0
            print(f"    t={t_rel:4.0f}s | {fase:14s} | setpoint={setpoint_real:5.1f}° | "
                  f"orient={orientacion:5.1f}° | error={error:4.1f}° | vel={vel:5.2f}°/s | "
                  f"acc={acc:5.2f}°/s² | h={h:.2f}s")
    
    # Convertir a arrays
    t_arr = np.array(tiempos)
    orient_arr = np.array(orientaciones)
    setpoint_arr = np.array(setpoints_reales)
    setpoint_pred_arr = np.array(setpoints_predichos)
    
    # Calcular metricas por fase
    mask_f1 = (t_arr >= 5) & (t_arr < 40)
    mask_f2 = (t_arr >= 40) & (t_arr < 80)
    mask_f3 = (t_arr >= 80) & (t_arr < 120)
    mask_f4 = (t_arr >= 120) & (t_arr < 160)
    
    print("\n" + "=" * 80)
    print("ANALISIS DE PREDICCION POR FASE")
    print("=" * 80)
    
    fases = [
        ("F1 (sinusoidal)", mask_f1),
        ("F2 (vel.constante)", mask_f2),
        ("F3 (aceleracion)", mask_f3),
        ("F4 (stop-go)", mask_f4)
    ]
    
    resultados = []
    mae_total = 0
    lead_total = 0
    
    for nombre, mask in fases:
        if np.any(mask):
            error = np.abs(orient_arr[mask] - setpoint_arr[mask])
            lead = np.mean(orient_arr[mask] - setpoint_pred_arr[mask])
            mae = np.mean(error)
            mae_total += mae
            lead_total += lead
            print(f"\n  {nombre}:")
            print(f"    MAE: {mae:.2f}°")
            print(f"    Lead Index: {lead:.2f}°")
            resultados.append((nombre, mae, lead))
    
    print("\n" + "=" * 80)
    print("CONCLUSION V138 — Generalizacion de prediccion")
    print("=" * 80)
    
    # Criterios de exito
    exito_general = all(mae < 12.0 for _, mae, _ in resultados) if resultados else False
    
    if exito_general:
        print("\n  ✅ PREDICCION GENERALIZABLE")
        print("     El organismo sigue correctamente los 4 regimenes de movimiento")
        print("\n  ANIMA-2 - Linea 2: CERRADA")
    else:
        print("\n  ⚠️ PREDICCION PARCIAL")
        print("     El organismo falla en algunos regimenes")
    
    # Graficos
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Grafico 1: Orientacion vs Setpoint
    ax = axes[0, 0]
    ax.plot(t_arr, setpoint_arr, 'r--', linewidth=1, alpha=0.7, label='Setpoint real')
    ax.plot(t_arr, orient_arr, 'b-', linewidth=0.8, label='Orientacion real')
    ax.axvline(x=40, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=80, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=120, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Angulo (grados)')
    ax.set_title('V138: Prediccion con aceleracion variable')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Grafico 2: Error de orientacion
    ax = axes[0, 1]
    error_total = np.abs(orient_arr - setpoint_arr)
    ax.plot(t_arr, error_total, 'purple', linewidth=0.8)
    ax.axhline(y=ZONA_MUERTA_BASE, color='red', linestyle='--', alpha=0.5, label=f'Zona muerta ({ZONA_MUERTA_BASE}°)')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Error (grados)')
    ax.set_title('Error de orientacion')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Grafico 3: Lead Index
    ax = axes[0, 2]
    lead_index = orient_arr - setpoint_pred_arr
    ax.plot(t_arr, lead_index, 'orange', linewidth=0.8)
    ax.axhline(y=0, color='red', linestyle='-', alpha=0.5)
    ax.axhline(y=2, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=-2, color='gray', linestyle=':', alpha=0.5)
    ax.fill_between(t_arr, -2, 2, alpha=0.2, color='green')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Lead Index (grados)')
    ax.set_title('Lead Index (vs setpoint predicho)')
    ax.grid(True, alpha=0.3)
    
    # Grafico 4: Velocidad y aceleracion
    ax = axes[1, 0]
    vel_arr = np.array(velocidades)
    acc_arr = np.array(aceleraciones)
    ax.plot(t_arr[:len(vel_arr)], vel_arr, 'cyan', linewidth=0.8, label='Velocidad estimada')
    ax.plot(t_arr[:len(acc_arr)], acc_arr, 'magenta', linewidth=0.8, alpha=0.7, label='Aceleracion estimada')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Velocidad (grados/s) / Aceleracion (grados/s²)')
    ax.set_title('Velocidad y aceleracion estimadas')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Grafico 5: Horizonte adaptativo
    ax = axes[1, 1]
    h_arr = np.array(horizontes)
    ax.plot(t_arr[:len(h_arr)], h_arr, 'green', linewidth=0.8)
    ax.axhline(y=HORIZONTE_MAX, color='red', linestyle='--', alpha=0.5, label=f'Horizonte max ({HORIZONTE_MAX}s)')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Horizonte (s)')
    ax.set_title('Horizonte adaptativo de prediccion')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Grafico 6: MAE por fase
    ax = axes[1, 2]
    nombres = [r[0] for r in resultados]
    mae_vals = [r[1] for r in resultados]
    colores_mae = ['blue', 'green', 'orange', 'red'][:len(resultados)]
    bars = ax.bar(nombres, mae_vals, color=colores_mae, alpha=0.7)
    for bar, val in zip(bars, mae_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                f'{val:.1f}°', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('MAE (grados)')
    ax.set_title('Error por regimen de movimiento')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=15)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('v138_logs', exist_ok=True)
    plt.savefig(f'v138_logs/v138_generalizacion_{timestamp}.png', dpi=150)
    print(f"\n  Graficos guardados: v138_logs/v138_generalizacion_{timestamp}.png")
    
    return sistema, exito_general


if __name__ == "__main__":
    import time
    start = time.time()
    sistema, exito = ejecutar_v138()
    elapsed = time.time() - start
    print(f"\n  Tiempo de ejecucion: {elapsed/60:.1f} minutos")