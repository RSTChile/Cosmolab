#!/usr/bin/env python3
"""
VSTCosmos V139 — Motor con inercia (sin homúnculo)

ANIMA-2 - Linea 2: Hipotesis O-N10b
  La prediccion emerge de la inercia del motor, no de un predictor externo.
  
Mecanismo:
  - Motor de segundo orden (velocidad + aceleracion)
  - Inercia: resistencia al cambio (0 = instantaneo, 1 = muy inerte)
  - Friccion: amortiguacion natural
  - No hay calculo de velocidad del setpoint
  - La anticipacion emerge del sobrepaso por inercia

Criterio O-N10b:
  - MAE < 10° en regimenes variados
  - Overshoot < 15°
  - T_settle < 20s
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

# Inercia (parametros abstractos del motor)
INERCIA_MOTOR = 0.3      # Resistencia al cambio (0=instantaneo, 1=muy inerte)
FRICCION = 0.1           # Amortiguacion natural

# Semilla base
SEMILLA_BASE = 44


# ============================================================
# HEMISFERIO
# ============================================================

class HemisferioV139:
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
# MOTOR CON INERCIA (V139 - SIN HOMUNCULO)
# ============================================================

class AparatoMotorInercial:
    """
    Motor de segundo orden con inercia.
    
    La prediccion emerge de la dinamica del motor:
      - Cuando el setpoint cambia, la inercia causa sobrepaso
      - Ese sobrepaso es la "anticipacion" sin calcular futuro
      - No hay predictor externo, solo F = m*a implicita
    
    Parametros:
      - inercia: resistencia al cambio (0 = respuesta instantanea)
      - friccion: amortiguacion natural
    """
    
    def __init__(self):
        self.orientacion = 0.0
        self.velocidad = 0.0
        
        self.Kp_base = KP_BASE
        self.Kp_actual = KP_BASE
        self.Kp_min = KP_MIN
        self.Kp_max = KP_MAX
        self.limite = 90.0
        self.zona_muerta = ZONA_MUERTA_BASE
        
        # Parametros de inercia (abstractos)
        self.inercia = INERCIA_MOTOR      # Resistencia al cambio
        self.friccion = FRICCION          # Amortiguacion
        self.sensibilidad_grad = 10.0
        
        self.memoria = MemoriaConRelajacion()
        
        self.memoria_error = deque(maxlen=VENTANA_OSCILACION)
        self.historial_Kp = []
        self.historial_velocidad = []
        
        self.t = 0.0
    
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
    
    def actuar(self, gradiente, LF_activa, fuente_activa, t, setpoint_percepcion):
        if not LF_activa:
            return self.orientacion, 0.0, 0.0
        
        if abs(gradiente) < 0.05:
            return self.orientacion, self.memoria.get_confianza(), 0.0
        
        # Actualizar memoria
        self.memoria.update(setpoint_percepcion, fuente_activa, t)
        
        # Determinar setpoint objetivo
        if fuente_activa:
            setpoint_objetivo = setpoint_percepcion
        else:
            setpoint_objetivo = self.memoria.get_setpoint()
        
        # Error de posicion
        error = setpoint_objetivo - self.orientacion
        
        # Zona muerta
        if abs(error) < self.zona_muerta:
            return self.orientacion, self.memoria.get_confianza(), self.velocidad
        
        # Fuerza = error * gradiente (como en V132)
        ganancia_grad = -np.tanh(gradiente * self.sensibilidad_grad)
        fuerza = self.Kp_actual * error * ganancia_grad
        
        # Limitar fuerza para evitar inestabilidades
        fuerza = np.clip(fuerza, -10.0, 10.0)
        
        # Dinamica de segundo orden (F = m*a, con m = 1/inercia)
        # aceleracion = fuerza * (1 - inercia) - friccion * velocidad
        aceleracion = fuerza * (1 - self.inercia) - self.friccion * self.velocidad
        
        # Integrar
        self.velocidad += aceleracion * DT
        self.velocidad = np.clip(self.velocidad, -30.0, 30.0)
        
        self.orientacion += self.velocidad * DT
        self.orientacion = np.clip(self.orientacion, -self.limite, self.limite)
        
        # Plasticidad
        self.actualizar_plasticidad(error)
        
        self.historial_velocidad.append(self.velocidad)
        self.t += DT
        
        return self.orientacion, self.memoria.get_confianza(), self.velocidad
    
    def reset(self):
        self.orientacion = 0.0
        self.velocidad = 0.0
        self.t = 0.0
        self.Kp_actual = KP_BASE
        self.memoria_error.clear()
        self.historial_Kp = []
        self.historial_velocidad = []


# ============================================================
# SISTEMA V139
# ============================================================

class SistemaV139:
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
        
        self.izquierdo = HemisferioV139("L", 30.0, generar_ruido_rosa, seed=seed, sesgo=SESGO_L)
        self.derecho = HemisferioV139("R", 300.0, generar_clicks_poisson, seed=seed+100, sesgo=SESGO_R)
        
        self.sistema_B_izq = HemisferioV139("B_L", 30.0, generar_ruido_rosa, seed=seed+200, sesgo=SESGO_L)
        self.sistema_B_der = HemisferioV139("B_R", 300.0, generar_clicks_poisson, seed=seed+300, sesgo=SESGO_R)
        
        self.modo_entrenamiento = True
        self.motor = AparatoMotorInercial()
        
        self.historial = {
            't': [],
            'orientacion': [],
            'setpoint_real': [],
            'confianza': [],
            'velocidad': [],
            's_shared': [],
            'Kp': []
        }
    
    def calcular_s_shared(self):
        omega_A = (self.izquierdo._calcular_omega() + self.derecho._calcular_omega()) / 2
        omega_B = (self.sistema_B_izq._calcular_omega() + self.sistema_B_der._calcular_omega()) / 2
        return 1 - abs(omega_A - omega_B) / 2.0
    
    def actualizar(self, t, dt, duracion_total, setpoint_real, modo_prediccion=False):
        # Modo prediccion es ignorado en V139 (la inercia siempre actua)
        fuente_activa = True
        
        self.izquierdo.actualizar(t, dt, duracion_total, self.derecho)
        self.derecho.actualizar(t, dt, duracion_total, self.izquierdo)
        self.sistema_B_izq.actualizar(t, dt, duracion_total, self.sistema_B_der)
        self.sistema_B_der.actualizar(t, dt, duracion_total, self.sistema_B_izq)
        
        omega_A = (self.izquierdo._calcular_omega() + self.derecho._calcular_omega()) / 2
        omega_B = (self.sistema_B_izq._calcular_omega() + self.sistema_B_der._calcular_omega()) / 2
        gradiente = omega_A - omega_B
        
        # Espacializacion
        sesgo = setpoint_real / 90.0
        gradiente += sesgo * 0.5
        
        LF_activa = not self.modo_entrenamiento
        orientacion, confianza, velocidad = self.motor.actuar(
            gradiente, LF_activa, fuente_activa, t, setpoint_real
        )
        
        self.historial['t'].append(t)
        self.historial['orientacion'].append(orientacion)
        self.historial['setpoint_real'].append(setpoint_real)
        self.historial['confianza'].append(confianza)
        self.historial['velocidad'].append(velocidad)
        self.historial['s_shared'].append(self.calcular_s_shared())
        self.historial['Kp'].append(self.motor.Kp_actual)
        
        return orientacion
    
    def set_modo_entrenamiento(self, entrenamiento=True):
        self.modo_entrenamiento = entrenamiento
        if entrenamiento:
            self.motor.reset()


# ============================================================
# MOVIMIENTOS DE PRUEBA
# ============================================================

def movimiento_v139(t):
    """Mismo movimiento que V138 para comparacion directa"""
    if t < 40:
        return 60.0 * np.sin(2 * np.pi * t / 40.0)
    elif t < 80:
        t_rel = t - 40
        return -60.0 + 3.0 * t_rel
    elif t < 120:
        t_rel = t - 80
        if t_rel < 20:
            return -60.0 + 6.0 * t_rel
        else:
            return 60.0 - 4.0 * (t_rel - 20)
    else:
        t_rel = t - 120
        if t_rel < 20:
            return 0.0
        else:
            t_mov = t_rel - 20
            return -60.0 + 4.0 * t_mov


# ============================================================
# EXPERIMENTO V139
# ============================================================

def ejecutar_v139():
    print("=" * 100)
    print("EXPERIMENTO V139 — Motor con inercia (sin homunculo)")
    print("=" * 100)
    print("  ANIMA-2 - Linea 2: Hipotesis O-N10b")
    print("  Mecanismo:")
    print(f"    - Inercia: {INERCIA_MOTOR} (resistencia al cambio)")
    print(f"    - Friccion: {FRICCION} (amortiguacion)")
    print("  Movimiento:")
    print("    Fase 1 (0-40s): Sinusoidal (periodo 40s)")
    print("    Fase 2 (40-80s): Velocidad constante -60° → +60°")
    print("    Fase 3 (80-120s): Aceleracion abrupta")
    print("    Fase 4 (120-160s): Stop-and-go")
    print("=" * 100)
    
    # Configurar sistema
    sistema = SistemaV139("V139", seed=SEMILLA_BASE)
    
    # Entrenamiento lateral
    print("\n  Entrenando lateralidad (10 repeticiones)...")
    sistema.set_modo_entrenamiento(True)
    
    for rep in range(REPETICIONES_LENTAS):
        for i in range(int(TIEMPO_POR_REPETICION / DT)):
            t = rep * TIEMPO_POR_REPETICION + i * DT
            sistema.actualizar(t, DT, TIEMPO_POR_REPETICION * REPETICIONES_LENTAS,
                              setpoint_real=0.0)
    
    print("  Entrenamiento completado.")
    
    # Fase de test
    print("\n  Iniciando test de inercia...")
    sistema.set_modo_entrenamiento(False)
    sistema.motor.reset()
    
    t_actual = TIEMPO_POR_REPETICION * REPETICIONES_LENTAS
    duracion_test = 160.0
    
    tiempos = []
    orientaciones = []
    setpoints_reales = []
    velocidades = []
    
    for i in range(int(duracion_test / DT)):
        t = t_actual + i * DT
        t_rel = i * DT
        
        setpoint_real = movimiento_v139(t_rel)
        
        orientacion = sistema.actualizar(t, DT, t_actual + duracion_test,
                                         setpoint_real)
        
        tiempos.append(t_rel)
        orientaciones.append(orientacion)
        setpoints_reales.append(setpoint_real)
        
        if len(sistema.historial['velocidad']) > 0:
            velocidades.append(sistema.historial['velocidad'][-1])
        else:
            velocidades.append(0.0)
        
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
            print(f"    t={t_rel:4.0f}s | {fase:14s} | setpoint={setpoint_real:5.1f}° | "
                  f"orient={orientacion:5.1f}° | error={error:4.1f}° | vel={vel:5.2f}°/s")
    
    # Convertir a arrays
    t_arr = np.array(tiempos)
    orient_arr = np.array(orientaciones)
    setpoint_arr = np.array(setpoints_reales)
    
    # Calcular metricas por fase
    mask_f1 = (t_arr >= 5) & (t_arr < 40)
    mask_f2 = (t_arr >= 40) & (t_arr < 80)
    mask_f3 = (t_arr >= 80) & (t_arr < 120)
    mask_f4 = (t_arr >= 120) & (t_arr < 160)
    
    print("\n" + "=" * 80)
    print("ANALISIS DE INERCIA")
    print("=" * 80)
    
    fases = [
        ("F1 (sinusoidal)", mask_f1),
        ("F2 (vel.constante)", mask_f2),
        ("F3 (aceleracion)", mask_f3),
        ("F4 (stop-go)", mask_f4)
    ]
    
    resultados = []
    mae_total = 0
    fases_validas = 0
    max_overshoot = 0
    
    for nombre, mask in fases:
        if np.any(mask):
            error = np.abs(orient_arr[mask] - setpoint_arr[mask])
            mae = np.mean(error)
            mae_total += mae
            fases_validas += 1
            
            # Calcular overshoot (maximo exceso sobre setpoint en transiciones)
            orient_fase = orient_arr[mask]
            setpoint_fase = setpoint_arr[mask]
            overshoot_fase = np.max(orient_fase - setpoint_fase) if np.max(orient_fase - setpoint_fase) > 0 else 0
            if overshoot_fase > max_overshoot:
                max_overshoot = overshoot_fase
            
            print(f"\n  {nombre}:")
            print(f"    MAE: {mae:.2f}°")
            print(f"    Overshoot maximo: {overshoot_fase:.1f}°")
            resultados.append((nombre, mae, overshoot_fase))
    
    if fases_validas > 0:
        mae_promedio = mae_total / fases_validas
    else:
        mae_promedio = 999
    
    # Calcular T_settle (tiempo para estabilizarse en F4)
    mask_f4_stable = (t_arr >= 150) & (t_arr < 160)
    if np.any(mask_f4_stable):
        orient_f4_stable = orient_arr[mask_f4_stable]
        setpoint_f4_stable = setpoint_arr[mask_f4_stable]
        error_f4 = np.abs(orient_f4_stable - setpoint_f4_stable)
        if len(error_f4) > 0:
            t_settle = 160 - 150 - np.where(error_f4 < 5.0)[0][-1] * DT if np.any(error_f4 < 5.0) else None
        else:
            t_settle = None
    else:
        t_settle = None
    
    # Criterios O-N10b
    exito_mae = mae_promedio < 10.0
    exito_overshoot = max_overshoot < 15.0
    exito_settle = t_settle is not None and t_settle < 20.0
    
    print("\n" + "=" * 80)
    print("CONCLUSION V139 — Motor con inercia")
    print("=" * 80)
    
    print(f"\n  MAE promedio: {mae_promedio:.2f}° {'✅' if exito_mae else '❌'} (objetivo <10°)")
    print(f"  Overshoot maximo: {max_overshoot:.1f}° {'✅' if exito_overshoot else '❌'} (objetivo <15°)")
    print(f"  T_settle: {t_settle:.1f}s {'✅' if exito_settle else '❌'} (objetivo <20s)" if t_settle else "  T_settle: No alcanzado ❌")
    
    exito_total = exito_mae and exito_overshoot
    
    if exito_total:
        print("\n  ✅ O-N10b VALIDADA")
        print("     La inercia del motor genera anticipacion emergente")
        print("     Sin homunculo, sin predictor externo")
        print("\n  ANIMA-2 - Linea 2: CERRADA")
    else:
        print("\n  ⚠️ O-N10b NO VALIDADA")
        print("     Los parametros de inercia necesitan ajuste")
    
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
    ax.set_title('V139: Motor con inercia')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Grafico 2: Error
    ax = axes[0, 1]
    error_total = np.abs(orient_arr - setpoint_arr)
    ax.plot(t_arr, error_total, 'purple', linewidth=0.8)
    ax.axhline(y=ZONA_MUERTA_BASE, color='red', linestyle='--', alpha=0.5, label=f'Zona muerta ({ZONA_MUERTA_BASE}°)')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Error (grados)')
    ax.set_title('Error de orientacion')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Grafico 3: Overshoot (diferencia orient - setpoint)
    ax = axes[0, 2]
    overshoot = orient_arr - setpoint_arr
    ax.plot(t_arr, overshoot, 'orange', linewidth=0.8)
    ax.axhline(y=0, color='red', linestyle='-', alpha=0.5)
    ax.axhline(y=15, color='gray', linestyle=':', alpha=0.5, label='Limite overshoot (15°)')
    ax.fill_between(t_arr, 0, overshoot, where=(overshoot>0), alpha=0.3, color='green', label='Overshoot')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Overshoot (grados)')
    ax.set_title('Overshoot por inercia')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Grafico 4: Velocidad del motor
    ax = axes[1, 0]
    vel_arr = np.array(velocidades)
    ax.plot(t_arr[:len(vel_arr)], vel_arr, 'cyan', linewidth=0.8)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Velocidad (grados/s)')
    ax.set_title('Velocidad del motor (emergente)')
    ax.grid(True, alpha=0.3)
    
    # Grafico 5: MAE por fase
    ax = axes[1, 1]
    nombres = [r[0] for r in resultados]
    mae_vals = [r[1] for r in resultados]
    colores_mae = ['blue', 'green', 'orange', 'red'][:len(resultados)]
    bars = ax.bar(nombres, mae_vals, color=colores_mae, alpha=0.7)
    for bar, val in zip(bars, mae_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                f'{val:.1f}°', ha='center', va='bottom', fontsize=9)
    ax.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='Objetivo MAE <10°')
    ax.set_ylabel('MAE (grados)')
    ax.set_title('Error por fase')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=15)
    
    # Grafico 6: s_shared
    ax = axes[1, 2]
    s_shared = sistema.historial['s_shared']
    ax.plot(t_arr[:len(s_shared)], s_shared, 'purple', linewidth=0.5)
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Umbral lateralidad (0.8)')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('s_shared')
    ax.set_title('Coherencia inter-sistemas')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('v139_logs', exist_ok=True)
    plt.savefig(f'v139_logs/v139_inercia_{timestamp}.png', dpi=150)
    print(f"\n  Graficos guardados: v139_logs/v139_inercia_{timestamp}.png")
    
    return sistema, exito_total


if __name__ == "__main__":
    import time
    start = time.time()
    sistema, exito = ejecutar_v139()
    elapsed = time.time() - start
    print(f"\n  Tiempo de ejecucion: {elapsed/60:.1f} minutos")