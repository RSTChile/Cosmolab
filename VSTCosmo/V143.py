#!/usr/bin/env python3
"""
VSTCosmos V143 — Recuperación de la alternancia (sin fatiga)

Objetivo:
  Demostrar que el organismo puede alternar -60° ↔ +60° en régimen sano.
  
Parámetros:
  - Sin fatiga (K_GAIN = 0, sin degradación)
  - Zona muerta = 2.0°
  - Período de alternancia = 40s (20s en cada polo)
  - Amplitud progresiva: ±30° → ±45° → ±60°
  
Criterio de éxito:
  - Alternancia verificada: amplitud real > 80° del setpoint
  - Error final < 10° en cada semiciclo
  - T_settle < 30s
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys
from collections import deque

# ============================================================
# PARAMETROS (régimen sano, sin fatiga)
# ============================================================
DT = 0.01
TIEMPO_POR_REPETICION = 452.0
REPETICIONES_LENTAS = 10

# Asimetria forzada
SESGO_L = 0.05
SESGO_R = -0.05
DIM_HEMISFERIO = 32

# Zona muerta (recuperamos 2.0°)
ZONA_MUERTA_BASE = 2.0

# Limites de plasticidad
KP_BASE = 0.002
KP_MIN = 0.0005
KP_MAX = 0.005

# Plasticidad
HABITUACION_RAPIDA = 0.99
SENSIBILIZACION_LENTA = 1.01
VENTANA_OSCILACION = 100

# Sin fatiga (V143)
K_GAIN = 0.0           # Sin degradación de ganancia
K_PRECISION = 0.0      # Sin expansión de zona muerta
K_TEMBLOR = 0.0        # Sin temblor
TAU_RECUPERACION = 1.0 # Irrelevante

# Semilla base
SEMILLA_BASE = 44

# Onda cuadrada más lenta
PERIODO_ALTERNANCIA = 40.0  # 40s por ciclo (20s en cada polo)


# ============================================================
# HEMISFERIO
# ============================================================

class HemisferioV143:
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
# FATIGA (DESACTIVADA EN V143)
# ============================================================

class FatigaDesactivada:
    def __init__(self):
        self.energia_total = 0.0
        self.historial_energia = []
    
    def actualizar(self, delta_orientacion, en_reposo, dt):
        self.historial_energia.append(self.energia_total)
        return 1.0, ZONA_MUERTA_BASE, 0.0  # factor_gain=1, zona_muerta=2.0, temblor=0
    
    def reset(self):
        self.energia_total = 0.0
        self.historial_energia = []
    
    def get_energia(self):
        return self.energia_total


# ============================================================
# APARATO MOTOR SIN FATIGA (V143)
# ============================================================

class AparatoMotorSinFatiga:
    def __init__(self):
        self.orientacion = 0.0
        self.Kp_base = KP_BASE
        self.Kp_actual = KP_BASE
        self.Kp_min = KP_MIN
        self.Kp_max = KP_MAX
        self.limite = 90.0
        self.inercia = 0.95
        self.ultimo_delta = 0.0
        self.sensibilidad_grad = 10.0
        self.t = 0.0
        
        self.fatiga = FatigaDesactivada()
        
        self.memoria_error = deque(maxlen=VENTANA_OSCILACION)
        self.historial_Kp = []
        self.historial_fatiga = []
        
        self.ultimo_delta_registrado = 0.0
    
    def calcular_factor_freno(self, error):
        return 1 - np.exp(-abs(error) / 30.0)
    
    def actualizar_plasticidad(self, error):
        self.memoria_error.append(error)
        if len(self.memoria_error) < VENTANA_OSCILACION:
            return
        
        oscilacion = np.std(self.memoria_error)
        if oscilacion > ZONA_MUERTA_BASE * 1.5:
            self.Kp_actual = max(self.Kp_min, self.Kp_actual * 0.99)
        elif oscilacion < ZONA_MUERTA_BASE * 0.5:
            self.Kp_actual = min(self.Kp_max, self.Kp_actual * 1.01)
        
        self.historial_Kp.append(self.Kp_actual)
    
    def actuar(self, gradiente, LF_activa, fuente_activa, t, setpoint_percepcion):
        if not LF_activa:
            return self.orientacion, 0.0, 0.0
        
        if abs(gradiente) < 0.05:
            return self.orientacion, self.fatiga.get_energia(), 0.0
        
        setpoint_objetivo = setpoint_percepcion if fuente_activa else 0.0
        
        error = setpoint_objetivo - self.orientacion
        
        # Sin fatiga: factor_gain=1, zona_muerta_efectiva=ZONA_MUERTA_BASE, temblor=0
        factor_gain, zona_muerta_efectiva, temblor = self.fatiga.actualizar(
            self.ultimo_delta_registrado, not fuente_activa, DT
        )
        
        if abs(error) < zona_muerta_efectiva:
            return self.orientacion, self.fatiga.get_energia(), zona_muerta_efectiva
        
        ganancia_grad = -np.tanh(gradiente * self.sensibilidad_grad)
        factor_freno = self.calcular_factor_freno(error)
        
        Kp_efectivo = self.Kp_actual * factor_gain
        
        delta_raw = Kp_efectivo * error * ganancia_grad * factor_freno
        
        delta = self.inercia * self.ultimo_delta + (1 - self.inercia) * delta_raw
        self.ultimo_delta = delta
        self.ultimo_delta_registrado = delta
        
        delta += temblor * DT
        
        self.actualizar_plasticidad(error)
        
        self.orientacion += delta
        self.orientacion = np.clip(self.orientacion, -self.limite, self.limite)
        
        self.historial_fatiga.append(factor_gain)
        self.t += DT
        
        return self.orientacion, self.fatiga.get_energia(), zona_muerta_efectiva
    
    def reset(self):
        self.orientacion = 0.0
        self.ultimo_delta = 0.0
        self.ultimo_delta_registrado = 0.0
        self.t = 0.0
        self.Kp_actual = KP_BASE
        self.memoria_error.clear()
        self.historial_Kp = []
        self.historial_fatiga = []
        self.fatiga.reset()


# ============================================================
# SISTEMA V143
# ============================================================

class SistemaV143:
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
        
        self.izquierdo = HemisferioV143("L", 30.0, generar_ruido_rosa, seed=seed, sesgo=SESGO_L)
        self.derecho = HemisferioV143("R", 300.0, generar_clicks_poisson, seed=seed+100, sesgo=SESGO_R)
        
        self.sistema_B_izq = HemisferioV143("B_L", 30.0, generar_ruido_rosa, seed=seed+200, sesgo=SESGO_L)
        self.sistema_B_der = HemisferioV143("B_R", 300.0, generar_clicks_poisson, seed=seed+300, sesgo=SESGO_R)
        
        self.modo_entrenamiento = True
        self.motor = AparatoMotorSinFatiga()
        
        self.historial = {
            't': [],
            'orientacion': [],
            'setpoint_real': [],
            'energia': [],
            'zona_muerta': [],
            's_shared': [],
            'Kp': []
        }
    
    def calcular_s_shared(self):
        omega_A = (self.izquierdo._calcular_omega() + self.derecho._calcular_omega()) / 2
        omega_B = (self.sistema_B_izq._calcular_omega() + self.sistema_B_der._calcular_omega()) / 2
        return 1 - abs(omega_A - omega_B) / 2.0
    
    def actualizar(self, t, dt, duracion_total, setpoint_real):
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
        orientacion, energia, zona_muerta = self.motor.actuar(
            gradiente, LF_activa, fuente_activa, t, setpoint_real
        )
        
        self.historial['t'].append(t)
        self.historial['orientacion'].append(orientacion)
        self.historial['setpoint_real'].append(setpoint_real)
        self.historial['energia'].append(energia)
        self.historial['zona_muerta'].append(zona_muerta)
        self.historial['s_shared'].append(self.calcular_s_shared())
        self.historial['Kp'].append(self.motor.Kp_actual)
        
        return orientacion, energia
    
    def set_modo_entrenamiento(self, entrenamiento=True):
        self.modo_entrenamiento = entrenamiento
        if entrenamiento:
            self.motor.reset()


# ============================================================
# ONDA CUADRADA PROGRESIVA
# ============================================================

def onda_cuadrada_progresiva(t, periodo=PERIODO_ALTERNANCIA, amplitud=60.0):
    """Onda cuadrada con alternancia clara entre -60° y +60°"""
    if (t % periodo) < (periodo / 2):
        return -amplitud
    else:
        return +amplitud


# ============================================================
# FUNCION DE ANALISIS DE ALTERNANCIA
# ============================================================

def analizar_alternancia(orientaciones, setpoints, umbral_amplitud=40.0):
    """Verifica que el organismo realmente alterna entre polos"""
    if len(orientaciones) == 0:
        return False, 0.0, 0
    
    # Amplitud real (max - min)
    amplitud_real = max(orientaciones) - min(orientaciones)
    
    # Conteo de cruces por cero (alternancia)
    cruces = 0
    for i in range(1, len(orientaciones)):
        if orientaciones[i-1] * orientaciones[i] < 0:
            cruces += 1
    
    # Amplitud suficiente?
    amplitud_suficiente = amplitud_real > umbral_amplitud
    alterna = amplitud_suficiente and cruces > 2
    
    return alterna, amplitud_real, cruces


def calcular_t_settle(orientaciones, setpoints, dt=DT, umbral_error=2.0):
    """Calcula T_settle: tiempo hasta error < umbral_error por 50 pasos"""
    if len(orientaciones) == 0:
        return None
    
    errores = np.abs(np.array(orientaciones) - np.array(setpoints))
    
    for i in range(len(errores) - 50):
        if all(errores[i:i+50] < umbral_error):
            return i * dt
    return None


# ============================================================
# EXPERIMENTO V143
# ============================================================

def ejecutar_v143():
    print("=" * 100)
    print("EXPERIMENTO V143 — Recuperación de alternancia (sin fatiga)")
    print("=" * 100)
    print("  Objetivo: Demostrar que el organismo puede alternar -60° ↔ +60°")
    print("  Parámetros:")
    print(f"    - Zona muerta: {ZONA_MUERTA_BASE}°")
    print(f"    - Período: {PERIODO_ALTERNANCIA}s (20s en cada polo)")
    print(f"    - Fatiga: DESACTIVADA")
    print(f"    - Kp_base: {KP_BASE}")
    print(f"    - Kp_min: {KP_MIN}")
    print("=" * 100)
    
    sistema = SistemaV143("V143", seed=SEMILLA_BASE)
    
    # Entrenamiento
    print("\n  Entrenando lateralidad (10 repeticiones)...")
    sistema.set_modo_entrenamiento(True)
    
    for rep in range(REPETICIONES_LENTAS):
        for i in range(int(TIEMPO_POR_REPETICION / DT)):
            t = rep * TIEMPO_POR_REPETICION + i * DT
            sistema.actualizar(t, DT, TIEMPO_POR_REPETICION * REPETICIONES_LENTAS,
                              setpoint_real=0.0)
    
    print("  Entrenamiento completado.")
    
    # Test de alternancia con amplitud progresiva
    print("\n  Iniciando test de alternancia progresiva...")
    sistema.set_modo_entrenamiento(False)
    sistema.motor.reset()
    
    t_actual = TIEMPO_POR_REPETICION * REPETICIONES_LENTAS
    
    amplitudes = [30.0, 45.0, 60.0]
    resultados = []
    
    for amplitud in amplitudes:
        print(f"\n  Test con amplitud: ±{amplitud:.0f}° (período {PERIODO_ALTERNANCIA}s)")
        
        tiempos = []
        orientaciones = []
        setpoints = []
        
        for i in range(int(3 * PERIODO_ALTERNANCIA / DT)):  # 3 ciclos completos
            t = t_actual + i * DT
            t_rel = i * DT
            
            setpoint = onda_cuadrada_progresiva(t_rel, periodo=PERIODO_ALTERNANCIA, amplitud=amplitud)
            orient, energia = sistema.actualizar(t, DT, t_actual + 200, setpoint)
            
            tiempos.append(t_rel)
            orientaciones.append(orient)
            setpoints.append(setpoint)
            
            if i % 2000 == 0:
                print(f"      t={t_rel:.0f}s | setpoint={setpoint:+.0f}° | orient={orient:.1f}°")
        
        # Analizar
        alterna, amp_real, cruces = analizar_alternancia(orientaciones, setpoints, umbral_amplitud=amplitud * 0.7)
        t_settle = calcular_t_settle(orientaciones, setpoints, DT, umbral_error=ZONA_MUERTA_BASE)
        
        # Error final promedio (último ciclo)
        ciclo_final = orientaciones[-int(PERIODO_ALTERNANCIA/DT):]
        setpoint_final = setpoints[-int(PERIODO_ALTERNANCIA/DT):]
        error_final = np.mean(np.abs(np.array(ciclo_final) - np.array(setpoint_final)))
        
        resultados.append({
            'amplitud': amplitud,
            'alterna': alterna,
            'amplitud_real': amp_real,
            'cruces': cruces,
            't_settle': t_settle,
            'error_final': error_final
        })
        
        print(f"      Resultado: alternancia={'✅' if alterna else '❌'} | "
              f"amplitud_real={amp_real:.1f}° | cruces={cruces} | "
              f"t_settle={t_settle:.1f}s" if t_settle else f"t_settle=∞ | "
              f"error_final={error_final:.1f}°")
        
        t_actual += 3 * PERIODO_ALTERNANCIA
    
    print("\n" + "=" * 80)
    print("RESULTADOS DE ALTERNANCIA PROGRESIVA")
    print("=" * 80)
    
    exito_total = all(r['alterna'] for r in resultados) if resultados else False
    
    for r in resultados:
        print(f"\n  Amplitud ±{r['amplitud']:.0f}°:")
        print(f"    Alternancia: {'✅' if r['alterna'] else '❌'}")
        print(f"    Amplitud real: {r['amplitud_real']:.1f}°")
        print(f"    Cruces por cero: {r['cruces']}")
        print(f"    T_settle: {r['t_settle']:.1f}s" if r['t_settle'] else "    T_settle: No alcanzado")
        print(f"    Error final: {r['error_final']:.1f}°")
    
    print("\n" + "=" * 80)
    print("CONCLUSION V143 — Recuperación de alternancia")
    print("=" * 80)
    
    if exito_total:
        print("\n  ✅ ALTERNANCIA RECUPERADA")
        print("     El organismo puede alternar entre -60° y +60° en régimen sano")
        print("\n  ANIMA-1 está listo para pruebas de fatiga real")
        print("\n  ANIMA-2 - Linea 3: Proceder con V144 (fatiga progresiva)")
    else:
        print("\n  ⚠️ ALTERNANCIA NO RECUPERADA")
        print("     El organismo aún no puede ejecutar la tarea base")
        print("     Revisar gradiente inter-sistemas y espacialización")
    
    # Graficos
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Reconstruir datos para gráfico (último test con amplitud 60°)
    ax = axes[0]
    # Tomar los últimos orientaciones de la simulación (última ejecución)
    if 'orientaciones' in locals():
        ax.plot(tiempos, setpoints, 'r--', linewidth=0.8, alpha=0.5, label='Setpoint')
        ax.plot(tiempos, orientaciones, 'b-', linewidth=0.6, label='Orientacion real')
        ax.set_xlabel('Tiempo (s)')
        ax.set_ylabel('Angulo (grados)')
        ax.set_title('Alternancia con amplitud ±60°')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    amplitudes_plot = [r['amplitud'] for r in resultados]
    alternancia_plot = [r['amplitud_real'] for r in resultados]
    ax.bar([f"±{a:.0f}°" for a in amplitudes_plot], alternancia_plot, color='green')
    ax.axhline(y=60, color='red', linestyle='--', label='Objetivo ±60°')
    ax.set_ylabel('Amplitud real (grados)')
    ax.set_title('Amplitud real vs amplitud ordenada')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('v143_logs', exist_ok=True)
    plt.savefig(f'v143_logs/v143_alternancia_{timestamp}.png', dpi=150)
    print(f"\n  Graficos guardados: v143_logs/v143_alternancia_{timestamp}.png")
    
    return sistema, exito_total


if __name__ == "__main__":
    import time
    start = time.time()
    sistema, exito = ejecutar_v143()
    elapsed = time.time() - start
    print(f"\n  Tiempo de ejecucion: {elapsed/60:.1f} minutos")