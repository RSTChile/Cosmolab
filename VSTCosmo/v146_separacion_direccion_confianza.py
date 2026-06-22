#!/usr/bin/env python3
"""
VSTCosmos V146 — Separación dirección/confianza

Corrección sobre V143-V145:
  - La dirección del movimiento viene del ERROR (setpoint - orientacion)
  - La confianza viene del gradiente inter-sistemas (modula ganancia)
  - El gradiente ya NO determina el signo del movimiento

Esto resuelve:
  - V143: alternaba con signo invertido (gradiente tenía sesgo)
  - V144-V145: saturación en +90° (gradiente positivo fijo)

Hipótesis:
  - El organismo alternará correctamente sin saturación
  - T_settle < 30s, error final < 10°
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys
from collections import deque

# ============================================================
# PARAMETROS (basados en V143 que funcionaba parcialmente)
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

# Inercia (como V143)
INERCIA = 0.95

# Sensibilidad del gradiente para confianza
SENSIBILIDAD_GRAD = 10.0

# Fatiga DESACTIVADA para validar baseline
K_GAIN = 0.0
K_PRECISION = 0.0
K_TEMBLOR = 0.0
TAU_RECUPERACION = 1.0

# Semilla base
SEMILLA_BASE = 44

# Onda cuadrada
PERIODO_ALTERNANCIA = 40.0  # 20s en cada polo


# ============================================================
# HEMISFERIO (idéntico a V143)
# ============================================================

class HemisferioV146:
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
# FATIGA DESACTIVADA
# ============================================================

class FatigaDesactivada:
    def __init__(self):
        self.energia_total = 0.0
        self.historial_energia = []
    
    def actualizar(self, delta_orientacion, en_reposo, dt):
        self.historial_energia.append(self.energia_total)
        return 1.0, ZONA_MUERTA_BASE, 0.0
    
    def reset(self):
        self.energia_total = 0.0
        self.historial_energia = []
    
    def get_energia(self):
        return self.energia_total


# ============================================================
# APARATO MOTOR V146 - SEPARACIÓN DIRECCIÓN/CONFIANZA
# ============================================================

class AparatoMotorV146:
    def __init__(self):
        self.orientacion = 0.0
        self.Kp_base = KP_BASE
        self.Kp_actual = KP_BASE
        self.Kp_min = KP_MIN
        self.Kp_max = KP_MAX
        self.limite = 90.0
        self.zona_muerta = ZONA_MUERTA_BASE
        self.inercia = INERCIA
        self.ultimo_delta = 0.0
        self.sensibilidad_grad = SENSIBILIDAD_GRAD
        self.t = 0.0
        
        self.fatiga = FatigaDesactivada()
        
        self.memoria_error = deque(maxlen=VENTANA_OSCILACION)
        self.historial_Kp = []
        
        self.ultimo_delta_registrado = 0.0
        self.temblor_actual = 0.0
    
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
    
    def actuar(self, gradiente, LF_activa, fuente_activa, t, setpoint_percepcion):
        if not LF_activa:
            return self.orientacion, 0.0, 0.0
        
        # Ignorar gradiente muy pequeño
        if abs(gradiente) < 0.01:
            return self.orientacion, self.fatiga.get_energia(), 0.0
        
        setpoint_objetivo = setpoint_percepcion if fuente_activa else 0.0
        error = setpoint_objetivo - self.orientacion
        
        # Fatiga desactivada
        factor_gain, zona_muerta_efectiva, temblor = self.fatiga.actualizar(
            self.ultimo_delta_registrado, not fuente_activa, DT
        )
        self.temblor_actual = temblor
        
        # Zona muerta
        if abs(error) < zona_muerta_efectiva:
            return self.orientacion, self.fatiga.get_energia(), zona_muerta_efectiva
        
        # ============================================================
        # NUEVA LÓGICA V146: Separación dirección/confianza
        # ============================================================
        
        # 1. DIRECCIÓN: viene del ERROR (setpoint - orientacion)
        direccion = np.sign(error)
        
        # 2. CONFIANZA: viene del gradiente inter-sistemas
        #    Cuanto mayor |gradiente|, más clara es la señal
        confianza = min(1.0, abs(gradiente) * self.sensibilidad_grad)
        
        # 3. FRENO: reduce velocidad cerca del objetivo
        factor_freno = self.calcular_factor_freno(error)
        
        # 4. Kp efectivo: base * confianza * factor_fatiga
        Kp_efectivo = self.Kp_actual * factor_gain * confianza
        
        # 5. Delta: dirección fija, magnitud proporcional al error
        delta_raw = Kp_efectivo * abs(error) * direccion * factor_freno
        
        # 6. Inercia (suavizado)
        delta = self.inercia * self.ultimo_delta + (1 - self.inercia) * delta_raw
        self.ultimo_delta = delta
        self.ultimo_delta_registrado = delta
        
        # 7. Temblor (desactivado)
        delta += self.temblor_actual * DT
        
        # 8. Actualizar plasticidad
        self.actualizar_plasticidad(error)
        
        # 9. Actualizar orientación
        self.orientacion += delta
        self.orientacion = np.clip(self.orientacion, -self.limite, self.limite)
        
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
        self.fatiga.reset()


# ============================================================
# SISTEMA V146
# ============================================================

class SistemaV146:
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
        
        self.izquierdo = HemisferioV146("L", 30.0, generar_ruido_rosa, seed=seed, sesgo=SESGO_L)
        self.derecho = HemisferioV146("R", 300.0, generar_clicks_poisson, seed=seed+100, sesgo=SESGO_R)
        
        self.sistema_B_izq = HemisferioV146("B_L", 30.0, generar_ruido_rosa, seed=seed+200, sesgo=SESGO_L)
        self.sistema_B_der = HemisferioV146("B_R", 300.0, generar_clicks_poisson, seed=seed+300, sesgo=SESGO_R)
        
        self.modo_entrenamiento = True
        self.motor = AparatoMotorV146()
        
        self.historial = {
            't': [],
            'orientacion': [],
            'setpoint_real': [],
            'gradiente': [],
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
        self.historial['gradiente'].append(gradiente)
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
# ONDA CUADRADA
# ============================================================

def onda_cuadrada(t, periodo=PERIODO_ALTERNANCIA, amplitud=60.0):
    if (t % periodo) < (periodo / 2):
        return -amplitud
    else:
        return +amplitud


# ============================================================
# ANALISIS
# ============================================================

def analizar_alternancia(orientaciones, setpoints, umbral_amplitud=40.0):
    if len(orientaciones) == 0:
        return False, 0.0, 0, False
    
    amplitud_real = max(orientaciones) - min(orientaciones)
    
    cruces = 0
    for i in range(1, len(orientaciones)):
        if orientaciones[i-1] * orientaciones[i] < 0:
            cruces += 1
    
    setpoint_medio = np.mean(setpoints[-100:]) if len(setpoints) > 100 else 0
    orient_medio = np.mean(orientaciones[-100:]) if len(orientaciones) > 100 else 0
    
    signo_correcto = (setpoint_medio * orient_medio) > 0 if abs(setpoint_medio) > 10 else True
    
    alterna = (amplitud_real > umbral_amplitud) and (cruces > 2) and signo_correcto
    
    return alterna, amplitud_real, cruces, signo_correcto


def calcular_t_settle(orientaciones, setpoints, dt=DT, umbral_error=2.0):
    if len(orientaciones) == 0:
        return None
    
    errores = np.abs(np.array(orientaciones) - np.array(setpoints))
    
    for i in range(len(errores) - 50):
        if all(errores[i:i+50] < umbral_error):
            return i * dt
    return None


# ============================================================
# EXPERIMENTO V146
# ============================================================

def ejecutar_v146():
    print("=" * 100)
    print("EXPERIMENTO V146 — Separación dirección/confianza")
    print("=" * 100)
    print("  Corrección sobre V143-V145:")
    print("    - La dirección del movimiento viene del ERROR")
    print("    - La confianza viene del gradiente (modula ganancia)")
    print("    - El gradiente ya NO determina el signo del movimiento")
    print("")
    print("  Parámetros:")
    print(f"    - Kp_base: {KP_BASE}")
    print(f"    - Inercia: {INERCIA}")
    print(f"    - Zona muerta: {ZONA_MUERTA_BASE}°")
    print(f"    - Período: {PERIODO_ALTERNANCIA}s (20s en cada polo)")
    print("=" * 100)
    
    sistema = SistemaV146("V146", seed=SEMILLA_BASE)
    
    print("\n  Entrenando lateralidad (10 repeticiones)...")
    sistema.set_modo_entrenamiento(True)
    
    for rep in range(REPETICIONES_LENTAS):
        for i in range(int(TIEMPO_POR_REPETICION / DT)):
            t = rep * TIEMPO_POR_REPETICION + i * DT
            sistema.actualizar(t, DT, TIEMPO_POR_REPETICION * REPETICIONES_LENTAS,
                              setpoint_real=0.0)
    
    print("  Entrenamiento completado.")
    print("\n  Iniciando test de alternancia con dirección por error...")
    
    sistema.set_modo_entrenamiento(False)
    sistema.motor.reset()
    
    t_actual = TIEMPO_POR_REPETICION * REPETICIONES_LENTAS
    
    amplitudes = [30.0, 45.0, 60.0]
    resultados = []
    
    for amplitud in amplitudes:
        print(f"\n  Test con amplitud: ±{amplitud:.0f}°")
        
        tiempos = []
        orientaciones = []
        setpoints = []
        gradientes = []
        
        for i in range(int(3 * PERIODO_ALTERNANCIA / DT)):
            t = t_actual + i * DT
            t_rel = i * DT
            
            setpoint = onda_cuadrada(t_rel, periodo=PERIODO_ALTERNANCIA, amplitud=amplitud)
            orient, energia = sistema.actualizar(t, DT, t_actual + 200, setpoint)
            
            tiempos.append(t_rel)
            orientaciones.append(orient)
            setpoints.append(setpoint)
            gradientes.append(sistema.historial['gradiente'][-1] if sistema.historial['gradiente'] else 0)
            
            if i % 1000 == 0:
                grad = gradientes[-1] if gradientes else 0
                print(f"      t={t_rel:.0f}s | setpoint={setpoint:+.0f}° | orient={orient:.1f}° | grad={grad:.3f}")
        
        alterna, amp_real, cruces, signo_correcto = analizar_alternancia(orientaciones, setpoints, umbral_amplitud=amplitud * 0.7)
        t_settle = calcular_t_settle(orientaciones, setpoints, DT, umbral_error=ZONA_MUERTA_BASE)
        
        ciclo_final = orientaciones[-int(PERIODO_ALTERNANCIA/DT):]
        setpoint_final = setpoints[-int(PERIODO_ALTERNANCIA/DT):]
        error_final = np.mean(np.abs(np.array(ciclo_final) - np.array(setpoint_final)))
        
        resultados.append({
            'amplitud': amplitud,
            'alterna': alterna,
            'amplitud_real': amp_real,
            'cruces': cruces,
            'signo_correcto': signo_correcto,
            't_settle': t_settle,
            'error_final': error_final
        })
        
        print(f"      Resultado: alternancia={'✅' if alterna else '❌'} | "
              f"signo={'✅' if signo_correcto else '❌'} | "
              f"amplitud_real={amp_real:.1f}° | cruces={cruces} | "
              f"t_settle={t_settle:.1f}s" if t_settle else f"t_settle=∞ | "
              f"error_final={error_final:.1f}°")
        
        t_actual += 3 * PERIODO_ALTERNANCIA
    
    print("\n" + "=" * 80)
    print("RESULTADOS V146")
    print("=" * 80)
    
    for r in resultados:
        print(f"\n  Amplitud ±{r['amplitud']:.0f}°:")
        print(f"    Alternancia: {'✅' if r['alterna'] else '❌'}")
        print(f"    Signo correcto: {'✅' if r['signo_correcto'] else '❌'}")
        print(f"    Amplitud real: {r['amplitud_real']:.1f}°")
        print(f"    T_settle: {r['t_settle']:.1f}s" if r['t_settle'] else "    T_settle: No alcanzado")
        print(f"    Error final: {r['error_final']:.1f}°")
    
    exito_60 = next((r for r in resultados if r['amplitud'] == 60.0), None)
    
    if exito_60:
        baseline_ok = exito_60['signo_correcto'] and exito_60['t_settle'] is not None and exito_60['t_settle'] < 30.0 and exito_60['error_final'] < 10.0
    else:
        baseline_ok = False
    
    print("\n" + "=" * 80)
    print("CONCLUSION V146")
    print("=" * 80)
    
    if baseline_ok:
        print("\n  ✅ BASELINE SANO CONFIRMADO")
        print("     Separación dirección/confianza funciona")
        print("     El organismo orienta correctamente (±60°)")
        print("     T_settle < 30s, Error final < 10°")
        print("\n  ANIMA-1 listo para V147 (Fatiga progresiva)")
    else:
        print("\n  ⚠️ BASELINE PARCIAL")
        if exito_60:
            if not exito_60['signo_correcto']:
                print("     Problema: Signo de orientación aún invertido")
            if exito_60['t_settle'] is None:
                print("     Problema: T_settle no alcanzado")
            if exito_60['error_final'] > 10:
                print(f"     Problema: Error final alto ({exito_60['error_final']:.1f}°)")
        else:
            print("     Problema: No se obtuvieron datos para ±60°")
    
    # Grafico
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    if 'tiempos' in locals() and 'orientaciones' in locals():
        ax = axes[0]
        ax.plot(tiempos, setpoints, 'r--', linewidth=0.8, alpha=0.5, label='Setpoint')
        ax.plot(tiempos, orientaciones, 'b-', linewidth=0.6, label='Orientacion real')
        ax.axhline(y=ZONA_MUERTA_BASE, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(y=-ZONA_MUERTA_BASE, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Tiempo (s)')
        ax.set_ylabel('Angulo (grados)')
        ax.set_title('V146: Separación dirección/confianza (±60°)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    amps_plot = [r['amplitud'] for r in resultados]
    errors_plot = [r['error_final'] for r in resultados]
    ax.plot(amps_plot, errors_plot, 'o-', color='red', linewidth=2, markersize=8)
    ax.axhline(y=10, color='green', linestyle='--', label='Objetivo error <10°')
    ax.set_xlabel('Amplitud ordenada (grados)')
    ax.set_ylabel('Error final (grados)')
    ax.set_title('Error vs amplitud')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('v146_logs', exist_ok=True)
    plt.savefig(f'v146_logs/v146_direccion_confianza_{timestamp}.png', dpi=150)
    print(f"\n  Graficos guardados: v146_logs/v146_direccion_confianza_{timestamp}.png")
    
    return sistema, baseline_ok


if __name__ == "__main__":
    import time
    start = time.time()
    sistema, exito = ejecutar_v146()
    elapsed = time.time() - start
    print(f"\n  Tiempo de ejecucion: {elapsed/60:.1f} minutos")