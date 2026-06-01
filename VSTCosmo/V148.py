#!/usr/bin/env python3
"""
VSTCosmos V148 — Fatiga progresiva sobre baseline sano

Basado en V147 (baseline confirmado):
  - Kp_base: 0.002
  - Inercia: 0.95
  - Zona muerta: 2.0°
  - Período: 80s (40s por polo)
  - Dirección por error, confianza por gradiente

Fatiga gradual (calibrada para degradación mensurable, no letal):
  - K_GAIN: 0.0002 (degrada Kp a 0.55 en E=3000°)
  - K_PRECISION: 0.001 (zona muerta: 2.0° → 5.0° en E=3000°)
  - K_TEMBLOR: 0.0005 (temblor: 1.5° en E=3000°)
  - TAU_RECUPERACION: 300.0s (5 minutos)

Hipótesis O-N11:
  - T_settle_fat / T_settle_fresco > 1.3 (degradación mensurable)
  - Recuperación > 20%
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys
from collections import deque

# ============================================================
# PARAMETROS (basados en V147)
# ============================================================
DT = 0.01
TIEMPO_POR_REPETICION = 452.0
REPETICIONES_LENTAS = 10

# Asimetria forzada
SESGO_L = 0.05
SESGO_R = -0.05
DIM_HEMISFERIO = 32

# Zona muerta base
ZONA_MUERTA_BASE = 2.0
ZONA_MUERTA_MAX = 10.0

# Limites de plasticidad
KP_BASE = 0.002
KP_MIN = 0.0005
KP_MAX = 0.005

# Plasticidad
HABITUACION_RAPIDA = 0.99
SENSIBILIZACION_LENTA = 1.01
VENTANA_OSCILACION = 100

# Inercia
INERCIA = 0.95
SENSIBILIDAD_GRAD = 10.0

# Fatiga V148 (gradual, no letal)
K_GAIN = 0.0002        # E=3000° → factor=0.55
K_PRECISION = 0.001    # E=3000° → zona_muerta=2+3=5°
K_TEMBLOR = 0.0005     # E=3000° → temblor=1.5°
TAU_RECUPERACION = 300.0

# Semilla base
SEMILLA_BASE = 44

# Período (como V147)
PERIODO_ALTERNANCIA = 80.0


# ============================================================
# HEMISFERIO
# ============================================================

class HemisferioV148:
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
# FATIGA METABOLICA REAL (V148 - GRADUAL)
# ============================================================

class FatigaMetabolicaReal:
    def __init__(self, k_gain=K_GAIN, k_precision=K_PRECISION,
                 k_temblor=K_TEMBLOR, tau_recuperacion=TAU_RECUPERACION):
        self.k_gain = k_gain
        self.k_precision = k_precision
        self.k_temblor = k_temblor
        self.tau_recuperacion = tau_recuperacion
        
        self.energia_total = 0.0
        self.historial_energia = []
        self.historial_factor_gain = []
    
    def actualizar(self, delta_orientacion, en_reposo, dt):
        if not en_reposo:
            self.energia_total += abs(delta_orientacion)
        else:
            self.energia_total *= np.exp(-dt / self.tau_recuperacion)
        
        # Calcular efectos
        factor_gain = np.exp(-self.k_gain * self.energia_total)
        zona_muerta_efectiva = ZONA_MUERTA_BASE + self.k_precision * self.energia_total
        temblor = self.k_temblor * self.energia_total * np.random.randn()
        
        # Limitar
        factor_gain = max(0.3, min(1.0, factor_gain))
        zona_muerta_efectiva = min(ZONA_MUERTA_MAX, zona_muerta_efectiva)
        temblor = np.clip(temblor, -2.0, 2.0)
        
        self.historial_energia.append(self.energia_total)
        self.historial_factor_gain.append(factor_gain)
        
        return factor_gain, zona_muerta_efectiva, temblor
    
    def reset(self):
        self.energia_total = 0.0
        self.historial_energia = []
        self.historial_factor_gain = []
    
    def get_energia(self):
        return self.energia_total


# ============================================================
# APARATO MOTOR V148 (con fatiga gradual)
# ============================================================

class AparatoMotorV148:
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
        
        # Fatiga activada
        self.fatiga = FatigaMetabolicaReal()
        
        self.memoria_error = deque(maxlen=VENTANA_OSCILACION)
        self.historial_Kp = []
        
        self.ultimo_delta_registrado = 0.0
    
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
        
        if abs(gradiente) < 0.01:
            return self.orientacion, self.fatiga.get_energia(), 0.0
        
        setpoint_objetivo = setpoint_percepcion if fuente_activa else 0.0
        error = setpoint_objetivo - self.orientacion
        
        # Actualizar fatiga
        factor_gain, zona_muerta_efectiva, temblor = self.fatiga.actualizar(
            self.ultimo_delta_registrado, not fuente_activa, DT
        )
        
        if abs(error) < zona_muerta_efectiva:
            return self.orientacion, self.fatiga.get_energia(), zona_muerta_efectiva
        
        # Dirección: viene del error
        direccion = np.sign(error)
        
        # Confianza: viene del gradiente
        confianza = min(1.0, abs(gradiente) * self.sensibilidad_grad)
        
        # Freno exponencial
        factor_freno = self.calcular_factor_freno(error)
        
        # Kp efectivo (fatiga + confianza)
        Kp_efectivo = self.Kp_actual * factor_gain * confianza
        
        # Delta
        delta_raw = Kp_efectivo * abs(error) * direccion * factor_freno
        
        # Inercia
        delta = self.inercia * self.ultimo_delta + (1 - self.inercia) * delta_raw
        self.ultimo_delta = delta
        self.ultimo_delta_registrado = delta
        
        # Temblor
        delta += temblor * DT
        
        self.actualizar_plasticidad(error)
        
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
# SISTEMA V148
# ============================================================

class SistemaV148:
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
        
        self.izquierdo = HemisferioV148("L", 30.0, generar_ruido_rosa, seed=seed, sesgo=SESGO_L)
        self.derecho = HemisferioV148("R", 300.0, generar_clicks_poisson, seed=seed+100, sesgo=SESGO_R)
        
        self.sistema_B_izq = HemisferioV148("B_L", 30.0, generar_ruido_rosa, seed=seed+200, sesgo=SESGO_L)
        self.sistema_B_der = HemisferioV148("B_R", 300.0, generar_clicks_poisson, seed=seed+300, sesgo=SESGO_R)
        
        self.modo_entrenamiento = True
        self.motor = AparatoMotorV148()
        
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

def analizar_ciclo(orientaciones, setpoints, dt=DT, umbral_error=2.0, ventana=100):
    """Analiza un ciclo completo (desde inicio hasta entrada en zona muerta)"""
    if len(orientaciones) == 0:
        return None, None, None, None
    
    errores = np.abs(np.array(orientaciones) - np.array(setpoints))
    
    # T_settle
    t_settle = None
    for i in range(len(errores) - ventana):
        if all(errores[i:i+ventana] < umbral_error):
            t_settle = i * dt
            break
    
    # Error final (promedio últimos ventana pasos)
    if len(errores) > ventana:
        error_final = np.mean(errores[-ventana:])
    else:
        error_final = errores[-1] if len(errores) > 0 else None
    
    # Amplitud real del ciclo
    amplitud_real = max(orientaciones) - min(orientaciones)
    
    # Velocidad media
    if len(orientaciones) > 1:
        velocidad_media = np.mean(np.abs(np.diff(orientaciones))) / dt
    else:
        velocidad_media = 0
    
    return t_settle, error_final, amplitud_real, velocidad_media


def ejecutar_ciclos(sistema, t_actual, num_ciclos, nombre_fase, verbose=True):
    """Ejecuta num_ciclos de alternancia ±60°"""
    tiempos = []
    orientaciones = []
    setpoints = []
    energias = []
    zonas = []
    
    for ciclo in range(num_ciclos):
        for i in range(int(PERIODO_ALTERNANCIA / DT)):
            t = t_actual + (ciclo * PERIODO_ALTERNANCIA + i) * DT
            t_rel = i * DT
            
            setpoint = onda_cuadrada(t_rel, periodo=PERIODO_ALTERNANCIA, amplitud=60.0)
            orient, energia = sistema.actualizar(t, DT, t_actual + 1000, setpoint)
            
            tiempos.append(t)
            orientaciones.append(orient)
            setpoints.append(setpoint)
            energias.append(energia)
            zonas.append(sistema.historial['zona_muerta'][-1])
        
        if verbose and (ciclo + 1) % 10 == 0:
            print(f"      Ciclo {ciclo + 1}/{num_ciclos} completado, energia={energia:.0f}°")
    
    return t_actual + num_ciclos * PERIODO_ALTERNANCIA, tiempos, orientaciones, setpoints, energias, zonas


# ============================================================
# EXPERIMENTO V148
# ============================================================

def ejecutar_v148():
    print("=" * 100)
    print("EXPERIMENTO V148 — Fatiga progresiva sobre baseline sano")
    print("=" * 100)
    print("  Basado en V147 (baseline confirmado)")
    print("  Parámetros fatiga (gradual, no letal):")
    print(f"    - K_GAIN: {K_GAIN}")
    print(f"    - K_PRECISION: {K_PRECISION}")
    print(f"    - K_TEMBLOR: {K_TEMBLOR}")
    print(f"    - TAU_RECUPERACION: {TAU_RECUPERACION}s")
    print("")
    print("  Protocolo:")
    print("    F1: Baseline fresco (3 ciclos)")
    print("    F2: Fatiga inducida (50 ciclos)")
    print("    F3: Test fatigado (3 ciclos)")
    print("    F4: Recuperacion (300s reposo)")
    print("    F5: Test post-recuperacion (3 ciclos)")
    print("=" * 100)
    
    sistema = SistemaV148("V148", seed=SEMILLA_BASE)
    
    print("\n  Entrenando lateralidad (10 repeticiones)...")
    sistema.set_modo_entrenamiento(True)
    
    for rep in range(REPETICIONES_LENTAS):
        for i in range(int(TIEMPO_POR_REPETICION / DT)):
            t = rep * TIEMPO_POR_REPETICION + i * DT
            sistema.actualizar(t, DT, TIEMPO_POR_REPETICION * REPETICIONES_LENTAS,
                              setpoint_real=0.0)
    
    print("  Entrenamiento completado.")
    print("\n  Iniciando test de fatiga progresiva...")
    
    sistema.set_modo_entrenamiento(False)
    sistema.motor.reset()
    t_actual = TIEMPO_POR_REPETICION * REPETICIONES_LENTAS
    
    # F1: Baseline fresco (3 ciclos)
    print("\n  F1: Baseline fresco (3 ciclos)...")
    t_actual, t1, o1, s1, e1, z1 = ejecutar_ciclos(sistema, t_actual, 3, "F1", verbose=True)
    
    # F2: Fatiga inducida (50 ciclos)
    print("\n  F2: Fatiga inducida (50 ciclos)...")
    t_actual, t2, o2, s2, e2, z2 = ejecutar_ciclos(sistema, t_actual, 50, "F2", verbose=True)
    
    # F3: Test fatigado (3 ciclos)
    print("\n  F3: Test fatigado (3 ciclos)...")
    t_actual, t3, o3, s3, e3, z3 = ejecutar_ciclos(sistema, t_actual, 3, "F3", verbose=True)
    
    # F4: Recuperacion (300s reposo)
    print("\n  F4: Recuperacion (300s reposo)...")
    for i in range(int(TAU_RECUPERACION / DT)):
        t = t_actual + i * DT
        orient, energia = sistema.actualizar(t, DT, t_actual + TAU_RECUPERACION, 0.0)
    t_actual += TAU_RECUPERACION
    
    # F5: Test post-recuperacion (3 ciclos)
    print("\n  F5: Test post-recuperacion (3 ciclos)...")
    t_actual, t5, o5, s5, e5, z5 = ejecutar_ciclos(sistema, t_actual, 3, "F5", verbose=True)
    
    # ============================================================
    # ANALISIS
    # ============================================================
    print("\n" + "=" * 80)
    print("ANALISIS DE FATIGA PROGRESIVA")
    print("=" * 80)
    
    # Analizar primer ciclo de cada fase (para comparación directa)
    def get_primer_ciclo(orientaciones, setpoints):
        # Tomar los primeros PERIODO_ALTERNANCIA segundos
        paso_por_segundo = int(1.0 / DT)
        fin_ciclo = int(PERIODO_ALTERNANCIA / DT)
        
        o_ciclo = orientaciones[:fin_ciclo]
        s_ciclo = setpoints[:fin_ciclo]
        
        return analizar_ciclo(o_ciclo, s_ciclo)
    
    # F1: Primer ciclo
    t_settle_f1, error_f1, amp_f1, vel_f1 = get_primer_ciclo(o1, s1)
    energia_f1 = e1[-1] if e1 else 0
    
    # F3: Primer ciclo fatigado
    t_settle_f3, error_f3, amp_f3, vel_f3 = get_primer_ciclo(o3, s3)
    energia_f3 = e3[-1] if e3 else 0
    
    # F5: Primer ciclo post-recuperacion
    t_settle_f5, error_f5, amp_f5, vel_f5 = get_primer_ciclo(o5, s5)
    energia_f5 = e5[-1] if e5 else 0
    
    print(f"\n  F1 - Baseline fresco:")
    print(f"    T_settle: {t_settle_f1:.1f}s" if t_settle_f1 else "    T_settle: No alcanzado")
    print(f"    Error final: {error_f1:.1f}°" if error_f1 else "    Error final: N/A")
    print(f"    Amplitud real: {amp_f1:.1f}°")
    print(f"    Velocidad media: {vel_f1:.2f}°/s")
    print(f"    Energia acumulada: {energia_f1:.0f}°")
    
    print(f"\n  F3 - Fatigado (despues de 50 ciclos):")
    print(f"    T_settle: {t_settle_f3:.1f}s" if t_settle_f3 else "    T_settle: No alcanzado")
    print(f"    Error final: {error_f3:.1f}°" if error_f3 else "    Error final: N/A")
    print(f"    Amplitud real: {amp_f3:.1f}°")
    print(f"    Velocidad media: {vel_f3:.2f}°/s")
    print(f"    Energia acumulada: {energia_f3:.0f}°")
    
    print(f"\n  F5 - Post-recuperacion (300s reposo):")
    print(f"    T_settle: {t_settle_f5:.1f}s" if t_settle_f5 else "    T_settle: No alcanzado")
    print(f"    Error final: {error_f5:.1f}°" if error_f5 else "    Error final: N/A")
    print(f"    Amplitud real: {amp_f5:.1f}°")
    print(f"    Velocidad media: {vel_f5:.2f}°/s")
    print(f"    Energia acumulada: {energia_f5:.0f}°")
    
    # Degradacion y recuperacion
    if t_settle_f1 and t_settle_f3:
        degradacion = t_settle_f3 / t_settle_f1
        print(f"\n  Degradacion por fatiga: {degradacion:.2f}x")
    else:
        degradacion = None
    
    if t_settle_f1 and t_settle_f3 and t_settle_f5:
        recuperacion = (t_settle_f3 - t_settle_f5) / (t_settle_f3 - t_settle_f1) * 100 if t_settle_f3 != t_settle_f1 else 0
        print(f"  Recuperacion post-reposo: {recuperacion:.1f}%")
    else:
        recuperacion = None
    
    exito_fatiga = degradacion is not None and degradacion > 1.3
    exito_recuperacion = recuperacion is not None and recuperacion > 20
    
    print("\n" + "=" * 80)
    print("CONCLUSION V148")
    print("=" * 80)
    
    if exito_fatiga and exito_recuperacion:
        print("\n  ✅ O-N11 VALIDADA")
        print("     La fatiga degrada el rendimiento significativamente")
        print("     El reposo restaura parcialmente la funcion")
        print("\n  ANIMA-2 - Linea 3: CERRADA")
    elif exito_fatiga:
        print("\n  ✅ FATIGA DEMOSTRADA, RECUPERACION PARCIAL")
        print("     O-N11 validada parcialmente")
    else:
        print("\n  ⚠️ O-N11 NO VALIDADA")
        if degradacion:
            print(f"     Degradacion: {degradacion:.2f}x (<1.3x)")
        if recuperacion:
            print(f"     Recuperacion: {recuperacion:.1f}% (<20%)")
    
    # Grafico
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Grafico 1: Comparativa de orientacion
    ax = axes[0]
    # Tomar primeros 80s de F1 y F3
    fin = int(PERIODO_ALTERNANCIA / DT)
    ax.plot(s1[:fin], 'r--', linewidth=0.8, alpha=0.5, label='Setpoint')
    ax.plot(o1[:fin], 'b-', linewidth=0.6, label='F1 (fresco)')
    ax.plot(o3[:fin], 'orange', linewidth=0.6, label='F3 (fatigado)')
    ax.set_xlabel('Paso')
    ax.set_ylabel('Angulo (grados)')
    ax.set_title('Comparativa: Fresco vs Fatigado')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Grafico 2: Energia acumulada
    ax = axes[1]
    todas_energias = []
    for e in [e1, e2, e3, e5]:
        todas_energias.extend(e)
    ax.plot(todas_energias, 'r-', linewidth=0.8)
    ax.set_xlabel('Paso')
    ax.set_ylabel('Energia acumulada (grados)')
    ax.set_title('Energia metabolica global')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('v148_logs', exist_ok=True)
    plt.savefig(f'v148_logs/v148_fatiga_progresiva_{timestamp}.png', dpi=150)
    print(f"\n  Graficos guardados: v148_logs/v148_fatiga_progresiva_{timestamp}.png")
    
    return sistema, exito_fatiga and exito_recuperacion


if __name__ == "__main__":
    import time
    start = time.time()
    sistema, exito = ejecutar_v148()
    elapsed = time.time() - start
    print(f"\n  Tiempo de ejecucion: {elapsed/60:.1f} minutos")