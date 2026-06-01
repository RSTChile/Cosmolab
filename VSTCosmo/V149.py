#!/usr/bin/env python3
"""
VSTCosmos V149 — Separación entre historia y fatiga activa

Correcciones sobre V148:
  1. Separación conceptual: E_historia (permanente) vs E_fatiga (recuperable)
  2. Solo la fatiga activa degrada el rendimiento
  3. La historia solo acumula (registro permanente, sin efecto en acción)
  4. Reposo recupera fatiga_activa, no afecta historia

Hipótesis O-N11 (revisada):
  - Fatiga activa degrada T_settle y aumenta error
  - Reposo recupera parcialmente el rendimiento
  - La historia permanece como registro (no afecta acción presente)
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys
from collections import deque

# ============================================================
# PARAMETROS (basados en V147 baseline)
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
ZONA_MUERTA_MAX = 15.0

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

# Fatiga V149 (separación historia/fatiga)
K_GAIN = 0.0003        # fatiga_activa=3000° → factor=0.41
K_PRECISION = 0.002    # fatiga_activa=3000° → zona_muerta=2+6=8.0°
K_TEMBLOR = 0.001      # fatiga_activa=3000° → temblor=3.0°
TAU_RECUPERACION = 180.0  # 3 minutos para recuperación completa

# Semilla base
SEMILLA_BASE = 44

# Período (como V147)
PERIODO_ALTERNANCIA = 80.0


# ============================================================
# HEMISFERIO (idéntico a V147)
# ============================================================

class HemisferioV149:
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
# FATIGA METABOLICA CON SEPARACIÓN HISTORIA/FATIGA (V149)
# ============================================================

class FatigaMetabolicaV149:
    """
    Separa:
      - historia: registro permanente (nunca decae)
      - fatiga_activa: estado recuperable (decae con reposo)
    
    Solo la fatiga_activa afecta el rendimiento actual.
    """
    
    def __init__(self, k_gain=K_GAIN, k_precision=K_PRECISION,
                 k_temblor=K_TEMBLOR, tau_recuperacion=TAU_RECUPERACION):
        self.k_gain = k_gain
        self.k_precision = k_precision
        self.k_temblor = k_temblor
        self.tau_recuperacion = tau_recuperacion
        
        self.historia = 0.0        # Permanente, nunca decae
        self.fatiga_activa = 0.0   # Recuperable, decae con reposo
        
        self.historial_historia = []
        self.historial_fatiga = []
        self.historial_factor_gain = []
    
    def actualizar(self, delta_orientacion, en_reposo, dt):
        # 1. HISTORIA: acumula SIEMPRE (registro permanente)
        self.historia += abs(delta_orientacion)
        
        # 2. FATIGA ACTIVA: acumula durante movimiento, decae durante reposo
        if not en_reposo:
            self.fatiga_activa += abs(delta_orientacion)
        else:
            self.fatiga_activa *= np.exp(-dt / self.tau_recuperacion)
        
        # 3. Calcular efectos SOLO con fatiga_activa (no historia)
        factor_gain = np.exp(-self.k_gain * self.fatiga_activa)
        zona_muerta_efectiva = ZONA_MUERTA_BASE + self.k_precision * self.fatiga_activa
        temblor = self.k_temblor * self.fatiga_activa * np.random.randn()
        
        # Limitar
        factor_gain = max(0.2, min(1.0, factor_gain))
        zona_muerta_efectiva = min(ZONA_MUERTA_MAX, zona_muerta_efectiva)
        temblor = np.clip(temblor, -3.0, 3.0)
        
        # Guardar historial
        self.historial_historia.append(self.historia)
        self.historial_fatiga.append(self.fatiga_activa)
        self.historial_factor_gain.append(factor_gain)
        
        return factor_gain, zona_muerta_efectiva, temblor
    
    def reset(self):
        self.historia = 0.0
        self.fatiga_activa = 0.0
        self.historial_historia = []
        self.historial_fatiga = []
        self.historial_factor_gain = []
    
    def get_historia(self):
        return self.historia
    
    def get_fatiga(self):
        return self.fatiga_activa


# ============================================================
# APARATO MOTOR V149 (con fatiga aplicada correctamente)
# ============================================================

class AparatoMotorV149:
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
        
        # Fatiga con separación historia/fatiga
        self.fatiga = FatigaMetabolicaV149()
        
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
            return self.orientacion, 0.0, 0.0, 0.0
        
        if abs(gradiente) < 0.01:
            return self.orientacion, self.fatiga.get_historia(), self.fatiga.get_fatiga(), 0.0
        
        setpoint_objetivo = setpoint_percepcion if fuente_activa else 0.0
        error = setpoint_objetivo - self.orientacion
        
        # Actualizar fatiga con el delta REAL del paso anterior
        factor_gain, zona_muerta_efectiva, temblor = self.fatiga.actualizar(
            self.ultimo_delta_registrado, not fuente_activa, DT
        )
        
        # Zona muerta expandida por fatiga
        if abs(error) < zona_muerta_efectiva:
            return self.orientacion, self.fatiga.get_historia(), self.fatiga.get_fatiga(), zona_muerta_efectiva
        
        # Dirección: viene del error
        direccion = np.sign(error)
        
        # Confianza: viene del gradiente
        confianza = min(1.0, abs(gradiente) * self.sensibilidad_grad)
        
        # Freno exponencial
        factor_freno = self.calcular_factor_freno(error)
        
        # Kp EFECTIVO: aplica factor_gain de la fatiga_activa
        Kp_efectivo = self.Kp_actual * factor_gain * confianza
        
        # Asegurar que no baja demasiado (suelo metabólico)
        Kp_efectivo = max(self.Kp_min, Kp_efectivo)
        
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
        
        return (self.orientacion, self.fatiga.get_historia(), 
                self.fatiga.get_fatiga(), zona_muerta_efectiva)
    
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
# SISTEMA V149
# ============================================================

class SistemaV149:
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
        
        self.izquierdo = HemisferioV149("L", 30.0, generar_ruido_rosa, seed=seed, sesgo=SESGO_L)
        self.derecho = HemisferioV149("R", 300.0, generar_clicks_poisson, seed=seed+100, sesgo=SESGO_R)
        
        self.sistema_B_izq = HemisferioV149("B_L", 30.0, generar_ruido_rosa, seed=seed+200, sesgo=SESGO_L)
        self.sistema_B_der = HemisferioV149("B_R", 300.0, generar_clicks_poisson, seed=seed+300, sesgo=SESGO_R)
        
        self.modo_entrenamiento = True
        self.motor = AparatoMotorV149()
        
        self.historial = {
            't': [],
            'orientacion': [],
            'setpoint_real': [],
            'gradiente': [],
            'historia': [],
            'fatiga': [],
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
        orientacion, historia, fatiga, zona_muerta = self.motor.actuar(
            gradiente, LF_activa, fuente_activa, t, setpoint_real
        )
        
        self.historial['t'].append(t)
        self.historial['orientacion'].append(orientacion)
        self.historial['setpoint_real'].append(setpoint_real)
        self.historial['gradiente'].append(gradiente)
        self.historial['historia'].append(historia)
        self.historial['fatiga'].append(fatiga)
        self.historial['zona_muerta'].append(zona_muerta)
        self.historial['s_shared'].append(self.calcular_s_shared())
        self.historial['Kp'].append(self.motor.Kp_actual)
        
        return orientacion, historia, fatiga
    
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
# ANALISIS DE SEMICICLO
# ============================================================

def analizar_semiciclo(orientaciones, setpoints, dt=DT, umbral_error=2.0, ventana=50):
    """Analiza un semiciclo completo (hasta 40s) buscando T_settle y error final"""
    if len(orientaciones) == 0:
        return None, None, None, None
    
    # Limitar a los primeros 40 segundos (primer semiciclo)
    fin = min(len(orientaciones), int(40.0 / dt))
    orient_ciclo = orientaciones[:fin]
    setpoint_ciclo = setpoints[:fin]
    
    errores = np.abs(np.array(orient_ciclo) - np.array(setpoint_ciclo))
    
    # T_settle: primer momento donde error < umbral_error por ventana pasos
    t_settle = None
    for i in range(len(errores) - ventana):
        if all(errores[i:i+ventana] < umbral_error):
            t_settle = i * dt
            break
    
    # Error final: promedio últimos 0.5 segundos del semiciclo
    if len(errores) > ventana:
        error_final = np.mean(errores[-ventana:])
    else:
        error_final = errores[-1] if len(errores) > 0 else None
    
    # Amplitud alcanzada en este semiciclo
    amplitud = max(orient_ciclo) if setpoint_ciclo[-1] > 0 else abs(min(orient_ciclo))
    
    return t_settle, error_final, amplitud, None


# ============================================================
# EXPERIMENTO V149
# ============================================================

def ejecutar_v149():
    print("=" * 100)
    print("EXPERIMENTO V149 — Separación historia/fatiga activa")
    print("=" * 100)
    print("  Correcciones sobre V148:")
    print("    - Historia: registro permanente (nunca decae)")
    print("    - Fatiga activa: estado recuperable (decae con reposo)")
    print("    - Solo la fatiga activa afecta el rendimiento")
    print("")
    print("  Parámetros fatiga:")
    print(f"    - K_GAIN: {K_GAIN}")
    print(f"    - K_PRECISION: {K_PRECISION}")
    print(f"    - K_TEMBLOR: {K_TEMBLOR}")
    print(f"    - TAU_RECUPERACION: {TAU_RECUPERACION}s")
    print("=" * 100)
    
    sistema = SistemaV149("V149", seed=SEMILLA_BASE)
    
    print("\n  Entrenando lateralidad (10 repeticiones)...")
    sistema.set_modo_entrenamiento(True)
    
    for rep in range(REPETICIONES_LENTAS):
        for i in range(int(TIEMPO_POR_REPETICION / DT)):
            t = rep * TIEMPO_POR_REPETICION + i * DT
            sistema.actualizar(t, DT, TIEMPO_POR_REPETICION * REPETICIONES_LENTAS,
                              setpoint_real=0.0)
    
    print("  Entrenamiento completado.")
    print("\n  Iniciando test de fatiga con separación historia/fatiga...")
    
    sistema.set_modo_entrenamiento(False)
    sistema.motor.reset()
    t_actual = TIEMPO_POR_REPETICION * REPETICIONES_LENTAS
    
    # Función para ejecutar ciclos con reporte
    def ejecutar_ciclos(sistema, t_actual, num_ciclos, nombre_fase, verbose=True):
        tiempos = []
        orientaciones = []
        setpoints = []
        historias = []
        fatigas = []
        zonas = []
        
        for ciclo in range(num_ciclos):
            for i in range(int(PERIODO_ALTERNANCIA / DT)):
                t = t_actual + (ciclo * PERIODO_ALTERNANCIA + i) * DT
                t_rel = i * DT
                
                setpoint = onda_cuadrada(t_rel, periodo=PERIODO_ALTERNANCIA, amplitud=60.0)
                orient, historia, fatiga = sistema.actualizar(t, DT, t_actual + 1000, setpoint)
                
                tiempos.append(t)
                orientaciones.append(orient)
                setpoints.append(setpoint)
                historias.append(historia)
                fatigas.append(fatiga)
                zonas.append(sistema.historial['zona_muerta'][-1])
            
            if verbose and (ciclo + 1) % 10 == 0:
                print(f"      Ciclo {ciclo + 1}/{num_ciclos} completado, "
                      f"fatiga={fatiga:.0f}°, historia={historia:.0f}°")
        
        return t_actual + num_ciclos * PERIODO_ALTERNANCIA, tiempos, orientaciones, setpoints, historias, fatigas, zonas
    
    # F1: Baseline fresco (3 ciclos)
    print("\n  F1: Baseline fresco (3 ciclos)...")
    t_actual, t1, o1, s1, h1, f1, z1 = ejecutar_ciclos(sistema, t_actual, 3, "F1", verbose=True)
    
    # F2: Fatiga inducida (50 ciclos)
    print("\n  F2: Fatiga inducida (50 ciclos)...")
    t_actual, t2, o2, s2, h2, f2, z2 = ejecutar_ciclos(sistema, t_actual, 50, "F2", verbose=True)
    
    # F3: Test fatigado (3 ciclos)
    print("\n  F3: Test fatigado (3 ciclos)...")
    t_actual, t3, o3, s3, h3, f3, z3 = ejecutar_ciclos(sistema, t_actual, 3, "F3", verbose=True)
    
    # F4: Recuperacion (TAU_RECUPERACION segundos)
    print(f"\n  F4: Recuperacion ({TAU_RECUPERACION}s reposo)...")
    for i in range(int(TAU_RECUPERACION / DT)):
        t = t_actual + i * DT
        orient, historia, fatiga = sistema.actualizar(t, DT, t_actual + TAU_RECUPERACION, 0.0)
    t_actual += TAU_RECUPERACION
    
    # F5: Test post-recuperacion (3 ciclos)
    print("\n  F5: Test post-recuperacion (3 ciclos)...")
    t_actual, t5, o5, s5, h5, f5, z5 = ejecutar_ciclos(sistema, t_actual, 3, "F5", verbose=True)
    
    # ============================================================
    # ANALISIS
    # ============================================================
    print("\n" + "=" * 80)
    print("ANALISIS DE FATIGA (separación historia/fatiga)")
    print("=" * 80)
    
    # Analizar primer semiciclo de cada fase (primeros 40s)
    fin = int(40.0 / DT)
    
    # F1
    t_settle_f1, error_f1, amp_f1, _ = analizar_semiciclo(o1[:fin], s1[:fin])
    fatiga_f1 = f1[-1] if f1 else 0
    historia_f1 = h1[-1] if h1 else 0
    
    # F3
    t_settle_f3, error_f3, amp_f3, _ = analizar_semiciclo(o3[:fin], s3[:fin])
    fatiga_f3 = f3[-1] if f3 else 0
    historia_f3 = h3[-1] if h3 else 0
    
    # F5
    t_settle_f5, error_f5, amp_f5, _ = analizar_semiciclo(o5[:fin], s5[:fin])
    fatiga_f5 = f5[-1] if f5 else 0
    historia_f5 = h5[-1] if h5 else 0
    
    print(f"\n  F1 - Baseline fresco:")
    print(f"    T_settle: {t_settle_f1:.1f}s" if t_settle_f1 else "    T_settle: No alcanzado")
    print(f"    Error final: {error_f1:.1f}°" if error_f1 else "    Error final: N/A")
    print(f"    Amplitud alcanzada: {amp_f1:.1f}°")
    print(f"    Fatiga activa: {fatiga_f1:.0f}°")
    print(f"    Historia acumulada: {historia_f1:.0f}°")
    
    print(f"\n  F3 - Fatigado (despues de 50 ciclos):")
    print(f"    T_settle: {t_settle_f3:.1f}s" if t_settle_f3 else "    T_settle: No alcanzado")
    print(f"    Error final: {error_f3:.1f}°" if error_f3 else "    Error final: N/A")
    print(f"    Amplitud alcanzada: {amp_f3:.1f}°")
    print(f"    Fatiga activa: {fatiga_f3:.0f}°")
    print(f"    Historia acumulada: {historia_f3:.0f}°")
    
    print(f"\n  F5 - Post-recuperacion ({TAU_RECUPERACION}s reposo):")
    print(f"    T_settle: {t_settle_f5:.1f}s" if t_settle_f5 else "    T_settle: No alcanzado")
    print(f"    Error final: {error_f5:.1f}°" if error_f5 else "    Error final: N/A")
    print(f"    Amplitud alcanzada: {amp_f5:.1f}°")
    print(f"    Fatiga activa: {fatiga_f5:.0f}°")
    print(f"    Historia acumulada: {historia_f5:.0f}°")
    
    # Degradacion y recuperacion
    if t_settle_f1 and t_settle_f3:
        degradacion = t_settle_f3 / t_settle_f1
        print(f"\n  Degradacion por fatiga: {degradacion:.2f}x")
    else:
        degradacion = None
        print(f"\n  Degradacion por fatiga: No calculable (T_settle faltante)")
    
    if error_f1 and error_f3:
        degradacion_error = error_f3 / error_f1
        print(f"  Degradacion error: {degradacion_error:.2f}x")
    
    if fatiga_f1 and fatiga_f3:
        recuperacion_fatiga = (fatiga_f3 - fatiga_f5) / (fatiga_f3 - fatiga_f1) * 100 if fatiga_f3 != fatiga_f1 else 0
        print(f"  Recuperacion fatiga: {recuperacion_fatiga:.1f}%")
    
    exito_fatiga = error_f3 and error_f1 and (error_f3 / error_f1) > 2.0
    exito_recuperacion = fatiga_f5 and fatiga_f3 and fatiga_f5 < fatiga_f3 * 0.5
    
    print("\n" + "=" * 80)
    print("CONCLUSION V149")
    print("=" * 80)
    
    if exito_fatiga and exito_recuperacion:
        print("\n  ✅ O-N11 VALIDADA")
        print("     La fatiga activa degrada el rendimiento")
        print("     El reposo recupera parcialmente la fatiga")
        print("\n  ANIMA-2 - Linea 3: CERRADA")
    elif exito_fatiga:
        print("\n  ✅ FATIGA DEMOSTRADA")
        print(f"     Error: {error_f1:.1f}° → {error_f3:.1f}° ({degradacion_error:.1f}x)")
        print("  ⚠️ RECUPERACION PENDIENTE")
    else:
        print("\n  ⚠️ O-N11 NO VALIDADA")
    
    # Graficos
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Grafico 1: Comparativa de orientacion F1 vs F3
    ax = axes[0, 0]
    ax.plot(s1[:fin], 'r--', linewidth=0.8, alpha=0.5, label='Setpoint')
    ax.plot(o1[:fin], 'b-', linewidth=0.6, label='F1 (fresco)')
    ax.plot(o3[:fin], 'orange', linewidth=0.6, label='F3 (fatigado)')
    ax.set_xlabel('Paso')
    ax.set_ylabel('Angulo (grados)')
    ax.set_title('Comparativa: Fresco vs Fatigado')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Grafico 2: Historia vs Fatiga activa
    ax = axes[0, 1]
    ax.plot(f1, 'b-', linewidth=0.6, label='Fatiga activa F1')
    ax.plot(f3, 'orange', linewidth=0.6, label='Fatiga activa F3')
    ax.plot(f5, 'green', linewidth=0.6, label='Fatiga activa F5')
    ax.set_xlabel('Paso')
    ax.set_ylabel('Fatiga activa (grados)')
    ax.set_title('Evolucion de la fatiga activa')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Grafico 3: Zona muerta efectiva
    ax = axes[1, 0]
    ax.plot(z1, 'b-', linewidth=0.6, label='Zona muerta F1')
    ax.plot(z3, 'orange', linewidth=0.6, label='Zona muerta F3')
    ax.plot(z5, 'green', linewidth=0.6, label='Zona muerta F5')
    ax.axhline(y=ZONA_MUERTA_BASE, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Paso')
    ax.set_ylabel('Zona muerta (grados)')
    ax.set_title('Expansion de zona muerta por fatiga')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Grafico 4: Error final comparativo
    ax = axes[1, 2]
    fases = ['F1 (fresco)', 'F3 (fatigado)', 'F5 (post)']
    errores = [error_f1 or 0, error_f3 or 0, error_f5 or 0]
    colores = ['green', 'red', 'blue']
    bars = ax.bar(fases, errores, color=colores, alpha=0.7)
    for bar, val in zip(bars, errores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.1f}°', ha='center', va='bottom', fontsize=10)
    ax.axhline(y=ZONA_MUERTA_BASE, color='red', linestyle='--', alpha=0.5, label='Zona muerta base')
    ax.set_ylabel('Error final (grados)')
    ax.set_title('Degradacion del error por fatiga')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Ocultar subplot vacío
    axes[1, 1].set_visible(False)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('v149_logs', exist_ok=True)
    plt.savefig(f'v149_logs/v149_fatiga_separada_{timestamp}.png', dpi=150)
    print(f"\n  Graficos guardados: v149_logs/v149_fatiga_separada_{timestamp}.png")
    
    return sistema, exito_fatiga and exito_recuperacion


if __name__ == "__main__":
    import time
    start = time.time()
    sistema, exito = ejecutar_v149()
    elapsed = time.time() - start
    print(f"\n  Tiempo de ejecucion: {elapsed/60:.1f} minutos")