#!/usr/bin/env python3
"""
VSTCosmos V141 — Fatiga real (3 mecanismos acoplados)

Correcciones sobre V140:
  1. Acumulador de energía global (nunca se reinicia)
  2. Degradación de Kp con exponencial (factor_gain = exp(-k_gain * E))
  3. Zona muerta crece con fatiga (zona_muerta += k_precision * E)
  4. Temblor aumenta con fatiga (ruido += k_temblor * E)
  5. Recuperación exponencial durante reposo

Hipotesis O-N11:
  - T_settle_fatigado / T_settle_fresco > 1.5
  - Recuperación > 30%
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

# Zona muerta base
ZONA_MUERTA_BASE = 2.0

# Limites de plasticidad
KP_BASE = 0.002
KP_MIN = 0.0005
KP_MAX = 0.005

# Plasticidad
HABITUACION_RAPIDA = 0.99
SENSIBILIZACION_LENTA = 1.01
VENTANA_OSCILACION = 100

# Fatiga V141
K_GAIN = 0.00008      # Kp_eff = Kp * exp(-K_GAIN * E)
K_PRECISION = 0.0005  # zona_muerta += K_PRECISION * E
K_TEMBLOR = 0.0003    # ruido_motor += K_TEMBLOR * E
TAU_RECUPERACION = 120.0  # segundos (constante de tiempo)

# Semilla base
SEMILLA_BASE = 44


# ============================================================
# HEMISFERIO
# ============================================================

class HemisferioV141:
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
# FATIGA METABOLICA REAL (V141)
# ============================================================

class FatigaMetabolicaReal:
    """
    Fatiga que afecta tres aspectos del motor:
      1. Ganancia (Kp) - gira más lento
      2. Precisión (zona muerta) - menos preciso
      3. Temblor - más ruido motor
    
    La energía se acumula GLOBALMENTE (nunca se reinicia excepto por reposo)
    """
    
    def __init__(self, k_gain=K_GAIN, k_precision=K_PRECISION, 
                 k_temblor=K_TEMBLOR, tau_recuperacion=TAU_RECUPERACION):
        self.k_gain = k_gain
        self.k_precision = k_precision
        self.k_temblor = k_temblor
        self.tau_recuperacion = tau_recuperacion
        
        self.energia_total = 0.0  # Acumulador GLOBAL
        self.historial_energia = []
        self.historial_factor_gain = []
        self.historial_zona_muerta = []
        self.historial_temblor = []
    
    def actualizar(self, delta_orientacion, en_reposo, dt):
        """
        Actualiza energía y calcula efectos de fatiga.
        
        Args:
            delta_orientacion: cambio de orientación en este paso (grados)
            en_reposo: True si el organismo está en reposo
            dt: paso de tiempo
        
        Returns:
            factor_gain: multiplicador de Kp (1.0 = sano, <1.0 = fatigado)
            zona_muerta_efectiva: zona muerta aumentada por fatiga
            temblor: ruido aditivo al motor
        """
        # Actualizar energía acumulada
        if not en_reposo:
            # Acumula SIEMPRE que hay movimiento (sin umbral)
            self.energia_total += abs(delta_orientacion)
        else:
            # Recuperación exponencial durante reposo
            self.energia_total *= np.exp(-dt / self.tau_recuperacion)
        
        # Calcular efectos de fatiga
        factor_gain = np.exp(-self.k_gain * self.energia_total)
        zona_muerta_efectiva = ZONA_MUERTA_BASE + self.k_precision * self.energia_total
        temblor = self.k_temblor * self.energia_total * np.random.randn()
        
        # Limitar valores extremos
        factor_gain = max(0.2, min(1.0, factor_gain))
        zona_muerta_efectiva = min(15.0, zona_muerta_efectiva)
        temblor = np.clip(temblor, -5.0, 5.0)
        
        # Guardar historial
        self.historial_energia.append(self.energia_total)
        self.historial_factor_gain.append(factor_gain)
        self.historial_zona_muerta.append(zona_muerta_efectiva)
        self.historial_temblor.append(temblor)
        
        return factor_gain, zona_muerta_efectiva, temblor
    
    def reset(self):
        """Reinicia el acumulador (nuevo experimento)"""
        self.energia_total = 0.0
        self.historial_energia = []
        self.historial_factor_gain = []
        self.historial_zona_muerta = []
        self.historial_temblor = []
    
    def get_energia(self):
        return self.energia_total


# ============================================================
# APARATO MOTOR CON FATIGA REAL (V141)
# ============================================================

class AparatoMotorConFatigaReal:
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
        
        # Fatiga (NUEVO V141)
        self.fatiga = FatigaMetabolicaReal()
        
        # Plasticidad
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
        
        # Setpoint objetivo
        setpoint_objetivo = setpoint_percepcion if fuente_activa else 0.0
        
        error = setpoint_objetivo - self.orientacion
        
        # Actualizar fatiga y obtener efectos
        factor_gain, zona_muerta_efectiva, temblor = self.fatiga.actualizar(
            self.ultimo_delta_registrado, not fuente_activa, DT
        )
        
        # Zona muerta aumentada por fatiga
        if abs(error) < zona_muerta_efectiva:
            return self.orientacion, self.fatiga.get_energia(), zona_muerta_efectiva
        
        # Control proporcional con freno
        ganancia_grad = -np.tanh(gradiente * self.sensibilidad_grad)
        factor_freno = self.calcular_factor_freno(error)
        
        # Kp reducido por fatiga
        Kp_efectivo = self.Kp_actual * factor_gain
        
        delta = Kp_efectivo * error * ganancia_grad * factor_freno
        
        # Inercia del motor (suavizado)
        delta = self.inercia * self.ultimo_delta + (1 - self.inercia) * delta
        self.ultimo_delta = delta
        self.ultimo_delta_registrado = delta
        
        # Añadir temblor por fatiga
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
# SISTEMA V141
# ============================================================

class SistemaV141:
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
        
        self.izquierdo = HemisferioV141("L", 30.0, generar_ruido_rosa, seed=seed, sesgo=SESGO_L)
        self.derecho = HemisferioV141("R", 300.0, generar_clicks_poisson, seed=seed+100, sesgo=SESGO_R)
        
        self.sistema_B_izq = HemisferioV141("B_L", 30.0, generar_ruido_rosa, seed=seed+200, sesgo=SESGO_L)
        self.sistema_B_der = HemisferioV141("B_R", 300.0, generar_clicks_poisson, seed=seed+300, sesgo=SESGO_R)
        
        self.modo_entrenamiento = True
        self.motor = AparatoMotorConFatigaReal()
        
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
# EXPERIMENTO V141 - FATIGA REAL
# ============================================================

def ejecutar_v141():
    print("=" * 100)
    print("EXPERIMENTO V141 — Fatiga real (3 mecanismos acoplados)")
    print("=" * 100)
    print("  ANIMA-2 - Linea 3: Hipotesis O-N11 (corregida)")
    print("")
    print("  El organismo:")
    print("    - Es una esfera con orejas que rota sobre su eje central")
    print("    - Acumula 'costo metabolico' con cada giro (ENERGIA GLOBAL)")
    print("    - La fatiga afecta 3 mecanismos:")
    print("      1. Ganancia (Kp) - gira mas lento")
    print("      2. Precision (zona muerta) - menos preciso")
    print("      3. Temblor - mas ruido motor")
    print("    - Con reposo, recupera exponencialmente")
    print("")
    print("  Protocolo:")
    print("    Fase 1: Baseline fresco (10 ciclos -60° ↔ +60°)")
    print("    Fase 2: Fatiga inducida (50 ciclos -60° ↔ +60°)")
    print("    Fase 3: Test fatiga (10 ciclos, medir degradacion)")
    print("    Fase 4: Recuperacion (60s reposo)")
    print("    Fase 5: Test post-recuperacion (10 ciclos)")
    print("=" * 100)
    
    # Configurar sistema
    sistema = SistemaV141("V141", seed=SEMILLA_BASE)
    
    # Entrenamiento lateral
    print("\n  Entrenando lateralidad (10 repeticiones)...")
    sistema.set_modo_entrenamiento(True)
    
    for rep in range(REPETICIONES_LENTAS):
        for i in range(int(TIEMPO_POR_REPETICION / DT)):
            t = rep * TIEMPO_POR_REPETICION + i * DT
            sistema.actualizar(t, DT, TIEMPO_POR_REPETICION * REPETICIONES_LENTAS,
                              setpoint_real=0.0)
    
    print("  Entrenamiento completado.")
    print("  Iniciando test de fatiga real...")
    
    sistema.set_modo_entrenamiento(False)
    sistema.motor.reset()
    
    t_actual = TIEMPO_POR_REPETICION * REPETICIONES_LENTAS
    
    # Funcion para ejecutar ciclos de giro
    def ejecutar_ciclos(sistema, t_actual, num_ciclos, nombre_fase, verbose=False):
        if verbose:
            print(f"  {nombre_fase}: {num_ciclos} ciclos -60° ↔ +60°")
        
        tiempos = []
        orientaciones = []
        setpoints = []
        energias = []
        zonas_muertas = []
        
        for ciclo in range(num_ciclos):
            # Giro a izquierda (-60°)
            for i in range(int(60.0 / DT)):
                t = t_actual + (ciclo * 120 + i) * DT
                t_rel = i * DT
                
                # Setpoint: -60° durante 60s, luego +60°
                if t_rel < 60:
                    setpoint = -60.0
                else:
                    setpoint = 60.0
                
                orient, energia = sistema.actualizar(t, DT, t_actual + 1000, setpoint)
                
                if i % 1000 == 0 and verbose and ciclo % 5 == 0:
                    print(f"      Ciclo {ciclo}, t={t_rel:.0f}s: orient={orient:.1f}°, energia={energia:.0f}°")
                
                tiempos.append(t)
                orientaciones.append(orient)
                setpoints.append(setpoint)
                energias.append(energia)
                zonas_muertas.append(sistema.historial['zona_muerta'][-1])
            
            t_actual += 120.0
        
        return t_actual, tiempos, orientaciones, setpoints, energias, zonas_muertas
    
    # Fase 1: Baseline fresco (10 ciclos)
    t_actual, t1, o1, s1, e1, z1 = ejecutar_ciclos(sistema, t_actual, 10, "Fase 1: Baseline fresco", verbose=True)
    
    # Fase 2: Fatiga inducida (50 ciclos) - silencioso para no saturar
    t_actual, t2, o2, s2, e2, z2 = ejecutar_ciclos(sistema, t_actual, 50, "Fase 2: Fatiga inducida", verbose=False)
    
    # Fase 3: Test fatiga (10 ciclos)
    t_actual, t3, o3, s3, e3, z3 = ejecutar_ciclos(sistema, t_actual, 10, "Fase 3: Test fatiga", verbose=True)
    
    # Fase 4: Recuperacion (60s reposo)
    print("\n  Fase 4: Recuperacion (60s reposo)...")
    for i in range(int(60.0 / DT)):
        t = t_actual + i * DT
        orient, energia = sistema.actualizar(t, DT, t_actual + 60, 0.0)
    t_actual += 60.0
    
    # Fase 5: Test post-recuperacion (10 ciclos)
    t_actual, t5, o5, s5, e5, z5 = ejecutar_ciclos(sistema, t_actual, 10, "Fase 5: Post-recuperacion", verbose=True)
    
    # ============================================================
    # ANALISIS
    # ============================================================
    print("\n" + "=" * 80)
    print("ANALISIS DE FATIGA REAL")
    print("=" * 80)
    
    def analizar_fase(orientaciones, setpoints, energias):
        errores = np.abs(np.array(orientaciones) - np.array(setpoints))
        
        # Encontrar T_settle (entrada estable en zona muerta)
        zona_muerta_actual = ZONA_MUERTA_BASE
        t_settle = None
        
        for i in range(len(errores)):
            if errores[i] < zona_muerta_actual:
                # Verificar que se mantiene por 100 pasos (1 segundo)
                if i + 100 < len(errores):
                    if all(errores[i:i+100] < zona_muerta_actual + 1.0):
                        t_settle = i * DT
                        break
        
        # Error final
        error_final = errores[-1] if len(errores) > 0 else None
        
        # Energia acumulada
        energia_final = energias[-1] if len(energias) > 0 else 0
        
        # Velocidad media (como proxy de fatiga)
        if len(orientaciones) > 1:
            diffs = np.abs(np.diff(orientaciones))
            velocidad_media = np.mean(diffs) / DT
        else:
            velocidad_media = 0
        
        return {
            't_settle': t_settle,
            'error_final': error_final,
            'energia': energia_final,
            'velocidad_media': velocidad_media
        }
    
    # Analizar cada fase
    fresco = analizar_fase(o1, s1, e1)
    fatigado = analizar_fase(o3, s3, e3)
    recuperado = analizar_fase(o5, s5, e5)
    
    print(f"\n  Fase 1 - Baseline fresco:")
    print(f"    T_settle: {fresco['t_settle']:.1f}s" if fresco['t_settle'] else "    T_settle: No alcanzado")
    print(f"    Error final: {fresco['error_final']:.2f}°" if fresco['error_final'] else "    Error final: N/A")
    print(f"    Energia acumulada: {fresco['energia']:.0f}°")
    print(f"    Velocidad media: {fresco['velocidad_media']:.2f}°/s")
    
    print(f"\n  Fase 3 - Fatigado (despues de 50 ciclos):")
    print(f"    T_settle: {fatigado['t_settle']:.1f}s" if fatigado['t_settle'] else "    T_settle: No alcanzado")
    print(f"    Error final: {fatigado['error_final']:.2f}°" if fatigado['error_final'] else "    Error final: N/A")
    print(f"    Energia acumulada: {fatigado['energia']:.0f}°")
    print(f"    Velocidad media: {fatigado['velocidad_media']:.2f}°/s")
    
    print(f"\n  Fase 5 - Post-recuperacion (60s reposo):")
    print(f"    T_settle: {recuperado['t_settle']:.1f}s" if recuperado['t_settle'] else "    T_settle: No alcanzado")
    print(f"    Error final: {recuperado['error_final']:.2f}°" if recuperado['error_final'] else "    Error final: N/A")
    print(f"    Energia acumulada: {recuperado['energia']:.0f}°")
    print(f"    Velocidad media: {recuperado['velocidad_media']:.2f}°/s")
    
    # Calcular degradacion y recuperacion
    if fresco['t_settle'] and fatigado['t_settle']:
        degradacion = fatigado['t_settle'] / fresco['t_settle']
        print(f"\n  Degradacion por fatiga: {degradacion:.2f}x (objetivo >1.5x)")
    else:
        degradacion = None
        print(f"\n  Degradacion por fatiga: No calculable (T_settle faltante)")
    
    if fresco['t_settle'] and fatigado['t_settle'] and recuperado['t_settle']:
        recuperacion = (fatigado['t_settle'] - recuperado['t_settle']) / (fatigado['t_settle'] - fresco['t_settle']) * 100
        print(f"  Recuperacion post-reposo: {recuperacion:.1f}% (objetivo >30%)")
    else:
        recuperacion = None
    
    # Criterios O-N11
    exito_fatiga = degradacion and degradacion > 1.5
    exito_recuperacion = recuperacion and recuperacion > 30
    
    print("\n" + "=" * 80)
    print("CONCLUSION V141 — Fatiga real")
    print("=" * 80)
    
    if exito_fatiga:
        print("\n  ✅ FATIGA DEMOSTRADA: El organismo se cansa con el uso")
        print(f"     T_settle aumento {degradacion:.2f}x (>1.5x)")
    else:
        print(f"\n  ⚠️ FATIGA NO DEMOSTRADA: Degradacion insuficiente")
    
    if exito_recuperacion:
        print("\n  ✅ RECUPERACION PARCIAL: El reposo restaura funcion")
        print(f"     Recuperacion del {recuperacion:.1f}% (>30%)")
    else:
        print("\n  ⚠️ RECUPERACION INSUFICIENTE")
    
    exito_total = exito_fatiga and exito_recuperacion
    
    # Graficos
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Grafico 1: Energia acumulada global
    ax = axes[0, 0]
    todas_energias = e1 + e2 + e3 + [e3[-1]]*6000 + e5 if e3 else e1 + e2 + e5
    ax.plot(todas_energias, 'r-', linewidth=0.8)
    ax.set_xlabel('Paso')
    ax.set_ylabel('Energia acumulada (grados)')
    ax.set_title('Energia metabolica global')
    ax.grid(True, alpha=0.3)
    
    # Grafico 2: T_settle por fase
    ax = axes[0, 1]
    fases = ['Fresco', 'Fatigado', 'Recuperado']
    t_settle_vals = [
        fresco['t_settle'] or 0,
        fatigado['t_settle'] or 0,
        recuperado['t_settle'] or 0
    ]
    colores = ['green', 'red', 'blue']
    bars = ax.bar(fases, t_settle_vals, color=colores, alpha=0.7)
    for bar, val in zip(bars, t_settle_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}s', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('T_settle (segundos)')
    ax.set_title('Degradacion por fatiga')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Grafico 3: Velocidad media por fase
    ax = axes[1, 0]
    vel_vals = [
        fresco['velocidad_media'],
        fatigado['velocidad_media'],
        recuperado['velocidad_media']
    ]
    bars = ax.bar(fases, vel_vals, color=colores, alpha=0.7)
    for bar, val in zip(bars, vel_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{val:.1f}°/s', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('Velocidad media (grados/s)')
    ax.set_title('Velocidad de giro (proxy de fatiga)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Grafico 4: Zona muerta efectiva durante fatiga
    ax = axes[1, 1]
    if len(z3) > 0:
        ax.plot(z3[:5000], 'orange', linewidth=0.8, label='Fatigado')
    if len(z1) > 0:
        ax.plot(z1[:5000], 'green', linewidth=0.8, label='Fresco')
    ax.set_xlabel('Paso')
    ax.set_ylabel('Zona muerta (grados)')
    ax.set_title('Zona muerta efectiva por fatiga')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('v141_logs', exist_ok=True)
    plt.savefig(f'v141_logs/v141_fatiga_real_{timestamp}.png', dpi=150)
    print(f"\n  Graficos guardados: v141_logs/v141_fatiga_real_{timestamp}.png")
    
    return sistema, exito_total


if __name__ == "__main__":
    import time
    start = time.time()
    sistema, exito = ejecutar_v141()
    elapsed = time.time() - start
    print(f"\n  Tiempo de ejecucion: {elapsed/60:.1f} minutos")