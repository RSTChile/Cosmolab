#!/usr/bin/env python3
"""
VSTCosmos V140 — Fatiga: La esfera con orejas se cansa

ANIMA-2 - Linea 3: Hipotesis O-N11
  El organismo acumula "costo energetico" con cada giro.
  Cuando la energia acumulada supera un umbral, el rendimiento se degrada.
  Con reposo, recupera parcialmente.

El organismo:
  - Es una esfera con orejas que rota sobre su eje central
  - No tiene inercia significativa (masa concentrada en el centro)
  - El motor aplica torque, pero la fatiga es metabolica, no inercial
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

# Limites de plasticidad (motor sano)
KP_BASE = 0.002
KP_MIN = 0.0005
KP_MAX = 0.005

# Plasticidad
HABITUACION_RAPIDA = 0.99
SENSIBILIZACION_LENTA = 1.01
VENTANA_OSCILACION = 100

# Fatiga (NUEVO V140)
ENERGIA_MAX = 5000.0      # grados acumulados hasta fatiga severa
FACTOR_FATIGA = 0.5       # cuánto reduce Kp cuando E_acumulada = ENERGIA_MAX
TASA_RECUPERACION = 0.01  # 1% de recuperacion por segundo en reposo

# Semilla base
SEMILLA_BASE = 44


# ============================================================
# HEMISFERIO
# ============================================================

class HemisferioV140:
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
# FATIGA METABOLICA (NUEVO V140)
# ============================================================

class FatigaMetabolica:
    """
    Acumula costo energetico de los movimientos.
    Cuando la energia acumulada es alta, reduce la ganancia efectiva.
    En reposo, recupera gradualmente.
    
    La esfera con orejas no tiene inercia, pero tiene "metabolismo":
    cada giro consume energia, y despues de muchos giros se cansa.
    """
    
    def __init__(self, energia_max=ENERGIA_MAX, factor_fatiga=FACTOR_FATIGA,
                 tasa_recuperacion=TASA_RECUPERACION):
        self.energia_max = energia_max
        self.factor_fatiga = factor_fatiga
        self.tasa_recuperacion = tasa_recuperacion
        self.energia_acumulada = 0.0
        self.historial_energia = []
        self.historial_factor = []
    
    def actualizar(self, delta_orientacion, en_reposo, dt):
        """
        Actualiza la energia acumulada y calcula factor de fatiga.
        
        Args:
            delta_orientacion: cambio de orientacion en este paso (grados)
            en_reposo: True si el organismo esta en reposo (sin estimulo)
            dt: paso de tiempo
        
        Returns:
            factor_fatiga: multiplicador de Kp (1.0 = sano, <1.0 = fatigado)
        """
        if not en_reposo and abs(delta_orientacion) > 0.01:
            # Movimiento: acumula energia
            self.energia_acumulada += abs(delta_orientacion)
        else:
            # Reposo: recupera gradualmente
            self.energia_acumulada *= max(0.0, 1.0 - self.tasa_recuperacion * dt)
        
        # Limitar energia maxima
        self.energia_acumulada = min(self.energia_acumulada, self.energia_max * 2.0)
        
        # Calcular factor de fatiga (reduce Kp cuando energia es alta)
        if self.energia_acumulada < self.energia_max:
            factor = 1.0 - (self.energia_acumulada / self.energia_max) * self.factor_fatiga
        else:
            factor = 1.0 - self.factor_fatiga
        
        factor = max(0.3, min(1.0, factor))
        
        self.historial_energia.append(self.energia_acumulada)
        self.historial_factor.append(factor)
        
        return factor
    
    def reset(self):
        self.energia_acumulada = 0.0
        self.historial_energia = []
        self.historial_factor = []
    
    def get_energia(self):
        return self.energia_acumulada


# ============================================================
# APARATO MOTOR CON FATIGA (V140)
# ============================================================

class AparatoMotorConFatiga:
    """
    Motor cinemático (sin inercia) que se fatiga con el uso.
    La esfera gira sobre su eje sin inercia significativa,
    pero acumula "costo metabolico" que degrada el rendimiento.
    """
    
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
        
        # Memoria episodica
        self.memoria = None  # Simplificado para V140
        
        # Fatiga (NUEVO)
        self.fatiga = FatigaMetabolica()
        
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
        if oscilacion > self.zona_muerta * 1.5:
            self.Kp_actual = max(self.Kp_min, self.Kp_actual * 0.99)
        elif oscilacion < self.zona_muerta * 0.5:
            self.Kp_actual = min(self.Kp_max, self.Kp_actual * 1.01)
        
        self.historial_Kp.append(self.Kp_actual)
    
    def actuar(self, gradiente, LF_activa, fuente_activa, t, setpoint_percepcion):
        if not LF_activa:
            return self.orientacion, 0.0
        
        if abs(gradiente) < 0.05:
            return self.orientacion, self.fatiga.get_energia()
        
        # Setpoint objetivo (sin memoria en V140 para simplificar)
        setpoint_objetivo = setpoint_percepcion if fuente_activa else 0.0
        
        error = setpoint_objetivo - self.orientacion
        
        if abs(error) < self.zona_muerta:
            return self.orientacion, self.fatiga.get_energia()
        
        # Control proporcional con freno
        ganancia_grad = -np.tanh(gradiente * self.sensibilidad_grad)
        factor_freno = self.calcular_factor_freno(error)
        
        # Aplicar fatiga: reduce Kp efectivo
        factor_fatiga = self.fatiga.actualizar(self.ultimo_delta_registrado, 
                                                not fuente_activa, DT)
        
        Kp_efectivo = self.Kp_actual * factor_fatiga
        
        delta = Kp_efectivo * error * ganancia_grad * factor_freno
        
        # Inercia del motor (suavizado, no inercia fisica)
        delta = self.inercia * self.ultimo_delta + (1 - self.inercia) * delta
        self.ultimo_delta = delta
        self.ultimo_delta_registrado = delta
        
        self.actualizar_plasticidad(error)
        
        self.orientacion += delta
        self.orientacion = np.clip(self.orientacion, -self.limite, self.limite)
        
        self.historial_fatiga.append(factor_fatiga)
        self.t += DT
        
        return self.orientacion, self.fatiga.get_energia()
    
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
# SISTEMA V140
# ============================================================

class SistemaV140:
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
        
        self.izquierdo = HemisferioV140("L", 30.0, generar_ruido_rosa, seed=seed, sesgo=SESGO_L)
        self.derecho = HemisferioV140("R", 300.0, generar_clicks_poisson, seed=seed+100, sesgo=SESGO_R)
        
        self.sistema_B_izq = HemisferioV140("B_L", 30.0, generar_ruido_rosa, seed=seed+200, sesgo=SESGO_L)
        self.sistema_B_der = HemisferioV140("B_R", 300.0, generar_clicks_poisson, seed=seed+300, sesgo=SESGO_R)
        
        self.modo_entrenamiento = True
        self.motor = AparatoMotorConFatiga()
        
        self.historial = {
            't': [],
            'orientacion': [],
            'setpoint_real': [],
            'energia': [],
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
        orientacion, energia = self.motor.actuar(
            gradiente, LF_activa, fuente_activa, t, setpoint_real
        )
        
        self.historial['t'].append(t)
        self.historial['orientacion'].append(orientacion)
        self.historial['setpoint_real'].append(setpoint_real)
        self.historial['energia'].append(energia)
        self.historial['s_shared'].append(self.calcular_s_shared())
        self.historial['Kp'].append(self.motor.Kp_actual)
        
        return orientacion, energia
    
    def set_modo_entrenamiento(self, entrenamiento=True):
        self.modo_entrenamiento = entrenamiento
        if entrenamiento:
            self.motor.reset()


# ============================================================
# EXPERIMENTO V140 - FATIGA
# ============================================================

def ejecutar_v140():
    print("=" * 100)
    print("EXPERIMENTO V140 — Fatiga: La esfera con orejas se cansa")
    print("=" * 100)
    print("  ANIMA-2 - Linea 3: Hipotesis O-N11")
    print("")
    print("  El organismo:")
    print("    - Es una esfera con orejas que rota sobre su eje central")
    print("    - No tiene inercia significativa (masa concentrada en el centro)")
    print("    - Acumula 'costo metabolico' con cada giro")
    print("    - Cuando se cansa, gira mas lento y menos preciso")
    print("    - Con reposo, recupera parcialmente")
    print("")
    print("  Protocolo:")
    print("    Fase 1: Baseline fresco (10 ciclos -60° ↔ +60°)")
    print("    Fase 2: Fatiga inducida (50 ciclos -60° ↔ +60°)")
    print("    Fase 3: Test fatiga (10 ciclos, medir degradacion)")
    print("    Fase 4: Recuperacion (60s reposo)")
    print("    Fase 5: Test post-recuperacion (10 ciclos)")
    print("=" * 100)
    
    # Configurar sistema
    sistema = SistemaV140("V140", seed=SEMILLA_BASE)
    
    # Entrenamiento lateral
    print("\n  Entrenando lateralidad (10 repeticiones)...")
    sistema.set_modo_entrenamiento(True)
    
    for rep in range(REPETICIONES_LENTAS):
        for i in range(int(TIEMPO_POR_REPETICION / DT)):
            t = rep * TIEMPO_POR_REPETICION + i * DT
            sistema.actualizar(t, DT, TIEMPO_POR_REPETICION * REPETICIONES_LENTAS,
                              setpoint_real=0.0)
    
    print("  Entrenamiento completado.")
    print("  Iniciando test de fatiga...")
    
    sistema.set_modo_entrenamiento(False)
    sistema.motor.reset()
    
    t_actual = TIEMPO_POR_REPETICION * REPETICIONES_LENTAS
    
    # Funcion para ejecutar ciclos de giro
    def ejecutar_ciclos(sistema, t_actual, num_ciclos, nombre_fase):
        print(f"\n  {nombre_fase}: {num_ciclos} ciclos -60° ↔ +60°")
        
        tiempos = []
        orientaciones = []
        setpoints = []
        energias = []
        
        for ciclo in range(num_ciclos):
            # Giro a izquierda (-60°)
            for i in range(int(60.0 / DT)):  # 60 segundos para llegar
                t = t_actual + (ciclo * 120 + i) * DT
                t_rel = i * DT
                
                # Setpoint: -60° durante 60s, luego +60°
                if t_rel < 60:
                    setpoint = -60.0
                else:
                    setpoint = 60.0
                
                orient, energia = sistema.actualizar(t, DT, t_actual + 1000, setpoint)
                
                tiempos.append(t)
                orientaciones.append(orient)
                setpoints.append(setpoint)
                energias.append(energia)
            
            # Giro a derecha (+60°) ya incluido en el mismo loop
            
            t_actual += 120.0  # Avanzar 120 segundos por ciclo
        
        return t_actual, tiempos, orientaciones, setpoints, energias
    
    # Fase 1: Baseline fresco (10 ciclos)
    t_actual, t1, o1, s1, e1 = ejecutar_ciclos(sistema, t_actual, 10, "Fase 1: Baseline fresco")
    
    # Fase 2: Fatiga inducida (50 ciclos)
    t_actual, t2, o2, s2, e2 = ejecutar_ciclos(sistema, t_actual, 50, "Fase 2: Fatiga inducida")
    
    # Fase 3: Test fatiga (10 ciclos, fatigado)
    t_actual, t3, o3, s3, e3 = ejecutar_ciclos(sistema, t_actual, 10, "Fase 3: Test fatiga")
    
    # Fase 4: Recuperacion (60s reposo)
    print("\n  Fase 4: Recuperacion (60s reposo)...")
    for i in range(int(60.0 / DT)):
        t = t_actual + i * DT
        # Sin estimulo: setpoint=0
        orient, energia = sistema.actualizar(t, DT, t_actual + 60, 0.0)
    t_actual += 60.0
    
    # Fase 5: Test post-recuperacion (10 ciclos)
    t_actual, t5, o5, s5, e5 = ejecutar_ciclos(sistema, t_actual, 10, "Fase 5: Post-recuperacion")
    
    # ============================================================
    # ANALISIS
    # ============================================================
    print("\n" + "=" * 80)
    print("ANALISIS DE FATIGA")
    print("=" * 80)
    
    def analizar_ciclos(orientaciones, setpoints, energia_final=None):
        # Calcular T_settle promedio (tiempo para estabilizarse)
        # Simplificado: buscar el tiempo hasta que entra en zona muerta
        errores = np.abs(np.array(orientaciones) - np.array(setpoints))
        
        # Encontrar indices donde entra en zona muerta
        zona_muerta = ZONA_MUERTA_BASE
        indice_estable = None
        for i, err in enumerate(errores):
            if err < zona_muerta:
                # Verificar que se mantiene por 100 pasos
                if i + 100 < len(errores) and all(errores[i:i+100] < zona_muerta):
                    indice_estable = i
                    break
        
        if indice_estable:
            t_settle = indice_estable * DT
        else:
            t_settle = None
        
        # Error final
        error_final = errores[-1] if len(errores) > 0 else None
        
        # Energia acumulada final
        energia = energia_final if energia_final else 0
        
        return {
            't_settle': t_settle,
            'error_final': error_final,
            'energia': energia
        }
    
    # Analizar cada fase
    baseline = analizar_ciclos(o1, s1, e1[-1] if e1 else 0)
    fatigado = analizar_ciclos(o3, s3, e3[-1] if e3 else 0)
    recuperado = analizar_ciclos(o5, s5, e5[-1] if e5 else 0)
    
    print(f"\n  Fase 1 - Baseline fresco:")
    print(f"    T_settle: {baseline['t_settle']:.1f}s" if baseline['t_settle'] else "    T_settle: No alcanzado")
    print(f"    Error final: {baseline['error_final']:.2f}°" if baseline['error_final'] else "    Error final: N/A")
    print(f"    Energia acumulada: {baseline['energia']:.0f}°")
    
    print(f"\n  Fase 3 - Fatigado (despues de 50 ciclos):")
    print(f"    T_settle: {fatigado['t_settle']:.1f}s" if fatigado['t_settle'] else "    T_settle: No alcanzado")
    print(f"    Error final: {fatigado['error_final']:.2f}°" if fatigado['error_final'] else "    Error final: N/A")
    print(f"    Energia acumulada: {fatigado['energia']:.0f}°")
    
    print(f"\n  Fase 5 - Post-recuperacion (60s reposo):")
    print(f"    T_settle: {recuperado['t_settle']:.1f}s" if recuperado['t_settle'] else "    T_settle: No alcanzado")
    print(f"    Error final: {recuperado['error_final']:.2f}°" if recuperado['error_final'] else "    Error final: N/A")
    print(f"    Energia acumulada: {recuperado['energia']:.0f}°")
    
    # Calcular degradacion y recuperacion
    if baseline['t_settle'] and fatigado['t_settle']:
        degradacion = (fatigado['t_settle'] - baseline['t_settle']) / baseline['t_settle'] * 100
        print(f"\n  Degradacion por fatiga: +{degradacion:.1f}% T_settle")
    else:
        degradacion = None
    
    if baseline['t_settle'] and recuperado['t_settle']:
        recuperacion = (baseline['t_settle'] - recuperado['t_settle']) / baseline['t_settle'] * 100
        print(f"  Recuperacion post-reposo: {recuperacion:.1f}% (negativo = aun fatigado)")
    else:
        recuperacion = None
    
    # Criterios O-N11
    exito_fatiga = degradacion and degradacion > 30
    exito_recuperacion = recuperacion and recuperacion > -20  # Al menos no empeoro
    
    print("\n" + "=" * 80)
    print("CONCLUSION V140 — Fatiga")
    print("=" * 80)
    
    if exito_fatiga:
        print("\n  ✅ FATIGA DEMOSTRADA: El organismo se cansa con el uso")
        print(f"     T_settle aumento {degradacion:.1f}% (>30%)")
    else:
        print(f"\n  ⚠️ FATIGA NO DEMOSTRADA: Degradacion insuficiente ({degradacion:.1f}%)")
    
    if exito_recuperacion:
        print("\n  ✅ RECUPERACION PARCIAL: El reposo restaura funcion")
    else:
        print("\n  ⚠️ RECUPERACION INSUFICIENTE: El organismo no recupera")
    
    # Graficos
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Grafico 1: Energia acumulada
    ax = axes[0, 0]
    # Combinar energias de todas las fases
    todas_energias = e1 + e2 + e3 + [0]*6000 + e5  # Aproximado
    ax.plot(todas_energias, 'r-', linewidth=0.8)
    ax.axhline(y=ENERGIA_MAX, color='orange', linestyle='--', label=f'Umbral fatiga ({ENERGIA_MAX}°)')
    ax.set_xlabel('Paso')
    ax.set_ylabel('Energia acumulada (grados)')
    ax.set_title('Energia metabolica acumulada')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Grafico 2: T_settle por fase
    ax = axes[0, 1]
    fases = ['Baseline', 'Fatigado', 'Recuperado']
    t_settle_vals = [baseline['t_settle'] or 0, fatigado['t_settle'] or 0, recuperado['t_settle'] or 0]
    colores = ['green', 'red', 'blue']
    bars = ax.bar(fases, t_settle_vals, color=colores, alpha=0.7)
    for bar, val in zip(bars, t_settle_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}s', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('T_settle (segundos)')
    ax.set_title('Tiempo de asentamiento por fase')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Grafico 3: Error final por fase
    ax = axes[1, 0]
    error_vals = [baseline['error_final'] or 0, fatigado['error_final'] or 0, recuperado['error_final'] or 0]
    bars = ax.bar(fases, error_vals, color=colores, alpha=0.7)
    for bar, val in zip(bars, error_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{val:.1f}°', ha='center', va='bottom', fontsize=10)
    ax.axhline(y=ZONA_MUERTA_BASE, color='gray', linestyle='--', label=f'Zona muerta ({ZONA_MUERTA_BASE}°)')
    ax.set_ylabel('Error final (grados)')
    ax.set_title('Precision por fase')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Grafico 4: Orientacion en fase fatigada (ultimos ciclos)
    ax = axes[1, 1]
    # Tomar los ultimos 5000 puntos de la fase 3
    if len(o3) > 5000:
        o3_muestra = o3[-5000:]
        s3_muestra = s3[-5000:]
        t3_muestra = np.linspace(0, len(o3_muestra) * DT, len(o3_muestra))
        ax.plot(t3_muestra, s3_muestra, 'r--', linewidth=0.8, alpha=0.7, label='Setpoint')
        ax.plot(t3_muestra, o3_muestra, 'b-', linewidth=0.6, label='Orientacion')
        ax.set_xlabel('Tiempo (s)')
        ax.set_ylabel('Angulo (grados)')
        ax.set_title('Comportamiento fatigado (ultimos ciclos)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('v140_logs', exist_ok=True)
    plt.savefig(f'v140_logs/v140_fatiga_{timestamp}.png', dpi=150)
    print(f"\n  Graficos guardados: v140_logs/v140_fatiga_{timestamp}.png")
    
    return sistema, exito_fatiga and exito_recuperacion


if __name__ == "__main__":
    import time
    start = time.time()
    sistema, exito = ejecutar_v140()
    elapsed = time.time() - start
    print(f"\n  Tiempo de ejecucion: {elapsed/60:.1f} minutos")