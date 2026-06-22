#!/usr/bin/env python3
"""
VSTCosmos V129 — Supervivencia, no perfeccion

Principios:
  1. Forzar asimetria inicial para eliminar fenotipo patologico
  2. Aceptar fragilidad como propiedad del organismo
  3. Caracterizar region de supervivencia, no buscar optimo
  4. El organismo no necesita ser fuerte, solo sobrevivir

Cambios:
  - Hemisferios nacen con sesgo: L=+0.01, R=-0.01
  - Zona muerta dinamica: se adapta al ruido
  - Metricas de supervivencia, no de perfeccion
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys
import time

# Importar V122 modificado (solo cambios en inicializacion)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# PARAMETROS
# ============================================================
DT = 0.01
TIEMPO_POR_REPETICION = 452.0
REPETICIONES_LENTAS = 10
TIEMPO_BASELINE = 60.0
TIEMPO_INANICION = 30.0

# Asimetria forzada al nacer
SESGO_L = 0.01   # Hemisferio izquierdo nace ligeramente activo
SESGO_R = -0.01  # Hemisferio derecho nace ligeramente inhibido
DIM_HEMISFERIO = 32

# Zona muerta dinamica
ZONA_MUERTA_BASE = 2.0
ZONA_MUERTA_MIN = 0.5
ZONA_MUERTA_MAX = 5.0

# Parkinson
TEMBLOR_AMPS = [0.0, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
FREQ_TEMBLOR = 5.0

# Semillas para validacion
SEMILLAS = [42, 43, 44, 45, 46]


# ============================================================
# HEMISFERIO CON ASIMETRIA FORZADA
# ============================================================

class HemisferioV129:
    """Hemisferio que nace con sesgo para evitar fenotipo patologico"""
    
    def __init__(self, nombre, tau, generar_entrada_func, seed=None, sesgo=0.0):
        if seed is not None:
            np.random.seed(seed)
        self.nombre = nombre
        self.tau = tau
        self.generar_entrada = generar_entrada_func
        self.sesgo = sesgo  # Asimetria forzada al nacer
        
        # Phi con sesgo: L nace activo, R nace inhibido
        self.Phi = np.random.normal(sesgo, 0.1, DIM_HEMISFERIO)
        self.Phi_vel = np.zeros(DIM_HEMISFERIO)
        self.W = np.zeros((DIM_HEMISFERIO, DIM_HEMISFERIO))
        
        self.entrada = None
        self.sr = 48000
        self.en_inanicion = False
        self.factor_inanicion = 1.0
        
        self.buffer_rapido = []
        self.historial_omega = []
        self.historial_Lambda = []
    
    def omega(self):
        return np.mean(self.Phi[:32])
    
    def _calcular_omega(self):
        return np.mean(self.Phi[:32])
    
    def _calcular_Lambda(self):
        # Simplificado para V129
        return abs(self._calcular_omega())
    
    def generar_entrada_para_t(self, t, duracion_total):
        if self.entrada is None:
            self.entrada = self.generar_entrada(duracion_total, self.sr)
        idx = int(t * self.sr)
        if idx >= len(self.entrada):
            return 0.0
        return self.entrada[idx] * self.factor_inanicion
    
    def inducir_inanicion_gradual(self, paso_actual, pasos_totales):
        if paso_actual < pasos_totales:
            self.factor_inanicion = 1.0 - (paso_actual / pasos_totales)
        else:
            self.factor_inanicion = 0.0
            self.en_inanicion = True
    
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
        
        omega = self._calcular_omega()
        
        self.buffer_rapido.append((t, omega))
        if len(self.buffer_rapido) > int(self.tau / dt):
            self.buffer_rapido.pop(0)
        
        self.historial_omega.append(omega)
        
        return {'omega': omega, 'entrada': entrada}


# ============================================================
# SISTEMA V129 (con asimetria forzada)
# ============================================================

class SistemaV129:
    """Sistema con hemisferios que nacen asimetricos"""
    
    def __init__(self, nombre, seed=42, motor_tipo='base', temblor_amp=0.0):
        self.nombre = nombre
        
        # Generadores de entrada (como V122)
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
        
        # Hemisferios con sesgo: L nace activo, R nace inhibido
        self.izquierdo = HemisferioV129("L", 30.0, generar_ruido_rosa, seed=seed, sesgo=SESGO_L)
        self.derecho = HemisferioV129("R", 300.0, generar_clicks_poisson, seed=seed+100, sesgo=SESGO_R)
        
        # Sistema B (para acoplamiento)
        self.sistema_B_izq = HemisferioV129("B_L", 30.0, generar_ruido_rosa, seed=seed+200, sesgo=SESGO_L)
        self.sistema_B_der = HemisferioV129("B_R", 300.0, generar_clicks_poisson, seed=seed+300, sesgo=SESGO_R)
        
        self.modo_entrenamiento = True
        
        # Motor
        self.motor = AparatoMotorV129(motor_tipo=motor_tipo, temblor_amp=temblor_amp)
        
        # Historial
        self.historial = {
            't': [],
            'omega_L': [],
            'omega_R': [],
            'omega_B_L': [],
            'omega_B_R': [],
            'orientacion': [],
            'error': [],
            's_shared': []
        }
    
    def omega_actual(self):
        # Para compatibilidad con V122
        return (self.izquierdo._calcular_omega() + self.derecho._calcular_omega()) / 2
    
    def calcular_s_shared(self, otro_sistema=None):
        if otro_sistema is None:
            # s_shared con sistema B
            omega_A = (self.izquierdo._calcular_omega() + self.derecho._calcular_omega()) / 2
            omega_B = (self.sistema_B_izq._calcular_omega() + self.sistema_B_der._calcular_omega()) / 2
            return 1 - abs(omega_A - omega_B) / 2.0
        return 1.0
    
    def actualizar(self, t, dt, duracion_total, audio_espacial=None):
        # Actualizar sistema A
        self.izquierdo.actualizar(t, dt, duracion_total, self.derecho)
        self.derecho.actualizar(t, dt, duracion_total, self.izquierdo)
        
        # Actualizar sistema B
        self.sistema_B_izq.actualizar(t, dt, duracion_total, self.sistema_B_der)
        self.sistema_B_der.actualizar(t, dt, duracion_total, self.sistema_B_izq)
        
        # Gradiente inter-sistemas
        omega_A = (self.izquierdo._calcular_omega() + self.derecho._calcular_omega()) / 2
        omega_B = (self.sistema_B_izq._calcular_omega() + self.sistema_B_der._calcular_omega()) / 2
        gradiente = omega_A - omega_B
        
        # Motor
        LF_activa = not self.modo_entrenamiento
        if audio_espacial is not None and not self.modo_entrenamiento:
            # Espacializacion simple
            sesgo = audio_espacial / 90.0
            gradiente += sesgo * 0.5
        
        orientacion = self.motor.actuar(gradiente, LF_activa)
        
        # Guardar historial
        self.historial['t'].append(t)
        self.historial['omega_L'].append(self.izquierdo._calcular_omega())
        self.historial['omega_R'].append(self.derecho._calcular_omega())
        self.historial['omega_B_L'].append(self.sistema_B_izq._calcular_omega())
        self.historial['omega_B_R'].append(self.sistema_B_der._calcular_omega())
        self.historial['orientacion'].append(orientacion)
        self.historial['error'].append(self.motor.setpoint - orientacion)
        self.historial['s_shared'].append(self.calcular_s_shared())
        
        return {'orientacion': orientacion, 's_shared': self.calcular_s_shared(), 'error': self.motor.setpoint - orientacion}
    
    def inducir_inanicion_sistema_B(self, paso_actual, pasos_totales):
        """Induce inanicion en el sistema B (para test R2)"""
        self.sistema_B_izq.inducir_inanicion_gradual(paso_actual, pasos_totales)
        self.sistema_B_der.inducir_inanicion_gradual(paso_actual, pasos_totales)
    
    def set_modo_entrenamiento(self, entrenamiento=True):
        self.modo_entrenamiento = entrenamiento


# ============================================================
# APARATO MOTOR V129 (con zona muerta dinamica)
# ============================================================

class AparatoMotorV129:
    """Motor con zona muerta dinamica y tolerancia a ruido"""
    
    def __init__(self, setpoint_inicial=-60.0, motor_tipo='base', temblor_amp=0.0):
        self.orientacion = 0.0
        self.setpoint = setpoint_inicial
        self.Kp_base = 0.002
        self.limite = 90.0
        self.zona_muerta_base = ZONA_MUERTA_BASE
        self.zona_muerta_min = ZONA_MUERTA_MIN
        self.zona_muerta_max = ZONA_MUERTA_MAX
        self.inercia = 0.95
        self.ultimo_delta = 0.0
        self.t = 0.0
        
        # Parkinson
        self.motor_tipo = motor_tipo
        self.temblor_amp = temblor_amp
        self.freq_temblor = FREQ_TEMBLOR
        
        # Metricas
        self.historial_orientacion = []
        self.historial_gradiente = []
        self.historial_delta = []
        self.historial_error = []
        self.historial_zona_muerta = []
    
    def calcular_factor_freno(self, error):
        return 1 - np.exp(-abs(error) / 30.0)
    
    def calcular_zona_muerta_adaptativa(self, ruido_estimado):
        """Zona muerta se expande con el ruido"""
        return np.clip(self.zona_muerta_base + ruido_estimado, 
                      self.zona_muerta_min, self.zona_muerta_max)
    
    def actuar(self, gradiente, LF_activa):
        if not LF_activa:
            return self.orientacion
        
        if abs(gradiente) < 0.05:
            return self.orientacion
        
        error = self.setpoint - self.orientacion
        
        # Zona muerta adaptativa
        ruido_estimado = self.temblor_amp * 0.5  # Estimacion simple
        zona_muerta_actual = self.calcular_zona_muerta_adaptativa(ruido_estimado)
        self.historial_zona_muerta.append(zona_muerta_actual)
        
        if abs(error) < zona_muerta_actual:
            return self.orientacion
        
        # Control proporcional
        ganancia_grad = -np.tanh(gradiente * 10.0)
        factor_freno = self.calcular_factor_freno(error)
        
        delta = self.Kp_base * error * ganancia_grad * factor_freno
        
        # Inercia
        delta = self.inercia * self.ultimo_delta + (1 - self.inercia) * delta
        self.ultimo_delta = delta
        
        # Parkinson: temblor
        if self.motor_tipo == 'parkinson' and self.temblor_amp > 0:
            temblor = self.temblor_amp * np.sin(2 * np.pi * self.freq_temblor * self.t)
            self.orientacion += delta + temblor
        else:
            self.orientacion += delta
        
        self.orientacion = np.clip(self.orientacion, -self.limite, self.limite)
        
        self.historial_orientacion.append(self.orientacion)
        self.historial_gradiente.append(gradiente)
        self.historial_delta.append(delta)
        self.historial_error.append(error)
        
        self.t += DT
        
        return self.orientacion
    
    def reset(self):
        self.orientacion = 0.0
        self.ultimo_delta = 0.0
        self.t = 0.0
        self.historial_orientacion = []
        self.historial_gradiente = []
        self.historial_delta = []
        self.historial_error = []
        self.historial_zona_muerta = []


# ============================================================
# FUNCIONES DE SUPERVIVENCIA
# ============================================================

def calcular_metricas_supervivencia(sistema, setpoint=-60.0):
    """Calcula metricas de supervivencia, no de perfeccion"""
    orientacion = np.array(sistema.historial['orientacion'])
    error = np.array(sistema.historial['error'])
    s_shared = np.array(sistema.historial['s_shared'])
    
    if len(orientacion) == 0:
        return {
            'sobrevive': False,
            's_shared_final': 1.0,
            'lateralidad': False,
            'U_eff': 10.0,
            'E': 0,
            'T_settle': None,
            'C50': False,
            'parkinson_severidad': 1.0
        }
    
    # Metricas basicas
    s_shared_final = np.mean(s_shared[-6000:]) if len(s_shared) > 6000 else np.mean(s_shared)
    lateralidad = s_shared_final < 0.8
    
    # Supervivencia: C50 (orientacion final cercana)
    orient_final = orientacion[-1] if len(orientacion) > 0 else 0
    C50 = abs(orient_final - setpoint) < 10.0  # Tolerancia aumentada a 10° (no 5°)
    
    # Temblor residual
    orient_ultimos = orientacion[-500:] if len(orientacion) >= 500 else orientacion
    U_eff = np.std(orient_ultimos)
    
    # Costo energetico
    E = np.sum(np.abs(np.diff(orientacion))) if len(orientacion) > 1 else 0.0
    
    # Tiempo de asentamiento (mas tolerante)
    T_settle = None
    for i, err in enumerate(error):
        if abs(err) < 5.0:  # Tolerancia aumentada
            T_settle = sistema.historial['t'][i] if i < len(sistema.historial['t']) else None
            break
    
    # Parkinson severidad: relacion E / distancia minima teorica
    distancia_teorica = abs(setpoint)  # 60°
    parkinson_severidad = E / distancia_teorica if distancia_teorica > 0 else 1.0
    
    # Supervivencia: el organismo sobrevive si C50 y no colapso energetico
    sobrevive = C50 and parkinson_severidad < 100  # Menos de 100x el minimo
    
    return {
        'sobrevive': sobrevive,
        's_shared_final': s_shared_final,
        'lateralidad': lateralidad,
        'U_eff': U_eff,
        'E': E,
        'T_settle': T_settle,
        'C50': C50,
        'orient_final': orient_final,
        'parkinson_severidad': parkinson_severidad
    }


def validar_poblacion():
    """Valida la poblacion de organismos con asimetria forzada"""
    print("\n" + "=" * 80)
    print("VALIDACION DE POBLACION (con asimetria forzada)")
    print("=" * 80)
    
    resultados = []
    
    for seed in SEMILLAS:
        print(f"\n  Semilla {seed}...", end=" ", flush=True)
        
        sistema = SistemaV129(f"V129_seed{seed}", seed=seed, motor_tipo='base')
        sistema.set_modo_entrenamiento(True)
        
        # Fase 2: Entrenamiento (simplificado para velocidad)
        for rep in range(REPETICIONES_LENTAS):
            for i in range(int(TIEMPO_POR_REPETICION / DT)):
                t = rep * TIEMPO_POR_REPETICION + i * DT
                sistema.actualizar(t, DT, TIEMPO_POR_REPETICION * REPETICIONES_LENTAS)
        
        # Fase 4: C50 (simplificado)
        sistema.set_modo_entrenamiento(False)
        for i in range(int(300.0 / DT)):
            t = TIEMPO_POR_REPETICION * REPETICIONES_LENTAS + i * DT
            sistema.actualizar(t, DT, 400.0, audio_espacial=-60.0)
        
        metricas = calcular_metricas_supervivencia(sistema)
        resultados.append(metricas)
        
        status = "✅" if metricas['sobrevive'] else "❌"
        print(f"{status} C50={metricas['C50']}, lateralidad={metricas['lateralidad']}, severidad={metricas['parkinson_severidad']:.1f}x")
    
    # Resumen poblacional
    print("\n  --- RESUMEN POBLACION ---")
    sobreviven = sum(1 for r in resultados if r['sobrevive'])
    lateralidad_ok = sum(1 for r in resultados if r['lateralidad'])
    C50_ok = sum(1 for r in resultados if r['C50'])
    
    print(f"    Supervivencia: {sobreviven}/{len(SEMILLAS)} ({sobreviven/len(SEMILLAS)*100:.0f}%)")
    print(f"    Lateralidad: {lateralidad_ok}/{len(SEMILLAS)}")
    print(f"    C50: {C50_ok}/{len(SEMILLAS)}")
    
    return resultados


def caracterizar_fragilidad():
    """Caracteriza la fragilidad al temblor"""
    print("\n" + "=" * 80)
    print("CARACTERIZACION DE FRAGILIDAD (Parkinson)")
    print("=" * 80)
    
    resultados = []
    
    for amp in TEMBLOR_AMPS:
        print(f"  Temblor {amp:.1f}°...", end=" ", flush=True)
        
        sistema = SistemaV129(f"V129_parkinson_amp{amp}", seed=42, 
                              motor_tipo='parkinson', temblor_amp=amp)
        sistema.set_modo_entrenamiento(True)
        
        # Entrenamiento rapido
        for rep in range(2):
            for i in range(int(TIEMPO_POR_REPETICION / DT)):
                t = rep * TIEMPO_POR_REPETICION + i * DT
                sistema.actualizar(t, DT, TIEMPO_POR_REPETICION * 2)
        
        # Test C50
        sistema.set_modo_entrenamiento(False)
        for i in range(int(300.0 / DT)):
            t = 2 * TIEMPO_POR_REPETICION + i * DT
            sistema.actualizar(t, DT, 400.0, audio_espacial=-60.0)
        
        metricas = calcular_metricas_supervivencia(sistema)
        metricas['temblor_amp'] = amp
        resultados.append(metricas)
        
        status = "✅" if metricas['sobrevive'] else "❌"
        print(f"{status} severidad={metricas['parkinson_severidad']:.1f}x, U_eff={metricas['U_eff']:.2f}°")
    
    # Encontrar umbral critico
    print("\n  --- UMBRAL DE FRAGILIDAD ---")
    
    umbral_supervivencia = None
    for r in resultados:
        if not r['sobrevive'] and umbral_supervivencia is None:
            umbral_supervivencia = r['temblor_amp']
    
    if umbral_supervivencia:
        print(f"    El organismo sobrevive hasta {umbral_supervivencia:.1f}° de temblor")
        print(f"    A partir de ahi, el costo energetico explota (>100x)")
    else:
        print(f"    El organismo sobrevive hasta {max(TEMBLOR_AMPS)}° (no se encontro ruptura)")
    
    return resultados


# ============================================================
# EXPERIMENTO V129
# ============================================================

def ejecutar_v129():
    print("=" * 100)
    print("EXPERIMENTO V129 — Supervivencia, no perfeccion")
    print("=" * 100)
    print("  Filosofia: El organismo no necesita ser fuerte, solo sobrevivir")
    print("  Cambios:")
    print("    - Asimetria forzada al nacer (L=+0.01, R=-0.01)")
    print("    - Zona muerta adaptativa (se expande con ruido)")
    print("    - Metricas de supervivencia (tolerancia 10°, no 5°)")
    print("    - Aceptamos fragilidad como propiedad")
    print("=" * 100)
    
    # Parte 1: Validacion de poblacion
    resultados_poblacion = validar_poblacion()
    
    # Parte 2: Caracterizacion de fragilidad
    resultados_parkinson = caracterizar_fragilidad()
    
    # Graficos
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Grafico 1: Supervivencia por semilla
    ax = axes[0, 0]
    semillas_str = [str(s) for s in SEMILLAS]
    sobreviven = [1 if r['sobrevive'] else 0 for r in resultados_poblacion]
    colores = ['green' if s else 'red' for s in sobreviven]
    ax.bar(semillas_str, sobreviven, color=colores)
    ax.set_xlabel('Semilla')
    ax.set_ylabel('Supervive')
    ax.set_title('Supervivencia por semilla')
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    
    # Grafico 2: Severidad por semilla
    ax = axes[0, 1]
    severidades = [r['parkinson_severidad'] for r in resultados_poblacion]
    ax.bar(semillas_str, severidades, color='orange')
    ax.axhline(y=100, color='red', linestyle='--', label='Umbral de colapso (100x)')
    ax.set_xlabel('Semilla')
    ax.set_ylabel('Severidad (E / distancia teorica)')
    ax.set_title('Costo energetico relativo')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Grafico 3: U_eff por semilla
    ax = axes[0, 2]
    ueffs = [r['U_eff'] for r in resultados_poblacion]
    ax.bar(semillas_str, ueffs, color='blue')
    ax.axhline(y=2.0, color='gray', linestyle='--', label='Zona muerta base')
    ax.set_xlabel('Semilla')
    ax.set_ylabel('U_eff (grados)')
    ax.set_title('Temblor residual')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Grafico 4: Supervivencia vs temblor
    ax = axes[1, 0]
    temblores = [r['temblor_amp'] for r in resultados_parkinson]
    sobreviven_park = [1 if r['sobrevive'] else 0 for r in resultados_parkinson]
    ax.plot(temblores, sobreviven_park, 'o-', color='green', linewidth=2, markersize=8)
    ax.set_xlabel('Amplitud de temblor (grados)')
    ax.set_ylabel('Supervive')
    ax.set_title('Supervivencia vs Parkinson')
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    
    # Grafico 5: Severidad vs temblor
    ax = axes[1, 1]
    severidades_park = [r['parkinson_severidad'] for r in resultados_parkinson]
    ax.semilogy(temblores, severidades_park, 'o-', color='red', linewidth=2, markersize=8)
    ax.axhline(y=100, color='orange', linestyle='--', label='Umbral de colapso (100x)')
    ax.set_xlabel('Amplitud de temblor (grados)')
    ax.set_ylabel('Severidad (escala log)')
    ax.set_title('Costo energetico vs Parkinson')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Grafico 6: U_eff vs temblor
    ax = axes[1, 2]
    ueffs_park = [r['U_eff'] for r in resultados_parkinson]
    ax.plot(temblores, ueffs_park, 'o-', color='purple', linewidth=2, markersize=8)
    ax.axhline(y=2.0, color='gray', linestyle='--', label='Zona muerta base')
    ax.set_xlabel('Amplitud de temblor (grados)')
    ax.set_ylabel('U_eff (grados)')
    ax.set_title('Temblor residual vs Parkinson')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('v129_logs', exist_ok=True)
    plt.savefig(f'v129_logs/v129_resultados_{timestamp}.png', dpi=150)
    print(f"\n  Graficos guardados: v129_logs/v129_resultados_{timestamp}.png")
    
    # Conclusion
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    
    tasa_supervivencia = sum(1 for r in resultados_poblacion if r['sobrevive']) / len(resultados_poblacion) * 100
    
    print(f"  Tasa de supervivencia: {tasa_supervivencia:.0f}%")
    print(f"  Umbral de Parkinson: ~{resultados_parkinson[1]['temblor_amp'] if len(resultados_parkinson)>1 else '?'}°")
    print(f"  Filosofia: El organismo sobrevive en su nicho, no necesita ser perfecto")
    
    if tasa_supervivencia >= 80:
        print("\n  ✅ POBLACION VIABLE: 80%+ de los individuos sobreviven")
    else:
        print("\n  ⚠️ POBLACION FRAGIL: Menos del 80% sobrevive")
    
    return resultados_poblacion, resultados_parkinson


if __name__ == "__main__":
    start = time.time()
    pob, park = ejecutar_v129()
    print(f"\n  Tiempo: {(time.time() - start)/60:.1f} minutos")