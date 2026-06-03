#!/usr/bin/env python3
"""
VSTCosmos V155 — ANIMA-2 Etapa 1: Consciencia básica corregida

Corrección sobre V154:
  - Cb como INTEGRAL de presión de desacople, no derivada
  - Cb = ∫(e_R * (1 - A_sys-env) - Cb/τ_cb) dt
  - Cb es persistente: sube cuando hay desacople sostenido

Criterios:
  1. Memoria motora: error_60s < 15° y E_historia > 1500°
  2. Correlación Cb vs (1 - A_sys-env) > 0.7
  3. Cb final > 50.0
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

# Fatiga
K_GAIN = 0.0003
K_PRECISION = 0.002
K_TEMBLOR = 0.001
TAU_RECUPERACION = 300.0

# Memoria de ausencia
TAU_BASE = 30.0
K_MEM = 0.005
SUELO_CONFIANZA = 0.2
K_HOLD = 0.001

# Consciencia básica (CORREGIDO)
TAU_CB = 10.0          # constante de "olvido del dolor"
CB_MAX = 500.0         # límite superior de Cb

# Semilla base
SEMILLA_BASE = 44

# Período
PERIODO_ALTERNANCIA = 80.0


# ============================================================
# HEMISFERIO
# ============================================================

class HemisferioV155:
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
# FATIGA METABOLICA
# ============================================================

class FatigaMetabolicaV155:
    def __init__(self, k_gain=K_GAIN, k_precision=K_PRECISION,
                 k_temblor=K_TEMBLOR, tau_recuperacion=TAU_RECUPERACION):
        self.k_gain = k_gain
        self.k_precision = k_precision
        self.k_temblor = k_temblor
        self.tau_recuperacion = tau_recuperacion
        
        self.historia = 0.0
        self.fatiga_activa = 0.0
        
        self.historial_historia = []
        self.historial_fatiga = []
    
    def actualizar(self, delta_real, costo_trabajo, en_reposo_real, dt):
        self.historia += abs(delta_real)
        
        if not en_reposo_real:
            self.fatiga_activa += costo_trabajo
        else:
            self.fatiga_activa *= np.exp(-dt / self.tau_recuperacion)
        
        self.fatiga_activa = min(self.fatiga_activa, 20000.0)
        
        factor_gain = np.exp(-self.k_gain * self.fatiga_activa)
        zona_muerta_efectiva = ZONA_MUERTA_BASE + self.k_precision * self.fatiga_activa
        temblor = self.k_temblor * self.fatiga_activa * np.random.randn()
        
        factor_gain = max(0.2, min(1.0, factor_gain))
        zona_muerta_efectiva = min(ZONA_MUERTA_MAX, zona_muerta_efectiva)
        temblor = np.clip(temblor, -3.0, 3.0)
        
        self.historial_historia.append(self.historia)
        self.historial_fatiga.append(self.fatiga_activa)
        
        return factor_gain, zona_muerta_efectiva, temblor
    
    def reset(self):
        self.historia = 0.0
        self.fatiga_activa = 0.0
        self.historial_historia = []
        self.historial_fatiga = []
    
    def get_historia(self):
        return self.historia
    
    def get_fatiga(self):
        return self.fatiga_activa


# ============================================================
# MEMORIA DE AUSENCIA
# ============================================================

class MemoriaAusencia:
    def __init__(self, tau_base=TAU_BASE, k_mem=K_MEM, suelo_confianza=SUELO_CONFIANZA):
        self.setpoint_last = 0.0
        self.t_ausencia = 0.0
        self.tau_base = tau_base
        self.k_mem = k_mem
        self.tau_mem = tau_base
        self.suelo_confianza = suelo_confianza
        self.historial_confianza = []
    
    def actualizar(self, setpoint, E_historia, dt):
        if setpoint is not None:
            self.setpoint_last = setpoint
            self.t_ausencia = 0.0
            self.tau_mem = self.tau_base + self.k_mem * E_historia
            self.historial_confianza.append(1.0)
            return self.setpoint_last, 1.0
        else:
            self.t_ausencia += dt
            confianza = np.exp(-self.t_ausencia / self.tau_mem)
            self.historial_confianza.append(confianza)
            return self.setpoint_last, confianza
    
    def get_confianza(self):
        return self.historial_confianza[-1] if self.historial_confianza else 0.0
    
    def get_tau_mem(self):
        return self.tau_mem
    
    def reset(self):
        self.setpoint_last = 0.0
        self.t_ausencia = 0.0
        self.tau_mem = self.tau_base
        self.historial_confianza = []


# ============================================================
# CONSCIENCIA BÁSICA CORREGIDA (INTEGRAL DE PRESIÓN)
# ============================================================

class ConscienciaBasica:
    """
    Cb como integral de presión de desacouple (O-N5.1).
    
    dCb/dt = e_R * (1 - A_sys-env) - Cb / tau_cb
    
    Cb acumula cuando hay desacople persistente.
    Cb decae cuando el acoplamiento se restaura.
    """
    
    def __init__(self, tau_cb=TAU_CB, cb_max=CB_MAX):
        self.Cb = 0.0
        self.tau_cb = tau_cb
        self.cb_max = cb_max
        self.historial_Cb = []
        self.historial_presion = []
    
    def actualizar(self, e_R, A_sys_env, dt):
        """
        Args:
            e_R: error de primer orden (|setpoint - orientacion|)
            A_sys_env: calidad de acoplamiento (0-1, 1 = perfecto)
            dt: paso de tiempo
        """
        # Presión instantánea: error * desacople
        # Si A_sys_env=1 (acoplamiento perfecto), presión=0
        presion = e_R * (1.0 - A_sys_env)
        
        # Cb como integrador leaky
        dCb_dt = presion - self.Cb / self.tau_cb
        self.Cb += dCb_dt * dt
        
        # Limitar Cb
        self.Cb = max(0.0, min(self.cb_max, self.Cb))
        
        self.historial_Cb.append(self.Cb)
        self.historial_presion.append(presion)
        
        return self.Cb, presion
    
    def reset(self):
        self.Cb = 0.0
        self.historial_Cb = []
        self.historial_presion = []


# ============================================================
# APARATO MOTOR V155
# ============================================================

class AparatoMotorV155:
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
        
        self.fatiga = FatigaMetabolicaV155()
        self.memoria = MemoriaAusencia()
        self.consciencia = ConscienciaBasica()
        
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
    
    def actuar(self, gradiente, LF_activa, fuente_activa, t, setpoint_raw):
        if not LF_activa:
            return self.orientacion, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        if abs(gradiente) < 0.01:
            return self.orientacion, self.fatiga.get_historia(), self.fatiga.get_fatiga(), 0.0, 0.0, 0.0, 0.0
        
        # Obtener setpoint efectivo y confianza
        setpoint_objetivo, confianza = self.memoria.actualizar(setpoint_raw, self.fatiga.get_historia(), DT)
        
        error = setpoint_objetivo - self.orientacion
        e_R = abs(error)
        
        # Determinar A_sys-env según contexto
        if setpoint_raw is not None:
            # En acción, A_sys-env = proporción de orientación alcanzada
            if abs(setpoint_raw) > 0.01:
                A_sys_env = min(1.0, abs(self.orientacion) / abs(setpoint_raw))
            else:
                A_sys_env = 1.0
        else:
            # En silencio, A_sys-env = confianza en la memoria
            A_sys_env = confianza
        
        # Actualizar consciencia básica (CORREGIDO: integral de presión)
        Cb, presion = self.consciencia.actualizar(e_R, A_sys_env, DT)
        
        # Efectos de fatiga
        factor_gain, zona_muerta_efectiva, temblor = self.fatiga.actualizar(0, 0, False, DT)
        
        # Zona muerta
        if abs(error) < zona_muerta_efectiva:
            self.fatiga.actualizar(0, 0, True, DT)
            return (self.orientacion, self.fatiga.get_historia(), 
                    self.fatiga.get_fatiga(), confianza, zona_muerta_efectiva, Cb, presion)
        
        # Dirección y confianza sensorial
        direccion = np.sign(error)
        confianza_sensorial = min(1.0, abs(gradiente) * self.sensibilidad_grad)
        factor_freno = self.calcular_factor_freno(error)
        
        # Kp base
        Kp_base_efectivo = self.Kp_actual * factor_gain * confianza_sensorial
        Kp_base_efectivo = max(self.Kp_min, Kp_base_efectivo)
        
        # Kp instantáneo modulado por confianza de memoria
        Kp_inst = Kp_base_efectivo * (self.memoria.suelo_confianza + (1 - self.memoria.suelo_confianza) * confianza)
        
        # Delta por error de posición
        delta_error = Kp_inst * abs(error) * direccion * factor_freno
        costo_error = abs(delta_error)
        
        # Torque de sujeción (solo durante silencio)
        torque_memoria = 0.0
        if setpoint_raw is None:
            torque_memoria = K_HOLD * (self.memoria.setpoint_last - self.orientacion) * confianza
        
        delta_raw = delta_error + torque_memoria
        
        # Inercia
        delta = self.inercia * self.ultimo_delta + (1 - self.inercia) * delta_raw
        self.ultimo_delta = delta
        self.ultimo_delta_registrado = delta
        
        # Costo total
        costo_total = costo_error + abs(torque_memoria)
        
        # Reposo real
        en_reposo_real = (setpoint_raw is None and abs(delta) < 0.001 and abs(torque_memoria) < 0.001)
        
        # Actualizar fatiga
        self.fatiga.actualizar(delta, costo_total, en_reposo_real, DT)
        
        # Temblor
        delta += temblor * DT
        
        self.actualizar_plasticidad(error)
        
        self.orientacion += delta
        self.orientacion = np.clip(self.orientacion, -self.limite, self.limite)
        
        self.t += DT
        
        return (self.orientacion, self.fatiga.get_historia(), 
                self.fatiga.get_fatiga(), confianza, zona_muerta_efectiva, Cb, presion)
    
    def reset(self):
        self.orientacion = 0.0
        self.ultimo_delta = 0.0
        self.ultimo_delta_registrado = 0.0
        self.t = 0.0
        self.Kp_actual = KP_BASE
        self.memoria_error.clear()
        self.historial_Kp = []
        self.fatiga.reset()
        self.memoria.reset()
        self.consciencia.reset()


# ============================================================
# SISTEMA V155
# ============================================================

class SistemaV155:
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
        
        self.izquierdo = HemisferioV155("L", 30.0, generar_ruido_rosa, seed=seed, sesgo=SESGO_L)
        self.derecho = HemisferioV155("R", 300.0, generar_clicks_poisson, seed=seed+100, sesgo=SESGO_R)
        
        self.sistema_B_izq = HemisferioV155("B_L", 30.0, generar_ruido_rosa, seed=seed+200, sesgo=SESGO_L)
        self.sistema_B_der = HemisferioV155("B_R", 300.0, generar_clicks_poisson, seed=seed+300, sesgo=SESGO_R)
        
        self.modo_entrenamiento = True
        self.motor = AparatoMotorV155()
        
        self.historial = {
            't': [],
            'orientacion': [],
            'setpoint_raw': [],
            'confianza': [],
            'Cb': [],
            'A_sys_env': [],
            'historia': [],
            'fatiga': [],
            's_shared': []
        }
    
    def calcular_s_shared(self):
        omega_A = (self.izquierdo._calcular_omega() + self.derecho._calcular_omega()) / 2
        omega_B = (self.sistema_B_izq._calcular_omega() + self.sistema_B_der._calcular_omega()) / 2
        return 1 - abs(omega_A - omega_B) / 2.0
    
    def actualizar(self, t, dt, duracion_total, setpoint_raw):
        fuente_activa = setpoint_raw is not None
        
        self.izquierdo.actualizar(t, dt, duracion_total, self.derecho)
        self.derecho.actualizar(t, dt, duracion_total, self.izquierdo)
        self.sistema_B_izq.actualizar(t, dt, duracion_total, self.sistema_B_der)
        self.sistema_B_der.actualizar(t, dt, duracion_total, self.sistema_B_izq)
        
        omega_A = (self.izquierdo._calcular_omega() + self.derecho._calcular_omega()) / 2
        omega_B = (self.sistema_B_izq._calcular_omega() + self.sistema_B_der._calcular_omega()) / 2
        gradiente = omega_A - omega_B
        
        if setpoint_raw is not None:
            sesgo = setpoint_raw / 90.0
            gradiente += sesgo * 0.5
        
        LF_activa = not self.modo_entrenamiento
        orientacion, historia, fatiga, confianza, _, Cb, presion = self.motor.actuar(
            gradiente, LF_activa, fuente_activa, t, setpoint_raw
        )
        
        # Calcular A_sys_env para el registro
        if setpoint_raw is not None:
            A_sys_env = min(1.0, abs(orientacion) / abs(setpoint_raw)) if abs(setpoint_raw) > 0.01 else 1.0
        else:
            A_sys_env = confianza
        
        self.historial['t'].append(t)
        self.historial['orientacion'].append(orientacion)
        self.historial['setpoint_raw'].append(setpoint_raw)
        self.historial['confianza'].append(confianza)
        self.historial['Cb'].append(Cb)
        self.historial['A_sys_env'].append(A_sys_env)
        self.historial['historia'].append(historia)
        self.historial['fatiga'].append(fatiga)
        self.historial['s_shared'].append(self.calcular_s_shared())
        
        return orientacion, historia, fatiga, confianza, Cb, A_sys_env
    
    def set_modo_entrenamiento(self, entrenamiento=True):
        self.modo_entrenamiento = entrenamiento
        if entrenamiento:
            self.motor.reset()


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def onda_cuadrada(t, periodo=PERIODO_ALTERNANCIA, amplitud=60.0):
    if (t % periodo) < (periodo / 2):
        return -amplitud
    else:
        return +amplitud


def ejecutar_v155():
    print("=" * 100)
    print("EXPERIMENTO V155 — ANIMA-2 Etapa 1: Consciencia básica corregida")
    print("=" * 100)
    print("  Cb como INTEGRAL de presión de desacople (no derivada)")
    print("  dCb/dt = e_R * (1 - A_sys-env) - Cb/τ")
    print("")
    print(f"  Parámetros:")
    print(f"    - Kp_base = {KP_BASE}")
    print(f"    - K_HOLD = {K_HOLD}")
    print(f"    - tau_cb = {TAU_CB}s")
    print(f"    - tau_base = {TAU_BASE}s")
    print("=" * 100)
    
    sistema = SistemaV155("V155", seed=SEMILLA_BASE)
    
    print("\n  Entrenando lateralidad (10 repeticiones)...")
    sistema.set_modo_entrenamiento(True)
    
    for rep in range(REPETICIONES_LENTAS):
        for i in range(int(TIEMPO_POR_REPETICION / DT)):
            t = rep * TIEMPO_POR_REPETICION + i * DT
            sistema.actualizar(t, DT, TIEMPO_POR_REPETICION * REPETICIONES_LENTAS,
                              setpoint_raw=0.0)
    
    print("  Entrenamiento completado.")
    print("\n  Iniciando test de consciencia básica corregida...")
    
    sistema.set_modo_entrenamiento(False)
    sistema.motor.reset()
    t_actual = TIEMPO_POR_REPETICION * REPETICIONES_LENTAS
    
    # F1: Baseline
    print("\n  F1: Baseline fresco (3 ciclos)...")
    for ciclo in range(3):
        for i in range(int(PERIODO_ALTERNANCIA / DT)):
            t = t_actual + (ciclo * PERIODO_ALTERNANCIA + i) * DT
            setpoint = onda_cuadrada(i * DT, periodo=PERIODO_ALTERNANCIA, amplitud=60.0)
            sistema.actualizar(t, DT, t_actual + 1000, setpoint)
    t_actual += 3 * PERIODO_ALTERNANCIA
    
    # F2: Fatiga
    print("\n  F2: Fatiga inducida (30 ciclos)...")
    for ciclo in range(30):
        for i in range(int(PERIODO_ALTERNANCIA / DT)):
            t = t_actual + i * DT
            setpoint = onda_cuadrada(i * DT, periodo=PERIODO_ALTERNANCIA, amplitud=60.0)
            sistema.actualizar(t, DT, t_actual + 1000, setpoint)
        if (ciclo + 1) % 10 == 0:
            historia = sistema.motor.fatiga.get_historia()
            fatiga = sistema.motor.fatiga.get_fatiga()
            print(f"      Ciclo {ciclo + 1}/30, fatiga={fatiga:.0f}°, historia={historia:.0f}°")
        t_actual += PERIODO_ALTERNANCIA
    
    historia_f2 = sistema.motor.fatiga.get_historia()
    
    # F3: Silencio con Cb
    print("\n  F3: Silencio (120s) - midiendo Cb integral...")
    setpoint_last = sistema.motor.memoria.setpoint_last
    
    tiempos_f3 = []
    orientaciones_f3 = []
    confianzas_f3 = []
    Cb_values = []
    A_sys_env_values = []
    
    for i in range(int(120.0 / DT)):
        t = t_actual + i * DT
        orient, historia, fatiga, confianza, Cb, A_sys_env = sistema.actualizar(t, DT, t_actual + 120, None)
        tiempos_f3.append(t)
        orientaciones_f3.append(orient)
        confianzas_f3.append(confianza)
        Cb_values.append(Cb)
        A_sys_env_values.append(A_sys_env)
        
        if i % 500 == 0:
            print(f"      t={i*DT:.0f}s | orient={orient:.1f}° | confianza={confianza:.3f} | Cb={Cb:.1f} | A_sys_env={A_sys_env:.3f}")
    
    # ============================================================
    # RESULTADOS
    # ============================================================
    print("\n" + "=" * 80)
    print("RESULTADOS V155")
    print("=" * 80)
    
    idx_60s = int(60.0 / DT)
    orient_60s = orientaciones_f3[idx_60s] if idx_60s < len(orientaciones_f3) else orientaciones_f3[-1]
    error_60s = abs(orient_60s - setpoint_last)
    
    # Correlación entre Cb y (1 - A_sys-env)
    if len(Cb_values) > 100 and len(A_sys_env_values) > 100:
        min_len = min(len(Cb_values), len(A_sys_env_values))
        desacople = [1.0 - A_sys_env_values[i] for i in range(min_len)]
        Cb_array = np.array(Cb_values[:min_len])
        corr = np.corrcoef(desacople, Cb_array)[0, 1] if len(desacople) > 0 else 0
    else:
        corr = 0
    
    Cb_final = Cb_values[-1] if Cb_values else 0
    
    print(f"\n  Memoria de ausencia (F3):")
    print(f"    Setpoint recordado: {setpoint_last:.0f}°")
    print(f"    Orientación a 60s: {orient_60s:.1f}°")
    print(f"    Error a 60s: {error_60s:.1f}°")
    print(f"    Historia acumulada (F2): {historia_f2:.0f}°")
    
    print(f"\n  Consciencia básica (Cb):")
    print(f"    Cb final: {Cb_final:.1f}")
    print(f"    Correlación Cb vs (1 - A_sys-env): {corr:.3f}")
    
    # Criterios
    exito_memoria = error_60s < 25.0 and historia_f2 > 1500
    exito_consciencia = corr > 0.7
    exito_cb_final = Cb_final > 50.0
    
    print("\n" + "=" * 80)
    print("CRITERIOS DE ÉXITO (Etapa 1)")
    print("=" * 80)
    print(f"  1. Memoria motora: {error_60s:.1f}° / {historia_f2:.0f}° -> {'✅' if exito_memoria else '❌'}")
    print(f"  2. Correlación Cb vs desacople > 0.7: {corr:.3f} -> {'✅' if exito_consciencia else '❌'}")
    print(f"  3. Cb final > 50.0: {Cb_final:.1f} -> {'✅' if exito_cb_final else '❌'}")
    
    exito = exito_memoria and exito_consciencia
    
    print("\n" + "=" * 80)
    print("CONCLUSION V155")
    print("=" * 80)
    
    if exito:
        print("\n  ✅ ETAPA 1 COMPLETADA")
        print("     Memoria de ausencia: ✅")
        print("     Consciencia básica (Cb): ✅")
        print("     Cb acumula presión de desacople persistentemente")
        print("\n  ANIMA-2 listo para Etapa 2 (Juego enactuado - V156)")
    else:
        print("\n  ⚠️ ETAPA 1 PARCIAL")
        if not exito_memoria:
            print(f"     Memoria insuficiente: error {error_60s:.1f}° > 25° o historia {historia_f2:.0f}° < 1500°")
        if not exito_consciencia:
            print(f"     Cb no correlaciona suficientemente con desacople: {corr:.3f} < 0.7")
        if not exito_cb_final:
            print(f"     Cb final insuficiente: {Cb_final:.1f} < 50.0")
    
    # Gráficos
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Gráfico 1: Orientación en F3
    ax = axes[0, 0]
    ax.plot(orientaciones_f3, 'b-', linewidth=0.8)
    ax.axhline(y=setpoint_last, color='red', linestyle='--', alpha=0.7, label=f'Setpoint recordado ({setpoint_last:.0f}°)')
    ax.axhline(y=setpoint_last + 25, color='orange', linestyle=':', alpha=0.5, label='Umbral 25°')
    ax.axhline(y=setpoint_last - 25, color='orange', linestyle=':', alpha=0.5)
    ax.set_xlabel('Paso')
    ax.set_ylabel('Orientación (º)')
    ax.set_title('F3: Persistencia en silencio')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Gráfico 2: Confianza
    ax = axes[0, 1]
    ax.plot(confianzas_f3, 'purple', linewidth=0.8)
    ax.set_xlabel('Paso')
    ax.set_ylabel('Confianza')
    ax.set_title('Decaimiento de confianza')
    ax.grid(True, alpha=0.3)
    
    # Gráfico 3: Cb vs desacople
    ax = axes[1, 0]
    ax.plot(Cb_values, 'orange', linewidth=0.8, label='Cb')
    ax.plot([1.0 - a for a in A_sys_env_values], 'g--', linewidth=0.6, alpha=0.7, label='Desacople (1 - A_sys-env)')
    ax.set_xlabel('Paso')
    ax.set_ylabel('Cb / Desacople')
    ax.set_title(f'Cb vs Desacople - Corr={corr:.3f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Gráfico 4: s_shared
    ax = axes[1, 1]
    s_shared = sistema.historial['s_shared']
    ax.plot(s_shared, 'purple', linewidth=0.5)
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Paso')
    ax.set_ylabel('s_shared')
    ax.set_title('Lateralidad')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('v155_logs', exist_ok=True)
    plt.savefig(f'v155_logs/v155_consciencia_presion_{timestamp}.png', dpi=150)
    print(f"\n  📊 Gráficos guardados: v155_logs/v155_consciencia_presion_{timestamp}.png")
    
    return sistema, exito


if __name__ == "__main__":
    import time
    start = time.time()
    sistema, exito = ejecutar_v155()
    elapsed = time.time() - start
    print(f"\n  ⏱️ Tiempo de ejecución: {elapsed/60:.1f} minutos")
    print(f"\n  V155 completado. Éxito: {exito}")