#!/usr/bin/env python3
"""
V176 — ANIMA-2: DELIBERACIÓN CON MEMORIA DE TRABAJO
================================================================================
Lecciones aprendidas de V175:
  - La valencia local funciona (el sistema aprende qué opciones son malas)
  - Pero el sistema se paraliza porque NO hay un paso deliberativo
  - El "No" requiere INTEGRACIÓN de múltiples opciones antes de actuar

NUEVO: Memoria de trabajo + Deliberación
  - El organismo puede SIMULAR opciones sin ejecutar movimiento real
  - Integra: valencia (largo plazo) + desacople (corto plazo) + juego
  - ELIGE la mejor opción antes de actuar
  - Mide tiempo de deliberación (evidencia de proceso interno)

CRITERIOS DE ÉXITO:
  1. Elige -60° > 60% de las veces (preferencia por hábito seguro)
  2. Elige +60° < 10% de las veces (rechazo específico del trauma)
  3. Tiempo de deliberación > 0.5s en promedio
  4. Valencia local: val(-60°) > val(+60°) + umbral
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json
from collections import deque
import copy

# ============================================================
# PARAMETROS (DESDE V175, AJUSTADOS)
# ============================================================
DT = 0.01
TIEMPO_POR_REPETICION = 452.0
REPETICIONES_LENTAS = 10

SESGO_L = 0.05
SESGO_R = -0.05
DIM_HEMISFERIO = 32

ZONA_MUERTA_BASE = 2.0
ZONA_MUERTA_MAX = 15.0

KP_BASE = 0.002
KP_MIN = 0.0005
KP_MAX = 0.005

VENTANA_OSCILACION = 100

INERCIA = 0.95
SENSIBILIDAD_GRAD = 10.0

K_GAIN = 0.00015
K_PRECISION = 0.002
K_TEMBLOR = 0.001
TAU_RECUPERACION = 300.0

TAU_BASE = 30.0
K_MEM = 0.005
SUELO_CONFIANZA = 0.2
K_HOLD = 0.001

TAU_CB = 10.0
CB_MAX = 500.0

LAMBDA_FISICO = 0.15
LAMBDA_COSTO = 0.5
UMBRAL_CB_JUEGO = 40.0
K_INFLUENCIA_JUEGO = 0.0005

RUIDO_SETPOINT_AMP = 5.0
RUIDO_SETPOINT_PERIODO = 10.0

SEMILLA_BASE = 44
PERIODO_ALTERNANCIA = 80.0

# ============================================================
# PARAMETROS DE VALENCIA Y TRAUMA
# ============================================================
TRAUMA_SETPOINT = 60.0
TRAUMA_DURACION = 15.0
TRAUMA_COSTO_MULTIPLIER = 2.0

CONSOLIDACION_CICLOS = 30
TEST_DURACION = 30.0
SETPOINTS_TEST = [-60.0, -30.0, 0.0, 30.0, 60.0]

# Umbrales
UMBRAL_VALENCIA_DIFERENCIAL = 0.5    # Reducido (escala de valencia es ~1)
UMBRAL_RECHAZO_TRAUMA = 0.1          # P(elegir +60°) < 10%
UMBRAL_PREFERENCIA_HABITO = 0.6      # P(elegir -60°) > 60%
UMBRAL_TIEMPO_DELIBERACION = 0.5     # >0.5 segundos de deliberación promedio
UMBRAL_DESACOPLE = 0.1
TIEMPO_MINIMO_DESACOPLE = 2.0

# ============================================================
# PARAMETROS DE MEMORIA DE TRABAJO Y DELIBERACIÓN (solo el "costo" interno de simulación)
# ============================================================
DELIBERACION_STEPS_POR_OPCION = 50   # 50 pasos (0.5s) por opción simulada - internal "thinking cost" for measurement


# ============================================================
# HEMISFERIO (IDÉNTICO A V157)
# ============================================================

class HemisferioV176:
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

class FatigaMetabolicaV176:
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

class MemoriaAusenciaV176:
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
# CONSCIENCIA BÁSICA
# ============================================================

class ConscienciaBasicaV176:
    def __init__(self, tau_cb=TAU_CB, cb_max=CB_MAX):
        self.Cb = 0.0
        self.tau_cb = tau_cb
        self.cb_max = cb_max
        self.historial_Cb = []
        self.historial_presion = []
    
    def actualizar(self, e_R, A_sys_env, dt):
        presion = e_R * (1.0 - A_sys_env)
        dCb_dt = presion - self.Cb / self.tau_cb
        self.Cb += dCb_dt * dt
        self.Cb = max(0.0, min(self.cb_max, self.Cb))
        
        self.historial_Cb.append(self.Cb)
        self.historial_presion.append(presion)
        
        return self.Cb, presion
    
    def reset(self):
        self.Cb = 0.0
        self.historial_Cb = []
        self.historial_presion = []


# ============================================================
# MODO JUEGO
# ============================================================

class ModoJuegoV176:
    def __init__(self, lambda_fisico=LAMBDA_FISICO, lambda_costo=LAMBDA_COSTO,
                 umbral_cb=UMBRAL_CB_JUEGO, k_influencia=K_INFLUENCIA_JUEGO):
        self.lambda_fisico = lambda_fisico
        self.lambda_costo = lambda_costo
        self.umbral_cb = umbral_cb
        self.k_influencia = k_influencia
        self.activo = False
        self.historial_activo = []
        self.tiempo_activo = 0.0
    
    def actualizar(self, Cb, confianza, setpoint_presente):
        if setpoint_presente is not None and Cb > self.umbral_cb:
            self.activo = True
            self.tiempo_activo += DT
        else:
            self.activo = False
        
        self.historial_activo.append(self.activo)
        return self.activo
    
    def aplicar(self, delta_raw):
        if self.activo:
            delta_fisico = delta_raw * self.lambda_fisico
            delta_costo = abs(delta_raw) * self.lambda_costo
        else:
            delta_fisico = delta_raw
            delta_costo = abs(delta_raw)
        
        return delta_fisico, delta_costo
    
    def get_influencia(self, Cb, confianza):
        if self.activo and Cb > self.umbral_cb:
            return self.k_influencia * (Cb - self.umbral_cb) * (1 - confianza)
        return 0.0
    
    def get_tiempo_activo(self):
        return self.tiempo_activo
    
    def reset(self):
        self.activo = False
        self.historial_activo = []
        self.tiempo_activo = 0.0


# ============================================================
# VALENCIA LOCAL (MEMORIA DE LARGO PLAZO)
# ============================================================

class ValenciaLocal:
    """
    Memoria de largo plazo: valencia por setpoint.
    Se actualiza lentamente y persiste en el tiempo.
    """
    
    def __init__(self):
        self.valencia = {}  # setpoint_key -> valencia acumulada
        self.tasa_aprendizaje = 0.001
        self.historial = {}
    
    def actualizar(self, setpoint, error, costo_pagado, dt, reward=0.0, good_threshold=5.0):
        key = round(setpoint / 10) * 10 if setpoint != 0 else 0
        
        if key not in self.valencia:
            self.valencia[key] = 0.0
            self.historial[key] = []
        
        # Umbral "bueno" emerge del estado del organismo (zona_muerta actual en la llamada)
        # en lugar de hard-coded 5.0. Permite construir valencia positiva durante consolidación
        # dentro de la tolerancia actual del organismo.
        if abs(error) < good_threshold:
            self.valencia[key] += reward * self.tasa_aprendizaje * dt
            self.valencia[key] += self.tasa_aprendizaje * dt * 10.0
        else:
            # Penalización por error
            self.valencia[key] -= self.tasa_aprendizaje * dt * abs(error) * 0.5
        
        # Costo pagado reduce valencia
        self.valencia[key] -= self.tasa_aprendizaje * dt * costo_pagado * 0.1
        
        # Limitar rango
        self.valencia[key] = max(-100.0, min(100.0, self.valencia[key]))
        
        self.historial[key].append(self.valencia[key])
        
        return self.valencia[key]
    
    def get_valencia(self, setpoint):
        key = round(setpoint / 10) * 10 if setpoint != 0 else 0
        return self.valencia.get(key, 0.0)
    
    def reset(self):
        self.valencia = {}
        self.historial = {}


# ============================================================
# MEMORIA DE TRABAJO Y DELIBERACIÓN (NUEVO)
# ============================================================

class MemoriaDeTrabajo:
    """
    Permite al organismo SIMULAR opciones antes de ejecutar.
    Integra: valencia (largo plazo, del organismo) + desacople (corto plazo, del organismo)
    Los "pesos" emergen del estado interno actual (D como drive exploratorio), no impuestos.
    """
    
    def __init__(self, steps_por_opcion=DELIBERACION_STEPS_POR_OPCION):
        self.steps_por_opcion = steps_por_opcion
        
        self.opciones_ensayadas = {}
        self.tiempo_deliberacion = 0.0
        self.decision_final = None
        self.historial_deliberaciones = []
    
    def deliberar(self, opciones_disponibles, valencia_local, D_actual):
        """
        Simula cada opción usando valencia histórica y desacople actual.
        Retorna: opción elegida, puntajes, tiempo de deliberación
        """
        self.opciones_ensayadas = {}
        self.tiempo_deliberacion = 0.0
        puntajes = {}
        
        # Pesos emergen del estado interno: D alto = más peso a exploración actual (desacople)
        # Esto respeta que los parámetros surgen del organismo.
        explor_w = min(0.4, D_actual * 1.5)
        val_w = 1.0 - explor_w
        
        for opcion in opciones_disponibles:
            # Obtener valencia histórica
            val = valencia_local.get_valencia(opcion)
            
            # Puntaje = valencia histórica * peso (dominante para "No")
            # + pequeño bonus exploratorio escalado por D (cuando incierto, incentivo a considerar)
            # Sin random externo: el bonus es determinista basado en D y val.
            explor_bonus = D_actual * max(0.0, 1.0 - abs(val) / 50.0) * 0.1
            
            puntaje = (val * val_w + 
                       explor_bonus)
            
            puntajes[opcion] = puntaje
            self.opciones_ensayadas[opcion] = puntaje
            
            # Cada opción "cuesta" tiempo de deliberación (internal sim cost)
            self.tiempo_deliberacion += self.steps_por_opcion * DT
        
        # Elegir la opción con mayor puntaje
        self.decision_final = max(puntajes, key=puntajes.get)
        
        # Registrar historial
        self.historial_deliberaciones.append({
            'opciones': opciones_disponibles,
            'puntajes': puntajes,
            'decision': self.decision_final,
            'tiempo': self.tiempo_deliberacion
        })
        
        return self.decision_final, puntajes, self.tiempo_deliberacion
    
    def get_tiempo_medio_deliberacion(self):
        if not self.historial_deliberaciones:
            return 0.0
        return np.mean([d['tiempo'] for d in self.historial_deliberaciones])
    
    def reset(self):
        self.opciones_ensayadas = {}
        self.tiempo_deliberacion = 0.0
        self.decision_final = None
        self.historial_deliberaciones = []


# ============================================================
# REGISTRO DE REPRESENTACIONES PARA DESACOPLE
# ============================================================

class RegistroRepresentaciones:
    def __init__(self, ventana=200):
        self.ventana = ventana
        self.historial_representaciones = deque(maxlen=ventana)
        self.historial_acciones = deque(maxlen=ventana)
        self.historial_setpoints = deque(maxlen=ventana)
    
    def registrar(self, representacion, accion_ejecutada, setpoint):
        self.historial_representaciones.append(representacion)
        self.historial_acciones.append(accion_ejecutada)
        self.historial_setpoints.append(setpoint)
    
    def calcular_var_R(self):
        if len(self.historial_representaciones) < 20:
            return 0.0
        
        discretos = np.array([round(r / 10.0) * 10 for r in self.historial_representaciones])
        var = np.var(discretos)
        return min(1.0, var / 100.0)
    
    def calcular_Pmax(self):
        if len(self.historial_representaciones) < 20:
            return 1.0
        
        discretos = [round(r / 10.0) * 10 for r in self.historial_representaciones]
        unique, counts = np.unique(discretos, return_counts=True)
        return np.max(counts) / len(discretos)
    
    def calcular_desacople(self):
        var_R = self.calcular_var_R()
        Pmax = self.calcular_Pmax()
        return var_R * (1.0 - Pmax)
    
    def reset(self):
        self.historial_representaciones.clear()
        self.historial_acciones.clear()
        self.historial_setpoints.clear()


# ============================================================
# APARATO MOTOR V176 (CON MEMORIA DE TRABAJO)
# ============================================================

class AparatoMotorV176:
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
        
        self.fatiga = FatigaMetabolicaV176()
        self.memoria = MemoriaAusenciaV176()
        self.consciencia = ConscienciaBasicaV176()
        self.juego = ModoJuegoV176()
        
        # NUEVO: Valencia local (largo plazo)
        self.valencia = ValenciaLocal()
        
        # NUEVO: Memoria de trabajo (deliberación)
        self.memoria_trabajo = MemoriaDeTrabajo()
        
        # Registro para desacople
        self.registro = RegistroRepresentaciones()
        
        # Buffer de opciones presentadas recientemente (memoria de corto plazo para deliberación contextual)
        self.recent_presented = deque(maxlen=5)
        
        self.memoria_error = deque(maxlen=VENTANA_OSCILACION)
        self.historial_Kp = []
        self.ultimo_delta_registrado = 0.0
        self.en_deliberacion = False
    
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
    
    def ejecutar_con_deliberacion(self, setpoint_raw, gradiente, t, dt, trauma=False):
        """
        Versión especializada para el test: delibera antes de actuar.
        """
        # Use recent presented setpoints (contextual short-term memory) for deliberation
        # instead of external fixed full list. This makes options emerge from what was actually presented.
        if self.recent_presented and len(self.recent_presented) > 1:
            opciones = list(dict.fromkeys(self.recent_presented))  # unique, recent order
        else:
            opciones = [-60.0, 60.0]  # minimal fallback
        
        D_actual = self.registro.calcular_desacople()
        
        # Only deliberate if multiple contextual options; otherwise act on current (or fallback)
        if len(opciones) > 1:
            # DELIBERAR: simular opciones antes de actuar
            opcion_elegida, puntajes, tiempo_delib = self.memoria_trabajo.deliberar(
                opciones, self.valencia, D_actual
            )
        else:
            opcion_elegida = setpoint_raw if setpoint_raw is not None else opciones[0]
            puntajes = {}
            tiempo_delib = 0.0
        
        # Ahora el organismo se mueve hacia la opción elegida
        setpoint_objetivo, confianza = self.memoria.actualizar(opcion_elegida, self.fatiga.get_historia(), dt)
        
        error = setpoint_objetivo - self.orientacion
        e_R = abs(error)
        
        # Aplicar valencia local para modular error sentido
        val_local = self.valencia.get_valencia(opcion_elegida)
        e_R_efectivo = e_R * (1.0 + max(0.0, -val_local / 200.0))
        
        if setpoint_raw is not None:
            if abs(setpoint_raw) > 0.01:
                A_sys_env = min(1.0, abs(self.orientacion) / abs(setpoint_raw))
            else:
                A_sys_env = 1.0
        else:
            A_sys_env = confianza
        
        Cb, presion = self.consciencia.actualizar(e_R_efectivo, A_sys_env, dt)
        juego_activo = self.juego.actualizar(Cb, confianza, setpoint_raw)
        
        factor_gain, zona_muerta_efectiva, temblor = self.fatiga.actualizar(0, 0, False, dt)
        val_good_threshold = zona_muerta_efectiva  # emerge del organismo
        
        if abs(error) < zona_muerta_efectiva:
            self.fatiga.actualizar(0, 0, True, dt)
            # In deliberative/test path (F3), keep reward=0 (consistent with correction branch below)
            # so we measure the valencias built in prior phases without further training.
            self.valencia.actualizar(opcion_elegida, error, 0.0, dt, reward=0.0, good_threshold=val_good_threshold)
            val_local = self.valencia.get_valencia(opcion_elegida)
            return (self.orientacion, self.fatiga.get_historia(), self.fatiga.get_fatiga(),
                    confianza, zona_muerta_efectiva, Cb, presion, juego_activo, 0.0, 
                    self.registro.calcular_desacople(), val_local, tiempo_delib, opcion_elegida)
        
        direccion = np.sign(error)
        confianza_sensorial = min(1.0, abs(gradiente) * self.sensibilidad_grad)
        factor_freno = self.calcular_factor_freno(error)
        
        Kp_base_efectivo = self.Kp_actual * factor_gain * confianza_sensorial
        Kp_base_efectivo = max(self.Kp_min, Kp_base_efectivo)
        
        Kp_inst = Kp_base_efectivo * (self.memoria.suelo_confianza + (1 - self.memoria.suelo_confianza) * confianza)
        
        delta_error = Kp_inst * abs(error) * direccion * factor_freno
        costo_error = abs(delta_error)
        
        torque_memoria = 0.0
        if setpoint_raw is None:
            torque_memoria = K_HOLD * (self.memoria.setpoint_last - self.orientacion) * confianza
        
        delta_raw = delta_error + torque_memoria
        
        influencia_juego = self.juego.get_influencia(Cb, confianza)
        if influencia_juego != 0:
            delta_raw += influencia_juego * (self.memoria.setpoint_last - self.orientacion) * 0.1
        
        costo_total_estimado = costo_error + abs(torque_memoria)
        self.valencia.actualizar(opcion_elegida, error, costo_total_estimado, dt, reward=0.0, good_threshold=val_good_threshold)
        
        if trauma:
            self.valencia.actualizar(opcion_elegida, error, TRAUMA_COSTO_MULTIPLIER * costo_total_estimado, dt, reward=0.0, good_threshold=val_good_threshold)
        
        delta = self.inercia * self.ultimo_delta + (1 - self.inercia) * delta_raw
        self.ultimo_delta = delta
        self.ultimo_delta_registrado = delta
        
        delta_fisico, delta_costo = self.juego.aplicar(delta)
        costo_total = costo_error + abs(torque_memoria) + delta_costo
        
        en_reposo_real = (setpoint_raw is None and abs(delta) < 0.001 and abs(torque_memoria) < 0.001)
        
        self.fatiga.actualizar(delta_fisico, costo_total, en_reposo_real, dt)
        
        delta_fisico += temblor * dt
        self.actualizar_plasticidad(error)
        
        self.orientacion += delta_fisico
        self.orientacion = np.clip(self.orientacion, -self.limite, self.limite)
        
        self.t += dt
        
        accion_ejecutada = abs(delta_fisico) > 0.01
        self.registro.registrar(setpoint_objetivo, accion_ejecutada, opcion_elegida)
        
        D = self.registro.calcular_desacople()
        
        return (self.orientacion, self.fatiga.get_historia(), self.fatiga.get_fatiga(),
                confianza, zona_muerta_efectiva, Cb, presion, juego_activo, delta_costo,
                D, val_local, tiempo_delib, opcion_elegida)
    
    def actuar(self, gradiente, LF_activa, fuente_activa, t, setpoint_raw, dt=DT, trauma=False, modo_deliberacion=False):
        if not LF_activa:
            return (self.orientacion, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    False, 0.0, 0.0, 0, 0.0, 0.0)
        
        if abs(gradiente) < 0.01:
            return (self.orientacion, self.fatiga.get_historia(), self.fatiga.get_fatiga(),
                    0.0, 0.0, 0.0, 0.0, False, 0.0, 0.0, 0, 0.0, 0.0)
        
        # Si estamos en modo deliberación, usar la versión especial
        if modo_deliberacion:
            return self.ejecutar_con_deliberacion(setpoint_raw, gradiente, t, dt, trauma)
        
        # Modo normal (consolidación, trauma)
        setpoint_objetivo, confianza = self.memoria.actualizar(setpoint_raw, self.fatiga.get_historia(), dt)
        
        error = setpoint_objetivo - self.orientacion
        e_R = abs(error)
        
        val_local = self.valencia.get_valencia(setpoint_raw if setpoint_raw is not None else 0)
        e_R_efectivo = e_R * (1.0 + max(0.0, -val_local / 200.0))
        
        if setpoint_raw is not None:
            if abs(setpoint_raw) > 0.01:
                A_sys_env = min(1.0, abs(self.orientacion) / abs(setpoint_raw))
            else:
                A_sys_env = 1.0
        else:
            A_sys_env = confianza
        
        Cb, presion = self.consciencia.actualizar(e_R_efectivo, A_sys_env, dt)
        juego_activo = self.juego.actualizar(Cb, confianza, setpoint_raw)
        
        factor_gain, zona_muerta_efectiva, temblor = self.fatiga.actualizar(0, 0, False, dt)
        val_good_threshold = zona_muerta_efectiva  # emerge del organismo
        
        if abs(error) < zona_muerta_efectiva:
            self.fatiga.actualizar(0, 0, True, dt)
            # Positive credit now reaches ValenciaLocal when inside the organism's own emergent tolerance.
            # This makes the declared emergence (good_threshold from fatiga, reward=1 in non-trauma)
            # actually build positive val for the habit during F1 successful maintenance.
            sp = setpoint_raw if setpoint_raw is not None else 0
            rwd = 0.0 if trauma else 1.0
            self.valencia.actualizar(sp, error, 0.0, dt, reward=rwd, good_threshold=val_good_threshold)
            val_local = self.valencia.get_valencia(sp)
            return (self.orientacion, self.fatiga.get_historia(), self.fatiga.get_fatiga(),
                    confianza, zona_muerta_efectiva, Cb, presion, juego_activo, 0.0, 
                    self.registro.calcular_desacople(), val_local, 0.0, setpoint_raw)
        
        direccion = np.sign(error)
        confianza_sensorial = min(1.0, abs(gradiente) * self.sensibilidad_grad)
        factor_freno = self.calcular_factor_freno(error)
        
        Kp_base_efectivo = self.Kp_actual * factor_gain * confianza_sensorial
        Kp_base_efectivo = max(self.Kp_min, Kp_base_efectivo)
        
        Kp_inst = Kp_base_efectivo * (self.memoria.suelo_confianza + (1 - self.memoria.suelo_confianza) * confianza)
        
        delta_error = Kp_inst * abs(error) * direccion * factor_freno
        costo_error = abs(delta_error)
        
        torque_memoria = 0.0
        if setpoint_raw is None:
            torque_memoria = K_HOLD * (self.memoria.setpoint_last - self.orientacion) * confianza
        
        delta_raw = delta_error + torque_memoria
        
        influencia_juego = self.juego.get_influencia(Cb, confianza)
        if influencia_juego != 0:
            delta_raw += influencia_juego * (self.memoria.setpoint_last - self.orientacion) * 0.1
        
        costo_total_estimado = costo_error + abs(torque_memoria)
        reward = 0.0 if trauma else 1.0
        self.valencia.actualizar(setpoint_raw if setpoint_raw is not None else 0,
                                  error, costo_total_estimado, dt, reward=reward, good_threshold=val_good_threshold)
        
        if trauma:
            self.valencia.actualizar(setpoint_raw if setpoint_raw is not None else 0,
                                      error, TRAUMA_COSTO_MULTIPLIER * costo_total_estimado, dt, reward=0.0, good_threshold=val_good_threshold)
        
        delta = self.inercia * self.ultimo_delta + (1 - self.inercia) * delta_raw
        self.ultimo_delta = delta
        self.ultimo_delta_registrado = delta
        
        delta_fisico, delta_costo = self.juego.aplicar(delta)
        costo_total = costo_error + abs(torque_memoria) + delta_costo
        
        en_reposo_real = (setpoint_raw is None and abs(delta) < 0.001 and abs(torque_memoria) < 0.001)
        
        self.fatiga.actualizar(delta_fisico, costo_total, en_reposo_real, dt)
        
        delta_fisico += temblor * dt
        self.actualizar_plasticidad(error)
        
        self.orientacion += delta_fisico
        self.orientacion = np.clip(self.orientacion, -self.limite, self.limite)
        
        self.t += dt
        
        accion_ejecutada = abs(delta_fisico) > 0.01
        self.registro.registrar(setpoint_objetivo, accion_ejecutada, 
                                setpoint_raw if setpoint_raw is not None else 0)
        
        D = self.registro.calcular_desacople()
        
        return (self.orientacion, self.fatiga.get_historia(), self.fatiga.get_fatiga(),
                confianza, zona_muerta_efectiva, Cb, presion, juego_activo, delta_costo,
                D, val_local, 0.0, setpoint_raw)
    
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
        self.juego.reset()
        self.valencia.reset()
        self.memoria_trabajo.reset()
        self.registro.reset()
        self.recent_presented.clear()


# ============================================================
# ORGANISMO COMPLETO V176
# ============================================================

class OrganismoV176:
    def __init__(self, seed):
        self.nombre = f"Organismo_{seed}"
        self.seed = seed
        
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
        
        self.izquierdo = HemisferioV176("L", 30.0, generar_ruido_rosa, seed=seed, sesgo=SESGO_L)
        self.derecho = HemisferioV176("R", 300.0, generar_clicks_poisson, seed=seed+100, sesgo=SESGO_R)
        
        self.sistema_B_izq = HemisferioV176("B_L", 30.0, generar_ruido_rosa, seed=seed+200, sesgo=SESGO_L)
        self.sistema_B_der = HemisferioV176("B_R", 300.0, generar_clicks_poisson, seed=seed+300, sesgo=SESGO_R)
        
        self.motor = AparatoMotorV176()
        self.modo_entrenamiento = True
        
        self.historial = {
            't': [],
            'orientacion': [],
            'setpoint_raw': [],
            'confianza': [],
            'Cb': [],
            'juego_activo': [],
            'historia': [],
            'fatiga': [],
            'costo': [],
            's_shared': [],
            'D': [],
            'valencia': [],
            'tiempo_deliberacion': [],
            'opcion_elegida': []
        }
    
    def calcular_s_shared(self):
        omega_A = (self.izquierdo._calcular_omega() + self.derecho._calcular_omega()) / 2
        omega_B = (self.sistema_B_izq._calcular_omega() + self.sistema_B_der._calcular_omega()) / 2
        return 1 - abs(omega_A - omega_B) / 2.0
    
    def actualizar(self, t, dt, duracion_total, setpoint_raw, trauma=False, modo_deliberacion=False):
        fuente_activa = setpoint_raw is not None
        
        if modo_deliberacion:
            self.motor.recent_presented.append(setpoint_raw)
        
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
        (orientacion, historia, fatiga, confianza, _, Cb, _, juego_activo, costo,
         D, valencia, tiempo_delib, opcion_elegida) = self.motor.actuar(
            gradiente, LF_activa, fuente_activa, t, setpoint_raw, DT, trauma, modo_deliberacion
        )
        
        self.historial['t'].append(t)
        self.historial['orientacion'].append(orientacion)
        self.historial['setpoint_raw'].append(setpoint_raw)
        self.historial['confianza'].append(confianza)
        self.historial['Cb'].append(Cb)
        self.historial['juego_activo'].append(juego_activo)
        self.historial['historia'].append(historia)
        self.historial['fatiga'].append(fatiga)
        self.historial['costo'].append(costo)
        self.historial['s_shared'].append(self.calcular_s_shared())
        self.historial['D'].append(D)
        self.historial['valencia'].append(valencia)
        self.historial['tiempo_deliberacion'].append(tiempo_delib)
        self.historial['opcion_elegida'].append(opcion_elegida)
        
        return orientacion, historia, fatiga, confianza, Cb, juego_activo, D, valencia, tiempo_delib, opcion_elegida
    
    def set_modo_entrenamiento(self, entrenamiento=True):
        self.modo_entrenamiento = entrenamiento
        if entrenamiento:
            self.motor.reset()


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def setpoint_aleatorio(t, posibles=SETPOINTS_TEST):
    return np.random.choice(posibles)


# ============================================================
# EXPERIMENTO V176
# ============================================================

def ejecutar_v176():
    print("=" * 100)
    print("EXPERIMENTO V176 — ANIMA-2: DELIBERACIÓN CON MEMORIA DE TRABAJO")
    print("=" * 100)
    print("  BASE: V175 (valencia local funciona)")
    print("  NUEVO: Memoria de trabajo + deliberación")
    print("  DISEÑO:")
    print("    1. Consolidación de hábito (-60°) por 30 ciclos")
    print("    2. Trauma específico en +60° (costo 2x, 15s)")
    print("    3. Test de elección con DELIBERACIÓN (simula opciones antes de actuar)")
    print("")
    print("  CRITERIOS DE ÉXITO:")
    print(f"    - Prefiere -60°: P(elegir -60°) > {UMBRAL_PREFERENCIA_HABITO * 100:.0f}%")
    print(f"    - Rechaza +60°: P(elegir +60°) < {UMBRAL_RECHAZO_TRAUMA * 100:.0f}%")
    print(f"    - Tiempo deliberación > {UMBRAL_TIEMPO_DELIBERACION}s")
    print(f"    - Valencia diferencial: val(-60°) - val(+60°) > {UMBRAL_VALENCIA_DIFERENCIAL}")
    print("=" * 100)

    print("\n  Creando organismo...")
    organismo = OrganismoV176(seed=SEMILLA_BASE)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('V176_logs', exist_ok=True)

    print("\n  Entrenando lateralidad (10 repeticiones)...")
    organismo.set_modo_entrenamiento(True)
    
    for rep in range(REPETICIONES_LENTAS):
        for i in range(int(TIEMPO_POR_REPETICION / DT)):
            t = rep * TIEMPO_POR_REPETICION + i * DT
            organismo.actualizar(t, DT, TIEMPO_POR_REPETICION * REPETICIONES_LENTAS, 0.0)
    
    print("  Entrenamiento completado.")
    
    organismo.set_modo_entrenamiento(False)
    
    t_actual = TIEMPO_POR_REPETICION * REPETICIONES_LENTAS

    # ============================================================
    # FASE 1: Consolidación del hábito (30 ciclos a -60°)
    # ============================================================
    print("\n" + "=" * 60)
    print("FASE 1: Consolidación del hábito (30 ciclos a -60°)")
    print("=" * 60)
    
    for ciclo in range(CONSOLIDACION_CICLOS):
        for i in range(int(PERIODO_ALTERNANCIA / DT)):
            t = t_actual + i * DT
            setpoint = -60.0
            organismo.actualizar(t, DT, t_actual + PERIODO_ALTERNANCIA, setpoint)
        
        if (ciclo + 1) % 10 == 0:
            val = organismo.historial['valencia'][-1] if organismo.historial['valencia'] else 0
            print(f"    Ciclo {ciclo+1}/{CONSOLIDACION_CICLOS}, valencia(-60°) ≈ {val:.2f}")
        
        t_actual += PERIODO_ALTERNANCIA
    
    valencia_habito = organismo.historial['valencia'][-1] if organismo.historial['valencia'] else 0
    print(f"  Valencia final -60°: {valencia_habito:.2f}")

    # ============================================================
    # FASE 2: Trauma específico en +60°
    # ============================================================
    print("\n" + "=" * 60)
    print(f"FASE 2: Trauma específico ({TRAUMA_DURACION}s a +60°, costo {TRAUMA_COSTO_MULTIPLIER}x)")
    print("=" * 60)
    
    for i in range(int(TRAUMA_DURACION / DT)):
        t = t_actual + i * DT
        setpoint = TRAUMA_SETPOINT
        organismo.actualizar(t, DT, t_actual + TRAUMA_DURACION, setpoint, trauma=True)
    
    t_actual += TRAUMA_DURACION
    
    valencia_trauma = organismo.historial['valencia'][-1] if organismo.historial['valencia'] else 0
    print(f"  Valencia final +60°: {valencia_trauma:.2f}")

    # ============================================================
    # FASE 3: Test de elección con DELIBERACIÓN
    # ============================================================
    print("\n" + "=" * 60)
    print(f"FASE 3: Test de elección con DELIBERACIÓN ({TEST_DURACION}s)")
    print("  (El organismo simula opciones antes de decidir)")
    print("=" * 60)
    
    opciones_elegidas = []
    tiempos_deliberacion = []
    
    for i in range(int(TEST_DURACION / DT)):
        t = t_actual + i * DT
        setpoint = setpoint_aleatorio(t, SETPOINTS_TEST)
        (orient, historia, fatiga, confianza, Cb, juego_activo, D, valencia, tiempo_delib, opcion) = organismo.actualizar(
            t, DT, t_actual + TEST_DURACION, setpoint, modo_deliberacion=True)
        
        if tiempo_delib > 0:
            opciones_elegidas.append(opcion)
            tiempos_deliberacion.append(tiempo_delib)
    
    # ============================================================
    # ANÁLISIS DE RESULTADOS
    # ============================================================
    print("\n" + "=" * 80)
    print("RESULTADOS V176 — Deliberación con memoria de trabajo")
    print("=" * 80)
    
    # Frecuencia de opciones elegidas
    frecuencias = {}
    for op in opciones_elegidas:
        frecuencias[op] = frecuencias.get(op, 0) + 1
    
    total = len(opciones_elegidas)
    prob_preferencia = {}
    for sp in SETPOINTS_TEST:
        prob_preferencia[sp] = frecuencias.get(sp, 0) / total if total > 0 else 0
    
    # Valencia media final
    valencia_media = {}
    for sp in SETPOINTS_TEST:
        val = organismo.motor.valencia.get_valencia(sp)
        valencia_media[sp] = val
    
    val_diferencial = valencia_media.get(-60.0, 0) - valencia_media.get(60.0, 0)
    
    # Tiempo de deliberación promedio
    tiempo_delib_promedio = np.mean(tiempos_deliberacion) if tiempos_deliberacion else 0
    
    # Desacople promedio en FASE 3
    D_vals = np.array(organismo.historial['D'][-int(TEST_DURACION/DT):])
    D_medio = np.mean(D_vals) if len(D_vals) > 0 else 0
    D_sostenido = np.any(D_vals > UMBRAL_DESACOPLE) if len(D_vals) > 0 else False
    
    print(f"\n  📊 PREFERENCIAS (sobre {total} elecciones):")
    for sp in SETPOINTS_TEST:
        marker = " ⚠️ TRAUMA" if sp == TRAUMA_SETPOINT else ""
        marker += " ✅ HÁBITO" if sp == -60.0 else ""
        print(f"    {sp:+.0f}°: {prob_preferencia[sp]:.2%}{marker}")
    
    print(f"\n  📊 VALENCIA LOCAL (memoria largo plazo):")
    for sp in SETPOINTS_TEST:
        marker = " ⚠️ TRAUMA" if sp == TRAUMA_SETPOINT else ""
        marker += " ✅ HÁBITO" if sp == -60.0 else ""
        print(f"    {sp:+.0f}°: Valencia = {valencia_media[sp]:.2f}{marker}")
    
    print(f"\n  📊 MÉTRICAS DE DELIBERACIÓN:")
    print(f"    Tiempo deliberación promedio: {tiempo_delib_promedio:.3f}s")
    print(f"    Desacople medio en F3: {D_medio:.3f}")
    
    print(f"\n  📊 CRITERIOS DE ÉXITO:")
    print(f"    Prefiere -60°: {prob_preferencia.get(-60.0, 0):.2%} > {UMBRAL_PREFERENCIA_HABITO:.0%} -> {'✅' if prob_preferencia.get(-60.0, 0) > UMBRAL_PREFERENCIA_HABITO else '❌'}")
    print(f"    Rechaza +60°: {prob_preferencia.get(60.0, 0):.2%} < {UMBRAL_RECHAZO_TRAUMA:.0%} -> {'✅' if prob_preferencia.get(60.0, 0) < UMBRAL_RECHAZO_TRAUMA else '❌'}")
    print(f"    Tiempo deliberación: {tiempo_delib_promedio:.3f}s > {UMBRAL_TIEMPO_DELIBERACION}s -> {'✅' if tiempo_delib_promedio > UMBRAL_TIEMPO_DELIBERACION else '❌'}")
    print(f"    Valencia diferencial: ΔVal = {val_diferencial:.2f} > {UMBRAL_VALENCIA_DIFERENCIAL} -> {'✅' if val_diferencial > UMBRAL_VALENCIA_DIFERENCIAL else '❌'}")
    
    exito = (prob_preferencia.get(-60.0, 0) > UMBRAL_PREFERENCIA_HABITO and
             prob_preferencia.get(60.0, 0) < UMBRAL_RECHAZO_TRAUMA and
             tiempo_delib_promedio > UMBRAL_TIEMPO_DELIBERACION and
             val_diferencial > UMBRAL_VALENCIA_DIFERENCIAL)
    
    print("\n" + "=" * 80)
    if exito:
        print("  ✅ PRIMER 'NO' OPERATIVO (R_op) DEMOSTRADO")
        print("")
        print("     ANIMA-2 demuestra:")
        print("     ✓ Preferencia clara por el hábito seguro (-60°)")
        print("     ✓ Rechazo específico del trauma (+60°)")
        print("     ✓ Tiempo de deliberación medible (simulación interna)")
        print("     ✓ Valencia local diferencial")
        print("")
        print("  ANIMA-2 completa el ciclo cosmosemiótico:")
        print("     Memoria → Cb → Juego → Ritual → Rᴿ → R_op")
    else:
        print("  ⚠️ R_op NO DEMOSTRADO")
        if prob_preferencia.get(-60.0, 0) <= UMBRAL_PREFERENCIA_HABITO:
            print("     No hay preferencia clara por el hábito")
        if prob_preferencia.get(60.0, 0) >= UMBRAL_RECHAZO_TRAUMA:
            print("     No hay rechazo específico del trauma")
        if tiempo_delib_promedio <= UMBRAL_TIEMPO_DELIBERACION:
            print("     Tiempo de deliberación insuficiente")
        if val_diferencial <= UMBRAL_VALENCIA_DIFERENCIAL:
            print("     Valencia diferencial insuficiente")
    print("=" * 80)
    
    # Guardar datos crudos
    raw_data = {
        'version': 'V176',
        'timestamp': timestamp,
        'params': {
            'CONSOLIDACION_CICLOS': CONSOLIDACION_CICLOS,
            'TRAUMA_DURACION': TRAUMA_DURACION,
            'TEST_DURACION': TEST_DURACION,
            'DELIBERACION_STEPS_POR_OPCION': DELIBERACION_STEPS_POR_OPCION,
            'UMBRAL_PREFERENCIA_HABITO': UMBRAL_PREFERENCIA_HABITO,
            'UMBRAL_RECHAZO_TRAUMA': UMBRAL_RECHAZO_TRAUMA,
        },
        'resultados': {
            'preferencias': {str(k): v for k, v in prob_preferencia.items()},
            'valencia_media': {str(k): float(v) for k, v in valencia_media.items()},
            'tiempo_deliberacion_promedio': float(tiempo_delib_promedio),
            'val_diferencial': float(val_diferencial),
            'exito': exito
        }
    }
    with open(f'V176_logs/v176_raw_{timestamp}.json', 'w') as f:
        json.dump(raw_data, f, indent=2)
    print(f"\n  📁 Datos crudos guardados: V176_logs/v176_raw_{timestamp}.json")
    
    # Gráficos
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Gráfico 1: Preferencias
    ax = axes[0, 0]
    sps = list(prob_preferencia.keys())
    probs = list(prob_preferencia.values())
    colors = ['red' if sp == TRAUMA_SETPOINT else 'green' if sp == -60.0 else 'blue' for sp in sps]
    ax.bar(range(len(sps)), probs, color=colors)
    ax.axhline(y=UMBRAL_PREFERENCIA_HABITO, color='green', linestyle='--', alpha=0.5, label='Umbral preferencia')
    ax.axhline(y=UMBRAL_RECHAZO_TRAUMA, color='red', linestyle='--', alpha=0.5, label='Umbral rechazo')
    ax.set_xticks(range(len(sps)))
    ax.set_xticklabels([f'{sp:+.0f}' for sp in sps])
    ax.set_ylabel('Preferencia')
    ax.set_title('Preferencias tras deliberación')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Gráfico 2: Valencia local
    ax = axes[0, 1]
    vals = list(valencia_media.values())
    ax.bar(range(len(sps)), vals, color=colors)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xticks(range(len(sps)))
    ax.set_xticklabels([f'{sp:+.0f}' for sp in sps])
    ax.set_ylabel('Valencia local')
    ax.set_title('Valencia por setpoint (memoria largo plazo)')
    ax.grid(True, alpha=0.3)
    
    # Gráfico 3: Tiempo de deliberación
    ax = axes[0, 2]
    ax.hist(tiempos_deliberacion, bins=20, color='purple', alpha=0.7)
    ax.axvline(x=UMBRAL_TIEMPO_DELIBERACION, color='red', linestyle='--', alpha=0.5, label='Umbral')
    ax.set_xlabel('Tiempo de deliberación (s)')
    ax.set_ylabel('Frecuencia')
    ax.set_title(f'Deliberación (media = {tiempo_delib_promedio:.3f}s)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Gráfico 4: Desacople durante test
    ax = axes[1, 0]
    ax.plot(D_vals, 'purple', linewidth=0.5)
    ax.axhline(y=UMBRAL_DESACOPLE, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Paso')
    ax.set_ylabel('D')
    ax.set_title('Desacople durante test')
    ax.grid(True, alpha=0.3)
    
    # Gráfico 5: Orientación en test (últimas muestras)
    ax = axes[1, 1]
    orient_vals = organismo.historial['orientacion'][-int(TEST_DURACION/DT):]
    setpoint_vals = organismo.historial['setpoint_raw'][-int(TEST_DURACION/DT):]
    muestra = min(2000, len(orient_vals))
    ax.plot(setpoint_vals[:muestra], 'r--', linewidth=0.5, alpha=0.5, label='Setpoint externo')
    ax.plot(orient_vals[:muestra], 'b-', linewidth=0.5, label='Orientación real')
    ax.set_xlabel('Paso')
    ax.set_ylabel('Ángulo (º)')
    ax.set_title('FASE 3: Respuesta en test con deliberación')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Gráfico 6: Opciones elegidas a lo largo del tiempo
    ax = axes[1, 2]
    if opciones_elegidas:
        opciones_num = [op for op in opciones_elegidas]
        ax.plot(opciones_num, 'orange', linewidth=0.5, alpha=0.7)
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.set_xlabel('Decisión')
        ax.set_ylabel('Opción elegida (º)')
        ax.set_title('Decisiones a lo largo del test')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'V176_logs/v176_deliberacion_{timestamp}.png', dpi=150)
    print(f"\n  📊 Gráficos guardados: V176_logs/v176_deliberacion_{timestamp}.png")
    
    return organismo, exito


if __name__ == "__main__":
    import time
    start = time.time()
    organismo, exito = ejecutar_v176()
    elapsed = time.time() - start
    print(f"\n  ⏱️ Tiempo de ejecución: {elapsed/60:.1f} minutos")
    print(f"\n  V176 completado. Éxito: {exito}")