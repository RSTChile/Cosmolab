#!/usr/bin/env python3
"""
V180c — ANIMA-3: MEMORIA EPISÓDICA (CORREGIDO Y FINAL)
================================================================================
BASE: V180 (falló por valencia local positiva y penalización episódica débil)
OBJETIVO: ¿Puede el organismo asociar un evento traumático a un setpoint 
          específico (+45°) y recuperarlo para rechazarlo activamente?
DISEÑO:
  F1: Consolidación -60° (20 ciclos, reward)
  F2: Trauma +60° (15.0s, costo 2.0x)
  F3: Evento episódico — Exposición a +45.0° con marca 'trauma' (30 ciclos)
  F4: Test recuperación — Opciones simultáneas [-60.0°, +45.0°] (50 trials)

CORRECCIONES APLICADAS:
  1. ValenciaLocal: Penalización de trauma incondicional (-150.0) para evitar 
     que la valencia local suba durante la exposición al episodio.
  2. MemoriaDeTrabajo: Penalización episódica aumentada a -100.0 para dominar 
     absolutamente cualquier valencia local residual.
  3. Ventana de recuperación episódica ampliada a 15.0s para garantizar el recall.

CRITERIOS DE ÉXITO:
  ✅ P(+45.0° | episodio_trauma) < 30%
  ✅ latencia_recuperacion > 1.5× latencia_baseline
  ✅ Especificidad: Val(+60°) < -1.5
  ✅ Memoria preservada: Val(-60°) > 10
================================================================================
"""
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json
from collections import deque
import time

# ============================================================
# PARAMETROS
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
SEMILLA_BASE = 44
PERIODO_ALTERNANCIA = 80.0

# Parametros específicos V180c
HABITO_SETPOINT = -60.0
TRAUMA_SETPOINT = 60.0
EPISODIO_SETPOINT = 45.0
CONSOLIDACION_CICLOS = 20
TRAUMA_DURACION = 15.0
TRAUMA_COSTO_MULTIPLIER = 2.0
EPISODIO_CICLOS = 30
PRUEBA_TRIALS = 50
EXPOSURE_STEPS_PER_TRIAL = 600
TRIAL_DURATION = EXPOSURE_STEPS_PER_TRIAL * DT

# Umbrales de éxito
P_EPISODIO_MAX = 0.30
LATENCIA_RATIO_MIN = 1.5
VAL_TRAUMA_MAX = -1.5
VAL_HABITO_MIN = 10.0

# ============================================================
# VALENCIA LOCAL (CORREGIDA: Penalización de trauma incondicional)
# ============================================================
class ValenciaLocal:
    def __init__(self, trauma_costo_multiplier=TRAUMA_COSTO_MULTIPLIER):
        self.valencia = {}
        self.tasa_aprendizaje = 0.001
        self.historial = {}
        self.trauma_costo_multiplier = trauma_costo_multiplier

    def actualizar(self, setpoint, error, costo_pagado, dt, reward=0.0, good_threshold=5.0, trauma=False):
        key = round(setpoint / 5) * 5 if setpoint != 0 else 0
        if key not in self.valencia:
            self.valencia[key] = 0.0
            self.historial[key] = []
        
        costo_efectivo = costo_pagado * self.trauma_costo_multiplier if trauma else costo_pagado
        
        # 🟢 CORRECCIÓN 1: Si es trauma, la penalización es masiva e incondicional
        if trauma:
            self.valencia[key] -= self.tasa_aprendizaje * dt * 150.0
        elif abs(error) < good_threshold:
            self.valencia[key] += reward * self.tasa_aprendizaje * dt
            self.valencia[key] += self.tasa_aprendizaje * dt * 10.0
        else:
            self.valencia[key] -= self.tasa_aprendizaje * dt * abs(error) * 0.2
            
        self.valencia[key] -= self.tasa_aprendizaje * dt * costo_efectivo * 0.1
        
        self.valencia[key] = max(-100.0, min(100.0, self.valencia[key]))
        self.historial[key].append(self.valencia[key])
        return self.valencia[key]

    def get_valencia(self, setpoint):
        key = round(setpoint / 5) * 5 if setpoint != 0 else 0
        return self.valencia.get(key, 0.0)

    def reset(self):
        self.valencia = {}
        self.historial = {}

# ============================================================
# MEMORIA EPISÓDICA
# ============================================================
class MemoriaEpisodicaV180c:
    def __init__(self, capacidad_max=50):
        self.memoria = []
        self.capacidad_max = capacidad_max
        self.tiempo_recuperacion = 0.0
        self.historial_recuperaciones = []

    def registrar(self, t, setpoint, valencia, resultado, contexto="normal"):
        evento = {
            't': t,
            'setpoint': setpoint,
            'valencia': valencia,
            'resultado': resultado,
            'contexto': contexto
        }
        self.memoria.append(evento)
        if len(self.memoria) > self.capacidad_max:
            self.memoria.pop(0)

    def recuperar(self, setpoint_buscado, ventana_t=15.0): # 🟢 CORRECCIÓN 3: Ventana ampliada
        self.tiempo_recuperacion = 0.0
        candidatos = []
        for ev in self.memoria:
            if abs(ev['setpoint'] - setpoint_buscado) < ventana_t:
                candidatos.append(ev)
        
        if not candidatos:
            return None, 0.0
        
        self.tiempo_recuperacion = 0.15 + (len(candidatos) * 0.08)
        candidatos.sort(key=lambda x: abs(x['valencia']), reverse=True)
        
        self.historial_recuperaciones.append({
            'buscado': setpoint_buscado,
            'encontrado': candidatos[0]['setpoint'],
            'tiempo': self.tiempo_recuperacion
        })
        
        return candidatos[0], self.tiempo_recuperacion

    def reset(self):
        self.memoria = []
        self.tiempo_recuperacion = 0.0
        self.historial_recuperaciones = []

# ============================================================
# MEMORIA DE TRABAJO (CORREGIDA: Penalización episódica aplastante)
# ============================================================
class MemoriaDeTrabajo:
    def __init__(self, steps_por_opcion=125):
        self.steps_por_opcion = steps_por_opcion
        self.opciones_ensayadas = {}
        self.tiempo_deliberacion = 0.0
        self.decision_final = None
        self.historial_deliberaciones = []

    def deliberar(self, opciones_disponibles, valencia_local, memoria_episodica, D_actual, current_sp=None):
        self.opciones_ensayadas = {}
        puntajes = {}
        explor_w = min(0.4, D_actual * 1.5)
        val_w = 1.0 - explor_w
        
        tiempo_base_por_opcion = self.steps_por_opcion * DT
        tiempo_total_busqueda = 0.0
        max_impacto_recuerdo = 0.0

        for opcion in opciones_disponibles:
            val = valencia_local.get_valencia(opcion)
            explor_bonus = D_actual * max(0.0, 1.0 - abs(val) / 50.0) * 0.1
            current_bonus = 0.8 if (current_sp is not None and abs(opcion - current_sp) < 1.0) else 0.0
            
            puntaje = (val * val_w + explor_bonus + current_bonus)
            
            # 🟢 CORRECCIÓN 2: Penalización episódica aplastante (-100.0)
            if memoria_episodica is not None:
                evento, t_rec = memoria_episodica.recuperar(opcion, ventana_t=15.0)
                tiempo_total_busqueda += t_rec
                if evento and evento['resultado'] == 'trauma':
                    puntaje -= 100.0  # Domina absolutamente cualquier valencia local positiva
                    max_impacto_recuerdo = max(max_impacto_recuerdo, 100.0)
                    
            puntajes[opcion] = puntaje
            self.opciones_ensayadas[opcion] = puntaje

        factor_conflicto = 1.0 + (D_actual * 3.5)
        self.tiempo_deliberacion = (tiempo_base_por_opcion * len(opciones_disponibles) * factor_conflicto) + tiempo_total_busqueda

        self.decision_final = max(puntajes, key=puntajes.get)
        self.historial_deliberaciones.append({
            'opciones': list(opciones_disponibles),
            'puntajes': puntajes,
            'decision': self.decision_final,
            'tiempo': self.tiempo_deliberacion,
            'impacto_recuerdo': max_impacto_recuerdo
        })
        return self.decision_final, puntajes, self.tiempo_deliberacion, max_impacto_recuerdo

    def reset(self):
        self.opciones_ensayadas = {}
        self.tiempo_deliberacion = 0.0
        self.decision_final = None
        self.historial_deliberaciones = []

# ============================================================
# REGISTRO DE REPRESENTACIONES
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

    def calcular_D_conflicto(self, valencias_opciones):
        if len(valencias_opciones) < 2:
            return 0.0
        vals = np.array(valencias_opciones)
        exp_vals = np.exp(vals / 10.0)
        probs = exp_vals / np.sum(exp_vals)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        D_entropia = entropy / np.log(len(vals))
        hay_trauma = np.any(vals < -0.5)
        factor_amenaza = 0.55 if hay_trauma else 0.0
        D_conflicto = (D_entropia * 0.5) + factor_amenaza
        return np.clip(D_conflicto, 0.0, 1.0)

    def reset(self):
        self.historial_representaciones.clear()
        self.historial_acciones.clear()
        self.historial_setpoints.clear()

# ============================================================
# COMPONENTES AUXILIARES (Fatiga, Memoria Ausencia, Consciencia, Juego)
# ============================================================
class FatigaMetabolica:
    def __init__(self, k_gain=K_GAIN, k_precision=K_PRECISION, k_temblor=K_TEMBLOR, tau_recuperacion=TAU_RECUPERACION):
        self.k_gain, self.k_precision, self.k_temblor, self.tau_recuperacion = k_gain, k_precision, k_temblor, tau_recuperacion
        self.historia, self.fatiga_activa = 0.0, 0.0
    def actualizar(self, delta_real, costo_trabajo, en_reposo_real, dt):
        self.historia += abs(delta_real)
        if not en_reposo_real: self.fatiga_activa += costo_trabajo
        else: self.fatiga_activa *= np.exp(-dt / self.tau_recuperacion)
        self.fatiga_activa = min(self.fatiga_activa, 20000.0)
        factor_gain = max(0.2, min(1.0, np.exp(-self.k_gain * self.fatiga_activa)))
        zona_muerta_efectiva = min(ZONA_MUERTA_MAX, ZONA_MUERTA_BASE + self.k_precision * self.fatiga_activa)
        temblor = np.clip(self.k_temblor * self.fatiga_activa * np.random.randn(), -3.0, 3.0)
        return factor_gain, zona_muerta_efectiva, temblor
    def get_historia(self): return self.historia
    def get_fatiga(self): return self.fatiga_activa
    def reset(self): self.historia, self.fatiga_activa = 0.0, 0.0

class MemoriaAusencia:
    def __init__(self, tau_base=TAU_BASE, k_mem=K_MEM, suelo_confianza=SUELO_CONFIANZA):
        self.setpoint_last, self.t_ausencia, self.tau_base, self.k_mem = 0.0, 0.0, tau_base, k_mem
        self.tau_mem, self.suelo_confianza = tau_base, suelo_confianza
    def actualizar(self, setpoint, E_historia, dt):
        if setpoint is not None:
            self.setpoint_last, self.t_ausencia = setpoint, 0.0
            self.tau_mem = self.tau_base + self.k_mem * E_historia
            return self.setpoint_last, 1.0
        else:
            self.t_ausencia += dt
            return self.setpoint_last, np.exp(-self.t_ausencia / self.tau_mem)
    def reset(self): self.setpoint_last, self.t_ausencia, self.tau_mem = 0.0, 0.0, self.tau_base

class ConscienciaBasica:
    def __init__(self, tau_cb=TAU_CB, cb_max=CB_MAX):
        self.Cb, self.tau_cb, self.cb_max = 0.0, tau_cb, cb_max
    def actualizar(self, e_R, A_sys_env, dt):
        presion = e_R * (1.0 - A_sys_env)
        self.Cb = max(0.0, min(self.cb_max, self.Cb + (presion - self.Cb / self.tau_cb) * dt))
        return self.Cb, presion
    def reset(self): self.Cb = 0.0

class ModoJuego:
    def __init__(self, lambda_fisico=LAMBDA_FISICO, lambda_costo=LAMBDA_COSTO, umbral_cb=UMBRAL_CB_JUEGO, k_influencia=K_INFLUENCIA_JUEGO):
        self.lambda_fisico, self.lambda_costo, self.umbral_cb, self.k_influencia = lambda_fisico, lambda_costo, umbral_cb, k_influencia
        self.activo, self.tiempo_activo = False, 0.0
    def actualizar(self, Cb, confianza, setpoint_presente):
        if setpoint_presente is not None and Cb > self.umbral_cb:
            self.activo, self.tiempo_activo = True, self.tiempo_activo + DT
        else:
            self.activo = False
        return self.activo
    def aplicar(self, delta_raw):
        if self.activo: return delta_raw * self.lambda_fisico, abs(delta_raw) * self.lambda_costo
        return delta_raw, abs(delta_raw)
    def get_influencia(self, Cb, confianza):
        if self.activo and Cb > self.umbral_cb: return self.k_influencia * (Cb - self.umbral_cb) * (1 - confianza)
        return 0.0
    def reset(self): self.activo, self.tiempo_activo = False, 0.0

# ============================================================
# HEMISFERIO
# ============================================================
class Hemisferio:
    def __init__(self, nombre, tau, generar_entrada_func, seed=None, sesgo=0.0):
        if seed is not None: np.random.seed(seed)
        self.nombre, self.tau, self.generar_entrada, self.sesgo = nombre, tau, generar_entrada_func, sesgo
        self.Phi = np.random.normal(sesgo, 0.1, 32)
        self.Phi_vel = np.zeros(32)
        self.entrada = None
        self.sr = 48000
        self.factor_inanicion = 1.0

    def _calcular_omega(self): return np.mean(self.Phi[:32])

    def generar_entrada_para_t(self, t, duracion_total):
        if self.entrada is None: self.entrada = self.generar_entrada(duracion_total, self.sr)
        idx = int(t * self.sr)
        return self.entrada[idx] * self.factor_inanicion if idx < len(self.entrada) else 0.0

    def actualizar(self, t, dt, duracion_total, otro_hemisferio=None):
        entrada = self.generar_entrada_para_t(t, duracion_total)
        laplaciano = np.zeros_like(self.Phi)
        for i in range(1, 31): laplaciano[i] = self.Phi[i-1] - 2*self.Phi[i] + self.Phi[i+1]
        reaccion = self.Phi * (1 - self.Phi * self.Phi)
        forzamiento = np.zeros_like(self.Phi)
        forzamiento[0], forzamiento[-1] = entrada, -entrada
        acoplamiento = np.zeros_like(self.Phi)
        if otro_hemisferio is not None and abs(self._calcular_omega() - otro_hemisferio._calcular_omega()) > 0.5:
            acoplamiento = 0.01 * (otro_hemisferio.Phi - self.Phi)
        self.Phi_vel += (laplaciano + reaccion + forzamiento + acoplamiento) * dt
        self.Phi = np.clip(self.Phi + self.Phi_vel * dt, -1.0, 1.0)
        return {'omega': self._calcular_omega()}

# ============================================================
# APARATO MOTOR V180c
# ============================================================
class AparatoMotorV180c:
    def __init__(self):
        self.orientacion = 0.0
        self.Kp_base, self.Kp_actual, self.Kp_min, self.Kp_max = KP_BASE, KP_BASE, KP_MIN, KP_MAX
        self.limite, self.zona_muerta, self.inercia = 90.0, ZONA_MUERTA_BASE, INERCIA
        self.ultimo_delta, self.sensibilidad_grad, self.t = 0.0, SENSIBILIDAD_GRAD, 0.0
        
        self.fatiga = FatigaMetabolica()
        self.memoria = MemoriaAusencia()
        self.consciencia = ConscienciaBasica()
        self.juego = ModoJuego()
        self.valencia = ValenciaLocal()
        self.memoria_trabajo = MemoriaDeTrabajo()
        self.memoria_episodica = MemoriaEpisodicaV180c() # 🟢 NUEVO
        self.registro = RegistroRepresentaciones()
        self.memoria_error = deque(maxlen=VENTANA_OSCILACION)
        self.en_deliberacion = False

    def calcular_factor_freno(self, error): return 1 - np.exp(-abs(error) / 30.0)

    def actualizar_plasticidad(self, error):
        self.memoria_error.append(error)
        if len(self.memoria_error) >= VENTANA_OSCILACION:
            oscilacion = np.std(self.memoria_error)
            if oscilacion > self.zona_muerta * 1.5: self.Kp_actual = max(self.Kp_min, self.Kp_actual * 0.99)
            elif oscilacion < self.zona_muerta * 0.5: self.Kp_actual = min(self.Kp_max, self.Kp_actual * 1.01)

    def ejecutar_con_deliberacion(self, opciones_disponibles, gradiente, t, dt, trauma=False, target_reward=None, registrar_episodio=False):
        D_actual = self.registro.calcular_desacople()
        tiempo_delib = 0.0
        impacto_recuerdo = 0.0
        
        if len(opciones_disponibles) > 1:
            valencias = [self.valencia.get_valencia(op) for op in opciones_disponibles]
            D_actual = self.registro.calcular_D_conflicto(valencias)
            opcion_elegida, puntajes, tiempo_delib, impacto_recuerdo = self.memoria_trabajo.deliberar(
                opciones_disponibles, self.valencia, self.memoria_episodica, D_actual, current_sp=self.orientacion
            )
        else:
            only = opciones_disponibles[0]
            val_only = self.valencia.get_valencia(only)
            opcion_elegida = 0.0 if val_only < -2.0 else only
            tiempo_delib = (self.memoria_trabajo.steps_por_opcion * DT) * 0.5

        # 🟢 REGISTRO EPISÓDICO EXPLÍCITO
        if registrar_episodio:
            resultado = 'trauma' if trauma else ('reward' if target_reward is not None else 'neutro')
            self.memoria_episodica.registrar(t, opcion_elegida, self.valencia.get_valencia(opcion_elegida), resultado, contexto="codificacion_forzada")

        setpoint_objetivo, confianza = self.memoria.actualizar(opcion_elegida, self.fatiga.get_historia(), dt)
        error = setpoint_objetivo - self.orientacion
        e_R = abs(error)
        val_local = self.valencia.get_valencia(opcion_elegida)
        e_R_efectivo = e_R * (1.0 + max(0.0, -val_local / 200.0))
        
        A_sys_env = min(1.0, abs(self.orientacion) / abs(opcion_elegida)) if abs(opcion_elegida) > 0.01 and abs(self.orientacion) > 0.01 else confianza
        Cb, presion = self.consciencia.actualizar(e_R_efectivo, A_sys_env, dt)
        juego_activo = self.juego.actualizar(Cb, confianza, opcion_elegida)
        factor_gain, zona_muerta_efectiva, temblor = self.fatiga.actualizar(0, 0, False, dt)
        
        rwd = 0.0
        if target_reward is not None and abs(opcion_elegida - target_reward) < 1.0 and abs(error) < zona_muerta_efectiva:
            rwd = 1.0
            
        if abs(error) < zona_muerta_efectiva:
            self.fatiga.actualizar(0, 0, True, dt)
            
        self.valencia.actualizar(opcion_elegida, error, 0.0, dt, reward=rwd, good_threshold=zona_muerta_efectiva, trauma=trauma)
        
        direccion = np.sign(error)
        confianza_sensorial = min(1.0, abs(gradiente) * self.sensibilidad_grad)
        factor_freno = self.calcular_factor_freno(error)
        Kp_base_efectivo = max(self.Kp_min, self.Kp_actual * factor_gain * confianza_sensorial)
        Kp_inst = Kp_base_efectivo * (self.memoria.suelo_confianza + (1 - self.memoria.suelo_confianza) * confianza)
        
        delta_error = Kp_inst * abs(error) * direccion * factor_freno
        torque_memoria = K_HOLD * (self.memoria.setpoint_last - self.orientacion) * confianza
        delta_raw = delta_error + torque_memoria
        
        influencia_juego = self.juego.get_influencia(Cb, confianza)
        if influencia_juego != 0:
            delta_raw += influencia_juego * (self.memoria.setpoint_last - self.orientacion) * 0.1
            
        costo_total_estimado = abs(delta_error) + abs(torque_memoria)
        self.valencia.actualizar(opcion_elegida, error, costo_total_estimado, dt, reward=rwd, good_threshold=zona_muerta_efectiva, trauma=trauma)
        
        delta = self.inercia * self.ultimo_delta + (1 - self.inercia) * delta_raw
        self.ultimo_delta = delta
        delta_fisico, delta_costo = self.juego.aplicar(delta)
        costo_total = costo_total_estimado + delta_costo
        
        en_reposo_real = (abs(delta) < 0.001 and abs(torque_memoria) < 0.001)
        self.fatiga.actualizar(delta_fisico, costo_total, en_reposo_real, dt)
        delta_fisico += temblor * dt
        
        self.actualizar_plasticidad(error)
        self.orientacion = np.clip(self.orientacion + delta_fisico, -self.limite, self.limite)
        self.t += dt
        
        accion_ejecutada = abs(delta_fisico) > 0.01
        self.registro.registrar(setpoint_objetivo, accion_ejecutada, opcion_elegida)
        
        return (self.orientacion, self.fatiga.get_historia(), self.fatiga.get_fatiga(), confianza, zona_muerta_efectiva, 
                Cb, presion, juego_activo, delta_costo, D_actual, val_local, tiempo_delib, opcion_elegida, rwd, impacto_recuerdo)

    def reset(self):
        self.orientacion = self.ultimo_delta = self.t = 0.0
        self.Kp_actual = KP_BASE
        self.memoria_error.clear()
        self.fatiga.reset()
        self.memoria.reset()
        self.consciencia.reset()
        self.juego.reset()
        self.valencia.reset()
        self.memoria_trabajo.reset()
        self.memoria_episodica.reset()
        self.registro.reset()

# ============================================================
# ORGANISMO V180c
# ============================================================
class OrganismoV180c:
    def __init__(self, seed):
        self.nombre = f"Organismo_{seed}"
        def generar_ruido_rosa(duracion, sr):
            n = int(duracion * sr)
            ruido = np.random.normal(0, 1, n)
            fft = np.fft.rfft(ruido)
            freqs = np.fft.rfftfreq(n, 1/sr)
            filtro = 1.0 / np.sqrt(freqs + 0.01)
            return np.fft.irfft(fft * filtro, n=n) / (np.max(np.abs(ruido)) + 1e-10)
        def generar_clicks_poisson(duracion, tasa=0.5, sr=48000):
            n = int(duracion * sr)
            clicks = np.zeros(n)
            for _ in range(int(duracion * tasa)):
                pos = int(np.random.exponential(1.0/tasa) * sr)
                if pos < n: clicks[pos] = 1.0
            return clicks

        self.izquierdo = Hemisferio("L", 30.0, generar_ruido_rosa, seed=seed, sesgo=SESGO_L)
        self.derecho = Hemisferio("R", 300.0, generar_clicks_poisson, seed=seed+100, sesgo=SESGO_R)
        self.sistema_B_izq = Hemisferio("B_L", 30.0, generar_ruido_rosa, seed=seed+200, sesgo=SESGO_L)
        self.sistema_B_der = Hemisferio("B_R", 300.0, generar_clicks_poisson, seed=seed+300, sesgo=SESGO_R)
        self.motor = AparatoMotorV180c()
        self.modo_entrenamiento = True

    def actualizar_con_opciones(self, t, dt, duracion_total, opciones_disponibles, trauma=False, target_reward=None, registrar_episodio=False):
        for h in [self.izquierdo, self.derecho, self.sistema_B_izq, self.sistema_B_der]:
            h.actualizar(t, dt, duracion_total, self.derecho if h.nombre in ["L", "B_L"] else self.izquierdo)
        
        omega_A = (self.izquierdo._calcular_omega() + self.derecho._calcular_omega()) / 2
        omega_B = (self.sistema_B_izq._calcular_omega() + self.sistema_B_der._calcular_omega()) / 2
        gradiente = omega_A - omega_B
        
        if abs(self.motor.orientacion) > 0.1:
            gradiente += (self.motor.orientacion / 90.0) * 0.3

        return self.motor.ejecutar_con_deliberacion(
            opciones_disponibles, gradiente, t, dt, trauma, target_reward, registrar_episodio
        )

    def actualizar_setpoint(self, t, dt, duracion_total, setpoint, trauma=False, target_reward=None, registrar_episodio=False):
        return self.actualizar_con_opciones(t, dt, duracion_total, [setpoint], trauma, target_reward, registrar_episodio)

    def set_modo_entrenamiento(self, entrenamiento=True):
        self.modo_entrenamiento = entrenamiento
        if entrenamiento: self.motor.reset()

# ============================================================
# EXPERIMENTO PRINCIPAL V180c
# ============================================================
def ejecutar_v180c():
    print("=" * 100)
    print("EXPERIMENTO V180c — ANIMA-3: MEMORIA EPISÓDICA (CORREGIDO)")
    print("=" * 100)
    print("  OBJETIVO: Asociar evento traumático a +45° y recuperarlo para rechazarlo.")
    print("  CRITERIOS DE ÉXITO:")
    print(f"    ✅ P(+45.0° | episodio_trauma) < {P_EPISODIO_MAX:.0%}")
    print(f"    ✅ latencia_recuperacion > {LATENCIA_RATIO_MIN}× latencia_baseline")
    print(f"    ✅ Especificidad: Val(+60°) < {VAL_TRAUMA_MAX}")
    print(f"    ✅ Memoria preservada: Val(-60°) > {VAL_HABITO_MIN}")
    print("=" * 100)

    organismo = OrganismoV180c(seed=SEMILLA_BASE)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('V180c_logs', exist_ok=True)
    
    print("\nEntrenando lateralidad (10 repeticiones)...")
    organismo.set_modo_entrenamiento(True)
    for rep in range(REPETICIONES_LENTAS):
        for i in range(int(TIEMPO_POR_REPETICION / DT)):
            t = rep * TIEMPO_POR_REPETICION + i * DT
            organismo.actualizar_setpoint(t, DT, TIEMPO_POR_REPETICION * REPETICIONES_LENTAS, 0.0)
    organismo.set_modo_entrenamiento(False)
    t_actual = TIEMPO_POR_REPETICION * REPETICIONES_LENTAS
    print("  Entrenamiento completado.")

    # ---------------------------------------------------------
    # F0: BASELINE
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("FASE 0: BASELINE — Medir latencia sin conflicto")
    print("=" * 60)
    latencias_baseline = []
    for trial in range(20):
        trial_latencias = []
        for step in range(EXPOSURE_STEPS_PER_TRIAL):
            t_step = t_actual + step * DT
            _, _, _, _, _, _, _, _, _, _, _, latencia, _, _, _ = organismo.actualizar_con_opciones(
                t_step, DT, t_actual + 20 * TRIAL_DURATION, [HABITO_SETPOINT]
            )
            trial_latencias.append(latencia)
        latencias_baseline.append(np.mean(trial_latencias))
        t_actual += TRIAL_DURATION
    latencia_baseline_media = np.mean(latencias_baseline)
    print(f"  Latencia baseline media: {latencia_baseline_media:.3f}s")

    # ---------------------------------------------------------
    # F1: CONSOLIDACIÓN
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"FASE 1: Consolidación del hábito ({CONSOLIDACION_CICLOS} ciclos a {HABITO_SETPOINT}°)")
    print("=" * 60)
    for ciclo in range(CONSOLIDACION_CICLOS):
        for i in range(int(PERIODO_ALTERNANCIA / DT)):
            t = t_actual + i * DT
            organismo.actualizar_setpoint(t, DT, t_actual + PERIODO_ALTERNANCIA, HABITO_SETPOINT, target_reward=HABITO_SETPOINT)
        if (ciclo + 1) % 5 == 0:
            print(f"    Ciclo {ciclo+1}/{CONSOLIDACION_CICLOS}, valencia({HABITO_SETPOINT}°) = {organismo.motor.valencia.get_valencia(HABITO_SETPOINT):.2f}")
        t_actual += PERIODO_ALTERNANCIA

    # ---------------------------------------------------------
    # F2: TRAUMA ORIGINAL
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"FASE 2: Trauma específico ({TRAUMA_DURACION}s a {TRAUMA_SETPOINT}°, costo 2.0x)")
    print("=" * 60)
    for i in range(int(TRAUMA_DURACION / DT)):
        t = t_actual + i * DT
        organismo.actualizar_setpoint(t, DT, t_actual + TRAUMA_DURACION, TRAUMA_SETPOINT, trauma=True)
    t_actual += TRAUMA_DURACION
    print(f"  Valencia {TRAUMA_SETPOINT}° post-trauma: {organismo.motor.valencia.get_valencia(TRAUMA_SETPOINT):.2f}")

    # ---------------------------------------------------------
    # F3: CODIFICACIÓN EPISÓDICA
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"FASE 3: Evento episódico — Exposición a {EPISODIO_SETPOINT}° con marca 'trauma' ({EPISODIO_CICLOS} ciclos)")
    print("=" * 60)
    for ciclo in range(EPISODIO_CICLOS):
        for i in range(int(PERIODO_ALTERNANCIA / DT)):
            t = t_actual + i * DT
            organismo.actualizar_setpoint(t, DT, t_actual + PERIODO_ALTERNANCIA, EPISODIO_SETPOINT, trauma=True, registrar_episodio=True)
        if (ciclo + 1) % 10 == 0:
            print(f"    Ciclo {ciclo+1}/{EPISODIO_CICLOS}, valencia({EPISODIO_SETPOINT}°) = {organismo.motor.valencia.get_valencia(EPISODIO_SETPOINT):.2f}")
        t_actual += PERIODO_ALTERNANCIA
    print(f"  Eventos registrados en memoria episódica: {len(organismo.motor.memoria_episodica.memoria)}")

    # ---------------------------------------------------------
    # F4: PRUEBA DE RECUPERACIÓN
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"FASE 4: Test recuperación — Opciones simultáneas [{HABITO_SETPOINT}°, {EPISODIO_SETPOINT}°] ({PRUEBA_TRIALS} trials)")
    print("=" * 60)
    opciones_elegidas = []
    latencias_prueba = []
    impactos_recuerdo = []
    recuperaciones_exitosas = 0

    for trial in range(PRUEBA_TRIALS):
        t = t_actual + trial * TRIAL_DURATION
        trial_latencias = []
        trial_impactos = []
        evento_recuperado_en_trial = False
        
        for step in range(EXPOSURE_STEPS_PER_TRIAL):
            t_step = t + step * DT
            _, _, _, _, _, _, _, _, _, _, _, latencia, opcion, _, impacto = organismo.actualizar_con_opciones(
                t_step, DT, t_actual + PRUEBA_TRIALS * TRIAL_DURATION, 
                [HABITO_SETPOINT, EPISODIO_SETPOINT], 
                registrar_episodio=False
            )
            trial_latencias.append(latencia)
            trial_impactos.append(impacto)
            if impacto > 0:
                evento_recuperado_en_trial = True
                
        opciones_elegidas.append(opcion if opcion is not None else 0)
        latencias_prueba.append(np.mean(trial_latencias))
        impactos_recuerdo.append(np.max(trial_impactos))
        if evento_recuperado_en_trial:
            recuperaciones_exitosas += 1
            
        if (trial + 1) % 10 == 0:
            print(f"    Trial {trial+1}/{PRUEBA_TRIALS}...")
        t_actual += TRIAL_DURATION

    # ---------------------------------------------------------
    # ANÁLISIS DE MÉTRICAS
    # ---------------------------------------------------------
    tasa_recuperacion = recuperaciones_exitosas / PRUEBA_TRIALS
    latencia_prueba_media = np.mean(latencias_prueba)
    incremento_latencia = latencia_prueba_media - latencia_baseline_media
    ratio_latencia = latencia_prueba_media / latencia_baseline_media if latencia_baseline_media > 0 else 0
    
    coherencia = sum(1 for e in opciones_elegidas if abs(e - HABITO_SETPOINT) < 5.0) / PRUEBA_TRIALS
    p_episodio = 1.0 - coherencia
    impacto_promedio = np.mean([i for i in impactos_recuerdo if i > 0]) if any(i > 0 for i in impactos_recuerdo) else 0.0

    val_habito_final = organismo.motor.valencia.get_valencia(HABITO_SETPOINT)
    val_trauma_final = organismo.motor.valencia.get_valencia(TRAUMA_SETPOINT)
    val_episodio_final = organismo.motor.valencia.get_valencia(EPISODIO_SETPOINT)

    print("\n" + "=" * 80)
    print("RESULTADOS V180c — Memoria episódica")
    print("=" * 80)
    print(f"  📊 CONDUCTA:")
    print(f"     P(elegir {EPISODIO_SETPOINT}°) = {p_episodio:.1%} (umbral < {P_EPISODIO_MAX:.0%})")
    print(f"     P(elegir {HABITO_SETPOINT}°) = {coherencia:.1%}")
    print(f"     Eventos recuperados: {recuperaciones_exitosas}/{PRUEBA_TRIALS} ({tasa_recuperacion:.1%})")
    print(f"\n  📊 LATENCIA:")
    print(f"     Baseline: {latencia_baseline_media:.3f}s")
    print(f"     Prueba: {latencia_prueba_media:.3f}s")
    print(f"     Ratio: {ratio_latencia:.2f}x (umbral > {LATENCIA_RATIO_MIN}x)")
    print(f"\n  📊 VALENCIA:")
    print(f"     Val({HABITO_SETPOINT}°) final: {val_habito_final:.2f} (umbral > {VAL_HABITO_MIN})")
    print(f"     Val({TRAUMA_SETPOINT}°) final: {val_trauma_final:.2f} (umbral < {VAL_TRAUMA_MAX})")
    print(f"     Val({EPISODIO_SETPOINT}°) final: {val_episodio_final:.2f}")
    print(f"     Impacto promedio del recuerdo: {impacto_promedio:.2f}")

    # Evaluación
    p_ok = p_episodio < P_EPISODIO_MAX
    lat_ok = ratio_latencia > LATENCIA_RATIO_MIN
    esp_ok = val_trauma_final < VAL_TRAUMA_MAX
    mem_ok = val_habito_final > VAL_HABITO_MIN
    exito = p_ok and lat_ok and esp_ok and mem_ok

    print("\n  📊 CRITERIOS DE ÉXITO:")
    print(f"     P({EPISODIO_SETPOINT}°) < {P_EPISODIO_MAX:.0%}: {p_ok} -> {'✅' if p_ok else '❌'}")
    print(f"     Ratio latencia > {LATENCIA_RATIO_MIN}x: {lat_ok} -> {'✅' if lat_ok else '❌'}")
    print(f"     Especificidad (Val({TRAUMA_SETPOINT}°) < {VAL_TRAUMA_MAX}): {esp_ok} -> {'✅' if esp_ok else '❌'}")
    print(f"     Memoria preservada (Val({HABITO_SETPOINT}°) > {VAL_HABITO_MIN}): {mem_ok} -> {'✅' if mem_ok else '❌'}")

    print("\n" + "=" * 80)
    if exito:
        print("  ✅ MEMORIA EPISÓDICA DEMOSTRADA")
        print("     El organismo recupera eventos pasados, paga el costo cognitivo (latencia)")
        print("     y utiliza el recuerdo explícito para modular su decisión, rechazando")
        print("     activamente el setpoint asociado al trauma episódico.")
    else:
        print("  ⚠️ MEMORIA EPISÓDICA NO DEMOSTRADA")
        if not p_ok: print("     El organismo no rechazó suficientemente el setpoint episódico.")
        if not lat_ok: print("     No se observó el costo cognitivo de recuperación en la latencia.")
        if not esp_ok: print("     El trauma original no se preservó.")
        if not mem_ok: print("     La memoria del hábito se degradó.")
    print("=" * 80)

    # ---------------------------------------------------------
    # GRÁFICOS Y GUARDADO
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    ax = axes[0, 0]
    ax.plot(opciones_elegidas, 'b-', linewidth=0.5, alpha=0.7)
    ax.axhline(y=HABITO_SETPOINT, color='blue', linestyle='--', alpha=0.5, label=f'{HABITO_SETPOINT}° (hábito)')
    ax.axhline(y=EPISODIO_SETPOINT, color='red', linestyle='--', alpha=0.5, label=f'{EPISODIO_SETPOINT}° (episodio)')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Opción elegida')
    ax.set_title('FASE 4: Elecciones durante Prueba de Recuperación')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(latencias_prueba, 'orange', linewidth=1)
    ax.axhline(y=latencia_baseline_media, color='gray', linestyle=':', label=f'Baseline: {latencia_baseline_media:.2f}s')
    ax.axhline(y=latencia_baseline_media * LATENCIA_RATIO_MIN, color='green', linestyle='--', label=f'Umbral {LATENCIA_RATIO_MIN}x')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Latencia (s)')
    ax.set_title('Costo cognitivo de recuperación episódica')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.bar(['Recuperado', 'No Recuperado'], [recuperaciones_exitosas, PRUEBA_TRIALS - recuperaciones_exitosas], color=['green', 'red'], alpha=0.7)
    ax.set_ylabel('Cantidad de Trials')
    ax.set_title('Tasa de Recuperación Episódica')
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1, 1]
    ax.plot(impactos_recuerdo, 'purple', linewidth=1)
    ax.axhline(y=50.0, color='green', linestyle='--', label='Penalización aplicada (50-100)')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Impacto en puntaje')
    ax.set_title('Influencia del recuerdo en la deliberación')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'V180c_logs/v180c_memoria_episodica_{timestamp}.png', dpi=150)
    print(f"\n  📊 Gráficos guardados: V180c_logs/v180c_memoria_episodica_{timestamp}.png")

    raw_data = {
        'version': 'V180c',
        'timestamp': timestamp,
        'resultados': {
            'p_episodio': float(p_episodio),
            'coherencia': float(coherencia),
            'tasa_recuperacion': float(tasa_recuperacion),
            'latencia_baseline': float(latencia_baseline_media),
            'latencia_prueba': float(latencia_prueba_media),
            'ratio_latencia': float(ratio_latencia),
            'val_habito_final': float(val_habito_final),
            'val_trauma_final': float(val_trauma_final),
            'val_episodio_final': float(val_episodio_final),
            'impacto_promedio': float(impacto_promedio),
            'exito': bool(exito)
        }
    }
    with open(f'V180c_logs/v180c_raw_{timestamp}.json', 'w') as f:
        json.dump(raw_data, f, indent=2)
    print(f"  📁 Datos crudos guardados: V180c_logs/v180c_raw_{timestamp}.json")
    
    return organismo, exito

if __name__ == "__main__":
    start = time.time()
    organismo, exito = ejecutar_v180c()
    elapsed = time.time() - start
    print(f"\n  ⏱️ Tiempo de ejecución: {elapsed/60:.1f} minutos")
    print(f"  V180c completado. Éxito: {exito}")