#!/usr/bin/env python3
"""
V182.3 — ANIMA-4: COMUNICACIÓN COMO ESTÍMULO MUTUO (CORREGIDO)
================================================================================
PRINCIPIO:
  El mensaje de otro organismo es ONTOLÓGICAMENTE IGUAL a cualquier otro estímulo.
  No hay distinción entre "entorno" y "otro organismo". Solo hay ESTÍMULOS.

CORRECCIONES APLICADAS (Qwen):
  1. recibir_estimulo ahora INYECTA la señal en los hemisferios (no está vacío)
  2. Transducción física: np.tanh(valencia) evita saturación del canal sensorial
  3. El estímulo es ESCALAR, no dict (evita semántica explícita)

FLUJO:
  1. A recibe estímulos ambientales
  2. B recibe estímulos ambientales
  3. A EMITE su estado como estímulo escalar (tanh(valencia))
  4. B recibe ESE estímulo y lo procesa como cualquier otro
  5. B EMITE su estado como estímulo escalar
  6. A recibe ESE estímulo y lo procesa
  7. REPETIR

CRITERIOS DE ÉXITO:
  ✅ Cambio en P_B(+60°) > 15% después de la comunicación
  ✅ Aumento de latencia > 10% (costo de procesar estímulos)
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json
from collections import deque
import time
import random

# ============================================================
# PARÁMETROS
# ============================================================
DT = 0.01
TIEMPO_POR_REPETICION = 452.0
REPETICIONES_LENTAS = 10
SESGO_L, SESGO_R = 0.05, -0.05
ZONA_MUERTA_BASE, ZONA_MUERTA_MAX = 2.0, 15.0
KP_BASE, KP_MIN, KP_MAX = 0.002, 0.0005, 0.005
VENTANA_OSCILACION = 100
INERCIA = 0.95
SENSIBILIDAD_GRAD = 10.0
K_GAIN, K_PRECISION, K_TEMBLOR = 0.00015, 0.002, 0.001
TAU_RECUPERACION, TAU_BASE, K_MEM = 300.0, 30.0, 0.005
SUELO_CONFIANZA, K_HOLD = 0.2, 0.001
TAU_CB, CB_MAX = 10.0, 500.0
LAMBDA_FISICO, LAMBDA_COSTO = 0.15, 0.5
UMBRAL_CB_JUEGO, K_INFLUENCIA_JUEGO = 40.0, 0.0005

SEMILLA_A, SEMILLA_B = 44, 444
HABITO_SETPOINT = -60.0
TRAUMA_SETPOINT = 60.0

# Parámetros estructurales
CONSOLIDACION_CICLOS = 20
TRAUMA_DURACION = 30.0
TRAUMA_REPETICIONES = 3
EXPOSURE_STEPS = 600
TRIAL_DURATION = EXPOSURE_STEPS * DT
N_TRIALS_BASELINE = 30
N_TRIALS_POST = 50
RONDAS_COMUNICACION = 3

# Umbrales de éxito
CAMBIO_MIN = 0.15
LATENCIA_AUMENTO_MIN = 0.10


# ============================================================
# VALENCIA LOCAL
# ============================================================
class ValenciaLocal:
    def __init__(self):
        self.valencia = {}
        self.lr = 0.001
        self.historial = {}
    
    def actualizar(self, setpoint, error, costo_pagado, dt, reward=0.0, 
                   good_th=5.0, trauma=False):
        key = round(setpoint/5)*5 if setpoint != 0 else 0
        if key not in self.valencia:
            self.valencia[key] = 0.0
            self.historial[key] = []
        
        if abs(error) < good_th:
            self.valencia[key] += reward * self.lr * dt
            if not trauma:
                self.valencia[key] += self.lr * dt * 10.0
        else:
            self.valencia[key] -= self.lr * dt * abs(error) * 0.2
        
        self.valencia[key] -= self.lr * dt * costo_pagado * 0.01
        
        if trauma:
            self.valencia[key] -= 0.5 * dt
        
        self.valencia[key] = np.clip(self.valencia[key], -100, 100)
        self.historial[key].append(self.valencia[key])
        return self.valencia[key]
    
    def get(self, setpoint):
        key = round(setpoint/5)*5 if setpoint != 0 else 0
        return self.valencia.get(key, 0.0)
    
    def set(self, setpoint, valor):
        key = round(setpoint/5)*5 if setpoint != 0 else 0
        self.valencia[key] = valor
        if key not in self.historial:
            self.historial[key] = []
        self.historial[key].append(valor)
    
    def reset(self):
        self.valencia = {}
        self.historial = {}


# ============================================================
# MEMORIA DE TRABAJO
# ============================================================
class MemoriaDeTrabajo:
    def __init__(self, steps=125):
        self.steps = steps
        self.tiempo = 0.0
        self.decision = None
    
    def deliberar(self, opciones, valencia, D_actual, current_sp=None):
        puntajes = {}
        val_w = 0.85
        explor_w = 0.15
        t_base = self.steps * DT
        
        for op in opciones:
            val = valencia.get(op)
            ex_bonus = D_actual * max(0.0, 1.0 - abs(val)/50.0) * 0.05
            cur_bonus = 0.5 if (current_sp is not None and abs(op - current_sp) < 1.0) else 0.0
            puntajes[op] = (val * val_w) + (ex_bonus * explor_w) + cur_bonus
        
        factor = 1.0 + (D_actual * 2.5)
        self.tiempo = t_base * len(opciones) * factor
        self.decision = max(puntajes, key=puntajes.get)
        return self.decision, puntajes, self.tiempo
    
    def reset(self):
        self.tiempo = 0.0
        self.decision = None


# ============================================================
# REGISTRO DE REPRESENTACIONES
# ============================================================
class RegistroRepresentaciones:
    def __init__(self, vent=200):
        self.hist_R = deque(maxlen=vent)
        self.hist_A = deque(maxlen=vent)
        self.hist_SP = deque(maxlen=vent)
    
    def registrar(self, R, A, SP):
        self.hist_R.append(R)
        self.hist_A.append(A)
        self.hist_SP.append(SP)
    
    def calcular_D_conflicto(self, val_ops):
        if len(val_ops) < 2:
            return 0.0
        vals = np.array(val_ops)
        p = np.exp(vals / 10.0) / np.sum(np.exp(vals / 10.0))
        ent = -np.sum(p * np.log(p + 1e-10))
        D_ent = ent / np.log(len(vals))
        D_amenaza = 0.4 if np.any(vals < -0.5) else 0.15
        return np.clip((D_ent * 0.6) + D_amenaza, 0.0, 1.0)
    
    def reset(self):
        self.hist_R.clear()
        self.hist_A.clear()
        self.hist_SP.clear()


# ============================================================
# FATIGA
# ============================================================
class Fatiga:
    def __init__(self):
        self.h, self.f = 0.0, 0.0
    def actualizar(self, d, c, reposo, dt):
        self.h += abs(d)
        self.f = max(0, self.f + c) if not reposo else self.f * np.exp(-dt / TAU_RECUPERACION)
        fg = np.clip(np.exp(-K_GAIN * self.f), 0.2, 1.0)
        zm = np.clip(ZONA_MUERTA_BASE + K_PRECISION * self.f, ZONA_MUERTA_BASE, ZONA_MUERTA_MAX)
        tb = np.clip(K_TEMBLOR * self.f * np.random.randn(), -3, 3)
        return fg, zm, tb
    def reset(self):
        self.h, self.f = 0.0, 0.0


# ============================================================
# MEMORIA DE AUSENCIA
# ============================================================
class MemoriaAusencia:
    def __init__(self):
        self.sp, self.t, self.tau = 0.0, 0.0, TAU_BASE
    def actualizar(self, sp, E, dt):
        if sp is not None:
            self.sp, self.t = sp, 0.0
            self.tau = TAU_BASE + K_MEM * E
            return self.sp, 1.0
        self.t += dt
        return self.sp, np.exp(-self.t / self.tau)
    def reset(self):
        self.sp, self.t, self.tau = 0.0, 0.0, TAU_BASE


# ============================================================
# CONSCIENCIA
# ============================================================
class Consciencia:
    def __init__(self):
        self.Cb = 0.0
    def actualizar(self, eR, A, dt):
        self.Cb = np.clip(self.Cb + (eR * (1 - A) - self.Cb / TAU_CB) * dt, 0, CB_MAX)
        return self.Cb
    def reset(self):
        self.Cb = 0.0


# ============================================================
# MODO JUEGO
# ============================================================
class Juego:
    def __init__(self):
        self.on = False
    def actualizar(self, Cb, conf, sp):
        self.on = (sp is not None and Cb > UMBRAL_CB_JUEGO)
        return self.on
    def apl(self, dr):
        return (dr * LAMBDA_FISICO, abs(dr) * LAMBDA_COSTO) if self.on else (dr, abs(dr))
    def reset(self):
        self.on = False


# ============================================================
# HEMISFERIO (CON ESTÍMULOS EXTERNOS Y TRANSDUCCIÓN)
# ============================================================
class Hemisferio:
    def __init__(self, n, tau, gen_f, seed, sesgo):
        np.random.seed(seed)
        self.n, self.tau, self.gen, self.sesgo = n, tau, gen_f, sesgo
        self.Phi = np.random.normal(sesgo, 0.1, 32)
        self.v = np.zeros(32)
        self.entrada = None
        self.sr = 48000
        self.estímulos_externos = deque()  # 🟢 Cola de estímulos (incluye mensajes)
    
    def añadir_estimulo(self, valor):
        """Añade un estímulo externo (del entorno o de otro organismo)"""
        self.estímulos_externos.append(valor)
    
    def _omega(self):
        return np.mean(self.Phi)
    
    def entrada_t(self, t, dur):
        # Si hay estímulos externos, procesarlos como prioridad
        if self.estímulos_externos:
            estimulo = self.estímulos_externos.popleft()
            return estimulo
        
        if self.entrada is None:
            self.entrada = self.gen(dur, self.sr)
        idx = int(t * self.sr)
        if idx >= len(self.entrada):
            return 0.0
        return self.entrada[idx]
    
    def actualizar(self, t, dt, dur, otro):
        e = self.entrada_t(t, dur)
        
        laplaciano = np.zeros_like(self.Phi)
        for i in range(1, 31):
            laplaciano[i] = self.Phi[i-1] - 2*self.Phi[i] + self.Phi[i+1]
        
        reaccion = self.Phi * (1 - self.Phi * self.Phi)
        
        forzamiento = np.zeros_like(self.Phi)
        forzamiento[0] = e
        forzamiento[-1] = -e
        
        acoplamiento = np.zeros_like(self.Phi)
        if otro is not None and abs(self._omega() - otro._omega()) > 0.5:
            acoplamiento = 0.01 * (otro.Phi - self.Phi)
        
        self.v += (laplaciano + reaccion + forzamiento + acoplamiento) * dt
        self.Phi = np.clip(self.Phi + self.v * dt, -1, 1)
        return {'omega': self._omega()}
    
    def reset(self):
        self.estímulos_externos.clear()


# ============================================================
# APARATO MOTOR V182
# ============================================================
class AparatoMotorV182:
    def __init__(self):
        self.orient, self.Kp, self.Kp_min, self.Kp_max = 0.0, KP_BASE, KP_MIN, KP_MAX
        self.lim, self.zm, self.iner = 90.0, ZONA_MUERTA_BASE, INERCIA
        self.ult_d, self.sens, self.t = 0.0, SENSIBILIDAD_GRAD, 0.0
        self.fat, self.mem, self.consc, self.juego = Fatiga(), MemoriaAusencia(), Consciencia(), Juego()
        self.val, self.mt, self.reg = ValenciaLocal(), MemoriaDeTrabajo(), RegistroRepresentaciones()
        self.mem_err = deque(maxlen=VENTANA_OSCILACION)
        self.nombre = None
    
    def set_nombre(self, nombre):
        self.nombre = nombre
    
    def recibir_estimulo(self, valor, hemisferios):
        """Inyecta el estímulo directamente en los hemisferios"""
        for h in hemisferios:
            h.añadir_estimulo(valor)
    
    def upd_plast(self, err):
        self.mem_err.append(err)
        if len(self.mem_err) >= VENTANA_OSCILACION:
            osc = np.std(self.mem_err)
            if osc > self.zm * 1.5:
                self.Kp = max(self.Kp_min, self.Kp * 0.99)
            elif osc < self.zm * 0.5:
                self.Kp = min(self.Kp_max, self.Kp * 1.01)
    
    def ejec(self, ops, grad, t, dt, trauma=False, reward_setpoint=None, reward_amount=0.0):
        vals = [self.val.get(o) for o in ops]
        D = self.reg.calcular_D_conflicto(vals)
        
        if len(ops) > 1:
            op_e, _, t_del = self.mt.deliberar(ops, self.val, D, current_sp=self.orient)
        else:
            op_e = ops[0]
            t_del = self.mt.steps * DT * 0.5
        
        sp_obj, conf = self.mem.actualizar(op_e, self.fat.h, dt)
        err = sp_obj - self.orient
        eR = abs(err)
        A_sys = min(1.0, abs(self.orient) / abs(op_e)) if abs(op_e) > 0.01 else conf
        Cb = self.consc.actualizar(eR * (1 + max(0, -self.val.get(op_e)/200)), A_sys, dt)
        self.juego.actualizar(Cb, conf, op_e)
        fg, zm_ef, tem = self.fat.actualizar(0, 0, False, dt)
        
        rwd = 0.0
        if reward_setpoint is not None and abs(op_e - reward_setpoint) < 1.0 and abs(err) < zm_ef:
            rwd = reward_amount
        
        if abs(err) < zm_ef:
            self.fat.actualizar(0, 0, True, dt)
        
        self.val.actualizar(op_e, err, 0.0, dt, reward=rwd, good_th=zm_ef, trauma=trauma)
        
        dir_sign = np.sign(err)
        conf_s = min(1.0, abs(grad) * self.sens)
        fre = 1 - np.exp(-abs(err) / 30.0)
        Kp_i = self.Kp * fg * conf_s * (TAU_BASE / (TAU_BASE + 1))
        d_err = Kp_i * abs(err) * dir_sign * fre
        tor = K_HOLD * (self.mem.sp - self.orient) * conf
        dr = d_err + tor
        c_est = abs(d_err) + abs(tor)
        
        self.val.actualizar(op_e, err, c_est, dt, reward=rwd, good_th=zm_ef, trauma=trauma)
        
        d = self.iner * self.ult_d + (1 - self.iner) * dr
        self.ult_d = d
        df, dc = self.juego.apl(d)
        self.fat.actualizar(df, c_est + dc, abs(d) < 0.001 and abs(tor) < 0.001, dt)
        df += tem * dt
        self.upd_plast(err)
        self.orient = np.clip(self.orient + df, -self.lim, self.lim)
        self.t += dt
        self.reg.registrar(sp_obj, abs(df) > 0.01, op_e)
        
        return {
            'orient': self.orient, 'Cb': Cb, 'lat': t_del, 'opcion': op_e,
            'val': self.val.get(op_e), 'D': D
        }
    
    def reset(self):
        self.orient = self.ult_d = self.t = 0.0
        self.Kp = KP_BASE
        self.mem_err.clear()
        self.fat.reset()
        self.mem.reset()
        self.consc.reset()
        self.juego.reset()
        self.val.reset()
        self.mt.reset()
        self.reg.reset()


# ============================================================
# ORGANISMO V182
# ============================================================
class OrganismoV182:
    def __init__(self, seed, nombre):
        self.nombre = nombre
        
        def rosa(d, s):
            n = int(d * s)
            r = np.random.normal(0, 1, n)
            f = 1.0 / np.sqrt(np.fft.rfftfreq(n, 1/s) + 0.01)
            signal = np.fft.irfft(np.fft.rfft(r) * f)
            return signal / (np.max(np.abs(signal)) + 1e-10)
        
        def click(d, s):
            n = int(d * s)
            c = np.zeros(n)
            for _ in range(int(d * 0.5)):
                pos = int(np.random.exponential(2.0) * s)
                if pos < n:
                    c[pos] = 1.0
            return c
        
        self.L = Hemisferio("L", 30, rosa, seed, SESGO_L)
        self.R = Hemisferio("R", 300, click, seed+100, SESGO_R)
        self.BL = Hemisferio("BL", 30, rosa, seed+200, SESGO_L)
        self.BR = Hemisferio("BR", 300, click, seed+300, SESGO_R)
        self.motor = AparatoMotorV182()
        self.motor.set_nombre(nombre)
        self.hemisferios = [self.L, self.R, self.BL, self.BR]
    
    def act(self, t, dt, dur, ops, trauma=False, reward_setpoint=None, reward_amount=0.0):
        for h in self.hemisferios:
            h.actualizar(t, dt, dur, self.R if h.n in ["L", "BL"] else self.L)
        g = (self.L._omega() + self.R._omega())/2 - (self.BL._omega() + self.BR._omega())/2
        g += (self.motor.orient / 90) * 0.3 if abs(self.motor.orient) > 0.1 else 0
        return self.motor.ejec(ops, g, t, dt, trauma, reward_setpoint, reward_amount)
    
    def get_valencia(self, setpoint):
        return self.motor.val.get(setpoint)
    
    def get_Cb(self):
        return self.motor.consc.Cb
    
    def emitir_estimulo(self, setpoint):
        """EMITE su estado como estímulo escalar (con transducción física)"""
        val = self.get_valencia(setpoint)
        # 🟢 TRANSDUCCIÓN FÍSICA: tanh evita saturación del canal sensorial
        estimulo = np.tanh(val / 20.0)  # Mapea valencia a rango [-1, 1]
        return estimulo
    
    def recibir_estimulo(self, valor):
        """RECIBE un estímulo (del entorno o de otro organismo) y lo inyecta en hemisferios"""
        for h in self.hemisferios:
            h.añadir_estimulo(valor)
    
    def reset(self):
        self.motor.reset()
        for h in self.hemisferios:
            h.reset()


# ============================================================
# FUNCIONES DE PREPARACIÓN
# ============================================================
def consolidar_habito(org, setpoint, reward=2.0, ciclos=20, dt=DT):
    for _ in range(ciclos):
        for _ in range(int(80.0 / dt)):
            org.act(0, dt, 80.0, [setpoint], reward_setpoint=setpoint, reward_amount=reward)
    return org.get_valencia(setpoint)


def aplicar_trauma(org, setpoint, duracion=30.0, repeticiones=3, dt=DT):
    for _ in range(repeticiones):
        for _ in range(int(duracion / dt)):
            org.act(0, dt, duracion, [setpoint], trauma=True)
    return org.get_valencia(setpoint)


def medir_p_B(org, n_trials=30):
    """Mide P_B(+60°) sin comunicación"""
    decisiones = []
    for _ in range(n_trials):
        # Pequeña variación para evitar determinismo
        val_B = org.get_valencia(TRAUMA_SETPOINT)
        p_60 = 1.0 / (1.0 + np.exp(-val_B / 5.0))
        decision = TRAUMA_SETPOINT if random.random() < p_60 else HABITO_SETPOINT
        decisiones.append(decision)
    return sum(1 for d in decisiones if abs(d - TRAUMA_SETPOINT) < 5.0) / n_trials


# ============================================================
# EXPERIMENTO PRINCIPAL
# ============================================================
def ejecutar_v182_3():
    print("=" * 100)
    print("EXPERIMENTO V182.3 — COMUNICACIÓN COMO ESTÍMULO MUTUO")
    print("=" * 100)
    print("  PRINCIPIO:")
    print("    El mensaje de otro organismo es ONTOLÓGICAMENTE IGUAL")
    print("    a cualquier otro estímulo del entorno.")
    print("")
    print("  TRANSDUCCIÓN FÍSICA:")
    print("    estimulo = tanh(valencia / 20.0)  # Mapeo a [-1, 1]")
    print("")
    print("  CRITERIOS DE ÉXITO:")
    print(f"    ✅ Cambio en P_B(+60°) > {CAMBIO_MIN:.0%}")
    print(f"    ✅ Aumento de latencia > {LATENCIA_AUMENTO_MIN:.0%}")
    print("=" * 100)

    A = OrganismoV182(SEMILLA_A, "A")
    B = OrganismoV182(SEMILLA_B, "B")
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('V182_logs', exist_ok=True)
    
    # ============================================================
    # FASE 1: Preparación de A (trauma)
    # ============================================================
    print("\n" + "=" * 60)
    print("FASE 1: Preparación de A")
    print("=" * 60)
    
    print("  Consolidando hábito de A en -60°...")
    consolidar_habito(A, HABITO_SETPOINT, reward=2.0, ciclos=CONSOLIDACION_CICLOS)
    
    print("  Aplicando trauma a A en +60°...")
    val_A_trauma = aplicar_trauma(A, TRAUMA_SETPOINT, duracion=TRAUMA_DURACION, 
                                   repeticiones=TRAUMA_REPETICIONES)
    print(f"  Valencia de A en +60°: {val_A_trauma:.2f}")
    
    # B recibe estímulo inicial (curiosidad)
    print("\n  Preparando B...")
    B.recibir_estimulo(0.5)  # Estímulo neutro inicial
    print(f"  Valencia inicial de B en +60°: {B.get_valencia(TRAUMA_SETPOINT):.2f}")
    
    # ============================================================
    # FASE 2: Baseline — Sin comunicación
    # ============================================================
    print("\n" + "=" * 60)
    print("FASE 2: BASELINE — Sin comunicación")
    print("=" * 60)
    
    start_baseline = time.time()
    decisiones_B = []
    for tr in range(N_TRIALS_BASELINE):
        res = B.act(0, DT, TRIAL_DURATION, [HABITO_SETPOINT, TRAUMA_SETPOINT])
        decisiones_B.append(res['opcion'])
        if (tr + 1) % 10 == 0:
            print(f"    Trial {tr+1}/{N_TRIALS_BASELINE}...")
    
    p_B_sin = sum(1 for d in decisiones_B if abs(d - TRAUMA_SETPOINT) < 5.0) / N_TRIALS_BASELINE
    lat_B_sin = (time.time() - start_baseline) / N_TRIALS_BASELINE
    print(f"\n  Resultados Baseline:")
    print(f"    P_B(+60°) = {p_B_sin:.1%}")
    print(f"    Latencia media = {lat_B_sin:.3f}s")
    
    # ============================================================
    # FASE 3: Comunicación — Intercambio de estímulos
    # ============================================================
    print("\n" + "=" * 60)
    print(f"FASE 3: COMUNICACIÓN — Intercambio mutuo de estímulos ({RONDAS_COMUNICACION} rondas)")
    print("=" * 60)
    
    start_com = time.time()
    
    for ronda in range(RONDAS_COMUNICACION):
        # A emite su estado como estímulo (transducido)
        estimulo_A = A.emitir_estimulo(TRAUMA_SETPOINT)
        print(f"  Ronda {ronda+1}: A emite {estimulo_A:.3f}")
        
        # B recibe el estímulo de A
        B.recibir_estimulo(estimulo_A)
        
        # B emite su estado como estímulo
        estimulo_B = B.emitir_estimulo(TRAUMA_SETPOINT)
        print(f"           B emite {estimulo_B:.3f}")
        
        # A recibe el estímulo de B
        A.recibir_estimulo(estimulo_B)
    
    lat_com = (time.time() - start_com) / RONDAS_COMUNICACION
    print(f"\n  Latencia por ronda de comunicación: {lat_com:.3f}s")
    
    # ============================================================
    # FASE 4: Post-comunicación — Medir efecto
    # ============================================================
    print("\n" + "=" * 60)
    print("FASE 4: POST-COMUNICACIÓN — Medir efecto en B")
    print("=" * 60)
    
    # Guardar estado de B después de comunicación
    val_B_post_com = B.get_valencia(TRAUMA_SETPOINT)
    print(f"  Valencia de B después de comunicación: {val_B_post_com:.2f}")
    
    start_post = time.time()
    decisiones_B_post = []
    for tr in range(N_TRIALS_POST):
        # Restaurar estado post-comunicación en cada trial (para medir efecto puro)
        B.motor.val.set(TRAUMA_SETPOINT, val_B_post_com)
        res = B.act(0, DT, TRIAL_DURATION, [HABITO_SETPOINT, TRAUMA_SETPOINT])
        decisiones_B_post.append(res['opcion'])
        if (tr + 1) % 10 == 0:
            print(f"    Trial {tr+1}/{N_TRIALS_POST}...")
    
    p_B_con = sum(1 for d in decisiones_B_post if abs(d - TRAUMA_SETPOINT) < 5.0) / N_TRIALS_POST
    lat_B_post = (time.time() - start_post) / N_TRIALS_POST
    
    cambio = p_B_sin - p_B_con
    aumento_latencia = (lat_B_post - lat_B_sin) / lat_B_sin if lat_B_sin > 0 else 0
    
    # ============================================================
    # RESULTADOS
    # ============================================================
    print("\n" + "=" * 80)
    print("RESULTADOS V182.3 — Comunicación como estímulo mutuo")
    print("=" * 80)
    
    print(f"\n  📊 VALENCIAS:")
    print(f"     A en +60°: {val_A_trauma:.2f}")
    print(f"     B en +60° (baseline): {B.get_valencia(TRAUMA_SETPOINT):.2f}")
    print(f"     B en +60° (post-com): {val_B_post_com:.2f}")
    
    print(f"\n  📊 CONDUCTA DE B:")
    print(f"     P_B(+60°) baseline: {p_B_sin:.1%}")
    print(f"     P_B(+60°) post-com: {p_B_con:.1%}")
    print(f"     Cambio: {cambio:.1%} (>{CAMBIO_MIN:.0%}) -> {'✅' if cambio > CAMBIO_MIN else '❌'}")
    
    print(f"\n  📊 LATENCIA:")
    print(f"     Baseline: {lat_B_sin:.3f}s")
    print(f"     Post-com: {lat_B_post:.3f}s")
    print(f"     Aumento: {aumento_latencia:.1%} (>{LATENCIA_AUMENTO_MIN:.0%}) -> {'✅' if aumento_latencia > LATENCIA_AUMENTO_MIN else '❌'}")
    
    exito = (cambio > CAMBIO_MIN) and (aumento_latencia > LATENCIA_AUMENTO_MIN)
    
    print("\n" + "=" * 80)
    if exito:
        print("  ✅ COMUNICACIÓN DEMOSTRADA")
        print("")
        print("     El organismo B modificó su conducta basándose en")
        print("     los estímulos emitidos por A, sin distinción ontológica")
        print("     entre 'entorno' y 'otro organismo'.")
        print("")
        print("  → Ψ_alma > 0: Reconocimiento funcional del otro como sujeto")
    else:
        print("  ⚠️ COMUNICACIÓN NO DEMOSTRADA")
        if cambio <= CAMBIO_MIN:
            print(f"     Cambio insuficiente ({cambio:.1%} < {CAMBIO_MIN:.0%})")
        if aumento_latencia <= LATENCIA_AUMENTO_MIN:
            print(f"     Aumento de latencia insuficiente ({aumento_latencia:.1%} < {LATENCIA_AUMENTO_MIN:.0%})")
    print("=" * 80)
    
    # ============================================================
    # GRÁFICOS
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].bar(['Baseline', 'Post-comunicación'], [p_B_sin, p_B_con], 
                color=['blue', 'green'], alpha=0.7)
    axes[0].axhline(y=p_B_sin - CAMBIO_MIN, color='red', linestyle='--', 
                    label=f'Umbral ({CAMBIO_MIN:.0%})')
    axes[0].set_ylabel('P(elegir +60°)')
    axes[0].set_title('Efecto de la comunicación en B')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].bar(['Baseline', 'Post-comunicación'], [lat_B_sin, lat_B_post], 
                color=['blue', 'orange'], alpha=0.7)
    axes[1].axhline(y=lat_B_sin * (1 + LATENCIA_AUMENTO_MIN), color='red', linestyle='--', 
                    label=f'Umbral (+{LATENCIA_AUMENTO_MIN:.0%})')
    axes[1].set_ylabel('Latencia (s)')
    axes[1].set_title('Costo cognitivo de la comunicación')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'V182_logs/v182_3_comunicacion_{ts}.png', dpi=150)
    print(f"\n  📊 Gráficos guardados: V182_logs/v182_3_comunicacion_{ts}.png")
    
    # ============================================================
    # GUARDAR DATOS
    # ============================================================
    raw_data = {
        'version': 'V182.3',
        'timestamp': ts,
        'params': {
            'RONDAS_COMUNICACION': RONDAS_COMUNICACION,
            'CAMBIO_MIN': CAMBIO_MIN,
            'LATENCIA_AUMENTO_MIN': LATENCIA_AUMENTO_MIN,
        },
        'resultados': {
            'val_A_trauma': float(val_A_trauma),
            'val_B_post_com': float(val_B_post_com),
            'p_B_sin': float(p_B_sin),
            'p_B_con': float(p_B_con),
            'cambio': float(cambio),
            'lat_B_sin': float(lat_B_sin),
            'lat_B_post': float(lat_B_post),
            'aumento_latencia': float(aumento_latencia),
            'exito': bool(exito)
        }
    }
    
    with open(f'V182_logs/v182_3_raw_{ts}.json', 'w') as f:
        json.dump(raw_data, f, indent=2)
    print(f"  📁 Datos guardados: V182_logs/v182_3_raw_{ts}.json")
    
    return exito, data


if __name__ == "__main__":
    start = time.time()
    exito, data = ejecutar_v182_3()
    elapsed = time.time() - start
    print(f"\n⏱️ Tiempo: {elapsed/60:.1f} min | Éxito: {exito}")