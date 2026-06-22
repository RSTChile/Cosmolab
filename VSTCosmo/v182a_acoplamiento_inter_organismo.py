#!/usr/bin/env python3
"""
V182A — ANIMA-4: ACOPLAMIENTO INTER-ORGANISMO
================================================================================
OBJETIVO: ¿El estado de A altera la dinámica de B sin que haya comunicación explícita?
HIPÓTESIS:
  - Acoplamiento básico: La presencia de A crea una perturbación en B
  - B no procesa información de A como "señal", solo como "ruido" o "fuerza"
  - Efecto observable: D, latencia, valencia se modifican por la presencia de A

CRITERIOS DE ÉXITO:
  ✅ |ΔD_coupled| > 0.15
  ✅ Corr(A_estado, ΔB) > 0.25
  ✅ Cambios revierten en F3 (decoupling)
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
NEUTRAL_SETPOINT = 0.0

# Parámetros estructurales
CONSOLIDACION_CICLOS = 20
EXPOSURE_STEPS = 600
TRIAL_DURATION = EXPOSURE_STEPS * DT
BASELINE_TRIALS = 20
COUPLED_TRIALS = 50
DECOUPLING_TRIALS = 20

# Umbrales de éxito
DELTA_D_MIN = 0.15
CORR_MIN = 0.25


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
# COMPONENTES AUXILIARES
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

class Consciencia:
    def __init__(self):
        self.Cb = 0.0
    def actualizar(self, eR, A, dt):
        self.Cb = np.clip(self.Cb + (eR * (1 - A) - self.Cb / TAU_CB) * dt, 0, CB_MAX)
        return self.Cb
    def reset(self):
        self.Cb = 0.0

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
# HEMISFERIO
# ============================================================
class Hemisferio:
    def __init__(self, n, tau, gen_f, seed, sesgo):
        np.random.seed(seed)
        self.n, self.tau, self.gen, self.sesgo = n, tau, gen_f, sesgo
        self.Phi = np.random.normal(sesgo, 0.1, 32)
        self.v = np.zeros(32)
        self.entrada = None
        self.sr = 48000

    def _omega(self):
        return np.mean(self.Phi)

    def entrada_t(self, t, dur):
        if self.entrada is None:
            self.entrada = self.gen(dur, self.sr)
        i = int(t * self.sr)
        return self.entrada[i] if i < len(self.entrada) else 0.0

    def actualizar(self, t, dt, dur, otro):
        e = self.entrada_t(t, dur)
        lap = np.zeros_like(self.Phi)
        for i in range(1, 31):
            lap[i] = self.Phi[i-1] - 2*self.Phi[i] + self.Phi[i+1]
        ac = 0.01 * (otro.Phi - self.Phi) if otro and abs(self._omega() - otro._omega()) > 0.5 else 0.0
        forz = np.zeros_like(self.Phi)
        forz[0], forz[-1] = e, -e
        self.v += (lap + self.Phi * (1 - self.Phi**2) + forz + ac) * dt
        self.Phi = np.clip(self.Phi + self.v * dt, -1, 1)
        return {'omega': self._omega()}
    
    def reset(self):
        self.Phi = np.random.normal(self.sesgo, 0.1, 32)
        self.v = np.zeros(32)
        self.entrada = None


# ============================================================
# APARATO MOTOR
# ============================================================
class AparatoMotor:
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
# ORGANISMO
# ============================================================
class Organismo:
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
        self.motor = AparatoMotor()
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
    
    def get_D(self):
        """Retorna el desacople actual del organismo"""
        if hasattr(self.motor, 'reg') and hasattr(self.motor.reg, 'calcular_D_conflicto'):
            # Para simplificar, usamos una aproximación
            val_habito = self.get_valencia(HABITO_SETPOINT)
            val_trauma = self.get_valencia(TRAUMA_SETPOINT)
            return self.motor.reg.calcular_D_conflicto([val_habito, val_trauma])
        return 0.0
    
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


def medir_baseline(org, setpoints, n_trials=20):
    """Mide D, latencia y valencia del organismo solo"""
    D_vals = []
    latencias = []
    opciones = []
    
    for tr in range(n_trials):
        t = tr * TRIAL_DURATION
        for step in range(EXPOSURE_STEPS):
            t_step = t + step * DT
            res = org.act(t_step, DT, TRIAL_DURATION, setpoints)
        D_vals.append(res['D'])
        latencias.append(res['lat'])
        opciones.append(res['opcion'])
    
    return {
        'D_mean': np.mean(D_vals),
        'D_std': np.std(D_vals),
        'lat_mean': np.mean(latencias),
        'lat_std': np.std(latencias),
        'opciones': opciones,
        'valencia_habito': org.get_valencia(HABITO_SETPOINT),
        'valencia_trauma': org.get_valencia(TRAUMA_SETPOINT)
    }


def medir_acoplado(A, B, setpoints, n_trials=50):
    """Mide A y B juntos, registrando estados de ambos"""
    D_A_vals = []
    D_B_vals = []
    lat_A_vals = []
    lat_B_vals = []
    opciones_A = []
    opciones_B = []
    estados_A = []
    
    for tr in range(n_trials):
        t = tr * TRIAL_DURATION
        for step in range(EXPOSURE_STEPS):
            t_step = t + step * DT
            res_A = A.act(t_step, DT, TRIAL_DURATION, setpoints)
            res_B = B.act(t_step, DT, TRIAL_DURATION, setpoints)
        
        D_A_vals.append(res_A['D'])
        D_B_vals.append(res_B['D'])
        lat_A_vals.append(res_A['lat'])
        lat_B_vals.append(res_B['lat'])
        opciones_A.append(res_A['opcion'])
        opciones_B.append(res_B['opcion'])
        estados_A.append({
            'D': res_A['D'],
            'val_habito': A.get_valencia(HABITO_SETPOINT),
            'val_trauma': A.get_valencia(TRAUMA_SETPOINT)
        })
    
    return {
        'A': {
            'D_mean': np.mean(D_A_vals),
            'D_std': np.std(D_A_vals),
            'lat_mean': np.mean(lat_A_vals),
            'opciones': opciones_A
        },
        'B': {
            'D_mean': np.mean(D_B_vals),
            'D_std': np.std(D_B_vals),
            'lat_mean': np.mean(lat_B_vals),
            'opciones': opciones_B
        },
        'estados_A': estados_A,
        'correlacion': np.corrcoef(D_A_vals, D_B_vals)[0, 1] if len(D_A_vals) > 1 else 0.0
    }


# ============================================================
# EXPERIMENTO PRINCIPAL V182A
# ============================================================
def ejecutar_v182a():
    print("=" * 100)
    print("EXPERIMENTO V182A — ACOPLAMIENTO INTER-ORGANISMO")
    print("=" * 100)
    print("  OBJETIVO: ¿El estado de A altera la dinámica de B?")
    print("")
    print("  DISEÑO:")
    print("    F1: Baseline — A solo, B solo (20 trials cada uno)")
    print("    F2: Acoplamiento — A y B juntos (50 trials)")
    print("    F3: Decoupling — Separados nuevamente (20 trials)")
    print("")
    print("  CRITERIOS DE ÉXITO:")
    print(f"    ✅ |ΔD_coupled| > {DELTA_D_MIN}")
    print(f"    ✅ Corr(A_estado, ΔB) > {CORR_MIN}")
    print("=" * 100)

    # Inicializar organismos
    A = Organismo(SEMILLA_A, "A")
    B = Organismo(SEMILLA_B, "B")
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('V182_logs', exist_ok=True)
    
    # ============================================================
    # PREPARACIÓN (unificar historias)
    # ============================================================
    print("\n" + "=" * 60)
    print("PREPARACIÓN: Consolidación de hábitos")
    print("=" * 60)
    
    print("  Consolidando hábito de A en -60°...")
    consolidar_habito(A, HABITO_SETPOINT, reward=2.0, ciclos=CONSOLIDACION_CICLOS)
    print(f"    Valencia A(-60°) = {A.get_valencia(HABITO_SETPOINT):.2f}")
    
    print("  Consolidando hábito de B en -60°...")
    consolidar_habito(B, HABITO_SETPOINT, reward=2.0, ciclos=CONSOLIDACION_CICLOS)
    print(f"    Valencia B(-60°) = {B.get_valencia(HABITO_SETPOINT):.2f}")
    
    # ============================================================
    # F1: BASELINE (A solo, B solo)
    # ============================================================
    print("\n" + "=" * 60)
    print("FASE 1: BASELINE — Organismos solos")
    print("=" * 60)
    
    print("  Midiendo A solo...")
    baseline_A = medir_baseline(A, [HABITO_SETPOINT, TRAUMA_SETPOINT], BASELINE_TRIALS)
    print(f"    D_A = {baseline_A['D_mean']:.3f}, Lat_A = {baseline_A['lat_mean']:.3f}s")
    
    print("  Midiendo B solo...")
    baseline_B = medir_baseline(B, [HABITO_SETPOINT, TRAUMA_SETPOINT], BASELINE_TRIALS)
    print(f"    D_B = {baseline_B['D_mean']:.3f}, Lat_B = {baseline_B['lat_mean']:.3f}s")
    
    # ============================================================
    # F2: ACOPLAMIENTO (A y B juntos)
    # ============================================================
    print("\n" + "=" * 60)
    print(f"FASE 2: ACOPLAMIENTO — A y B juntos ({COUPLED_TRIALS} trials)")
    print("=" * 60)
    
    coupled = medir_acoplado(A, B, [HABITO_SETPOINT, TRAUMA_SETPOINT], COUPLED_TRIALS)
    
    print(f"  Resultados acoplados:")
    print(f"    D_A = {coupled['A']['D_mean']:.3f} (Δ = {coupled['A']['D_mean'] - baseline_A['D_mean']:.3f})")
    print(f"    D_B = {coupled['B']['D_mean']:.3f} (Δ = {coupled['B']['D_mean'] - baseline_B['D_mean']:.3f})")
    print(f"    Correlación D(A,B) = {coupled['correlacion']:.3f}")
    
    # ============================================================
    # F3: DECOUPLING (separados nuevamente)
    # ============================================================
    print("\n" + "=" * 60)
    print(f"FASE 3: DECOUPLING — Separados nuevamente ({DECOUPLING_TRIALS} trials)")
    print("=" * 60)
    
    print("  Midiendo B solo después del acoplamiento...")
    decoupling_B = medir_baseline(B, [HABITO_SETPOINT, TRAUMA_SETPOINT], DECOUPLING_TRIALS)
    print(f"    D_B = {decoupling_B['D_mean']:.3f}")
    
    # ============================================================
    # ANÁLISIS
    # ============================================================
    delta_D = abs(coupled['B']['D_mean'] - baseline_B['D_mean'])
    reversibilidad = abs(decoupling_B['D_mean'] - baseline_B['D_mean']) < 0.1
    
    exito_d = delta_D > DELTA_D_MIN
    exito_corr = coupled['correlacion'] > CORR_MIN
    exito_rev = reversibilidad
    
    exito = exito_d and exito_corr and exito_rev
    
    # ============================================================
    # RESULTADOS
    # ============================================================
    print("\n" + "=" * 80)
    print("RESULTADOS V182A — Acoplamiento Inter-Organismo")
    print("=" * 80)
    
    print(f"\n  📊 MÉTRICAS DE ACOPLAMIENTO:")
    print(f"     ΔD_B = {delta_D:.3f} (>{DELTA_D_MIN:.2f}) -> {'✅' if exito_d else '❌'}")
    print(f"     Corr(D_A, D_B) = {coupled['correlacion']:.3f} (>{CORR_MIN}) -> {'✅' if exito_corr else '❌'}")
    print(f"     Reversibilidad: {'✅' if exito_rev else '❌'}")
    
    print(f"\n  📊 DATOS DETALLADOS:")
    print(f"     Baseline A: D={baseline_A['D_mean']:.3f}, Lat={baseline_A['lat_mean']:.3f}s")
    print(f"     Baseline B: D={baseline_B['D_mean']:.3f}, Lat={baseline_B['lat_mean']:.3f}s")
    print(f"     Acoplado A: D={coupled['A']['D_mean']:.3f}, Lat={coupled['A']['lat_mean']:.3f}s")
    print(f"     Acoplado B: D={coupled['B']['D_mean']:.3f}, Lat={coupled['B']['lat_mean']:.3f}s")
    print(f"     Decoupling B: D={decoupling_B['D_mean']:.3f}")
    
    print("\n" + "=" * 80)
    if exito:
        print("  ✅ ACOPLAMIENTO DEMOSTRADO")
        print("")
        print("     El organismo B modifica su dinámica interna")
        print("     (desacople D) cuando A está presente.")
        print("")
        print("  PRÓXIMO: V182B — Comunicación")
    else:
        print("  ⚠️ ACOPLAMIENTO NO DEMOSTRADO")
        if not exito_d:
            print("     El desacople de B no cambió significativamente")
        if not exito_corr:
            print("     No hubo correlación entre estados de A y B")
        if not exito_rev:
            print("     Los cambios no revirtieron al separarlos")
    print("=" * 80)
    
    # ============================================================
    # GRÁFICOS
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Gráfico 1: Comparación de D
    axes[0].bar(['Baseline B', 'Acoplado B', 'Decoupling B'], 
                [baseline_B['D_mean'], coupled['B']['D_mean'], decoupling_B['D_mean']],
                color=['blue', 'orange', 'green'], alpha=0.7,
                yerr=[baseline_B['D_std'], coupled['B']['D_std'], decoupling_B['D_std']])
    axes[0].axhline(y=baseline_B['D_mean'] + DELTA_D_MIN, color='red', linestyle='--',
                    label=f'Umbral ΔD > {DELTA_D_MIN}')
    axes[0].set_ylabel('Desacople (D)')
    axes[0].set_title('Cambio en D de B por presencia de A')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Gráfico 2: Correlación D_A vs D_B
    # Simular datos para el scatter (usando los valores medios)
    axes[1].scatter([baseline_A['D_mean']], [baseline_B['D_mean']], 
                    color='blue', s=100, label='Baseline', alpha=0.7)
    axes[1].scatter([coupled['A']['D_mean']], [coupled['B']['D_mean']], 
                    color='orange', s=100, label='Acoplado', alpha=0.7)
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Identidad')
    axes[1].set_xlabel('D de A')
    axes[1].set_ylabel('D de B')
    axes[1].set_title(f'Correlación D(A,B) = {coupled["correlacion"]:.3f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'V182_logs/v182a_acoplamiento_{ts}.png', dpi=150)
    print(f"\n  📊 Gráficos guardados: V182_logs/v182a_acoplamiento_{ts}.png")
    
    # Guardar datos
    raw_data = {
        'version': 'V182A',
        'timestamp': ts,
        'params': {
            'BASELINE_TRIALS': BASELINE_TRIALS,
            'COUPLED_TRIALS': COUPLED_TRIALS,
            'DECOUPLING_TRIALS': DECOUPLING_TRIALS,
            'DELTA_D_MIN': DELTA_D_MIN,
            'CORR_MIN': CORR_MIN,
        },
        'resultados': {
            'baseline_A_D': float(baseline_A['D_mean']),
            'baseline_B_D': float(baseline_B['D_mean']),
            'coupled_A_D': float(coupled['A']['D_mean']),
            'coupled_B_D': float(coupled['B']['D_mean']),
            'decoupling_B_D': float(decoupling_B['D_mean']),
            'delta_D': float(delta_D),
            'correlacion': float(coupled['correlacion']),
            'reversibilidad': bool(exito_rev),
            'exito_d': bool(exito_d),
            'exito_corr': bool(exito_corr),
            'exito': bool(exito)
        }
    }
    
    with open(f'V182_logs/v182a_raw_{ts}.json', 'w') as f:
        json.dump(raw_data, f, indent=2)
    print(f"  📁 Datos guardados: V182_logs/v182a_raw_{ts}.json")
    
    return exito


if __name__ == "__main__":
    start = time.time()
    exito = ejecutar_v182a()
    elapsed = time.time() - start
    print(f"\n  ⏱️ Tiempo: {elapsed/60:.1f} min | Éxito: {exito}")