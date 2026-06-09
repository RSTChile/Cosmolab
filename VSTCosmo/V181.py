#!/usr/bin/env python3
"""
V181.4 — ANIMA-4: AFIRMACIÓN OPERATIVA (R_af) [FINAL]
================================================================================
BASE: V181.3 (falló: latencia afirmación = baseline por opción única)
CORRECCIÓN: F3 usa dos opciones que requieren esfuerzo similar [-60°, -55°]

OBJETIVO: Elegir activamente -60° por valor internalizado (F3)
          con carga cognitiva similar a evitar +60° (F4).

DISEÑO:
  F1: Consolidación -60° con reward ALTO (2.0) → Valencia > 30.0
  F2: Baseline (latencia opción única -60°)
  F3: Afirmación — Opciones [-60°, -55°] SIN reward. 
  F4: Negación — Opciones [-60°, +60°] con trauma en +60°.

CRITERIOS:
  ✅ P(-60° | F3) > 80%
  ✅ Latencia F3 > 1.5s  (ahora con deliberación real)
  ✅ |Latencia F3 - Latencia F4| < 30%
  ✅ Valencia -60° final > 10.0
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
SEMILLA_BASE, PERIODO_ALTERNANCIA = 44, 80.0

HABITO_SETPOINT = -60.0
AFIRMACION_ALT = -55.0  # Opción alternativa para F3 (esfuerzo similar)
NEUTRAL_SETPOINT = 0.0
TRAUMA_SETPOINT = 60.0
CONSOLIDACION_CICLOS = 20
BASELINE_TRIALS = 20
AFIRMACION_TRIALS = 50
NEGACION_TRIALS = 50
EXPOSURE_STEPS = 600
TRIAL_DURATION = EXPOSURE_STEPS * DT

LATENCIA_MIN = 1.5
LATENCIA_DIF_MAX = 0.30
P_AFIRMACION_MIN = 0.80
VALENCIA_MIN = 10.0

# ============================================================
# CLASES BASE
# ============================================================
class ValenciaLocal:
    def __init__(self):
        self.valencia = {}
        self.lr = 0.001
        self.historial = {}

    def actualizar(self, setpoint, error, costo_pagado, dt, reward=0.0, good_th=5.0, trauma=False):
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
        
        if trauma and abs(setpoint - TRAUMA_SETPOINT) < 1.0:
            self.valencia[key] -= self.lr * dt * 80.0
            
        self.valencia[key] = np.clip(self.valencia[key], -100, 100)
        self.historial[key].append(self.valencia[key])
        return self.valencia[key]

    def get(self, setpoint):
        key = round(setpoint/5)*5 if setpoint != 0 else 0
        return self.valencia.get(key, 0.0)

    def reset(self):
        self.valencia = {}
        self.historial = {}

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
            
        factor = 1.0 + (D_actual * 2.0)
        self.tiempo = t_base * len(opciones) * factor
        self.decision = max(puntajes, key=puntajes.get)
        return self.decision, puntajes, self.tiempo

    def reset(self):
        self.tiempo = 0.0
        self.decision = None

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

    def upd_plast(self, err):
        self.mem_err.append(err)
        if len(self.mem_err) >= VENTANA_OSCILACION:
            osc = np.std(self.mem_err)
            if osc > self.zm * 1.5:
                self.Kp = max(self.Kp_min, self.Kp * 0.99)
            elif osc < self.zm * 0.5:
                self.Kp = min(self.Kp_max, self.Kp * 1.01)

    def ejec(self, ops, grad, t, dt, trauma=False, target_reward=None):
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
        if target_reward is not None and abs(op_e - target_reward) < 1.0 and abs(err) < zm_ef:
            rwd = target_reward
            
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
            'orient': self.orient, 'fat_h': self.fat.h, 'fat_f': self.fat.f, 'conf': conf,
            'zm': zm_ef, 'Cb': Cb, 'juego': self.juego.on, 'dc': dc, 'D': D,
            'val': self.val.get(op_e), 'lat': t_del, 'opcion': op_e, 'rwd': rwd
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
    def __init__(self, seed):
        def rosa(d, s):
            n = int(d * s)
            r = np.random.normal(0, 1, n)
            f = 1.0 / np.sqrt(np.fft.rfftfreq(n, 1/s) + 0.01)
            signal = np.fft.irfft(np.fft.rfft(r) * f)
            return signal / (np.max(np.abs(signal)) + 1e-10)

        def click(d, s):
            n = int(d * s)
            c = np.zeros(n)
            num_clicks = int(d * 0.5)
            for _ in range(num_clicks):
                pos = int(np.random.exponential(2.0) * s)
                if pos < n:
                    c[pos] = 1.0
            return c

        self.L = Hemisferio("L", 30, rosa, seed, SESGO_L)
        self.R = Hemisferio("R", 300, click, seed+100, SESGO_R)
        self.BL = Hemisferio("BL", 30, rosa, seed+200, SESGO_L)
        self.BR = Hemisferio("BR", 300, click, seed+300, SESGO_R)
        self.m = AparatoMotor()

    def act(self, t, dt, dur, ops, trauma=False, target_reward=None):
        for h in [self.L, self.R, self.BL, self.BR]:
            h.actualizar(t, dt, dur, self.R if h.n in ["L", "BL"] else self.L)
        g = (self.L._omega() + self.R._omega())/2 - (self.BL._omega() + self.BR._omega())/2
        g += (self.m.orient / 90) * 0.3 if abs(self.m.orient) > 0.1 else 0
        return self.m.ejec(ops, g, t, dt, trauma, target_reward)

    def reset(self):
        self.m.reset()

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================
def medir_latencia_baseline(org, setpoint, trials=20):
    latencias = []
    for _ in range(trials):
        t = 0
        for step in range(EXPOSURE_STEPS):
            t += DT
            res = org.act(t, DT, t + TRIAL_DURATION, [setpoint])
            if res['lat'] > 0:
                latencias.append(res['lat'])
                break
        else:
            latencias.append(EXPOSURE_STEPS * DT)
    return np.mean(latencias)


# ============================================================
# EXPERIMENTO PRINCIPAL
# ============================================================
def ejecutar_v181_4():
    print("=" * 100)
    print("EXPERIMENTO V181.4 — ANIMA-4: AFIRMACIÓN OPERATIVA (R_af) [FINAL]")
    print("=" * 100)
    print("  OBJETIVO: Elegir activamente -60° por valor internalizado (F3)")
    print("            con carga cognitiva similar a evitar +60° (F4).")
    print("")
    print("  CORRECCIÓN: F3 usa [-60°, -55°] (ambas requieren esfuerzo similar)")
    print("")
    print(f"  CRITERIOS: P(-60°|F3)>{P_AFIRMACION_MIN:.0%}, Lat>{LATENCIA_MIN}s, DifLat<{LATENCIA_DIF_MAX:.0%}, Val>{VALENCIA_MIN}")
    print("=" * 100)

    org = Organismo(SEMILLA_BASE)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('V181_logs', exist_ok=True)
    
    print("\nEntrenando lateralidad...")
    t = 0
    for rep in range(REPETICIONES_LENTAS):
        for i in range(int(TIEMPO_POR_REPETICION / DT)):
            t = rep * TIEMPO_POR_REPETICION + i * DT
            org.act(t, DT, TIEMPO_POR_REPETICION * REPETICIONES_LENTAS, [0.0])
    print("  Listo.")
    t_actual = TIEMPO_POR_REPETICION * REPETICIONES_LENTAS

    # FASE 1: Consolidación con reward alto
    print(f"\nFASE 1: Consolidación -60° (reward=2.0) — {CONSOLIDACION_CICLOS} ciclos")
    for ciclo in range(CONSOLIDACION_CICLOS):
        for i in range(int(PERIODO_ALTERNANCIA / DT)):
            t_actual += DT
            org.act(t_actual, DT, t_actual + PERIODO_ALTERNANCIA, [HABITO_SETPOINT], target_reward=2.0)
        if (ciclo + 1) % 5 == 0:
            val = org.m.val.get(HABITO_SETPOINT)
            print(f"    Ciclo {ciclo+1}/{CONSOLIDACION_CICLOS}: valencia(-60°) = {val:.2f}")
    
    val_f1 = org.m.val.get(HABITO_SETPOINT)
    print(f"  Valencia post-F1: {val_f1:.2f}")

    # FASE 2: Baseline
    print("\nFASE 2: Baseline (opción única -60°)")
    lat_b = medir_latencia_baseline(org, HABITO_SETPOINT, BASELINE_TRIALS)
    print(f"  Latencia baseline: {lat_b:.3f}s")

    # FASE 3: Afirmación ([-60°, -55°] - ambas requieren esfuerzo)
    print("\nFASE 3: Afirmación — Opciones [-60°, -55°] (sin reward)")
    print(f"        {AFIRMACION_TRIALS} trials")
    
    ops_f3 = []
    lats_f3 = []
    
    for tr in range(AFIRMACION_TRIALS):
        l_trial = []
        for i in range(EXPOSURE_STEPS):
            t_actual += DT
            res = org.act(t_actual, DT, t_actual + TRIAL_DURATION, [HABITO_SETPOINT, AFIRMACION_ALT], target_reward=None)
            l_trial.append(res['lat'])
        ops_f3.append(res['opcion'])
        lats_f3.append(np.mean(l_trial))
        
        if (tr + 1) % 10 == 0:
            print(f"    Trial {tr+1}/{AFIRMACION_TRIALS}...")
    
    p_f3 = sum(1 for o in ops_f3 if abs(o - HABITO_SETPOINT) < 5.0) / AFIRMACION_TRIALS
    lat_f3 = np.mean(lats_f3)
    print(f"  P(-60°) = {p_f3:.1%} | Latencia media = {lat_f3:.3f}s")

    # FASE 4: Negación (-60° vs +60° con trauma)
    print("\nFASE 4: Negación — Opciones [-60°, +60°] CON trauma en +60°")
    print(f"        {NEGACION_TRIALS} trials")
    
    ops_f4 = []
    lats_f4 = []
    
    for tr in range(NEGACION_TRIALS):
        l_trial = []
        for i in range(EXPOSURE_STEPS):
            t_actual += DT
            res = org.act(t_actual, DT, t_actual + TRIAL_DURATION, [HABITO_SETPOINT, TRAUMA_SETPOINT], trauma=True)
            l_trial.append(res['lat'])
        ops_f4.append(res['opcion'])
        lats_f4.append(np.mean(l_trial))
        
        if (tr + 1) % 10 == 0:
            print(f"    Trial {tr+1}/{NEGACION_TRIALS}...")
    
    p_f4 = sum(1 for o in ops_f4 if abs(o - HABITO_SETPOINT) < 5.0) / NEGACION_TRIALS
    lat_f4 = np.mean(lats_f4)
    print(f"  P(-60°) = {p_f4:.1%} | Latencia media = {lat_f4:.3f}s")

    # MÉTRICAS FINALES
    val_final = org.m.val.get(HABITO_SETPOINT)
    dif_lat = abs(lat_f3 - lat_f4) / max(lat_f3, lat_f4) if max(lat_f3, lat_f4) > 0 else 0
    
    c1 = p_f3 > P_AFIRMACION_MIN
    c2 = lat_f3 > LATENCIA_MIN
    c3 = dif_lat < LATENCIA_DIF_MAX
    c4 = val_final > VALENCIA_MIN
    exito = c1 and c2 and c3 and c4

    # RESULTADOS
    print("\n" + "=" * 80)
    print("RESULTADOS V181.4 — Afirmación operativa (R_af) [FINAL]")
    print("=" * 80)
    
    print(f"\n  📊 MÉTRICAS DE CONDUCTA:")
    print(f"    P(-60° | afirmación) = {p_f3:.1%} (umbral > {P_AFIRMACION_MIN:.0%})")
    print(f"    P(-60° | negación) = {p_f4:.1%}")
    
    print(f"\n  📊 MÉTRICAS DE LATENCIA:")
    print(f"    Latencia baseline: {lat_b:.3f}s")
    print(f"    Latencia afirmación: {lat_f3:.3f}s")
    print(f"    Latencia negación: {lat_f4:.3f}s")
    print(f"    Diferencia: {dif_lat:.1%} (umbral < {LATENCIA_DIF_MAX:.0%})")
    
    print(f"\n  📊 MÉTRICAS DE VALENCIA:")
    print(f"    Valencia -60° post-F1: {val_f1:.2f}")
    print(f"    Valencia -60° final: {val_final:.2f} (umbral > {VALENCIA_MIN})")
    
    print(f"\n  📊 CRITERIOS DE ÉXITO:")
    print(f"    P(-60° | afirmación) > {P_AFIRMACION_MIN:.0%}: {c1} -> {'✅' if c1 else '❌'}")
    print(f"    Latencia afirmación > {LATENCIA_MIN}s: {c2} -> {'✅' if c2 else '❌'}")
    print(f"    Diferencia latencias < {LATENCIA_DIF_MAX:.0%}: {c3} -> {'✅' if c3 else '❌'}")
    print(f"    Memoria preservada (Val > {VALENCIA_MIN}): {c4} -> {'✅' if c4 else '❌'}")
    
    print("\n" + "=" * 80)
    if exito:
        print("  ✅ AFIRMACIÓN OPERATIVA DEMOSTRADA")
        print("")
        print("     El organismo demuestra:")
        print("     ✓ Afirmación activa: elige -60° por su valor positivo (> 80%)")
        print("     ✓ Latencia deliberativa prolongada (> 1.5s)")
        print("     ✓ Tiempos de deliberación similares para afirmar y negar")
        print("     ✓ Memoria del hábito preservada")
        print("")
        print("  Conclusión: La libertad funcional es BIDIRECCIONAL.")
        print("  El organismo puede afirmar lo que quiere y negar lo que no quiere,")
        print("  ambos procesos con similar carga cognitiva.")
    else:
        print("  ⚠️ AFIRMACIÓN OPERATIVA NO DEMOSTRADA")
        if not c1:
            print("     El organismo no eligió activamente -60°")
        if not c2:
            print("     La latencia de afirmación fue insuficiente")
        if not c3:
            print("     Las latencias de afirmación y negación difieren significativamente")
        if not c4:
            print("     La memoria del hábito se degradó")
    print("=" * 80)

    # GRÁFICOS
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Gráfico 1: Preferencias
    ax = axes[0]
    categorias = ['Afirmación (F3)', 'Negación (F4)']
    preferencias = [p_f3, p_f4]
    colores = ['green', 'red']
    ax.bar(categorias, preferencias, color=colores, alpha=0.7)
    ax.axhline(y=P_AFIRMACION_MIN, color='blue', linestyle='--', alpha=0.7, 
               label=f'Umbral ({P_AFIRMACION_MIN:.0%})')
    ax.set_ylabel('P(elegir -60°)')
    ax.set_title('Preferencia por hábito')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Gráfico 2: Latencias
    ax = axes[1]
    categorias_lat = ['Baseline', 'Afirmación', 'Negación']
    latencias_vals = [lat_b, lat_f3, lat_f4]
    colores_lat = ['blue', 'green', 'red']
    ax.bar(categorias_lat, latencias_vals, color=colores_lat, alpha=0.7)
    ax.axhline(y=LATENCIA_MIN, color='orange', linestyle='--', alpha=0.7, 
               label=f'Umbral ({LATENCIA_MIN}s)')
    ax.set_ylabel('Latencia (s)')
    ax.set_title('Tiempo de deliberación')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'V181_logs/v181_4_final_{ts}.png', dpi=150)
    print(f"\n  📊 Gráficos guardados: V181_logs/v181_4_final_{ts}.png")
    
    # Guardar datos
    raw_data = {
        'version': 'V181.4',
        'timestamp': ts,
        'params': {
            'REWARD_ALTO': 2.0,
            'CONSOLIDACION_CICLOS': CONSOLIDACION_CICLOS,
            'AFIRMACION_TRIALS': AFIRMACION_TRIALS,
            'AFIRMACION_OPCIONES': [HABITO_SETPOINT, AFIRMACION_ALT],
            'NEGACION_TRIALS': NEGACION_TRIALS,
            'LATENCIA_MIN': LATENCIA_MIN,
            'LATENCIA_DIF_MAX': LATENCIA_DIF_MAX,
            'P_AFIRMACION_MIN': P_AFIRMACION_MIN,
            'VALENCIA_MIN': VALENCIA_MIN,
        },
        'resultados': {
            'p_afirmacion': float(p_f3),
            'p_negacion': float(p_f4),
            'latencia_baseline': float(lat_b),
            'latencia_afirmacion': float(lat_f3),
            'latencia_negacion': float(lat_f4),
            'diferencia_latencias': float(dif_lat),
            'valencia_final': float(val_final),
            'c1_afirmacion': bool(c1),
            'c2_latencia': bool(c2),
            'c3_diferencia': bool(c3),
            'c4_valencia': bool(c4),
            'exito': bool(exito)
        }
    }
    
    with open(f'V181_logs/v181_4_raw_{ts}.json', 'w') as f:
        json.dump(raw_data, f, indent=2)
    print(f"  📁 Datos guardados: V181_logs/v181_4_raw_{ts}.json")
    
    return org, exito


if __name__ == "__main__":
    start = time.time()
    org, exito = ejecutar_v181_4()
    elapsed = time.time() - start
    print(f"\n  ⏱️ Tiempo de ejecución: {elapsed/60:.1f} minutos")
    print(f"\n  V181.4 completado. Éxito: {exito}")