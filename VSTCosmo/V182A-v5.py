#!/usr/bin/env python3
"""
V182A-v5 — ACOPLAMIENTO BIDIRECCIONAL CON REFORZAMIENTO (FINAL)
================================================================================
PRINCIPIOS (consenso equipo):
  1. 100 iteraciones para historia compartida (como niño aprendiendo)
  2. Reforzamiento = ganancia mutua de viabilidad (NO es Shannon oculto)
  3. Organismos completos: Cb, D reales, hemisferios funcionales
  4. Parada temprana si diferencia < 5.0
  5. Sincronización de dinámicas internas como métrica

CRITERIOS DE ÉXITO:
  ✅ Reducción de diferencia > 40%
  ✅ Diferencia final < 20.0
  ✅ Estabilización (std últimas 10) < 5.0
  ✅ Simetría: ambos se movieron > 12.0
  ✅ Correlación Cb(A,B) > 0.3 (sincronización emergente)
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
DIM_HEMISFERIO = 32
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

# Parámetros de acoplamiento (consenso)
RONDAS_ACP = 100                    # 100 iteraciones
DT_PROCESAMIENTO = 1.0              # 1s por ronda → 100s total
PESO_ESTIMULO = 0.05                # Aprendizaje gradual
TASA_APRENDIZAJE = 0.0003
REWARD_BASE = 0.3                   # Recompensa por punto de reducción
ESCALA_REWARD = 10.0                # Factor de escalado (Meta)
PARADA_TEMPRANA_DIF = 5.0           # Si diferencia < 5, convergió
RANGO_VALENCIA_INICIAL = 50.0

# Umbrales de éxito
REDUCCION_MIN = 0.40
DIFERENCIA_FINAL_MAX = 20.0
ESTABILIZACION_MAX = 5.0
MOVIMIENTO_MIN = 12.0
CORRELACION_CB_MIN = 0.30
CORRELACION_D_MIN = 0.30


# ============================================================
# HEMISFERIO (COMPLETO)
# ============================================================
class Hemisferio:
    def __init__(self, nombre, tau, generar_entrada_func, seed=None, sesgo=0.0):
        if seed is not None:
            np.random.seed(seed)
        self.nombre = nombre
        self.tau = tau
        self.generar_entrada = generar_entrada_func
        self.sesgo = sesgo
        self.Phi = np.random.normal(sesgo, 0.1, DIM_HEMISFERIO)
        self.Phi_vel = np.zeros(DIM_HEMISFERIO)
        self.entrada = None
        self.sr = 48000
        self.en_inanicion = False
        self.factor_inanicion = 1.0
        self.estímulos_externos = deque()
    
    def añadir_estimulo(self, valor):
        """Añade estímulo externo (del otro organismo)"""
        self.estímulos_externos.append(valor)
    
    def _calcular_omega(self):
        return np.mean(self.Phi[:DIM_HEMISFERIO])
    
    def entrada_t(self, t, duracion_total):
        # Priorizar estímulos de otro organismo
        if self.estímulos_externos:
            return self.estímulos_externos.popleft()
        
        if self.entrada is None:
            self.entrada = self.generar_entrada(duracion_total, self.sr)
        idx = int(t * self.sr)
        if idx >= len(self.entrada):
            return 0.0
        return self.entrada[idx] * self.factor_inanicion
    
    def actualizar(self, t, dt, duracion_total, otro_hemisferio=None):
        entrada = self.entrada_t(t, duracion_total)
        
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
        
        return {'omega': self._calcular_omega()}
    
    def reset(self):
        self.Phi = np.random.normal(self.sesgo, 0.1, DIM_HEMISFERIO)
        self.Phi_vel = np.zeros(DIM_HEMISFERIO)
        self.entrada = None
        self.estímulos_externos.clear()


# ============================================================
# VALENCIA LOCAL
# ============================================================
class ValenciaLocal:
    def __init__(self):
        self.valencia = {}
        self.lr = TASA_APRENDIZAJE
        self.historial = {}
    
    def actualizar_con_estimulo(self, setpoint, estimulo, dt, peso=PESO_ESTIMULO, recompensa=0.0):
        key = round(setpoint/5)*5 if setpoint != 0 else 0
        if key not in self.valencia:
            self.valencia[key] = 0.0
            self.historial[key] = []
        
        # Desplazamiento gradual hacia el estímulo
        self.valencia[key] += peso * (estimulo - self.valencia[key]) * self.lr * dt
        
        # Recompensa por convergencia (refuerzo de viabilidad mutua)
        if recompensa > 0:
            self.valencia[key] += recompensa * self.lr * dt * ESCALA_REWARD
        
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
# ORGANISMO COMPLETO (CON HEMISFERIOS)
# ============================================================
class OrganismoCompleto:
    def __init__(self, seed, nombre):
        self.nombre = nombre
        self.seed = seed
        
        def generar_ruido_rosa(duracion, sr):
            n = int(duracion * sr)
            ruido = np.random.normal(0, 1, n)
            fft = np.fft.rfft(ruido)
            freqs = np.fft.rfftfreq(n, 1/sr)
            filtro = 1.0 / np.sqrt(freqs + 0.01)
            fft_filtrado = fft * filtro
            ruido_rosa = np.fft.irfft(fft_filtrado, n=n)
            return ruido_rosa / (np.max(np.abs(ruido_rosa)) + 1e-10)
        
        def generar_clicks_poisson(duracion, tasa=0.5, sr=48000):
            n = int(duracion * sr)
            clicks = np.zeros(n)
            n_clicks = int(duracion * tasa)
            for _ in range(n_clicks):
                pos = int(np.random.exponential(1.0/tasa) * sr)
                if pos < n:
                    clicks[pos] = 1.0
            return clicks
        
        self.L = Hemisferio("L", 30, generar_ruido_rosa, seed, SESGO_L)
        self.R = Hemisferio("R", 300, generar_clicks_poisson, seed+100, SESGO_R)
        self.BL = Hemisferio("BL", 30, generar_ruido_rosa, seed+200, SESGO_L)
        self.BR = Hemisferio("BR", 300, generar_clicks_poisson, seed+300, SESGO_R)
        self.hemisferios = [self.L, self.R, self.BL, self.BR]
        
        self.Cb = 0.0
        self.D = 0.0
        self.valencia = ValenciaLocal()
        self.memoria_trabajo = None  # Simplificado para acoplamiento
        self.buffer = None
        
        # Historial
        self.historial_valencia = []
        self.historial_Cb = []
        self.historial_D = []
    
    def set_estado_inicial(self, setpoint, valencia, Cb=0.0, D=0.0):
        self.valencia.set(setpoint, valencia)
        self.Cb = Cb
        self.D = D
    
    def get_valencia(self, setpoint):
        return self.valencia.get(setpoint)
    
    def get_estado(self, setpoint):
        return {
            'valencia': self.valencia.get(setpoint),
            'Cb': self.Cb,
            'D': self.D
        }
    
    def procesar_senal(self, setpoint, dt):
        """Actualiza hemisferios y estados internos"""
        for h in self.hemisferios:
            h.actualizar(0, dt, 1.0, None)
        
        # Calcular Cb basado en desacople entre hemisferios
        omega_L = self.L._calcular_omega()
        omega_R = self.R._calcular_omega()
        gradiente = omega_L - omega_R
        self.Cb = min(CB_MAX, self.Cb + abs(gradiente) * dt)
        self.Cb *= (1 - dt / TAU_CB)
        
        # Calcular D basado en conflicto de valencias
        val_habito = self.valencia.get(HABITO_SETPOINT)
        val_trauma = self.valencia.get(TRAUMA_SETPOINT)
        self.D = min(1.0, abs(val_habito - val_trauma) / 100.0)
    
    def recibir_estimulo(self, estimulo, setpoint, dt, peso=PESO_ESTIMULO, recompensa=0.0):
        """Recibe estímulo del otro organismo"""
        # Inyectar estímulo en hemisferios
        for h in self.hemisferios:
            h.añadir_estimulo(estimulo)
        
        # Actualizar valencia con reforzamiento
        self.valencia.actualizar_con_estimulo(setpoint, estimulo, dt, peso, recompensa)
        
        # Actualizar Cb y D
        self.procesar_senal(setpoint, dt)
    
    def obtener_resultado(self, setpoint):
        return self.valencia.get(setpoint)
    
    def registrar_estado(self):
        self.historial_valencia.append(self.valencia.get(TRAUMA_SETPOINT))
        self.historial_Cb.append(self.Cb)
        self.historial_D.append(self.D)
    
    def reset(self):
        for h in self.hemisferios:
            h.reset()
        self.valencia.reset()
        self.Cb = 0.0
        self.D = 0.0
        self.historial_valencia = []
        self.historial_Cb = []
        self.historial_D = []


# ============================================================
# BUFFER DE ACOPLAMIENTO
# ============================================================
class BufferAcoplamiento:
    def __init__(self, capacidad=100):
        self.historial = deque(maxlen=capacidad)
        self.recompensa_acumulada = 0.0
    
    def calcular_recompensa(self, diferencia_actual, diferencia_anterior):
        if diferencia_anterior <= 0:
            return 0.0
        
        reduccion = (diferencia_anterior - diferencia_actual) / diferencia_anterior
        if reduccion > 0:
            return reduccion * REWARD_BASE
        else:
            return 0.0  # Sin penalidad, solo refuerzo positivo
    
    def reset(self):
        self.historial.clear()
        self.recompensa_acumulada = 0.0


# ============================================================
# RONDA DE ACOPLAMIENTO (8 PASOS)
# ============================================================
def ronda_acoplamiento(A, B, setpoint, ronda_num, dt=DT_PROCESAMIENTO):
    # PASO 1: Procesan señal
    A.procesar_senal(setpoint, dt)
    B.procesar_senal(setpoint, dt)
    
    # PASO 2: Envían resultados y almacenan
    resultado_A = A.obtener_resultado(setpoint)
    resultado_B = B.obtener_resultado(setpoint)
    
    # Intercambio de estímulos
    A.recibir_estimulo(resultado_B, setpoint, dt)
    B.recibir_estimulo(resultado_A, setpoint, dt)
    
    # PASO 3: Procesan otra señal
    A.procesar_senal(setpoint, dt)
    B.procesar_senal(setpoint, dt)
    
    # PASO 4: Nuevos resultados
    nuevo_resultado_A = A.obtener_resultado(setpoint)
    nuevo_resultado_B = B.obtener_resultado(setpoint)
    
    return nuevo_resultado_A, nuevo_resultado_B


# ============================================================
# EXPERIMENTO PRINCIPAL
# ============================================================
def ejecutar_v182a_v5():
    print("=" * 100)
    print("EXPERIMENTO V182A-v5 — ACOPLAMIENTO BIDIRECCIONAL (FINAL)")
    print("=" * 100)
    print("  PRINCIPIOS (consenso equipo):")
    print(f"    • 100 iteraciones para historia compartida")
    print(f"    • Reforzamiento = ganancia mutua de viabilidad")
    print(f"    • Organismos completos (Cb, D reales)")
    print(f"    • Parada temprana si diferencia < {PARADA_TEMPRANA_DIF}")
    print("")
    print("  CRITERIOS DE ÉXITO:")
    print(f"    ✅ Reducción de diferencia > {REDUCCION_MIN:.0%}")
    print(f"    ✅ Diferencia final < {DIFERENCIA_FINAL_MAX}")
    print(f"    ✅ Estabilización (std últimas 10) < {ESTABILIZACION_MAX}")
    print(f"    ✅ Simetría: ambos se movieron > {MOVIMIENTO_MIN}")
    print(f"    ✅ Correlación Cb(A,B) > {CORRELACION_CB_MIN}")
    print("=" * 100)

    A = OrganismoCompleto(SEMILLA_A, "A")
    B = OrganismoCompleto(SEMILLA_B, "B")
    buffer = BufferAcoplamiento()
    
    print("\n" + "=" * 60)
    print("CONDICIÓN INICIAL: Divergencia")
    print("=" * 60)
    
    A.set_estado_inicial(TRAUMA_SETPOINT, -RANGO_VALENCIA_INICIAL/2, Cb=50.0, D=0.6)
    B.set_estado_inicial(TRAUMA_SETPOINT, +RANGO_VALENCIA_INICIAL/2, Cb=10.0, D=0.2)
    
    print(f"  Valencia A: {A.get_valencia(TRAUMA_SETPOINT):.2f}")
    print(f"  Valencia B: {B.get_valencia(TRAUMA_SETPOINT):.2f}")
    print(f"  Diferencia inicial: {abs(A.get_valencia(TRAUMA_SETPOINT) - B.get_valencia(TRAUMA_SETPOINT)):.2f}")
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('V182_logs', exist_ok=True)
    
    print("\n" + "=" * 60)
    print(f"COMUNICACIÓN BIDIRECCIONAL ({RONDAS_ACP} rondas máx)")
    print("=" * 60)
    
    historial_A = []
    historial_B = []
    diferencias = []
    recompensas = []
    cb_A_hist = []
    cb_B_hist = []
    
    for ronda in range(RONDAS_ACP):
        val_A, val_B = ronda_acoplamiento(A, B, TRAUMA_SETPOINT, ronda)
        
        historial_A.append(val_A)
        historial_B.append(val_B)
        diferencia_actual = abs(val_A - val_B)
        diferencias.append(diferencia_actual)
        
        A.registrar_estado()
        B.registrar_estado()
        cb_A_hist.append(A.Cb)
        cb_B_hist.append(B.Cb)
        
        # Calcular recompensa por convergencia
        if ronda > 0:
            recompensa = buffer.calcular_recompensa(diferencia_actual, diferencias[-2])
            recompensas.append(recompensa)
            
            # Aplicar recompensa a ambos (beneficio mutuo)
            if recompensa > 0:
                A.recibir_estimulo(recompensa, TRAUMA_SETPOINT, DT_PROCESAMIENTO * 0.1, recompensa=recompensa)
                B.recibir_estimulo(recompensa, TRAUMA_SETPOINT, DT_PROCESAMIENTO * 0.1, recompensa=recompensa)
        else:
            recompensas.append(0.0)
        
        if (ronda + 1) % 10 == 0:
            print(f"\n  Ronda {ronda+1}:")
            print(f"    A: {val_A:.2f}, B: {val_B:.2f}, diff: {diferencia_actual:.2f}")
            print(f"    Recompensa acumulada: {buffer.recompensa_acumulada:.4f}")
        
        # Parada temprana
        if diferencia_actual < PARADA_TEMPRANA_DIF and ronda > 20:
            print(f"\n  ✅ Convergencia temprana en ronda {ronda+1}")
            break
    
    # ============================================================
    # ANÁLISIS DE MÉTRICAS
    # ============================================================
    diferencia_inicial = diferencias[0]
    diferencia_final = diferencias[-1]
    reduccion = 1.0 - (diferencia_final / diferencia_inicial) if diferencia_inicial > 0 else 0
    
    ultimas_diferencias = diferencias[-10:] if len(diferencias) >= 10 else diferencias
    estabilizacion = np.std(ultimas_diferencias) if len(ultimas_diferencias) > 1 else 0
    
    movimiento_A = abs(historial_A[-1] - historial_A[0])
    movimiento_B = abs(historial_B[-1] - historial_B[0])
    simetria = movimiento_A > MOVIMIENTO_MIN and movimiento_B > MOVIMIENTO_MIN
    
    # Correlación de Cb (sincronización emergente)
    if len(cb_A_hist) > 10 and len(cb_B_hist) > 10:
        correlacion_cb = np.corrcoef(cb_A_hist, cb_B_hist)[0, 1]
        if np.isnan(correlacion_cb):
            correlacion_cb = 0.0
    else:
        correlacion_cb = 0.0
    
    # Evaluación
    exito_reduccion = reduccion > REDUCCION_MIN
    exito_diferencia = diferencia_final < DIFERENCIA_FINAL_MAX
    exito_estabilizacion = estabilizacion < ESTABILIZACION_MAX
    exito_simetria = simetria
    exito_correlacion = correlacion_cb > CORRELACION_CB_MIN
    
    exito = exito_reduccion and exito_diferencia and exito_estabilizacion and exito_simetria and exito_correlacion
    
    # ============================================================
    # RESULTADOS
    # ============================================================
    print("\n" + "=" * 80)
    print("RESULTADOS V182A-v5 — Acoplamiento Bidireccional")
    print("=" * 80)
    
    print(f"\n  📊 MÉTRICAS DE CONVERGENCIA:")
    print(f"     Diferencia inicial: {diferencia_inicial:.2f}")
    print(f"     Diferencia final: {diferencia_final:.2f}")
    print(f"     Reducción: {reduccion:.1%} -> {'✅' if exito_reduccion else '❌'}")
    print(f"     Estabilización (std): {estabilizacion:.2f} -> {'✅' if exito_estabilizacion else '❌'}")
    
    print(f"\n  📊 MÉTRICAS DE SIMETRÍA:")
    print(f"     Movimiento A: {movimiento_A:.2f}")
    print(f"     Movimiento B: {movimiento_B:.2f}")
    print(f"     Simetría: {'✅' if exito_simetria else '❌'}")
    
    print(f"\n  📊 SINCRONIZACIÓN EMERGENTE:")
    print(f"     Correlación Cb(A,B): {correlacion_cb:.3f} -> {'✅' if exito_correlacion else '❌'}")
    
    print(f"\n  📊 VALENCIAS FINALES:")
    print(f"     A: {historial_A[-1]:.2f}")
    print(f"     B: {historial_B[-1]:.2f}")
    print(f"     Diferencia final: {diferencia_final:.2f} -> {'✅' if exito_diferencia else '❌'}")
    
    print(f"\n  📊 REFUERZO ACUMULADO:")
    print(f"     Recompensa total: {buffer.recompensa_acumulada:.4f}")
    
    print("\n" + "=" * 80)
    if exito:
        print("  ✅ ACOPLAMIENTO BIDIRECCIONAL DEMOSTRADO")
        print("")
        print("     Los organismos demostraron:")
        print("     ✓ Convergencia progresiva en ~100 iteraciones")
        print("     ✓ Capacidad de respuesta simétrica")
        print("     ✓ Sincronización emergente de Cb")
        print("     ✓ Reforzamiento como ganancia mutua de viabilidad")
        print("")
        print("  PRÓXIMO: V182B — Comunicación funcional")
    else:
        print("  ⚠️ ACOPLAMIENTO BIDIRECCIONAL NO DEMOSTRADO")
        if not exito_reduccion:
            print("     No hubo convergencia suficiente")
        if not exito_diferencia:
            print("     Las valencias finales siguen muy distantes")
        if not exito_estabilizacion:
            print("     El sistema no se estabilizó")
        if not exito_simetria:
            print("     Uno de los organismos no respondió activamente")
        if not exito_correlacion:
            print("     No hubo sincronización de Cb (falta acoplamiento real)")
    print("=" * 80)
    
    # ============================================================
    # GRÁFICOS
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    ax = axes[0, 0]
    ax.plot(historial_A, 'b-', linewidth=0.8, alpha=0.7, label='A')
    ax.plot(historial_B, 'r-', linewidth=0.8, alpha=0.7, label='B')
    ax.set_xlabel('Ronda')
    ax.set_ylabel('Valencia en +60°')
    ax.set_title('Evolución de valencias')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(diferencias, 'purple', linewidth=0.8)
    ax.axhline(y=DIFERENCIA_FINAL_MAX, color='green', linestyle='--', label=f'Umbral ({DIFERENCIA_FINAL_MAX})')
    ax.set_xlabel('Ronda')
    ax.set_ylabel('Diferencia')
    ax.set_title('Convergencia')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.bar(['A', 'B'], [movimiento_A, movimiento_B], color=['blue', 'red'], alpha=0.7)
    ax.axhline(y=MOVIMIENTO_MIN, color='green', linestyle='--', label=f'Umbral ({MOVIMIENTO_MIN})')
    ax.set_ylabel('Cambio absoluto')
    ax.set_title('Movimiento de cada organismo')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.plot(cb_A_hist, 'b-', linewidth=0.8, alpha=0.5, label='Cb A')
    ax.plot(cb_B_hist, 'r-', linewidth=0.8, alpha=0.5, label='Cb B')
    ax.set_xlabel('Ronda')
    ax.set_ylabel('Cb (Consciencia)')
    ax.set_title(f'Sincronización de Cb (corr={correlacion_cb:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'V182_logs/v182a_v5_acoplamiento_{ts}.png', dpi=150)
    print(f"\n  📊 Gráficos guardados: V182_logs/v182a_v5_acoplamiento_{ts}.png")
    
    # Guardar datos
    raw_data = {
        'version': 'V182A-v5',
        'timestamp': ts,
        'params': {
            'RONDAS_ACP': RONDAS_ACP,
            'DT_PROCESAMIENTO': DT_PROCESAMIENTO,
            'PESO_ESTIMULO': PESO_ESTIMULO,
            'REWARD_BASE': REWARD_BASE,
            'ESCALA_REWARD': ESCALA_REWARD,
            'REDUCCION_MIN': REDUCCION_MIN,
            'DIFERENCIA_FINAL_MAX': DIFERENCIA_FINAL_MAX,
            'ESTABILIZACION_MAX': ESTABILIZACION_MAX,
            'MOVIMIENTO_MIN': MOVIMIENTO_MIN,
            'CORRELACION_CB_MIN': CORRELACION_CB_MIN,
        },
        'resultados': {
            'diferencia_inicial': float(diferencia_inicial),
            'diferencia_final': float(diferencia_final),
            'reduccion': float(reduccion),
            'estabilizacion': float(estabilizacion),
            'movimiento_A': float(movimiento_A),
            'movimiento_B': float(movimiento_B),
            'correlacion_cb': float(correlacion_cb),
            'recompensa_total': float(buffer.recompensa_acumulada),
            'exito_reduccion': bool(exito_reduccion),
            'exito_diferencia': bool(exito_diferencia),
            'exito_estabilizacion': bool(exito_estabilizacion),
            'exito_simetria': bool(exito_simetria),
            'exito_correlacion': bool(exito_correlacion),
            'exito': bool(exito)
        }
    }
    
    with open(f'V182_logs/v182a_v5_raw_{ts}.json', 'w') as f:
        json.dump(raw_data, f, indent=2)
    print(f"  📁 Datos guardados: V182_logs/v182a_v5_raw_{ts}.json")
    
    return exito


if __name__ == "__main__":
    start = time.time()
    exito = ejecutar_v182a_v5()
    elapsed = time.time() - start
    print(f"\n  ⏱️ Tiempo total: {elapsed/60:.1f} min | Éxito: {exito}")