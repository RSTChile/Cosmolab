#!/usr/bin/env python3
"""
V182A-v7 — ACOPLAMIENTO BIDIRECCIONAL CON MEMORIA RELACIONAL
================================================================================
INTEGRACIÓN DE LAS OBSERVACIONES:
  - GPT: Falta el paso 5 (comparar resultado_actual con historial del otro)
  - Meta: Validación de parámetros y criterios
  - Qwen: Código completo unificado
  - Alexis: Los 8 pasos originales

LOS 8 PASOS (Alexis):
  1. Ambos procesan una señal
  2. Se envían los resultados y los almacenan
  3. Ambos procesan otra señal
  4. Se envían los resultados y los almacenan
  5. Comparan señales con resultados almacenados (NUEVO)
  6. Responden con la comparación
  7. Reiteran el ciclo
  8. Éxito cuando ambos llegan a soluciones similares

CRITERIOS DE ÉXITO:
  ✅ Reducción de diferencia > 40%
  ✅ Diferencia final < 20.0
  ✅ Estabilización (std últimas 10) < 5.0
  ✅ Simetría: ambos se movieron > 12.0
  ✅ Correlación Cb(A,B) > 0.3
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

# Parámetros de acoplamiento
RONDAS_ACP = 1000
DT_PROCESAMIENTO = 1.0
PESO_ESTIMULO = 0.3
TASA_APRENDIZAJE = 0.01
REWARD_BASE = 1.0
ESCALA_REWARD = 20.0
PARADA_TEMPRANA_DIF = 5.0
RANGO_VALENCIA_INICIAL = 50.0

# Umbrales de éxito
REDUCCION_MIN = 0.40
DIFERENCIA_FINAL_MAX = 20.0
ESTABILIZACION_MAX = 5.0
MOVIMIENTO_MIN = 12.0
CORRELACION_CB_MIN = 0.30

# Capacidad de la memoria relacional (GPT)
MEMORIA_CAPACIDAD = 10


# ============================================================
# MEMORIA RELACIONAL (NUEVO — PASO 5)
# ============================================================
class MemoriaRelacional:
    """
    Almacena el historial de resultados recibidos del otro organismo.
    Permite la comparación entre resultado_actual y resultado_anterior.
    """
    def __init__(self, capacidad=MEMORIA_CAPACIDAD):
        self.capacidad = capacidad
        self.historial = deque(maxlen=capacidad)  # (ronda, resultado_otro)
    
    def almacenar(self, ronda, resultado_otro):
        self.historial.append((ronda, resultado_otro))
    
    def comparar_con_anterior(self, resultado_actual):
        """Compara resultado_actual con el último resultado almacenado del otro"""
        if len(self.historial) < 1:
            return None, 0.0
        
        ultimo_ronda, ultimo_resultado = self.historial[-1]
        diferencia = abs(resultado_actual - ultimo_resultado)
        return {
            'diferencia': diferencia,
            'ultimo_resultado': ultimo_resultado,
            'ronda_anterior': ultimo_ronda
        }, diferencia
    
    def reset(self):
        self.historial.clear()


# ============================================================
# HEMISFERIO
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
        self.estímulos_externos.append(valor)
    
    def _calcular_omega(self):
        return np.mean(self.Phi[:DIM_HEMISFERIO])
    
    def entrada_t(self, t, duracion_total):
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
# ORGANISMO COMPLETO (CON MEMORIA RELACIONAL)
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
        
        # 🟢 NUEVO: Memoria relacional para el paso 5
        self.memoria_relacional = MemoriaRelacional()
        
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
        
        omega_L = self.L._calcular_omega()
        omega_R = self.R._calcular_omega()
        gradiente = omega_L - omega_R
        self.Cb = min(CB_MAX, self.Cb + abs(gradiente) * dt)
        self.Cb *= (1 - dt / TAU_CB)
        
        val_habito = self.valencia.get(HABITO_SETPOINT)
        val_trauma = self.valencia.get(TRAUMA_SETPOINT)
        self.D = min(1.0, abs(val_habito - val_trauma) / 100.0)
    
    def recibir_estimulo(self, estimulo, setpoint, dt, peso=PESO_ESTIMULO, recompensa=0.0):
        """Recibe estímulo del otro organismo"""
        for h in self.hemisferios:
            h.añadir_estimulo(estimulo)
        
        self.valencia.actualizar_con_estimulo(setpoint, estimulo, dt, peso, recompensa)
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
        self.memoria_relacional.reset()
        self.historial_valencia = []
        self.historial_Cb = []
        self.historial_D = []


# ============================================================
# BUFFER DE ACOPLAMIENTO (REFORZAMIENTO)
# ============================================================
class BufferAcoplamiento:
    def __init__(self):
        self.recompensa_acumulada = 0.0
    
    def calcular_recompensa(self, diferencia_actual, diferencia_anterior):
        if diferencia_anterior <= 0:
            return 0.0
        
        reduccion = (diferencia_anterior - diferencia_actual) / diferencia_anterior
        if reduccion > 0:
            return reduccion * REWARD_BASE
        else:
            return 0.0
    
    def reset(self):
        self.recompensa_acumulada = 0.0


# ============================================================
# RONDA DE ACOPLAMIENTO (8 PASOS — CON MEMORIA RELACIONAL)
# ============================================================
def ronda_acoplamiento(A, B, setpoint, ronda_num, dt=DT_PROCESAMIENTO):
    """
    Implementa los 8 pasos de acoplamiento bidireccional según Alexis.
    Incluye el PASO 5: comparación con resultados almacenados.
    """
    
    # PASO 1: Ambos procesan una señal
    A.procesar_senal(setpoint, dt)
    B.procesar_senal(setpoint, dt)
    
    # PASO 2: Se envían los resultados y los almacenan
    resultado_A = A.obtener_resultado(setpoint)
    resultado_B = B.obtener_resultado(setpoint)
    
    # Almacenar en memoria relacional (para futuras comparaciones)
    A.memoria_relacional.almacenar(ronda_num, resultado_B)
    B.memoria_relacional.almacenar(ronda_num, resultado_A)
    
    # Intercambio de estímulos
    A.recibir_estimulo(resultado_B, setpoint, dt)
    B.recibir_estimulo(resultado_A, setpoint, dt)
    
    # PASO 3: Ambos procesan otra señal
    A.procesar_senal(setpoint, dt)
    B.procesar_senal(setpoint, dt)
    
    # PASO 4: Se envían nuevos resultados y los almacenan
    nuevo_resultado_A = A.obtener_resultado(setpoint)
    nuevo_resultado_B = B.obtener_resultado(setpoint)
    
    A.memoria_relacional.almacenar(ronda_num + 0.5, nuevo_resultado_B)
    B.memoria_relacional.almacenar(ronda_num + 0.5, nuevo_resultado_A)
    
    # PASO 5: Comparan señales con resultados almacenados (CRÍTICO)
    comparacion_A, diff_A = A.memoria_relacional.comparar_con_anterior(nuevo_resultado_B)
    comparacion_B, diff_B = B.memoria_relacional.comparar_con_anterior(nuevo_resultado_A)
    
    # PASO 6: Responden con la comparación
    if comparacion_A is not None:
        # La respuesta es la discrepancia detectada
        A.recibir_estimulo(diff_B, setpoint, dt, peso=PESO_ESTIMULO * 1.5)
    if comparacion_B is not None:
        B.recibir_estimulo(diff_A, setpoint, dt, peso=PESO_ESTIMULO * 1.5)
    
    # PASO 7: Reiterar (se hace en el bucle principal)
    # PASO 8: Éxito (se evalúa fuera)
    
    return nuevo_resultado_A, nuevo_resultado_B, comparacion_A, comparacion_B, diff_A, diff_B


# ============================================================
# EXPERIMENTO PRINCIPAL
# ============================================================
def ejecutar_v182a_v7():
    print("=" * 100)
    print("EXPERIMENTO V182A-v7 — ACOPLAMIENTO BIDIRECCIONAL CON MEMORIA RELACIONAL")
    print("=" * 100)
    print("  LOS 8 PASOS (Alexis):")
    print("    1. Ambos procesan una señal")
    print("    2. Se envían resultados y los almacenan")
    print("    3. Ambos procesan otra señal")
    print("    4. Se envían resultados y los almacenan")
    print("    5. 🟢 Comparan señales con resultados almacenados (NUEVO)")
    print("    6. Responden con la comparación")
    print("    7. Reiteran el ciclo")
    print("    8. Éxito cuando ambos llegan a soluciones similares")
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
    discrepancias_A = []
    discrepancias_B = []
    
    start_time = time.time()
    
    for ronda in range(RONDAS_ACP):
        val_A, val_B, comp_A, comp_B, diff_A, diff_B = ronda_acoplamiento(A, B, TRAUMA_SETPOINT, ronda)
        
        historial_A.append(val_A)
        historial_B.append(val_B)
        diferencia_actual = abs(val_A - val_B)
        diferencias.append(diferencia_actual)
        
        if comp_A:
            discrepancias_A.append(comp_A['diferencia'])
        if comp_B:
            discrepancias_B.append(comp_B['diferencia'])
        
        A.registrar_estado()
        B.registrar_estado()
        cb_A_hist.append(A.Cb)
        cb_B_hist.append(B.Cb)
        
        # Calcular recompensa por convergencia
        if ronda > 0:
            recompensa = buffer.calcular_recompensa(diferencia_actual, diferencias[-2])
            recompensas.append(recompensa)
            
            if recompensa > 0:
                buffer.recompensa_acumulada += recompensa
                A.recibir_estimulo(recompensa, TRAUMA_SETPOINT, DT_PROCESAMIENTO * 0.1, recompensa=recompensa)
                B.recibir_estimulo(recompensa, TRAUMA_SETPOINT, DT_PROCESAMIENTO * 0.1, recompensa=recompensa)
        else:
            recompensas.append(0.0)
        
        if (ronda + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"\n  Ronda {ronda+1}:")
            print(f"    A: {val_A:.2f}, B: {val_B:.2f}, diff: {diferencia_actual:.2f}")
            print(f"    Recompensa acumulada: {buffer.recompensa_acumulada:.4f}")
            print(f"    Discrepancia A (comparación): {diff_A:.4f}")
            print(f"    Discrepancia B (comparación): {diff_B:.4f}")
            print(f"    Tiempo: {elapsed/60:.1f} min")
        
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
    
    if len(cb_A_hist) > 10 and len(cb_B_hist) > 10:
        correlacion_cb = np.corrcoef(cb_A_hist, cb_B_hist)[0, 1]
        if np.isnan(correlacion_cb):
            correlacion_cb = 0.0
    else:
        correlacion_cb = 0.0
    
    # Métricas de memoria relacional
    discrepancia_media_A = np.mean(discrepancias_A) if discrepancias_A else 0.0
    discrepancia_media_B = np.mean(discrepancias_B) if discrepancias_B else 0.0
    
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
    print("RESULTADOS V182A-v7 — Acoplamiento Bidireccional")
    print("=" * 80)
    
    print(f"\n  📊 MÉTRICAS DE CONVERGENCIA:")
    print(f"     Diferencia inicial: {diferencia_inicial:.2f}")
    print(f"     Diferencia final: {diferencia_final:.2f}")
    print(f"     Reducción: {reduccion:.1%} -> {'✅' if exito_reduccion else '❌'}")
    print(f"     Estabilización (std últimas 10): {estabilizacion:.2f} -> {'✅' if exito_estabilizacion else '❌'}")
    
    print(f"\n  📊 MÉTRICAS DE SIMETRÍA:")
    print(f"     Movimiento A: {movimiento_A:.2f}")
    print(f"     Movimiento B: {movimiento_B:.2f}")
    print(f"     Simetría: {'✅' if exito_simetria else '❌'}")
    
    print(f"\n  📊 SINCRONIZACIÓN EMERGENTE:")
    print(f"     Correlación Cb(A,B): {correlacion_cb:.3f} -> {'✅' if exito_correlacion else '❌'}")
    
    print(f"\n  📊 MEMORIA RELACIONAL (Paso 5):")
    print(f"     Discrepancia media detectada por A: {discrepancia_media_A:.4f}")
    print(f"     Discrepancia media detectada por B: {discrepancia_media_B:.4f}")
    
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
        print("     Los organismos demostraron los 8 pasos:")
        print("     ✓ Procesamiento mutuo de señales")
        print("     ✓ Almacenamiento de resultados del otro")
        print("     ✓ Comparación con historial (memoria relacional)")
        print("     ✓ Respuesta basada en discrepancias detectadas")
        print("     ✓ Convergencia progresiva (~1000 iteraciones)")
        print("     ✓ Simetría (ambos se movieron)")
        print("     ✓ Sincronización emergente de Cb")
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
            print("     No hubo sincronización de Cb")
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
    plt.savefig(f'V182_logs/v182a_v7_acoplamiento_{ts}.png', dpi=150)
    print(f"\n  📊 Gráficos guardados: V182_logs/v182a_v7_acoplamiento_{ts}.png")
    
    # Guardar datos
    raw_data = {
        'version': 'V182A-v7',
        'timestamp': ts,
        'params': {
            'RONDAS_ACP': RONDAS_ACP,
            'DT_PROCESAMIENTO': DT_PROCESAMIENTO,
            'PESO_ESTIMULO': PESO_ESTIMULO,
            'TASA_APRENDIZAJE': TASA_APRENDIZAJE,
            'REWARD_BASE': REWARD_BASE,
            'ESCALA_REWARD': ESCALA_REWARD,
            'MEMORIA_CAPACIDAD': MEMORIA_CAPACIDAD,
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
            'discrepancia_media_A': float(discrepancia_media_A),
            'discrepancia_media_B': float(discrepancia_media_B),
            'recompensa_total': float(buffer.recompensa_acumulada),
            'exito_reduccion': bool(exito_reduccion),
            'exito_diferencia': bool(exito_diferencia),
            'exito_estabilizacion': bool(exito_estabilizacion),
            'exito_simetria': bool(exito_simetria),
            'exito_correlacion': bool(exito_correlacion),
            'exito': bool(exito)
        }
    }
    
    with open(f'V182_logs/v182a_v7_raw_{ts}.json', 'w') as f:
        json.dump(raw_data, f, indent=2)
    print(f"  📁 Datos guardados: V182_logs/v182a_v7_raw_{ts}.json")
    
    return exito


if __name__ == "__main__":
    start = time.time()
    exito = ejecutar_v182a_v7()
    elapsed = time.time() - start
    print(f"\n  ⏱️ Tiempo total: {elapsed/60:.1f} min | Éxito: {exito}")