#!/usr/bin/env python3
"""
V182A-v3 — ACOPLAMIENTO BIDIRECCIONAL (CON TIEMPO REAL DE PROCESAMIENTO)
================================================================================
CORRECCIÓN CRÍTICA:
  Los organismos necesitan TIEMPO REAL para procesar e integrar la señal del otro.
  dt_procesamiento = 2.0s (antes 0.5s)
  RONDAS_ACP = 20 (antes 8)
  peso_estimulo = 0.1 (antes 0.3)
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

# Parámetros de acoplamiento (CORREGIDOS para tiempo real)
RONDAS_ACP = 20                    # 8 → 20 rondas
DT_PROCESAMIENTO = 2.0             # 0.5s → 2.0s por ronda
PESO_ESTIMULO = 0.1                # 0.3 → 0.1 (aprendizaje gradual)
TASA_APRENDIZAJE = 0.0005          # 0.001 → 0.0005 (más lento)

RANGO_VALENCIA_INICIAL = 50.0

# Umbrales de éxito
REDUCCION_MIN = 0.50
DIFERENCIA_FINAL_MAX = 15.0
ESTABILIZACION_MAX = 3.0
MOVIMIENTO_MIN = 10.0              # 2.0 → 10.0 (cambio significativo)


# ============================================================
# BUFFER DE ACOPLAMIENTO
# ============================================================
class BufferAcoplamiento:
    def __init__(self, capacidad=20):
        self.historial = deque(maxlen=capacidad)
        self.mis_estados = deque(maxlen=capacidad)
    
    def almacenar(self, ronda, estado_otro, estado_propio):
        self.historial.append({
            'ronda': ronda,
            'valencia': estado_otro.get('valencia', 0),
            'Cb': estado_otro.get('Cb', 0),
            'D': estado_otro.get('D', 0)
        })
        self.mis_estados.append({
            'ronda': ronda,
            'valencia': estado_propio.get('valencia', 0),
            'Cb': estado_propio.get('Cb', 0),
            'D': estado_propio.get('D', 0)
        })
    
    def comparar_con_anterior(self, estado_otro_actual):
        if len(self.historial) < 1:
            return None, 1.0
        
        anterior = self.historial[-1]
        delta_valencia = abs(estado_otro_actual.get('valencia', 0) - anterior['valencia'])
        delta_Cb = abs(estado_otro_actual.get('Cb', 0) - anterior['Cb'])
        delta_D = abs(estado_otro_actual.get('D', 0) - anterior['D'])
        
        comparacion = {
            'delta_valencia': delta_valencia,
            'delta_Cb': delta_Cb,
            'delta_D': delta_D,
            'magnitud': np.sqrt(delta_valencia**2 + delta_Cb**2 + delta_D**2) / 100.0
        }
        return comparacion, comparacion['magnitud']
    
    def convergencia(self, estado_actual):
        if len(self.historial) < 2:
            return 0.0
        
        diferencias = []
        for i, h in enumerate(self.historial):
            diff = abs(h['valencia'] - self.mis_estados[i]['valencia'])
            diferencias.append(diff)
        
        if diferencias[0] == 0:
            return 1.0
        
        reduccion = 1.0 - (diferencias[-1] / diferencias[0])
        return max(0.0, min(1.0, reduccion))
    
    def reset(self):
        self.historial.clear()
        self.mis_estados.clear()


# ============================================================
# VALENCIA LOCAL
# ============================================================
class ValenciaLocal:
    def __init__(self):
        self.valencia = {}
        self.lr = TASA_APRENDIZAJE
        self.historial = {}
    
    def actualizar_con_estimulo(self, setpoint, estimulo, dt, peso=PESO_ESTIMULO):
        key = round(setpoint/5)*5 if setpoint != 0 else 0
        if key not in self.valencia:
            self.valencia[key] = 0.0
            self.historial[key] = []
        
        # Desplazamiento gradual hacia el estímulo
        self.valencia[key] += peso * (estimulo - self.valencia[key]) * self.lr * dt
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
# ORGANISMO
# ============================================================
class OrganismoAcoplamiento:
    def __init__(self, seed, nombre):
        self.nombre = nombre
        self.seed = seed
        self.valencia = ValenciaLocal()
        self.Cb = 0.0
        self.D = 0.0
        self.buffer = BufferAcoplamiento()
        
        self.historial_valencia = []
        self.historial_Cb = []
        self.historial_D = []
    
    def get_estado(self, setpoint):
        return {
            'valencia': self.valencia.get(setpoint),
            'Cb': self.Cb,
            'D': self.D
        }
    
    def set_estado_inicial(self, setpoint, valencia, Cb=0.0, D=0.0):
        self.valencia.set(setpoint, valencia)
        self.Cb = Cb
        self.D = D
    
    def procesar_senal(self, setpoint, dt):
        """Tiempo real de procesamiento - simula integración neural"""
        # Durante el procesamiento, los hemisferios integran estímulos
        # Aquí simulamos con un pequeño drift basado en Cb
        val_actual = self.valencia.get(setpoint)
        # Cb alta produce más cambio (el organismo está más receptivo)
        cambio = self.Cb / CB_MAX * 0.1 * dt
        nueva_val = val_actual + cambio * (0 - val_actual)  # Drift hacia cero
        self.valencia.set(setpoint, nueva_val)
    
    def recibir_estimulo(self, estimulo, setpoint, dt, peso=PESO_ESTIMULO):
        """Recibe estímulo del otro organismo y actualiza estado gradualmente"""
        if estimulo is not None:
            self.valencia.actualizar_con_estimulo(setpoint, estimulo, dt, peso)
            # El estímulo también afecta Cb y D
            self.Cb = min(CB_MAX, self.Cb + abs(estimulo) * 0.01 * dt)
            self.D = self.calcular_D()
    
    def calcular_D(self):
        val_habito = self.valencia.get(HABITO_SETPOINT)
        val_trauma = self.valencia.get(TRAUMA_SETPOINT)
        return min(1.0, abs(val_habito - val_trauma) / 100.0)
    
    def obtener_resultado(self, setpoint):
        return self.valencia.get(setpoint)
    
    def registrar_estado(self):
        self.historial_valencia.append(self.valencia.get(TRAUMA_SETPOINT))
        self.historial_Cb.append(self.Cb)
        self.historial_D.append(self.D)
    
    def reset(self):
        self.valencia.reset()
        self.Cb = 0.0
        self.D = 0.0
        self.buffer.reset()
        self.historial_valencia = []
        self.historial_Cb = []
        self.historial_D = []


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
    
    estado_A = A.get_estado(setpoint)
    estado_B = B.get_estado(setpoint)
    
    A.buffer.almacenar(ronda_num, estado_B, estado_A)
    B.buffer.almacenar(ronda_num, estado_A, estado_B)
    
    A.recibir_estimulo(resultado_B, setpoint, dt)
    B.recibir_estimulo(resultado_A, setpoint, dt)
    
    # PASO 3: Procesan otra señal
    A.procesar_senal(setpoint, dt)
    B.procesar_senal(setpoint, dt)
    
    # PASO 4: Envían nuevos resultados y almacenan
    nuevo_resultado_A = A.obtener_resultado(setpoint)
    nuevo_resultado_B = B.obtener_resultado(setpoint)
    
    nuevo_estado_A = A.get_estado(setpoint)
    nuevo_estado_B = B.get_estado(setpoint)
    
    A.buffer.almacenar(ronda_num + 0.5, nuevo_estado_B, nuevo_estado_A)
    B.buffer.almacenar(ronda_num + 0.5, nuevo_estado_A, nuevo_estado_B)
    
    # PASO 5: Comparan señales con resultados almacenados
    comparacion_A, magnitud_A = A.buffer.comparar_con_anterior(nuevo_estado_B)
    comparacion_B, magnitud_B = B.buffer.comparar_con_anterior(nuevo_estado_A)
    
    # PASO 6: Responden con la comparación
    if comparacion_A:
        A.recibir_estimulo(magnitud_B, setpoint, dt, peso=PESO_ESTIMULO * 2)
    if comparacion_B:
        B.recibir_estimulo(magnitud_A, setpoint, dt, peso=PESO_ESTIMULO * 2)
    
    return nuevo_resultado_A, nuevo_resultado_B, comparacion_A, comparacion_B


# ============================================================
# EXPERIMENTO PRINCIPAL
# ============================================================
def ejecutar_v182a_v3():
    print("=" * 100)
    print("EXPERIMENTO V182A-v3 — ACOPLAMIENTO BIDIRECCIONAL (CON TIEMPO REAL)")
    print("=" * 100)
    print("  CORRECCIÓN: Los organismos necesitan TIEMPO para procesar.")
    print(f"    dt_procesamiento = {DT_PROCESAMIENTO}s por ronda")
    print(f"    Rondas = {RONDAS_ACP}")
    print(f"    Tiempo total de interacción: {RONDAS_ACP * DT_PROCESAMIENTO:.1f}s")
    print("")
    print("  CRITERIOS DE ÉXITO:")
    print(f"    ✅ Reducción de diferencia > {REDUCCION_MIN:.0%}")
    print(f"    ✅ Diferencia final < {DIFERENCIA_FINAL_MAX}")
    print(f"    ✅ Estabilización < {ESTABILIZACION_MAX}")
    print(f"    ✅ Simetría: ambos se movieron > {MOVIMIENTO_MIN}")
    print("=" * 100)

    A = OrganismoAcoplamiento(SEMILLA_A, "A")
    B = OrganismoAcoplamiento(SEMILLA_B, "B")
    
    print("\n" + "=" * 60)
    print("CONDICIÓN INICIAL: Divergencia")
    print("=" * 60)
    
    A.set_estado_inicial(TRAUMA_SETPOINT, -RANGO_VALENCIA_INICIAL/2, Cb=50.0, D=0.6)
    B.set_estado_inicial(TRAUMA_SETPOINT, +RANGO_VALENCIA_INICIAL/2, Cb=10.0, D=0.2)
    
    print(f"  Valencia A({TRAUMA_SETPOINT}) = {A.obtener_resultado(TRAUMA_SETPOINT):.2f}")
    print(f"  Valencia B({TRAUMA_SETPOINT}) = {B.obtener_resultado(TRAUMA_SETPOINT):.2f}")
    print(f"  Diferencia inicial: {abs(A.obtener_resultado(TRAUMA_SETPOINT) - B.obtener_resultado(TRAUMA_SETPOINT)):.2f}")
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('V182_logs', exist_ok=True)
    
    print("\n" + "=" * 60)
    print(f"COMUNICACIÓN BIDIRECCIONAL ({RONDAS_ACP} rondas de {DT_PROCESAMIENTO}s c/u)")
    print("=" * 60)
    
    historial_A = []
    historial_B = []
    diferencias = []
    
    for ronda in range(RONDAS_ACP):
        val_A, val_B, comp_A, comp_B = ronda_acoplamiento(A, B, TRAUMA_SETPOINT, ronda)
        
        historial_A.append(val_A)
        historial_B.append(val_B)
        diferencias.append(abs(val_A - val_B))
        
        A.registrar_estado()
        B.registrar_estado()
        
        if (ronda + 1) % 5 == 0:
            print(f"\n  Ronda {ronda+1}:")
            print(f"    Valencia A: {val_A:.2f}")
            print(f"    Valencia B: {val_B:.2f}")
            print(f"    Diferencia: {diferencias[-1]:.2f}")
    
    print("\n" + "=" * 80)
    print("RESULTADOS V182A-v3")
    print("=" * 80)
    
    diferencia_inicial = diferencias[0]
    diferencia_final = diferencias[-1]
    reduccion = 1.0 - (diferencia_final / diferencia_inicial) if diferencia_inicial > 0 else 0
    
    ultimas_diferencias = diferencias[-3:] if len(diferencias) >= 3 else diferencias
    estabilizacion = np.std(ultimas_diferencias) if len(ultimas_diferencias) > 1 else 0
    
    movimiento_A = abs(historial_A[-1] - historial_A[0])
    movimiento_B = abs(historial_B[-1] - historial_B[0])
    simetria = movimiento_A > MOVIMIENTO_MIN and movimiento_B > MOVIMIENTO_MIN
    
    exito_reduccion = reduccion > REDUCCION_MIN
    exito_diferencia = diferencia_final < DIFERENCIA_FINAL_MAX
    exito_estabilizacion = estabilizacion < ESTABILIZACION_MAX
    exito_simetria = simetria
    
    exito = exito_reduccion and exito_diferencia and exito_estabilizacion and exito_simetria
    
    print(f"\n  📊 MÉTRICAS DE CONVERGENCIA:")
    print(f"     Diferencia inicial: {diferencia_inicial:.2f}")
    print(f"     Diferencia final: {diferencia_final:.2f}")
    print(f"     Reducción: {reduccion:.1%} -> {'✅' if exito_reduccion else '❌'}")
    print(f"     Estabilización (std): {estabilizacion:.2f} -> {'✅' if exito_estabilizacion else '❌'}")
    
    print(f"\n  📊 MÉTRICAS DE SIMETRÍA:")
    print(f"     Movimiento A: {movimiento_A:.2f}")
    print(f"     Movimiento B: {movimiento_B:.2f}")
    print(f"     Simetría: {'✅' if exito_simetria else '❌'}")
    
    print(f"\n  📊 VALENCIAS FINALES:")
    print(f"     A: {historial_A[-1]:.2f}")
    print(f"     B: {historial_B[-1]:.2f}")
    print(f"     Diferencia final: {diferencia_final:.2f} -> {'✅' if exito_diferencia else '❌'}")
    
    print("\n" + "=" * 80)
    if exito:
        print("  ✅ ACOPLAMIENTO BIDIRECCIONAL DEMOSTRADO")
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
    print("=" * 80)
    
    # Gráficos
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    ax = axes[0, 0]
    ax.plot(historial_A, 'b-o', linewidth=1.5, markersize=4, label='A')
    ax.plot(historial_B, 'r-o', linewidth=1.5, markersize=4, label='B')
    ax.set_xlabel('Ronda')
    ax.set_ylabel('Valencia en +60°')
    ax.set_title('Evolución de valencias')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(diferencias, 'purple', linewidth=1.5)
    ax.axhline(y=DIFERENCIA_FINAL_MAX, color='green', linestyle='--')
    ax.set_xlabel('Ronda')
    ax.set_ylabel('Diferencia')
    ax.set_title('Convergencia')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.bar(['A', 'B'], [movimiento_A, movimiento_B], color=['blue', 'red'], alpha=0.7)
    ax.axhline(y=MOVIMIENTO_MIN, color='green', linestyle='--')
    ax.set_ylabel('Cambio absoluto')
    ax.set_title('Movimiento de cada organismo')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.plot(historial_A[-10:], 'b-o', label='A')
    ax.plot(historial_B[-10:], 'r-o', label='B')
    ax.set_xlabel('Ronda (últimas 10)')
    ax.set_ylabel('Valencia')
    ax.set_title(f'Estabilización final (std={estabilizacion:.2f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'V182_logs/v182a_v3_acoplamiento_{ts}.png', dpi=150)
    print(f"\n  📊 Gráficos guardados: V182_logs/v182a_v3_acoplamiento_{ts}.png")
    
    return exito


if __name__ == "__main__":
    start = time.time()
    exito = ejecutar_v182a_v3()
    elapsed = time.time() - start
    print(f"\n  ⏱️ Tiempo total: {elapsed/60:.1f} min | Éxito: {exito}")