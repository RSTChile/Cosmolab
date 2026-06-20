#!/usr/bin/env python3
"""
V182B-v3 — COMUNICACIÓN FUNCIONAL (ESTÍMULOS EXPLÍCITOS)
================================================================================
OBJETIVO: Demostrar que B mejora su estimación del setpoint
          cuando A le comunica su estimación correcta.

DISEÑO:
  FASE 1: Baseline — B solo con estímulo ruidoso (80% de ruido)
  FASE 2: Comunicación — A (estímulo limpio) + B (ruidoso + comunicación de A)

CRITERIOS DE ÉXITO:
  ✅ Mejora en error > 20%
  ✅ Latencia comunicación > baseline + 10%
  ✅ Correlación |val_A| vs error_B > 0.5
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json
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

# Parámetros de comunicación
RONDAS_POR_CONDICION = 200
SEGUNDOS_POR_RONDA = 0.1  # 100ms por ronda (rápido para prueba)
PESO_ESTIMULO = 0.3
TASA_APRENDIZAJE = 0.01

# Estímulos (explícitos, no audios)
SETPOINTS_TEST = [-60.0, 60.0]
RUIDO_B_AMPLITUD = 0.80  # 80% de ruido para B

# Umbrales de éxito
MEJORA_MIN = 0.20
LATENCIA_AUMENTO_MIN = 0.10
CORRELACION_MIN = 0.50


# ============================================================
# CLASES SIMPLIFICADAS (PARA PRUEBA RÁPIDA)
# ============================================================
class HemisferioSimple:
    def __init__(self, sesgo=0.0):
        self.omega = np.random.normal(sesgo, 0.1)
        self.estimulo_externo = None
    
    def recibir_estimulo(self, valor):
        self.estimulo_externo = valor
    
    def actualizar(self, dt):
        if self.estimulo_externo is not None:
            self.omega += 0.1 * (self.estimulo_externo - self.omega) * dt
            self.estimulo_externo = None
        return self.omega


class OrganismoSimple:
    def __init__(self, nombre):
        self.nombre = nombre
        self.L = HemisferioSimple(SESGO_L)
        self.R = HemisferioSimple(SESGO_R)
        self.valencia = {sp: 0.0 for sp in SETPOINTS_TEST}
        self.Cb = 0.0
        self.tiempo_deliberacion = 0.0
    
    def set_valencia(self, setpoint, valor):
        self.valencia[setpoint] = np.clip(valor, -100, 100)
    
    def get_valencia(self, setpoint):
        return self.valencia.get(setpoint, 0.0)
    
    def procesar(self, dt, setpoint_real, estimulo_externo=None):
        # Recibir estímulo externo (audio o comunicación)
        if estimulo_externo is not None:
            self.L.recibir_estimulo(estimulo_externo)
            self.R.recibir_estimulo(estimulo_externo)
        
        # Actualizar hemisferios
        self.L.actualizar(dt)
        self.R.actualizar(dt)
        
        # Calcular Cb (desacople)
        gradiente = abs(self.L.omega - self.R.omega)
        self.Cb = min(CB_MAX, self.Cb + gradiente * dt)
        self.Cb *= (1 - dt / TAU_CB)
        
        # Actualizar valencia según el estímulo procesado
        # La valencia se mueve hacia el setpoint que más resuena con los hemisferios
        omega_promedio = (self.L.omega + self.R.omega) / 2
        # Normalizar omega a [-1, 1] y usar para actualizar valencia
        influencia = np.tanh(omega_promedio)
        
        if setpoint_real == TRAUMA_SETPOINT:
            self.valencia[setpoint_real] += 0.01 * (influencia * 50 - self.valencia[setpoint_real]) * dt
        else:
            self.valencia[setpoint_real] += 0.01 * (-influencia * 50 - self.valencia[setpoint_real]) * dt
        
        self.valencia[setpoint_real] = np.clip(self.valencia[setpoint_real], -100, 100)
    
    def estimar_setpoint(self):
        """Estima el setpoint basado en las valencias (el que tiene mayor valencia)"""
        return max(self.valencia, key=self.valencia.get)
    
    def get_confianza(self):
        """Confianza basada en la valencia del setpoint estimado"""
        estimado = self.estimar_setpoint()
        return min(1.0, abs(self.valencia[estimado]) / 50.0)
    
    def reset(self):
        self.L = HemisferioSimple(SESGO_L)
        self.R = HemisferioSimple(SESGO_R)
        self.valencia = {sp: 0.0 for sp in SETPOINTS_TEST}
        self.Cb = 0.0
        self.tiempo_deliberacion = 0.0


# ============================================================
# FUNCIONES DE ESTÍMULO
# ============================================================
def generar_estimulo(setpoint, ruido=0.0):
    """Genera un estímulo numérico con ruido gaussiano"""
    estimulo = setpoint
    if ruido > 0:
        estimulo += np.random.normal(0, abs(setpoint) * ruido)
    return estimulo


# ============================================================
# FASE 1: BASELINE — B solo con ruido
# ============================================================
def fase_baseline(B, setpoint_real, ruido_amplitud, rondas):
    """B solo intenta estimar el setpoint a partir de estímulo ruidoso"""
    
    errores = []
    latencias = []
    valencias_hist = []
    
    for ronda in range(rondas):
        start = time.time()
        
        # Generar estímulo ruidoso para B
        estimulo = generar_estimulo(setpoint_real, ruido_amplitud)
        
        # B procesa
        B.procesar(DT, setpoint_real, estimulo)
        
        # B estima el setpoint
        estimado = B.estimar_setpoint()
        error = abs(setpoint_real - estimado)
        errores.append(error)
        
        latencias.append(time.time() - start)
        valencias_hist.append(B.get_valencia(setpoint_real))
        
        if (ronda + 1) % 50 == 0:
            print(f"        Ronda {ronda+1}: val={B.get_valencia(setpoint_real):.2f}, estimado={estimado}°, error={error}")
    
    error_medio = np.mean(errores)
    latencia_media = np.mean(latencias)
    
    return error_medio, latencia_media, valencias_hist


# ============================================================
# FASE 2: COMUNICACIÓN — A + B acoplados
# ============================================================
def fase_comunicacion(A, B, setpoint_real, ruido_amplitud, rondas):
    """A recibe estímulo limpio, B recibe ruidoso + comunicación de A"""
    
    errores = []
    latencias = []
    valencias_A = []
    valencias_B = []
    
    for ronda in range(rondas):
        start = time.time()
        
        # A: estímulo limpio (sin ruido)
        estimulo_A = generar_estimulo(setpoint_real, 0.0)
        A.procesar(DT, setpoint_real, estimulo_A)
        
        # B: estímulo ruidoso
        estimulo_B = generar_estimulo(setpoint_real, ruido_amplitud)
        B.procesar(DT, setpoint_real, estimulo_B)
        
        # COMUNICACIÓN: A envía su estimación (valencia) a B
        val_A = A.get_valencia(setpoint_real)
        
        # B recibe la valencia de A como estímulo adicional (refuerzo)
        B.procesar(DT, setpoint_real, val_A / 10.0)  # Escalar para no saturar
        
        # B estima el setpoint
        estimado_B = B.estimar_setpoint()
        error = abs(setpoint_real - estimado_B)
        errores.append(error)
        
        latencias.append(time.time() - start)
        valencias_A.append(val_A)
        valencias_B.append(B.get_valencia(setpoint_real))
        
        if (ronda + 1) % 50 == 0:
            print(f"        Ronda {ronda+1}: val_A={val_A:.2f}, val_B={B.get_valencia(setpoint_real):.2f}, estimado={estimado_B}°, error={error}")
    
    error_medio = np.mean(errores)
    latencia_media = np.mean(latencias)
    
    return error_medio, latencia_media, valencias_A, valencias_B


# ============================================================
# EXPERIMENTO PRINCIPAL
# ============================================================
def ejecutar_v182b_v3():
    print("=" * 100)
    print("EXPERIMENTO V182B-v3 — COMUNICACIÓN FUNCIONAL (ESTÍMULOS EXPLÍCITOS)")
    print("=" * 100)
    print("  SIMPLIFICACIÓN (Qwen):")
    print("    • Usar estímulos numéricos explícitos en lugar de audios")
    print("    • Validar el principio de comunicación primero")
    print("    • Luego, si funciona, reemplazar por audios reales")
    print("")
    print("  CRITERIOS DE ÉXITO:")
    print(f"    ✅ Mejora en error > {MEJORA_MIN:.0%}")
    print(f"    ✅ Aumento latencia > {LATENCIA_AUMENTO_MIN:.0%}")
    print(f"    ✅ Correlación |val_A| vs error_B > {CORRELACION_MIN}")
    print("=" * 100)

    resultados = []
    
    for setpoint_real in SETPOINTS_TEST:
        print(f"\n{'='*60}")
        print(f"PROCESANDO: Setpoint real = {setpoint_real}°")
        print(f"{'='*60}")
        
        # ============================================================
        # FASE 1: BASELINE (B solo)
        # ============================================================
        print(f"\n  FASE 1: BASELINE — B solo con ruido ({RUIDO_B_AMPLITUD*100:.0f}%)")
        
        B = OrganismoSimple("B")
        error_solo, lat_solo, val_B_solo = fase_baseline(B, setpoint_real, RUIDO_B_AMPLITUD, RONDAS_POR_CONDICION)
        
        print(f"    Error medio B solo: {error_solo:.1f}°")
        print(f"    Latencia media: {lat_solo:.4f}s")
        
        # ============================================================
        # FASE 2: COMUNICACIÓN (A + B)
        # ============================================================
        print(f"\n  FASE 2: COMUNICACIÓN — A (limpio) + B (ruidoso + ayuda de A)")
        
        A = OrganismoSimple("A")
        B = OrganismoSimple("B")
        
        error_con, lat_con, val_A_hist, val_B_hist = fase_comunicacion(
            A, B, setpoint_real, RUIDO_B_AMPLITUD, RONDAS_POR_CONDICION
        )
        
        print(f"    Error medio B con A: {error_con:.1f}°")
        print(f"    Latencia media: {lat_con:.4f}s")
        
        # ============================================================
        # MÉTRICAS
        # ============================================================
        mejora = (error_solo - error_con) / error_solo if error_solo > 0 else 0
        aumento_latencia = (lat_con - lat_solo) / lat_solo if lat_solo > 0 else 0
        
        # Correlación entre |val_A| y error_B (negativa esperada)
        val_A_abs = [abs(v) for v in val_A_hist]
        errores_B = []
        for i in range(RONDAS_POR_CONDICION):
            estimado = A.estimar_setpoint() if i < len(val_A_hist) else 0
            errores_B.append(abs(setpoint_real - estimado))
        
        if len(val_A_abs) > 10 and len(errores_B) > 10:
            correlacion = np.corrcoef(val_A_abs, errores_B)[0, 1]
            if np.isnan(correlacion):
                correlacion = 0.0
        else:
            correlacion = 0.0
        
        mejora_ok = mejora > MEJORA_MIN
        latencia_ok = aumento_latencia > LATENCIA_AUMENTO_MIN
        correlacion_ok = correlacion < -CORRELACION_MIN  # Negativa: más val_A = menos error
        
        exito = mejora_ok and latencia_ok and correlacion_ok
        
        print(f"\n  RESULTADOS:")
        print(f"    Mejora: {mejora:.1%} -> {'✅' if mejora_ok else '❌'}")
        print(f"    Aumento latencia: {aumento_latencia:.1%} -> {'✅' if latencia_ok else '❌'}")
        print(f"    Correlación |val_A| vs error_B: {correlacion:.3f} -> {'✅' if correlacion_ok else '❌'}")
        
        resultados.append({
            'setpoint': setpoint_real,
            'error_solo': error_solo,
            'error_con': error_con,
            'mejora': mejora,
            'lat_solo': lat_solo,
            'lat_con': lat_con,
            'aumento_latencia': aumento_latencia,
            'correlacion': correlacion,
            'exito': exito
        })
    
    # ============================================================
    # RESUMEN
    # ============================================================
    print("\n" + "=" * 80)
    print("RESUMEN V182B-v3 — Comunicación Funcional")
    print("=" * 80)
    
    for r in resultados:
        status = "✅" if r['exito'] else "❌"
        print(f"  {status} Setpoint {r['setpoint']}°: mejora={r['mejora']:.1%}, error: {r['error_solo']:.1f}° → {r['error_con']:.1f}°")
    
    exitos = sum(1 for r in resultados if r['exito'])
    mejora_media = np.mean([r['mejora'] for r in resultados])
    
    print(f"\n  📊 MÉTRICAS GLOBALES:")
    print(f"     Éxito en {exitos}/{len(resultados)} condiciones")
    print(f"     Mejora media: {mejora_media:.1%}")
    
    exito_global = exitos == len(resultados) and mejora_media > MEJORA_MIN
    
    print("\n" + "=" * 80)
    if exito_global:
        print("  ✅ COMUNICACIÓN FUNCIONAL DEMOSTRADA")
        print("")
        print("     El principio de comunicación funciona:")
        print("     ✓ B mejora su estimación cuando recibe ayuda de A")
        print("     ✓ La latencia aumenta (costo de procesar comunicación)")
        print("     ✓ Mayor confianza de A se correlaciona con menor error de B")
        print("")
        print("  PRÓXIMO: Reemplazar estímulos numéricos por audios reales")
    else:
        print("  ⚠️ COMUNICACIÓN FUNCIONAL NO DEMOSTRADA")
        if not any(r['exito'] for r in resultados):
            print("     Revisar parámetros: ruido, pesos de comunicación, rondas")
    print("=" * 80)
    
    # Gráficos
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Gráfico 1: Comparación de errores
    ax = axes[0]
    nombres = [f"{r['setpoint']}°" for r in resultados]
    x = np.arange(len(nombres))
    width = 0.35
    ax.bar(x - width/2, [r['error_solo'] for r in resultados], width, label='B solo', color='red', alpha=0.7)
    ax.bar(x + width/2, [r['error_con'] for r in resultados], width, label='B con A', color='green', alpha=0.7)
    ax.set_xlabel('Setpoint real')
    ax.set_ylabel('Error (°)')
    ax.set_title('Error de estimación de B')
    ax.set_xticks(x)
    ax.set_xticklabels(nombres)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 2: Mejora por condición
    ax = axes[1]
    mejoras = [r['mejora'] for r in resultados]
    colores = ['green' if m > MEJORA_MIN else 'red' for m in mejoras]
    ax.bar(nombres, mejoras, color=colores, alpha=0.7)
    ax.axhline(y=MEJORA_MIN, color='blue', linestyle='--', label=f'Umbral ({MEJORA_MIN:.0%})')
    ax.set_xlabel('Setpoint real')
    ax.set_ylabel('Mejora')
    ax.set_title('Reducción de error gracias a A')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('V182_logs', exist_ok=True)
    plt.savefig(f'V182_logs/v182b_v3_comunicacion_{ts}.png', dpi=150)
    print(f"\n  📊 Gráficos guardados: V182_logs/v182b_v3_comunicacion_{ts}.png")
    
    # Guardar datos
    raw_data = {
        'version': 'V182B-v3',
        'timestamp': ts,
        'params': {
            'RONDAS_POR_CONDICION': RONDAS_POR_CONDICION,
            'RUIDO_B_AMPLITUD': RUIDO_B_AMPLITUD,
            'PESO_ESTIMULO': PESO_ESTIMULO,
            'MEJORA_MIN': MEJORA_MIN,
            'LATENCIA_AUMENTO_MIN': LATENCIA_AUMENTO_MIN,
            'CORRELACION_MIN': CORRELACION_MIN,
        },
        'resultados': resultados
    }
    
    with open(f'V182_logs/v182b_v3_raw_{ts}.json', 'w') as f:
        json.dump(raw_data, f, indent=2)
    print(f"  📁 Datos guardados: V182_logs/v182b_v3_raw_{ts}.json")
    
    return exito_global


if __name__ == "__main__":
    start = time.time()
    exito = ejecutar_v182b_v3()
    elapsed = time.time() - start
    print(f"\n  ⏱️ Tiempo total real: {elapsed:.1f} segundos | Éxito: {exito}")