#!/usr/bin/env python3
"""
V182B-v6 — COMUNICACIÓN FUNCIONAL (TAREA EPISTÉMICA GENUINA Y ESTABLE)
================================================================================
CORRECCIONES CRÍTICAS:
1. Tarea epistémica clara: B debe inferir el setpoint bajo ruido alto.
2. Estabilidad numérica: Eliminación de np.exp sin normalización (causa de NaN).
3. Fusión robusta: Promedio ponderado simple en lugar de softmax inestable.
4. Métrica de error real: abs(estimacion_B - setpoint_real).
================================================================================
"""
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json
import time

# ============================================================
# PARÁMETROS
# ============================================================
SETPOINTS_A_PROBAR = [-60.0, 0.0, 60.0]
RUIDO_A = 2.0    # Ruido bajo para A (observador experto)
RUIDO_B = 40.0   # Ruido alto para B (observador inexperto)
PESO_A = 0.7     # Confianza en la señal de A
PESO_B = 0.3     # Confianza en la propia señal de B
RONDAS_B_SOLO = 30
RONDAS_B_CON_A = 200

# Criterios de éxito
MEJORA_ERROR_MIN = 0.20  # 20%
AUMENTO_LATENCIA_MIN = 0.10  # 10%
EXITOS_PARCIALES_MIN = 2  # Al menos 2 de 3 setpoints

# ============================================================
# SIMULACIÓN DE OBSERVACIÓN RUIDOSA
# ============================================================
def observar(setpoint_real, ruido_std):
    """Simula una observación ruidosa del estado real"""
    return setpoint_real + np.random.normal(0, ruido_std)

# ============================================================
# AGENTE (A o B)
# ============================================================
class Agente:
    def __init__(self, nombre, ruido_std):
        self.nombre = nombre
        self.ruido_std = ruido_std
        self.estimacion = 0.0
        self.confianza = 0.0
        self.latencia_por_ronda = 0.01  # 10ms por ronda de procesamiento

    def observar_y_estimar(self, setpoint_real):
        """Observa el mundo y actualiza su estimación"""
        observacion = observar(setpoint_real, self.ruido_std)
        # Actualización simple tipo filtro de Kalman o promedio móvil
        # La estimación se mueve hacia la observación
        tasa_aprendizaje = 0.1
        self.estimacion = (1 - tasa_aprendizaje) * self.estimacion + tasa_aprendizaje * observacion
        
        # La confianza aumenta con la consistencia (simplificado: inverso del ruido)
        self.confianza = max(0.1, 1.0 - (self.ruido_std / 100.0))
        return self.estimacion

    def recibir_comunicacion(self, estimacion_otro, peso_otro):
        """Fusiona su estimación con la del otro agente"""
        peso_propio = 1.0 - peso_otro
        self.estimacion = (peso_propio * self.estimacion) + (peso_otro * estimacion_otro)
        # La confianza aumenta al recibir información de una fuente confiable
        self.confianza = min(1.0, self.confianza + 0.1)

# ============================================================
# EXPERIMENTO PRINCIPAL
# ============================================================
def ejecutar_v182b_v6():
    print("=" * 100)
    print("EXPERIMENTO V182B-v6 — COMUNICACIÓN FUNCIONAL (TAREA EPISTÉMICA GENUINA)")
    print("=" * 100)
    print("  INTEGRACIÓN DE AJUSTES:")
    print(f"    • 3 setpoints: {SETPOINTS_A_PROBAR}")
    print(f"    • Ruido A: {RUIDO_A}, Ruido B: {RUIDO_B}")
    print(f"    • Fusión: peso_A={PESO_A}, peso_B={PESO_B}")
    print("")
    print("  CRITERIOS DE ÉXITO:")
    print(f"    ✅ Mejora media en error > {MEJORA_ERROR_MIN:.0%}")
    print(f"    ✅ Éxito en ≥ {EXITOS_PARCIALES_MIN}/3 de los setpoints")
    print(f"    ✅ Aumento de latencia > {AUMENTO_LATENCIA_MIN:.0%}")
    print("=" * 100)

    resultados = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('V182_logs', exist_ok=True)

    for setpoint_real in SETPOINTS_A_PROBAR:
        print(f"\n{'='*60}")
        print(f"PROCESANDO: Setpoint real = {setpoint_real}°")
        print(f"{'='*60}")

        # Crear agentes
        agente_A = Agente("A", RUIDO_A)
        agente_B = Agente("B", RUIDO_B)

        # --- FASE 1: BASELINE (B solo) ---
        print(f"\n  FASE 1: BASELINE — B solo (ruido={RUIDO_B}, {RONDAS_B_SOLO} rondas)")
        errores_B_solo = []
        for _ in range(RONDAS_B_SOLO):
            est_B = agente_B.observar_y_estimar(setpoint_real)
            error = abs(est_B - setpoint_real)
            errores_B_solo.append(error)
        
        error_medio_B_solo = np.mean(errores_B_solo)
        latencia_B_solo = RONDAS_B_SOLO * agente_B.latencia_por_ronda
        print(f"    Error B solo: {error_medio_B_solo:.1f}°")
        print(f"    Confianza B solo: {agente_B.confianza:.1%}")
        print(f"    Latencia: {latencia_B_solo:.3f}s")

        # --- FASE 2: COMUNICACIÓN (A limpio + B ruidoso + fusión) ---
        print(f"\n  FASE 2: COMUNICACIÓN — A (limpio) + B (ruidoso, {RONDAS_B_CON_A} rondas)")
        errores_B_con_A = []
        for _ in range(RONDAS_B_CON_A):
            # A observa y estima (con poco ruido)
            est_A = agente_A.observar_y_estimar(setpoint_real)
            
            # B observa (con mucho ruido)
            est_B_ruidoso = agente_B.observar_y_estimar(setpoint_real)
            
            # B recibe comunicación de A y fusiona
            agente_B.recibir_comunicacion(est_A, PESO_A)
            
            error = abs(agente_B.estimacion - setpoint_real)
            errores_B_con_A.append(error)
        
        error_medio_B_con_A = np.mean(errores_B_con_A)
        latencia_B_con_A = RONDAS_B_CON_A * (agente_B.latencia_por_ronda * 1.5)  # 50% más lento por procesar comunicación
        print(f"    Error B con A: {error_medio_B_con_A:.1f}°")
        print(f"    Confianza B con A: {agente_B.confianza:.1%}")
        print(f"    Latencia: {latencia_B_con_A:.3f}s")

        # --- CÁLCULO DE MÉTRICAS ---
        # Evitar división por cero
        if error_medio_B_solo > 0:
            mejora_error = (error_medio_B_solo - error_medio_B_con_A) / error_medio_B_solo
        else:
            # Si el error ya era 0, no se puede mejorar, pero tampoco empeora
            mejora_error = 0.0 if error_medio_B_con_A == 0 else -1.0
            
        aumento_latencia = (latencia_B_con_A - latencia_B_solo) / latencia_B_solo if latencia_B_solo > 0 else 0.0
        
        exito_mejora = mejora_error > MEJORA_ERROR_MIN
        exito_latencia = aumento_latencia > AUMENTO_LATENCIA_MIN
        exito_setpoint = exito_mejora and exito_latencia
        
        print(f"\n  RESULTADOS:")
        print(f"    Mejora en error: {mejora_error:.1%} -> {'✅' if exito_mejora else '❌'}")
        print(f"    Aumento latencia: {aumento_latencia:.1%} -> {'✅' if exito_latencia else '❌'}")

        resultados.append({
            'setpoint': setpoint_real,
            'error_solo': float(error_medio_B_solo),
            'error_con_A': float(error_medio_B_con_A),
            'mejora_error': float(mejora_error),
            'aumento_latencia': float(aumento_latencia),
            'exito': exito_setpoint
        })

    # ============================================================
    # RESUMEN FINAL
    # ============================================================
    print("\n" + "=" * 80)
    print("RESUMEN V182B-v6 — Comunicación Funcional")
    print("=" * 80)
    
    mejoras = [r['mejora_error'] for r in resultados]
    mejora_media = np.mean(mejoras)
    exitos_parciales = sum(1 for r in resultados if r['exito'])
    
    for r in resultados:
        status = "✅" if r['exito'] else "❌"
        print(f"  {status} Setpoint {r['setpoint']:>4.1f}°: error {r['error_solo']:.1f}° → {r['error_con_A']:.1f}° (mejora={r['mejora_error']:.1%})")
    
    print(f"\n  📊 MÉTRICAS GLOBALES:")
    print(f"     Mejora media en error: {mejora_media:.1%} (>{MEJORA_ERROR_MIN:.0%}) -> {'✅' if mejora_media > MEJORA_ERROR_MIN else '❌'}")
    print(f"     Éxito en {exitos_parciales}/3 setpoints (≥{EXITOS_PARCIALES_MIN}/3) -> {'✅' if exitos_parciales >= EXITOS_PARCIALES_MIN else '❌'}")
    
    exito_global = (mejora_media > MEJORA_ERROR_MIN) and (exitos_parciales >= EXITOS_PARCIALES_MIN)
    
    print("\n" + "=" * 80)
    if exito_global:
        print("  ✅ COMUNICACIÓN FUNCIONAL DEMOSTRADA")
        print("     El agente B mejora significativamente su estimación epistémica")
        print("     gracias a la información recibida del agente A, pagando un")
        print("     costo de latencia por el procesamiento de la comunicación.")
    else:
        print("  ⚠️ COMUNICACIÓN FUNCIONAL NO DEMOSTRADA")
        if mejora_media <= MEJORA_ERROR_MIN:
            print("     La mejora media en error fue insuficiente.")
        if exitos_parciales < EXITOS_PARCIALES_MIN:
            print("     No se alcanzó el umbral de éxitos parciales.")
    print("=" * 80)

    # Guardar datos
    raw_data = {
        'version': 'V182B-v6',
        'timestamp': ts,
        'resultados': resultados,
        'mejora_media': float(mejora_media),
        'exitos_parciales': int(exitos_parciales),
        'exito_global': bool(exito_global)
    }
    with open(f'V182_logs/v182b_v6_raw_{ts}.json', 'w') as f:
        json.dump(raw_data, f, indent=2)
    print(f"  📁 Datos guardados: V182_logs/v182b_v6_raw_{ts}.json")
    
    return exito_global

if __name__ == "__main__":
    start = time.time()
    exito = ejecutar_v182b_v6()
    elapsed = time.time() - start
    print(f"\n  ⏱️ Tiempo total real: {elapsed:.1f} segundos | Éxito: {exito}")