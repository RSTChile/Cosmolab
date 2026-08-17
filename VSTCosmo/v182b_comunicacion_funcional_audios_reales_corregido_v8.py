#!/usr/bin/env python3
"""
V182B-v8 — COMUNICACIÓN FUNCIONAL CON AUDIOS REALES (CORREGIDO)
================================================================================
CORRECCIONES:
  1. Extraer evidencia por CORRELACIÓN con firma, no por amplitud media
  2. Firmas por frecuencia (220 Hz = -60°, 440 Hz = 0°, 880 Hz = +60°)
  3. Confianza basada en la fuerza de la evidencia, no automática
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json
import time
import wave

# ============================================================
# PARÁMETROS
# ============================================================
SETPOINTS = [-60, 0, 60]
RUIDO_B_STD = 0.3        # Ruido gaussiano para B (menor, para no destruir correlación)
DROP_OUT_B = 0.5         # 50% dropout (menor que antes)
PESO_A = 0.7
PESO_B = 0.3

RONDAS_B_SOLO = 30
RONDAS_B_CON_A = 200
SEGUNDOS_POR_RONDA = 1.0
DT = 0.01
PASOS_POR_RONDA = int(SEGUNDOS_POR_RONDA / DT)
FRAMERATE = 48000

# Audios a probar
AUDIO_DIR = "/Users/alexis/Desktop/RMD/Cosmolab/VSTCosmo/audio_binaural"
AUDIOS_PRUEBA = [
    ("BigBang_neg60deg.wav", -60),
    ("BigBang_pos60deg.wav", 60),
]

# Umbrales de éxito
MEJORA_ERROR_MIN = 0.20
EXITOS_PARCIALES_MIN = 2

# ============================================================
# FIRMAS POR FRECUENCIA
# ============================================================
def generar_firma_setpoint(setpoint, n_samples, freq_base=440):
    """Genera firma sinusoidal para cada setpoint"""
    if setpoint == -60:
        freq = freq_base * 0.5   # 220 Hz
    elif setpoint == 0:
        freq = freq_base          # 440 Hz
    else:  # +60
        freq = freq_base * 2      # 880 Hz
    t = np.arange(n_samples) / FRAMERATE
    return np.sin(2 * np.pi * freq * t)


# ============================================================
# CARGA DE AUDIOS
# ============================================================
def cargar_audio(filepath):
    try:
        with wave.open(filepath, 'rb') as wf:
            n_frames = wf.getnframes()
            framerate = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            
            frames = wf.readframes(n_frames)
            
            if sampwidth == 2:
                audio_data = np.frombuffer(frames, dtype=np.int16)
                audio_data = audio_data / 32768.0
            else:
                return None, None
            
            if n_channels == 2:
                audio_data = audio_data.reshape(-1, 2).mean(axis=1)
            
            return audio_data, framerate
    except Exception as e:
        print(f"      Error: {e}")
        return None, None


# ============================================================
# EXTRAER EVIDENCIA POR CORRELACIÓN
# ============================================================
def extraer_evidencia(audio_segment, setpoints=SETPOINTS):
    """Extrae evidencia correlacionando el audio con firmas de cada setpoint"""
    n_samples = len(audio_segment)
    evidencias = {}
    
    for sp in setpoints:
        firma = generar_firma_setpoint(sp, n_samples)
        # Correlación de Pearson
        corr = np.corrcoef(audio_segment, firma)[0, 1]
        if np.isnan(corr):
            corr = 0.0
        evidencias[sp] = corr
    
    return evidencias


def evidencia_a_estimacion(evidencias):
    """Convierte evidencias en estimación (setpoint con mayor correlación)"""
    mejor_sp = max(evidencias, key=evidencias.get)
    confianza = abs(evidencias[mejor_sp])
    return mejor_sp, confianza


# ============================================================
# AGENTE PARA AUDIOS REALES
# ============================================================
class AgenteAudio:
    def __init__(self, nombre):
        self.nombre = nombre
        self.estimacion = 0.0
        self.confianza = 0.0
        self.audio_completo = None
        self.posicion = 0
    
    def cargar_audio(self, audio_data):
        self.audio_completo = audio_data
        self.posicion = 0
    
    def procesar_ronda(self, usar_dropout=False):
        """Procesa una ronda (1s de audio) y actualiza estimación"""
        # Extraer segmento de 1 segundo
        segmento = self.audio_completo[self.posicion:self.posicion + FRAMERATE]
        self.posicion += FRAMERATE
        
        if len(segmento) < FRAMERATE:
            # Si no hay suficiente audio, volver al inicio
            self.posicion = 0
            segmento = self.audio_completo[:FRAMERATE]
        
        # Aplicar dropout si es necesario
        if usar_dropout and np.random.random() < DROP_OUT_B:
            segmento = np.zeros_like(segmento)
        
        # Añadir ruido gaussiano
        segmento += np.random.normal(0, RUIDO_B_STD, len(segmento))
        segmento = np.clip(segmento, -1, 1)
        
        # Extraer evidencia por correlación
        evidencias = extraer_evidencia(segmento)
        estimado, confianza = evidencia_a_estimacion(evidencias)
        
        # Actualizar estado interno
        self.estimacion = estimado
        self.confianza = confianza
        
        return self.estimacion, self.confianza, evidencias
    
    def recibir_comunicacion(self, estimacion_otro, confianza_otro):
        """Fusiona la estimación de A con la propia (bayesiana)"""
        # Si la confianza de A es alta, B se mueve hacia esa estimación
        if confianza_otro > self.confianza:
            self.estimacion = (1 - PESO_A) * self.estimacion + PESO_A * estimacion_otro
            self.confianza = max(self.confianza, confianza_otro * 0.8)
    
    def reset(self):
        self.estimacion = 0.0
        self.confianza = 0.0
        self.posicion = 0


# ============================================================
# EXPERIMENTO PRINCIPAL
# ============================================================
def ejecutar_v182b_v8():
    print("=" * 100)
    print("EXPERIMENTO V182B-v8 — COMUNICACIÓN FUNCIONAL CON AUDIOS REALES (CORREGIDO)")
    print("=" * 100)
    print("  CORRECCIONES:")
    print("    • Evidencia por CORRELACIÓN con firma (no amplitud media)")
    print("    • Firmas por frecuencia: 220Hz (-60°), 440Hz (0°), 880Hz (+60°)")
    print("    • Confianza basada en fuerza de correlación")
    print("")
    print("  CRITERIOS DE ÉXITO:")
    print(f"    ✅ Mejora media en error > {MEJORA_ERROR_MIN:.0%}")
    print(f"    ✅ Éxito en ≥ {EXITOS_PARCIALES_MIN}/2 audios")
    print("=" * 100)

    resultados = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('V182_logs', exist_ok=True)

    for audio_file, setpoint_real in AUDIOS_PRUEBA:
        print(f"\n{'='*60}")
        print(f"PROCESANDO: {audio_file} (setpoint {setpoint_real}°)")
        print(f"{'='*60}")
        
        audio_path = os.path.join(AUDIO_DIR, audio_file)
        audio_data, framerate = cargar_audio(audio_path)
        if audio_data is None:
            print(f"  ❌ Error cargando audio, saltando...")
            continue
        
        print(f"  Audio cargado: {len(audio_data)} muestras")
        
        # ============================================================
        # FASE 1: BASELINE (B solo)
        # ============================================================
        print(f"\n  FASE 1: BASELINE — B solo ({RONDAS_B_SOLO} rondas)")
        
        B = AgenteAudio("B")
        B.cargar_audio(audio_data)
        
        errores_B_solo = []
        for _ in range(RONDAS_B_SOLO):
            estimado, conf, _ = B.procesar_ronda(usar_dropout=True)
            error = abs(estimado - setpoint_real)
            errores_B_solo.append(error)
        
        error_medio_B_solo = np.mean(errores_B_solo)
        print(f"    Error B solo: {error_medio_B_solo:.1f}°")
        print(f"    Confianza B solo: {B.confianza:.1%}")
        
        # ============================================================
        # FASE 2: COMUNICACIÓN (A + B)
        # ============================================================
        print(f"\n  FASE 2: COMUNICACIÓN — A + B ({RONDAS_B_CON_A} rondas)")
        
        A = AgenteAudio("A")
        B = AgenteAudio("B")
        A.cargar_audio(audio_data)
        B.cargar_audio(audio_data)
        
        errores_B_con_A = []
        
        for ronda in range(RONDAS_B_CON_A):
            # A: sin dropout (escucha limpio)
            estimado_A, conf_A, _ = A.procesar_ronda(usar_dropout=False)
            
            # B: con dropout
            estimado_B, conf_B, _ = B.procesar_ronda(usar_dropout=True)
            
            # Comunicación: A envía su estimación a B
            B.recibir_comunicacion(estimado_A, conf_A)
            
            error = abs(B.estimacion - setpoint_real)
            errores_B_con_A.append(error)
        
        error_medio_B_con_A = np.mean(errores_B_con_A)
        print(f"    Error B con A: {error_medio_B_con_A:.1f}°")
        print(f"    Confianza B con A: {B.confianza:.1%}")
        
        # Métricas
        mejora_error = (error_medio_B_solo - error_medio_B_con_A) / error_medio_B_solo if error_medio_B_solo > 0 else 0
        exito_parcial = mejora_error > MEJORA_ERROR_MIN
        
        print(f"\n  RESULTADOS:")
        print(f"    Mejora en error: {mejora_error:.1%} -> {'✅' if exito_parcial else '❌'}")
        
        resultados.append({
            'audio': audio_file,
            'setpoint': setpoint_real,
            'error_solo': error_medio_B_solo,
            'error_con_A': error_medio_B_con_A,
            'mejora_error': mejora_error,
            'exito': exito_parcial
        })
    
    # ============================================================
    # RESUMEN
    # ============================================================
    print("\n" + "=" * 80)
    print("RESUMEN V182B-v8 — Comunicación Funcional con Audios Reales")
    print("=" * 80)
    
    for r in resultados:
        status = "✅" if r['exito'] else "❌"
        print(f"  {status} {r['audio']}: error {r['error_solo']:.1f}° → {r['error_con_A']:.1f}° (mejora={r['mejora_error']:.1%})")
    
    mejoras = [r['mejora_error'] for r in resultados]
    mejora_media = np.mean(mejoras) if mejoras else 0
    exitos_parciales = sum(1 for r in resultados if r['exito'])
    
    print(f"\n  📊 MÉTRICAS GLOBALES:")
    print(f"     Mejora media: {mejora_media:.1%} (>{MEJORA_ERROR_MIN:.0%}) -> {'✅' if mejora_media > MEJORA_ERROR_MIN else '❌'}")
    print(f"     Éxito en {exitos_parciales}/{len(resultados)} audios (≥{EXITOS_PARCIALES_MIN}) -> {'✅' if exitos_parciales >= EXITOS_PARCIALES_MIN else '❌'}")
    
    exito_global = (mejora_media > MEJORA_ERROR_MIN) and (exitos_parciales >= EXITOS_PARCIALES_MIN)
    
    print("\n" + "=" * 80)
    if exito_global:
        print("  ✅ COMUNICACIÓN FUNCIONAL DEMOSTRADA CON AUDIOS REALES")
    else:
        print("  ⚠️ COMUNICACIÓN FUNCIONAL NO DEMOSTRADA CON AUDIOS REALES")
        print("     Verificar: las firmas de frecuencia deben coincidir con el contenido del audio")
    print("=" * 80)
    
    return exito_global


if __name__ == "__main__":
    start = time.time()
    exito = ejecutar_v182b_v8()
    elapsed = time.time() - start
    print(f"\n  ⏱️ Tiempo total real: {elapsed/60:.1f} minutos | Éxito: {exito}")