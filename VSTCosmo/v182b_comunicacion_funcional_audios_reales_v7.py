#!/usr/bin/env python3
"""
V182B-v7 — COMUNICACIÓN FUNCIONAL CON AUDIOS REALES
================================================================================
BASE: V182B-v6 (éxito con estímulos numéricos)
EXTENSIÓN: Reemplazar números por audios reales del directorio /audio_binaural
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
SETPOINTS_A_PROBAR = [-60.0, 0.0, 60.0]
RUIDO_B_STD = 40.0    # Ruido gaussiano para B (cuando se añade al audio)
DROP_OUT_B = 0.8      # 80% dropout: B solo procesa 20% del audio
PESO_A = 0.7          # Confianza en la señal de A
PESO_B = 0.3          # Confianza en la propia señal de B

RONDAS_B_SOLO = 30
RONDAS_B_CON_A = 200
SEGUNDOS_POR_RONDA = 1.0
DT = 0.01
PASOS_POR_RONDA = int(SEGUNDOS_POR_RONDA / DT)

# Audios a probar
AUDIO_DIR = "/Volumes/LaCie/RMD/Cosmolab/VSTCosmo/audio_binaural"
AUDIOS_PRUEBA = [
    ("BigBang_neg60deg.wav", -60, "BigBang -60°"),
    ("BigBang_pos60deg.wav", 60, "BigBang +60°"),
    ("Silencio", None, "Silencio (control)"),
]

# Umbrales de éxito
MEJORA_ERROR_MIN = 0.20
EXITOS_PARCIALES_MIN = 2

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
            comptype = wf.getcompname()
            
            if comptype != 'not compressed':
                return None, None
            
            frames = wf.readframes(n_frames)
            
            if sampwidth == 2:
                audio_data = np.frombuffer(frames, dtype=np.int16)
                audio_data = audio_data / 32768.0
            elif sampwidth == 1:
                audio_data = np.frombuffer(frames, dtype=np.uint8)
                audio_data = (audio_data - 128) / 128.0
            else:
                return None, None
            
            if n_channels == 2:
                audio_data = audio_data.reshape(-1, 2).mean(axis=1)
            
            return audio_data, framerate
    except Exception as e:
        print(f"      Error: {e}")
        return None, None


def generar_silencio(duracion=5.0, framerate=48000):
    n = int(duracion * framerate)
    return np.zeros(n), framerate


def aplicar_dropout(audio_data, dropout_rate=DROP_OUT_B):
    """Aplica dropout: mantiene solo (1-dropout) del audio, resto ruido"""
    if dropout_rate <= 0:
        return audio_data
    
    resultado = audio_data.copy()
    mask = np.random.random(len(audio_data)) > dropout_rate
    resultado[~mask] = 0
    return resultado


def añadir_ruido_gaussiano(audio_data, std=RUIDO_B_STD / 100.0):
    """Añade ruido gaussiano al audio"""
    ruido = np.random.normal(0, std, len(audio_data))
    return np.clip(audio_data + ruido, -1.0, 1.0)


# ============================================================
# AGENTE PARA AUDIOS REALES
# ============================================================
class AgenteAudio:
    def __init__(self, nombre, ruido_std=RUIDO_B_STD):
        self.nombre = nombre
        self.ruido_std = ruido_std
        self.estimacion = 0.0
        self.confianza = 0.6
        self.audio_actual = None
        self.framerate = 48000
    
    def cargar_audio(self, audio_data, framerate):
        self.audio_actual = audio_data
        self.framerate = framerate
        self.posicion = 0
    
    def _extraer_muestra(self):
        if self.audio_actual is None or self.posicion >= len(self.audio_actual):
            return 0.0
        val = self.audio_actual[self.posicion]
        self.posicion += 1
        return val
    
    def procesar_ronda(self, setpoint_real, usar_dropout=False):
        """Procesa una ronda completa (1 segundo de audio)"""
        # Acumular evidencia del audio durante la ronda
        muestra = 0.0
        for _ in range(PASOS_POR_RONDA):
            muestra += self._extraer_muestra()
        
        # La evidencia es el promedio del audio procesado
        evidencia = muestra / PASOS_POR_RONDA
        
        # Aplicar dropout si es necesario
        if usar_dropout and np.random.random() < DROP_OUT_B:
            evidencia = 0.0
        
        # Añadir ruido gaussiano
        evidencia += np.random.normal(0, self.ruido_std / 100.0)
        
        # Actualizar estimación
        tasa = 0.1
        self.estimacion = (1 - tasa) * self.estimacion + tasa * (evidencia * 60.0)
        self.estimacion = np.clip(self.estimacion, -100, 100)
        
        # Actualizar confianza
        self.confianza = min(0.9, self.confianza + 0.01)
        
        return self.estimacion
    
    def recibir_comunicacion(self, estimacion_otro, peso):
        self.estimacion = (1 - peso) * self.estimacion + peso * estimacion_otro
        self.confianza = min(0.9, self.confianza + 0.05)
    
    def reset(self):
        self.estimacion = 0.0
        self.confianza = 0.6
        self.posicion = 0


# ============================================================
# EXPERIMENTO PRINCIPAL
# ============================================================
def ejecutar_v182b_v7():
    print("=" * 100)
    print("EXPERIMENTO V182B-v7 — COMUNICACIÓN FUNCIONAL CON AUDIOS REALES")
    print("=" * 100)
    print("  EXTENSIÓN: Replicar V182B-v6 (éxito) con audios reales binaurales")
    print("")
    print("  CRITERIOS DE ÉXITO:")
    print(f"    ✅ Mejora media en error > {MEJORA_ERROR_MIN:.0%}")
    print(f"    ✅ Éxito en ≥ {EXITOS_PARCIALES_MIN}/3 de los setpoints")
    print("=" * 100)

    resultados = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('V182_logs', exist_ok=True)

    for audio_file, setpoint_real, nombre in AUDIOS_PRUEBA:
        print(f"\n{'='*60}")
        print(f"PROCESANDO: {nombre} (setpoint {setpoint_real}°)")
        print(f"{'='*60}")
        
        # Cargar audio
        if audio_file is None:
            audio_data, framerate = generar_silencio(duracion=5.0)
            print(f"  Silencio generado (control)")
        else:
            audio_path = os.path.join(AUDIO_DIR, audio_file)
            audio_data, framerate = cargar_audio(audio_path)
            if audio_data is None:
                print(f"  ❌ Error cargando audio, saltando...")
                continue
            print(f"  Audio cargado: {len(audio_data)} muestras, {framerate}Hz")
        
        # Crear agentes
        agente_A = AgenteAudio("A", ruido_std=2.0)
        agente_B = AgenteAudio("B", ruido_std=RUIDO_B_STD)
        
        # Copiar audio para A y B
        agente_A.cargar_audio(audio_data, framerate)
        agente_B.cargar_audio(audio_data, framerate)
        
        # --- FASE 1: BASELINE (B solo) ---
        print(f"\n  FASE 1: BASELINE — B solo ({RONDAS_B_SOLO} rondas)")
        
        agente_B.reset()
        agente_B.cargar_audio(audio_data, framerate)
        
        errores_B_solo = []
        for ronda in range(RONDAS_B_SOLO):
            est_B = agente_B.procesar_ronda(setpoint_real, usar_dropout=True)
            error = abs(est_B - setpoint_real)
            errores_B_solo.append(error)
        
        error_medio_B_solo = np.mean(errores_B_solo)
        latencia_B_solo = RONDAS_B_SOLO * SEGUNDOS_POR_RONDA
        print(f"    Error B solo: {error_medio_B_solo:.1f}°")
        print(f"    Confianza B solo: {agente_B.confianza:.1%}")
        print(f"    Latencia: {latencia_B_solo:.1f}s")
        
        # --- FASE 2: COMUNICACIÓN (A + B) ---
        print(f"\n  FASE 2: COMUNICACIÓN — A + B ({RONDAS_B_CON_A} rondas)")
        
        agente_A.reset()
        agente_B.reset()
        agente_A.cargar_audio(audio_data, framerate)
        agente_B.cargar_audio(audio_data, framerate)
        
        errores_B_con_A = []
        
        for ronda in range(RONDAS_B_CON_A):
            # A: sin dropout (escucha limpio)
            est_A = agente_A.procesar_ronda(setpoint_real, usar_dropout=False)
            
            # B: con dropout
            est_B = agente_B.procesar_ronda(setpoint_real, usar_dropout=True)
            
            # Comunicación: A envía su estimación a B
            agente_B.recibir_comunicacion(est_A, PESO_A)
            
            error = abs(agente_B.estimacion - setpoint_real)
            errores_B_con_A.append(error)
        
        error_medio_B_con_A = np.mean(errores_B_con_A)
        latencia_B_con_A = RONDAS_B_CON_A * SEGUNDOS_POR_RONDA * 1.5
        print(f"    Error B con A: {error_medio_B_con_A:.1f}°")
        print(f"    Confianza B con A: {agente_B.confianza:.1%}")
        print(f"    Latencia: {latencia_B_con_A:.1f}s")
        
        # Métricas
        if error_medio_B_solo > 0:
            mejora_error = (error_medio_B_solo - error_medio_B_con_A) / error_medio_B_solo
        else:
            mejora_error = 0.0
        
        exito_parcial = mejora_error > MEJORA_ERROR_MIN
        
        print(f"\n  RESULTADOS:")
        print(f"    Mejora en error: {mejora_error:.1%} -> {'✅' if exito_parcial else '❌'}")
        
        resultados.append({
            'audio': nombre,
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
    print("RESUMEN V182B-v7 — Comunicación Funcional con Audios Reales")
    print("=" * 80)
    
    for r in resultados:
        status = "✅" if r['exito'] else "❌"
        print(f"  {status} {r['audio']}: error {r['error_solo']:.1f}° → {r['error_con_A']:.1f}° (mejora={r['mejora_error']:.1%})")
    
    mejoras = [r['mejora_error'] for r in resultados if r['setpoint'] is not None]
    mejora_media = np.mean(mejoras) if mejoras else 0
    exitos_parciales = sum(1 for r in resultados if r['exito'])
    
    print(f"\n  📊 MÉTRICAS GLOBALES:")
    print(f"     Mejora media: {mejora_media:.1%} (>{MEJORA_ERROR_MIN:.0%}) -> {'✅' if mejora_media > MEJORA_ERROR_MIN else '❌'}")
    print(f"     Éxito en {exitos_parciales}/{len(resultados)} audios (≥{EXITOS_PARCIALES_MIN}) -> {'✅' if exitos_parciales >= EXITOS_PARCIALES_MIN else '❌'}")
    
    exito_global = (mejora_media > MEJORA_ERROR_MIN) and (exitos_parciales >= EXITOS_PARCIALES_MIN)
    
    print("\n" + "=" * 80)
    if exito_global:
        print("  ✅ COMUNICACIÓN FUNCIONAL DEMOSTRADA CON AUDIOS REALES")
        print("")
        print("     El agente B mejora su estimación del setpoint")
        print("     cuando recibe información del agente A,")
        print("     incluso cuando el estímulo es un audio real binaural.")
        print("")
        print("  → V182B validado con audios reales")
    else:
        print("  ⚠️ COMUNICACIÓN FUNCIONAL NO DEMOSTRADA CON AUDIOS REALES")
        print("     Revisar: dropout, ruido, o integración audio-valencia")
    print("=" * 80)
    
    return exito_global


if __name__ == "__main__":
    start = time.time()
    exito = ejecutar_v182b_v7()
    elapsed = time.time() - start
    print(f"\n  ⏱️ Tiempo total real: {elapsed:.1f} segundos | Éxito: {exito}")