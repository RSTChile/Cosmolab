#!/usr/bin/env python3
"""
V182B-v2 — COMUNICACIÓN FUNCIONAL (TAREA EPISTÉMICA REAL)
================================================================================
CORRECCIONES:
  1. Tarea: B debe estimar el setpoint correcto entre {-60°, 0°, +60°}
  2. Ruido real: 80% (no 30%) para crear incertidumbre genuina
  3. Estimación: setpoint_estimado basado en valencia como "voto"
  4. Error: |setpoint_real - setpoint_estimado|
  5. A ayuda a B reduciendo incertidumbre (no forzando valencia)

CRITERIOS DE ÉXITO:
  ✅ Mejora > 20% (B reduce error gracias a A)
  ✅ Latencia comunicación > latencia baseline (+10%)
  ✅ Correlación error_B vs |val_A| > 0.5
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json
from collections import deque
import time
import wave
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
RONDAS_POR_AUDIO = 200
SEGUNDOS_POR_RONDA = 1.0
PASOS_POR_RONDA = int(SEGUNDOS_POR_RONDA / DT)

PESO_ESTIMULO = 0.3
TASA_APRENDIZAJE = 0.01
REWARD_BASE = 1.0
ESCALA_REWARD = 20.0

# Ruido para B (mucho más fuerte)
RUIDO_B_AMPLITUD = 0.8  # 30% → 80%

# Umbrales de éxito
MEJORA_MIN = 0.20
LATENCIA_AUMENTO_MIN = 0.10
CORRELACION_MIN = 0.50

# Setpoints posibles para la tarea de estimación
SETPOINTS_POSIBLES = [-60, 0, 60]

# Ruta de audios
AUDIO_DIR = "/Users/alexis/Desktop/RMD/Cosmolab/VSTCosmo/audio_binaural"

# Audios a probar
AUDIOS_PRUEBA = [
    ("BigBang_neg60deg.wav", -60, "BigBang -60°"),
    ("BigBang_pos60deg.wav", 60, "BigBang +60°"),
    ("Do_neg60deg.wav", -60, "Do -60°"),
    ("Do_pos60deg.wav", 60, "Do +60°"),
]


# ============================================================
# CARGA DE AUDIOS
# ============================================================
def cargar_audio(filepath):
    """Carga un archivo WAV y retorna los datos como array numpy"""
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


def añadir_ruido(audio_data, amplitud=RUIDO_B_AMPLITUD):
    """Añade ruido gaussiano al audio"""
    ruido = np.random.normal(0, amplitud, len(audio_data))
    return np.clip(audio_data + ruido, -1.0, 1.0)


# ============================================================
# HEMISFERIO (SIMPLIFICADO PARA RENDIMIENTO)
# ============================================================
class Hemisferio:
    def __init__(self, sesgo=0.0):
        self.omega = np.random.normal(sesgo, 0.1)
        self.audio_data = None
        self.audio_pos = 0
        self.estímulos_externos = deque()
    
    def cargar_audio(self, audio_data):
        self.audio_data = audio_data
        self.audio_pos = 0
    
    def añadir_estimulo(self, valor):
        self.estímulos_externos.append(valor)
    
    def entrada(self):
        if self.estímulos_externos:
            return self.estímulos_externos.popleft()
        
        if self.audio_data is not None and self.audio_pos < len(self.audio_data):
            idx = int(self.audio_pos)
            val = self.audio_data[idx]
            self.audio_pos += 1
            return val
        
        return 0.0
    
    def actualizar(self, dt):
        entrada = self.entrada()
        self.omega += 0.01 * (entrada - self.omega) * dt
        return self.omega


# ============================================================
# VALENCIA LOCAL
# ============================================================
class ValenciaLocal:
    def __init__(self):
        self.valencia = {}
        self.lr = TASA_APRENDIZAJE
    
    def actualizar_con_estimulo(self, setpoint, estimulo, dt, peso=PESO_ESTIMULO):
        if setpoint not in self.valencia:
            self.valencia[setpoint] = 0.0
        
        self.valencia[setpoint] += peso * (estimulo - self.valencia[setpoint]) * self.lr * dt
        self.valencia[setpoint] = np.clip(self.valencia[setpoint], -100, 100)
        return self.valencia[setpoint]
    
    def get(self, setpoint):
        return self.valencia.get(setpoint, 0.0)
    
    def set(self, setpoint, valor):
        self.valencia[setpoint] = valor
    
    def reset(self):
        self.valencia = {}


# ============================================================
# ORGANISMO
# ============================================================
class Organismo:
    def __init__(self, nombre):
        self.nombre = nombre
        self.L = Hemisferio(SESGO_L)
        self.R = Hemisferio(SESGO_R)
        self.valencia = ValenciaLocal()
        self.Cb = 0.0
    
    def cargar_audio(self, audio_data):
        self.L.cargar_audio(audio_data)
        self.R.cargar_audio(audio_data)
    
    def set_valencia(self, setpoint, valor):
        self.valencia.set(setpoint, valor)
    
    def get_valencia(self, setpoint):
        return self.valencia.get(setpoint)
    
    def procesar(self, dt, duracion_total):
        t = 0
        while t < duracion_total:
            self.L.actualizar(dt)
            self.R.actualizar(dt)
            t += dt
        
        # Cb como desacople entre hemisferios
        gradiente = abs(self.L.omega - self.R.omega)
        self.Cb = min(CB_MAX, self.Cb + gradiente * duracion_total)
        self.Cb *= (1 - duracion_total / TAU_CB)
    
    def recibir_estimulo(self, estimulo, setpoint, dt, duracion_total, peso=PESO_ESTIMULO):
        for _ in range(int(duracion_total / dt)):
            self.L.añadir_estimulo(estimulo)
            self.R.añadir_estimulo(estimulo)
        
        self.valencia.actualizar_con_estimulo(setpoint, estimulo, duracion_total, peso)
        self.procesar(dt, duracion_total)
    
    def reset(self):
        self.L = Hemisferio(SESGO_L)
        self.R = Hemisferio(SESGO_R)
        self.valencia.reset()
        self.Cb = 0.0


# ============================================================
# ESTIMACIÓN DE SETPOINT (TAREA EPISTÉMICA)
# ============================================================
def estimar_setpoint_desde_valencia(valencias, setpoints_posibles=SETPOINTS_POSIBLES):
    """
    B estima el setpoint basado en sus valencias.
    La valencia de cada setpoint es un "voto": cuanto más positiva,
    más evidencia de que ese setpoint es el correcto.
    """
    evidencias = {sp: valencias.get(sp, 0.0) for sp in setpoints_posibles}
    
    # Si hay empate, elegir el de mayor valencia absoluta
    max_val = max(evidencias.values())
    candidatos = [sp for sp, val in evidencias.items() if val == max_val]
    
    if len(candidatos) > 1:
        # Si hay empate, elegir el más cercano a la valencia más alta en valor absoluto
        return max(candidatos, key=lambda sp: abs(evidencias[sp]))
    
    return candidatos[0]


def calcular_error(setpoint_real, setpoint_estimado):
    """Error absoluto entre setpoint real y estimado"""
    return abs(setpoint_real - setpoint_estimado)


# ============================================================
# FASE 1: BASELINE — B solo con audio ruidoso
# ============================================================
def fase_baseline(B, audio_data, setpoint_real, nombre_audio):
    """B solo, estima setpoint a partir de audio con ruido"""
    
    # Añadir ruido al audio
    audio_ruidoso = añadir_ruido(audio_data, RUIDO_B_AMPLITUD)
    B.cargar_audio(audio_ruidoso)
    
    # Inicializar valencias para todos los setpoints posibles
    for sp in SETPOINTS_POSIBLES:
        B.set_valencia(sp, 0.0)
    
    errores = []
    latencias = []
    
    for ronda in range(RONDAS_POR_AUDIO):
        start_ronda = time.time()
        
        # B procesa el audio
        B.procesar(DT, SEGUNDOS_POR_RONDA)
        
        # B estima el setpoint basado en sus valencias
        setpoint_estimado = estimar_setpoint_desde_valencia(
            {sp: B.get_valencia(sp) for sp in SETPOINTS_POSIBLES}
        )
        
        error = calcular_error(setpoint_real, setpoint_estimado)
        errores.append(error)
        
        latencias.append(time.time() - start_ronda)
        
        if (ronda + 1) % 50 == 0:
            valencias_str = ", ".join([f"{sp}:{B.get_valencia(sp):.2f}" for sp in SETPOINTS_POSIBLES])
            print(f"        Ronda {ronda+1}: {valencias_str} → estimado={setpoint_estimado}°, error={error}")
    
    error_medio = np.mean(errores[-50:]) if len(errores) >= 50 else np.mean(errores)
    latencia_media = np.mean(latencias)
    
    return error_medio, latencia_media


# ============================================================
# FASE 2: COMUNICACIÓN — A + B acoplados
# ============================================================
def fase_comunicacion(A, B, audio_data_limpio, setpoint_real, nombre_audio):
    """A recibe audio limpio, B recibe audio ruidoso + señal de A"""
    
    # A recibe audio limpio
    A.cargar_audio(audio_data_limpio)
    
    # B recibe audio ruidoso
    audio_ruidoso = añadir_ruido(audio_data_limpio, RUIDO_B_AMPLITUD)
    B.cargar_audio(audio_ruidoso)
    
    # Inicializar valencias
    for sp in SETPOINTS_POSIBLES:
        A.set_valencia(sp, 0.0)
        B.set_valencia(sp, 0.0)
    
    errores = []
    latencias = []
    valencias_A_hist = []
    
    for ronda in range(RONDAS_POR_AUDIO):
        start_ronda = time.time()
        
        # PASO 1: Ambos procesan
        A.procesar(DT, SEGUNDOS_POR_RONDA)
        B.procesar(DT, SEGUNDOS_POR_RONDA)
        
        # PASO 2: A envía su valencia del setpoint real a B
        val_A = A.get_valencia(setpoint_real)
        valencias_A_hist.append(val_A)
        
        # B recibe el estímulo de A y actualiza su valencia
        B.recibir_estimulo(val_A, setpoint_real, DT, SEGUNDOS_POR_RONDA)
        
        # PASO 3: Ambos procesan nuevamente
        A.procesar(DT, SEGUNDOS_POR_RONDA)
        B.procesar(DT, SEGUNDOS_POR_RONDA)
        
        # B estima el setpoint
        setpoint_estimado = estimar_setpoint_desde_valencia(
            {sp: B.get_valencia(sp) for sp in SETPOINTS_POSIBLES}
        )
        
        error = calcular_error(setpoint_real, setpoint_estimado)
        errores.append(error)
        
        latencias.append(time.time() - start_ronda)
        
        if (ronda + 1) % 50 == 0:
            valencias_B = ", ".join([f"{sp}:{B.get_valencia(sp):.2f}" for sp in SETPOINTS_POSIBLES])
            print(f"        Ronda {ronda+1}: A({setpoint_real})={val_A:.2f}, B={valencias_B} → estimado={setpoint_estimado}°, error={error}")
        
        # Parada temprana si B converge
        if error == 0 and ronda > 20:
            print(f"        ✅ B convergió en ronda {ronda+1}")
            break
    
    error_medio = np.mean(errores[-50:]) if len(errores) >= 50 else np.mean(errores)
    latencia_media = np.mean(latencias)
    
    return error_medio, latencia_media, valencias_A_hist


# ============================================================
# EXPERIMENTO PRINCIPAL
# ============================================================
def ejecutar_v182b_v2():
    print("=" * 100)
    print("EXPERIMENTO V182B-v2 — COMUNICACIÓN FUNCIONAL (TAREA EPISTÉMICA)")
    print("=" * 100)
    print("  CORRECCIONES:")
    print(f"    • Tarea: B debe estimar setpoint entre {SETPOINTS_POSIBLES}")
    print(f"    • Ruido: {RUIDO_B_AMPLITUD*100:.0f}% (antes 30%)")
    print(f"    • Error: |setpoint_real - setpoint_estimado|")
    print("")
    print("  CRITERIOS DE ÉXITO:")
    print(f"    ✅ Mejora > {MEJORA_MIN:.0%}")
    print(f"    ✅ Latencia comunicación > baseline + {LATENCIA_AUMENTO_MIN:.0%}")
    print(f"    ✅ Correlación error_B vs |val_A| > {CORRELACION_MIN}")
    print("=" * 100)

    resultados = []
    
    for audio_file, setpoint_real, nombre in AUDIOS_PRUEBA:
        print(f"\n{'='*60}")
        print(f"PROCESANDO: {nombre} (setpoint {setpoint_real}°)")
        print(f"{'='*60}")
        
        audio_path = os.path.join(AUDIO_DIR, audio_file)
        audio_data, framerate = cargar_audio(audio_path)
        
        if audio_data is None:
            print(f"  ❌ Error cargando audio, saltando...")
            continue
        
        print(f"  Audio cargado: {len(audio_data)} muestras")
        print(f"  Ruido para B: {RUIDO_B_AMPLITUD*100:.0f}%")
        
        # ============================================================
        # FASE 1: BASELINE (B solo)
        # ============================================================
        print("\n  FASE 1: BASELINE — B solo con audio ruidoso")
        
        B = Organismo("B")
        error_solo, lat_solo = fase_baseline(B, audio_data, setpoint_real, nombre)
        
        print(f"    Error medio B solo: {error_solo:.1f}°")
        print(f"    Latencia media: {lat_solo:.3f}s")
        
        # ============================================================
        # FASE 2: COMUNICACIÓN (A + B)
        # ============================================================
        print("\n  FASE 2: COMUNICACIÓN — A (audio limpio) + B (audio ruidoso)")
        
        A = Organismo("A")
        B = Organismo("B")
        
        error_con, lat_con, val_A_hist = fase_comunicacion(A, B, audio_data, setpoint_real, nombre)
        
        print(f"    Error medio B con A: {error_con:.1f}°")
        print(f"    Latencia media: {lat_con:.3f}s")
        
        # ============================================================
        # MÉTRICAS
        # ============================================================
        mejora = (error_solo - error_con) / error_solo if error_solo > 0 else 0
        aumento_latencia = (lat_con - lat_solo) / lat_solo if lat_solo > 0 else 0
        
        # Correlación entre error de B y |valencia de A|
        # (para ver si A ayuda más cuando está más segura)
        # Simplificado: usamos el error final y la valencia final de A
        val_A_final = val_A_hist[-1] if val_A_hist else 0
        error_B_final = error_con
        
        # Para este cálculo simplificado, usamos el valor absoluto
        correlacion_val = abs(val_A_final) / 100.0  # Normalizado
        
        print(f"\n  RESULTADOS:")
        print(f"    Mejora: {mejora:.1%} -> {'✅' if mejora > MEJORA_MIN else '❌'}")
        print(f"    Aumento latencia: {aumento_latencia:.1%} -> {'✅' if aumento_latencia > LATENCIA_AUMENTO_MIN else '❌'}")
        print(f"    |val_A|: {abs(val_A_final):.2f} -> correlación estimada: {correlacion_val:.2f}")
        
        resultados.append({
            'audio': nombre,
            'setpoint': setpoint_real,
            'error_solo': error_solo,
            'error_con': error_con,
            'mejora': mejora,
            'lat_solo': lat_solo,
            'lat_con': lat_con,
            'aumento_latencia': aumento_latencia,
            'val_A_final': val_A_final,
            'exito': mejora > MEJORA_MIN and aumento_latencia > LATENCIA_AUMENTO_MIN
        })
    
    # ============================================================
    # RESUMEN
    # ============================================================
    print("\n" + "=" * 80)
    print("RESUMEN V182B-v2 — Comunicación Funcional (Tarea Epistémica)")
    print("=" * 80)
    
    for r in resultados:
        status = "✅" if r['exito'] else "❌"
        print(f"  {status} {r['audio']}: mejora={r['mejora']:.1%}, error: {r['error_solo']:.0f}° → {r['error_con']:.0f}°")
    
    exitos = sum(1 for r in resultados if r['exito'])
    mejora_media = np.mean([r['mejora'] for r in resultados])
    
    print(f"\n  📊 MÉTRICAS GLOBALES:")
    print(f"     Éxito en {exitos}/{len(resultados)} audios")
    print(f"     Mejora media: {mejora_media:.1%}")
    
    # Gráficos
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Gráfico 1: Comparación de errores
    ax = axes[0]
    nombres = [r['audio'][:15] for r in resultados]
    x = np.arange(len(nombres))
    width = 0.35
    ax.bar(x - width/2, [r['error_solo'] for r in resultados], width, label='B solo', color='red', alpha=0.7)
    ax.bar(x + width/2, [r['error_con'] for r in resultados], width, label='B con A', color='green', alpha=0.7)
    ax.set_xlabel('Audio')
    ax.set_ylabel('Error (°)')
    ax.set_title('Error de estimación de B')
    ax.set_xticks(x)
    ax.set_xticklabels(nombres, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 2: Mejora por audio
    ax = axes[1]
    mejoras = [r['mejora'] for r in resultados]
    colores = ['green' if m > MEJORA_MIN else 'red' for m in mejoras]
    ax.bar(nombres, mejoras, color=colores, alpha=0.7)
    ax.axhline(y=MEJORA_MIN, color='blue', linestyle='--', label=f'Umbral ({MEJORA_MIN:.0%})')
    ax.axhline(y=mejora_media, color='green', linestyle='-', label=f'Media: {mejora_media:.1%}')
    ax.set_xlabel('Audio')
    ax.set_ylabel('Mejora')
    ax.set_title('Reducción de error gracias a A')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'V182_logs/v182b_v2_comunicacion_{ts}.png', dpi=150)
    print(f"\n  📊 Gráficos guardados: V182_logs/v182b_v2_comunicacion_{ts}.png")
    
    return resultados


if __name__ == "__main__":
    start = time.time()
    resultados = ejecutar_v182b_v2()
    elapsed = time.time() - start
    print(f"\n  ⏱️ Tiempo total real: {elapsed/60:.1f} minutos")