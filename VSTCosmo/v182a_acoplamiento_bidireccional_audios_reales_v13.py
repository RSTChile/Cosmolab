#!/usr/bin/env python3
"""
V182A-v9-AUDIO — ACOPLAMIENTO BIDIRECCIONAL CON AUDIOS REALES
================================================================================
BASE: V182A-v9 (funciona, tiempo real)
MODIFICACIÓN: Reemplazar ruido rosa/clicks por audios reales del directorio
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

# ============================================================
# PARÁMETROS (IDÉNTICOS A V182A-v9)
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
SEGUNDOS_POR_RONDA = 1.0
PASOS_POR_RONDA = int(SEGUNDOS_POR_RONDA / DT)

PESO_ESTIMULO = 0.3
TASA_APRENDIZAJE = 0.01
REWARD_BASE = 1.0
ESCALA_REWARD = 20.0
PARADA_TEMPRANA_DIF = 5.0
RANGO_VALENCIA_INICIAL = 50.0

# Umbrales
REDUCCION_MIN = 0.40
DIFERENCIA_FINAL_MAX = 20.0
ESTABILIZACION_MAX = 5.0
MOVIMIENTO_MIN = 12.0
CORRELACION_CB_MIN = 0.30

MEMORIA_CAPACIDAD = 10

# Ruta de audios
AUDIO_DIR = "/Volumes/LaCie/RMD/Cosmolab/VSTCosmo/audio_binaural"


# ============================================================
# CARGA DE AUDIOS REALES
# ============================================================
def cargar_audio_real(filepath):
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
            
            # Si es estéreo, convertir a mono
            if n_channels == 2:
                audio_data = audio_data.reshape(-1, 2).mean(axis=1)
            
            return audio_data, framerate
    except Exception as e:
        print(f"      Error cargando {filepath}: {e}")
        return None, None


def generar_ruido_rosa_duracion(duracion, sr):
    """Genera ruido rosa de una duración específica (reemplazo para generar_ruido_rosa)"""
    n = int(duracion * sr)
    ruido = np.random.normal(0, 1, n)
    fft = np.fft.rfft(ruido)
    freqs = np.fft.rfftfreq(n, 1/sr)
    filtro = 1.0 / np.sqrt(freqs + 0.01)
    fft_filtrado = fft * filtro
    ruido_rosa = np.fft.irfft(fft_filtrado, n=n)
    return ruido_rosa / (np.max(np.abs(ruido_rosa)) + 1e-10)


def generar_clicks_poisson_duracion(duracion, tasa=0.5, sr=48000):
    """Genera clicks de Poisson de una duración específica"""
    n = int(duracion * sr)
    clicks = np.zeros(n)
    n_clicks = int(duracion * tasa)
    for _ in range(n_clicks):
        pos = int(np.random.exponential(1.0/tasa) * sr)
        if pos < n:
            clicks[pos] = 1.0
    return clicks


# ============================================================
# MEMORIA RELACIONAL
# ============================================================
class MemoriaRelacional:
    def __init__(self, capacidad=MEMORIA_CAPACIDAD):
        self.capacidad = capacidad
        self.historial = deque(maxlen=capacidad)
    
    def almacenar(self, ronda, resultado_otro):
        self.historial.append((ronda, resultado_otro))
    
    def comparar_con_anterior(self, resultado_actual):
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
# HEMISFERIO (MODIFICADO: RECIBE AUDIOS REALES)
# ============================================================
class Hemisferio:
    def __init__(self, nombre, tau, seed=None, sesgo=0.0):
        if seed is not None:
            np.random.seed(seed)
        self.nombre = nombre
        self.tau = tau
        self.sesgo = sesgo
        self.Phi = np.random.normal(sesgo, 0.1, DIM_HEMISFERIO)
        self.Phi_vel = np.zeros(DIM_HEMISFERIO)
        self.entrada = None  # Audio real cargado
        self.sr = 48000
        self.en_inanicion = False
        self.factor_inanicion = 1.0
        self.estímulos_externos = deque()
        self.audio_data = None
        self.audio_pos = 0
    
    def cargar_audio(self, audio_data):
        """Carga un audio real para ser usado como entrada"""
        self.audio_data = audio_data
        self.audio_pos = 0
    
    def añadir_estimulo(self, valor):
        self.estímulos_externos.append(valor)
    
    def _calcular_omega(self):
        return np.mean(self.Phi[:DIM_HEMISFERIO])
    
    def entrada_t(self, t):
        # Priorizar estímulos de otro organismo
        if self.estímulos_externos:
            return self.estímulos_externos.popleft()
        
        # Si hay audio real cargado, reproducirlo
        if self.audio_data is not None and self.audio_pos < len(self.audio_data):
            idx = int(self.audio_pos)
            val = self.audio_data[idx]
            self.audio_pos += 1
            return val
        
        # Si no hay audio, usar ruido rosa (comportamiento original)
        if self.entrada is None:
            self.entrada = generar_ruido_rosa_duracion(1.0, self.sr)
        idx = int(t * self.sr)
        if idx >= len(self.entrada):
            return 0.0
        return self.entrada[idx] * self.factor_inanicion
    
    def actualizar(self, t, dt, duracion_total, otro_hemisferio=None):
        entrada = self.entrada_t(t)
        
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
        self.audio_data = None
        self.audio_pos = 0


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
        
        self.valencia[key] += peso * (estimulo - self.valencia[key]) * self.lr * dt
        
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
# ORGANISMO COMPLETO
# ============================================================
class OrganismoCompleto:
    def __init__(self, seed, nombre):
        self.nombre = nombre
        self.seed = seed
        
        self.L = Hemisferio("L", 30, seed, SESGO_L)
        self.R = Hemisferio("R", 300, seed+100, SESGO_R)
        self.BL = Hemisferio("BL", 30, seed+200, SESGO_L)
        self.BR = Hemisferio("BR", 300, seed+300, SESGO_R)
        self.hemisferios = [self.L, self.R, self.BL, self.BR]
        
        self.Cb = 0.0
        self.D = 0.0
        self.valencia = ValenciaLocal()
        self.memoria_relacional = MemoriaRelacional()
        
        self.historial_valencia = []
        self.historial_Cb = []
        self.historial_D = []
    
    def cargar_audio(self, audio_data):
        """Carga un audio real en todos los hemisferios"""
        for h in self.hemisferios:
            h.cargar_audio(audio_data)
    
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
    
    def procesar_senal(self, setpoint, dt, duracion_total):
        t = 0
        while t < duracion_total:
            for h in self.hemisferios:
                h.actualizar(t, dt, duracion_total, None)
            t += dt
        
        omega_L = self.L._calcular_omega()
        omega_R = self.R._calcular_omega()
        gradiente = omega_L - omega_R
        self.Cb = min(CB_MAX, self.Cb + abs(gradiente) * duracion_total)
        self.Cb *= (1 - duracion_total / TAU_CB)
        
        val_habito = self.valencia.get(HABITO_SETPOINT)
        val_trauma = self.valencia.get(TRAUMA_SETPOINT)
        self.D = min(1.0, abs(val_habito - val_trauma) / 100.0)
    
    def recibir_estimulo(self, estimulo, setpoint, dt, duracion_total, peso=PESO_ESTIMULO, recompensa=0.0):
        for h in self.hemisferios:
            h.añadir_estimulo(estimulo)
        
        t = 0
        while t < duracion_total:
            for h in self.hemisferios:
                h.actualizar(t, dt, duracion_total, None)
            t += dt
        
        self.valencia.actualizar_con_estimulo(setpoint, estimulo, duracion_total, peso, recompensa)
        
        omega_L = self.L._calcular_omega()
        omega_R = self.R._calcular_omega()
        gradiente = omega_L - omega_R
        self.Cb = min(CB_MAX, self.Cb + abs(gradiente) * duracion_total)
        self.Cb *= (1 - duracion_total / TAU_CB)
        
        val_habito = self.valencia.get(HABITO_SETPOINT)
        val_trauma = self.valencia.get(TRAUMA_SETPOINT)
        self.D = min(1.0, abs(val_habito - val_trauma) / 100.0)
    
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
# BUFFER DE ACOPLAMIENTO
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
        return 0.0
    
    def reset(self):
        self.recompensa_acumulada = 0.0


# ============================================================
# RONDA DE ACOPLAMIENTO
# ============================================================
def ronda_acoplamiento(A, B, setpoint, ronda_num, dt=DT, duracion=SEGUNDOS_POR_RONDA):
    # PASO 1
    A.procesar_senal(setpoint, dt, duracion)
    B.procesar_senal(setpoint, dt, duracion)
    
    # PASO 2
    resultado_A = A.obtener_resultado(setpoint)
    resultado_B = B.obtener_resultado(setpoint)
    
    A.memoria_relacional.almacenar(ronda_num, resultado_B)
    B.memoria_relacional.almacenar(ronda_num, resultado_A)
    
    A.recibir_estimulo(resultado_B, setpoint, dt, duracion)
    B.recibir_estimulo(resultado_A, setpoint, dt, duracion)
    
    # PASO 3
    A.procesar_senal(setpoint, dt, duracion)
    B.procesar_senal(setpoint, dt, duracion)
    
    # PASO 4
    nuevo_resultado_A = A.obtener_resultado(setpoint)
    nuevo_resultado_B = B.obtener_resultado(setpoint)
    
    # PASO 5
    comparacion_A, diff_A = A.memoria_relacional.comparar_con_anterior(nuevo_resultado_B)
    comparacion_B, diff_B = B.memoria_relacional.comparar_con_anterior(nuevo_resultado_A)
    
    # Almacenar nuevos
    A.memoria_relacional.almacenar(ronda_num + 0.5, nuevo_resultado_B)
    B.memoria_relacional.almacenar(ronda_num + 0.5, nuevo_resultado_A)
    
    # PASO 6
    if comparacion_A is not None:
        A.recibir_estimulo(diff_B, setpoint, dt, duracion, peso=PESO_ESTIMULO * 1.5)
    if comparacion_B is not None:
        B.recibir_estimulo(diff_A, setpoint, dt, duracion, peso=PESO_ESTIMULO * 1.5)
    
    return nuevo_resultado_A, nuevo_resultado_B, comparacion_A, comparacion_B, diff_A, diff_B


# ============================================================
# EXPERIMENTO PRINCIPAL
# ============================================================
def ejecutar_v182a_v9_audio():
    print("=" * 100)
    print("EXPERIMENTO V182A-v9-AUDIO — ACOPLAMIENTO CON AUDIOS REALES")
    print("=" * 100)
    print("  BASE: V182A-v9 (funciona, tiempo real)")
    print("  MODIFICACIÓN: Reemplazar estímulos sintéticos por audios reales")
    print("")
    print(f"  TIEMPO REAL:")
    print(f"    • {SEGUNDOS_POR_RONDA}s por ronda")
    print(f"    • {RONDAS_ACP} rondas máximas")
    print(f"    • Tiempo total: ~{RONDAS_ACP * SEGUNDOS_POR_RONDA / 60:.1f} minutos")
    print("")
    print("  CRITERIOS DE ÉXITO (idénticos a V182A-v9):")
    print(f"    ✅ Reducción de diferencia > {REDUCCION_MIN:.0%}")
    print(f"    ✅ Diferencia final < {DIFERENCIA_FINAL_MAX}")
    print(f"    ✅ Estabilización < {ESTABILIZACION_MAX}")
    print(f"    ✅ Simetría: ambos se movieron > {MOVIMIENTO_MIN}")
    print(f"    ✅ Correlación Cb(A,B) > {CORRELACION_CB_MIN}")
    print("=" * 100)

    # AUDIOS A PROBAR
    audios_a_probar = [
        ("BigBang_neg60deg.wav", -60, "BigBang -60°"),
        ("BigBang_pos60deg.wav", 60, "BigBang +60°"),
        ("Do_neg60deg.wav", -60, "Do -60°"),
        ("Do_pos60deg.wav", 60, "Do +60°"),
    ]
    
    resultados = []
    
    for audio_file, setpoint, nombre in audios_a_probar:
        print(f"\n{'='*60}")
        print(f"PROCESANDO: {nombre}")
        print(f"{'='*60}")
        
        audio_path = os.path.join(AUDIO_DIR, audio_file)
        audio_data, framerate = cargar_audio_real(audio_path)
        
        if audio_data is None:
            print(f"  ❌ Error cargando audio: {audio_file}")
            continue
        
        print(f"  Audio cargado: {len(audio_data)} muestras, {framerate}Hz")
        print(f"  Setpoint: {setpoint}°")
        
        # Crear organismos
        A = OrganismoCompleto(SEMILLA_A, "A")
        B = OrganismoCompleto(SEMILLA_B, "B")
        
        # Cargar audio en los organismos
        A.cargar_audio(audio_data)
        B.cargar_audio(audio_data)
        
        # Condiciones iniciales
        A.set_estado_inicial(setpoint, -RANGO_VALENCIA_INICIAL/2, Cb=50.0, D=0.6)
        B.set_estado_inicial(setpoint, +RANGO_VALENCIA_INICIAL/2, Cb=10.0, D=0.2)
        
        print(f"  Condiciones iniciales: A={A.get_valencia(setpoint):.2f}, B={B.get_valencia(setpoint):.2f}")
        print(f"  Ejecutando {RONDAS_ACP} rondas...")
        
        # Variables para tracking
        historial_A = []
        historial_B = []
        diferencias = []
        cb_A_hist = []
        cb_B_hist = []
        buffer = BufferAcoplamiento()
        
        start_time = time.time()
        
        for ronda in range(RONDAS_ACP):
            val_A, val_B, _, _, diff_A, diff_B = ronda_acoplamiento(A, B, setpoint, ronda)
            
            historial_A.append(val_A)
            historial_B.append(val_B)
            diferencia_actual = abs(val_A - val_B)
            diferencias.append(diferencia_actual)
            
            A.registrar_estado()
            B.registrar_estado()
            cb_A_hist.append(A.Cb)
            cb_B_hist.append(B.Cb)
            
            if ronda > 0:
                recompensa = buffer.calcular_recompensa(diferencia_actual, diferencias[-2])
                if recompensa > 0:
                    buffer.recompensa_acumulada += recompensa
                    A.recibir_estimulo(recompensa, setpoint, DT, 0.1, recompensa=recompensa)
                    B.recibir_estimulo(recompensa, setpoint, DT, 0.1, recompensa=recompensa)
            
            if (ronda + 1) % 100 == 0:
                elapsed = time.time() - start_time
                print(f"    Ronda {ronda+1}: A={val_A:.2f}, B={val_B:.2f}, diff={diferencia_actual:.2f}, tiempo={elapsed/60:.1f}min")
            
            if diferencia_actual < PARADA_TEMPRANA_DIF and ronda > 20:
                print(f"  ✅ Convergencia temprana en ronda {ronda+1}")
                break
        
        elapsed = time.time() - start_time
        
        # Análisis
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
            correlacion_cb = 0.0 if np.isnan(correlacion_cb) else correlacion_cb
        else:
            correlacion_cb = 0.0
        
        exito_reduccion = reduccion > REDUCCION_MIN
        exito_diferencia = diferencia_final < DIFERENCIA_FINAL_MAX
        exito_estabilizacion = estabilizacion < ESTABILIZACION_MAX
        exito_simetria = simetria
        exito_correlacion = correlacion_cb > CORRELACION_CB_MIN
        
        exito = exito_reduccion and exito_diferencia and exito_estabilizacion and exito_simetria and exito_correlacion
        
        print(f"\n  RESULTADOS:")
        print(f"    Diferencia inicial: {diferencia_inicial:.2f}")
        print(f"    Diferencia final: {diferencia_final:.2f}")
        print(f"    Reducción: {reduccion:.1%} -> {'✅' if exito_reduccion else '❌'}")
        print(f"    Estabilización: {estabilizacion:.2f} -> {'✅' if exito_estabilizacion else '❌'}")
        print(f"    Movimiento A: {movimiento_A:.2f}, B: {movimiento_B:.2f} -> {'✅' if exito_simetria else '❌'}")
        print(f"    Correlación Cb(A,B): {correlacion_cb:.3f} -> {'✅' if exito_correlacion else '❌'}")
        print(f"    Recompensa acumulada: {buffer.recompensa_acumulada:.4f}")
        print(f"    ÉXITO: {'✅' if exito else '❌'}")
        print(f"    Tiempo real: {elapsed/60:.1f} minutos")
        
        resultados.append({
            'audio': nombre,
            'setpoint': setpoint,
            'reduccion': reduccion,
            'diferencia_final': diferencia_final,
            'estabilizacion': estabilizacion,
            'correlacion_cb': correlacion_cb,
            'exito': exito
        })
    
    # Resumen final
    print("\n" + "=" * 80)
    print("RESUMEN V182A-v9-AUDIO")
    print("=" * 80)
    
    for r in resultados:
        status = "✅" if r['exito'] else "❌"
        print(f"  {status} {r['audio']}: reducción={r['reduccion']:.1%}, diff_final={r['diferencia_final']:.2f}")
    
    exitos = sum(1 for r in resultados if r['exito'])
    print(f"\n  ÉXITO EN {exitos}/{len(resultados)} audios")
    print("=" * 80)
    
    return resultados


if __name__ == "__main__":
    start = time.time()
    resultados = ejecutar_v182a_v9_audio()
    elapsed = time.time() - start
    print(f"\n  ⏱️ Tiempo total real: {elapsed/60:.1f} minutos")