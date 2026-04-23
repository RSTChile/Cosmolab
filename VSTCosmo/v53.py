#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experimento Cosmosemiótico v5.3 - Modo dominante y supervivencia global
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

FS = 48000
VENTANA_MS = 30
HOP_MS = 10
VENTANA_MUESTRAS = int(FS * VENTANA_MS / 1000)
HOP_MUESTRAS = int(FS * HOP_MS / 1000)

N_BANDAS = 32
F_MIN = 50
F_MAX = FS // 2
BANDAS = np.logspace(np.log10(F_MIN), np.log10(F_MAX), N_BANDAS + 1)
BANDAS_VOZ = (BANDAS[:-1] >= 100) & (BANDAS[:-1] <= 4000)

def cargar_wav_seguro(nombre):
    resultado = wav.read(nombre)
    if len(resultado) == 2:
        sr, data = resultado
    else:
        sr, data, _ = resultado
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    if data.ndim == 2:
        data = data.mean(axis=1)
    return sr, data

def espectrograma_con_energia(audio, ventana, hop):
    n_ventanas = (len(audio) - ventana) // hop + 1
    espectro = np.zeros((N_BANDAS, n_ventanas))
    rms = np.zeros(n_ventanas)
    for i in range(n_ventanas):
        inicio = i * hop
        fragmento = audio[inicio:inicio + ventana]
        rms[i] = np.sqrt(np.mean(fragmento ** 2))
        ventana_hann = np.hanning(len(fragmento))
        fragmento = fragmento * ventana_hann
        fft = np.fft.rfft(fragmento)
        potencia = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(len(fragmento), 1/FS)
        for b in range(N_BANDAS):
            mask = (freqs >= BANDAS[b]) & (freqs < BANDAS[b+1])
            if np.any(mask):
                espectro[b, i] = np.mean(potencia[mask])
        max_banda = np.max(espectro[:, i])
        if max_banda > 0:
            espectro[:, i] /= max_banda
    return espectro, rms

def correlacion_coseno(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class SistemaV53:
    def __init__(self, perfil_voz=None):
        self.perfil_voz = perfil_voz
        self.campo_actual = None
        self.campo_anterior = None
        self.acoplamiento = 0.5
        self.acoplamiento_anterior = 0.5
        self.historial_modos = []
        self.historial_acoplamiento = []
        self.historial_rms = []
    
    def procesar_ventana(self, espectro, rms):
        self.campo_anterior = self.campo_actual
        self.campo_actual = espectro[:, 0] if espectro.ndim > 1 else espectro
        
        # Calcular acoplamiento
        if self.perfil_voz is not None:
            acop = correlacion_coseno(self.campo_actual, self.perfil_voz)
            acop = max(0.0, min(1.0, acop))
            self.acoplamiento = 0.9 * self.acoplamiento_anterior + 0.1 * acop
            self.acoplamiento_anterior = self.acoplamiento
        
        # Determinar modo por ventana (sin histéresis)
        if rms < 0.001:
            modo = "SILENCIO"
        elif self.acoplamiento > 0.55:
            modo = "ZONA_VOZ"
        elif self.acoplamiento > 0.35:
            modo = "DISCREPO"
        else:
            modo = "NO_SE"
        
        self.historial_modos.append(modo)
        self.historial_acoplamiento.append(self.acoplamiento)
        self.historial_rms.append(rms)
    
    def modo_dominante(self):
        if not self.historial_modos:
            return "DESCONOCIDO"
        # Excluir SILENCIO del cómputo del modo dominante (solo importa cuando hay señal)
        modos_sin_silencio = [m for m in self.historial_modos if m != "SILENCIO"]
        if not modos_sin_silencio:
            return "SILENCIO"
        return Counter(modos_sin_silencio).most_common(1)[0][0]
    
    def esta_vivo(self):
        if len(self.historial_acoplamiento) < 100:
            return True
        
        # Excluir ventanas de silencio del análisis de supervivencia
        mascara_voz = np.array(self.historial_rms) > 0.005
        if not np.any(mascara_voz):
            return False  # Nunca hubo señal significativa
        
        acop_con_voz = np.array(self.historial_acoplamiento)[mascara_voz]
        
        # Si el acoplamiento promedio en zonas con voz es bajo, colapsa
        if np.mean(acop_con_voz) < 0.4:
            return False
        
        return True
    
    def resumen(self):
        return {
            'vivo': self.esta_vivo(),
            'modo_dominante': self.modo_dominante(),
            'acoplamiento_promedio': np.mean(self.historial_acoplamiento),
            'porcentaje_voz': 100 * np.mean(np.array(self.historial_rms) > 0.005)
        }

# ============================================================
# EXPERIMENTO PRINCIPAL
# ============================================================

def main():
    print("=" * 60)
    print("EXPERIMENTO COSMOSEMIÓTICO v5.3")
    print("= Modo dominante y supervivencia global =")
    print("=" * 60)
    
    # Cargar archivos
    print("\n[1] Cargando archivos...")
    sr, voz_estudio = cargar_wav_seguro('Voz_Estudio.wav')
    _, voz_viento1 = cargar_wav_seguro('Voz+Viento_1.wav')
    _, voz_viento2 = cargar_wav_seguro('Voz+Viento_2.wav')
    _, viento_puro = cargar_wav_seguro('Viento.wav')
    
    print(f"  Frecuencia: {sr} Hz")
    
    # Recortar
    min_len = min(len(voz_estudio), len(voz_viento1), len(voz_viento2), len(viento_puro))
    voz_estudio = voz_estudio[:min_len]
    voz_viento1 = voz_viento1[:min_len]
    voz_viento2 = voz_viento2[:min_len]
    viento_puro = viento_puro[:min_len]
    
    # Calcular perfil de referencia
    print("\n[2] Calibrando referencia de voz...")
    espectro_ref, _ = espectrograma_con_energia(voz_estudio, VENTANA_MUESTRAS, HOP_MUESTRAS)
    perfil_voz = np.mean(espectro_ref, axis=1)
    print(f"  Perfil de voz calculado ({len(perfil_voz)} bandas)")
    
    # Experimentos
    print("\n[3] Ejecutando experimentos...")
    resultados = []
    
    for nombre, audio in [("Viento_Puro", viento_puro),
                          ("Voz_Viento_1", voz_viento1),
                          ("Voz_Viento_2", voz_viento2)]:
        print(f"\n--- {nombre} ---")
        
        espectro, rms = espectrograma_con_energia(audio, VENTANA_MUESTRAS, HOP_MUESTRAS)
        sistema = SistemaV53(perfil_voz=perfil_voz)
        
        for i in range(espectro.shape[1]):
            sistema.procesar_ventana(espectro[:, i:i+1], rms[i])
        
        res = sistema.resumen()
        resultados.append((nombre, res['vivo'], res['modo_dominante'], 
                          res['acoplamiento_promedio'], res['porcentaje_voz']))
        
        print(f"  Resultado: {'✅ VIVO' if res['vivo'] else '❌ COLAPSADO'}")
        print(f"  Modo dominante: {res['modo_dominante']}")
        print(f"  Acoplamiento promedio: {res['acoplamiento_promedio']:.3f}")
        print(f"  % ventanas con voz: {res['porcentaje_voz']:.1f}%")
    
    # Resumen final
    print("\n" + "=" * 60)
    print("RESUMEN FINAL")
    print("=" * 60)
    print(f"{'Archivo':<18} {'Estado':<12} {'Modo dominante':<14} {'Acoplamiento':<12} {'% Voz':<8}")
    print("-" * 60)
    
    for nombre, vivo, modo, acop, pct_voz in resultados:
        estado = "✅ VIVO" if vivo else "❌ COLAPSADO"
        print(f"{nombre:<18} {estado:<12} {modo:<14} {acop:.3f}       {pct_voz:.1f}%")
    
    print("\n" + "=" * 60)
    print("CRITERIOS DE ÉXITO:")
    print("  Viento_Puro  → ❌ COLAPSADO (poco voz, acoplamiento bajo)")
    print("  Voz_Viento_1 → ✅ VIVO (voz presente, acoplamiento > 0.4)")
    print("  Voz_Viento_2 → ✅ VIVO (voz presente, acoplamiento > 0.4)")
    print("=" * 60)

if __name__ == "__main__":
    main()