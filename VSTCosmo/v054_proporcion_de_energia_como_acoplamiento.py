#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experimento Cosmosemiótico v5.4 - Proporción de energía como acoplamiento
"""

import numpy as np
import scipy.io.wavfile as wav
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
BANDAS_VIENTO = (BANDAS[:-1] <= 200)

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

def energia_por_bandas(audio, ventana, hop):
    n_ventanas = (len(audio) - ventana) // hop + 1
    energia_voz = np.zeros(n_ventanas)
    energia_viento = np.zeros(n_ventanas)
    energia_total = np.zeros(n_ventanas)
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
                e_banda = np.mean(potencia[mask])
                if BANDAS_VOZ[b]:
                    energia_voz[i] += e_banda
                if BANDAS_VIENTO[b]:
                    energia_viento[i] += e_banda
                energia_total[i] += e_banda
        
        if energia_total[i] == 0:
            energia_total[i] = 0.01
    
    return energia_voz, energia_viento, energia_total, rms

class SistemaV54:
    def __init__(self):
        self.historial_acoplamiento = []
        self.historial_modos = []
        self.historial_rms = []
    
    def calcular_acoplamiento(self, energia_voz, energia_total):
        if energia_total > 0:
            return energia_voz / energia_total
        return 0.5
    
    def procesar_ventana(self, energia_voz, energia_viento, energia_total, rms):
        acop = self.calcular_acoplamiento(energia_voz, energia_total)
        self.historial_acoplamiento.append(acop)
        self.historial_rms.append(rms)
        
        # Determinar modo
        if rms < 0.001:
            modo = "SILENCIO"
        elif acop > 0.5:
            modo = "ZONA_VOZ"
        elif acop > 0.3:
            modo = "DISCREPO"
        else:
            modo = "NO_SE"
        
        self.historial_modos.append(modo)
    
    def modo_dominante(self):
        if not self.historial_modos:
            return "DESCONOCIDO"
        from collections import Counter
        modos_sin_silencio = [m for m in self.historial_modos if m != "SILENCIO"]
        if not modos_sin_silencio:
            return "SILENCIO"
        return Counter(modos_sin_silencio).most_common(1)[0][0]
    
    def esta_vivo(self):
        if len(self.historial_acoplamiento) < 100:
            return True
        
        # Excluir silencios
        mascara_voz = np.array(self.historial_rms) > 0.005
        if not np.any(mascara_voz):
            return False
        
        acop_con_voz = np.array(self.historial_acoplamiento)[mascara_voz]
        
        # Viento puro debería tener acoplamiento bajo (< 0.4)
        # Voz debería tener acoplamiento alto ( > 0.5)
        if np.mean(acop_con_voz) > 0.48:
            return True
        return False
    
    def resumen(self):
        return {
            'vivo': self.esta_vivo(),
            'modo_dominante': self.modo_dominante(),
            'acoplamiento_promedio': np.mean(self.historial_acoplamiento),
            'acoplamiento_con_voz': np.mean(np.array(self.historial_acoplamiento)[np.array(self.historial_rms) > 0.005]) if np.any(np.array(self.historial_rms) > 0.005) else 0
        }

def main():
    print("=" * 60)
    print("EXPERIMENTO COSMOSEMIÓTICO v5.4 - Proporción de energía")
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
    
    # Experimentos
    print("\n[2] Ejecutando experimentos...")
    resultados = []
    
    for nombre, audio in [("Viento_Puro", viento_puro),
                          ("Voz_Viento_1", voz_viento1),
                          ("Voz_Viento_2", voz_viento2)]:
        print(f"\n--- {nombre} ---")
        
        evoz, eviento, etotal, rms = energia_por_bandas(audio, VENTANA_MUESTRAS, HOP_MUESTRAS)
        sistema = SistemaV54()
        
        for i in range(len(evoz)):
            sistema.procesar_ventana(evoz[i], eviento[i], etotal[i], rms[i])
        
        res = sistema.resumen()
        resultados.append((nombre, res['vivo'], res['modo_dominante'], 
                          res['acoplamiento_promedio'], res['acoplamiento_con_voz']))
        
        print(f"  Resultado: {'✅ VIVO' if res['vivo'] else '❌ COLAPSADO'}")
        print(f"  Modo dominante: {res['modo_dominante']}")
        print(f"  Acoplamiento promedio total: {res['acoplamiento_promedio']:.3f}")
        print(f"  Acoplamiento solo en zonas con voz: {res['acoplamiento_con_voz']:.3f}")
    
    # Resumen final
    print("\n" + "=" * 60)
    print("RESUMEN FINAL")
    print("=" * 60)
    print(f"{'Archivo':<18} {'Estado':<12} {'Modo':<12} {'Acoplamiento (solo voz)':<22}")
    print("-" * 60)
    
    for nombre, vivo, modo, acop_total, acop_voz in resultados:
        estado = "✅ VIVO" if vivo else "❌ COLAPSADO"
        print(f"{nombre:<18} {estado:<12} {modo:<12} {acop_voz:.3f}")
    
    print("\n" + "=" * 60)
    print("CRITERIOS DE ÉXITO:")
    print("  Viento_Puro  → ❌ COLAPSADO (acoplamiento < 0.48)")
    print("  Voz_Viento_1 → ✅ VIVO (acoplamiento > 0.48)")
    print("  Voz_Viento_2 → ✅ VIVO (acoplamiento > 0.48)")
    print("=" * 60)

if __name__ == "__main__":
    main()