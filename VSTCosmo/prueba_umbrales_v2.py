#!/usr/bin/env python3
# Prueba de estrés del umbral 0.48 - excluyendo silencios
import numpy as np
import scipy.io.wavfile as wav

FS = 48000
VENTANA_MS = 30
HOP_MS = 10
VENTANA_MUESTRAS = int(FS * VENTANA_MS / 1000)
HOP_MUESTRAS = int(FS * HOP_MS / 1000)

def acoplamiento_por_ventana(audio):
    n = (len(audio) - VENTANA_MUESTRAS) // HOP_MUESTRAS + 1
    acops = []
    rms_vals = []
    for i in range(n):
        inicio = i * HOP_MUESTRAS
        fragmento = audio[inicio:inicio + VENTANA_MUESTRAS]
        rms = np.sqrt(np.mean(fragmento ** 2))
        rms_vals.append(rms)
        
        fft = np.fft.rfft(fragmento * np.hanning(len(fragmento)))
        pot = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(len(fragmento), 1/FS)
        
        energia_voz = np.sum(pot[(freqs >= 100) & (freqs <= 4000)])
        energia_total = np.sum(pot)
        
        if energia_total > 0:
            acops.append(energia_voz / energia_total)
        else:
            acops.append(0.5)
    return np.array(acops), np.array(rms_vals)

umbrales = [0.45, 0.48, 0.50, 0.55]
archivos = ['Viento.wav', 'Voz+Viento_1.wav', 'Voz+Viento_2.wav', 'Voz_Estudio.wav']

print("Prueba de estrés del umbral de acoplamiento (excluyendo silencios)")
print("=" * 70)
print(f"{'Archivo':<20} {'Acop medio':<10} ", end="")
for u in umbrales:
    print(f"umbral={u:<6} ", end="")
print()
print("-" * 70)

for archivo in archivos:
    sr, data = wav.read(archivo)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    if data.ndim == 2:
        data = data.mean(axis=1)
    
    acops, rms_vals = acoplamiento_por_ventana(data)
    
    # EXCLUIR ventanas de silencio (RMS < 0.005)
    mascara = np.array(rms_vals) > 0.005
    acops_filtrados = acops[mascara]
    
    if len(acops_filtrados) == 0:
        acop_medio = 0.0
    else:
        acop_medio = np.mean(acops_filtrados)
    
    print(f"{archivo:<20} {acop_medio:.3f}      ", end="")
    for u in umbrales:
        estado = "VIVO" if acop_medio > u else "COLAPSADO"
        print(f"{estado:<10} ", end="")
    print(f"(ventanas con señal: {len(acops_filtrados)}/{len(acops)})")