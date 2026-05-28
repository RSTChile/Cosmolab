#!/usr/bin/env python3
"""
Generador de tonos largos binaurales (30s) para todas las notas

Para cada nota (Do, Do_alto, Re, Mi, Fa, Sol, La, Si):
- Genera tono puro a 48000 Hz
- Aplica ITD + ILD para direcciones +60° y -60°
- Guarda como {nota}_pos60deg_largo.wav y {nota}_neg60deg_largo.wav
- Duración: 30 segundos
"""

import numpy as np
import os
import soundfile as sf

# ============================================================
# PARAMETROS
# ============================================================
SR = 48000  # Sample rate
DURACION = 30.0  # segundos
N_MUESTRAS = int(SR * DURACION)

# Frecuencias de las notas (Hz)
FRECUENCIAS = {
    'Do': 261.63,      # C4
    'Do_alto': 523.25, # C5 (una octava arriba)
    'Re': 293.66,      # D4
    'Mi': 329.63,      # E4
    'Fa': 349.23,      # F4
    'Sol': 392.00,     # G4
    'La': 440.00,      # A4
    'Si': 493.88,      # B4
}

# Parámetros binaurales
DIAMETRO_CABEZA = 0.175  # metros
VELOCIDAD_SONIDO = 343.0  # m/s
RADIO_CABEZA = DIAMETRO_CABEZA / 2

# Atenuación para ILD (dB a 60°)
ILD_DB_60 = 6.0

def calcular_itd_samples(angulo_grados, sr):
    """Calcula ITD en muestras para un ángulo dado"""
    theta = np.radians(min(abs(angulo_grados), 90))
    itd_segundos = (RADIO_CABEZA / VELOCIDAD_SONIDO) * (np.sin(theta) + theta)
    return int(round(itd_segundos * sr))

def calcular_atenuacion_ild(angulo_grados, max_db=6.0):
    """Calcula atenuación lineal para ILD"""
    theta = min(abs(angulo_grados), 90)
    atenuacion_db = max_db * (theta / 90.0)
    return 10 ** (-atenuacion_db / 20)

def generar_tono(frecuencia, sr, duracion):
    """Genera tono puro mono"""
    t = np.linspace(0, duracion, int(sr * duracion), endpoint=False)
    return np.sin(2 * np.pi * frecuencia * t)

def generar_binaural(mono, sr, angulo_grados):
    """
    Genera señal binaural a partir de mono para un ángulo dado.
    angulo_grados: positivo = derecha (+60°), negativo = izquierda (-60°)
    """
    itd_samples = calcular_itd_samples(angulo_grados, sr)
    atenuacion_lejano = calcular_atenuacion_ild(angulo_grados, ILD_DB_60)
    
    if angulo_grados > 0:
        # Derecha: R cercano (fuerte), L lejano (débil y retrasado)
        canal_R = mono.copy()
        canal_L = mono * atenuacion_lejano
        if itd_samples > 0:
            canal_L = np.pad(canal_L, (itd_samples, 0))[:-itd_samples]
    else:
        # Izquierda: L cercano (fuerte), R lejano (débil y retrasado)
        canal_L = mono.copy()
        canal_R = mono * atenuacion_lejano
        if itd_samples > 0:
            canal_R = np.pad(canal_R, (itd_samples, 0))[:-itd_samples]
    
    # Asegurar misma longitud
    min_len = min(len(canal_L), len(canal_R))
    return canal_L[:min_len], canal_R[:min_len]

def main():
    output_dir = "audio_binaural"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Generador de tonos largos binaurales (30s)")
    print("=" * 80)
    print(f"  Sample rate: {SR} Hz")
    print(f"  Duración: {DURACION} s")
    print(f"  ITD máximo: {calcular_itd_samples(60, SR) * 1000 / SR:.3f} ms")
    print(f"  ILD a 60°: {ILD_DB_60} dB")
    print()
    
    resultados = []
    
    for nota, frecuencia in FRECUENCIAS.items():
        print(f"\n[Generando] {nota} ({frecuencia:.2f} Hz)")
        
        # Generar tono mono
        tono_mono = generar_tono(frecuencia, SR, DURACION)
        
        # Normalizar
        tono_mono = tono_mono / np.max(np.abs(tono_mono))
        
        # Generar versiones binaurales
        for sufijo, angulo in [('pos60deg_largo', 60), ('neg60deg_largo', -60)]:
            canal_L, canal_R = generar_binaural(tono_mono, SR, angulo)
            stereo = np.column_stack((canal_L, canal_R))
            
            output_name = f"{nota}_{sufijo}.wav"
            output_path = os.path.join(output_dir, output_name)
            
            sf.write(output_path, stereo, SR)
            
            itd = calcular_itd_samples(angulo, SR)
            atenuacion = calcular_atenuacion_ild(angulo, ILD_DB_60)
            
            print(f"    {output_name}: {angulo:+d}°, ITD={itd} muestras, atenuación={atenuacion:.3f}")
            
            resultados.append({
                'nota': nota,
                'frecuencia': frecuencia,
                'archivo': output_name,
                'angulo': angulo,
                'duracion': DURACION
            })
    
    print("\n" + "=" * 80)
    print("RESUMEN")
    print("=" * 80)
    print(f"  Archivos generados: {len(resultados)}")
    print(f"  Notas procesadas: {len(FRECUENCIAS)}")
    print()
    print("  Archivos creados:")
    for r in resultados:
        print(f"    {r['archivo']:40s} ({r['frecuencia']:.2f}Hz, {r['angulo']:+d}°)")
    
    print("\n  ✅ COMPLETADO!")
    print("  Ahora ejecuta V105d.py para probar los tonos largos (30s)")

if __name__ == "__main__":
    main()