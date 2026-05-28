#!/usr/bin/env python3
"""
Generar versiones binaurales sintéticas de tonos puros

Para cada archivo de tono monofónico, crea:
- {nombre}_pos60deg.wav: dirección +60° (L adelantado y más fuerte)
- {nombre}_neg60deg.wav: dirección -60° (R adelantado y más fuerte)

Basado en HRTF simplificado usando ITD (diferencia temporal) e ILD (diferencia de nivel)
para una cabeza esférica de 17.5cm de diámetro.
"""

import numpy as np
import os
import soundfile as sf
from scipy import signal

# ============================================================
# PARAMETROS FISICOS
# ============================================================
DIAMETRO_CABEZA = 0.175  # metros
VELOCIDAD_SONIDO = 343.0  # m/s
RADIO_CABEZA = DIAMETRO_CABEZA / 2  # 0.0875 m

# Retardo máximo (cuando la fuente está a 90°)
ITD_MAX = (RADIO_CABEZA / VELOCIDAD_SONIDO)  # ≈ 255 microsegundos
ITD_MAX_MUESTRAS = None  # Se calculará según sample rate

# Atenuación por cabeza (ILD simplificado)
# Para 60°, atenuación ~6dB en el oído opuesto
ILD_DB_POS_60 = 6.0  # dB de atenuación para el oído lejano
ILD_DB_NEG_60 = 6.0

# Direcciones a generar
DIRECCIONES = [
    ('pos60deg', 60),   # +60°: fuente a la derecha
    ('neg60deg', -60),  # -60°: fuente a la izquierda
]

# ============================================================
# FUNCIONES
# ============================================================

def calcular_itd_samples(angulo_grados, sr):
    """
    Calcula el ITD (diferencia temporal interaural) en muestras.
    
    Formula simplificada para cabeza esférica:
    ITD = (r / c) * (sin(theta) + theta)
    
    donde:
    - r = radio de la cabeza
    - c = velocidad del sonido
    - theta = ángulo en radianes
    
    Para ángulos > 90°, se usa 90°.
    """
    theta = np.radians(min(abs(angulo_grados), 90))
    itd_segundos = (RADIO_CABEZA / VELOCIDAD_SONIDO) * (np.sin(theta) + theta)
    itd_muestras = int(round(itd_segundos * sr))
    return itd_muestras


def calcular_atenuacion_ild(angulo_grados, max_db=6.0):
    """
    Calcula la atenuación en dB para el oído lejano.
    
    Formula simplificada: lineal desde 0dB a 0° hasta max_db a 90°.
    """
    theta = min(abs(angulo_grados), 90)
    atenuacion_db = max_db * (theta / 90.0)
    # Convertir dB a factor lineal
    atenuacion_linear = 10 ** (-atenuacion_db / 20)
    return atenuacion_linear


def generar_binaural(mono, sr, angulo_grados):
    """
    Genera señal binaural a partir de mono para un ángulo dado.
    
    Args:
        mono: array 1D con la señal mono
        sr: sample rate
        angulo_grados: ángulo en grados (-90 a 90, positivo = derecha)
    
    Returns:
        (canal_L, canal_R) como arrays
    """
    itd_samples = calcular_itd_samples(angulo_grados, sr)
    
    # Calcular atenuación ILD
    # Oído cercano: sin atenuación
    # Oído lejano: atenuado según ángulo
    atenuacion_lejano = calcular_atenuacion_ild(angulo_grados, ILD_DB_POS_60)
    
    # Para ángulo positivo (derecha):
    # - L es lejano (atenuado, retrasado)
    # - R es cercano (fuerte, sin retraso)
    if angulo_grados > 0:
        # Derecha: R adelantado, L retrasado y atenuado
        canal_R = mono.copy()
        canal_L = mono * atenuacion_lejano
        # Aplicar retraso a L
        if itd_samples > 0:
            canal_L = np.pad(canal_L, (itd_samples, 0))[:-itd_samples]
    else:
        # Izquierda: L adelantado, R retrasado y atenuado
        canal_L = mono.copy()
        canal_R = mono * atenuacion_lejano
        # Aplicar retraso a R
        if itd_samples > 0:
            canal_R = np.pad(canal_R, (itd_samples, 0))[:-itd_samples]
    
    # Asegurar misma longitud
    min_len = min(len(canal_L), len(canal_R))
    canal_L = canal_L[:min_len]
    canal_R = canal_R[:min_len]
    
    return canal_L, canal_R


def procesar_archivo(input_path, output_dir, sr=None, verbose=True):
    """
    Procesa un archivo mono y genera versiones binaurales.
    """
    # Cargar archivo
    data, file_sr = sf.read(input_path)
    
    # Si es estéreo, convertir a mono (promedio)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    
    sr = sr or file_sr
    
    # Obtener nombre base
    basename = os.path.splitext(os.path.basename(input_path))[0]
    
    resultados = []
    
    for sufijo, angulo in DIRECCIONES:
        # Generar binaural
        canal_L, canal_R = generar_binaural(data, sr, angulo)
        
        # Combinar en estéreo
        stereo = np.column_stack((canal_L, canal_R))
        
        # Nombre de salida
        output_name = f"{basename}_{sufijo}.wav"
        output_path = os.path.join(output_dir, output_name)
        
        # Guardar
        sf.write(output_path, stereo, sr)
        
        resultados.append({
            'input': basename,
            'output': output_name,
            'angle': angulo,
            'itd_samples': calcular_itd_samples(angulo, sr),
            'attenuation_db': ILD_DB_POS_60 if angulo > 0 else ILD_DB_NEG_60,
            'size': len(stereo) / sr
        })
        
        if verbose:
            print(f"    Generado: {output_name}")
            print(f"      Ángulo: {angulo:+d}°, ITD: {calcular_itd_samples(angulo, sr)} muestras, atenuación: {ILD_DB_POS_60:.1f}dB")
    
    return resultados


def main():
    # Configuración
    input_dir = "audio_binaural"
    output_dir = "audio_binaural"  # Mismo directorio
    sample_rate = 48000  # Forzar 48kHz
    
    # Tonos a procesar
    tonos = [
        "Do",
        "Do_alto", 
        "Re",
        "Mi",
        "Fa",
        "Sol",
        "La",
        "Si",
        "escala_do_mayor_piano_like"
    ]
    
    print("=" * 80)
    print("Generador de tonos binaurales sintéticos")
    print("=" * 80)
    print(f"  Input dir: {input_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  ITD máximo: {ITD_MAX*1000:.3f} ms")
    print()
    
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    resultados_totales = []
    
    for tono in tonos:
        input_path = os.path.join(input_dir, f"{tono}.wav")
        
        if not os.path.exists(input_path):
            print(f"[X] {tono}: archivo no encontrado")
            continue
        
        print(f"\n[Procesando] {tono}")
        resultados = procesar_archivo(input_path, output_dir, sample_rate)
        resultados_totales.extend(resultados)
    
    print("\n" + "=" * 80)
    print("RESUMEN")
    print("=" * 80)
    print(f"  Archivos generados: {len(resultados_totales)}")
    
    # Listar archivos creados
    print("\n  Archivos creados:")
    for r in resultados_totales:
        print(f"    {r['output']:35s} ({r['size']:.1f}s, {r['angle']:+d}°)")
    
    print("\n  Completado!")
    print("  Ahora ejecuta V105b.py para probar los tonos binaurales.")


if __name__ == "__main__":
    main()