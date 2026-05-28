#!/usr/bin/env python3
"""
Generador binaural de "Blue Monday" - New Order

Convierte un archivo estéreo normal en una versión binaural
con dirección simulada (+60° para un canal, -60° para el otro)
para crear una experiencia de "campo estéreo expandido".

Entrada: Blue_Monday.wav (estéreo normal)
Salida: Blue_Monday_binaural.wav (binaural con dirección sintética)
"""

import numpy as np
import os
import soundfile as sf
from scipy import signal

# ============================================================
# PARAMETROS BINAURALES
# ============================================================
DIAMETRO_CABEZA = 0.175  # metros
VELOCIDAD_SONIDO = 343.0  # m/s
RADIO_CABEZA = DIAMETRO_CABEZA / 2  # 0.0875 m

# Direcciones a aplicar
ANGULO_IZQUIERDO = -60  # grados (fuente a la izquierda)
ANGULO_DERECHO = 60     # grados (fuente a la derecha)

# Atenuación ILD (dB a 90°)
ILD_MAX_DB = 6.0

def calcular_itd_samples(angulo_grados, sr):
    """Calcula ITD (diferencia temporal interaural) en muestras"""
    theta = np.radians(min(abs(angulo_grados), 90))
    itd_segundos = (RADIO_CABEZA / VELOCIDAD_SONIDO) * (np.sin(theta) + theta)
    itd_muestras = int(round(itd_segundos * sr))
    return itd_muestras

def calcular_atenuacion_ild(angulo_grados):
    """Calcula atenuación lineal para ILD"""
    theta = min(abs(angulo_grados), 90)
    atenuacion_db = ILD_MAX_DB * (theta / 90.0)
    return 10 ** (-atenuacion_db / 20)

def aplicar_filtro_hrtf_simplificado(canal, sr, angulo_grados, es_oido_cercano):
    """
    Aplica filtro HRTF simplificado a un canal.
    
    Args:
        canal: señal mono
        sr: sample rate
        angulo_grados: ángulo de la fuente (-90 a 90)
        es_oido_cercano: True si este es el oído cercano, False si es el lejano
    """
    resultado = canal.copy()
    
    if not es_oido_cercano:
        # Oído lejano: atenuar
        atenuacion = calcular_atenuacion_ild(angulo_grados)
        resultado = resultado * atenuacion
    
    # Aplicar ITD solo al oído lejano (el sonido tarda más en llegar)
    if not es_oido_cercano:
        itd_samples = calcular_itd_samples(angulo_grados, sr)
        if itd_samples > 0:
            resultado = np.pad(resultado, (itd_samples, 0))[:-itd_samples]
    
    return resultado

def generar_binaural_desde_estéreo(estéreo, sr, angulo_izq=-60, angulo_der=60):
    """
    Convierte audio estéreo normal a binaural.
    
    Estrategia:
    - Canal izquierdo original → se mueve a posición -60° (izquierda)
    - Canal derecho original → se mueve a posición +60° (derecha)
    
    Esto expande el campo estéreo original a una experiencia más envolvente.
    """
    # Separar canales
    if estéreo.ndim == 1:
        # Mono: duplicar
        canal_izq_original = estéreo
        canal_der_original = estéreo
    else:
        canal_izq_original = estéreo[:, 0]
        canal_der_original = estéreo[:, 1] if estéreo.shape[1] > 1 else estéreo[:, 0]
    
    # Procesar cada canal con su dirección
    # Canal izquierdo original → fuente a -60° (oído izquierdo cercano, derecho lejano)
    canal_izq_binaural = aplicar_filtro_hrtf_simplificado(
        canal_izq_original, sr, angulo_izq, es_oido_cercano=True
    )
    canal_der_desde_izq = aplicar_filtro_hrtf_simplificado(
        canal_izq_original, sr, angulo_izq, es_oido_cercano=False
    )
    
    # Canal derecho original → fuente a +60° (oído derecho cercano, izquierdo lejano)
    canal_der_binaural = aplicar_filtro_hrtf_simplificado(
        canal_der_original, sr, angulo_der, es_oido_cercano=True
    )
    canal_izq_desde_der = aplicar_filtro_hrtf_simplificado(
        canal_der_original, sr, angulo_der, es_oido_cercano=False
    )
    
    # Mezclar: oído izquierdo recibe de ambas fuentes
    oido_izquierdo = canal_izq_binaural + canal_izq_desde_der
    oido_derecho = canal_der_binaural + canal_der_desde_izq
    
    # Normalizar para evitar clipping
    max_val = max(np.max(np.abs(oido_izquierdo)), np.max(np.abs(oido_derecho)))
    if max_val > 0:
        oido_izquierdo = oido_izquierdo / max_val * 0.95
        oido_derecho = oido_derecho / max_val * 0.95
    
    return np.column_stack((oido_izquierdo, oido_derecho))

def generar_binaural_desde_mono(mono, sr, angulo=60):
    """
    Convierte audio mono a binaural con una dirección específica.
    
    Args:
        mono: señal mono
        sr: sample rate
        angulo: ángulo de la fuente (-90 a 90, positivo = derecha)
    """
    canal_cercano = aplicar_filtro_hrtf_simplificado(mono, sr, angulo, es_oido_cercano=True)
    canal_lejano = aplicar_filtro_hrtf_simplificado(mono, sr, angulo, es_oido_cercano=False)
    
    if angulo > 0:
        # Fuente a la derecha: oído derecho cercano, izquierdo lejano
        return np.column_stack((canal_lejano, canal_cercano))
    else:
        # Fuente a la izquierda: oído izquierdo cercano, derecho lejano
        return np.column_stack((canal_cercano, canal_lejano))

def procesar_blue_monday(input_file, output_file, sr=None):
    """
    Procesa Blue Monday a versión binaural.
    """
    print("=" * 80)
    print("Generador binaural de 'Blue Monday' - New Order")
    print("=" * 80)
    
    # Cargar archivo
    print(f"\n[Carga] {input_file}")
    data, file_sr = sf.read(input_file)
    
    # Usar sample rate especificado o el del archivo
    if sr is None:
        sr = file_sr
    
    # Si es necesario, resamplear
    if sr != file_sr:
        print(f"  Resampleando de {file_sr} a {sr} Hz...")
        # Número de muestras después del resampleo
        num_muestras = int(len(data) * sr / file_sr)
        data = signal.resample(data, num_muestras)
    
    print(f"  Canales: {data.ndim}, Duración: {len(data)/sr:.1f}s")
    
    # Generar versión binaural
    print("\n[Procesamiento] Generando audio binaural...")
    print(f"  Estrategia: expandir campo estéreo original")
    print(f"  Ángulos: izquierdo = {ANGULO_IZQUIERDO}°, derecho = {ANGULO_DERECHO}°")
    
    binaural = generar_binaural_desde_estéreo(data, sr, ANGULO_IZQUIERDO, ANGULO_DERECHO)
    
    # También generar versiones mono-direccionales para experimentos
    print("\n[Generando variantes]")
    
    # Versión izquierda (fuente a -60°)
    print("  Generando Blue_Monday_left_binaural.wav...")
    if data.ndim == 1:
        mono = data
    else:
        mono = np.mean(data, axis=1)
    left_binaural = generar_binaural_desde_mono(mono, sr, ANGULO_IZQUIERDO)
    
    # Versión derecha (fuente a +60°)
    print("  Generando Blue_Monday_right_binaural.wav...")
    right_binaural = generar_binaural_desde_mono(mono, sr, ANGULO_DERECHO)
    
    # Guardar archivos
    print("\n[Guardado]")
    sf.write(output_file, binaural, sr)
    print(f"  {output_file}: {len(binaural)/sr:.1f}s, estéreo binaural expandido")
    
    left_file = output_file.replace('.wav', '_left_binaural.wav')
    sf.write(left_file, left_binaural, sr)
    print(f"  {left_file}: fuente a {ANGULO_IZQUIERDO}°")
    
    right_file = output_file.replace('.wav', '_right_binaural.wav')
    sf.write(right_file, right_binaural, sr)
    print(f"  {right_file}: fuente a {ANGULO_DERECHO}°")
    
    print("\n✅ COMPLETADO!")
    print(f"  Archivos generados en el directorio: {os.path.dirname(output_file)}")
    
    return {
        'binaural_estéreo': output_file,
        'binaural_left': left_file,
        'binaural_right': right_file,
        'duración': len(data)/sr,
        'sr': sr
    }

def main():
    # Configuración
    input_file = "audio_binaural/Blue_Monday.wav"
    output_file = "audio_binaural/Blue_Monday_binaural_expandido.wav"
    
    # Verificar que el archivo existe
    if not os.path.exists(input_file):
        print(f"ERROR: No se encuentra {input_file}")
        print("\nPor favor, asegúrate de que:")
        print("  1. El archivo 'Blue_Monday.wav' está en audio_binaural/")
        print("  2. El nombre es exactamente 'Blue_Monday.wav'")
        print("\nO modifica el script con la ruta correcta.")
        return
    
    # Procesar
    resultado = procesar_blue_monday(input_file, output_file, sr=48000)
    
    print("\n" + "=" * 80)
    print("RESUMEN PARA V108")
    print("=" * 80)
    print(f"""
    Archivos generados:
      1. {resultado['binaural_estéreo']} - Campo estéreo expandido
      2. {resultado['binaural_left']} - Fuente única a {ANGULO_IZQUIERDO}°
      3. {resultado['binaural_right']} - Fuente única a {ANGULO_DERECHO}°
    
    Duración: {resultado['duración']:.1f}s (7:0{int(resultado['duración']%60)})
    Sample rate: {resultado['sr']} Hz
    
    Para V108, usar:
      - Estímulo largo: Blue_Monday_binaural_expandido.wav
      - Controles: Blue_Monday_left_binaural.wav y Blue_Monday_right_binaural.wav
    """)


if __name__ == "__main__":
    main()