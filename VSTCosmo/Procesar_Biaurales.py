#!/usr/bin/env python3
"""
Preprocesamiento binaural — VSTCosmos v84+

Convierte todos los archivos de audio del directorio a estéreo binaural
real (ITD + ILD) y los guarda en audio_binaural/.

Reglas:
  - Archivos mono → preprocesamiento ITD+ILD a +60° y -60°
  - Archivos estéreo → copiados sin modificar (ya tienen diferencia L/R real)
  - BigBang.wav → cargado completo (sin límite de duración)
  - Todos los demás → 35 segundos

Ejecutar una sola vez antes de cualquier experimento.
El script de simulación solo carga desde audio_binaural/ — nunca convierte.

Física binaural:
  Diámetro de cabeza = 0.175 m
  Velocidad del sonido = 343 m/s
  ITD_max = 0.175/343 ≈ 0.51 ms
  Con θ=60°: ITD = 0.44 ms, ILD = 5.2 dB
"""

import numpy as np
import os
import shutil
from scipy import signal as scipy_signal

try:
    import soundfile as sf
    HAS_SF = True
except ImportError:
    print("ERROR: soundfile no está instalado.")
    print("Ejecutar: pip install soundfile")
    exit(1)

# ============================================================
# PARÁMETROS
# ============================================================
DIRECTORIO_SALIDA  = 'audio_binaural'
ANGULO_PRINCIPAL   = 60.0      # grados — derivado de arcsin(DIM_AUD/DIM_INTERNA)×2
DURACION_ESTANDAR  = 35.0      # segundos para todos los archivos excepto BigBang

DIAMETRO_CABEZA    = 0.175     # metros
VELOCIDAD_SONIDO   = 343.0     # m/s
ITD_MAX_SEG        = DIAMETRO_CABEZA / VELOCIDAD_SONIDO

# Lista completa de archivos del directorio
# Formato: (nombre_archivo, duracion_seg_o_None, descripcion)
# None = cargar completo
ARCHIVOS = [
    ('BigBang.wav',          None,             'Poesía de la Teoría musicalizada — completo'),
    ('Brandemburgo.wav',     DURACION_ESTANDAR, 'Música clásica'),
    ('Ondas mixtas.wav',     DURACION_ESTANDAR, 'Cuadrada/triangular/sierra con variaciones'),
    ('Pulso logaritmico.wav',DURACION_ESTANDAR, 'Chirp 0.5 Hz → 22050 Hz'),
    ('Ritmos aleatorios.wav',DURACION_ESTANDAR, 'Tono modulado con intervalos irregulares'),
    ('Ruido blanco.wav',     DURACION_ESTANDAR, 'Ruido uniforme normalizado'),
    ('Tono puro.wav',        DURACION_ESTANDAR, 'Sinusoide 440 Hz'),
    ('Viento.wav',           DURACION_ESTANDAR, 'Ruido de viento'),
    ('Voz+Viento_1.wav',     DURACION_ESTANDAR, 'Voz con viento — versión 1'),
    ('Voz+Viento_2.wav',     DURACION_ESTANDAR, 'Voz con viento — versión 2'),
    ('Voz_Estudio.wav',      DURACION_ESTANDAR, 'Voz limpia en estudio'),
]

# Archivos que ya son estéreo con diferencia L/R real — copiar sin modificar
ESTEREO_DIRECTO = {
    'Binaural LR mixto.wav': 'Binaural LR mixto (canal L: voz, canal R: ruido)',
}

# ============================================================
# NÚCLEO DE PREPROCESAMIENTO
# ============================================================
def convertir_mono_a_binaural(audio_mono, sr, angulo_grados):
    """
    ITD + ILD. Todos los parámetros derivan de constantes físicas.

    ITD(θ) = ITD_max × sin(θ)
    ILD(θ) = 6 × sin(θ) dB en frecuencias > f_transicion
    f_transicion = velocidad_sonido / diametro_cabeza ≈ 1960 Hz
    """
    angulo_rad   = np.radians(angulo_grados)
    ITD_seg      = ITD_MAX_SEG * np.sin(abs(angulo_rad))
    ITD_muestras = int(ITD_seg * sr)
    n            = len(audio_mono)

    if angulo_grados >= 0:
        canal_L = np.concatenate([np.zeros(ITD_muestras), audio_mono])[:n]
        canal_R = np.concatenate([audio_mono, np.zeros(ITD_muestras)])[:n]
    else:
        canal_L = np.concatenate([audio_mono, np.zeros(ITD_muestras)])[:n]
        canal_R = np.concatenate([np.zeros(ITD_muestras), audio_mono])[:n]

    canal_L = np.pad(canal_L, (0, max(0, n - len(canal_L))))[:n]
    canal_R = np.pad(canal_R, (0, max(0, n - len(canal_R))))[:n]

    # ILD: atenuar frecuencias altas en el canal sombreado
    F_TRANS   = VELOCIDAD_SONIDO / DIAMETRO_CABEZA
    ILD_dB    = 6.0 * np.sin(abs(angulo_rad))
    ILD_lin   = 10 ** (-ILD_dB / 20.0)
    nyquist   = sr / 2.0
    freq_norm = min(0.99, F_TRANS / nyquist)

    if freq_norm < 1.0 and ILD_lin < 0.99:
        b, a = scipy_signal.butter(2, freq_norm, btype='high')
        if angulo_grados >= 0:
            altas_L = scipy_signal.filtfilt(b, a, canal_L)
            canal_L = (canal_L - altas_L) + altas_L * ILD_lin
        else:
            altas_R = scipy_signal.filtfilt(b, a, canal_R)
            canal_R = (canal_R - altas_R) + altas_R * ILD_lin

    max_val = max(np.max(np.abs(canal_L)), np.max(np.abs(canal_R))) + 1e-10
    return (canal_L / max_val).astype(np.float32), \
           (canal_R / max_val).astype(np.float32)


def cargar_audio(filepath, duracion=None):
    """
    Carga audio como mono.
    Si duracion=None, carga completo.
    Si el archivo es estéreo, promedia los canales.
    Retorna (sr, audio_mono, n_canales_original, duracion_real_seg)
    """
    try:
        info = sf.info(filepath)
        sr   = info.samplerate
        n_canales = info.channels

        if duracion is None:
            data, sr = sf.read(filepath, dtype='float32')
        else:
            n_target = int(sr * duracion)
            data, sr = sf.read(filepath, frames=n_target, dtype='float32')

        if data.ndim > 1:
            audio = data.mean(axis=1)
        else:
            audio = data

        # Padding si el archivo es más corto que la duración pedida
        if duracion is not None:
            n_target = int(sr * duracion)
            if len(audio) < n_target:
                audio = np.pad(audio, (0, n_target - len(audio)))

        duracion_real = len(audio) / sr
        return sr, audio.astype(np.float32), n_canales, duracion_real

    except Exception as e:
        raise RuntimeError(f"No se pudo cargar '{filepath}': {e}")


def nombre_salida(nombre_base, angulo):
    """
    Construye el nombre del archivo de salida.
    Ejemplo: 'Voz_Estudio.wav' → 'Voz_Estudio_pos60deg.wav'
    """
    raiz = os.path.splitext(nombre_base)[0]
    signo = 'pos' if angulo >= 0 else 'neg'
    return f"{raiz}_{signo}{int(abs(angulo))}deg.wav"


# ============================================================
# PROCESAMIENTO PRINCIPAL
# ============================================================
def procesar_todos():
    os.makedirs(DIRECTORIO_SALIDA, exist_ok=True)

    ILD_dB = 6.0 * np.sin(np.radians(ANGULO_PRINCIPAL))
    ITD_ms = ITD_MAX_SEG * np.sin(np.radians(ANGULO_PRINCIPAL)) * 1000

    print("=" * 70)
    print("Preprocesamiento binaural — VSTCosmos v84+")
    print(f"  Directorio de salida: {DIRECTORIO_SALIDA}/")
    print(f"  Ángulo: ±{ANGULO_PRINCIPAL}°")
    print(f"  ITD: {ITD_ms:.2f} ms  |  ILD: {ILD_dB:.1f} dB")
    print("=" * 70)

    generados  = []
    copiados   = []
    omitidos   = []
    errores    = []

    # ---- Archivos estéreo directo (copiar sin modificar) ----
    print(f"\n[1/3] Archivos estéreo — copiar sin modificar:")
    for nombre, descripcion in ESTEREO_DIRECTO.items():
        if not os.path.exists(nombre):
            print(f"  ⚠️  '{nombre}' no encontrado — omitido")
            omitidos.append(nombre)
            continue
        destino = os.path.join(DIRECTORIO_SALIDA, nombre)
        shutil.copy2(nombre, destino)
        mb = os.path.getsize(destino) / (1024 * 1024)
        print(f"  ✅ '{nombre}' → copiado ({mb:.1f} MB)")
        print(f"     {descripcion}")
        copiados.append(nombre)

    # ---- Archivos mono → preprocesamiento binaural ----
    print(f"\n[2/3] Archivos mono → binaural ±{ANGULO_PRINCIPAL}°:")
    for nombre, duracion, descripcion in ARCHIVOS:
        if not os.path.exists(nombre):
            print(f"  ⚠️  '{nombre}' no encontrado — omitido")
            omitidos.append(nombre)
            continue

        try:
            sr, audio, n_canales, dur_real = cargar_audio(nombre, duracion)

            # Si llegó estéreo de todas formas, advertir
            if n_canales > 1:
                print(f"  ⚠️  '{nombre}' tiene {n_canales} canales — "
                      f"promediando antes de binaural")

            # Generar +angulo y -angulo
            for angulo in [+ANGULO_PRINCIPAL, -ANGULO_PRINCIPAL]:
                cL, cR    = convertir_mono_a_binaural(audio, sr, angulo)
                estereo   = np.stack([cL, cR], axis=1)
                nom_sal   = nombre_salida(nombre, angulo)
                ruta_sal  = os.path.join(DIRECTORIO_SALIDA, nom_sal)
                sf.write(ruta_sal, estereo, sr)
                mb        = os.path.getsize(ruta_sal) / (1024 * 1024)
                signo_str = f"+{angulo}°" if angulo >= 0 else f"{angulo}°"
                print(f"  ✅ '{nom_sal}' ({signo_str}, "
                      f"{dur_real:.1f}s, {mb:.1f} MB)")

            print(f"     {descripcion}")
            generados.append(nombre)

        except Exception as e:
            print(f"  ❌ '{nombre}': {e}")
            errores.append((nombre, str(e)))

    # ---- Resumen ----
    print(f"\n[3/3] Resumen:")

    # Tamaño total del directorio de salida
    total_mb = sum(
        os.path.getsize(os.path.join(DIRECTORIO_SALIDA, f)) / (1024 * 1024)
        for f in os.listdir(DIRECTORIO_SALIDA)
        if f.endswith('.wav')
    )
    n_archivos = len([f for f in os.listdir(DIRECTORIO_SALIDA)
                      if f.endswith('.wav')])

    print(f"  Procesados:   {len(generados)} archivos → "
          f"{len(generados)*2} binaurales")
    print(f"  Copiados:     {len(copiados)} archivos estéreo")
    print(f"  Omitidos:     {len(omitidos)}")
    print(f"  Errores:      {len(errores)}")
    print(f"  Total en {DIRECTORIO_SALIDA}/: "
          f"{n_archivos} archivos, {total_mb:.1f} MB")

    if errores:
        print(f"\n  Errores detallados:")
        for nombre, msg in errores:
            print(f"    {nombre}: {msg}")

    # ---- Índice de archivos generados ----
    print(f"\n{'='*70}")
    print(f"Índice de archivos en {DIRECTORIO_SALIDA}/")
    print(f"{'='*70}")

    archivos_sal = sorted([
        f for f in os.listdir(DIRECTORIO_SALIDA) if f.endswith('.wav')
    ])
    for f in archivos_sal:
        ruta = os.path.join(DIRECTORIO_SALIDA, f)
        mb   = os.path.getsize(ruta) / (1024 * 1024)
        try:
            info = sf.info(ruta)
            print(f"  {f:<45} {info.channels}ch  "
                  f"{info.duration:6.1f}s  {mb:6.1f} MB")
        except Exception:
            print(f"  {f:<45} {mb:6.1f} MB")

    print(f"{'='*70}")
    print()
    print("Clave de nombres:")
    print("  _pos60deg.wav  → fuente a la derecha (+60°)")
    print("  _neg60deg.wav  → fuente a la izquierda (-60°)")
    print("  sin sufijo     → estéreo original (copiado sin modificar)")
    print()
    print("Uso en v84 — diccionario de archivos:")
    print()
    print("  ARCHIVOS_BINAURAL = {")
    for nombre, _, _ in ARCHIVOS:
        if nombre in [n for n, _ in [(n, _) for n, _, _ in ARCHIVOS]]:
            raiz = os.path.splitext(nombre)[0]
            clave = raiz.lower().replace(' ', '_').replace('+', '_')
            print(f"    '{clave}_pos': "
                  f"f'{{DIRECTORIO_BINAURAL}}/{raiz}_pos60deg.wav',")
            print(f"    '{clave}_neg': "
                  f"f'{{DIRECTORIO_BINAURAL}}/{raiz}_neg60deg.wav',")
    for nombre in ESTEREO_DIRECTO:
        raiz  = os.path.splitext(nombre)[0]
        clave = raiz.lower().replace(' ', '_')
        print(f"    '{clave}':     "
              f"f'{{DIRECTORIO_BINAURAL}}/{nombre}',  # estéreo directo")
    print("  }")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    procesar_todos()