#!/usr/bin/env python3
"""
Generador de estímulos para VSTCosmos v84+

Genera 6 archivos WAV listos para usar como entrada al experimento:
  1. Tono puro.wav              — sinusoide 440 Hz
  2. Ruido blanco.wav           — ruido uniforme normalizado
  3. Ritmos aleatorios.wav      — tono 440 Hz modulado en amplitud con intervalos irregulares
  4. Ondas mixtas.wav           — cuadrada, triangular y diente de sierra con variaciones
  5. Binaural LR mixto.wav      — canal L: voz, canal R: ruido blanco
  6. Pulso logaritmico.wav      — pulsos desde infrasonido hasta ultrasonido

Todos los archivos tienen la misma duración y sample rate para ser
intercambiables en el protocolo del experimento.

Ejecutar una sola vez antes de correr v84.
"""

import numpy as np
import soundfile as sf
import os

# ============================================================
# PARÁMETROS GLOBALES
# ============================================================
SR       = 48000    # Hz — mismo que los archivos del experimento
DURACION = 35.0     # segundos — mismo que generar_binaurales_preprocesados
N        = int(SR * DURACION)
t        = np.arange(N) / SR

# ============================================================
# UTILIDADES
# ============================================================
def normalizar(señal, headroom=0.02):
    """Normaliza a [-1+headroom, 1-headroom] sin clip."""
    max_val = np.max(np.abs(señal)) + 1e-10
    return (señal / max_val * (1.0 - headroom)).astype(np.float32)

def guardar(nombre, señal_mono_o_estereo):
    sf.write(nombre, señal_mono_o_estereo, SR)
    if señal_mono_o_estereo.ndim == 1:
        print(f"  ✅ '{nombre}' — mono, {DURACION}s, sr={SR} Hz")
    else:
        print(f"  ✅ '{nombre}' — estéreo {señal_mono_o_estereo.shape[1]}ch, "
              f"{DURACION}s, sr={SR} Hz")

# ============================================================
# 1. TONO PURO
# ============================================================
def generar_tono_puro():
    """
    Sinusoide de 440 Hz (La4 — referencia estándar).
    """
    FREQ = 440.0
    señal = normalizar(np.sin(2 * np.pi * FREQ * t))
    guardar('Tono puro.wav', señal)

# ============================================================
# 2. RUIDO BLANCO
# ============================================================
def generar_ruido_blanco():
    """
    Ruido uniforme en [-1, 1]. Semilla None — no reproducible,
    como el entorno real.
    """
    np.random.seed(None)
    señal = normalizar(np.random.uniform(-1.0, 1.0, N).astype(np.float32))
    guardar('Ruido blanco.wav', señal)

# ============================================================
# 3. RITMOS ALEATORIOS
# ============================================================
def generar_ritmos_aleatorios():
    """
    Tono de 440 Hz modulado en amplitud con intervalos de encendido/apagado
    de duración aleatoria.

    Los intervalos son irregulares (no rítmicos en sentido musical)
    pero la estructura temporal es más rica que el ruido blanco.
    Cada ráfaga tiene duración entre 50 ms y 500 ms.
    Cada silencio tiene duración entre 20 ms y 300 ms.
    La modulación tiene un ataque y decaimiento suave (envolvente
    trapezoidal) para evitar clicks.

    La frecuencia de portadora varía aleatoriamente entre ráfagas
    en el rango 200–2000 Hz — estructura espectral cambiante.
    """
    np.random.seed(None)
    señal = np.zeros(N, dtype=np.float32)
    pos   = 0

    FADE_MS    = 10   # ms de ataque/decaimiento
    FADE_N     = int(FADE_MS * SR / 1000)

    while pos < N:
        # Duración de la ráfaga (muestras)
        dur_on  = int(np.random.uniform(0.05, 0.50) * SR)
        # Frecuencia de esta ráfaga
        freq    = np.random.uniform(200.0, 2000.0)
        # Intensidad de esta ráfaga (0.3 a 1.0)
        amp     = np.random.uniform(0.3, 1.0)

        fin_on  = min(pos + dur_on, N)
        n_on    = fin_on - pos
        t_local = np.arange(n_on) / SR

        # Envolvente trapezoidal
        env = np.ones(n_on)
        fade_actual = min(FADE_N, n_on // 4)
        if fade_actual > 0:
            ramp = np.linspace(0, 1, fade_actual)
            env[:fade_actual]  = ramp
            env[-fade_actual:] = ramp[::-1]

        señal[pos:fin_on] = amp * env * np.sin(2 * np.pi * freq * t_local)
        pos = fin_on

        if pos >= N:
            break

        # Silencio entre ráfagas
        dur_off = int(np.random.uniform(0.02, 0.30) * SR)
        pos    += dur_off

    guardar('Ritmos aleatorios.wav', normalizar(señal))

# ============================================================
# 4. ONDAS MIXTAS
# ============================================================
def generar_ondas_mixtas():
    """
    Un solo archivo que recorre cuadrada, triangular y diente de sierra
    con variaciones aleatorias de orden, intensidad y frecuencia.

    El archivo se divide en segmentos de duración aleatoria (1–4 s).
    Cada segmento elige aleatoriamente:
      - Forma de onda: cuadrada, triangular, diente de sierra ascendente,
        diente de sierra descendente, sinusoide (como referencia)
      - Frecuencia: 50–3000 Hz
      - Intensidad: 0.3–1.0
      - Duty cycle (solo para cuadrada): 0.3–0.7

    Los armónicos de cada forma distinguen espectralmente los segmentos
    — estructura que W_prof puede aprender a reconocer.
    """
    np.random.seed(None)
    señal = np.zeros(N, dtype=np.float32)
    pos   = 0

    FORMAS    = ['cuadrada', 'triangular', 'sierra_asc', 'sierra_desc', 'seno']
    FADE_MS   = 15
    FADE_N    = int(FADE_MS * SR / 1000)

    while pos < N:
        dur     = int(np.random.uniform(1.0, 4.0) * SR)
        fin     = min(pos + dur, N)
        n_seg   = fin - pos
        t_local = np.arange(n_seg) / SR

        freq    = np.random.uniform(50.0, 3000.0)
        amp     = np.random.uniform(0.3, 1.0)
        forma   = np.random.choice(FORMAS)
        fase    = t_local * freq   # ciclos transcurridos

        if forma == 'cuadrada':
            duty   = np.random.uniform(0.3, 0.7)
            frac   = fase % 1.0
            seg    = np.where(frac < duty, 1.0, -1.0).astype(np.float32)
        elif forma == 'triangular':
            frac   = fase % 1.0
            seg    = (2.0 * np.abs(2.0 * frac - 1.0) - 1.0).astype(np.float32)
        elif forma == 'sierra_asc':
            frac   = fase % 1.0
            seg    = (2.0 * frac - 1.0).astype(np.float32)
        elif forma == 'sierra_desc':
            frac   = fase % 1.0
            seg    = (1.0 - 2.0 * frac).astype(np.float32)
        else:   # seno
            seg    = np.sin(2 * np.pi * fase).astype(np.float32)

        # Envolvente trapezoidal
        env = np.ones(n_seg)
        fade_actual = min(FADE_N, n_seg // 4)
        if fade_actual > 0:
            ramp = np.linspace(0, 1, fade_actual)
            env[:fade_actual]  = ramp
            env[-fade_actual:] = ramp[::-1]

        señal[pos:fin] = amp * env * seg
        pos = fin

    guardar('Ondas mixtas.wav', normalizar(señal))

# ============================================================
# 5. BINAURAL LR MIXTO
# ============================================================
def generar_binaural_lr_mixto():
    """
    Canal L: voz (Voz_Estudio.wav)
    Canal R: ruido blanco (Ruido blanco.wav recién generado)

    Es el primer estímulo con información espacial real sin
    preprocesamiento ITD/ILD — la diferencia entre canales
    viene del contenido, no de la geometría de la cabeza.

    Si alguno de los archivos fuente no existe, usa fallback sintético.
    El canal más corto se repite (loop) hasta completar la duración.
    """
    np.random.seed(None)

    # Canal L: voz
    try:
        data_voz, sr_voz = sf.read('Voz_Estudio.wav', dtype='float32')
        if data_voz.ndim > 1:
            data_voz = data_voz.mean(axis=1)
        # Resamplear si sr distinto (simple: solo aceptamos mismo sr)
        if sr_voz != SR:
            print(f"  [ADVERTENCIA] Voz_Estudio.wav tiene sr={sr_voz}, "
                  f"esperado {SR}. Usando tono 220 Hz como fallback L.")
            data_voz = (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    except Exception:
        print("  [ADVERTENCIA] No se encontró Voz_Estudio.wav. "
              "Usando tono 220 Hz como canal L.")
        data_voz = (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)

    # Canal R: ruido blanco
    try:
        data_ruido, sr_ruido = sf.read('Ruido blanco.wav', dtype='float32')
        if data_ruido.ndim > 1:
            data_ruido = data_ruido.mean(axis=1)
        if sr_ruido != SR:
            data_ruido = np.random.uniform(-1.0, 1.0, N).astype(np.float32)
    except Exception:
        print("  [ADVERTENCIA] No se encontró 'Ruido blanco.wav'. "
              "Generando ruido en canal R.")
        data_ruido = np.random.uniform(-1.0, 1.0, N).astype(np.float32)

    # Loop para completar N muestras
    def loop_hasta(arr, n):
        if len(arr) >= n:
            return arr[:n]
        repeticiones = (n // len(arr)) + 1
        return np.tile(arr, repeticiones)[:n]

    canal_L = loop_hasta(data_voz,   N)
    canal_R = loop_hasta(data_ruido, N)

    canal_L = normalizar(canal_L)
    canal_R = normalizar(canal_R)

    estereo = np.stack([canal_L, canal_R], axis=1)
    guardar('Binaural LR mixto.wav', estereo)

# ============================================================
# 6. PULSO LOGARÍTMICO
# ============================================================
def generar_pulso_logaritmico():
    """
    Secuencia de pulsos cuya frecuencia de repetición crece
    logarítmicamente en el tiempo.

    Cubre todo el espectro audible con excesos no audibles:
      - Inicio: 0.5 Hz  (infrasonido — por debajo de los 20 Hz audibles)
      - Final:  22050 Hz (ultrasónico — por encima de los 20 kHz audibles)

    La frecuencia instantánea en el tiempo t es:
        f(t) = f_inicio × (f_final / f_inicio)^(t / T)

    Eso es un chirp logarítmico — cada octava ocupa la misma
    duración en escala logarítmica.

    Cada pulso es una sinusoide breve de 2 ciclos a la frecuencia
    instantánea, con envolvente gaussiana para evitar artefactos.

    Este estímulo es el "test de audición" del sistema:
    si el campo responde de forma diferencial a distintas
    frecuencias, se manifestará como variación de GED a lo largo
    del archivo.
    """
    F_INICIO  = 0.5       # Hz — infrasonido
    F_FINAL   = 22050.0   # Hz — Nyquist = límite físico del archivo

    # Frecuencia instantánea por muestra (chirp logarítmico)
    # f(t) = f_inicio × r^t donde r = (f_final/f_inicio)^(1/T)
    ratio     = F_FINAL / F_INICIO
    f_inst    = F_INICIO * (ratio ** (t / DURACION))   # Hz en cada muestra

    # Fase acumulada (integral de f(t))
    # ∫ f(t) dt = f_inicio × T / ln(ratio) × (r^t - 1)
    ln_ratio  = np.log(ratio)
    fase_acum = 2 * np.pi * F_INICIO * (DURACION / ln_ratio) * \
                (ratio ** (t / DURACION) - 1.0)

    señal = np.sin(fase_acum).astype(np.float32)

    # Envolvente: fade in y fade out de 500 ms para evitar clicks
    FADE_N = int(0.5 * SR)
    ramp   = np.linspace(0, 1, FADE_N)
    señal[:FADE_N]  *= ramp
    señal[-FADE_N:] *= ramp[::-1]

    guardar('Pulso logaritmico.wav', normalizar(señal))

    # Información de diagnóstico
    t_20hz   = DURACION * np.log(20.0    / F_INICIO) / np.log(ratio)
    t_20khz  = DURACION * np.log(20000.0 / F_INICIO) / np.log(ratio)
    print(f"       Rango audible (20 Hz – 20 kHz):")
    print(f"       20 Hz alcanzado en t={t_20hz:.2f}s")
    print(f"       20 kHz alcanzado en t={t_20khz:.2f}s")
    print(f"       Infrasonido: 0.0s – {t_20hz:.2f}s")
    print(f"       Audible:     {t_20hz:.2f}s – {t_20khz:.2f}s")
    print(f"       Ultrasonido: {t_20khz:.2f}s – {DURACION:.1f}s")

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("Generador de estímulos — VSTCosmos v84+")
    print(f"  sr={SR} Hz, duración={DURACION}s, muestras={N:,}")
    print("=" * 70)
    print()

    print("Generando archivos base:")
    generar_tono_puro()
    generar_ruido_blanco()
    print()

    print("Generando estímulos adicionales:")
    generar_ritmos_aleatorios()
    generar_ondas_mixtas()
    generar_binaural_lr_mixto()
    print()
    generar_pulso_logaritmico()
    print()

    print("=" * 70)
    print("Archivos generados:")
    archivos = [
        'Tono puro.wav',
        'Ruido blanco.wav',
        'Ritmos aleatorios.wav',
        'Ondas mixtas.wav',
        'Binaural LR mixto.wav',
        'Pulso logaritmico.wav',
    ]
    total_mb = 0
    for nombre in archivos:
        if os.path.exists(nombre):
            mb = os.path.getsize(nombre) / (1024 * 1024)
            total_mb += mb
            print(f"  {nombre:<30} {mb:.1f} MB")
        else:
            print(f"  {nombre:<30} ❌ no generado")
    print(f"  {'TOTAL':<30} {total_mb:.1f} MB")
    print("=" * 70)
    print()
    print("Listos para usar en el protocolo de v84.")
    print("Agregar al diccionario 'estimulos' en generar_binaurales_preprocesados:")
    print()
    print("  'ritmos':   'Ritmos aleatorios.wav',")
    print("  'ondas':    'Ondas mixtas.wav',")
    print("  'lr_mixto': 'Binaural LR mixto.wav',   # ya es estéreo — no preprocesar")
    print("  'pulso':    'Pulso logaritmico.wav',")
    print()
    print("Nota: 'Binaural LR mixto.wav' ya tiene canales L/R diferenciados.")
    print("No aplicar preprocesamiento binaural adicional — la diferencia")
    print("espacial viene del contenido, no del ITD/ILD.")


if __name__ == "__main__":
    main()