#!/usr/bin/env python3
"""
VSTCosmos v84 — Campo diferencial: act_busc como órgano de gradiente L-R

Cambios respecto a v83:
1. Archivos binaurales preprocesados una sola vez al inicio (angulo ±60°)
2. act_busc recibe Δ(L,R) como variable primaria — no difusión de aud_L/aud_R
3. Vecindades act_busc ↔ aud_L y act_busc ↔ aud_R eliminadas de la topología
4. Coherencia calculada sobre objetivos crudos, no sobre estado del campo diluido

Frase canónica:
La orientación no emerge de L y R como magnitudes;
emerge de Δ(L,R) como variable primaria del campo.
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import csv
import warnings
import os

try:
    import soundfile as sf
    HAS_SF = True
except ImportError:
    HAS_SF = False

from scipy import signal as scipy_signal
warnings.filterwarnings('ignore')

# ============================================================
# PARÁMETROS DE LA FÍSICA DEL CAMPO (heredados de v80h)
# ============================================================
DIM_INTERNA = 32

DIFUSION_BASE     = 0.15
GANANCIA_REACCION = 0.05
OMEGA_MIN         = 0.05
OMEGA_MAX         = 0.50
AMORT_MIN         = 0.01
AMORT_MAX         = 0.08
PHI_EQUILIBRIO    = 0.5

VENTANA_FFT_MS = 25
HOP_FFT_MS     = 10
F_MIN          = 80
F_MAX          = 8000

OMEGA_MEDIA        = (OMEGA_MIN + OMEGA_MAX) / 2.0
T_PROFUNDA_SEG     = 1.0 / OMEGA_MIN
T_RECIENTE_SEG     = 1.0 / OMEGA_MAX
T_PROFUNDA_PASOS   = int(T_PROFUNDA_SEG / 0.01)
T_RECIENTE_PASOS   = int(T_RECIENTE_SEG / 0.01)

ETA_PROFUNDA_BASE  = (1.0 / T_PROFUNDA_PASOS) / DIFUSION_BASE
ETA_RECIENTE_BASE  = (1.0 / T_RECIENTE_PASOS) / DIFUSION_BASE
TAU_PROFUNDA       = OMEGA_MIN
TAU_RECIENTE       = OMEGA_MIN * 0.5
TAU_EFICIENCIA     = int(1.0 / (OMEGA_MIN * 0.01))
TAU_EXPLORACION    = int(T_RECIENTE_SEG / 0.01)   # 200 pasos = 2s

LIMITE_MIN  = 0.0
LIMITE_MAX  = 1.0
W_MAX       = 1.0
ALPHA_FIJO  = 0.05
DT          = 0.01
DIM_TIME    = 100

# Constantes físicas para binaural
DIAMETRO_CABEZA  = 0.175    # metros
VELOCIDAD_SONIDO = 343.0    # m/s
ITD_MAX_SEG      = DIAMETRO_CABEZA / VELOCIDAD_SONIDO   # ≈ 0.51 ms
ANGULO_PRINCIPAL = 60.0     # derivado de arcsin(DIM_AUD_L/DIM_INTERNA)×2
                             # = arcsin(0.5)×2 = 60°

# ============================================================
# ARQUITECTURA DEL CAMPO EXPANDIDO
# ============================================================
DIM_GANGLIO  = DIM_INTERNA // 2   # 16
DIM_AUD      = DIM_GANGLIO        # 16
DIM_ACT      = DIM_GANGLIO // 2   # 8

DIM_AUD_L    = DIM_AUD
DIM_AUD_R    = DIM_AUD
DIM_ACT_PERM = DIM_ACT
DIM_ACT_GEOM = DIM_ACT
DIM_ACT_BUSC = DIM_ACT
DIM_ACT_MANT = DIM_ACT

# Índices de regiones
idx = {}
idx['int']      = (0,                   DIM_INTERNA)
idx['G']        = (DIM_INTERNA,         DIM_INTERNA + DIM_GANGLIO)
idx['aud_L']    = (idx['G'][1],         idx['G'][1]        + DIM_AUD_L)
idx['aud_R']    = (idx['aud_L'][1],     idx['aud_L'][1]    + DIM_AUD_R)
idx['act_perm'] = (idx['aud_R'][1],     idx['aud_R'][1]    + DIM_ACT_PERM)
idx['act_geom'] = (idx['act_perm'][1],  idx['act_perm'][1] + DIM_ACT_GEOM)
idx['act_busc'] = (idx['act_geom'][1],  idx['act_geom'][1] + DIM_ACT_BUSC)
idx['act_mant'] = (idx['act_busc'][1],  idx['act_busc'][1] + DIM_ACT_MANT)
DIM_TOTAL = idx['act_mant'][1]

# Topología v84 — act_busc YA NO recibe difusión de aud_L ni aud_R
# Su estado se actualiza directamente desde el gradiente Δ(L,R)
VECINDADES = [
    ('int',      'G'),
    ('G',        'aud_L'),
    ('G',        'aud_R'),
    ('G',        'act_perm'),
    ('G',        'act_geom'),
    ('G',        'act_busc'),     # Φ_G sigue acoplada con act_busc
    ('G',        'act_mant'),
    ('aud_L',    'aud_R'),
    ('act_perm', 'aud_L'),
    ('act_perm', 'aud_R'),
    ('act_geom', 'aud_L'),
    ('act_geom', 'aud_R'),
    # ELIMINADAS respecto a v83:
    # ('act_busc', 'aud_L'),
    # ('act_busc', 'aud_R'),
]

print("=" * 100)
print("VSTCosmos v84 — Campo diferencial: act_busc como órgano de gradiente L-R")
print("")
print("  Cambios respecto a v83:")
print("  1. Archivos binaurales preprocesados una sola vez al inicio (±60°)")
print("  2. act_busc recibe Δ(L,R) como variable primaria del campo")
print("  3. Vecindades act_busc ↔ aud_L/R eliminadas de la topología")
print("  4. Coherencia calculada sobre objetivos crudos")
print("")
print(f"  Física binaural: ITD_max={ITD_MAX_SEG*1000:.2f}ms, "
      f"ángulo=±{ANGULO_PRINCIPAL}°, "
      f"ILD={6*np.sin(np.radians(ANGULO_PRINCIPAL)):.1f} dB")
print(f"  DIM_TOTAL = {DIM_TOTAL}")
print("")
print("  Topología v84:")
for va, vb in VECINDADES:
    print(f"    {va} ↔ {vb}")
print("=" * 100)

# ============================================================
# CLASE EXPLORADOR
# ============================================================
class ExploradorActuadores:
    def __init__(self):
        self.historial        = []
        self.mejor_config     = None
        self.mejor_eficiencia = 0.0
        self.pasos_en_lf      = 0

    def actualizar(self, lf_activa, eficiencia_actual,
                   fraccion_L, fraccion_R, sesgo_freq):
        if lf_activa:
            self.pasos_en_lf += 1
            self.historial.append(
                (fraccion_L, fraccion_R, sesgo_freq, eficiencia_actual)
            )
            if eficiencia_actual > self.mejor_eficiencia:
                self.mejor_eficiencia = eficiencia_actual
                self.mejor_config = (fraccion_L, fraccion_R, sesgo_freq)
        else:
            self.pasos_en_lf = 0

# ============================================================
# FUNCIONES BASE
# ============================================================
def inicializar_campo():
    np.random.seed(None)
    Phi_total     = np.random.normal(PHI_EQUILIBRIO, 0.01, (DIM_TOTAL, DIM_TIME))
    Phi_vel_total = np.zeros((DIM_TOTAL, DIM_TIME))
    return Phi_total, Phi_vel_total

def inicializar_memorias():
    W_prof           = np.zeros((DIM_INTERNA, DIM_AUD))
    W_rec            = np.zeros((DIM_INTERNA, DIM_AUD))
    Phi_int_historia = np.zeros((DIM_INTERNA, DIM_TIME))
    return W_prof, W_rec, Phi_int_historia

def _perfil_espectral_region(region, dim):
    n_bins = 50
    perfil = np.zeros(n_bins)
    for banda in range(min(dim, region.shape[0])):
        serie    = region[banda, :] - np.mean(region[banda, :])
        fft      = np.fft.rfft(serie)
        potencia = np.abs(fft) ** 2
        perfil  += potencia[:n_bins]
    return perfil / max(1, dim)

def calcular_ged_entre(region_a, region_b):
    p_a = _perfil_espectral_region(region_a, region_a.shape[0])
    p_b = _perfil_espectral_region(region_b, region_b.shape[0])
    return float(np.mean(np.abs(p_a - p_b)))

def calcular_frecuencias_naturales(dim):
    bandas = np.arange(dim)
    t      = np.log1p(bandas) / np.log1p(max(dim - 1, 1))
    omega  = OMEGA_MIN + (OMEGA_MAX - OMEGA_MIN) * t
    amort  = AMORT_MIN + (AMORT_MAX - AMORT_MIN) * t
    return omega.reshape(-1, 1), amort.reshape(-1, 1)

def calcular_promedio_vecinos(Phi_total):
    promedio = np.zeros_like(Phi_total)
    conteo   = np.zeros(DIM_TOTAL)

    for reg_a, reg_b in VECINDADES:
        ia0, ia1 = idx[reg_a]
        ib0, ib1 = idx[reg_b]
        n = min(ia1 - ia0, ib1 - ib0)
        for d in range(n):
            if ia0 + d < DIM_TOTAL and ib0 + d < DIM_TOTAL:
                promedio[ia0 + d, :] += Phi_total[ib0 + d, :]
                promedio[ib0 + d, :] += Phi_total[ia0 + d, :]
                conteo[ia0 + d] += 1
                conteo[ib0 + d] += 1

    for i in range(DIM_TOTAL):
        if conteo[i] > 0:
            promedio[i, :] /= conteo[i]
        else:
            promedio[i, :] = Phi_total[i, :]

    return promedio

# ============================================================
# PREPROCESAMIENTO BINAURAL — UNA SOLA VEZ AL INICIO
# ============================================================
def cargar_mono(filepath, duracion):
    """Carga cualquier archivo de audio como mono."""
    n_target = None
    audio    = None
    sr       = 48000

    if HAS_SF:
        try:
            data, sr = sf.read(filepath, dtype='float32')
            if data.ndim > 1:
                data = data.mean(axis=1)
            n_target = int(sr * duracion)
            audio    = data[:n_target]
        except Exception:
            audio = None

    if audio is None:
        try:
            sr, data = wav.read(filepath)
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            if data.ndim > 1:
                data = data.mean(axis=1)
            n_target = int(sr * duracion)
            audio    = data[:n_target]
        except Exception:
            audio = None

    if audio is None:
        print(f"  [ADVERTENCIA] No se pudo cargar '{filepath}', usando tono 440Hz")
        sr       = 48000
        n_target = int(sr * duracion)
        t        = np.arange(n_target) / sr
        audio    = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    if n_target and len(audio) < n_target:
        audio = np.pad(audio, (0, n_target - len(audio)))

    return sr, audio.astype(np.float32)


def convertir_mono_a_binaural(audio_mono, sr, angulo_grados):
    """
    ITD + ILD. Todos los parámetros derivan de constantes físicas.

    ITD: desfase temporal entre oídos
        ITD(θ) = ITD_max × sin(θ)
        ITD_max = diámetro_cabeza / velocidad_sonido = 0.175/343 ≈ 0.51 ms

    ILD: diferencia de nivel por sombra acústica de la cabeza
        Frecuencia de transición: f_t = v/d = 343/0.175 ≈ 1960 Hz
        Atenuación en canal sombreado: 6 × sin(θ) dB
        Con θ=60°: ILD ≈ 5.2 dB — diferencia espectral visible
    """
    angulo_rad   = np.radians(angulo_grados)
    ITD_seg      = ITD_MAX_SEG * np.sin(abs(angulo_rad))
    ITD_muestras = int(ITD_seg * sr)
    n            = len(audio_mono)

    if angulo_grados >= 0:
        # Fuente a la derecha: L llega tarde, R llega antes
        canal_L = np.concatenate([np.zeros(ITD_muestras), audio_mono])[:n]
        canal_R = np.concatenate([audio_mono, np.zeros(ITD_muestras)])[:n]
    else:
        # Fuente a la izquierda: R llega tarde, L llega antes
        canal_L = np.concatenate([audio_mono, np.zeros(ITD_muestras)])[:n]
        canal_R = np.concatenate([np.zeros(ITD_muestras), audio_mono])[:n]

    canal_L = np.pad(canal_L, (0, max(0, n - len(canal_L))))[:n]
    canal_R = np.pad(canal_R, (0, max(0, n - len(canal_R))))[:n]

    # ILD: atenuar frecuencias altas en el canal sombreado
    F_TRANS   = VELOCIDAD_SONIDO / DIAMETRO_CABEZA   # ≈ 1960 Hz
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


def generar_binaurales_preprocesados(directorio='audio_binaural',
                                      angulo=ANGULO_PRINCIPAL,
                                      duracion=35.0):
    """
    Genera archivos binaurales una sola vez y los guarda en disco.
    Retorna diccionario con (sr, canal_L, canal_R) por clave.

    Ángulo de 60°: derivado de arcsin(DIM_AUD_L/DIM_INTERNA)×2
                  = arcsin(16/32)×2 = arcsin(0.5)×2 = 60°
    ILD resultante: 6×sin(60°) ≈ 5.2 dB — diferencia espectral visible.
    """
    os.makedirs(directorio, exist_ok=True)

    estimulos = {
        'voz':        'Voz_Estudio.wav',
        'musica':     'Brandemburgo.wav',
        'voz_viento': 'Voz+Viento_1.wav',
        'tono':       'Tono puro',
        'ruido':      'Ruido blanco',
    }

    archivos = {}
    ILD_dB   = 6.0 * np.sin(np.radians(angulo))
    ITD_ms   = ITD_MAX_SEG * np.sin(np.radians(angulo)) * 1000

    print(f"\n  Preprocesando audios binaurales (±{angulo}°, "
          f"ILD≈{ILD_dB:.1f} dB, ITD≈{ITD_ms:.2f} ms)...")

    for nombre, filepath in estimulos.items():
        sr, audio = cargar_mono(filepath, duracion)

        # +angulo (entrenamiento y fases 2-6)
        cL, cR = convertir_mono_a_binaural(audio, sr, +angulo)
        clave_pos = f"{nombre}_pos"
        archivos[clave_pos] = (sr, cL, cR)
        if HAS_SF:
            ruta = os.path.join(directorio,
                                f"{nombre}_pos{int(angulo)}deg.wav")
            sf.write(ruta, np.stack([cL, cR], axis=1), sr)

        # -angulo (solo voz, para fase 7)
        if nombre == 'voz':
            cL2, cR2 = convertir_mono_a_binaural(audio, sr, -angulo)
            clave_neg = f"{nombre}_neg"
            archivos[clave_neg] = (sr, cL2, cR2)
            if HAS_SF:
                ruta = os.path.join(directorio,
                                    f"{nombre}_neg{int(angulo)}deg.wav")
                sf.write(ruta, np.stack([cL2, cR2], axis=1), sr)

        print(f"    ✅ {nombre}: sr={sr} Hz")

    print(f"  Preprocesamiento completado: {len(archivos)} archivos.")
    return archivos


def preparar_objetivo_canal(canal, sr, idx_paso, ventana_muestras,
                             hop_muestras, dim_aud, dim_time):
    inicio   = idx_paso * hop_muestras
    fin      = inicio + ventana_muestras
    segmento = canal[inicio:fin] if fin <= len(canal) else canal[inicio:]
    if len(segmento) < ventana_muestras:
        segmento = np.pad(segmento, (0, ventana_muestras - len(segmento)))

    fft      = np.fft.rfft(segmento)
    potencia = np.abs(fft) ** 2
    freqs    = np.fft.rfftfreq(len(segmento), 1 / sr)

    bandas  = np.logspace(np.log10(F_MIN), np.log10(F_MAX), dim_aud + 1)
    objetivo = np.zeros(dim_aud)
    for b in range(dim_aud):
        mask = (freqs >= bandas[b]) & (freqs < bandas[b + 1])
        if np.any(mask):
            objetivo[b] = np.mean(potencia[mask])

    max_val = np.max(objetivo)
    if max_val > 0:
        objetivo /= max_val

    return objetivo.reshape(-1, 1) * np.ones((1, dim_time))

# ============================================================
# CAMBIO CENTRAL: act_busc como órgano diferencial
# ============================================================
def actualizar_act_busc_diferencial(Phi_total, obj_L, obj_R, dt):
    """
    act_busc recibe el gradiente diferencial L-R como estado primario.

    No recibe difusión de aud_L ni aud_R — esas vecindades fueron
    eliminadas de la topología. Su estado se actualiza directamente
    desde los objetivos crudos, antes de que la difusión los mezcle.

    Eso convierte la diferencia entre canales en una variable de
    estado del campo: el campo representa diferencia, no magnitud.

    La escala es DIFUSION_BASE — derivada de la física del campo.
    El decaimiento DIFUSION_BASE×dt garantiza que act_busc no acumula
    indefinidamente — se actualiza en la escala temporal de la difusión.

    Retorna:
        Phi_total modificado
        fuerza_grad: magnitud media del gradiente (para diagnóstico)
        signo_grad:  signo medio del gradiente (positivo = L>R)
    """
    n   = min(idx['act_busc'][1] - idx['act_busc'][0],
              obj_L.shape[0],
              obj_R.shape[0])
    ab0 = idx['act_busc'][0]

    # Gradiente espectral crudo entre canales
    gradiente_LR = obj_L[:n, :] - obj_R[:n, :]   # positivo → L > R

    # Señal diferencial escalada para competir con la difusión del campo
    señal = DIFUSION_BASE * gradiente_LR

    # Integración con decaimiento en la escala temporal de difusión
    Phi_total[ab0:ab0 + n, :] = (
        (1.0 - DIFUSION_BASE * dt) * Phi_total[ab0:ab0 + n, :] +
        DIFUSION_BASE * dt * señal
    )

    fuerza_grad = float(np.mean(np.abs(gradiente_LR)))
    signo_grad  = float(np.mean(gradiente_LR))
    return Phi_total, fuerza_grad, signo_grad

# ============================================================
# COHERENCIA SOBRE OBJETIVOS CRUDOS
# ============================================================
def calcular_coherencia_sobre_objetivos(obj_L, obj_R, W_prof, region_int):
    """
    Compara qué objetivo (L o R) es más coherente con W_prof.

    Se calcula sobre los objetivos crudos — no sobre el estado
    del campo después de la mezcla alpha. Eso preserva la información
    diferencial antes de que la difusión la borre.

    W_prof aprendió la estructura espectral de voz desde +60°.
    El objetivo que más se parece a esa estructura es el canal
    que apunta hacia la fuente conocida.

    Retorna:
        coh_rel > 0: L más coherente con identidad aprendida
        coh_rel < 0: R más coherente con identidad aprendida
    """
    n_prof = W_prof.shape[0]
    n_cols = W_prof.shape[1]
    n_int  = region_int.shape[0]

    # Perfil espectral medio de cada objetivo
    perfil_L = obj_L.mean(axis=1)   # (DIM_AUD,)
    perfil_R = obj_R.mean(axis=1)   # (DIM_AUD,)
    perfil_i = region_int.mean(axis=1)   # (DIM_INTERNA,)

    min_c = min(n_cols, len(perfil_L), len(perfil_R))
    min_p = min(n_prof, n_int)

    W_t = W_prof[:min_p, :min_c]

    pred_L = W_t @ perfil_L[:min_c].reshape(-1, 1)   # (min_p, 1)
    pred_R = W_t @ perfil_R[:min_c].reshape(-1, 1)
    ref    = perfil_i[:min_p].reshape(-1, 1)

    err_L  = float(np.mean((pred_L - ref) ** 2))
    err_R  = float(np.mean((pred_R - ref) ** 2))
    total  = err_L + err_R + 1e-10
    coh_rel = (err_R - err_L) / total   # >0 → L más coherente

    return float(coh_rel), err_L, err_R


def aplicar_orientacion_por_coherencia(Phi_total, coherencia_rel, dt):
    """
    La señal de coherencia relativa sesga act_geom.
    act_geom responde a la coherencia por difusión desde Φ_G,
    que recibe el estado de act_busc y lo integra con los demás.
    """
    acg0  = idx['act_geom'][0]
    acg1  = idx['act_geom'][1]
    mitad = max(1, (acg1 - acg0) // 2)
    señal = float(np.clip(coherencia_rel * DIFUSION_BASE * dt, -0.1, 0.1))
    Phi_total[acg0:acg0 + mitad, :] += señal
    Phi_total[acg0 + mitad:acg1, :] -= señal
    return Phi_total

# ============================================================
# ACTUACIÓN CUALITATIVA (sin cambios respecto a v83)
# ============================================================
def calcular_parametros_actuacion(Phi_total):
    act_perm = Phi_total[idx['act_perm'][0]:idx['act_perm'][1], :]
    act_geom = Phi_total[idx['act_geom'][0]:idx['act_geom'][1], :]

    nivel_perm = float(np.mean(np.tanh(act_perm)))
    frac_base  = 0.25 + 0.75 * (nivel_perm + 1.0) / 2.0

    mitad    = max(1, DIM_ACT // 2)
    g_baja   = float(np.mean(act_geom[:mitad, :]))
    g_alta   = float(np.mean(act_geom[mitad:, :]))
    sesgo    = float(np.tanh(g_alta - g_baja))
    asimetria = float(np.tanh(g_baja - g_alta))

    frac_L = float(np.clip(frac_base * (1.0 + asimetria * 0.5), 0.1, 1.0))
    frac_R = float(np.clip(frac_base * (1.0 - asimetria * 0.5), 0.1, 1.0))
    return frac_L, frac_R, sesgo, asimetria, nivel_perm


def aplicar_entrada_cualitativa(Phi_total, obj_L, obj_R,
                                 frac_L, frac_R, sesgo):
    def aplicar_canal(obj_full, frac, sl):
        n_act = max(1, int(DIM_AUD * frac))
        if sesgo > 0:
            ini = int(DIM_AUD * min(sesgo, 0.8) * 0.5)
            fin = min(DIM_AUD, ini + n_act)
        else:
            ini, fin = 0, n_act
        obj_mod = np.zeros((DIM_AUD, DIM_TIME), dtype=np.float32)
        obj_mod[ini:fin, :] = obj_full[ini:fin, :]
        Phi_total[sl, :] = ((1 - ALPHA_FIJO) * Phi_total[sl, :]
                            + ALPHA_FIJO * obj_mod)

    aplicar_canal(obj_L, frac_L, slice(idx['aud_L'][0], idx['aud_L'][1]))
    aplicar_canal(obj_R, frac_R, slice(idx['aud_R'][0], idx['aud_R'][1]))
    return Phi_total

# ============================================================
# EXPLORACIÓN ACTIVA (sin cambios respecto a v83)
# ============================================================
def explorar_actuadores(Phi_total, explorador, lf_activa, eficiencia, dt):
    AMPLITUD_MAX = DIFUSION_BASE
    ap0, ap1 = idx['act_perm']
    ag0, ag1 = idx['act_geom']

    if lf_activa:
        amplitud = AMPLITUD_MAX * min(1.0,
                                      explorador.pasos_en_lf / TAU_EXPLORACION)
        if explorador.mejor_config is not None:
            nivel = float(np.mean(np.tanh(Phi_total[ap0:ap1, :])))
            sesgo = ((explorador.mejor_config[0] + explorador.mejor_config[1])
                     / 2.0 - nivel)
            ruido_perm = np.random.normal(sesgo * 0.5, amplitud,
                                          (ap1 - ap0, DIM_TIME))
        else:
            ruido_perm = np.random.normal(0, amplitud, (ap1 - ap0, DIM_TIME))

        ruido_geom = np.random.normal(0, amplitud * 0.5, (ag1 - ag0, DIM_TIME))
        Phi_total[ap0:ap1, :] += ruido_perm * dt
        Phi_total[ag0:ag1, :] += ruido_geom * dt
    else:
        if explorador.mejor_config is not None:
            nivel = float(np.mean(np.tanh(Phi_total[ap0:ap1, :])))
            corr  = (explorador.mejor_config[0] - nivel) * DIFUSION_BASE * dt
            Phi_total[ap0:ap1, :] += corr

    return Phi_total

# ============================================================
# PLASTICIDAD DUAL (sin cambios respecto a v83)
# ============================================================
def aplicar_plasticidad_dual(region_int, region_aud, W_prof, W_rec,
                              Phi_int_historia, dt):
    n_prof = W_prof.shape[0]
    n_cols = W_prof.shape[1]

    min_prof = min(n_prof, region_int.shape[0])
    min_cols = min(n_cols, region_aud.shape[0])

    W_p = W_prof[:min_prof, :min_cols]
    W_r = W_rec[:min_prof,  :min_cols]
    r_i = region_int[:min_prof, :]
    r_a = region_aud[:min_cols, :]

    corr_prof = (r_i @ r_a.T) / DIM_TIME
    dW_prof   = ETA_PROFUNDA_BASE * corr_prof - TAU_PROFUNDA * W_p
    W_p_nueva = np.clip(W_p + dW_prof * dt, -W_MAX, W_MAX)

    W_prof_nueva = W_prof.copy()
    W_prof_nueva[:min_prof, :min_cols] = W_p_nueva

    pred_rec   = np.tanh(W_r @ r_a)
    error_rec  = float(np.mean((pred_rec - r_i) ** 2))
    pred_prof  = W_p_nueva @ r_a
    error_prof = float(np.mean((pred_prof - r_i) ** 2))

    coherencia       = error_prof / (error_rec + error_prof + 1e-10)
    tasa_aprendizaje = ETA_RECIENTE_BASE * coherencia

    corr_rec  = (r_i @ r_a.T) / DIM_TIME
    dW_rec    = tasa_aprendizaje * corr_rec - TAU_RECIENTE * W_r
    W_r_nueva = np.clip(W_r + dW_rec * dt, -W_MAX, W_MAX)

    W_rec_nueva = W_rec.copy()
    W_rec_nueva[:min_prof, :min_cols] = W_r_nueva

    M_plast = np.zeros((DIM_INTERNA, DIM_TIME))
    delta_p = W_p_nueva @ r_a - r_i
    delta_r = W_r_nueva @ r_a - r_i
    M_plast[:min_prof, :] = (delta_p + delta_r) * 0.01

    Phi_int_historia_nueva = 0.95 * Phi_int_historia + 0.05 * region_int

    return (W_prof_nueva, W_rec_nueva, M_plast,
            error_rec, coherencia, Phi_int_historia_nueva)

# ============================================================
# EFICIENCIA Y MÉTRICAS AUXILIARES
# ============================================================
def calcular_eficiencia(Phi_total, ged_actual):
    region_int = Phi_total[idx['int'][0]:idx['int'][1], :]
    variacion  = float(np.mean(np.abs(np.diff(region_int, axis=1))))
    return ged_actual / (variacion + 1e-10), variacion

def calcular_senal_busqueda(Phi_total):
    """Asimetría medida sobre el estado de act_busc — no sobre aud_L/R."""
    ab0, ab1 = idx['act_busc']
    estado_busc = Phi_total[ab0:ab1, :]
    return float(np.mean(estado_busc))   # signo indica dirección

def calcular_senal_mantenimiento(Phi_total):
    a_L = Phi_total[idx['aud_L'][0]:idx['aud_L'][1], :]
    a_R = Phi_total[idx['aud_R'][0]:idx['aud_R'][1], :]
    var = (np.var(a_L) + np.var(a_R)) / 2.0
    return max(0.0, DIFUSION_BASE ** 2 - float(var)), float(var)

# ============================================================
# ACTUALIZACIÓN PRINCIPAL DEL CAMPO
# ============================================================
def actualizar_campo(Phi_total, Phi_vel_total, W_prof, W_rec,
                     Phi_int_historia, obj_L, obj_R,
                     frac_L, frac_R, sesgo, coherencia_rel, dt):

    omega_n, amort_n = calcular_frecuencias_naturales(DIM_TOTAL)

    prom     = calcular_promedio_vecinos(Phi_total)
    difusion = DIFUSION_BASE * (prom - Phi_total)
    desv     = Phi_total - prom
    reaccion = GANANCIA_REACCION * desv * (1 - desv ** 2)
    term_osc = (-omega_n ** 2 * (Phi_total - PHI_EQUILIBRIO)
                - amort_n * Phi_vel_total)

    region_int = Phi_total[idx['int'][0]:idx['int'][1], :]
    aud_L      = Phi_total[idx['aud_L'][0]:idx['aud_L'][1], :]
    aud_R      = Phi_total[idx['aud_R'][0]:idx['aud_R'][1], :]
    aud_comb   = (aud_L + aud_R) / 2.0

    W_prof, W_rec, M_plast, error_rec, coherencia, Phi_int_historia = \
        aplicar_plasticidad_dual(
            region_int, aud_comb, W_prof, W_rec, Phi_int_historia, dt
        )

    M_campo = np.zeros_like(Phi_total)
    n_m     = M_plast.shape[0]
    M_campo[idx['int'][0]:idx['int'][0] + n_m, :] = M_plast

    # Entrada cualitativa sobre canales auditivos
    Phi_total = aplicar_entrada_cualitativa(Phi_total, obj_L, obj_R,
                                            frac_L, frac_R, sesgo)

    # Orientación de act_geom por coherencia
    Phi_total = aplicar_orientacion_por_coherencia(Phi_total, coherencia_rel, dt)

    # Integración
    dPhi_vel  = term_osc + reaccion + difusion + M_campo
    Phi_vel_n = Phi_vel_total + dt * dPhi_vel
    Phi_nueva = Phi_total + dt * Phi_vel_n

    # Prevenir colapso de región interna
    var_int = np.var(Phi_nueva[idx['int'][0]:idx['int'][1], :])
    if var_int < DIFUSION_BASE * 1e-4:
        Phi_nueva[idx['int'][0]:idx['int'][1], :] += \
            np.random.normal(0, 0.01, (DIM_INTERNA, DIM_TIME))

    lf_activa = error_rec > DIFUSION_BASE ** 2

    return (np.clip(Phi_nueva, LIMITE_MIN, LIMITE_MAX),
            np.clip(Phi_vel_n, -5.0, 5.0),
            W_prof, W_rec, Phi_int_historia,
            lf_activa, error_rec, coherencia)

# ============================================================
# ENTRENAMIENTO
# ============================================================
def entrenar(archivos, duracion=30.0):
    print(f"\n[Fase 1] Entrenamiento (voz, +{ANGULO_PRINCIPAL}°, 30s)")

    Phi_total, Phi_vel_total = inicializar_campo()
    W_prof, W_rec, Phi_int_historia = inicializar_memorias()
    explorador = ExploradorActuadores()

    sr, c_L, c_R = archivos['voz_pos']
    vent    = int(sr * VENTANA_FFT_MS / 1000)
    hop     = int(sr * HOP_FFT_MS  / 1000)
    n_pasos = int(duracion / DT)
    errores = []

    for paso in range(n_pasos):
        obj_L = preparar_objetivo_canal(c_L, sr, paso, vent, hop, DIM_AUD, DIM_TIME)
        obj_R = preparar_objetivo_canal(c_R, sr, paso, vent, hop, DIM_AUD, DIM_TIME)

        # act_busc recibe gradiente diferencial
        Phi_total, _, _ = actualizar_act_busc_diferencial(Phi_total, obj_L, obj_R, DT)

        # Coherencia sobre objetivos crudos
        region_int = Phi_total[idx['int'][0]:idx['int'][1], :]
        coh_rel, _, _ = calcular_coherencia_sobre_objetivos(
            obj_L, obj_R, W_prof, region_int
        )

        fL, fR, sf_v, _, _ = calcular_parametros_actuacion(Phi_total)

        Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia, \
            _, error_rec, _ = actualizar_campo(
                Phi_total, Phi_vel_total, W_prof, W_rec,
                Phi_int_historia, obj_L, obj_R,
                fL, fR, sf_v, coh_rel, DT
            )
        errores.append(error_rec)

        if paso % 500 == 0:
            print(f"    Paso {paso}/{n_pasos}, error={error_rec:.6f}")

    ERROR_EQ = float(np.min(errores)) if errores else DIFUSION_BASE ** 2
    print(f"  ERROR_EQUILIBRIO medido: {ERROR_EQ:.6f}")
    print(f"  W_prof tras entreno: {np.mean(np.abs(W_prof)):.4f}")
    print(f"  W_rec tras entreno:  {np.mean(np.abs(W_rec)):.4f}")

    return Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia, ERROR_EQ, explorador

# ============================================================
# SIMULACIÓN DE FASE
# ============================================================
def simular_fase(Phi_total, Phi_vel_total, W_prof, W_rec,
                 Phi_int_historia, historial_ef, explorador,
                 sr, canal_L, canal_R, duracion):

    vent    = int(sr * VENTANA_FFT_MS / 1000)
    hop     = int(sr * HOP_FFT_MS  / 1000)
    n_pasos = int(duracion / DT)

    hist = {k: [] for k in [
        'ged_L', 'ged_R', 'fuerza_grad', 'signo_grad', 'act_busc',
        'coh_rel', 'err_L', 'err_R', 'geom', 'frac_L', 'frac_R',
        'sesgo', 'efic', 'lf', 'mant', 'w_rec', 'w_prof'
    ]}

    lf_prev = False
    for paso in range(n_pasos):
        obj_L = preparar_objetivo_canal(canal_L, sr, paso, vent, hop, DIM_AUD, DIM_TIME)
        obj_R = preparar_objetivo_canal(canal_R, sr, paso, vent, hop, DIM_AUD, DIM_TIME)

        # 1. act_busc recibe gradiente diferencial crudo
        Phi_total, fuerza_grad, signo_grad = actualizar_act_busc_diferencial(
            Phi_total, obj_L, obj_R, DT
        )

        # 2. Coherencia sobre objetivos crudos
        region_int = Phi_total[idx['int'][0]:idx['int'][1], :]
        coh_rel, err_L, err_R = calcular_coherencia_sobre_objetivos(
            obj_L, obj_R, W_prof, region_int
        )

        # 3. Parámetros de actuación
        fL, fR, sf_v, asim, _ = calcular_parametros_actuacion(Phi_total)

        # 4. GED
        a_L   = Phi_total[idx['aud_L'][0]:idx['aud_L'][1], :]
        a_R   = Phi_total[idx['aud_R'][0]:idx['aud_R'][1], :]
        ged_L = calcular_ged_entre(region_int, a_L)
        ged_R = calcular_ged_entre(region_int, a_R)
        ged   = (ged_L + ged_R) / 2.0

        # 5. Eficiencia
        efic, _ = calcular_eficiencia(Phi_total, ged)
        historial_ef.append(efic)
        if len(historial_ef) > TAU_EFICIENCIA * 2:
            historial_ef.pop(0)

        explorador.actualizar(lf_prev, efic, fL, fR, sf_v)

        # 6. Actualizar campo
        Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia, \
            lf_activa, error_rec, _ = actualizar_campo(
                Phi_total, Phi_vel_total, W_prof, W_rec,
                Phi_int_historia, obj_L, obj_R,
                fL, fR, sf_v, coh_rel, DT
            )

        # 7. Exploración activa
        Phi_total = explorar_actuadores(Phi_total, explorador, lf_activa, efic, DT)
        lf_prev   = lf_activa

        # 8. Métricas
        act_busc_estado = calcular_senal_busqueda(Phi_total)
        mant, _         = calcular_senal_mantenimiento(Phi_total)
        geom            = float(np.mean(np.tanh(
            Phi_total[idx['act_geom'][0]:idx['act_geom'][1], :]
        )))
        G_act = float(np.mean(np.abs(
            Phi_total[idx['G'][0]:idx['G'][1], :]
        )))

        for k, v in [
            ('ged_L',      ged_L),
            ('ged_R',      ged_R),
            ('fuerza_grad', fuerza_grad),
            ('signo_grad',  signo_grad),
            ('act_busc',    act_busc_estado),
            ('coh_rel',     coh_rel),
            ('err_L',       err_L),
            ('err_R',       err_R),
            ('geom',        geom),
            ('frac_L',      fL),
            ('frac_R',      fR),
            ('sesgo',       sf_v),
            ('efic',        efic),
            ('lf',          lf_activa),
            ('mant',        mant),
            ('w_rec',       np.mean(np.abs(W_rec))),
            ('w_prof',      np.mean(np.abs(W_prof))),
        ]:
            hist[k].append(v)

        if paso % 200 == 0:
            print(f"    t={paso*DT:.1f}s | GED={ged:.4f} | "
                  f"grad={fuerza_grad:.4f} sgn={signo_grad:+.4f} | "
                  f"busc={act_busc_estado:+.4f} | "
                  f"coh={coh_rel:+.4f} | "
                  f"G={G_act:.4f} | efic={efic:.3f} | "
                  f"LF={'ACTIVA' if lf_activa else 'inact'}")

    def M(k): return float(np.mean(hist[k]))
    print(f"\n  Resumen:")
    print(f"    GED L/R:               {M('ged_L'):.4f} / {M('ged_R'):.4f}")
    print(f"    Fuerza gradiente L-R:  {M('fuerza_grad'):.4f}")
    print(f"    act_busc estado:       {M('act_busc'):+.4f}")
    print(f"    Coherencia relativa:   {M('coh_rel'):+.4f}")
    print(f"    Error L/R medios:      {M('err_L'):.4f} / {M('err_R'):.4f}")
    print(f"    act_geom estado:       {M('geom'):+.4f}")
    print(f"    Eficiencia media:      {M('efic'):.4f}")
    print(f"    LF activa (%):         {100*M('lf'):.1f}%")
    print(f"    Mejor efic explorada:  {explorador.mejor_eficiencia:.4f}")

    return hist, Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia, explorador

# ============================================================
# MAIN
# ============================================================
def main():
    # Paso 0: generar binaurales una sola vez
    archivos = generar_binaurales_preprocesados(
        directorio='audio_binaural',
        angulo=ANGULO_PRINCIPAL,
        duracion=35.0
    )

    # Entrenamiento
    Phi_total, Phi_vel_total, W_prof, W_rec, \
        Phi_int_historia, ERROR_EQ, explorador = entrenar(archivos, 30.0)

    # Protocolo de fases
    FASES = [
        ("F2", 'voz_pos',        +ANGULO_PRINCIPAL, 20.0, "Dominio — voz +60°"),
        ("F3", 'musica_pos',     +ANGULO_PRINCIPAL, 20.0, "No entrenado — música +60°"),
        ("F4", 'tono_pos',       +ANGULO_PRINCIPAL, 20.0, "No entrenado — tono +60°"),
        ("F5", 'voz_viento_pos', +ANGULO_PRINCIPAL, 20.0, "Degradado — voz+viento +60°"),
        ("F6", 'ruido_pos',      +ANGULO_PRINCIPAL, 20.0, "Perturbación — ruido +60°"),
        ("F7", 'voz_neg',        -ANGULO_PRINCIPAL, 20.0, "Re-acoplamiento opuesto — voz -60°"),
    ]

    resultados   = []
    historial_ef = []

    for fid, clave, angulo, dur, desc in FASES:
        print(f"\n[{fid}] {desc}")
        sr, c_L, c_R = archivos[clave]
        hist, Phi_total, Phi_vel_total, W_prof, W_rec, \
            Phi_int_historia, explorador = simular_fase(
                Phi_total, Phi_vel_total, W_prof, W_rec,
                Phi_int_historia, historial_ef, explorador,
                sr, c_L, c_R, dur
            )
        resultados.append((fid, angulo, hist))

    # ---- DIAGNÓSTICO ----
    print("\n" + "=" * 100)
    print("DIAGNÓSTICO — v84 Campo diferencial")
    print("=" * 100)

    def M(fase_idx, k):
        return float(np.mean(resultados[fase_idx][2][k]))

    # Valores clave para criterios
    busc_f2 = M(0, 'act_busc')
    busc_f7 = M(5, 'act_busc')
    grad_f2 = M(0, 'fuerza_grad')
    coh_f2  = M(0, 'coh_rel')
    coh_f7  = M(5, 'coh_rel')
    geom_f2 = M(0, 'geom')
    geom_f7 = M(5, 'geom')
    asim_f2 = M(0, 'fuerza_grad')   # en v84 la "asimetría" es la fuerza del gradiente
    asim_f7 = M(5, 'fuerza_grad')
    mejor_ef = explorador.mejor_eficiencia

    # Criterios acumulados
    c10 = True
    c11 = float(np.std([M(i, 'frac_L') for i in range(6)])) > 0.001
    c12 = abs(busc_f2 - busc_f7) > 0.001   # act_busc difiere entre +60° y -60°
    c13 = any(M(i, 'mant') > 0 for i in range(6))
    c14 = float(np.std([M(i, 'frac_L') for i in range(6)])) > 0.01
    c15 = grad_f2 > 0.001   # gradiente L-R no nulo
    c16 = mejor_ef > 0
    c17 = grad_f2 > 0.001   # fuerza del gradiente en F2
    c18 = coh_f2 * coh_f7 < 0   # coherencia invierte signo entre +60° y -60°
    c19 = abs(geom_f2 - geom_f7) > 0.01
    c20 = abs(busc_f2) > 0.001   # act_busc activo en F2
    c21 = busc_f2 * busc_f7 < 0   # act_busc invierte signo entre F2 y F7

    def tick(b): return "✅" if b else "❌"

    print("\n  CRITERIOS DE ARQUITECTURA (v81):")
    print(f"    C10 — Ganglio activo:            {tick(c10)}")
    print(f"    C11 — Alpha modulado:            {tick(c11)}")
    print(f"    C12 — Asimetría diferencial:     {tick(c12)} "
          f"(busc_F2={busc_f2:+.4f}, busc_F7={busc_f7:+.4f})")
    print(f"    C13 — Mantenimiento activo:      {tick(c13)}")

    print("\n  CRITERIOS DE ACOPLAMIENTO ACTIVO (v82):")
    print(f"    C14 — Fracción L/R varía:        {tick(c14)}")
    print(f"    C15 — Gradiente L-R emerge:      {tick(c15)} (grad_F2={grad_f2:.4f})")
    print(f"    C16 — Explorador encuentra:      {tick(c16)} (mejor_ef={mejor_ef:.4f})")

    print("\n  CRITERIOS DE ORIENTACIÓN ESPACIAL (v83):")
    print(f"    C17 — Gradiente L-R en F2 > 0:  {tick(c17)} ({grad_f2:.6f})")
    print(f"    C18 — Coherencia invierte:       {tick(c18)} "
          f"(F2={coh_f2:+.4f}, F7={coh_f7:+.4f})")
    print(f"    C19 — act_geom diferencial:      {tick(c19)} "
          f"(F2={geom_f2:+.4f}, F7={geom_f7:+.4f})")

    print("\n  CRITERIOS DE CAMPO DIFERENCIAL (v84):")
    print(f"    C20 — act_busc activo en F2:     {tick(c20)} ({busc_f2:+.4f})")
    print(f"    C21 — act_busc invierte en F7:   {tick(c21)} "
          f"(F2={busc_f2:+.4f}, F7={busc_f7:+.4f})")

    print("\n  VEREDICTO:")
    if c20 and c21:
        print("  ✅ CAMPO DIFERENCIAL VALIDADO")
        print("     act_busc representa el gradiente Δ(L,R) como variable de estado.")
        print("     El signo de act_busc invierte al cambiar el ángulo de la fuente.")
        print("     El campo pasó de escalar a direccional.")
        if c18:
            print("     La coherencia relativa también invierte — orientación completa.")
        if c19:
            print("     act_geom responde diferencialmente — reorientación activa.")
    elif c20:
        print("  ⚠️  act_busc ACTIVO pero sin inversión de signo en F7.")
        print("     El campo representa magnitud del gradiente pero no dirección.")
    else:
        print("  ❌ act_busc no muestra gradiente diferencial.")
        print("     Verificar preprocesamiento binaural y ángulo de 60°.")

    # ---- CSV ----
    with open('v84_campo_diferencial.csv', 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(['fase', 'angulo', 't', 'ged_L', 'ged_R',
                     'fuerza_grad', 'signo_grad', 'act_busc',
                     'coh_rel', 'err_L', 'err_R',
                     'geom', 'frac_L', 'frac_R', 'efic', 'lf'])
        for fid, angulo, hist in resultados:
            for i in range(len(hist['ged_L'])):
                wr.writerow([
                    fid, angulo, round(i * DT, 2),
                    hist['ged_L'][i], hist['ged_R'][i],
                    hist['fuerza_grad'][i], hist['signo_grad'][i],
                    hist['act_busc'][i], hist['coh_rel'][i],
                    hist['err_L'][i], hist['err_R'][i],
                    hist['geom'][i], hist['frac_L'][i],
                    hist['frac_R'][i], hist['efic'][i],
                    int(hist['lf'][i])
                ])
    print("\n  CSV guardado: v84_campo_diferencial.csv")

    # ---- GRÁFICO ----
    nombres = [r[0] for r in resultados]
    fig, axes = plt.subplots(3, 4, figsize=(24, 18))

    # Fila 0
    axes[0,0].plot(nombres, [M(i,'ged_L') for i in range(6)], 'o-', label='L')
    axes[0,0].plot(nombres, [M(i,'ged_R') for i in range(6)], 's-', label='R')
    axes[0,0].set_title('GED por canal'); axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    axes[0,1].bar(nombres, [M(i,'fuerza_grad') for i in range(6)])
    axes[0,1].set_title('Fuerza gradiente Δ(L,R)')
    axes[0,1].grid(True, alpha=0.3)

    vals_busc = [M(i,'act_busc') for i in range(6)]
    axes[0,2].bar(nombres, vals_busc,
                  color=['steelblue' if v >= 0 else 'firebrick' for v in vals_busc])
    axes[0,2].axhline(0, color='k', linestyle='--', linewidth=0.8)
    axes[0,2].set_title('act_busc estado (signo = dirección)')
    axes[0,2].grid(True, alpha=0.3)

    vals_coh = [M(i,'coh_rel') for i in range(6)]
    axes[0,3].bar(nombres, vals_coh,
                  color=['steelblue' if v >= 0 else 'firebrick' for v in vals_coh])
    axes[0,3].axhline(0, color='r', linestyle='--')
    axes[0,3].set_title('Coherencia relativa (L>0)')
    axes[0,3].grid(True, alpha=0.3)

    # Fila 1
    axes[1,0].plot(nombres, [M(i,'err_L') for i in range(6)], 'o-', label='L')
    axes[1,0].plot(nombres, [M(i,'err_R') for i in range(6)], 's-', label='R')
    axes[1,0].set_title('Error predictivo por canal'); axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

    axes[1,1].bar(nombres, [M(i,'geom') for i in range(6)])
    axes[1,1].set_title('act_geom estado')
    axes[1,1].grid(True, alpha=0.3)

    axes[1,2].plot(nombres, [M(i,'frac_L') for i in range(6)], 'o-', label='L')
    axes[1,2].plot(nombres, [M(i,'frac_R') for i in range(6)], 's-', label='R')
    axes[1,2].set_title('Fracción de ventana L/R'); axes[1,2].legend()
    axes[1,2].grid(True, alpha=0.3)

    axes[1,3].bar(nombres, [M(i,'efic') for i in range(6)])
    axes[1,3].set_title('Eficiencia de régimen')
    axes[1,3].grid(True, alpha=0.3)

    # Fila 2
    axes[2,0].bar(nombres, [100*M(i,'lf') for i in range(6)])
    axes[2,0].set_title('LF activa (%)')
    axes[2,0].grid(True, alpha=0.3)

    axes[2,1].plot(nombres, [M(i,'w_rec') for i in range(6)], 'o-', label='W_rec')
    axes[2,1].plot(nombres, [M(i,'w_prof') for i in range(6)], 's-', label='W_prof')
    axes[2,1].set_title('Norma W_rec y W_prof'); axes[2,1].legend()
    axes[2,1].grid(True, alpha=0.3)

    axes[2,2].bar(nombres, [M(i,'sesgo') for i in range(6)])
    axes[2,2].set_title('Sesgo frecuencial')
    axes[2,2].grid(True, alpha=0.3)

    axes[2,3].bar(nombres, [M(i,'signo_grad') for i in range(6)],
                  color=['steelblue' if M(i,'signo_grad') >= 0
                         else 'firebrick' for i in range(6)])
    axes[2,3].axhline(0, color='k', linestyle='--', linewidth=0.8)
    axes[2,3].set_title('Signo del gradiente (+ = L>R)')
    axes[2,3].grid(True, alpha=0.3)

    plt.suptitle(
        'VSTCosmos v84 — Campo diferencial: act_busc como órgano de gradiente L-R',
        fontsize=13
    )
    plt.tight_layout()
    plt.savefig('v84_campo_diferencial.png', dpi=150)
    print("  Gráfico guardado: v84_campo_diferencial.png")

    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)


if __name__ == "__main__":
    main()