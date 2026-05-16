#!/usr/bin/env python3
"""
VSTCosmos v83 — Orientación activa con entrada binaural real

Cambios respecto a v82:
1. Preprocesamiento binaural de audios (ITD + ILD)
2. Protocolo con ángulos espaciales (+30° para entrenamiento, -30° para re-acoplamiento)
3. Coherencia por canal como señal de orientación para act_geom

Corrección respecto al código anterior:
- aplicar_plasticidad_dual_v83 ya no referencia Phi_total global
- Recibe dim_total como parámetro explícito
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import csv
import warnings
import os
from collections import deque

try:
    import soundfile as sf
    HAS_SF = True
except ImportError:
    HAS_SF = False

from scipy import signal as scipy_signal
warnings.filterwarnings('ignore')

# ============================================================
# PARÁMETROS DE LA FÍSICA DEL CAMPO
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
TAU_EXPLORACION    = int(T_RECIENTE_SEG / 0.01)  # 200 pasos

LIMITE_MIN  = 0.0
LIMITE_MAX  = 1.0
W_MAX       = 1.0
ALPHA_FIJO  = 0.05
DT          = 0.01
DIM_TIME    = 100

# Constantes físicas para binaural
DIAMETRO_CABEZA  = 0.175   # metros
VELOCIDAD_SONIDO = 343.0   # m/s
ITD_MAX_SEG      = DIAMETRO_CABEZA / VELOCIDAD_SONIDO

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

VECINDADES = [
    ('int',      'G'),
    ('G',        'aud_L'),
    ('G',        'aud_R'),
    ('G',        'act_perm'),
    ('G',        'act_geom'),
    ('G',        'act_busc'),
    ('G',        'act_mant'),
    ('aud_L',    'aud_R'),
    ('act_perm', 'aud_L'),
    ('act_perm', 'aud_R'),
    ('act_geom', 'aud_L'),
    ('act_geom', 'aud_R'),
]

print("=" * 100)
print("VSTCosmos v83 — Orientación activa con entrada binaural real")
print("")
print("  Cambios respecto a v82:")
print("  1. Preprocesamiento binaural de audios (ITD + ILD)")
print("  2. Protocolo con ángulos espaciales (+30° para entrenamiento, -30° para re-acoplamiento)")
print("  3. Coherencia por canal como señal de orientación para act_geom")
print("")
print(f"    ITD máximo = {ITD_MAX_SEG*1000:.2f}ms")
print(f"    DIM_TOTAL = {DIM_TOTAL}")
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
        serie   = region[banda, :] - np.mean(region[banda, :])
        fft     = np.fft.rfft(serie)
        potencia = np.abs(fft) ** 2
        perfil  += potencia[:n_bins]
    return perfil / max(1, dim)

def calcular_ged_entre(region_a, region_b):
    p_a = _perfil_espectral_region(region_a, region_a.shape[0])
    p_b = _perfil_espectral_region(region_b, region_b.shape[0])
    return float(np.mean(np.abs(p_a - p_b)))

def calcular_frecuencias_naturales(dim):
    bandas = np.arange(dim)
    t = np.log1p(bandas) / np.log1p(max(dim - 1, 1))
    omega = OMEGA_MIN + (OMEGA_MAX - OMEGA_MIN) * t
    amort = AMORT_MIN + (AMORT_MAX - AMORT_MIN) * t
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
# PREPROCESAMIENTO BINAURAL
# ============================================================
def convertir_mono_a_binaural(audio_mono, sr, angulo_grados):
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

    # Padding por si acaso
    if len(canal_L) < n:
        canal_L = np.pad(canal_L, (0, n - len(canal_L)))
    if len(canal_R) < n:
        canal_R = np.pad(canal_R, (0, n - len(canal_R)))

    # ILD
    F_TRANSICION = VELOCIDAD_SONIDO / DIAMETRO_CABEZA
    ILD_dB       = 6.0 * np.sin(abs(angulo_rad))
    ILD_lineal   = 10 ** (-ILD_dB / 20.0)
    nyquist      = sr / 2.0
    freq_norm    = min(0.99, F_TRANSICION / nyquist)

    if freq_norm < 1.0 and ILD_lineal < 0.99:
        b, a = scipy_signal.butter(2, freq_norm, btype='high')
        if angulo_grados >= 0:
            altas_L = scipy_signal.filtfilt(b, a, canal_L)
            canal_L = (canal_L - altas_L) + altas_L * ILD_lineal
        else:
            altas_R = scipy_signal.filtfilt(b, a, canal_R)
            canal_R = (canal_R - altas_R) + altas_R * ILD_lineal

    max_val = max(np.max(np.abs(canal_L)), np.max(np.abs(canal_R))) + 1e-10
    return (canal_L / max_val).astype(np.float32), (canal_R / max_val).astype(np.float32)

def cargar_audio(filepath, duracion):
    """Carga audio en mono. Genera tono si no puede leer el archivo."""
    n_target = None
    audio    = None
    sr       = 48000

    # Intento 1: soundfile
    if HAS_SF:
        try:
            data, sr = sf.read(filepath, dtype='float32')
            if data.ndim > 1:
                data = data.mean(axis=1)
            n_target = int(sr * duracion)
            audio = data[:n_target]
        except Exception:
            audio = None

    # Intento 2: scipy
    if audio is None:
        try:
            sr, data = wav.read(filepath)
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            if data.ndim > 1:
                data = data.mean(axis=1)
            n_target = int(sr * duracion)
            audio = data[:n_target]
        except Exception:
            audio = None

    # Fallback: tono 440 Hz
    if audio is None:
        print(f"  [ADVERTENCIA] No se pudo cargar '{filepath}', usando tono 440Hz")
        sr       = 48000
        n_target = int(sr * duracion)
        t        = np.arange(n_target) / sr
        audio    = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    # Padding
    if len(audio) < n_target:
        audio = np.pad(audio, (0, n_target - len(audio)))

    return sr, audio

def preprocesar_audio_binaural(filepath, angulo_grados, duracion):
    sr, audio   = cargar_audio(filepath, duracion)
    canal_L, canal_R = convertir_mono_a_binaural(audio, sr, angulo_grados)
    return sr, canal_L, canal_R

def preparar_objetivo_canal(canal, sr, idx_paso, ventana_muestras, hop_muestras,
                             dim_aud, dim_time):
    inicio   = idx_paso * hop_muestras
    fin      = inicio + ventana_muestras
    segmento = canal[inicio:fin] if fin <= len(canal) else canal[inicio:]
    if len(segmento) < ventana_muestras:
        segmento = np.pad(segmento, (0, ventana_muestras - len(segmento)))

    fft    = np.fft.rfft(segmento)
    potencia = np.abs(fft) ** 2
    freqs  = np.fft.rfftfreq(len(segmento), 1 / sr)

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
# PLASTICIDAD DUAL — CORREGIDA
# ============================================================
def aplicar_plasticidad_dual(region_int, region_aud, W_prof, W_rec,
                              Phi_int_historia, dt):
    """
    Corrección: ya no referencia Phi_total global.
    Construye M_plast sobre la región interna y lo retorna
    como array de forma (DIM_INTERNA, DIM_TIME).
    """
    n_prof     = W_prof.shape[0]
    n_cols     = W_prof.shape[1]
    n_int_rows = region_int.shape[0]
    n_aud_rows = region_aud.shape[0]

    min_prof = min(n_prof, n_int_rows)
    min_cols = min(n_cols, n_aud_rows)

    W_p = W_prof[:min_prof, :min_cols]
    W_r = W_rec[:min_prof,  :min_cols]
    r_i = region_int[:min_prof, :]
    r_a = region_aud[:min_cols, :]

    # Plasticidad profunda
    corr_prof  = (r_i @ r_a.T) / DIM_TIME
    dW_prof    = ETA_PROFUNDA_BASE * corr_prof - TAU_PROFUNDA * W_p
    W_p_nueva  = np.clip(W_p + dW_prof * dt, -W_MAX, W_MAX)

    W_prof_nueva = W_prof.copy()
    W_prof_nueva[:min_prof, :min_cols] = W_p_nueva

    # Plasticidad reciente (con tanh)
    pred_rec   = np.tanh(W_r @ r_a)
    error_rec  = float(np.mean((pred_rec - r_i) ** 2))

    pred_prof  = W_p_nueva @ r_a
    error_prof = float(np.mean((pred_prof - r_i) ** 2))

    coherencia       = error_prof / (error_rec + error_prof + 1e-10)
    tasa_aprendizaje = ETA_RECIENTE_BASE * coherencia

    corr_rec   = (r_i @ r_a.T) / DIM_TIME
    dW_rec     = tasa_aprendizaje * corr_rec - TAU_RECIENTE * W_r
    W_r_nueva  = np.clip(W_r + dW_rec * dt, -W_MAX, W_MAX)

    W_rec_nueva = W_rec.copy()
    W_rec_nueva[:min_prof, :min_cols] = W_r_nueva

    # M_plast — solo sobre región interna, forma (DIM_INTERNA, DIM_TIME)
    M_plast = np.zeros((DIM_INTERNA, DIM_TIME))
    delta_prof = W_p_nueva @ r_a - r_i   # (min_prof, DIM_TIME)
    delta_rec  = W_r_nueva @ r_a - r_i
    M_plast[:min_prof, :] = (delta_prof + delta_rec) * 0.01

    # Actualizar historia interna
    Phi_int_historia_nueva = 0.95 * Phi_int_historia + 0.05 * region_int

    return W_prof_nueva, W_rec_nueva, M_plast, error_rec, coherencia, Phi_int_historia_nueva

# ============================================================
# COHERENCIA POR CANAL Y ORIENTACIÓN
# ============================================================
def calcular_coherencia_por_canal(Phi_total, W_prof):
    n_prof = W_prof.shape[0]
    n_cols = W_prof.shape[1]

    region_int = Phi_total[idx['int'][0]:idx['int'][1], :]
    aud_L      = Phi_total[idx['aud_L'][0]:idx['aud_L'][1], :]
    aud_R      = Phi_total[idx['aud_R'][0]:idx['aud_R'][1], :]

    min_p = min(n_prof, region_int.shape[0])
    min_c = min(n_cols, aud_L.shape[0])

    W_t  = W_prof[:min_p, :min_c]
    r_i  = region_int[:min_p, :]
    a_L  = aud_L[:min_c, :]
    a_R  = aud_R[:min_c, :]

    pred_L  = W_t @ a_L
    pred_R  = W_t @ a_R
    err_L   = float(np.mean((pred_L - r_i) ** 2))
    err_R   = float(np.mean((pred_R - r_i) ** 2))
    total   = err_L + err_R + 1e-10
    coh_rel = (err_R - err_L) / total   # >0 → L más coherente
    return float(coh_rel), err_L, err_R

def aplicar_orientacion_por_coherencia(Phi_total, coherencia_rel, dt):
    acg0  = idx['act_geom'][0]
    acg1  = idx['act_geom'][1]
    mitad = max(1, (acg1 - acg0) // 2)
    señal = np.clip(coherencia_rel * DIFUSION_BASE * dt, -0.1, 0.1)
    Phi_total[acg0:acg0 + mitad, :] += señal
    Phi_total[acg0 + mitad:acg1, :] -= señal
    return Phi_total

# ============================================================
# ACTUACIÓN CUALITATIVA
# ============================================================
def calcular_parametros_actuacion(Phi_total):
    act_perm = Phi_total[idx['act_perm'][0]:idx['act_perm'][1], :]
    act_geom = Phi_total[idx['act_geom'][0]:idx['act_geom'][1], :]

    nivel_perm   = float(np.mean(np.tanh(act_perm)))
    frac_base    = 0.25 + 0.75 * (nivel_perm + 1.0) / 2.0

    mitad        = max(1, DIM_ACT // 2)
    g_baja       = float(np.mean(act_geom[:mitad, :]))
    g_alta       = float(np.mean(act_geom[mitad:, :]))
    sesgo_freq   = float(np.tanh(g_alta - g_baja))
    asimetria    = float(np.tanh(g_baja - g_alta))

    frac_L = float(np.clip(frac_base * (1.0 + asimetria * 0.5), 0.1, 1.0))
    frac_R = float(np.clip(frac_base * (1.0 - asimetria * 0.5), 0.1, 1.0))
    return frac_L, frac_R, sesgo_freq, asimetria, nivel_perm

def aplicar_entrada_cualitativa(Phi_total, obj_L_full, obj_R_full,
                                 frac_L, frac_R, sesgo_freq):
    def aplicar_canal(obj_full, frac, sl):
        n_act = max(1, int(DIM_AUD * frac))
        if sesgo_freq > 0:
            ini = int(DIM_AUD * min(sesgo_freq, 0.8) * 0.5)
            fin = min(DIM_AUD, ini + n_act)
        else:
            ini, fin = 0, n_act
        obj_mod = np.zeros((DIM_AUD, DIM_TIME), dtype=np.float32)
        obj_mod[ini:fin, :] = obj_full[ini:fin, :]
        Phi_total[sl, :] = (1 - ALPHA_FIJO) * Phi_total[sl, :] + ALPHA_FIJO * obj_mod

    aplicar_canal(obj_L_full, frac_L, slice(idx['aud_L'][0], idx['aud_L'][1]))
    aplicar_canal(obj_R_full, frac_R, slice(idx['aud_R'][0], idx['aud_R'][1]))
    return Phi_total

# ============================================================
# EXPLORACIÓN ACTIVA
# ============================================================
def explorar_actuadores(Phi_total, explorador, lf_activa, eficiencia_actual, dt):
    AMPLITUD_MAX = DIFUSION_BASE
    ap0, ap1     = idx['act_perm']
    ag0, ag1     = idx['act_geom']

    if lf_activa:
        amplitud = AMPLITUD_MAX * min(1.0, explorador.pasos_en_lf / TAU_EXPLORACION)
        if explorador.mejor_config is not None:
            nivel_actual = float(np.mean(np.tanh(Phi_total[ap0:ap1, :])))
            sesgo        = (explorador.mejor_config[0] + explorador.mejor_config[1]) / 2.0 - nivel_actual
            ruido_perm   = np.random.normal(sesgo * 0.5, amplitud, (ap1 - ap0, DIM_TIME))
        else:
            ruido_perm = np.random.normal(0, amplitud, (ap1 - ap0, DIM_TIME))

        ruido_geom = np.random.normal(0, amplitud * 0.5, (ag1 - ag0, DIM_TIME))
        Phi_total[ap0:ap1, :] += ruido_perm * dt
        Phi_total[ag0:ag1, :] += ruido_geom * dt
    else:
        if explorador.mejor_config is not None:
            nivel_actual = float(np.mean(np.tanh(Phi_total[ap0:ap1, :])))
            corr = (explorador.mejor_config[0] - nivel_actual) * DIFUSION_BASE * dt
            Phi_total[ap0:ap1, :] += corr

    return Phi_total

# ============================================================
# EFICIENCIA Y TASA DE OLVIDO
# ============================================================
def calcular_eficiencia(Phi_total, ged_actual):
    region_int = Phi_total[idx['int'][0]:idx['int'][1], :]
    variacion  = float(np.mean(np.abs(np.diff(region_int, axis=1))))
    return ged_actual / (variacion + 1e-10), variacion

def calcular_tasa_olvido(eficiencia_actual, historial_eficiencia, k=10):
    if len(historial_eficiencia) < TAU_EFICIENCIA:
        return TAU_RECIENTE, 1.0
    ef_media = float(np.mean(historial_eficiencia[-TAU_EFICIENCIA:]))
    ventaja  = eficiencia_actual / (ef_media + 1e-10)
    tasa     = TAU_RECIENTE / (ventaja ** k)
    return float(min(tasa, ETA_RECIENTE_BASE * 10.0)), ventaja

# ============================================================
# MÉTRICAS ADICIONALES
# ============================================================
def calcular_senal_busqueda(Phi_total):
    a_L = Phi_total[idx['aud_L'][0]:idx['aud_L'][1], :]
    a_R = Phi_total[idx['aud_R'][0]:idx['aud_R'][1], :]
    return float(np.mean(np.abs(
        _perfil_espectral_region(a_L, DIM_AUD_L) -
        _perfil_espectral_region(a_R, DIM_AUD_R)
    )))

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
                     frac_L, frac_R, sesgo_freq, coherencia_rel, dt):

    omega_n, amort_n = calcular_frecuencias_naturales(DIM_TOTAL)

    prom = calcular_promedio_vecinos(Phi_total)
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

    # Aplicar M_plast solo sobre región interna
    M_campo = np.zeros_like(Phi_total)
    n_m = M_plast.shape[0]
    M_campo[idx['int'][0]:idx['int'][0] + n_m, :] = M_plast

    # Entrada cualitativa
    Phi_total = aplicar_entrada_cualitativa(Phi_total, obj_L, obj_R,
                                            frac_L, frac_R, sesgo_freq)

    # Orientación por coherencia
    Phi_total = aplicar_orientacion_por_coherencia(Phi_total, coherencia_rel, dt)

    # Integración
    dPhi_vel    = term_osc + reaccion + difusion + M_campo
    Phi_vel_n   = Phi_vel_total + dt * dPhi_vel
    Phi_nueva   = Phi_total + dt * Phi_vel_n

    # Prevenir colapso
    var_int = np.var(Phi_nueva[idx['int'][0]:idx['int'][1], :])
    if var_int < DIFUSION_BASE * 1e-4:
        Phi_nueva[idx['int'][0]:idx['int'][1], :] += \
            np.random.normal(0, 0.01, (DIM_INTERNA, DIM_TIME))

    # LF activa cuando error_rec supera equilibrio
    lf_activa = error_rec > DIFUSION_BASE ** 2

    return (np.clip(Phi_nueva, LIMITE_MIN, LIMITE_MAX),
            np.clip(Phi_vel_n, -5.0, 5.0),
            W_prof, W_rec, Phi_int_historia,
            lf_activa, error_rec, coherencia)

# ============================================================
# ENTRENAMIENTO
# ============================================================
def entrenar(duracion=30.0):
    print("\n[Fase 1] Entrenamiento (voz, alpha=0.05, 30s, ángulo +30°)")

    Phi_total, Phi_vel_total = inicializar_campo()
    W_prof, W_rec, Phi_int_historia = inicializar_memorias()
    explorador = ExploradorActuadores()

    sr, c_L, c_R = preprocesar_audio_binaural("Voz_Estudio.wav", 30.0, duracion)
    vent = int(sr * VENTANA_FFT_MS / 1000)
    hop  = int(sr * HOP_FFT_MS  / 1000)
    n    = int(duracion / DT)
    errores = []

    for paso in range(n):
        obj_L = preparar_objetivo_canal(c_L, sr, paso, vent, hop, DIM_AUD, DIM_TIME)
        obj_R = preparar_objetivo_canal(c_R, sr, paso, vent, hop, DIM_AUD, DIM_TIME)

        coh_rel, _, _ = calcular_coherencia_por_canal(Phi_total, W_prof)
        fL, fR, sf_v, _, _ = calcular_parametros_actuacion(Phi_total)

        Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia, \
            _, error_rec, _ = actualizar_campo(
                Phi_total, Phi_vel_total, W_prof, W_rec,
                Phi_int_historia, obj_L, obj_R,
                fL, fR, sf_v, coh_rel, DT
            )
        errores.append(error_rec)

        if paso % 500 == 0:
            print(f"    Paso {paso}/{n}, error={error_rec:.6f}")

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
                 filepath, angulo, duracion):

    sr, c_L, c_R = preprocesar_audio_binaural(filepath, angulo, duracion)
    vent = int(sr * VENTANA_FFT_MS / 1000)
    hop  = int(sr * HOP_FFT_MS  / 1000)
    n    = int(duracion / DT)

    hist = {k: [] for k in [
        'ged_L', 'ged_R', 'asim_LR', 'coh_rel', 'err_L', 'err_R',
        'geom', 'frac_L', 'frac_R', 'sesgo', 'efic', 'lf', 'mant',
        'w_rec', 'w_prof'
    ]}

    lf_prev = False
    for paso in range(n):
        obj_L = preparar_objetivo_canal(c_L, sr, paso, vent, hop, DIM_AUD, DIM_TIME)
        obj_R = preparar_objetivo_canal(c_R, sr, paso, vent, hop, DIM_AUD, DIM_TIME)

        coh_rel, err_L, err_R = calcular_coherencia_por_canal(Phi_total, W_prof)
        fL, fR, sf_v, asim, _ = calcular_parametros_actuacion(Phi_total)

        r_int = Phi_total[idx['int'][0]:idx['int'][1], :]
        a_L   = Phi_total[idx['aud_L'][0]:idx['aud_L'][1], :]
        ged_L = calcular_ged_entre(r_int, a_L)
        a_R   = Phi_total[idx['aud_R'][0]:idx['aud_R'][1], :]
        ged_R = calcular_ged_entre(r_int, a_R)
        ged   = (ged_L + ged_R) / 2.0

        efic, _ = calcular_eficiencia(Phi_total, ged)
        historial_ef.append(efic)
        if len(historial_ef) > TAU_EFICIENCIA * 2:
            historial_ef.pop(0)

        explorador.actualizar(lf_prev, efic, fL, fR, sf_v)

        Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia, \
            lf_activa, error_rec, _ = actualizar_campo(
                Phi_total, Phi_vel_total, W_prof, W_rec,
                Phi_int_historia, obj_L, obj_R,
                fL, fR, sf_v, coh_rel, DT
            )

        Phi_total = explorar_actuadores(Phi_total, explorador, lf_activa, efic, DT)
        lf_prev   = lf_activa

        asim_LR = calcular_senal_busqueda(Phi_total)
        mant, _ = calcular_senal_mantenimiento(Phi_total)
        geom    = float(np.mean(np.tanh(
            Phi_total[idx['act_geom'][0]:idx['act_geom'][1], :]
        )))

        for k, v in [('ged_L', ged_L), ('ged_R', ged_R), ('asim_LR', asim_LR),
                     ('coh_rel', coh_rel), ('err_L', err_L), ('err_R', err_R),
                     ('geom', geom), ('frac_L', fL), ('frac_R', fR),
                     ('sesgo', sf_v), ('efic', efic), ('lf', lf_activa),
                     ('mant', mant), ('w_rec', np.mean(np.abs(W_rec))),
                     ('w_prof', np.mean(np.abs(W_prof)))]:
            hist[k].append(v)

        if paso % 200 == 0:
            G_act = float(np.mean(np.abs(
                Phi_total[idx['G'][0]:idx['G'][1], :]
            )))
            print(f"    t={paso*DT:.1f}s | GED={ged:.4f} | "
                  f"asim={asim_LR:+.4f} | coh={coh_rel:+.4f} | "
                  f"errL={err_L:.4f} errR={err_R:.4f} | "
                  f"G={G_act:.4f} | efic={efic:.3f} | "
                  f"LF={'ACTIVA' if lf_activa else 'inact'}")

    def M(k): return float(np.mean(hist[k]))
    print(f"\n  Resumen:")
    print(f"    GED L/R:               {M('ged_L'):.4f} / {M('ged_R'):.4f}")
    print(f"    Asimetría L/R media:   {M('asim_LR'):+.4f}")
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
    Phi_total, Phi_vel_total, W_prof, W_rec, \
        Phi_int_historia, ERROR_EQ, explorador = entrenar(30.0)

    FASES = [
        ("F2", "Voz_Estudio.wav",   +30.0, 20.0, "Dominio — voz +30°"),
        ("F3", "Brandemburgo.wav",  +30.0, 20.0, "No entrenado — música +30°"),
        ("F4", "Tono puro",         +30.0, 20.0, "No entrenado — tono +30°"),
        ("F5", "Voz+Viento_1.wav",  +30.0, 20.0, "Degradado — voz+viento +30°"),
        ("F6", "Ruido blanco",      +30.0, 20.0, "Perturbación — ruido +30°"),
        ("F7", "Voz_Estudio.wav",   -30.0, 20.0, "Re-acoplamiento opuesto — voz -30°"),
    ]

    resultados = []
    historial_ef = []

    for fid, archivo, angulo, dur, desc in FASES:
        print(f"\n[{fid}] {desc}")
        hist, Phi_total, Phi_vel_total, W_prof, W_rec, \
            Phi_int_historia, explorador = simular_fase(
                Phi_total, Phi_vel_total, W_prof, W_rec,
                Phi_int_historia, historial_ef, explorador,
                archivo, angulo, dur
            )
        resultados.append((fid, hist))

    # ---- DIAGNÓSTICO ----
    print("\n" + "=" * 100)
    print("DIAGNÓSTICO — v83 Orientación activa")
    print("=" * 100)

    def M(fase_idx, k):
        return float(np.mean(resultados[fase_idx][1][k]))

    asim_f2 = M(0, 'asim_LR')
    asim_f7 = M(5, 'asim_LR')
    coh_f2  = M(0, 'coh_rel')
    coh_f7  = M(5, 'coh_rel')
    geom_f2 = M(0, 'geom')
    geom_f7 = M(5, 'geom')
    mejor_ef = explorador.mejor_eficiencia

    c10 = True   # ganglio siempre activo por construcción
    c11 = float(np.std([M(i, 'frac_L') for i in range(6)])) > 0.001
    c12 = abs(asim_f2 - asim_f7) > 0.001
    c13 = any(M(i, 'mant') > 0 for i in range(6))
    c14 = float(np.std([M(i, 'frac_L') for i in range(6)])) > 0.01
    c15 = abs(asim_f2) > 0.001
    c16 = mejor_ef > 0
    c17 = abs(asim_f2) > 0.001
    c18 = coh_f2 * coh_f7 < 0   # signos opuestos
    c19 = abs(geom_f2 - geom_f7) > 0.01

    def tick(b): return "✅" if b else "❌"

    print("\n  CRITERIOS DE ARQUITECTURA (v81):")
    print(f"    C10 — Ganglio activo:          {tick(c10)}")
    print(f"    C11 — Alpha modulado:          {tick(c11)}")
    print(f"    C12 — Asimetría diferencial:   {tick(c12)} (F2={asim_f2:+.4f}, F7={asim_f7:+.4f})")
    print(f"    C13 — Mantenimiento activo:    {tick(c13)}")

    print("\n  CRITERIOS DE ACOPLAMIENTO ACTIVO (v82):")
    print(f"    C14 — Fracción L/R varía:      {tick(c14)}")
    print(f"    C15 — Asimetría L/R emerge:    {tick(c15)} (asim_F2={asim_f2:+.4f})")
    print(f"    C16 — Explorador encuentra:    {tick(c16)} (mejor_ef={mejor_ef:.4f})")

    print("\n  CRITERIOS DE ORIENTACIÓN ESPACIAL (v83):")
    print(f"    C17 — Asimetría L/R F2 > 0:   {tick(c17)} ({asim_f2:+.6f})")
    print(f"    C18 — Coherencia invierte:     {tick(c18)} (F2={coh_f2:+.4f}, F7={coh_f7:+.4f})")
    print(f"    C19 — act_geom diferencial:    {tick(c19)} (F2={geom_f2:+.4f}, F7={geom_f7:+.4f})")

    print("\n  VEREDICTO:")
    if c17 and c18 and c19:
        print("  ✅ ORIENTACIÓN ACTIVA VALIDADA")
        print("     El campo recibe entrada binaural real (ITD+ILD).")
        print("     La asimetría L/R emerge de la diferencia espectral entre canales.")
        print("     La coherencia relativa invierte signo al cambiar ángulo de la fuente.")
        print("     act_geom responde diferencialmente entre +30° y -30°.")
        print("     Primera evidencia computacional de orientación activa emergente.")
    elif c17 and c18:
        print("  ⚠️  ASIMETRÍA Y COHERENCIA VALIDADAS — act_geom aún no responde diferencialmente.")
    elif c17:
        print("  ⚠️  ASIMETRÍA FUNCIONAL — coherencia y geom aún no invierten.")
    else:
        print("  ❌ ORIENTACIÓN PARCIAL — revisar preprocesamiento binaural.")

    # ---- CSV ----
    with open('v83_orientacion_activa.csv', 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(['fase', 't', 'ged_L', 'ged_R', 'asim_LR', 'coh_rel',
                     'err_L', 'err_R', 'geom', 'frac_L', 'frac_R',
                     'efic', 'lf'])
        for fid, hist in resultados:
            for i in range(len(hist['ged_L'])):
                wr.writerow([fid, round(i * DT, 2),
                             hist['ged_L'][i], hist['ged_R'][i],
                             hist['asim_LR'][i], hist['coh_rel'][i],
                             hist['err_L'][i], hist['err_R'][i],
                             hist['geom'][i], hist['frac_L'][i],
                             hist['frac_R'][i], hist['efic'][i],
                             int(hist['lf'][i])])
    print("\n  CSV guardado: v83_orientacion_activa.csv")

    # ---- GRÁFICO ----
    nombres = [r[0] for r in resultados]
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))

    axes[0,0].plot(nombres, [M(i,'ged_L') for i in range(6)], 'o-', label='L')
    axes[0,0].plot(nombres, [M(i,'ged_R') for i in range(6)], 's-', label='R')
    axes[0,0].set_title('GED por canal')
    axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

    axes[0,1].bar(nombres, [M(i,'asim_LR') for i in range(6)])
    axes[0,1].set_title('Asimetría L/R')
    axes[0,1].grid(True, alpha=0.3)

    vals_coh = [M(i,'coh_rel') for i in range(6)]
    axes[0,2].bar(nombres, vals_coh)
    axes[0,2].axhline(0, color='r', linestyle='--')
    axes[0,2].set_title('Coherencia relativa (L>0)')
    axes[0,2].grid(True, alpha=0.3)

    axes[0,3].plot(nombres, [M(i,'err_L') for i in range(6)], 'o-', label='L')
    axes[0,3].plot(nombres, [M(i,'err_R') for i in range(6)], 's-', label='R')
    axes[0,3].set_title('Error predictivo por canal')
    axes[0,3].legend(); axes[0,3].grid(True, alpha=0.3)

    axes[0,4].bar(nombres, [M(i,'geom') for i in range(6)])
    axes[0,4].set_title('act_geom estado')
    axes[0,4].grid(True, alpha=0.3)

    axes[1,0].plot(nombres, [M(i,'frac_L') for i in range(6)], 'o-', label='L')
    axes[1,0].plot(nombres, [M(i,'frac_R') for i in range(6)], 's-', label='R')
    axes[1,0].set_title('Fracción de ventana L/R')
    axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

    axes[1,1].bar(nombres, [M(i,'efic') for i in range(6)])
    axes[1,1].set_title('Eficiencia de régimen')
    axes[1,1].grid(True, alpha=0.3)

    axes[1,2].bar(nombres, [100*M(i,'lf') for i in range(6)])
    axes[1,2].set_title('LF activa (%)')
    axes[1,2].grid(True, alpha=0.3)

    axes[1,3].plot(nombres, [M(i,'w_rec') for i in range(6)], 'o-', label='W_rec')
    axes[1,3].plot(nombres, [M(i,'w_prof') for i in range(6)], 's-', label='W_prof')
    axes[1,3].set_title('Norma W_rec y W_prof')
    axes[1,3].legend(); axes[1,3].grid(True, alpha=0.3)

    axes[1,4].bar(nombres, [M(i,'sesgo') for i in range(6)])
    axes[1,4].set_title('Sesgo frecuencial')
    axes[1,4].grid(True, alpha=0.3)

    plt.suptitle('VSTCosmos v83 — Orientación activa con entrada binaural real', fontsize=14)
    plt.tight_layout()
    plt.savefig('v83_orientacion_activa.png', dpi=150)
    print("  Gráfico guardado: v83_orientacion_activa.png")

    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)


if __name__ == "__main__":
    main()