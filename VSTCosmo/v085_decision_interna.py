#!/usr/bin/env python3
"""
VSTCosmos v85 — Decisión interna: act_busc como integrador de coherencia

Cambio único respecto a v84b:
- act_busc recibe la coherencia relativa (estable) en lugar del gradiente crudo (oscilante)
- La estabilidad emerge de W_prof, que el campo construyó durante el entrenamiento
- El campo puede sostener una dirección porque la coherencia tiene memoria
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import csv
import warnings
import os
from collections import deque
from scipy import signal as scipy_signal

try:
    import soundfile as sf
    HAS_SF = True
except ImportError:
    HAS_SF = False

warnings.filterwarnings('ignore')

# ============================================================
# PARÁMETROS DE LA FÍSICA DEL CAMPO
# ============================================================
DIM_INTERNA = 32

DIFUSION_BASE = 0.15
GANANCIA_REACCION = 0.05

OMEGA_MIN = 0.05
OMEGA_MAX = 0.50
AMORT_MIN = 0.01
AMORT_MAX = 0.08
PHI_EQUILIBRIO = 0.5

VENTANA_FFT_MS = 25
HOP_FFT_MS = 10
F_MIN = 80
F_MAX = 8000

OMEGA_MEDIA = (OMEGA_MIN + OMEGA_MAX) / 2.0
T_PROFUNDA_SEG = 1.0 / OMEGA_MIN
T_RECIENTE_SEG = 1.0 / OMEGA_MAX
T_PROFUNDA_PASOS = int(T_PROFUNDA_SEG / 0.01)
T_RECIENTE_PASOS = int(T_RECIENTE_SEG / 0.01)

ETA_PROFUNDA_BASE = (1.0 / T_PROFUNDA_PASOS) / DIFUSION_BASE
ETA_RECIENTE_BASE = (1.0 / T_RECIENTE_PASOS) / DIFUSION_BASE
TAU_PROFUNDA = OMEGA_MIN
TAU_RECIENTE = OMEGA_MIN * 0.5
TAU_EFICIENCIA = int(1.0 / (OMEGA_MIN * 0.01))
TAU_EXPLORACION = int(T_RECIENTE_SEG / 0.01)

LIMITE_MIN = 0.0
LIMITE_MAX = 1.0
W_MAX = 1.0
ALPHA_FIJO = 0.05
DT = 0.01
DIM_TIME = 100

# Constantes físicas para binaural
DIAMETRO_CABEZA = 0.175
VELOCIDAD_SONIDO = 343.0
ITD_MAX_SEG = DIAMETRO_CABEZA / VELOCIDAD_SONIDO

# Decaimiento de act_busc (escala T_RECIENTE)
DECAIMIENTO_ACT_BUSC = DT / T_RECIENTE_SEG  # ≈ 0.005

# ============================================================
# ARQUITECTURA DEL CAMPO EXPANDIDO
# ============================================================
DIM_GANGLIO = DIM_INTERNA // 2
DIM_AUD = DIM_GANGLIO
DIM_ACT = DIM_GANGLIO // 2

DIM_AUD_L = DIM_AUD
DIM_AUD_R = DIM_AUD
DIM_ACT_PERM = DIM_ACT
DIM_ACT_GEOM = DIM_ACT
DIM_ACT_BUSC = DIM_ACT
DIM_ACT_MANT = DIM_ACT

idx = {}
idx['int'] = (0, DIM_INTERNA)
idx['G'] = (idx['int'][1], idx['int'][1] + DIM_GANGLIO)
idx['aud_L'] = (idx['G'][1], idx['G'][1] + DIM_AUD_L)
idx['aud_R'] = (idx['aud_L'][1], idx['aud_L'][1] + DIM_AUD_R)
idx['act_perm'] = (idx['aud_R'][1], idx['aud_R'][1] + DIM_ACT_PERM)
idx['act_geom'] = (idx['act_perm'][1], idx['act_perm'][1] + DIM_ACT_GEOM)
idx['act_busc'] = (idx['act_geom'][1], idx['act_geom'][1] + DIM_ACT_BUSC)
idx['act_mant'] = (idx['act_busc'][1], idx['act_busc'][1] + DIM_ACT_MANT)

DIM_TOTAL = idx['act_mant'][1]

# Vecindades v85
VECINDADES = [
    ('int', 'G'),
    ('G', 'aud_L'),
    ('G', 'aud_R'),
    ('G', 'act_perm'),
    ('G', 'act_geom'),
    ('G', 'act_busc'),
    ('G', 'act_mant'),
    ('aud_L', 'aud_R'),
    ('act_perm', 'aud_L'),
    ('act_perm', 'aud_R'),
    ('act_geom', 'aud_L'),
    ('act_geom', 'aud_R'),
]

print("=" * 100)
print("VSTCosmos v85 — Decisión interna: act_busc como integrador de coherencia")
print("")
print("  Cambio único respecto a v84b:")
print("  - act_busc recibe la coherencia relativa (estable) en lugar del gradiente crudo")
print("  - La estabilidad emerge de W_prof, que el campo construyó durante el entrenamiento")
print("  - El campo puede sostener una dirección porque la coherencia tiene memoria")
print("")
print("  Arquitectura del campo:")
print(f"    DIM_INTERNA = {DIM_INTERNA}")
print(f"    DIM_GANGLIO = {DIM_GANGLIO}")
print(f"    DIM_AUD = {DIM_AUD}")
print(f"    DIM_ACT = {DIM_ACT}")
print(f"    DIM_TOTAL = {DIM_TOTAL}")
print("=" * 100)

# ============================================================
# PREPROCESAMIENTO BINAURAL Y CARGA
# ============================================================
def cargar_audio_binaural(filepath, duracion):
    """Carga archivo binaural preprocesado"""
    try:
        data, sr = sf.read(filepath, dtype='float32')
        n_muestras = int(sr * duracion)
        if data.ndim == 1:
            canal_L = data[:n_muestras]
            canal_R = data[:n_muestras]
        else:
            canal_L = data[:n_muestras, 0]
            canal_R = data[:n_muestras, 1] if data.shape[1] > 1 else data[:n_muestras, 0]
        return sr, canal_L, canal_R
    except Exception as e:
        print(f"  [ERROR] No se pudo cargar {filepath}: {e}")
        sr = 48000
        n_muestras = int(sr * duracion)
        t = np.arange(n_muestras) / sr
        senal = 0.5 * np.sin(2 * np.pi * 440 * t)
        return sr, senal, senal

def cargar_todos_binaurales(directorio='audio_binaural', duracion=35.0):
    """Carga todos los archivos binaurales desde el directorio"""
    archivos = {}
    
    mapping = {
        'voz_pos': 'Voz_Estudio_pos60deg.wav',
        'voz_neg': 'Voz_Estudio_neg60deg.wav',
        'musica_pos': 'Brandemburgo_pos60deg.wav',
        'voz_viento_pos': 'Voz+Viento_1_pos60deg.wav',
        'tono_pos': 'Tono puro_pos60deg.wav',
        'ruido_pos': 'Ruido blanco_pos60deg.wav',
    }
    
    for clave, filename in mapping.items():
        filepath = os.path.join(directorio, filename)
        if os.path.exists(filepath):
            sr, canal_L, canal_R = cargar_audio_binaural(filepath, duracion)
            archivos[clave] = (filepath, sr, canal_L, canal_R)
            print(f"    ✅ {clave:15s} {filename}")
        else:
            print(f"    ❌ {clave:15s} {filename} no encontrado")
    
    return archivos

# ============================================================
# CLASE EXPLORADOR
# ============================================================
class ExploradorActuadores:
    def __init__(self):
        self.historial = []
        self.mejor_config = None
        self.mejor_eficiencia = 0.0
        self.pasos_en_lf = 0

    def actualizar(self, lf_activa, eficiencia_actual, fraccion_L, fraccion_R, sesgo_freq):
        if lf_activa:
            self.pasos_en_lf += 1
            self.historial.append((fraccion_L, fraccion_R, sesgo_freq, eficiencia_actual))
            if eficiencia_actual > self.mejor_eficiencia:
                self.mejor_eficiencia = eficiencia_actual
                self.mejor_config = (fraccion_L, fraccion_R, sesgo_freq)
        else:
            self.pasos_en_lf = 0

# ============================================================
# FUNCIONES BASE
# ============================================================
def inicializar_campo_v85():
    np.random.seed(None)
    Phi_total = np.random.normal(PHI_EQUILIBRIO, 0.01, (DIM_TOTAL, DIM_TIME))
    Phi_vel_total = np.zeros((DIM_TOTAL, DIM_TIME))
    return Phi_total, Phi_vel_total

def inicializar_memorias():
    W_prof = np.zeros((DIM_INTERNA, DIM_AUD))
    W_rec = np.zeros((DIM_INTERNA, DIM_AUD))
    Phi_int_historia = np.zeros((DIM_INTERNA, DIM_TIME))
    return W_prof, W_rec, Phi_int_historia

def _perfil_espectral_region(region, dim):
    perfil = np.zeros(50)
    for banda in range(min(dim, len(region))):
        serie = region[banda, :] - np.mean(region[banda, :])
        fft = np.fft.rfft(serie)
        perfil += np.abs(fft)[:50] ** 2
    return perfil / dim

def calcular_ged_entre(region_int, region_aud):
    perfil_int = _perfil_espectral_region(region_int, len(region_int))
    perfil_aud = _perfil_espectral_region(region_aud, len(region_aud))
    return float(np.mean(np.abs(perfil_int - perfil_aud)))

def calcular_frecuencias_naturales(dim_total):
    bandas = np.arange(dim_total)
    t = np.log1p(bandas) / np.log1p(dim_total - 1) if dim_total > 1 else np.zeros_like(bandas)
    omega = OMEGA_MIN + (OMEGA_MAX - OMEGA_MIN) * t
    amort = AMORT_MIN + (AMORT_MAX - AMORT_MIN) * t
    return omega.reshape(-1, 1), amort.reshape(-1, 1)

def calcular_promedio_vecinos_v85(Phi_total):
    promedio = np.zeros_like(Phi_total)
    conteo = np.zeros(DIM_TOTAL)

    for reg_a, reg_b in VECINDADES:
        ia0, ia1 = idx[reg_a]
        ib0, ib1 = idx[reg_b]
        n_a = ia1 - ia0
        n_b = ib1 - ib0

        for d in range(min(n_a, n_b)):
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
# PREPARAR OBJETIVO
# ============================================================
def preparar_objetivo_para_canal(canal, sr, idx_paso, ventana_muestras, hop_muestras, dim_aud, dim_time):
    inicio = idx_paso * hop_muestras
    fin = inicio + ventana_muestras

    if fin > len(canal):
        segmento = canal[inicio:]
        if len(segmento) < ventana_muestras:
            segmento = np.pad(segmento, (0, ventana_muestras - len(segmento)))
    else:
        segmento = canal[inicio:fin]

    fft = np.fft.rfft(segmento)
    potencia = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(len(segmento), 1/sr)

    bandas = np.logspace(np.log10(F_MIN), np.log10(F_MAX), dim_aud + 1)
    objetivo = np.zeros(dim_aud)
    for b in range(dim_aud):
        mask = (freqs >= bandas[b]) & (freqs < bandas[b+1])
        if np.any(mask):
            objetivo[b] = np.mean(potencia[mask])

    max_val = np.max(objetivo)
    if max_val > 0:
        objetivo = objetivo / max_val

    return objetivo.reshape(-1, 1) * np.ones((1, dim_time))

# ============================================================
# COHERENCIA SOBRE OBJETIVOS CRUDOS
# ============================================================
def calcular_coherencia_sobre_objetivos(obj_L, obj_R, W_prof, region_int):
    n_prof = W_prof.shape[0]
    n_cols = W_prof.shape[1]
    n_int = min(n_prof, region_int.shape[0])

    perfil_L = obj_L.mean(axis=1)[:n_cols]
    perfil_R = obj_R.mean(axis=1)[:n_cols]
    perfil_i = region_int.mean(axis=1)[:n_int]

    min_c = min(n_cols, len(perfil_L), len(perfil_R))
    min_p = min(n_int, n_prof)

    W_t = W_prof[:min_p, :min_c]

    pred_L = W_t @ perfil_L[:min_c].reshape(-1, 1)
    pred_R = W_t @ perfil_R[:min_c].reshape(-1, 1)
    ref = perfil_i[:min_p].reshape(-1, 1)

    err_L = float(np.mean((pred_L - ref) ** 2))
    err_R = float(np.mean((pred_R - ref) ** 2))
    total = err_L + err_R + 1e-10
    coh_rel = (err_R - err_L) / total

    return float(coh_rel), err_L, err_R

# ============================================================
# ACT_BUSC DESDE COHERENCIA (CAMBIO ARQUITECTÓNICO)
# ============================================================
def actualizar_act_busc_desde_coherencia(Phi_total, coherencia_rel, dt):
    """
    act_busc representa la coherencia relativa — no el gradiente crudo.
    
    La coherencia es estable porque W_prof es estable.
    No oscila con cada ventana FFT.
    El signo persiste mientras la fuente esté en la misma posición.
    
    Eso es memoria del signo sin umbral externo:
    la estabilidad emerge de W_prof, que el campo construyó
    durante el entrenamiento.
    """
    ab0 = idx['act_busc'][0]
    ab1 = idx['act_busc'][1]
    
    # La señal es coherencia_rel escalada a la amplitud del campo
    senal = coherencia_rel * DIFUSION_BASE
    
    # Decaimiento en escala T_RECIENTE — mismo que W_rec
    Phi_total[ab0:ab1, :] = (
        (1.0 - DECAIMIENTO_ACT_BUSC) * Phi_total[ab0:ab1, :] +
        DECAIMIENTO_ACT_BUSC * senal
    )
    
    return Phi_total

# ============================================================
# ORIENTACIÓN Y ACTUACIÓN
# ============================================================
def aplicar_orientacion_por_coherencia(Phi_total, coherencia_rel, dt):
    """La señal de coherencia relativa influye sobre act_geom por difusión"""
    acg_inicio = idx['act_geom'][0]
    acg_fin = idx['act_geom'][1]
    dim_act = acg_fin - acg_inicio
    mitad = max(1, dim_act // 2)
    
    # Fuerza de orientación derivada de la relación de escalas
    K_ORIENT = T_PROFUNDA_SEG / T_RECIENTE_SEG  # = 10
    
    senal_sesgo = coherencia_rel * DIFUSION_BASE * K_ORIENT * dt
    senal_sesgo = np.clip(senal_sesgo, -0.1, 0.1)
    
    Phi_total[acg_inicio:acg_inicio + mitad, :] += senal_sesgo
    Phi_total[acg_inicio + mitad:acg_fin, :] -= senal_sesgo
    
    return Phi_total

def calcular_parametros_actuacion(Phi_total):
    act_perm = Phi_total[idx['act_perm'][0]:idx['act_perm'][1], :]
    act_geom = Phi_total[idx['act_geom'][0]:idx['act_geom'][1], :]

    nivel_perm = float(np.mean(np.tanh(act_perm)))
    fraccion_base = 0.25 + 0.75 * (nivel_perm + 1.0) / 2.0

    mitad = max(1, DIM_ACT // 2)
    if mitad < DIM_ACT:
        banda_baja = float(np.mean(act_geom[:mitad, :]))
        banda_alta = float(np.mean(act_geom[mitad:, :]))
        sesgo_freq = float(np.tanh(banda_alta - banda_baja))
        geom_L_val = float(np.mean(act_geom[:mitad, :]))
        geom_R_val = float(np.mean(act_geom[mitad:, :]))
        asimetria = float(np.tanh(geom_L_val - geom_R_val))
    else:
        sesgo_freq = 0.0
        asimetria = 0.0

    fraccion_L = np.clip(fraccion_base * (1.0 + asimetria * 0.5), 0.1, 1.0)
    fraccion_R = np.clip(fraccion_base * (1.0 - asimetria * 0.5), 0.1, 1.0)

    return fraccion_L, fraccion_R, sesgo_freq, asimetria, nivel_perm

def aplicar_entrada_cualitativa(Phi_total, objetivo_L, objetivo_R, fraccion_L, fraccion_R, sesgo_freq):
    dim_aud = DIM_AUD_L
    dim_time = DIM_TIME

    def aplicar_canal(objetivo_full, fraccion, region_slice):
        n_bandas_activas = max(1, int(dim_aud * fraccion))
        if sesgo_freq > 0:
            inicio_banda = int(dim_aud * min(sesgo_freq, 0.8) * 0.5)
            fin_banda = min(dim_aud, inicio_banda + n_bandas_activas)
        else:
            inicio_banda = 0
            fin_banda = n_bandas_activas

        objetivo_mod = np.zeros((dim_aud, dim_time), dtype=np.float32)
        objetivo_mod[inicio_banda:fin_banda, :] = objetivo_full[inicio_banda:fin_banda, :]

        Phi_total[region_slice, :] = (
            (1 - ALPHA_FIJO) * Phi_total[region_slice, :] + ALPHA_FIJO * objetivo_mod
        )

    aplicar_canal(objetivo_L, fraccion_L, slice(idx['aud_L'][0], idx['aud_L'][1]))
    aplicar_canal(objetivo_R, fraccion_R, slice(idx['aud_R'][0], idx['aud_R'][1]))

    return Phi_total

# ============================================================
# EXPLORACIÓN ACTIVA
# ============================================================
def explorar_actuadores(Phi_total, explorador, lf_activa, eficiencia_actual, dt):
    act_perm = Phi_total[idx['act_perm'][0]:idx['act_perm'][1], :]
    act_geom = Phi_total[idx['act_geom'][0]:idx['act_geom'][1], :]

    if lf_activa:
        amplitud = DIFUSION_BASE * min(1.0, explorador.pasos_en_lf / TAU_EXPLORACION)

        if explorador.mejor_config is not None:
            mejor_perm = explorador.mejor_config[0]
            nivel_actual = float(np.mean(np.tanh(act_perm)))
            sesgo_perm = mejor_perm - nivel_actual
            ruido_perm = np.random.normal(sesgo_perm * 0.5, amplitud, act_perm.shape)
        else:
            ruido_perm = np.random.normal(0, amplitud, act_perm.shape)

        ruido_geom = np.random.normal(0, amplitud * 0.5, act_geom.shape)

        Phi_total[idx['act_perm'][0]:idx['act_perm'][1], :] += ruido_perm * dt
        Phi_total[idx['act_geom'][0]:idx['act_geom'][1], :] += ruido_geom * dt
    else:
        if explorador.mejor_config is not None:
            mejor_perm = explorador.mejor_config[0]
            nivel_actual = float(np.mean(np.tanh(act_perm)))
            correccion = (mejor_perm - nivel_actual) * DIFUSION_BASE * dt
            Phi_total[idx['act_perm'][0]:idx['act_perm'][1], :] += correccion

    return Phi_total

# ============================================================
# EFICIENCIA Y PLASTICIDAD
# ============================================================
def calcular_eficiencia_regimen(Phi_total, ged_actual):
    region_int = Phi_total[:DIM_INTERNA, :]
    variacion = float(np.mean(np.abs(np.diff(region_int, axis=1))))
    return ged_actual / (variacion + 1e-10), variacion

def calcular_tasa_olvido_por_eficiencia(eficiencia_actual, historial_eficiencia, k=10):
    if len(historial_eficiencia) < TAU_EFICIENCIA:
        return TAU_RECIENTE, 1.0, 1.0

    eficiencia_media = float(np.mean(historial_eficiencia[-TAU_EFICIENCIA:]))
    ventaja = eficiencia_actual / (eficiencia_media + 1e-10)
    ventaja_amp = ventaja ** k
    tasa_olvido = TAU_RECIENTE / ventaja_amp
    return min(tasa_olvido, ETA_RECIENTE_BASE * 10.0), ventaja, ventaja_amp

def aplicar_plasticidad_dual_v85(region_int, region_aud, W_prof, W_rec, Phi_int_historia, dt):
    min_prof = min(W_prof.shape[0], region_int.shape[0])
    min_cols = min(W_prof.shape[1], region_aud.shape[0])

    W_p = W_prof[:min_prof, :min_cols]
    W_r = W_rec[:min_prof, :min_cols]
    reg_int_t = region_int[:min_prof, :]
    reg_aud_t = region_aud[:min_cols, :]

    pred_prof = W_p @ reg_aud_t
    error_prof = np.mean((pred_prof - reg_int_t) ** 2)

    corr_prof = (reg_int_t @ reg_aud_t.T) / DIM_TIME
    dW_prof = ETA_PROFUNDA_BASE * corr_prof - TAU_PROFUNDA * W_p
    W_prof_n = np.zeros_like(W_prof)
    W_prof_n[:min_prof, :min_cols] = np.clip(W_p + dW_prof * dt, -W_MAX, W_MAX)

    pred_rec = np.tanh(W_r @ reg_aud_t)
    error_rec = np.mean((pred_rec - reg_int_t) ** 2)

    coherencia = error_prof / (error_rec + error_prof + 1e-10) if error_rec + error_prof > 0 else 0.5
    tasa_aprendizaje = ETA_RECIENTE_BASE * coherencia

    corr_rec = (reg_int_t @ reg_aud_t.T) / DIM_TIME
    dW_rec = tasa_aprendizaje * corr_rec - TAU_RECIENTE * W_r
    W_rec_n = np.zeros_like(W_rec)
    W_rec_n[:min_prof, :min_cols] = np.clip(W_r + dW_rec * dt, -W_MAX, W_MAX)

    M_plast = np.zeros_like(region_int)
    M_plast[:min_prof, :] = (W_p @ reg_aud_t - reg_int_t) + (W_r @ reg_aud_t - reg_int_t)
    M_plast = M_plast * 0.01

    Phi_int_historia_n = (1 - 0.05) * Phi_int_historia + 0.05 * region_int

    return W_prof_n, W_rec_n, M_plast, float(error_rec), float(coherencia)

def calcular_senal_busqueda(Phi_total):
    """Estado de act_busc centrado en PHI_EQUILIBRIO"""
    ab0 = idx['act_busc'][0]
    ab1 = idx['act_busc'][1]
    return float(np.mean(Phi_total[ab0:ab1, :])) - PHI_EQUILIBRIO

def calcular_senal_mantenimiento(Phi_total):
    aud_L = Phi_total[idx['aud_L'][0]:idx['aud_L'][1], :]
    aud_R = Phi_total[idx['aud_R'][0]:idx['aud_R'][1], :]
    var_media = (np.var(aud_L) + np.var(aud_R)) / 2.0
    return max(0.0, DIFUSION_BASE ** 2 - var_media), var_media

# ============================================================
# ACTUALIZACIÓN PRINCIPAL
# ============================================================
def actualizar_campo_v85(Phi_total, Phi_vel_total, W_prof, W_rec,
                          Phi_int_historia, historial_eficiencia,
                          objetivo_L, objetivo_R,
                          dt, fraccion_L, fraccion_R, sesgo_freq,
                          coherencia_rel):

    omega_nat, amort_nat = calcular_frecuencias_naturales(DIM_TOTAL)
    promedio_vecinos = calcular_promedio_vecinos_v85(Phi_total)
    difusion = DIFUSION_BASE * (promedio_vecinos - Phi_total)
    desviacion = Phi_total - promedio_vecinos
    reaccion = GANANCIA_REACCION * desviacion * (1 - desviacion**2)
    term_osc = (-omega_nat**2 * (Phi_total - PHI_EQUILIBRIO) - amort_nat * Phi_vel_total)

    M_campo = np.zeros_like(Phi_total)
    region_int = Phi_total[:DIM_INTERNA, :]
    region_aud = (Phi_total[idx['aud_L'][0]:idx['aud_L'][1], :] + Phi_total[idx['aud_R'][0]:idx['aud_R'][1], :]) / 2.0

    W_prof_n, W_rec_n, M_plast, error_rec, _ = aplicar_plasticidad_dual_v85(
        region_int, region_aud, W_prof, W_rec, Phi_int_historia, dt
    )
    M_campo[:DIM_INTERNA, :] = M_plast

    # Aplicar entrada cualitativa
    Phi_total = aplicar_entrada_cualitativa(Phi_total, objetivo_L, objetivo_R, fraccion_L, fraccion_R, sesgo_freq)

    # Aplicar orientación por coherencia
    Phi_total = aplicar_orientacion_por_coherencia(Phi_total, coherencia_rel, dt)

    # Actualizar act_busc desde coherencia (CAMBIO ARQUITECTÓNICO)
    Phi_total = actualizar_act_busc_desde_coherencia(Phi_total, coherencia_rel, dt)

    dPhi_vel = term_osc + reaccion + difusion + M_campo
    Phi_vel_n = Phi_vel_total + dt * dPhi_vel
    Phi_n = Phi_total + dt * Phi_vel_n

    varianza_int = np.var(Phi_n[:DIM_INTERNA, :])
    if varianza_int < DIFUSION_BASE * 1e-4:
        Phi_n[:DIM_INTERNA, :] += np.random.normal(0, 0.01, (DIM_INTERNA, DIM_TIME))

    lf_activa = error_rec > DIFUSION_BASE ** 2
    Phi_int_historia_n = (1 - 0.05) * Phi_int_historia + 0.05 * region_int

    return (np.clip(Phi_n, LIMITE_MIN, LIMITE_MAX),
            np.clip(Phi_vel_n, -5.0, 5.0),
            W_prof_n, W_rec_n, Phi_int_historia_n,
            lf_activa, error_rec)

# ============================================================
# SIMULACIÓN DE FASE
# ============================================================
def simular_fase(Phi_total, Phi_vel_total, W_prof, W_rec,
                  Phi_int_historia, historial_eficiencia, explorador,
                  canal_L, canal_R, sr, duracion, idx, dt, fase_nombre):

    ventana_muestras = int(sr * VENTANA_FFT_MS / 1000)
    hop_muestras = int(sr * HOP_FFT_MS / 1000)
    n_pasos = int(duracion / dt)

    metricas = []
    lf_hist = []
    w_rec_norma_hist = []
    w_prof_norma_hist = []
    act_busc_hist = []
    coherencia_hist = []

    for paso in range(n_pasos):
        obj_L = preparar_objetivo_para_canal(canal_L, sr, paso, ventana_muestras, hop_muestras, DIM_AUD, DIM_TIME)
        obj_R = preparar_objetivo_para_canal(canal_R, sr, paso, ventana_muestras, hop_muestras, DIM_AUD, DIM_TIME)

        # Calcular coherencia sobre objetivos crudos
        region_int = Phi_total[:DIM_INTERNA, :]
        coh_rel, err_L, err_R = calcular_coherencia_sobre_objetivos(obj_L, obj_R, W_prof, region_int)

        # Calcular parámetros de actuación
        fL, fR, sesgo, asim_geom, nivel_perm = calcular_parametros_actuacion(Phi_total)

        # Calcular eficiencia
        ged_actual = calcular_ged_entre(region_int, Phi_total[idx['aud_L'][0]:idx['aud_L'][1], :])
        eficiencia, _ = calcular_eficiencia_regimen(Phi_total, ged_actual)

        # Actualizar historial de eficiencia
        historial_eficiencia.append(eficiencia)
        if len(historial_eficiencia) > TAU_EFICIENCIA * 2:
            historial_eficiencia.pop(0)

        # Calcular tasa de olvido
        tasa_olvido, ventaja, _ = calcular_tasa_olvido_por_eficiencia(eficiencia, historial_eficiencia)

        # Actualizar campo (incluye act_busc desde coherencia)
        (Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia,
         lf_activa, error_rec) = actualizar_campo_v85(
            Phi_total, Phi_vel_total, W_prof, W_rec,
            Phi_int_historia, historial_eficiencia,
            obj_L, obj_R, dt, fL, fR, sesgo, coh_rel
        )

        # Actualizar explorador
        explorador.actualizar(lf_activa, eficiencia, fL, fR, sesgo)

        # Exploración activa
        Phi_total = explorar_actuadores(Phi_total, explorador, lf_activa, eficiencia, dt)

        # Métricas
        G = Phi_total[idx['G'][0]:idx['G'][1], :]
        G_act = float(np.mean(np.abs(G)))
        act_busc = calcular_senal_busqueda(Phi_total)
        asim_LR = calcular_senal_busqueda(Phi_total)  # placeholder
        geom_estado = float(np.mean(np.tanh(Phi_total[idx['act_geom'][0]:idx['act_geom'][1], :])))

        metricas.append({
            'ged_L': ged_actual,
            'ged_R': calcular_ged_entre(region_int, Phi_total[idx['aud_R'][0]:idx['aud_R'][1], :]),
            'act_busc': act_busc,
            'coherencia_rel': coh_rel,
            'geom_estado': geom_estado,
            'frac_L': fL,
            'frac_R': fR,
            'eficiencia': eficiencia,
            'lf_activa': lf_activa
        })

        lf_hist.append(lf_activa)
        w_rec_norma_hist.append(np.mean(np.abs(W_rec)))
        w_prof_norma_hist.append(np.mean(np.abs(W_prof)))
        act_busc_hist.append(act_busc)
        coherencia_hist.append(coh_rel)

        if paso % 200 == 0:
            print(f"    t={paso*dt:.1f}s | GED={ged_actual:.4f} | "
                  f"busc={act_busc:+.4f} | coh={coh_rel:+.4f} | "
                  f"G_act={G_act:.4f} | efic={eficiencia:.3f} | "
                  f"LF={'ACTIVA' if lf_activa else 'inact'}")

    # Calcular convergencia de act_busc
    n_half = n_pasos // 2
    act_busc_primera = np.mean(act_busc_hist[:n_half])
    act_busc_segunda = np.mean(act_busc_hist[n_half:])
    convergente = act_busc_primera * act_busc_segunda > 0

    return {
        'metricas': metricas,
        'lf_pct': 100 * np.mean(lf_hist),
        'w_rec_norma': np.mean(w_rec_norma_hist),
        'w_prof_norma': np.mean(w_prof_norma_hist),
        'act_busc_primera': act_busc_primera,
        'act_busc_segunda': act_busc_segunda,
        'act_busc_convergente': convergente,
        'mejor_eficiencia': explorador.mejor_eficiencia,
        'phi_total': Phi_total,
        'w_prof': W_prof,
        'w_rec': W_rec
    }

# ============================================================
# ENTRENAMIENTO INICIAL
# ============================================================
def entrenar_inicial(archivos, duracion=30.0):
    print("\n[Fase 1] Entrenamiento (voz +60°, alpha=0.05, 30s)")

    Phi_total, Phi_vel_total = inicializar_campo_v85()
    W_prof, W_rec, Phi_int_historia = inicializar_memorias()
    historial_eficiencia = []
    explorador = ExploradorActuadores()

    _, sr, canal_L, canal_R = archivos['voz_pos']

    ventana_muestras = int(sr * VENTANA_FFT_MS / 1000)
    hop_muestras = int(sr * HOP_FFT_MS / 1000)
    n_pasos = int(duracion / DT)

    for paso in range(n_pasos):
        obj_L = preparar_objetivo_para_canal(canal_L, sr, paso, ventana_muestras, hop_muestras, DIM_AUD, DIM_TIME)
        obj_R = preparar_objetivo_para_canal(canal_R, sr, paso, ventana_muestras, hop_muestras, DIM_AUD, DIM_TIME)

        coh_rel, _, _ = calcular_coherencia_sobre_objetivos(obj_L, obj_R, W_prof, Phi_total[:DIM_INTERNA, :])
        fL, fR, sesgo, _, _ = calcular_parametros_actuacion(Phi_total)

        (Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia,
         _, _) = actualizar_campo_v85(
            Phi_total, Phi_vel_total, W_prof, W_rec,
            Phi_int_historia, [], obj_L, obj_R,
            DT, fL, fR, sesgo, coh_rel
        )

        if paso % 500 == 0:
            ged = calcular_ged_entre(Phi_total[:DIM_INTERNA, :], Phi_total[idx['aud_L'][0]:idx['aud_L'][1], :])
            print(f"    Paso {paso}/{n_pasos}, GED={ged:.6f}")

    print(f"  W_prof tras entreno: {np.mean(np.abs(W_prof)):.4f}")
    print(f"  W_rec tras entreno: {np.mean(np.abs(W_rec)):.4f}")

    return Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia, explorador

# ============================================================
# MAIN
# ============================================================
def main():
    print("\n[Carga] Cargando archivos binaurales desde 'audio_binaural/'...")
    archivos = cargar_todos_binaurales('audio_binaural', 35.0)

    if not archivos:
        print("  ERROR: No se encontraron archivos binaurales.")
        print("  Ejecute primero v84b_campo_diferencial.py para generarlos.")
        return

    Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia, explorador = entrenar_inicial(archivos)

    protocolo = [
        ("Fase 2", "voz_pos", "Dominio (voz +60°)"),
        ("Fase 3", "musica_pos", "No entrenado (música +60°)"),
        ("Fase 4", "tono_pos", "No entrenado (tono +60°)"),
        ("Fase 5", "voz_viento_pos", "Degradado (voz+viento +60°)"),
        ("Fase 6", "ruido_pos", "Perturbación (ruido +60°)"),
        ("Fase 7", "voz_neg", "Re-acoplamiento opuesto (voz -60°)"),
    ]

    resultados = []
    historial_eficiencia_global = []

    for fase_id, clave, desc in protocolo:
        print(f"\n[{fase_id}] {desc}")
        _, sr, canal_L, canal_R = archivos[clave]

        res = simular_fase(
            Phi_total, Phi_vel_total, W_prof, W_rec,
            Phi_int_historia, historial_eficiencia_global, explorador,
            canal_L, canal_R, sr, 20.0, idx, DT, fase_id
        )

        resultados.append(res)
        Phi_total = res['phi_total']
        W_prof = res['w_prof']
        W_rec = res['w_rec']

        m = res['metricas']
        print(f"\n  Resumen {fase_id}:")
        print(f"    GED L/R:            {np.mean([mm['ged_L'] for mm in m]):.6f} / {np.mean([mm['ged_R'] for mm in m]):.6f}")
        print(f"    act_busc — primera mitad: {res['act_busc_primera']:+.4f}")
        print(f"    act_busc — segunda mitad: {res['act_busc_segunda']:+.4f}")
        print(f"    Convergencia:       {'✅ estable' if res['act_busc_convergente'] else '⚠️ oscilante'}")
        print(f"    Coherencia media:   {np.mean([mm['coherencia_rel'] for mm in m]):+.4f}")
        print(f"    act_geom estado:    {np.mean([mm['geom_estado'] for mm in m]):+.4f}")
        print(f"    Eficiencia media:   {np.mean([mm['eficiencia'] for mm in m]):.4f}")
        print(f"    LF activa:          {res['lf_pct']:.1f}%")
        print(f"    Mejor efic explorada: {res['mejor_eficiencia']:.4f}")

    print("\n" + "=" * 100)
    print("DIAGNÓSTICO — v85 Decisión interna")
    print("=" * 100)

    # Criterios
    act_busc_f2 = resultados[0]['act_busc_segunda']
    act_busc_f7 = resultados[5]['act_busc_segunda']
    c21 = act_busc_f2 * act_busc_f7 < 0

    c22_f2 = resultados[0]['act_busc_convergente']
    c22_f7 = resultados[5]['act_busc_convergente']
    c22 = c22_f2 and c22_f7

    coh_f2 = np.mean([m['coherencia_rel'] for m in resultados[0]['metricas']])
    coh_f7 = np.mean([m['coherencia_rel'] for m in resultados[5]['metricas']])
    c18 = coh_f2 * coh_f7 < 0

    mejor_efic = max([res['mejor_eficiencia'] for res in resultados])

    print("\n  CRITERIOS v85:")
    print(f"    C18 — Coherencia invierte:     {coh_f2:+.4f} → {coh_f7:+.4f} {'✅' if c18 else '❌'}")
    print(f"    C21 — act_busc signo opuesto:  F2={act_busc_f2:+.4f}, F7={act_busc_f7:+.4f} {'✅' if c21 else '❌'}")
    print(f"    C22 — act_busc converge:       F2={c22_f2}, F7={c22_f7} {'✅' if c22 else '❌'}")
    print(f"    Mejor eficiencia explorada:    {mejor_efic:.4f}")

    print("\n  VEREDICTO:")
    if c21 and c22:
        print("  ✅ DECISIÓN INTERNA VALIDADA")
        print("     act_busc tiene signos opuestos entre +60° y -60°.")
        print("     El signo es consistente dentro de cada fase.")
        print("     El campo pasó de detección diferencial a selección direccional.")
    elif c22:
        print("  ⚠️ act_busc CONVERGE pero no invierte signo.")
        print("     W_prof puede no ser suficientemente asimétrico.")
    elif c18:
        print("  ⚠️ COHERENCIA INVIERTE pero act_busc no.")
        print("     La integración temporal de act_busc necesita ajuste.")
    else:
        print("  ❌ DECISIÓN NO ALCANZADA")

    # Guardar CSV
    with open('v85_decision_interna.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['fase', 't', 'ged_L', 'ged_R', 'act_busc',
                         'coherencia_rel', 'geom_estado', 'frac_L', 'frac_R',
                         'eficiencia', 'lf_activa'])

        for fase_idx, (fase, res) in enumerate(zip(protocolo, resultados)):
            for t_idx, m in enumerate(res['metricas']):
                writer.writerow([
                    fase[0], t_idx * DT,
                    m['ged_L'], m['ged_R'], m['act_busc'],
                    m['coherencia_rel'], m['geom_estado'],
                    m['frac_L'], m['frac_R'],
                    m['eficiencia'], 1 if m['lf_activa'] else 0
                ])

    print("\n  CSV guardado: v85_decision_interna.csv")

    # Gráfico
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    nombres = ['F2', 'F3', 'F4', 'F5', 'F6', 'F7']

    # act_busc
    act_busc_vals = [res['act_busc_segunda'] for res in resultados]
    axes[0,0].bar(nombres, act_busc_vals)
    axes[0,0].axhline(y=0, color='r', linestyle='--')
    axes[0,0].set_title('act_busc (segunda mitad)')
    axes[0,0].grid(True, alpha=0.3)

    # Coherencia
    coh_vals = [np.mean([m['coherencia_rel'] for m in res['metricas']]) for res in resultados]
    axes[0,1].bar(nombres, coh_vals)
    axes[0,1].axhline(y=0, color='r', linestyle='--')
    axes[0,1].set_title('Coherencia relativa')
    axes[0,1].grid(True, alpha=0.3)

    # GED
    ged_L = [np.mean([m['ged_L'] for m in res['metricas']]) for res in resultados]
    ged_R = [np.mean([m['ged_R'] for m in res['metricas']]) for res in resultados]
    axes[0,2].plot(nombres, ged_L, 'o-', label='GED L')
    axes[0,2].plot(nombres, ged_R, 's-', label='GED R')
    axes[0,2].set_title('GED por canal')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)

    # act_geom
    geom = [np.mean([m['geom_estado'] for m in res['metricas']]) for res in resultados]
    axes[0,3].bar(nombres, geom)
    axes[0,3].set_title('act_geom estado')
    axes[0,3].grid(True, alpha=0.3)

    # Eficiencia
    efic = [np.mean([m['eficiencia'] for m in res['metricas']]) for res in resultados]
    axes[1,0].bar(nombres, efic)
    axes[1,0].set_title('Eficiencia de régimen')
    axes[1,0].grid(True, alpha=0.3)

    # LF activa
    lf_pct = [res['lf_pct'] for res in resultados]
    axes[1,1].bar(nombres, lf_pct)
    axes[1,1].set_title('LF activa (%)')
    axes[1,1].grid(True, alpha=0.3)

    # Mejor eficiencia explorada
    mejor_exp = [res['mejor_eficiencia'] for res in resultados]
    axes[1,2].bar(nombres, mejor_exp)
    axes[1,2].set_title('Mejor eficiencia explorada')
    axes[1,2].grid(True, alpha=0.3)

    # Convergencia
    convergencia = ['✅' if res['act_busc_convergente'] else '❌' for res in resultados]
    axes[1,3].bar(nombres, [1 if c else 0 for c in [res['act_busc_convergente'] for res in resultados]])
    axes[1,3].set_title('Convergencia de act_busc')
    axes[1,3].grid(True, alpha=0.3)

    plt.suptitle('VSTCosmos v85 — Decisión interna: act_busc como integrador de coherencia', fontsize=14)
    plt.tight_layout()
    plt.savefig('v85_decision_interna.png', dpi=150)
    print("  Gráfico guardado: v85_decision_interna.png")

    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)


if __name__ == "__main__":
    main()