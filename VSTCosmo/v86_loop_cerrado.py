#!/usr/bin/env python3
"""
VSTCosmos v86 — Loop cerrado: forzamiento dirigido act_busc → G

Cambios arquitectónicos:
1. Eliminada vecindad G ↔ act_busc (act_busc ya no recibe difusión)
2. act_busc integra coherencia con tanh amplificado
3. Forzamiento dirigido act_busc → G (sesgo sobre el ganglio)
4. El cierre sensoriomotor es completo: coherencia → act_busc → G → act_geom → filtrado
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

# Parámetros para act_busc
K_BUSC = T_PROFUNDA_SEG / T_RECIENTE_SEG  # = 10
DECAIMIENTO_ACT_BUSC = DT / T_RECIENTE_SEG  # = 0.005
EPSILON_BUSC_G = DIFUSION_BASE * K_BUSC * DT  # = 0.015

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

# Vecindades v86 — act_busc ya NO participa en la difusión
VECINDADES = [
    ('int', 'G'),
    ('G', 'aud_L'),
    ('G', 'aud_R'),
    ('G', 'act_perm'),
    ('G', 'act_geom'),
    # ('G', 'act_busc'),  ← ELIMINADA
    ('G', 'act_mant'),
    ('aud_L', 'aud_R'),
    ('act_perm', 'aud_L'),
    ('act_perm', 'aud_R'),
    ('act_geom', 'aud_L'),
    ('act_geom', 'aud_R'),
]

print("=" * 100)
print("VSTCosmos v86 — Loop cerrado: forzamiento dirigido act_busc → G")
print("")
print("  Cambios arquitectónicos:")
print("  1. Eliminada vecindad G ↔ act_busc (act_busc ya no recibe difusión)")
print("  2. act_busc integra coherencia con tanh amplificado")
print("  3. Forzamiento dirigido act_busc → G (sesgo sobre el ganglio)")
print("  4. El cierre sensoriomotor es completo:")
print("     coherencia → act_busc → G → act_geom → filtrado")
print("")
print("  Arquitectura del campo:")
print(f"    DIM_INTERNA = {DIM_INTERNA}")
print(f"    DIM_GANGLIO = {DIM_GANGLIO}")
print(f"    DIM_AUD = {DIM_AUD}")
print(f"    DIM_ACT = {DIM_ACT}")
print(f"    DIM_TOTAL = {DIM_TOTAL}")
print("=" * 100)

# ============================================================
# CARGA DE ARCHIVOS BINAURALES
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
        if len(canal_L) < n_muestras:
            canal_L = np.pad(canal_L, (0, n_muestras - len(canal_L)))
            canal_R = np.pad(canal_R, (0, n_muestras - len(canal_R)))
        return sr, canal_L, canal_R
    except Exception as e:
        print(f"  [ERROR] No se pudo cargar {filepath}: {e}")
        sr = 48000
        n_muestras = int(sr * duracion)
        t = np.arange(n_muestras) / sr
        senal = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        return sr, senal, senal

def cargar_todos_binaurales(directorio='audio_binaural', duracion=35.0):
    archivos = {}
    mapping = {
        'voz_pos':        'Voz_Estudio_pos60deg.wav',
        'voz_neg':        'Voz_Estudio_neg60deg.wav',
        'musica_pos':     'Brandemburgo_pos60deg.wav',
        'voz_viento_pos': 'Voz+Viento_1_pos60deg.wav',
        'tono_pos':       'Tono puro_pos60deg.wav',
        'ruido_pos':      'Ruido blanco_pos60deg.wav',
    }
    print(f"\n[Carga] Cargando archivos binaurales desde '{directorio}/'...")
    for clave, filename in mapping.items():
        filepath = os.path.join(directorio, filename)
        if os.path.exists(filepath):
            sr, canal_L, canal_R = cargar_audio_binaural(filepath, duracion)
            archivos[clave] = (filepath, sr, canal_L, canal_R)
            print(f"    ✅ {clave:20s} {filename}")
        else:
            print(f"    ❌ {clave:20s} {filename} no encontrado")
    return archivos

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
                (fraccion_L, fraccion_R, sesgo_freq, eficiencia_actual))
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
        perfil  += np.abs(fft)[:n_bins] ** 2
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
# PREPARAR OBJETIVO
# ============================================================
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
# COHERENCIA SOBRE OBJETIVOS CRUDOS
# ============================================================
def calcular_coherencia_sobre_objetivos(obj_L, obj_R, W_prof, region_int):
    n_prof = W_prof.shape[0]
    n_cols = W_prof.shape[1]
    n_int  = region_int.shape[0]

    perfil_L = obj_L.mean(axis=1)
    perfil_R = obj_R.mean(axis=1)
    perfil_i = region_int.mean(axis=1)

    min_c = min(n_cols, len(perfil_L), len(perfil_R))
    min_p = min(n_prof, n_int)

    W_t    = W_prof[:min_p, :min_c]
    pred_L = W_t @ perfil_L[:min_c].reshape(-1, 1)
    pred_R = W_t @ perfil_R[:min_c].reshape(-1, 1)
    ref    = perfil_i[:min_p].reshape(-1, 1)

    err_L  = float(np.mean((pred_L - ref) ** 2))
    err_R  = float(np.mean((pred_R - ref) ** 2))
    total  = err_L + err_R + 1e-10
    coh_rel = (err_R - err_L) / total

    return float(coh_rel), err_L, err_R

# ============================================================
# ACT_BUSC — integrador de coherencia (fuera del equilibrio difusivo)
# ============================================================
def actualizar_act_busc_desde_coherencia(Phi_total, coherencia_rel, dt):
    """
    act_busc integra coherencia_rel con amplificación no lineal tanh.
    No recibe difusión de ninguna región — fue eliminado de VECINDADES.

    La señal está centrada en PHI_EQUILIBRIO:
        señal = PHI_EQUILIBRIO + tanh(K_BUSC × coh) × DIFUSION_BASE

    Esto garantiza que act_busc converge hacia PHI_EQUILIBRIO ± perturbación,
    no hacia tanh(K×coh)×DIFUSION_BASE (valor absoluto ≈ 0.00045, que
    al centrar daría -0.4995 — incorrecto).

    Con K_BUSC=10 y coh=+0.0003: equilibrio = 0.5 + 0.00045 = 0.500450
    Con K_BUSC=10 y coh=-0.0002: equilibrio = 0.5 - 0.0003  = 0.499700
    Centrados: +0.000450 y -0.000300 → signos opuestos → C21 se cumple.
    """
    ab0, ab1 = idx['act_busc']
    señal = PHI_EQUILIBRIO + float(np.tanh(K_BUSC * coherencia_rel)) * DIFUSION_BASE
    Phi_total[ab0:ab1, :] = (
        (1.0 - DECAIMIENTO_ACT_BUSC) * Phi_total[ab0:ab1, :] +
        DECAIMIENTO_ACT_BUSC * señal
    )
    return Phi_total

# ============================================================
# FORZAMIENTO DIRIGIDO act_busc → G
# ============================================================
def aplicar_forzamiento_busc_a_ganglio(Phi_total, dt):
    """
    act_busc fuerza Φ_G con intensidad proporcional a su estado centrado.
    Acoplamiento dirigido — no es difusión recíproca.
    Sin este forzamiento, act_busc quedaría aislado y no influiría en nada.

    EPSILON_BUSC_G = DIFUSION_BASE × K_BUSC × dt = 0.015
    Derivado del mismo producto que K_ORIENT en act_geom.
    """
    ab0, ab1 = idx['act_busc']
    g0,  g1  = idx['G']

    estado_busc = float(np.mean(Phi_total[ab0:ab1, :])) - PHI_EQUILIBRIO
    n = min(ab1 - ab0, g1 - g0)
    Phi_total[g0:g0 + n, :] += EPSILON_BUSC_G * estado_busc
    return Phi_total

# ============================================================
# ORIENTACIÓN Y ACTUACIÓN
# ============================================================
def aplicar_orientacion_por_coherencia(Phi_total, coherencia_rel, dt):
    K_ORIENT = T_PROFUNDA_SEG / T_RECIENTE_SEG   # = 10
    acg0  = idx['act_geom'][0]
    acg1  = idx['act_geom'][1]
    mitad = max(1, (acg1 - acg0) // 2)
    señal = float(np.clip(
        coherencia_rel * DIFUSION_BASE * K_ORIENT * dt, -0.1, 0.1
    ))
    Phi_total[acg0:acg0 + mitad, :] += señal
    Phi_total[acg0 + mitad:acg1, :] -= señal
    return Phi_total

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
# EXPLORACIÓN ACTIVA
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
# PLASTICIDAD DUAL
# ============================================================
def aplicar_plasticidad_dual(region_int, region_aud, W_prof, W_rec,
                              Phi_int_historia, dt):
    min_prof = min(W_prof.shape[0], region_int.shape[0])
    min_cols = min(W_prof.shape[1], region_aud.shape[0])

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
# EFICIENCIA Y MÉTRICAS
# ============================================================
def calcular_eficiencia(Phi_total, ged_actual):
    region_int = Phi_total[idx['int'][0]:idx['int'][1], :]
    variacion  = float(np.mean(np.abs(np.diff(region_int, axis=1))))
    return ged_actual / (variacion + 1e-10), variacion

def calcular_senal_busqueda(Phi_total):
    ab0, ab1 = idx['act_busc']
    return float(np.mean(Phi_total[ab0:ab1, :])) - PHI_EQUILIBRIO

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

    Phi_total = aplicar_entrada_cualitativa(Phi_total, obj_L, obj_R,
                                            frac_L, frac_R, sesgo)
    Phi_total = aplicar_orientacion_por_coherencia(Phi_total, coherencia_rel, dt)

    dPhi_vel  = term_osc + reaccion + difusion + M_campo
    Phi_vel_n = Phi_vel_total + dt * dPhi_vel
    Phi_nueva = Phi_total + dt * Phi_vel_n

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
    print(f"\n[Fase 1] Entrenamiento (voz +60°, 30s)")

    Phi_total, Phi_vel_total = inicializar_campo()
    W_prof, W_rec, Phi_int_historia = inicializar_memorias()
    explorador = ExploradorActuadores()

    _, sr, c_L, c_R = archivos['voz_pos']
    vent    = int(sr * VENTANA_FFT_MS / 1000)
    hop     = int(sr * HOP_FFT_MS  / 1000)
    n_pasos = int(duracion / DT)
    errores = []

    for paso in range(n_pasos):
        obj_L = preparar_objetivo_canal(c_L, sr, paso, vent, hop, DIM_AUD, DIM_TIME)
        obj_R = preparar_objetivo_canal(c_R, sr, paso, vent, hop, DIM_AUD, DIM_TIME)

        region_int = Phi_total[idx['int'][0]:idx['int'][1], :]
        coh_rel, _, _ = calcular_coherencia_sobre_objetivos(
            obj_L, obj_R, W_prof, region_int
        )

        Phi_total = actualizar_act_busc_desde_coherencia(Phi_total, coh_rel, DT)
        Phi_total = aplicar_forzamiento_busc_a_ganglio(Phi_total, DT)

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
        'ged_L', 'ged_R', 'act_busc', 'coh_rel',
        'geom', 'frac_L', 'frac_R', 'efic', 'lf',
        'w_rec', 'w_prof', 'G_act'
    ]}

    lf_prev = False
    for paso in range(n_pasos):
        obj_L = preparar_objetivo_canal(canal_L, sr, paso, vent, hop, DIM_AUD, DIM_TIME)
        obj_R = preparar_objetivo_canal(canal_R, sr, paso, vent, hop, DIM_AUD, DIM_TIME)

        # 1. Coherencia sobre objetivos crudos
        region_int = Phi_total[idx['int'][0]:idx['int'][1], :]
        coh_rel, err_L, err_R = calcular_coherencia_sobre_objetivos(
            obj_L, obj_R, W_prof, region_int
        )

        # 2. act_busc integra coherencia (fuera del equilibrio difusivo)
        Phi_total = actualizar_act_busc_desde_coherencia(Phi_total, coh_rel, DT)

        # 3. Forzamiento dirigido act_busc → G
        Phi_total = aplicar_forzamiento_busc_a_ganglio(Phi_total, DT)

        # 4. Parámetros de actuación
        fL, fR, sf_v, asim, _ = calcular_parametros_actuacion(Phi_total)

        # 5. GED y eficiencia
        a_L   = Phi_total[idx['aud_L'][0]:idx['aud_L'][1], :]
        a_R   = Phi_total[idx['aud_R'][0]:idx['aud_R'][1], :]
        ged_L = calcular_ged_entre(region_int, a_L)
        ged_R = calcular_ged_entre(region_int, a_R)
        ged   = (ged_L + ged_R) / 2.0
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
        act_busc_val = calcular_senal_busqueda(Phi_total)
        geom         = float(np.mean(np.tanh(
            Phi_total[idx['act_geom'][0]:idx['act_geom'][1], :]
        )))
        G_act = float(np.mean(np.abs(
            Phi_total[idx['G'][0]:idx['G'][1], :]
        )))

        for k, v in [
            ('ged_L',   ged_L),
            ('ged_R',   ged_R),
            ('act_busc', act_busc_val),
            ('coh_rel', coh_rel),
            ('geom',    geom),
            ('frac_L',  fL),
            ('frac_R',  fR),
            ('efic',    efic),
            ('lf',      lf_activa),
            ('w_rec',   np.mean(np.abs(W_rec))),
            ('w_prof',  np.mean(np.abs(W_prof))),
            ('G_act',   G_act),
        ]:
            hist[k].append(v)

        if paso % 200 == 0:
            print(f"    t={paso*DT:.1f}s | GED={ged:.4f} | "
                  f"busc={act_busc_val:+.4f} | coh={coh_rel:+.4f} | "
                  f"geom={geom:+.4f} | G={G_act:.4f} | "
                  f"efic={efic:.3f} | LF={'ACTIVA' if lf_activa else 'inact'}")

    def M(k): return float(np.mean(hist[k]))
    n_half = n_pasos // 2

    geom_primera = float(np.mean(hist['geom'][:n_half]))
    geom_segunda = float(np.mean(hist['geom'][n_half:]))
    geom_conv    = geom_primera * geom_segunda > 0

    busc_segunda = float(np.mean(hist['act_busc'][n_half:]))

    print(f"\n  Resumen:")
    print(f"    GED L/R:                   {M('ged_L'):.4f} / {M('ged_R'):.4f}")
    print(f"    act_busc (segunda mitad):  {busc_segunda:+.4f}")
    print(f"    act_geom (segunda mitad):  {geom_segunda:+.4f}")
    print(f"    Convergencia geom:         {'✅ estable' if geom_conv else '⚠️ oscilante'}")
    print(f"    Coherencia media:          {M('coh_rel'):+.4f}")
    print(f"    Ganglio actividad media:   {M('G_act'):.4f}")
    print(f"    Eficiencia media:          {M('efic'):.4f}")
    print(f"    LF activa (%):             {100*M('lf'):.1f}%")
    print(f"    Mejor efic explorada:      {explorador.mejor_eficiencia:.4f}")

    return hist, geom_primera, geom_segunda, geom_conv, busc_segunda, \
           Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia, explorador

# ============================================================
# MAIN
# ============================================================
def main():
    archivos = cargar_todos_binaurales('audio_binaural', 35.0)
    if not archivos:
        print("\nERROR: No se encontraron archivos binaurales.")
        print("Ejecutar primero: py preprocesar_binaurales.py")
        return

    Phi_total, Phi_vel_total, W_prof, W_rec, \
        Phi_int_historia, ERROR_EQ, explorador = entrenar(archivos, 30.0)

    FASES = [
        ("F2", 'voz_pos',        +60.0, 20.0, "Dominio — voz +60°"),
        ("F3", 'musica_pos',     +60.0, 20.0, "No entrenado — música +60°"),
        ("F4", 'tono_pos',       +60.0, 20.0, "No entrenado — tono +60°"),
        ("F5", 'voz_viento_pos', +60.0, 20.0, "Degradado — voz+viento +60°"),
        ("F6", 'ruido_pos',      +60.0, 20.0, "Perturbación — ruido +60°"),
        ("F7", 'voz_neg',        -60.0, 20.0, "Re-acoplamiento opuesto — voz -60°"),
    ]

    resultados   = []
    historial_ef = []

    for fid, clave, angulo, dur, desc in FASES:
        print(f"\n[{fid}] {desc}")
        _, sr, c_L, c_R = archivos[clave]

        hist, geom_p, geom_s, geom_c, busc_s, \
            Phi_total, Phi_vel_total, W_prof, W_rec, \
            Phi_int_historia, explorador = simular_fase(
                Phi_total, Phi_vel_total, W_prof, W_rec,
                Phi_int_historia, historial_ef, explorador,
                sr, c_L, c_R, dur
            )
        resultados.append({
            'fid': fid, 'angulo': angulo, 'hist': hist,
            'geom_primera': geom_p, 'geom_segunda': geom_s,
            'geom_conv': geom_c, 'busc_segunda': busc_s,
            'mejor_ef': explorador.mejor_eficiencia
        })

    # ---- DIAGNÓSTICO ----
    print("\n" + "=" * 100)
    print("DIAGNÓSTICO — v86 Loop cerrado")
    print("=" * 100)

    def M(idx_fase, k):
        return float(np.mean(resultados[idx_fase]['hist'][k]))

    coh_f2  = M(0, 'coh_rel')
    coh_f7  = M(5, 'coh_rel')
    busc_f2 = resultados[0]['busc_segunda']
    busc_f7 = resultados[5]['busc_segunda']
    geom_f2 = resultados[0]['geom_segunda']
    geom_f7 = resultados[5]['geom_segunda']
    mejor_ef = resultados[-1]['mejor_ef']

    c18 = coh_f2 * coh_f7 < 0
    c21 = busc_f2 * busc_f7 < 0
    c22 = resultados[0]['geom_conv'] and resultados[5]['geom_conv']
    c23 = abs(geom_f2 - geom_f7) > 0.01

    def tick(b): return "✅" if b else "❌"

    print(f"\n  CRITERIOS v86:")
    print(f"    C18 — Coherencia invierte:     {tick(c18)} "
          f"(F2={coh_f2:+.4f}, F7={coh_f7:+.4f})")
    print(f"    C21 — act_busc invierte:       {tick(c21)} "
          f"(F2={busc_f2:+.4f}, F7={busc_f7:+.4f})")
    print(f"    C22 — act_geom converge:       {tick(c22)}")
    print(f"    C23 — act_geom diferencial:    {tick(c23)} "
          f"(F2={geom_f2:+.4f}, F7={geom_f7:+.4f})")
    print(f"    Mejor eficiencia explorada:    {mejor_ef:.4f}")

    print("\n  VEREDICTO:")
    if c23:
        print("  ✅ LOOP CERRADO VALIDADO")
        print("     act_geom se orienta de forma diferencial entre +60° y -60°.")
        print("     El forzamiento dirigido act_busc → G funciona.")
        print("     El cierre sensoriomotor está completo.")
        if c21:
            print("     ✅ act_busc también invierte signo.")
    elif c21:
        print("  ⚠️  act_busc INVIERTE pero act_geom aún no responde.")
        print("     El loop existe pero EPSILON_BUSC_G puede necesitar más ganancia.")
    elif c18:
        print("  ⚠️  COHERENCIA INVIERTE pero act_busc y act_geom no responden.")
        print("     Verificar forzamiento y vecindades.")
    else:
        print("  ❌ LOOP NO CERRADO.")

    # ---- CSV ----
    with open('v86_loop_cerrado.csv', 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(['fase', 'angulo', 't', 'ged_L', 'ged_R',
                     'act_busc', 'coh_rel', 'geom', 'frac_L', 'frac_R',
                     'efic', 'lf'])
        for r in resultados:
            h = r['hist']
            for i in range(len(h['ged_L'])):
                wr.writerow([
                    r['fid'], r['angulo'], round(i * DT, 2),
                    h['ged_L'][i], h['ged_R'][i],
                    h['act_busc'][i], h['coh_rel'][i],
                    h['geom'][i], h['frac_L'][i], h['frac_R'][i],
                    h['efic'][i], int(h['lf'][i])
                ])
    print("\n  CSV guardado: v86_loop_cerrado.csv")

    # ---- GRÁFICO ----
    nombres = [r['fid'] for r in resultados]
    fig, axes = plt.subplots(3, 4, figsize=(24, 18))

    # Fila 0
    vals_geom = [r['geom_segunda'] for r in resultados]
    c_geom = ['steelblue' if v >= 0 else 'firebrick' for v in vals_geom]
    axes[0,0].bar(nombres, vals_geom, color=c_geom)
    axes[0,0].axhline(0, color='k', linestyle='--', lw=0.8)
    axes[0,0].set_title('act_geom (segunda mitad)'); axes[0,0].grid(True, alpha=0.3)

    vals_busc = [r['busc_segunda'] for r in resultados]
    c_busc = ['steelblue' if v >= 0 else 'firebrick' for v in vals_busc]
    axes[0,1].bar(nombres, vals_busc, color=c_busc)
    axes[0,1].axhline(0, color='k', linestyle='--', lw=0.8)
    axes[0,1].set_title('act_busc (segunda mitad)'); axes[0,1].grid(True, alpha=0.3)

    vals_coh = [M(i, 'coh_rel') for i in range(6)]
    c_coh = ['steelblue' if v >= 0 else 'firebrick' for v in vals_coh]
    axes[0,2].bar(nombres, vals_coh, color=c_coh)
    axes[0,2].axhline(0, color='r', linestyle='--')
    axes[0,2].set_title('Coherencia relativa'); axes[0,2].grid(True, alpha=0.3)

    axes[0,3].plot(nombres, [M(i,'ged_L') for i in range(6)], 'o-', label='L')
    axes[0,3].plot(nombres, [M(i,'ged_R') for i in range(6)], 's-', label='R')
    axes[0,3].set_title('GED por canal'); axes[0,3].legend()
    axes[0,3].grid(True, alpha=0.3)

    # Fila 1: evolución temporal de act_busc y act_geom en F2 y F7
    # act_busc temporal
    t_ax = [i * DT for i in range(len(resultados[0]['hist']['act_busc']))]
    axes[1,0].plot(t_ax, resultados[0]['hist']['act_busc'],
                   color='steelblue', alpha=0.7, label='F2 +60°')
    axes[1,0].plot(t_ax, resultados[5]['hist']['act_busc'],
                   color='firebrick', alpha=0.7, label='F7 -60°')
    axes[1,0].axhline(0, color='k', linestyle='--', lw=0.8)
    axes[1,0].set_title('act_busc temporal F2 vs F7')
    axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

    # act_geom temporal
    axes[1,1].plot(t_ax, resultados[0]['hist']['geom'],
                   color='steelblue', alpha=0.7, label='F2 +60°')
    axes[1,1].plot(t_ax, resultados[5]['hist']['geom'],
                   color='firebrick', alpha=0.7, label='F7 -60°')
    axes[1,1].axhline(0, color='k', linestyle='--', lw=0.8)
    axes[1,1].set_title('act_geom temporal F2 vs F7')
    axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

    axes[1,2].bar(nombres, [M(i,'G_act') for i in range(6)])
    axes[1,2].set_title('Ganglio actividad media'); axes[1,2].grid(True, alpha=0.3)

    axes[1,3].bar(nombres, [M(i,'efic') for i in range(6)])
    axes[1,3].set_title('Eficiencia de régimen'); axes[1,3].grid(True, alpha=0.3)

    # Fila 2
    axes[2,0].bar(nombres, [100*M(i,'lf') for i in range(6)])
    axes[2,0].set_title('LF activa (%)'); axes[2,0].grid(True, alpha=0.3)

    axes[2,1].plot(nombres, [M(i,'w_rec') for i in range(6)], 'o-', label='W_rec')
    axes[2,1].plot(nombres, [M(i,'w_prof') for i in range(6)], 's-', label='W_prof')
    axes[2,1].set_title('Norma W'); axes[2,1].legend(); axes[2,1].grid(True, alpha=0.3)

    axes[2,2].plot(nombres, [M(i,'frac_L') for i in range(6)], 'o-', label='L')
    axes[2,2].plot(nombres, [M(i,'frac_R') for i in range(6)], 's-', label='R')
    axes[2,2].set_title('Fracción L/R'); axes[2,2].legend(); axes[2,2].grid(True, alpha=0.3)

    axes[2,3].bar(nombres, [r['mejor_ef'] for r in resultados])
    axes[2,3].set_title('Mejor eficiencia explorada'); axes[2,3].grid(True, alpha=0.3)

    plt.suptitle(
        'VSTCosmos v86 — Loop cerrado: forzamiento dirigido act_busc → G',
        fontsize=13
    )
    plt.tight_layout()
    plt.savefig('v86_loop_cerrado.png', dpi=150)
    print("  Gráfico guardado: v86_loop_cerrado.png")

    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)


if __name__ == "__main__":
    main()