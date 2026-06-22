#!/usr/bin/env python3
"""
VSTCosmos v87 — Coherencia espectral dirigida y protocolo completo

Cambios respecto a v86:
1. Coherencia calculada solo en bandas altas (donde existe ILD binaural)
2. Protocolo extendido con 13 fases — todos los estímulos disponibles
3. act_busc mantiene fix de centrado en PHI_EQUILIBRIO (de v86 corregido)

Correcciones respecto al código de DeepSeek:
- BANDA_TRANS movido después de DIM_AUD (orden de definición)
- actualizar_act_busc_desde_coherencia: señal centrada en PHI_EQUILIBRIO
- Diagnóstico robusto a archivos faltantes (índices por nombre, no posición)
- Condición de completitud no bloquea diagnóstico parcial
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import csv
import warnings
import os
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

T_PROFUNDA_SEG   = 1.0 / OMEGA_MIN
T_RECIENTE_SEG   = 1.0 / OMEGA_MAX
T_PROFUNDA_PASOS = int(T_PROFUNDA_SEG / 0.01)
T_RECIENTE_PASOS = int(T_RECIENTE_SEG / 0.01)

ETA_PROFUNDA_BASE = (1.0 / T_PROFUNDA_PASOS) / DIFUSION_BASE
ETA_RECIENTE_BASE = (1.0 / T_RECIENTE_PASOS) / DIFUSION_BASE
TAU_PROFUNDA      = OMEGA_MIN
TAU_RECIENTE      = OMEGA_MIN * 0.5
TAU_EFICIENCIA    = int(1.0 / (OMEGA_MIN * 0.01))
TAU_EXPLORACION   = int(T_RECIENTE_SEG / 0.01)

LIMITE_MIN  = 0.0
LIMITE_MAX  = 1.0
W_MAX       = 1.0
ALPHA_FIJO  = 0.05
DT          = 0.01
DIM_TIME    = 100

# Constantes físicas para binaural
DIAMETRO_CABEZA  = 0.175
VELOCIDAD_SONIDO = 343.0
ITD_MAX_SEG      = DIAMETRO_CABEZA / VELOCIDAD_SONIDO
F_TRANS_HZ       = VELOCIDAD_SONIDO / DIAMETRO_CABEZA   # ≈ 1960 Hz

# ============================================================
# ARQUITECTURA DEL CAMPO EXPANDIDO
# ============================================================
DIM_GANGLIO  = DIM_INTERNA // 2   # 16
DIM_AUD      = DIM_GANGLIO        # 16   ← BANDA_TRANS se calcula después de este
DIM_ACT      = DIM_GANGLIO // 2   # 8

DIM_AUD_L    = DIM_AUD
DIM_AUD_R    = DIM_AUD
DIM_ACT_PERM = DIM_ACT
DIM_ACT_GEOM = DIM_ACT
DIM_ACT_BUSC = DIM_ACT
DIM_ACT_MANT = DIM_ACT

# BANDA_TRANS: aquí, después de DIM_AUD
# Banda en la que F_TRANS_HZ cae dentro del espectro logarítmico [F_MIN, F_MAX]
BANDA_TRANS = int(DIM_AUD * np.log10(F_TRANS_HZ / F_MIN)
                  / np.log10(F_MAX / F_MIN))
BANDA_TRANS = max(1, min(BANDA_TRANS, DIM_AUD - 1))   # = 11 con F_MIN=80, F_MAX=8000

# Parámetros para act_busc
K_BUSC              = T_PROFUNDA_SEG / T_RECIENTE_SEG   # = 10
DECAIMIENTO_ACT_BUSC = DT / T_RECIENTE_SEG             # = 0.005
EPSILON_BUSC_G      = DIFUSION_BASE * K_BUSC * DT       # = 0.015

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

# Topología v87 — igual que v86: act_busc fuera del equilibrio difusivo
VECINDADES = [
    ('int',      'G'),
    ('G',        'aud_L'),
    ('G',        'aud_R'),
    ('G',        'act_perm'),
    ('G',        'act_geom'),
    # ('G',      'act_busc'),  ← ELIMINADA desde v86
    ('G',        'act_mant'),
    ('aud_L',    'aud_R'),
    ('act_perm', 'aud_L'),
    ('act_perm', 'aud_R'),
    ('act_geom', 'aud_L'),
    ('act_geom', 'aud_R'),
]

print("=" * 100)
print("VSTCosmos v87 — Coherencia espectral dirigida y protocolo completo")
print("")
print("  Cambios respecto a v86:")
print(f"  1. Coherencia en bandas altas únicamente "
      f"(banda {BANDA_TRANS}-{DIM_AUD-1}, F>{F_TRANS_HZ:.0f}Hz)")
print("  2. Protocolo extendido con 13 fases — todos los estímulos disponibles")
print("  3. act_busc centrado en PHI_EQUILIBRIO (fix de v86)")
print("")
print(f"  BANDA_TRANS = {BANDA_TRANS} → bandas ILD = {DIM_AUD - BANDA_TRANS} "
      f"de {DIM_AUD} totales")
print(f"  K_BUSC = {K_BUSC}, EPSILON_BUSC_G = {EPSILON_BUSC_G}")
print(f"  DIM_TOTAL = {DIM_TOTAL}")
print("=" * 100)

# ============================================================
# CARGA DE ARCHIVOS BINAURALES
# ============================================================
def cargar_todos_binaurales(directorio='audio_binaural', duracion=35.0):
    """
    Carga todos los archivos preprocesados.
    BigBang se carga completo (sin límite de duración).
    """
    mapping = {
        'voz_pos':         'Voz_Estudio_pos60deg.wav',
        'voz_neg':         'Voz_Estudio_neg60deg.wav',
        'musica_pos':      'Brandemburgo_pos60deg.wav',
        'voz_viento1_pos': 'Voz+Viento_1_pos60deg.wav',
        'voz_viento2_pos': 'Voz+Viento_2_pos60deg.wav',
        'tono_pos':        'Tono puro_pos60deg.wav',
        'ruido_pos':       'Ruido blanco_pos60deg.wav',
        'ritmos_pos':      'Ritmos aleatorios_pos60deg.wav',
        'ondas_pos':       'Ondas mixtas_pos60deg.wav',
        'pulso_pos':       'Pulso logaritmico_pos60deg.wav',
        'viento_pos':      'Viento_pos60deg.wav',
        'bigbang_pos':     'BigBang_pos60deg.wav',
    }

    archivos = {}
    print(f"\n[Carga] Desde '{directorio}/'...")
    for clave, filename in mapping.items():
        filepath = os.path.join(directorio, filename)
        if not os.path.exists(filepath):
            print(f"    ⚠️  {clave:22s} no encontrado")
            continue
        try:
            data, sr = sf.read(filepath, dtype='float32')
            # BigBang: duración completa; resto: truncar a duracion
            if clave != 'bigbang_pos' and duracion is not None:
                n = int(sr * duracion)
                if data.ndim == 1:
                    data = data[:n]
                    if len(data) < n:
                        data = np.pad(data, (0, n - len(data)))
                else:
                    data = data[:n, :]
                    if data.shape[0] < n:
                        data = np.pad(data, ((0, n - data.shape[0]), (0, 0)))

            if data.ndim == 1:
                canal_L = data
                canal_R = data.copy()
            else:
                canal_L = data[:, 0]
                canal_R = data[:, 1] if data.shape[1] > 1 else data[:, 0].copy()

            dur_real = len(canal_L) / sr
            archivos[clave] = (filepath, sr, canal_L, canal_R)
            print(f"    ✅ {clave:22s} {filename} ({dur_real:.1f}s)")
        except Exception as e:
            print(f"    ❌ {clave:22s} {e}")

    print(f"  Carga completada: {len(archivos)} archivos.")
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

    def actualizar(self, lf_activa, efic, fL, fR, sesgo):
        if lf_activa:
            self.pasos_en_lf += 1
            self.historial.append((fL, fR, sesgo, efic))
            if efic > self.mejor_eficiencia:
                self.mejor_eficiencia = efic
                self.mejor_config = (fL, fR, sesgo)
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
# COHERENCIA ESPECTRAL DIRIGIDA (cambio central de v87)
# ============================================================
def calcular_coherencia_dirigida(obj_L, obj_R, W_prof, region_int):
    """
    Coherencia calculada solo sobre las bandas frecuenciales altas
    (banda BANDA_TRANS en adelante) donde el ILD produce diferencia
    real entre canales L y R.

    Las bandas bajas (0 a BANDA_TRANS-1) no tienen ILD binaural
    y diluyen la señal cuando se incluyen. Excluirlas amplifica
    la coherencia estructuralmente — sin parámetros arbitrarios.

    BANDA_TRANS deriva de F_TRANS_HZ = VELOCIDAD_SONIDO/DIAMETRO_CABEZA.
    """
    n_prof = W_prof.shape[0]
    n_cols = W_prof.shape[1]
    n_int  = region_int.shape[0]

    # Solo bandas altas (donde existe ILD)
    n_altas = DIM_AUD - BANDA_TRANS
    if n_altas <= 0:
        return 0.0, 0.0, 0.0

    perfil_L_alto = obj_L[BANDA_TRANS:, :].mean(axis=1)   # (n_altas,)
    perfil_R_alto = obj_R[BANDA_TRANS:, :].mean(axis=1)
    perfil_i      = region_int.mean(axis=1)                # (DIM_INTERNA,)

    # W_prof columnas BANDA_TRANS:DIM_AUD corresponden a bandas altas
    min_c = min(n_cols - BANDA_TRANS, n_altas)
    min_p = min(n_prof, n_int)

    if min_c <= 0 or min_p <= 0:
        return 0.0, 0.0, 0.0

    W_alto = W_prof[:min_p, BANDA_TRANS:BANDA_TRANS + min_c]

    pred_L = W_alto @ perfil_L_alto[:min_c].reshape(-1, 1)
    pred_R = W_alto @ perfil_R_alto[:min_c].reshape(-1, 1)
    ref    = perfil_i[:min_p].reshape(-1, 1)

    err_L   = float(np.mean((pred_L - ref) ** 2))
    err_R   = float(np.mean((pred_R - ref) ** 2))
    total   = err_L + err_R + 1e-10
    coh_rel = (err_R - err_L) / total

    return float(coh_rel), err_L, err_R

# ============================================================
# ACT_BUSC — integrador de coherencia (fuera del equilibrio difusivo)
# ============================================================
def actualizar_act_busc_desde_coherencia(Phi_total, coherencia_rel, dt):
    """
    Señal centrada en PHI_EQUILIBRIO para que act_busc oscile
    alrededor de 0.5 y el signo centrado refleje la dirección.

    señal = PHI_EQUILIBRIO + tanh(K_BUSC × coh) × DIFUSION_BASE

    Con coh=+0.005: señal ≈ 0.5 + 0.0075 = 0.5075 → centrado = +0.0075
    Con coh=-0.005: señal ≈ 0.5 - 0.0075 = 0.4925 → centrado = -0.0075
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
    Acoplamiento dirigido — no difusión recíproca.
    El estado diferencial de act_busc (centrado) sesga Φ_G.
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
    """Estado de act_busc centrado en PHI_EQUILIBRIO."""
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
    print(f"\n[Fase 1] Entrenamiento (voz +60°, {duracion}s)")

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
        coh_rel, _, _ = calcular_coherencia_dirigida(obj_L, obj_R, W_prof, region_int)

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

    print(f"  ERROR_EQUILIBRIO: {min(errores):.6f}")
    print(f"  W_prof: {np.mean(np.abs(W_prof)):.4f}")
    print(f"  W_rec:  {np.mean(np.abs(W_rec)):.4f}")

    return Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia, explorador

# ============================================================
# SIMULACIÓN DE FASE
# ============================================================
def simular_fase(Phi_total, Phi_vel_total, W_prof, W_rec,
                 Phi_int_historia, historial_ef, explorador,
                 sr, canal_L, canal_R, duracion):

    vent    = int(sr * VENTANA_FFT_MS / 1000)
    hop     = int(sr * HOP_FFT_MS  / 1000)

    if duracion is None:
        n_pasos_max = len(canal_L) // hop
    else:
        n_pasos_max = int(duracion / DT)

    # Límite de cómputo para archivos muy largos (BigBang: ~23000 pasos)
    MAX_PASOS = 5000
    n_pasos = min(n_pasos_max, MAX_PASOS)

    hist = {k: [] for k in [
        'ged_L', 'ged_R', 'act_busc', 'coh_rel', 'geom',
        'frac_L', 'frac_R', 'efic', 'lf', 'w_rec', 'w_prof', 'G_act'
    ]}

    lf_prev = False
    for paso in range(n_pasos):
        obj_L = preparar_objetivo_canal(canal_L, sr, paso, vent, hop, DIM_AUD, DIM_TIME)
        obj_R = preparar_objetivo_canal(canal_R, sr, paso, vent, hop, DIM_AUD, DIM_TIME)

        region_int = Phi_total[idx['int'][0]:idx['int'][1], :]
        coh_rel, err_L, err_R = calcular_coherencia_dirigida(
            obj_L, obj_R, W_prof, region_int
        )

        Phi_total = actualizar_act_busc_desde_coherencia(Phi_total, coh_rel, DT)
        Phi_total = aplicar_forzamiento_busc_a_ganglio(Phi_total, DT)

        fL, fR, sf_v, asim, _ = calcular_parametros_actuacion(Phi_total)

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

        Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia, \
            lf_activa, error_rec, _ = actualizar_campo(
                Phi_total, Phi_vel_total, W_prof, W_rec,
                Phi_int_historia, obj_L, obj_R,
                fL, fR, sf_v, coh_rel, DT
            )

        Phi_total = explorar_actuadores(Phi_total, explorador, lf_activa, efic, DT)
        lf_prev   = lf_activa

        act_busc_val = calcular_senal_busqueda(Phi_total)
        geom         = float(np.mean(np.tanh(
            Phi_total[idx['act_geom'][0]:idx['act_geom'][1], :]
        )))
        G_act = float(np.mean(np.abs(
            Phi_total[idx['G'][0]:idx['G'][1], :]
        )))

        for k, v in [
            ('ged_L',   ged_L), ('ged_R',   ged_R),
            ('act_busc', act_busc_val), ('coh_rel', coh_rel),
            ('geom',    geom), ('frac_L',  fL), ('frac_R',  fR),
            ('efic',    efic), ('lf',      lf_activa),
            ('w_rec',   np.mean(np.abs(W_rec))),
            ('w_prof',  np.mean(np.abs(W_prof))),
            ('G_act',   G_act),
        ]:
            hist[k].append(v)

        if paso % 200 == 0:
            print(f"    t={paso*DT:.1f}s | GED={ged:.4f} | "
                  f"busc={act_busc_val:+.5f} | coh={coh_rel:+.5f} | "
                  f"geom={geom:+.4f} | G={G_act:.4f} | "
                  f"efic={efic:.3f} | LF={'ACTIVA' if lf_activa else 'inact'}")

    def M(k): return float(np.mean(hist[k])) if hist[k] else 0.0
    n_half = len(hist['geom']) // 2

    geom_primera = float(np.mean(hist['geom'][:n_half])) if n_half > 0 else 0.0
    geom_segunda = float(np.mean(hist['geom'][n_half:])) if n_half > 0 else 0.0
    geom_conv    = (geom_primera * geom_segunda > 0) if n_half > 0 else False
    busc_segunda = float(np.mean(hist['act_busc'][n_half:])) if n_half > 0 else 0.0

    print(f"\n  Resumen:")
    print(f"    GED L/R:                  {M('ged_L'):.4f} / {M('ged_R'):.4f}")
    print(f"    act_busc (2ª mitad):      {busc_segunda:+.5f}")
    print(f"    act_geom (2ª mitad):      {geom_segunda:+.4f}")
    print(f"    Convergencia geom:        {'✅ estable' if geom_conv else '⚠️ oscilante'}")
    print(f"    Coherencia dirigida:      {M('coh_rel'):+.5f}")
    print(f"    Eficiencia media:         {M('efic'):.4f}")
    print(f"    LF activa (%):            {100*M('lf'):.1f}%")
    print(f"    Mejor efic explorada:     {explorador.mejor_eficiencia:.4f}")

    return {
        'hist': hist,
        'geom_primera': geom_primera, 'geom_segunda': geom_segunda,
        'geom_conv': geom_conv, 'busc_segunda': busc_segunda,
        'coh_media': M('coh_rel'), 'mejor_ef': explorador.mejor_eficiencia,
        'phi_total': Phi_total, 'phi_vel': Phi_vel_total,
        'W_prof': W_prof, 'W_rec': W_rec,
        'Phi_int_historia': Phi_int_historia,
    }

# ============================================================
# MAIN
# ============================================================
def main():
    archivos = cargar_todos_binaurales('audio_binaural', 35.0)
    if not archivos:
        print("\nERROR: No se encontraron archivos. Ejecutar preprocesar_binaurales.py")
        return

    Phi_total, Phi_vel_total, W_prof, W_rec, \
        Phi_int_historia, explorador = entrenar(archivos, 30.0)

    FASES = [
        ("F2",  'voz_pos',         +60.0, 20.0, "Dominio — voz +60°"),
        ("F3",  'musica_pos',      +60.0, 20.0, "No entrenado — música +60°"),
        ("F4",  'tono_pos',        +60.0, 20.0, "No entrenado — tono +60°"),
        ("F5",  'ritmos_pos',      +60.0, 20.0, "Ritmos irregulares +60°"),
        ("F6",  'ondas_pos',       +60.0, 20.0, "Ondas mixtas +60°"),
        ("F7",  'pulso_pos',       +60.0, 20.0, "Pulso logarítmico +60°"),
        ("F8",  'viento_pos',      +60.0, 20.0, "Viento +60°"),
        ("F9",  'voz_viento1_pos', +60.0, 20.0, "Degradado — voz+viento1 +60°"),
        ("F10", 'voz_viento2_pos', +60.0, 20.0, "Degradado — voz+viento2 +60°"),
        ("F11", 'ruido_pos',       +60.0, 20.0, "Perturbación basal — ruido +60°"),
        ("F12", 'bigbang_pos',     +60.0, None, "Poesía musicalizada — BigBang +60°"),
        ("F13", 'voz_neg',         -60.0, 20.0, "Re-acoplamiento opuesto — voz -60°"),
    ]

    # Resultados indexados por id de fase para acceso robusto
    resultados = {}   # fid → (angulo, res)
    historial_ef = []

    for fid, clave, angulo, dur, desc in FASES:
        if clave not in archivos:
            print(f"\n[{fid}] {desc} — archivo no disponible, omitido")
            continue

        print(f"\n[{fid}] {desc}")
        _, sr, c_L, c_R = archivos[clave]

        res = simular_fase(
            Phi_total, Phi_vel_total, W_prof, W_rec,
            Phi_int_historia, historial_ef, explorador,
            sr, c_L, c_R, dur
        )
        resultados[fid] = (angulo, res)

        Phi_total    = res['phi_total']
        Phi_vel_total = res['phi_vel']
        W_prof       = res['W_prof']
        W_rec        = res['W_rec']
        Phi_int_historia = res['Phi_int_historia']

    # ---- DIAGNÓSTICO (robusto a archivos faltantes) ----
    print("\n" + "=" * 100)
    print("DIAGNÓSTICO — v87 Coherencia espectral dirigida")
    print("=" * 100)

    def get_coh(fid):
        return resultados[fid][1]['coh_media'] if fid in resultados else None
    def get_busc(fid):
        return resultados[fid][1]['busc_segunda'] if fid in resultados else None
    def get_geom(fid):
        return resultados[fid][1]['geom_segunda'] if fid in resultados else None

    coh_f2   = get_coh('F2')
    coh_f13  = get_coh('F13')
    busc_f2  = get_busc('F2')
    busc_f13 = get_busc('F13')
    geom_f2  = get_geom('F2')
    geom_f13 = get_geom('F13')
    coh_f11  = get_coh('F11')   # ruido
    coh_f12  = get_coh('F12')   # bigbang
    coh_f7   = get_coh('F7')    # pulso

    def tick(b): return "✅" if b else "❌"

    print(f"\n  BANDA_TRANS = {BANDA_TRANS} (F>{F_TRANS_HZ:.0f}Hz, "
          f"{DIM_AUD - BANDA_TRANS} bandas ILD de {DIM_AUD})")
    print()

    c18 = (coh_f2 is not None and coh_f13 is not None
           and coh_f2 * coh_f13 < 0)
    c21 = (busc_f2 is not None and busc_f13 is not None
           and busc_f2 * busc_f13 < 0)
    c23 = (geom_f2 is not None and geom_f13 is not None
           and abs(geom_f2 - geom_f13) > 0.01)
    c24 = coh_f2 is not None and abs(coh_f2) > 0.001

    coh_f11_str = f"{coh_f11:+.5f}" if coh_f11 is not None else "N/A"
    coh_f12_str = f"{coh_f12:+.5f}" if coh_f12 is not None else "N/A"
    coh_f7_str  = f"{coh_f7:+.5f}"  if coh_f7  is not None else "N/A"

    c25 = (coh_f11 is not None and coh_f12 is not None
           and abs(coh_f12 - coh_f11) > 0.005)

    todas_coh = [v[1]['coh_media'] for v in resultados.values()
                 if v[0] > 0]   # solo fases con ángulo positivo
    c26 = (coh_f7 is not None and todas_coh
           and abs(coh_f7) >= max(abs(c) for c in todas_coh))

    mejor_ef = max([v[1]['mejor_ef'] for v in resultados.values()],
                   default=0.0)

    coh_f2_str  = f"{coh_f2:+.5f}"  if coh_f2  is not None else "N/A"
    coh_f13_str = f"{coh_f13:+.5f}" if coh_f13 is not None else "N/A"
    busc_f2_str  = f"{busc_f2:+.6f}"  if busc_f2  is not None else "N/A"
    busc_f13_str = f"{busc_f13:+.6f}" if busc_f13 is not None else "N/A"
    geom_f2_str  = f"{geom_f2:+.4f}"  if geom_f2  is not None else "N/A"
    geom_f13_str = f"{geom_f13:+.4f}" if geom_f13 is not None else "N/A"

    print(f"  CRITERIOS v87:")
    print(f"    C18 — Coherencia invierte F2/F13:  "
          f"{tick(c18)} (F2={coh_f2_str}, F13={coh_f13_str})")
    print(f"    C21 — act_busc invierte F2/F13:    "
          f"{tick(c21)} (F2={busc_f2_str}, F13={busc_f13_str})")
    print(f"    C23 — act_geom diferencial:        "
          f"{tick(c23)} (F2={geom_f2_str}, F13={geom_f13_str})")
    print(f"    C24 — Coherencia amplificada>0.001: "
          f"{tick(c24)} ({coh_f2_str})")
    print(f"    C25 — BigBang ≠ ruido:             "
          f"{tick(c25)} (BigBang={coh_f12_str}, ruido={coh_f11_str})")
    print(f"    C26 — Pulso max coherencia:        "
          f"{tick(c26)} (pulso={coh_f7_str})")
    print(f"    Mejor eficiencia explorada:        {mejor_ef:.4f}")

    print("\n  VEREDICTO:")
    if c24 and c21 and c23:
        print("  ✅ ORIENTACIÓN ESPACIAL VALIDADA")
        print("     La coherencia dirigida es suficientemente fuerte.")
        print("     act_busc y act_geom responden a la dirección de la fuente.")
        print("     El campo distingue espacialmente +60° de -60°.")
    elif c24 and c18:
        print("  ⚠️  COHERENCIA AMPLIFICADA Y DIRIGIDA — act_geom aún no responde.")
        print("     La señal es más fuerte pero no alcanza umbral de orientación.")
    elif c24:
        print("  ⚠️  COHERENCIA AMPLIFICADA — inversión de signo no lograda.")
    elif c18:
        print("  ⚠️  Coherencia invierte pero señal insuficiente (C24 falla).")
        print("     BANDA_TRANS puede necesitar ajuste.")
    else:
        print("  ❌ ORIENTACIÓN NO LOGRADA.")

    # ---- CSV ----
    with open('v87_coherencia_dirigida.csv', 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(['fase', 'angulo', 't', 'ged_L', 'ged_R',
                     'act_busc', 'coh_rel', 'geom', 'frac_L', 'frac_R',
                     'efic', 'lf'])
        for fid, (angulo, res) in resultados.items():
            h = res['hist']
            for i in range(len(h['ged_L'])):
                wr.writerow([
                    fid, angulo, round(i * DT, 2),
                    h['ged_L'][i], h['ged_R'][i], h['act_busc'][i],
                    h['coh_rel'][i], h['geom'][i],
                    h['frac_L'][i], h['frac_R'][i],
                    h['efic'][i], int(h['lf'][i])
                ])
    print("\n  CSV guardado: v87_coherencia_dirigida.csv")

    # ---- GRÁFICO ----
    fids_ord = [fid for fid, _, _, _, _ in FASES if fid in resultados]
    n_fases  = len(fids_ord)

    fig, axes = plt.subplots(3, 4, figsize=(24, 18))

    vals_coh  = [resultados[f][1]['coh_media']    for f in fids_ord]
    vals_busc = [resultados[f][1]['busc_segunda']  for f in fids_ord]
    vals_geom = [resultados[f][1]['geom_segunda']  for f in fids_ord]
    vals_efic = [float(np.mean(resultados[f][1]['hist']['efic'])) for f in fids_ord]
    vals_lf   = [float(np.mean(resultados[f][1]['hist']['lf']))*100 for f in fids_ord]
    vals_wrec = [float(np.mean(resultados[f][1]['hist']['w_rec'])) for f in fids_ord]
    vals_mej  = [resultados[f][1]['mejor_ef'] for f in fids_ord]

    def bar_colored(ax, names, vals, title):
        colors = ['steelblue' if v >= 0 else 'firebrick' for v in vals]
        ax.bar(names, vals, color=colors)
        ax.axhline(0, color='k', linestyle='--', lw=0.8)
        ax.set_title(title); ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

    bar_colored(axes[0,0], fids_ord, vals_coh,  'Coherencia dirigida (bandas altas)')
    bar_colored(axes[0,1], fids_ord, vals_busc, 'act_busc (2ª mitad)')
    bar_colored(axes[0,2], fids_ord, vals_geom, 'act_geom (2ª mitad)')

    ged_L_vals = [float(np.mean(resultados[f][1]['hist']['ged_L'])) for f in fids_ord]
    ged_R_vals = [float(np.mean(resultados[f][1]['hist']['ged_R'])) for f in fids_ord]
    axes[0,3].plot(fids_ord, ged_L_vals, 'o-', label='L')
    axes[0,3].plot(fids_ord, ged_R_vals, 's-', label='R')
    axes[0,3].set_title('GED por canal'); axes[0,3].legend()
    axes[0,3].grid(True, alpha=0.3)
    axes[0,3].tick_params(axis='x', rotation=45)

    # Evolución temporal act_busc F2 vs F13
    if 'F2' in resultados and 'F13' in resultados:
        t_f2  = [i*DT for i in range(len(resultados['F2'][1]['hist']['act_busc']))]
        t_f13 = [i*DT for i in range(len(resultados['F13'][1]['hist']['act_busc']))]
        axes[1,0].plot(t_f2,  resultados['F2'][1]['hist']['act_busc'],
                       color='steelblue', alpha=0.7, label='F2 +60°')
        axes[1,0].plot(t_f13, resultados['F13'][1]['hist']['act_busc'],
                       color='firebrick', alpha=0.7, label='F13 -60°')
        axes[1,0].axhline(0, color='k', linestyle='--', lw=0.8)
        axes[1,0].set_title('act_busc temporal F2 vs F13')
        axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

    # Evolución temporal act_geom F2 vs F13
    if 'F2' in resultados and 'F13' in resultados:
        axes[1,1].plot(t_f2,  resultados['F2'][1]['hist']['geom'],
                       color='steelblue', alpha=0.7, label='F2 +60°')
        axes[1,1].plot(t_f13, resultados['F13'][1]['hist']['geom'],
                       color='firebrick', alpha=0.7, label='F13 -60°')
        axes[1,1].axhline(0, color='k', linestyle='--', lw=0.8)
        axes[1,1].set_title('act_geom temporal F2 vs F13')
        axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

    axes[1,2].bar(fids_ord, vals_efic)
    axes[1,2].set_title('Eficiencia de régimen')
    axes[1,2].grid(True, alpha=0.3)
    axes[1,2].tick_params(axis='x', rotation=45)

    axes[1,3].bar(fids_ord, vals_lf)
    axes[1,3].set_title('LF activa (%)')
    axes[1,3].grid(True, alpha=0.3)
    axes[1,3].tick_params(axis='x', rotation=45)

    axes[2,0].bar(fids_ord, vals_wrec)
    axes[2,0].set_title('W_rec norma')
    axes[2,0].grid(True, alpha=0.3)
    axes[2,0].tick_params(axis='x', rotation=45)

    axes[2,1].bar(fids_ord, vals_mej)
    axes[2,1].set_title('Mejor eficiencia explorada')
    axes[2,1].grid(True, alpha=0.3)
    axes[2,1].tick_params(axis='x', rotation=45)

    frac_L_vals = [float(np.mean(resultados[f][1]['hist']['frac_L'])) for f in fids_ord]
    frac_R_vals = [float(np.mean(resultados[f][1]['hist']['frac_R'])) for f in fids_ord]
    axes[2,2].plot(fids_ord, frac_L_vals, 'o-', label='L')
    axes[2,2].plot(fids_ord, frac_R_vals, 's-', label='R')
    axes[2,2].set_title('Fracción L/R')
    axes[2,2].legend(); axes[2,2].grid(True, alpha=0.3)
    axes[2,2].tick_params(axis='x', rotation=45)

    G_vals = [float(np.mean(resultados[f][1]['hist']['G_act'])) for f in fids_ord]
    axes[2,3].bar(fids_ord, G_vals)
    axes[2,3].set_title('Ganglio actividad media')
    axes[2,3].grid(True, alpha=0.3)
    axes[2,3].tick_params(axis='x', rotation=45)

    plt.suptitle(
        f'VSTCosmos v87 — Coherencia espectral dirigida '
        f'(BANDA_TRANS={BANDA_TRANS}, F>{F_TRANS_HZ:.0f}Hz)',
        fontsize=13
    )
    plt.tight_layout()
    plt.savefig('v87_coherencia_dirigida.png', dpi=150)
    print("  Gráfico guardado: v87_coherencia_dirigida.png")

    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)


if __name__ == "__main__":
    main()