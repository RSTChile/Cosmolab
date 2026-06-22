#!/usr/bin/env python3
"""
VSTCosmos v101 — Entrenamiento multi-estimulo, evaluacion SOLO BigBang

Pregunta: ¿El aplanamiento de v100 se debe al entrenamiento multi-estimulo
o a la presencia de otros estimulos durante la evaluacion?

Diseño:
  - Entrenamiento: 28 estimulos, 10s c/u (igual a v100)
  - Evaluacion: SOLO BigBang (sin mezcla de estimulos)
  - 9 instancias con diferentes duraciones de exposicion inicial a +60°
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
from datetime import datetime

try:
    import soundfile as sf
    HAS_SF = True
except ImportError:
    HAS_SF = False

warnings.filterwarnings('ignore')

# ============================================================
# PARAMETROS DE LA FISICA DEL CAMPO (identicos a v100)
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

# Constantes fisicas para binaural
DIAMETRO_CABEZA  = 0.175
VELOCIDAD_SONIDO = 343.0
F_TRANS_HZ       = VELOCIDAD_SONIDO / DIAMETRO_CABEZA   # = 1960 Hz

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

BANDA_TRANS = int(DIM_AUD * np.log10(F_TRANS_HZ / F_MIN)
                  / np.log10(F_MAX / F_MIN))
BANDA_TRANS = max(1, min(BANDA_TRANS, DIM_AUD - 1))   # = 11

K_BUSC               = T_PROFUNDA_SEG / T_RECIENTE_SEG
K_ORIENT             = T_PROFUNDA_SEG / T_RECIENTE_SEG
DECAIMIENTO_ACT_BUSC = DT / T_RECIENTE_SEG
EPSILON_BUSC_G       = DIFUSION_BASE * K_BUSC * DT

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
    ('G',        'act_mant'),
    ('aud_L',    'aud_R'),
    ('act_perm', 'aud_L'),
    ('act_perm', 'aud_R'),
    ('act_geom', 'aud_L'),
    ('act_geom', 'aud_R'),
]

# Logging fino
LOG_FINO_DT = 0.5
LOG_FINO_PASOS = int(LOG_FINO_DT / DT)
VARIACION_FLOOR = 1e-6

# Ventana para analisis de estabilidad (ultimos 30s)
VENTANA_FINAL_SEG = 30
VENTANA_FINAL_PASOS = int(VENTANA_FINAL_SEG / DT)

# Duracion por estimulo en entrenamiento
DURACION_ENTRENAMIENTO = 10.0

# NOMBRES EXACTOS DE LOS ARCHIVOS (igual que v100)
ESTIMULOS_NEG = [
    'BigBang_neg60deg', 'Voz_Estudio_neg60deg', 'Ritmos aleatorios_neg60deg',
    'Ondas mixtas_neg60deg', 'Pulso logaritmico_neg60deg', 'Viento_neg60deg',
    'Voz+Viento_1_neg60deg', 'Voz+Viento_2_neg60deg', 'Ruido blanco_neg60deg',
    'Tono puro_neg60deg', 'Brandemburgo_neg60deg', 'voz_neg60deg'
]

ESTIMULOS_POS = [
    'BigBang_pos60deg', 'Voz_Estudio_pos60deg', 'Brandemburgo_pos60deg',
    'Tono puro_pos60deg', 'Ruido blanco_pos60deg', 'Ritmos aleatorios_pos60deg',
    'Ondas mixtas_pos60deg', 'Pulso logaritmico_pos60deg', 'Viento_pos60deg',
    'Voz+Viento_1_pos60deg', 'Voz+Viento_2_pos60deg', 'musica_pos60deg',
    'tono_pos60deg', 'ruido_pos60deg', 'voz_pos60deg', 'voz_viento_pos60deg'
]

# Todos los estimulos para entrenamiento
ESTIMULOS_ENTRENAMIENTO = list(set(ESTIMULOS_NEG + ESTIMULOS_POS))

print("=" * 100)
print("VSTCosmos v101 — Entrenamiento multi-estimulo, evaluacion SOLO BigBang")
print()
print("  Pregunta: El aplanamiento de v100 se debe al entrenamiento")
print("  multi-estimulo o a la presencia de otros estimulos durante")
print("  la evaluacion?")
print()
print("  Diseño:")
print("    - Entrenamiento: 28 estimulos, 10s c/u (igual a v100)")
print("    - Evaluacion: SOLO BigBang (sin mezcla de estimulos)")
print("    - 9 instancias con diferentes duraciones de exposicion inicial")
print()
print(f"  BANDA_TRANS={BANDA_TRANS} (F>{F_TRANS_HZ:.0f}Hz)")
print(f"  DIM_TOTAL={DIM_TOTAL}")
print("=" * 100)


# ============================================================
# CARGA DE TODOS LOS ARCHIVOS BINAURALES
# ============================================================
def cargar_todos_sonidos(directorio='audio_binaural'):
    """Carga todos los archivos WAV del directorio"""
    archivos = {}
    
    print(f"\n[Carga] Desde '{directorio}/'...")
    
    for clave in ESTIMULOS_ENTRENAMIENTO:
        filepath = os.path.join(directorio, clave + '.wav')
        if not os.path.exists(filepath):
            print(f"    [X] {clave:35s} no encontrado")
            continue
        
        try:
            data, sr = sf.read(filepath, dtype='float32')
            if data.ndim == 1:
                canal_L = data
                canal_R = data.copy()
            else:
                canal_L = data[:, 0]
                canal_R = data[:, 1] if data.shape[1] > 1 else data[:, 0].copy()
            
            dur_real = len(canal_L) / sr
            archivos[clave] = (filepath, sr, canal_L, canal_R)
            print(f"    [OK] {clave:35s} ({dur_real:.1f}s, {sr}Hz)")
            
        except Exception as e:
            print(f"    [X] {clave:35s} {e}")
    
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
# FUNCIONES BASE (identicas a v100)
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
        perfil += np.abs(fft)[:n_bins] ** 2
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
# GRADIENTE ENERGETICO
# ============================================================
def calcular_gradiente_energetico_dirigido(obj_L, obj_R):
    if BANDA_TRANS >= DIM_AUD:
        return 0.0
    energia_L = float(np.mean(obj_L[BANDA_TRANS:, :] ** 2))
    energia_R = float(np.mean(obj_R[BANDA_TRANS:, :] ** 2))
    total     = energia_L + energia_R + 1e-10
    return (energia_R - energia_L) / total


# ============================================================
# COHERENCIA (solo diagnostico)
# ============================================================
def calcular_coherencia_dirigida(obj_L, obj_R, W_prof, region_int):
    if BANDA_TRANS >= DIM_AUD:
        return 0.0, 0.0, 0.0
    n_prof  = W_prof.shape[0]
    n_cols  = W_prof.shape[1]
    n_int   = region_int.shape[0]
    n_altas = DIM_AUD - BANDA_TRANS
    if n_altas <= 0:
        return 0.0, 0.0, 0.0
    perfil_L = obj_L[BANDA_TRANS:, :].mean(axis=1)
    perfil_R = obj_R[BANDA_TRANS:, :].mean(axis=1)
    perfil_i = region_int.mean(axis=1)
    min_c = min(n_cols - BANDA_TRANS, n_altas)
    min_p = min(n_prof, n_int)
    if min_c <= 0 or min_p <= 0:
        return 0.0, 0.0, 0.0
    W_alto = W_prof[:min_p, BANDA_TRANS:BANDA_TRANS + min_c]
    pred_L = W_alto @ perfil_L[:min_c].reshape(-1, 1)
    pred_R = W_alto @ perfil_R[:min_c].reshape(-1, 1)
    ref    = perfil_i[:min_p].reshape(-1, 1)
    err_L  = float(np.mean((pred_L - ref) ** 2))
    err_R  = float(np.mean((pred_R - ref) ** 2))
    total  = err_L + err_R + 1e-10
    return float((err_R - err_L) / total), err_L, err_R


# ============================================================
# ACT_BUSC
# ============================================================
def actualizar_act_busc_desde_gradiente(Phi_total, gradiente_E, dt):
    ab0, ab1 = idx['act_busc']
    senal = PHI_EQUILIBRIO + float(np.tanh(K_BUSC * gradiente_E)) * DIFUSION_BASE
    Phi_total[ab0:ab1, :] = (
        (1.0 - DECAIMIENTO_ACT_BUSC) * Phi_total[ab0:ab1, :] +
        DECAIMIENTO_ACT_BUSC * senal
    )
    return Phi_total

def aplicar_forzamiento_busc_a_ganglio(Phi_total, dt):
    ab0, ab1 = idx['act_busc']
    g0,  g1  = idx['G']
    estado_busc = float(np.mean(Phi_total[ab0:ab1, :])) - PHI_EQUILIBRIO
    n = min(ab1 - ab0, g1 - g0)
    Phi_total[g0:g0 + n, :] += EPSILON_BUSC_G * estado_busc
    return Phi_total


# ============================================================
# ACT_GEOM - ADITIVA CON PROYECCION DIRECCIONAL
# ============================================================
def aplicar_orientacion_v1_aditiva(Phi_total, gradiente_E, W_rec, dt):
    acg0  = idx['act_geom'][0]
    acg1  = idx['act_geom'][1]
    mitad = max(1, (acg1 - acg0) // 2)

    senal_grad = float(np.clip(
        gradiente_E * DIFUSION_BASE * K_ORIENT * dt, -0.1, 0.1
    ))

    aud_L = Phi_total[idx['aud_L'][0]:idx['aud_L'][1], :]
    aud_R = Phi_total[idx['aud_R'][0]:idx['aud_R'][1], :]
    aud_dir = (aud_L - aud_R).mean(axis=1)
    norm_dir = np.linalg.norm(aud_dir)

    if norm_dir > 1e-10:
        aud_dir_n = aud_dir / norm_dir
        min_dim = min(W_rec.shape[1], aud_dir_n.shape[0])
        sesgo_dir = float(np.mean(W_rec[:, :min_dim] @ aud_dir_n[:min_dim]))
    else:
        sesgo_dir = 0.0

    sesgo_rec   = float(np.tanh(sesgo_dir)) * DIFUSION_BASE * dt
    senal_total = senal_grad + sesgo_rec

    Phi_total[acg0:acg0 + mitad, :] += senal_total
    Phi_total[acg0 + mitad:acg1, :] -= senal_total
    return Phi_total


# ============================================================
# ACTUACION
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

def aplicar_entrada_cualitativa(Phi_total, obj_L, obj_R, frac_L, frac_R, sesgo):
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
# EXPLORACION ACTIVA
# ============================================================
def explorar_actuadores(Phi_total, explorador, lf_activa, eficiencia, dt):
    AMPLITUD_MAX = DIFUSION_BASE
    ap0, ap1 = idx['act_perm']
    ag0, ag1 = idx['act_geom']
    if lf_activa:
        amplitud = AMPLITUD_MAX * min(1.0, explorador.pasos_en_lf / TAU_EXPLORACION)
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
                              Phi_int_historia, dt, modo_aud='dir'):
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
# OMEGA_ORIENT
# ============================================================
def calcular_omega_orient(Phi_total, gradiente_hist_fase):
    if len(gradiente_hist_fase) < 2:
        return 0.0

    ag0, ag1 = idx['act_geom']
    ab0, ab1 = idx['act_busc']

    geom_medio = float(np.mean(np.tanh(Phi_total[ag0:ag1, :])))
    busc_medio = float(np.mean(Phi_total[ab0:ab1, :])) - PHI_EQUILIBRIO

    config_interna = np.array([geom_medio, busc_medio])

    grads    = np.array(gradiente_hist_fase)
    grad_pos = float(np.mean(grads[grads >= 0])) if np.any(grads >= 0) else 0.0
    grad_neg = float(np.mean(np.abs(grads[grads < 0]))) if np.any(grads < 0) else 0.0
    firma_entorno = np.array([grad_pos, -grad_neg])

    norma_c = np.linalg.norm(config_interna)
    norma_f = np.linalg.norm(firma_entorno)

    if norma_c < 1e-10 or norma_f < 1e-10:
        return 0.0

    return float(np.dot(config_interna, firma_entorno) / (norma_c * norma_f))


# ============================================================
# EFICIENCIA
# ============================================================
def calcular_eficiencia(Phi_total, ged_actual):
    region_int = Phi_total[idx['int'][0]:idx['int'][1], :]
    variacion_real = float(np.mean(np.abs(np.diff(region_int, axis=1))))
    variacion_floor = max(variacion_real, VARIACION_FLOOR)
    efic = ged_actual / variacion_floor
    return efic, variacion_real

def calcular_senal_busqueda(Phi_total):
    ab0, ab1 = idx['act_busc']
    return float(np.mean(Phi_total[ab0:ab1, :])) - PHI_EQUILIBRIO


# ============================================================
# ACTUALIZACION PRINCIPAL DEL CAMPO
# ============================================================
def actualizar_campo(Phi_total, Phi_vel_total, W_prof, W_rec,
                     Phi_int_historia, obj_L, obj_R,
                     frac_L, frac_R, sesgo, dt, modo_aud='dir'):

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

    if modo_aud == 'dir':
        aud_comb = aud_L - aud_R
    else:
        aud_comb = (aud_L + aud_R) / 2.0

    W_prof, W_rec, M_plast, error_rec, coherencia, Phi_int_historia = \
        aplicar_plasticidad_dual(
            region_int, aud_comb, W_prof, W_rec, Phi_int_historia, dt,
            modo_aud=modo_aud
        )

    M_campo = np.zeros_like(Phi_total)
    n_m     = M_plast.shape[0]
    M_campo[idx['int'][0]:idx['int'][0] + n_m, :] = M_plast

    Phi_total = aplicar_entrada_cualitativa(Phi_total, obj_L, obj_R,
                                            frac_L, frac_R, sesgo)

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
# ENTRENAMIENTO CON MULTIPLES ESTIMULOS
# ============================================================
def entrenar_multiestimulo(archivos, duracion_por_estimulo=10.0, modo_aud='dir'):
    """Entrena el campo rotando por todos los estimulos disponibles"""
    estimulos_disponibles = [e for e in ESTIMULOS_ENTRENAMIENTO if e in archivos]
    
    print(f"\n[Entrenamiento multi-estimulo] {len(estimulos_disponibles)} estimulos, "
          f"{duracion_por_estimulo}s c/u = {len(estimulos_disponibles)*duracion_por_estimulo:.0f}s total")
    
    Phi_total, Phi_vel_total = inicializar_campo()
    W_prof, W_rec, Phi_int_historia = inicializar_memorias()
    explorador = ExploradorActuadores()
    
    errores = []
    
    for estimulo in estimulos_disponibles:
        _, sr, c_L, c_R = archivos[estimulo]
        vent = int(sr * VENTANA_FFT_MS / 1000)
        hop = int(sr * HOP_FFT_MS / 1000)
        n_pasos = int(duracion_por_estimulo / DT)
        
        print(f"    Entrenando con {estimulo}...", end=" ", flush=True)
        
        for paso in range(n_pasos):
            obj_L = preparar_objetivo_canal(c_L, sr, paso, vent, hop, DIM_AUD, DIM_TIME)
            obj_R = preparar_objetivo_canal(c_R, sr, paso, vent, hop, DIM_AUD, DIM_TIME)
            
            gradiente_E = calcular_gradiente_energetico_dirigido(obj_L, obj_R)
            Phi_total = actualizar_act_busc_desde_gradiente(Phi_total, gradiente_E, DT)
            Phi_total = aplicar_forzamiento_busc_a_ganglio(Phi_total, DT)
            Phi_total = aplicar_orientacion_v1_aditiva(Phi_total, gradiente_E, W_rec, DT)
            
            fL, fR, sf_v, _, _ = calcular_parametros_actuacion(Phi_total)
            
            Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia, \
                _, error_rec, _ = actualizar_campo(
                    Phi_total, Phi_vel_total, W_prof, W_rec,
                    Phi_int_historia, obj_L, obj_R, fL, fR, sf_v, DT,
                    modo_aud=modo_aud
                )
            errores.append(error_rec)
        
        print(f"error_final={error_rec:.6f}")
    
    print(f"  ERROR_EQUILIBRIO: {min(errores):.6f}")
    print(f"  W_prof: {np.mean(np.abs(W_prof)):.4f}")
    print(f"  W_rec:  {np.mean(np.abs(W_rec)):.4f}")
    
    return Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia, explorador


# ============================================================
# SIMULACION DE FASE (con logging reducido para BigBang)
# ============================================================
def simular_fase_bigbang(Phi_total, Phi_vel_total, W_prof, W_rec,
                         Phi_int_historia, historial_ef, explorador,
                         archivos, clave, duracion, verbose=False,
                         modo_aud='dir'):
    """Simula una fase con un unico estimulo (BigBang)"""
    
    if clave not in archivos:
        print(f"  [ERROR] {clave} no disponible")
        return None
    
    _, sr, c_L, c_R = archivos[clave]
    vent = int(sr * VENTANA_FFT_MS / 1000)
    hop = int(sr * HOP_FFT_MS / 1000)
    n_pasos = int(duracion / DT)
    n_pasos = min(n_pasos, len(c_L) // hop + 1)

    hist = {k: [] for k in [
        'ged_L', 'ged_R', 'grad_E', 'act_busc', 'coh_rel',
        'geom', 'frac_L', 'frac_R', 'efic', 'lf',
        'w_rec', 'w_prof', 'G_act', 'omega', 'var_int',
        'act_mant_media', 'act_mant_var'
    ]}

    gradiente_hist_fase = []
    lf_prev = False

    for paso in range(n_pasos):
        obj_L = preparar_objetivo_canal(c_L, sr, paso, vent, hop, DIM_AUD, DIM_TIME)
        obj_R = preparar_objetivo_canal(c_R, sr, paso, vent, hop, DIM_AUD, DIM_TIME)

        gradiente_E = calcular_gradiente_energetico_dirigido(obj_L, obj_R)
        gradiente_hist_fase.append(gradiente_E)

        region_int = Phi_total[idx['int'][0]:idx['int'][1], :]
        coh_rel, _, _ = calcular_coherencia_dirigida(
            obj_L, obj_R, W_prof, region_int
        )

        Phi_total = actualizar_act_busc_desde_gradiente(Phi_total, gradiente_E, DT)
        Phi_total = aplicar_forzamiento_busc_a_ganglio(Phi_total, DT)
        Phi_total = aplicar_orientacion_v1_aditiva(Phi_total, gradiente_E, W_rec, DT)

        fL, fR, sf_v, asim, _ = calcular_parametros_actuacion(Phi_total)

        a_L   = Phi_total[idx['aud_L'][0]:idx['aud_L'][1], :]
        a_R   = Phi_total[idx['aud_R'][0]:idx['aud_R'][1], :]
        ged_L = calcular_ged_entre(region_int, a_L)
        ged_R = calcular_ged_entre(region_int, a_R)
        ged   = (ged_L + ged_R) / 2.0

        efic, var_int_real = calcular_eficiencia(Phi_total, ged)

        act_mant = Phi_total[idx['act_mant'][0]:idx['act_mant'][1], :]
        act_mant_media = float(np.mean(act_mant))
        act_mant_var   = float(np.var(act_mant))

        historial_ef.append(efic)
        if len(historial_ef) > TAU_EFICIENCIA * 2:
            historial_ef.pop(0)

        explorador.actualizar(lf_prev, efic, fL, fR, sf_v)

        Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia, \
            lf_activa, error_rec, _ = actualizar_campo(
                Phi_total, Phi_vel_total, W_prof, W_rec,
                Phi_int_historia, obj_L, obj_R, fL, fR, sf_v, DT,
                modo_aud=modo_aud
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
        omega = calcular_omega_orient(Phi_total, gradiente_hist_fase)

        for k, v in [
            ('ged_L',   ged_L), ('ged_R',   ged_R),
            ('grad_E',  gradiente_E), ('act_busc', act_busc_val),
            ('coh_rel', coh_rel), ('geom',    geom),
            ('frac_L',  fL), ('frac_R',  fR),
            ('efic',    efic), ('lf',      lf_activa),
            ('w_rec',   np.mean(np.abs(W_rec))),
            ('w_prof',  np.mean(np.abs(W_prof))),
            ('G_act',   G_act), ('omega',   omega),
            ('var_int', var_int_real),
            ('act_mant_media', act_mant_media),
            ('act_mant_var',   act_mant_var),
        ]:
            hist[k].append(v)

    def M(k): return float(np.mean(hist[k])) if hist[k] else 0.0
    n_half = len(hist['geom']) // 2

    omega_segunda = float(np.mean(hist['omega'][n_half:])) if n_half > 0 else M('omega')
    act_mant_final = float(np.mean(hist['act_mant_media'][-VENTANA_FINAL_PASOS:])) if len(hist['act_mant_media']) > VENTANA_FINAL_PASOS else M('act_mant_media')

    return {
        'hist': hist,
        'omega_segunda': omega_segunda,
        'act_mant_final': act_mant_final,
        'phi_total': Phi_total, 'phi_vel': Phi_vel_total,
        'W_prof': W_prof, 'W_rec': W_rec,
        'Phi_int_historia': Phi_int_historia,
    }


# ============================================================
# EVALUACION COMPLETA DE UNA INSTANCIA (SOLO BIGBANG)
# ============================================================
def evaluar_instancia(archivos, duracion_inicial, nombre_instancia, modo_aud='dir'):
    """
    duracion_inicial: segundos de exposicion a +60 grados (BigBang) al inicio
    """
    print(f"\n{'█'*100}")
    print(f"INSTANCIA {nombre_instancia} — Entrenada multi-estimulo, "
          f"evaluada SOLO BigBang: +60({duracion_inicial}s) -> -60(120s) -> +60(120s)")
    print(f"{'█'*100}")
    
    # Entrenamiento con todos los estimulos
    Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia, explorador = \
        entrenar_multiestimulo(archivos, duracion_por_estimulo=10.0, modo_aud=modo_aud)
    
    resultados = {
        'duracion_inicial': duracion_inicial,
    }
    
    # Fase 1: Exposicion inicial a +60 grados (BigBang)
    if duracion_inicial > 0:
        print(f"\n  Fase 1: +60 (BigBang) - {duracion_inicial}s")
        historial_ef = []
        
        res = simular_fase_bigbang(
            Phi_total, Phi_vel_total, W_prof, W_rec,
            Phi_int_historia, historial_ef, explorador,
            archivos, 'BigBang_pos60deg', duracion_inicial, verbose=False,
            modo_aud=modo_aud
        )
        
        if res:
            resultados['omega_pos1'] = res['omega_segunda']
            resultados['act_mant_pos1'] = res['act_mant_final']
            Phi_total = res['phi_total']
            Phi_vel_total = res['phi_vel']
            W_prof = res['W_prof']
            W_rec = res['W_rec']
            Phi_int_historia = res['Phi_int_historia']
    else:
        resultados['omega_pos1'] = None
    
    # Fase 2: Exposicion a -60 grados (BigBang)
    print(f"\n  Fase 2: -60 (BigBang) - 120s")
    historial_ef = []
    
    res_neg = simular_fase_bigbang(
        Phi_total, Phi_vel_total, W_prof, W_rec,
        Phi_int_historia, historial_ef, explorador,
        archivos, 'BigBang_neg60deg', 120.0, verbose=False,
        modo_aud=modo_aud
    )
    
    if res_neg:
        resultados['omega_neg'] = res_neg['omega_segunda']
        resultados['act_mant_neg'] = res_neg['act_mant_final']
        Phi_total = res_neg['phi_total']
        Phi_vel_total = res_neg['phi_vel']
        W_prof = res_neg['W_prof']
        W_rec = res_neg['W_rec']
        Phi_int_historia = res_neg['Phi_int_historia']
    
    # Fase 3: Exposicion a +60 grados (BigBang)
    print(f"\n  Fase 3: +60 (BigBang) - 120s")
    historial_ef = []
    
    res_pos = simular_fase_bigbang(
        Phi_total, Phi_vel_total, W_prof, W_rec,
        Phi_int_historia, historial_ef, explorador,
        archivos, 'BigBang_pos60deg', 120.0, verbose=False,
        modo_aud=modo_aud
    )
    
    if res_pos:
        resultados['omega_pos2'] = res_pos['omega_segunda']
        resultados['act_mant_pos2'] = res_pos['act_mant_final']
    
    return resultados


# ============================================================
# MAIN
# ============================================================
def main():
    archivos = cargar_todos_sonidos('audio_binaural')
    
    if 'BigBang_pos60deg' not in archivos:
        print("\n[ERROR] BigBang_pos60deg es necesario y no se cargo.")
        return
    
    # Duraciones a probar (segundos)
    duraciones = [0, 15, 30, 45, 60, 75, 90, 105, 120]
    nombres = ['H0', 'H15', 'H30', 'H45', 'H60', 'H75', 'H90', 'H105', 'H120']
    
    resultados_totales = {}
    
    for dur, nombre in zip(duraciones, nombres):
        resultados_totales[nombre] = evaluar_instancia(archivos, dur, nombre)
    
    # ============================================================
    # REPORTE DE OBSERVACIONES
    # ============================================================
    print()
    print("=" * 100)
    print("REPORTE DE OBSERVACIONES - v101")
    print("=" * 100)
    
    print("\n  OMEGA PARA BIGBANG SEGUN DURACION INICIAL")
    print("  " + "-" * 70)
    print(f"  {'Instancia':<10} {'Dur. inicial':<12} {'Ω BigBang Neg':<16} {'Ω BigBang Pos final':<20}")
    print("  " + "-" * 70)
    
    for nombre, dur in zip(nombres, duraciones):
        res = resultados_totales[nombre]
        omega_neg = res.get('omega_neg', None)
        omega_pos = res.get('omega_pos2', None)
        
        neg_str = f"{omega_neg:.4f}" if omega_neg is not None else "N/D"
        pos_str = f"{omega_pos:.4f}" if omega_pos is not None else "N/D"
        
        print(f"  {nombre:<10} {dur:<12} {neg_str:<16} {pos_str:<20}")
    
    # Curva de dosis-respuesta para BigBang
    fig, ax = plt.subplots(figsize=(12, 7))
    
    omega_pos_vals = []
    omega_neg_vals = []
    
    for nombre, dur in zip(nombres, duraciones):
        res = resultados_totales[nombre]
        omega_pos = res.get('omega_pos2', None)
        omega_neg = res.get('omega_neg', None)
        omega_pos_vals.append(omega_pos if omega_pos is not None else np.nan)
        omega_neg_vals.append(omega_neg if omega_neg is not None else np.nan)
    
    ax.plot(duraciones, omega_pos_vals, 'o-', color='steelblue', linewidth=2, markersize=10, label='Ω para BigBang +60° (fase final)')
    ax.plot(duraciones, omega_neg_vals, 's-', color='firebrick', linewidth=2, markersize=8, label='Ω para BigBang -60°')
    
    # Referencias de v99 y v100
    ax.axhline(y=0.762, color='gray', linestyle='--', alpha=0.7, label='Referencia v99 H0')
    ax.axhline(y=0.939, color='red', linestyle='--', alpha=0.7, label='Pico v99 H60')
    ax.axhline(y=0.650, color='green', linestyle='--', alpha=0.7, label='Valor plano v100')
    
    ax.set_xlabel('Duracion de exposicion inicial a +60 (s)')
    ax.set_ylabel('Omega final')
    ax.set_title('VSTCosmos v101 - Entrenamiento multi-estimulo, evaluacion SOLO BigBang\nOmega vs duracion de exposicion inicial')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('v101_logs', exist_ok=True)
    plt.savefig(f'v101_logs/v101_dosis_respuesta_{timestamp}.png', dpi=150)
    
    # Guardar resultados en CSV
    with open(f'v101_logs/v101_resultados_{timestamp}.csv', 'w') as f:
        f.write("instancia,duracion_inicial,omega_bigbang_neg,omega_bigbang_pos_final\n")
        for nombre, dur in zip(nombres, duraciones):
            res = resultados_totales[nombre]
            neg = res.get('omega_neg', '')
            pos = res.get('omega_pos2', '')
            f.write(f"{nombre},{dur},{neg},{pos}\n")
    
    print(f"\n  Grafico guardado: v101_logs/v101_dosis_respuesta_{timestamp}.png")
    print(f"  Resultados guardados: v101_logs/v101_resultados_{timestamp}.csv")
    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO - Revise los datos y extraiga sus propias conclusiones")
    print("=" * 100)


if __name__ == "__main__":
    main()