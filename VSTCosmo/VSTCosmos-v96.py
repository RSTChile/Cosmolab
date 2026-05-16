#!/usr/bin/env python3
"""
VSTCosmos v96 — Diagnóstico de rigidez geométrica

Tres ejes de investigación (fenómeno no predicho en v95):

  EJE 1 — ¿La rigidez de act_geom tiene umbral?
    F13 extendida a 60s (vs 20s en v95).
    Si act_geom no cede en 60s de gradE negativo continuo → atractor absoluto.
    Si cede gradualmente → inercia con escala temporal larga.

  EJE 2 — ¿El fenómeno depende del entrenamiento previo?
    Instancia C: entrenada directamente en -60° desde cero.
    Compara act_geom de C en F2 (+60°) vs instancia B (entrenada en +60°,
    expuesta a -60°). Si C también mantiene su orientación → fenómeno
    simétrico y estructural. Si no → la historia del entrenamiento importa.

  EJE 3 — ¿Qué ocurre adentro cuando Ω se mantiene positivo?
    Logging fino de region_int cada 0.5s (en lugar de cada 2s).
    GED registrado con 6 decimales sin truncamiento.
    Variación de region_int registrada explícitamente.
    Activo en todas las fases F2 y F13 de todos los ciclos.

Correcciones sobre v95:
  - Floor en variación para cálculo de eficiencia (floor = 1e-6).
  - GED con 6 dígitos en log.
  - Variación de region_int registrada en hist como 'var_int'.

Instancias:
  A (+60°, aditiva)  — control de referencia
  B (-60°, aditiva)  — control inversión (entrenado en opuesto al protocolo)
  C (-60°, aditiva)  — nuevo: entrenado en -60°, evaluado en entorno simétrico
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

# Logging fino: muestreo cada 0.5s en fases de diagnóstico
LOG_FINO_DT = 0.5
LOG_FINO_PASOS = int(LOG_FINO_DT / DT)   # = 50

# Umbral de floor para cálculo de eficiencia (corrige bug v95)
VARIACION_FLOOR = 1e-6

print("=" * 100)
print("VSTCosmos v96 — Diagnóstico de rigidez geométrica")
print("")
print("  EJE 1: F13 extendida a 60s — ¿act_geom tiene umbral de rendición?")
print("  EJE 2: Instancia C (-60° directo) — ¿el fenómeno depende del entrenamiento previo?")
print("  EJE 3: Logging fino de region_int cada 0.5s — ¿qué ocurre adentro?")
print("")
print(f"  BANDA_TRANS={BANDA_TRANS} (F>{F_TRANS_HZ:.0f}Hz)")
print(f"  DIM_TOTAL={DIM_TOTAL}")
print(f"  VARIACION_FLOOR={VARIACION_FLOOR} (corrección bug eficiencia)")
print("=" * 100)

# ============================================================
# CARGA DE ARCHIVOS BINAURALES
# ============================================================
def cargar_todos_binaurales(directorio='audio_binaural', duracion=35.0):
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
# GRADIENTE ENERGÉTICO
# ============================================================
def calcular_gradiente_energetico_dirigido(obj_L, obj_R):
    if BANDA_TRANS >= DIM_AUD:
        return 0.0
    energia_L = float(np.mean(obj_L[BANDA_TRANS:, :] ** 2))
    energia_R = float(np.mean(obj_R[BANDA_TRANS:, :] ** 2))
    total     = energia_L + energia_R + 1e-10
    return (energia_R - energia_L) / total

# ============================================================
# COHERENCIA (solo diagnóstico)
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
    señal = PHI_EQUILIBRIO + float(np.tanh(K_BUSC * gradiente_E)) * DIFUSION_BASE
    Phi_total[ab0:ab1, :] = (
        (1.0 - DECAIMIENTO_ACT_BUSC) * Phi_total[ab0:ab1, :] +
        DECAIMIENTO_ACT_BUSC * señal
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
# ACT_GEOM — ADITIVA CON PROYECCIÓN DIRECCIONAL (v95)
# ============================================================
def aplicar_orientacion_v1_aditiva(Phi_total, gradiente_E, W_rec, dt):
    acg0  = idx['act_geom'][0]
    acg1  = idx['act_geom'][1]
    mitad = max(1, (acg1 - acg0) // 2)

    señal_grad = float(np.clip(
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
    señal_total = señal_grad + sesgo_rec

    Phi_total[acg0:acg0 + mitad, :] += señal_total
    Phi_total[acg0 + mitad:acg1, :] -= señal_total
    return Phi_total

# ============================================================
# ACTUACIÓN
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
# EXPLORACIÓN ACTIVA
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
                              Phi_int_historia, dt, modo_aud='suma'):
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
# Ω_ORIENT
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
# EFICIENCIA — CORREGIDA (floor en variación)
# ============================================================
def calcular_eficiencia(Phi_total, ged_actual):
    """
    v96: variacion con floor = VARIACION_FLOOR para evitar explosión numérica.
    Retorna también la variación real (sin floor) para diagnóstico.
    """
    region_int = Phi_total[idx['int'][0]:idx['int'][1], :]
    variacion_real = float(np.mean(np.abs(np.diff(region_int, axis=1))))
    variacion_floor = max(variacion_real, VARIACION_FLOOR)
    efic = ged_actual / variacion_floor
    return efic, variacion_real

def calcular_senal_busqueda(Phi_total):
    ab0, ab1 = idx['act_busc']
    return float(np.mean(Phi_total[ab0:ab1, :])) - PHI_EQUILIBRIO

# ============================================================
# ACTUALIZACIÓN PRINCIPAL DEL CAMPO
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
# ENTRENAMIENTO
# ============================================================
def entrenar(archivos, duracion=30.0, clave_audio='voz_pos', etiqueta=None,
             modo_aud='dir'):
    if etiqueta is None:
        etiqueta = clave_audio
    print(f"\n[Entrenamiento] {etiqueta} — modo={modo_aud} ({duracion}s)")
    Phi_total, Phi_vel_total = inicializar_campo()
    W_prof, W_rec, Phi_int_historia = inicializar_memorias()
    explorador = ExploradorActuadores()

    _, sr, c_L, c_R = archivos[clave_audio]
    vent    = int(sr * VENTANA_FFT_MS / 1000)
    hop     = int(sr * HOP_FFT_MS  / 1000)
    n_pasos = int(duracion / DT)
    errores = []

    for paso in range(n_pasos):
        obj_L = preparar_objetivo_canal(c_L, sr, paso, vent, hop, DIM_AUD, DIM_TIME)
        obj_R = preparar_objetivo_canal(c_R, sr, paso, vent, hop, DIM_AUD, DIM_TIME)

        gradiente_E = calcular_gradiente_energetico_dirigido(obj_L, obj_R)
        Phi_total   = actualizar_act_busc_desde_gradiente(Phi_total, gradiente_E, DT)
        Phi_total   = aplicar_forzamiento_busc_a_ganglio(Phi_total, DT)
        Phi_total   = aplicar_orientacion_v1_aditiva(Phi_total, gradiente_E, W_rec, DT)

        fL, fR, sf_v, _, _ = calcular_parametros_actuacion(Phi_total)

        Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia, \
            _, error_rec, _ = actualizar_campo(
                Phi_total, Phi_vel_total, W_prof, W_rec,
                Phi_int_historia, obj_L, obj_R, fL, fR, sf_v, DT,
                modo_aud=modo_aud
            )
        errores.append(error_rec)

        if paso % 500 == 0:
            print(f"    Paso {paso}/{n_pasos}, error={error_rec:.6f}")

    print(f"  ERROR_EQUILIBRIO: {min(errores):.6f}")
    print(f"  W_prof: {np.mean(np.abs(W_prof)):.4f}")
    print(f"  W_rec:  {np.mean(np.abs(W_rec)):.4f}")

    return Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia, explorador

# ============================================================
# SIMULACIÓN DE FASE — CON LOGGING FINO INTEGRADO
# ============================================================
def simular_fase(Phi_total, Phi_vel_total, W_prof, W_rec,
                 Phi_int_historia, historial_ef, explorador,
                 sr, canal_L, canal_R, duracion, verbose=True,
                 modo_aud='dir', fase_id='', log_fino=False):
    """
    log_fino=True: registra métricas de region_int cada LOG_FINO_DT segundos.
    Activo en F2 y F13 para todos los ciclos (eje 3).
    """
    vent    = int(sr * VENTANA_FFT_MS / 1000)
    hop     = int(sr * HOP_FFT_MS  / 1000)

    if duracion is None:
        n_pasos_max = len(canal_L) // hop
    else:
        n_pasos_max = int(duracion / DT)

    n_pasos = min(n_pasos_max, int(60.0 / DT))  # máx 60s (eje 1)

    hist = {k: [] for k in [
        'ged_L', 'ged_R', 'grad_E', 'act_busc', 'coh_rel',
        'geom', 'frac_L', 'frac_R', 'efic', 'lf',
        'w_rec', 'w_prof', 'G_act', 'omega', 'var_int'
    ]}

    # Log fino: lista de dicts con snapshot de region_int
    log_fino_registros = []

    gradiente_hist_fase = []
    lf_prev = False

    for paso in range(n_pasos):
        obj_L = preparar_objetivo_canal(canal_L, sr, paso, vent, hop, DIM_AUD, DIM_TIME)
        obj_R = preparar_objetivo_canal(canal_R, sr, paso, vent, hop, DIM_AUD, DIM_TIME)

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

        # Eficiencia con floor (eje 3 — corrige bug v95)
        efic, var_int_real = calcular_eficiencia(Phi_total, ged)

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
        ]:
            hist[k].append(v)

        # ---- LOGGING FINO (eje 3) ----
        if log_fino and paso % LOG_FINO_PASOS == 0:
            ri_now    = Phi_total[idx['int'][0]:idx['int'][1], :]
            ri_media  = float(np.mean(ri_now))
            ri_std    = float(np.std(ri_now))
            ri_var    = float(np.var(ri_now))
            ri_diff   = float(np.mean(np.abs(np.diff(ri_now, axis=1))))
            log_fino_registros.append({
                't':         paso * DT,
                'ged':       ged,
                'ged_6d':    f"{ged:.6f}",
                'var_int':   var_int_real,
                'ri_media':  ri_media,
                'ri_std':    ri_std,
                'ri_var':    ri_var,
                'ri_diff':   ri_diff,
                'efic':      efic,
                'geom':      geom,
                'omega':     omega,
                'gradE':     gradiente_E,
                'lf':        lf_activa,
            })

        if verbose and paso % 200 == 0:
            print(f"    t={paso*DT:.1f}s | GED={ged:.6f} | "
                  f"gradE={gradiente_E:+.4f} | busc={act_busc_val:+.4f} | "
                  f"geom={geom:+.4f} | Ω={omega:+.3f} | "
                  f"G={G_act:.4f} | efic={efic:.3f} | "
                  f"var_int={var_int_real:.2e} | "
                  f"LF={'ACTIVA' if lf_activa else 'inact'}")

    def M(k): return float(np.mean(hist[k])) if hist[k] else 0.0
    n_half = len(hist['geom']) // 2

    geom_primera  = float(np.mean(hist['geom'][:n_half])) if n_half > 0 else 0.0
    geom_segunda  = float(np.mean(hist['geom'][n_half:])) if n_half > 0 else 0.0
    geom_conv     = (geom_primera * geom_segunda > 0) if n_half > 0 else False
    busc_segunda  = float(np.mean(hist['act_busc'][n_half:])) if n_half > 0 else 0.0
    grad_medio    = M('grad_E')
    omega_media   = M('omega')
    omega_segunda = float(np.mean(hist['omega'][n_half:])) if n_half > 0 else 0.0

    # Eje 1: tendencia de geom en F13 larga — cuartiles
    n_q = max(1, len(hist['geom']) // 4)
    geom_q1 = float(np.mean(hist['geom'][:n_q]))
    geom_q4 = float(np.mean(hist['geom'][-n_q:]))
    geom_tendencia = geom_q4 - geom_q1   # negativo = act_geom bajó

    var_int_media = M('var_int')

    if verbose:
        print(f"\n  Resumen:")
        print(f"    GED L/R:                  {M('ged_L'):.6f} / {M('ged_R'):.6f}")
        print(f"    Gradiente energético:     {grad_medio:+.4f}")
        print(f"    act_busc (2ª mitad):      {busc_segunda:+.4f}")
        print(f"    act_geom (2ª mitad):      {geom_segunda:+.4f}")
        print(f"    act_geom Q1→Q4:           {geom_q1:+.4f} → {geom_q4:+.4f}  (Δ={geom_tendencia:+.4f})")
        print(f"    Convergencia geom:        {'✅ estable' if geom_conv else '⚠️ oscilante'}")
        print(f"    Coherencia (diagnóstico): {M('coh_rel'):+.5f}")
        print(f"    Ω_orient (medio):         {omega_media:+.4f}")
        print(f"    Ω_orient (2ª mitad):      {omega_segunda:+.4f}")
        print(f"    Eficiencia media:         {M('efic'):.4f}")
        print(f"    var_int media:            {var_int_media:.2e}")
        print(f"    LF activa (%):            {100*M('lf'):.1f}%")
        print(f"    Mejor efic explorada:     {explorador.mejor_eficiencia:.4f}")

        if log_fino and log_fino_registros:
            print(f"\n  --- LOG FINO (cada {LOG_FINO_DT}s) ---")
            print(f"    {'t':>6} | {'GED':>10} | {'var_int':>10} | "
                  f"{'ri_std':>8} | {'efic':>8} | {'geom':>7} | {'Ω':>7} | LF")
            for r in log_fino_registros:
                print(f"    {r['t']:>6.1f} | {r['ged_6d']:>10} | "
                      f"{r['var_int']:>10.2e} | {r['ri_std']:>8.4f} | "
                      f"{r['efic']:>8.3f} | {r['geom']:>+7.4f} | "
                      f"{r['omega']:>+7.3f} | {'LF' if r['lf'] else '--'}")

    return {
        'hist': hist,
        'geom_primera': geom_primera, 'geom_segunda': geom_segunda,
        'geom_conv': geom_conv, 'busc_segunda': busc_segunda,
        'grad_medio': grad_medio, 'coh_media': M('coh_rel'),
        'omega_media': omega_media, 'omega_segunda': omega_segunda,
        'geom_tendencia': geom_tendencia,
        'geom_q1': geom_q1, 'geom_q4': geom_q4,
        'var_int_media': var_int_media,
        'mejor_ef': explorador.mejor_eficiencia,
        'log_fino': log_fino_registros,
        'phi_total': Phi_total, 'phi_vel': Phi_vel_total,
        'W_prof': W_prof, 'W_rec': W_rec,
        'Phi_int_historia': Phi_int_historia,
    }

# ============================================================
# PROTOCOLO CÍCLICO
# ============================================================
def correr_protocolo(archivos, clave_entrenamiento, etiqueta,
                     N_CICLOS=5, CICLOS_LOG=None, modo_aud='dir'):
    """
    Fases:
      F2  — voz +60°   (20s)  — log fino siempre
      F3–F12 — varios +60° (20s)
      F13 — voz -60°  (60s)   — extendida (eje 1), log fino siempre
    """
    if CICLOS_LOG is None:
        CICLOS_LOG = {1, 3, 5}

    Phi_total, Phi_vel_total, W_prof, W_rec, \
        Phi_int_historia, explorador = entrenar(
            archivos, 30.0,
            clave_audio=clave_entrenamiento,
            etiqueta=etiqueta,
            modo_aud=modo_aud
        )

    FASES = [
        ("F2",  'voz_pos',         +60.0, 20.0,  "Dominio — voz +60°"),
        ("F3",  'musica_pos',      +60.0, 20.0,  "No entrenado — música +60°"),
        ("F4",  'tono_pos',        +60.0, 20.0,  "No entrenado — tono +60°"),
        ("F5",  'ritmos_pos',      +60.0, 20.0,  "Ritmos irregulares +60°"),
        ("F6",  'ondas_pos',       +60.0, 20.0,  "Ondas mixtas +60°"),
        ("F7",  'pulso_pos',       +60.0, 20.0,  "Pulso logarítmico +60°"),
        ("F8",  'viento_pos',      +60.0, 20.0,  "Viento +60°"),
        ("F9",  'voz_viento1_pos', +60.0, 20.0,  "Degradado — voz+viento1 +60°"),
        ("F10", 'voz_viento2_pos', +60.0, 20.0,  "Degradado — voz+viento2 +60°"),
        ("F11", 'ruido_pos',       +60.0, 20.0,  "Perturbación basal — ruido +60°"),
        ("F12", 'bigbang_pos',     +60.0, 20.0,  "BigBang +60° (20s por ciclo)"),
        ("F13", 'voz_neg',         -60.0, 60.0,  "Re-acoplamiento opuesto — voz -60° [60s]"),
    ]

    punteros     = {clave: 0 for clave in archivos}
    historial_ef = []
    registro     = []

    print()
    print("=" * 80)
    print("PROTOCOLO — " + etiqueta + " — " + str(N_CICLOS) + " ciclos")
    print("=" * 80)

    for ciclo in range(1, N_CICLOS + 1):
        verbose = ciclo in CICLOS_LOG

        if verbose:
            print()
            print("-" * 50)
            print("CICLO " + str(ciclo) + "/" + str(N_CICLOS) + "  [" + etiqueta + "]")
            print("-" * 50)
        else:
            print("  Ciclo " + str(ciclo).rjust(2) + "...", end="", flush=True)

        metricas_ciclo = {}

        for fid, clave, angulo, dur, desc in FASES:
            if clave not in archivos:
                continue

            _, sr, c_L_full, c_R_full = archivos[clave]
            n_pasos = int(dur / DT)
            hop     = int(sr * HOP_FFT_MS    / 1000)
            vent    = int(sr * VENTANA_FFT_MS / 1000)

            inicio_m = punteros[clave] * hop
            if inicio_m + n_pasos * hop > len(c_L_full):
                punteros[clave] = 0
                inicio_m        = 0

            fin_m = inicio_m + n_pasos * hop + vent
            c_L   = c_L_full[inicio_m : min(fin_m, len(c_L_full))]
            c_R   = c_R_full[inicio_m : min(fin_m, len(c_R_full))]

            needed = n_pasos * hop + vent
            if len(c_L) < needed:
                c_L = np.pad(c_L, (0, needed - len(c_L)))
                c_R = np.pad(c_R, (0, needed - len(c_R)))

            punteros[clave] += n_pasos

            # Log fino en F2 y F13 (eje 3)
            log_fino = (fid in ('F2', 'F13'))

            if verbose:
                print("  [" + fid + "] " + desc)

            res = simular_fase(
                Phi_total, Phi_vel_total, W_prof, W_rec,
                Phi_int_historia, historial_ef, explorador,
                sr, c_L, c_R, dur, verbose=verbose,
                modo_aud=modo_aud, fase_id=fid, log_fino=log_fino
            )

            metricas_ciclo[fid] = res
            Phi_total        = res['phi_total']
            Phi_vel_total    = res['phi_vel']
            W_prof           = res['W_prof']
            W_rec            = res['W_rec']
            Phi_int_historia = res['Phi_int_historia']

        def gc(fid, k):
            return metricas_ciclo[fid][k] if fid in metricas_ciclo else None

        grad_f2    = gc('F2',  'grad_medio')
        grad_f13   = gc('F13', 'grad_medio')
        busc_f2    = gc('F2',  'busc_segunda')
        busc_f13   = gc('F13', 'busc_segunda')
        geom_f2    = gc('F2',  'geom_segunda')
        geom_f13   = gc('F13', 'geom_segunda')
        omega_f2   = gc('F2',  'omega_segunda')
        omega_f13  = gc('F13', 'omega_segunda')
        tend_f13   = gc('F13', 'geom_tendencia')   # eje 1
        geom_q1_f13= gc('F13', 'geom_q1')
        geom_q4_f13= gc('F13', 'geom_q4')
        var_int_f2 = gc('F2',  'var_int_media')
        var_int_f13= gc('F13', 'var_int_media')

        mejor_ef = max([v.get('mejor_ef', 0.0)
                        for v in metricas_ciclo.values()
                        if isinstance(v, dict)], default=0.0)

        efic_f2_val  = float(np.mean(metricas_ciclo['F2']['hist']['efic']))  if 'F2'  in metricas_ciclo else 0.0
        efic_f13_val = float(np.mean(metricas_ciclo['F13']['hist']['efic'])) if 'F13' in metricas_ciclo else 0.0

        asys_f2  = (omega_f2  * efic_f2_val)  if omega_f2  is not None else None
        asys_f13 = (omega_f13 * efic_f13_val) if omega_f13 is not None else None

        c31 = (omega_f2 is not None and omega_f13 is not None
               and omega_f2 > 0 and omega_f2 > omega_f13)
        c32 = (asys_f2 is not None and asys_f13 is not None and asys_f2 > asys_f13)

        # Eje 1: ¿tendencia significativa en F13?
        eje1_umbral = (tend_f13 is not None and abs(tend_f13) > 0.005)
        eje1_dir    = "↓" if (tend_f13 is not None and tend_f13 < -0.005) else \
                      "↑" if (tend_f13 is not None and tend_f13 > 0.005) else "="

        registro.append({
            'ciclo': ciclo,
            'grad_f2': grad_f2,   'grad_f13': grad_f13,
            'busc_f2': busc_f2,   'busc_f13': busc_f13,
            'geom_f2': geom_f2,   'geom_f13': geom_f13,
            'omega_f2': omega_f2, 'omega_f13': omega_f13,
            'efic_f2': efic_f2_val, 'efic_f13': efic_f13_val,
            'asys_f2': asys_f2,   'asys_f13': asys_f13,
            'mejor_ef': mejor_ef,
            'tend_f13': tend_f13,
            'geom_q1_f13': geom_q1_f13, 'geom_q4_f13': geom_q4_f13,
            'var_int_f2': var_int_f2, 'var_int_f13': var_int_f13,
            'c31': c31, 'c32': c32,
        })

        def fmt(v):
            return str(round(v, 4)).rjust(7) if v is not None else "    N/A"

        tend_str = f"Δgeom_F13={tend_f13:+.4f}({eje1_dir})" if tend_f13 is not None else ""

        resumen = (
            "Ω F2=" + fmt(omega_f2) + " F13=" + fmt(omega_f13) + " | "
            "geomF13: " + fmt(geom_q1_f13) + "→" + fmt(geom_q4_f13) + " " + tend_str + " | "
            "efic=" + str(round(mejor_ef, 4)) + " | "
            "C31=" + ("OK" if c31 else "--") + " "
            "C32=" + ("OK" if c32 else "--")
        )

        if verbose:
            print("  Ciclo " + str(ciclo).rjust(2) + " — " + resumen)
        else:
            print(" " + resumen)

    return registro, metricas_ciclo  # metricas_ciclo = último ciclo

# ============================================================
# MAIN
# ============================================================
def main():
    archivos = cargar_todos_binaurales('audio_binaural', 35.0)
    if not archivos:
        print("\nERROR: No se encontraron archivos.")
        return

    N_CICLOS   = 5
    CICLOS_LOG = {1, 3, 5}

    # ================================================================
    # INSTANCIA A — entrenada en +60°, evaluada en protocolo estándar
    # ================================================================
    print()
    print("█" * 100)
    print("INSTANCIA A — Entrenada +60°  [referencia]")
    print("█" * 100)
    reg_A, mc_A = correr_protocolo(
        archivos, 'voz_pos', 'A (+60°)', N_CICLOS, CICLOS_LOG
    )

    # ================================================================
    # INSTANCIA B — entrenada en +60°, evaluada con F13=-60°
    # Mismo que en v95: expuesta al opuesto
    # ================================================================
    print()
    print("█" * 100)
    print("INSTANCIA B — Entrenada +60°, expuesta a -60° en F13")
    print("█" * 100)
    reg_B, mc_B = correr_protocolo(
        archivos, 'voz_pos', 'B (+60°→-60°)', N_CICLOS, CICLOS_LOG
    )

    # ================================================================
    # INSTANCIA C — entrenada directamente en -60° (eje 2)
    # Pregunta: ¿el fenómeno es simétrico?
    # ================================================================
    print()
    print("█" * 100)
    print("INSTANCIA C — Entrenada -60°  [EJE 2: ¿fenómeno depende de historia?]")
    print("█" * 100)
    reg_C, mc_C = correr_protocolo(
        archivos, 'voz_neg', 'C (-60°)', N_CICLOS, CICLOS_LOG
    )

    # ================================================================
    # DIAGNÓSTICO FINAL
    # ================================================================
    print()
    print("=" * 100)
    print("DIAGNÓSTICO FINAL — v96")
    print("=" * 100)

    def stats(reg, key):
        vals = [r[key] for r in reg if r.get(key) is not None]
        if not vals:
            return 0.0, 0.0
        return float(np.mean(vals)), float(np.std(vals))

    def c31_ok(reg):
        return sum(r['c31'] for r in reg)

    def c32_ok(reg):
        return sum(r['c32'] for r in reg)

    n = N_CICLOS

    print()
    print(f"  {'Instancia':<20} {'Ω_F2':>8} {'Ω_F13':>8} "
          f"{'geomF13_Q1':>12} {'geomF13_Q4':>12} {'Δgeom_F13':>12} "
          f"{'C31':>5} {'C32':>5}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*5} {'-'*5}")

    for nombre, reg in [('A (+60°)', reg_A), ('B (+60°→-60°)', reg_B), ('C (-60°)', reg_C)]:
        m_f2,  _ = stats(reg, 'omega_f2')
        m_f13, _ = stats(reg, 'omega_f13')
        m_q1,  _ = stats(reg, 'geom_q1_f13')
        m_q4,  _ = stats(reg, 'geom_q4_f13')
        m_tend,_ = stats(reg, 'tend_f13')
        c31 = c31_ok(reg)
        c32 = c32_ok(reg)
        print(f"  {nombre:<20} {m_f2:>+8.4f} {m_f13:>+8.4f} "
              f"{m_q1:>+12.4f} {m_q4:>+12.4f} {m_tend:>+12.4f} "
              f"{c31:>3}/{n} {c32:>3}/{n}")

    print()
    print("  EJE 1 — ¿Rigidez de act_geom tiene umbral? (F13 = 60s)")
    print(f"  {'Instancia':<20} {'Ciclos con Δgeom>0.005':>24}")
    print(f"  {'-'*20} {'-'*24}")
    for nombre, reg in [('A (+60°)', reg_A), ('B (+60°→-60°)', reg_B), ('C (-60°)', reg_C)]:
        ceden = sum(1 for r in reg if r.get('tend_f13') is not None and abs(r['tend_f13']) > 0.005)
        print(f"  {nombre:<20} {ceden}/{n}")

    print()
    print("  EJE 2 — ¿Fenómeno depende del entrenamiento previo?")
    m_f2_B,  _ = stats(reg_B, 'omega_f2')
    m_f2_C,  _ = stats(reg_C, 'omega_f2')
    m_f13_B, _ = stats(reg_B, 'omega_f13')
    m_f13_C, _ = stats(reg_C, 'omega_f13')
    print(f"  B (entrenado +60°): Ω_F2={m_f2_B:+.4f}  Ω_F13={m_f13_B:+.4f}")
    print(f"  C (entrenado -60°): Ω_F2={m_f2_C:+.4f}  Ω_F13={m_f13_C:+.4f}")
    if abs(m_f2_B - m_f2_C) < 0.05:
        print("  → Ω_F2 similar entre B y C — el fenómeno es SIMÉTRICO (independiente del entrenamiento)")
    else:
        print("  → Ω_F2 diferente entre B y C — el entrenamiento previo SÍ importa")

    print()
    print("  EJE 3 — Logging fino de region_int (ver logs de F2 y F13 arriba)")
    m_var_f2_B,  _ = stats(reg_B, 'var_int_f2')
    m_var_f13_B, _ = stats(reg_B, 'var_int_f13')
    m_var_f2_C,  _ = stats(reg_C, 'var_int_f2')
    m_var_f13_C, _ = stats(reg_C, 'var_int_f13')
    print(f"  B — var_int media F2={m_var_f2_B:.2e}  F13={m_var_f13_B:.2e}")
    print(f"  C — var_int media F2={m_var_f2_C:.2e}  F13={m_var_f13_C:.2e}")

    # CSV
    with open('v96_ciclos.csv', 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(['instancia', 'ciclo',
                     'omega_f2', 'omega_f13', 'efic_f2', 'efic_f13',
                     'asys_f2', 'asys_f13', 'mejor_ef',
                     'tend_f13', 'geom_q1_f13', 'geom_q4_f13',
                     'var_int_f2', 'var_int_f13', 'c31', 'c32'])
        for nombre, reg in [('A', reg_A), ('B', reg_B), ('C', reg_C)]:
            for r in reg:
                wr.writerow([nombre, r['ciclo'],
                             r['omega_f2'], r['omega_f13'],
                             r['efic_f2'], r['efic_f13'],
                             r['asys_f2'], r['asys_f13'], r['mejor_ef'],
                             r['tend_f13'], r['geom_q1_f13'], r['geom_q4_f13'],
                             r['var_int_f2'], r['var_int_f13'],
                             int(r['c31']), int(r['c32'])])
    print("\n  CSV guardado: v96_ciclos.csv")

    # Gráfico
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    ciclos_x = list(range(1, N_CICLOS + 1))

    colores = {'A': 'steelblue', 'B': 'firebrick', 'C': 'forestgreen'}
    for row, (nombre, reg) in enumerate([('A', reg_A), ('B', reg_B), ('C', reg_C)]):
        c = colores[nombre]
        ax = axes[row, 0]
        ax.plot(ciclos_x, [r['omega_f2']  for r in reg], 'o-', color=c, lw=2, label='Ω F2')
        ax.plot(ciclos_x, [r['omega_f13'] for r in reg], 's--', color=c, lw=1.5, alpha=0.7, label='Ω F13')
        ax.axhline(0, color='k', lw=0.8, ls='--')
        ax.set_title(f'Omega_orient — Instancia {nombre}')
        ax.set_xlabel('Ciclo')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[row, 1]
        q1_vals = [r.get('geom_q1_f13') or 0 for r in reg]
        q4_vals = [r.get('geom_q4_f13') or 0 for r in reg]
        ax.plot(ciclos_x, q1_vals, 'o-', color=c, lw=2, label='geom Q1 (inicio F13)')
        ax.plot(ciclos_x, q4_vals, 's--', color=c, lw=1.5, alpha=0.7, label='geom Q4 (fin F13)')
        ax.axhline(0, color='k', lw=0.8, ls='--')
        ax.set_title(f'act_geom Q1→Q4 en F13 — Instancia {nombre} [Eje 1]')
        ax.set_xlabel('Ciclo')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('VSTCosmos v96 — Diagnóstico rigidez geométrica\n'
                 'Eje1: umbral act_geom | Eje2: simetría | Eje3: var_int logging fino',
                 fontsize=11)
    plt.tight_layout()
    plt.savefig('v96_diagnostico.png', dpi=150)
    print("  Gráfico guardado: v96_diagnostico.png")

    print()
    print("=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)


if __name__ == "__main__":
    main()