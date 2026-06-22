#!/usr/bin/env python3
"""
VSTCosmos v97.1 — BigBang con diagnóstico post-entrenamiento

Idéntico a v97 EXCEPTO:
  - Tras cada entrenamiento, imprime y guarda diagnóstico de estado interno:
      * Norma de W_prof y W_rec
      * Norma de Phi_total por región
      * Estadísticas de act_geom, act_busc, region_int
  - Tras los tres entrenamientos (antes de cualquier evaluación), calcula y guarda:
      * Distancias entre estados post-entrenamiento:
          ‖Phi_A − Phi_B‖, ‖Phi_A − Phi_C‖, ‖Phi_B − Phi_C‖
      * Lo mismo para W_prof y W_rec
      * Lo mismo restringido a act_geom, act_busc, region_int por separado

Pregunta que resuelve:
  ¿El entrenamiento de 60s deja a B y C en estados internos distinguibles?
  Si Phi_B ≈ Phi_C tras el entrenamiento, la pregunta de v97 era irrespondible.
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import csv
import warnings
import os
import sys
from datetime import datetime

try:
    import soundfile as sf
    HAS_SF = True
except ImportError:
    HAS_SF = False
    print("ERROR: soundfile no disponible. Instalar con: pip install soundfile")
    sys.exit(1)

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACIÓN GLOBAL
# ============================================================
SEED = 42
np.random.seed(SEED)

DURACION_ENTRENAMIENTO = 60.0
DURACION_EVALUACION    = 232.0
N_CICLOS               = 1
WARMUP_OMEGA_SEG       = 10.0
VARIACION_FLOOR        = 1e-4
LOG_FINO_DT            = 0.5
VENTANA_SOSTENIDO_SEG  = 5.0
UMBRAL_CLASIFICACION   = 0.1
VENTANA_FINAL_SEG      = 30.0

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

DIAMETRO_CABEZA  = 0.175
VELOCIDAD_SONIDO = 343.0
ITD_MAX_SEG      = DIAMETRO_CABEZA / VELOCIDAD_SONIDO
F_TRANS_HZ       = VELOCIDAD_SONIDO / DIAMETRO_CABEZA

# ============================================================
# ARQUITECTURA DEL CAMPO EXPANDIDO
# ============================================================
DIM_GANGLIO  = DIM_INTERNA // 2
DIM_AUD      = DIM_GANGLIO
DIM_ACT      = DIM_GANGLIO // 2

DIM_AUD_L    = DIM_AUD
DIM_AUD_R    = DIM_AUD
DIM_ACT_PERM = DIM_ACT
DIM_ACT_GEOM = DIM_ACT
DIM_ACT_BUSC = DIM_ACT
DIM_ACT_MANT = DIM_ACT

BANDA_TRANS = int(DIM_AUD * np.log10(F_TRANS_HZ / F_MIN)
                  / np.log10(F_MAX / F_MIN))
BANDA_TRANS = max(1, min(BANDA_TRANS, DIM_AUD - 1))

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

LOG_FINO_PASOS = int(LOG_FINO_DT / DT)

# ============================================================
# IMPRESIÓN DE CONFIGURACIÓN
# ============================================================
def imprimir_configuracion():
    print("=" * 100)
    print("VSTCosmos v97.1 — BigBang con diagnóstico post-entrenamiento")
    print("=" * 100)
    print()
    print("CONFIGURACIÓN:")
    print(f"  Semilla:                {SEED}")
    print(f"  Entrenamiento:          {DURACION_ENTRENAMIENTO}s por instancia")
    print(f"  Evaluación:             {DURACION_EVALUACION}s continua (BigBang completo)")
    print(f"  Ciclos:                 {N_CICLOS}")
    print(f"  Warmup Ω:               {WARMUP_OMEGA_SEG}s")
    print(f"  Variación floor:        {VARIACION_FLOOR}")
    print(f"  Logging fino:           cada {LOG_FINO_DT}s ({int(DURACION_EVALUACION/LOG_FINO_DT)} muestras/instancia)")
    print(f"  Ventana sostenido:      {VENTANA_SOSTENIDO_SEG}s")
    print(f"  Umbral clasificación:   {UMBRAL_CLASIFICACION}")
    print(f"  Ventana final (H1-H3):  últimos {VENTANA_FINAL_SEG}s")
    print(f"  Reinyección de ruido:   NO (eliminada)")
    print(f"  Reinicio puntero audio: NO (error explícito)")
    print(f"  DIAGNÓSTICO POST-ENTRENAMIENTO: SÍ (nuevo en v97.1)")
    print()
    print("INSTANCIAS:")
    print("  A — Entrenada BigBang +60°, evaluada BigBang +60°  [control de identidad]")
    print("  B — Entrenada BigBang +60°, evaluada BigBang -60°  [inversión post-entrenamiento]")
    print("  C — Entrenada BigBang -60°, evaluada BigBang -60°  [control simétrico]")
    print()
    print("HIPÓTESIS (clasificación post-hoc, últimos 30s sostenidos):")
    print("  H1 mantención:      |Ω_B(t) - Ω_A(t)| < 0.1")
    print("  H2 desorientación:  |Ω_B(t)|          < 0.1")
    print("  H3 reorientación:   |Ω_B(t) - Ω_C(t)| < 0.1")
    print()
    print(f"  BANDA_TRANS={BANDA_TRANS} (F>{F_TRANS_HZ:.0f}Hz)")
    print(f"  DIM_TOTAL={DIM_TOTAL}")
    print(f"  Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)

# ============================================================
# CARGA DE BIGBANG
# ============================================================
def cargar_bigbang(directorio='audio_binaural'):
    archivos = {}
    print(f"\n[Carga] Desde '{directorio}/'...")
    for clave, filename in [
        ('bigbang_pos', 'BigBang_pos60deg.wav'),
        ('bigbang_neg', 'BigBang_neg60deg.wav'),
    ]:
        filepath = os.path.join(directorio, filename)
        if not os.path.exists(filepath):
            print(f"    ❌ {clave}: no encontrado en {filepath}")
            sys.exit(1)
        try:
            data, sr = sf.read(filepath, dtype='float32')
            if data.ndim == 1:
                canal_L = data
                canal_R = data.copy()
            else:
                canal_L = data[:, 0]
                canal_R = data[:, 1] if data.shape[1] > 1 else data[:, 0].copy()
            dur = len(canal_L) / sr
            archivos[clave] = (filepath, sr, canal_L, canal_R)
            print(f"    ✅ {clave:15s} {filename} ({dur:.2f}s, {sr}Hz)")
        except Exception as e:
            print(f"    ❌ {clave}: {e}")
            sys.exit(1)
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
            if efic > self.mejor_eficiencia and not np.isnan(efic):
                self.mejor_eficiencia = efic
                self.mejor_config = (fL, fR, sesgo)
        else:
            self.pasos_en_lf = 0

# ============================================================
# FUNCIONES BASE
# ============================================================
def inicializar_campo():
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
    inicio = idx_paso * hop_muestras
    fin    = inicio + ventana_muestras

    if fin > len(canal):
        raise IndexError(
            f"Audio insuficiente: paso {idx_paso}, fin={fin}, "
            f"len(canal)={len(canal)}. NO se reinicia el puntero."
        )

    segmento = canal[inicio:fin]

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
# COHERENCIA (diagnóstico)
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
# ACT_GEOM
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
# EFICIENCIA
# ============================================================
def calcular_eficiencia(Phi_total, ged_actual):
    region_int = Phi_total[idx['int'][0]:idx['int'][1], :]
    variacion_real = float(np.mean(np.abs(np.diff(region_int, axis=1))))
    if variacion_real < VARIACION_FLOOR:
        return float('nan'), variacion_real
    efic = ged_actual / variacion_real
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

    lf_activa = error_rec > DIFUSION_BASE ** 2

    return (np.clip(Phi_nueva, LIMITE_MIN, LIMITE_MAX),
            np.clip(Phi_vel_n, -5.0, 5.0),
            W_prof, W_rec, Phi_int_historia,
            lf_activa, error_rec, coherencia)

# ============================================================
# DIAGNÓSTICO POST-ENTRENAMIENTO (NUEVO EN v97.1)
# ============================================================
def diagnosticar_estado_post_entrenamiento(Phi_total, W_prof, W_rec, etiqueta):
    """
    Imprime diagnóstico del estado interno tras el entrenamiento.
    Retorna un dict con métricas para comparación posterior.
    """
    diag = {}

    # Normas globales
    diag['W_prof_norm']  = float(np.mean(np.abs(W_prof)))
    diag['W_rec_norm']   = float(np.mean(np.abs(W_rec)))
    diag['W_prof_max']   = float(np.max(np.abs(W_prof)))
    diag['W_rec_max']    = float(np.max(np.abs(W_rec)))
    diag['Phi_norm']     = float(np.mean(np.abs(Phi_total - PHI_EQUILIBRIO)))

    # Por región
    for region in ['int', 'G', 'aud_L', 'aud_R',
                   'act_perm', 'act_geom', 'act_busc', 'act_mant']:
        i0, i1 = idx[region]
        slice_phi = Phi_total[i0:i1, :]
        diag[f'{region}_mean']  = float(np.mean(slice_phi))
        diag[f'{region}_std']   = float(np.std(slice_phi))
        diag[f'{region}_norm']  = float(np.mean(np.abs(slice_phi - PHI_EQUILIBRIO)))

    # act_geom específico (relevante para Ω)
    ag0, ag1 = idx['act_geom']
    geom = Phi_total[ag0:ag1, :]
    diag['act_geom_tanh_mean'] = float(np.mean(np.tanh(geom)))
    mitad = max(1, (ag1 - ag0) // 2)
    diag['act_geom_baja_mean']  = float(np.mean(geom[:mitad, :]))
    diag['act_geom_alta_mean']  = float(np.mean(geom[mitad:, :]))
    diag['act_geom_asimetria']  = diag['act_geom_alta_mean'] - diag['act_geom_baja_mean']

    # Guardar referencias completas para distancias posteriores
    diag['_Phi']    = Phi_total.copy()
    diag['_W_prof'] = W_prof.copy()
    diag['_W_rec']  = W_rec.copy()

    print(f"\n  [DIAGNÓSTICO POST-ENTRENAMIENTO — Instancia {etiqueta}]")
    print(f"    W_prof: media|·|={diag['W_prof_norm']:.6f}  max|·|={diag['W_prof_max']:.6f}")
    print(f"    W_rec:  media|·|={diag['W_rec_norm']:.6f}  max|·|={diag['W_rec_max']:.6f}")
    print(f"    Phi global: media|·-eq|={diag['Phi_norm']:.6f}")
    print(f"    Por región (media):")
    print(f"      int:      {diag['int_mean']:+.6f}  (std={diag['int_std']:.4f})")
    print(f"      G:        {diag['G_mean']:+.6f}  (std={diag['G_std']:.4f})")
    print(f"      aud_L:    {diag['aud_L_mean']:+.6f}  aud_R: {diag['aud_R_mean']:+.6f}")
    print(f"      act_perm: {diag['act_perm_mean']:+.6f}  act_geom: {diag['act_geom_mean']:+.6f}")
    print(f"      act_busc: {diag['act_busc_mean']:+.6f}  act_mant: {diag['act_mant_mean']:+.6f}")
    print(f"    act_geom (clave para Ω):")
    print(f"      tanh medio:    {diag['act_geom_tanh_mean']:+.6f}")
    print(f"      mitad baja:    {diag['act_geom_baja_mean']:+.6f}")
    print(f"      mitad alta:    {diag['act_geom_alta_mean']:+.6f}")
    print(f"      asimetría:     {diag['act_geom_asimetria']:+.6f}")

    return diag

def comparar_estados_entre_instancias(diag_A, diag_B, diag_C):
    """
    Calcula distancias entre los tres estados post-entrenamiento.
    Pregunta clave: ¿el entrenamiento dejó a A, B, C en estados distinguibles?
    """
    print("\n" + "=" * 100)
    print("COMPARACIÓN DE ESTADOS POST-ENTRENAMIENTO (antes de evaluación)")
    print("=" * 100)

    pares = [('A', 'B', diag_A, diag_B),
             ('A', 'C', diag_A, diag_C),
             ('B', 'C', diag_B, diag_C)]

    resultados = {}

    for n1, n2, d1, d2 in pares:
        key = f'{n1}_vs_{n2}'
        # Distancia euclidiana en Phi_total completo
        dist_phi    = float(np.linalg.norm(d1['_Phi'] - d2['_Phi']))
        # Distancia normalizada por número de elementos
        n_elem      = d1['_Phi'].size
        dist_phi_n  = dist_phi / np.sqrt(n_elem)

        dist_wprof  = float(np.linalg.norm(d1['_W_prof'] - d2['_W_prof']))
        dist_wrec   = float(np.linalg.norm(d1['_W_rec']  - d2['_W_rec']))

        # Distancia restringida a regiones críticas
        dist_geom = float(np.linalg.norm(
            d1['_Phi'][idx['act_geom'][0]:idx['act_geom'][1], :] -
            d2['_Phi'][idx['act_geom'][0]:idx['act_geom'][1], :]
        ))
        dist_busc = float(np.linalg.norm(
            d1['_Phi'][idx['act_busc'][0]:idx['act_busc'][1], :] -
            d2['_Phi'][idx['act_busc'][0]:idx['act_busc'][1], :]
        ))
        dist_int = float(np.linalg.norm(
            d1['_Phi'][idx['int'][0]:idx['int'][1], :] -
            d2['_Phi'][idx['int'][0]:idx['int'][1], :]
        ))

        # Diferencia en métricas escalares clave
        d_geom_asim = abs(d1['act_geom_asimetria'] - d2['act_geom_asimetria'])
        d_geom_tanh = abs(d1['act_geom_tanh_mean'] - d2['act_geom_tanh_mean'])

        resultados[key] = {
            'dist_phi':       dist_phi,
            'dist_phi_norm':  dist_phi_n,
            'dist_wprof':     dist_wprof,
            'dist_wrec':      dist_wrec,
            'dist_geom':      dist_geom,
            'dist_busc':      dist_busc,
            'dist_int':       dist_int,
            'd_geom_asim':    d_geom_asim,
            'd_geom_tanh':    d_geom_tanh,
        }

        print(f"\n  {n1} vs {n2}:")
        print(f"    ‖Phi_{n1} - Phi_{n2}‖         = {dist_phi:.6f}  (normalizada: {dist_phi_n:.6f})")
        print(f"    ‖W_prof_{n1} - W_prof_{n2}‖   = {dist_wprof:.6f}")
        print(f"    ‖W_rec_{n1}  - W_rec_{n2}‖    = {dist_wrec:.6f}")
        print(f"    ‖Phi[act_geom]‖ ({n1} vs {n2}) = {dist_geom:.6f}")
        print(f"    ‖Phi[act_busc]‖ ({n1} vs {n2}) = {dist_busc:.6f}")
        print(f"    ‖Phi[int]‖      ({n1} vs {n2}) = {dist_int:.6f}")
        print(f"    Δ act_geom_asimetria        = {d_geom_asim:.6f}")
        print(f"    Δ act_geom_tanh_mean        = {d_geom_tanh:.6f}")

    # Diagnóstico interpretativo
    print("\n  DIAGNÓSTICO INTERPRETATIVO:")
    bc = resultados['B_vs_C']
    ab = resultados['A_vs_B']
    ac = resultados['A_vs_C']

    print(f"    Pregunta clave: ¿B y C están en estados distinguibles tras el entrenamiento?")
    print(f"    ‖Phi_B - Phi_C‖ normalizada = {bc['dist_phi_norm']:.6f}")

    if bc['dist_phi_norm'] < 0.001:
        print(f"    → B y C son INDISTINGUIBLES (dist < 0.001).")
        print(f"      El entrenamiento NO produjo separación entre las dos condiciones.")
        print(f"      La pregunta de v97 era IRRESPONDIBLE por construcción.")
    elif bc['dist_phi_norm'] < 0.01:
        print(f"    → B y C están MUY POCO separados (dist < 0.01).")
        print(f"      El entrenamiento produjo separación marginal.")
    elif bc['dist_phi_norm'] < 0.1:
        print(f"    → B y C están MODERADAMENTE separados (0.01 < dist < 0.1).")
        print(f"      Hay separación detectable, pero potencialmente erosionable por la dinámica.")
    else:
        print(f"    → B y C están BIEN separados (dist > 0.1).")
        print(f"      El entrenamiento estableció estados internos distinguibles.")

    print(f"\n    Comparación de magnitudes:")
    print(f"      A vs B (mismo entrenamiento, +60°): {ab['dist_phi_norm']:.6f}")
    print(f"      A vs C (entrenamientos opuestos):   {ac['dist_phi_norm']:.6f}")
    print(f"      B vs C (entrenamientos opuestos):   {bc['dist_phi_norm']:.6f}")
    print(f"    Si A-B (mismo entrenamiento) ≈ B-C (opuesto), el entrenamiento es irrelevante.")

    return resultados

# ============================================================
# ENTRENAMIENTO
# ============================================================
def entrenar(archivos, clave_audio, etiqueta, duracion=DURACION_ENTRENAMIENTO,
             modo_aud='dir'):
    print(f"\n[Entrenamiento] {etiqueta} con {clave_audio} ({duracion}s, modo={modo_aud})")
    Phi_total, Phi_vel_total = inicializar_campo()
    W_prof, W_rec, Phi_int_historia = inicializar_memorias()
    explorador = ExploradorActuadores()

    _, sr, c_L, c_R = archivos[clave_audio]
    vent    = int(sr * VENTANA_FFT_MS / 1000)
    hop     = int(sr * HOP_FFT_MS  / 1000)
    n_pasos = int(duracion / DT)

    if (n_pasos - 1) * hop + vent > len(c_L):
        raise ValueError(
            f"Audio {clave_audio} insuficiente para entrenamiento de {duracion}s. "
            f"Necesario: {(n_pasos-1)*hop + vent} muestras. Disponible: {len(c_L)}."
        )

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

        if paso % 1000 == 0:
            print(f"    Paso {paso}/{n_pasos} ({paso*DT:.1f}s), error={error_rec:.6f}")

    print(f"  ERROR_EQUILIBRIO: {min(errores):.6f}")
    print(f"  W_prof: {np.mean(np.abs(W_prof)):.4f}")
    print(f"  W_rec:  {np.mean(np.abs(W_rec)):.4f}")

    return Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia, explorador

# ============================================================
# EVALUACIÓN
# ============================================================
def evaluar(Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia,
            explorador, archivos, clave_eval, etiqueta_inst,
            duracion=DURACION_EVALUACION, modo_aud='dir',
            csv_log_path=None, csv_estimulo_path=None):
    print(f"\n[Evaluación] {etiqueta_inst} con {clave_eval} ({duracion}s)")

    _, sr, c_L, c_R = archivos[clave_eval]
    vent    = int(sr * VENTANA_FFT_MS / 1000)
    hop     = int(sr * HOP_FFT_MS  / 1000)
    n_pasos = int(duracion / DT)

    audio_necesario = (n_pasos - 1) * hop + vent
    if audio_necesario > len(c_L):
        raise ValueError(
            f"Audio {clave_eval} insuficiente para evaluación de {duracion}s. "
            f"Necesario: {audio_necesario} muestras ({audio_necesario/sr:.2f}s). "
            f"Disponible: {len(c_L)} muestras ({len(c_L)/sr:.2f}s)."
        )
    print(f"  Audio disponible: {len(c_L)/sr:.2f}s. Audio requerido: {audio_necesario/sr:.2f}s. OK.")

    f_log = open(csv_log_path, 'w', newline='', encoding='utf-8')
    w_log = csv.writer(f_log)
    w_log.writerow([
        't', 'ged', 'var_int', 'efic', 'geom', 'busc', 'omega',
        'gradE', 'coh', 'lf', 'warmup'
    ])

    f_est = open(csv_estimulo_path, 'w', newline='', encoding='utf-8')
    w_est = csv.writer(f_est)
    w_est.writerow([
        't', 'energia_L_baja', 'energia_L_alta',
        'energia_R_baja', 'energia_R_alta', 'gradE_calc'
    ])

    historial_ef = []
    gradiente_hist_fase = []
    lf_prev = False

    omega_traj = []
    t_traj     = []

    print(f"  {'t(s)':>6} | {'GED':>9} | {'var_int':>9} | {'efic':>8} | "
          f"{'geom':>7} | {'Ω':>7} | {'gradE':>7} | LF | warmup")

    n_pasos_warmup = int(WARMUP_OMEGA_SEG / DT)

    for paso in range(n_pasos):
        t_actual = paso * DT
        en_warmup = paso < n_pasos_warmup

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

        fL, fR, sf_v, _, _ = calcular_parametros_actuacion(Phi_total)

        a_L   = Phi_total[idx['aud_L'][0]:idx['aud_L'][1], :]
        a_R   = Phi_total[idx['aud_R'][0]:idx['aud_R'][1], :]
        ged_L = calcular_ged_entre(region_int, a_L)
        ged_R = calcular_ged_entre(region_int, a_R)
        ged   = (ged_L + ged_R) / 2.0

        efic, var_int_real = calcular_eficiencia(Phi_total, ged)
        if not np.isnan(efic):
            historial_ef.append(efic)
        if len(historial_ef) > TAU_EFICIENCIA * 2:
            historial_ef.pop(0)

        explorador.actualizar(lf_prev, efic if not np.isnan(efic) else 0.0,
                              fL, fR, sf_v)

        Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia, \
            lf_activa, error_rec, _ = actualizar_campo(
                Phi_total, Phi_vel_total, W_prof, W_rec,
                Phi_int_historia, obj_L, obj_R, fL, fR, sf_v, DT,
                modo_aud=modo_aud
            )

        Phi_total = explorar_actuadores(Phi_total, explorador, lf_activa, efic, DT)
        lf_prev   = lf_activa

        act_busc_val = calcular_senal_busqueda(Phi_total)
        geom = float(np.mean(np.tanh(
            Phi_total[idx['act_geom'][0]:idx['act_geom'][1], :]
        )))
        omega = calcular_omega_orient(Phi_total, gradiente_hist_fase)

        omega_traj.append(omega)
        t_traj.append(t_actual)

        if paso % LOG_FINO_PASOS == 0:
            efic_str = f"{efic:>8.4f}" if not np.isnan(efic) else "     NaN"
            print(f"  {t_actual:>6.1f} | {ged:>9.6f} | {var_int_real:>9.2e} | "
                  f"{efic_str} | {geom:>+7.4f} | {omega:>+7.3f} | "
                  f"{gradiente_E:>+7.3f} | {'LF' if lf_activa else '--'} | "
                  f"{'WARMUP' if en_warmup else '------'}")
            sys.stdout.flush()

            w_log.writerow([
                f"{t_actual:.4f}",
                f"{ged:.8f}",
                f"{var_int_real:.4e}",
                f"{efic:.6f}" if not np.isnan(efic) else "NaN",
                f"{geom:.6f}",
                f"{act_busc_val:.6f}",
                f"{omega:.6f}",
                f"{gradiente_E:.6f}",
                f"{coh_rel:.6f}",
                int(lf_activa),
                int(en_warmup),
            ])

            n_mid = DIM_AUD // 2
            e_L_baja = float(np.mean(obj_L[:n_mid, :] ** 2))
            e_L_alta = float(np.mean(obj_L[n_mid:, :] ** 2))
            e_R_baja = float(np.mean(obj_R[:n_mid, :] ** 2))
            e_R_alta = float(np.mean(obj_R[n_mid:, :] ** 2))
            w_est.writerow([
                f"{t_actual:.4f}",
                f"{e_L_baja:.6f}", f"{e_L_alta:.6f}",
                f"{e_R_baja:.6f}", f"{e_R_alta:.6f}",
                f"{gradiente_E:.6f}",
            ])

    f_log.close()
    f_est.close()

    print(f"  Logs guardados: {csv_log_path}, {csv_estimulo_path}")

    return {
        't_traj':     np.array(t_traj),
        'omega_traj': np.array(omega_traj),
        'phi_total':  Phi_total,
        'W_prof':     W_prof,
        'W_rec':      W_rec,
    }

# ============================================================
# ANÁLISIS POST-HOC
# ============================================================
def clasificar_hipotesis(t_A, omega_A, t_B, omega_B, t_C, omega_C):
    t_max = min(t_A[-1], t_B[-1], t_C[-1])
    t_ini_ventana = t_max - VENTANA_FINAL_SEG

    mask_A = (t_A >= t_ini_ventana) & (t_A <= t_max)
    mask_B = (t_B >= t_ini_ventana) & (t_B <= t_max)
    mask_C = (t_C >= t_ini_ventana) & (t_C <= t_max)

    om_A_v = omega_A[mask_A]
    om_B_v = omega_B[mask_B]
    om_C_v = omega_C[mask_C]

    n_min = min(len(om_A_v), len(om_B_v), len(om_C_v))
    om_A_v = om_A_v[-n_min:]
    om_B_v = om_B_v[-n_min:]
    om_C_v = om_C_v[-n_min:]

    diff_BA = np.abs(om_B_v - om_A_v)
    abs_B   = np.abs(om_B_v)
    diff_BC = np.abs(om_B_v - om_C_v)

    H1 = float(np.mean(diff_BA)) < UMBRAL_CLASIFICACION
    H2 = float(np.mean(abs_B))   < UMBRAL_CLASIFICACION
    H3 = float(np.mean(diff_BC)) < UMBRAL_CLASIFICACION

    return {
        'mean_diff_BA': float(np.mean(diff_BA)),
        'mean_abs_B':   float(np.mean(abs_B)),
        'mean_diff_BC': float(np.mean(diff_BC)),
        'H1_mantencion':     H1,
        'H2_desorientacion': H2,
        'H3_reorientacion':  H3,
        'omega_A_final': float(np.mean(om_A_v)),
        'omega_B_final': float(np.mean(om_B_v)),
        'omega_C_final': float(np.mean(om_C_v)),
    }

def calcular_tiempo_rendicion(t_A, omega_A, t_B, omega_B, t_C, omega_C):
    omega_A_interp = np.interp(t_B, t_A, omega_A)
    omega_C_interp = np.interp(t_B, t_C, omega_C)

    dist_a_A = np.abs(omega_B - omega_A_interp)
    dist_a_C = np.abs(omega_B - omega_C_interp)
    rendido = dist_a_C < dist_a_A

    mask_post_warmup = t_B >= WARMUP_OMEGA_SEG
    rendido = rendido & mask_post_warmup

    n_sostenido = int(VENTANA_SOSTENIDO_SEG / LOG_FINO_DT)
    for i in range(len(rendido) - n_sostenido):
        if np.all(rendido[i:i + n_sostenido]):
            return float(t_B[i])
    return None

# ============================================================
# MAIN
# ============================================================
def main():
    imprimir_configuracion()

    archivos = cargar_bigbang('audio_binaural')

    os.makedirs('v97_1_logs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # ================================================================
    # ENTRENAMIENTO DE LAS TRES INSTANCIAS (con diagnóstico)
    # ================================================================
    print()
    print("█" * 100)
    print("FASE 1 — ENTRENAMIENTO DE LAS TRES INSTANCIAS")
    print("█" * 100)

    # Instancia A
    print()
    print("-" * 100)
    print("INSTANCIA A — Entrenada BigBang +60°")
    print("-" * 100)
    np.random.seed(SEED)
    Phi_A, Phi_vel_A, W_prof_A, W_rec_A, Phi_int_A, expl_A = \
        entrenar(archivos, 'bigbang_pos', 'A')
    diag_A = diagnosticar_estado_post_entrenamiento(Phi_A, W_prof_A, W_rec_A, 'A')

    # Instancia B
    print()
    print("-" * 100)
    print("INSTANCIA B — Entrenada BigBang +60° (igual que A pero será evaluada con -60°)")
    print("-" * 100)
    np.random.seed(SEED)
    Phi_B, Phi_vel_B, W_prof_B, W_rec_B, Phi_int_B, expl_B = \
        entrenar(archivos, 'bigbang_pos', 'B')
    diag_B = diagnosticar_estado_post_entrenamiento(Phi_B, W_prof_B, W_rec_B, 'B')

    # Instancia C
    print()
    print("-" * 100)
    print("INSTANCIA C — Entrenada BigBang -60°")
    print("-" * 100)
    np.random.seed(SEED)
    Phi_C, Phi_vel_C, W_prof_C, W_rec_C, Phi_int_C, expl_C = \
        entrenar(archivos, 'bigbang_neg', 'C')
    diag_C = diagnosticar_estado_post_entrenamiento(Phi_C, W_prof_C, W_rec_C, 'C')

    # ================================================================
    # COMPARACIÓN INTER-INSTANCIA (NUEVO EN v97.1)
    # ================================================================
    distancias = comparar_estados_entre_instancias(diag_A, diag_B, diag_C)

    # Guardar diagnóstico a archivo
    diag_path = f'v97_1_logs/v97_1_diagnostico_{timestamp}.txt'
    with open(diag_path, 'w', encoding='utf-8') as f:
        f.write(f"VSTCosmos v97.1 — Diagnóstico post-entrenamiento\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Seed: {SEED}\n\n")

        for nombre, d in [('A', diag_A), ('B', diag_B), ('C', diag_C)]:
            f.write(f"=== Instancia {nombre} ===\n")
            for k, v in d.items():
                if not k.startswith('_'):
                    f.write(f"  {k}: {v}\n")
            f.write("\n")

        f.write("=== Distancias entre instancias ===\n")
        for par, d in distancias.items():
            f.write(f"\n{par}:\n")
            for k, v in d.items():
                f.write(f"  {k}: {v}\n")

    print(f"\n  Diagnóstico guardado: {diag_path}")

    # ================================================================
    # FASE 2 — EVALUACIÓN DE LAS TRES INSTANCIAS
    # ================================================================
    print()
    print("█" * 100)
    print("FASE 2 — EVALUACIÓN DE LAS TRES INSTANCIAS")
    print("█" * 100)

    # Evaluación A
    print()
    print("-" * 100)
    print("INSTANCIA A — Evaluada BigBang +60°  [identidad]")
    print("-" * 100)
    res_A = evaluar(
        Phi_A, Phi_vel_A, W_prof_A, W_rec_A, Phi_int_A, expl_A,
        archivos, 'bigbang_pos', 'A',
        csv_log_path=f'v97_1_logs/v97_1_A_log_{timestamp}.csv',
        csv_estimulo_path=f'v97_1_logs/v97_1_A_estimulo_{timestamp}.csv',
    )

    # Evaluación B
    print()
    print("-" * 100)
    print("INSTANCIA B — Evaluada BigBang -60°  [inversión]")
    print("-" * 100)
    res_B = evaluar(
        Phi_B, Phi_vel_B, W_prof_B, W_rec_B, Phi_int_B, expl_B,
        archivos, 'bigbang_neg', 'B',
        csv_log_path=f'v97_1_logs/v97_1_B_log_{timestamp}.csv',
        csv_estimulo_path=f'v97_1_logs/v97_1_B_estimulo_{timestamp}.csv',
    )

    # Evaluación C
    print()
    print("-" * 100)
    print("INSTANCIA C — Evaluada BigBang -60°  [simetría]")
    print("-" * 100)
    res_C = evaluar(
        Phi_C, Phi_vel_C, W_prof_C, W_rec_C, Phi_int_C, expl_C,
        archivos, 'bigbang_neg', 'C',
        csv_log_path=f'v97_1_logs/v97_1_C_log_{timestamp}.csv',
        csv_estimulo_path=f'v97_1_logs/v97_1_C_estimulo_{timestamp}.csv',
    )

    # ================================================================
    # ANÁLISIS POST-HOC
    # ================================================================
    print()
    print("=" * 100)
    print("ANÁLISIS POST-HOC")
    print("=" * 100)

    clasif = clasificar_hipotesis(
        res_A['t_traj'], res_A['omega_traj'],
        res_B['t_traj'], res_B['omega_traj'],
        res_C['t_traj'], res_C['omega_traj'],
    )

    print(f"\nVentana de clasificación: últimos {VENTANA_FINAL_SEG}s")
    print(f"  Ω_A medio (final):  {clasif['omega_A_final']:+.4f}")
    print(f"  Ω_B medio (final):  {clasif['omega_B_final']:+.4f}")
    print(f"  Ω_C medio (final):  {clasif['omega_C_final']:+.4f}")
    print()
    print(f"  |Ω_B - Ω_A| medio:  {clasif['mean_diff_BA']:.4f}")
    print(f"  |Ω_B|       medio:  {clasif['mean_abs_B']:.4f}")
    print(f"  |Ω_B - Ω_C| medio:  {clasif['mean_diff_BC']:.4f}")
    print()
    print(f"  H1 (mantención):     {'✅ SE CUMPLE' if clasif['H1_mantencion']     else '❌ no se cumple'}")
    print(f"  H2 (desorientación): {'✅ SE CUMPLE' if clasif['H2_desorientacion'] else '❌ no se cumple'}")
    print(f"  H3 (reorientación):  {'✅ SE CUMPLE' if clasif['H3_reorientacion']  else '❌ no se cumple'}")

    cumplidas = sum([clasif['H1_mantencion'], clasif['H2_desorientacion'],
                     clasif['H3_reorientacion']])
    if cumplidas == 0:
        print("\n  → INDETERMINADO: ninguna hipótesis se cumple bajo umbral 0.1.")
    elif cumplidas == 1:
        print("\n  → CLASIFICACIÓN ÚNICA.")
    else:
        print(f"\n  → MÚLTIPLES HIPÓTESIS CUMPLIDAS ({cumplidas}). Posible solapamiento.")

    t_rend = calcular_tiempo_rendicion(
        res_A['t_traj'], res_A['omega_traj'],
        res_B['t_traj'], res_B['omega_traj'],
        res_C['t_traj'], res_C['omega_traj'],
    )

    print()
    if t_rend is not None:
        print(f"  Tiempo de rendición:  t = {t_rend:.2f}s")
    else:
        print(f"  Tiempo de rendición:  NO ocurre en {DURACION_EVALUACION}s")

    # ================================================================
    # LECTURA INTEGRADA DIAGNÓSTICO + RESULTADO
    # ================================================================
    print()
    print("=" * 100)
    print("LECTURA INTEGRADA — Diagnóstico vs Resultado")
    print("=" * 100)

    bc_dist = distancias['B_vs_C']['dist_phi_norm']
    ac_dist = distancias['A_vs_C']['dist_phi_norm']
    ab_dist = distancias['A_vs_B']['dist_phi_norm']

    print(f"\n  Estados post-entrenamiento (dist normalizada):")
    print(f"    A vs B (mismo entrenamiento): {ab_dist:.6f}")
    print(f"    A vs C (entren. opuesto):     {ac_dist:.6f}")
    print(f"    B vs C (entren. opuesto):     {bc_dist:.6f}")

    print(f"\n  Trayectorias de Ω (final):")
    print(f"    Ω_A = {clasif['omega_A_final']:+.4f}")
    print(f"    Ω_B = {clasif['omega_B_final']:+.4f}")
    print(f"    Ω_C = {clasif['omega_C_final']:+.4f}")

    print(f"\n  Interpretación:")
    if bc_dist < 0.001:
        print(f"  → B y C ya eran INDISTINGUIBLES tras el entrenamiento.")
        print(f"    Que Ω_B converja a Ω_C NO es 'reorientación', es 'mismo cálculo'.")
        print(f"    H3 cumplido es un artefacto de B y C arrancar en el mismo estado.")
    elif bc_dist < 0.01 and clasif['mean_diff_BC'] < 0.01:
        print(f"  → B y C arrancaban poco separados y la dinámica los igualó.")
        print(f"    El entrenamiento estableció separación marginal, y la evaluación la borró.")
        print(f"    H3 cumplido refleja erosión de memoria, no reorientación activa.")
    elif bc_dist > 0.1 and clasif['mean_diff_BC'] < 0.1:
        print(f"  → B y C arrancaban BIEN separados y convergieron.")
        print(f"    Esto sí es reorientación genuina: la dinámica del estímulo presente")
        print(f"    superó la traza del entrenamiento previo.")
    else:
        print(f"  → Resultado mixto. Inspeccionar gráfico de trayectorias.")

    # ================================================================
    # GUARDAR RESUMEN
    # ================================================================
    resumen_path = f'v97_1_logs/v97_1_resumen_{timestamp}.txt'
    with open(resumen_path, 'w', encoding='utf-8') as f:
        f.write(f"VSTCosmos v97.1 — Resumen\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Seed: {SEED}\n")
        f.write(f"Entrenamiento: {DURACION_ENTRENAMIENTO}s\n")
        f.write(f"Evaluación: {DURACION_EVALUACION}s\n\n")
        f.write(f"--- Distancias post-entrenamiento (normalizadas) ---\n")
        f.write(f"A vs B: {ab_dist:.6f}\n")
        f.write(f"A vs C: {ac_dist:.6f}\n")
        f.write(f"B vs C: {bc_dist:.6f}\n\n")
        f.write(f"--- Resultados de evaluación ---\n")
        f.write(f"Ω_A final: {clasif['omega_A_final']:+.6f}\n")
        f.write(f"Ω_B final: {clasif['omega_B_final']:+.6f}\n")
        f.write(f"Ω_C final: {clasif['omega_C_final']:+.6f}\n\n")
        f.write(f"|Ω_B - Ω_A| medio: {clasif['mean_diff_BA']:.6f}\n")
        f.write(f"|Ω_B|       medio: {clasif['mean_abs_B']:.6f}\n")
        f.write(f"|Ω_B - Ω_C| medio: {clasif['mean_diff_BC']:.6f}\n\n")
        f.write(f"H1 mantención:     {clasif['H1_mantencion']}\n")
        f.write(f"H2 desorientación: {clasif['H2_desorientacion']}\n")
        f.write(f"H3 reorientación:  {clasif['H3_reorientacion']}\n\n")
        f.write(f"Tiempo de rendición: {t_rend if t_rend else 'NO OCURRE'}\n")

    print(f"\n  Resumen guardado: {resumen_path}")

    # ================================================================
    # GRÁFICO
    # ================================================================
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax = axes[0]
    ax.plot(res_A['t_traj'], res_A['omega_traj'], color='steelblue',
            lw=1.2, alpha=0.8, label='A (+60° → +60°)')
    ax.plot(res_B['t_traj'], res_B['omega_traj'], color='firebrick',
            lw=1.5, alpha=0.9, label='B (+60° → -60°)')
    ax.plot(res_C['t_traj'], res_C['omega_traj'], color='forestgreen',
            lw=1.2, alpha=0.8, label='C (-60° → -60°)')
    ax.axhline(0, color='k', lw=0.5, ls='--', alpha=0.5)
    ax.axvline(WARMUP_OMEGA_SEG, color='gray', lw=0.5, ls=':',
               alpha=0.7, label=f'fin warmup ({WARMUP_OMEGA_SEG}s)')
    if t_rend is not None:
        ax.axvline(t_rend, color='red', lw=1.0, ls='-.',
                   alpha=0.7, label=f'rendición t={t_rend:.1f}s')
    ax.set_xlabel('t (s)')
    ax.set_ylabel('Ω_orient')
    ax.set_title(f'VSTCosmos v97.1 — Ω_orient (BigBang completo)\n'
                 f'B-C dist post-entrenamiento: {bc_dist:.6f}')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    om_A_int = np.interp(res_B['t_traj'], res_A['t_traj'], res_A['omega_traj'])
    om_C_int = np.interp(res_B['t_traj'], res_C['t_traj'], res_C['omega_traj'])
    ax.plot(res_B['t_traj'], np.abs(res_B['omega_traj'] - om_A_int),
            color='steelblue', lw=1.2, label='|Ω_B - Ω_A|')
    ax.plot(res_B['t_traj'], np.abs(res_B['omega_traj'] - om_C_int),
            color='forestgreen', lw=1.2, label='|Ω_B - Ω_C|')
    ax.axhline(UMBRAL_CLASIFICACION, color='red', lw=0.8, ls='--',
               label=f'umbral={UMBRAL_CLASIFICACION}')
    ax.set_xlabel('t (s)')
    ax.set_ylabel('distancia')
    ax.set_title('Distancia de B a controles A y C')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    grafico_path = f'v97_1_logs/v97_1_omega_{timestamp}.png'
    plt.savefig(grafico_path, dpi=150)
    print(f"  Gráfico guardado: {grafico_path}")

    print()
    print("=" * 100)
    print(f"EXPERIMENTO COMPLETADO — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)


if __name__ == "__main__":
    main()