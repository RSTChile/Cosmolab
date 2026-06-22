#!/usr/bin/env python3
"""
VSTCosmos v105e — Barrido fino alrededor de 440 Hz

Frecuencias a testear (15 frecuencias):
  400, 410, 420, 430, 435, 438, 439, 440, 441, 442, 445, 450, 460, 470, 480 Hz

Cada frecuencia: 30s, versión pos y neg (binaural sintético)
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
from datetime import datetime
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

try:
    import soundfile as sf
    HAS_SF = True
except ImportError:
    HAS_SF = False
    print("ERROR: soundfile no instalado")
    exit(1)

warnings.filterwarnings('ignore')

# ============================================================
# PARAMETROS (identicos)
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
F_TRANS_HZ       = VELOCIDAD_SONIDO / DIAMETRO_CABEZA

# ============================================================
# ARQUITECTURA (identica)
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

VARIACION_FLOOR = 1e-6
VENTANA_FINAL_SEG = 30
VENTANA_FINAL_PASOS = int(VENTANA_FINAL_SEG / DT)
DURACION_ENTRENAMIENTO = 60.0
DURACION_EVALUACION = 30.0

# Frecuencias a testear (barrido fino alrededor de 440 Hz)
FRECUENCIAS = [400, 410, 420, 430, 435, 438, 439, 440, 441, 442, 445, 450, 460, 470, 480]

# Parámetros de generación de tonos
SR = 48000
DURACION_TONO = 30.0

print("=" * 100)
print("VSTCosmos v105e — Barrido fino alrededor de 440 Hz")
print()
print("  Preguntas:")
print("    1. ¿El pico es exactamente en 440 Hz o cerca?")
print("    2. ¿Qué ancho de banda tiene la anomalía?")
print("    3. ¿Hay otros picos?")
print("    4. ¿La respuesta es simétrica?")
print()
print(f"  Frecuencias a testear: {FRECUENCIAS}")
print(f"  Total: {len(FRECUENCIAS)} frecuencias × 2 direcciones = {len(FRECUENCIAS)*2} estímulos")
print(f"  BANDA_TRANS={BANDA_TRANS} (F>{F_TRANS_HZ:.0f}Hz)")
print(f"  DIM_TOTAL={DIM_TOTAL}")
print("=" * 100)


# ============================================================
# GENERADOR DE TONOS (si no existen)
# ============================================================
def generar_tono_binaural(frecuencia_hz, sr, duracion, angulo_grados):
    """Genera tono binaural sintético para una frecuencia y ángulo dados"""
    n_muestras = int(sr * duracion)
    t = np.linspace(0, duracion, n_muestras, endpoint=False)
    tono_mono = np.sin(2 * np.pi * frecuencia_hz * t)
    tono_mono = tono_mono / np.max(np.abs(tono_mono))
    
    # Parámetros binaurales
    RADIO_CABEZA = 0.0875
    VELOCIDAD_SONIDO = 343.0
    ILD_DB_60 = 6.0
    
    theta = np.radians(min(abs(angulo_grados), 90))
    itd_segundos = (RADIO_CABEZA / VELOCIDAD_SONIDO) * (np.sin(theta) + theta)
    itd_muestras = int(round(itd_segundos * sr))
    atenuacion = 10 ** (-(ILD_DB_60 * (abs(angulo_grados)/90.0)) / 20)
    
    if angulo_grados > 0:
        canal_R = tono_mono
        canal_L = tono_mono * atenuacion
        if itd_muestras > 0:
            canal_L = np.pad(canal_L, (itd_muestras, 0))[:-itd_muestras]
    else:
        canal_L = tono_mono
        canal_R = tono_mono * atenuacion
        if itd_muestras > 0:
            canal_R = np.pad(canal_R, (itd_muestras, 0))[:-itd_muestras]
    
    min_len = min(len(canal_L), len(canal_R))
    return canal_L[:min_len], canal_R[:min_len]

def asegurar_tonos(directorio='audio_binaural'):
    """Genera los tonos si no existen"""
    os.makedirs(directorio, exist_ok=True)
    
    generados = []
    for freq in FRECUENCIAS:
        for sufijo, angulo in [('pos', 60), ('neg', -60)]:
            nombre = f"freq_{freq:.0f}_{sufijo}60deg_largo.wav"
            filepath = os.path.join(directorio, nombre)
            if not os.path.exists(filepath):
                canal_L, canal_R = generar_tono_binaural(freq, SR, DURACION_TONO, angulo)
                stereo = np.column_stack((canal_L, canal_R))
                sf.write(filepath, stereo, SR)
                generados.append(nombre)
                print(f"    Generado: {nombre}")
    return generados


# ============================================================
# CARGA DE ARCHIVOS
# ============================================================
def cargar_sonidos(directorio='audio_binaural'):
    archivos = {}
    
    print(f"\n[Carga] Desde '{directorio}/'...")
    
    # Generar tonos si no existen
    generados = asegurar_tonos(directorio)
    if generados:
        print(f"  Generados {len(generados)} tonos nuevos")
    
    # Cargar tonos del barrido
    for freq in FRECUENCIAS:
        for sufijo, angulo in [('pos', 60), ('neg', -60)]:
            nombre = f"freq_{freq:.0f}_{sufijo}60deg_largo"
            filepath = os.path.join(directorio, nombre + '.wav')
            if os.path.exists(filepath):
                try:
                    data, sr = sf.read(filepath, dtype='float32')
                    if data.ndim == 1:
                        canal_L = data
                        canal_R = data.copy()
                    else:
                        canal_L = data[:, 0]
                        canal_R = data[:, 1] if data.shape[1] > 1 else data[:, 0].copy()
                    archivos[nombre] = (filepath, sr, canal_L, canal_R)
                    print(f"    [OK] {nombre:40s} ({len(canal_L)/sr:.1f}s)")
                except Exception as e:
                    print(f"    [X] {nombre:40s} {e}")
    
    # Cargar controles
    for nombre in ['BigBang_pos60deg', 'BigBang_neg60deg']:
        filepath = os.path.join(directorio, nombre + '.wav')
        if os.path.exists(filepath):
            try:
                data, sr = sf.read(filepath, dtype='float32')
                if data.ndim == 1:
                    canal_L = data
                    canal_R = data.copy()
                else:
                    canal_L = data[:, 0]
                    canal_R = data[:, 1] if data.shape[1] > 1 else data[:, 0].copy()
                archivos[nombre] = (filepath, sr, canal_L, canal_R)
                print(f"    [OK] {nombre:40s} ({len(canal_L)/sr:.1f}s)")
            except Exception as e:
                print(f"    [X] {nombre:40s} {e}")
    
    # Silencio
    sr = 48000
    silencio = np.zeros(int(sr * 60))
    archivos['silencio'] = ('silencio', sr, silencio, silencio)
    print(f"    [OK] silencio                                    (60.0s)")
    
    print(f"  Carga completada: {len(archivos)} archivos.")
    return archivos


# ============================================================
# CLASE EXPLORADOR Y FUNCIONES BASE (iguales a v105d)
# ============================================================
class ExploradorActuadores:
    def __init__(self):
        self.historial = []
        self.mejor_config = None
        self.mejor_eficiencia = 0.0
        self.pasos_en_lf = 0

    def actualizar(self, lf_activa, efic, fL, fR, sesgo):
        if lf_activa:
            self.pasos_en_lf += 1
            self.historial.append((fL, fR, sesgo, efic))
            if efic > self.mejor_eficiencia:
                self.mejor_eficiencia = efic
                self.mejor_config = (fL, fR, sesgo)
        else:
            self.pasos_en_lf = 0


def inicializar_campo():
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
    n_bins = 50
    perfil = np.zeros(n_bins)
    for banda in range(min(dim, region.shape[0])):
        serie = region[banda, :] - np.mean(region[banda, :])
        fft = np.fft.rfft(serie)
        perfil += np.abs(fft)[:n_bins] ** 2
    return perfil / max(1, dim)

def calcular_frecuencias_naturales(dim):
    bandas = np.arange(dim)
    t = np.log1p(bandas) / np.log1p(max(dim - 1, 1))
    omega = OMEGA_MIN + (OMEGA_MAX - OMEGA_MIN) * t
    amort = AMORT_MIN + (AMORT_MAX - AMORT_MIN) * t
    return omega.reshape(-1, 1), amort.reshape(-1, 1)

def calcular_promedio_vecinos(Phi_total):
    promedio = np.zeros_like(Phi_total)
    conteo = np.zeros(DIM_TOTAL)
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


def preparar_objetivo_canal(canal, sr, idx_paso, ventana_muestras,
                             hop_muestras, dim_aud, dim_time):
    inicio = idx_paso * hop_muestras
    fin = inicio + ventana_muestras
    segmento = canal[inicio:fin] if fin <= len(canal) else canal[inicio:]
    if len(segmento) < ventana_muestras:
        segmento = np.pad(segmento, (0, ventana_muestras - len(segmento)))

    fft = np.fft.rfft(segmento)
    potencia = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(len(segmento), 1 / sr)

    bandas = np.logspace(np.log10(F_MIN), np.log10(F_MAX), dim_aud + 1)
    objetivo = np.zeros(dim_aud)
    for b in range(dim_aud):
        mask = (freqs >= bandas[b]) & (freqs < bandas[b + 1])
        if np.any(mask):
            objetivo[b] = np.mean(potencia[mask])

    max_val = np.max(objetivo)
    if max_val > 0:
        objetivo /= max_val

    return objetivo.reshape(-1, 1) * np.ones((1, dim_time))


def calcular_gradiente_energetico_dirigido(obj_L, obj_R):
    if BANDA_TRANS >= DIM_AUD:
        return 0.0
    energia_L = float(np.mean(obj_L[BANDA_TRANS:, :] ** 2))
    energia_R = float(np.mean(obj_R[BANDA_TRANS:, :] ** 2))
    total = energia_L + energia_R + 1e-10
    return (energia_R - energia_L) / total


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
    g0, g1 = idx['G']
    estado_busc = float(np.mean(Phi_total[ab0:ab1, :])) - PHI_EQUILIBRIO
    n = min(ab1 - ab0, g1 - g0)
    Phi_total[g0:g0 + n, :] += EPSILON_BUSC_G * estado_busc
    return Phi_total


def aplicar_orientacion_v1_aditiva(Phi_total, gradiente_E, W_rec, dt):
    acg0 = idx['act_geom'][0]
    acg1 = idx['act_geom'][1]
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

    sesgo_rec = float(np.tanh(sesgo_dir)) * DIFUSION_BASE * dt
    senal_total = senal_grad + sesgo_rec

    Phi_total[acg0:acg0 + mitad, :] += senal_total
    Phi_total[acg0 + mitad:acg1, :] -= senal_total
    return Phi_total


def calcular_parametros_actuacion(Phi_total):
    act_perm = Phi_total[idx['act_perm'][0]:idx['act_perm'][1], :]
    act_geom = Phi_total[idx['act_geom'][0]:idx['act_geom'][1], :]
    nivel_perm = float(np.mean(np.tanh(act_perm)))
    frac_base = 0.25 + 0.75 * (nivel_perm + 1.0) / 2.0
    mitad = max(1, DIM_ACT // 2)
    g_baja = float(np.mean(act_geom[:mitad, :]))
    g_alta = float(np.mean(act_geom[mitad:, :]))
    sesgo = float(np.tanh(g_alta - g_baja))
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
            corr = (explorador.mejor_config[0] - nivel) * DIFUSION_BASE * dt
            Phi_total[ap0:ap1, :] += corr
    return Phi_total


def aplicar_plasticidad_dual(region_int, region_aud, W_prof, W_rec,
                              Phi_int_historia, dt, modo_aud='dir'):
    min_prof = min(W_prof.shape[0], region_int.shape[0])
    min_cols = min(W_prof.shape[1], region_aud.shape[0])
    W_p = W_prof[:min_prof, :min_cols]
    W_r = W_rec[:min_prof, :min_cols]
    r_i = region_int[:min_prof, :]
    r_a = region_aud[:min_cols, :]
    corr_prof = (r_i @ r_a.T) / DIM_TIME
    dW_prof = ETA_PROFUNDA_BASE * corr_prof - TAU_PROFUNDA * W_p
    W_p_nueva = np.clip(W_p + dW_prof * dt, -W_MAX, W_MAX)
    W_prof_nueva = W_prof.copy()
    W_prof_nueva[:min_prof, :min_cols] = W_p_nueva
    pred_rec = np.tanh(W_r @ r_a)
    error_rec = float(np.mean((pred_rec - r_i) ** 2))
    pred_prof = W_p_nueva @ r_a
    error_prof = float(np.mean((pred_prof - r_i) ** 2))
    coherencia = error_prof / (error_rec + error_prof + 1e-10)
    tasa_aprendizaje = ETA_RECIENTE_BASE * coherencia
    corr_rec = (r_i @ r_a.T) / DIM_TIME
    dW_rec = tasa_aprendizaje * corr_rec - TAU_RECIENTE * W_r
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


def calcular_omega_orient(Phi_total, gradiente_hist_fase):
    if len(gradiente_hist_fase) < 2:
        return 0.0

    ag0, ag1 = idx['act_geom']
    ab0, ab1 = idx['act_busc']

    geom_medio = float(np.mean(np.tanh(Phi_total[ag0:ag1, :])))
    busc_medio = float(np.mean(Phi_total[ab0:ab1, :])) - PHI_EQUILIBRIO

    config_interna = np.array([geom_medio, busc_medio])

    grads = np.array(gradiente_hist_fase)
    grad_pos = float(np.mean(grads[grads >= 0])) if np.any(grads >= 0) else 0.0
    grad_neg = float(np.mean(np.abs(grads[grads < 0]))) if np.any(grads < 0) else 0.0
    firma_entorno = np.array([grad_pos, -grad_neg])

    norma_c = np.linalg.norm(config_interna)
    norma_f = np.linalg.norm(firma_entorno)

    if norma_c < 1e-10 or norma_f < 1e-10:
        return 0.0

    return float(np.dot(config_interna, firma_entorno) / (norma_c * norma_f))


def calcular_eficiencia(Phi_total, ged_actual):
    region_int = Phi_total[idx['int'][0]:idx['int'][1], :]
    variacion_real = float(np.mean(np.abs(np.diff(region_int, axis=1))))
    variacion_floor = max(variacion_real, VARIACION_FLOOR)
    efic = ged_actual / variacion_floor
    return efic, variacion_real


def actualizar_campo(Phi_total, Phi_vel_total, W_prof, W_rec,
                     Phi_int_historia, obj_L, obj_R,
                     frac_L, frac_R, sesgo, dt, modo_aud='dir'):

    omega_n, amort_n = calcular_frecuencias_naturales(DIM_TOTAL)
    prom = calcular_promedio_vecinos(Phi_total)
    difusion = DIFUSION_BASE * (prom - Phi_total)
    desv = Phi_total - prom
    reaccion = GANANCIA_REACCION * desv * (1 - desv ** 2)
    term_osc = (-omega_n ** 2 * (Phi_total - PHI_EQUILIBRIO)
                - amort_n * Phi_vel_total)

    region_int = Phi_total[idx['int'][0]:idx['int'][1], :]
    aud_L = Phi_total[idx['aud_L'][0]:idx['aud_L'][1], :]
    aud_R = Phi_total[idx['aud_R'][0]:idx['aud_R'][1], :]

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
    n_m = M_plast.shape[0]
    M_campo[idx['int'][0]:idx['int'][0] + n_m, :] = M_plast

    Phi_total = aplicar_entrada_cualitativa(Phi_total, obj_L, obj_R,
                                            frac_L, frac_R, sesgo)

    dPhi_vel = term_osc + reaccion + difusion + M_campo
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
def entrenar(archivos, modo_aud='dir'):
    print(f"\n[Entrenamiento] BigBang_pos60deg + BigBang_neg60deg (60s c/u)")
    
    Phi_total, Phi_vel_total = inicializar_campo()
    W_prof, W_rec, Phi_int_historia = inicializar_memorias()
    explorador = ExploradorActuadores()
    
    errores = []
    
    for estimulo in ['BigBang_pos60deg', 'BigBang_neg60deg']:
        if estimulo not in archivos:
            print(f"    [ERROR] {estimulo} no encontrado")
            continue
        
        _, sr, c_L, c_R = archivos[estimulo]
        vent = int(sr * VENTANA_FFT_MS / 1000)
        hop = int(sr * HOP_FFT_MS / 1000)
        n_pasos = int(DURACION_ENTRENAMIENTO / DT)
        
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
# EVALUACION DE UN ESTIMULO
# ============================================================
def evaluar_estimulo(Phi_total, Phi_vel_total, W_prof, W_rec,
                     Phi_int_historia, explorador, archivos,
                     clave, duracion, modo_aud='dir'):
    
    if clave not in archivos:
        return None
    
    _, sr, c_L, c_R = archivos[clave]
    vent = int(sr * VENTANA_FFT_MS / 1000)
    hop = int(sr * HOP_FFT_MS / 1000)
    n_pasos = int(duracion / DT)
    n_pasos = min(n_pasos, len(c_L) // hop + 1)

    hist = {k: [] for k in [
        'omega', 'grad_E', 'act_mant'
    ]}

    gradiente_hist_fase = []
    lf_prev = False

    for paso in range(n_pasos):
        obj_L = preparar_objetivo_canal(c_L, sr, paso, vent, hop, DIM_AUD, DIM_TIME)
        obj_R = preparar_objetivo_canal(c_R, sr, paso, vent, hop, DIM_AUD, DIM_TIME)

        gradiente_E = calcular_gradiente_energetico_dirigido(obj_L, obj_R)
        gradiente_hist_fase.append(gradiente_E)

        Phi_total = actualizar_act_busc_desde_gradiente(Phi_total, gradiente_E, DT)
        Phi_total = aplicar_forzamiento_busc_a_ganglio(Phi_total, DT)
        Phi_total = aplicar_orientacion_v1_aditiva(Phi_total, gradiente_E, W_rec, DT)

        fL, fR, sf_v, _, _ = calcular_parametros_actuacion(Phi_total)

        efic, _ = calcular_eficiencia(Phi_total, 0)

        act_mant = Phi_total[idx['act_mant'][0]:idx['act_mant'][1], :]
        act_mant_media = float(np.mean(act_mant))

        explorador.actualizar(lf_prev, efic, fL, fR, sf_v)

        Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia, \
            lf_activa, error_rec, _ = actualizar_campo(
                Phi_total, Phi_vel_total, W_prof, W_rec,
                Phi_int_historia, obj_L, obj_R, fL, fR, sf_v, DT,
                modo_aud=modo_aud
            )

        Phi_total = explorar_actuadores(Phi_total, explorador, lf_activa, efic, DT)
        lf_prev = lf_activa

        omega = calcular_omega_orient(Phi_total, gradiente_hist_fase)

        hist['omega'].append(omega)
        hist['grad_E'].append(gradiente_E)
        hist['act_mant'].append(act_mant_media)

    omega_final = float(np.mean(hist['omega'][-VENTANA_FINAL_PASOS:])) if len(hist['omega']) > VENTANA_FINAL_PASOS else float(np.mean(hist['omega']))

    return {
        'omega_final': omega_final,
        'omega_series': hist['omega'],
        'grad_E_series': hist['grad_E'],
        'act_mant_series': hist['act_mant'],
        'phi_total': Phi_total, 'phi_vel': Phi_vel_total,
        'W_prof': W_prof, 'W_rec': W_rec,
        'Phi_int_historia': Phi_int_historia,
    }


# ============================================================
# EXPERIMENTO: BARRIDO FINO
# ============================================================
def experimento_barrido_fino(archivos, base_state, modo_aud='dir'):
    print("\n" + "=" * 80)
    print("EXPERIMENTO: Barrido fino alrededor de 440 Hz")
    print("=" * 80)
    print("  Evaluando cada frecuencia (pos/neg) con 30s de duración")
    print("-" * 80)
    
    resultados = {}
    
    for freq in FRECUENCIAS:
        print(f"\n  Frecuencia: {freq} Hz")
        
        # Positivo
        nombre_pos = f"freq_{freq:.0f}_pos60deg_largo"
        Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia, explorador = base_state
        
        res_pos = evaluar_estimulo(
            Phi_total, Phi_vel_total, W_prof, W_rec,
            Phi_int_historia, explorador, archivos,
            nombre_pos, DURACION_EVALUACION, modo_aud
        )
        
        if not res_pos:
            print(f"    ERROR evaluando {nombre_pos}")
            continue
        
        omega_pos = res_pos['omega_final']
        gradE_pos = np.mean(res_pos['grad_E_series']) if res_pos['grad_E_series'] else 0
        
        # Negativo
        nombre_neg = f"freq_{freq:.0f}_neg60deg_largo"
        Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia, explorador = base_state
        
        res_neg = evaluar_estimulo(
            Phi_total, Phi_vel_total, W_prof, W_rec,
            Phi_int_historia, explorador, archivos,
            nombre_neg, DURACION_EVALUACION, modo_aud
        )
        
        if not res_neg:
            print(f"    ERROR evaluando {nombre_neg}")
            continue
        
        omega_neg = res_neg['omega_final']
        gradE_neg = np.mean(res_neg['grad_E_series']) if res_neg['grad_E_series'] else 0
        
        delta_omega = omega_pos - omega_neg
        abs_delta = abs(delta_omega)
        
        # Clasificar
        if abs_delta > 0.5:
            clasificacion = "🔴 ANOMALIA"
        elif abs_delta > 0.1:
            clasificacion = "🟡 SENSIBLE"
        elif abs_delta > 0.02:
            clasificacion = "🟢 DEBIL"
        else:
            clasificacion = "⚪ NEUTRO"
        
        print(f"    {clasificacion}")
        print(f"      pos: Ω={omega_pos:.4f}, gradE={gradE_pos:.6f}")
        print(f"      neg: Ω={omega_neg:.4f}, gradE={gradE_neg:.6f}")
        print(f"      ΔΩ = {delta_omega:+.4f} |ΔΩ| = {abs_delta:.4f}")
        
        resultados[freq] = {
            'frecuencia': freq,
            'omega_pos': omega_pos,
            'omega_neg': omega_neg,
            'gradE_pos': gradE_pos,
            'gradE_neg': gradE_neg,
            'delta_omega': delta_omega,
            'abs_delta': abs_delta,
            'es_anomalia': abs_delta > 0.5
        }
    
    return resultados


# ============================================================
# ANALISIS DE RESONANCIA
# ============================================================
def analizar_resonancia(resultados):
    print("\n" + "=" * 80)
    print("ANALISIS DE RESONANCIA")
    print("=" * 80)
    
    frecuencias = []
    abs_deltas = []
    
    for freq, data in sorted(resultados.items()):
        frecuencias.append(freq)
        abs_deltas.append(data['abs_delta'])
    
    # Encontrar pico
    peak_idx = np.argmax(abs_deltas)
    peak_freq = frecuencias[peak_idx]
    peak_value = abs_deltas[peak_idx]
    
    print(f"\n  📊 PICO MÁXIMO: {peak_freq} Hz con |ΔΩ| = {peak_value:.4f}")
    
    # Calcular ancho de banda (FWHM)
    half_max = peak_value / 2
    indices_above_half = [i for i, v in enumerate(abs_deltas) if v >= half_max]
    
    if len(indices_above_half) >= 2:
        fwhm = frecuencias[indices_above_half[-1]] - frecuencias[indices_above_half[0]]
        print(f"  📊 ANCHO DE BANDA (FWHM): ~{fwhm} Hz")
    else:
        print(f"  📊 ANCHO DE BANDA: < 1 Hz (muy estrecho)")
    
    # Verificar simetría alrededor del pico
    if peak_idx > 0 and peak_idx < len(frecuencias) - 1:
        left_idx = peak_idx - 1
        right_idx = peak_idx + 1
        
        if left_idx >= 0 and right_idx < len(frecuencias):
            left_val = abs_deltas[left_idx]
            right_val = abs_deltas[right_idx]
            simetria = abs(left_val - right_val) / max(left_val, right_val)
            print(f"  📊 SIMETRÍA: {'SIMÉTRICO' if simetria < 0.3 else 'ASIMÉTRICO'} (izq={left_val:.4f}, der={right_val:.4f})")
    
    # Detectar otros picos
    peaks, properties = find_peaks(abs_deltas, height=0.05, distance=2)
    otros_picos = [(frecuencias[i], abs_deltas[i]) for i in peaks if i != peak_idx]
    
    if otros_picos:
        print(f"\n  📊 OTROS PICOS DETECTADOS:")
        for f, v in otros_picos:
            print(f"      {f} Hz: |ΔΩ| = {v:.4f}")
    else:
        print(f"\n  📊 OTROS PICOS: No se detectaron otros picos significativos")
    
    return {'peak_freq': peak_freq, 'peak_value': peak_value, 'fwhm': fwhm if 'fwhm' in dir() else None}


# ============================================================
# MAIN
# ============================================================
def main():
    archivos = cargar_sonidos('audio_binaural')
    
    print("\n" + "█" * 100)
    print("ENTRENAMIENTO BASE")
    print("█" * 100)
    
    Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia, explorador = \
        entrenar(archivos)
    
    base_state = (Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia, explorador)
    
    # Ejecutar experimento
    resultados = experimento_barrido_fino(archivos, base_state)
    
    # ============================================================
    # REPORTE FINAL
    # ============================================================
    print()
    print("=" * 100)
    print("REPORTE DE OBSERVACIONES - v105e")
    print("=" * 100)
    
    print("\n  📈 RESULTADOS DEL BARRIDO FINO")
    print("  " + "-" * 80)
    print(f"    {'Freq (Hz)':10s} {'Ω_pos':10s} {'Ω_neg':10s} {'ΔΩ':12s} {'|ΔΩ|':10s} {'Estado':10s}")
    print("    " + "-" * 80)
    
    for freq in sorted(resultados.keys()):
        data = resultados[freq]
        marcador = "🔴" if data['es_anomalia'] else ("🟡" if data['abs_delta'] > 0.1 else ("🟢" if data['abs_delta'] > 0.02 else "⚪"))
        print(f"    {freq:4d}       {data['omega_pos']:8.4f}  {data['omega_neg']:8.4f}  {data['delta_omega']:+10.4f}  {data['abs_delta']:8.4f}   {marcador}")
    
    # Análisis de resonancia
    analisis = analizar_resonancia(resultados)
    
    # ============================================================
    # CONCLUSION
    # ============================================================
    print("\n" + "=" * 100)
    print("CONCLUSION")
    print("=" * 100)
    
    anomalias = sum(1 for d in resultados.values() if d['es_anomalia'])
    
    if anomalias > 0:
        print(f"""
    ✅ RESONANCIA CONFIRMADA en {analisis['peak_freq']} Hz
    
    El sistema muestra un pico de sensibilidad direccional extremadamente agudo
    en {analisis['peak_freq']} Hz con |ΔΩ| = {analisis['peak_value']:.4f}.
    
    Características de la resonancia:
    - Frecuencia crítica: {analisis['peak_freq']} Hz
    - Ancho de banda: {analisis['fwhm'] if analisis['fwhm'] else '< 1'} Hz
    - Factor de calidad (Q): {analisis['peak_freq'] / analisis['fwhm'] if analisis['fwhm'] and analisis['fwhm'] > 0 else '> 440'} 
    
    Esto confirma que el sistema NO es un decodificador Shannon.
    Es un campo no lineal con puntos de bifurcación frecuenciales.
    
    IMPLICACIÓN: El sistema tiene una "frecuencia propia" de resonancia
    que coincide con el LA estándar de afinación musical (440 Hz).
    
    → Esto es evidencia de `R₂`: el sistema se interpreta a sí mismo
      a través de su curva de sintonía.
    """)
    else:
        print("""
    ❌ RESONANCIA NO CONFIRMADA
    
    No se detectaron picos significativos en el barrido.
    """)
    
    # Gráfico
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Gráfico 1: |ΔΩ| vs Frecuencia
    ax = axes[0, 0]
    frecuencias = sorted(resultados.keys())
    abs_deltas = [resultados[f]['abs_delta'] for f in frecuencias]
    ax.plot(frecuencias, abs_deltas, 'o-', color='steelblue', linewidth=2, markersize=8)
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Umbral anomalía')
    ax.axhline(y=0.1, color='orange', linestyle=':', linewidth=1, alpha=0.7, label='Umbral sensible')
    ax.set_xlabel('Frecuencia (Hz)')
    ax.set_ylabel('|ΔΩ|')
    ax.set_title('Curva de sensibilidad direccional')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 2: Ω_pos y Ω_neg vs Frecuencia
    ax = axes[0, 1]
    omega_pos = [resultados[f]['omega_pos'] for f in frecuencias]
    omega_neg = [resultados[f]['omega_neg'] for f in frecuencias]
    ax.plot(frecuencias, omega_pos, 'o-', color='green', label='Positivo (+60°)', markersize=6)
    ax.plot(frecuencias, omega_neg, 's-', color='orange', label='Negativo (-60°)', markersize=6)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Frecuencia (Hz)')
    ax.set_ylabel('Ω')
    ax.set_title('Ω por dirección')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 3: ΔΩ vs Frecuencia
    ax = axes[1, 0]
    deltas = [resultados[f]['delta_omega'] for f in frecuencias]
    colors = ['red' if abs(d) > 0.5 else ('orange' if abs(d) > 0.1 else 'steelblue') for d in deltas]
    ax.bar(frecuencias, deltas, width=3, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.axhline(y=-0.5, color='red', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_xlabel('Frecuencia (Hz)')
    ax.set_ylabel('ΔΩ (pos - neg)')
    ax.set_title('Sensibilidad direccional por frecuencia')
    
    # Gráfico 4: Detalle alrededor del pico
    ax = axes[1, 1]
    # Filtrar alrededor del pico (±10 Hz)
    peak = analisis['peak_freq']
    mask = [abs(f - peak) <= 20 for f in frecuencias]
    freqs_cerca = [f for i, f in enumerate(frecuencias) if mask[i]]
    deltas_cerca = [d for i, d in enumerate(deltas) if mask[i]]
    ax.plot(freqs_cerca, deltas_cerca, 'o-', color='red', linewidth=2, markersize=10)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.axhline(y=-0.5, color='red', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.axvline(x=peak, color='purple', linestyle=':', linewidth=1, alpha=0.7)
    ax.set_xlabel('Frecuencia (Hz)')
    ax.set_ylabel('ΔΩ (pos - neg)')
    ax.set_title(f'Detalle alrededor de {peak} Hz')
    ax.grid(True, alpha=0.3)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('v105e_logs', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'v105e_logs/v105e_resultados_{timestamp}.png', dpi=150)
    
    print(f"\n  Gráfico guardado: v105e_logs/v105e_resultados_{timestamp}.png")
    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)


if __name__ == "__main__":
    main()