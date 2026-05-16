#!/usr/bin/env python3
"""
VSTCosmos v82 — Acoplamiento activo: entrada estéreo real, actuación cualitativa y exploración deliberada

Cambios respecto a v81:
1. Entrada estéreo real (archivos con dos canales, o generación de desfase)
2. Actuación cualitativa: fracción de ventana y sesgo frecuencial
3. Exploración activa bajo LF con memoria de eficiencia
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import csv
import warnings
from collections import deque
import soundfile as sf
warnings.filterwarnings('ignore')

# ============================================================
# PARÁMETROS DE LA FÍSICA DEL CAMPO (heredados de v80h)
# ============================================================
DIM_INTERNA = 32

# Parámetros de dinámica
DIFUSION_BASE = 0.15
GANANCIA_REACCION = 0.05

OMEGA_MIN = 0.05
OMEGA_MAX = 0.50
AMORT_MIN = 0.01
AMORT_MAX = 0.08
PHI_EQUILIBRIO = 0.5

# Parámetros de FFT
VENTANA_FFT_MS = 25
HOP_FFT_MS = 10
F_MIN = 80
F_MAX = 8000

# Parámetros de plasticidad (derivados estructuralmente)
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
TAU_EXPLORACION = int(T_RECIENTE_SEG / 0.01)  # 200 pasos = 2s

# Limites
LIMITE_MIN = 0.0
LIMITE_MAX = 1.0
W_MAX = 1.0
ALPHA_FIJO = 0.05
DT = 0.01
DIM_TIME = 100

# ============================================================
# ARQUITECTURA DEL CAMPO EXPANDIDO
# ============================================================
DIM_GANGLIO = DIM_INTERNA // 2      # = 16
DIM_AUD = DIM_GANGLIO               # = 16 (cada canal)
DIM_ACT = DIM_GANGLIO // 2          # = 8

DIM_AUD_L = DIM_AUD
DIM_AUD_R = DIM_AUD
DIM_ACT_PERM = DIM_ACT
DIM_ACT_GEOM = DIM_ACT
DIM_ACT_BUSC = DIM_ACT
DIM_ACT_MANT = DIM_ACT

# Cálculo de índices de regiones
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

# Topología de vecindades
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
print("VSTCosmos v82 — Acoplamiento activo: entrada estéreo real")
print("")
print("  Cambios respecto a v81:")
print("  1. Entrada estéreo real (archivos con dos canales)")
print("  2. Actuación cualitativa: fracción de ventana y sesgo frecuencial")
print("  3. Exploración activa bajo LF con memoria de eficiencia")
print("")
print("  Arquitectura del campo expandido:")
print(f"    DIM_INTERNA = {DIM_INTERNA} (identidad)")
print(f"    DIM_GANGLIO = {DIM_GANGLIO}")
print(f"    DIM_AUD_L/R = {DIM_AUD_L}")
print(f"    DIM_ACT = {DIM_ACT} × 5")
print(f"    DIM_TOTAL = {DIM_TOTAL}")
print("")
print("  Vecindades topológicas:")
for v in VECINDADES:
    print(f"    {v[0]} ↔ {v[1]}")
print("=" * 100)

# ============================================================
# CLASE EXPLORADOR DE ACTUADORES
# ============================================================
class ExploradorActuadores:
    def __init__(self):
        self.historial = []
        self.mejor_config = None
        self.mejor_eficiencia = 0.0
        self.pasos_en_lf = 0
    
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
def cargar_audio_estereo(filepath, duracion):
    """Carga archivo estéreo o genera estímulo sintético"""
    
    # Detectar estímulos sintéticos
    if "Tono puro" in filepath:
        sr = 48000
        t = np.arange(int(sr * duracion)) / sr
        senal = 0.5 * np.sin(2 * np.pi * 440 * t)
        return sr, senal, senal
    
    if "Ruido blanco" in filepath:
        sr = 48000
        ruido = np.random.normal(0, 0.5, int(sr * duracion))
        return sr, ruido, ruido
    
    try:
        data, sr = sf.read(filepath, dtype='float32')
        n_muestras = int(sr * duracion)
        
        if data.ndim == 1:
            # Mono: generar canal R con desfase de un paso
            canal_L = data[:n_muestras]
            hop = int(sr * HOP_FFT_MS / 1000)
            canal_R = np.concatenate([np.zeros(hop), data])[:n_muestras]
            if len(canal_R) < n_muestras:
                canal_R = np.pad(canal_R, (0, n_muestras - len(canal_R)))
        else:
            # Estéreo: usar canales directamente
            canal_L = data[:n_muestras, 0]
            canal_R = data[:n_muestras, 1] if data.shape[1] > 1 else data[:n_muestras, 0]
        
        return sr, canal_L, canal_R
    except Exception as e:
        # Fallback a scipy
        try:
            sr, data = wav.read(filepath)
            n_muestras = int(sr * duracion)
            
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            if data.dtype == np.uint8:
                data = (data - 128).astype(np.float32) / 128.0
            
            if data.ndim == 2:
                canal_L = data[:n_muestras, 0]
                canal_R = data[:n_muestras, 1] if data.shape[1] > 1 else data[:n_muestras, 0]
            else:
                canal_L = data[:n_muestras]
                hop = int(sr * HOP_FFT_MS / 1000)
                canal_R = np.concatenate([np.zeros(hop), canal_L])[:n_muestras]
                if len(canal_R) < n_muestras:
                    canal_R = np.pad(canal_R, (0, n_muestras - len(canal_R)))
            
            return sr, canal_L, canal_R
        except Exception as e2:
            print(f"  [ERROR] No se pudo cargar {filepath}: {e2}")
            sr = 48000
            t = np.arange(int(sr * duracion)) / sr
            senal = 0.5 * np.sin(2 * np.pi * 440 * t)
            return sr, senal, senal

def preparar_objetivo_canal(canal, sr, idx_paso, ventana_muestras, hop_muestras,
                             dim_aud, dim_time):
    """Prepara objetivo para un solo canal"""
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

def inicializar_campo_v82():
    """Todas las regiones se inicializan con ruido pequeño alrededor de PHI_EQUILIBRIO"""
    np.random.seed(None)
    Phi_total = np.random.normal(PHI_EQUILIBRIO, 0.01, (DIM_TOTAL, DIM_TIME))
    Phi_vel_total = np.zeros((DIM_TOTAL, DIM_TIME))
    return Phi_total, Phi_vel_total

def inicializar_memorias():
    """Memorias con dimensiones compatibles con región_int y región_aud"""
    W_prof = np.zeros((DIM_INTERNA, DIM_AUD))
    W_rec = np.zeros((DIM_INTERNA, DIM_AUD))
    Phi_int_historia = np.zeros((DIM_INTERNA, DIM_TIME))
    return W_prof, W_rec, Phi_int_historia

def _perfil_espectral_region(region, dim):
    perfil = np.zeros(50)
    for banda in range(min(dim, len(region))):
        serie = region[banda, :]
        serie = serie - np.mean(serie)
        fft = np.fft.rfft(serie)
        potencia = np.abs(fft) ** 2
        perfil += potencia[:50]
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

# ============================================================
# DIFUSIÓN ENTRE VECINOS (TOPOLÓGICA)
# ============================================================
def calcular_promedio_vecinos_v82(Phi_total, idx):
    """Promedio de vecinos según topología definida"""
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
# ACTUACIÓN CUALITATIVA
# ============================================================
def calcular_parametros_actuacion(Phi_total, idx):
    """Parámetros de actuación derivados del estado del campo"""
    act_perm = Phi_total[idx['act_perm'][0]:idx['act_perm'][1], :]
    act_geom = Phi_total[idx['act_geom'][0]:idx['act_geom'][1], :]
    
    # Permeabilidad → fracción de ventana usada
    nivel_perm = float(np.mean(np.tanh(act_perm)))
    fraccion_base = 0.25 + 0.75 * (nivel_perm + 1.0) / 2.0
    
    # Geometría → sesgo frecuencial
    mitad = max(1, DIM_ACT // 2)
    banda_baja = float(np.mean(act_geom[:mitad, :]))
    banda_alta = float(np.mean(act_geom[mitad:, :])) if mitad < DIM_ACT else banda_baja
    sesgo_freq = float(np.tanh(banda_alta - banda_baja))
    
    # Asimetría L/R
    mitad_geom = max(1, DIM_ACT // 2)
    geom_L_val = float(np.mean(act_geom[:mitad_geom, :]))
    geom_R_val = float(np.mean(act_geom[mitad_geom:, :])) if mitad_geom < DIM_ACT else geom_L_val
    asimetria = float(np.tanh(geom_L_val - geom_R_val))
    
    # Aplicar asimetría a las fracciones
    fraccion_L = np.clip(fraccion_base * (1.0 + asimetria * 0.5), 0.1, 1.0)
    fraccion_R = np.clip(fraccion_base * (1.0 - asimetria * 0.5), 0.1, 1.0)
    
    return fraccion_L, fraccion_R, sesgo_freq, asimetria, nivel_perm

def aplicar_entrada_cualitativa(objetivo_L_full, objetivo_R_full,
                                  Phi_total, idx,
                                  fraccion_L, fraccion_R, sesgo_freq):
    """Aplica entrada al campo con resolución y sesgo frecuencial"""
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
        objetivo_mod[inicio_banda:fin_banda, :] = \
            objetivo_full[inicio_banda:fin_banda, :]
        
        Phi_total[region_slice, :] = (
            (1 - ALPHA_FIJO) * Phi_total[region_slice, :] + ALPHA_FIJO * objetivo_mod
        )
    
    aplicar_canal(objetivo_L_full, fraccion_L, slice(idx['aud_L'][0], idx['aud_L'][1]))
    aplicar_canal(objetivo_R_full, fraccion_R, slice(idx['aud_R'][0], idx['aud_R'][1]))
    
    return Phi_total

# ============================================================
# EXPLORACIÓN ACTIVA
# ============================================================
def explorar_actuadores(Phi_total, idx, explorador, lf_activa,
                         eficiencia_actual, dt):
    """Genera variación deliberada en actuadores bajo LF"""
    act_perm = Phi_total[idx['act_perm'][0]:idx['act_perm'][1], :]
    act_geom = Phi_total[idx['act_geom'][0]:idx['act_geom'][1], :]
    
    AMPLITUD_MAX = DIFUSION_BASE
    
    if lf_activa:
        amplitud = AMPLITUD_MAX * min(1.0, explorador.pasos_en_lf / TAU_EXPLORACION)
        
        if explorador.mejor_config is not None:
            mejor_perm = explorador.mejor_config[0]
            mejor_frac_R = explorador.mejor_config[1]
            nivel_actual = float(np.mean(np.tanh(act_perm)))
            sesgo_perm = (mejor_perm + mejor_frac_R) / 2.0 - nivel_actual
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
# EFICIENCIA DE RÉGIMEN Y PLASTICIDAD (heredados)
# ============================================================
def calcular_eficiencia_regimen(Phi_total, dim_interna, ged_actual):
    region_int = Phi_total[:dim_interna, :]
    variacion = float(np.mean(np.abs(np.diff(region_int, axis=1))))
    eficiencia = ged_actual / (variacion + 1e-10)
    return eficiencia, variacion

def calcular_tasa_olvido_por_eficiencia(eficiencia_actual, historial_eficiencia, k=10):
    if len(historial_eficiencia) < TAU_EFICIENCIA:
        return TAU_RECIENTE, 1.0, 1.0
    
    eficiencia_media = float(np.mean(historial_eficiencia[-TAU_EFICIENCIA:]))
    ventaja = eficiencia_actual / (eficiencia_media + 1e-10)
    ventaja_amplificada = ventaja ** k
    tasa_olvido = TAU_RECIENTE / ventaja_amplificada
    tasa_max = ETA_RECIENTE_BASE * 10.0
    
    return min(tasa_olvido, tasa_max), ventaja, ventaja_amplificada

def aplicar_plasticidad_dual_v82(region_int, region_aud, W_prof, W_rec,
                                   Phi_int_historia, historial_eficiencia, dt):
    DIM_TIME = region_int.shape[1]
    
    assert region_int.shape[0] == DIM_INTERNA, f"region_int dim {region_int.shape[0]} != {DIM_INTERNA}"
    assert region_aud.shape[0] == DIM_AUD, f"region_aud dim {region_aud.shape[0]} != {DIM_AUD}"
    
    prediccion_prof = W_prof @ region_aud
    error_prof = np.mean((prediccion_prof - region_int) ** 2)
    
    correlacion_prof = (region_int @ region_aud.T) / DIM_TIME
    dW_prof = ETA_PROFUNDA_BASE * correlacion_prof - TAU_PROFUNDA * W_prof
    W_prof_nueva = np.clip(W_prof + dW_prof * dt, -W_MAX, W_MAX)
    
    prediccion_rec = np.tanh(W_rec @ region_aud)
    error_rec = np.mean((prediccion_rec - region_int) ** 2)
    
    coherencia = error_prof / (error_rec + error_prof + 1e-10) if error_rec + error_prof > 0 else 0.5
    tasa_aprendizaje = ETA_RECIENTE_BASE * coherencia
    
    correlacion_rec = (region_int @ region_aud.T) / DIM_TIME
    dW_rec = tasa_aprendizaje * correlacion_rec - TAU_RECIENTE * W_rec
    W_rec_nueva = np.clip(W_rec + dW_rec * dt, -W_MAX, W_MAX)
    
    M_plast = (W_prof_nueva @ region_aud - region_int) + (W_rec_nueva @ region_aud - region_int)
    M_plast = M_plast * 0.01
    
    Phi_int_historia_nueva = (1 - 0.05) * Phi_int_historia + 0.05 * region_int
    
    return W_prof_nueva, W_rec_nueva, M_plast, float(error_rec), float(coherencia)

def calcular_senal_busqueda(Phi_total, idx):
    aud_L = Phi_total[idx['aud_L'][0]:idx['aud_L'][1], :]
    aud_R = Phi_total[idx['aud_R'][0]:idx['aud_R'][1], :]
    
    perfil_L = _perfil_espectral_region(aud_L, len(aud_L))
    perfil_R = _perfil_espectral_region(aud_R, len(aud_R))
    
    return float(np.mean(np.abs(perfil_L - perfil_R)))

def calcular_senal_mantenimiento(Phi_total, idx):
    aud_L = Phi_total[idx['aud_L'][0]:idx['aud_L'][1], :]
    aud_R = Phi_total[idx['aud_R'][0]:idx['aud_R'][1], :]
    
    varianza_media = (np.var(aud_L) + np.var(aud_R)) / 2.0
    umbral = DIFUSION_BASE ** 2
    
    return max(0.0, umbral - varianza_media), float(varianza_media)

# ============================================================
# ACTUALIZACIÓN PRINCIPAL DEL CAMPO
# ============================================================
def actualizar_campo_v82(Phi_total, Phi_vel_total, W_prof, W_rec,
                          Phi_int_historia, historial_eficiencia,
                          objetivo_L, objetivo_R,
                          idx, dt, entrenando, fraccion_L, fraccion_R, sesgo_freq):
    
    omega_natural, amort_natural = calcular_frecuencias_naturales(DIM_TOTAL)
    
    promedio_vecinos = calcular_promedio_vecinos_v82(Phi_total, idx)
    difusion = DIFUSION_BASE * (promedio_vecinos - Phi_total)
    desviacion = Phi_total - promedio_vecinos
    reaccion = GANANCIA_REACCION * desviacion * (1 - desviacion**2)
    term_osc = (-omega_natural**2 * (Phi_total - PHI_EQUILIBRIO)
                - amort_natural * Phi_vel_total)
    
    M_campo = np.zeros_like(Phi_total)
    region_int = Phi_total[idx['int'][0]:idx['int'][1], :]
    region_aud_L = Phi_total[idx['aud_L'][0]:idx['aud_L'][1], :]
    region_aud_R = Phi_total[idx['aud_R'][0]:idx['aud_R'][1], :]
    region_aud_combinada = (region_aud_L + region_aud_R) / 2.0
    
    W_prof_nueva, W_rec_nueva, M_plast, error_rec, coherencia = \
        aplicar_plasticidad_dual_v82(
            region_int, region_aud_combinada,
            W_prof, W_rec, Phi_int_historia, historial_eficiencia, dt
        )
    M_campo[idx['int'][0]:idx['int'][1], :] = M_plast
    
    # Aplicar entrada cualitativa
    Phi_total = aplicar_entrada_cualitativa(
        objetivo_L, objetivo_R, Phi_total, idx,
        fraccion_L, fraccion_R, sesgo_freq
    )
    
    dPhi_vel = term_osc + reaccion + difusion + M_campo
    Phi_vel_nueva = Phi_vel_total + dt * dPhi_vel
    Phi_nueva = Phi_total + dt * Phi_vel_nueva
    
    varianza_int = np.var(Phi_nueva[idx['int'][0]:idx['int'][1], :])
    if varianza_int < DIFUSION_BASE * 1e-4:
        ruido = np.random.normal(0, 0.01, (DIM_INTERNA, DIM_TIME))
        Phi_nueva[idx['int'][0]:idx['int'][1], :] += ruido
    
    error_equilibrio = DIFUSION_BASE ** 2
    lf_activa = error_rec > error_equilibrio
    
    Phi_int_historia_nueva = (1 - 0.05) * Phi_int_historia + 0.05 * region_int
    
    return (np.clip(Phi_nueva, LIMITE_MIN, LIMITE_MAX),
            np.clip(Phi_vel_nueva, -5.0, 5.0),
            W_prof_nueva, W_rec_nueva, Phi_int_historia_nueva,
            lf_activa, error_rec, coherencia)

# ============================================================
# SIMULACIÓN DE FASE
# ============================================================
def simular_fase(Phi_total, Phi_vel_total, W_prof, W_rec,
                  Phi_int_historia, historial_eficiencia, explorador,
                  estimulo_file, duracion, idx, dt):
    
    sr, canal_L, canal_R = cargar_audio_estereo(estimulo_file, duracion)
    ventana_muestras = int(sr * VENTANA_FFT_MS / 1000)
    hop_muestras = int(sr * HOP_FFT_MS / 1000)
    n_pasos = int(duracion / dt)
    
    metricas = []
    lf_hist = []
    w_rec_norma_hist = []
    w_prof_norma_hist = []
    fraccion_L_hist = []
    fraccion_R_hist = []
    sesgo_hist = []
    asimetria_hist = []
    pasos_exp_hist = []
    mejor_efic_hist = []
    
    for paso in range(n_pasos):
        objetivo_L = preparar_objetivo_canal(canal_L, sr, paso, ventana_muestras,
                                              hop_muestras, DIM_AUD, DIM_TIME)
        objetivo_R = preparar_objetivo_canal(canal_R, sr, paso, ventana_muestras,
                                              hop_muestras, DIM_AUD, DIM_TIME)
        
        # Calcular parámetros de actuación
        frac_L, frac_R, sesgo_freq, asimetria, nivel_perm = calcular_parametros_actuacion(Phi_total, idx)
        fraccion_L_hist.append(frac_L)
        fraccion_R_hist.append(frac_R)
        sesgo_hist.append(sesgo_freq)
        asimetria_hist.append(asimetria)
        
        # Calcular eficiencia actual
        region_int = Phi_total[idx['int'][0]:idx['int'][1], :]
        aud_L_state = Phi_total[idx['aud_L'][0]:idx['aud_L'][1], :]
        ged_actual = calcular_ged_entre(region_int, aud_L_state)
        eficiencia_actual, variacion = calcular_eficiencia_regimen(Phi_total, DIM_INTERNA, ged_actual)
        
        # Calcular tasa de olvido
        tasa_olvido, ventaja, _ = calcular_tasa_olvido_por_eficiencia(
            eficiencia_actual, historial_eficiencia
        )
        
        # Actualizar campo
        (Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia,
         lf_activa, error_rec, coherencia) = actualizar_campo_v82(
            Phi_total, Phi_vel_total, W_prof, W_rec,
            Phi_int_historia, historial_eficiencia,
            objetivo_L, objetivo_R,
            idx, dt, False, frac_L, frac_R, sesgo_freq
        )
        
        # Actualizar historial de eficiencia
        historial_eficiencia.append(eficiencia_actual)
        if len(historial_eficiencia) > TAU_EFICIENCIA * 2:
            historial_eficiencia.pop(0)
        
        # Actualizar explorador
        explorador.actualizar(lf_activa, eficiencia_actual, frac_L, frac_R, sesgo_freq)
        
        # Aplicar exploración activa
        Phi_total = explorar_actuadores(Phi_total, idx, explorador, lf_activa, eficiencia_actual, dt)
        
        # Calcular métricas adicionales
        G = Phi_total[idx['G'][0]:idx['G'][1], :]
        G_actividad = float(np.mean(np.abs(G)))
        asimetria_LR = calcular_senal_busqueda(Phi_total, idx)
        senal_mant, var_aud = calcular_senal_mantenimiento(Phi_total, idx)
        
        metricas.append({
            'ged_L': ged_actual,
            'ged_R': calcular_ged_entre(region_int, Phi_total[idx['aud_R'][0]:idx['aud_R'][1], :]),
            'frac_L': frac_L,
            'frac_R': frac_R,
            'sesgo_freq': sesgo_freq,
            'asimetria_geom': asimetria,
            'asimetria_LR': asimetria_LR,
            'G_actividad': G_actividad,
            'perm_nivel': nivel_perm,
            'senal_mant': senal_mant,
            'var_aud_media': var_aud,
            'eficiencia': eficiencia_actual,
            'error_rec': error_rec
        })
        lf_hist.append(lf_activa)
        w_rec_norma_hist.append(np.mean(np.abs(W_rec)))
        w_prof_norma_hist.append(np.mean(np.abs(W_prof)))
        pasos_exp_hist.append(explorador.pasos_en_lf)
        mejor_efic_hist.append(explorador.mejor_eficiencia)
        
        if paso % 200 == 0:
            print(f"    t={paso*dt:.1f}s | GED={ged_actual:.4f} | "
                  f"fL={frac_L:.3f} | fR={frac_R:.3f} | sesgo={sesgo_freq:+.3f} | "
                  f"asim={asimetria_LR:.4f} | G_act={G_actividad:.4f} | "
                  f"efic={eficiencia_actual:.3f} | LF={'ACTIVA' if lf_activa else 'inact'} | "
                  f"exp={explorador.pasos_en_lf}")
    
    return {
        'metricas': metricas,
        'lf_pct': 100 * np.mean(lf_hist),
        'w_rec_norma': np.mean(w_rec_norma_hist),
        'w_prof_norma': np.mean(w_prof_norma_hist),
        'fraccion_L_mean': np.mean(fraccion_L_hist),
        'fraccion_R_mean': np.mean(fraccion_R_hist),
        'sesgo_mean': np.mean(sesgo_hist),
        'pasos_exp_total': sum(pasos_exp_hist),
        'mejor_eficiencia': explorador.mejor_eficiencia,
        'phi_total': Phi_total,
        'w_prof': W_prof,
        'w_rec': W_rec,
        'phi_int_historia': Phi_int_historia,
        'explorador': explorador
    }

# ============================================================
# ENTRENAMIENTO INICIAL
# ============================================================
def entrenar_inicial(duracion=30.0):
    print("\n[Fase 1] Entrenamiento (voz, alpha=0.05, 30s)")
    
    Phi_total, Phi_vel_total = inicializar_campo_v82()
    W_prof, W_rec, Phi_int_historia = inicializar_memorias()
    historial_eficiencia = []
    explorador = ExploradorActuadores()
    
    sr, canal_L, canal_R = cargar_audio_estereo("Voz_Estudio.wav", duracion)
    ventana_muestras = int(sr * VENTANA_FFT_MS / 1000)
    hop_muestras = int(sr * HOP_FFT_MS / 1000)
    n_pasos = int(duracion / DT)
    
    errores = []
    
    for paso in range(n_pasos):
        objetivo_L = preparar_objetivo_canal(canal_L, sr, paso, ventana_muestras,
                                              hop_muestras, DIM_AUD, DIM_TIME)
        objetivo_R = preparar_objetivo_canal(canal_R, sr, paso, ventana_muestras,
                                              hop_muestras, DIM_AUD, DIM_TIME)
        
        frac_L, frac_R, sesgo_freq, _, _ = calcular_parametros_actuacion(Phi_total, idx)
        
        Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia, _, error_rec, _ = \
            actualizar_campo_v82(
                Phi_total, Phi_vel_total, W_prof, W_rec,
                Phi_int_historia, [], objetivo_L, objetivo_R,
                idx, DT, True, frac_L, frac_R, sesgo_freq
            )
        
        errores.append(error_rec)
        
        if paso % 500 == 0:
            print(f"    Paso {paso}/{n_pasos}, error={error_rec:.6f}")
    
    error_equilibrio = np.min(errores)
    print(f"  ERROR_EQUILIBRIO medido: {error_equilibrio:.6f}")
    print(f"  W_prof tras entreno: {np.mean(np.abs(W_prof)):.4f}")
    print(f"  W_rec tras entreno: {np.mean(np.abs(W_rec)):.4f}")
    
    return Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia, error_equilibrio, explorador

# ============================================================
# MAIN
# ============================================================
def main():
    Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia, ERROR_EQUILIBRIO, explorador_global = entrenar_inicial()
    
    fases = [
        ("Fase 2", "Voz_Estudio.wav", "Dominio (voz)"),
        ("Fase 3", "Brandemburgo.wav", "No entrenado (música)"),
        ("Fase 4", "Tono puro", "No entrenado (tono)"),
        ("Fase 5", "Voz+Viento_1.wav", "Degradado"),
        ("Fase 6", "Ruido blanco", "Perturbación basal"),
        ("Fase 7", "Voz_Estudio.wav", "Re-acoplamiento"),
    ]
    
    resultados = []
    historial_eficiencia_global = []
    
    for fase_id, archivo, desc in fases:
        print(f"\n[{fase_id}] {desc}")
        
        res = simular_fase(
            Phi_total, Phi_vel_total, W_prof, W_rec,
            Phi_int_historia, historial_eficiencia_global, explorador_global,
            archivo, 20.0, idx, DT
        )
        
        resultados.append(res)
        
        Phi_total = res['phi_total']
        W_prof = res['w_prof']
        W_rec = res['w_rec']
        Phi_int_historia = res['phi_int_historia']
        explorador_global = res['explorador']
        
        m = res['metricas']
        print(f"\n  Resumen {fase_id}:")
        print(f"    GED L/R medios:        {np.mean([mm['ged_L'] for mm in m]):.4f} / {np.mean([mm['ged_R'] for mm in m]):.4f}")
        print(f"    Fracción L/R medias:   {res['fraccion_L_mean']:.3f} / {res['fraccion_R_mean']:.3f}")
        print(f"    Sesgo frecuencial:     {res['sesgo_mean']:+.4f}")
        print(f"    Ganglio actividad:     {np.mean([mm['G_actividad'] for mm in m]):.4f}")
        print(f"    Asimetría L/R real:    {np.mean([mm['asimetria_LR'] for mm in m]):.4f}")
        print(f"    Eficiencia media:      {np.mean([mm['eficiencia'] for mm in m]):.4f}")
        print(f"    LF activa (%):         {res['lf_pct']:.1f}%")
        print(f"    Mejor eficiencia exp:  {res['mejor_eficiencia']:.4f}")
        print(f"    Pasos en exploración:  {res['pasos_exp_total']}")
    
    print("\n" + "=" * 100)
    print("DIAGNÓSTICO — v82 Acoplamiento activo")
    print("=" * 100)
    
    # Criterios
    G_actividades = []
    for res in resultados:
        G_actividades.extend([m['G_actividad'] for m in res['metricas']])
    criterio10 = np.mean(G_actividades) > 0.01
    
    fracciones = []
    for res in resultados:
        fracciones.extend([m['frac_L'] for m in res['metricas']])
    criterio11 = np.std(fracciones) > 0.01
    
    asims_f2 = [m['asimetria_LR'] for m in resultados[0]['metricas']]
    asims_f6 = [m['asimetria_LR'] for m in resultados[4]['metricas']]
    criterio12 = abs(np.mean(asims_f2) - np.mean(asims_f6)) > 0.001
    
    mants = []
    for res in resultados:
        mants.extend([m['senal_mant'] for m in res['metricas']])
    criterio13 = any(m > 0.001 for m in mants)
    
    fracciones_por_fase = [res['fraccion_L_mean'] for res in resultados]
    criterio14 = np.std(fracciones_por_fase) > 0.01
    
    asim_f2 = np.mean([m['asimetria_LR'] for m in resultados[0]['metricas']])
    criterio15 = abs(asim_f2) > 0.001
    
    criterio16 = resultados[-1]['mejor_eficiencia'] > 0
    
    print("\n  CRITERIOS v81 (arquitectura):")
    print(f"    C10 — Ganglio activo:          {'✅' if criterio10 else '❌'} (media={np.mean(G_actividades):.4f})")
    print(f"    C11 — Alpha modulado:          {'✅' if criterio11 else '❌'} (std={np.std(fracciones):.4f})")
    print(f"    C12 — Asimetría diferencial:   {'✅' if criterio12 else '❌'} (diff={abs(np.mean(asims_f2)-np.mean(asims_f6)):.6f})")
    print(f"    C13 — Mantenimiento activo:    {'✅' if criterio13 else '❌'} (max={max(mants):.6f})")
    
    print("\n  CRITERIOS v82 (acoplamiento activo):")
    print(f"    C14 — Fracción L/R varía:      {'✅' if criterio14 else '❌'} (std={np.std(fracciones_por_fase):.4f})")
    print(f"    C15 — Asimetría L/R emerge:    {'✅' if criterio15 else '❌'} (asim_F2={asim_f2:.6f})")
    print(f"    C16 — Explorador encuentra:    {'✅' if criterio16 else '❌'} (mejor_ef={resultados[-1]['mejor_eficiencia']:.4f})")
    
    print("\n  VEREDICTO:")
    if all([criterio10, criterio11, criterio12, criterio13, criterio14, criterio15, criterio16]):
        print("  ✅ ACOPLAMIENTO ACTIVO VALIDADO")
        print("     El campo recibe entrada estéreo real.")
        print("     Los actuadores modifican cualitativamente el acoplamiento.")
        print("     La exploración deliberada bajo LF funciona con memoria.")
    elif criterio14 and criterio15:
        print("  ✅ ACOPLAMIENTO FUNCIONAL")
        print("     La actuación modula la entrada cualitativamente.")
        print("     La asimetría L/R emerge de la entrada estéreo.")
    else:
        print("  ⚠️ VALIDACIÓN PARCIAL")
        print("     Algunos criterios no se cumplen aún.")
    
    # Guardar CSV
    with open('v82_acoplamiento_activo.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['fase', 't', 'ged_L', 'ged_R', 'frac_L', 'frac_R',
                         'sesgo_freq', 'asimetria_LR', 'G_actividad',
                         'senal_mant', 'eficiencia', 'error_rec'])
        
        for fase_idx, (fase, res) in enumerate(zip(fases, resultados)):
            for t_idx, m in enumerate(res['metricas']):
                writer.writerow([
                    fase[0], t_idx * DT,
                    m['ged_L'], m['ged_R'], m['frac_L'], m['frac_R'],
                    m['sesgo_freq'], m['asimetria_LR'], m['G_actividad'],
                    m['senal_mant'], m['eficiencia'], m['error_rec']
                ])
    
    print("\n  CSV guardado: v82_acoplamiento_activo.csv")
    
    # Gráfico
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    nombres = [f[0] for f in fases]
    
    GED_L = [np.mean([m['ged_L'] for m in res['metricas']]) for res in resultados]
    GED_R = [np.mean([m['ged_R'] for m in res['metricas']]) for res in resultados]
    axes[0,0].plot(nombres, GED_L, 'o-', label='GED L')
    axes[0,0].plot(nombres, GED_R, 's-', label='GED R')
    axes[0,0].set_title('GED por canal')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    frac_L = [res['fraccion_L_mean'] for res in resultados]
    frac_R = [res['fraccion_R_mean'] for res in resultados]
    axes[0,1].plot(nombres, frac_L, 'o-', label='Fracción L')
    axes[0,1].plot(nombres, frac_R, 's-', label='Fracción R')
    axes[0,1].set_title('Fracción de ventana (permeabilidad)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    sesgo = [res['sesgo_mean'] for res in resultados]
    axes[0,2].bar(nombres, sesgo)
    axes[0,2].set_title('Sesgo frecuencial')
    axes[0,2].grid(True, alpha=0.3)
    
    G_act = [np.mean([m['G_actividad'] for m in res['metricas']]) for res in resultados]
    axes[0,3].bar(nombres, G_act)
    axes[0,3].set_title('Actividad del ganglio')
    axes[0,3].grid(True, alpha=0.3)
    
    asim_LR = [np.mean([m['asimetria_LR'] for m in res['metricas']]) for res in resultados]
    axes[1,0].bar(nombres, asim_LR)
    axes[1,0].set_title('Asimetría L/R (dirección)')
    axes[1,0].grid(True, alpha=0.3)
    
    efic = [np.mean([m['eficiencia'] for m in res['metricas']]) for res in resultados]
    axes[1,1].bar(nombres, efic)
    axes[1,1].set_title('Eficiencia de régimen')
    axes[1,1].grid(True, alpha=0.3)
    
    lf_pct = [res['lf_pct'] for res in resultados]
    axes[1,2].bar(nombres, lf_pct)
    axes[1,2].set_title('LF activa (%)')
    axes[1,2].grid(True, alpha=0.3)
    
    mejor_exp = [res['mejor_eficiencia'] for res in resultados]
    axes[1,3].bar(nombres, mejor_exp)
    axes[1,3].set_title('Mejor eficiencia explorada')
    axes[1,3].grid(True, alpha=0.3)
    
    plt.suptitle('VSTCosmos v82 — Acoplamiento activo', fontsize=14)
    plt.tight_layout()
    plt.savefig('v82_acoplamiento_activo.png', dpi=150)
    print("  Gráfico guardado: v82_acoplamiento_activo.png")
    
    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)


if __name__ == "__main__":
    main()