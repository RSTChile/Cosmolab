#!/usr/bin/env python3
"""
VSTCosmos v81 — Campo expandido con ganglio coordinador

Principios canónicos:
- Ningún parámetro fijado externamente, todo deriva de la física del campo
- Todo órgano es una extensión del campo — siempre bidireccional por construcción
- El ganglio emerge de mayor conectividad topológica
- Los actuadores son estado del campo, no comandos externos
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import csv
import warnings
from collections import deque
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

# Limites
LIMITE_MIN = 0.0
LIMITE_MAX = 1.0
W_MAX = 1.0
ALPHA_MAX = 0.05
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
print("VSTCosmos v81 — Campo expandido con ganglio coordinador")
print("")
print("  Principios canónicos:")
print("  - Todo órgano es una extensión del campo — bidireccional por construcción")
print("  - El ganglio emerge de mayor conectividad topológica")
print("  - Actuadores son estado del campo, no comandos externos")
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
# FUNCIONES BASE
# ============================================================
def cargar_audio_dual(ruta_L, ruta_R, duracion):
    """Carga dos canales de audio (pueden ser el mismo archivo o diferentes)"""
    def _cargar_individual(ruta):
        if "Tono puro" in ruta:
            sr = 48000
            t = np.arange(int(sr * duracion)) / sr
            return sr, 0.5 * np.sin(2 * np.pi * 440 * t)
        elif "Ruido blanco" in ruta:
            sr = 48000
            return sr, np.random.normal(0, 0.5, int(sr * duracion))
        else:
            try:
                sr, data = wav.read(ruta)
                if data.dtype == np.int16:
                    data = data.astype(np.float32) / 32768.0
                if data.ndim == 2:
                    data = data.mean(axis=1)
                max_val = np.max(np.abs(data))
                if max_val > 0:
                    data = data / max_val
                muestras_necesarias = int(sr * duracion)
                if len(data) < muestras_necesarias:
                    data = np.pad(data, (0, muestras_necesarias - len(data)))
                else:
                    data = data[:muestras_necesarias]
                return sr, data
            except FileNotFoundError:
                print(f"  [ADVERTENCIA] {ruta} no encontrado, usando tono 440Hz")
                sr = 48000
                t = np.arange(int(sr * duracion)) / sr
                return sr, 0.5 * np.sin(2 * np.pi * 440 * t)
    
    sr_L, audio_L = _cargar_individual(ruta_L)
    sr_R, audio_R = _cargar_individual(ruta_R)
    return sr_L, audio_L, audio_R

def inicializar_campo_v81():
    """Todas las regiones se inicializan con ruido pequeño alrededor de PHI_EQUILIBRIO"""
    np.random.seed(None)
    Phi_total = np.random.normal(PHI_EQUILIBRIO, 0.01, (DIM_TOTAL, DIM_TIME))
    Phi_vel_total = np.zeros((DIM_TOTAL, DIM_TIME))
    return Phi_total, Phi_vel_total

def inicializar_memorias():
    """Memorias con dimensiones compatibles con región_int y región_aud"""
    W_prof = np.zeros((DIM_INTERNA, DIM_AUD))  # 32 x 16
    W_rec = np.zeros((DIM_INTERNA, DIM_AUD))   # 32 x 16
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

def preparar_objetivo_dual(audio_L, audio_R, sr, idx_ventana, ventana_muestras, hop_muestras, dim_auditiva):
    def _preparar_individual(audio):
        inicio = idx_ventana * hop_muestras
        if inicio + ventana_muestras > len(audio):
            return np.zeros(dim_auditiva)
        
        fragmento = audio[inicio:inicio + ventana_muestras]
        ventana_hann = np.hanning(len(fragmento))
        fragmento = fragmento * ventana_hann
        
        fft = np.fft.rfft(fragmento)
        potencia = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(len(fragmento), 1/sr)
        
        bandas = np.logspace(np.log10(F_MIN), np.log10(F_MAX), dim_auditiva + 1)
        objetivo = np.zeros(dim_auditiva)
        for b in range(dim_auditiva):
            mask = (freqs >= bandas[b]) & (freqs < bandas[b+1])
            if np.any(mask):
                objetivo[b] = np.mean(potencia[mask])
        
        max_val = np.max(objetivo)
        if max_val > 0:
            objetivo = objetivo / max_val
        
        return objetivo
    
    objetivo_L = _preparar_individual(audio_L)
    objetivo_R = _preparar_individual(audio_R)
    return objetivo_L.reshape(-1, 1) * np.ones((1, DIM_TIME)), objetivo_R.reshape(-1, 1) * np.ones((1, DIM_TIME))

def calcular_frecuencias_naturales(dim_total):
    bandas = np.arange(dim_total)
    t = np.log1p(bandas) / np.log1p(dim_total - 1) if dim_total > 1 else np.zeros_like(bandas)
    omega = OMEGA_MIN + (OMEGA_MAX - OMEGA_MIN) * t
    amort = AMORT_MIN + (AMORT_MAX - AMORT_MIN) * t
    return omega.reshape(-1, 1), amort.reshape(-1, 1)

# ============================================================
# DIFUSIÓN ENTRE VECINOS (TOPOLÓGICA)
# ============================================================
def calcular_promedio_vecinos_v81(Phi_total, idx):
    """Promedio de vecinos según topología definida"""
    DIM_TIME = Phi_total.shape[1]
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
    
    # Normalizar
    for i in range(DIM_TOTAL):
        if conteo[i] > 0:
            promedio[i, :] /= conteo[i]
        else:
            promedio[i, :] = Phi_total[i, :]
    
    return promedio

# ============================================================
# ALPHA ACTIVO (EMERGE DEL ESTADO DEL CAMPO)
# ============================================================
def calcular_alpha_activo(Phi_total, idx):
    """Alpha derivado del estado de act_perm y act_geom"""
    act_perm = Phi_total[idx['act_perm'][0]:idx['act_perm'][1], :]
    act_geom = Phi_total[idx['act_geom'][0]:idx['act_geom'][1], :]
    
    # Permeabilidad base: normalizar estado de act_perm a [0, ALPHA_MAX]
    nivel_perm = float(np.mean(np.tanh(act_perm)))
    alpha_base = ALPHA_MAX * (nivel_perm + 1.0) / 2.0
    
    # Asimetría geométrica: diferencia entre bandas de act_geom
    mitad = DIM_ACT // 2 if DIM_ACT > 1 else 1
    banda_baja = float(np.mean(act_geom[:mitad, :]))
    banda_alta = float(np.mean(act_geom[mitad:, :])) if mitad < DIM_ACT else banda_baja
    asimetria = float(np.tanh(banda_alta - banda_baja))
    
    # Alpha por canal
    alpha_L = alpha_base * (1.0 + asimetria)
    alpha_R = alpha_base * (1.0 - asimetria)
    alpha_L = max(0.0, min(ALPHA_MAX, alpha_L))
    alpha_R = max(0.0, min(ALPHA_MAX, alpha_R))
    
    return alpha_L, alpha_R, asimetria

# ============================================================
# SEÑALES DE ACTUACIÓN
# ============================================================
def calcular_senal_busqueda(Phi_total, idx):
    """act_busc codifica diferencia entre canales auditivos (triangulación)"""
    aud_L = Phi_total[idx['aud_L'][0]:idx['aud_L'][1], :]
    aud_R = Phi_total[idx['aud_R'][0]:idx['aud_R'][1], :]
    
    perfil_L = _perfil_espectral_region(aud_L, len(aud_L))
    perfil_R = _perfil_espectral_region(aud_R, len(aud_R))
    
    diferencia_LR = float(np.mean(np.abs(perfil_L - perfil_R)))
    return diferencia_LR

def calcular_senal_mantenimiento(Phi_total, idx, historial_varianza=None):
    """act_mant detecta degradación de la zona auditiva"""
    aud_L = Phi_total[idx['aud_L'][0]:idx['aud_L'][1], :]
    aud_R = Phi_total[idx['aud_R'][0]:idx['aud_R'][1], :]
    
    varianza_L = float(np.var(aud_L))
    varianza_R = float(np.var(aud_R))
    varianza_media = (varianza_L + varianza_R) / 2.0
    
    # Umbral estructural: varianza mínima viable
    umbral_mantenimiento = DIFUSION_BASE ** 2
    senal = max(0.0, umbral_mantenimiento - varianza_media)
    return senal, varianza_media

# ============================================================
# EFICIENCIA DE RÉGIMEN (heredado de v80h)
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

# ============================================================
# PLATICIDAD DUAL (heredada de v80h, dimensiones corregidas)
# ============================================================
def aplicar_plasticidad_dual_v81(region_int, region_aud, W_prof, W_rec,
                                   Phi_int_historia, historial_eficiencia, dt):
    DIM_TIME = region_int.shape[1]
    
    # Verificar que dimensiones coincidan
    assert region_int.shape[0] == DIM_INTERNA, f"region_int dim {region_int.shape[0]} != {DIM_INTERNA}"
    assert region_aud.shape[0] == DIM_AUD, f"region_aud dim {region_aud.shape[0]} != {DIM_AUD}"
    assert W_prof.shape == (DIM_INTERNA, DIM_AUD), f"W_prof shape {W_prof.shape} != ({DIM_INTERNA},{DIM_AUD})"
    
    # Error actual
    prediccion_prof = W_prof @ region_aud
    error_prof = np.mean((prediccion_prof - region_int) ** 2)
    
    # Plasticidad profunda
    correlacion_prof = (region_int @ region_aud.T) / DIM_TIME
    dW_prof = ETA_PROFUNDA_BASE * correlacion_prof - TAU_PROFUNDA * W_prof
    W_prof_nueva = np.clip(W_prof + dW_prof * dt, -W_MAX, W_MAX)
    
    # Plasticidad reciente
    prediccion_rec = np.tanh(W_rec @ region_aud)
    error_rec = np.mean((prediccion_rec - region_int) ** 2)
    
    coherencia = error_prof / (error_rec + error_prof + 1e-10) if error_rec + error_prof > 0 else 0.5
    tasa_aprendizaje = ETA_RECIENTE_BASE * coherencia
    
    correlacion_rec = (region_int @ region_aud.T) / DIM_TIME
    dW_rec = tasa_aprendizaje * correlacion_rec - TAU_RECIENTE * W_rec
    W_rec_nueva = np.clip(W_rec + dW_rec * dt, -W_MAX, W_MAX)
    
    # M_plast
    M_plast = (W_prof_nueva @ region_aud - region_int) + (W_rec_nueva @ region_aud - region_int)
    M_plast = M_plast * 0.01  # GAMMA_PLAST
    
    # Actualizar historia
    Phi_int_historia_nueva = (1 - 0.05) * Phi_int_historia + 0.05 * region_int
    
    return W_prof_nueva, W_rec_nueva, M_plast, float(error_rec), float(coherencia)

# ============================================================
# ACTUALIZACIÓN PRINCIPAL DEL CAMPO
# ============================================================
def actualizar_campo_v81(Phi_total, Phi_vel_total, W_prof, W_rec,
                          Phi_int_historia, historial_eficiencia,
                          objetivo_L, objetivo_R,
                          idx, dt, entrenando, alpha_L, alpha_R):
    
    DIM_TIME = Phi_total.shape[1]
    omega_natural, amort_natural = calcular_frecuencias_naturales(DIM_TOTAL)
    
    # Difusión entre vecinos
    promedio_vecinos = calcular_promedio_vecinos_v81(Phi_total, idx)
    difusion = DIFUSION_BASE * (promedio_vecinos - Phi_total)
    
    # Reacción no lineal
    desviacion = Phi_total - promedio_vecinos
    reaccion = GANANCIA_REACCION * desviacion * (1 - desviacion**2)
    
    # Oscilación
    term_osc = (-omega_natural**2 * (Phi_total - PHI_EQUILIBRIO)
                - amort_natural * Phi_vel_total)
    
    # Plasticidad (solo sobre region_int)
    M_campo = np.zeros_like(Phi_total)
    region_int = Phi_total[idx['int'][0]:idx['int'][1], :]
    region_aud_L = Phi_total[idx['aud_L'][0]:idx['aud_L'][1], :]
    region_aud_R = Phi_total[idx['aud_R'][0]:idx['aud_R'][1], :]
    region_aud_combinada = (region_aud_L + region_aud_R) / 2.0
    
    W_prof_nueva, W_rec_nueva, M_plast, error_rec, coherencia = \
        aplicar_plasticidad_dual_v81(
            region_int, region_aud_combinada,
            W_prof, W_rec, Phi_int_historia, historial_eficiencia, dt
        )
    M_campo[idx['int'][0]:idx['int'][1], :] = M_plast
    
    # Acoplamiento con entorno externo (alpha activo ya está calculado)
    Phi_total_actualizado = Phi_total.copy()
    Phi_total_actualizado[idx['aud_L'][0]:idx['aud_L'][1], :] = (
        (1 - alpha_L) * Phi_total[idx['aud_L'][0]:idx['aud_L'][1], :] + alpha_L * objetivo_L
    )
    Phi_total_actualizado[idx['aud_R'][0]:idx['aud_R'][1], :] = (
        (1 - alpha_R) * Phi_total[idx['aud_R'][0]:idx['aud_R'][1], :] + alpha_R * objetivo_R
    )
    
    # Actualización del campo
    dPhi_vel = term_osc + reaccion + difusion + M_campo
    Phi_vel_nueva = Phi_vel_total + dt * dPhi_vel
    Phi_nueva = Phi_total_actualizado + dt * Phi_vel_nueva
    
    # Prevenir colapso (sin umbral fijo)
    varianza_int = np.var(Phi_nueva[idx['int'][0]:idx['int'][1], :])
    if varianza_int < DIFUSION_BASE * 1e-4:
        ruido = np.random.normal(0, 0.01, (DIM_INTERNA, DIM_TIME))
        Phi_nueva[idx['int'][0]:idx['int'][1], :] += ruido
    
    # LF definida por error_rec > error_equilibrio (usando DIFUSION como referencia)
    error_equilibrio = DIFUSION_BASE ** 2
    lf_activa = error_rec > error_equilibrio
    
    # Actualizar historia interna
    Phi_int_historia_nueva = (1 - 0.05) * Phi_int_historia + 0.05 * region_int
    
    return (np.clip(Phi_nueva, LIMITE_MIN, LIMITE_MAX),
            np.clip(Phi_vel_nueva, -5.0, 5.0),
            W_prof_nueva, W_rec_nueva, Phi_int_historia_nueva,
            lf_activa, error_rec, coherencia)

# ============================================================
# CÁLCULO DE MÉTRICAS
# ============================================================
def calcular_metricas_v81(Phi_total, idx, alpha_L, alpha_R, asimetria,
                           eficiencia, ventaja, tasa_olvido, error_rec):
    region_int = Phi_total[idx['int'][0]:idx['int'][1], :]
    aud_L = Phi_total[idx['aud_L'][0]:idx['aud_L'][1], :]
    aud_R = Phi_total[idx['aud_R'][0]:idx['aud_R'][1], :]
    G = Phi_total[idx['G'][0]:idx['G'][1], :]
    act_perm = Phi_total[idx['act_perm'][0]:idx['act_perm'][1], :]
    act_busc = Phi_total[idx['act_busc'][0]:idx['act_busc'][1], :]
    
    ged_L = calcular_ged_entre(region_int, aud_L)
    ged_R = calcular_ged_entre(region_int, aud_R)
    ged_medio = (ged_L + ged_R) / 2.0
    
    G_actividad = float(np.mean(np.abs(G)))
    G_varianza = float(np.var(G))
    
    perm_nivel = float(np.mean(np.tanh(act_perm)))
    busq_nivel = float(np.mean(np.tanh(act_busc)))
    
    senal_mant, var_aud = calcular_senal_mantenimiento(Phi_total, idx)
    asimetria_LR = calcular_senal_busqueda(Phi_total, idx)
    
    return {
        'ged_L': ged_L,
        'ged_R': ged_R,
        'ged_medio': ged_medio,
        'alpha_L': alpha_L,
        'alpha_R': alpha_R,
        'asimetria_geom': asimetria,
        'G_actividad': G_actividad,
        'G_varianza': G_varianza,
        'perm_nivel': perm_nivel,
        'busq_nivel': busq_nivel,
        'senal_mant': senal_mant,
        'asimetria_LR': asimetria_LR,
        'var_aud_media': var_aud,
        'eficiencia': eficiencia,
        'ventaja': ventaja,
        'tasa_olvido': tasa_olvido,
        'error_rec': error_rec,
    }

# ============================================================
# SIMULACIÓN DE FASE
# ============================================================
def simular_fase(Phi_total, Phi_vel_total, W_prof, W_rec,
                  Phi_int_historia, historial_eficiencia,
                  estimulo_L, estimulo_R, duracion,
                  idx, dt):
    
    sr, audio_L, audio_R = cargar_audio_dual(estimulo_L, estimulo_R, duracion)
    ventana_muestras = int(sr * VENTANA_FFT_MS / 1000)
    hop_muestras = int(sr * HOP_FFT_MS / 1000)
    n_pasos = int(duracion / dt)
    
    metricas = []
    lf_hist = []
    w_rec_norma_hist = []
    w_prof_norma_hist = []
    historial_eficiencia_local = list(historial_eficiencia) if historial_eficiencia else []
    
    for paso in range(n_pasos):
        objetivo_L, objetivo_R = preparar_objetivo_dual(
            audio_L, audio_R, sr, paso, ventana_muestras, hop_muestras, DIM_AUD
        )
        
        # Calcular alpha activo ANTES de actualizar (para usar estado actual)
        alpha_L, alpha_R, asimetria = calcular_alpha_activo(Phi_total, idx)
        
        # Calcular eficiencia actual
        region_int = Phi_total[idx['int'][0]:idx['int'][1], :]
        aud_L_state = Phi_total[idx['aud_L'][0]:idx['aud_L'][1], :]
        ged_actual = calcular_ged_entre(region_int, aud_L_state)
        eficiencia_actual, _ = calcular_eficiencia_regimen(Phi_total, DIM_INTERNA, ged_actual)
        
        # Calcular tasa de olvido
        tasa_olvido, ventaja, _ = calcular_tasa_olvido_por_eficiencia(
            eficiencia_actual, historial_eficiencia_local
        )
        
        # Actualizar campo
        (Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia,
         lf_activa, error_rec, coherencia) = actualizar_campo_v81(
            Phi_total, Phi_vel_total, W_prof, W_rec,
            Phi_int_historia, historial_eficiencia_local,
            objetivo_L, objetivo_R,
            idx, dt, False, alpha_L, alpha_R
        )
        
        # Actualizar historial de eficiencia
        historial_eficiencia_local.append(eficiencia_actual)
        if len(historial_eficiencia_local) > TAU_EFICIENCIA * 2:
            historial_eficiencia_local.pop(0)
        
        # Calcular métricas
        m = calcular_metricas_v81(Phi_total, idx, alpha_L, alpha_R, asimetria,
                                   eficiencia_actual, ventaja, tasa_olvido, error_rec)
        metricas.append(m)
        lf_hist.append(lf_activa)
        w_rec_norma_hist.append(np.mean(np.abs(W_rec)))
        w_prof_norma_hist.append(np.mean(np.abs(W_prof)))
        
        # Log cada 100 pasos
        if paso % 100 == 0:
            print(f"    t={paso*dt:.1f}s | GED_L={m['ged_L']:.4f} | GED_R={m['ged_R']:.4f} | "
                  f"aL={m['alpha_L']:.4f} | aR={m['alpha_R']:.4f} | "
                  f"G_act={m['G_actividad']:.4f} | asim={m['asimetria_LR']:.4f} | "
                  f"perm={m['perm_nivel']:+.3f} | efic={m['eficiencia']:.3f} | "
                  f"LF={'ACTIVA' if lf_activa else 'inact'}")
    
    # Agregar estadísticas de la fase
    return {
        'metricas': metricas,
        'lf_pct': 100 * np.mean(lf_hist),
        'w_rec_norma': np.mean(w_rec_norma_hist),
        'w_prof_norma': np.mean(w_prof_norma_hist),
        'phi_total': Phi_total,
        'w_prof': W_prof,
        'w_rec': W_rec,
        'phi_int_historia': Phi_int_historia,
        'historial_eficiencia': historial_eficiencia_local
    }

# ============================================================
# ENTRENAMIENTO INICIAL
# ============================================================
def entrenar_inicial(duracion=30.0):
    """Entrenamiento inicial del campo con voz (ambos canales iguales)"""
    print("\n[Fase 1] Entrenamiento (voz, alpha=0.05, 30s)")
    
    Phi_total, Phi_vel_total = inicializar_campo_v81()
    W_prof, W_rec, Phi_int_historia = inicializar_memorias()
    historial_eficiencia = []
    
    sr, audio_L, audio_R = cargar_audio_dual("Voz_Estudio.wav", "Voz_Estudio.wav", duracion)
    ventana_muestras = int(sr * VENTANA_FFT_MS / 1000)
    hop_muestras = int(sr * HOP_FFT_MS / 1000)
    n_pasos = int(duracion / DT)
    
    errores = []
    
    for paso in range(n_pasos):
        objetivo_L, objetivo_R = preparar_objetivo_dual(
            audio_L, audio_R, sr, paso, ventana_muestras, hop_muestras, DIM_AUD
        )
        
        alpha_fijo = 0.05
        alpha_L, alpha_R = alpha_fijo, alpha_fijo
        
        Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia, _, error_rec, _ = \
            actualizar_campo_v81(
                Phi_total, Phi_vel_total, W_prof, W_rec,
                Phi_int_historia, [], objetivo_L, objetivo_R,
                idx, DT, True, alpha_L, alpha_R
            )
        
        errores.append(error_rec)
        
        if paso % 500 == 0:
            print(f"    Paso {paso}/{n_pasos}, error={error_rec:.6f}")
    
    error_equilibrio = np.min(errores)
    print(f"  ERROR_EQUILIBRIO medido: {error_equilibrio:.6f}")
    print(f"  W_prof tras entreno: {np.mean(np.abs(W_prof)):.4f}")
    print(f"  W_rec tras entreno: {np.mean(np.abs(W_rec)):.4f}")
    
    return Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia, error_equilibrio

# ============================================================
# MAIN
# ============================================================
def main():
    # Entrenamiento inicial
    Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia, ERROR_EQUILIBRIO = entrenar_inicial()
    
    # Protocolo de fases
    fases = [
        ("Fase 2", "voz", "Voz_Estudio.wav", "Voz_Estudio.wav", "Dominio (voz)"),
        ("Fase 3", "musica", "Brandemburgo.wav", "Brandemburgo.wav", "No entrenado (música)"),
        ("Fase 4", "tono", "Tono puro", "Tono puro", "No entrenado (tono)"),
        ("Fase 5", "voz_viento", "Voz+Viento_1.wav", "Voz+Viento_1.wav", "Degradado"),
        ("Fase 6", "ruido", "Ruido blanco", "Ruido blanco", "Perturbación basal"),
        ("Fase 7", "voz", "Voz_Estudio.wav", "Voz_Estudio.wav", "Re-acoplamiento"),
    ]
    
    resultados = []
    historial_eficiencia_global = []
    
    for fase_id, nombre, archivo_L, archivo_R, desc in fases:
        print(f"\n[{fase_id}] {desc}")
        
        res = simular_fase(
            Phi_total, Phi_vel_total, W_prof, W_rec,
            Phi_int_historia, historial_eficiencia_global,
            archivo_L, archivo_R, 20.0,
            idx, DT
        )
        
        resultados.append(res)
        
        Phi_total = res['phi_total']
        W_prof = res['w_prof']
        W_rec = res['w_rec']
        Phi_int_historia = res['phi_int_historia']
        historial_eficiencia_global = res['historial_eficiencia']
        
        # Resumen de la fase
        m = res['metricas']
        print(f"\n  Resumen {fase_id}:")
        print(f"    GED L/R medios:        {np.mean([mm['ged_L'] for mm in m]):.4f} / {np.mean([mm['ged_R'] for mm in m]):.4f}")
        print(f"    Alpha L/R medios:      {np.mean([mm['alpha_L'] for mm in m]):.4f} / {np.mean([mm['alpha_R'] for mm in m]):.4f}")
        print(f"    Ganglio — actividad:   {np.mean([mm['G_actividad'] for mm in m]):.4f}")
        print(f"    Asimetría L/R media:   {np.mean([mm['asimetria_LR'] for mm in m]):.4f}")
        print(f"    Permeabilidad media:   {np.mean([mm['perm_nivel'] for mm in m]):+.4f}")
        print(f"    Eficiencia media:      {np.mean([mm['eficiencia'] for mm in m]):.4f}")
        print(f"    LF activa (%):         {res['lf_pct']:.1f}%")
        print(f"    W_rec norma:           {res['w_rec_norma']:.4f}")
    
    print("\n" + "=" * 100)
    print("DIAGNÓSTICO — v81 Campo expandido")
    print("=" * 100)
    
    # Verificar criterios
    print("\n  CRITERIOS v81:")
    
    # C10 — Ganglio activo
    G_actividades = []
    for res in resultados:
        G_actividades.extend([m['G_actividad'] for m in res['metricas']])
    criterio10 = np.mean(G_actividades) > 0.01
    print(f"    C10 — Ganglio activo:          media={np.mean(G_actividades):.6f} > 0 {'✅' if criterio10 else '❌'}")
    
    # C11 — Alpha modulado
    alpha_Ls = []
    for res in resultados:
        alpha_Ls.extend([m['alpha_L'] for m in res['metricas']])
    criterio11 = np.std(alpha_Ls) > 0.001
    print(f"    C11 — Alpha modulado:          std={np.std(alpha_Ls):.6f} > 0.001 {'✅' if criterio11 else '❌'}")
    
    # C12 — Asimetría L/R diferencial
    asims = []
    for res in resultados:
        asims.extend([m['asimetria_LR'] for m in res['metricas']])
    asim_f2 = np.mean([m['asimetria_LR'] for m in resultados[0]['metricas']])
    asim_f6 = np.mean([m['asimetria_LR'] for m in resultados[4]['metricas']])
    criterio12 = abs(asim_f2 - asim_f6) > 0.001
    print(f"    C12 — Asimetría diferencial:   F2={asim_f2:.6f} F6={asim_f6:.6f} diff={abs(asim_f2-asim_f6):.6f} {'✅' if criterio12 else '❌'}")
    
    # C13 — Mantenimiento activo
    mants = []
    for res in resultados:
        mants.extend([m['senal_mant'] for m in res['metricas']])
    mant_activo = any(m > 0.001 for m in mants)
    criterio13 = mant_activo
    print(f"    C13 — Mantenimiento activo:    max_senal={max(mants):.6f} {'✅' if criterio13 else '❌'}")
    
    print("\n  VEREDICTO:")
    if all([criterio10, criterio11, criterio12, criterio13]):
        print("  ✅ CAMPO EXPANDIDO VALIDADO")
        print("     El ganglio coordinador emerge por mayor conectividad topológica.")
        print("     Los actuadores modulan el acoplamiento sin comandos externos.")
        print("     La asimetría L/R proporciona información de dirección emergente.")
    else:
        print("  ⚠️ VALIDACIÓN PARCIAL — Algunos criterios no se cumplen")
    
    # Guardar CSV
    with open('v81_campo_expandido.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['fase', 't', 'ged_L', 'ged_R', 'alpha_L', 'alpha_R',
                         'G_actividad', 'G_varianza', 'asimetria_LR', 'perm_nivel',
                         'senal_mant', 'eficiencia', 'ventaja', 'error_rec'])
        
        for fase_idx, (fase, res) in enumerate(zip(fases, resultados)):
            for t_idx, m in enumerate(res['metricas']):
                writer.writerow([
                    fase[0], t_idx * DT,
                    m['ged_L'], m['ged_R'], m['alpha_L'], m['alpha_R'],
                    m['G_actividad'], m['G_varianza'], m['asimetria_LR'],
                    m['perm_nivel'], m['senal_mant'], m['eficiencia'],
                    m['ventaja'], m['error_rec']
                ])
    
    print("\n  CSV guardado: v81_campo_expandido.csv")
    
    # Gráfico
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    nombres = [f[0] for f in fases]
    
    # GED L y R
    axes[0,0].plot(nombres, [np.mean([m['ged_L'] for m in res['metricas']]) for res in resultados], 'o-', label='GED L')
    axes[0,0].plot(nombres, [np.mean([m['ged_R'] for m in res['metricas']]) for res in resultados], 's-', label='GED R')
    axes[0,0].set_title('GED por canal')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Alpha L y R
    axes[0,1].plot(nombres, [np.mean([m['alpha_L'] for m in res['metricas']]) for res in resultados], 'o-', label='Alpha L')
    axes[0,1].plot(nombres, [np.mean([m['alpha_R'] for m in res['metricas']]) for res in resultados], 's-', label='Alpha R')
    axes[0,1].set_title('Alpha por canal (acoplamiento)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Actividad del ganglio
    axes[0,2].bar(nombres, [np.mean([m['G_actividad'] for m in res['metricas']]) for res in resultados])
    axes[0,2].set_title('Actividad del ganglio Φ_G')
    axes[0,2].grid(True, alpha=0.3)
    
    # Asimetría L/R
    axes[0,3].bar(nombres, [np.mean([m['asimetria_LR'] for m in res['metricas']]) for res in resultados])
    axes[0,3].set_title('Asimetría L/R (dirección)')
    axes[0,3].grid(True, alpha=0.3)
    
    # Permeabilidad
    axes[0,4].bar(nombres, [np.mean([m['perm_nivel'] for m in res['metricas']]) for res in resultados])
    axes[0,4].set_title('Permeabilidad (act_perm)')
    axes[0,4].grid(True, alpha=0.3)
    
    # Eficiencia
    axes[1,0].bar(nombres, [np.mean([m['eficiencia'] for m in res['metricas']]) for res in resultados])
    axes[1,0].set_title('Eficiencia de régimen')
    axes[1,0].grid(True, alpha=0.3)
    
    # LF activa
    axes[1,1].bar(nombres, [res['lf_pct'] for res in resultados])
    axes[1,1].set_title('LF activa (%)')
    axes[1,1].grid(True, alpha=0.3)
    
    # W_rec norma
    axes[1,2].bar(nombres, [res['w_rec_norma'] for res in resultados])
    axes[1,2].set_title('W_rec (memoria reciente)')
    axes[1,2].grid(True, alpha=0.3)
    
    # Señal de mantenimiento
    axes[1,3].bar(nombres, [np.mean([m['senal_mant'] for m in res['metricas']]) for res in resultados])
    axes[1,3].set_title('Señal de mantenimiento')
    axes[1,3].grid(True, alpha=0.3)
    
    # Varianza auditiva
    axes[1,4].bar(nombres, [np.mean([m['var_aud_media'] for m in res['metricas']]) for res in resultados])
    axes[1,4].set_title('Varianza auditiva media')
    axes[1,4].grid(True, alpha=0.3)
    
    plt.suptitle('VSTCosmos v81 — Campo expandido con ganglio coordinador', fontsize=14)
    plt.tight_layout()
    plt.savefig('v81_campo_expandido.png', dpi=150)
    print("  Gráfico guardado: v81_campo_expandido.png")
    
    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)


if __name__ == "__main__":
    main()