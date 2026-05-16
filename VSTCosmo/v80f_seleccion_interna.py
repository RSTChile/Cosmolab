#!/usr/bin/env python3
"""
VSTCosmos v80f — Selección interna por eficiencia de régimen

Un solo mecanismo nuevo: la función de valor interna del campo.
El olvido de W_rec es inversamente proporcional a la ventaja de eficiencia
del régimen actual respecto a su propia historia.

Eficiencia = GED / variación_temporal = ICR/IRDE computacional.
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import csv
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PARÁMETROS BASE (definen la física del campo)
# ============================================================
DIM_INTERNA = 32
DIM_AUDITIVA = 32
DIM_TOTAL = DIM_INTERNA + DIM_AUDITIVA

DIM_TIME = 100
DT = 0.01

DURACION_ENTRENO = 30.0
DURACION_FASE = 20.0
DURACION_REACOPLAMIENTO = 30.0

N_PASOS_ENTRENO = int(DURACION_ENTRENO / DT)
N_PASOS_FASE = int(DURACION_FASE / DT)
N_PASOS_REACOP = int(DURACION_REACOPLAMIENTO / DT)

# Parámetros de dinámica
DIFUSION_BASE = 0.15
GANANCIA_REACCION = 0.05
GAMMA_PLAST = 0.01

OMEGA_MIN = 0.05
OMEGA_MAX = 0.50
AMORT_MIN = 0.01
AMORT_MAX = 0.08
PHI_EQUILIBRIO = 0.5

# Límites
LIMITE_MIN = 0.0
LIMITE_MAX = 1.0
W_MAX = 1.0

# Parámetros de FFT
VENTANA_FFT_MS = 25
HOP_FFT_MS = 10
F_MIN = 80
F_MAX = 8000

# ============================================================
# PARÁMETROS DERIVADOS ESTRUCTURALMENTE
# ============================================================

# Períodos naturales del campo
T_PROFUNDA_SEG = 1.0 / OMEGA_MIN           # = 20.0 s
T_RECIENTE_SEG = 1.0 / OMEGA_MAX           # = 2.0 s

T_PROFUNDA_PASOS = int(T_PROFUNDA_SEG / DT)  # = 2000
T_RECIENTE_PASOS = int(T_RECIENTE_SEG / DT)  # = 200

# Tasas de aprendizaje base
ETA_PROFUNDA_BASE = (1.0 / T_PROFUNDA_PASOS) / DIFUSION_BASE   # ≈ 0.00333
ETA_RECIENTE_BASE = (1.0 / T_RECIENTE_PASOS) / DIFUSION_BASE   # ≈ 0.03333

# Tasas de olvido base
TAU_PROFUNDA = OMEGA_MIN                    # = 0.05
TAU_RECIENTE = OMEGA_MIN * 0.5              # = 0.025

# Error de equilibrio de difusión (referencia)
ERROR_DIFUSION = DIFUSION_BASE ** 2         # = 0.0225

# Escala temporal del historial de eficiencia
TAU_EFICIENCIA = int(1.0 / (OMEGA_MIN * DT))  # = 2000 pasos = 20s

# Fuerza del atractor
GAMMA_ATRACTOR = 0.05

# Separación espectral
BANDA_BAJA = slice(0, DIM_INTERNA // 2)    # modos 0-15: identidad
BANDA_ALTA = slice(DIM_INTERNA // 2, None)  # modos 16-31: contexto

print("=" * 100)
print("VSTCosmos v80f — Selección interna por eficiencia de régimen")
print("")
print("  Mecanismo nuevo: función de valor interna del campo")
print("  Eficiencia = GED / variación_temporal = ICR/IRDE computacional")
print("  Olvido de W_rec es inversamente proporcional a ventaja de eficiencia")
print("  Sin umbrales externos — solo comparación con historia propia")
print("")
print("  Parámetros derivados de la física del campo:")
print(f"    T_PROFUNDA = {T_PROFUNDA_SEG:.1f}s ({T_PROFUNDA_PASOS} pasos)")
print(f"    T_RECIENTE = {T_RECIENTE_SEG:.1f}s ({T_RECIENTE_PASOS} pasos)")
print(f"    ETA_PROFUNDA_BASE = {ETA_PROFUNDA_BASE:.6f}")
print(f"    ETA_RECIENTE_BASE = {ETA_RECIENTE_BASE:.6f}")
print(f"    TAU_PROFUNDA = {TAU_PROFUNDA:.4f}")
print(f"    TAU_RECIENTE = {TAU_RECIENTE:.4f} (base, modulada por eficiencia)")
print(f"    TAU_EFICIENCIA = {TAU_EFICIENCIA} pasos ({TAU_EFICIENCIA*DT:.1f}s)")
print(f"    ERROR_DIFUSION = {ERROR_DIFUSION:.4f} (difusión²)")
print("=" * 100)

# ============================================================
# FUNCIONES BASE
# ============================================================
def cargar_audio(ruta, duracion):
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

def inicializar_campo_total():
    np.random.seed(42)
    return np.random.rand(DIM_TOTAL, DIM_TIME) * 0.2 + 0.4

def inicializar_memoria_profunda():
    return np.zeros((DIM_INTERNA // 2, DIM_AUDITIVA // 2))

def inicializar_memoria_reciente():
    return np.zeros((DIM_INTERNA // 2, DIM_AUDITIVA // 2))

def inicializar_historia_interna():
    return np.zeros((DIM_INTERNA, DIM_TIME))

def vecinos(X):
    return (np.roll(X, 1, axis=0) + np.roll(X, -1, axis=0) +
            np.roll(X, 1, axis=1) + np.roll(X, -1, axis=1)) / 4.0

def preparar_objetivo_audio(audio, sr, idx_ventana, ventana_muestras, hop_muestras,
                            dim_auditiva, DIM_TIME):
    inicio = idx_ventana * hop_muestras
    if inicio + ventana_muestras > len(audio):
        return np.zeros((dim_auditiva, DIM_TIME))
    
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
    
    return objetivo.reshape(-1, 1) * np.ones((1, DIM_TIME))

def calcular_frecuencias_naturales(dim_total, dim_interna):
    bandas = np.arange(dim_total)
    t = np.log1p(bandas) / np.log1p(dim_total - 1)
    omega = OMEGA_MIN + (OMEGA_MAX - OMEGA_MIN) * t
    amort = AMORT_MIN + (AMORT_MAX - AMORT_MIN) * t
    return omega.reshape(-1, 1), amort.reshape(-1, 1)

def _perfil_espectral(region, dim, DIM_TIME):
    perfil = np.zeros(DIM_TIME // 2)
    for banda in range(dim):
        serie = region[banda, :]
        serie = serie - np.mean(serie)
        fft = np.fft.rfft(serie)
        potencia = np.abs(fft) ** 2
        perfil += potencia[:DIM_TIME // 2]
    return perfil / dim

def calcular_gradiente_espectral_diferencial(Phi_total, dim_interna, DIM_TIME):
    region_int = Phi_total[:dim_interna, :]
    region_aud = Phi_total[dim_interna:, :]
    
    perfil_int = _perfil_espectral(region_int, dim_interna, DIM_TIME)
    perfil_aud = _perfil_espectral(region_aud, dim_interna, DIM_TIME)
    
    ged = np.mean(np.abs(perfil_int - perfil_aud))
    return float(ged)

def calcular_balance_plastica_difusiva(Phi_total, W_rec, dim_interna):
    region_int = Phi_total[:dim_interna, :]
    region_aud = Phi_total[dim_interna:, :]
    reg_int_alta = region_int[BANDA_ALTA, :]
    reg_aud_alta = region_aud[BANDA_ALTA, :]
    
    prediccion = np.tanh(W_rec @ reg_aud_alta)
    M_rec = GAMMA_PLAST * (prediccion - reg_int_alta)
    potencia_plast = float(np.mean(M_rec ** 2))
    
    promedio_local_completo = vecinos(Phi_total)
    promedio_local_int = promedio_local_completo[:dim_interna, :]
    difusion = DIFUSION_BASE * (promedio_local_int[BANDA_ALTA, :] - reg_int_alta)
    potencia_difus = float(np.mean(difusion ** 2))
    
    balance = potencia_plast / (potencia_difus + 1e-10)
    return balance, potencia_plast, potencia_difus

# ============================================================
# NUEVO: EFICIENCIA DE RÉGIMEN (ICR/IRDE computacional)
# ============================================================
def calcular_eficiencia_regimen(Phi_total, dim_interna, ged_actual):
    """
    Eficiencia = diferenciación sostenida / costo dinámico.
    
    GED mide la diferenciación entre regiones del campo.
    La variación interna mide cuánto está cambiando el campo
    para mantener su estado — el costo disipativo.
    
    Alta eficiencia: mucha diferenciación con poca variación.
    Baja eficiencia: poca diferenciación o mucha variación.
    
    Equivalente computacional de ICR/IRDE.
    """
    region_int = Phi_total[:dim_interna, :]
    
    # Costo disipativo: variación temporal media del campo interno
    variacion = float(np.mean(np.abs(np.diff(region_int, axis=1))))
    
    eficiencia = ged_actual / (variacion + 1e-10)
    return eficiencia, variacion

def calcular_tasa_olvido_por_eficiencia(eficiencia_actual,
                                         historial_eficiencia):
    """
    El olvido de W_rec es inversamente proporcional a la ventaja
    de eficiencia del régimen actual respecto al historial.
    
    Retorna: (tasa_olvido, ventaja)
    """
    if len(historial_eficiencia) < TAU_EFICIENCIA:
        return TAU_RECIENTE, 1.0  # sin historia: tasa base, ventaja neutra
    
    eficiencia_media = float(np.mean(historial_eficiencia[-TAU_EFICIENCIA:]))
    
    # Ventaja relativa
    ventaja = eficiencia_actual / (eficiencia_media + 1e-10)
    
    # Tasa de olvido inversamente proporcional a la ventaja
    tasa_olvido = TAU_RECIENTE / ventaja
    
    # Límites derivados de la física del campo
    tasa_min = TAU_PROFUNDA
    tasa_max = ETA_RECIENTE_BASE * 10.0
    
    return float(np.clip(tasa_olvido, tasa_min, tasa_max)), float(ventaja)

# ============================================================
# NÚCLEO: MEMORIAS DUALES CON SELECCIÓN POR EFICIENCIA
# ============================================================
def aplicar_plasticidad_memoria_profunda(W_prof, region_int, region_aud, dt):
    """Identidad en banda baja — sin cambios"""
    reg_int_baja = region_int[BANDA_BAJA, :]
    reg_aud_baja = region_aud[BANDA_BAJA, :]
    
    correlacion = (reg_int_baja @ reg_aud_baja.T) / DIM_TIME
    dW = ETA_PROFUNDA_BASE * correlacion - TAU_PROFUNDA * W_prof
    W_nueva = np.clip(W_prof + dW * dt, -W_MAX, W_MAX)
    
    prediccion = W_nueva @ reg_aud_baja
    M_hebb = np.zeros((DIM_INTERNA, DIM_TIME))
    M_hebb[BANDA_BAJA, :] = prediccion - reg_int_baja
    
    return W_nueva, M_hebb

def aplicar_plasticidad_memoria_reciente_v80f(W_rec, region_int, region_aud,
                                               dt, tasa_olvido, error_equilibrio):
    """
    Plasticidad de W_rec con olvido por eficiencia.
    Aprendizaje proporcional a coherencia (error bajo).
    Olvido determinado por eficiencia del régimen.
    """
    reg_int_alta = region_int[BANDA_ALTA, :]
    reg_aud_alta = region_aud[BANDA_ALTA, :]
    
    # Error predictivo actual de W_rec
    prediccion = np.tanh(W_rec @ reg_aud_alta)
    error_rec = float(np.mean((prediccion - reg_int_alta) ** 2))
    
    # Aprendizaje proporcional a coherencia (como en v80e)
    coherencia = error_equilibrio / (error_rec + error_equilibrio)
    tasa_aprendizaje = ETA_RECIENTE_BASE * coherencia
    
    # Olvido por eficiencia (el cambio clave)
    correlacion = (reg_int_alta @ reg_aud_alta.T) / DIM_TIME
    dW = tasa_aprendizaje * correlacion - tasa_olvido * W_rec
    W_nueva = np.clip(W_rec + dW * dt, -W_MAX, W_MAX)
    
    M_reciente = np.zeros((DIM_INTERNA, DIM_TIME))
    M_reciente[BANDA_ALTA, :] = GAMMA_PLAST * (prediccion - reg_int_alta)
    
    return W_nueva, M_reciente, error_rec, coherencia, tasa_aprendizaje

def atractor_dual(Phi_int_historia, region_int):
    return GAMMA_ATRACTOR * (Phi_int_historia - region_int)

# ============================================================
# ACTUALIZACIÓN PRINCIPAL
# ============================================================
def actualizar_campo_con_seleccion_interna(
        Phi_total, Phi_vel_total, W_prof, W_rec,
        Phi_int_historia,
        objetivo_audio, alpha,
        omega_natural, amort_natural,
        dt, entrenando, error_equilibrio_w_rec,
        historial_eficiencia,
        DIM_TIME):
    
    # Dinámica base
    promedio_local = vecinos(Phi_total)
    difusion = DIFUSION_BASE * (promedio_local - Phi_total)
    desviacion = Phi_total - promedio_local
    reaccion = GANANCIA_REACCION * desviacion * (1 - desviacion**2)
    term_osc = (-omega_natural**2 * (Phi_total - PHI_EQUILIBRIO)
                - amort_natural * Phi_vel_total)
    
    region_int = Phi_total[:DIM_INTERNA, :]
    region_aud = Phi_total[DIM_INTERNA:, :]
    
    # Plasticidad profunda
    W_prof_nueva, M_prof = aplicar_plasticidad_memoria_profunda(
        W_prof, region_int, region_aud, dt
    )
    
    # Calcular eficiencia actual para determinar olvido
    ged = calcular_gradiente_espectral_diferencial(Phi_total, DIM_INTERNA, DIM_TIME)
    eficiencia_actual, variacion = calcular_eficiencia_regimen(
        Phi_total, DIM_INTERNA, ged
    )
    
    # Calcular tasa de olvido por eficiencia
    tasa_olvido, ventaja = calcular_tasa_olvido_por_eficiencia(
        eficiencia_actual, historial_eficiencia if not entrenando else []
    )
    
    # Plasticidad reciente con olvido modulado
    W_rec_nueva, M_rec, error_rec, coherencia, tasa_aprendizaje = \
        aplicar_plasticidad_memoria_reciente_v80f(
            W_rec, region_int, region_aud, dt, tasa_olvido, error_equilibrio_w_rec
        )
    
    # Atractor
    M_atractor = atractor_dual(Phi_int_historia, region_int)
    
    # Campo total
    M_campo = np.zeros_like(Phi_total)
    M_campo[:DIM_INTERNA, :] = M_prof + M_rec + M_atractor
    
    # LF definida por error > equilibrio
    lf_activa = error_rec > error_equilibrio_w_rec
    
    # Actualizar historial de eficiencia (solo en test, no durante entrenamiento)
    if not entrenando:
        historial_eficiencia.append(eficiencia_actual)
        if len(historial_eficiencia) > TAU_EFICIENCIA * 2:
            historial_eficiencia.pop(0)
    
    # Actualización
    dPhi_vel = term_osc + reaccion + difusion + M_campo
    Phi_vel_nueva = Phi_vel_total + dt * dPhi_vel
    Phi_nueva = Phi_total + dt * Phi_vel_nueva
    
    if alpha > 0:
        region_auditiva_nueva = Phi_nueva[DIM_INTERNA:, :]
        region_auditiva_nueva = (1 - alpha) * region_auditiva_nueva + alpha * objetivo_audio
        Phi_nueva[DIM_INTERNA:, :] = region_auditiva_nueva
    
    # Prevenir colapso
    varianza_campo = np.var(Phi_nueva[:DIM_INTERNA, :])
    if varianza_campo < 1e-6:
        ruido_minimo = np.random.normal(0, DIFUSION_BASE * 0.1, Phi_nueva[:DIM_INTERNA, :].shape)
        Phi_nueva[:DIM_INTERNA, :] += ruido_minimo
    
    # Actualizar historia interna
    Phi_int_historia_nueva = (1 - 0.05) * Phi_int_historia + 0.05 * region_int
    
    return (np.clip(Phi_nueva, LIMITE_MIN, LIMITE_MAX),
            np.clip(Phi_vel_nueva, -5.0, 5.0),
            W_prof_nueva, W_rec_nueva, Phi_int_historia_nueva,
            lf_activa, error_rec, eficiencia_actual, variacion,
            tasa_olvido, tasa_aprendizaje, ventaja, coherencia)

# ============================================================
# ENTRENAMIENTO CON MEDICIÓN DEL ERROR DE EQUILIBRIO
# ============================================================
def entrenar_y_medir_error_equilibrio():
    """Entrena el campo y mide ERROR_EQUILIBRIO_W_REC como error mínimo alcanzado"""
    Phi_total = inicializar_campo_total()
    Phi_vel_total = np.zeros_like(Phi_total)
    W_prof = inicializar_memoria_profunda()
    W_rec = inicializar_memoria_reciente()
    Phi_int_historia = inicializar_historia_interna()
    
    omega_natural, amort_natural = calcular_frecuencias_naturales(DIM_TOTAL, DIM_INTERNA)
    
    sr_voz, audio_voz = cargar_audio("Voz_Estudio.wav", duracion=DURACION_ENTRENO)
    ventana_muestras = int(sr_voz * VENTANA_FFT_MS / 1000)
    hop_muestras = int(sr_voz * HOP_FFT_MS / 1000)
    
    errores = []
    historial_eficiencia_dummy = []
    
    for idx in range(N_PASOS_ENTRENO):
        objetivo = preparar_objetivo_audio(audio_voz, sr_voz, idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
        
        (Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia,
         _, error_rec, _, _, _, _, _, _) = actualizar_campo_con_seleccion_interna(
            Phi_total, Phi_vel_total, W_prof, W_rec,
            Phi_int_historia,
            objetivo, alpha=0.05,
            omega_natural=omega_natural, amort_natural=amort_natural,
            dt=DT, entrenando=True, error_equilibrio_w_rec=ERROR_DIFUSION,
            historial_eficiencia=historial_eficiencia_dummy,
            DIM_TIME=DIM_TIME
        )
        
        errores.append(error_rec)
    
    # Usar el error mínimo alcanzado como equilibrio
    error_equilibrio = np.min(errores[-2000:]) if len(errores) > 2000 else np.min(errores)
    print(f"  ERROR_EQUILIBRIO_W_REC medido: {error_equilibrio:.6f} (difusión: {ERROR_DIFUSION:.4f})")
    
    return (Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia,
            omega_natural, amort_natural, error_equilibrio)

# ============================================================
# SIMULACIÓN DE FASE
# ============================================================
def simular_fase(Phi_total, Phi_vel_total, W_prof, W_rec,
                 Phi_int_historia,
                 estimulo, alpha, duracion, fase_nombre,
                 omega_natural, amort_natural, error_equilibrio_w_rec,
                 DIM_TIME, historial_eficiencia_global):
    sr, audio = cargar_audio(estimulo, duracion=duracion)
    ventana_muestras = int(sr * VENTANA_FFT_MS / 1000)
    hop_muestras = int(sr * HOP_FFT_MS / 1000)
    n_pasos = int(duracion / DT)
    
    ged_hist = []
    lf_hist = []
    error_rec_hist = []
    w_prof_norma_hist = []
    w_rec_norma_hist = []
    eficiencia_hist = []
    variacion_hist = []
    ventaja_hist = []
    tasa_olvido_hist = []
    tasa_aprendizaje_hist = []
    coherencia_hist = []
    
    for idx in range(n_pasos):
        objetivo = preparar_objetivo_audio(audio, sr, idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
        
        ged = calcular_gradiente_espectral_diferencial(Phi_total, DIM_INTERNA, DIM_TIME)
        
        (Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia,
         lf_activa, error_rec, eficiencia, variacion,
         tasa_olvido, tasa_aprendizaje, ventaja, coherencia) = actualizar_campo_con_seleccion_interna(
            Phi_total, Phi_vel_total, W_prof, W_rec,
            Phi_int_historia,
            objetivo, alpha,
            omega_natural, amort_natural,
            DT, entrenando=False, error_equilibrio_w_rec=error_equilibrio_w_rec,
            historial_eficiencia=historial_eficiencia_global,
            DIM_TIME=DIM_TIME
        )
        
        ged_hist.append(ged)
        lf_hist.append(1 if lf_activa else 0)
        error_rec_hist.append(error_rec)
        w_prof_norma_hist.append(np.mean(np.abs(W_prof)))
        w_rec_norma_hist.append(np.mean(np.abs(W_rec)))
        eficiencia_hist.append(eficiencia)
        variacion_hist.append(variacion)
        ventaja_hist.append(ventaja)
        tasa_olvido_hist.append(tasa_olvido)
        tasa_aprendizaje_hist.append(tasa_aprendizaje)
        coherencia_hist.append(coherencia)
    
    return {
        'ged_mean': np.mean(ged_hist),
        'lf_pct': 100 * np.mean(lf_hist),
        'w_prof_norma': np.mean(w_prof_norma_hist),
        'w_rec_norma': np.mean(w_rec_norma_hist),
        'w_rec_inicio': w_rec_norma_hist[0] if w_rec_norma_hist else 0,
        'w_rec_fin': w_rec_norma_hist[-1] if w_rec_norma_hist else 0,
        'error_rec_mean': np.mean(error_rec_hist),
        'eficiencia_mean': np.mean(eficiencia_hist),
        'variacion_mean': np.mean(variacion_hist),
        'ventaja_mean': np.mean(ventaja_hist),
        'tasa_olvido_mean': np.mean(tasa_olvido_hist),
        'tasa_aprendizaje_mean': np.mean(tasa_aprendizaje_hist),
        'coherencia_mean': np.mean(coherencia_hist),
        'phi_total': Phi_total.copy(),
        'w_prof': W_prof.copy(),
        'w_rec': W_rec.copy(),
        'phi_int_historia': Phi_int_historia.copy()
    }

# ============================================================
# MAIN
# ============================================================
def main():
    # Entrenamiento y medición
    print("\n[Fase 1] Entrenamiento (voz, alpha=0.05, 30s)")
    (Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia,
     omega_natural, amort_natural, ERROR_EQUILIBRIO_W_REC) = entrenar_y_medir_error_equilibrio()
    
    print(f"  W_prof tras entreno: {np.mean(np.abs(W_prof)):.4f}")
    print(f"  W_rec tras entreno: {np.mean(np.abs(W_rec)):.4f}")
    
    # Historial global de eficiencia (compartido entre fases)
    historial_eficiencia_global = []
    
    # Fases de test
    fases = [
        ("Fase 2", "Voz_Estudio.wav", "Dominio (voz)", DURACION_FASE),
        ("Fase 3", "Brandemburgo.wav", "No entrenado (música)", DURACION_FASE),
        ("Fase 4", "Tono puro", "No entrenado (tono)", DURACION_FASE),
        ("Fase 5", "Voz+Viento_1.wav", "Degradado (voz+viento)", DURACION_FASE),
        ("Fase 6", "Ruido blanco", "Perturbación basal", DURACION_FASE),
        ("Fase 7", "Voz_Estudio.wav", "Re-acoplamiento (voz)", DURACION_REACOPLAMIENTO)
    ]
    
    resultados = []
    
    for fase_id, estimulo, desc, duracion in fases:
        print(f"\n[{fase_id}] {desc}")
        res = simular_fase(
            Phi_total, Phi_vel_total, W_prof, W_rec,
            Phi_int_historia,
            estimulo, 0.0, duracion, fase_id,
            omega_natural, amort_natural, ERROR_EQUILIBRIO_W_REC,
            DIM_TIME, historial_eficiencia_global
        )
        resultados.append(res)
        
        Phi_total = res['phi_total']
        W_prof = res['w_prof']
        W_rec = res['w_rec']
        Phi_int_historia = res['phi_int_historia']
        
        balance, _, _ = calcular_balance_plastica_difusiva(Phi_total, W_rec, DIM_INTERNA)
        balance_status = ""
        if 0.1 < balance < 10:
            balance_status = "✅ equilibrado"
        elif balance <= 0.1:
            balance_status = "⚠️ plástica débil"
        else:
            balance_status = "❌ plástica dominante"
        
        print(f"    GED: {res['ged_mean']:.6f} | LF: {res['lf_pct']:.1f}%")
        print(f"    W_prof: {res['w_prof_norma']:.4f} | W_rec: {res['w_rec_norma']:.4f}")
        print(f"    Error: {res['error_rec_mean']:.6f} (eq: {ERROR_EQUILIBRIO_W_REC:.6f})")
        print(f"    Eficiencia: {res['eficiencia_mean']:.4f} | Variación: {res['variacion_mean']:.6f}")
        print(f"    Ventaja: {res['ventaja_mean']:.4f} | Olvido: {res['tasa_olvido_mean']:.6f}")
        print(f"    Balance: {balance:.3f} {balance_status}")
    
    # Diagnóstico final
    print("\n" + "=" * 100)
    print("DIAGNÓSTICO — v80f Selección interna por eficiencia")
    print("=" * 100)
    
    print(f"  ERROR_EQUILIBRIO_W_REC medido: {ERROR_EQUILIBRIO_W_REC:.6f}")
    
    # Criterios 1-7 (de v80e)
    error_f2 = resultados[0]['error_rec_mean']
    error_f6 = resultados[4]['error_rec_mean']
    criterio1 = error_f2 < 1.0
    criterio2 = error_f2 < error_f6
    
    balance_f7, _, _ = calcular_balance_plastica_difusiva(
        resultados[5]['phi_total'], resultados[5]['w_rec'], DIM_INTERNA
    )
    criterio3 = 0.1 < balance_f7 < 10
    
    w_prof_f2 = resultados[0]['w_prof_norma']
    w_prof_f7 = resultados[5]['w_prof_norma']
    criterio4 = w_prof_f7 > w_prof_f2 * 1.5
    
    w_rec_f6 = resultados[4]['w_rec_fin']
    w_rec_f7 = resultados[5]['w_rec_fin']
    criterio5 = w_rec_f7 < w_rec_f6
    
    coherencia_f6 = resultados[4]['coherencia_mean']
    coherencia_f7 = resultados[5]['coherencia_mean']
    criterio6 = coherencia_f7 > coherencia_f6
    
    error_f7 = resultados[5]['error_rec_mean']
    criterio7 = error_f7 < ERROR_EQUILIBRIO_W_REC
    
    # CRITERIOS NUEVOS (8 y 9)
    eficiencia_f2 = resultados[0]['eficiencia_mean']
    eficiencia_f6 = resultados[4]['eficiencia_mean']
    criterio8 = eficiencia_f2 > eficiencia_f6
    
    olvido_f6 = resultados[4]['tasa_olvido_mean']
    olvido_f7 = resultados[5]['tasa_olvido_mean']
    criterio9 = olvido_f7 < olvido_f6
    
    lf_f7 = resultados[5]['lf_pct']
    
    print(f"\n  CRITERIOS DE ESTABILIDAD (v80e):")
    print(f"    C1 — Error F2 < 1.0:                {error_f2:.6f} {'✅' if criterio1 else '❌'}")
    print(f"    C2 — Error F2 < Error F6:            {error_f2:.6f} < {error_f6:.6f} {'✅' if criterio2 else '❌'}")
    print(f"    C3 — Balance 0.1-10 en F7:           {balance_f7:.3f} {'✅' if criterio3 else '❌'}")
    print(f"    C4 — W_prof creció:                  {w_prof_f2:.4f} → {w_prof_f7:.4f} {'✅' if criterio4 else '❌'}")
    print(f"    C5 — W_rec olvida en F7:             {w_rec_f6:.4f} → {w_rec_f7:.4f} {'✅' if criterio5 else '❌'}")
    print(f"    C6 — Coherencia F7 > F6:             {coherencia_f6:.4f} → {coherencia_f7:.4f} {'✅' if criterio6 else '❌'}")
    print(f"    C7 — Error F7 < Equilibrio:          {error_f7:.6f} < {ERROR_EQUILIBRIO_W_REC:.6f} {'✅' if criterio7 else '❌'}")
    
    print(f"\n  CRITERIOS DE SELECCIÓN INTERNA:")
    print(f"    C8 — Eficiencia F2 > F6:             {eficiencia_f2:.4f} > {eficiencia_f6:.4f} {'✅' if criterio8 else '❌'}")
    print(f"    C9 — Olvido F7 < Olvido F6:          {olvido_f7:.6f} < {olvido_f6:.6f} {'✅' if criterio9 else '❌'}")
    
    print(f"\n  LF activa en Fase 7:                 {lf_f7:.1f}%")
    
    print("\n  VEREDICTO:")
    if criterio8 and criterio9:
        print("  ✅ SELECCIÓN INTERNA FUNCIONAL")
        print("     El campo prefiere el régimen de voz sobre el de ruido")
        print("     porque es más eficiente (mayor ICR/IRDE).")
        print("     El olvido se modula correctamente por eficiencia.")
        if all([criterio1, criterio2, criterio3, criterio4, criterio5, criterio6, criterio7]):
            print("     ✅ También cumple todos los criterios de estabilidad.")
            print("     v80f es la primera versión con ciclo completo validado.")
        else:
            print("     ⚠️ Estabilidad parcial — ajuste fino aún necesario.")
    else:
        print("  ❌ SELECCIÓN INTERNA NO FUNCIONAL")
        print("     La eficiencia no distingue voz de ruido o")
        print("     el olvido no se modula como se esperaba.")
    
    # Guardar CSV
    with open('v80f_seleccion_interna.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['fase', 'ged_mean', 'lf_pct', 'w_prof_norma', 'w_rec_norma',
                         'error_rec', 'eficiencia', 'variacion', 'ventaja',
                         'tasa_olvido', 'tasa_aprendizaje', 'coherencia'])
        for i, (fase, res) in enumerate(zip(fases, resultados)):
            writer.writerow([fase[0], res['ged_mean'], res['lf_pct'],
                            res['w_prof_norma'], res['w_rec_norma'],
                            res['error_rec_mean'], res['eficiencia_mean'],
                            res['variacion_mean'], res['ventaja_mean'],
                            res['tasa_olvido_mean'], res['tasa_aprendizaje_mean'],
                            res['coherencia_mean']])
    
    print("\n  CSV guardado: v80f_seleccion_interna.csv")
    
    # Gráfico
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    nombres = ['Voz', 'Música', 'Tono', 'Voz+Viento', 'Ruido', 'Reacop']
    
    # GED
    axes[0,0].bar(nombres, [r['ged_mean'] for r in resultados])
    axes[0,0].set_title('GED')
    axes[0,0].grid(True, alpha=0.3)
    
    # LF activa
    axes[0,1].bar(nombres, [r['lf_pct'] for r in resultados])
    axes[0,1].set_title('LF activa (%)')
    axes[0,1].grid(True, alpha=0.3)
    
    # W_rec
    axes[0,2].bar(nombres, [r['w_rec_norma'] for r in resultados])
    axes[0,2].set_title('W_rec (memoria de contexto)')
    axes[0,2].grid(True, alpha=0.3)
    
    # Eficiencia
    axes[0,3].bar(nombres, [r['eficiencia_mean'] for r in resultados])
    axes[0,3].set_title('Eficiencia (ICR/IRDE)')
    axes[0,3].grid(True, alpha=0.3)
    
    # Variación
    axes[1,0].bar(nombres, [r['variacion_mean'] for r in resultados])
    axes[1,0].set_title('Variación (costo disipativo)')
    axes[1,0].grid(True, alpha=0.3)
    
    # Tasa olvido
    axes[1,1].bar(nombres, [r['tasa_olvido_mean'] for r in resultados])
    axes[1,1].set_title('Tasa de olvido de W_rec')
    axes[1,1].grid(True, alpha=0.3)
    
    # Ventaja
    axes[1,2].bar(nombres, [r['ventaja_mean'] for r in resultados])
    axes[1,2].axhline(y=1.0, color='r', linestyle='--', label='Ventaja = 1')
    axes[1,2].set_title('Ventaja de eficiencia')
    axes[1,2].legend()
    axes[1,2].grid(True, alpha=0.3)
    
    # Coherencia
    axes[1,3].bar(nombres, [r['coherencia_mean'] for r in resultados])
    axes[1,3].set_title('Coherencia de W_rec')
    axes[1,3].grid(True, alpha=0.3)
    
    plt.suptitle('VSTCosmos v80f — Selección interna por eficiencia de régimen', fontsize=14)
    plt.tight_layout()
    plt.savefig('v80f_seleccion_interna.png', dpi=150)
    print("  Gráfico guardado: v80f_seleccion_interna.png")
    
    # Guardar TXT
    with open('v80f_seleccion_interna.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("VSTCosmos v80f — Diagnóstico\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"ERROR_EQUILIBRIO_W_REC = {ERROR_EQUILIBRIO_W_REC:.6f}\n\n")
        
        f.write("CRITERIOS DE ESTABILIDAD:\n")
        f.write(f"  C1 — Error F2 < 1.0:                {error_f2:.6f} {'✅' if criterio1 else '❌'}\n")
        f.write(f"  C2 — Error F2 < Error F6:            {error_f2:.6f} < {error_f6:.6f} {'✅' if criterio2 else '❌'}\n")
        f.write(f"  C3 — Balance 0.1-10 en F7:           {balance_f7:.3f} {'✅' if criterio3 else '❌'}\n")
        f.write(f"  C4 — W_prof creció:                  {w_prof_f2:.4f} → {w_prof_f7:.4f} {'✅' if criterio4 else '❌'}\n")
        f.write(f"  C5 — W_rec olvida en F7:             {w_rec_f6:.4f} → {w_rec_f7:.4f} {'✅' if criterio5 else '❌'}\n")
        f.write(f"  C6 — Coherencia F7 > F6:             {coherencia_f6:.4f} → {coherencia_f7:.4f} {'✅' if criterio6 else '❌'}\n")
        f.write(f"  C7 — Error F7 < Equilibrio:          {error_f7:.6f} < {ERROR_EQUILIBRIO_W_REC:.6f} {'✅' if criterio7 else '❌'}\n\n")
        
        f.write("CRITERIOS DE SELECCIÓN INTERNA:\n")
        f.write(f"  C8 — Eficiencia F2 > F6:             {eficiencia_f2:.4f} > {eficiencia_f6:.4f} {'✅' if criterio8 else '❌'}\n")
        f.write(f"  C9 — Olvido F7 < Olvido F6:          {olvido_f7:.6f} < {olvido_f6:.6f} {'✅' if criterio9 else '❌'}\n\n")
        
        if criterio8 and criterio9:
            f.write("VEREDICTO: ✅ SELECCIÓN INTERNA FUNCIONAL\n")
            f.write("El campo prefiere regímenes más eficientes.\n")
            f.write("La función de valor emerge de la dinámica del campo.\n")
        else:
            f.write("VEREDICTO: ❌ SELECCIÓN INTERNA NO FUNCIONAL\n")
        f.write("=" * 60 + "\n")
    
    print("  TXT guardado: v80f_seleccion_interna.txt")
    
    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)


if __name__ == "__main__":
    main()