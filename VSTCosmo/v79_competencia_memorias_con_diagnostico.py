#!/usr/bin/env python3
"""
VSTCosmos v79 — Competencia de memorias con diagnóstico de estímulo nuevo

Principio canónico: ningún umbral fijado externamente.
El campo decide qué régimen sostener por coherencia predictiva.

Al final, prueba diagnóstica con estímulo nuevo (barrido de frecuencias)
para determinar si la integración amplió funcionalmente el dominio.
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import csv
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PARÁMETROS BASE DE LA FÍSICA DEL CAMPO
# ============================================================
DIM_INTERNA = 32
DIM_AUDITIVA = 32
DIM_TOTAL = DIM_INTERNA + DIM_AUDITIVA

DIM_TIME = 100
DT = 0.01

DURACION_ENTRENO = 30.0
DURACION_FASE = 20.0
DURACION_REACOPLAMIENTO = 30.0
DURACION_DIAGNOSTICO = 10.0

N_PASOS_ENTRENO = int(DURACION_ENTRENO / DT)
N_PASOS_FASE = int(DURACION_FASE / DT)
N_PASOS_REACOP = int(DURACION_REACOPLAMIENTO / DT)
N_PASOS_DIAGNOSTICO = int(DURACION_DIAGNOSTICO / DT)

# Parámetros de dinámica
DIFUSION_BASE = 0.15
GANANCIA_REACCION = 0.05

OMEGA_MIN = 0.05
OMEGA_MAX = 0.50
AMORT_MIN = 0.01
AMORT_MAX = 0.08
PHI_EQUILIBRIO = 0.5

# Plasticidad hebbiana de W
ETA_HEBB = 0.05
TAU_W = 0.008
GAMMA_PLAST = 0.01
UMBRAL_CORRELACION = 0.1

# Plasticidad de W_exploracion
ETA_APRENDIZAJE_EXPL = 0.03
TAU_W_EXP = 0.01

# Fuerza del atractor competitivo
GAMMA_ATRACTOR = 0.05

# Fuerzas de los mecanismos LF
ETA_EXPLORACION = 0.02
GAMMA_GENERACION = 0.04

# Límites duros del campo
LIMITE_MIN = 0.0
LIMITE_MAX = 1.0

# Parámetros de FFT
VENTANA_FFT_MS = 25
HOP_FFT_MS = 10
F_MIN = 80
F_MAX = 8000

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

def generar_estimulo_nuevo(duracion=10.0):
    """
    Estímulo sintético que NO está en el conjunto de entrenamiento:
    - Barrido logarítmico de frecuencias (200Hz → 2000Hz)
    - Modulación de amplitud asimétrica (ritmo no estacionario)
    """
    sr = 48000
    t = np.arange(int(sr * duracion)) / sr
    # Barrido logarítmico de frecuencias
    freqs = np.exp(np.linspace(np.log(200), np.log(2000), len(t)))
    fase = 2 * np.pi * np.cumsum(freqs) / sr
    senal = 0.5 * np.sin(fase)
    # Modulación de amplitud asimétrica
    modulacion = (1 + 0.3 * np.sin(2 * np.pi * 3.7 * t)) * (1 + 0.2 * np.sin(2 * np.pi * 1.3 * t))
    senal = senal * modulacion
    return sr, senal.astype(np.float32)

def inicializar_campo_total():
    np.random.seed(42)
    return np.random.rand(DIM_TOTAL, DIM_TIME) * 0.2 + 0.4

def inicializar_plasticidad():
    return np.zeros((DIM_INTERNA, DIM_AUDITIVA))

def inicializar_w_exploracion():
    return np.zeros((DIM_INTERNA, DIM_AUDITIVA))

def inicializar_historia_interna():
    return np.zeros((DIM_INTERNA, DIM_TIME))

def inicializar_phi_generado():
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

def actualizar_hebb_y_plasticidad(Phi_total, W, dt):
    region_int = Phi_total[:DIM_INTERNA, :]
    region_aud = Phi_total[DIM_INTERNA:, :]

    correlacion = (region_int @ region_aud.T) / DIM_TIME
    correlacion_filtrada = np.where(
        np.abs(correlacion) > UMBRAL_CORRELACION,
        correlacion,
        0.0
    )
    dW = ETA_HEBB * correlacion_filtrada - TAU_W * W
    W_nueva = np.clip(W + dW * dt, -1.0, 1.0)

    M_hebb = GAMMA_PLAST * (W_nueva @ region_aud - region_int)

    return W_nueva, M_hebb

def actualizar_plasticidad_exploratoria(Phi_total, W_exploracion, dt):
    region_int = Phi_total[:DIM_INTERNA, :]
    region_aud = Phi_total[DIM_INTERNA:, :]

    correlacion = (region_int @ region_aud.T) / DIM_TIME
    umbral_expl = UMBRAL_CORRELACION * 0.5
    correlacion_filtrada = np.where(
        np.abs(correlacion) > umbral_expl,
        correlacion,
        0.0
    )
    dW_exp = ETA_APRENDIZAJE_EXPL * correlacion_filtrada - TAU_W_EXP * W_exploracion
    W_nueva = np.clip(W_exploracion + dW_exp * dt, -1.0, 1.0)

    return W_nueva

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

# ============================================================
# NÚCLEO: COMPETENCIA DE MEMORIAS
# ============================================================
def calcular_pesos_competitivos(W, W_exploracion, region_int, region_aud):
    prediccion_W = W @ region_aud
    prediccion_W_exp = W_exploracion @ region_aud
    
    error_W = np.mean(np.abs(prediccion_W - region_aud))
    error_W_exp = np.mean(np.abs(prediccion_W_exp - region_aud))
    
    peso_W = 1.0 / (error_W + 1e-10)
    peso_W_exp = 1.0 / (error_W_exp + 1e-10)
    
    lf_activa = peso_W_exp > peso_W
    
    return peso_W, peso_W_exp, lf_activa

def atractor_competitivo(Phi_total, W, W_exploracion, Phi_int_historia, dim_interna):
    region_int = Phi_total[:dim_interna, :]
    region_aud = Phi_total[dim_interna:, :]
    
    peso_W, peso_W_exp, lf_activa = calcular_pesos_competitivos(
        W, W_exploracion, region_int, region_aud
    )
    peso_total = peso_W + peso_W_exp
    
    atractor_W = GAMMA_ATRACTOR * (Phi_int_historia - region_int)
    
    prediccion_W_exp = W_exploracion @ region_aud
    atractor_W_exp = GAMMA_PLAST * (prediccion_W_exp - region_int)
    
    if peso_total > 0:
        M_atractor = ((peso_W / peso_total) * atractor_W +
                      (peso_W_exp / peso_total) * atractor_W_exp)
    else:
        M_atractor = atractor_W * 0.5 + atractor_W_exp * 0.5
    
    fraccion_W = peso_W / (peso_total + 1e-10)
    
    return M_atractor, lf_activa, fraccion_W

# ============================================================
# MECANISMOS LF
# ============================================================
def lf_exploracion(Phi_total, dim_interna):
    region_int = Phi_total[:dim_interna, :]
    region_aud = Phi_total[dim_interna:, :]
    return ETA_EXPLORACION * (region_aud - region_int)

def lf_generacion(Phi_total, Phi_generado, dim_interna, dt, DIM_TIME):
    region_int = Phi_total[:dim_interna, :]
    Phi_generado_nueva = 0.99 * Phi_generado + 0.01 * region_int
    
    perfil = _perfil_espectral(region_int, dim_interna, DIM_TIME)
    modo_dominante = np.argmax(perfil) if len(perfil) > 0 else 0
    mascara_modos = np.zeros(DIM_TIME)
    if modo_dominante < DIM_TIME:
        mascara_modos[modo_dominante] = 1.0
    
    signal_amplificada = np.zeros_like(region_int)
    for banda in range(dim_interna):
        fft_banda = np.fft.rfft(region_int[banda, :])
        fft_banda[:DIM_TIME//2] *= (1.0 + GAMMA_GENERACION *
                                     mascara_modos[:DIM_TIME//2])
        signal_amplificada[banda, :] = np.real(np.fft.irfft(
            fft_banda, n=DIM_TIME
        ))
    
    M_generacion = GAMMA_GENERACION * (signal_amplificada - region_int)
    return Phi_generado_nueva, M_generacion

# ============================================================
# ACTUALIZACIÓN PRINCIPAL
# ============================================================
def actualizar_campo_con_competencia(
        Phi_total, Phi_vel_total, W, W_exploracion,
        Phi_int_historia, Phi_generado,
        objetivo_audio, alpha,
        omega_natural, amort_natural,
        dt, entrenando,
        DIM_TIME):
    
    # Dinámica base
    promedio_local = vecinos(Phi_total)
    difusion = DIFUSION_BASE * (promedio_local - Phi_total)
    desviacion = Phi_total - promedio_local
    reaccion = GANANCIA_REACCION * desviacion * (1 - desviacion**2)
    term_osc = (-omega_natural**2 * (Phi_total - PHI_EQUILIBRIO)
                - amort_natural * Phi_vel_total)
    
    # Plasticidad de W
    W_nueva, M_hebb = actualizar_hebb_y_plasticidad(Phi_total, W, dt)
    
    # Plasticidad de W_exploracion
    W_exp_nueva = actualizar_plasticidad_exploratoria(Phi_total, W_exploracion, dt)
    
    # Atractor competitivo
    M_atractor, lf_activa, fraccion_W = atractor_competitivo(
        Phi_total, W_nueva, W_exp_nueva, Phi_int_historia, DIM_INTERNA
    )
    
    # Mecanismos LF
    M_campo = np.zeros_like(Phi_total)
    M_campo[:DIM_INTERNA, :] = M_hebb + M_atractor
    
    Phi_generado_nueva = Phi_generado.copy()
    
    if lf_activa and not entrenando:
        M_exp = lf_exploracion(Phi_total, DIM_INTERNA)
        Phi_generado_nueva, M_gen = lf_generacion(Phi_total, Phi_generado, DIM_INTERNA, dt, DIM_TIME)
        M_campo[:DIM_INTERNA, :] += M_exp + M_gen
    
    # Actualizar historia interna
    region_int = Phi_total[:DIM_INTERNA, :]
    Phi_int_historia_nueva = (1 - TAU_W) * Phi_int_historia + TAU_W * region_int
    
    # Actualización del campo
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
    
    return (np.clip(Phi_nueva, LIMITE_MIN, LIMITE_MAX),
            np.clip(Phi_vel_nueva, -5.0, 5.0),
            W_nueva, W_exp_nueva, Phi_int_historia_nueva,
            Phi_generado_nueva, lf_activa, fraccion_W)

# ============================================================
# SIMULACIÓN DE FASE PRINCIPAL
# ============================================================
def simular_fase(Phi_total, Phi_vel_total, W, W_exploracion,
                 Phi_int_historia, Phi_generado,
                 estimulo, alpha, duracion, fase_nombre,
                 omega_natural, amort_natural, DIM_TIME):
    sr, audio = cargar_audio(estimulo, duracion=duracion)
    ventana_muestras = int(sr * VENTANA_FFT_MS / 1000)
    hop_muestras = int(sr * HOP_FFT_MS / 1000)
    n_pasos = int(duracion / DT)
    
    ged_hist = []
    lf_hist = []
    fraccion_W_hist = []
    w_exp_hist = []
    w_norma_hist = []
    
    for idx in range(n_pasos):
        objetivo = preparar_objetivo_audio(audio, sr, idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
        
        ged = calcular_gradiente_espectral_diferencial(Phi_total, DIM_INTERNA, DIM_TIME)
        
        (Phi_total, Phi_vel_total, W, W_exploracion, Phi_int_historia,
         Phi_generado, lf_activa, fraccion_W) = actualizar_campo_con_competencia(
            Phi_total, Phi_vel_total, W, W_exploracion,
            Phi_int_historia, Phi_generado,
            objetivo, alpha,
            omega_natural, amort_natural,
            DT, entrenando=False,
            DIM_TIME=DIM_TIME
        )
        
        ged_hist.append(ged)
        lf_hist.append(1 if lf_activa else 0)
        fraccion_W_hist.append(fraccion_W)
        w_exp_hist.append(np.mean(np.abs(W_exploracion)))
        w_norma_hist.append(np.mean(np.abs(W)))
    
    return {
        'ged_mean': np.mean(ged_hist),
        'lf_pct': 100 * np.mean(lf_hist),
        'w_exp_mean': np.mean(w_exp_hist),
        'w_norma_mean': np.mean(w_norma_hist),
        'w_norma_max': np.max(w_norma_hist),
        'fraccion_W_mean': np.mean(fraccion_W_hist),
        'fraccion_W_min': np.min(fraccion_W_hist),
        'fraccion_W_max': np.max(fraccion_W_hist),
        'phi_total': Phi_total.copy(),
        'w': W.copy(),
        'w_exp': W_exploracion.copy(),
        'phi_int_historia': Phi_int_historia.copy()
    }

# ============================================================
# PRUEBA DIAGNÓSTICA CON ESTÍMULO NUEVO
# ============================================================
def probar_campo_con_estimulo(W, Phi_int_historia, estimulo, duracion,
                               omega_natural, amort_natural):
    """Prueba un campo con un estímulo, W congelado, mide GED medio"""
    np.random.seed(42)  # Misma inicialización
    Phi_total = inicializar_campo_total()
    Phi_vel_total = np.zeros_like(Phi_total)
    
    sr, audio = estimulo
    ventana_muestras = int(sr * VENTANA_FFT_MS / 1000)
    hop_muestras = int(sr * HOP_FFT_MS / 1000)
    n_pasos = int(duracion / DT)
    
    ged_hist = []
    alpha_test = 0.01  # Muy bajo para no saturar, solo suficiente para acoplar
    
    for idx in range(n_pasos):
        objetivo = preparar_objetivo_audio(audio, sr, idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
        
        region_int = Phi_total[:DIM_INTERNA, :]
        region_aud = Phi_total[DIM_INTERNA:, :]
        
        # Usar W fijo (congelado)
        prediccion_W = W @ region_aud
        M_hebb = GAMMA_PLAST * (prediccion_W - region_int)
        
        # Atractor con historia dada
        M_atractor = GAMMA_ATRACTOR * (Phi_int_historia - region_int)
        
        M_campo = np.zeros_like(Phi_total)
        M_campo[:DIM_INTERNA, :] = M_hebb + M_atractor
        
        # Dinámica base
        promedio_local = vecinos(Phi_total)
        difusion = DIFUSION_BASE * (promedio_local - Phi_total)
        desviacion = Phi_total - promedio_local
        reaccion = GANANCIA_REACCION * desviacion * (1 - desviacion**2)
        term_osc = (-omega_natural**2 * (Phi_total - PHI_EQUILIBRIO)
                    - amort_natural * Phi_vel_total)
        
        dPhi_vel = term_osc + reaccion + difusion + M_campo
        Phi_vel_nueva = Phi_vel_total + DT * dPhi_vel
        Phi_nueva = Phi_total + DT * Phi_vel_nueva
        
        # Mezcla con estímulo
        region_auditiva_nueva = Phi_nueva[DIM_INTERNA:, :]
        region_auditiva_nueva = (1 - alpha_test) * region_auditiva_nueva + alpha_test * objetivo
        Phi_nueva[DIM_INTERNA:, :] = region_auditiva_nueva
        
        Phi_total = np.clip(Phi_nueva, LIMITE_MIN, LIMITE_MAX)
        Phi_vel_total = np.clip(Phi_vel_nueva, -5.0, 5.0)
        
        ged = calcular_gradiente_espectral_diferencial(Phi_total, DIM_INTERNA, DIM_TIME)
        ged_hist.append(ged)
    
    return np.mean(ged_hist), np.std(ged_hist)

def entrenar_campo_original():
    """Entrena un campo desde cero con voz (30s, alpha=0.05) y retorna W y Phi_int_historia"""
    np.random.seed(42)
    Phi_total = inicializar_campo_total()
    Phi_vel_total = np.zeros_like(Phi_total)
    W = inicializar_plasticidad()
    W_exploracion = inicializar_w_exploracion()
    Phi_int_historia = inicializar_historia_interna()
    Phi_generado = inicializar_phi_generado()
    
    omega_natural, amort_natural = calcular_frecuencias_naturales(DIM_TOTAL, DIM_INTERNA)
    
    sr_voz, audio_voz = cargar_audio("Voz_Estudio.wav", duracion=DURACION_ENTRENO)
    ventana_muestras = int(sr_voz * VENTANA_FFT_MS / 1000)
    hop_muestras = int(sr_voz * HOP_FFT_MS / 1000)
    
    for idx in range(N_PASOS_ENTRENO):
        objetivo = preparar_objetivo_audio(audio_voz, sr_voz, idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
        
        (Phi_total, Phi_vel_total, W, W_exploracion, Phi_int_historia,
         Phi_generado, _, _) = actualizar_campo_con_competencia(
            Phi_total, Phi_vel_total, W, W_exploracion,
            Phi_int_historia, Phi_generado,
            objetivo, alpha=0.05,
            omega_natural=omega_natural, amort_natural=amort_natural,
            dt=DT, entrenando=True,
            DIM_TIME=DIM_TIME
        )
    
    return W.copy(), Phi_int_historia.copy(), omega_natural, amort_natural

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 100)
    print("VSTCosmos v79 — Competencia de memorias con diagnóstico")
    print("")
    print("Principio canónico: ningún umbral fijado externamente.")
    print("El campo decide qué régimen sostener por coherencia predictiva.")
    print("")
    print("Al final: prueba diagnóstica con estímulo nuevo para evaluar")
    print("si la integración amplió funcionalmente el dominio.")
    print("=" * 100)

    # Inicialización
    Phi_total = inicializar_campo_total()
    Phi_vel_total = np.zeros_like(Phi_total)
    W = inicializar_plasticidad()
    W_exploracion = inicializar_w_exploracion()
    Phi_int_historia = inicializar_historia_interna()
    Phi_generado = inicializar_phi_generado()
    
    omega_natural, amort_natural = calcular_frecuencias_naturales(DIM_TOTAL, DIM_INTERNA)
    
    # Entrenamiento
    print("\n[Fase 1] Entrenamiento (voz, alpha=0.05, 30s)")
    sr_voz, audio_voz = cargar_audio("Voz_Estudio.wav", duracion=DURACION_ENTRENO)
    ventana_muestras = int(sr_voz * VENTANA_FFT_MS / 1000)
    hop_muestras = int(sr_voz * HOP_FFT_MS / 1000)
    
    for idx in range(N_PASOS_ENTRENO):
        objetivo = preparar_objetivo_audio(audio_voz, sr_voz, idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
        
        (Phi_total, Phi_vel_total, W, W_exploracion, Phi_int_historia,
         Phi_generado, _, _) = actualizar_campo_con_competencia(
            Phi_total, Phi_vel_total, W, W_exploracion,
            Phi_int_historia, Phi_generado,
            objetivo, alpha=0.05,
            omega_natural=omega_natural, amort_natural=amort_natural,
            dt=DT, entrenando=True,
            DIM_TIME=DIM_TIME
        )
    
    # Guardar estado original para diagnóstico
    W_original = W.copy()
    Phi_int_original = Phi_int_historia.copy()
    
    print(f"  W_tras_entreno: {np.mean(np.abs(W)):.4f}")
    print(f"  W_exp_tras_entreno: {np.mean(np.abs(W_exploracion)):.4f}")
    
    # Fases de test
    fases = [
        ("Fase 2", "Voz_Estudio.wav", 0.0, "Dominio (voz)", DURACION_FASE),
        ("Fase 3", "Brandemburgo.wav", 0.0, "No entrenado (música)", DURACION_FASE),
        ("Fase 4", "Tono puro", 0.0, "No entrenado (tono)", DURACION_FASE),
        ("Fase 5", "Voz+Viento_1.wav", 0.0, "Degradado (voz+viento)", DURACION_FASE),
        ("Fase 6", "Ruido blanco", 0.0, "Perturbación basal", DURACION_FASE),
        ("Fase 7", "Voz_Estudio.wav", 0.0, "Re-acoplamiento (voz)", DURACION_REACOPLAMIENTO)
    ]
    
    resultados = []
    
    for fase_id, estimulo, alpha, desc, duracion in fases:
        print(f"\n[{fase_id}] {desc}")
        res = simular_fase(
            Phi_total, Phi_vel_total, W, W_exploracion,
            Phi_int_historia, Phi_generado,
            estimulo, alpha, duracion, fase_id,
            omega_natural, amort_natural, DIM_TIME
        )
        resultados.append(res)
        
        Phi_total = res['phi_total']
        W = res['w']
        W_exploracion = res['w_exp']
        Phi_int_historia = res['phi_int_historia']
        
        print(f"    GED: {res['ged_mean']:.6f} | LF: {res['lf_pct']:.1f}%")
        print(f"    W: {res['w_norma_mean']:.4f} | W_exp: {res['w_exp_mean']:.4f}")
        print(f"    Fracción W: media={res['fraccion_W_mean']:.3f} min={res['fraccion_W_min']:.3f}")
    
    # Diagnóstico principal
    print("\n" + "=" * 100)
    print("DIAGNÓSTICO — COMPETENCIA DE MEMORIAS")
    print("=" * 100)
    
    w_f2 = resultados[0]['w_norma_mean']
    w_f7 = resultados[5]['w_norma_mean']
    frac_W_f2 = resultados[0]['fraccion_W_mean']
    frac_W_f7 = resultados[5]['fraccion_W_mean']
    lf_f7 = resultados[5]['lf_pct']
    
    print(f"\n  W norma Fase 2:        {w_f2:.4f}")
    print(f"  W norma Fase 7:        {w_f7:.4f}")
    print(f"  Dominio ampliado:      {'✅' if w_f7 > w_f2 * 1.2 else '❌'} ({w_f7/w_f2:.2f}x)")
    print(f"\n  Fracción W Fase 2:     {frac_W_f2:.3f} (W domina si >0.5)")
    print(f"  Fracción W Fase 7:     {frac_W_f7:.3f} (W domina si >0.5)")
    print(f"  LF activa Fase 7:      {lf_f7:.1f}%")
    
    # ============================================================
    # PRUEBA DIAGNÓSTICA CON ESTÍMULO NUEVO
    # ============================================================
    print("\n" + "=" * 100)
    print("PRUEBA DIAGNÓSTICA: Estímulo nuevo")
    print("=" * 100)
    
    # Generar estímulo nuevo
    sr_nuevo, audio_nuevo = generar_estimulo_nuevo(duracion=DURACION_DIAGNOSTICO)
    
    # Guardar el estado integrado (post-Fase 7)
    W_integrado = resultados[-1]['w']
    Phi_int_integrado = resultados[-1]['phi_int_historia']
    
    print(f"\n  Campo original (post-entrenamiento):")
    print(f"    W norma: {np.mean(np.abs(W_original)):.4f}")
    print(f"    Phi_int norma: {np.mean(np.abs(Phi_int_original)):.4f}")
    
    print(f"\n  Campo integrado (post-Fase 7):")
    print(f"    W norma: {np.mean(np.abs(W_integrado)):.4f}")
    print(f"    Phi_int norma: {np.mean(np.abs(Phi_int_integrado)):.4f}")
    
    print(f"\n  Estímulo nuevo: Barrido de frecuencias (200-2000Hz) con modulación asimétrica")
    
    # Probar campo original
    print("\n  Probando campo ORIGINAL...")
    ged_orig_mean, ged_orig_std = probar_campo_con_estimulo(
        W_original, Phi_int_original, (sr_nuevo, audio_nuevo),
        DURACION_DIAGNOSTICO, omega_natural, amort_natural
    )
    
    # Probar campo integrado
    print("  Probando campo INTEGRADO...")
    ged_int_mean, ged_int_std = probar_campo_con_estimulo(
        W_integrado, Phi_int_integrado, (sr_nuevo, audio_nuevo),
        DURACION_DIAGNOSTICO, omega_natural, amort_natural
    )
    
    print("\n" + "=" * 100)
    print("RESULTADO DE LA PRUEBA DIAGNÓSTICA")
    print("=" * 100)
    print(f"  Campo original:   GED = {ged_orig_mean:.6f} ± {ged_orig_std:.6f}")
    print(f"  Campo integrado:  GED = {ged_int_mean:.6f} ± {ged_int_std:.6f}")
    print(f"  Ganancia: {ged_int_mean / (ged_orig_mean + 1e-10):.2f}x")
    
    if ged_int_mean > ged_orig_mean * 1.5:
        print("\n  ✅ INTEGRACIÓN AMPLIÓ EL DOMINIO FUNCIONAL")
        print("     El campo integrado se acopla mejor al estímulo nuevo")
        print("     que el campo original. La exaptación es funcional.")
    elif ged_int_mean > ged_orig_mean:
        print("\n  ⚠️  GANANCIA MODESTA: dominio ampliado pero cerca del original")
    else:
        print("\n  ❌ INTEGRACIÓN SIN GANANCIA FUNCIONAL")
        print("     El dominio se amplió pero perdió especificidad.")
        print("     W integrada es más grande pero no mejor.")
    
    # ============================================================
    # GUARDAR RESULTADOS
    # ============================================================
    
    # Guardar CSV principal
    with open('v79_competencia_memorias.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['fase', 'ged_mean', 'lf_pct', 'w_norma', 'w_exp_norma', 'fraccion_W_mean'])
        for i, (fase, res) in enumerate(zip(fases, resultados)):
            writer.writerow([fase[0], res['ged_mean'], res['lf_pct'],
                            res['w_norma_mean'], res['w_exp_mean'],
                            res['fraccion_W_mean']])
    
    # Guardar diagnóstico
    with open('v79_diagnostico_estimulo_nuevo.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['campo', 'w_norma', 'ged_mean', 'ged_std'])
        writer.writerow(['original', np.mean(np.abs(W_original)), ged_orig_mean, ged_orig_std])
        writer.writerow(['integrado', np.mean(np.abs(W_integrado)), ged_int_mean, ged_int_std])
        writer.writerow(['ganancia_x', '', ged_int_mean / (ged_orig_mean + 1e-10), ''])
    
    print("\n  CSV guardado: v79_competencia_memorias.csv")
    print("  CSV guardado: v79_diagnostico_estimulo_nuevo.csv")
    
    # Gráfico
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    nombres = ['Voz', 'Música', 'Tono', 'Voz+Viento', 'Ruido', 'Reacop']
    
    axes[0,0].bar(nombres, [r['ged_mean'] for r in resultados])
    axes[0,0].set_title('GED')
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].bar(nombres, [r['lf_pct'] for r in resultados])
    axes[0,1].set_title('LF activa (%)')
    axes[0,1].grid(True, alpha=0.3)
    
    axes[0,2].bar(nombres, [r['w_norma_mean'] for r in resultados], alpha=0.7, label='W')
    axes[0,2].bar(nombres, [r['w_exp_mean'] for r in resultados], alpha=0.5, label='W_exp')
    axes[0,2].set_title('Norma de memorias')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    axes[1,0].bar(nombres, [r['fraccion_W_mean'] for r in resultados])
    axes[1,0].axhline(y=0.5, color='r', linestyle='--', label='Equilibrio (0.5)')
    axes[1,0].set_title('Fracción de peso de W')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Gráfico de diagnóstico
    axes[1,1].bar(['Original', 'Integrado'], [ged_orig_mean, ged_int_mean], 
                  yerr=[ged_orig_std, ged_int_std], capsize=5)
    axes[1,1].set_title('GED con estímulo nuevo')
    axes[1,1].set_ylabel('GED')
    axes[1,1].grid(True, alpha=0.3)
    
    axes[1,2].bar(['Original', 'Integrado'], 
                  [np.mean(np.abs(W_original)), np.mean(np.abs(W_integrado))])
    axes[1,2].set_title('W norma')
    axes[1,2].grid(True, alpha=0.3)
    
    plt.suptitle('VSTCosmos v79 — Competencia de memorias + diagnóstico', fontsize=14)
    plt.tight_layout()
    plt.savefig('v79_competencia_memorias.png', dpi=150)
    print("  Gráfico guardado: v79_competencia_memorias.png")
    
    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)


if __name__ == "__main__":
    main()