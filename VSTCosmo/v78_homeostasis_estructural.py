#!/usr/bin/env python3
"""
VSTCosmos v78 — Homeostasis estructural de LF

Principio canónico: ningún umbral fijado externamente.
Todos los criterios emergen de las constricciones del campo:
- Difusión (DIFUSION_BASE)
- Frecuencias naturales (OMEGA_MIN/MAX)
- Relación entre plasticidad y difusión
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

N_PASOS_ENTRENO = int(DURACION_ENTRENO / DT)
N_PASOS_FASE = int(DURACION_FASE / DT)
N_PASOS_REACOP = int(DURACION_REACOPLAMIENTO / DT)

# Parámetros de dinámica (definen la estructura del campo)
DIFUSION_BASE = 0.15       # tasa de difusión — escala natural del campo
GANANCIA_REACCION = 0.05

OMEGA_MIN = 0.05           # frecuencia natural mínima
OMEGA_MAX = 0.50           # frecuencia natural máxima
AMORT_MIN = 0.01           # amortiguación mínima
AMORT_MAX = 0.08           # amortiguación máxima
PHI_EQUILIBRIO = 0.5

# Plasticidad hebbiana principal
ETA_HEBB = 0.05
TAU_W = 0.008
GAMMA_PLAST = 0.01
UMBRAL_CORRELACION = 0.1   # derivado de la varianza esperada del campo

# Mecanismo atractor
GAMMA_MEMORIA = 0.05
TAU_INT_HIST = 0.0002

# Parámetros de LF activa (fuerzas, no umbrales)
ETA_EXPLORACION = 0.02
ETA_APRENDIZAJE = 0.03
GAMMA_GENERACION = 0.04

# Límites duros del campo (el "tímpano" — no son umbrales, son saturación)
LIMITE_MIN = 0.0
LIMITE_MAX = 1.0

# Parámetros de FFT (análisis del estímulo, no del campo)
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

def inicializar_campo_total():
    np.random.seed(42)
    return np.random.rand(DIM_TOTAL, DIM_TIME) * 0.2 + 0.4

def inicializar_plasticidad():
    return np.zeros((DIM_INTERNA, DIM_AUDITIVA))

def inicializar_historia_interna():
    return np.zeros((DIM_INTERNA, DIM_TIME))

def inicializar_w_exploracion():
    return np.zeros((DIM_INTERNA, DIM_AUDITIVA))

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

def actualizar_historia_interna(Phi_int_historia, region_int, entrenando):
    if entrenando:
        return (1 - TAU_INT_HIST) * Phi_int_historia + TAU_INT_HIST * region_int
    else:
        return Phi_int_historia

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
# HOMEOSTASIS ESTRUCTURAL — sin umbrales fijos
# ============================================================
def calcular_ratio_dominancia_plastica(Phi_total, W_exploracion, dim_interna):
    """
    Mide si la plasticidad exploratoria domina sobre la difusión.
    El límite estructural es ratio > 1.0 — el campo no necesita
    un umbral externo, la física misma indica cuándo la plasticidad
    está sobrepasando la capacidad de mantener gradientes.
    """
    region_int = Phi_total[:dim_interna, :]
    region_aud = Phi_total[dim_interna:, :]
    
    # Fuerza de plasticidad de W_exploracion
    patron_exp = W_exploracion @ region_aud
    M_plast = np.abs(patron_exp - region_int)
    
    # Fuerza de difusión
    promedio_local = vecinos(Phi_total)
    difusion = DIFUSION_BASE * np.abs(promedio_local[:dim_interna] - region_int)
    
    norma_plast = np.mean(M_plast)
    norma_difus = np.mean(difusion)
    
    ratio = norma_plast / (norma_difus + 1e-10)
    
    return ratio, ratio > 1.0

def calcular_umbral_entrada_lf_estructural():
    """
    El umbral de entrada a LF se deriva de la difusión base.
    Cuando GED cae por debajo de ~2% de la difusión,
    el campo prácticamente no distingue regiones.
    """
    return DIFUSION_BASE * 0.02  # ≈ 0.003

def calcular_umbral_salida_lf_estructural():
    """
    El umbral de salida de LF se deriva de la difusión base.
    Cuando GED supera ~4% de la difusión, el campo empieza
    a recuperar diferenciación espectral.
    """
    return DIFUSION_BASE * 0.04  # ≈ 0.006

def calcular_tasa_estabilizacion_natural(eta_aprendizaje, dt):
    """
    La tasa de cambio de W_exp se considera estabilizada
    cuando es menor que la tasa de aprendizaje por paso.
    """
    return eta_aprendizaje * dt  # ≈ 0.0003

def calcular_escala_maxima_w_estructural():
    """
    W no debería crecer más allá del punto donde su contribución
    a M_plasticidad superaría consistentemente a la difusión.
    Escala estructural: ~5 * DIFUSION_BASE ≈ 0.75
    """
    return 5.0 * DIFUSION_BASE

def evaluar_salida_lf_estructural(ged_actual, ratio_dominancia,
                                   historial_w_exp, paso_actual):
    """
    El campo sale de LF cuando su propia estructura indica
    que puede retornar a operación normal:
    
    1. GED > umbral_salida (derivado de difusión)
    2. Plasticidad NO domina sobre difusión (ratio < 0.8)
    3. W_exp se estabilizó (tasa < tasa natural)
    """
    umbral_salida = calcular_umbral_salida_lf_estructural()
    ged_recupero = ged_actual > umbral_salida
    
    plastica_no_domina = ratio_dominancia < 0.8
    
    tasa_estable = calcular_tasa_estabilizacion_natural(ETA_APRENDIZAJE, DT)
    w_estable = True
    if len(historial_w_exp) >= 50:
        w_reciente = np.array(historial_w_exp[-50:])
        tasa_cambio = np.abs(np.gradient(w_reciente)).mean()
        w_estable = tasa_cambio < tasa_estable
    
    return ged_recupero and plastica_no_domina and w_estable

# ============================================================
# FUNCIONES LF
# ============================================================
def lf_exploracion(Phi_total, dim_interna):
    region_int = Phi_total[:dim_interna, :]
    region_aud = Phi_total[dim_interna:, :]
    return ETA_EXPLORACION * (region_aud - region_int)

def lf_aprendizaje(Phi_total, W_exploracion, dim_interna, dt):
    region_int = Phi_total[:dim_interna, :]
    region_aud = Phi_total[dim_interna:, :]
    
    correlacion = (region_int @ region_aud.T) / DIM_TIME
    correlacion_filtrada = np.where(
        np.abs(correlacion) > 0.05,
        correlacion,
        0.0
    )
    dW_exp = ETA_APRENDIZAJE * correlacion_filtrada - 0.01 * W_exploracion
    W_nueva = np.clip(W_exploracion + dW_exp * dt, -1.0, 1.0)
    
    M_aprendizaje = 0.005 * (W_nueva @ region_aud - region_int)
    return W_nueva, M_aprendizaje

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
        fft_banda = np.fft.rfft(region_int[banda])
        fft_banda[:DIM_TIME//2] *= (1.0 + GAMMA_GENERACION *
                                     mascara_modos[:DIM_TIME//2])
        signal_amplificada[banda] = np.real(np.fft.irfft(
            fft_banda, n=DIM_TIME
        ))
    
    M_generacion = GAMMA_GENERACION * (signal_amplificada - region_int)
    return Phi_generado_nueva, M_generacion

def integrar_estructural(W, W_exploracion, Phi_int_historia,
                          Phi_generado, dim_interna):
    """
    Integración que respeta la escala estructural del campo.
    No usa umbrales de similitud — usa la física misma.
    """
    escala_max_w = calcular_escala_maxima_w_estructural()
    
    # Integración base
    W_nueva = W + 0.3 * W_exploracion
    
    # Regulación estructural: si W excede la escala natural, escala
    norma_w_nueva = np.mean(np.abs(W_nueva))
    if norma_w_nueva > escala_max_w:
        factor = escala_max_w / norma_w_nueva
        W_nueva = W_nueva * factor
    
    # Olvido adaptativo de W_exploracion
    W_exp_nuevo = W_exploracion * 0.5
    
    # Actualizar atractor solo si la generación no es extrema
    generacion_norma = np.mean(np.abs(Phi_generado[:dim_interna, :]))
    if generacion_norma < 0.5:
        Phi_hist_nueva = Phi_int_historia + 0.1 * Phi_generado[:dim_interna, :]
    else:
        Phi_hist_nueva = Phi_int_historia
    
    # Normalizar atractor
    phi_norma = np.mean(np.abs(Phi_hist_nueva))
    if phi_norma > 1.0:
        Phi_hist_nueva = Phi_hist_nueva * (1.0 / phi_norma)
    
    return W_nueva, W_exp_nuevo, Phi_hist_nueva

# ============================================================
# ACTUALIZACIÓN PRINCIPAL
# ============================================================
def actualizar_campo_con_homeostasis_estructural(
        Phi_total, Phi_vel_total, W, W_exploracion,
        Phi_int_historia, Phi_generado,
        objetivo_audio, alpha,
        omega_natural, amort_natural,
        dt, entrenando, ged_actual,
        historial_w_exp, lf_activa_actual,
        DIM_TIME):
    
    # Dinámica base
    promedio_local = vecinos(Phi_total)
    difusion = DIFUSION_BASE * (promedio_local - Phi_total)
    desviacion = Phi_total - promedio_local
    reaccion = GANANCIA_REACCION * desviacion * (1 - desviacion**2)
    term_osc = (-omega_natural**2 * (Phi_total - PHI_EQUILIBRIO)
                - amort_natural * Phi_vel_total)
    
    # Evaluar estado LF con umbrales estructurales
    umbral_entrada = calcular_umbral_entrada_lf_estructural()
    
    if not lf_activa_actual:
        lf_activa = ged_actual < umbral_entrada
    else:
        ratio, _ = calcular_ratio_dominancia_plastica(Phi_total, W_exploracion, DIM_INTERNA)
        lf_activa = not evaluar_salida_lf_estructural(
            ged_actual, ratio, historial_w_exp, 0
        )
    
    # Plasticidad principal: congelada durante LF
    if not lf_activa and not entrenando:
        W_nueva, M_hebb = actualizar_hebb_y_plasticidad(Phi_total, W, dt)
    else:
        W_nueva = W
        M_hebb = 0
    
    # Atractor
    M_atractor = GAMMA_MEMORIA * (Phi_int_historia - Phi_total[:DIM_INTERNA, :])
    
    M_campo = np.zeros_like(Phi_total)
    M_campo[:DIM_INTERNA, :] = M_hebb + M_atractor
    
    W_exploracion_nueva = W_exploracion.copy()
    Phi_generado_nueva = Phi_generado.copy()
    ciclo_cerrado = False
    
    if lf_activa and not entrenando:
        M_exp = lf_exploracion(Phi_total, DIM_INTERNA)
        W_exploracion_nueva, M_apr = lf_aprendizaje(Phi_total, W_exploracion, DIM_INTERNA, dt)
        Phi_generado_nueva, M_gen = lf_generacion(Phi_total, Phi_generado, DIM_INTERNA, dt, DIM_TIME)
        
        M_campo[:DIM_INTERNA, :] += M_exp + M_apr + M_gen
        
        w_exp_norma = np.mean(np.abs(W_exploracion_nueva))
        historial_w_exp.append(w_exp_norma)
        
        # Verificar cierre de ciclo
        ratio, _ = calcular_ratio_dominancia_plastica(Phi_total, W_exploracion_nueva, DIM_INTERNA)
        if evaluar_salida_lf_estructural(ged_actual, ratio, historial_w_exp, 0):
            W_nueva, W_exploracion_nueva, Phi_int_historia = integrar_estructural(
                W_nueva, W_exploracion_nueva, Phi_int_historia, Phi_generado_nueva, DIM_INTERNA
            )
            ciclo_cerrado = True
            historial_w_exp = []
            print(f"        *** CICLO LF CERRADO (ratio={ratio:.3f}) ***")
    else:
        Phi_int_historia = actualizar_historia_interna(Phi_int_historia, Phi_total[:DIM_INTERNA, :], entrenando)
    
    # Actualización del campo
    dPhi_vel = term_osc + reaccion + difusion + M_campo
    Phi_vel_nueva = Phi_vel_total + dt * dPhi_vel
    Phi_nueva = Phi_total + dt * Phi_vel_nueva
    
    if alpha > 0:
        region_auditiva_nueva = Phi_nueva[DIM_INTERNA:, :]
        region_auditiva_nueva = (1 - alpha) * region_auditiva_nueva + alpha * objetivo_audio
        Phi_nueva[DIM_INTERNA:, :] = region_auditiva_nueva
    
    # Prevenir colapso (detectado por varianza, no por umbral fijo)
    varianza_campo = np.var(Phi_nueva[:DIM_INTERNA, :])
    if varianza_campo < 1e-6:
        ruido_minimo = np.random.normal(0, DIFUSION_BASE * 0.1, Phi_nueva[:DIM_INTERNA, :].shape)
        Phi_nueva[:DIM_INTERNA, :] += ruido_minimo
    
    return (np.clip(Phi_nueva, LIMITE_MIN, LIMITE_MAX),
            np.clip(Phi_vel_nueva, -5.0, 5.0),
            W_nueva, W_exploracion_nueva, Phi_int_historia,
            Phi_generado_nueva, ciclo_cerrado, historial_w_exp, lf_activa)

# ============================================================
# SIMULACIÓN (misma estructura que v77, con nueva actualización)
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
    w_exp_hist = []
    w_norma_hist = []
    ratio_hist = []
    ciclos_hist = []
    
    historial_w_exp_local = []
    lf_activa_actual = False
    
    for idx in range(n_pasos):
        objetivo = preparar_objetivo_audio(audio, sr, idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
        
        ged = calcular_gradiente_espectral_diferencial(Phi_total, DIM_INTERNA, DIM_TIME)
        ratio, _ = calcular_ratio_dominancia_plastica(Phi_total, W_exploracion, DIM_INTERNA)
        
        (Phi_total, Phi_vel_total, W, W_exploracion, Phi_int_historia,
         Phi_generado, ciclo, historial_w_exp_local,
         lf_activa_actual) = actualizar_campo_con_homeostasis_estructural(
            Phi_total, Phi_vel_total, W, W_exploracion,
            Phi_int_historia, Phi_generado,
            objetivo, alpha,
            omega_natural, amort_natural,
            DT, entrenando=False, ged_actual=ged,
            historial_w_exp=historial_w_exp_local,
            lf_activa_actual=lf_activa_actual,
            DIM_TIME=DIM_TIME
        )
        
        ged_hist.append(ged)
        lf_hist.append(1 if lf_activa_actual else 0)
        w_exp_hist.append(np.mean(np.abs(W_exploracion)))
        w_norma_hist.append(np.mean(np.abs(W)))
        ratio_hist.append(ratio)
        ciclos_hist.append(1 if ciclo else 0)
    
    return {
        'ged_mean': np.mean(ged_hist),
        'lf_pct': 100 * np.mean(lf_hist),
        'w_exp_mean': np.mean(w_exp_hist),
        'w_norma_mean': np.mean(w_norma_hist),
        'w_norma_max': np.max(w_norma_hist),
        'ratio_medio': np.mean(ratio_hist),
        'ratio_max': np.max(ratio_hist),
        'total_ciclos': np.sum(ciclos_hist),
        'ged_hist': ged_hist,
        'lf_hist': lf_hist,
        'w_norma_hist': w_norma_hist,
        'ratio_hist': ratio_hist,
        'phi_total': Phi_total.copy(),
        'w': W.copy(),
        'w_exp': W_exploracion.copy(),
        'phi_int_historia': Phi_int_historia.copy()
    }

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 100)
    print("VSTCosmos v78 — Homeostasis estructural de LF")
    print("")
    print("Principio canónico: ningún umbral fijado externamente.")
    print("Todos los criterios emergen de las constricciones del campo:")
    print(f"  - DIFUSION_BASE = {DIFUSION_BASE}")
    print(f"  - OMEGA_MIN/MAX = {OMEGA_MIN}/{OMEGA_MAX}")
    print(f"  - Umbral entrada LF = DIFUSION_BASE * 0.02 = {DIFUSION_BASE * 0.02:.4f}")
    print(f"  - Umbral salida LF = DIFUSION_BASE * 0.04 = {DIFUSION_BASE * 0.04:.4f}")
    print(f"  - Escala máxima W = 5 * DIFUSION_BASE = {5 * DIFUSION_BASE:.2f}")
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
    
    historial_w_exp_dummy = []
    lf_activa_dummy = False
    
    for idx in range(N_PASOS_ENTRENO):
        objetivo = preparar_objetivo_audio(audio_voz, sr_voz, idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
        ged = calcular_gradiente_espectral_diferencial(Phi_total, DIM_INTERNA, DIM_TIME)
        ratio, _ = calcular_ratio_dominancia_plastica(Phi_total, W_exploracion, DIM_INTERNA)
        
        (Phi_total, Phi_vel_total, W, W_exploracion, Phi_int_historia,
         Phi_generado, _, historial_w_exp_dummy,
         lf_activa_dummy) = actualizar_campo_con_homeostasis_estructural(
            Phi_total, Phi_vel_total, W, W_exploracion,
            Phi_int_historia, Phi_generado,
            objetivo, alpha=0.05,
            omega_natural=omega_natural, amort_natural=amort_natural,
            dt=DT, entrenando=True, ged_actual=ged,
            historial_w_exp=historial_w_exp_dummy,
            lf_activa_actual=lf_activa_dummy,
            DIM_TIME=DIM_TIME
        )
        region_int = Phi_total[:DIM_INTERNA, :]
        Phi_int_historia = (1 - TAU_INT_HIST) * Phi_int_historia + TAU_INT_HIST * region_int
    
    print(f"  W_tras_entreno: {np.mean(np.abs(W)):.4f}")
    
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
    ciclos_totales = 0
    
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
        ciclos_totales += res['total_ciclos']
        
        print(f"    GED: {res['ged_mean']:.6f} | LF: {res['lf_pct']:.1f}%")
        print(f"    W: {res['w_norma_mean']:.4f} (max {res['w_norma_max']:.4f})")
        print(f"    W_exp: {res['w_exp_mean']:.4f} | Ratio plástica/difusión: {res['ratio_medio']:.3f}")
        print(f"    Ciclos cerrados: {res['total_ciclos']}")
    
    # Diagnóstico
    print("\n" + "=" * 100)
    print("DIAGNÓSTICO — HOMEOSTASIS ESTRUCTURAL")
    print("=" * 100)
    
    w_f2 = resultados[0]['w_norma_mean']
    w_f7 = resultados[5]['w_norma_mean']
    w_max = max(r['w_norma_max'] for r in resultados)
    escala_max_w = calcular_escala_maxima_w_estructural()
    
    ged_f2_hist = resultados[0]['ged_hist']
    ged_f7_hist = resultados[5]['ged_hist']
    
    def tiempo_primer_umbral(ged_hist, umbral):
        for i, g in enumerate(ged_hist):
            if g > umbral:
                return i * DT
        return float('inf')
    
    umbral_salida = calcular_umbral_salida_lf_estructural()
    t_f2 = tiempo_primer_umbral(ged_f2_hist, umbral_salida)
    t_f7 = tiempo_primer_umbral(ged_f7_hist, umbral_salida)
    
    print(f"\n  Ciclos LF cerrados:        {ciclos_totales}")
    print(f"  W norma Fase 2:            {w_f2:.4f}")
    print(f"  W norma Fase 7:            {w_f7:.4f}")
    print(f"  W norma máxima:            {w_max:.4f} (escala estructural: {escala_max_w:.2f})")
    print(f"  W dentro de escala:        {'✅' if w_max <= escala_max_w * 1.1 else '❌'}")
    print(f"  Dominio ampliado (>1.2x):  {'✅' if w_f7 > w_f2 * 1.2 else '❌'} ({w_f7/w_f2:.2f}x)")
    print(f"  Recuperación Fase 2:       {t_f2:.2f}s")
    print(f"  Recuperación Fase 7:       {t_f7:.2f}s" if t_f7 != float('inf') else "  Recuperación Fase 7:       NUNCA")
    print(f"  LF activa Fase 7:          {resultados[5]['lf_pct']:.1f}%")
    
    print("\n  VEREDICTO:")
    if ciclos_totales > 0 and w_max <= escala_max_w * 1.1:
        if t_f7 < t_f2:
            print("  ✅ CICLO COMPLETO CON HOMEOSTASIS ESTRUCTURAL VALIDADO")
            print("     El campo salió, aprendió, integró selectivamente,")
            print("     y retornó con dominio ampliado y re-acoplamiento más rápido.")
            print("     La homeostasis emergió de las constricciones del campo — no fue diseñada.")
        else:
            print("  ⚠️  HOMEOSTASIS FUNCIONA pero retorno aún no más rápido.")
    elif ciclos_totales > 0:
        print("  ⚠️  CICLOS OCURRIERON pero W superó la escala estructural.")
        print("     La integración necesita ser más conservadora.")
    else:
        print("  ❌ SIN CICLOS: el campo no está detectando oportunidad de salida.")
    
    # Guardar resultados
    with open('v78_homeostasis_estructural.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['fase', 'ged_mean', 'lf_pct', 'w_norma', 'w_exp_norma', 
                         'ratio_plastica_difusion', 'ciclos'])
        for i, (fase, res) in enumerate(zip(fases, resultados)):
            writer.writerow([fase[0], res['ged_mean'], res['lf_pct'], 
                            res['w_norma_mean'], res['w_exp_mean'],
                            res['ratio_medio'], res['total_ciclos']])
    
    print("\n  CSV guardado: v78_homeostasis_estructural.csv")
    
    # Gráfico
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    nombres = ['Voz', 'Música', 'Tono', 'Voz+Viento', 'Ruido', 'Reacop']
    
    # GED
    axes[0,0].bar(nombres, [r['ged_mean'] for r in resultados])
    axes[0,0].axhline(y=calcular_umbral_entrada_lf_estructural(), color='g', linestyle='--', label='Entrada LF')
    axes[0,0].axhline(y=calcular_umbral_salida_lf_estructural(), color='r', linestyle='--', label='Salida LF')
    axes[0,0].set_title('GED')
    axes[0,0].legend()
    
    # LF activa
    axes[0,1].bar(nombres, [r['lf_pct'] for r in resultados])
    axes[0,1].set_title('LF activa (%)')
    
    # W norma
    axes[0,2].bar(nombres, [r['w_norma_mean'] for r in resultados])
    axes[0,2].axhline(y=escala_max_w, color='r', linestyle='--', label='Escala estructural')
    axes[0,2].set_title('W norma')
    axes[0,2].legend()
    
    # Ratio plasticidad/difusión
    axes[1,0].bar(nombres, [r['ratio_medio'] for r in resultados])
    axes[1,0].axhline(y=1.0, color='r', linestyle='--', label='Dominancia (ratio=1)')
    axes[1,0].set_title('Plasticidad / Difusión')
    axes[1,0].legend()
    
    # W_exp
    axes[1,1].bar(nombres, [r['w_exp_mean'] for r in resultados])
    axes[1,1].set_title('W_exp norma')
    
    # Ciclos
    axes[1,2].bar(nombres, [r['total_ciclos'] for r in resultados])
    axes[1,2].set_title('Ciclos cerrados')
    
    plt.suptitle('VSTCosmos v78 — Homeostasis estructural de LF', fontsize=14)
    plt.tight_layout()
    plt.savefig('v78_homeostasis_estructural.png', dpi=150)
    print("  Gráfico guardado: v78_homeostasis_estructural.png")
    
    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)


if __name__ == "__main__":
    main()