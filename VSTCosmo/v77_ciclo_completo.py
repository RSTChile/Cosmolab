#!/usr/bin/env python3
"""
VSTCosmos v77 — El ciclo completo: salida, exploración, retorno
Implementa el ciclo evolutivo LF:
    activación → aprendizaje/generación → integración → retorno al dominio ampliado
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import csv
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PARÁMETROS GLOBALES
# ============================================================
DIM_INTERNA = 32
DIM_AUDITIVA = 32
DIM_TOTAL = DIM_INTERNA + DIM_AUDITIVA

DIM_TIME = 100
DT = 0.01

DURACION_ENTRENO = 30.0
DURACION_FASE = 20.0          # todas las fases de test duran 20s
DURACION_REACOPLAMIENTO = 30.0 # Fase 7 más larga para ver retorno

N_PASOS_ENTRENO = int(DURACION_ENTRENO / DT)
N_PASOS_FASE = int(DURACION_FASE / DT)
N_PASOS_REACOP = int(DURACION_REACOPLAMIENTO / DT)

# Parámetros de dinámica
DIFUSION_BASE = 0.15
GANANCIA_REACCION = 0.05

OMEGA_MIN = 0.05
OMEGA_MAX = 0.50
AMORT_MIN = 0.01
AMORT_MAX = 0.08
PHI_EQUILIBRIO = 0.5

# Plasticidad hebbiana principal
ETA_HEBB = 0.05
TAU_W = 0.008
GAMMA_PLAST = 0.01
W_MAX = 1.0
UMBRAL_CORRELACION = 0.1

# Mecanismo atractor
GAMMA_MEMORIA = 0.05
TAU_INT_HIST = 0.0002

# Parámetros para LF activa (mismos que v76)
UMBRAL_GED_LF = 0.003
ETA_EXPLORACION = 0.02
ETA_APRENDIZAJE = 0.03
GAMMA_GENERACION = 0.04
RUIDO_GENERACION = 0.05

# NUEVOS: Criterio de salida de LF
TAU_ESTABILIZACION = 50      # pasos para evaluar estabilidad de W_exp
UMBRAL_SALIDA_LF = 0.01      # tasa de cambio de W_exp por debajo de la cual se considera estabilizado

# NUEVO: Integración de aprendizaje LF
GAMMA_INTEGRACION = 0.3      # fracción de W_exp que se integra a W principal

# NUEVO: Balance de reserva
W_MAX_TOTAL = 1.5            # norma máxima combinada de W + W_exploracion

# Parámetros de FFT
VENTANA_FFT_MS = 25
HOP_FFT_MS = 10
F_MIN = 80
F_MAX = 8000

LIMITE_MIN = 0.0
LIMITE_MAX = 1.0

# ============================================================
# FUNCIONES BASE (sin cambios relevantes)
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
            # Si no existe el archivo, generar tono sintético
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
    W_nueva = np.clip(W + dW * dt, -W_MAX, W_MAX)

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
# NUEVAS FUNCIONES DEL CICLO COMPLETO
# ============================================================
def evaluar_salida_lf(historial_w_exp, ged_actual, umbral_ged=0.003):
    """
    El campo sale de LF cuando SE CUMPLEN LAS DOS condiciones:
    1. W_exploracion se estabilizó (tasa de cambio baja)
    2. GED sube por encima del umbral (recuperando diferenciación)
    """
    if len(historial_w_exp) < TAU_ESTABILIZACION:
        return False
    
    w_reciente = np.array(historial_w_exp[-TAU_ESTABILIZACION:])
    tasa_cambio = np.abs(np.gradient(w_reciente)).mean()
    w_estabilizado = tasa_cambio < UMBRAL_SALIDA_LF
    
    ged_recuperando = ged_actual > umbral_ged * 0.5  # umbral suavizado
    
    return w_estabilizado and ged_recuperando

def integrar_aprendizaje_lf(W, W_exploracion, Phi_int_historia,
                             Phi_generado, dim_interna):
    """
    Cuando el ciclo LF cierra, integrar parcialmente lo aprendido y generado.
    """
    # Ampliar W con lo aprendido bajo LF
    W_integrada = W + GAMMA_INTEGRACION * W_exploracion
    
    # Normalizar para que la norma total no explote
    norma_total = np.mean(np.abs(W_integrada))
    if norma_total > W_MAX:
        W_integrada = W_integrada * (W_MAX / norma_total)
    
    # Ampliar el atractor con lo generado bajo LF
    Phi_historia_ampliada = (Phi_int_historia +
                             GAMMA_INTEGRACION * Phi_generado[:dim_interna, :])
    
    # Normalizar atractor
    phi_norma = np.mean(np.abs(Phi_historia_ampliada))
    if phi_norma > 1.0:
        Phi_historia_ampliada = Phi_historia_ampliada * (1.0 / phi_norma)
    
    # Resetear W_exploracion para el próximo ciclo LF
    W_exploracion_reset = np.zeros_like(W_exploracion)
    
    return W_integrada, Phi_historia_ampliada, W_exploracion_reset

def balancear_reserva(W, W_exploracion):
    """Preservar parte del dominio original mientras LF aprende."""
    norma_combinada = np.mean(np.abs(W)) + np.mean(np.abs(W_exploracion))
    if norma_combinada > W_MAX_TOTAL:
        factor = W_MAX_TOTAL / norma_combinada
        W = W * factor
        W_exploracion = W_exploracion * factor
    return W.copy(), W_exploracion.copy()

def lf_exploracion(Phi_total, dim_interna, ged_actual):
    region_int = Phi_total[:dim_interna, :]
    region_aud = Phi_total[dim_interna:, :]
    M_exploracion = ETA_EXPLORACION * (region_aud - region_int)
    return M_exploracion

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
    W_nueva = np.clip(W_exploracion + dW_exp * dt, -W_MAX, W_MAX)
    
    M_aprendizaje = 0.005 * (W_nueva @ region_aud - region_int)
    return W_nueva, M_aprendizaje

def lf_generacion(Phi_total, Phi_generado, dim_interna, dt, DIM_TIME):
    region_int = Phi_total[:dim_interna, :]
    Phi_generado_nueva = 0.99 * Phi_generado + 0.01 * region_int
    
    perfil = _perfil_espectral(region_int, dim_interna, DIM_TIME)
    modo_dominante = np.argmax(perfil)
    mascara_modos = np.zeros(DIM_TIME)
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

def actualizar_campo_con_ciclo_lf(Phi_total, Phi_vel_total, W, W_exploracion,
                                    Phi_int_historia, Phi_generado,
                                    objetivo_audio, alpha,
                                    omega_natural, amort_natural,
                                    dt, entrenando, ged_actual,
                                    historial_w_exp, DIM_TIME):
    """Actualización con los tres mecanismos LF + criterio de salida + balance"""
    
    # Dinámica base
    promedio_local = vecinos(Phi_total)
    difusion = DIFUSION_BASE * (promedio_local - Phi_total)
    desviacion = Phi_total - promedio_local
    reaccion = GANANCIA_REACCION * desviacion * (1 - desviacion**2)
    term_osc = (-omega_natural**2 * (Phi_total - PHI_EQUILIBRIO)
                - amort_natural * Phi_vel_total)
    
    # Plasticidad hebbiana principal
    W_nueva, M_hebb = actualizar_hebb_y_plasticidad(Phi_total, W, dt)
    
    # Atractor aprendido
    M_atractor = GAMMA_MEMORIA * (Phi_int_historia - Phi_total[:DIM_INTERNA, :])
    
    M_campo = np.zeros_like(Phi_total)
    M_campo[:DIM_INTERNA, :] = M_hebb + M_atractor
    
    lf_activa = ged_actual < UMBRAL_GED_LF
    
    W_exploracion_nueva = W_exploracion.copy()
    Phi_generado_nueva = Phi_generado.copy()
    ciclo_cerrado = False
    
    if lf_activa:
        # Los tres mecanismos LF
        M_exp = lf_exploracion(Phi_total, DIM_INTERNA, ged_actual)
        W_exploracion_nueva, M_apr = lf_aprendizaje(Phi_total, W_exploracion, DIM_INTERNA, dt)
        Phi_generado_nueva, M_gen = lf_generacion(Phi_total, Phi_generado, DIM_INTERNA, dt, DIM_TIME)
        
        M_campo[:DIM_INTERNA, :] += M_exp + M_apr + M_gen
        
        # Balance de reserva
        W_nueva, W_exploracion_nueva = balancear_reserva(W_nueva, W_exploracion_nueva)
        
        # Verificar criterio de salida
        w_exp_norma_actual = np.mean(np.abs(W_exploracion_nueva))
        historial_w_exp.append(w_exp_norma_actual)
        
        if evaluar_salida_lf(historial_w_exp, ged_actual, UMBRAL_GED_LF):
            # Cerrar el ciclo: integrar aprendizaje al dominio principal
            W_nueva, Phi_int_historia_nueva, W_exploracion_nueva = integrar_aprendizaje_lf(
                W_nueva, W_exploracion_nueva, Phi_int_historia, Phi_generado_nueva, DIM_INTERNA
            )
            ciclo_cerrado = True
            # Resetear historial para próximos ciclos
            historial_w_exp = []
        else:
            Phi_int_historia_nueva = Phi_int_historia
    else:
        Phi_int_historia_nueva = Phi_int_historia
    
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
        ruido_minimo = np.random.normal(0, 0.01, Phi_nueva[:DIM_INTERNA, :].shape)
        Phi_nueva[:DIM_INTERNA, :] += ruido_minimo
    
    return (np.clip(Phi_nueva, LIMITE_MIN, LIMITE_MAX),
            np.clip(Phi_vel_nueva, -5.0, 5.0),
            W_nueva, W_exploracion_nueva, Phi_int_historia_nueva,
            Phi_generado_nueva, ciclo_cerrado, historial_w_exp)

def simular_fase(Phi_total, Phi_vel_total, W, W_exploracion,
                 Phi_int_historia, Phi_generado,
                 estimulo, alpha, duracion, fase_nombre,
                 omega_natural, amort_natural, DIM_TIME):
    """Simula una fase completa y retorna métricas + historial de ciclos"""
    sr, audio = cargar_audio(estimulo, duracion=duracion)
    ventana_muestras = int(sr * VENTANA_FFT_MS / 1000)
    hop_muestras = int(sr * HOP_FFT_MS / 1000)
    n_pasos = int(duracion / DT)
    
    ged_hist = []
    lf_hist = []
    w_exp_hist = []
    w_norma_hist = []
    phi_gen_vs_hist_hist = []
    phi_gen_vs_aud_hist = []
    ciclos_cerrados_hist = []
    historial_w_exp_local = []
    
    for idx in range(n_pasos):
        objetivo = preparar_objetivo_audio(audio, sr, idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
        
        ged = calcular_gradiente_espectral_diferencial(Phi_total, DIM_INTERNA, DIM_TIME)
        lf_activa = ged < UMBRAL_GED_LF
        
        (Phi_total, Phi_vel_total, W, W_exploracion, Phi_int_historia,
         Phi_generado, ciclo_cerrado, historial_w_exp_local) = actualizar_campo_con_ciclo_lf(
            Phi_total, Phi_vel_total, W, W_exploracion,
            Phi_int_historia, Phi_generado,
            objetivo, alpha,
            omega_natural, amort_natural,
            DT, entrenando=False, ged_actual=ged,
            historial_w_exp=historial_w_exp_local, DIM_TIME=DIM_TIME
        )
        
        ged_hist.append(ged)
        lf_hist.append(lf_activa)
        w_exp_hist.append(np.mean(np.abs(W_exploracion)))
        w_norma_hist.append(np.mean(np.abs(W)))
        phi_gen_vs_hist_hist.append(np.mean(np.abs(Phi_generado - Phi_int_historia[:DIM_INTERNA, :])))
        phi_gen_vs_aud_hist.append(np.mean(np.abs(Phi_generado - Phi_total[DIM_INTERNA:, :])))
        ciclos_cerrados_hist.append(1 if ciclo_cerrado else 0)
    
    return {
        'ged_mean': np.mean(ged_hist),
        'ged_std': np.std(ged_hist),
        'lf_pct': 100 * np.mean(lf_hist),
        'w_exp_mean': np.mean(w_exp_hist),
        'w_norma_mean': np.mean(w_norma_hist),
        'phi_gen_vs_hist_mean': np.mean(phi_gen_vs_hist_hist),
        'phi_gen_vs_aud_mean': np.mean(phi_gen_vs_aud_hist),
        'total_ciclos_cerrados': np.sum(ciclos_cerrados_hist),
        'ged_hist': ged_hist,
        'lf_hist': lf_hist,
        'w_exp_hist': w_exp_hist,
        'w_norma_hist': w_norma_hist,
        'ciclos_cerrados_hist': ciclos_cerrados_hist,
        'phi_total': Phi_total.copy(),
        'w': W.copy(),
        'w_exp': W_exploracion.copy(),
        'phi_int_historia': Phi_int_historia.copy(),
        'phi_generado': Phi_generado.copy()
    }

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 100)
    print("VSTCosmos v77 — El ciclo completo: salida, exploración, retorno")
    print("Implementa el ciclo evolutivo LF:")
    print("    activación → aprendizaje/generación → integración → retorno al dominio ampliado")
    print("=" * 100)

    # Inicialización
    Phi_total = inicializar_campo_total()
    Phi_vel_total = np.zeros_like(Phi_total)
    W = inicializar_plasticidad()
    W_exploracion = inicializar_w_exploracion()
    Phi_int_historia = inicializar_historia_interna()
    Phi_generado = inicializar_phi_generado()
    omega_natural, amort_natural = calcular_frecuencias_naturales(DIM_TOTAL, DIM_INTERNA)
    
    # ============================================================
    # FASE 1: ENTRENAMIENTO (voz, alpha=0.05)
    # ============================================================
    print("\n[Fase 1] Entrenamiento (voz, alpha=0.05, 30s)")
    sr_voz, audio_voz = cargar_audio("Voz_Estudio.wav", duracion=DURACION_ENTRENO)
    ventana_muestras = int(sr_voz * VENTANA_FFT_MS / 1000)
    hop_muestras = int(sr_voz * HOP_FFT_MS / 1000)
    
    historial_w_exp_dummy = []
    for idx in range(N_PASOS_ENTRENO):
        objetivo = preparar_objetivo_audio(audio_voz, sr_voz, idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
        ged = calcular_gradiente_espectral_diferencial(Phi_total, DIM_INTERNA, DIM_TIME)
        (Phi_total, Phi_vel_total, W, W_exploracion, Phi_int_historia,
         Phi_generado, _, historial_w_exp_dummy) = actualizar_campo_con_ciclo_lf(
            Phi_total, Phi_vel_total, W, W_exploracion,
            Phi_int_historia, Phi_generado,
            objetivo, alpha=0.05,
            omega_natural=omega_natural, amort_natural=amort_natural,
            dt=DT, entrenando=True, ged_actual=ged,
            historial_w_exp=historial_w_exp_dummy, DIM_TIME=DIM_TIME
        )
        # Actualizar historia interna durante entrenamiento
        region_int = Phi_total[:DIM_INTERNA, :]
        Phi_int_historia = (1 - TAU_INT_HIST) * Phi_int_historia + TAU_INT_HIST * region_int
    
    w_tras_entreno = np.mean(np.abs(W))
    print(f"  W_tras_entreno: {w_tras_entreno:.4f}")
    
    # ============================================================
    # FASES DE TEST (con ciclos LF)
    # ============================================================
    fases = [
        ("Fase 2", "Voz_Estudio.wav", 0.0, "Dominio (voz)", DURACION_FASE),
        ("Fase 3", "Brandemburgo.wav", 0.0, "No entrenado (música)", DURACION_FASE),
        ("Fase 4", "Tono puro", 0.0, "No entrenado (tono)", DURACION_FASE),
        ("Fase 5", "Voz+Viento_1.wav", 0.0, "Degradado (voz+viento)", DURACION_FASE),
        ("Fase 6", "Ruido blanco", 0.0, "Perturbación basal", DURACION_FASE),
        ("Fase 7", "Voz_Estudio.wav", 0.0, "Re-acoplamiento (voz)", DURACION_REACOPLAMIENTO)
    ]
    
    resultados = []
    n_ciclos_cerrados_total = 0
    
    for fase_id, estimulo, alpha, desc, duracion in fases:
        print(f"\n[{fase_id}] {desc} ({estimulo}, alpha={alpha})")
        res = simular_fase(
            Phi_total, Phi_vel_total, W, W_exploracion,
            Phi_int_historia, Phi_generado,
            estimulo, alpha, duracion, fase_id,
            omega_natural, amort_natural, DIM_TIME
        )
        resultados.append(res)
        
        # Actualizar estado para la siguiente fase
        Phi_total = res['phi_total']
        W = res['w']
        W_exploracion = res['w_exp']
        Phi_int_historia = res['phi_int_historia']
        Phi_generado = res['phi_generado']
        n_ciclos_cerrados_total += res['total_ciclos_cerrados']
        
        print(f"    GED medio: {res['ged_mean']:.6f}")
        print(f"    LF activa: {res['lf_pct']:.1f}%")
        print(f"    W_exp norma: {res['w_exp_mean']:.4f}")
        print(f"    W norma: {res['w_norma_mean']:.4f}")
        print(f"    Ciclos cerrados: {res['total_ciclos_cerrados']}")
    
    # ============================================================
    # MÉTRICAS DE ÉXITO DEL CICLO COMPLETO
    # ============================================================
    print("\n" + "=" * 100)
    print("DIAGNÓSTICO DEL CICLO COMPLETO")
    print("=" * 100)
    
    # Métrica 1: Ciclos cerrados
    ciclos_totales = n_ciclos_cerrados_total
    
    # Métrica 2: Dominio ampliado (comparar W norma Fase 2 vs Fase 7)
    w_norma_f2 = resultados[0]['w_norma_mean']
    w_norma_f7 = resultados[5]['w_norma_mean']
    dominio_ampliado = w_norma_f7 > w_norma_f2 * 1.2
    
    # Métrica 3: Re-acoplamiento más rápido (tiempo hasta GED > 0.008)
    ged_f2_hist = resultados[0]['ged_hist']
    ged_f7_hist = resultados[5]['ged_hist']
    
    def tiempo_primer_umbral(ged_hist, umbral=0.008):
        for i, g in enumerate(ged_hist):
            if g > umbral:
                return i * DT
        return float('inf')
    
    t_reacop_f2 = tiempo_primer_umbral(ged_f2_hist, 0.008)
    t_reacop_f7 = tiempo_primer_umbral(ged_f7_hist, 0.008)
    reacop_mas_rapido = t_reacop_f7 < t_reacop_f2
    
    lf_f7 = resultados[5]['lf_pct']
    
    print(f"\n  Métrica 1 — Ciclos LF cerrados:")
    print(f"    Total ciclos cerrados:        {ciclos_totales}")
    print(f"    {'✅' if ciclos_totales > 0 else '❌'} Ciclos ocurrieron")
    
    print(f"\n  Métrica 2 — Dominio ampliado:")
    print(f"    W norma Fase 2:               {w_norma_f2:.4f}")
    print(f"    W norma Fase 7:               {w_norma_f7:.4f}")
    print(f"    Relación:                     {w_norma_f7/w_norma_f2:.2f}")
    print(f"    Dominio ampliado (>1.2x):     {'✅' if dominio_ampliado else '❌'}")
    
    print(f"\n  Métrica 3 — Re-acoplamiento:")
    print(f"    GED > 0.008 en Fase 2:        {t_reacop_f2:.2f}s")
    print(f"    GED > 0.008 en Fase 7:        {t_reacop_f7:.2f}s")
    print(f"    Re-acoplamiento más rápido:   {'✅' if reacop_mas_rapido else '❌'}")
    print(f"    LF activa Fase 7:             {lf_f7:.1f}%")
    
    print("\n  VEREDICTO:")
    if ciclos_totales > 0 and dominio_ampliado:
        if reacop_mas_rapido:
            print(f"  ✅ CICLO COMPLETO VALIDADO")
            print(f"     El campo salió del dominio, aprendió, generó,")
            print(f"     integró y retornó con dominio ampliado.")
            print(f"     Primera evidencia computacional de exaptación funcional.")
            print(f"     Tipo de exaptación: Aprendizaje estructural bajo baja frecuencia")
            print(f"     → dominio expandido y re-acoplamiento facilitado.")
        else:
            print(f"  ⚠️  CICLO PARCIAL: dominio ampliado pero retorno no más rápido.")
            print(f"     La integración ocurrió pero no facilitó el re-acoplamiento.")
    elif ciclos_totales > 0:
        print(f"  ⚠️  CICLOS CERRADOS pero dominio no ampliado suficientemente.")
        print(f"     Aumentar GAMMA_INTEGRACION o reducir UMBRAL_SALIDA_LF.")
    else:
        print(f"  ❌ SIN CICLOS CERRADOS: criterio de salida demasiado estricto.")
        print(f"     Reducir TAU_ESTABILIZACION o UMBRAL_SALIDA_LF.")
    
    # ============================================================
    # VISUALIZACIÓN
    # ============================================================
    print("\n[Generando visualización...]")
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    
    nombres_fases = ['Voz', 'Música', 'Tono', 'Voz+Viento', 'Ruido', 'Reacop']
    
    # Gráfico 1: GED por fase
    ax = axes[0, 0]
    ged_vals = [r['ged_mean'] for r in resultados]
    ax.bar(nombres_fases, ged_vals)
    ax.axhline(y=UMBRAL_GED_LF, color='r', linestyle='--', label=f'Umbral LF = {UMBRAL_GED_LF}')
    ax.set_ylabel('GED medio')
    ax.set_title('Gradiente Espectral Diferencial')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 2: LF activa por fase
    ax = axes[0, 1]
    lf_vals = [r['lf_pct'] for r in resultados]
    ax.bar(nombres_fases, lf_vals, color=['green', 'orange', 'orange', 'orange', 'red', 'green'])
    ax.axhline(y=50, color='red', linestyle='--', label='Umbral activación')
    ax.set_ylabel('LF activa (%)')
    ax.set_title('LF activa por fase')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 3: W norma por fase
    ax = axes[0, 2]
    w_norma_vals = [r['w_norma_mean'] for r in resultados]
    ax.bar(nombres_fases, w_norma_vals)
    ax.set_ylabel('W norma')
    ax.set_title('Plasticidad principal (dominio)')
    ax.grid(True, alpha=0.3)
    
    # Gráfico 4: W_exp norma por fase
    ax = axes[0, 3]
    w_exp_vals = [r['w_exp_mean'] for r in resultados]
    ax.bar(nombres_fases, w_exp_vals)
    ax.set_ylabel('W_exploracion norma')
    ax.set_title('Aprendizaje bajo LF')
    ax.grid(True, alpha=0.3)
    
    # Gráfico 5: Phi_generado vs historia
    ax = axes[1, 0]
    gen_vs_hist = [r['phi_gen_vs_hist_mean'] for r in resultados]
    gen_vs_aud = [r['phi_gen_vs_aud_mean'] for r in resultados]
    x = np.arange(len(nombres_fases))
    ax.bar(x - 0.2, gen_vs_hist, width=0.4, label='vs historia', alpha=0.7)
    ax.bar(x + 0.2, gen_vs_aud, width=0.4, label='vs estímulo', alpha=0.7)
    ax.axhline(y=0.03, color='r', linestyle='--', label='Umbral novedad')
    ax.set_xticks(x)
    ax.set_xticklabels(nombres_fases, rotation=45)
    ax.set_ylabel('Diferencia')
    ax.set_title('Generación interna (Phi_generado)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 6: Ciclos cerrados por fase
    ax = axes[1, 1]
    ciclos_por_fase = [r['total_ciclos_cerrados'] for r in resultados]
    ax.bar(nombres_fases, ciclos_por_fase)
    ax.set_ylabel('Ciclos cerrados')
    ax.set_title('Ciclos LF completados')
    ax.grid(True, alpha=0.3)
    
    # Gráfico 7: Comparación GED Fase 2 vs Fase 7
    ax = axes[1, 2]
    # Submuestrear para no saturar
    subsample = max(1, len(ged_f2_hist) // 200)
    tiempos = np.arange(0, len(ged_f2_hist[:2000])) * DT
    ax.plot(tiempos, ged_f2_hist[:2000], label='Fase 2 (voz, primer acoplamiento)', alpha=0.7)
    ax.plot(tiempos, ged_f7_hist[:2000], label='Fase 7 (voz, re-acoplamiento)', alpha=0.7)
    ax.axhline(y=0.008, color='g', linestyle='--', label='Umbral rápido (0.008)')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('GED')
    ax.set_title('Re-acoplamiento: Fase 2 vs Fase 7')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 8: Diagrama de estado LF (inactiva/activa/cerrando)
    ax = axes[1, 3]
    # Tomar Fase 3 como ejemplo de ciclo
    lf_hist_f3 = resultados[1]['lf_hist']
    ciclos_f3 = resultados[1]['ciclos_cerrados_hist']
    tiempo_eje = np.arange(len(lf_hist_f3)) * DT
    ax.fill_between(tiempo_eje, 0, lf_hist_f3, step='mid', alpha=0.5, label='LF activa', color='orange')
    # Marcar ciclos cerrados
    for i, cerrado in enumerate(ciclos_f3):
        if cerrado:
            ax.axvline(x=i*DT, color='red', alpha=0.5, linestyle=':', linewidth=1)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Estado')
    ax.set_title('LF: activa → cerrando (líneas rojas)')
    ax.legend()
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('VSTCosmos v77 — El ciclo completo: salida, exploración, retorno', fontsize=14)
    plt.tight_layout()
    plt.savefig('v77_ciclo_completo.png', dpi=150)
    print("  Gráfico guardado: v77_ciclo_completo.png")
    
    # ============================================================
    # GUARDAR CSV
    # ============================================================
    with open('v77_ciclo_completo.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['fase', 'estimulo', 't', 'ged', 'lf_activa', 
                         'ciclo_cerrado', 'w_norma', 'w_exp_norma', 
                         'phi_gen_vs_hist', 'phi_gen_vs_aud'])
        
        for i, (fase, res) in enumerate(zip(fases, resultados)):
            ged_hist = res['ged_hist']
            lf_hist = res['lf_hist']
            ciclos_hist = res['ciclos_cerrados_hist']
            w_norma_hist = res['w_norma_hist']
            w_exp_hist = res['w_exp_hist']
            phi_hist_hist = res['phi_gen_vs_hist_mean'] * np.ones(len(ged_hist))
            phi_aud_hist = res['phi_gen_vs_aud_mean'] * np.ones(len(ged_hist))
            
            for j in range(len(ged_hist)):
                writer.writerow([
                    fase[0], fase[1], j * DT,
                    ged_hist[j], lf_hist[j], ciclos_hist[j],
                    w_norma_hist[j], w_exp_hist[j],
                    phi_hist_hist[j], phi_aud_hist[j]
                ])
    
    print("  CSV guardado: v77_ciclo_completo.csv")
    
    # ============================================================
    # GUARDAR TXT
    # ============================================================
    with open('v77_ciclo_completo.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("VSTCosmos v77 — El ciclo completo\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("MÉTRICAS DE ÉXITO\n")
        f.write("-" * 40 + "\n")
        f.write(f"Ciclos LF cerrados:        {ciclos_totales}\n")
        f.write(f"W norma Fase 2:            {w_norma_f2:.4f}\n")
        f.write(f"W norma Fase 7:            {w_norma_f7:.4f}\n")
        f.write(f"Dominio ampliado (>1.2x):  {w_norma_f7/w_norma_f2:.2f}\n")
        f.write(f"GED > 0.008 en Fase 2:     {t_reacop_f2:.2f}s\n")
        f.write(f"GED > 0.008 en Fase 7:     {t_reacop_f7:.2f}s\n")
        f.write(f"LF activa final:           {lf_f7:.1f}%\n\n")
        
        f.write("VEREDICTO\n")
        f.write("-" * 40 + "\n")
        if ciclos_totales > 0 and dominio_ampliado:
            if reacop_mas_rapido:
                f.write("✅ CICLO COMPLETO VALIDADO\n\n")
                f.write("El campo salió del dominio, aprendió bajo LF, generó\n")
                f.write("estructuras nuevas, las integró al dominio principal,\n")
                f.write("y retornó con dominio ampliado y re-acoplamiento facilitado.\n\n")
                f.write("Tipo de exaptación demostrada:\n")
                f.write("- Aprendizaje estructural en régimen de baja frecuencia\n")
                f.write("- Expansión del dominio competente por integración selectiva\n")
                f.write("- Facilitación del retorno al dominio expandido\n")
                f.write("- Ciclo evolutivo completo: apertura → exploración → integración → cierre\n")
            else:
                f.write("⚠️ CICLO PARCIAL: dominio ampliado pero retorno no más rápido.\n")
        elif ciclos_totales > 0:
            f.write("⚠️ CICLOS CERRADOS pero dominio no ampliado suficientemente.\n")
        else:
            f.write("❌ SIN CICLOS CERRADOS\n")
        f.write("=" * 60 + "\n")
    
    print("  TXT guardado: v77_ciclo_completo.txt")
    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)


if __name__ == "__main__":
    main()