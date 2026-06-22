#!/usr/bin/env python3
"""
VSTCosmos - v73: HETA (Hipótesis de Exaptación Tecnológica Autónoma)
Primera prueba computacional de O-N8.22.

El campo distingue dominio de competencia (voz) de dominio de operación (ruido),
activa LF cuando está fuera de competencia, y mantiene esa señal bajo perturbación.

Mecanismos nuevos (sobre region_int, emergentes):
1. Detector de resonancia (comparación con atractor aprendido)
2. Reserva estructural (modos propios no ocupados)
3. Respuesta diferenciada (LF) cuando resonancia baja y reserva alta
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import csv
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PARÁMETROS DEL CAMPO TOTAL
# ============================================================
DIM_INTERNA = 32
DIM_AUDITIVA = 32
DIM_TOTAL = DIM_INTERNA + DIM_AUDITIVA

DIM_TIME = 100
DT = 0.01

DURACION_ENTRENO = 30.0
DURACION_TEST_DOMINIO = 20.0
DURACION_PERTURBACION_LEVE = 20.0
DURACION_PERTURBACION_SOSTENIDA = 60.0

N_PASOS_ENTRENO = int(DURACION_ENTRENO / DT)
N_PASOS_DOMINIO = int(DURACION_TEST_DOMINIO / DT)
N_PASOS_PERT_LEVE = int(DURACION_PERTURBACION_LEVE / DT)
N_PASOS_PERT_SOST = int(DURACION_PERTURBACION_SOSTENIDA / DT)

# Parámetros de dinámica
DIFUSION_BASE = 0.15
GANANCIA_REACCION = 0.05

OMEGA_MIN = 0.05
OMEGA_MAX = 0.50
AMORT_MIN = 0.01
AMORT_MAX = 0.08
PHI_EQUILIBRIO = 0.5

# Parámetros de plasticidad hebbiana
ETA_HEBB = 0.05
TAU_W = 0.008
GAMMA_PLAST = 0.01
W_MAX = 1.0
UMBRAL_CORRELACION = 0.1

# Parámetros del mecanismo atractor
GAMMA_MEMORIA = 0.05
TAU_INT_HIST = 0.0002

# Parámetros de HETA
UMBRAL_LF = 0.3          # Umbral de LF: por debajo de este valor, el campo está fuera de competencia
UMBRAL_RESERVA = 0.1     # Reserva mínima para activar LF

# Parámetros de FFT
VENTANA_FFT_MS = 25
HOP_FFT_MS = 10
F_MIN = 80
F_MAX = 8000

LIMITE_MIN = 0.0
LIMITE_MAX = 1.0


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


def inicializar_campo_total():
    np.random.seed(42)
    return np.random.rand(DIM_TOTAL, DIM_TIME) * 0.2 + 0.4


def inicializar_plasticidad():
    return np.zeros((DIM_INTERNA, DIM_AUDITIVA))


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


def actualizar_historia_interna(Phi_int_historia, region_int, entrenando):
    if entrenando:
        return (1 - TAU_INT_HIST) * Phi_int_historia + TAU_INT_HIST * region_int
    else:
        return Phi_int_historia


def actualizar_hebb_y_plasticidad_v2(Phi_total, W, Phi_int_historia, dt, entrenando):
    region_int = Phi_total[:DIM_INTERNA, :]
    region_aud = Phi_total[DIM_INTERNA:, :]

    # Mecanismo Hebbiano
    correlacion = (region_int @ region_aud.T) / DIM_TIME
    correlacion_filtrada = np.where(
        np.abs(correlacion) > UMBRAL_CORRELACION,
        correlacion,
        0.0
    )
    dW = ETA_HEBB * correlacion_filtrada - TAU_W * W
    W_nueva = np.clip(W + dW * dt, -W_MAX, W_MAX)

    M_hebb = GAMMA_PLAST * (W_nueva @ region_aud - region_int)

    # Mecanismo Atractor
    if np.mean(np.abs(Phi_int_historia)) > 1e-6:
        M_atractor = GAMMA_MEMORIA * (Phi_int_historia - region_int)
    else:
        M_atractor = 0.0

    M_total = M_hebb + M_atractor

    Phi_int_historia_nueva = actualizar_historia_interna(Phi_int_historia, region_int, entrenando)

    return W_nueva, M_total, Phi_int_historia_nueva


def actualizar_campo_total(Phi_total, Phi_vel_total, W, Phi_int_historia,
                           objetivo_audio, alpha, omega_natural, amort_natural,
                           dt, entrenando):
    promedio_local = vecinos(Phi_total)
    difusion = DIFUSION_BASE * (promedio_local - Phi_total)
    
    desviacion = Phi_total - promedio_local
    reaccion = GANANCIA_REACCION * desviacion * (1 - desviacion**2)
    
    term_osc = (-omega_natural**2 * (Phi_total - PHI_EQUILIBRIO)
                - amort_natural * Phi_vel_total)
    
    W_nueva, M_plasticidad, Phi_int_historia_nueva = actualizar_hebb_y_plasticidad_v2(
        Phi_total, W, Phi_int_historia, dt, entrenando
    )
    
    M_campo = np.zeros_like(Phi_total)
    M_campo[:DIM_INTERNA, :] = M_plasticidad
    
    dPhi_vel = term_osc + reaccion + difusion + M_campo
    Phi_vel_nueva = Phi_vel_total + dt * dPhi_vel
    Phi_nueva = Phi_total + dt * Phi_vel_nueva
    
    if alpha > 0:
        region_auditiva_nueva = Phi_nueva[DIM_INTERNA:, :]
        region_auditiva_nueva = (1 - alpha) * region_auditiva_nueva + alpha * objetivo_audio
        Phi_nueva[DIM_INTERNA:, :] = region_auditiva_nueva
    
    return (np.clip(Phi_nueva, LIMITE_MIN, LIMITE_MAX),
            np.clip(Phi_vel_nueva, -5.0, 5.0),
            W_nueva,
            Phi_int_historia_nueva)


def calcular_gradiente(Phi_total, dim_interna):
    region_int = Phi_total[:dim_interna, :]
    region_aud = Phi_total[dim_interna:, :]
    return np.mean(np.abs(region_int - region_aud))


def calcular_acoplamiento(Phi_total, dim_interna):
    region_int = Phi_total[:dim_interna, :]
    region_aud = Phi_total[dim_interna:, :]
    return float(np.mean(region_int * region_aud))


def _perfil_espectral(region, dim, DIM_TIME):
    """Calcula el perfil espectral de una región (interna o historia)"""
    perfil = np.zeros(DIM_TIME // 2)
    for banda in range(dim):
        serie = region[banda, :]
        serie = serie - np.mean(serie)
        fft = np.fft.rfft(serie)
        potencia = np.abs(fft) ** 2
        perfil += potencia[:DIM_TIME // 2]
    return perfil / dim


def calcular_resonancia_actual(Phi_total, Phi_int_historia, dim_interna, DIM_TIME):
    """
    Compara el perfil espectral actual de region_int con el perfil
    del atractor aprendido (Phi_int_historia).
    
    Alta resonancia: el estímulo actual activa modos compatibles.
    Baja resonancia: el estímulo actual no encuentra modos compatibles.
    """
    region_int = Phi_total[:dim_interna, :]
    historia = Phi_int_historia
    
    perfil_actual = _perfil_espectral(region_int, dim_interna, DIM_TIME)
    perfil_historia = _perfil_espectral(historia, dim_interna, DIM_TIME)
    
    # Similitud coseno entre perfiles
    dot = np.dot(perfil_actual, perfil_historia)
    norma = np.linalg.norm(perfil_actual) * np.linalg.norm(perfil_historia) + 1e-10
    resonancia = float(dot / norma)
    
    return resonancia


def calcular_reserva(Phi_total, dim_interna, resonancia_actual, DIM_TIME):
    """
    Reserva = modos propios del campo no ocupados por el acoplamiento actual.
    """
    region_int = Phi_total[:dim_interna, :]
    perfil = _perfil_espectral(region_int, dim_interna, DIM_TIME)
    
    # Modos activos: por encima del umbral
    umbral_activacion = np.mean(perfil) * 1.5
    modos_activos = int(np.sum(perfil > umbral_activacion))
    modos_totales = DIM_TIME // 2
    
    # Reserva = fracción de modos no utilizados
    reserva = 1.0 - (modos_activos / modos_totales)
    
    # Reserva efectiva ponderada por inverso de resonancia
    reserva_efectiva = reserva * (1.0 - resonancia_actual)
    
    return float(reserva_efectiva), modos_activos


def calcular_respuesta_lf(resonancia, reserva_efectiva, historial_resonancia, dt):
    """
    El campo genera una respuesta diferenciada cuando:
    - La resonancia cae bajo el umbral (fuera de dominio de competencia)
    - La reserva es suficiente (hay modos disponibles)
    """
    fuera_de_competencia = resonancia < UMBRAL_LF
    reserva_suficiente = reserva_efectiva > UMBRAL_RESERVA
    
    lf_activa = fuera_de_competencia and reserva_suficiente
    
    # Estabilidad: LF se considera estable si se mantiene por 2 segundos
    if len(historial_resonancia) >= int(2.0 / dt):
        resonancia_media = np.mean(historial_resonancia[-int(2.0 / dt):])
        lf_estable = resonancia_media < UMBRAL_LF
    else:
        lf_estable = False
    
    return {
        'lf_activa': lf_activa,
        'lf_estable': lf_estable,
        'fuera_de_competencia': fuera_de_competencia,
        'reserva_suficiente': reserva_suficiente
    }


def calcular_gradiente_espectral_diferencial(Phi_total, dim_interna, DIM_TIME):
    region_int = Phi_total[:dim_interna, :]
    region_aud = Phi_total[dim_interna:, :]
    
    perfil_int = _perfil_espectral(region_int, dim_interna, DIM_TIME)
    perfil_aud = _perfil_espectral(region_aud, dim_interna, DIM_TIME)
    
    diferencia_espectral = np.abs(perfil_int - perfil_aud)
    gradiente_espectral_medio = float(np.mean(diferencia_espectral))
    
    return gradiente_espectral_medio


def calcular_perfil_espectral_modos(Phi_total, dim_interna, DIM_TIME):
    region_int = Phi_total[:dim_interna, :]
    perfil = _perfil_espectral(region_int, dim_interna, DIM_TIME)
    frecuencia_dominante = int(np.argmax(perfil))
    umbral_riqueza = np.mean(perfil)
    riqueza_modal = int(np.sum(perfil > umbral_riqueza))
    return frecuencia_dominante, riqueza_modal


def simular_heta():
    print("=" * 100)
    print("VSTCosmos - v73: HETA (Hipótesis de Exaptación Tecnológica Autónoma)")
    print("Fase 1: Entrenamiento (voz, alpha=0.05, 30s)")
    print("Fase 2: Test en dominio (voz, alpha=0.0, 20s)")
    print("Fase 3: Perturbación leve (ruido, alpha=0.0, 20s)")
    print("Fase 4: Perturbación sostenida (ruido, alpha=0.0, 60s)")
    print("=" * 100)

    # Inicialización
    Phi_total = inicializar_campo_total()
    Phi_vel_total = np.zeros_like(Phi_total)
    W = inicializar_plasticidad()
    Phi_int_historia = inicializar_historia_interna()
    omega_natural, amort_natural = calcular_frecuencias_naturales(DIM_TOTAL, DIM_INTERNA)
    
    # Cargar audios
    sr_voz, audio_voz = cargar_audio("Voz_Estudio.wav", duracion=DURACION_ENTRENO + DURACION_TEST_DOMINIO)
    sr_ruido, audio_ruido = cargar_audio("Ruido blanco", duracion=DURACION_PERTURBACION_LEVE + DURACION_PERTURBACION_SOSTENIDA)
    
    ventana_muestras = int(sr_voz * VENTANA_FFT_MS / 1000)
    hop_muestras = int(sr_voz * HOP_FFT_MS / 1000)
    
    historial_resonancia = []
    historial_reserva = []
    historial_lf_activa = []
    historial_lf_estable = []
    historial_grad_esp = []
    
    # ============================================================
    # FASE 1: ENTRENAMIENTO (voz, alpha=0.05)
    # ============================================================
    print("\n[Fase 1] Entrenamiento (voz, alpha=0.05)")
    for idx in range(N_PASOS_ENTRENO):
        objetivo = preparar_objetivo_audio(audio_voz, sr_voz, idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
        Phi_total, Phi_vel_total, W, Phi_int_historia = actualizar_campo_total(
            Phi_total, Phi_vel_total, W, Phi_int_historia,
            objetivo, alpha=0.05, omega_natural=omega_natural, 
            amort_natural=amort_natural, dt=DT, entrenando=True
        )
    
    w_tras_entreno = np.mean(np.abs(W))
    norma_historia = np.mean(np.abs(Phi_int_historia))
    print(f"  W_tras_entreno: {w_tras_entreno:.4f}")
    print(f"  Phi_int_historia: norma {norma_historia:.6f}")
    
    # ============================================================
    # FASE 2: TEST EN DOMINIO (voz, alpha=0.0)
    # ============================================================
    print("\n[Fase 2] Test en dominio (voz, alpha=0.0)")
    
    resonancia_f2 = []
    reserva_f2 = []
    lf_activa_f2 = []
    lf_estable_f2 = []
    
    for idx in range(N_PASOS_ENTRENO, N_PASOS_ENTRENO + N_PASOS_DOMINIO):
        objetivo = preparar_objetivo_audio(audio_voz, sr_voz, idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
        Phi_total, Phi_vel_total, W, Phi_int_historia = actualizar_campo_total(
            Phi_total, Phi_vel_total, W, Phi_int_historia,
            objetivo, alpha=0.0, omega_natural=omega_natural, 
            amort_natural=amort_natural, dt=DT, entrenando=False
        )
        
        resonancia = calcular_resonancia_actual(Phi_total, Phi_int_historia, DIM_INTERNA, DIM_TIME)
        reserva_ef, modos_activos = calcular_reserva(Phi_total, DIM_INTERNA, resonancia, DIM_TIME)
        lf = calcular_respuesta_lf(resonancia, reserva_ef, historial_resonancia, DT)
        
        resonancia_f2.append(resonancia)
        reserva_f2.append(reserva_ef)
        lf_activa_f2.append(lf['lf_activa'])
        lf_estable_f2.append(lf['lf_estable'])
        
        historial_resonancia.append(resonancia)
        historial_reserva.append(reserva_ef)
        historial_lf_activa.append(lf['lf_activa'])
        historial_lf_estable.append(lf['lf_estable'])
        
        if idx % 100 == 0:
            print(f"    t={idx*DT:.1f}s | resonancia={resonancia:.3f} | reserva={reserva_ef:.3f} | modos={modos_activos} | LF={'ACTIVA' if lf['lf_activa'] else 'inactiva'}")
    
    grad_esp_f2 = calcular_gradiente_espectral_diferencial(Phi_total, DIM_INTERNA, DIM_TIME)
    freq_dom_f2, riq_f2 = calcular_perfil_espectral_modos(Phi_total, DIM_INTERNA, DIM_TIME)
    historial_grad_esp.append(('F2', grad_esp_f2))
    
    print(f"\n  Resumen Fase 2:")
    print(f"    Resonancia media:          {np.mean(resonancia_f2):.4f}")
    print(f"    Resonancia mínima:         {np.min(resonancia_f2):.4f}")
    print(f"    Reserva media:             {np.mean(reserva_f2):.4f}")
    print(f"    LF activa (% del tiempo):  {100*np.mean(lf_activa_f2):.1f}%")
    print(f"    LF estable (% del tiempo): {100*np.mean(lf_estable_f2):.1f}%")
    print(f"    Gradiente espectral diff:  {grad_esp_f2:.6f}")
    print(f"    Frecuencia dominante:      modo {freq_dom_f2}")
    print(f"    Riqueza modal:             {riq_f2} modos")
    
    # ============================================================
    # FASE 3: PERTURBACIÓN LEVE (ruido, alpha=0.0, 20s)
    # ============================================================
    print("\n[Fase 3] Perturbación leve (ruido, alpha=0.0, 20s)")
    
    resonancia_f3 = []
    reserva_f3 = []
    lf_activa_f3 = []
    lf_estable_f3 = []
    
    for idx in range(N_PASOS_PERT_LEVE):
        objetivo = preparar_objetivo_audio(audio_ruido, sr_ruido, idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
        Phi_total, Phi_vel_total, W, Phi_int_historia = actualizar_campo_total(
            Phi_total, Phi_vel_total, W, Phi_int_historia,
            objetivo, alpha=0.0, omega_natural=omega_natural, 
            amort_natural=amort_natural, dt=DT, entrenando=False
        )
        
        resonancia = calcular_resonancia_actual(Phi_total, Phi_int_historia, DIM_INTERNA, DIM_TIME)
        reserva_ef, modos_activos = calcular_reserva(Phi_total, DIM_INTERNA, resonancia, DIM_TIME)
        lf = calcular_respuesta_lf(resonancia, reserva_ef, historial_resonancia, DT)
        
        resonancia_f3.append(resonancia)
        reserva_f3.append(reserva_ef)
        lf_activa_f3.append(lf['lf_activa'])
        lf_estable_f3.append(lf['lf_estable'])
        
        historial_resonancia.append(resonancia)
        historial_reserva.append(reserva_ef)
        historial_lf_activa.append(lf['lf_activa'])
        historial_lf_estable.append(lf['lf_estable'])
        
        if idx % 100 == 0:
            print(f"    t={idx*DT:.1f}s | resonancia={resonancia:.3f} | reserva={reserva_ef:.3f} | modos={modos_activos} | LF={'ACTIVA' if lf['lf_activa'] else 'inactiva'}")
    
    grad_esp_f3 = calcular_gradiente_espectral_diferencial(Phi_total, DIM_INTERNA, DIM_TIME)
    freq_dom_f3, riq_f3 = calcular_perfil_espectral_modos(Phi_total, DIM_INTERNA, DIM_TIME)
    historial_grad_esp.append(('F3', grad_esp_f3))
    
    print(f"\n  Resumen Fase 3:")
    print(f"    Resonancia media:          {np.mean(resonancia_f3):.4f}")
    print(f"    Resonancia mínima:         {np.min(resonancia_f3):.4f}")
    print(f"    Reserva media:             {np.mean(reserva_f3):.4f}")
    print(f"    LF activa (% del tiempo):  {100*np.mean(lf_activa_f3):.1f}%")
    print(f"    LF estable (% del tiempo): {100*np.mean(lf_estable_f3):.1f}%")
    print(f"    Gradiente espectral diff:  {grad_esp_f3:.6f}")
    print(f"    Frecuencia dominante:      modo {freq_dom_f3}")
    print(f"    Riqueza modal:             {riq_f3} modos")
    
    # ============================================================
    # FASE 4: PERTURBACIÓN SOSTENIDA (ruido, alpha=0.0, 60s)
    # ============================================================
    print("\n[Fase 4] Perturbación sostenida (ruido, alpha=0.0, 60s)")
    
    resonancia_f4 = []
    reserva_f4 = []
    lf_activa_f4 = []
    lf_estable_f4 = []
    
    for idx in range(N_PASOS_PERT_SOST):
        objetivo = preparar_objetivo_audio(audio_ruido, sr_ruido, idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
        Phi_total, Phi_vel_total, W, Phi_int_historia = actualizar_campo_total(
            Phi_total, Phi_vel_total, W, Phi_int_historia,
            objetivo, alpha=0.0, omega_natural=omega_natural, 
            amort_natural=amort_natural, dt=DT, entrenando=False
        )
        
        resonancia = calcular_resonancia_actual(Phi_total, Phi_int_historia, DIM_INTERNA, DIM_TIME)
        reserva_ef, modos_activos = calcular_reserva(Phi_total, DIM_INTERNA, resonancia, DIM_TIME)
        lf = calcular_respuesta_lf(resonancia, reserva_ef, historial_resonancia[-len(historial_resonancia):], DT)
        
        resonancia_f4.append(resonancia)
        reserva_f4.append(reserva_ef)
        lf_activa_f4.append(lf['lf_activa'])
        lf_estable_f4.append(lf['lf_estable'])
        
        historial_resonancia.append(resonancia)
        historial_reserva.append(reserva_ef)
        historial_lf_activa.append(lf['lf_activa'])
        historial_lf_estable.append(lf['lf_estable'])
        
        if idx % 200 == 0:
            print(f"    t={idx*DT:.1f}s | resonancia={resonancia:.3f} | reserva={reserva_ef:.3f} | modos={modos_activos} | LF={'ACTIVA' if lf['lf_activa'] else 'inactiva'}")
    
    grad_esp_f4 = calcular_gradiente_espectral_diferencial(Phi_total, DIM_INTERNA, DIM_TIME)
    freq_dom_f4, riq_f4 = calcular_perfil_espectral_modos(Phi_total, DIM_INTERNA, DIM_TIME)
    historial_grad_esp.append(('F4', grad_esp_f4))
    
    print(f"\n  Resumen Fase 4:")
    print(f"    Resonancia media:          {np.mean(resonancia_f4):.4f}")
    print(f"    Resonancia mínima:         {np.min(resonancia_f4):.4f}")
    print(f"    Reserva media:             {np.mean(reserva_f4):.4f}")
    print(f"    LF activa (% del tiempo):  {100*np.mean(lf_activa_f4):.1f}%")
    print(f"    LF estable (% del tiempo): {100*np.mean(lf_estable_f4):.1f}%")
    print(f"    Gradiente espectral diff:  {grad_esp_f4:.6f}")
    print(f"    Frecuencia dominante:      modo {freq_dom_f4}")
    print(f"    Riqueza modal:             {riq_f4} modos")
    
    # ============================================================
    # CRITERIOS DE HETA
    # ============================================================
    print("\n" + "=" * 100)
    print("CRITERIOS DE HETA")
    print("=" * 100)
    
    # Criterio 1: Caída de resonancia bajo perturbación
    resonancia_f2_media = np.mean(resonancia_f2)
    resonancia_f3_media = np.mean(resonancia_f3)
    caida_resonancia = resonancia_f3_media / max(resonancia_f2_media, 0.001)
    criterio_1 = caida_resonancia < 0.5
    
    # Criterio 2: Activación de LF en Fase 3
    lf_activa_f3_pct = 100 * np.mean(lf_activa_f3)
    criterio_2 = lf_activa_f3_pct > 50
    
    # Criterio 3: Persistencia de LF bajo perturbación sostenida
    # Buscar secuencia continua de LF estable > 30s
    racha_max = 0
    racha_actual = 0
    for val in lf_estable_f4:
        if val:
            racha_actual += 1
        else:
            racha_actual = 0
        racha_max = max(racha_max, racha_actual)
    tiempo_lf_estable = racha_max * DT
    criterio_3 = tiempo_lf_estable > 30.0
    
    # Criterio 4: Reserva disponible bajo perturbación
    reserva_f2_media = np.mean(reserva_f2)
    reserva_f3_media = np.mean(reserva_f3)
    incremento_reserva = reserva_f3_media / max(reserva_f2_media, 0.001)
    criterio_4 = incremento_reserva > 2.0
    
    print(f"\n  Criterio 1 (Caída de resonancia < 0.5×):")
    print(f"    Resonancia F2 media: {resonancia_f2_media:.4f}")
    print(f"    Resonancia F3 media: {resonancia_f3_media:.4f}")
    print(f"    Relación: {caida_resonancia:.3f} -> {'✅' if criterio_1 else '❌'}")
    
    print(f"\n  Criterio 2 (LF activa > 50% en Fase 3):")
    print(f"    LF activa F3: {lf_activa_f3_pct:.1f}% -> {'✅' if criterio_2 else '❌'}")
    
    print(f"\n  Criterio 3 (LF estable > 30s en Fase 4):")
    print(f"    Racha máxima LF estable: {tiempo_lf_estable:.1f}s -> {'✅' if criterio_3 else '❌'}")
    
    print(f"\n  Criterio 4 (Reserva F3 > 2× F2):")
    print(f"    Reserva F2 media: {reserva_f2_media:.4f}")
    print(f"    Reserva F3 media: {reserva_f3_media:.4f}")
    print(f"    Relación: {incremento_reserva:.2f} -> {'✅' if criterio_4 else '❌'}")
    
    print("\n" + "=" * 100)
    print("CONCLUSIÓN HETA")
    print("=" * 100)
    
    if criterio_1 and criterio_2 and criterio_3 and criterio_4:
        print("\n  ✅ HETA VALIDADO")
        print("     El campo distingue dominio de competencia de dominio de operación,")
        print("     activa LF cuando está fuera de competencia,")
        print("     y mantiene esa señal bajo perturbación sostenida.")
        print("     → Primera evidencia computacional de HETA (O-N8.22).")
    elif criterio_1 and criterio_2:
        print("\n  ✅ HETA PARCIAL (LF reactiva)")
        print("     El campo distingue dominios y activa LF,")
        print("     pero la señal no persiste bajo perturbación sostenida.")
        print("     → La reserva se agota o el detector de resonancia se adapta.")
    elif criterio_1:
        print("\n  ⚠️ HETA INCOMPLETO")
        print("     El campo distingue dominios,")
        print("     pero LF no se activa consistentemente.")
        print("     → UMBRAL_LF demasiado alto o reserva insuficiente.")
    else:
        print("\n  ❌ HETA NO VALIDADO")
        print("     El detector de resonancia no distingue voz de ruido.")
        print("     → Revisar la función de similitud coseno.")
    
    # ============================================================
    # VISUALIZACIÓN
    # ============================================================
    print("\n[Generando visualización...]")
    
    # Construir series temporales completas
    tiempos_f1 = np.arange(0, DURACION_ENTRENO, DT)
    tiempos_f2 = np.arange(DURACION_ENTRENO, DURACION_ENTRENO + DURACION_TEST_DOMINIO, DT)
    tiempos_f3 = np.arange(DURACION_ENTRENO + DURACION_TEST_DOMINIO, 
                          DURACION_ENTRENO + DURACION_TEST_DOMINIO + DURACION_PERTURBACION_LEVE, DT)
    tiempos_f4 = np.arange(DURACION_ENTRENO + DURACION_TEST_DOMINIO + DURACION_PERTURBACION_LEVE,
                          DURACION_ENTRENO + DURACION_TEST_DOMINIO + DURACION_PERTURBACION_LEVE + DURACION_PERTURBACION_SOSTENIDA, DT)
    
    # Solo graficamos F2-F4 para claridad
    tiempos = np.concatenate([tiempos_f2, tiempos_f3, tiempos_f4])
    resonancias = np.concatenate([resonancia_f2, resonancia_f3, resonancia_f4])
    reservas = np.concatenate([reserva_f2, reserva_f3, reserva_f4])
    lf_activas = np.concatenate([lf_activa_f2, lf_activa_f3, lf_activa_f4])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Gráfico 1: Resonancia
    ax = axes[0, 0]
    ax.plot(tiempos, resonancias)
    ax.axhline(y=UMBRAL_LF, color='r', linestyle='--', label=f'Umbral LF = {UMBRAL_LF}')
    ax.axvline(x=DURACION_ENTRENO + DURACION_TEST_DOMINIO, color='gray', linestyle=':', alpha=0.7)
    ax.axvline(x=DURACION_ENTRENO + DURACION_TEST_DOMINIO + DURACION_PERTURBACION_LEVE, color='gray', linestyle=':', alpha=0.7)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Resonancia (similitud coseno)')
    ax.set_title('Resonancia con atractor aprendido')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 2: Reserva estructural
    ax = axes[0, 1]
    ax.plot(tiempos, reservas)
    ax.axhline(y=UMBRAL_RESERVA, color='g', linestyle='--', label=f'Umbral reserva = {UMBRAL_RESERVA}')
    ax.axvline(x=DURACION_ENTRENO + DURACION_TEST_DOMINIO, color='gray', linestyle=':', alpha=0.7)
    ax.axvline(x=DURACION_ENTRENO + DURACION_TEST_DOMINIO + DURACION_PERTURBACION_LEVE, color='gray', linestyle=':', alpha=0.7)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Reserva efectiva')
    ax.set_title('Reserva estructural (modos no utilizados)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 3: LF Activa (binario)
    ax = axes[1, 0]
    ax.fill_between(tiempos, 0, lf_activas, step='mid', alpha=0.7, color='red')
    ax.axvline(x=DURACION_ENTRENO + DURACION_TEST_DOMINIO, color='gray', linestyle=':', alpha=0.7)
    ax.axvline(x=DURACION_ENTRENO + DURACION_TEST_DOMINIO + DURACION_PERTURBACION_LEVE, color='gray', linestyle=':', alpha=0.7)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('LF Activa')
    ax.set_title('Respuesta diferenciada (LF)')
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    
    # Gráfico 4: Gradiente espectral diferencial por fase
    ax = axes[1, 1]
    fases = ['F2\n(voz)', 'F3\n(ruido leve)', 'F4\n(ruido sost.)']
    valores = [grad_esp_f2, grad_esp_f3, grad_esp_f4]
    ax.bar(fases, valores, color=['green', 'orange', 'red'])
    ax.set_ylabel('Gradiente espectral diferencial')
    ax.set_title('Diferenciación espectral entre regiones')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('VSTCosmos v73 - HETA: Hipótesis de Exaptación Tecnológica Autónoma', fontsize=14)
    plt.tight_layout()
    plt.savefig('v73_heta_resultados.png', dpi=150)
    print("  Gráfico guardado: v73_heta_resultados.png")
    
    # Guardar CSV
    with open('v73_heta_resultado.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['fase', 't', 'resonancia', 'reserva_efectiva', 'lf_activa', 'lf_estable'])
        
        t_acum = 0
        for t_val, r_val, res_val, lf_a, lf_e in zip(tiempos_f2, resonancia_f2, reserva_f2, lf_activa_f2, lf_estable_f2):
            writer.writerow(['F2', t_val, r_val, res_val, int(lf_a), int(lf_e)])
        for t_val, r_val, res_val, lf_a, lf_e in zip(tiempos_f3, resonancia_f3, reserva_f3, lf_activa_f3, lf_estable_f3):
            writer.writerow(['F3', t_val, r_val, res_val, int(lf_a), int(lf_e)])
        for t_val, r_val, res_val, lf_a, lf_e in zip(tiempos_f4, resonancia_f4, reserva_f4, lf_activa_f4, lf_estable_f4):
            writer.writerow(['F4', t_val, r_val, res_val, int(lf_a), int(lf_e)])
    
    print("  CSV guardado: v73_heta_resultado.csv")
    
    # Guardar TXT
    with open('v73_heta_resultado.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("VSTCosmos v73 - HETA: Resultado\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Criterio 1 (Caída resonancia < 0.5×): {caida_resonancia:.3f} -> {'PASA' if criterio_1 else 'FALLA'}\n")
        f.write(f"  Resonancia F2: {resonancia_f2_media:.4f}, F3: {resonancia_f3_media:.4f}\n")
        f.write(f"Criterio 2 (LF activa > 50% en F3): {lf_activa_f3_pct:.1f}% -> {'PASA' if criterio_2 else 'FALLA'}\n")
        f.write(f"Criterio 3 (LF estable > 30s): {tiempo_lf_estable:.1f}s -> {'PASA' if criterio_3 else 'FALLA'}\n")
        f.write(f"Criterio 4 (Reserva F3 > 2× F2): {incremento_reserva:.2f} -> {'PASA' if criterio_4 else 'FALLA'}\n")
        f.write(f"  Reserva F2: {reserva_f2_media:.4f}, F3: {reserva_f3_media:.4f}\n\n")
        f.write("=" * 60 + "\n")
        if criterio_1 and criterio_2 and criterio_3 and criterio_4:
            f.write("CONCLUSION: HETA VALIDADO\n")
            f.write("Primera evidencia computacional de HETA (O-N8.22).\n")
        elif criterio_1 and criterio_2:
            f.write("CONCLUSION: HETA PARCIAL - LF reactiva\n")
        elif criterio_1:
            f.write("CONCLUSION: HETA INCOMPLETO\n")
        else:
            f.write("CONCLUSION: HETA NO VALIDADO\n")
        f.write("=" * 60 + "\n")
    
    print("  TXT guardado: v73_heta_resultado.txt")
    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)


if __name__ == "__main__":
    simular_heta()