#!/usr/bin/env python3
"""
VSTCosmos - v74: HETA Redefinido
Tres hipótesis en un experimento:
A) Detección de señal (res_v2 discrimina voz vs ruido)
B) Detección de régimen (LF activa por colapso sostenido)
C) Disponibilidad para nuevo acoplamiento (re-acoplamiento más rápido desde LF)
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import csv
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PARÁMETROS DEL CAMPO TOTAL (basados en v73)
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
DURACION_REACOPLAMIENTO = 20.0

N_PASOS_ENTRENO = int(DURACION_ENTRENO / DT)
N_PASOS_DOMINIO = int(DURACION_TEST_DOMINIO / DT)
N_PASOS_PERT_LEVE = int(DURACION_PERTURBACION_LEVE / DT)
N_PASOS_PERT_SOST = int(DURACION_PERTURBACION_SOSTENIDA / DT)
N_PASOS_REACOPLAMIENTO = int(DURACION_REACOPLAMIENTO / DT)

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

# Parámetros del mecanismo atractor (aumentados)
GAMMA_MEMORIA = 0.08
TAU_INT_HIST = 0.0002

# Parámetros de HETA
UMBRAL_LF = 0.3
UMBRAL_RESERVA = 0.1

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


def actualizar_hebb_y_plasticidad(Phi_total, W, Phi_int_historia, dt, entrenando):
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
    
    W_nueva, M_plasticidad, Phi_int_historia_nueva = actualizar_hebb_y_plasticidad(
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
    
    # Prevenir colapso: si varianza demasiado baja, agregar ruido mínimo
    varianza_campo = np.var(Phi_nueva[:DIM_INTERNA, :])
    if varianza_campo < 1e-6:
        ruido_minimo = np.random.normal(0, 0.01, Phi_nueva[:DIM_INTERNA, :].shape)
        Phi_nueva[:DIM_INTERNA, :] += ruido_minimo
    
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
    perfil = np.zeros(DIM_TIME // 2)
    for banda in range(dim):
        serie = region[banda, :]
        serie = serie - np.mean(serie)
        fft = np.fft.rfft(serie)
        potencia = np.abs(fft) ** 2
        perfil += potencia[:DIM_TIME // 2]
    return perfil / dim


def calcular_resonancia_original(Phi_total, Phi_int_historia, dim_interna, DIM_TIME):
    """Resonancia original: region_int vs atractor"""
    region_int = Phi_total[:dim_interna, :]
    historia = Phi_int_historia
    
    perfil_int = _perfil_espectral(region_int, dim_interna, DIM_TIME)
    perfil_hist = _perfil_espectral(historia, dim_interna, DIM_TIME)
    
    dot = np.dot(perfil_int, perfil_hist)
    norma = np.linalg.norm(perfil_int) * np.linalg.norm(perfil_hist) + 1e-10
    return float(dot / norma)


def calcular_resonancia_v2(Phi_total, Phi_int_historia, dim_interna, DIM_TIME):
    """
    Versión A: compara region_AUD con el atractor aprendido.
    La pregunta es: ¿el entorno actual tiene estructura compatible
    con los modos que aprendí?
    """
    region_aud = Phi_total[dim_interna:, :]
    historia = Phi_int_historia
    
    perfil_aud = _perfil_espectral(region_aud, dim_interna, DIM_TIME)
    perfil_hist = _perfil_espectral(historia, dim_interna, DIM_TIME)
    
    # Selector de modos coherentes: ignorar modos de baja energía
    perfil_norm = perfil_hist / (np.max(perfil_hist) + 1e-10)
    mascara = perfil_norm > 0.3
    if np.sum(mascara) == 0:
        return 0.0
    
    perfil_aud_sel = perfil_aud * mascara
    perfil_hist_sel = perfil_hist * mascara
    
    dot = np.dot(perfil_aud_sel, perfil_hist_sel)
    norma = (np.linalg.norm(perfil_aud_sel) * np.linalg.norm(perfil_hist_sel) + 1e-10)
    return float(dot / norma)


def calcular_reserva(Phi_total, dim_interna, resonancia_actual, DIM_TIME):
    region_int = Phi_total[:dim_interna, :]
    perfil = _perfil_espectral(region_int, dim_interna, DIM_TIME)
    
    umbral_activacion = np.mean(perfil) * 1.5
    modos_activos = int(np.sum(perfil > umbral_activacion))
    modos_totales = DIM_TIME // 2
    
    reserva = 1.0 - (modos_activos / modos_totales)
    reserva_efectiva = reserva * (1.0 - resonancia_actual)
    
    return float(reserva_efectiva), modos_activos


def calcular_respuesta_lf(resonancia, reserva_efectiva, historial_resonancia, dt):
    fuera_de_competencia = resonancia < UMBRAL_LF
    reserva_suficiente = reserva_efectiva > UMBRAL_RESERVA
    
    lf_activa = fuera_de_competencia and reserva_suficiente
    
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


def calcular_tiempo_reacoplamiento(Phi_total, Phi_vel_total, W, Phi_int_historia,
                                    omega_natural, amort_natural, duracion, DIM_TIME):
    """Mide tiempo para alcanzar resonancia_v2 > 0.7 desde inicio de fase"""
    sr, audio = cargar_audio("Voz_Estudio.wav", duracion=duracion)
    ventana_muestras = int(sr * VENTANA_FFT_MS / 1000)
    hop_muestras = int(sr * HOP_FFT_MS / 1000)
    n_pasos = int(duracion / DT)
    
    for idx in range(n_pasos):
        objetivo = preparar_objetivo_audio(audio, sr, idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
        Phi_total, Phi_vel_total, W, Phi_int_historia = actualizar_campo_total(
            Phi_total, Phi_vel_total, W, Phi_int_historia,
            objetivo, alpha=0.0, omega_natural=omega_natural,
            amort_natural=amort_natural, dt=DT, entrenando=False
        )
        
        res_v2 = calcular_resonancia_v2(Phi_total, Phi_int_historia, DIM_INTERNA, DIM_TIME)
        
        if res_v2 > 0.7:
            return idx * DT, Phi_total, Phi_vel_total, W, Phi_int_historia
    
    return None, Phi_total, Phi_vel_total, W, Phi_int_historia


def simular_experimento_control_basal():
    """Corrida control sin entrenamiento ni colapso: mide tiempo basal de reacoplamiento"""
    print("\n[Experimento Control] Tiempo de reacoplamiento basal")
    
    np.random.seed(999)
    Phi_total = inicializar_campo_total()
    Phi_vel_total = np.zeros_like(Phi_total)
    W = inicializar_plasticidad()
    Phi_int_historia = inicializar_historia_interna()
    omega_natural, amort_natural = calcular_frecuencias_naturales(DIM_TOTAL, DIM_INTERNA)
    
    tiempo, _, _, _, _ = calcular_tiempo_reacoplamiento(
        Phi_total, Phi_vel_total, W, Phi_int_historia,
        omega_natural, amort_natural, DURACION_REACOPLAMIENTO, DIM_TIME
    )
    
    print(f"  Tiempo basal hasta resonancia > 0.7: {tiempo:.2f}s" if tiempo else "  No alcanzó resonancia > 0.7")
    return tiempo


def main():
    print("=" * 100)
    print("VSTCosmos - v74: HETA Redefinido")
    print("Tres hipótesis:")
    print("  A) Detección de señal (res_v2 discrimina voz vs ruido)")
    print("  B) Detección de régimen (LF activa por colapso sostenido)")
    print("  C) Disponibilidad para nuevo acoplamiento (re-acoplamiento más rápido desde LF)")
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
    
    # Historiales
    historial_res_orig = []
    historial_res_v2 = []
    historial_reserva = []
    historial_lf_activa = []
    historial_grad_esp = []
    
    # ============================================================
    # FASE 1: ENTRENAMIENTO
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
    # FASE 2: TEST EN DOMINIO
    # ============================================================
    print("\n[Fase 2] Test en dominio (voz, alpha=0.0)")
    
    res_orig_f2 = []
    res_v2_f2 = []
    reserva_f2 = []
    lf_f2 = []
    
    for idx in range(N_PASOS_DOMINIO):
        objetivo = preparar_objetivo_audio(audio_voz, sr_voz, N_PASOS_ENTRENO + idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
        Phi_total, Phi_vel_total, W, Phi_int_historia = actualizar_campo_total(
            Phi_total, Phi_vel_total, W, Phi_int_historia,
            objetivo, alpha=0.0, omega_natural=omega_natural,
            amort_natural=amort_natural, dt=DT, entrenando=False
        )
        
        res_orig = calcular_resonancia_original(Phi_total, Phi_int_historia, DIM_INTERNA, DIM_TIME)
        res_v2 = calcular_resonancia_v2(Phi_total, Phi_int_historia, DIM_INTERNA, DIM_TIME)
        reserva_ef, modos = calcular_reserva(Phi_total, DIM_INTERNA, res_v2, DIM_TIME)
        lf = calcular_respuesta_lf(res_v2, reserva_ef, historial_res_v2, DT)
        
        res_orig_f2.append(res_orig)
        res_v2_f2.append(res_v2)
        reserva_f2.append(reserva_ef)
        lf_f2.append(lf['lf_activa'])
        
        historial_res_orig.append(res_orig)
        historial_res_v2.append(res_v2)
        historial_reserva.append(reserva_ef)
        historial_lf_activa.append(lf['lf_activa'])
    
    grad_esp_f2 = calcular_gradiente_espectral_diferencial(Phi_total, DIM_INTERNA, DIM_TIME)
    historial_grad_esp.append(('F2', grad_esp_f2))
    
    print(f"  Resumen Fase 2:")
    print(f"    Resonancia orig (media):   {np.mean(res_orig_f2):.4f}")
    print(f"    Resonancia v2 (media):     {np.mean(res_v2_f2):.4f}")
    print(f"    Reserva media:             {np.mean(reserva_f2):.4f}")
    print(f"    LF activa (%):             {100*np.mean(lf_f2):.1f}%")
    print(f"    Gradiente espectral diff:  {grad_esp_f2:.6f}")
    
    # ============================================================
    # FASE 3: PERTURBACIÓN LEVE
    # ============================================================
    print("\n[Fase 3] Perturbación leve (ruido, alpha=0.0)")
    
    res_orig_f3 = []
    res_v2_f3 = []
    reserva_f3 = []
    lf_f3 = []
    
    for idx in range(N_PASOS_PERT_LEVE):
        objetivo = preparar_objetivo_audio(audio_ruido, sr_ruido, idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
        Phi_total, Phi_vel_total, W, Phi_int_historia = actualizar_campo_total(
            Phi_total, Phi_vel_total, W, Phi_int_historia,
            objetivo, alpha=0.0, omega_natural=omega_natural,
            amort_natural=amort_natural, dt=DT, entrenando=False
        )
        
        res_orig = calcular_resonancia_original(Phi_total, Phi_int_historia, DIM_INTERNA, DIM_TIME)
        res_v2 = calcular_resonancia_v2(Phi_total, Phi_int_historia, DIM_INTERNA, DIM_TIME)
        reserva_ef, modos = calcular_reserva(Phi_total, DIM_INTERNA, res_v2, DIM_TIME)
        lf = calcular_respuesta_lf(res_v2, reserva_ef, historial_res_v2, DT)
        
        res_orig_f3.append(res_orig)
        res_v2_f3.append(res_v2)
        reserva_f3.append(reserva_ef)
        lf_f3.append(lf['lf_activa'])
        
        historial_res_orig.append(res_orig)
        historial_res_v2.append(res_v2)
        historial_reserva.append(reserva_ef)
        historial_lf_activa.append(lf['lf_activa'])
        
        if idx % 200 == 0:
            print(f"    t={idx*DT:.1f}s | res_orig={res_orig:.3f} | res_v2={res_v2:.3f} | reserva={reserva_ef:.3f} | LF={'ACTIVA' if lf['lf_activa'] else 'inactiva'}")
    
    grad_esp_f3 = calcular_gradiente_espectral_diferencial(Phi_total, DIM_INTERNA, DIM_TIME)
    historial_grad_esp.append(('F3', grad_esp_f3))
    
    print(f"\n  Resumen Fase 3:")
    print(f"    Resonancia orig (media):   {np.mean(res_orig_f3):.4f}")
    print(f"    Resonancia v2 (media):     {np.mean(res_v2_f3):.4f}")
    print(f"    Reserva media:             {np.mean(reserva_f3):.4f}")
    print(f"    LF activa (%):             {100*np.mean(lf_f3):.1f}%")
    print(f"    Gradiente espectral diff:  {grad_esp_f3:.6f}")
    
    # ============================================================
    # FASE 4: PERTURBACIÓN SOSTENIDA
    # ============================================================
    print("\n[Fase 4] Perturbación sostenida (ruido, alpha=0.0)")
    
    res_orig_f4 = []
    res_v2_f4 = []
    reserva_f4 = []
    lf_f4 = []
    
    for idx in range(N_PASOS_PERT_SOST):
        objetivo = preparar_objetivo_audio(audio_ruido, sr_ruido, N_PASOS_PERT_LEVE + idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
        Phi_total, Phi_vel_total, W, Phi_int_historia = actualizar_campo_total(
            Phi_total, Phi_vel_total, W, Phi_int_historia,
            objetivo, alpha=0.0, omega_natural=omega_natural,
            amort_natural=amort_natural, dt=DT, entrenando=False
        )
        
        res_orig = calcular_resonancia_original(Phi_total, Phi_int_historia, DIM_INTERNA, DIM_TIME)
        res_v2 = calcular_resonancia_v2(Phi_total, Phi_int_historia, DIM_INTERNA, DIM_TIME)
        reserva_ef, modos = calcular_reserva(Phi_total, DIM_INTERNA, res_v2, DIM_TIME)
        lf = calcular_respuesta_lf(res_v2, reserva_ef, historial_res_v2, DT)
        
        res_orig_f4.append(res_orig)
        res_v2_f4.append(res_v2)
        reserva_f4.append(reserva_ef)
        lf_f4.append(lf['lf_activa'])
        
        historial_res_orig.append(res_orig)
        historial_res_v2.append(res_v2)
        historial_reserva.append(reserva_ef)
        historial_lf_activa.append(lf['lf_activa'])
        
        if idx % 400 == 0:
            print(f"    t={idx*DT:.1f}s | res_orig={res_orig:.3f} | res_v2={res_v2:.3f} | reserva={reserva_ef:.3f} | LF={'ACTIVA' if lf['lf_activa'] else 'inactiva'}")
    
    grad_esp_f4 = calcular_gradiente_espectral_diferencial(Phi_total, DIM_INTERNA, DIM_TIME)
    historial_grad_esp.append(('F4', grad_esp_f4))
    
    print(f"\n  Resumen Fase 4:")
    print(f"    Resonancia orig (media):   {np.mean(res_orig_f4):.4f}")
    print(f"    Resonancia v2 (media):     {np.mean(res_v2_f4):.4f}")
    print(f"    Reserva media:             {np.mean(reserva_f4):.4f}")
    print(f"    LF activa (%):             {100*np.mean(lf_f4):.1f}%")
    print(f"    Gradiente espectral diff:  {grad_esp_f4:.6f}")
    
    # ============================================================
    # FASE 5: RE-ACOPLAMIENTO DESDE LF
    # ============================================================
    print("\n[Fase 5] Re-acoplamiento desde LF activa (voz, alpha=0.0)")
    
    tiempo_desde_lf, Phi_total, Phi_vel_total, W, Phi_int_historia = calcular_tiempo_reacoplamiento(
        Phi_total, Phi_vel_total, W, Phi_int_historia,
        omega_natural, amort_natural, DURACION_REACOPLAMIENTO, DIM_TIME
    )
    
    print(f"  Tiempo hasta resonancia > 0.7: {tiempo_desde_lf:.2f}s" if tiempo_desde_lf else "  No alcanzó resonancia > 0.7")
    
    # ============================================================
    # EXPERIMENTO CONTROL BASAL
    # ============================================================
    tiempo_basal = simular_experimento_control_basal()
    
    # ============================================================
    # DIAGNÓSTICO DE HIPÓTESIS
    # ============================================================
    print("\n" + "=" * 100)
    print("DIAGNÓSTICO DE HIPÓTESIS")
    print("=" * 100)
    
    # Opción A: Detección de señal (res_v2 discrimina voz vs ruido)
    res_v2_f2_media = np.mean(res_v2_f2)
    res_v2_f3_media = np.mean(res_v2_f3)
    ratio_vs_ruido = res_v2_f2_media / max(res_v2_f3_media, 1e-6)
    
    opcion_a = ratio_vs_ruido > 2.0
    
    # Opción B: Detección de régimen (LF activa solo bajo perturbación sostenida)
    lf_f3_pct = 100 * np.mean(lf_f3)
    lf_f4_pct = 100 * np.mean(lf_f4)
    
    opcion_b = (lf_f3_pct < 20 and lf_f4_pct > 50)
    
    # Opción C: Disponibilidad para nuevo acoplamiento
    opcion_c = (tiempo_desde_lf is not None and tiempo_basal is not None and tiempo_desde_lf < tiempo_basal)
    
    print(f"\n  Opción A (señal): res_v2 voz={res_v2_f2_media:.3f} vs ruido={res_v2_f3_media:.3f} | ratio={ratio_vs_ruido:.2f}")
    print(f"    -> {'✅ A validada' if opcion_a else '❌ A no validada'}")
    
    print(f"\n  Opción B (régimen): LF_f3={lf_f3_pct:.1f}% vs LF_f4={lf_f4_pct:.1f}%")
    print(f"    -> {'✅ B validada' if opcion_b else '❌ B no validada'}")
    
    print(f"\n  Opción C (disponibilidad): t_LF={tiempo_desde_lf:.2f}s vs t_basal={tiempo_basal:.2f}s")
    print(f"    -> {'✅ C validada' if opcion_c else '❌ C no validada'}")
    
    # ============================================================
    # CONCLUSIONES
    # ============================================================
    print("\n" + "=" * 100)
    print("CONCLUSIONES")
    print("=" * 100)
    
    if opcion_a and opcion_b and opcion_c:
        print("\n  ✅ A + B + C: HETA COMPLETO")
        print("     El sistema distingue señales, opera por régimen Y el estado LF facilita acoplamiento.")
    elif opcion_b and opcion_c:
        print("\n  ✅ B + C: HETA POR RÉGIMEN")
        print("     El sistema no distingue señales pero sí regímenes, y el estado LF facilita acoplamiento.")
        print("     La exaptación no ocurre por reconocimiento de señal sino por transición de fase.")
    elif opcion_c:
        print("\n  ✅ C: HETA DÉBIL")
        print("     El estado de colapso sí facilita re-acoplamiento.")
        print("     Hay disponibilidad pero no reconocimiento de señal o régimen.")
    elif opcion_b:
        print("\n  ⚠️ B: RÉGIMEN PERO SIN DISPONIBILIDAD")
        print("     El campo opera por transición de régimen, pero el estado LF no facilita re-acoplamiento.")
    elif opcion_a:
        print("\n  ⚠️ A: SEÑAL PERO SIN RÉGIMEN NI DISPONIBILIDAD")
        print("     El detector distingue voz de ruido, pero el campo no transiciona por régimen.")
    else:
        print("\n  ❌ NINGUNA HIPÓTESIS VALIDADA")
        print("     El estado LF activa en v73 era artefacto numérico o colapso sin disponibilidad.")
    
    # Visualización
    print("\n[Generando visualización...]")
    
    tiempos_f2 = np.arange(0, DURACION_TEST_DOMINIO, DT)
    tiempos_f3 = np.arange(0, DURACION_PERTURBACION_LEVE, DT)
    tiempos_f4 = np.arange(0, DURACION_PERTURBACION_SOSTENIDA, DT)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Gráfico 1: Resonancia original vs v2 en Fase 2
    ax = axes[0, 0]
    ax.plot(tiempos_f2, res_orig_f2, label='res_orig (int vs hist)', alpha=0.7)
    ax.plot(tiempos_f2, res_v2_f2, label='res_v2 (aud vs hist)', alpha=0.7)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Resonancia')
    ax.set_title('Resonancia en dominio (voz)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 2: Resonancia v2 Fase 3 vs Fase 4
    ax = axes[0, 1]
    ax.plot(tiempos_f3, res_v2_f3, label='Fase 3 (ruido leve)', alpha=0.7)
    ax.plot(tiempos_f4, res_v2_f4, label='Fase 4 (ruido sost.)', alpha=0.7)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Resonancia v2')
    ax.set_title('Resonancia bajo perturbación')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 3: Reserva vs LF activa
    ax = axes[0, 2]
    ax2 = ax.twinx()
    ax.plot(tiempos_f4, reserva_f4, label='Reserva', color='blue', alpha=0.7)
    ax2.fill_between(tiempos_f4, 0, lf_f4, step='mid', alpha=0.3, color='red', label='LF activa')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Reserva', color='blue')
    ax2.set_ylabel('LF activa', color='red')
    ax.set_title('Reserva y LF en Fase 4')
    ax.grid(True, alpha=0.3)
    
    # Gráfico 4: Comparación tiempos de re-acoplamiento
    ax = axes[1, 0]
    if tiempo_desde_lf and tiempo_basal:
        ax.bar(['Desde LF activa', 'Basal (sin entrenamiento)'], [tiempo_desde_lf, tiempo_basal])
        ax.axhline(y=tiempo_basal, color='r', linestyle='--', alpha=0.5)
    ax.set_ylabel('Tiempo (s)')
    ax.set_title('Velocidad de re-acoplamiento')
    ax.grid(True, alpha=0.3)
    
    # Gráfico 5: Gradiente espectral diferencial por fase
    ax = axes[1, 1]
    fases = ['F2', 'F3', 'F4']
    valores = [grad_esp_f2, grad_esp_f3, grad_esp_f4]
    ax.bar(fases, valores, color=['green', 'orange', 'red'])
    ax.set_ylabel('Gradiente espectral diferencial')
    ax.set_title('Diferenciación espectral entre regiones')
    ax.grid(True, alpha=0.3)
    
    # Gráfico 6: Diagnóstico de hipótesis
    ax = axes[1, 2]
    ax.text(0.1, 0.8, f"Opción A: {'✅' if opcion_a else '❌'}\n"
                      f"  ratio={ratio_vs_ruido:.2f} (necesita >2.0)", fontsize=10)
    ax.text(0.1, 0.5, f"Opción B: {'✅' if opcion_b else '❌'}\n"
                      f"  LF_f3={lf_f3_pct:.1f}% vs LF_f4={lf_f4_pct:.1f}%\n"
                      f"  necesita <20% y >50%", fontsize=10)
    ax.text(0.1, 0.2, f"Opción C: {'✅' if opcion_c else '❌'}\n"
                      f"  t_LF={tiempo_desde_lf:.2f}s vs t_basal={tiempo_basal:.2f}s", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Diagnóstico de hipótesis')
    
    plt.suptitle('VSTCosmos v74 - HETA Redefinido', fontsize=14)
    plt.tight_layout()
    plt.savefig('v74_heta_resultados.png', dpi=150)
    print("  Gráfico guardado: v74_heta_resultados.png")
    
    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)


if __name__ == "__main__":
    main()