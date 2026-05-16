#!/usr/bin/env python3
"""
VSTCosmos v80e — Memorias duales con olvido invertido

Cambios respecto a v80d:
- Olvido proporcional a INCOHERENCIA (error alto → olvido alto)
- ERROR_EQUILIBRIO_W_REC medido durante entrenamiento (derivado del campo)
- Aprendizaje proporcional a coherencia (error bajo → aprendizaje alto)

Efecto esperado:
- Fuera de dominio: olvido alto → W_rec disipa contexto
- En dominio: aprendizaje alto → W_rec aprende/re-aprende
- Retorno a voz: W_rec puede re-aprender frescamente
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

# Error de equilibrio de difusión (referencia basal)
ERROR_DIFUSION = DIFUSION_BASE ** 2         # = 0.0225

# ERROR_EQUILIBRIO_W_REC será medido durante entrenamiento
ERROR_EQUILIBRIO_W_REC = None

# Fuerza del atractor
GAMMA_ATRACTOR = 0.05

# Separación espectral
BANDA_BAJA = slice(0, DIM_INTERNA // 2)    # modos 0-15: identidad
BANDA_ALTA = slice(DIM_INTERNA // 2, None)  # modos 16-31: contexto

print("=" * 100)
print("VSTCosmos v80e — Memorias duales con olvido invertido")
print("")
print("  Cambios respecto a v80d:")
print("  1. Olvido proporcional a INCOHERENCIA (error alto → olvido alto)")
print("  2. ERROR_EQUILIBRIO_W_REC medido durante entrenamiento")
print("  3. Aprendizaje proporcional a coherencia")
print("")
print("  Parámetros derivados de la física del campo:")
print(f"    T_PROFUNDA = {T_PROFUNDA_SEG:.1f}s ({T_PROFUNDA_PASOS} pasos)")
print(f"    T_RECIENTE = {T_RECIENTE_SEG:.1f}s ({T_RECIENTE_PASOS} pasos)")
print(f"    ETA_PROFUNDA_BASE = {ETA_PROFUNDA_BASE:.6f}")
print(f"    ETA_RECIENTE_BASE = {ETA_RECIENTE_BASE:.6f}")
print(f"    TAU_PROFUNDA = {TAU_PROFUNDA:.4f}")
print(f"    TAU_RECIENTE = {TAU_RECIENTE:.4f} (modulada por incoherencia)")
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
# NÚCLEO: MEMORIAS DUALES CON OLVIDO INVERTIDO
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

def aplicar_plasticidad_memoria_reciente_invertida(
        W_rec, region_int, region_aud, dt, error_rec, error_equilibrio):
    """
    Memoria reciente con olvido invertido.
    
    INCOHERENCIA = error / (error + error_equilibrio)
    - Cuando error alto (fuera de dominio): incoherencia → 1, olvido alto
    - Cuando error bajo (en dominio): incoherencia → 0, olvido bajo
    
    COHERENCIA = 1 - incoherencia = error_equilibrio / (error + error_equilibrio)
    - Aprendizaje proporcional a coherencia
    """
    reg_int_alta = region_int[BANDA_ALTA, :]
    reg_aud_alta = region_aud[BANDA_ALTA, :]
    
    # Incoherencia: alta fuera de dominio, baja en dominio
    incoherencia = error_rec / (error_rec + error_equilibrio)
    coherencia = 1.0 - incoherencia  # = error_equilibrio / (error + error_equilibrio)
    
    # Tasas moduladas
    tasa_aprendizaje = ETA_RECIENTE_BASE * coherencia
    tasa_olvido = TAU_RECIENTE * incoherencia
    
    # Plasticidad hebbiana con tasas moduladas
    correlacion = (reg_int_alta @ reg_aud_alta.T) / DIM_TIME
    dW = tasa_aprendizaje * correlacion - tasa_olvido * W_rec
    W_nueva = np.clip(W_rec + dW * dt, -W_MAX, W_MAX)
    
    # Predicción normalizada con tanh
    prediccion = np.tanh(W_nueva @ reg_aud_alta)
    M_hebb = np.zeros((DIM_INTERNA, DIM_TIME))
    M_hebb[BANDA_ALTA, :] = GAMMA_PLAST * (prediccion - reg_int_alta)
    
    return W_nueva, M_hebb, tasa_olvido, tasa_aprendizaje, incoherencia, coherencia

def calcular_error_memoria_reciente(W_rec, region_int, region_aud):
    reg_int_alta = region_int[BANDA_ALTA, :]
    reg_aud_alta = region_aud[BANDA_ALTA, :]
    
    prediccion = np.tanh(W_rec @ reg_aud_alta)
    error = np.mean((prediccion - reg_int_alta) ** 2)
    
    return float(error)

def atractor_dual(Phi_int_historia, region_int):
    return GAMMA_ATRACTOR * (Phi_int_historia - region_int)

# ============================================================
# ACTUALIZACIÓN PRINCIPAL
# ============================================================
def actualizar_campo_con_memorias_duales(
        Phi_total, Phi_vel_total, W_prof, W_rec,
        Phi_int_historia,
        objetivo_audio, alpha,
        omega_natural, amort_natural,
        dt, entrenando, error_equilibrio_w_rec,
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
    
    # Plasticidad profunda (identidad)
    W_prof_nueva, M_prof = aplicar_plasticidad_memoria_profunda(
        W_prof, region_int, region_aud, dt
    )
    
    # Calcular error actual antes de actualizar W_rec
    error_actual = calcular_error_memoria_reciente(W_rec, region_int, region_aud)
    
    # Plasticidad reciente con olvido invertido
    W_rec_nueva, M_rec, tasa_olvido, tasa_aprendizaje, incoherencia, coherencia = \
        aplicar_plasticidad_memoria_reciente_invertida(
            W_rec, region_int, region_aud, dt, error_actual, error_equilibrio_w_rec
        )
    
    # Atractor
    M_atractor = atractor_dual(Phi_int_historia, region_int)
    
    # Campo total
    M_campo = np.zeros_like(Phi_total)
    M_campo[:DIM_INTERNA, :] = M_prof + M_rec + M_atractor
    
    # Calcular error actualizado
    error_rec = calcular_error_memoria_reciente(W_rec_nueva, region_int, region_aud)
    lf_activa = error_rec > error_equilibrio_w_rec
    
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
            lf_activa, error_rec, tasa_olvido, tasa_aprendizaje,
            incoherencia, coherencia)

# ============================================================
# SIMULACIÓN DE FASE
# ============================================================
def simular_fase(Phi_total, Phi_vel_total, W_prof, W_rec,
                 Phi_int_historia,
                 estimulo, alpha, duracion, fase_nombre,
                 omega_natural, amort_natural, error_equilibrio_w_rec, DIM_TIME):
    sr, audio = cargar_audio(estimulo, duracion=duracion)
    ventana_muestras = int(sr * VENTANA_FFT_MS / 1000)
    hop_muestras = int(sr * HOP_FFT_MS / 1000)
    n_pasos = int(duracion / DT)
    
    ged_hist = []
    lf_hist = []
    error_rec_hist = []
    w_prof_norma_hist = []
    w_rec_norma_hist = []
    balance_hist = []
    incoherencia_hist = []
    coherencia_hist = []
    tasa_aprendizaje_hist = []
    tasa_olvido_hist = []
    
    for idx in range(n_pasos):
        objetivo = preparar_objetivo_audio(audio, sr, idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
        
        ged = calcular_gradiente_espectral_diferencial(Phi_total, DIM_INTERNA, DIM_TIME)
        
        (Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia,
         lf_activa, error_rec, tasa_olvido, tasa_aprendizaje,
         incoherencia, coherencia) = actualizar_campo_con_memorias_duales(
            Phi_total, Phi_vel_total, W_prof, W_rec,
            Phi_int_historia,
            objetivo, alpha,
            omega_natural, amort_natural,
            DT, entrenando=False, error_equilibrio_w_rec=error_equilibrio_w_rec,
            DIM_TIME=DIM_TIME
        )
        
        balance, _, _ = calcular_balance_plastica_difusiva(Phi_total, W_rec, DIM_INTERNA)
        
        ged_hist.append(ged)
        lf_hist.append(1 if lf_activa else 0)
        error_rec_hist.append(error_rec)
        w_prof_norma_hist.append(np.mean(np.abs(W_prof)))
        w_rec_norma_hist.append(np.mean(np.abs(W_rec)))
        balance_hist.append(balance)
        incoherencia_hist.append(incoherencia)
        coherencia_hist.append(coherencia)
        tasa_aprendizaje_hist.append(tasa_aprendizaje)
        tasa_olvido_hist.append(tasa_olvido)
    
    return {
        'ged_mean': np.mean(ged_hist),
        'lf_pct': 100 * np.mean(lf_hist),
        'w_prof_norma': np.mean(w_prof_norma_hist),
        'w_rec_norma': np.mean(w_rec_norma_hist),
        'w_rec_inicio': w_rec_norma_hist[0],
        'w_rec_fin': w_rec_norma_hist[-1],
        'error_rec_mean': np.mean(error_rec_hist),
        'balance_mean': np.mean(balance_hist),
        'incoherencia_mean': np.mean(incoherencia_hist),
        'coherencia_mean': np.mean(coherencia_hist),
        'tasa_aprendizaje_mean': np.mean(tasa_aprendizaje_hist),
        'tasa_olvido_mean': np.mean(tasa_olvido_hist),
        'phi_total': Phi_total.copy(),
        'w_prof': W_prof.copy(),
        'w_rec': W_rec.copy(),
        'phi_int_historia': Phi_int_historia.copy()
    }

# ============================================================
# ENTRENAMIENTO CON MEDICIÓN DEL ERROR DE EQUILIBRIO
# ============================================================
def entrenar_y_medir_error_equilibrio():
    """Entrena el campo y mide ERROR_EQUILIBRIO_W_REC como media del error en últimos pasos"""
    Phi_total = inicializar_campo_total()
    Phi_vel_total = np.zeros_like(Phi_total)
    W_prof = inicializar_memoria_profunda()
    W_rec = inicializar_memoria_reciente()
    Phi_int_historia = inicializar_historia_interna()
    
    omega_natural, amort_natural = calcular_frecuencias_naturales(DIM_TOTAL, DIM_INTERNA)
    
    sr_voz, audio_voz = cargar_audio("Voz_Estudio.wav", duracion=DURACION_ENTRENO)
    ventana_muestras = int(sr_voz * VENTANA_FFT_MS / 1000)
    hop_muestras = int(sr_voz * HOP_FFT_MS / 1000)
    
    # Para medir error al final del entrenamiento
    errores_ultimos_pasos = []
    ERROR_EQUILIBRIO_DEFAULT = ERROR_DIFUSION  # fallback
    
    for idx in range(N_PASOS_ENTRENO):
        objetivo = preparar_objetivo_audio(audio_voz, sr_voz, idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
        
        # Usar un error_equilibrio temporal (difusión) durante entrenamiento
        (Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia,
         _, error_rec, _, _, _, _) = actualizar_campo_con_memorias_duales(
            Phi_total, Phi_vel_total, W_prof, W_rec,
            Phi_int_historia,
            objetivo, alpha=0.05,
            omega_natural=omega_natural, amort_natural=amort_natural,
            dt=DT, entrenando=True, error_equilibrio_w_rec=ERROR_DIFUSION,
            DIM_TIME=DIM_TIME
        )
        
        # Guardar errores de los últimos 1000 pasos
        if idx >= N_PASOS_ENTRENO - 1000:
            errores_ultimos_pasos.append(error_rec)
    
    error_equilibrio = np.mean(errores_ultimos_pasos) if errores_ultimos_pasos else ERROR_DIFUSION
    print(f"  ERROR_EQUILIBRIO_W_REC medido: {error_equilibrio:.6f} (difusión: {ERROR_DIFUSION:.4f})")
    
    return (Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia,
            omega_natural, amort_natural, error_equilibrio)

# ============================================================
# MAIN
# ============================================================
def main():
    # Entrenamiento y medición del error de equilibrio
    print("\n[Fase 1] Entrenamiento (voz, alpha=0.05, 30s)")
    (Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia,
     omega_natural, amort_natural, ERROR_EQUILIBRIO_W_REC) = entrenar_y_medir_error_equilibrio()
    
    print(f"  W_prof tras entreno: {np.mean(np.abs(W_prof)):.4f}")
    print(f"  W_rec tras entreno: {np.mean(np.abs(W_rec)):.4f}")
    
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
            omega_natural, amort_natural, ERROR_EQUILIBRIO_W_REC, DIM_TIME
        )
        resultados.append(res)
        
        Phi_total = res['phi_total']
        W_prof = res['w_prof']
        W_rec = res['w_rec']
        Phi_int_historia = res['phi_int_historia']
        
        balance_status = ""
        if 0.1 < res['balance_mean'] < 10:
            balance_status = "✅ equilibrado"
        elif res['balance_mean'] <= 0.1:
            balance_status = "⚠️ plástica débil"
        else:
            balance_status = "❌ plástica dominante"
        
        print(f"    GED: {res['ged_mean']:.6f} | LF: {res['lf_pct']:.1f}%")
        print(f"    W_prof: {res['w_prof_norma']:.4f} | W_rec: {res['w_rec_norma']:.4f}")
        print(f"    Error: {res['error_rec_mean']:.6f} (eq: {ERROR_EQUILIBRIO_W_REC:.6f})")
        print(f"    Coherencia: {res['coherencia_mean']:.4f} | Incoherencia: {res['incoherencia_mean']:.4f}")
        print(f"    Aprendizaje: {res['tasa_aprendizaje_mean']:.6f} | Olvido: {res['tasa_olvido_mean']:.6f}")
        print(f"    Balance: {res['balance_mean']:.3f} {balance_status}")
    
    # Diagnóstico
    print("\n" + "=" * 100)
    print("DIAGNÓSTICO — v80e con olvido invertido")
    print("=" * 100)
    
    error_f2 = resultados[0]['error_rec_mean']
    error_f6 = resultados[4]['error_rec_mean']
    criterio1 = error_f2 < 1.0
    criterio2 = error_f2 < error_f6
    
    balance_f7 = resultados[5]['balance_mean']
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
    
    lf_f7 = resultados[5]['lf_pct']
    
    print(f"\n  ERROR_EQUILIBRIO_W_REC medido: {ERROR_EQUILIBRIO_W_REC:.6f}")
    print(f"\n  Criterio 1 — Error F2 < 1.0:                {error_f2:.6f} {'✅' if criterio1 else '❌'}")
    print(f"  Criterio 2 — Error F2 < Error F6:            {error_f2:.6f} < {error_f6:.6f} {'✅' if criterio2 else '❌'}")
    print(f"  Criterio 3 — Balance 0.1-10 en F7:           {balance_f7:.3f} {'✅' if criterio3 else '❌'}")
    print(f"  Criterio 4 — W_prof creció:                  {w_prof_f2:.4f} → {w_prof_f7:.4f} {'✅' if criterio4 else '❌'}")
    print(f"  Criterio 5 — W_rec olvida en F7:             {w_rec_f6:.4f} → {w_rec_f7:.4f} {'✅' if criterio5 else '❌'}")
    print(f"  Criterio 6 — Coherencia F7 > Coherencia F6:  {coherencia_f6:.4f} → {coherencia_f7:.4f} {'✅' if criterio6 else '❌'}")
    print(f"  Criterio 7 — Error F7 < Equilibrio:          {error_f7:.6f} < {ERROR_EQUILIBRIO_W_REC:.6f} {'✅' if criterio7 else '❌'}")
    print(f"\n  LF activa en Fase 7:                       {lf_f7:.1f}%")
    
    print("\n  VEREDICTO:")
    if all([criterio1, criterio2, criterio3, criterio4, criterio5, criterio6, criterio7]):
        print("  ✅ v80e VALIDADO — Olvido invertido funciona")
        print("     El error de equilibrio se midió desde el entrenamiento,")
        print("     el olvido es proporcional a la incoherencia,")
        print("     W_rec disipa contexto fuera de dominio y re-aprende al retornar.")
    elif criterio5 and criterio6:
        print("  ⚠️ CRITERIOS PARCIALES — Olvido funcional pero ajuste fino necesario")
    else:
        print("  ❌ v80e NO VALIDADO — Verificar implementación o ajustar parámetros")
    
    # Guardar CSV
    with open('v80e_memorias_duales.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['fase', 'ged_mean', 'lf_pct', 'w_prof_norma', 'w_rec_norma',
                         'error_rec', 'balance', 'coherencia', 'incoherencia',
                         'tasa_aprendizaje', 'tasa_olvido'])
        for i, (fase, res) in enumerate(zip(fases, resultados)):
            writer.writerow([fase[0], res['ged_mean'], res['lf_pct'],
                            res['w_prof_norma'], res['w_rec_norma'],
                            res['error_rec_mean'], res['balance_mean'],
                            res['coherencia_mean'], res['incoherencia_mean'],
                            res['tasa_aprendizaje_mean'], res['tasa_olvido_mean']])
    
    print("\n  CSV guardado: v80e_memorias_duales.csv")
    
    # Gráfico
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    nombres = ['Voz', 'Música', 'Tono', 'Voz+Viento', 'Ruido', 'Reacop']
    
    axes[0,0].bar(nombres, [r['ged_mean'] for r in resultados])
    axes[0,0].set_title('GED')
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].bar(nombres, [r['lf_pct'] for r in resultados])
    axes[0,1].set_title('LF activa (%)')
    axes[0,1].grid(True, alpha=0.3)
    
    axes[0,2].bar(nombres, [r['w_prof_norma'] for r in resultados], alpha=0.7, label='W_prof')
    axes[0,2].bar(nombres, [r['w_rec_norma'] for r in resultados], alpha=0.5, label='W_rec')
    axes[0,2].set_title('Norma de memorias')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    axes[1,0].bar(nombres, [r['error_rec_mean'] for r in resultados])
    axes[1,0].axhline(y=ERROR_EQUILIBRIO_W_REC, color='r', linestyle='--', label='Equilibrio')
    axes[1,0].set_title('Error W_rec')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].bar(nombres, [r['coherencia_mean'] for r in resultados], alpha=0.7, label='Coherencia')
    axes[1,1].bar(nombres, [r['incoherencia_mean'] for r in resultados], alpha=0.5, label='Incoherencia')
    axes[1,1].set_title('Coherencia vs Incoherencia')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    axes[1,2].plot(nombres, [r['tasa_aprendizaje_mean'] for r in resultados], 'o-', label='Aprendizaje')
    axes[1,2].plot(nombres, [r['tasa_olvido_mean'] for r in resultados], 's-', label='Olvido')
    axes[1,2].set_title('Tasas moduladas')
    axes[1,2].legend()
    axes[1,2].grid(True, alpha=0.3)
    
    plt.suptitle('VSTCosmos v80e — Memorias duales con olvido invertido', fontsize=14)
    plt.tight_layout()
    plt.savefig('v80e_memorias_duales.png', dpi=150)
    print("  Gráfico guardado: v80e_memorias_duales.png")
    
    # Guardar TXT
    with open('v80e_memorias_duales.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("VSTCosmos v80e — Diagnóstico\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"ERROR_EQUILIBRIO_W_REC = {ERROR_EQUILIBRIO_W_REC:.6f} (medido)\n\n")
        f.write(f"Criterio 1 — Error F2 < 1.0:                {error_f2:.6f} {'✅' if criterio1 else '❌'}\n")
        f.write(f"Criterio 2 — Error F2 < Error F6:            {error_f2:.6f} < {error_f6:.6f} {'✅' if criterio2 else '❌'}\n")
        f.write(f"Criterio 3 — Balance 0.1-10 en F7:           {balance_f7:.3f} {'✅' if criterio3 else '❌'}\n")
        f.write(f"Criterio 4 — W_prof creció:                  {w_prof_f2:.4f} → {w_prof_f7:.4f} {'✅' if criterio4 else '❌'}\n")
        f.write(f"Criterio 5 — W_rec olvida en F7:             {w_rec_f6:.4f} → {w_rec_f7:.4f} {'✅' if criterio5 else '❌'}\n")
        f.write(f"Criterio 6 — Coherencia F7 > F6:             {coherencia_f6:.4f} → {coherencia_f7:.4f} {'✅' if criterio6 else '❌'}\n")
        f.write(f"Criterio 7 — Error F7 < Equilibrio:          {error_f7:.6f} < {ERROR_EQUILIBRIO_W_REC:.6f} {'✅' if criterio7 else '❌'}\n\n")
        
        if all([criterio1, criterio2, criterio3, criterio4, criterio5, criterio6, criterio7]):
            f.write("VEREDICTO: ✅ v80e VALIDADO\n")
            f.write("Olvido invertido + equilibrio medido funcionan.\n")
        else:
            f.write("VEREDICTO: ❌ v80e NO VALIDADO\n")
        f.write("=" * 60 + "\n")
    
    print("  TXT guardado: v80e_memorias_duales.txt")
    
    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)


if __name__ == "__main__":
    main()