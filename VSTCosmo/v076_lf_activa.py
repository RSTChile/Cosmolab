#!/usr/bin/env python3
"""
VSTCosmos - v76: LF Activa: ¿Hace algo?
Tres formas posibles de exaptación computacional en una sola corrida:
1) Exploración: LF facilita re-acoplamiento con estímulos nuevos
2) Aprendizaje: W_exploracion aprende correlaciones bajo LF
3) Generación: Phi_generado produce actividad nueva
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import csv
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PARÁMETROS
# ============================================================
DIM_INTERNA = 32
DIM_AUDITIVA = 32
DIM_TOTAL = DIM_INTERNA + DIM_AUDITIVA

DIM_TIME = 100
DT = 0.01

DURACION_ENTRENO = 30.0
DURACION_FASE = 20.0  # todas las fases de test duran 20s

N_PASOS_ENTRENO = int(DURACION_ENTRENO / DT)
N_PASOS_FASE = int(DURACION_FASE / DT)

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

# Parámetros para LF activa
UMBRAL_GED_LF = 0.003  # GED por debajo de este valor → LF activa

# Parámetros de los tres mecanismos de LF
ETA_EXPLORACION = 0.02   # Opción 1: tasa de acoplamiento con estímulo nuevo
ETA_APRENDIZAJE = 0.03   # Opción 2: tasa de aprendizaje no supervisado
GAMMA_GENERACION = 0.04   # Opción 3: fuerza de generación interna
RUIDO_GENERACION = 0.05   # amplitud del ruido estructurado

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


def lf_exploracion(Phi_total, dim_interna, ged_actual):
    """Opción 1: Exploración de modos no entrenados"""
    region_int = Phi_total[:dim_interna, :]
    region_aud = Phi_total[dim_interna:, :]
    
    # Acoplamiento exploratorio: empujar region_int hacia region_aud
    M_exploracion = ETA_EXPLORACION * (region_aud - region_int)
    return M_exploracion


def lf_aprendizaje(Phi_total, W_exploracion, dim_interna, dt):
    """Opción 2: Aprendizaje no supervisado bajo LF"""
    region_int = Phi_total[:dim_interna, :]
    region_aud = Phi_total[dim_interna:, :]
    
    correlacion = (region_int @ region_aud.T) / DIM_TIME
    correlacion_filtrada = np.where(
        np.abs(correlacion) > 0.05,  # umbral más bajo que W principal
        correlacion,
        0.0
    )
    dW_exp = ETA_APRENDIZAJE * correlacion_filtrada - 0.01 * W_exploracion
    W_nueva = np.clip(W_exploracion + dW_exp * dt, -W_MAX, W_MAX)
    
    M_aprendizaje = 0.005 * (W_nueva @ region_aud - region_int)
    return W_nueva, M_aprendizaje


def _perfil_espectral(region, dim, DIM_TIME):
    perfil = np.zeros(DIM_TIME // 2)
    for banda in range(dim):
        serie = region[banda, :]
        serie = serie - np.mean(serie)
        fft = np.fft.rfft(serie)
        potencia = np.abs(fft) ** 2
        perfil += potencia[:DIM_TIME // 2]
    return perfil / dim


def lf_generacion(Phi_total, Phi_generado, dim_interna, dt, DIM_TIME):
    """Opción 3: Generación interna bajo LF"""
    region_int = Phi_total[:dim_interna, :]
    
    # Actualizar Phi_generado: acumular el estado actual de region_int
    Phi_generado_nueva = 0.99 * Phi_generado + 0.01 * region_int
    
    # Amplificar modos dominantes actuales de region_int
    perfil = _perfil_espectral(region_int, dim_interna, DIM_TIME)
    modo_dominante = np.argmax(perfil)
    mascara_modos = np.zeros(DIM_TIME)
    mascara_modos[modo_dominante] = 1.0
    
    # Reconstruir señal amplificando el modo dominante
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


def calcular_gradiente_espectral_diferencial(Phi_total, dim_interna, DIM_TIME):
    region_int = Phi_total[:dim_interna, :]
    region_aud = Phi_total[dim_interna:, :]
    
    perfil_int = _perfil_espectral(region_int, dim_interna, DIM_TIME)
    perfil_aud = _perfil_espectral(region_aud, dim_interna, DIM_TIME)
    
    ged = np.mean(np.abs(perfil_int - perfil_aud))
    return float(ged)


def actualizar_campo_con_lf(Phi_total, Phi_vel_total, W, W_exploracion,
                             Phi_int_historia, Phi_generado,
                             objetivo_audio, alpha,
                             omega_natural, amort_natural,
                             dt, entrenando, ged_actual, DIM_TIME):
    """Actualización del campo con los tres mecanismos de LF"""
    # Dinámica base
    promedio_local = vecinos(Phi_total)
    difusion = DIFUSION_BASE * (promedio_local - Phi_total)
    
    desviacion = Phi_total - promedio_local
    reaccion = GANANCIA_REACCION * desviacion * (1 - desviacion**2)
    
    term_osc = (-omega_natural**2 * (Phi_total - PHI_EQUILIBRIO)
                - amort_natural * Phi_vel_total)
    
    # Plasticidad hebbiana (siempre activa)
    W_nueva, M_hebb = actualizar_hebb_y_plasticidad(Phi_total, W, dt)
    
    # Atractor aprendido (siempre activo)
    M_atractor = GAMMA_MEMORIA * (Phi_int_historia - Phi_total[:DIM_INTERNA, :])
    
    M_campo = np.zeros_like(Phi_total)
    M_campo[:DIM_INTERNA, :] = M_hebb + M_atractor
    
    lf_activa = ged_actual < UMBRAL_GED_LF
    
    # Inicializar salidas de LF
    W_exploracion_nueva = W_exploracion.copy()
    Phi_generado_nueva = Phi_generado.copy()
    
    if lf_activa:
        # Opción 1: exploración
        M_exp = lf_exploracion(Phi_total, DIM_INTERNA, ged_actual)
        
        # Opción 2: aprendizaje no supervisado
        W_exploracion_nueva, M_apr = lf_aprendizaje(
            Phi_total, W_exploracion, DIM_INTERNA, dt
        )
        
        # Opción 3: generación interna
        Phi_generado_nueva, M_gen = lf_generacion(
            Phi_total, Phi_generado, DIM_INTERNA, dt, DIM_TIME
        )
        
        # Los tres se suman sobre region_int
        M_campo[:DIM_INTERNA, :] += M_exp + M_apr + M_gen
    
    # Actualización del campo
    dPhi_vel = term_osc + reaccion + difusion + M_campo
    Phi_vel_nueva = Phi_vel_total + dt * dPhi_vel
    Phi_nueva = Phi_total + dt * Phi_vel_nueva
    
    # Sesgo operativo
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
            W_nueva, W_exploracion_nueva, Phi_generado_nueva)


def simular_fase(Phi_total, Phi_vel_total, W, W_exploracion,
                 Phi_int_historia, Phi_generado,
                 estimulo, alpha, duracion,
                 omega_natural, amort_natural, fase_nombre,
                 DIM_TIME):
    """Simula una fase completa y retorna métricas"""
    sr, audio = cargar_audio(estimulo, duracion=duracion)
    ventana_muestras = int(sr * VENTANA_FFT_MS / 1000)
    hop_muestras = int(sr * HOP_FFT_MS / 1000)
    n_pasos = int(duracion / DT)
    
    ged_hist = []
    lf_hist = []
    w_exp_hist = []
    phi_gen_dist_hist = []
    phi_gen_vs_aud_hist = []
    
    for idx in range(n_pasos):
        objetivo = preparar_objetivo_audio(audio, sr, idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
        
        ged = calcular_gradiente_espectral_diferencial(Phi_total, DIM_INTERNA, DIM_TIME)
        lf_activa = ged < UMBRAL_GED_LF
        
        Phi_total, Phi_vel_total, W, W_exploracion, Phi_generado = actualizar_campo_con_lf(
            Phi_total, Phi_vel_total, W, W_exploracion,
            Phi_int_historia, Phi_generado,
            objetivo, alpha,
            omega_natural, amort_natural,
            DT, entrenando=False, ged_actual=ged, DIM_TIME=DIM_TIME
        )
        
        ged_hist.append(ged)
        lf_hist.append(lf_activa)
        w_exp_hist.append(np.mean(np.abs(W_exploracion)))
        phi_gen_dist_hist.append(np.mean(np.abs(Phi_generado - Phi_int_historia)))
        phi_gen_vs_aud_hist.append(np.mean(np.abs(Phi_generado - Phi_total[DIM_INTERNA:, :])))
    
    return {
        'ged_mean': np.mean(ged_hist),
        'ged_std': np.std(ged_hist),
        'lf_pct': 100 * np.mean(lf_hist),
        'w_exp_mean': np.mean(w_exp_hist),
        'phi_gen_dist_mean': np.mean(phi_gen_dist_hist),
        'phi_gen_vs_aud_mean': np.mean(phi_gen_vs_aud_hist),
        'ged_hist': ged_hist,
        'lf_hist': lf_hist,
        'phi_total': Phi_total,
        'w': W,
        'w_exp': W_exploracion,
        'phi_generado': Phi_generado
    }


def main():
    print("=" * 100)
    print("VSTCosmos - v76: LF Activa: ¿Hace algo?")
    print("Tres formas de exaptación computacional:")
    print("  1) Exploración: LF facilita re-acoplamiento")
    print("  2) Aprendizaje: W_exploracion aprende bajo LF")
    print("  3) Generación: Phi_generado produce actividad nueva")
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
    
    for idx in range(N_PASOS_ENTRENO):
        objetivo = preparar_objetivo_audio(audio_voz, sr_voz, idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
        # Durante entrenamiento, no hay LF (se usa la misma función sin LF)
        Phi_total, Phi_vel_total, W, W_exploracion, Phi_generado = actualizar_campo_con_lf(
            Phi_total, Phi_vel_total, W, W_exploracion,
            Phi_int_historia, Phi_generado,
            objetivo, alpha=0.05,
            omega_natural=omega_natural, amort_natural=amort_natural,
            dt=DT, entrenando=True, ged_actual=1.0, DIM_TIME=DIM_TIME
        )
        # Actualizar historia interna durante entrenamiento
        region_int = Phi_total[:DIM_INTERNA, :]
        Phi_int_historia = (1 - TAU_INT_HIST) * Phi_int_historia + TAU_INT_HIST * region_int
    
    w_tras_entreno = np.mean(np.abs(W))
    print(f"  W_tras_entreno: {w_tras_entreno:.4f}")
    
    # ============================================================
    # FASES DE TEST
    # ============================================================
    fases = [
        ("Fase 2", "Voz_Estudio.wav", 0.0, "Dominio (voz)"),
        ("Fase 3", "Brandemburgo.wav", 0.0, "No entrenado (música)"),
        ("Fase 4", "Tono puro", 0.0, "No entrenado (tono)"),
        ("Fase 5", "Voz+Viento_1.wav", 0.0, "Degradado (voz+viento)"),
        ("Fase 6", "Ruido blanco", 0.0, "Perturbación basal"),
        ("Fase 7", "Voz_Estudio.wav", 0.0, "Re-acoplamiento (voz)")
    ]
    
    resultados = []
    
    for fase_id, estimulo, alpha, desc in fases:
        print(f"\n[{fase_id}] {desc} ({estimulo}, alpha={alpha})")
        res = simular_fase(
            Phi_total, Phi_vel_total, W, W_exploracion,
            Phi_int_historia, Phi_generado,
            estimulo, alpha, DURACION_FASE,
            omega_natural, amort_natural, fase_id, DIM_TIME
        )
        resultados.append(res)
        
        # Actualizar estado para la siguiente fase
        Phi_total = res['phi_total']
        W = res['w']
        W_exploracion = res['w_exp']
        Phi_generado = res['phi_generado']
        
        print(f"    GED medio: {res['ged_mean']:.6f}")
        print(f"    LF activa: {res['lf_pct']:.1f}%")
        print(f"    W_exp norma: {res['w_exp_mean']:.4f}")
        print(f"    Phi_gen vs historia: {res['phi_gen_dist_mean']:.4f}")
        print(f"    Phi_gen vs estímulo: {res['phi_gen_vs_aud_mean']:.4f}")
    
    # ============================================================
    # DIAGNÓSTICO DE LAS TRES OPCIONES
    # ============================================================
    print("\n" + "=" * 100)
    print("DIAGNÓSTICO DE LF ACTIVA")
    print("=" * 100)
    
    # Opción 1: Exploración (comparar Fase 2 vs Fase 7)
    ged_f2 = resultados[0]['ged_mean']
    ged_f7 = resultados[5]['ged_mean']
    lf_f7 = resultados[5]['lf_pct']
    
    # Calcular velocidad de reacoplamiento (pendiente inicial en Fase 7)
    ged_f7_hist = resultados[5]['ged_hist'][:200]  # primeras 2 segundos
    if len(ged_f7_hist) > 10:
        t_reac = np.linspace(0, 2.0, len(ged_f7_hist))
        grad = np.gradient(ged_f7_hist)
        velocidad_reac = np.mean(grad[10:50]) if len(grad) > 50 else np.mean(grad)
        exploracion_exitosa = velocidad_reac > 0.0005
    else:
        velocidad_reac = 0
        exploracion_exitosa = False
    
    print(f"\n  Opción 1 (Exploración):")
    print(f"    GED Fase 2 (voz, sin LF activa): {ged_f2:.6f}")
    print(f"    GED Fase 7 (re-acoplamiento con voz): {ged_f7:.6f}")
    print(f"    LF activa en Fase 7: {lf_f7:.1f}%")
    print(f"    Velocidad reacoplamiento: {velocidad_reac:.6f}")
    print(f"    {'✅ LF facilita exploración' if exploracion_exitosa else '❌ Sin efecto exploratorio'}")
    
    # Opción 2: Aprendizaje (comparar W_exp en diferentes fases)
    w_exp_f2 = resultados[0]['w_exp_mean']
    w_exp_f3 = resultados[1]['w_exp_mean']
    w_exp_f6 = resultados[4]['w_exp_mean']
    
    aprendizaje_exitoso = (w_exp_f3 > w_exp_f2 * 1.5 or w_exp_f6 > w_exp_f2 * 1.5)
    
    print(f"\n  Opción 2 (Aprendizaje):")
    print(f"    W_exp bajo voz (Dominio): {w_exp_f2:.4f}")
    print(f"    W_exp bajo música: {w_exp_f3:.4f}")
    print(f"    W_exp bajo ruido: {w_exp_f6:.4f}")
    print(f"    {'✅ LF activa aprendizaje nuevo' if aprendizaje_exitoso else '❌ Sin aprendizaje diferencial'}")
    
    # Opción 3: Generación (Phi_generado diferente del atractor y del estímulo)
    phi_gen_vs_hist_f3 = resultados[1]['phi_gen_dist_mean']
    phi_gen_vs_aud_f3 = resultados[1]['phi_gen_vs_aud_mean']
    phi_gen_vs_hist_f6 = resultados[4]['phi_gen_dist_mean']
    phi_gen_vs_aud_f6 = resultados[4]['phi_gen_vs_aud_mean']
    
    generacion_exitosa = (phi_gen_vs_hist_f3 > 0.03 and phi_gen_vs_aud_f3 > 0.03) or \
                         (phi_gen_vs_hist_f6 > 0.03 and phi_gen_vs_aud_f6 > 0.03)
    
    print(f"\n  Opción 3 (Generación):")
    print(f"    Phi_gen vs historia (música): {phi_gen_vs_hist_f3:.4f}")
    print(f"    Phi_gen vs estímulo (música): {phi_gen_vs_aud_f3:.4f}")
    print(f"    Phi_gen vs historia (ruido): {phi_gen_vs_hist_f6:.4f}")
    print(f"    Phi_gen vs estímulo (ruido): {phi_gen_vs_aud_f6:.4f}")
    print(f"    {'✅ LF genera algo nuevo' if generacion_exitosa else '❌ Sin generación genuina'}")
    
    # ============================================================
    # VISUALIZACIÓN
    # ============================================================
    print("\n[Generando visualización...]")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Gráfico 1: GED por fase
    ax = axes[0, 0]
    nombres_fases = ['Voz', 'Música', 'Tono', 'Voz+Viento', 'Ruido', 'Reacop']
    ged_vals = [r['ged_mean'] for r in resultados]
    ax.bar(nombres_fases, ged_vals)
    ax.axhline(y=UMBRAL_GED_LF, color='r', linestyle='--', label=f'Umbral LF = {UMBRAL_GED_LF}')
    ax.set_ylabel('GED medio')
    ax.set_title('Gradiente Espectral Diferencial por fase')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 2: LF activa por fase
    ax = axes[0, 1]
    lf_vals = [r['lf_pct'] for r in resultados]
    ax.bar(nombres_fases, lf_vals, color=['green', 'orange', 'orange', 'orange', 'red', 'green'])
    ax.axhline(y=50, color='red', linestyle='--', label='Umbral activación (50%)')
    ax.set_ylabel('LF activa (%)')
    ax.set_title('LF activa por fase')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 3: Evolución del GED en Fase 3 (música)
    ax = axes[0, 2]
    ax.plot(resultados[1]['ged_hist'], alpha=0.7)
    ax.axhline(y=UMBRAL_GED_LF, color='r', linestyle='--', label=f'Umbral LF = {UMBRAL_GED_LF}')
    ax.set_xlabel('Paso')
    ax.set_ylabel('GED')
    ax.set_title('Evolución del GED (música)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 4: W_exploracion por fase
    ax = axes[1, 0]
    w_exp_vals = [r['w_exp_mean'] for r in resultados]
    ax.bar(nombres_fases, w_exp_vals)
    ax.set_ylabel('W_exploracion (norma)')
    ax.set_title('Aprendizaje no supervisado bajo LF')
    ax.grid(True, alpha=0.3)
    
    # Gráfico 5: Phi_generado vs historia
    ax = axes[1, 1]
    gen_vs_hist = [r['phi_gen_dist_mean'] for r in resultados]
    gen_vs_aud = [r['phi_gen_vs_aud_mean'] for r in resultados]
    x = np.arange(len(nombres_fases))
    ax.bar(x - 0.2, gen_vs_hist, width=0.4, label='vs historia', alpha=0.7)
    ax.bar(x + 0.2, gen_vs_aud, width=0.4, label='vs estímulo', alpha=0.7)
    ax.axhline(y=0.03, color='r', linestyle='--', label='Umbral novedad (0.03)')
    ax.set_xticks(x)
    ax.set_xticklabels(nombres_fases, rotation=45)
    ax.set_ylabel('Diferencia')
    ax.set_title('Generación interna (Phi_generado)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 6: Resumen de criterios
    ax = axes[1, 2]
    criterios = ['Exploración', 'Aprendizaje', 'Generación']
    exitos = [exploracion_exitosa, aprendizaje_exitoso, generacion_exitosa]
    ax.bar(criterios, [1 if e else 0 for e in exitos], color=['green' if e else 'red' for e in exitos])
    ax.set_ylabel('Éxito')
    ax.set_title('Resultado de las tres hipótesis')
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('VSTCosmos v76 - LF Activa: ¿Hace algo?', fontsize=14)
    plt.tight_layout()
    plt.savefig('v76_lf_activa.png', dpi=150)
    print("  Gráfico guardado: v76_lf_activa.png")
    
    # Guardar CSV
    with open('v76_lf_activa.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['fase', 'estimulo', 'ged_mean', 'lf_pct', 'w_exp_mean', 
                         'phi_gen_vs_hist', 'phi_gen_vs_aud'])
        for i, (fase, res) in enumerate(zip(fases, resultados)):
            writer.writerow([fase[0], fase[1], res['ged_mean'], res['lf_pct'], 
                             res['w_exp_mean'], res['phi_gen_dist_mean'], 
                             res['phi_gen_vs_aud_mean']])
    
    print("  CSV guardado: v76_lf_activa.csv")
    
    # Guardar TXT
    with open('v76_lf_activa.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("VSTCosmos v76 - LF Activa: ¿Hace algo?\n")
        f.write("=" * 60 + "\n\n")
        f.write("Opción 1 (Exploración):\n")
        f.write(f"  GED Fase 2: {ged_f2:.6f}\n")
        f.write(f"  GED Fase 7: {ged_f7:.6f}\n")
        f.write(f"  Velocidad reacoplamiento: {velocidad_reac:.6f}\n")
        f.write(f"  Resultado: {'✅ LF facilita exploración' if exploracion_exitosa else '❌ Sin efecto exploratorio'}\n\n")
        f.write("Opción 2 (Aprendizaje):\n")
        f.write(f"  W_exp bajo voz: {w_exp_f2:.4f}\n")
        f.write(f"  W_exp bajo música: {w_exp_f3:.4f}\n")
        f.write(f"  W_exp bajo ruido: {w_exp_f6:.4f}\n")
        f.write(f"  Resultado: {'✅ LF activa aprendizaje nuevo' if aprendizaje_exitoso else '❌ Sin aprendizaje diferencial'}\n\n")
        f.write("Opción 3 (Generación):\n")
        f.write(f"  Phi_gen vs historia (música): {phi_gen_vs_hist_f3:.4f}\n")
        f.write(f"  Phi_gen vs estímulo (música): {phi_gen_vs_aud_f3:.4f}\n")
        f.write(f"  Resultado: {'✅ LF genera algo nuevo' if generacion_exitosa else '❌ Sin generación genuina'}\n\n")
        f.write("=" * 60 + "\n")
        if exploracion_exitosa or aprendizaje_exitoso or generacion_exitosa:
            f.write("CONCLUSION: LF Activa SÍ produce exaptación computacional\n")
            if exploracion_exitosa:
                f.write("- Explora estímulos nuevos más rápido\n")
            if aprendizaje_exitoso:
                f.write("- Aprende correlaciones bajo LF\n")
            if generacion_exitosa:
                f.write("- Genera actividad nueva distinta del atractor y del estímulo\n")
        else:
            f.write("CONCLUSION: LF Activa NO produce exaptación detectable\n")
        f.write("=" * 60 + "\n")
    
    print("  TXT guardado: v76_lf_activa.txt")
    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)


if __name__ == "__main__":
    main()