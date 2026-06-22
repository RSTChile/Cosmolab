#!/usr/bin/env python3
"""
VSTCosmos - v75: GED como detector de LF
El detector de LF ya no es similitud coseno con el atractor.
Es el GRADIENTE ESPECTRAL DIFERENCIAL (GED) entre region_int y region_aud.

Predicción:
- Voz (dominio de competencia): GED alto → LF inactiva
- Ruido (fuera de dominio): GED bajo → LF activa
- No requiere colapso del campo. Usa una distinción real que ya existe.
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
DURACION_TEST_DOMINIO = 20.0
DURACION_PERTURBACION = 20.0  # una sola fase de perturbación

N_PASOS_ENTRENO = int(DURACION_ENTRENO / DT)
N_PASOS_DOMINIO = int(DURACION_TEST_DOMINIO / DT)
N_PASOS_PERTURBACION = int(DURACION_PERTURBACION / DT)

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
GAMMA_MEMORIA = 0.05  # mantenemos el valor que funcionó en v72c
TAU_INT_HIST = 0.0002

# NUEVOS: Parámetros para GED como detector de LF
UMBRAL_GED = 0.004   # Umbral para considerar que el campo está en dominio
WINDOW_LF = 2.0      # Ventana de estabilidad para LF (segundos)

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

    correlacion = (region_int @ region_aud.T) / DIM_TIME
    correlacion_filtrada = np.where(
        np.abs(correlacion) > UMBRAL_CORRELACION,
        correlacion,
        0.0
    )
    dW = ETA_HEBB * correlacion_filtrada - TAU_W * W
    W_nueva = np.clip(W + dW * dt, -W_MAX, W_MAX)

    M_hebb = GAMMA_PLAST * (W_nueva @ region_aud - region_int)

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
    
    # Prevenir colapso
    varianza_campo = np.var(Phi_nueva[:DIM_INTERNA, :])
    if varianza_campo < 1e-6:
        ruido_minimo = np.random.normal(0, 0.01, Phi_nueva[:DIM_INTERNA, :].shape)
        Phi_nueva[:DIM_INTERNA, :] += ruido_minimo
    
    return (np.clip(Phi_nueva, LIMITE_MIN, LIMITE_MAX),
            np.clip(Phi_vel_nueva, -5.0, 5.0),
            W_nueva,
            Phi_int_historia_nueva)


def calcular_gradiente_espectral_diferencial(Phi_total, dim_interna, DIM_TIME):
    """Calcula el GED entre region_int y region_aud."""
    region_int = Phi_total[:dim_interna, :]
    region_aud = Phi_total[dim_interna:, :]
    
    perfil_int = np.zeros(DIM_TIME // 2)
    perfil_aud = np.zeros(DIM_TIME // 2)
    
    for banda in range(dim_interna):
        serie_int = region_int[banda, :] - np.mean(region_int[banda, :])
        serie_aud = region_aud[banda, :] - np.mean(region_aud[banda, :])
        
        fft_int = np.fft.rfft(serie_int)
        fft_aud = np.fft.rfft(serie_aud)
        
        perfil_int += np.abs(fft_int[:DIM_TIME//2]) ** 2
        perfil_aud += np.abs(fft_aud[:DIM_TIME//2]) ** 2
    
    perfil_int /= dim_interna
    perfil_aud /= dim_interna
    
    ged = np.mean(np.abs(perfil_int - perfil_aud))
    return float(ged)


def calcular_respuesta_lf_ged(ged, historial_ged, dt):
    """
    LF se activa cuando GED cae por debajo del umbral.
    No requiere colapso del campo. Usa la distinción real que ya existe.
    """
    fuera_de_dominio = ged < UMBRAL_GED
    
    # Estabilidad: LF se considera activa si se mantiene por WINDOW_LF segundos
    ventana_pasos = int(WINDOW_LF / dt)
    if len(historial_ged) >= ventana_pasos:
        ged_media = np.mean(historial_ged[-ventana_pasos:])
        lf_estable = ged_media < UMBRAL_GED
    else:
        lf_estable = False
    
    return {
        'lf_activa': fuera_de_dominio,
        'lf_estable': lf_estable,
        'ged_actual': ged,
        'umbral': UMBRAL_GED
    }


def simular_experimento_v75():
    print("=" * 100)
    print("VSTCosmos - v75: GED como detector de LF")
    print("El detector de LF es el GRADIENTE ESPECTRAL DIFERENCIAL (GED)")
    print("Fase 1: Entrenamiento (voz, alpha=0.05, 30s)")
    print("Fase 2: Test en dominio (voz, alpha=0.0, 20s)")
    print("Fase 3: Perturbación (ruido, alpha=0.0, 20s)")
    print("=" * 100)

    # Inicialización
    Phi_total = inicializar_campo_total()
    Phi_vel_total = np.zeros_like(Phi_total)
    W = inicializar_plasticidad()
    Phi_int_historia = inicializar_historia_interna()
    omega_natural, amort_natural = calcular_frecuencias_naturales(DIM_TOTAL, DIM_INTERNA)
    
    # Cargar audios
    sr_voz, audio_voz = cargar_audio("Voz_Estudio.wav", duracion=DURACION_ENTRENO + DURACION_TEST_DOMINIO)
    sr_ruido, audio_ruido = cargar_audio("Ruido blanco", duracion=DURACION_PERTURBACION)
    
    ventana_muestras = int(sr_voz * VENTANA_FFT_MS / 1000)
    hop_muestras = int(sr_voz * HOP_FFT_MS / 1000)
    
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
    print(f"  W_tras_entreno: {w_tras_entreno:.4f}")
    
    # ============================================================
    # FASE 2: TEST EN DOMINIO (voz)
    # ============================================================
    print("\n[Fase 2] Test en dominio (voz, alpha=0.0)")
    
    ged_f2 = []
    lf_f2 = []
    
    for idx in range(N_PASOS_DOMINIO):
        objetivo = preparar_objetivo_audio(audio_voz, sr_voz, N_PASOS_ENTRENO + idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
        Phi_total, Phi_vel_total, W, Phi_int_historia = actualizar_campo_total(
            Phi_total, Phi_vel_total, W, Phi_int_historia,
            objetivo, alpha=0.0, omega_natural=omega_natural,
            amort_natural=amort_natural, dt=DT, entrenando=False
        )
        
        ged = calcular_gradiente_espectral_diferencial(Phi_total, DIM_INTERNA, DIM_TIME)
        ged_f2.append(ged)
        
        lf = calcular_respuesta_lf_ged(ged, ged_f2, DT)
        lf_f2.append(lf['lf_activa'])
        
        if idx % 100 == 0:
            print(f"    t={idx*DT:.1f}s | GED={ged:.6f} | LF={'ACTIVA' if lf['lf_activa'] else 'inactiva'}")
    
    ged_f2_media = np.mean(ged_f2)
    lf_f2_pct = 100 * np.mean(lf_f2)
    
    print(f"\n  Resumen Fase 2:")
    print(f"    GED medio: {ged_f2_media:.6f}")
    print(f"    LF activa: {lf_f2_pct:.1f}%")
    
    # ============================================================
    # FASE 3: PERTURBACIÓN (ruido)
    # ============================================================
    print("\n[Fase 3] Perturbación (ruido, alpha=0.0)")
    
    ged_f3 = []
    lf_f3 = []
    
    for idx in range(N_PASOS_PERTURBACION):
        objetivo = preparar_objetivo_audio(audio_ruido, sr_ruido, idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
        Phi_total, Phi_vel_total, W, Phi_int_historia = actualizar_campo_total(
            Phi_total, Phi_vel_total, W, Phi_int_historia,
            objetivo, alpha=0.0, omega_natural=omega_natural,
            amort_natural=amort_natural, dt=DT, entrenando=False
        )
        
        ged = calcular_gradiente_espectral_diferencial(Phi_total, DIM_INTERNA, DIM_TIME)
        ged_f3.append(ged)
        
        lf = calcular_respuesta_lf_ged(ged, ged_f3, DT)
        lf_f3.append(lf['lf_activa'])
        
        if idx % 100 == 0:
            print(f"    t={idx*DT:.1f}s | GED={ged:.6f} | LF={'ACTIVA' if lf['lf_activa'] else 'inactiva'}")
    
    ged_f3_media = np.mean(ged_f3)
    lf_f3_pct = 100 * np.mean(lf_f3)
    
    print(f"\n  Resumen Fase 3:")
    print(f"    GED medio: {ged_f3_media:.6f}")
    print(f"    LF activa: {lf_f3_pct:.1f}%")
    
    # ============================================================
    # DIAGNÓSTICO
    # ============================================================
    print("\n" + "=" * 100)
    print("DIAGNÓSTICO")
    print("=" * 100)
    
    ratio_ged = ged_f2_media / max(ged_f3_media, 1e-6)
    deteccion_ok = ratio_ged > 3.0
    lf_ok = (lf_f2_pct < 20 and lf_f3_pct > 50)
    
    print(f"\n  GED voz:     {ged_f2_media:.6f}")
    print(f"  GED ruido:   {ged_f3_media:.6f}")
    print(f"  Ratio:       {ratio_ged:.2f} {'✅' if deteccion_ok else '❌'}")
    print(f"\n  LF activa en dominio:    {lf_f2_pct:.1f}% (<20%: {'✅' if lf_f2_pct < 20 else '❌'})")
    print(f"  LF activa en ruido:      {lf_f3_pct:.1f}% (>50%: {'✅' if lf_f3_pct > 50 else '❌'})")
    
    print("\n" + "=" * 100)
    print("CONCLUSIÓN")
    print("=" * 100)
    
    if deteccion_ok and lf_ok:
        print("\n  ✅ HETA VALIDADO")
        print("     El detector GED distingue dominio de no-dominio.")
        print("     LF se activa consistentemente fuera del dominio.")
        print("     La exaptación no requiere colapso — usa una distinción real que ya existe.")
    elif deteccion_ok:
        print("\n  ⚠️ DETECCIÓN OK, PERO LF NO SE ACTIVA")
        print("     El GED distingue voz de ruido, pero el umbral de LF está mal calibrado.")
        print("     → Ajustar UMBRAL_GED (actual={UMBRAL_GED})")
    elif lf_ok:
        print("\n  ⚠️ LF SE ACTIVA, PERO NO DISCRIMINA")
        print("     El GED no distingue voz de ruido — la métrica falla.")
        print("     → Revisar cálculo del gradiente espectral diferencial.")
    else:
        print("\n  ❌ NINGÚN CRITERIO CUMPLIDO")
        print("     El GED no distingue y LF no se activa.")
    
    # ============================================================
    # VISUALIZACIÓN
    # ============================================================
    print("\n[Generando visualización...]")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Gráfico 1: Evolución del GED
    ax = axes[0, 0]
    tiempos = np.arange(0, DURACION_TEST_DOMINIO, DT)
    ax.plot(tiempos, ged_f2, label='Fase 2 (voz)', alpha=0.7)
    ax.plot(tiempos, ged_f3, label='Fase 3 (ruido)', alpha=0.7)
    ax.axhline(y=UMBRAL_GED, color='r', linestyle='--', label=f'Umbral GED = {UMBRAL_GED}')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('GED')
    ax.set_title('Gradiente Espectral Diferencial')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 2: LF activa
    ax = axes[0, 1]
    ax.fill_between(tiempos, 0, lf_f2, step='mid', alpha=0.5, label='Fase 2 (voz)', color='green')
    ax.fill_between(tiempos, 0, lf_f3, step='mid', alpha=0.5, label='Fase 3 (ruido)', color='red')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('LF activa')
    ax.set_title('Respuesta LF por GED')
    ax.legend()
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    
    # Gráfico 3: Barras comparativas
    ax = axes[1, 0]
    ax.bar(['Voz', 'Ruido'], [ged_f2_media, ged_f3_media], color=['green', 'red'])
    ax.axhline(y=UMBRAL_GED, color='r', linestyle='--', label=f'Umbral GED = {UMBRAL_GED}')
    ax.set_ylabel('GED medio')
    ax.set_title('Discriminación GED')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 4: LF activa por fase
    ax = axes[1, 1]
    ax.bar(['Voz', 'Ruido'], [lf_f2_pct, lf_f3_pct], color=['green', 'red'])
    ax.axhline(y=20, color='orange', linestyle='--', label='Umbral máximo en dominio (20%)')
    ax.axhline(y=50, color='red', linestyle='--', label='Umbral mínimo fuera (50%)')
    ax.set_ylabel('LF activa (%)')
    ax.set_title('Activación de LF por estímulo')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('VSTCosmos v75 - GED como detector de LF', fontsize=14)
    plt.tight_layout()
    plt.savefig('v75_ged_detector.png', dpi=150)
    print("  Gráfico guardado: v75_ged_detector.png")
    
    # Guardar CSV
    with open('v75_ged_resultado.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['fase', 'ged_mean', 'lf_pct'])
        writer.writerow(['F2', ged_f2_media, lf_f2_pct])
        writer.writerow(['F3', ged_f3_media, lf_f3_pct])
        writer.writerow(['ratio', ratio_ged, ''])
        writer.writerow(['umbral_ged', UMBRAL_GED, ''])
        writer.writerow(['deteccion_ok', deteccion_ok, lf_ok])
    
    print("  CSV guardado: v75_ged_resultado.csv")
    
    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)


if __name__ == "__main__":
    simular_experimento_v75()