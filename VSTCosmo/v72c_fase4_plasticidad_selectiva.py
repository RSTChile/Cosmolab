#!/usr/bin/env python3
"""
VSTCosmos - v72-B Fase 4: Plasticidad Hebbiana Selectiva + Espectro de Modos Propios
Reorientación Fourier cosmosemiótica (C-N2.0, F-N1 a F-N4).

Cambios principales:
- Parámetros: ETA_HEBB=0.05, TAU_W=0.008, GAMMA_PLAST=0.01
- Regla Hebb con umbral de correlación (0.1) para evitar ruido débil.
- Nueva métrica: perfil espectral de modos propios de la región interna.
- Criterios: ratio A/C > 3.0 (gradiente) y riqueza modal A/C > 2.0.
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
DURACION_TEST = 30.0
N_PASOS_ENTRENO = int(DURACION_ENTRENO / DT)
N_PASOS_TEST = int(DURACION_TEST / DT)

# Parámetros de dinámica (sin cambio)
DIFUSION_BASE = 0.15
GANANCIA_REACCION = 0.05

OMEGA_MIN = 0.05
OMEGA_MAX = 0.50
AMORT_MIN = 0.01
AMORT_MAX = 0.08
PHI_EQUILIBRIO = 0.5

# Parámetros de plasticidad hebbiana (ajustados según especificación)
ETA_HEBB = 0.05        # antes 0.02
TAU_W = 0.008          # antes 0.005
GAMMA_PLAST = 0.01     # antes 0.03
W_MAX = 1.0

# Umbral para regla Hebb
UMBRAL_CORRELACION = 0.1

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


def inicializar_campo_total(seed=42):
    np.random.seed(seed)
    return np.random.rand(DIM_TOTAL, DIM_TIME) * 0.2 + 0.4


def inicializar_plasticidad():
    return np.zeros((DIM_INTERNA, DIM_AUDITIVA))


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

    # Correlación entre regiones
    correlacion = (region_int @ region_aud.T) / DIM_TIME

    # Umbral: solo reforzar correlaciones significativas
    correlacion_filtrada = np.where(
        np.abs(correlacion) > UMBRAL_CORRELACION,
        correlacion,
        0.0
    )

    # Regla de Hebb con decaimiento
    dW = ETA_HEBB * correlacion_filtrada - TAU_W * W
    W_nueva = np.clip(W + dW * dt, -W_MAX, W_MAX)

    # Término plástico (solo sobre región interna)
    patron_esperado = W_nueva @ region_aud
    M_plasticidad = GAMMA_PLAST * (patron_esperado - region_int)

    return W_nueva, M_plasticidad


def actualizar_campo_total(Phi_total, Phi_vel_total, W,
                           objetivo_audio, alpha,
                           omega_natural, amort_natural):
    # Difusión
    promedio_local = vecinos(Phi_total)
    difusion = DIFUSION_BASE * (promedio_local - Phi_total)
    
    # Reacción
    desviacion = Phi_total - promedio_local
    reaccion = GANANCIA_REACCION * desviacion * (1 - desviacion**2)
    
    # Oscilación
    term_osc = (-omega_natural**2 * (Phi_total - PHI_EQUILIBRIO)
                - amort_natural * Phi_vel_total)
    
    # Plasticidad hebbiana
    W_nueva, M_plasticidad = actualizar_hebb_y_plasticidad(Phi_total, W, DT)
    
    # M_plasticidad actúa solo sobre region_int
    M_campo = np.zeros_like(Phi_total)
    M_campo[:DIM_INTERNA, :] = M_plasticidad
    
    # Actualización del campo
    dPhi_vel = term_osc + reaccion + difusion + M_campo
    Phi_vel_nueva = Phi_vel_total + DT * dPhi_vel
    Phi_nueva = Phi_total + DT * Phi_vel_nueva
    
    # Sesgo operativo
    if alpha > 0:
        region_auditiva_nueva = Phi_nueva[DIM_INTERNA:, :]
        region_auditiva_nueva = (1 - alpha) * region_auditiva_nueva + alpha * objetivo_audio
        Phi_nueva[DIM_INTERNA:, :] = region_auditiva_nueva
    
    return (np.clip(Phi_nueva, LIMITE_MIN, LIMITE_MAX),
            np.clip(Phi_vel_nueva, -5.0, 5.0),
            W_nueva)


def calcular_gradiente(Phi_total, dim_interna):
    region_int = Phi_total[:dim_interna, :]
    region_aud = Phi_total[dim_interna:, :]
    return np.mean(np.abs(region_int - region_aud))


def calcular_acoplamiento(Phi_total, dim_interna):
    region_int = Phi_total[:dim_interna, :]
    region_aud = Phi_total[dim_interna:, :]
    return float(np.mean(region_int * region_aud))


def calcular_perfil_espectral_modos(Phi_total, dim_interna):
    """
    Calcula el perfil de potencia espectral de la región interna.
    Retorna dict con perfil y métricas derivadas.
    """
    region_int = Phi_total[:dim_interna, :]  # (dim_interna, DIM_TIME)
    perfil = np.zeros(DIM_TIME // 2)
    for banda in range(dim_interna):
        serie = region_int[banda, :]
        serie = serie - np.mean(serie)          # centrar
        fft = np.fft.rfft(serie)
        potencia = np.abs(fft) ** 2
        perfil += potencia[:DIM_TIME // 2]
    perfil = perfil / dim_interna               # normalizar por bandas

    frecuencia_dominante = int(np.argmax(perfil))
    umbral_riqueza = np.mean(perfil)
    riqueza_modal = int(np.sum(perfil > umbral_riqueza))
    p_norm = perfil / (np.sum(perfil) + 1e-10)
    entropia_espectral = float(-np.sum(p_norm * np.log(p_norm + 1e-10)))

    return {
        'perfil': perfil,
        'frecuencia_dominante': frecuencia_dominante,
        'riqueza_modal': riqueza_modal,
        'entropia_espectral': entropia_espectral
    }


# ============================================================
# EXPERIMENTOS
# ============================================================
def experimento_persistencia(seed=42):
    """Test A: Entrena voz, testea voz"""
    print("  Test A: Persistencia (voz -> voz)")
    np.random.seed(seed)
    Phi_total = inicializar_campo_total()
    Phi_vel_total = np.zeros_like(Phi_total)
    W = inicializar_plasticidad()
    omega_natural, amort_natural = calcular_frecuencias_naturales(DIM_TOTAL, DIM_INTERNA)

    sr, audio = cargar_audio("Voz_Estudio.wav", duracion=DURACION_ENTRENO + DURACION_TEST)
    ventana_muestras = int(sr * VENTANA_FFT_MS / 1000)
    hop_muestras = int(sr * HOP_FFT_MS / 1000)

    # Entrenamiento (alpha=0.05)
    for idx in range(N_PASOS_ENTRENO):
        objetivo = preparar_objetivo_audio(audio, sr, idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
        Phi_total, Phi_vel_total, W = actualizar_campo_total(
            Phi_total, Phi_vel_total, W, objetivo, alpha=0.05,
            omega_natural=omega_natural, amort_natural=amort_natural
        )
    w_tras_entreno = np.mean(np.abs(W))

    # Test (alpha=0.0)
    gradientes_test = []
    acoplamientos_test = []
    for idx in range(N_PASOS_ENTRENO, N_PASOS_ENTRENO + N_PASOS_TEST):
        objetivo = preparar_objetivo_audio(audio, sr, idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
        Phi_total, Phi_vel_total, W = actualizar_campo_total(
            Phi_total, Phi_vel_total, W, objetivo, alpha=0.0,
            omega_natural=omega_natural, amort_natural=amort_natural
        )
        gradientes_test.append(calcular_gradiente(Phi_total, DIM_INTERNA))
        acoplamientos_test.append(calcular_acoplamiento(Phi_total, DIM_INTERNA))

    grad_estable = np.mean(gradientes_test[-int(10.0/DT):])
    persistencia = sum(1 for g in gradientes_test if g > 0.08) * DT
    min_acop = min(acoplamientos_test)

    # Perfil espectral (últimos 1000 pasos = ~10s)
    perfil_data = calcular_perfil_espectral_modos(Phi_total, DIM_INTERNA)

    # También devolver las listas para gráficos si se necesita
    return {
        'label': 'A',
        'w_tras_entreno': w_tras_entreno,
        'grad_estable': grad_estable,
        'persistencia': persistencia,
        'min_acop': min_acop,
        'perfil': perfil_data,
        'gradientes_test': gradientes_test,
        'w_norma_historia': []  # no guardamos evolución aquí por simplicidad
    }


def experimento_asimetria(seed=43):
    """Test B: Entrena voz, testea ruido blanco"""
    print("  Test B: Asimetría (voz -> ruido)")
    np.random.seed(seed)
    Phi_total = inicializar_campo_total()
    Phi_vel_total = np.zeros_like(Phi_total)
    W = inicializar_plasticidad()
    omega_natural, amort_natural = calcular_frecuencias_naturales(DIM_TOTAL, DIM_INTERNA)

    sr_voz, audio_voz = cargar_audio("Voz_Estudio.wav", duracion=DURACION_ENTRENO)
    sr_ruido, audio_ruido = cargar_audio("Ruido blanco", duracion=DURACION_TEST)
    ventana_muestras = int(sr_voz * VENTANA_FFT_MS / 1000)
    hop_muestras = int(sr_voz * HOP_FFT_MS / 1000)

    # Entrenamiento (alpha=0.05)
    for idx in range(N_PASOS_ENTRENO):
        objetivo = preparar_objetivo_audio(audio_voz, sr_voz, idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
        Phi_total, Phi_vel_total, W = actualizar_campo_total(
            Phi_total, Phi_vel_total, W, objetivo, alpha=0.05,
            omega_natural=omega_natural, amort_natural=amort_natural
        )
    w_tras_entreno = np.mean(np.abs(W))

    # Test (alpha=0.0) con ruido
    gradientes_test = []
    acoplamientos_test = []
    for idx in range(N_PASOS_TEST):
        objetivo = preparar_objetivo_audio(audio_ruido, sr_ruido, idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
        Phi_total, Phi_vel_total, W = actualizar_campo_total(
            Phi_total, Phi_vel_total, W, objetivo, alpha=0.0,
            omega_natural=omega_natural, amort_natural=amort_natural
        )
        gradientes_test.append(calcular_gradiente(Phi_total, DIM_INTERNA))
        acoplamientos_test.append(calcular_acoplamiento(Phi_total, DIM_INTERNA))

    grad_estable = np.mean(gradientes_test[-int(10.0/DT):])
    min_acop = min(acoplamientos_test)
    perfil_data = calcular_perfil_espectral_modos(Phi_total, DIM_INTERNA)

    return {
        'label': 'B',
        'w_tras_entreno': w_tras_entreno,
        'grad_estable': grad_estable,
        'min_acop': min_acop,
        'perfil': perfil_data,
        'gradientes_test': gradientes_test
    }


def experimento_control_ruido(seed=44):
    """Test C: Ruido sin entrenamiento (control)"""
    print("  Test C: Control - Ruido sin entrenamiento")
    np.random.seed(seed)
    Phi_total = inicializar_campo_total()
    Phi_vel_total = np.zeros_like(Phi_total)
    W = inicializar_plasticidad()
    omega_natural, amort_natural = calcular_frecuencias_naturales(DIM_TOTAL, DIM_INTERNA)

    sr_ruido, audio_ruido = cargar_audio("Ruido blanco", duracion=DURACION_TEST)
    ventana_muestras = int(sr_ruido * VENTANA_FFT_MS / 1000)
    hop_muestras = int(sr_ruido * HOP_FFT_MS / 1000)

    gradientes_test = []
    acoplamientos_test = []
    for idx in range(N_PASOS_TEST):
        objetivo = preparar_objetivo_audio(audio_ruido, sr_ruido, idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
        Phi_total, Phi_vel_total, W = actualizar_campo_total(
            Phi_total, Phi_vel_total, W, objetivo, alpha=0.0,
            omega_natural=omega_natural, amort_natural=amort_natural
        )
        gradientes_test.append(calcular_gradiente(Phi_total, DIM_INTERNA))
        acoplamientos_test.append(calcular_acoplamiento(Phi_total, DIM_INTERNA))

    grad_estable = np.mean(gradientes_test[-int(10.0/DT):])
    min_acop = min(acoplamientos_test)
    perfil_data = calcular_perfil_espectral_modos(Phi_total, DIM_INTERNA)

    return {
        'label': 'C',
        'grad_estable': grad_estable,
        'min_acop': min_acop,
        'perfil': perfil_data,
        'gradientes_test': gradientes_test
    }


def experimento_evolucion_w(seed=45):
    """Ejecuta entrenamiento de voz (30s) y guarda evolución de norma de W para gráfico."""
    np.random.seed(seed)
    Phi_total = inicializar_campo_total()
    Phi_vel_total = np.zeros_like(Phi_total)
    W = inicializar_plasticidad()
    omega_natural, amort_natural = calcular_frecuencias_naturales(DIM_TOTAL, DIM_INTERNA)

    sr, audio = cargar_audio("Voz_Estudio.wav", duracion=DURACION_ENTRENO)
    ventana_muestras = int(sr * VENTANA_FFT_MS / 1000)
    hop_muestras = int(sr * HOP_FFT_MS / 1000)

    w_norma = []
    for idx in range(N_PASOS_ENTRENO):
        objetivo = preparar_objetivo_audio(audio, sr, idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
        Phi_total, Phi_vel_total, W = actualizar_campo_total(
            Phi_total, Phi_vel_total, W, objetivo, alpha=0.05,
            omega_natural=omega_natural, amort_natural=amort_natural
        )
        w_norma.append(np.mean(np.abs(W)))
    return w_norma


# ============================================================
# SIMULACIÓN PRINCIPAL
# ============================================================
def main():
    print("=" * 100)
    print("VSTCosmos - v72-B Fase 4: Plasticidad Hebbiana Selectiva + Espectro de Modos Propios")
    print(f"ETA_HEBB = {ETA_HEBB}, TAU_W = {TAU_W}, GAMMA_PLAST = {GAMMA_PLAST}")
    print("Test A: Persistencia (entrena voz, testea voz)")
    print("Test B: Asimetría (entrena voz, testea ruido)")
    print("Test C: Control (ruido sin entrenamiento)")
    print("=" * 100)

    print("\n[Ejecutando experimentos...]")
    test_A = experimento_persistencia(seed=42)
    test_B = experimento_asimetria(seed=43)
    test_C = experimento_control_ruido(seed=44)
    w_evolucion = experimento_evolucion_w(seed=45)

    # Extraer métricas
    grad_A = test_A['grad_estable']
    grad_C = test_C['grad_estable']
    ratio_grad = grad_A / grad_C if grad_C > 0 else 0

    ric_A = test_A['perfil']['riqueza_modal']
    ric_C = test_C['perfil']['riqueza_modal']
    ratio_ric = ric_A / max(ric_C, 1)

    pers_A = test_A['persistencia']
    min_acop = min(test_A['min_acop'], test_B['min_acop'], test_C['min_acop'])

    # Criterios
    criterio_ratio = ratio_grad > 3.0
    criterio_riqueza = ratio_ric > 2.0
    criterio_persistencia = pers_A > 20.0
    criterio_acop = min_acop > 0.01

    print("\n" + "=" * 100)
    print("RESULTADOS")
    print("=" * 100)

    for test, name in [(test_A, 'A'), (test_B, 'B'), (test_C, 'C')]:
        print(f"\n  Test {name}:")
        if 'w_tras_entreno' in test:
            print(f"    W_tras_entreno:           {test['w_tras_entreno']:.4f}")
        print(f"    Gradiente estable (10s):  {test['grad_estable']:.4f}")
        if 'persistencia' in test:
            print(f"    Persistencia (>0.08):     {test['persistencia']:.1f}s")
        print(f"    min(A_sys-env):           {test['min_acop']:.4f}")
        print(f"    --- Perfil espectral ---")
        print(f"    Frecuencia dominante:     modo {test['perfil']['frecuencia_dominante']}")
        print(f"    Riqueza modal:            {test['perfil']['riqueza_modal']} modos activos")
        print(f"    Entropía espectral:       {test['perfil']['entropia_espectral']:.4f}")

    print("\n" + "=" * 100)
    print("CRITERIOS DE DECISIÓN")
    print("=" * 100)
    print(f"    Ratio A/C (gradiente):    {ratio_grad:.3f}  (criterio >3.0: {'✅' if criterio_ratio else '❌'})")
    print(f"    Riqueza modal A/C:        {ratio_ric:.2f}   (criterio >2.0: {'✅' if criterio_riqueza else '❌'})")
    print(f"    Persistencia Test A:      {pers_A:.1f}s (criterio >20s: {'✅' if criterio_persistencia else '❌'})")
    print(f"    Acoplamiento mínimo:      {min_acop:.4f}  (criterio >0.01: {'✅' if criterio_acop else '❌'})")

    print("\n" + "=" * 100)
    print("DECISIÓN")
    print("=" * 100)

    if criterio_ratio and criterio_riqueza and criterio_persistencia and criterio_acop:
        print("\n  ✅ PRE OPERATIVO con evidencia de modos propios específicos.")
        print("     El sistema ha aprendido una relación I·E y sostiene estructura sin estímulo externo.")
        print("     → Proceder a HETA (Homeostasis Estructural Temporal Adaptativa).")
    elif criterio_ratio and not criterio_riqueza:
        print("\n  ✅ ÉXITO PARCIAL: ratio A/C > 3.0, pero riqueza modal A ≈ C.")
        print("     W amplifica gradiente pero no construye estructura espectral diferenciada.")
        print("     → Proceder a HETA con esta limitación documentada.")
    elif 1.5 < ratio_grad < 3.0:
        print("\n  ⚠️ DIRECCIÓN CORRECTA, INSUFICIENTE.")
        print("     W aprendió pero aún débil. Subir ETA_HEBB a 0.08 y repetir.")
    elif ratio_grad < 1.5 and test_A.get('w_tras_entreno', 0) < 0.01:
        print("\n  ❌ BLOQUEO POR UMBRAL DE CORRELACIÓN.")
        print("     W_tras_entreno muy bajo. Bajar umbral_correlacion a 0.05.")
    elif ratio_grad < 1.5 and test_A.get('w_tras_entreno', 0) > 0.05:
        print("\n  ❌ GAMMA_PLAST INSUFICIENTE.")
        print("     W aprendió pero no activa modos. Subir GAMMA_PLAST a 0.02.")
    else:
        print("\n  ❌ VALIDACIÓN INCOMPLETA. Revisar parámetros.")

    # ============================================================
    # VISUALIZACIÓN — seis subgráficos
    # ============================================================
    print("\n[Generando visualizaciones...]")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # [0,0] Evolución gradiente Tests A, B, C superpuestos
    ax = axes[0, 0]
    ax.plot(test_A['gradientes_test'], label='Test A (voz->voz)', alpha=0.7)
    ax.plot(test_B['gradientes_test'], label='Test B (voz->ruido)', alpha=0.7)
    ax.plot(test_C['gradientes_test'], label='Test C (ruido control)', alpha=0.7)
    ax.axhline(y=0.08, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Paso')
    ax.set_ylabel('Gradiente')
    ax.set_title('Evolución del gradiente durante test (alpha=0)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # [0,1] Perfil espectral Test A vs Test C (superpuestos)
    ax = axes[0, 1]
    perfil_A = test_A['perfil']['perfil']
    perfil_C = test_C['perfil']['perfil']
    freqs = np.arange(len(perfil_A))
    ax.plot(freqs, perfil_A, label='Test A (voz aprendida)', linewidth=1.5)
    ax.plot(freqs, perfil_C, label='Test C (ruido control)', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Índice de modo')
    ax.set_ylabel('Potencia espectral')
    ax.set_title('Perfil de modos propios (Test A vs Test C)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # [0,2] Evolución norma de W durante entrenamiento
    ax = axes[0, 2]
    ax.plot(w_evolucion)
    ax.set_xlabel('Paso')
    ax.set_ylabel('Norma de W')
    ax.set_title('Evolución de W durante entrenamiento (voz, alpha=0.05)')
    ax.grid(True, alpha=0.3)

    # [1,0] Gradiente medio por test (barras)
    ax = axes[1, 0]
    labels = ['Test A', 'Test B', 'Test C']
    grad_medios = [test_A['grad_estable'], test_B['grad_estable'], test_C['grad_estable']]
    bars = ax.bar(labels, grad_medios, color=['green', 'blue', 'red'])
    ax.axhline(y=0.08, color='gray', linestyle='--', linewidth=0.8, label='Umbral persistencia')
    ax.set_ylabel('Gradiente estable (últimos 10s)')
    ax.set_title('Gradiente medio por test')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # [1,1] Riqueza modal por test
    ax = axes[1, 1]
    riquezas = [test_A['perfil']['riqueza_modal'],
                test_B['perfil']['riqueza_modal'],
                test_C['perfil']['riqueza_modal']]
    bars = ax.bar(labels, riquezas, color=['green', 'blue', 'red'])
    ax.set_ylabel('Número de modos activos')
    ax.set_title('Riqueza modal (modos > media)')
    ax.grid(True, alpha=0.3)

    # [1,2] Entropía espectral por test
    ax = axes[1, 2]
    entropias = [test_A['perfil']['entropia_espectral'],
                 test_B['perfil']['entropia_espectral'],
                 test_C['perfil']['entropia_espectral']]
    bars = ax.bar(labels, entropias, color=['green', 'blue', 'red'])
    ax.set_ylabel('Entropía espectral')
    ax.set_title('Diversidad espectral de modos')
    ax.grid(True, alpha=0.3)

    plt.suptitle('VSTCosmos v72-B Fase 4 — Plasticidad Hebbiana Selectiva + Modos Propios', fontsize=14)
    plt.tight_layout()
    plt.savefig('v72b_fase4_resultados.png', dpi=150)
    print("  Gráfico guardado: v72b_fase4_resultados.png")

    # Guardar CSV y TXT
    with open('v72b_fase4_resultado.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['test', 'w_tras_entreno', 'gradiente_estable', 'persistencia',
                         'min_acop', 'frecuencia_dominante', 'riqueza_modal', 'entropia_espectral'])
        for test, name in [(test_A, 'A'), (test_B, 'B'), (test_C, 'C')]:
            w_val = test.get('w_tras_entreno', '')
            pers = test.get('persistencia', '')
            writer.writerow([name, w_val, test['grad_estable'], pers,
                             test['min_acop'], test['perfil']['frecuencia_dominante'],
                             test['perfil']['riqueza_modal'], test['perfil']['entropia_espectral']])
    print("  CSV guardado: v72b_fase4_resultado.csv")

    with open('v72b_fase4_resultado.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("VSTCosmos v72-B Fase 4 - Resultado\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"ETA_HEBB = {ETA_HEBB}, TAU_W = {TAU_W}, GAMMA_PLAST = {GAMMA_PLAST}\n")
        f.write(f"Umbral correlación = {UMBRAL_CORRELACION}\n\n")
        f.write(f"Test A (voz->voz): grad_estable={test_A['grad_estable']:.4f}, persistencia={test_A['persistencia']:.1f}s, W={test_A['w_tras_entreno']:.4f}, riqueza={test_A['perfil']['riqueza_modal']}\n")
        f.write(f"Test B (voz->ruido): grad_estable={test_B['grad_estable']:.4f}, riqueza={test_B['perfil']['riqueza_modal']}\n")
        f.write(f"Test C (ruido control): grad_estable={test_C['grad_estable']:.4f}, riqueza={test_C['perfil']['riqueza_modal']}\n\n")
        f.write(f"Criterios:\n")
        f.write(f"  Ratio A/C gradiente: {ratio_grad:.3f} (criterio >3.0)\n")
        f.write(f"  Ratio A/C riqueza: {ratio_ric:.2f} (criterio >2.0)\n")
        f.write(f"  Persistencia Test A: {test_A['persistencia']:.1f}s (criterio >20s)\n")
        f.write(f"  Acoplamiento mínimo: {min_acop:.4f} (criterio >0.01)\n\n")
        f.write("=" * 60 + "\n")
        if criterio_ratio and criterio_riqueza and criterio_persistencia and criterio_acop:
            f.write("DECISION: PRE OPERATIVO con evidencia de modos propios especificos.\n")
            f.write("Proceder a HETA.\n")
        elif criterio_ratio and not criterio_riqueza:
            f.write("DECISION: EXITO PARCIAL - W amplifica gradiente pero no crea modos propios diferenciados.\n")
        elif 1.5 < ratio_grad < 3.0:
            f.write("DECISION: DIRECCION CORRECTA - Insuficiente. Subir ETA_HEBB a 0.08.\n")
        elif ratio_grad < 1.5 and test_A.get('w_tras_entreno', 0) < 0.01:
            f.write("DECISION: BLOQUEO POR UMBRAL - Bajar umbral_correlacion a 0.05.\n")
        elif ratio_grad < 1.5 and test_A.get('w_tras_entreno', 0) > 0.05:
            f.write("DECISION: GAMMA_PLAST INSUFICIENTE - Subir a 0.02.\n")
        else:
            f.write("DECISION: VALIDACION INCOMPLETA - Revisar parametros.\n")
        f.write("=" * 60 + "\n")
    print("  TXT guardado: v72b_fase4_resultado.txt")

    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)


if __name__ == "__main__":
    main()