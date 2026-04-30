#!/usr/bin/env python3
"""
VSTCosmos - v72-B Fase 3: Plasticidad Hebbiana
Mecanismo: matriz W aprende correlaciones entre región interna y auditiva.
Durante test (alpha=0.0), W reconstruye el patrón aprendido.
No hay histéresis. No hay inercia. Solo plasticidad.
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

# Parámetros de dinámica
DIFUSION_BASE = 0.15
GANANCIA_REACCION = 0.05

OMEGA_MIN = 0.05
OMEGA_MAX = 0.50
AMORT_MIN = 0.01
AMORT_MAX = 0.08
PHI_EQUILIBRIO = 0.5

# Parámetros de plasticidad hebbiana
ETA_HEBB = 0.02        # tasa de aprendizaje hebbiano
TAU_W = 0.005          # decaimiento de pesos
GAMMA_PLAST = 0.15     # fuerza del término plástico sobre región interna
W_MAX = 1.0            # límite para estabilidad numérica

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
    """W: matriz de correlaciones aprendidas entre región interna y auditiva."""
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
    """
    Dos operaciones:
    1. Regla de Hebb: W aprende correlaciones entre region_int y region_aud.
    2. Término plástico: usa W para reconstituir region_int desde region_aud.
    """
    region_int = Phi_total[:DIM_INTERNA, :]
    region_aud = Phi_total[DIM_INTERNA:, :]

    # --- Regla de Hebb ---
    # Producto externo promediado sobre DIM_TIME
    correlacion = (region_int @ region_aud.T) / DIM_TIME
    dW = ETA_HEBB * correlacion - TAU_W * W
    W_nueva = np.clip(W + dW * dt, -W_MAX, W_MAX)

    # --- Término plástico (actúa solo sobre region_int) ---
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
    
    # M_plasticidad solo actúa sobre region_int
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


# ============================================================
# EXPERIMENTOS
# ============================================================
def experimento_fase3(estimulo_entrenamiento, estimulo_test,
                      duracion_entreno=30.0, duracion_test=30.0):
    """
    Etapa 1 — Entrenamiento (alpha=0.05): W aprende correlaciones.
    Etapa 2 — Test (alpha=0.0): Campo expuesto a estimulo_test sin sesgo.
    """
    print(f"    Entrenando: {estimulo_entrenamiento}")
    print(f"    Testeando: {estimulo_test}")
    
    Phi_total = inicializar_campo_total()
    Phi_vel_total = np.zeros_like(Phi_total)
    W = inicializar_plasticidad()
    omega_natural, amort_natural = calcular_frecuencias_naturales(DIM_TOTAL, DIM_INTERNA)
    
    sr, audio_e = cargar_audio(estimulo_entrenamiento, duracion=duracion_entreno)
    ventana_muestras = int(sr * VENTANA_FFT_MS / 1000)
    hop_muestras = int(sr * HOP_FFT_MS / 1000)
    
    gradientes_entreno = []
    w_norma_historia = []
    acoplamientos_entreno = []
    
    # --- Etapa 1: Entrenamiento ---
    for idx in range(N_PASOS_ENTRENO):
        objetivo = preparar_objetivo_audio(audio_e, sr, idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
        Phi_total, Phi_vel_total, W = actualizar_campo_total(
            Phi_total, Phi_vel_total, W,
            objetivo, alpha=0.05,
            omega_natural=omega_natural, amort_natural=amort_natural
        )
        g = calcular_gradiente(Phi_total, DIM_INTERNA)
        a = calcular_acoplamiento(Phi_total, DIM_INTERNA)
        gradientes_entreno.append(g)
        acoplamientos_entreno.append(a)
        w_norma_historia.append(np.mean(np.abs(W)))
    
    w_tras_entreno = np.mean(np.abs(W))
    print(f"      W media tras entrenamiento: {w_tras_entreno:.4f}")
    
    # --- Etapa 2: Test (alpha=0.0) ---
    sr_t, audio_t = cargar_audio(estimulo_test, duracion=duracion_test)
    gradientes_test = []
    acoplamientos_test = []
    
    for idx in range(N_PASOS_TEST):
        objetivo = preparar_objetivo_audio(audio_t, sr_t, idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
        Phi_total, Phi_vel_total, W = actualizar_campo_total(
            Phi_total, Phi_vel_total, W,
            objetivo, alpha=0.0,
            omega_natural=omega_natural, amort_natural=amort_natural
        )
        g = calcular_gradiente(Phi_total, DIM_INTERNA)
        a = calcular_acoplamiento(Phi_total, DIM_INTERNA)
        gradientes_test.append(g)
        acoplamientos_test.append(a)
    
    gradiente_pico = max(gradientes_test)
    gradiente_estable = np.mean(gradientes_test[-int(10.0/DT):])
    acoplamiento_min_test = min(acoplamientos_test)
    persistencia_s = sum(1 for g in gradientes_test if g > 0.08) * DT
    
    print(f"      Gradiente pico: {gradiente_pico:.4f}")
    print(f"      Gradiente estable: {gradiente_estable:.4f}")
    print(f"      Persistencia (>0.08): {persistencia_s:.1f}s")
    
    return {
        'w_tras_entreno': w_tras_entreno,
        'gradiente_pico': gradiente_pico,
        'gradiente_estable': gradiente_estable,
        'persistencia_s': persistencia_s,
        'acoplamiento_min': acoplamiento_min_test,
        'gradientes_entreno': gradientes_entreno,
        'gradientes_test': gradientes_test,
        'w_norma_historia': w_norma_historia,
        'acoplamientos_entreno': acoplamientos_entreno,
        'acoplamientos_test': acoplamientos_test
    }


def experimento_fase3_sin_entreno(estimulo_test):
    """Sin etapa de entrenamiento: W=0 desde inicio, alpha=0.0"""
    print(f"    Testeando sin entrenamiento: {estimulo_test}")
    
    Phi_total = inicializar_campo_total()
    Phi_vel_total = np.zeros_like(Phi_total)
    W = inicializar_plasticidad()
    omega_natural, amort_natural = calcular_frecuencias_naturales(DIM_TOTAL, DIM_INTERNA)
    
    sr_t, audio_t = cargar_audio(estimulo_test, duracion=DURACION_TEST)
    ventana_muestras = int(sr_t * VENTANA_FFT_MS / 1000)
    hop_muestras = int(sr_t * HOP_FFT_MS / 1000)
    
    gradientes_test = []
    acoplamientos_test = []
    
    for idx in range(N_PASOS_TEST):
        objetivo = preparar_objetivo_audio(audio_t, sr_t, idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
        Phi_total, Phi_vel_total, W = actualizar_campo_total(
            Phi_total, Phi_vel_total, W,
            objetivo, alpha=0.0,
            omega_natural=omega_natural, amort_natural=amort_natural
        )
        g = calcular_gradiente(Phi_total, DIM_INTERNA)
        a = calcular_acoplamiento(Phi_total, DIM_INTERNA)
        gradientes_test.append(g)
        acoplamientos_test.append(a)
    
    gradiente_estable = np.mean(gradientes_test[-int(10.0/DT):])
    acoplamiento_min = min(acoplamientos_test)
    
    print(f"      Gradiente estable: {gradiente_estable:.4f}")
    
    return {
        'gradiente_estable': gradiente_estable,
        'acoplamiento_min': acoplamiento_min,
        'gradientes_test': gradientes_test
    }


def main():
    print("=" * 100)
    print("VSTCosmos - v72-B Fase 3: Plasticidad Hebbiana")
    print(f"ETA_HEBB = {ETA_HEBB}, TAU_W = {TAU_W}, GAMMA_PLAST = {GAMMA_PLAST}")
    print("Test A: Persistencia | Test B: Asimetría | Test C: Especificidad | Test D: Acoplamiento")
    print("=" * 100)
    
    # Test A: Persistencia (entrena voz, testa voz)
    print("\n[Test A] Persistencia")
    test_A = experimento_fase3("Voz_Estudio.wav", "Voz_Estudio.wav")
    
    # Test B: Asimetría (entrena voz, testa ruido)
    print("\n[Test B] Asimetría")
    test_B = experimento_fase3("Voz_Estudio.wav", "Ruido blanco")
    
    # Test C: Especificidad (sin entrenamiento, testa ruido)
    print("\n[Test C] Especificidad")
    test_C = experimento_fase3_sin_entreno("Ruido blanco")
    
    # Test D: Acoplamiento (verificar min acoplamiento en Tests A y B)
    print("\n[Test D] Acoplamiento constitutivo")
    min_acop_A = test_A['acoplamiento_min']
    min_acop_B = test_B['acoplamiento_min']
    min_acop_C = test_C['acoplamiento_min']
    print(f"    Test A: min_acop = {min_acop_A:.6f}")
    print(f"    Test B: min_acop = {min_acop_B:.6f}")
    print(f"    Test C: min_acop = {min_acop_C:.6f}")
    
    # ============================================================
    # CRITERIOS DE DECISIÓN
    # ============================================================
    print("\n" + "=" * 100)
    print("CRITERIOS DE DECISIÓN")
    print("=" * 100)
    
    # Criterio A: Persistencia
    criterio_A = (test_A['gradiente_estable'] > 0.15 and test_A['persistencia_s'] > 20.0)
    
    # Criterio B: Asimetría
    criterio_B = test_B['gradiente_estable'] > 0.08
    
    # Criterio C: Especificidad
    criterio_C = test_C['gradiente_estable'] < 0.03
    
    # Criterio D: Acoplamiento
    criterio_D = (min_acop_A > 0.01 and min_acop_B > 0.01)
    
    print(f"\n  Test A (Persistencia):")
    print(f"    grad_estable = {test_A['gradiente_estable']:.4f} (>0.15: {'✅' if test_A['gradiente_estable'] > 0.15 else '❌'})")
    print(f"    persistencia = {test_A['persistencia_s']:.1f}s (>20s: {'✅' if test_A['persistencia_s'] > 20 else '❌'})")
    print(f"    → {'✅' if criterio_A else '❌'}")
    
    print(f"\n  Test B (Asimetría):")
    print(f"    grad_estable ruido->voz = {test_B['gradiente_estable']:.4f} (>0.08: {'✅' if test_B['gradiente_estable'] > 0.08 else '❌'})")
    print(f"    → {'✅' if criterio_B else '❌'}")
    
    print(f"\n  Test C (Especificidad):")
    print(f"    grad_estable sin entreno = {test_C['gradiente_estable']:.4f} (<0.03: {'✅' if test_C['gradiente_estable'] < 0.03 else '❌'})")
    print(f"    → {'✅' if criterio_C else '❌'}")
    
    print(f"\n  Test D (Acoplamiento):")
    print(f"    min_acop A = {min_acop_A:.6f} (>0.01: {'✅' if min_acop_A > 0.01 else '❌'})")
    print(f"    min_acop B = {min_acop_B:.6f} (>0.01: {'✅' if min_acop_B > 0.01 else '❌'})")
    print(f"    → {'✅' if criterio_D else '❌'}")
    
    print("\n" + "=" * 100)
    print("DECISIÓN")
    print("=" * 100)
    
    if criterio_A and criterio_B and criterio_C and criterio_D:
        print("\n  ✅ VALIDACIÓN COMPLETA: v72-B Fase 3 exitosa")
        print("     El campo aprendió una relación I·E y la sostiene sin estímulo externo.")
        print("     → Primera implementación de PRE en VSTCosmos")
    elif criterio_A and criterio_B:
        print("\n  ✅ VALIDACIÓN PARCIAL: Persistencia y Asimetría OK")
        print("     El campo aprendió, pero hay problemas de especificidad o acoplamiento.")
        if not criterio_C:
            print("     → W inventa estructura sin entrenamiento. Subir TAU_W o bajar ETA_HEBB")
        if not criterio_D:
            print("     → GAMMA_PLAST demasiado fuerte. Reducir a 0.1")
    elif criterio_A:
        print("\n  ⚠️ CRITERIO PARCIAL: Solo Persistencia")
        print("     W aprendió el patrón de voz, pero no generaliza a ruido en región auditiva.")
        print("     → GAMMA_PLAST insuficiente. Subir a 0.25")
    elif test_A['w_tras_entreno'] < 0.01:
        print("\n  ❌ NO APRENDIZAJE")
        print("     W_tras_entreno < 0.01. No hubo aprendizaje durante entrenamiento.")
        print("     → Subir ETA_HEBB a 0.05")
    else:
        print("\n  ❌ VALIDACIÓN INCOMPLETA")
        print("     El mecanismo no produce persistencia ni asimetría.")
        print("     → Revisar parámetros o arquitectura")
    
    # ============================================================
    # VISUALIZACIÓN
    # ============================================================
    print("\n[Generando visualizaciones...]")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Gráfico 1: Evolución del gradiente durante entrenamiento
    ax = axes[0, 0]
    ax.plot(test_A['gradientes_entreno'])
    ax.set_xlabel('Paso')
    ax.set_ylabel('Gradiente')
    ax.set_title('Test A: Gradiente durante entrenamiento')
    ax.grid(True, alpha=0.3)
    
    # Gráfico 2: Gradiente durante test (persistencia)
    ax = axes[0, 1]
    ax.plot(test_A['gradientes_test'])
    ax.axhline(y=0.15, color='r', linestyle='--', label='Umbral persistencia')
    ax.set_xlabel('Paso')
    ax.set_ylabel('Gradiente')
    ax.set_title(f'Test A: Test (voz, alpha=0)\nPersistencia: {test_A["persistencia_s"]:.1f}s')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 3: Test B (voz entrenada, test ruido)
    ax = axes[0, 2]
    ax.plot(test_B['gradientes_test'])
    ax.axhline(y=0.08, color='r', linestyle='--', label='Umbral asimetría')
    ax.set_xlabel('Paso')
    ax.set_ylabel('Gradiente')
    ax.set_title(f'Test B: Test (ruido, alpha=0)\nGrad_estable: {test_B["gradiente_estable"]:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 4: Evolución de W durante entrenamiento
    ax = axes[1, 0]
    ax.plot(test_A['w_norma_historia'])
    ax.axhline(y=test_A['w_tras_entreno'], color='r', linestyle='--', label='W final')
    ax.set_xlabel('Paso')
    ax.set_ylabel('Norma de W')
    ax.set_title(f'Evolución de W (Test A)\nW final: {test_A["w_tras_entreno"]:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 5: Test C (sin entrenamiento)
    ax = axes[1, 1]
    ax.plot(test_C['gradientes_test'])
    ax.axhline(y=0.03, color='r', linestyle='--', label='Umbral especificidad')
    ax.set_xlabel('Paso')
    ax.set_ylabel('Gradiente')
    ax.set_title(f'Test C: Sin entrenamiento\nGrad_estable: {test_C["gradiente_estable"]:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 6: Acoplamiento durante test A y B
    ax = axes[1, 2]
    ax.plot(test_A['acoplamientos_test'], label='Test A (voz->voz)', alpha=0.7)
    ax.plot(test_B['acoplamientos_test'], label='Test B (voz->ruido)', alpha=0.7)
    ax.axhline(y=0.01, color='r', linestyle='--', label='Umbral acoplamiento')
    ax.set_xlabel('Paso')
    ax.set_ylabel('A_sys-env')
    ax.set_title('Acoplamiento durante test')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('VSTCosmos v72-B Fase 3 - Plasticidad Hebbiana', fontsize=14)
    plt.tight_layout()
    plt.savefig('v72b_fase3_resultados.png', dpi=150)
    print("  Gráfico guardado: v72b_fase3_resultados.png")
    
    # Guardar CSV
    with open('v72b_fase3_resultado.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['test', 'metric', 'value', 'criterio'])
        writer.writerow(['A', 'grad_estable', f'{test_A["gradiente_estable"]:.4f}', '>0.15'])
        writer.writerow(['A', 'persistencia_s', f'{test_A["persistencia_s"]:.1f}', '>20'])
        writer.writerow(['A', 'w_tras_entreno', f'{test_A["w_tras_entreno"]:.4f}', '>0.01'])
        writer.writerow(['B', 'grad_estable_ruido', f'{test_B["gradiente_estable"]:.4f}', '>0.08'])
        writer.writerow(['C', 'grad_estable_sin_entreno', f'{test_C["gradiente_estable"]:.4f}', '<0.03'])
        writer.writerow(['D', 'min_acop_A', f'{test_A["acoplamiento_min"]:.6f}', '>0.01'])
        writer.writerow(['D', 'min_acop_B', f'{test_B["acoplamiento_min"]:.6f}', '>0.01'])
        writer.writerow(['decision', 'v72_b_fase3', 'VALIDADO' if (criterio_A and criterio_B and criterio_C and criterio_D) else 'PENDIENTE', ''])
    
    print("  CSV guardado: v72b_fase3_resultado.csv")
    
    # Guardar TXT
    with open('v72b_fase3_resultado.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("VSTCosmos v72-B Fase 3 - Resultado\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"ETA_HEBB = {ETA_HEBB}\n")
        f.write(f"TAU_W = {TAU_W}\n")
        f.write(f"GAMMA_PLAST = {GAMMA_PLAST}\n\n")
        f.write(f"Test A (Persistencia):\n")
        f.write(f"  grad_estable = {test_A['gradiente_estable']:.4f}\n")
        f.write(f"  persistencia = {test_A['persistencia_s']:.1f}s\n")
        f.write(f"  w_tras_entreno = {test_A['w_tras_entreno']:.4f}\n\n")
        f.write(f"Test B (Asimetría):\n")
        f.write(f"  grad_estable ruido->voz = {test_B['gradiente_estable']:.4f}\n\n")
        f.write(f"Test C (Especificidad):\n")
        f.write(f"  grad_estable sin entreno = {test_C['gradiente_estable']:.4f}\n\n")
        f.write(f"Test D (Acoplamiento):\n")
        f.write(f"  min_acop A = {test_A['acoplamiento_min']:.6f}\n")
        f.write(f"  min_acop B = {test_B['acoplamiento_min']:.6f}\n\n")
        f.write("=" * 60 + "\n")
        if criterio_A and criterio_B and criterio_C and criterio_D:
            f.write("DECISION: VALIDADO - v72-B Fase 3 exitosa\n")
            f.write("Primera implementacion de PRE en VSTCosmos.\n")
        else:
            f.write("DECISION: PENDIENTE - Ajustar parametros\n")
            if not criterio_A:
                f.write("- Aumentar GAMMA_PLAST a 0.25 o ETA_HEBB a 0.05\n")
            if not criterio_B:
                f.write("- Aumentar GAMMA_PLAST a 0.25\n")
            if not criterio_C:
                f.write("- Subir TAU_W a 0.01 o bajar ETA_HEBB a 0.01\n")
            if not criterio_D:
                f.write("- Reducir GAMMA_PLAST a 0.1\n")
        f.write("=" * 60 + "\n")
    
    print("  TXT guardado: v72b_fase3_resultado.txt")
    
    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)


if __name__ == "__main__":
    main()