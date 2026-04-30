#!/usr/bin/env python3
"""
VSTCosmos - v72-B Fase 4.5: Gradiente Espectral Diferencial
Agrega métrica de diferenciación espectral entre region_int y region_aud.
No modifica parámetros. Solo diagnostico adicional.
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import csv
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PARÁMETROS (idénticos a Fase 4)
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

DIFUSION_BASE = 0.15
GANANCIA_REACCION = 0.05

OMEGA_MIN = 0.05
OMEGA_MAX = 0.50
AMORT_MIN = 0.01
AMORT_MAX = 0.08
PHI_EQUILIBRIO = 0.5

ETA_HEBB = 0.05
TAU_W = 0.008
GAMMA_PLAST = 0.01
W_MAX = 1.0
UMBRAL_CORRELACION = 0.1

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

    correlacion = (region_int @ region_aud.T) / DIM_TIME
    correlacion_filtrada = np.where(
        np.abs(correlacion) > UMBRAL_CORRELACION,
        correlacion,
        0.0
    )

    dW = ETA_HEBB * correlacion_filtrada - TAU_W * W
    W_nueva = np.clip(W + dW * dt, -W_MAX, W_MAX)

    patron_esperado = W_nueva @ region_aud
    M_plasticidad = GAMMA_PLAST * (patron_esperado - region_int)

    return W_nueva, M_plasticidad


def actualizar_campo_total(Phi_total, Phi_vel_total, W,
                           objetivo_audio, alpha,
                           omega_natural, amort_natural):
    promedio_local = vecinos(Phi_total)
    difusion = DIFUSION_BASE * (promedio_local - Phi_total)
    
    desviacion = Phi_total - promedio_local
    reaccion = GANANCIA_REACCION * desviacion * (1 - desviacion**2)
    
    term_osc = (-omega_natural**2 * (Phi_total - PHI_EQUILIBRIO)
                - amort_natural * Phi_vel_total)
    
    W_nueva, M_plasticidad = actualizar_hebb_y_plasticidad(Phi_total, W, DT)
    
    M_campo = np.zeros_like(Phi_total)
    M_campo[:DIM_INTERNA, :] = M_plasticidad
    
    dPhi_vel = term_osc + reaccion + difusion + M_campo
    Phi_vel_nueva = Phi_vel_total + DT * dPhi_vel
    Phi_nueva = Phi_total + DT * Phi_vel_nueva
    
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
    region_int = Phi_total[:dim_interna, :]
    perfil = np.zeros(DIM_TIME // 2)
    for banda in range(dim_interna):
        serie = region_int[banda, :]
        serie = serie - np.mean(serie)
        fft = np.fft.rfft(serie)
        potencia = np.abs(fft) ** 2
        perfil += potencia[:DIM_TIME // 2]
    perfil = perfil / dim_interna
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


def calcular_gradiente_espectral_diferencial(Phi_total, dim_interna):
    """
    NUEVA MÉTRICA: diferencia espectral entre region_int y region_aud.
    """
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
    
    diferencia_espectral = np.abs(perfil_int - perfil_aud)
    gradiente_espectral_medio = np.mean(diferencia_espectral)
    modo_max_diferencia = int(np.argmax(diferencia_espectral))
    
    return {
        'perfil_int': perfil_int,
        'perfil_aud': perfil_aud,
        'diferencia_espectral': diferencia_espectral,
        'gradiente_espectral': float(gradiente_espectral_medio),
        'modo_max_diferencia': modo_max_diferencia
    }


# ============================================================
# EXPERIMENTOS
# ============================================================
def experimento_persistencia(seed=42):
    print("  Test A: Persistencia (voz -> voz)")
    np.random.seed(seed)
    Phi_total = inicializar_campo_total()
    Phi_vel_total = np.zeros_like(Phi_total)
    W = inicializar_plasticidad()
    omega_natural, amort_natural = calcular_frecuencias_naturales(DIM_TOTAL, DIM_INTERNA)

    sr, audio = cargar_audio("Voz_Estudio.wav", duracion=DURACION_ENTRENO + DURACION_TEST)
    ventana_muestras = int(sr * VENTANA_FFT_MS / 1000)
    hop_muestras = int(sr * HOP_FFT_MS / 1000)

    for idx in range(N_PASOS_ENTRENO):
        objetivo = preparar_objetivo_audio(audio, sr, idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
        Phi_total, Phi_vel_total, W = actualizar_campo_total(
            Phi_total, Phi_vel_total, W, objetivo, alpha=0.05,
            omega_natural=omega_natural, amort_natural=amort_natural
        )
    w_tras_entreno = np.mean(np.abs(W))

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
    perfil_modos = calcular_perfil_espectral_modos(Phi_total, DIM_INTERNA)
    grad_espectral = calcular_gradiente_espectral_diferencial(Phi_total, DIM_INTERNA)

    return {
        'w_tras_entreno': w_tras_entreno,
        'grad_estable': grad_estable,
        'persistencia': persistencia,
        'min_acop': min_acop,
        'perfil_modos': perfil_modos,
        'grad_espectral': grad_espectral,
        'gradientes_test': gradientes_test
    }


def experimento_asimetria(seed=43):
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

    for idx in range(N_PASOS_ENTRENO):
        objetivo = preparar_objetivo_audio(audio_voz, sr_voz, idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
        Phi_total, Phi_vel_total, W = actualizar_campo_total(
            Phi_total, Phi_vel_total, W, objetivo, alpha=0.05,
            omega_natural=omega_natural, amort_natural=amort_natural
        )
    w_tras_entreno = np.mean(np.abs(W))

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
    perfil_modos = calcular_perfil_espectral_modos(Phi_total, DIM_INTERNA)
    grad_espectral = calcular_gradiente_espectral_diferencial(Phi_total, DIM_INTERNA)

    return {
        'w_tras_entreno': w_tras_entreno,
        'grad_estable': grad_estable,
        'min_acop': min_acop,
        'perfil_modos': perfil_modos,
        'grad_espectral': grad_espectral,
        'gradientes_test': gradientes_test
    }


def experimento_control_ruido(seed=44):
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
    perfil_modos = calcular_perfil_espectral_modos(Phi_total, DIM_INTERNA)
    grad_espectral = calcular_gradiente_espectral_diferencial(Phi_total, DIM_INTERNA)

    return {
        'grad_estable': grad_estable,
        'min_acop': min_acop,
        'perfil_modos': perfil_modos,
        'grad_espectral': grad_espectral,
        'gradientes_test': gradientes_test
    }


def main():
    print("=" * 100)
    print("VSTCosmos - v72-B Fase 4.5: Gradiente Espectral Diferencial")
    print(f"Parámetros: ETA_HEBB={ETA_HEBB}, TAU_W={TAU_W}, GAMMA_PLAST={GAMMA_PLAST}")
    print("Test A: Persistencia | Test B: Asimetría | Test C: Control")
    print("=" * 100)

    print("\n[Ejecutando experimentos...]")
    A = experimento_persistencia(42)
    B = experimento_asimetria(43)
    C = experimento_control_ruido(44)

    print("\n" + "=" * 100)
    print("RESULTADOS")
    print("=" * 100)

    for test, label in [(A, 'A'), (B, 'B'), (C, 'C')]:
        print(f"\n  Test {label}:")
        if 'w_tras_entreno' in test:
            print(f"    W_tras_entreno:           {test['w_tras_entreno']:.4f}")
        print(f"    Gradiente estable (10s):  {test['grad_estable']:.4f}")
        if 'persistencia' in test:
            print(f"    Persistencia (>0.08):     {test['persistencia']:.1f}s")
        print(f"    Acoplamiento mínimo:      {test['min_acop']:.4f}")
        print(f"    --- Perfil de modos (region_int) ---")
        print(f"    Frecuencia dominante:     modo {test['perfil_modos']['frecuencia_dominante']}")
        print(f"    Riqueza modal:            {test['perfil_modos']['riqueza_modal']} modos")
        print(f"    Entropía espectral:       {test['perfil_modos']['entropia_espectral']:.4f}")
        print(f"    --- Gradiente espectral diferencial ---")
        print(f"    Diferencia espectral media: {test['grad_espectral']['gradiente_espectral']:.6f}")
        print(f"    Modo de máxima diferencia:  {test['grad_espectral']['modo_max_diferencia']}")

    # ============================================================
    # CRITERIOS ACTUALIZADOS
    # ============================================================
    ratio_grad = A['grad_estable'] / C['grad_estable']
    ratio_grad_espectral = A['grad_espectral']['gradiente_espectral'] / C['grad_espectral']['gradiente_espectral']

    print("\n" + "=" * 100)
    print("DIAGNÓSTICO")
    print("=" * 100)
    print(f"  Ratio A/C (gradiente medio):        {ratio_grad:.3f}")
    print(f"  Ratio A/C (gradiente espectral):    {ratio_grad_espectral:.3f}")
    print(f"  Modo dominante A vs C:              {A['perfil_modos']['frecuencia_dominante']} vs {C['perfil_modos']['frecuencia_dominante']}")
    print(f"  Riqueza modal A vs C:               {A['perfil_modos']['riqueza_modal']} vs {C['perfil_modos']['riqueza_modal']}")

    print("\n" + "=" * 100)
    print("CONCLUSIÓN")
    print("=" * 100)

    if ratio_grad_espectral > 2.0 and A['perfil_modos']['frecuencia_dominante'] != C['perfil_modos']['frecuencia_dominante']:
        print("\n  ✅ W está creando diferenciación espectral efectiva.")
        print("     El gradiente espectral diferencial es alto.")
        print("     La frecuencia dominante de region_int ha cambiado.")
        print("     → Subir GAMMA_PLAST a 0.02 para amplificar el efecto en el gradiente medio.")
    elif ratio_grad_espectral > 1.5:
        print("\n  📈 W está creando diferenciación espectral incipiente.")
        print("     El gradiente espectral diferencial aumenta pero aún es débil.")
        print("     → Subir GAMMA_PLAST a 0.02 e iterar.")
    else:
        print("\n  ❌ La plásticidad hebbiana no está creando diferenciación espectral detectable.")
        print("     → Aumentar ETA_HEBB a 0.08 o bajar umbral de correlación a 0.05.")

    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)


if __name__ == "__main__":
    main()