#!/usr/bin/env python3
"""
VSTCosmos - v72-B Fase 1.5: Diagnóstico de Gradiente Máximo
Ejecutar ruido->voz con alpha=0.0 y registrar:
- gradiente máximo alcanzado
- gradiente medio en últimos 5 segundos
- si en algún momento superó 0.05
"""

import numpy as np
import scipy.io.wavfile as wav
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
DURACION_TRANSICION = 20.0
N_PASOS_TRANSICION = int(DURACION_TRANSICION / DT)

# Parámetros de dinámica
DIFUSION_BASE = 0.15
GANANCIA_REACCION = 0.05

OMEGA_MIN = 0.05
OMEGA_MAX = 0.50
AMORT_MIN = 0.01
AMORT_MAX = 0.08
PHI_EQUILIBRIO = 0.5

# Parámetros de memoria estructural (así estaban en v72-B Fase 1)
GAMMA_MEMORIA = 0.08
BETA_MEMORIA = 2.0
TAU_HISTORIA = 0.0005

# Parámetros de FFT
VENTANA_FFT_MS = 25
HOP_FFT_MS = 10
F_MIN = 80
F_MAX = 8000

LIMITE_MIN = 0.0
LIMITE_MAX = 1.0


# ============================================================
# FUNCIONES
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


def inicializar_historia():
    return inicializar_campo_total()


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


def calcular_memoria(Phi_total, Phi_historia):
    diferencia = Phi_total - Phi_historia
    M = -GAMMA_MEMORIA * np.sign(diferencia) * np.abs(diferencia) ** BETA_MEMORIA
    return M


def actualizar_historia(Phi_historia, Phi_total):
    return (1 - TAU_HISTORIA) * Phi_historia + TAU_HISTORIA * Phi_total


def actualizar_campo_total(Phi_total, Phi_vel_total, Phi_historia,
                           objetivo_audio, alpha,
                           omega_natural, amort_natural):
    promedio_local = vecinos(Phi_total)
    difusion = DIFUSION_BASE * (promedio_local - Phi_total)
    
    desviacion = Phi_total - promedio_local
    reaccion = GANANCIA_REACCION * desviacion * (1 - desviacion**2)
    
    term_osc = (-omega_natural**2 * (Phi_total - PHI_EQUILIBRIO)
                - amort_natural * Phi_vel_total)
    
    M = calcular_memoria(Phi_total, Phi_historia)
    
    dPhi_vel = term_osc + reaccion + difusion + M
    Phi_vel_nueva = Phi_vel_total + DT * dPhi_vel
    Phi_nueva = Phi_total + DT * Phi_vel_nueva
    
    if alpha > 0:
        region_auditiva_nueva = Phi_nueva[DIM_INTERNA:, :]
        region_auditiva_nueva = (1 - alpha) * region_auditiva_nueva + alpha * objetivo_audio
        Phi_nueva[DIM_INTERNA:, :] = region_auditiva_nueva
    
    return (np.clip(Phi_nueva, LIMITE_MIN, LIMITE_MAX),
            np.clip(Phi_vel_nueva, -5.0, 5.0))


def calcular_gradiente(Phi_total, dim_interna):
    region_int = Phi_total[:dim_interna, :]
    region_aud = Phi_total[dim_interna:, :]
    return np.mean(np.abs(region_int - region_aud))


def calcular_acoplamiento(Phi_total, dim_interna):
    region_int = Phi_total[:dim_interna, :]
    region_aud = Phi_total[dim_interna:, :]
    return float(np.mean(region_int * region_aud))


def test_diagnostico():
    """Versión diagnóstica que registra gradiente, M y acoplamiento."""
    print("=" * 100)
    print("VSTCosmos - v72-B Fase 1.5: Diagnóstico de Gradiente Máximo")
    print(f"GAMMA_MEMORIA = {GAMMA_MEMORIA}, BETA_MEMORIA = {BETA_MEMORIA}, TAU_HISTORIA = {TAU_HISTORIA}")
    print("Transición: Ruido blanco -> Voz_Estudio.wav con alpha=0.0")
    print("=" * 100)
    
    est_a = "Ruido blanco"
    est_b = "Voz_Estudio.wav"
    alpha = 0.0
    
    Phi_total = inicializar_campo_total()
    Phi_vel_total = np.zeros_like(Phi_total)
    Phi_historia = inicializar_historia()
    omega_natural, amort_natural = calcular_frecuencias_naturales(DIM_TOTAL, DIM_INTERNA)
    
    sr_a, audio_a = cargar_audio(est_a, duracion=DURACION_TRANSICION)
    sr_b, audio_b = cargar_audio(est_b, duracion=DURACION_TRANSICION)
    
    ventana_muestras = int(sr_a * VENTANA_FFT_MS / 1000)
    hop_muestras = int(sr_a * HOP_FFT_MS / 1000)
    
    gradientes = []
    tiempos = []
    
    print("\n[1] Segmento A: Ruido blanco (alpha=0.0)")
    for idx in range(N_PASOS_TRANSICION):
        obj = preparar_objetivo_audio(audio_a, sr_a, idx, ventana_muestras, hop_muestras,
                                      DIM_AUDITIVA, DIM_TIME)
        Phi_total, Phi_vel_total = actualizar_campo_total(
            Phi_total, Phi_vel_total, Phi_historia, obj, alpha,
            omega_natural, amort_natural
        )
        Phi_historia = actualizar_historia(Phi_historia, Phi_total)
        grad = calcular_gradiente(Phi_total, DIM_INTERNA)
        gradientes.append(grad)
        tiempos.append(idx * DT)
    
    print(f"    Gradiente final (ruido): {gradientes[-1]:.4f}")
    print(f"    Gradiente medio (ruido): {np.mean(gradientes[-100:]):.4f}")
    
    print("\n[2] Segmento B: Voz (alpha=0.0)")
    for idx in range(N_PASOS_TRANSICION):
        obj = preparar_objetivo_audio(audio_b, sr_b, idx, ventana_muestras, hop_muestras,
                                      DIM_AUDITIVA, DIM_TIME)
        Phi_total, Phi_vel_total = actualizar_campo_total(
            Phi_total, Phi_vel_total, Phi_historia, obj, alpha,
            omega_natural, amort_natural
        )
        Phi_historia = actualizar_historia(Phi_historia, Phi_total)
        grad = calcular_gradiente(Phi_total, DIM_INTERNA)
        gradientes.append(grad)
        tiempos.append((N_PASOS_TRANSICION + idx) * DT)
    
    print(f"    Gradiente final (voz): {gradientes[-1]:.4f}")
    print(f"    Gradiente medio (voz, últimos 5s): {np.mean(gradientes[-int(5/DT):]):.4f}")
    
    gradiente_maximo = np.max(gradientes)
    gradiente_medio_voz = np.mean(gradientes[-int(5/DT):])
    gradiente_supero_05 = gradiente_maximo > 0.05
    
    print("\n" + "=" * 100)
    print("RESULTADO DEL DIAGNÓSTICO")
    print("=" * 100)
    print(f"  Gradiente máximo alcanzado: {gradiente_maximo:.6f}")
    print(f"  Gradiente medio (voz, últimos 5s): {gradiente_medio_voz:.6f}")
    print(f"  ¿Superó 0.05? {'SÍ' if gradiente_supero_05 else 'NO'}")
    
    print("\n" + "=" * 100)
    print("CONCLUSIÓN")
    print("=" * 100)
    
    if gradiente_maximo < 0.03:
        print("  ❌ CASO 1: gradiente máximo < 0.03")
        print("     La histéresis simple NO produce diferenciación sin sesgo externo.")
        print("     El problema es ARQUITECTÓNICO.")
        print("     → v72-B Fase 2 debe rediseñar el mecanismo de memoria,")
        print("       no solo ajustar parámetros.")
    elif gradiente_maximo < 0.05:
        print("  ⚠️ CASO INTERMEDIO: 0.03 ≤ gradiente < 0.05")
        print("     El campo se diferencia débilmente.")
        print("     Ajustar parámetros podría funcionar, pero el margen es pequeño.")
        print("     → Probar GAMMA_MEMORIA=0.15, BETA_MEMORIA=2.5, TAU_HISTORIA=0.0002")
    else:
        print("  ✅ CASO 2: gradiente máximo ≥ 0.05")
        print("     El campo sí se diferencia, pero no alcanza el umbral del 90%.")
        print("     El problema es de INTENSIDAD — ajustar parámetros puede funcionar.")
        print("     → Aumentar GAMMA_MEMORIA a 0.12-0.15")
        print("     → Aumentar BETA_MEMORIA a 2.5")
        print("     → Reducir TAU_HISTORIA a 0.0002")
    
    return {
        'gradiente_maximo': gradiente_maximo,
        'gradiente_medio_voz': gradiente_medio_voz,
        'gradiente_supero_05': gradiente_supero_05,
        'gradientes': gradientes,
        'tiempos': tiempos
    }


def main():
    resultado = test_diagnostico()
    
    # Visualización rápida
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(resultado['tiempos'], resultado['gradientes'])
    ax.axvline(x=DURACION_TRANSICION, color='r', linestyle='--', label='Cambio de estímulo')
    ax.axhline(y=0.05, color='g', linestyle='--', label='Umbral 0.05')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Gradiente')
    ax.set_title(f'Gradiente máximo: {resultado["gradiente_maximo"]:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('v72b_fase1.5_diagnostico.png', dpi=150)
    print("\n  Gráfico guardado: v72b_fase1.5_diagnostico.png")


if __name__ == "__main__":
    main()