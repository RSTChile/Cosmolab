#!/usr/bin/env python3
"""
VSTCosmos - v72c Fase 5: Mecanismos Complementarios
Arquitectura que combina:
1. Mecanismo Hebbiano (acoplamiento resonante): region_int sigue a region_aud
2. Mecanismo Atractor (memoria interna): region_int tiende a su patrón histórico

Tres fases en una ejecución sin reinicializar el campo:
- Entrenamiento (alpha=0.05, voz): W aprende, Phi_int_historia acumula atractor
- Test acoplado (alpha=0.0, voz): ambos mecanismos activos
- Test desacoplado (alpha=0.0, ruido): W intenta reconstruir desde señal equivocada,
  Phi_int_historia resiste el cambio
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
DURACION_TEST_ACOPLADO = 20.0
DURACION_TEST_DESACOPLADO = 20.0

N_PASOS_ENTRENO = int(DURACION_ENTRENO / DT)
N_PASOS_TEST_ACOPLADO = int(DURACION_TEST_ACOPLADO / DT)
N_PASOS_TEST_DESACOPLADO = int(DURACION_TEST_DESACOPLADO / DT)

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


def simular_experimento_combinado():
    print("=" * 100)
    print("VSTCosmos - v72c Fase 5: Mecanismos Complementarios")
    print(f"GAMMA_PLAST = {GAMMA_PLAST}, GAMMA_MEMORIA = {GAMMA_MEMORIA}")
    print("Fase 1: Entrenamiento (voz, alpha=0.05)")
    print("Fase 2: Test acoplado (voz, alpha=0.0)")
    print("Fase 3: Test desacoplado (ruido, alpha=0.0)")
    print("=" * 100)

    # Inicialización
    Phi_total = inicializar_campo_total()
    Phi_vel_total = np.zeros_like(Phi_total)
    W = inicializar_plasticidad()
    Phi_int_historia = inicializar_historia_interna()
    omega_natural, amort_natural = calcular_frecuencias_naturales(DIM_TOTAL, DIM_INTERNA)
    
    # Cargar audios
    sr_voz, audio_voz = cargar_audio("Voz_Estudio.wav", duracion=DURACION_ENTRENO + DURACION_TEST_ACOPLADO)
    sr_ruido, audio_ruido = cargar_audio("Ruido blanco", duracion=DURACION_TEST_DESACOPLADO)
    
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
    print(f"  Phi_int_historia: norma {np.mean(np.abs(Phi_int_historia)):.6f}")
    
    # ============================================================
    # FASE 2: TEST ACOPLADO
    # ============================================================
    print("\n[Fase 2] Test acoplado (voz, alpha=0.0)")
    gradientes_acoplado = []
    acoplamientos_acoplado = []
    
    for idx in range(N_PASOS_ENTRENO, N_PASOS_ENTRENO + N_PASOS_TEST_ACOPLADO):
        objetivo = preparar_objetivo_audio(audio_voz, sr_voz, idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
        Phi_total, Phi_vel_total, W, Phi_int_historia = actualizar_campo_total(
            Phi_total, Phi_vel_total, W, Phi_int_historia,
            objetivo, alpha=0.0, omega_natural=omega_natural, 
            amort_natural=amort_natural, dt=DT, entrenando=False
        )
        gradientes_acoplado.append(calcular_gradiente(Phi_total, DIM_INTERNA))
        acoplamientos_acoplado.append(calcular_acoplamiento(Phi_total, DIM_INTERNA))
    
    grad_acoplado_estable = np.mean(gradientes_acoplado[-int(5.0/DT):])
    perfil_acoplado = calcular_perfil_espectral_modos(Phi_total, DIM_INTERNA)
    grad_esp_acoplado = calcular_gradiente_espectral_diferencial(Phi_total, DIM_INTERNA)
    
    print(f"  Gradiente estable: {grad_acoplado_estable:.4f}")
    print(f"  Frecuencia dominante: modo {perfil_acoplado['frecuencia_dominante']}")
    print(f"  Riqueza modal: {perfil_acoplado['riqueza_modal']} modos")
    print(f"  Gradiente espectral diferencial: {grad_esp_acoplado['gradiente_espectral']:.6f}")
    
    # ============================================================
    # FASE 3: TEST DESACOPLADO
    # ============================================================
    print("\n[Fase 3] Test desacoplado (ruido, alpha=0.0)")
    gradientes_desacoplado = []
    acoplamientos_desacoplado = []
    
    for idx in range(N_PASOS_TEST_DESACOPLADO):
        objetivo = preparar_objetivo_audio(audio_ruido, sr_ruido, idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
        Phi_total, Phi_vel_total, W, Phi_int_historia = actualizar_campo_total(
            Phi_total, Phi_vel_total, W, Phi_int_historia,
            objetivo, alpha=0.0, omega_natural=omega_natural, 
            amort_natural=amort_natural, dt=DT, entrenando=False
        )
        gradientes_desacoplado.append(calcular_gradiente(Phi_total, DIM_INTERNA))
        acoplamientos_desacoplado.append(calcular_acoplamiento(Phi_total, DIM_INTERNA))
    
    grad_desacoplado_estable = np.mean(gradientes_desacoplado[-int(5.0/DT):])
    perfil_desacoplado = calcular_perfil_espectral_modos(Phi_total, DIM_INTERNA)
    grad_esp_desacoplado = calcular_gradiente_espectral_diferencial(Phi_total, DIM_INTERNA)
    
    print(f"  Gradiente estable: {grad_desacoplado_estable:.4f}")
    print(f"  Frecuencia dominante: modo {perfil_desacoplado['frecuencia_dominante']}")
    print(f"  Riqueza modal: {perfil_desacoplado['riqueza_modal']} modos")
    print(f"  Gradiente espectral diferencial: {grad_esp_desacoplado['gradiente_espectral']:.6f}")
    
    # ============================================================
    # DIAGNÓSTICO
    # ============================================================
    print("\n" + "=" * 100)
    print("DIAGNÓSTICO FINAL")
    print("=" * 100)
    
    ratio_grad = grad_acoplado_estable / max(grad_desacoplado_estable, 0.001)
    ratio_esp = grad_esp_acoplado['gradiente_espectral'] / max(grad_esp_desacoplado['gradiente_espectral'], 0.001)
    
    print(f"\n  Gradiente estable (voz):   {grad_acoplado_estable:.4f}")
    print(f"  Gradiente estable (ruido): {grad_desacoplado_estable:.4f}")
    print(f"  Ratio gradiente:           {ratio_grad:.3f}")
    print(f"\n  Frecuencia dominante (voz):   modo {perfil_acoplado['frecuencia_dominante']}")
    print(f"  Frecuencia dominante (ruido): modo {perfil_desacoplado['frecuencia_dominante']}")
    print(f"\n  Gradiente espectral diff (voz):   {grad_esp_acoplado['gradiente_espectral']:.6f}")
    print(f"  Gradiente espectral diff (ruido): {grad_esp_desacoplado['gradiente_espectral']:.6f}")
    print(f"  Ratio espectral:                  {ratio_esp:.3f}")
    
    # Criterios
    resistencia_ok = (perfil_desacoplado['frecuencia_dominante'] >= 30 and grad_desacoplado_estable > 0.3)
    acoplamiento_ok = ratio_esp > 1.5
    w_aprendio = w_tras_entreno > 0.01
    
    print("\n" + "=" * 100)
    print("CONCLUSIÓN")
    print("=" * 100)
    
    if resistencia_ok and acoplamiento_ok and w_aprendio:
        print("\n  ✅ VALIDACIÓN COMPLETA")
        print("     El atractor aprendido resiste el cambio de estímulo.")
        print("     El gradiente espectral diferencial es mayor con voz.")
        print("     W aprendió y está activo.")
        print("     → Primera demostración de modos propios aprendidos.")
    elif resistencia_ok:
        print("\n  ✅ RESISTENCIA DEL ATRACTOR")
        print("     Frecuencia dominante se mantiene alta con ruido,")
        print("     pero la diferenciación espectral entre regiones es débil.")
        print("     → Aumentar GAMMA_MEMORIA a 0.08.")
    elif acoplamiento_ok:
        print("\n  ✅ ACOPLAMIENTO RESONANTE")
        print("     El gradiente espectral diferencial es alto con voz,")
        print("     pero el atractor no resiste el cambio a ruido.")
        print("     → El atractor no se consolidó. Subir TAU_INT_HIST a 0.0005.")
    elif w_aprendio:
        print("\n  ⚠️ APRENDIZAJE SIN EFECTO")
        print("     W aprendió (w={w_tras_entreno:.4f}), pero no produce ni resistencia ni acoplamiento.")
        print("     → GAMMA_PLAST y GAMMA_MEMORIA insuficientes. Subir ambos.")
    else:
        print("\n  ❌ NINGÚN MECANISMO ES EFECTIVO")
        print("     W no aprendió ni el atractor se consolidó.")
        print("     → Aumentar ETA_HEBB a 0.08 y TAU_INT_HIST a 0.0005.")
    
    return {
        'w_tras_entreno': w_tras_entreno,
        'grad_acoplado': grad_acoplado_estable,
        'grad_desacoplado': grad_desacoplado_estable,
        'freq_dom_acoplado': perfil_acoplado['frecuencia_dominante'],
        'freq_dom_desacoplado': perfil_desacoplado['frecuencia_dominante'],
        'grad_esp_acoplado': grad_esp_acoplado['gradiente_espectral'],
        'grad_esp_desacoplado': grad_esp_desacoplado['gradiente_espectral'],
        'ratio_esp': ratio_esp,
        'riqueza_acoplado': perfil_acoplado['riqueza_modal'],
        'riqueza_desacoplado': perfil_desacoplado['riqueza_modal']
    }


def main():
    resultados = simular_experimento_combinado()
    
    # Visualización
    print("\n[Generando visualización...]")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].text(0.5, 0.5, 
                    f"W final: {resultados['w_tras_entreno']:.4f}\n"
                    f"Gradiente voz: {resultados['grad_acoplado']:.3f}\n"
                    f"Gradiente ruido: {resultados['grad_desacoplado']:.3f}\n"
                    f"Ratio: {resultados['grad_acoplado']/max(resultados['grad_desacoplado'],0.001):.3f}",
                    ha='center', va='center', fontsize=10)
    axes[0, 0].set_title("Gradiente medio")
    axes[0, 0].axis('off')
    
    axes[0, 1].text(0.5, 0.5,
                    f"Freq dominio voz: modo {resultados['freq_dom_acoplado']}\n"
                    f"Freq dominio ruido: modo {resultados['freq_dom_desacoplado']}\n"
                    f"Riqueza voz: {resultados['riqueza_acoplado']} modos\n"
                    f"Riqueza ruido: {resultados['riqueza_desacoplado']} modos",
                    ha='center', va='center', fontsize=10)
    axes[0, 1].set_title("Perfil espectral (region_int)")
    axes[0, 1].axis('off')
    
    axes[1, 0].text(0.5, 0.5,
                    f"Grad esp voz: {resultados['grad_esp_acoplado']:.6f}\n"
                    f"Grad esp ruido: {resultados['grad_esp_desacoplado']:.6f}\n"
                    f"Ratio espectral: {resultados['ratio_esp']:.3f}",
                    ha='center', va='center', fontsize=10)
    axes[1, 0].set_title("Gradiente espectral diferencial")
    axes[1, 0].axis('off')
    
    axes[1, 1].text(0.5, 0.5,
                    "Criterios:\n"
                    f"Resistencia: {'✅' if resultados['freq_dom_desacoplado'] >= 30 else '❌'}\n"
                    f"Acoplamiento: {'✅' if resultados['ratio_esp'] > 1.5 else '❌'}\n"
                    f"W_aprendio: {'✅' if resultados['w_tras_entreno'] > 0.01 else '❌'}",
                    ha='center', va='center', fontsize=10)
    axes[1, 1].set_title("Diagnóstico")
    axes[1, 1].axis('off')
    
    plt.suptitle('VSTCosmos v72c Fase 5 - Mecanismos Complementarios', fontsize=14)
    plt.tight_layout()
    plt.savefig('v72c_fase5_mecanismos_complementarios.png', dpi=150)
    print("  Gráfico guardado: v72c_fase5_mecanismos_complementarios.png")
    
    # Guardar CSV
    with open('v72c_fase5_resultado.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for k, v in resultados.items():
            writer.writerow([k, v])
    
    print("  CSV guardado: v72c_fase5_resultado.csv")
    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)


if __name__ == "__main__":
    main()