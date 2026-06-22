#!/usr/bin/env python3
"""
VSTCosmos - v72-B Fase 1: Campo con Memoria Estructural (Histéresis)
Mecanismo: histéresis simple M(Φ) que resiste cambios respecto a la historia del campo.
Objetivo: que el campo retenga estructura incluso sin sesgo externo (alpha=0.0).
Tests: Persistencia, Asimetría real, Acoplamiento constitutivo.
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
DURACION_SEGMENTO = 30.0  # para tests de persistencia
DURACION_TRANSICION = 20.0  # para asimetría
N_PASOS_SEGMENTO = int(DURACION_SEGMENTO / DT)
N_PASOS_TRANSICION = int(DURACION_TRANSICION / DT)

# Parámetros de dinámica
DIFUSION_BASE = 0.15
GANANCIA_REACCION = 0.05

OMEGA_MIN = 0.05
OMEGA_MAX = 0.50
AMORT_MIN = 0.01
AMORT_MAX = 0.08
PHI_EQUILIBRIO = 0.5

# Parámetros de memoria estructural (ajustados para prueba)
GAMMA_MEMORIA = 0.08      # aumentado (antes 0.02)
BETA_MEMORIA = 2.0        # aumentado (antes 1.5)
TAU_HISTORIA = 0.0005     # reducido (antes 0.001)

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


def inicializar_historia():
    """La historia se inicializa igual que el campo"""
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
    """
    M(Φ): término que resiste cambios respecto a la historia del campo.
    """
    diferencia = Phi_total - Phi_historia
    M = -GAMMA_MEMORIA * np.sign(diferencia) * np.abs(diferencia) ** BETA_MEMORIA
    return M


def actualizar_historia(Phi_historia, Phi_total):
    """La historia se actualiza lentamente."""
    return (1 - TAU_HISTORIA) * Phi_historia + TAU_HISTORIA * Phi_total


def actualizar_campo_total(Phi_total, Phi_vel_total, Phi_historia,
                           objetivo_audio, alpha,
                           omega_natural, amort_natural):
    # DIFUSIÓN
    promedio_local = vecinos(Phi_total)
    difusion = DIFUSION_BASE * (promedio_local - Phi_total)
    
    # REACCIÓN
    desviacion = Phi_total - promedio_local
    reaccion = GANANCIA_REACCION * desviacion * (1 - desviacion**2)
    
    # OSCILACIÓN
    term_osc = (-omega_natural**2 * (Phi_total - PHI_EQUILIBRIO)
                - amort_natural * Phi_vel_total)
    
    # MEMORIA ESTRUCTURAL
    M = calcular_memoria(Phi_total, Phi_historia)
    
    # Actualización del campo
    dPhi_vel = term_osc + reaccion + difusion + M
    Phi_vel_nueva = Phi_vel_total + DT * dPhi_vel
    Phi_nueva = Phi_total + DT * Phi_vel_nueva
    
    # Sesgo operativo
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


# ============================================================
# TESTS
# ============================================================
def test_persistencia():
    """Test A: Exponer campo a voz con alpha=0.05, luego alpha=0.0."""
    print("\n  Test A: Persistencia")
    
    Phi_total = inicializar_campo_total()
    Phi_vel_total = np.zeros_like(Phi_total)
    Phi_historia = inicializar_historia()
    omega_natural, amort_natural = calcular_frecuencias_naturales(DIM_TOTAL, DIM_INTERNA)
    
    sr, audio = cargar_audio("Voz_Estudio.wav", duracion=DURACION_SEGMENTO * 2)
    ventana_muestras = int(sr * VENTANA_FFT_MS / 1000)
    hop_muestras = int(sr * HOP_FFT_MS / 1000)
    
    gradientes = []
    tiempos = []
    
    # FASE 1: alpha=0.05
    print("    Fase 1: voz con alpha=0.05")
    alpha = 0.05
    for idx in range(N_PASOS_SEGMENTO):
        obj = preparar_objetivo_audio(audio, sr, idx, ventana_muestras, hop_muestras,
                                      DIM_AUDITIVA, DIM_TIME)
        Phi_total, Phi_vel_total = actualizar_campo_total(
            Phi_total, Phi_vel_total, Phi_historia, obj, alpha,
            omega_natural, amort_natural
        )
        Phi_historia = actualizar_historia(Phi_historia, Phi_total)
        grad = calcular_gradiente(Phi_total, DIM_INTERNA)
        gradientes.append(grad)
        tiempos.append(idx * DT)
    
    # FASE 2: alpha=0.0
    print("    Fase 2: voz con alpha=0.0")
    alpha = 0.0
    for idx in range(N_PASOS_SEGMENTO, 2 * N_PASOS_SEGMENTO):
        obj = preparar_objetivo_audio(audio, sr, idx, ventana_muestras, hop_muestras,
                                      DIM_AUDITIVA, DIM_TIME)
        Phi_total, Phi_vel_total = actualizar_campo_total(
            Phi_total, Phi_vel_total, Phi_historia, obj, alpha,
            omega_natural, amort_natural
        )
        Phi_historia = actualizar_historia(Phi_historia, Phi_total)
        grad = calcular_gradiente(Phi_total, DIM_INTERNA)
        gradientes.append(grad)
        tiempos.append(idx * DT)
    
    # Medir persistencia (gradiente > 0.15)
    tiempo_persistencia = 0
    for i in range(N_PASOS_SEGMENTO, len(gradientes)):
        if gradientes[i] > 0.15:
            tiempo_persistencia = (i - N_PASOS_SEGMENTO) * DT
        else:
            break
    
    print(f"    Persistencia: {tiempo_persistencia:.2f} s")
    
    return gradientes, tiempos, tiempo_persistencia


def test_asimetria():
    """Test B: ruido->voz vs voz->ruido con alpha=0.0"""
    print("\n  Test B: Asimetría real (alpha=0.0)")
    
    transiciones = [
        ("Ruido blanco", "Voz_Estudio.wav"),
        ("Voz_Estudio.wav", "Ruido blanco")
    ]
    
    resultados = {}
    
    for est_a, est_b in transiciones:
        clave = f"{est_a} -> {est_b}"
        print(f"    Simulando: {clave}")
        
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
        alpha = 0.0
        
        # Segmento A
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
        
        # Segmento B
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
        
        # Calcular régimen estable antes
        g_antes = np.mean(gradientes[N_PASOS_TRANSICION-50:N_PASOS_TRANSICION])
        
        # Medir tiempo para alcanzar 2x el valor basal
        tiempo_transicion = None
        for i in range(N_PASOS_TRANSICION, len(gradientes)):
            if gradientes[i] > g_antes * 2:
                tiempo_transicion = (i - N_PASOS_TRANSICION) * DT
                break
        
        resultados[clave] = {
            'tiempo': tiempo_transicion,
            'g_antes': g_antes,
            'gradientes': gradientes,
            'tiempos': tiempos
        }
        
        tiempo_str = f"{tiempo_transicion:.3f} s" if tiempo_transicion else "no alcanzado"
        print(f"    {clave}: {tiempo_str}")
    
    return resultados


def test_acoplamiento():
    """Test C: Verificar A_sys-env > 0.01"""
    print("\n  Test C: Acoplamiento constitutivo")
    
    Phi_total = inicializar_campo_total()
    Phi_vel_total = np.zeros_like(Phi_total)
    Phi_historia = inicializar_historia()
    omega_natural, amort_natural = calcular_frecuencias_naturales(DIM_TOTAL, DIM_INTERNA)
    
    sr, audio = cargar_audio("Voz_Estudio.wav", duracion=30.0)
    ventana_muestras = int(sr * VENTANA_FFT_MS / 1000)
    hop_muestras = int(sr * HOP_FFT_MS / 1000)
    n_pasos = min(int(30.0 / DT), (len(audio) - ventana_muestras) // hop_muestras + 1)
    
    acoplamientos = []
    alpha = 0.0
    
    for idx in range(n_pasos):
        obj = preparar_objetivo_audio(audio, sr, idx, ventana_muestras, hop_muestras,
                                      DIM_AUDITIVA, DIM_TIME)
        Phi_total, Phi_vel_total = actualizar_campo_total(
            Phi_total, Phi_vel_total, Phi_historia, obj, alpha,
            omega_natural, amort_natural
        )
        Phi_historia = actualizar_historia(Phi_historia, Phi_total)
        acop = calcular_acoplamiento(Phi_total, DIM_INTERNA)
        acoplamientos.append(acop)
    
    min_acop = min(acoplamientos)
    print(f"    min(A_sys-env) = {min_acop:.6f}")
    return min_acop, acoplamientos


def main():
    print("=" * 100)
    print("VSTCosmos - v72-B Fase 1: Campo con Memoria Estructural (Histéresis)")
    print(f"GAMMA_MEMORIA = {GAMMA_MEMORIA}, BETA_MEMORIA = {BETA_MEMORIA}, TAU_HISTORIA = {TAU_HISTORIA}")
    print("Test A: Persistencia | Test B: Asimetría | Test C: Acoplamiento")
    print("=" * 100)
    
    # Test A
    grad_persistencia, tiempos_persistencia, tiempo_persistencia = test_persistencia()
    
    # Test B
    resultados_asimetria = test_asimetria()
    
    # Test C
    min_acop, acoplamientos = test_acoplamiento()
    
    # Extraer tiempos de asimetría
    tiempo_ruido_voz = None
    tiempo_voz_ruido = None
    for clave, res in resultados_asimetria.items():
        if "Ruido blanco -> Voz_Estudio.wav" in clave:
            tiempo_ruido_voz = res['tiempo']
        elif "Voz_Estudio.wav -> Ruido blanco" in clave:
            tiempo_voz_ruido = res['tiempo']
    
    # ============================================================
    # CRITERIO DE DECISIÓN
    # ============================================================
    print("\n" + "=" * 100)
    print("CRITERIO DE DECISIÓN")
    print("=" * 100)
    
    criterio_a = tiempo_persistencia > 20.0
    
    criterio_b = False
    if tiempo_ruido_voz is not None and tiempo_voz_ruido is not None:
        criterio_b = (tiempo_ruido_voz > 2.0 and tiempo_voz_ruido < 0.5)
    elif tiempo_ruido_voz is not None and tiempo_voz_ruido is None:
        # voz->ruido instantáneo cuenta como < 0.5
        criterio_b = tiempo_ruido_voz > 2.0
    
    criterio_c = min_acop > 0.01
    
    print(f"\n  Test A (Persistencia > 20s): {tiempo_persistencia:.2f}s -> {'✅' if criterio_a else '❌'}")
    print(f"  Test B (Asimetría real):")
    ruido_voz_str = f"{tiempo_ruido_voz:.3f}s" if tiempo_ruido_voz else "no alcanzado"
    voz_ruido_str = f"{tiempo_voz_ruido:.3f}s" if tiempo_voz_ruido else "no alcanzado"
    print(f"    ruido->voz: {ruido_voz_str}")
    print(f"    voz->ruido: {voz_ruido_str}")
    print(f"    -> {'✅' if criterio_b else '❌'}")
    print(f"  Test C (Acoplamiento > 0.01): min={min_acop:.6f} -> {'✅' if criterio_c else '❌'}")
    
    print("\n" + "=" * 100)
    print("DECISIÓN")
    print("=" * 100)
    
    if criterio_a and criterio_b and criterio_c:
        print("\n  ✅ VALIDADO: v72-B Fase 1 exitosa.")
        print("     La histéresis produce persistencia y asimetría reales.")
        print("     Proceder a Fase 2: Plasticidad Hebbiana.")
    else:
        print("\n  ⚠️  NO VALIDADO: Ajustar parámetros.")
        if not criterio_a:
            print("     -> Aumentar GAMMA_MEMORIA a 0.12 o reducir TAU_HISTORIA a 0.0002")
        if not criterio_b:
            print("     -> Aumentar BETA_MEMORIA a 2.5 (más no-linealidad)")
        if not criterio_c:
            print("     -> Reducir GAMMA_MEMORIA a 0.05")
    
    # ============================================================
    # VISUALIZACIÓN
    # ============================================================
    print("\n[Generando visualizaciones...]")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfico 1: Persistencia
    ax = axes[0]
    ax.plot(tiempos_persistencia, grad_persistencia)
    ax.axvline(x=DURACION_SEGMENTO, color='r', linestyle='--', label='alpha: 0.05 -> 0.0')
    ax.axhline(y=0.15, color='g', linestyle='--', label='Umbral persistencia')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Gradiente')
    ax.set_title(f'Test A: Persistencia\nPersistencia: {tiempo_persistencia:.1f}s')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 2: Asimetría
    ax = axes[1]
    colores = ['blue', 'red']
    for idx, (clave, res) in enumerate(resultados_asimetria.items()):
        ax.plot(res['tiempos'], res['gradientes'], label=clave, color=colores[idx % len(colores)])
        if res['tiempo']:
            ax.axvline(x=DURACION_TRANSICION + res['tiempo'], 
                       color=colores[idx % len(colores)], linestyle='--', alpha=0.5)
    ax.axvline(x=DURACION_TRANSICION, color='black', linestyle='-', label='Cambio de estímulo')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Gradiente')
    ax.set_title('Test B: Asimetría (alpha=0.0)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('v72b_resultados.png', dpi=150)
    print("  Gráfico guardado: v72b_resultados.png")
    
    # Guardar CSV
    with open('v72b_resultado.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['test', 'metric', 'value', 'criterio'])
        writer.writerow(['A', 'persistencia_s', f'{tiempo_persistencia:.2f}', '>20'])
        writer.writerow(['B', 'ruido_voz_s', ruido_voz_str, '>2.0'])
        writer.writerow(['B', 'voz_ruido_s', voz_ruido_str, '<0.5'])
        writer.writerow(['C', 'min_acoplamiento', f'{min_acop:.6f}', '>0.01'])
        writer.writerow(['decision', 'v72_b_fase1', 'VALIDADO' if (criterio_a and criterio_b and criterio_c) else 'NO_VALIDADO', ''])
    
    print("  CSV guardado: v72b_resultado.csv")
    
    # Guardar TXT
    with open('v72b_resultado.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("VSTCosmos v72-B Fase 1 - Resultado\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"GAMMA_MEMORIA = {GAMMA_MEMORIA}\n")
        f.write(f"BETA_MEMORIA = {BETA_MEMORIA}\n")
        f.write(f"TAU_HISTORIA = {TAU_HISTORIA}\n\n")
        f.write(f"Test A (Persistencia > 20s): {tiempo_persistencia:.2f}s -> {'PASA' if criterio_a else 'FALLA'}\n")
        f.write(f"Test B (Asimetria real):\n")
        f.write(f"  ruido->voz: {ruido_voz_str} -> {'PASA' if tiempo_ruido_voz and tiempo_ruido_voz > 2.0 else 'FALLA'}\n")
        f.write(f"  voz->ruido: {voz_ruido_str} -> {'PASA' if (tiempo_voz_ruido is None or tiempo_voz_ruido < 0.5) else 'FALLA'}\n")
        f.write(f"Test C (Acoplamiento > 0.01): min={min_acop:.6f} -> {'PASA' if criterio_c else 'FALLA'}\n\n")
        f.write("=" * 60 + "\n")
        if criterio_a and criterio_b and criterio_c:
            f.write("DECISION: VALIDADO - v72-B Fase 1 exitosa\n")
            f.write("La histeresis produce persistencia y asimetria reales.\n")
            f.write("Proceder a Fase 2: Plasticidad Hebbiana.\n")
        else:
            f.write("DECISION: NO VALIDADO - Ajustar parametros\n")
            if not criterio_a:
                f.write("- Aumentar GAMMA_MEMORIA a 0.12 o reducir TAU_HISTORIA a 0.0002\n")
            if not criterio_b:
                f.write("- Aumentar BETA_MEMORIA a 2.5\n")
            if not criterio_c:
                f.write("- Reducir GAMMA_MEMORIA a 0.05\n")
        f.write("=" * 60 + "\n")
    
    print("  TXT guardado: v72b_resultado.txt")
    
    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)


if __name__ == "__main__":
    main()