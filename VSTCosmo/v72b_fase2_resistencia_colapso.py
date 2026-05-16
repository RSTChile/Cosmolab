#!/usr/bin/env python3
"""
VSTCosmos - v72-B Fase 2: Histéresis + Resistencia al Colapso
Mecanismo: histéresis con parámetros ajustados + término de inercia
que tira hacia la historia cuando hay gradiente reciente.
NO multiplica Phi_update — evita amplificación artificial.
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
DURACION_SEGMENTO = 30.0      # para tests de persistencia
DURACION_TRANSICION = 20.0    # para asimetría
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

# Parámetros de memoria estructural (ajustados)
GAMMA_MEMORIA = 0.15      # aumentado (antes 0.08)
BETA_MEMORIA = 3.0        # aumentado (antes 2.0)
TAU_HISTORIA = 0.0001     # reducido (antes 0.0005)
K_INERCIA = 0.05          # nuevo: fuerza de resistencia al colapso

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


def calcular_memoria_fase2(Phi_total, Phi_historia, grad_inercia):
    """
    Dos componentes:
    1. Histéresis (igual que Fase 1, parámetros más fuertes)
    2. Resistencia al colapso: tira hacia la historia
       proporcionalmente a cuánto régimen hubo recientemente.
    
    NO multiplica Phi_update — evita amplificación artificial.
    """
    diferencia = Phi_total - Phi_historia

    # Componente 1: histéresis
    M_histeresis = (-GAMMA_MEMORIA
                    * np.sign(diferencia)
                    * np.abs(diferencia) ** BETA_MEMORIA)

    # Componente 2: resistencia al colapso
    # grad_inercia mide cuánto régimen hubo recientemente.
    # El término tira Phi_total hacia Phi_historia
    # solo cuando hay historia de gradiente.
    M_resistencia = K_INERCIA * grad_inercia * (Phi_historia - Phi_total)

    return M_histeresis + M_resistencia


def actualizar_historia(Phi_historia, Phi_total):
    """La historia se actualiza lentamente."""
    return (1 - TAU_HISTORIA) * Phi_historia + TAU_HISTORIA * Phi_total


def actualizar_campo_total(Phi_total, Phi_vel_total, Phi_historia,
                           objetivo_audio, alpha,
                           omega_natural, amort_natural, grad_inercia):
    # DIFUSIÓN
    promedio_local = vecinos(Phi_total)
    difusion = DIFUSION_BASE * (promedio_local - Phi_total)
    
    # REACCIÓN
    desviacion = Phi_total - promedio_local
    reaccion = GANANCIA_REACCION * desviacion * (1 - desviacion**2)
    
    # OSCILACIÓN
    term_osc = (-omega_natural**2 * (Phi_total - PHI_EQUILIBRIO)
                - amort_natural * Phi_vel_total)
    
    # MEMORIA ESTRUCTURAL (histéresis + resistencia al colapso)
    M = calcular_memoria_fase2(Phi_total, Phi_historia, grad_inercia)
    
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
    """Test A: Exponer campo a voz con alpha=0.05, luego alpha=0.0.
    Medir cuánto tiempo el gradiente se mantiene > 0.15"""
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
    grad_inercia = 0.0
    
    # FASE 1: alpha=0.05
    print("    Fase 1: voz con alpha=0.05")
    alpha = 0.05
    for idx in range(N_PASOS_SEGMENTO):
        obj = preparar_objetivo_audio(audio, sr, idx, ventana_muestras, hop_muestras,
                                      DIM_AUDITIVA, DIM_TIME)
        Phi_total, Phi_vel_total = actualizar_campo_total(
            Phi_total, Phi_vel_total, Phi_historia, obj, alpha,
            omega_natural, amort_natural, grad_inercia
        )
        Phi_historia = actualizar_historia(Phi_historia, Phi_total)
        grad = calcular_gradiente(Phi_total, DIM_INTERNA)
        grad_inercia = 0.95 * grad_inercia + 0.05 * grad
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
            omega_natural, amort_natural, grad_inercia
        )
        Phi_historia = actualizar_historia(Phi_historia, Phi_total)
        grad = calcular_gradiente(Phi_total, DIM_INTERNA)
        grad_inercia = 0.95 * grad_inercia + 0.05 * grad
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
        grad_inercia = 0.0
        
        # Segmento A: ruido
        for idx in range(N_PASOS_TRANSICION):
            obj = preparar_objetivo_audio(audio_a, sr_a, idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
            Phi_total, Phi_vel_total = actualizar_campo_total(
                Phi_total, Phi_vel_total, Phi_historia, obj, alpha,
                omega_natural, amort_natural, grad_inercia
            )
            Phi_historia = actualizar_historia(Phi_historia, Phi_total)
            grad = calcular_gradiente(Phi_total, DIM_INTERNA)
            grad_inercia = 0.95 * grad_inercia + 0.05 * grad
            gradientes.append(grad)
            tiempos.append(idx * DT)
        
        gradiente_basal = np.mean(gradientes[-50:]) if len(gradientes) >= 50 else gradientes[0]
        
        # Segmento B: voz
        for idx in range(N_PASOS_TRANSICION):
            obj = preparar_objetivo_audio(audio_b, sr_b, idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
            Phi_total, Phi_vel_total = actualizar_campo_total(
                Phi_total, Phi_vel_total, Phi_historia, obj, alpha,
                omega_natural, amort_natural, grad_inercia
            )
            Phi_historia = actualizar_historia(Phi_historia, Phi_total)
            grad = calcular_gradiente(Phi_total, DIM_INTERNA)
            grad_inercia = 0.95 * grad_inercia + 0.05 * grad
            gradientes.append(grad)
            tiempos.append((N_PASOS_TRANSICION + idx) * DT)
        
        # Métricas detalladas
        gradientes_B = gradientes[N_PASOS_TRANSICION:]
        gradiente_max = max(gradientes_B)
        gradiente_pico_idx = gradientes_B.index(gradiente_max)
        gradiente_pico_t = gradiente_pico_idx * DT
        
        ventana_estable = min(50, len(gradientes_B))
        gradiente_estable = np.mean(gradientes_B[-ventana_estable:])
        gradiente_basal = np.mean(gradientes[:50]) if len(gradientes) >= 50 else gradientes[0]
        
        ratio_pico_basal = gradiente_max / max(gradiente_basal, 1e-6)
        ratio_estable_basal = gradiente_estable / max(gradiente_basal, 1e-6)
        
        print(f"      Gradiente máximo:         {gradiente_max:.4f}  (t={gradiente_pico_t:.2f}s)")
        print(f"      Gradiente estable (5s):   {gradiente_estable:.4f}")
        print(f"      Gradiente basal:          {gradiente_basal:.4f}")
        print(f"      Ratio pico/basal:         {ratio_pico_basal:.3f}")
        print(f"      Ratio estable/basal:      {ratio_estable_basal:.3f}")
        
        resultados[clave] = {
            'gradiente_max': gradiente_max,
            'gradiente_pico_t': gradiente_pico_t,
            'gradiente_estable': gradiente_estable,
            'gradiente_basal': gradiente_basal,
            'ratio_pico_basal': ratio_pico_basal,
            'ratio_estable_basal': ratio_estable_basal,
            'gradientes': gradientes,
            'tiempos': tiempos
        }
    
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
    grad_inercia = 0.0
    
    for idx in range(n_pasos):
        obj = preparar_objetivo_audio(audio, sr, idx, ventana_muestras, hop_muestras,
                                      DIM_AUDITIVA, DIM_TIME)
        Phi_total, Phi_vel_total = actualizar_campo_total(
            Phi_total, Phi_vel_total, Phi_historia, obj, alpha,
            omega_natural, amort_natural, grad_inercia
        )
        Phi_historia = actualizar_historia(Phi_historia, Phi_total)
        acop = calcular_acoplamiento(Phi_total, DIM_INTERNA)
        grad = calcular_gradiente(Phi_total, DIM_INTERNA)
        grad_inercia = 0.95 * grad_inercia + 0.05 * grad
        acoplamientos.append(acop)
    
    min_acop = min(acoplamientos)
    print(f"    min(A_sys-env) = {min_acop:.6f}")
    return min_acop, acoplamientos


def main():
    print("=" * 100)
    print("VSTCosmos - v72-B Fase 2: Histéresis + Resistencia al Colapso")
    print(f"GAMMA_MEMORIA = {GAMMA_MEMORIA}, BETA_MEMORIA = {BETA_MEMORIA}")
    print(f"TAU_HISTORIA = {TAU_HISTORIA}, K_INERCIA = {K_INERCIA}")
    print("Test A: Persistencia | Test B: Asimetría | Test C: Acoplamiento")
    print("=" * 100)
    
    # Test A
    grad_persistencia, tiempos_persistencia, tiempo_persistencia = test_persistencia()
    
    # Test B
    resultados_asimetria = test_asimetria()
    
    # Test C
    min_acop, acoplamientos = test_acoplamiento()
    
    # Extraer métricas clave de asimetría
    ruido_voz = resultados_asimetria.get("Ruido blanco -> Voz_Estudio.wav", {})
    voz_ruido = resultados_asimetria.get("Voz_Estudio.wav -> Ruido blanco", {})
    
    grad_estable_ruido_voz = ruido_voz.get('gradiente_estable', 0)
    grad_estable_voz_ruido = voz_ruido.get('gradiente_estable', 0)
    ratio_ruido_voz = ruido_voz.get('ratio_estable_basal', 0)
    
    # ============================================================
    # CRITERIOS DE DECISIÓN (dos niveles)
    # ============================================================
    print("\n" + "=" * 100)
    print("CRITERIOS DE DECISIÓN")
    print("=" * 100)
    
    criterio_minimo = grad_estable_ruido_voz > 0.05
    criterio_validacion = (tiempo_persistencia > 20.0 and 
                           grad_estable_ruido_voz > 0.15 and
                           ratio_ruido_voz > 2.0)
    criterio_abandono = grad_estable_ruido_voz < 0.03
    
    print(f"\n  Test A (Persistencia > 20s): {tiempo_persistencia:.2f}s")
    print(f"  Test B (Gradiente estable ruido->voz): {grad_estable_ruido_voz:.4f}")
    print(f"  Test B (Ratio estable/basal): {ratio_ruido_voz:.3f}")
    print(f"  Test C (Acoplamiento > 0.01): min={min_acop:.6f} -> {'✅' if min_acop > 0.01 else '❌'}")
    
    print("\n" + "=" * 100)
    print("DECISIÓN")
    print("=" * 100)
    
    if criterio_abandono:
        print("\n  ❌ CRITERIO DE ABANDONO DE HISTÉRESIS")
        print("     gradiente_estable < 0.03")
        print("     El mecanismo es arquitectónicamente insuficiente.")
        print("     → Pasar directamente a v72-B Fase 3: Plasticidad Hebbiana")
    elif criterio_validacion:
        print("\n  ✅ VALIDACIÓN COMPLETA: v72-B Fase 2 exitosa")
        print("     Persistencia > 20s Y gradiente_estable > 0.15")
        print("     → Proceder a Fase 3: Plasticidad Hebbiana")
    elif criterio_minimo:
        print("\n  ⚠️ CRITERIO MÍNIMO CUMPLIDO")
        print("     gradiente_estable > 0.05")
        print("     El mecanismo tiene dirección correcta pero escala insuficiente.")
        print("     → Iterar GAMMA_MEMORIA hacia 0.25")
    else:
        print("\n  ❌ CRITERIO MÍNIMO NO CUMPLIDO")
        print("     gradiente_estable < 0.05")
        print("     El mecanismo no produce diferenciación sostenida.")
        print("     → Revisar parámetros o pasar a Plasticidad Hebbiana")
    
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
    ax.axvline(x=DURACION_TRANSICION, color='black', linestyle='-', label='Cambio de estímulo')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Gradiente')
    ax.set_title('Test B: Asimetría (alpha=0.0)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('v72b_fase2_resultados.png', dpi=150)
    print("  Gráfico guardado: v72b_fase2_resultados.png")
    
    # Guardar CSV
    with open('v72b_fase2_resultado.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['test', 'metric', 'value', 'criterio'])
        writer.writerow(['A', 'persistencia_s', f'{tiempo_persistencia:.2f}', '>20'])
        writer.writerow(['B', 'grad_estable_ruido_voz', f'{grad_estable_ruido_voz:.4f}', '>0.15'])
        writer.writerow(['B', 'ratio_estable_basal', f'{ratio_ruido_voz:.3f}', '>2.0'])
        writer.writerow(['C', 'min_acoplamiento', f'{min_acop:.6f}', '>0.01'])
        
        if criterio_validacion:
            writer.writerow(['decision', 'v72_b_fase2', 'VALIDADO', ''])
        elif criterio_minimo:
            writer.writerow(['decision', 'v72_b_fase2', 'CRITERIO_MINIMO', 'aumentar_gamma'])
        elif criterio_abandono:
            writer.writerow(['decision', 'v72_b_fase2', 'ABANDONAR_HISTERESIS', 'pasar_a_hebbiana'])
        else:
            writer.writerow(['decision', 'v72_b_fase2', 'NO_VALIDADO', 'revisar_parametros'])
    
    print("  CSV guardado: v72b_fase2_resultado.csv")
    
    # Guardar TXT
    with open('v72b_fase2_resultado.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("VSTCosmos v72-B Fase 2 - Resultado\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"GAMMA_MEMORIA = {GAMMA_MEMORIA}\n")
        f.write(f"BETA_MEMORIA = {BETA_MEMORIA}\n")
        f.write(f"TAU_HISTORIA = {TAU_HISTORIA}\n")
        f.write(f"K_INERCIA = {K_INERCIA}\n\n")
        f.write(f"Test A (Persistencia > 20s): {tiempo_persistencia:.2f}s\n")
        f.write(f"Test B (Gradiente estable ruido->voz): {grad_estable_ruido_voz:.4f}\n")
        f.write(f"Test B (Ratio estable/basal): {ratio_ruido_voz:.3f}\n")
        f.write(f"Test C (Acoplamiento > 0.01): min={min_acop:.6f}\n\n")
        f.write("=" * 60 + "\n")
        
        if criterio_abandono:
            f.write("DECISION: ABANDONAR HISTERESIS\n")
            f.write("Pasar directamente a v72-B Fase 3: Plasticidad Hebbiana\n")
        elif criterio_validacion:
            f.write("DECISION: VALIDADO - v72-B Fase 2 exitosa\n")
            f.write("Proceder a Fase 3: Plasticidad Hebbiana\n")
        elif criterio_minimo:
            f.write("DECISION: CRITERIO MINIMO - Iterar GAMMA_MEMORIA hacia 0.25\n")
        else:
            f.write("DECISION: NO VALIDADO - Revisar parametros\n")
        f.write("=" * 60 + "\n")
    
    print("  TXT guardado: v72b_fase2_resultado.txt")
    
    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)


if __name__ == "__main__":
    main()