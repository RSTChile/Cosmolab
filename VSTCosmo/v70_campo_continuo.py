#!/usr/bin/env python3
"""
VSTCosmos - v71: Campo Continuo con Métricas Completas (C-N2.0)
Validación ontológica expandida.
Métricas:
1. Gradiente de diferenciación (media, std, varianza de fluctuaciones)
2. Estabilidad del régimen (autocorrelación)
3. Velocidad de transición (umbral 90%)
4. Varianza de fluctuaciones del gradiente (firma de estructura)

Transiciones de control: ruido→tono, voz→tono, ruido→voz, voz→ruido
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
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
DURACION_SIM = 60.0
N_PASOS = int(DURACION_SIM / DT)

# Parámetros de dinámica
DIFUSION_BASE = 0.15
GANANCIA_REACCION = 0.05

# Parámetros oscilatorios
OMEGA_MIN = 0.05
OMEGA_MAX = 0.50
AMORT_MIN = 0.01
AMORT_MAX = 0.08
PHI_EQUILIBRIO = 0.5

# Asimetría operativa
ALPHA_AUDIO = 0.05

# Parámetros de FFT
VENTANA_FFT_MS = 25
HOP_FFT_MS = 10
F_MIN = 80
F_MAX = 8000

LIMITE_MIN = 0.0
LIMITE_MAX = 1.0

# Umbral para transiciones (90% del nuevo régimen)
UMBRAL_TRANSICION = 0.90


# ============================================================
# FUNCIONES BASE
# ============================================================
def cargar_audio(ruta, duracion=DURACION_SIM):
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


def actualizar_campo_total(Phi_total, Phi_vel_total, objetivo_audio, alpha,
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
    
    # Actualización
    dPhi_vel = term_osc + reaccion + difusion
    Phi_vel_nueva = Phi_vel_total + DT * dPhi_vel
    
    dPhi = Phi_vel_nueva
    Phi_nueva = Phi_total + DT * dPhi
    
    # Sesgo operativo
    region_auditiva_nueva = Phi_nueva[DIM_INTERNA:, :]
    region_auditiva_nueva = (1 - alpha) * region_auditiva_nueva + alpha * objetivo_audio
    Phi_nueva[DIM_INTERNA:, :] = region_auditiva_nueva
    
    return np.clip(Phi_nueva, LIMITE_MIN, LIMITE_MAX), np.clip(Phi_vel_nueva, -5.0, 5.0)


def calcular_metricas_campo(Phi_total, historial_gradientes, dim_interna):
    region_int = Phi_total[:dim_interna, :]
    region_aud = Phi_total[dim_interna:, :]
    
    # Métrica 1: Gradiente
    gradiente = np.mean(np.abs(region_int - region_aud))
    historial_gradientes.append(gradiente)
    
    # Métrica 2: Estabilidad (autocorrelación)
    if len(historial_gradientes) >= 20:
        G = np.array(historial_gradientes[-50:])
        if np.std(G) > 1e-6:
            estabilidad = np.corrcoef(G[:-1], G[1:])[0, 1]
        else:
            estabilidad = 1.0
    else:
        estabilidad = 0.0
    
    return gradiente, estabilidad


def simular_estimulo(estimulo_nombre, duracion_seg=DURACION_SIM):
    print(f"  Simulando: {estimulo_nombre}")
    
    Phi_total = inicializar_campo_total()
    Phi_vel_total = np.zeros_like(Phi_total)
    omega_natural, amort_natural = calcular_frecuencias_naturales(DIM_TOTAL, DIM_INTERNA)
    
    sr, audio = cargar_audio(estimulo_nombre, duracion=duracion_seg)
    ventana_muestras = int(sr * VENTANA_FFT_MS / 1000)
    hop_muestras = int(sr * HOP_FFT_MS / 1000)
    n_ventanas = (len(audio) - ventana_muestras) // hop_muestras + 1
    n_pasos = min(N_PASOS, n_ventanas)
    
    historial_gradientes = []
    gradientes_por_paso = []
    
    for idx in range(n_pasos):
        objetivo = preparar_objetivo_audio(audio, sr, idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
        
        Phi_total, Phi_vel_total = actualizar_campo_total(
            Phi_total, Phi_vel_total, objetivo, ALPHA_AUDIO,
            omega_natural, amort_natural
        )
        
        grad, _ = calcular_metricas_campo(Phi_total, historial_gradientes, DIM_INTERNA)
        gradientes_por_paso.append(grad)
    
    grad_array = np.array(gradientes_por_paso)
    
    # Métricas
    gradiente_media = np.mean(grad_array)
    gradiente_std = np.std(grad_array)
    
    # Métrica 4: Varianza de fluctuaciones (diferencias entre pasos sucesivos)
    fluctuaciones = np.diff(grad_array)
    varianza_fluctuaciones = np.var(fluctuaciones)
    
    # Estabilidad (última ventana)
    if len(grad_array) >= 50:
        G = grad_array[-50:]
        estabilidad = np.corrcoef(G[:-1], G[1:])[0, 1] if np.std(G) > 1e-6 else 1.0
    else:
        estabilidad = 0.0
    
    return {
        'nombre': estimulo_nombre,
        'gradiente_media': gradiente_media,
        'gradiente_std': gradiente_std,
        'estabilidad': estabilidad,
        'varianza_fluctuaciones': varianza_fluctuaciones,
        'historial_gradientes': grad_array
    }


def simular_transicion(estimulo_a, estimulo_b, duracion_segmento=20.0):
    """
    Simula transición con umbral al 90% del nuevo régimen.
    """
    print(f"  Transición: {estimulo_a} → {estimulo_b}")
    n_pasos_segmento = int(duracion_segmento / DT)
    
    Phi_total = inicializar_campo_total()
    Phi_vel_total = np.zeros_like(Phi_total)
    omega_natural, amort_natural = calcular_frecuencias_naturales(DIM_TOTAL, DIM_INTERNA)
    
    sr_a, audio_a = cargar_audio(estimulo_a, duracion=duracion_segmento)
    sr_b, audio_b = cargar_audio(estimulo_b, duracion=duracion_segmento)
    
    ventana_muestras = int(sr_a * VENTANA_FFT_MS / 1000)
    hop_muestras = int(sr_a * HOP_FFT_MS / 1000)
    
    gradientes = []
    
    # Segmento A (estabilización)
    for idx in range(n_pasos_segmento):
        obj = preparar_objetivo_audio(audio_a, sr_a, idx, ventana_muestras, hop_muestras,
                                      DIM_AUDITIVA, DIM_TIME)
        Phi_total, Phi_vel_total = actualizar_campo_total(
            Phi_total, Phi_vel_total, obj, ALPHA_AUDIO,
            omega_natural, amort_natural
        )
        grad = np.mean(np.abs(Phi_total[:DIM_INTERNA, :] - Phi_total[DIM_INTERNA:, :]))
        gradientes.append(grad)
    
    # Segmento B (transición)
    for idx in range(n_pasos_segmento):
        obj = preparar_objetivo_audio(audio_b, sr_b, idx, ventana_muestras, hop_muestras,
                                      DIM_AUDITIVA, DIM_TIME)
        Phi_total, Phi_vel_total = actualizar_campo_total(
            Phi_total, Phi_vel_total, obj, ALPHA_AUDIO,
            omega_natural, amort_natural
        )
        grad = np.mean(np.abs(Phi_total[:DIM_INTERNA, :] - Phi_total[DIM_INTERNA:, :]))
        gradientes.append(grad)
    
    # Calcular régimen estable después de la transición (últimos 100 pasos)
    g_despues = np.mean(gradientes[-100:]) if len(gradientes) >= 100 else gradientes[-1]
    
    # Buscar cuándo alcanza el UMBRAL_TRANSICION * g_despues
    tiempo_umbral = 0
    for i in range(n_pasos_segmento, len(gradientes)):
        if gradientes[i] >= UMBRAL_TRANSICION * g_despues:
            tiempo_umbral = i - n_pasos_segmento
            break
    
    velocidad = tiempo_umbral * DT  # segundos
    
    return velocidad, gradientes


def main():
    print("=" * 100)
    print("VSTCosmos - v71: Campo Continuo con Métricas Completas (C-N2.0)")
    print("Métricas: gradiente, estabilidad, varianza de fluctuaciones, transición al 90%")
    print("=" * 100)
    
    # ============================================================
    # ESTÍMULOS INDIVIDUALES
    # ============================================================
    estimulos = [
        "Ruido blanco",
        "Tono puro",
        "Voz_Estudio.wav",
        "Brandemburgo.wav"
    ]
    
    print("\n[1] Simulando estímulos individuales...")
    resultados = {}
    for est in estimulos:
        res = simular_estimulo(est, duracion_seg=60.0)
        resultados[est] = res
        print(f"    {est}:")
        print(f"      Gradiente: {res['gradiente_media']:.4f} ± {res['gradiente_std']:.4f}")
        print(f"      Estabilidad: {res['estabilidad']:.4f}")
        print(f"      Varianza fluctuaciones: {res['varianza_fluctuaciones']:.6f}")
    
    print("\n" + "=" * 100)
    print("[2] ANÁLISIS DE REGÍMENES")
    print("=" * 100)
    
    print(f"\n{'Estímulo':<20} | {'Gradiente':>12} | {'Estabilidad':>12} | {'Varianza fluctuaciones':>24}")
    print("-" * 80)
    for est in estimulos:
        r = resultados[est]
        print(f"{est:<20} | {r['gradiente_media']:12.4f} | {r['estabilidad']:12.4f} | {r['varianza_fluctuaciones']:24.6f}")
    
    # ============================================================
    # TRANSICIONES
    # ============================================================
    print("\n" + "=" * 100)
    print("[3] TRANSICIONES")
    print("=" * 100)
    
    transiciones = [
        ("Voz_Estudio.wav", "Ruido blanco"),
        ("Ruido blanco", "Voz_Estudio.wav"),
        ("Ruido blanco", "Tono puro"),
        ("Tono puro", "Ruido blanco"),
        ("Voz_Estudio.wav", "Tono puro"),
        ("Tono puro", "Voz_Estudio.wav")
    ]
    
    velocidades = {}
    for a, b in transiciones:
        vel, _ = simular_transicion(a, b, duracion_segmento=20.0)
        velocidades[f"{a[:10]}→{b[:10]}"] = vel
        print(f"    {a[:15]} → {b[:15]}: {vel:.3f} s")
    
    print("\n" + "=" * 100)
    print("[4] DIAGNÓSTICO FINAL")
    print("=" * 100)
    
    # Extraer métricas
    ruido = resultados["Ruido blanco"]
    tono = resultados["Tono puro"]
    voz = resultados["Voz_Estudio.wav"]
    musica = resultados["Brandemburgo.wav"]
    
    print("\n  📊 GRADIENTE DE DIFERENCIACIÓN:")
    print(f"    Ruido: {ruido['gradiente_media']:.4f}")
    print(f"    Tono: {tono['gradiente_media']:.4f}")
    print(f"    Voz: {voz['gradiente_media']:.4f}")
    print(f"    Música: {musica['gradiente_media']:.4f}")
    
    if voz['gradiente_media'] > ruido['gradiente_media'] and musica['gradiente_media'] > ruido['gradiente_media']:
        print("    ✅ Estímulos estructurados producen mayor gradiente.")
    else:
        print("    ⚠️  Gradiente no discrimina.")
    
    print("\n  📊 VARIANZA DE FLUCTUACIONES (firma de estructura):")
    print(f"    Ruido: {ruido['varianza_fluctuaciones']:.6f}")
    print(f"    Tono: {tono['varianza_fluctuaciones']:.6f}")
    print(f"    Voz: {voz['varianza_fluctuaciones']:.6f}")
    print(f"    Música: {musica['varianza_fluctuaciones']:.6f}")
    
    # Predicción: tono varianza baja, voz varianza media-alta, ruido varianza alta
    if (tono['varianza_fluctuaciones'] < voz['varianza_fluctuaciones'] < ruido['varianza_fluctuaciones'] or
        tono['varianza_fluctuaciones'] < musica['varianza_fluctuaciones'] < ruido['varianza_fluctuaciones']):
        print("    ✅ Varianza de fluctuaciones discrimina por tipo de estructura.")
    else:
        print("    ⚠️  Varianza de fluctuaciones no discrimina como se esperaba.")
    
    print("\n  🚀 VELOCIDAD DE TRANSICIÓN (umbral 90%):")
    for trans, vel in velocidades.items():
        print(f"    {trans}: {vel:.3f} s")
    
    # Verificar asimetría (ruido→voz más lento que voz→ruido)
    vel_ruido_voz = velocidades.get("Ruido blanc→Voz_Estudi", 0)
    vel_voz_ruido = velocidades.get("Voz_Estudio→Ruido blan", 0)
    
    if vel_ruido_voz > vel_voz_ruido:
        print("\n    ✅ Transición asimétrica: construir estructura (ruido→voz) es más lento que destruirla (voz→ruido).")
    else:
        print("\n    ⚠️  Transición no asimétrica (ajustar umbral o resolución).")
    
    # ============================================================
    # VISUALIZACIÓN
    # ============================================================
    print("\n[5] Generando visualización...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Gráfico 1: Gradiente medio
    ax = axes[0, 0]
    nombres = [est[:15] for est in estimulos]
    medias = [resultados[est]['gradiente_media'] for est in estimulos]
    stds = [resultados[est]['gradiente_std'] for est in estimulos]
    ax.bar(nombres, medias, yerr=stds, capsize=5, color=['gray', 'blue', 'green', 'orange'])
    ax.set_ylabel('Gradiente medio')
    ax.set_title('Gradiente de diferenciación')
    ax.grid(True, alpha=0.3)
    
    # Gráfico 2: Varianza de fluctuaciones
    ax = axes[0, 1]
    varianzas = [resultados[est]['varianza_fluctuaciones'] for est in estimulos]
    ax.bar(nombres, varianzas, color=['gray', 'blue', 'green', 'orange'])
    ax.set_ylabel('Varianza de fluctuaciones')
    ax.set_title('Firma de estructura (varianza del gradiente)')
    ax.grid(True, alpha=0.3)
    
    # Gráfico 3: Evolución del gradiente (voz vs ruido)
    ax = axes[0, 2]
    ax.plot(resultados["Voz_Estudio.wav"]['historial_gradientes'][:500], label='Voz', alpha=0.8)
    ax.plot(resultados["Ruido blanco"]['historial_gradientes'][:500], label='Ruido', alpha=0.8)
    ax.set_xlabel('Paso de tiempo (x100)')
    ax.set_ylabel('Gradiente')
    ax.set_title('Evolución del gradiente')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 4: Transiciones (todas)
    ax = axes[1, 0]
    trans_nombres = list(velocidades.keys())
    trans_vals = list(velocidades.values())
    colores = ['green' if 'ruido→voz' in t.lower() or 'blanc→voz' in t.lower() 
               else 'red' if 'voz→ruido' in t.lower()
               else 'blue' for t in trans_nombres]
    ax.bar(trans_nombres, trans_vals, color=colores)
    ax.set_ylabel('Tiempo (s)')
    ax.set_title('Velocidad de transición (umbral 90%)')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.grid(True, alpha=0.3)
    
    # Gráfico 5: Gradiente vs Varianza (espacio de fases)
    ax = axes[1, 1]
    for est in estimulos:
        r = resultados[est]
        ax.scatter(r['gradiente_media'], r['varianza_fluctuaciones'], 
                   s=150, label=est[:15], alpha=0.7)
        ax.annotate(est[:10], (r['gradiente_media'], r['varianza_fluctuaciones']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax.set_xlabel('Gradiente medio')
    ax.set_ylabel('Varianza de fluctuaciones')
    ax.set_title('Espacio de fases de regímenes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 6: Estabilidad
    ax = axes[1, 2]
    estabs = [resultados[est]['estabilidad'] for est in estimulos]
    ax.bar(nombres, estabs, color=['gray', 'blue', 'green', 'orange'])
    ax.set_ylabel('Estabilidad (autocorrelación)')
    ax.set_title('Estabilidad del régimen')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('VSTCosmos v71 - Campo Continuo con Métricas Completas', fontsize=14)
    plt.tight_layout()
    plt.savefig('v71_metricas_completas.png', dpi=150)
    print("  Gráfico guardado: v71_metricas_completas.png")
    
    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)


if __name__ == "__main__":
    main()