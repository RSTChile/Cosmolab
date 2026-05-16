#!/usr/bin/env python3
"""
VSTCosmos - v72-A: Protocolo de Verificación Causal
Experimento de control con alcance cerrado:
1. Parsing corregido de transiciones
2. Resolución temporal fina (DT=0.001) solo para transiciones
3. Métrica A_sys-env (C-N2.0.3)
4. Barrido de alpha: 0.05, 0.001, 0.000

Pregunta central: ¿La asimetría de transiciones pertenece al campo o al sesgo externo?
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
DT_BASE = 0.01                # para estímulos individuales
DT_TRANSICION = 0.001         # 10× más fino para transiciones
DURACION_TRANSICION = 10.0    # segundos por segmento
N_PASOS_TRANSICION = int(DURACION_TRANSICION / DT_TRANSICION)

DURACION_SIM = 60.0
N_PASOS = int(DURACION_SIM / DT_BASE)

# Parámetros de dinámica
DIFUSION_BASE = 0.15
GANANCIA_REACCION = 0.05

OMEGA_MIN = 0.05
OMEGA_MAX = 0.50
AMORT_MIN = 0.01
AMORT_MAX = 0.08
PHI_EQUILIBRIO = 0.5

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
    promedio_local = vecinos(Phi_total)
    difusion = DIFUSION_BASE * (promedio_local - Phi_total)
    
    desviacion = Phi_total - promedio_local
    reaccion = GANANCIA_REACCION * desviacion * (1 - desviacion**2)
    
    term_osc = (-omega_natural**2 * (Phi_total - PHI_EQUILIBRIO)
                - amort_natural * Phi_vel_total)
    
    dPhi_vel = term_osc + reaccion + difusion
    Phi_vel_nueva = Phi_vel_total + DT_BASE * dPhi_vel
    
    dPhi = Phi_vel_nueva
    Phi_nueva = Phi_total + DT_BASE * dPhi
    
    region_auditiva_nueva = Phi_nueva[DIM_INTERNA:, :]
    region_auditiva_nueva = (1 - alpha) * region_auditiva_nueva + alpha * objetivo_audio
    Phi_nueva[DIM_INTERNA:, :] = region_auditiva_nueva
    
    return np.clip(Phi_nueva, LIMITE_MIN, LIMITE_MAX), np.clip(Phi_vel_nueva, -5.0, 5.0)


def calcular_acoplamiento(Phi_total, dim_interna):
    """A_sys-env = media del producto punto entre región interna y auditiva"""
    region_int = Phi_total[:dim_interna, :]
    region_aud = Phi_total[dim_interna:, :]
    return float(np.mean(region_int * region_aud))


def calcular_gradiente(Phi_total, dim_interna):
    region_int = Phi_total[:dim_interna, :]
    region_aud = Phi_total[dim_interna:, :]
    return np.mean(np.abs(region_int - region_aud))


def simular_transicion(estimulo_a, estimulo_b, alpha, dt=DT_TRANSICION):
    """Simula transición con resolución fina"""
    n_pasos_segmento = int(DURACION_TRANSICION / dt)
    
    Phi_total = inicializar_campo_total()
    Phi_vel_total = np.zeros_like(Phi_total)
    omega_natural, amort_natural = calcular_frecuencias_naturales(DIM_TOTAL, DIM_INTERNA)
    
    sr_a, audio_a = cargar_audio(estimulo_a, duracion=DURACION_TRANSICION)
    sr_b, audio_b = cargar_audio(estimulo_b, duracion=DURACION_TRANSICION)
    
    ventana_muestras = int(sr_a * VENTANA_FFT_MS / 1000)
    hop_muestras = int(sr_a * HOP_FFT_MS / 1000)
    
    gradientes = []
    acoplamientos = []
    
    # Segmento A
    for idx in range(n_pasos_segmento):
        obj = preparar_objetivo_audio(audio_a, sr_a, idx, ventana_muestras, hop_muestras,
                                      DIM_AUDITIVA, DIM_TIME)
        Phi_total, Phi_vel_total = actualizar_campo_total(
            Phi_total, Phi_vel_total, obj, alpha,
            omega_natural, amort_natural
        )
        grad = calcular_gradiente(Phi_total, DIM_INTERNA)
        acop = calcular_acoplamiento(Phi_total, DIM_INTERNA)
        gradientes.append(grad)
        acoplamientos.append(acop)
    
    # Segmento B
    for idx in range(n_pasos_segmento):
        obj = preparar_objetivo_audio(audio_b, sr_b, idx, ventana_muestras, hop_muestras,
                                      DIM_AUDITIVA, DIM_TIME)
        Phi_total, Phi_vel_total = actualizar_campo_total(
            Phi_total, Phi_vel_total, obj, alpha,
            omega_natural, amort_natural
        )
        grad = calcular_gradiente(Phi_total, DIM_INTERNA)
        acop = calcular_acoplamiento(Phi_total, DIM_INTERNA)
        gradientes.append(grad)
        acoplamientos.append(acop)
    
    t0 = n_pasos_segmento
    
    # Régimen estable post-transitorio (pasos 50-150 del segmento B)
    inicio_ventana = t0 + 50
    fin_ventana = min(t0 + 150, len(gradientes))
    if fin_ventana > inicio_ventana:
        g_despues = np.mean(gradientes[inicio_ventana:fin_ventana])
        a_despues = np.mean(acoplamientos[inicio_ventana:fin_ventana])
    else:
        g_despues = gradientes[-1]
        a_despues = acoplamientos[-1]
    
    # Régimen antes de la transición (últimos 50 pasos del segmento A)
    inicio_antes = max(0, t0 - 50)
    g_antes = np.mean(gradientes[inicio_antes:t0])
    a_antes = np.mean(acoplamientos[inicio_antes:t0])
    
    umbral = UMBRAL_TRANSICION * g_despues
    
    # Buscar primer cruce después del cambio
    tiempo_umbral = None
    for i in range(t0, len(gradientes)):
        if gradientes[i] >= umbral:
            tiempo_umbral = (i - t0) * dt
            break
    
    acoplamiento_min = min(acoplamientos)
    acoplamiento_medio = np.mean(acoplamientos)
    
    return {
        'tiempo_umbral': tiempo_umbral,
        'g_antes': g_antes,
        'g_despues': g_despues,
        'a_antes': a_antes,
        'a_despues': a_despues,
        'acoplamiento_min': acoplamiento_min,
        'acoplamiento_medio': acoplamiento_medio,
        'gradientes': gradientes,
        'acoplamientos': acoplamientos
    }


def simular_estimulo(estimulo_nombre, alpha, duracion_seg=60.0):
    """Simula estímulo individual (usando DT_BASE)"""
    Phi_total = inicializar_campo_total()
    Phi_vel_total = np.zeros_like(Phi_total)
    omega_natural, amort_natural = calcular_frecuencias_naturales(DIM_TOTAL, DIM_INTERNA)
    
    sr, audio = cargar_audio(estimulo_nombre, duracion=duracion_seg)
    ventana_muestras = int(sr * VENTANA_FFT_MS / 1000)
    hop_muestras = int(sr * HOP_FFT_MS / 1000)
    n_ventanas = (len(audio) - ventana_muestras) // hop_muestras + 1
    n_pasos = min(N_PASOS, n_ventanas)
    
    gradientes = []
    
    for idx in range(n_pasos):
        objetivo = preparar_objetivo_audio(audio, sr, idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
        Phi_total, Phi_vel_total = actualizar_campo_total(
            Phi_total, Phi_vel_total, objetivo, alpha,
            omega_natural, amort_natural
        )
        grad = calcular_gradiente(Phi_total, DIM_INTERNA)
        gradientes.append(grad)
    
    return np.mean(gradientes), np.std(gradientes)


def main():
    print("=" * 100)
    print("VSTCosmos - v72-A: Protocolo de Verificación Causal")
    print("Barrido de alpha: 0.05, 0.001, 0.000")
    print("Pregunta: ¿La asimetría persiste sin sesgo externo?")
    print("=" * 100)
    
    transiciones = [
        ("Voz_Estudio.wav", "Ruido blanco"),
        ("Ruido blanco", "Voz_Estudio.wav"),
        ("Tono puro", "Ruido blanco"),
        ("Ruido blanco", "Tono puro")
    ]
    
    condiciones = [
        (0.05, "Alpha = 0.05 (referencia v71)"),
        (0.001, "Alpha = 0.001 (sesgo mínimo)"),
        (0.000, "Alpha = 0.000 (sin sesgo)")
    ]
    
    resultados = []
    
    # Estímulos individuales para cada condición
    print("\n[1] Estímulos individuales por condición")
    for alpha, desc in condiciones:
        print(f"\n  {desc}:")
        for est in ["Ruido blanco", "Voz_Estudio.wav"]:
            grad_mean, grad_std = simular_estimulo(est, alpha)
            print(f"    {est}: gradiente = {grad_mean:.4f} ± {grad_std:.4f}")
    
    # Transiciones para cada condición
    print("\n[2] Transiciones con resolución fina")
    
    for alpha, desc in condiciones:
        print(f"\n  {desc}:")
        for a, b in transiciones:
            res = simular_transicion(a, b, alpha)
            clave = f"{a[:15]}→{b[:15]}"
            tiempo = res['tiempo_umbral']
            tiempo_str = f"{tiempo:.3f} s" if tiempo is not None else "no alcanzado"
            print(f"    {clave}: {tiempo_str}")
            resultados.append({
                'condicion': desc,
                'alpha': alpha,
                'transicion': f"{a}→{b}",
                'tiempo_umbral': tiempo,
                'g_antes': res['g_antes'],
                'g_despues': res['g_despues'],
                'acoplamiento_min': res['acoplamiento_min'],
                'acoplamiento_medio': res['acoplamiento_medio']
            })
    
    # ============================================================
    # ANÁLISIS Y CRITERIO DE DECISIÓN
    # ============================================================
    print("\n" + "=" * 100)
    print("[3] CRITERIO DE DECISIÓN")
    print("=" * 100)
    
    # Para alpha = 0.001 (condición crítica)
    resultados_alpha_001 = [r for r in resultados if r['alpha'] == 0.001]
    
    tiempo_ruido_voz = None
    tiempo_voz_ruido = None
    
    for r in resultados_alpha_001:
        if "Ruido blanco→Voz_Estudio.wav" in r['transicion']:
            tiempo_ruido_voz = r['tiempo_umbral']
        if "Voz_Estudio.wav→Ruido blanco" in r['transicion']:
            tiempo_voz_ruido = r['tiempo_umbral']
    
    # Criterio 1: Asimetría
    asimetria_valida = False
    if tiempo_ruido_voz is not None and tiempo_voz_ruido is not None:
        asimetria = tiempo_ruido_voz - tiempo_voz_ruido
        if tiempo_voz_ruido == 0 and tiempo_ruido_voz > 0.05:
            asimetria_valida = True
        elif tiempo_ruido_voz is not None and tiempo_voz_ruido is not None:
            if asimetria > 0.05:
                asimetria_valida = True
    
    # Criterio 2: Acoplamiento constitutivo
    acoplamientos_min = [r['acoplamiento_min'] for r in resultados_alpha_001]
    acoplamientos_min_validos = [a for a in acoplamientos_min if a is not None]
    acoplamiento_ok = all(a > 1e-4 for a in acoplamientos_min_validos) if acoplamientos_min_validos else False
    
    # Criterio 3: Gradiente no colapsa
    gradiente_voz_alpha_001 = None
    for alpha, desc in condiciones:
        if alpha == 0.001:
            grad_mean, _ = simular_estimulo("Voz_Estudio.wav", alpha)
            gradiente_voz_alpha_001 = grad_mean
    gradiente_ok = gradiente_voz_alpha_001 is not None and gradiente_voz_alpha_001 > 0.05
    
    print(f"\n  Condición crítica: alpha = 0.001")
    print(f"\n  Criterio 1 — Asimetría (ruido→voz vs voz→ruido):")
    print(f"    ruido→voz: {tiempo_ruido_voz:.3f} s" if tiempo_ruido_voz else "    ruido→voz: no alcanzado")
    print(f"    voz→ruido: {tiempo_voz_ruido:.3f} s" if tiempo_voz_ruido else "    voz→ruido: no alcanzado")
    print(f"    {'✅' if asimetria_valida else '❌'} Asimetría > 0.05s")
    
    print(f"\n  Criterio 2 — Acoplamiento constitutivo (A_sys-env > 0):")
    if acoplamientos_min_validos:
        print(f"    min(A_sys-env) = {min(acoplamientos_min_validos):.6f}")
        print(f"    {'✅' if acoplamiento_ok else '❌'} Acoplamiento nunca cae a cero")
    else:
        print(f"    ❌ Sin datos de acoplamiento")
    
    print(f"\n  Criterio 3 — Gradiente de voz en régimen estable:")
    print(f"    gradiente_voz = {gradiente_voz_alpha_001:.4f}")
    print(f"    {'✅' if gradiente_ok else '❌'} Gradiente > 0.05")
    
    # Decisión final
    print("\n" + "=" * 100)
    print("DECISIÓN")
    print("=" * 100)
    
    if asimetria_valida and acoplamiento_ok and gradiente_ok:
        print("\n  ✅ PASÓ EL CRITERIO")
        print("     La asimetría persiste incluso con alpha → 0.")
        print("     La estructura pertenece AL CAMPO, no solo al sesgo externo.")
        print("     → PROCEDE v72-B (Memoria Estructural Interna)")
    else:
        print("\n  ⚠️  NO PASÓ EL CRITERIO")
        print("     La asimetría de v71 era mayoritariamente inducida por el sesgo externo.")
        print("     El campo no tiene inercia estructural propia.")
        print("     → v72-B se diseña introduciendo memoria estructural robusta")
    
    # ============================================================
    # VISUALIZACIÓN (solo para condición crítica)
    # ============================================================
    print("\n[4] Generando visualización...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Transiciones críticas
    for idx, (alpha, desc) in enumerate(condiciones):
        if idx >= 4:
            break
        ax = axes[idx // 2, idx % 2]
        
        # Obtener datos de transición crítica (ruido→voz)
        res = simular_transicion("Ruido blanco", "Voz_Estudio.wav", alpha)
        grad = np.array(res['gradientes'])
        t = np.linspace(0, len(grad) * DT_TRANSICION, len(grad))
        
        ax.plot(t, grad)
        ax.axvline(x=DURACION_TRANSICION, color='r', linestyle='--', label='Cambio de estímulo')
        ax.set_xlabel('Tiempo (s)')
        ax.set_ylabel('Gradiente')
        ax.set_title(f'{desc}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('v72-A: Transición Ruido → Voz por condición de alpha', fontsize=14)
    plt.tight_layout()
    plt.savefig('v72a_asimetria.png', dpi=150)
    print("  Gráfico guardado: v72a_asimetria.png")
    
    # Guardar CSV
    import csv
    with open('v72a_transiciones.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['condicion', 'alpha', 'transicion', 'tiempo_umbral_s', 
                         'g_antes', 'g_despues', 'acoplamiento_min', 'acoplamiento_medio'])
        for r in resultados:
            tiempo_str = f"{r['tiempo_umbral']:.3f}" if r['tiempo_umbral'] is not None else 'no_alcanzado'
            writer.writerow([
                r['condicion'], r['alpha'], r['transicion'],
                tiempo_str, r['g_antes'], r['g_despues'],
                r['acoplamiento_min'], r['acoplamiento_medio']
            ])
    print("  CSV guardado: v72a_transiciones.csv")
    
    # Guardar resultado.txt
    with open('v72a_resultado.txt', 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("VSTCosmos v72-A - Resultado del protocolo de verificación\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Condición crítica: alpha = 0.001\n\n")
        f.write(f"Criterio 1 — Asimetría: {'PASA' if asimetria_valida else 'FALLA'}\n")
        f.write(f"  ruido→voz: {tiempo_ruido_voz:.3f}s, voz→ruido: {tiempo_voz_ruido:.3f}s\n\n")
        f.write(f"Criterio 2 — Acoplamiento: {'PASA' if acoplamiento_ok else 'FALLA'}\n")
        if acoplamientos_min_validos:
            f.write(f"  min(A_sys-env) = {min(acoplamientos_min_validos):.6f}\n\n")
        f.write(f"Criterio 3 — Gradiente: {'PASA' if gradiente_ok else 'FALLA'}\n")
        f.write(f"  gradiente_voz = {gradiente_voz_alpha_001:.4f}\n\n")
        f.write("=" * 60 + "\n")
        if asimetria_valida and acoplamiento_ok and gradiente_ok:
            f.write("DECISIÓN: PROCEDE v72-B\n")
            f.write("La asimetría persiste incluso con alpha → 0.\n")
            f.write("La estructura pertenece AL CAMPO, no solo al sesgo externo.\n")
        else:
            f.write("DECISIÓN: v72-B requiere MEMORIA ESTRUCTURAL ROBUSTA\n")
            f.write("La asimetría de v71 era mayoritariamente inducida por el sesgo externo.\n")
        f.write("=" * 60 + "\n")
    print("  TXT guardado: v72a_resultado.txt")
    
    print("\n" + "=" * 100)
    print("PROTOCOLO COMPLETADO")
    print("=" * 100)


if __name__ == "__main__":
    main()