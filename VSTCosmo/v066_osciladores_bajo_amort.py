#!/usr/bin/env python3
"""
VSTCosmos - v66: Osciladores con Amortiguamiento Bajo
EXPERIMENTO DE DIAGNÓSTICO - Sin aprendizaje, sin memoria, sin selección.
El campo Φ es un sistema de osciladores con frecuencias naturales propias.
Amortiguamiento reducido para permitir oscilación real.
GANANCIA_ESPECTRAL aumentada para que la perturbación tenga efecto.
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PARÁMETROS MÍNIMOS
# ============================================================
DIM_FREQ = 32
DIM_TIME = 100
DT = 0.01
DURACION_SIM = 20.0
N_PASOS = int(DURACION_SIM / DT)

DECAIMIENTO_PHI = 0.01
GANANCIA_GENERACION_BASE = 0.05
GANANCIA_SOSTENIMIENTO = 0.25
DIFUSION_BASE = 0.20

REFUERZO_A = 0.15
INHIBICION_A = 0.2
DIFUSION_A = 0.08
FUERZA_RELIEVE = 0.08
K_COMP_BASE = 0.05

LIMITE_ATENCION = DIM_FREQ * DIM_TIME * 0.35
INHIB_GLOBAL = 0.5

MOD_DECAY = 1.0
MOD_GENERACION = 1.5

TASA_CRECIMIENTO = 0.10
TASA_DISIPACION = 0.03
GANANCIA_HISTORIA = 0.5
FUERZA_ESTABILIDAD = 0.1
BLOQUEO_MAXIMO_BASE = 0.8

TASA_OMEGA = 0.15
DISIPACION_OMEGA = 0.05
FUERZA_COHERENCIA = 0.2

LIMITE_MIN = 0.0
LIMITE_MAX = 1.0

VENTANA_FFT_MS = 25
HOP_FFT_MS = 10
F_MIN = 80
F_MAX = 8000

HOMEOSTASIS_INTERVALO = 500

# Parámetros oscilatorios corticales (AMORTIGUAMIENTO REDUCIDO)
OMEGA_MIN = 0.05   # frecuencia natural mínima (bandas bajas)
OMEGA_MAX = 0.50   # frecuencia natural máxima (bandas altas)
AMORT_MIN = 0.01   # MUCHO más bajo (antes 0.05)
AMORT_MAX = 0.08   # MUCHO más bajo (antes 0.40)
GANANCIA_ESPECTRAL = 0.15   # Aumentada (antes 0.08)
PHI_EQUILIBRIO = 0.5        # punto de equilibrio del oscilador

# Semillas para reproducibilidad
N_SEMILLAS = 10


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


def inicializar_campo(seed):
    np.random.seed(seed)
    return np.random.rand(DIM_FREQ, DIM_TIME) * 0.2 + 0.4


def inicializar_atencion(seed):
    np.random.seed(seed + 1)
    return np.ones((DIM_FREQ, DIM_TIME), dtype=np.float32) * 0.1


def inicializar_memoria():
    return np.zeros((DIM_FREQ, DIM_TIME), dtype=np.float32)


def inicializar_anclaje():
    return np.zeros((DIM_FREQ, DIM_TIME), dtype=np.float32)


def vecinos(X):
    return (np.roll(X, 1, axis=0) + np.roll(X, -1, axis=0) +
            np.roll(X, 1, axis=1) + np.roll(X, -1, axis=1)) / 4.0


def calcular_frecuencias_naturales():
    """Frecuencias naturales por banda, escaladas logarítmicamente."""
    bandas = np.arange(DIM_FREQ)
    # Escala logarítmica: más resolución en frecuencias bajas
    t = np.log1p(bandas) / np.log1p(DIM_FREQ - 1)
    
    omega = OMEGA_MIN + (OMEGA_MAX - OMEGA_MIN) * t
    amort = AMORT_MIN + (AMORT_MAX - AMORT_MIN) * t
    
    return omega.reshape(-1, 1), amort.reshape(-1, 1)


def perfil_espectral_ventana(audio, sr, idx_ventana, ventana_muestras, hop_muestras):
    inicio = idx_ventana * hop_muestras
    if inicio + ventana_muestras > len(audio):
        return np.zeros(DIM_FREQ)
    
    fragmento = audio[inicio:inicio + ventana_muestras]
    ventana_hann = np.hanning(len(fragmento))
    fragmento = fragmento * ventana_hann
    
    fft = np.fft.rfft(fragmento)
    potencia = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(len(fragmento), 1/sr)
    
    bandas = np.logspace(np.log10(F_MIN), np.log10(F_MAX), DIM_FREQ + 1)
    perfil = np.zeros(DIM_FREQ)
    
    for b in range(DIM_FREQ):
        mask = (freqs >= bandas[b]) & (freqs < bandas[b+1])
        if np.any(mask):
            perfil[b] = np.mean(potencia[mask])
    
    max_energia = np.max(perfil)
    if max_energia > 0:
        perfil = perfil / max_energia
    
    return perfil


def actualizar_memoria_estabilidad(Psi, Phi, A):
    cambio_local = np.abs(Phi - vecinos(Phi))
    cambio_norm = cambio_local / (cambio_local + 0.1)
    dPsi = (TASA_CRECIMIENTO * A * (1 - cambio_norm) * (1 - Psi) -
            TASA_DISIPACION * Psi) * DT
    Psi = Psi + dPsi
    return np.clip(Psi, 0.0, 1.0)


def actualizar_memoria_coherencia(Omega, Phi, Phi_prev, Phi_prev2, A):
    cambio_actual = Phi - Phi_prev
    cambio_anterior = Phi_prev - Phi_prev2
    signo_actual = np.tanh(cambio_actual * 10)
    signo_anterior = np.tanh(cambio_anterior * 10)
    coherencia = (1 + signo_actual * signo_anterior) / 2
    dOmega = (TASA_OMEGA * A * coherencia * (1 - Omega) -
              DISIPACION_OMEGA * Omega) * DT
    Omega = Omega + dOmega
    return np.clip(Omega, 0.0, 1.0)


def actualizar_campo_v66(Phi, Phi_vel, A, perfil, Psi, 
                          ganancia_gen, bloqueo_max, K,
                          omega_natural, amort_natural):
    """
    v66: Campo con frecuencias naturales + amortiguamiento bajo.
    """
    es_silencio = np.max(perfil) < 1e-6
    
    if es_silencio:
        perfil_2d = np.zeros((DIM_FREQ, 1))
        mod_entrada = 1.0
        mod_entrada_decay = 1.0
    else:
        perfil_2d = perfil.reshape(-1, 1)
        mod_entrada = (1 + MOD_GENERACION * perfil_2d) * K
        mod_entrada_decay = (1 - MOD_DECAY * perfil_2d) * K
    
    promedio_local = vecinos(Phi)
    
    difusion = DIFUSION_BASE * (promedio_local - Phi)
    
    desviacion = Phi - promedio_local
    generacion_base = ganancia_gen * desviacion * (1 - desviacion**2)
    mod_memoria = (1 - GANANCIA_HISTORIA * Psi)
    generacion = generacion_base * mod_entrada * mod_memoria
    
    decaimiento = -DECAIMIENTO_PHI * (Phi - promedio_local) * mod_entrada_decay
    
    sostenimiento = GANANCIA_SOSTENIMIENTO * A * (Phi - promedio_local)
    
    # Término espectral (perturbación)
    termino_espectral = GANANCIA_ESPECTRAL * (perfil_2d - Phi)
    
    # Término oscilatorio con amortiguamiento bajo
    termino_oscilatorio = (-omega_natural**2 * (Phi - PHI_EQUILIBRIO) 
                           - amort_natural * Phi_vel)
    
    # Actualización de velocidad (segundo orden)
    dPhi_vel = termino_oscilatorio + termino_espectral
    Phi_vel_nueva = Phi_vel + DT * dPhi_vel
    
    # Actualización del campo
    dPhi_propuesto = (difusion + generacion + decaimiento + 
                      sostenimiento + Phi_vel_nueva)
    dPhi_real = dPhi_propuesto * (1 - bloqueo_max * Psi)
    
    Phi_nueva = Phi + DT * dPhi_real
    
    return (np.clip(Phi_nueva, LIMITE_MIN, LIMITE_MAX), 
            np.clip(Phi_vel_nueva, -5.0, 5.0))  # Rango ampliado para oscilación


def actualizar_atencion(A, Phi, Psi, Omega, k_comp):
    vA = vecinos(A)
    auto = REFUERZO_A * A * (1.0 - A)
    inhib_local = -INHIBICION_A * vA
    difusion = DIFUSION_A * (vA - A)
    
    relieve_local = np.abs(Phi - vecinos(Phi))
    max_relieve = np.max(relieve_local)
    if max_relieve > 0:
        relieve_local = relieve_local / max_relieve
    
    acoplamiento_relieve = FUERZA_RELIEVE * (relieve_local - A)
    acoplamiento_estabilidad = FUERZA_ESTABILIDAD * (Psi - A)
    acoplamiento_coherencia = FUERZA_COHERENCIA * (Omega - A)
    
    A_mean = np.mean(A)
    competencia = -k_comp * (A - A_mean)
    
    dA = (auto + inhib_local + difusion +
          acoplamiento_relieve +
          acoplamiento_estabilidad +
          acoplamiento_coherencia +
          competencia)
    
    atencion_total = np.sum(A)
    if atencion_total > LIMITE_ATENCION:
        exceso = (atencion_total - LIMITE_ATENCION) / LIMITE_ATENCION
        dA += -INHIB_GLOBAL * exceso * A
    
    dA += np.random.randn(*A.shape) * 0.001
    A = A + DT * dA
    return np.clip(A, LIMITE_MIN, LIMITE_MAX)


def simular_estimulo_v66(entrada_nombre, seed):
    np.random.seed(seed)
    
    Phi = inicializar_campo(seed)
    Phi_vel = np.zeros((DIM_FREQ, DIM_TIME))
    A = inicializar_atencion(seed)
    Psi = inicializar_memoria()
    Omega = inicializar_memoria()
    
    # Precalcular frecuencias naturales (constantes durante la simulación)
    omega_natural, amort_natural = calcular_frecuencias_naturales()
    
    sr, audio = cargar_audio(entrada_nombre)
    
    ventana_muestras = int(sr * VENTANA_FFT_MS / 1000)
    hop_muestras = int(sr * HOP_FFT_MS / 1000)
    n_ventanas = (len(audio) - ventana_muestras) // hop_muestras + 1
    
    ganancia_gen = GANANCIA_GENERACION_BASE
    bloqueo_max = BLOQUEO_MAXIMO_BASE
    k_comp = K_COMP_BASE
    
    evolucion_phi = []
    evolucion_energia_vel = []
    evolucion_phi_std = []
    
    for idx_ventana in range(min(N_PASOS, n_ventanas)):
        perfil = perfil_espectral_ventana(audio, sr, idx_ventana, ventana_muestras, hop_muestras)
        
        A = actualizar_atencion(A, Phi, Psi, Omega, k_comp)
        Phi, Phi_vel = actualizar_campo_v66(
            Phi, Phi_vel, A, perfil, Psi,
            ganancia_gen, bloqueo_max, K=1.0,
            omega_natural=omega_natural, amort_natural=amort_natural
        )
        Psi = actualizar_memoria_estabilidad(Psi, Phi, A)
        Omega = actualizar_memoria_coherencia(Omega, Phi, Phi, Phi, A)
        
        if idx_ventana % 100 == 0:
            evolucion_phi.append(np.mean(Phi))
            evolucion_energia_vel.append(np.mean(Phi_vel ** 2))
            evolucion_phi_std.append(np.std(Phi))
    
    # Perfil promedio para correlación
    sr, audio_long = cargar_audio(entrada_nombre, duracion=2.0)
    ventana_muestras_long = int(sr * VENTANA_FFT_MS / 1000)
    hop_muestras_long = int(sr * HOP_FFT_MS / 1000)
    n_ventanas_long = min(100, (len(audio_long) - ventana_muestras_long) // hop_muestras_long + 1)
    perfiles = []
    for i in range(n_ventanas_long):
        perf = perfil_espectral_ventana(audio_long, sr, i, ventana_muestras_long, hop_muestras_long)
        perfiles.append(perf)
    perfil_promedio = np.mean(perfiles, axis=0)
    
    # Energía oscilatoria y estadísticas
    energia_vel_por_banda = np.mean(Phi_vel ** 2, axis=1)
    energia_vel_total = np.mean(Phi_vel ** 2)
    
    return {
        'Phi': Phi,
        'Phi_vel': Phi_vel,
        'rango_Phi': np.max(Phi) - np.min(Phi),
        'media_Phi': np.mean(Phi),
        'energia_oscilatoria_total': energia_vel_total,
        'energia_oscilatoria_por_banda': energia_vel_por_banda,
        'evolucion_phi': evolucion_phi,
        'evolucion_energia': evolucion_energia_vel,
        'perfil_promedio': perfil_promedio,
        'std_final_phi': np.std(Phi)
    }


def distancia_estados(estado1, estado2):
    return np.mean((estado1['Phi'] - estado2['Phi']) ** 2)


def main():
    opciones = [
        "Ruido blanco",
        "Tono puro",
        "Voz_Estudio.wav",
        "Brandemburgo.wav",
        "Voz+Viento_1.wav"
    ]
    
    print("=" * 100)
    print("VSTCosmos - v66: Osciladores con Amortiguamiento Bajo")
    print("EXPERIMENTO DE DIAGNÓSTICO")
    print(f"AMORT_MIN = {AMORT_MIN}, AMORT_MAX = {AMORT_MAX}")
    print(f"GANANCIA_ESPECTRAL = {GANANCIA_ESPECTRAL}")
    print("=" * 100)
    
    estados = {opcion: [] for opcion in opciones}
    
    print("\n[1] Ejecutando simulaciones...")
    print("-" * 80)
    
    for opcion in opciones:
        print(f"\n  Estímulo: {opcion}")
        for rep in range(N_SEMILLAS):
            seed = rep * 100
            estado = simular_estimulo_v66(opcion, seed)
            estados[opcion].append(estado)
            print(f"    Semilla {rep:2d}: rango_Φ={estado['rango_Phi']:.4f}, "
                  f"energía_vel={estado['energia_oscilatoria_total']:.6f}, "
                  f"std_Φ={estado['std_final_phi']:.4f}")
    
    print("\n" + "=" * 100)
    print("[2] ANÁLISIS DE DISTANCIAS (inter vs intra)")
    print("=" * 100)
    
    intra_distancias = {}
    for opcion in opciones:
        distancias = []
        for i in range(len(estados[opcion])):
            for j in range(i+1, len(estados[opcion])):
                d = distancia_estados(estados[opcion][i], estados[opcion][j])
                distancias.append(d)
        intra = np.mean(distancias) if distancias else 0
        intra_distancias[opcion] = intra
        print(f"  {opcion:<25}: intra={intra:.8f}")
    
    print("\n[3] RESONANCIA Y ACOPLAMIENTO")
    print("=" * 100)
    
    correlaciones = {}
    for opcion in opciones:
        estado = estados[opcion][-1]
        Phi_flat = np.mean(estado['Phi'], axis=1)
        perfil_prom = estado['perfil_promedio']
        correlacion = np.corrcoef(Phi_flat, perfil_prom)[0, 1]
        correlaciones[opcion] = correlacion
        
        print(f"\n  {opcion}:")
        print(f"    Energía oscilatoria total: {estado['energia_oscilatoria_total']:.6f}")
        print(f"    Desviación estándar de Φ: {estado['std_final_phi']:.4f}")
        print(f"    Correlación Φ-perfil: {correlacion:.3f}")
        
        if estado['energia_oscilatoria_total'] > 0.0005:
            print(f"    ✅ OSCILACIÓN DETECTADA")
        else:
            print(f"    ❌ OSCILACIÓN MUY BAJA")
        
        if 0.4 <= correlacion <= 0.7:
            print(f"    ✅ ACOPLAMIENTO MODERADO (resonancia)")
        elif correlacion > 0.7:
            print(f"    ⚠️  ALTA CORRELACIÓN (posible copia)")
        else:
            print(f"    ❌ BAJA CORRELACIÓN")
    
    print("\n" + "=" * 100)
    print("[4] DIAGNÓSTICO FINAL")
    print("=" * 100)
    
    # Discriminación entre ruido y voz
    ruido_intra = intra_distancias["Ruido blanco"]
    voz_intra = intra_distancias["Voz_Estudio.wav"]
    
    dists = []
    for rep in range(N_SEMILLAS):
        d = distancia_estados(estados["Ruido blanco"][rep], estados["Voz_Estudio.wav"][rep])
        dists.append(d)
    ruido_voz_inter = np.mean(dists)
    
    relacion_ruido = ruido_voz_inter / ruido_intra if ruido_intra > 0 else 0
    relacion_voz = ruido_voz_inter / voz_intra if voz_intra > 0 else 0
    
    print(f"\n  Discriminación ruido vs voz:")
    print(f"    Relación inter/intra (ruido): {relacion_ruido:.2f}")
    print(f"    Relación inter/intra (voz): {relacion_voz:.2f}")
    
    # Energías oscilatorias
    energias = [estados[op][0]['energia_oscilatoria_total'] for op in opciones]
    print(f"\n  Energías oscilatorias:")
    for op, eng in zip(opciones, energias):
        print(f"    {op[:20]}: {eng:.6f}")
    
    oscilacion_ok = max(energias) > 0.001
    
    if relacion_ruido > 100 and relacion_voz > 100 and oscilacion_ok:
        print("\n  ✅ V66 EXITOSO")
        print("     → El sistema discrimina Y hay oscilación real.")
        print("     → El mecanismo de resonancia está funcionando.")
    elif relacion_ruido > 100 and relacion_voz > 100:
        print("\n  ⚠️  V66 PARCIAL")
        print("     → Discriminación lograda, pero oscilación aún baja.")
        print("     → Reducir aún más AMORT_MIN/MAX o aumentar GANANCIA_ESPECTRAL.")
    else:
        print("\n  ❌ V66 FALLIDO")
        print("     → Revisar parámetros del oscilador.")
    
    # ============================================================
    # VISUALIZACIÓN
    # ============================================================
    print("\n[5] Generando visualización...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Gráfico 1: Evolución de Φ
    ax = axes[0, 0]
    for opcion in opciones:
        evol = estados[opcion][-1]['evolucion_phi']
        ax.plot(evol, label=opcion[:15])
    ax.set_xlabel('Ventana (x100)')
    ax.set_ylabel('Φ medio')
    ax.set_title('Evolución del campo Φ (v66)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 2: Energía oscilatoria por banda
    ax = axes[0, 1]
    for opcion in opciones:
        energia = estados[opcion][0]['energia_oscilatoria_por_banda']
        ax.plot(energia, label=opcion[:15])
    ax.set_xlabel('Banda espectral')
    ax.set_ylabel('Energía oscilatoria')
    ax.set_title('Patrón de resonancia por estímulo')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 3: Evolución de la energía oscilatoria
    ax = axes[0, 2]
    for opcion in opciones:
        evol_energia = estados[opcion][-1]['evolucion_energia']
        ax.plot(evol_energia, label=opcion[:15])
    ax.set_xlabel('Ventana (x100)')
    ax.set_ylabel('Energía oscilatoria')
    ax.set_title('Evolución de la resonancia')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 4: Correlaciones
    ax = axes[1, 0]
    corrs = [correlaciones[op] for op in opciones]
    barras = ax.bar(range(len(opciones)), corrs)
    ax.axhline(y=0.7, color='r', linestyle='--', label='Copia')
    ax.axhline(y=0.4, color='orange', linestyle='--', label='Zona resonancia')
    ax.set_xticks(range(len(opciones)))
    ax.set_xticklabels([op[:12] for op in opciones], rotation=45, ha='right')
    ax.set_ylabel('Correlación Φ-perfil')
    ax.set_title('Acoplamiento por estímulo (v66)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 5: Frecuencias naturales por banda
    ax = axes[1, 1]
    omega_natural, amort_natural = calcular_frecuencias_naturales()
    ax.plot(omega_natural, label='Frecuencia natural', color='b')
    ax.set_xlabel('Banda')
    ax.set_ylabel('Frecuencia natural (Hz)')
    ax.set_title('Frecuencias naturales por banda')
    ax.grid(True, alpha=0.3)
    
    ax2 = ax.twinx()
    ax2.plot(amort_natural, label='Amortiguamiento', color='r', linestyle='--')
    ax2.set_ylabel('Amortiguamiento')
    
    # Gráfico 6: Energía oscilatoria comparativa
    ax = axes[1, 2]
    energias_plot = [estados[op][0]['energia_oscilatoria_total'] for op in opciones]
    barras = ax.bar(range(len(opciones)), energias_plot)
    ax.set_xticks(range(len(opciones)))
    ax.set_xticklabels([op[:12] for op in opciones], rotation=45, ha='right')
    ax.set_ylabel('Energía oscilatoria total')
    ax.set_title('Resonancia por estímulo')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('VSTCosmos v66 - Osciladores con Amortiguamiento Bajo', fontsize=14)
    plt.tight_layout()
    plt.savefig('v66_osciladores_bajo_amort.png', dpi=150)
    print("  Gráfico guardado: v66_osciladores_bajo_amort.png")
    
    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)


if __name__ == "__main__":
    main()