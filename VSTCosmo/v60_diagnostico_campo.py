#!/usr/bin/env python3
"""
VSTCosmos - v60: Caracterización Basal de Φ
EXPERIMENTO DE DIAGNÓSTICO - Sin aprendizaje, sin memoria, sin selección.
¿Puede el campo Φ sostener estados distintos para estímulos distintos?
Criterio: distancia intra-estímulo < distancia inter-estímulo
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PARÁMETROS MÍNIMOS (sin selección, sin memoria)
# ============================================================
DIM_FREQ = 32
DIM_TIME = 100
DT = 0.01
DURACION_SIM = 20.0
N_PASOS = int(DURACION_SIM / DT)

# Parámetros de campo (sin decaimiento adaptativo, sin K)
DECAIMIENTO_PHI = 0.01
GANANCIA_GENERACION_BASE = 0.05
GANANCIA_SOSTENIMIENTO = 0.25
DIFUSION_BASE = 0.20

# Parámetros de atención
REFUERZO_A = 0.15
INHIBICION_A = 0.2
DIFUSION_A = 0.08
FUERZA_RELIEVE = 0.08
K_COMP_BASE = 0.05

LIMITE_ATENCION = DIM_FREQ * DIM_TIME * 0.35
INHIB_GLOBAL = 0.5

MOD_DECAY = 1.0
MOD_GENERACION = 1.5

# Parámetros de memoria de estabilidad y coherencia
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

# Parámetros de entrada espectral
VENTANA_FFT_MS = 25
HOP_FFT_MS = 10
F_MIN = 80
F_MAX = 8000

HOMEOSTASIS_INTERVALO = 500

# Semillas para reproducibilidad
N_SEMILLAS = 10
N_REPETICIONES_POR_SEMILLA = 3  # Mismo estímulo, misma semilla, diferente ruido interno


# ============================================================
# FUNCIONES BASE (sin aprendizaje, sin selección)
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


def actualizar_campo(Phi, A, perfil, Psi, ganancia_gen, bloqueo_max):
    es_silencio = np.max(perfil) < 1e-6
    
    if es_silencio:
        perfil_2d = np.zeros((DIM_FREQ, 1))
        mod_entrada = 1.0
        mod_entrada_decay = 1.0
    else:
        perfil_2d = perfil.reshape(-1, 1)
        mod_entrada = (1 + MOD_GENERACION * perfil_2d)
        mod_entrada_decay = (1 - MOD_DECAY * perfil_2d)
    
    promedio_local = vecinos(Phi)
    
    difusion = DIFUSION_BASE * (promedio_local - Phi)
    
    desviacion = Phi - promedio_local
    generacion_base = ganancia_gen * desviacion * (1 - desviacion**2)
    mod_memoria = (1 - GANANCIA_HISTORIA * Psi)
    generacion = generacion_base * mod_entrada * mod_memoria
    
    decaimiento = -DECAIMIENTO_PHI * (Phi - promedio_local) * mod_entrada_decay
    
    sostenimiento = GANANCIA_SOSTENIMIENTO * A * (Phi - promedio_local)
    
    dPhi_propuesto = difusion + generacion + decaimiento + sostenimiento
    dPhi_real = dPhi_propuesto * (1 - bloqueo_max * Psi)
    
    Phi = Phi + DT * dPhi_real
    return np.clip(Phi, LIMITE_MIN, LIMITE_MAX)


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


def simular_estimulo(entrada_nombre, seed):
    """Simula una experiencia completa con el estímulo dado, partiendo de un estado inicial fijado por seed"""
    np.random.seed(seed)
    
    # Inicializar todo con la misma semilla (estado inicial reproducible)
    Phi = inicializar_campo(seed)
    A = inicializar_atencion(seed)
    Psi = inicializar_memoria()
    Omega = inicializar_memoria()
    L = inicializar_anclaje()
    M = inicializar_memoria()
    
    sr, audio = cargar_audio(entrada_nombre)
    
    ventana_muestras = int(sr * VENTANA_FFT_MS / 1000)
    hop_muestras = int(sr * HOP_FFT_MS / 1000)
    n_ventanas = (len(audio) - ventana_muestras) // hop_muestras + 1
    
    Phi_prev = Phi.copy()
    Phi_prev2 = Phi.copy()
    
    ganancia_gen = GANANCIA_GENERACION_BASE
    bloqueo_max = BLOQUEO_MAXIMO_BASE
    k_comp = K_COMP_BASE
    
    # Registrar evolución cada 100 pasos para análisis
    evolucion_phi = []
    evolucion_a = []
    
    for idx_ventana in range(min(N_PASOS, n_ventanas)):
        perfil = perfil_espectral_ventana(audio, sr, idx_ventana, ventana_muestras, hop_muestras)
        
        A = actualizar_atencion(A, Phi, Psi, Omega, k_comp)
        Phi = actualizar_campo(Phi, A, perfil, Psi, ganancia_gen, bloqueo_max)
        Psi = actualizar_memoria_estabilidad(Psi, Phi, A)
        Omega = actualizar_memoria_coherencia(Omega, Phi, Phi_prev, Phi_prev2, A)
        
        if idx_ventana % 100 == 0:
            evolucion_phi.append(np.mean(Phi))
            evolucion_a.append(np.mean(A))
        
        Phi_prev2 = Phi_prev.copy()
        Phi_prev = Phi.copy()
    
    return {
        'Phi': Phi,
        'A': A,
        'Psi': Psi,
        'Omega': Omega,
        'rango_Phi': np.max(Phi) - np.min(Phi),
        'media_Phi': np.mean(Phi),
        'energia_Phi': np.mean(Phi ** 2),
        'evolucion_phi': evolucion_phi,
        'evolucion_a': evolucion_a
    }


def distancia_estados(estado1, estado2):
    """Distancia Euclidiana normalizada entre dos campos Φ"""
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
    print("VSTCosmos - v60: Caracterización Basal de Φ")
    print("EXPERIMENTO DE DIAGNÓSTICO - Sin aprendizaje, sin memoria, sin selección")
    print("¿Puede el campo Φ sostener estados distintos para estímulos distintos?")
    print("=" * 100)
    
    # Almacenar todos los estados finales por estímulo y semilla
    estados = {opcion: [] for opcion in opciones}
    
    print("\n[1] Ejecutando simulaciones...")
    print("-" * 80)
    
    for opcion in opciones:
        print(f"\n  Estímulo: {opcion}")
        for rep in range(N_SEMILLAS):
            seed = rep * 100  # Semillas distintas para cada repetición
            estado = simular_estimulo(opcion, seed)
            estados[opcion].append(estado)
            print(f"    Semilla {rep:2d}: rango_Φ={estado['rango_Phi']:.4f}, media_Φ={estado['media_Phi']:.4f}")
    
    print("\n" + "=" * 100)
    print("[2] ANÁLISIS DE DISTANCIAS")
    print("=" * 100)
    
    # Calcular distancias intra-estímulo
    print("\n📊 DISTANCIAS INTRA-ESTÍMULO (reproducibilidad):")
    print("   Compara diferentes semillas para el MISMO estímulo")
    print("   Valores pequeños indican que el campo converge al mismo punto")
    print("-" * 80)
    
    intra_distancias = {}
    for opcion in opciones:
        distancias = []
        for i in range(len(estados[opcion])):
            for j in range(i+1, len(estados[opcion])):
                d = distancia_estados(estados[opcion][i], estados[opcion][j])
                distancias.append(d)
        media = np.mean(distancias) if distancias else 0
        std = np.std(distancias) if distancias else 0
        intra_distancias[opcion] = media
        print(f"  {opcion:<25}: media={media:.6f}, std={std:.6f}, n={len(distancias)}")
    
    # Calcular distancias inter-estímulo
    print("\n📊 DISTANCIAS INTER-ESTÍMULO (discriminabilidad):")
    print("   Compara diferentes estímulos con la MISMA semilla")
    print("   Valores grandes indican que Φ responde diferencialmente")
    print("-" * 80)
    
    inter_distancias = {}
    for i, op1 in enumerate(opciones):
        for j, op2 in enumerate(opciones):
            if i < j:
                distancias = []
                for rep in range(N_SEMILLAS):
                    d = distancia_estados(estados[op1][rep], estados[op2][rep])
                    distancias.append(d)
                media = np.mean(distancias)
                std = np.std(distancias)
                inter_distancias[f"{op1[:15]} vs {op2[:15]}"] = media
                print(f"  {op1[:15]} vs {op2[:15]}: media={media:.6f}, std={std:.6f}")
    
    # ============================================================
    # CRITERIO DE ÉXITO
    # ============================================================
    print("\n" + "=" * 100)
    print("[3] DIAGNÓSTICO FINAL")
    print("=" * 100)
    
    # Comparación clave: Ruido Blanco vs Voz_Estudio
    ruido_intra = intra_distancias["Ruido blanco"]
    voz_intra = intra_distancias["Voz_Estudio.wav"]
    ruido_voz_inter = inter_distancias["Ruido blanco vs Voz_Estudio.wav"]
    
    print(f"\n  Distancia intra-ruido: {ruido_intra:.6f}")
    print(f"  Distancia intra-voz:   {voz_intra:.6f}")
    print(f"  Distancia inter (ruido vs voz): {ruido_voz_inter:.6f}")
    print(f"  Relación inter/intra_ruido: {ruido_voz_inter / ruido_intra:.2f}")
    print(f"  Relación inter/intra_voz:   {ruido_voz_inter / voz_intra:.2f}")
    
    criterio_ruido = ruido_voz_inter > ruido_intra * 2
    criterio_voz = ruido_voz_inter > voz_intra * 2
    
    print("\n" + "=" * 100)
    if criterio_ruido and criterio_voz:
        print("  ✅ EL CAMPO Φ SÍ PUEDE DISTINGUIR ESTÍMULOS")
        print("     La distancia inter-estímulo es significativamente mayor")
        print("     que la distancia intra-estímulo.")
        print("\n     → El problema NO está en la dinámica basal de Φ.")
        print("     → Está en cómo usamos esa información (memoria, identidad, etc.)")
    else:
        print("  ❌ EL CAMPO Φ NO PUEDE DISTINGUIR ESTÍMULOS")
        print("     La distancia inter-estímulo es comparable o menor")
        print("     que la distancia intra-estímulo.")
        print("\n     → El problema está en la DINÁMICA BASAL DEL CAMPO.")
        print("     → Φ converge al mismo atractor para todos los estímulos.")
        print("     → Hay que rediseñar la ecuación de actualización de Φ.")
    print("=" * 100)
    
    # ============================================================
    # VISUALIZACIÓN ADICIONAL
    # ============================================================
    print("\n[4] Generando visualización...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Gráfico 1: Evolución de Φ durante la simulación (última semilla)
    ax = axes[0, 0]
    for opcion in opciones:
        evol = estados[opcion][-1]['evolucion_phi']
        ax.plot(evol, label=opcion[:15])
    ax.set_xlabel('Ventana (x100)')
    ax.set_ylabel('Φ medio')
    ax.set_title('Evolución del campo Φ durante la experiencia')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 2: Evolución de A
    ax = axes[0, 1]
    for opcion in opciones:
        evol = estados[opcion][-1]['evolucion_a']
        ax.plot(evol, label=opcion[:15])
    ax.set_xlabel('Ventana (x100)')
    ax.set_ylabel('A medio')
    ax.set_title('Evolución de la atención A')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 3: rango_Φ final por estímulo
    ax = axes[0, 2]
    rangos = {op: [est['rango_Phi'] for est in estados[op]] for op in opciones}
    bp = ax.boxplot(rangos.values(), labels=[op[:12] for op in opciones])
    ax.set_ylabel('rango_Φ final')
    ax.set_title('Distribución de rango_Φ por estímulo')
    ax.grid(True, alpha=0.3)
    
    # Gráfico 4: media_Φ final
    ax = axes[1, 0]
    medias = {op: [est['media_Phi'] for est in estados[op]] for op in opciones}
    bp = ax.boxplot(medias.values(), labels=[op[:12] for op in opciones])
    ax.set_ylabel('media_Φ final')
    ax.set_title('Distribución de media_Φ por estímulo')
    ax.grid(True, alpha=0.3)
    
    # Gráfico 5: Matriz de distancias
    ax = axes[1, 1]
    n_ops = len(opciones)
    matriz_distancias = np.zeros((n_ops, n_ops))
    for i, op1 in enumerate(opciones):
        for j, op2 in enumerate(opciones):
            dists = []
            for rep in range(N_SEMILLAS):
                dists.append(distancia_estados(estados[op1][rep], estados[op2][rep]))
            matriz_distancias[i, j] = np.mean(dists)
    im = ax.imshow(matriz_distancias, cmap='hot', vmin=0, vmax=matriz_distancias.max())
    ax.set_xticks(range(n_ops))
    ax.set_yticks(range(n_ops))
    ax.set_xticklabels([op[:12] for op in opciones], rotation=45, ha='right')
    ax.set_yticklabels([op[:12] for op in opciones])
    ax.set_title('Matriz de distancias medias entre estímulos')
    plt.colorbar(im, ax=ax)
    
    # Gráfico 6: Relación inter/intra
    ax = axes[1, 2]
    relaciones = []
    nombres = []
    for i, op1 in enumerate(opciones):
        for j, op2 in enumerate(opciones):
            if i < j:
                intra = (intra_distancias[op1] + intra_distancias[op2]) / 2
                inter = inter_distancias[f"{op1[:15]} vs {op2[:15]}"]
                relacion = inter / intra if intra > 0 else 0
                relaciones.append(relacion)
                nombres.append(f"{op1[:8]}/{op2[:8]}")
    barras = ax.bar(range(len(relaciones)), relaciones)
    ax.set_xticks(range(len(relaciones)))
    ax.set_xticklabels(nombres, rotation=45, ha='right')
    ax.axhline(y=2, color='r', linestyle='--', label='Umbral (x2)')
    ax.set_ylabel('Relación inter / intra')
    ax.set_title('Relación distancia inter-estímulo / intra-estímulo')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('VSTCosmos v60 - Caracterización Basal de Φ', fontsize=14)
    plt.tight_layout()
    plt.savefig('v60_diagnostico_campo.png', dpi=150)
    print("  Gráfico guardado: v60_diagnostico_campo.png")
    
    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)


if __name__ == "__main__":
    main()