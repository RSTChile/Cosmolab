#!/usr/bin/env python3
"""
VSTCosmos - v62: Ganancia Espectral Adaptativa
EXPERIMENTO DE DIAGNÓSTICO - Sin aprendizaje, sin memoria, sin selección.
La ganancia espectral se ajusta según la complejidad del estímulo.
Objetivo: correlación Φ-perfil moderada (0.6-0.85) para TODOS los estímulos.
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

# Parámetros de ganancia adaptativa
GANANCIA_BASE = 0.12           # Ganancia base
GANANCIA_MAX = 0.20            # Máxima (para estímulos complejos que necesitan más empuje)
GANANCIA_MIN = 0.04            # Mínima (para estímulos simples que copian fácil)
SENSIBILIDAD_COMPLEJIDAD = 2.0 # Qué tan rápido responde la ganancia a la complejidad

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


def calcular_complejidad_espectral(perfil):
    """
    Calcula la complejidad del espectro usando entropía normalizada.
    Alto valor → espectro plano/rico (música, mezcla)
    Bajo valor → espectro concentrado (tono)
    """
    # Entropía de Shannon
    perfil_norm = perfil / (np.sum(perfil) + 1e-6)
    entropia = -np.sum(perfil_norm * np.log(perfil_norm + 1e-6))
    entropia_max = np.log(DIM_FREQ)
    complejidad = entropia / entropia_max
    return complejidad


def calcular_ganancia_adaptativa(perfil):
    """
    Ganancia adaptativa basada en complejidad espectral.
    Estímulos simples (complejidad baja) → ganancia baja (evitar copia)
    Estímulos complejos (complejidad alta) → ganancia alta (necesitan más empuje)
    """
    complejidad = calcular_complejidad_espectral(perfil)
    
    # Mapeo: complejidad baja (0) → ganancia baja, complejidad alta (1) → ganancia alta
    # Exponencial para mayor sensibilidad
    factor = complejidad ** SENSIBILIDAD_COMPLEJIDAD
    
    ganancia = GANANCIA_BASE + (GANANCIA_MAX - GANANCIA_BASE) * factor
    
    return np.clip(ganancia, GANANCIA_MIN, GANANCIA_MAX), complejidad


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


def actualizar_campo_v62(Phi, A, perfil, Psi, ganancia_gen, bloqueo_max, ganancia_espectral):
    """
    v62: Ganancia espectral adaptativa.
    """
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
    
    # Término espectral con ganancia adaptativa
    termino_espectral = ganancia_espectral * (perfil_2d - Phi)
    
    dPhi_propuesto = (difusion + generacion + decaimiento + sostenimiento + 
                      termino_espectral)
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


def simular_estimulo_v62(entrada_nombre, seed):
    """Simula una experiencia con ganancia espectral adaptativa"""
    np.random.seed(seed)
    
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
    
    evolucion_phi = []
    evolucion_a = []
    ganancias_espectrales = []
    complejidades = []
    
    for idx_ventana in range(min(N_PASOS, n_ventanas)):
        perfil = perfil_espectral_ventana(audio, sr, idx_ventana, ventana_muestras, hop_muestras)
        
        # Calcular ganancia adaptativa basada en el perfil
        ganancia_esp, complejidad = calcular_ganancia_adaptativa(perfil)
        ganancias_espectrales.append(ganancia_esp)
        complejidades.append(complejidad)
        
        A = actualizar_atencion(A, Phi, Psi, Omega, k_comp)
        Phi = actualizar_campo_v62(Phi, A, perfil, Psi, ganancia_gen, bloqueo_max, ganancia_esp)
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
        'evolucion_a': evolucion_a,
        'ganancia_media': np.mean(ganancias_espectrales),
        'complejidad_media': np.mean(complejidades)
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
    print("VSTCosmos - v62: Ganancia Espectral Adaptativa")
    print("EXPERIMENTO DE DIAGNÓSTICO")
    print("La ganancia se ajusta según la complejidad del estímulo.")
    print("Objetivo: correlación Φ-perfil moderada (0.6-0.85) para TODOS.")
    print("=" * 100)
    
    estados = {opcion: [] for opcion in opciones}
    complejidades = {}
    ganancias_medias = {}
    
    print("\n[1] Ejecutando simulaciones...")
    print("-" * 80)
    
    for opcion in opciones:
        print(f"\n  Estímulo: {opcion}")
        complejidades[opcion] = []
        ganancias_medias[opcion] = []
        for rep in range(N_SEMILLAS):
            seed = rep * 100
            estado = simular_estimulo_v62(opcion, seed)
            estados[opcion].append(estado)
            complejidades[opcion].append(estado['complejidad_media'])
            ganancias_medias[opcion].append(estado['ganancia_media'])
            print(f"    Semilla {rep:2d}: rango_Φ={estado['rango_Phi']:.4f}, "
                  f"ganancia={estado['ganancia_media']:.3f}, "
                  f"complejidad={estado['complejidad_media']:.3f}")
    
    print("\n" + "=" * 100)
    print("[2] ANÁLISIS DE DISTANCIAS")
    print("=" * 100)
    
    intra_distancias = {}
    for opcion in opciones:
        distancias = []
        for i in range(len(estados[opcion])):
            for j in range(i+1, len(estados[opcion])):
                d = distancia_estados(estados[opcion][i], estados[opcion][j])
                distancias.append(d)
        media = np.mean(distancias) if distancias else 0
        intra_distancias[opcion] = media
        print(f"  {opcion:<25}: intra={media:.6f}")
    
    print("\n[3] VERIFICACIÓN: Acoplamiento moderado (0.6-0.85)")
    print("-" * 80)
    
    correlaciones = {}
    for opcion in opciones:
        estado = estados[opcion][-1]
        Phi_flat = np.mean(estado['Phi'], axis=1)
        sr, audio = cargar_audio(opcion, duracion=2.0)
        ventana_muestras = int(sr * VENTANA_FFT_MS / 1000)
        hop_muestras = int(sr * HOP_FFT_MS / 1000)
        n_ventanas = min(50, (len(audio) - ventana_muestras) // hop_muestras + 1)
        perfiles = []
        for i in range(n_ventanas):
            perf = perfil_espectral_ventana(audio, sr, i, ventana_muestras, hop_muestras)
            perfiles.append(perf)
        perfil_prom = np.mean(perfiles, axis=0)
        correlacion = np.corrcoef(Phi_flat, perfil_prom)[0, 1]
        correlaciones[opcion] = correlacion
        
        ganancia_media = np.mean(ganancias_medias[opcion])
        complejidad_media = np.mean(complejidades[opcion])
        
        print(f"\n  {opcion}:")
        print(f"    Correlación Φ-perfil: {correlacion:.3f}")
        print(f"    Ganancia adaptativa media: {ganancia_media:.3f}")
        print(f"    Complejidad espectral media: {complejidad_media:.3f}")
        
        if 0.6 <= correlacion <= 0.85:
            print(f"    ✅ ACOPLAMIENTO ÓPTIMO (zona objetivo)")
        elif correlacion > 0.85:
            print(f"    ⚠️  SOBREACOPLAMIENTO (Φ copiando al perfil)")
        elif correlacion > 0.4:
            print(f"    📈 ACOPLAMIENTO DÉBIL (necesita más ganancia)")
        else:
            print(f"    ❌ DESACOPLAMIENTO (casi sin influencia espectral)")
    
    print("\n" + "=" * 100)
    print("[4] DIAGNÓSTICO FINAL")
    print("=" * 100)
    
    print(f"\n  Ganancia adaptativa - Fórmula: G = {GANANCIA_BASE} + ({GANANCIA_MAX} - {GANANCIA_BASE}) * complejidad^{SENSIBILIDAD_COMPLEJIDAD}")
    
    # Resumen
    print("\n  Resumen de acoplamiento:")
    for opcion, corr in correlaciones.items():
        estado = "✅ ÓPTIMO" if 0.6 <= corr <= 0.85 else ("⚠️ SOBRE" if corr > 0.85 else ("📈 DÉBIL" if corr > 0.4 else "❌ NULO"))
        print(f"    {opcion[:20]:20}: {corr:.3f} → {estado}")
    
    # Criterio de éxito
    exitosos = [corr for corr in correlaciones.values() if 0.6 <= corr <= 0.85]
    print(f"\n  Estímulos en zona óptima: {len(exitosos)}/{len(opciones)}")
    
    if len(exitosos) >= 3:
        print("\n  ✅ V62 EXITOSO: El sistema regula su acoplamiento según el estímulo.")
        print("     → Listo para integrar con identidad, memoria y selección autónoma.")
    else:
        print("\n  ⚠️  V62 PARCIAL: Ajustar parámetros de ganancia adaptativa.")
        print("     → Modificar GANANCIA_BASE, GANANCIA_MAX o SENSIBILIDAD_COMPLEJIDAD.")
    
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
    ax.set_title('Evolución del campo Φ (v62)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 2: Comparación Φ vs perfil espectral
    ax = axes[0, 1]
    colores = ['blue', 'red', 'green', 'orange', 'purple']
    for idx, opcion in enumerate(opciones):
        estado = estados[opcion][-1]
        Phi_flat = np.mean(estado['Phi'], axis=1)
        ax.plot(Phi_flat, label=f"{opcion[:12]} (Φ)", color=colores[idx], linestyle='-')
        sr, audio = cargar_audio(opcion, duracion=2.0)
        ventana_muestras = int(sr * VENTANA_FFT_MS / 1000)
        hop_muestras = int(sr * HOP_FFT_MS / 1000)
        n_ventanas = min(50, (len(audio) - ventana_muestras) // hop_muestras + 1)
        perfiles = []
        for i in range(n_ventanas):
            perf = perfil_espectral_ventana(audio, sr, i, ventana_muestras, hop_muestras)
            perfiles.append(perf)
        perfil_prom = np.mean(perfiles, axis=0)
        ax.plot(perfil_prom, label=f"{opcion[:12]} (perfil)", color=colores[idx], linestyle='--', alpha=0.5)
    ax.set_xlabel('Banda espectral')
    ax.set_ylabel('Actividad')
    ax.set_title('Φ final vs Perfil espectral (v62)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Gráfico 3: Correlaciones por estímulo
    ax = axes[0, 2]
    barras = ax.bar(range(len(opciones)), list(correlaciones.values()))
    ax.axhline(y=0.85, color='r', linestyle='--', label='Sobreacoplamiento')
    ax.axhline(y=0.6, color='orange', linestyle='--', label='Zona óptima')
    ax.axhline(y=0.4, color='g', linestyle='--', label='Desacoplamiento')
    ax.set_xticks(range(len(opciones)))
    ax.set_xticklabels([op[:12] for op in opciones], rotation=45, ha='right')
    ax.set_ylabel('Correlación Φ-perfil')
    ax.set_title('Acoplamiento por estímulo (v62)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 4: Ganancia adaptativa vs complejidad
    ax = axes[1, 0]
    for opcion in opciones:
        ganancia = np.mean(ganancias_medias[opcion])
        complejidad = np.mean(complejidades[opcion])
        ax.scatter(complejidad, ganancia, s=100, label=opcion[:15])
    ax.set_xlabel('Complejidad espectral (entropía normalizada)')
    ax.set_ylabel('Ganancia espectral adaptativa')
    ax.set_title('Relación ganancia vs complejidad')
    ax.legend()
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
    im = ax.imshow(matriz_distancias, cmap='hot', vmin=0)
    ax.set_xticks(range(n_ops))
    ax.set_yticks(range(n_ops))
    ax.set_xticklabels([op[:12] for op in opciones], rotation=45, ha='right')
    ax.set_yticklabels([op[:12] for op in opciones])
    ax.set_title('Matriz de distancias entre estímulos')
    plt.colorbar(im, ax=ax)
    
    # Gráfico 6: Relación inter/intra
    ax = axes[1, 2]
    relaciones = []
    nombres = []
    for i, op1 in enumerate(opciones):
        for j, op2 in enumerate(opciones):
            if i < j:
                intra = (intra_distancias[op1] + intra_distancias[op2]) / 2
                dists = []
                for rep in range(N_SEMILLAS):
                    dists.append(distancia_estados(estados[op1][rep], estados[op2][rep]))
                inter = np.mean(dists)
                relacion = inter / intra if intra > 0 else 0
                relaciones.append(relacion)
                nombres.append(f"{op1[:8]}/{op2[:8]}")
    ax.bar(range(len(relaciones)), relaciones)
    ax.axhline(y=2, color='r', linestyle='--', label='Discriminación fuerte')
    ax.axhline(y=1, color='orange', linestyle='--', label='Línea base')
    ax.set_xticks(range(len(relaciones)))
    ax.set_xticklabels(nombres, rotation=45, ha='right')
    ax.set_ylabel('Relación inter / intra')
    ax.set_title('Discriminación entre estímulos')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('VSTCosmos v62 - Ganancia Espectral Adaptativa', fontsize=14)
    plt.tight_layout()
    plt.savefig('v62_ganancia_adaptativa.png', dpi=150)
    print("  Gráfico guardado: v62_ganancia_adaptativa.png")
    
    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)


if __name__ == "__main__":
    main()