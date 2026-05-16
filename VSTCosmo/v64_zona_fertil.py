#!/usr/bin/env python3
"""
VSTCosmos - v64: Zona Fértil de Acoplamiento
EXPERIMENTO DE DIAGNÓSTICO - Sin aprendizaje, sin memoria, sin selección.
La ganancia espectral sigue una campana de Gauss centrada en la magnitud óptima.
Ruido (magnitud alta) y Tono (magnitud baja) tienen ganancia baja.
Voz (magnitud óptima) tiene ganancia alta.
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

# Parámetros de la zona fértil
GANANCIA_BASE = 0.15
MAGNITUD_OPTIMA = 0.065
ANCHURA_ZONA = 0.04

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


def ganancia_por_magnitud(magnitud):
    """
    Campana de Gauss centrada en MAGNITUD_OPTIMA.
    Ganancia alta para magnitud ~0.065 (voz).
    Ganancia baja para magnitud → 0 (tono) y magnitud → alta (ruido).
    """
    return np.exp(-((magnitud - MAGNITUD_OPTIMA)**2) / (2 * ANCHURA_ZONA**2))


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


def actualizar_campo_v64(Phi, A, perfil, Psi, ganancia_gen, bloqueo_max, ganancia_espectral):
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


def simular_estimulo_v64(entrada_nombre, seed):
    np.random.seed(seed)
    
    Phi = inicializar_campo(seed)
    A = inicializar_atencion(seed)
    Psi = inicializar_memoria()
    Omega = inicializar_memoria()
    
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
    ganancias = []
    magnitudes = []
    
    perfil_anterior = None
    
    for idx_ventana in range(min(N_PASOS, n_ventanas)):
        perfil = perfil_espectral_ventana(audio, sr, idx_ventana, ventana_muestras, hop_muestras)
        
        # Calcular magnitud de cambio entre ventanas consecutivas
        if perfil_anterior is not None:
            magnitud = np.mean(np.abs(perfil - perfil_anterior))
        else:
            magnitud = 0.0
        
        # Ganancia por zona fértil (campana de Gauss)
        factor_ganancia = ganancia_por_magnitud(magnitud)
        ganancia_espectral = GANANCIA_BASE * factor_ganancia
        
        ganancias.append(ganancia_espectral)
        magnitudes.append(magnitud)
        
        A = actualizar_atencion(A, Phi, Psi, Omega, k_comp)
        Phi = actualizar_campo_v64(Phi, A, perfil, Psi, ganancia_gen, bloqueo_max, ganancia_espectral)
        Psi = actualizar_memoria_estabilidad(Psi, Phi, A)
        Omega = actualizar_memoria_coherencia(Omega, Phi, Phi_prev, Phi_prev2, A)
        
        if idx_ventana % 100 == 0:
            evolucion_phi.append(np.mean(Phi))
            evolucion_a.append(np.mean(A))
        
        Phi_prev2 = Phi_prev.copy()
        Phi_prev = Phi.copy()
        perfil_anterior = perfil.copy()
    
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
    
    return {
        'Phi': Phi,
        'A': A,
        'rango_Phi': np.max(Phi) - np.min(Phi),
        'media_Phi': np.mean(Phi),
        'evolucion_phi': evolucion_phi,
        'evolucion_a': evolucion_a,
        'ganancia_media': np.mean(ganancias),
        'magnitud_media': np.mean(magnitudes),
        'factor_ganancia_medio': np.mean([ganancia_por_magnitud(m) for m in magnitudes]),
        'perfil_promedio': perfil_promedio
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
    print("VSTCosmos - v64: Zona Fértil de Acoplamiento")
    print("EXPERIMENTO DE DIAGNÓSTICO")
    print("La ganancia espectral sigue una campana de Gauss centrada en la magnitud óptima.")
    print(f"G = {GANANCIA_BASE} * exp(-((magnitud - {MAGNITUD_OPTIMA})²) / (2 * {ANCHURA_ZONA}²))")
    print("=" * 100)
    
    estados = {opcion: [] for opcion in opciones}
    
    print("\n[1] Ejecutando simulaciones...")
    print("-" * 80)
    
    for opcion in opciones:
        print(f"\n  Estímulo: {opcion}")
        for rep in range(N_SEMILLAS):
            seed = rep * 100
            estado = simular_estimulo_v64(opcion, seed)
            estados[opcion].append(estado)
            print(f"    Semilla {rep:2d}: rango_Φ={estado['rango_Phi']:.4f}, "
                  f"ganancia={estado['ganancia_media']:.3f}, "
                  f"magnitud={estado['magnitud_media']:.4f}, "
                  f"factor={estado['factor_ganancia_medio']:.3f}")
    
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
        intra = np.mean(distancias) if distancias else 0
        intra_distancias[opcion] = intra
        print(f"  {opcion:<25}: intra={intra:.8f}")
    
    print("\n[3] ACOPLAMIENTO")
    print("=" * 100)
    
    correlaciones = {}
    for opcion in opciones:
        estado = estados[opcion][-1]
        Phi_flat = np.mean(estado['Phi'], axis=1)
        perfil_prom = estado['perfil_promedio']
        correlacion = np.corrcoef(Phi_flat, perfil_prom)[0, 1]
        correlaciones[opcion] = correlacion
        
        print(f"\n  {opcion}:")
        print(f"    Magnitud de cambio: {estado['magnitud_media']:.4f}")
        print(f"    Factor de ganancia: {estado['factor_ganancia_medio']:.3f}")
        print(f"    Ganancia resultante: {estado['ganancia_media']:.3f}")
        print(f"    Correlación Φ-perfil: {correlacion:.3f}")
        
        if 0.6 <= correlacion <= 0.85:
            print(f"    ✅ ACOPLAMIENTO ÓPTIMO")
        elif correlacion > 0.85:
            print(f"    ⚠️  SOBREACOPLAMIENTO")
        else:
            print(f"    ❌ DESACOPLAMIENTO")
    
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
    
    discriminacion_ok = relacion_ruido > 100 and relacion_voz > 100
    
    # Acoplamiento óptimo
    acoplamientos_ok = [c for c in correlaciones.values() if 0.6 <= c <= 0.85]
    print(f"\n  Acoplamiento óptimo: {len(acoplamientos_ok)}/5")
    
    print(f"\n  {opciones[0]}: {correlaciones[opciones[0]]:.3f} → " + 
          ("✅" if 0.6 <= correlaciones[opciones[0]] <= 0.85 else "❌"))
    print(f"  {opciones[1]}: {correlaciones[opciones[1]]:.3f} → " + 
          ("✅" if 0.6 <= correlaciones[opciones[1]] <= 0.85 else "❌"))
    print(f"  {opciones[2]}: {correlaciones[opciones[2]]:.3f} → " + 
          ("✅" if 0.6 <= correlaciones[opciones[2]] <= 0.85 else "❌"))
    print(f"  {opciones[3]}: {correlaciones[opciones[3]]:.3f} → " + 
          ("✅" if 0.6 <= correlaciones[opciones[3]] <= 0.85 else "❌"))
    print(f"  {opciones[4]}: {correlaciones[opciones[4]]:.3f} → " + 
          ("✅" if 0.6 <= correlaciones[opciones[4]] <= 0.85 else "❌"))
    
    if discriminacion_ok and len(acoplamientos_ok) >= 4:
        print("\n  ✅ V64 EXITOSO")
        print("     → El sistema discrimina Y se acopla en la zona óptima.")
        print("     → La campana de Gauss centrada en la magnitud es la métrica correcta.")
    elif discriminacion_ok and len(acoplamientos_ok) >= 2:
        print("\n  ✅ V64 PARCIAL")
        print("     → Discriminación lograda, acoplamiento cerca del objetivo.")
        print("     → Ajustar GANANCIA_BASE, MAGNITUD_OPTIMA o ANCHURA_ZONA.")
    else:
        print("\n  ❌ V64 FALLIDO")
        print("     → Revisar la función de ganancia.")
    
    # Visualización
    print("\n[5] Generando visualización...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Gráfico 1: Correlaciones
    ax = axes[0, 0]
    corrs = [correlaciones[op] for op in opciones]
    barras = ax.bar(range(len(opciones)), corrs)
    ax.axhline(y=0.85, color='r', linestyle='--', label='Sobreacoplamiento')
    ax.axhline(y=0.6, color='orange', linestyle='--', label='Zona óptima')
    ax.set_xticks(range(len(opciones)))
    ax.set_xticklabels([op[:12] for op in opciones], rotation=45, ha='right')
    ax.set_ylabel('Correlación Φ-perfil')
    ax.set_title('Acoplamiento por estímulo (v64)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 2: Ganancia vs Magnitud
    ax = axes[0, 1]
    for opcion in opciones:
        g = estados[opcion][0]['ganancia_media']
        m = estados[opcion][0]['magnitud_media']
        ax.scatter(m, g, s=150, label=opcion[:15])
    # Curva teórica
    m_teorica = np.linspace(0, 0.25, 100)
    g_teorica = GANANCIA_BASE * ganancia_por_magnitud(m_teorica)
    ax.plot(m_teorica, g_teorica, 'k--', alpha=0.5, label='Curva teórica')
    ax.set_xlabel('Magnitud de cambio espectral')
    ax.set_ylabel('Ganancia adaptativa')
    ax.set_title('Ganancia vs Magnitud')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 3: Evolución de Φ
    ax = axes[0, 2]
    for opcion in opciones:
        evol = estados[opcion][-1]['evolucion_phi']
        ax.plot(evol, label=opcion[:15])
    ax.set_xlabel('Ventana (x100)')
    ax.set_ylabel('Φ medio')
    ax.set_title('Evolución del campo Φ')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 4: Matriz de distancias
    ax = axes[1, 0]
    n_ops = len(opciones)
    matriz = np.zeros((n_ops, n_ops))
    for i, op1 in enumerate(opciones):
        for j, op2 in enumerate(opciones):
            dists = [distancia_estados(estados[op1][rep], estados[op2][rep]) for rep in range(N_SEMILLAS)]
            matriz[i, j] = np.mean(dists)
    im = ax.imshow(matriz, cmap='hot', vmin=0)
    ax.set_xticks(range(n_ops))
    ax.set_yticks(range(n_ops))
    ax.set_xticklabels([op[:12] for op in opciones], rotation=45, ha='right')
    ax.set_yticklabels([op[:12] for op in opciones])
    ax.set_title('Matriz de distancias')
    plt.colorbar(im, ax=ax)
    
    # Gráfico 5: Relación inter/intra
    ax = axes[1, 1]
    relaciones = [relacion_ruido, relacion_voz]
    nombres = ["Ruido/Voz", "Voz/Ruido"]
    ax.bar(nombres, relaciones)
    ax.axhline(y=100, color='r', linestyle='--', label='Discriminación fuerte')
    ax.set_ylabel('Relación inter/intra')
    ax.set_title('Discriminación')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 6: Curva de ganancia teórica
    ax = axes[1, 2]
    m_vals = np.linspace(0, 0.25, 100)
    g_vals = GANANCIA_BASE * ganancia_por_magnitud(m_vals)
    ax.plot(m_vals, g_vals, 'b-', linewidth=2)
    ax.axvline(x=MAGNITUD_OPTIMA, color='g', linestyle='--', label='Optimo')
    ax.set_xlabel('Magnitud de cambio espectral')
    ax.set_ylabel('Ganancia espectral')
    ax.set_title('Curva de ganancia teórica')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.suptitle('VSTCosmos v64 - Zona Fértil de Acoplamiento', fontsize=14)
    plt.tight_layout()
    plt.savefig('v64_zona_fertil.png', dpi=150)
    print("  Gráfico guardado: v64_zona_fertil.png")
    
    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)


if __name__ == "__main__":
    main()