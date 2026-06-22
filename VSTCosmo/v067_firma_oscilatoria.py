#!/usr/bin/env python3
"""
VSTCosmos - v67: Discriminación por Firma Oscilatoria
EXPERIMENTO DE DIAGNÓSTICO
La discriminación no se basa en el estado final de Φ sino en el perfil de
energía oscilatoria por banda. Cada estímulo produce una firma característica.
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

# Parámetros oscilatorios (de v66, conservados)
OMEGA_MIN = 0.05
OMEGA_MAX = 0.50
AMORT_MIN = 0.01
AMORT_MAX = 0.08
GANANCIA_ESPECTRAL = 0.15
PHI_EQUILIBRIO = 0.5

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
    bandas = np.arange(DIM_FREQ)
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


def actualizar_campo_v67(Phi, Phi_vel, A, perfil, Psi, 
                          ganancia_gen, bloqueo_max, K,
                          omega_natural, amort_natural):
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
    
    termino_espectral = GANANCIA_ESPECTRAL * (perfil_2d - Phi)
    
    termino_oscilatorio = (-omega_natural**2 * (Phi - PHI_EQUILIBRIO) 
                           - amort_natural * Phi_vel)
    
    dPhi_vel = termino_oscilatorio + termino_espectral
    Phi_vel_nueva = Phi_vel + DT * dPhi_vel
    
    dPhi_propuesto = (difusion + generacion + decaimiento + 
                      sostenimiento + Phi_vel_nueva)
    dPhi_real = dPhi_propuesto * (1 - bloqueo_max * Psi)
    
    Phi_nueva = Phi + DT * dPhi_real
    
    return (np.clip(Phi_nueva, LIMITE_MIN, LIMITE_MAX), 
            np.clip(Phi_vel_nueva, -5.0, 5.0))


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


def simular_estimulo_v67(entrada_nombre, seed):
    np.random.seed(seed)
    
    Phi = inicializar_campo(seed)
    Phi_vel = np.zeros((DIM_FREQ, DIM_TIME))
    A = inicializar_atencion(seed)
    Psi = inicializar_memoria()
    Omega = inicializar_memoria()
    
    omega_natural, amort_natural = calcular_frecuencias_naturales()
    
    sr, audio = cargar_audio(entrada_nombre)
    
    ventana_muestras = int(sr * VENTANA_FFT_MS / 1000)
    hop_muestras = int(sr * HOP_FFT_MS / 1000)
    n_ventanas = (len(audio) - ventana_muestras) // hop_muestras + 1
    
    ganancia_gen = GANANCIA_GENERACION_BASE
    bloqueo_max = BLOQUEO_MAXIMO_BASE
    k_comp = K_COMP_BASE
    
    # Acumulador de energía oscilatoria por banda
    energia_por_banda = np.zeros(DIM_FREQ)
    n_pasos_acumulados = 0
    
    evolucion_energia = []
    
    for idx_ventana in range(min(N_PASOS, n_ventanas)):
        perfil = perfil_espectral_ventana(audio, sr, idx_ventana, ventana_muestras, hop_muestras)
        
        A = actualizar_atencion(A, Phi, Psi, Omega, k_comp)
        Phi, Phi_vel = actualizar_campo_v67(
            Phi, Phi_vel, A, perfil, Psi,
            ganancia_gen, bloqueo_max, K=1.0,
            omega_natural=omega_natural, amort_natural=amort_natural
        )
        Psi = actualizar_memoria_estabilidad(Psi, Phi, A)
        Omega = actualizar_memoria_coherencia(Omega, Phi, Phi, Phi, A)
        
        # Acumular energía oscilatoria por banda
        energia_por_banda += np.mean(Phi_vel**2, axis=1)
        n_pasos_acumulados += 1
        
        if idx_ventana % 100 == 0:
            evolucion_energia.append(np.mean(Phi_vel**2))
    
    # Normalizar
    energia_por_banda /= max(n_pasos_acumulados, 1)
    energia_total = np.mean(energia_por_banda)
    
    # Normalizar firma para comparación
    max_energia = np.max(energia_por_banda)
    if max_energia > 0:
        firma_normalizada = energia_por_banda / max_energia
    else:
        firma_normalizada = energia_por_banda
    
    # Para comparación con versiones anteriores, también guardamos el estado de Φ
    # y calculamos su perfil promedio
    Phi_flat_promedio = np.mean(Phi, axis=1)
    
    return {
        'Phi': Phi,
        'Phi_vel': Phi_vel,
        'rango_Phi': np.max(Phi) - np.min(Phi),
        'media_Phi': np.mean(Phi),
        'energia_total': energia_total,
        'energia_por_banda': energia_por_banda,
        'firma_normalizada': firma_normalizada,
        'Phi_flat_promedio': Phi_flat_promedio,
        'evolucion_energia': evolucion_energia
    }


def distancia_firmas(estado1, estado2):
    """Distancia entre firmas oscilatorias normalizadas"""
    return np.mean((estado1['firma_normalizada'] - estado2['firma_normalizada'])**2)


def distancia_estados_phi(estado1, estado2):
    """Distancia entre estados de Φ (para comparación)"""
    return np.mean((estado1['Phi'] - estado2['Phi'])**2)


def main():
    opciones = [
        "Ruido blanco",
        "Tono puro",
        "Voz_Estudio.wav",
        "Brandemburgo.wav",
        "Voz+Viento_1.wav"
    ]
    
    print("=" * 100)
    print("VSTCosmos - v67: Discriminación por Firma Oscilatoria")
    print("EXPERIMENTO DE DIAGNÓSTICO")
    print("La discriminación se basa en el perfil de energía oscilatoria por banda,")
    print("no en el estado final de Φ.")
    print(f"AMORT_MIN={AMORT_MIN}, AMORT_MAX={AMORT_MAX}, G_ESPECTRAL={GANANCIA_ESPECTRAL}")
    print("=" * 100)
    
    estados = {opcion: [] for opcion in opciones}
    
    print("\n[1] Ejecutando simulaciones...")
    print("-" * 80)
    
    for opcion in opciones:
        print(f"\n  Estímulo: {opcion}")
        for rep in range(N_SEMILLAS):
            seed = rep * 100
            estado = simular_estimulo_v67(opcion, seed)
            estados[opcion].append(estado)
            print(f"    Semilla {rep:2d}: rango_Φ={estado['rango_Phi']:.4f}, "
                  f"energía_total={estado['energia_total']:.6f}")
    
    print("\n" + "=" * 100)
    print("[2] FIRMAS OSCILATORIAS (discriminación principal)")
    print("=" * 100)
    
    # Distancias intra-estímulo (firmas)
    intra_firmas = {}
    for opcion in opciones:
        distancias = []
        for i in range(len(estados[opcion])):
            for j in range(i+1, len(estados[opcion])):
                d = distancia_firmas(estados[opcion][i], estados[opcion][j])
                distancias.append(d)
        intra = np.mean(distancias) if distancias else 0
        intra_firmas[opcion] = intra
        print(f"  {opcion:<25}: intra_firma={intra:.8f}")
    
    # Distancias inter-estímulo (firmas)
    print("\n  Distancias INTER-estímulo (firmas):")
    inter_firmas = {}
    for i, op1 in enumerate(opciones):
        for j, op2 in enumerate(opciones):
            if i < j:
                distancias = []
                for rep in range(N_SEMILLAS):
                    d = distancia_firmas(estados[op1][rep], estados[op2][rep])
                    distancias.append(d)
                inter = np.mean(distancias)
                inter_firmas[f"{op1[:15]} vs {op2[:15]}"] = inter
                print(f"    {op1[:15]} vs {op2[:15]}: inter={inter:.8f}")
    
    # Relaciones inter/intra para firmas
    print("\n  Relaciones inter/intra (firmas):")
    ruido_firma_intra = intra_firmas["Ruido blanco"]
    voz_firma_intra = intra_firmas["Voz_Estudio.wav"]
    ruido_voz_firma_inter = inter_firmas["Ruido blanco vs Voz_Estudio.wav"]
    
    rel_ruido_firma = ruido_voz_firma_inter / ruido_firma_intra if ruido_firma_intra > 0 else 0
    rel_voz_firma = ruido_voz_firma_inter / voz_firma_intra if voz_firma_intra > 0 else 0
    
    print(f"    Ruido vs Voz (firma): inter={ruido_voz_firma_inter:.8f}")
    print(f"    Relación (ruido): {rel_ruido_firma:.2f}")
    print(f"    Relación (voz): {rel_voz_firma:.2f}")
    
    print("\n" + "=" * 100)
    print("[3] ESTADOS DE Φ (para comparación)")
    print("=" * 100)
    
    # Distancias intra-estímulo (Φ)
    intra_phi = {}
    for opcion in opciones:
        distancias = []
        for i in range(len(estados[opcion])):
            for j in range(i+1, len(estados[opcion])):
                d = distancia_estados_phi(estados[opcion][i], estados[opcion][j])
                distancias.append(d)
        intra = np.mean(distancias) if distancias else 0
        intra_phi[opcion] = intra
        print(f"  {opcion:<25}: intra_phi={intra:.8f}")
    
    # Distancias inter-estímulo (Φ)
    print("\n  Distancias INTER-estímulo (Φ):")
    inter_phi = {}
    for i, op1 in enumerate(opciones):
        for j, op2 in enumerate(opciones):
            if i < j:
                distancias = []
                for rep in range(N_SEMILLAS):
                    d = distancia_estados_phi(estados[op1][rep], estados[op2][rep])
                    distancias.append(d)
                inter = np.mean(distancias)
                inter_phi[f"{op1[:15]} vs {op2[:15]}"] = inter
                print(f"    {op1[:15]} vs {op2[:15]}: inter={inter:.8f}")
    
    # Relaciones inter/intra para Φ
    print("\n  Relaciones inter/intra (Φ):")
    ruido_phi_intra = intra_phi["Ruido blanco"]
    voz_phi_intra = intra_phi["Voz_Estudio.wav"]
    ruido_voz_phi_inter = inter_phi["Ruido blanco vs Voz_Estudio.wav"]
    
    rel_ruido_phi = ruido_voz_phi_inter / ruido_phi_intra if ruido_phi_intra > 0 else 0
    rel_voz_phi = ruido_voz_phi_inter / voz_phi_intra if voz_phi_intra > 0 else 0
    
    print(f"    Ruido vs Voz (Φ): inter={ruido_voz_phi_inter:.8f}")
    print(f"    Relación (ruido): {rel_ruido_phi:.2f}")
    print(f"    Relación (voz): {rel_voz_phi:.2f}")
    
    print("\n" + "=" * 100)
    print("[4] ENERGÍAS TOTALES")
    print("=" * 100)
    
    energias_totales = {}
    for opcion in opciones:
        energias = [est['energia_total'] for est in estados[opcion]]
        energias_totales[opcion] = (np.mean(energias), np.std(energias))
    
    for opcion, (media, std) in energias_totales.items():
        print(f"  {opcion:<25}: energía_total = {media:.6f} ± {std:.6f}")
    
    print("\n" + "=" * 100)
    print("[5] DIAGNÓSTICO FINAL")
    print("=" * 100)
    
    print(f"\n  FIRMAS OSCILATORIAS:")
    print(f"    Relación inter/intra (ruido): {rel_ruido_firma:.2f}")
    print(f"    Relación inter/intra (voz): {rel_voz_firma:.2f}")
    
    print(f"\n  ESTADOS DE Φ (referencia):")
    print(f"    Relación inter/intra (ruido): {rel_ruido_phi:.2f}")
    print(f"    Relación inter/intra (voz): {rel_voz_phi:.2f}")
    
    if rel_ruido_firma > 10 and rel_voz_firma > 10:
        print("\n  ✅ V67 EXITOSO")
        print("     → Las firmas oscilatorias discriminan robustamente.")
        print("     → El mecanismo de resonancia por banda funciona como se esperaba.")
    elif rel_ruido_firma > 3 and rel_voz_firma > 3:
        print("\n  ✅ V67 PARCIAL")
        print("     → Discriminación por firma oscilatoria lograda, pero aún débil.")
        print("     → Revisar acumulación de energía o parámetros de amortiguamiento.")
    else:
        print("\n  ❌ V67 FALLIDO")
        print("     → Las firmas oscilatorias no discriminan.")
        print("     → Revisar implementación del oscilador o acumulación de energía.")
    
    # ============================================================
    # VISUALIZACIÓN
    # ============================================================
    print("\n[6] Generando visualización...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Gráfico 1: Firmas oscilatorias (primeras semillas)
    ax = axes[0, 0]
    for opcion in opciones:
        firma = estados[opcion][0]['firma_normalizada']
        ax.plot(firma, label=opcion[:15], linewidth=1.5)
    ax.set_xlabel('Banda espectral')
    ax.set_ylabel('Energía oscilatoria (normalizada)')
    ax.set_title('Firmas oscilatorias por estímulo')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 2: Energías totales
    ax = axes[0, 1]
    nombres = [op[:15] for op in opciones]
    medias = [energias_totales[op][0] for op in opciones]
    desvios = [energias_totales[op][1] for op in opciones]
    ax.bar(nombres, medias, yerr=desvios, capsize=5)
    ax.set_ylabel('Energía oscilatoria total')
    ax.set_title('Energía total por estímulo')
    ax.set_xticklabels(nombres, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # Gráfico 3: Evolución de la energía
    ax = axes[0, 2]
    for opcion in opciones:
        evol = estados[opcion][0]['evolucion_energia']
        ax.plot(evol, label=opcion[:15])
    ax.set_xlabel('Ventana (x100)')
    ax.set_ylabel('Energía oscilatoria media')
    ax.set_title('Evolución de la resonancia')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 4: Comparación firmas vs Φ
    ax = axes[1, 0]
    relaciones = [rel_ruido_firma, rel_voz_firma, rel_ruido_phi, rel_voz_phi]
    nombres_rel = ['Firma (ruido)', 'Firma (voz)', 'Φ (ruido)', 'Φ (voz)']
    ax.bar(nombres_rel, relaciones)
    ax.axhline(y=10, color='g', linestyle='--', label='Umbral éxito')
    ax.axhline(y=3, color='orange', linestyle='--', label='Umbral parcial')
    ax.set_ylabel('Relación inter/intra')
    ax.set_title('Comparación de métricas de discriminación')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 5: Matriz de distancias de firmas
    ax = axes[1, 1]
    n_ops = len(opciones)
    matriz_firmas = np.zeros((n_ops, n_ops))
    for i, op1 in enumerate(opciones):
        for j, op2 in enumerate(opciones):
            if i == j:
                matriz_firmas[i, j] = intra_firmas[op1]
            else:
                key = f"{op1[:15]} vs {op2[:15]}"
                if key in inter_firmas:
                    matriz_firmas[i, j] = inter_firmas[key]
                else:
                    key_rev = f"{op2[:15]} vs {op1[:15]}"
                    matriz_firmas[i, j] = inter_firmas[key_rev] if key_rev in inter_firmas else 0
    im = ax.imshow(matriz_firmas, cmap='hot')
    ax.set_xticks(range(n_ops))
    ax.set_yticks(range(n_ops))
    ax.set_xticklabels([op[:12] for op in opciones], rotation=45, ha='right')
    ax.set_yticklabels([op[:12] for op in opciones])
    ax.set_title('Matriz de distancias (firmas)')
    plt.colorbar(im, ax=ax)
    
    # Gráfico 6: Comparación de perfiles Φ promedio
    ax = axes[1, 2]
    for opcion in opciones:
        phi_flat = estados[opcion][0]['Phi_flat_promedio']
        ax.plot(phi_flat, label=opcion[:15], linewidth=1.5)
    ax.set_xlabel('Banda')
    ax.set_ylabel('Φ promedio')
    ax.set_title('Perfil de Φ final (para referencia)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('VSTCosmos v67 - Discriminación por Firma Oscilatoria', fontsize=14)
    plt.tight_layout()
    plt.savefig('v67_firma_oscilatoria.png', dpi=150)
    print("  Gráfico guardado: v67_firma_oscilatoria.png")
    
    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)


if __name__ == "__main__":
    main()