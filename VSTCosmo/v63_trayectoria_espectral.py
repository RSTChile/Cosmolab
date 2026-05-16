#!/usr/bin/env python3
"""
VSTCosmos - v63: Trayectoria Temporal del Espectro
EXPERIMENTO DE DIAGNÓSTICO - Sin aprendizaje, sin memoria, sin selección.
La ganancia espectral se modula por la COHERENCIA DE TRAYECTORIA del espectro,
no por entropía estática. El ruido salta incoherentemente; la voz evoluciona suavemente.
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

# Parámetros de ganancia por trayectoria
GANANCIA_BASE = 0.08
GANANCIA_MIN = 0.04
GANANCIA_MAX = 0.20

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


def calcular_coherencia_trayectoria(perfil_actual, perfil_anterior, perfil_ante_anterior):
    """
    Mide la coherencia de la trayectoria espectral.
    
    Ruido blanco → variaciones grandes e incoherentes → coherencia baja
    Voz → variaciones pequeñas y estructuradas → coherencia alta
    Música → variaciones medianas y periódicas → coherencia media-alta
    """
    if perfil_anterior is None or perfil_ante_anterior is None:
        return 0.5, np.zeros(DIM_FREQ), 0.0
    
    # Cambio entre ventanas consecutivas
    delta_actual = perfil_actual - perfil_anterior
    delta_anterior = perfil_anterior - perfil_ante_anterior
    
    # Coherencia: los cambios van en la misma dirección?
    signo_actual = np.sign(delta_actual)
    signo_anterior = np.sign(delta_anterior)
    coherencia_local = (signo_actual * signo_anterior + 1) / 2  # [0, 1] por banda
    
    coherencia_global = np.mean(coherencia_local)
    
    # Magnitud del cambio
    magnitud = np.mean(np.abs(delta_actual))
    
    return coherencia_global, coherencia_local, magnitud


def calcular_ganancia_por_trayectoria(coherencia_global, magnitud):
    """
    Ganancia basada en coherencia de trayectoria.
    
    Busca que Φ se acople a la trayectoria, no al perfil estático.
    """
    # El ruido tiene alta magnitud y baja coherencia → ganancia baja
    # La voz tiene baja magnitud y alta coherencia → ganancia alta
    # La música tiene magnitud variable y coherencia media → ganancia media
    
    factor_coherencia = coherencia_global ** 2
    factor_estabilidad = 1.0 / (1.0 + magnitud * 10)
    
    ganancia = GANANCIA_BASE * (1 + factor_coherencia * factor_estabilidad)
    return np.clip(ganancia, GANANCIA_MIN, GANANCIA_MAX)


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


def actualizar_campo_v63(Phi, A, perfil, Psi, ganancia_gen, bloqueo_max, ganancia_espectral):
    """
    v63: Ganancia espectral modulada por coherencia de trayectoria.
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
    
    # Término espectral con ganancia adaptativa por trayectoria
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


def simular_estimulo_v63(entrada_nombre, seed):
    """Simula una experiencia con ganancia modulada por coherencia de trayectoria"""
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
    coherencias = []
    magnitudes = []
    
    # Historial de perfiles para calcular trayectoria
    perfil_ante_anterior = None
    perfil_anterior = None
    
    for idx_ventana in range(min(N_PASOS, n_ventanas)):
        perfil_actual = perfil_espectral_ventana(audio, sr, idx_ventana, ventana_muestras, hop_muestras)
        
        # Calcular coherencia de trayectoria
        coherencia, coherencia_local, magnitud = calcular_coherencia_trayectoria(
            perfil_actual, perfil_anterior, perfil_ante_anterior
        )
        
        # Calcular ganancia adaptativa por trayectoria
        ganancia_esp = calcular_ganancia_por_trayectoria(coherencia, magnitud)
        
        ganancias_espectrales.append(ganancia_esp)
        coherencias.append(coherencia)
        magnitudes.append(magnitud)
        
        A = actualizar_atencion(A, Phi, Psi, Omega, k_comp)
        Phi = actualizar_campo_v63(Phi, A, perfil_actual, Psi, ganancia_gen, bloqueo_max, ganancia_esp)
        Psi = actualizar_memoria_estabilidad(Psi, Phi, A)
        Omega = actualizar_memoria_coherencia(Omega, Phi, Phi_prev, Phi_prev2, A)
        
        if idx_ventana % 100 == 0:
            evolucion_phi.append(np.mean(Phi))
            evolucion_a.append(np.mean(A))
        
        Phi_prev2 = Phi_prev.copy()
        Phi_prev = Phi.copy()
        
        # Actualizar historial de perfiles
        perfil_ante_anterior = perfil_anterior
        perfil_anterior = perfil_actual.copy()
    
    # Perfil promedio del estímulo (para correlación final)
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
        'Psi': Psi,
        'Omega': Omega,
        'rango_Phi': np.max(Phi) - np.min(Phi),
        'media_Phi': np.mean(Phi),
        'energia_Phi': np.mean(Phi ** 2),
        'evolucion_phi': evolucion_phi,
        'evolucion_a': evolucion_a,
        'ganancia_media': np.mean(ganancias_espectrales),
        'coherencia_media': np.mean(coherencias),
        'magnitud_media': np.mean(magnitudes),
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
    print("VSTCosmos - v63: Trayectoria Temporal del Espectro")
    print("EXPERIMENTO DE DIAGNÓSTICO")
    print("La ganancia espectral se modula por la COHERENCIA DE TRAYECTORIA del espectro.")
    print("El ruido salta incoherentemente; la voz evoluciona suavemente.")
    print("=" * 100)
    
    estados = {opcion: [] for opcion in opciones}
    
    print("\n[1] Ejecutando simulaciones...")
    print("-" * 80)
    
    for opcion in opciones:
        print(f"\n  Estímulo: {opcion}")
        for rep in range(N_SEMILLAS):
            seed = rep * 100
            estado = simular_estimulo_v63(opcion, seed)
            estados[opcion].append(estado)
            print(f"    Semilla {rep:2d}: rango_Φ={estado['rango_Phi']:.4f}, "
                  f"ganancia={estado['ganancia_media']:.3f}, "
                  f"coherencia={estado['coherencia_media']:.3f}, "
                  f"magnitud={estado['magnitud_media']:.4f}")
    
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
    
    # Calcular inter-distancias
    inter_pares = {}
    for i, op1 in enumerate(opciones):
        for j, op2 in enumerate(opciones):
            if i < j:
                dists = []
                for rep in range(N_SEMILLAS):
                    d = distancia_estados(estados[op1][rep], estados[op2][rep])
                    dists.append(d)
                inter = np.mean(dists)
                inter_pares[f"{op1[:15]} vs {op2[:15]}"] = inter
                print(f"  {op1[:15]} vs {op2[:15]}: inter={inter:.8f}")
    
    print("\n" + "=" * 100)
    print("[3] VERIFICACIÓN: Coherencia de trayectoria y acoplamiento")
    print("=" * 100)
    
    correlaciones = {}
    coherencias_medias = {}
    ganancias_medias = {}
    magnitudes_medias = {}
    
    for opcion in opciones:
        estado = estados[opcion][-1]
        Phi_flat = np.mean(estado['Phi'], axis=1)
        perfil_prom = estado['perfil_promedio']
        correlacion = np.corrcoef(Phi_flat, perfil_prom)[0, 1]
        correlaciones[opcion] = correlacion
        coherencias_medias[opcion] = estado['coherencia_media']
        ganancias_medias[opcion] = estado['ganancia_media']
        magnitudes_medias[opcion] = estado['magnitud_media']
        
        print(f"\n  {opcion}:")
        print(f"    Coherencia de trayectoria: {estado['coherencia_media']:.3f}")
        print(f"    Magnitud de cambio: {estado['magnitud_media']:.4f}")
        print(f"    Ganancia adaptativa media: {estado['ganancia_media']:.3f}")
        print(f"    Correlación Φ-perfil: {correlacion:.3f}")
        
        if 0.6 <= correlacion <= 0.85:
            print(f"    ✅ ACOPLAMIENTO ÓPTIMO")
        elif correlacion > 0.85:
            print(f"    ⚠️  SOBREACOPLAMIENTO (Φ copiando al perfil)")
        else:
            print(f"    ❌ DESACOPLAMIENTO (poca influencia espectral)")
    
    print("\n" + "=" * 100)
    print("[4] PREDICCIONES FALSABLES")
    print("=" * 100)
    
    print("\n  Predicción vs Resultado:")
    for opcion in opciones:
        coh = coherencias_medias[opcion]
        mag = magnitudes_medias[opcion]
        corr = correlaciones[opcion]
        
        if opcion == "Ruido blanco":
            esperado = "coherencia baja (0.45-0.55), magnitud alta, ganancia baja"
            ok = (0.45 <= coh <= 0.55) and (corr < 0.6)
        elif opcion == "Tono puro":
            esperado = "coherencia alta (0.80+), magnitud muy baja, ganancia alta"
            ok = (coh >= 0.75) and (corr > 0.6)
        elif opcion == "Voz_Estudio.wav":
            esperado = "coherencia media-alta (0.65-0.75), magnitud baja-media"
            ok = (0.60 <= coh <= 0.80) and (0.6 <= corr <= 0.85)
        elif opcion == "Brandemburgo.wav":
            esperado = "coherencia media (0.60-0.70), magnitud media"
            ok = (0.55 <= coh <= 0.75)
        else:  # Voz+Viento
            esperado = "coherencia media (0.55-0.65), magnitud media-alta"
            ok = (0.50 <= coh <= 0.70)
        
        print(f"    {opcion[:20]}: coh={coh:.3f} | mag={mag:.4f} | corr={corr:.3f}")
        print(f"      Esperado: {esperado}")
        print(f"      {'✅ CUMPLE' if ok else '❌ NO CUMPLE'}")
    
    print("\n" + "=" * 100)
    print("[5] DIAGNÓSTICO FINAL")
    print("=" * 100)
    
    # Criterio de éxito: relación inter/intra
    ruido_intra = intra_distancias["Ruido blanco"]
    voz_intra = intra_distancias["Voz_Estudio.wav"]
    ruido_voz_inter = inter_pares["Ruido blanco vs Voz_Estudio.wav"]
    
    relacion_ruido = ruido_voz_inter / ruido_intra if ruido_intra > 0 else 0
    relacion_voz = ruido_voz_inter / voz_intra if voz_intra > 0 else 0
    
    print(f"\n  Ruido blanco intra: {ruido_intra:.8f}")
    print(f"  Voz intra: {voz_intra:.8f}")
    print(f"  Ruido vs Voz inter: {ruido_voz_inter:.8f}")
    print(f"  Relación inter/intra (ruido): {relacion_ruido:.2f}")
    print(f"  Relación inter/intra (voz): {relacion_voz:.2f}")
    
    discriminacion_ok = relacion_ruido > 100 and relacion_voz > 100
    
    # Correlaciones óptimas
    corr_voz = correlaciones["Voz_Estudio.wav"]
    corr_ruido = correlaciones["Ruido blanco"]
    corr_tono = correlaciones["Tono puro"]
    corr_musica = correlaciones["Brandemburgo.wav"]
    corr_mezcla = correlaciones["Voz+Viento_1.wav"]
    
    acoplamientos_ok = [c for c in [corr_voz, corr_ruido, corr_tono, corr_musica, corr_mezcla] if 0.6 <= c <= 0.85]
    
    print(f"\n  Discriminación (inter/intra > 100): {'✅' if discriminacion_ok else '❌'}")
    print(f"  Acoplamiento óptimo (4/5 estímulos): {len(acoplamientos_ok)}/5")
    
    if discriminacion_ok and len(acoplamientos_ok) >= 3:
        print("\n  ✅ V63 EXITOSO: El sistema discrimina Y regula su acoplamiento.")
        print("     → La métrica de coherencia de trayectoria captura estructura real.")
    elif discriminacion_ok:
        print("\n  ⚠️  V63 PARCIAL: Discriminación lograda, acoplamiento por ajustar.")
        print("     → Ajustar parámetros de ganancia (GANANCIA_BASE, MIN, MAX).")
    else:
        print("\n  ❌ V63 FALLIDO: El sistema no discrimina o el acoplamiento es incorrecto.")
        print("     → Revisar definición de coherencia de trayectoria.")
    
    # ============================================================
    # VISUALIZACIÓN
    # ============================================================
    print("\n[6] Generando visualización...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Gráfico 1: Coherencia de trayectoria por estímulo
    ax = axes[0, 0]
    cohs = [coherencias_medias[op] for op in opciones]
    barras = ax.bar(range(len(opciones)), cohs)
    ax.axhline(y=0.7, color='g', linestyle='--', label='Alta coherencia')
    ax.axhline(y=0.5, color='orange', linestyle='--', label='Baja coherencia')
    ax.set_xticks(range(len(opciones)))
    ax.set_xticklabels([op[:12] for op in opciones], rotation=45, ha='right')
    ax.set_ylabel('Coherencia de trayectoria')
    ax.set_title('Coherencia temporal del espectro')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 2: Correlación Φ-perfil
    ax = axes[0, 1]
    corrs = [correlaciones[op] for op in opciones]
    barras = ax.bar(range(len(opciones)), corrs)
    ax.axhline(y=0.85, color='r', linestyle='--', label='Sobreacoplamiento')
    ax.axhline(y=0.6, color='orange', linestyle='--', label='Zona óptima')
    ax.axhline(y=0.4, color='g', linestyle='--', label='Desacoplamiento')
    ax.set_xticks(range(len(opciones)))
    ax.set_xticklabels([op[:12] for op in opciones], rotation=45, ha='right')
    ax.set_ylabel('Correlación Φ-perfil')
    ax.set_title('Acoplamiento por estímulo (v63)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 3: Ganancia adaptativa vs coherencia
    ax = axes[0, 2]
    for opcion in opciones:
        gan = ganancias_medias[opcion]
        coh = coherencias_medias[opcion]
        ax.scatter(coh, gan, s=150, label=opcion[:15])
    ax.set_xlabel('Coherencia de trayectoria')
    ax.set_ylabel('Ganancia espectral')
    ax.set_title('Relación ganancia vs coherencia')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 4: Evolución de Φ
    ax = axes[1, 0]
    for opcion in opciones:
        evol = estados[opcion][-1]['evolucion_phi']
        ax.plot(evol, label=opcion[:15])
    ax.set_xlabel('Ventana (x100)')
    ax.set_ylabel('Φ medio')
    ax.set_title('Evolución del campo Φ (v63)')
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
    ax.set_title('Matriz de distancias')
    plt.colorbar(im, ax=ax)
    
    # Gráfico 6: Relación inter/intra
    ax = axes[1, 2]
    relaciones = []
    nombres = []
    for i, op1 in enumerate(opciones):
        for j, op2 in enumerate(opciones):
            if i < j:
                intra = (intra_distancias[op1] + intra_distancias[op2]) / 2
                inter = inter_pares[f"{op1[:15]} vs {op2[:15]}"]
                rel = inter / intra if intra > 0 else 0
                relaciones.append(rel)
                nombres.append(f"{op1[:8]}/{op2[:8]}")
    ax.bar(range(len(relaciones)), relaciones)
    ax.axhline(y=100, color='r', linestyle='--', label='Discriminación fuerte')
    ax.axhline(y=10, color='orange', linestyle='--', label='Discriminación débil')
    ax.set_yscale('log')
    ax.set_xticks(range(len(relaciones)))
    ax.set_xticklabels(nombres, rotation=45, ha='right')
    ax.set_ylabel('Relación inter / intra (escala log)')
    ax.set_title('Discriminación entre estímulos')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('VSTCosmos v63 - Trayectoria Temporal del Espectro', fontsize=14)
    plt.tight_layout()
    plt.savefig('v63_trayectoria_espectral.png', dpi=150)
    print("  Gráfico guardado: v63_trayectoria_espectral.png")
    
    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)


if __name__ == "__main__":
    main()