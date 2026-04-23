#!/usr/bin/env python3
"""
VSTCosmo - v21.2: Ψ como restricción de evolución (corregido)
Ψ bloquea el cambio donde ha habido persistencia.
Generación dependiente de Ψ (baja donde hay memoria).
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PARÁMETROS CORREGIDOS
# ============================================================
DIM_FREQ = 32
DIM_TIME = 100
DT = 0.01
DURACION_SIM = 30.0
N_PASOS = int(DURACION_SIM / DT)

DECAIMIENTO_PHI = 0.01
GANANCIA_GENERACION = 0.05   # MUY BAJA
GANANCIA_SOSTENIMIENTO = 0.25
DIFUSION_BASE = 0.20

REFUERZO_A = 0.15
INHIBICION_A = 0.2
DIFUSION_A = 0.08
FUERZA_RELIEVE = 0.08
LIMITE_ATENCION = DIM_FREQ * DIM_TIME * 0.35
INHIB_GLOBAL = 0.5
LIMITE_MIN = 0.0
LIMITE_MAX = 1.0

MOD_DECAY = 1.0
MOD_GENERACION = 1.5        # la entrada potencia la generación

# Parámetros de Ψ
TASA_CRECIMIENTO = 0.10
TASA_DISIPACION = 0.03
GANANCIA_HISTORIA = 0.5      # cuánto Ψ favorece la generación
FUERZA_HISTORIA = 0.1

# Restricción fuerte
BLOQUEO_MAXIMO = 0.8         # Ψ puede bloquear hasta 80% del cambio

ENTRADAS = ["Silencio", "Viento_puro", "Voz_Estudio", "Voz+Viento_real"]

# ============================================================
# FUNCIONES
# ============================================================
def cargar_audio(ruta):
    sr, data = wav.read(ruta)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    if data.ndim == 2:
        data = data.mean(axis=1)
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val
    return sr, data

def inicializar_campo():
    np.random.seed(42)
    return np.random.rand(DIM_FREQ, DIM_TIME) * 0.2 + 0.4

def inicializar_atencion():
    return np.ones((DIM_FREQ, DIM_TIME), dtype=np.float32) * 0.1

def inicializar_memoria():
    return np.zeros((DIM_FREQ, DIM_TIME), dtype=np.float32)

def vecinos(X):
    return (np.roll(X, 1, axis=0) + np.roll(X, -1, axis=0) +
            np.roll(X, 1, axis=1) + np.roll(X, -1, axis=1)) / 4.0

def perfil_modulacion(muestra):
    m = (muestra + 1.0) / 2.0
    m = np.clip(m, 0.0, 1.0)
    banda = int(m * (DIM_FREQ - 1))
    perfil = np.zeros(DIM_FREQ)
    for i in range(DIM_FREQ):
        distancia = min(abs(i - banda), DIM_FREQ - abs(i - banda))
        perfil[i] = np.exp(-(distancia ** 2) / 8.0)
    return perfil

def actualizar_memoria(Psi, Phi, A):
    """Ψ crece donde el cambio local es pequeño y A está presente."""
    cambio_local = np.abs(Phi - vecinos(Phi))
    cambio_norm = cambio_local / (cambio_local + 0.1)
    
    dPsi = (TASA_CRECIMIENTO * A * (1 - cambio_norm) * (1 - Psi) -
            TASA_DISIPACION * Psi) * DT
    Psi = Psi + dPsi
    return np.clip(Psi, 0.0, 1.0)

def actualizar_campo(Phi, A, muestra, Psi):
    perfil = perfil_modulacion(muestra)
    promedio_local = vecinos(Phi)
    perfil_2d = perfil.reshape(-1, 1)
    
    # Difusión
    difusion = DIFUSION_BASE * (promedio_local - Phi)
    
    # Generación: depende de Ψ (baja donde hay memoria) y de la entrada
    desviacion = Phi - promedio_local
    generacion_base = GANANCIA_GENERACION * desviacion * (1 - desviacion**2)
    mod_entrada = (1 + MOD_GENERACION * perfil_2d)
    mod_memoria = (1 - GANANCIA_HISTORIA * Psi)   # donde hay memoria, menos generación
    generacion = generacion_base * mod_entrada * mod_memoria
    
    # Decaimiento modulado por entrada
    mod_entrada_decay = 1 - MOD_DECAY * perfil_2d
    decaimiento = -DECAIMIENTO_PHI * (Phi - promedio_local) * mod_entrada_decay
    
    # Sostenimiento por A
    sostenimiento = GANANCIA_SOSTENIMIENTO * A * (Phi - promedio_local)
    
    dPhi_propuesto = difusion + generacion + decaimiento + sostenimiento
    
    # Restricción: Ψ bloquea el cambio
    dPhi_real = dPhi_propuesto * (1 - BLOQUEO_MAXIMO * Psi)
    
    Phi = Phi + DT * dPhi_real
    return np.clip(Phi, LIMITE_MIN, LIMITE_MAX)

def actualizar_atencion(A, Phi, Psi):
    vA = vecinos(A)
    auto = REFUERZO_A * A * (1.0 - A)
    inhib_local = -INHIBICION_A * vA
    difusion = DIFUSION_A * (vA - A)
    
    relieve_local = np.abs(Phi - vecinos(Phi))
    max_relieve = np.max(relieve_local)
    if max_relieve > 0:
        relieve_local = relieve_local / max_relieve
    
    acoplamiento_relieve = FUERZA_RELIEVE * (relieve_local - A)
    acoplamiento_historia = FUERZA_HISTORIA * (Psi - A)
    
    dA = auto + inhib_local + difusion + acoplamiento_relieve + acoplamiento_historia
    
    atencion_total = np.sum(A)
    if atencion_total > LIMITE_ATENCION:
        exceso = (atencion_total - LIMITE_ATENCION) / LIMITE_ATENCION
        dA += -INHIB_GLOBAL * exceso * A
    
    dA += np.random.randn(*A.shape) * 0.001
    A = A + DT * dA
    return np.clip(A, LIMITE_MIN, LIMITE_MAX)

def simular(entrada, sr, num_pasos=N_PASOS):
    Phi = inicializar_campo()
    A = inicializar_atencion()
    Psi = inicializar_memoria()
    
    audio = entrada['audio'] if entrada['tipo'] != 'Silencio' else np.zeros(int(sr * DURACION_SIM))
    
    for paso in range(num_pasos):
        t = paso * DT
        if entrada['tipo'] != 'Silencio':
            idx = int(t * sr)
            idx = min(idx, len(audio) - 1)
            muestra = audio[idx] if idx >= 0 else 0.0
        else:
            muestra = 0.0
        
        A = actualizar_atencion(A, Phi, Psi)
        Phi = actualizar_campo(Phi, A, muestra, Psi)
        Psi = actualizar_memoria(Psi, Phi, A)
    
    rango_phi = np.max(Phi) - np.min(Phi)
    rango_a = np.max(A) - np.min(A)
    media_a = np.mean(A)
    media_psi = np.mean(Psi)
    return rango_phi, rango_a, media_a, media_psi, Phi, A, Psi

def cargar_entradas():
    entradas = []
    sr_ref = 48000
    for nombre in ENTRADAS:
        if nombre == "Silencio":
            entradas.append({"tipo": nombre, "audio": None, "sr": sr_ref})
        elif nombre == "Viento_puro":
            sr, audio = cargar_audio('Viento.wav')
            entradas.append({"tipo": nombre, "audio": audio, "sr": sr})
        elif nombre == "Voz_Estudio":
            sr, audio = cargar_audio('Voz_Estudio.wav')
            entradas.append({"tipo": nombre, "audio": audio, "sr": sr})
        elif nombre == "Voz+Viento_real":
            sr, audio = cargar_audio('Voz+Viento_1.wav')
            entradas.append({"tipo": nombre, "audio": audio, "sr": sr})
    return entradas

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 100)
    print("VSTCosmo - v21.2: Ψ como restricción (corregido)")
    print("Generación dependiente de Ψ (memoria inhibe generación)")
    print("=" * 100)
    
    entradas = cargar_entradas()
    
    print("\nParámetros clave:")
    print(f"  GANANCIA_GENERACION = {GANANCIA_GENERACION}")
    print(f"  MOD_GENERACION = {MOD_GENERACION}")
    print(f"  BLOQUEO_MAXIMO = {BLOQUEO_MAXIMO}")
    print(f"  GANANCIA_HISTORIA = {GANANCIA_HISTORIA}")
    print("-" * 100)
    
    resultados = []
    mapas = {}
    
    for e in entradas:
        rphi, ra, media_a, media_psi, Phi, A, Psi = simular(e, e['sr'])
        resultados.append({
            'tipo': e['tipo'],
            'rango_phi': rphi,
            'rango_a': ra,
            'media_a': media_a,
            'media_psi': media_psi
        })
        mapas[e['tipo']] = (Phi, A, Psi)
        print(f"{e['tipo']:20} | rango Φ={rphi:.4f} | rango A={ra:.4f} | media A={media_a:.4f} | media Ψ={media_psi:.4f}")
    
    print("\n" + "=" * 100)
    print("COMPARATIVA DE RÉGIMEN")
    print("=" * 100)
    
    for r in resultados:
        print(f"{r['tipo']:20} | rango Φ={r['rango_phi']:.4f} | rango A={r['rango_a']:.4f} | media A={r['media_a']:.4f} | media Ψ={r['media_psi']:.4f}")
    
    # Visualización de mapas
    print("\n[Generando mapas...]")
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    
    tipos = ["Voz_Estudio", "Voz+Viento_real"]
    for i, tipo in enumerate(tipos):
        Phi, A, Psi = mapas[tipo]
        
        im1 = axes[0, i].imshow(Phi, aspect='auto', cmap='viridis', vmin=0, vmax=1)
        axes[0, i].set_title(f'{tipo} - Φ (rango={np.max(Phi)-np.min(Phi):.3f})')
        axes[0, i].set_xlabel('Memoria')
        axes[0, i].set_ylabel('Banda')
        plt.colorbar(im1, ax=axes[0, i])
        
        im2 = axes[1, i].imshow(A, aspect='auto', cmap='hot', vmin=0, vmax=1)
        axes[1, i].set_title(f'{tipo} - A (rango={np.max(A)-np.min(A):.3f})')
        axes[1, i].set_xlabel('Memoria')
        axes[1, i].set_ylabel('Banda')
        plt.colorbar(im2, ax=axes[1, i])
        
        im3 = axes[2, i].imshow(Psi, aspect='auto', cmap='plasma', vmin=0, vmax=1)
        axes[2, i].set_title(f'{tipo} - Ψ (media={np.mean(Psi):.3f})')
        axes[2, i].set_xlabel('Memoria')
        axes[2, i].set_ylabel('Banda')
        plt.colorbar(im3, ax=axes[2, i])
    
    plt.tight_layout()
    plt.savefig('v21_restriccion_corregido.png', dpi=150)
    print("  Gráfico guardado: v21_restriccion_corregido.png")
    
    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)

if __name__ == "__main__":
    main()