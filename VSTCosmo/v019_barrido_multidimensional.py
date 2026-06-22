#!/usr/bin/env python3
"""
VSTCosmo - v19: Barrido multidimensional
La entrada modula la dinámica, no el estado.
Barremos DIFUSION_BASE, MOD_GENERACION, MOD_DECAY.
Observamos en qué región las entradas se diferencian.
"""

import numpy as np
import scipy.io.wavfile as wav
import itertools
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PARÁMETROS FIJOS
# ============================================================
DIM_FREQ = 32
DIM_TIME = 100
DT = 0.01
DURACION_SIM = 15.0
N_PASOS = int(DURACION_SIM / DT)

# Fijos (valores de v17 que funcionaban)
DECAIMIENTO_PHI = 0.04
GANANCIA_GENERACION = 0.15
GANANCIA_SOSTENIMIENTO = 0.25
REFUERZO_A = 0.15
INHIBICION_A = 0.2
DIFUSION_A = 0.08
FUERZA_RELIEVE = 0.08
LIMITE_ATENCION = DIM_FREQ * DIM_TIME * 0.35
INHIB_GLOBAL = 0.5
LIMITE_MIN = 0.0
LIMITE_MAX = 1.0

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

def vecinos(X):
    return (np.roll(X, 1, axis=0) + np.roll(X, -1, axis=0) +
            np.roll(X, 1, axis=1) + np.roll(X, -1, axis=1)) / 4.0

def perfil_modulacion(muestra):
    """Calcula el perfil de modulación basado en la entrada."""
    m = (muestra + 1.0) / 2.0
    m = np.clip(m, 0.0, 1.0)
    banda = int(m * (DIM_FREQ - 1))
    perfil = np.zeros(DIM_FREQ)
    for i in range(DIM_FREQ):
        distancia = min(abs(i - banda), DIM_FREQ - abs(i - banda))
        perfil[i] = np.exp(-(distancia ** 2) / 8.0)
    return perfil

def actualizar_campo(Phi, A, muestra, params):
    """Evolución del campo con dinámica modulada por entrada."""
    perfil = perfil_modulacion(muestra)
    promedio_local = vecinos(Phi)
    
    # Parámetros modulados
    difusion_base = params['difusion_base']
    mod_generacion = params['mod_generacion']
    mod_decay = params['mod_decay']
    
    # 1. Difusión modulada
    difusion = difusion_base * (promedio_local - Phi)
    
    # 2. Generación modulada (más fuerte donde hay entrada)
    desviacion = Phi - promedio_local
    generacion_base = GANANCIA_GENERACION * desviacion * (1 - desviacion**2)
    perfil_2d = perfil.reshape(-1, 1)  # expandir a 2D
    generacion = generacion_base * (1 + mod_generacion * perfil_2d)
    
    # 3. Decaimiento modulado (más lento donde hay entrada)
    decaimiento = -DECAIMIENTO_PHI * (Phi - promedio_local) * (1 - mod_decay * perfil_2d)
    
    # 4. Sostenimiento por A (sin modulación)
    sostenimiento = GANANCIA_SOSTENIMIENTO * A * (Phi - promedio_local)
    
    dPhi = difusion + generacion + decaimiento + sostenimiento
    Phi = Phi + DT * dPhi
    
    return np.clip(Phi, LIMITE_MIN, LIMITE_MAX)

def actualizar_atencion(A, Phi):
    vA = vecinos(A)
    auto = REFUERZO_A * A * (1.0 - A)
    inhib_local = -INHIBICION_A * vA
    difusion = DIFUSION_A * (vA - A)
    
    relieve_local = np.abs(Phi - vecinos(Phi))
    max_relieve = np.max(relieve_local)
    if max_relieve > 0:
        relieve_local = relieve_local / max_relieve
    
    acoplamiento_local = FUERZA_RELIEVE * (relieve_local - A)
    dA = auto + inhib_local + difusion + acoplamiento_local
    
    atencion_total = np.sum(A)
    if atencion_total > LIMITE_ATENCION:
        exceso = (atencion_total - LIMITE_ATENCION) / LIMITE_ATENCION
        dA += -INHIB_GLOBAL * exceso * A
    
    dA += np.random.randn(*A.shape) * 0.001
    A = A + DT * dA
    return np.clip(A, LIMITE_MIN, LIMITE_MAX)

def simular(entrada, sr, params):
    Phi = inicializar_campo()
    A = inicializar_atencion()
    
    audio = entrada['audio'] if entrada['tipo'] != 'Silencio' else np.zeros(int(sr * DURACION_SIM))
    
    for paso in range(N_PASOS):
        t = paso * DT
        if entrada['tipo'] != 'Silencio':
            idx = int(t * sr)
            idx = min(idx, len(audio) - 1)
            muestra = audio[idx] if idx >= 0 else 0.0
        else:
            muestra = 0.0
        
        A = actualizar_atencion(A, Phi)
        Phi = actualizar_campo(Phi, A, muestra, params)
    
    return np.max(Phi) - np.min(Phi)

# ============================================================
# Cargar entradas
# ============================================================
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
# BARRIDO MULTIDIMENSIONAL
# ============================================================
def barrido_multidimensional():
    print("=" * 100)
    print("VSTCosmo - v19: Barrido multidimensional")
    print("La entrada modula la dinámica (difusión, generación, decaimiento)")
    print("Buscando región donde las entradas se diferencian")
    print("=" * 100)
    
    entradas = cargar_entradas()
    
    # Diales a barrer
    difusion_vals = [0.04, 0.08, 0.12, 0.16, 0.20]
    mod_gen_vals = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0]
    mod_decay_vals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    print("\nBarriendo combinaciones...")
    print(f"Difusión: {difusion_vals}")
    print(f"Mod Generación: {mod_gen_vals}")
    print(f"Mod Decaimiento: {mod_decay_vals}")
    print(f"Total combinaciones: {len(difusion_vals) * len(mod_gen_vals) * len(mod_decay_vals)}")
    print("-" * 100)
    
    resultados = []
    
    for dif, mod_gen, mod_decay in itertools.product(difusion_vals, mod_gen_vals, mod_decay_vals):
        params = {
            'difusion_base': dif,
            'mod_generacion': mod_gen,
            'mod_decay': mod_decay
        }
        
        valores = []
        for e in entradas:
            rphi = simular(e, e['sr'], params)
            valores.append(rphi)
        
        # Calcular diferenciación: qué tan distintos son los valores
        silencio, viento, voz, voz_viento = valores
        rango = max(valores) - min(valores)
        # Diferenciación específica: voz > viento > silencio
        orden_correcto = (voz > viento and viento > silencio)
        
        if rango > 0.01 and orden_correcto:
            resultados.append({
                'difusion': dif,
                'mod_generacion': mod_gen,
                'mod_decay': mod_decay,
                'silencio': silencio,
                'viento': viento,
                'voz': voz,
                'voz_viento': voz_viento,
                'rango': rango
            })
            print(f"✓ dif={dif:.2f}, mod_gen={mod_gen:.1f}, mod_decay={mod_decay:.1f} | "
                  f"S={silencio:.4f}, Vi={viento:.4f}, Vo={voz:.4f}, Vx={voz_viento:.4f} | "
                  f"rango={rango:.4f}")
    
    # ============================================================
    # RESUMEN
    # ============================================================
    print("\n" + "=" * 100)
    print("RESUMEN: Combinaciones donde las entradas se diferencian")
    print("(silencio < viento < voz, rango > 0.01)")
    print("=" * 100)
    
    if resultados:
        print(f"\nSe encontraron {len(resultados)} combinaciones viables:")
        print("-" * 100)
        for r in resultados[:20]:  # mostrar primeras 20
            print(f"  dif={r['difusion']:.2f}, mod_gen={r['mod_generacion']:.1f}, mod_decay={r['mod_decay']:.1f} | "
                  f"S={r['silencio']:.4f}, Vi={r['viento']:.4f}, Vo={r['voz']:.4f}, Vx={r['voz_viento']:.4f}")
        
        # Mejor combinación (mayor rango)
        mejor = max(resultados, key=lambda x: x['rango'])
        print("\n" + "=" * 100)
        print("MEJOR COMBINACIÓN ENCONTRADA (mayor diferenciación):")
        print(f"  difusion_base = {mejor['difusion']:.2f}")
        print(f"  MOD_GENERACION = {mejor['mod_generacion']:.1f}")
        print(f"  MOD_DECAY = {mejor['mod_decay']:.1f}")
        print(f"\n  Rangos Φ:")
        print(f"    Silencio:     {mejor['silencio']:.4f}")
        print(f"    Viento:       {mejor['viento']:.4f}")
        print(f"    Voz_Estudio:  {mejor['voz']:.4f}")
        print(f"    Voz+Viento:   {mejor['voz_viento']:.4f}")
        print(f"    Rango total:  {mejor['rango']:.4f}")
    else:
        print("\n  No se encontraron combinaciones con diferenciación clara.")
        print("  Ajustar rangos de búsqueda o criterios.")
    
    print("\n" + "=" * 100)
    print("BARRIDO COMPLETADO")
    print("=" * 100)

if __name__ == "__main__":
    barrido_multidimensional()