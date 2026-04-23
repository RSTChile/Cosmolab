#!/usr/bin/env python3
"""
VSTCosmo - v19.2: Barrido multidimensional (criterios relajados)
DECAIMIENTO_PHI reducido, mayor generación.
Buscamos cualquier diferenciación (rango > 0.005).
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

# Version ajustada
DECAIMIENTO_PHI = 0.01  # Reducido drásticamente
GANANCIA_GENERACION = 0.20  # Aumentado
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
    m = (muestra + 1.0) / 2.0
    m = np.clip(m, 0.0, 1.0)
    banda = int(m * (DIM_FREQ - 1))
    perfil = np.zeros(DIM_FREQ)
    for i in range(DIM_FREQ):
        distancia = min(abs(i - banda), DIM_FREQ - abs(i - banda))
        perfil[i] = np.exp(-(distancia ** 2) / 8.0)
    return perfil

def actualizar_campo(Phi, A, muestra, params):
    perfil = perfil_modulacion(muestra)
    promedio_local = vecinos(Phi)
    
    difusion_base = params['difusion_base']
    mod_generacion = params['mod_generacion']
    mod_decay = params['mod_decay']
    
    difusion = difusion_base * (promedio_local - Phi)
    
    desviacion = Phi - promedio_local
    generacion_base = GANANCIA_GENERACION * desviacion * (1 - desviacion**2)
    perfil_2d = perfil.reshape(-1, 1)
    generacion = generacion_base * (1 + mod_generacion * perfil_2d)
    
    decaimiento = -DECAIMIENTO_PHI * (Phi - promedio_local) * (1 - mod_decay * perfil_2d)
    
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
# BARRIDO
# ============================================================
def barrido_multidimensional():
    print("=" * 100)
    print("VSTCosmo - v19.2: Barrido multidimensional (criterios relajados)")
    print("DECAIMIENTO_PHI = 0.01, GANANCIA_GENERACION = 0.20")
    print("Buscando cualquier diferenciación (rango > 0.005)")
    print("=" * 100)
    
    entradas = cargar_entradas()
    
    difusion_vals = [0.04, 0.08, 0.12, 0.16, 0.20]
    mod_gen_vals = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0]
    mod_decay_vals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    print(f"\nDifusión: {difusion_vals}")
    print(f"Mod Generación: {mod_gen_vals}")
    print(f"Mod Decaimiento: {mod_decay_vals}")
    print(f"Total: {len(difusion_vals) * len(mod_gen_vals) * len(mod_decay_vals)}")
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
        
        silencio, viento, voz, voz_viento = valores
        rango = max(valores) - min(valores)
        
        if rango > 0.005:
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
    
    print("\n" + "=" * 100)
    if resultados:
        print(f"Se encontraron {len(resultados)} combinaciones con diferenciación (rango > 0.005):")
        for r in resultados[:10]:
            print(f"  dif={r['difusion']:.2f}, mod_gen={r['mod_generacion']:.1f}, mod_decay={r['mod_decay']:.1f} | "
                  f"S={r['silencio']:.4f}, Vi={r['viento']:.4f}, Vo={r['voz']:.4f}, Vx={r['voz_viento']:.4f}")
        
        mejor = max(resultados, key=lambda x: x['rango'])
        print("\n" + "=" * 100)
        print("MEJOR COMBINACIÓN (mayor rango):")
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
        print("  No se encontraron combinaciones con diferenciación > 0.005.")
        print("  El sistema no logra diferenciar entradas en este espacio.")
    
    print("\n" + "=" * 100)
    print("BARRIDO COMPLETADO")
    print("=" * 100)

if __name__ == "__main__":
    barrido_multidimensional()