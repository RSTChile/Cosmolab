#!/usr/bin/env python3
"""
VSTCosmo - Experimento v11: Pastor Cosmosemiótico
El Pastor ajusta dinámicamente los parámetros para mantener el sistema en la zona fértil.
Múltiples iteraciones para observar convergencia.
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PARÁMETROS BASE
# ============================================================
DIM_FREQ = 32
DIM_TIME = 100
DT = 0.01
DURACION = 20.0
N_PASOS = int(DURACION / DT)

# Parámetros de A (fijos)
REFUERZO_A = 0.15
INHIBICION_A = 0.2
DIFUSION_A = 0.08
DISIPACION_A = 0.01
BASAL_A = 0.05

# Acoplamiento A → Φ
DIFUSION_ACOPLAMIENTO = 0.2

# Límites
LIMITE_MAX = 1.0
LIMITE_MIN = 0.0

# ============================================================
# PASTOR COSMOSEMIÓTICO
# ============================================================
class PastorCosmosemiotico:
    def __init__(self):
        # Parámetros ajustables por el pastor
        self.ganancia_inest = 0.2
        self.intensidad_entrada = 0.05
        self.difusion_phi = 0.1
        
        # Historial
        self.historial_ganancia = []
        self.historial_intensidad = []
        self.historial_rango_phi = []
        self.historial_rango_a = []
        
    def ajustar(self, estado):
        """
        Ajusta parámetros basado en el estado del sistema.
        El pastor busca:
        - rango_Phi entre 0.3 y 0.7 (no saturado, no homogéneo)
        - rango_A > 0.01 (atención diferenciada)
        """
        rango_phi = estado['rango_phi']
        rango_a = estado['rango_a']
        
        # Ajustar inestabilidad basado en saturación de Φ
        if rango_phi > 0.85:
            self.ganancia_inest *= 0.95
            accion = "reduciendo inestabilidad"
        elif rango_phi < 0.25:
            self.ganancia_inest *= 1.08
            accion = "aumentando inestabilidad"
        else:
            accion = "inestabilidad OK"
        
        # Ajustar intensidad de entrada basado en diferenciación de A
        if rango_a < 0.008:
            self.intensidad_entrada *= 1.05
            accion_entrada = "aumentando intensidad"
        elif rango_a > 0.05:
            self.intensidad_entrada *= 0.97
            accion_entrada = "reduciendo intensidad"
        else:
            accion_entrada = "intensidad OK"
        
        # Mantener rangos
        self.ganancia_inest = np.clip(self.ganancia_inest, 0.01, 0.5)
        self.intensidad_entrada = np.clip(self.intensidad_entrada, 0.01, 0.3)
        
        print(f"    Pastor: {accion} -> G={self.ganancia_inest:.3f} | {accion_entrada} -> I={self.intensidad_entrada:.3f}")
        
        # Guardar historial
        self.historial_ganancia.append(self.ganancia_inest)
        self.historial_intensidad.append(self.intensidad_entrada)
        self.historial_rango_phi.append(rango_phi)
        self.historial_rango_a.append(rango_a)
        
        return self.ganancia_inest, self.intensidad_entrada

# ============================================================
# FUNCIONES DEL SISTEMA
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
    return np.random.rand(DIM_FREQ, DIM_TIME) * 0.2 + 0.4

def inicializar_atencion():
    return np.ones((DIM_FREQ, DIM_TIME)) * BASAL_A

def vecinos_phi(Phi):
    return (np.roll(Phi, 1, axis=0) + np.roll(Phi, -1, axis=0) +
            np.roll(Phi, 1, axis=1) + np.roll(Phi, -1, axis=1)) / 4

def reconfigurar_campo(Phi, muestra, paso, ganancia_inest, intensidad_entrada):
    """La perturbación suprime la inestabilidad donde es fuerte."""
    m = (muestra + 1) / 2
    m = np.clip(m, 0, 1)
    banda_preferida = int(m * (DIM_FREQ - 1))
    
    Phi_nuevo = Phi.copy()
    
    for i in range(DIM_FREQ):
        distancia = min(abs(i - banda_preferida), DIM_FREQ - abs(i - banda_preferida))
        preferencia = np.exp(-distancia**2 / 4)
        
        desviacion = Phi[i] - 0.5
        inestabilidad_base = ganancia_inest * desviacion * (1 - desviacion**2)
        
        # La perturbación suprime la inestabilidad donde es fuerte
        supresion = 0.8 * preferencia
        inestabilidad = inestabilidad_base * (1 - supresion)
        
        Phi_nuevo[i] = Phi[i] + DT * inestabilidad
    
    # Difusión
    vecinos = vecinos_phi(Phi_nuevo)
    difusion = 0.1 * (vecinos - Phi_nuevo)
    Phi_nuevo = Phi_nuevo + DT * difusion
    
    # Perturbación adicional (suave)
    if intensidad_entrada > 0:
        perturbacion = intensidad_entrada * muestra
        Phi_nuevo[banda_preferida] += DT * perturbacion
    
    return np.clip(Phi_nuevo, LIMITE_MIN, LIMITE_MAX)

def vecinos_a(A):
    return (np.roll(A, 1, axis=0) + np.roll(A, -1, axis=0) +
            np.roll(A, 1, axis=1) + np.roll(A, -1, axis=1)) / 4

def actualizar_atencion(A, Phi, Phi_prev):
    auto = REFUERZO_A * A * (1 - A)
    inhib = -INHIBICION_A * vecinos_a(A)
    dif = DIFUSION_A * (vecinos_a(A) - A)
    dis = -DISIPACION_A * (A - BASAL_A)
    
    dA_base = auto + inhib + dif + dis
    
    grad_temporal = Phi - Phi_prev
    prop = 0.02 * np.roll(A, 1, axis=1) * np.maximum(grad_temporal, 0)
    prop += 0.01 * np.roll(A, -1, axis=1) * np.maximum(-grad_temporal, 0)
    
    dA = dA_base + prop
    dA += np.random.randn(*A.shape) * 0.001
    
    A = A + DT * dA
    return np.clip(A, LIMITE_MIN, LIMITE_MAX)

def acoplamiento_atencion_campo(Phi, A):
    vecinos = vecinos_phi(Phi)
    mezcla = (1 - 0.5 * A) * Phi + 0.5 * A * vecinos
    flujo = mezcla - Phi
    Phi = Phi + DT * 0.2 * flujo
    return np.clip(Phi, LIMITE_MIN, LIMITE_MAX)

def simular(audio, sr, nombre, pastor, num_iteraciones=10):
    print(f"\n  Simulando: {nombre} ({num_iteraciones} iteraciones)")
    
    historial = []
    
    for iteracion in range(num_iteraciones):
        print(f"    Iteración {iteracion + 1}/{num_iteraciones}")
        
        Phi = inicializar_campo()
        A = inicializar_atencion()
        Phi_prev = Phi.copy()
        
        # Usar parámetros actuales del pastor
        ganancia = pastor.ganancia_inest
        intensidad = pastor.intensidad_entrada
        
        for paso in range(N_PASOS):
            t = paso * DT
            idx = int(t * sr)
            idx = min(idx, len(audio) - 1) if len(audio) > 0 else 0
            muestra = audio[idx] if idx >= 0 and len(audio) > 0 else 0.0
            
            Phi = reconfigurar_campo(Phi, muestra, paso, ganancia, intensidad)
            A = actualizar_atencion(A, Phi, Phi_prev)
            Phi = acoplamiento_atencion_campo(Phi, A)
            Phi_prev = Phi.copy()
        
        rango_phi = np.max(Phi) - np.min(Phi)
        rango_a = np.max(A) - np.min(A)
        
        print(f"      Resultado: rango Φ={rango_phi:.3f}, rango A={rango_a:.4f}")
        
        # Pastor ajusta basado en el resultado
        estado = {'rango_phi': rango_phi, 'rango_a': rango_a}
        pastor.ajustar(estado)
        
        historial.append({
            'iteracion': iteracion + 1,
            'rango_phi': rango_phi,
            'rango_a': rango_a,
            'ganancia': pastor.ganancia_inest,
            'intensidad': pastor.intensidad_entrada
        })
    
    return historial

def main():
    print("=" * 60)
    print("VSTCosmo - Experimento v11: Pastor Cosmosemiótico")
    print("El Pastor ajusta parámetros para mantener la zona fértil")
    print("10 iteraciones por tipo de entrada")
    print("=" * 60)
    
    # Cargar archivos
    print("\n[1] Cargando archivos...")
    sr_v, voz_viento = cargar_audio('Voz+Viento_1.wav')
    sr_w, viento = cargar_audio('Viento.wav')
    sr_vc, voz_limpia = cargar_audio('Voz_Estudio.wav')
    
    print(f"    Voz+Viento: {len(voz_viento)/sr_v:.2f}s")
    print(f"    Viento: {len(viento)/sr_w:.2f}s")
    print(f"    Voz_Estudio: {len(voz_limpia)/sr_vc:.2f}s")
    
    # Datasets
    datasets = [
        (voz_viento, sr_v, "Voz+Viento"),
        (viento, sr_w, "Viento"),
        (voz_limpia, sr_vc, "Voz_Estudio")
    ]
    
    # Correr con silencio también
    audio_silencio = np.zeros(int(sr_v * DURACION))
    datasets.append((audio_silencio, sr_v, "Silencio"))
    
    resultados = {}
    
    for audio, sr, nombre in datasets:
        print(f"\n[2] Ejecutando para: {nombre}")
        pastor = PastorCosmosemiotico()
        historial = simular(audio, sr, nombre, pastor, num_iteraciones=10)
        resultados[nombre] = historial
    
    # ============================================================
    # VISUALIZACIÓN DE CONVERGENCIA
    # ============================================================
    print("\n[3] Generando visualizaciones...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, (nombre, historial) in enumerate(resultados.items()):
        ax = axes[idx // 2, idx % 2]
        
        iteraciones = [h['iteracion'] for h in historial]
        rango_phi = [h['rango_phi'] for h in historial]
        rango_a = [h['rango_a'] for h in historial]
        ganancia = [h['ganancia'] for h in historial]
        
        ax.plot(iteraciones, rango_phi, 'b-', label='rango Φ', marker='o')
        ax.plot(iteraciones, rango_a, 'r-', label='rango A', marker='s')
        ax.plot(iteraciones, ganancia, 'g--', label='ganancia inest', marker='^')
        
        ax.axhline(y=0.7, color='b', linestyle='--', alpha=0.5)
        ax.axhline(y=0.3, color='b', linestyle='--', alpha=0.5)
        ax.axhline(y=0.02, color='r', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Iteración')
        ax.set_ylabel('Valor')
        ax.set_title(f'{nombre}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('VSTCosmo v11 - Convergencia del Pastor Cosmosemiótico', fontsize=14)
    plt.tight_layout()
    plt.savefig('vstcosmo_v11_pastor.png', dpi=150)
    print("  Gráfico guardado: vstcosmo_v11_pastor.png")
    
    # ============================================================
    # RESUMEN FINAL
    # ============================================================
    print("\n" + "=" * 60)
    print("RESUMEN DE CONVERGENCIA")
    print("=" * 60)
    
    for nombre, historial in resultados.items():
        final = historial[-1]
        print(f"\n{nombre}:")
        print(f"  rango Φ final: {final['rango_phi']:.3f}")
        print(f"  rango A final: {final['rango_a']:.4f}")
        print(f"  ganancia inest final: {final['ganancia']:.3f}")
        
        if 0.3 < final['rango_phi'] < 0.7:
            print("  → Φ en zona fértil (0.3-0.7) ✓")
        elif final['rango_phi'] > 0.85:
            print("  → Φ saturado ✗")
        elif final['rango_phi'] < 0.2:
            print("  → Φ homogéneo ✗")
        
        if final['rango_a'] > 0.01:
            print("  → Atención diferenciada ✓")
        else:
            print("  → Atención uniforme ✗")
    
    print("\n" + "=" * 60)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 60)

if __name__ == "__main__":
    main()