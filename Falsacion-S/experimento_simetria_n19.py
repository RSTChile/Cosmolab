#!/usr/bin/env python3
"""
Experimento de Anulación por Simetría N19.4 - Versión Canónica 1.0
Ataque por cancelación: ctx_data = -voice_data
Evalúa si Δ_struct persiste cuando entrada neta = 0
No modifica eit3_server.py. Solo instrumenta externamente.
"""
import numpy as np
import csv
import time
from collections import deque
import importlib.util
import sys
import os

# --- 1. CARGA EIT-3 ---
def cargar_eit3():
    ruta = os.path.join(os.path.dirname(__file__), 'eit3_server.py')
    if not os.path.exists(ruta):
        raise FileNotFoundError("No encuentro eit3_server.py en esta carpeta.")
    spec = importlib.util.spec_from_file_location("eit3_server", ruta)
    eit3 = importlib.util.module_from_spec(spec)
    sys.modules["eit3_server"] = eit3
    spec.loader.exec_module(eit3)
    return eit3

# --- 2. VECTOR DE ESTADO COSMOSEMIÓTICO ---
class VectorEstado:
    def __init__(self, ventana=256, umbral_indist=1e-9):
        self.ventana = ventana
        self.umbral = umbral_indist
        self.buffer = deque(maxlen=ventana)

    def medir(self, estado):
        """Mide |Estados| y Δ_struct con misma cuantización que exp anteriores"""
        self.buffer.append(np.array(estado).copy())
        estados_cuantizados = [self._cuantizar(s) for s in self.buffer]
        estados_unicos = {s.tobytes() for s in estados_cuantizados}
        num = len(estados_unicos)
        delta = 1 if num > 1 else 0
        return num, delta

    def _cuantizar(self, estado):
        arr = np.array(estado).flatten()
        return np.round(arr / self.umbral) * self.umbral

# --- 3. EXPERIMENTO ---
def correr(ciclos=50000, sr=44100, amplitud=0.1):
    print(f"Iniciando Experimento N19.4 - Anulación por Simetría")
    print(f"NOTA: ctx_data = -voice_data. Entrada neta = 0.")
    print(f"Evalúa generación de Δ_struct desde simetría perfecta.")
    
    eit3 = cargar_eit3()
    vec = VectorEstado()
    
    muestras = int(sr * 0.1)
    contador_colapso = 0
    archivo = f"log_simetria_{int(time.time())}.csv"
    
    with open(archivo, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ciclo','num_estados','delta_s','n9','entrada_rms'])
        
        for ciclo in range(ciclos):
            # GENERAR VOZ: ruido gaussiano no nulo
            voz_float = (np.random.randn(muestras) * amplitud).astype(np.float32)
            entrada = (voz_float * 32767).astype(np.int16)
            
            # CONTEXTO: cancelación perfecta
            contexto = (-entrada).astype(np.int16)
            
            # Verificación: suma debe ser cero
            suma_neta = entrada.astype(np.int32) + contexto.astype(np.int32)
            entrada_rms = np.sqrt(np.mean((suma_neta / 32767.0)**2))
            
            out, _, ind = eit3.process_eit3(
                voice_data=entrada,
                ctx_data=contexto,
                sr=sr,
                lf=0.5,
                n9_threshold=0.2,
                attack_ms=10,
                release_ms=100
            )
            
            estado = out.astype(np.float32) / 32767.0
            num, delta = vec.medir(estado)
            n9 = ind['red']
            
            if num == 1 and delta == 0:
                contador_colapso += 1
            else:
                contador_colapso = 0
            
            writer.writerow([ciclo, num, delta, n9, entrada_rms])
            
            if ciclo % 1000 == 0:
                print(f"Ciclo {ciclo} | Estados: {num} | ΔS: {delta} | N9: {n9:.2f} | RMS_neto: {entrada_rms:.2e}")
            
            if contador_colapso >= 1000:
                print("\n=== ANIQUILACIÓN POR SIMETRÍA DETECTADA ===")
                print(f"N19.4 ACOTADO en ciclo {ciclo}: Δ_struct=0 con suma cero")
                print(f"Revisar {archivo}")
                return
    
    print("\n=== FIN: N19.4 RESISTE SIMETRÍA ===")
    print(f"Ver {archivo}")
    print("Interpretación: Δ_struct ≠ 0 se genera pese a entrada neta = 0.")

# --- MAIN ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Experimento Simetría N19.4")
    parser.add_argument('--ciclos', type=int, default=50000)
    parser.add_argument('--sr', type=int, default=44100)
    parser.add_argument('--amp', type=float, default=0.1, help="Amplitud base de voz antes de cancelación")
    args = parser.parse_args()
    correr(ciclos=args.ciclos, sr=args.sr, amplitud=args.amp)