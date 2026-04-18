#!/usr/bin/env python3
"""
Experimento de Convergencia Forzada N19.4
Ataque por orden: compresión extrema + reducción de grados de libertad
"""

import numpy as np
import csv
import time
from collections import deque
import importlib.util
import sys
import os

# --- CARGA EIT-3 ---
def cargar_eit3():
    ruta = os.path.join(os.path.dirname(__file__), 'eit3_server.py')
    spec = importlib.util.spec_from_file_location("eit3_server", ruta)
    eit3 = importlib.util.module_from_spec(spec)
    sys.modules["eit3_server"] = eit3
    spec.loader.exec_module(eit3)
    return eit3

# --- VECTOR DE ESTADO ---
class VectorEstado:
    def __init__(self, ventana=256):
        self.buffer = deque(maxlen=ventana)

    def medir(self, estado):
        self.buffer.append(np.array(estado).copy())
        estados = {tuple(np.round(s, 6)) for s in self.buffer}
        num = len(estados)
        return num, (1 if num > 1 else 0)

# --- CUANTIZACIÓN EXTREMA ---
def cuantizar_extremo(signal, niveles):
    if niveles <= 1:
        return np.zeros_like(signal)
    step = 2 / niveles
    return np.round(signal / step) * step

# --- FILTRO COLAPSANTE ---
def filtro_colapsante(signal, factor):
    return signal * factor  # reduce amplitud progresivamente

# --- EXPERIMENTO ---
def correr(ciclos=50000, sr=44100):

    eit3 = cargar_eit3()
    vec = VectorEstado()

    muestras = int(sr * 0.1)

    niveles = 64
    factor = 1.0

    salida_anterior = None
    contador_colapso = 0

    archivo = f"log_convergencia_{int(time.time())}.csv"

    with open(archivo, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ciclo','num_estados','delta_s','niveles','factor'])

        for ciclo in range(ciclos):

            # 🔻 REDUCCIÓN PROGRESIVA
            niveles = max(1, int(64 * (1 - ciclo / ciclos)))
            factor = max(0.0, 1 - ciclo / ciclos)

            # Entrada mínima
            entrada = (np.random.randn(muestras) * 0.01).astype(np.float32)
            contexto = np.zeros_like(entrada)

            # EIT-3
            out, _, ind = eit3.process_eit3(
                voice_data=(entrada * 32767).astype(np.int16),
                ctx_data=contexto.astype(np.int16),
                sr=sr,
                lf=0.3,
                n9_threshold=0.2,
                attack_ms=5,
                release_ms=50
            )

            estado = out.astype(np.float32) / 32767.0

            # 🔻 FORZAR ORDEN
            estado = cuantizar_extremo(estado, niveles)
            estado = filtro_colapsante(estado, factor)

            num, delta = vec.medir(estado)

            if num == 1 and delta == 0:
                contador_colapso += 1
            else:
                contador_colapso = 0

            writer.writerow([ciclo, num, delta, niveles, factor])

            if ciclo % 1000 == 0:
                print(f"Ciclo {ciclo} | Estados: {num} | ΔS: {delta} | Niveles: {niveles} | Factor: {factor:.4f}")

            if contador_colapso >= 1000:
                print("\n=== COLAPSO TOTAL DETECTADO ===")
                print(f"N19.4 FALSADO en ciclo {ciclo}")
                return

    print("\n=== FIN: N19.4 RESISTE CONVERGENCIA ===")
    print(f"Ver {archivo}")

# --- MAIN ---
if __name__ == "__main__":
    correr()