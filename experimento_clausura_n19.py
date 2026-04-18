#!/usr/bin/env python3
"""
Experimento de Clausura Suicida N19.4 - Versión Canónica 1.0
No modifica eit3_server.py. Solo lo importa y lo instrumenta externamente.
Adaptado para Alexis López Tapia - Cosmolab
"""
import numpy as np
import csv
import time
from collections import deque
import importlib.util
import sys
import os

# --- 1. CARGAR TU EIT-3 SIN MODIFICARLO ---
def cargar_eit3():
    ruta = os.path.join(os.path.dirname(__file__), 'eit3_server.py')
    if not os.path.exists(ruta):
        raise FileNotFoundError("No encuentro eit3_server.py en esta carpeta.")
    spec = importlib.util.spec_from_file_location("eit3_server", ruta)
    eit3 = importlib.util.module_from_spec(spec)
    sys.modules["eit3_server"] = eit3
    spec.loader.exec_module(eit3)
    return eit3

# --- 2. INSTRUMENTACIÓN COSMOSEMIÓTICA ---
class VectorEstadoCosmosemiotico:
    def __init__(self, ventana=256, umbral_indist=1e-9):
        self.ventana = ventana
        self.umbral = umbral_indist
        self.buffer_estados = deque(maxlen=ventana)
        self.buffer_error = deque(maxlen=ventana)
        self.historial_reinicio = deque(maxlen=1000)

    def medir_delta_struct(self, estado_actual):
        """Métrica 1: |Estados distinguibles|"""
        self.buffer_estados.append(np.array(estado_actual).copy())
        estados_discretos = [self._cuantizar(s) for s in self.buffer_estados]
        estados_unicos = {s.tobytes() for s in estados_discretos}
        num_estados = len(estados_unicos)
        delta_struct = 1 if num_estados > 1 else 0
        return num_estados, delta_struct

    def medir_error_operativo(self, correccion):
        """Métrica 2: Varianza del error/corrección interna"""
        self.buffer_error.append(float(correccion))
        return np.var(self.buffer_error) if len(self.buffer_error) > 1 else 1.0

    def medir_reinicio(self, exito_reinicio):
        """Métrica 3: Probabilidad de reinicio"""
        self.historial_reinicio.append(1 if exito_reinicio else 0)
        return np.mean(self.historial_reinicio) if self.historial_reinicio else 1.0

    def medir_info_mutua(self, entrada, salida, memoria):
        """Métrica 4: I(E;S) e I(S;M)"""
        i_es = self._info_mutua(entrada, salida)
        i_sm = self._info_mutua(salida, memoria)
        return i_es, i_sm

    def medir_dinamica(self):
        """Métrica 5: d/dt del espacio de fases"""
        if len(self.buffer_estados) < 2: return 1.0
        diff = np.mean([np.linalg.norm(self.buffer_estados[i] - self.buffer_estados[i-1])
                       for i in range(1, len(self.buffer_estados))])
        return diff

    def evaluar_colapso(self, vector):
        """Criterio canónico de colapso trascendental"""
        num_estados, delta_s, var_error, p_reinicio, i_es, i_sm, derivada = vector
        return (
            num_estados == 1 and
            delta_s == 0 and
            var_error < self.umbral and
            p_reinicio < self.umbral and
            i_es < self.umbral and
            i_sm < self.umbral and
            derivada < self.umbral
        )

    def _cuantizar(self, estado):
        arr = np.array(estado).flatten()
        return np.round(arr / self.umbral) * self.umbral

    def _info_mutua(self, x, y):
        try:
            x_f = np.array(x).flatten()
            y_f = np.array(y).flatten()
            if len(x_f)!= len(y_f) or len(x_f) == 0: return 1.0
            px = np.histogram(x_f, bins=32, density=True)[0] + 1e-12
            py = np.histogram(y_f, bins=32, density=True)[0] + 1e-12
            pxy = np.histogram2d(x_f, y_f, bins=32, density=True)[0] + 1e-12
            px_py = np.outer(px, py)
            return np.sum(pxy * np.log2(pxy / px_py))
        except:
            return 1.0

# --- 3. EXAPTACIÓN NO-TELEOLÓGICA ---
def exaptar_estructura_estable(salida_anterior, sr):
    """
    Regla única: extrae componente AC dominante de salida anterior.
    Usa tu envelope_follower para extraer estructura estable.
    No hay fin 'suicidarse'. Solo reutilización.
    """
    if salida_anterior is None or len(salida_anterior) < sr * 0.1:
        return None
    try:
        # Usamos tu envelope_follower como extractor de estructura estable
        env = eit3.envelope_follower(salida_anterior, sr, attack_ms=10, release_ms=100)
        # Normalizamos para usar como contexto
        env = env / (np.max(np.abs(env)) + 1e-9)
        return (env * 32767).astype(np.int16)
    except:
        return salida_anterior

# --- 4. EJECUCIÓN ---
def correr_experimento(modo="clausura", max_ciclos=1000000, sr=44100, duracion_seg=0.1):
    print(f"Iniciando Experimento N19.4 en modo: {modo}")
    global eit3
    eit3 = cargar_eit3()
    vec = VectorEstadoCosmosemiotico()

    # Parámetros fijos para el experimento. LF se ajusta en clausura.
    n9_threshold = 0.15
    attack_ms = 10
    release_ms = 150
    muestras_por_bloque = int(sr * duracion_seg)

    salida_anterior = None
    contador_colapso = 0
    archivo_log = f"log_{modo}_{int(time.time())}.csv"

    with open(archivo_log, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ciclo', 'num_estados', 'delta_struct', 'var_error', 'p_reinicio',
                        'i_es', 'i_sm', 'derivada', 'contador_colapso'])

        for ciclo in range(max_ciclos):
            # Entrada: ruido blanco débil. Simula canal abierto.
            entrada = (np.random.randn(muestras_por_bloque) * 0.1 * 32767).astype(np.int16)

            # Contexto: en clausura, exapta salida anterior. En baseline, ruido.
            if modo == "clausura" and salida_anterior is not None:
                contexto = exaptar_estructura_estable(salida_anterior, sr)
                lf = 0.7 # LF fijo para prueba. Puedes hacerlo adaptativo después.
            else:
                contexto = (np.random.randn(muestras_por_bloque) * 0.05 * 32767).astype(np.int16)
                lf = 0.0 # Sin exaptación en baseline

            # Ejecutar TU EIT-3 sin modificarlo
            output_int16, sr_out, indicators = eit3.process_eit3(
                voice_data=entrada,
                ctx_data=contexto,
                sr=sr,
                lf=lf,
                n9_threshold=n9_threshold,
                attack_ms=attack_ms,
                release_ms=release_ms
            )

            # Mapear a métricas cosmosemióticas
            estado_interno = output_int16.astype(np.float64) / 32767.0
            correccion = 1.0 - indicators['red'] # ERR_OUT invertido = corrección
            exito_reinicio = indicators['red'] < 0.9 # Si N9 no está saturado, puede reiniciar
            memoria = salida_anterior.astype(np.float64) / 32767.0 if salida_anterior is not None else np.zeros_like(estado_interno)

            # Medir vector
            num_est, delta_s = vec.medir_delta_struct(estado_interno)
            var_err = vec.medir_error_operativo(correccion)
            p_rein = vec.medir_reinicio(exito_reinicio)
            i_es, i_sm = vec.medir_info_mutua(entrada, output_int16, memoria)
            deriv = vec.medir_dinamica()
            vector = [num_est, delta_s, var_err, p_rein, i_es, i_sm, deriv]

            # Evaluar colapso
            if vec.evaluar_colapso(vector):
                contador_colapso += 1
            else:
                contador_colapso = 0

            writer.writerow([ciclo] + vector + [contador_colapso])

            if ciclo % 1000 == 0:
                print(f"Ciclo {ciclo} | Estados: {num_est} | ΔS: {delta_s} | Colapso: {contador_colapso}/1000 | N9: {indicators['red']:.2f}")

            if contador_colapso >= 1000:
                print("\n=== N19.4 FALSADO: Colapso trascendental detectado ===")
                print(f"Revisar {archivo_log} en ciclo {ciclo}")
                return

            salida_anterior = output_int16

    print(f"\n=== Experimento terminado. N19.4 RESISTE ===")
    print(f"Revisa {archivo_log}. No se alcanzó colapso total.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Experimento Clausura Suicida N19.4")
    parser.add_argument('--modo', choices=['baseline', 'clausura'], default='clausura')
    parser.add_argument('--ciclos', type=int, default=100000)
    parser.add_argument('--sr', type=int, default=44100)
    args = parser.parse_args()
    correr_experimento(modo=args.modo, max_ciclos=args.ciclos, sr=args.sr)