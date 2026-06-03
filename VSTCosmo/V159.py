#!/usr/bin/env python3
"""
EXPERIMENTO V159 — ANIMA-2 Etapa 3: RITUAL (SIN ENTRENAMIENTO INICIAL)
================================================================================
CORRECCIÓN CRÍTICA:
  - Eliminar entrenamiento de lateralidad (el organismo aprende durante el experimento)
  - Mismo flujo que V158: empezar desde cero en F1
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Tuple, List, Dict
import time
from datetime import datetime
import os

# ============================================================================
# PARÁMETROS EXACTOS DE V158
# ============================================================================

@dataclass
class ParamsRitual:
    tau_ritual: float = 120.0
    repeticion_min: int = 3
    ritual_gain: float = 0.05
    patron_temporal: float = 30.0
    tolerancia_patron: float = 0.3
    umbral_ritual_activacion: float = 0.7


@dataclass
class ParamsCosmosemiotica:
    dt: float = 0.1
    K_GAIN: float = 0.00003
    Kp_base: float = 0.002
    
    tau_cb: float = 10.0
    umbral_cb_baseline: float = 35.0
    
    tau_mem_base: float = 30.0
    k_mem: float = 0.05
    
    K_COG: float = 0.5
    lambda_fisico: float = 0.1
    lambda_costo_motor: float = 0.3
    
    ritual: ParamsRitual = field(default_factory=ParamsRitual)
    
    FATIGA_MAX: float = 100000.0
    ORIENT_MIN: float = -60.0
    ORIENT_MAX: float = 60.0


# ============================================================================
# ORGANISMO V159
# ============================================================================

class OrganismoV159:
    def __init__(self, name: str, params: ParamsCosmosemiotica, seed: int = None, verbose: bool = False):
        self.name = name
        self.p = params
        self.verbose = verbose
        
        if seed is not None:
            np.random.seed(seed)
        
        self.reset()
    
    def reset(self):
        self.orientacion = 0.0
        self.fatiga = 0.0
        self.historia_fisica = 0.0
        self.historia_intencional = 0.0
        self.Cb = 0.0
        self.setpoint_recordado = 0.0
        self.confianza_memoria = 1.0
        self.tau_mem_actual = self.p.tau_mem_base
        
        self.ritual_activation = 0.0
        self.patron_buffer = []
        self.repeticiones_consecutivas = 0
        self.ritual_active = False
        
        self.tiempo_total = 0.0
        self.tiempo_juego = 0.0
        self.juego_active = False
        self.refractory_counter = 0
        self.episodios_juego = 0
        self._error_prev = 0.0
        
        self.history = {
            't': [], 'orientacion': [], 'Cb': [], 'fatiga': [],
            'ritual_activation': [], 'setpoint': [], 'error': []
        }
    
    def actualizar_memoria_ausencia(self, error: float, setpoint_objetivo: float):
        self.confianza_memoria *= np.exp(-self.p.dt / self.tau_mem_actual)
        
        if abs(error) < 5.0:
            alpha = 0.1
            self.setpoint_recordado = (alpha * setpoint_objetivo + 
                                        (1-alpha) * self.setpoint_recordado)
            self.confianza_memoria = min(1.0, self.confianza_memoria + 0.05)
        else:
            self.confianza_memoria *= 0.995
        
        self.tau_mem_actual = min(100.0, self.p.tau_mem_base + 
                                   self.p.k_mem * self.historia_intencional / 100)
    
    def actualizar_consciencia(self, error: float, as_sys_env: float = 0.0):
        entrada_cb = error * (1.0 - as_sys_env)
        dCb = (entrada_cb - self.Cb / self.p.tau_cb) * self.p.dt
        self.Cb += dCb
        self.Cb = max(0.0, self.Cb)
    
    def detectar_patron(self, delta_intencional: float) -> bool:
        if abs(delta_intencional) < 0.5:
            return False
        
        direccion = np.sign(delta_intencional)
        magnitud = abs(delta_intencional)
        
        for t_prev, mag_prev, dir_prev in self.patron_buffer:
            dt_desde_prev = self.tiempo_total - t_prev
            
            timing_ok = abs(dt_desde_prev - self.p.ritual.patron_temporal) <= (
                self.p.ritual.patron_temporal * self.p.ritual.tolerancia_patron
            )
            
            magnitud_ok = abs(magnitud - mag_prev) / max(magnitud, mag_prev, 0.1) < 0.3
            direccion_ok = dir_prev == direccion
            
            if timing_ok and magnitud_ok and direccion_ok:
                return True
        
        return False
    
    def actualizar_ritual(self, delta_intencional: float):
        es_patron = self.detectar_patron(delta_intencional)
        
        if es_patron:
            self.repeticiones_consecutivas += 1
            
            if self.repeticiones_consecutivas >= self.p.ritual.repeticion_min:
                incremento = self.Cb * self.repeticiones_consecutivas / 100.0
                self.ritual_activation += incremento * self.p.dt
        else:
            self.repeticiones_consecutivas = max(0, self.repeticiones_consecutivas - 0.5)
        
        self.ritual_activation *= np.exp(-self.p.dt / self.p.ritual.tau_ritual)
        self.ritual_activation = max(0.0, min(2.0, self.ritual_activation))
        
        self.ritual_active = (self.ritual_activation > 
                               self.p.ritual.umbral_ritual_activacion)
        
        if abs(delta_intencional) > 0.5:
            self.patron_buffer.append((self.tiempo_total, abs(delta_intencional),
                                        np.sign(delta_intencional)))
            if len(self.patron_buffer) > 10:
                self.patron_buffer.pop(0)
    
    def calcular_correccion(self, error: float) -> Tuple[float, float]:
        error_deriv = (error - self._error_prev) / self.p.dt
        self._error_prev = error
        
        correccion_base = self.p.Kp_base * error + 0.001 * error_deriv
        
        if self.ritual_active and self.patron_buffer and self.repeticiones_consecutivas >= 1:
            _, ultima_mag, ultima_dir = self.patron_buffer[-1]
            correccion_ritual = ultima_dir * ultima_mag * self.p.ritual.ritual_gain
            correccion = correccion_base * (1 - self.ritual_activation * 0.2) + correccion_ritual * self.ritual_activation
        else:
            correccion = correccion_base
        
        delta_max = 5.0 * self.p.dt
        delta_intencional = np.clip(correccion, -delta_max, delta_max)
        
        juego_activado = False
        if not self.ritual_active:
            if len(self.history['Cb']) > 200:
                umbral_dinamico = np.percentile(self.history['Cb'][-200:], 70)
            else:
                umbral_dinamico = self.p.umbral_cb_baseline
            
            if self.Cb > umbral_dinamico and self.refractory_counter <= 0:
                juego_activado = True
                self.episodios_juego += 1
        
        if juego_activado and not self.ritual_active:
            self.juego_active = True
            self.tiempo_juego += self.p.dt
            
            delta_fisico = delta_intencional * self.p.lambda_fisico
            costo_motor = abs(delta_intencional) * self.p.lambda_costo_motor
            costo_cognitivo = self.p.K_COG * self.Cb * self.p.dt
            costo_total = costo_motor + costo_cognitivo
        else:
            if self.refractory_counter <= 0:
                self.juego_active = False
            
            delta_fisico = delta_intencional
            costo_total = abs(delta_intencional)
        
        if not juego_activado and self.refractory_counter > 0:
            self.refractory_counter -= self.p.dt
        
        self.historia_intencional += abs(delta_intencional)
        self.historia_fisica += abs(delta_fisico)
        
        self.fatiga += costo_total * self.p.K_GAIN
        self.fatiga = min(self.fatiga, self.p.FATIGA_MAX)
        
        return delta_intencional, delta_fisico
    
    def step(self, setpoint_externo: float, ruido: float = 0.0, 
             modo_ritual_forzado: bool = None) -> float:
        self.tiempo_total += self.p.dt
        
        setpoint_efectivo = setpoint_externo + ruido
        
        error_antes = setpoint_efectivo - self.orientacion
        self.actualizar_memoria_ausencia(error_antes, setpoint_efectivo)
        
        if self.confianza_memoria < 0.3:
            setpoint_objetivo = self.setpoint_recordado
        else:
            setpoint_objetivo = setpoint_efectivo
        
        error = setpoint_objetivo - self.orientacion
        
        if modo_ritual_forzado is not None:
            self.ritual_active = modo_ritual_forzado
        
        delta_intencional, delta_fisico = self.calcular_correccion(error)
        
        if modo_ritual_forzado is None:
            self.actualizar_ritual(delta_intencional)
        
        self.actualizar_consciencia(abs(error), 0.0)
        
        self.orientacion += delta_fisico
        self.orientacion = np.clip(self.orientacion, self.p.ORIENT_MIN, self.p.ORIENT_MAX)
        
        self.history['t'].append(self.tiempo_total)
        self.history['orientacion'].append(self.orientacion)
        self.history['Cb'].append(self.Cb)
        self.history['fatiga'].append(self.fatiga)
        self.history['ritual_activation'].append(self.ritual_activation)
        self.history['setpoint'].append(setpoint_objetivo)
        self.history['error'].append(error)
        
        return error
    
    def get_error_rms(self, segundos_ultimos: float = 10.0) -> float:
        if len(self.history['t']) < 2:
            return 0.0
        
        t_max = self.tiempo_total
        t_min = max(0, t_max - segundos_ultimos)
        
        errores = [err for t, err in zip(self.history['t'], self.history['error']) if t >= t_min]
        
        if not errores:
            return 0.0
        return np.sqrt(np.mean(np.square(errores)))


# ============================================================================
# EXPERIMENTO V159 (SIN ENTRENAMIENTO INICIAL)
# ============================================================================

class ExperimentoV159:
    def __init__(self, seed: int = 42, verbose: bool = True):
        self.seed = seed
        self.verbose = verbose
        self.params = ParamsCosmosemiotica()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = f"v159_logs"
        os.makedirs(self.log_dir, exist_ok=True)
    
    def ejecutar_ciclo(self, organismo: OrganismoV159, duracion: float,
                       ruido_amplitud: float = 0.0, intervalo_ruido: float = 10.0,
                       setpoint_func=None, ritual_forzado: bool = None) -> Dict:
        dt = self.params.dt
        pasos = int(duracion / dt)
        
        if setpoint_func is None:
            def setpoint_default(t):
                periodo = 30.0
                fase = (t % periodo) / periodo
                return 40.0 if fase < 0.5 else -40.0
            setpoint_func = setpoint_default
        
        ruido_actual = 0.0
        tiempo_prox_ruido = intervalo_ruido
        
        for _ in range(pasos):
            t = organismo.tiempo_total
            
            if t >= tiempo_prox_ruido and ruido_amplitud > 0:
                ruido_actual = np.random.uniform(-ruido_amplitud, ruido_amplitud)
                tiempo_prox_ruido = t + intervalo_ruido
            
            setpoint = setpoint_func(t)
            organismo.step(setpoint, ruido_actual, modo_ritual_forzado=ritual_forzado)
        
        return {
            'error_rms': organismo.get_error_rms(10.0),
            'fatiga': organismo.fatiga,
            'historia_fisica': organismo.historia_fisica,
            'historia_intencional': organismo.historia_intencional,
            'ritual_activation': organismo.ritual_activation,
            'ritual_active': organismo.ritual_active
        }
    
    def ejecutar(self) -> Dict:
        print("=" * 100)
        print("EXPERIMENTO V159 — ANIMA-2 Etapa 3: RITUAL")
        print("=" * 100)
        print("  BASE: Parámetros exactos de V158")
        print("  CORRECCIÓN: Sin entrenamiento inicial (aprende durante F1-F4)")
        print()
        print("  Parámetros:")
        print(f"    K_GAIN = {self.params.K_GAIN}")
        print(f"    dt = {self.params.dt}s")
        print(f"    τ_ritual = {self.params.ritual.tau_ritual}s")
        print(f"    repetición_min = {self.params.ritual.repeticion_min}")
        print("=" * 100)
        print()
        
        print("  Creando organismos paralelos...")
        organismo_control = OrganismoV159("Control", self.params, self.seed, self.verbose)
        organismo_ritual = OrganismoV159("Ritual", self.params, self.seed, self.verbose)
        
        # NO hay entrenamiento inicial. Empiezan desde cero.
        
        # ====================================================================
        # F1: Baseline (aprendizaje inicial)
        # ====================================================================
        print("\n  F1: Baseline (3 ciclos, 90s) - RITUAL FORZADO OFF...")
        
        self.ejecutar_ciclo(organismo_control, duracion=90.0, ruido_amplitud=0.0,
                            ritual_forzado=False)
        self.ejecutar_ciclo(organismo_ritual, duracion=90.0, ruido_amplitud=0.0,
                            ritual_forzado=False)
        
        print(f"    Control post-baseline: fatiga={organismo_control.fatiga:.0f}°, hist_int={organismo_control.historia_intencional:.0f}°")
        print(f"    Ritual post-baseline: fatiga={organismo_ritual.fatiga:.0f}°, hist_int={organismo_ritual.historia_intencional:.0f}°")
        
        # ====================================================================
        # F2: Control - SIN ritual
        # ====================================================================
        print("\n  F2: Control - 20 ciclos SIN ritual...")
        
        for ciclo in range(20):
            self.ejecutar_ciclo(organismo_control, duracion=40.0,
                                ruido_amplitud=5.0, intervalo_ruido=10.0,
                                ritual_forzado=False)
            
            if (ciclo + 1) % 5 == 0:
                print(f"    Control ciclo {ciclo+1}/20, fatiga={organismo_control.fatiga:.0f}°, "
                      f"hist_int={organismo_control.historia_intencional:.0f}°")
        
        # ====================================================================
        # F3: Experimental - CON ritual
        # ====================================================================
        print("\n  F3: Experimental - 20 ciclos CON ritual...")
        
        episodios_ritual = 0
        for ciclo in range(20):
            self.ejecutar_ciclo(organismo_ritual, duracion=40.0,
                                ruido_amplitud=5.0, intervalo_ruido=10.0,
                                ritual_forzado=None)
            
            if organismo_ritual.ritual_active:
                episodios_ritual += 1
            
            if (ciclo + 1) % 5 == 0:
                print(f"    Ritual ciclo {ciclo+1}/20, fatiga={organismo_ritual.fatiga:.0f}°, "
                      f"hist_int={organismo_ritual.historia_intencional:.0f}°, "
                      f"ritual_act={organismo_ritual.ritual_activation:.3f}")
        
        print(f"\n    Episodios con ritual activo: {episodios_ritual}/20")
        
        # ====================================================================
        # F4: Test post
        # ====================================================================
        print("\n  F4: Test post (3 ciclos) - RITUAL FORZADO OFF...")
        
        for _ in range(3):
            self.ejecutar_ciclo(organismo_control, duracion=40.0,
                                ruido_amplitud=5.0, intervalo_ruido=10.0,
                                ritual_forzado=False)
        
        for _ in range(3):
            self.ejecutar_ciclo(organismo_ritual, duracion=40.0,
                                ruido_amplitud=5.0, intervalo_ruido=10.0,
                                ritual_forzado=False)
        
        # ====================================================================
        # Resultados
        # ====================================================================
        
        error_control = organismo_control.get_error_rms(10.0)
        error_ritual = organismo_ritual.get_error_rms(10.0)
        mejora = (error_control - error_ritual) / error_control if error_control > 0 else 0
        
        ratio_int_fis = (organismo_ritual.historia_intencional / 
                         max(1, organismo_ritual.historia_fisica))
        
        ahorro_fatiga = ((organismo_control.fatiga - organismo_ritual.fatiga) / 
                         max(1, organismo_control.fatiga))
        
        mejora_ok = mejora > 0.15
        ritual_act_ok = 0.5 < organismo_ritual.ritual_activation < 1.5
        ratio_ok = ratio_int_fis > 1.5
        ahorro_ok = ahorro_fatiga > 0.05
        
        exito = (mejora_ok + ritual_act_ok + ratio_ok + ahorro_ok) >= 3
        
        print("\n" + "=" * 80)
        print("RESULTADOS V159 — Ritual")
        print("=" * 80)
        print(f"\n  Tiempo total simulado: {organismo_control.tiempo_total:.0f}s")
        print(f"\n  Historia total Control: {organismo_control.historia_intencional:.0f}°")
        print(f"  Historia total Ritual:  {organismo_ritual.historia_intencional:.0f}°")
        print(f"\n  Resultados post-entrenamiento (F4):")
        print(f"    Error RMS Control: {error_control:.1f}°")
        print(f"    Error RMS Ritual:  {error_ritual:.1f}°")
        print(f"    Mejora: {mejora*100:.1f}% {'✅' if mejora_ok else '❌'}")
        print(f"    Fatiga Control: {organismo_control.fatiga:.0f}°")
        print(f"    Fatiga Ritual:  {organismo_ritual.fatiga:.0f}°")
        print(f"    Ahorro fatiga: {ahorro_fatiga*100:.1f}% {'✅' if ahorro_ok else '❌'}")
        print(f"    Ratio int/fis: {ratio_int_fis:.2f} {'✅' if ratio_ok else '❌'}")
        print(f"    Activación ritual final: {organismo_ritual.ritual_activation:.3f} {'✅' if ritual_act_ok else '❌'}")
        
        print("\n" + "=" * 80)
        if exito:
            print("  ✅ ETAPA 3 COMPLETADA — RITUAL VALIDADO")
        else:
            print("  ⚠️ ETAPA 3 PARCIAL")
        print("=" * 80)
        
        # Gráfico
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        axes[0, 0].plot(organismo_control.history['t'], organismo_control.history['Cb'], label='Control')
        axes[0, 0].plot(organismo_ritual.history['t'], organismo_ritual.history['Cb'], label='Ritual')
        axes[0, 0].set_xlabel('Tiempo (s)')
        axes[0, 0].set_ylabel('Cb')
        axes[0, 0].set_title('Consciencia Básica')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(organismo_ritual.history['t'], organismo_ritual.history['ritual_activation'], color='purple')
        axes[0, 1].axhline(y=0.7, color='red', linestyle='--', label='Umbral')
        axes[0, 1].set_xlabel('Tiempo (s)')
        axes[0, 1].set_ylabel('Activación')
        axes[0, 1].set_title('Activación Ritual')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(organismo_control.history['t'], organismo_control.history['fatiga'], label='Control')
        axes[1, 0].plot(organismo_ritual.history['t'], organismo_ritual.history['fatiga'], label='Ritual')
        axes[1, 0].set_xlabel('Tiempo (s)')
        axes[1, 0].set_ylabel('Fatiga (°)')
        axes[1, 0].set_title('Fatiga')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(organismo_control.history['t'], organismo_control.history['error'], label='Control', alpha=0.5)
        axes[1, 1].plot(organismo_ritual.history['t'], organismo_ritual.history['error'], label='Ritual', alpha=0.5)
        axes[1, 1].set_xlabel('Tiempo (s)')
        axes[1, 1].set_ylabel('Error (°)')
        axes[1, 1].set_title('Error')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.log_dir}/v159_ritual_{self.timestamp}.png", dpi=150)
        plt.close()
        
        print(f"\n  📊 Gráficos guardados: {self.log_dir}/v159_ritual_{self.timestamp}.png")
        
        return {'exito': exito, 'mejora': mejora, 'hist_control': organismo_control.historia_intencional, 'hist_ritual': organismo_ritual.historia_intencional}
    
    def run(self):
        start = time.time()
        resultado = self.ejecutar()
        print(f"\n  ⏱️ Tiempo de ejecución: {time.time() - start:.1f} segundos")
        return resultado


if __name__ == "__main__":
    print("\n" + "=" * 100)
    print("V159 — ANIMA-2 Etapa 3: RITUAL (sin entrenamiento previo)")
    print("=" * 100 + "\n")
    
    experimento = ExperimentoV159(seed=42, verbose=True)
    resultados = experimento.run()
    
    print("\n" + "=" * 100)
    print(f"V159 COMPLETADO — Éxito: {resultados['exito']}")
    print(f"Historia Control: {resultados['hist_control']:.0f}°")
    print(f"Historia Ritual:  {resultados['hist_ritual']:.0f}°")
    print("=" * 100)