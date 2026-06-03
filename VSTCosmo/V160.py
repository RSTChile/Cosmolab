#!/usr/bin/env python3
"""
EXPERIMENTO V160 — ANIMA-2 Etapa 3: RITUAL CON MUNDO
================================================================================
CORRECCIONES CRÍTICAS:
  1. Setpoint alternante real: ±60° (restaurado)
  2. K_GAIN aumentado a 0.0008 (25x V159)
  3. Medición de rigidez en F4 (inversión de gradiente)
  4. Verificación de actuación (assert si no hay movimiento)
  5. Activación ritual solo bajo carga (Cb > umbral + patrón repetido)
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
# PARÁMETROS V160
# ============================================================================

@dataclass
class ParamsRitual:
    tau_ritual: float = 60.0            # Reducido (más fácil activar)
    repeticion_min: int = 3
    ritual_gain: float = 0.1
    patron_temporal: float = 30.0
    tolerancia_patron: float = 0.3
    umbral_ritual_activacion: float = 0.5   # Bajado
    umbral_cb_para_ritual: float = 28.0     # NUEVO: Cb mínima para activar


@dataclass
class ParamsCosmosemiotica:
    dt: float = 0.1
    K_GAIN: float = 0.0008              # AUMENTADO (25x V159)
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
# ORGANISMO V160
# ============================================================================

class OrganismoV160:
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
        self.secuencia_acciones = []     # Para detección de patrones
        
        self.tiempo_total = 0.0
        self.tiempo_juego = 0.0
        self.juego_active = False
        self.refractory_counter = 0
        self.episodios_juego = 0
        self._error_prev = 0.0
        
        self.history = {
            't': [], 'orientacion': [], 'Cb': [], 'fatiga': [],
            'ritual_activation': [], 'setpoint': [], 'error': [], 'delta_raw': []
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
    
    def detectar_patron_conductual(self) -> bool:
        """
        Detecta si la secuencia reciente de acciones forma un patrón repetitivo.
        Patrón: [mov_derecha, movimiento_izquierda, movimiento_derecha] con timing similar.
        """
        if len(self.secuencia_acciones) < 3:
            return False
        
        # Tomar últimos 3 movimientos significativos
        ultimos = self.secuencia_acciones[-3:]
        
        # Verificar alternancia derecha-izquierda-derecha (o izquierda-derecha-izquierda)
        patron_valido = (ultimos[0][0] > 0 and ultimos[1][0] < 0 and ultimos[2][0] > 0) or \
                        (ultimos[0][0] < 0 and ultimos[1][0] > 0 and ultimos[2][0] < 0)
        
        if not patron_valido:
            return False
        
        # Verificar timing similar (±30%)
        t0, t1, t2 = ultimos[0][1], ultimos[1][1], ultimos[2][1]
        dt1 = t1 - t0
        dt2 = t2 - t1
        
        timing_ok = abs(dt1 - dt2) / max(dt1, dt2, 0.1) < 0.3
        
        # Verificar magnitud similar (±30%)
        m0, m1, m2 = abs(ultimos[0][0]), abs(ultimos[1][0]), abs(ultimos[2][0])
        magnitud_ok = (abs(m0 - m1) / max(m0, m1, 0.1) < 0.3 and
                       abs(m1 - m2) / max(m1, m2, 0.1) < 0.3)
        
        return timing_ok and magnitud_ok
    
    def actualizar_ritual(self, delta_intencional: float):
        """Actualiza activación ritual con condición de carga (Cb > umbral)"""
        
        # Registrar acción significativa para detección de patrones
        if abs(delta_intencional) > 0.5:
            self.secuencia_acciones.append((delta_intencional, self.tiempo_total))
            if len(self.secuencia_acciones) > 10:
                self.secuencia_acciones.pop(0)
        
        # Detectar patrón conductual
        es_patron = self.detectar_patron_conductual()
        
        if es_patron and self.Cb > self.p.ritual.umbral_cb_para_ritual:
            self.repeticiones_consecutivas += 1
            
            if self.repeticiones_consecutivas >= self.p.ritual.repeticion_min:
                incremento = self.Cb * self.repeticiones_consecutivas / 100.0
                self.ritual_activation += incremento * self.p.dt
        else:
            self.repeticiones_consecutivas = max(0, self.repeticiones_consecutivas - 0.5)
        
        # Decaimiento natural
        self.ritual_activation *= np.exp(-self.p.dt / self.p.ritual.tau_ritual)
        self.ritual_activation = max(0.0, min(2.0, self.ritual_activation))
        
        self.ritual_active = (self.ritual_activation > 
                               self.p.ritual.umbral_ritual_activacion)
    
    def calcular_correccion(self, error: float) -> Tuple[float, float]:
        error_deriv = (error - self._error_prev) / self.p.dt
        self._error_prev = error
        
        correccion_base = self.p.Kp_base * error + 0.001 * error_deriv
        
        # Influencia ritual (modula la corrección cuando está activo)
        if self.ritual_active:
            # El ritual introduce rigidez: reduce adaptación
            correccion = correccion_base * 0.7  # 30% menos sensible
        else:
            correccion = correccion_base
        
        delta_max = 5.0 * self.p.dt
        delta_intencional = np.clip(correccion, -delta_max, delta_max)
        
        # Modo juego (inhibido por ritual)
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
        self.history['delta_raw'].append(delta_intencional)
        
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
    
    def hubo_movimiento(self) -> bool:
        """Verifica si hubo movimiento significativo durante la simulación"""
        return max(abs(np.array(self.history['delta_raw']))) > 0.01


# ============================================================================
# EXPERIMENTO V160
# ============================================================================

class ExperimentoV160:
    def __init__(self, seed: int = 42, verbose: bool = True):
        self.seed = seed
        self.verbose = verbose
        self.params = ParamsCosmosemiotica()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = f"v160_logs"
        os.makedirs(self.log_dir, exist_ok=True)
    
    def ejecutar_ciclo(self, organismo: OrganismoV160, duracion: float,
                       setpoint_func, ruido_amplitud: float = 5.0, 
                       intervalo_ruido: float = 10.0,
                       ritual_forzado: bool = None) -> Dict:
        """Ejecuta un ciclo con setpoint dinámico"""
        dt = self.params.dt
        pasos = int(duracion / dt)
        
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
        print("EXPERIMENTO V160 — ANIMA-2 Etapa 3: RITUAL CON MUNDO")
        print("=" * 100)
        print("  CORRECCIONES:")
        print("    ✓ Setpoint alternante ±60° (restaurado)")
        print("    ✓ K_GAIN = 0.0008 (25x V159)")
        print("    ✓ Medición de rigidez en F4 (inversión de gradiente)")
        print("    ✓ Activación ritual solo bajo carga (Cb > 28)")
        print()
        print("  Parámetros:")
        print(f"    K_GAIN = {self.params.K_GAIN}")
        print(f"    dt = {self.params.dt}s")
        print(f"    τ_ritual = {self.params.ritual.tau_ritual}s")
        print(f"    repetición_min = {self.params.ritual.repeticion_min}")
        print(f"    umbral_Cb_ritual = {self.params.ritual.umbral_cb_para_ritual}")
        print("=" * 100)
        print()
        
        print("  Creando organismos paralelos...")
        organismo_control = OrganismoV160("Control", self.params, self.seed, self.verbose)
        organismo_ritual = OrganismoV160("Ritual", self.params, self.seed, self.verbose)
        
        # Función setpoint alternante ±60° cada 40 segundos
        def setpoint_alternante(t):
            ciclo = int(t / 40.0) % 2
            return 60.0 if ciclo == 0 else -60.0
        
        # Función setpoint INVERTIDO para F4 (rigidez)
        def setpoint_invertido(t):
            ciclo = int(t / 40.0) % 2
            return -60.0 if ciclo == 0 else 60.0  # Invertido
        
        # ====================================================================
        # F1: Baseline (aprendizaje inicial, ritual OFF)
        # ====================================================================
        print("\n  F1: Baseline (3 ciclos, 120s) - RITUAL FORZADO OFF...")
        
        self.ejecutar_ciclo(organismo_control, duracion=120.0, setpoint_func=setpoint_alternante,
                            ritual_forzado=False)
        self.ejecutar_ciclo(organismo_ritual, duracion=120.0, setpoint_func=setpoint_alternante,
                            ritual_forzado=False)
        
        print(f"    Control: fatiga={organismo_control.fatiga:.0f}°, hist_int={organismo_control.historia_intencional:.0f}°")
        print(f"    Ritual: fatiga={organismo_ritual.fatiga:.0f}°, hist_int={organismo_ritual.historia_intencional:.0f}°")
        
        # Verificar movimiento
        assert organismo_control.hubo_movimiento(), "ERROR: Control no se movió en F1"
        assert organismo_ritual.hubo_movimiento(), "ERROR: Ritual no se movió en F1"
        
        # ====================================================================
        # F2: Control - SIN ritual (20 ciclos)
        # ====================================================================
        print("\n  F2: Control - 20 ciclos SIN ritual...")
        
        for ciclo in range(20):
            self.ejecutar_ciclo(organismo_control, duracion=40.0, setpoint_func=setpoint_alternante,
                                ritual_forzado=False)
            
            if (ciclo + 1) % 5 == 0:
                print(f"    Control ciclo {ciclo+1}/20, fatiga={organismo_control.fatiga:.0f}°, "
                      f"hist_int={organismo_control.historia_intencional:.0f}°")
        
        # ====================================================================
        # F3: Experimental - CON ritual (20 ciclos)
        # ====================================================================
        print("\n  F3: Experimental - 20 ciclos CON ritual...")
        
        episodios_ritual = 0
        for ciclo in range(20):
            self.ejecutar_ciclo(organismo_ritual, duracion=40.0, setpoint_func=setpoint_alternante,
                                ritual_forzado=None)
            
            if organismo_ritual.ritual_active:
                episodios_ritual += 1
            
            if (ciclo + 1) % 5 == 0:
                print(f"    Ritual ciclo {ciclo+1}/20, fatiga={organismo_ritual.fatiga:.0f}°, "
                      f"hist_int={organismo_ritual.historia_intencional:.0f}°, "
                      f"ritual_act={organismo_ritual.ritual_activation:.3f}")
        
        print(f"\n    Episodios con ritual activo: {episodios_ritual}/20")
        
        # ====================================================================
        # F4: Test post CON INVERSIÓN DE GRADIENTE (mide rigidez)
        # ====================================================================
        print("\n  F4: Test post (3 ciclos) - SETPOINT INVERTIDO, ritual FORZADO OFF...")
        
        # Guardar estado de ritual antes de forzar OFF
        ritual_estaba_activo = organismo_ritual.ritual_active
        
        for _ in range(3):
            self.ejecutar_ciclo(organismo_control, duracion=40.0, setpoint_func=setpoint_invertido,
                                ritual_forzado=False)
        
        for _ in range(3):
            self.ejecutar_ciclo(organismo_ritual, duracion=40.0, setpoint_func=setpoint_invertido,
                                ritual_forzado=False)
        
        # ====================================================================
        # Resultados
        # ====================================================================
        
        error_control = organismo_control.get_error_rms(10.0)
        error_ritual = organismo_ritual.get_error_rms(10.0)
        
        # Rigidez: error del ritual en entorno invertido vs control
        rigidez = error_ritual / max(error_control, 0.1)
        
        ratio_int_fis = (organismo_ritual.historia_intencional / 
                         max(1, organismo_ritual.historia_fisica))
        
        ahorro_fatiga = ((organismo_control.fatiga - organismo_ritual.fatiga) / 
                         max(1, organismo_control.fatiga))
        
        # Criterios V160
        activacion_ok = episodios_ritual >= 10  # 50% del tiempo
        rigidez_ok = rigidez > 1.5
        persistencia_ok = ritual_estaba_activo  # El ritual estaba activo antes de F4
        movimiento_ok = organismo_ritual.historia_intencional > 500
        
        exito = (activacion_ok and rigidez_ok and persistencia_ok and movimiento_ok)
        
        print("\n" + "=" * 80)
        print("RESULTADOS V160 — Ritual con mundo")
        print("=" * 80)
        print(f"\n  Tiempo total simulado: {organismo_control.tiempo_total:.0f}s")
        print(f"\n  Historia total Control: {organismo_control.historia_intencional:.0f}°")
        print(f"  Historia total Ritual:  {organismo_ritual.historia_intencional:.0f}°")
        print(f"\n  Resultados post-inversión (F4):")
        print(f"    Error RMS Control: {error_control:.1f}°")
        print(f"    Error RMS Ritual:  {error_ritual:.1f}°")
        print(f"    Rigidez (error_ritual/error_control): {rigidez:.2f} {'✅' if rigidez_ok else '❌'}")
        print(f"\n  Métricas ritual:")
        print(f"    Episodios activos: {episodios_ritual}/20 {'✅' if activacion_ok else '❌'}")
        print(f"    Ritual activo antes de F4: {ritual_estaba_activo} {'✅' if persistencia_ok else '❌'}")
        print(f"    Historia intencional > 500°: {organismo_ritual.historia_intencional:.0f}° {'✅' if movimiento_ok else '❌'}")
        print(f"    Ratio int/fis: {ratio_int_fis:.2f}")
        print(f"    Activación ritual final: {organismo_ritual.ritual_activation:.3f}")
        
        print("\n" + "=" * 80)
        if exito:
            print("  ✅ ETAPA 3 COMPLETADA — RITUAL VALIDADO")
            print("     El organismo desarrolló rigidez de marco conductual")
        else:
            print("  ⚠️ ETAPA 3 PARCIAL")
        print("=" * 80)
        
        # Gráficos
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # Cb
        axes[0, 0].plot(organismo_control.history['t'], organismo_control.history['Cb'], label='Control')
        axes[0, 0].plot(organismo_ritual.history['t'], organismo_ritual.history['Cb'], label='Ritual')
        axes[0, 0].axhline(y=28, color='red', linestyle='--', label='Umbral Cb ritual')
        axes[0, 0].set_xlabel('Tiempo (s)')
        axes[0, 0].set_ylabel('Cb')
        axes[0, 0].set_title('Consciencia Básica')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Ritual activation
        axes[0, 1].plot(organismo_ritual.history['t'], organismo_ritual.history['ritual_activation'], color='purple')
        axes[0, 1].axhline(y=0.5, color='red', linestyle='--', label='Umbral activación')
        axes[0, 1].set_xlabel('Tiempo (s)')
        axes[0, 1].set_ylabel('Activación')
        axes[0, 1].set_title('Activación Ritual')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Error
        axes[0, 2].plot(organismo_control.history['t'], organismo_control.history['error'], label='Control', alpha=0.5)
        axes[0, 2].plot(organismo_ritual.history['t'], organismo_ritual.history['error'], label='Ritual', alpha=0.5)
        # Marcar inicio de F4 (inversión)
        t_f4 = organismo_control.tiempo_total - 120
        axes[0, 2].axvline(x=t_f4, color='black', linestyle='--', label='Inicio F4 (inversión)')
        axes[0, 2].set_xlabel('Tiempo (s)')
        axes[0, 2].set_ylabel('Error (°)')
        axes[0, 2].set_title('Error de orientación')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Fatiga
        axes[1, 0].plot(organismo_control.history['t'], organismo_control.history['fatiga'], label='Control')
        axes[1, 0].plot(organismo_ritual.history['t'], organismo_ritual.history['fatiga'], label='Ritual')
        axes[1, 0].axvline(x=t_f4, color='black', linestyle='--')
        axes[1, 0].set_xlabel('Tiempo (s)')
        axes[1, 0].set_ylabel('Fatiga (°)')
        axes[1, 0].set_title('Fatiga')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Historia intencional acumulada
        hist_int_control = np.cumsum(organismo_control.history['delta_raw'])
        hist_int_ritual = np.cumsum(organismo_ritual.history['delta_raw'])
        axes[1, 1].plot(organismo_control.history['t'], np.abs(hist_int_control), label='Control')
        axes[1, 1].plot(organismo_ritual.history['t'], np.abs(hist_int_ritual), label='Ritual')
        axes[1, 1].axvline(x=t_f4, color='black', linestyle='--')
        axes[1, 1].set_xlabel('Tiempo (s)')
        axes[1, 1].set_ylabel('Historia intencional acumulada (°)')
        axes[1, 1].set_title('Historia Intencional')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Barras de criterios
        axes[1, 2].bar(['Activación', 'Rigidez', 'Persistencia', 'Movimiento'], 
                       [activacion_ok, rigidez_ok, persistencia_ok, movimiento_ok],
                       color=['green' if x else 'red' for x in [activacion_ok, rigidez_ok, persistencia_ok, movimiento_ok]])
        axes[1, 2].set_ylim(0, 1.2)
        axes[1, 2].set_ylabel('Cumplimiento')
        axes[1, 2].set_title('Criterios V160')
        
        plt.tight_layout()
        plt.savefig(f"{self.log_dir}/v160_ritual_{self.timestamp}.png", dpi=150)
        plt.close()
        
        print(f"\n  📊 Gráficos guardados: {self.log_dir}/v160_ritual_{self.timestamp}.png")
        
        return {
            'exito': exito,
            'episodios_ritual': episodios_ritual,
            'rigidez': rigidez,
            'historia_ritual': organismo_ritual.historia_intencional
        }
    
    def run(self):
        start = time.time()
        resultado = self.ejecutar()
        print(f"\n  ⏱️ Tiempo de ejecución: {time.time() - start:.1f} segundos")
        return resultado


if __name__ == "__main__":
    print("\n" + "=" * 100)
    print("V160 — ANIMA-2 Etapa 3: RITUAL CON MUNDO")
    print("=" * 100 + "\n")
    
    experimento = ExperimentoV160(seed=42, verbose=True)
    resultados = experimento.run()
    
    print("\n" + "=" * 100)
    print(f"V160 COMPLETADO — Éxito: {resultados['exito']}")
    print(f"Episodios ritual activo: {resultados['episodios_ritual']}/20")
    print(f"Rigidez (error ratio): {resultados['rigidez']:.2f}")
    print(f"Historia ritual: {resultados['historia_ritual']:.0f}°")
    print("=" * 100)