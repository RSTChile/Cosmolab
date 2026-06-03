#!/usr/bin/env python3
"""
EXPERIMENTO V159 — ANIMA-2 Etapa 3: RITUAL
================================================================================
Objetivo: Fijar marcos conductuales mediante repetición enactuada

Parámetros:
  - τ_ritual = 120s (decaimiento de memoria ritual)
  - repetición_min = 3 ciclos idénticos para activar ritual
  - ritual_activation = ∫(Cb * repetición) / τ_ritual dt
  - Meta: crear expectativa temporal rígida (patrón fijo cada 30s)

Precondiciones:
  - Etapa 2 completada (Juego enactuado validado en V158)
  - Cb operativa, memoria de ausencia activa
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
# PARÁMETROS GLOBALES
# ============================================================================

@dataclass
class ParamsRitual:
    """Parámetros específicos de Etapa 3 - RITUAL"""
    tau_ritual: float = 120.0          # Constante de decaimiento ritual (segundos)
    repeticion_min: int = 3            # Ciclos idénticos mínimos para activar ritual
    ritual_gain: float = 0.05          # Ganancia de influencia ritual sobre acción
    patron_temporal: float = 30.0      # Intervalo esperado entre patrones (segundos)
    tolerancia_patron: float = 0.3     # 30% de tolerancia en timing (±9s)
    umbral_ritual_activacion: float = 0.7  # Ritual activation > 0.7 → influye


@dataclass
class ParamsCosmosemiotica:
    """Parámetros base del sistema (heredados de V158)"""
    dt: float = 0.1                    # Paso temporal (segundos)
    K_GAIN: float = 0.00003            # Ganancia de fatiga
    Kp_base: float = 0.002             # Ganancia proporcional base
    
    # Consciencia básica (Cb)
    tau_cb: float = 10.0               # Constante de tiempo Cb
    umbral_cb_baseline: float = 35.0   # Umbral de referencia
    
    # Memoria de ausencia
    tau_mem_base: float = 30.0         # Base de memoria
    k_mem: float = 0.05                # Influencia de historia
    
    # Juego (de V158)
    K_COG: float = 0.5                 # Costo cognitivo
    lambda_fisico: float = 0.1         # Amortiguación física
    lambda_costo_motor: float = 0.3    # Costo motor reducido
    
    # Ritual (nuevo)
    ritual: ParamsRitual = field(default_factory=ParamsRitual)
    
    # Límites
    FATIGA_MAX: float = 100000.0
    ORIENT_MIN: float = -60.0
    ORIENT_MAX: float = 60.0


# ============================================================================
# ORGANISMO CON RITUAL
# ============================================================================

class OrganismoRitual:
    """
    ANIMA-2 con capacidad ritual (Etapa 3)
    Extiende el organismo de V158 añadiendo:
      - Memoria de patrones temporales
      - Activación ritual por repetición
      - Influencia ritual sobre acción
    """
    
    def __init__(self, name: str, params: ParamsCosmosemiotica, seed: int = None):
        self.name = name
        self.p = params
        
        if seed is not None:
            np.random.seed(seed)
        
        # Estado físico
        self.orientacion = 0.0          # Orientación actual (°)
        self.fatiga = 0.0               # Fatiga acumulada
        self.historia_fisica = 0.0      # Desplazamiento físico real
        self.historia_intencional = 0.0 # Desplazamiento intencional (raw)
        
        # Estado cognitivo
        self.Cb = 0.0                   # Consciencia básica
        self.setpoint_recordado = 0.0   # Memoria de ausencia
        self.confianza_memoria = 1.0    # Confianza en el setpoint recordado
        self.tau_mem_actual = params.tau_mem_base
        
        # Estado ritual
        self.ritual_activation = 0.0    # Nivel de activación ritual (0-1+)
        self.patron_buffer = []         # Buffer de últimos patrones temporales
        self.ultimo_patron_time = 0.0   # Tiempo del último patrón reconocido
        self.ritual_active = False      # Si el ritual está influyendo
        self.repeticiones_consecutivas = 0  # Conteo de patrones iguales
        
        # Métricas temporales
        self.tiempo_total = 0.0
        self.tiempo_juego = 0.0
        self.juego_active = False
        self.refractory_counter = 0
        self.episodios_juego = 0
        
        # Registro de historial
        self.history = {
            't': [], 'orientacion': [], 'Cb': [], 'fatiga': [],
            'ritual_activation': [], 'setpoint': [], 'error': []
        }
        
        # Patrón actual para detección de ritual
        self.patron_actual = None
        self.tiempo_ultimo_patron = 0.0
    
    def actualizar_memoria_ausencia(self, error: float, setpoint_objetivo: float):
        """Actualiza memoria de ausencia con decaimiento por historia"""
        self.confianza_memoria *= np.exp(-self.p.dt / self.tau_mem_actual)
        
        if abs(error) < 5.0:
            # Error bajo -> actualizar memoria
            alpha = 0.1
            self.setpoint_recordado = (alpha * setpoint_objetivo + 
                                        (1-alpha) * self.setpoint_recordado)
            self.confianza_memoria = min(1.0, self.confianza_memoria + 0.05)
        else:
            # Error alto -> degradar memoria
            self.confianza_memoria *= 0.995
        
        # Actualizar tau_mem según historia
        self.tau_mem_actual = min(100.0, self.p.tau_mem_base + 
                                   self.p.k_mem * self.historia_intencional / 100)
    
    def actualizar_consciencia(self, error: float, as_sys_env: float = 0.0):
        """Actualiza Consciencia Básica (Cb)"""
        # A_sys-env: desacople sistema-entorno
        entrada_cb = error * (1.0 - as_sys_env)
        
        # Ecuación diferencial: dCb/dt = entrada - Cb/τ
        dCb = (entrada_cb - self.Cb / self.p.tau_cb) * self.p.dt
        self.Cb += dCb
        self.Cb = max(0.0, self.Cb)
    
    def detectar_patron(self, delta_intencional: float, delta_tiempo: float) -> bool:
        """
        Detecta si la acción actual forma parte de un patrón repetitivo.
        Un patrón se define por:
          - Magnitud similar (dentro de 30%)
          - Temporalidad similar (cada ~30s)
        """
        if abs(delta_intencional) < 0.5:
            return False
        
        # Normalizar signo (importa dirección)
        direccion = np.sign(delta_intencional)
        magnitud = abs(delta_intencional)
        
        # Buscar en buffer patrones similares
        for t_prev, mag_prev, dir_prev in self.patron_buffer:
            dt_desde_prev = self.tiempo_total - t_prev
            
            # Verificar timing esperado
            timing_ok = abs(dt_desde_prev - self.p.ritual.patron_temporal) <= (
                self.p.ritual.patron_temporal * self.p.ritual.tolerancia_patron
            )
            
            # Verificar magnitud similar (±30%)
            magnitud_ok = abs(magnitud - mag_prev) / max(magnitud, mag_prev) < 0.3
            
            # Verificar misma dirección
            direccion_ok = dir_prev == direccion
            
            if timing_ok and magnitud_ok and direccion_ok:
                return True
        
        return False
    
    def actualizar_ritual(self, delta_intencional: float):
        """
        Actualiza activación ritual basada en repetición de patrones.
        
        Ritual_activation = ∫(Cb * repetición) / τ_ritual dt
        """
        # Detectar patrón en acción actual
        es_patron = self.detectar_patron(delta_intencional, self.p.dt)
        
        if es_patron:
            # Encontrar patrón consecutivo
            self.repeticiones_consecutivas += 1
            
            # Si alcanza umbral mínimo de repeticiones, activar ritual
            if self.repeticiones_consecutivas >= self.p.ritual.repeticion_min:
                # Ritual activation aumenta con Cb y repeticiones
                incremento = self.Cb * self.repeticiones_consecutivas / 100.0
                self.ritual_activation += incremento * self.p.dt
        else:
            # Ruptura de patrón: decae repetición pero no necesariamente ritual_activation
            self.repeticiones_consecutivas = max(0, self.repeticiones_consecutivas - 1)
        
        # Decaimiento natural de ritual_activation
        self.ritual_activation *= np.exp(-self.p.dt / self.p.ritual.tau_ritual)
        self.ritual_activation = max(0.0, min(2.0, self.ritual_activation))
        
        # Determinar si ritual está activo (influye en acción)
        self.ritual_active = (self.ritual_activation > 
                               self.p.ritual.umbral_ritual_activacion)
        
        # Registrar patrón actual en buffer (mantener últimos 10)
        if abs(delta_intencional) > 0.5:
            self.patron_buffer.append((self.tiempo_total, abs(delta_intencional),
                                        np.sign(delta_intencional)))
            if len(self.patron_buffer) > 10:
                self.patron_buffer.pop(0)
    
    def calcular_correccion(self, error: float, setpoint: float) -> Tuple[float, float]:
        """
        Calcula corrección enactuada con influencia ritual.
        
        Retorna: (delta_intencional, delta_fisico)
        """
        # Error derivativo para anticipación
        if not hasattr(self, '_error_prev'):
            self._error_prev = error
        error_deriv = (error - self._error_prev) / self.p.dt
        self._error_prev = error
        
        # Corrección base proporcional-derivativa
        correccion_base = (self.p.Kp_base * error + 
                          0.001 * error_deriv)  # K_derivativa pequeña
        
        # INFLUENCIA RITUAL
        # El ritual fuerza repetición del patrón cuando está activo
        if self.ritual_active and self.repeticiones_consecutivas >= 1:
            # Obtener última magnitud y dirección del patrón
            if self.patron_buffer:
                _, ultima_mag, ultima_dir = self.patron_buffer[-1]
                # Forzar corrección hacia el patrón ritual
                correccion_ritual = ultima_dir * ultima_mag * self.p.ritual.ritual_gain
                # El ritual modula la corrección base
                correccion = (correccion_base * (1 - self.ritual_activation * 0.3) + 
                              correccion_ritual * self.ritual_activation)
            else:
                correccion = correccion_base
        else:
            correccion = correccion_base
        
        # Limitar corrección
        delta_max = 5.0 * self.p.dt
        delta_intencional = np.clip(correccion, -delta_max, delta_max)
        
        # Modo juego (heredado de V158) - episódico
        # El ritual puede inhibir juego cuando está activo
        juego_activado = False
        if not self.ritual_active:  # Ritual inhibe juego (prioridad ritual)
            # Umbral adaptativo de Cb (percentil 70 de historial reciente)
            if len(self.history['Cb']) > 200:
                umbral_dinamico = np.percentile(self.history['Cb'][-200:], 70)
            else:
                umbral_dinamico = self.p.umbral_cb_baseline
            
            if self.Cb > umbral_dinamico and self.refractory_counter <= 0:
                juego_activado = True
                self.episodios_juego += 1
        
        if juego_activado and not self.ritual_active:
            # MODO JUEGO: física amortiguada, costo cognitivo
            self.juego_active = True
            self.tiempo_juego += self.p.dt
            
            delta_fisico = delta_intencional * self.p.lambda_fisico
            costo_motor = abs(delta_intencional) * self.p.lambda_costo_motor
            costo_cognitivo = self.p.K_COG * self.Cb * self.p.dt
            costo_total = costo_motor + costo_cognitivo
        else:
            # MODO SERIO
            if self.refractory_counter <= 0:
                self.juego_active = False
            
            delta_fisico = delta_intencional
            costo_total = abs(delta_intencional)
        
        # Actualizar refractory period (post-juego)
        if not juego_activado and self.refractory_counter > 0:
            self.refractory_counter -= self.p.dt
        
        # Registrar historia dual
        self.historia_intencional += abs(delta_intencional)
        self.historia_fisica += abs(delta_fisico)
        
        # Actualizar fatiga
        self.fatiga += costo_total * self.p.K_GAIN
        self.fatiga = min(self.fatiga, self.p.FATIGA_MAX)
        
        return delta_intencional, delta_fisico
    
    def step(self, setpoint_externo: float, ruido: float = 0.0, 
             modo_ritual_forzado: bool = None) -> float:
        """
        Avanza un paso temporal del organismo.
        
        Args:
            setpoint_externo: Objetivo de orientación (°)
            ruido: Ruido aplicado al setpoint (simula perturbación)
            modo_ritual_forzado: None=normal, True=forzar ritual activo,
                                 False=forzar ritual inactivo
        
        Returns:
            error_actual: Error después de corrección (°)
        """
        self.tiempo_total += self.p.dt
        
        # Aplicar ruido al setpoint si se especifica
        setpoint_efectivo = setpoint_externo + ruido
        
        # Actualizar memoria de ausencia
        error_antes = setpoint_efectivo - self.orientacion
        self.actualizar_memoria_ausencia(error_antes, setpoint_efectivo)
        
        # Si confianza es baja, usar setpoint_recordado
        if self.confianza_memoria < 0.3:
            setpoint_objetivo = self.setpoint_recordado
        else:
            setpoint_objetivo = setpoint_efectivo
        
        error = setpoint_objetivo - self.orientacion
        
        # Forzar modo ritual si se especifica (para debugging)
        if modo_ritual_forzado is not None:
            self.ritual_active = modo_ritual_forzado
        
        # Calcular corrección enactuada
        delta_intencional, delta_fisico = self.calcular_correccion(error, setpoint_objetivo)
        
        # Actualizar ritual basado en intención
        self.actualizar_ritual(delta_intencional)
        
        # Actualizar consciencia básica (error como desacople)
        self.actualizar_consciencia(abs(error), 0.0)
        
        # Aplicar movimiento
        self.orientacion += delta_fisico
        self.orientacion = np.clip(self.orientacion, self.p.ORIENT_MIN, self.p.ORIENT_MAX)
        
        # Registrar historial
        self.history['t'].append(self.tiempo_total)
        self.history['orientacion'].append(self.orientacion)
        self.history['Cb'].append(self.Cb)
        self.history['fatiga'].append(self.fatiga)
        self.history['ritual_activation'].append(self.ritual_activation)
        self.history['setpoint'].append(setpoint_objetivo)
        self.history['error'].append(error)
        
        return error
    
    def get_error_rms(self, segundos_ultimos: float = 10.0) -> float:
        """Calcula RMS del error en los últimos N segundos"""
        if len(self.history['t']) < 2:
            return 0.0
        
        t_max = self.tiempo_total
        t_min = max(0, t_max - segundos_ultimos)
        
        errores = []
        for t, err in zip(self.history['t'], self.history['error']):
            if t >= t_min:
                errores.append(err)
        
        if not errores:
            return 0.0
        return np.sqrt(np.mean(np.square(errores)))
    
    def reset(self):
        """Resetea el organismo a estado inicial"""
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
        self.patron_actual = None
        self._error_prev = 0.0
        
        for key in self.history:
            self.history[key] = []
    
    def print_status(self):
        """Imprime estado actual del organismo"""
        print(f"  [{self.name}] t={self.tiempo_total:.1f}s | "
              f"orient={self.orientacion:.1f}° | "
              f"Cb={self.Cb:.1f} | "
              f"ritual_act={self.ritual_activation:.2f} | "
              f"ritual_active={self.ritual_active} | "
              f"fatiga={self.fatiga:.0f}° | "
              f"hist_int={self.historia_intencional:.0f}° | "
              f"juego_t={self.tiempo_juego:.0f}s")


# ============================================================================
# EXPERIMENTO V159
# ============================================================================

class ExperimentoV159:
    """
    Experimento V159: Validación de Etapa 3 - RITUAL
    
    Estructura:
      F1: Baseline (3 ciclos) - sin ritual, sin juego
      F2: Control (20 ciclos) - sin ritual
      F3: Experimental (20 ciclos) - CON ritual
      F4: Test post (3 ciclos) - ambos sin ritual
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.params = ParamsCosmosemiotica()
        self.resultados = {}
        
        # Crear directorio de logs
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = f"v159_logs"
        os.makedirs(self.log_dir, exist_ok=True)
    
    def ejecutar_ciclo(self, organismo: OrganismoRitual, duracion: float,
                       ruido_amplitud: float = 0.0, intervalo_ruido: float = 10.0,
                       setpoint_func=None, ritual_forzado: bool = None) -> Dict:
        """
        Ejecuta un ciclo de entrenamiento/test.
        
        Args:
            organismo: Organismo a ejecutar
            duracion: Duración en segundos
            ruido_amplitud: Amplitud del ruido (± grados)
            intervalo_ruido: Intervalo entre cambios de ruido
            setpoint_func: Función que genera setpoint(t)
            ritual_forzado: Forzar ritual activo/inactivo (None=normal)
        """
        dt = self.params.dt
        pasos = int(duracion / dt)
        
        # Setpoint por defecto: onda cuadrada entre -40 y 40 cada 15s
        if setpoint_func is None:
            def setpoint_default(t):
                periodo = 30.0
                fase = (t % periodo) / periodo
                return 40.0 if fase < 0.5 else -40.0
            setpoint_func = setpoint_default
        
        # Ruido por defecto
        ruido_actual = 0.0
        tiempo_prox_ruido = intervalo_ruido
        
        for paso in range(pasos):
            t = organismo.tiempo_total
            
            # Actualizar ruido periódico
            if t >= tiempo_prox_ruido and ruido_amplitud > 0:
                ruido_actual = np.random.uniform(-ruido_amplitud, ruido_amplitud)
                tiempo_prox_ruido = t + intervalo_ruido
            
            setpoint = setpoint_func(t)
            
            # Paso del organismo
            organismo.step(setpoint, ruido_actual, ritual_forzado)
        
        # Recolectar métricas
        return {
            'error_rms': organismo.get_error_rms(10.0),
            'fatiga': organismo.fatiga,
            'historia_fisica': organismo.historia_fisica,
            'historia_intencional': organismo.historia_intencional,
            'tiempo_juego': organismo.tiempo_juego,
            'episodios_juego': organismo.episodios_juego,
            'ritual_activation_final': organismo.ritual_activation,
            'ritual_active_final': organismo.ritual_active,
            'repeticiones': organismo.repeticiones_consecutivas,
            'Cb_final': organismo.Cb
        }
    
    def entrenar_lateralidad(self, organismo: OrganismoRitual, repeticiones: int = 10):
        """Entrenamiento inicial de lateralidad"""
        print("  Entrenando lateralidad...")
        for i in range(repeticiones):
            # Movimiento hacia derecha
            for _ in range(50):
                organismo.step(30.0, 0.0, ritual_forzado=False)
            # Movimiento hacia izquierda
            for _ in range(50):
                organismo.step(-30.0, 0.0, ritual_forzado=False)
        print("  Entrenamiento completado.")
    
    def ejecutar(self) -> Dict:
        """Ejecuta el experimento V159 completo"""
        print("=" * 100)
        print("EXPERIMENTO V159 — ANIMA-2 Etapa 3: RITUAL")
        print("=" * 100)
        print("  Objetivo: Fijar marcos conductuales mediante repetición enactuada")
        print()
        print("  Parámetros:")
        print(f"    τ_ritual = {self.params.ritual.tau_ritual}s")
        print(f"    repetición_min = {self.params.ritual.repeticion_min}")
        print(f"    patron_temporal = {self.params.ritual.patron_temporal}s")
        print(f"    umbral_ritual = {self.params.ritual.umbral_ritual_activacion}")
        print("=" * 100)
        print()
        
        # Crear dos organismos paralelos (misma semilla)
        print("  Creando organismos paralelos...")
        organismo_control = OrganismoRitual("Control", self.params, self.seed)
        organismo_ritual = OrganismoRitual("Ritual", self.params, self.seed)
        
        # Entrenamiento inicial
        self.entrenar_lateralidad(organismo_control)
        self.entrenar_lateralidad(organismo_ritual)
        
        # ====================================================================
        # F1: Baseline (sin ritual, sin juego)
        # ====================================================================
        print("\n  F1: Baseline (3 ciclos) - RITUAL FORZADO OFF...")
        
        # Reset ambos
        organismo_control.reset()
        organismo_ritual.reset()
        
        # Forzar ritual inactivo en baseline
        baseline_resultados = []
        for org in [organismo_control, organismo_ritual]:
            self.ejecutar_ciclo(org, duracion=90.0, ruido_amplitud=0.0,
                                ritual_forzado=False)
            baseline_resultados.append({
                'error_rms': org.get_error_rms(10.0),
                'fatiga': org.fatiga
            })
        
        print(f"    Control error RMS: {baseline_resultados[0]['error_rms']:.1f}°")
        print(f"    Ritual error RMS: {baseline_resultados[1]['error_rms']:.1f}°")
        
        # ====================================================================
        # F2: Control - 20 ciclos SIN ritual
        # ====================================================================
        print("\n  F2: Control - 20 ciclos SIN ritual...")
        
        organismo_control.reset()
        self.entrenar_lateralidad(organismo_control)
        
        # Ruido presente para forzar adaptación
        for ciclo in range(20):
            self.ejecutar_ciclo(organismo_control, duracion=40.0,
                                ruido_amplitud=5.0, intervalo_ruido=10.0,
                                ritual_forzado=False)  # Ritual siempre OFF
            
            if (ciclo + 1) % 5 == 0:
                print(f"    Control ciclo {ciclo+1}/20, fatiga={organismo_control.fatiga:.0f}°, "
                      f"hist_int={organismo_control.historia_intencional:.0f}°, "
                      f"ritual_act={organismo_control.ritual_activation:.2f}")
        
        # ====================================================================
        # F3: Experimental - 20 ciclos CON ritual
        # ====================================================================
        print("\n  F3: Experimental - 20 ciclos CON ritual...")
        
        organismo_ritual.reset()
        self.entrenar_lateralidad(organismo_ritual)
        
        # Ruido presente, ritual NATURAL (no forzado)
        episodios_ritual = 0
        for ciclo in range(20):
            self.ejecutar_ciclo(organismo_ritual, duracion=40.0,
                                ruido_amplitud=5.0, intervalo_ruido=10.0,
                                ritual_forzado=None)  # Ritual natural
            
            if organismo_ritual.ritual_active:
                episodios_ritual += 1
            
            if (ciclo + 1) % 5 == 0:
                print(f"    Ritual ciclo {ciclo+1}/20, fatiga={organismo_ritual.fatiga:.0f}°, "
                      f"hist_int={organismo_ritual.historia_intencional:.0f}°, "
                      f"ritual_act={organismo_ritual.ritual_activation:.2f}, "
                      f"ritual_active={organismo_ritual.ritual_active}")
        
        print(f"\n    Episodios con ritual activo: {episodios_ritual}/20")
        
        # ====================================================================
        # F4: Test post - ambos SIN ritual
        # ====================================================================
        print("\n  F4: Test post (3 ciclos) - RITUAL FORZADO OFF...")
        
        # Forzar ritual inactivo en ambos para test justo
        post_resultados = []
        for org, name in [(organismo_control, "Control"), (organismo_ritual, "Ritual")]:
            # Ejecutar 3 ciclos de test sin ritual
            for _ in range(3):
                self.ejecutar_ciclo(org, duracion=40.0,
                                    ruido_amplitud=5.0, intervalo_ruido=10.0,
                                    ritual_forzado=False)
            
            post_resultados.append({
                'error_rms': org.get_error_rms(10.0),
                'fatiga': org.fatiga,
                'historia_fisica': org.historia_fisica,
                'historia_intencional': org.historia_intencional,
                'tiempo_juego': org.tiempo_juego,
                'ritual_activation_final': org.ritual_activation
            })
            print(f"    {name} error RMS post: {post_resultados[-1]['error_rms']:.1f}°, "
                  f"hist_int={org.historia_intencional:.0f}°")
        
        # ====================================================================
        # Procesar resultados
        # ====================================================================
        
        control_post = post_resultados[0]
        ritual_post = post_resultados[1]
        
        # Métricas de exaptación ritual
        tiempo_total = 20 * 40.0  # 20 ciclos * 40s = 800s
        tiempo_ritual_activo = organismo_ritual.ritual_activation * tiempo_total / 2.0  # Aprox
        
        resultados = {
            'exito': False,
            'control': {
                'error_rms_post': control_post['error_rms'],
                'fatiga_final': control_post['fatiga'],
                'historia_intencional': control_post['historia_intencional']
            },
            'ritual': {
                'error_rms_post': ritual_post['error_rms'],
                'fatiga_final': ritual_post['fatiga'],
                'historia_intencional': ritual_post['historia_intencional'],
                'historia_fisica': ritual_post['historia_fisica'],
                'ritual_activation_final': ritual_post['ritual_activation_final'],
                'tiempo_juego': ritual_post['tiempo_juego']
            },
            'criterios': {}
        }
        
        # Criterio 1: Mejora post-ritual (error reducido)
        mejora = (resultados['control']['error_rms_post'] - 
                  resultados['ritual']['error_rms_post']) / resultados['control']['error_rms_post']
        resultados['criterios']['mejora_error'] = mejora
        resultados['criterios']['mejora_error_ok'] = mejora > 0.15  # 15% mejora
        
        # Criterio 2: Activación ritual durante entrenamiento
        resultados['criterios']['ritual_activation'] = organismo_ritual.ritual_activation
        resultados['criterios']['ritual_activation_ok'] = (0.5 < organismo_ritual.ritual_activation < 1.5)
        
        # Criterio 3: Historia intencional vs física (exaptación ritual)
        if ritual_post['historia_intencional'] > 0:
            ratio_ritual = ritual_post['historia_intencional'] / max(1, ritual_post['historia_fisica'])
        else:
            ratio_ritual = 0
        resultados['criterios']['ratio_int_fis'] = ratio_ritual
        resultados['criterios']['ratio_ok'] = ratio_ritual > 1.5  # Más intención que física
        
        # Criterio 4: Fatiga menor en ritual
        ahorro_fatiga = (resultados['control']['fatiga_final'] - 
                         resultados['ritual']['fatiga_final']) / max(1, resultados['control']['fatiga_final'])
        resultados['criterios']['ahorro_fatiga'] = ahorro_fatiga
        resultados['criterios']['ahorro_ok'] = ahorro_fatiga > 0.05  # 5% menos fatiga
        
        # Éxito general: cumplir al menos 3 de 4 criterios
        ok_count = sum([
            resultados['criterios']['mejora_error_ok'],
            resultados['criterios']['ritual_activation_ok'],
            resultados['criterios']['ratio_ok'],
            resultados['criterios']['ahorro_ok']
        ])
        resultados['exito'] = ok_count >= 3
        
        # ====================================================================
        # Generar gráficos
        # ====================================================================
        self.generar_graficos(organismo_control, organismo_ritual, resultados)
        
        # ====================================================================
        # Imprimir resultados
        # ====================================================================
        print("\n" + "=" * 80)
        print("RESULTADOS V159 — Ritual")
        print("=" * 80)
        print(f"\n  Baseline (F1):")
        print(f"    Control error RMS: {baseline_resultados[0]['error_rms']:.1f}°")
        print(f"    Ritual error RMS: {baseline_resultados[1]['error_rms']:.1f}°")
        
        print(f"\n  Resultados post-entrenamiento (F4):")
        print(f"    ┌─────────────────────┬─────────────┬─────────────┬─────────────┐")
        print(f"    │ Métrica             │ Control     │ Ritual      │ Mejora      │")
        print(f"    ├─────────────────────┼─────────────┼─────────────┼─────────────┤")
        print(f"    │ Error RMS (10s)     │ {resultados['control']['error_rms_post']:5.1f}°      │ {resultados['ritual']['error_rms_post']:5.1f}°      │ {mejora*100:5.1f}% {'✅' if mejora>0 else '❌'}    │")
        print(f"    │ Fatiga final        │ {resultados['control']['fatiga_final']:7.0f}°  │ {resultados['ritual']['fatiga_final']:7.0f}°  │ {ahorro_fatiga*100:5.1f}% {'✅' if ahorro_fatiga>0 else '❌'}    │")
        print(f"    │ Historia intencional│ {resultados['control']['historia_intencional']:7.0f}°  │ {resultados['ritual']['historia_intencional']:7.0f}°  │         │")
        print(f"    │ Ratio int/fis       │ 1.00        │ {ratio_ritual:5.2f}        │         │")
        print(f"    └─────────────────────┴─────────────┴─────────────┴─────────────┘")
        
        print(f"\n  Métricas ritual (F3):")
        print(f"    Activación ritual final: {organismo_ritual.ritual_activation:.3f}")
        print(f"    Episodios ritual activo: {episodios_ritual}/20")
        print(f"    Repeticiones consecutivas máx: {max(organismo_ritual.repeticiones_consecutivas, 0)}")
        
        print("\n" + "=" * 80)
        print("CRITERIOS DE ÉXITO V159")
        print("=" * 80)
        print(f"  1. Mejora error post-ritual > 15%: {mejora*100:.1f}% → {'✅' if resultados['criterios']['mejora_error_ok'] else '❌'}")
        print(f"  2. Activación ritual 0.5-1.5: {organismo_ritual.ritual_activation:.3f} → {'✅' if resultados['criterios']['ritual_activation_ok'] else '❌'}")
        print(f"  3. Ratio intencional/físico > 1.5: {ratio_ritual:.2f} → {'✅' if resultados['criterios']['ratio_ok'] else '❌'}")
        print(f"  4. Menor fatiga que control: {ahorro_fatiga*100:.1f}% → {'✅' if resultados['criterios']['ahorro_ok'] else '❌'}")
        
        print("\n" + "=" * 80)
        if resultados['exito']:
            print("  ✅ ETAPA 3 COMPLETADA — RITUAL VALIDADO")
            print("     El organismo desarrolló marcos conductuales por repetición enactuada.")
        else:
            print("  ⚠️ ETAPA 3 PARCIAL")
            print("     Ritual emergiendo pero criterios no cumplidos.")
        print("=" * 80)
        
        print(f"\n  📊 Gráficos guardados: {self.log_dir}/v159_ritual_{self.timestamp}.png")
        print(f"\n  ⏱️ Tiempo de ejecución: {time.time() - self.start_time:.1f} segundos")
        
        return resultados
    
    def generar_graficos(self, control: OrganismoRitual, ritual: OrganismoRitual, 
                         resultados: Dict):
        """Genera gráficos comparativos"""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # 1. Evolución de Cb
        ax = axes[0, 0]
        ax.plot(control.history['t'], control.history['Cb'], label='Control', alpha=0.8)
        ax.plot(ritual.history['t'], ritual.history['Cb'], label='Ritual', alpha=0.8)
        ax.axhline(y=35, color='gray', linestyle='--', alpha=0.5, label='Umbral Cb')
        ax.set_xlabel('Tiempo (s)')
        ax.set_ylabel('Consciencia Básica (Cb)')
        ax.set_title('Evolución de Cb')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Activación ritual
        ax = axes[0, 1]
        ax.plot(ritual.history['t'], ritual.history['ritual_activation'], 
                color='purple', label='Ritual Activation')
        ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, 
                   label='Umbral activación')
        ax.set_xlabel('Tiempo (s)')
        ax.set_ylabel('Activación Ritual')
        ax.set_title('Activación Ritual')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Error comparativo
        ax = axes[0, 2]
        ax.plot(control.history['t'], control.history['error'], label='Control', alpha=0.5)
        ax.plot(ritual.history['t'], ritual.history['error'], label='Ritual', alpha=0.5)
        ax.set_xlabel('Tiempo (s)')
        ax.set_ylabel('Error (°)')
        ax.set_title('Error de orientación')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Fatiga
        ax = axes[1, 0]
        ax.plot(control.history['t'], control.history['fatiga'], label='Control')
        ax.plot(ritual.history['t'], ritual.history['fatiga'], label='Ritual')
        ax.set_xlabel('Tiempo (s)')
        ax.set_ylabel('Fatiga (°)')
        ax.set_title('Acumulación de fatiga')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Historia intencional vs física (solo ritual)
        ax = axes[1, 1]
        if len(ritual.history['t']) > 0:
            # Calcular acumulados
            t_int = ritual.history['t']
            # Aproximar historias acumuladas
            ax.plot(t_int, [ritual.historia_intencional * t/t_int[-1] 
                           if t_int[-1] > 0 else 0 for t in t_int], 
                   label='Intencional', color='green')
            ax.plot(t_int, [ritual.historia_fisica * t/t_int[-1] 
                           if t_int[-1] > 0 else 0 for t in t_int], 
                   label='Física', color='orange')
        ax.set_xlabel('Tiempo (s)')
        ax.set_ylabel('Historia acumulada (°)')
        ax.set_title('Historia: Intencional vs Física (Ritual)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Barras de criterios
        ax = axes[1, 2]
        criterios = list(resultados['criterios'].keys())
        valores = [1.0 if resultados['criterios'][c] else 0.0 
                   for c in criterios if c.endswith('_ok')]
        nombres = [c.replace('_ok', '').replace('_', ' ') for c in criterios if c.endswith('_ok')]
        
        colores = ['green' if v == 1 else 'red' for v in valores]
        ax.bar(nombres, valores, color=colores)
        ax.set_ylim(0, 1.2)
        ax.set_ylabel('Cumplimiento')
        ax.set_title('Criterios de Éxito')
        ax.set_xticklabels(nombres, rotation=15, ha='right')
        
        plt.tight_layout()
        plt.savefig(f"{self.log_dir}/v159_ritual_{self.timestamp}.png", dpi=150)
        plt.close()
    
    def run(self):
        """Ejecuta el experimento y retorna resultados"""
        self.start_time = time.time()
        return self.ejecutar()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 100)
    print("V159 — ANIMA-2 Etapa 3: RITUAL")
    print("Fijación de marcos conductuales por repetición enactuada")
    print("=" * 100 + "\n")
    
    experimento = ExperimentoV159(seed=42)
    resultados = experimento.run()
    
    print("\n" + "=" * 100)
    print("V159 COMPLETADO")
    print(f"Éxito: {resultados['exito']}")
    print("=" * 100)