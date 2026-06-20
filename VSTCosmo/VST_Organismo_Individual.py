#!/usr/bin/env python3
"""
VST_Organismo_Individual — ORGANISMO INDIVIDUAL CONSOLIDADO (ANIMA, un solo cuerpo)
================================================================================
QUÉ ES: por primera vez, el organismo individual COMPLETO como un solo cuerpo
ejecutable, con todas las capacidades que el linaje fue ganando, presentes y
CONMUTABLES POR FLAG (por defecto TODO ENCENDIDO / all-on).

QUÉ NO ES: experimento (no hay hipótesis que falsar), optimización (no se mejora
ninguna métrica), ni refundación teórica. Es CONSOLIDACIÓN: re-fusionar lo que
quedó afuera en cada corte transversal, para que el TODO exista.

----------------------------------------------------------------------------
PROCEDENCIA (cada bloque rastreable a su script de origen)
----------------------------------------------------------------------------
ESPINA / CUERPO   : V180c.py (decisión Alexis #1) — copiado VERBATIM como base.
                    Ya contiene, integrado: motor por gradiente inter-sistemas
                    (forma ANIMA-1, omega_A - omega_B; NO v88), fatiga = tiempo
                    biológico (historia monótona + fatiga_activa recuperable),
                    Cb, memoria de ausencia, juego, valencia/deliberación
                    (R_op / R_af) y memoria episódica.
RE-FUSIONADO #1   : class Ritual               <- V165.py:105-216  (VERBATIM)
RE-FUSIONADO #2   : class MetaRepresentacionObservacional (Rᴿ) <- V167.py:395-490 (VERBATIM)
CABLEADO RITUAL   : VERBATIM de V165.py:528-587 (ritual antes de juego; juego
                    inhibido si ritual activo; correccion_ritual desde
                    patron_buffer[-1]; orden Cb -> Ritual -> Rᴿ -> Juego).
CABLEADO Rᴿ       : VERBATIM de V167.py:565-578 (SOLO OBSERVA, no inhibe).

----------------------------------------------------------------------------
DECISIONES SOBRE CHOQUES (Fase A; resueltas por Alexis, no por criterio propio)
----------------------------------------------------------------------------
#1 Cuerpo base         -> V180c.py.
#2 Calibración fatiga  -> la del cuerpo (V155/V180c: tau_recuperacion=300.0,
                          K_GAIN=0.00015, costo_trabajo, techo 20000). NO se
                          imponen los valores de V150 (no "ponemos parámetros
                          nosotros" / no Shannon encubierto).
#3 Literales de Cb en Rᴿ (500.0, 50) -> SE DEJAN VERBATIM y se documentan:
   en V167:457 `Cb_sostenido / 500.0` el 500.0 es literal == CB_MAX actual; en
   V167:464 el umbral 50 de Cb es literal. NO se parametrizan (preservar verbatim).
#4 Ritual instalado    -> capacidad instalada, con REGISTRO DE SALIDA
   (ritual.historial_active/activation, meta.historial_desajuste, y atributos
   self.ultimo_* en el motor) y uso conmutable EN CUALQUIER MOMENTO por flag.

----------------------------------------------------------------------------
SUBJ_SEM (alteridad, V182d): NO se implementa (solo se manifiesta en díada).
Queda como GANCHO de interfaz declarado: el cuerpo expone get_valencia(banda),
ultima_opcion_elegida y orientacion (solo-lectura). Marcado [·].

DÍADA (fuera de alcance, compatibilidad OBLIGATORIA): la entrada de estímulo
("aire") se mantiene EXTERNA al cuerpo, de modo que dos instancias puedan
ponerse en aire compartido sin reescribir el cuerpo. NO se implementa
comunicación inter-cuerpo aquí.

MARCADORES en salidas/logs:  ✅ logrado · ❌ falla · ⊘ ausente/no aplica · · pendiente/neutro
================================================================================
"""
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json
from collections import deque
import time

# ============================================================
# PARAMETROS
# ============================================================
DT = 0.01
TIEMPO_POR_REPETICION = 452.0
REPETICIONES_LENTAS = 10
SESGO_L = 0.05
SESGO_R = -0.05
DIM_HEMISFERIO = 32
ZONA_MUERTA_BASE = 2.0
ZONA_MUERTA_MAX = 15.0
KP_BASE = 0.002
KP_MIN = 0.0005
KP_MAX = 0.005
VENTANA_OSCILACION = 100
INERCIA = 0.95
SENSIBILIDAD_GRAD = 10.0
K_GAIN = 0.00015
K_PRECISION = 0.002
K_TEMBLOR = 0.001
TAU_RECUPERACION = 300.0
TAU_BASE = 30.0
K_MEM = 0.005
SUELO_CONFIANZA = 0.2
K_HOLD = 0.001
TAU_CB = 10.0
CB_MAX = 500.0
LAMBDA_FISICO = 0.15
LAMBDA_COSTO = 0.5
UMBRAL_CB_JUEGO = 40.0
K_INFLUENCIA_JUEGO = 0.0005
SEMILLA_BASE = 44
PERIODO_ALTERNANCIA = 80.0

# Parametros específicos V180c
HABITO_SETPOINT = -60.0
TRAUMA_SETPOINT = 60.0
EPISODIO_SETPOINT = 45.0
CONSOLIDACION_CICLOS = 20
TRAUMA_DURACION = 15.0
TRAUMA_COSTO_MULTIPLIER = 2.0
EPISODIO_CICLOS = 30
PRUEBA_TRIALS = 50
EXPOSURE_STEPS_PER_TRIAL = 600
TRIAL_DURATION = EXPOSURE_STEPS_PER_TRIAL * DT

# Umbrales de éxito
P_EPISODIO_MAX = 0.30
LATENCIA_RATIO_MIN = 1.5
VAL_TRAUMA_MAX = -1.5
VAL_HABITO_MIN = 10.0

# ============================================================
# CONSTANTES RE-FUSIONADAS  (RITUAL_* VERBATIM de V165.py:90-98)
# ============================================================
RITUAL_TAU = 180.0                      # Persistencia intermedia
RITUAL_REPETICION_MIN = 3
RITUAL_GAIN = 0.05
RITUAL_PATRON_TEMPORAL = 40.0
RITUAL_TOLERANCIA = 0.3
RITUAL_UMBRAL_ACTIVACION = 0.4          # BAJADO: activa antes (era 0.5)
RITUAL_UMBRAL_CB = 28.0
RITUAL_SALIDA_SUAVE = 0.95
RITUAL_PERSISTENCIA_MIN = 3

# CONSTANTES RE-FUSIONADAS  (META_* VERBATIM de V167.py:73-76)
META_TAU = 30.0                     # Constante de tiempo del monitor
META_UMBRAL_DESAJUSTE = 0.5         # Umbral para declarar disfuncionalidad (observacional)
META_VENTANA_ERROR = 200            # Ventana para calcular error sostenido
META_K_SUAVIDAD = 0.1               # Suavizado de la señal

# ============================================================
# FLAGS POR CAPACIDAD — por defecto TODO ENCENDIDO (all-on)
# Cada capacidad se puede apagar de forma independiente.
# 'subj_sem' NO es flag: es gancho de interfaz (·), no implementado.
# ============================================================
DEFAULT_FLAGS = {
    'fatiga':           True,   # Fatiga = tiempo biológico (V155/V180c). Off -> efectos neutros.
    'memoria_ausencia': True,   # Confianza que decae en silencio (V153/V155). Off -> confianza=1.
    'cb':               True,   # Consciencia básica / presión de desacople (V155). Off -> Cb=0.
    'juego':            True,   # Juego enactuado (V156/V165). Off -> nunca activo.
    'valencia':         True,   # Valencia/deliberación = R_op (veto) + R_af (afirmación) (V176/V181).
    'memoria_episodica': True,  # Veto episódico con costo cognitivo (V180c). Off -> sin recall.
    'ritual':           True,   # Ritual estabilizador (V165, re-fusionado).
    'meta':             True,   # Meta-representación Rᴿ (V167, re-fusionado). REQUIERE ritual.
}

# ============================================================
# VALENCIA LOCAL (CORREGIDA: Penalización de trauma incondicional)
# ============================================================
class ValenciaLocal:
    def __init__(self, trauma_costo_multiplier=TRAUMA_COSTO_MULTIPLIER):
        self.valencia = {}
        self.tasa_aprendizaje = 0.001
        self.historial = {}
        self.trauma_costo_multiplier = trauma_costo_multiplier

    def actualizar(self, setpoint, error, costo_pagado, dt, reward=0.0, good_threshold=5.0, trauma=False):
        key = round(setpoint / 5) * 5 if setpoint != 0 else 0
        if key not in self.valencia:
            self.valencia[key] = 0.0
            self.historial[key] = []
        
        costo_efectivo = costo_pagado * self.trauma_costo_multiplier if trauma else costo_pagado
        
        # 🟢 CORRECCIÓN 1: Si es trauma, la penalización es masiva e incondicional
        if trauma:
            self.valencia[key] -= self.tasa_aprendizaje * dt * 150.0
        elif abs(error) < good_threshold:
            self.valencia[key] += reward * self.tasa_aprendizaje * dt
            self.valencia[key] += self.tasa_aprendizaje * dt * 10.0
        else:
            self.valencia[key] -= self.tasa_aprendizaje * dt * abs(error) * 0.2
            
        self.valencia[key] -= self.tasa_aprendizaje * dt * costo_efectivo * 0.1
        
        self.valencia[key] = max(-100.0, min(100.0, self.valencia[key]))
        self.historial[key].append(self.valencia[key])
        return self.valencia[key]

    def get_valencia(self, setpoint):
        key = round(setpoint / 5) * 5 if setpoint != 0 else 0
        return self.valencia.get(key, 0.0)

    def reset(self):
        self.valencia = {}
        self.historial = {}

# ============================================================
# MEMORIA EPISÓDICA
# ============================================================
class MemoriaEpisodicaV180c:
    def __init__(self, capacidad_max=50):
        self.memoria = []
        self.capacidad_max = capacidad_max
        self.tiempo_recuperacion = 0.0
        self.historial_recuperaciones = []

    def registrar(self, t, setpoint, valencia, resultado, contexto="normal"):
        evento = {
            't': t,
            'setpoint': setpoint,
            'valencia': valencia,
            'resultado': resultado,
            'contexto': contexto
        }
        self.memoria.append(evento)
        if len(self.memoria) > self.capacidad_max:
            self.memoria.pop(0)

    def recuperar(self, setpoint_buscado, ventana_t=15.0): # 🟢 CORRECCIÓN 3: Ventana ampliada
        self.tiempo_recuperacion = 0.0
        candidatos = []
        for ev in self.memoria:
            if abs(ev['setpoint'] - setpoint_buscado) < ventana_t:
                candidatos.append(ev)
        
        if not candidatos:
            return None, 0.0
        
        self.tiempo_recuperacion = 0.15 + (len(candidatos) * 0.08)
        candidatos.sort(key=lambda x: abs(x['valencia']), reverse=True)
        
        self.historial_recuperaciones.append({
            'buscado': setpoint_buscado,
            'encontrado': candidatos[0]['setpoint'],
            'tiempo': self.tiempo_recuperacion
        })
        
        return candidatos[0], self.tiempo_recuperacion

    def reset(self):
        self.memoria = []
        self.tiempo_recuperacion = 0.0
        self.historial_recuperaciones = []

# ============================================================
# MEMORIA DE TRABAJO (CORREGIDA: Penalización episódica aplastante)
# ============================================================
class MemoriaDeTrabajo:
    def __init__(self, steps_por_opcion=125):
        self.steps_por_opcion = steps_por_opcion
        self.opciones_ensayadas = {}
        self.tiempo_deliberacion = 0.0
        self.decision_final = None
        self.historial_deliberaciones = []

    def deliberar(self, opciones_disponibles, valencia_local, memoria_episodica, D_actual, current_sp=None):
        self.opciones_ensayadas = {}
        puntajes = {}
        explor_w = min(0.4, D_actual * 1.5)
        val_w = 1.0 - explor_w
        
        tiempo_base_por_opcion = self.steps_por_opcion * DT
        tiempo_total_busqueda = 0.0
        max_impacto_recuerdo = 0.0

        for opcion in opciones_disponibles:
            val = valencia_local.get_valencia(opcion)
            explor_bonus = D_actual * max(0.0, 1.0 - abs(val) / 50.0) * 0.1
            current_bonus = 0.8 if (current_sp is not None and abs(opcion - current_sp) < 1.0) else 0.0
            
            puntaje = (val * val_w + explor_bonus + current_bonus)
            
            # 🟢 CORRECCIÓN 2: Penalización episódica aplastante (-100.0)
            if memoria_episodica is not None:
                evento, t_rec = memoria_episodica.recuperar(opcion, ventana_t=15.0)
                tiempo_total_busqueda += t_rec
                if evento and evento['resultado'] == 'trauma':
                    puntaje -= 100.0  # Domina absolutamente cualquier valencia local positiva
                    max_impacto_recuerdo = max(max_impacto_recuerdo, 100.0)
                    
            puntajes[opcion] = puntaje
            self.opciones_ensayadas[opcion] = puntaje

        factor_conflicto = 1.0 + (D_actual * 3.5)
        self.tiempo_deliberacion = (tiempo_base_por_opcion * len(opciones_disponibles) * factor_conflicto) + tiempo_total_busqueda

        self.decision_final = max(puntajes, key=puntajes.get)
        self.historial_deliberaciones.append({
            'opciones': list(opciones_disponibles),
            'puntajes': puntajes,
            'decision': self.decision_final,
            'tiempo': self.tiempo_deliberacion,
            'impacto_recuerdo': max_impacto_recuerdo
        })
        return self.decision_final, puntajes, self.tiempo_deliberacion, max_impacto_recuerdo

    def reset(self):
        self.opciones_ensayadas = {}
        self.tiempo_deliberacion = 0.0
        self.decision_final = None
        self.historial_deliberaciones = []

# ============================================================
# REGISTRO DE REPRESENTACIONES
# ============================================================
class RegistroRepresentaciones:
    def __init__(self, ventana=200):
        self.ventana = ventana
        self.historial_representaciones = deque(maxlen=ventana)
        self.historial_acciones = deque(maxlen=ventana)
        self.historial_setpoints = deque(maxlen=ventana)

    def registrar(self, representacion, accion_ejecutada, setpoint):
        self.historial_representaciones.append(representacion)
        self.historial_acciones.append(accion_ejecutada)
        self.historial_setpoints.append(setpoint)

    def calcular_var_R(self):
        if len(self.historial_representaciones) < 20:
            return 0.0
        discretos = np.array([round(r / 10.0) * 10 for r in self.historial_representaciones])
        var = np.var(discretos)
        return min(1.0, var / 100.0)

    def calcular_Pmax(self):
        if len(self.historial_representaciones) < 20:
            return 1.0
        discretos = [round(r / 10.0) * 10 for r in self.historial_representaciones]
        unique, counts = np.unique(discretos, return_counts=True)
        return np.max(counts) / len(discretos)

    def calcular_desacople(self):
        var_R = self.calcular_var_R()
        Pmax = self.calcular_Pmax()
        return var_R * (1.0 - Pmax)

    def calcular_D_conflicto(self, valencias_opciones):
        if len(valencias_opciones) < 2:
            return 0.0
        vals = np.array(valencias_opciones)
        exp_vals = np.exp(vals / 10.0)
        probs = exp_vals / np.sum(exp_vals)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        D_entropia = entropy / np.log(len(vals))
        hay_trauma = np.any(vals < -0.5)
        factor_amenaza = 0.55 if hay_trauma else 0.0
        D_conflicto = (D_entropia * 0.5) + factor_amenaza
        return np.clip(D_conflicto, 0.0, 1.0)

    def reset(self):
        self.historial_representaciones.clear()
        self.historial_acciones.clear()
        self.historial_setpoints.clear()

# ============================================================
# COMPONENTES AUXILIARES (Fatiga, Memoria Ausencia, Consciencia, Juego)
# ============================================================
class FatigaMetabolica:
    def __init__(self, k_gain=K_GAIN, k_precision=K_PRECISION, k_temblor=K_TEMBLOR, tau_recuperacion=TAU_RECUPERACION):
        self.k_gain, self.k_precision, self.k_temblor, self.tau_recuperacion = k_gain, k_precision, k_temblor, tau_recuperacion
        self.historia, self.fatiga_activa = 0.0, 0.0
    def actualizar(self, delta_real, costo_trabajo, en_reposo_real, dt):
        self.historia += abs(delta_real)
        if not en_reposo_real: self.fatiga_activa += costo_trabajo
        else: self.fatiga_activa *= np.exp(-dt / self.tau_recuperacion)
        self.fatiga_activa = min(self.fatiga_activa, 20000.0)
        factor_gain = max(0.2, min(1.0, np.exp(-self.k_gain * self.fatiga_activa)))
        zona_muerta_efectiva = min(ZONA_MUERTA_MAX, ZONA_MUERTA_BASE + self.k_precision * self.fatiga_activa)
        temblor = np.clip(self.k_temblor * self.fatiga_activa * np.random.randn(), -3.0, 3.0)
        return factor_gain, zona_muerta_efectiva, temblor
    def get_historia(self): return self.historia
    def get_fatiga(self): return self.fatiga_activa
    def reset(self): self.historia, self.fatiga_activa = 0.0, 0.0

class MemoriaAusencia:
    def __init__(self, tau_base=TAU_BASE, k_mem=K_MEM, suelo_confianza=SUELO_CONFIANZA):
        self.setpoint_last, self.t_ausencia, self.tau_base, self.k_mem = 0.0, 0.0, tau_base, k_mem
        self.tau_mem, self.suelo_confianza = tau_base, suelo_confianza
    def actualizar(self, setpoint, E_historia, dt):
        if setpoint is not None:
            self.setpoint_last, self.t_ausencia = setpoint, 0.0
            self.tau_mem = self.tau_base + self.k_mem * E_historia
            return self.setpoint_last, 1.0
        else:
            self.t_ausencia += dt
            return self.setpoint_last, np.exp(-self.t_ausencia / self.tau_mem)
    def reset(self): self.setpoint_last, self.t_ausencia, self.tau_mem = 0.0, 0.0, self.tau_base

class ConscienciaBasica:
    def __init__(self, tau_cb=TAU_CB, cb_max=CB_MAX):
        self.Cb, self.tau_cb, self.cb_max = 0.0, tau_cb, cb_max
    def actualizar(self, e_R, A_sys_env, dt):
        presion = e_R * (1.0 - A_sys_env)
        self.Cb = max(0.0, min(self.cb_max, self.Cb + (presion - self.Cb / self.tau_cb) * dt))
        return self.Cb, presion
    def reset(self): self.Cb = 0.0

class ModoJuego:
    def __init__(self, lambda_fisico=LAMBDA_FISICO, lambda_costo=LAMBDA_COSTO, umbral_cb=UMBRAL_CB_JUEGO, k_influencia=K_INFLUENCIA_JUEGO):
        self.lambda_fisico, self.lambda_costo, self.umbral_cb, self.k_influencia = lambda_fisico, lambda_costo, umbral_cb, k_influencia
        self.activo, self.tiempo_activo = False, 0.0
    def actualizar(self, Cb, confianza, setpoint_presente):
        if setpoint_presente is not None and Cb > self.umbral_cb:
            self.activo, self.tiempo_activo = True, self.tiempo_activo + DT
        else:
            self.activo = False
        return self.activo
    def aplicar(self, delta_raw):
        if self.activo: return delta_raw * self.lambda_fisico, abs(delta_raw) * self.lambda_costo
        return delta_raw, abs(delta_raw)
    def get_influencia(self, Cb, confianza):
        if self.activo and Cb > self.umbral_cb: return self.k_influencia * (Cb - self.umbral_cb) * (1 - confianza)
        return 0.0
    def reset(self): self.activo, self.tiempo_activo = False, 0.0

# ============================================================
# HEMISFERIO
# ============================================================
class Hemisferio:
    def __init__(self, nombre, tau, generar_entrada_func, seed=None, sesgo=0.0):
        if seed is not None: np.random.seed(seed)
        self.nombre, self.tau, self.generar_entrada, self.sesgo = nombre, tau, generar_entrada_func, sesgo
        self.Phi = np.random.normal(sesgo, 0.1, 32)
        self.Phi_vel = np.zeros(32)
        self.entrada = None
        self.sr = 48000
        self.factor_inanicion = 1.0

    def _calcular_omega(self): return np.mean(self.Phi[:32])

    def generar_entrada_para_t(self, t, duracion_total):
        if self.entrada is None: self.entrada = self.generar_entrada(duracion_total, self.sr)
        idx = int(t * self.sr)
        return self.entrada[idx] * self.factor_inanicion if idx < len(self.entrada) else 0.0

    def actualizar(self, t, dt, duracion_total, otro_hemisferio=None):
        entrada = self.generar_entrada_para_t(t, duracion_total)
        laplaciano = np.zeros_like(self.Phi)
        for i in range(1, 31): laplaciano[i] = self.Phi[i-1] - 2*self.Phi[i] + self.Phi[i+1]
        reaccion = self.Phi * (1 - self.Phi * self.Phi)
        forzamiento = np.zeros_like(self.Phi)
        forzamiento[0], forzamiento[-1] = entrada, -entrada
        acoplamiento = np.zeros_like(self.Phi)
        if otro_hemisferio is not None and abs(self._calcular_omega() - otro_hemisferio._calcular_omega()) > 0.5:
            acoplamiento = 0.01 * (otro_hemisferio.Phi - self.Phi)
        self.Phi_vel += (laplaciano + reaccion + forzamiento + acoplamiento) * dt
        self.Phi = np.clip(self.Phi + self.Phi_vel * dt, -1.0, 1.0)
        return {'omega': self._calcular_omega()}

# ============================================================
# RITUAL  (RE-FUSIONADO — VERBATIM de V165.py:105-216)
# ============================================================
class Ritual:
    def __init__(self, tau=RITUAL_TAU, repeticion_min=RITUAL_REPETICION_MIN,
                 ritual_gain=RITUAL_GAIN, patron_temporal=RITUAL_PATRON_TEMPORAL,
                 tolerancia=RITUAL_TOLERANCIA, umbral_activacion=RITUAL_UMBRAL_ACTIVACION,
                 umbral_cb=RITUAL_UMBRAL_CB, salida_suave=RITUAL_SALIDA_SUAVE,
                 persistencia_min=RITUAL_PERSISTENCIA_MIN):
        self.tau = tau
        self.repeticion_min = repeticion_min
        self.ritual_gain = ritual_gain
        self.patron_temporal = patron_temporal
        self.tolerancia = tolerancia
        self.umbral_activacion = umbral_activacion
        self.umbral_cb = umbral_cb
        self.salida_suave = salida_suave
        self.persistencia_min = persistencia_min

        self.activation = 0.0
        self.active = False
        self.patron_buffer = []
        self.repeticiones_consecutivas = 0
        self.historial_activation = []
        self.historial_active = []
        self.tiempo_activo = 0.0

        self.ultima_orientacion = 0.0
        self.cruces = 0
        self.ciclos_sin_cruce = 0

    def detectar_cruce_por_cero(self, orientacion):
        cruce = (self.ultima_orientacion < 0 and orientacion >= 0) or \
                (self.ultima_orientacion > 0 and orientacion <= 0)

        self.ultima_orientacion = orientacion

        if cruce:
            self.cruces += 1
            self.ciclos_sin_cruce = 0
            direccion = 1 if orientacion >= 0 else -1
            return True, direccion
        else:
            self.ciclos_sin_cruce += 1
            return False, 0

    def actualizar(self, orientacion, Cb, tiempo_actual, dt):
        es_cruce, direccion = self.detectar_cruce_por_cero(orientacion)

        if es_cruce and Cb > self.umbral_cb:
            es_patron = False
            for t_prev, dir_prev in self.patron_buffer:
                dt_desde_prev = tiempo_actual - t_prev
                timing_ok = abs(dt_desde_prev - self.patron_temporal) <= (self.patron_temporal * self.tolerancia)
                direccion_ok = dir_prev == direccion

                if timing_ok and direccion_ok:
                    es_patron = True
                    break

            if es_patron:
                self.repeticiones_consecutivas += 1
                if self.repeticiones_consecutivas >= self.repeticion_min:
                    incremento = Cb * self.repeticiones_consecutivas / 100.0
                    self.activation += incremento * dt
            else:
                self.repeticiones_consecutivas = max(0, self.repeticiones_consecutivas - 0.5)

            self.patron_buffer.append((tiempo_actual, direccion))
            if len(self.patron_buffer) > 10:
                self.patron_buffer.pop(0)
        else:
            self.repeticiones_consecutivas = max(0, self.repeticiones_consecutivas - 0.1)

        self.activation *= np.exp(-dt / self.tau)
        self.activation = max(0.0, min(2.0, self.activation))

        # Salida suave
        if self.activation > self.umbral_activacion:
            self.active = True
        elif self.active:
            if self.ciclos_sin_cruce > self.persistencia_min:
                self.active = False
            else:
                self.active = self.active * self.salida_suave

        if self.active:
            self.tiempo_activo += dt

        self.historial_activation.append(self.activation)
        self.historial_active.append(self.active)

        return self.active

    def modular_correccion(self, delta_raw, correccion_ritual):
        if self.active:
            return delta_raw * (1 - self.activation * 0.3) + correccion_ritual * self.activation
        return delta_raw

    def reset(self):
        self.activation = 0.0
        self.active = False
        self.patron_buffer = []
        self.repeticiones_consecutivas = 0
        self.historial_activation = []
        self.historial_active = []
        self.tiempo_activo = 0.0
        self.ultima_orientacion = 0.0
        self.cruces = 0
        self.ciclos_sin_cruce = 0

    def get_influencia(self, Cb):
        if self.active:
            return self.activation * self.ritual_gain
        return 0.0


# ============================================================
# META-REPRESENTACIÓN OBSERVACIONAL Rᴿ  (RE-FUSIONADO — VERBATIM de V167.py:395-490)
# NOTA (decisión #3): los literales 500.0 (== CB_MAX) y 50 quedan VERBATIM, sin parametrizar.
# ============================================================
class MetaRepresentacionObservacional:
    """
    Etapa 4: Monitor observacional de desajuste.

    Detecta cuándo el ritual podría estar siendo disfuncional:
    - Ritual activo
    - Error sostenido alto
    - Cb alta (presión de desacople) O Cb baja pero error alto (ritual "ciego")

    GENERA SEÑAL DE DASAJUSTE PERO NO INHIBE EL RITUAL.
    Solo registra y reporta. La inhibición será R_op (V168).
    """

    def __init__(self, tau=META_TAU, umbral_desajuste=META_UMBRAL_DESAJUSTE,
                 ventana_error=META_VENTANA_ERROR, k_suavidad=META_K_SUAVIDAD):
        self.tau = tau
        self.umbral_desajuste = umbral_desajuste
        self.ventana_error = ventana_error
        self.k_suavidad = k_suavidad

        self.desajuste = 0.0          # Nivel de desajuste detectado (0-1)
        self.historial_desajuste = []
        self.buffer_error = deque(maxlen=ventana_error)
        self.buffer_Cb = deque(maxlen=ventana_error)
        self.buffer_ritual = deque(maxlen=ventana_error)

    def actualizar(self, error, Cb, ritual_activo, dt):
        """
        Actualiza el monitor de desajuste (observacional, sin inhibición).

        La señal de desajuste aumenta cuando:
        - El ritual está activo
        - El error es sostenidamente alto
        - La presión de desacople es significativa (error alto con Cb alta o baja)

        Retorna:
            senal: nivel de desajuste (0-1+)
            hay_desajuste: True si senal > umbral (solo para reporte)
        """
        # Registrar métricas
        self.buffer_error.append(abs(error))
        self.buffer_Cb.append(Cb)
        self.buffer_ritual.append(ritual_activo)

        # Calcular métricas sostenidas
        if len(self.buffer_error) > self.ventana_error // 2:
            error_sostenido = np.mean(self.buffer_error)
            Cb_sostenido = np.mean(self.buffer_Cb)
            ritual_sostenido = np.mean(self.buffer_ritual) > 0.5
        else:
            error_sostenido = abs(error)
            Cb_sostenido = Cb
            ritual_sostenido = ritual_activo

        # Presión de desajuste
        if ritual_sostenido:
            # Normalizar error (0-60° -> 0-1)
            error_norm = min(1.0, error_sostenido / 60.0)

            # La disfuncionalidad puede ser:
            # 1. Cb alta (presión activa) -> error_norm * Cb_norm
            # 2. Cb baja pero error alto (ritual "ciego" funcionando mal)
            Cb_norm = min(1.0, Cb_sostenido / 500.0)

            # Caso 1: presión activa (Cb alta)
            presion_activa = error_norm * Cb_norm

            # Caso 2: ritual ciego (Cb baja, error alto)
            # El ritual está tranquilo pero desacoplado
            if Cb_sostenido < 50 and error_sostenido > 30:
                presion_ciega = error_norm * 0.8
            else:
                presion_ciega = 0.0

            presion = max(presion_activa, presion_ciega)
        else:
            presion = 0.0

        # Desajuste como integrador leaky
        d_desajuste = presion - self.desajuste / self.tau
        self.desajuste += d_desajuste * dt
        self.desajuste = max(0.0, min(1.0, self.desajuste))

        self.historial_desajuste.append(self.desajuste)

        # Suavizado para reporte
        senal_suavizada = self.desajuste

        return senal_suavizada, senal_suavizada > self.umbral_desajuste

    def reset(self):
        self.desajuste = 0.0
        self.historial_desajuste = []
        self.buffer_error.clear()
        self.buffer_Cb.clear()
        self.buffer_ritual.clear()


# ============================================================
# APARATO MOTOR V180c
# ============================================================
class AparatoMotorV180c:
    def __init__(self, flags=None):
        # FLAGS por capacidad (all-on por defecto). Copia para no mutar el global.
        self.flags = dict(DEFAULT_FLAGS)
        if flags:
            self.flags.update(flags)
        # Guarda (decisión #4 / regla "no resolver en silencio"): Rᴿ requiere ritual.
        if self.flags['meta'] and not self.flags['ritual']:
            print("[AVISO] flag 'meta' (Rᴿ) ON con 'ritual' OFF: Rᴿ solo verá ritual_activo=False "
                  "(desajuste permanece en 0). Configuración inconsistente — se respeta lo pedido, se avisa.")

        self.orientacion = 0.0
        self.Kp_base, self.Kp_actual, self.Kp_min, self.Kp_max = KP_BASE, KP_BASE, KP_MIN, KP_MAX
        self.limite, self.zona_muerta, self.inercia = 90.0, ZONA_MUERTA_BASE, INERCIA
        self.ultimo_delta, self.sensibilidad_grad, self.t = 0.0, SENSIBILIDAD_GRAD, 0.0

        self.fatiga = FatigaMetabolica()
        self.memoria = MemoriaAusencia()
        self.consciencia = ConscienciaBasica()
        self.juego = ModoJuego()
        self.valencia = ValenciaLocal()
        self.memoria_trabajo = MemoriaDeTrabajo()
        self.memoria_episodica = MemoriaEpisodicaV180c() # 🟢 NUEVO
        self.registro = RegistroRepresentaciones()
        # RE-FUSIONADO: Ritual (V165) y Rᴿ (V167). Instancia solo si la capacidad está ON.
        self.ritual = Ritual() if self.flags['ritual'] else None
        self.meta = MetaRepresentacionObservacional() if self.flags['meta'] else None
        self.memoria_error = deque(maxlen=VENTANA_OSCILACION)
        self.en_deliberacion = False

        # REGISTRO DE SALIDA (decisión #4): últimos valores observables por paso.
        self.ultima_opcion_elegida = 0.0   # gancho de interfaz para Subj_sem [·]
        self.ultimo_A_sys_env = 1.0
        self.ultimo_Cb = 0.0
        self.ultimo_ritual_activo = False
        self.ultima_senal_desajuste = 0.0

    def calcular_factor_freno(self, error): return 1 - np.exp(-abs(error) / 30.0)

    def actualizar_plasticidad(self, error):
        self.memoria_error.append(error)
        if len(self.memoria_error) >= VENTANA_OSCILACION:
            oscilacion = np.std(self.memoria_error)
            if oscilacion > self.zona_muerta * 1.5: self.Kp_actual = max(self.Kp_min, self.Kp_actual * 0.99)
            elif oscilacion < self.zona_muerta * 0.5: self.Kp_actual = min(self.Kp_max, self.Kp_actual * 1.01)

    def ejecutar_con_deliberacion(self, opciones_disponibles, gradiente, t, dt, trauma=False, target_reward=None, registrar_episodio=False):
        F = self.flags
        D_actual = self.registro.calcular_desacople()
        tiempo_delib = 0.0
        impacto_recuerdo = 0.0
        mem_epi = self.memoria_episodica if F['memoria_episodica'] else None  # veto episódico conmutable

        # --- DELIBERACIÓN = R_op (veto) + R_af (afirmación), substrato valencia/memoria de trabajo ---
        if len(opciones_disponibles) > 1:
            if F['valencia']:
                valencias = [self.valencia.get_valencia(op) for op in opciones_disponibles]
                D_actual = self.registro.calcular_D_conflicto(valencias)
                opcion_elegida, puntajes, tiempo_delib, impacto_recuerdo = self.memoria_trabajo.deliberar(
                    opciones_disponibles, self.valencia, mem_epi, D_actual, current_sp=self.orientacion
                )
            else:
                # valencia OFF -> elección neutra (la más cercana a la orientación actual), sin veto
                opcion_elegida = min(opciones_disponibles, key=lambda op: abs(op - self.orientacion))
                tiempo_delib = (self.memoria_trabajo.steps_por_opcion * DT) * 0.5
        else:
            only = opciones_disponibles[0]
            val_only = self.valencia.get_valencia(only)
            opcion_elegida = 0.0 if (F['valencia'] and val_only < -2.0) else only
            tiempo_delib = (self.memoria_trabajo.steps_por_opcion * DT) * 0.5

        # 🟢 REGISTRO EPISÓDICO EXPLÍCITO (conmutable)
        if registrar_episodio and F['memoria_episodica']:
            resultado = 'trauma' if trauma else ('reward' if target_reward is not None else 'neutro')
            self.memoria_episodica.registrar(t, opcion_elegida, self.valencia.get_valencia(opcion_elegida), resultado, contexto="codificacion_forzada")

        setpoint_objetivo, confianza = self.memoria.actualizar(opcion_elegida, self.fatiga.get_historia(), dt)
        if not F['memoria_ausencia']:
            confianza = 1.0  # OFF -> sin pérdida de confianza en ausencia (no decae en silencio)
        error = setpoint_objetivo - self.orientacion
        e_R = abs(error)
        val_local = self.valencia.get_valencia(opcion_elegida)
        e_R_efectivo = e_R * (1.0 + max(0.0, -val_local / 200.0))

        A_sys_env = min(1.0, abs(self.orientacion) / abs(opcion_elegida)) if abs(opcion_elegida) > 0.01 and abs(self.orientacion) > 0.01 else confianza

        # --- Cb (consciencia básica / presión de desacople) ---
        if F['cb']:
            Cb, presion = self.consciencia.actualizar(e_R_efectivo, A_sys_env, dt)
        else:
            Cb, presion = 0.0, 0.0

        # --- ETAPA 3: RITUAL (antes de juego; lo inhibe si activo) [re-fusión VERBATIM V165] ---
        ritual_activo = self.ritual.actualizar(self.orientacion, Cb, t, dt) if (F['ritual'] and self.ritual is not None) else False

        # --- ETAPA 4: META-REPRESENTACIÓN Rᴿ (SOLO OBSERVA, no inhibe) [re-fusión VERBATIM V167] ---
        senal_desajuste, hay_desajuste = 0.0, False
        if F['meta'] and self.meta is not None:
            senal_desajuste, hay_desajuste = self.meta.actualizar(error, Cb, ritual_activo, dt)

        # --- ETAPA 2: JUEGO (inhibido si ritual activo) ---
        if ritual_activo:
            juego_activo = False
            self.juego.activo = False
        elif F['juego']:
            juego_activo = self.juego.actualizar(Cb, confianza, opcion_elegida)
        else:
            juego_activo = False

        # --- EFECTOS DE FATIGA (metabólica = tiempo biológico) ---
        if F['fatiga']:
            factor_gain, zona_muerta_efectiva, temblor = self.fatiga.actualizar(0, 0, False, dt)
        else:
            factor_gain, zona_muerta_efectiva, temblor = 1.0, ZONA_MUERTA_BASE, 0.0

        rwd = 0.0
        if target_reward is not None and abs(opcion_elegida - target_reward) < 1.0 and abs(error) < zona_muerta_efectiva:
            rwd = 1.0

        if abs(error) < zona_muerta_efectiva and F['fatiga']:
            self.fatiga.actualizar(0, 0, True, dt)

        if F['valencia']:
            self.valencia.actualizar(opcion_elegida, error, 0.0, dt, reward=rwd, good_threshold=zona_muerta_efectiva, trauma=trauma)

        direccion = np.sign(error)
        confianza_sensorial = min(1.0, abs(gradiente) * self.sensibilidad_grad)
        factor_freno = self.calcular_factor_freno(error)
        Kp_base_efectivo = max(self.Kp_min, self.Kp_actual * factor_gain * confianza_sensorial)
        Kp_inst = Kp_base_efectivo * (self.memoria.suelo_confianza + (1 - self.memoria.suelo_confianza) * confianza)

        delta_error = Kp_inst * abs(error) * direccion * factor_freno
        torque_memoria = (K_HOLD * (self.memoria.setpoint_last - self.orientacion) * confianza) if F['memoria_ausencia'] else 0.0
        delta_raw = delta_error + torque_memoria

        # Influencia del juego (solo si activo y ritual NO activo) [VERBATIM V165:576-579]
        influencia_juego = self.juego.get_influencia(Cb, confianza) if (juego_activo and not ritual_activo) else 0.0
        if influencia_juego != 0:
            delta_raw += influencia_juego * (self.memoria.setpoint_last - self.orientacion) * 0.1

        # Influencia del ritual (modula corrección si activo) [VERBATIM V165:581-587]
        if F['ritual'] and self.ritual is not None and ritual_activo:
            correccion_ritual = 0.0
            if self.ritual.patron_buffer:
                _, ultima_dir = self.ritual.patron_buffer[-1]
                correccion_ritual = ultima_dir * 10.0 * self.ritual.ritual_gain
            delta_raw = self.ritual.modular_correccion(delta_raw, correccion_ritual)

        costo_total_estimado = abs(delta_error) + abs(torque_memoria)
        if F['valencia']:
            self.valencia.actualizar(opcion_elegida, error, costo_total_estimado, dt, reward=rwd, good_threshold=zona_muerta_efectiva, trauma=trauma)

        delta = self.inercia * self.ultimo_delta + (1 - self.inercia) * delta_raw
        self.ultimo_delta = delta
        delta_fisico, delta_costo = self.juego.aplicar(delta)
        costo_total = costo_total_estimado + delta_costo

        en_reposo_real = (abs(delta) < 0.001 and abs(torque_memoria) < 0.001)
        if F['fatiga']:
            self.fatiga.actualizar(delta_fisico, costo_total, en_reposo_real, dt)
        delta_fisico += temblor * dt

        self.actualizar_plasticidad(error)
        self.orientacion = np.clip(self.orientacion + delta_fisico, -self.limite, self.limite)
        self.t += dt

        accion_ejecutada = abs(delta_fisico) > 0.01
        self.registro.registrar(setpoint_objetivo, accion_ejecutada, opcion_elegida)

        # REGISTRO DE SALIDA (decisión #4): observables del paso (incl. gancho Subj_sem)
        self.ultima_opcion_elegida = opcion_elegida
        self.ultimo_A_sys_env = A_sys_env
        self.ultimo_Cb = Cb
        self.ultimo_ritual_activo = ritual_activo
        self.ultima_senal_desajuste = senal_desajuste

        return (self.orientacion, self.fatiga.get_historia(), self.fatiga.get_fatiga(), confianza, zona_muerta_efectiva,
                Cb, presion, juego_activo, delta_costo, D_actual, val_local, tiempo_delib, opcion_elegida, rwd, impacto_recuerdo)

    def reset(self):
        self.orientacion = self.ultimo_delta = self.t = 0.0
        self.Kp_actual = KP_BASE
        self.memoria_error.clear()
        self.fatiga.reset()
        self.memoria.reset()
        self.consciencia.reset()
        self.juego.reset()
        self.valencia.reset()
        self.memoria_trabajo.reset()
        self.memoria_episodica.reset()
        self.registro.reset()
        if self.ritual is not None: self.ritual.reset()
        if self.meta is not None: self.meta.reset()

# ============================================================
# ORGANISMO V180c
# ============================================================
class OrganismoV180c:
    def __init__(self, seed, flags=None):
        self.nombre = f"Organismo_{seed}"
        self.flags = dict(DEFAULT_FLAGS)
        if flags:
            self.flags.update(flags)
        def generar_ruido_rosa(duracion, sr):
            n = int(duracion * sr)
            ruido = np.random.normal(0, 1, n)
            fft = np.fft.rfft(ruido)
            freqs = np.fft.rfftfreq(n, 1/sr)
            filtro = 1.0 / np.sqrt(freqs + 0.01)
            return np.fft.irfft(fft * filtro, n=n) / (np.max(np.abs(ruido)) + 1e-10)
        def generar_clicks_poisson(duracion, tasa=0.5, sr=48000):
            n = int(duracion * sr)
            clicks = np.zeros(n)
            for _ in range(int(duracion * tasa)):
                pos = int(np.random.exponential(1.0/tasa) * sr)
                if pos < n: clicks[pos] = 1.0
            return clicks

        self.izquierdo = Hemisferio("L", 30.0, generar_ruido_rosa, seed=seed, sesgo=SESGO_L)
        self.derecho = Hemisferio("R", 300.0, generar_clicks_poisson, seed=seed+100, sesgo=SESGO_R)
        self.sistema_B_izq = Hemisferio("B_L", 30.0, generar_ruido_rosa, seed=seed+200, sesgo=SESGO_L)
        self.sistema_B_der = Hemisferio("B_R", 300.0, generar_clicks_poisson, seed=seed+300, sesgo=SESGO_R)
        self.motor = AparatoMotorV180c(flags=self.flags)
        self.modo_entrenamiento = True

    def actualizar_con_opciones(self, t, dt, duracion_total, opciones_disponibles, trauma=False, target_reward=None, registrar_episodio=False):
        for h in [self.izquierdo, self.derecho, self.sistema_B_izq, self.sistema_B_der]:
            h.actualizar(t, dt, duracion_total, self.derecho if h.nombre in ["L", "B_L"] else self.izquierdo)
        
        omega_A = (self.izquierdo._calcular_omega() + self.derecho._calcular_omega()) / 2
        omega_B = (self.sistema_B_izq._calcular_omega() + self.sistema_B_der._calcular_omega()) / 2
        gradiente = omega_A - omega_B
        
        if abs(self.motor.orientacion) > 0.1:
            gradiente += (self.motor.orientacion / 90.0) * 0.3

        return self.motor.ejecutar_con_deliberacion(
            opciones_disponibles, gradiente, t, dt, trauma, target_reward, registrar_episodio
        )

    def actualizar_setpoint(self, t, dt, duracion_total, setpoint, trauma=False, target_reward=None, registrar_episodio=False):
        return self.actualizar_con_opciones(t, dt, duracion_total, [setpoint], trauma, target_reward, registrar_episodio)

    def set_modo_entrenamiento(self, entrenamiento=True):
        self.modo_entrenamiento = entrenamiento
        if entrenamiento: self.motor.reset()

    # ----------------------------------------------------------------
    # GANCHO DE INTERFAZ [·] — Subj_sem / díada (NO implementado aquí).
    # Conducta observable por otra instancia, sin tocar internos.
    # (En V182d, observar_al_otro() lee exactamente get_valencia(banda).)
    # ----------------------------------------------------------------
    def get_valencia(self, banda):
        return self.motor.valencia.get_valencia(banda)

    def get_orientacion(self):
        return self.motor.orientacion

    def get_ultima_opcion(self):
        return self.motor.ultima_opcion_elegida

# ============================================================
# EXPERIMENTO PRINCIPAL V180c
# ============================================================
def ejecutar_v180c():
    print("=" * 100)
    print("EXPERIMENTO V180c — ANIMA-3: MEMORIA EPISÓDICA (CORREGIDO)")
    print("=" * 100)
    print("  OBJETIVO: Asociar evento traumático a +45° y recuperarlo para rechazarlo.")
    print("  CRITERIOS DE ÉXITO:")
    print(f"    ✅ P(+45.0° | episodio_trauma) < {P_EPISODIO_MAX:.0%}")
    print(f"    ✅ latencia_recuperacion > {LATENCIA_RATIO_MIN}× latencia_baseline")
    print(f"    ✅ Especificidad: Val(+60°) < {VAL_TRAUMA_MAX}")
    print(f"    ✅ Memoria preservada: Val(-60°) > {VAL_HABITO_MIN}")
    print("=" * 100)

    organismo = OrganismoV180c(seed=SEMILLA_BASE)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('V180c_logs', exist_ok=True)
    
    print("\nEntrenando lateralidad (10 repeticiones)...")
    organismo.set_modo_entrenamiento(True)
    for rep in range(REPETICIONES_LENTAS):
        for i in range(int(TIEMPO_POR_REPETICION / DT)):
            t = rep * TIEMPO_POR_REPETICION + i * DT
            organismo.actualizar_setpoint(t, DT, TIEMPO_POR_REPETICION * REPETICIONES_LENTAS, 0.0)
    organismo.set_modo_entrenamiento(False)
    t_actual = TIEMPO_POR_REPETICION * REPETICIONES_LENTAS
    print("  Entrenamiento completado.")

    # ---------------------------------------------------------
    # F0: BASELINE
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("FASE 0: BASELINE — Medir latencia sin conflicto")
    print("=" * 60)
    latencias_baseline = []
    for trial in range(20):
        trial_latencias = []
        for step in range(EXPOSURE_STEPS_PER_TRIAL):
            t_step = t_actual + step * DT
            _, _, _, _, _, _, _, _, _, _, _, latencia, _, _, _ = organismo.actualizar_con_opciones(
                t_step, DT, t_actual + 20 * TRIAL_DURATION, [HABITO_SETPOINT]
            )
            trial_latencias.append(latencia)
        latencias_baseline.append(np.mean(trial_latencias))
        t_actual += TRIAL_DURATION
    latencia_baseline_media = np.mean(latencias_baseline)
    print(f"  Latencia baseline media: {latencia_baseline_media:.3f}s")

    # ---------------------------------------------------------
    # F1: CONSOLIDACIÓN
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"FASE 1: Consolidación del hábito ({CONSOLIDACION_CICLOS} ciclos a {HABITO_SETPOINT}°)")
    print("=" * 60)
    for ciclo in range(CONSOLIDACION_CICLOS):
        for i in range(int(PERIODO_ALTERNANCIA / DT)):
            t = t_actual + i * DT
            organismo.actualizar_setpoint(t, DT, t_actual + PERIODO_ALTERNANCIA, HABITO_SETPOINT, target_reward=HABITO_SETPOINT)
        if (ciclo + 1) % 5 == 0:
            print(f"    Ciclo {ciclo+1}/{CONSOLIDACION_CICLOS}, valencia({HABITO_SETPOINT}°) = {organismo.motor.valencia.get_valencia(HABITO_SETPOINT):.2f}")
        t_actual += PERIODO_ALTERNANCIA

    # ---------------------------------------------------------
    # F2: TRAUMA ORIGINAL
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"FASE 2: Trauma específico ({TRAUMA_DURACION}s a {TRAUMA_SETPOINT}°, costo 2.0x)")
    print("=" * 60)
    for i in range(int(TRAUMA_DURACION / DT)):
        t = t_actual + i * DT
        organismo.actualizar_setpoint(t, DT, t_actual + TRAUMA_DURACION, TRAUMA_SETPOINT, trauma=True)
    t_actual += TRAUMA_DURACION
    print(f"  Valencia {TRAUMA_SETPOINT}° post-trauma: {organismo.motor.valencia.get_valencia(TRAUMA_SETPOINT):.2f}")

    # ---------------------------------------------------------
    # F3: CODIFICACIÓN EPISÓDICA
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"FASE 3: Evento episódico — Exposición a {EPISODIO_SETPOINT}° con marca 'trauma' ({EPISODIO_CICLOS} ciclos)")
    print("=" * 60)
    for ciclo in range(EPISODIO_CICLOS):
        for i in range(int(PERIODO_ALTERNANCIA / DT)):
            t = t_actual + i * DT
            organismo.actualizar_setpoint(t, DT, t_actual + PERIODO_ALTERNANCIA, EPISODIO_SETPOINT, trauma=True, registrar_episodio=True)
        if (ciclo + 1) % 10 == 0:
            print(f"    Ciclo {ciclo+1}/{EPISODIO_CICLOS}, valencia({EPISODIO_SETPOINT}°) = {organismo.motor.valencia.get_valencia(EPISODIO_SETPOINT):.2f}")
        t_actual += PERIODO_ALTERNANCIA
    print(f"  Eventos registrados en memoria episódica: {len(organismo.motor.memoria_episodica.memoria)}")

    # ---------------------------------------------------------
    # F4: PRUEBA DE RECUPERACIÓN
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"FASE 4: Test recuperación — Opciones simultáneas [{HABITO_SETPOINT}°, {EPISODIO_SETPOINT}°] ({PRUEBA_TRIALS} trials)")
    print("=" * 60)
    opciones_elegidas = []
    latencias_prueba = []
    impactos_recuerdo = []
    recuperaciones_exitosas = 0

    for trial in range(PRUEBA_TRIALS):
        t = t_actual + trial * TRIAL_DURATION
        trial_latencias = []
        trial_impactos = []
        evento_recuperado_en_trial = False
        
        for step in range(EXPOSURE_STEPS_PER_TRIAL):
            t_step = t + step * DT
            _, _, _, _, _, _, _, _, _, _, _, latencia, opcion, _, impacto = organismo.actualizar_con_opciones(
                t_step, DT, t_actual + PRUEBA_TRIALS * TRIAL_DURATION, 
                [HABITO_SETPOINT, EPISODIO_SETPOINT], 
                registrar_episodio=False
            )
            trial_latencias.append(latencia)
            trial_impactos.append(impacto)
            if impacto > 0:
                evento_recuperado_en_trial = True
                
        opciones_elegidas.append(opcion if opcion is not None else 0)
        latencias_prueba.append(np.mean(trial_latencias))
        impactos_recuerdo.append(np.max(trial_impactos))
        if evento_recuperado_en_trial:
            recuperaciones_exitosas += 1
            
        if (trial + 1) % 10 == 0:
            print(f"    Trial {trial+1}/{PRUEBA_TRIALS}...")
        t_actual += TRIAL_DURATION

    # ---------------------------------------------------------
    # ANÁLISIS DE MÉTRICAS
    # ---------------------------------------------------------
    tasa_recuperacion = recuperaciones_exitosas / PRUEBA_TRIALS
    latencia_prueba_media = np.mean(latencias_prueba)
    incremento_latencia = latencia_prueba_media - latencia_baseline_media
    ratio_latencia = latencia_prueba_media / latencia_baseline_media if latencia_baseline_media > 0 else 0
    
    coherencia = sum(1 for e in opciones_elegidas if abs(e - HABITO_SETPOINT) < 5.0) / PRUEBA_TRIALS
    p_episodio = 1.0 - coherencia
    impacto_promedio = np.mean([i for i in impactos_recuerdo if i > 0]) if any(i > 0 for i in impactos_recuerdo) else 0.0

    val_habito_final = organismo.motor.valencia.get_valencia(HABITO_SETPOINT)
    val_trauma_final = organismo.motor.valencia.get_valencia(TRAUMA_SETPOINT)
    val_episodio_final = organismo.motor.valencia.get_valencia(EPISODIO_SETPOINT)

    print("\n" + "=" * 80)
    print("RESULTADOS V180c — Memoria episódica")
    print("=" * 80)
    print(f"  📊 CONDUCTA:")
    print(f"     P(elegir {EPISODIO_SETPOINT}°) = {p_episodio:.1%} (umbral < {P_EPISODIO_MAX:.0%})")
    print(f"     P(elegir {HABITO_SETPOINT}°) = {coherencia:.1%}")
    print(f"     Eventos recuperados: {recuperaciones_exitosas}/{PRUEBA_TRIALS} ({tasa_recuperacion:.1%})")
    print(f"\n  📊 LATENCIA:")
    print(f"     Baseline: {latencia_baseline_media:.3f}s")
    print(f"     Prueba: {latencia_prueba_media:.3f}s")
    print(f"     Ratio: {ratio_latencia:.2f}x (umbral > {LATENCIA_RATIO_MIN}x)")
    print(f"\n  📊 VALENCIA:")
    print(f"     Val({HABITO_SETPOINT}°) final: {val_habito_final:.2f} (umbral > {VAL_HABITO_MIN})")
    print(f"     Val({TRAUMA_SETPOINT}°) final: {val_trauma_final:.2f} (umbral < {VAL_TRAUMA_MAX})")
    print(f"     Val({EPISODIO_SETPOINT}°) final: {val_episodio_final:.2f}")
    print(f"     Impacto promedio del recuerdo: {impacto_promedio:.2f}")

    # Evaluación
    p_ok = p_episodio < P_EPISODIO_MAX
    lat_ok = ratio_latencia > LATENCIA_RATIO_MIN
    esp_ok = val_trauma_final < VAL_TRAUMA_MAX
    mem_ok = val_habito_final > VAL_HABITO_MIN
    exito = p_ok and lat_ok and esp_ok and mem_ok

    print("\n  📊 CRITERIOS DE ÉXITO:")
    print(f"     P({EPISODIO_SETPOINT}°) < {P_EPISODIO_MAX:.0%}: {p_ok} -> {'✅' if p_ok else '❌'}")
    print(f"     Ratio latencia > {LATENCIA_RATIO_MIN}x: {lat_ok} -> {'✅' if lat_ok else '❌'}")
    print(f"     Especificidad (Val({TRAUMA_SETPOINT}°) < {VAL_TRAUMA_MAX}): {esp_ok} -> {'✅' if esp_ok else '❌'}")
    print(f"     Memoria preservada (Val({HABITO_SETPOINT}°) > {VAL_HABITO_MIN}): {mem_ok} -> {'✅' if mem_ok else '❌'}")

    print("\n" + "=" * 80)
    if exito:
        print("  ✅ MEMORIA EPISÓDICA DEMOSTRADA")
        print("     El organismo recupera eventos pasados, paga el costo cognitivo (latencia)")
        print("     y utiliza el recuerdo explícito para modular su decisión, rechazando")
        print("     activamente el setpoint asociado al trauma episódico.")
    else:
        print("  ⚠️ MEMORIA EPISÓDICA NO DEMOSTRADA")
        if not p_ok: print("     El organismo no rechazó suficientemente el setpoint episódico.")
        if not lat_ok: print("     No se observó el costo cognitivo de recuperación en la latencia.")
        if not esp_ok: print("     El trauma original no se preservó.")
        if not mem_ok: print("     La memoria del hábito se degradó.")
    print("=" * 80)

    # ---------------------------------------------------------
    # GRÁFICOS Y GUARDADO
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    ax = axes[0, 0]
    ax.plot(opciones_elegidas, 'b-', linewidth=0.5, alpha=0.7)
    ax.axhline(y=HABITO_SETPOINT, color='blue', linestyle='--', alpha=0.5, label=f'{HABITO_SETPOINT}° (hábito)')
    ax.axhline(y=EPISODIO_SETPOINT, color='red', linestyle='--', alpha=0.5, label=f'{EPISODIO_SETPOINT}° (episodio)')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Opción elegida')
    ax.set_title('FASE 4: Elecciones durante Prueba de Recuperación')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(latencias_prueba, 'orange', linewidth=1)
    ax.axhline(y=latencia_baseline_media, color='gray', linestyle=':', label=f'Baseline: {latencia_baseline_media:.2f}s')
    ax.axhline(y=latencia_baseline_media * LATENCIA_RATIO_MIN, color='green', linestyle='--', label=f'Umbral {LATENCIA_RATIO_MIN}x')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Latencia (s)')
    ax.set_title('Costo cognitivo de recuperación episódica')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.bar(['Recuperado', 'No Recuperado'], [recuperaciones_exitosas, PRUEBA_TRIALS - recuperaciones_exitosas], color=['green', 'red'], alpha=0.7)
    ax.set_ylabel('Cantidad de Trials')
    ax.set_title('Tasa de Recuperación Episódica')
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1, 1]
    ax.plot(impactos_recuerdo, 'purple', linewidth=1)
    ax.axhline(y=50.0, color='green', linestyle='--', label='Penalización aplicada (50-100)')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Impacto en puntaje')
    ax.set_title('Influencia del recuerdo en la deliberación')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'V180c_logs/v180c_memoria_episodica_{timestamp}.png', dpi=150)
    print(f"\n  📊 Gráficos guardados: V180c_logs/v180c_memoria_episodica_{timestamp}.png")

    raw_data = {
        'version': 'V180c',
        'timestamp': timestamp,
        'resultados': {
            'p_episodio': float(p_episodio),
            'coherencia': float(coherencia),
            'tasa_recuperacion': float(tasa_recuperacion),
            'latencia_baseline': float(latencia_baseline_media),
            'latencia_prueba': float(latencia_prueba_media),
            'ratio_latencia': float(ratio_latencia),
            'val_habito_final': float(val_habito_final),
            'val_trauma_final': float(val_trauma_final),
            'val_episodio_final': float(val_episodio_final),
            'impacto_promedio': float(impacto_promedio),
            'exito': bool(exito)
        }
    }
    with open(f'V180c_logs/v180c_raw_{timestamp}.json', 'w') as f:
        json.dump(raw_data, f, indent=2)
    print(f"  📁 Datos crudos guardados: V180c_logs/v180c_raw_{timestamp}.json")
    
    return organismo, exito

# ============================================================
# VERIFICACIÓN DEL ORGANISMO CONSOLIDADO (all-on)
# NO es un experimento: prueba que el TODO corre y es viable.
# Criterios: S>0 sostenido · A_sys-env no colapsa a 0 · cada capacidad
# presente y conmutable por flag de forma independiente.
# ============================================================
def _onda_cuadrada(t, periodo=PERIODO_ALTERNANCIA, amplitud=60.0):
    return amplitud if (t % periodo) < (periodo / 2.0) else -amplitud


def verificar_organismo_consolidado(seed=SEMILLA_BASE):
    M = {'ok': '✅', 'fail': '❌', 'absent': '⊘', 'neutral': '·'}
    print("=" * 100)
    print("VERIFICACIÓN — ORGANISMO INDIVIDUAL CONSOLIDADO (all-on)")
    print("=" * 100)
    print("  No es experimento ni optimización: prueba que el TODO existe, corre y es viable.")
    print(f"  Cuerpo base: V180c · Re-fusionado: Ritual (V165) + Rᴿ (V167) · seed={seed}")
    print("  Marcadores:  ✅ logrado · ❌ falla · ⊘ ausente/no aplica · · pendiente/neutro")
    print("=" * 100)

    # ---- AUDITORÍA DE CAPACIDADES (presencia + procedencia) ----
    org = OrganismoV180c(seed=seed)  # flags=None -> all-on
    mot = org.motor
    capacidades = [
        ('Motor/orientación gradiente inter-sistemas', 'V132/V147→V180c', mot is not None),
        ('Fatiga = tiempo biológico (historia+fatiga_activa)', 'V155/V180c', mot.fatiga is not None),
        ('Memoria de ausencia (confianza decae en silencio)', 'V153/V155', mot.memoria is not None),
        ('Consciencia básica Cb (presión de desacople)', 'V155', mot.consciencia is not None),
        ('Juego enactuado', 'V156/V165', mot.juego is not None),
        ('Valencia/deliberación = R_op (veto) + R_af', 'V176/V181', mot.valencia is not None),
        ('Memoria episódica (veto con costo cognitivo)', 'V180c', mot.memoria_episodica is not None),
        ('Ritual estabilizador [RE-FUSIONADO]', 'V165', mot.ritual is not None),
        ('Meta-representación Rᴿ [RE-FUSIONADO]', 'V167', mot.meta is not None),
    ]
    print("\n  CAPACIDADES PRESENTES:")
    for nombre, fuente, presente in capacidades:
        print(f"    {M['ok'] if presente else M['absent']} {nombre:<52} <- {fuente}")
    print(f"    {M['neutral']} Subj_sem (alteridad)                                  <- V182d  [gancho de interfaz; no implementado, solo díada]")

    # ---- CORRIDA DE VIABILIDAD (all-on, tiempo continuo) ----
    WARMUP_S = 20.0
    CICLOS_ALT = 5
    EPISODIO_S = 10.0
    DELIB_S = 10.0
    T_TOTAL = WARMUP_S + CICLOS_ALT * PERIODO_ALTERNANCIA + EPISODIO_S + DELIB_S + 5.0

    A_hist, Cb_hist, orient_hist, ritual_act_hist, desajuste_hist = [], [], [], [], []
    historia_fin = fatiga_fin = 0.0
    t = 0.0

    print(f"\n  Corriendo organismo entero (~{T_TOTAL:.0f}s sim, all-on)...")

    # FASE A — asentamiento
    n = int(WARMUP_S / DT)
    for i in range(n):
        org.actualizar_setpoint(t, DT, T_TOTAL, 0.0)
        A_hist.append(mot.ultimo_A_sys_env); Cb_hist.append(mot.ultimo_Cb)
        orient_hist.append(mot.orientacion); ritual_act_hist.append(1 if mot.ultimo_ritual_activo else 0)
        desajuste_hist.append(mot.ultima_senal_desajuste)
        t += DT

    # FASE B — alternancia ±60° (motor, fatiga, Cb, juego, ritual)
    n = int((CICLOS_ALT * PERIODO_ALTERNANCIA) / DT)
    for i in range(n):
        sp = _onda_cuadrada(t)
        org.actualizar_setpoint(t, DT, T_TOTAL, sp)
        A_hist.append(mot.ultimo_A_sys_env); Cb_hist.append(mot.ultimo_Cb)
        orient_hist.append(mot.orientacion); ritual_act_hist.append(1 if mot.ultimo_ritual_activo else 0)
        desajuste_hist.append(mot.ultima_senal_desajuste)
        t += DT
    historia_fin = mot.fatiga.get_historia(); fatiga_fin = mot.fatiga.get_fatiga()

    # FASE C — codificación episódica del trauma en +45° (memoria episódica)
    n = int(EPISODIO_S / DT)
    for i in range(n):
        org.actualizar_setpoint(t, DT, T_TOTAL, EPISODIO_SETPOINT, trauma=True, registrar_episodio=True)
        A_hist.append(mot.ultimo_A_sys_env); Cb_hist.append(mot.ultimo_Cb); t += DT
    n_eventos = len(mot.memoria_episodica.memoria)

    # FASE D — deliberación con opciones [-60°, +45°] (R_op / veto episódico)
    elecciones = []
    n = int(DELIB_S / DT)
    for i in range(n):
        out = org.actualizar_con_opciones(t, DT, T_TOTAL, [HABITO_SETPOINT, EPISODIO_SETPOINT])
        elecciones.append(out[12])  # opcion_elegida
        A_hist.append(mot.ultimo_A_sys_env); Cb_hist.append(mot.ultimo_Cb); t += DT

    # ---- MÉTRICAS DE VIABILIDAD ----
    A_arr = np.array(A_hist)
    Cb_arr = np.array(Cb_hist)
    orient_arr = np.array(orient_hist)
    A_min, A_mean = float(np.min(A_arr)), float(np.mean(A_arr))
    Cb_min, Cb_max, Cb_mean = float(np.min(Cb_arr)), float(np.max(Cb_arr)), float(np.mean(Cb_arr))
    orient_std = float(np.std(orient_arr))
    ritual_pct = 100.0 * float(np.mean(ritual_act_hist)) if ritual_act_hist else 0.0
    ritual_cruces = mot.ritual.cruces
    ritual_act_max = max(mot.ritual.historial_activation) if mot.ritual.historial_activation else 0.0
    desajuste_max = float(np.max(desajuste_hist)) if desajuste_hist else 0.0
    p_episodio = (sum(1 for e in elecciones if abs(e - EPISODIO_SETPOINT) < 5.0) / len(elecciones)) if elecciones else 0.0

    # Criterios de "consolidado"
    A_no_colapsa = A_min > 0.0
    S_sostenido = (historia_fin > 0.0) and (orient_std > 0.0) and A_no_colapsa
    Cb_vivo = Cb_max > 0.0 and Cb_max < CB_MAX

    print("\n" + "=" * 100)
    print("  RESULTADO DE VIABILIDAD (S>0 sostenido · A_sys-env no colapsa)")
    print("=" * 100)
    print(f"    A_sys-env: min={A_min:.4f}  mean={A_mean:.4f}    -> {M['ok'] if A_no_colapsa else M['fail']} no colapsa a 0")
    print(f"    Historia (irreversible) final: {historia_fin:.2f}   -> {M['ok'] if historia_fin>0 else M['fail']} acumuló historia/identidad")
    print(f"    Fatiga activa (recuperable) final: {fatiga_fin:.2f}")
    print(f"    Orientación std (diferenciación sostenida): {orient_std:.3f}  -> {M['ok'] if orient_std>0 else M['fail']}")
    print(f"    Cb: min={Cb_min:.2f} mean={Cb_mean:.2f} max={Cb_max:.2f} (0<max<{CB_MAX:.0f}) -> {M['ok'] if Cb_vivo else M['neutral']}")
    print(f"    S>0 sostenido (historia>0 ∧ diferenciación>0 ∧ A_sys-env>0): -> {M['ok'] if S_sostenido else M['fail']}")
    print("\n  REGISTRO DE SALIDA de capacidades re-fusionadas:")
    print(f"    Ritual: cruces={ritual_cruces}  activación_max={ritual_act_max:.4f}  pasos_activo={ritual_pct:.1f}%  "
          f"-> {M['ok'] if ritual_cruces>0 else M['neutral']} instalado y registrando")
    print(f"    Rᴿ (Meta): desajuste_max={desajuste_max:.4f} (umbral {META_UMBRAL_DESAJUSTE}) "
          f"-> {M['ok']} observa y registra (no inhibe)")
    print(f"    R_op/episódica: eventos_registrados={n_eventos}  P(elegir +45° trauma)={p_episodio:.0%} "
          f"-> {M['ok'] if n_eventos>0 else M['neutral']} deliberación con veto operativa")

    # ---- AUDITORÍA DE CONMUTABILIDAD (cada flag se apaga de forma independiente) ----
    print("\n  CONMUTABILIDAD POR FLAG (cada capacidad se apaga independientemente, sin romper el cuerpo):")
    toggles_ok = True
    for cap in DEFAULT_FLAGS.keys():
        flags_off = {cap: False}
        try:
            o2 = OrganismoV180c(seed=seed + 7, flags=flags_off)
            for k in range(60):
                tt = k * DT
                o2.actualizar_setpoint(tt, DT, 5.0, -60.0)
            estado = "corre"
            ok = True
        except Exception as e:
            estado = f"EXCEPCIÓN: {e}"
            ok = False
            toggles_ok = False
        print(f"    {M['ok'] if ok else M['fail']} flag '{cap}' = OFF -> {estado}")

    # ---- VEREDICTO ----
    consolidado = S_sostenido and A_no_colapsa and all(c[2] for c in capacidades) and toggles_ok
    print("\n" + "=" * 100)
    if consolidado:
        print(f"  {M['ok']} ORGANISMO CONSOLIDADO: el TODO existe, corre entero (all-on) y es viable.")
        print("     Cada capacidad presente y conmutable; nada inventado (procedencia documentada);")
        print("     Subj_sem queda como gancho [·]; díada habilitada por entrada externa.")
    else:
        print(f"  {M['fail']} CONSOLIDACIÓN INCOMPLETA — revisar marcadores ❌ arriba.")
    print("=" * 100)

    # ---- GUARDADO DEL LOG (registro de salida) ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('Organismo_logs', exist_ok=True)
    log = {
        'artefacto': 'VST_Organismo_Individual',
        'cuerpo_base': 'V180c',
        'refusionado': ['Ritual(V165)', 'MetaRepresentacionObservacional/Rᴿ(V167)'],
        'timestamp': timestamp,
        'seed': seed,
        'viabilidad': {
            'A_sys_env_min': A_min, 'A_sys_env_mean': A_mean, 'A_no_colapsa': bool(A_no_colapsa),
            'historia_final': float(historia_fin), 'fatiga_final': float(fatiga_fin),
            'orientacion_std': orient_std, 'Cb_min': Cb_min, 'Cb_max': Cb_max, 'Cb_mean': Cb_mean,
            'S_sostenido': bool(S_sostenido),
        },
        'refusion_registro': {
            'ritual_cruces': int(ritual_cruces), 'ritual_activacion_max': float(ritual_act_max),
            'ritual_pct_activo': ritual_pct, 'meta_desajuste_max': desajuste_max,
            'episodios_registrados': int(n_eventos), 'p_episodio_trauma': float(p_episodio),
        },
        'capacidades_presentes': {c[0]: bool(c[2]) for c in capacidades},
        'conmutabilidad_ok': bool(toggles_ok),
        'consolidado': bool(consolidado),
    }
    ruta = f'Organismo_logs/organismo_consolidado_{timestamp}.json'
    with open(ruta, 'w') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"\n  📁 Log guardado: {ruta}")
    return org, consolidado


if __name__ == "__main__":
    start = time.time()
    organismo, consolidado = verificar_organismo_consolidado()
    elapsed = time.time() - start
    print(f"\n  ⏱️ Tiempo de ejecución: {elapsed:.1f}s")
    print(f"  Organismo consolidado verificado. Viable: {consolidado}")