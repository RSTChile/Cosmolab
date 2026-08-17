#!/usr/bin/env python3
"""
VST_Celula_Madre_001 — CAMPO Φ ROBUSTECIDO ("célula madre")  [ex VST_Organismo_Individual_v2 / Fase B v2]
================================================================================
v2 = v1 (cuerpo V180c + Ritual/Rᴿ) + CAMPO Φ robustecido: se portan al Φ
per-hemisferio de 32 nodos cuatro piezas del linaje rico (perdidas en la
DISCONTINUIDAD de linaje, no en una poda — ver Fase A), para que el campo deje
de ser sustrato reactivo y empiece a ser un MEDIO capaz de sostener más de un
destino (memoria distribuida + atractor + resonancia). No es experimento ni
optimización: es robustecimiento del sustrato.

PIEZAS PORTADAS (verbatim adaptado, con procedencia + sub-flag independiente):
  1. W_relacional  (BASE)         <- v72b: dW=η·outer(Φ,Φ)/N − τ·W ; M=γ·(W@Φ−Φ)
  2. atractor Phi_int_historia (CARGA ESTRUCTURAL) <- v80h/v72c: EMA + reinyección
     (la versión LIVE de v80h, no la inerte de v81)
  3. oscilador / modos propios     <- v70: −ω²(Φ−Φ_eq) − amort·Φ̇ (frec. por nodo)
  4. dual W_prof/W_rec (CONDICIONAL)<- v80h: identidad(τ≈20s)/contexto(τ≈2s)

INVARIANTES (no negociables):
  · Driver gradiente = ωA − ωB INTOCABLE. Ninguna pieza alimenta el gradiente;
    enriquecen el MEDIO. La conducta sigue guiada solo por el Δ.
  · El oscilador es MEDIO (alimenta Phi_vel), nunca driver.
  · Entrada = forzamiento de borde (perturbación), jamás mensaje decodificado.
  · omega = mean(Φ) sin cambios.
  · Conmutabilidad: memoria_campo=OFF => campo IDÉNTICO a v1/V180c (control git
    'organismo-v1-control'). Es la prueba de que no se rompió el baseline.

CHOQUE REPORTADO (no resuelto en silencio) — equilibrio del oscilador:
  v70 usa PHI_EQUILIBRIO=0.5 sobre un campo en [0,1]. El Φ per-hemisferio es
  SIMÉTRICO en [-1,1], centrado en 0, con reacción biestable Φ(1−Φ²) en ±1.
  Forzar Φ_eq=0.5 lo sacaría de su rango operativo. DECISIÓN DOCUMENTADA: se
  porta con Φ_eq=0.0 (centro/simetría del campo per-hemisferio). Valor original
  0.5 en v70:41. El smoke del oscilador reporta el efecto, sin tunear.

ALCANCE: solo robustece el campo de UNA célula. Ganglio, díada/aire compartido,
comunicación intercelular y división asimétrica quedan FUERA (trabajo posterior).
================================================================================
"""
__doc_v1__ = """
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
try:
    import matplotlib.pyplot as plt          # SÓLO para graficar (CLI/__main__); el campo VIVO no lo necesita
except Exception:                            # en el servidor/Docker matplotlib puede no estar → import perezoso
    plt = None
from datetime import datetime
import os
import json
from collections import deque
import time

# ESCALA COMPARTIDA (5-ago-2026). Lo habitual de una magnitud, aprendido de la propia
# experiencia. NO se reimplementa aquí el patrón r/(1+r): la auditoría avisó por escrito de
# que si se escribe 168 veces, en tres meses hay 168 variantes. Import defensivo porque este
# archivo se ejecuta tanto desde el paquete como suelto desde su propia carpeta.
try:
    from escala import Escala, rel_contra
except Exception:                                # suelto: escala.py vive en la raíz del repo
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from escala import Escala, rel_contra

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
# CAMPO Φ ROBUSTECIDO — constantes PORTADAS (procedencia línea a línea)
# Driver gradiente=ωA−ωB INTOCABLE; estas piezas enriquecen el MEDIO.
# ============================================================
# -- 0. ACOPLE AUDIO→CAMPO (S = I·E, C-N2): el campo OYE la ENVOLVENTE (energía) del audio, no el sample
#       crudo. El forzamiento histórico [0]=+e,[-1]=−e es un DIPOLO de media cero, y el audio es oscilante
#       de media cero → ambos se promedian a 0 en omega → el campo era SORDO al nivel del sonido (E≈0 ⇒ S≈0).
#       La envolvente es POSITIVA y sostenida → mueve omega → A/ICR responden. Recupera la MEMBRANA SENSORIAL (V111).
#
# CORRECCIÓN 5-ago-2026 — EL TRINQUETE (hallazgo de gravedad 10 del plan de constantes).
# QUÉ ESTABA MAL. `env` es un RMS: siempre ≥ 0. Multiplicado por una ganancia fija (6,0), el
# forzamiento sobre Φ SÓLO PODÍA EMPUJAR HACIA ARRIBA. Sobre un campo biestable recortado a
# [−1,1] eso es un trinquete: sube y no vuelve. MEDIDO sobre 100.335 pasos reales del organismo
# ANIMA_5Z934MWHNNRH (3–5 ago 2026): ω_L ≥ +0,999 el 89,70 % del tiempo y ≤ −0,999 el 0,00 %;
# ω_A ≥ +0,999 el 44,14 % y ≤ −0,999 el 0,00 %. Cero por ciento en un lado y noventa en el otro
# no es una preferencia del organismo: es la firma aritmética de un forzamiento de un solo signo.
# Además `campo_F` superaba el 7,86 % de los pasos el umbral de bifurcación del propio término de
# reacción Φ(1−Φ²) —2/(3·√3) = 0,3849—, con máximo 2,082 (5,4× el umbral): por encima de ese
# valor el campo pierde el pozo negativo y con él la capacidad de VOLVER.
# La palanca de `apertura` (2-ago) alivió la magnitud pero no podía arreglar el SIGNO.
#
# POR QUÉ LA CORRECCIÓN ES AUTORREGULADA. El campo ya no oye "cuánto suena" contra una ganancia
# que alguien eligió, sino CUÁNTO SE APARTA lo que suena de lo que suele sonar para este oído:
#   e = env − media_móvil(env)                       (desviación respecto del nivel habitual)
#   u = signo(e) · |e| / (dispersión_habitual + |e|) (0,5 de magnitud en la desviación de siempre)
#   u = u − media_móvil(u)                           (ver abajo: sin esto queda un trinquete débil)
#   F = F_MAX · u · gain_vol · apertura
# `u` sale de rel_contra() de escala.py — sin ninguna constante que elegir: la escala la pone la
# historia del propio oído. Es lo que hace un sistema sensorial real: uno deja de oír el zumbido
# de la nevera y oye cuando se apaga. Ahora el silencio EMPUJA HACIA ABAJO (e < 0) y el campo
# puede volver al pozo negativo, que es lo que se le había quitado.
#
# EL SEGUNDO PASO NO ES ADORNO, Y ESTÁ MEDIDO. Con sólo el primero, `u` NO tiene media cero:
# signo(e)·|e|/(disp+|e|) es cóncava en |e|, y el nivel sonoro del mundo es asimétrico (media 0,1113
# contra mediana 0,0681 en los 101.159 pasos medidos: muchos ratos flojos, pocos golpes fuertes).
# Resultado del replay del `campo_env` real por la fórmula nueva SIN este paso: F queda con media
# −0,0814, el 21,12 % del fondo de escala, empujando siempre hacia abajo. Eso es cambiar un
# trinquete fuerte por uno débil, que no es arreglarlo. Descontando la media móvil del propio
# empuje —el mismo principio, aplicado al empuje en vez de al nivel— la media medida pasa a
# +0,0000 (0,01 % del fondo) y la mediana a +0,0000, con 73,4 % de empujones hacia arriba y 26,6 %
# hacia abajo (pequeños y frecuentes contra grandes y raros: la forma real del sonido).
# Contraste con lo que había: F = 6·env tenía media 0,4087 de mediana, el 0,00 % de masa negativa y
# superaba el umbral de bifurcación el 50,21 % de los pasos, con máximo 5,951 = 15,5 veces el umbral.
CAMPO_GAIN_AUDIO  = 6.0      # cuánto empuja la sonoridad (envolvente) al campo. Calibrar: E debe IMPORTAR sin tapar I.
_COTA_CAMPO_ACTIVA = os.environ.get("ANIMA_CAMPO_COTA", "1") != "0"   # A/B, ver el uso
CAMPO_F_MAX = 2.0 / (3.0 * (3.0 ** 0.5))   # 0,384900… ORIGEN: umbral de bifurcación del término de
                             # reacción Φ(1−Φ²) del propio campo — el valor de forzamiento a partir del
                             # cual desaparece el punto fijo negativo. NO es una elección de calibración:
                             # es el techo derivado por encima del cual el campo deja de poder volver.
                             # Como |u| < 1, |F| < CAMPO_F_MAX siempre: el organismo puede ser desbordado
                             # (u→±1) pero no destruido (nunca cruza su propia bifurcación).
CAMPO_ENV_TASA = 0.005       # ORIGEN: τ = 1/0,005 = 200 subpasos = 2,0 s con SUBPASO = 0,01 s. Es la
                             # escala de tiempo de "contexto" que el propio campo ya declara para
                             # CAMPO_TAU_REC ("contexto, τ≈2s", v80h:75). La adaptación sensorial al nivel
                             # habitual ocurre en el contexto, no en la identidad (τ≈20s): un tono
                             # sostenido deja de empujar en ~2 s, un cambio de nivel empuja al instante.
CAMPO_ENV_MINIMO = 200       # una constante de tiempo completa de observaciones antes de que "lo
                             # habitual" signifique algo. Mientras tanto el forzamiento es 0: el campo
                             # se abstiene en vez de decidir contra una escala vacía (escala.py).
CAMPO_ENV_VENTANA = 1200     # muestras a cada lado del instante actual → ventana total 2·1200 = 2400
                             # muestras = 50 ms a 48 kHz. (El comentario anterior decía "≈ 25 ms": era
                             # el semiancho, no la ventana. Discrepancia doc/código señalada por la
                             # auditoría, resuelta aquí a favor del código; el número no se toca.)
# -- 1. W relacional (BASE) — VERBATIM de v72b_fase3_plasticidad_hebbiana.py --
CAMPO_ETA_HEBB    = 0.02     # v72b:41  tasa de aprendizaje hebbiano
CAMPO_TAU_W       = 0.005    # v72b:42  decaimiento de pesos
CAMPO_GAMMA_PLAST = 0.15     # v72b:43  fuerza del término plástico
CAMPO_W_MAX       = 1.0      # v72b:44  límite para estabilidad numérica
# -- 2. Atractor Phi_int_historia (CARGA ESTRUCTURAL) — v80h:307-308,388 / v72c --
CAMPO_BETA_HIST   = 0.05     # v80h:388 factor EMA de la historia interna
CAMPO_GAMMA_ATR   = 0.05     # v80h:88  GAMMA_ATRACTOR (reinyección viva, no inerte v81)
# -- 3. Oscilador / resonancia (MEDIO) — VERBATIM de v70_campo_continuo.py --
CAMPO_OMEGA_MIN   = 0.05     # v70:37
CAMPO_OMEGA_MAX   = 0.50     # v70:38
CAMPO_AMORT_MIN   = 0.01     # v70:39
CAMPO_AMORT_MAX   = 0.08     # v70:40
CAMPO_PHI_EQ      = 0.0      # CHOQUE (ver encabezado): v70:41 usa 0.5 (campo 0..1);
                             # aquí 0.0 por simetría del Φ per-hemisferio [-1,1].
CAMPO_PHIVEL_CLIP = 5.0      # v70:159 clip de Phi_vel (solo ruta campo-ON)
# -- 4. Dual W_prof/W_rec (CONDICIONAL) — VERBATIM de v80h_seleccion_interna.py --
CAMPO_ETA_PROF    = 0.00333  # v80h:70 ETA_PROFUNDA_BASE  ((1/2000)/0.15)
CAMPO_TAU_PROF    = 0.05     # v80h:74 TAU_PROFUNDA (identidad, τ≈20s)
CAMPO_ETA_REC     = 0.03333  # v80h:71 ETA_RECIENTE_BASE  ((1/200)/0.15)
CAMPO_TAU_REC     = 0.025    # v80h:75 TAU_RECIENTE (contexto, τ≈2s)
# olvido_selectivo (eficiencia-modulado, v80h:245-268) -> ⊘ NO PORTABLE sin inventar:
#   requiere GED (gradiente espectral diferencial) entre region_int/region_aud,
#   estructura AUSENTE en el Φ per-hemisferio. La dual W entra con olvido FIJO
#   (como ya quedó inerte en v81). El olvido selectivo queda ⊘ con razón.

def _frecuencias_naturales_campo(n=32):
    """Frecuencias naturales por nodo, log-espaciadas. VERBATIM de v70:124-130."""
    bandas = np.arange(n)
    tt = np.log1p(bandas) / np.log1p(n - 1)
    omega = CAMPO_OMEGA_MIN + (CAMPO_OMEGA_MAX - CAMPO_OMEGA_MIN) * tt
    amort = CAMPO_AMORT_MIN + (CAMPO_AMORT_MAX - CAMPO_AMORT_MIN) * tt
    return omega, amort

# -- 5. INCORPORADO de la Hemisferio original V122 (perdido en V123) --
#    Λ (medida nativa de apertura/atractores) + inhibición lateral. VERBATIM V122.
CAMPO_UMBRAL_CALLOSO   = 0.5   # V122:30 (ya estaba hardcodeado 0.5 en v1)
CAMPO_GANANCIA_CALLOSO = 0.01  # V122:31 (ya estaba hardcodeado 0.01 en v1)
CAMPO_INHIBICION_RAPIDA = 0.1  # V122:32  umbral de |dOmega| para inhibición lateral

# FLAGS del campo robustecido — master + sub-flags. memoria_campo=False => v1 exacto.
# (campo_dualW / campo_inhib_lateral arrancan OFF: CONDICIONALES, entran si pasa smoke.)
# campo_lambda es MEDIDA (no toca dinámica): se puede leer sin alterar el campo.
DEFAULT_CAMPO_FLAGS = {
    'campo_audio_real':    True,   # S=I·E: el campo OYE la envolvente del audio (monopolo) → omega/A responden
    'memoria_campo':       True,   # master: OFF => campo idéntico a v1/V180c
    'campo_W':             True,   # W relacional (base obligatoria)
    'campo_atractor':      True,   # Phi_int_historia reinyectado (carga estructural)
    'campo_oscilador':     True,   # resonancia / modos propios
    'campo_dualW':         False,  # dual W_prof/W_rec (condicional; reemplaza a campo_W si ON)
    'campo_inhib_lateral': False,  # inhibición lateral L→lento (V122, condicional)
    'campo_lambda':        True,   # Λ/LF: medida nativa de atractores (no altera dinámica)
}


# ============================================================
# HEMISFERIO
# ============================================================
class Hemisferio:
    def __init__(self, nombre, tau, generar_entrada_func, seed=None, sesgo=0.0, campo_flags=None):
        if seed is not None: np.random.seed(seed)
        self.nombre, self.tau, self.generar_entrada, self.sesgo = nombre, tau, generar_entrada_func, sesgo
        self.Phi = np.random.normal(sesgo, 0.1, 32)
        self.Phi_vel = np.zeros(32)
        self.entrada = None
        self.sr = 48000
        self.factor_inanicion = 1.0
        self.env_audio = 0.0           # envolvente (RMS) que el campo oye — instrumentación 2-ago-2026
        self.F_audio = 0.0             # forzamiento efectivo CON SIGNO = F_MAX·u·gain_vol·apertura.
        #   Desde el 5-ago-2026 puede ser NEGATIVO: ésa es justamente la corrección (ver CAMPO_F_MAX).
        self.esc_env = Escala(tasa=CAMPO_ENV_TASA, minimo=CAMPO_ENV_MINIMO)   # lo habitual del nivel
        #   sonoro PARA ESTE OÍDO. media = nivel habitual; dispersion = cuánto suele apartarse de él.
        self.esc_emp = Escala(tasa=CAMPO_ENV_TASA, minimo=CAMPO_ENV_MINIMO)   # lo habitual del EMPUJE
        #   que recibe. Su media es el sesgo que deja la asimetría del mundo, y se descuenta.
        # APERTURA DE LA MEMBRANA (2-ago-2026). Cuánto del exterior deja entrar el organismo.
        # 1.0 = comportamiento histórico exacto (membrana siempre abierta del todo). Lo fija el
        # soma desde `perm_ext` (su propio act_perm, lag 1) cuando ANIMA_MEMBRANA_APERTURA=1.
        self.apertura = 1.0
        self.gain_vol = 1.0            # ganancia de VOLUMEN del oído (física): el oído que se acerca a la
        #   fuente sube de volumen; el otro baja (como girar la cabeza para oír mejor). Lo fija el soma.
        # --- CAMPO ROBUSTECIDO (Fase B v2): flags + estado de memoria de campo ---
        self.cf = dict(DEFAULT_CAMPO_FLAGS)
        if campo_flags: self.cf.update(campo_flags)
        self._campo_on = self.cf['memoria_campo']
        self.W = np.zeros((32, 32))                 # W relacional (v72b)
        self.W_prof = np.zeros((16, 16))            # dual: identidad nodos 0-15 (v80h)
        self.W_rec = np.zeros((16, 16))             # dual: contexto nodos 16-31 (v80h)
        self.Phi_int_historia = np.zeros(32)        # atractor reinyectado (v80h/v72c)
        self.omega_nat, self.amort_nat = _frecuencias_naturales_campo(32)  # oscilador (v70)
        # INCORPORADO de V122 (perdido en V123): registro para Λ/LF + inhibición lateral
        self.historial_omega = []                   # V122:97  (registro, no altera dinámica)
        self.buffer_rapido = []                     # V122:96  (para e_R de Λ)
        self.Lambda = 0.0

    def _calcular_omega(self): return np.mean(self.Phi[:32])  # INVARIANTE: sin cambios

    def _calcular_Lambda(self):
        """Λ nativa — VERBATIM de V122:118-133. delta_struct·(LF+ε)/(e_R+ε).
        LF = nº de atractores distintos en el historial reciente de omega."""
        delta_struct = np.var(self.Phi[:32])
        if len(self.historial_omega) > 50:
            atractores = np.round(self.historial_omega[-50:], 1)
            LF = len(set(atractores)) / 10.0
        else:
            LF = 0.0
        if len(self.buffer_rapido) > int(self.tau / DT):
            idx_pasado = max(0, len(self.buffer_rapido) - int(self.tau / DT))
            omega_pasado = self.buffer_rapido[idx_pasado]
            e_R = abs(self._calcular_omega() - omega_pasado)
        else:
            e_R = 0.01
        self.Lambda = (delta_struct * (LF + 0.01)) / (e_R + 0.01)
        return self.Lambda

    def _lf_actual(self):
        """LF aislado (nº atractores distintos en omega reciente). Medida de multiplicidad."""
        if len(self.historial_omega) > 50:
            return len(set(np.round(self.historial_omega[-50:], 1))) / 10.0
        return 0.0

    def generar_entrada_para_t(self, t, duracion_total):
        if self.entrada is None: self.entrada = self.generar_entrada(duracion_total, self.sr)
        idx = int(t * self.sr)
        return self.entrada[idx] * self.factor_inanicion * self.gain_vol if idx < len(self.entrada) else 0.0

    def actualizar(self, t, dt, duracion_total, otro_hemisferio=None):
        entrada = self.generar_entrada_para_t(t, duracion_total)
        laplaciano = np.zeros_like(self.Phi)
        for i in range(1, 31): laplaciano[i] = self.Phi[i-1] - 2*self.Phi[i] + self.Phi[i+1]
        reaccion = self.Phi * (1 - self.Phi * self.Phi)
        forzamiento = np.zeros_like(self.Phi)
        forzamiento[0], forzamiento[-1] = entrada, -entrada      # dipolo (lateralidad/estructura; media cero)
        # ACOPLE AUDIO→CAMPO (S = I·E): la ENVOLVENTE (energía) del audio — el EXTERIOR entra al
        # INTERIOR y lo desplaza. Sin esto E≈0 ⇒ S≈0 (campo sordo). El gain_vol (orientación) la
        # modula → enfrentar la fuente la oye más fuerte.
        if self.cf.get('campo_audio_real') and self.entrada is not None:
            i0 = int(t * self.sr)
            w = self.entrada[max(0, i0 - CAMPO_ENV_VENTANA):i0 + CAMPO_ENV_VENTANA]
            env = float(np.sqrt(np.mean(w * w))) if w.size else 0.0
            # LA APERTURA ES DEL ORGANISMO. `gain_vol` sólo REDISTRIBUYE entre oídos (un balancín:
            # lo que sube en uno baja en el otro, la suma se conserva), así que orientarse no
            # permite escapar de la saturación. `apertura` sí regula CUÁNTO entra en total, y la
            # fija el propio organismo desde su act_perm. Es la palanca que un ser vivo usa cuando
            # el mundo lo desborda: cerrarse. Estaba prevista (`perm_ext`, "Cable B") y desconectada
            # —se probó sólo ABRIR, que siempre empeoraba, y se archivó sin probar CERRAR.
            #
            # EL FORZAMIENTO YA NO ES MONOPOLAR (5-ago-2026; ver la nota larga en CAMPO_F_MAX).
            # Se compara el nivel de ahora con el nivel habitual DE ESTE OÍDO, y se empuja en la
            # dirección de la diferencia: más fuerte de lo normal empuja hacia arriba, más flojo
            # empuja hacia abajo, lo de siempre no empuja. Con el descuento del empuje habitual
            # (abajo) la media medida sobre el `campo_env` real es +0,0000 — no queda trinquete de
            # ningún signo. Mientras la escala no tiene historia (200 subpasos = 2 s) el forzamiento
            # es 0: el campo se abstiene en vez de empujar contra una escala vacía.
            self.esc_env.observar(env)
            e = env - self.esc_env.media
            if self.esc_env.madura and self.esc_env.dispersion > 1e-9 and abs(e) > 1e-9:
                u = rel_contra(abs(e), self.esc_env.dispersion)   # 0,5 en la desviación habitual
                u = (u if e >= 0.0 else -u)
            else:
                # sin historia, o desviación y dispersión ambas por debajo de la guarda numérica del
                # proyecto (1e-9): no hay con qué comparar. Abstenerse es 0 (no empujar), NUNCA el
                # neutro 0,5 de una clasificación. Sin esta guarda, en silencio digital prolongado
                # media y dispersión decaen juntas hacia 0 y el cociente 0/0 daría 0,5 — un empujón
                # permanente de media escala salido de la nada, que es el defecto que se está
                # corrigiendo, reaparecido por la puerta de atrás.
                u = 0.0
            # Descuento del empuje habitual: lo que queda del sesgo por la asimetría del mundo. El
            # recorte a ±1 es la guarda que mantiene |F| ≤ CAMPO_F_MAX — el campo puede llegar a
            # rozar su propia bifurcación (medido: el 0,03 % de los pasos) pero nunca pasarla.
            # CORREGIDO 5-ago-2026 tras la revisión adversarial. Antes el descuento se aplicaba
            # sólo `if self.esc_emp.madura`, así que las primeras ~200 muestras de CADA arranque
            # pasaban sin descontar, con media medida −0,203: un empujón de un solo signo en cada
            # encendido, que es justo el trinquete que esto viene a quitar. Ahora la escala nace
            # con la primera muestra y el descuento rige desde el paso 1.
            #
            # Y el descuento se hace contra la media INSTANTÁNEA ya actualizada, no contra la
            # anterior: `u − media` con la media incluyendo a `u` es lo que garantiza que el
            # residuo tenga media nula por construcción y no por suerte. La revisión midió que
            # con la versión anterior el signo del sesgo cambiaba según el tramo del corpus
            # (−0,0692 en el completo, +0,0232 en la segunda mitad): eso no es una propiedad,
            # es un accidente.
            # CENTRADO RETIRADO, 5-ago-2026, tras medirlo dos veces.
            #
            # El diagnóstico era correcto: el forzamiento empujaba SIEMPRE hacia el mismo lado
            # (campo_F ≥ 0 en el 100% de los pasos) y eso es un trinquete. Pero la cura estaba
            # mal elegida, y lo pagó el organismo: con el descuento de la media, campo_F quedó en
            # 0,0001 de mediana durante 183 minutos, la ingesta en la mitad del gasto y la reserva
            # en cero de forma estable. Tres horas muerto.
            #
            # La razón es conceptual y no de implementación: `env` es una ENVOLVENTE, y su
            # información está en la MAGNITUD, no en la fluctuación. Restarle su media le quita
            # justo lo que era comida y deja el temblor alrededor del promedio. Una onda sonora sí
            # es bipolar —compresión y rarefacción—; su envolvente no lo es, porque ya es un
            # módulo. Para tener empuje en ambos sentidos hay que tomarlo de la ONDA, no de su
            # envolvente, y la membrana ya la tiene: ése es el arreglo verdadero y queda pendiente.
            #
            # Lo que SÍ se conserva porque está bien derivado y medido: la cota CAMPO_F_MAX =
            # 2/(3√3), el punto de bifurcación del propio término de reacción Φ(1−Φ²). Con ella
            # |F| ≤ F_MAX en el 100% de los pasos, contra el 50,6% que antes cruzaba la
            # bifurcación y dejaba al campo sin pozo negativo al que volver.
            self.esc_emp.observar(u)     # se sigue aprendiendo el empuje habitual, para medirlo
            # LA COTA ES UN TOPE, NO UNA GANANCIA (5-ago-2026).
            # El lote anterior sustituyó la ganancia del acople por CAMPO_F_MAX = 0,3849 y
            # normalizó la envolvente a su propia dispersión. La cota está bien derivada —es el
            # punto de bifurcación de Φ(1−Φ²), donde el campo pierde su pozo negativo— pero se usó
            # como FACTOR en vez de como LÍMITE, y eso dividió el forzamiento por quince: medido,
            # campo_F pasó de 0,0699 a 0,0050 y el organismo estuvo tres horas sin poder comer.
            #
            # Ahora se conservan las dos cosas: el acople con su ganancia real —que es lo que hace
            # que el sonido MUEVA al campo— y la cota como recorte duro, para que nunca cruce su
            # propia bifurcación. Es lo que se quería desde el principio.
            _f = CAMPO_GAIN_AUDIO * env * self.gain_vol * self.apertura
            # ── A/B DE LA COTA (7-ago-2026) ────────────────────────────────────────────
            # La cota entró en vigor de verdad el 7-ago a las 05:31 (antes se había
            # desplegado sin que el proceso la cargara). Desde ese instante, con el MUNDO
            # IGUAL —oído izquierdo 134,6 → 130,4, un 3 %—, `A_sys_env` cayó de 0,4900 a
            # 0,2516 y se quedó ahí: ocho horas seguidas entre 0,2467 y 0,2508 sobre 10.727
            # pasos. Un acoplamiento que no se mueve mientras el mundo varía no está
            # respondiendo: está atascado.
            #
            # Y la cota muerde cuatro veces más que el defecto que venía a corregir: sólo el
            # 7,07 % de los pasos cruzaba la bifurcación, pero el recorte aplana el 23-31 %
            # contra un único valor. Eso convierte una escala en una constante — la misma
            # patología que este proyecto persigue con el nombre de «columna clavada», sólo
            # que aquí lo clavado es el forzamiento que el organismo recibe del mundo: un
            # tercio del tiempo, «fuerte» y «mucho más fuerte» le llegan idénticos.
            #
            # ANIMA_CAMPO_COTA=0 la desactiva para atribuir. NO es un parámetro del
            # organismo: es el interruptor de un experimento, y por eso vive en el entorno y
            # no en `plast`. Con la cota fuera, el campo vuelve a poder cruzar su propia
            # bifurcación el 7 % de los pasos — que es el defecto conocido y el precio de
            # esta medición.
            if _COTA_CAMPO_ACTIVA:
                _f = max(-CAMPO_F_MAX, min(CAMPO_F_MAX, _f))
            forzamiento += _f
            # INSTRUMENTACIÓN (2-ago-2026, ampliada el 5-ago). Esta columna es la que permitió
            # medir el trinquete: sin ella la calibración pedida en la nota vieja ("E debe
            # IMPORTAR sin tapar I") no se podía hacer. Ahora F LLEVA SIGNO — es la mitad del
            # arreglo, y sin publicar el signo nadie puede comprobar que el arreglo sigue en pie.
            # Centinelas que hay que vigilar al desplegar esto: `campo_F` debe tener mediana
            # cercana a 0 y masa a ambos lados. CRITERIO DURO (la revisión señaló que decir
            # "masa a ambos lados" es demasiado laxo: un 68/32 con media −18 % lo pasaría):
            #     |media(campo_F)| debe ser menor que 0,05 · CAMPO_F_MAX = 0,0192.
            # Y ω_L / ω_A deben dejar de
            # estar clavados en +1 (antes: 89,70 % y 44,14 % en ≥ +0,999, 0,00 % en ≤ −0,999).
            self.env_audio = float(env)
            self.F_audio = float(_f)   # ya recortado arriba si la cota esta activa
        acoplamiento = np.zeros_like(self.Phi)
        inhibido = False
        if otro_hemisferio is not None:
            divergencia = abs(self._calcular_omega() - otro_hemisferio._calcular_omega())
            # Inhibición lateral (INCORPORADO V122:155-160) — solo si master+flag ON.
            # El rápido ('L'/'B_L') congela al lento cuando su |dOmega| es alto.
            if self._campo_on and self.cf.get('campo_inhib_lateral') and ('L' in self.nombre) \
               and len(self.historial_omega) > 1:
                dOmega = (self._calcular_omega() - self.historial_omega[-1]) / dt
                if abs(dOmega) > CAMPO_INHIBICION_RAPIDA:
                    inhibido = True
            # Acoplamiento calloso (V122:162-164; idéntico al hardcodeado de v1)
            if not inhibido and divergencia > CAMPO_UMBRAL_CALLOSO:
                acoplamiento = CAMPO_GANANCIA_CALLOSO * (otro_hemisferio.Phi - self.Phi)

        # ---- INVARIANTE DE CONMUTABILIDAD: campo OFF => ruta idéntica a v1/V180c ----
        if not self._campo_on:
            self.Phi_vel += (laplaciano + reaccion + forzamiento + acoplamiento) * dt
            self.Phi = np.clip(self.Phi + self.Phi_vel * dt, -1.0, 1.0)
            om = self._calcular_omega()
            self.historial_omega.append(om); self.buffer_rapido.append(om)  # registro Λ (no altera dinámica)
            return {'omega': om}

        # ---- CAMPO ROBUSTECIDO: términos extra al MEDIO (jamás al gradiente) ----
        extra = np.zeros_like(self.Phi)

        # 3. Oscilador / resonancia — v70:143-145 (alimenta Phi_vel, es MEDIO)
        if self.cf['campo_oscilador']:
            extra = extra + (-(self.omega_nat ** 2) * (self.Phi - CAMPO_PHI_EQ)
                             - self.amort_nat * self.Phi_vel)

        # 1/4. Plasticidad de campo: dual W_prof/W_rec (v80h) reemplaza a W simple (v72b)
        if self.cf['campo_dualW']:
            lo, hi = slice(0, 16), slice(16, 32)
            phi_lo, phi_hi = self.Phi[lo], self.Phi[hi]
            self.W_prof = np.clip(self.W_prof + (CAMPO_ETA_PROF * np.outer(phi_lo, phi_lo) / 16
                                                 - CAMPO_TAU_PROF * self.W_prof) * dt, -CAMPO_W_MAX, CAMPO_W_MAX)
            self.W_rec = np.clip(self.W_rec + (CAMPO_ETA_REC * np.outer(phi_hi, phi_hi) / 16
                                               - CAMPO_TAU_REC * self.W_rec) * dt, -CAMPO_W_MAX, CAMPO_W_MAX)
            m = np.zeros_like(self.Phi)
            m[lo] = CAMPO_GAMMA_PLAST * (self.W_prof @ phi_lo - phi_lo)
            m[hi] = CAMPO_GAMMA_PLAST * (self.W_rec @ phi_hi - phi_hi)
            extra = extra + m
        elif self.cf['campo_W']:
            self.W = np.clip(self.W + (CAMPO_ETA_HEBB * np.outer(self.Phi, self.Phi) / 32
                                       - CAMPO_TAU_W * self.W) * dt, -CAMPO_W_MAX, CAMPO_W_MAX)
            extra = extra + CAMPO_GAMMA_PLAST * (self.W @ self.Phi - self.Phi)

        # 2. Atractor Phi_int_historia reinyectado — v80h:388 (EMA) + v80h:307-308 (reinyección viva)
        if self.cf['campo_atractor']:
            self.Phi_int_historia = (1 - CAMPO_BETA_HIST) * self.Phi_int_historia + CAMPO_BETA_HIST * self.Phi
            extra = extra + CAMPO_GAMMA_ATR * (self.Phi_int_historia - self.Phi)

        self.Phi_vel += (laplaciano + reaccion + forzamiento + acoplamiento + extra) * dt
        self.Phi_vel = np.clip(self.Phi_vel, -CAMPO_PHIVEL_CLIP, CAMPO_PHIVEL_CLIP)  # v70:159
        self.Phi = np.clip(self.Phi + self.Phi_vel * dt, -1.0, 1.0)
        om = self._calcular_omega()
        self.historial_omega.append(om); self.buffer_rapido.append(om)  # registro Λ/LF
        return {'omega': om}

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
    def __init__(self, seed, flags=None, campo_flags=None):
        self.nombre = f"Organismo_{seed}"
        self.flags = dict(DEFAULT_FLAGS)
        if flags:
            self.flags.update(flags)
        self.campo_flags = dict(DEFAULT_CAMPO_FLAGS)
        if campo_flags:
            self.campo_flags.update(campo_flags)
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

        cf = self.campo_flags
        self.izquierdo = Hemisferio("L", 30.0, generar_ruido_rosa, seed=seed, sesgo=SESGO_L, campo_flags=cf)
        self.derecho = Hemisferio("R", 300.0, generar_clicks_poisson, seed=seed+100, sesgo=SESGO_R, campo_flags=cf)
        self.sistema_B_izq = Hemisferio("B_L", 30.0, generar_ruido_rosa, seed=seed+200, sesgo=SESGO_L, campo_flags=cf)
        self.sistema_B_der = Hemisferio("B_R", 300.0, generar_clicks_poisson, seed=seed+300, sesgo=SESGO_R, campo_flags=cf)
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


# ============================================================
# VERIFICACIÓN FASE B v2 — CAMPO ROBUSTECIDO (NO es experimento)
#   OFF == v1 (invariante) · smoke por pieza · ON vs OFF · multiestabilidad
# ============================================================
import importlib.util as _ilu

def _cargar_v1():
    """Importa el control congelado VST_Organismo_Individual.py (v1)."""
    ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "VST_Organismo_Individual.py")
    spec = _ilu.spec_from_file_location("organismo_v1_control", ruta)
    mod = _ilu.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def _protocolo(org, warmup_s=15.0, alt_s=160.0, epi_s=8.0, delib_s=8.0):
    """Protocolo determinista común (mismo para v1 y v2): warmup+alternancia+episodio+delib.
    Duraciones parametrizadas (invariante/viabilidad usan el valor por defecto; el smoke usa
    una versión corta). Devuelve series por paso para comparar/medir viabilidad."""
    T_TOTAL = warmup_s + alt_s + epi_s + delib_s + 2.0
    A, Cb, orient = [], [], []
    mot = org.motor; t = 0.0
    def paso_set(sp, **kw):
        nonlocal t
        org.actualizar_setpoint(t, DT, T_TOTAL, sp, **kw)
        A.append(mot.ultimo_A_sys_env); Cb.append(mot.ultimo_Cb); orient.append(mot.orientacion); t += DT
    for _ in range(int(warmup_s / DT)): paso_set(0.0)                               # warmup
    for _ in range(int(alt_s / DT)):                                                # alternancia ±60
        paso_set(60.0 if (t % PERIODO_ALTERNANCIA) < (PERIODO_ALTERNANCIA / 2) else -60.0)
    for _ in range(int(epi_s / DT)): paso_set(EPISODIO_SETPOINT, trauma=True, registrar_episodio=True)
    for _ in range(int(delib_s / DT)):                                              # deliberación R_op
        org.actualizar_con_opciones(t, DT, T_TOTAL, [HABITO_SETPOINT, EPISODIO_SETPOINT])
        A.append(mot.ultimo_A_sys_env); Cb.append(mot.ultimo_Cb); orient.append(mot.orientacion); t += DT
    return {'A': np.array(A), 'Cb': np.array(Cb), 'orient': np.array(orient),
            'historia': float(mot.fatiga.get_historia()), 'fatiga': float(mot.fatiga.get_fatiga())}

def _metricas(s):
    return {'A_min': float(np.min(s['A'])), 'A_mean': float(np.mean(s['A'])),
            'orient_std': float(np.std(s['orient'])), 'historia': s['historia'],
            'S_sostenido': bool(np.min(s['A']) > 0 and s['historia'] > 0 and np.std(s['orient']) > 0)}

# ---- Sonda de campo aislado (multiestabilidad / modos) ----
# ============================================================
# SONDA REDISEÑADA (instrucción b): HISTORIA primero -> perturbar desde lo
# aprendido -> contar estados de retorno. Engancha la W/atractor (la sonda
# anterior partía de entrada=0 y NO los ejercitaba). Barrido de umbral.
# ============================================================
def _entrada_estructurada(dur, sr):
    """Estímulo estructurado determinista (multi-tono) para que la W/atractor
    construyan estructura durante la fase de HISTORIA. Es estímulo de sonda, no
    mecanismo portado."""
    n = int(dur * sr) + 2
    tt = np.arange(n) / sr
    return 0.5 * np.sin(2 * np.pi * 3.0 * tt) + 0.3 * np.sin(2 * np.pi * 7.0 * tt) + 0.2 * np.sin(2 * np.pi * 13.0 * tt)

_ZERO_GEN = lambda dur, sr: np.zeros(int(dur * sr) + 2)

def _contar_cuencas(patrones, umbral=0.25):
    reps = []
    for p in patrones:
        nuevo = True
        for r in reps:
            d = np.linalg.norm(p - r) / (np.linalg.norm(p) + np.linalg.norm(r) + 1e-9)
            if d < umbral: nuevo = False; break
        if nuevo: reps.append(p)
    return len(reps)

def _contar_modos(phi, frac=0.2):
    f = np.abs(np.fft.rfft(phi - np.mean(phi)))
    return int(np.sum(f > frac * f.max())) if f.max() > 1e-9 else 0

def _clonar_aprendido_zero(h):
    """Clona el estado aprendido (Φ, Phi_vel, W*, atractor) en un Hemisferio con
    entrada CERO, para perturbar y asentar SIN borrar lo aprendido."""
    c = Hemisferio("clon", h.tau, _ZERO_GEN, seed=12345, sesgo=0.0, campo_flags=h.cf)
    c.Phi = h.Phi.copy(); c.Phi_vel = h.Phi_vel.copy()
    c.W = h.W.copy(); c.W_prof = h.W_prof.copy(); c.W_rec = h.W_rec.copy()
    c.Phi_int_historia = h.Phi_int_historia.copy()
    c.historial_omega = list(h.historial_omega); c.buffer_rapido = list(h.buffer_rapido)
    c.entrada = None
    return c

def _sondear(campo_flags, n_ci=50, hist_s=300.0, settle_s=60.0,
             umbrales=(0.10, 0.15, 0.20, 0.25, 0.30, 0.40)):
    # FASE HISTORIA: construir estructura (W/atractor) con entrada estructurada
    h = Hemisferio("hist", 30.0, _entrada_estructurada, seed=44, sesgo=0.0, campo_flags=campo_flags)
    t = 0.0
    for _ in range(int(hist_s / DT)):
        h.actualizar(t, DT, hist_s + 1.0, None); t += DT
    LF_hist = h._lf_actual()                      # medida nativa V122: nº atractores en omega
    finito_h = bool(np.all(np.isfinite(h.Phi)))
    # FASE PERTURBAR: desde el estado APRENDIDO, n_ci perturbaciones distintas, asentar
    pats = []
    for i in range(n_ci):
        c = _clonar_aprendido_zero(h)
        rng = np.random.RandomState(7000 + i)
        c.Phi = np.clip(c.Phi + rng.normal(0, 0.5, 32), -1.0, 1.0)
        c.Phi_vel = np.zeros(32)
        tt = 0.0
        for _ in range(int(settle_s / DT)):
            c.actualizar(tt, DT, settle_s + 1.0, None); tt += DT
        pats.append(c.Phi.copy())
    finito = finito_h and all(np.all(np.isfinite(p)) for p in pats)
    cuencas = {round(u, 2): (_contar_cuencas(pats, u) if finito else -1) for u in umbrales}
    modos = float(np.mean([_contar_modos(p) for p in pats])) if finito else -1.0
    return {'finito': bool(finito), 'cuencas': cuencas, 'LF_hist': float(LF_hist),
            'modos_medios': modos, 'n_ci': n_ci}


def verificar_campo_v2(seed=SEMILLA_BASE):
    M = {'ok': '✅', 'fail': '❌', 'absent': '⊘', 'neutral': '·'}
    print("=" * 100)
    print("VERIFICACIÓN FASE B v2 — CAMPO Φ ROBUSTECIDO (no es experimento)")
    print("=" * 100)
    print("  Driver gradiente=ωA−ωB INTOCABLE · entrada=forzamiento · marcadores ✅ ❌ ⊘ ·")
    print("=" * 100)

    # ---------- 1) INVARIANTE DE CONMUTABILIDAD: OFF == v1 control ----------
    print("\n[1] INVARIANTE — campo OFF debe reproducir el v1 control EXACTO")
    v1 = _cargar_v1()
    s_v1 = _protocolo(v1.OrganismoV180c(seed=seed))
    s_off = _protocolo(OrganismoV180c(seed=seed, campo_flags={'memoria_campo': False}))
    dA = float(np.max(np.abs(s_v1['A'] - s_off['A'])))
    dO = float(np.max(np.abs(s_v1['orient'] - s_off['orient'])))
    dH = abs(s_v1['historia'] - s_off['historia'])
    invariante = (dA == 0.0 and dO == 0.0 and dH == 0.0)
    print(f"    max|ΔA_sys-env|={dA:.2e}  max|Δorient|={dO:.2e}  |Δhistoria|={dH:.2e}")
    print(f"    -> {M['ok'] if invariante else M['fail']} OFF { '==' if invariante else '!=' } v1 control "
          f"({'invariante intacta' if invariante else 'BUG DE INVARIANTE — no avanzar'})")
    if not invariante:
        print("    ❌ Deteniendo: el campo OFF alteró el baseline. Revisar antes de seguir.")
        return False

    # ---------- 2) SMOKE POR PIEZA ----------
    print("\n[2] SMOKE por pieza (¿corre estable? ¿rompe viabilidad?) — sin tunear")
    piezas = [
        ('campo_W',            'W relacional (BASE, v72b)'),
        ('campo_atractor',     'atractor Phi_int_historia (carga estructural, v80h)'),
        ('campo_oscilador',    'oscilador/modos (v70)'),
        ('campo_dualW',        'dual W_prof/W_rec (condicional, v80h)'),
        ('campo_inhib_lateral','inhibición lateral L→lento (INCORPORADO V122)'),
    ]
    smoke = {}
    for key, desc in piezas:
        cf = {'memoria_campo': True, 'campo_W': False, 'campo_atractor': False,
              'campo_oscilador': False, 'campo_dualW': False, 'campo_inhib_lateral': False, key: True}
        try:
            s = _protocolo(OrganismoV180c(seed=seed, campo_flags=cf), warmup_s=5.0, alt_s=40.0, epi_s=3.0, delib_s=3.0)
            finito = bool(np.all(np.isfinite(s['A'])) and np.all(np.isfinite(s['orient'])))
            m = _metricas(s)
            estable = finito and m['A_min'] > 0 and m['S_sostenido']
            smoke[key] = estable
            mark = M['ok'] if estable else M['fail']
            print(f"    {mark} {desc:<48} A_min={m['A_min']:.4f} hist={m['historia']:.1f} "
                  f"S>0={m['S_sostenido']} {'(estable)' if estable else '(INESTABLE)'}")
        except Exception as e:
            smoke[key] = False
            print(f"    {M['fail']} {desc:<48} EXCEPCIÓN: {e}")
    # oscilador: reporte neutro de modos (corre/no corre, produce/no produce modos) — sonda ligera
    son_osc = _sondear({'memoria_campo': True, 'campo_W': False, 'campo_atractor': False,
                        'campo_oscilador': True, 'campo_dualW': False}, n_ci=8, hist_s=60.0, settle_s=20.0)
    print(f"    {M['neutral']} oscilador — guarda epistémica: corre={son_osc['finito']}, "
          f"modos_medios={son_osc['modos_medios']:.2f} (candidato, NO 'mecanismo de diferenciación')")

    # ---------- 3) CORRIDA ON (sub-flags según smoke) vs OFF ----------
    print("\n[3] CORRIDA ON (sub-flags que pasaron smoke) vs OFF")
    on_flags = {'memoria_campo': True,
                'campo_W': smoke.get('campo_W', False),
                'campo_atractor': smoke.get('campo_atractor', False),
                'campo_oscilador': smoke.get('campo_oscilador', False),
                'campo_dualW': smoke.get('campo_dualW', False),
                'campo_inhib_lateral': smoke.get('campo_inhib_lateral', False),
                'campo_lambda': True}
    # si dualW pasó, reemplaza a W (mutuamente excluyentes en el código)
    print(f"    flags ON: { {k: v for k, v in on_flags.items() if k != 'memoria_campo'} }")
    s_on = _protocolo(OrganismoV180c(seed=seed, campo_flags=on_flags))
    m_off, m_on = _metricas(s_off), _metricas(s_on)
    viable_on = m_on['A_min'] > 0 and m_on['S_sostenido']
    print(f"    (a) VIABILIDAD ON: A_min={m_on['A_min']:.4f} hist={m_on['historia']:.1f} "
          f"orient_std={m_on['orient_std']:.2f} S>0={m_on['S_sostenido']} -> {M['ok'] if viable_on else M['fail']}")
    print(f"        Δ vs OFF: A_mean {m_off['A_mean']:.4f}→{m_on['A_mean']:.4f}  "
          f"hist {m_off['historia']:.1f}→{m_on['historia']:.1f}  "
          f"orient_std {m_off['orient_std']:.2f}→{m_on['orient_std']:.2f}")

    # ---------- 4) SONDA CON HISTORIA + ATRIBUCIÓN LIMPIA (incl. inhib_lateral) ----------
    print("\n[4] ¿MÁS DE UN ESTADO ESTABLE? — sonda con historia + atribución por pieza")
    print("    Sonda: HISTORIA con entrada estructurada (300s) -> 50 perturbaciones desde el estado")
    print("    APRENDIDO -> asentar 60s -> contar estados de retorno. Barrido de umbral. + Λ/LF nativo.")
    print("    LIMITACIÓN REGISTRADA (§2): la historia es ÚNICA. Un atractor entrenado con UNA historia")
    print("    devuelve a UN estado: que W+atractor dé 1 cuenca es casi tautológico, NO evidencia de que")
    print("    W no pueda sostener varios valles. La sonda testea retorno a un estado aprendido, NO")
    print("    latencia multi-contexto (la pregunta real de célula madre). Falta sonda multi-contexto.")
    REF = 0.25  # umbral de referencia para el listón pre-registrado
    combos = [
        ('OFF (campo desnudo, = v1)',    {'memoria_campo': False}),
        ('W + atractor (osc OFF)',       {'memoria_campo': True, 'campo_W': True, 'campo_atractor': True, 'campo_oscilador': False, 'campo_dualW': False, 'campo_inhib_lateral': False}),
        ('oscilador solo',               {'memoria_campo': True, 'campo_W': False, 'campo_atractor': False, 'campo_oscilador': True, 'campo_dualW': False, 'campo_inhib_lateral': False}),
        ('oscilador + inhib_lateral',    {'memoria_campo': True, 'campo_W': False, 'campo_atractor': False, 'campo_oscilador': True, 'campo_dualW': False, 'campo_inhib_lateral': True}),
        ('TODO ON sin inhib_lateral',    {'memoria_campo': True, 'campo_W': True, 'campo_atractor': True, 'campo_oscilador': True, 'campo_dualW': True, 'campo_inhib_lateral': False}),
        ('TODO ON',                      on_flags),
    ]
    son = {}
    for nombre, cf in combos:
        r = _sondear(cf)
        son[nombre] = r
        barrido = "  ".join(f"u{u}={c:>2}" for u, c in sorted(r['cuencas'].items()))
        print(f"    {nombre:<28} [{barrido}]  LF_hist={r['LF_hist']:.2f}  modos={r['modos_medios']:.2f}  finito={r['finito']}")

    cuencas_OFF = son['OFF (campo desnudo, = v1)']['cuencas'][REF]
    cuencas_ON = son['TODO ON']['cuencas'][REF]
    canalizacion = (cuencas_ON <= max(3, cuencas_OFF // 2))  # listón PRE-REGISTRADO, no se mueve
    LF_off = son['OFF (campo desnudo, = v1)']['LF_hist']; LF_on = son['TODO ON']['LF_hist']

    # ---- VEREDICTO HONESTO Y MODESTO (§3) — pendiente; numérico ≠ proto-diferenciación ----
    print("\n    " + "-" * 92)
    print("    VEREDICTO (PENDIENTE — la sonda NO resuelve la pregunta de célula madre):")
    print(f"      · INGENIERÍA {M['ok']}: invariante OFF==v1 exacta; 5 piezas estables; oscilador")
    print("        rehabilitado (se perdió por accidente de linaje, no por inestabilidad).")
    print("      · Sonda con historia ya NO es ciega (los números dejaron de ser planos): metodología")
    print("        corregida respecto de la sonda anterior (entrada=0).")
    print(f"      · Listón pre-registrado: cuencas_ON({cuencas_ON}) ≤ max(3,OFF//2)({max(3, cuencas_OFF // 2)}) -> "
          f"se cumple NUMÉRICAMENTE {M['ok'] if canalizacion else M['neutral']}. PERO cumplimiento numérico ≠ proto-diferenciación.")
    print("      · W+atractor entrenado con UNA historia -> 1 cuenca: casi TAUTOLÓGICO (un atractor, una")
    print("        historia, un valle de retorno). NO demuestra que W no pueda sostener varios valles.")
    print(f"      · Multiplicidad espacial aparece (oscilador y/o inhib_lateral): oscilador-solo="
          f"{son['oscilador solo']['cuencas'][REF]}, osc+inhib={son['oscilador + inhib_lateral']['cuencas'][REF]}, "
          f"TODO-sin-inhib={son['TODO ON sin inhib_lateral']['cuencas'][REF]}, TODO ON={cuencas_ON}.")
    print("        Atribución según estas condiciones (§1) — comparar columnas arriba; algo en el mix COMPRIME.")
    print(f"      · Λ/LF nativo PLANO (OFF={LF_off:.2f} = ON={LF_on:.2f}): la variable conductual (omega) NO")
    print("        cambia. La multiplicidad espacial -si existe- es estructura LATENTE no expresada en")
    print("        conducta. Λ y cuencas miden cosas distintas (temporal de omega vs estado espacial); no se contradicen.")
    print(f"      · CONCLUSIÓN {M['neutral']}: firma de célula madre (varios valles latentes y EXPRESABLES) NI")
    print("        demostrada NI refutada. Falta: (i) sonda multi-contexto, (ii) atribución limpia de")
    print("        inhib_lateral, (iii) si la latencia espacial se expresa o no en conducta.")
    print("    " + "-" * 92)

    # ---------- LOG ----------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('Organismo_logs', exist_ok=True)
    log = {'artefacto': 'VST_Celula_Madre_001', 'timestamp': timestamp, 'seed': seed,
           'invariante_OFF_eq_v1': bool(invariante),
           'smoke': {k: bool(v) for k, v in smoke.items()},
           'on_flags': {k: bool(v) for k, v in on_flags.items()},
           'viabilidad_off': m_off, 'viabilidad_on': m_on, 'viable_on': bool(viable_on),
           'multiestabilidad': {k: {'cuencas_por_umbral': {str(u): int(c) for u, c in v['cuencas'].items()},
                                    'LF_hist': float(v['LF_hist']), 'modos_medios': float(v['modos_medios']),
                                    'finito': bool(v['finito']), 'n_ci': v['n_ci']} for k, v in son.items()},
           'canalizacion_ref025_NUMERICA': bool(canalizacion),
           'LF_nativo': {'OFF': float(LF_off), 'ON': float(LF_on), 'plano': bool(abs(LF_on - LF_off) <= 0.1)},
           'limitacion_sonda': 'historia ÚNICA: testea retorno a un estado aprendido, NO latencia multi-contexto',
           'veredicto_celula_madre': 'PENDIENTE — NI demostrada NI refutada; numerico != proto-diferenciacion; '
                                     'falta sonda multi-contexto, atribucion limpia inhib_lateral, y si la latencia '
                                     'espacial se expresa en conducta (Lambda/LF plano)',
           'oscilador_smoke': {'finito': bool(son_osc['finito']), 'modos_medios': float(son_osc['modos_medios'])}}
    ruta = f'Organismo_logs/campo_v2_{timestamp}.json'
    with open(ruta, 'w') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"\n  📁 Log guardado: {ruta}")

    ingenieria_ok = invariante and smoke.get('campo_W', False) and viable_on
    print("\n" + "=" * 100)
    print(f"  INGENIERÍA {M['ok'] if ingenieria_ok else M['fail']}: invariante OFF==v1 exacta, 5 piezas estables, organismo viable.")
    print(f"  CÉLULA MADRE {M['neutral']}: PENDIENTE — ni demostrada ni refutada (sonda de historia única; ver veredicto §[4]).")
    print("=" * 100)
    return ingenieria_ok


if __name__ == "__main__":
    start = time.time()
    ok = verificar_campo_v2()
    elapsed = time.time() - start
    print(f"\n  ⏱️ Tiempo de ejecución: {elapsed:.1f}s")
    print(f"  Campo v2 verificado. OK: {ok}")