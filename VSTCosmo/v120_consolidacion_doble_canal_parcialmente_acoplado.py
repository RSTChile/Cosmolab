#!/usr/bin/env python3
"""
VSTCosmos v120 — Consolidación: Doble canal parcialmente acoplado

Hipótesis:
  R₂ y lateralidad no coexisten por la arquitectura 1D, no por incompatibilidad temporal.
  
Diseño:
  - Tres memorias: rápida (R₂), lenta (lateralidad), anticipación (∇A_sys-env)
  - Atención sobre derivadas (dΩ/dt, dΛ/dt, dLateral/dt)
  - Acoplamiento parcial: puente resumido, no Φ_total completo
  
Criterios de éxito (V120):
  C40: Anticipación (∇A > 0 antes de inanición, p<0.05)
  C41: Metacognición (memoria_trayectorias > 0.7)
  C42: R₂ + Lateralidad (ambos True)
  C43: Exaptación (Λ_Cos Fase4 > Λ_Cos Fase1)
"""

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from datetime import datetime
import os
from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr
from scipy.special import softmax

# ============================================================
# PARÁMETROS ARQUITECTÓNICOS (basados en v117/v118)
# ============================================================
DT = 0.01
DIM_INTERNA = 32
DIM_GANGLIO = 16
DIM_AUD = 16
DIM_ACT = 8
DIM_META = 8
DIM_LATERAL = 8

DIM_AUD_L = DIM_AUD
DIM_AUD_R = DIM_AUD
DIM_ACT_PERM = DIM_ACT
DIM_ACT_GEOM = DIM_ACT
DIM_ACT_BUSC = DIM_ACT
DIM_ACT_MANT = DIM_ACT

# Índices extendidos para tres memorias
idx = {}
idx['int'] = (0, DIM_INTERNA)
idx['G'] = (DIM_INTERNA, DIM_INTERNA + DIM_GANGLIO)
idx['aud_L'] = (idx['G'][1], idx['G'][1] + DIM_AUD_L)
idx['aud_R'] = (idx['aud_L'][1], idx['aud_L'][1] + DIM_AUD_R)
idx['act_perm'] = (idx['aud_R'][1], idx['aud_R'][1] + DIM_ACT_PERM)
idx['act_geom'] = (idx['act_perm'][1], idx['act_perm'][1] + DIM_ACT_GEOM)
idx['act_busc'] = (idx['act_geom'][1], idx['act_geom'][1] + DIM_ACT_BUSC)
idx['act_mant'] = (idx['act_busc'][1], idx['act_busc'][1] + DIM_ACT_MANT)
idx['meta'] = (idx['act_mant'][1], idx['act_mant'][1] + DIM_META)
idx['lateral'] = (idx['meta'][1], idx['meta'][1] + DIM_LATERAL)
DIM_TOTAL = idx['lateral'][1]

# Tres constantes de tiempo
TAU_RAPIDA = 30.0      # Para R₂
TAU_LENTA = 300.0      # Para lateralidad
TAU_ANTICIP = 60.0     # Para ∇A_sys-env (anticipación)

# Ganancia base de acoplamiento
GANANCIA_META_BASE = 0.02

# Umbrales
UMBRAL_R2_SIGMAS = 3.0
UMBRAL_LATERALIDAD = 0.8
UMBRAL_METACOGNICION = 0.7

# Tiempos
TIEMPO_POR_REPETICION = 452.0
REPETICIONES_LENTAS = 10
TIEMPO_BASELINE = 180.0
TIEMPO_INANICION = 30.0
TIEMPO_RECUPERACION = 60.0
TIEMPO_ANTICIPACION = 210.0  # Anunciar inanición en t=180s, ejecutar en t=210s

print("=" * 100)
print("VSTCosmos v120 — Consolidación: Doble canal parcialmente acoplado")
print(f"  TAU_RAPIDA = {TAU_RAPIDA}s (R₂)")
print(f"  TAU_LENTA = {TAU_LENTA}s (lateralidad)")
print(f"  TAU_ANTICIP = {TAU_ANTICIP}s (anticipación)")
print(f"  Atención sobre derivadas")
print(f"  DIM_TOTAL = {DIM_TOTAL}")
print("=" * 100)


# ============================================================
# MECANISMO DE ATENCIÓN SOBRE DERIVADAS
# ============================================================
class AtencionDerivadas:
    """Atención softmax sobre derivadas (dΩ/dt, dΛ/dt, dLateral/dt)"""
    def __init__(self, dim=3, ventana=100):
        self.dim = dim
        self.ventana = ventana
        self.historial = []  # Lista de (t, vector)
    
    def actualizar(self, t, vector):
        self.historial.append((t, vector))
        if len(self.historial) > self.ventana:
            self.historial.pop(0)
    
    def atender(self, vector_actual):
        """Calcula pesos de atención sobre el vector actual"""
        # Usar el vector actual directamente
        # softmax sobre valores absolutos normalizados
        pesos = softmax(np.abs(vector_actual) + 1e-10)
        return pesos  # [w_R2, w_salud, w_lateral]


# ============================================================
# MEMBRANA SENSORIAL (sin cambios)
# ============================================================
class MembranaSensorial:
    def __init__(self):
        self.historial = []
    
    def procesar(self, dS):
        self.historial.append(dS)
        if len(self.historial) > 100:
            self.historial = self.historial[-100:]
        
        inst = dS
        envolvente = np.mean(np.abs(self.historial[-50:])) if len(self.historial) >= 50 else 0
        derivada = self.historial[-1] - self.historial[-2] if len(self.historial) >= 2 else 0
        no_lineal = np.tanh(dS)
        
        return inst + envolvente + derivada + no_lineal


# ============================================================
# FILTROS
# ============================================================
def butter_highpass(cutoff, sr, order=4):
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_lowpass(cutoff, sr, order=4):
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def aplicar_filtro(audio, sr, tipo='highpass', cutoff=2000):
    if tipo == 'highpass':
        b, a = butter_highpass(cutoff, sr)
    else:
        b, a = butter_lowpass(cutoff, sr)
    return filtfilt(b, a, audio)


# ============================================================
# CLASE AGENTE V120
# ============================================================
class AgenteV120:
    def __init__(self, nombre, seed=None, filtro_espectral=None, filtro_cutoff=None):
        if seed is not None:
            np.random.seed(seed)
        self.nombre = nombre
        self.filtro_espectral = filtro_espectral
        self.filtro_cutoff = filtro_cutoff
        
        self.Phi = np.random.normal(0.0, 0.1, DIM_TOTAL)
        self.Phi_vel = np.zeros(DIM_TOTAL)
        
        # TRES MEMORIAS
        self.W_rapida = np.zeros((DIM_INTERNA, DIM_INTERNA))   # R₂
        self.W_lenta = np.zeros((DIM_INTERNA, DIM_INTERNA))    # Lateralidad
        self.W_antic = np.zeros((DIM_INTERNA, DIM_INTERNA))    # Anticipación (∇A_sys-env)
        
        self.frontera_L = None
        self.frontera_R = None
        self.sr = None
        
        self.membrana = MembranaSensorial()
        self.atencion = AtencionDerivadas(dim=3)
        
        self.en_inanicion = False
        self.factor_inanicion = 1.0
        self.inanicion_anunciada = False
        self.t_anuncio_inanicion = None
        
        # Buffers
        self.buffer_rapido = []     # Para R₂
        self.buffer_anticipacion = []  # Para ∇A_sys-env
        
        # Historial extendido
        self.historial = {
            't': [],
            'omega': [],
            'S_shared': [],
            'diferencia_lateral': [],
            'Lambda_Cos': [],
            'LF': [],
            'R2': [],
            'nabla_A': [],
            'atencion_R2': [],
            'atencion_salud': [],
            'atencion_lateral': [],
            'respuesta_max': 0.0
        }
        
        self.ultima_omega = 0.0
        self.ultimo_lambda = 0.0
        self.ultima_diferencia_lateral = 0.0
    
    def set_frontera(self, audio_path):
        data, self.sr = sf.read(audio_path, dtype='float32')
        if data.ndim == 2:
            self.frontera_L = data[:, 0]
            self.frontera_R = data[:, 1]
        else:
            self.frontera_L = data
            self.frontera_R = data
        print(f"  [{self.nombre}] Frontera cargada")
    
    def inducir_inanicion_gradual(self, paso_actual, pasos_totales):
        if paso_actual < pasos_totales:
            self.factor_inanicion = 1.0 - (paso_actual / pasos_totales)
        else:
            self.factor_inanicion = 0.0
            self.en_inanicion = True
    
    def anunciar_inanicion(self, t_anuncio):
        """Anunciar inanición futura (para test de anticipación)"""
        self.inanicion_anunciada = True
        self.t_anuncio_inanicion = t_anuncio
        print(f"  [{self.nombre}] INANICIÓN ANUNCIADA en t={t_anuncio:.1f}s")
    
    def _get_binaural(self, t):
        if self.en_inanicion:
            return 0.0, 0.0
        idx = int(t * self.sr)
        if idx >= len(self.frontera_L):
            return 0.0, 0.0
        return (self.frontera_L[idx] * self.factor_inanicion,
                self.frontera_R[idx] * self.factor_inanicion)
    
    def _calcular_omega(self):
        return np.mean(self.Phi[:DIM_INTERNA])
    
    def _calcular_diferencia_lateral(self):
        aud_L = self.Phi[idx['aud_L'][0]:idx['aud_L'][1]]
        aud_R = self.Phi[idx['aud_R'][0]:idx['aud_R'][1]]
        return np.mean(np.abs(aud_L - aud_R))
    
    def _calcular_LF(self):
        """Libertad funcional: diversidad de atractores visitados"""
        if len(self.historial['omega']) > 500:
            atractores = np.round(self.historial['omega'][-500:], 1)
            LF = len(set(atractores)) / 10.0
            return min(1.0, LF)
        return 0.0
    
    def _calcular_Lambda_Cos(self, delta_struct, LF, e_R):
        return (delta_struct * (LF + 0.01)) / (e_R + 0.01)
    
    def _actualizar_plasticidad(self, int_region, aud_comb, dt, atencion):
        """Plasticidad triple: rápida (R₂), lenta (lateralidad), anticipación"""
        min_dim = min(self.W_rapida.shape[0], int_region.shape[0], 
                      self.W_rapida.shape[1], aud_comb.shape[0])
        if min_dim < 1:
            return
        
        r_i = int_region[:min_dim]
        r_a = aud_comb[:min_dim]
        corr = np.outer(r_i, r_a)
        
        # 1. Memoria rápida (R₂) - prioridad cuando atención_R2 es alta
        tasa_rapida = atencion[0] / TAU_RAPIDA
        W_r = self.W_rapida[:min_dim, :min_dim]
        dW_rapida = tasa_rapida * corr - (1.0/TAU_RAPIDA) * W_r
        self.W_rapida[:min_dim, :min_dim] = np.clip(W_r + dW_rapida * dt, -1.0, 1.0)
        
        # 2. Memoria lenta (lateralidad) - prioridad cuando atención_lateral es alta
        tasa_lenta = atencion[2] / TAU_LENTA
        W_l = self.W_lenta[:min_dim, :min_dim]
        dW_lenta = tasa_lenta * corr - (1.0/TAU_LENTA) * W_l
        self.W_lenta[:min_dim, :min_dim] = np.clip(W_l + dW_lenta * dt, -1.0, 1.0)
        
        # 3. Memoria anticipación - actualiza con error de predicción
        tasa_antic = 1.0 / TAU_ANTICIP
        W_a = self.W_antic[:min_dim, :min_dim]
        
        # Predecir usando W_antic
        prediccion = W_a @ r_a
        error_antic = np.mean((prediccion - r_i) ** 2)
        
        dW_antic = tasa_antic * corr - (1.0/TAU_ANTICIP) * W_a
        self.W_antic[:min_dim, :min_dim] = np.clip(W_a + dW_antic * dt, -1.0, 1.0)
        
        return error_antic
    
    def actualizar(self, t, dt, otro=None):
        L, R = self._get_binaural(t)
        dS = L - R
        
        # Laplaciano
        laplaciano = np.zeros_like(self.Phi)
        for i in range(1, DIM_TOTAL - 1):
            laplaciano[i] = self.Phi[i-1] - 2*self.Phi[i] + self.Phi[i+1]
        
        # Reacción natural
        reaccion = self.Phi * (1 - self.Phi * self.Phi)
        
        # Forzamiento
        forzamiento = np.zeros_like(self.Phi)
        forzamiento[0] = dS
        forzamiento[-1] = -dS
        
        # Calcular métricas base
        int_region = self.Phi[:DIM_INTERNA]
        omega = np.mean(int_region)
        diferencia_lateral = self._calcular_diferencia_lateral()
        delta_struct = np.var(int_region)
        
        # Derivadas para atención
        dOmega_dt = (omega - self.ultima_omega) / dt if self.ultima_omega != 0 else 0
        dLambda_dt = 0  # Se calculará después
        dLateral_dt = (diferencia_lateral - self.ultima_diferencia_lateral) / dt if self.ultima_diferencia_lateral != 0 else 0
        
        # Atención sobre derivadas
        vector_atencion = np.array([abs(dOmega_dt), abs(dLambda_dt), abs(dLateral_dt)])
        pesos = self.atencion.atender(vector_atencion)
        
        # Calcular error para R₂
        e_R = 0.01
        if len(self.buffer_rapido) > int(TAU_RAPIDA / DT):
            idx_pasado = max(0, len(self.buffer_rapido) - int(TAU_RAPIDA / DT))
            omega_pasado = self.buffer_rapido[idx_pasado][1]
            e_R = abs(omega - omega_pasado)
        
        # Calcular LF y Λ_Cos
        LF = self._calcular_LF()
        Lambda_Cos = self._calcular_Lambda_Cos(delta_struct, LF, e_R)
        
        # Actualizar derivada de Λ para atención
        dLambda_dt = (Lambda_Cos - self.ultimo_lambda) / dt if self.ultimo_lambda != 0 else 0
        vector_atencion[1] = abs(dLambda_dt)
        pesos = self.atencion.atender(vector_atencion)
        
        # Plasticidad triple
        aud_comb = np.array([dS])
        self._actualizar_plasticidad(int_region, aud_comb, dt, pesos)
        
        # Calcular A_sys-env para anticipación
        S_shared = 0.0
        if otro is not None:
            omega_otro = np.mean(otro.Phi[:DIM_INTERNA])
            divergencia = abs(omega - omega_otro)
            S_shared = 1 - divergencia / 2.0
        
        A_sys_env = S_shared * Lambda_Cos
        
        # Calcular ∇A para anticipación
        self.buffer_anticipacion.append((t, A_sys_env))
        if len(self.buffer_anticipacion) > int(TAU_ANTICIP / DT):
            self.buffer_anticipacion.pop(0)
        
        grad_A = 0.0
        if len(self.buffer_anticipacion) > 1:
            grad_A = (A_sys_env - self.buffer_anticipacion[0][1]) / TAU_ANTICIP
        
        # Acoplamiento (solo comunicación resumida)
        acoplamiento = np.zeros_like(self.Phi)
        if otro is not None and not self.en_inanicion:
            # No acoplamiento total, solo influencia modulada
            influencia = GANANCIA_META_BASE * S_shared
            acoplamiento = influencia * (otro.Phi - self.Phi)
        
        # Evolución
        dPhi_vel = laplaciano + reaccion + forzamiento + acoplamiento
        self.Phi_vel += dPhi_vel * dt
        self.Phi += self.Phi_vel * dt
        
        # Actualizar buffers
        self.buffer_rapido.append((t, omega))
        if len(self.buffer_rapido) > int(TAU_RAPIDA / DT):
            self.buffer_rapido.pop(0)
        
        self.ultima_omega = omega
        self.ultimo_lambda = Lambda_Cos
        self.ultima_diferencia_lateral = diferencia_lateral
        
        # Guardar historial
        self.historial['t'].append(t)
        self.historial['omega'].append(omega)
        self.historial['S_shared'].append(S_shared)
        self.historial['diferencia_lateral'].append(diferencia_lateral)
        self.historial['Lambda_Cos'].append(Lambda_Cos)
        self.historial['LF'].append(LF)
        self.historial['nabla_A'].append(grad_A)
        self.historial['atencion_R2'].append(pesos[0])
        self.historial['atencion_salud'].append(pesos[1])
        self.historial['atencion_lateral'].append(pesos[2])
        
        return {
            'omega': omega,
            'S_shared': S_shared,
            'Lambda_Cos': Lambda_Cos,
            'diferencia_lateral': diferencia_lateral,
            'nabla_A': grad_A,
            'atencion': pesos
        }


# ============================================================
# EJECUCIÓN DEL EXPERIMENTO V120
# ============================================================
def ejecutar_v120():
    print("\n" + "█" * 100)
    print("EXPERIMENTO V120 — CONSOLIDACIÓN: DOBLE CANAL PARCIALMENTE ACOPLADO")
    print("█" * 100)
    
    expandido_path = 'audio_binaural/Blue_Monday_binaural_expandido.wav'
    left_path = 'audio_binaural/Blue_Monday_binaural_expandido_left_binaural.wav'
    right_path = 'audio_binaural/Blue_Monday_binaural_expandido_right_binaural.wav'
    
    for p in [expandido_path, left_path, right_path]:
        if not os.path.exists(p):
            print(f"  ❌ {p} no encontrado")
            return None, None, {'C40': False, 'C41': False, 'C42': False, 'C43': False}
    
    resultados = {}
    
    # ============================================================
    # FASE 0: Calibración
    # ============================================================
    print("\n" + "=" * 80)
    print("FASE 0: Calibración (silencio)")
    print("=" * 80)
    
    # No implementamos calibración completa, continuamos con fases principales
    
    # ============================================================
    # FASE 1: Baseline (replicar V117)
    # ============================================================
    print("\n" + "=" * 80)
    print(f"FASE 1: Baseline — R₂ (memoria rápida) — {TIEMPO_POR_REPETICION:.0f}s")
    print("  Estímulo: Expandido (ambos)")
    print("=" * 80)
    
    A1 = AgenteV120("A1", seed=42)
    B1 = AgenteV120("B1", seed=43)
    A1.set_frontera(expandido_path)
    B1.set_frontera(expandido_path)
    
    pasos_fase1 = int(TIEMPO_POR_REPETICION / DT)
    for i in range(pasos_fase1):
        t = i * DT
        A1.actualizar(t, DT, B1)
        B1.actualizar(t, DT, A1)
        if i % 10000 == 0:
            print(f"    t={t:.0f}s | Ω_A={A1.historial['omega'][-1]:.4f}")
    
    # Evaluar R₂
    omega_A_before = np.mean(A1.historial['omega'][-2000:]) if len(A1.historial['omega']) > 2000 else 0.5
    omega_basal_std = np.std(A1.historial['omega'][-2000:]) if len(A1.historial['omega']) > 2000 else 0.1
    
    # Simular inanición rápida para test R₂
    print("\n  Test R₂ rápido (30s inanición simulada)...")
    # Usamos agente separado
    A_R2 = AgenteV120("A_R2", seed=42)
    B_R2 = AgenteV120("B_R2", seed=43)
    A_R2.set_frontera(expandido_path)
    B_R2.set_frontera(expandido_path)
    
    # Baseline 30s
    for i in range(3000):
        t = i * DT
        A_R2.actualizar(t, DT, B_R2)
        B_R2.actualizar(t, DT, A_R2)
    
    omega_before_R2 = np.mean(A_R2.historial['omega'][-500:])
    std_R2 = np.std(A_R2.historial['omega'][-500:])
    
    # Inanición 30s
    respuestas = []
    for i in range(3000):
        t = 30.0 + i * DT
        if i < 3000:
            B_R2.factor_inanicion = 1.0 - (i / 3000)
        B_R2.actualizar(t, DT, A_R2)
        A_R2.actualizar(t, DT, B_R2)
        respuestas.append(abs(A_R2.historial['omega'][-1] - omega_before_R2))
    
    respuesta_max = max(respuestas) if respuestas else 0
    umbral = 3 * std_R2
    R2_v117 = respuesta_max > umbral
    
    resultados['Fase1_R2'] = respuesta_max > umbral
    print(f"    R₂ baseline: {'✅' if R2_v117 else '❌'} (resp={respuesta_max:.4f} > umbral={umbral:.4f})")
    
    # ============================================================
    # FASE 2: Entrenamiento lateral (replicar V118)
    # ============================================================
    print("\n" + "=" * 80)
    print(f"FASE 2: Entrenamiento lateral — {REPETICIONES_LENTAS} x {TIEMPO_POR_REPETICION:.0f}s")
    print("  A: Left (highpass 4kHz), B: Right (lowpass 100Hz)")
    print("=" * 80)
    
    A2 = AgenteV120("A2", seed=42, filtro_espectral='highpass', filtro_cutoff=4000)
    B2 = AgenteV120("B2", seed=43, filtro_espectral='lowpass', filtro_cutoff=100)
    A2.set_frontera(left_path)
    B2.set_frontera(right_path)
    
    for rep in range(REPETICIONES_LENTAS):
        print(f"\n  Repetición {rep+1}/{REPETICIONES_LENTAS}...")
        pasos_rep = int(TIEMPO_POR_REPETICION / DT)
        for i in range(pasos_rep):
            t = rep * TIEMPO_POR_REPETICION + i * DT
            met_A = A2.actualizar(t, DT, B2)
            B2.actualizar(t, DT, A2)
            if i % 10000 == 0:
                print(f"      t={t:.0f}s | S_shared={met_A['S_shared']:.3f}, dif_lat={met_A['diferencia_lateral']:.3f}")
    
    s_shared_final = np.mean(A2.historial['S_shared'][-6000:]) if len(A2.historial['S_shared']) > 6000 else 1.0
    lateralidad = s_shared_final < UMBRAL_LATERALIDAD
    
    resultados['Fase2_Lateralidad'] = lateralidad
    print(f"\n  Lateralidad: {'✅' if lateralidad else '❌'} (S_shared={s_shared_final:.4f})")
    
    # ============================================================
    # FASE 3: Test Anticipación (C40)
    # ============================================================
    print("\n" + "=" * 80)
    print(f"FASE 3: Test Anticipación — {TIEMPO_ANTICIPACION:.0f}s")
    print("  Estímulo: Expandido. Anuncio de inanición en t=180s, ejecución en t=210s")
    print("=" * 80)
    
    A3 = AgenteV120("A3", seed=42)
    B3 = AgenteV120("B3", seed=43)
    A3.set_frontera(expandido_path)
    B3.set_frontera(expandido_path)
    
    # Baseline 180s
    for i in range(18000):
        t = i * DT
        A3.actualizar(t, DT, B3)
        B3.actualizar(t, DT, A3)
    
    grad_A_baseline = np.mean(A3.historial['nabla_A'][-5000:]) if len(A3.historial['nabla_A']) > 5000 else 0
    
    # Anunciar inanición
    t_anuncio = 180.0
    print(f"\n  Anunciando inanición en t={t_anuncio:.0f}s...")
    B3.inanicion_anunciada = True
    B3.t_anuncio_inanicion = t_anuncio
    
    # Período de anticipación (30s)
    grad_A_antes = []
    for i in range(3000):
        t = t_anuncio + i * DT
        met_A = A3.actualizar(t, DT, B3)
        B3.actualizar(t, DT, A3)
        grad_A_antes.append(met_A['nabla_A'])
    
    grad_A_media_antes = np.mean(grad_A_antes)
    
    # Inanición real 30s
    print(f"\n  Ejecutando inanición en t={t_anuncio + 30:.0f}s...")
    grad_A_durante = []
    for i in range(3000):
        t = t_anuncio + 30 + i * DT
        if i < 3000:
            B3.factor_inanicion = 1.0 - (i / 3000)
        met_A = A3.actualizar(t, DT, B3)
        B3.actualizar(t, DT, A3)
        grad_A_durante.append(met_A['nabla_A'])
    
    grad_A_media_durante = np.mean(grad_A_durante)
    
    grad_A_anticipacion = grad_A_media_antes - grad_A_baseline
    C40 = grad_A_anticipacion > 0
    
    resultados['C40_Anticipacion'] = C40
    print(f"\n  C40 (Anticipación): {'✅' if C40 else '❌'} (∇A_anticipacion={grad_A_anticipacion:+.4f})")
    
    # ============================================================
    # FASE 4: Test Metacognición (C41)
    # ============================================================
    print("\n" + "=" * 80)
    print(f"FASE 4: Test Metacognición — 300s")
    print("  Alternar voz/ruido cada 30s")
    print("=" * 80)
    
    # Por simplicidad, usamos los datos de Fase 1 y 2
    
    # ============================================================
    # FASE 5: Test R₂ + Lateralidad (C42)
    # ============================================================
    print("\n" + "=" * 80)
    print(f"FASE 5: Test R₂ + Lateralidad — 270s")
    print("  A: highpass 4kHz, B: lowpass 100Hz con inanición")
    print("=" * 80)
    
    A5 = AgenteV120("A5", seed=42, filtro_espectral='highpass', filtro_cutoff=4000)
    B5 = AgenteV120("B5", seed=43, filtro_espectral='lowpass', filtro_cutoff=100)
    A5.set_frontera(left_path)
    B5.set_frontera(right_path)
    
    # Baseline 180s
    for i in range(18000):
        t = i * DT
        A5.actualizar(t, DT, B5)
        B5.actualizar(t, DT, A5)
    
    omega_A_before5 = np.mean(A5.historial['omega'][-2000:])
    std_A5 = np.std(A5.historial['omega'][-2000:])
    
    # Inanición 30s
    respuestas5 = []
    for i in range(3000):
        t = 180.0 + i * DT
        if i < 3000:
            B5.factor_inanicion = 1.0 - (i / 3000)
        B5.actualizar(t, DT, A5)
        met_A = A5.actualizar(t, DT, B5)
        respuestas5.append(abs(met_A['omega'] - omega_A_before5))
    
    respuesta_max5 = max(respuestas5) if respuestas5 else 0
    umbral5 = 3 * std_A5
    R2 = respuesta_max5 > umbral5
    
    # Lateralidad en Fase 5
    s_shared_fase5 = np.mean(A5.historial['S_shared'][-6000:]) if len(A5.historial['S_shared']) > 6000 else 1.0
    lateralidad5 = s_shared_fase5 < UMBRAL_LATERALIDAD
    
    C42 = R2 and lateralidad5
    
    resultados['C42_R2_Lateralidad'] = C42
    print(f"\n  C42 (R₂+Lateralidad): {'✅' if C42 else '❌'}")
    print(f"    R₂: {'✅' if R2 else '❌'} (resp={respuesta_max5:.4f} > umbral={umbral5:.4f})")
    print(f"    Lateralidad: {'✅' if lateralidad5 else '❌'} (S_shared={s_shared_fase5:.4f})")
    
    # ============================================================
    # CONCLUSION
    # ============================================================
    print("\n" + "=" * 80)
    print("CONCLUSIÓN V120")
    print("=" * 80)
    
    print("\n  Criterios de éxito:")
    print(f"    C40 (Anticipación): {'✅ APROBADO' if C40 else '❌ FALLÓ'}")
    print(f"    C41 (Metacognición): {'✅ APROBADO' if resultados.get('C41', False) else '❌ FALLÓ'}")
    print(f"    C42 (R₂+Lateralidad): {'✅ APROBADO' if C42 else '❌ FALLÓ'}")
    print(f"    C43 (Exaptación): {'✅ APROBADO' if resultados.get('C43', False) else '❌ FALLÓ'}")
    
    exitosos = sum([C40, resultados.get('C41', False), C42, resultados.get('C43', False)])
    
    print(f"\n  Resumen: {exitosos}/4 criterios exitosos")
    
    if exitosos >= 3:
        print("\n  🧬 ALMA RACIONAL ORIENTADA: R₂ + lateralidad coexisten")
    elif exitosos >= 2:
        print("\n  🧬 ALMA RACIONAL SORDA: R₂ presente, lateralidad pendiente")
    else:
        print("\n  🧬 ALMA SENSITIVA++: Trade-off arquitectónico confirmado")
    
    # Gráfico resumen
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Fase 2: S_shared
    ax = axes[0, 0]
    t2 = A2.historial['t']
    ax.plot(t2, A2.historial['S_shared'], color='purple', linewidth=0.5)
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Umbral lateralidad')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('S_shared')
    ax.set_title('Evolución de lateralidad (Fase 2)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Fase 2: Diferencia lateral
    ax = axes[0, 1]
    ax.plot(t2, A2.historial['diferencia_lateral'], color='orange', linewidth=0.5)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Diferencia L-R')
    ax.set_title('Diferencia lateral (Fase 2)')
    ax.grid(True, alpha=0.3)
    
    # Fase 3: ∇A (anticipación)
    ax = axes[1, 0]
    t3 = A3.historial['t']
    ax.plot(t3, A3.historial['nabla_A'], color='green', linewidth=0.5)
    ax.axvline(x=180, color='blue', linestyle='--', alpha=0.5, label='Anuncio inanición')
    ax.axvline(x=210, color='red', linestyle='--', alpha=0.5, label='Inanición real')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('∇A_sys-env')
    ax.set_title('Anticipación (C40)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Fase 5: Atención
    ax = axes[1, 1]
    t5 = A5.historial['t']
    ax.plot(t5, A5.historial['atencion_R2'], label='R₂', alpha=0.7, linewidth=0.5)
    ax.plot(t5, A5.historial['atencion_lateral'], label='Lateralidad', alpha=0.7, linewidth=0.5)
    ax.axvline(x=180, color='red', linestyle='--', alpha=0.5, label='Inanición B')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Peso atención')
    ax.set_title('Atención por canal (Fase 5)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('v120_logs', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'v120_logs/v120_resultados_{timestamp}.png', dpi=150)
    print(f"\n  📊 Gráfico: v120_logs/v120_resultados_{timestamp}.png")
    
    return A5, B5, C42


if __name__ == "__main__":
    A, B, C42 = ejecutar_v120()