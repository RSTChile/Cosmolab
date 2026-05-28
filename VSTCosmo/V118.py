#!/usr/bin/env python3
"""
VSTCosmos v118 — Memoria larga y atención para lateralidad

Rediseño:
  - TAU_PROFUNDA = 300s (memoria de 5 minutos)
  - TAU_RECIENTE = 60s (ventana de integración)
  - Mecanismo de atención softmax sobre historial
  - W_lateral: memoria separada para canales izquierdo/derecho
  - Buffer de estado anterior para comparación temporal
  - Exposición prolongada: 10 repeticiones de Blue Monday (75 min)
"""

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from datetime import datetime
import os
from scipy.signal import butter, filtfilt
from scipy.special import softmax

# ============================================================
# PARÁMETROS RECALIBRADOS CON MEMORIA LARGA
# ============================================================
DT = 0.01
DIM_INTERNA = 32
DIM_GANGLIO = 16
DIM_AUD = 16
DIM_ACT = 8
DIM_META = 8
DIM_LATERAL = 8  # Nueva: memoria lateral para L/R

DIM_AUD_L = DIM_AUD
DIM_AUD_R = DIM_AUD
DIM_ACT_PERM = DIM_ACT
DIM_ACT_GEOM = DIM_ACT
DIM_ACT_BUSC = DIM_ACT
DIM_ACT_MANT = DIM_ACT

# Índices expandidos
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
idx['lateral'] = (idx['meta'][1], idx['meta'][1] + DIM_LATERAL)  # Nueva región
DIM_TOTAL = idx['lateral'][1]

# Constantes de tiempo aumentadas (memoria larga)
TAU_PROFUNDA = 300.0   # 5 minutos de memoria (antes 20s)
TAU_RECIENTE = 60.0    # 1 minuto de memoria reciente (antes 10s)

# Parámetros de atención
ATENCION_VENTANA = 1000  # 10 segundos de historial para atención
ATENCION_DIM = 8

# Acoplamiento
GANANCIA_META_BASE = 0.02

# Umbrales
UMBRAL_R2_SIGMAS = 3.0

# Tiempos
TIEMPO_POR_REPETICION = 452.0
REPETICIONES_BASELINE = 1
REPETICIONES_ENTRENAMIENTO = 10  # 75 minutos total
TIEMPO_BASELINE = 180.0
TIEMPO_INANICION = 30.0
TIEMPO_RECUPERACION = 60.0

print("=" * 100)
print("VSTCosmos v118 — Memoria larga y atención")
print(f"  TAU_PROFUNDA = {TAU_PROFUNDA}s (5 minutos)")
print(f"  TAU_RECIENTE = {TAU_RECIENTE}s (1 minuto)")
print(f"  Atención ventana = {ATENCION_VENTANA*DT:.1f}s")
print(f"  Repeticiones entrenamiento = {REPETICIONES_ENTRENAMIENTO} ({REPETICIONES_ENTRENAMIENTO * TIEMPO_POR_REPETICION/60:.1f} min)")
print(f"  DIM_TOTAL = {DIM_TOTAL}")
print("=" * 100)


# ============================================================
# MECANISMO DE ATENCIÓN
# ============================================================
class MecanismoAtencion:
    """Atención softmax sobre historial de estados"""
    def __init__(self, dim, ventana=1000):
        self.dim = dim
        self.ventana = ventana
        self.historial = []  # Lista de (t, estado)
    
    def actualizar(self, t, estado):
        self.historial.append((t, estado))
        if len(self.historial) > self.ventana:
            self.historial.pop(0)
    
    def atender(self, estado_actual, clave='diferencia_lateral'):
        if len(self.historial) < 10:
            return 0.0
        
        # Calcular similitud entre estado actual y cada estado histórico
        similitudes = []
        for _, estado_hist in self.historial[-self.ventana:]:
            sim = np.dot(estado_actual, estado_hist) / (np.linalg.norm(estado_actual) * np.linalg.norm(estado_hist) + 1e-10)
            similitudes.append(sim)
        
        # Softmax para obtener pesos de atención
        pesos = softmax(np.array(similitudes))
        
        # Atención ponderada (por ahora, solo retornamos la máxima similitud)
        atencion = np.max(pesos)
        
        return atencion


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
# CLASE AGENTE CON MEMORIA LARGA Y ATENCIÓN
# ============================================================
class Agente:
    def __init__(self, nombre, seed=None, filtro_espectral=None, filtro_cutoff=None):
        if seed is not None:
            np.random.seed(seed)
        self.nombre = nombre
        self.filtro_espectral = filtro_espectral
        self.filtro_cutoff = filtro_cutoff
        
        self.Phi = np.random.normal(0.0, 0.1, DIM_TOTAL)
        self.Phi_vel = np.zeros(DIM_TOTAL)
        
        # Memorias con constantes de tiempo largas
        self.W_prof = np.zeros((DIM_INTERNA, DIM_INTERNA))
        self.W_rec = np.zeros((DIM_INTERNA, DIM_INTERNA))
        self.W_lateral = np.zeros((DIM_LATERAL, DIM_LATERAL))  # Memoria lateral
        
        self.frontera_L = None
        self.frontera_R = None
        self.sr = None
        
        self.membrana = MembranaSensorial()
        self.atencion = MecanismoAtencion(ATENCION_DIM, ATENCION_VENTANA)
        
        self.en_inanicion = False
        self.factor_inanicion = 1.0
        
        # Buffer de estado anterior para comparación temporal
        self.buffer_estado = []  # Lista de (t, omega)
        self.buffer_max_size = int(TAU_PROFUNDA / DT)  # 5 minutos de buffer
        
        self.historial = {
            't': [],
            'omega': [],
            'delta_struct': [],
            'LF': [],
            'Lambda_Cos': [],
            'S_shared': [],
            'divergencia': [],
            'e_R': [],
            'dOmega_dt': [],
            'saturacion': [],
            'k_efectivo': [],
            'atencion': [],
            'diferencia_lateral': []
        }
    
    def set_frontera(self, audio_path):
        data, self.sr = sf.read(audio_path, dtype='float32')
        
        if data.ndim == 2:
            canal_L = data[:, 0]
            canal_R = data[:, 1]
        else:
            canal_L = data
            canal_R = data
        
        if self.filtro_espectral == 'highpass':
            canal_L = aplicar_filtro(canal_L, self.sr, 'highpass', self.filtro_cutoff)
            canal_R = aplicar_filtro(canal_R, self.sr, 'highpass', self.filtro_cutoff)
        elif self.filtro_espectral == 'lowpass':
            canal_L = aplicar_filtro(canal_L, self.sr, 'lowpass', self.filtro_cutoff)
            canal_R = aplicar_filtro(canal_R, self.sr, 'lowpass', self.filtro_cutoff)
        
        self.frontera_L = canal_L
        self.frontera_R = canal_R
        print(f"  [{self.nombre}] Frontera: {os.path.basename(audio_path)} [{self.filtro_espectral} {self.filtro_cutoff}Hz] ({len(self.frontera_L)/self.sr:.1f}s)")
    
    def inducir_inanicion_gradual(self, paso_actual, pasos_totales):
        if paso_actual < pasos_totales:
            self.factor_inanicion = 1.0 - (paso_actual / pasos_totales)
        else:
            self.factor_inanicion = 0.0
            self.en_inanicion = True
    
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
        """Calcula la diferencia entre regiones auditivas izquierda y derecha"""
        aud_L = self.Phi[idx['aud_L'][0]:idx['aud_L'][1]]
        aud_R = self.Phi[idx['aud_R'][0]:idx['aud_R'][1]]
        return np.mean(np.abs(aud_L - aud_R))
    
    def _actualizar_plasticidad_larga(self, int_region, aud_comb):
        """Plasticidad con constante de tiempo larga (TAU_PROFUNDA)"""
        min_dim = min(self.W_prof.shape[0], int_region.shape[0], 
                      self.W_prof.shape[1], aud_comb.shape[0])
        if min_dim < 1:
            return
        
        W_p = self.W_prof[:min_dim, :min_dim]
        r_i = int_region[:min_dim]
        r_a = aud_comb[:min_dim]
        
        # Tasa de aprendizaje ajustada a TAU_PROFUNDA
        tasa = 1.0 / TAU_PROFUNDA
        corr = np.outer(r_i, r_a)
        dW = tasa * corr - (1.0/TAU_PROFUNDA) * W_p
        self.W_prof[:min_dim, :min_dim] = np.clip(W_p + dW * DT, -1.0, 1.0)
    
    def _actualizar_plasticidad_reciente(self, int_region, aud_comb):
        """Plasticidad con constante de tiempo corta (TAU_RECIENTE)"""
        min_dim = min(self.W_rec.shape[0], int_region.shape[0], 
                      self.W_rec.shape[1], aud_comb.shape[0])
        if min_dim < 1:
            return 0.0
        
        W_r = self.W_rec[:min_dim, :min_dim]
        r_i = int_region[:min_dim]
        r_a = aud_comb[:min_dim]
        
        pred = np.tanh(W_r @ r_a)
        error = np.mean((pred - r_i) ** 2)
        
        # Tasa adaptativa basada en error
        tasa = min(0.1, 1.0 / TAU_RECIENTE) / (error + 0.01)
        corr = np.outer(r_i, r_a)
        dW = tasa * corr - (1.0/TAU_RECIENTE) * W_r
        self.W_rec[:min_dim, :min_dim] = np.clip(W_r + dW * DT, -1.0, 1.0)
        
        return error
    
    def _actualizar_memoria_lateral(self, diferencia_lateral):
        """Actualiza memoria lateral con integración larga"""
        min_dim = min(self.W_lateral.shape[0], self.W_lateral.shape[1])
        if min_dim < 1:
            return
        
        W_l = self.W_lateral[:min_dim, :min_dim]
        sig = np.tanh(diferencia_lateral)
        
        # Integración lenta de la diferencia lateral
        dW = 0.01 * sig - (1.0/TAU_PROFUNDA) * W_l
        self.W_lateral[:min_dim, :min_dim] = np.clip(W_l + dW * DT, -1.0, 1.0)
    
    def _calcular_metricas(self, otro=None):
        int_region = self.Phi[:DIM_INTERNA]
        omega = np.mean(int_region)
        delta_struct = np.var(int_region)
        
        # e_R: error de predicción con memoria larga
        if len(self.buffer_estado) > int(TAU_RECIENTE / DT):
            # Predecir usando estado de hace TAU_RECIENTE
            idx_pasado = max(0, len(self.buffer_estado) - int(TAU_RECIENTE / DT))
            omega_pasado = self.buffer_estado[idx_pasado][1]
            e_R = abs(omega - omega_pasado)
        elif len(self.historial['omega']) > 1:
            e_R = abs(omega - self.historial['omega'][-1])
        else:
            e_R = 0.01
        
        dOmega = 0.0
        if len(self.historial['omega']) > 1:
            dOmega = abs(self.historial['omega'][-1] - omega) / DT
        
        if len(self.historial['omega']) > 500:
            atractores_recientes = np.round(self.historial['omega'][-500:], 1)
            LF = len(set(atractores_recientes)) / 10.0
        else:
            LF = 0.0
        
        Lambda_Cos = (delta_struct * (LF + 0.01)) / (e_R + 0.01)
        
        S_shared = 0.0
        divergencia = 0.0
        if otro is not None:
            omega_otro = np.mean(otro.Phi[:DIM_INTERNA])
            divergencia = abs(omega - omega_otro)
            S_shared = 1 - divergencia / 2.0
        
        return {
            'omega': omega,
            'delta_struct': delta_struct,
            'dOmega': dOmega,
            'e_R': e_R,
            'LF': LF,
            'Lambda_Cos': Lambda_Cos,
            'S_shared': S_shared,
            'divergencia': divergencia
        }
    
    def actualizar(self, t, dt, otro=None):
        L, R = self._get_binaural(t)
        dS = L - R
        perturbacion = self.membrana.procesar(dS)
        
        # Calcular diferencia lateral para atención
        diferencia_lateral = self._calcular_diferencia_lateral()
        
        # Actualizar atención con el estado actual
        estado_atencion = np.array([diferencia_lateral, perturbacion, np.sin(t), np.cos(t)])
        if estado_atencion.shape[0] < ATENCION_DIM:
            estado_atencion = np.pad(estado_atencion, (0, ATENCION_DIM - len(estado_atencion)))
        self.atencion.actualizar(t, estado_atencion[:ATENCION_DIM])
        peso_atencion = self.atencion.atender(estado_atencion[:ATENCION_DIM])
        
        # Laplaciano
        laplaciano = np.zeros_like(self.Phi)
        for i in range(1, DIM_TOTAL - 1):
            laplaciano[i] = self.Phi[i-1] - 2*self.Phi[i] + self.Phi[i+1]
        
        # Reacción natural
        reaccion = self.Phi * (1 - self.Phi * self.Phi)
        
        # Forzamiento modulado por atención
        forzamiento = np.zeros_like(self.Phi)
        forzamiento[0] = perturbacion * (1 + peso_atencion)
        forzamiento[-1] = -perturbacion * (1 + peso_atencion)
        
        # Acoplamiento adaptativo con memoria lateral
        acoplamiento = np.zeros_like(self.Phi)
        k_efectivo = 0.0
        if otro is not None and not self.en_inanicion:
            divergencia = abs(np.mean(self.Phi[:DIM_INTERNA]) - np.mean(otro.Phi[:DIM_INTERNA]))
            # La memoria lateral influye en el acoplamiento
            influencia_lateral = np.mean(np.abs(self.W_lateral))
            k_efectivo = GANANCIA_META_BASE / (divergencia + 0.05) * (1 + influencia_lateral)
            acoplamiento = k_efectivo * (otro.Phi - self.Phi)
        
        # Evolución
        dPhi_vel = laplaciano + reaccion + forzamiento + acoplamiento
        self.Phi_vel += dPhi_vel * dt
        self.Phi += self.Phi_vel * dt
        
        # Actualizar plasticidades
        int_region = self.Phi[:DIM_INTERNA]
        aud_comb = L - R
        self._actualizar_plasticidad_larga(int_region, np.array([aud_comb]))
        error_rec = self._actualizar_plasticidad_reciente(int_region, np.array([aud_comb]))
        self._actualizar_memoria_lateral(diferencia_lateral)
        
        # Actualizar buffer de estado
        omega = self._calcular_omega()
        self.buffer_estado.append((t, omega))
        if len(self.buffer_estado) > self.buffer_max_size:
            self.buffer_estado.pop(0)
        
        # Observación de saturación
        saturados = np.sum(np.abs(self.Phi) > 1.0)
        
        met = self._calcular_metricas(otro)
        
        self.historial['t'].append(t)
        self.historial['omega'].append(met['omega'])
        self.historial['delta_struct'].append(met['delta_struct'])
        self.historial['LF'].append(met['LF'])
        self.historial['Lambda_Cos'].append(met['Lambda_Cos'])
        self.historial['S_shared'].append(met['S_shared'])
        self.historial['divergencia'].append(met['divergencia'])
        self.historial['e_R'].append(met['e_R'])
        self.historial['dOmega_dt'].append(met['dOmega'])
        self.historial['saturacion'].append(saturados)
        self.historial['k_efectivo'].append(k_efectivo)
        self.historial['atencion'].append(peso_atencion)
        self.historial['diferencia_lateral'].append(diferencia_lateral)
        
        return met


# ============================================================
# EXPERIMENTO V118
# ============================================================
def ejecutar_v118():
    print("\n" + "█" * 100)
    print("EXPERIMENTO V118 — MEMORIA LARGA Y ATENCIÓN")
    print("█" * 100)
    
    expandido_path = 'audio_binaural/Blue_Monday_binaural_expandido.wav'
    left_path = 'audio_binaural/Blue_Monday_binaural_expandido_left_binaural.wav'
    right_path = 'audio_binaural/Blue_Monday_binaural_expandido_right_binaural.wav'
    
    for p in [expandido_path, left_path, right_path]:
        if not os.path.exists(p):
            print(f"  ❌ {p} no encontrado")
            return None, None, False
    
    # ============================================================
    # FASE 1: Baseline con expandido (misma dieta)
    # ============================================================
    print("\n" + "=" * 80)
    print(f"FASE 1: Baseline — Expandido (ambos)")
    print(f"  Duración: {TIEMPO_POR_REPETICION * REPETICIONES_BASELINE:.0f}s")
    print("=" * 80)
    
    A1 = Agente("A1", seed=42)
    B1 = Agente("B1", seed=43)
    A1.set_frontera(expandido_path)
    B1.set_frontera(expandido_path)
    
    pasos_baseline_total = int(TIEMPO_POR_REPETICION * REPETICIONES_BASELINE / DT)
    
    for i in range(pasos_baseline_total):
        t = i * DT
        A1.actualizar(t, DT, B1)
        B1.actualizar(t, DT, A1)
        if i % 10000 == 0 and i > 0:
            print(f"    t={t:.0f}s | Ω_A={A1.historial['omega'][-1]:.4f}, Ω_B={B1.historial['omega'][-1]:.4f}")
    
    print(f"\n  Fase 1 completada. Ω_A final = {A1.historial['omega'][-1]:.4f}")
    
    # ============================================================
    # FASE 2: Entrenamiento lateral (múltiples repeticiones)
    # ============================================================
    print("\n" + "=" * 80)
    print(f"FASE 2: Entrenamiento lateral — A:Left, B:Right")
    print(f"  Repeticiones: {REPETICIONES_ENTRENAMIENTO} x {TIEMPO_POR_REPETICION:.0f}s = {REPETICIONES_ENTRENAMIENTO * TIEMPO_POR_REPETICION/60:.1f} minutos")
    print("=" * 80)
    
    A2 = Agente("A2", seed=42)
    B2 = Agente("B2", seed=43)
    A2.set_frontera(left_path)
    B2.set_frontera(right_path)
    
    for rep in range(REPETICIONES_ENTRENAMIENTO):
        print(f"\n  Repetición {rep+1}/{REPETICIONES_ENTRENAMIENTO}...")
        pasos_rep = int(TIEMPO_POR_REPETICION / DT)
        for i in range(pasos_rep):
            t = rep * TIEMPO_POR_REPETICION + i * DT
            A2.actualizar(t, DT, B2)
            B2.actualizar(t, DT, A2)
            if i % 10000 == 0:
                print(f"      t={t:.0f}s | Ω_A={A2.historial['omega'][-1]:.4f}, Ω_B={B2.historial['omega'][-1]:.4f}, S_shared={A2.historial['S_shared'][-1]:.3f}")
    
    print(f"\n  Fase 2 completada. Ω_A final = {A2.historial['omega'][-1]:.4f}")
    
    # Calcular S_shared en últimos 60s
    s_shared_final = np.mean(A2.historial['S_shared'][-6000:]) if len(A2.historial['S_shared']) > 6000 else np.mean(A2.historial['S_shared'])
    lateralidad = s_shared_final < 0.8
    
    print(f"\n  Resultados Fase 2:")
    print(f"    S_shared medio últimos 60s: {s_shared_final:.4f}")
    print(f"    Lateralidad detectable: {'✅ SI' if lateralidad else '❌ NO'}")
    print(f"    Atención media: {np.mean(A2.historial['atencion'][-6000:]):.4f}")
    print(f"    Diferencia lateral media: {np.mean(A2.historial['diferencia_lateral'][-6000:]):.4f}")
    
    # ============================================================
    # FASE 3: Test de orientación (cambio repentino de dieta)
    # ============================================================
    print("\n" + "=" * 80)
    print(f"FASE 3: Test de orientación")
    print(f"  Baseline: {TIEMPO_BASELINE:.0f}s | Inanición: {TIEMPO_INANICION:.0f}s | Recuperación: {TIEMPO_RECUPERACION:.0f}s")
    print("=" * 80)
    
    A3 = Agente("A3", seed=42)
    B3 = Agente("B3", seed=43)
    A3.set_frontera(left_path)
    B3.set_frontera(right_path)
    
    # Baseline extendido
    pasos_baseline = int(TIEMPO_BASELINE / DT)
    print(f"\n  Baseline ({TIEMPO_BASELINE:.0f}s) con A=Left, B=Right...")
    for i in range(pasos_baseline):
        t = i * DT
        A3.actualizar(t, DT, B3)
        B3.actualizar(t, DT, A3)
        if i % 5000 == 0:
            print(f"    t={t:.0f}s | Ω_A={A3.historial['omega'][-1]:.4f}, Ω_B={B3.historial['omega'][-1]:.4f}")
    
    omega_A_before = np.mean(A3.historial['omega'][-2000:])
    omega_basal_std = np.std(A3.historial['omega'][-2000:])
    print(f"    Ω_A medio baseline (últimos 20s): {omega_A_before:.4f} ± {omega_basal_std:.4f}")
    
    # Inanición de B
    pasos_inanicion = int(TIEMPO_INANICION / DT)
    print(f"\n  Inanición gradual de B ({TIEMPO_INANICION:.0f}s)...")
    respuestas = []
    
    for i in range(pasos_inanicion):
        t = TIEMPO_BASELINE + i * DT
        B3.inducir_inanicion_gradual(i, pasos_inanicion)
        met_A = A3.actualizar(t, DT, B3)
        B3.actualizar(t, DT, A3)
        respuesta = abs(met_A['omega'] - omega_A_before)
        respuestas.append(respuesta)
        if i % 500 == 0:
            print(f"    t={t:.1f}s | Ω_A={met_A['omega']:.4f} (resp={respuesta:.4f}) | factor={B3.factor_inanicion:.2f}")
    
    # Recuperación
    pasos_recuperacion = int(TIEMPO_RECUPERACION / DT)
    print(f"\n  Recuperación ({TIEMPO_RECUPERACION:.0f}s)...")
    B3.en_inanicion = False
    B3.factor_inanicion = 1.0
    
    for i in range(pasos_recuperacion):
        t = TIEMPO_BASELINE + TIEMPO_INANICION + i * DT
        A3.actualizar(t, DT, B3)
        B3.actualizar(t, DT, A3)
        if i % 1000 == 0:
            print(f"    t={t:.1f}s | Ω_A={A3.historial['omega'][-1]:.4f}")
    
    omega_A_after = np.mean(A3.historial['omega'][-2000:])
    respuesta_max = max(respuestas) if respuestas else 0
    umbral = UMBRAL_R2_SIGMAS * omega_basal_std
    
    print(f"\n  Resultados Fase 3:")
    print(f"    Ω_A después: {omega_A_after:.4f}")
    print(f"    ΔA = {omega_A_after - omega_A_before:+.4f}")
    print(f"    Respuesta máxima: {respuesta_max:.4f}")
    print(f"    Umbral ({UMBRAL_R2_SIGMAS}σ): {umbral:.4f}")
    
    R2 = respuesta_max > umbral
    print(f"    R₂: {'✅ CONFIRMADO' if R2 else '❌ NO'}")
    
    # ============================================================
    # CONCLUSION
    # ============================================================
    print("\n" + "=" * 80)
    print("CONCLUSIÓN")
    print("=" * 80)
    
    omega_min = min(A2.historial['omega'])
    omega_max = max(A2.historial['omega'])
    rango = omega_max - omega_min
    sistema_vivo = rango > 0.5
    
    print(f"\n  Estado del sistema:")
    print(f"    Rango Ω en Fase 2: {rango:.3f}")
    print(f"    {'✅ VIVO' if sistema_vivo else '❌ COLMAPSO'}")
    print(f"    Lateralidad (S_shared < 0.8): {'✅' if lateralidad else '❌'}")
    print(f"    R₂: {'✅' if R2 else '❌'}")
    print(f"    Atención media: {np.mean(A2.historial['atencion'][-6000:]):.4f}")
    
    if sistema_vivo and lateralidad and R2:
        print("""
    🧬 ALMA RACIONAL COMPLETA: El sistema distingue lateralidad,
       se mantiene vivo tras entrenamiento prolongado, y responde a la inanición.
    """)
    elif sistema_vivo and R2:
        print("""
    🧬 ALMA RACIONAL CON FUSIÓN: El sistema tiene R₂ pero no distingue lateralidad,
       incluso tras entrenamiento prolongado.
    """)
    elif sistema_vivo and lateralidad:
        print("""
    🧬 ALMA SENSITIVA++: El sistema distingue lateralidad pero no tiene R₂.
    """)
    else:
        print("""
    🧬 ALMA VEGETATIVA: El sistema colapsa incluso con entrenamiento prolongado.
    """)
    
    # GRÁFICOS
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Fase 2: Ω
    ax = axes[0, 0]
    t2 = A2.historial['t']
    ax.plot(t2, A2.historial['omega'], label='A (Left)', alpha=0.7, linewidth=0.5)
    ax.plot(t2[:len(B2.historial['omega'])], B2.historial['omega'], label='B (Right)', alpha=0.7, linewidth=0.5)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Ω')
    ax.set_title('Fase 2: Entrenamiento lateral')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Fase 2: S_shared
    ax = axes[0, 1]
    ax.plot(t2, A2.historial['S_shared'], color='purple', linewidth=0.5)
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Umbral lateralidad (0.8)')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('S_shared')
    ax.set_title('Sentido compartido')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Fase 2: Atención
    ax = axes[0, 2]
    ax.plot(t2, A2.historial['atencion'], color='green', linewidth=0.5)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Atención')
    ax.set_title('Mecanismo de atención')
    ax.grid(True, alpha=0.3)
    
    # Fase 3: Respuesta
    ax = axes[1, 0]
    t_resp = TIEMPO_BASELINE + np.arange(len(respuestas)) * DT
    ax.plot(t_resp, respuestas, color='red', linewidth=1.5)
    ax.axhline(y=umbral, color='green', linestyle='--', label=f'Umbral ({UMBRAL_R2_SIGMAS}σ={umbral:.3f})')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('|ΔΩ_A|')
    ax.set_title('Respuesta a inanición')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Diferencia lateral
    ax = axes[1, 1]
    ax.plot(t2, A2.historial['diferencia_lateral'], color='orange', linewidth=0.5)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Diferencia L-R')
    ax.set_title('Diferencia lateral en campo auditivo')
    ax.grid(True, alpha=0.3)
    
    # Λ_Cos
    ax = axes[1, 2]
    ax.plot(t2, A2.historial['Lambda_Cos'], color='blue', linewidth=0.5)
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Umbral Racional')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Λ_Cos')
    ax.set_title('Salud dinámica')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('v118_logs', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'v118_logs/v118_resultados_{timestamp}.png', dpi=150)
    print(f"\n  📊 Gráfico: v118_logs/v118_resultados_{timestamp}.png")
    
    return A3, B3, R2


if __name__ == "__main__":
    A, B, R2 = ejecutar_v118()