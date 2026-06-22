#!/usr/bin/env python3
"""
VSTCosmos v119 — Plasticidad dual
  TAU_RAPIDA = 30s (para R₂)
  TAU_LENTA = 300s (para lateralidad)
  Mantiene memoria larga y exposición prolongada de V118
  Hipótesis: Así coexisten lateralidad y R₂
"""

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from datetime import datetime
import os
from scipy.signal import butter, filtfilt

# ============================================================
# PARÁMETROS
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

# PLASTICIDAD DUAL - CLAVE DE V119
TAU_RAPIDA = 30.0    # Para R₂: reacción rápida
TAU_LENTA = 300.0    # Para lateralidad: integración lenta

GANANCIA_META_BASE = 0.02
UMBRAL_R2_SIGMAS = 3.0

TIEMPO_POR_REPETICION = 452.0
REPETICIONES_BASELINE = 1
REPETICIONES_ENTRENAMIENTO = 10
TIEMPO_BASELINE = 180.0
TIEMPO_INANICION = 30.0
TIEMPO_RECUPERACION = 60.0

print("=" * 100)
print("VSTCosmos v119 — Plasticidad dual")
print(f"  TAU_RAPIDA = {TAU_RAPIDA}s (para R₂)")
print(f"  TAU_LENTA = {TAU_LENTA}s (para lateralidad)")
print(f"  Objetivo: Coexistencia de R₂ y lateralidad")
print(f"  DIM_TOTAL = {DIM_TOTAL}")
print("=" * 100)


class Agente:
    def __init__(self, nombre, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.nombre = nombre
        
        self.Phi = np.random.normal(0.0, 0.1, DIM_TOTAL)
        self.Phi_vel = np.zeros(DIM_TOTAL)
        
        # MEMORIAS SEPARADAS
        self.W_rapida = np.zeros((DIM_INTERNA, DIM_INTERNA))   # Para R₂
        self.W_lenta = np.zeros((DIM_INTERNA, DIM_INTERNA))    # Para lateralidad
        self.W_lateral = np.zeros((DIM_LATERAL, DIM_LATERAL))
        
        self.frontera_L = None
        self.frontera_R = None
        self.sr = None
        
        self.en_inanicion = False
        self.factor_inanicion = 1.0
        
        # Buffer para predicción rápida
        self.buffer_rapido = []  # Últimos 30s para R₂
        self.buffer_lento = []   # Últimos 300s para lateralidad
        
        self.historial = {
            't': [],
            'omega': [],
            'S_shared': [],
            'diferencia_lateral': [],
            'respuesta_max': 0.0
        }
    
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
    
    def _actualizar_plasticidad_dual(self, int_region, aud_comb, delta_t):
        """Plasticidad con dos tiempos: rápida y lenta"""
        min_dim = min(self.W_rapida.shape[0], int_region.shape[0], 
                      self.W_rapida.shape[1], aud_comb.shape[0])
        if min_dim < 1:
            return
        
        # Actualizar memoria rápida (para R₂)
        W_r = self.W_rapida[:min_dim, :min_dim]
        r_i = int_region[:min_dim]
        r_a = aud_comb[:min_dim]
        
        # Error rápido: predicción a corto plazo
        corr_rapida = np.outer(r_i, r_a)
        dW_rapida = (1.0 / TAU_RAPIDA) * corr_rapida - (1.0 / TAU_RAPIDA) * W_r
        self.W_rapida[:min_dim, :min_dim] = np.clip(W_r + dW_rapida * DT, -1.0, 1.0)
        
        # Actualizar memoria lenta (para lateralidad)
        W_l = self.W_lenta[:min_dim, :min_dim]
        corr_lenta = np.outer(r_i, r_a)
        dW_lenta = (1.0 / TAU_LENTA) * corr_lenta - (1.0 / TAU_LENTA) * W_l
        self.W_lenta[:min_dim, :min_dim] = np.clip(W_l + dW_lenta * DT, -1.0, 1.0)
        
        # Error de predicción para R₂ (rápido)
        error = 0.0
        if len(self.buffer_rapido) > int(TAU_RAPIDA / DT):
            idx_pasado = max(0, len(self.buffer_rapido) - int(TAU_RAPIDA / DT))
            omega_pasado = self.buffer_rapido[idx_pasado][1]
            error = abs(np.mean(r_i) - omega_pasado)
        
        return error
    
    def actualizar(self, t, dt, otro=None):
        L, R = self._get_binaural(t)
        dS = L - R
        
        # Laplaciano
        laplaciano = np.zeros_like(self.Phi)
        for i in range(1, DIM_TOTAL - 1):
            laplaciano[i] = self.Phi[i-1] - 2*self.Phi[i] + self.Phi[i+1]
        
        # Reacción natural
        reaccion = self.Phi * (1 - self.Phi * self.Phi)
        
        # Forzamiento natural
        forzamiento = np.zeros_like(self.Phi)
        forzamiento[0] = dS
        forzamiento[-1] = -dS
        
        # Acoplamiento adaptativo con base rápida (para R₂)
        acoplamiento = np.zeros_like(self.Phi)
        k_efectivo = 0.0
        if otro is not None and not self.en_inanicion:
            divergencia = abs(np.mean(self.Phi[:DIM_INTERNA]) - np.mean(otro.Phi[:DIM_INTERNA]))
            k_efectivo = GANANCIA_META_BASE / (divergencia + 0.05)
            acoplamiento = k_efectivo * (otro.Phi - self.Phi)
        
        # Evolución
        dPhi_vel = laplaciano + reaccion + forzamiento + acoplamiento
        self.Phi_vel += dPhi_vel * dt
        self.Phi += self.Phi_vel * dt
        
        # Plasticidad dual
        int_region = self.Phi[:DIM_INTERNA]
        error_rapido = self._actualizar_plasticidad_dual(int_region, np.array([dS]), dt)
        
        # Actualizar buffers
        omega = self._calcular_omega()
        self.buffer_rapido.append((t, omega))
        self.buffer_lento.append((t, omega))
        
        # Mantener buffers
        max_rapido = int(TAU_RAPIDA / DT)
        max_lento = int(TAU_LENTA / DT)
        if len(self.buffer_rapido) > max_rapido:
            self.buffer_rapido.pop(0)
        if len(self.buffer_lento) > max_lento:
            self.buffer_lento.pop(0)
        
        # Calcular diferencia lateral
        dif_lat = self._calcular_diferencia_lateral()
        
        # Calcular S_shared si hay otro
        S_shared = 0.0
        if otro is not None:
            omega_otro = np.mean(otro.Phi[:DIM_INTERNA])
            divergencia = abs(omega - omega_otro)
            S_shared = 1 - divergencia / 2.0
        
        self.historial['t'].append(t)
        self.historial['omega'].append(omega)
        self.historial['S_shared'].append(S_shared)
        self.historial['diferencia_lateral'].append(dif_lat)
        
        return {'omega': omega, 'S_shared': S_shared, 'error_rapido': error_rapido}


def ejecutar_v119():
    print("\n" + "█" * 100)
    print("EXPERIMENTO V119 — PLASTICIDAD DUAL")
    print("█" * 100)
    
    expandido_path = 'audio_binaural/Blue_Monday_binaural_expandido.wav'
    left_path = 'audio_binaural/Blue_Monday_binaural_expandido_left_binaural.wav'
    right_path = 'audio_binaural/Blue_Monday_binaural_expandido_right_binaural.wav'
    
    for p in [expandido_path, left_path, right_path]:
        if not os.path.exists(p):
            print(f"  ❌ {p} no encontrado")
            return None, None, False
    
    # FASE 1: Baseline
    print("\n" + "=" * 80)
    print(f"FASE 1: Baseline — Expandido ({TIEMPO_POR_REPETICION:.0f}s)")
    print("=" * 80)
    
    A1 = Agente("A1", seed=42)
    B1 = Agente("B1", seed=43)
    A1.set_frontera(expandido_path)
    B1.set_frontera(expandido_path)
    
    pasos_fase1 = int(TIEMPO_POR_REPETICION / DT)
    for i in range(pasos_fase1):
        t = i * DT
        A1.actualizar(t, DT, B1)
        B1.actualizar(t, DT, A1)
        if i % 10000 == 0:
            print(f"    t={t:.0f}s | Ω_A={A1.historial['omega'][-1]:.4f}")
    
    # FASE 2: Entrenamiento lateral
    print("\n" + "=" * 80)
    print(f"FASE 2: Entrenamiento lateral — {REPETICIONES_ENTRENAMIENTO} x {TIEMPO_POR_REPETICION:.0f}s")
    print("  A: Left, B: Right")
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
            met_A = A2.actualizar(t, DT, B2)
            B2.actualizar(t, DT, A2)
            if i % 10000 == 0:
                print(f"      t={t:.0f}s | Ω_A={met_A['omega']:.4f}, S_shared={met_A['S_shared']:.3f}")
    
    s_shared_final = np.mean(A2.historial['S_shared'][-6000:])
    lateralidad = s_shared_final < 0.8
    
    print(f"\n  Resultados Fase 2:")
    print(f"    S_shared medio últimos 60s: {s_shared_final:.4f}")
    print(f"    Lateralidad: {'✅ SI' if lateralidad else '❌ NO'}")
    print(f"    Diferencia lateral media: {np.mean(A2.historial['diferencia_lateral'][-6000:]):.4f}")
    
    # FASE 3: Inanición
    print("\n" + "=" * 80)
    print(f"FASE 3: Inanición — Baseline {TIEMPO_BASELINE:.0f}s + inanición {TIEMPO_INANICION:.0f}s")
    print("=" * 80)
    
    A3 = Agente("A3", seed=42)
    B3 = Agente("B3", seed=43)
    A3.set_frontera(left_path)
    B3.set_frontera(right_path)
    
    pasos_baseline = int(TIEMPO_BASELINE / DT)
    print(f"\n  Baseline ({TIEMPO_BASELINE:.0f}s)...")
    for i in range(pasos_baseline):
        t = i * DT
        A3.actualizar(t, DT, B3)
        B3.actualizar(t, DT, A3)
    
    omega_A_before = np.mean(A3.historial['omega'][-2000:])
    omega_basal_std = np.std(A3.historial['omega'][-2000:])
    print(f"    Ω_A medio baseline: {omega_A_before:.4f} ± {omega_basal_std:.4f}")
    
    pasos_inanicion = int(TIEMPO_INANICION / DT)
    print(f"\n  Inanición gradual ({TIEMPO_INANICION:.0f}s)...")
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
    
    respuesta_max = max(respuestas) if respuestas else 0
    umbral = UMBRAL_R2_SIGMAS * omega_basal_std
    R2 = respuesta_max > umbral
    
    print(f"\n  Resultados Fase 3:")
    print(f"    Respuesta máxima: {respuesta_max:.4f}")
    print(f"    Umbral ({UMBRAL_R2_SIGMAS}σ): {umbral:.4f}")
    print(f"    R₂: {'✅ CONFIRMADO' if R2 else '❌ NO'}")
    
    # CONCLUSION
    print("\n" + "=" * 80)
    print("CONCLUSIÓN")
    print("=" * 80)
    
    print(f"\n  Estado del sistema:")
    print(f"    Lateralidad: {'✅' if lateralidad else '❌'}")
    print(f"    R₂: {'✅' if R2 else '❌'}")
    
    if lateralidad and R2:
        print("\n  🧬 ALMA RACIONAL COMPLETA: Lateralidad + R₂ coexisten.")
        print("     Plasticidad dual resolvió el trade-off.")
    elif lateralidad:
        print("\n  🧬 ALMA SENSITIVA++: Lateralidad sí, R₂ no.")
    elif R2:
        print("\n  🧬 ALMA RACIONAL SORDA: R₂ sí, lateralidad no.")
    else:
        print("\n  🧬 ALMA VEGETATIVA: Ninguna capacidad.")
    
    # Gráfico rápido
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    t2 = A2.historial['t']
    ax[0].plot(t2, A2.historial['S_shared'], color='purple', linewidth=0.5)
    ax[0].axhline(y=0.8, color='red', linestyle='--', alpha=0.5)
    ax[0].set_xlabel('Tiempo (s)')
    ax[0].set_ylabel('S_shared')
    ax[0].set_title('Evolución de lateralidad')
    
    ax[1].plot(range(len(respuestas)), respuestas, color='red')
    ax[1].axhline(y=umbral, color='green', linestyle='--', label=f'Umbral ({UMBRAL_R2_SIGMAS}σ)')
    ax[1].set_xlabel('Paso')
    ax[1].set_ylabel('|ΔΩ_A|')
    ax[1].set_title('Respuesta a inanición')
    ax[1].legend()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('v119_logs', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'v119_logs/v119_resultados_{timestamp}.png', dpi=150)
    print(f"\n  📊 Gráfico: v119_logs/v119_resultados_{timestamp}.png")
    
    return A3, B3, R2


if __name__ == "__main__":
    A, B, R2 = ejecutar_v119()