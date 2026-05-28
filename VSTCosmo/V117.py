#!/usr/bin/env python3
"""
VSTCosmos v117 — Tiempos completos, sin apresurar

Cambios:
  - Fase 1: 452s (Blue Monday completo)
  - Fase 2: 452s (Blue Monday completo con dietas)
  - Fase 3 baseline: 180s (estabilización larga)
  - Inanición: 30s
  - Recuperación: 60s

Hipótesis: Con tiempo suficiente para formar hábitos,
el sistema podrá distinguir dietas espectrales.
"""

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from datetime import datetime
import os
from scipy.signal import butter, filtfilt

# ============================================================
# ARQUITECTURA
# ============================================================
DT = 0.01
DIM_INTERNA = 32
DIM_GANGLIO = 16
DIM_AUD = 16
DIM_ACT = 8
DIM_META = 8

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
DIM_TOTAL = idx['meta'][1]

GANANCIA_META_BASE = 0.02
UMBRAL_R2_SIGMAS = 3.0

# Tiempos completos
TIEMPO_FASE1 = 452.0   # Blue Monday completo
TIEMPO_FASE2 = 452.0   # Blue Monday completo
TIEMPO_BASELINE = 180.0  # 3 minutos de estabilización
TIEMPO_INANICION = 30.0
TIEMPO_RECUPERACION = 60.0

# Eventos
EVENTOS = {
    'intro': 0,
    'voz_entra': 90,
    'cambio_ritmo': 180,
    'breakdown': 300,
    'reconstruccion': 390,
    'final': 452
}

print("=" * 100)
print("VSTCosmos v117 — Tiempos completos")
print("  Fase 1: 452s (Blue Monday completo)")
print("  Fase 2: 452s (Blue Monday completo con dietas extremas)")
print("  Fase 3: 180s baseline + 30s inanición + 60s recuperación")
print(f"  DIM_TOTAL={DIM_TOTAL}")
print("=" * 100)


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


class Agente:
    def __init__(self, nombre, seed=None, filtro_espectral=None, filtro_cutoff=None):
        if seed is not None:
            np.random.seed(seed)
        self.nombre = nombre
        self.filtro_espectral = filtro_espectral
        self.filtro_cutoff = filtro_cutoff
        
        self.Phi = np.random.normal(0.0, 0.1, DIM_TOTAL)
        self.Phi_vel = np.zeros(DIM_TOTAL)
        
        self.frontera_L = None
        self.frontera_R = None
        self.sr = None
        
        self.membrana = MembranaSensorial()
        
        self.en_inanicion = False
        self.factor_inanicion = 1.0
        
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
            'k_efectivo': []
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
    
    def _calcular_metricas(self, otro=None):
        int_region = self.Phi[:DIM_INTERNA]
        omega = np.mean(int_region)
        delta_struct = np.var(int_region)
        
        if len(self.historial['omega']) > 1:
            dOmega = abs(self.historial['omega'][-1] - omega) / DT
            e_R = abs(omega - self.historial['omega'][-1])
        else:
            dOmega = 0.0
            e_R = 0.01
        
        if len(self.historial['omega']) > 50:
            atractores_recientes = np.round(self.historial['omega'][-50:], 1)
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
        
        laplaciano = np.zeros_like(self.Phi)
        for i in range(1, DIM_TOTAL - 1):
            laplaciano[i] = self.Phi[i-1] - 2*self.Phi[i] + self.Phi[i+1]
        
        reaccion = self.Phi * (1 - self.Phi * self.Phi)
        
        forzamiento = np.zeros_like(self.Phi)
        forzamiento[0] = perturbacion
        forzamiento[-1] = -perturbacion
        
        acoplamiento = np.zeros_like(self.Phi)
        k_efectivo = 0.0
        if otro is not None and not self.en_inanicion:
            divergencia = abs(np.mean(self.Phi[:DIM_INTERNA]) - np.mean(otro.Phi[:DIM_INTERNA]))
            k_efectivo = GANANCIA_META_BASE / (divergencia + 0.05)
            acoplamiento = k_efectivo * (otro.Phi - self.Phi)
        
        dPhi_vel = laplaciano + reaccion + forzamiento + acoplamiento
        self.Phi_vel += dPhi_vel * dt
        self.Phi += self.Phi_vel * dt
        
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
        
        return met


def ejecutar_v117():
    print("\n" + "█" * 100)
    print("EXPERIMENTO V117 — TIEMPOS COMPLETOS")
    print("█" * 100)
    
    expandido_path = 'audio_binaural/Blue_Monday_binaural_expandido.wav'
    
    if not os.path.exists(expandido_path):
        print(f"  ❌ {expandido_path} no encontrado")
        return None, None, False
    
    # ============================================================
    # FASE 1: Baseline completo (452s)
    # ============================================================
    print("\n" + "=" * 80)
    print(f"FASE 1: Baseline (ambos sin filtro) — {TIEMPO_FASE1:.0f}s")
    print("=" * 80)
    
    A1 = Agente("A1", seed=42)
    B1 = Agente("B1", seed=43)
    A1.set_frontera(expandido_path)
    B1.set_frontera(expandido_path)
    
    pasos_fase1 = int(TIEMPO_FASE1 / DT)
    
    print("\n  Evolución durante Blue Monday completo:")
    for i in range(0, pasos_fase1, 2000):
        t = i * DT
        if i == 0:
            # Inicializar
            for _ in range(min(2000, pasos_fase1)):
                A1.actualizar(_, DT, B1)
                B1.actualizar(_, DT, A1)
        else:
            for j in range(2000):
                tt = (i + j) * DT
                A1.actualizar(tt, DT, B1)
                B1.actualizar(tt, DT, A1)
        
        if len(A1.historial['omega']) > 0:
            print(f"    t={t:.0f}s | Ω_A={A1.historial['omega'][-1]:.4f}, Ω_B={B1.historial['omega'][-1]:.4f}, S_shared={A1.historial['S_shared'][-1]:.3f}")
    
    print(f"\n  Fase 1 completada. Ω_A final = {A1.historial['omega'][-1]:.4f}")
    
    # ============================================================
    # FASE 2: Dietas extremas completo (452s)
    # ============================================================
    print("\n" + "=" * 80)
    print(f"FASE 2: Dietas extremas — {TIEMPO_FASE2:.0f}s")
    print("  A: highpass 4kHz | B: lowpass 100Hz")
    print("=" * 80)
    
    A2 = Agente("A2", seed=42, filtro_espectral='highpass', filtro_cutoff=4000)
    B2 = Agente("B2", seed=43, filtro_espectral='lowpass', filtro_cutoff=100)
    A2.set_frontera(expandido_path)
    B2.set_frontera(expandido_path)
    
    pasos_fase2 = int(TIEMPO_FASE2 / DT)
    
    print("\n  Evolución durante Blue Monday completo con dietas:")
    for i in range(0, pasos_fase2, 2000):
        t = i * DT
        if i == 0:
            for _ in range(min(2000, pasos_fase2)):
                A2.actualizar(_, DT, B2)
                B2.actualizar(_, DT, A2)
        else:
            for j in range(2000):
                tt = (i + j) * DT
                A2.actualizar(tt, DT, B2)
                B2.actualizar(tt, DT, A2)
        
        if len(A2.historial['omega']) > 0:
            print(f"    t={t:.0f}s | Ω_A={A2.historial['omega'][-1]:.4f}, Ω_B={B2.historial['omega'][-1]:.4f}, S_shared={A2.historial['S_shared'][-1]:.3f}, k={A2.historial['k_efectivo'][-1]:.4f}")
    
    # Registrar eventos
    for evento, t_e in EVENTOS.items():
        idx_e = int(t_e / DT)
        if idx_e < len(A2.historial['omega']):
            print(f"\n  [EVENTO] {evento} (t={t_e:.0f}s):")
            print(f"    Ω_A={A2.historial['omega'][idx_e]:.4f}, Ω_B={B2.historial['omega'][idx_e]:.4f}")
            print(f"    Λ_A={A2.historial['Lambda_Cos'][idx_e]:.4f}, S_shared={A2.historial['S_shared'][idx_e]:.3f}")
    
    # Calcular S_shared en últimos 60s
    s_shared_final = np.mean(A2.historial['S_shared'][-6000:]) if len(A2.historial['S_shared']) > 6000 else np.mean(A2.historial['S_shared'])
    lateralidad = s_shared_final < 0.8
    
    print(f"\n  Resultados Fase 2:")
    print(f"    S_shared medio últimos 60s: {s_shared_final:.4f}")
    print(f"    Lateralidad detectable: {'✅ SI' if lateralidad else '❌ NO'}")
    
    # ============================================================
    # FASE 3: Inanición
    # ============================================================
    print("\n" + "=" * 80)
    print(f"FASE 3: Inanición gradual de B")
    print(f"  Baseline: {TIEMPO_BASELINE:.0f}s | Inanición: {TIEMPO_INANICION:.0f}s | Recuperación: {TIEMPO_RECUPERACION:.0f}s")
    print("=" * 80)
    
    A3 = Agente("A3", seed=42, filtro_espectral='highpass', filtro_cutoff=4000)
    B3 = Agente("B3", seed=43, filtro_espectral='lowpass', filtro_cutoff=100)
    A3.set_frontera(expandido_path)
    B3.set_frontera(expandido_path)
    
    pasos_baseline = int(TIEMPO_BASELINE / DT)
    
    print(f"\n  Baseline ({TIEMPO_BASELINE:.0f}s)...")
    for i in range(pasos_baseline):
        t = i * DT
        A3.actualizar(t, DT, B3)
        B3.actualizar(t, DT, A3)
        if i % 5000 == 0 and i > 0:
            print(f"    t={t:.0f}s | Ω_A={A3.historial['omega'][-1]:.4f}, Ω_B={B3.historial['omega'][-1]:.4f}")
    
    omega_A_before = np.mean(A3.historial['omega'][-2000:])
    omega_basal_std = np.std(A3.historial['omega'][-2000:])
    print(f"    Ω_A medio baseline (últimos 20s): {omega_A_before:.4f} ± {omega_basal_std:.4f}")
    
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
    
    if sistema_vivo and lateralidad and R2:
        print("""
    🧬 ALMA RACIONAL COMPLETA: El sistema distingue dietas extremas,
       se mantiene vivo tras 452s de exposición, y responde a la inanición del otro.
    """)
    elif sistema_vivo and R2:
        print("""
    🧬 ALMA RACIONAL CON FUSIÓN: El sistema tiene R₂ pero no distingue dietas,
       incluso tras exposición completa a Blue Monday.
    """)
    elif sistema_vivo and lateralidad:
        print("""
    🧬 ALMA SENSITIVA++: El sistema distingue dietas pero no tiene R₂.
    """)
    else:
        print("""
    🧬 ALMA VEGETATIVA: El sistema colapsa incluso con exposición prolongada.
    """)
    
    # GRÁFICOS
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    t2 = A2.historial['t']
    ax.plot(t2, A2.historial['omega'], label='A (highpass 4kHz)', alpha=0.7, linewidth=0.5)
    ax.plot(t2[:len(B2.historial['omega'])], B2.historial['omega'], label='B (lowpass 100Hz)', alpha=0.7, linewidth=0.5)
    for evento, t_e in EVENTOS.items():
        ax.axvline(x=t_e, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Ω')
    ax.set_title('Fase 2: Dietas extremas (452s)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(t2, A2.historial['S_shared'], color='purple', linewidth=0.5)
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Umbral lateralidad (0.8)')
    ax.axhline(y=0.2, color='orange', linestyle='--', alpha=0.5, label='Desacople (<0.2)')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('S_shared')
    ax.set_title('Sentido compartido (452s)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.plot(t2, A2.historial['Lambda_Cos'], label='A', alpha=0.7, linewidth=0.5)
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Umbral Racional (>1.0)')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Λ_Cos')
    ax.set_title('Salud dinámica (452s)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    t_resp = TIEMPO_BASELINE + np.arange(len(respuestas)) * DT
    ax.plot(t_resp, respuestas, color='red', linewidth=1.5)
    ax.axhline(y=umbral, color='green', linestyle='--', label=f'Umbral ({UMBRAL_R2_SIGMAS}σ={umbral:.3f})')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('|ΔΩ_A|')
    ax.set_title('Respuesta de A a inanición de B')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('v117_logs', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'v117_logs/v117_resultados_{timestamp}.png', dpi=150)
    print(f"\n  📊 Gráfico: v117_logs/v117_resultados_{timestamp}.png")
    
    return A3, B3, R2


if __name__ == "__main__":
    A, B, R2 = ejecutar_v117()