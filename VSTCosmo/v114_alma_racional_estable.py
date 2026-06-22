#!/usr/bin/env python3
"""
VSTCosmos v114 — Alma Racional estable
Acoplamiento adaptativo real para evitar fusión.
Todo lo demás: sin parámetros, sin clips, sin coeficientes.
"""

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from datetime import datetime
import os
from scipy.signal import butter, filtfilt

# ============================================================
# ARQUITECTURA (igual que V113)
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

# Índices
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
print("VSTCosmos v114 — Alma Racional estable")
print("  Sin parámetros externos (como V113)")
print("  Acoplamiento adaptativo: k = 0.05 / (divergencia + 0.05)")
print("  Dietas incompatibles: A=left+highpass, B=right+lowpass")
print(f"  DIM_TOTAL={DIM_TOTAL}")
print("=" * 100)


# ============================================================
# MEMBRANA SENSORIAL (sin coeficientes)
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
# CLASE AGENTE — CON ACOPLAMIENTO ADAPTATIVO
# ============================================================
class Agente:
    def __init__(self, nombre, seed=None, filtro_tipo=None, filtro_cutoff=None):
        if seed is not None:
            np.random.seed(seed)
        self.nombre = nombre
        self.filtro_tipo = filtro_tipo
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
            'saturacion': []
        }
    
    def set_frontera_binaural(self, left_path, right_path, aplicar_filtros=False):
        data_left, sr_left = sf.read(left_path, dtype='float32')
        data_right, sr_right = sf.read(right_path, dtype='float32')
        
        if data_left.ndim == 2:
            canal_L = data_left[:, 0]
            canal_R = data_left[:, 1] if data_left.shape[1] > 1 else data_left[:, 0]
        else:
            canal_L = data_left
            canal_R = data_left
        
        if data_right.ndim == 2:
            canal_R_right = data_right[:, 1] if data_right.shape[1] > 1 else data_right[:, 0]
        else:
            canal_R_right = data_right
        
        self.sr = sr_left
        
        if aplicar_filtros and self.filtro_tipo:
            if self.filtro_tipo == 'highpass':
                canal_L = aplicar_filtro(canal_L, self.sr, 'highpass', self.filtro_cutoff)
                canal_R = aplicar_filtro(canal_R, self.sr, 'highpass', self.filtro_cutoff)
            elif self.filtro_tipo == 'lowpass':
                canal_L = aplicar_filtro(canal_L, self.sr, 'lowpass', self.filtro_cutoff)
                canal_R = aplicar_filtro(canal_R, self.sr, 'lowpass', self.filtro_cutoff)
        
        self.frontera_L = canal_L
        self.frontera_R = canal_R_right
        print(f"  [{self.nombre}] Frontera binaural ({len(self.frontera_L)/self.sr:.1f}s)")
    
    def set_frontera_expandido(self, audio_path, aplicar_filtros=False):
        data, sr = sf.read(audio_path, dtype='float32')
        self.sr = sr
        
        if data.ndim == 2:
            canal_L = data[:, 0]
            canal_R = data[:, 1]
        else:
            canal_L = data
            canal_R = data
        
        if aplicar_filtros and self.filtro_tipo:
            if self.filtro_tipo == 'highpass':
                canal_L = aplicar_filtro(canal_L, self.sr, 'highpass', self.filtro_cutoff)
                canal_R = aplicar_filtro(canal_R, self.sr, 'highpass', self.filtro_cutoff)
            elif self.filtro_tipo == 'lowpass':
                canal_L = aplicar_filtro(canal_L, self.sr, 'lowpass', self.filtro_cutoff)
                canal_R = aplicar_filtro(canal_R, self.sr, 'lowpass', self.filtro_cutoff)
        
        self.frontera_L = canal_L
        self.frontera_R = canal_R
        print(f"  [{self.nombre}] Frontera expandido ({len(self.frontera_L)/self.sr:.1f}s)")
    
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
        
        # Laplaciano
        laplaciano = np.zeros_like(self.Phi)
        for i in range(1, DIM_TOTAL - 1):
            laplaciano[i] = self.Phi[i-1] - 2*self.Phi[i] + self.Phi[i+1]
        
        # Reacción natural
        reaccion = self.Phi * (1 - self.Phi * self.Phi)
        
        # Forzamiento natural
        forzamiento = np.zeros_like(self.Phi)
        forzamiento[0] = perturbacion
        forzamiento[-1] = -perturbacion
        
        # ============================================================
        # ACOPLAMIENTO ADAPTATIVO REAL (la única diferencia con V113)
        # ============================================================
        acoplamiento = np.zeros_like(self.Phi)
        if otro is not None and not self.en_inanicion:
            # Divergencia actual entre agentes
            divergencia = abs(np.mean(self.Phi[:DIM_INTERNA]) - np.mean(otro.Phi[:DIM_INTERNA]))
            # k adaptativo: si divergen mucho, acopla poco; si están cerca, acopla más
            k = 0.05 / (divergencia + 0.05)  # k ∈ [0, 1]
            acoplamiento = k * (otro.Phi - self.Phi)
        
        # Evolución
        dPhi_vel = laplaciano + reaccion + forzamiento + acoplamiento
        self.Phi_vel += dPhi_vel * dt
        self.Phi += self.Phi_vel * dt
        
        # Medir saturación (solo observación)
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
        
        return met


# ============================================================
# EXPERIMENTO V114
# ============================================================
def ejecutar_v114():
    print("\n" + "█" * 100)
    print("EXPERIMENTO V114 — ALMA RACIONAL ESTABLE")
    print("█" * 100)
    
    expandido_path = 'audio_binaural/Blue_Monday_binaural_expandido.wav'
    left_path = 'audio_binaural/Blue_Monday_binaural_expandido_left_binaural.wav'
    right_path = 'audio_binaural/Blue_Monday_binaural_expandido_right_binaural.wav'
    
    for p in [expandido_path, left_path, right_path]:
        if not os.path.exists(p):
            print(f"  ❌ {p} no encontrado")
            return None, None, False
    
    # ============================================================
    # FASE 1: Baseline con EXPANDIDO
    # ============================================================
    print("\n" + "=" * 80)
    print("FASE 1: Baseline con EXPANDIDO")
    print("=" * 80)
    
    A1 = Agente("A1", seed=42)
    B1 = Agente("B1", seed=43)
    A1.set_frontera_expandido(expandido_path)
    B1.set_frontera_expandido(expandido_path)
    
    T_fase1 = 90.0
    pasos_fase1 = int(T_fase1 / DT)
    
    for i in range(pasos_fase1):
        t = i * DT
        met_A = A1.actualizar(t, DT, B1)
        met_B = B1.actualizar(t, DT, A1)
        if i % 2000 == 0:
            print(f"  t={t:.1f}s | Ω_A={met_A['omega']:.4f}, Ω_B={met_B['omega']:.4f}, S_shared={met_A['S_shared']:.3f}")
    
    print(f"\n  Fase 1: Ω_A medio = {np.mean(A1.historial['omega'][-1000:]):.4f}")
    
    # ============================================================
    # FASE 2: Left vs Right con acoplamiento adaptativo
    # ============================================================
    print("\n" + "=" * 80)
    print("FASE 2: Left vs Right (acoplamiento adaptativo)")
    print("  A: LEFT + highpass 2kHz")
    print("  B: RIGHT + lowpass 200Hz")
    print("=" * 80)
    
    A2 = Agente("A2", seed=42, filtro_tipo='highpass', filtro_cutoff=2000)
    B2 = Agente("B2", seed=43, filtro_tipo='lowpass', filtro_cutoff=200)
    A2.set_frontera_binaural(left_path, left_path, aplicar_filtros=True)
    B2.set_frontera_binaural(right_path, right_path, aplicar_filtros=True)
    
    T_fase2 = 180.0
    pasos_fase2 = int(T_fase2 / DT)
    
    for i in range(pasos_fase2):
        t = i * DT
        met_A = A2.actualizar(t, DT, B2)
        met_B = B2.actualizar(t, DT, A2)
        if i % 2000 == 0:
            print(f"  t={t:.1f}s | Ω_A={met_A['omega']:.4f}, Ω_B={met_B['omega']:.4f}, S_shared={met_A['S_shared']:.3f}, k_efectivo={0.05/(met_A['divergencia']+0.05):.3f}")
    
    # Calcular S_shared en últimos 30s
    s_shared_final = np.mean(A2.historial['S_shared'][-3000:])
    lateralidad = s_shared_final < 0.8
    
    print(f"\n  Resultados Fase 2:")
    print(f"    S_shared medio últimos 30s: {s_shared_final:.4f}")
    print(f"    Lateralidad detectable: {'✅ SI' if lateralidad else '❌ NO'}")
    
    # ============================================================
    # FASE 3: Inanición gradual
    # ============================================================
    print("\n" + "=" * 80)
    print("FASE 3: Inanición gradual de B (30s)")
    print("=" * 80)
    
    A3 = Agente("A3", seed=42, filtro_tipo='highpass', filtro_cutoff=2000)
    B3 = Agente("B3", seed=43, filtro_tipo='lowpass', filtro_cutoff=200)
    A3.set_frontera_binaural(left_path, left_path, aplicar_filtros=True)
    B3.set_frontera_binaural(right_path, right_path, aplicar_filtros=True)
    
    T_baseline = 60.0
    pasos_baseline = int(T_baseline / DT)
    
    print(f"\n  Baseline ({T_baseline}s)...")
    for i in range(pasos_baseline):
        t = i * DT
        met_A = A3.actualizar(t, DT, B3)
        met_B = B3.actualizar(t, DT, A3)
    
    omega_A_before = np.mean(A3.historial['omega'][-1000:])
    omega_basal_std = np.std(A3.historial['omega'][-1000:])
    print(f"    Ω_A medio baseline: {omega_A_before:.4f} ± {omega_basal_std:.4f}")
    
    T_inanicion = 30.0
    pasos_inanicion = int(T_inanicion / DT)
    
    print(f"\n  Inanición gradual ({T_inanicion}s)...")
    respuestas = []
    
    for i in range(pasos_inanicion):
        t = T_baseline + i * DT
        B3.inducir_inanicion_gradual(i, pasos_inanicion)
        met_A = A3.actualizar(t, DT, B3)
        met_B = B3.actualizar(t, DT, A3)
        respuesta = abs(met_A['omega'] - omega_A_before)
        respuestas.append(respuesta)
        if i % 500 == 0:
            print(f"    t={t:.1f}s | Ω_A={met_A['omega']:.4f} (resp={respuesta:.4f}) | factor={B3.factor_inanicion:.2f}")
    
    T_recuperacion = 30.0
    pasos_recuperacion = int(T_recuperacion / DT)
    
    print(f"\n  Recuperación ({T_recuperacion}s)...")
    B3.en_inanicion = False
    B3.factor_inanicion = 1.0
    
    for i in range(pasos_recuperacion):
        t = T_baseline + T_inanicion + i * DT
        met_A = A3.actualizar(t, DT, B3)
        met_B = B3.actualizar(t, DT, A3)
    
    omega_A_after = np.mean(A3.historial['omega'][-1000:])
    respuesta_max = max(respuestas) if respuestas else 0
    umbral = 3 * omega_basal_std  # Más estricto: 3σ
    
    print(f"\n  Resultados Fase 3:")
    print(f"    Ω_A después: {omega_A_after:.4f}")
    print(f"    ΔA = {omega_A_after - omega_A_before:+.4f}")
    print(f"    Respuesta máxima: {respuesta_max:.4f}")
    print(f"    Umbral (3σ): {umbral:.4f}")
    
    R2 = respuesta_max > umbral
    print(f"    R₂: {'✅ CONFIRMADO' if R2 else '❌ NO'}")
    
    # ============================================================
    # CONCLUSION
    # ============================================================
    print("\n" + "=" * 80)
    print("CONCLUSIÓN")
    print("=" * 80)
    
    # Calcular rango real de Ω
    omega_min = min(A2.historial['omega'])
    omega_max = max(A2.historial['omega'])
    rango = omega_max - omega_min
    sistema_vivo = rango > 0.5  # Si explora más de 0.5 de rango
    
    print(f"\n  Estado del sistema:")
    print(f"    Rango Ω en Fase 2: {rango:.3f}")
    print(f"    {'✅ VIVO' if sistema_vivo else '❌ COLMAPSO'}")
    print(f"    Lateralidad: {'✅' if lateralidad else '❌'}")
    print(f"    R₂: {'✅' if R2 else '❌'}")
    
    if sistema_vivo and lateralidad and R2:
        print("""
    🧬 ALMA RACIONAL ESTABLE: El sistema distingue lateralidad,
       se mantiene vivo sin parámetros, y responde a la inanición del otro.
    """)
    elif sistema_vivo and lateralidad:
        print("""
    🧬 ALMA SENSITIVA++: El sistema distingue lateralidad y se mantiene vivo,
       pero no responde selectivamente a la inanición.
    """)
    elif sistema_vivo and R2:
        print("""
    🧬 ALMA RACIONAL CON FUSIÓN: El sistema tiene R₂ pero no distingue lateralidad.
       Ajustar acoplamiento adaptativo.
    """)
    elif sistema_vivo:
        print("""
    🧬 ALMA SENSITIVA: El sistema se mantiene vivo, pero no distingue lateralidad.
    """)
    else:
        print("""
    🧬 ALMA VEGETATIVA: El sistema colapsa. La arquitectura actual no es viable.
    """)
    
    # GRÁFICOS
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Fase 2
    ax = axes[0, 0]
    t2 = A2.historial['t']
    ax.plot(t2, A2.historial['omega'], label='A (Left+HP)', alpha=0.7)
    ax.plot(t2[:len(B2.historial['omega'])], B2.historial['omega'], label='B (Right+LP)', alpha=0.7)
    for evento, t_e in EVENTOS.items():
        if t_e < T_fase2:
            ax.axvline(x=t_e, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Ω')
    ax.set_title('Fase 2: Left vs Right (acoplamiento adaptativo)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # S_shared
    ax = axes[0, 1]
    ax.plot(t2, A2.historial['S_shared'], color='purple')
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Fusión (>0.8)')
    ax.axhline(y=0.2, color='orange', linestyle='--', alpha=0.5, label='Desacople (<0.2)')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('S_shared')
    ax.set_title('Sentido compartido (Fase 2)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Λ_Cos
    ax = axes[1, 0]
    ax.plot(t2, A2.historial['Lambda_Cos'], label='A', alpha=0.7)
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Umbral Racional (>1.0)')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Λ_Cos')
    ax.set_title('Salud dinámica (Fase 2)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Respuesta a inanición
    ax = axes[1, 1]
    t_resp = T_baseline + np.arange(len(respuestas)) * DT
    ax.plot(t_resp, respuestas, color='red')
    ax.axhline(y=umbral, color='green', linestyle='--', label=f'Umbral (3σ={umbral:.3f})')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('|ΔΩ_A|')
    ax.set_title('Respuesta de A a inanición de B')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('v114_logs', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'v114_logs/v114_resultados_{timestamp}.png', dpi=150)
    print(f"\n  📊 Gráfico: v114_logs/v114_resultados_{timestamp}.png")
    
    return A3, B3, R2


if __name__ == "__main__":
    A, B, R2 = ejecutar_v114()