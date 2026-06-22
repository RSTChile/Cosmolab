#!/usr/bin/env python3
"""
VSTCosmos v113 — Experimento completo sin parámetros

Principios:
  1. Los 3 audios como frontera (expandido, left, right)
  2. Sin parámetros externos (sin coeficientes, sin clips)
  3. Dietas incompatibles (A: left+highpass, B: right+lowpass)
  4. Observador separado (medimos todo, no interferimos)
  5. Inanición gradual (B se apaga en 30s)

Métricas registradas en cada paso:
  - t, omega, delta_struct, LF, Lambda_Cos, S_shared
  - divergencia, e_R, dOmega_dt, dLambda_dt
  - saturacion (cuántos nodos exceden |1|)
  - phi_max, phi_min (rango real del campo)
"""

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from datetime import datetime
import os
from scipy.signal import butter, filtfilt

# ============================================================
# ARQUITECTURA (única decisión nuestra)
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

# Índices (topología fija de V100-V111)
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

# Vecindades
VECINDADES = [
    ('int', 'G'),
    ('G', 'aud_L'),
    ('G', 'aud_R'),
    ('G', 'act_perm'),
    ('G', 'act_geom'),
    ('G', 'act_mant'),
    ('G', 'meta'),
    ('aud_L', 'aud_R'),
    ('act_perm', 'aud_L'),
    ('act_perm', 'aud_R'),
    ('act_geom', 'aud_L'),
    ('act_geom', 'aud_R'),
]

# Eventos de Blue Monday (solo para observación)
EVENTOS = {
    'intro': 0,
    'voz_entra': 90,
    'cambio_ritmo': 180,
    'breakdown': 300,
    'reconstruccion': 390,
    'final': 452
}

print("=" * 100)
print("VSTCosmos v113 — Experimento completo sin parámetros")
print("  Los 3 audios: expandido, left, right")
print("  Sin coeficientes, sin clips, sin parámetros externos")
print("  Dietas incompatibles: A=highpass/left, B=lowpass/right")
print(f"  DIM_TOTAL={DIM_TOTAL}")
print("=" * 100)


# ============================================================
# MEMBRANA SENSORIAL (sin coeficientes)
# ============================================================
class MembranaSensorial:
    """Transforma diferencia binaural sin factores de escala."""
    def __init__(self):
        self.historial = []
    
    def procesar(self, dS):
        self.historial.append(dS)
        if len(self.historial) > 100:
            self.historial = self.historial[-100:]
        
        # Sin 0.5, 0.3, 0.2. Solo combinación natural.
        inst = dS
        envolvente = np.mean(np.abs(self.historial[-50:])) if len(self.historial) >= 50 else 0
        derivada = self.historial[-1] - self.historial[-2] if len(self.historial) >= 2 else 0
        no_lineal = np.tanh(dS)
        
        # La perturbación es la suma natural, sin coeficientes
        return inst + envolvente + derivada + no_lineal


# ============================================================
# FILTROS PARA DIETAS INCOMPATIBLES
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
# CLASE AGENTE — SIN PARÁMETROS EXTERNOS
# ============================================================
class Agente:
    def __init__(self, nombre, seed=None, filtro_tipo=None, filtro_cutoff=None):
        if seed is not None:
            np.random.seed(seed)
        self.nombre = nombre
        self.filtro_tipo = filtro_tipo
        self.filtro_cutoff = filtro_cutoff
        
        # Estado inicial con pequeña asimetría
        self.Phi = np.random.normal(0.0, 0.1, DIM_TOTAL)
        self.Phi_vel = np.zeros(DIM_TOTAL)
        
        # Frontera (se setea externamente)
        self.frontera_L = None
        self.frontera_R = None
        self.sr = None
        
        # Membrana sensorial
        self.membrana = MembranaSensorial()
        
        # Estado de inanición
        self.en_inanicion = False
        self.factor_inanicion = 1.0
        
        # Historial COMPLETO
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
            'dLambda_dt': [],
            'saturacion': [],
            'phi_max': [],
            'phi_min': [],
            'phi_mean': [],
            'phi_std': []
        }
    
    def set_frontera_binaural(self, left_path, right_path, aplicar_filtros=False):
        """Carga archivos binaurales (L y R separados)."""
        data_left, sr_left = sf.read(left_path, dtype='float32')
        data_right, sr_right = sf.read(right_path, dtype='float32')
        
        if data_left.ndim == 2:
            canal_L = data_left[:, 0]
            canal_R = data_left[:, 1] if data_left.shape[1] > 1 else data_left[:, 0]
        else:
            canal_L = data_left
            canal_R = data_left
        
        if data_right.ndim == 2:
            # Para right, usar el canal derecho como principal
            canal_R_right = data_right[:, 1] if data_right.shape[1] > 1 else data_right[:, 0]
        else:
            canal_R_right = data_right
        
        self.sr = sr_left
        
        # Aplicar filtros si es necesario (dietas incompatibles)
        if aplicar_filtros and self.filtro_tipo:
            if self.filtro_tipo == 'highpass':
                canal_L = aplicar_filtro(canal_L, self.sr, 'highpass', self.filtro_cutoff)
                canal_R = aplicar_filtro(canal_R, self.sr, 'highpass', self.filtro_cutoff)
            elif self.filtro_tipo == 'lowpass':
                canal_L = aplicar_filtro(canal_L, self.sr, 'lowpass', self.filtro_cutoff)
                canal_R = aplicar_filtro(canal_R, self.sr, 'lowpass', self.filtro_cutoff)
        
        self.frontera_L = canal_L
        self.frontera_R = canal_R_right
        
        print(f"  [{self.nombre}] Frontera binaural cargada ({len(self.frontera_L)/self.sr:.1f}s)")
    
    def set_frontera_expandido(self, audio_path, aplicar_filtros=False):
        """Carga archivo expandido (un solo archivo con L y R)."""
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
        
        print(f"  [{self.nombre}] Frontera expandido cargada ({len(self.frontera_L)/self.sr:.1f}s)")
    
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
        
        # Perturbación de la membrana (sin coeficientes)
        perturbacion = self.membrana.procesar(dS)
        
        # Laplaciano (difusión natural)
        laplaciano = np.zeros_like(self.Phi)
        for i in range(1, DIM_TOTAL - 1):
            laplaciano[i] = self.Phi[i-1] - 2*self.Phi[i] + self.Phi[i+1]
        
        # Reacción natural (sin ganancia)
        reaccion = self.Phi * (1 - self.Phi * self.Phi)
        
        # Forzamiento de frontera natural (sin coeficiente)
        forzamiento = np.zeros_like(self.Phi)
        forzamiento[0] = perturbacion
        forzamiento[-1] = -perturbacion
        
        # Acoplamiento natural (sin coeficiente)
        acoplamiento = np.zeros_like(self.Phi)
        if otro is not None and not self.en_inanicion:
            acoplamiento = otro.Phi - self.Phi
        
        # Evolución (sin parámetros)
        dPhi_vel = laplaciano + reaccion + forzamiento + acoplamiento
        self.Phi_vel += dPhi_vel * dt
        self.Phi += self.Phi_vel * dt
        
        # SIN CLIPS - observamos saturación
        saturados = np.sum(np.abs(self.Phi) > 1.0)
        
        met = self._calcular_metricas(otro)
        
        # Derivadas
        dLambda = 0.0
        if len(self.historial['Lambda_Cos']) > 1:
            dLambda = (met['Lambda_Cos'] - self.historial['Lambda_Cos'][-1]) / dt
        
        # Guardar historial
        self.historial['t'].append(t)
        self.historial['omega'].append(met['omega'])
        self.historial['delta_struct'].append(met['delta_struct'])
        self.historial['LF'].append(met['LF'])
        self.historial['Lambda_Cos'].append(met['Lambda_Cos'])
        self.historial['S_shared'].append(met['S_shared'])
        self.historial['divergencia'].append(met['divergencia'])
        self.historial['e_R'].append(met['e_R'])
        self.historial['dOmega_dt'].append(met['dOmega'])
        self.historial['dLambda_dt'].append(dLambda)
        self.historial['saturacion'].append(saturados)
        self.historial['phi_max'].append(np.max(self.Phi))
        self.historial['phi_min'].append(np.min(self.Phi))
        self.historial['phi_mean'].append(np.mean(self.Phi))
        self.historial['phi_std'].append(np.std(self.Phi))
        
        return met


# ============================================================
# EXPERIMENTO V113 — COMPLETO
# ============================================================
def ejecutar_v113():
    print("\n" + "█" * 100)
    print("EXPERIMENTO V113 — COMPLETO SIN PARÁMETROS")
    print("█" * 100)
    
    # Archivos
    expandido_path = 'audio_binaural/Blue_Monday_binaural_expandido.wav'
    left_path = 'audio_binaural/Blue_Monday_binaural_expandido_left_binaural.wav'
    right_path = 'audio_binaural/Blue_Monday_binaural_expandido_right_binaural.wav'
    
    for p in [expandido_path, left_path, right_path]:
        if not os.path.exists(p):
            print(f"  ❌ {p} no encontrado")
            return None, None, False
    
    # ============================================================
    # FASE 1: Baseline con EXPANDIDO (misma dieta)
    # ============================================================
    print("\n" + "=" * 80)
    print("FASE 1: Baseline con EXPANDIDO (misma dieta)")
    print("  Propósito: Ver comportamiento base del sistema")
    print("=" * 80)
    
    A1 = Agente("A1", seed=42)
    B1 = Agente("B1", seed=43)
    
    A1.set_frontera_expandido(expandido_path, aplicar_filtros=False)
    B1.set_frontera_expandido(expandido_path, aplicar_filtros=False)
    
    T_fase1 = 90.0  # Hasta que entra la voz
    pasos_fase1 = int(T_fase1 / DT)
    
    for i in range(pasos_fase1):
        t = i * DT
        met_A = A1.actualizar(t, DT, B1)
        met_B = B1.actualizar(t, DT, A1)
        
        if i % 2000 == 0:
            print(f"  t={t:.1f}s | Ω_A={met_A['omega']:.4f}, Ω_B={met_B['omega']:.4f}, S_shared={met_A['S_shared']:.3f}")
    
    print(f"\n  Resultados Fase 1 (Expandido):")
    print(f"    Ω_A final: {A1.historial['omega'][-1]:.4f}")
    print(f"    Ω_B final: {B1.historial['omega'][-1]:.4f}")
    print(f"    S_shared final: {A1.historial['S_shared'][-1]:.3f}")
    print(f"    Saturación A: {sum(A1.historial['saturacion'])/len(A1.historial['saturacion']):.1f} nodos/paso")
    
    # ============================================================
    # FASE 2: Dietas incompatibles (A: left+highpass, B: right+lowpass)
    # ============================================================
    print("\n" + "=" * 80)
    print("FASE 2: Dietas incompatibles")
    print("  A: LEFT + highpass 2kHz")
    print("  B: RIGHT + lowpass 200Hz")
    print("  Propósito: ¿El sistema distingue lateralidad?")
    print("=" * 80)
    
    A2 = Agente("A2", seed=42, filtro_tipo='highpass', filtro_cutoff=2000)
    B2 = Agente("B2", seed=43, filtro_tipo='lowpass', filtro_cutoff=200)
    
    A2.set_frontera_binaural(left_path, left_path, aplicar_filtros=True)
    B2.set_frontera_binaural(right_path, right_path, aplicar_filtros=True)
    
    T_fase2 = 180.0  # Desde voz hasta breakdown
    pasos_fase2 = int(T_fase2 / DT)
    
    for i in range(pasos_fase2):
        t = i * DT
        met_A = A2.actualizar(t, DT, B2)
        met_B = B2.actualizar(t, DT, A2)
        
        if i % 2000 == 0:
            print(f"  t={t:.1f}s | Ω_A={met_A['omega']:.4f}, Ω_B={met_B['omega']:.4f}, S_shared={met_A['S_shared']:.3f}")
    
    # Detectar eventos
    for evento, t_e in EVENTOS.items():
        if t_e < T_fase2:
            idx_e = int(t_e / DT)
            if idx_e < len(A2.historial['omega']):
                print(f"\n  [EVENTO] {evento} (t={t_e:.0f}s):")
                print(f"    Ω_A={A2.historial['omega'][idx_e]:.4f}, Ω_B={B2.historial['omega'][idx_e]:.4f}")
                print(f"    Λ_A={A2.historial['Lambda_Cos'][idx_e]:.4f}, S_shared={A2.historial['S_shared'][idx_e]:.3f}")
    
    print(f"\n  Resultados Fase 2 (Left vs Right):")
    print(f"    Ω_A medio últimos 30s: {np.mean(A2.historial['omega'][-3000:]):.4f}")
    print(f"    Ω_B medio últimos 30s: {np.mean(B2.historial['omega'][-3000:]):.4f}")
    print(f"    S_shared medio últimos 30s: {np.mean(A2.historial['S_shared'][-3000:]):.4f}")
    
    lateralidad = np.mean(A2.historial['S_shared'][-3000:]) < 0.8
    print(f"    Lateralidad detectable: {'✅ SI' if lateralidad else '❌ NO'}")
    
    # ============================================================
    # FASE 3: Inanición gradual de B (con dietas incompatibles)
    # ============================================================
    print("\n" + "=" * 80)
    print("FASE 3: Inanición gradual de B (30s fadeout)")
    print("  Propósito: ¿A responde a la pérdida de B?")
    print("=" * 80)
    
    A3 = Agente("A3", seed=42, filtro_tipo='highpass', filtro_cutoff=2000)
    B3 = Agente("B3", seed=43, filtro_tipo='lowpass', filtro_cutoff=200)
    
    A3.set_frontera_binaural(left_path, left_path, aplicar_filtros=True)
    B3.set_frontera_binaural(right_path, right_path, aplicar_filtros=True)
    
    # Baseline (60s)
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
    
    # Inanición gradual (30s)
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
    
    # Recuperación (30s)
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
    umbral = 2 * omega_basal_std
    
    print(f"\n  Resultados Fase 3 (Inanición):")
    print(f"    Ω_A después: {omega_A_after:.4f}")
    print(f"    ΔA = {omega_A_after - omega_A_before:+.4f}")
    print(f"    Respuesta máxima: {respuesta_max:.4f}")
    print(f"    Umbral (2σ): {umbral:.4f}")
    
    R2_detectado = respuesta_max > umbral
    print(f"    R₂: {'✅ CONFIRMADO' if R2_detectado else '❌ NO'}")
    
    # ============================================================
    # CONCLUSION
    # ============================================================
    print("\n" + "=" * 80)
    print("CONCLUSIÓN")
    print("=" * 80)
    
    # Evaluar estado del sistema
    omega_rango = max(A2.historial['omega']) - min(A2.historial['omega'])
    sistema_vivo = 0.1 < np.mean(A2.historial['omega'][-3000:]) < 0.9 and omega_rango > 0.2
    
    print(f"\n  Estado del sistema:")
    print(f"    Rango Ω en Fase 2: {omega_rango:.3f}")
    print(f"    {'✅ VIVO' if sistema_vivo else '❌ COLMAPSO'}")
    print(f"    Lateralidad: {'✅' if lateralidad else '❌'}")
    print(f"    R₂: {'✅' if R2_detectado else '❌'}")
    
    if sistema_vivo and lateralidad and R2_detectado:
        print("""
    🧬 ALMA RACIONAL: El sistema distingue lateralidad,
       se mantiene vivo sin parámetros, y responde a la inanición del otro.
    """)
    elif sistema_vivo and lateralidad:
        print("""
    🧬 ALMA SENSITIVA++: El sistema distingue lateralidad y se mantiene vivo,
       pero no responde selectivamente a la inanición.
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
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Fase 1: Expandido
    ax = axes[0, 0]
    t1 = A1.historial['t']
    ax.plot(t1, A1.historial['omega'], label='A', alpha=0.7)
    ax.plot(t1[:len(B1.historial['omega'])], B1.historial['omega'], label='B', alpha=0.7)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Ω')
    ax.set_title('Fase 1: Expandido (baseline)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Fase 2: Left vs Right
    ax = axes[0, 1]
    t2 = A2.historial['t']
    ax.plot(t2, A2.historial['omega'], label='A (Left+HP)', alpha=0.7)
    ax.plot(t2[:len(B2.historial['omega'])], B2.historial['omega'], label='B (Right+LP)', alpha=0.7)
    for evento, t_e in EVENTOS.items():
        if t_e < T_fase2:
            ax.axvline(x=t_e, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Ω')
    ax.set_title('Fase 2: Left vs Right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Fase 2: S_shared
    ax = axes[0, 2]
    ax.plot(t2, A2.historial['S_shared'], color='purple')
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Fusión')
    ax.axhline(y=0.2, color='orange', linestyle='--', alpha=0.5, label='Desacople')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('S_shared')
    ax.set_title('Sentido compartido (Fase 2)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Fase 3: Respuesta a inanición
    ax = axes[1, 0]
    t3 = A3.historial['t']
    ax.plot(t3, A3.historial['omega'], label='A', alpha=0.7)
    ax.axvline(x=T_baseline, color='red', linestyle='--', label='Inicio inanición B')
    ax.axvline(x=T_baseline + T_inanicion, color='orange', linestyle='--', label='Fin inanición')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Ω')
    ax.set_title('Fase 3: Respuesta a inanición')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Fase 3: Respuesta cuantitativa
    ax = axes[1, 1]
    t_resp = T_baseline + np.arange(len(respuestas)) * DT
    ax.plot(t_resp, respuestas, color='red')
    ax.axhline(y=umbral, color='green', linestyle='--', label=f'Umbral (2σ={umbral:.3f})')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('|ΔΩ_A|')
    ax.set_title('Respuesta de A a inanición de B')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Saturación
    ax = axes[1, 2]
    ax.plot(t2, A2.historial['saturacion'], label='A', alpha=0.7)
    ax.plot(t2[:len(B2.historial['saturacion'])], B2.historial['saturacion'], label='B', alpha=0.7)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Nodos con |Φ|>1')
    ax.set_title('Saturación del campo')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('v113_logs', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'v113_logs/v113_resultados_{timestamp}.png', dpi=150)
    print(f"\n  📊 Gráfico: v113_logs/v113_resultados_{timestamp}.png")
    
    return A3, B3, R2_detectado


if __name__ == "__main__":
    A, B, R2 = ejecutar_v113()