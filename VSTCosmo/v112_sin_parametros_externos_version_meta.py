#!/usr/bin/env python3
"""
VSTCosmos v112 — Sin parámetros externos (versión Meta)
Con registro ampliado para auditar si los clips son necesarios.

Registramos:
- Φ y Φ_vel antes y después del clip
- Valores de saturación (cuánto se recortó)
- Todas las métricas canónicas

Esto nos permitirá ver si el sistema "quiere" salirse de los límites.
"""

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from datetime import datetime
import os
from scipy.ndimage import gaussian_filter1d
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

# Vecindades (topología fija)
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

# Límites para clips (los mantenemos pero registramos saturación)
PHI_CLIP_MIN = -1.0
PHI_CLIP_MAX = 1.0
PHI_VEL_CLIP_MIN = -5.0
PHI_VEL_CLIP_MAX = 5.0

print("=" * 100)
print("VSTCosmos v112 — Sin parámetros externos (Registro ampliado)")
print("  El sistema encuentra su propia escala.")
print("  Registramos saturación para auditar necesidad de clips.")
print(f"  DIM_TOTAL={DIM_TOTAL}")
print("=" * 100)


# ============================================================
# MEMBRANA SENSORIAL (con coeficientes fijos, registramos)
# ============================================================
class MembranaSensorial:
    def __init__(self, buffer_size=100):
        self.buffer_size = buffer_size
        self.buffer = np.zeros(buffer_size)
        self.pos = 0
        self.historial = []
    
    def procesar(self, dS):
        inst = dS
        
        self.buffer[self.pos] = abs(dS)
        self.pos = (self.pos + 1) % self.buffer_size
        envolvente = np.mean(self.buffer)
        
        if len(self.historial) > 10:
            derivada = np.mean(np.diff(self.historial[-10:]))
        else:
            derivada = 0.0
        
        no_lineal = np.tanh(dS * 2.0)
        
        self.historial.append(dS)
        if len(self.historial) > self.buffer_size * 2:
            self.historial = self.historial[-self.buffer_size:]
        
        return inst + 0.5 * envolvente * np.sign(dS) + 0.3 * derivada + 0.2 * no_lineal
    
    def reset(self):
        self.buffer = np.zeros(self.buffer_size)
        self.pos = 0
        self.historial = []


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
# CLASE AGENTE V112 — CON REGISTRO DE SATURACIÓN
# ============================================================
class AgenteVST:
    def __init__(self, nombre, seed=None, filtro_tipo=None, filtro_cutoff=None):
        if seed is not None:
            np.random.seed(seed)
        self.nombre = nombre
        
        # Estado inicial
        self.Phi = np.random.normal(0.5, 0.05, DIM_TOTAL)
        self.Phi_vel = np.zeros(DIM_TOTAL)
        
        # Memorias
        self.W_prof = np.zeros((DIM_INTERNA, DIM_INTERNA))
        self.W_rec = np.zeros((DIM_INTERNA, DIM_INTERNA))
        
        # Filtros
        self.filtro_tipo = filtro_tipo
        self.filtro_cutoff = filtro_cutoff
        self.sr = None
        self.frontera_L = None
        self.frontera_R = None
        
        self.membrana = MembranaSensorial()
        
        self.en_inanicion = False
        self.factor_inanicion = 1.0
        
        # Historial AMPLIADO
        self.historial = {
            't': [],
            'omega': [],
            'delta_struct': [],
            'LF': [],
            'Lambda_Cos': [],
            'S_shared': [],
            'divergencia': [],
            'dOmega_dt': [],
            'dLambda_dt': [],
            # Registro de saturación
            'phi_saturado': [],      # Cantidad de Phi que excedió límites
            'phi_vel_saturado': [],  # Cantidad de Phi_vel que excedió límites
            'max_phi': [],           # Valor máximo de Phi en este paso
            'min_phi': [],           # Valor mínimo de Phi en este paso
            'max_phi_vel': [],       # Valor máximo de Phi_vel
            'min_phi_vel': []        # Valor mínimo de Phi_vel
        }
    
    def set_frontera(self, audio_path):
        data, self.sr = sf.read(audio_path, dtype='float32')
        if data.ndim == 2:
            canal_L = data[:, 0]
            canal_R = data[:, 1]
        else:
            canal_L = data
            canal_R = data
        
        if self.filtro_tipo == 'highpass':
            canal_L = aplicar_filtro(canal_L, self.sr, 'highpass', self.filtro_cutoff)
            canal_R = aplicar_filtro(canal_R, self.sr, 'highpass', self.filtro_cutoff)
        elif self.filtro_tipo == 'lowpass':
            canal_L = aplicar_filtro(canal_L, self.sr, 'lowpass', self.filtro_cutoff)
            canal_R = aplicar_filtro(canal_R, self.sr, 'lowpass', self.filtro_cutoff)
        
        self.frontera_L = canal_L
        self.frontera_R = canal_R
        print(f"  [{self.nombre}] Frontera lista ({len(data)/self.sr:.1f}s)")
    
    def inducir_inanicion_gradual(self, paso_actual, pasos_totales):
        if paso_actual < pasos_totales:
            self.factor_inanicion = 1.0 - (paso_actual / pasos_totales)
        else:
            self.factor_inanicion = 0.0
            self.en_inanicion = True
    
    def _get_frontera_t(self, t):
        if self.en_inanicion:
            return 0.0, 0.0
        idx = int(t * self.sr)
        if idx >= len(self.frontera_L):
            return 0.0, 0.0
        return (self.frontera_L[idx] * self.factor_inanicion,
                self.frontera_R[idx] * self.factor_inanicion)
    
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
        
        RC = dOmega + np.var(self.Phi_vel)
        
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
            'RC': RC,
            'LF': LF,
            'Lambda_Cos': Lambda_Cos,
            'S_shared': S_shared,
            'divergencia': divergencia,
            'e_R': e_R,
            'dOmega': dOmega
        }
    
    def _registrar_saturacion(self):
        """Registra cuánto se saturó Phi y Phi_vel antes del clip"""
        # Phi
        phi_saturado_arriba = np.sum(self.Phi > PHI_CLIP_MAX)
        phi_saturado_abajo = np.sum(self.Phi < PHI_CLIP_MIN)
        phi_saturado = phi_saturado_arriba + phi_saturado_abajo
        
        # Phi_vel
        vel_saturado_arriba = np.sum(self.Phi_vel > PHI_VEL_CLIP_MAX)
        vel_saturado_abajo = np.sum(self.Phi_vel < PHI_VEL_CLIP_MIN)
        vel_saturado = vel_saturado_arriba + vel_saturado_abajo
        
        self.historial['phi_saturado'].append(phi_saturado)
        self.historial['phi_vel_saturado'].append(vel_saturado)
        self.historial['max_phi'].append(np.max(self.Phi))
        self.historial['min_phi'].append(np.min(self.Phi))
        self.historial['max_phi_vel'].append(np.max(self.Phi_vel))
        self.historial['min_phi_vel'].append(np.min(self.Phi_vel))
    
    def actualizar(self, t, dt, otro=None):
        L, R = self._get_frontera_t(t)
        dS = L - R
        perturbacion = self.membrana.procesar(dS)
        
        # Laplaciano (difusión natural)
        laplaciano = np.zeros_like(self.Phi)
        for i in range(1, DIM_TOTAL - 1):
            laplaciano[i] = self.Phi[i-1] - 2*self.Phi[i] + self.Phi[i+1]
        
        # Reacción natural (sin factor de escala)
        reaccion = self.Phi * (1 - self.Phi**2)
        
        # Forzamiento de frontera natural
        forzamiento = np.zeros_like(self.Phi)
        forzamiento[0] = perturbacion
        forzamiento[-1] = -perturbacion
        
        # Acoplamiento natural (con 0.1, registramos si es necesario)
        acoplamiento = np.zeros_like(self.Phi)
        if otro is not None and not self.en_inanicion:
            acoplamiento = (otro.Phi - self.Phi) * 0.1
        
        # Evolución
        dPhi_vel = laplaciano + reaccion + forzamiento + acoplamiento
        self.Phi_vel += dPhi_vel * dt
        self.Phi += self.Phi_vel * dt
        
        # Registrar antes del clip
        self._registrar_saturacion()
        
        # Clips (los mantenemos pero sabemos cuánto recortan)
        self.Phi = np.clip(self.Phi, PHI_CLIP_MIN, PHI_CLIP_MAX)
        self.Phi_vel = np.clip(self.Phi_vel, PHI_VEL_CLIP_MIN, PHI_VEL_CLIP_MAX)
        
        met = self._calcular_metricas(otro)
        
        # Derivadas
        dLambda = 0.0
        if len(self.historial['Lambda_Cos']) > 1:
            dLambda = (met['Lambda_Cos'] - self.historial['Lambda_Cos'][-1]) / dt
        
        self.historial['t'].append(t)
        self.historial['omega'].append(met['omega'])
        self.historial['delta_struct'].append(met['delta_struct'])
        self.historial['LF'].append(met['LF'])
        self.historial['Lambda_Cos'].append(met['Lambda_Cos'])
        self.historial['S_shared'].append(met['S_shared'])
        self.historial['divergencia'].append(met['divergencia'])
        self.historial['dOmega_dt'].append(met['dOmega'])
        self.historial['dLambda_dt'].append(dLambda)
        
        return met


# ============================================================
# EXPERIMENTO V112
# ============================================================
def ejecutar_v112():
    print("\n" + "█" * 100)
    print("EXPERIMENTO V112 — REGISTRO AMPLIADO")
    print("█" * 100)
    
    # Crear agentes con dietas incompatibles
    A = AgenteVST("A", seed=42, filtro_tipo='highpass', filtro_cutoff=2000)
    B = AgenteVST("B", seed=43, filtro_tipo='lowpass', filtro_cutoff=200)
    
    left_path = 'audio_binaural/Blue_Monday_binaural_expandido_left_binaural.wav'
    right_path = 'audio_binaural/Blue_Monday_binaural_expandido_right_binaural.wav'
    
    if not os.path.exists(left_path) or not os.path.exists(right_path):
        print("  ❌ Archivos no encontrados")
        return None, None, False
    
    A.set_frontera(left_path)
    B.set_frontera(right_path)
    
    T_total = EVENTOS['final']
    pasos = int(T_total / DT)
    t_inanicion = EVENTOS['breakdown']
    pasos_inanicion = int(30.0 / DT)
    
    print(f"\n  Simulación: {T_total:.0f}s ({pasos} pasos)")
    print(f"  Inanición de B en t={t_inanicion:.0f}s (duración 30s)")
    print("  Registrando saturación de Φ y Φ_vel")
    
    print("\n" + "=" * 80)
    print("FASE 1: Co-regulación")
    print("=" * 80)
    
    eventos_reportados = set()
    pasos_inicio = int(t_inanicion / DT)
    
    for i in range(pasos_inicio):
        t = i * DT
        met_A = A.actualizar(t, DT, B)
        met_B = B.actualizar(t, DT, A)
        
        for evento, t_e in EVENTOS.items():
            if evento not in eventos_reportados and t >= t_e - 1.0:
                eventos_reportados.add(evento)
                print(f"\n  [EVENTO] {evento} (t={t:.1f}s)")
                print(f"    A: Ω={met_A['omega']:.4f}, Λ={met_A['Lambda_Cos']:.4f}, LF={met_A['LF']:.3f}")
                print(f"    B: Ω={met_B['omega']:.4f}, Λ={met_B['Lambda_Cos']:.4f}, LF={met_B['LF']:.3f}")
                print(f"    S_shared={met_A['S_shared']:.3f}")
                print(f"    SATURACIÓN A: Φ={A.historial['phi_saturado'][-1]}, Φ_vel={A.historial['phi_vel_saturado'][-1]}")
                print(f"    SATURACIÓN B: Φ={B.historial['phi_saturado'][-1]}, Φ_vel={B.historial['phi_vel_saturado'][-1]}")
        
        if i % 2000 == 0 and i > 0:
            print(f"\n  t={t:.1f}s")
            print(f"    A: Ω={met_A['omega']:.4f}, Λ={met_A['Lambda_Cos']:.4f}, LF={met_A['LF']:.3f}")
            print(f"    B: Ω={met_B['omega']:.4f}, Λ={met_B['Lambda_Cos']:.4f}, LF={met_B['LF']:.3f}")
            print(f"    S_shared={met_A['S_shared']:.3f}")
            print(f"    SAT A: Φ=[{A.historial['min_phi'][-1]:.3f}, {A.historial['max_phi'][-1]:.3f}]")
            print(f"    SAT B: Φ=[{B.historial['min_phi'][-1]:.3f}, {B.historial['max_phi'][-1]:.3f}]")
    
    # Calcular zona semiótica empírica
    s_shared_fase1 = A.historial['S_shared'][-2000:] if len(A.historial['S_shared']) > 2000 else A.historial['S_shared']
    S_shared_medio = np.mean(s_shared_fase1) if s_shared_fase1 else 0
    S_shared_std = np.std(s_shared_fase1) if s_shared_fase1 else 0
    umbral_alto = S_shared_medio + 2 * S_shared_std if S_shared_std > 0 else 0.8
    umbral_bajo = max(0, S_shared_medio - 2 * S_shared_std) if S_shared_std > 0 else 0.2
    
    print(f"\n  📊 S_shared medio Fase 1: {S_shared_medio:.3f} ± {S_shared_std:.3f}")
    print(f"  📊 Zona semiótica empírica: {umbral_bajo:.2f}-{umbral_alto:.2f}")
    
    # Calcular saturación total
    sat_total_A = sum(A.historial['phi_saturado'])
    sat_total_B = sum(B.historial['phi_saturado'])
    print(f"  📊 Saturación Φ total: A={sat_total_A}, B={sat_total_B}")
    
    omega_A_before = A.historial['omega'][-1] if A.historial['omega'] else 0.5
    
    print("\n" + "=" * 80)
    print("FASE 2: Inanición gradual en B")
    print("=" * 80)
    
    respuestas_A = []
    
    for i in range(pasos_inanicion):
        t = t_inanicion + i * DT
        B.inducir_inanicion_gradual(i, pasos_inanicion)
        met_A = A.actualizar(t, DT, B)
        met_B = B.actualizar(t, DT, A)
        
        respuesta = abs(met_A['omega'] - omega_A_before)
        respuestas_A.append(respuesta)
        
        if i % 500 == 0:
            print(f"  t={t:.1f}s | Ω_A={met_A['omega']:.4f} (resp={respuesta:.4f}) | Ω_B={met_B['omega']:.4f} | sat_A={A.historial['phi_saturado'][-1]}")
    
    print("\n" + "=" * 80)
    print("FASE 3: Recuperación")
    print("=" * 80)
    
    B.en_inanicion = False
    B.factor_inanicion = 1.0
    B.membrana.reset()
    
    for i in range(pasos_inicio + pasos_inanicion, pasos):
        t = i * DT
        met_A = A.actualizar(t, DT, B)
        met_B = B.actualizar(t, DT, A)
        
        if i % 2000 == 0 and i > pasos_inicio + pasos_inanicion:
            print(f"  t={t:.1f}s | Ω_A={met_A['omega']:.4f} | Ω_B={met_B['omega']:.4f}")
    
    print("\n" + "=" * 80)
    print("ANÁLISIS R₂")
    print("=" * 80)
    
    omega_A_after = A.historial['omega'][-1] if A.historial['omega'] else 0.5
    delta_A = omega_A_after - omega_A_before
    respuesta_max = max(respuestas_A) if respuestas_A else 0
    
    # Umbral de respuesta emergente (2σ de variación basal)
    omega_basal_std = np.std(A.historial['omega'][:2000]) if len(A.historial['omega']) > 2000 else 0.01
    umbral_respuesta = 2 * omega_basal_std
    
    print(f"\n  📊 Datos:")
    print(f"    Variación basal Ω_A: σ={omega_basal_std:.5f}")
    print(f"    Umbral respuesta (2σ): {umbral_respuesta:.5f}")
    print(f"    Ω_A antes: {omega_A_before:.6f}")
    print(f"    Ω_A después: {omega_A_after:.6f}")
    print(f"    ΔA = {delta_A:+.6f}")
    print(f"    Respuesta máxima a inanición: {respuesta_max:.6f}")
    
    # Estadísticas de saturación
    sat_pct_A = sat_total_A / (pasos * DIM_TOTAL) * 100
    sat_pct_B = sat_total_B / (pasos * DIM_TOTAL) * 100
    print(f"\n  📊 Saturación Φ total:")
    print(f"    A: {sat_pct_A:.2f}% del espacio de estados")
    print(f"    B: {sat_pct_B:.2f}% del espacio de estados")
    
    R2_emergente = respuesta_max > umbral_respuesta
    
    print(f"\n  🧬 Diagnóstico R₂ (emergente): {'✅ CONFIRMADO' if R2_emergente else '❌ NO CONFIRMADO'}")
    
    if R2_emergente:
        print("""
    → ALMA RACIONAL: A detectó la inanición de B y respondió.
    """)
    elif umbral_bajo < S_shared_medio < umbral_alto:
        print("""
    → ALMA SENSITIVA++: Co-regulación en zona semiótica.
    """)
    else:
        print("""
    → ALMA VEGETATIVA: Sin co-regulación sostenida.
    """)
    
    # GRÁFICOS
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Ω
    ax = axes[0, 0]
    t_A = A.historial['t']
    ax.plot(t_A, A.historial['omega'], label='A', alpha=0.7)
    ax.plot(t_A[:len(B.historial['omega'])], B.historial['omega'], label='B', alpha=0.7)
    for evento, t_e in EVENTOS.items():
        ax.axvline(x=t_e, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=t_inanicion, color='red', linestyle='-', alpha=0.7)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Ω')
    ax.set_title('Evolución de Ω')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # S_shared
    ax = axes[0, 1]
    ax.plot(t_A[:len(A.historial['S_shared'])], A.historial['S_shared'], color='purple')
    ax.axhline(y=umbral_alto, color='red', linestyle='--', alpha=0.5, label=f'Emergente alto')
    ax.axhline(y=umbral_bajo, color='orange', linestyle='--', alpha=0.5, label=f'Emergente bajo')
    ax.axvline(x=t_inanicion, color='red', linestyle='-', alpha=0.7)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('S_shared')
    ax.set_title('Sentido Compartido')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Λ_Cos
    ax = axes[0, 2]
    ax.plot(t_A[:len(A.historial['Lambda_Cos'])], A.historial['Lambda_Cos'], label='A')
    ax.plot(t_A[:len(B.historial['Lambda_Cos'])], B.historial['Lambda_Cos'], label='B')
    ax.axvline(x=t_inanicion, color='red', linestyle='-', alpha=0.7)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Λ_Cos')
    ax.set_title('Salud dinámica')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Saturación de Φ
    ax = axes[1, 0]
    ax.plot(t_A, A.historial['phi_saturado'], label='A', alpha=0.7)
    ax.plot(t_A[:len(B.historial['phi_saturado'])], B.historial['phi_saturado'], label='B', alpha=0.7)
    ax.axvline(x=t_inanicion, color='red', linestyle='-', alpha=0.7)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Nodos saturados')
    ax.set_title('Saturación de Φ (clip)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Rango de Φ
    ax = axes[1, 1]
    ax.plot(t_A, A.historial['max_phi'], label='A max', alpha=0.5)
    ax.plot(t_A, A.historial['min_phi'], label='A min', alpha=0.5)
    ax.axhline(y=PHI_CLIP_MAX, color='red', linestyle='--', alpha=0.5, label='Clip superior')
    ax.axhline(y=PHI_CLIP_MIN, color='red', linestyle='--', alpha=0.5, label='Clip inferior')
    ax.axvline(x=t_inanicion, color='red', linestyle='-', alpha=0.7)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Φ')
    ax.set_title('Rango de Φ (A)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Respuesta a inanición
    ax = axes[1, 2]
    t_respuesta = np.linspace(t_inanicion, t_inanicion + 30, len(respuestas_A))
    ax.plot(t_respuesta, respuestas_A, color='red')
    ax.axhline(y=umbral_respuesta, color='green', linestyle='--', alpha=0.5, label=f'Umbral (2σ)')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('|ΔΩ_A|')
    ax.set_title('Respuesta de A a inanición de B')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('v112_logs', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'v112_logs/v112_resultados_{timestamp}.png', dpi=150)
    print(f"\n  📊 Gráfico guardado: v112_logs/v112_resultados_{timestamp}.png")
    
    # Exportar datos de saturación
    import json
    datos_saturacion = {
        'sat_A': A.historial['phi_saturado'],
        'sat_B': B.historial['phi_saturado'],
        'max_phi_A': A.historial['max_phi'],
        'min_phi_A': A.historial['min_phi'],
        'max_phi_B': B.historial['max_phi'],
        'min_phi_B': B.historial['min_phi'],
        't': A.historial['t']
    }
    with open(f'v112_logs/v112_saturacion_{timestamp}.json', 'w') as f:
        # Guardar solo primeros 10000 puntos para no explotar
        trimmed = {k: v[:10000] for k, v in datos_saturacion.items() if isinstance(v, list)}
        json.dump(trimmed, f)
    print(f"  📊 Datos de saturación: v112_logs/v112_saturacion_{timestamp}.json")
    
    return A, B, R2_emergente


if __name__ == "__main__":
    A, B, R2 = ejecutar_v112()