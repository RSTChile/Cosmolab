#!/usr/bin/env python3
"""
VSTCosmos v111b — Fisiología corregida + Auditoría completa

Principios:
  1. Membrana sensorial no-shannoniana
  2. Recalibración física (DIFUSION=0.30, GANANCIA=0.005)
  3. Acoplamiento débil y adaptativo
  4. Inanición gradual
  5. e_R como error de predicción
  6. Auditoría COMPLETA: todas las métricas guardadas

Métricas guardadas en cada paso:
  - t, omega, atractor, delta_struct, RC, ICR, IRDE
  - LF, e_R, Lambda_Cos, S_shared, sincronia, fusion, divergencia
  - dOmega_dt, dLambda_dt, dDelta_dt
  - respuesta_evento, respuesta_inanicion
  - A_sys_env (acoplamiento efectivo)
"""

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from datetime import datetime
import os
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt

# ============================================================
# PARÁMETROS RECALIBRADOS
# ============================================================
DT = 0.01
DIM_TOTAL = 120
DIM_INTERNA = 32

DIFUSION_BASE = 0.30
GANANCIA_REACCION = 0.005
AMORT_MIN = 0.05
AMORT_MAX = 0.20
W_MAX = 0.5

GANANCIA_META_BASE = 0.005
PHI_EQUILIBRIO = 0.5
FORZAMIENTO_FRONTERA = 0.1

# Umbrales
UMBRAL_R2 = 0.03
UMBRAL_S_SHARED_BAJO = 0.15
UMBRAL_S_SHARED_ALTO = 0.85
UMBRAL_RESPUESTA_EVENTO = 0.02

EVENTOS = {
    'intro': 0,
    'voz_entra': 90,
    'cambio_ritmo': 180,
    'breakdown': 300,
    'reconstruccion': 390,
    'final': 452
}

print("=" * 100)
print("VSTCosmos v111b — Fisiología corregida + Auditoría completa")
print("  Membrana sensorial no-shannoniana")
print("  Recalibración: DIFUSION=0.30, GANANCIA=0.005")
print("  Auditoría COMPLETA: 20+ métricas guardadas")
print("=" * 100)


# ============================================================
# MEMBRANA SENSORIAL
# ============================================================
class MembranaSensorial:
    def __init__(self, nombre, buffer_size=100):
        self.nombre = nombre
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
        
        perturbacion = (0.4 * inst + 0.3 * envolvente * np.sign(dS) +
                        0.2 * derivada + 0.1 * no_lineal)
        
        return np.clip(perturbacion, -0.2, 0.2)
    
    def reset(self):
        self.buffer = np.zeros(self.buffer_size)
        self.pos = 0
        self.historial = []


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
# CLASE AGENTE V111b (CON AUDITORÍA COMPLETA)
# ============================================================
class AgenteVST_Canonico:
    def __init__(self, nombre, seed=None, filtro_tipo='highpass', filtro_cutoff=2000):
        if seed is not None:
            np.random.seed(seed)
        self.nombre = nombre
        self.filtro_tipo = filtro_tipo
        self.filtro_cutoff = filtro_cutoff

        self.Phi = np.random.normal(PHI_EQUILIBRIO, 0.005, DIM_TOTAL)
        self.Phi_vel = np.zeros(DIM_TOTAL)

        self.W_prof = np.zeros((DIM_INTERNA, DIM_INTERNA))
        self.W_rec = np.zeros((DIM_INTERNA, DIM_INTERNA))

        self.frontera_L = None
        self.frontera_R = None
        self.sr = None
        self.membrana = MembranaSensorial(nombre)
        self.en_inanicion = False
        self.factor_inanicion = 1.0

        # AUDITORÍA COMPLETA
        self.historial = {
            # Básicas
            't': [],
            'omega': [],
            'atractor': [],
            'gradE': [],
            
            # Métricas canónicas
            'delta_struct': [],
            'ICR': [],
            'IRDE': [],
            'RC': [],
            'LF': [],
            'e_R': [],
            'Lambda_Cos': [],
            'S_shared': [],
            'A_sys_env': [],
            
            # Derivadas temporales
            'dOmega_dt': [],
            'dLambda_dt': [],
            'dDelta_dt': [],
            
            # Diagnósticas
            'sincronia': [],      # Correlación con el otro
            'fusion': [],         # 1 - |omega - omega_otro|/2
            'divergencia': [],    # |omega - omega_otro|
            
            # Respuestas
            'respuesta_evento': [],
            'respuesta_inanicion': [],
            
            # Estado
            'en_inanicion': [],
            'factor_inanicion': []
        }

    def set_frontera(self, audio_path):
        data, self.sr = sf.read(audio_path, dtype='float32')
        if data.ndim == 2:
            canal_L = data[:, 0]
            canal_R = data[:, 1]
        else:
            canal_L = data
            canal_R = data
        
        canal_L = aplicar_filtro(canal_L, self.sr, self.filtro_tipo, self.filtro_cutoff)
        canal_R = aplicar_filtro(canal_R, self.sr, self.filtro_tipo, self.filtro_cutoff)
        
        self.frontera_L = canal_L
        self.frontera_R = canal_R
        nombre = os.path.basename(audio_path).replace('.wav', '')
        print(f"  [{self.nombre}] Frontera: {nombre} [{self.filtro_tipo} {self.filtro_cutoff}Hz] ({len(data)/self.sr:.1f}s)")

    def inducir_inanicion_gradual(self, duracion=30.0, paso_actual=0, pasos_totales=3000):
        if paso_actual < pasos_totales:
            self.factor_inanicion = 1.0 - (paso_actual / pasos_totales)
        else:
            self.factor_inanicion = 0.0
            self.en_inanicion = True
        
        if paso_actual % 500 == 0:
            print(f"    [{self.nombre}] Inanición: factor={self.factor_inanicion:.2f}")

    def _get_frontera_t(self, t):
        if self.en_inanicion:
            return 0.0, 0.0
        idx = int(t * self.sr)
        if idx >= len(self.frontera_L):
            return 0.0, 0.0
        L = self.frontera_L[idx] * self.factor_inanicion
        R = self.frontera_R[idx] * self.factor_inanicion
        return L, R

    def _calcular_gradE(self, L, R):
        if abs(L + R) < 1e-10:
            return 0.0
        return (R - L) / (abs(L) + abs(R) + 1e-10)

    def _detectar_atractor(self, omega):
        atractores = [-0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
        return atractores[np.argmin([abs(omega - a) for a in atractores])]

    def _calcular_metricas(self, t, otro_agente=None):
        int_region = self.Phi[:DIM_INTERNA]
        omega = np.mean(int_region)
        
        # 1. Δ_struct
        delta_struct = np.var(int_region)
        
        # 2. ICR, IRDE, RC
        if len(self.historial['omega']) > 1:
            ICR = abs(self.historial['omega'][-1] - self.historial['omega'][-2]) / DT
        else:
            ICR = 0.0
        IRDE = np.var(self.Phi_vel)
        RC = ICR + IRDE
        
        # 3. LF (diversidad de atractores)
        if len(self.historial['atractor']) > 50:
            atractores_recientes = self.historial['atractor'][-50:]
            LF = len(set(atractores_recientes)) / 8.0
        else:
            LF = 0.0
        
        # 4. e_R (error de predicción)
        if len(self.historial['omega']) > 1:
            prediccion = self.historial['omega'][-1]
            e_R = abs(omega - prediccion)
        else:
            e_R = 0.01
        
        # 5. Λ_Cos
        Lambda_Cos = (delta_struct * (LF + 0.01)) / (e_R + 0.01)
        
        # 6. S_shared, sincronia, fusion, divergencia
        S_shared = 0.0
        sincronia = 0.0
        fusion = 0.0
        divergencia = 0.0
        
        if otro_agente is not None:
            omega_otro = np.mean(otro_agente.Phi[:DIM_INTERNA])
            divergencia = abs(omega - omega_otro)
            fusion = 1 - divergencia / 2.0
            
            if len(self.historial['omega']) > 20 and len(otro_agente.historial['omega']) > 20:
                omega_self_reciente = self.historial['omega'][-20:]
                omega_other_reciente = otro_agente.historial['omega'][-20:]
                if len(omega_self_reciente) == len(omega_other_reciente) and len(omega_self_reciente) > 1:
                    if np.std(omega_self_reciente) > 1e-6 and np.std(omega_other_reciente) > 1e-6:
                        corr = np.corrcoef(omega_self_reciente, omega_other_reciente)[0, 1]
                        if not np.isnan(corr):
                            sincronia = max(0.0, min(1.0, corr))
            
            S_shared = fusion if sincronia == 0 else (sincronia + fusion) / 2.0
        
        # 7. A_sys_env (acoplamiento efectivo)
        A_sys_env = 0.0
        if otro_agente is not None:
            diff = np.mean(np.abs(self.Phi - otro_agente.Phi))
            A_sys_env = GANANCIA_META_BASE * (1 - S_shared) * diff
        
        # 8. Derivadas
        dOmega_dt = 0.0
        dLambda_dt = 0.0
        dDelta_dt = 0.0
        
        if len(self.historial['omega']) > 1:
            dOmega_dt = (omega - self.historial['omega'][-1]) / DT
        if len(self.historial['Lambda_Cos']) > 1:
            dLambda_dt = (Lambda_Cos - self.historial['Lambda_Cos'][-1]) / DT
        if len(self.historial['delta_struct']) > 1:
            dDelta_dt = (delta_struct - self.historial['delta_struct'][-1]) / DT
        
        return {
            'omega': omega,
            'delta_struct': delta_struct,
            'ICR': ICR,
            'IRDE': IRDE,
            'RC': RC,
            'LF': LF,
            'e_R': e_R,
            'Lambda_Cos': Lambda_Cos,
            'S_shared': S_shared,
            'sincronia': sincronia,
            'fusion': fusion,
            'divergencia': divergencia,
            'A_sys_env': A_sys_env,
            'dOmega_dt': dOmega_dt,
            'dLambda_dt': dLambda_dt,
            'dDelta_dt': dDelta_dt
        }

    def actualizar(self, t, dt, otro_agente=None):
        L, R = self._get_frontera_t(t)
        dS = L - R
        gradE = self._calcular_gradE(L, R)
        
        perturbacion = self.membrana.procesar(dS)
        
        laplaciano = np.zeros_like(self.Phi)
        for i in range(1, DIM_TOTAL - 1):
            laplaciano[i] = self.Phi[i-1] - 2*self.Phi[i] + self.Phi[i+1]
        difusion = DIFUSION_BASE * laplaciano
        
        reaccion = GANANCIA_REACCION * self.Phi * (1 - self.Phi**2)
        
        forzamiento = np.zeros_like(self.Phi)
        forzamiento[0] = perturbacion * FORZAMIENTO_FRONTERA
        forzamiento[-1] = -perturbacion * FORZAMIENTO_FRONTERA
        
        acoplamiento = np.zeros_like(self.Phi)
        if otro_agente is not None and not self.en_inanicion:
            # Usar S_shared del paso anterior si existe
            s_actual = self.historial['S_shared'][-1] if self.historial['S_shared'] else 0
            k_adaptativo = GANANCIA_META_BASE * (1 - min(1.0, s_actual))
            acoplamiento = k_adaptativo * (otro_agente.Phi - self.Phi)
        
        dPhi_vel = difusion + reaccion + forzamiento + acoplamiento
        self.Phi_vel += dPhi_vel * dt
        self.Phi += self.Phi_vel * dt
        self.Phi = np.clip(self.Phi, -W_MAX, W_MAX)
        self.Phi_vel = np.clip(self.Phi_vel, -2.0, 2.0)
        
        met = self._calcular_metricas(t, otro_agente)
        atractor = self._detectar_atractor(met['omega'])
        
        # Respuesta a evento (se calcula externamente)
        respuesta_evento = 0.0
        
        self.historial['t'].append(t)
        self.historial['omega'].append(met['omega'])
        self.historial['atractor'].append(atractor)
        self.historial['gradE'].append(gradE)
        self.historial['delta_struct'].append(met['delta_struct'])
        self.historial['ICR'].append(met['ICR'])
        self.historial['IRDE'].append(met['IRDE'])
        self.historial['RC'].append(met['RC'])
        self.historial['LF'].append(met['LF'])
        self.historial['e_R'].append(met['e_R'])
        self.historial['Lambda_Cos'].append(met['Lambda_Cos'])
        self.historial['S_shared'].append(met['S_shared'])
        self.historial['sincronia'].append(met['sincronia'])
        self.historial['fusion'].append(met['fusion'])
        self.historial['divergencia'].append(met['divergencia'])
        self.historial['A_sys_env'].append(met['A_sys_env'])
        self.historial['dOmega_dt'].append(met['dOmega_dt'])
        self.historial['dLambda_dt'].append(met['dLambda_dt'])
        self.historial['dDelta_dt'].append(met['dDelta_dt'])
        self.historial['respuesta_evento'].append(respuesta_evento)
        self.historial['en_inanicion'].append(self.en_inanicion)
        self.historial['factor_inanicion'].append(self.factor_inanicion)
        
        return met


# ============================================================
# REPORTE DOBLE DESCRIPCIÓN (ENRIQUECIDO)
# ============================================================
def reporte_doble_descripcion(agente, nombre, evento=None, momento="normal"):
    if len(agente.historial['omega']) < 10:
        return
    
    # Obtener último valor y tendencias
    omega = agente.historial['omega'][-1]
    delta = agente.historial['delta_struct'][-1]
    lf = agente.historial['LF'][-1]
    lam = agente.historial['Lambda_Cos'][-1]
    s_shared = agente.historial['S_shared'][-1]
    dOmega = agente.historial['dOmega_dt'][-1]
    
    # Fenomenología
    if lam < 0.05:
        salud = "coma metabólico"
    elif lam < 0.2:
        salud = "salud frágil"
    elif lam < 0.5:
        salud = "salud estable"
    else:
        salud = "salud robusta"
    
    if lf < 0.2:
        plasticidad = "rígido"
    elif lf < 0.5:
        plasticidad = "plástico moderado"
    else:
        plasticidad = "altamente plástico"
    
    if delta < 0.001:
        alimento = "sin nutrientes"
    elif delta < 0.01:
        alimento = "nutrientes escasos"
    else:
        alimento = "nutrientes abundantes"
    
    if s_shared > UMBRAL_S_SHARED_ALTO:
        vinculo = "fusión"
    elif s_shared > UMBRAL_S_SHARED_BAJO:
        vinculo = "co-regulación"
    else:
        vinculo = "desacople"
    
    tendencia = "↑" if dOmega > 0.01 else ("↓" if dOmega < -0.01 else "→")
    
    evento_str = f" [{evento}]" if evento else ""
    print(f"\n  [{nombre}{evento_str}] {momento}")
    print(f"    🧬 Fenomenológico: {salud}, {plasticidad}, {alimento}, {vinculo}, tendencia {tendencia}")
    print(f"    📊 Operacional: Ω={omega:.4f}, Λ_Cos={lam:.4f}, LF={lf:.3f}, Δ_struct={delta:.5f}, S_shared={s_shared:.3f}")


# ============================================================
# EXPERIMENTO V111b
# ============================================================
def ejecutar_v111b():
    print("\n" + "█" * 100)
    print("EXPERIMENTO V111b — FISIOLOGÍA CORREGIDA + AUDITORÍA COMPLETA")
    print("█" * 100)
    
    A = AgenteVST_Canonico("A", seed=42, filtro_tipo='highpass', filtro_cutoff=2000)
    B = AgenteVST_Canonico("B", seed=43, filtro_tipo='lowpass', filtro_cutoff=200)
    
    print("\n  Verificando archivos...")
    left_path = 'audio_binaural/Blue_Monday_binaural_expandido_left_binaural.wav'
    right_path = 'audio_binaural/Blue_Monday_binaural_expandido_right_binaural.wav'
    
    if not os.path.exists(left_path) or not os.path.exists(right_path):
        print("  ❌ Archivos no encontrados.")
        return None, None, False
    
    A.set_frontera(left_path)
    B.set_frontera(right_path)
    
    T_total = EVENTOS['final']
    pasos = int(T_total / DT)
    t_inanicion = EVENTOS['breakdown']
    
    print(f"\n  Simulación: {T_total:.0f}s ({pasos} pasos)")
    print(f"  Inanición gradual de B en t={t_inanicion:.0f}s (duración 30s)")
    
    print("\n" + "=" * 80)
    print("FASE 1: Co-regulación con dietas incompatibles")
    print("  A: highpass 2kHz, B: lowpass 200Hz")
    print("=" * 80)
    
    eventos_reportados = set()
    pasos_inicio_inanicion = int(t_inanicion / DT)
    
    for i in range(pasos_inicio_inanicion):
        t = i * DT
        met_A = A.actualizar(t, DT, B)
        met_B = B.actualizar(t, DT, A)
        
        for evento, t_evento in EVENTOS.items():
            if evento not in eventos_reportados and t >= t_evento - 1.0:
                eventos_reportados.add(evento)
                # Registrar respuesta al evento (cambio en Ω)
                if len(A.historial['omega']) > 20:
                    omega_antes = np.mean(A.historial['omega'][-20:-1])
                    respuesta_A = abs(met_A['omega'] - omega_antes)
                    A.historial['respuesta_evento'][-1] = respuesta_A
                reporte_doble_descripcion(A, "A", evento, "EVENTO")
                reporte_doble_descripcion(B, "B", evento, "EVENTO")
        
        if i % 2000 == 0 and i > 0:
            print(f"\n  t={t:.1f}s (reporte periódico)")
            print(f"    A: Ω={met_A['omega']:.4f}, Λ={met_A['Lambda_Cos']:.4f}, LF={met_A['LF']:.3f}, dΩ/dt={met_A['dOmega_dt']:+.4f}")
            print(f"    B: Ω={met_B['omega']:.4f}, Λ={met_B['Lambda_Cos']:.4f}, LF={met_B['LF']:.3f}, dΩ/dt={met_B['dOmega_dt']:+.4f}")
            print(f"    S_shared={met_A['S_shared']:.3f}, fusión={met_A['fusion']:.3f}, sincronía={met_A['sincronia']:.3f}")
    
    s_shared_fase1 = A.historial['S_shared'][-2000:] if len(A.historial['S_shared']) > 2000 else A.historial['S_shared']
    S_shared_medio = np.mean(s_shared_fase1) if s_shared_fase1 else 0
    print(f"\n  📊 S_shared medio Fase 1: {S_shared_medio:.3f}")
    
    zona_semiotica = UMBRAL_S_SHARED_BAJO <= S_shared_medio <= UMBRAL_S_SHARED_ALTO
    print(f"  📊 Zona semiótica ({UMBRAL_S_SHARED_BAJO}-{UMBRAL_S_SHARED_ALTO}): {'✅ ALCANZADA' if zona_semiotica else '❌ NO ALCANZADA'}")
    
    omega_A_before = A.historial['omega'][-1] if A.historial['omega'] else 0.5
    omega_B_before = B.historial['omega'][-1] if B.historial['omega'] else 0.5
    
    print("\n" + "=" * 80)
    print("FASE 2: Inanición gradual en B (30s fadeout)")
    print("=" * 80)
    
    pasos_inanicion = int(30.0 / DT)
    
    for i in range(pasos_inanicion):
        t = t_inanicion + i * DT
        B.inducir_inanicion_gradual(30.0, i, pasos_inanicion)
        met_A = A.actualizar(t, DT, B)
        met_B = B.actualizar(t, DT, A)
        
        # Registrar respuesta a inanición
        if i == 0:
            omega_A_base_inanicion = met_A['omega']
        respuesta_inanicion = abs(met_A['omega'] - omega_A_base_inanicion)
        A.historial['respuesta_inanicion'].append(respuesta_inanicion)
        
        if i % 500 == 0:
            print(f"  t={t:.1f}s | Ω_A={met_A['omega']:.4f} (Δ={respuesta_inanicion:.4f}) | Ω_B={met_B['omega']:.4f} | Λ_B={met_B['Lambda_Cos']:.4f} | factor={B.factor_inanicion:.2f}")
    
    print("\n" + "=" * 80)
    print("FASE 3: Recuperación (B vuelve a tener alimento)")
    print("=" * 80)
    
    B.en_inanicion = False
    B.factor_inanicion = 1.0
    B.membrana.reset()
    
    for i in range(pasos_inicio_inanicion + pasos_inanicion, pasos):
        t = i * DT
        met_A = A.actualizar(t, DT, B)
        met_B = B.actualizar(t, DT, A)
        
        if i % 2000 == 0 and i > pasos_inicio_inanicion + pasos_inanicion:
            print(f"  t={t:.1f}s | Ω_A={met_A['omega']:.4f} | Ω_B={met_B['omega']:.4f} | Λ_B={met_B['Lambda_Cos']:.4f}")
    
    print("\n" + "=" * 80)
    print("ANÁLISIS R₂")
    print("=" * 80)
    
    idx_inicio = pasos_inicio_inanicion
    idx_fin = idx_inicio + pasos_inanicion
    
    omega_A_durante = A.historial['omega'][idx_inicio:idx_fin] if len(A.historial['omega']) > idx_fin else []
    omega_B_durante = B.historial['omega'][idx_inicio:idx_fin] if len(B.historial['omega']) > idx_fin else []
    respuesta_inanicion_lista = A.historial['respuesta_inanicion']
    
    omega_A_after = omega_A_durante[-1] if len(omega_A_durante) > 0 else omega_A_before
    delta_A = omega_A_after - omega_A_before
    respuesta_max = max(respuesta_inanicion_lista) if respuesta_inanicion_lista else 0
    
    if len(omega_B_durante) > 10:
        dOmega_B = np.mean(np.diff(omega_B_durante[-100:])) if len(omega_B_durante) > 100 else np.mean(np.diff(omega_B_durante))
    else:
        dOmega_B = 0.0
    
    print(f"\n  📊 Datos:")
    print(f"    Omega_A antes inanición: {omega_A_before:.6f}")
    print(f"    Omega_A durante/después: {omega_A_after:.6f}")
    print(f"    ΔA = {delta_A:+.6f}")
    print(f"    Respuesta máxima A a inanición: {respuesta_max:.6f}")
    print(f"    Tendencia B durante inanición: {dOmega_B:+.6f}")
    
    # R₂ si: A cambia significativamente Y en dirección opuesta a B
    R2_detectado = (abs(delta_A) > UMBRAL_R2 or respuesta_max > UMBRAL_R2) and dOmega_B != 0 and np.sign(delta_A) == -np.sign(dOmega_B)
    
    print(f"\n  🧬 Diagnóstico R₂: {'✅ CONFIRMADO' if R2_detectado else '❌ NO CONFIRMADO'}")
    
    if R2_detectado:
        print("""
    → INTERPRETACIÓN BIOLÓGICA: A detectó la inanición de B y modificó
       su postura en dirección opuesta. DISENSO confirmado.

    → CONCLUSIÓN CANÓNICA: R₂ = True. ALMA RACIONAL.
    """)
    elif zona_semiotica:
        print("""
    → INTERPRETACIÓN BIOLÓGICA: Co-regulación en zona semiótica,
       pero A no respondió selectivamente a la inanición de B.

    → CONCLUSIÓN CANÓNICA: Alma Sensitiva++. Hay vínculo, no hay R₂.
    """)
    else:
        print("""
    → INTERPRETACIÓN BIOLÓGICA: No hubo co-regulación sostenida.

    → CONCLUSIÓN CANÓNICA: Alma Vegetativa.
    """)
    
    # GRÁFICOS
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Ω
    ax = axes[0, 0]
    t_A = A.historial['t'][:len(A.historial['omega'])]
    t_B = B.historial['t'][:len(B.historial['omega'])]
    ax.plot(t_A, A.historial['omega'], label='A (highpass)', alpha=0.7)
    ax.plot(t_B, B.historial['omega'], label='B (lowpass)', alpha=0.7)
    for evento, t_e in EVENTOS.items():
        ax.axvline(x=t_e, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=t_inanicion, color='red', linestyle='-', linewidth=2, alpha=0.7)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Ω')
    ax.set_title('Evolución de Ω')
    ax.legend()
    ax.set_ylim(-0.5, 1.0)
    ax.grid(True, alpha=0.3)
    
    # S_shared
    ax = axes[0, 1]
    ax.plot(t_A[:len(A.historial['S_shared'])], A.historial['S_shared'], color='purple')
    ax.axhline(y=UMBRAL_S_SHARED_ALTO, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=UMBRAL_S_SHARED_BAJO, color='orange', linestyle='--', alpha=0.5)
    ax.axvline(x=t_inanicion, color='red', linestyle='-', linewidth=2, alpha=0.7)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('S_shared')
    ax.set_title('Sentido Compartido')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Λ_Cos
    ax = axes[0, 2]
    ax.plot(t_A[:len(A.historial['Lambda_Cos'])], A.historial['Lambda_Cos'], label='A')
    ax.plot(t_B[:len(B.historial['Lambda_Cos'])], B.historial['Lambda_Cos'], label='B')
    ax.axvline(x=t_inanicion, color='red', linestyle='-', linewidth=2, alpha=0.7)
    ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Umbral (>0.5)')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Λ_Cos')
    ax.set_title('Salud dinámica')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Δ_struct
    ax = axes[1, 0]
    ax.plot(t_A[:len(A.historial['delta_struct'])], A.historial['delta_struct'], label='A')
    ax.plot(t_B[:len(B.historial['delta_struct'])], B.historial['delta_struct'], label='B')
    ax.axvline(x=t_inanicion, color='red', linestyle='-', linewidth=2, alpha=0.7)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Δ_struct')
    ax.set_title('Alimento disponible')
    ax.legend()
    ax.set_ylim(0, 0.1)
    ax.grid(True, alpha=0.3)
    
    # LF (Plasticidad)
    ax = axes[1, 1]
    ax.plot(t_A[:len(A.historial['LF'])], A.historial['LF'], label='A')
    ax.plot(t_B[:len(B.historial['LF'])], B.historial['LF'], label='B')
    ax.axvline(x=t_inanicion, color='red', linestyle='-', linewidth=2, alpha=0.7)
    ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Alta plasticidad')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('LF')
    ax.set_title('Plasticidad')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Respuesta a inanición
    ax = axes[1, 2]
    if respuesta_inanicion_lista:
        t_inanicion_axis = np.linspace(t_inanicion, t_inanicion + 30, len(respuesta_inanicion_lista))
        ax.plot(t_inanicion_axis, respuesta_inanicion_lista, color='red')
        ax.axhline(y=UMBRAL_R2, color='green', linestyle='--', alpha=0.5, label=f'Umbral R₂ ({UMBRAL_R2})')
        ax.set_xlabel('Tiempo (s)')
        ax.set_ylabel('|ΔΩ_A|')
        ax.set_title('Respuesta de A a inanición de B')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('v111b_logs', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'v111b_logs/v111b_resultados_{timestamp}.png', dpi=150)
    print(f"\n  📊 Gráfico guardado: v111b_logs/v111b_resultados_{timestamp}.png")
    
    # Exportar datos para auditoría externa
    import json
    datos_exportar = {
        't': A.historial['t'],
        'omega_A': A.historial['omega'],
        'omega_B': B.historial['omega'],
        'Lambda_Cos_A': A.historial['Lambda_Cos'],
        'Lambda_Cos_B': B.historial['Lambda_Cos'],
        'S_shared': A.historial['S_shared'],
        'LF_A': A.historial['LF'],
        'LF_B': B.historial['LF'],
        'delta_struct_A': A.historial['delta_struct'],
        'delta_struct_B': B.historial['delta_struct'],
        'respuesta_A': respuesta_inanicion_lista
    }
    
    with open(f'v111b_logs/v111b_datos_{timestamp}.json', 'w') as f:
        json.dump({k: [float(x) if not isinstance(x, (int, float)) else x for x in v[:1000]] 
                   for k, v in datos_exportar.items()}, f)
    print(f"  📊 Datos exportados: v111b_logs/v111b_datos_{timestamp}.json")
    
    return A, B, R2_detectado


if __name__ == "__main__":
    A, B, R2 = ejecutar_v111b()