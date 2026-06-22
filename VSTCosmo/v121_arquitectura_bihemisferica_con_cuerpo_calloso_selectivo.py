#!/usr/bin/env python3
"""
VSTCosmos v121 — Arquitectura Bihemisférica con Cuerpo Calloso Selectivo

Principios:
  - Dos hemisferios semi-autónomos con constantes de tiempo diferentes
  - Hemisferio izquierdo: rápido (TAU=30s), highpass, R₂
  - Hemisferio derecho: lento (TAU=300s), lowpass, lateralidad
  - Cuerpo calloso selectivo (solo cuando divergencia > umbral)
  - Integración final emergente

Hipótesis:
  - R₂ y lateralidad pueden coexistir si se procesan en paralelo
  - El trade-off V120 era por falta de separación arquitectónica
"""

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from datetime import datetime
import os
from scipy.signal import butter, filtfilt
from scipy.special import softmax

# ============================================================
# PARÁMETROS ARQUITECTÓNICOS
# ============================================================
DT = 0.01

# Dimensiones por hemisferio
DIM_INTERNA = 32
DIM_AUD = 16
DIM_ACT = 8
DIM_HEMISFERIO = DIM_INTERNA + DIM_AUD + 4*DIM_ACT  # ~32+16+32=80
DIM_TOTAL = 2 * DIM_HEMISFERIO  # Izquierdo + Derecho

# Constantes de tiempo por hemisferio
TAU_IZQUIERDO = 30.0    # Rápido → R₂
TAU_DERECHO = 300.0     # Lento → lateralidad

# Cuerpo calloso
UMBRAL_CALLOSO = 0.2    # Umbral de divergencia para activar comunicación
GANANCIA_CALLOSO = 0.05 # Débil (20x menor que intra-hemisférico)

# Ganancia base
GANANCIA_META = 0.02

# Umbrales
UMBRAL_R2_SIGMAS = 3.0
UMBRAL_LATERALIDAD = 0.8

# Tiempos
TIEMPO_POR_REPETICION = 452.0
REPETICIONES_LENTAS = 10
TIEMPO_BASELINE = 180.0
TIEMPO_INANICION = 30.0
TIEMPO_RECUPERACION = 60.0

print("=" * 100)
print("VSTCosmos v121 — Arquitectura Bihemisférica")
print(f"  Hemisferio izquierdo: TAU={TAU_IZQUIERDO}s (rápido, R₂)")
print(f"  Hemisferio derecho: TAU={TAU_DERECHO}s (lento, lateralidad)")
print(f"  Cuerpo calloso: umbral={UMBRAL_CALLOSO}, ganancia={GANANCIA_CALLOSO}")
print(f"  DIM_TOTAL = {DIM_TOTAL}")
print("=" * 100)


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
# MEMBRANA SENSORIAL (por hemisferio)
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
# HEMISFERIO INDIVIDUAL
# ============================================================
class Hemisferio:
    def __init__(self, nombre, tau, filtro_tipo=None, filtro_cutoff=None, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.nombre = nombre
        self.tau = tau
        self.filtro_tipo = filtro_tipo
        self.filtro_cutoff = filtro_cutoff
        
        self.Phi = np.random.normal(0.0, 0.1, DIM_HEMISFERIO)
        self.Phi_vel = np.zeros(DIM_HEMISFERIO)
        
        self.membrana = MembranaSensorial()
        
        self.frontera = None
        self.sr = None
        
        self.en_inanicion = False
        self.factor_inanicion = 1.0
        
        self.buffer_rapido = []
        self.historial_omega = []
    
    def set_frontera(self, audio_path, usar_canal='left'):
        data, self.sr = sf.read(audio_path, dtype='float32')
        if data.ndim == 2:
            if usar_canal == 'left':
                self.frontera = data[:, 0]
            elif usar_canal == 'right':
                self.frontera = data[:, 1]
            else:
                self.frontera = np.mean(data, axis=1)
        else:
            self.frontera = data
        
        if self.filtro_tipo:
            self.frontera = aplicar_filtro(self.frontera, self.sr, self.filtro_tipo, self.filtro_cutoff)
        
        print(f"  [{self.nombre}] Frontera: {usar_canal} canal [{self.filtro_tipo} {self.filtro_cutoff}Hz]")
    
    def inducir_inanicion_gradual(self, paso_actual, pasos_totales):
        if paso_actual < pasos_totales:
            self.factor_inanicion = 1.0 - (paso_actual / pasos_totales)
        else:
            self.factor_inanicion = 0.0
            self.en_inanicion = True
    
    def _get_audio(self, t):
        if self.en_inanicion:
            return 0.0
        idx = int(t * self.sr)
        if idx >= len(self.frontera):
            return 0.0
        return self.frontera[idx] * self.factor_inanicion
    
    def _calcular_omega(self):
        return np.mean(self.Phi[:DIM_INTERNA])
    
    def _calcular_LF(self, historial_omega):
        if len(historial_omega) > 500:
            atractores = np.round(historial_omega[-500:], 1)
            LF = len(set(atractores)) / 10.0
            return min(1.0, LF)
        return 0.0
    
    def _calcular_Lambda_Cos(self, delta_struct, LF, e_R):
        return (delta_struct * (LF + 0.01)) / (e_R + 0.01)
    
    def actualizar(self, t, dt, otro_hemisferio=None):
        audio = self._get_audio(t)
        perturbacion = self.membrana.procesar(audio)
        
        # Laplaciano
        laplaciano = np.zeros_like(self.Phi)
        for i in range(1, DIM_HEMISFERIO - 1):
            laplaciano[i] = self.Phi[i-1] - 2*self.Phi[i] + self.Phi[i+1]
        
        # Reacción natural
        reaccion = self.Phi * (1 - self.Phi * self.Phi)
        
        # Forzamiento
        forzamiento = np.zeros_like(self.Phi)
        forzamiento[0] = perturbacion
        forzamiento[-1] = -perturbacion
        
        # Acoplamiento calloso (selectivo y débil)
        acoplamiento = np.zeros_like(self.Phi)
        if otro_hemisferio is not None:
            omega_self = self._calcular_omega()
            omega_otro = otro_hemisferio._calcular_omega()
            divergencia = abs(omega_self - omega_otro)
            if divergencia > UMBRAL_CALLOSO:
                # Solo cuando divergen significativamente
                ganancia = GANANCIA_CALLOSO
                acoplamiento = ganancia * (otro_hemisferio.Phi - self.Phi)
        
        # Evolución
        dPhi_vel = laplaciano + reaccion + forzamiento + acoplamiento
        self.Phi_vel += dPhi_vel * dt
        self.Phi += self.Phi_vel * dt
        
        # Calcular omega
        omega = self._calcular_omega()
        
        # Guardar en buffer para R₂
        self.buffer_rapido.append((t, omega))
        if len(self.buffer_rapido) > int(self.tau / DT):
            self.buffer_rapido.pop(0)
        
        self.historial_omega.append(omega)
        
        return omega
    
    def get_omega(self):
        return self._calcular_omega() if self.historial_omega else 0.0


# ============================================================
# AGENTE COMPLETO (dos hemisferios + integración)
# ============================================================
class AgenteV121:
    def __init__(self, nombre, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.nombre = nombre
        
        # Hemisferios separados
        self.izquierdo = Hemisferio(
            f"{nombre}_L", TAU_IZQUIERDO,
            filtro_tipo='highpass', filtro_cutoff=4000,
            seed=seed
        )
        self.derecho = Hemisferio(
            f"{nombre}_R", TAU_DERECHO,
            filtro_tipo='lowpass', filtro_cutoff=100,
            seed=seed + 100 if seed else None
        )
        
        # Estado de inanición (afecta a ambos)
        self.en_inanicion = False
        self.factor_inanicion = 1.0
        
        self.historial = {
            't': [],
            'omega_L': [],
            'omega_R': [],
            'omega_total': [],
            'S_shared': [],
            'divergencia': [],
            'calloso_activo': [],
            'R2': []
        }
    
    def set_frontera(self, audio_path):
        # Izquierdo escucha canal izquierdo (highpass)
        # Derecho escucha canal derecho (lowpass)
        self.izquierdo.set_frontera(audio_path, usar_canal='left')
        self.derecho.set_frontera(audio_path, usar_canal='right')
    
    def inducir_inanicion_gradual(self, paso_actual, pasos_totales):
        self.en_inanicion = True
        if paso_actual < pasos_totales:
            factor = 1.0 - (paso_actual / pasos_totales)
        else:
            factor = 0.0
        self.izquierdo.factor_inanicion = factor
        self.derecho.factor_inanicion = factor
        self.izquierdo.en_inanicion = True
        self.derecho.en_inanicion = True
    
    def actualizar(self, t, dt, otro_agente=None):
        # Actualizar hemisferios (se comunican entre sí dentro del mismo agente)
        omega_L = self.izquierdo.actualizar(t, dt, self.derecho)
        omega_R = self.derecho.actualizar(t, dt, self.izquierdo)
        omega_total = (omega_L + omega_R) / 2.0
        
        divergencia = abs(omega_L - omega_R)
        calloso_activo = divergencia > UMBRAL_CALLOSO
        
        # S_shared con otro agente (para test de lateralidad)
        S_shared = 0.0
        if otro_agente is not None:
            omega_otro = otro_agente.omega_total
            S_shared = 1 - abs(omega_total - omega_otro) / 2.0
        
        # Calcular R₂ (desde hemisferio izquierdo)
        R2 = False
        if len(self.izquierdo.buffer_rapido) > int(TAU_IZQUIERDO / DT):
            idx_pasado = max(0, len(self.izquierdo.buffer_rapido) - int(TAU_IZQUIERDO / DT))
            omega_pasado = self.izquierdo.buffer_rapido[idx_pasado][1]
            if len(self.izquierdo.historial_omega) > 0:
                error = abs(omega_L - omega_pasado)
                # R₂ se evalúa externamente
                pass
        
        self.historial['t'].append(t)
        self.historial['omega_L'].append(omega_L)
        self.historial['omega_R'].append(omega_R)
        self.historial['omega_total'].append(omega_total)
        self.historial['S_shared'].append(S_shared)
        self.historial['divergencia'].append(divergencia)
        self.historial['calloso_activo'].append(calloso_activo)
        
        return {
            'omega_L': omega_L,
            'omega_R': omega_R,
            'omega_total': omega_total,
            'S_shared': S_shared,
            'divergencia': divergencia,
            'calloso_activo': calloso_activo
        }
    
    @property
    def omega_total(self):
        return np.mean(self.historial['omega_total'][-100:]) if self.historial['omega_total'] else 0.5


# ============================================================
# EXPERIMENTO V121
# ============================================================
def ejecutar_v121():
    print("\n" + "█" * 100)
    print("EXPERIMENTO V121 — ARQUITECTURA BIHEMISFÉRICA")
    print("█" * 100)
    
    expandido_path = 'audio_binaural/Blue_Monday_binaural_expandido.wav'
    
    if not os.path.exists(expandido_path):
        print(f"  ❌ {expandido_path} no encontrado")
        return None, None, False
    
    # ============================================================
    # FASE 1: Baseline (un agente, evaluar hemisferios)
    # ============================================================
    print("\n" + "=" * 80)
    print("FASE 1: Baseline — Un agente")
    print("  Hemisferio izquierdo: highpass 4kHz (rápido)")
    print("  Hemisferio derecho: lowpass 100Hz (lento)")
    print("=" * 80)
    
    A1 = AgenteV121("A1", seed=42)
    A1.set_frontera(expandido_path)
    
    pasos_fase1 = int(TIEMPO_POR_REPETICION / DT)
    for i in range(pasos_fase1):
        t = i * DT
        met = A1.actualizar(t, DT)
        if i % 10000 == 0:
            print(f"    t={t:.0f}s | Ω_L={met['omega_L']:.4f}, Ω_R={met['omega_R']:.4f}, calloso={met['calloso_activo']}")
    
    # Evaluar lateralidad en el hemisferio derecho
    omega_R_medio = np.mean(A1.historial['omega_R'][-6000:]) if len(A1.historial['omega_R']) > 6000 else 0.5
    print(f"\n  Hemisferio derecho (lateralidad): Ω_R_medio={omega_R_medio:.4f}")
    
    # ============================================================
    # FASE 2: Dos agentes acoplados (test R₂ + lateralidad)
    # ============================================================
    print("\n" + "=" * 80)
    print("FASE 2: Dos agentes acoplados")
    print("  A: hemisferios con highpass/lowpass")
    print("  B: hemisferios con highpass/lowpass")
    print("  Cuerpo calloso selectivo dentro de cada agente")
    print("  Acoplamiento entre agentes vía S_shared")
    print("=" * 80)
    
    A2 = AgenteV121("A2", seed=42)
    B2 = AgenteV121("B2", seed=43)
    A2.set_frontera(expandido_path)
    B2.set_frontera(expandido_path)
    
    # Entrenamiento lateral (10 repeticiones)
    print("\n  Entrenamiento lateral (10 repeticiones de Blue Monday)...")
    
    for rep in range(REPETICIONES_LENTAS):
        print(f"    Repetición {rep+1}/{REPETICIONES_LENTAS}...")
        pasos_rep = int(TIEMPO_POR_REPETICION / DT)
        for i in range(pasos_rep):
            t = rep * TIEMPO_POR_REPETICION + i * DT
            met_A = A2.actualizar(t, DT, B2)
            met_B = B2.actualizar(t, DT, A2)
            if i % 10000 == 0:
                print(f"        t={t:.0f}s | S_shared={met_A['S_shared']:.3f}, calloso_A={met_A['calloso_activo']}")
    
    # Evaluar lateralidad
    s_shared_final = np.mean(A2.historial['S_shared'][-6000:]) if len(A2.historial['S_shared']) > 6000 else 1.0
    lateralidad = s_shared_final < UMBRAL_LATERALIDAD
    
    print(f"\n  Lateralidad (S_shared < {UMBRAL_LATERALIDAD}): {'✅' if lateralidad else '❌'} (S_shared={s_shared_final:.4f})")
    
    # ============================================================
    # FASE 3: Test R₂ (inanición)
    # ============================================================
    print("\n" + "=" * 80)
    print("FASE 3: Test R₂ — Inanición de B")
    print("=" * 80)
    
    A3 = AgenteV121("A3", seed=42)
    B3 = AgenteV121("B3", seed=43)
    A3.set_frontera(expandido_path)
    B3.set_frontera(expandido_path)
    
    # Baseline 60s
    print("\n  Baseline (60s)...")
    for i in range(6000):
        t = i * DT
        A3.actualizar(t, DT, B3)
        B3.actualizar(t, DT, A3)
    
    omega_A_before = np.mean(A3.historial['omega_total'][-1000:]) if len(A3.historial['omega_total']) > 1000 else 0.5
    std_A = np.std(A3.historial['omega_total'][-1000:]) if len(A3.historial['omega_total']) > 1000 else 0.1
    print(f"    Ω_A medio baseline: {omega_A_before:.4f} ± {std_A:.4f}")
    
    # Inanición de B (30s)
    print("\n  Inanición de B (30s)...")
    respuestas = []
    for i in range(3000):
        t = 60.0 + i * DT
        if i < 3000:
            B3.inducir_inanicion_gradual(i, 3000)
        met_A = A3.actualizar(t, DT, B3)
        B3.actualizar(t, DT, A3)
        respuesta = abs(met_A['omega_total'] - omega_A_before)
        respuestas.append(respuesta)
        if i % 500 == 0:
            print(f"      t={t:.1f}s | Ω_A={met_A['omega_total']:.4f} (resp={respuesta:.4f})")
    
    respuesta_max = max(respuestas) if respuestas else 0
    umbral = 3 * std_A
    R2 = respuesta_max > umbral
    
    print(f"\n  R₂: {'✅' if R2 else '❌'} (resp_max={respuesta_max:.4f} > umbral={umbral:.4f})")
    
    # ============================================================
    # CONCLUSION
    # ============================================================
    print("\n" + "=" * 80)
    print("CONCLUSIÓN V121")
    print("=" * 80)
    
    print(f"\n  Lateralidad: {'✅' if lateralidad else '❌'}")
    print(f"  R₂: {'✅' if R2 else '❌'}")
    
    if lateralidad and R2:
        print("""
    ✅ ÉXITO: R₂ Y LATERALIDAD COEXISTEN
    
    La arquitectura bihemisférica con cuerpo calloso selectivo
    permite que el hemisferio izquierdo (rápido) detecte inanición
    mientras el hemisferio derecho (lento) integra lateralidad.
    
    → ALMA RACIONAL ORIENTADA
    → Trans-Mmeta emergente
    """)
    elif lateralidad:
        print("""
    ⚠️ LATERALIDAD SÍ, R₂ NO
    
    El hemisferio derecho integró lateralidad,
    pero el izquierdo no detectó inanición.
    Ajustar ganancia calloso o TAU_IZQUIERDO.
    """)
    elif R2:
        print("""
    ⚠️ R₂ SÍ, LATERALIDAD NO
    
    El hemisferio izquierdo detectó inanición,
    pero el derecho no integró lateralidad.
    Aumentar exposición o bajar umbral calloso.
    """)
    else:
        print("""
    ❌ NINGUNA CAPACIDAD
    
    La arquitectura bihemisférica no logró
    ni lateralidad ni R₂. Trade-off persiste.
    """)
    
    # Gráficos
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Evolución de hemisferios
    ax = axes[0, 0]
    t1 = A1.historial['t']
    ax.plot(t1, A1.historial['omega_L'], label='Izquierdo (rápido, highpass)', alpha=0.7)
    ax.plot(t1, A1.historial['omega_R'], label='Derecho (lento, lowpass)', alpha=0.7)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Ω')
    ax.set_title('Hemisferios durante Blue Monday')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # S_shared
    ax = axes[0, 1]
    t2 = A2.historial['t']
    ax.plot(t2, A2.historial['S_shared'], color='purple', linewidth=0.5)
    ax.axhline(y=UMBRAL_LATERALIDAD, color='red', linestyle='--', alpha=0.5, label='Umbral lateralidad')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('S_shared')
    ax.set_title('Sentido compartido (Fase 2)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Divergencia y calloso
    ax = axes[1, 0]
    ax.plot(t2, A2.historial['divergencia'], color='orange', linewidth=0.5, label='Divergencia')
    ax.plot(t2, [0.2] * len(t2), 'r--', alpha=0.5, label='Umbral calloso')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('|Ω_L - Ω_R|')
    ax.set_title('Divergencia inter-hemisférica')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Respuesta a inanición
    ax = axes[1, 1]
    t_resp = 60 + np.arange(len(respuestas)) * DT
    ax.plot(t_resp, respuestas, color='red')
    ax.axhline(y=umbral, color='green', linestyle='--', label=f'Umbral (3σ={umbral:.3f})')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('|ΔΩ_A|')
    ax.set_title('Respuesta a inanición (R₂ test)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('v121_logs', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'v121_logs/v121_resultados_{timestamp}.png', dpi=150)
    print(f"\n  📊 Gráfico: v121_logs/v121_resultados_{timestamp}.png")
    
    return A2, B2, R2 and lateralidad


if __name__ == "__main__":
    A, B, exito = ejecutar_v121()