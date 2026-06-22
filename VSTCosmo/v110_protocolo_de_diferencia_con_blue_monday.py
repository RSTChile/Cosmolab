#!/usr/bin/env python3
"""
VSTCosmos v110 — Protocolo de Diferencia con Blue Monday

Principios Canónicos:
  - C-N4: S = f(I, ∂S). Audio es frontera, no se procesa.
  - C-N9.2: compatibilidad(R₁, R₂) > 0 y < 1
  - O-N3.4a: Test R₂ = A nota inanición de B
  - Doble descripción: biológica + operacional

Dietas:
  A: Blue_Monday_left_binaural.wav (fuente -60°)
  B: Blue_Monday_right_binaural.wav (fuente +60°)

Eventos Blue Monday:
  - t=90s: entrada de voz (sobresalto esperado)
  - t=180s: cambio de ritmo (transición)
  - t=300s: breakdown (inanición)
  - t=390s: reconstrucción
"""

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from datetime import datetime
import os
from scipy.ndimage import gaussian_filter1d

# ============================================================
# PARÁMETROS CANÓNICOS
# ============================================================
DT = 0.01
DIM_TOTAL = 120
DIM_INTERNA = 32
DIFUSION_BASE = 0.15
GANANCIA_REACCION = 0.05
GANANCIA_META = 0.08
PHI_EQUILIBRIO = 0.5
FORZAMIENTO_FRONTERA = 0.1

# Eventos de Blue Monday (segundos)
EVENTOS = {
    'intro': 0,
    'voz_entra': 90,
    'cambio_ritmo': 180,
    'breakdown': 300,
    'reconstruccion': 390,
    'final': 452
}

print("=" * 100)
print("VSTCosmos v110 — Protocolo de Diferencia")
print("  Dietas incompatibles: A = Left, B = Right")
print("  Test R₂: ¿A nota inanición de B?")
print(f"  Eventos: voz={EVENTOS['voz_entra']}s, breakdown={EVENTOS['breakdown']}s")
print("=" * 100)


# ============================================================
# CLASE AGENTE CANÓNICO V110
# ============================================================
class AgenteVST_Canonico:
    def __init__(self, nombre, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.nombre = nombre

        self.Phi = np.random.normal(PHI_EQUILIBRIO, 0.01, DIM_TOTAL)
        self.Phi_vel = np.zeros(DIM_TOTAL)

        self.W_prof = np.zeros((DIM_INTERNA, DIM_INTERNA))
        self.W_rec = np.zeros((DIM_INTERNA, DIM_INTERNA))

        self.frontera_L = None
        self.frontera_R = None
        self.sr = None
        self.en_inanicion = False

        self.historial = {
            't': [],
            'omega': [],
            'atractor': [],
            'delta_struct': [],
            'RC': [],
            'LF': [],
            'S_shared': [],
            'Lambda_Cos': [],
            'en_inanicion': []
        }

    def set_frontera(self, audio_path):
        data, self.sr = sf.read(audio_path, dtype='float32')
        if data.ndim == 2:
            self.frontera_L = data[:, 0]
            self.frontera_R = data[:, 1]
        else:
            self.frontera_L = data
            self.frontera_R = data
        nombre = os.path.basename(audio_path).replace('.wav', '')
        print(f"  [{self.nombre}] Frontera: {nombre} ({len(data)/self.sr:.1f}s)")

    def inducir_inanicion(self):
        self.en_inanicion = True
        print(f"  [{self.nombre}] INANICIÓN INDUCIDA (silencio)")

    def _get_frontera_t(self, t):
        if self.en_inanicion:
            return 0.0, 0.0
        idx = int(t * self.sr)
        if idx >= len(self.frontera_L):
            return 0.0, 0.0
        return self.frontera_L[idx], self.frontera_R[idx]

    def _detectar_atractor(self, omega):
        atractores = [-0.5, 0.0, 0.1, 0.6, 0.7, 0.9, 1.0]
        return atractores[np.argmin([abs(omega - a) for a in atractores])]

    def _calcular_metricas(self, otro_agente=None):
        int_region = self.Phi[:DIM_INTERNA]
        omega = np.mean(int_region)

        delta_struct = np.var(int_region)

        if len(self.historial['omega']) > 1:
            ICR = abs(self.historial['omega'][-1] - self.historial['omega'][-2]) / DT
        else:
            ICR = 0.0
        IRDE = np.var(self.Phi_vel)
        RC = ICR + IRDE

        if len(self.historial['atractor']) > 50:
            atractores_recientes = self.historial['atractor'][-50:]
            LF = len(set(atractores_recientes)) / 7.0
        else:
            LF = 0.0

        e_R = abs(omega - PHI_EQUILIBRIO)
        Lambda_Cos = (delta_struct * LF) / (e_R + 1e-10)

        # S_shared: sincronía
        S_shared = 0.0
        if otro_agente is not None:
            omega_otro = np.mean(otro_agente.Phi[:DIM_INTERNA])
            if len(self.historial['omega']) > 10 and len(otro_agente.historial['omega']) > 10:
                omega_self_reciente = self.historial['omega'][-10:]
                omega_other_reciente = otro_agente.historial['omega'][-10:]
                if len(omega_self_reciente) == len(omega_other_reciente) and len(omega_self_reciente) > 1:
                    corr = np.corrcoef(omega_self_reciente, omega_other_reciente)[0, 1]
                    if not np.isnan(corr):
                        S_shared = max(0.0, min(1.0, corr))
            if S_shared == 0.0:
                S_shared = 1 - abs(omega - omega_otro) / 2.0

        return {
            'omega': omega,
            'delta_struct': delta_struct,
            'RC': RC,
            'LF': LF,
            'Lambda_Cos': Lambda_Cos,
            'S_shared': S_shared,
            'e_R': e_R
        }

    def actualizar(self, t, dt, otro_agente=None):
        L, R = self._get_frontera_t(t)
        dS = L - R

        laplaciano = np.zeros_like(self.Phi)
        for i in range(1, DIM_TOTAL - 1):
            laplaciano[i] = self.Phi[i-1] - 2*self.Phi[i] + self.Phi[i+1]
        difusion = DIFUSION_BASE * laplaciano
        reaccion = GANANCIA_REACCION * self.Phi * (1 - self.Phi**2)

        forzamiento = np.zeros_like(self.Phi)
        forzamiento[0] = dS * FORZAMIENTO_FRONTERA
        forzamiento[-1] = -dS * FORZAMIENTO_FRONTERA

        acoplamiento = np.zeros_like(self.Phi)
        if otro_agente is not None and not self.en_inanicion:
            acoplamiento = GANANCIA_META * (otro_agente.Phi - self.Phi)

        dPhi_vel = difusion + reaccion + forzamiento + acoplamiento
        self.Phi_vel += dPhi_vel * dt
        self.Phi += self.Phi_vel * dt
        self.Phi = np.clip(self.Phi, -1.0, 1.0)
        self.Phi_vel = np.clip(self.Phi_vel, -5.0, 5.0)

        met = self._calcular_metricas(otro_agente)
        atractor = self._detectar_atractor(met['omega'])

        self.historial['t'].append(t)
        self.historial['omega'].append(met['omega'])
        self.historial['atractor'].append(atractor)
        self.historial['delta_struct'].append(met['delta_struct'])
        self.historial['RC'].append(met['RC'])
        self.historial['LF'].append(met['LF'])
        self.historial['S_shared'].append(met['S_shared'])
        self.historial['Lambda_Cos'].append(met['Lambda_Cos'])
        self.historial['en_inanicion'].append(self.en_inanicion)

        return met


# ============================================================
# REPORTE DOBLE DESCRIPCIÓN
# ============================================================
def reporte_doble_descripcion(agente, nombre, evento=None):
    if len(agente.historial['omega']) < 10:
        return

    omega_actual = agente.historial['omega'][-1]
    delta_actual = agente.historial['delta_struct'][-1]
    lf_actual = agente.historial['LF'][-1]
    lambda_actual = agente.historial['Lambda_Cos'][-1]

    if lambda_actual < 0.1:
        salud = "coma metabólico"
    elif lambda_actual < 0.5:
        salud = "salud frágil"
    elif lambda_actual < 1.0:
        salud = "salud estable"
    else:
        salud = "salud robusta"

    if lf_actual < 0.2:
        plasticidad = "rígido"
    elif lf_actual < 0.6:
        plasticidad = "plástico moderado"
    else:
        plasticidad = "altamente plástico"

    if delta_actual < 0.01:
        alimento = "sin nutrientes"
    elif delta_actual < 0.1:
        alimento = "nutrientes escasos"
    else:
        alimento = "nutrientes abundantes"

    evento_str = f" [{evento}]" if evento else ""
    print(f"\n  [{nombre}{evento_str}]")
    print(f"    🧬 Fenomenológico: {salud}, {plasticidad}, {alimento}")
    print(f"    📊 Operacional: Ω={omega_actual:.3f}, Λ_Cos={lambda_actual:.3f}, LF={lf_actual:.3f}, Δ_struct={delta_actual:.4f}")


# ============================================================
# EXPERIMENTO V110
# ============================================================
def ejecutar_v110():
    print("\n" + "█" * 100)
    print("EXPERIMENTO V110 — PROTOCOLO DE DIFERENCIA")
    print("█" * 100)

    A = AgenteVST_Canonico("A", seed=42)
    B = AgenteVST_Canonico("B", seed=43)

    # Verificar archivos
    print("\n  Verificando archivos...")
    left_path = 'audio_binaural/Blue_Monday_binaural_expandido_left_binaural.wav'
    right_path = 'audio_binaural/Blue_Monday_binaural_expandido_right_binaural.wav'

    if not os.path.exists(left_path):
        print(f"    ❌ {left_path} no encontrado")
        return None, None, False
    if not os.path.exists(right_path):
        print(f"    ❌ {right_path} no encontrado")
        return None, None, False

    print(f"    ✅ Blue_Monday_left_binaural.wav")
    print(f"    ✅ Blue_Monday_right_binaural.wav")

    A.set_frontera(left_path)
    B.set_frontera(right_path)

    T_total = EVENTOS['final']
    pasos = int(T_total / DT)
    t_inanicion = EVENTOS['breakdown']

    print(f"\n  Simulación: {T_total:.0f}s ({pasos} pasos)")
    print(f"  Inanición de B en t={t_inanicion:.0f}s (breakdown)")

    # FASE 1: Co-regulación
    print("\n" + "=" * 80)
    print("FASE 1: Co-regulación (A:Left, B:Right)")
    print("=" * 80)

    eventos_reportados = set()

    for i in range(int(t_inanicion / DT)):
        t = i * DT
        met_A = A.actualizar(t, DT, B)
        met_B = B.actualizar(t, DT, A)

        for evento, t_evento in EVENTOS.items():
            if evento not in eventos_reportados and t >= t_evento - 1.0:
                eventos_reportados.add(evento)
                reporte_doble_descripcion(A, "A", evento)
                reporte_doble_descripcion(B, "B", evento)

        if i % 2000 == 0 and i > 0:
            print(f"\n  t={t:.1f}s")
            print(f"    A: Ω={met_A['omega']:.3f}, Λ={met_A['Lambda_Cos']:.3f}, LF={met_A['LF']:.3f}")
            print(f"    B: Ω={met_B['omega']:.3f}, Λ={met_B['Lambda_Cos']:.3f}, LF={met_B['LF']:.3f}")
            print(f"    S_shared={met_A['S_shared']:.3f}")

    s_shared_fase1 = A.historial['S_shared'][-2000:] if len(A.historial['S_shared']) > 2000 else A.historial['S_shared']
    S_shared_medio = np.mean(s_shared_fase1)
    print(f"\n  📊 S_shared medio Fase 1: {S_shared_medio:.3f}")
    print(f"  📊 Zona semiótica (0.2-0.8): {'✅ ALCANZADA' if 0.2 <= S_shared_medio <= 0.8 else '❌ NO ALCANZADA'}")

    omega_A_before = A.historial['omega'][-1]

    # FASE 2: Inanición en B
    print("\n" + "=" * 80)
    print("FASE 2: Inanición en B (pérdida de Δ_struct)")
    print("=" * 80)

    B.inducir_inanicion()
    t_fin_inanicion = t_inanicion + 30.0

    for i in range(int(30.0 / DT)):
        t = t_inanicion + i * DT
        met_A = A.actualizar(t, DT, B)
        met_B = B.actualizar(t, DT, A)

        if i % 500 == 0:
            print(f"  t={t:.1f}s | Ω_A={met_A['omega']:.3f} | Ω_B={met_B['omega']:.3f} | Λ_B={met_B['Lambda_Cos']:.3f}")

    # FASE 3: Recuperación
    print("\n" + "=" * 80)
    print("FASE 3: Recuperación (B vuelve a tener alimento)")
    print("=" * 80)

    B.en_inanicion = False

    for i in range(int(t_fin_inanicion / DT), pasos):
        t = i * DT
        met_A = A.actualizar(t, DT, B)
        met_B = B.actualizar(t, DT, A)

        if i % 2000 == 0 and i > int(t_fin_inanicion / DT):
            print(f"  t={t:.1f}s | Ω_A={met_A['omega']:.3f} | Ω_B={met_B['omega']:.3f} | Λ_B={met_B['Lambda_Cos']:.3f}")

    # TEST R₂
    print("\n" + "=" * 80)
    print("ANÁLISIS R₂")
    print("=" * 80)

    idx_inicio = int(t_inanicion / DT)
    idx_fin = int(t_fin_inanicion / DT)

    omega_A_durante = A.historial['omega'][idx_inicio:idx_fin]
    omega_B_durante = B.historial['omega'][idx_inicio:idx_fin]

    omega_A_after = omega_A_durante[-1] if len(omega_A_durante) > 0 else omega_A_before
    delta_A = omega_A_after - omega_A_before

    if len(omega_B_durante) > 10:
        dOmega_B = np.mean(np.diff(omega_B_durante[-100:])) if len(omega_B_durante) > 100 else np.mean(np.diff(omega_B_durante))
    else:
        dOmega_B = 0.0

    print(f"\n  📊 Datos:")
    print(f"    Omega_A antes inanición: {omega_A_before:.4f}")
    print(f"    Omega_A durante/después: {omega_A_after:.4f}")
    print(f"    ΔA = {delta_A:+.4f}")
    print(f"    Tendencia B durante inanición: {dOmega_B:+.4f}")

    R2_detectado = abs(delta_A) > 0.05 and dOmega_B != 0 and np.sign(delta_A) == -np.sign(dOmega_B)

    print(f"\n  🧬 Diagnóstico R₂: {'✅ CONFIRMADO' if R2_detectado else '❌ NO CONFIRMADO'}")

    if R2_detectado:
        print("""
    → INTERPRETACIÓN BIOLÓGICA:
       El organismo A detectó la inanición de B y modificó su postura
       en dirección opuesta a la tendencia de B. Esto es DISENSO.
       A modela el estado de B y actúa en consecuencia.

    → CONCLUSIÓN CANÓNICA:
       R₂ = True. Meta-representación alcanzada.
       VSTCosmos ha cruzado a ALMA RACIONAL.
    """)
    elif S_shared_medio > 0.2:
        print("""
    → INTERPRETACIÓN BIOLÓGICA:
       A y B desarrollaron co-regulación (S_shared > 0.2), pero A no
       respondió selectivamente a la inanición de B. Hay vínculo, no hay
       teoría del otro.

    → CONCLUSIÓN CANÓNICA:
       Alma Sensitiva++. Hay S_shared, no hay R₂.
    """)
    else:
        print("""
    → INTERPRETACIÓN BIOLÓGICA:
       No hubo vínculo sostenido entre A y B. Cada uno comió por su lado.

    → CONCLUSIÓN CANÓNICA:
       Alma Sensitiva. Sin co-regulación.
    """)

    # GRÁFICOS
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Gráfico 1: Ω
    ax = axes[0, 0]
    t_A = A.historial['t'][:len(A.historial['omega'])]
    t_B = B.historial['t'][:len(B.historial['omega'])]
    ax.plot(t_A, gaussian_filter1d(A.historial['omega'], sigma=50), label='A (Left)', linewidth=1.5)
    ax.plot(t_B, gaussian_filter1d(B.historial['omega'], sigma=50), label='B (Right)', linewidth=1.5)
    for evento, t_e in EVENTOS.items():
        ax.axvline(x=t_e, color='gray', linestyle='--', alpha=0.5)
        ax.text(t_e, 1.05, evento, rotation=45, fontsize=8)
    ax.axvline(x=t_inanicion, color='red', linestyle='-', linewidth=2, alpha=0.7, label='Inanición B')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Ω')
    ax.set_title('Evolución de Ω')
    ax.legend()
    ax.set_ylim(-0.2, 1.2)
    ax.grid(True, alpha=0.3)

    # Gráfico 2: S_shared
    ax = axes[0, 1]
    s_shared = A.historial['S_shared'][:len(A.historial['t'])]
    ax.plot(t_A[:len(s_shared)], gaussian_filter1d(s_shared, sigma=50), color='purple', linewidth=1.5)
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Fusión (>0.8)')
    ax.axhline(y=0.2, color='orange', linestyle='--', alpha=0.5, label='Desacople (<0.2)')
    ax.axvline(x=t_inanicion, color='red', linestyle='-', linewidth=2, alpha=0.7)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('S_shared')
    ax.set_title('Sentido Compartido')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Gráfico 3: Λ_Cos
    ax = axes[1, 0]
    ax.plot(t_A[:len(A.historial['Lambda_Cos'])], gaussian_filter1d(A.historial['Lambda_Cos'], sigma=50), label='A')
    ax.plot(t_B[:len(B.historial['Lambda_Cos'])], gaussian_filter1d(B.historial['Lambda_Cos'], sigma=50), label='B')
    ax.axvline(x=t_inanicion, color='red', linestyle='-', linewidth=2, alpha=0.7)
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Umbral Racional (>1.0)')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Λ_Cos')
    ax.set_title('Salud dinámica')
    ax.legend()
    ax.set_ylim(0, 1.5)
    ax.grid(True, alpha=0.3)

    # Gráfico 4: Δ_struct
    ax = axes[1, 1]
    ax.plot(t_A[:len(A.historial['delta_struct'])], gaussian_filter1d(A.historial['delta_struct'], sigma=50), label='A')
    ax.plot(t_B[:len(B.historial['delta_struct'])], gaussian_filter1d(B.historial['delta_struct'], sigma=50), label='B')
    ax.axvline(x=t_inanicion, color='red', linestyle='-', linewidth=2, alpha=0.7)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Δ_struct')
    ax.set_title('Alimento disponible')
    ax.legend()
    ax.set_ylim(0, 0.5)
    ax.grid(True, alpha=0.3)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('v110_logs', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'v110_logs/v110_resultados_{timestamp}.png', dpi=150)
    print(f"\n  📊 Gráfico guardado: v110_logs/v110_resultados_{timestamp}.png")

    return A, B, R2_detectado


if __name__ == "__main__":
    A, B, R2 = ejecutar_v110()