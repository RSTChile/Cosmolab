#!/usr/bin/env python3
"""
VSTCosmos v122-Anima-1 — Segregación Física + Órganos Mínimos
AJUSTADO: Menor ganancia para orientación estable hacia -60°
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# ============================================================
# PARÁMETROS
# ============================================================
DT = 0.01
DIM_INTERNA = 32
DIM_AUD = 16
DIM_ACT = 8
DIM_HEMISFERIO = DIM_INTERNA + DIM_AUD + 4 * DIM_ACT

TAU_IZQUIERDO = 30.0
TAU_DERECHO = 300.0

UMBRAL_CALLOSO = 0.5
GANANCIA_CALLOSO = 0.01
INHIBICION_RAPIDA = 0.1

UMBRAL_R2_SIGMAS = 3.0
UMBRAL_LATERALIDAD = 0.8

TIEMPO_POR_REPETICION = 452.0
REPETICIONES_LENTAS = 10
TIEMPO_BASELINE = 60.0
TIEMPO_INANICION = 30.0

print("=" * 100)
print("VSTCosmos v122-Anima-1 — Segregación Física + Órganos Mínimos")
print(f"  TAU_Izq={TAU_IZQUIERDO}s (R₂, ruido_rosa)")
print(f"  TAU_Der={TAU_DERECHO}s (lateralidad, clicks)")
print(f"  Calloso: umbral={UMBRAL_CALLOSO}, ganancia={GANANCIA_CALLOSO}")
print(f"  Inhibición: umbral={INHIBICION_RAPIDA}")
print("  ANIMA-1: Aparato motor + Atención (O-N3.3, O-N4.1)")
print("  AJUSTE: ganancia=0.01, factor_grad=20.0 (C50: -60°)")
print("=" * 100)


# ============================================================
# GENERADORES DE ENTRADAS ORTOGONALES
# ============================================================
def generar_ruido_rosa(duracion, sr=48000):
    n = int(duracion * sr)
    ruido = np.random.normal(0, 1, n)
    fft = np.fft.rfft(ruido)
    freqs = np.fft.rfftfreq(n, 1/sr)
    filtro = 1.0 / np.sqrt(freqs + 0.01)
    fft_filtrado = fft * filtro
    ruido_rosa = np.fft.irfft(fft_filtrado, n=n)
    ruido_rosa = ruido_rosa / (np.max(np.abs(ruido_rosa)) + 1e-10)
    return ruido_rosa

def generar_clicks_poisson(duracion, tasa=0.5, sr=48000):
    n = int(duracion * sr)
    clicks = np.zeros(n)
    n_clicks = int(duracion * tasa)
    for _ in range(n_clicks):
        pos = int(np.random.exponential(1.0/tasa) * sr)
        if pos < n:
            clicks[pos] = 1.0
    return clicks


# ============================================================
# HEMISFERIO
# ============================================================
class Hemisferio:
    def __init__(self, nombre, tau, generar_entrada_func, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.nombre = nombre
        self.tau = tau
        self.generar_entrada = generar_entrada_func
        
        self.Phi = np.random.normal(0, 0.01, DIM_HEMISFERIO)
        self.Phi_vel = np.zeros(DIM_HEMISFERIO)
        self.W = np.zeros((DIM_INTERNA, DIM_INTERNA))
        
        self.entrada = None
        self.sr = 48000
        self.en_inanicion = False
        self.factor_inanicion = 1.0
        
        self.buffer_rapido = []
        self.historial_omega = []
        self.historial_Lambda = []
    
    def omega(self):
        return self._calcular_omega()
    
    def lambda_cos(self):
        return self._calcular_Lambda()
    
    def generar_entrada_para_t(self, t, duracion_total):
        if self.entrada is None:
            self.entrada = self.generar_entrada(duracion_total, self.sr)
        idx = int(t * self.sr)
        if idx >= len(self.entrada):
            return 0.0
        return self.entrada[idx] * self.factor_inanicion
    
    def inducir_inanicion_gradual(self, paso_actual, pasos_totales):
        if paso_actual < pasos_totales:
            self.factor_inanicion = 1.0 - (paso_actual / pasos_totales)
        else:
            self.factor_inanicion = 0.0
            self.en_inanicion = True
    
    def _calcular_omega(self):
        return np.mean(self.Phi[:DIM_INTERNA])
    
    def _calcular_Lambda(self):
        int_region = self.Phi[:DIM_INTERNA]
        delta_struct = np.var(int_region)
        if len(self.historial_omega) > 50:
            atractores = np.round(self.historial_omega[-50:], 1)
            LF = len(set(atractores)) / 10.0
        else:
            LF = 0.0
        if len(self.buffer_rapido) > int(self.tau / DT):
            idx_pasado = max(0, len(self.buffer_rapido) - int(self.tau / DT))
            omega_pasado = self.buffer_rapido[idx_pasado][1]
            e_R = abs(self._calcular_omega() - omega_pasado)
        else:
            e_R = 0.01
        Lambda = (delta_struct * (LF + 0.01)) / (e_R + 0.01)
        return Lambda
    
    def actualizar(self, t, dt, duracion_total, otro_hemisferio=None):
        entrada = self.generar_entrada_para_t(t, duracion_total)
        
        laplaciano = np.zeros_like(self.Phi)
        for i in range(1, DIM_HEMISFERIO - 1):
            laplaciano[i] = self.Phi[i-1] - 2*self.Phi[i] + self.Phi[i+1]
        
        reaccion = self.Phi * (1 - self.Phi * self.Phi)
        
        forzamiento = np.zeros_like(self.Phi)
        forzamiento[0] = entrada
        forzamiento[-1] = -entrada
        
        acoplamiento = np.zeros_like(self.Phi)
        inhibido = False
        if otro_hemisferio is not None:
            omega_self = self._calcular_omega()
            omega_otro = otro_hemisferio._calcular_omega()
            divergencia = abs(omega_self - omega_otro)
            
            if self.nombre == 'L' and len(self.historial_omega) > 1:
                dOmega = (omega_self - self.historial_omega[-1]) / dt
                if abs(dOmega) > INHIBICION_RAPIDA:
                    inhibido = True
                    acoplamiento = np.zeros_like(self.Phi)
            
            if not inhibido and divergencia > UMBRAL_CALLOSO:
                acoplamiento = GANANCIA_CALLOSO * (otro_hemisferio.Phi - self.Phi)
        
        dPhi_vel = laplaciano + reaccion + forzamiento + acoplamiento
        self.Phi_vel += dPhi_vel * dt
        self.Phi += self.Phi_vel * dt
        self.Phi = np.clip(self.Phi, -1.0, 1.0)
        
        omega = self._calcular_omega()
        Lambda = self._calcular_Lambda()
        
        self.buffer_rapido.append((t, omega))
        if len(self.buffer_rapido) > int(self.tau / DT):
            self.buffer_rapido.pop(0)
        
        self.historial_omega.append(omega)
        self.historial_Lambda.append(Lambda)
        
        return {'omega': omega, 'Lambda': Lambda, 'inhibido': inhibido, 'entrada': entrada}


# ============================================================
# FUNCIÓN S_SHARED PROTEGIDA
# ============================================================
def calcular_S_shared(H_L, H_R):
    if np.std(H_L.Phi) < 1e-9 or np.std(H_R.Phi) < 1e-9:
        return 1.0
    return np.corrcoef(H_L.Phi, H_R.Phi)[0, 1]


# ============================================================
# ANIMA-1: ÓRGANOS NUEVOS (AJUSTADOS)
# ============================================================
class AparatoMotor:
    """Órgano de acción O-N3.3. Solo lee, no escribe en hemisferios.
    AJUSTADO: ganancia reducida para evitar saturación en ±90°"""
    def __init__(self):
        self.orientacion = 0.0
        self.ganancia = 0.01          # REDUCIDO: 0.05 → 0.01
        self.factor_grad = 20.0       # REDUCIDO: 50.0 → 20.0
        self.historial = []

    def actuar(self, H_L, H_R, LF_activa):
        if not LF_activa:
            return self.orientacion
        grad_binaural = H_L.omega() - H_R.omega()
        self.orientacion += self.ganancia * grad_binaural * self.factor_grad
        self.orientacion = np.clip(self.orientacion, -90.0, 90.0)
        self.historial.append(self.orientacion)
        return self.orientacion

class Atencion:
    """Pondera por Lambda_Cos. O-N4.1"""
    def ponderar(self, H_L, H_R):
        L_L = H_L.lambda_cos()
        L_R = H_R.lambda_cos()
        total = L_L + L_R + 1e-9
        peso_L = L_L / total
        Omega_at = peso_L * H_L.omega() + (1-peso_L) * H_R.omega()
        return Omega_at, peso_L


# ============================================================
# SISTEMA COMPLETO
# ============================================================
class SistemaV122:
    def __init__(self, nombre, seed=None):
        self.nombre = nombre
        self.izquierdo = Hemisferio("L", TAU_IZQUIERDO, generar_ruido_rosa, seed=seed)
        self.derecho = Hemisferio("R", TAU_DERECHO, generar_clicks_poisson, seed=seed+100 if seed else None)
        
        self.motor = AparatoMotor()
        self.atencion = Atencion()
        
        self.historial = {
            't': [],
            'omega_L': [],
            'omega_R': [],
            'omega_total': [],
            'Lambda_L': [],
            'Lambda_R': [],
            's_shared': [],
            'calloso_activo': [],
            'inhibicion': [],
            'orientacion': [],
            'LF_activa': []
        }
    
    def inducir_inanicion_gradual(self, paso_actual, pasos_totales):
        self.izquierdo.inducir_inanicion_gradual(paso_actual, pasos_totales)
        self.derecho.inducir_inanicion_gradual(paso_actual, pasos_totales)
    
    def actualizar(self, t, dt, duracion_total, otro_sistema=None):
        res_L = self.izquierdo.actualizar(t, dt, duracion_total, self.derecho)
        res_R = self.derecho.actualizar(t, dt, duracion_total, self.izquierdo)
        
        total = res_L['Lambda'] + res_R['Lambda'] + 1e-10
        omega_total = (res_L['Lambda'] * res_L['omega'] + res_R['Lambda'] * res_R['omega']) / total
        
        s_shared = 0.0
        if otro_sistema is not None:
            s_shared = calcular_S_shared(self.izquierdo, otro_sistema.izquierdo)
        
        calloso_activo = abs(res_L['omega'] - res_R['omega']) > UMBRAL_CALLOSO
        
        delta_A = (np.mean(np.abs(np.gradient(self.izquierdo.Phi))) + 
                   np.mean(np.abs(np.gradient(self.derecho.Phi)))) / 2 - 0.15
        LF_activa = delta_A < -0.1
        
        orientacion = self.motor.actuar(self.izquierdo, self.derecho, LF_activa)
        Omega_at, peso_L = self.atencion.ponderar(self.izquierdo, self.derecho)
        
        self.historial['t'].append(t)
        self.historial['omega_L'].append(res_L['omega'])
        self.historial['omega_R'].append(res_R['omega'])
        self.historial['omega_total'].append(omega_total)
        self.historial['Lambda_L'].append(res_L['Lambda'])
        self.historial['Lambda_R'].append(res_R['Lambda'])
        self.historial['s_shared'].append(s_shared)
        self.historial['calloso_activo'].append(calloso_activo)
        self.historial['inhibicion'].append(res_L['inhibido'] or res_R['inhibido'])
        self.historial['orientacion'].append(orientacion)
        self.historial['LF_activa'].append(LF_activa)
        
        return {
            'omega_total': omega_total,
            's_shared': s_shared,
            'orientacion': orientacion,
            'LF_activa': LF_activa
        }
    
    def omega_actual(self):
        if self.historial['omega_total']:
            return self.historial['omega_total'][-1]
        return 0.5


# ============================================================
# EXPERIMENTO
# ============================================================
def ejecutar_v122():
    print("\n" + "█" * 100)
    print("EXPERIMENTO V122-ANIMA-1 (AJUSTADO)")
    print("█" * 100)
    
    duracion_total = TIEMPO_POR_REPETICION * REPETICIONES_LENTAS + TIEMPO_BASELINE + TIEMPO_INANICION
    
    print("\n" + "=" * 80)
    print("FASE 1: Baseline — Un sistema")
    print("=" * 80)
    
    S1 = SistemaV122("S1", seed=42)
    
    for i in range(int(TIEMPO_POR_REPETICION / DT)):
        t = i * DT
        S1.actualizar(t, DT, duracion_total)
        if i % 10000 == 0:
            print(f"    t={t:.0f}s | Ω_L={S1.historial['omega_L'][-1]:.4f}, Ω_R={S1.historial['omega_R'][-1]:.4f}")
    
    print("\n" + "=" * 80)
    print("FASE 2: Dos sistemas acoplados (entrenamiento lateral)")
    print("=" * 80)
    
    S2 = SistemaV122("S2", seed=42)
    S3 = SistemaV122("S3", seed=43)
    
    for rep in range(REPETICIONES_LENTAS):
        print(f"  Repetición {rep+1}/{REPETICIONES_LENTAS}...")
        for i in range(int(TIEMPO_POR_REPETICION / DT)):
            t = rep * TIEMPO_POR_REPETICION + i * DT
            res2 = S2.actualizar(t, DT, duracion_total, S3)
            S3.actualizar(t, DT, duracion_total, S2)
            if i % 10000 == 0:
                print(f"      t={t:.0f}s | s_shared={res2['s_shared']:.3f}, orient={res2['orientacion']:.1f}°")
    
    s_shared_final = np.mean(S2.historial['s_shared'][-6000:]) if len(S2.historial['s_shared']) > 6000 else 1.0
    lateralidad = s_shared_final < UMBRAL_LATERALIDAD
    print(f"\n  Lateralidad: {'✅' if lateralidad else '❌'} (s_shared={s_shared_final:.4f})")
    
    print("\n" + "=" * 80)
    print("FASE 3: Test R₂ (inanición)")
    print("=" * 80)
    
    S4 = SistemaV122("S4", seed=42)
    S5 = SistemaV122("S5", seed=43)
    
    for i in range(6000):
        t = i * DT
        S4.actualizar(t, DT, duracion_total, S5)
        S5.actualizar(t, DT, duracion_total, S4)
    
    omega_before = np.mean(S4.historial['omega_total'][-1000:])
    std_before = np.std(S4.historial['omega_total'][-1000:])
    print(f"  Baseline: Ω_S4 = {omega_before:.4f} ± {std_before:.4f}")
    
    respuestas = []
    for i in range(3000):
        t = 60.0 + i * DT
        if i < 3000:
            S5.inducir_inanicion_gradual(i, 3000)
        res4 = S4.actualizar(t, DT, duracion_total, S5)
        S5.actualizar(t, DT, duracion_total, S4)
        omega_actual = res4['omega_total']
        respuestas.append(abs(omega_actual - omega_before))
        if i % 500 == 0:
            print(f"      t={t:.1f}s | Ω_S4={omega_actual:.4f}, orient={res4['orientacion']:.1f}°")
    
    respuesta_max = max(respuestas)
    umbral = 3 * std_before
    R2 = respuesta_max > umbral
    print(f"  R₂: {'✅' if R2 else '❌'} (resp_max={respuesta_max:.4f} > umbral={umbral:.4f})")
    
    print("\n" + "=" * 80)
    print("CONCLUSIÓN")
    print("=" * 80)
    print(f"  Lateralidad: {'✅' if lateralidad else '❌'} (s_shared={s_shared_final:.4f})")
    print(f"  R₂: {'✅' if R2 else '❌'}")
    
    orientacion_final = S2.historial['orientacion'][-1] if S2.historial['orientacion'] else 0
    C50 = abs(orientacion_final + 60) < 15
    print(f"  C50 (Orientación → -60°): {'✅' if C50 else '❌'} (final={orientacion_final:.1f}°)")
    
    if lateralidad and R2:
        print("\n  ✅ ÉXITO: R₂ Y LATERALIDAD COEXISTEN")
        if C50:
            print("  ✅ ANIMA-1: Orientación lograda (-60°)")
        else:
            print(f"  ⚠️ ANIMA-1: Orientación = {orientacion_final:.1f}° (objetivo -60°)")
    elif lateralidad:
        print("\n  ⚠️ LATERALIDAD SÍ, R₂ NO")
    elif R2:
        print("\n  ⚠️ R₂ SÍ, LATERALIDAD NO")
    else:
        print("\n  ❌ NINGUNA CAPACIDAD")
    
    # Gráficos
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    t1 = S1.historial['t']
    ax.plot(t1, S1.historial['omega_L'], label='Izquierdo (ruido_rosa)', alpha=0.7)
    ax.plot(t1, S1.historial['omega_R'], label='Derecho (clicks)', alpha=0.7)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Ω')
    ax.set_title('Hemisferios con entradas ortogonales')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    t2 = S2.historial['t']
    ax.plot(t2, S2.historial['s_shared'], color='purple', linewidth=0.5)
    ax.axhline(y=UMBRAL_LATERALIDAD, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('s_shared')
    ax.set_title('Sentido compartido (lateralidad)')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.plot(t2, np.array(S2.historial['calloso_activo']).astype(float), color='orange', linewidth=0.5)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Calloso activo')
    ax.set_title('Comunicación inter-hemisférica')
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.plot(t2, S2.historial['orientacion'], color='green', linewidth=0.5)
    ax.axhline(y=-60, color='blue', linestyle='--', alpha=0.5, label='Objetivo -60°')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Orientación (°)')
    ax.set_title('Aparato motor (C50)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    t_resp = 60 + np.arange(len(respuestas)) * DT
    ax2.plot(t_resp, respuestas, color='red')
    ax2.axhline(y=umbral, color='green', linestyle='--', label=f'Umbral (3σ={umbral:.3f})')
    ax2.set_xlabel('Tiempo (s)')
    ax2.set_ylabel('|ΔΩ|')
    ax2.set_title('Respuesta a inanición (R₂)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('v122_anima1_logs', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'v122_anima1_logs/v122_anima1_resultados_{timestamp}.png', dpi=150)
    fig2.savefig(f'v122_anima1_logs/v122_anima1_respuesta_{timestamp}.png', dpi=150)
    print(f"\n  📊 Gráficos: v122_anima1_logs/v122_anima1_resultados_{timestamp}.png")
    print(f"  📊 Respuesta: v122_anima1_logs/v122_anima1_respuesta_{timestamp}.png")
    
    return S2, S3, R2 and lateralidad


if __name__ == "__main__":
    S2, S3, exito = ejecutar_v122()