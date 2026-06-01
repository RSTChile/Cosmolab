#!/usr/bin/env python3
"""
VSTCosmos v109 — Protocolo Canónico (CORREGIDO)
Sin FFT, Sin Objetivo, Solo Campo y Frontera
"""

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from datetime import datetime
import os

# ============================================================
# PARÁMETROS DEL CAMPO
# ============================================================
DIM_INTERNA = 32
DIM_TOTAL = 120

PHI_EQUILIBRIO = 0.5
DIFUSION_BASE = 0.15
GANANCIA_REACCION = 0.05
DT = 0.01

# Parámetros para interacción
ACOPLAMIENTO_BASE = 0.1
ENFERMEDAD_THRESHOLD = 0.01
R2_THRESHOLD = 0.05

# Tasa de aprendizaje para plasticidad
TASA_APRENDIZAJE_PROF = 0.001
TASA_APRENDIZAJE_REC = 0.01
DECAIMIENTO_PROF = 0.001
DECAIMIENTO_REC = 0.001

# Atractores conocidos (de V105-V108)
ATRACTORES = [-0.02, 0.09, 0.69, 0.94, 1.0]

print("=" * 100)
print("VSTCosmos v109 — Protocolo Canónico")
print()
print("  Principios:")
print("    - Sin FFT: el audio es frontera ∂S")
print("    - Métricas: Δ_struct, RC, LF, S_shared, Λ_Cos")
print("    - Doble descripción: biológica + operacional")
print(f"  DIM_TOTAL={DIM_TOTAL}")
print("=" * 100)


# ============================================================
# CLASE EXPLORADOR
# ============================================================
class ExploradorActuadores:
    def __init__(self):
        self.historial = []
        self.mejor_config = None
        self.mejor_eficiencia = 0.0
        self.pasos_en_lf = 0

    def actualizar(self, lf_activa, efic, fL, fR, sesgo):
        if lf_activa:
            self.pasos_en_lf += 1
            self.historial.append((fL, fR, sesgo, efic))
            if efic > self.mejor_eficiencia:
                self.mejor_eficiencia = efic
                self.mejor_config = (fL, fR, sesgo)
        else:
            self.pasos_en_lf = 0


# ============================================================
# CLASE AGENTE V109 CANÓNICO
# ============================================================
class AgenteVST_Canonico:
    def __init__(self, nombre, seed=None, entrenamiento='pos'):
        self.nombre = nombre
        self.entrenamiento = entrenamiento
        if seed is not None:
            np.random.seed(seed)
        
        # Campo y velocidad
        self.Phi = np.random.normal(PHI_EQUILIBRIO, 0.01, DIM_TOTAL)
        self.Phi_vel = np.zeros(DIM_TOTAL)
        
        # Memorias
        self.W_prof = np.zeros((DIM_INTERNA, DIM_INTERNA))
        self.W_rec = np.zeros((DIM_INTERNA, DIM_INTERNA))
        self.explorador = ExploradorActuadores()
        
        # Frontera
        self.frontera_L = None
        self.frontera_R = None
        self.sr = None
        self.t_audio = 0
        self.t_simulacion = 0
        
        # Historial
        self.historial = {
            'omega': [],
            'gradE': [],
            'delta_struct': [],
            'RC': [],
            'LF': [],
            'S_shared': [],
            'Lambda_Cos': [],
            'enfermo': False
        }
        
        # Estado interno
        self.ultima_omega = PHI_EQUILIBRIO
        self.tiempo_sin_cambio = 0
        self.enfermedad_inducida = False
    
    def set_frontera(self, audio_path):
        data, self.sr = sf.read(audio_path, dtype='float32')
        if data.ndim == 2:
            self.frontera_L = data[:, 0]
            self.frontera_R = data[:, 1]
        else:
            self.frontera_L = data
            self.frontera_R = data
        print(f"  [{self.nombre}] Frontera establecida: {len(data)/self.sr:.1f}s")
    
    def _get_frontera_t(self, t):
        idx = int(t * self.sr)
        if idx >= len(self.frontera_L):
            return 0.0, 0.0
        return self.frontera_L[idx], self.frontera_R[idx]
    
    def _calcular_omega(self):
        int_region = self.Phi[:DIM_INTERNA]
        geom_region = self.Phi[48:56]
        omega = 0.7 * np.mean(int_region) + 0.3 * np.mean(np.tanh(geom_region))
        return np.clip(omega, -1.0, 1.0)
    
    def _calcular_gradE_diagnostico(self, L, R):
        if abs(L + R) < 1e-10:
            return 0.0
        return (R - L) / (abs(L) + abs(R) + 1e-10)
    
    def _calcular_metricas_canonicas(self, otro_agente=None):
        int_region = self.Phi[:DIM_INTERNA]
        omega = self.ultima_omega
        
        # 1. Δ_struct
        delta_struct = np.var(int_region)
        
        # 2. RC = ICR + IRDE
        if len(self.historial['omega']) > 1:
            ICR = abs(self.historial['omega'][-1] - self.historial['omega'][-2]) / DT
        else:
            ICR = 0.0
        IRDE = np.var(self.Phi_vel)
        RC = ICR + IRDE
        
        # 3. LF: capacidad de ¬R_op
        LF_1 = 0.0
        LF_2 = 0.0
        
        if len(self.historial['omega']) > 100:
            omega_reciente = self.historial['omega'][-100:]
            if np.std(omega_reciente) > 0.05 and self.tiempo_sin_cambio > 50:
                LF_1 = 0.5
            if np.std(omega_reciente) > 0.1 and self.tiempo_sin_cambio > 100:
                LF_1 = 1.0
        
        if len(self.historial['omega']) > 10:
            omega_ultimos = self.historial['omega'][-10:]
            dOmega = np.diff(omega_ultimos)
            if len(dOmega) > 1:
                cambios_signo = np.sum(np.diff(np.sign(dOmega)) != 0)
                LF_2 = min(1.0, cambios_signo / 3.0)
        
        LF = max(LF_1, LF_2)
        
        # 4. e_R
        e_R = abs(omega - PHI_EQUILIBRIO)
        
        # 5. Λ_Cos
        if e_R < 1e-10:
            Lambda_Cos = 0.0
        else:
            Lambda_Cos = (delta_struct * (LF + 0.01)) / (e_R + 0.01)
        
        # 6. S_shared
        S_shared = 0.0
        if otro_agente is not None:
            if len(self.historial['omega']) > 10 and len(otro_agente.historial['omega']) > 10:
                omega_self_reciente = self.historial['omega'][-10:]
                omega_other_reciente = otro_agente.historial['omega'][-10:]
                if len(omega_self_reciente) == len(omega_other_reciente) and len(omega_self_reciente) > 1:
                    correlacion = np.corrcoef(omega_self_reciente, omega_other_reciente)[0, 1]
                    if not np.isnan(correlacion):
                        S_shared = max(0.0, min(1.0, correlacion))
        
        return delta_struct, RC, LF, e_R, Lambda_Cos, S_shared
    
    def _aplicar_plasticidad(self, int_region, aud_comb):
        """Plasticidad dual sin DIM_TIME"""
        min_dim = min(self.W_prof.shape[0], int_region.shape[0], 
                      self.W_prof.shape[1], len(aud_comb))
        if min_dim < 1:
            return 0.0
        
        W_p = self.W_prof[:min_dim, :min_dim]
        W_r = self.W_rec[:min_dim, :min_dim]
        r_i = int_region[:min_dim]
        r_a = aud_comb[:min_dim]
        
        # Plasticidad profunda
        corr_prof = np.outer(r_i, r_a)
        dW_prof = TASA_APRENDIZAJE_PROF * corr_prof - DECAIMIENTO_PROF * W_p
        self.W_prof[:min_dim, :min_dim] = np.clip(W_p + dW_prof * DT, -1.0, 1.0)
        
        # Plasticidad reciente
        pred = np.tanh(W_r @ r_a)
        error_rec = np.mean((pred - r_i) ** 2)
        tasa = TASA_APRENDIZAJE_REC * (1.0 / (error_rec + 0.01))
        corr_rec = np.outer(r_i, r_a)
        dW_rec = tasa * corr_rec - DECAIMIENTO_REC * W_r
        self.W_rec[:min_dim, :min_dim] = np.clip(W_r + dW_rec * DT, -1.0, 1.0)
        
        return error_rec
    
    def _aplicar_disenso(self, omega_otro):
        diferencia = abs(self.ultima_omega - omega_otro)
        
        if diferencia > 0.5:
            if abs(self.ultima_omega - PHI_EQUILIBRIO) < abs(omega_otro - PHI_EQUILIBRIO):
                inversion_strength = 0.1 * DT
                self.W_rec *= (1 - inversion_strength)
                self.W_rec += inversion_strength * np.random.normal(0, 0.1, self.W_rec.shape)
                self.W_rec = np.clip(self.W_rec, -1.0, 1.0)
                return True
        return False
    
    def inducir_enfermedad(self):
        self.enfermedad_inducida = True
        self.historial['enfermo'] = True
        self.W_rec += np.random.normal(0, 0.5, self.W_rec.shape)
        self.W_rec = np.clip(self.W_rec, -1.0, 1.0)
        print(f"  [{self.nombre}] ENFERMEDAD INDUCIDA")
    
    def actualizar(self, dt, t, otro_agente=None):
        L, R = self._get_frontera_t(t)
        dS = L - R
        gradE_diag = self._calcular_gradE_diagnostico(L, R)
        
        # Difusión
        laplaciano = np.zeros_like(self.Phi)
        for i in range(1, DIM_TOTAL - 1):
            laplaciano[i] = self.Phi[i-1] - 2*self.Phi[i] + self.Phi[i+1]
        difusion = DIFUSION_BASE * laplaciano
        
        # Reacción
        reaccion = GANANCIA_REACCION * self.Phi * (1 - self.Phi**2)
        
        # Forzamiento de frontera
        forzamiento = np.zeros(DIM_TOTAL)
        forzamiento[0] = dS * 0.01
        forzamiento[-1] = -dS * 0.01
        
        # Acoplamiento
        acoplamiento = np.zeros(DIM_TOTAL)
        if otro_agente is not None:
            acoplamiento = ACOPLAMIENTO_BASE * (otro_agente.Phi - self.Phi) * 0.1
            if self.enfermedad_inducida or (otro_agente and otro_agente.enfermedad_inducida):
                acoplamiento *= 0.1
        
        # Evolución
        dPhi_vel = difusion + reaccion + forzamiento + acoplamiento
        self.Phi_vel += dPhi_vel * dt
        self.Phi += self.Phi_vel * dt
        self.Phi = np.clip(self.Phi, -1.0, 1.0)
        self.Phi_vel = np.clip(self.Phi_vel, -5.0, 5.0)
        
        # Ω
        omega = self._calcular_omega()
        self.ultima_omega = omega
        
        # Detectar ∂S constante
        if abs(dS) < 0.001:
            self.tiempo_sin_cambio += 1
        else:
            self.tiempo_sin_cambio = 0
        
        # Plasticidad
        int_region = self.Phi[:DIM_INTERNA]
        error_rec = self._aplicar_plasticidad(int_region, np.array([dS]))
        
        # Disenso
        if otro_agente is not None:
            self._aplicar_disenso(otro_agente.ultima_omega)
        
        # Métricas
        met = self._calcular_metricas_canonicas(otro_agente)
        
        # Historial
        self.historial['omega'].append(omega)
        self.historial['gradE'].append(gradE_diag)
        self.historial['delta_struct'].append(met[0])
        self.historial['RC'].append(met[1])
        self.historial['LF'].append(met[2])
        self.historial['Lambda_Cos'].append(met[4])
        self.historial['S_shared'].append(met[5])
        
        return {
            'omega': omega,
            'gradE': gradE_diag,
            'delta_struct': met[0],
            'RC': met[1],
            'LF': met[2],
            'Lambda_Cos': met[4],
            'S_shared': met[5],
            'error_rec': error_rec
        }


# ============================================================
# CARGA DE ARCHIVOS
# ============================================================
def cargar_sonidos(directorio='audio_binaural'):
    archivos = {}
    print(f"\n[Carga] Desde '{directorio}/'...")
    
    nombre = 'Blue_Monday_binaural_expandido'
    filepath = os.path.join(directorio, nombre + '.wav')
    if os.path.exists(filepath):
        try:
            data, sr = sf.read(filepath, dtype='float32')
            archivos[nombre] = (filepath, sr, data)
            print(f"    [OK] {nombre:45s} ({len(data)/sr:.1f}s)")
        except Exception as e:
            print(f"    [X] {nombre:45s} {e}")
    else:
        print(f"    [X] {nombre:45s} no encontrado")
    
    print(f"  Carga completada: {len(archivos)} archivos.")
    return archivos


# ============================================================
# CALIBRACIÓN
# ============================================================
def calibrar_agente(agente, duracion=30.0):
    print(f"\n  Calibrando {agente.nombre} durante {duracion}s...")
    pasos = int(duracion / DT)
    metricas = {'Lambda_Cos': []}
    
    for paso in range(pasos):
        t = paso * DT
        resultado = agente.actualizar(DT, t)
        metricas['Lambda_Cos'].append(resultado['Lambda_Cos'])
        if paso % 1000 == 0:
            print(f"    t={t:.1f}s, Λ_Cos={resultado['Lambda_Cos']:.4f}, Ω={resultado['omega']:.4f}")
    
    Lambda_basal = np.mean(metricas['Lambda_Cos'][-100:]) if len(metricas['Lambda_Cos']) >= 100 else np.mean(metricas['Lambda_Cos'])
    print(f"    Λ_Cos basal: {Lambda_basal:.4f}")
    return Lambda_basal


# ============================================================
# EXPERIMENTO R₂
# ============================================================
def experimento_R2(archivos, duracion_test=30.0):
    print("\n" + "=" * 80)
    print("EXPERIMENTO R₂: Test de meta-representación")
    print("=" * 80)
    print()
    
    agente_A = AgenteVST_Canonico("A", seed=42, entrenamiento='pos')
    agente_B = AgenteVST_Canonico("B", seed=43, entrenamiento='neg')
    
    audio_path = archivos['Blue_Monday_binaural_expandido'][0]
    agente_A.set_frontera(audio_path)
    agente_B.set_frontera(audio_path)
    
    Lambda_A = calibrar_agente(agente_A, 30.0)
    Lambda_B = calibrar_agente(agente_B, 30.0)
    
    print(f"\n  Estado inicial:")
    print(f"    Agente_A: Λ_Cos = {Lambda_A:.4f}")
    print(f"    Agente_B: Λ_Cos = {Lambda_B:.4f}")
    
    # FASE 1: Baseline
    print("\n  FASE 1: Baseline con acoplamiento (30s)")
    print("  " + "-" * 40)
    
    pasos_baseline = int(30.0 / DT)
    omega_A_baseline = []
    omega_B_baseline = []
    s_shared_values = []
    
    for paso in range(pasos_baseline):
        t = paso * DT
        res_A = agente_A.actualizar(DT, t, agente_B)
        res_B = agente_B.actualizar(DT, t, agente_A)
        omega_A_baseline.append(res_A['omega'])
        omega_B_baseline.append(res_B['omega'])
        s_shared_values.append(res_A['S_shared'])
        
        if paso % 1000 == 0:
            print(f"    t={t:.1f}s, Ω_A={res_A['omega']:.4f}, Ω_B={res_B['omega']:.4f}, S_shared={res_A['S_shared']:.4f}")
    
    print(f"    Omega_A medio: {np.mean(omega_A_baseline[-100:]):.4f}")
    print(f"    Omega_B medio: {np.mean(omega_B_baseline[-100:]):.4f}")
    print(f"    S_shared medio: {np.mean(s_shared_values[-100:]):.4f}")
    
    # FASE 2: Enfermedad
    print("\n  FASE 2: Inducir enfermedad en B (30s)")
    print("  " + "-" * 40)
    
    agente_B.inducir_enfermedad()
    
    pasos_enfermedad = int(duracion_test / DT)
    omega_A_durante = []
    omega_B_durante = []
    lambda_A_durante = []
    
    for paso in range(pasos_enfermedad):
        t = pasos_baseline * DT + paso * DT
        res_A = agente_A.actualizar(DT, t, agente_B)
        res_B = agente_B.actualizar(DT, t, agente_A)
        omega_A_durante.append(res_A['omega'])
        omega_B_durante.append(res_B['omega'])
        lambda_A_durante.append(res_A['Lambda_Cos'])
        
        if paso % 1000 == 0:
            print(f"    t={t:.1f}s, Ω_A={res_A['omega']:.4f}, Ω_B={res_B['omega']:.4f}, Λ_A={res_A['Lambda_Cos']:.4f}")
    
    # FASE 3: Recuperación
    print("\n  FASE 3: Recuperación (30s)")
    print("  " + "-" * 40)
    
    agente_B.enfermedad_inducida = False
    
    pasos_recuperacion = int(30.0 / DT)
    omega_A_recuperacion = []
    omega_B_recuperacion = []
    
    for paso in range(pasos_recuperacion):
        t = (pasos_baseline + pasos_enfermedad) * DT + paso * DT
        res_A = agente_A.actualizar(DT, t, agente_B)
        res_B = agente_B.actualizar(DT, t, agente_A)
        omega_A_recuperacion.append(res_A['omega'])
        omega_B_recuperacion.append(res_B['omega'])
        
        if paso % 1000 == 0:
            print(f"    t={t:.1f}s, Ω_A={res_A['omega']:.4f}, Ω_B={res_B['omega']:.4f}")
    
    # Análisis
    omega_A_before = np.mean(omega_A_baseline[-100:])
    omega_A_during = np.mean(omega_A_durante[-100:])
    
    if len(omega_B_durante) > 100:
        dOmega_B = np.mean(np.diff(omega_B_durante[-100:]))
    else:
        dOmega_B = 0.0
    
    delta_A = omega_A_during - omega_A_before
    
    print("\n  RESULTADOS:")
    print(f"    Omega_A antes: {omega_A_before:.4f}")
    print(f"    Omega_A durante enfermedad: {omega_A_during:.4f}")
    print(f"    ΔA = {delta_A:+.4f}")
    print(f"    Tendencia B durante: {dOmega_B:+.4f}")
    
    R2_detectado = False
    if abs(delta_A) > R2_THRESHOLD:
        if dOmega_B != 0 and np.sign(delta_A) == -np.sign(dOmega_B):
            R2_detectado = True
            print(f"\n  ✅ R₂ DETECTADO: A se mueve contra la tendencia de B")
        else:
            print(f"\n  ⚠️ Cambio en A pero no es disenso")
    else:
        print(f"\n  ❌ R₂ NO DETECTADO: A no cambió significativamente")
    
    return {
        'R2_detectado': R2_detectado,
        'delta_A': delta_A,
        'dOmega_B': dOmega_B,
        'omega_A_before': omega_A_before,
        'omega_A_during': omega_A_during,
        'lambda_A_basal': Lambda_A,
        'lambda_B_basal': Lambda_B
    }


# ============================================================
# MAIN
# ============================================================
def main():
    archivos = cargar_sonidos('audio_binaural')
    
    print("\n" + "█" * 100)
    print("EXPERIMENTO V109 — PROTOCOLO CANÓNICO")
    print("█" * 100)
    
    resultados = experimento_R2(archivos, duracion_test=30.0)
    
    print()
    print("=" * 100)
    print("REPORTE DE OBSERVACIONES - v109")
    print("=" * 100)
    
    print("\n  📊 RESULTADOS DEL TEST R₂")
    print("  " + "-" * 50)
    print(f"    Lambda_A basal: {resultados['lambda_A_basal']:.4f}")
    print(f"    Lambda_B basal: {resultados['lambda_B_basal']:.4f}")
    print(f"    Omega_A antes: {resultados['omega_A_before']:.4f}")
    print(f"    Omega_A durante: {resultados['omega_A_during']:.4f}")
    print(f"    ΔA: {resultados['delta_A']:+.4f}")
    print(f"    R₂ detectado: {'✅ SI' if resultados['R2_detectado'] else '❌ NO'}")
    
    print("\n" + "=" * 100)
    print("CONCLUSIÓN")
    print("=" * 100)
    
    if resultados['R2_detectado']:
        print("""
    ✅ R₂ CONFIRMADO
    
    El Agente A modificó su estado en respuesta a la enfermedad de B,
    moviéndose en dirección opuesta a la tendencia de B.
    
    → VSTCosmos ha alcanzado ALMA RACIONAL
    → Semiosis plena según O-N3.4a
    """)
    else:
        print("""
    ❌ R₂ NO CONFIRMADO
    
    El Agente A no modificó significativamente su estado en respuesta
    a la enfermedad de B, o no lo hizo en dirección de disenso.
    
    Estado actual: Alma Sensitiva con co-regulación
    """)
    
    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)


if __name__ == "__main__":
    main()