#!/usr/bin/env python3
"""
VSTCosmos v108 — Blue Monday: Estímulo largo y complejo

Preguntas:
  1. ¿Cómo evoluciona Ω a lo largo de 7.5 minutos?
  2. ¿Hay habituación o sensibilización?
  3. ¿El sistema detecta cambios estructurales en la música?
  4. ¿Dos agentes acoplados desarrollan S_shared con música compleja?

Estímulos:
  - Blue_Monday_binaural_expandido.wav (campo estéreo expandido, 7.5 min)
  - Blue_Monday_binaural_expandido_left_binaural.wav (fuente izquierda)
  - Blue_Monday_binaural_expandido_right_binaural.wav (fuente derecha)
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
from datetime import datetime
from scipy import signal
from scipy.ndimage import gaussian_filter1d

try:
    import soundfile as sf
    HAS_SF = True
except ImportError:
    HAS_SF = False
    print("ERROR: soundfile no instalado")
    exit(1)

warnings.filterwarnings('ignore')

# ============================================================
# PARAMETROS (identicos a v107)
# ============================================================
DIM_INTERNA = 32

DIFUSION_BASE     = 0.15
GANANCIA_REACCION = 0.05
OMEGA_MIN         = 0.05
OMEGA_MAX         = 0.50
AMORT_MIN         = 0.01
AMORT_MAX         = 0.08
PHI_EQUILIBRIO    = 0.5

VENTANA_FFT_MS = 25
HOP_FFT_MS     = 10
F_MIN          = 80
F_MAX          = 8000

T_PROFUNDA_SEG   = 1.0 / OMEGA_MIN
T_RECIENTE_SEG   = 1.0 / OMEGA_MAX
T_PROFUNDA_PASOS = int(T_PROFUNDA_SEG / 0.01)
T_RECIENTE_PASOS = int(T_RECIENTE_SEG / 0.01)

ETA_PROFUNDA_BASE = (1.0 / T_PROFUNDA_PASOS) / DIFUSION_BASE
ETA_RECIENTE_BASE = (1.0 / T_RECIENTE_PASOS) / DIFUSION_BASE
TAU_PROFUNDA      = OMEGA_MIN
TAU_RECIENTE      = OMEGA_MIN * 0.5
TAU_EFICIENCIA    = int(1.0 / (OMEGA_MIN * 0.01))
TAU_EXPLORACION   = int(T_RECIENTE_SEG / 0.01)

LIMITE_MIN  = -1.0
LIMITE_MAX  = 1.0
W_MAX       = 1.0
ALPHA_FIJO  = 0.05
DT          = 0.01
DIM_TIME    = 100

# Parámetros para meta-representación
DIM_META = 8
GANANCIA_META = 0.1
DISSENSO_THRESHOLD = 0.5
DISSENSO_STRENGTH = 0.5

DIAMETRO_CABEZA  = 0.175
VELOCIDAD_SONIDO = 343.0
F_TRANS_HZ       = VELOCIDAD_SONIDO / DIAMETRO_CABEZA

# ============================================================
# ARQUITECTURA (expandida con región meta)
# ============================================================
DIM_GANGLIO  = DIM_INTERNA // 2
DIM_AUD      = DIM_GANGLIO
DIM_ACT      = DIM_GANGLIO // 2

DIM_AUD_L    = DIM_AUD
DIM_AUD_R    = DIM_AUD
DIM_ACT_PERM = DIM_ACT
DIM_ACT_GEOM = DIM_ACT
DIM_ACT_BUSC = DIM_ACT
DIM_ACT_MANT = DIM_ACT
DIM_META_REG = DIM_META

BANDA_TRANS = int(DIM_AUD * np.log10(F_TRANS_HZ / F_MIN)
                  / np.log10(F_MAX / F_MIN))
BANDA_TRANS = max(1, min(BANDA_TRANS, DIM_AUD - 1))

K_BUSC               = T_PROFUNDA_SEG / T_RECIENTE_SEG
K_ORIENT             = T_PROFUNDA_SEG / T_RECIENTE_SEG
DECAIMIENTO_ACT_BUSC = DT / T_RECIENTE_SEG
EPSILON_BUSC_G       = DIFUSION_BASE * K_BUSC * DT

idx = {}
idx['int']      = (0,                   DIM_INTERNA)
idx['G']        = (DIM_INTERNA,         DIM_INTERNA + DIM_GANGLIO)
idx['aud_L']    = (idx['G'][1],         idx['G'][1]        + DIM_AUD_L)
idx['aud_R']    = (idx['aud_L'][1],     idx['aud_L'][1]    + DIM_AUD_R)
idx['act_perm'] = (idx['aud_R'][1],     idx['aud_R'][1]    + DIM_ACT_PERM)
idx['act_geom'] = (idx['act_perm'][1],  idx['act_perm'][1] + DIM_ACT_GEOM)
idx['act_busc'] = (idx['act_geom'][1],  idx['act_geom'][1] + DIM_ACT_BUSC)
idx['act_mant'] = (idx['act_busc'][1],  idx['act_busc'][1] + DIM_ACT_MANT)
idx['meta']     = (idx['act_mant'][1],  idx['act_mant'][1] + DIM_META_REG)
DIM_TOTAL = idx['meta'][1]

VECINDADES = [
    ('int',      'G'),
    ('G',        'aud_L'),
    ('G',        'aud_R'),
    ('G',        'act_perm'),
    ('G',        'act_geom'),
    ('G',        'act_mant'),
    ('G',        'meta'),
    ('aud_L',    'aud_R'),
    ('act_perm', 'aud_L'),
    ('act_perm', 'aud_R'),
    ('act_geom', 'aud_L'),
    ('act_geom', 'aud_R'),
]

VARIACION_FLOOR = 1e-6
VENTANA_FINAL_SEG = 30
VENTANA_FINAL_PASOS = int(VENTANA_FINAL_SEG / DT)
DURACION_ENTRENAMIENTO = 60.0
DURACION_EVALUACION_BASE = 30.0

# Blue Monday dura 452.5 segundos (~7.5 min)
DURACION_BLUE_MONDAY = 452.5

# Estímulos de Blue Monday
ESTIMULOS_BLUE = [
    'Blue_Monday_binaural_expandido',
    'Blue_Monday_binaural_expandido_left_binaural',
    'Blue_Monday_binaural_expandido_right_binaural'
]

print("=" * 100)
print("VSTCosmos v108 — Blue Monday: Estímulo largo y complejo")
print()
print("  Preguntas:")
print("    1. ¿Cómo evoluciona Ω a lo largo de 7.5 minutos?")
print("    2. ¿Hay habituación o sensibilización?")
print("    3. ¿El sistema detecta cambios estructurales?")
print("    4. ¿Dos agentes desarrollan S_shared con música compleja?")
print()
print(f"  Duración del estímulo: {DURACION_BLUE_MONDAY:.1f}s (7.5 min)")
print(f"  DIM_TOTAL={DIM_TOTAL}")
print(f"  BANDA_TRANS={BANDA_TRANS} (F>{F_TRANS_HZ:.0f}Hz)")
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
# CLASE AGENTE VST (expandida)
# ============================================================
class AgenteVST:
    def __init__(self, nombre, seed=None):
        self.nombre = nombre
        if seed is not None:
            np.random.seed(seed)
        self.Phi_total, self.Phi_vel_total = self._inicializar_campo()
        self.W_prof, self.W_rec, self.W_meta, self.Phi_int_historia = self._inicializar_memorias()
        self.explorador = ExploradorActuadores()
        self.historial_omega = []
        self.historial_omega_tiempo = []  # Serie temporal completa
        self.historial_gradE = []
        self.historial_s_shared = []
        self.historial_disenso = []
        self.ultima_omega = 0.0
    
    def _inicializar_campo(self):
        Phi_total = np.random.normal(PHI_EQUILIBRIO, 0.01, (DIM_TOTAL, DIM_TIME))
        Phi_vel_total = np.zeros((DIM_TOTAL, DIM_TIME))
        return Phi_total, Phi_vel_total
    
    def _inicializar_memorias(self):
        W_prof = np.zeros((DIM_INTERNA, DIM_AUD))
        W_rec = np.zeros((DIM_INTERNA, DIM_AUD))
        W_meta = np.zeros((DIM_META_REG, DIM_TOTAL))
        Phi_int_historia = np.zeros((DIM_INTERNA, DIM_TIME))
        return W_prof, W_rec, W_meta, Phi_int_historia
    
    def _calcular_frecuencias_naturales(self, dim):
        bandas = np.arange(dim)
        t = np.log1p(bandas) / np.log1p(max(dim - 1, 1))
        omega = OMEGA_MIN + (OMEGA_MAX - OMEGA_MIN) * t
        amort = AMORT_MIN + (AMORT_MAX - AMORT_MIN) * t
        return omega.reshape(-1, 1), amort.reshape(-1, 1)
    
    def _calcular_promedio_vecinos(self):
        promedio = np.zeros_like(self.Phi_total)
        conteo = np.zeros(DIM_TOTAL)
        for reg_a, reg_b in VECINDADES:
            ia0, ia1 = idx[reg_a]
            ib0, ib1 = idx[reg_b]
            n = min(ia1 - ia0, ib1 - ib0)
            for d in range(n):
                if ia0 + d < DIM_TOTAL and ib0 + d < DIM_TOTAL:
                    promedio[ia0 + d, :] += self.Phi_total[ib0 + d, :]
                    promedio[ib0 + d, :] += self.Phi_total[ia0 + d, :]
                    conteo[ia0 + d] += 1
                    conteo[ib0 + d] += 1
        for i in range(DIM_TOTAL):
            if conteo[i] > 0:
                promedio[i, :] /= conteo[i]
            else:
                promedio[i, :] = self.Phi_total[i, :]
        return promedio
    
    def _preparar_objetivo_canal(self, canal, sr, idx_paso, ventana_muestras, hop_muestras):
        inicio = idx_paso * hop_muestras
        fin = inicio + ventana_muestras
        segmento = canal[inicio:fin] if fin <= len(canal) else canal[inicio:]
        if len(segmento) < ventana_muestras:
            segmento = np.pad(segmento, (0, ventana_muestras - len(segmento)))
        
        fft = np.fft.rfft(segmento)
        potencia = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(len(segmento), 1 / sr)
        
        bandas = np.logspace(np.log10(F_MIN), np.log10(F_MAX), DIM_AUD + 1)
        objetivo = np.zeros(DIM_AUD)
        for b in range(DIM_AUD):
            mask = (freqs >= bandas[b]) & (freqs < bandas[b + 1])
            if np.any(mask):
                objetivo[b] = np.mean(potencia[mask])
        
        max_val = np.max(objetivo)
        if max_val > 0:
            objetivo /= max_val
        
        return objetivo.reshape(-1, 1) * np.ones((1, DIM_TIME))
    
    def _calcular_gradiente_energetico(self, obj_L, obj_R):
        if BANDA_TRANS >= DIM_AUD:
            return 0.0
        energia_L = float(np.mean(obj_L[BANDA_TRANS:, :] ** 2))
        energia_R = float(np.mean(obj_R[BANDA_TRANS:, :] ** 2))
        total = energia_L + energia_R + 1e-10
        return (energia_R - energia_L) / total
    
    def _actualizar_act_busc(self, gradiente_E):
        ab0, ab1 = idx['act_busc']
        senal = PHI_EQUILIBRIO + float(np.tanh(K_BUSC * gradiente_E)) * DIFUSION_BASE
        self.Phi_total[ab0:ab1, :] = (
            (1.0 - DECAIMIENTO_ACT_BUSC) * self.Phi_total[ab0:ab1, :] +
            DECAIMIENTO_ACT_BUSC * senal
        )
    
    def _aplicar_forzamiento_busc_a_ganglio(self):
        ab0, ab1 = idx['act_busc']
        g0, g1 = idx['G']
        estado_busc = float(np.mean(self.Phi_total[ab0:ab1, :])) - PHI_EQUILIBRIO
        n = min(ab1 - ab0, g1 - g0)
        self.Phi_total[g0:g0 + n, :] += EPSILON_BUSC_G * estado_busc
    
    def _aplicar_orientacion(self, gradiente_E, campo_otro_avg=0.0):
        acg0 = idx['act_geom'][0]
        acg1 = idx['act_geom'][1]
        mitad = max(1, (acg1 - acg0) // 2)
        
        senal_grad = float(np.clip(
            gradiente_E * DIFUSION_BASE * K_ORIENT * DT, -0.1, 0.1
        ))
        
        senal_acople = campo_otro_avg * GANANCIA_META * DT
        senal_acople = np.clip(senal_acople, -0.2, 0.2)
        
        aud_L = self.Phi_total[idx['aud_L'][0]:idx['aud_L'][1], :]
        aud_R = self.Phi_total[idx['aud_R'][0]:idx['aud_R'][1], :]
        aud_dir = (aud_L - aud_R).mean(axis=1)
        norm_dir = np.linalg.norm(aud_dir)
        
        if norm_dir > 1e-10:
            aud_dir_n = aud_dir / norm_dir
            min_dim = min(self.W_rec.shape[1], aud_dir_n.shape[0])
            sesgo_dir = float(np.mean(self.W_rec[:, :min_dim] @ aud_dir_n[:min_dim]))
        else:
            sesgo_dir = 0.0
        
        sesgo_rec = float(np.tanh(sesgo_dir)) * DIFUSION_BASE * DT
        senal_total = senal_grad + sesgo_rec + senal_acople
        
        self.Phi_total[acg0:acg0 + mitad, :] += senal_total
        self.Phi_total[acg0 + mitad:acg1, :] -= senal_total
    
    def _aplicar_protocolo_disenso(self, omega_propio, omega_otro):
        diferencia = abs(omega_propio - omega_otro)
        
        if diferencia > DISSENSO_THRESHOLD:
            if abs(omega_propio - PHI_EQUILIBRIO) < abs(omega_otro - PHI_EQUILIBRIO):
                self.historial_disenso.append(1)
                inversion_strength = DISSENSO_STRENGTH * DT
                self.W_rec *= (1 - inversion_strength)
                self.W_rec += inversion_strength * np.random.normal(0, 0.1, self.W_rec.shape)
                self.W_rec = np.clip(self.W_rec, -W_MAX, W_MAX)
                return True
        else:
            self.historial_disenso.append(0)
        
        return False
    
    def _calcular_parametros_actuacion(self):
        act_perm = self.Phi_total[idx['act_perm'][0]:idx['act_perm'][1], :]
        act_geom = self.Phi_total[idx['act_geom'][0]:idx['act_geom'][1], :]
        nivel_perm = float(np.mean(np.tanh(act_perm)))
        frac_base = 0.25 + 0.75 * (nivel_perm + 1.0) / 2.0
        mitad = max(1, DIM_ACT // 2)
        g_baja = float(np.mean(act_geom[:mitad, :]))
        g_alta = float(np.mean(act_geom[mitad:, :]))
        sesgo = float(np.tanh(g_alta - g_baja))
        asimetria = float(np.tanh(g_baja - g_alta))
        frac_L = float(np.clip(frac_base * (1.0 + asimetria * 0.5), 0.1, 1.0))
        frac_R = float(np.clip(frac_base * (1.0 - asimetria * 0.5), 0.1, 1.0))
        return frac_L, frac_R, sesgo, asimetria, nivel_perm
    
    def _aplicar_entrada_cualitativa(self, obj_L, obj_R, frac_L, frac_R, sesgo):
        def aplicar_canal(obj_full, frac, sl):
            n_act = max(1, int(DIM_AUD * frac))
            if sesgo > 0:
                ini = int(DIM_AUD * min(sesgo, 0.8) * 0.5)
                fin = min(DIM_AUD, ini + n_act)
            else:
                ini, fin = 0, n_act
            obj_mod = np.zeros((DIM_AUD, DIM_TIME), dtype=np.float32)
            obj_mod[ini:fin, :] = obj_full[ini:fin, :]
            self.Phi_total[sl, :] = ((1 - ALPHA_FIJO) * self.Phi_total[sl, :]
                                     + ALPHA_FIJO * obj_mod)
        aplicar_canal(obj_L, frac_L, slice(idx['aud_L'][0], idx['aud_L'][1]))
        aplicar_canal(obj_R, frac_R, slice(idx['aud_R'][0], idx['aud_R'][1]))
    
    def _explorar_actuadores(self, lf_activa, eficiencia):
        AMPLITUD_MAX = DIFUSION_BASE
        ap0, ap1 = idx['act_perm']
        ag0, ag1 = idx['act_geom']
        if lf_activa:
            amplitud = AMPLITUD_MAX * min(1.0, self.explorador.pasos_en_lf / TAU_EXPLORACION)
            if self.explorador.mejor_config is not None:
                nivel = float(np.mean(np.tanh(self.Phi_total[ap0:ap1, :])))
                sesgo = ((self.explorador.mejor_config[0] + self.explorador.mejor_config[1])
                         / 2.0 - nivel)
                ruido_perm = np.random.normal(sesgo * 0.5, amplitud,
                                              (ap1 - ap0, DIM_TIME))
            else:
                ruido_perm = np.random.normal(0, amplitud, (ap1 - ap0, DIM_TIME))
            ruido_geom = np.random.normal(0, amplitud * 0.5, (ag1 - ag0, DIM_TIME))
            self.Phi_total[ap0:ap1, :] += ruido_perm * DT
            self.Phi_total[ag0:ag1, :] += ruido_geom * DT
        else:
            if self.explorador.mejor_config is not None:
                nivel = float(np.mean(np.tanh(self.Phi_total[ap0:ap1, :])))
                corr = (self.explorador.mejor_config[0] - nivel) * DIFUSION_BASE * DT
                self.Phi_total[ap0:ap1, :] += corr
    
    def _aplicar_plasticidad_dual(self, region_int, region_aud):
        min_prof = min(self.W_prof.shape[0], region_int.shape[0])
        min_cols = min(self.W_prof.shape[1], region_aud.shape[0])
        W_p = self.W_prof[:min_prof, :min_cols]
        W_r = self.W_rec[:min_prof, :min_cols]
        r_i = region_int[:min_prof, :]
        r_a = region_aud[:min_cols, :]
        corr_prof = (r_i @ r_a.T) / DIM_TIME
        dW_prof = ETA_PROFUNDA_BASE * corr_prof - TAU_PROFUNDA * W_p
        W_p_nueva = np.clip(W_p + dW_prof * DT, -W_MAX, W_MAX)
        self.W_prof[:min_prof, :min_cols] = W_p_nueva
        pred_rec = np.tanh(W_r @ r_a)
        error_rec = float(np.mean((pred_rec - r_i) ** 2))
        pred_prof = W_p_nueva @ r_a
        error_prof = float(np.mean((pred_prof - r_i) ** 2))
        coherencia = error_prof / (error_rec + error_prof + 1e-10)
        tasa_aprendizaje = ETA_RECIENTE_BASE * coherencia
        corr_rec = (r_i @ r_a.T) / DIM_TIME
        dW_rec = tasa_aprendizaje * corr_rec - TAU_RECIENTE * W_r
        self.W_rec[:min_prof, :min_cols] = np.clip(W_r + dW_rec * DT, -W_MAX, W_MAX)
        M_plast = np.zeros((DIM_INTERNA, DIM_TIME))
        delta_p = W_p_nueva @ r_a - r_i
        delta_r = self.W_rec[:min_prof, :min_cols] @ r_a - r_i
        M_plast[:min_prof, :] = (delta_p + delta_r) * 0.01
        self.Phi_total[idx['int'][0]:idx['int'][0] + DIM_INTERNA, :] += M_plast
        self.Phi_int_historia = 0.95 * self.Phi_int_historia + 0.05 * region_int
        return error_rec, coherencia
    
    def _aplicar_plasticidad_meta(self, region_meta, campo_otro):
        min_meta = min(self.W_meta.shape[0], region_meta.shape[0])
        min_campo = min(self.W_meta.shape[1], campo_otro.shape[0])
        
        W_m = self.W_meta[:min_meta, :min_campo]
        r_m = region_meta[:min_meta, :]
        c_o = campo_otro[:min_campo, :]
        
        corr_meta = (r_m @ c_o.T) / DIM_TIME
        dW_meta = 0.01 * corr_meta - 0.001 * W_m
        self.W_meta[:min_meta, :min_campo] = np.clip(W_m + dW_meta * DT, -W_MAX, W_MAX)
    
    def _actualizar_campo_principal(self, obj_L, obj_R, frac_L, frac_R, sesgo):
        omega_n, amort_n = self._calcular_frecuencias_naturales(DIM_TOTAL)
        prom = self._calcular_promedio_vecinos()
        difusion = DIFUSION_BASE * (prom - self.Phi_total)
        desv = self.Phi_total - prom
        reaccion = GANANCIA_REACCION * desv * (1 - desv ** 2)
        term_osc = (-omega_n ** 2 * (self.Phi_total - PHI_EQUILIBRIO)
                    - amort_n * self.Phi_vel_total)
        
        region_int = self.Phi_total[idx['int'][0]:idx['int'][1], :]
        aud_L = self.Phi_total[idx['aud_L'][0]:idx['aud_L'][1], :]
        aud_R = self.Phi_total[idx['aud_R'][0]:idx['aud_R'][1], :]
        aud_comb = aud_L - aud_R
        
        error_rec, coherencia = self._aplicar_plasticidad_dual(region_int, aud_comb)
        
        self._aplicar_entrada_cualitativa(obj_L, obj_R, frac_L, frac_R, sesgo)
        
        dPhi_vel = term_osc + reaccion + difusion
        self.Phi_vel_total += DT * dPhi_vel
        self.Phi_total += DT * self.Phi_vel_total
        
        var_int = np.var(self.Phi_total[idx['int'][0]:idx['int'][1], :])
        if var_int < DIFUSION_BASE * 1e-4:
            self.Phi_total[idx['int'][0]:idx['int'][1], :] += \
                np.random.normal(0, 0.01, (DIM_INTERNA, DIM_TIME))
        
        self.Phi_total = np.clip(self.Phi_total, LIMITE_MIN, LIMITE_MAX)
        self.Phi_vel_total = np.clip(self.Phi_vel_total, -5.0, 5.0)
        
        lf_activa = error_rec > DIFUSION_BASE ** 2
        return lf_activa, error_rec
    
    def _calcular_omega(self, gradiente_hist_fase):
        if len(gradiente_hist_fase) < 2:
            return 0.0
        ag0, ag1 = idx['act_geom']
        ab0, ab1 = idx['act_busc']
        geom_medio = float(np.mean(np.tanh(self.Phi_total[ag0:ag1, :])))
        busc_medio = float(np.mean(self.Phi_total[ab0:ab1, :])) - PHI_EQUILIBRIO
        config_interna = np.array([geom_medio, busc_medio])
        grads = np.array(gradiente_hist_fase)
        grad_pos = float(np.mean(grads[grads >= 0])) if np.any(grads >= 0) else 0.0
        grad_neg = float(np.mean(np.abs(grads[grads < 0]))) if np.any(grads < 0) else 0.0
        firma_entorno = np.array([grad_pos, -grad_neg])
        norma_c = np.linalg.norm(config_interna)
        norma_f = np.linalg.norm(firma_entorno)
        if norma_c < 1e-10 or norma_f < 1e-10:
            return 0.0
        return float(np.dot(config_interna, firma_entorno) / (norma_c * norma_f))
    
    def _calcular_s_shared(self, omega_otro):
        if abs(self.ultima_omega + omega_otro) < 1e-10:
            return 0.0
        s = abs(self.ultima_omega - omega_otro) / (abs(self.ultima_omega) + abs(omega_otro) + 1e-10)
        return min(1.0, s)
    
    def entrenar(self, archivos, estimulo, duracion):
        if estimulo not in archivos:
            print(f"  [{self.nombre}] ERROR: {estimulo} no encontrado")
            return False
        
        _, sr, c_L, c_R = archivos[estimulo]
        vent = int(sr * VENTANA_FFT_MS / 1000)
        hop = int(sr * HOP_FFT_MS / 1000)
        n_pasos = int(duracion / DT)
        n_pasos = min(n_pasos, len(c_L) // hop + 1)
        
        gradiente_hist = []
        
        for paso in range(n_pasos):
            obj_L = self._preparar_objetivo_canal(c_L, sr, paso, vent, hop)
            obj_R = self._preparar_objetivo_canal(c_R, sr, paso, vent, hop)
            grad_E = self._calcular_gradiente_energetico(obj_L, obj_R)
            gradiente_hist.append(grad_E)
            
            self._actualizar_act_busc(grad_E)
            self._aplicar_forzamiento_busc_a_ganglio()
            self._aplicar_orientacion(grad_E)
            
            fL, fR, sesgo, _, _ = self._calcular_parametros_actuacion()
            self._actualizar_campo_principal(obj_L, obj_R, fL, fR, sesgo)
            
            omega = self._calcular_omega(gradiente_hist)
            self.ultima_omega = omega
        
        print(f"  [{self.nombre}] Entrenado con {estimulo}")
        return True
    
    def evaluar_con_historial(self, archivos, estimulo, duracion, otro_agente=None):
        if estimulo not in archivos:
            return None
        
        _, sr, c_L, c_R = archivos[estimulo]
        vent = int(sr * VENTANA_FFT_MS / 1000)
        hop = int(sr * HOP_FFT_MS / 1000)
        n_pasos = int(duracion / DT)
        n_pasos = min(n_pasos, len(c_L) // hop + 1)
        
        historial_omega = []
        historial_gradE = []
        gradiente_hist = []
        
        for paso in range(n_pasos):
            obj_L = self._preparar_objetivo_canal(c_L, sr, paso, vent, hop)
            obj_R = self._preparar_objetivo_canal(c_R, sr, paso, vent, hop)
            grad_E = self._calcular_gradiente_energetico(obj_L, obj_R)
            gradiente_hist.append(grad_E)
            historial_gradE.append(grad_E)
            
            campo_otro_avg = 0.0
            omega_otro = 0.0
            if otro_agente is not None:
                campo_otro_avg = float(np.mean(otro_agente.Phi_total))
                omega_otro = otro_agente.ultima_omega
            
            self._actualizar_act_busc(grad_E)
            self._aplicar_forzamiento_busc_a_ganglio()
            self._aplicar_orientacion(grad_E, campo_otro_avg)
            
            fL, fR, sesgo, _, _ = self._calcular_parametros_actuacion()
            self._actualizar_campo_principal(obj_L, obj_R, fL, fR, sesgo)
            
            omega = self._calcular_omega(gradiente_hist)
            historial_omega.append(omega)
            
            if otro_agente is not None:
                region_meta = self.Phi_total[idx['meta'][0]:idx['meta'][1], :]
                self._aplicar_plasticidad_meta(region_meta, otro_agente.Phi_total)
                self._aplicar_protocolo_disenso(omega, omega_otro)
                s_shared = self._calcular_s_shared(omega_otro)
                self.historial_s_shared.append(s_shared)
            
            self.ultima_omega = omega
        
        self.historial_omega_tiempo = historial_omega
        self.historial_gradE = historial_gradE
        
        omega_final = float(np.mean(historial_omega[-VENTANA_FINAL_PASOS:])) if len(historial_omega) > VENTANA_FINAL_PASOS else float(np.mean(historial_omega))
        return omega_final
    
    def get_historial_omega(self):
        return self.historial_omega_tiempo
    
    def get_historial_gradE(self):
        return self.historial_gradE
    
    def get_s_shared_historial(self):
        return self.historial_s_shared


# ============================================================
# CARGA DE ARCHIVOS
# ============================================================
def cargar_sonidos(directorio='audio_binaural'):
    archivos = {}
    
    print(f"\n[Carga] Desde '{directorio}/'...")
    
    nombres = ['BigBang_pos60deg', 'BigBang_neg60deg'] + ESTIMULOS_BLUE
    
    for nombre in nombres:
        filepath = os.path.join(directorio, nombre + '.wav')
        if os.path.exists(filepath):
            try:
                data, sr = sf.read(filepath, dtype='float32')
                if data.ndim == 1:
                    canal_L = data
                    canal_R = data.copy()
                else:
                    canal_L = data[:, 0]
                    canal_R = data[:, 1] if data.shape[1] > 1 else data[:, 0].copy()
                archivos[nombre] = (filepath, sr, canal_L, canal_R)
                duracion = len(canal_L)/sr
                print(f"    [OK] {nombre:45s} ({duracion:.1f}s)")
            except Exception as e:
                print(f"    [X] {nombre:45s} {e}")
        else:
            print(f"    [X] {nombre:45s} no encontrado")
    
    print(f"  Carga completada: {len(archivos)} archivos.")
    return archivos


# ============================================================
# EXPERIMENTO: BLUE MONDAY
# ============================================================
def experimento_blue_monday(archivos):
    print("\n" + "=" * 80)
    print("EXPERIMENTO: Blue Monday - 7.5 minutos")
    print("=" * 80)
    print()
    
    resultados = {}
    
    # === FASE 1: Entrenamiento base ===
    print("  FASE 1: Entrenamiento base con BigBang")
    print("  " + "-" * 50)
    
    agente_base = AgenteVST("Base", seed=42)
    agente_base.entrenar(archivos, 'BigBang_pos60deg', DURACION_ENTRENAMIENTO)
    
    # === FASE 2: Evaluación de cada estímulo con historial completo ===
    print("\n  FASE 2: Evaluación de Blue Monday (7.5 min)")
    print("  " + "-" * 50)
    
    resultados['evolucion'] = {}
    
    for estimulo in ESTIMULOS_BLUE:
        print(f"\n    Evaluando {estimulo}...")
        
        agente = AgenteVST(f"Agent_{estimulo[:20]}", seed=42)
        agente.entrenar(archivos, 'BigBang_pos60deg', DURACION_ENTRENAMIENTO)
        
        omega_final = agente.evaluar_con_historial(archivos, estimulo, DURACION_BLUE_MONDAY)
        
        historial = agente.get_historial_omega()
        gradE_hist = agente.get_historial_gradE()
        
        print(f"      Ω final: {omega_final:.4f}")
        print(f"      Ω medio: {np.mean(historial):.4f}")
        print(f"      Ω std:   {np.std(historial):.4f}")
        print(f"      Rango Ω: [{np.min(historial):.4f}, {np.max(historial):.4f}]")
        
        resultados['evolucion'][estimulo] = {
            'omega_final': omega_final,
            'omega_medio': np.mean(historial),
            'omega_std': np.std(historial),
            'omega_min': np.min(historial),
            'omega_max': np.max(historial),
            'historial': historial,
            'gradE_historial': gradE_hist
        }
    
    # === FASE 3: Dos agentes acoplados escuchando Blue Monday ===
    print("\n  FASE 3: Dos agentes acoplados escuchando Blue Monday expandido")
    print("  " + "-" * 50)
    
    estimulo_expandido = 'Blue_Monday_binaural_expandido'
    
    agente_A = AgenteVST("Agente_A", seed=42)
    agente_B = AgenteVST("Agente_B", seed=43)
    
    agente_A.entrenar(archivos, 'BigBang_pos60deg', DURACION_ENTRENAMIENTO)
    agente_B.entrenar(archivos, 'BigBang_neg60deg', DURACION_ENTRENAMIENTO)
    
    print("\n    Evaluando con acoplamiento mutuo...")
    omega_A = agente_A.evaluar_con_historial(archivos, estimulo_expandido, DURACION_BLUE_MONDAY, agente_B)
    omega_B = agente_B.evaluar_con_historial(archivos, estimulo_expandido, DURACION_BLUE_MONDAY, agente_A)
    
    historial_A = agente_A.get_historial_omega()
    historial_B = agente_B.get_historial_omega()
    s_shared_A = agente_A.get_s_shared_historial()
    s_shared_B = agente_B.get_s_shared_historial()
    
    print(f"      Agente_A: Ω final = {omega_A:.4f}")
    print(f"      Agente_B: Ω final = {omega_B:.4f}")
    print(f"      Correlación Ω_A vs Ω_B: {np.corrcoef(historial_A, historial_B)[0,1]:.4f}")
    
    resultados['acoplados'] = {
        'omega_A_final': omega_A,
        'omega_B_final': omega_B,
        'historial_A': historial_A,
        'historial_B': historial_B,
        's_shared_A': s_shared_A,
        's_shared_B': s_shared_B,
        'correlacion': np.corrcoef(historial_A, historial_B)[0,1]
    }
    
    return resultados


# ============================================================
# MAIN
# ============================================================
def main():
    archivos = cargar_sonidos('audio_binaural')
    
    print("\n" + "█" * 100)
    print("EXPERIMENTO V108 — BLUE MONDAY")
    print("█" * 100)
    
    resultados = experimento_blue_monday(archivos)
    
    # ============================================================
    # REPORTE FINAL
    # ============================================================
    print()
    print("=" * 100)
    print("REPORTE DE OBSERVACIONES - v108")
    print("=" * 100)
    
    print("\n  📊 EVOLUCIÓN DE Ω DURANTE BLUE MONDAY")
    print("  " + "-" * 70)
    print(f"    {'Estímulo':50s} {'Ω final':10s} {'Ω medio':10s} {'Rango':15s}")
    print("    " + "-" * 70)
    
    for estimulo, data in resultados['evolucion'].items():
        nombre_corto = estimulo.replace('Blue_Monday_binaural_expandido', 'Expandido').replace('_binaural', '')
        print(f"    {nombre_corto:50s} {data['omega_final']:8.4f}   {data['omega_medio']:8.4f}   [{data['omega_min']:.3f}, {data['omega_max']:.3f}]")
    
    print("\n  📊 AGENTES ACOPLADOS")
    print("  " + "-" * 50)
    data = resultados['acoplados']
    print(f"    Agente_A (BigBang_pos): Ω final = {data['omega_A_final']:.4f}")
    print(f"    Agente_B (BigBang_neg): Ω final = {data['omega_B_final']:.4f}")
    print(f"    Correlación entre agentes: {data['correlacion']:.4f}")
    
    # ============================================================
    # GRÁFICOS
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Gráfico 1: Evolución temporal de Ω para los 3 estímulos
    ax = axes[0, 0]
    tiempo = np.arange(len(resultados['evolucion'][ESTIMULOS_BLUE[0]]['historial'])) * DT
    for estimulo in ESTIMULOS_BLUE:
        nombre = estimulo.replace('Blue_Monday_binaural_expandido', 'Expandido').replace('_binaural', '')
        historial = resultados['evolucion'][estimulo]['historial']
        # Suavizar para visualización
        historial_suave = gaussian_filter1d(historial, sigma=50)
        ax.plot(tiempo / 60, historial_suave, label=nombre, linewidth=1.5)
    ax.set_xlabel('Tiempo (minutos)')
    ax.set_ylabel('Ω')
    ax.set_title('Evolución de Ω durante Blue Monday (7.5 min)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.2, 1.2)
    
    # Gráfico 2: Histograma de Ω
    ax = axes[0, 1]
    for estimulo in ESTIMULOS_BLUE:
        nombre = estimulo.replace('Blue_Monday_binaural_expandido', 'Expandido').replace('_binaural', '')
        historial = resultados['evolucion'][estimulo]['historial']
        ax.hist(historial, bins=50, alpha=0.5, label=nombre)
    ax.set_xlabel('Ω')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Distribución de Ω')
    ax.legend()
    
    # Gráfico 3: Agentes acoplados
    ax = axes[1, 0]
    data_acoplados = resultados['acoplados']
    tiempo_acoplados = np.arange(len(data_acoplados['historial_A'])) * DT
    # Suavizar
    hist_A_suave = gaussian_filter1d(data_acoplados['historial_A'], sigma=50)
    hist_B_suave = gaussian_filter1d(data_acoplados['historial_B'], sigma=50)
    ax.plot(tiempo_acoplados / 60, hist_A_suave, label='Agente_A (BigBang_pos)', linewidth=1.5)
    ax.plot(tiempo_acoplados / 60, hist_B_suave, label='Agente_B (BigBang_neg)', linewidth=1.5)
    ax.set_xlabel('Tiempo (minutos)')
    ax.set_ylabel('Ω')
    ax.set_title('Agentes acoplados escuchando Blue Monday expandido')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.2, 1.2)
    
    # Gráfico 4: S_shared a lo largo del tiempo
    ax = axes[1, 1]
    if data_acoplados['s_shared_A'] and len(data_acoplados['s_shared_A']) > 0:
        s_shared_suave = gaussian_filter1d(data_acoplados['s_shared_A'], sigma=50)
        ax.plot(tiempo_acoplados[:len(s_shared_suave)] / 60, s_shared_suave, color='purple', linewidth=1.5)
        ax.axhline(y=0.3, color='green', linestyle='--', alpha=0.7, label='Umbral semiótico (0.3)')
        ax.axhline(y=0.1, color='orange', linestyle=':', alpha=0.7, label='Umbral débil (0.1)')
        ax.set_xlabel('Tiempo (minutos)')
        ax.set_ylabel('S_shared')
        ax.set_title('Representación compartida durante Blue Monday')
        ax.legend()
        ax.set_ylim(0, 1)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('v108_logs', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'v108_logs/v108_resultados_{timestamp}.png', dpi=150)
    
    print(f"\n  Gráfico guardado: v108_logs/v108_resultados_{timestamp}.png")
    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)


if __name__ == "__main__":
    main()