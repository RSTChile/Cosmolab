#!/usr/bin/env python3
"""
VSTCosmos v94 — Cierre W_rec → act_geom: dos variantes

Base: v93 limpio (sin código suelto fuera de main).
Corrección estructural: W_rec ahora retroalimenta act_geom.

El problema de v93: act_geom respondía solo al gradiente instantáneo.
El entrenamiento (W_rec) no afectaba la orientación persistente.
Resultado: Instancia B (-60°) no invertía patrón de Instancia A (+60°).

Cierre implementado:
Las dos variantes derivan su escala de los parámetros existentes —
sin constantes arbitrarias.

VARIANTE 1 — Aditiva:
  sesgo_rec = tanh(mean(W_rec @ region_int_medio)) * DIFUSION_BASE
  señal_total = señal_gradiente + sesgo_rec
  El sesgo compite con el gradiente en igualdad de escala.
  Fuerza del cierre proporcional a cuánto aprendió W_rec.

VARIANTE 2 — Escala de K_ORIENT:
  sesgo_norm = tanh(mean(W_rec @ region_int_medio)) / W_MAX
  k_mod = K_ORIENT * (1 + sesgo_norm)
  K_ORIENT se amplifica o atenúa según la predicción de W_rec.
  Cuando W_rec y gradiente apuntan igual: respuesta amplificada.
  Cuando apuntan opuesto: respuesta atenuada.

Experimento: 4 instancias × 5 ciclos
  A_V1: entrena +60°, variante aditiva
  A_V2: entrena +60°, variante escala
  B_V1: entrena -60°, variante aditiva    ← control inversión
  B_V2: entrena -60°, variante escala     ← control inversión

Criterio clave: C31_inv — ¿Ω_F13_B > Ω_F2_B?
Si sí: W_rec retroalimenta orientación → cierre semiótico verificado.
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import csv
import warnings
import os
from scipy import signal as scipy_signal

try:
    import soundfile as sf
    HAS_SF = True
except ImportError:
    HAS_SF = False

warnings.filterwarnings('ignore')

# ============================================================
# PARÁMETROS DE LA FÍSICA DEL CAMPO
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

LIMITE_MIN  = 0.0
LIMITE_MAX  = 1.0
W_MAX       = 1.0
ALPHA_FIJO  = 0.05
DT          = 0.01
DIM_TIME    = 100

# Constantes físicas para binaural
DIAMETRO_CABEZA  = 0.175
VELOCIDAD_SONIDO = 343.0
ITD_MAX_SEG      = DIAMETRO_CABEZA / VELOCIDAD_SONIDO
F_TRANS_HZ       = VELOCIDAD_SONIDO / DIAMETRO_CABEZA   # ≈ 1960 Hz

# ============================================================
# ARQUITECTURA DEL CAMPO EXPANDIDO
# ============================================================
DIM_GANGLIO  = DIM_INTERNA // 2   # 16
DIM_AUD      = DIM_GANGLIO        # 16  ← definido UNA SOLA VEZ, antes de BANDA_TRANS
DIM_ACT      = DIM_GANGLIO // 2   # 8

DIM_AUD_L    = DIM_AUD
DIM_AUD_R    = DIM_AUD
DIM_ACT_PERM = DIM_ACT
DIM_ACT_GEOM = DIM_ACT
DIM_ACT_BUSC = DIM_ACT
DIM_ACT_MANT = DIM_ACT

# BANDA_TRANS: después de DIM_AUD
BANDA_TRANS = int(DIM_AUD * np.log10(F_TRANS_HZ / F_MIN)
                  / np.log10(F_MAX / F_MIN))
BANDA_TRANS = max(1, min(BANDA_TRANS, DIM_AUD - 1))   # = 11

# Parámetros para orientación
K_BUSC               = T_PROFUNDA_SEG / T_RECIENTE_SEG   # = 10
K_ORIENT             = T_PROFUNDA_SEG / T_RECIENTE_SEG   # = 10
DECAIMIENTO_ACT_BUSC = DT / T_RECIENTE_SEG              # = 0.005
EPSILON_BUSC_G       = DIFUSION_BASE * K_BUSC * DT      # = 0.015

# Índices de regiones
idx = {}
idx['int']      = (0,                   DIM_INTERNA)
idx['G']        = (DIM_INTERNA,         DIM_INTERNA + DIM_GANGLIO)
idx['aud_L']    = (idx['G'][1],         idx['G'][1]        + DIM_AUD_L)
idx['aud_R']    = (idx['aud_L'][1],     idx['aud_L'][1]    + DIM_AUD_R)
idx['act_perm'] = (idx['aud_R'][1],     idx['aud_R'][1]    + DIM_ACT_PERM)
idx['act_geom'] = (idx['act_perm'][1],  idx['act_perm'][1] + DIM_ACT_GEOM)
idx['act_busc'] = (idx['act_geom'][1],  idx['act_geom'][1] + DIM_ACT_BUSC)
idx['act_mant'] = (idx['act_busc'][1],  idx['act_busc'][1] + DIM_ACT_MANT)
DIM_TOTAL = idx['act_mant'][1]

# Topología — act_busc fuera del equilibrio difusivo (desde v86)
VECINDADES = [
    ('int',      'G'),
    ('G',        'aud_L'),
    ('G',        'aud_R'),
    ('G',        'act_perm'),
    ('G',        'act_geom'),
    # ('G',      'act_busc'),  ← ELIMINADA desde v86
    ('G',        'act_mant'),
    ('aud_L',    'aud_R'),
    ('act_perm', 'aud_L'),
    ('act_perm', 'aud_R'),
    ('act_geom', 'aud_L'),
    ('act_geom', 'aud_R'),
]

print("=" * 100)
print("VSTCosmos v94 — Cierre W_rec → act_geom: variante aditiva vs escala de K_ORIENT")
print("")
print("  V1 (aditiva):  señal_total = señal_gradiente + sesgo_rec")
print("  V2 (escala):   k_mod = K_ORIENT × (1 + sesgo_norm)")
print("  4 instancias × 5 ciclos: A_V1, A_V2, B_V1, B_V2")
print("  Control clave: ¿B invierte patrón de A?")
print("")
print(f"  BANDA_TRANS={BANDA_TRANS} (F>{F_TRANS_HZ:.0f}Hz, {DIM_AUD-BANDA_TRANS} bandas ILD)")
print(f"  DIM_TOTAL={DIM_TOTAL}")
print("=" * 100)

# ============================================================
# CARGA DE ARCHIVOS BINAURALES
# ============================================================
def cargar_todos_binaurales(directorio='audio_binaural', duracion=35.0):
    mapping = {
        'voz_pos':         'Voz_Estudio_pos60deg.wav',
        'voz_neg':         'Voz_Estudio_neg60deg.wav',
        'musica_pos':      'Brandemburgo_pos60deg.wav',
        'voz_viento1_pos': 'Voz+Viento_1_pos60deg.wav',
        'voz_viento2_pos': 'Voz+Viento_2_pos60deg.wav',
        'tono_pos':        'Tono puro_pos60deg.wav',
        'ruido_pos':       'Ruido blanco_pos60deg.wav',
        'ritmos_pos':      'Ritmos aleatorios_pos60deg.wav',
        'ondas_pos':       'Ondas mixtas_pos60deg.wav',
        'pulso_pos':       'Pulso logaritmico_pos60deg.wav',
        'viento_pos':      'Viento_pos60deg.wav',
        'bigbang_pos':     'BigBang_pos60deg.wav',
    }
    archivos = {}
    print(f"\n[Carga] Desde '{directorio}/'...")
    for clave, filename in mapping.items():
        filepath = os.path.join(directorio, filename)
        if not os.path.exists(filepath):
            print(f"    ⚠️  {clave:22s} no encontrado")
            continue
        try:
            data, sr = sf.read(filepath, dtype='float32')
            if clave != 'bigbang_pos' and duracion is not None:
                n = int(sr * duracion)
                if data.ndim == 1:
                    data = data[:n]
                    if len(data) < n:
                        data = np.pad(data, (0, n - len(data)))
                else:
                    data = data[:n, :]
                    if data.shape[0] < n:
                        data = np.pad(data, ((0, n - data.shape[0]), (0, 0)))
            if data.ndim == 1:
                canal_L = data
                canal_R = data.copy()
            else:
                canal_L = data[:, 0]
                canal_R = data[:, 1] if data.shape[1] > 1 else data[:, 0].copy()
            dur_real = len(canal_L) / sr
            archivos[clave] = (filepath, sr, canal_L, canal_R)
            print(f"    ✅ {clave:22s} {filename} ({dur_real:.1f}s)")
        except Exception as e:
            print(f"    ❌ {clave:22s} {e}")
    print(f"  Carga completada: {len(archivos)} archivos.")
    return archivos

# ============================================================
# CLASE EXPLORADOR
# ============================================================
class ExploradorActuadores:
    def __init__(self):
        self.historial        = []
        self.mejor_config     = None
        self.mejor_eficiencia = 0.0
        self.pasos_en_lf      = 0

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
# FUNCIONES BASE
# ============================================================
def inicializar_campo():
    np.random.seed(None)
    Phi_total     = np.random.normal(PHI_EQUILIBRIO, 0.01, (DIM_TOTAL, DIM_TIME))
    Phi_vel_total = np.zeros((DIM_TOTAL, DIM_TIME))
    return Phi_total, Phi_vel_total

def inicializar_memorias():
    W_prof           = np.zeros((DIM_INTERNA, DIM_AUD))
    W_rec            = np.zeros((DIM_INTERNA, DIM_AUD))
    Phi_int_historia = np.zeros((DIM_INTERNA, DIM_TIME))
    return W_prof, W_rec, Phi_int_historia

def _perfil_espectral_region(region, dim):
    n_bins = 50
    perfil = np.zeros(n_bins)
    for banda in range(min(dim, region.shape[0])):
        serie    = region[banda, :] - np.mean(region[banda, :])
        fft      = np.fft.rfft(serie)
        perfil  += np.abs(fft)[:n_bins] ** 2
    return perfil / max(1, dim)

def calcular_ged_entre(region_a, region_b):
    p_a = _perfil_espectral_region(region_a, region_a.shape[0])
    p_b = _perfil_espectral_region(region_b, region_b.shape[0])
    return float(np.mean(np.abs(p_a - p_b)))

def calcular_frecuencias_naturales(dim):
    bandas = np.arange(dim)
    t      = np.log1p(bandas) / np.log1p(max(dim - 1, 1))
    omega  = OMEGA_MIN + (OMEGA_MAX - OMEGA_MIN) * t
    amort  = AMORT_MIN + (AMORT_MAX - AMORT_MIN) * t
    return omega.reshape(-1, 1), amort.reshape(-1, 1)

def calcular_promedio_vecinos(Phi_total):
    promedio = np.zeros_like(Phi_total)
    conteo   = np.zeros(DIM_TOTAL)
    for reg_a, reg_b in VECINDADES:
        ia0, ia1 = idx[reg_a]
        ib0, ib1 = idx[reg_b]
        n = min(ia1 - ia0, ib1 - ib0)
        for d in range(n):
            if ia0 + d < DIM_TOTAL and ib0 + d < DIM_TOTAL:
                promedio[ia0 + d, :] += Phi_total[ib0 + d, :]
                promedio[ib0 + d, :] += Phi_total[ia0 + d, :]
                conteo[ia0 + d] += 1
                conteo[ib0 + d] += 1
    for i in range(DIM_TOTAL):
        if conteo[i] > 0:
            promedio[i, :] /= conteo[i]
        else:
            promedio[i, :] = Phi_total[i, :]
    return promedio

# ============================================================
# PREPARAR OBJETIVO
# ============================================================
def preparar_objetivo_canal(canal, sr, idx_paso, ventana_muestras,
                             hop_muestras, dim_aud, dim_time):
    inicio   = idx_paso * hop_muestras
    fin      = inicio + ventana_muestras
    segmento = canal[inicio:fin] if fin <= len(canal) else canal[inicio:]
    if len(segmento) < ventana_muestras:
        segmento = np.pad(segmento, (0, ventana_muestras - len(segmento)))

    fft      = np.fft.rfft(segmento)
    potencia = np.abs(fft) ** 2
    freqs    = np.fft.rfftfreq(len(segmento), 1 / sr)

    bandas  = np.logspace(np.log10(F_MIN), np.log10(F_MAX), dim_aud + 1)
    objetivo = np.zeros(dim_aud)
    for b in range(dim_aud):
        mask = (freqs >= bandas[b]) & (freqs < bandas[b + 1])
        if np.any(mask):
            objetivo[b] = np.mean(potencia[mask])

    max_val = np.max(objetivo)
    if max_val > 0:
        objetivo /= max_val

    return objetivo.reshape(-1, 1) * np.ones((1, dim_time))

# ============================================================
# GRADIENTE ENERGÉTICO DIRIGIDO (variable primaria de orientación)
# ============================================================
def calcular_gradiente_energetico_dirigido(obj_L, obj_R):
    """
    Diferencia de energía espectral entre canales L y R
    solo en las bandas altas donde existe ILD binaural.

    gradiente ∈ (-1, 1) por construcción — sin clip externo.

    Convención canónica:
        positivo → fuente a la DERECHA (+60°)
        negativo → fuente a la IZQUIERDA (-60°)

    El preprocesador asigna canal_L=sombreado para +60°,
    por lo tanto R tiene más energía en altas para +60°:
        gradiente = (E_R - E_L) / total → positivo para +60°
    """
    if BANDA_TRANS >= DIM_AUD:
        return 0.0
    energia_L = float(np.mean(obj_L[BANDA_TRANS:, :] ** 2))
    energia_R = float(np.mean(obj_R[BANDA_TRANS:, :] ** 2))
    total     = energia_L + energia_R + 1e-10
    return (energia_R - energia_L) / total

# ============================================================
# COHERENCIA (solo diagnóstico — no driver)
# ============================================================
def calcular_coherencia_dirigida(obj_L, obj_R, W_prof, region_int):
    """
    Coherencia en bandas altas. Solo para diagnóstico en v88.
    """
    if BANDA_TRANS >= DIM_AUD:
        return 0.0, 0.0, 0.0
    n_prof = W_prof.shape[0]
    n_cols = W_prof.shape[1]
    n_int  = region_int.shape[0]
    n_altas = DIM_AUD - BANDA_TRANS
    if n_altas <= 0:
        return 0.0, 0.0, 0.0
    perfil_L = obj_L[BANDA_TRANS:, :].mean(axis=1)
    perfil_R = obj_R[BANDA_TRANS:, :].mean(axis=1)
    perfil_i = region_int.mean(axis=1)
    min_c = min(n_cols - BANDA_TRANS, n_altas)
    min_p = min(n_prof, n_int)
    if min_c <= 0 or min_p <= 0:
        return 0.0, 0.0, 0.0
    W_alto = W_prof[:min_p, BANDA_TRANS:BANDA_TRANS + min_c]
    pred_L = W_alto @ perfil_L[:min_c].reshape(-1, 1)
    pred_R = W_alto @ perfil_R[:min_c].reshape(-1, 1)
    ref    = perfil_i[:min_p].reshape(-1, 1)
    err_L  = float(np.mean((pred_L - ref) ** 2))
    err_R  = float(np.mean((pred_R - ref) ** 2))
    total  = err_L + err_R + 1e-10
    return float((err_R - err_L) / total), err_L, err_R

# ============================================================
# ACT_BUSC — respuesta rápida al gradiente energético
# ============================================================
def actualizar_act_busc_desde_gradiente(Phi_total, gradiente_E, dt):
    """
    act_busc recibe el gradiente energético. Respuesta rápida (T_RECIENTE).
    Señal centrada en PHI_EQUILIBRIO para signo informativo.

    Con gradiente=+0.2: señal = 0.5 + tanh(10×0.2)×0.15 ≈ 0.649
    Con gradiente=-0.2: señal ≈ 0.351
    act_busc centrado: ±0.149 — comparable con Φ_G
    """
    ab0, ab1 = idx['act_busc']
    señal = PHI_EQUILIBRIO + float(np.tanh(K_BUSC * gradiente_E)) * DIFUSION_BASE
    Phi_total[ab0:ab1, :] = (
        (1.0 - DECAIMIENTO_ACT_BUSC) * Phi_total[ab0:ab1, :] +
        DECAIMIENTO_ACT_BUSC * señal
    )
    return Phi_total

# ============================================================
# FORZAMIENTO DIRIGIDO act_busc → G
# ============================================================
def aplicar_forzamiento_busc_a_ganglio(Phi_total, dt):
    """Acoplamiento dirigido — propaga gradiente desde act_busc a Φ_G."""
    ab0, ab1 = idx['act_busc']
    g0,  g1  = idx['G']
    estado_busc = float(np.mean(Phi_total[ab0:ab1, :])) - PHI_EQUILIBRIO
    n = min(ab1 - ab0, g1 - g0)
    Phi_total[g0:g0 + n, :] += EPSILON_BUSC_G * estado_busc
    return Phi_total

# ============================================================
# ACT_GEOM — VARIANTE 1: ADITIVA (cierre W_rec)
# ============================================================
def aplicar_orientacion_v1_aditiva(Phi_total, gradiente_E, W_rec, dt):
    """
    Variante 1: señal_total = señal_gradiente + sesgo_rec

    sesgo_rec = tanh(mean(W_rec @ region_int_medio)) * DIFUSION_BASE
    Mismo peso que la señal del gradiente — sin constante arbitraria.
    La fuerza del cierre emerge del aprendizaje de W_rec.

    Si W_rec aprendió +60°: sesgo_rec positivo → refuerza cuando gradE > 0
    Si W_rec aprendió -60°: sesgo_rec negativo → contrarresta cuando gradE > 0
    """
    acg0  = idx['act_geom'][0]
    acg1  = idx['act_geom'][1]
    mitad = max(1, (acg1 - acg0) // 2)

    # Señal del gradiente (igual que versiones anteriores)
    señal_grad = float(np.clip(
        gradiente_E * DIFUSION_BASE * K_ORIENT * dt, -0.1, 0.1
    ))

    # Sesgo de W_rec: predicción interna del campo sobre region_int
    region_int = Phi_total[idx['int'][0]:idx['int'][1], :]
    region_int_medio = region_int.mean(axis=1)  # (DIM_INTERNA,)

    # W_rec: (DIM_INTERNA, DIM_AUD) — proyectamos region_int sobre W_rec
    # La norma de W_rec determina la fuerza del sesgo
    min_dim = min(W_rec.shape[0], region_int_medio.shape[0])
    proyeccion = float(np.mean(np.tanh(
        W_rec[:min_dim, :] @ np.ones(W_rec.shape[1]) * region_int_medio[:min_dim].mean()
    )))
    sesgo_rec = proyeccion * DIFUSION_BASE * dt

    señal_total = señal_grad + sesgo_rec

    Phi_total[acg0:acg0 + mitad, :] += señal_total
    Phi_total[acg0 + mitad:acg1, :] -= señal_total
    return Phi_total


# ============================================================
# ACT_GEOM — VARIANTE 2: ESCALA DE K_ORIENT (cierre W_rec)
# ============================================================
def aplicar_orientacion_v2_escala(Phi_total, gradiente_E, W_rec, dt):
    """
    Variante 2: k_mod = K_ORIENT × (1 + sesgo_norm)

    sesgo_norm = tanh(mean(W_rec @ region_int_medio)) / W_MAX ∈ [-1, +1]
    K_ORIENT se amplifica (hasta 2×) o atenúa (hasta 0×) según W_rec.

    Si W_rec y gradiente apuntan igual: respuesta amplificada.
    Si W_rec apunta opuesto al gradiente: respuesta atenuada.
    La escala de modulación es K_ORIENT — sin constante adicional.
    """
    acg0  = idx['act_geom'][0]
    acg1  = idx['act_geom'][1]
    mitad = max(1, (acg1 - acg0) // 2)

    # Predicción interna normalizada por W_MAX
    region_int = Phi_total[idx['int'][0]:idx['int'][1], :]
    region_int_medio = region_int.mean(axis=1)

    min_dim = min(W_rec.shape[0], region_int_medio.shape[0])
    proyeccion = float(np.mean(np.tanh(
        W_rec[:min_dim, :] @ np.ones(W_rec.shape[1]) * region_int_medio[:min_dim].mean()
    )))
    sesgo_norm = float(np.tanh(proyeccion)) / W_MAX  # ∈ [-1/W_MAX, +1/W_MAX]

    # K_ORIENT modulado — acotado a [0, 2×K_ORIENT]
    k_mod = K_ORIENT * (1.0 + sesgo_norm)

    señal = float(np.clip(
        gradiente_E * DIFUSION_BASE * k_mod * dt, -0.1, 0.1
    ))

    Phi_total[acg0:acg0 + mitad, :] += señal
    Phi_total[acg0 + mitad:acg1, :] -= señal
    return Phi_total

# ============================================================
# ACTUACIÓN CUALITATIVA
# ============================================================
def calcular_parametros_actuacion(Phi_total):
    act_perm = Phi_total[idx['act_perm'][0]:idx['act_perm'][1], :]
    act_geom = Phi_total[idx['act_geom'][0]:idx['act_geom'][1], :]
    nivel_perm = float(np.mean(np.tanh(act_perm)))
    frac_base  = 0.25 + 0.75 * (nivel_perm + 1.0) / 2.0
    mitad    = max(1, DIM_ACT // 2)
    g_baja   = float(np.mean(act_geom[:mitad, :]))
    g_alta   = float(np.mean(act_geom[mitad:, :]))
    sesgo    = float(np.tanh(g_alta - g_baja))
    asimetria = float(np.tanh(g_baja - g_alta))
    frac_L = float(np.clip(frac_base * (1.0 + asimetria * 0.5), 0.1, 1.0))
    frac_R = float(np.clip(frac_base * (1.0 - asimetria * 0.5), 0.1, 1.0))
    return frac_L, frac_R, sesgo, asimetria, nivel_perm

def aplicar_entrada_cualitativa(Phi_total, obj_L, obj_R, frac_L, frac_R, sesgo):
    def aplicar_canal(obj_full, frac, sl):
        n_act = max(1, int(DIM_AUD * frac))
        if sesgo > 0:
            ini = int(DIM_AUD * min(sesgo, 0.8) * 0.5)
            fin = min(DIM_AUD, ini + n_act)
        else:
            ini, fin = 0, n_act
        obj_mod = np.zeros((DIM_AUD, DIM_TIME), dtype=np.float32)
        obj_mod[ini:fin, :] = obj_full[ini:fin, :]
        Phi_total[sl, :] = ((1 - ALPHA_FIJO) * Phi_total[sl, :]
                            + ALPHA_FIJO * obj_mod)
    aplicar_canal(obj_L, frac_L, slice(idx['aud_L'][0], idx['aud_L'][1]))
    aplicar_canal(obj_R, frac_R, slice(idx['aud_R'][0], idx['aud_R'][1]))
    return Phi_total

# ============================================================
# EXPLORACIÓN ACTIVA
# ============================================================
def explorar_actuadores(Phi_total, explorador, lf_activa, eficiencia, dt):
    AMPLITUD_MAX = DIFUSION_BASE
    ap0, ap1 = idx['act_perm']
    ag0, ag1 = idx['act_geom']
    if lf_activa:
        amplitud = AMPLITUD_MAX * min(1.0, explorador.pasos_en_lf / TAU_EXPLORACION)
        if explorador.mejor_config is not None:
            nivel = float(np.mean(np.tanh(Phi_total[ap0:ap1, :])))
            sesgo = ((explorador.mejor_config[0] + explorador.mejor_config[1])
                     / 2.0 - nivel)
            ruido_perm = np.random.normal(sesgo * 0.5, amplitud,
                                          (ap1 - ap0, DIM_TIME))
        else:
            ruido_perm = np.random.normal(0, amplitud, (ap1 - ap0, DIM_TIME))
        ruido_geom = np.random.normal(0, amplitud * 0.5, (ag1 - ag0, DIM_TIME))
        Phi_total[ap0:ap1, :] += ruido_perm * dt
        Phi_total[ag0:ag1, :] += ruido_geom * dt
    else:
        if explorador.mejor_config is not None:
            nivel = float(np.mean(np.tanh(Phi_total[ap0:ap1, :])))
            corr  = (explorador.mejor_config[0] - nivel) * DIFUSION_BASE * dt
            Phi_total[ap0:ap1, :] += corr
    return Phi_total

# ============================================================
# PLASTICIDAD DUAL
# ============================================================
def aplicar_plasticidad_dual(region_int, region_aud, W_prof, W_rec,
                              Phi_int_historia, dt):
    min_prof = min(W_prof.shape[0], region_int.shape[0])
    min_cols = min(W_prof.shape[1], region_aud.shape[0])
    W_p = W_prof[:min_prof, :min_cols]
    W_r = W_rec[:min_prof,  :min_cols]
    r_i = region_int[:min_prof, :]
    r_a = region_aud[:min_cols, :]
    corr_prof = (r_i @ r_a.T) / DIM_TIME
    dW_prof   = ETA_PROFUNDA_BASE * corr_prof - TAU_PROFUNDA * W_p
    W_p_nueva = np.clip(W_p + dW_prof * dt, -W_MAX, W_MAX)
    W_prof_nueva = W_prof.copy()
    W_prof_nueva[:min_prof, :min_cols] = W_p_nueva
    pred_rec   = np.tanh(W_r @ r_a)
    error_rec  = float(np.mean((pred_rec - r_i) ** 2))
    pred_prof  = W_p_nueva @ r_a
    error_prof = float(np.mean((pred_prof - r_i) ** 2))
    coherencia       = error_prof / (error_rec + error_prof + 1e-10)
    tasa_aprendizaje = ETA_RECIENTE_BASE * coherencia
    corr_rec  = (r_i @ r_a.T) / DIM_TIME
    dW_rec    = tasa_aprendizaje * corr_rec - TAU_RECIENTE * W_r
    W_r_nueva = np.clip(W_r + dW_rec * dt, -W_MAX, W_MAX)
    W_rec_nueva = W_rec.copy()
    W_rec_nueva[:min_prof, :min_cols] = W_r_nueva
    M_plast = np.zeros((DIM_INTERNA, DIM_TIME))
    delta_p = W_p_nueva @ r_a - r_i
    delta_r = W_r_nueva @ r_a - r_i
    M_plast[:min_prof, :] = (delta_p + delta_r) * 0.01
    Phi_int_historia_nueva = 0.95 * Phi_int_historia + 0.05 * region_int
    return (W_prof_nueva, W_rec_nueva, M_plast,
            error_rec, coherencia, Phi_int_historia_nueva)

# ============================================================
# Ω_ORIENT — RESPUESTA DE ORIENTACIÓN (CAMBIO ÚNICO v92)
# ============================================================
def calcular_omega_orient(Phi_total, gradiente_hist_fase):
    """
    Ω_orient = correlación coseno entre config_interna y firma_entorno.

    config_interna: [geom_medio, busc_medio]
        geom_medio: estado integrado de act_geom (orientación lenta)
        busc_medio: desplazamiento de act_busc (orientación rápida)

    firma_entorno: [grad_pos, grad_neg]
        grad_pos: promedio de gradientes positivos acumulados en la fase
        grad_neg: promedio de gradientes negativos (como magnitud opuesta)

    Esto evita que firma_entorno sea [k, k] — los dos componentes
    capturan lados complementarios del gradiente, haciendo la
    correlación coseno sensible a la dirección real de la orientación.

    Retorna escalar en [-1, +1].
    Positivo: orientación interna alineada con el entorno.
    Negativo: orientación interna contradice el entorno.
    """
    if len(gradiente_hist_fase) < 2:
        return 0.0

    ag0, ag1 = idx['act_geom']
    ab0, ab1 = idx['act_busc']

    geom_medio = float(np.mean(np.tanh(Phi_total[ag0:ag1, :])))
    busc_medio = float(np.mean(Phi_total[ab0:ab1, :])) - PHI_EQUILIBRIO

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

# ============================================================
# EFICIENCIA Y MÉTRICAS
# ============================================================
def calcular_eficiencia(Phi_total, ged_actual):
    region_int = Phi_total[idx['int'][0]:idx['int'][1], :]
    variacion  = float(np.mean(np.abs(np.diff(region_int, axis=1))))
    return ged_actual / (variacion + 1e-10), variacion

def calcular_senal_busqueda(Phi_total):
    ab0, ab1 = idx['act_busc']
    return float(np.mean(Phi_total[ab0:ab1, :])) - PHI_EQUILIBRIO

# ============================================================
# ACTUALIZACIÓN PRINCIPAL DEL CAMPO
# ============================================================
def actualizar_campo(Phi_total, Phi_vel_total, W_prof, W_rec,
                     Phi_int_historia, obj_L, obj_R,
                     frac_L, frac_R, sesgo, dt):

    omega_n, amort_n = calcular_frecuencias_naturales(DIM_TOTAL)
    prom     = calcular_promedio_vecinos(Phi_total)
    difusion = DIFUSION_BASE * (prom - Phi_total)
    desv     = Phi_total - prom
    reaccion = GANANCIA_REACCION * desv * (1 - desv ** 2)
    term_osc = (-omega_n ** 2 * (Phi_total - PHI_EQUILIBRIO)
                - amort_n * Phi_vel_total)

    region_int = Phi_total[idx['int'][0]:idx['int'][1], :]
    aud_L      = Phi_total[idx['aud_L'][0]:idx['aud_L'][1], :]
    aud_R      = Phi_total[idx['aud_R'][0]:idx['aud_R'][1], :]
    aud_comb   = (aud_L + aud_R) / 2.0

    W_prof, W_rec, M_plast, error_rec, coherencia, Phi_int_historia = \
        aplicar_plasticidad_dual(
            region_int, aud_comb, W_prof, W_rec, Phi_int_historia, dt
        )

    M_campo = np.zeros_like(Phi_total)
    n_m     = M_plast.shape[0]
    M_campo[idx['int'][0]:idx['int'][0] + n_m, :] = M_plast

    Phi_total = aplicar_entrada_cualitativa(Phi_total, obj_L, obj_R,
                                            frac_L, frac_R, sesgo)

    dPhi_vel  = term_osc + reaccion + difusion + M_campo
    Phi_vel_n = Phi_vel_total + dt * dPhi_vel
    Phi_nueva = Phi_total + dt * Phi_vel_n

    var_int = np.var(Phi_nueva[idx['int'][0]:idx['int'][1], :])
    if var_int < DIFUSION_BASE * 1e-4:
        Phi_nueva[idx['int'][0]:idx['int'][1], :] += \
            np.random.normal(0, 0.01, (DIM_INTERNA, DIM_TIME))

    lf_activa = error_rec > DIFUSION_BASE ** 2

    return (np.clip(Phi_nueva, LIMITE_MIN, LIMITE_MAX),
            np.clip(Phi_vel_n, -5.0, 5.0),
            W_prof, W_rec, Phi_int_historia,
            lf_activa, error_rec, coherencia)

# ============================================================
# ENTRENAMIENTO
# ============================================================
def entrenar(archivos, duracion=30.0, clave_audio='voz_pos', etiqueta=None, variante=1):
    if etiqueta is None:
        etiqueta = clave_audio
    print(f"\n[Entrenamiento] {etiqueta} — V{variante} ({duracion}s)")
    Phi_total, Phi_vel_total = inicializar_campo()
    W_prof, W_rec, Phi_int_historia = inicializar_memorias()
    explorador = ExploradorActuadores()

    _, sr, c_L, c_R = archivos[clave_audio]
    vent    = int(sr * VENTANA_FFT_MS / 1000)
    hop     = int(sr * HOP_FFT_MS  / 1000)
    n_pasos = int(duracion / DT)
    errores = []

    for paso in range(n_pasos):
        obj_L = preparar_objetivo_canal(c_L, sr, paso, vent, hop, DIM_AUD, DIM_TIME)
        obj_R = preparar_objetivo_canal(c_R, sr, paso, vent, hop, DIM_AUD, DIM_TIME)

        gradiente_E = calcular_gradiente_energetico_dirigido(obj_L, obj_R)
        Phi_total   = actualizar_act_busc_desde_gradiente(Phi_total, gradiente_E, DT)
        Phi_total   = aplicar_forzamiento_busc_a_ganglio(Phi_total, DT)

        # Durante entrenamiento usamos V1 base (W_rec aún no está aprendido)
        # En pasos avanzados del entrenamiento, W_rec ya tiene información
        if variante == 1:
            Phi_total = aplicar_orientacion_v1_aditiva(Phi_total, gradiente_E, W_rec, DT)
        else:
            Phi_total = aplicar_orientacion_v2_escala(Phi_total, gradiente_E, W_rec, DT)

        fL, fR, sf_v, _, _ = calcular_parametros_actuacion(Phi_total)

        Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia, \
            _, error_rec, _ = actualizar_campo(
                Phi_total, Phi_vel_total, W_prof, W_rec,
                Phi_int_historia, obj_L, obj_R, fL, fR, sf_v, DT
            )
        errores.append(error_rec)

        if paso % 500 == 0:
            print(f"    Paso {paso}/{n_pasos}, error={error_rec:.6f}")

    print(f"  ERROR_EQUILIBRIO: {min(errores):.6f}")
    print(f"  W_prof: {np.mean(np.abs(W_prof)):.4f}")
    print(f"  W_rec:  {np.mean(np.abs(W_rec)):.4f}")

    return Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia, explorador

# ============================================================
# SIMULACIÓN DE FASE
# ============================================================
def simular_fase(Phi_total, Phi_vel_total, W_prof, W_rec,
                 Phi_int_historia, historial_ef, explorador,
                 sr, canal_L, canal_R, duracion, verbose=True, variante=1):

    vent    = int(sr * VENTANA_FFT_MS / 1000)
    hop     = int(sr * HOP_FFT_MS  / 1000)

    if duracion is None:
        n_pasos_max = len(canal_L) // hop
    else:
        n_pasos_max = int(duracion / DT)

    n_pasos = min(n_pasos_max, 5000)

    hist = {k: [] for k in [
        'ged_L', 'ged_R', 'grad_E', 'act_busc', 'coh_rel',
        'geom', 'frac_L', 'frac_R', 'efic', 'lf',
        'w_rec', 'w_prof', 'G_act', 'omega'
    ]}

    gradiente_hist_fase = []  # para calcular firma_entorno en Ω_orient
    lf_prev = False
    for paso in range(n_pasos):
        obj_L = preparar_objetivo_canal(canal_L, sr, paso, vent, hop, DIM_AUD, DIM_TIME)
        obj_R = preparar_objetivo_canal(canal_R, sr, paso, vent, hop, DIM_AUD, DIM_TIME)

        # 1. Gradiente energético (variable primaria de orientación)
        gradiente_E = calcular_gradiente_energetico_dirigido(obj_L, obj_R)
        gradiente_hist_fase.append(gradiente_E)

        # 2. Coherencia (solo diagnóstico)
        region_int = Phi_total[idx['int'][0]:idx['int'][1], :]
        coh_rel, _, _ = calcular_coherencia_dirigida(
            obj_L, obj_R, W_prof, region_int
        )

        # 3. act_busc: respuesta rápida
        Phi_total = actualizar_act_busc_desde_gradiente(Phi_total, gradiente_E, DT)

        # 4. Forzamiento act_busc → G
        Phi_total = aplicar_forzamiento_busc_a_ganglio(Phi_total, DT)

        # 5. act_geom: orientación con cierre W_rec (v94)
        if variante == 1:
            Phi_total = aplicar_orientacion_v1_aditiva(Phi_total, gradiente_E, W_rec, DT)
        else:
            Phi_total = aplicar_orientacion_v2_escala(Phi_total, gradiente_E, W_rec, DT)

        # 6. Actuación
        fL, fR, sf_v, asim, _ = calcular_parametros_actuacion(Phi_total)

        # 7. GED y eficiencia
        a_L   = Phi_total[idx['aud_L'][0]:idx['aud_L'][1], :]
        a_R   = Phi_total[idx['aud_R'][0]:idx['aud_R'][1], :]
        ged_L = calcular_ged_entre(region_int, a_L)
        ged_R = calcular_ged_entre(region_int, a_R)
        ged   = (ged_L + ged_R) / 2.0
        efic, _ = calcular_eficiencia(Phi_total, ged)

        historial_ef.append(efic)
        if len(historial_ef) > TAU_EFICIENCIA * 2:
            historial_ef.pop(0)

        explorador.actualizar(lf_prev, efic, fL, fR, sf_v)

        # 8. Actualizar campo
        Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia, \
            lf_activa, error_rec, _ = actualizar_campo(
                Phi_total, Phi_vel_total, W_prof, W_rec,
                Phi_int_historia, obj_L, obj_R, fL, fR, sf_v, DT
            )

        # 9. Exploración activa
        Phi_total = explorar_actuadores(Phi_total, explorador, lf_activa, efic, DT)
        lf_prev   = lf_activa

        # 10. Métricas
        act_busc_val = calcular_senal_busqueda(Phi_total)
        geom         = float(np.mean(np.tanh(
            Phi_total[idx['act_geom'][0]:idx['act_geom'][1], :]
        )))
        G_act = float(np.mean(np.abs(
            Phi_total[idx['G'][0]:idx['G'][1], :]
        )))
        omega = calcular_omega_orient(Phi_total, gradiente_hist_fase)

        for k, v in [
            ('ged_L',   ged_L), ('ged_R',   ged_R),
            ('grad_E',  gradiente_E), ('act_busc', act_busc_val),
            ('coh_rel', coh_rel), ('geom',    geom),
            ('frac_L',  fL), ('frac_R',  fR),
            ('efic',    efic), ('lf',      lf_activa),
            ('w_rec',   np.mean(np.abs(W_rec))),
            ('w_prof',  np.mean(np.abs(W_prof))),
            ('G_act',   G_act), ('omega',   omega),
        ]:
            hist[k].append(v)

        if verbose and paso % 200 == 0:
            print(f"    t={paso*DT:.1f}s | GED={ged:.4f} | "
                  f"gradE={gradiente_E:+.4f} | busc={act_busc_val:+.4f} | "
                  f"geom={geom:+.4f} | Ω={omega:+.3f} | "
                  f"G={G_act:.4f} | efic={efic:.3f} | "
                  f"LF={'ACTIVA' if lf_activa else 'inact'}")

    def M(k): return float(np.mean(hist[k])) if hist[k] else 0.0
    n_half = len(hist['geom']) // 2

    geom_primera = float(np.mean(hist['geom'][:n_half])) if n_half > 0 else 0.0
    geom_segunda = float(np.mean(hist['geom'][n_half:])) if n_half > 0 else 0.0
    geom_conv    = (geom_primera * geom_segunda > 0) if n_half > 0 else False
    busc_segunda = float(np.mean(hist['act_busc'][n_half:])) if n_half > 0 else 0.0
    grad_medio   = M('grad_E')
    omega_media  = M('omega')
    omega_segunda = float(np.mean(hist['omega'][n_half:])) if n_half > 0 else 0.0

    if verbose:
        print(f"\n  Resumen:")
        print(f"    GED L/R:                  {M('ged_L'):.4f} / {M('ged_R'):.4f}")
        print(f"    Gradiente energético:     {grad_medio:+.4f}")
        print(f"    act_busc (2ª mitad):      {busc_segunda:+.4f}")
        print(f"    act_geom (2ª mitad):      {geom_segunda:+.4f}")
        print(f"    Convergencia geom:        {'✅ estable' if geom_conv else '⚠️ oscilante'}")
        print(f"    Coherencia (diagnóstico): {M('coh_rel'):+.5f}")
        print(f"    Ω_orient (medio):         {omega_media:+.4f}")
        print(f"    Ω_orient (2ª mitad):      {omega_segunda:+.4f}")
        print(f"    Eficiencia media:         {M('efic'):.4f}")
        print(f"    LF activa (%):            {100*M('lf'):.1f}%")
        print(f"    Mejor efic explorada:     {explorador.mejor_eficiencia:.4f}")

    return {
        'hist': hist,
        'geom_primera': geom_primera, 'geom_segunda': geom_segunda,
        'geom_conv': geom_conv, 'busc_segunda': busc_segunda,
        'grad_medio': grad_medio, 'coh_media': M('coh_rel'),
        'omega_media': omega_media, 'omega_segunda': omega_segunda,
        'mejor_ef': explorador.mejor_eficiencia,
        'phi_total': Phi_total, 'phi_vel': Phi_vel_total,
        'W_prof': W_prof, 'W_rec': W_rec,
        'Phi_int_historia': Phi_int_historia,
    }


# ============================================================
# MAIN — DOS INSTANCIAS: CONTROL DE INVERSIÓN
# ============================================================
def correr_protocolo(archivos, clave_entrenamiento, etiqueta, N_CICLOS=5, CICLOS_LOG=None, variante=1):
    """
    Corre el protocolo cíclico completo para una instancia.
    clave_entrenamiento: 'voz_pos' o 'voz_neg'
    variante: 1 (aditiva) o 2 (escala K_ORIENT)
    """
    if CICLOS_LOG is None:
        CICLOS_LOG = {1, 3, 5}

    Phi_total, Phi_vel_total, W_prof, W_rec, \
        Phi_int_historia, explorador = entrenar(
            archivos, 30.0,
            clave_audio=clave_entrenamiento,
            etiqueta=etiqueta,
            variante=variante
        )

    FASES = [
        ("F2",  'voz_pos',         +60.0, 20.0, "Dominio — voz +60°"),
        ("F3",  'musica_pos',      +60.0, 20.0, "No entrenado — música +60°"),
        ("F4",  'tono_pos',        +60.0, 20.0, "No entrenado — tono +60°"),
        ("F5",  'ritmos_pos',      +60.0, 20.0, "Ritmos irregulares +60°"),
        ("F6",  'ondas_pos',       +60.0, 20.0, "Ondas mixtas +60°"),
        ("F7",  'pulso_pos',       +60.0, 20.0, "Pulso logarítmico +60°"),
        ("F8",  'viento_pos',      +60.0, 20.0, "Viento +60°"),
        ("F9",  'voz_viento1_pos', +60.0, 20.0, "Degradado — voz+viento1 +60°"),
        ("F10", 'voz_viento2_pos', +60.0, 20.0, "Degradado — voz+viento2 +60°"),
        ("F11", 'ruido_pos',       +60.0, 20.0, "Perturbación basal — ruido +60°"),
        ("F12", 'bigbang_pos',     +60.0, 20.0, "BigBang +60° (20s por ciclo)"),
        ("F13", 'voz_neg',         -60.0, 20.0, "Re-acoplamiento opuesto — voz -60°"),
    ]

    punteros     = {clave: 0 for clave in archivos}
    historial_ef = []
    registro     = []

    print()
    print("=" * 80)
    print("PROTOCOLO — " + etiqueta + " — " + str(N_CICLOS) + " ciclos")
    print("=" * 80)

    for ciclo in range(1, N_CICLOS + 1):
        verbose = ciclo in CICLOS_LOG

        if verbose:
            print()
            print("-" * 50)
            print("CICLO " + str(ciclo) + "/" + str(N_CICLOS) + "  [" + etiqueta + "]")
            print("-" * 50)
        else:
            print("  Ciclo " + str(ciclo).rjust(2) + "...", end="", flush=True)

        metricas_ciclo = {}

        for fid, clave, angulo, dur, desc in FASES:
            if clave not in archivos:
                continue

            _, sr, c_L_full, c_R_full = archivos[clave]
            n_pasos = int(dur / DT)
            hop     = int(sr * HOP_FFT_MS    / 1000)
            vent    = int(sr * VENTANA_FFT_MS / 1000)

            inicio_m = punteros[clave] * hop
            if inicio_m + n_pasos * hop > len(c_L_full):
                punteros[clave] = 0
                inicio_m        = 0

            fin_m = inicio_m + n_pasos * hop + vent
            c_L   = c_L_full[inicio_m : min(fin_m, len(c_L_full))]
            c_R   = c_R_full[inicio_m : min(fin_m, len(c_R_full))]

            needed = n_pasos * hop + vent
            if len(c_L) < needed:
                c_L = np.pad(c_L, (0, needed - len(c_L)))
                c_R = np.pad(c_R, (0, needed - len(c_R)))

            punteros[clave] += n_pasos

            if verbose:
                print("  [" + fid + "] " + desc)

            res = simular_fase(
                Phi_total, Phi_vel_total, W_prof, W_rec,
                Phi_int_historia, historial_ef, explorador,
                sr, c_L, c_R, dur, verbose=verbose, variante=variante
            )

            metricas_ciclo[fid] = res
            Phi_total        = res['phi_total']
            Phi_vel_total    = res['phi_vel']
            W_prof           = res['W_prof']
            W_rec            = res['W_rec']
            Phi_int_historia = res['Phi_int_historia']

        def gc(fid, k):
            return metricas_ciclo[fid][k] if fid in metricas_ciclo else None

        grad_f2   = gc('F2',  'grad_medio')
        grad_f13  = gc('F13', 'grad_medio')
        busc_f2   = gc('F2',  'busc_segunda')
        busc_f13  = gc('F13', 'busc_segunda')
        geom_f2   = gc('F2',  'geom_segunda')
        geom_f13  = gc('F13', 'geom_segunda')
        omega_f2  = gc('F2',  'omega_segunda')
        omega_f13 = gc('F13', 'omega_segunda')
        efic_f2   = gc('F2',  'omega_media')   # usamos omega_media como proxy de efic_media
        # eficiencia de fase: tomamos mejor_ef del ciclo
        mejor_ef  = max([v.get('mejor_ef', 0.0)
                         for v in metricas_ciclo.values()
                         if isinstance(v, dict)], default=0.0)
        efic_f2_val  = float(np.mean(metricas_ciclo['F2']['hist']['efic']))  if 'F2'  in metricas_ciclo else 0.0
        efic_f13_val = float(np.mean(metricas_ciclo['F13']['hist']['efic'])) if 'F13' in metricas_ciclo else 0.0

        # A_sys-env = Ω_orient × eficiencia_media_de_fase
        asys_f2  = (omega_f2  * efic_f2_val)  if omega_f2  is not None else None
        asys_f13 = (omega_f13 * efic_f13_val) if omega_f13 is not None else None

        c28 = (grad_f2  is not None and grad_f13 is not None and grad_f2 * grad_f13 < 0)
        c21 = (busc_f2  is not None and busc_f13 is not None and busc_f2 * busc_f13 < 0)
        c29 = (geom_f2  is not None and geom_f13 is not None and abs(geom_f2 - geom_f13) > 0.01)
        c31 = (omega_f2 is not None and omega_f13 is not None
               and omega_f2 > 0 and omega_f2 > omega_f13)
        c32 = (asys_f2  is not None and asys_f13 is not None and asys_f2 > asys_f13)

        registro.append({
            'ciclo': ciclo,
            'grad_f2': grad_f2,   'grad_f13': grad_f13,
            'busc_f2': busc_f2,   'busc_f13': busc_f13,
            'geom_f2': geom_f2,   'geom_f13': geom_f13,
            'omega_f2': omega_f2, 'omega_f13': omega_f13,
            'efic_f2': efic_f2_val, 'efic_f13': efic_f13_val,
            'asys_f2': asys_f2,   'asys_f13': asys_f13,
            'mejor_ef': mejor_ef,
            'c28': c28, 'c21': c21, 'c29': c29, 'c31': c31, 'c32': c32,
        })

        def fmt(v):
            return str(round(v, 4)).rjust(7) if v is not None else "    N/A"

        resumen = (
            "Ω F2=" + fmt(omega_f2) + " F13=" + fmt(omega_f13) + " | "
            "A F2=" + fmt(asys_f2)  + " F13=" + fmt(asys_f13)  + " | "
            "efic=" + str(round(mejor_ef, 4)) + " | "
            "C31=" + ("OK" if c31 else "--") + " "
            "C32=" + ("OK" if c32 else "--")
        )

        if verbose:
            print("  Ciclo " + str(ciclo).rjust(2) + " — " + resumen)
        else:
            print(" " + resumen)

    return registro



def main():
    archivos = cargar_todos_binaurales('audio_binaural', 35.0)
    if not archivos:
        print("\nERROR: No se encontraron archivos. Ejecutar preprocesar_binaurales.py")
        return

    N_CICLOS   = 5
    CICLOS_LOG = {1, 3, 5}

    def correr(clave, etiqueta, v):
        print()
        print("█" * 100)
        print(f"{etiqueta} — Variante {v}")
        print("█" * 100)
        return correr_protocolo(archivos, clave, etiqueta, N_CICLOS, CICLOS_LOG, variante=v)

    reg_AV1 = correr('voz_pos', 'A_V1 (+60°, aditiva)',    1)
    reg_AV2 = correr('voz_pos', 'A_V2 (+60°, escala)',     2)
    reg_BV1 = correr('voz_neg', 'B_V1 (-60°, aditiva)',    1)
    reg_BV2 = correr('voz_neg', 'B_V2 (-60°, escala)',     2)

    print()
    print("=" * 100)
    print("DIAGNÓSTICO FINAL — v94 Cierre W_rec → act_geom")
    print("=" * 100)

    def stats(reg, key):
        vals = [r[key] for r in reg if r[key] is not None]
        if not vals:
            return 0.0, 0.0
        return float(np.mean(vals)), float(np.std(vals))

    def inv_ok(reg):
        return sum(1 for r in reg
                   if r['omega_f13'] is not None and r['omega_f2'] is not None
                   and r['omega_f13'] > r['omega_f2'])

    def c31_ok(reg):
        return sum(r['c31'] for r in reg)

    def c32_ok(reg):
        return sum(r['c32'] for r in reg)

    n = N_CICLOS
    instancias = [
        ('A_V1 (+60°, aditiva)',  reg_AV1, False),
        ('A_V2 (+60°, escala)',   reg_AV2, False),
        ('B_V1 (-60°, aditiva)',  reg_BV1, True),
        ('B_V2 (-60°, escala)',   reg_BV2, True),
    ]

    print()
    print(f"  {'Instancia':<28} {'Ω_F2':>8} {'Ω_F13':>8} {'C31':>6} {'C31_inv':>8} {'C32':>6}")
    print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*6} {'-'*8} {'-'*6}")

    resultados = {}
    for nombre, reg, es_B in instancias:
        m_f2,  _ = stats(reg, 'omega_f2')
        m_f13, _ = stats(reg, 'omega_f13')
        c31 = c31_ok(reg)
        inv = inv_ok(reg)
        c32 = c32_ok(reg)
        resultados[nombre] = {'inv': inv, 'c31': c31, 'c32': c32, 'es_B': es_B}
        inv_str = f"{inv}/{n}" if es_B else "  —"
        c31_str = f"{c31}/{n}" if not es_B else "  —"
        print(f"  {nombre:<28} {m_f2:>+8.4f} {m_f13:>+8.4f} {c31_str:>6} {inv_str:>8} {c32:>4}/{n}")

    print()
    print("  TABLA DE INVERSIÓN (criterio clave):")
    print(f"  {'Variante':<14} {'A: C31':>8} {'B: C31_inv':>12} {'Inversión':>12}")
    print(f"  {'-'*14} {'-'*8} {'-'*12} {'-'*12}")
    for v_label, r_A, r_B in [
        ('Aditiva (V1)', resultados['A_V1 (+60°, aditiva)'],  resultados['B_V1 (-60°, aditiva)']),
        ('Escala  (V2)', resultados['A_V2 (+60°, escala)'],   resultados['B_V2 (-60°, escala)']),
    ]:
        inv_lograda = "LOGRADA" if r_B['inv'] == n else f"PARCIAL {r_B['inv']}/{n}"
        print(f"  {v_label:<14} {r_A['c31']:>5}/{n}    {r_B['inv']:>6}/{n}      {inv_lograda}")

    # CSV
    with open('v94_ciclos.csv', 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(['instancia', 'variante', 'ciclo',
                     'omega_f2', 'omega_f13', 'efic_f2', 'efic_f13',
                     'asys_f2', 'asys_f13', 'mejor_ef', 'c31', 'c32'])
        for inst_label, reg, _ in instancias:
            v = '1' if 'aditiva' in inst_label else '2'
            for r in reg:
                wr.writerow([inst_label, v, r['ciclo'],
                             r['omega_f2'], r['omega_f13'],
                             r['efic_f2'], r['efic_f13'],
                             r['asys_f2'], r['asys_f13'], r['mejor_ef'],
                             int(r['c31']), int(r['c32'])])
    print("\n  CSV guardado: v94_ciclos.csv")

    # Gráfico 2x2: columna por variante, fila por métrica
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ciclos_x = list(range(1, N_CICLOS + 1))

    for col, (r_A, r_B, titulo_v) in enumerate([
        (reg_AV1, reg_BV1, 'V1 — Aditiva'),
        (reg_AV2, reg_BV2, 'V2 — Escala K_ORIENT'),
    ]):
        ax = axes[0, col]
        ax.plot(ciclos_x, [r['omega_f2']  for r in r_A], 'o-', color='steelblue', lw=2, label='A: Ω F2')
        ax.plot(ciclos_x, [r['omega_f13'] for r in r_A], 's--', color='steelblue', lw=1.5, alpha=0.7, label='A: Ω F13')
        ax.plot(ciclos_x, [r['omega_f2']  for r in r_B], 'o-', color='firebrick', lw=2, label='B: Ω F2')
        ax.plot(ciclos_x, [r['omega_f13'] for r in r_B], 's--', color='firebrick', lw=1.5, alpha=0.7, label='B: Ω F13')
        ax.axhline(0, color='k', lw=0.8, ls='--')
        ax.set_title(f'Ω_orient — {titulo_v}')
        ax.set_xlabel('Ciclo')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        ax = axes[1, col]
        diff_A = [(r['omega_f2'] or 0) - (r['omega_f13'] or 0) for r in r_A]
        diff_B = [(r['omega_f13'] or 0) - (r['omega_f2'] or 0)  for r in r_B]
        ax.plot(ciclos_x, diff_A, 'o-', color='steelblue', lw=2, label='A: Ω_F2 - Ω_F13')
        ax.plot(ciclos_x, diff_B, 'o-', color='firebrick', lw=2, label='B: Ω_F13 - Ω_F2')
        ax.axhline(0, color='k', lw=0.8, ls='--')
        ax.set_title(f'Diferencia Ω — {titulo_v}')
        ax.set_xlabel('Ciclo')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('VSTCosmos v94 — Cierre W_rec -> act_geom\nControl de inversion: B invierte patron de A?',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig('v94_evolucion.png', dpi=150)
    print("  Grafico guardado: v94_evolucion.png")

    print()
    print("=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)


if __name__ == "__main__":
    main()