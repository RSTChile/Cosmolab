#!/usr/bin/env python3
"""
VSTCosmos v80 — Memorias duales por separación espectral

Principio canónico:
- Dos memorias con escalas temporales derivadas de las frecuencias naturales
- Operan en bandas espectrales distintas del mismo campo
- Ortogonalidad estructural: no es diseño, es física del campo
- LF definida por error de memoria reciente (contexto)
- Identidad preservada en memoria profunda (banda baja)
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import csv
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PARÁMETROS BASE (definen la física del campo)
# ============================================================
DIM_INTERNA = 32
DIM_AUDITIVA = 32
DIM_TOTAL = DIM_INTERNA + DIM_AUDITIVA

DIM_TIME = 100
DT = 0.01

DURACION_ENTRENO = 30.0
DURACION_FASE = 20.0
DURACION_REACOPLAMIENTO = 30.0

N_PASOS_ENTRENO = int(DURACION_ENTRENO / DT)
N_PASOS_FASE = int(DURACION_FASE / DT)
N_PASOS_REACOP = int(DURACION_REACOPLAMIENTO / DT)

# Parámetros de dinámica
DIFUSION_BASE = 0.15
GANANCIA_REACCION = 0.05

OMEGA_MIN = 0.05
OMEGA_MAX = 0.50
AMORT_MIN = 0.01
AMORT_MAX = 0.08
PHI_EQUILIBRIO = 0.5

# Límites
LIMITE_MIN = 0.0
LIMITE_MAX = 1.0
W_MAX = 1.0

# Parámetros de FFT
VENTANA_FFT_MS = 25
HOP_FFT_MS = 10
F_MIN = 80
F_MAX = 8000

# ============================================================
# ESCALAS TEMPORALES DERIVADAS DE FRECUENCIAS NATURALES
# ============================================================
OMEGA_MEDIA = (OMEGA_MIN + OMEGA_MAX) / 2.0  # = 0.275

# Memoria profunda — escala temporal lenta (identidad)
ETA_PROFUNDA = OMEGA_MIN * DT      # = 0.0005 — aprende muy lento
TAU_PROFUNDA = OMEGA_MIN           # = 0.05   — olvida muy lento (vida media ~20s)

# Memoria reciente — escala temporal rápida (contexto)
ETA_RECIENTE = OMEGA_MAX * DT      # = 0.005  — aprende rápido
TAU_RECIENTE = OMEGA_MAX           # = 0.50   — olvida rápido (vida media ~2s)

# Separación espectral
BANDA_BAJA = slice(0, DIM_INTERNA // 2)   # modos 0-15: lentos, identidad
BANDA_ALTA = slice(DIM_INTERNA // 2, None) # modos 16-31: rápidos, contexto

# Error de equilibrio (derivado de difusión)
ERROR_EQUILIBRIO = DIFUSION_BASE ** 2  # = 0.0225

# ============================================================
# FUNCIONES BASE
# ============================================================
def cargar_audio(ruta, duracion):
    if "Tono puro" in ruta:
        sr = 48000
        t = np.arange(int(sr * duracion)) / sr
        return sr, 0.5 * np.sin(2 * np.pi * 440 * t)
    elif "Ruido blanco" in ruta:
        sr = 48000
        return sr, np.random.normal(0, 0.5, int(sr * duracion))
    else:
        try:
            sr, data = wav.read(ruta)
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            if data.ndim == 2:
                data = data.mean(axis=1)
            max_val = np.max(np.abs(data))
            if max_val > 0:
                data = data / max_val
            muestras_necesarias = int(sr * duracion)
            if len(data) < muestras_necesarias:
                data = np.pad(data, (0, muestras_necesarias - len(data)))
            else:
                data = data[:muestras_necesarias]
            return sr, data
        except FileNotFoundError:
            print(f"  [ADVERTENCIA] {ruta} no encontrado, usando tono 440Hz")
            sr = 48000
            t = np.arange(int(sr * duracion)) / sr
            return sr, 0.5 * np.sin(2 * np.pi * 440 * t)

def inicializar_campo_total():
    np.random.seed(42)
    return np.random.rand(DIM_TOTAL, DIM_TIME) * 0.2 + 0.4

def inicializar_memoria_profunda():
    return np.zeros((DIM_INTERNA // 2, DIM_AUDITIVA // 2))

def inicializar_memoria_reciente():
    return np.zeros((DIM_INTERNA // 2, DIM_AUDITIVA // 2))

def inicializar_historia_interna():
    return np.zeros((DIM_INTERNA, DIM_TIME))

def vecinos(X):
    return (np.roll(X, 1, axis=0) + np.roll(X, -1, axis=0) +
            np.roll(X, 1, axis=1) + np.roll(X, -1, axis=1)) / 4.0

def preparar_objetivo_audio(audio, sr, idx_ventana, ventana_muestras, hop_muestras,
                            dim_auditiva, DIM_TIME):
    inicio = idx_ventana * hop_muestras
    if inicio + ventana_muestras > len(audio):
        return np.zeros((dim_auditiva, DIM_TIME))
    
    fragmento = audio[inicio:inicio + ventana_muestras]
    ventana_hann = np.hanning(len(fragmento))
    fragmento = fragmento * ventana_hann
    
    fft = np.fft.rfft(fragmento)
    potencia = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(len(fragmento), 1/sr)
    
    bandas = np.logspace(np.log10(F_MIN), np.log10(F_MAX), dim_auditiva + 1)
    objetivo = np.zeros(dim_auditiva)
    for b in range(dim_auditiva):
        mask = (freqs >= bandas[b]) & (freqs < bandas[b+1])
        if np.any(mask):
            objetivo[b] = np.mean(potencia[mask])
    
    max_val = np.max(objetivo)
    if max_val > 0:
        objetivo = objetivo / max_val
    
    return objetivo.reshape(-1, 1) * np.ones((1, DIM_TIME))

def calcular_frecuencias_naturales(dim_total, dim_interna):
    bandas = np.arange(dim_total)
    t = np.log1p(bandas) / np.log1p(dim_total - 1)
    omega = OMEGA_MIN + (OMEGA_MAX - OMEGA_MIN) * t
    amort = AMORT_MIN + (AMORT_MAX - AMORT_MIN) * t
    return omega.reshape(-1, 1), amort.reshape(-1, 1)

def _perfil_espectral(region, dim, DIM_TIME):
    perfil = np.zeros(DIM_TIME // 2)
    for banda in range(dim):
        serie = region[banda, :]
        serie = serie - np.mean(serie)
        fft = np.fft.rfft(serie)
        potencia = np.abs(fft) ** 2
        perfil += potencia[:DIM_TIME // 2]
    return perfil / dim

def calcular_gradiente_espectral_diferencial(Phi_total, dim_interna, DIM_TIME):
    region_int = Phi_total[:dim_interna, :]
    region_aud = Phi_total[dim_interna:, :]
    
    perfil_int = _perfil_espectral(region_int, dim_interna, DIM_TIME)
    perfil_aud = _perfil_espectral(region_aud, dim_interna, DIM_TIME)
    
    ged = np.mean(np.abs(perfil_int - perfil_aud))
    return float(ged)

# ============================================================
# NÚCLEO: MEMORIAS DUALES POR BANDA ESPECTRAL
# ============================================================
def aplicar_plasticidad_memoria_profunda(W_prof, region_int, region_aud, dt):
    """Aprende en banda baja, escala temporal lenta"""
    reg_int_baja = region_int[BANDA_BAJA, :]
    reg_aud_baja = region_aud[BANDA_BAJA, :]
    
    correlacion = (reg_int_baja @ reg_aud_baja.T) / DIM_TIME
    dW = ETA_PROFUNDA * correlacion - TAU_PROFUNDA * W_prof
    W_nueva = np.clip(W_prof + dW * dt, -W_MAX, W_MAX)
    
    # Contribución a M_hebb
    prediccion = W_nueva @ reg_aud_baja
    M_hebb = np.zeros((DIM_INTERNA, DIM_TIME))
    M_hebb[BANDA_BAJA, :] = prediccion - reg_int_baja
    
    return W_nueva, M_hebb

def aplicar_plasticidad_memoria_reciente(W_rec, region_int, region_aud, dt):
    """Aprende en banda alta, escala temporal rápida"""
    reg_int_alta = region_int[BANDA_ALTA, :]
    reg_aud_alta = region_aud[BANDA_ALTA, :]
    
    correlacion = (reg_int_alta @ reg_aud_alta.T) / DIM_TIME
    dW = ETA_RECIENTE * correlacion - TAU_RECIENTE * W_rec
    W_nueva = np.clip(W_rec + dW * dt, -W_MAX, W_MAX)
    
    # Contribución a M_hebb
    prediccion = W_nueva @ reg_aud_alta
    M_hebb = np.zeros((DIM_INTERNA, DIM_TIME))
    M_hebb[BANDA_ALTA, :] = prediccion - reg_int_alta
    
    return W_nueva, M_hebb

def calcular_error_memoria_reciente(W_rec, region_int, region_aud):
    """Error predictivo de la memoria reciente — define LF"""
    reg_int_alta = region_int[BANDA_ALTA, :]
    reg_aud_alta = region_aud[BANDA_ALTA, :]
    
    prediccion = W_rec @ reg_aud_alta
    error = np.mean((prediccion - reg_int_alta) ** 2)
    
    return error

def atractor_dual(Phi_int_historia, region_int):
    """Atractor único (identidad) — opera en todo el campo"""
    return 0.05 * (Phi_int_historia - region_int)

# ============================================================
# ACTUALIZACIÓN PRINCIPAL
# ============================================================
def actualizar_campo_con_memorias_duales(
        Phi_total, Phi_vel_total, W_prof, W_rec,
        Phi_int_historia,
        objetivo_audio, alpha,
        omega_natural, amort_natural,
        dt, entrenando,
        DIM_TIME):
    
    # Dinámica base
    promedio_local = vecinos(Phi_total)
    difusion = DIFUSION_BASE * (promedio_local - Phi_total)
    desviacion = Phi_total - promedio_local
    reaccion = GANANCIA_REACCION * desviacion * (1 - desviacion**2)
    term_osc = (-omega_natural**2 * (Phi_total - PHI_EQUILIBRIO)
                - amort_natural * Phi_vel_total)
    
    region_int = Phi_total[:DIM_INTERNA, :]
    region_aud = Phi_total[DIM_INTERNA:, :]
    
    # Plasticidad dual
    W_prof_nueva, M_prof = aplicar_plasticidad_memoria_profunda(
        W_prof, region_int, region_aud, dt
    )
    W_rec_nueva, M_rec = aplicar_plasticidad_memoria_reciente(
        W_rec, region_int, region_aud, dt
    )
    
    # Atractor
    M_atractor = atractor_dual(Phi_int_historia, region_int)
    
    # Campo total
    M_campo = np.zeros_like(Phi_total)
    M_campo[:DIM_INTERNA, :] = M_prof + M_rec + M_atractor
    
    # Calcular error de memoria reciente (para diagnóstico LF)
    error_rec = calcular_error_memoria_reciente(W_rec_nueva, region_int, region_aud)
    lf_activa = error_rec > ERROR_EQUILIBRIO
    
    # Actualización
    dPhi_vel = term_osc + reaccion + difusion + M_campo
    Phi_vel_nueva = Phi_vel_total + dt * dPhi_vel
    Phi_nueva = Phi_total + dt * Phi_vel_nueva
    
    if alpha > 0:
        region_auditiva_nueva = Phi_nueva[DIM_INTERNA:, :]
        region_auditiva_nueva = (1 - alpha) * region_auditiva_nueva + alpha * objetivo_audio
        Phi_nueva[DIM_INTERNA:, :] = region_auditiva_nueva
    
    # Prevenir colapso
    varianza_campo = np.var(Phi_nueva[:DIM_INTERNA, :])
    if varianza_campo < 1e-6:
        ruido_minimo = np.random.normal(0, DIFUSION_BASE * 0.1, Phi_nueva[:DIM_INTERNA, :].shape)
        Phi_nueva[:DIM_INTERNA, :] += ruido_minimo
    
    # Actualizar historia interna
    Phi_int_historia_nueva = (1 - 0.05) * Phi_int_historia + 0.05 * region_int
    
    return (np.clip(Phi_nueva, LIMITE_MIN, LIMITE_MAX),
            np.clip(Phi_vel_nueva, -5.0, 5.0),
            W_prof_nueva, W_rec_nueva, Phi_int_historia_nueva,
            lf_activa, error_rec)

# ============================================================
# SIMULACIÓN
# ============================================================
def simular_fase(Phi_total, Phi_vel_total, W_prof, W_rec,
                 Phi_int_historia,
                 estimulo, alpha, duracion, fase_nombre,
                 omega_natural, amort_natural, DIM_TIME):
    sr, audio = cargar_audio(estimulo, duracion=duracion)
    ventana_muestras = int(sr * VENTANA_FFT_MS / 1000)
    hop_muestras = int(sr * HOP_FFT_MS / 1000)
    n_pasos = int(duracion / DT)
    
    ged_hist = []
    lf_hist = []
    error_rec_hist = []
    w_prof_norma_hist = []
    w_rec_norma_hist = []
    
    for idx in range(n_pasos):
        objetivo = preparar_objetivo_audio(audio, sr, idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
        
        ged = calcular_gradiente_espectral_diferencial(Phi_total, DIM_INTERNA, DIM_TIME)
        
        (Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia,
         lf_activa, error_rec) = actualizar_campo_con_memorias_duales(
            Phi_total, Phi_vel_total, W_prof, W_rec,
            Phi_int_historia,
            objetivo, alpha,
            omega_natural, amort_natural,
            DT, entrenando=False,
            DIM_TIME=DIM_TIME
        )
        
        ged_hist.append(ged)
        lf_hist.append(1 if lf_activa else 0)
        error_rec_hist.append(error_rec)
        w_prof_norma_hist.append(np.mean(np.abs(W_prof)))
        w_rec_norma_hist.append(np.mean(np.abs(W_rec)))
    
    return {
        'ged_mean': np.mean(ged_hist),
        'lf_pct': 100 * np.mean(lf_hist),
        'w_prof_norma': np.mean(w_prof_norma_hist),
        'w_rec_norma': np.mean(w_rec_norma_hist),
        'error_rec_mean': np.mean(error_rec_hist),
        'phi_total': Phi_total.copy(),
        'w_prof': W_prof.copy(),
        'w_rec': W_rec.copy(),
        'phi_int_historia': Phi_int_historia.copy()
    }

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 100)
    print("VSTCosmos v80 — Memorias duales por separación espectral")
    print("")
    print("Principio canónico:")
    print(f"  - Memoria profunda (identidad): banda baja, escala lenta (τ≈{TAU_PROFUNDA:.2f}s)")
    print(f"  - Memoria reciente (contexto): banda alta, escala rápida (τ≈{TAU_RECIENTE:.2f}s)")
    print(f"  - LF definida por error de memoria reciente > {ERROR_EQUILIBRIO:.4f}")
    print("=" * 100)

    # Inicialización
    Phi_total = inicializar_campo_total()
    Phi_vel_total = np.zeros_like(Phi_total)
    W_prof = inicializar_memoria_profunda()
    W_rec = inicializar_memoria_reciente()
    Phi_int_historia = inicializar_historia_interna()
    
    omega_natural, amort_natural = calcular_frecuencias_naturales(DIM_TOTAL, DIM_INTERNA)
    
    # Entrenamiento
    print("\n[Fase 1] Entrenamiento (voz, alpha=0.05, 30s)")
    sr_voz, audio_voz = cargar_audio("Voz_Estudio.wav", duracion=DURACION_ENTRENO)
    ventana_muestras = int(sr_voz * VENTANA_FFT_MS / 1000)
    hop_muestras = int(sr_voz * HOP_FFT_MS / 1000)
    
    for idx in range(N_PASOS_ENTRENO):
        objetivo = preparar_objetivo_audio(audio_voz, sr_voz, idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
        
        (Phi_total, Phi_vel_total, W_prof, W_rec, Phi_int_historia,
         _, _) = actualizar_campo_con_memorias_duales(
            Phi_total, Phi_vel_total, W_prof, W_rec,
            Phi_int_historia,
            objetivo, alpha=0.05,
            omega_natural=omega_natural, amort_natural=amort_natural,
            dt=DT, entrenando=True,
            DIM_TIME=DIM_TIME
        )
    
    print(f"  W_prof (identidad) tras entreno: {np.mean(np.abs(W_prof)):.4f}")
    print(f"  W_rec (contexto) tras entreno: {np.mean(np.abs(W_rec)):.4f}")
    
    # Fases de test
    fases = [
        ("Fase 2", "Voz_Estudio.wav", "Dominio (voz)", DURACION_FASE),
        ("Fase 3", "Brandemburgo.wav", "No entrenado (música)", DURACION_FASE),
        ("Fase 4", "Tono puro", "No entrenado (tono)", DURACION_FASE),
        ("Fase 5", "Voz+Viento_1.wav", "Degradado (voz+viento)", DURACION_FASE),
        ("Fase 6", "Ruido blanco", "Perturbación basal", DURACION_FASE),
        ("Fase 7", "Voz_Estudio.wav", "Re-acoplamiento (voz)", DURACION_REACOPLAMIENTO)
    ]
    
    resultados = []
    
    for fase_id, estimulo, desc, duracion in fases:
        print(f"\n[{fase_id}] {desc}")
        res = simular_fase(
            Phi_total, Phi_vel_total, W_prof, W_rec,
            Phi_int_historia,
            estimulo, 0.0, duracion, fase_id,
            omega_natural, amort_natural, DIM_TIME
        )
        resultados.append(res)
        
        Phi_total = res['phi_total']
        W_prof = res['w_prof']
        W_rec = res['w_rec']
        Phi_int_historia = res['phi_int_historia']
        
        print(f"    GED: {res['ged_mean']:.6f} | LF: {res['lf_pct']:.1f}%")
        print(f"    W_prof: {res['w_prof_norma']:.4f} | W_rec: {res['w_rec_norma']:.4f}")
        print(f"    Error W_rec: {res['error_rec_mean']:.6f}")
    
    # Diagnóstico
    print("\n" + "=" * 100)
    print("DIAGNÓSTICO — MEMORIAS DUALES")
    print("=" * 100)
    
    w_prof_f2 = resultados[0]['w_prof_norma']
    w_prof_f7 = resultados[5]['w_prof_norma']
    w_rec_f2 = resultados[0]['w_rec_norma']
    w_rec_f7 = resultados[5]['w_rec_norma']
    lf_f7 = resultados[5]['lf_pct']
    error_f7 = resultados[5]['error_rec_mean']
    
    print(f"\n  Memoria profunda (identidad):")
    print(f"    W_prof Fase 2: {w_prof_f2:.4f}")
    print(f"    W_prof Fase 7: {w_prof_f7:.4f}")
    print(f"    Crecimiento: {'✅' if w_prof_f7 > w_prof_f2 * 1.1 else '❌'} ({w_prof_f7/w_prof_f2:.2f}x)")
    
    print(f"\n  Memoria reciente (contexto):")
    print(f"    W_rec Fase 2: {w_rec_f2:.4f}")
    print(f"    W_rec Fase 7: {w_rec_f7:.4f}")
    print(f"    Crecimiento: {w_rec_f7/w_rec_f2:.2f}x")
    
    print(f"\n  Estado del campo en Fase 7:")
    print(f"    LF activa: {lf_f7:.1f}%")
    print(f"    Error W_rec: {error_f7:.6f} (equilibrio: {ERROR_EQUILIBRIO:.4f})")
    print(f"    {'✅' if error_f7 < ERROR_EQUILIBRIO else '❌'} Campo en dominio")
    
    print("\n  VEREDICTO:")
    if error_f7 < ERROR_EQUILIBRIO and lf_f7 < 30:
        print("  ✅ MEMORIAS DUALES FUNCIONALES")
        print("     Error W_rec bajo → campo en dominio.")
        print("     Identidad preservada en banda baja.")
        print("     Contexto aprendido en banda alta sin sobrescribir identidad.")
    elif w_prof_f7 > w_prof_f2 * 1.1:
        print("  ⚠️ IDENTIDAD AMPLIADA pero campo aún fuera de dominio.")
    else:
        print("  ❌ SEPARACIÓN ESPECTRAL NO FUNCIONÓ")
        print("     Verificar parámetros de bandas y escalas temporales.")
    
    # Guardar resultados
    with open('v80_memorias_duales.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['fase', 'ged_mean', 'lf_pct', 'w_prof_norma', 'w_rec_norma', 'error_rec'])
        for i, (fase, res) in enumerate(zip(fases, resultados)):
            writer.writerow([fase[0], res['ged_mean'], res['lf_pct'],
                            res['w_prof_norma'], res['w_rec_norma'],
                            res['error_rec_mean']])
    
    print("\n  CSV guardado: v80_memorias_duales.csv")
    
    # Gráfico
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    nombres = ['Voz', 'Música', 'Tono', 'Voz+Viento', 'Ruido', 'Reacop']
    
    axes[0,0].bar(nombres, [r['ged_mean'] for r in resultados])
    axes[0,0].axhline(y=ERROR_EQUILIBRIO**0.5, color='r', linestyle='--', label='Umbral referencia')
    axes[0,0].set_title('GED')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].bar(nombres, [r['lf_pct'] for r in resultados])
    axes[0,1].set_title('LF activa (%)')
    axes[0,1].grid(True, alpha=0.3)
    
    axes[0,2].bar(nombres, [r['w_prof_norma'] for r in resultados], label='W_prof (identidad)')
    axes[0,2].bar(nombres, [r['w_rec_norma'] for r in resultados], alpha=0.5, label='W_rec (contexto)')
    axes[0,2].set_title('Norma de memorias')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    axes[1,0].bar(nombres, [r['error_rec_mean'] for r in resultados])
    axes[1,0].axhline(y=ERROR_EQUILIBRIO, color='r', linestyle='--', label=f'Equilibrio ({ERROR_EQUILIBRIO:.4f})')
    axes[1,0].set_title('Error de memoria reciente')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].bar(nombres, [r['w_prof_norma'] for r in resultados])
    axes[1,1].set_title('W_prof (identidad, banda baja)')
    axes[1,1].grid(True, alpha=0.3)
    
    axes[1,2].bar(nombres, [r['w_rec_norma'] for r in resultados])
    axes[1,2].set_title('W_rec (contexto, banda alta)')
    axes[1,2].grid(True, alpha=0.3)
    
    plt.suptitle('VSTCosmos v80 — Memorias duales por separación espectral', fontsize=14)
    plt.tight_layout()
    plt.savefig('v80_memorias_duales.png', dpi=150)
    print("  Gráfico guardado: v80_memorias_duales.png")
    
    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)


if __name__ == "__main__":
    main()