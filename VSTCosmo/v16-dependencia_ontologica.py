#!/usr/bin/env python3
"""
VSTCosmo - v16: Dependencia ontológica de Φ respecto a A

Idea central:
- A compite globalmente por persistencia.
- Φ ya no tiene estructura sostenible por sí misma.
- Sin A, Φ se aplana y pierde relieve.
- Con A, Φ puede conservar diferencia local.
- No hay métrica externa de "qué es mejor".
- No hay control teleológico.
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# PARÁMETROS
# ============================================================
DIM_FREQ = 32
DIM_TIME = 100
DT = 0.01
DURACION_SIM = 30.0
N_PASOS = int(DURACION_SIM / DT)

# Campo Φ
GANANCIA_ENTRADA = 0.02
DIFUSION_PHI = 0.08
DECAIMIENTO_PHI = 0.06          # NUEVO: Φ pierde estructura si no hay A
GANANCIA_TARGET = 0.12          # target móvil por entrada, más suave
GANANCIA_SOSTENIMIENTO = 0.18   # NUEVO: A permite a Φ conservar estructura

# Atención A
REFUERZO_A = 0.15
INHIBICION_A = 0.2
DIFUSION_A = 0.08
FUERZA_ACOPLAMIENTO_A = 0.08    # A siente Φ, pero sin colapsar al promedio

# Competencia global en A
LIMITE_ATENCION = DIM_FREQ * DIM_TIME * 0.35
INHIB_GLOBAL = 0.5

# Límites
LIMITE_MIN = 0.0
LIMITE_MAX = 1.0

# ============================================================
# UTILIDADES
# ============================================================
def cargar_audio(ruta):
    sr, data = wav.read(ruta)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)

    if data.ndim == 2:
        data = data.mean(axis=1)

    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val

    return sr, data

def inicializar_campo(semilla=None):
    if semilla is not None:
        np.random.seed(semilla)
    return np.random.rand(DIM_FREQ, DIM_TIME) * 0.2 + 0.4

def inicializar_atencion():
    return np.ones((DIM_FREQ, DIM_TIME), dtype=np.float32) * 0.1

def vecinos(X):
    return (
        np.roll(X, 1, axis=0) +
        np.roll(X, -1, axis=0) +
        np.roll(X, 1, axis=1) +
        np.roll(X, -1, axis=1)
    ) / 4.0

# ============================================================
# CAMPO Φ
# ============================================================
def construir_target(muestra):
    """
    La entrada no se suma como 'información'.
    Deforma el paisaje local de Φ.
    """
    m = (muestra + 1.0) / 2.0
    m = np.clip(m, 0.0, 1.0)

    target_banda = 0.3 + 0.4 * m
    banda = int(m * (DIM_FREQ - 1))

    target = np.ones((DIM_FREQ, DIM_TIME), dtype=np.float32) * 0.5

    for i in range(DIM_FREQ):
        distancia = min(abs(i - banda), DIM_FREQ - abs(i - banda))
        influencia = np.exp(-(distancia ** 2) / 10.0)
        target[i, :] = target_banda * influencia + 0.5 * (1.0 - influencia)

    return target

def actualizar_campo(Phi, A, muestra):
    """
    Φ no se sostiene solo.
    - La entrada deforma el target.
    - La difusión mantiene continuidad.
    - El decaimiento aplana el campo.
    - A permite conservar diferencia local.
    """
    target = construir_target(muestra)
    v = vecinos(Phi)

    # 1. Tendencia del entorno a deformar el campo
    arrastre_entrada = GANANCIA_TARGET * (target - Phi)

    # 2. Difusión local
    difusion = DIFUSION_PHI * (v - Phi)

    # 3. Desorganización base: Φ pierde relieve si nadie lo sostiene
    #    Lo llevamos hacia el entorno local medio (no a una constante global).
    decaimiento = -DECAIMIENTO_PHI * (Phi - v)

    # 4. Sostenimiento por A:
    #    donde A es alta, Φ conserva más su diferencia respecto del entorno local
    #    No impone valor, solo evita que el relieve colapse.
    sostenimiento = GANANCIA_SOSTENIMIENTO * A * (Phi - v)

    # 5. Pequeña inyección directa de la muestra
    entrada_directa = GANANCIA_ENTRADA * muestra

    dPhi = arrastre_entrada + difusion + decaimiento + sostenimiento
    Phi = Phi + DT * dPhi + entrada_directa

    return np.clip(Phi, LIMITE_MIN, LIMITE_MAX)

# ============================================================
# ATENCIÓN A
# ============================================================
def actualizar_atencion(A, Phi):
    """
    A compite por persistencia.
    No tiene basal externo.
    No colapsa linealmente al promedio.
    """
    vA = vecinos(A)

    auto = REFUERZO_A * A * (1.0 - A)
    inhib_local = -INHIBICION_A * vA
    difusion = DIFUSION_A * (vA - A)

    # A siente el relieve local de Φ, no su media global
    vPhi = vecinos(Phi)
    relieve_local = np.abs(Phi - vPhi)

    max_relieve = np.max(relieve_local)
    if max_relieve > 0:
        relieve_local = relieve_local / max_relieve

    acoplamiento_local = FUERZA_ACOPLAMIENTO_A * (relieve_local - A)

    dA = auto + inhib_local + difusion + acoplamiento_local

    # Competencia global: no toda A puede persistir a la vez
    atencion_total = np.sum(A)
    if atencion_total > LIMITE_ATENCION:
        exceso = (atencion_total - LIMITE_ATENCION) / LIMITE_ATENCION
        dA += -INHIB_GLOBAL * exceso * A

    # Ruido mínimo
    dA += np.random.randn(*A.shape) * 0.001

    A = A + DT * dA
    return np.clip(A, LIMITE_MIN, LIMITE_MAX)

# ============================================================
# SIMULACIÓN
# ============================================================
def simular(audio, sr, nombre, semilla=None, num_pasos=N_PASOS):
    print(f"    {nombre}...", end=" ", flush=True)

    Phi = inicializar_campo(semilla)
    A = inicializar_atencion()

    n_muestras = int(num_pasos * DT * sr)
    if len(audio) > n_muestras:
        audio = audio[:n_muestras]

    for paso in range(num_pasos):
        t = paso * DT
        idx = int(t * sr)
        idx = min(idx, len(audio) - 1) if len(audio) > 0 else 0
        muestra = audio[idx] if len(audio) > 0 else 0.0

        A = actualizar_atencion(A, Phi)
        Phi = actualizar_campo(Phi, A, muestra)

    rango_phi = float(np.max(Phi) - np.min(Phi))
    rango_a = float(np.max(A) - np.min(A))
    media_a = float(np.mean(A))
    total_a = float(np.sum(A))

    print(f"rango Φ={rango_phi:.3f}, rango A={rango_a:.4f}, media A={media_a:.4f}, total A={total_a:.1f}")
    return rango_phi, rango_a, media_a, total_a

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("VSTCosmo - v16: Φ depende de A para sostener estructura")
    print("=" * 70)

    print("\n[1] Cargando archivo...")
    sr, voz_viento = cargar_audio("Voz+Viento_1.wav")
    print(f"    Voz+Viento_1.wav: {len(voz_viento)/sr:.1f}s")

    semillas = [42, 123, 987, 1, 2, 3, 100, 200, 300, 400]

    print("\n[2] Ejecutando 10 simulaciones...")
    print("-" * 70)

    resultados_phi = []
    resultados_a = []
    resultados_media_a = []
    resultados_total_a = []

    for i, semilla in enumerate(semillas):
        rphi, ra, media_a, total_a = simular(
            voz_viento, sr, f"Run_{i}", semilla=semilla
        )
        resultados_phi.append(rphi)
        resultados_a.append(ra)
        resultados_media_a.append(media_a)
        resultados_total_a.append(total_a)

    print("-" * 70)

    media_phi = np.mean(resultados_phi)
    std_phi = np.std(resultados_phi)
    media_a = np.mean(resultados_a)
    std_a = np.std(resultados_a)
    media_total = np.mean(resultados_total_a)
    std_total = np.std(resultados_total_a)

    print("\n[3] Análisis estadístico")
    print("=" * 70)
    print(f"  rango Φ: media={media_phi:.3f}, std={std_phi:.4f}")
    print(f"  rango A: media={media_a:.4f}, std={std_a:.4f}")
    print(f"  total A: media={media_total:.1f}, std={std_total:.1f}")

    print("\n[4] Interpretación")
    print("=" * 70)
    if std_phi < 0.05:
        print("  ✓ Φ muestra estabilidad consistente.")
    else:
        print("  ✗ Φ sigue cambiando mucho entre ejecuciones.")

    if std_a < 0.02:
        print("  ✓ A muestra régimen consistente.")
    else:
        print("  ✗ A sigue sin régimen consistente.")

    if std_phi < 0.05 and std_a < 0.02:
        print("\n  ★ Posible régimen co-sostenido Φ↔A.")
    else:
        print("\n  → Aún no aparece co-sostenimiento fuerte.")

    print("\n[5] Generando visualización...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(range(1, 11), resultados_phi, "bo-", markersize=8)
    axes[0, 0].axhline(y=media_phi, color="r", linestyle="--")
    axes[0, 0].set_title("rango Φ")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(range(1, 11), resultados_a, "go-", markersize=8)
    axes[0, 1].axhline(y=media_a, color="r", linestyle="--")
    axes[0, 1].set_title("rango A")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(range(1, 11), resultados_total_a, "mo-", markersize=8)
    axes[1, 0].axhline(y=LIMITE_ATENCION, color="r", linestyle="--", label="límite")
    axes[1, 0].set_title("total A")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].scatter(resultados_phi, resultados_a, s=100, alpha=0.7)
    axes[1, 1].set_xlabel("rango Φ")
    axes[1, 1].set_ylabel("rango A")
    axes[1, 1].set_title("Relación Φ ↔ A")
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("VSTCosmo v16 - Co-sostenimiento Φ↔A", fontsize=14)
    plt.tight_layout()
    plt.savefig("v16_cosostenimiento.png", dpi=150)
    print("    Gráfico guardado: v16_cosostenimiento.png")

    print("\n" + "=" * 70)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 70)

if __name__ == "__main__":
    main()