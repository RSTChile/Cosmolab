"""
CS072-II -- MOTOR SEPARADO, sin sustrato previo (MANIFIESTO_FOLD_CS072.md ADDENDUM + ADJUDICACION_CS072_II_
transicion_sin_sustrato_CS.md + PROPUESTA_CODEX_CS072_II_transicion_sin_sustrato_PARA_CS.md).
==============================================================================
NO parchea cs072_v6/v7/v8_nucleo.py -- son Track I (motor condicionado a GR.aleatorio), CERRADOS, conservados
intactos como evidencia histórica. Este es el motor CS072-II: construido para que la pregunta "¿emerge el
primer 'al lado de' desde una singularidad permutacionalmente simétrica, sin medida previa?" se responda sin
que la respuesta pueda venir de aristas, índices, umbrales o ruido escondido.

ESTADO: T (N,) temperatura; W (N,N) simétrica, diagonal 0, w_ij(0)=w0 IGUAL para TODO par (matriz de
afinidad UNIFORME -- no hay grafo, no hay "vecino" privilegiado). ε rompe SÓLO T (invariante §3.1: el
conjunto de focos es CANÓNICO -- los primeros n_focos índices -- tratado como gauge, NUNCA sorteado).

ESTA ES LA VERSIÓN DETERMINISTA (II-DET): CERO llamadas a RNG en todo el paso. Es el CONTROL del no-go de
simetría (Codex + CS, verificado con código): una dinámica determinista y equivariante sobre un estado
exactamente simétrico salvo ε NO PUEDE fabricar más que las clases que ε induce (con 1 foco: frío + tibio,
y TODOS los tibios deben permanecer IDÉNTICOS entre sí para siempre, hasta precisión de punto flotante).
Si esta implementación viola eso (como el motor ingenuo que CS probó, que amplificó ruido 1e-15 -> O(1) en
40 pasos), es un BUG de implementación, no un hallazgo -- por eso existe la Puerta S (cs072_ii_puerta_s.py):
detecta exactamente esa amplificación antes de creer cualquier resultado.

Los 4 mecanismos de NÚCLEO-II (DISEÑO §8 del manifiesto post-Codex), todos leyendo (T,W) AL INICIO del paso
y combinados en una sola actualización simultánea (invariante §3.8, G-PROCESO-NO-SUCESION):
1. ROCE PONDERADO (flujo anti-difusivo, generalización continua de v6): el más frío cede al más tibio,
   proporcional a w_ij (contacto continuo), normalizado por FORTALEZA (Σw, invariante §3.4: nunca por
   conteo N). Piso T>=0 exacto (conservación).
2. GRAVEDAD (#2, inestabilidad): "modifica afinidades ya potenciales; no crea pares ni sortea candidatos"
   (Codex §5) -- refuerzo aditivo entre pares fríos, proporcional al peso pairwise PROMEDIO ACTUAL
   (w0_efectivo = fortaleza_media/(N-1)) para ser GAUGE-INVARIANTE a la escala de W (invariante §3.3).
3. MEMORIA (mecanismo de origen, CS071): refuerzo continuo por roce real, decaimiento -- SIN corte binario.
4. EXPANSIÓN CONTINUA (R-EXPANSIÓN, fórmula adjudicada): W_ij <- W_ij * exp[-p_t*(s_i+s_j)/(2*s_bar)],
   ciega a longitud (sólo lee fortaleza s=Σw, propiedad de la propia W). REDUCE capacidad total -- NO
   redistribuye/renormaliza (ruling del director: "no hay de dónde llenarse, no hay más que lo que hubo").

Codea/ejecuta: CC. Diseño/ruling: CS + director + Codex (revisor propositivo).
"""
from __future__ import annotations
import numpy as np

W0_DEFAULT = 1.0
TASA_FLUJO_DEFAULT = 0.15
GRAV_RATE_DEFAULT = 0.30
REFUERZO_DEFAULT = 0.04
DECAY_DEFAULT = 0.99
P_EXP_DEFAULT = 0.02


def estado_inicial(N, n_focos, delta, w0=W0_DEFAULT):
    """Invariantes §3(1-2): T_i=1 salvo los primeros n_focos indices (CANÓNICO, no sorteado, gauge) a
    1-delta. W_ii=0, W_ij=w0 para TODO i!=j. CERO RNG."""
    T = np.ones(N, dtype=np.float64)
    T[:n_focos] = 1.0 - delta
    W = np.full((N, N), float(w0), dtype=np.float64)
    np.fill_diagonal(W, 0.0)
    return T, W


def paso_ii_det(T, W, tasa_flujo=TASA_FLUJO_DEFAULT, grav_rate=GRAV_RATE_DEFAULT,
                refuerzo=REFUERZO_DEFAULT, decay=DECAY_DEFAULT, p_exp=P_EXP_DEFAULT):
    """Un paso, íntegramente determinista (CERO RNG -- auditable: no hay una sola llamada a np.random en
    esta función). Todos los sub-términos (roce, gravedad, refuerzo, expansión) se calculan desde el MISMO
    (T,W) de entrada -- no hay cascada. Devuelve (T_nuevo, W_nuevo, roce_pair) -- roce_pair se expone para
    diagnóstico/memoria externa si hiciera falta, no se re-lee dentro del paso."""
    N = T.shape[0]

    # --- fortaleza (grado pesado) del estado DE ENTRADA -- se usa para normalizar TODO (invariantes 3,4) ---
    s = W.sum(axis=1)
    s_safe = np.maximum(s, 1e-12)
    s_bar = max(float(s.mean()), 1e-12)
    w0_efectivo = s_bar / max(N - 1, 1)     # peso pairwise promedio actual -- gauge de escala (invariante 3)

    # --- 1. ROCE PONDERADO: el frío cede al tibio, normalizado por fortaleza, piso T>=0 exacto ---
    D = T[None, :] - T[:, None]                       # D[i,j] = T_j - T_i
    contraste = np.clip(D, 0.0, None)                  # >0 sólo si j es más tibio que i (i es el frío)
    raw = tasa_flujo * W * contraste / s_safe[:, None]
    raw_out = raw.sum(axis=1)
    escala = np.where(raw_out > 1e-12, np.minimum(1.0, T / np.maximum(raw_out, 1e-12)), 1.0)
    sent = raw * escala[:, None]                       # sent[i,j]: i (frío) cede a j (tibio)
    T_nuevo = T - sent.sum(axis=1) + sent.sum(axis=0)
    T_nuevo = np.clip(T_nuevo, 0.0, None)
    roce_pair = sent + sent.T                          # simétrico: roce real transitado en el par (i,j)

    # --- 2. GRAVEDAD: refuerzo aditivo entre fríos, escalado por w0_efectivo (gauge-invariante) ---
    cold = np.clip(1.0 - T, 0.0, None)
    dW_grav = grav_rate * np.outer(cold, cold) * w0_efectivo
    np.fill_diagonal(dW_grav, 0.0)

    # --- 3. MEMORIA: refuerzo continuo por roce real + decaimiento, SIN corte binario ---
    reinforce_factor = np.where(roce_pair > 1e-15, 1.0 + refuerzo, 1.0)

    # --- 4. EXPANSIÓN CONTINUA (fórmula adjudicada): ciega a longitud, sólo lee fortaleza s=Sigma(w) ---
    exp_factor = np.exp(-p_exp * (s[:, None] + s[None, :]) / (2.0 * s_bar))

    W_nuevo = (W + dW_grav) * reinforce_factor * decay * exp_factor
    np.fill_diagonal(W_nuevo, 0.0)
    W_nuevo = np.clip(W_nuevo, 0.0, None)

    return T_nuevo, W_nuevo, roce_pair


def corre_ii_det(N, n_focos, delta, pasos, w0=W0_DEFAULT, **kw):
    """Corrida completa II-DET (cero RNG de principio a fin). kw se pasa a paso_ii_det (tasas)."""
    T, W = estado_inicial(N, n_focos, delta, w0=w0)
    historia_T = [T.copy()]
    for _ in range(pasos):
        T, W, _ = paso_ii_det(T, W, **kw)
        historia_T.append(T.copy())
    return dict(T_final=T, W_final=W, historia_T=historia_T)


if __name__ == "__main__":
    print("cs072_ii_nucleo.py -- Motor II determinista (II-DET), cero RNG. Correr cs072_ii_puerta_s.py.",
          flush=True)
