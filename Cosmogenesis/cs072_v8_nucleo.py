"""
CS072 v8 -- núcleo FOLD (II): sustrato SIN grafo previo (MANIFIESTO_FOLD_CS072.md, "SUSTRATO INICIAL").
==============================================================================
G-NO-SUSTRATO-PREVIO (dura, del director vía CS): el código NO arranca de GR.aleatorio, random_regular, ni
ningún grafo con aristas binarias privilegiadas. El único estado inicial legítimo es una MATRIZ DE AFINIDAD
continua W (N×N, simétrica, diagonal 0), con w_ij = W0 IGUAL para TODO par -- simetría de permutación TOTAL,
nada privilegiado de entrada. ε rompe SÓLO la temperatura (campo T, vía V5.campo_inicial, sin tocar), NUNCA
la topología (que no existe aún). El "al lado de" (la topología que leen los jueces: diám, δ-Gromov, d_s,
frac_gigante, grado_max) se LEE DESPUÉS de que W diverja por la dinámica de roce I⟷E -- se define como
"i,j en contacto" sii w_ij > media(W) en ese instante (umbral ADAPTATIVO, lee sólo la propia W -- una
variable de estado del roce, nunca coordenada ni distancia -- G-DONDE-ES-SOMBRA respetado). Nunca se
siembra la topología: es SALIDA, no entrada.

Esta es la FASE 1 del fold (Task tracker): el sustrato nuevo + los 4 mecanismos YA VALIDADOS en v6/v7
(gravedad #2, flujo-enfriamiento -- realización de #2, memoria de enlace -- mecanismo de origen CS071,
poda-por-grado #9/18) reescritos como operaciones MATRICIALES VECTORIZADAS (numpy, sin loop Python sobre
pares -- a N=1600 un doble loop sería inviable). El resto de los 18 elementos + sector cohesión son fases
posteriores (declaradas, no escondidas -- ver MANIFIESTO §1).

TODOS los operadores leen el estado (T,W) AL INICIO del paso y escriben sobre el estado nuevo -- no hay
cascada (G-PROCESO-NO-SUCESION): gravedad lee T_antes (no W ya actualizada por flujo), flujo lee W_antes
(no la W ya reforzada por gravedad); sólo memoria y poda, que actúan sobre TOPOLOGÍA (W), se compone
secuencialmente entre sí porque ambas son la MISMA variable de estado (W) y no pueden verse ambas la
versión "antes" sin perder una -- igual que en v6/v7 (memoria luego poda, mismo orden ya validado).

Codea/ejecuta: CC. Diseño/ruling: CS + director (MANIFIESTO_FOLD_CS072.md).
"""
from __future__ import annotations
import os, sys
import numpy as np

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)
import cs072_v5_nucleo as V5   # campo_inicial, enfria, cv_heterogeneidad -- sin tocar, substrate-agnostic
import cs071_histeresis as S71  # REFUERZO, DECAY (parámetros YA validados, G-NO-CALIBRAR)

RNG = np.random.default_rng

W0 = 1.0                  # peso relacional uniforme inicial, TODO par (matriz de afinidad, G-NO-SUSTRATO-PREVIO)
GRAV_RATE = 0.30           # mismo orden que R_GRAV del resto del arco
TASA_FLUJO = 0.15          # heredado de v6 (flujo_enfriamiento)
ENFRIAMIENTO = V5.ENFRIAMIENTO
REFUERZO = S71.REFUERZO
DECAY = S71.DECAY


def _W_inicial(N):
    """Matriz de afinidad UNIFORME: TODOS los pares con el MISMO peso W0, diagonal 0. Simetría de
    permutación total -- ningún par privilegiado. NO es un grafo (no hay "aristas": todo par empieza
    igual de conectado)."""
    W = np.full((N, N), W0, dtype=np.float64)
    np.fill_diagonal(W, 0.0)
    return W


def _gravedad_matriz(W, T, rate):
    """Elemento #2 (gravedad ∝ peso-masa, cs054/062 `_grav_peso`), reescrita como REFUERZO DE PESO entre
    pares fríos (en vez de 'añadir arista nueva' -- aquí no hay aristas que añadir, todo par ya existe;
    lo frío REFUERZA el peso relacional que ya tenía con lo frío). Presupuesto acotado por paso (análogo
    al CAP/E del motor viejo) para que no diverja -- el freno es CONSERVACIÓN de presupuesto, no una
    perilla de escala."""
    N = W.shape[0]
    cold = np.clip(1.0 - T, 0.0, None) + 1e-9
    attract = np.outer(cold, cold)
    np.fill_diagonal(attract, 0.0)
    total = attract.sum()
    if total <= 0:
        return W
    CAP = max(1.0, N / 4.0) * W0
    budget = min(rate * total, CAP)
    W = W + attract * (budget / total)
    return W


def _flujo_enfriamiento_matriz(T, W, tasa):
    """Flujo-de-enfriamiento (ADJUDICACION_CS072_v6_gravedad...): el MÁS FRÍO cede al más tibio,
    proporcional al peso w_ij (contacto continuo, no binario) y normalizado por el GRADO PESADO del frío
    (Σ_k w_ik) -- generaliza v6 (que normalizaba por grado entero) al sustrato continuo. PISO DURO T>=0
    (nadie cede más de lo que tiene) vía escalado por fila. Conserva ΣT exacto (lo que sale de una fila
    entra en una columna)."""
    N = T.shape[0]
    deg_w = np.maximum(W.sum(axis=1), 1e-9)
    D = T[None, :] - T[:, None]                 # D[i,j] = T_j - T_i
    contraste = np.clip(D, 0.0, None)            # >0 sólo donde j es más tibio que i (i es el frío)
    raw = tasa * W * contraste / deg_w[:, None]
    raw_out = raw.sum(axis=1)
    scale = np.where(raw_out > 1e-12, np.minimum(1.0, T / np.maximum(raw_out, 1e-12)), 1.0)
    sent = raw * scale[:, None]                  # sent[i,j]: i (frío) cede a j (tibio)
    T_nuevo = T - sent.sum(axis=1) + sent.sum(axis=0)
    T_nuevo = np.clip(T_nuevo, 0.0, None)
    roce_pair = sent + sent.T                    # simétrico: cuánto roce hubo en el par (i,j) este paso
    return T_nuevo, roce_pair


def _memoria_matriz(W, roce_pair, N, deg0_w):
    """Memoria de enlace (mecanismo de origen, CS071 -- NO es uno de los 18): refuerza w_ij donde hubo
    roce real este paso, decae TODO, homeostasis como TECHO (nunca como piso): reescala hacia ABAJO sólo
    si el presupuesto del nodo (Σ_j w_ij) EXCEDE deg0_w = (N-1)*W0 -- evita runaway de la gravedad, igual
    que el CAP/E del motor viejo. CORRECCIÓN (bug cazado en el smoke-test del sustrato nuevo): si
    homeostasis reescalara TAMBIÉN hacia ARRIBA (como en el CS071 original, pensado para un grafo con
    aristas que se ELIMINAN de la adyacencia), en este sustrato continuo (matriz completa, ningún par
    'desaparece', sólo baja de peso) eso REINFLARÍA cada paso el presupuesto completo y BORRARÍA el efecto
    de la poda (#9/18) y del propio decaimiento -- verificado: con reescalado bidireccional, grado_max y CV
    salían IDÉNTICOS para toda tasa de poda de 0.02 a 0.3. El techo unidireccional deja que la poda y el
    decaimiento se acumulen (persistan), que es lo que #9/18 y la 2ª ley exigen."""
    touched = roce_pair > 1e-12
    W = W.copy()
    W[touched] *= (1.0 + REFUERZO)
    W *= DECAY
    row_sum = np.maximum(W.sum(axis=1), 1e-9)
    factor = np.minimum(1.0, deg0_w / row_sum)   # SÓLO techo: nunca infla por debajo del presupuesto
    W = W * np.sqrt(np.outer(factor, factor))
    np.fill_diagonal(W, 0.0)
    return W


def _poda_grado_matriz(W, N, poda_tasa):
    """Elemento #9/18 (expansión = poda, ADJUDICACION_CS072_v6_expansion_poda + validada en v7): SUPRIME
    peso (no 'corta arista' -- aquí no hay aristas discretas) proporcional al grado PESADO de los extremos
    relativo al grado medio -- CIEGA A LA LONGITUD (sólo lee Σw, propiedad de la matriz de estado, nunca
    coordenada ni distancia). Generaliza v7 (poda binaria con probabilidad p) a supresión continua
    (shrink por factor (1-p)) -- mismo espíritu, sustrato continuo."""
    if poda_tasa <= 0:
        return W
    deg_w = W.sum(axis=1)
    meandeg_w = max(float(deg_w.mean()), 1e-9)
    supresion = poda_tasa * (deg_w[:, None] + deg_w[None, :]) / (2.0 * meandeg_w)
    W = W * np.clip(1.0 - supresion, 0.0, 1.0)
    np.fill_diagonal(W, 0.0)
    return W


def lee_topologia(W, N):
    """La topología ("al lado de") es SALIDA, nunca entrada: i,j en contacto sii w_ij > media(W fuera de
    diagonal) EN ESE INSTANTE -- umbral ADAPTATIVO (se autoajusta a la escala de W, que crece/decae con
    la dinámica), lee SÓLO la propia matriz de estado (roce persistido), jamás coordenada/distancia."""
    iu = np.triu_indices(N, k=1)
    off_vals = W[iu]
    umbral = float(off_vals.mean()) if off_vals.size else 0.0
    mask = off_vals > umbral
    ii = iu[0][mask]; jj = iu[1][mask]
    adj = [set() for _ in range(N)]
    for i, j in zip(ii.tolist(), jj.tolist()):
        adj[i].add(j); adj[j].add(i)
    return adj, umbral


def corre_nucleo_v8(N, n_focos, delta, pasos, seed, poda_tasa=0.0, grav_on=True):
    rng = RNG(seed)
    T = V5.campo_inicial(N, n_focos, delta, rng)     # ÚNICA asimetría: la temperatura (ε), no la topología
    W = _W_inicial(N)                                 # G-NO-SUSTRATO-PREVIO: matriz uniforme, sin grafo
    deg0_w = (N - 1) * W0

    cvs = [V5.cv_heterogeneidad(T)]
    for _ in range(pasos):
        T_antes = T
        W_antes = W
        # --- todos leen el estado AL INICIO del paso (paralelo, no cascada, G-PROCESO-NO-SUCESION) ---
        W_grav = _gravedad_matriz(W_antes, T_antes, GRAV_RATE) if grav_on else W_antes
        T_nuevo, roce_pair = _flujo_enfriamiento_matriz(T_antes, W_antes, TASA_FLUJO)
        # memoria y poda SÍ se componen entre sí (ambas escriben W, no pueden ambas leer "antes" sin perder
        # la otra -- mismo orden ya validado en v6/v7: memoria (refuerza lo recién tocado) luego poda).
        W_mem = _memoria_matriz(W_grav, roce_pair, N, deg0_w)
        W_nuevo = _poda_grado_matriz(W_mem, N, poda_tasa)

        T = V5.enfria(T_nuevo, factor=ENFRIAMIENTO)
        W = W_nuevo
        cvs.append(V5.cv_heterogeneidad(T))

    adj, umbral = lee_topologia(W, N)
    return dict(T_final=T, W_final=W, adj=adj, umbral_lectura=umbral, cvs=cvs)


if __name__ == "__main__":
    print("cs072_v8_nucleo.py -- FASE 1 del fold: sustrato sin grafo previo + 4 mecanismos validados.",
          flush=True)
    print("Correr cs072_v8_smoke.py para el smoke-test de ruptura de simetría.", flush=True)
