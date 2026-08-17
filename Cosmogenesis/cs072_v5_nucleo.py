"""
CS072 v5 — núcleo: campo de temperatura, contacto SIN grid, intercambio I⟷E, enfriamiento, entropía.
==============================================================================
Diseño: CS (DISENO_CS072_experimento_unico_CS.md v5). Implementa §2-3-bis. Declaraciones de CC (el diseño
exige "CC elige, declara y audita" para la construcción libre-de-índice):

**G-DONDE-ES-SOMBRA — construcción del contacto:** el estado es SOLO un array T (N temperaturas), sin
posición. "Vecindad" se lee ordenando las parcelas por su VALOR de temperatura (argsort) y conectando cada
una con sus k más cercanas EN ESE ORDEN DE VALOR (no en el índice original del array -- dos parcelas con
temperaturas casi iguales quedan "al lado" aunque sus índices de array estén lejos; dos con temperaturas
muy distintas quedan lejos aunque sus índices sean consecutivos). El grafo se RECONSTRUYE cada paso desde
las temperaturas ACTUALES -- la topología es una sombra de la temperatura, nunca del layout de memoria.

**Entropía/heterogeneidad — proxy declarado:** coeficiente de variación CV=std(T)/mean(T). Se elige sobre
la varianza cruda porque el enfriamiento GLOBAL (multiplicativo, uniforme) escala TODOS los T_i por el
mismo factor cada paso -- eso encogería la varianza cruda trivialmente sin decir nada sobre si la
diferencia ε se está lavando o ahondando. CV es INVARIANTE a un reescalado uniforme (std y mean escalan
igual), así que solo cambia por la diferenciación real del intercambio I⟷E, no por el enfriamiento del
todo. Arranca en ~0 (T casi uniforme) y el diseño predice que debe CRECER si el trinquete funciona.

**Intercambio I⟷E:** difusivo, pairwise, a lo largo de los enlaces de contacto -- T_i += tasa*(T_j-T_i),
simétrico, conserva EXACTAMENTE T_i+T_j en cada intercambio (por construcción, sin importar el orden de
procesamiento de los enlaces -- la suma total del sistema es invariante).

Codea/ejecuta: CC. Diseño/ruling: CS.
"""
from __future__ import annotations
import os, sys, math
import numpy as np

RNG = np.random.default_rng

K_CONTACTO = 6           # vecinos en orden-de-valor de temperatura (mismo orden de magnitud que WS_K del arco)
TASA_INTERCAMBIO = 0.15  # fracción de la diferencia que se transfiere por contacto y por paso
ENFRIAMIENTO = 0.97      # T *= ENFRIAMIENTO cada paso, global, uniforme (monótono por construcción)


# ============================ estado inicial (G-SINGULARIDAD) ============================
def campo_inicial(N, n_focos, delta, rng):
    """T=1.0 en todas las parcelas salvo n_focos elegidas AL AZAR (rng, no por índice -- cualquier parcela
    puede ser la fría, el índice no significa nada) que arrancan en (1-delta). n_focos y delta son los dos
    parámetros de la invarianza (§6)."""
    T = np.ones(N, dtype=float)
    focos = rng.choice(N, size=min(n_focos, N), replace=False)
    T[focos] = 1.0 - delta
    return T


# ============================ contacto (G-DONDE-ES-SOMBRA) ============================
def contacto_por_temperatura(T, k=K_CONTACTO):
    """Grafo de contacto DERIVADO del valor de T (argsort), nunca del índice del array. Devuelve lista de
    pares (i,j) -- cada parcela conectada a sus ~k vecinas más cercanas EN VALOR de temperatura."""
    N = len(T)
    orden = np.argsort(T, kind="stable")  # posición p en orden creciente de T -> índice real orden[p]
    edges = set()
    for p in range(N):
        i = int(orden[p])
        for d in range(1, k // 2 + 1):
            if p + d < N:
                j = int(orden[p + d])
                edges.add((i, j) if i < j else (j, i))
    return edges


def contacto_barajado(T, k=K_CONTACTO, rng=None):
    """NULL: mismo Nº de enlaces que contacto_por_temperatura, pero BARAJADOS -- destruye la correlación
    entre 'quién está en contacto' y 'qué tan parecida es su temperatura' (misma magnitud, G-NULL-MISMA-
    MAGNITUD), conservando la estadística de T intacta (T no se toca aquí, solo el grafo)."""
    N = len(T)
    n_edges = len(contacto_por_temperatura(T, k))
    edges = set()
    intentos = 0
    while len(edges) < n_edges and intentos < n_edges * 30:
        intentos += 1
        i, j = int(rng.integers(0, N)), int(rng.integers(0, N))
        if i == j:
            continue
        e = (i, j) if i < j else (j, i)
        edges.add(e)
    return edges


# ============================ intercambio I⟷E (difusivo, conserva suma) ============================
def intercambio(T, edges, tasa=TASA_INTERCAMBIO):
    """Para cada enlace de contacto, transfiere tasa*(T_j-T_i) de j hacia i (y lo simétrico) -- conserva
    T_i+T_j EXACTAMENTE en cada intercambio, sin importar el orden de procesamiento (conservación global
    telescópica). Devuelve T actualizado (copia)."""
    T = T.copy()
    for (i, j) in edges:
        d = tasa * (T[j] - T[i])
        T[i] += d; T[j] -= d
    return T


def enfria(T, factor=ENFRIAMIENTO):
    """Enfriamiento global monótono, uniforme (todas las parcelas por igual) -- G-CONSERVACION: no crea
    ni destruye diferencias RELATIVAS (todo se escala por el mismo factor), solo baja la escala absoluta."""
    return T * factor


# ============================ entropía / heterogeneidad (proxy declarado) ============================
def cv_heterogeneidad(T):
    m = float(np.mean(T))
    if abs(m) < 1e-12:
        return float("nan")
    return float(np.std(T) / abs(m))


# ============================ un run completo del núcleo (sin leyes plegadas aún) ============================
def corre_nucleo(N, n_focos, delta, pasos, seed, barajado=False):
    rng = RNG(seed)
    T = campo_inicial(N, n_focos, delta, rng)
    suma0 = float(T.sum())
    cvs = [cv_heterogeneidad(T)]
    sumas = [float(T.sum())]
    for _ in range(pasos):
        edges = contacto_barajado(T, rng=rng) if barajado else contacto_por_temperatura(T)
        T = intercambio(T, edges)
        T = enfria(T)
        cvs.append(cv_heterogeneidad(T))
        sumas.append(float(T.sum()))
    return dict(T_final=T, cvs=cvs, sumas=sumas, suma0=suma0)


if __name__ == "__main__":
    print("cs072_v5_nucleo.py -- núcleo temperatura+contacto+entropía. Correr cs072_v5_exploratoria.py.",
          flush=True)
