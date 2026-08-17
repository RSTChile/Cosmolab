"""
CS072 v6 — núcleo: contacto por ROCE (memoria de enlace), no por orden de un escalar.
==============================================================================
Diseño: CS (DISENO_CS072_experimento_unico_CS.md v6), corrigiendo el bug fatal de v5 (cazado por Gemini,
verificado por CS con código): ordenar parcelas por VALOR de T y conectar vecinas-en-ese-orden fuerza una
cadena 1D (diám/N≈0.20 constante) -- CUALQUIER "cercanía = proximidad de un escalar" es 1D por naturaleza.

LA VECINDAD AHORA ES EL ROCE, no el parecido:
- Bootstrap: grafo ALEATORIO (GR.aleatorio, mismo generador que TODO el arco desde CS064) -- NO hay ORDEN
  de ningún escalar involucrado; es la sopa indiferenciada, "nada distingue un lugar de otro" antes de que
  el roce empiece a actuar. Este es el único punto donde CC declara una elección no derivada del roce
  mismo (tiene que arrancar de algo); se declara aquí, auditable.
- Cada paso, la vecindad se REFUERZA o se DILUYE según el roce real: cuánta energía intercambió cada
  enlace (memoria de CS071, ya validada: REFUERZO=0.04 DECAY=0.99 PRUNE_FRAC=0.15 -- NO se re-tunea). Un
  enlace que no carga roce se apaga y se poda; uno que sí, persiste. El estado de cada parcela no es solo
  T (escalar): también carga su historial de roce (w_hist por enlace) -- eso es lo que impide el colapso a
  1D (Gemini + CS, DISEÑO §3).
- INESTABILIDAD (gravedad, cs062 `_grav_peso`, SIN TOCAR): lo frío (w=1-T, "peso" por coldness, misma
  convención que el resto del arco) atrae más roce -- añade enlaces nuevos preferentemente entre parcelas
  frías. Ya trae su propio freno (CAP/E, el mismo de siempre) -- no se inventa un freno nuevo.
- DIFUSIÓN: intercambio de T a lo largo del contacto VIVO (idéntico a v5, ya validado que suaviza).
- ENFRIAMIENTO: global, monótono, multiplicativo (idéntico a v5).

G-DIMENSION-EMERGE: la dimensión se MIDE (d_s por bola, β por escalamiento, δ-Gromov), nunca se fija.

Codea/ejecuta: CC. Diseño/ruling: CS.
"""
from __future__ import annotations
import os, sys, math
import numpy as np

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)
import cg003_diagnostico_gromov as GR
import cs062_paisaje_peso as C62
import cs064_smoke as SM
import cs071_histeresis as S71
import cs072_v5_nucleo as V5   # campo_inicial, cv_heterogeneidad -- sin tocar, ya validados

RNG = np.random.default_rng

MEANDEG_BOOTSTRAP = 6.0    # grafo aleatorio inicial, MISMA convención que GR.aleatorio en todo el arco
GRAV_RATE = 0.30           # mismo orden de magnitud que R_GRAV=0.30 del resto del arco (cs057/062/064/067)
GRAV_DMAX = 3
TASA_FLUJO = 0.15          # tasa del flujo-de-enfriamiento (mismo orden que la TASA_INTERCAMBIO de v5, fijo)
ENFRIAMIENTO = V5.ENFRIAMIENTO


def _bootstrap(N, seed):
    """Grafo ALEATORIO, sin orden de ningún escalar -- el único punto de partida no-derivado-del-roce,
    declarado y auditable (DISEÑO §3: 'CC elige la implementación concreta, la declara')."""
    adj0, _ = GR.aleatorio(N, meandeg=MEANDEG_BOOTSTRAP, seed=seed)
    return [set(a) for a in adj0]


def flujo_enfriamiento(T, edges, N, tasa=TASA_FLUJO):
    """LA REGLA (CS, ADJUDICACION_CS072_v6_gravedad_flujo_enfriamiento_CS.md): ANTI-difusiva -- en cada
    enlace, el MÁS FRÍO CEDE energía al más tibio (lo opuesto de la difusión, que rellena el pozo hacia la
    media). Tasa ∝ contraste (T_tibio-T_frío), normalizada por el GRADO del frío (estabilidad numérica --
    un hub no se vacía de golpe por todos sus enlaces a la vez). PISO DURO T>=0: nadie cede más energía de
    la que TIENE ("no hay de dónde llenarse") -- eso es lo que ACOTA la inestabilidad sin una perilla; el
    límite es conservación pura. Conserva T_i+T_j exacto en cada transferencia (lo que cede uno, lo gana
    el otro, ni más ni menos)."""
    T = T.copy()
    grado = np.zeros(N)
    for (i, j) in edges:
        grado[i] += 1; grado[j] += 1
    for (i, j) in edges:
        if T[i] < T[j]:
            frio, tibio = i, j
        else:
            frio, tibio = j, i
        contraste = T[tibio] - T[frio]
        if contraste <= 0:
            continue
        d = max(grado[frio], 1.0)
        cede = min(tasa * contraste / d, T[frio])
        T[frio] -= cede; T[tibio] += cede
    return T


def _poda_grado(adj, N, rng, tasa):
    """EXPANSIÓN como operador de PODA sobre la topología (ADJUDICACION_CS072_v6_expansion_poda_CS.md):
    la mitad del balance gravedad-vs-expansión que faltaba -- antes la expansión sólo enfriaba T, nunca
    tocaba el tejido, y la gravedad ganaba sin freno (colapso a hub). Aquí la expansión CORTA enlaces,
    preferentemente en los nodos de MÁS grado (más expuestos/frágiles a la ruptura) -- efecto anti-hub.

    CIEGA A LA LONGITUD (dura, G del auditor): jamás lee distancia ni coordenada, sólo grado(i)+grado(j),
    una propiedad del GRAFO, no de una métrica supuesta. p_corte(i,j) = tasa * (grado_i+grado_j)/(2*grado_medio),
    capada a 1 -- un enlace entre dos nodos de grado medio se corta con prob=tasa; un hub (grado >> medio) se
    poda desproporcionadamente más rápido; nunca hay tope de grado puesto a mano (emerge del balance)."""
    if tasa <= 0:
        return
    grado = np.array([len(a) for a in adj], dtype=float)
    meandeg = max(float(grado.mean()), 1.0)
    edges = [(i, j) for i in range(N) for j in adj[i] if i < j]
    if not edges:
        return
    probs = tasa * (grado[[e[0] for e in edges]] + grado[[e[1] for e in edges]]) / (2.0 * meandeg)
    corta = rng.random(len(edges)) < np.clip(probs, 0.0, 1.0)
    for (i, j), c in zip(edges, corta):
        if c:
            adj[i].discard(j); adj[j].discard(i)


def _gravedad(adj, N, T, rng):
    """Inestabilidad: cs062._grav_peso SIN TOCAR. w=coldness (1-T) por NODO -- lo frío atrae, misma
    convención del resto del arco. Freno YA incluido en _grav_peso vía `rate` (CAP/E).

    CORRECCIÓN encontrada al smoke-testear (bug mecánico mío, no una decisión de diseño): la fórmula
    original gateaba `rate` por (1-T_MEDIA GLOBAL) -- convención correcta en el motor viejo, donde T es
    UNA temperatura compartida que baja igual para todos con el tiempo. Aquí T es un CAMPO por nodo con
    una fracción ε diminuta más fría: la media global queda ≈1.0 SIEMPRE (por diseño, ε es minúsculo), así
    que ese gate apagaba la gravedad casi del todo (nadd≈0) sin importar cuán fría fuera la brizna. El
    freno correcto no depende de la media global -- ya está en el propio w (solo los nodos fríos tienen
    peso real) y en el CAP/E de siempre."""
    w = np.clip(1.0 - T, 0.0, None) + 1e-9
    Tm = float(np.mean(T))
    E = sum(len(a) for a in adj) // 2
    if E < 1:
        return
    CAP = max(1, N // 4)
    rate = min(GRAV_RATE, CAP / max(1.0, (0.5 + Tm) * E))
    C62._grav_peso(adj, N, rng, rate, dmax=GRAV_DMAX, T=Tm, w=w)


def _memoria_roce(adj, w_hist, N, deg0_hist, T_antes, T_despues, rng, barajado):
    """Refuerza los enlaces que cargaron ROCE real este paso (proporcional a |ΔT| que fluyó por ellos),
    decae+poda con los parámetros YA validados de CS071. barajado: el refuerzo se asigna a enlaces AL AZAR
    de la MISMA magnitud total, no a los que realmente rozaron (G-NULL-MISMA-MAGNITUD)."""
    vivos = set()
    for i in range(N):
        for j in adj[i]:
            if i < j:
                vivos.add((i, j))
    for e in list(w_hist.keys()):
        if e not in vivos:
            del w_hist[e]
    for e in vivos:
        if e not in w_hist:
            w_hist[e] = 1.0

    if not barajado:
        for (i, j) in vivos:
            roce = abs(T_despues[i] - T_antes[i]) + abs(T_despues[j] - T_antes[j])
            if roce > 1e-12:
                w_hist[(i, j)] *= (1 + S71.REFUERZO)
    else:
        edges_vivos = list(vivos)
        n_roces = sum(1 for (i, j) in vivos
                      if abs(T_despues[i] - T_antes[i]) + abs(T_despues[j] - T_antes[j]) > 1e-12)
        if edges_vivos and n_roces:
            idxs = rng.integers(0, len(edges_vivos), size=n_roces)
            for t in idxs:
                e = edges_vivos[int(t)]
                w_hist[e] *= (1 + S71.REFUERZO)

    for e in list(w_hist.keys()):
        w_hist[e] *= S71.DECAY
    w_hist = S71._homeostasis_y_poda(adj, w_hist, N, deg0_hist)
    return w_hist


def corre_nucleo_v6(N, n_focos, delta, pasos, seed, barajado=False, grav_on=True, poda_tasa=0.0):
    rng = RNG(seed)
    T = V5.campo_inicial(N, n_focos, delta, rng)
    adj = _bootstrap(N, int(rng.integers(1 << 30)))
    deg0_hist = np.array([len(a) for a in adj], float)
    w_hist = {(i, j): 1.0 for i in range(N) for j in adj[i] if i < j}

    cvs = [V5.cv_heterogeneidad(T)]
    sumas = [float(T.sum())]
    diams = []
    for _ in range(pasos):
        if grav_on:
            _gravedad(adj, N, T, rng)
        T_antes = T.copy()
        edges_vivos = [(i, j) for i in range(N) for j in adj[i] if i < j]
        T = flujo_enfriamiento(T, edges_vivos, N)
        w_hist = _memoria_roce(adj, w_hist, N, deg0_hist, T_antes, T, rng, barajado=barajado)
        _poda_grado(adj, N, rng, poda_tasa)   # EXPANSIÓN: poda topológica ciega a longitud (por grado)
        T = V5.enfria(T, factor=ENFRIAMIENTO)
        cvs.append(V5.cv_heterogeneidad(T))
        sumas.append(float(T.sum()))
    return dict(T_final=T, adj=adj, cvs=cvs, sumas=sumas)


if __name__ == "__main__":
    print("cs072_v6_nucleo.py -- núcleo roce+gravedad+difusión+memoria. Correr cs072_v6_exploratoria.py.",
          flush=True)
