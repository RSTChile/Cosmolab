"""
CS070 — La semilla: ¿una asimetría primordial mínima se amplifica en direcciones, o se lava?
==============================================================================
Diseño/ruling: CS (DISENO_CS070_semilla_amplificacion_CS.md, 17-jul-2026). Fundado en C-N2.5.5 de la
canónica (una asimetría primordial mínima, tipo violación CP, fue NECESARIA para que el universo no se
aniquilara en simetría perfecta). El arco CS064-069 pidió SIEMPRE que la dirección emergiera de una sopa
simétrica -- nunca puso esta semilla. CS070 la pone: UN bit de asimetría (una dirección débil común), no un
ingrediente relacional ni un operador que obliga (G-SEMILLA-MINIMA, G-SEMILLA-UN-EJE).

MECANISMO -- reusa el motor de los 17 y su juez SIN TOCARLOS (adj final de E._sustrato = H.proceso067
'completo', Aw = H._pesos_correlacion ingrediente 14, V0 = C9._spins, juez = H.cuenta_ejes_gap +
SM.tensor_orientacion, igual que juzga067). Lo ÚNICO nuevo: un proceso de consenso POST-HOC (como CS068/069
sobre el blob congelado) que corre SSB Potts CON una semilla de asimetría inyectada en el voto.

_ssb_potts_semilla = H.ssb_potts + UN término: antes del argmax, se suma peso_semilla al estado
(eje_semilla_i, signo_semilla_i) de cada nodo. Si eje/signo son el MISMO valor para todo nodo -> SEMILLA
COHERENTE (un eje débil común); si se sortean independientes por nodo -> SEMILLA BARAJADA (misma magnitud,
sin coherencia, G-NULL-MISMA-MAGNITUD). peso_semilla=0 -> SIN_SEMILLA, reduce exactamente al Potts de CS067.

JUEZ -- NUNCA coherencia (el toy cazó la trampa: en mundo-pequeño la coherencia de 0.98 es colapso a 1 eje,
no dimensión). El veredicto es n_ejes MÚLTIPLES con el candado picado-por-nodo (pico_medio>0.85) del arco.

Codea/ejecuta: CC. Diseño/ruling: CS.
"""
from __future__ import annotations
import os, sys, time, math
import numpy as np
import scipy.sparse as sp

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)
import cs067_habitacion_completa as H
import cs068_inflacion_estirar_enfriar as E
import cs066_localidad_geometrogenesis as C66
import cs059_espin_como_marco as C9
import cs064_smoke as SM

RNG = np.random.default_rng

PESO_SEMILLA_RANGO = (0.05, 0.20)   # G-SEMILLA-MINIMA: sorteado bajo, nunca calibrado para que "salga"
T_PASOS_CONSENSO = 20                # pasos del proceso post-hoc de consenso. Fijo -- G-NO-CALIBRAR.
K_LOCAL_FUERTE = 4                   # k_local del gate CS066 para el brazo SUSTRATO_LOCAL (persistencia dura)
INERTIA = 0.3                        # misma inercia que H.ssb_potts usa por defecto


# ============================ consenso SSB Potts + semilla (única pieza nueva) ============================
def _ssb_potts_semilla(V, Aw, K, peso_semilla, eje_semilla, signo_semilla, inertia=INERTIA):
    """H.ssb_potts + un término de semilla en el voto: counts[estado_semilla_i] += peso_semilla, antes del
    argmax. eje_semilla/signo_semilla son arrays (N,) -- COHERENTE si son constantes, BARAJADA si varían por
    nodo (mismo peso_semilla en ambos casos: G-NULL-MISMA-MAGNITUD). peso_semilla=0 reduce exactamente a
    H.ssb_potts (SIN_SEMILLA). No lee ninguna verdad de fondo del grafo -- la semilla es una condición
    inicial impuesta UNA vez, no un operador que reevalúa qué es atajo o local."""
    N, D = V.shape
    P = V[:, :K]
    axis = np.argmax(np.abs(P), axis=1)
    sign = (P[np.arange(N), axis] >= 0).astype(int)
    state = (axis * 2 + sign).astype(int)
    A = Aw.tocsr(); indptr, indices, wdata = A.indptr, A.indices, A.data
    new = state.copy()
    conf = np.ones(N)
    estado_semilla = (eje_semilla * 2 + signo_semilla).astype(int)
    for i in range(N):
        a, b = indptr[i], indptr[i + 1]
        nb = indices[a:b]; wt = wdata[a:b]
        counts = np.zeros(2 * K)
        if len(nb) > 0:
            wsum = wt.sum()
            if wsum > 0:
                np.add.at(counts, state[nb], wt / wsum)
        counts[state[i]] += inertia
        counts[estado_semilla[i]] += peso_semilla
        new[i] = int(np.argmax(counts))
        axis_counts = np.clip(counts, 0, None).reshape(K, 2).sum(axis=1)
        tot = float(axis_counts.sum())
        conf[i] = float(axis_counts[new[i] // 2] / tot) if tot > 0 else 1.0
    ax = new // 2; sg = np.where(new % 2 == 1, 1.0, -1.0)
    Vn = np.zeros((N, D)); Vn[np.arange(N), ax] = sg
    return Vn, conf


# ============================ sustrato + Aw (motor de los 17, SIN TOCAR) ============================
def _sustrato_y_Aw(N, seed, sustrato_local=False):
    rng = RNG(seed)
    par = H._sorteo(seed)
    adj = E._sustrato(N, seed)  # adj FINAL de H.proceso067 'completo' -- motor heredado sin tocar
    if sustrato_local:
        C66.gate_localidad(adj, N, rng, K_LOCAL_FUERTE, barajado=False)
    wmap = H._pesos_correlacion(adj, N, par["gamma"], "real", rng)
    A = SM.adj_sparse(adj, N)
    Ac = A.tocoo()
    if wmap:
        wd = np.array([wmap.get((r, c) if r < c else (c, r), 1.0) for r, c in zip(Ac.row, Ac.col)])
        Aw = sp.coo_matrix((wd, (Ac.row, Ac.col)), shape=(N, N)).tocsr()
    else:
        Aw = A
    return adj, Aw, par["K"]


# ============================ semillas ============================
def _semilla_coherente(N, K, rng, peso=None):
    if peso is None:
        peso = float(rng.uniform(*PESO_SEMILLA_RANGO))
    eje = np.zeros(N, dtype=int)           # UN eje fijo, el mismo para TODO nodo (G-SEMILLA-UN-EJE)
    signo = np.ones(N, dtype=int)          # signo fijo
    return peso, eje, signo


def _semilla_barajada(N, K, rng, peso):
    eje = rng.integers(0, K, size=N)       # eje AL AZAR por nodo -- misma magnitud, sin coherencia
    signo = rng.integers(0, 2, size=N)
    return peso, eje, signo


# ============================ el proceso de consenso (post-hoc, sobre el blob congelado) ============================
def _corre_consenso(Aw, N, K, rng, peso_semilla, eje_semilla, signo_semilla, t_pasos=T_PASOS_CONSENSO):
    V = C9._spins(N, K, rng)
    conf = np.ones(N)
    for _ in range(t_pasos):
        V, conf = _ssb_potts_semilla(V, Aw, K, peso_semilla, eje_semilla, signo_semilla)
    return V, conf


def _juzga(V, conf):
    ev = SM.tensor_orientacion(V)
    n_ejes, PR, gap_interno, r_thr = H.cuenta_ejes_gap(ev)
    pico_medio = float(np.mean(conf)); frac_picados = float(np.mean(np.asarray(conf) > 0.9))
    return dict(n_ejes=n_ejes, PR=PR, gap_interno=gap_interno, pico_medio=pico_medio,
                frac_picados=frac_picados, certificado=(pico_medio > 0.85))


# ============================ LOS 4 BRAZOS ============================
def brazo_semilla_coherente(N, seed, sustrato_local=False):
    adj, Aw, K = _sustrato_y_Aw(N, seed, sustrato_local=sustrato_local)
    rng = RNG(seed + 1)
    peso, eje, signo = _semilla_coherente(N, K, rng)
    V, conf = _corre_consenso(Aw, N, K, RNG(seed + 2), peso, eje, signo)
    r = _juzga(V, conf); r["peso_semilla"] = peso
    return r


def brazo_semilla_barajada(N, seed, sustrato_local=False):
    adj, Aw, K = _sustrato_y_Aw(N, seed, sustrato_local=sustrato_local)
    rng = RNG(seed + 1)
    peso_ref, _e, _s = _semilla_coherente(N, K, rng)  # MISMO sorteo de peso que la coherente (misma magnitud)
    peso, eje, signo = _semilla_barajada(N, K, rng, peso_ref)
    V, conf = _corre_consenso(Aw, N, K, RNG(seed + 2), peso, eje, signo)
    r = _juzga(V, conf); r["peso_semilla"] = peso
    return r


def brazo_sin_semilla(N, seed, sustrato_local=False):
    adj, Aw, K = _sustrato_y_Aw(N, seed, sustrato_local=sustrato_local)
    eje = np.zeros(N, dtype=int); signo = np.ones(N, dtype=int)
    V, conf = _corre_consenso(Aw, N, K, RNG(seed + 2), 0.0, eje, signo)
    r = _juzga(V, conf); r["peso_semilla"] = 0.0
    return r


def brazo_semilla_sustrato_local(N, seed):
    return brazo_semilla_coherente(N, seed, sustrato_local=True)


BRAZOS = dict(semilla_coherente=brazo_semilla_coherente, semilla_barajada=brazo_semilla_barajada,
              sin_semilla=brazo_sin_semilla, semilla_sustrato_local=brazo_semilla_sustrato_local)


if __name__ == "__main__":
    print("cs070_semilla.py -- módulo de mecanismo. Correr cs070_smoke.py para las anclas obligatorias.")
