"""
CS072 — EL EXPERIMENTO ÚNICO. La singularidad como TODO.
==============================================================================
Diseño/ruling: CS (DISENO_CS072_experimento_unico_CS.md, 17-jul-2026). Instrucción literal de Alexis: "la
interacción, la relación compleja, la que provoca la emergencia, no las partes... todo lo que fueron
haciendo debe incluirse en UN experimento único, complejo y multidimensional (cada cosa ocurriendo al mismo
tiempo del proceso de cálculo)".

NO hay motor nuevo ni proxy: `proceso072` es H.proceso067 (los 17 ingredientes, cs064+CS067) con TRES
mecanismos plegados DENTRO del mismo bucle por-paso, co-actuando desde t=0, reusando cantidades que el
motor YA calcula (no se inventan clasificadores nuevos):

- **[CS070] semilla primordial:** el paso SSB Potts usa `_ssb_potts_semilla` (cs070_semilla.py) en vez de
  `H.ssb_potts` -- un sesgo mínimo (peso∈[0.05,0.20], UN eje) presente en la votación desde el primer paso.
- **[CS068] enfriamiento/inflación:** reusa el wmap de correlación YA calculado para el voto SSB pesado
  (ingrediente 14, `H._pesos_correlacion`) -- cada enlace sobrevive el paso con p=exp(-d_ij/T), d_ij=-log(w_ij),
  T = la MISMA temperatura que ya gobierna el resto de la física ese paso. Local (w≈1,d≈0) sobrevive
  siempre; atajo (w→0,d grande) muere preferentemente al enfriar. Sin clasificador nuevo, sin ℓ por BFS.
- **[CS071] memoria de enlace:** cada paso, unos pocos caminantes (N//4, 2 saltos c/u -- escalado porque
  esto corre DENTRO de los 20 pasos del motor, no como proceso propio de 30) recorren el adj VIVO de ese
  paso, refuerzan lo transitado; homeostasis+poda con los parámetros YA validados en CS071
  (REFUERZO=0.04, DECAY=0.99, PRUNE_FRAC=0.15 -- G-NO-CALIBRAR, no se re-explora aquí).

**[CS069] fase cuántica relacional: DECLARADA FUERA de esta corrida** (acoplar el propagador complejo al
motor real de partículas es una integración mayor, no una llamada de función; se declara pendiente para
CS073, no se esconde).

NULL_TODO_BARAJADO: identidades de partícula shuffled (como null_tipos de CS064) + semilla BARAJADA
(orientación al azar por nodo, misma magnitud) + refuerzo de memoria BARAJADO (mismo nº de toques,
edges al azar) -- destruye las correlaciones entre QUIÉN es cada nodo y CÓMO se refuerza/siembra, preserva
cantidades globales. El enfriamiento-CS068 (topológico, no de identidad) se deja intacto en ambos brazos --
es parte del motor de fuerzas, no una "pieza plegada" con etiqueta de nodo que barajar.

Codea/ejecuta: CC. Diseño/ruling: CS. PROTOCOLO CERRADO (§8): nada se ajusta a mitad de tanda.
"""
from __future__ import annotations
import os, sys, time, math
import numpy as np

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)
import cs057_paisaje_completo as C7
import cs062_paisaje_peso as C62
import cs059_espin_como_marco as C9
import cg003_diagnostico_gromov as GR
import cs064_smoke as SM
import cs064_sistema_completo as C64
import cs065_exclusion_pauli as C65
import cs066_localidad_geometrogenesis as C66
import cs067_habitacion_completa as H
import cs070_semilla as S70
import cs071_histeresis as S71

RNG = np.random.default_rng
DMAX_INT = C64.DMAX_INT
STEPS = int(os.environ.get("CS072_STEPS", 20))

# memoria de enlace, ESCALADA para correr cada uno de los STEPS del motor (no cada macro-paso propio) --
# mismo cronograma cualitativo de CS071 (REFUERZO/DECAY/PRUNE_FRAC intactos, G-NO-CALIBRAR), menos
# caminantes/saltos por paso porque aquí hay ~20 pasos, no 30, y cada paso YA es costoso (17 ingredientes).
MEM_FRAC_WALKERS = 0.25
MEM_PASOS_POR_CAMINANTE = 2


# ============================ semilla (CS070) — al inicio, presente desde t=0 ============================
def _arma_semilla(N, K, arm, rng):
    peso = float(rng.uniform(*S70.PESO_SEMILLA_RANGO))
    if arm == "todo":
        eje = np.zeros(N, dtype=int); signo = np.ones(N, dtype=int)
    else:  # null_todo_barajado
        eje = rng.integers(0, K, size=N); signo = rng.integers(0, 2, size=N)
    return peso, eje, signo


# ============================ memoria de enlace (CS071) — un paso ligero, dentro del loop ============================
def _memoria_un_paso(adj, w_hist, N, deg0_hist, rng, barajado):
    """Sincroniza w_hist con el adj VIVO (enlaces nuevos entran a 1.0, los que murieron por otras fuerzas
    salen solos), corre caminantes (o toques al azar de la MISMA magnitud si barajado), decae, aplica
    homeostasis+poda de CS071 sobre el adj. Devuelve w_hist actualizado."""
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

    n_walkers = max(1, int(N * MEM_FRAC_WALKERS))
    if not barajado:
        pos = rng.integers(0, N, size=n_walkers)
        for _ in range(MEM_PASOS_POR_CAMINANTE):
            nuevas = pos.copy()
            for idx in range(n_walkers):
                i = int(pos[idx])
                if not adj[i]:
                    continue
                j = S71._elige_vecino(adj, w_hist, i, rng)
                if j is None:
                    continue
                e = (i, j) if i < j else (j, i)
                if e in w_hist:
                    w_hist[e] *= (1 + S71.REFUERZO)
                nuevas[idx] = j
            pos = nuevas
    else:
        edges_vivos = list(w_hist.keys())
        if edges_vivos:
            n_toques = n_walkers * MEM_PASOS_POR_CAMINANTE
            idxs = rng.integers(0, len(edges_vivos), size=n_toques)
            for t in idxs:
                e = edges_vivos[int(t)]
                w_hist[e] *= (1 + S71.REFUERZO)

    for e in list(w_hist.keys()):
        w_hist[e] *= S71.DECAY
    w_hist = S71._homeostasis_y_poda(adj, w_hist, N, deg0_hist)
    return w_hist


# ============================ EL MOTOR ÚNICO: proceso067 + semilla + memoria + enfriamiento-CS068 ============================
def proceso072(N, cat, arm, par, rng):
    """arm ∈ {todo, null_todo_barajado}. Devuelve (adj, V, conf_final, dark, D) -- misma forma que
    proceso067 + conf (para el juez de picado de CS070)."""
    fam, color, carga, masa, es_anti = (cat["fam"].copy(), cat["color"].copy(), cat["carga"].copy(),
                                        cat["masa"].copy(), cat["es_anti"].copy())
    es_ferm = cat["es_ferm"].copy()
    if arm == "null_todo_barajado":       # identidades barajadas -- MISMA estructura que null_tipos de CS064
        for a in (fam, color, carga, es_anti, es_ferm):
            rng.shuffle(a)
        rng.shuffle(masa)

    f = H._flags("completo")               # los 17 ingredientes SIEMPRE activos -- el NULL no apaga piezas,
                                            # baraja correlaciones (la tesis del director, §5 del diseño)
    dark = np.zeros(N, bool)
    if f["dark"]:
        dark = rng.random(N) < par["dark"]
        color = color.copy(); carga = carga.copy()
        color[dark] = -1; carga[dark] = 0
    emask = es_ferm.copy()
    Dm = C65._mask_diag(emask)

    adj0, _ = GR.aleatorio(N, meandeg=6.0, seed=int(rng.integers(1 << 30)))
    adj = [set(a) for a in adj0]
    col = np.where(color >= 0, color, rng.integers(0, 3, N)).astype(np.int8)
    car = (carga > 0).astype(np.int8)
    deg0 = [len(a) for a in adj]
    t = np.zeros(N, dtype=np.int32)
    V = C9._spins(N, DMAX_INT, rng)
    t_birth = np.full(N, STEPS, dtype=float); deg_prev = np.array(deg0, float)
    R = getattr(C7, "R_GRAV", 1.0); T_CONF = getattr(C7, "T_CONF", 0.5); CAP_E = 12 * N
    k_local = par["k_local"]

    peso_semilla, eje_semilla, signo_semilla = _arma_semilla(N, par["K"], arm, RNG(int(rng.integers(1 << 30))))
    w_hist = {(i, j): 1.0 for i in range(N) for j in adj[i] if i < j}
    deg0_hist = np.array(deg0, float)

    D, G = [], []
    conf_final = None
    for step in range(STEPS):
        T = C7._T_de_paso(step, 1.0) if hasattr(C7, "_T_de_paso") else max(0.02, 1.6 * (1 - step / STEPS))
        E = sum(len(a) for a in adj) // 2
        if E > CAP_E:
            break
        w = masa * max(0.0, 1.0 - T); CAP = max(1, N // 4)
        try:
            rg = min(0.30 * R * (1 - T), CAP / max(1.0, (0.5 + T) * E))
            C62._grav_peso(adj, N, rng, rg, dmax=3, T=T, w=w + 1e-9)
        except Exception: pass
        if T < T_CONF:
            try: C7._confin(adj, N, col, t, rng, min(0.8, CAP / max(1.0, E)))
            except Exception: pass
        try: C7._em(adj, N, car, deg0, rng, 0.12)
        except Exception: pass
        try: C7._debil(N, col, car, rng, 0.05)
        except Exception: pass
        try: C7._despliegue(adj, N, rng, 0.14 * T)
        except Exception: pass
        if T > 0.4 and step % 2 == 0:
            for i in rng.choice(N, size=max(1, N // 60), replace=False):
                if adj[i] and es_anti[i] != es_anti[list(adj[i])[0]]:
                    j = list(adj[i])[0]; adj[i].discard(j); adj[j].discard(i)
        if f["local"]:
            C66.gate_localidad(adj, N, rng, k_local, barajado=False)

        # correlación (ingrediente 14) -- se calcula UNA vez, se reusa para el voto SSB Y para el
        # enfriamiento-CS068 (mismo wmap, dos lecturas, G-MECANISMO-UNIFICADO)
        wmap = H._pesos_correlacion(adj, N, par["gamma"], "real", rng)

        # [CS068 PLEGADO] enfriamiento: cada enlace sobrevive con p=exp(-d_ij/T), d_ij=-log(w_ij) YA
        # calculado arriba. Local (w≈1) sobrevive casi siempre; atajo (w→0) muere preferente al enfriar --
        # MISMO T que gobierna el resto de la física ese paso, sin proceso aparte.
        if wmap:
            for (i, j), wij in list(wmap.items()):
                dij = -math.log(max(wij, 1e-6))
                p_sobrevive = math.exp(-dij / max(T, 1e-3))
                if rng.random() > p_sobrevive:
                    adj[i].discard(j); adj[j].discard(i)

        # [CS071 PLEGADO] memoria de enlace: caminantes sobre el adj YA actualizado este paso
        w_hist = _memoria_un_paso(adj, w_hist, N, deg0_hist, rng, barajado=(arm == "null_todo_barajado"))

        degnow = np.array([len(a) for a in adj], float)
        cambiaron = np.abs(degnow - deg_prev) > 0.5
        nc = int(cambiaron.sum())
        if nc: t_birth[cambiaron] = step + rng.random(nc)
        deg_prev = degnow

        if f["marco_vivo"]:
            A = SM.adj_sparse(adj, N)
            tt = t_birth.copy()
            A_use = H._A_causal(A, tt, par["c"], N) if f["causal"] else A
            # ingrediente 14 (b): repesar Aw con la correlación YA calculada (wmap) para el voto SSB
            import scipy.sparse as sp
            Ac = A_use.tocoo()
            if wmap:
                wd = np.array([wmap.get((r, c) if r < c else (c, r), 1.0) for r, c in zip(Ac.row, Ac.col)])
                Aw = sp.coo_matrix((wd, (Ac.row, Ac.col)), shape=(N, N)).tocsr()
            else:
                Aw = A_use
            # [CS070 PLEGADO] SSB Potts CON semilla -- presente desde el primer paso, ciega a es_atajo
            V, conf_final = S70._ssb_potts_semilla(V, Aw, par["K"], peso_semilla, eje_semilla, signo_semilla)
        D.append(C7._diam(adj, N)); G.append(C7._giant(adj, N))
    return adj, V, conf_final, dark, D


def _cataloga_completo(N, rng):
    cat = C65._cataloga065(N, rng)
    return cat


if __name__ == "__main__":
    print("cs072_experimento_unico.py -- módulo del motor. Correr cs072_exploratoria.py primero (declarada,", flush=True)
    print("§8), luego cs072_tanda.py solo si la exploratoria confirma que el motor arranca con todo plegado.",
          flush=True)
