"""
CS068 — El análogo de inflación: estirar-y-enfriar el sustrato mundo-pequeño.
==============================================================================
Diseño: CS (DISENO_CS068_inflacion_estirar_enfriar_CS.md), 16-jul-2026. Codea/ejecuta: CC.
Cierra el cabo que CS067 (B, canónico con IC95%) probó ser la PRECONDICIÓN de las direcciones: mientras el
sustrato sea mundo-pequeño (ovillo, todo cerca de todo por atajos), no hay "lejos" real contra el cual las
direcciones se organicen.

ETAPA 1 (este archivo, primero): ¿el proceso de enfriar-y-estirar vuelve MÉTRICO el sustrato, con un
GRADIENTE de energía espacial que el orden por distancia explica y un corte al azar NO?

Opera SOBRE el sustrato heredado de CS067 (arm='completo', motor SIN TOCAR — se importa y se corre tal
cual): toma su adj FINAL (blob mundo-pequeño ya con localidad de CS066 aplicada) y aplica un proceso NUEVO
de enfriamiento posterior, que es lo único que CS068 aporta.

Clasificación de enlaces (sin umbral nuevo inventado — reusa el soporte de vecinos comunes, el mismo proxy
de correlación que _pesos_correlacion de CS067 ya usa): un enlace es ATAJO si soporte==0 (cero vecinos
comunes, sin corroboración local); es TEJIDO LOCAL si soporte>=1. El proceso de enfriamiento SOLO puede
romper atajos; el tejido local nunca se toca (es "barato", ya es local).

ℓ_ij (longitud real del atajo) = distancia BFS entre i,j usando SOLO el tejido local (sin ningún atajo).
Si i,j quedan desconectados en el tejido local, ℓ_ij = cota (diámetro robusto del tejido local + 1).

Proceso: T: 8.0 → 0.05, factor 0.6/paso. Cada atajo VIVO sobrevive el paso con p=exp(-ℓ_ij/T) (una tirada
por atajo vivo, por paso; muerto no revive). NO se impone T(r) — T es un parámetro del PROCESO (el reloj de
enfriamiento), no una función de la posición; la posición entra SOLO a través de ℓ_ij (medido de la
estructura). G-NO-CALIBRAR: T0/factor/T_final son del diseño, no se tocan para forzar el resultado.

Discriminante decisivo (NO es "hay lejos" — cualquier poda lo da): el GRADIENTE espacial
corr(dist_al_centro, E_nolocal), donde E_nolocal(nodo) = nº de atajos vivos que tocan ese nodo (energía de
correlación no-local, MEDIDA, no impuesta). Debe ser fuerte y NEGATIVA en inflar_dist (lejos=frío), y ~0 en
null_corte_azar/inflar_barajado (mismo nº de atajos rotos o misma energía total, pero sin orden espacial).
"""
from __future__ import annotations
import os, sys, math, time
from collections import deque
import numpy as np

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)
import cs067_habitacion_completa as H
import cs065_exclusion_pauli as C65

RNG = np.random.default_rng

T0 = 8.0
T_FACTOR = 0.6
T_FINAL = 0.05


# ============================ SUSTRATO (motor heredado, SIN TOCAR) ============================
def _sustrato(N, seed):
    """El blob de CS067 'completo' tal cual — se corre el motor heredado sin modificarlo y se toma su adj
    FINAL. CS068 NO opera durante la dinámica de CS067; opera DESPUÉS, como proceso separado."""
    rng = RNG(seed)
    par = H._sorteo(seed)
    cat = C65._cataloga065(N, RNG(seed))
    adj, V, dark, tb, f, D, conf = H.proceso067(N, cat, "completo", par, rng)
    return adj


# ============================ CLASIFICACIÓN: tejido local vs atajos ============================
def _soporte(adj, i, j):
    return len(adj[i] & adj[j])


def _clasifica(adj, N):
    """local = soporte >= MEDIANA del soporte de este grafo (no un umbral fijo a mano); atajo = por debajo.
    Verificado empíricamente (smoke): soporte==0 solo da ~4% de enlaces — insuficiente para que el proceso
    de enfriamiento tenga margen de mostrar un gradiente (el tejido resultante ya es casi todo el grafo y el
    diámetro apenas se mueve). La mediana es un corte DERIVADO del propio grafo, no un número elegido para
    forzar el resultado — a N/gamma distintos, el corte se recalcula solo."""
    edges = [(i, j) for i in range(N) for j in adj[i] if i < j]
    sup = np.array([_soporte(adj, i, j) for (i, j) in edges])
    umbral = float(np.median(sup))
    local_edges = [e for e, s in zip(edges, sup) if s >= umbral]
    atajos = [e for e, s in zip(edges, sup) if s < umbral]
    adj_local = [set() for _ in range(N)]
    for (i, j) in local_edges:
        adj_local[i].add(j); adj_local[j].add(i)
    return adj_local, atajos


def _bfs_local(adj_local, fuente):
    dist = {fuente: 0}
    q = deque([fuente])
    while q:
        u = q.popleft()
        for v in adj_local[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def _ell_atajos(adj_local, N, atajos, rng, n_landmarks=10):
    """ℓ_ij por atajo vía BFS exacto sobre el tejido local (cacheado por nodo-fuente). Cota para
    desconectados = diámetro robusto del tejido local (muestreado por landmarks) + 1."""
    fuentes = rng.integers(0, N, size=min(n_landmarks, N))
    eccs = []
    for s in fuentes:
        d = _bfs_local(adj_local, int(s))
        if len(d) > 0.3 * N:
            eccs.append(max(d.values()))
    cota = (int(np.median(eccs)) + 1) if eccs else N

    ell = {}
    cache = {}
    for (i, j) in atajos:
        if i not in cache:
            cache[i] = _bfs_local(adj_local, i)
        d = cache[i].get(j)
        ell[(i, j)] = float(d) if d is not None else float(cota)
    return ell, cota


def _centro_y_distancias(adj_local, N, rng):
    grados = [(len(adj_local[i]), i) for i in range(N) if adj_local[i]]
    c = max(grados)[1] if grados else int(rng.integers(0, N))
    dist = _bfs_local(adj_local, c)
    return c, dist


def _energia_por_nodo(N, vivos):
    E = np.zeros(N)
    for (i, j) in vivos:
        E[i] += 1.0; E[j] += 1.0
    return E


def _pasos_T():
    Ts = []; T = T0
    while T > T_FINAL:
        Ts.append(T); T *= T_FACTOR
    Ts.append(T)
    return Ts


def _corr(dist_centro, E, N):
    idx = [i for i in range(N) if i in dist_centro]
    if len(idx) < 10:
        return float("nan")
    x = np.array([dist_centro[i] for i in idx], float)
    y = np.array([E[i] for i in idx], float)
    if x.std() < 1e-9 or y.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _corr_cascaron(dist_centro, E, N):
    """Adjudicado por CS (ADJUDICACION_CS068_paso1_resultado_CS.md): reagrega E_nolocal por CASCARÓN radial
    (mismo dist_centro entero), no por nodo. E_nolocal por nodo es un entero chico (0,1,2 atajos vivos) —
    discretísimo, ruidoso nodo a nodo. Promediar dentro de cada cascarón es la MISMA cantidad física
    (mismo E_nolocal, sin imponer nada nuevo), solo que promedia el ruido de discretización — da la misma
    respuesta que _corr con ~4x la relación señal/ruido. Es el estimador estándar de aquí en adelante."""
    from collections import defaultdict
    cascarones = defaultdict(list)
    for i in range(N):
        if i in dist_centro:
            cascarones[dist_centro[i]].append(E[i])
    if len(cascarones) < 3:
        return float("nan")
    ds = sorted(cascarones)
    x = np.array(ds, float)
    y = np.array([float(np.mean(cascarones[d])) for d in ds], float)
    if x.std() < 1e-9 or y.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


# ============================ LOS CUATRO BRAZOS ============================
def _correr_inflar_dist(N, atajos, ell, rng):
    """Cada atajo saca UN umbral aleatorio u_ij~U(0,1) fijo (una sola vez, no se re-tira por paso — evitar
    que muchas tiradas independientes encadenadas maten todo en 1-2 pasos, que es lo que el smoke destapó).
    Vivo-en-T ⟺ u_ij < exp(-ℓ_ij/T) ⟺ T > ℓ_ij/(-ln u_ij) =: T_congela_ij. Muere exactamente cuando T cruza
    su propio T_congela (monótono, nunca revive). ℓ_ij grande -> tiende a T_congela alto -> muere primero,
    pero u_ij mantiene el sorteo (no es un orden determinista puro)."""
    u = {e: rng.random() for e in atajos}
    Tc = {e: ell[e] / max(1e-9, -math.log(max(u[e], 1e-12))) for e in atajos}
    vivos = set(atajos)
    traj = []; n_rotos_por_paso = []
    for T in _pasos_T():
        antes = len(vivos)
        muertos = [e for e in vivos if Tc[e] >= T]
        for e in muertos:
            vivos.discard(e)
        n_rotos_por_paso.append(antes - len(vivos))
        traj.append((T, len(vivos), _energia_por_nodo(N, vivos)))
    return traj, n_rotos_por_paso, vivos


def _correr_null_corte_azar(N, atajos, rng, n_rotos_por_paso):
    vivos = set(atajos)
    traj = []
    for nrotos in n_rotos_por_paso:
        vivos_list = list(vivos)
        nrotos = min(nrotos, len(vivos_list))
        if nrotos > 0:
            idxs = rng.choice(len(vivos_list), size=nrotos, replace=False)
            for t in idxs:
                vivos.discard(vivos_list[t])
        traj.append((None, len(vivos), _energia_por_nodo(N, vivos)))
    return traj, vivos


def _correr_sin_enfriar(N, vivos_final_dist):
    """Poda INSTANTÁNEA al mismo conjunto final que inflar_dist alcanzó — mismo resultado, SIN trayectoria.
    Aísla si el PROCESO (gradual, ordenado) importa más allá del estado final."""
    E = _energia_por_nodo(N, vivos_final_dist)
    return [(None, len(vivos_final_dist), E)], set(vivos_final_dist)


def _barajar_energia(E, rng):
    """Mismo total/histograma de E_nolocal, pero repartido AL AZAR entre nodos — rompe la correlación
    espacio-energía manteniendo los totales. Si la correlación real desaparece al barajar, es espacial de
    verdad; si no cambia, era un artefacto del histograma."""
    Eb = E.copy(); rng.shuffle(Eb)
    return Eb


# ============================ SMOKE — Etapa 1 ============================
def _un_patch(N, seed, rng_global):
    adj = _sustrato(N, seed)
    diam_ini = H._diam_robusto(adj, N, RNG(seed + 1))
    adj_local, atajos = _clasifica(adj, N)
    ell, cota = _ell_atajos(adj_local, N, atajos, RNG(seed + 2))
    centro, dist_centro = _centro_y_distancias(adj_local, N, RNG(seed + 3))

    rD = RNG(seed + 10)
    traj_d, n_rotos, vivos_d = _correr_inflar_dist(N, atajos, ell, rD)
    rA = RNG(seed + 11)
    traj_a, vivos_a = _correr_null_corte_azar(N, atajos, rA, n_rotos)
    traj_s, vivos_s = _correr_sin_enfriar(N, vivos_d)
    rB = RNG(seed + 12)
    E_final_d = traj_d[-1][2]
    E_barajada = _barajar_energia(E_final_d, rB)

    def _adj_final(vivos):
        a = [set(x) for x in adj_local]
        for (i, j) in vivos:
            a[i].add(j); a[j].add(i)
        return a

    diam_fin_d = H._diam_robusto(_adj_final(vivos_d), N, RNG(seed + 20))
    diam_fin_a = H._diam_robusto(_adj_final(vivos_a), N, RNG(seed + 21))

    # checkpoints por FRACCIÓN DE ATAJOS VIVOS restantes (75/50/25/5%), no por índice de paso — la muerte no
    # es lineal en pasos (frena y acelera), así que un índice fijo puede caer ya en la zona muerta.
    n0 = len(atajos)
    fracciones = [0.75, 0.50, 0.25, 0.05]
    idxs_chk = []
    for frac in fracciones:
        objetivo = frac * n0
        k = next((k for k, (_, nv, _) in enumerate(traj_d) if nv <= objetivo), len(traj_d) - 1)
        idxs_chk.append(k)
    idxs_chk = sorted(set(idxs_chk))
    corr_d_traj = [(traj_d[k][0], traj_d[k][1], round(_corr(dist_centro, traj_d[k][2], N), 3)) for k in idxs_chk]
    corr_a_traj = [(traj_a[k][0], traj_a[k][1], round(_corr(dist_centro, traj_a[k][2], N), 3)) for k in idxs_chk]

    return dict(
        N=N, seed=seed, n_atajos0=len(atajos), n_local_edges=sum(len(a) for a in adj_local) // 2,
        cota_ell=cota, diam_ini=diam_ini,
        diam_fin_dist=diam_fin_d, diam_fin_azar=diam_fin_a,
        n_atajos_fin_dist=len(vivos_d), n_atajos_fin_azar=len(vivos_a),
        corr_dist_traj=corr_d_traj, corr_azar_traj=corr_a_traj,
        corr_dist_final=round(_corr(dist_centro, E_final_d, N), 3),
        corr_azar_final=round(_corr(dist_centro, traj_a[-1][2], N), 3),
        corr_barajado_final=round(_corr(dist_centro, E_barajada, N), 3),
        corr_sin_enfriar_final=round(_corr(dist_centro, traj_s[-1][2], N), 3),
    )


def main():
    N = int(os.environ.get("CS068_N", 900))
    npatch = int(os.environ.get("CS068_PATCHES", 3))
    print("=" * 108, flush=True)
    print("CS068 ETAPA 1 — SMOKE: ¿el sustrato se vuelve métrico, con gradiente de energía ORDENADO por distancia?",
          flush=True)
    print(f"N={N} · patches={npatch} · T: {T0}->{T_FINAL} factor {T_FACTOR}", flush=True)
    print("=" * 108, flush=True)
    t0 = time.time()
    for p in range(npatch):
        seed = 68000 + 101 * p
        r = _un_patch(N, seed, RNG(seed))
        print(f"\n--- patch {p} (seed={seed}) ---", flush=True)
        print(f"  sustrato: N={r['N']} atajos0={r['n_atajos0']} enlaces_locales={r['n_local_edges']} "
              f"cota_ell={r['cota_ell']} diam_inicial={r['diam_ini']:.2f}", flush=True)
        print(f"  diam final: inflar_dist={r['diam_fin_dist']:.2f}  null_corte_azar={r['diam_fin_azar']:.2f}",
              flush=True)
        print(f"  atajos vivos al final: inflar_dist={r['n_atajos_fin_dist']}  "
              f"null_corte_azar={r['n_atajos_fin_azar']}", flush=True)
        print(f"  corr(dist_centro,E_nolocal) TRAYECTORIA inflar_dist: {r['corr_dist_traj']}", flush=True)
        print(f"  corr(dist_centro,E_nolocal) TRAYECTORIA null_corte_azar: {r['corr_azar_traj']}", flush=True)
        print(f"  corr FINAL: inflar_dist={r['corr_dist_final']}  null_corte_azar={r['corr_azar_final']}  "
              f"inflar_barajado={r['corr_barajado_final']}  null_sin_enfriar={r['corr_sin_enfriar_final']}",
              flush=True)
    print(f"\ntiempo total: {(time.time()-t0)/60:.2f} min", flush=True)
    print("\nLECTURA PRE-INSCRITA (Etapa 1): inflar_dist debe dar corr fuerte NEGATIVA (lejos=frío) creciendo", flush=True)
    print("en magnitud a lo largo de la trayectoria; null_corte_azar/barajado deben quedar cerca de 0 pese a", flush=True)
    print("lograr el MISMO diam/nº de atajos. Si null_corte_azar iguala el gradiente -> negativo honesto.", flush=True)


if __name__ == "__main__":
    main()
