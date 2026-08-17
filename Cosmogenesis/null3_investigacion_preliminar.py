"""
null3_investigacion_preliminar.py — PASO DE INVESTIGACIÓN (no implementación completa) para NULL-3,
escalón 3 de 6 de la jerarquía de controles Fase II CS073.

Encargo de NULL-3 (palabras de Alexis): un control que conserva "grado de cada nodo y longitudes [de
enlace]" y destruye "motivos, ciclos e historia", para identificar el "efecto de la topología de
orden superior" -- el escalón siguiente a NULL-1 (conserva sólo el radio/perfil de densidad de cada
partícula, destruye el ángulo) y NULL-2 (conserva P(k)/2-puntos del campo, aproxima ξ(r) de
partícula vía Zel'dovich).

Por qué esto NO es lo mismo que `barajar_aristas` (double-edge-swap de Maslov-Sneppen, ya existente
en `p_semilla_causal.py`, usado por los NULL1-8 ORIGINALES de CS073, `bateria_n2000/ic_null1..8`):
ese swap preserva la SECUENCIA DE GRADOS exactamente, pero NO restringe la LONGITUD de las aristas
nuevas -- por eso, cuando se re-corre `layout_resortes` (relajación de resortes) sobre el grafo
barajado, la nube resultante cambia su perfil radial completo (r_mean/r_std) respecto de REAL (ver
docstring de `null1_generar_ic.py`: "los NULL1-8 existentes destruyen MÁS que la correspondencia
relacional -- también destruyen el perfil radial/densidad de la nube", KS<1e-113 en las 8
comparaciones). Esa es exactamente la razón por la que Alexis pidió un NULL-1 AISLADO nuevo (que
conserva el radio EXACTO por construcción, sin pasar por el grafo) en vez de reusar esos NULL1-8.

Hipótesis de trabajo para NULL-3: si se restringe el double-edge-swap para que sólo acepte
intercambios cuyas dos aristas NUEVAS tengan una longitud geométrica (medida sobre las posiciones
REALES ya existentes, `pos_real`) parecida a las dos aristas VIEJAS que reemplazan, el grafo
resultante debería, al pasar por `layout_resortes`, producir una nube con un perfil radial mucho más
parecido a REAL que el swap sin restricción -- porque la escala local de conexión (qué tan lejos
tiende a estar cada partícula de sus vecinas en el grafo) es precisamente lo que determina la escala
global de la relajación de resortes. Si eso se confirma (perfil radial de NULL-3 ≈ REAL, mucho más
cerca que NULL1-8 originales), la comparación de sumideros NULL-3 vs REAL aislaría el efecto de la
topología de orden superior (motivos/ciclos/triángulos específicos, "quién-con-quién" más allá de la
escala local) de forma mucho más limpia que el swap sin restringir.

Qué hace ESTE script (sólo investigación, NO la batería completa -- no corre Phantom, no genera 8
condiciones iniciales): reconstruye el grafo causal REAL exacto (determinista, mismo `dens_bar`,
`seed_ejes=2000`, `D=3`, `k=4` que usó `traducir_pool` para `ic_real`), calcula la longitud de cada
arista sobre las posiciones REAL ya escritas en disco, implementa y prueba
`barajar_aristas_preservando_longitud` (extensión de `barajar_aristas`, misma mecánica de
Maslov-Sneppen, con un filtro geométrico de tolerancia), y verifica en el propio grafo (sin
`layout_resortes`, sin Phantom) que: (a) el grado de cada nodo queda exactamente preservado (por
construcción del swap, igual que el original), y (b) la distribución de longitudes de las aristas
NUEVAS se parece MUCHO más a la distribución original que un swap sin restringir.

No toca `p_semilla_causal.py`, `fase1_traducir_a_phantom.py`, ni ningún archivo de `bateria_n2000/`
(sólo lectura de `dens_bar.npy` y `ic_real/cosmogenesis_ic.txt`). No genera condiciones iniciales de
Phantom. El siguiente paso, NO hecho aquí, sería: re-correr `layout_resortes` sobre el grafo NULL-3
para obtener posiciones, verificar el perfil radial contra REAL (como hace
`null2_zeldovich_disenar_verificar.py` para NULL-2) y recién si eso confirma la hipótesis, escalar a
8 semillas + Phantom con el mismo patrón de `null1_bateria_generar.py`/`_correr.py`/`_comparar.py`.
"""
import time

import numpy as np

from cs072_modulos.piezas.p_semilla_causal import malla_causal_atomos
from null1_generar_ic import leer_ic_txt

RUTA_DENS_BAR = "/Users/alexis/phantom_cs073/bateria_n2000/dens_bar.npy"
RUTA_IC_REAL = "/Users/alexis/phantom_cs073/bateria_n2000/ic_real/cosmogenesis_ic.txt"


# --------------------------------------------------------------------------------------------------
# Reconstrucción del grafo causal REAL exacto (determinista -- mismos parámetros que traducir_pool)
# --------------------------------------------------------------------------------------------------
def reconstruir_grafo_real():
    dens_bar = np.load(RUTA_DENS_BAR)
    adj, m = malla_causal_atomos(dens_bar, D=3, k=4, seed_ejes=2000)
    pos_real, _vel, _h, _mp, n = leer_ic_txt(RUTA_IC_REAL)
    assert len(dens_bar) == n, f"dens_bar (n={len(dens_bar)}) no coincide con ic_real (n={n})"
    return adj, pos_real, n


def aristas_de(adj, n):
    return [(i, j) for i in range(n) for j in adj[i] if j > i]


def longitudes(edges, pos):
    edges = np.asarray(edges)
    d = pos[edges[:, 0]] - pos[edges[:, 1]]
    return np.sqrt((d ** 2).sum(axis=1))


# --------------------------------------------------------------------------------------------------
# Double-edge-swap con filtro geométrico de longitud (extiende barajar_aristas de p_semilla_causal.py
# -- misma mecánica Maslov-Sneppen, no reescrita desde cero, sólo se añade la condición de aceptación
# por longitud). pos_referencia = posiciones REAL ya existentes (usadas SÓLO para medir longitud, no
# para mover ninguna partícula aquí).
# --------------------------------------------------------------------------------------------------
def barajar_aristas_preservando_longitud(adj, n, pos_referencia, seed, factor_swaps=10,
                                          tol_relativa=0.2):
    """Igual que barajar_aristas (Maslov-Sneppen): elige 2 aristas (a,b),(c,d) al azar y las reconecta
    a (a,d),(c,b). Acepta el intercambio SÓLO si AMBAS longitudes nuevas quedan dentro de
    tol_relativa (fracción) de la longitud de la arista que reemplazan -- |L_nuevo - L_viejo| <=
    tol_relativa * L_viejo. Esto preserva la secuencia de grados EXACTA (igual que el original, nunca
    se toca degree) y, además, mantiene la distribución de longitudes mucho más cerca de la original
    -- lo que se destruye es la identidad específica de qué par de nodos está conectado (motivos,
    ciclos, triángulos, "historia" de la malla), no la escala local de conexión."""
    rng = np.random.default_rng(seed)
    edges = aristas_de(adj, n)
    if len(edges) < 2:
        return {i: set(adj[i]) for i in range(n)}, 0, 0

    edges = [list(e) for e in edges]
    edge_set = set(tuple(e) for e in edges)
    n_intentos = factor_swaps * len(edges)
    aceptados = 0

    def L(u, v):
        d = pos_referencia[u] - pos_referencia[v]
        return float(np.sqrt((d ** 2).sum()))

    for _ in range(n_intentos):
        i1, i2 = rng.integers(0, len(edges), size=2)
        if i1 == i2:
            continue
        a, b = edges[i1]
        c, d = edges[i2]
        if len({a, b, c, d}) < 4:
            continue
        nuevo1 = tuple(sorted((a, d)))
        nuevo2 = tuple(sorted((c, b)))
        if nuevo1 in edge_set or nuevo2 in edge_set or nuevo1 == nuevo2:
            continue

        l_viejo1, l_viejo2 = L(a, b), L(c, d)
        l_nuevo1, l_nuevo2 = L(*nuevo1), L(*nuevo2)
        if abs(l_nuevo1 - l_viejo1) > tol_relativa * l_viejo1:
            continue
        if abs(l_nuevo2 - l_viejo2) > tol_relativa * l_viejo2:
            continue

        viejo1, viejo2 = tuple(sorted((a, b))), tuple(sorted((c, d)))
        edge_set.discard(viejo1)
        edge_set.discard(viejo2)
        edge_set.add(nuevo1)
        edge_set.add(nuevo2)
        edges[i1] = [a, d]
        edges[i2] = [c, b]
        aceptados += 1

    adj_null = {i: set() for i in range(n)}
    for (u, v) in edge_set:
        adj_null[u].add(v)
        adj_null[v].add(u)
    return adj_null, aceptados, n_intentos


# --------------------------------------------------------------------------------------------------
# Swap SIN restricción (idéntico en método a barajar_aristas de p_semilla_causal.py, duplicado aquí
# tal cual sólo para poder medirlo lado a lado en este mismo script de investigación -- no se importa
# porque comparten la misma firma que la versión con filtro y así ambas corridas quedan
# reproducibles/comparables con el mismo bucle de arriba). Es la MISMA función, factor_swaps default
# igual, sólo referenciada aquí para el diagnóstico -- no se usa en producción (esa es
# p_semilla_causal.barajar_aristas, congelada).
# --------------------------------------------------------------------------------------------------
from cs072_modulos.piezas.p_semilla_causal import barajar_aristas as barajar_aristas_sin_restriccion


def main():
    t0 = time.time()
    print("[1] reconstruyendo grafo causal REAL (determinista, mismos parámetros que traducir_pool)...",
          flush=True)
    adj_real, pos_real, n = reconstruir_grafo_real()
    edges_real = aristas_de(adj_real, n)
    L_real = longitudes(edges_real, pos_real)
    grados_real = np.array([len(adj_real[i]) for i in range(n)])
    print(f"    n={n}  n_aristas={len(edges_real)}  grado: min={grados_real.min()} "
          f"max={grados_real.max()} mean={grados_real.mean():.3f}  "
          f"L: mean={L_real.mean():.2f} std={L_real.std():.2f} "
          f"tiempo={time.time()-t0:.2f}s", flush=True)

    seed = 501
    print(f"\n[2] swap SIN restricción de longitud (Maslov-Sneppen original, seed={seed})...",
          flush=True)
    t1 = time.time()
    adj_sin, aceptados_sin, intentos_sin = None, None, None
    adj_sin = barajar_aristas_sin_restriccion(adj_real, n, seed=seed)
    edges_sin = aristas_de(adj_sin, n)
    L_sin = longitudes(edges_sin, pos_real)
    grados_sin = np.array([len(adj_sin[i]) for i in range(n)])
    print(f"    n_aristas={len(edges_sin)}  grado preservado exacto: "
          f"{np.array_equal(np.sort(grados_real), np.sort(grados_sin))}  "
          f"L: mean={L_sin.mean():.2f} std={L_sin.std():.2f}  tiempo={time.time()-t1:.2f}s",
          flush=True)

    print(f"\n[3] swap CON restricción de longitud (tol_relativa=0.2, seed={seed})...", flush=True)
    t2 = time.time()
    adj_con, aceptados, intentos = barajar_aristas_preservando_longitud(
        adj_real, n, pos_real, seed=seed, tol_relativa=0.2)
    edges_con = aristas_de(adj_con, n)
    L_con = longitudes(edges_con, pos_real)
    grados_con = np.array([len(adj_con[i]) for i in range(n)])
    n_aristas_distintas = len(set(tuple(e) for e in edges_con) - set(tuple(e) for e in edges_real))
    print(f"    n_aristas={len(edges_con)}  grado preservado exacto: "
          f"{np.array_equal(np.sort(grados_real), np.sort(grados_con))}  "
          f"L: mean={L_con.mean():.2f} std={L_con.std():.2f}  "
          f"swaps aceptados/intentados={aceptados}/{intentos} "
          f"({100*aceptados/intentos:.1f}%)  "
          f"aristas distintas de REAL={n_aristas_distintas}/{len(edges_real)} "
          f"({100*n_aristas_distintas/len(edges_real):.1f}%)  tiempo={time.time()-t2:.2f}s",
          flush=True)

    print("\n[4] resumen comparativo de la distribución de longitudes (vs REAL):")
    print(f"    REAL                        mean={L_real.mean():7.2f}  std={L_real.std():7.2f}")
    print(f"    swap SIN restricción        mean={L_sin.mean():7.2f}  std={L_sin.std():7.2f}  "
          f"(diff mean = {100*(L_sin.mean()-L_real.mean())/L_real.mean():+.1f}%)")
    print(f"    swap CON restricción (0.2)  mean={L_con.mean():7.2f}  std={L_con.std():7.2f}  "
          f"(diff mean = {100*(L_con.mean()-L_real.mean())/L_real.mean():+.1f}%)")

    from scipy.stats import ks_2samp
    ks_sin = ks_2samp(L_real, L_sin)
    ks_con = ks_2samp(L_real, L_con)
    print(f"\n    KS(L_real, L_sin_restriccion)  = {ks_sin.statistic:.4f}  p={ks_sin.pvalue:.2e}")
    print(f"    KS(L_real, L_con_restriccion)  = {ks_con.statistic:.4f}  p={ks_con.pvalue:.2e}")

    print(f"\n[TOTAL] {time.time()-t0:.2f}s", flush=True)
    print("\nEsto es SÓLO la verificación a nivel de grafo (grado + longitud), sin layout_resortes ni")
    print("Phantom -- punto de partida para NULL-3, no un resultado de la jerarquía. Ver")
    print("NULL3_investigacion_preliminar_CS.md para la lectura y los próximos pasos pendientes.")


if __name__ == "__main__":
    main()
