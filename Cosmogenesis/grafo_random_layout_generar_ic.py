"""
grafo_random_layout_generar_ic.py — grafo de control INDEPENDIENTE (Erdős-Rényi G(n,m), SIN ninguna
relación con la malla causal REAL) + `layout_resortes` + pipeline físico idéntico al resto de la
jerarquía CS073 (NULL-1/NULL-2/NULL-3). Control nuevo pedido por Alexis, Fase II CS073.

Pregunta que testea (formulada por el propio patrón de datos): `NULL3_robustecido_motivos_dosis_CS.md`
señaló, como hallazgo especulativo (no verificado ahí), que los NULL1-8 ORIGINALES de CS073
(`bateria_n2000/ic_null1..8`, swap Maslov-Sneppen SIN filtro de longitud sobre la malla causal REAL --
que destruye ~99.9% de los triángulos/motivos, ver `null3_motivos_directos.py`: 4 triángulos de 2780)
TODAVÍA formaron sumideros PARCIALMENTE (masa promedio ≈680-770), mientras que NULL-1 (radio exacto/
ángulo aleatorio) y NULL-2 (Zel'dovich) -- que NO pasan por ningún grafo de vecindad ni por
`layout_resortes` -- dieron CERO sumideros en 16 corridas combinadas. Lectura especulativa a testear:
¿es el mero HECHO de pasar por un grafo de vecindad + `layout_resortes` (relajación de resortes,
proceso FÍSICO) lo que siembra la estructura local necesaria para sumideros -- sea cual sea la
identidad de ese grafo -- o hace falta algo de la estructura/identidad real de la malla causal? Este
script testea la versión más limpia posible de esa pregunta: un grafo Erdős-Rényi G(n,m) genuinamente
INDEPENDIENTE de REAL (no comparte grado, no comparte aristas, no comparte ninguna propiedad salvo el
orden de magnitud de n y de nº de aristas), puesto a pasar por el MISMO `layout_resortes` + misma
dilatación + mismo campo de velocidad + misma escritura ASCII que toda la jerarquía (REAL/NULL-1/
NULL-2/NULL-3).

Construcción del grafo random -- Erdős-Rényi G(n,m) por rejection sampling de pares únicos (el método
correcto y más simple para generar exactamente m aristas simples sin repetición: la densidad de la
malla causal es baja, ~0.25% de C(n,2), así que la tasa de colisión al samplear pares al azar es
despreciable y el rechazo converge en pocas pasadas). n y m objetivo se toman CONTANDO la malla causal
REAL ya existente (misma función `malla_causal_atomos`/`aristas_de` que usa el resto de la jerarquía,
sólo para CONTAR nodos/aristas -- nunca se usa ninguna arista ni ninguna posición de la malla REAL para
construir este grafo random: es estadísticamente independiente por construcción, no un rebarajado).

Diferencia explícita con NULL-3: NULL-3 parte del grafo REAL exacto y sólo rebaraja aristas preservando
grado+longitud (destruye motivos/ciclos, conserva identidad local). Este control NO parte del grafo
REAL en absoluto -- es un grafo nuevo, generado desde cero, sin ninguna relación con la secuencia de
grados, las posiciones, ni la identidad de ningún nodo de REAL. Sólo comparte con REAL el orden de
magnitud de n y de nº de aristas.

Reusa (importado tal cual, NO modificado): `layout_resortes` de `p_semilla_causal.py`, `Expansion` de
`p_expansion.py`, `T0`/`_T_reloj` de `cs073_cierre_holistico.py`, `HFACT`/`POLYK` de
`fase1_traducir_a_phantom.py`, `malla_causal_atomos`/`aristas_de` (sólo para CONTAR la malla REAL, ver
`contar_aristas_malla_real` abajo) de `p_semilla_causal.py`/`null3_investigacion_preliminar.py`.
No toca ningún archivo congelado ni ninguna carpeta de batería/piloto anterior -- sólo lectura/
importación. Mismo patrón de módulo que `null3_generar_ic.py` (duplica la FORMA del pipeline, no la
lógica de cada paso).
"""
import numpy as np

from cs073_cierre_holistico import T0, _T_reloj
from cs072_modulos.piezas.p_expansion import Expansion
from cs072_modulos.piezas.p_semilla_causal import malla_causal_atomos, layout_resortes
from fase1_traducir_a_phantom import HFACT, POLYK
from null3_investigacion_preliminar import aristas_de


def contar_aristas_malla_real(dens_bar, D=3, k=4, seed_ejes=2000):
    """Cuenta nodos/aristas de la malla causal REAL construida sobre ESTE pool de bariones (mismos
    parámetros D/k/seed_ejes que usa `traducir_pool`/`generar_null3` en toda la jerarquía) -- SÓLO
    para dimensionar el grafo random (mismo n, misma m aproximada). No se usa ninguna arista de acá
    para construir el grafo random: es sólo un conteo."""
    n = len(dens_bar)
    adj_real, _m = malla_causal_atomos(dens_bar, D=D, k=k, seed_ejes=seed_ejes)
    n_aristas = len(aristas_de(adj_real, n))
    return n, n_aristas, adj_real


def generar_grafo_erdos_renyi(n, n_aristas, seed):
    """G(n,m): elige `n_aristas` pares únicos (i,j), i≠j, uniformes sobre TODOS los C(n,2) pares
    posibles de n nodos -- rejection sampling (densidad baja, colisión rarísima, converge en pocas
    pasadas). Devuelve dict nodo->set(vecinos), MISMO formato (adj) que `malla_causal_atomos`, para
    poder alimentar `layout_resortes` sin cambiar su firma. Grafo genuinamente independiente: ninguna
    arista, ningún grado, ninguna posición se toma de REAL -- sólo n y n_aristas coinciden en orden de
    magnitud (pedido explícito de la tarea: 'no necesitás preservar grado, ni longitud, ni ninguna otra
    propiedad de REAL')."""
    rng = np.random.default_rng(seed)
    adj = {i: set() for i in range(n)}
    edge_set = set()
    intentos = 0
    intentos_max = max(n_aristas * 200, 20000)  # margen amplio; densidad ~0.25% => colisión rarísima
    while len(edge_set) < n_aristas and intentos < intentos_max:
        i, j = rng.integers(0, n, size=2)
        intentos += 1
        if i == j:
            continue
        par = (int(i), int(j)) if i < j else (int(j), int(i))
        if par in edge_set:
            continue
        edge_set.add(par)
        adj[par[0]].add(par[1])
        adj[par[1]].add(par[0])
    assert len(edge_set) == n_aristas, (
        f"no se alcanzaron {n_aristas} aristas únicas en {intentos_max} intentos "
        f"(quedó en {len(edge_set)}/{n_aristas}) -- ¿n_aristas demasiado grande para n={n}?")
    return adj, edge_set, intentos


def generar_control_random(masa_bar, dens_bar, n_aristas, seed_random, seed_layout=None,
                            iters_layout=100, n_pasos_expansion=60, vel_generador=None,
                            hfact=HFACT, polyk=POLYK, ruta_salida="grafo_random_ic.txt"):
    """Genera UNA condición inicial de control: grafo Erdős-Rényi G(n,n_aristas) (independiente de
    REAL, construido desde cero con `seed_random`) -> `layout_resortes` (MISMA función/parámetros
    -- iters=100 -- que usa toda la jerarquía REAL/NULL-1-grafo/NULL-2/NULL-3) -> MISMA dilatación
    isótropa estática (`Expansion`, n_pasos_expansion=60, a_final=√60≈7.75, no depende de ninguna
    semilla) -> mismo campo de velocidad turbulento opcional -> h inicial uniforme -> escritura ASCII
    idéntica al resto de la jerarquía.

    `seed_layout` por defecto = `seed_random` (varía por semilla de corrida -- a diferencia de NULL-3,
    que fija `seed_layout=12345` en las 8 corridas y sólo varía la semilla del swap; acá se pidió
    explícitamente que `seed_layout` variara por semilla, ya que no hay una "semilla de swap"
    separada -- la única fuente de aleatoriedad de este control es la construcción del grafo + el
    layout, y ambas se hacen variar juntas para no reusar la misma relajación de resortes en las 8
    corridas)."""
    n = len(masa_bar)
    assert len(dens_bar) == n, (
        f"masa_bar(n={n}) / dens_bar(n={len(dens_bar)}) no coinciden -- ¿pool distinto?")
    if seed_layout is None:
        seed_layout = seed_random

    lado = float(n) ** (1.0 / 3.0)
    adj_random, edge_set, intentos_rechazo = generar_grafo_erdos_renyi(n, n_aristas, seed=seed_random)

    pos = layout_resortes(adj_random, n, lado=lado, iters=iters_layout, seed=seed_layout)

    # MISMA dilatación isótropa estática que toda la jerarquía -- depende sólo de n_pasos_expansion y
    # del reloj, nunca de seed_random/seed_layout.
    expansion = Expansion(T0=T0)
    for step in range(n_pasos_expansion):
        expansion.paso_de_estiramiento(_T_reloj(step))
    a_final = expansion._a_prev
    pos = pos * a_final

    vel = vel_generador(pos, adj_random, dens_bar) if vel_generador is not None else np.zeros_like(pos)
    h_guess = np.full(n, hfact)
    masa_media = float(masa_bar.mean())

    with open(ruta_salida, "w") as f:
        f.write(f"# cosmogenesis_ic v2 CONTROL grafo-random (Erdos-Renyi G(n,m) INDEPENDIENTE de la "
                 f"malla causal REAL) -- npart={n} n_aristas={n_aristas} "
                 f"masa_particula={masa_media:.17g} hfact={hfact} polyk={polyk:.17g} "
                 f"seed_random={seed_random} seed_layout={seed_layout}\n")
        f.write(f"{n} {masa_media:.17g} {hfact} {polyk:.17g}\n")
        for i in range(n):
            f.write(f"{float(pos[i,0]):.17g} {float(pos[i,1]):.17g} {float(pos[i,2]):.17g} "
                     f"{float(vel[i,0]):.17g} {float(vel[i,1]):.17g} {float(vel[i,2]):.17g} "
                     f"{float(h_guess[i]):.17g}\n")

    return dict(ruta=ruta_salida, n=n, pos=pos, masa_particula=masa_media, a_final=a_final,
                seed_random=seed_random, seed_layout=seed_layout, n_aristas=n_aristas,
                grado_medio=2.0 * n_aristas / n, intentos_rechazo=intentos_rechazo,
                adj=adj_random)


if __name__ == "__main__":
    print("Uso como módulo. Ver grafo_random_piloto_generar.py / grafo_random_bateria_generar.py.")
