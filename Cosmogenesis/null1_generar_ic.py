"""
null1_generar_ic.py — Fase II CS073, escalón NULL-1 ("blindaje de la jerarquía de 6 controles").

Qué problema resuelve, en simple: el arco CS073 ya tenía 8 controles NULL (`ic_null1`..`ic_null8` en
/Users/alexis/phantom_cs073/bateria_n2000/) construidos así: la misma malla causal de REAL (grafo de
vecindad, `p_semilla_causal.malla_causal_atomos`), con las ARISTAS barajadas por double-edge-swap
(`barajar_aristas`, preserva grado exacto), pasada por el MISMO layout de resortes
(Fruchterman-Reingold, `layout_resortes`). Se verificó empíricamente (ver
NULL1_piloto_distribucion_radial_CS.md) que ese barajado de aristas, aunque preserva el grado de cada
nodo, SÍ cambia la forma global de la nube: la distancia de cada partícula al centro de masa en REAL
(r_mean=72.78, r_std=8.20, N=2000) es sistemáticamente distinta de la de los 8 NULL existentes
(r_mean≈63.2-63.5, r_std≈13.3-13.7), con test de Kolmogorov-Smirnov altamente significativo en las 8
comparaciones (p < 1e-113 en todos los casos). Es decir: los NULL1-8 existentes destruyen MÁS que la
correspondencia relacional -- también destruyen el perfil radial/densidad de la nube. No son el
equivalente de "NULL-1" (el escalón más aislado de la jerarquía de 6 controles, que debería tocar SOLO
la correspondencia relacional).

Qué hace este módulo: construye una condición inicial NULL-1 genuina, que:
  CONSERVA -- EXACTAMENTE, no sólo en distribución -- el conjunto de distancias al centro de masa que ya
    tiene la nube REAL (se leen las posiciones REALES ya generadas por `fase1_traducir_a_phantom.
    traducir_pool`, se calcula r_i = |pos_i - COM| para cada partícula, y NULL-1 hereda ese MISMO
    multiconjunto de radios -- mismo histograma de distancia al centro, mismo perfil rho(r), por
    construcción, no por muestreo).
  DESTRUYE -- la dirección angular de cada partícula se reasigna a un vector aleatorio isótropo
    (uniforme sobre la esfera), independiente de cualquier vecino, de la malla causal, o del layout de
    resortes. No hay grafo, no hay Fruchterman-Reingold: es una permutación pura del ángulo, a radio
    fijo. Esto rompe cualquier estructura de "quién quedó cerca de quién" (el objeto que la malla causal
    + FR sí codifica), sin tocar la única cantidad que a NULL-1 le toca conservar.

Por qué esto y no "reasignar partículas a posiciones ya existentes de REAL": con la convención actual
de fase1_traducir_a_phantom.py, la masa es la MISMA para las 2000 (o N) partículas
(`masa_media = masa_bar.mean()`) y la velocidad se calcula PURAMENTE a partir de la posición final
(`campo_velocidad_turbulento` interpola un campo en `pos`, no depende de qué "partícula" ocupa ese punto).
Es decir: en el archivo de condición inicial que Phantom realmente lee, las partículas son
indistinguibles salvo por su posición -- "reasignar identidades mantiendo las posiciones" produciría un
archivo BIT-IDÉNTICO al de REAL (ningún observable cambiaría). El único grado de libertad real y no
trivial es la posición misma. Por eso NULL-1 actúa sobre el ángulo, conservando el radio.

No toca: p_semilla_causal.py, fase1_traducir_a_phantom.py, campo_velocidad_turbulento.py,
cs073_cierre_holistico.py, leer_volcado_phantom.py (todos congelados) -- sólo los importa. No escribe
ni modifica nada dentro de /Users/alexis/phantom_cs073/bateria_n2000/ (datos originales verificados,
sólo lectura para leer ic_real/cosmogenesis_ic.txt si se quiere comparar contra la batería N=2000).
"""
import numpy as np

from fase1_traducir_a_phantom import traducir_pool, HFACT, POLYK


def leer_ic_txt(ruta):
    """Lee un archivo cosmogenesis_ic.txt (formato v2, el mismo que escribe traducir_pool) y devuelve
    (pos, vel, h, masa_particula, n). No reimplementa el parser de Phantom -- sólo el ASCII plano que
    nuestro propio traducir_pool ya produce, formato de 3 líneas de header + 1 línea por partícula."""
    with open(ruta) as f:
        f.readline()  # comentario
        n, masa_particula, hfact, polyk = f.readline().split()
        n = int(n)
        masa_particula = float(masa_particula)
        data = np.loadtxt(f, max_rows=n)
    pos = data[:, 0:3]
    vel = data[:, 3:6]
    h = data[:, 6]
    return pos, vel, h, masa_particula, n


def radios_desde_real(masa_bar, dens_bar, seed_layout=12345, ruta_tmp="/tmp/_null1_real_tmp_ic.txt",
                       **kw_traducir):
    """Genera (o reutiliza) la condición REAL con la pieza YA VALIDADA (traducir_pool, seed_null=None) y
    devuelve el multiconjunto de radios r_i=|pos_i-COM| que NULL-1 va a heredar EXACTO, junto con la
    propia condición REAL (para poder correrla como punto de comparación limpio del piloto)."""
    info = traducir_pool(masa_bar, dens_bar, seed_null=None, seed_layout=seed_layout,
                          ruta_salida=ruta_tmp, **kw_traducir)
    pos_real, vel_real, h_real, masa_particula, n = leer_ic_txt(ruta_tmp)
    com = pos_real.mean(axis=0)
    r = np.linalg.norm(pos_real - com, axis=1)
    return dict(pos_real=pos_real, vel_real=vel_real, h_real=h_real, com=com, r=r,
                masa_particula=masa_particula, n=n, info_real=info)


def generar_null1(r, com, masa_particula, n, seed, vel_generador=None, hfact=HFACT, polyk=POLYK,
                   ruta_salida="null1_ic.txt"):
    """EL GENERADOR NULL-1: mismo multiconjunto de radios `r` (ya calculado sobre la nube REAL), ángulo
    de cada partícula redistribuido a un punto aleatorio ISÓTROPO de la esfera unidad (Marsaglia 1972 --
    normal 3D normalizada, el método estándar para muestrear direcciones uniformes en la esfera, no
    inventado para este experimento). seed distingue las 2-3 semillas del piloto -- ortogonal a
    seed_layout/seed_null de la pieza congelada (este módulo no los usa, no hay malla ni FR aquí)."""
    rng = np.random.default_rng(seed)
    n_r = len(r)
    assert n_r == n, f"radios ({n_r}) no coincide con n ({n}) -- ¿pos_real truncada?"

    direcciones = rng.normal(size=(n, 3))
    normas = np.linalg.norm(direcciones, axis=1, keepdims=True)
    direcciones = direcciones / normas  # uniforme en la esfera unidad (Marsaglia)

    pos = com + r[:, None] * direcciones

    vel = vel_generador(pos, None, None) if vel_generador is not None else np.zeros_like(pos)
    h_guess = np.full(n, hfact)

    with open(ruta_salida, "w") as f:
        f.write(f"# cosmogenesis_ic v2 NULL-1 (angulo aleatorio, radio heredado de REAL) -- npart={n} "
                 f"masa_particula={masa_particula:.17g} hfact={hfact} polyk={polyk:.17g} seed={seed}\n")
        f.write(f"{n} {masa_particula:.17g} {hfact} {polyk:.17g}\n")
        for i in range(n):
            f.write(f"{float(pos[i,0]):.17g} {float(pos[i,1]):.17g} {float(pos[i,2]):.17g} "
                     f"{float(vel[i,0]):.17g} {float(vel[i,1]):.17g} {float(vel[i,2]):.17g} "
                     f"{float(h_guess[i]):.17g}\n")

    return dict(ruta=ruta_salida, n=n, masa_particula=masa_particula, seed=seed,
                r_mean=float(r.mean()), r_std=float(r.std()))


if __name__ == "__main__":
    import sys
    print("Uso como módulo (ver NULL1_piloto_distribucion_radial_CS.md para el orquestador del piloto).")
    sys.exit(0)
