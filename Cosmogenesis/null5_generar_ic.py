"""
null5_generar_ic.py — NULL-5 (topología COMPLETA de REAL intacta, CONJUNTO de posiciones finales de REAL
intacto, pero la correspondencia nodo-del-grafo <-> posición física destruida), Fase II CS073, escalón
6 de 6 (el último).

Pregunta que responde: ¿importa que el nodo A (con sus vecinos causales específicos) haya terminado
EXACTAMENTE en la posición X que la física real (`layout_resortes` sobre la malla causal REAL) le
asignó, o alcanza con que "algún nodo cualquiera" esté en esa posición y "algún otro" en la de al lado,
sin que la identidad de sus conexiones tenga que ver con dónde cayó?

Verificación previa OBLIGATORIA (ver `null5_verificar_colapso.py`, corrida antes de escribir este
módulo) sobre si la pregunta es operacionalizable sin colapsar trivial -- resultado: NO lo es, en este
pipeline. Se deja documentado aquí también, junto al código, para que quien lea este archivo no necesite
ir a buscar el otro:

  1. La masa por partícula es UNIFORME (un solo `masa_media` global en la cabecera del IC, no una
     columna por partícula) en TODA la jerarquía CS073 (REAL/NULL-1/2/3/4) -- confirmado leyendo
     `fase1_traducir_a_phantom.py` línea 109 y `null4_generar_ic.py` línea 86: `masa_bar.mean()`, nunca
     `masa_bar[i]`. Ninguna partícula "recuerda" su masa individual real extraída del pool -- da igual
     qué nodo ocupe cuál posición, la masa escrita es la misma constante para las 2000.
  2. `h` (longitud de suavizado) también se escribe UNIFORME (`hfact=1.2` para las 2000 partículas) --
     es sólo una semilla para el solver grad-h nativo de Phantom, que la reemplaza en tiempo de corrida
     por el `h` de equilibrio real (función de la densidad LOCAL de partículas vecinas en el espacio,
     no de qué nodo del grafo causal fue cada partícula).
  3. La velocidad (`campo_velocidad_turbulento.factory`) es una función PURA de la posición final --
     la propia factory documenta "adj/dens_bar se ignoran a propósito" y ni siquiera los usa dentro de
     `_gen(pos, adj, dens_bar)`. Verificado empíricamente (no sólo leyendo el código) en
     `null5_verificar_colapso.py`: recomputar `campo_turbulento` únicamente a partir de las posiciones
     del archivo IC real reproduce la columna de velocidad del archivo real BIT A BIT (diff máxima =
     0.0), confirmando que no hay ninguna dependencia oculta de identidad de nodo.
  4. La topología (`adj`) sólo se usa para construir el layout (paso ya completado en REAL) -- el
     formato de IC que lee Phantom (posición/velocidad/masa/h por partícula) no contiene NINGÚN campo
     de identidad de nodo ni de adyacencia. Una vez escritas las posiciones, el grafo se descarta -- ni
     `phantomsetup` ni `phantom` lo vuelven a leer.

Consecuencia: como NINGÚN atributo físico escrito en el IC (masa/h/velocidad) depende de la identidad
del nodo -- sólo del VALOR de la posición final, que es igual sea cual sea el nodo que "la tenga" --
permutar qué nodo ocupa cuál posición NO puede cambiar ningún valor físico de ninguna partícula. Lo
único que la permutación puede cambiar es el ORDEN DE FILA en que esas tuplas (posición, velocidad,
masa, h) -- idénticas en contenido a las de REAL -- se escriben en el archivo ASCII. Este módulo
construye exactamente eso: toma la malla causal REAL exacta + el layout REAL exacto (reconstrucción
bit-idéntica, ver assert de comparación contra el archivo real en `null5_verificar_colapso.py`), aplica
una permutación de FILA (`seed_permutacion`) y escribe el resultado. La verificación de que el archivo
resultante es una permutación de fila EXACTA del archivo REAL (mismo multiset de tuplas) se hace con un
assert antes de escribir.

No toca ningún archivo congelado ni ninguna carpeta de batería anterior (`bateria_n2000/`,
`bateria_null1/2/3/4_n2000/`, `bateria_real_extra_n2000/`, `p_semilla_causal.py`,
`grafo_random_layout_generar_ic_masa_fija.py`, `leer_volcado_phantom.py`, `null4_generar_ic.py`,
`null4_verificar_invarianza_orden.py`) -- sólo los importa/lee.
"""
import numpy as np

from cs073_cierre_holistico import T0, _T_reloj
from cs072_modulos.piezas.p_expansion import Expansion
from cs072_modulos.piezas.p_semilla_causal import malla_causal_atomos, layout_resortes
from fase1_traducir_a_phantom import HFACT, POLYK

SEED_EJES_MALLA = 2000     # mismo seed_ejes que traducir_pool/null3/null4 para reconstruir la malla REAL
SEED_LAYOUT_REAL = 12345   # misma semilla de layout que toda la jerarquía


def reconstruir_malla_y_layout_real(dens_bar, D_causal=3, k_causal=4, iters_layout=100,
                                     n_pasos_expansion=60, seed_layout=SEED_LAYOUT_REAL):
    """Reconstruye la malla causal REAL exacta y el layout físico final REAL exacto (posiciones YA
    dilatadas por la expansión estática) -- mismo método/parámetros que `null4_generar_ic.py` y
    `traducir_pool`. Verificado (en `null5_verificar_colapso.py`) que esto reproduce
    `bateria_n2000/ic_real/cosmogenesis_ic.txt` BIT A BIT (diff máxima = 0.0) antes de construir NULL-5
    sobre este resultado."""
    n = len(dens_bar)
    lado = float(n) ** (1.0 / 3.0)
    adj_real, _m = malla_causal_atomos(dens_bar, D=D_causal, k=k_causal, seed_ejes=SEED_EJES_MALLA)
    pos = layout_resortes(adj_real, n, lado=lado, iters=iters_layout, seed=seed_layout)

    expansion = Expansion(T0=T0)
    for step in range(n_pasos_expansion):
        expansion.paso_de_estiramiento(_T_reloj(step))
    a_final = expansion._a_prev
    pos = pos * a_final
    return adj_real, pos, a_final


def generar_null5(masa_bar, dens_bar, seed_permutacion, seed_layout=SEED_LAYOUT_REAL,
                   D_causal=3, k_causal=4, iters_layout=100, n_pasos_expansion=60,
                   vel_generador=None, hfact=HFACT, polyk=POLYK,
                   ruta_salida="null5_ic.txt"):
    """Genera UNA condición inicial NULL-5: misma malla causal REAL exacta y mismo layout físico final
    REAL exacto (reconstrucción bit-idéntica, ver `reconstruir_malla_y_layout_real`), pero permuta al
    azar (según `seed_permutacion`) qué FILA del archivo de salida recibe cuál posición -- es decir, qué
    "nodo" (índice de partícula 0..n-1) queda asociado a cuál de las posiciones finales de REAL. La
    velocidad se recalcula a partir de la posición YA permutada (pura función de posición, ver docstring
    del módulo) -- por construcción reproduce el mismo VALOR que tenía esa posición en REAL, sólo que
    ahora en la fila del nodo permutado. Un assert compara el MULTISET completo de tuplas
    (pos,vel,masa,h) de NULL-5 contra el de REAL (recalculado con la MISMA malla/layout/velocidad, orden
    natural) -- deben ser exactamente iguales como conjunto, sólo el orden de fila puede diferir; si no
    lo son, algo del pipeline introdujo una dependencia de identidad de nodo no documentada y hay que
    investigarla antes de seguir."""
    n = len(masa_bar)
    assert len(dens_bar) == n, f"masa_bar(n={n}) / dens_bar(n={len(dens_bar)}) no coinciden"

    adj_real, pos_real, a_final = reconstruir_malla_y_layout_real(
        dens_bar, D_causal=D_causal, k_causal=k_causal, iters_layout=iters_layout,
        n_pasos_expansion=n_pasos_expansion, seed_layout=seed_layout)

    rng = np.random.default_rng(seed_permutacion)
    permutacion = rng.permutation(n)   # permutacion[i] = qué nodo de REAL ocupa la fila i de NULL-5
    pos_null5 = pos_real[permutacion]

    vel_real = vel_generador(pos_real, adj_real, dens_bar) if vel_generador is not None else np.zeros_like(pos_real)
    vel_null5 = vel_generador(pos_null5, adj_real, dens_bar) if vel_generador is not None else np.zeros_like(pos_null5)

    h_guess = np.full(n, hfact)
    masa_media = float(masa_bar.mean())

    # --- verificación de colapso trivial: multiset (pos,vel) de NULL-5 == multiset (pos,vel) de REAL ---
    tabla_real = np.round(np.hstack([pos_real, vel_real]), decimals=8)
    tabla_null5 = np.round(np.hstack([pos_null5, vel_null5]), decimals=8)
    orden_real = np.lexsort(tabla_real.T[::-1])
    orden_null5 = np.lexsort(tabla_null5.T[::-1])
    assert np.array_equal(tabla_real[orden_real], tabla_null5[orden_null5]), (
        "¡el multiset de (posicion,velocidad) de NULL-5 difiere del de REAL! -- esto significaría que "
        "algún atributo SÍ depende de la identidad de nodo, contradiciendo la verificación previa de "
        "null5_verificar_colapso.py. Investigar antes de escribir el IC.")

    with open(ruta_salida, "w") as f:
        f.write(f"# cosmogenesis_ic v2 NULL-5 (topologia + conjunto de posiciones finales IDENTICOS a "
                 f"REAL, correspondencia nodo<->posicion permutada -- ver null5_generar_ic.py para la "
                 f"verificacion de colapso trivial) -- npart={n} masa_particula={masa_media:.17g} "
                 f"hfact={hfact} polyk={polyk:.17g} seed_permutacion={seed_permutacion}\n")
        f.write(f"{n} {masa_media:.17g} {hfact} {polyk:.17g}\n")
        for i in range(n):
            f.write(f"{float(pos_null5[i,0]):.17g} {float(pos_null5[i,1]):.17g} {float(pos_null5[i,2]):.17g} "
                     f"{float(vel_null5[i,0]):.17g} {float(vel_null5[i,1]):.17g} {float(vel_null5[i,2]):.17g} "
                     f"{float(h_guess[i]):.17g}\n")

    return dict(ruta=ruta_salida, n=n, pos=pos_null5, masa_particula=masa_media, a_final=a_final,
                seed_permutacion=seed_permutacion, seed_layout=seed_layout,
                n_aristas=sum(len(v) for v in adj_real.values()) // 2)


if __name__ == "__main__":
    print("Uso como módulo. Ver null5_verificar_colapso.py y null5_bateria_generar.py.")
