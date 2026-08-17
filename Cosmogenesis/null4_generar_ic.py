"""
null4_generar_ic.py — NULL-4 (topología COMPLETA idéntica a REAL, orden temporal de formación
rebarajado), Fase II CS073, escalón 5 de 6.

Continúa `null4_verificar_invarianza_orden.py` (ya en disco, NO tocado por este módulo — sólo se le
IMPORTAN las dos funciones de reconstrucción que ya usó para responder la pregunta previa: ¿el layout
final depende del orden de inserción de las aristas en la malla causal? Respuesta empírica ya obtenida
ahí: SÍ, de forma no trivial — peor diferencia = 38% de la escala típica de coordenada. Ese resultado es
lo que hace a NULL-4 operacionalizable: si el orden no hubiera importado, NULL-4 habría colapsado a ser
idéntico a REAL por construcción).

Qué es NULL-4 (a diferencia de NULL-1/2/3, que cambian la TOPOLOGÍA — ángulo, espectro de potencia,
motivos — dejando algo del proceso de formación intacto): NULL-4 deja la topología EXACTA de la malla
causal REAL sin tocar un solo enlace — el mismo `adj_real` que ve `traducir_pool` con `seed_null=None` —
y en cambio destruye el ORDEN en que esas aristas se insertaron en la estructura de adyacencia antes de
correr `layout_resortes`. La pregunta que responde: ¿la HISTORIA de cómo se construyó la malla (no sólo
su forma final como conjunto) importa para que nazcan sumideros?

Reusa (sin reescribir) tres piezas ya congeladas/validadas:
  - `malla_causal_atomos`/`layout_resortes` de `cs072_modulos/piezas/p_semilla_causal.py` (congelado).
  - `adj_a_lista_aristas`/`construir_adj_en_orden` de `null4_verificar_invarianza_orden.py` (ya en disco,
    ya usado para verificar empíricamente la dependencia de orden — se importa tal cual, no se toca).
  - la FORMA del pipeline downstream (dilatación estática `Expansion`, campo de velocidad turbulento
    opcional, escritura ASCII v2) — mismo patrón que `null3_generar_ic.py` y la plantilla de
    `grafo_random_layout_generar_ic_masa_fija.py` (mismo orden de pasos: malla -> [operación NULL] ->
    layout -> expansión -> velocidad -> h uniforme -> escritura).

No toca `p_semilla_causal.py`, `null4_verificar_invarianza_orden.py`, `fase1_traducir_a_phantom.py`, ni
ningún archivo de `bateria_n2000/`/`bateria_null1_n2000/`/`bateria_null2_n2000/`/`bateria_null3_n2000/`/
`bateria_real_extra_n2000/` — sólo los importa/lee.
"""
import numpy as np

from cs073_cierre_holistico import T0, _T_reloj
from cs072_modulos.piezas.p_expansion import Expansion
from cs072_modulos.piezas.p_semilla_causal import malla_causal_atomos, layout_resortes
from fase1_traducir_a_phantom import HFACT, POLYK
from null4_verificar_invarianza_orden import adj_a_lista_aristas, construir_adj_en_orden

SEED_EJES_MALLA = 2000  # mismo seed_ejes que traducir_pool/null3_generar_ic para reconstruir la malla REAL


def generar_null4(masa_bar, dens_bar, seed_reorden, seed_layout=12345,
                   D_causal=3, k_causal=4, iters_layout=100, n_pasos_expansion=60,
                   vel_generador=None, hfact=HFACT, polyk=POLYK,
                   ruta_salida="null4_ic.txt"):
    """Genera UNA condición inicial NULL-4: misma malla causal REAL exacta (mismo CONJUNTO de aristas,
    `malla_causal_atomos` con los mismos D_causal/k_causal/seed_ejes=2000 que usa toda la jerarquía —
    NINGÚN enlace cambia), pero la secuencia de INSERCIÓN de esas aristas en `adj` se rebaraja al azar
    según `seed_reorden` (permutación de índices, `construir_adj_en_orden` de
    `null4_verificar_invarianza_orden.py`) antes de correr `layout_resortes` con el mismo `seed_layout`
    de siempre. La única diferencia con REAL es esa: el orden de construcción — verificado por conjunto
    (assert) que el grafo final es topológicamente idéntico a REAL, no una aproximación."""
    n = len(masa_bar)
    assert len(dens_bar) == n, f"masa_bar(n={n}) / dens_bar(n={len(dens_bar)}) no coinciden"

    lado = float(n) ** (1.0 / 3.0)
    adj_real, _m = malla_causal_atomos(dens_bar, D=D_causal, k=k_causal, seed_ejes=SEED_EJES_MALLA)
    lista_aristas = adj_a_lista_aristas(adj_real, n)
    n_aristas = len(lista_aristas)

    rng = np.random.default_rng(seed_reorden)
    orden = rng.permutation(n_aristas).tolist()
    adj_null4 = construir_adj_en_orden(lista_aristas, n, orden)

    # verificación de que el CONJUNTO de aristas es IDÉNTICO al de REAL (topología completa preservada,
    # la propiedad que define a NULL-4 frente a NULL-1/2/3) -- no basta con el número, se compara la
    # lista canónica ordenada completa.
    lista_null4 = sorted(adj_a_lista_aristas(adj_null4, n))
    assert lista_null4 == sorted(lista_aristas), (
        "¡el conjunto de aristas de NULL-4 difiere del de REAL! -- construir_adj_en_orden debería "
        "preservar el conjunto exacto, sólo cambiar el orden de inserción.")

    pos = layout_resortes(adj_null4, n, lado=lado, iters=iters_layout, seed=seed_layout)

    # MISMA dilatación isótropa estática que toda la jerarquía (REAL/NULL-1/NULL-2/NULL-3) -- depende
    # sólo de n_pasos_expansion y del reloj, nunca de seed_reorden/seed_layout.
    expansion = Expansion(T0=T0)
    for step in range(n_pasos_expansion):
        expansion.paso_de_estiramiento(_T_reloj(step))
    a_final = expansion._a_prev
    pos = pos * a_final

    vel = vel_generador(pos, adj_null4, dens_bar) if vel_generador is not None else np.zeros_like(pos)
    h_guess = np.full(n, hfact)
    masa_media = float(masa_bar.mean())

    with open(ruta_salida, "w") as f:
        f.write(f"# cosmogenesis_ic v2 NULL-4 (topologia completa idéntica a REAL, orden de insercion "
                 f"de aristas rebarajado antes de layout_resortes) -- npart={n} "
                 f"masa_particula={masa_media:.17g} hfact={hfact} polyk={polyk:.17g} "
                 f"seed_reorden={seed_reorden}\n")
        f.write(f"{n} {masa_media:.17g} {hfact} {polyk:.17g}\n")
        for i in range(n):
            f.write(f"{float(pos[i,0]):.17g} {float(pos[i,1]):.17g} {float(pos[i,2]):.17g} "
                     f"{float(vel[i,0]):.17g} {float(vel[i,1]):.17g} {float(vel[i,2]):.17g} "
                     f"{float(h_guess[i]):.17g}\n")

    return dict(ruta=ruta_salida, n=n, pos=pos, masa_particula=masa_media, a_final=a_final,
                seed_reorden=seed_reorden, seed_layout=seed_layout, n_aristas=n_aristas)


if __name__ == "__main__":
    print("Uso como módulo. Ver null4_bateria_generar.py.")
