"""
null3_generar_ic.py — NULL-3 (double-edge-swap con filtro geométrico de longitud), Fase II CS073,
escalón 3 de 6.

Continúa `null3_investigacion_preliminar.py` (diagnóstico SOLO de grafo: grado exacto preservado,
KS(L,L_real)=0.004 con tol_relativa=0.2, 12.5% de aristas cambiadas -- ya verificado, no repetido acá,
sólo importado). Este módulo añade el paso que faltaba: pasar el grafo NULL-3 por `layout_resortes`
(la MISMA función Fruchterman-Reingold que usa `fase1_traducir_a_phantom.traducir_pool` para REAL/
NULL-1-grafo/NULL-2, importada tal cual, NO reescrita) para obtener posiciones físicas 3D, y escribir
la condición inicial de Phantom en el mismo formato que toda la jerarquía.

Por qué esto es un módulo nuevo y no una edición de `fase1_traducir_a_phantom.py` (congelado, prohibido
tocar): `traducir_pool` sólo conoce DOS operaciones sobre la malla (idéntica, o `barajar_aristas` sin
restricción de longitud vía `seed_null`). NULL-3 necesita una TERCERA operación (`barajar_aristas_
preservando_longitud`, que además necesita `pos_referencia` -- un argumento que `traducir_pool` no
tiene). Se duplica aquí la FORMA del pipeline (mismo orden: malla -> barajado -> layout -> expansión
estática -> velocidad -> h uniforme -> escritura ASCII), no la lógica de cada paso (todos los pasos se
IMPORTAN de las piezas ya congeladas/validadas: `p_semilla_causal.malla_causal_atomos`/`layout_resortes`,
`p_expansion.Expansion`, `null3_investigacion_preliminar.barajar_aristas_preservando_longitud`).
Mismo patrón que ya usó `null2_zeldovich_generar_ic.py` para NULL-2 (pipeline paralelo, piezas
reutilizadas, ningún archivo congelado tocado).

`pos_referencia`: las posiciones REAL YA ESCRITAS en disco (post-layout, post-expansión -- el mismo
archivo que usa el resto de la jerarquía como REAL de referencia) -- se usan SÓLO para medir la
longitud de cada arista candidata en `barajar_aristas_preservando_longitud`, nunca para mover ninguna
partícula del propio NULL-3 (el layout de NULL-3 vuelve a arrancar de una posición aleatoria y se
relaja con Fruchterman-Reingold, exactamente como REAL/NULL-2-grafo).

No toca `p_semilla_causal.py`, `fase1_traducir_a_phantom.py`, `null3_investigacion_preliminar.py`, ni
ningún archivo de `bateria_n2000/`/`bateria_null1_n2000/`/`bateria_null2_n2000/`/
`bateria_real_extra_n2000/`/`piloto_null2_zeldovich/`/`piloto_null1/` -- sólo los importa/lee.
"""
import numpy as np

from cs073_cierre_holistico import T0, TASA_EXPANSION, _T_reloj
from cs072_modulos.piezas.p_expansion import Expansion
from cs072_modulos.piezas.p_semilla_causal import malla_causal_atomos, layout_resortes
from fase1_traducir_a_phantom import HFACT, POLYK
from null3_investigacion_preliminar import barajar_aristas_preservando_longitud


def generar_null3(masa_bar, dens_bar, pos_referencia, seed_null3, seed_layout=12345,
                   D_causal=3, k_causal=4, iters_layout=100, tol_relativa=0.2, factor_swaps=10,
                   n_pasos_expansion=60, vel_generador=None, hfact=HFACT, polyk=POLYK,
                   ruta_salida="null3_ic.txt"):
    """Genera UNA condición inicial NULL-3: misma malla causal REAL (malla_causal_atomos, MISMOS
    D_causal/k_causal/seed_ejes=2000 que traducir_pool), aristas barajadas por double-edge-swap CON
    filtro geométrico de longitud (`seed_null3` distingue la realización del barajado, análogo a
    `seed_null` de traducir_pool), layout de resortes idéntico en método a REAL (misma función, misma
    `seed_layout`, mismos `iters_layout`), MISMA dilatación isótropa estática que traducir_pool
    (Expansion, n_pasos_expansion=60, a_final=√60≈7.75, no depende de la semilla), mismo campo de
    velocidad turbulento opcional, h inicial uniforme. Devuelve dict con pos/diagnóstico del swap
    (grado preservado, % aristas distintas, KS de longitud) para poder reportarlo junto al perfil
    radial en el mismo log."""
    n = len(masa_bar)
    assert len(dens_bar) == n == len(pos_referencia), (
        f"masa_bar(n={n}) / dens_bar(n={len(dens_bar)}) / pos_referencia(n={len(pos_referencia)}) "
        "no coinciden -- ¿pool distinto del que generó pos_referencia?")

    lado = float(n) ** (1.0 / 3.0)
    adj_real, _m = malla_causal_atomos(dens_bar, D=D_causal, k=k_causal, seed_ejes=2000)
    adj_null3, aceptados, intentos = barajar_aristas_preservando_longitud(
        adj_real, n, pos_referencia, seed=seed_null3, factor_swaps=factor_swaps,
        tol_relativa=tol_relativa)

    pos = layout_resortes(adj_null3, n, lado=lado, iters=iters_layout, seed=seed_layout)

    # MISMA dilatación isótropa estática que traducir_pool (idéntica en REAL/NULL-1-grafo/NULL-2/NULL-3
    # por construcción -- depende sólo de n_pasos_expansion y del reloj, nunca de seed_null3/seed_layout).
    expansion = Expansion(T0=T0)
    for step in range(n_pasos_expansion):
        expansion.paso_de_estiramiento(_T_reloj(step))
    a_final = expansion._a_prev
    pos = pos * a_final

    vel = vel_generador(pos, adj_null3, dens_bar) if vel_generador is not None else np.zeros_like(pos)
    h_guess = np.full(n, hfact)
    masa_media = float(masa_bar.mean())

    with open(ruta_salida, "w") as f:
        f.write(f"# cosmogenesis_ic v2 NULL-3 (double-edge-swap + filtro geometrico de longitud, "
                 f"tol_relativa={tol_relativa}) -- npart={n} masa_particula={masa_media:.17g} "
                 f"hfact={hfact} polyk={polyk:.17g} seed={seed_null3}\n")
        f.write(f"{n} {masa_media:.17g} {hfact} {polyk:.17g}\n")
        for i in range(n):
            f.write(f"{float(pos[i,0]):.17g} {float(pos[i,1]):.17g} {float(pos[i,2]):.17g} "
                     f"{float(vel[i,0]):.17g} {float(vel[i,1]):.17g} {float(vel[i,2]):.17g} "
                     f"{float(h_guess[i]):.17g}\n")

    return dict(ruta=ruta_salida, n=n, pos=pos, masa_particula=masa_media, a_final=a_final,
                seed_null3=seed_null3, seed_layout=seed_layout,
                swap_aceptados=aceptados, swap_intentos=intentos)


if __name__ == "__main__":
    print("Uso como módulo. Ver null3_paso1_verificar_perfil_radial.py / null3_piloto_generar.py.")
