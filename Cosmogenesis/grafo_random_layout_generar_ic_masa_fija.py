"""
grafo_random_layout_generar_ic_masa_fija.py — variante de MASA TOTAL FIJA de
`grafo_random_layout_generar_ic.py` (script congelado — NO se toca, sólo se importa de él lo que ya
estaba validado: `generar_grafo_erdos_renyi`, `contar_aristas_malla_real`).

Por qué existe (encargo de Alexis, 7-ago-2026, sobre el hallazgo de `MISTERIO_N500_vs_N2000_CS.md`):
  El generador original hace crecer el lado de la caja con `lado = n ** (1.0/3.0)` (línea 116 del
  original) y usa SIEMPRE la masa por partícula real del pool físico (`masa_bar.mean()`, ronda 9.4 sin
  importar N). Consecuencia verificada en `MISTERIO_N500_vs_N2000_CS.md`: a N=500 la masa total del
  sistema fue 4700, a N=2000 fue 18800 — razón EXACTA 0.2655 ≈ N500/N2000. Es decir: bajar N no sólo
  baja la resolución (menos partículas resolviendo el mismo sistema) — también baja la masa TOTAL
  absoluta, así que N=500 y N=2000 terminan siendo dos sistemas físicos DISTINTOS (menos modos de
  perturbación super-Jeans disponibles a menor masa), no el mismo sistema muestreado más fino. Ese es un
  confound de física real mezclado con el efecto de resolución que se quería aislar.

Qué cambia acá (y nada más):
  1. `lado` deja de depender de `n` — se fija en `LADO_FIJO`, el mismo valor que el generador ORIGINAL
     calcula para N=2000 (`2000 ** (1/3)`), para CUALQUIER N. La caja física ya no se achica a menor N.
  2. La masa por partícula deja de heredarse del pool físico (`masa_bar.mean()`, ~9.4 fijo) — se fuerza
     `masa_particula = MASA_TOTAL_OBJETIVO / n`, con `MASA_TOTAL_OBJETIVO=18800` (la masa total real y
     ya verificada de `bateria_n2000/ic_real`, ver `MISTERIO_N500_vs_N2000_CS.md` y
     `NULL0_masa_total_verificacion_CS.md`). Así, `n * masa_particula == MASA_TOTAL_OBJETIVO` para
     cualquier N por construcción — ya no escala con N.

Qué NO cambia: la construcción del grafo Erdős-Rényi (importada tal cual del original), `layout_resortes`
(mismos parámetros/semilla), la dilatación estática (`Expansion`, mismos 60 pasos), el campo de velocidad
turbulento opcional, el formato de escritura ASCII (mismo que lee `setup_cosmogenesis.f90`).

Consecuencia esperada en `h` (longitud de suavizado, la pregunta que pidió Alexis documentar con
precisión): el `h` que escribe este script en el IC (`h_guess = hfact` uniforme) es sólo una SEMILLA
para el solver grad-h nativo de Phantom — el `h` físico real emerge de la propia iteración de Phantom
(h = hfact·(m/ρ)^(1/3), ver `fase1_traducir_a_phantom.py`) una vez que arranca la corrida, no de este
script. Con masa total FIJA y caja FIJA: a menor N, la densidad numérica de muestreo baja (menos
partículas en el MISMO volumen) pero la densidad de MASA por unidad de volumen también baja (mismo
volumen, misma masa total, repartida entre menos partículas — cada partícula pesa más pero hay menos
partículas por unidad de volumen para promediarla). El resultado neto esperado es que el `h` de
equilibrio que reporte Phantom sea MAYOR a menor N — igual que con el generador original — pero ahora esa
subida de `h` refleja SÓLO "menos partículas resolviendo el mismo sistema" (resolución pura), porque el
sistema físico (caja + masa total) ya no cambia con N. Esto se mide después, leyendo el dump con
`leer_volcado_phantom.py` (no se puede afirmar el valor exacto de antemano — Phantom decide `h` vía su
propio solver, no este script).

No toca ningún archivo congelado ni ninguna carpeta de piloto/batería anterior — sólo importa/lee.
"""
import numpy as np

from cs073_cierre_holistico import T0, _T_reloj
from cs072_modulos.piezas.p_expansion import Expansion
from cs072_modulos.piezas.p_semilla_causal import layout_resortes
from fase1_traducir_a_phantom import HFACT, POLYK
from grafo_random_layout_generar_ic import generar_grafo_erdos_renyi  # importado tal cual, NO tocado

N_REFERENCIA_LADO = 2000  # el N de la batería original cuyo lado de caja se congela para todo N
LADO_FIJO = float(N_REFERENCIA_LADO) ** (1.0 / 3.0)  # MISMA fórmula que el original, evaluada una vez
MASA_TOTAL_OBJETIVO = 18800.0  # masa total (gas+sumideros) de bateria_n2000/ic_real, verificada en
                                 # MISTERIO_N500_vs_N2000_CS.md y NULL0_masa_total_verificacion_CS.md


def generar_control_random_masa_fija(n, dens_bar, n_aristas, seed_random, seed_layout=None,
                                      iters_layout=100, n_pasos_expansion=60, vel_generador=None,
                                      hfact=HFACT, polyk=POLYK,
                                      masa_total_objetivo=MASA_TOTAL_OBJETIVO, lado_fijo=LADO_FIJO,
                                      ruta_salida="grafo_random_ic_masa_fija.txt"):
    """Igual construcción que `generar_control_random` del original (grafo Erdős-Rényi G(n,n_aristas)
    -> layout_resortes -> dilatación estática -> velocidad turbulenta opcional -> escritura ASCII), con
    los DOS cambios documentados arriba: `lado` fijo (no depende de n) y masa por partícula derivada de
    `masa_total_objetivo / n` (no heredada del pool físico).

    `dens_bar` se recibe sólo para alimentar `vel_generador` (el campo turbulento pesa por densidad
    relativa) — ya NO determina la masa ni la caja, a diferencia del original."""
    assert len(dens_bar) == n, f"dens_bar (len={len(dens_bar)}) no coincide con n={n}"
    if seed_layout is None:
        seed_layout = seed_random

    adj_random, edge_set, intentos_rechazo = generar_grafo_erdos_renyi(n, n_aristas, seed=seed_random)

    pos = layout_resortes(adj_random, n, lado=lado_fijo, iters=iters_layout, seed=seed_layout)

    # MISMA dilatación isótropa estática que toda la jerarquía — depende sólo de n_pasos_expansion y del
    # reloj, nunca de seed_random/seed_layout/n.
    expansion = Expansion(T0=T0)
    for step in range(n_pasos_expansion):
        expansion.paso_de_estiramiento(_T_reloj(step))
    a_final = expansion._a_prev
    pos = pos * a_final

    vel = vel_generador(pos, adj_random, dens_bar) if vel_generador is not None else np.zeros_like(pos)
    h_guess = np.full(n, hfact)  # sólo semilla inicial para el solver grad-h de Phantom, ver docstring
    masa_particula = masa_total_objetivo / n

    with open(ruta_salida, "w") as f:
        f.write(f"# cosmogenesis_ic v2 CONTROL grafo-random MASA FIJA (variante de "
                f"grafo_random_layout_generar_ic.py -- lado_caja_fijo={lado_fijo:.6f} "
                f"(=N_referencia={N_REFERENCIA_LADO}^(1/3), NO depende de n) "
                f"masa_total_objetivo={masa_total_objetivo:.6g} (NO escala con n)) -- "
                f"npart={n} n_aristas={n_aristas} masa_particula={masa_particula:.17g} hfact={hfact} "
                f"polyk={polyk:.17g} seed_random={seed_random} seed_layout={seed_layout}\n")
        f.write(f"{n} {masa_particula:.17g} {hfact} {polyk:.17g}\n")
        for i in range(n):
            f.write(f"{float(pos[i, 0]):.17g} {float(pos[i, 1]):.17g} {float(pos[i, 2]):.17g} "
                    f"{float(vel[i, 0]):.17g} {float(vel[i, 1]):.17g} {float(vel[i, 2]):.17g} "
                    f"{float(h_guess[i]):.17g}\n")

    return dict(ruta=ruta_salida, n=n, pos=pos, masa_particula=masa_particula,
                masa_total=masa_particula * n, lado=lado_fijo, a_final=a_final,
                seed_random=seed_random, seed_layout=seed_layout, n_aristas=n_aristas,
                grado_medio=2.0 * n_aristas / n, intentos_rechazo=intentos_rechazo, adj=adj_random)


if __name__ == "__main__":
    print("Uso como módulo. Ver grafo_random_masa_fija_generar.py / grafo_random_masa_fija_correr.py.")
