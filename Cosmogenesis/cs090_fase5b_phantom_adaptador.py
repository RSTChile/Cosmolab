"""
cs090_fase5b_phantom_adaptador.py — FASE V-B: adaptador A2-B0-C2 -> condiciones iniciales de Phantom
=========================================================================================================
QUIÉN SOY: primer archivo de la Fase V-B (piloto de validación física con Phantom sobre el candidato
A2-B0-C2 de Fase V-A). Alexis autorizó explícitamente correr Phantom para esta línea de investigación
(10-ago-2026) — es la primera vez que se corre este pipeline concreto (grafo A2-B0-C2 → Phantom), así
que el costo real por corrida se mide en `cs090_fase5b_correr.py`, no se asume aquí.

Qué hace este archivo (dos piezas, cada una reusa lo ya validado, ninguna reimplementa desde cero):

  1. `reconstruir_regla_a2b0c2(seed, N, n_sweeps)` — reconstruye, BIT A BIT, el grafo final de una regla
     A2-B0-C2 ya generada y clasificada en `FASE5A_profundizar_A2B0C2_resultado_CS.md`
     (`cs090_fase5_profundizar_a2b0c2_resumen.csv`), usando SÓLO su columna `seed` — reproduce
     exactamente lo que hizo `cs090_fase5_motor.correr_regla_coarse()` (líneas 433-436: mismo `p` vía
     `cs090_fase5_generador.generar_regla`, mismo rng derivado de `seed*5000+N`, mismo
     `construir_A2`+`dinamica_B0`+`medir`). `cs090_fase5_generador.py` y `cs090_fase5_motor.py` se
     IMPORTAN tal cual — no se tocan ni se copian sus funciones.

  2. `generar_ic_masa_fija_desde_grafo(...)` — MISMA receta de MASA TOTAL FIJA que
     `grafo_random_layout_generar_ic_masa_fija.generar_control_random_masa_fija` (script congelado, sólo
     se REUSAN sus constantes/decisiones: lado de caja fijo = 2000**(1/3) para cualquier N, masa por
     partícula = 18800/N, mismo `layout_resortes`, misma dilatación estática `Expansion` de 60 pasos,
     mismo campo de turbulencia opcional Mach=3/seed=42) — pero en vez de generar un grafo Erdős-Rényi
     nuevo, recibe el grafo YA CONSTRUIDO por (1): el grafo dinámico co-emergente A2-B0-C2 que
     `cs090_fase5_clasificador.py` ya clasificó como Clase I o Clase III, no un control random. Esta es
     la pieza central pedida por Alexis: el mismo patrón grafo→layout→masa fija, reusado sobre un grafo
     de origen distinto.

Ningún script congelado se modifica (`cs090_fase5_generador.py`, `cs090_fase5_motor.py`,
`cs090_fase5_clasificador.py`, `grafo_random_layout_generar_ic_masa_fija.py`, `real_extra_generar_ic.py`,
`leer_volcado_phantom.py`) — todos sólo se importan. No corre Phantom (eso es `cs090_fase5b_correr.py`).
No declara cierre ni veredicto.
"""
from __future__ import annotations
import sys
import numpy as np

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)

import cs090_fase5_generador as GEN          # generar_regla() -- sólo import, no se toca
import cs090_fase5_motor as MOT              # construir_A2/dinamica_B0/medir -- sólo import, no se toca

from cs073_cierre_holistico import T0, _T_reloj
from cs072_modulos.piezas.p_expansion import Expansion
from cs072_modulos.piezas.p_semilla_causal import layout_resortes
from fase1_traducir_a_phantom import HFACT, POLYK
from campo_velocidad_turbulento import factory as factory_turb, MACH_OBJETIVO

# ---- constantes heredadas TAL CUAL de grafo_random_layout_generar_ic_masa_fija.py (mismos valores) ----
N_REFERENCIA_LADO = 2000
LADO_FIJO = float(N_REFERENCIA_LADO) ** (1.0 / 3.0)      # mismo lado para CUALQUIER N (no escala)
MASA_TOTAL_OBJETIVO = 18800.0                              # misma masa total objetivo que toda CS073

TURB_SEED = 42                                              # misma semilla de turbulencia que CS073
CS_SONORA = float(np.sqrt(POLYK))


# ============================================================================================
# 1) RECONSTRUIR el grafo final de una regla A2-B0-C2 ya clasificada, a partir de su `seed`
# ============================================================================================
def reconstruir_regla_a2b0c2(seed: int, N: int = 2000, n_sweeps: int = 14):
    """Reproduce bit a bit lo que corrió `cs090_fase5_motor.correr_regla_coarse(p, N=2000, ...)` para
    la regla A2-B0-C2 identificada por `seed` en `cs090_fase5_profundizar_a2b0c2_resumen.csv`: mismo
    `p` (generar_regla sólo depende de `seed`, el `idx` únicamente etiqueta el rule_id/descripción —
    verificado por inspección: `generar_regla` arma `rng = np.random.default_rng(seed)` y TODO el
    muestreo sale de ese rng, `idx` nunca se usa como fuente de aleatoriedad), mismo rng derivado de
    `seed*5000+N` (línea 433 de cs090_fase5_motor.py), mismo `construir_A2`+`dinamica_B0`+`medir`.

    Devuelve (p, m): `p` es el dict de parámetros de la regla (K,J,noise,meandeg,kcap,sim_thr_frac,...),
    `m` es el dict que devuelve `medir()` — el grafo final está en `m["adj_final"]` (lista de sets,
    índice = nodo, formato nativo de cs090_fase5_motor)."""
    p = GEN.generar_regla("A2", "B0", "C2", idx=0, seed=seed)
    rng = np.random.default_rng(p["seed"] * 5000 + N)
    sustrato = MOT.construir_A2(N, rng, p)
    sustrato = MOT.dinamica_B0(sustrato, p, rng, n_sweeps, "C2")
    m = MOT.medir(sustrato, p, rng)
    return p, m


# ============================================================================================
# 2) GENERAR condición inicial de Phantom, MASA FIJA, a partir de un grafo YA CONSTRUIDO
# ============================================================================================
def generar_ic_masa_fija_desde_grafo(adj_lista, N, seed_layout, ruta_salida,
                                      masa_total_objetivo=MASA_TOTAL_OBJETIVO, lado_fijo=LADO_FIJO,
                                      iters_layout=100, n_pasos_expansion=60,
                                      hfact=HFACT, polyk=POLYK, con_turbulencia=True,
                                      turb_seed=TURB_SEED):
    """MISMA receta que `generar_control_random_masa_fija` (grafo_random_layout_generar_ic_masa_fija.py,
    congelado): lado de caja FIJO (no depende de N), masa por partícula = masa_total_objetivo/N (no
    escala con N — el fix del confound caja/masa∝N de CS073, ver MISTERIO_N500_vs_N2000_CS.md), mismo
    `layout_resortes` + dilatación estática `Expansion` + campo de turbulencia Mach=3.

    Diferencia con el original: NO genera un grafo Erdős-Rényi nuevo — recibe `adj_lista` (lista de
    sets, formato nativo de cs090_fase5_motor: adj_lista[i] = vecinos del nodo i), el grafo YA
    CONSTRUIDO+EVOLUCIONADO+PODADO de una regla A2-B0-C2 (kind A2, con recableo co-emergente + límite de
    escala duro C2 ya aplicados por el motor de Fase V-A). `layout_resortes` espera un dict-like con
    `.get(i, ())` (ver p_semilla_causal.py línea 113) — se convierte `adj_lista` (indexable por entero,
    NO tiene `.get`) a dict antes de llamarlo; ninguna arista ni nodo se altera en esa conversión.

    `con_turbulencia=True` (default): usa el MISMO campo de turbulencia (Mach=3, espectro Burgers k^-2,
    seed=42) que toda la jerarquía CS073 — `campo_velocidad_turbulento.factory()._gen` IGNORA a propósito
    `adj`/`dens_bar` (verificado por inspección, ver docstring de ese módulo: "adj/dens_bar se ignoran a
    propósito"), así que no hace falta un `dens_bar` real del pool físico — se pasa un array de unos del
    tamaño correcto sólo para satisfacer la firma de la función, sin ningún efecto sobre el resultado."""
    adj_dict = {i: adj_lista[i] for i in range(N)}
    n_aristas = sum(len(v) for v in adj_lista) // 2

    pos = layout_resortes(adj_dict, N, lado=lado_fijo, iters=iters_layout, seed=seed_layout)

    # misma dilatación isótropa estática que TODA la jerarquía (REAL/NULL/grafo_random_masa_fija)
    expansion = Expansion(T0=T0)
    for step in range(n_pasos_expansion):
        expansion.paso_de_estiramiento(_T_reloj(step))
    a_final = expansion._a_prev
    pos = pos * a_final

    if con_turbulencia:
        vel_gen = factory_turb(CS_SONORA, seed=turb_seed, mach=MACH_OBJETIVO)
        vel = vel_gen(pos, adj_dict, np.ones(N))   # dens_bar dummy -- factory_turb la ignora (ver arriba)
    else:
        vel = np.zeros_like(pos)

    h_guess = np.full(N, hfact)
    masa_particula = masa_total_objetivo / N

    with open(ruta_salida, "w") as f:
        f.write(f"# cosmogenesis_ic v2 FASE V-B A2-B0-C2 (piloto Phantom) -- lado_caja_fijo="
                f"{lado_fijo:.6f} (=N_referencia={N_REFERENCIA_LADO}^(1/3), NO depende de n) "
                f"masa_total_objetivo={masa_total_objetivo:.6g} (NO escala con n) -- "
                f"npart={N} n_aristas={n_aristas} masa_particula={masa_particula:.17g} hfact={hfact} "
                f"polyk={polyk:.17g} seed_layout={seed_layout} con_turbulencia={con_turbulencia}\n")
        f.write(f"{N} {masa_particula:.17g} {hfact} {polyk:.17g}\n")
        for i in range(N):
            f.write(f"{float(pos[i, 0]):.17g} {float(pos[i, 1]):.17g} {float(pos[i, 2]):.17g} "
                    f"{float(vel[i, 0]):.17g} {float(vel[i, 1]):.17g} {float(vel[i, 2]):.17g} "
                    f"{float(h_guess[i]):.17g}\n")

    return dict(ruta=ruta_salida, n=N, masa_particula=masa_particula, masa_total=masa_particula * N,
                lado=lado_fijo, a_final=a_final, n_aristas=n_aristas, seed_layout=seed_layout,
                grado_medio=2.0 * n_aristas / N)


if __name__ == "__main__":
    print("Uso como módulo. Ver cs090_fase5b_generar_pares.py / cs090_fase5b_correr.py.")
    print("Smoke test rápido (N=200, una sola regla, sin escribir a phantom_cs073/):")
    p, m = reconstruir_regla_a2b0c2(seed=272702, N=200, n_sweeps=14)  # r9, Clase I, N chico p/smoke
    print(f"  regla reconstruida: K={p['K']} J={p['J']} noise={p['noise']} meandeg={p['meandeg']} "
          f"kcap={p['kcap']} -- n_aristas grafo final={m['n_aristas']} diam={m['diam']}")
    info = generar_ic_masa_fija_desde_grafo(m["adj_final"], N=200, seed_layout=12345,
                                             ruta_salida="/tmp/_smoke_fase5b_ic.txt")
    print(f"  IC escrito: {info}")
