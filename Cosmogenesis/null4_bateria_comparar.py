"""
null4_bateria_comparar.py — REAL (n=6, `bateria_n2000/ic_real` + `bateria_real_extra_n2000/`) vs las
3 NULL-4 (N=2000, semillas de reordenamiento 601-603) recién corridas, y NULL-4 vs NULL-1/NULL-2/NULL-3
(ambos ya en disco) -- Fase II CS073, escalón NULL-4.

`masa_y_n_sumideros` y `test_permutacion_2_grupos` se IMPORTAN tal cual de `null3_bateria_comparar.py`
(mismo test de permutación exacto ya validado, H1 pre-registrada de una cola REAL>NULL, sobre TODAS las
C(n_REAL+n_NULL, n_REAL) asignaciones posibles bajo H0) -- no se reimplementan, para no arriesgar una
divergencia silenciosa de la matemática entre escalones de la jerarquía.

No corre ninguna simulación. No toca ninguna carpeta de batería anterior (sólo lectura). No declara
cierre de CS073 ni de la jerarquía de 6 -- reporta números, la lectura es de Alexis.
"""
from pathlib import Path

import numpy as np

from null3_bateria_comparar import masa_y_n_sumideros, test_permutacion_2_grupos

BASE_REAL_ORIG = Path("/Users/alexis/phantom_cs073/bateria_n2000/ic_real")
BASE_REAL_EXTRA = Path("/Users/alexis/phantom_cs073/bateria_real_extra_n2000")
REAL_EXTRA_SEEDS = [301, 302, 303, 304, 305]
BASE_NULL1 = Path("/Users/alexis/phantom_cs073/bateria_null1_n2000")
NULL1_CARPETAS = [BASE_NULL1 / f"ic_null1_s{i}" for i in range(1, 9)]
BASE_NULL2 = Path("/Users/alexis/phantom_cs073/bateria_null2_n2000")
NULL2_SEEDS = list(range(401, 409))
NULL2_CARPETAS = [BASE_NULL2 / f"ic_null2_s{s}" for s in NULL2_SEEDS]
BASE_NULL3 = Path("/Users/alexis/phantom_cs073/bateria_null3_n2000")
NULL3_SEEDS = list(range(501, 509))
NULL3_CARPETAS = [BASE_NULL3 / f"ic_null3_s{s}" for s in NULL3_SEEDS]
BASE_NULL4 = Path("/Users/alexis/phantom_cs073/bateria_null4_n2000")
NULL4_SEEDS = [601, 602, 603]
NULL4_CARPETAS = [BASE_NULL4 / f"ic_null4_s{s}" for s in NULL4_SEEDS]


def _reportar_grupo(nombre, dicts):
    print(f"\n{nombre} (n={len(dicts)}):")
    for d in dicts:
        print(f"  {d['carpeta']:14s} masa_total={d['masa_total']:.1f}  n_sumideros={d['n_sumideros']}")
    masas = np.array([d["masa_total"] for d in dicts])
    de_str = f"{masas.std(ddof=1):.4f}" if len(masas) > 1 and masas.std(ddof=1) > 0 else "0.0000 (todas iguales o n=1)"
    print(f"  media={masas.mean():.4f}  DE={de_str}  min={masas.min():.1f}  max={masas.max():.1f}")
    return masas


def _reportar_comparacion(titulo, r):
    print("\n" + "-" * 92)
    print(titulo)
    print(f"    estadístico observado (media_{r['nombre_a']} - media_{r['nombre_b']}) = "
          f"{r['estadistico_observado']:.4f}")
    print(f"    C({r['n_a']}+{r['n_b']},{r['n_a']}) = {r['n_asignaciones']} asignaciones posibles bajo H0")
    print(f"    rank de la asignación observada = {r['rank']} de {r['n_asignaciones']}")
    print(f"    p (una cola, {r['nombre_a']}>{r['nombre_b']}) = {r['p_una_cola']:.6f}  "
          f"(piso teórico = 1/{r['n_asignaciones']} = {1/r['n_asignaciones']:.6f})")
    print(f"    p (dos colas) = {r['p_dos_colas']:.6f}")


def main():
    print("=" * 92)
    print("NULL-4 batería completa (N=2000, 3 semillas de reordenamiento 601-603) vs REAL (n=6), "
          "NULL-1 (n=8), NULL-2 (n=8), NULL-3 (n=8)")
    print("Observable: masa total acumulada en sumideros al final (mismo que toda la jerarquía CS073)")
    print("=" * 92)

    real_orig = masa_y_n_sumideros(BASE_REAL_ORIG)
    reales_extra = [masa_y_n_sumideros(BASE_REAL_EXTRA / f"ic_real_s{s}") for s in REAL_EXTRA_SEEDS]
    reales = [real_orig] + reales_extra
    nulls1 = [masa_y_n_sumideros(c) for c in NULL1_CARPETAS]
    nulls2 = [masa_y_n_sumideros(c) for c in NULL2_CARPETAS]
    nulls3 = [masa_y_n_sumideros(c) for c in NULL3_CARPETAS]
    nulls4 = [masa_y_n_sumideros(c) for c in NULL4_CARPETAS]

    masas_real = _reportar_grupo("REAL", reales)
    masas_null1 = _reportar_grupo("NULL-1 (ángulo isótropo aleatorio, mismo radio que REAL)", nulls1)
    masas_null2 = _reportar_grupo("NULL-2 (Zel'dovich, P(k) parcialmente preservado)", nulls2)
    masas_null3 = _reportar_grupo(
        "NULL-3 (double-edge-swap + filtro geométrico de longitud, grado+longitud preservados)", nulls3)
    masas_null4 = _reportar_grupo(
        "NULL-4 (topología COMPLETA idéntica a REAL, orden de inserción/formación rebarajado)", nulls4)

    r_a = test_permutacion_2_grupos(masas_real, masas_null4, nombre_a="REAL", nombre_b="NULL-4")
    _reportar_comparacion("(a) REAL (n=6) vs NULL-4 (n=3) -- test de permutación EXACTO", r_a)
    if masas_null4.std(ddof=1) > 0 if len(masas_null4) > 1 else False:
        z = (masas_real.mean() - masas_null4.mean()) / masas_null4.std(ddof=1)
        print(f"    z-score ((media_REAL - media_NULL4) / DE_NULL4) = {z:.3f}")
    else:
        print("    z-score: INDEFINIDO (DE_NULL4 = 0 o n=1).")

    r_b = test_permutacion_2_grupos(masas_null4, masas_null1, nombre_a="NULL-4", nombre_b="NULL-1")
    _reportar_comparacion(
        "(b) NULL-4 (n=3) vs NULL-1 (n=8) -- ¿preservar la topología COMPLETA (vs sólo ángulo) forma "
        "MÁS estructura?", r_b)

    r_c = test_permutacion_2_grupos(masas_null4, masas_null2, nombre_a="NULL-4", nombre_b="NULL-2")
    _reportar_comparacion(
        "(c) NULL-4 (n=3) vs NULL-2 (n=8) -- ¿preservar la topología COMPLETA supera preservar sólo el "
        "espectro de potencia (2-puntos)?", r_c)

    r_d = test_permutacion_2_grupos(masas_null4, masas_null3, nombre_a="NULL-4", nombre_b="NULL-3")
    _reportar_comparacion(
        "(d) NULL-4 (n=3) vs NULL-3 (n=8) -- ¿el ORDEN de formación agrega algo sobre preservar "
        "grado+longitud (NULL-3), dado que NULL-4 preserva la topología aún más fielmente (100% de "
        "las aristas, no ~87.5%)?", r_d)

    r_e = test_permutacion_2_grupos(masas_real, masas_null3, nombre_a="REAL", nombre_b="NULL-3")
    _reportar_comparacion("(e) referencia: REAL (n=6) vs NULL-3 (n=8) -- repetido de "
                           "null3_bateria_comparar.py para contexto en el mismo informe", r_e)

    print("\n" + "=" * 92)
    print("FIN. Números arriba; la lectura/veredicto es del informe adjunto y, en última")
    print("instancia, del director del proyecto -- este script no declara cierres.")
    print("=" * 92)


if __name__ == "__main__":
    main()
