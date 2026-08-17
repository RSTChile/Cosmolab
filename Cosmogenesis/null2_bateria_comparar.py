"""
null2_bateria_comparar.py — REAL (n=6, `bateria_n2000/ic_real` + `bateria_real_extra_n2000/`) vs las
8 NULL-2-Zel'dovich (N=2000) recién corridas, y opcionalmente NULL-2 vs NULL-1 (8 vs 8, ambos ya en
disco) -- Fase II CS073, escalón NULL-2.

Qué hace: lee, tal cual están en disco, sólo lectura:
    /Users/alexis/phantom_cs073/bateria_n2000/ic_real/cosmog01.sink                  (REAL original)
    /Users/alexis/phantom_cs073/bateria_real_extra_n2000/ic_real_s{301..305}/        (REAL, 5 más)
    /Users/alexis/phantom_cs073/bateria_null1_n2000/ic_null1_s{1..8}/                (NULL-1, 8)
    /Users/alexis/phantom_cs073/bateria_null2_n2000/ic_null2_s{401..408}/            (NULL-2, 8)
y calcula, por corrida, el MISMO observable que toda la jerarquía (masa total acumulada en sumideros
al final; columna 12 del .sink == columna 5 "mass", un sumidero sólo crece por acreción). Si una
corrida no formó ningún sumidero, su masa es 0 y su nº de sumideros es 0 -- no es un error.

`masa_y_n_sumideros` y `test_permutacion_nivel_corrida` son reimplementaciones línea a línea de
`null1_bateria_comparar.py` (congelado, no se toca) -- misma matemática, generalizada aquí a n_REAL=6
y a comparar además NULL-2 vs NULL-1 (dos grupos NULL, no NULL vs REAL) con el mismo test exacto de
permutación aplicado a dos grupos de 8 (C(16,8)=12870 asignaciones posibles bajo H0).

No corre ninguna simulación. No toca ninguna carpeta de batería anterior (sólo lectura). No declara
cierre de CS073 ni de la jerarquía de 6 -- reporta números, la lectura es de Alexis.
"""
from itertools import combinations
from pathlib import Path

import numpy as np

BASE_REAL_ORIG = Path("/Users/alexis/phantom_cs073/bateria_n2000/ic_real")
BASE_REAL_EXTRA = Path("/Users/alexis/phantom_cs073/bateria_real_extra_n2000")
REAL_EXTRA_SEEDS = [301, 302, 303, 304, 305]
BASE_NULL1 = Path("/Users/alexis/phantom_cs073/bateria_null1_n2000")
NULL1_CARPETAS = [BASE_NULL1 / f"ic_null1_s{i}" for i in range(1, 9)]
BASE_NULL2 = Path("/Users/alexis/phantom_cs073/bateria_null2_n2000")
NULL2_SEEDS = list(range(401, 409))
NULL2_CARPETAS = [BASE_NULL2 / f"ic_null2_s{s}" for s in NULL2_SEEDS]

COL_T = 0
COL_MACC = 11    # columna 12: masa acretada acumulada (== "mass", col 5, verificado en cs078)
COL_SINKID = 18  # columna 19: ID del sumidero


def masa_y_n_sumideros(carpeta: Path) -> dict:
    """Masa total en sumideros vivos al final + nº de sumideros de una corrida. 0/0 si la corrida
    no formó ningún sumidero (no existe cosmog01.sink)."""
    ruta = carpeta / "cosmog01.sink"
    if not ruta.exists():
        return dict(carpeta=carpeta.name, masa_total=0.0, n_sumideros=0, masas_individuales=[])

    data = np.loadtxt(ruta, skiprows=2)
    if data.ndim == 1:
        data = data[None, :]

    t_final = data[:, COL_T].max()
    en_t_final = data[np.isclose(data[:, COL_T], t_final)]
    sink_ids = np.unique(en_t_final[:, COL_SINKID].astype(int))
    masas = [float(en_t_final[en_t_final[:, COL_SINKID].astype(int) == sid, COL_MACC][0])
             for sid in sink_ids]

    return dict(carpeta=carpeta.name, masa_total=float(sum(masas)), n_sumideros=len(masas),
                masas_individuales=masas)


def test_permutacion_2_grupos(grupo_a: np.ndarray, grupo_b: np.ndarray, nombre_a="A", nombre_b="B") -> dict:
    """Test de permutación EXACTO de 2 muestras independientes (sin apareamiento), estadístico =
    media(grupo_a) - media(grupo_b), H1 pre-registrada de una cola (a > b), sobre las C(n_a+n_b, n_a)
    asignaciones posibles bajo H0 (etiquetas intercambiables). Si C(n_a+n_b,n_a) es grande se enumera
    completo igual (aquí n_a=n_b=8 o 6/8, manejable: C(16,8)=12870, C(14,6)=3003, C(14,8)=3003)."""
    todos = np.concatenate([grupo_a, grupo_b])
    n_total = len(todos)
    n_a = len(grupo_a)
    obs_stat = grupo_a.mean() - grupo_b.mean()

    stats = []
    for idx_a in combinations(range(n_total), n_a):
        idx_a = set(idx_a)
        sub_a = np.array([todos[i] for i in range(n_total) if i in idx_a])
        sub_b = np.array([todos[i] for i in range(n_total) if i not in idx_a])
        stats.append(sub_a.mean() - sub_b.mean())
    stats = np.array(stats)

    p_una_cola = float(np.mean(stats >= obs_stat - 1e-9))
    p_dos_colas = float(np.mean(np.abs(stats) >= abs(obs_stat) - 1e-9))
    n_asignaciones = len(stats)
    rank = int(np.sum(stats >= obs_stat - 1e-9))

    return dict(nombre_a=nombre_a, nombre_b=nombre_b, n_a=n_a, n_b=len(grupo_b),
                estadistico_observado=float(obs_stat), n_asignaciones=n_asignaciones,
                p_una_cola=p_una_cola, p_dos_colas=p_dos_colas, rank=rank)


def test_permutacion_nivel_corrida(masa_real_vec: np.ndarray, masas_null: np.ndarray) -> dict:
    """Generalización de `null1_bateria_comparar.test_permutacion_nivel_corrida` a n_REAL>1: n_REAL +
    n_NULL unidades, C(n_REAL+n_NULL, n_REAL) asignaciones posibles bajo H0, estadístico =
    media(grupo etiquetado REAL) - media(grupo etiquetado NULL). Idéntico en método a
    `real_extra_comparar.py` (que ya generalizó esto para REAL vs NULL/NULL-1)."""
    return test_permutacion_2_grupos(masa_real_vec, masas_null, nombre_a="REAL", nombre_b="NULL")


def main():
    print("=" * 92)
    print("NULL-2-Zel'dovich batería completa (N=2000, 8 semillas 401-408) vs REAL (n=6) y vs NULL-1 (n=8)")
    print("Observable: masa total acumulada en sumideros al final (mismo que toda la jerarquía CS073)")
    print("=" * 92)

    real_orig = masa_y_n_sumideros(BASE_REAL_ORIG)
    reales_extra = [masa_y_n_sumideros(BASE_REAL_EXTRA / f"ic_real_s{s}") for s in REAL_EXTRA_SEEDS]
    reales = [real_orig] + reales_extra
    nulls1 = [masa_y_n_sumideros(c) for c in NULL1_CARPETAS]
    nulls2 = [masa_y_n_sumideros(c) for c in NULL2_CARPETAS]

    print(f"\nREAL (n={len(reales)}):")
    for d in reales:
        print(f"  {d['carpeta']:14s} masa_total={d['masa_total']:.1f}  n_sumideros={d['n_sumideros']}")
    masas_real = np.array([d["masa_total"] for d in reales])
    print(f"  media={masas_real.mean():.4f}  DE={masas_real.std(ddof=1):.4f}  "
          f"min={masas_real.min():.1f}  max={masas_real.max():.1f}")

    print(f"\nNULL-1 (n={len(nulls1)}, ángulo isótropo aleatorio, mismo radio que REAL):")
    for d in nulls1:
        print(f"  {d['carpeta']:14s} masa_total={d['masa_total']:.1f}  n_sumideros={d['n_sumideros']}")
    masas_null1 = np.array([d["masa_total"] for d in nulls1])
    print(f"  media={masas_null1.mean():.4f}  DE={masas_null1.std(ddof=1):.4f}  "
          f"min={masas_null1.min():.1f}  max={masas_null1.max():.1f}")

    print(f"\nNULL-2-Zel'dovich (n={len(nulls2)}, fases aleatorizadas en P(k) + desplazamiento Lagrangiano):")
    for d in nulls2:
        print(f"  {d['carpeta']:14s} masa_total={d['masa_total']:.1f}  n_sumideros={d['n_sumideros']}")
    masas_null2 = np.array([d["masa_total"] for d in nulls2])
    if masas_null2.std(ddof=1) > 0:
        de2_str = f"{masas_null2.std(ddof=1):.4f}"
    else:
        de2_str = "0.0000 (todas iguales)"
    print(f"  media={masas_null2.mean():.4f}  DE={de2_str}  "
          f"min={masas_null2.min():.1f}  max={masas_null2.max():.1f}")

    # ---- (a) REAL (n=6) vs NULL-2 (n=8) ----
    print("\n" + "-" * 92)
    print("(a) REAL (n=6) vs NULL-2-Zel'dovich (n=8) -- test de permutación EXACTO de 2 grupos")
    r_a = test_permutacion_nivel_corrida(masas_real, masas_null2)
    print(f"    estadístico observado (media_REAL - media_NULL2) = {r_a['estadistico_observado']:.4f}")
    print(f"    C({r_a['n_a']}+{r_a['n_b']},{r_a['n_a']}) = {r_a['n_asignaciones']} asignaciones posibles bajo H0")
    print(f"    rank de la asignación observada = {r_a['rank']} de {r_a['n_asignaciones']}")
    print(f"    p (una cola, H1 pre-registrada REAL>NULL2) = {r_a['p_una_cola']:.6f}  "
          f"(piso teórico = 1/{r_a['n_asignaciones']} = {1/r_a['n_asignaciones']:.6f})")
    print(f"    p (dos colas) = {r_a['p_dos_colas']:.6f}")
    if masas_null2.std(ddof=1) > 0:
        z = (masas_real.mean() - masas_null2.mean()) / masas_null2.std(ddof=1)
        print(f"    z-score ((media_REAL - media_NULL2) / DE_NULL2) = {z:.3f}")
    else:
        print("    z-score: INDEFINIDO (DE_NULL2 = 0 -- las 8 NULL-2 dieron exactamente la misma masa).")

    # ---- (b) NULL-1 (n=8) vs NULL-2 (n=8) -- comparación entre escalones de la jerarquía ----
    print("\n" + "-" * 92)
    print("(b) NULL-1 (n=8) vs NULL-2-Zel'dovich (n=8) -- ¿el 2-puntos parcialmente preservado de")
    print("    Zel'dovich forma MÁS estructura que NULL-1 (que no preserva nada de eso)?")
    r_b = test_permutacion_2_grupos(masas_null2, masas_null1, nombre_a="NULL-2", nombre_b="NULL-1")
    print(f"    estadístico observado (media_NULL2 - media_NULL1) = {r_b['estadistico_observado']:.4f}")
    print(f"    C({r_b['n_a']}+{r_b['n_b']},{r_b['n_a']}) = {r_b['n_asignaciones']} asignaciones posibles bajo H0")
    print(f"    rank de la asignación observada = {r_b['rank']} de {r_b['n_asignaciones']}")
    print(f"    p (una cola, H1 exploratoria NULL2>NULL1) = {r_b['p_una_cola']:.6f}")
    print(f"    p (dos colas) = {r_b['p_dos_colas']:.6f}")
    if masas_null2.std(ddof=1) == 0 and masas_null1.std(ddof=1) == 0:
        print("    Nota: ambos grupos con DE=0 -- si ambas masas medias son iguales, el test es no")
        print("    informativo por construcción (no hay varianza combinada para permutar contra).")

    print("\n" + "=" * 92)
    print("FIN. Números arriba; la lectura/veredicto es del informe adjunto y, en última")
    print("instancia, del director del proyecto -- este script no declara cierres.")
    print("=" * 92)


if __name__ == "__main__":
    main()
