"""
null3_bateria_comparar.py — REAL (n=6, `bateria_n2000/ic_real` + `bateria_real_extra_n2000/`) vs las
8 NULL-3 (N=2000, semillas 501-508) recién corridas, y NULL-3 vs NULL-1/NULL-2 (8 vs 8, ambos ya en
disco) -- Fase II CS073, escalón NULL-3.

`masa_y_n_sumideros` y `test_permutacion_2_grupos` son reimplementaciones línea a línea de
`null2_bateria_comparar.py` (congelado, no se toca) -- misma matemática exacta, mismas columnas del
.sink, mismo test de permutación exacto (H1 pre-registrada de una cola REAL>NULL, sobre TODAS las
C(n_REAL+n_NULL, n_REAL) asignaciones posibles bajo H0).

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
BASE_NULL3 = Path("/Users/alexis/phantom_cs073/bateria_null3_n2000")
NULL3_SEEDS = list(range(501, 509))
NULL3_CARPETAS = [BASE_NULL3 / f"ic_null3_s{s}" for s in NULL3_SEEDS]

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
    """Test de permutación EXACTO de 2 muestras independientes, estadístico = media(a) - media(b),
    H1 de una cola (a > b), enumerando TODAS las C(n_a+n_b, n_a) asignaciones bajo H0."""
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


def _reportar_grupo(nombre, dicts):
    print(f"\n{nombre} (n={len(dicts)}):")
    for d in dicts:
        print(f"  {d['carpeta']:14s} masa_total={d['masa_total']:.1f}  n_sumideros={d['n_sumideros']}")
    masas = np.array([d["masa_total"] for d in dicts])
    de_str = f"{masas.std(ddof=1):.4f}" if masas.std(ddof=1) > 0 else "0.0000 (todas iguales)"
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
    print("NULL-3 batería completa (N=2000, 8 semillas 501-508) vs REAL (n=6), NULL-1 (n=8), NULL-2 (n=8)")
    print("Observable: masa total acumulada en sumideros al final (mismo que toda la jerarquía CS073)")
    print("=" * 92)

    real_orig = masa_y_n_sumideros(BASE_REAL_ORIG)
    reales_extra = [masa_y_n_sumideros(BASE_REAL_EXTRA / f"ic_real_s{s}") for s in REAL_EXTRA_SEEDS]
    reales = [real_orig] + reales_extra
    nulls1 = [masa_y_n_sumideros(c) for c in NULL1_CARPETAS]
    nulls2 = [masa_y_n_sumideros(c) for c in NULL2_CARPETAS]
    nulls3 = [masa_y_n_sumideros(c) for c in NULL3_CARPETAS]

    masas_real = _reportar_grupo("REAL", reales)
    masas_null1 = _reportar_grupo("NULL-1 (ángulo isótropo aleatorio, mismo radio que REAL)", nulls1)
    masas_null2 = _reportar_grupo("NULL-2 (Zel'dovich, P(k) parcialmente preservado)", nulls2)
    masas_null3 = _reportar_grupo(
        "NULL-3 (double-edge-swap + filtro geométrico de longitud, grado+longitud preservados)", nulls3)

    r_a = test_permutacion_2_grupos(masas_real, masas_null3, nombre_a="REAL", nombre_b="NULL-3")
    _reportar_comparacion("(a) REAL (n=6) vs NULL-3 (n=8) -- test de permutación EXACTO", r_a)
    if masas_null3.std(ddof=1) > 0:
        z = (masas_real.mean() - masas_null3.mean()) / masas_null3.std(ddof=1)
        print(f"    z-score ((media_REAL - media_NULL3) / DE_NULL3) = {z:.3f}")
    else:
        print("    z-score: INDEFINIDO (DE_NULL3 = 0).")

    r_b = test_permutacion_2_grupos(masas_null3, masas_null1, nombre_a="NULL-3", nombre_b="NULL-1")
    _reportar_comparacion(
        "(b) NULL-3 (n=8) vs NULL-1 (n=8) -- ¿preservar grado+longitud forma MÁS estructura que "
        "NULL-1 (ángulo puro)?", r_b)

    r_c = test_permutacion_2_grupos(masas_null3, masas_null2, nombre_a="NULL-3", nombre_b="NULL-2")
    _reportar_comparacion(
        "(c) NULL-3 (n=8) vs NULL-2 (n=8) -- ¿la topología de orden superior (motivos/ciclos) agrega "
        "algo sobre el 2-puntos de campo (Zel'dovich)?", r_c)

    print("\n" + "=" * 92)
    print("FIN. Números arriba; la lectura/veredicto es del informe adjunto y, en última")
    print("instancia, del director del proyecto -- este script no declara cierres.")
    print("=" * 92)


if __name__ == "__main__":
    main()
