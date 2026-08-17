"""
real_extra_comparar.py -- CS073, el test de permutación EXACTO recalculado con n_REAL>1.

Antecedente exacto del cuello de botella (pedido por Alexis): con 1 sola corrida REAL y 8 NULL, el test
de permutación exacto a nivel de corrida de cs078_kappaV_permutacion.py / null1_bateria_comparar.py sólo
puede asignar la etiqueta "REAL" a 1 de 9 unidades -> C(9,1)=9 asignaciones posibles bajo H0 -> el propio
diseño del test IMPIDE p<1/9=0.1111, sin importar la magnitud del efecto observado. Esa cota no es un
problema de potencia estadística que "más NULL" pueda arreglar -- depende sólo de cuántas etiquetas REAL
hay para repartir.

Qué hace este script: generaliza el mismo test (permutación exacta, estadístico = media(grupo REAL) -
media(grupo NULL), H1 pre-registrada REAL>NULL, una cola) al caso n_REAL>1 -- ahora hay C(n_total,
n_real) formas de elegir qué corridas llevan la etiqueta REAL bajo H0, no sólo n_total. Con n_real=6
(1 original de bateria_n2000/ic_real + 5 nuevas de bateria_real_extra_n2000) y n_null=8, n_total=14,
C(14,6)=3003 -- el piso baja de 1/9=0.1111 a 1/3003=0.000333, un factor >300 sin haber corrido NI UN
NULL adicional (la salvaguarda que pidió Alexis: la cura era más REAL, no más NULL).

Se corre DOS VECES el mismo test: (a) REAL(n) vs los 8 NULL originales de bateria_n2000 (barajado de
ARISTAS + barajado de densidad legado); (b) REAL(n) vs los 8 NULL-1 de bateria_null1_n2000 (mismo radio
que REAL, ángulo isótropo aleatorio -- el escalón más aislado de la jerarquía de 6 controles).

También reporta la variabilidad ENTRE semillas REAL (media, DE, min, max de masa_total en sumideros) --
algo que con n_REAL=1 era estructuralmente invisible: no había forma de saber si la corrida original era
representativa o un outlier hasta tener más de una.

Reutiliza (no reimplementa) `masa_y_n_sumideros` de null1_bateria_comparar.py (congelado) para leer los
.sink -- mismo criterio de "columna 12 = masa acretada acumulada, corrida contribuye la SUMA sobre todos
sus sumideros vivos en el último paso grabado, 0/0 si no formó ninguno".

No corre ninguna simulación. No toca bateria_n2000/ ni bateria_null1_n2000/ (sólo lectura). No declara
cierre de CS073 -- reporta números, la lectura es de Alexis.
"""
import itertools
from pathlib import Path

import numpy as np

from null1_bateria_comparar import masa_y_n_sumideros

BASE_REAL_ORIGINAL = Path("/Users/alexis/phantom_cs073/bateria_n2000/ic_real")
BASE_REAL_EXTRA = Path("/Users/alexis/phantom_cs073/bateria_real_extra_n2000")
SEMILLAS_REAL_EXTRA = [301, 302, 303, 304, 305]

BASE_NULL_ORIGINAL = Path("/Users/alexis/phantom_cs073/bateria_n2000")
NULL_ORIGINAL_CARPETAS = [BASE_NULL_ORIGINAL / f"ic_null{i}" for i in range(1, 9)]

BASE_NULL1 = Path("/Users/alexis/phantom_cs073/bateria_null1_n2000")
NULL1_CARPETAS = [BASE_NULL1 / f"ic_null1_s{i}" for i in range(1, 9)]


def test_permutacion_dos_grupos(masas_real: np.ndarray, masas_null: np.ndarray) -> dict:
    """Test de permutación EXACTO, generalización de `test_permutacion_nivel_corrida`
    (cs078_kappaV_permutacion.py / null1_bateria_comparar.py) a n_real>1: bajo H0 cualquier subconjunto
    de tamaño n_real de las n_total=n_real+n_null corridas pudo llevar la etiqueta REAL. Se enumeran
    TODAS las C(n_total, n_real) asignaciones (exacto, no muestreado -- factible mientras
    C(n_total,n_real) sea manejable, como aquí). Estadístico = media(grupo REAL) - media(grupo NULL);
    H1 pre-registrada de CS073, una cola (REAL>NULL)."""
    valores = np.concatenate([masas_real, masas_null])
    n_total = len(valores)
    n_real = len(masas_real)

    obs_stat = float(masas_real.mean() - masas_null.mean())

    stats = []
    for idx_real in itertools.combinations(range(n_total), n_real):
        idx_real = set(idx_real)
        idx_null = [i for i in range(n_total) if i not in idx_real]
        g_real = valores[list(idx_real)]
        g_null = valores[idx_null]
        stats.append(float(g_real.mean() - g_null.mean()))
    stats = np.array(stats)
    n_asignaciones = len(stats)

    p_una_cola = float(np.mean(stats >= obs_stat - 1e-9))
    p_dos_colas = float(np.mean(np.abs(stats) >= abs(obs_stat) - 1e-9))
    rank = int(np.sum(stats >= obs_stat - 1e-9))

    return dict(estadistico_observado=obs_stat, n_asignaciones=n_asignaciones,
                p_una_cola=p_una_cola, p_dos_colas=p_dos_colas, rank=rank,
                p_minimo_posible=1.0 / n_asignaciones)


def recolectar_real():
    """Lee (sólo lectura) la corrida REAL original (bateria_n2000/ic_real, seed_layout=12345, la de
    CS073) + las que efectivamente terminaron de correr en bateria_real_extra_n2000/ic_real_s{301..305}
    -- si real_extra_correr.py se detuvo por la salvaguarda de tiempo antes de completar las 5, sólo se
    incluyen las que sí tienen cosmog01.sink o cuya corrida terminó (se detecta por la presencia de
    run.log con al menos una línea, y se usa 0/0 si no formó sumidero -- igual criterio que NULL-1)."""
    filas = []
    d0 = masa_y_n_sumideros(BASE_REAL_ORIGINAL)
    d0["seed_layout"] = 12345
    d0["etiqueta"] = "REAL_original"
    filas.append(d0)

    for s in SEMILLAS_REAL_EXTRA:
        carpeta = BASE_REAL_EXTRA / f"ic_real_s{s}"
        run_log = carpeta / "run.log"
        if not run_log.exists():
            print(f"[aviso] {carpeta.name}: no tiene run.log -- no se corrió, se omite del análisis.")
            continue
        d = masa_y_n_sumideros(carpeta)
        d["seed_layout"] = s
        d["etiqueta"] = f"REAL_s{s}"
        filas.append(d)
    return filas


def main():
    print("=" * 92)
    print("CS073 -- test de permutación EXACTO con n_REAL>1 (5 semillas REAL adicionales, seed_layout)")
    print("=" * 92)

    real_filas = recolectar_real()
    print(f"\n--- {len(real_filas)} corridas REAL disponibles para el análisis ---")
    for d in real_filas:
        print(f"  {d['etiqueta']:16s} seed_layout={d['seed_layout']:6d}  "
              f"masa_total={d['masa_total']:.1f}  n_sumideros={d['n_sumideros']}")

    masas_real = np.array([d["masa_total"] for d in real_filas])
    print(f"\nVariabilidad ENTRE semillas REAL (n={len(masas_real)}): "
          f"media={masas_real.mean():.2f}  DE={masas_real.std(ddof=1) if len(masas_real) > 1 else 0.0:.2f}  "
          f"min={masas_real.min():.1f}  max={masas_real.max():.1f}")
    print("(con n_REAL=1 esta fila no existía -- no había forma de saber si la corrida original era "
          "representativa o un outlier.)")

    for nombre_null, carpetas_null in (("NULL original (bateria_n2000, 8 corridas)", NULL_ORIGINAL_CARPETAS),
                                        ("NULL-1 (bateria_null1_n2000, 8 corridas)", NULL1_CARPETAS)):
        print("\n" + "-" * 92)
        print(f"REAL (n={len(masas_real)}) vs {nombre_null}")
        print("-" * 92)
        nulls = [masa_y_n_sumideros(c) for c in carpetas_null]
        for d in nulls:
            print(f"  {d['carpeta']:14s} masa_total={d['masa_total']:.1f}  n_sumideros={d['n_sumideros']}")
        masas_null = np.array([d["masa_total"] for d in nulls])
        print(f"  NULL (n={len(masas_null)}): media={masas_null.mean():.2f}  "
              f"DE={masas_null.std(ddof=1) if len(masas_null) > 1 else 0.0:.2f}  "
              f"min={masas_null.min():.1f}  max={masas_null.max():.1f}")

        r = test_permutacion_dos_grupos(masas_real, masas_null)
        print(f"\n  Test de permutación EXACTO (H1 pre-registrada REAL>NULL, una cola):")
        print(f"    estadístico observado (media_REAL - media_NULL) = {r['estadistico_observado']:.4f}")
        print(f"    nº de asignaciones posibles bajo H0 = C({len(masas_real)+len(masas_null)},"
              f"{len(masas_real)}) = {r['n_asignaciones']}")
        print(f"    rank de la asignación observada = {r['rank']} de {r['n_asignaciones']} "
              f"(1 = más extremo)")
        print(f"    p (una cola) = {r['p_una_cola']:.6f}   p (dos colas) = {r['p_dos_colas']:.6f}")
        print(f"    piso de p alcanzable con este n = {r['p_minimo_posible']:.6f}  "
              f"(vs 1/9={1/9:.6f} con n_REAL=1)")

    print("\n" + "=" * 92)
    print("FIN. Números arriba; la lectura/veredicto de CS073 es del director del proyecto -- este "
          "script no declara cierres.")
    print("=" * 92)


if __name__ == "__main__":
    main()
