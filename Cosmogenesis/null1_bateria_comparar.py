"""
null1_bateria_comparar.py — REAL (bateria_n2000, YA EXISTENTE) vs las 8 NULL-1 (N=2000) recién
corridas, batería completa (Fase II CS073, escalón NULL-1).

Qué hace: lee, tal cual están en disco, sólo lectura:
    /Users/alexis/phantom_cs073/bateria_n2000/ic_real/cosmog01.sink        (REAL, 1 corrida)
    /Users/alexis/phantom_cs073/bateria_null1_n2000/ic_null1_s{1..8}/      (NULL-1, 8 corridas)
y calcula, por corrida, el MISMO observable que usó CS073 original (masa total acumulada en
sumideros al final -- ver RESULTADO_bateria_ignicion_sumideros_N2000_CS.md): para cada sumidero vivo
al final de la corrida, su masa acretada acumulada (columna 12 del .sink, == columna 5 "mass" --
verificado, un sumidero sólo crece por acreción); la corrida aporta la SUMA sobre todos sus sumideros
en el ÚLTIMO paso de tiempo grabado. Si una corrida no formó ningún sumidero (no existe cosmog01.sink),
su masa es 0 y su nº de sumideros es 0 -- no es un error, es el resultado (ver
NULL1_bateria_completa_CS.md).

Estadístico de separación: el más riguroso ya validado en este proyecto es el test de permutación
EXACTO a nivel de CORRIDA de cs078_kappaV_permutacion.py (sección 3) -- 9 unidades (1 REAL + 8 NULL),
C(9,1)=9 asignaciones posibles bajo H0 (cualquiera de las 9 pudo ser "REAL"), estadístico =
masa(corrida etiquetada REAL) - media(masa de las 8 restantes), p (una cola, H1 pre-registrada
REAL>NULL) = fracción de las 9 asignaciones con estadístico >= al observado. Se reutiliza esa MISMA
función (se reimplementa aquí en 15 líneas porque cs078 la define inline sobre su propio diccionario
de datos -- no se le pueden pasar valores externos sin tocar ese script congelado -- pero es la
matemática idéntica: se puede diffear contra `test_permutacion_nivel_corrida` de cs078 línea a línea).
También se reporta el z-score (REAL - media_NULL)/DE_NULL, el estadístico original de CS073, con el
caveat explícito si DE_NULL=0 (no está definido, no se disimula con un número).

No corre ninguna simulación. No toca bateria_n2000/ (sólo lectura). No declara cierre de CS073 ni de
la jerarquía de 6 -- reporta números, la lectura es de Alexis.
"""
from pathlib import Path

import numpy as np

BASE_REAL = Path("/Users/alexis/phantom_cs073/bateria_n2000/ic_real")
BASE_NULL1 = Path("/Users/alexis/phantom_cs073/bateria_null1_n2000")
NULL1_CARPETAS = [BASE_NULL1 / f"ic_null1_s{i}" for i in range(1, 9)]

COL_T = 0
COL_MACC = 11    # columna 12: masa acretada acumulada (== "mass", col 5, verificado en cs078)
COL_SINKID = 18  # columna 19: ID del sumidero


def masa_y_n_sumideros(carpeta: Path) -> dict:
    """Masa total en sumideros vivos al final + nº de sumideros de una corrida. 0/0 si la corrida
    no formó ningún sumidero (no existe cosmog01.sink -- caso real de las 8 NULL-1, no un fallo)."""
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


def test_permutacion_nivel_corrida(masa_real: float, masas_null: np.ndarray) -> dict:
    """Idéntico en método a `test_permutacion_nivel_corrida` de cs078_kappaV_permutacion.py
    (sección 3), aplicado aquí a masa_total_en_sumideros en vez de kappa_V. 9 unidades (1 REAL +
    8 NULL-1), C(9,1)=9 asignaciones posibles bajo H0."""
    valores = np.concatenate(([masa_real], masas_null))
    n = len(valores)
    obs_stat = valores[0] - np.mean(valores[1:])

    stats = []
    for i in range(n):
        resto = np.concatenate((valores[:i], valores[i + 1:]))
        stats.append(valores[i] - resto.mean())
    stats = np.array(stats)

    p_una_cola = float(np.mean(stats >= obs_stat - 1e-9))
    p_dos_colas = float(np.mean(np.abs(stats) >= abs(obs_stat) - 1e-9))
    rank = int(np.sum(stats >= obs_stat - 1e-9))

    return dict(estadistico_observado=float(obs_stat), distribucion_nula_9=stats.tolist(),
                p_una_cola=p_una_cola, p_dos_colas=p_dos_colas, rank_de_9=rank)


def main():
    print("=" * 88)
    print("NULL-1 batería completa (N=2000, 8 semillas) vs REAL ya existente (bateria_n2000/ic_real)")
    print("Observable: masa total acumulada en sumideros al final (mismo que CS073 original)")
    print("=" * 88)

    real = masa_y_n_sumideros(BASE_REAL)
    nulls = [masa_y_n_sumideros(c) for c in NULL1_CARPETAS]

    print(f"\nREAL       masa_total={real['masa_total']:.1f}  n_sumideros={real['n_sumideros']}  "
          f"masas_individuales={[round(m, 1) for m in real['masas_individuales']]}")
    for d in nulls:
        print(f"{d['carpeta']:12s} masa_total={d['masa_total']:.1f}  n_sumideros={d['n_sumideros']}")

    masas_null = np.array([d["masa_total"] for d in nulls])
    media_null = masas_null.mean()
    de_null = masas_null.std(ddof=1)

    print(f"\nNULL-1 (n=8): media={media_null:.4f}  DE={de_null:.4f}  "
          f"min={masas_null.min():.1f}  max={masas_null.max():.1f}")

    if de_null > 0:
        z = (real["masa_total"] - media_null) / de_null
        print(f"z-score (REAL - media_NULL) / DE_NULL = {z:.3f}")
    else:
        print("z-score: INDEFINIDO (DE_NULL=0 -- las 8 corridas NULL-1 dieron EXACTAMENTE la misma "
              "masa, 0.0 -- ver nota abajo). No se rellena con un número inventado.")
        if media_null == 0.0:
            print("  Nota: las 8 corridas NULL-1 formaron 0 sumideros y 0.0 masa cada una -- "
                  "resultado categórico, no al límite. REAL formó "
                  f"{real['n_sumideros']} sumideros con masa {real['masa_total']:.1f}.")

    r = test_permutacion_nivel_corrida(real["masa_total"], masas_null)
    print(f"\nTest de permutación EXACTO a nivel de corrida (9 unidades: 1 REAL + 8 NULL-1, "
          f"C(9,1)=9 asignaciones posibles, idéntico en método a cs078_kappaV_permutacion.py sección 3):")
    print(f"  estadístico observado (REAL - media(resto de 8)) = {r['estadistico_observado']:.4f}")
    print(f"  distribución nula exacta (9 valores): {[round(v, 4) for v in r['distribucion_nula_9']]}")
    print(f"  rank de REAL entre las 9 asignaciones = {r['rank_de_9']} de 9 (1 = más extremo)")
    print(f"  p (una cola, H1 pre-registrada REAL>NULL) = {r['p_una_cola']:.4f}  "
          f"(mínimo posible con n=9: 1/9={1/9:.4f})")
    print(f"  p (dos colas) = {r['p_dos_colas']:.4f}")

    print("\n" + "=" * 88)
    print("FIN. Números arriba; la lectura/veredicto es del informe adjunto y, en última")
    print("instancia, del director del proyecto -- este script no declara cierres.")
    print("=" * 88)


if __name__ == "__main__":
    main()
