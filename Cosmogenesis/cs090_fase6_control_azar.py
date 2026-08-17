"""
cs090_fase6_control_azar.py -- FASE VI, Analisis 1: control de parejas al azar (VI-A / A1 del equipo).

Pregunta que responde este script: el resultado de Fase V-B (Clase III > Clase I en fraccion de masa
acretada y en kappa_V, en 40 pares reales emparejados por K=kcap) -- sobrevive si a los MISMOS 80 grafos
se les ignora la clase real y se los re-empareja al azar? O el resultado depende especificamente de que
el emparejamiento real respete la distincion Clase I / Clase III?

Metodologia:
  1. Cargar las 80 filas de cs090_fase5b_TOTAL_40pares.csv (40 reglas Clase I + 40 Clase III, verificado
     exactamente 40+40 antes de seguir).
  2. Reproducir el resultado REAL (referencia): 40 diferencias pareadas (fila III - fila I) usando el
     emparejamiento real por columna 'par', sobre 'fraccion_masa_en_sumideros' y 'kappa_v_agregado'.
     Test de signos (binomial exacto) + Wilcoxon signed-rank sobre esas 40 diferencias.
  3. Control al azar: se re-empareja el POOL COMPLETO de 80 filas (ignorando la columna clase/rol) N_PERM
     veces (semilla fija np.random.default_rng(2026) para reproducibilidad). En cada permutacion:
       - se baraja el orden de las 80 filas,
       - se forman 40 pares consecutivos (fila 0-1, 2-3, ..., 78-79),
       - la diferencia de cada par es (segunda fila del par) - (primera fila del par) -- el orden ya es
         arbitrario por el barajado completo, así que no hay sesgo hacia "la que era III menos la que
         era I": la clase real de cada miembro del par al azar es irrelevante y en general mixta (un par
         al azar puede ser I-I, I-III, III-I o III-III).
       - se corre el mismo test de signos + Wilcoxon sobre esas 40 diferencias al azar.
  4. Se compara el resultado REAL contra la distribucion de resultados AL AZAR: se reporta que fraccion
     de las permutaciones da un conteo de "victorias" (diferencias positivas) tan o mas extremo que el
     observado en el emparejamiento real, y que fraccion da un p-valor de signos/Wilcoxon tan bajo o mas
     bajo que el observado.

Interpretacion (sin forzar, se deja para Alexis):
  - Si el resultado REAL cae en la cola extrema (ej. percentil <1%) de la distribucion al azar: evidencia
    de que el efecto SI depende de la clase real, no solo del pool de 80 grafos.
  - Si el resultado REAL cae dentro del rango tipico de los emparejamientos al azar: evidencia de que el
    efecto observado en Fase V-B podria no depender especificamente de la distincion I/III.

No modifica ningun script/CSV existente (solo lectura de cs090_fase5b_TOTAL_40pares.csv). No corre
Phantom. No declara cierre ni veredicto -- solo reporta numeros.
"""
import numpy as np
import pandas as pd
from scipy import stats

RUTA_TOTAL = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs090_fase5b_TOTAL_40pares.csv"
N_PERM = 10000
SEMILLA = 2026
METRICAS = ["fraccion_masa_en_sumideros", "kappa_v_agregado"]


def cargar_datos():
    df = pd.read_csv(RUTA_TOTAL)
    n_I = (df["clase"] == "I").sum()
    n_III = (df["clase"] == "III").sum()
    assert len(df) == 80, f"esperaba 80 filas, hay {len(df)}"
    assert n_I == 40 and n_III == 40, f"esperaba 40+40, hay I={n_I} III={n_III}"
    assert (df["clase"] == df["rol"]).all(), "columna clase y rol no coinciden en alguna fila"
    assert df["par"].nunique() == 40, f"esperaba 40 pares unicos, hay {df['par'].nunique()}"
    return df


def diffs_reales(df, metrica):
    """40 diferencias pareadas (III - I) usando el emparejamiento REAL (columna 'par')."""
    diffs = []
    for par, g in df.groupby("par"):
        assert len(g) == 2, f"par {par} no tiene exactamente 2 filas"
        fila_III = g[g["rol"] == "III"]
        fila_I = g[g["rol"] == "I"]
        assert len(fila_III) == 1 and len(fila_I) == 1, f"par {par} no tiene exactamente 1 I y 1 III"
        diffs.append(fila_III[metrica].values[0] - fila_I[metrica].values[0])
    return np.array(diffs)


def test_signos_wilcoxon(diffs):
    n = len(diffs)
    n_pos = int((diffs > 0).sum())
    n_neg = int((diffs < 0).sum())
    n_cero = int((diffs == 0).sum())
    # test de signos: binomial exacto de 2 colas sobre n_pos de (n_pos+n_neg) con p=0.5
    n_efectivo = n_pos + n_neg
    if n_efectivo == 0:
        p_signos = 1.0
    else:
        p_signos = stats.binomtest(max(n_pos, n_neg), n_efectivo, 0.5, alternative="two-sided").pvalue
    try:
        W, p_wilcoxon = stats.wilcoxon(diffs, alternative="two-sided")
    except ValueError:
        W, p_wilcoxon = np.nan, np.nan
    return dict(n=n, n_pos=n_pos, n_neg=n_neg, n_cero=n_cero, p_signos=p_signos, W=W, p_wilcoxon=p_wilcoxon)


def permutaciones_azar(df, metrica, rng, n_perm):
    """N_PERM re-emparejamientos al azar del pool de 80 filas, ignorando clase real."""
    valores = df[metrica].to_numpy()
    n_filas = len(valores)
    assert n_filas == 80
    resultados = []
    idx = np.arange(n_filas)
    for _ in range(n_perm):
        perm = rng.permutation(idx)
        barajado = valores[perm]
        a = barajado[0::2]  # 40 valores "primer miembro del par"
        b = barajado[1::2]  # 40 valores "segundo miembro del par"
        diffs = b - a
        r = test_signos_wilcoxon(diffs)
        resultados.append(r)
    return pd.DataFrame(resultados)


def resumen_percentil(real_wins_extremos, dist_wins, real_p_signos, dist_p_signos,
                       real_p_wilcoxon, dist_p_wilcoxon):
    frac_wins_tan_extremo = float((dist_wins >= real_wins_extremos).mean())
    frac_p_signos_tan_bajo = float((dist_p_signos <= real_p_signos).mean())
    frac_p_wilcoxon_tan_bajo = float((dist_p_wilcoxon <= real_p_wilcoxon).mean())
    return frac_wins_tan_extremo, frac_p_signos_tan_bajo, frac_p_wilcoxon_tan_bajo


def main():
    df = cargar_datos()
    print(f"[verificacion] 80 filas, 40 Clase I + 40 Clase III, columna clase==rol en las 80, "
          f"40 pares unicos -- OK")

    rng = rng_maestro = np.random.default_rng(SEMILLA)
    reporte_lineas = []

    for metrica in METRICAS:
        print(f"\n===== METRICA: {metrica} =====")
        diffs_real = diffs_reales(df, metrica)
        r_real = test_signos_wilcoxon(diffs_real)
        print(f"[REAL] n={r_real['n']} n_pos={r_real['n_pos']} n_neg={r_real['n_neg']} "
              f"p_signos={r_real['p_signos']:.6f} W={r_real['W']:.4f} p_wilcoxon={r_real['p_wilcoxon']:.6f}")

        # extremidad del resultado real en "distancia de 20" (mitad de 40) -- dos colas
        real_extremidad = abs(r_real["n_pos"] - 20)

        # sub-generador propio para esta metrica para que ambas metricas usen la MISMA secuencia base
        # de permutaciones del pool (misma semilla raiz, distinta metrica -> distinto valor barajado)
        rng_local = np.random.default_rng(SEMILLA)
        dist = permutaciones_azar(df, metrica, rng_local, N_PERM)
        dist["n_pos_extremidad"] = (dist["n_pos"] - 20).abs()

        frac_wins, frac_psig, frac_pwil = resumen_percentil(
            real_extremidad, dist["n_pos_extremidad"].to_numpy(),
            r_real["p_signos"], dist["p_signos"].to_numpy(),
            r_real["p_wilcoxon"], dist["p_wilcoxon"].to_numpy(),
        )

        media_wins_azar = dist["n_pos"].mean()
        std_wins_azar = dist["n_pos"].std()
        pct_wins_real = float((dist["n_pos"] <= r_real["n_pos"]).mean()) * 100

        print(f"[AZAR n_perm={N_PERM}] media n_pos={media_wins_azar:.2f} (std={std_wins_azar:.2f}), "
              f"real n_pos={r_real['n_pos']} cae en percentil {pct_wins_real:.2f} de la distribucion azar")
        print(f"  fraccion de permutaciones con |n_pos-20| >= |real-20|={real_extremidad}: {frac_wins:.5f}")
        print(f"  fraccion de permutaciones con p_signos <= real ({r_real['p_signos']:.6f}): {frac_psig:.5f}")
        print(f"  fraccion de permutaciones con p_wilcoxon <= real ({r_real['p_wilcoxon']:.6f}): {frac_pwil:.5f}")

        dist.to_csv(
            f"/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs090_fase6_azar_distribucion_{metrica}.csv",
            index=False,
        )

        reporte_lineas.append(dict(
            metrica=metrica,
            n_pos_real=r_real["n_pos"], n_neg_real=r_real["n_neg"],
            p_signos_real=r_real["p_signos"], W_real=r_real["W"], p_wilcoxon_real=r_real["p_wilcoxon"],
            media_n_pos_azar=media_wins_azar, std_n_pos_azar=std_wins_azar,
            percentil_real_en_azar=pct_wins_real,
            frac_perm_tan_extrema_en_wins=frac_wins,
            frac_perm_p_signos_tan_bajo=frac_psig,
            frac_perm_p_wilcoxon_tan_bajo=frac_pwil,
            n_perm=N_PERM, semilla=SEMILLA,
        ))

    resumen = pd.DataFrame(reporte_lineas)
    resumen.to_csv(
        "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs090_fase6_control_azar_resumen.csv",
        index=False,
    )
    print("\n[guardado] cs090_fase6_control_azar_resumen.csv")
    print("[guardado] cs090_fase6_azar_distribucion_<metrica>.csv (una por metrica, N_PERM filas c/u)")


if __name__ == "__main__":
    main()
