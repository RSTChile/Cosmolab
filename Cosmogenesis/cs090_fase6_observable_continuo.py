"""
cs090_fase6_observable_continuo.py -- FASE VI, Analisis 2: observable continuo sin clases (F6-01).

Pregunta que responde este script: en vez de "es Clase I o Clase III" (un corte binario en pendiente=0.7),
la PENDIENTE CONTINUA (el numero real log(diam)-vs-log(N) del que depende la clasificacion) predice de
forma monotona la fraccion de masa acretada en Phantom?

PASO 1 -- JOIN VERIFICADO (la parte critica, documentada con cuidado porque esta linea de trabajo tuvo
al menos un bug real de colision de nombres de regla entre lotes -- ver
FASE5B_investigacion_8sumideros_y_escala_CS.md, seccion 2.1):

  Las 80 filas de cs090_fase5b_TOTAL_40pares.csv vienen de 4 "lotes" de origen distintos, cada uno con su
  propio CSV de reglas clasificadas (que SI tiene la columna 'pendiente'):
    - cs090_fase5_profundizar_a2b0c2_resumen.csv (seccion origen=='nueva_profundizar' unicamente -- la
      seccion 'original_barrido180' no tiene seed y no se usa para el join)
    - cs090_fase5b_candidatas_v2.csv
    - cs090_fase5b_candidatas_v3.csv
    - cs090_fase5b_candidatas_v4.csv

  Se verifico primero (ver salida de consola) que estos 4 CSVs, combinados, tienen 430 filas con 430
  SEEDS UNICOS -- CERO colisiones de seed entre lotes (el seed_base de cada lote esta separado por >90000
  del anterior, tal como documenta cada informe de Fase V-B). Por eso el JOIN SE HACE POR SEED, no por
  rule_id: el rule_id es justamente lo que colisiono en el bug documentado (dos reglas fisicamente
  distintas compartieron el mismo nombre 'A2-B0-C2-r2' etc. en un momento de la linea de trabajo), pero
  el seed nunca colisiono.

  Para cada una de las 80 filas de cs090_fase5b_TOTAL_40pares.csv:
    1. buscar su 'seed' en la tabla maestra combinada (430 filas de los 4 CSVs de origen).
    2. si el seed no aparece: fila invalida, razon='seed no encontrado en ningun CSV de origen'.
    3. si aparece: comparar K, kcap y clase de la fila de origen contra K, kcap y clase que YA estan en
       cs090_fase5b_TOTAL_40pares.csv para esa fila. Los TRES deben coincidir exactamente.
    4. si algo no coincide: fila invalida, razon describe que campo no calzo (no se adivina, se excluye).
    5. si todo coincide: fila valida, se le asigna la 'pendiente' del CSV de origen.

PASO 2 -- con las filas validas (idealmente las 80, se documenta cuantas quedaron fuera y por que):
  - Spearman (pendiente vs fraccion_masa_en_sumideros) y (pendiente vs kappa_v_agregado), sobre las 80 (o
    N validas) filas INDIVIDUALES, no sobre diferencias de pares.
  - Regresion polinomial grado 1 y grado 2 (numpy.polyfit) para ver si aparece curvatura relevante.
  - LOWESS (implementacion manual liviana, sin dependencia de statsmodels que no esta instalado en este
    entorno -- kernel tricubico local, span configurable) como suavizado no parametrico adicional.
  - Grafico pendiente (x) vs fraccion_masa_en_sumideros (y), coloreado por clase ORIGINAL (I/III), con
    linea vertical en el umbral de clasificacion (pendiente=0.7) y la curva LOWESS superpuesta.
  - Nota metodologica honesta: se reporta el rango de pendientes efectivamente cubierto por los 80 (o N
    validas) puntos, y si hay o no puntos cerca del umbral 0.7 (la pregunta "que tan bien cubierto esta
    el rango intermedio" es tan importante como el numero de Spearman).

No modifica ningun script/CSV existente (solo lectura). No corre Phantom. No declara cierre ni veredicto.
"""
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
RUTA_TOTAL = f"{BASE}/cs090_fase5b_TOTAL_40pares.csv"
UMBRAL_CLASIFICACION = 0.7


def cargar_tabla_maestra():
    """Combina las 4 fuentes con columna 'pendiente', clave = seed (int). Verifica cero colisiones."""
    prof = pd.read_csv(f"{BASE}/cs090_fase5_profundizar_a2b0c2_resumen.csv")
    prof = prof[prof["origen"] == "nueva_profundizar"].copy()
    prof["seed"] = prof["seed"].astype(int)
    prof["fuente"] = "cs090_fase5_profundizar_a2b0c2_resumen.csv (nueva_profundizar)"

    v2 = pd.read_csv(f"{BASE}/cs090_fase5b_candidatas_v2.csv")
    v2["fuente"] = "cs090_fase5b_candidatas_v2.csv"
    v3 = pd.read_csv(f"{BASE}/cs090_fase5b_candidatas_v3.csv")
    v3["fuente"] = "cs090_fase5b_candidatas_v3.csv"
    v4 = pd.read_csv(f"{BASE}/cs090_fase5b_candidatas_v4.csv")
    v4["fuente"] = "cs090_fase5b_candidatas_v4.csv"

    cols = ["seed", "rule_id", "clase", "K", "kcap", "pendiente", "fuente"]
    maestra = pd.concat([prof[cols], v2[cols], v3[cols], v4[cols]], ignore_index=True)

    dup = maestra["seed"][maestra["seed"].duplicated()]
    assert dup.empty, f"COLISION DE SEED entre fuentes -- investigar antes de seguir: {dup.tolist()}"
    print(f"[tabla maestra] {len(maestra)} filas combinadas de 4 fuentes, seeds unicos: "
          f"{maestra['seed'].nunique()} -- sin colisiones, OK")
    return maestra.set_index("seed", drop=False)


def hacer_join(df_total, maestra):
    filas = []
    for _, row in df_total.iterrows():
        seed = int(row["seed"])
        registro = dict(
            par=row["par"], rule_id=row["rule_id"], clase=row["clase"], seed=seed,
            K_total=row["K"], kcap_total=row["kcap"],
            fraccion_masa_en_sumideros=row["fraccion_masa_en_sumideros"],
            kappa_v_agregado=row["kappa_v_agregado"],
            match_exacto_K_kcap=row["match_exacto_K_kcap"],
        )
        if seed not in maestra.index:
            registro.update(valido=False, razon="seed no encontrado en ningun CSV de origen",
                             pendiente=np.nan, fuente=None)
            filas.append(registro)
            continue

        origen = maestra.loc[seed]
        if isinstance(origen, pd.DataFrame):
            # no deberia pasar (ya verificamos seeds unicos), pero por robustez:
            registro.update(valido=False, razon="seed con multiples filas de origen (inesperado)",
                             pendiente=np.nan, fuente=None)
            filas.append(registro)
            continue

        problemas = []
        if int(origen["K"]) != int(row["K"]):
            problemas.append(f"K difiere (total={row['K']}, origen={origen['K']})")
        if int(origen["kcap"]) != int(row["kcap"]):
            problemas.append(f"kcap difiere (total={row['kcap']}, origen={origen['kcap']})")
        if str(origen["clase"]) != str(row["clase"]):
            problemas.append(f"clase difiere (total={row['clase']}, origen={origen['clase']})")

        if problemas:
            registro.update(valido=False, razon="; ".join(problemas),
                             pendiente=np.nan, fuente=origen["fuente"])
        else:
            registro.update(valido=True, razon="OK", pendiente=float(origen["pendiente"]),
                             fuente=origen["fuente"])
        filas.append(registro)
    return pd.DataFrame(filas)


def lowess_manual(x, y, frac=0.6, n_puntos=200):
    """LOWESS liviano (kernel tricubico local, grado 1) -- statsmodels no esta instalado en este venv."""
    orden = np.argsort(x)
    x_o, y_o = x[orden], y[orden]
    n = len(x_o)
    k = max(3, int(np.ceil(frac * n)))
    x_eval = np.linspace(x_o.min(), x_o.max(), n_puntos)
    y_eval = np.empty(n_puntos)
    for i, x0 in enumerate(x_eval):
        d = np.abs(x_o - x0)
        idx_vecinos = np.argsort(d)[:k]
        d_k = d[idx_vecinos]
        h = d_k.max() if d_k.max() > 0 else 1.0
        w = (1 - (d_k / h) ** 3) ** 3
        w = np.clip(w, 0, None)
        X = np.vstack([np.ones(k), x_o[idx_vecinos]]).T
        W = np.diag(w)
        try:
            beta = np.linalg.lstsq(W @ X, W @ y_o[idx_vecinos], rcond=None)[0]
            y_eval[i] = beta[0] + beta[1] * x0
        except np.linalg.LinAlgError:
            y_eval[i] = np.average(y_o[idx_vecinos], weights=w)
    return x_eval, y_eval


def analizar_metrica(dfv, metrica, nombre_corto):
    x = dfv["pendiente"].to_numpy()
    y = dfv[metrica].to_numpy()
    rho, p_rho = stats.spearmanr(x, y)

    # regresion lineal y cuadratica (numpy.polyfit) + R^2
    def r2(yhat):
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / ss_tot

    c1 = np.polyfit(x, y, 1)
    yhat1 = np.polyval(c1, x)
    r2_1 = r2(yhat1)

    c2 = np.polyfit(x, y, 2)
    yhat2 = np.polyval(c2, x)
    r2_2 = r2(yhat2)

    print(f"\n--- {metrica} (n={len(x)}) ---")
    print(f"Spearman rho={rho:.4f}, p={p_rho:.6f}")
    print(f"Lineal: y = {c1[1]:.4f} + {c1[0]:.4f}*x | R^2={r2_1:.4f}")
    print(f"Cuadratico: y = {c2[2]:.4f} + {c2[1]:.4f}*x + {c2[0]:.4f}*x^2 | R^2={r2_2:.4f} "
          f"(mejora sobre lineal: {r2_2 - r2_1:+.4f})")

    return dict(metrica=metrica, n=len(x), spearman_rho=rho, spearman_p=p_rho,
                lineal_intercepto=c1[1], lineal_pendiente=c1[0], lineal_R2=r2_1,
                cuadratico_c0=c2[2], cuadratico_c1=c2[1], cuadratico_c2=c2[0], cuadratico_R2=r2_2,
                mejora_R2_cuadratico_vs_lineal=r2_2 - r2_1)


def graficar(dfv, resumen_fraccion, ruta_png):
    x = dfv["pendiente"].to_numpy()
    y = dfv["fraccion_masa_en_sumideros"].to_numpy()
    x_low, y_low = lowess_manual(x, y, frac=0.6)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    for clase, color, marker in [("I", "#3b7dd8", "o"), ("III", "#e07b39", "^")]:
        sub = dfv[dfv["clase"] == clase]
        ax.scatter(sub["pendiente"], sub["fraccion_masa_en_sumideros"],
                   label=f"Clase {clase} (n={len(sub)})", color=color, marker=marker,
                   s=55, alpha=0.85, edgecolor="white", linewidth=0.6)

    ax.plot(x_low, y_low, color="#444444", linewidth=2.2, label="LOWESS (manual, frac=0.6)")
    ax.axvline(UMBRAL_CLASIFICACION, color="#999999", linestyle="--", linewidth=1.3,
               label=f"umbral clasificacion (pendiente={UMBRAL_CLASIFICACION})")

    ax.set_xlabel("Pendiente continua (log-log, observable de clasificacion I/III)")
    ax.set_ylabel("Fraccion de masa acretada en sumideros (Phantom)")
    ax.set_title(
        f"FASE VI -- Analisis 2: observable continuo vs respuesta gravitacional (n={len(dfv)})\n"
        f"Spearman rho={resumen_fraccion['spearman_rho']:.3f}, p={resumen_fraccion['spearman_p']:.4f}"
    )
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(ruta_png, dpi=150)
    print(f"\n[guardado] {ruta_png}")


def main():
    df_total = pd.read_csv(RUTA_TOTAL)
    maestra = cargar_tabla_maestra()
    joined = hacer_join(df_total, maestra)

    n_validas = int(joined["valido"].sum())
    n_invalidas = len(joined) - n_validas
    print(f"\n[join] {n_validas}/{len(joined)} filas validas (K, kcap y clase coinciden entre "
          f"TOTAL_40pares y el CSV de origen). {n_invalidas} filas excluidas.")
    if n_invalidas:
        print("[filas excluidas]")
        print(joined[~joined["valido"]][["par", "rule_id", "seed", "razon"]].to_string(index=False))

    ruta_join_csv = f"{BASE}/cs090_fase6_pendientes_unidas.csv"
    joined.to_csv(ruta_join_csv, index=False)
    print(f"[guardado] {ruta_join_csv}")

    dfv = joined[joined["valido"]].copy()

    # cobertura del rango de pendientes -- nota metodologica honesta
    print(f"\n[cobertura de pendiente] min={dfv['pendiente'].min():.4f} max={dfv['pendiente'].max():.4f}")
    banda = 0.05
    cerca_umbral = dfv[(dfv["pendiente"] >= UMBRAL_CLASIFICACION - banda) &
                        (dfv["pendiente"] <= UMBRAL_CLASIFICACION + banda)]
    print(f"[cobertura de pendiente] puntos dentro de +-{banda} del umbral {UMBRAL_CLASIFICACION}: "
          f"{len(cerca_umbral)}/{len(dfv)}")
    print("[distribucion por clase, resumen de pendiente]")
    print(dfv.groupby("clase")["pendiente"].describe()[["count", "min", "25%", "50%", "75%", "max"]]
          .to_string())

    resumen_fraccion = analizar_metrica(dfv, "fraccion_masa_en_sumideros", "fraccion")
    resumen_kappa = analizar_metrica(dfv, "kappa_v_agregado", "kappa_v")

    resumen = pd.DataFrame([resumen_fraccion, resumen_kappa])
    ruta_resumen_csv = f"{BASE}/cs090_fase6_observable_continuo_resumen.csv"
    resumen.to_csv(ruta_resumen_csv, index=False)
    print(f"\n[guardado] {ruta_resumen_csv}")

    ruta_png = f"{BASE}/cs090_fase6_observable_continuo.png"
    graficar(dfv, resumen_fraccion, ruta_png)


if __name__ == "__main__":
    main()
