"""
cs090_fase8_f803_robustez.py — FASE VIII F8-03: ¿el residuo que queda cuando se condiciona por el pico
local depende de CÓMO se mide el pico, o de la forma que se le supone a la relación?
=======================================================================================================

El número central de F8-03 sale de un modelo: masa ~ brazo + pico, con efecto fijo de grafo. Ese modelo
supone (a) que el pico es `p90/mediana`, (b) que entra linealmente. Acá se sacude las dos suposiciones:

  - se repite con **otras tres varas del pico** (CV de la densidad, máximo/mediana, y el rango),
  - se repite en **rangos** (Spearman parcial), que no supone ninguna forma,
  - se ajusta, a nivel de PAR (n=12), `Δmasa ~ Δpico + Δsoporte`: los dos canales a la vez,
  - se reporta cuánto vale el residuo en partículas y contra el piso práctico (~5 partículas, F8-01).

No corre Phantom, no toca ninguna carpeta: es re-análisis puro sobre los CSV que ya escribió
`cs090_fase8_f803_analizar.py`. No declara cierre ni veredicto.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
GRANO = 0.0005


def ols(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    gl = max(len(y) - X.shape[1], 1)
    s2 = float(resid @ resid / gl)
    ee = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    t = b / ee
    p = 2 * stats.t.sf(np.abs(t), gl)
    return b, ee, p


def main():
    D = pd.read_csv(f"{HERE}/cs090_fase8_f803_phantom_crudo.csv")
    C = pd.read_csv(f"{HERE}/cs090_fase8_f803_pares_con_masa.csv")
    D["es_solap"] = (D.brazo == "solap").astype(float)
    filas = []

    # ---------- 1) el residuo del brazo con distintas varas del pico (centrado por grafo) ----------
    for vara in ("pico_p90_med", "pico_cv", "pico_max_med", "log_pico_p90_med"):
        Z = D.copy()
        if vara.startswith("log_"):
            Z[vara] = np.log(Z[vara[4:]])
        for c in ("frac_masa", "es_solap", vara):
            Z[c + "_c"] = Z[c] - Z.groupby("clave")[c].transform("mean")
        X = np.column_stack([Z["es_solap_c"], Z[vara + "_c"], np.ones(len(Z))])
        b, ee, p = ols(X, Z["frac_masa_c"].values)
        # el mismo contraste sin la vara, para la fracción que sobrevive
        X0 = np.column_stack([Z["es_solap_c"], np.ones(len(Z))])
        b0, *_ = np.linalg.lstsq(X0, Z["frac_masa_c"].values, rcond=None)
        rp = stats.spearmanr(Z["es_solap_c"], Z["frac_masa_c"])
        filas.append(dict(analisis=f"residuo del brazo condicionando por {vara}",
                          n=len(Z), efecto_bruto_part=b0[0] / GRANO,
                          residuo_part=b[0] / GRANO, ee_part=ee[0] / GRANO, p=p[0],
                          frac_sobrevive=b[0] / b0[0]))

    # ---------- 2) en rangos: Spearman parcial (sin suponer forma) ----------
    Z = D.copy()
    for c in ("frac_masa", "es_solap", "pico_p90_med", "frac_aristas_en_triangulo"):
        Z[c + "_c"] = Z[c] - Z.groupby("clave")[c].transform("mean")

    def parcial(x, y, z):
        rxz = stats.spearmanr(x, z).statistic
        ryz = stats.spearmanr(y, z).statistic
        rxy = stats.spearmanr(x, y).statistic
        return (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))

    filas.append(dict(analisis="Spearman parcial: masa vs SOPORTE descontando el pico",
                      n=len(Z), residuo_part=float("nan"),
                      rho=parcial(Z["frac_aristas_en_triangulo_c"], Z["frac_masa_c"],
                                  Z["pico_p90_med_c"])))
    filas.append(dict(analisis="Spearman parcial: masa vs PICO descontando el soporte",
                      n=len(Z), residuo_part=float("nan"),
                      rho=parcial(Z["pico_p90_med_c"], Z["frac_masa_c"],
                                  Z["frac_aristas_en_triangulo_c"])))

    # ---------- 3) a nivel de PAR: los dos canales juntos (n=12) ----------
    y = C["d_masa_part"].values
    X = np.column_stack([C["d_pico"].values, C["d_soporte"].values, np.ones(len(C))])
    b, ee, p = ols(X, y)
    filas.append(dict(analisis="PAR: Δmasa ~ Δpico + Δsoporte — coef. Δpico [part./unidad]",
                      n=len(C), residuo_part=b[0], ee_part=ee[0], p=p[0]))
    filas.append(dict(analisis="PAR: Δmasa ~ Δpico + Δsoporte — coef. Δsoporte [part./unidad]",
                      n=len(C), residuo_part=b[1], ee_part=ee[1], p=p[1]))
    filas.append(dict(analisis="PAR: Δmasa ~ Δpico + Δsoporte — ORDENADA (Δpico=0, Δsoporte=0)",
                      n=len(C), residuo_part=b[2], ee_part=ee[2], p=p[2]))
    # sólo con el pico, para leer la ordenada como "Δmasa esperado con el pico igualado"
    X1 = np.column_stack([C["d_pico"].values, np.ones(len(C))])
    b1, ee1, p1 = ols(X1, y)
    filas.append(dict(analisis="PAR: Δmasa ~ Δpico — ORDENADA (extrapolación a Δpico=0)",
                      n=len(C), residuo_part=b1[1], ee_part=ee1[1], p=p1[1],
                      pendiente=b1[0]))

    R = pd.DataFrame(filas)
    R.to_csv(f"{HERE}/cs090_fase8_f803_robustez.csv", index=False)
    pd.set_option("display.width", 200)
    print(R.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---------- 4) el efecto por unidad de contraste topológico, igualados vs no ----------
    C["masa_por_soporte"] = C["d_masa_part"] / (-C["d_soporte"])
    print("\nΔmasa por unidad de Δsoporte (partículas por unidad de fracción de aristas):")
    print(C.groupby("igualado")["masa_por_soporte"].describe()[["count", "mean", "50%"]]
          .to_string(float_format=lambda v: f"{v:.1f}"))
    print("\nΔsoporte medio: igualados %.4f  no igualados %.4f" % (
        C[C.igualado]["d_soporte"].mean(), C[~C.igualado]["d_soporte"].mean()))


if __name__ == "__main__":
    main()
