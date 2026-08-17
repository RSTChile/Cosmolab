"""
cs090_fase6_o3a_seleccionar_pares.py — FASE VI, tarea O3-A: elige los 12 pares que se re-corren a
resolución mayor, con un criterio ESCRITO ANTES de mirar ningún resultado nuevo (11-ago-2026).

POR QUÉ IMPORTA EL CRITERIO
---------------------------
Si uno eligiera "los pares donde el efecto se vio más lindo", el test de convergencia estaría amañado de
entrada: se estaría preguntando si sobreviven los casos favorables, no si sobrevive el efecto. Por eso el
criterio es mecánico y cubre TODO el rango, incluidos los pares donde a N=2000 ganó la Clase I (Δ<0):

  1. Se parte de los 40 pares de `cs090_fase5b_TOTAL_40pares.csv` (Fase V-B, N=2000).
  2. Se filtran los 37 pares LIMPIOS: `match_exacto_K_kcap == True` en AMBOS miembros, es decir la regla
     Clase III y la Clase I tienen EXACTAMENTE el mismo K y el mismo kcap. (Prioridad pedida por el
     equipo: los pares "sucios" mezclan la diferencia de clase con una diferencia de parámetros.)
  3. Se ordenan esos 37 por Δmasa = fracción_masa(III) − fracción_masa(I) a N=2000, de menor a mayor
     (desempate alfabético por nombre de par, para que el resultado sea reproducible bit a bit).
  4. Se toman 12 posiciones EQUIESPACIADAS en ese ranking: índices round(i·36/11) para i=0..11, o sea
     0, 3, 7, 10, 13, 16, 20, 23, 26, 29, 33, 36. El primero es el par MÁS invertido (peor caso para la
     hipótesis) y el último el más favorable; en el medio quedan los Δ chicos y los cercanos a cero.
  5. EMPATES: la fracción de masa está cuantizada (la masa por partícula es fija), así que hay muchos
     Δmasa repetidos; el desempate es alfabético por nombre de par y punto. En el índice 20 hay un
     empate exacto en Δ=+0.0085 entre `batch4-r18_vs_batch4-r19` y `batch4-r51_vs_batch4-r36`: gana el
     primero por orden alfabético. El par `r51_vs_r36` igual se corrió (había sido lanzado con una
     versión previa del desempate) y se REPORTA como par #13 en vez de tirar el cómputo — no introduce
     sesgo porque los dos empatados tienen el MISMO Δ a N=2000, y en el CSV/figura queda marcado como
     `extra_por_empate` para que se pueda recalcular todo sin él.

Resultado: 2 pares invertidos (Δ<0), 1 prácticamente en cero (Δ=+0.0005) y 9 positivos de todos los
tamaños (10 contando el par #13) — un muestreo del rango completo, no una selección de los favorables.

Escribe `cs090_fase6_o3a_pares_seleccionados.json` (lo consume `cs090_fase6_o3a_analizar.py` y de ahí
salen los trabajos que corre `cs090_fase6_o3a_convergencia_resolucion.py worker`). No modifica ningún
script congelado ni ningún CSV existente.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path("/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis")
CSV_ORIGEN = HERE / "cs090_fase5b_TOTAL_40pares.csv"
SALIDA_JSON = HERE / "cs090_fase6_o3a_pares_seleccionados.json"
N_PARES = 12
PAR_EXTRA_POR_EMPATE = "A2-B0-C2-batch4-r51_vs_A2-B0-C2-batch4-r36"


def main() -> pd.DataFrame:
    d = pd.read_csv(CSV_ORIGEN)
    filas = []
    for par, g in d.groupby("par"):
        assert len(g) == 2, f"{par}: se esperaban 2 filas (I y III), hay {len(g)}"
        I = g[g.rol == "I"].iloc[0]
        III = g[g.rol == "III"].iloc[0]
        filas.append(dict(
            par=par, rid_I=I.rule_id, seed_I=int(I.seed), rid_III=III.rule_id, seed_III=int(III.seed),
            K=int(I.K), kcap=int(I.kcap),
            limpio=bool(I.match_exacto_K_kcap and III.match_exacto_K_kcap),
            dfm_n2000=float(III.fraccion_masa_en_sumideros - I.fraccion_masa_en_sumideros),
            dkv_n2000=float(III.kappa_v_agregado - I.kappa_v_agregado)))
    r = pd.DataFrame(filas)
    limpios = r[r.limpio].sort_values(["dfm_n2000", "par"]).reset_index(drop=True)
    n = len(limpios)
    idx = sorted({int(round(i * (n - 1) / (N_PARES - 1))) for i in range(N_PARES)})
    sel = limpios.loc[idx].reset_index(drop=True)
    sel["extra_por_empate"] = False
    # par #13: empatado exactamente en Δ=+0.0085 con el del índice 20 (ver docstring, punto 5)
    extra = limpios[limpios.par == PAR_EXTRA_POR_EMPATE].copy()
    extra["extra_por_empate"] = True
    sel = pd.concat([sel, extra], ignore_index=True)
    SALIDA_JSON.write_text(json.dumps(sel.to_dict(orient="records"), indent=1))
    print(f"{n} pares limpios de {len(r)} totales; seleccionados {len(sel)} en los índices {idx}")
    print(sel.to_string(index=False))
    return sel


if __name__ == "__main__":
    main()
