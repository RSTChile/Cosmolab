"""
cs090_fase6_o3a_grafos_por_N.py — FASE VI, tarea O3-A: ¿el par sigue siendo el mismo par al subir N?
(11-ago-2026)

POR QUÉ HACE FALTA ESTE CHEQUEO
-------------------------------
En este pipeline los nodos del grafo SON las partículas SPH, así que subir la resolución reconstruye la
misma REGLA sobre un grafo MÁS GRANDE (ver §1.1 de `FASE6_O3A_convergencia_resolucion_CS.md`). La
etiqueta Clase I / Clase III se la puso el clasificador de Fase V-A mirando métricas del grafo a N=2000.
Si al pasar a N=4000 la regla "Clase III" dejara de comportarse como Clase III, el par dejaría de ser el
par y la comparación de Δmasa entre resoluciones estaría comparando otra cosa.

Volver a correr el clasificador completo a N=4000 (que necesita el barrido de escalas b=1,2,4,8,16 por
regla) costaría del orden de 20-40 s por regla × 26 reglas, fuera del presupuesto de esta tarea. En su
lugar se hace el chequeo BARATO y directo: se comparan, regla por regla, las métricas del grafo final a
N=2000 (guardadas en los `meta_regla.json` de las carpetas de Fase V-B) contra las mismas métricas a
N=4000 (guardadas por el worker de esta tarea), y se mira si **el ORDEN entre la Clase III y la Clase I
de cada par se conserva** en grado medio, diámetro de la componente gigante, fracción de componente
gigante y holonomía.

El diámetro que se compara a N=4000 es el de la medición OFICIAL `cs090_diam_corregido.diam_gigante`
(regla vigente desde el 11-ago-2026); el de N=2000 quedó guardado con el `_diam` viejo de cs055, así que
la columna se marca como tal y NO se usa para concluir nada — se reporta sólo como referencia. El
chequeo de orden se apoya en grado medio, giant y holonomía, que sí se midieron igual en las dos.

Salida: `cs090_fase6_o3a_grafos_por_N.csv`. No declara cierre ni veredicto.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd

HERE = Path("/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis")
SEL_JSON = HERE / "cs090_fase6_o3a_pares_seleccionados.json"
SALIDA = HERE / "cs090_fase6_o3a_grafos_por_N.csv"
BASES_N2000 = ["/Users/alexis/phantom_cs073/bateria_fase5b_a2b0c2_piloto",
               "/Users/alexis/phantom_cs073/bateria_fase5b_a2b0c2_escala_v2",
               "/Users/alexis/phantom_cs073/bateria_fase5b_a2b0c2_escala_v3",
               "/Users/alexis/phantom_cs073/bateria_fase5b_a2b0c2_escala_v4"]
BASE_N4000 = "/Users/alexis/phantom_cs073/bateria_fase6_o3a_resolucion/N4000"


def main() -> pd.DataFrame:
    m2000 = {}
    for b in BASES_N2000:
        for p in glob.glob(b + "/*/meta_regla.json"):
            m = json.loads(Path(p).read_text())
            m2000[m["rule_id"]] = m
    m4000 = {}
    for p in glob.glob(BASE_N4000 + "/*/meta_regla.json"):
        m = json.loads(Path(p).read_text())
        m4000[m["rule_id"]] = m

    sel = json.loads(SEL_JSON.read_text())
    filas = []
    for s in sel:
        if s["rid_I"] not in m4000 or s["rid_III"] not in m4000:
            continue
        f = dict(par=s["par"])
        for rol, rid in (("I", s["rid_I"]), ("III", s["rid_III"])):
            a, b = m2000[rid], m4000[rid]
            f[f"{rol}_rule_id"] = rid
            f[f"{rol}_grado_2000"] = a["grado_medio_grafo_final"]
            f[f"{rol}_grado_4000"] = b["grado_medio_grafo_final"]
            f[f"{rol}_giant_2000"] = a["giant_grafo_final"]
            f[f"{rol}_giant_4000"] = b["giant_grafo_final"]
            f[f"{rol}_holon_2000"] = a["holon_grafo_final"]
            f[f"{rol}_holon_4000"] = b["holon_grafo_final"]
            f[f"{rol}_diam_2000_cs055_viejo"] = a["diam_grafo_final"]
            f[f"{rol}_diam_4000_OFICIAL"] = b["diam_grafo_final_OFICIAL"]
        for met in ("grado", "giant", "holon"):
            for N in (2000, 4000):
                f[f"orden_{met}_{N}"] = (
                    "III>I" if f[f"III_{met}_{N}"] > f[f"I_{met}_{N}"]
                    else ("III<I" if f[f"III_{met}_{N}"] < f[f"I_{met}_{N}"] else "="))
            f[f"orden_{met}_se_conserva"] = f[f"orden_{met}_2000"] == f[f"orden_{met}_4000"]
        filas.append(f)

    d = pd.DataFrame(filas)
    d.to_csv(SALIDA, index=False)
    print(f"[grafos] {len(d)} pares -> {SALIDA}")
    if len(d):
        for met in ("grado", "giant", "holon"):
            print(f"  orden III vs I en '{met}' se conserva de N=2000 a N=4000 en "
                  f"{int(d[f'orden_{met}_se_conserva'].sum())}/{len(d)} pares")
        pd.set_option("display.width", 240)
        print(d[["par"] + [c for c in d.columns if c.startswith("orden_")]].to_string(index=False))
    return d


if __name__ == "__main__":
    main()
