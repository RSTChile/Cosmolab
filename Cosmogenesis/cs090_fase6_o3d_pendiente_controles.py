"""
cs090_fase6_o3d_pendiente_controles.py — FASE VI, tarea O3-D, chequeo adicional.

POR QUÉ EXISTE. El barrido encontró que los controles Erdős-Rényi emparejados en aristas acretan
prácticamente la misma masa que las reglas del kcap correspondiente. Eso deja una pregunta abierta que
se puede contestar sin correr ni un segundo más de Phantom: **¿los controles caen SOBRE la recta
masa-vs-pendiente de las reglas, o al costado?**

  - Si caen SOBRE la recta: la pendiente (la geometría medida) es una buena variable resumen — el
    control simplemente tiene la misma geometría que la regla, y por eso la misma masa.
  - Si caen AL COSTADO (misma masa, pendiente muy distinta): la pendiente NO es lo que está moviendo la
    masa; sería un tercer factor común (la densidad de aristas del grafo que entra al layout).

Se mide la pendiente de los 6 grafos de control con EXACTAMENTE la misma vara que las reglas: se
regenera el grafo Erdős-Rényi con `generar_grafo_erdos_renyi(2000, n_aristas, seed)` (la misma llamada,
misma semilla, que usó `generar_control_random_masa_fija` para construir su condición inicial, así que
es el MISMO grafo bit a bit), se lo agrupa con `cs080_renormalizacion.cajas_bfs`/`grafo_grueso` a las
escalas b = 1, 2, 4, 8, 16, se mide el diámetro con `cs090_diam_corregido.diam_gigante` y se ajusta la
recta log-log con `cs090_fase5_clasificador._pendiente_loglog`. Ninguna pieza se reimplementa.

Escribe `cs090_fase6_o3d_pendiente_controles.csv`. No modifica nada. No declara cierre ni veredicto.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)

import cs080_renormalizacion as CS80
from cs090_diam_corregido import diam_gigante
from cs090_fase5_clasificador import _pendiente_loglog
from grafo_random_layout_generar_ic import generar_grafo_erdos_renyi

BASE = Path("/Users/alexis/phantom_cs073/bateria_fase6_o3d_kcap")
N = 2000
ESCALAS_B = (1, 2, 4, 8, 16)
RUTA_SALIDA = Path(f"{_HERE}/cs090_fase6_o3d_pendiente_controles.csv")


def pendiente_de_grafo(adj_dict, seed_para_cajas):
    """MISMO procedimiento de coarse-graining que `cs090_fase5_motor.correr_regla_coarse`: se agrupa el
    grafo en cajas por BFS a escalas b crecientes y se ajusta log(diámetro) vs log(nº de supernodos).
    El rng de las cajas se deriva de la semilla del grafo, igual que en el motor."""
    adj = [set(adj_dict[i]) for i in range(N)]
    Ns, diams = [], []
    for b in ESCALAS_B:
        if b == 1:
            adj_g, n_cajas = adj, N
        else:
            rng_b = np.random.default_rng(seed_para_cajas * 7000 + b * 31)
            asign, n_cajas = CS80.cajas_bfs(adj, N, b, rng_b)
            adj_g = CS80.grafo_grueso(adj, N, asign, n_cajas)
        if n_cajas > 1:
            Ns.append(n_cajas)
            diams.append(float(diam_gigante(adj_g, n_cajas)))
    return _pendiente_loglog(Ns, diams), list(zip(Ns, diams))


def main():
    filas = []
    for carpeta in sorted(BASE.glob("CONTROL-ER-*")):
        meta = json.loads((carpeta / "meta_regla.json").read_text())
        res = json.loads((carpeta / "resultado_o3d.json").read_text())
        seed = int(meta["seed"]); n_ar = int(meta["n_aristas_grafo_final"])
        adj, edge_set, _ = generar_grafo_erdos_renyi(N, n_ar, seed=seed)
        assert len(edge_set) == n_ar, f"{carpeta.name}: el grafo regenerado tiene {len(edge_set)} aristas, no {n_ar}"
        pend, detalle = pendiente_de_grafo(adj, seed)
        fila = dict(rule_id=meta["rule_id"], seed=seed, n_aristas=n_ar,
                    grado_medio=round(2.0 * n_ar / N, 3),
                    pendiente_corregida=round(pend, 4),
                    fraccion_masa_en_sumideros=res["fraccion_masa_en_sumideros"],
                    kappa_v_agregado=res["kappa_v_agregado"],
                    diam_b1=detalle[0][1] if detalle else None)
        filas.append(fila)
        print(f"  {fila['rule_id']}: aristas={n_ar} pendiente={pend:.3f} "
              f"frac_masa={fila['fraccion_masa_en_sumideros']:.4f}", flush=True)

    with open(RUTA_SALIDA, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader(); w.writerows(filas)
    print(f"[pendiente controles] {len(filas)} filas -> {RUTA_SALIDA}")


if __name__ == "__main__":
    main()
