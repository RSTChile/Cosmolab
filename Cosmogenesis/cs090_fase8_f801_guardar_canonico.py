"""
cs090_fase8_f801_guardar_canonico.py — FASE VIII F8-01: pasa los grafos ya guardados por esta tarea al
formato CANÓNICO de F8-00.

Cuando F8-01 arrancó, la utilidad `cs090_fase8_f800_grafos.py` todavía no existía, así que cada brazo
guardó su grafo como `grafo_f801.npz` (numpy comprimido: `aristas` E×2 int32 con i<j ordenado, `grados`
N int32). Este script lee ese npz, reconstruye la lista de adyacencia y la vuelve a escribir con
`F800.guardar_grafo` como `grafo_f801.grafo.gz` — con el sha256 canónico de F8-00 — y anota el sello en
el `meta_regla.json` de la carpeta. Los dos archivos quedan: el npz no se borra.

Idempotente: se puede correr muchas veces (por ejemplo mientras la batería todavía está generando).
No modifica ningún script ajeno; sólo importa `cs090_fase8_f800_grafos`.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, HERE)

import cs090_fase8_f800_grafos as F800          # sólo import

BASE = Path("/Users/alexis/phantom_cs073/bateria_fase8_f801_desacople")


def adj_desde_npz(ruta):
    d = np.load(ruta)
    ar, gr = d["aristas"], d["grados"]
    adj = [set() for _ in range(len(gr))]
    for i, j in ar:
        adj[int(i)].add(int(j))
        adj[int(j)].add(int(i))
    assert all(len(adj[k]) == int(gr[k]) for k in range(len(gr))), f"{ruta}: grados no coinciden"
    return adj, len(gr)


def main():
    hechos, saltados = 0, 0
    for carpeta in sorted(c for c in BASE.iterdir() if c.is_dir()):
        npz = carpeta / "grafo_f801.npz"
        if not npz.exists():
            saltados += 1
            continue
        destino = carpeta / "grafo_f801.grafo.gz"
        adj, N = adj_desde_npz(npz)
        sello = F800.hash_grafo(adj, N)
        F800.guardar_grafo(adj, destino, N=N, meta=dict(tarea="FASE8_F801", carpeta=carpeta.name))
        mp = carpeta / "meta_regla.json"
        if mp.exists():
            m = json.loads(mp.read_text())
            m["sha256_grafo"] = sello
            m["grafo_canonico_f800"] = destino.name
            mp.write_text(json.dumps(m, indent=2))
        hechos += 1
        print(f"  {carpeta.name}: sha256={sello[:16]}…", flush=True)
    print(f"[f801] {hechos} grafos en formato canónico F8-00; {saltados} carpetas sin npz")


if __name__ == "__main__":
    main()
