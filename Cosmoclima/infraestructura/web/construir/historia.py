"""
LA HISTORIA CLIMÁTICA, A CUALQUIER ESCALA · país, región, provincia o comuna
=============================================================================

INSTRUCCIÓN (Alexis, 25-ago-2026): «para que nos sirva deberíamos poder
seleccionar qué comuna, región, provincia o país muestra».

El gráfico «Pluviosidad real en vivo» de Cosmoclima hace exactamente lo que hace
falta —60 años de lluvia con zoom, bandas de El Niño y La Niña— pero está atado a
UNA zona: la hiperárida costera de Huasco. Puesto bajo el mapa de la MICR estaría
hablando de Atacama mientras alguien mira Puerto Montt.

Aquí se construye la misma serie para **418 territorios**: el país, las 16
regiones, las 56 provincias y las 345 comunas.

★★ QUÉ ESTADÍSTICO SE USA AL AGREGAR, Y POR QUÉ TRES
------------------------------------------------------
Una región no tiene «una» lluvia: tiene un campo de lluvia. Coquimbo mezcla costa
seca y cordillera húmeda en la misma etiqueta, así que cualquier número único
miente un poco. Se guardan tres y el gráfico muestra lo que corresponda:

    mediana  lo típico del territorio — la línea principal
    p75      el cuartil alto, para ver la dispersión interna
    maximo   el peor punto — es el criterio que usa el resto de la aplicación
             para decidir riesgo, y tiene que poder compararse con él

⚠️ Para una comuna con una sola celda los tres coinciden, y está bien: no hay
dispersión que mostrar porque no hay dónde variar.

★ MENSUAL PARA LA VISTA LARGA, DIARIO PARA EL ZOOM
----------------------------------------------------
Mensual son 440 números por territorio y entra todo junto en 1,3 MB. Diario son
13.505 y sólo se baja de la comuna que se esté mirando (81 KB).

No es sólo peso: **son preguntas distintas**. Un mes con 100 mm repartidos no
rompe nada; tres días con 100 mm sí. La vista mensual sirve para «¿este año es
raro?» y la diaria para «¿este temporal cruza el umbral?».

USO
---
    ../../.venv-esa/bin/python construir/historia.py
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
DATOS = AQUI.parent / "publico" / "datos"
SERIE = RAIZ / "datos" / "clima_diario_celdas_era5land.csv"
SAL_MES = DATOS / "historia_mensual.json"
SAL_DIA = DATOS / "historia_diaria"


def cargar_serie():
    """Matriz (celdas × días) en float32. Son 3.647 × 13.505 ≈ 197 MB: cabe de
    sobra, y evita el diccionario de 45 millones de entradas que no cabría."""
    print("  leyendo la serie ERA5-Land…", flush=True)
    celdas, fechas = {}, {}
    filas = []
    with SERIE.open(encoding="utf-8") as fh:
        r = csv.reader(fh)
        next(r)
        for c, f, v in r:
            if v in ("", "None"):
                continue
            i = celdas.setdefault(c, len(celdas))
            j = fechas.setdefault(f, len(fechas))
            filas.append((i, j, float(v)))
    orden_f = sorted(fechas, key=lambda f: f)
    remap = {fechas[f]: k for k, f in enumerate(orden_f)}
    M = np.zeros((len(celdas), len(orden_f)), dtype=np.float32)
    for i, j, v in filas:
        M[i, remap[j]] = v
    print(f"  celdas {len(celdas):,} · días {len(orden_f):,} · "
          f"{M.nbytes/1e6:.0f} MB en memoria")
    return M, celdas, orden_f


def main():
    if not SERIE.exists():
        print("  falta la serie ERA5-Land")
        return 1
    M, idx_celda, fechas = cargar_serie()

    terr = json.loads((DATOS / "territorios.json").read_text(encoding="utf-8"))
    cel = json.loads((DATOS / "celdas_por_comuna.json").read_text(
        encoding="utf-8"))["por_comuna"]

    # ── qué celdas tiene cada territorio ────────────────────────────────────
    grupos = {"CL": set()}
    nombres = {"CL": "Chile"}
    for c in terr["comunas"]:
        cs = {k for k in cel.get(c["cut"], {}).get("celdas", []) if k in idx_celda}
        if not cs:
            continue
        grupos[f"C{c['cut']}"] = cs
        nombres[f"C{c['cut']}"] = c["comuna"]
        grupos.setdefault(f"P{c['cut_prov']}", set()).update(cs)
        grupos.setdefault(f"R{c['cut_reg']}", set()).update(cs)
        grupos["CL"] |= cs
    for p in terr["provincias"]:
        nombres[f"P{p['cut']}"] = p["nombre"]
    for r in terr["regiones"]:
        nombres[f"R{r['cut']}"] = r.get("nombre") or r.get("region") or r["cut"]

    print(f"  territorios: {len(grupos)}")

    # ── índices de mes, para agrupar los días ───────────────────────────────
    meses = sorted({f[:7] for f in fechas})
    pos_mes = {m: k for k, m in enumerate(meses)}
    col_mes = np.array([pos_mes[f[:7]] for f in fechas])

    salida = {}
    for clave, celdas in grupos.items():
        if not celdas:
            continue
        filas = [idx_celda[c] for c in celdas]
        sub = M[filas, :]                      # (celdas del territorio × días)
        # suma mensual de cada celda, y recién después el estadístico entre
        # celdas: al revés daría la mediana de días sueltos, que no es una lluvia
        # mensual de ningún lugar real.
        mens = np.zeros((sub.shape[0], len(meses)), dtype=np.float32)
        for k in range(len(meses)):
            mens[:, k] = sub[:, col_mes == k].sum(axis=1)
        salida[clave] = {
            "n": nombres.get(clave, clave),
            "celdas": len(celdas),
            "mediana": [round(float(v), 1) for v in np.median(mens, axis=0)],
            "p75": [round(float(v), 1) for v in np.percentile(mens, 75, axis=0)],
            "maximo": [round(float(v), 1) for v in mens.max(axis=0)],
        }

    SAL_MES.write_text(json.dumps({
        "meses": meses,
        "desde": fechas[0], "hasta": fechas[-1],
        "fuente": "ERA5-Land 0,1° · Copernicus",
        "estadisticos": {
            "mediana": "lo típico del territorio",
            "p75": "cuartil alto, muestra la dispersión interna",
            "maximo": "el peor punto — el criterio que usa el resto de la app",
        },
        "territorios": salida,
    }, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"\n  mensual: {SAL_MES.name} · {SAL_MES.stat().st_size/1e6:.2f} MB "
          f"· {len(salida)} territorios × {len(meses)} meses")

    # ── diario por comuna, para el zoom ─────────────────────────────────────
    SAL_DIA.mkdir(parents=True, exist_ok=True)
    for viejo in SAL_DIA.glob("*.json"):
        viejo.unlink()
    n = 0
    for clave, g in grupos.items():
        if not clave.startswith("C") or clave == "CL":
            continue
        filas = [idx_celda[c] for c in g]
        # el máximo entre celdas, igual que hace el mapa: para decidir si hay que
        # preocuparse manda el peor punto del territorio, no el promedio.
        serie = M[filas, :].max(axis=0)
        (SAL_DIA / f"{clave[1:]}.json").write_text(json.dumps({
            "desde": fechas[0],
            "mm": [round(float(v), 1) for v in serie],
        }, separators=(",", ":")), encoding="utf-8")
        n += 1
    pesos = [p.stat().st_size for p in SAL_DIA.glob("*.json")]
    print(f"  diario : {n} comunas · mediano {sorted(pesos)[len(pesos)//2]/1e3:.0f} KB"
          f" · total {sum(pesos)/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    print("=" * 74)
    print("HISTORIA CLIMÁTICA · país, región, provincia y comuna")
    print("=" * 74)
    sys.exit(main())
