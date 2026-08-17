"""
cs090_fase5b_analizar.py — FASE V-B: extrae métricas de las corridas de Phantom sobre A2-B0-C2
(masa fija, N=2000) generadas por `cs090_fase5b_generar_pares.py` y corridas por `cs090_fase5b_correr.py`.

Reusa `leer_volcado_phantom.py` (congelado, sólo import) para leer el dump binario final de cada
corrida, y el mismo método de `cs078_kappaV_permutacion.py` (NO se importa ese archivo -- se reimplementa
la lectura de `.sink`, 5 líneas, para no acoplar esta tarea a un script de otra fase; misma lógica,
mismas columnas verificadas contra el header real del .sink, ver docstring de cada función abajo) para
calcular κ_V por sumidero y agregado por corrida.

Métricas por corrida (una fila por regla/carpeta):
  - n_gas_inicial: partículas de gas al t=0 (siempre 2000 salvo pérdida por setup).
  - n_sumideros: cuántos sumideros nacieron durante la corrida (t=0 a t=tmax=0.5).
  - masa_sumideros_final / masa_gas_final / masa_total_final: leídas del ÚLTIMO dump binario disponible.
  - fraccion_masa_en_sumideros: masa_sumideros_final / masa_total_final.
  - t_primer_sumidero: primer tiempo que aparece en el .sink (si existe) -- None si nunca se formó ninguno.
  - masa_acretada_total: suma de macc final (última fila por sink ID) del .sink -- mismo valor que
    masa_sumideros_final salvo redondeo, se reporta igual como chequeo cruzado independiente (una viene
    del dump binario final, la otra del log incremental .sink).
  - kappa_v_agregado / kappa_v_medio_valido / n_kappa_indefinidos: MISMA definición que CS073
    (masa_ultimo_tercio/masa_primer_tercio de la vida de cada sumidero, agregado Σúltimo/Σprimero y
    promedio de razones válidas -- NaN cuando masa_primer_tercio==0, ver cs078_kappaV_permutacion.py).

Escribe `cs090_fase5b_metricas.csv` en el mismo directorio de este script. No declara cierre ni
veredicto -- sólo extrae y tabula números.
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis")
from leer_volcado_phantom import leer_dump, listar_dumps   # congelado -- sólo import

BASE = Path("/Users/alexis/phantom_cs073/bateria_fase5b_a2b0c2_piloto")
RUTA_SALIDA_CSV = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs090_fase5b_metricas.csv"

# columnas del .sink -- verificadas contra el header real (bateria_n2000/ic_real/cosmog01.sink),
# MISMAS que usa cs078_kappaV_permutacion.py (COL_T=0, COL_MACC=11, COL_SINKID=18)
COL_T, COL_MACC, COL_SINKID = 0, 11, 18


def cargar_sink(path: Path) -> np.ndarray:
    data = np.loadtxt(path, skiprows=2)
    if data.ndim == 1:
        data = data[None, :]
    return data


def kappa_v_por_sumidero(t: np.ndarray, macc: np.ndarray) -> tuple[float, float]:
    """MISMA definición que cs078_kappaV_permutacion.py: masa acretada en el primer tercio de la vida
    del sumidero vs. en el último tercio -- interpolación lineal sobre las fronteras temporales."""
    t0, t1 = t[0], t[-1]
    duracion = t1 - t0
    frontera_1 = t0 + duracion / 3.0
    frontera_2 = t1 - duracion / 3.0
    m_inicio = np.interp(t0, t, macc)
    m_frontera1 = np.interp(frontera_1, t, macc)
    m_frontera2 = np.interp(frontera_2, t, macc)
    m_final = np.interp(t1, t, macc)
    return float(m_frontera1 - m_inicio), float(m_final - m_frontera2)


def analizar_sink(path_sink: Path) -> dict:
    if not path_sink.exists():
        return dict(n_sumideros=0, t_primer_sumidero=None, masa_acretada_total=0.0,
                    kappa_v_agregado=None, kappa_v_medio_valido=None, n_kappa_indefinidos=0)
    data = cargar_sink(path_sink)
    sink_ids = data[:, COL_SINKID].astype(int)
    t_primer_sumidero = float(data[:, COL_T].min())

    primeros, ultimos, razones = [], [], []
    masa_acretada_total = 0.0
    for sid in np.unique(sink_ids):
        sub = data[sink_ids == sid]
        orden = np.argsort(sub[:, COL_T])
        t = sub[orden, COL_T]
        macc = sub[orden, COL_MACC]
        masa_acretada_total += float(macc[-1])   # última masa acretada acumulada de este sumidero
        primero, ultimo = kappa_v_por_sumidero(t, macc)
        primeros.append(primero); ultimos.append(ultimo)
        razones.append(ultimo / primero if primero > 0 else np.nan)

    primeros = np.array(primeros); ultimos = np.array(ultimos); razones = np.array(razones)
    n_indef = int(np.isnan(razones).sum())
    kappa_agg = float(ultimos.sum() / primeros.sum()) if primeros.sum() > 0 else None
    kappa_med = float(np.nanmean(razones)) if n_indef < len(razones) else None

    return dict(n_sumideros=len(np.unique(sink_ids)), t_primer_sumidero=t_primer_sumidero,
                masa_acretada_total=masa_acretada_total, kappa_v_agregado=kappa_agg,
                kappa_v_medio_valido=kappa_med, n_kappa_indefinidos=n_indef)


def analizar_carpeta(carpeta: Path) -> dict:
    meta_path = carpeta / "meta_regla.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    dumps = listar_dumps(carpeta)
    fila = dict(carpeta=carpeta.name, rule_id=meta.get("rule_id"), clase=meta.get("clase"),
                seed=meta.get("seed"), K=meta.get("K"), J=meta.get("J"), noise=meta.get("noise"),
                meandeg=meta.get("meandeg"), kcap=meta.get("kcap"),
                n_aristas_grafo_final=meta.get("n_aristas_grafo_final"),
                diam_grafo_final=meta.get("diam_grafo_final"))

    if not dumps:
        fila.update(n_gas_inicial=None, n_dump_final=None, masa_gas_final=None,
                    masa_sumideros_final=None, masa_total_final=None, fraccion_masa_en_sumideros=None)
    else:
        gas0, sinks0 = leer_dump(dumps[0])
        fila["n_gas_inicial"] = len(gas0)

        dump_final = dumps[-1]
        gas, sinks = leer_dump(dump_final)
        fila["n_dump_final"] = dump_final.name
        m_gas = float(gas["m"].sum()) if "m" in gas.columns else \
            float(gas.params.get("massoftype", 0.0)) * len(gas)
        m_sinks = float(sinks["m"].sum()) if sinks is not None else 0.0
        fila["masa_gas_final"] = m_gas
        fila["masa_sumideros_final"] = m_sinks
        fila["masa_total_final"] = m_gas + m_sinks
        fila["fraccion_masa_en_sumideros"] = m_sinks / (m_gas + m_sinks) if (m_gas + m_sinks) > 0 else None

    path_sink = carpeta / "cosmog01.sink"
    fila.update(analizar_sink(path_sink))
    return fila


def main(carpetas=None):
    if carpetas is None:
        carpetas = sorted(p for p in BASE.iterdir() if p.is_dir())
    filas = []
    for carpeta in carpetas:
        dumps_existen = bool(listar_dumps(carpeta))
        if not dumps_existen:
            print(f"[{carpeta.name}] sin dumps todavía -- ¿corriste cs090_fase5b_correr.py?")
            continue
        print(f"[{carpeta.name}] analizando...", flush=True)
        fila = analizar_carpeta(carpeta)
        filas.append(fila)
        print(f"  clase={fila['clase']} n_sumideros={fila['n_sumideros']} "
              f"fraccion_masa_en_sumideros={fila.get('fraccion_masa_en_sumideros')} "
              f"kappa_v_agregado={fila.get('kappa_v_agregado')} "
              f"t_primer_sumidero={fila.get('t_primer_sumidero')}", flush=True)

    if filas:
        campos = list(filas[0].keys())
        for f in filas:
            for c in f:
                if c not in campos:
                    campos.append(c)
        with open(RUTA_SALIDA_CSV, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=campos)
            w.writeheader()
            w.writerows(filas)
        print(f"\n[TOTAL] {len(filas)} corridas analizadas -> {RUTA_SALIDA_CSV}")
    return filas


if __name__ == "__main__":
    main()
