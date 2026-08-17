"""
cs090_fase6_outliers_paso3_phantom.py -- FASE VI, investigacion de los "casos raros" (PASO 3).

QUIEN SOY: corro Phantom sobre las reglas A2-B0-C2 de PENDIENTE MUY NEGATIVA que todavia NO habian
pasado por Phantom (identificadas por el Paso 2, ver cs090_fase6_outliers_candidatas.csv), con EXACTAMENTE
el mismo protocolo que las 80 corridas de Fase V-B, para responder una sola pregunta:

    los 3 outliers originales (A2-B0-C2-batch3-r100/-batch4-r51/-batch3-r143) dieron fraccion de masa
    ALTA (0.10-0.15) pese a tener pendiente muy negativa. ¿Las OTRAS reglas de pendiente muy negativa
    hacen lo mismo (regimen reproducible) o se dispersan por todo el rango (los 3 eran ruido de n chico)?

Protocolo -- identico al de Fase V-B, sin ninguna perilla nueva:
  - reconstruccion del grafo: `reconstruir_regla_a2b0c2` (cs090_fase5b_phantom_adaptador, solo import)
  - condicion inicial: `generar_ic_masa_fija_desde_grafo` (mismo adaptador), N=2000, masa total fija
    18800, lado de caja fijo 2000^(1/3), seed_layout=12345, turbulencia Mach=3 seed=42
  - Phantom: mismo binario y misma edicion de `cosmog.in` que `cs090_fase5b_correr.py` (icreate_sinks=1,
    rho_crit_cgs=1000, r_crit=0.6, h_acc=0.3, tmax=0.500, dtmax=0.001) -- se REUSA `correr_una` de ese
    script congelado, no se reimplementa
  - metricas: `analizar_sink` de `cs090_fase5b_analizar.py` (solo esa funcion; NO se llama a su `main()`,
    que escribiria sobre cs090_fase5b_metricas.csv de la fase anterior)

NOTA HONESTA SOBRE UNA DESVIACION FORZADA POR EL ENTORNO (no es una eleccion de metodo): en esta maquina
`sarracen` (la libreria que lee los volcados BINARIOS de Phantom, usada por `leer_volcado_phantom.py`) ya
no esta instalada en ningun interprete -- verificado en python3.9/3.10/3.11/3.13 y por busqueda en todo el
disco. Por eso la fraccion de masa NO se lee del dump binario final sino del log `.sink`, como
masa_acretada_total / 18800 (la masa total del sistema es fija por construccion de la IC). Esto NO es una
metrica distinta: se comprobo sobre las 80 corridas ya existentes de Fase V-B (que tienen AMBOS numeros en
`cs090_fase5b_TOTAL_40pares.csv`) que `masa_acretada_total/18800` reproduce
`fraccion_masa_en_sumideros` con una diferencia maxima de 5.6e-17 en las 80 filas -- es el mismo numero
hasta el ultimo bit del punto flotante. `analizar_sink` (que da n_sumideros, t_primer_sumidero, kappa_V y
masa_acretada_total) no usa sarracen y se importa tal cual del script congelado.

DOBLE VERIFICACION CRUZADA (esta linea tuvo un bug real de colision de nombres de regla entre lotes, ver
FASE5B_investigacion_8sumideros_y_escala_CS.md 2.1 -- no se repite):
  (1) ANTES de generar la IC: el `p` que devuelve el generador a partir del `seed` debe coincidir en
      K, J, noise, meandeg, kcap y sim_thr_frac con lo que dice el CSV de origen para ese rule_id; y el
      seed no puede estar entre los que ya corrieron Phantom en Fase V-B.
  (2) DESPUES de correr: se relee `meta_regla.json` de la carpeta y se exige que su rule_id/seed/K/kcap
      coincidan con los del CSV -- si una carpeta quedo con la meta de otra regla, el script lo grita.

Carpeta de salida NUEVA (no pisa ninguna bateria anterior):
/Users/alexis/phantom_cs073/bateria_fase6_outliers_negativos

No modifica ningun script ni CSV existente. No declara cierre ni veredicto -- reporta numeros.
"""
from __future__ import annotations
import csv
import json
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis")

import cs090_fase5_generador as GEN
from cs090_fase5b_phantom_adaptador import reconstruir_regla_a2b0c2, generar_ic_masa_fija_desde_grafo
from cs090_fase5b_correr import correr_una          # congelado -- mismo setup+run+edicion de cosmog.in

# cs090_fase5b_analizar.py (congelado) importa `leer_volcado_phantom`, que a su vez importa `sarracen`
# -- ausente en esta maquina (ver nota en el docstring). Para poder reusar su `analizar_sink` SIN tocar
# ni una linea del archivo congelado, se registra un modulo-stub con la firma que ese import espera; las
# dos funciones del stub NUNCA se llaman (solo se usa `analizar_sink`, que lee el `.sink` con numpy).
if "leer_volcado_phantom" not in sys.modules:
    _stub = types.ModuleType("leer_volcado_phantom")
    def _no_disponible(*a, **k):
        raise RuntimeError("leer_dump/listar_dumps no disponibles: sarracen no esta instalado; "
                           "este paso mide la fraccion de masa desde el .sink (ver docstring)")
    _stub.leer_dump = _no_disponible
    _stub.listar_dumps = _no_disponible
    sys.modules["leer_volcado_phantom"] = _stub
from cs090_fase5b_analizar import analizar_sink     # congelado -- solo esta funcion, nunca su main()

MASA_TOTAL_SISTEMA = 18800.0   # fija por construccion de la IC (misma constante que el adaptador)

HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
BASE_SALIDA = Path("/Users/alexis/phantom_cs073/bateria_fase6_outliers_negativos")

RUTA_CANDIDATAS = f"{HERE}/cs090_fase6_outliers_candidatas.csv"
RUTA_80 = f"{HERE}/cs090_fase5b_TOTAL_40pares.csv"
RUTA_METRICAS = f"{HERE}/cs090_fase6_outliers_phantom_metricas.csv"

N = 2000
SEED_LAYOUT = 12345      # MISMO que Fase V-B (v1..v4) -- no se cambia
N_SWEEPS = 14


def seeds_ya_corridos():
    s = set()
    with open(RUTA_80) as f:
        for row in csv.DictReader(f):
            s.add(int(row["seed"]))
    return s


def cargar_candidatas():
    with open(RUTA_CANDIDATAS) as f:
        return list(csv.DictReader(f))


def verificacion_1_antes(fila, p):
    """Cruce CSV <-> parametros regenerados desde el seed. Devuelve lista de discrepancias (vacia=OK)."""
    problemas = []
    for campo, tol in (("K", 0), ("kcap", 0), ("J", 1e-9), ("noise", 1e-9),
                       ("meandeg", 1e-9), ("sim_thr_frac", 1e-9)):
        v_csv = float(fila[campo]); v_p = float(p[campo])
        if abs(v_csv - v_p) > tol:
            problemas.append(f"{campo}: CSV={v_csv} vs regenerado={v_p}")
    return problemas


def verificacion_2_despues(fila, carpeta):
    """Relee meta_regla.json ya escrito en disco y lo cruza contra el CSV."""
    meta = json.loads((carpeta / "meta_regla.json").read_text())
    problemas = []
    if meta["rule_id"] != fila["rule_id"]:
        problemas.append(f"rule_id: meta={meta['rule_id']} vs CSV={fila['rule_id']}")
    if int(meta["seed"]) != int(fila["seed"]):
        problemas.append(f"seed: meta={meta['seed']} vs CSV={fila['seed']}")
    if int(meta["K"]) != int(float(fila["K"])) or int(meta["kcap"]) != int(float(fila["kcap"])):
        problemas.append(f"K/kcap: meta={meta['K']}/{meta['kcap']} vs CSV={fila['K']}/{fila['kcap']}")
    return problemas, meta


def generar_ic(fila):
    rid, seed = fila["rule_id"], int(fila["seed"])
    p = GEN.generar_regla("A2", "B0", "C2", idx=0, seed=seed)
    problemas = verificacion_1_antes(fila, p)
    assert not problemas, f"[VERIF-1 FALLA] {rid}: " + "; ".join(problemas)

    carpeta = BASE_SALIDA / f"{rid}_pendNEG"
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_ic = carpeta / "cosmogenesis_ic.txt"

    t0 = time.time()
    p2, m = reconstruir_regla_a2b0c2(seed=seed, N=N, n_sweeps=N_SWEEPS)
    t_rec = time.time() - t0
    t1 = time.time()
    info = generar_ic_masa_fija_desde_grafo(m["adj_final"], N=N, seed_layout=SEED_LAYOUT,
                                            ruta_salida=str(ruta_ic))
    t_ic = time.time() - t1

    meta = dict(rule_id=rid, clase=fila["clase"], seed=seed, N=N, seed_layout=SEED_LAYOUT,
                pendiente_csv=float(fila["pendiente"]), fuente_csv=fila["fuente"],
                K=p2["K"], J=p2["J"], noise=p2["noise"], meandeg=p2["meandeg"], kcap=p2["kcap"],
                sim_thr_frac=p2["sim_thr_frac"], n_aristas_grafo_final=m["n_aristas"],
                diam_grafo_final=m["diam"], giant_grafo_final=m["giant"],
                holon_grafo_final=m["holonomia"], grado_medio_grafo_final=2.0 * m["n_aristas"] / N,
                masa_total_ic=info["masa_total"], carpeta=str(carpeta), ruta_ic=str(ruta_ic),
                t_reconstruir_grafo_s=round(t_rec, 2), t_generar_ic_s=round(t_ic, 2))
    (carpeta / "meta_regla.json").write_text(json.dumps(meta, indent=2))
    print(f"  [IC] {rid} seed={seed} pend={float(fila['pendiente']):+.3f} aristas={m['n_aristas']} "
          f"giant={m['giant']:.3f} ({t_rec:.0f}s+{t_ic:.0f}s)", flush=True)
    return carpeta, meta


def main():
    cands = cargar_candidatas()
    ya = seeds_ya_corridos()
    seeds = [int(c["seed"]) for c in cands]
    assert len(set(seeds)) == len(seeds), "seeds repetidos entre candidatas"
    solapadas = [c["rule_id"] for c in cands if int(c["seed"]) in ya]
    assert not solapadas, f"candidatas que YA corrieron Phantom en Fase V-B: {solapadas}"
    print(f"[paso3] {len(cands)} candidatas de pendiente muy negativa, ninguna corrio Phantom antes.\n"
          f"        salida: {BASE_SALIDA}", flush=True)
    BASE_SALIDA.mkdir(parents=True, exist_ok=True)

    t_ini = time.time()
    carpetas = []
    for c in cands:
        carpeta, _ = generar_ic(c)
        carpetas.append((c, carpeta))
    print(f"[paso3] ICs listas en {time.time()-t_ini:.0f}s. Corriendo Phantom...", flush=True)

    filas = []
    for c, carpeta in carpetas:
        t0 = time.time()
        info = correr_una(carpeta)
        problemas, meta = verificacion_2_despues(c, carpeta)
        assert not problemas, f"[VERIF-2 FALLA] {c['rule_id']}: " + "; ".join(problemas)
        fila = dict(carpeta=carpeta.name, rule_id=meta["rule_id"], clase=meta["clase"],
                    seed=meta["seed"], K=meta["K"], J=meta["J"], noise=meta["noise"],
                    meandeg=meta["meandeg"], kcap=meta["kcap"],
                    n_aristas_grafo_final=meta["n_aristas_grafo_final"],
                    diam_grafo_final=meta["diam_grafo_final"])
        fila.update(analizar_sink(carpeta / "cosmog01.sink"))
        fila["masa_total_sistema"] = MASA_TOTAL_SISTEMA
        fila["fraccion_masa_en_sumideros"] = fila["masa_acretada_total"] / MASA_TOTAL_SISTEMA
        fila["pendiente"] = float(c["pendiente"])
        fila["clase_csv"] = c["clase"]
        fila["fuente_csv"] = c["fuente"]
        fila["giant_grafo_final"] = meta["giant_grafo_final"]
        fila["grado_medio_grafo_final"] = meta["grado_medio_grafo_final"]
        fila["t_run_s"] = info.get("t_run", 0.0)
        filas.append(fila)
        print(f"  [PHANTOM] {c['rule_id']} pend={fila['pendiente']:+.3f} -> "
              f"fraccion_masa={fila.get('fraccion_masa_en_sumideros')} "
              f"n_sumideros={fila.get('n_sumideros')} kappa_v={fila.get('kappa_v_agregado')} "
              f"({time.time()-t0:.0f}s)", flush=True)

        campos = []
        for f in filas:
            for k in f:
                if k not in campos:
                    campos.append(k)
        with open(RUTA_METRICAS, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
            w.writeheader(); w.writerows(filas)

    print(f"\n[paso3] {len(filas)} corridas -> {RUTA_METRICAS}  (total {time.time()-t_ini:.0f}s)")


if __name__ == "__main__":
    main()
