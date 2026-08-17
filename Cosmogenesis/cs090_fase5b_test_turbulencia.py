"""
cs090_fase5b_test_turbulencia.py — FASE V-B, investigacion del "8 sumideros reiterado" (10-ago-2026).

Alexis vio que las 6 corridas del piloto Fase V-B dieron EXACTAMENTE 8 sumideros y dijo "me huele a
fallo del instrumento". Antes de escalar la bateria de pares, este script hace el TEST BARATO de
confirmacion pedido: toma UN grafo YA GENERADO del piloto (A2-B0-C2-r9, Clase I) y vuelve a generar su
condicion inicial de Phantom con 2 semillas de turbulencia DISTINTAS (seed=7, seed=99), dejando TODO lo
demas identico (mismo grafo, mismo seed_layout=12345, mismo N=2000, misma masa fija) -- la corrida con
seed=42 ya existe en bateria_fase5b_a2b0c2_piloto/A2-B0-C2-r9_I y NO se recomputa.

No modifica ningun script congelado. Reusa (solo import, sin tocar):
  - cs090_fase5b_phantom_adaptador.reconstruir_regla_a2b0c2 (reconstruye el grafo bit a bit)
  - cs090_fase5b_phantom_adaptador.generar_ic_masa_fija_desde_grafo (acepta turb_seed como parametro,
    NO hace falta editar el adaptador -- ya expone ese grado de libertad)
  - cs090_fase5b_correr.main (corre Phantom sobre una lista arbitraria de carpetas)
  - cs090_fase5b_analizar.analizar_carpeta (extrae metricas de cualquier carpeta)

Escribe en /Users/alexis/phantom_cs073/test_turbulencia_r9/turbseed_<N>/ (carpeta NUEVA, no toca la
carpeta original del piloto). No declara cierre ni veredicto.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis")

from cs090_fase5b_phantom_adaptador import reconstruir_regla_a2b0c2, generar_ic_masa_fija_desde_grafo
import cs090_fase5b_correr as CORRER
from cs090_fase5b_analizar import analizar_carpeta

BASE = Path("/Users/alexis/phantom_cs073/test_turbulencia_r9")
SEED_REGLA_R9 = 272702   # A2-B0-C2-r9, Clase I -- mismo que el piloto
SEED_LAYOUT = 12345      # mismo seed_layout que el piloto (unico grado de libertad = grafo, aqui fijo)
N = 2000

SEEDS_TURB_NUEVOS = [7, 99]   # seed=42 ya existe en bateria_fase5b_a2b0c2_piloto/A2-B0-C2-r9_I


def generar_ics():
    print("Reconstruyendo grafo A2-B0-C2-r9 (Clase I, seed={})...".format(SEED_REGLA_R9), flush=True)
    p, m = reconstruir_regla_a2b0c2(seed=SEED_REGLA_R9, N=N, n_sweeps=14)
    print(f"  grafo reconstruido: K={p['K']} J={p['J']} noise={p['noise']} meandeg={p['meandeg']} "
          f"kcap={p['kcap']} n_aristas={m['n_aristas']} diam={m['diam']}", flush=True)

    carpetas = []
    for ts in SEEDS_TURB_NUEVOS:
        carpeta = BASE / f"turbseed_{ts}"
        carpeta.mkdir(parents=True, exist_ok=True)
        ruta_ic = carpeta / "cosmogenesis_ic.txt"
        info = generar_ic_masa_fija_desde_grafo(
            m["adj_final"], N=N, seed_layout=SEED_LAYOUT, ruta_salida=str(ruta_ic),
            con_turbulencia=True, turb_seed=ts,
        )
        print(f"  IC turb_seed={ts} escrito en {ruta_ic} (masa_total={info['masa_total']:.6g})", flush=True)
        carpetas.append(carpeta)

    return carpetas


def main():
    carpetas = generar_ics()
    print("\nCorriendo Phantom sobre las carpetas nuevas...", flush=True)
    CORRER.main(carpetas)

    print("\nAnalizando (incluye tambien la corrida original turb_seed=42 del piloto, para comparar)...",
          flush=True)
    carpeta_original = Path(
        "/Users/alexis/phantom_cs073/bateria_fase5b_a2b0c2_piloto/A2-B0-C2-r9_I")
    todas = [carpeta_original] + carpetas
    filas = []
    for c in todas:
        fila = analizar_carpeta(c)
        fila["turb_seed_test"] = 42 if c == carpeta_original else int(c.name.split("_")[-1])
        filas.append(fila)
        print(f"  [{c.name}] n_sumideros={fila['n_sumideros']} "
              f"fraccion_masa_en_sumideros={fila.get('fraccion_masa_en_sumideros')} "
              f"kappa_v_agregado={fila.get('kappa_v_agregado')} "
              f"t_primer_sumidero={fila.get('t_primer_sumidero')}", flush=True)

    import csv
    ruta_out = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs090_fase5b_test_turbulencia_resultados.csv"
    campos = list(filas[0].keys())
    for f in filas:
        for c in f:
            if c not in campos:
                campos.append(c)
    with open(ruta_out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)
    print(f"\n[TOTAL] {len(filas)} filas -> {ruta_out}")
    return filas


if __name__ == "__main__":
    main()
