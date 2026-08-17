#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infra_verificar_poda_no_cambia_analisis.py — control de que podar volcados NO cambia ningún número
====================================================================================================

QUIÉN SOY / QUÉ HAGO
---------------------
Soy el control de calidad de `infra_podar_volcados.py`. Corro el analizador REAL del proyecto
(`cs090_fase5b_analizar.analizar_carpeta`, importado tal cual, sin modificarlo) sobre un puñado de
corridas y comparo, número por número, contra los CSV que ya estaban guardados en el proyecto ANTES
de podar. Si algún valor cambia, lo grito.

Por qué esto y no otra cosa: los CSV guardados (`cs090_fase7_f704_phantom_crudo.csv`,
`cs090_fase7_f702_phantom_crudo.csv`) son la evidencia congelada de lo que el análisis daba con los
501 volcados en disco. Compararse contra ellos es la prueba más fuerte disponible: no depende de que
yo haya medido bien el "antes", porque el "antes" ya estaba escrito por otro script en otra sesión.

CÓMO SE USA
------------
    ./venv/bin/python infra_verificar_poda_no_cambia_analisis.py            # todas las muestras
    ./venv/bin/python infra_verificar_poda_no_cambia_analisis.py --etiqueta antes-de-podar

Devuelve código de salida 1 si algún número difiere.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, HERE)
from cs090_fase5b_analizar import analizar_carpeta   # noqa: E402 — script congelado, sólo import

# (CSV guardado, columna con el nombre de la corrida, raíz de la batería, cuántas corridas muestrear)
FUENTES = [
    (f"{HERE}/cs090_fase7_f704_phantom_crudo.csv", "carpeta",
     "/Users/alexis/phantom_cs073/bateria_fase7_f704_cortar_bien", 3),
    (f"{HERE}/cs090_fase7_f702_phantom_crudo.csv", "carpeta",
     "/Users/alexis/phantom_cs073/bateria_fase7_f702_escalera", 2),
]

# métricas que salen del dump binario y del .sink — las que la poda podría romper
METRICAS = ["n_gas_inicial", "n_dump_final", "masa_gas_final", "masa_sumideros_final",
            "masa_total_final", "fraccion_masa_en_sumideros", "n_sumideros",
            "t_primer_sumidero", "masa_acretada_total", "kappa_v_agregado",
            "kappa_v_medio_valido", "n_kappa_indefinidos"]


def igual(a, b) -> bool:
    """Compara el valor recomputado contra el del CSV (que es texto). Exige igualdad EXACTA de la
    representación numérica: no se tolera deriva, la poda no debe cambiar ni el último dígito."""
    if a is None or a == "":
        return b in (None, "", "None")
    try:
        return f"{float(a):.17g}" == f"{float(b):.17g}"
    except (TypeError, ValueError):
        return str(a) == str(b)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--etiqueta", default="verificacion")
    args = ap.parse_args()

    total = difs = 0
    print(f"=== {args.etiqueta}: recomputando el análisis y comparando contra los CSV guardados\n")
    for ruta_csv, col, base, n in FUENTES:
        filas = list(csv.DictReader(open(ruta_csv)))
        muestra = filas[:n]
        print(f"--- {Path(ruta_csv).name} ({len(muestra)} corridas de {len(filas)})")
        for guardada in muestra:
            carpeta = Path(base) / guardada[col]
            if not carpeta.exists():
                print(f"  [SALTEADA] {carpeta} no existe")
                continue
            recomputada = analizar_carpeta(carpeta)
            malas = []
            for m in METRICAS:
                if m not in guardada:
                    continue
                total += 1
                if not igual(recomputada.get(m), guardada[m]):
                    malas.append(f"{m}: guardado={guardada[m]!r} recomputado={recomputada.get(m)!r}")
                    difs += 1
            estado = "IDÉNTICO" if not malas else "¡¡DIFERENCIAS!!"
            print(f"  {carpeta.name}: {estado}  "
                  f"(dump_final={recomputada.get('n_dump_final')}, "
                  f"frac_masa={recomputada.get('fraccion_masa_en_sumideros')}, "
                  f"kappaV={recomputada.get('kappa_v_agregado')})")
            for m in malas:
                print(f"      {m}")
    print(f"\n{total} valores comparados, {difs} diferencias.")
    if difs:
        print("FALLO: el análisis NO da los mismos números.")
        return 1
    print("OK: el análisis da EXACTAMENTE los mismos números.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
