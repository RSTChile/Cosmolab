#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infra_podar_volcados.py — Poda de volcados intermedios de Phantom (libera disco sin tocar el análisis)
========================================================================================================

QUIÉN SOY / QUÉ HAGO
---------------------
Soy la herramienta de infraestructura que borra los volcados binarios INTERMEDIOS de las corridas de
Phantom de Cosmogénesis, conservando exactamente los archivos que los analizadores del proyecto sí leen.
No soy un experimento: no calculo física, no produzco métricas, no declaro nada. Sólo libero disco.

POR QUÉ EXISTO (el problema medido)
------------------------------------
El protocolo validado de esta línea corre Phantom con `tmax=0.500` y `dtmax=0.001`, y Phantom escribe
UN volcado por cada `dtmax` -> 501 volcados por corrida (`cosmog_00000` .. `cosmog_00500`). A N=2000 eso
son ~24 MB de volcados intermedios por corrida que NINGÚN análisis abre. Con decenas de corridas por
batería, y con la intención de subir la resolución a N=8000 (donde cada volcado pesa ~4x), el disco se
llena antes de poder correr el experimento.

QUÉ SE CONSERVA Y POR QUÉ (auditoría de lo que los analizadores realmente leen)
--------------------------------------------------------------------------------
Verificado leyendo el código (no por suposición). El contrato de lectura del proyecto es:

  * `cs090_fase5b_analizar.py::analizar_carpeta` — el analizador central, importado TAL CUAL por
    `cs090_fase6_o3b_analizar.py`, `cs090_fase6_o3e_correr.py`, `cs090_fase7_f702_analizar.py`,
    `cs090_fase7_f704_analizar.py` y `cs090_fase6_o3a_convergencia_resolucion.py`. Usa
    `listar_dumps(carpeta)` y de esa lista toca SÓLO `dumps[0]` (n_gas_inicial) y `dumps[-1]`
    (masa de gas / masa de sumideros / fracción de masa). Nunca recorre los del medio.
  * `cosmog01.sink` — de ahí salen n_sumideros, t_primer_sumidero, masa_acretada_total y κ_V
    (`cs078_kappaV_permutacion.py`, `null{1,2,3}_bateria_comparar.py`, `real_extra_comparar.py`,
    `grafo_random_bateria_comparar.py`, `ON77_sistemaA/B*.py`, `cs090_fase6_outliers_paso3_phantom.py`).
  * `cosmog_00000` explícito — `cs088_espectro_proximidad_null12.py`, `null2_zeldovich_disenar_verificar.py`,
    `grafo_random_masa_fija_verificar.py`, `cs090_fase6_o3f_extraer_gas.py`.
  * `cosmog_00500` explícito — `cn4_delimitacion_fof.py`, `cs079_delimitacion_cn4.py`,
    `cs090_fase6_o4a_observable_comun.py`, `cs090_fase6_o3f_extraer_gas.py`, y los `*_correr.py`
    lo usan como marca de "esta corrida ya está hecha, no recomputar".
  * `cosmogenesis_ic.txt` — condición inicial en texto; la lee `cs090_fase7_f705_geometria_ic_todas.py`,
    `cs090_fase6_o3a_geometria_ic.py`, `cs090_fase6_o4a_observable_comun.py`.
  * `meta_regla.json` — metadatos de la regla (rule_id, clase, seed, K, J, kcap...); lo leen todos los
    analizadores de Fase V-B en adelante.
  * `cosmog.in`, `*.ev`, `*.log`, `*.setup` — configuración y bitácora; pesan poco y son la evidencia
    de qué se corrió. No se tocan.

  BÚSQUEDA NEGATIVA (lo que autoriza a borrar): un `grep` sobre TODOS los `.py` del proyecto por
  `for ... in listar_dumps`, `dumps[1:-1]`, `dumps[1:]`, `len(dumps)` no encuentra NINGÚN analizador que
  recorra los volcados del medio. Los dos únicos usos de `dumps[1:-1]` son podas ya existentes
  (`cs090_fase6_o3d_barrido_kcap.py:356` y `cs090_fase7_f701_factorial.py:372`), o sea que podar ya es
  práctica establecida del proyecto. El único lector de un volcado intermedio es el `_smoke_test()` de
  `leer_volcado_phantom.py`, que toma `dumps[len(dumps)//2]`; tras podar, esa lista tiene 2 elementos y
  `dumps[1]` es el volcado FINAL -> el smoke test sigue pasando (se verificó corriéndolo).

REGLA DE SEGURIDAD: UNA CORRIDA INCOMPLETA NO SE PODA
------------------------------------------------------
Sólo se poda una carpeta si:
  (a) tiene al menos 3 volcados (si no, no hay nada que podar),
  (b) el último volcado corresponde al final esperado del protocolo, calculado como
      round(tmax/dtmax) leído del propio `.in` de la corrida (para el protocolo estándar: 0.500/0.001=500),
  (c) ese volcado final se abre sin error (verificación de integridad real, no sólo `os.path.exists`);
      con `--verificar-sarracen` se abre con `sarracen` (el mismo lector del análisis), y sin esa bandera
      se hace la verificación barata de cabecera binaria de Phantom.
Si (b) o (c) fallan, la carpeta se marca INCOMPLETA, se deja INTACTA y se documenta en el CSV. Motivo:
en una corrida incompleta `dumps[-1]` es un volcado intermedio, y borrarlo destruiría el único estado
final que esa corrida tiene.

CÓMO SE USA
------------
    # 1. DRY RUN (por defecto: NO borra nada, sólo informa y escribe el CSV)
    ./venv/bin/python infra_podar_volcados.py --csv infra_poda_detalle.csv

    # 2. Podar de verdad (requiere la bandera explícita)
    ./venv/bin/python infra_podar_volcados.py --ejecutar --csv infra_poda_detalle.csv

    # 3. Podar UNA batería recién corrida (uso previsto hacia adelante, al final de cada batería)
    ./venv/bin/python infra_podar_volcados.py --ejecutar --raiz /Users/alexis/phantom_cs073/bateria_X

    # 4. Con verificación de integridad fuerte (abre el volcado final con sarracen)
    ./venv/bin/python infra_podar_volcados.py --ejecutar --verificar-sarracen --raiz ...

Banderas: `--raiz` (default /Users/alexis/phantom_cs073), `--ejecutar`, `--csv`, `--verificar-sarracen`,
`--excluir` (patrones de carpeta de primer nivel a saltear, repetible).

QUÉ NO HAGO
------------
No borro carpetas enteras, no borro `.sink`/`.ev`/`.log`/`.in`/`.txt`/`.json`, no toco corridas
incompletas, no toco ningún script congelado, no cambio ningún parámetro de física.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

RAIZ_DEFECTO = Path("/Users/alexis/phantom_cs073")

# un volcado de Phantom es <prefijo>_NNNNN sin extensión (p.ej. cosmog_00123, sphere_00648)
RE_VOLCADO = re.compile(r"^(?P<pref>.+)_(?P<num>\d{5})$")

# archivos que jamás se tocan aunque casen algo raro
SUFIJOS_INTOCABLES = {".sink", ".ev", ".log", ".in", ".txt", ".json", ".setup", ".csv", ".md",
                      ".restart", ".png", ".pdf"}

# identificador binario embebido al principio de un volcado de Phantom (verificación barata)
MARCA_PHANTOM = b"FT:Phantom"


def es_volcado(p: Path) -> tuple[str, int] | None:
    """Devuelve (prefijo, numero) si `p` es un volcado binario de Phantom; None si no lo es."""
    if not p.is_file() or p.suffix in SUFIJOS_INTOCABLES:
        return None
    m = RE_VOLCADO.match(p.name)
    if not m:
        return None
    return m.group("pref"), int(m.group("num"))


def leer_protocolo(carpeta: Path) -> tuple[float | None, float | None, int | None]:
    """Lee tmax y dtmax del `.in` de la corrida y devuelve (tmax, dtmax, indice_final_esperado).
    No modifica el archivo. Si no hay `.in` legible devuelve (None, None, None) y la corrida se
    tratará como 'sin protocolo declarado' -> NO se poda."""
    candidatos = sorted(carpeta.glob("*.in"))
    if not candidatos:
        return None, None, None
    tmax = dtmax = None
    try:
        for linea in candidatos[0].read_text(errors="replace").splitlines():
            if "=" not in linea:
                continue
            clave = linea.split("=", 1)[0].strip()
            valor = linea.split("=", 1)[1].split("!")[0].strip()
            if clave == "tmax":
                tmax = float(valor)
            elif clave == "dtmax":
                dtmax = float(valor)
    except Exception:
        return None, None, None
    if not tmax or not dtmax or dtmax <= 0:
        return tmax, dtmax, None
    return tmax, dtmax, int(round(tmax / dtmax))


def volcado_legible(p: Path, con_sarracen: bool) -> tuple[bool, str]:
    """Verificación de integridad del volcado FINAL antes de autorizar la poda de esa corrida."""
    try:
        if p.stat().st_size < 1024:
            return False, "volcado final sospechosamente chico (<1 KB)"
        if con_sarracen:
            import warnings
            warnings.filterwarnings("ignore")
            import sarracen
            r = sarracen.read_phantom(str(p))
            gas = r[0] if isinstance(r, list) else r
            if len(gas) == 0:
                return False, "sarracen abrió el volcado pero tiene 0 partículas"
            return True, f"sarracen OK ({len(gas)} partículas)"
        with open(p, "rb") as fh:
            cabecera = fh.read(256)
        if MARCA_PHANTOM not in cabecera:
            return False, "sin la marca binaria 'FT:Phantom' en la cabecera"
        return True, "cabecera Phantom OK"
    except Exception as e:  # noqa: BLE001 — cualquier fallo de lectura = no podar
        return False, f"no se pudo leer: {type(e).__name__}: {e}"


def auditar_corrida(carpeta: Path, con_sarracen: bool) -> dict | None:
    """Audita UNA carpeta de corrida. Devuelve la fila del informe, o None si no es una corrida."""
    volcados: dict[str, list[tuple[int, Path]]] = {}
    for p in carpeta.iterdir():
        v = es_volcado(p)
        if v:
            volcados.setdefault(v[0], []).append((v[1], p))
    if not volcados:
        return None

    # si hubiera más de un prefijo (raro), se toma el que tenga más volcados
    pref = max(volcados, key=lambda k: len(volcados[k]))
    serie = sorted(volcados[pref])
    n_volcados = len(serie)
    idx_min, idx_max = serie[0][0], serie[-1][0]

    tmax, dtmax, idx_esperado = leer_protocolo(carpeta)
    bytes_totales = sum(p.stat().st_size for _, p in serie)
    intermedios = serie[1:-1]
    bytes_intermedios = sum(p.stat().st_size for _, p in intermedios)

    fila = dict(
        carpeta=str(carpeta), bateria=carpeta.parent.name, corrida=carpeta.name, prefijo=pref,
        n_volcados=n_volcados, idx_primero=idx_min, idx_ultimo=idx_max,
        tmax=tmax, dtmax=dtmax, idx_final_esperado=idx_esperado,
        mb_volcados=round(bytes_totales / 1e6, 2),
        n_intermedios=len(intermedios), mb_liberables=round(bytes_intermedios / 1e6, 2),
        podable=False, motivo="", verificacion="",
    )

    if n_volcados < 3:
        fila["motivo"] = "nada que podar (menos de 3 volcados)"
        fila["mb_liberables"] = 0.0
        fila["n_intermedios"] = 0
        return fila
    if idx_esperado is None:
        fila["motivo"] = "INCOMPLETA/NO-ESTÁNDAR: no se pudo leer tmax/dtmax del .in -> NO se poda"
        return fila
    if idx_max != idx_esperado:
        fila["motivo"] = (f"INCOMPLETA: último volcado {idx_max} != final esperado {idx_esperado} "
                          f"(tmax/dtmax) -> NO se poda, queda intacta")
        return fila
    if idx_min != 0:
        fila["motivo"] = f"ANÓMALA: falta el volcado 0 (el primero es {idx_min}) -> NO se poda"
        return fila

    ok, detalle = volcado_legible(serie[-1][1], con_sarracen)
    fila["verificacion"] = detalle
    if not ok:
        fila["motivo"] = f"INTEGRIDAD: el volcado final no verifica ({detalle}) -> NO se poda"
        return fila

    fila["podable"] = True
    fila["motivo"] = (f"completa y verificada; se conservan {pref}_{idx_min:05d}, {pref}_{idx_max:05d} "
                      f"y todos los archivos no-volcado (.sink/.ev/.in/.log/.txt/.json)")
    return fila


def recorrer(raiz: Path, excluir: list[str], con_sarracen: bool) -> list[dict]:
    """Recorre la raíz buscando carpetas de corrida (cualquier profundidad) y las audita."""
    filas = []
    for dirpath, dirnames, _filenames in os.walk(raiz):
        d = Path(dirpath)
        if any(pat in d.parts for pat in excluir):
            dirnames[:] = []
            continue
        try:
            fila = auditar_corrida(d, con_sarracen)
        except PermissionError:
            continue
        if fila:
            filas.append(fila)
            dirnames[:] = []  # una carpeta de corrida no contiene otras corridas
    return sorted(filas, key=lambda f: (-f["mb_liberables"], f["carpeta"]))


def podar(fila: dict) -> tuple[int, int]:
    """Borra los volcados intermedios de UNA corrida ya auditada como podable.
    Devuelve (n_borrados, bytes_borrados). Vuelve a verificar la condición en el momento de borrar
    (no confía en la auditoría previa: el disco pudo cambiar entre el dry run y la ejecución)."""
    carpeta = Path(fila["carpeta"])
    pref = fila["prefijo"]
    serie = []
    for p in carpeta.iterdir():
        v = es_volcado(p)
        if v and v[0] == pref:
            serie.append((v[1], p))
    serie.sort()
    if len(serie) < 3:
        return 0, 0
    if serie[0][0] != fila["idx_primero"] or serie[-1][0] != fila["idx_ultimo"]:
        raise RuntimeError(f"{carpeta}: la serie de volcados cambió desde la auditoría — abortando poda")
    n = b = 0
    for _, p in serie[1:-1]:
        b += p.stat().st_size
        p.unlink()
        n += 1
    return n, b


def podar_una_corrida(carpeta: str | Path, con_sarracen: bool = False, verboso: bool = True) -> dict:
    """EL ARREGLO HACIA ADELANTE — llamable desde cualquier `*_correr.py` NUEVO, inmediatamente
    después de que Phantom termina una corrida, para que los 499 volcados inútiles no lleguen a
    acumularse nunca.

    Uso previsto en un runner nuevo (dos líneas, no toca la física ni el `.in`):

        from infra_podar_volcados import podar_una_corrida
        ...
        subprocess.run([phantom, "cosmog.in"], cwd=carpeta, check=True)
        podar_una_corrida(carpeta)      # <- acá, apenas termina esa corrida

    Es la misma poda que ya hacen `cs090_fase6_o3d_barrido_kcap.py:356` y
    `cs090_fase7_f701_factorial.py:372` inline, pero con la verificación de completitud e
    integridad que aquellas no tienen: si la corrida quedó incompleta (Phantom abortó, se cortó
    por wall time, etc.) NO borra nada y lo dice. Es idempotente: correrla dos veces sobre la
    misma carpeta no hace daño (la segunda vez ya no hay intermedios).

    Devuelve la fila de auditoría con `podable`, `motivo`, `n_borrados` y `mb_liberados`.
    """
    carpeta = Path(carpeta)
    fila = auditar_corrida(carpeta, con_sarracen)
    if fila is None:
        fila = dict(carpeta=str(carpeta), podable=False, motivo="no es una carpeta de corrida",
                    n_borrados=0, mb_liberados=0.0)
        if verboso:
            print(f"[poda] {carpeta.name}: {fila['motivo']}")
        return fila
    if not fila["podable"]:
        fila["n_borrados"], fila["mb_liberados"] = 0, 0.0
        if verboso:
            print(f"[poda] {carpeta.name}: NO se poda -- {fila['motivo']}")
        return fila
    n, b = podar(fila)
    fila["n_borrados"], fila["mb_liberados"] = n, round(b / 1e6, 2)
    if verboso:
        print(f"[poda] {carpeta.name}: {n} volcados intermedios borrados, "
              f"{fila['mb_liberados']} MB liberados (se conservan el inicial, el final y el .sink)")
    return fila


def main() -> int:
    ap = argparse.ArgumentParser(description="Poda volcados intermedios de Phantom (dry run por defecto)")
    ap.add_argument("--raiz", default=str(RAIZ_DEFECTO))
    ap.add_argument("--ejecutar", action="store_true", help="borra de verdad (sin esto es dry run)")
    ap.add_argument("--csv", default=None, help="ruta del CSV de detalle por corrida")
    ap.add_argument("--verificar-sarracen", action="store_true",
                    help="abre el volcado final con sarracen (más lento, verificación fuerte)")
    ap.add_argument("--excluir", action="append", default=[],
                    help="nombre de carpeta a saltear (repetible)")
    args = ap.parse_args()

    raiz = Path(args.raiz)
    filas = recorrer(raiz, args.excluir, args.verificar_sarracen)

    podables = [f for f in filas if f["podable"]]
    intactas = [f for f in filas if not f["podable"] and f["n_intermedios"] > 0]

    por_bateria: dict[str, dict] = {}
    for f in filas:
        b = por_bateria.setdefault(f["bateria"], dict(corridas=0, podables=0, intactas=0,
                                                      mb_volcados=0.0, mb_liberables=0.0))
        b["corridas"] += 1
        b["mb_volcados"] += f["mb_volcados"]
        if f["podable"]:
            b["podables"] += 1
            b["mb_liberables"] += f["mb_liberables"]
        elif f["n_intermedios"] > 0:
            b["intactas"] += 1

    modo = "EJECUTANDO PODA" if args.ejecutar else "DRY RUN (no se borra nada)"
    print(f"=== infra_podar_volcados — {modo} — raíz={raiz}\n")
    print(f"{'batería':<42} {'corr':>5} {'podab':>6} {'intac':>6} {'MB volc':>10} {'MB libera':>10}")
    print("-" * 84)
    for nombre, b in sorted(por_bateria.items(), key=lambda kv: -kv[1]["mb_liberables"]):
        print(f"{nombre[:42]:<42} {b['corridas']:>5} {b['podables']:>6} {b['intactas']:>6} "
              f"{b['mb_volcados']:>10.1f} {b['mb_liberables']:>10.1f}")
    print("-" * 84)
    total_lib = sum(f["mb_liberables"] for f in podables)
    print(f"{'TOTAL':<42} {len(filas):>5} {len(podables):>6} {len(intactas):>6} "
          f"{sum(f['mb_volcados'] for f in filas):>10.1f} {total_lib:>10.1f}")
    print(f"\nLiberable: {total_lib/1000:.2f} GB en {len(podables)} corridas completas y verificadas.")
    if intactas:
        print(f"\nCorridas que NO se podan ({len(intactas)}) — se dejan intactas:")
        for f in intactas[:40]:
            print(f"  {f['bateria']}/{f['corrida']}: {f['motivo']}")
        if len(intactas) > 40:
            print(f"  ... y {len(intactas)-40} más (ver CSV)")

    if args.csv:
        campos = list(filas[0].keys()) if filas else []
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=campos)
            w.writeheader()
            w.writerows(filas)
        print(f"\nCSV de detalle: {args.csv}")

    if args.ejecutar:
        n_tot = b_tot = 0
        for f in podables:
            n, b = podar(f)
            n_tot += n
            b_tot += b
        print(f"\nPODA HECHA: {n_tot} volcados intermedios borrados, "
              f"{b_tot/1e9:.2f} GB liberados en {len(podables)} corridas.")
    else:
        print("\n(dry run — volvé a correr con --ejecutar para borrar)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
