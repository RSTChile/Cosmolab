#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F8-00 · MÓDULO DE PERSISTENCIA DE GRAFOS — `guardar_grafo` / `cargar_grafo`
==========================================================================

QUÉ ES ESTO (a nivel módulo)
-----------------------------
El arreglo hacia adelante de la Fase VIII: una forma **compacta y verificable** de dejar el grafo de
cada corrida escrito en disco, para que ningún experimento vuelva a perderlo.

El problema que resuelve, con analogía: hasta ahora cada corrida era como una foto de una construcción
de la que se anotaba "tenía 2308 ladrillos" pero se tiraban los planos. Si después uno quería saber
*cómo* estaban puestos esos ladrillos, había que volver a construir todo el edificio desde cero (y
rezar para que la reconstrucción fuera idéntica). Este módulo guarda los planos: 10 KB por grafo,
con un sello (hash) que permite comprobar después que los planos son de ESE edificio y no de otro.

FORMATO (`.grafo.gz`) — texto plano comprimido con gzip, legible con `zcat`
---------------------------------------------------------------------------
    # cosmogenesis_grafo v1 N=<n> E=<m> sha256=<64 hex> fmt=lista_aristas [clave=valor ...]
    i j        <- una arista por línea, i<j, ordenadas lexicográficamente por (i, j)
    ...

  - Compacto: ~7 caracteres por arista; con gzip, un grafo de N=2000 y 2300 aristas pesa ~7 KB.
  - Verificable: el `sha256` se calcula sobre la lista canónica de aristas (ver `hash_grafo`), así que
    dos grafos con el MISMO conjunto de aristas dan SIEMPRE el mismo sello, sin importar en qué orden
    se armaron en memoria. `cargar_grafo` lo recalcula y falla si no coincide.
  - Autodescriptivo: la cabecera admite metadatos libres (`rule_id`, `seed`, `brazo`, ...) que
    `cargar_grafo` devuelve como diccionario.

ADVERTENCIA IMPORTANTE — LO QUE EL ARCHIVO **NO** GUARDA
---------------------------------------------------------
Guarda la TOPOLOGÍA (qué nodo toca a qué nodo), no el ORDEN DE ITERACIÓN de los `set` de Python en
que el motor dejó cada vecindario. Eso no es un detalle cosmético: `cs080_renormalizacion.cajas_bfs`
recorre `for v in adj[u]`, así que un conjunto de vecinos con los mismos elementos pero armado en otro
orden puede dar una partición en cajas distinta y mover la **pendiente** en la 2ª-3ª cifra decimal
(está documentado en `cs090_fase6_o3b_rewiring._a_lista`: en A2-B0-C2-r14 la pendiente pasaba de
0.7729 a 0.7517 sólo por eso).

Consecuencia práctica, y hay que respetarla:
  * Todo lo que dependa SÓLO del conjunto de aristas (triángulos, clustering, Gini, solapamiento,
    grados, componentes, asortatividad, diámetro) se puede medir sobre el grafo cargado: da idéntico.
  * La **pendiente** de coarse-graining se mide sobre el objeto NATIVO que devuelve el motor si lo que
    se quiere es reproducir el histórico bit a bit. Sobre el grafo cargado se obtiene la variante
    "canónica" (vecinos insertados de menor a mayor), que es igual de legítima pero es OTRA medición.
    Por eso `cs090_fase8_f800_medir_254.py` reporta las dos y publica su diferencia.

EJEMPLO DE USO PARA UN RUNNER FUTURO (copiar y pegar)
------------------------------------------------------
    import cs090_fase8_f800_grafos as G8

    # ... el runner ya tiene `m = MOT.medir(...)`, `carpeta`, `rule_id`, `seed`, `N` ...
    ruta = G8.guardar_grafo(m["adj_final"], carpeta / "grafo_final.grafo.gz", N=N,
                            meta=dict(rule_id=rule_id, seed=seed, brazo="unico", n_sweeps=14))

    # y en el meta_regla.json que el runner ya escribía, una línea más:
    meta["grafo_sha256"] = G8.hash_grafo(m["adj_final"], N)
    meta["grafo_archivo"] = ruta.name

    # (o, en un solo paso, si el meta_regla.json ya está escrito:)
    G8.anotar_hash_en_meta(carpeta / "meta_regla.json", m["adj_final"], N,
                           archivo_grafo="grafo_final.grafo.gz")

    # ... y meses después, sin volver a correr el motor:
    adj, N, meta = G8.cargar_grafo(carpeta / "grafo_final.grafo.gz")   # verifica el sello solo

QUÉ NO HACE
-----------
No corre Phantom. No genera reglas. No modifica ningún script ni CSV existente.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path

CABECERA_MAGICA = "# cosmogenesis_grafo v1"
SUFIJO = ".grafo.gz"


# =============================================================================================
# 1) NORMALIZACIÓN — de cualquier formato de adyacencia de la línea a una lista de aristas canónica
# =============================================================================================
def _n_de(adj, N=None):
    """Deduce N. `adj` puede ser lista/tupla de sets (formato nativo del motor) o dict {i: vecinos}."""
    if N is not None:
        return int(N)
    if isinstance(adj, dict):
        return (max(adj) + 1) if adj else 0
    return len(adj)


def _vecinos(adj, i):
    if isinstance(adj, dict):
        return adj.get(i, ())
    return adj[i]


def aristas_canonicas(adj, N=None):
    """Lista de aristas (i, j) con i<j, ordenada lexicográficamente. Es LA representación canónica:
    no depende del orden en que los vecindarios quedaron armados en memoria."""
    n = _n_de(adj, N)
    E = set()
    for i in range(n):
        for j in _vecinos(adj, i):
            j = int(j)
            if j == i:
                continue                      # los bucles no existen en esta línea; si aparecen, se ignoran
            E.add((i, j) if i < j else (j, i))
    return sorted(E)


def hash_grafo(adj, N=None):
    """sha256 (64 hex) del grafo, calculado sobre `N` + la lista canónica de aristas.

    Dos grafos con el mismo conjunto de aristas dan el mismo sello SIEMPRE, en cualquier máquina y
    en cualquier orden de construcción. Sirve para (a) verificar que un archivo no se corrompió,
    (b) comprobar meses después que el grafo que uno tiene en la mano es el de esa corrida y no el de
    otra (el bug de colisión de nombres de Fase V-B, pero resuelto por contenido en vez de por nombre)."""
    n = _n_de(adj, N)
    h = hashlib.sha256()
    h.update(f"N={n}\n".encode())
    for (i, j) in aristas_canonicas(adj, n):
        h.update(b"%d %d\n" % (i, j))
    return h.hexdigest()


# =============================================================================================
# 2) GUARDAR / CARGAR
# =============================================================================================
def guardar_grafo(adj, ruta, N=None, meta=None):
    """Escribe el grafo en `ruta` (formato descrito arriba) y devuelve la ruta como `Path`.

    `meta` es un diccionario opcional de metadatos simples (sin espacios en los valores; se
    serializan como `clave=valor` en la cabecera). Se crea el directorio padre si falta."""
    n = _n_de(adj, N)
    E = aristas_canonicas(adj, n)
    sello = hash_grafo(adj, n)
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    extras = ""
    for k, v in (meta or {}).items():
        v = str(v).replace(" ", "_").replace("\n", "_")
        extras += f" {k}={v}"

    with gzip.open(ruta, "wt", compresslevel=6) as f:
        f.write(f"{CABECERA_MAGICA} N={n} E={len(E)} sha256={sello} fmt=lista_aristas{extras}\n")
        for (i, j) in E:
            f.write(f"{i} {j}\n")
    return ruta


def cargar_grafo(ruta, verificar=True):
    """Lee un `.grafo.gz` y devuelve `(adj, N, meta)`.

    `adj` es una lista de `set` indexable por entero (mismo formato que usa toda la línea), con los
    vecinos insertados de MENOR A MAYOR: es la forma CANÓNICA (ver advertencia del docstring del
    módulo sobre la pendiente). `meta` incluye siempre `N`, `E`, `sha256` y `sha256_recalculado`.
    Si `verificar=True` y el sello no coincide, levanta `ValueError`."""
    ruta = Path(ruta)
    with gzip.open(ruta, "rt") as f:
        cabecera = f.readline().rstrip("\n")
        if not cabecera.startswith(CABECERA_MAGICA):
            raise ValueError(f"{ruta.name}: no es un archivo de grafo v1 (cabecera: {cabecera[:60]!r})")
        meta = {}
        for tok in cabecera[len(CABECERA_MAGICA):].split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                meta[k] = v
        n = int(meta["N"])
        adj = [set() for _ in range(n)]
        m = 0
        for linea in f:
            if not linea.strip():
                continue
            a, b = linea.split()
            a, b = int(a), int(b)
            adj[a].add(b)
            adj[b].add(a)
            m += 1

    # los sets se rearman en orden creciente para que la forma cargada sea siempre la misma
    adj = [set(sorted(s)) for s in adj]

    meta["N"] = n
    meta["E"] = m
    if verificar:
        recal = hash_grafo(adj, n)
        meta["sha256_recalculado"] = recal
        if meta.get("sha256") and recal != meta["sha256"]:
            raise ValueError(f"{ruta.name}: SELLO NO COINCIDE — archivo corrupto o alterado "
                             f"(guardado {meta['sha256'][:12]}…, recalculado {recal[:12]}…)")
    return adj, n, meta


# =============================================================================================
# 3) ANOTAR EL SELLO EN EL meta_regla.json QUE EL RUNNER YA ESCRIBE
# =============================================================================================
def anotar_hash_en_meta(ruta_meta, adj, N=None, archivo_grafo=None):
    """Agrega (o actualiza) las claves `grafo_sha256`, `grafo_n_aristas` y opcionalmente
    `grafo_archivo` dentro de un `meta_regla.json` ya existente, sin tocar el resto de su contenido.

    Pensado para que un runner futuro lo llame justo después de escribir su meta. NO se usa sobre las
    corridas históricas de Fases V-B/VI/VII: los `meta_regla.json` ya escritos no se modifican."""
    ruta_meta = Path(ruta_meta)
    datos = json.loads(ruta_meta.read_text()) if ruta_meta.exists() else {}
    n = _n_de(adj, N)
    datos["grafo_sha256"] = hash_grafo(adj, n)
    datos["grafo_n_aristas"] = len(aristas_canonicas(adj, n))
    if archivo_grafo is not None:
        datos["grafo_archivo"] = str(archivo_grafo)
    ruta_meta.write_text(json.dumps(datos, indent=2))
    return datos["grafo_sha256"]


# =============================================================================================
# 4) AUTOPRUEBA — se corre con `python3.9 cs090_fase8_f800_grafos.py`
# =============================================================================================
if __name__ == "__main__":
    import random
    import tempfile

    rnd = random.Random(12345)
    N = 300
    adj_nativo = [set() for _ in range(N)]
    for _ in range(900):
        a, b = rnd.randrange(N), rnd.randrange(N)
        if a != b:
            adj_nativo[a].add(b)
            adj_nativo[b].add(a)

    with tempfile.TemporaryDirectory() as tmp:
        ruta = guardar_grafo(adj_nativo, os.path.join(tmp, "prueba" + SUFIJO), N=N,
                             meta=dict(rule_id="AUTOPRUEBA", seed=12345))
        adj2, n2, meta = cargar_grafo(ruta)
        tam = os.path.getsize(ruta)

        iguales = (n2 == N) and all(adj_nativo[i] == adj2[i] for i in range(N))
        print(f"autoprueba: N={N} aristas={meta['E']} archivo={tam} bytes "
              f"({tam / max(1, meta['E']):.1f} bytes/arista)")
        print(f"  ida y vuelta idéntica arista por arista: {iguales}")
        print(f"  sello guardado == recalculado: {meta['sha256'] == meta['sha256_recalculado']}")
        print(f"  sello: {meta['sha256'][:24]}…")

        # el sello NO depende del orden de armado en memoria: se rearma al revés y debe dar igual
        adj_rev = []
        for i in range(N):
            s = set()
            for v in sorted(adj_nativo[i], reverse=True):
                s.add(v)
            adj_rev.append(s)
        print(f"  sello invariante al orden de armado: {hash_grafo(adj_rev, N) == meta['sha256']}")

        # detección de corrupción
        adj2[0].add((max(adj2[0]) + 1) % N if adj2[0] else 1)
        try:
            ruta_mal = guardar_grafo(adj2, os.path.join(tmp, "mal" + SUFIJO), N=N)
            with gzip.open(ruta_mal, "rt") as f:
                lineas = f.readlines()
            lineas[1] = "0 999\n" if N > 999 else "0 %d\n" % (N - 1)
            with gzip.open(ruta_mal, "wt") as f:
                f.writelines(lineas)
            cargar_grafo(ruta_mal)
            print("  deteccion de corrupcion: NO detectada (mal)")
        except ValueError as e:
            print(f"  deteccion de corrupcion: OK -> {str(e)[:70]}…")
