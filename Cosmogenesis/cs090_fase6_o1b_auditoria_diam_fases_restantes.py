"""
cs090_fase6_o1b_auditoria_diam_fases_restantes.py — ¿el bug de `_diam` toca a FASE IV y a CS07x-CS08x?
=======================================================================================================

QUÉ PREGUNTA RESPONDE (tarea O1-B, Parte 1)
-------------------------------------------
`FASE6_adopcion_diam_corregido_CS.md` §Cola-de-pendientes, punto 5, dejó explícito:

    "esta tarea cubrió Fase III [Exp.1] y toda la Fase V; **Fase IV y la línea CS07x-CS08x no se
     auditaron**, y usan el mismo `_diam`."

Este script audita ese resto. Hace TRES cosas, de la más barata a la más cara:

  PARTE A — AUDITORÍA ESTÁTICA (¿siquiera se usa el diámetro?)
      Recorre el grafo de imports LOCALES de cada script objetivo (Fase IV: cs082/cs083/cs083b/cs087;
      línea CS07x-CS08x: cs076..cs089) y busca, por AST, toda llamada a una función de diámetro
      (`_diam`, `diam_gigante`, `metricas_escala`, ...) tanto en el script como en cualquier módulo del
      proyecto que ese script importe. Distingue tres estados:
        - USA          : el script llama al diámetro (directa o indirectamente) y el número entra en su salida
        - IMPORTA-NO-USA : puede alcanzarlo por la cadena de imports pero nunca lo llama
        - NO-TOCA      : el diámetro no aparece por ningún lado
      Esto se hace por AST y no por `grep`, para que "aparece la palabra en un comentario" no cuente.

  PARTE B — DETECTOR BARATO sobre los CSV históricos que guardan diámetro POR ESCALA
      El detector propuesto en el informe anterior: un grafo "descarrila" si
      `diám(b=1) < diám(b=2)` y `diám(b=1) <= 3`. Se agrega el criterio más general y más sensible,
      el de MONOTONÍA: agrupar cajas conexas nunca puede ALARGAR un camino, así que cualquier
      `diám(b_k) < diám(b_{k+1})` es geométricamente imposible y delata que alguna escala midió otra
      cosa. (El informe anterior ya anotó que "pendiente < 0" se le escapa un caso: el detector correcto
      es el de monotonía, no el del signo de la pendiente.)
      Se aplica a `cs080_renormalizacion.csv` (control positivo: ya auditado, debe dar 0) y a
      `cs081_poda_dinamica.csv` (Fase III Exp.2, la poda — NO auditado hasta ahora).

  PARTE C — RE-MEDICIÓN DIRECTA de los grafos de FASE IV
      Aunque la Parte A muestre que Fase IV no mide diámetro, conviene saber si sus grafos SIQUIERA son
      del tipo que puede descarrilar (es decir: ¿se fragmentan?). Se reconstruyen, con las funciones
      propias de `cs082` sin tocarlas (`construir_base`, `_linea_adyacencia`), los grafos de los 4
      sustratos para las 20 semillas de `cs083`/`cs083b`, y se mide el diámetro de las dos maneras.
      Si nada se fragmenta, la respuesta es doblemente robusta: no sólo el diámetro no entra en la
      conclusión — es que ni siquiera habría cambiado si hubiera entrado.

SALIDAS
-------
  - `cs090_fase6_o1b_auditoria_estatica.csv`   (Parte A: una fila por script)
  - `cs090_fase6_o1b_detector_csv_historicos.csv` (Parte B: una fila por serie de escalas)
  - `cs090_fase6_o1b_fase4_grafos.csv`         (Parte C: una fila por grafo medido)

No toca ningún script congelado (todo por import o por lectura de CSV). No corre Phantom.
No declara cierre ni veredicto: reporta números.
"""
from __future__ import annotations
import ast
import csv
import os
import sys
from collections import defaultdict

import numpy as np

HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, HERE)

import cs090_diam_corregido as DC   # diam_original (cs055 tal cual) / diam_gigante / diagnostico

# ---------------------------------------------------------------------------------------------
# Nombres que, si son LLAMADOS, significan "acá se está midiendo un diámetro".
# `metricas_escala` (cs080) y `correr_regla_coarse` (fase5) son envoltorios que llaman a `_diam`
# adentro: llamarlos ES usar el diámetro aunque el nombre no lo diga.
# ---------------------------------------------------------------------------------------------
FUNCS_DIAMETRO = {"_diam", "diam_original", "diam_gigante", "diagnostico"}
ENVOLTORIOS    = {"metricas_escala"}

OBJETIVOS = [
    # (script, fase, qué mide su conclusión publicada)
    ("cs076_direccion_temporal.py",        "CS07x", "asimetría temporal de campo (skew, gradientes) — sin grafo"),
    ("cs077_gradientes_atractores.py",     "CS07x", "gradientes y atractores del motor holístico"),
    ("cs078_kappaV_permutacion.py",        "CS07x", "kappa_V por permutación"),
    ("cs079_delimitacion_cn4.py",          "CS07x", "delimitación C-N4 sobre volcados Phantom"),
    ("cs080_renormalizacion.py",           "Fase III Exp.1", "PENDIENTE diám vs N_cajas (YA AUDITADO: 0/54)"),
    ("cs081_poda_dinamica.py",             "Fase III Exp.2", "PENDIENTE diám vs N_cajas (auditado en esta tarea)"),
    ("cs082_fase4_4sustratos.py",          "Fase IV", "HOLONOMÍA de triángulos, REAL vs NULL vs SHUF"),
    ("cs083_fase4_robustecer.py",          "Fase IV", "HOLONOMÍA, 20 semillas + control fino (92%/8%)"),
    ("cs083b_fase4_control_local_global.py","Fase IV", "HOLONOMÍA, NULL-GLOBAL vs NULL-LOCAL-ROTO"),
    ("cs084_espectro_laplaciano.py",       "CS08x", "espectro del laplaciano (lambda_max, dispersión)"),
    ("cs085_espectro_jerarquia_cs073.py",  "CS08x", "espectro sobre la jerarquía CS073"),
    ("cs086_espectro_renorm_poda.py",      "CS08x", "espectro bajo renormalización y poda"),
    ("cs087_hodge_fase4.py",               "CS08x/Fase IV", "descomposición de Hodge sobre los sustratos de Fase IV"),
    ("cs088_espectro_proximidad_null12.py","CS08x", "espectro de proximidad NULL-1/NULL-2"),
    ("cs089_on77_espectral.py",            "CS08x", "espectro del sistema A/B de O-N7.7"),
]


# =============================================================================================
# PARTE A — auditoría estática por AST del grafo de imports locales
# =============================================================================================
def _modulos_locales():
    """Conjunto de nombres de módulo (sin .py) que viven en la carpeta del proyecto."""
    return {f[:-3] for f in os.listdir(HERE) if f.endswith(".py")}


def _arbol(nombre_modulo):
    p = os.path.join(HERE, nombre_modulo + ".py")
    if not os.path.exists(p):
        return None
    try:
        return ast.parse(open(p, encoding="utf-8").read(), filename=p)
    except SyntaxError:
        return None


def _imports_locales_de(arbol, locales):
    """Módulos locales del proyecto que este árbol importa (import X / from X import ...)."""
    out = set()
    for n in ast.walk(arbol):
        if isinstance(n, ast.Import):
            for a in n.names:
                raiz = a.name.split(".")[0]
                if raiz in locales:
                    out.add(raiz)
        elif isinstance(n, ast.ImportFrom) and n.module:
            raiz = n.module.split(".")[0]
            if raiz in locales:
                out.add(raiz)
    return out


def _llamadas_a_diametro(arbol):
    """Nombres de función de diámetro efectivamente LLAMADOS en este árbol (no sólo mencionados).
    Cubre tanto `_diam(...)` como `C7._diam(...)` (atributo de módulo)."""
    hits = set()
    for n in ast.walk(arbol):
        if isinstance(n, ast.Call):
            f = n.func
            nombre = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else None)
            if nombre in FUNCS_DIAMETRO:
                hits.add(nombre)
            elif nombre in ENVOLTORIOS:
                hits.add(nombre + "()->_diam")
    return hits


def parte_a():
    locales = _modulos_locales()
    filas = []
    print("=" * 108)
    print("PARTE A — auditoría estática por AST: ¿quién LLAMA al diámetro, directa o indirectamente?")
    print("=" * 108)
    print(f"  {'script':<40} {'fase':<16} {'estado':<16} {'llamadas propias / vía':<40}")
    for script, fase, conclusion in OBJETIVOS:
        mod = script[:-3]
        arbol = _arbol(mod)
        if arbol is None:
            filas.append(dict(script=script, fase=fase, estado="NO-PARSEA", llamadas_propias="",
                              modulos_con_diam="", n_modulos_alcanzables=0, conclusion_publicada=conclusion))
            continue
        propias = _llamadas_a_diametro(arbol)

        # cierre transitivo de imports locales
        vistos, cola = {mod}, list(_imports_locales_de(arbol, locales))
        con_diam = set()
        while cola:
            m = cola.pop()
            if m in vistos:
                continue
            vistos.add(m)
            a2 = _arbol(m)
            if a2 is None:
                continue
            if _llamadas_a_diametro(a2):
                con_diam.add(m)
            cola.extend(_imports_locales_de(a2, locales) - vistos)

        if propias:
            estado = "USA"
        elif con_diam:
            estado = "IMPORTA-NO-USA"
        else:
            estado = "NO-TOCA"
        filas.append(dict(script=script, fase=fase, estado=estado,
                          llamadas_propias="|".join(sorted(propias)),
                          modulos_con_diam="|".join(sorted(con_diam)),
                          n_modulos_alcanzables=len(vistos) - 1,
                          conclusion_publicada=conclusion))
        print(f"  {script:<40} {fase:<16} {estado:<16} "
              f"{('|'.join(sorted(propias)) or '(ninguna)'):<40}")

    with open(os.path.join(HERE, "cs090_fase6_o1b_auditoria_estatica.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["script", "fase", "estado", "llamadas_propias",
                                           "modulos_con_diam", "n_modulos_alcanzables",
                                           "conclusion_publicada"])
        wr.writeheader()
        for fila in filas:
            wr.writerow(fila)
    n_usa = sum(1 for f in filas if f["estado"] == "USA")
    print(f"\n  -> de {len(filas)} scripts objetivo, {n_usa} LLAMAN al diámetro; "
          f"{sum(1 for f in filas if f['estado']=='IMPORTA-NO-USA')} lo alcanzan por import sin usarlo; "
          f"{sum(1 for f in filas if f['estado']=='NO-TOCA')} no lo tocan.")
    return filas


# =============================================================================================
# PARTE B — detector barato sobre CSV históricos con diámetro por escala
# =============================================================================================
def detector_serie(bs, diams):
    """Devuelve (descarrila_b1, viola_monotonia, detalle).
      - descarrila_b1  : criterio propuesto en el informe anterior -> diám(b=1) < diám(b=2) y diám(b=1) <= 3
      - viola_monotonia: criterio general -> existe k con diám(b_k) < diám(b_{k+1}); agrupar cajas conexas
                         nunca puede alargar un camino, así que esto es geométricamente imposible."""
    orden = sorted(range(len(bs)), key=lambda i: bs[i])
    b = [bs[i] for i in orden]
    d = [diams[i] for i in orden]
    desc_b1 = bool(len(d) >= 2 and b[0] == 1 and d[0] < d[1] and d[0] <= 3)
    viola = [(b[k], d[k], b[k + 1], d[k + 1]) for k in range(len(d) - 1) if d[k] < d[k + 1]]
    return desc_b1, bool(viola), viola, b, d


def parte_b():
    fuentes = [
        ("cs080_renormalizacion.csv", ("seed", "arm"), "Fase III Exp.1 (renormalización) — CONTROL, ya auditado"),
        ("cs081_poda_dinamica.csv",   ("seed", "variante"), "Fase III Exp.2 (poda por costo) — NO auditado antes"),
    ]
    filas = []
    print("\n" + "=" * 108)
    print("PARTE B — detector barato sobre los CSV históricos que guardan diámetro por escala")
    print("=" * 108)
    for nombre, claves, etiqueta in fuentes:
        p = os.path.join(HERE, nombre)
        if not os.path.exists(p):
            print(f"  [!] {nombre} no existe — no auditable desde CSV")
            continue
        series = defaultdict(lambda: ([], []))
        with open(p) as f:
            for r in csv.DictReader(f):
                k = tuple(r[c] for c in claves)
                series[k][0].append(int(float(r["b"])))
                series[k][1].append(float(r["diam"]))
        n_desc = n_viola = 0
        for k, (bs, ds) in sorted(series.items()):
            desc, viola, detalle, b, d = detector_serie(bs, ds)
            n_desc += desc; n_viola += viola
            filas.append(dict(fuente=nombre, serie="|".join(k), n_escalas=len(b),
                              b_serie="|".join(str(x) for x in b),
                              diam_serie="|".join(f"{x:g}" for x in d),
                              descarrila_b1=desc, viola_monotonia=viola,
                              detalle_violacion=str(detalle) if detalle else ""))
        print(f"  {nombre:<32} {etiqueta}")
        print(f"    series (grafos) evaluadas: {len(series)}   "
              f"descarrila_b1: {n_desc}   viola_monotonia: {n_viola}")
    with open(os.path.join(HERE, "cs090_fase6_o1b_detector_csv_historicos.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["fuente", "serie", "n_escalas", "b_serie", "diam_serie",
                                           "descarrila_b1", "viola_monotonia", "detalle_violacion"])
        wr.writeheader()
        for fila in filas:
            wr.writerow(fila)
    return filas


# =============================================================================================
# PARTE C — re-medición directa de los grafos de FASE IV (¿se fragmentan siquiera?)
# =============================================================================================
def parte_c(n_seeds=20):
    """Reconstruye los grafos sobre los que corre la dinámica de Fase IV y los mide con las dos varas.

    Los 4 sustratos de cs082 comparten el MISMO grafo base (Erdos-Renyi N=110, p=0.09) y difieren en
    qué objeto-relación se define encima. Los grafos efectivamente recorridos por la dinámica son:
      - `base`         : el grafo de nodos (sustratos 1 y 3 lo usan como soporte)
      - `linea_aristas`: line-graph de las ~540 aristas (sustrato 1: grafo diádico)
      - `linea_trios`  : line-graph de los ~200 triángulos (sustrato 2: hipergrafo / sustrato 4: caras)
    Se mide el diámetro de los tres con `diam_original` (cs055) y `diam_gigante` (corregida)."""
    import cs082_fase4_4sustratos as CS82   # import puro, no se modifica

    filas = []
    print("\n" + "=" * 108)
    print(f"PARTE C — re-medición de los grafos de FASE IV (cs082/cs083/cs083b), semillas 1..{n_seeds}")
    print("=" * 108)
    for seed in range(1, n_seeds + 1):
        adj, edges, triangles = CS82.construir_base(seed)
        grafos = [("base_nodos", adj, CS82.N)]

        adj_e = CS82._linea_adyacencia(edges)
        grafos.append(("linea_aristas_sust1", [set(x) for x in adj_e], len(adj_e)))

        adj_t = CS82._linea_adyacencia(triangles)
        grafos.append(("linea_trios_sust2y4", [set(x) for x in adj_t], len(adj_t)))

        for etiqueta, a, n in grafos:
            d = DC.diagnostico(a, n)
            filas.append(dict(seed=seed, grafo=etiqueta, N=n,
                              n_aristas=sum(len(x) for x in a) // 2,
                              diam_viejo=d["diam_orig"], diam_corregido=d["diam_corr"],
                              tam_comp_medida=d["tam_comp_medida"], tam_gigante=d["tam_gigante"],
                              n_componentes=d["n_componentes"], n_aislados=d["n_aislados"],
                              descarrila=d["descarrila"],
                              difiere=bool(d["diam_orig"] != d["diam_corr"])))
    with open(os.path.join(HERE, "cs090_fase6_o1b_fase4_grafos.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["seed", "grafo", "N", "n_aristas", "diam_viejo", "diam_corregido",
                                           "tam_comp_medida", "tam_gigante", "n_componentes", "n_aislados",
                                           "descarrila", "difiere"])
        wr.writeheader()
        for fila in filas:
            wr.writerow(fila)

    por_grafo = defaultdict(list)
    for fila in filas:
        por_grafo[fila["grafo"]].append(fila)
    print(f"  {'grafo':<22} {'n':>4} {'N medio':>8} {'comps: 1 / >1':>14} {'aislados medio':>15} "
          f"{'descarrila':>11} {'difiere':>8}")
    for g, sub in por_grafo.items():
        una = sum(1 for f in sub if f["n_componentes"] == 1)
        print(f"  {g:<22} {len(sub):>4} {np.mean([f['N'] for f in sub]):>8.1f} "
              f"{f'{una} / {len(sub)-una}':>14} {np.mean([f['n_aislados'] for f in sub]):>15.2f} "
              f"{sum(1 for f in sub if f['descarrila']):>11} {sum(1 for f in sub if f['difiere']):>8}")
    n_desc = sum(1 for f in filas if f["descarrila"])
    n_dif = sum(1 for f in filas if f["difiere"])
    print(f"\n  -> {len(filas)} grafos de Fase IV medidos: descarrilan {n_desc}; "
          f"viejo != corregido en {n_dif}.")
    return filas


if __name__ == "__main__":
    a = parte_a()
    b = parte_b()
    c = parte_c()
    print("\n" + "=" * 108)
    print("SALIDAS: cs090_fase6_o1b_auditoria_estatica.csv | cs090_fase6_o1b_detector_csv_historicos.csv "
          "| cs090_fase6_o1b_fase4_grafos.csv")
    print("=" * 108)
