#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cs090_f804_grano_n8000.py — F8-04: ¿cuál es el GRANO del instrumento a N=8000?
==============================================================================

POR QUÉ EXISTE
--------------
`INFRA_layout_barnes_hut_CS.md` §8.1 dejó escrito el único número que falta antes de comprometer una
batería a N=8000: **cuánto tiembla el instrumento a esa resolución**. A N=2000 ese temblor está medido
(`FASE7_F704...`, y §4.1 del informe de infraestructura): perturbar SÓLO el redondeo (nivel 1e-16) mueve
el observable FoF en 0,0112, y una partícula vale 0,0005 de fracción de masa. A N=8000 no se sabe.

Si el ruido crece más rápido que la señal, una batería a N=8000 no es interpretable. Los dos efectos que
se persiguen son **+0,0143** (F7-03, apiñamiento) y **+0,0016** (F7-04, residual).

QUÉ MIDE, EXACTAMENTE
---------------------
Para cada uno de 2 grafos reales de la serie (el par extremo de O3-A, uno Clase I y otro Clase III), se
generan 8 RÉPLICAS que difieren ÚNICAMENTE en una perturbación de redondeo de orden 1e-16, y se corre la
cadena completa grafo → layout → IC → phantomsetup → Phantom sobre todas. El grano es la **desviación
estándar de la fracción de masa acretada entre réplicas del mismo grafo**, leída a un **dump común**
(mismo tiempo simulado en todas: comparar dumps de tiempos distintos mezclaría ruido con evolución).

DÓNDE SE INYECTA LA PERTURBACIÓN, Y POR QUÉ AHÍ
-----------------------------------------------
El control de redondeo de `cs090_layout_barnes_hut.py` (`metodo="exacta_reordenada"`: la misma suma N²
en orden inverso) da UNA sola variante alternativa, y a N=8000 cuesta la suma N² completa (~73 min de
layout). Para conseguir 8 réplicas hace falta una familia de perturbaciones del mismo orden y del mismo
carácter — que cambien el ÚLTIMO BIT y nada más.

Se usa `lado`, el lado de la caja del layout, movido **k ULPs** (`np.nextafter`, k = 0,1,...,7). El ULP
de `lado = 2000^(1/3) = 12,5992` es 1,78e-15 absoluto = **1,4e-16 relativo**: exactamente el orden del
control de redondeo. Y es la inyección más limpia disponible porque:

  * el layout Fruchterman-Reingold es **covariante de escala**: pos ~ lado, k_FR ~ lado, la fuerza
    repulsiva k_FR²·δ/d² ~ lado, la atractiva d²/k_FR ~ lado, el paso lado·0,1 ~ lado. Multiplicar `lado`
    por (1+ε) da, EN ARITMÉTICA EXACTA, el mismo layout multiplicado por (1+ε) — o sea, la MISMA
    configuración salvo un reescalado de 1e-16, que en el observable FoF (longitud de enlace relativa a
    la separación media) es rigurosamente nada. Todo lo que aparezca por encima de 1e-16 en las posiciones
    finales es **redondeo amplificado por las 100 iteraciones**, que es justo lo que se quiere medir;
  * no obliga a tocar ni una línea de código congelado ni del módulo Barnes-Hut: `lado` es un argumento.

(El regularizador aditivo `+1e-6` de la fuerza rompe la covariancia exacta, pero su contribución relativa
es del orden de ε·1e-6/d — más chica todavía que el propio ε. El subcomando `sanity` verifica
EMPÍRICAMENTE, a N=2000, que esta perturbación produce una dispersión de posiciones finales del mismo
tamaño que el control de redondeo documentado (RMS 0,321 / máx 3,11); si diera mucho menos, el
instrumento no serviría y habría que buscar otro punto de inyección.)

QUÉ SE MANTIENE IDÉNTICO
------------------------
Todo el resto del protocolo de la serie: masa total fija 18800, lado nominal 2000^(1/3), `Expansion` de
60 pasos, turbulencia Mach=3 semilla 42, 100 iteraciones de layout, `seed_layout=12345`, 14 sweeps del
motor relacional, `icreate_sinks=1`, `rho_crit_cgs=1000`, `r_crit=0.600`, `h_acc=0.300`, `f_acc=0.800`,
`tmax=0.500`, `dtmax=0.001`. El bloque de `cosmog.in` se reescribe con `editar_cosmog_in` IMPORTADA de
`cs090_fase5b_correr.py` (congelado). El lector es `leer_volcado_phantom.leer_dump` (congelado).

**θ = 0,5 en TODAS las réplicas.** No es el θ operativo (0,3) que eligió el informe de infraestructura;
se usa 0,5 porque cuesta ×2,9 menos y porque lo que aquí se mide es una DISPERSIÓN ENTRE RÉPLICAS QUE
COMPARTEN θ: el sesgo de θ es una constante común a las 8 réplicas y se cancela exacto en la σ. Queda
dicho como limitación: no se midió si σ misma depende de θ.

**Los grafos se guardan** (`grafo_adj.json.gz` + `grafo_meta.json` por grafo, compartidos por sus 8
réplicas, porque el motor relacional es idéntico en todas — lo único que cambia es el layout).

USO
    ./venv/bin/python cs090_f804_grano_n8000.py sanity            # verifica el instrumento a N=2000
    ./venv/bin/python cs090_f804_grano_n8000.py ic   <graf> <rep>          # grafo (cache) + layout + IC
    ./venv/bin/python cs090_f804_grano_n8000.py run  <graf> <rep> <cap_s>  # phantomsetup + Phantom
    ./venv/bin/python cs090_f804_grano_n8000.py medir [dump_forzado]       # lee al dump común -> CSV
"""
from __future__ import annotations

import gzip
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis")

from cs090_layout_barnes_hut import layout_barnes_hut
from cs090_fase5b_phantom_adaptador import (reconstruir_regla_a2b0c2, LADO_FIJO, MASA_TOTAL_OBJETIVO,
                                            TURB_SEED, CS_SONORA, N_REFERENCIA_LADO)
from cs073_cierre_holistico import T0, _T_reloj
from cs072_modulos.piezas.p_expansion import Expansion
from fase1_traducir_a_phantom import HFACT, POLYK
from campo_velocidad_turbulento import factory as factory_turb, MACH_OBJETIVO
from cs090_fase5b_correr import editar_cosmog_in, PHANTOMSETUP, PHANTOM   # congelado: sólo import
from leer_volcado_phantom import leer_dump, listar_dumps                  # congelado: sólo import
import cs090_diam_corregido as DIAMC

BASE = Path("/Users/alexis/phantom_cs073/f804_grano_n8000")
DIR_GRAFOS = Path("/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs090_f804_grafos")
CSV_SALIDA = Path("/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs090_f804_grano_n8000.csv")

SEED_LAYOUT = 12345
N_SWEEPS = 14
ITERS_LAYOUT = 100
THETA = 0.5
N_DEF = 8000
N_REPLICAS = 8

# El par extremo de O3-A, el mismo que corrió la demo de infraestructura a N=8000.
GRAFOS = {
    "r23_I":   dict(rule_id="A2-B0-C2-batch4-r23", seed=574060, clase="I"),
    "r10_III": dict(rule_id="A2-B0-C2-batch4-r10", seed=572799, clase="III"),
}


# ==========================================================================================
# 1) La perturbación de redondeo
# ==========================================================================================
def lado_perturbado(k: int, lado: float = LADO_FIJO) -> float:
    """`lado` movido k ULPs hacia +infinito. k=0 devuelve el valor sin tocar (réplica de referencia).

    Un ULP de 12,5992 es 1,78e-15 en absoluto, o sea 1,4e-16 relativo: el orden del épsilon de máquina,
    el mismo que separa a `repulsion_exacta` de `repulsion_exacta_reordenada`."""
    x = float(lado)
    for _ in range(int(k)):
        x = float(np.nextafter(x, np.inf))
    return x


# ==========================================================================================
# 2) Grafo (idéntico en las 8 réplicas) — se genera una vez y se GUARDA
# ==========================================================================================
def grafo_cacheado(clave: str, N: int) -> tuple[list, dict]:
    DIR_GRAFOS.mkdir(parents=True, exist_ok=True)
    p_adj = DIR_GRAFOS / f"{clave}_N{N}_adj.json.gz"
    p_meta = DIR_GRAFOS / f"{clave}_N{N}_meta.json"
    if p_adj.exists() and p_meta.exists():
        with gzip.open(p_adj, "rt") as f:
            adj_lista = [set(v) for v in json.load(f)]
        return adj_lista, json.loads(p_meta.read_text())

    g = GRAFOS[clave]
    t0 = time.time()
    p, m = reconstruir_regla_a2b0c2(seed=g["seed"], N=N, n_sweeps=N_SWEEPS)
    t_grafo = time.time() - t0
    adj_lista = m["adj_final"]
    meta = dict(clave=clave, rule_id=g["rule_id"], clase=g["clase"], seed=g["seed"], N=N,
                n_sweeps=N_SWEEPS, t_grafo_s=round(t_grafo, 1),
                K=p["K"], J=p["J"], noise=p["noise"], meandeg=p["meandeg"], kcap=p["kcap"],
                n_aristas_grafo_final=m["n_aristas"],
                grado_medio_grafo_final=2.0 * m["n_aristas"] / N,
                diam_grafo_final_OFICIAL=DIAMC.diam_gigante(adj_lista, N),
                holon_grafo_final=m["holonomia"])
    with gzip.open(p_adj, "wt") as f:
        json.dump([sorted(int(x) for x in s) for s in adj_lista], f)
    p_meta.write_text(json.dumps(meta, indent=2, default=str))
    return adj_lista, meta


# ==========================================================================================
# 3) IC — MISMA receta que `cs090_layout_bh_demo_n8000.escribir_ic_bh`, con `lado` parametrizado
# ==========================================================================================
def escribir_ic(adj_lista, N, ruta_salida, theta, lado, seed_layout=SEED_LAYOUT,
                iters_layout=ITERS_LAYOUT, n_pasos_expansion=60):
    adj = {i: adj_lista[i] for i in range(N)}
    n_aristas = sum(len(v) for v in adj_lista) // 2

    diag = {}
    t0 = time.time()
    pos = layout_barnes_hut(adj, N, lado=lado, iters=iters_layout, seed=seed_layout,
                            theta=theta, diagnostico=diag)
    t_layout = time.time() - t0

    expansion = Expansion(T0=T0)
    for step in range(n_pasos_expansion):
        expansion.paso_de_estiramiento(_T_reloj(step))
    a_final = expansion._a_prev
    pos = pos * a_final

    vel_gen = factory_turb(CS_SONORA, seed=TURB_SEED, mach=MACH_OBJETIVO)
    vel = vel_gen(pos, adj, np.ones(N))
    h_guess = np.full(N, HFACT)
    masa_particula = MASA_TOTAL_OBJETIVO / N

    with open(ruta_salida, "w") as f:
        f.write(f"# cosmogenesis_ic v2 F8-04 grano N={N} layout Barnes-Hut theta={theta} -- "
                f"lado_caja={lado:.17g} (nominal {LADO_FIJO:.17g} = N_referencia={N_REFERENCIA_LADO}^(1/3)) "
                f"masa_total_objetivo={MASA_TOTAL_OBJETIVO:.6g} (NO escala con n) -- "
                f"npart={N} n_aristas={n_aristas} masa_particula={masa_particula:.17g} hfact={HFACT} "
                f"polyk={POLYK:.17g} seed_layout={seed_layout} con_turbulencia=True\n")
        f.write(f"{N} {masa_particula:.17g} {HFACT} {POLYK:.17g}\n")
        for i in range(N):
            f.write(f"{float(pos[i, 0]):.17g} {float(pos[i, 1]):.17g} {float(pos[i, 2]):.17g} "
                    f"{float(vel[i, 0]):.17g} {float(vel[i, 1]):.17g} {float(vel[i, 2]):.17g} "
                    f"{float(h_guess[i]):.17g}\n")

    return dict(ruta=str(ruta_salida), n=N, masa_particula=masa_particula, lado_usado=lado,
                a_final=a_final, n_aristas=n_aristas, seed_layout=seed_layout, theta=theta,
                t_layout_s=round(t_layout, 1), prof_octree=diag.get("prof"),
                interacciones_por_iter=diag.get("interacciones"),
                pos_rms=float(np.sqrt(np.mean(np.sum(pos ** 2, axis=1)))))


def carpeta_de(clave, rep, N):
    return BASE / f"N{N}" / f"{clave}_rep{rep:02d}"


def cmd_ic(clave, rep, N=N_DEF, theta=THETA):
    carpeta = carpeta_de(clave, rep, N)
    carpeta.mkdir(parents=True, exist_ok=True)
    ic = carpeta / "cosmogenesis_ic.txt"
    meta_p = carpeta / "meta_regla.json"
    if ic.exists() and meta_p.exists():
        print(f"[IC-cache] {clave} rep{rep}", flush=True)
        return json.loads(meta_p.read_text())

    adj_lista, gmeta = grafo_cacheado(clave, N)
    lado = lado_perturbado(rep)
    info = escribir_ic(adj_lista, N, ic, theta, lado)
    meta = dict(gmeta)
    meta.update(rep=rep, k_ulp=rep, lado_nominal=LADO_FIJO,
                delta_lado_rel=(lado - LADO_FIJO) / LADO_FIJO, layout="barnes_hut", **info)
    meta_p.write_text(json.dumps(meta, indent=2, default=str))
    print(f"[IC] {clave} rep{rep} lado={lado:.17g} (drel={meta['delta_lado_rel']:.3e}) "
          f"t_layout={info['t_layout_s']}s", flush=True)
    return meta


# ==========================================================================================
# 4) Phantom
# ==========================================================================================
def cmd_run(clave, rep, cap_s, N=N_DEF):
    carpeta = carpeta_de(clave, rep, N)
    t0 = time.time()
    with open(carpeta / "setup.log", "w") as f:
        r = subprocess.run([PHANTOMSETUP, "cosmog"], cwd=carpeta, stdin=subprocess.DEVNULL,
                           stdout=f, stderr=subprocess.STDOUT)
    t_setup = time.time() - t0
    assert r.returncode == 0, f"phantomsetup falló en {carpeta}"
    editar_cosmog_in(carpeta / "cosmog.in")
    t1 = time.time()
    timeout = False
    try:
        with open(carpeta / "run.log", "w") as f:
            rr = subprocess.run([PHANTOM, "cosmog.in"], cwd=carpeta, stdin=subprocess.DEVNULL,
                                stdout=f, stderr=subprocess.STDOUT, timeout=cap_s)
        exit_run = rr.returncode
    except subprocess.TimeoutExpired:
        exit_run, timeout = None, True
    t_run = time.time() - t1
    dumps = listar_dumps(carpeta)
    info = dict(exit_setup=r.returncode, t_setup_s=round(t_setup, 1), exit_run=exit_run,
                t_run_s=round(t_run, 1), timeout=timeout,
                ultimo_dump=dumps[-1].name if dumps else None, n_dumps=len(dumps),
                cap_s=cap_s)
    (carpeta / "resultado_run.json").write_text(json.dumps(info, indent=2))
    print(f"[RUN] {clave} rep{rep} t_run={info['t_run_s']}s timeout={timeout} "
          f"ultimo={info['ultimo_dump']}", flush=True)
    return info


# ==========================================================================================
# 5) Medición al DUMP COMÚN
# ==========================================================================================
def idx_dump(p: Path) -> int:
    return int(p.stem[7:])


def leer_fraccion(carpeta: Path, idx: int) -> dict:
    p = carpeta / f"cosmog_{idx:05d}"
    gas, sinks = leer_dump(p)
    m_gas = float(gas["m"].sum()) if "m" in gas.columns else \
        float(gas.params.get("massoftype", 0.0)) * len(gas)
    m_sinks = float(sinks["m"].sum()) if sinks is not None else 0.0
    n_sinks = 0 if sinks is None else len(sinks)
    tot = m_gas + m_sinks
    return dict(dump=p.name, n_gas=len(gas), n_sumideros=n_sinks,
                masa_gas=m_gas, masa_sumideros=m_sinks, masa_total=tot,
                fraccion_masa_en_sumideros=(m_sinks / tot) if tot > 0 else None)


def cmd_curva(N=N_DEF):
    """σ entre réplicas COMO FUNCIÓN del dump (del tiempo simulado). Sirve para separar dos cosas que
    se confunden fácil: el ruido del método, y el hecho de que a t chico el sistema está en plena
    formación de sumideros y cualquier desfase temporal se lee como diferencia de masa."""
    import csv
    filas = []
    for clave in GRAFOS:
        ds = [d for d in sorted((BASE / f"N{N}").iterdir())
              if d.name.startswith(clave + "_rep") and listar_dumps(d)]
        if len(ds) < 2:
            continue
        comun = min(idx_dump(listar_dumps(d)[-1]) for d in ds)
        print(f"\n== {clave}: n={len(ds)} dump comun={comun}")
        print(f"{'dump':>5} {'t':>6} {'media':>9} {'sigma':>9} {'rango':>9} {'sig/med':>8} nsink")
        for k in list(range(20, comun, 20)) + [comun]:
            v, ns = [], []
            for d in ds:
                r = leer_fraccion(d, k)
                v.append(r["fraccion_masa_en_sumideros"]); ns.append(r["n_sumideros"])
            v = np.array(v, dtype=float)
            sd = float(v.std(ddof=1))
            filas.append(dict(grafo=clave, n_replicas=len(ds), dump=k, t=k / 1000.0,
                              media=float(v.mean()), sigma=sd, rango=float(v.max() - v.min()),
                              sigma_rel=sd / max(float(v.mean()), 1e-12),
                              sigma_en_particulas=sd * N,
                              n_sumideros_min=int(min(ns)), n_sumideros_max=int(max(ns)),
                              n_sumideros_media=float(np.mean(ns))))
            print(f"{k:5d} {k/1000:6.3f} {v.mean():9.6f} {sd:9.6f} {v.max()-v.min():9.6f} "
                  f"{sd/max(v.mean(),1e-12):8.3f} {min(ns)}-{max(ns)}")
    if filas:
        p = CSV_SALIDA.with_name("cs090_f804_sigma_vs_tiempo.csv")
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
            w.writeheader(); w.writerows(filas)
        print(f"\n[CSV] {len(filas)} filas -> {p}")
    return filas


def cmd_medir(dump_forzado=None, N=N_DEF):
    import csv
    filas = []
    resumen = []
    for clave in GRAFOS:
        carpetas = sorted(p for p in (BASE / f"N{N}").iterdir()
                          if p.is_dir() and p.name.startswith(clave + "_rep"))
        maxs = {}
        for c in carpetas:
            d = listar_dumps(c)
            if d:
                maxs[c] = idx_dump(d[-1])
        if not maxs:
            continue
        comun = min(maxs.values()) if dump_forzado is None else int(dump_forzado)
        for c, mx in sorted(maxs.items()):
            meta = json.loads((c / "meta_regla.json").read_text())
            run = json.loads((c / "resultado_run.json").read_text()) if (c / "resultado_run.json").exists() else {}
            f = leer_fraccion(c, comun)
            filas.append(dict(grafo=clave, clase=meta.get("clase"), rep=meta.get("rep"),
                              k_ulp=meta.get("k_ulp"), delta_lado_rel=meta.get("delta_lado_rel"),
                              theta=meta.get("theta"), N=N,
                              t_layout_s=meta.get("t_layout_s"), t_run_s=run.get("t_run_s"),
                              timeout=run.get("timeout"), max_dump_alcanzado=mx,
                              dump_comun=comun, **f))
        sub = [x for x in filas if x["grafo"] == clave]
        v = np.array([x["fraccion_masa_en_sumideros"] for x in sub], dtype=float)
        ns = np.array([x["n_sumideros"] for x in sub], dtype=float)
        resumen.append(dict(grafo=clave, n_replicas=len(v), dump_comun=comun,
                            media=float(v.mean()), sigma=float(v.std(ddof=1)) if len(v) > 1 else None,
                            rango=float(v.max() - v.min()),
                            sigma_rel=float(v.std(ddof=1) / v.mean()) if len(v) > 1 else None,
                            n_sumideros_media=float(ns.mean()),
                            n_sumideros_min=int(ns.min()), n_sumideros_max=int(ns.max())))
    if filas:
        campos = list(filas[0].keys())
        with open(CSV_SALIDA, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=campos)
            w.writeheader()
            w.writerows(filas)
        print(f"\n[CSV] {len(filas)} filas -> {CSV_SALIDA}")
    for r in resumen:
        print(f"[GRANO] {r['grafo']}: n={r['n_replicas']} dump={r['dump_comun']} "
              f"media={r['media']:.6f} sigma={r['sigma']} rango={r['rango']:.6f} "
              f"nsink {r['n_sumideros_min']}-{r['n_sumideros_max']}")
    (CSV_SALIDA.with_suffix(".resumen.json")).write_text(json.dumps(resumen, indent=2))
    return filas, resumen


# ==========================================================================================
# 6) Sanity — ¿la perturbación de ULP tiembla como el control de redondeo documentado?
# ==========================================================================================
def cmd_sanity(N=2000, theta=THETA):
    """A N=2000, compara las posiciones finales del layout con `lado` a 0, 1 y 2 ULPs.
    Vara documentada (INFRA §4.1): control de redondeo -> RMS 0,321 / máx 3,11 en caja de lado 12,6."""
    adj_lista, _ = grafo_cacheado("r23_I", N)
    adj = {i: adj_lista[i] for i in range(N)}
    out = []
    base = None
    for k in (0, 1, 2):
        lado = lado_perturbado(k)
        t0 = time.time()
        pos = layout_barnes_hut(adj, N, lado=lado, iters=ITERS_LAYOUT, seed=SEED_LAYOUT, theta=theta)
        t = time.time() - t0
        if base is None:
            base = pos
            out.append(dict(k=k, t_s=round(t, 1), rms=0.0, maxdif=0.0))
        else:
            d = pos - base
            out.append(dict(k=k, t_s=round(t, 1),
                            rms=float(np.sqrt(np.mean(np.sum(d ** 2, axis=1)))),
                            maxdif=float(np.abs(d).max())))
        print(f"  k={k} ULP lado={lado:.17g} t={t:.1f}s "
              f"RMS_vs_k0={out[-1]['rms']:.4f} max={out[-1]['maxdif']:.4f}", flush=True)
    (DIR_GRAFOS / "sanity_ulp_n2000.json").write_text(json.dumps(out, indent=2))
    print("  vara documentada (control de redondeo, N=2000): RMS 0,321 / max 3,11")
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "sanity":
        cmd_sanity()
    elif a[0] == "ic":
        cmd_ic(a[1], int(a[2]))
    elif a[0] == "run":
        cmd_run(a[1], int(a[2]), int(a[3]))
    elif a[0] == "curva":
        cmd_curva()
    elif a[0] == "medir":
        cmd_medir(a[1] if len(a) > 1 else None)
    else:
        raise SystemExit(__doc__)
