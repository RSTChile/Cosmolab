#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cs090_layout_bh_validar.py — validación del layout Barnes-Hut contra el layout congelado
=========================================================================================

QUÉ CONTESTA
------------
`cs090_layout_barnes_hut.py` propone reemplazar la suma N² de repulsión del layout Fruchterman-Reingold
por una aproximación de árbol O(N log N). Antes de usarlo en producción hay que contestar tres cosas con
números, no con confianza:

  1. ¿El árbol está bien programado? — Con θ=0 no se puede aceptar ninguna celda, así que la suma del
     árbol tiene que ser la MISMA suma partícula-a-partícula del original. Si no lo es, hay un bug.
  2. ¿Cuánto error mete cada θ, y cuánto tiempo ahorra? — la curva error-vs-velocidad.
  3. ¿Ese error es tolerable? — el patrón de comparación NO es "error cero": el propio método tiene un
     piso de ruido, porque la relajación FR amplifica caóticamente diferencias de nivel 1e-16
     (`FASE6_O3B_control_rewiring_CS.md`: re-correr lo mismo dio 15-23 unidades de diferencia en 4 de 12
     casos). Acá ese piso se REPRODUCE A PEDIDO con un control limpio: el mismo layout exacto, con la
     misma cuenta sumada en orden inverso (`metodo="exacta_reordenada"`) — mismos sumandos, distinto
     último bit. Si el error de un θ queda por debajo de ese control, el layout nuevo no es más ruidoso
     que el viejo consigo mismo.

EN SIMPLE, CON ANALOGÍA
-----------------------
Es como cambiar una balanza de precisión por otra más rápida. No se le pide a la nueva que dé el mismo
número hasta el último decimal: se le pide que la diferencia con la vieja sea menor que la diferencia que
la vieja tiene consigo misma cuando se pesa dos veces lo mismo. Eso último es el "temblor" propio de la
balanza, y es la vara honesta.

QUÉ SE MIDE EN CADA LAYOUT
--------------------------
- posiciones: RMS y máximo de la diferencia contra el layout original;
- el observable de aglomeración de las condiciones iniciales: `fof_masa` (importado tal cual de
  `cs090_fase6_o4a_observable_comun.py`) con b = 0.20 / 0.30 / 0.50 y umbral de masa fija 47.0, y la
  dispersión de densidad local `dens_k8_cv` — EXACTAMENTE las mismas cuatro varas y las mismas
  constantes que usa `cs090_fase6_o3a_geometria_ic.py` (mismo LADO_ESCRITO=97.593, misma masa total
  18800). Las posiciones se dilatan con el mismo `Expansion` de 60 pasos que la IC real antes de medir.

MODOS
-----
    ./venv/bin/python cs090_layout_bh_validar.py fuerza          # test 1: θ=0 exacto + curva de error
    ./venv/bin/python cs090_layout_bh_validar.py worker <i>      # un grafo: todos los layouts
    ./venv/bin/python cs090_layout_bh_validar.py tabla           # junta JSONs -> CSV + PNG
    ./venv/bin/python cs090_layout_bh_validar.py escala          # cronometraje N=1000..16000

Ningún archivo congelado se modifica: `p_semilla_causal.py`, `cs090_fase5b_phantom_adaptador.py`,
`cs090_fase6_o4a_observable_comun.py` sólo se importan. No se declara cierre ni veredicto.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis")

from cs072_modulos.piezas.p_semilla_causal import layout_resortes          # CONGELADO: sólo import
from cs090_layout_barnes_hut import (layout_barnes_hut, repulsion_barnes_hut, repulsion_exacta,
                                     repulsion_exacta_reordenada, _profundidad_octree)
from cs090_fase5b_phantom_adaptador import (reconstruir_regla_a2b0c2, LADO_FIJO, MASA_TOTAL_OBJETIVO)
from cs073_cierre_holistico import T0, _T_reloj
from cs072_modulos.piezas.p_expansion import Expansion

AQUI = Path("/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis")
SEL_JSON = AQUI / "cs090_fase6_o3a_pares_seleccionados.json"
SALIDA = Path("/Users/alexis/phantom_cs073/infra_layout_bh")
SALIDA.mkdir(parents=True, exist_ok=True)

THETAS = (0.0, 0.3, 0.5, 0.7, 1.0)
CON_THETA0 = (0, 13)       # índices de grafo en los que también se corre el layout completo con θ=0
N_VALID = 2000
ITERS = 100
SEED_LAYOUT = 12345
N_SWEEPS = 14
LADO_ESCRITO = 97.593      # misma constante que cs090_fase6_o3a_geometria_ic.py
MASA_MIN_FIJA = 47.0
BS = (0.20, 0.30, 0.50)


def _a_final() -> float:
    """La misma dilatación isótropa estática de 60 pasos que aplica la IC real."""
    e = Expansion(T0=T0)
    for step in range(60):
        e.paso_de_estiramiento(_T_reloj(step))
    return e._a_prev


A_FINAL = _a_final()


def metricas_ic(pos: np.ndarray) -> dict:
    """Las MISMAS cuatro varas de `cs090_fase6_o3a_geometria_ic.py`, sobre las posiciones ya dilatadas."""
    from cs090_fase6_o4a_observable_comun import fof_masa        # import puro
    from scipy.spatial import cKDTree
    p = pos * A_FINAL
    n = len(p)
    masas = np.full(n, MASA_TOTAL_OBJETIVO / n)
    sep_media = LADO_ESCRITO / n ** (1.0 / 3.0)
    out = {}
    for b in BS:
        frac, ngr = fof_masa(p, masas, b * sep_media, MASA_MIN_FIJA)
        out[f"fof_b{b:.2f}"] = float(frac)
        out[f"ngrupos_b{b:.2f}"] = int(ngr)
    dist, _ = cKDTree(p).query(p, k=9)
    r8 = dist[:, 8]
    out["dens_k8_cv"] = float(np.std(1.0 / r8 ** 3) / np.mean(1.0 / r8 ** 3))
    return out


def reglas() -> list[dict]:
    """Las 12 reglas de validación: una por par de los seleccionados en O3-A (la Clase III de cada par),
    más las Clase I hasta completar 12. Son grafos YA usados en la serie, con sus semillas."""
    sel = json.loads(SEL_JSON.read_text())
    r = [dict(rule_id=s["rid_III"], seed=s["seed_III"], clase="III", par=s["par"]) for s in sel]
    r += [dict(rule_id=s["rid_I"], seed=s["seed_I"], clase="I", par=s["par"]) for s in sel]
    return r


# ==========================================================================================
# TEST 1 — a nivel de una sola evaluación de fuerza (sin dinámica): θ=0 debe ser exacto
# ==========================================================================================
def modo_fuerza() -> None:
    import csv
    filas = []
    for N in (500, 1000, 2000, 4000):
        rng = np.random.default_rng(31337)
        pos = rng.uniform(0.0, LADO_FIJO, size=(N, 3))
        k_fr = (LADO_FIJO ** 3 / N) ** (1.0 / 3.0)
        t0 = time.time(); fe = repulsion_exacta(pos, k_fr); t_ex = time.time() - t0
        fr = repulsion_exacta_reordenada(pos, k_fr)
        norma = np.linalg.norm(fe, axis=1)
        ref = float(np.mean(norma))
        # control de redondeo: misma cuenta, otro orden de suma
        e_round = np.linalg.norm(fr - fe, axis=1)
        filas.append(dict(N=N, theta="control_redondeo(N2 orden inverso)", t_s=round(t_ex, 3),
                          err_rel_medio=float(np.mean(e_round) / ref),
                          err_rel_max=float(e_round.max() / ref),
                          err_abs_max=float(e_round.max()), interacciones=N * (N - 1)))
        for th in THETAS:
            cont = {}
            t0 = time.time()
            fb = repulsion_barnes_hut(pos, k_fr, LADO_FIJO, th, contadores=cont)
            t_bh = time.time() - t0
            e = np.linalg.norm(fb - fe, axis=1)
            filas.append(dict(N=N, theta=th, t_s=round(t_bh, 3),
                              err_rel_medio=float(np.mean(e) / ref),
                              err_rel_max=float(e.max() / ref), err_abs_max=float(e.max()),
                              interacciones=cont["interacciones"]))
            print(f"N={N:5d} theta={th:<4} t={t_bh:7.3f}s (exacta {t_ex:6.3f}s) "
                  f"err_rel_medio={np.mean(e)/ref:.3e} interacc={cont['interacciones']:,}", flush=True)
        print(f"N={N:5d} control redondeo: err_rel_medio={filas[-6]['err_rel_medio']:.3e}", flush=True)
    ruta = AQUI / "cs090_layout_bh_fuerza.csv"
    with open(ruta, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader(); w.writerows(filas)
    print(f"[fuerza] {len(filas)} filas -> {ruta}")


# ==========================================================================================
# TEST 2 — layout completo (100 iteraciones) sobre grafos REALES de la serie
# ==========================================================================================
def worker(i: int) -> None:
    r = reglas()[i]
    destino = SALIDA / f"valid_{r['rule_id']}_{r['clase']}.json"
    t0 = time.time()
    p, m = reconstruir_regla_a2b0c2(seed=r["seed"], N=N_VALID, n_sweeps=N_SWEEPS)
    adj = {j: m["adj_final"][j] for j in range(N_VALID)}
    res = dict(r, n_aristas=m["n_aristas"], t_grafo_s=round(time.time() - t0, 2), N=N_VALID)

    t0 = time.time()
    pos_ref = layout_resortes(adj, N_VALID, LADO_FIJO, iters=ITERS, seed=SEED_LAYOUT)
    res["t_original_s"] = round(time.time() - t0, 2)
    res["original"] = metricas_ic(pos_ref)

    # θ=0 en el layout COMPLETO (100 iteraciones) cuesta más que el propio original (es la misma suma
    # N² más el costo del árbol), y su papel — probar que el árbol no tiene bugs — ya lo cumple el test
    # de fuerza (`modo fuerza`), que lo mide con precisión de último bit. Por eso acá sólo se corre en
    # los grafos indicados en `CON_THETA0`, como testigo, y no en los 12.
    variantes = [("control_redondeo", dict(metodo="exacta_reordenada", theta=0.0))]
    if i in CON_THETA0:
        variantes += [("bh_theta0.0", dict(metodo="bh", theta=0.0))]
    variantes += [(f"bh_theta{th}", dict(metodo="bh", theta=th)) for th in THETAS if th > 0]

    for nombre, kw in variantes:
        t0 = time.time()
        pos = layout_barnes_hut(adj, N_VALID, LADO_FIJO, iters=ITERS, seed=SEED_LAYOUT, **kw)
        dt = time.time() - t0
        d = pos - pos_ref
        res[nombre] = dict(t_s=round(dt, 2),
                           rms=float(np.sqrt(np.mean(np.sum(d ** 2, axis=1)))),
                           max_abs=float(np.abs(d).max()),
                           **metricas_ic(pos))
        print(f"[{r['rule_id']}] {nombre:20s} t={dt:7.2f}s rms={res[nombre]['rms']:.3e} "
              f"fof020={res[nombre]['fof_b0.20']:.4f} (orig {res['original']['fof_b0.20']:.4f})",
              flush=True)

    destino.write_text(json.dumps(res, indent=2))
    print(f"[OK] {r['rule_id']} -> {destino}", flush=True)


# ==========================================================================================
# TEST 3 — escalamiento real del layout nuevo
# ==========================================================================================
def modo_escala(ns=(1000, 2000, 4000, 8000, 16000), theta=0.5, iters=10, seed_regla=576000) -> None:
    """Cronometra el layout nuevo. Se cronometran `iters` iteraciones y se extrapola linealmente a 100
    (el costo por iteración es constante: mismo árbol, misma cuenta), para no gastar el presupuesto en
    los puntos grandes. Se anota el número de interacciones por iteración, que es la medida
    ALGORÍTMICA del costo (independiente de la máquina)."""
    import csv
    filas = []
    for N in ns:
        t0 = time.time()
        p, m = reconstruir_regla_a2b0c2(seed=seed_regla, N=N, n_sweeps=N_SWEEPS)
        t_grafo = time.time() - t0
        adj = {j: m["adj_final"][j] for j in range(N)}
        d = {}
        t0 = time.time()
        layout_barnes_hut(adj, N, LADO_FIJO, iters=iters, seed=SEED_LAYOUT, theta=theta, diagnostico=d)
        dt = time.time() - t0
        fila = dict(N=N, theta=theta, iters_medidos=iters, t_medido_s=round(dt, 2),
                    t_100iters_s=round(dt * 100.0 / iters, 1), t_grafo_s=round(t_grafo, 2),
                    prof_octree=d["prof"], interacciones_por_iter=d["interacciones"],
                    interacciones_por_particula=round(d["interacciones"] / N, 1))
        filas.append(fila)
        print(fila, flush=True)
    ruta = AQUI / "cs090_layout_bh_escala.csv"
    with open(ruta, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader(); w.writerows(filas)
    print(f"[escala] -> {ruta}")


# ==========================================================================================
# TABLA + FIGURAS
# ==========================================================================================
def tabla() -> None:
    import csv
    import pandas as pd
    js = sorted(SALIDA.glob("valid_*.json"))
    filas = []
    for j in js:
        r = json.loads(j.read_text())
        for nombre in [k for k in r if isinstance(r[k], dict) and k != "original"]:
            v = r[nombre]
            f = dict(rule_id=r["rule_id"], clase=r["clase"], par=r["par"], N=r["N"],
                     n_aristas=r["n_aristas"], variante=nombre, t_s=v["t_s"],
                     t_original_s=r["t_original_s"],
                     aceleracion=round(r["t_original_s"] / v["t_s"], 2) if v["t_s"] else None,
                     rms=v["rms"], max_abs=v["max_abs"])
            for c in [f"fof_b{b:.2f}" for b in BS] + ["dens_k8_cv"]:
                f[c] = v[c]
                f[f"d_{c}"] = v[c] - r["original"][c]
            for b in BS:
                f[f"ngrupos_b{b:.2f}"] = v[f"ngrupos_b{b:.2f}"]
                f[f"d_ngrupos_b{b:.2f}"] = v[f"ngrupos_b{b:.2f}"] - r["original"][f"ngrupos_b{b:.2f}"]
            filas.append(f)
    d = pd.DataFrame(filas)
    ruta = AQUI / "cs090_layout_bh_validacion.csv"
    d.to_csv(ruta, index=False)
    print(f"[tabla] {len(d)} filas ({len(js)} grafos) -> {ruta}")

    cols = [f"d_fof_b{b:.2f}" for b in BS] + ["d_dens_k8_cv"]
    orden = ["control_redondeo"] + [f"bh_theta{t}" for t in THETAS]
    res = []
    for v in orden:
        g = d[d.variante == v]
        if not len(g):
            continue
        fila = dict(variante=v, n_grafos=len(g), t_medio_s=round(g.t_s.mean(), 1),
                    aceleracion_media=round(g.aceleracion.mean(), 2),
                    rms_medio=g.rms.mean(), max_abs_medio=g.max_abs.mean())
        for c in cols:
            fila[f"{c}_absmedio"] = float(np.mean(np.abs(g[c])))
            fila[f"{c}_max"] = float(np.max(np.abs(g[c])))
        res.append(fila)
    rr = pd.DataFrame(res)
    rr.to_csv(AQUI / "cs090_layout_bh_resumen.csv", index=False)
    pd.set_option("display.width", 250)
    print(rr.to_string(index=False))

    # ---------- figuras ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    piso = rr[rr.variante == "control_redondeo"]
    for c, mk in zip(cols, "osd^"):
        g = rr[rr.variante != "control_redondeo"]
        ax[0].plot([float(v.replace("bh_theta", "")) for v in g.variante],
                   g[f"{c}_absmedio"], marker=mk, label=c.replace("d_", "Δ "))
        if len(piso):
            ax[0].axhline(float(piso[f"{c}_absmedio"].iloc[0]), ls=":", lw=1, alpha=.6)
    ax[0].set_yscale("log"); ax[0].set_xlabel("θ (criterio de apertura)")
    ax[0].set_ylabel("|Δ| medio del observable vs layout original")
    ax[0].set_title("Error inducido por θ\n(líneas punteadas = piso de ruido del propio método)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

    # Panel derecho: se usa FoF b=0.50 y NO b=0.20. Razón: b=0.20 es la vara con el piso de ruido más
    # alto (0.0119) y con n=10 no ordena los θ (0.3 < 0.7 < 0.5 < 1.0) -- dibujarla haría ver un
    # zigzag que es ruido de muestreo, no la curva error-vs-velocidad. b=0.50 tiene el piso más bajo
    # y sí ordena monótonamente. Se marca el sesgo medio CON SIGNO además del valor absoluto.
    g = rr[rr.variante != "control_redondeo"]
    ax[1].plot(g.t_medio_s, g["d_fof_b0.50_absmedio"], "o-", label="|Δ FoF b=0.50| medio")
    for _, row in g.iterrows():
        ax[1].annotate(f"θ={row.variante.replace('bh_theta','')}",
                       (row.t_medio_s, row["d_fof_b0.50_absmedio"]),
                       textcoords="offset points", xytext=(5, -12), fontsize=9)
    if len(piso):
        ax[1].axhline(float(piso["d_fof_b0.50_absmedio"].iloc[0]), ls=":", color="k",
                      label="piso de ruido (mismo algoritmo, otro redondeo)")
        ax[1].axvline(float(d.t_original_s.mean()), ls="--", color="r", alpha=.6,
                      label=f"layout original ({d.t_original_s.mean():.0f} s)")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("tiempo por grafo, N=2000 (s) — más a la izquierda = más rápido")
    ax[1].set_ylabel("|Δ FoF b=0.50| medio vs layout original")
    ax[1].set_title("Error vs velocidad"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(AQUI / "cs090_layout_bh_error_vs_velocidad.png", dpi=140)
    print("[fig] cs090_layout_bh_error_vs_velocidad.png")

    # ---------- escalamiento ----------
    ruta_esc = AQUI / "cs090_layout_bh_escala.csv"
    if ruta_esc.exists():
        e = pd.read_csv(ruta_esc)
        fig, ax = plt.subplots(figsize=(6.6, 5))
        ax.plot(e.N, e.t_100iters_s, "o-", label="Barnes-Hut θ=0.5 (medido)")
        # puntos del layout original, medidos en FASE6_O3A_convergencia_resolucion_CS.md §8.1
        n_o = np.array([1000, 1414, 2000, 2828, 4000])
        t_o = np.array([23.3, 58.2, 98.4, 188.0, 656.0])
        ax.plot(n_o, t_o, "s--", color="r", label="original O(N²) (medido en O3-A §8.1)")
        ax.plot([8000], [4380], "*", color="r", ms=14, label="original, extrapolado (73 min)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("N (partículas)"); ax.set_ylabel("segundos por grafo, 100 iteraciones")
        ax.set_title("Escalamiento del layout")
        ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(AQUI / "cs090_layout_bh_escalamiento.png", dpi=140)
        print("[fig] cs090_layout_bh_escalamiento.png")
        lx, ly = np.log(e.N.values.astype(float)), np.log(e.t_100iters_s.values.astype(float))
        print(f"[escala] exponente global BH = {np.polyfit(lx, ly, 1)[0]:.3f}")
        for a, b in zip(range(len(lx) - 1), range(1, len(lx))):
            print(f"   tramo {e.N.iloc[a]}->{e.N.iloc[b]}: exponente "
                  f"{(ly[b]-ly[a])/(lx[b]-lx[a]):.3f}")


# ==========================================================================================
# TEST 3-bis — escalamiento POR EVALUACIÓN de fuerza (barato y robusto a la contención)
# ==========================================================================================
def modo_escala_rapida(ns=(1000, 2000, 4000, 8000, 16000), theta=0.5, repes=3) -> None:
    """Cronometra UNA evaluación de la repulsión (que es el 100 % del costo que cambia; el término
    atractivo es O(M) y ya era barato) y extrapola a las 100 iteraciones del layout.

    Por qué así y no corriendo layouts enteros: la máquina está compartida con ~20 agentes y la
    contención inflaría los puntos grandes de forma irregular. Se toma el MÍNIMO de `repes`
    evaluaciones, que es el estimador robusto habitual bajo contención (el mínimo se acerca al costo sin
    competencia). Además se reporta el nº de INTERACCIONES por iteración, que es la medida algorítmica
    del costo y no depende de la máquina en absoluto.
    """
    import csv
    filas = []
    for N in ns:
        rng = np.random.default_rng(31337)
        pos = rng.uniform(0.0, LADO_FIJO, size=(N, 3))
        k_fr = (LADO_FIJO ** 3 / N) ** (1.0 / 3.0)
        cont = {}
        ts = []
        for _ in range(repes):
            t0 = time.time()
            repulsion_barnes_hut(pos, k_fr, LADO_FIJO, theta, contadores=cont)
            ts.append(time.time() - t0)
        fila = dict(N=N, theta=theta, t_eval_min_s=round(min(ts), 4),
                    t_eval_mediana_s=round(float(np.median(ts)), 4),
                    t_100iters_s=round(min(ts) * 100.0, 1),
                    prof_octree=cont["prof"], interacciones_por_iter=cont["interacciones"],
                    interacciones_por_particula=round(cont["interacciones"] / N, 1))
        filas.append(fila)
        print(fila, flush=True)
    ruta = AQUI / "cs090_layout_bh_escala.csv"
    with open(ruta, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader(); w.writerows(filas)
    lx = np.log([f["N"] for f in filas]); ly = np.log([f["t_100iters_s"] for f in filas])
    li = np.log([f["interacciones_por_iter"] for f in filas])
    print(f"[escala] exponente TIEMPO   = {np.polyfit(lx, ly, 1)[0]:.3f}")
    print(f"[escala] exponente INTERACC = {np.polyfit(lx, li, 1)[0]:.3f}")
    for a in range(len(lx) - 1):
        print(f"   tramo {filas[a]['N']}->{filas[a+1]['N']}: exp_tiempo "
              f"{(ly[a+1]-ly[a])/(lx[a+1]-lx[a]):.3f}  exp_interacc "
              f"{(li[a+1]-li[a])/(lx[a+1]-lx[a]):.3f}")
    print(f"[escala] -> {ruta}")


if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "tabla"
    if modo == "fuerza":
        modo_fuerza()
    elif modo == "worker":
        worker(int(sys.argv[2]))
    elif modo in ("escala", "escala_rapida"):
        ns = tuple(int(x) for x in sys.argv[2].split(",")) if len(sys.argv) > 2 else \
            (1000, 2000, 4000, 8000, 16000)
        (modo_escala if modo == "escala" else modo_escala_rapida)(ns=ns)
    elif modo == "tabla":
        tabla()
    elif modo == "listar":
        for i, r in enumerate(reglas()):
            print(i, r)
    else:
        print(__doc__)
