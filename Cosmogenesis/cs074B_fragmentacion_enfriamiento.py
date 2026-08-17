#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs074B_fragmentacion_enfriamiento.py — ¿Dónde actúa el enfriamiento?
=========================================================================================

Quién soy / qué hago (código autodescriptivo):
  Implementa PROTOCOLO_cs074B_fragmentacion_enfriamiento_PREREGISTRO.md (leer primero).
  Reusa `correr_holistico_energia()` de cs074_energia_holistica.py (motor verificado) con
  dos parámetros aditivos nuevos (`tasa_enfriamiento`, `seed_dens_null`) que NO cambian el
  comportamiento por defecto de cs074 original. Mide fragmentación (n_clusters_finales,
  frac_masa_en_mayor_cluster) vs intensidad de enfriamiento, REAL vs NULL barajado.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cs074_energia_holistica import correr_holistico_energia  # noqa: E402

OUT = HERE / "resultados_cs074B_fragmentacion_enfriamiento"
OUT.mkdir(exist_ok=True)

TASA_ENFRIAMIENTO_B = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]   # protocolo §4
AMP_RUGOSIDAD_B = [0.5, 1.0, 1.5, 2.5, 4.0]                                       # protocolo §4
E_RESERVA_B = [1e-2, 1.0, 1e2]                                                    # protocolo §4
SEMILLAS_B = list(range(12))                                                     # protocolo §4
SEED_NULL_OFFSET = 90_000


def correr_experimento_b(nq=300, naq=210, ne=100, npos=70, pasos_basal=150, log_fn=print):
    t0 = time.time()
    n_celdas = len(TASA_ENFRIAMIENTO_B) * len(AMP_RUGOSIDAD_B) * len(E_RESERVA_B) * len(SEMILLAS_B)
    log_fn(f"[B] barrido: {len(TASA_ENFRIAMIENTO_B)} tasa_enf x {len(AMP_RUGOSIDAD_B)} eps x "
           f"{len(E_RESERVA_B)} E_reserva x {len(SEMILLAS_B)} semillas = {n_celdas} REAL + "
           f"{n_celdas} NULL = {2*n_celdas} corridas")

    filas = []
    n_done = 0
    for te in TASA_ENFRIAMIENTO_B:
        for amp in AMP_RUGOSIDAD_B:
            for er in E_RESERVA_B:
                for s in SEMILLAS_B:
                    real = correr_holistico_energia(
                        nq=nq, naq=naq, ne=ne, npos=npos, pasos_basal=pasos_basal,
                        amp_rugosidad=amp, E_reserva=er, tasa_enfriamiento=te,
                        seed_layout=12345 + s, guardar_curva=False,
                    )
                    null = correr_holistico_energia(
                        nq=nq, naq=naq, ne=ne, npos=npos, pasos_basal=pasos_basal,
                        amp_rugosidad=amp, E_reserva=er, tasa_enfriamiento=te,
                        seed_layout=12345 + s, seed_dens_null=SEED_NULL_OFFSET + s,
                        guardar_curva=False,
                    )
                    real["seed"] = s
                    null["seed"] = s
                    filas.append(dict(real=real, null=null,
                                       tasa_enfriamiento=te, amp_rugosidad=amp, E_reserva=er, seed=s))
                    n_done += 1
            log_fn(f"[B] tasa_enf={te} eps={amp} listo (celda) "
                   f"({n_done}/{n_celdas}) t={time.time()-t0:.0f}s")

    elapsed = time.time() - t0
    log_fn(f"[B] TOTAL elapsed={elapsed:.0f}s")
    return dict(filas=filas, elapsed_s=elapsed,
                grid=dict(tasa_enfriamiento=TASA_ENFRIAMIENTO_B, amp_rugosidad=AMP_RUGOSIDAD_B,
                          E_reserva=E_RESERVA_B, semillas=SEMILLAS_B))


def analizar_b(resultado):
    """Protocolo §6: curva de fragmentación vs tasa_enfriamiento (agregada sobre eps/E_reserva/
    semillas), z-score REAL vs NULL, y veredicto PASS/FAIL."""
    filas = [f for f in resultado["filas"] if f["real"].get("ok") and f["null"].get("ok")]

    por_te = {}
    for f in filas:
        te = f["tasa_enfriamiento"]
        por_te.setdefault(te, {"n_clusters_real": [], "n_clusters_null": [],
                                "frac_mayor_real": [], "frac_mayor_null": []})
        por_te[te]["n_clusters_real"].append(f["real"]["n_clusters_finales"])
        por_te[te]["n_clusters_null"].append(f["null"]["n_clusters_finales"])
        por_te[te]["frac_mayor_real"].append(f["real"]["frac_masa_en_mayor_cluster"])
        por_te[te]["frac_mayor_null"].append(f["null"]["frac_masa_en_mayor_cluster"])

    curva = {}
    for te, d in sorted(por_te.items()):
        nr, nn = np.array(d["n_clusters_real"]), np.array(d["n_clusters_null"])
        fr, fn = np.array(d["frac_mayor_real"]), np.array(d["frac_mayor_null"])
        sd_n = max(np.sqrt((nr.var() + nn.var()) / 2.0), 1e-9)
        z_n = float((nr.mean() - nn.mean()) / sd_n)
        curva[te] = dict(
            n_clusters_real_media=float(nr.mean()), n_clusters_real_std=float(nr.std()),
            n_clusters_null_media=float(nn.mean()),
            frac_mayor_real_media=float(fr.mean()), frac_mayor_real_std=float(fr.std()),
            frac_mayor_null_media=float(fn.mean()),
            z_n_clusters=z_n, n=len(nr),
        )

    tes = sorted(curva.keys())
    n_real_curva = [curva[t]["n_clusters_real_media"] for t in tes]
    frac_mayor_curva = [curva[t]["frac_mayor_real_media"] for t in tes]
    corr_monotona = float(np.corrcoef(tes, n_real_curva)[0, 1]) if len(tes) > 2 else None

    z_extremo = curva[tes[-1]]["z_n_clusters"] - curva[tes[0]]["z_n_clusters"] if len(tes) >= 2 else None
    n_celdas_z_mayor_2 = sum(1 for t in tes if abs(curva[t]["z_n_clusters"]) > 2.0)
    frac_celdas_significativas = n_celdas_z_mayor_2 / len(tes) if tes else 0.0

    pass_b = bool(
        corr_monotona is not None and corr_monotona > 0.5 and
        frac_celdas_significativas >= 0.5
    )

    return dict(
        curva_vs_tasa_enfriamiento=curva,
        corr_tasa_enfriamiento_vs_n_clusters=corr_monotona,
        n_celdas_con_z_mayor_2=n_celdas_z_mayor_2, n_celdas_total=len(tes),
        frac_celdas_significativas=frac_celdas_significativas,
        PASS_cs074B=pass_b,
        n_ok=len(filas), n_total=len(resultado["filas"]),
    )


def main():
    log_lines = []

    def p(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, file=sys.stderr, flush=True)
        log_lines.append(line)

    full = "--full" in sys.argv
    if full:
        nq, naq, ne, npos, pasos_basal = 300, 210, 100, 70, 150
    else:
        p("=== SMOKE TEST (escala reducida) ===")
        global TASA_ENFRIAMIENTO_B, AMP_RUGOSIDAD_B, E_RESERVA_B, SEMILLAS_B
        TASA_ENFRIAMIENTO_B = [0.0, 0.5, 3.0]
        AMP_RUGOSIDAD_B = [1.5]
        E_RESERVA_B = [1.0]
        SEMILLAS_B = [0, 1]
        nq, naq, ne, npos, pasos_basal = 60, 42, 20, 14, 40

    resultado = correr_experimento_b(nq=nq, naq=naq, ne=ne, npos=npos,
                                      pasos_basal=pasos_basal, log_fn=p)
    balance = analizar_b(resultado)
    resultado["analisis"] = balance
    resultado["log"] = log_lines
    p(f"[B] PASS_cs074B = {balance['PASS_cs074B']}")
    p(f"[B] corr(tasa_enfriamiento, n_clusters) = {balance['corr_tasa_enfriamiento_vs_n_clusters']}")
    p(f"[B] celdas con |z|>2: {balance['n_celdas_con_z_mayor_2']}/{balance['n_celdas_total']}")

    nombre = "cs074B_result_FULL.json" if full else "cs074B_result_smoke.json"
    out_json = OUT / nombre
    out_json.write_text(json.dumps(resultado, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    p(f"[archivo] {out_json}")


if __name__ == "__main__":
    main()
