#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs074A_asimetria_techo.py — ¿Por qué demasiada asimetría produce menos estructura?
=========================================================================================

Quién soy / qué hago (código autodescriptivo):
  Implementa PROTOCOLO_cs074A_asimetria_techo_PREREGISTRO.md (leer ese documento primero,
  congelado ANTES de este script). Reusa `correr_holistico_energia()` de
  `cs074_energia_holistica.py` (motor YA verificado, no se toca su física) para barrer ε
  fino y amplio, midiendo en paralelo 3 observables independientes que pueden explicar el
  techo no-monótono en ε que apareció (sin buscarlo) en el barrido original de cs074:
  cuánta masa queda ligada (el techo a explicar), cuánta reserva se gasta temprano, y qué
  tan fragmentada queda la materia.

No edita cs074_energia_holistica.py salvo por los dos campos aditivos de fragmentación ya
agregados (frac_masa_en_mayor_cluster, masas_clusters_finales) -- no cambian ningún valor
ya reportado del barrido original.
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

OUT = HERE / "resultados_cs074A_asimetria_techo"
OUT.mkdir(exist_ok=True)

AMP_RUGOSIDAD_A = list(np.logspace(-3, 1, 20))    # protocolo §2: 1e-3 a 10, 20 pts log
E_RESERVA_A = list(np.logspace(-3, 3, 7))         # protocolo §2: mismo grid que cs074
SEMILLAS_A = list(range(12))                      # protocolo §2: 12 semillas


def _frac_reserva_gastada_temprano(r):
    """Protocolo §3.2: de la curva por-paso, qué fracción de la reserva total ya se gastó
    (ligada_acum) al llegar a 1/3 del total de pasos. r debe traer guardar_curva=True."""
    curva = r.get("curva")
    if not curva or not np.isfinite(r.get("E_reserva_abs", float("inf"))):
        return None
    n = len(curva)
    idx_temprano = max(0, n // 3 - 1)
    ligada_temprano = curva[idx_temprano]["ligada_acum"]
    reserva_total = r["E_reserva_abs"]
    return ligada_temprano / reserva_total if reserva_total > 0 else 0.0


def correr_experimento_a(nq=300, naq=210, ne=100, npos=70, pasos_basal=150, log_fn=print):
    t0 = time.time()
    log_fn(f"[A] barrido: {len(AMP_RUGOSIDAD_A)} eps x {len(E_RESERVA_A)} E_reserva x "
           f"{len(SEMILLAS_A)} semillas = "
           f"{len(AMP_RUGOSIDAD_A)*len(E_RESERVA_A)*len(SEMILLAS_A)} finitas + "
           f"{len(AMP_RUGOSIDAD_A)*len(SEMILLAS_A)} de control (sin energía)")

    # --- brazo SIN energía (control mecánico, independiente de E_reserva) ---
    control = {}
    n_ctrl = 0
    for amp in AMP_RUGOSIDAD_A:
        for s in SEMILLAS_A:
            r = correr_holistico_energia(nq=nq, naq=naq, ne=ne, npos=npos,
                                          pasos_basal=pasos_basal, amp_rugosidad=amp,
                                          energia_on=False, seed_layout=12345 + s,
                                          guardar_curva=False)
            control[(round(amp, 8), s)] = r
            n_ctrl += 1
        log_fn(f"[A] control (sin energía) eps={amp:.4g} listo ({n_ctrl}/"
               f"{len(AMP_RUGOSIDAD_A)*len(SEMILLAS_A)}) t={time.time()-t0:.0f}s")

    # --- brazo CON energía (barrido principal, con curva para observable 2) ---
    filas = []
    n_done, n_total = 0, len(AMP_RUGOSIDAD_A) * len(E_RESERVA_A) * len(SEMILLAS_A)
    for amp in AMP_RUGOSIDAD_A:
        for er in E_RESERVA_A:
            for s in SEMILLAS_A:
                r = correr_holistico_energia(nq=nq, naq=naq, ne=ne, npos=npos,
                                              pasos_basal=pasos_basal, amp_rugosidad=amp,
                                              E_reserva=er, energia_on=True,
                                              seed_layout=12345 + s, guardar_curva=True)
                r["seed"] = s
                if r.get("ok"):
                    r["frac_reserva_gastada_temprano"] = _frac_reserva_gastada_temprano(r)
                    r["curva"] = None  # ya extraído lo que hacía falta; no persistir crudo (tamaño)
                filas.append(r)
                n_done += 1
        log_fn(f"[A] eps={amp:.4g} listo ({n_done}/{n_total}) t={time.time()-t0:.0f}s")

    elapsed = time.time() - t0
    log_fn(f"[A] TOTAL elapsed={elapsed:.0f}s")
    return dict(filas=filas,
                control={f"{k[0]}_{k[1]}": v for k, v in control.items()},
                elapsed_s=elapsed,
                grid=dict(amp_rugosidad=AMP_RUGOSIDAD_A, E_reserva=E_RESERVA_A, semillas=SEMILLAS_A))


def analizar_a(resultado):
    """Protocolo §5: curva de los 3 observables vs eps (media +- std entre semillas y
    E_reserva), para ambos brazos, y la lectura (energética/mecánica/mixta/no explicado)."""
    filas = [r for r in resultado["filas"] if r.get("ok")]
    control = [r for r in resultado["control"].values() if r.get("ok")]

    por_eps_con = {}
    for r in filas:
        eps = round(r["params"]["amp_rugosidad"], 8)
        por_eps_con.setdefault(eps, {"frac_masa_ligada": [], "frac_reserva_gastada_temprano": [],
                                      "frac_masa_en_mayor_cluster": [], "n_clusters_finales": []})
        por_eps_con[eps]["frac_masa_ligada"].append(r["frac_masa_ligada"])
        if r.get("frac_reserva_gastada_temprano") is not None:
            por_eps_con[eps]["frac_reserva_gastada_temprano"].append(r["frac_reserva_gastada_temprano"])
        por_eps_con[eps]["frac_masa_en_mayor_cluster"].append(r["frac_masa_en_mayor_cluster"])
        por_eps_con[eps]["n_clusters_finales"].append(r["n_clusters_finales"])

    por_eps_sin = {}
    for r in control:
        eps = round(r["params"]["amp_rugosidad"], 8)
        por_eps_sin.setdefault(eps, {"frac_masa_ligada": [], "frac_masa_en_mayor_cluster": [],
                                      "n_clusters_finales": []})
        por_eps_sin[eps]["frac_masa_ligada"].append(r["frac_masa_ligada"])
        por_eps_sin[eps]["frac_masa_en_mayor_cluster"].append(r["frac_masa_en_mayor_cluster"])
        por_eps_sin[eps]["n_clusters_finales"].append(r["n_clusters_finales"])

    def resumen(d):
        out = {}
        for eps, obs in sorted(d.items()):
            fila = {"eps": eps}
            for k, v in obs.items():
                if v:
                    fila[f"{k}_media"] = float(np.mean(v))
                    fila[f"{k}_std"] = float(np.std(v))
            out[eps] = fila
        return out

    curva_con = resumen(por_eps_con)
    curva_sin = resumen(por_eps_sin)

    eps_sorted = sorted(curva_con.keys())
    fm_con = np.array([curva_con[e]["frac_masa_ligada_media"] for e in eps_sorted])
    fm_sin = np.array([curva_sin[e]["frac_masa_ligada_media"] for e in eps_sorted if e in curva_sin])
    # correlación (spearman-like simple: signo de la pendiente log-log) del techo con eps
    corr_con = float(np.corrcoef(np.log(eps_sorted), fm_con)[0, 1]) if len(eps_sorted) > 2 else None
    corr_sin = (float(np.corrcoef(np.log(eps_sorted), fm_sin)[0, 1])
                if len(fm_sin) > 2 and len(fm_sin) == len(eps_sorted) else None)

    gasto_temprano = [curva_con[e].get("frac_reserva_gastada_temprano_media") for e in eps_sorted]
    gasto_valido = [g for g in gasto_temprano if g is not None]
    corr_gasto = (float(np.corrcoef(np.log(eps_sorted), gasto_temprano)[0, 1])
                  if len(gasto_valido) == len(eps_sorted) and len(eps_sorted) > 2 else None)

    frag_con = [curva_con[e]["frac_masa_en_mayor_cluster_media"] for e in eps_sorted]
    corr_frag = float(np.corrcoef(np.log(eps_sorted), frag_con)[0, 1]) if len(eps_sorted) > 2 else None

    techo_desaparece_sin_energia = None
    if corr_sin is not None and corr_con is not None:
        # "desaparece" = la pendiente negativa fuerte con energia deja de serlo sin energia
        techo_desaparece_sin_energia = (corr_con < -0.3) and (corr_sin > corr_con + 0.3)

    energetica = bool(techo_desaparece_sin_energia) and (corr_gasto is not None and corr_gasto > 0.3)
    mecanica = bool(corr_sin is not None and corr_sin < -0.3) and (corr_frag is not None and corr_frag < -0.3)
    if energetica and mecanica:
        lectura = "mixta"
    elif energetica:
        lectura = "energetica"
    elif mecanica:
        lectura = "mecanica"
    else:
        lectura = "no_explicado_por_estos_3_observables"

    return dict(
        curva_con_energia=curva_con, curva_sin_energia=curva_sin,
        corr_logeps_fracligada_con=corr_con, corr_logeps_fracligada_sin=corr_sin,
        corr_logeps_gastotemprano=corr_gasto, corr_logeps_fragmentacion=corr_frag,
        techo_desaparece_sin_energia=techo_desaparece_sin_energia,
        lectura=lectura,
        n_ok_con=len(filas), n_ok_sin=len(control),
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
        global AMP_RUGOSIDAD_A, E_RESERVA_A, SEMILLAS_A
        AMP_RUGOSIDAD_A = [1e-3, 0.5, 4.0, 10.0]
        E_RESERVA_A = [1e-2, 1.0, 1e2]
        SEMILLAS_A = [0, 1]
        nq, naq, ne, npos, pasos_basal = 60, 42, 20, 14, 40

    resultado = correr_experimento_a(nq=nq, naq=naq, ne=ne, npos=npos,
                                      pasos_basal=pasos_basal, log_fn=p)
    balance = analizar_a(resultado)
    resultado["analisis"] = balance
    resultado["log"] = log_lines
    p(f"[A] lectura = {balance['lectura']}")
    p(f"[A] corr(log eps, frac_ligada) con energia={balance['corr_logeps_fracligada_con']} "
      f"sin energia={balance['corr_logeps_fracligada_sin']}")
    p(f"[A] corr(log eps, gasto temprano)={balance['corr_logeps_gastotemprano']} "
      f"corr(log eps, fragmentacion)={balance['corr_logeps_fragmentacion']}")

    nombre = "cs074A_result_FULL.json" if full else "cs074A_result_smoke.json"
    out_json = OUT / nombre
    out_json.write_text(json.dumps(resultado, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    p(f"[archivo] {out_json}")


if __name__ == "__main__":
    main()
