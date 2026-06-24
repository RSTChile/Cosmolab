#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
campana_exaptacion.py — HARNESS de experimentos sistemáticos (Célula Madre)
================================================================================
QUIÉN SOY: la espina dorsal REPRODUCIBLE de la campaña. No decido nada "inteligente";
solo corro la célula madre validada (correr() de VST_CelulaMadre_WebLive) sobre un
manifiesto determinista de entradas (los .wav del proyecto + demos), guardo el CSV de
cada experimento y resumo cada corrida en métricas — con FOCO EN EXAPTACIÓN (qué
material la despierta y cuál no). Semillas fijas → mismas entradas dan mismos números.

POR QUÉ ASÍ: los agentes (workflow) NO deben reinventar la ejecución; ejecutan ESTE
harness por lotes (--rango) y luego analizan/verifican/sintetizan los resultados. La
ejecución es determinista y barata; el valor multi-agente está en el análisis.

USO
  python campana_exaptacion.py --manifest                 # imprime el work-list (JSON) y N
  python campana_exaptacion.py --rango 0:20 --out res/p0.jsonl   # corre [0,20) y guarda
  python campana_exaptacion.py --merge                    # une los .jsonl → resumen.csv/json
Salidas en experimentos/resultados/: <id>.csv por experimento + resumen.csv/json.
================================================================================
"""
from __future__ import annotations
import os, sys, json, glob, argparse
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
RES = os.path.join(AQUI, "resultados")
sys.path.insert(0, RAIZ)                       # para importar el laboratorio
import VST_CelulaMadre_WebLive as W            # motor validado + correr()

AUDIO_DIR = os.path.join(RAIZ, "audio_binaural")
SIM_S = 20.0                                   # segundos simulados por experimento (cap a la duración real)
TODO_ON = {n: True for n, *_ in W.ORG_UI}      # todos los organelos encendidos


def manifiesto() -> list:
    """Work-list DETERMINISTA: cada .wav del proyecto (binaural real, sus propios L/R) + 3 demos
    (mono, controles). Orden estable (sorted) para que los rangos sean reproducibles."""
    exps = []
    for p in sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav"))):
        nom = os.path.basename(p)
        exps.append({"id": "wav__" + os.path.splitext(nom)[0].replace(" ", "_")[:60],
                     "clase": "archivo", "spec": nom,
                     "cfg": {"fuente": "archivo", "left": nom, "right": None, "binaural": True}})
    for d in ["demo:tono", "demo:rosa", "demo:clicks"]:
        exps.append({"id": "demo__" + d.split(":")[1], "clase": "demo", "spec": d,
                     "cfg": {"fuente": "demo", "left": d, "binaural": False}})
    return exps


def _resumen(rows, meta, exp, error=None) -> dict:
    """Métricas de UNA corrida, con foco en exaptación + correlatos + controles (deriva/saturación)."""
    base = {"id": exp["id"], "clase": exp["clase"], "spec": exp["spec"]}
    if error is not None:
        base.update(status="error", error=str(error)[:200]); return base
    a = lambda c: np.array([r[c] for r in rows], dtype=float)
    n = len(rows)
    seg = lambda x, k=10: (round(x[:max(1, n // k)].mean(), 4), round(x[-max(1, n // k):].mean(), 4))
    xe = a("XE"); ex = a("exaptacion_activa"); idx = np.where(ex > 0)[0]
    oi_i, oi_f = seg(a("OI")); h_i, h_f = seg(a("H_homeostasis")); A_i, A_f = seg(a("A_sys_env"))
    base.update(
        status="ok", pasos=n, vida_s=round(rows[-1]["t"], 1),
        # --- EXAPTACIÓN (foco) ---
        XE_media=round(xe.mean(), 4), XE_max=round(xe.max(), 4), XE_fin=round(xe[-1], 4),
        XE_std=round(xe.std(), 4),
        exapt_pct=round(100 * (ex > 0).mean(), 1), exapt_n=int((ex > 0).sum()),
        t_primera_exapt=(round(rows[int(idx[0])]["t"], 1) if idx.size else None),
        Omega_op_max=round(a("Omega_op").max(), 4),
        # --- CORRELATOS de entrada (¿lateralidad? ¿energía? ¿riqueza temporal?) ---
        lat_real=bool(meta.get("lateralidad_real")),
        energia_L=round(a("energia_L").mean(), 5), energia_R=round(a("energia_R").mean(), 5),
        energia_std=round((a("energia_L") + a("energia_R")).std(), 5),   # variabilidad temporal del volumen
        balance=round(a("balance_LR").mean(), 4), coherencia=round(a("coherencia_biaural").mean(), 4),
        lateralidad=round(a("lateralidad").mean(), 4),
        gradiente_std=round(a("gradiente").std(), 4),                    # variabilidad del desajuste (sorpresa)
        # --- CONTROLES (deriva / saturación / genealogía LF) ---
        OI_media=round(a("OI").mean(), 4), OI_ini=oi_i, OI_fin=oi_f,
        H_ini=h_i, H_fin=h_f, A_ini=A_i, A_fin=A_f,
        R2_media=round(a("R2").mean(), 4), C_m_max=round(a("C_m").max(), 4),
        ritual_pct=round(100 * (a("ritual") > 0).mean(), 1),
        juego_pct=round(100 * (a("juego") > 0).mean(), 1),
        negacion_pct=round(100 * (a("negacion") > 0).mean(), 1),
        inv_ok_min=int(a("invariantes_ok").min()),
    )
    return base


def correr_uno(exp) -> dict:
    """Corre un experimento, guarda su CSV y devuelve el resumen (captura errores por-experimento)."""
    os.makedirs(RES, exist_ok=True)
    try:
        out = W.correr(exp["cfg"], dict(TODO_ON), sim_s=SIM_S)
        with open(os.path.join(RES, exp["id"] + ".csv"), "w", encoding="utf-8") as f:
            f.write(out["csv"])
        return _resumen(out["rows"], out["meta"], exp)
    except Exception as e:                                   # un archivo corrupto no debe tumbar el lote
        return _resumen(None, {}, exp, error=e)


def cmd_rango(a, b, out):
    exps = manifiesto()[a:b]
    res = []
    for i, exp in enumerate(exps):
        r = correr_uno(exp); res.append(r)
        print(f"  [{a+i:3d}] {r['status']:5s} {exp['id'][:48]:48s}"
              + (f" XE={r.get('XE_media')} exapt={r.get('exapt_pct')}% lat_real={r.get('lat_real')}"
                 if r["status"] == "ok" else f" ERROR {r.get('error','')[:60]}"))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = sum(1 for r in res if r["status"] == "ok")
    print(json.dumps({"rango": [a, b], "n": len(res), "ok": ok, "fail": len(res) - ok, "out": out}, ensure_ascii=False))


def cmd_merge():
    filas = []
    for p in sorted(glob.glob(os.path.join(RES, "resumen_*.jsonl"))):
        for ln in open(p, encoding="utf-8"):
            if ln.strip():
                filas.append(json.loads(ln))
    vistos = {}                                            # dedup por id (último gana), orden por id
    for r in filas:
        vistos[r["id"]] = r
    filas = [vistos[k] for k in sorted(vistos)]
    cols = ["id", "clase", "spec", "status", "XE_media", "XE_max", "XE_fin", "exapt_pct", "exapt_n",
            "t_primera_exapt", "Omega_op_max", "lat_real", "energia_L", "energia_R", "energia_std",
            "balance", "coherencia", "lateralidad", "gradiente_std", "OI_media", "OI_ini", "OI_fin",
            "H_ini", "H_fin", "A_ini", "A_fin", "R2_media", "C_m_max", "ritual_pct", "juego_pct",
            "negacion_pct", "inv_ok_min", "pasos", "vida_s"]
    with open(os.path.join(RES, "resumen.csv"), "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in filas:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    with open(os.path.join(RES, "resumen.json"), "w", encoding="utf-8") as f:
        json.dump(filas, f, ensure_ascii=False, indent=1)
    ok = sum(1 for r in filas if r.get("status") == "ok")
    print(json.dumps({"merged": len(filas), "ok": ok, "fail": len(filas) - ok,
                      "csv": os.path.join(RES, "resumen.csv")}, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description="Harness de campaña de experimentos (foco exaptación).")
    p.add_argument("--manifest", action="store_true", help="imprime el work-list (JSON) y termina")
    p.add_argument("--rango", help="A:B → corre los experimentos [A,B)")
    p.add_argument("--out", default=None, help="archivo .jsonl de salida del rango")
    p.add_argument("--merge", action="store_true", help="une los resumen_*.jsonl → resumen.csv/json")
    args = p.parse_args()
    if args.manifest:
        m = manifiesto(); print(json.dumps({"N": len(m), "experimentos": m}, ensure_ascii=False)); return
    if args.merge:
        cmd_merge(); return
    if args.rango:
        a, b = (int(x) for x in args.rango.split(":"))
        out = args.out or os.path.join(RES, f"resumen_{a:04d}_{b:04d}.jsonl")
        cmd_rango(a, b, out); return
    p.print_help()


if __name__ == "__main__":
    main()
