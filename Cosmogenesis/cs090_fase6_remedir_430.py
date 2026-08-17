"""
cs090_fase6_remedir_430.py — PASO 2 de la adopción del diámetro corregido (sólo lectura + cómputo).
====================================================================================================

QUÉ HACE
--------
Vuelve a medir, con la medición CORREGIDA de `cs090_diam_corregido.py` (diámetro sobre la componente
gigante), las 430 reglas A2-B0-C2 ya clasificadas que están consolidadas en
`cs090_fase6_outliers_430_todas.csv` (que a su vez viene de los 4 CSV de origen:
cs090_fase5_profundizar_a2b0c2_resumen.csv, cs090_fase5b_candidatas_v2/v3/v4.csv).

Para cada regla:
  1. Reconstruye el grafo con la MISMA cadena determinista de `cs090_fase5_motor.correr_regla_coarse`
     (mismo `p` regenerado desde el `seed`, mismo rng `seed*5000+2000`, `construir_A2` + `dinamica_B0`
     + `medir`, cajas `cs080.cajas_bfs` con rng `seed*7000+b*31`, NULL_topo ER con `seed*6000+...`).
  2. Mide el diámetro de las DOS maneras a la vez (vieja y corregida), en REAL y en los 3 NULL_topo.
  3. Aplica `cs090_fase5_clasificador.clasificar_regla` a los dos juegos de filas — LOS MISMOS
     UMBRALES OFICIALES, sin cambiar ni uno (Clase I / II 0.35-0.45 / III >0.7 o z sostenido / IV +
     holonomía ≥5x / "intermedio" si no encaja).
  4. VERIFICACIÓN DURA: la pendiente y el z_agg reconstruidos con la medición VIEJA tienen que
     coincidir con los del CSV histórico hasta 1e-9. Si no coinciden, la reconstrucción no es fiel y
     el resto no valdría nada; se cuenta y se reporta.

SALIDA
------
`cs090_fase6_remedicion_430.csv` — una fila por regla con: pendiente vieja/corregida, z_agg
vieja/corregida, clase vieja/corregida, si cambió de clase, y el diagnóstico de descarrilamiento
(tamaño de la componente que midió la versión vieja vs la gigante, por escala).

No corre Phantom. No toca ningún script congelado. No declara veredicto.
"""
from __future__ import annotations
import csv
import sys
import time

import numpy as np

HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, HERE)

import cs090_fase5_generador as GEN
import cs090_diam_corregido as DC
from cs090_fase5_clasificador import clasificar_regla

ENTRADA = f"{HERE}/cs090_fase6_outliers_430_todas.csv"
SALIDA = f"{HERE}/cs090_fase6_remedicion_430.csv"


def main(limite=None):
    with open(ENTRADA) as f:
        base = list(csv.DictReader(f))
    if limite:
        base = base[:limite]
    print(f"[remedir] {len(base)} reglas desde {ENTRADA.split('/')[-1]}", flush=True)

    filas_out, t0 = [], time.time()
    n_mismatch = 0
    for k, r in enumerate(base):
        seed = int(r["seed"])
        p = GEN.generar_regla("A2", "B0", "C2", idx=0, seed=seed)
        f_orig, f_corr, diag = DC.correr_regla_coarse_doble(p)
        c_orig = clasificar_regla(f_orig)
        c_corr = clasificar_regla(f_corr)

        # --- verificación dura contra el CSV histórico ---
        d_pend = abs(c_orig["pendiente_real"] - float(r["pendiente"]))
        d_z = abs(c_orig["z_agg"] - float(r["z_agg"]))
        fiel = (d_pend < 1e-9) and (d_z < 1e-9) and (c_orig["clase"] == r["clase"])
        if not fiel:
            n_mismatch += 1
            print(f"   !! reconstrucción NO fiel: {r['rule_id']} seed={seed} "
                  f"d_pend={d_pend:.2e} d_z={d_z:.2e} clase_csv={r['clase']} clase_recon={c_orig['clase']}",
                  flush=True)

        filas_out.append(dict(
            rule_id=r["rule_id"], seed=seed, fuente=r.get("fuente", ""),
            K=p["K"], J=p["J"], noise=p["noise"], meandeg=p["meandeg"], kcap=p["kcap"],
            clase_csv=r["clase"], pendiente_csv=float(r["pendiente"]), z_agg_csv=float(r["z_agg"]),
            reconstruccion_fiel=fiel,
            pendiente_vieja=c_orig["pendiente_real"], pendiente_corregida=c_corr["pendiente_real"],
            z_agg_vieja=c_orig["z_agg"], z_agg_corregida=c_corr["z_agg"],
            z_sost_vieja=c_orig["z_sostenido"], z_sost_corregida=c_corr["z_sostenido"],
            holon_ratio=c_orig["holon_ratio"],
            clase_vieja=c_orig["clase"], clase_corregida=c_corr["clase"],
            cambia_clase=(c_orig["clase"] != c_corr["clase"]),
            diam_viejo=[d["diam_real"] for d in f_orig],
            diam_corregido=[d["diam_real"] for d in f_corr],
            diam_null_viejo=[round(d["diam_null_topo"], 3) for d in f_orig],
            diam_null_corregido=[round(d["diam_null_topo"], 3) for d in f_corr],
            n_cajas=[d["N"] for d in f_orig],
            tam_comp_medida=[d.get("tam_comp_medida") for d in diag],
            tam_gigante=[d.get("tam_gigante") for d in diag],
            descarrila_b1=bool(diag[0].get("descarrila")),
            escalas_descarriladas=sum(1 for d in diag if d.get("descarrila")),
        ))
        if (k + 1) % 25 == 0:
            el = time.time() - t0
            print(f"   ...{k+1}/{len(base)}  ({el:.0f}s, ETA {el/(k+1)*(len(base)-k-1):.0f}s)", flush=True)

    campos = list(filas_out[0].keys())
    with open(SALIDA, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        for d in filas_out:
            w.writerow(d)
    print(f"\n[csv] {SALIDA.split('/')[-1]} ({len(filas_out)} filas)")
    print(f"[verificación] reconstrucciones NO fieles: {n_mismatch}/{len(filas_out)}")

    # ---------------- resumen inmediato ----------------
    import collections
    cv = collections.Counter(d["clase_vieja"] for d in filas_out)
    cc = collections.Counter(d["clase_corregida"] for d in filas_out)
    print(f"\n[clases] vieja:     {dict(cv)}")
    print(f"[clases] corregida: {dict(cc)}")
    cam = [d for d in filas_out if d["cambia_clase"]]
    print(f"\n[cambian de clase] {len(cam)}/{len(filas_out)} ({100.0*len(cam)/len(filas_out):.1f}%)")
    trans = collections.Counter((d["clase_vieja"], d["clase_corregida"]) for d in cam)
    for (a, b), n in trans.most_common():
        print(f"   {a}  ->  {b}   : {n}")
    print(f"\n[descarrilan en b=1] {sum(1 for d in filas_out if d['descarrila_b1'])}/{len(filas_out)}")
    neg = [d for d in filas_out if d["pendiente_vieja"] < 0]
    print(f"[pendiente vieja < 0] {len(neg)}   de ellas cambian de clase: "
          f"{sum(1 for d in neg if d['cambia_clase'])}")
    nonneg_cam = [d for d in filas_out if d["pendiente_vieja"] >= 0 and d["cambia_clase"]]
    print(f"[pendiente vieja >= 0 que IGUAL cambian de clase] {len(nonneg_cam)}")
    for d in nonneg_cam[:20]:
        print(f"   {d['rule_id']:<26} pend {d['pendiente_vieja']:+.4f} -> {d['pendiente_corregida']:+.4f} "
              f"| z {d['z_agg_vieja']:.2f} -> {d['z_agg_corregida']:.2f} "
              f"| {d['clase_vieja']} -> {d['clase_corregida']}")


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(lim)
