"""
cs090_fase6_o3b_analizar.py — FASE VI O3-B: lee los dumps de Phantom de los pares
(original, gemelo reconfigurado) y hace la comparación PAREADA.

Reusa `cs090_fase5b_analizar.analizar_carpeta` TAL CUAL (sólo import) — la misma extracción de métricas
que toda la línea Fase V-B: fracción de masa en sumideros (principal), κ_V agregado (secundario), nº de
sumideros, t del primer sumidero, masa acretada total. No se reimplementa nada de eso acá.

VERIFICACIÓN CRUZADA OBLIGATORIA (lección del bug de colisión de nombres de Fase V-B): antes de armar
ningún par se comprueba, contra el `meta_regla.json` de CADA carpeta, que:
  - la carpeta `<rule_id>_orig` diga variante="orig" y la `<rule_id>_rewire` diga variante="rewire",
  - las dos digan el MISMO rule_id y el MISMO seed,
  - las dos tengan el mismo nº de aristas (la reconfiguración conserva aristas por construcción),
  - la carpeta declarada dentro del meta coincida con la carpeta donde está el meta.
Un par que falle cualquiera de esos chequeos NO entra en la estadística; se reporta aparte.

Estadística pareada (los mismos dos tests que el resto de la línea):
  - test de signos (binomial exacto, dos colas, empates descartados),
  - Wilcoxon de rangos con signo (dos colas).
Se aplican a la diferencia (original − gemelo) en fracción de masa acretada y en κ_V agregado.

Escribe `cs090_fase6_o3b_phantom_pares.csv` (datos crudos, una fila por par) y
`cs090_fase6_o3b_estadistica.csv`. No declara cierre ni veredicto: sólo números.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, HERE)

from cs090_fase5b_analizar import analizar_carpeta          # sólo import, script congelado

BASE = Path("/Users/alexis/phantom_cs073/bateria_fase6_o3b_rewiring")
RUTA_PARES = f"{HERE}/cs090_fase6_o3b_phantom_pares.csv"
RUTA_CRUDO = f"{HERE}/cs090_fase6_o3b_phantom_crudo.csv"
RUTA_ESTAD = f"{HERE}/cs090_fase6_o3b_estadistica.csv"
RUTA_ESTRUCTURA = f"{HERE}/cs090_fase6_o3b_estructura.csv"
RUTA_TOTAL_40 = f"{HERE}/cs090_fase5b_TOTAL_40pares.csv"


def test_de_signos(dif):
    """Binomial exacto de dos colas sobre cuántas diferencias son positivas (empates descartados)."""
    d = np.asarray([x for x in dif if x != 0.0], dtype=float)
    n, k = len(d), int((d > 0).sum())
    if n == 0:
        return dict(n=0, n_positivos=0, p=float("nan"))
    p = float(stats.binomtest(k, n, 0.5, alternative="two-sided").pvalue)
    return dict(n=n, n_positivos=k, p=p)


def main():
    metas, problemas = {}, []
    for carpeta in sorted(c for c in BASE.iterdir() if c.is_dir()):
        mp = carpeta / "meta_regla.json"
        if not mp.exists():
            problemas.append(f"{carpeta.name}: sin meta_regla.json")
            continue
        m = json.loads(mp.read_text())
        if Path(m.get("carpeta", "")).name != carpeta.name:
            problemas.append(f"{carpeta.name}: el meta declara carpeta={m.get('carpeta')}")
            continue
        metas[carpeta.name] = (carpeta, m)

    # ---- armar pares con verificación cruzada ----
    pares = []
    for nombre, (carpeta, m) in sorted(metas.items()):
        if m.get("variante") != "orig":
            continue
        rid = m["rule_id"]
        nombre_rw = f"{rid}_rewire"
        if nombre != f"{rid}_orig":
            problemas.append(f"{nombre}: el meta dice rule_id={rid} (nombre de carpeta no concuerda)")
            continue
        if nombre_rw not in metas:
            problemas.append(f"{rid}: falta la carpeta del gemelo ({nombre_rw})")
            continue
        carpeta_rw, m_rw = metas[nombre_rw]
        chequeos = {
            "variante_rewire": m_rw.get("variante") == "rewire",
            "mismo_rule_id": m_rw.get("rule_id") == rid,
            "mismo_seed": m_rw.get("seed") == m.get("seed"),
            "mismas_aristas": m_rw.get("n_aristas_grafo_final") == m.get("n_aristas_grafo_final"),
            "misma_seed_layout": m_rw.get("seed_layout") == m.get("seed_layout"),
            "solape_declarado": m_rw.get("solapamiento_aristas") is not None,
        }
        if not all(chequeos.values()):
            problemas.append(f"{rid}: verificación cruzada FALLA -> {chequeos}")
            continue
        pares.append((rid, carpeta, m, carpeta_rw, m_rw))

    print(f"[o3b] {len(pares)} pares pasan la verificación cruzada; {len(problemas)} problemas")
    for pr in problemas:
        print(f"   !! {pr}")

    # ---- métricas de Phantom ----
    historico = {}
    with open(RUTA_TOTAL_40) as f:
        for r in csv.DictReader(f):
            if r["rol"] == "III":
                historico[(r["rule_id"], int(r["seed"]))] = r

    estructura = {}
    for suf in ("", "_shard0", "_shard1", "_shard2", "_shard3"):
        p = Path(RUTA_ESTRUCTURA.replace(".csv", f"{suf}.csv"))
        if p.exists():
            for r in csv.DictReader(open(p)):
                estructura[r["rule_id"]] = r

    filas_crudo, filas_par = [], []
    for rid, c_o, m_o, c_r, m_r in pares:
        f_o = analizar_carpeta(c_o)
        f_r = analizar_carpeta(c_r)
        f_o["variante"], f_r["variante"] = "orig", "rewire"
        f_o["rule_id"] = f_r["rule_id"] = rid
        filas_crudo.extend([f_o, f_r])

        est = estructura.get(rid, {})
        hist = historico.get((rid, m_o["seed"]), {})
        frac_hist = float(hist["fraccion_masa_en_sumideros"]) if hist else None
        kv_hist = float(hist["kappa_v_agregado"]) if hist and hist["kappa_v_agregado"] else None

        fila = dict(
            rule_id=rid, seed=m_o["seed"], lote=m_o.get("lote"), K=m_o.get("K"), kcap=m_o.get("kcap"),
            n_aristas=m_o.get("n_aristas_grafo_final"),
            solapamiento_aristas=m_r.get("solapamiento_aristas"),
            clustering_orig=m_o.get("clustering_local"), clustering_rewire=m_r.get("clustering_local"),
            transitividad_orig=m_o.get("transitividad"), transitividad_rewire=m_r.get("transitividad"),
            pendiente_corr_orig=est.get("pendiente_corr_orig"),
            pendiente_corr_rewire=est.get("pendiente_corr_rewire"),
            frac_masa_orig=f_o["fraccion_masa_en_sumideros"],
            frac_masa_rewire=f_r["fraccion_masa_en_sumideros"],
            d_frac_masa=(f_o["fraccion_masa_en_sumideros"] - f_r["fraccion_masa_en_sumideros"]),
            kappa_v_orig=f_o["kappa_v_agregado"], kappa_v_rewire=f_r["kappa_v_agregado"],
            d_kappa_v=((f_o["kappa_v_agregado"] - f_r["kappa_v_agregado"])
                       if (f_o["kappa_v_agregado"] is not None and f_r["kappa_v_agregado"] is not None)
                       else None),
            n_sumideros_orig=f_o["n_sumideros"], n_sumideros_rewire=f_r["n_sumideros"],
            t_primer_sumidero_orig=f_o["t_primer_sumidero"],
            t_primer_sumidero_rewire=f_r["t_primer_sumidero"],
            masa_acretada_orig=f_o["masa_acretada_total"],
            masa_acretada_rewire=f_r["masa_acretada_total"],
            dump_final_orig=f_o.get("n_dump_final"), dump_final_rewire=f_r.get("n_dump_final"),
            # --- réplica: la mitad "original" debería reproducir su propia corrida de Fase V-B ---
            frac_masa_fase5b=frac_hist, kappa_v_fase5b=kv_hist,
            replica_frac_ok=(abs(f_o["fraccion_masa_en_sumideros"] - frac_hist) < 1e-12
                             if frac_hist is not None else None),
            delta_replica_frac=(f_o["fraccion_masa_en_sumideros"] - frac_hist
                                if frac_hist is not None else None),
        )
        filas_par.append(fila)
        print(f"   {rid:<26} frac {fila['frac_masa_orig']:.4f} vs {fila['frac_masa_rewire']:.4f} "
              f"(d={fila['d_frac_masa']:+.4f})  κ_V {fila['kappa_v_orig']} vs {fila['kappa_v_rewire']} "
              f" réplica_frac_ok={fila['replica_frac_ok']}", flush=True)

    for ruta, filas in ((RUTA_CRUDO, filas_crudo), (RUTA_PARES, filas_par)):
        campos = []
        for f in filas:
            for c in f:
                if c not in campos:
                    campos.append(c)
        with open(ruta, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=campos)
            w.writeheader()
            w.writerows(filas)
        print(f"[csv] {ruta.split('/')[-1]} ({len(filas)} filas)")

    # ---------------- estadística pareada ----------------
    resumen = []
    for nombre, clave, col_o, col_r in (
            ("fraccion_masa_en_sumideros", "d_frac_masa", "frac_masa_orig", "frac_masa_rewire"),
            ("kappa_v_agregado", "d_kappa_v", "kappa_v_orig", "kappa_v_rewire")):
        usables = [f for f in filas_par if f[clave] is not None]
        d = np.asarray([f[clave] for f in usables], dtype=float)
        sg = test_de_signos(d)
        try:
            w_stat, w_p = stats.wilcoxon(d, alternative="two-sided")
        except Exception as e:
            w_stat, w_p = float("nan"), float("nan")
            print(f"   (wilcoxon no aplicable a {nombre}: {e})")
        resumen.append(dict(
            observable=nombre, n_pares=len(d),
            media_orig=float(np.mean([f[col_o] for f in usables])),
            media_rewire=float(np.mean([f[col_r] for f in usables])),
            media_dif=float(np.mean(d)), mediana_dif=float(np.median(d)),
            n_orig_gana=sg["n_positivos"], n_sin_empate=sg["n"], p_signos=sg["p"],
            wilcoxon_W=float(w_stat), p_wilcoxon=float(w_p),
        ))
        print(f"\n[{nombre}] n={len(d)}  media orig={resumen[-1]['media_orig']:.5f} "
              f"vs rewire={resumen[-1]['media_rewire']:.5f}  media dif={np.mean(d):+.5f} "
              f"mediana dif={np.median(d):+.5f}")
        print(f"   test de signos: original gana {sg['n_positivos']}/{sg['n']}  p={sg['p']:.4f}")
        print(f"   Wilcoxon: W={w_stat}  p={w_p:.4f}")

    with open(RUTA_ESTAD, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(resumen[0].keys()))
        w.writeheader()
        w.writerows(resumen)
    print(f"[csv] {RUTA_ESTAD.split('/')[-1]}")
    return filas_par, resumen


if __name__ == "__main__":
    main()
