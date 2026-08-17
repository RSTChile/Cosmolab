"""
cs090_fase6_outliers_paso2_distribucion.py -- FASE VI, investigacion de los "casos raros" (PASO 2).

QUIEN SOY: script de SOLA LECTURA. Junta los 4 CSV de origen donde viven TODAS las reglas A2-B0-C2 ya
generadas y clasificadas por el motor de Fase V-A (430 filas en total, de las cuales solo 80 llegaron a
correr en Phantom) y responde una pregunta barata antes de gastar un segundo de computo:

    los 3 puntos de pendiente muy negativa que aparecieron en FASE6_reanalisis_azar_continuo_CS.md
    (A2-B0-C2-batch3-r100, A2-B0-C2-batch4-r51, A2-B0-C2-batch3-r143) -- ¿son 3 rarezas entre 430,
    o hay muchas mas y simplemente solo 3 cayeron dentro de la muestra de 80 que paso por Phantom?

Que hace, en orden:
  1. Combina los 4 CSV (misma seleccion de filas que uso cs090_fase6_observable_continuo.py: del CSV
     "profundizar" solo la seccion origen=='nueva_profundizar', que es la unica que trae `seed`) y
     verifica que los `seed` no colisionen -- misma salvaguarda anti-bug-de-nombres de la linea.
  2. Describe la distribucion completa de `pendiente` (percentiles, histograma) y busca huecos: ordena
     las pendientes y mide la separacion entre valores consecutivos, para elegir el corte de "muy
     negativa" MIRANDO LOS DATOS en vez de poniendolo a dedo.
  3. Marca cuales de las reglas por debajo del corte YA pasaron por Phantom (cruzando contra
     cs090_fase5b_TOTAL_40pares.csv por `seed`) y cuales NO -- las que no son las candidatas del Paso 3.
  4. Compara los parametros (K, J, noise, meandeg, kcap, sim_thr_frac) del grupo de pendiente muy
     negativa contra el resto de las 430, para ver si comparten algo obvio.

Salidas: cs090_fase6_outliers_430_todas.csv (las 430 con su fuente y si paso por Phantom),
cs090_fase6_outliers_candidatas.csv (las de pendiente muy negativa sin Phantom),
cs090_fase6_outliers_histograma.png. No modifica ningun script ni CSV existente. No corre Phantom.
No declara cierre ni veredicto -- reporta numeros.
"""
from __future__ import annotations
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"

FUENTES = [
    ("cs090_fase5_profundizar_a2b0c2_resumen.csv", "nueva_profundizar"),  # 2do elem = filtro de 'origen'
    ("cs090_fase5b_candidatas_v2.csv", None),
    ("cs090_fase5b_candidatas_v3.csv", None),
    ("cs090_fase5b_candidatas_v4.csv", None),
]
RUTA_80 = f"{HERE}/cs090_fase5b_TOTAL_40pares.csv"

# Los 3 outliers reportados en FASE6_reanalisis_azar_continuo_CS.md 3.2 -- se usan solo para verificar
# que este script los vuelve a encontrar (no para definir el corte, que sale de los datos).
OUTLIERS_CONOCIDOS = {"A2-B0-C2-batch3-r100", "A2-B0-C2-batch4-r51", "A2-B0-C2-batch3-r143"}

NUM = ("pendiente", "z_agg", "holon_ratio", "J", "noise", "meandeg", "sim_thr_frac")
INT = ("seed", "K", "kcap")


def cargar_todas():
    filas = []
    for archivo, filtro_origen in FUENTES:
        with open(f"{HERE}/{archivo}") as f:
            for row in csv.DictReader(f):
                if filtro_origen is not None and row.get("origen") != filtro_origen:
                    continue
                d = dict(rule_id=row["rule_id"], clase=row["clase"], fuente=archivo)
                for c in NUM:
                    d[c] = float(row[c]) if row.get(c) not in (None, "") else float("nan")
                for c in INT:
                    d[c] = int(row[c]) if row.get(c) not in (None, "") else -1
                filas.append(d)
    seeds = [f["seed"] for f in filas]
    assert len(set(seeds)) == len(seeds), "COLISION de seed entre los CSV de origen -- no se sigue a ciegas"
    return filas


def seeds_ya_en_phantom():
    s = set()
    with open(RUTA_80) as f:
        for row in csv.DictReader(f):
            s.add(int(row["seed"]))
    return s


def main():
    filas = cargar_todas()
    ph = seeds_ya_en_phantom()
    for d in filas:
        d["paso_phantom"] = d["seed"] in ph
    n = len(filas)
    pend = np.array([d["pendiente"] for d in filas])
    print(f"[carga] {n} reglas clasificadas, {len(ph)} seeds distintos con corrida de Phantom "
          f"({sum(d['paso_phantom'] for d in filas)} de las {n} filas)")

    # ---------------- distribucion ----------------
    qs = [0, 0.5, 1, 2.5, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    print("\n[distribucion de pendiente sobre las %d]" % n)
    for q in qs:
        print(f"   p{q:<5} = {np.percentile(pend, q):+.4f}")
    print(f"   media={pend.mean():+.4f} sd={pend.std(ddof=1):.4f}")

    # huecos entre valores consecutivos -> el corte sale de mirar los datos
    orden = np.sort(pend)
    huecos = np.diff(orden)
    idx = np.argsort(huecos)[::-1][:8]
    print("\n[los 8 huecos mas grandes entre pendientes consecutivas (todo el rango)]")
    for i in sorted(idx, key=lambda k: orden[k]):
        print(f"   hueco {huecos[i]:.4f} entre {orden[i]:+.4f} y {orden[i+1]:+.4f}")

    # ---------------- cortes candidatos ----------------
    print("\n[cuantas reglas caen bajo distintos cortes]")
    for corte in (0.0, -0.1, -0.2, -0.3, -0.4, -0.5):
        sel = [d for d in filas if d["pendiente"] < corte]
        sinph = [d for d in sel if not d["paso_phantom"]]
        print(f"   pendiente < {corte:+.2f}: {len(sel):3d} reglas  "
              f"({len(sel)-len(sinph)} ya en Phantom, {len(sinph)} SIN Phantom)")

    # ---------------- histograma ----------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    ax.hist(pend, bins=60, color="#4477aa", edgecolor="white", linewidth=0.4)
    ax.axvline(0.7, color="#cc3311", ls="--", lw=1.2, label="umbral de clase 0.7")
    ax.axvline(0.0, color="#999999", ls=":", lw=1.2, label="pendiente = 0")
    ax.set_xlabel("pendiente log(diam) vs log(N_cajas)")
    ax.set_ylabel("nº de reglas")
    ax.set_title(f"Distribucion de pendiente — las {n} reglas A2-B0-C2 ya clasificadas")
    ax.legend(fontsize=8)

    ax = axes[1]
    col = {"I": "#4477aa", "III": "#ee7733", "II": "#009988"}
    for cl in sorted(set(d["clase"] for d in filas)):
        xs = [d["pendiente"] for d in filas if d["clase"] == cl]
        ys = [1 if d["paso_phantom"] else 0 for d in filas if d["clase"] == cl]
        ys = np.array(ys, float) + np.random.default_rng(7).normal(0, 0.06, len(ys))
        ax.scatter(xs, ys, s=16, alpha=0.6, label=f"clase {cl} (n={len(xs)})",
                   color=col.get(cl, "#777777"))
    ax.set_yticks([0, 1]); ax.set_yticklabels(["sin Phantom", "corrio Phantom"])
    ax.axvline(0.0, color="#999999", ls=":", lw=1.2)
    ax.set_xlabel("pendiente")
    ax.set_title("Quien de las 430 llego a Phantom")
    ax.legend(fontsize=8, loc="center left")
    fig.tight_layout()
    fig.savefig(f"{HERE}/cs090_fase6_outliers_histograma.png", dpi=130)
    print(f"\n[png] {HERE}/cs090_fase6_outliers_histograma.png")

    # ---------------- grupo negativo: parametros ----------------
    CORTE = -0.3   # se justifica en el informe con la salida de arriba
    grupo = [d for d in filas if d["pendiente"] < CORTE]
    resto = [d for d in filas if d["pendiente"] >= CORTE]
    print(f"\n[grupo pendiente < {CORTE}] n={len(grupo)}  (resto n={len(resto)})")
    print(f"   verificacion: los 3 outliers ya conocidos estan en el grupo? "
          f"{OUTLIERS_CONOCIDOS <= set(d['rule_id'] for d in grupo)}")
    print("   clases del grupo:", {c: sum(1 for d in grupo if d['clase'] == c)
                                    for c in sorted(set(d['clase'] for d in grupo))})
    print("\n   parametro        grupo (media±sd, min..max)          resto (media±sd, min..max)")
    for c in ("K", "J", "noise", "meandeg", "kcap", "sim_thr_frac", "z_agg", "holon_ratio"):
        a = np.array([d[c] for d in grupo], float)
        b = np.array([d[c] for d in resto], float)
        print(f"   {c:<14} {np.nanmean(a):8.3f}±{np.nanstd(a,ddof=1) if len(a)>1 else 0:6.3f} "
              f"[{np.nanmin(a):7.3f}..{np.nanmax(a):7.3f}]   "
              f"{np.nanmean(b):8.3f}±{np.nanstd(b,ddof=1):6.3f} [{np.nanmin(b):7.3f}..{np.nanmax(b):7.3f}]")

    print(f"\n[las {len(grupo)} reglas del grupo]")
    for d in sorted(grupo, key=lambda x: x["pendiente"]):
        print(f"   {d['rule_id']:<26} pend={d['pendiente']:+.4f} clase={d['clase']:<3} seed={d['seed']} "
              f"K={d['K']} J={d['J']:.3f} noise={d['noise']:.3f} meandeg={d['meandeg']:.2f} "
              f"kcap={d['kcap']} z_agg={d['z_agg']:.2f} phantom={d['paso_phantom']} [{d['fuente']}]")

    # ---------------- CSVs ----------------
    campos = ["rule_id", "clase", "seed", "K", "J", "noise", "meandeg", "kcap", "sim_thr_frac",
              "pendiente", "z_agg", "holon_ratio", "fuente", "paso_phantom"]
    with open(f"{HERE}/cs090_fase6_outliers_430_todas.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos); w.writeheader()
        for d in sorted(filas, key=lambda x: x["pendiente"]):
            w.writerow({k: d[k] for k in campos})
    cand = [d for d in grupo if not d["paso_phantom"]]
    with open(f"{HERE}/cs090_fase6_outliers_candidatas.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos); w.writeheader()
        for d in sorted(cand, key=lambda x: x["pendiente"]):
            w.writerow({k: d[k] for k in campos})
    print(f"\n[csv] cs090_fase6_outliers_430_todas.csv ({len(filas)} filas)")
    print(f"[csv] cs090_fase6_outliers_candidatas.csv ({len(cand)} candidatas SIN Phantom)")


if __name__ == "__main__":
    main()
