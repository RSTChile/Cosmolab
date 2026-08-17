"""
CS092 — agregacion: tabla OBSERVABLE x SUSTRATO con el veredicto "detecta / no detecta"
segun los MISMOS criterios que uso el arco original (pre-registrados en
CONTROL_POSITIVO_orden_global_CN264_CS.md §1.2, no modificados).
Salidas: cs092_control_positivo_tabla.csv  +  cs092_control_positivo.png
"""
from __future__ import annotations
import sys, json, csv
import numpy as np

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)
import cs092_control_positivo_orden_global as C91

U_P, U_CV, U_PICO = C91.UMBRAL_PENDIENTE, C91.UMBRAL_PI_CV, C91.UMBRAL_PICO
DIAM_MIN = C91.FACTOR_JB_PRIMA * C91.DIAM_SW_REF_2500      # J-B': >= 39 a N~2500


def _pend(Ns, vals):
    Ns = np.array(Ns, float); vals = np.array(vals, float)
    ok = np.isfinite(Ns) & np.isfinite(vals) & (vals > 0)
    if ok.sum() < 2:
        return float("nan")
    x = np.log(Ns[ok]); y = np.log(vals[ok])
    return float(np.polyfit(x, y, 1)[0])


def main():
    filas = json.load(open(f"{_HERE}/cs092_control_positivo_crudo.json"))
    subs = []
    for r in filas:
        if r["sustrato"] not in subs:
            subs.append(r["sustrato"])
    seeds = sorted(set(r["seed"] for r in filas))
    out = []
    for nom in subs:
        rs = [r for r in filas if r["sustrato"] == nom]
        orden = C91.SUSTRATOS[nom][2]
        Ns = sorted(set(r["N"] for r in rs))
        Nmax = Ns[-1]
        pM, pQ = [], []
        for s in seeds:
            rr = [r for r in rs if r["seed"] == s]
            if len(rr) < 2:
                continue
            rr.sort(key=lambda r: r["N"])
            pM.append(_pend([r["N"] for r in rr], [r["M_diam_gigante"] for r in rr]))
            pQ.append(_pend([r["N"] for r in rr], [r.get("Q_diam", float("nan")) for r in rr]))
        gr = [r for r in rs if r["N"] == Nmax]
        f = lambda k: float(np.nanmean([r.get(k, np.nan) for r in gr])) if gr else float("nan")
        d = dict(
            sustrato=nom, orden_global_conocido=orden, N_max=Nmax,
            grado_medio=round(f("grado_medio"), 2),
            # --- J-A ---
            M_pi_cv=round(f("M_pi_cv"), 3), Q_pi_cv=round(f("Q_pi_cv"), 3),
            # --- J-B ---
            M_pendiente=round(float(np.nanmean(pM)), 3) if pM else float("nan"),
            Q_pendiente=round(float(np.nanmean(pQ)), 3) if pQ else float("nan"),
            # --- J-B' ---
            M_diam=round(f("M_diam_gigante"), 1), M_diam_robusto=round(f("M_diam_robusto"), 1),
            Q_diam=round(f("Q_diam"), 2),
            # --- J-C ---
            M_n_ejes=round(f("M_n_ejes"), 2), M_pico=round(f("M_pico_medio"), 3),
            Q_n_ejes=round(f("Q_n_ejes"), 2), Q_pico=round(f("Q_pico_medio"), 3),
        )
        det = lambda b: "DETECTA" if b else "no detecta"
        nan = lambda v: (isinstance(v, float) and not np.isfinite(v))
        d["JA_M"] = "NO MEDIBLE" if nan(d["M_pi_cv"]) else det(d["M_pi_cv"] < U_CV)
        d["JA_Q"] = "NO MEDIBLE" if nan(d["Q_pi_cv"]) else det(d["Q_pi_cv"] < U_CV)
        d["JB_M"] = "NO MEDIBLE" if nan(d["M_pendiente"]) else det(d["M_pendiente"] > U_P)
        d["JB_Q"] = "NO MEDIBLE" if nan(d["Q_pendiente"]) else det(d["Q_pendiente"] > U_P)
        d["JBp_M"] = det(d["M_diam"] >= DIAM_MIN)
        d["JBp_Q"] = "NO MEDIBLE" if nan(d["Q_diam"]) else det(d["Q_diam"] >= DIAM_MIN)
        d["JC_M"] = det(d["M_pico"] > U_PICO and d["M_n_ejes"] >= 1)
        d["JC_Q"] = "NO MEDIBLE" if nan(d["Q_pico"]) else det(d["Q_pico"] > U_PICO and d["Q_n_ejes"] >= 1)
        out.append(d)

    campos = list(out[0].keys())
    with open(f"{_HERE}/cs092_control_positivo_tabla.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos); w.writeheader(); w.writerows(out)

    # ---- impresion ----
    print("=" * 132)
    print("TABLA OBSERVABLE x SUSTRATO — criterios pre-registrados: J-A pi_cv<0.5 | J-B pend>0.3 | "
          f"J-B' diam>={DIAM_MIN:.0f} | J-C pico>0.85 y n_ejes>=1")
    print("=" * 132)
    hdr = (f"{'sustrato':>15} {'orden':>6} | {'pi_cv M':>8} {'J-A M':>11} {'pi_cv Q':>8} {'J-A Q':>11} | "
           f"{'pend M':>7} {'J-B M':>11} {'pend Q':>7} {'J-B Q':>11}")
    print(hdr); print("-" * len(hdr))
    for d in out:
        print(f"{d['sustrato']:>15} {str(d['orden_global_conocido']):>6} | {d['M_pi_cv']:>8} {d['JA_M']:>11} "
              f"{d['Q_pi_cv']:>8} {d['JA_Q']:>11} | {d['M_pendiente']:>7} {d['JB_M']:>11} "
              f"{d['Q_pendiente']:>7} {d['JB_Q']:>11}")
    print()
    jbp = "J-B' "
    hdr2 = (f"{'sustrato':>15} {'orden':>6} | {'diam M':>8} {jbp+'M':>11} {'diam Q':>8} {jbp+'Q':>11} | "
            f"{'ejes M':>6} {'pico M':>7} {'J-C M':>11} {'ejes Q':>6} {'pico Q':>7} {'J-C Q':>11}")
    print(hdr2); print("-" * len(hdr2))
    for d in out:
        print(f"{d['sustrato']:>15} {str(d['orden_global_conocido']):>6} | {d['M_diam']:>8} {d['JBp_M']:>11} "
              f"{d['Q_diam']:>8} {d['JBp_Q']:>11} | {d['M_n_ejes']:>6} {d['M_pico']:>7} {d['JC_M']:>11} "
              f"{d['Q_n_ejes']:>6} {d['Q_pico']:>7} {d['JC_Q']:>11}")
    print("\ncsv -> cs092_control_positivo_tabla.csv")

    # ---- PNG ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        jueces = [("J-A pi_cv", "JA_M", "JA_Q"), ("J-B pendiente", "JB_M", "JB_Q"),
                  ("J-B' magnitud", "JBp_M", "JBp_Q"), ("J-C gap+picado", "JC_M", "JC_Q")]
        cols = []
        etiquetas = []
        for nom, kM, kQ in jueces:
            cols.append((nom + "\n(via M)", kM)); cols.append((nom + "\n(via Q)", kQ))
        M = np.zeros((len(out), len(cols)))
        for i, d in enumerate(out):
            for j, (_, k) in enumerate(cols):
                M[i, j] = {"DETECTA": 1.0, "no detecta": 0.0, "NO MEDIBLE": 0.5}[d[k]]
        fig, ax = plt.subplots(figsize=(13, 6.2))
        cmap = matplotlib.colors.ListedColormap(["#c0392b", "#7f8c8d", "#27ae60"])
        ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(cols))); ax.set_xticklabels([c[0] for c in cols], fontsize=8)
        ax.set_yticks(range(len(out)))
        ax.set_yticklabels([f"{d['sustrato']}  {'[ORDEN]' if d['orden_global_conocido'] else '[sin orden]'}"
                            for d in out], fontsize=9)
        for i in range(len(out)):
            for j in range(len(cols)):
                txt = {1.0: "detecta", 0.0: "no", 0.5: "no medible"}[M[i, j]]
                ax.text(j, i, txt, ha="center", va="center", fontsize=7, color="white")
        ax.set_title("CS092 — Control positivo: ¿los jueces del arco detectan orden global donde SÍ lo hay?\n"
                     "verde=detecta  rojo=no detecta  gris=no medible (NaN por construcción)", fontsize=10)
        fig.tight_layout()
        fig.savefig(f"{_HERE}/cs092_control_positivo.png", dpi=150)
        print("png  -> cs092_control_positivo.png")
    except Exception as e:
        print(f"(png omitido: {e})")


if __name__ == "__main__":
    main()
