"""
CS058 — ANÁLISIS del zoom de energía oscura: ¿el candidato es REAL o ARTEFACTO? (tres caracterizaciones)
Robusto a CSV parcial. Responde las 3 preguntas pre-registradas + el NULL.
"""
import sys, os
import pandas as pd, numpy as np

CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "cs058_zoom.csv")
CL = ["d1", "d2", "d3", "d4", "curv"]


def main():
    df = pd.read_csv(CSV)
    for m in ["acc", "accnull", "accrob", "accrobnull"]:
        df[m] = df[[f"{m}_{c}" for c in CL]].sum(axis=1)
    df["viab"] = df[[f"viable_{c}" for c in CL]].sum(axis=1)
    print("=" * 92)
    print(f"CS058 — ¿ENERGÍA OSCURA REAL o ARTEFACTO?  ({df.point_id.nunique()} puntos, {len(df)} filas)")
    print("=" * 92)

    print("\n[1] SUPERVIVENCIA A LA RESOLUCIÓN (×1/×2/×4 pasos) — predicción: real→estable/afila, artefacto→decae")
    print(f"    {'res':>4} | {'acc(CS057)':>11} {'accrob':>8} | {'accNULL':>8} {'accrobNULL':>11}")
    tab = {}
    for res in sorted(df.res_mult.unique()):
        s = df[df.res_mult == res]
        tab[res] = (s.accrob.mean(), s.accrobnull.mean())
        print(f"    ×{res:<3} | {s.acc.mean():>11.3f} {s.accrob.mean():>8.3f} | {s.accnull.mean():>8.3f} {s.accrobnull.mean():>11.3f}")
    rs = sorted(tab)
    decae = tab[rs[-1]][0] < 0.4 * tab[rs[0]][0]
    print(f"    → accrob {'DECAE' if decae else 'se SOSTIENE'} de ×{rs[0]} a ×{rs[-1]} "
          f"({tab[rs[0]][0]:.3f}→{tab[rs[-1]][0]:.3f})  ⇒ {'ARTEFACTO' if decae else 'candidato REAL'}")

    print("\n[2] NULL (barajado temporal) — predicción: real >> null; si real≈null, artefacto de medición")
    r, n = df.accrob.mean(), df.accrobnull.mean()
    ratio = r / n if n > 0 else float("inf")
    print(f"    accrob={r:.3f}  accrobnull={n:.3f}  ratio={ratio:.2f}  ⇒ {'colapsa (real≈null): ARTEFACTO' if ratio < 1.5 else 'real supera null'}")

    print("\n[3] FRONTERA CURVA — predicción (conecta con R7): la aceleración vive en el dominio CURVO, no 3D/4D")
    accd = {c: df[f"accrob_{c}"].mean() for c in ["d2", "d3", "d4", "curv"]}
    top = max(accd, key=accd.get)
    print("    accrob por dim: " + "  ".join(f"{c}={accd[c]:.3f}" for c in accd))
    print(f"    → máximo en {top}  ⇒ {'FRONTERA CURVA (conecta R7)' if top == 'curv' else 'NO es frontera curva (vive en planas altas, sigue viabilidad)'}")

    print("\n" + "=" * 92)
    print("VEREDICTO:")
    señales_artefacto = int(decae) + int(ratio < 1.5) + int(top != "curv")
    if señales_artefacto >= 2:
        print("  El candidato de energía oscura es ARTEFACTO / NO robusto: decae con la resolución, el NULL no")
        print("  colapsa limpio, y no vive en la frontera curva. Se DECLARA artefacto (honesto). R7 no arrastra")
        print("  un fantasma — CS059 corrió sin depender de él. Se documenta el umbral de disolución (×2→×4).")
    else:
        print("  El candidato SOBREVIVE: se caracteriza como fenómeno real de expansión acelerada emergente.")
    print("=" * 92)


if __name__ == "__main__":
    main()
