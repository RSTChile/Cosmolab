"""
CS057 — ANÁLISIS DEL PAISAJE: lee cs057_paisaje.csv y responde las preguntas del diseño.
========================================================================================
Robusto a CSV PARCIAL (se puede correr mientras la tanda sigue). Responde CIEGO (nada busca 3D):
  1. ¿QUÉ combinaciones estabilizan un universo VIABLE (estable + expande), y de qué dimensión?
  2. sync vs async — la FALSACIÓN del "es un proceso": ¿hay universos viables en sync que NO en async?
  3. El PUNTO FÍSICO real (phys=1) y su vecindad densa (phys=2): ¿cae entre los viables? ¿de qué dim?
  4. SECTOR OSCURO emergente: ¿hay región con expansión que ACELERA sola (candidato energía oscura)?
  5. ¿Qué pesos de fuerza separan lo viable de lo no-viable (regiones del hipercubo)?
Uso: python3.11 cs057_analisis.py [ruta_csv]   → imprime el informe + guarda cs057_resumen.json
"""
import sys, os, json
import numpy as np
import pandas as pd

CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "cs057_paisaje.csv")
CLASES = ["d1", "d2", "d3", "d4", "curv"]
WCOLS = ["w_grav", "w_strong", "w_em", "w_weak", "w_exp", "w_cool", "alc"]


def cargar():
    df = pd.read_csv(CSV)
    for c in CLASES:
        for m in ("viable", "estable", "expande", "acelera"):
            col = f"{m}_{c}"
            if col not in df.columns:
                df[col] = 0
    df["viab_tot"] = df[[f"viable_{c}" for c in CLASES]].sum(axis=1)
    df["acel_tot"] = df[[f"acelera_{c}" for c in CLASES]].sum(axis=1)
    return df


def resumen(df):
    R = {}
    R["n_filas"] = int(len(df))
    R["n_puntos"] = int(df.point_id.nunique())
    R["por_brazo"] = df.arm.value_counts().to_dict()
    # 1. viables por clase de dimensión (fracción de corridas donde esa dim quedó viable)
    R["viable_frac_por_dim"] = {c: round(float(df[f"viable_{c}"].mean()), 4) for c in CLASES}
    R["estable_frac_por_dim"] = {c: round(float(df[f"estable_{c}"].mean()), 4) for c in CLASES}
    R["expande_frac_por_dim"] = {c: round(float(df[f"expande_{c}"].mean()), 4) for c in CLASES}
    # 2. sync vs async (la falsación del proceso)
    sv = {}
    for arm in ("sync", "async"):
        d = df[df.arm == arm]
        sv[arm] = {c: round(float(d[f"viable_{c}"].mean()), 4) for c in CLASES}
        sv[arm]["viab_tot_medio"] = round(float(d["viab_tot"].mean()), 4)
    R["sync_vs_async"] = sv
    # ¿el sync estabiliza universos que el async no? por dim: viable_sync - viable_async
    R["sync_menos_async_por_dim"] = {c: round(sv["sync"][c] - sv["async"][c], 4) for c in CLASES}
    # 3. punto físico y vecindad densa
    for etiqueta, ph in (("punto_fisico", 1), ("vecindad_densa", 2)):
        d = df[df.phys == ph]
        if len(d):
            R[etiqueta] = {
                "n": int(len(d)),
                "viable_por_dim": {c: round(float(d[f"viable_{c}"].mean()), 4) for c in CLASES},
                "viab_tot_medio": round(float(d["viab_tot"].mean()), 4),
                "acelera_frac": round(float((d["acel_tot"] > 0).mean()), 4),
                "sync_vs_async_viab": {a: round(float(d[d.arm == a]["viab_tot"].mean()), 4)
                                       for a in ("sync", "async") if len(d[d.arm == a])},
            }
        else:
            R[etiqueta] = {"n": 0}
    # 4. sector oscuro: expansión acelerada emergente
    R["acelera_frac_global"] = round(float((df["acel_tot"] > 0).mean()), 4)
    R["acelera_por_dim"] = {c: round(float(df[f"acelera_{c}"].mean()), 4) for c in CLASES}
    # ¿la aceleración coincide con viabilidad? (co-ocurrencia acelera Y viable en la misma dim)
    coacc = {}
    for c in CLASES:
        both = ((df[f"acelera_{c}"] == 1) & (df[f"viable_{c}"] == 1)).sum()
        acc = (df[f"acelera_{c}"] == 1).sum()
        coacc[c] = round(float(both / acc), 4) if acc else None
    R["acelera_que_es_viable_por_dim"] = coacc
    # 5. qué pesos separan viable de no-viable (media de cada peso en viables vs no-viables, brazo sync)
    ds = df[df.arm == "sync"]
    reg = {}
    for c in CLASES:
        viab = ds[ds[f"viable_{c}"] == 1]
        novi = ds[ds[f"viable_{c}"] == 0]
        if len(viab) and len(novi):
            reg[c] = {w: [round(float(viab[w].mean()), 3), round(float(novi[w].mean()), 3)] for w in WCOLS}
    R["pesos_viable_vs_noviable_sync"] = reg  # [media_en_viable, media_en_noviable] por peso
    return R


def imprimir(R):
    print("=" * 100)
    print(f"CS057 — ANÁLISIS DEL PAISAJE  ({R['n_puntos']} puntos, {R['n_filas']} filas)")
    print("=" * 100)
    print("\n[1] VIABLE (estable+expande) por dimensión de partida — fracción de corridas:")
    for c in CLASES:
        print(f"    {c:>5}: viable={R['viable_frac_por_dim'][c]:.3f}  estable={R['estable_frac_por_dim'][c]:.3f}  expande={R['expande_frac_por_dim'][c]:.3f}")
    print("\n[2] SYNC vs ASYNC (falsación del 'es un proceso') — viable medio por dim:")
    print(f"    {'dim':>5} {'sync':>8} {'async':>8} {'sync-async':>12}")
    for c in CLASES:
        s = R["sync_vs_async"]["sync"][c]; a = R["sync_vs_async"]["async"][c]
        print(f"    {c:>5} {s:>8.3f} {a:>8.3f} {s-a:>+12.3f}")
    print(f"    viab_tot medio: sync={R['sync_vs_async']['sync']['viab_tot_medio']:.3f}  async={R['sync_vs_async']['async']['viab_tot_medio']:.3f}")
    print("\n[3] PUNTO FÍSICO real y vecindad densa:")
    for et in ("punto_fisico", "vecindad_densa"):
        d = R[et]
        if d.get("n", 0):
            vd = d["viable_por_dim"]
            print(f"    {et} (n={d['n']}): viab_tot={d['viab_tot_medio']:.3f}  acelera={d['acelera_frac']:.3f}")
            print(f"       viable por dim: " + "  ".join(f"{c}={vd[c]:.2f}" for c in CLASES))
    print("\n[4] SECTOR OSCURO — expansión que ACELERA sola (candidato energía oscura, emergente):")
    print(f"    fracción global con aceleración: {R['acelera_frac_global']:.3f}")
    print(f"    por dim: " + "  ".join(f"{c}={R['acelera_por_dim'][c]:.3f}" for c in CLASES))
    print(f"    de lo que acelera, cuánto es viable: " + "  ".join(f"{c}={R['acelera_que_es_viable_por_dim'][c]}" for c in CLASES))
    print("\n[5] PESOS que separan viable/no-viable (sync) — [media en viable, media en no-viable]:")
    for c, reg in R["pesos_viable_vs_noviable_sync"].items():
        difs = {w: round(reg[w][0] - reg[w][1], 2) for w in WCOLS}
        top = sorted(difs.items(), key=lambda kv: -abs(kv[1]))[:3]
        print(f"    {c:>5}: " + "  ".join(f"{w}Δ{d:+.2f}" for w, d in top))
    print("\n" + "=" * 100)


def main():
    if not os.path.exists(CSV):
        print(f"No existe {CSV} todavía."); return
    df = cargar()
    R = resumen(df)
    imprimir(R)
    out = os.path.join(os.path.dirname(os.path.abspath(CSV)), "cs057_resumen.json")
    with open(out, "w") as f:
        json.dump(R, f, indent=2, ensure_ascii=False)
    print(f"Resumen guardado en {out}")


if __name__ == "__main__":
    main()
