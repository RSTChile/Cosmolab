#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXPERIMENTO EN VIVO — falsación de la alteridad en la díada Docker.
Corre la díada bajo ANIMA_CONTROL = real | null | shuffled, deja vivir ~SEG s en cada condición,
lee la biografía (fisiologia CSV de A y B escrita en ESE tramo) y compara las señales de alteridad.
HIPÓTESIS (principio rector): las señales deben COLAPSAR bajo null/shuffled. Si no, son confound.
NO modifica órganos. Restaura ANIMA_CONTROL=real al terminar.
"""
import os, sys, csv, glob, time, subprocess

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCKER = os.path.join(RAIZ, "docker")
HIST = os.path.join(os.path.dirname(RAIZ), "Docker_Historia")   # /Volumes/.../Docker_Historia
SEG = int(os.environ.get("SEG", "150"))
SIG = ["alt_otro_presente", "alt_intencion_comunicativa",
       "alt_contingencia_social", "alt_agencia_otro",   # presencia (sobrevive) vs AGENCIA (debe colapsar)
       "voz_otro_valor_ecologico", "voz_otro_confianza_ecologica", "voz_otro_historia_beneficio",
       "voz_otro_efecto_real"]                            # ¿la voz del par IMPORTA para persistir? (debe caer NULL/SHUFFLED)
CONDS = ["real", "null", "shuffled"]

def recrear(cond):
    env = dict(os.environ); env["ANIMA_CONTROL"] = cond
    subprocess.run(["docker", "compose", "up", "-d", "--force-recreate", "anima-a", "anima-b"],
                   cwd=DOCKER, env=env, check=True, capture_output=True)

def ult_csv(org):
    fs = glob.glob(os.path.join(HIST, f"organismo_{org}", "fisiologia", "fisiologia_*.csv"))
    return max(fs, key=os.path.getmtime) if fs else None

def lineas(org):
    f = ult_csv(org)
    if not f:
        return 0
    with open(f, encoding="utf-8", errors="replace") as fh:
        return sum(1 for _ in fh)

def medias(org, desde):
    """Promedia las señales en las filas escritas DESDE el marcador (las de esta condición)."""
    f = ult_csv(org)
    if not f:
        return {}
    rows = [r for r in csv.reader(open(f, encoding="utf-8", errors="replace")) if r and not r[0].startswith("#")]
    if not rows:
        return {}
    cab = rows[0]; data = [dict(zip(cab, r)) for r in rows[1:] if len(r) == len(cab)]
    # si el archivo es el mismo, toma desde 'desde'; si cambió (sesión nueva), toma todo
    seg = data[max(0, desde - 1):] if desde < len(data) else data
    seg = seg[-1200:]
    out = {}
    for k in SIG:
        vs = []
        for d in seg:
            try:
                vs.append(float(d.get(k, "") or 0))
            except Exception:
                pass
        out[k] = (sum(vs) / len(vs)) if vs else float("nan")
    out["_n"] = len(seg)
    return out

def main():
    print(f"[control-vivo] SEG={SEG}s por condición · conds={CONDS}", flush=True)
    res = {}
    for cond in CONDS:
        print(f"\n[control-vivo] === condición {cond.upper()} : recreando A/B ===", flush=True)
        n0A, n0B = lineas("ANIMA_A"), lineas("ANIMA_B")
        recrear(cond)
        time.sleep(SEG)
        mA, mB = medias("ANIMA_A", n0A), medias("ANIMA_B", n0B)
        res[cond] = {"A": mA, "B": mB}
        print(f"[control-vivo] {cond}: A(n={mA.get('_n')}) B(n={mB.get('_n')})", flush=True)
        for k in SIG:
            print(f"    {k:30s} A={mA.get(k, float('nan')):.4f}  B={mB.get(k, float('nan')):.4f}", flush=True)

    print("\n[control-vivo] restaurando ANIMA_CONTROL=real …", flush=True)
    recrear("real")

    print("\n" + "=" * 78)
    print("RESULTADO — señales de alteridad por condición (promedio A+B)")
    print("=" * 78)
    def prom(cond, k):
        a = res[cond]["A"].get(k, float("nan")); b = res[cond]["B"].get(k, float("nan"))
        xs = [x for x in (a, b) if x == x]
        return sum(xs) / len(xs) if xs else float("nan")
    print(f"  {'señal':30s} {'REAL':>9} {'NULL':>9} {'SHUFFLED':>9}   cae/NULL  cae/SHUF")
    for k in SIG:
        r, n, s = prom("real", k), prom("null", k), prom("shuffled", k)
        caeN = "sí" if (r > 0.01 and n < 0.5 * r) else "no"
        caeS = "sí" if (r > 0.01 and s < 0.5 * r) else "no"
        print(f"  {k:30s} {r:9.4f} {n:9.4f} {s:9.4f}     {caeN:>4}      {caeS:>4}")
    print("\n  Lectura esperada (Belbo): PRESENCIA (otro_presente, efecto, intención) cae con NULL pero")
    print("  SOBREVIVE a shuffle (el otro sigue ahí). AGENCIA (contingencia_social, agencia_otro) debe")
    print("  CAER con NULL **y** con SHUFFLE: si el otro cambia decorrelacionado de mi acto, no hay agencia.")

if __name__ == "__main__":
    main()
