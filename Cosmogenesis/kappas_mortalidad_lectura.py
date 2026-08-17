"""Lectura de los CSV del barrido: tablas listas para el informe. Read-only."""
import csv, json, math
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent


def leer(n):
    with (R / n).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fl(v, d=float("nan")):
    try:
        return float(v)
    except Exception:
        return d


A = leer("kappas_mortalidad_curva_alpha.csv")
C = leer("kappas_mortalidad_curva_S0.csv")
B = leer("kappas_mortalidad_curva_delta.csv")
res = json.load((R / "kappas_mortalidad_resumen.json").open(encoding="utf-8"))

print("=" * 78)
print("A · CURVA lambda(alpha) — fraccion de semillas con lambda>0 y lambda medio")
for prec in ("float64", "float80"):
    for brazo in ("REAL", "BARAJADO", "ALFA_0"):
        sub = [r for r in A if r["barrido"] == "A_alpha" and r["precision"] == prec
               and r["brazo"] == brazo]
        if not sub:
            continue
        xs = sorted({fl(r["alpha"]) for r in sub})
        print(f"\n--- {brazo} / {prec}  (n_semillas={len({r['semilla'] for r in sub})})")
        print(f"{'alpha':>11} {'lambda_medio':>13} {'frac_lam>0':>10} {'frac_vivo_k1500':>15} {'frac_vivos>=2':>13}")
        for x in xs:
            s = [r for r in sub if fl(r["alpha"]) == x]
            lam = [fl(r["lambda_puro"]) for r in s]
            lam = [v for v in lam if math.isfinite(v)]
            print(f"{x:11.5g} {np.mean(lam) if lam else float('nan'):13.6f} "
                  f"{np.mean([fl(r['persiste_puro']) for r in s]):10.2f} "
                  f"{np.mean([fl(r['sobrevive_finito']) for r in s]):15.2f} "
                  f"{np.mean([fl(r['sobrevive_ge2']) for r in s]):13.2f}")

print("\n" + "=" * 78)
print("A' · CURVA lambda(MU) con alpha fijo=0.05  (umbral esperado: MU/(1-MU)=ETA*a*lmax)")
sub = [r for r in A if r["barrido"] == "Ap_mu"]
for brazo in ("REAL", "BARAJADO"):
    s2 = [r for r in sub if r["brazo"] == brazo]
    xs = sorted({fl(r["MU"]) for r in s2})
    print(f"--- {brazo}")
    for x in xs:
        s = [r for r in s2 if fl(r["MU"]) == x]
        lam = [fl(r["lambda_puro"]) for r in s]
        print(f"  MU={x:9.3g}  lambda_medio={np.mean(lam):+.6f}  frac_lam>0={np.mean([fl(r['persiste_puro']) for r in s]):.2f}")

print("\n" + "=" * 78)
print("UMBRALES alpha_c por semilla y por precision")
print(f"{'sem':>4} {'lmaxG':>8} {'teorico':>10} {'float64':>10} {'float80':>10} {'mpmath50':>10} {'k400':>10} {'BAR_f64':>10} {'BAR_f80':>10}")
for u in res["umbrales_alpha_c"]:
    g = lambda k: fl(u.get(k))
    print(f"{u['semilla']:>4} {g('lambda_max_G'):8.4f} {g('alpha_c_teorico'):10.6f} "
          f"{g('alpha_c_float64'):10.6f} {g('alpha_c_float80'):10.6f} {g('alpha_c_mpmath50'):10.6f} "
          f"{g('alpha_c_REAL_k400'):10.6f} {g('alpha_c_BARAJADO_float64'):10.6g} {g('alpha_c_BARAJADO_float80'):10.6g}")
d64 = [fl(u["alpha_c_float64"]) for u in res["umbrales_alpha_c"]]
d80 = [fl(u["alpha_c_float80"]) for u in res["umbrales_alpha_c"]]
dmp = [fl(u["alpha_c_mpmath50"]) for u in res["umbrales_alpha_c"]]
dte = [fl(u["alpha_c_teorico"]) for u in res["umbrales_alpha_c"]]
print(f"\ndesplazamiento relativo f64->f80 : max |d|/a = {max(abs(a-b)/a for a,b in zip(d64,d80)):.3e}")
print(f"desplazamiento relativo f64->mp50: max |d|/a = {max(abs(a-b)/a for a,b in zip(d64,dmp)):.3e}")
print(f"desviacion contra la formula      : max |d|/a = {max(abs(a-b)/a for a,b in zip(d64,dte)):.3e}")

print("\n" + "=" * 78)
print("C · S0 en 15 decadas (test de tautologia)")
for modo in ("absoluto", "escalado"):
    for brazo in ("REAL", "BARAJADO"):
        s2 = [r for r in C if r["modo"] == modo and r["brazo"] == brazo]
        xs = sorted({fl(r["S0"]) for r in s2})
        fr = [np.mean([fl(r["sobrevive_finito"]) for r in s2 if fl(r["S0"]) == x]) for x in xs]
        km = [np.mean([fl(r["k_micro"]) for r in s2 if fl(r["S0"]) == x]) for x in xs]
        nv = [np.mean([fl(r["n_vivos_fin"]) for r in s2 if fl(r["S0"]) == x]) for x in xs]
        print(f"--- {modo:9s} {brazo}")
        for x, a, b, c in zip(xs, fr, km, nv):
            print(f"   S0={x:9.3g}  frac_vivo={a:.2f}  k_micro_medio={b:7.1f}  n_vivos_medio={c:.2f}")

print("\n" + "=" * 78)
print("B · delta en 20 decadas — fraccion distinguible / operable")
for prec in ("float64", "float80", "mpmath_dps40"):
    for et in sorted({fl(r["eps_tau"]) for r in B}):
        s2 = [r for r in B if r["precision"] == prec and fl(r["eps_tau"]) == et]
        if not s2:
            continue
        xs = sorted({fl(r["delta"]) for r in s2})
        print(f"--- {prec}  eps_tau={et:g}")
        for x in xs:
            s = [r for r in s2 if fl(r["delta"]) == x]
            dist = np.mean([fl(r["distinguible"]) for r in s])
            op = np.mean([fl(r["operable"]) for r in s]) if s[0]["operable"] != "" else float("nan")
            div = np.mean([fl(r["div_rel_S"]) for r in s])
            print(f"   d={x:9.2e} dist={dist:.2f} oper={op:.2f} div_rel_media={div:.3e}")
print("\nRESUMEN kappa_delta:", json.dumps(res["kappa_delta"], indent=2, ensure_ascii=False))
print("\nGUARDAS:", json.dumps(res["guardas"], indent=2, ensure_ascii=False)[:4000])
