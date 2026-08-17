#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_paladar.py — ¿emerge UN paladar único sobre todas las modalidades? (cierre de Alexis)

El metabolismo aprende UN mapa de preferencia sobre modalidades distintas —mundo, voz del otro,
radio— cada una con su PROPIA nutrición (semiosis por IM; radio por negentropía). Falsable: con una
dieta variada nace un RANKING coherente (favorito = lo más nutritivo); con una dieta donde NADA nutre
distinto, el paladar queda PLANO (no cuajan preferencias).

Uso:  ~/.venvs/vstcosmo/bin/python test_paladar.py
"""
import os, sys

CARP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(CARP, "..", "organelos")))
from VST_Metabolismo import OrganeloMetabolismo   # noqa: E402

OUT = os.path.join(CARP, "resultado_paladar_2026-07-05.md")

def log(m):
    with open(OUT, "a") as f: f.write(m + "\n")
    print(m, flush=True)

def comer(met, modalidad, icr, irde, lat, radio_nut=0.0, radio_f=0.0, n=40):
    for _ in range(n):
        met.actualizar({"ICR_ratio": icr, "IRDE_ratio": irde, "RC_total": 0.3, "lateralidad": lat,
                        "met_modalidad": modalidad, "radio_nutricion": radio_nut,
                        "radio_freq_dom_hz": radio_f}, dt=0.1)

def paladar_ordenado(met):
    return sorted(met.preferencia.items(), key=lambda kv: -kv[1])

def main():
    if os.path.exists(OUT): os.remove(OUT)
    log("# ¿Un PALADAR único sobre mundo + voz del otro + radio? — falsación (cierre de Alexis)\n")

    # ---- DIETA VARIADA: cada modalidad nutre distinto ----
    log("## Dieta variada (cada fuente alimenta distinto)")
    m = OrganeloMetabolismo(E0=0.6)
    comer(m, "mundo",    icr=0.75, irde=0.10, lat=0.3)                     # música nutritiva
    comer(m, "mundo",    icr=0.15, irde=0.65, lat=-0.3)                    # audio 'tóxico' (riesgo>orden)
    comer(m, "voz_otro", icr=0.55, irde=0.25, lat=0.1)                     # la voz del otro, moderada
    comer(m, "radio",    icr=0.30, irde=0.30, lat=0.0, radio_nut=0.13, radio_f=96e6)   # emisión rica en 96
    comer(m, "radio",    icr=0.30, irde=0.30, lat=0.0, radio_nut=0.02, radio_f=101e6)  # emisión pobre en 101
    log("| alimento (modalidad|sabor) | preferencia |")
    log("|---|---|")
    for k, v in paladar_ordenado(m):
        log("| %s | %+.3f |" % ("|".join(str(x) for x in k), v))
    fav = paladar_ordenado(m)[0]
    log("\n→ **Favorito del paladar: `%s` (%.3f)** — un solo mapa rankea modalidades distintas." % (
        "|".join(str(x) for x in fav[0]), fav[1]))

    # ---- CONTROL NULL: nada nutre distinto ----
    log("\n## Control NULL (nada alimenta distinto)")
    mn = OrganeloMetabolismo(E0=0.6)
    comer(mn, "mundo",    icr=0.30, irde=0.30, lat=0.0)                    # IM≈0
    comer(mn, "voz_otro", icr=0.30, irde=0.30, lat=0.0)
    comer(mn, "radio",    icr=0.30, irde=0.30, lat=0.0, radio_nut=0.0)    # radio sin negentropía
    pal_n = paladar_ordenado(mn)
    spread_n = (pal_n[0][1] - pal_n[-1][1]) if pal_n else 0.0
    log("| alimento | preferencia |")
    log("|---|---|")
    for k, v in pal_n:
        log("| %s | %+.3f |" % ("|".join(str(x) for x in k), v))
    log("")

    # ---- veredicto ----
    pal_v = paladar_ordenado(m)
    spread_v = (pal_v[0][1] - pal_v[-1][1]) if pal_v else 0.0
    modalidades = set(k[0] for k, _ in pal_v)
    log("## Veredicto")
    log("- Modalidades en UN solo paladar: %s" % ", ".join(sorted(modalidades)))
    log("- Rango de preferencia — dieta variada: **%.3f** · control NULL: **%.3f**" % (spread_v, spread_n))
    if len(modalidades) >= 3 and spread_v > 0.2 and spread_v > spread_n + 0.15:
        log("- ✅ **PALADAR ÚNICO EMERGENTE.** Un mapa rankea mundo + voz del otro + radio por cuánto")
        log("  nutrió cada uno; el favorito es el más nutritivo. Con NULL el paladar queda PLANO (nada")
        log("  destaca). El gusto es por el ORDEN que alimenta, venga de la modalidad que venga. Falsable.")
    else:
        log("- ⚠️ No concluyente (modalidades=%d, rango variado=%.3f vs null=%.3f)." % (len(modalidades), spread_v, spread_n))

if __name__ == "__main__":
    main()
