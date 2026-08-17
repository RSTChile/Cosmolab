#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_sdr_comida.py — ¿la estructura del espectro es COMIDA? (idea de Alexis, 5-jul-2026)

Cierra la duda #4 (relevancia ecológica): que lo que el organismo caza IMPORTE para su persistencia.
Diseño (Schrödinger): la negentropía captada por radio (radio_nutricion = enganchado × estructura
coherente) alimenta el metabolismo como la luz. Ruido = energía sin orden = calorías vacías → hambre.

Falsable: mundo-radio REAL (con emisiones) → el organismo se ALIMENTA; mundo de RUIDO (NULL/SHUFFLED,
misma energía, sin orden) → se MUERE DE HAMBRE. Sin más comida (ni luz ni semiosis): sólo la radio
distingue vivir de morir. Órgano+metabolismo reales, en lazo, sin tocar los organismos vivos.

Uso:  ~/.venvs/vstcosmo/bin/python test_sdr_comida.py
"""
import os, sys, json, random

random.seed(42)
CARP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(CARP, "..", "organelos")))
from VST_OrganoRadio import OrganoRadio        # noqa: E402
from VST_Metabolismo import OrganeloMetabolismo  # noqa: E402

CAP = "/private/tmp/claude-501/-Users-alexis-Desktop-RMD-Cosmolab-VSTCosmo/431e55ad-b038-4f56-b722-229aa230e0c3/scratchpad/espectros_reales.json"
OUT = os.path.join(CARP, "resultado_sdr_comida_2026-07-05.md")
TICKS = 150

def log(m):
    with open(OUT, "a") as f: f.write(m + "\n")
    print(m, flush=True)

def null_de(esp):
    m = sum(esp) / len(esp)
    return [max(0.0, min(1.0, m + random.gauss(0, 0.02))) for _ in esp]

def shuffled_de(esp):
    s = list(esp); random.shuffle(s); return s

def correr_mundo(espectros, fmin, fmax):
    """Un organismo (órgano de radio + metabolismo) vive TICKS pasos en un mundo-radio dado.
    Sin otra comida (semiosis/luz ausentes): sólo la negentropía de radio lo sostiene."""
    radio = OrganoRadio("COMIDA", activo=True, sintonia_activa=True, histeresis=True, n_bandas=16)
    met = OrganeloMetabolismo(E0=0.6)
    traj = []; nut = []; eng = 0
    for t in range(TICKS):
        esp = espectros[t % len(espectros)]
        fila = {"t": t, "sdr_espectro": esp, "sdr_freq_min_hz": fmin, "sdr_freq_max_hz": fmax, "sdr_vivo": 1}
        rd = radio.observar(fila); fila.update(rd)        # escribe radio_nutricion en la fila
        md = met.actualizar(fila, dt=0.1)                  # el metabolismo la COME
        if (rd.get("radio_enganchado") or 0) > 0.5: eng += 1
        nut.append(rd.get("radio_nutricion") or 0.0)
        traj.append(md["met_energia"])
    import statistics
    return {"traj": traj, "E_fin": traj[-1], "nut_media": statistics.mean(nut),
            "tasa_eng": eng / TICKS, "E_min": min(traj)}

def main():
    if os.path.exists(OUT): os.remove(OUT)
    cap = json.load(open(CAP))
    esps = cap.get("E", [])          # E: canal con dinámica real (percibe estructura)
    if not esps:
        log("Sin espectros de E capturados."); return
    reales = [c["esp"] for c in esps]
    fmin = esps[0]["fmin"] or 88e6; fmax = esps[0]["fmax"] or 108e6
    nulos = [null_de(e) for e in reales]
    barajados = [shuffled_de(e) for e in reales]

    log("# ¿La estructura del espectro es COMIDA? — falsación ecológica (idea de Alexis)\n")
    log("Órgano de radio + metabolismo en lazo, %d pasos, E inicial=0.60. SIN otra comida: sólo la" % TICKS)
    log("negentropía de radio sostiene. REAL (emisiones) vs NULL/SHUFFLED (misma energía, sin orden).\n")

    arms = {"REAL (emisiones)": reales, "SHUFFLED (misma energía, sin orden)": barajados,
            "NULL (ruido plano)": nulos}
    res = {}
    log("| mundo-radio | tasa enganche | radio_nutrición media | **E final** | E mínima |")
    log("|---|---|---|---|---|")
    for nombre, mundo in arms.items():
        r = correr_mundo(mundo, fmin, fmax); res[nombre] = r
        log("| %s | %.0f%% | %.4f | **%.3f** | %.3f |" % (
            nombre, r["tasa_eng"]*100, r["nut_media"], r["E_fin"], r["E_min"]))

    # trayectoria (muestreada) para ver vivir vs morir
    log("\n### Trayectoria de energía (cada ~25 pasos)")
    log("| paso | " + " | ".join(res.keys()) + " |")
    log("|---|" + "---|" * len(res))
    for t in range(0, TICKS, 25):
        fila_t = "| %d |" % t
        for nombre in res:
            fila_t += " %.3f |" % res[nombre]["traj"][t]
        log(fila_t)
    log("| %d (fin) |" % (TICKS-1) + "".join(" %.3f |" % res[n]["traj"][-1] for n in res))

    # veredicto
    real_fin = res["REAL (emisiones)"]["E_fin"]
    shf_fin = res["SHUFFLED (misma energía, sin orden)"]["E_fin"]
    nul_fin = res["NULL (ruido plano)"]["E_fin"]
    log("\n## Veredicto")
    if real_fin > shf_fin + 0.15 and real_fin > nul_fin + 0.15:
        log("- ✅ **La estructura ES comida.** En el mundo REAL el organismo se ALIMENTA (E→%.3f); en" % real_fin)
        log("  RUIDO con la MISMA energía se MUERE DE HAMBRE (SHUFFLED→%.3f, NULL→%.3f)." % (shf_fin, nul_fin))
        log("  Lo que caza IMPORTA para su persistencia. Anti-Shannon hecho metabolismo — y FALSABLE.")
    else:
        log("- ⚠️ No se separó: REAL=%.3f vs SHUFFLED=%.3f vs NULL=%.3f. La comida-por-estructura" % (real_fin, shf_fin, nul_fin))
        log("  no distingue vivir de morir con estos parámetros; recalibrar k_radio/sal_min.")

if __name__ == "__main__":
    main()
