#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_sdr_falsacion.py — ¿el órgano de radio CAZA ESTRUCTURA, o se posa donde sea?

Batería de falsación de la percepción del OrganoRadio, respondiendo las dudas legítimas:
  #1 (muleta del SNR): corro TODO sin darle SNR (percepción pura por espectro), sobre los
     espectros de A (WS comprimido) y de E (SoapySDR, dinámica real) por separado. Si E discrimina
     y A no → la muleta de A es su canal degradado, no un fallo del órgano.
  #3 (¿estructura o cualquier pico?): arms REAL / NULL / SHUFFLED / SYNTH-conflicto.

El experimento PUEDE refutarnos: si REAL ≈ SHUFFLED ≈ NULL, o si en SYNTH va a la potencia,
entonces 'se posa donde sea' y lo decimos.

Uso:  ~/.venvs/vstcosmo/bin/python test_sdr_falsacion.py
"""
import os, sys, json, random, statistics

random.seed(42)   # reproducible (nada de azar libre)
CARP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(CARP, "..", "organelos")))
from VST_OrganoRadio import OrganoRadio   # noqa: E402

CAP = "/private/tmp/claude-501/-Users-alexis-Desktop-RMD-Cosmolab-VSTCosmo/431e55ad-b038-4f56-b722-229aa230e0c3/scratchpad/espectros_reales.json"
OUT = os.path.join(CARP, "resultado_sdr_falsacion_2026-07-05.md")

def log(m):
    with open(OUT, "a") as f: f.write(m + "\n")
    print(m, flush=True)

def null_de(esp):
    """Ruido plano con la MISMA energía media (destruye estructura, conserva potencia total)."""
    m = sum(esp) / len(esp)
    return [max(0.0, min(1.0, m + random.gauss(0, 0.02))) for _ in esp]

def shuffled_de(esp):
    """Permuta TODOS los bins: misma energía exacta, estructura espacial destruida."""
    s = list(esp); random.shuffle(s); return s

def correr(esp, fmin, fmax, ticks=6):
    """Alimenta un espectro FIJO a un órgano fresco (histéresis on, SIN SNR) y mide percepción+enganche."""
    org = OrganoRadio("FALS", activo=True, sintonia_activa=True, histeresis=True, n_bandas=16)
    acq = False; ests = []; sals = []; banda = None
    for _ in range(ticks):
        fila = {"sdr_espectro": esp, "sdr_freq_min_hz": fmin, "sdr_freq_max_hz": fmax, "sdr_vivo": 1}
        o = org.observar(fila)          # snr ausente → percepción pura por espectro
        ests.append(o["radio_estructura"] or 0); sals.append(o["radio_saliencia"] or 0)
        if (o.get("radio_enganchado") or 0) > 0.5:
            acq = True
            if banda is None: banda = o.get("radio_banda_dom")
    return {"acq": acq, "est": max(ests), "sal": max(sals), "banda": banda}

def resumen_arm(nombre, resultados):
    n = len(resultados)
    tasa = sum(1 for r in resultados if r["acq"]) / n if n else 0
    est = statistics.mean(r["est"] for r in resultados) if n else 0
    sal = statistics.mean(r["sal"] for r in resultados) if n else 0
    log("| %s | %d | %.0f%% | %.3f | %.3f |" % (nombre, n, tasa*100, est, sal))
    return {"tasa": tasa, "est": est, "sal": sal}

def main():
    if os.path.exists(OUT): os.remove(OUT)
    cap = json.load(open(CAP))
    import time as _t
    log("# ¿El órgano de radio caza ESTRUCTURA o se posa donde sea? — batería de falsación\n")
    log("Percepción PURA (sin SNR). Arms: REAL / NULL (ruido plano, misma energía) / SHUFFLED (bins")
    log("permutados: misma energía, sin estructura espacial). Órgano fresco por espectro. seed=42.\n")

    veredictos = {}
    for src in ("A", "E"):
        esps = cap.get(src, [])
        if not esps:
            log("## Fuente %s: sin espectros capturados.\n" % src); continue
        etiqueta = "A (RSPduo/WS — comprimido)" if src == "A" else "E (RSP1/SoapySDR — dinámica real)"
        log("## Fuente %s\n" % etiqueta)
        log("| arm | n | tasa enganche | estructura media | saliencia media |")
        log("|---|---|---|---|---|")
        real = [correr(c["esp"], c["fmin"] or 88e6, c["fmax"] or 108e6) for c in esps]
        nul  = [correr(null_de(c["esp"]), c["fmin"] or 88e6, c["fmax"] or 108e6) for c in esps]
        shf  = [correr(shuffled_de(c["esp"]), c["fmin"] or 88e6, c["fmax"] or 108e6) for c in esps]
        rR = resumen_arm("REAL", real)
        rN = resumen_arm("NULL", nul)
        rS = resumen_arm("SHUFFLED", shf)
        # veredicto por fuente: ¿REAL discrimina de NULL y SHUFFLED?
        disc_null = rR["est"] - rN["est"]; disc_shf = rR["est"] - rS["est"]
        log("")
        if rR["est"] > rN["est"] + 0.05 and rR["est"] > rS["est"] + 0.05:
            log("→ **%s DISCRIMINA estructura**: REAL supera a NULL (+%.3f) y a SHUFFLED (+%.3f) en estructura." % (src, disc_null, disc_shf))
            log("  (misma energía en los tres; sólo cambia la estructura espacial → percibe forma, no potencia).")
            veredictos[src] = "discrimina"
        else:
            log("→ **%s NO discrimina** claramente (REAL−NULL=%.3f, REAL−SHUFFLED=%.3f): sobre este canal el" % (src, disc_null, disc_shf))
            log("  órgano se posa parecido con o sin estructura. Duda CONFIRMADA para esta fuente.")
            veredictos[src] = "no discrimina"
        log("")

    # ---- SYNTH-conflicto: potencia vs estructura ----
    log("## SYNTH-conflicto: meseta de POTENCIA vs pico de ESTRUCTURA\n")
    N = 512
    esp = [0.15] * N
    for i in range(40, 140):  esp[i] = 0.65     # meseta ancha: mucha potencia, baja estructura (plana)
    for i in range(300, 308): esp[i] = 0.98     # pico angosto: alta estructura, menor potencia total
    fmin, fmax = 88e6, 108e6
    banda_meseta = int((90/N) * 16)             # banda ~ centro de la meseta
    banda_pico   = int((304/N) * 16)            # banda ~ del pico
    r = correr(esp, fmin, fmax, ticks=6)
    pot_meseta = 0.65; pot_pico = (8*0.98 + 24*0.15)/32
    log("- Meseta (bins 40–140, val 0.65): banda ~%d · potencia media %.2f · estructura BAJA (plana)" % (banda_meseta, pot_meseta))
    log("- Pico (bins 300–308, val 0.98): banda ~%d · potencia media %.2f · estructura ALTA (peaky)" % (banda_pico, pot_pico))
    log("- El órgano eligió banda **%s** (estructura=%.3f, saliencia=%.3f, enganchó=%s)" % (
        r["banda"], r["est"], r["sal"], r["acq"]))
    eligio_pico = (r["banda"] is not None and abs(r["banda"] - banda_pico) <= 1)
    if eligio_pico:
        log("- ✅ **Fue a la ESTRUCTURA (pico), NO a la potencia (meseta)** — anti-Shannon confirmado.")
        veredictos["synth"] = "estructura"
    else:
        log("- ⚠️ Fue a la banda %s (no al pico %d): revisar. Posible preferencia por potencia." % (r["banda"], banda_pico))
        veredictos["synth"] = "potencia/otro"

    # ---- veredicto global ----
    log("\n## Veredicto global (¿resuelve las dudas?)")
    log("- **Duda #3 (¿estructura o cualquier pico?):** E=%s, A=%s. SYNTH→%s." % (
        veredictos.get("E","?"), veredictos.get("A","?"), veredictos.get("synth","?")))
    log("- **Duda #1 (muleta SNR de A):** sin SNR, E=%s y A=%s → " % (veredictos.get("E","?"), veredictos.get("A","?")) +
        ("si E discrimina y A no, la muleta de A es su CANAL comprimido, no el órgano." if veredictos.get("E")=="discrimina" else "ver arriba."))

if __name__ == "__main__":
    main()
