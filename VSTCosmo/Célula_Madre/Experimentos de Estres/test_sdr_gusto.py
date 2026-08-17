#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_sdr_gusto.py — ¿nace un GUSTO emergente? (cierre de Alexis, 5-jul-2026)

De 'come orden' a 'tiene apetito por ESTE orden'. El órgano aprende gusto[frecuencia] = nutrición
sacada ahí; el barrido, cada gusto_periodo saltos, vuelve a su favorita. Falsable con un MODELO DE
MUNDO: una emisora en 96.0 MHz que se DESVANECE a ratos (fuerza re-búsqueda).

Brazos:
  REAL      — emisora (pico contiguo) en 96.0 → debe nacer gusto[96] y VOLVER ahí más rápido/más tiempo.
  NULL      — ruido en 96.0 (misma energía, sin orden) → no nutre → no cuaja gusto → no dwelling.
  SIN-GUSTO — emisora real pero apetito APAGADO (ablación) → tras cada desvanecimiento vaga más.

Métrica: dwelling = fracción de la 2ª mitad ENGANCHADO cerca de 96 (±0.6 MHz). El gusto debe subir
el dwelling en REAL vs SIN-GUSTO, y no aparecer en NULL.

Uso:  ~/.venvs/vstcosmo/bin/python test_sdr_gusto.py
"""
import os, sys, random

CARP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(CARP, "..", "organelos")))
from VST_OrganoRadio import OrganoRadio   # noqa: E402

OUT = os.path.join(CARP, "resultado_sdr_gusto_2026-07-05.md")
TICKS = 260
BUENA = 96.0            # MHz donde vive la emisora
SR = 2.0               # MHz de ventana

def log(m):
    with open(OUT, "a") as f: f.write(m + "\n")
    print(m, flush=True)

def mundo(center_mhz, hay_emision, ruido_solo, rnd, amp=0.92, piso=0.12):
    """Espectro de la ventana centrada en center_mhz. Si hay_emision y la buena cae en la ventana:
    pico CONTIGUO en 96.0 (a menos que ruido_solo → energía sin orden). Piso de ruido siempre."""
    N = 256
    fmin = center_mhz - SR/2; fmax = center_mhz + SR/2
    esp = [piso + 0.04*rnd.random() for _ in range(N)]
    if hay_emision and fmin <= BUENA <= fmax:
        b = int((BUENA - fmin)/SR * N)
        if ruido_solo:                          # misma energía, DISPERSA (sin orden) → no es comida
            for _ in range(9):
                esp[rnd.randrange(N)] = amp * (0.4 + 0.6*rnd.random())
        else:                                   # emisión: bump contiguo
            for d, a in ((-2,0.45),(-1,0.72),(0,1.0),(1,0.72),(2,0.45)):
                i = b+d
                if 0 <= i < N: esp[i] = max(esp[i], amp*a)
    return esp, fmin*1e6, fmax*1e6

def correr(nombre, ruido_solo=False, gusto_on=True, seed=1):
    rnd = random.Random(seed)
    org = OrganoRadio("GUSTO", activo=True, sintonia_activa=True, histeresis=True, n_bandas=16,
                      settle_ticks=0,                            # sim sin latencia HW → barrido rápido (vaga lejos en el fade)
                      gusto_periodo=(3 if gusto_on else 10**9))  # ablación = nunca sesga
    center = 104.0                               # arranca lejos de la buena (cruzando la banda)
    dwell = {"1": [0,0], "2": [0,0]}             # [cerca_y_enganchado, total] por mitad
    for t in range(TICKS):
        hay = not (t % 40 < 18)                  # DESVANECIMIENTO largo (18/40): fuerza a vagar lejos de 96
        esp, fmin, fmax = mundo(center, hay, ruido_solo, rnd)
        o = org.observar({"t": t, "sdr_espectro": esp, "sdr_freq_min_hz": fmin, "sdr_freq_max_hz": fmax, "sdr_vivo": 1})
        orden = o.get("radio_orden_hz")
        if orden: center = orden/1e6
        eng = (o.get("radio_enganchado") or 0) > 0.5
        mitad = "1" if t < TICKS//2 else "2"
        dwell[mitad][1] += 1
        if eng and abs(center - BUENA) < 0.6: dwell[mitad][0] += 1
    d1 = dwell["1"][0]/max(1,dwell["1"][1]); d2 = dwell["2"][0]/max(1,dwell["2"][1])
    g96 = org._gusto.get(96.0, 0.0)
    top = sorted(org._gusto.items(), key=lambda kv: -kv[1])[:3]
    return {"d1": d1, "d2": d2, "g96": g96, "top": top}

def main():
    if os.path.exists(OUT): os.remove(OUT)
    log("# ¿Nace un GUSTO emergente por la frecuencia que nutre? — falsación (cierre de Alexis)\n")
    log("Modelo de mundo: emisora en 96.0 MHz que se desvanece 4/26 ticks. dwelling = fracción")
    log("ENGANCHADO cerca de 96 (±0.6 MHz), por mitad. El apetito debe SUBIR el dwelling en la 2ª mitad.\n")
    R = {"REAL (emisión en 96)": correr("REAL"),
         "NULL (ruido en 96, misma energía)": correr("NULL", ruido_solo=True),
         "SIN-GUSTO (emisión, apetito OFF)": correr("SINGUSTO", gusto_on=False)}
    log("| brazo | gusto[96] | dwelling 1ª mitad | dwelling 2ª mitad | Δ (aprendizaje) |")
    log("|---|---|---|---|---|")
    for n, r in R.items():
        log("| %s | %.3f | %.0f%% | %.0f%% | %+.0f pts |" % (n, r["g96"], r["d1"]*100, r["d2"]*100, (r["d2"]-r["d1"])*100))
    log("")
    real = R["REAL (emisión en 96)"]; null = R["NULL (ruido en 96, misma energía)"]; sing = R["SIN-GUSTO (emisión, apetito OFF)"]
    log("Gusto top-3 (REAL): %s" % ", ".join("%.1fMHz=%.3f" % (k, v) for k, v in real["top"]))
    log("\n## Veredicto")
    ok_gusto = real["g96"] > 0.05 and real["g96"] > null["g96"] + 0.03
    ok_dwell = real["d2"] > sing["d2"] + 0.10 and real["d2"] > null["d2"] + 0.10
    if ok_gusto and ok_dwell:
        log("- ✅ **GUSTO EMERGENTE.** En REAL nace un apetito por 96 (gusto=%.3f) y el organismo pasa" % real["g96"])
        log("  MÁS tiempo enganchado ahí en la 2ª mitad (%.0f%% vs %.0f%% sin apetito, %.0f%% en ruido)." % (
            real["d2"]*100, sing["d2"]*100, null["d2"]*100))
        log("  El apetito MUEVE la conducta: vuelve a lo que lo nutrió. Y colapsa en ruido (falsable).")
    else:
        log("- ⚠️ No concluyente: REAL g96=%.3f/dwell2=%.0f%% · SIN-GUSTO dwell2=%.0f%% · NULL g96=%.3f/dwell2=%.0f%%." % (
            real["g96"], real["d2"]*100, sing["d2"]*100, null["g96"], null["d2"]*100))
        log("  Ajustar gusto_periodo/umbral o el modelo de desvanecimiento.")

if __name__ == "__main__":
    main()
