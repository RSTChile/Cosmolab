#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_sdr_barrido.py — ¿el organismo BARRE el espectro hasta SELECCIONAR una emisión?

Responde en vivo, sobre el RSPduo real (vía SDRconnect headless WS :5454), la pregunta de
Alexis: hasta ahora los tests corrían el órgano de radio en modo OBSERVA (pasivo, enganchado
donde lo sintonizábamos a mano). Aquí ACTIVAMOS el actuador (sintonia_activa=True) y usamos la
POLÍTICA REAL del OrganoRadio (argmax de SALIENCIA = estructura+novedad, NO potencia bruta).

FASES:
  1) BARRIDO   — paso el centro del RSPduo por la banda FM (88→108 MHz) en saltos solapados.
                 En cada ventana (~2 MHz) corro el órgano y anoto la sub-banda más saliente:
                 estructura, saliencia, potencia y su frecuencia. Es el MAPA del espectro.
  2) SELECCIÓN — de todas las ventanas, elijo la de MAYOR saliencia (criterio del órgano). Esa
                 es la emisión que el organismo "escogería". Muestro también cuál habría ganado
                 por POTENCIA, para probar que el órgano NO elige por energía cruda.
  3) ENGANCHE  — sintonizo la emisión elegida y SOSTENGO ~45 s, muestreando: si freq_dom se queda
                 clavado y la saliencia se mantiene, el organismo se ENGANCHÓ (lock).

Aislado: no reinicia ni toca los organismos vivos; instancia su propio lector+órgano. El lector
de A (si está) corre en OBSERVA y no mueve la sintonía, así que sólo este test conduce el LO.

Uso:  ~/.venvs/vstcosmo/bin/python test_sdr_barrido.py
Env:  ANIMA_SDRWS_URI (def ws://127.0.0.1:5454), BARRIDO_F0/F1 (MHz), BARRIDO_PASO (MHz)
"""
import os, sys, time, statistics

CARP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(CARP, "..", "organelos")))

from VST_OrganoRadio import OrganoRadio          # noqa: E402
from VST_LectorSDRServidor import LectorSDRServidor  # noqa: E402

F0   = float(os.environ.get("BARRIDO_F0", "88.0"))   # MHz inicio
F1   = float(os.environ.get("BARRIDO_F1", "108.0"))  # MHz fin
PASO = float(os.environ.get("BARRIDO_PASO", "1.6"))  # MHz por salto (solape con ventana ~2 MHz)
ASENT = float(os.environ.get("BARRIDO_ASENT", "2.2"))  # s de asentamiento tras re-sintonizar
HOLD = float(os.environ.get("BARRIDO_HOLD", "45"))   # s de enganche sostenido
OUT = os.path.join(CARP, "resultado_sdr_barrido_2026-07-05.md")

def log(m):
    with open(OUT, "a") as f: f.write(m + "\n")
    print(m, flush=True)

def leer_ventana(lector, radio, freq_hz, asent):
    """Sintoniza freq_hz, deja asentar, y devuelve la observación del órgano (última, la más fresca)."""
    lector.sintonizar(freq_hz)
    t0 = time.time(); ultima = None
    while time.time() - t0 < asent:
        fila = {}
        lector.inyectar(fila)
        if fila.get("sdr_vivo"):
            ultima = radio.observar(fila)   # corre la POLÍTICA real del órgano
        time.sleep(0.3)
    return ultima

def main():
    if os.path.exists(OUT): os.remove(OUT)
    log("# ¿El organismo BARRE el espectro hasta SELECCIONAR una emisión? — %s\n" % time.strftime("%Y-%m-%d %H:%M"))
    log("Órgano de radio con **actuador ACTIVO** (sintonia_activa=True), política real = argmax de")
    log("SALIENCIA (estructura+novedad), sobre el RSPduo. Banda %.1f–%.1f MHz, paso %.1f MHz.\n" % (F0, F1, PASO))

    lector = LectorSDRServidor()
    if not lector.arrancar():
        log("**ERROR**: no arrancó el lector (¿headless :5454 sin RSPduo?). Aborto."); return
    # esperar a que enganche el device
    t0 = time.time()
    while time.time() - t0 < 12:
        fila = {}; lector.inyectar(fila)
        if fila.get("sdr_vivo"): break
        time.sleep(0.4)
    else:
        log("**ERROR**: el lector conectó pero no llega espectro (sdr_vivo=0). ¿RSPduo abierto?"); lector.cerrar(); return
    log("Lector vivo. Espectro fluyendo. Comienzo el BARRIDO.\n")

    # órgano con el ACTUADOR ENCENDIDO (lo que nunca activamos en los organismos)
    radio = OrganoRadio("BARRIDO", activo=True, sintonia_activa=True,
                        n_bandas=int(os.environ.get("ANIMA_RADIO_N_BANDAS", "16")))

    # ---- FASE 1: BARRIDO ----
    log("## 1) Barrido del espectro (mapa: qué encuentra en cada ventana)")
    log("| centro (MHz) | freq dom (MHz) | estructura | saliencia | potencia | bins señal |")
    log("|---|---|---|---|---|---|")
    mapa = []
    f = F0
    while f <= F1 + 1e-6:
        obs = leer_ventana(lector, radio, f * 1e6, ASENT)
        if obs:
            fdom = (obs.get("radio_freq_dom_hz") or 0) / 1e6
            est = obs.get("radio_estructura") or 0
            sal = obs.get("radio_saliencia") or 0
            pot = obs.get("radio_potencia_total") or 0
            # bins con señal: re-leer una fila para contar
            fila = {}; lector.inyectar(fila); esp = fila.get("sdr_espectro") or []
            nz = sum(1 for v in esp if v > 0.15) if esp else 0
            mapa.append({"centro": f, "fdom": fdom, "est": est, "sal": sal, "pot": pot, "nz": nz})
            log("| %.1f | %.2f | %.3f | %.3f | %.3f | %d |" % (f, fdom, est, sal, pot, nz))
        else:
            log("| %.1f | — sin espectro — |||||" % f)
        f += PASO

    if not mapa:
        log("\n**Sin datos de barrido.** Aborto."); lector.cerrar(); return

    # ---- FASE 2: SELECCIÓN ----
    elegida = max(mapa, key=lambda m: m["sal"])          # criterio del órgano: SALIENCIA
    por_potencia = max(mapa, key=lambda m: m["pot"])     # criterio ingenuo: POTENCIA
    log("\n## 2) Selección de emisión")
    log("- **Elegida por el órgano (saliencia=%.3f, estructura=%.3f)** → ventana %.1f MHz, emisión en **%.2f MHz**" % (
        elegida["sal"], elegida["est"], elegida["centro"], elegida["fdom"]))
    log("- Habría ganado por POTENCIA cruda → ventana %.1f MHz (pot=%.3f, saliencia sólo %.3f)" % (
        por_potencia["centro"], por_potencia["pot"], por_potencia["sal"]))
    if elegida["centro"] != por_potencia["centro"]:
        log("- ⇒ **el órgano NO elige por energía**: selecciona ESTRUCTURA (una portadora/emisión real), no el pico de potencia.")
    else:
        log("- (en esta banda la de más estructura coincide con la de más potencia)")

    # ---- FASE 3: ENGANCHE ----
    objetivo = elegida["fdom"] * 1e6 if elegida["fdom"] > 1 else elegida["centro"] * 1e6
    log("\n## 3) Enganche — sintonizo %.2f MHz y sostengo %.0f s (¿se queda clavado?)" % (objetivo / 1e6, HOLD))
    log("| t (s) | freq dom (MHz) | Δ vs objetivo (kHz) | estructura | saliencia |")
    log("|---|---|---|---|---|")
    fdoms = []; t0 = time.time(); prox = 0
    lector.sintonizar(objetivo)
    while time.time() - t0 < HOLD:
        fila = {}; lector.inyectar(fila)
        if fila.get("sdr_vivo"):
            obs = radio.observar(fila)
            # el lazo real: el órgano vuelve a ordenar (se re-centra en lo más saliente de la ventana)
            orden = obs.get("radio_orden_hz")
            if orden is not None: lector.sintonizar(orden)
            fdom = (obs.get("radio_freq_dom_hz") or 0) / 1e6
            if time.time() - t0 >= prox:
                dks = (fdom * 1e6 - objetivo) / 1e3
                log("| %4.0f | %.3f | %+.0f | %.3f | %.3f |" % (
                    time.time() - t0, fdom, dks, obs.get("radio_estructura") or 0, obs.get("radio_saliencia") or 0))
                prox += 5
            fdoms.append(fdom)
        time.sleep(0.5)

    # veredicto de lock: dispersión de freq_dom pequeña ⇒ enganchado
    disp = (max(fdoms) - min(fdoms)) if fdoms else 99
    log("\n## Veredicto")
    log("- Dispersión de freq_dom durante el enganche: **%.3f MHz** (%d muestras)." % (disp, len(fdoms)))
    if disp < 0.3:
        log("- ✅ **ENGANCHADO**: tras barrer, el organismo se quedó clavado en la emisión seleccionada (no deriva).")
    else:
        log("- ⚠️ freq_dom sigue moviéndose (disp %.2f MHz): o la ventana tiene varias emisiones que compiten, o no hay una portadora dominante clara." % disp)
    log("\n**Lectura:** Fase 1 = el organismo BARRE (mueve el LO por la banda). Fase 2 = SELECCIONA por")
    log("estructura, no por potencia (agencia perceptiva, no medidor de energía). Fase 3 = se ENGANCHA")
    log("y sostiene. Eso es 'oír activo': percibir haciendo algo, el linaje del oído pasivo → buscar.")
    lector.cerrar()

if __name__ == "__main__":
    main()
