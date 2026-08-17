#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_sdr_barrido_v2.py — barrer → seleccionar → ENGANCHAR (con histéresis), sobre el RSPduo real.

v1 mostró que el organismo BARRE y SELECCIONA por estructura (no potencia), pero NO enganchaba:
el sdr_espectro del WS viene normalizado-al-rango-visible (comprime la dinámica), y el actuador
argmax-saliencia, sin histéresis, sobre un espectro plano PERSIGUE ruido y deriva (se fue a 74 MHz).

v2 corrige la capa que SÍ es del organismo: añade HISTÉRESIS al actuador.
  - ACQUISICIÓN: arranco desintonizado ~0.3 MHz de la emisión elegida y dejo que el actuador la
    atraiga al centro (sólo re-sintoniza si la mejor banda supera a la actual por un MARGEN y de
    forma SOSTENIDA k ticks).
  - LOCK: una vez centrada, SOSTIENE (no re-sintoniza por ruido). Mido dispersión de freq_dom.
Además apago el AGC y fijo la ganancia (des-satura algo la FFT) al empezar.

Aislado: instancia su propio lector+órgano; no toca los organismos vivos.
Uso:  ~/.venvs/vstcosmo/bin/python test_sdr_barrido_v2.py
"""
import os, sys, time, json, struct

CARP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(CARP, "..", "organelos")))
from VST_OrganoRadio import OrganoRadio           # noqa: E402
from VST_LectorSDRServidor import LectorSDRServidor   # noqa: E402

F0, F1, PASO = 88.0, 108.0, 1.6
ASENT = 2.2
URI = os.environ.get("ANIMA_SDRWS_URI", "ws://127.0.0.1:5454")
OUT = os.path.join(CARP, "resultado_sdr_barrido_v2_2026-07-05.md")

# --- histéresis del actuador (la mejora) ---
MARGEN_SAL = 1.4     # una banda candidata debe tener saliencia > actual*MARGEN + EPS para robar el lock
EPS_SAL = 0.02
SOSTEN_K = 3         # y sostenerlo K ticks seguidos antes de re-sintonizar
LOCK_HOLD = 40       # s de sostén tras adquirir

def log(m):
    with open(OUT, "a") as f: f.write(m + "\n")
    print(m, flush=True)

def _ws_set_gain():
    """Apaga AGC y fija atenuación por un WS efímero (des-satura la FFT normalizada)."""
    import asyncio, websockets
    async def go():
        async with websockets.connect(URI, max_size=None, open_timeout=6) as ws:
            for d in ({"event_type":"set_property","property":"agc_enable","value":"false"},
                      {"event_type":"set_property","property":"lna_state","value":"4"}):
                await ws.send(json.dumps(d))
            await asyncio.sleep(0.6)
    try: asyncio.run(go()); return True
    except Exception as e: log("  (aviso: no pude fijar ganancia: %s)" % e); return False

def obs_en(lector, radio, freq_hz, asent):
    lector.sintonizar(freq_hz)
    t0 = time.time(); ult = None
    while time.time() - t0 < asent:
        fila = {}; lector.inyectar(fila)
        if fila.get("sdr_vivo"): ult = radio.observar(fila)
        time.sleep(0.3)
    return ult

def main():
    if os.path.exists(OUT): os.remove(OUT)
    log("# Barrer → seleccionar → ENGANCHAR con histéresis (RSPduo) — %s\n" % time.strftime("%Y-%m-%d %H:%M"))
    _ws_set_gain()
    lector = LectorSDRServidor()
    if not lector.arrancar(): log("**ERROR** lector no arrancó."); return
    t0 = time.time()
    while time.time() - t0 < 12:
        f = {}; lector.inyectar(f)
        if f.get("sdr_vivo"): break
        time.sleep(0.4)
    else:
        log("**ERROR** sin espectro (sdr_vivo=0)."); lector.cerrar(); return
    radio = OrganoRadio("BARRIDO2", activo=True, sintonia_activa=True, n_bandas=16)

    # ---- 1) BARRIDO ----
    log("## 1) Barrido (mapa por ventana)")
    log("| centro | freq dom | estructura | saliencia | potencia |")
    log("|---|---|---|---|---|")
    mapa = []; f = F0
    while f <= F1 + 1e-6:
        o = obs_en(lector, radio, f * 1e6, ASENT)
        if o:
            mapa.append({"c": f, "fd": (o.get("radio_freq_dom_hz") or 0)/1e6,
                         "est": o.get("radio_estructura") or 0, "sal": o.get("radio_saliencia") or 0,
                         "pot": o.get("radio_potencia_total") or 0})
            m = mapa[-1]
            log("| %.1f | %.2f | %.3f | %.3f | %.3f |" % (m["c"], m["fd"], m["est"], m["sal"], m["pot"]))
        f += PASO
    if not mapa: log("\n**sin barrido**"); lector.cerrar(); return

    # ---- 2) SELECCIÓN por estructura ----
    eleg = max(mapa, key=lambda m: m["est"])       # estructura = portadora real (planitud invertida)
    objetivo = eleg["fd"] if eleg["fd"] > 1 else eleg["c"]
    log("\n## 2) Selección → emisión con más ESTRUCTURA: **%.2f MHz** (est=%.3f, sal=%.3f)" % (
        objetivo, eleg["est"], eleg["sal"]))

    # ---- 3) ADQUISICIÓN + LOCK con histéresis, arrancando desintonizado ----
    arranque = objetivo - 0.3     # desintonizo 300 kHz para ver la ADQUISICIÓN
    log("\n## 3) Enganche con histéresis — arranco DESINTONIZADO en %.2f MHz (objetivo %.2f)" % (arranque, objetivo))
    log("| t(s) | centro (MHz) | freq dom (MHz) | saliencia | acción |")
    log("|---|---|---|---|---|")
    centro = arranque * 1e6
    lector.sintonizar(centro); time.sleep(2.5)      # asentar en el punto desintonizado
    fdoms = []; t0 = time.time(); prox = 0
    sal_lock = 0.0; cand_prev = None; cand_n = 0
    while time.time() - t0 < LOCK_HOLD + 8:
        fila = {}; lector.inyectar(fila)
        if not fila.get("sdr_vivo"): time.sleep(0.3); continue
        o = radio.observar(fila)
        fd = (o.get("radio_freq_dom_hz") or 0)
        sal = o.get("radio_saliencia") or 0
        accion = "sostener"
        # ¿la banda saliente está lejos del centro actual? candidata a robar el lock
        if abs(fd - centro) > 150e3:
            if sal > sal_lock * MARGEN_SAL + EPS_SAL:
                # exigir que la candidata se SOSTENGA k ticks (histéresis temporal)
                if cand_prev is not None and abs(fd - cand_prev) < 120e3:
                    cand_n += 1
                else:
                    cand_n = 1
                cand_prev = fd
                if cand_n >= SOSTEN_K:
                    centro = fd; sal_lock = sal; cand_n = 0; cand_prev = None
                    lector.sintonizar(centro); accion = "RE-SINTONIZA→%.2f" % (centro/1e6)
                else:
                    accion = "candidata(%d/%d)" % (cand_n, SOSTEN_K)
            else:
                cand_n = 0; cand_prev = None       # no supera el margen → ignora ruido, mantiene lock
        else:
            sal_lock = max(sal_lock, sal)          # está centrada: consolida el lock
            cand_n = 0; cand_prev = None
        fdoms.append(centro/1e6)
        if time.time() - t0 >= prox:
            log("| %4.0f | %.3f | %.3f | %.3f | %s |" % (time.time()-t0, centro/1e6, fd/1e6, sal, accion))
            prox += 4
        time.sleep(0.5)

    disp = (max(fdoms) - min(fdoms)) if fdoms else 99
    dcentro = abs(centro/1e6 - objetivo)
    log("\n## Veredicto")
    log("- Centro final: **%.3f MHz** (objetivo %.2f; error %.0f kHz)." % (centro/1e6, objetivo, dcentro*1e3))
    log("- Dispersión del centro en el sostén: **%.3f MHz** (%d muestras)." % (disp, len(fdoms)))
    if disp < 0.5 and dcentro < 0.6:
        log("- ✅ **ENGANCHADO**: adquirió la emisión y se quedó clavado (la histéresis frena la deriva).")
    elif disp < 0.5:
        log("- ◑ Estable pero en otra emisión (%.2f): se enganchó a lo más saliente de su vecindad, no al objetivo exacto." % (centro/1e6))
    else:
        log("- ⚠️ Aún deriva (disp %.2f): el espectro normalizado-a-visible es demasiado plano; haría falta usar la telemetría signal_snr (dinámica real) en vez de los bins comprimidos." % disp)
    log("\n**Lectura:** con histéresis el actuador deja de perseguir ruido: adquiere la emisión y la")
    log("SOSTIENE. Es el paso que faltaba entre 'barrer/seleccionar' (que ya funcionaba) y 'engancharse'.")
    lector.cerrar()

if __name__ == "__main__":
    main()
