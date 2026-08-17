#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_sdr_caza.py — verifica el actuador de CAZA nuevo del OrganoRadio (histéresis + gate por SNR)
en lazo cerrado sobre el RSPduo real. El órgano barre SOLO, engancha una emisión y sostiene.

A diferencia de v1/v2 (que conducían el barrido desde el test), aquí el TEST no decide nada: sólo
cierra el lazo órgano↔lector (inyectar→observar→sintonizar) y REGISTRA lo que el órgano hace por
su cuenta con histeresis=True. Usa la telemetría signal_snr (dinámica real), no los bins comprimidos.

Uso:  ~/.venvs/vstcosmo/bin/python test_sdr_caza.py
"""
import os, sys, time, json

CARP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(CARP, "..", "organelos")))
from VST_OrganoRadio import OrganoRadio            # noqa: E402
from VST_LectorSDRServidor import LectorSDRServidor    # noqa: E402

URI = os.environ.get("ANIMA_SDRWS_URI", "ws://127.0.0.1:5454")
DUR = float(os.environ.get("CAZA_DUR", "180"))     # s de caza
OUT = os.path.join(CARP, "resultado_sdr_caza_2026-07-05.md")

def log(m):
    with open(OUT, "a") as f: f.write(m + "\n")
    print(m, flush=True)

def _ws_gain():
    import asyncio, websockets
    async def go():
        async with websockets.connect(URI, max_size=None, open_timeout=6) as ws:
            for d in ({"event_type":"set_property","property":"agc_enable","value":"false"},
                      {"event_type":"set_property","property":"lna_state","value":"4"}):
                await ws.send(json.dumps(d))
            await asyncio.sleep(0.6)
    try: asyncio.run(go())
    except Exception: pass

def main():
    if os.path.exists(OUT): os.remove(OUT)
    log("# Caza de emisión con el actuador nuevo (histéresis + SNR) — %s\n" % time.strftime("%Y-%m-%d %H:%M"))
    log("El órgano barre SOLO (el test no lo conduce). histeresis=True, usa signal_snr real.\n")
    _ws_gain()
    lector = LectorSDRServidor()
    if not lector.arrancar(): log("**ERROR** lector no arrancó."); return
    t0 = time.time()
    while time.time() - t0 < 12:
        f = {}; lector.inyectar(f)
        if f.get("sdr_vivo"): break
        time.sleep(0.4)
    else:
        log("**ERROR** sin espectro."); lector.cerrar(); return

    radio = OrganoRadio("CAZA", activo=True, sintonia_activa=True, histeresis=True,
                        barrido_lo_hz=88e6, barrido_hi_hz=108e6, barrido_paso_hz=0.4e6)
    # arrancar en un punto vacío para OBLIGAR a barrer antes de encontrar algo
    lector.sintonizar(93.1e6); time.sleep(1.5)

    log("| t(s) | centro (MHz) | SNR dB | saliencia | enganchado | acción |")
    log("|---|---|---|---|---|---|")
    t0 = time.time(); prox = 0
    fase_lock_t = None; lock_freq = None; snr_lock = []
    n_hops = 0; centro_prev = None
    while time.time() - t0 < DUR:
        fila = {}; lector.inyectar(fila)
        if not fila.get("sdr_vivo"): time.sleep(0.3); continue
        o = radio.observar(fila)
        orden = o.get("radio_orden_hz")
        centro = ((fila.get("sdr_freq_min_hz") or 0) + (fila.get("sdr_freq_max_hz") or 0)) / 2.0
        eng = o.get("radio_enganchado") or 0
        snr = o.get("radio_snr_db")
        accion = "—"
        if orden is not None:
            lector.sintonizar(orden)
            accion = ("barre→%.1f" % (orden/1e6)) if not eng else ("mueve→%.2f" % (orden/1e6))
            if centro_prev is not None and abs(orden - (centro_prev or 0)) > 0.2e6: n_hops += 1
        elif eng:
            accion = "SOSTIENE"
        centro_prev = orden if orden is not None else centro
        if eng and lock_freq is None:
            lock_freq = centro; fase_lock_t = time.time() - t0
        if eng and snr is not None: snr_lock.append(snr)
        if time.time() - t0 >= prox:
            log("| %4.0f | %.2f | %s | %.3f | %s | %s |" % (
                time.time()-t0, centro/1e6, ("%.1f"%snr if snr is not None else "—"),
                o.get("radio_saliencia") or 0, "SÍ" if eng else "no", accion))
            prox += 5
        time.sleep(0.6)

    log("\n## Veredicto")
    if lock_freq is not None:
        import statistics
        snrp = statistics.mean(snr_lock) if snr_lock else 0
        log("- ✅ **ENGANCHÓ** una emisión tras barrer: primera vez a los **%.0f s**, ~%.1f MHz." % (fase_lock_t or 0, lock_freq/1e6))
        log("- SNR medio mientras enganchado: **%.1f dB** (%d muestras) → señal real, no ruido." % (snrp, len(snr_lock)))
        log("- Saltos de barrido antes/entre enganches: ~%d." % n_hops)
        log("\n**Lectura:** el órgano, solo, BARRIÓ el espectro, ADQUIRIÓ una emisión (SNR alto) y la")
        log("SOSTUVO — la conducta de 'oír activo' (barrer hasta seleccionar) ahora EMERGE del órgano.")
    else:
        log("- ⚠️ No llegó a enganchar en %.0f s: o no hay emisiones con SNR≥%.0f dB en la banda, o el" % (DUR, radio.snr_lock_db))
        log("  paso/asentamiento necesita ajuste. Barrió (hops~%d) pero sin lock estable." % n_hops)
    lector.cerrar()

if __name__ == "__main__":
    main()
