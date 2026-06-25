#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OBSERVACIÓN LONGITUDINAL — la díada vive y nosotros sólo MIRAMOS (no intervenimos).
Sondea /estado de A y B cada POLL s durante DURACION_S s, registra las variables INSTRUMENTALES
(expectativa, agencia, intención + valor ecológico) y reporta CUÁL abandona primero el nivel basal.
Arquitectura CONGELADA: esto NO construye nada; sólo observa. Corrida corta para verificar; luego, largo.

  DURACION_S (def 600=10min) · POLL (def 5s) · UMBRAL (def 0.05 = "salió de basal")
Para el estudio largo: DURACION_S=86400 POLL=60 (la biografía completa ya queda en Docker_Historia).
"""
import os, sys, json, time, urllib.request

A_URL = os.environ.get("ANIMA_A_URL", "http://localhost:7788")
B_URL = os.environ.get("ANIMA_B_URL", "http://localhost:7799")
DURACION = float(os.environ.get("DURACION_S", "600"))
POLL = float(os.environ.get("POLL", "5"))
UMBRAL = float(os.environ.get("UMBRAL", "0.05"))
OUT = os.environ.get("OUT", "/tmp/observacion_longitudinal.csv")

# Variables instrumentales que vigilamos (la genealogía: expectativa → agencia → intención).
VARS = ["expectativa", "expectativa_confianza", "expectativa_exploracion",
        "alt_contingencia_social", "alt_agencia_otro", "alt_intencion_comunicativa",
        "voz_otro_valor_ecologico", "voz_otro_confianza_ecologica", "OI"]

def estado(url):
    try:
        with urllib.request.urlopen(url + "/estado", timeout=3) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return {}

def num(v):
    try:
        return float(v)
    except Exception:
        return float("nan")

def main():
    print(f"[longitudinal] {DURACION:.0f}s · poll {POLL}s · umbral basal {UMBRAL} · salida {OUT}", flush=True)
    t0 = time.time(); n = 0
    pico = {f"{lado}:{v}": 0.0 for lado in ("A", "B") for v in VARS}
    primera_salida = {}     # clave -> (t_rel, valor) cuando supera UMBRAL por primera vez
    fh = open(OUT, "w", encoding="utf-8")
    fh.write("t_rel,lado," + ",".join(VARS) + "\n")
    while time.time() - t0 < DURACION:
        tr = round(time.time() - t0, 1)
        for lado, url in (("A", A_URL), ("B", B_URL)):
            d = estado(url)
            vals = [num(d.get(v)) for v in VARS]
            fh.write(f"{tr},{lado}," + ",".join(f"{x:.5f}" if x == x else "" for x in vals) + "\n")
            for v, x in zip(VARS, vals):
                if x == x:
                    k = f"{lado}:{v}"
                    if x > pico[k]:
                        pico[k] = x
                    if v != "OI" and v.split("_")[-1] not in ("confianza", "exploracion") and x > UMBRAL and k not in primera_salida:
                        primera_salida[k] = (tr, round(x, 4))
        fh.flush(); n += 1
        if n % 12 == 0:
            print(f"[longitudinal] t={tr:.0f}s · muestras={n} · salidas de basal hasta ahora: {len(primera_salida)}", flush=True)
        time.sleep(POLL)
    fh.close()

    print("\n" + "=" * 78)
    print(f"OBSERVACIÓN LONGITUDINAL — {n} muestras en {round(time.time()-t0)}s")
    print("=" * 78)
    print("  PICO por variable instrumental (máximo alcanzado · A / B):")
    for v in VARS:
        print(f"    {v:30s} A={pico['A:'+v]:.4f}   B={pico['B:'+v]:.4f}")
    print("\n  ¿QUÉ variable abandonó primero el nivel basal (> {:.2f})?".format(UMBRAL))
    if primera_salida:
        for k, (tr, x) in sorted(primera_salida.items(), key=lambda kv: kv[1][0]):
            print(f"    {tr:7.1f}s   {k}  = {x}")
    else:
        print("    NINGUNA salió del basal en esta corrida (esperable en una ventana corta).")
    print("\n  (Biografía COMPLETA en Docker_Historia; este CSV es el resumen instrumental: " + OUT + ")")

if __name__ == "__main__":
    main()
