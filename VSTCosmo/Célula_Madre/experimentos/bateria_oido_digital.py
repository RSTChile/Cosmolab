#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BATERÍA DEL OÍDO DIGITAL (VST_OrganoOidoDigital)
=================================================
Verifica adversarialmente que el oído digital:
  1) ROBUSTEZ   — no lanza con fila sin nrf / nrf_last_rx vacío / nrf_rx_delta ausente.
  2) OFF        — con activo=False la salida es EXACTAMENTE _NEUTRO (idempotente, cero cambio).
  3) FALSACIÓN  — REAL responde y COLAPSA en NULL y SHUFFLED:
                    · fiabilidad(REAL) > fiabilidad(SHUFFLED) ≈ fiabilidad(NULL)=0
                    · valor_ecológico(REAL) > valor(SHUFFLED) ≈ valor(NULL)=0
                    · modulación(NULL)=1.0 exacto; |mod-1| DÉBIL (≤ cap 0.15) siempre.
  4) DÉBIL      — la modulación jamás excede el tope duro ±cap_mod.

Escenario sintético: E (el otro) tiene un arousal que OSCILA. En REAL, E emite un símbolo
digital 'ping-alto' cuando su arousal es alto y 'ping-bajo' cuando es bajo (el símbolo COVARÍA
con el estado real del otro, y atender ANTECEDE una mejora de persistencia del receptor).
En SHUFFLED el mismo flujo de símbolos se baraja contra el estado (rompe la covariación).
En NULL no llega ningún símbolo.
"""
import os, sys, math, random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "organelos"))
from VST_OrganoOidoDigital import OrganoOidoDigital, _NEUTRO, COLS_OIDO_DIG  # noqa

DT = 0.1
N = 1200
random.seed(7)
BLOQUE = 40          # ticks por régimen (4.0 s) >> ventana del espejo (0.6 s): estado estable en la ventana
VENTANA = 0.6        # ventana corta del espejo: la mayoría de las ventanas caen dentro de UN solo régimen


def corre(modo):
    """Devuelve (org, series_self, series_otro, out_final, fiab_peak, valor_peak).
    Escenario en BLOQUES: el otro alterna régimen alto/bajo cada BLOQUE ticks; el estado PERSISTE
    a lo largo de la ventana del espejo, así el forward-model puede separar la firma de la línea-base."""
    org = OrganoOidoDigital("A", activo=True, ventana=VENTANA)
    a_self_series, a_otro_series = [], []

    # régimen por bloque (alto/bajo), y estado del otro CONSTANTE dentro del bloque (+ruido leve)
    estados_otro, regimen = [], []
    for i in range(N):
        alto = ((i // BLOQUE) % 2 == 0)
        a_otro = (0.9 if alto else 0.1) + 0.03 * random.uniform(-1, 1)
        v_otro = (0.7 if alto else 0.3) + 0.03 * random.uniform(-1, 1)
        estados_otro.append((a_otro, v_otro))
        regimen.append(alto)
    # símbolo que COVARÍA con el régimen real del otro (2 tipos + contador que se colapsa a '#')
    simbolos = [("ping-alto %d" % i) if regimen[i] else ("ping-bajo %d" % i) for i in range(N)]
    if modo == "shuffled":
        random.shuffle(simbolos)   # rompe la relación símbolo↔estado, MISMOS tiempos/contenido

    a_self = 0.5
    prev_rx = ""
    fiab_peak = valor_peak = 0.0
    out = dict(_NEUTRO)
    for i in range(N):
        a_otro, v_otro = estados_otro[i]
        emite = True                # el enlace nRF entrega un paquete cada tick (canal 100%, según el test 5-jul)
        rx = simbolos[i] if (modo != "null" and emite) else prev_rx
        delta = 1 if (modo != "null" and emite) else 0
        vivo = 0 if modo == "null" else 1

        fila = {
            "t": i * DT,
            "voz_arousal": a_self, "voz_valence": 0.4,
            "A_sys_env": 0.3 + 0.4 * a_self, "ICR": 0.2, "met_energia": 0.6,
            "necesidad": 0.2, "H_homeostasis": 0.5, "act_perm": 0.7,
            "nrf_last_rx": rx, "nrf_rx_delta": delta, "nrf_vivo": vivo,
        }
        # estado del otro por AUDIO (target del espejo). En NULL no hay co-presencia acústica.
        estado_otro = None if modo == "null" else {"fila": {"voz_arousal": a_otro, "voz_valence": v_otro}}
        out = org.observar(fila, estado_otro, dt=DT)

        # el arousal del receptor sigue al del otro con ganancia = (mod-1): la modulación ABRE el oído.
        gan = (out["oido_dig_modulacion"] - 1.0)
        a_self = 0.5 + 6.0 * gan * (a_otro - 0.5) + 0.02 * random.uniform(-1, 1)
        a_self = max(0.0, min(1.0, a_self))

        a_self_series.append(a_self)
        a_otro_series.append(a_otro)
        fiab_peak = max(fiab_peak, org.fiabilidad)
        valor_peak = max(valor_peak, org.valor_max())
        prev_rx = rx
    return org, a_self_series, a_otro_series, out, fiab_peak, valor_peak


def corr(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0 or sy == 0:
        return 0.0
    return sum((x[i] - mx) * (y[i] - my) for i in range(n)) / (sx * sy)


def main():
    fallas = []

    # ---- 1) ROBUSTEZ ------------------------------------------------------
    org = OrganoOidoDigital("A", activo=True)
    for fila in ({}, {"t": 0.0}, {"nrf_last_rx": ""}, {"nrf_rx_delta": None},
                 {"nrf_last_rx": None, "t": 1.0}, {"nrf_last_rx": "x", "nrf_rx_delta": 1}):
        try:
            r = org.observar(fila, None, dt=DT)
            assert isinstance(r, dict) and "oido_dig_modulacion" in r
        except Exception as e:
            fallas.append("ROBUSTEZ lanzó con %r: %r" % (fila, e))
    print("[1] ROBUSTEZ: no lanza con filas sin nrf / vacías ................ OK")

    # ---- 2) OFF idempotente ----------------------------------------------
    off = OrganoOidoDigital("A", activo=False)
    r_off = off.observar({"nrf_last_rx": "ping-alto 3", "nrf_rx_delta": 1, "nrf_vivo": 1}, None)
    if r_off != _NEUTRO:
        fallas.append("OFF no devolvió _NEUTRO: %r" % r_off)
    print("[2] OFF: activo=False ⇒ salida == _NEUTRO (mod=%.3f) ............ %s"
          % (r_off["oido_dig_modulacion"], "OK" if r_off == _NEUTRO else "FALLA"))

    # ---- 3) FALSACIÓN real vs null vs shuffled ---------------------------
    res = {}
    for modo in ("real", "shuffled", "null"):
        org, a_self, a_otro, last, fiab_peak, valor_peak = corre(modo)
        h = len(a_self) // 2                    # correlación de acople en la SEGUNDA mitad (tras aprender)
        res[modo] = {
            "fiab": fiab_peak,
            "valor": valor_peak,
            "conf": org.confianza,
            "r_acople": corr(a_self[h:], a_otro[h:]),
            "mod_last": last["oido_dig_modulacion"],
            "n_sym": len(org.n_sym),
            "eventos": org.eventos_n,
        }

    print("\n[3] FALSACIÓN (real vs shuffled vs null)  [fiab/valor = PICO; r = 2a mitad]:")
    print("    %-10s %9s %9s %9s %9s %6s" % ("modo", "fiab_pk", "valor_pk", "conf", "r_acopl", "nsym"))
    for m in ("real", "shuffled", "null"):
        d = res[m]
        print("    %-10s %9.4f %9.4f %9.4f %9.4f %6d"
              % (m, d["fiab"], d["valor"], d["conf"], d["r_acople"], d["n_sym"]))

    # criterios de falsación
    if not (res["null"]["fiab"] == 0.0 and res["null"]["valor"] == 0.0):
        fallas.append("NULL no colapsó (fiab=%.4f valor=%.4f)" % (res["null"]["fiab"], res["null"]["valor"]))
    if res["null"]["mod_last"] != 1.0:
        fallas.append("NULL modulación != 1.0 (%.4f)" % res["null"]["mod_last"])
    if not (res["real"]["fiab"] > 2.0 * res["shuffled"]["fiab"] + 1e-6):
        fallas.append("fiabilidad REAL (%.4f) no supera claramente SHUFFLED (%.4f)"
                      % (res["real"]["fiab"], res["shuffled"]["fiab"]))
    if not (res["real"]["valor"] >= res["shuffled"]["valor"]):
        fallas.append("valor REAL (%.4f) no >= SHUFFLED (%.4f)" % (res["real"]["valor"], res["shuffled"]["valor"]))
    if not (res["real"]["r_acople"] > res["shuffled"]["r_acople"] + 0.1 and res["real"]["r_acople"] > 0.2):
        fallas.append("acople r REAL (%.4f) no supera SHUFFLED (%.4f)/umbral"
                      % (res["real"]["r_acople"], res["shuffled"]["r_acople"]))

    # ---- 4) DÉBIL: tope duro nunca superado ------------------------------
    cap = OrganoOidoDigital("A").cap_mod
    # gate SATURADO (valor y fiabilidad forzados a valores enormes) no puede exceder el cap
    org = OrganoOidoDigital("A")
    org.valor["x"] = 1e6; org.fiabilidad = 1e6
    out_sat = org.observar({"t": 0.0, "nrf_last_rx": "x", "nrf_rx_delta": 1, "nrf_vivo": 1,
                            "act_perm": 1.0, "necesidad": 0.0, "voz_arousal": 0.5, "voz_valence": 0.5},
                           {"fila": {"voz_arousal": 0.9, "voz_valence": 0.9}})
    if abs(out_sat["oido_dig_modulacion"] - 1.0) > cap + 1e-9:
        fallas.append("modulación excede el cap con gate saturado: %.4f" % out_sat["oido_dig_modulacion"])
    if not (0.0 < cap <= 0.2):
        fallas.append("cap_mod fuera de rango débil: %.3f" % cap)
    print("\n[4] DÉBIL: |modulación-1| capada a ±cap_mod=%.2f; gate saturado ⇒ mod=%.4f ... OK"
          % (cap, out_sat["oido_dig_modulacion"]))

    # ---- veredicto -------------------------------------------------------
    print("\n" + "=" * 64)
    if fallas:
        print("RESULTADO: FALLA")
        for f in fallas:
            print("  - " + f)
        sys.exit(1)
    print("RESULTADO: PASA — el oído responde al REAL y colapsa en NULL/SHUFFLED,")
    print("           OFF idempotente, robusto a filas sin nrf, modulación DÉBIL.")
    sys.exit(0)


if __name__ == "__main__":
    main()
