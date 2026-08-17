#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
bateria_hemisferios.py — OrganoHemisferios AISLADO: ¿coexisten R₂ y lateralidad? ¿reúne el calloso?
================================================================================
Falsa la capa hemisférica funcional (VST_OrganoHemisferios, emulación de v121) SIN la célula ni
audio real: se inyectan secuencias de energía por oído (energia_L/energia_R) y se leen las columnas
hemi_*. Motivación (soma-ciego-localizacion): con un solo substrato simétrico, R₂ (predicción rápida)
y lateralidad (lenta) salían del mismo sitio y SE ANULABAN. Dos hemisferios asimétricos con cruce
contralateral + cuerpo calloso selectivo deben hacerlas COEXISTIR.

Esta batería NO está amañada para pasar: usa los readouts que de verdad expresan la dinámica
(divergencia, lateralidad, activación del calloso) y deja que un criterio FALLE si el organelo
no cumple lo que promete. Un FALLA aquí es un negativo honesto, no un bug del test.

CRITERIOS (anti-Shannon, falsables):
  C1  CRUCE CONTRALATERAL:  un drive FUERTE en el oído DERECHO (→ hemisferio rápido) mueve el sistema
      (divergencia↑, calloso↑) MUCHO más que el mismo drive en el oído IZQUIERDO (→ lento). La ruta
      contralateral + la asimetría de τ son reales.
  C2  DIFERENCIACIÓN HP/LP:  en el MISMO oído derecho, un TRANSITORIO produce más respuesta del
      hemisferio rápido (divergencia) que un SOSTENIDO de IGUAL amplitud → el rápido es highpass
      (sensible al cambio), el lento es lowpass.
  C3  COEXISTENCIA / DISOCIACIÓN:  con estímulo mixto, R₂ y lateralidad_func están AMBOS presentes
      en el mismo paso y son disociables (|corr| baja) → NO se anulan (lo que mataba al diseño
      simétrico anterior). NOTA: se reporta la magnitud de R₂ porque el forzamiento dipolar (media≈0)
      la mantiene pequeña (régimen de "autopredicción estable").
  C4  CALLOSO ¿REÚNE? (honesto):  bajo drive SOSTENIDO FUERTE, activar el calloso debe REDUCIR la
      divergencia respecto al control idéntico con ganancia=0. Es el caso de estrés; si falla, el
      calloso no cumple su rol de "reunión" de forma robusta (acople débil sobre Φ(1−Φ²) multiestable).
  C5  INVARIANTE DE SILENCIO:  en silencio (600 pasos) las salidas son FINITAS y ACOTADAS (|ω|≤1):
      el organelo corre indefinidamente sin explotar.
Corre:  venv/bin/python3 experimentos/bateria_hemisferios.py
================================================================================
"""
from __future__ import annotations
import os, sys, json
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__)); RAIZ = os.path.dirname(AQUI)
RES = os.path.join(AQUI, "resultados"); sys.path.insert(0, RAIZ)
[sys.path.insert(0, os.path.join(RAIZ, _d)) for _d in ("genoma", "campo", "organelos", "diada", "web", "audio")
 if os.path.isdir(os.path.join(RAIZ, _d))]   # Célula Madre en subcarpetas
import VST_OrganoHemisferios as H
from VST_OrganoHemisferios import OrganoHemisferios


# ----------------------------------------------------------------------------- estímulo y corrida
def _transitorio(n, amp=0.4, periodo=6):
    """On/off → cambios bruscos = transitorios (lo que el highpass del hemisferio rápido captura)."""
    h = max(1, periodo // 2)
    return [amp if (i // h) % 2 == 0 else 0.0 for i in range(n)]

def _const(n, nivel):
    return [float(nivel)] * n

def _correr(org, eL_seq, eR_seq):
    out = {c: [] for c in H.COLS_HEMI}
    for eL, eR in zip(eL_seq, eR_seq):
        fila = {"energia_L": float(eL), "energia_R": float(eR)}
        org.observar(fila)
        for c in H.COLS_HEMI:
            out[c].append(float(fila[c]))
    return out

def _t(serie, k=200):
    return float(np.mean(np.asarray(serie[-k:], dtype=float)))

def _frac(serie):
    return float(np.mean(np.asarray(serie, dtype=float)))

def _corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main():
    os.makedirs(RES, exist_ok=True)
    N = 400
    print("=" * 90)
    print("OrganoHemisferios AISLADO — cruce contralateral (der→rápido, izq→lento) + calloso selectivo")
    print("=" * 90)

    # ----------------------------------------------------------------- C1: cruce contralateral
    der = _correr(OrganoHemisferios(), _const(N, 0.0), _const(N, 1.0))      # oído DER fuerte → rápido
    izq = _correr(OrganoHemisferios(), _const(N, 1.0), _const(N, 0.0))      # oído IZQ fuerte → lento
    div_der, cal_der = _t(der["hemi_divergencia"]), _frac(der["hemi_calloso_activo"])
    div_izq, cal_izq = _t(izq["hemi_divergencia"]), _frac(izq["hemi_calloso_activo"])
    C1 = (div_der > 1.4 * div_izq) and (cal_der > 2.0 * max(cal_izq, 1e-3))
    print("\n[C1] cruce contralateral (drive fuerte 1.0)")
    print(f"     DER→rápido: div={div_der:.4f} calloso={cal_der*100:.0f}%   ||   IZQ→lento: div={div_izq:.4f} calloso={cal_izq*100:.0f}%")
    print(f"     el oído derecho mueve el sistema {div_der/max(div_izq,1e-6):.1f}× más en divergencia y {cal_der*100:.0f}% vs {cal_izq*100:.0f}% calloso")

    # ----------------------------------------------------------------- C2: diferenciación HP/LP (mismo oído der)
    tr = _correr(OrganoHemisferios(), _const(N, 0.0), _transitorio(N, 0.4, 6))   # transitorio DER
    so = _correr(OrganoHemisferios(), _const(N, 0.0), _const(N, 0.4))            # sostenido DER (igual amp)
    div_tr, div_so = _t(tr["hemi_divergencia"]), _t(so["hemi_divergencia"])
    C2 = div_tr > 1.3 * div_so
    print("\n[C2] diferenciación temporal HP/LP (mismo oído DER, amplitud 0.4)")
    print(f"     transitorio div={div_tr:.4f}  >  sostenido div={div_so:.4f}   → el rápido responde al CAMBIO (highpass)")

    # ----------------------------------------------------------------- C3: coexistencia / disociación
    mix = _correr(OrganoHemisferios(), _const(N, 0.35), _transitorio(N, 0.4, 6))
    R2_s = np.asarray(mix["hemi_R2"][-200:]); lat_s = np.asarray(mix["hemi_lateralidad_func"][-200:])
    R2_act, lat_act = float(R2_s.mean()), float(np.abs(lat_s).mean())
    r_R2_lat = _corr(R2_s, lat_s)
    C3 = (R2_act > 1e-4) and (lat_act > 1e-3) and (abs(r_R2_lat) < 0.7)
    print("\n[C3] coexistencia / disociación (estímulo mixto)")
    print(f"     R₂={R2_act:.4f} (pequeña: forzamiento dipolar)  ·  |lateralidad|={lat_act:.4f}  ·  corr={r_R2_lat:+.3f} (|corr|<0.7 ⇒ disociables)")

    # ----------------------------------------------------------------- C4: ¿el calloso REÚNE? (honesto)
    estim_estres = (_const(N, 0.0), _const(N, 1.0))                 # sostenido fuerte: caso de estrés
    on = _correr(OrganoHemisferios(), *estim_estres); div_on = _t(on["hemi_divergencia"]); cal_on = _frac(on["hemi_calloso_activo"])
    g0 = H.GANANCIA_CALLOSO; H.GANANCIA_CALLOSO = 0.0
    off = _correr(OrganoHemisferios(), *estim_estres); div_off = _t(off["hemi_divergencia"])
    H.GANANCIA_CALLOSO = g0
    # referencia: bajo transitorios sí reúne (contexto, no criterio)
    on_tr = _correr(OrganoHemisferios(), _const(N, 0.0), _transitorio(N, 0.6, 10)); div_on_tr = _t(on_tr["hemi_divergencia"])
    H.GANANCIA_CALLOSO = 0.0
    off_tr = _correr(OrganoHemisferios(), _const(N, 0.0), _transitorio(N, 0.6, 10)); div_off_tr = _t(off_tr["hemi_divergencia"])
    H.GANANCIA_CALLOSO = g0
    C4 = div_on < div_off
    print("\n[C4] ¿el cuerpo calloso REÚNE? (reduce divergencia vs ganancia=0)")
    print(f"     SOSTENIDO fuerte (estrés): div_ON={div_on:.4f}  vs  div_OFF={div_off:.4f}  → reúne={C4}  (calloso activo {cal_on*100:.0f}%)")
    print(f"     [contexto] TRANSITORIO:    div_ON={div_on_tr:.4f}  vs  div_OFF={div_off_tr:.4f}  → reúne={div_on_tr<div_off_tr}")
    if not C4:
        print("     ⚠ el calloso NO reduce la divergencia de forma robusta: acople débil sobre Φ(1−Φ²) multiestable")

    # ----------------------------------------------------------------- C5: invariante de silencio
    sil = _correr(OrganoHemisferios(), _const(600, 0.0), _const(600, 0.0))
    todos = [v for c in H.COLS_HEMI for v in sil[c]]
    finito = bool(np.all(np.isfinite(todos)))
    acotado = bool(np.all(np.abs(np.asarray(sil["hemi_rapido_omega"]) + np.asarray(sil["hemi_lento_omega"])) <= 1.0 + 1e-6))
    cal_sil = _frac(sil["hemi_calloso_activo"])
    C5 = finito and acotado
    print("\n[C5] invariante de silencio (600 pasos)")
    print(f"     finito={finito} · |ω|≤1={acotado} · calloso basal={cal_sil*100:.0f}% (activación intrínseca por asimetría de semillas)")

    # ----------------------------------------------------------------- veredicto
    ver = {"C1_cruce_contralateral": C1, "C2_diferenciacion_HP_LP": C2,
           "C3_coexistencia_disociacion": C3, "C4_calloso_reune": C4, "C5_invariante_silencio": C5}
    nombres = {"C1_cruce_contralateral": "oído der→rápido domina sobre izq→lento (ruta + τ asimétrico)",
               "C2_diferenciacion_HP_LP": "el rápido responde al transitorio más que al sostenido (highpass)",
               "C3_coexistencia_disociacion": "R₂ y lateralidad coexisten y son disociables (no se anulan)",
               "C4_calloso_reune": "el calloso reduce la divergencia (reunión robusta)",
               "C5_invariante_silencio": "estable y acotado en silencio (corre indefinido)"}
    print("\n" + "=" * 90)
    for k, v in ver.items():
        print(f"  {'PASS' if v else 'FALLA'}  {nombres[k]}")
    print(f"\n  RESUMEN: {sum(ver.values())}/{len(ver)} PASS  (un FALLA en C4 es un negativo honesto: ver informe)")
    payload = {"veredicto": ver,
               "C1": {"div_der": div_der, "cal_der": cal_der, "div_izq": div_izq, "cal_izq": cal_izq},
               "C2": {"div_transitorio": div_tr, "div_sostenido": div_so},
               "C3": {"R2_act": R2_act, "lat_act": lat_act, "corr_R2_lat": r_R2_lat},
               "C4": {"div_on": div_on, "div_off": div_off, "cal_on": cal_on,
                      "div_on_transitorio": div_on_tr, "div_off_transitorio": div_off_tr},
               "C5": {"finito": finito, "acotado": acotado, "calloso_basal": cal_sil}}
    with open(os.path.join(RES, "bateria_hemisferios.json"), "w", encoding="utf-8") as fj:
        json.dump(payload, fj, ensure_ascii=False, indent=1, default=float)
    print(f"  → {os.path.join(RES, 'bateria_hemisferios.json')}")


if __name__ == "__main__":
    main()
