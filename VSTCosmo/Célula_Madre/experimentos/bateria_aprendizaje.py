#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BATERÍA — Organelo de Aprendizaje Organísmico (OAO). Verifica que conecta percepción↔expresión por HISTORIA:
memoria ecoica de lo oído (no audio), IMITACIÓN emergente (atractor hacia lo oído, NUNCA copia, con variación),
LIBERTAD FUNCIONAL (aprender es probabilístico: puede NO), y OLVIDO. No prueba lenguaje.
"""
import os, sys
AQUI = os.path.dirname(os.path.abspath(__file__)); RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)
[sys.path.insert(0, os.path.join(RAIZ, _d)) for _d in ("genoma", "campo", "organelos", "diada", "web", "audio") if os.path.isdir(os.path.join(RAIZ, _d))]
import numpy as np
from VST_Aprendizaje import OrganeloAprendizajeOrganismico
from VST_Expresion import OrganoExpresion

def fila(t=0.0, energia=0.5, mundo=0.0):
    return {"t": t, "met_energia": energia, "act_fatiga": 0, "energia_L": mundo, "energia_R": mundo,
            "voz_arousal": 0.2, "voz_valence": 0.0, "necesidad": 0.3, "OI": 0.3, "H_homeostasis": 0.6,
            "expectativa": 0, "alt_intencion_comunicativa": 0, "alt_otro_presente": 1,
            "act_atencion_L": 0.1, "act_atencion_R": 0.1, "act_orientacion_deg": 0, "act_confianza": 0.3,
            "voz_otro_valor_ecologico": 0, "alt_contingencia_social": 0}

def par(gf, gi=0.0, gp=0.0, gr=0.0):
    return {"fila": {"g_freq": gf, "g_intensidad": gi, "g_pausa": gp, "g_repeticion": gr}, "ok": True}

res = []
def chk(n, c, extra=""):
    res.append((n, bool(c))); print(f"  {'PASS' if c else 'FALLA'}  {n}{('  · ' + extra) if extra else ''}")

print("=" * 84); print("BATERÍA — Organelo de Aprendizaje Organísmico (OAO)"); print("=" * 84)

# (1) MEMORIA ECOICA: oír (energía>umbral) almacena la ESTRUCTURA de lo oído; el sesgo apunta hacia ahí
o = OrganeloAprendizajeOrganismico("X1", p_aprender=1.0)
for k in range(40):
    o.observar(fila(t=k * 0.1, mundo=0.5), par(gf=0.8, gi=0.6))
b = o.bias_imitacion()
chk("MEMORIA ECOICA: lo oído se almacena y el sesgo apunta hacia su estructura", b is not None and b[0] > 0.4,
    f"bias≈{np.round(b,2).tolist() if b is not None else None}")

# (2) NO almacena audio: las entradas ecoicas son vectores de 4 (estructura), no señal/WAV
ok_estruct = all(isinstance(e[1], np.ndarray) and e[1].shape == (4,) for e in o.echoica)
chk("NO almacena audio: memoria ecoica = vectores de estructura (no WAV)", ok_estruct,
    f"entradas ecoicas: {len(o.echoica)} vectores de 4")

# (3) OLVIDO: si deja de oír, la memoria ecoica DECAE (el sesgo se desvanece)
mag_antes = float(np.linalg.norm(o.bias_imitacion()))
for k in range(120):
    o.observar(fila(t=(4 + k) * 0.1, mundo=0.0), None)   # silencio: ya no oye nada
mag_desp = o.bias_imitacion()
mag_desp = float(np.linalg.norm(mag_desp)) if mag_desp is not None else 0.0
chk("OLVIDO: sin oír, la memoria ecoica decae (sesgo ↓)", mag_desp < mag_antes,
    f"|bias| {mag_antes:.3f} → {mag_desp:.3f}")

# (4) LIBERTAD FUNCIONAL: aprender es PROBABILÍSTICO — no toda percepción se incorpora (a veces NO aprende)
o2 = OrganeloAprendizajeOrganismico("X2", p_aprender=0.5)
aprend = sum(o2.observar(fila(t=k * 0.1, mundo=0.5), par(gf=0.5))["oao_aprendio"] for k in range(400))
chk("LIBERTAD FUNCIONAL: aprender es probabilístico (puede NO incorporar lo oído)", 0 < aprend < 400,
    f"incorporó {aprend:.0f}/400 percepciones (resto: NO aprendió)")

# (5) IMITACIÓN EMERGE (no copia): la expresión, con sesgo hacia lo oído (freq alta), DERIVA hacia ahí…
oao = OrganeloAprendizajeOrganismico("Ximit", p_aprender=1.0)
ex = OrganoExpresion("Ximit", baseline_voz=1.0, baseline_sil=0.05)   # forzar vocalización para medir el gesto
exc = OrganoExpresion("Xctrl", baseline_voz=1.0, baseline_sil=0.05)  # control: sin sesgo de imitación
fr_imit = []; fr_ctrl = []
for k in range(2500):
    oao.observar(fila(t=k * 0.1, mundo=0.5), par(gf=0.9, gi=0.0))    # oye gestos de FRECUENCIA ALTA
    a = ex.proximo_gesto(fila(t=k * 0.1, energia=0.8, mundo=0.5), bias_imit=oao.bias_imitacion())
    c = exc.proximo_gesto(fila(t=k * 0.1, energia=0.8, mundo=0.5), bias_imit=None)
    if a["expr_vocalizando"] >= 0.5: fr_imit.append(a["g_freq"])
    if c["expr_vocalizando"] >= 0.5: fr_ctrl.append(c["g_freq"])
mi = np.mean(fr_imit) if fr_imit else 0; mc = np.mean(fr_ctrl) if fr_ctrl else 0
chk("IMITACIÓN EMERGE: la voz DERIVA hacia lo oído (freq↑ vs control), por historia", mi > mc + 0.1,
    f"freq media imitando={mi:.3f} vs control={mc:.3f}")

# (6) …pero NUNCA copia: preserva VARIACIÓN (no colapsa a un único gesto)
import statistics as st
chk("VARIACIÓN preservada: la imitación NO copia (dispersión > 0, no un único gesto)",
    len(set(round(f, 1) for f in fr_imit)) >= 3 and (st.pstdev(fr_imit) > 0.02 if len(fr_imit) > 1 else False),
    f"valores de freq distintos={len(set(round(f,1) for f in fr_imit))} std={st.pstdev(fr_imit):.3f}")

# (7) DIFERENCIACIÓN por historia: oír familias distintas → repertorios vocales distintos
def media_freq_oyendo(gf):
    oa = OrganeloAprendizajeOrganismico(f"o{gf}", p_aprender=1.0); e = OrganoExpresion(f"e{gf}", baseline_voz=1.0, baseline_sil=0.05)
    fs = []
    for k in range(2000):
        oa.observar(fila(t=k * 0.1, mundo=0.5), par(gf=gf))
        a = e.proximo_gesto(fila(t=k * 0.1, energia=0.8, mundo=0.5), bias_imit=oa.bias_imitacion())
        if a["expr_vocalizando"] >= 0.5: fs.append(a["g_freq"])
    return np.mean(fs) if fs else 0
chk("DIFERENCIACIÓN: distinta historia oída → distinto repertorio (no idénticos)",
    abs(media_freq_oyendo(0.9) - media_freq_oyendo(-0.9)) > 0.2,
    f"oyendo +0.9 vs −0.9 → repertorios distintos")

print("-" * 84)
nok = sum(1 for _, p in res if p)
print(f"  RESUMEN: {nok}/{len(res)} PASS")
print("  El OAO conecta lo oído con la expresión por HISTORIA: la imitación EMERGE (no se copia ni se programa);")
print("  aprender es libre (puede NO); se olvida. La aparición de semejanzas/estilos se verifica EN VIVO con mundo.")
sys.exit(0 if nok == len(res) else 1)
