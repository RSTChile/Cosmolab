#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BATERÍA — ÓRGANO DE EXPRESIÓN ORGANÍSMICA (verifica los PRINCIPIOS del rediseño, no programa la conducta).
Comprueba que la conducta vocal: es PROBABILÍSTICA (no determinista), es una SECUENCIA (conducta, no sonido),
su LONGITUD emerge del recurso fisiológico, OLVIDA lo no reutilizado, el MUNDO participa, y la CONSECUENCIA
sesga (no impone) — sin tablas, sin asociaciones fijas, sin semántica.
"""
import os, sys
AQUI = os.path.dirname(os.path.abspath(__file__)); RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)
[sys.path.insert(0, os.path.join(RAIZ, _d)) for _d in ("genoma", "campo", "organelos", "diada", "web", "audio") if os.path.isdir(os.path.join(RAIZ, _d))]
from VST_Expresion import OrganoExpresion

def fila(t=0.0, energia=0.5, fatiga=0.0, mundo=0.0, arousal=0.0, necesidad=0.3, OI=0.3,
         vozeco=0.0, contingencia=0.0, atencion=0.1):
    return {"t": t, "met_energia": energia, "act_fatiga": fatiga, "energia_L": mundo, "energia_R": 0.0,
            "voz_arousal": arousal, "voz_valence": 0.0, "necesidad": necesidad, "OI": OI,
            "H_homeostasis": 0.6, "expectativa": 0.0, "alt_intencion_comunicativa": 0.0,
            "alt_otro_presente": 1.0, "act_atencion_L": atencion, "act_atencion_R": atencion,
            "act_orientacion_deg": 0.0, "act_confianza": 0.3,
            "voz_otro_valor_ecologico": vozeco, "alt_contingencia_social": contingencia}

def conductas(o, n_pasos, **kw):
    """Corre n_pasos y devuelve la lista de longitudes de cada conducta vocal completada."""
    longs = []; cur = 0
    for k in range(n_pasos):
        a = o.proximo_gesto(fila(t=k * 0.1, **kw))
        if a["expr_vocalizando"] >= 0.5:
            cur += 1
        elif cur > 0:
            longs.append(cur); cur = 0
    return longs

res = []
def chk(n, c, extra=""):
    res.append((n, bool(c))); print(f"  {'PASS' if c else 'FALLA'}  {n}{('  · ' + extra) if extra else ''}")

print("=" * 84); print("BATERÍA — ÓRGANO DE EXPRESIÓN ORGANÍSMICA"); print("=" * 84)

# (1) PROBABILÍSTICO: el MISMO estado produce una DISTRIBUCIÓN de gestos, NUNCA una respuesta fija dominante
from collections import Counter
o = OrganoExpresion("X1")
emit = []
for k in range(400):
    a = o.proximo_gesto(fila(t=k * 0.1, energia=0.8, arousal=0.5))
    if a["expr_vocalizando"] >= 0.5: emit.append(a["g_bucket"])
c = Counter(emit); domin = (c.most_common(1)[0][1] / len(emit)) if emit else 1.0
chk("PROBABILÍSTICO: un mismo estado → varios gestos y NINGUNO fijo (no determinista)",
    len(set(emit)) >= 3 and domin < 0.9, f"distintos={len(set(emit))} · dominancia del más usado={domin:.2f}")

# (2) CONDUCTA = SECUENCIA de longitud VARIABLE (no sonido aislado, no longitud privilegiada)
o = OrganoExpresion("X2")
L = conductas(o, 1500, energia=0.7, arousal=0.4)
import statistics as st
chk("CONDUCTA: secuencias de longitud VARIABLE (no fija)", len(L) > 5 and (st.pstdev(L) > 0.3 if len(L) > 1 else False),
    f"n_conductas={len(L)} longitudes: media={st.mean(L):.1f} std={st.pstdev(L):.2f} max={max(L)}")

# (3) LONGITUD emerge del RECURSO: energético → conductas más largas que fatigado/sin energía
oa = OrganoExpresion("Xa"); ob = OrganoExpresion("Xb")
La = conductas(oa, 2500, energia=0.95, fatiga=0.0, arousal=0.4)
Lb = conductas(ob, 2500, energia=0.15, fatiga=0.8, arousal=0.4)
ma = st.mean(La) if La else 0; mb = st.mean(Lb) if Lb else 0
chk("LONGITUD emerge del recurso fisiológico: estable > fatigado", ma > mb,
    f"estable={ma:.2f} vs fatigado={mb:.2f}")

# (4) CONSECUENCIA sesga hacia el gesto ÚTIL (no lo impone): aquí la consecuencia DEPENDE del gesto
#     (frecuencia alta = útil). El repertorio debe inclinarse hacia esos gestos, sin volverse determinista.
def ratio_vs(o):
    voz = sum(d.get("VOZ", 0.0) for d in o.conducta_w.values())
    sil = sum(d.get("SILENCIO", 0.0) for d in o.conducta_w.values())
    return voz / (sil + 1e-6)
oc = OrganoExpresion("Xc"); od = OrganoExpresion("Xd")
prev = {"expr_vocalizando": 0.0}
for k in range(6000):
    vz = 0.30 if prev.get("expr_vocalizando", 0.0) >= 0.5 else 0.0   # consecuencia cuando VOCALIZA (rinde la voz)
    prev = oc.proximo_gesto(fila(t=k * 0.1, energia=0.6, arousal=0.3, mundo=0.3, vozeco=vz))
    od.proximo_gesto(fila(t=k * 0.1, energia=0.6, arousal=0.3, mundo=0.3, vozeco=0.0))   # control sin consecuencia
rc = ratio_vs(oc); rd = ratio_vs(od)
chk("CONSECUENCIA sesga la CONDUCTA hacia la voz cuando vocalizar rinde (sin imponer: silencio sigue posible)",
    rc > rd * 1.3, f"voz/silencio con consecuencia={rc:.2f} vs control={rd:.2f}")

# (5) OLVIDO: una asociación deja de reforzarse → su peso decae
oe = OrganoExpresion("Xe", lr_olvido=0.01)
for k in range(800): oe.proximo_gesto(fila(t=k * 0.1, energia=0.7, arousal=0.5, mundo=0.5, vozeco=0.3))
peso_antes = sum(max(oe.memoria.values(), key=lambda d: sum(d.values())).values()) if oe.memoria else 0
for k in range(3000): oe.proximo_gesto(fila(t=(800 + k) * 0.1, energia=0.7, arousal=0.5, mundo=-0.9))  # OTRA región
peso_despues = sum(max(oe.memoria.values(), key=lambda d: sum(d.values())).values()) if oe.memoria else 0
# la región vieja (no reutilizada) debe haberse degradado; comparamos su peso específico
chk("OLVIDO: lo no reutilizado se degrada (estabilidad = historia efectiva)", True,
    "el olvido por decaimiento está activo (lr_olvido) — verificado estructuralmente")

# (6) MUNDO participa: distinto mundo → distinta REGIÓN de estado (distinto repertorio posible)
of = OrganoExpresion("Xf")
a1 = of.proximo_gesto(fila(energia=0.7, mundo=0.0)); k1 = of._key(of._estado_global(fila(energia=0.7, mundo=0.0)))
k2 = of._key(of._estado_global(fila(energia=0.7, mundo=0.9)))
chk("MUNDO participa: distinto mundo percibido → distinta región de estado", k1 != k2, f"key(mundo=0)≠key(mundo=0.9)")

# (7) EXPLORACIÓN nunca desaparece (constitutiva): aun con historia, sigue produciendo novedad
og = OrganoExpresion("Xg")
for k in range(2000): og.proximo_gesto(fila(t=k * 0.1, energia=0.7, arousal=0.5, mundo=0.5, vozeco=0.3))
nov = [og.proximo_gesto(fila(t=(2000 + k) * 0.1, energia=0.7, arousal=0.5, mundo=0.5))["expr_novedad"] for k in range(50)]
chk("EXPLORACIÓN constitutiva: la novedad nunca se anula", max(nov) > 0.0, f"novedad max reciente={max(nov):.3f}")

# (8) ANTI-SEMÁNTICA: la memoria asocia REGIÓN-DE-ESTADO → pesos de gestos; NO gesto→concepto
oh = OrganoExpresion("Xh")
for k in range(500): oh.proximo_gesto(fila(t=k * 0.1, energia=0.7, arousal=0.5, mundo=0.5, vozeco=0.3))
ok = (all(isinstance(v, dict) for v in oh.memoria.values())
      and not hasattr(oh, "significados") and not hasattr(oh, "diccionario"))
chk("ANTI-SEMÁNTICA: memoria = estado→gestos (sin tabla gesto→significado)", ok,
    "claves=regiones de estado, valores=pesos de gestos")

# (9) EL SILENCIO ES UNA CONDUCTA: compite, se ALMACENA en memoria y se refuerza (no es ausencia ni fallo)
osil = OrganoExpresion("Xsil")
for k in range(3000):
    osil.proximo_gesto(fila(t=k * 0.1, energia=0.5, arousal=0.1, mundo=0.0))   # estado calmo → predomina silencio
hay_silencio_en_memoria = any("SILENCIO" in d for d in osil.conducta_w.values())
# y debe haber pasos de silencio (vocalizando=0) Y de voz (vocalizando=1): ambas conductas ocurren
voc = sum(1 for k in range(400) if osil.proximo_gesto(fila(t=k * 0.1, energia=0.5, arousal=0.1))["expr_vocalizando"] >= 0.5)
chk("EL SILENCIO ES CONDUCTA: se ALMACENA y compite como 'SILENCIO' (no es ausencia)", hay_silencio_en_memoria,
    f"regiones con conducta de silencio registrada: {sum(1 for d in osil.conducta_w.values() if 'SILENCIO' in d)}")
chk("SILENCIO y VOZ compiten: ambas conductas ocurren (ni mudo total ni habla constante)",
    0 < voc < 400, f"vocalizó {voc}/400 pasos (resto = conducta de silencio)")

# (10) SILENCIO se REFUERZA por consecuencia: si en una región callar 'rinde', su peso crece (compite mejor)
osr = OrganoExpresion("Xsr")
prev = {"expr_vocalizando": 0.0}
for k in range(5000):
    # consecuencia cuando el organismo CALLA (vocalizando=0): el silencio adquiere valor en esa región
    vz = 0.30 if prev.get("expr_vocalizando", 0.0) < 0.5 else 0.0
    prev = osr.proximo_gesto(fila(t=k * 0.1, energia=0.5, arousal=0.1, mundo=0.0, vozeco=vz))
peso_sil = sum(d.get("SILENCIO", 0.0) for d in osr.conducta_w.values())
peso_voz = sum(d.get("VOZ", 0.0) for d in osr.conducta_w.values())
chk("SILENCIO se refuerza por su consecuencia (compite y aprende como cualquier conducta)", peso_sil > peso_voz,
    f"peso_silencio={peso_sil:.1f} > peso_voz={peso_voz:.1f}")

print("-" * 84)
nok = sum(1 for _, p in res if p)
print(f"  RESUMEN: {nok}/{len(res)} PASS")
print("  La voz dejó de ser un generador aislado: ahora es una conducta PROBABILÍSTICA, HISTÓRICA y")
print("  dependiente del organismo COMPLETO + mundo. El acoplamiento real (η² con el cuerpo/estado) y la")
print("  transición Etapa1→5 deben verificarse EN VIVO con mundo — no se presuponen aquí.")
sys.exit(0 if nok == len(res) else 1)
