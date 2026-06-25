#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
BATERÍA — LIBERTAD EXPRESIVA (balbuceo) · FALSACIÓN
================================================================================
QUÉ PRUEBA (y qué NO)
  Prueba que la voz dejó de ser estado→patrón FIJO y pasó a estado+exploración→patrón:
  el organismo explora pequeñas variaciones ACÚSTICAS espontáneas (frecuencia, intensidad,
  pausa, repetición), NO etiquetas; y que el OrganeloAlteridad aprende por CONSECUENCIAS sobre
  esos GESTOS (no sobre palabras). NO prueba lenguaje (no hay convención estable ni significado).

  Esto es el equivalente biológico del BALBUCEO: explorar el espacio expresivo, condición previa
  (y única posible) para que una convención semiótica pueda emerger de la historia compartida.
================================================================================
"""
import os, sys
AQUI = os.path.dirname(os.path.abspath(__file__)); RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)
[sys.path.insert(0, os.path.join(RAIZ, _d)) for _d in ("genoma", "campo", "organelos", "diada", "web", "audio") if os.path.isdir(os.path.join(RAIZ, _d))]
import numpy as np
from VST_Alteridad import OrganeloAlteridad
from VST_OrganoComunicacion import OrganoComunicacion

DT = 0.1
res = []
def chk(n, c, extra=""):
    res.append((n, bool(c))); print(f"  {'PASS' if c else 'FALLA'}  {n}{('  · '+extra) if extra else ''}")

print("=" * 80); print("BATERÍA — LIBERTAD EXPRESIVA (balbuceo)"); print("=" * 80)

# (1) EXPLORACIÓN: con libertad ON, el gesto recorre el espacio acústico (varios buckets distintos)
os.environ["ANIMA_LIBERTAD_EXPRESIVA"] = "1"
o = OrganeloAlteridad("ANIMA_A")
fila = {"t": 0.0, "necesidad": 0.4, "OI": 0.3, "voz_emitida": "chat", "A_sys_env": 0.3, "energia": 0.5}
buckets = set(); rango = []
for i in range(400):
    fila["t"] = round(i * DT, 2)
    g = o.gesto_actual(fila); buckets.add(g["g_bucket"]); rango.append(g["g_freq"])
chk("EXPLORA el espacio acústico (≥4 gestos distintos)", len(buckets) >= 4, f"buckets distintos={len(buckets)}")
chk("La exploración es CONTINUA y acotada (|g_freq|≤1)", max(abs(min(rango)), abs(max(rango))) <= 1.0,
    f"rango g_freq=[{min(rango):.2f},{max(rango):.2f}]")

# (2) REVERSIBLE/PEQUEÑA: paso a paso el gesto cambia POCO (random walk, no saltos)
o2 = OrganeloAlteridad("ANIMA_B"); prev = None; saltos = []
for i in range(200):
    fila["t"] = round(i * DT, 2); g = o2.gesto_actual(fila)
    v = np.array([g["g_freq"], g["g_intensidad"], g["g_pausa"], g["g_repeticion"]])
    if prev is not None: saltos.append(float(np.linalg.norm(v - prev)))
    prev = v
chk("Variación PEQUEÑA y reversible (paso medio < 0.4)", np.mean(saltos) < 0.4, f"paso medio={np.mean(saltos):.3f}")

# (3) OFF: con libertad apagada, la voz es fisiológica pura (gesto neutro)
os.environ["ANIMA_LIBERTAD_EXPRESIVA"] = "0"
o3 = OrganeloAlteridad("ANIMA_A")
g = o3.gesto_actual(fila)
chk("OFF → voz fisiológica pura (gesto neutro)", g["g_bucket"] == "fisio" and abs(g["g_freq"]) < 1e-9, f"bucket={g['g_bucket']}")
os.environ["ANIMA_LIBERTAD_EXPRESIVA"] = "1"

# (4) ARBITRARIEDAD del balbuceo: distinto organismo (semilla) → distinta trayectoria de exploración
oa = OrganeloAlteridad("ANIMA_A"); ob = OrganeloAlteridad("ANIMA_B")
ta = []; tb = []
for i in range(100):
    fila["t"] = round(i * DT, 2)
    ta.append(oa.gesto_actual(fila)["g_freq"]); tb.append(ob.gesto_actual(fila)["g_freq"])
chk("ARBITRARIEDAD: A y B exploran distinto (semillas distintas)", np.std(np.array(ta) - np.array(tb)) > 0.05,
    f"divergencia A↔B (std)={np.std(np.array(ta)-np.array(tb)):.3f}")

# (5) APRENDE SOBRE EL GESTO: un gesto que mueve al otro Y beneficia gana valor (por consecuencia)
oc = OrganeloAlteridad("ANIMA_C", ventana=1.0)
oi_o = 0.2; orient_o = 0.0; oi_mi = 0.2; nec = 0.4
for c in range(60):
    g = oc.gesto_actual({"t": round(c * 16 * DT, 2), "necesidad": nec, "OI": oi_mi, "voz_emitida": "chat", "A_sys_env": 0.3, "energia": 0.5})
    for k in range(16):
        t = round((c * 16 + k) * DT, 2)
        voz = "chat" if k == 0 else "-"
        resp = (1 <= k <= 9)                       # el otro responde tras la emisión
        if resp: oi_o += 0.03; orient_o += 3.0; oi_mi += 0.02; nec = max(0.0, nec - 0.01)
        otro = {"fila": {"OI": oi_o, "necesidad": 0.3, "orientacion_deg": orient_o, "voz_emitida": ("chat" if resp else "-")}, "ok": True}
        f = {"t": t, "voz_emitida": voz, "g_bucket": g["g_bucket"], "OI": oi_mi, "necesidad": nec, "A_sys_env": 0.2 + 0.5 * oi_mi, "energia": 0.5, "mem_relacional_confianza": 0.3}
        oc.observar(f, otro, dt=DT)
    oi_o = 0.2; orient_o *= 0.7
gestos_valorados = [k for k, v in oc.valor.items() if v > 0.005]
chk("APRENDE por consecuencia sobre el GESTO (no etiqueta)", len(gestos_valorados) > 0 and all(str(k[0]).startswith("g") for k in gestos_valorados),
    f"gestos con valor>0: {len(gestos_valorados)}")

# (6) APLICACIÓN ACÚSTICA: el gesto MODIFICA de verdad la onda (duración/forma), no es decorativo
com = OrganoComunicacion("ANIMA_A")
base = np.sin(np.linspace(0, 40 * np.pi, com.sr // 2)) * 0.5   # 0.5 s de tono
com.gesto = None; w0 = com._aplicar_gesto(base.copy())
com.gesto = {"g_freq": 0.5, "g_intensidad": 0.4, "g_pausa": 0.5, "g_repeticion": 0.8}; w1 = com._aplicar_gesto(base.copy())
chk("El gesto MODIFICA la onda (longitud cambia con pausa+repetición)", w1.size != w0.size, f"len base={w0.size} → gesto={w1.size}")
chk("Gesto NEUTRO no altera la voz (reversibilidad)", w0.size == base.size and np.allclose(w0, base), "neutro = identidad")

print("-" * 80)
ok = sum(1 for _, p in res if p)
print(f"  RESUMEN: {ok}/{len(res)} PASS")
print("  NOTA honesta: esto valida que existe LIBERTAD EXPRESIVA (balbuceo) y que el órgano aprende por")
print("  consecuencias sobre el GESTO acústico. NO valida lenguaje: no hay convención estable ni significado.")
print("  La convención, si emerge, sólo podrá venir de la HISTORIA COMPARTIDA entre A y B (no programada).")
sys.exit(0 if ok == len(res) else 1)
