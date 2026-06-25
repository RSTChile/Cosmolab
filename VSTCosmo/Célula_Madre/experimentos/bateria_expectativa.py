#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
BATERÍA — NODO DE EXPECTATIVA  ·  FALSACIÓN
================================================================================
Prueba si el OrganeloExpectativa aprende, SÓLO por consecuencias y falsable, que tras cierta firma
acústica del otro EXPLORAR tiende a mejorar la situación — sin significado, sin etiquetas, sin control.
  1. Señal irrelevante → no genera expectativa.
  2. Señal útil repetida → la expectativa crece lentamente.
  3. Señal útil barajada (SHUFFLED) → no converge.
  4. Señal deja de ser útil → la expectativa decae.
  5. Dos organismos con historias distintas → expectativas distintas para el mismo gesto.
  6. Anti-Shannon → no existe tabla gesto→significado; sólo experiencia→expectativa.
NO es lenguaje. Es el mecanismo previo: expectativa ANTES que agencia, intención, convención, lenguaje.
================================================================================
"""
import os, sys, random
AQUI = os.path.dirname(os.path.abspath(__file__)); RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)
[sys.path.insert(0, os.path.join(RAIZ, _d)) for _d in ("genoma", "campo", "organelos", "diada", "web", "audio") if os.path.isdir(os.path.join(RAIZ, _d))]
from VST_Expectativa import OrganeloExpectativa, COLS_EXP

DT = 0.1

def sim(util, n_ciclos=140, paso=25, barajar=False, sin_voz=False, e_voz=0.6,
        util_hasta=10**9, seed=0, org="X"):
    """Cada ciclo: en k=0 llega (o no) la firma del otro. Si 'util', EXPLORAR tras la firma (k 1..8)
    mejora la situación (ICR/A_sys-env/necesidad/H). 'barajar' = la mejora ocurre des-correlacionada de la
    firma (SHUFFLED). 'sin_voz' = NULL. Modela voz→expectativa→exploración→resultado (no voz→persistencia)."""
    o = OrganeloExpectativa(org, ventana=1.2)
    rng = random.Random(seed)
    t = 0.0; ICR = 0.3; A = 0.3; nec = 0.4; H = 0.6
    a = {}
    for c in range(n_ciclos):
        util_ahora = util and (c < util_hasta)
        for k in range(paso):
            t += DT
            e = e_voz if (k == 0 and not sin_voz) else 0.0
            if barajar:
                mejora = util_ahora and (rng.random() < 0.30)
            else:
                mejora = util_ahora and (1 <= k <= 8)       # explorar tras la firma rinde
            if mejora:
                ICR = min(1.0, ICR + 0.010); A = min(1.0, A + 0.008); nec = max(0.0, nec - 0.006)
            else:
                ICR += (0.3 - ICR) * 0.02; A += (0.3 - A) * 0.02; nec += (0.4 - nec) * 0.02
            fila = {"t": round(t, 2), "ICR": ICR, "A_sys_env": A, "necesidad": nec, "H_homeostasis": H}
            a = o.observar(fila, energia_voz_otro=e, dt=DT)
    return o, a

res = []
def chk(n, c, extra=""):
    res.append((n, bool(c))); print(f"  {'PASS' if c else 'FALLA'}  {n}{('  · ' + extra) if extra else ''}")

print("=" * 84)
print("BATERÍA — NODO DE EXPECTATIVA (falsación)")
print("=" * 84)

# (1) SEÑAL IRRELEVANTE → no genera expectativa
o_irr, a_irr = sim(util=False)
chk("SEÑAL IRRELEVANTE → sin expectativa", o_irr.expect_max() < 0.01, f"exp_máx={o_irr.expect_max():.4f}")

# (2) SEÑAL ÚTIL REPETIDA → la expectativa crece lentamente
o_util, a_util = sim(util=True)
chk("SEÑAL ÚTIL REPETIDA → la expectativa CRECE", o_util.expect_max() > 0.02, f"exp_máx={o_util.expect_max():.4f}")
chk("Registra confirmaciones y confianza al explorar tras la voz útil",
    a_util["expectativa_confirmaciones"] > a_util["expectativa_falsaciones"] and a_util["expectativa_confianza"] > 0.2,
    f"confirm={a_util['expectativa_confirmaciones']} fals={a_util['expectativa_falsaciones']} conf={a_util['expectativa_confianza']}")

# (3) SEÑAL ÚTIL BARAJADA (SHUFFLED) → no converge
o_shf, a_shf = sim(util=True, barajar=True)
chk("SHUFFLED → la expectativa NO converge (≪ útil)", o_shf.expect_max() < 0.5 * o_util.expect_max(),
    f"shuffled={o_shf.expect_max():.4f} < útil={o_util.expect_max():.4f}")

# (3b) NULL → sin voz, sin expectativa
o_null, a_null = sim(util=True, sin_voz=True)
chk("NULL (sin voz) → sin expectativa", o_null.expect_max() < 1e-6, f"exp_máx={o_null.expect_max():.6f}")

# (4) SEÑAL DEJA DE SER ÚTIL → decae
o_dec, _ = sim(util=True, util_hasta=70, n_ciclos=160)
o_full, _ = sim(util=True, n_ciclos=160)
chk("DECAIMIENTO: si deja de ser útil, la expectativa decae (vs útil sostenido)",
    o_dec.expect_max() < o_full.expect_max(), f"cesa={o_dec.expect_max():.4f} < sostenido={o_full.expect_max():.4f}")

# (5) DOS HISTORIAS → expectativas distintas para firmas distintas según biografía
oA, _ = sim(util=True, e_voz=0.85, org="A")
oB, _ = sim(util=True, e_voz=0.30, org="B")
fA = max(oA.expect.items(), key=lambda kv: kv[1])[0] if oA.expect else None
fB = max(oB.expect.items(), key=lambda kv: kv[1])[0] if oB.expect else None
chk("DOS HISTORIAS → A y B asignan expectativa a firmas distintas", fA != fB, f"A→{fA} · B→{fB}")

# (6) ANTI-SHANNON → no hay tabla gesto=significado; sólo experiencia→expectativa
sin_dicc = (not hasattr(o_util, "significados") and not hasattr(o_util, "diccionario")
            and all(isinstance(v, float) for v in o_util.expect.values()))
chk("ANTI-SHANNON: sin tabla gesto=significado; sólo experiencia→expectativa", sin_dicc,
    "claves = firmas de estructura · valores = expectativa aprendida")

# (7) SALIDA ACOTADA: la expectativa sólo empuja EXPLORACIÓN, leve y acotada (no orienta/decide)
sin_control = all(k.startswith("expectativa") for k in a_util)
chk("SALIDA única y LEVE (sólo exploración acotada; sin orientación/decisión)",
    sin_control and 0.0 <= a_util["expectativa_exploracion"] <= 0.20,
    f"exploracion={a_util['expectativa_exploracion']} · sin otras salidas={sin_control}")

print("-" * 84)
ok = sum(1 for _, p in res if p)
print(f"  RESUMEN: {ok}/{len(res)} PASS")
print("  Lectura: la expectativa es el primer eslabón — aprender que tras ciertos patrones del otro vale la")
print("  pena seguir explorando. Falsable (NULL/SHUFFLED). NO es agencia, intención, convención ni lenguaje:")
print("  es el mecanismo biológico que, con historia suficiente, podría hacerlos POSIBLES.")
sys.exit(0 if ok == len(res) else 1)
