#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
BATERÍA — VALOR ECOLÓGICO DE LA VOZ DEL OTRO  ·  FALSACIÓN
================================================================================
Prueba si el OrganeloValorEcologicoVoz aprende, SÓLO por consecuencias y de forma falsable, que una
clase de voz recibida ANTECEDE una mejora REAL de la persistencia del receptor — y que ese valor:
  · no aparece si la voz no ayuda (irrelevante);
  · sube lento si la voz contingente útil precede mejora;
  · NO se consolida si la utilidad está barajada (SHUFFLED fuerte);
  · no aparece sin otro (NULL);
  · no controla orientación ni acción (sólo modula levemente la permeabilidad);
  · decae si la voz deja de ser útil;
  · puede ser distinto entre A y B según su biografía;
  · NO usa ninguna tabla gesto=significado (anti-Shannon).
NO es lenguaje. Es la PRECONDICIÓN: que la voz del otro pueda volverse ecológicamente relevante.
================================================================================
"""
import os, sys, random
AQUI = os.path.dirname(os.path.abspath(__file__)); RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)
[sys.path.insert(0, os.path.join(RAIZ, _d)) for _d in ("genoma", "campo", "organelos", "diada", "web", "audio") if os.path.isdir(os.path.join(RAIZ, _d))]
from VST_ValorEcologicoVoz import OrganeloValorEcologicoVoz, COLS_VOZECO

DT = 0.1

def sim(util, n_ciclos=120, paso=25, barajar=False, sin_voz=False, e_voz=0.6,
        util_desde=0, util_hasta=10**9, seed=0, org="X"):
    """Cada ciclo: en k=0 llega (o no) la voz del otro; si 'util' y el ciclo está en [desde,hasta], la
    persistencia del receptor MEJORA dentro de la ventana. 'barajar' = la mejora ocurre en momentos
    DES-correlacionados de la voz (SHUFFLED fuerte). 'sin_voz' = NULL (no llega voz)."""
    o = OrganeloValorEcologicoVoz(org, ventana=1.2)
    rng = random.Random(seed)
    t = 0.0; A = 0.3; E = 0.5; nec = 0.4; ICR = 0.3; H = 0.6; perm = 0.2
    a = {}
    for c in range(n_ciclos):
        util_ahora = util and (util_desde <= c < util_hasta)
        for k in range(paso):
            t += DT
            e = e_voz if (k == 0 and not sin_voz) else 0.0     # la voz llega al inicio del ciclo
            # mejora REAL de la persistencia: tras la voz (k 1..8) si es útil; si barajar, en momentos al azar
            if barajar:
                mejora = util_ahora and (rng.random() < 0.30)
            else:
                mejora = util_ahora and (1 <= k <= 8)
            if mejora:
                E = min(1.0, E + 0.010); nec = max(0.0, nec - 0.006); A = min(1.0, A + 0.006)
            else:
                E += (0.5 - E) * 0.02; nec += (0.4 - nec) * 0.02; A += (0.3 - A) * 0.02   # deriva suave a baseline
            fila = {"t": round(t, 2), "A_sys_env": A, "ICR": ICR, "met_energia": E,
                    "necesidad": nec, "H_homeostasis": H, "act_perm": perm}
            a = o.observar(fila, energia_voz_otro=e, dt=DT)
    return o, a

res = []
def chk(n, c, extra=""):
    res.append((n, bool(c))); print(f"  {'PASS' if c else 'FALLA'}  {n}{('  · ' + extra) if extra else ''}")

print("=" * 84)
print("BATERÍA — VALOR ECOLÓGICO DE LA VOZ DEL OTRO (falsación)")
print("=" * 84)

# (1) VOZ IRRELEVANTE: llega voz pero no mejora nada → valor ecológico permanece bajo
o_irr, a_irr = sim(util=False)
chk("VOZ IRRELEVANTE → valor ecológico BAJO", o_irr.valor_max() < 0.01,
    f"valor_máx={o_irr.valor_max():.4f}")

# (2) VOZ CONTINGENTE ÚTIL: la voz antecede mejora real → el valor ecológico sube (lento)
o_util, a_util = sim(util=True)
chk("VOZ CONTINGENTE ÚTIL → valor ecológico SUBE", o_util.valor_max() > 0.02,
    f"valor_máx={o_util.valor_max():.4f}")
chk("La voz útil registra beneficio histórico y confianza ecológica", a_util["voz_otro_historia_beneficio"] > 0.05 and a_util["voz_otro_confianza_ecologica"] > 0.2,
    f"historia={a_util['voz_otro_historia_beneficio']} · confianza={a_util['voz_otro_confianza_ecologica']}")

# (3) VOZ ÚTIL PERO BARAJADA (SHUFFLED fuerte): mejora des-correlacionada de la voz → NO se consolida
o_shf, a_shf = sim(util=True, barajar=True)
chk("SHUFFLED fuerte → valor ecológico NO se consolida (≪ contingente)", o_shf.valor_max() < 0.5 * o_util.valor_max(),
    f"shuffled={o_shf.valor_max():.4f} < útil={o_util.valor_max():.4f}")

# (4) NULL: sin voz no debe aparecer valor ecológico
o_null, a_null = sim(util=True, sin_voz=True)
chk("NULL (sin voz) → sin valor ecológico", o_null.valor_max() < 1e-6, f"valor_máx={o_null.valor_max():.6f}")

# (5) NO CONTROL DIRECTO: la voz valorada sólo MODULA permeabilidad (acotada), no orienta ni decide
sin_orient = all(k.startswith("voz_otro_") for k in a_util) and not any("orient" in k or "decision" in k for k in a_util)
mod = a_util["voz_otro_modulacion_aplicada"]
chk("NO control directo: sólo modula permeabilidad (acotada), sin orientación/decisión",
    sin_orient and (0.74 <= mod <= 1.26), f"modulación={mod} · sin salidas de orientación/decisión={sin_orient}")

# (6) DECAIMIENTO: la voz deja de ser útil a mitad → el valor ecológico baja respecto al pico
o_dec, _ = sim(util=True, util_hasta=60, n_ciclos=140)
pico = o_dec.valor_max()
o_full, _ = sim(util=True, n_ciclos=140)
chk("DECAIMIENTO: si la voz deja de ser útil, el valor decae (vs útil sostenido)", pico < o_full.valor_max(),
    f"útil-y-cesa={pico:.4f} < útil-sostenido={o_full.valor_max():.4f}")

# (7) DIFERENTES HISTORIAS: A valora la voz fuerte (e=0.8), B la voz suave (e=0.3) según qué le sirvió
oA, _ = sim(util=True, e_voz=0.85, org="A")
oB, _ = sim(util=True, e_voz=0.30, org="B")
fA = max(oA.valor.items(), key=lambda kv: kv[1])[0] if oA.valor else None
fB = max(oB.valor.items(), key=lambda kv: kv[1])[0] if oB.valor else None
chk("DIFERENTES HISTORIAS: A y B valoran firmas de voz distintas según su biografía", fA != fB,
    f"A→{fA} · B→{fB}")

# (8) ANTI-SHANNON: no existe ninguna tabla gesto=significado; el valor es SÓLO consecuencia
sin_diccionario = (not hasattr(o_util, "diccionario") and not hasattr(o_util, "significados")
                   and all(isinstance(v, float) for v in o_util.valor.values()))
chk("ANTI-SHANNON: sin tabla gesto=significado; valor SÓLO por consecuencia", sin_diccionario,
    f"claves de valor = firmas de estructura, valores = beneficio aprendido")

print("-" * 84)
ok = sum(1 for _, p in res if p)
print(f"  RESUMEN: {ok}/{len(res)} PASS")
print("  Lectura: si esto pasa, la voz del otro PUEDE volverse ecológicamente relevante por historia (no")
print("  por diseño), y sólo si de verdad sostiene la persistencia del receptor — y desaparece bajo NULL y")
print("  SHUFFLED. Recién cuando la voz IMPORTE para persistir tendrá sentido esperar agencia comunicativa.")
print("  Aún NO es lenguaje ni significado: es la PRECONDICIÓN ecológica de la alteridad.")
sys.exit(0 if ok == len(res) else 1)
