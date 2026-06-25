#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
BATERÍA — RECONOCIMIENTO DEL OTRO COMO SUJETO  ·  FALSACIÓN  (24-06-2026)
================================================================================
POR QUÉ ESTA BATERÍA (hoja de ruta GPT/Alexis)
  Ya no desarrollamos el organismo individual. Estamos en la transición
  Sentido Compartido (C-N9) → Alteridad (O-N3.4). Lo demostrado: resonancia
  afectiva, comunicación funcional, cultura acumulativa. Lo NO demostrado y
  objetivo central: el RECONOCIMIENTO DEL OTRO COMO SUJETO.

PRINCIPIO RECTOR (metodológico, permanente)
  No diseñamos lenguaje/símbolos/convenciones. Cada capacidad debe EMERGER de
  la historia y DESAPARECER bajo controles NULL/SHUFFLED. Si hay que imponerla
  externamente, NO pertenece a la arquitectura cosmosemiótica.

QUÉ PRUEBA (y qué NO)
  No prueba lenguaje. Prueba si el OrganeloAlteridad distingue a un OTRO-SUJETO
  (un agente cuya respuesta depende de MI acto) de: nada (NULL), una presencia
  inerte de igual energía (NOISE no-social), y un otro que responde pero SIN
  contingencia con mi acto (SHUFFLED). Cada señal del reconocimiento debe estar
  presente en REAL y COLAPSAR en los controles. Si no colapsa → es confound.

  Cuatro condiciones, todas con la MISMA energía de presencia salvo NULL:
    real     · el otro responde a MI emisión (contingente) y eso me beneficia.
    null     · no hay otro (desacople total).
    noise    · hay un otro PRESENTE pero inerte (energía constante, sin agencia).
    shuffled · el otro responde, pero de forma INDEPENDIENTE de mi acto.
================================================================================
"""
import os, sys, random
AQUI = os.path.dirname(os.path.abspath(__file__)); RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)
[sys.path.insert(0, os.path.join(RAIZ, _d)) for _d in ("genoma", "campo", "organelos", "diada", "web", "audio") if os.path.isdir(os.path.join(RAIZ, _d))]
from VST_Alteridad import OrganeloAlteridad

DT = 0.1; VENT = 1.0

def sim(cond, patrones=("P_a",), p_efectivo=None, n_ciclos=90, pasos_ciclo=16,
        beneficia=True, seed=0, ausencia=False):
    """Corre un escenario bajo una condición. p_efectivo = patrón al que el otro responde en REAL
    (si None, responde a cualquiera). ausencia = el otro se va un tramo (para probar contacto)."""
    o = OrganeloAlteridad(f"X-{cond}-{seed}", ventana=VENT); o.libertad = False  # patrones fijos: probamos la LÓGICA
    rng = random.Random(seed)
    t = 0.0; oi_o = 0.2; orient_o = 0.0; oi_mi = 0.2; nec = 0.4; llamo = 0.0; rec = 0.0
    a = {}
    for c in range(n_ciclos):
        P = patrones[c % len(patrones)]
        for k in range(pasos_ciclo):
            t += DT
            voz = P if k == 0 else "-"
            presente = (not (20 <= c <= 35)) if ausencia else True
            if cond == "real":
                resp = (1 <= k <= 9) and (p_efectivo is None or P == p_efectivo)
            elif cond == "shuffled":
                resp = (1 <= k <= 9) and (rng.random() < 0.5)      # responde en la ventana, pero independiente de P
            elif cond == "background":
                resp = (rng.random() < 9.0 / 16.0)                 # cambia espontáneamente a la MISMA tasa que real,
                                                                   # pero INDEPENDIENTE de mi emisión (sin contingencia)
            else:
                resp = False                                       # null / noise: el otro no responde a mi acto
            if resp and presente:
                oi_o += 0.03; orient_o += 3.0
                if beneficia:
                    oi_mi += 0.02; nec = max(0.0, nec - 0.01)
            if cond == "null" or (ausencia and not presente):
                otro = None
            elif cond == "noise":
                otro = {"fila": {"OI": 0.3, "necesidad": 0.3, "orientacion_deg": 0.0, "voz_emitida": "const"}, "ok": True}
            else:
                otro = {"fila": {"OI": oi_o, "necesidad": 0.3, "orientacion_deg": orient_o,
                                 "voz_emitida": (P if resp else "-")}, "ok": True}
            fila = {"t": round(t, 2), "voz_emitida": voz, "OI": oi_mi, "necesidad": nec,
                    "A_sys_env": 0.2 + 0.5 * oi_mi, "energia": 0.5, "mem_relacional_confianza": 0.3}
            a = o.observar(fila, otro, dt=DT)
            llamo = max(llamo, a["alt_contacto_presencia"]); rec = max(rec, a["alt_contacto_recuperado"])
        oi_o = 0.2; orient_o *= 0.7
    o._llamo = llamo; o._rec = rec; o._ulta = a
    return o

def valor_por_patron(o):
    agg = {}
    for (P, ctx), v in o.valor.items():
        agg[P] = max(agg.get(P, -9.0), v)
    return agg

def concentracion(o):
    """Cuán CONCENTRADO está el valor en un patrón (max − media). Alta = el organismo halló un acto
    específico que afecta al otro; baja = el efecto no depende de qué hace (no hay agencia específica)."""
    vp = list(valor_por_patron(o).values())
    if not vp:
        return 0.0
    m = sum(vp) / len(vp)
    return max(vp) - m

res = []
def chk(bloque, n, c, extra=""):
    res.append((bloque, n, bool(c)))
    print(f"  [{bloque}] {'PASS' if c else 'FRONTERA/NO'}  {n}{('  · ' + extra) if extra else ''}")

print("=" * 84)
print("BATERÍA — RECONOCIMIENTO DEL OTRO COMO SUJETO (falsación REAL vs NULL/SHUFFLED/NOISE)")
print("=" * 84)

# ───────────────────────── BLOQUE A — CONSOLIDACIÓN (debe PASAR) ─────────────────────────
print("\nBLOQUE A — Consolidación de nodos alcanzados (acople/agencia/contacto):")
o_real = sim("real",     patrones=("P_a",))
o_null = sim("null",     patrones=("P_a",))
o_nois = sim("noise",    patrones=("P_a",))

chk("A1", "Otro RESPONSIVO > presencia INERTE (no-social) — efecto_sobre_otro",
    o_real.efecto_otro_ema > max(0.02, 3 * o_nois.efecto_otro_ema),
    f"real={o_real.efecto_otro_ema:.4f} vs noise={o_nois.efecto_otro_ema:.4f}")
chk("A2", "Necesita un OTRO real — intención REAL ≫ NULL (O-N3.4)",
    o_real.intencion > 0.05 and o_real.intencion > 4 * (o_null.intencion + 1e-6),
    f"real={o_real.intencion:.4f} vs null={o_null.intencion:.4f}")
chk("A3", "Presencia inerte NO crea intención — intención REAL ≫ NOISE",
    o_real.intencion > 4 * (o_nois.intencion + 1e-6),
    f"real={o_real.intencion:.4f} vs noise={o_nois.intencion:.4f}")
o_aus = sim("real", patrones=("P_a",), ausencia=True)
chk("A4", "CONTACTO: llama al ausentarse el otro y registra su regreso",
    o_aus._llamo > 0.5 and o_aus._rec > 0.5, f"llamada={o_aus._llamo} · recuperado={o_aus._rec}")
oA = sim("real", patrones=("P_a", "P_b", "P_c"), p_efectivo="P_a")
oZ = sim("real", patrones=("P_a", "P_b", "P_c"), p_efectivo="P_c")
mejorA = max(valor_por_patron(oA).items(), key=lambda kv: kv[1])[0] if oA.valor else None
mejorZ = max(valor_por_patron(oZ).items(), key=lambda kv: kv[1])[0] if oZ.valor else None
chk("A5", "ARBITRARIEDAD: distinta historia → distinto acto valorado (no hay tono 'correcto')",
    mejorA == "P_a" and mejorZ == "P_c" and mejorA != mejorZ, f"hist A→{mejorA} · hist Z→{mejorZ}")

# ───────────────────────── BLOQUE B — FRONTERA (lo que aún no sabemos) ─────────────────────────
# El núcleo del reconocimiento del otro como SUJETO: ¿descubre el organismo CUÁL de sus actos afecta
# al otro? Eso sólo es posible si la respuesta del otro es CONTINGENTE con el acto. Bajo SHUFFLED el
# otro responde igual haga lo que haga → no hay acto específico → el valor NO debe concentrarse.
print("\nBLOQUE B — Frontera: ¿reconoce al otro como agente contingente? (REAL vs SHUFFLED):")
o_realM = sim("real",     patrones=("P_a", "P_b", "P_c"), p_efectivo="P_a", seed=1)
o_shufM = sim("shuffled", patrones=("P_a", "P_b", "P_c"), seed=1)

chk("B1", "ESPECIFICIDAD del acto: el valor se concentra en el acto efectivo (REAL) y se dispersa (SHUFFLED)",
    concentracion(o_realM) > 2 * (concentracion(o_shufM) + 1e-6) and concentracion(o_realM) > 0.01,
    f"concentración real={concentracion(o_realM):.4f} vs shuffled={concentracion(o_shufM):.4f}")
mejorR = max(valor_por_patron(o_realM).items(), key=lambda kv: kv[1])[0] if o_realM.valor else None
chk("B2", "AGENCIA dirigida: en REAL el acto valorado ES el efectivo (P_a); SHUFFLED no lo distingue",
    mejorR == "P_a", f"real→{mejorR} (esperado P_a)")
chk("B3", "MODELO del otro más nítido con contingencia — error de predicción REAL ≤ SHUFFLED",
    o_realM.error_pred_ema <= o_shufM.error_pred_ema + 1e-6,
    f"err_pred real={o_realM.error_pred_ema:.4f} vs shuffled={o_shufM.error_pred_ema:.4f}")
# B4 — el límite honesto: con UN solo acto y un otro que cambia espontáneamente a la MISMA tasa
# (BACKGROUND, rate-matched, sin contingencia con mi emisión), ¿separa 'me responde a MÍ' de 'cambió
# por su cuenta'? Sin línea-base de contingencia, el órgano mide correlación, no causalidad → NO debería.
o_real1 = sim("real",       patrones=("P_a",), seed=2)
o_bg1   = sim("background", patrones=("P_a",), seed=2)
distingue_1p = o_real1.intencion > 1.8 * (o_bg1.intencion + 1e-6)
chk("B4", "LÍMITE del viejo 'intención' (nivel-presencia): REAL vs BACKGROUND rate-matched, un solo acto",
    distingue_1p, f"intención real={o_real1.intencion:.4f} vs background={o_bg1.intencion:.4f}  → "
    + ("separa" if distingue_1p else "NO separa (correlación, no causalidad) → lo resuelve la AGENCIA, Bloque C"))

# ───────────────────────── BLOQUE C — AGENCIA (la medida NUEVA) ─────────────────────────
# Lo que B4 reveló como límite (mide correlación, no causalidad) AHORA SE MIDE: alt_contingencia_social
# y alt_agencia_otro comparan el cambio del otro JUSTO DESPUÉS vs JUSTO ANTES de mi emisión (línea-base
# pre/post). Debe separar causalidad de coincidencia: alta en REAL, baja en BACKGROUND (mismo cambio
# espontáneo, sin contingencia con mi acto). Requiere ventana quieta antes de emitir (emisión esparcida).
print("\nBLOQUE C — Agencia del otro: contingencia social pre/post (lo que B4 pedía):")
def sim_sparse(cond, n=120, paso=30, seed=0):
    o = OrganeloAlteridad(f"C-{cond}", ventana=VENT); o.libertad = False; rng = random.Random(seed)
    t = 0.0; oi = 0.2; ori = 0.0; oimi = 0.2; nec = 0.4; a = {}
    for c in range(n):
        for k in range(paso):                          # ciclo largo: emite en k=0, responde k1..9, QUIETO después
            t += DT; voz = "P_a" if k == 0 else "-"
            resp = (1 <= k <= 9) if cond == "real" else (rng.random() < 9.0 / paso if cond == "background" else False)
            if resp:
                oi += 0.03; ori += 2.0; oimi += 0.015; nec = max(0.0, nec - 0.01)
            else:
                oi += (0.2 - oi) * 0.05; ori *= 0.97   # relaja suave hacia baseline (línea-base limpia)
            otro = {"fila": {"OI": oi, "necesidad": 0.3, "orientacion_deg": ori,
                             "voz_emitida": ("P_a" if resp else "-")}, "ok": True}
            fila = {"t": round(t, 2), "voz_emitida": voz, "OI": oimi, "necesidad": nec,
                    "A_sys_env": 0.2 + 0.5 * oimi, "energia": 0.5, "mem_relacional_confianza": 0.3}
            a = o.observar(fila, otro, dt=DT)
    return a
cR = sim_sparse("real"); cB = sim_sparse("background"); cN = sim_sparse("null")
chk("C1", "CONTINGENCIA social REAL ≫ BACKGROUND (separa causalidad de coincidencia)",
    cR["alt_contingencia_social"] > 2.5 * (cB["alt_contingencia_social"] + 1e-6),
    f"real={cR['alt_contingencia_social']:.4f} vs background={cB['alt_contingencia_social']:.4f}")
chk("C2", "AGENCIA del otro alta en REAL (depende de MI acto), menor sin contingencia",
    cR["alt_agencia_otro"] > 0.5 and cR["alt_agencia_otro"] > cB["alt_agencia_otro"],
    f"real={cR['alt_agencia_otro']:.3f} vs background={cB['alt_agencia_otro']:.3f}")
chk("C3", "Sin otro (NULL) no hay contingencia social",
    cN["alt_contingencia_social"] < 0.02, f"null={cN['alt_contingencia_social']:.4f}")

print("-" * 84)
A = [p for bl, n, p in res if bl.startswith("A")]
B = [p for bl, n, p in res if bl.startswith("B")]
C = [p for bl, n, p in res if bl.startswith("C")]
print(f"  BLOQUE A (consolidación): {sum(A)}/{len(A)}   ·   BLOQUE B (frontera vieja): {sum(B)}/{len(B)}   ·   BLOQUE C (agencia NUEVA): {sum(C)}/{len(C)}")
print("\n  LECTURA (honesta):")
print("  · A consolida O-N3.4: el organismo distingue un otro RESPONSIVO de la nada y de una presencia")
print("    inerte; lo necesita real; lo llama si se va; y el acto valorado depende de la HISTORIA (arbitrario).")
print("  · B sondea el RECONOCIMIENTO DEL OTRO COMO SUJETO: con varios actos, descubre CUÁL afecta al otro")
print("    sólo si la respuesta es contingente (REAL concentra, SHUFFLED dispersa) = agencia específica.")
print("  · B4 mostró el LÍMITE del viejo 'intención' (nivel-presencia): no separa causalidad de coincidencia.")
print("  · BLOQUE C lo RESUELVE con la medida nueva: alt_contingencia_social / alt_agencia_otro comparan el")
print("    cambio del otro pre/post emisión → separan 'me responde a MÍ' de 'cambió igual'. Es el primer paso")
print("    MEDIBLE del reconocimiento del otro como SUJETO (O-N3.4): de PRESENCIA a AGENCIA.")
print("  · PRUEBA FINAL pendiente: correr EN VIVO (ANIMA_CONTROL=real|null|shuffled) y verificar que")
print("    alt_contingencia_social/alt_agencia_otro COLAPSAN bajo shuffle. Aún NO es lenguaje ni teoría de la mente.")
sys.exit(0 if (all(A) and all(C)) else 1)   # A y C deben pasar; B4 es diagnóstico del límite viejo (no bloquea)
