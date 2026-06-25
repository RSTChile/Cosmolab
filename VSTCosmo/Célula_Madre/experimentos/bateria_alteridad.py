#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BATERÍA — Alteridad / intención comunicativa (FALSACIÓN). Prueba la LÓGICA del OrganeloAlteridad
con escenarios sintéticos controlados: ¿el órgano DESCUBRE que su emisión modifica al otro, sólo
cuando de verdad lo modifica y le beneficia? ¿desaparece al cortar la comunicación (NULL)? ¿no
forma convención con voces barajadas (SHUFFLED)? ¿emerge contacto?
NO prueba lenguaje (no hay). Prueba la condición funcional de la intención comunicativa.
"""
import os, sys
AQUI = os.path.dirname(os.path.abspath(__file__)); RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)
[sys.path.insert(0, os.path.join(RAIZ, _d)) for _d in ("genoma", "campo", "organelos", "diada", "web", "audio") if os.path.isdir(os.path.join(RAIZ, _d))]
from VST_Alteridad import OrganeloAlteridad

DT = 0.1; VENT = 1.0   # ventana del órgano (1s = 10 pasos)

def sim(responde, beneficia, n_ciclos=40, presente=True, patrones=("P_a",), barajar=False, seed=0):
    """Corre un escenario: cada 'ciclo' el organismo emite un patrón; el OTRO responde (o no) en la
    ventana, y eso beneficia (o no) al organismo. Devuelve (organo, último alt)."""
    o = OrganeloAlteridad("X", ventana=VENT)
    t = 0.0; oi_mio = 0.2; nec = 0.3; oi_otro = 0.2; orient_otro = 0.0
    pasos_ciclo = 16
    alt = {}
    for c in range(n_ciclos):
        P = patrones[c % len(patrones)]
        # patrón que "responde" si barajar: el efecto NO sigue a P (se asigna a otro patrón)
        for k in range(pasos_ciclo):
            t += DT
            voz = P if k == 0 else "-"        # emite al inicio del ciclo (cambio de patrón = turno)
            # el otro responde dentro de la ventana (pasos 1..9 tras emitir), si 'responde'
            resp = responde and (1 <= k <= 9)
            if barajar:
                resp = responde and (1 <= k <= 9) and ((c + hash(P)) % 2 == 0)  # respuesta des-correlacionada de P
            if resp:
                oi_otro += 0.03; orient_otro += 3.0
            # el beneficio propio llega tras la respuesta del otro (pasos 5..12)
            if beneficia and resp:
                oi_mio += 0.02; nec = max(0.0, nec - 0.01)
            otro = {"OI": oi_otro, "necesidad": 0.3, "orientacion_deg": orient_otro,
                    "voz_emitida": (P if resp else "-"), "vivo": presente}
            fila = {"t": round(t, 2), "voz_emitida": voz, "OI": oi_mio, "necesidad": nec,
                    "A_sys_env": 0.2 + 0.5 * oi_mio, "energia": 0.5, "mem_relacional_confianza": 0.3}
            alt = o.observar(fila, otro, dt=DT)
        # relajación entre ciclos (vuelve hacia baseline para que el delta del próximo se note)
        oi_otro = 0.2 + 0.0; orient_otro *= 0.7
    return o, alt

res = []
def chk(n, c, extra=""):
    res.append((n, bool(c))); print(f"  {'PASS' if c else 'FALLA'}  {n}{('  · '+extra) if extra else ''}")

print("=" * 80)
print("BATERÍA — Alteridad / intención comunicativa (falsación)")
print("=" * 80)

# (1) BASAL: el otro NO responde a la emisión → poca intención
o_bas, a_bas = sim(responde=False, beneficia=False)
chk("BASAL: sin respuesta del otro → intención BAJA", a_bas["alt_intencion_comunicativa"] < 0.05,
    f"intención={a_bas['alt_intencion_comunicativa']}")

# (2) ACOPLE REAL: el otro cambia tras la emisión → efecto_sobre_otro sube
o_aco, a_aco = sim(responde=True, beneficia=False)
chk("ACOPLE: el otro cambia tras emitir → efecto_sobre_otro SUBE", a_aco["alt_efecto_sobre_otro"] > 0.05,
    f"efecto_otro={a_aco['alt_efecto_sobre_otro']}")

# (3) BENEFICIO: la respuesta del otro me ayuda → valor de emisión e intención suben
o_ben, a_ben = sim(responde=True, beneficia=True)
val_ben = max(o_ben.valor.values()) if o_ben.valor else 0.0   # valor APRENDIDO máx (no el del paso actual, que es "-")
chk("BENEFICIO: el otro me ayuda → valor de emisión APRENDIDO SUBE", val_ben > 0.01,
    f"valor_aprendido_máx={round(val_ben,4)}")
chk("INTENCIÓN emerge (otro responde Y me beneficia)", a_ben["alt_intencion_comunicativa"] > a_bas["alt_intencion_comunicativa"] + 0.02,
    f"intención acoplada={a_ben['alt_intencion_comunicativa']} vs basal={a_bas['alt_intencion_comunicativa']}")

# (4) NULL: cortar la comunicación (otro no responde) → intención << que acoplada
chk("NULL (sin comunicación) → intención cae vs acoplada",
    a_bas["alt_intencion_comunicativa"] < 0.5 * max(1e-6, a_ben["alt_intencion_comunicativa"]),
    f"NULL={a_bas['alt_intencion_comunicativa']} < acoplada={a_ben['alt_intencion_comunicativa']}")

# (5) SHUFFLED: respuesta del otro des-correlacionada del patrón → no hay convención estable (valor bajo)
o_shf, a_shf = sim(responde=True, beneficia=True, barajar=True, patrones=("P_a", "P_b", "P_c"))
maxv = max(o_shf.valor.values()) if o_shf.valor else 0.0
chk("SHUFFLED: voces des-correlacionadas → sin convención tan fuerte como la acoplada", maxv < val_ben,
    f"valor_máx_shuffled={round(maxv,4)} < acoplado={round(val_ben,4)}")

# (6) ARBITRARIEDAD: dos 'historias' distintas (distinto patrón funciona) → emerge un patrón distinto
oA, _ = sim(responde=True, beneficia=True, patrones=("P_a",))
oB, _ = sim(responde=True, beneficia=True, patrones=("P_z",))
mejorA = max(oA.valor.items(), key=lambda kv: kv[1])[0][0] if oA.valor else None
mejorB = max(oB.valor.items(), key=lambda kv: kv[1])[0][0] if oB.valor else None
chk("ARBITRARIEDAD: distinta historia → distinto patrón valorado", mejorA != mejorB,
    f"A→{mejorA} · B→{mejorB}")

# (7) CONTACTO: el otro se AUSENTA, el organismo emite (llamada) y el otro VUELVE → contacto_recuperado
o_c = OrganeloAlteridad("C", ventana=VENT); t = 0.0; rec = 0.0; llamo = 0.0
for k in range(60):
    t += DT
    presente = (k < 15) or (k > 35)            # se ausenta entre 15 y 35
    voz = "P_call" if (15 <= k <= 18) else "-"  # llama mientras está ausente
    otro = {"OI": (0.3 if presente else 0.0), "necesidad": 0.3, "orientacion_deg": 0.0,
            "voz_emitida": ("P_x" if presente else "-"), "vivo": presente}
    fila = {"t": round(t, 2), "voz_emitida": voz, "OI": 0.3, "necesidad": 0.4,
            "A_sys_env": 0.4, "energia": 0.5, "mem_relacional_confianza": 0.3}
    a = o_c.observar(fila, otro, dt=DT)
    llamo = max(llamo, a["alt_contacto_presencia"]); rec = max(rec, a["alt_contacto_recuperado"])
chk("CONTACTO: emite al ausentarse el otro (llamada)", llamo > 0.5, f"llamada={llamo}")
chk("CONTACTO: registra recuperación cuando el otro vuelve", rec > 0.5, f"recuperado={rec}")

print("-" * 80)
ok = sum(1 for _, p in res if p)
print(f"  RESUMEN: {ok}/{len(res)} PASS")
print("  NOTA honesta: esto valida la LÓGICA del órgano (descubre efecto+beneficio por consecuencia,")
print("  decae sin comunicación, no convenciona con voces barajadas, registra contacto). El AMBIENTE")
print("  COMPARTIDO (confound) sólo se distingue con el control NULL en organismos REALES — por eso el")
print("  órgano MIDE pero no AFIRMA: la prueba final es correr NULL/SHUFFLED en Docker. Aún NO es lenguaje.")
sys.exit(0 if ok == len(res) else 1)
