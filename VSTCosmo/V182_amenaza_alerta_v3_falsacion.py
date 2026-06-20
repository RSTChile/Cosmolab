#!/usr/bin/env python3
"""
================================================================================
V182_amenaza_alerta_v3_falsacion — PONER A PRUEBA v3 (avanzar falsando, no acertando)
================================================================================

v3 dio los 3 peldaños ✅, PERO dejo una gotera honesta: el daño no bajaba con el
tiempo (1a mitad 24 = 2a mitad 24). Eso abre DOS hipotesis, y el trabajo NO es elegir
una: es ponerlas a las dos contra la pared a ver cual cae. Avanzamos por lo que se
ROMPE.

  H1 — "FUE UN INTERRUPTOR, NO APRENDIZAJE":
       la confianza salto de 0 a tope en los primeros eventos y ahi se quedo; no hay
       cuesta de aprendizaje, solo un reflejo que se prendio.
    COMO LA MATO: registro la confianza evento a evento (la pelicula, no la foto) y
    comparo el DAÑO cuando el animal AUN NO confiaba vs cuando YA confiaba. Si el daño
    cae al pasar de "no aprendido" a "aprendido", hay curva de aprendizaje -> H1 muere.

  H2 — "EL DAÑO QUE QUEDA ES DE SONIDOS IMPOSIBLES, NO DE NO-APRENDER":
       los timpanos rotos con vision serian casi todos de rampas muy rapidas (suben tan
       de golpe que ni viendo al otro alcanzas), no de fallar el aviso.
    COMO LA MATO: parto el daño con vision por VELOCIDAD del sonido (rampa) y por estado
    (aprendido / no). Si queda daño en sonidos LENTOS —donde el aviso si deberia
    alcanzar—, entonces no es el ambiente, es que el aviso no se usa bien -> H2 muere.

LAS DOS PUEDEN CAER. No es un menu donde una tiene que ganar. Puede morir H1 y H2
(aprende, pero todavia deja timpanos evitables), pueden sobrevivir las dos, o una. Cada
combinacion dice algo distinto. No busco confirmar; miro que queda en pie.

MECANISMO INTACTO: oidos, regla de anticipacion, mundo y semilla IDENTICOS a v3. Solo
se AGREGAN registros (pelicula de confianza + rampa de cada daño) y los cortes que
matan o dejan viva a cada hipotesis. No se toca ninguna vara.
================================================================================
"""
import os, json, time
import numpy as np
import importlib.util

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("V180", os.path.join(_here, "V180.py"))
V180 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(V180)
DT = V180.DT

# ===== CONSTANTES IDENTICAS A v3 (no se tocan) =====
BANDAS = [-60.0, -30.0, 0.0, 30.0, 60.0]
VOL_INICIAL = 0.8
DOLOR_AGUDO = 3.6
PELIGRO_VOL = 3.2
MORTAL_VOL  = 4.0
CAP_FALSA   = 3.0
RAMPA_MIN, RAMPA_MAX = 0.40, 0.70
MAX_TICKS   = 30
SUBE_ANTICIPA = 0.15
BAJA_FALSA    = 0.20
OLVIDO        = 0.97
UMBRAL_CONF   = 0.5
N_MOMENTOS = 500
P_EVENTO   = 0.5
P_FALSA    = 0.2
PASOS_FUERTE=40000; PASOS_MEDIO=15000; PASOS_COMPART=20000; PASOS_DEBIL=5000
UMBRAL_ACEPTA=5.0; ESCALA_ACEPTA=2.0
SEED_A, SEED_B = 44, 77
TS = time.strftime("%Y%m%d_%H%M%S")


def consolidar(org, banda, pasos):
    for _ in range(pasos):
        org.actualizar_setpoint(0.0, DT, DT, banda, target_reward=banda)

def fase_exposicion(A, B):
    consolidar(A,-60.0,PASOS_FUERTE); consolidar(A,-30.0,PASOS_MEDIO)
    consolidar(A, 30.0,PASOS_DEBIL);  consolidar(A, 60.0,PASOS_DEBIL)
    consolidar(B, 60.0,PASOS_FUERTE); consolidar(B, 30.0,PASOS_MEDIO)
    consolidar(B,-30.0,PASOS_DEBIL);  consolidar(B,-60.0,PASOS_DEBIL)
    consolidar(A,0.0,PASOS_COMPART);  consolidar(B,0.0,PASOS_COMPART)

def p_acepta(org, b):
    v = org.get_valencia(b)
    return 1.0/(1.0+np.exp(-(v-UMBRAL_ACEPTA)/ESCALA_ACEPTA))

def construir_oido(org):
    ps = np.array([p_acepta(org, b) for b in BANDAS])
    sens = 1.0 + 1.6 * (ps - ps.mean())
    return {b: float(np.clip(sens[i], 0.55, 1.5)) for i, b in enumerate(BANDAS)}

class Animal:
    def __init__(self, nombre, oido):
        self.nombre = nombre; self.oido = oido; self.tapado = False
        self.confianza = {b: 0.0 for b in BANDAS}; self.daños = 0

# ===== EVENTO Y APRENDIZAJE: COPIA EXACTA DE v3 =====
def vivir_evento(A, B, banda, es_falsa, con_vision, rampa):
    A.tapado = B.tapado = False
    ct = {"A": None, "B": None}; cr = {"A": None, "B": None}; dño = {"A": False, "B": False}
    cap = CAP_FALSA if es_falsa else (MORTAL_VOL + 1.0); pico = VOL_INICIAL
    for t in range(MAX_TICKS):
        vol = min(cap, VOL_INICIAL + rampa * t); pico = max(pico, vol)
        prev = {"A": A.tapado, "B": B.tapado}
        for animal, otro in ((A, B), (B, A)):
            if animal.tapado: continue
            felt = vol * animal.oido[banda]; taparse, razon = False, None
            if felt > DOLOR_AGUDO: taparse, razon = True, 'propio'
            elif con_vision and prev[otro.nombre] and animal.confianza[banda] > UMBRAL_CONF:
                taparse, razon = True, 'social'
            if taparse:
                animal.tapado = True
                if ct[animal.nombre] is None: ct[animal.nombre] = t; cr[animal.nombre] = razon
            elif vol >= MORTAL_VOL:
                animal.daños += 1; animal.tapado = True; dño[animal.nombre] = True
                if ct[animal.nombre] is None: ct[animal.nombre] = t; cr[animal.nombre] = 'tarde'
        if A.tapado and B.tapado: break
        if es_falsa and vol >= cap and t > 4: break
    real = (not es_falsa) and (pico >= PELIGRO_VOL)
    return ct, cr, dño, real

def aprender(animal, ct_otro, ct_mio, otro_se_tapo, real, banda):
    if otro_se_tapo and real and ct_otro is not None and (ct_mio is None or ct_otro < ct_mio):
        animal.confianza[banda] = min(1.0, animal.confianza[banda] + SUBE_ANTICIPA)
    elif otro_se_tapo and not real:
        animal.confianza[banda] = max(0.0, animal.confianza[banda] - BAJA_FALSA)
    else:
        animal.confianza[banda] *= OLVIDO


# ===== BUCLE CON REGISTRO (la pelicula) =====
def warner_warned(A, B, banda):
    """quien avisa (oido fino) y quien es avisado (oido sordo) en esa banda."""
    if A.oido[banda] > B.oido[banda] + 0.05: return A, B
    if B.oido[banda] > A.oido[banda] + 0.05: return B, A
    return None, None   # 0°: nadie es claramente mejor

def vivir(con_vision, semilla_mundo):
    rmundo = np.random.default_rng(semilla_mundo)
    Ao = V180.OrganismoV180(seed=SEED_A, memoria_episodica=V180.MemoriaEpisodicaV180())
    Bo = V180.OrganismoV180(seed=SEED_B, memoria_episodica=V180.MemoriaEpisodicaV180())
    Ao.set_modo_entrenamiento(False); Bo.set_modo_entrenamiento(False)
    fase_exposicion(Ao, Bo)
    A = Animal("A", construir_oido(Ao)); B = Animal("B", construir_oido(Bo))

    registro = []
    for m in range(N_MOMENTOS):
        if rmundo.random() > P_EVENTO: continue
        banda = BANDAS[rmundo.integers(len(BANDAS))]
        es_falsa = (rmundo.random() < P_FALSA)
        rampa = RAMPA_MIN + (RAMPA_MAX - RAMPA_MIN) * rmundo.random()

        warner, warned = warner_warned(A, B, banda)
        conf_antes = warned.confianza[banda] if warned is not None else None   # ANTES del evento

        ct, cr, dño, real = vivir_evento(A, B, banda, es_falsa, con_vision, rampa)

        if con_vision:
            aprender(A, ct["B"], ct["A"], ct["B"] is not None, real, banda)
            aprender(B, ct["A"], ct["B"], ct["A"] is not None, real, banda)

        conf_despues = warned.confianza[banda] if warned is not None else None
        registro.append({'m': m, 'banda': banda, 'rampa': rampa, 'real': real, 'falsa': es_falsa,
                          'warned': warned.nombre if warned else None,
                          'conf_antes': conf_antes, 'conf_despues': conf_despues,
                          'warned_dañado': (dño[warned.nombre] if warned else False),
                          'daño_total': dño["A"] + dño["B"]})
    return registro, A, B


def tasa(dañados, total):
    return (dañados/total) if total else 0.0


def main():
    print("=" * 98)
    print("V182_amenaza_alerta_v3_falsacion — PONER A PRUEBA v3 (matar H1 y H2)")
    print("=" * 98)
    print("  Avanzamos FALSANDO. ✅ = la hipotesis MURIO (aprendimos algo). ⚠ = SOBREVIVE.")
    print("=" * 98)
    t0 = time.time()

    reg, A, B = vivir(con_vision=True, semilla_mundo=11)
    reg_sin, A0, B0 = vivir(con_vision=False, semilla_mundo=11)
    dcon = sum(e['daño_total'] for e in reg); dsin = sum(e['daño_total'] for e in reg_sin)
    print(f"\n  (recordatorio v3) timpanos: sin vision {dsin}, con vision {dcon}, evitados {dsin-dcon}")

    bandas_warned = [b for b in BANDAS if warner_warned(A, B, b)[1] is not None]

    # =========================================================
    # H1 — ¿INTERRUPTOR O APRENDIZAJE? (la pelicula de la confianza + daño antes/despues)
    # =========================================================
    print(f"\n{'#'*98}\n#  H1 — '¿fue interruptor, no aprendizaje?'  (la pelicula de la confianza)\n{'#'*98}")
    cruces = {}   # cuantos eventos de esa banda hasta cruzar el umbral
    wobble = {}   # cuantas veces la confianza, ya cruzada, volvio a caer bajo el umbral
    dmg_no_aprend = dmg_aprend = 0; ev_no_aprend = ev_aprend = 0
    for b in bandas_warned:
        evs = [e for e in reg if e['banda'] == b]
        traj = [e['conf_despues'] for e in evs]
        # eventos hasta cruzar
        cruce = next((i+1 for i,v in enumerate(traj) if v > UMBRAL_CONF), None)
        cruces[b] = cruce
        # wobble: caidas bajo umbral despues de haber cruzado
        w = 0; arriba = False
        for v in traj:
            if v > UMBRAL_CONF: arriba = True
            elif arriba and v <= UMBRAL_CONF: w += 1; arriba = False
        wobble[b] = w
        # daño antes vs despues de tener confianza (segun conf_antes del evento)
        for e in evs:
            if e['conf_antes'] is None: continue
            if e['conf_antes'] > UMBRAL_CONF:
                ev_aprend += 1; dmg_aprend += int(e['warned_dañado'])
            else:
                ev_no_aprend += 1; dmg_no_aprend += int(e['warned_dañado'])
        # pelicula compacta (primeros ~14 valores)
        peli = " ".join(f"{v:.2f}" for v in traj[:14])
        print(f"    banda {b:>+5.0f} (avisa {warner_warned(A,B,b)[0].nombre}): cruza umbral en evento #{cruce}"
              f"{'' if cruce else ' (nunca)'} ; recaidas={wobble[b]}")
        print(f"        pelicula confianza: {peli}{' ...' if len(traj)>14 else ''}")

    tasa_no = tasa(dmg_no_aprend, ev_no_aprend); tasa_si = tasa(dmg_aprend, ev_aprend)
    cruces_validos = [c for c in cruces.values() if c]
    hubo_cuesta = len(cruces_validos) > 0 and np.median(cruces_validos) >= 2   # no salto en 1 evento
    aprender_reduce = tasa_no > tasa_si + 0.05
    print(f"\n    daño al avisado cuando AUN NO confiaba: {dmg_no_aprend}/{ev_no_aprend} = {tasa_no:.0%}")
    print(f"    daño al avisado cuando YA confiaba:     {dmg_aprend}/{ev_aprend} = {tasa_si:.0%}")
    H1_muere = hubo_cuesta and aprender_reduce
    print(f"  -> {'✅ H1 FALSADA: hubo CUESTA (cruza tras varios eventos) y el daño cae al aprender ('+f'{tasa_no:.0%}->{tasa_si:.0%}'+'). Es aprendizaje, no interruptor.' if H1_muere else '⚠ H1 SOBREVIVE: no se ve cuesta+proteccion (pudo ser reflejo)'}")
    if any(w > 0 for w in wobble.values()):
        print(f"     [hallazgo extra] la confianza RECAE bajo el umbral en algunas bandas (recaidas={ {f'{b:+.0f}':wobble[b] for b in bandas_warned} }):")
        print(f"     las falsas alarmas le hacen 'bajar la guardia' y volver a exponerse -> explica por que el daño no cae parejo con el tiempo.")

    # =========================================================
    # H2 — ¿EL DAÑO RESIDUAL ES DE SONIDOS IMPOSIBLES (rapidos)?
    # =========================================================
    print(f"\n{'#'*98}\n#  H2 — '¿el daño que queda es de sonidos imposibles (rapidos)?'\n{'#'*98}")
    def tercil(r):
        if r < RAMPA_MIN + (RAMPA_MAX-RAMPA_MIN)/3: return 'lento'
        if r < RAMPA_MIN + 2*(RAMPA_MAX-RAMPA_MIN)/3: return 'medio'
        return 'rapido'
    dmg_por_rampa = {'lento':0,'medio':0,'rapido':0}; ev_por_rampa = {'lento':0,'medio':0,'rapido':0}
    dmg_aprend_por_rampa = {'lento':0,'medio':0,'rapido':0}
    for e in reg:
        if e['warned'] is None: continue
        tr = tercil(e['rampa']); ev_por_rampa[tr]+=1
        if e['warned_dañado']:
            dmg_por_rampa[tr]+=1
            if e['conf_antes'] is not None and e['conf_antes'] > UMBRAL_CONF:
                dmg_aprend_por_rampa[tr]+=1
    print(f"    daño con vision por velocidad del sonido:")
    print(f"      {'velocidad':>8} | daño | eventos | daño aun teniendo confianza")
    for tr in ('lento','medio','rapido'):
        print(f"      {tr:>8} | {dmg_por_rampa[tr]:>4} | {ev_por_rampa[tr]:>7} | {dmg_aprend_por_rampa[tr]:>4}")
    daño_en_lentos = dmg_por_rampa['lento']
    H2_muere = daño_en_lentos > 0
    print(f"  -> {'✅ H2 FALSADA: queda daño en sonidos LENTOS ('+str(daño_en_lentos)+'), donde el aviso SI alcanzaba. No es el ambiente: el aviso no se usa bien (la confianza recae).' if H2_muere else '⚠ H2 SOBREVIVE: el daño residual es solo de sonidos rapidos (imposibles)'}")

    # =========================================================
    # LO QUE QUEDA EN PIE
    # =========================================================
    print(f"\n{'='*98}\n  QUE QUEDO EN PIE (avanzamos por lo que se rompe)\n{'='*98}")
    print(f"    H1 'interruptor, no aprende'          -> {'MUERTA ✅' if H1_muere else 'viva ⚠'}")
    print(f"    H2 'daño residual = sonidos imposibles'-> {'MUERTA ✅' if H2_muere else 'viva ⚠'}")
    if H1_muere and H2_muere:
        print(f"\n    LECTURA: el animal SI aprende (hay cuesta y el daño cae al aprender), PERO todavia")
        print(f"    deja timpanos evitables en sonidos lentos. Causa probable: su confianza RECAE por")
        print(f"    falsas alarmas y baja la guardia. El proximo paso no es 'avanzar a V183': es que la")
        print(f"    confianza, una vez ganada, no se pierda tan facil. (Eso explica el daño plano de v3.)")
    elif H1_muere and not H2_muere:
        print(f"\n    LECTURA: aprende, y el daño que queda es de sonidos imposibles. La proteccion es tan")
        print(f"    buena como el mundo permite. Aqui SI se podria dar amenaza por cerrado.")
    elif not H1_muere:
        print(f"\n    LECTURA: no se ve aprendizaje real (pudo ser reflejo). Hay que revisar la regla de")
        print(f"    confianza antes que nada — no es un problema del mundo.")

    print(f"\n  tiempo {time.time()-t0:.1f}s")
    os.makedirs("V182_logs", exist_ok=True)
    salida = {'version':'V182_amenaza_alerta_v3_falsacion',
              'timpanos_sin':int(dsin),'timpanos_con':int(dcon),
              'cruces':{f'{b:+.0f}':cruces[b] for b in bandas_warned},
              'recaidas':{f'{b:+.0f}':wobble[b] for b in bandas_warned},
              'daño_no_aprendido':int(dmg_no_aprend),'eventos_no_aprendido':int(ev_no_aprend),
              'daño_aprendido':int(dmg_aprend),'eventos_aprendido':int(ev_aprend),
              'tasa_no_aprend':float(tasa_no),'tasa_aprend':float(tasa_si),
              'daño_por_rampa':dmg_por_rampa,'daño_aprendido_por_rampa':dmg_aprend_por_rampa,
              'H1_muere':bool(H1_muere),'H2_muere':bool(H2_muere)}
    with open(f"V182_logs/v182_amenaza_falsacion_{TS}.json","w") as f:
        json.dump(salida, f, indent=2)
    print(f"  log: V182_logs/v182_amenaza_falsacion_{TS}.json")


if __name__ == "__main__":
    main()
