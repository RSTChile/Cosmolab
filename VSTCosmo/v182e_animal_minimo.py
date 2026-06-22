#!/usr/bin/env python3
"""
V182E_animal_minimo — DE PLANTA A ANIMAL: ESTADO INTERNO QUE CAMBIA (cuerpo V180)
================================================================================

POR QUE HACEMOS ESTO (en simple)
--------------------------------
Hasta aqui (V182A–D) el organismo era una PLANTA: solo decia "esto me gusta /
no me gusta" y siempre lo mismo. Predecirlo era facil porque nunca cambiaba. Por
eso la prueba de "alma" de V183 sale amañada hacia el "no": si el otro siempre
hace lo mismo, anticiparlo no es informacion, es obviedad. Una planta medida por
sus impulsos electricos.

Un ANIMAL es distinto: tiene estados internos que cambian (hambre, cansancio) y
eso cambia lo que hace. El mismo estimulo, distinta respuesta segun como esta por
dentro. Y un segundo animal aprende a leer ESE estado que cambia, no una etiqueta
fija. Recien ahi "predecir al otro" es informacion de verdad, y V183 deja de estar
amañada.

QUE ENRIQUECEMOS (las tres cosas acordadas)
-------------------------------------------
1. ESTADO INTERNO QUE CAMBIA — y NO lo inventamos: usamos el que V180 YA tiene,
   la FATIGA (`motor.fatiga.fatiga_activa`). Sube cuando el organismo se esfuerza
   por orientarse; se recupera cuando descansa. Ya modula su conducta (cuando esta
   cansado: zona muerta mas ancha, mas temblor, menos ganancia -> orienta peor).
   => La "aceptacion" de una banda deja de ser fija: depende de la fatiga del
      momento. Fresco acepta crujiente; cansado, peor; descansado, se recupera.
2. MAS OPCIONES — 5 bandas [-60,-30,0,30,60] en vez de 3, para romper el "2 de 3
   siempre rechaza" que aplastaba la informacion mutua.
3. EL OTRO LEE EL ESTADO QUE CAMBIA — un modelo "planta" (supone aceptacion fija)
   vs un modelo "animal" (condiciona en la fatiga observable). Si el modelo animal
   predice mejor, el estado interno es LEGIBLE y aporta informacion -> es justo lo
   que V183 necesita.

HONESTIDAD SOBRE EL ALCANCE
---------------------------
Esto es el ANIMAL MINIMO: demuestra el MECANISMO (conducta atada a un estado
interno real que fluctua, + mas bandas + estado legible). Cuanto OSCILA la fatiga
y si alcanza para que I(A;B) suba en V183 lo dice la corrida. Si la oscilacion sale
debil (fresco ≈ cansado), eso NO es un fracaso: es señal de que hay que reforzar la
dinamica interna, y se lee. No se afirma de antemano que basta.

NOTA DE MAPA (importante)
-------------------------
En el roadmap "V182E" ya significaba NEGOCIACION. ESTO NO ES NEGOCIACION: es
enriquecer el cuerpo. Se usa el nombre de archivo pedido por el IP, pero queda
PENDIENTE decidir a que numero se mueve "negociacion" para no confundir el mapa.

CUERPO: V180 importado VERBATIM. No se reescribe ni se inventa ningun "organo".
================================================================================
"""
import os, json, time
import numpy as np
import importlib.util

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("V180", os.path.join(_here, "V180.py"))
V180 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(V180)
DT = V180.DT

# ============================================================
# CONFIG
# ============================================================
BANDAS = [-60.0, -30.0, 0.0, 30.0, 60.0]          # 5 bandas (antes 3): rompe el 2-1

# Exposicion graduada (FIDELIDAD chica para "minimo"; subir para corrida fina)
PASOS_FUERTE  = 40000   # banda nativa
PASOS_MEDIO   = 15000   # banda vecina
PASOS_COMPART = 20000   # banda compartida (0°)
PASOS_DEBIL   = 5000    # bandas lejanas

# Probes y bloques de actividad/descanso (es lo que hace fluctuar la fatiga)
PASOS_PROBE     = 600    # un probe de aceptacion de una banda
PASOS_TRABAJO   = 4000   # bloque de esfuerzo (sube fatiga)
PASOS_DESCANSO  = 4000   # bloque de reposo (recupera fatiga)
TOL_ACEPTA      = 8.0    # yardstick FIJO para "se asento en la banda" (cableado, no veredicto)

SEED_A, SEED_B = 44, 77
TS = time.strftime("%Y%m%d_%H%M%S")


# ============================================================
# CUERPO
# ============================================================
def consolidar(org, banda, pasos):
    for _ in range(pasos):
        org.actualizar_setpoint(0.0, DT, DT, banda, target_reward=banda)

def fase_exposicion(A, B):
    # A experto en -60 (y -30 medio); B experto en +60 (y +30 medio). Competencia GRADUADA.
    consolidar(A, -60.0, PASOS_FUERTE); consolidar(A, -30.0, PASOS_MEDIO)
    consolidar(A,  30.0, PASOS_DEBIL);  consolidar(A,  60.0, PASOS_DEBIL)
    consolidar(B,  60.0, PASOS_FUERTE); consolidar(B,  30.0, PASOS_MEDIO)
    consolidar(B, -30.0, PASOS_DEBIL);  consolidar(B, -60.0, PASOS_DEBIL)
    consolidar(A, 0.0, PASOS_COMPART);  consolidar(B, 0.0, PASOS_COMPART)


# ============================================================
# CONDUCTA ATADA AL ESTADO INTERNO (fatiga real de V180)
# ============================================================
def fatiga(org):
    return org.motor.fatiga.get_fatiga()

def probe_acepta(org, banda, t0):
    """Corre el motor real hacia 'banda' un rato. 'Acepta' = fraccion del tramo
    final en que quedo cerca de la banda (yardstick fijo TOL_ACEPTA). Como la fatiga
    ensancha la zona muerta y mete temblor, un organismo CANSADO se asienta peor:
    la aceptacion depende del estado interno del momento. Devuelve (acepta, fatiga)."""
    cerca = total = 0
    for s in range(PASOS_PROBE):
        t = t0 + s * DT
        org.actualizar_con_opciones(t, DT, t0 + PASOS_PROBE*DT, [banda], False, None)
        if s >= int(PASOS_PROBE * 0.5):
            total += 1
            if abs(org.motor.orientacion - banda) < TOL_ACEPTA:
                cerca += 1
    return (cerca / total if total else 0.0), fatiga(org)

def trabajar_duro(org, t0):
    """Esfuerzo: alterna blancos lejanos -> errores grandes -> sube la fatiga."""
    for s in range(PASOS_TRABAJO):
        t = t0 + s * DT
        objetivo = 60.0 if (s // 200) % 2 == 0 else -60.0
        org.actualizar_con_opciones(t, DT, t0 + PASOS_TRABAJO*DT, [objetivo], False, None)

def descansar(org, t0):
    """Reposo: orienta a su propia posicion -> error ~0 -> la fatiga se recupera."""
    pos = org.motor.orientacion
    for s in range(PASOS_DESCANSO):
        t = t0 + s * DT
        org.actualizar_con_opciones(t, DT, t0 + PASOS_DESCANSO*DT, [pos], False, None)


def sesion(org, etiqueta):
    """Fresco -> probar | Trabajo -> probar (cansado) | Descanso -> probar (recuperado).
    Registra aceptacion de cada banda en cada fase, con la fatiga del momento."""
    t = 0.0
    fases = {}
    for fase, accion in (("fresco", None), ("cansado", trabajar_duro), ("recuperado", descansar)):
        if accion is not None:
            accion(org, t); t += (PASOS_TRABAJO if accion is trabajar_duro else PASOS_DESCANSO) * DT
        fila = {}
        for b in BANDAS:
            acepta, fat = probe_acepta(org, b, t); t += PASOS_PROBE * DT
            fila[b] = {'acepta': acepta, 'fatiga': fat}
        fases[fase] = fila
    return fases


# ============================================================
# ¿ES ANIMAL O PLANTA?  (la disposicion cambia con el estado interno)
# ============================================================
def evaluar_animalidad(fases):
    """Planta = la aceptacion de una banda NO cambia entre fresco/cansado.
       Animal = cambia. Reporta el swing por banda y un veredicto simple."""
    out = {}
    for b in BANDAS:
        a_fresco = fases['fresco'][b]['acepta']
        a_cansado = fases['cansado'][b]['acepta']
        a_recup  = fases['recuperado'][b]['acepta']
        swing = abs(a_fresco - a_cansado)
        out[b] = {'fresco': a_fresco, 'cansado': a_cansado, 'recuperado': a_recup, 'swing': swing}
    swing_max = max(out[b]['swing'] for b in BANDAS)
    return out, swing_max


# ============================================================
# EL OTRO LEE EL ESTADO QUE CAMBIA  (modelo planta vs modelo animal)
# ============================================================
def lectura_del_otro(fases_obj):
    """Junta puntos (fatiga, acepta) del objetivo sobre todas las bandas/fases.
       Modelo PLANTA: predice la aceptacion media (ignora el estado interno).
       Modelo ANIMAL: predice segun la fatiga (2 regimenes: baja vs alta fatiga).
       Si el animal predice mejor, el estado interno es LEGIBLE -> aporta informacion."""
    pts = []
    for fase in fases_obj:
        for b in fases_obj[fase]:
            pts.append((fases_obj[fase][b]['fatiga'], fases_obj[fase][b]['acepta']))
    fats = np.array([p[0] for p in pts]); accs = np.array([p[1] for p in pts])
    media = float(accs.mean())
    err_planta = float(np.mean((accs - media) ** 2))
    corte = float(np.median(fats))
    baja = accs[fats <= corte]; alta = accs[fats > corte]
    m_baja = float(baja.mean()) if len(baja) else media
    m_alta = float(alta.mean()) if len(alta) else media
    pred = np.where(fats <= corte, m_baja, m_alta)
    err_animal = float(np.mean((accs - pred) ** 2))
    mejora = (err_planta - err_animal) / err_planta if err_planta > 1e-9 else 0.0
    return {'media_acepta': media, 'err_planta': err_planta, 'err_animal': err_animal,
            'acepta_baja_fatiga': m_baja, 'acepta_alta_fatiga': m_alta,
            'mejora_leyendo_estado': float(mejora)}


# ============================================================
# CORRIDA
# ============================================================
def main():
    print("=" * 96)
    print("V182E_animal_minimo — DE PLANTA A ANIMAL: estado interno (fatiga) que cambia")
    print("=" * 96)
    print("  Idea: la aceptacion de una banda depende de la FATIGA del momento (estado interno")
    print("  real de V180), no es fija. 5 bandas. El otro debe LEER ese estado que cambia.")
    print("=" * 96)
    t0 = time.time()

    A = V180.OrganismoV180(seed=SEED_A, memoria_episodica=V180.MemoriaEpisodicaV180())
    B = V180.OrganismoV180(seed=SEED_B, memoria_episodica=V180.MemoriaEpisodicaV180())
    A.set_modo_entrenamiento(False); B.set_modo_entrenamiento(False)
    fase_exposicion(A, B)

    sesion_A = sesion(A, "A"); sesion_B = sesion(B, "B")
    anim_A, swingmax_A = evaluar_animalidad(sesion_A)
    anim_B, swingmax_B = evaluar_animalidad(sesion_B)
    lee_A_de_B = lectura_del_otro(sesion_B)   # A leeria a B
    lee_B_de_A = lectura_del_otro(sesion_A)   # B leeria a A

    def tabla_animalidad(nombre, anim):
        print(f"\n  ¿{nombre} es animal? (aceptacion por banda: fresco -> cansado -> recuperado)")
        print(f"    {'banda':>6} | fresco cansado recup | swing")
        print(f"    {'-'*6}-+-{'-'*21}-+------")
        for b in BANDAS:
            r = anim[b]
            print(f"    {b:>+6.0f} | {r['fresco']:>5.0%} {r['cansado']:>6.0%} {r['recuperado']:>6.0%} | {r['swing']:>4.0%}")

    tabla_animalidad("A", anim_A)
    tabla_animalidad("B", anim_B)

    print(f"\n  [fatiga por fase]  A: fresco {sesion_A['fresco'][BANDAS[0]]['fatiga']:.0f} -> "
          f"cansado {sesion_A['cansado'][BANDAS[0]]['fatiga']:.0f} -> recup {sesion_A['recuperado'][BANDAS[0]]['fatiga']:.0f}")
    print(f"                     B: fresco {sesion_B['fresco'][BANDAS[0]]['fatiga']:.0f} -> "
          f"cansado {sesion_B['cansado'][BANDAS[0]]['fatiga']:.0f} -> recup {sesion_B['recuperado'][BANDAS[0]]['fatiga']:.0f}")

    print(f"\n{'#'*96}\n#  ¿ANIMAL O PLANTA?\n{'#'*96}")
    print(f"  swing maximo de aceptacion (fresco vs cansado):  A {swingmax_A:.0%}   B {swingmax_B:.0%}")
    es_animal = (swingmax_A > 0.10) and (swingmax_B > 0.10)
    print(f"  -> {'✅ ANIMAL: la disposicion cambia con el estado interno' if es_animal else '⚠ todavia PLANTA: la fatiga casi no movio la conducta (reforzar dinamica interna)'}")

    print(f"\n{'#'*96}\n#  ¿EL OTRO PUEDE LEER EL ESTADO INTERNO?\n{'#'*96}")
    for nom, L in (("A lee a B", lee_A_de_B), ("B lee a A", lee_B_de_A)):
        print(f"  {nom}: acepta(baja fatiga)={L['acepta_baja_fatiga']:.0%}  acepta(alta fatiga)={L['acepta_alta_fatiga']:.0%}  "
              f"-> leer el estado mejora la prediccion en {L['mejora_leyendo_estado']:+.0%}")
    legible = lee_A_de_B['mejora_leyendo_estado'] > 0.05 and lee_B_de_A['mejora_leyendo_estado'] > 0.05
    print(f"  -> {'✅ el estado interno es LEGIBLE: aporta informacion (lo que V183 necesita)' if legible else '⚠ el estado interno aun no aporta informacion legible'}")

    print(f"\n  LECTURA: si ANIMAL ✅ y LEGIBLE ✅, ANIMA dejo de ser planta y V183 tiene algo que medir.")
    print(f"  Si alguno sale ⚠, la dinamica interna es debil y hay que reforzarla (no es fracaso, es dato).")
    print(f"\n  tiempo {time.time()-t0:.1f}s")

    os.makedirs("V182_logs", exist_ok=True)
    salida = {'version': 'V182E_animal_minimo',
              'bandas': BANDAS,
              'animalidad_A': {str(k): v for k, v in anim_A.items()},
              'animalidad_B': {str(k): v for k, v in anim_B.items()},
              'swing_max_A': float(swingmax_A), 'swing_max_B': float(swingmax_B),
              'es_animal': bool(es_animal),
              'A_lee_a_B': lee_A_de_B, 'B_lee_a_A': lee_B_de_A,
              'estado_legible': bool(legible)}
    with open(f"V182_logs/v182e_animal_minimo_{TS}.json", "w") as f:
        json.dump(salida, f, indent=2)
    print(f"  log: V182_logs/v182e_animal_minimo_{TS}.json")


if __name__ == "__main__":
    main()