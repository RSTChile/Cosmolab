#!/usr/bin/env python3
"""
V182D — RECONOCIMIENTO DE ALTERIDAD (Subj_sem) SOBRE CUERPO V180 REAL
================================================================================
Sigue de V182C. Es el NODO-GATE de la escalera: hace interpretable a V183.
Sin D, la irreductibilidad de V183 (compresion, I(A;B), sincronizacion) es
desinflable como "solo dos variables acopladas". Con D —cada organismo modela al
OTRO como sujeto con disposicion propia y predice su conducta autonoma— V183 pasa
a medir irreductibilidad de una estructura de MODELADO MUTUO entre sujetos.

CUERPO: V180 importado VERBATIM, como A.3/A.4/A.5 (NO como B-v9 ni C). Innegociable:
el gate debe correr sobre la misma dinada de organismos reales que V183.

PREGUNTA (O-N3.4 / Subj_sem): tras la exposicion diferencial, ¿A puede PREDECIR la
disposicion de B (y B la de A) modelandola desde AFUERA —observando sus elecciones—,
y no solo PROYECTANDO la propia? La prediccion debe jugarse donde las disposiciones
DIVERGEN; si solo coinciden, proyectar ya acierta y no hay alteridad que demostrar.

LA TRAMPA (la misma que tumbo los primeros disenos de C):
  Tras A.5, A y B convergieron. Si D midiera "A predice a B y acierta", se pasa
  TRIVIALMENTE por proyeccion: A predice "B elige lo que yo elegiria" y acierta
  porque convergieron, sin modelar a B. Eso es ECO, no alteridad.

EL CONTROL QUE LO VUELVE HONESTO (misma disciplina ⊘ de B-v9):
  Por banda se mide la prediccion MODELANDO al otro contra la LINEA BASE DE
  PROYECCION (predecir "el otro hace lo que yo hago"):
        genuino(banda) = acierto_modelo - acierto_proyeccion
  Si en una banda la proyeccion YA acierta (banda compartida 0°, donde no
  divergen), esa posicion es DEGENERADA -> ⊘ (fuera de test), ni ✅ ni ❌, igual
  que el centroide en B-v9. Asi un ✅ significa SIEMPRE que el organismo modelo la
  disposicion DISTINTA del otro, no que proyecto la propia.

BIDIRECCIONAL (decision del IP): A modela a B Y B modela a A. El ✅ de D exige
  alteridad genuina en AMBAS direcciones, en las bandas medibles.

ADVERTENCIA MUTUA SIN OVERRIDE (F3, decision del IP — se MIDE, NO gatea):
  Cada uno, viendo por su modelo que el otro va hacia una banda donde el otro es
  incompetente, ajusta su PROPIA conducta como senal. NUNCA escribe la valencia
  del otro (no decide por el otro). Se reporta su magnitud; NO decide el ✅.
  Nota honesta: en el regimen post-exposicion las disposiciones son nitidas, asi
  que la advertencia tiene poco sobre que actuar; la advertencia con COSTO real es
  territorio de F (empatia), donde se reintroduce costo asimetrico.

MARGEN GENUINO: ⚠ PENDIENTE DE CONFIRMACION DEL IP. Valor provisional 0.15 por
  paralelo con UMBRAL_GENUINO de B-v9 (misma estructura real-nulo). DEBE fijarse
  ANTES de la corrida de fidelidad; mientras MARGEN_CONFIRMADO=False el veredicto
  se imprime como PROVISIONAL.

ABLACION de D: MODELO (Subj_sem: predice desde el modelo observacional del otro)
  vs PROYECCION (eco: predice que el otro hace lo propio). La proyeccion ES la
  linea base nula; no es un segundo run aparte.

MARCADORES: ✅ alteridad genuina · ❌ banda medible sin alteridad · ⊘ degenerada
  (proyeccion ya acierta: medidor sin poder) · · n/a
================================================================================
"""
import os, json, time
import numpy as np
import importlib.util

# ---- importar el cuerpo V180 real (debe estar junto a este archivo) ----
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("V180", os.path.join(_here, "V180.py"))
V180 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(V180)
DT = V180.DT

# ============================================================
# CONFIG
# ============================================================
BANDAS = [-60.0, 0.0, 60.0]

# Exposicion diferencial (FIDELIDAD, igual escala que A.5: val nativa ~10-12)
PASOS_FUERTE   = 80000     # banda nativa
PASOS_DEBIL    = 8000      # banda ajena
PASOS_COMPART  = 40000     # banda compartida (0°)

N_OBS   = 200    # episodios de observacion para construir el modelo del otro
N_PRED  = 200    # trials de prediccion por banda y direccion

# --- parametros de CABLEADO (no son el umbral de veredicto) ---
# Aceptacion de una banda a partir de la valencia: nativa(~12)->acepta,
# ajena(~1-3)->rechaza, compartida(~6-8)->ambos aceptan. Separa divergencia de
# coincidencia; NO decide el ✅.
UMBRAL_ACEPTA = 5.0
ESCALA_ACEPTA = 2.0

# --- UMBRAL DE VEREDICTO (gating) ---
# ⚠⚠⚠ PENDIENTE: confirmar/fijar con el IP ANTES de la corrida de fidelidad. ⚠⚠⚠
MARGEN_GENUINO   = 0.15     # provisional, por paralelo con UMBRAL_GENUINO de B-v9
MARGEN_CONFIRMADO = False   # poner True solo cuando el IP fije el valor

SEED_A, SEED_B = 44, 77
TS = time.strftime("%Y%m%d_%H%M%S")


# ============================================================
# CUERPO: exposicion = consolidacion REAL (como A.4/A.5), campo no contador
# ============================================================
def consolidar(org, banda, pasos):
    for _ in range(pasos):
        org.actualizar_setpoint(0.0, DT, DT, banda, target_reward=banda)


def fase_exposicion(A, B):
    # A nativo -60°, B nativo +60°; ambos comparten 0°. Disposiciones EMERGENTES.
    consolidar(A, -60.0, PASOS_FUERTE); consolidar(A, 60.0, PASOS_DEBIL)
    consolidar(B,  60.0, PASOS_FUERTE); consolidar(B, -60.0, PASOS_DEBIL)
    consolidar(A, 0.0, PASOS_COMPART);  consolidar(B, 0.0, PASOS_COMPART)


# ============================================================
# CONDUCTA EXTERNA OBSERVABLE: aceptar/rechazar una banda (depende de valencia)
# El otro SOLO observa esta conducta; nunca los internos.
# ============================================================
def p_acepta(org, banda):
    v = org.get_valencia(banda)
    return 1.0 / (1.0 + np.exp(-(v - UMBRAL_ACEPTA) / ESCALA_ACEPTA))

def actuar(org, banda, rng):
    """Conducta externa observable: 1 = acepta/orienta a la banda, 0 = rechaza."""
    return 1 if rng.random() < p_acepta(org, banda) else 0


# ============================================================
# MODELO OBSERVACIONAL DEL OTRO (Subj_sem): frecuencia de aceptacion por banda
# Se construye SOLO con la conducta externa observada del otro.
# ============================================================
def observar_al_otro(otro, rng):
    modelo = {}
    for b in BANDAS:
        acc = [actuar(otro, b, rng) for _ in range(N_OBS)]
        modelo[b] = float(np.mean(acc))     # P_obs(otro acepta b)
    return modelo


# ============================================================
# PREDICCION: modelo (Subj_sem) vs proyeccion (eco), por banda
# ============================================================
def evaluar_prediccion(predictor, objetivo, modelo_del_objetivo, rng_obj, rng_pred):
    """
    Por banda: el objetivo actua (verdad de campo). El predictor predice con:
      - MODELO    : argmax de su modelo observacional del objetivo
      - PROYECCION: lo que el predictor mismo haria en esa banda (eco)
    Devuelve, por banda: acc_modelo, acc_proyeccion, genuino, degenerada.
    """
    filas = {}
    for b in BANDAS:
        pred_modelo = 1 if modelo_del_objetivo[b] >= 0.5 else 0     # disposicion observada del otro
        pred_proy   = 1 if p_acepta(predictor, b) >= 0.5 else 0     # eco: "el otro hace lo que yo hago"
        aciertos_m, aciertos_p = 0, 0
        for _ in range(N_PRED):
            verdad = actuar(objetivo, b, rng_obj)                  # conducta real del objetivo
            aciertos_m += int(pred_modelo == verdad)
            aciertos_p += int(pred_proy   == verdad)
        acc_m = aciertos_m / N_PRED
        acc_p = aciertos_p / N_PRED
        genuino = acc_m - acc_p
        # degenerada: la proyeccion ya acierta sola -> el medidor no tiene poder aqui
        degenerada = acc_p >= (1.0 - MARGEN_GENUINO)
        filas[b] = {'acc_modelo': acc_m, 'acc_proyeccion': acc_p,
                    'genuino': genuino, 'degenerada': bool(degenerada)}
    return filas


# ============================================================
# F3 — ADVERTENCIA MUTUA SIN OVERRIDE (se mide, NO gatea, no decide por el otro)
# ============================================================
def medir_advertencia(predictor, objetivo, modelo_del_objetivo, banda_nativa_predictor):
    """
    Si el modelo del predictor anticipa que el objetivo va a RECHAZAR la banda en
    que el predictor es experto (banda donde podria 'avisarle'), el predictor
    ajusta su PROPIA conducta como senal. NO toca la valencia del objetivo.
    Reporta la magnitud del ajuste propio (0..1). NO gatea el veredicto de D.
    """
    b = banda_nativa_predictor
    objetivo_rechaza_b = modelo_del_objetivo[b] < 0.5        # el otro va a una eleccion subexperta segun mi modelo
    # 'ajuste' = cuanto refuerzo mi propia aceptacion de b como senal (no modifico al otro)
    ajuste = (p_acepta(predictor, b) if objetivo_rechaza_b else 0.0)
    return {'banda': b, 'predijo_rechazo_del_otro': bool(objetivo_rechaza_b),
            'ajuste_propio': float(ajuste)}


# ============================================================
# CORRIDA
# ============================================================
def correr():
    A = V180.OrganismoV180(seed=SEED_A, memoria_episodica=V180.MemoriaEpisodicaV180())
    B = V180.OrganismoV180(seed=SEED_B, memoria_episodica=V180.MemoriaEpisodicaV180())
    A.set_modo_entrenamiento(False); B.set_modo_entrenamiento(False)

    fase_exposicion(A, B)

    disp = {b: (round(p_acepta(A, b), 2), round(p_acepta(B, b), 2)) for b in BANDAS}

    rng = np.random.default_rng(12345)
    # fase de observacion bidireccional (cada uno modela las elecciones del otro)
    modelo_A_de_B = observar_al_otro(B, np.random.default_rng(101))
    modelo_B_de_A = observar_al_otro(A, np.random.default_rng(202))

    # prediccion bidireccional
    AB = evaluar_prediccion(A, B, modelo_A_de_B, np.random.default_rng(303), np.random.default_rng(304))
    BA = evaluar_prediccion(B, A, modelo_B_de_A, np.random.default_rng(305), np.random.default_rng(306))

    # F3 (no gatea): A experto en -60, B experto en +60
    adv_AB = medir_advertencia(A, B, modelo_A_de_B, -60.0)
    adv_BA = medir_advertencia(B, A, modelo_B_de_A,  60.0)

    return {'disp': disp, 'modelo_A_de_B': modelo_A_de_B, 'modelo_B_de_A': modelo_B_de_A,
            'AB': AB, 'BA': BA, 'adv_AB': adv_AB, 'adv_BA': adv_BA}


def _marcador(fila):
    if fila['degenerada']:
        return '⊘'
    return '✅' if fila['genuino'] > MARGEN_GENUINO else '❌'

def _imprimir_direccion(titulo, filas):
    print(f"\n  {titulo}")
    print(f"    {'banda':>6} | {'acc_modelo':>10} {'acc_proy':>9} | {'genuino':>8} | marca")
    print(f"    {'-'*6}-+-{'-'*20}-+-{'-'*8}-+------")
    for b in BANDAS:
        f = filas[b]; mk = _marcador(f)
        print(f"    {b:>+6.0f} | {f['acc_modelo']:>10.0%} {f['acc_proyeccion']:>9.0%} | {f['genuino']:>+8.0%} | [{mk}]")


def main():
    print("=" * 96)
    print("V182D — RECONOCIMIENTO DE ALTERIDAD (Subj_sem) — cuerpo V180 real")
    print("=" * 96)
    print("  Ablacion: MODELO (predice desde el modelo observacional del otro) vs")
    print("            PROYECCION (eco: 'el otro hace lo que yo hago' = linea base nula).")
    print("  genuino(banda) = acc_modelo - acc_proyeccion. Banda donde la proyeccion ya")
    print("  acierta (0° compartida) -> ⊘ (degenerada, fuera de test).")
    print(f"  CRITERIO (gating): genuino > MARGEN = {MARGEN_GENUINO:.0%}"
          + ("" if MARGEN_CONFIRMADO else "   ⚠ PROVISIONAL — sin confirmar por el IP"))
    print("  LEYENDA: ✅ alteridad genuina · ❌ medible sin alteridad · ⊘ degenerada")
    print("=" * 96)

    t0 = time.time()
    R = correr()

    print(f"\n  [disposicion emergente tras exposicion — P(acepta) por banda]")
    print(f"     {'banda':>6} |   A      B")
    for b in BANDAS:
        pa, pb = R['disp'][b]
        print(f"     {b:>+6.0f} | {pa:>4.2f}  {pb:>4.2f}")

    print(f"\n  [modelo observacional — P_obs(el otro acepta la banda)]")
    print(f"     {'banda':>6} | A-de-B  B-de-A")
    for b in BANDAS:
        print(f"     {b:>+6.0f} |  {R['modelo_A_de_B'][b]:>4.2f}   {R['modelo_B_de_A'][b]:>4.2f}")

    _imprimir_direccion("DIRECCION  A modela a B:", R['AB'])
    _imprimir_direccion("DIRECCION  B modela a A:", R['BA'])

    # veredicto bidireccional sobre bandas MEDIBLES (no degeneradas)
    def resumen(filas):
        medibles = [b for b in BANDAS if not filas[b]['degenerada']]
        verdes   = [b for b in medibles if filas[b]['genuino'] > MARGEN_GENUINO]
        degen    = [b for b in BANDAS if filas[b]['degenerada']]
        return medibles, verdes, degen

    med_AB, ver_AB, deg_AB = resumen(R['AB'])
    med_BA, ver_BA, deg_BA = resumen(R['BA'])
    ok_AB = len(ver_AB) >= 1 and len(ver_AB) == len(med_AB)
    ok_BA = len(ver_BA) >= 1 and len(ver_BA) == len(med_BA)
    alteridad_bidireccional = ok_AB and ok_BA

    print(f"\n{'#'*96}\n#  VEREDICTO\n{'#'*96}")
    print(f"  A->B: {len(ver_AB)}/{len(med_AB)} bandas medibles con alteridad genuina   degeneradas {deg_AB}")
    print(f"  B->A: {len(ver_BA)}/{len(med_BA)} bandas medibles con alteridad genuina   degeneradas {deg_BA}")
    print(f"\n  [F3 advertencia (NO gatea, no override): "
          f"A->B ajuste {R['adv_AB']['ajuste_propio']:.2f} (predijo rechazo={R['adv_AB']['predijo_rechazo_del_otro']}) | "
          f"B->A ajuste {R['adv_BA']['ajuste_propio']:.2f} (predijo rechazo={R['adv_BA']['predijo_rechazo_del_otro']})]")

    estado = "✅ ALTERIDAD BIDIRECCIONAL" if alteridad_bidireccional else "❌ no bidireccional"
    print(f"\n  {estado}")
    if not MARGEN_CONFIRMADO:
        print("  ⚠ VEREDICTO PROVISIONAL: el MARGEN ({:.0%}) no ha sido confirmado por el IP.".format(MARGEN_GENUINO))
        print("    Fijar MARGEN_GENUINO y poner MARGEN_CONFIRMADO=True antes de la corrida de fidelidad.")
    print(f"\n  tiempo {time.time()-t0:.1f}s")

    os.makedirs("V182_logs", exist_ok=True)
    salida = {'version': 'V182D-alteridad', 'margen': MARGEN_GENUINO,
              'margen_confirmado': bool(MARGEN_CONFIRMADO),
              'disposicion': {str(k): v for k, v in R['disp'].items()},
              'modelo_A_de_B': {str(k): v for k, v in R['modelo_A_de_B'].items()},
              'modelo_B_de_A': {str(k): v for k, v in R['modelo_B_de_A'].items()},
              'AB': {str(k): v for k, v in R['AB'].items()},
              'BA': {str(k): v for k, v in R['BA'].items()},
              'adv_AB': R['adv_AB'], 'adv_BA': R['adv_BA'],
              'ok_AB': bool(ok_AB), 'ok_BA': bool(ok_BA),
              'alteridad_bidireccional': bool(alteridad_bidireccional)}
    with open(f"V182_logs/v182d_alteridad_{TS}.json", "w") as f:
        json.dump(salida, f, indent=2)
    print(f"  log: V182_logs/v182d_alteridad_{TS}.json")


if __name__ == "__main__":
    main()