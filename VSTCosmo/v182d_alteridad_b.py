#!/usr/bin/env python3
"""
V182D_alteridad_B — RECONOCIMIENTO DE ALTERIDAD (Subj_sem) SOBRE CUERPO V180 REAL
================================================================================
Corrige V182D_alteridad.py. El experimento NO fallo; fallo el criterio de
degeneracion del medidor. Esta version (_B) lo alinea con la intencion
pre-declarada del protocolo.

BITACORA DE LA CORRECCION (honestidad — Libertad Funcional):
  Corrida A: ±60° dieron alteridad genuina masiva (+63% a +86% sobre proyeccion,
  bidireccional) y 0° dio genuino 0%. Pero 0° se marco ❌ en vez de ⊘, y eso
  envenenó el veredicto (`verdes == medibles` falla con 0° contado como medible).

  CAUSA: importe por analogia floja el criterio de B-v9, `degenerada = acc_proy >=
  (1 - MARGEN)` (≈0.85). En 0° la proyeccion acierta 0.64 (< 0.85), asi que NO se
  marco degenerada. El concepto declarado de degeneracion ("banda compartida donde
  las disposiciones NO divergen") no es "la proyeccion acierta casi perfecto".

  EQUIPO: unanime en el diagnostico (bug del gate, no fallo del fenomeno).
  Propusieron tres parches; NINGUNO se adopta, por meter parametro libre o
  misclasificar:
    - abs(disp_pred - disp_obj) < 0.10 : umbral arbitrario; falla cuando ambos
      estan del mismo lado del 0.5 con diferencia > 0.10 (degenerada real).
    - acc_proy >= 0.60 : una banda con acc_proy=0.62 y acc_modelo=0.90 es
      alteridad REAL (+28%); este criterio la marcaria ⊘. Misclasifica.
    - margen = 0.50 : es otra palanca (tamaño de efecto), no degeneracion.
  Tambien se descarta mi propia propuesta previa (`pred_modelo == pred_proy → ⊘`):
    enmascararia un FALLO de modelo en banda divergente si por azar coincide con la
    proyeccion. El criterio de degeneracion no puede esconder un ❌ legitimo.

  CRITERIO ADOPTADO (estructural, SIN parametro):
    Una banda es degenerada ⊘ si predictor y objetivo estan DEL MISMO LADO del 0.5
    (ambos aceptarian o ambos rechazarian la banda). Ahi, INCLUSO UN MODELO
    PERFECTO coincide con la proyeccion -> el medidor no tiene poder por la
    ESTRUCTURA de la tarea, no por como predijo el modelo. Es un juicio de nivel
    experimentador (como conocer el setpoint para definir el nulo en B-v9), no
    informacion que el organismo use para predecir.
    Propiedades:
      (a) coincide con la intencion pre-declarada ("0° compartida -> ⊘");
      (b) NO enmascara fallos de modelo en bandas divergentes (siguen medibles;
          si el modelo falla ahi, da ❌ real);
      (c) no tiene umbrales que calibrar.
    Edge case honesto: disposiciones casi en 0.5 (ambivalencia genuina) son
    inestables; con los datos actuales (0.82/0.08) estamos lejos del borde. Si una
    corrida futura cae cerca de 0.5, se evalua añadir una zona muerta (parametro).

CUERPO: V180 importado VERBATIM (como A.3/A.4/A.5). Innegociable: D es el gate de
  V183 y V183 corre sobre la dinada de organismos reales.

MARGEN GENUINO: ⚠ SIGUE PROVISIONAL (MARGEN_CONFIRMADO=False). El dato es robusto a
  cualquier margen < 60%, asi que el veredicto no depende de el; pero ratificarlo
  es decision del IP. No se pone en True aqui.

ABLACION: MODELO (Subj_sem: predice desde el modelo observacional del otro) vs
  PROYECCION (eco: "el otro hace lo que yo hago" = linea base nula).

MARCADORES: ✅ alteridad genuina · ❌ banda medible sin alteridad (incl. fallo de
  modelo) · ⊘ degenerada (mismo lado del 0.5: medidor sin poder) · · n/a
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
# ajena(~1-3)->rechaza, compartida(~6-8)->ambos aceptan o ambos rechazan segun
# escala. Separa divergencia de coincidencia; NO decide el ✅.
UMBRAL_ACEPTA = 5.0
ESCALA_ACEPTA = 2.0
LADO = 0.5        # frontera de decision (acepta si P>=LADO). Define divergencia.

# --- UMBRAL DE VEREDICTO (gating) ---
# ⚠⚠⚠ PENDIENTE: confirmar/fijar con el IP. El dato es robusto a margen < 60%. ⚠⚠⚠
MARGEN_GENUINO   = 0.15     # provisional (B-v9 usa 0.15 para real-nulo)
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

def lado_de(org, banda):
    """Lado de la disposicion respecto a la frontera 0.5 (True = aceptaria)."""
    return p_acepta(org, banda) >= LADO

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
def evaluar_prediccion(predictor, objetivo, modelo_del_objetivo, rng_obj):
    """
    Por banda: el objetivo actua (verdad de campo). El predictor predice con:
      - MODELO    : disposicion observada del objetivo (modelo_del_objetivo)
      - PROYECCION: lo que el predictor mismo haria en esa banda (eco)
    DEGENERACION (estructural, sin parametro): la banda es ⊘ si predictor y
    objetivo estan del MISMO lado del 0.5 -> incluso un modelo perfecto coincide
    con la proyeccion -> medidor sin poder. NO se basa en la prediccion del modelo
    (para no enmascarar fallos de modelo en bandas divergentes).
    """
    filas = {}
    for b in BANDAS:
        pred_modelo = 1 if modelo_del_objetivo[b] >= 0.5 else 0     # disposicion observada del otro
        pred_proy   = 1 if lado_de(predictor, b) else 0            # eco: "el otro hace lo que yo hago"

        aciertos_m, aciertos_p = 0, 0
        for _ in range(N_PRED):
            verdad = actuar(objetivo, b, rng_obj)                  # conducta real del objetivo
            aciertos_m += int(pred_modelo == verdad)
            aciertos_p += int(pred_proy   == verdad)
        acc_m = aciertos_m / N_PRED
        acc_p = aciertos_p / N_PRED
        genuino = acc_m - acc_p

        # --- DEGENERACION ESTRUCTURAL: mismo lado del 0.5 ---
        lado_pred = lado_de(predictor, b)
        lado_obj  = lado_de(objetivo, b)
        degenerada = (lado_pred == lado_obj)   # no divergen -> proyeccion estructuralmente acierta

        filas[b] = {'acc_modelo': acc_m, 'acc_proyeccion': acc_p, 'genuino': genuino,
                    'degenerada': bool(degenerada),
                    'disp_predictor': float(p_acepta(predictor, b)),
                    'disp_objetivo': float(p_acepta(objetivo, b)),
                    'lado_predictor': bool(lado_pred), 'lado_objetivo': bool(lado_obj)}
    return filas


# ============================================================
# F3 — ADVERTENCIA MUTUA SIN OVERRIDE (se mide, NO gatea, no decide por el otro)
# Nota: post-exposicion "el otro rechaza mi banda experta" es confiablemente
# cierto, asi que dispara casi siempre y SIN costo. NO es empatia (V182F); la
# advertencia con costo asimetrico es territorio de F.
# ============================================================
def medir_advertencia(predictor, objetivo, modelo_del_objetivo, banda_nativa_predictor):
    b = banda_nativa_predictor
    objetivo_rechaza_b = modelo_del_objetivo[b] < 0.5
    ajuste = (p_acepta(predictor, b) if objetivo_rechaza_b else 0.0)   # ajusta SU propia conducta; no toca al otro
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

    # observacion bidireccional (cada uno modela las elecciones externas del otro)
    modelo_A_de_B = observar_al_otro(B, np.random.default_rng(101))
    modelo_B_de_A = observar_al_otro(A, np.random.default_rng(202))

    # prediccion bidireccional
    AB = evaluar_prediccion(A, B, modelo_A_de_B, np.random.default_rng(303))
    BA = evaluar_prediccion(B, A, modelo_B_de_A, np.random.default_rng(305))

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
        nota = "  (mismo lado del 0.5 -> degenerada)" if f['degenerada'] else ""
        print(f"    {b:>+6.0f} | {f['acc_modelo']:>10.0%} {f['acc_proyeccion']:>9.0%} | {f['genuino']:>+8.0%} | [{mk}]{nota}")


def main():
    print("=" * 96)
    print("V182D_B — RECONOCIMIENTO DE ALTERIDAD (Subj_sem) — cuerpo V180 real")
    print("=" * 96)
    print("  Ablacion: MODELO (modelo observacional del otro) vs PROYECCION (eco = linea base nula).")
    print("  genuino(banda) = acc_modelo - acc_proyeccion.")
    print("  DEGENERACION (corregida, sin parametro): banda ⊘ si predictor y objetivo estan")
    print("  del MISMO lado del 0.5 (no divergen -> proyeccion estructuralmente acierta).")
    print(f"  CRITERIO (gating): genuino > MARGEN = {MARGEN_GENUINO:.0%}"
          + ("" if MARGEN_CONFIRMADO else "   ⚠ PROVISIONAL — sin confirmar por el IP"))
    print("  LEYENDA: ✅ alteridad genuina · ❌ medible sin alteridad · ⊘ degenerada")
    print("=" * 96)

    t0 = time.time()
    R = correr()

    print(f"\n  [disposicion emergente tras exposicion — P(acepta) por banda]")
    print(f"     {'banda':>6} |   A      B    | divergen?")
    for b in BANDAS:
        pa, pb = R['disp'][b]
        div = "SI (medible)" if (pa >= LADO) != (pb >= LADO) else "no (⊘)"
        print(f"     {b:>+6.0f} | {pa:>4.2f}  {pb:>4.2f}   | {div}")

    print(f"\n  [modelo observacional — P_obs(el otro acepta la banda)]")
    print(f"     {'banda':>6} | A-de-B  B-de-A")
    for b in BANDAS:
        print(f"     {b:>+6.0f} |  {R['modelo_A_de_B'][b]:>4.2f}   {R['modelo_B_de_A'][b]:>4.2f}")

    _imprimir_direccion("DIRECCION  A modela a B:", R['AB'])
    _imprimir_direccion("DIRECCION  B modela a A:", R['BA'])

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
    print(f"\n  [F3 advertencia (NO gatea, no override, sin costo -> NO es empatia/V182F): "
          f"A->B ajuste {R['adv_AB']['ajuste_propio']:.2f} | B->A ajuste {R['adv_BA']['ajuste_propio']:.2f}]")

    estado = "✅ ALTERIDAD BIDIRECCIONAL" if alteridad_bidireccional else "❌ no bidireccional"
    print(f"\n  {estado}")
    if not MARGEN_CONFIRMADO:
        print(f"  ⚠ VEREDICTO PROVISIONAL: el MARGEN ({MARGEN_GENUINO:.0%}) no ha sido confirmado por el IP.")
        print("    (El dato es robusto a cualquier margen < 60%; fijar MARGEN_GENUINO y MARGEN_CONFIRMADO=True")
        print("     antes de declarar D validado en el roadmap.)")
    print(f"\n  tiempo {time.time()-t0:.1f}s")

    os.makedirs("V182_logs", exist_ok=True)
    salida = {'version': 'V182D-alteridad-B', 'margen': MARGEN_GENUINO,
              'margen_confirmado': bool(MARGEN_CONFIRMADO),
              'criterio_degeneracion': 'mismo lado del 0.5 (estructural, sin parametro)',
              'disposicion': {str(k): v for k, v in R['disp'].items()},
              'modelo_A_de_B': {str(k): v for k, v in R['modelo_A_de_B'].items()},
              'modelo_B_de_A': {str(k): v for k, v in R['modelo_B_de_A'].items()},
              'AB': {str(k): v for k, v in R['AB'].items()},
              'BA': {str(k): v for k, v in R['BA'].items()},
              'adv_AB': R['adv_AB'], 'adv_BA': R['adv_BA'],
              'ok_AB': bool(ok_AB), 'ok_BA': bool(ok_BA),
              'alteridad_bidireccional': bool(alteridad_bidireccional)}
    with open(f"V182_logs/v182d_alteridad_B_{TS}.json", "w") as f:
        json.dump(salida, f, indent=2)
    print(f"  log: V182_logs/v182d_alteridad_B_{TS}.json")


if __name__ == "__main__":
    main()