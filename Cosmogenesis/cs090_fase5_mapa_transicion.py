"""
CS090 — FASE V-C: MAPA DE TRANSICIÓN kcap x K para A2-B0-C2 — ¿la bimodalidad I<->III es una
transición de fase genuina? (frente F5-C2-B, pedido del equipo-analisis-fase5-10ago2026)
=================================================================================================
QUIÉN SOY: archivo NUEVO (no toca ningún script congelado) que ataca la pregunta que dejó abierta
`FASE5A_profundizar_A2B0C2_resultado_CS.md` (Objetivo 2): dentro de la combinación A2-B0-C2 (grafo
dinámico co-emergente, sin retroalimentación relación-sobre-relación, con límite de escala duro), dos
parámetros correlacionaban moderadamente con caer en Clase III en vez de Clase I —
kcap (límite de escala, r=-0.43, n=18) y K (alfabeto de fase, r=+0.45, n=18) — pero con solape total de
rangos y sin umbral limpio. Este script barre kcap x K en grilla, GUARDA los observables continuos
ANTES de clasificar, arma la superficie P(Clase III | kcap, K), busca señales de borde nítido vs
gradiente y de comportamiento tipo transición de fase (varianza entre semillas cerca del borde), y hace
un test de histéresis.

Piezas reusadas TAL CUAL (sólo import, nunca se editan):
  cs090_fase5_generador.generar_regla / aplicar_filtro_P1_P5   -- generación de parámetros + filtro P1-P5
  cs090_fase5_motor.correr_regla_coarse                        -- motor de coarse-graining (N=2000)
  cs090_fase5_motor.CONSTRUCTORES_A / DINAMICAS_B               -- piezas atómicas del motor, para el
                                                                   test de histéresis (§4) y clustering
  cs090_fase5_clasificador.clasificar_regla                    -- clasificación I-IV

────────────────────────────────────────────────────────────────────────────────────────────────
DECISIÓN DE GRILLA (Paso 1, medida con el reloj real antes de comprometer nada — ver `medir_costo()`):
────────────────────────────────────────────────────────────────────────────────────────────────
Los rangos CALIBRADOS del generador congelado (`cs090_fase5_generador.py`, líneas 40/45) son
RANGO_K=(4,8) y RANGO_KCAP=(4,7), ambos ENTEROS. Eso significa que sólo existen 5 valores posibles de K
(4,5,6,7,8) y 4 valores posibles de kcap (4,5,6,7) dentro del espacio que la auditoría de C2
(`FASE5_auditoria_C2_resultado_CS.md`) puso a prueba. Un pedido de "8-10 valores por eje" EXCEDE ese
espacio de enteros calibrado.

Se decidió: quedarse DENTRO del rango auditado (grilla completa 5x4=20 celdas, el máximo que da el
espacio de enteros ya validado) en vez de extender kcap/K más allá de RANGO_KCAP/RANGO_K hacia un
régimen que la auditoría nunca puso a prueba. Es la opción más honesta: una grilla completa en
territorio conocido, no una grilla más ancha en territorio sin auditar. Esto se documenta explícito, no
se esconde — el tamaño final (20 celdas) es menor a los "8-10 x 8-10" ideales del pedido, y la razón es
estructural (el espacio de enteros del generador), no de presupuesto de cómputo (el motor resultó muy
rápido, ver Paso 1).

Cómo se controla kcap/K sin tocar el generador: se llama a `generar_regla()` normal (que sortea J,
noise, meandeg, sim_thr_frac, seed tal cual el método original) y LUEGO se sobreescriben `p["K"]` y
`p["kcap"]` con el valor de grilla deseado, ANTES de aplicar `aplicar_filtro_P1_P5()` (importada tal
cual). Si la regla resultante no admite, se reintenta con otra semilla (mismo kcap/K fijos) — misma
disciplina que `generar_reglas_clase()`, adaptada sólo para fijar kcap/K en vez de dejarlos aleatorios.

No se corre Phantom. No se declara cierre ni veredicto sobre si hay "una transición de fase real" — se
reportan los observables y la superficie, la lectura final es de Alexis. No se toca ningún script
congelado. No se hacen commits de git.
"""
from __future__ import annotations
import csv, sys, time
import numpy as np
from collections import defaultdict

sys.path.insert(0, "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis")
import cs090_fase5_generador as GEN
import cs090_fase5_motor as MOT
from cs090_fase5_clasificador import clasificar_regla

EJE_A, EJE_B, EJE_C = "A2", "B0", "C2"
N_GRANDE = 2000
N_SWEEPS = 14
ESCALAS_B = (1, 2, 4, 8, 16)
N_SEEDS_NULL_TOPO = 3

KCAP_VALORES = [4, 5, 6, 7]          # todo RANGO_KCAP entero, ver decisión de grilla arriba
K_VALORES = [4, 5, 6, 7, 8]          # todo RANGO_K entero

OUT_COSTO = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs090_fase5_mapa_transicion_costo.csv"
OUT_GRID = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs090_fase5_mapa_transicion_grid.csv"
OUT_HISTERESIS = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs090_fase5_mapa_transicion_histeresis.csv"

T_INICIO_GLOBAL = time.time()
PRESUPUESTO_TOTAL_SEG = 62 * 60   # salvaguarda de tiempo del pedido (55-65 min), con margen para reporte


def _tiempo_transcurrido():
    return time.time() - T_INICIO_GLOBAL


# ============================================================================================
# GENERACIÓN con kcap/K FIJOS (reusa generar_regla + aplicar_filtro_P1_P5 tal cual, sólo
# sobreescribe los dos parámetros bajo estudio antes del filtro)
# ============================================================================================
def generar_regla_kcap_K_fijos(kcap_fijo, K_fijo, idx, seed_base, max_reintentos=25):
    for intento in range(max_reintentos):
        seed = seed_base + intento * 97 + 1
        p = GEN.generar_regla(EJE_A, EJE_B, EJE_C, idx, seed)
        p["K"] = K_fijo
        p["kcap"] = kcap_fijo
        p["descripcion"] = (
            f"S en Z_{p['K']} por nodo/relación en sustrato {EJE_A}; actualización = media circular con "
            f"vecinos definidos por adyacencia previa (J={p['J']}, ruido={p['noise']}) [{EJE_B}]; "
            f"costo/localidad = {EJE_C} (grado máx duro={p['kcap']} + poda por costo)."
        )
        p = GEN.aplicar_filtro_P1_P5(p, seed_chequeo=seed + 500_000)
        if p["admitida"]:
            return p, intento + 1
    return None, max_reintentos


# ============================================================================================
# CLUSTERING (observable continuo barato de agregar): con C2 imponiendo grado máximo kcap<=7, el
# coeficiente de clustering EXACTO (no muestreado) cuesta O(N*kcap^2), barato incluso a N=2000.
# Se reconstruye el sustrato nativo con LA MISMA secuencia de rng que usa correr_regla_coarse()
# internamente (mismo seed*5000+N, mismas llamadas en el mismo orden) -- así el adj final que se
# mide para clustering es EXACTAMENTE el mismo grafo que correr_regla_coarse ya usó para diám/giant
# a escala b=1, no uno nuevo.
# ============================================================================================
def _clustering_promedio(adj, N):
    total, cont = 0.0, 0
    for i in range(N):
        vec = list(adj[i])
        k = len(vec)
        if k < 2:
            continue
        posibles = k * (k - 1) / 2.0
        conectados = 0
        for a in range(len(vec)):
            va = adj[vec[a]]
            for b in range(a + 1, len(vec)):
                if vec[b] in va:
                    conectados += 1
        total += conectados / posibles
        cont += 1
    return total / cont if cont else float("nan")


def correr_una_regla(p, N=N_GRANDE, n_sweeps=N_SWEEPS, escalas_b=ESCALAS_B,
                      n_seeds_null_topo=N_SEEDS_NULL_TOPO, con_clustering=True):
    filas = MOT.correr_regla_coarse(p, N=N, n_sweeps=n_sweeps, escalas_b=escalas_b,
                                     n_seeds_null_topo=n_seeds_null_topo)
    clus = float("nan")
    if con_clustering:
        rng = np.random.default_rng(p["seed"] * 5000 + N)
        sustrato = MOT.CONSTRUCTORES_A[p["eje_A"]](N, rng, p)
        sustrato = MOT.DINAMICAS_B[p["eje_B"]](sustrato, p, rng, n_sweeps, p["eje_C"])
        clus = _clustering_promedio(sustrato["adj"], N)
    return filas, clus


def resumen_de_regla(p, filas, clus, intentos_filtro):
    r = clasificar_regla(filas)
    fila_b1 = next(f for f in filas if f["escala_b"] == 1)
    fila_b16 = next(f for f in filas if f["escala_b"] == 16)
    return dict(
        rule_id=p["rule_id"], kcap=p["kcap"], K=p["K"], J=p["J"], noise=p["noise"],
        meandeg=p["meandeg"], seed=p["seed"], intentos_filtro=intentos_filtro,
        clase=r["clase"], pendiente_real=r["pendiente_real"], pendiente_null=r["pendiente_null"],
        z_agg=r["z_agg"], z_sostenido=r["z_sostenido"], holon_ratio=r["holon_ratio"], holon_ge5=r["holon_ge5"],
        diam_nativo=fila_b1["diam_real"], giant_nativo=fila_b1["giant_real"], n_aristas_nativo=fila_b1["n_aristas"],
        diam_b16=fila_b16["diam_real"], giant_b16=fila_b16["giant_real"],
        clustering_nativo=clus,
    )


# ============================================================================================
# PASO 1 — medir el costo real antes de comprometer una grilla grande
# ============================================================================================
def medir_costo():
    print("=" * 100)
    print("PASO 1 — medición de costo real (4 celdas esquina x 5 semillas)")
    print("=" * 100)
    celdas_prueba = [(min(KCAP_VALORES), min(K_VALORES)), (min(KCAP_VALORES), max(K_VALORES)),
                      (max(KCAP_VALORES), min(K_VALORES)), (max(KCAP_VALORES), max(K_VALORES))]
    filas_costo = []
    t0_paso1 = time.time()
    for (kcap, K) in celdas_prueba:
        for s in range(5):
            t0 = time.time()
            p, intentos = generar_regla_kcap_K_fijos(kcap, K, idx=s, seed_base=1_000_000 + kcap * 1009 + K * 97 + s * 7)
            if p is None:
                filas_costo.append(dict(kcap=kcap, K=K, seed_idx=s, dt=None, admitida=False))
                continue
            filas, clus = correr_una_regla(p)
            dt = time.time() - t0
            filas_costo.append(dict(kcap=kcap, K=K, seed_idx=s, dt=round(dt, 3), admitida=True, intentos_filtro=intentos))
            print(f"  kcap={kcap} K={K} seed_idx={s}: {dt:.2f}s (intentos_filtro={intentos})")
    dt_total = time.time() - t0_paso1
    dts = [f["dt"] for f in filas_costo if f["dt"] is not None]
    costo_medio = float(np.mean(dts))
    costo_max = float(np.max(dts))
    print(f"\nTotal Paso 1: {dt_total:.1f}s para {len(dts)} reglas -> costo medio/regla = {costo_medio:.2f}s "
          f"(máx observado = {costo_max:.2f}s)")

    with open(OUT_COSTO, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(filas_costo[0].keys()))
        wr.writeheader()
        for f in filas_costo:
            wr.writerow(f)
    print(f"CSV Paso 1: {OUT_COSTO}")
    return costo_medio, costo_max


# ============================================================================================
# PASO 2 — barrido preregistrado kcap x K, con el presupuesto medido
# ============================================================================================
def barrido_grid(costo_medio, n_seeds):
    n_celdas = len(KCAP_VALORES) * len(K_VALORES)
    costo_estimado = n_celdas * n_seeds * costo_medio
    print("\n" + "=" * 100)
    print(f"PASO 2 — barrido preregistrado: {len(KCAP_VALORES)} valores de kcap x {len(K_VALORES)} valores "
          f"de K = {n_celdas} celdas, {n_seeds} semillas/celda ({n_celdas*n_seeds} reglas totales). "
          f"Costo estimado = {costo_estimado/60:.1f} min (a {costo_medio:.2f}s/regla medido en Paso 1).")
    print("=" * 100)

    resultados = []
    t0 = time.time()
    idx_global = 0
    for kcap in KCAP_VALORES:
        for K in K_VALORES:
            for s in range(n_seeds):
                p, intentos = generar_regla_kcap_K_fijos(kcap, K, idx=idx_global,
                                                           seed_base=2_000_000 + kcap * 5003 + K * 311 + s * 13)
                idx_global += 1
                if p is None:
                    resultados.append(dict(kcap=kcap, K=K, seed_idx=s, admitida=False, rule_id=None))
                    continue
                filas, clus = correr_una_regla(p)
                resumen = resumen_de_regla(p, filas, clus, intentos)
                resumen["admitida"] = True
                resumen["seed_idx"] = s
                resultados.append(resumen)
            transcurrido = _tiempo_transcurrido()
            print(f"  celda (kcap={kcap}, K={K}) lista. t_total_script={transcurrido/60:.1f} min")
            if transcurrido > PRESUPUESTO_TOTAL_SEG:
                print("  *** SALVAGUARDA DE TIEMPO: presupuesto alcanzado, se corta el barrido acá ***")
                dt_barrido = time.time() - t0
                _guardar_grid(resultados)
                return resultados, dt_barrido
    dt_barrido = time.time() - t0
    _guardar_grid(resultados)
    print(f"\nBarrido completo en {dt_barrido/60:.1f} min ({len(resultados)} filas)")
    return resultados, dt_barrido


def _guardar_grid(resultados):
    campos = set()
    for r in resultados:
        campos |= set(r.keys())
    campos = ["kcap", "K", "seed_idx", "admitida"] + sorted(campos - {"kcap", "K", "seed_idx", "admitida"})
    with open(OUT_GRID, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=campos)
        wr.writeheader()
        for r in resultados:
            wr.writerow(r)
    print(f"CSV Paso 2 (grilla, {len(resultados)} filas): {OUT_GRID}")


# ============================================================================================
# PASO 3 — superficie P(Clase III | kcap, K) + análisis de borde
# ============================================================================================
def analizar_superficie(resultados):
    print("\n" + "=" * 100)
    print("PASO 3 — superficie P(Clase III | kcap, K) y análisis de borde")
    print("=" * 100)
    admitidas = [r for r in resultados if r.get("admitida")]
    superficie = {}
    for kcap in KCAP_VALORES:
        for K in K_VALORES:
            celda = [r for r in admitidas if r["kcap"] == kcap and r["K"] == K]
            n = len(celda)
            n_III = sum(1 for r in celda if r["clase"] == "III")
            n_I = sum(1 for r in celda if r["clase"] == "I")
            p_III = n_III / n if n else float("nan")
            pendientes = [r["pendiente_real"] for r in celda if np.isfinite(r["pendiente_real"])]
            std_pend = float(np.std(pendientes)) if len(pendientes) > 1 else float("nan")
            superficie[(kcap, K)] = dict(n=n, n_I=n_I, n_III=n_III, p_III=p_III,
                                          pendiente_media=float(np.mean(pendientes)) if pendientes else float("nan"),
                                          pendiente_std=std_pend)

    print(f"\n  Superficie P(Clase III | kcap, K)  [filas=kcap, columnas=K]")
    header = "kcap\\K  " + "  ".join(f"K={K}" for K in K_VALORES)
    print("  " + header)
    for kcap in KCAP_VALORES:
        fila = f"  kcap={kcap} "
        for K in K_VALORES:
            fila += f"  {superficie[(kcap,K)]['p_III']:.2f}"
        print(fila)

    print(f"\n  Desvío estándar de la pendiente entre semillas, por celda [filas=kcap, columnas=K]")
    print("  " + header)
    for kcap in KCAP_VALORES:
        fila = f"  kcap={kcap} "
        for K in K_VALORES:
            sp = superficie[(kcap, K)]["pendiente_std"]
            fila += f"  {sp:.3f}" if np.isfinite(sp) else "   nan "
        print(fila)

    # gradiente: diferencia de p_III entre celdas vecinas (kcap adyacente, K fijo) y (K adyacente, kcap fijo)
    saltos = []
    for kcap in KCAP_VALORES:
        for i in range(len(K_VALORES) - 1):
            K1, K2 = K_VALORES[i], K_VALORES[i + 1]
            p1, p2 = superficie[(kcap, K1)]["p_III"], superficie[(kcap, K2)]["p_III"]
            if np.isfinite(p1) and np.isfinite(p2):
                saltos.append(dict(eje="K", kcap=kcap, desde=K1, hasta=K2, salto=abs(p2 - p1)))
    for K in K_VALORES:
        for i in range(len(KCAP_VALORES) - 1):
            k1, k2 = KCAP_VALORES[i], KCAP_VALORES[i + 1]
            p1, p2 = superficie[(k1, K)]["p_III"], superficie[(k2, K)]["p_III"]
            if np.isfinite(p1) and np.isfinite(p2):
                saltos.append(dict(eje="kcap", K=K, desde=k1, hasta=k2, salto=abs(p2 - p1)))
    saltos.sort(key=lambda d: -d["salto"])
    salto_max = saltos[0]["salto"] if saltos else float("nan")
    salto_medio = float(np.mean([s["salto"] for s in saltos])) if saltos else float("nan")
    print(f"\n  Salto máximo entre celdas vecinas en P(III): {salto_max:.2f}  (media de todos los saltos vecinos: {salto_medio:.2f})")
    print(f"  Top 3 saltos más grandes: {saltos[:3]}")

    # varianza cerca del borde vs lejos: borde = celdas con 0.2<=p_III<=0.8; lejos = resto
    celdas_borde = [(k, K) for (k, K) in superficie if 0.2 <= superficie[(k, K)]["p_III"] <= 0.8]
    celdas_lejos = [(k, K) for (k, K) in superficie if (k, K) not in celdas_borde and np.isfinite(superficie[(k, K)]["p_III"])]
    std_borde = [superficie[c]["pendiente_std"] for c in celdas_borde if np.isfinite(superficie[c]["pendiente_std"])]
    std_lejos = [superficie[c]["pendiente_std"] for c in celdas_lejos if np.isfinite(superficie[c]["pendiente_std"])]
    print(f"\n  Celdas 'de borde' (0.2<=P(III)<=0.8): {celdas_borde}")
    print(f"  Celdas 'lejos del borde': {celdas_lejos}")
    if std_borde:
        print(f"  Desvío de pendiente ENTRE semillas -- borde: media={np.mean(std_borde):.3f} (n_celdas={len(std_borde)})")
    if std_lejos:
        print(f"  Desvío de pendiente ENTRE semillas -- lejos: media={np.mean(std_lejos):.3f} (n_celdas={len(std_lejos)})")

    return superficie, saltos, celdas_borde, celdas_lejos


# ============================================================================================
# PASO 4 — test de histéresis (kcap sube vs kcap baja, a K fijo)
# ============================================================================================
def test_histeresis(K_fijo, n_seeds_por_direccion=3):
    print("\n" + "=" * 100)
    print(f"PASO 4 — test de histéresis a K={K_fijo} fijo, kcap recorriendo {KCAP_VALORES} en dos direcciones")
    print("=" * 100)
    print("  Chequeo de continuidad de estado real en el motor congelado (cs090_fase5_motor.py):")
    print("  dinamica_B0(), rama A1/A2 (línea ~202): `S = rng.uniform(0, K, N)` -- RESAMPLEA el estado")
    print("  de cada nodo DESDE CERO al inicio de CADA llamada, sin leer nunca sustrato['S'] previo.")
    print("  -> El motor NO tiene un mecanismo de estado persistente entre puntos de parámetro para los")
    print("  VALORES (fases) de los nodos. Se documenta como limitación honesta, no se fabrica.")
    print("  SÍ es técnicamente posible encadenar la TOPOLOGÍA (sustrato['adj']) entre llamadas sucesivas")
    print("  de dinamica_B0/_enforce_kcap sin tocar el motor (son funciones expuestas, componibles desde")
    print("  este archivo nuevo) -- se corre esa variante como PARTE B, etiquetada 'continuidad SÓLO")
    print("  topológica' (no es continuidad de estado completa, y se reporta como tal).")

    resultados_indep = []
    for direccion, kcaps in [("alto_a_bajo", list(reversed(KCAP_VALORES))), ("bajo_a_alto", list(KCAP_VALORES))]:
        for kcap in kcaps:
            for s in range(n_seeds_por_direccion):
                p, intentos = generar_regla_kcap_K_fijos(kcap, K_fijo, idx=s,
                                                           seed_base=3_000_000 + kcap * 4001 + s * 11)
                if p is None:
                    continue
                filas, clus = correr_una_regla(p, con_clustering=False)
                r = clasificar_regla(filas)
                resultados_indep.append(dict(parte="A_independiente", direccion=direccion, kcap=kcap,
                                              seed_idx=s, clase=r["clase"], pendiente=r["pendiente_real"]))
    print(f"\n  PARTE A (independiente, SIN continuidad -- réplica honesta de cómo corre el motor tal cual):")
    for direccion in ("alto_a_bajo", "bajo_a_alto"):
        for kcap in KCAP_VALORES:
            clases = [r["clase"] for r in resultados_indep if r["direccion"] == direccion and r["kcap"] == kcap]
            print(f"    {direccion} kcap={kcap}: clases={clases}")

    # PARTE B -- continuidad SÓLO topológica: un único sustrato encadenado, cambiando p['kcap'] entre
    # llamadas sucesivas a dinamica_B0 (que sí sigue aplicando _enforce_kcap con el kcap del momento
    # sobre la MISMA adyacencia que traía de la llamada anterior). El estado de fase S se resamplea
    # igual (ver arriba) -- eso NO se puede evitar sin tocar el motor.
    resultados_topo = []
    for direccion, kcaps in [("alto_a_bajo", list(reversed(KCAP_VALORES))), ("bajo_a_alto", list(KCAP_VALORES))]:
        rng = np.random.default_rng(4_000_000 + hash(direccion) % 1000)
        p_base, intentos = generar_regla_kcap_K_fijos(kcaps[0], K_fijo, idx=0, seed_base=5_000_000 + hash(direccion) % 1000)
        if p_base is None:
            continue
        sustrato = MOT.CONSTRUCTORES_A[EJE_A](N_GRANDE, rng, p_base)
        for kcap in kcaps:
            p_paso = dict(p_base); p_paso["kcap"] = kcap
            sustrato = MOT.DINAMICAS_B[EJE_B](sustrato, p_paso, rng, N_SWEEPS, EJE_C)
            m = MOT.medir(sustrato, p_paso, rng)
            resultados_topo.append(dict(parte="B_topologia_encadenada", direccion=direccion, kcap=kcap,
                                         diam_nativo=m["diam"], giant_nativo=m["giant"], n_aristas=m["n_aristas"]))
    print(f"\n  PARTE B (continuidad SÓLO topológica, un sustrato encadenado por dirección):")
    for direccion in ("alto_a_bajo", "bajo_a_alto"):
        for r in [r for r in resultados_topo if r["direccion"] == direccion]:
            print(f"    {direccion} kcap={r['kcap']}: diam_nativo={r['diam_nativo']} giant_nativo={r['giant_nativo']:.3f} n_aristas={r['n_aristas']}")

    todas = resultados_indep + resultados_topo
    with open(OUT_HISTERESIS, "w", newline="") as fh:
        campos = sorted(set().union(*[set(r.keys()) for r in todas])) if todas else []
        wr = csv.DictWriter(fh, fieldnames=campos)
        wr.writeheader()
        for r in todas:
            wr.writerow(r)
    print(f"\nCSV Paso 4: {OUT_HISTERESIS}")
    return resultados_indep, resultados_topo


# ============================================================================================
# DRIVER
# ============================================================================================
def main():
    costo_medio, costo_max = medir_costo()
    tiempo_usado = _tiempo_transcurrido()
    tiempo_restante = PRESUPUESTO_TOTAL_SEG - tiempo_usado
    print(f"\nTiempo usado en Paso 1: {tiempo_usado/60:.1f} min. Tiempo restante estimado para Pasos 2-4: "
          f"{tiempo_restante/60:.1f} min.")

    n_celdas = len(KCAP_VALORES) * len(K_VALORES)
    # reservar ~8 min para Paso 4 (histéresis, barato) y margen; el resto para Paso 2
    presupuesto_paso2 = max(60, tiempo_restante - 8 * 60)
    n_seeds = int(presupuesto_paso2 // (n_celdas * costo_medio))
    n_seeds = max(3, min(20, n_seeds))
    print(f"Paso 2: con costo_medio={costo_medio:.2f}s/regla y {n_celdas} celdas, presupuesto "
          f"{presupuesto_paso2/60:.1f} min -> n_seeds/celda elegido = {n_seeds}")

    resultados_grid, dt_barrido = barrido_grid(costo_medio, n_seeds)
    superficie, saltos, celdas_borde, celdas_lejos = analizar_superficie(resultados_grid)

    # elegir K con mejor separación para el test de histéresis: el K cuya fila de P(III) tiene el
    # rango (max-min) más grande a lo largo de kcap
    mejor_K, mejor_rango = None, -1
    for K in K_VALORES:
        vals = [superficie[(kcap, K)]["p_III"] for kcap in KCAP_VALORES if np.isfinite(superficie[(kcap, K)]["p_III"])]
        if len(vals) >= 2:
            rango = max(vals) - min(vals)
            if rango > mejor_rango:
                mejor_rango, mejor_K = rango, K
    if mejor_K is None:
        mejor_K = K_VALORES[len(K_VALORES) // 2]
    print(f"\nK elegido para histéresis (mayor rango de P(III) a lo largo de kcap): K={mejor_K} (rango={mejor_rango:.2f})")

    resultados_indep, resultados_topo = test_histeresis(mejor_K, n_seeds_por_direccion=3)

    print(f"\n\nTiempo total del script: {_tiempo_transcurrido()/60:.1f} min")
    print("Fin. No se declara cierre ni veredicto -- números arriba, lectura final de Alexis.")


if __name__ == "__main__":
    main()
