"""
CS090 — FASE VI / O1-C: CIERRE DE LA PREGUNTA A0 CON MUESTRA SUFICIENTE
====================================================================================================
QUIÉN SOY
---------
Script NUEVO (no toca ningún congelado, no toca `cs090_fase5_a0_nativo.py` — lo IMPORTA y reusa sus
métricas). Su único trabajo es juntar una muestra GRANDE de reglas A0 y medir cada una con los DOS
métodos, para que la comparación entre "Clase II" y "Clase I" tenga n suficiente para un test formal.

EL PROBLEMA QUE VIENE DE ATRÁS (contexto en 3 líneas)
-----------------------------------------------------
- `FASE5A_completo_resultado_CS.md`: 27% de las reglas A0 (sustrato SIN grafo, campo continuo) cayeron
  en "Clase II — mundo-pequeño congelado". Sospecha: el grafo de medición derivado
  (`_grafo_medicion_A0`) conecta cada sitio con `n_cand=15` candidatos AL AZAR de todo el anillo — o
  sea, atajos de largo alcance sobre una base local: la receta literal de un grafo Watts-Strogatz, que
  es "mundo pequeño" POR CONSTRUCCIÓN, mida lo que mida el campo real.
- `FASE5_A0_metricas_nativas_CS.md`: se implementaron métricas nativas de campo (ξ(r) y dominios por
  adyacencia física, sin grafo derivado) y se compararon... pero en esa corrida sólo 2 de 35 reglas
  cayeron en Clase II. Con n=2 no hay test posible.
- ESTA TAREA (O1-C): repetir lo mismo con MUCHAS más Clase II (objetivo ≥12-15) más un grupo control
  de Clase I, y correr un test formal (Kolmogorov-Smirnov + Mann-Whitney + permutación) sobre las
  métricas nativas. El análisis estadístico va en `cs090_fase6_o1c_analisis.py`.

ANALOGÍA SIMPLE DE LO QUE SE ESTÁ MIDIENDO
-------------------------------------------
Imaginen 2000 personas en una ronda, cada una susurrando sólo a los dos vecinos de al lado (eso es el
sustrato A0: difusión local pura, sin teléfonos). Para decir "acá el mundo es pequeño", el método VIEJO
agarra parejas AL AZAR de cualquier parte de la ronda y las conecta por teléfono si dicen algo
parecido; sobre esa red de teléfonos mide "qué tan lejos está todo el mundo de todo el mundo". El
método NATIVO no arma ninguna red: sólo pregunta, caminando por la ronda, hasta dónde se sigue
pareciendo el susurro. La pregunta de esta tarea: cuando el método de los teléfonos dice "acá hay algo
raro" (Clase II), ¿la ronda se ve distinta caminándola de a pie? Si no se ve distinta en NINGUNA de
15 rondas marcadas, "Clase II" en A0 mide el método de medición, no la ronda.

QUÉ SE REUSA (sólo import — cero ediciones en congelados)
----------------------------------------------------------
  - `cs090_fase5_generador.generar_reglas_clase`  → generación + filtro P1-P5 REAL (no asumido)
  - `cs090_diam_corregido.correr_regla_coarse_doble` → método VIEJO corrido UNA vez pero con el
    diámetro medido de las DOS maneras: la histórica (`cs055._diam`, arranca el doble-BFS en el nodo
    no aislado de índice más bajo — puede caer en un fragmento suelto) y la corregida
    (`diam_gigante`, arranca dentro de la componente conexa más grande). Así se responde también la
    nota de la tarea: "¿el bug de diámetro afecta a este método viejo acá?".
    Su rama `filas_orig` reproduce exactamente lo que devolvería
    `cs090_fase5_motor.correr_regla_coarse` (mismas semillas derivadas, mismo coarse-graining de
    cs080), así que el método viejo NO cambia por usar este envoltorio.
  - `cs090_fase5_clasificador.clasificar_regla`   → Clase I-IV con los umbrales pre-registrados
  - `cs090_fase5_a0_nativo` (NAT)                 → `coarsear_campo_ring`, `metricas_nativas_A0`,
    `_pendiente_loglog`: las métricas nativas YA implementadas y ya validadas contra NULL. Se importan
    tal cual (la tarea pide reusarlas, no reinventarlas).
  - `cs090_fase5_motor.construir_A0` / `dinamica_B0` → reconstrucción bit a bit del MISMO campo que
    midió el método viejo (misma fórmula de semilla `seed*5000+N`), para comparar manzanas con manzanas.

QUÉ ES NUEVO ACÁ
-----------------
Nada de física ni de métrica: sólo (a) el barrido grande con contabilidad honesta de la tasa base de
Clase II, (b) el emparejamiento de las tres mediciones por regla (viejo-histórico / viejo-corregido /
nativo) en un solo CSV, y (c) la verificación explícita de qué combinaciones del Eje A0 son realmente
distintas entre sí (ver `verificar_combos_A0`).

POR QUÉ EL BARRIDO ES SÓLO A0-B0-C0 (verificado, no supuesto)
---------------------------------------------------------------
`verificar_combos_A0()` comprueba en la corrida, no de palabra:
  - A0-B1-*: `dinamica_B1` lee `sustrato["adj"]`, y A0 no tiene grafo → no existe como combinación.
  - A0-B0-C1 y A0-B0-C2: `dinamica_B0` con `kind=="A0"` devuelve ANTES de tocar cualquier bloque de
    costo/poda (motor.py, rama A0 termina en `return sustrato`), así que el eje C no entra en la
    dinámica; con la misma semilla el campo sale idéntico bit a bit al de C0. Se comprueba con
    `np.array_equal` (la regla de la casa: verificar determinismo antes de asumir azar).
Conclusión operativa: para A0 hay UNA sola combinación con dinámica propia; la muestra se junta
variando semillas dentro de A0-B0-C0, no combinando ejes (que sería duplicar filas).

SALIDAS
-------
  - cs090_fase6_o1c_a0_resumen.csv     — una fila por regla: parámetros, clase vieja (histórica y
                                          corregida), pendientes, métricas nativas REAL y NULL.
  - cs090_fase6_o1c_a0_viejo_raw.csv   — una fila por regla × escala b: diámetros orig/corr + diagnóstico
                                          de fragmentación (tamaño de componente medida vs. gigante).
  - cs090_fase6_o1c_a0_nativo_raw.csv  — una fila por regla × escala b: métricas nativas REAL y NULL.
Se escribe de forma incremental (cada `GUARDAR_CADA` reglas) para no perder el trabajo si la corrida
se corta.

No declara cierre ni veredicto: reporta números. La lectura final es de Alexis.
"""
from __future__ import annotations

import csv
import sys
import time

import numpy as np

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import cs090_fase5_generador as GEN       # generación + filtro P1-P5 -- SÓLO IMPORT
import cs090_fase5_motor as MOT           # construir_A0 / dinamica_B0 -- SÓLO IMPORT
import cs090_fase5_clasificador as CLS    # clasificar_regla          -- SÓLO IMPORT
import cs090_diam_corregido as DIAM       # correr_regla_coarse_doble -- SÓLO IMPORT
import cs090_fase5_a0_nativo as NAT       # métricas nativas ya implementadas -- SÓLO IMPORT

# ------------------------------------------------------------------ parámetros de la corrida
N_SITIOS = 2000                 # mismo N que usó el barrido de 180 reglas y el informe de nativas
N_SWEEPS = 14                   # idem
ESCALAS_B = (1, 2, 4, 8, 16)    # mismas escalas con que se calibraron los umbrales de clase (cs080)
SEED_BASE = 20260811            # lote nuevo (el informe de nativas usó 20260810) -- muestra fresca
N_REGLAS_OBJETIVO = 400         # techo del barrido; la tasa base decide cuántas Clase II salen
GUARDAR_CADA = 25


# ============================================================================================
# 0) VERIFICACIÓN DE QUÉ COMBINACIONES A0 EXISTEN DE VERDAD (se comprueba, no se supone)
# ============================================================================================
def verificar_combos_A0(seed=987654321):
    """Comprueba empíricamente (a) que A0-B1 no es una combinación ejecutable y (b) que el eje C no
    cambia NADA en A0-B0 (campo idéntico bit a bit con la misma semilla). Devuelve un dict con el
    resultado de cada comprobación, que se imprime en el log de la corrida."""
    res = {}
    campos = {}
    for eje_C in ("C0", "C1", "C2"):
        p = GEN.generar_regla("A0", "B0", eje_C, idx=0, seed=seed)
        rng = np.random.default_rng(p["seed"] * 5000 + N_SITIOS)
        s = MOT.construir_A0(N_SITIOS, rng, p)
        s = MOT.dinamica_B0(s, p, rng, N_SWEEPS, p["eje_C"])
        campos[eje_C] = s["S"].copy()
    res["C1_identico_a_C0"] = bool(np.array_equal(campos["C0"], campos["C1"]))
    res["C2_identico_a_C0"] = bool(np.array_equal(campos["C0"], campos["C2"]))

    p = GEN.generar_regla("A0", "B1", "C0", idx=0, seed=seed)
    rng = np.random.default_rng(1)
    try:
        s = MOT.construir_A0(200, rng, p)
        MOT.dinamica_B1(s, p, rng, 3, "C0")
        res["A0_B1_ejecutable"] = True
        res["A0_B1_error"] = ""
    except Exception as e:      # se documenta el error exacto, no se esconde
        res["A0_B1_ejecutable"] = False
        res["A0_B1_error"] = f"{type(e).__name__}: {e}"
    return res


# ============================================================================================
# 1) UNA REGLA: método VIEJO (histórico + corregido) y método NATIVO sobre EL MISMO campo
# ============================================================================================
def medir_regla(p):
    """Devuelve (fila_resumen, filas_viejo, filas_nativo) para UNA regla A0.

    Método viejo: `correr_regla_coarse_doble` corre la cadena completa (construir → dinámica → grafo
    de medición derivado → coarse-graining BFS a b=1..16 → diám vs n_cajas) midiendo el diámetro con
    las dos versiones. Cada juego de filas se clasifica por separado con el clasificador congelado.

    Método nativo: se reconstruye el MISMO campo con la misma fórmula de semilla (`seed*5000+N`) —
    array `S` idéntico bit a bit al que midió el método viejo — y se le aplican las métricas nativas
    de `cs090_fase5_a0_nativo` a las mismas escalas b, más el control NULL (campo barajado)."""
    filas_orig, filas_corr, diagnos = DIAM.correr_regla_coarse_doble(
        p, N=N_SITIOS, n_sweeps=N_SWEEPS, escalas_b=ESCALAS_B)
    r_orig = CLS.clasificar_regla(filas_orig)
    r_corr = CLS.clasificar_regla(filas_corr)

    # ---- mismo campo, reconstruido aparte (rng nuevo: no comparte estado, así que S es idéntico) ----
    rng = np.random.default_rng(p["seed"] * 5000 + N_SITIOS)
    sustrato = MOT.construir_A0(N_SITIOS, rng, p)
    sustrato = MOT.dinamica_B0(sustrato, p, rng, N_SWEEPS, p["eje_C"])
    S = sustrato["S"].copy()
    thr = p["sim_thr_frac"] * p["K"]        # MISMO umbral que usa _grafo_medicion_A0 (comparación justa)

    # NULL nativo: mismas posiciones barajadas (misma receta que el informe de métricas nativas)
    rng_shuf = np.random.default_rng(p["seed"] * 4000 + N_SITIOS)
    S_shuf = rng_shuf.permutation(S)

    filas_nativo = []
    for b in ESCALAS_B:
        S_b, n_cajas = NAT.coarsear_campo_ring(S, p["K"], b)
        S_b_null, _ = NAT.coarsear_campo_ring(S_shuf, p["K"], b)
        nat_real = NAT.metricas_nativas_A0(S_b, p["K"], thr)
        nat_null = NAT.metricas_nativas_A0(S_b_null, p["K"], thr)
        filas_nativo.append(dict(
            rule_id=p["rule_id"], seed=p["seed"], escala_b=b, n_cajas=n_cajas,
            corr_len_real=nat_real["corr_len"], corr_len_sat_real=nat_real["corr_len_saturada"],
            giant_nativo_frac_real=nat_real["giant_nativo_frac"],
            giant_nativo_size_real=nat_real["giant_nativo_size"], n_dominios_real=nat_real["n_dominios"],
            corr_len_null=nat_null["corr_len"], corr_len_sat_null=nat_null["corr_len_saturada"],
            giant_nativo_frac_null=nat_null["giant_nativo_frac"],
            giant_nativo_size_null=nat_null["giant_nativo_size"], n_dominios_null=nat_null["n_dominios"],
        ))

    cajas = [f["n_cajas"] for f in filas_nativo]
    corr_slope = NAT._pendiente_loglog(cajas, [f["corr_len_real"] for f in filas_nativo])
    dom_slope = NAT._pendiente_loglog(cajas, [f["giant_nativo_size_real"] for f in filas_nativo])
    corr_slope_null = NAT._pendiente_loglog(cajas, [f["corr_len_null"] for f in filas_nativo])
    dom_slope_null = NAT._pendiente_loglog(cajas, [f["giant_nativo_size_null"] for f in filas_nativo])
    f_b1 = filas_nativo[0]      # b=1: resolución nativa del anillo (n_cajas = N)

    # diagnóstico del bug de diámetro: ¿alguna escala midió un fragmento en vez de la gigante?
    algun_descarrile = any(d.get("descarrila", False) for d in diagnos)
    min_frac_comp = min((d["tam_comp_medida"] / max(1, d["tam_gigante"])) for d in diagnos
                        if d.get("tam_gigante"))

    fila = dict(
        rule_id=p["rule_id"], seed=p["seed"],
        K=p["K"], J=p["J"], noise=p["noise"], sim_thr_frac=p["sim_thr_frac"],
        # --------- método viejo, versión HISTÓRICA (cs055._diam) ---------
        clase_vieja=r_orig["clase"], pendiente_vieja=round(r_orig["pendiente_real"], 4),
        z_agg_vieja=round(r_orig["z_agg"], 3), holon_ratio_vieja=round(r_orig["holon_ratio"], 3),
        # --------- método viejo, versión CORREGIDA (diam_gigante) ---------
        clase_vieja_corr=r_corr["clase"], pendiente_vieja_corr=round(r_corr["pendiente_real"], 4),
        z_agg_vieja_corr=round(r_corr["z_agg"], 3),
        diam_b1_orig=filas_orig[0]["diam_real"], diam_b1_corr=filas_corr[0]["diam_real"],
        algun_descarrile=algun_descarrile, min_frac_comp_medida=round(min_frac_comp, 4),
        n_aristas_b1=filas_orig[0]["n_aristas"],
        # --------- método NATIVO (campo continuo, sin grafo derivado) ---------
        corr_len_b1=f_b1["corr_len_real"], corr_len_sat_b1=f_b1["corr_len_sat_real"],
        giant_frac_b1=round(f_b1["giant_nativo_frac_real"], 4), n_dominios_b1=f_b1["n_dominios_real"],
        corr_slope_nativo=round(corr_slope, 4), dom_slope_nativo=round(dom_slope, 4),
        # --------- control NULL nativo (campo barajado) ---------
        corr_slope_null=round(corr_slope_null, 4), dom_slope_null=round(dom_slope_null, 4),
        giant_frac_null_b1=round(f_b1["giant_nativo_frac_null"], 4),
        n_dominios_null_b1=f_b1["n_dominios_null"],
    )

    filas_viejo = []
    for fo, fc, dg in zip(filas_orig, filas_corr, diagnos):
        filas_viejo.append(dict(
            rule_id=p["rule_id"], seed=p["seed"], escala_b=fo["escala_b"], n_cajas=fo["N"],
            diam_real_orig=fo["diam_real"], diam_real_corr=fc["diam_real"],
            diam_null_orig=fo["diam_null_topo"], diam_null_corr=fc["diam_null_topo"],
            giant_real=fo["giant_real"], n_aristas=fo["n_aristas"],
            holon_real=fo["holon_real"], holon_null_valor=fo["holon_null_valor"],
            tam_comp_medida=dg.get("tam_comp_medida"), tam_gigante=dg.get("tam_gigante"),
            n_componentes=dg.get("n_componentes"), n_aislados=dg.get("n_aislados"),
            descarrila=dg.get("descarrila"),
        ))
    return fila, filas_viejo, filas_nativo


# ============================================================================================
# 2) BARRIDO
# ============================================================================================
def _guardar(nombre, filas):
    if not filas:
        return
    with open(f"{_HERE}/{nombre}", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)


def main(n_reglas=N_REGLAS_OBJETIVO, seed_base=SEED_BASE):
    t_ini = time.time()
    print("=" * 100)
    print("O1-C — cierre A0: método viejo (histórico + corregido) vs. métricas nativas, muestra grande")
    print("=" * 100)

    chk = verificar_combos_A0()
    print(f"[combos A0] C1 idéntico a C0: {chk['C1_identico_a_C0']} | C2 idéntico a C0: "
          f"{chk['C2_identico_a_C0']} | A0-B1 ejecutable: {chk['A0_B1_ejecutable']} "
          f"({chk['A0_B1_error']})")
    print("[combos A0] -> el barrido se hace variando semillas dentro de A0-B0-C0 (única combinación "
          "con dinámica propia en el Eje A0).\n")

    # el generador congelado ya sabe reintentar y documentar descartes: se le sube el techo de intentos
    t0 = time.time()
    admitidas, descartadas = GEN.generar_reglas_clase(
        "A0", "B0", "C0", n_reglas=n_reglas, seed_base=seed_base, max_intentos=n_reglas * 3)
    print(f"[gen] admitidas={len(admitidas)} descartadas={len(descartadas)} "
          f"(filtro P1-P5 real, {time.time()-t0:.1f}s)")
    if descartadas:
        motivos = {}
        for d in descartadas:
            for P, ok in d["filtro"].items():
                if not ok:
                    motivos[P] = motivos.get(P, 0) + 1
        print(f"[gen] descartes por criterio: {motivos}")

    resumen, raw_viejo, raw_nativo = [], [], []
    n_II = n_II_corr = 0
    for k, p in enumerate(admitidas):
        t0 = time.time()
        fila, fv, fn = medir_regla(p)
        resumen.append(fila); raw_viejo += fv; raw_nativo += fn
        if fila["clase_vieja"] == "II":
            n_II += 1
        if fila["clase_vieja_corr"] == "II":
            n_II_corr += 1
        marca = "  <-- CLASE II" if fila["clase_vieja"] == "II" else ""
        print(f"  [{k+1}/{len(admitidas)}] {p['rule_id']:<16} pend={fila['pendiente_vieja']:.3f} "
              f"clase={fila['clase_vieja']:<24} | nativo corr={fila['corr_slope_nativo']:.3f} "
              f"dom={fila['dom_slope_nativo']:.3f} giant_b1={fila['giant_frac_b1']:.3f} "
              f"| II={n_II} ({time.time()-t0:.1f}s){marca}", flush=True)
        if (k + 1) % GUARDAR_CADA == 0:
            _guardar("cs090_fase6_o1c_a0_resumen.csv", resumen)
            _guardar("cs090_fase6_o1c_a0_viejo_raw.csv", raw_viejo)
            _guardar("cs090_fase6_o1c_a0_nativo_raw.csv", raw_nativo)

    _guardar("cs090_fase6_o1c_a0_resumen.csv", resumen)
    _guardar("cs090_fase6_o1c_a0_viejo_raw.csv", raw_viejo)
    _guardar("cs090_fase6_o1c_a0_nativo_raw.csv", raw_nativo)

    n = len(resumen)
    print(f"\nTOTAL {n} reglas en {(time.time()-t_ini)/60:.1f} min")
    print(f"  Clase II (método viejo histórico):  {n_II}/{n} = {100.0*n_II/max(1,n):.1f}%")
    print(f"  Clase II (método viejo corregido):  {n_II_corr}/{n} = {100.0*n_II_corr/max(1,n):.1f}%")
    clases = {}
    for f in resumen:
        clases[f["clase_vieja"]] = clases.get(f["clase_vieja"], 0) + 1
    print(f"  distribución de clases (histórico): {clases}")
    desc = sum(1 for f in resumen if f["algun_descarrile"])
    print(f"  reglas con alguna escala descarrilada (bug de diámetro): {desc}/{n}")
    return resumen


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_REGLAS_OBJETIVO
    sb = int(sys.argv[2]) if len(sys.argv) > 2 else SEED_BASE
    main(n_reglas=n, seed_base=sb)
