"""
CS090 — FASE VI, tarea O2-E: RE-CORRIDA REPRODUCIBLE DEL BARRIDO DE FASE V-A
============================================================================================

QUIÉN SOY (y por qué existo)
----------------------------
El barrido original de Fase V-A (`cs090_fase5_completo.py`, informe `FASE5A_completo_resultado_CS.md`)
corrió 18 combinaciones de ejes (A0/A1/A2 × B0/B1 × C0/C1/C2) × 10 reglas = 180 reglas, de las que 150
se ejecutaron (los 3 combos A0+B1 lanzan `KeyError: 'adj'` con el motor congelado). Ese barrido tiene
DOS problemas de reproducibilidad, ninguno de ellos del método científico:

  1. las semillas de cada regla salían de `seed_base = abs(hash((eje_A, eje_B, eje_C, "paso1"))) % 100000`
     y `hash()` de strings/tuplas en Python está **aleatorizado por proceso** salvo que se fije
     `PYTHONHASHSEED` — que no quedó registrado;
  2. el CSV de salida **no guardaba la columna `seed`**, así que tampoco se puede recuperar a posteriori.

Consecuencia (documentada en `FASE6_adopcion_diam_corregido_CS.md` §3.1): cuando se descubrió el bug de
medición de diámetro (`_diam` de cs055 arranca el doble-BFS en el nodo no aislado de índice más bajo del
grafo ENTERO, y si ése cayó en un fragmento suelto mide el fragmento), Fase V-A quedó como **el único
resultado de la línea que no se pudo re-medir**: su impacto sólo se pudo inferir INDIRECTAMENTE por tres
pruebas sobre los datos guardados (las tres dieron 0/150 reglas afectadas). Es la última inferencia
indirecta que queda en pie tras la adopción de la corrección.

QUÉ HACE ESTE DRIVER
--------------------
Re-corre el MISMO barrido — mismos 18 combos, mismas 10 reglas por combo, mismo generador
(`cs090_fase5_generador`), mismo motor (`cs090_fase5_motor`), mismo clasificador y mismos umbrales,
mismos parámetros de corrida (N=2000, n_sweeps=14, escalas b=1/2/4/8/16, 3 semillas de NULL_topo) — con
tres diferencias, todas de METODO DE REGISTRO, ninguna de física:

  A. **`seed_base` explícito y fijo** (`SEED_RAIZ + PASO_COMBO * índice_de_combo`), nunca `hash()`.
     El barrido pasa a ser reproducible bit a bit de aquí en adelante.
  B. **La columna `seed` se guarda** en los dos CSV de salida (crudo y resumen), así que cualquier regla
     de este lote se puede reconstruir sola con `GEN.generar_regla(A, B, C, idx, seed)`.
  C. **Cada regla se mide con LAS DOS varas a la vez** — la vieja (`_diam` de cs055) y la corregida
     (`diam_gigante` de `cs090_diam_corregido.py`, oficial desde el 11-ago-2026 según
     `FASE6_adopcion_diam_corregido_CS.md`) — usando `correr_regla_coarse_doble`, que es copia exacta de
     la cadena de `correr_regla_coarse` y devuelve las filas medidas de las dos maneras SIN correr la
     dinámica dos veces. Así el efecto del bug se mide DIRECTAMENTE sobre las mismas reglas, en vez de
     inferirse.

Una sola pasada de 10 reglas por combo (el original corría en 2 pasadas de 5+5 con `seed_base` distinto,
lo que además hacía que los `rule_id` se repitieran entre pasadas: había dos `...-r0` por combo). Acá los
`rule_id` van r0..r9 y son únicos.

VERIFICACIÓN EXTRA (bloque 2): el Eje C no hace nada dentro de A0
-----------------------------------------------------------------
La tarea O1-C encontró que, dentro de A0, `dinamica_B0` retorna en la rama del campo en anillo ANTES de
cualquier bloque de costo (`cs090_fase5_motor.py`, líneas 190-199: `if kind == "A0": ... return sustrato`),
así que C0/C1/C2 producen campos bit a bit idénticos. Este driver lo VERIFICA numéricamente en vez de
darlo por bueno: corre A0-B0-C0, A0-B0-C1 y A0-B0-C2 con la MISMA semilla y compara las filas resultantes
campo por campo. Importa para leer el mapa global: las 3 filas A0-B0-* del mapa NO son 3 celdas
independientes, son la misma celda repetida 3 veces (30 reglas que en realidad son 10 configuraciones
distintas, cada una contada 3 veces con semillas distintas).

QUÉ NO HACE
-----------
No toca ningún script congelado ni existente (`cs055`, `cs080/81/82/83`, `cs090_fase5_generador/motor/
clasificador`, `cs090_fase5_completo.py`, `cs090_diam_corregido.py`) — sólo los importa. No cambia ni un
umbral del clasificador. No corre Phantom. No hace commits. **No declara cierre ni veredicto**: reporta
números; la lectura final es de Alexis.

USO
---
    python3.9 cs090_fase6_o2e_barrido_va_reproducible.py            # barrido completo + análisis
    python3.9 cs090_fase6_o2e_barrido_va_reproducible.py --analisis # sólo re-analiza los CSV ya escritos
    python3.9 cs090_fase6_o2e_barrido_va_reproducible.py --verificar-a0   # sólo el bloque 2
"""
from __future__ import annotations

import csv
import sys
import time
from collections import Counter, defaultdict

import numpy as np

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import cs090_fase5_generador as GEN            # generador + filtro P1-P5, SIN TOCAR
import cs090_diam_corregido as DIAM            # medición oficial corregida + correr_regla_coarse_doble
from cs090_fase5_clasificador import clasificar_regla   # umbrales originales, SIN TOCAR

# ============================ PARÁMETROS — IDÉNTICOS AL BARRIDO ORIGINAL ============================
EJES_A = ("A0", "A1", "A2")
EJES_B = ("B0", "B1")
EJES_C = ("C0", "C1", "C2")
TODOS_COMBOS = [(a, b, c) for a in EJES_A for b in EJES_B for c in EJES_C]   # 18, mismo orden que el original

N_REGLAS_POR_COMBO = 10       # igual que el objetivo del original
N_GRANDE = 2000               # igual
N_SWEEPS = 14                 # igual
ESCALAS_B = (1, 2, 4, 8, 16)  # igual
N_SEEDS_NULL_TOPO = 3         # igual

# ============================ LA DIFERENCIA: SEMILLAS EXPLÍCITAS Y FIJAS ============================
# El original: seed_base = abs(hash((eje_A, eje_B, eje_C, "paso1"))) % 100000  -> NO reproducible.
# Acá: función pura del índice del combo dentro de TODOS_COMBOS. Nada de hash(), nada de reloj, nada de
# entorno. `PASO_COMBO` (5000) es mayor que el máximo desplazamiento que puede usar el generador dentro
# de un combo (max_intentos=20 × 97 = 1940), así que dos combos NUNCA comparten una semilla.
# SEED_RAIZ=620000 está fuera del rango de semillas ya usadas por los barridos previos de esta línea
# (que van de ~270k a ~590k), así que este lote es una muestra genuinamente nueva.
SEED_RAIZ = 620_000
PASO_COMBO = 5_000


def seed_base_de_combo(eje_A, eje_B, eje_C):
    """Semilla base de un combo: determinista, explícita, sin `hash()`. Reproducible en cualquier
    máquina y con cualquier PYTHONHASHSEED."""
    idx = TODOS_COMBOS.index((eje_A, eje_B, eje_C))
    return SEED_RAIZ + PASO_COMBO * idx


# ============================ SALIDAS ============================
OUT_RAW = f"{_HERE}/cs090_fase6_o2e_barrido_va_raw.csv"          # una fila por regla × escala
OUT_RESUMEN = f"{_HERE}/cs090_fase6_o2e_barrido_va_resumen.csv"  # una fila por regla (CON seed)
OUT_A0 = f"{_HERE}/cs090_fase6_o2e_verificacion_a0_ejeC.csv"     # bloque 2

# Números publicados en FASE5A_completo_resultado_CS.md, para comparar contra el lote nuevo.
MAPA_PUBLICADO = {
    "A0-B0-C0": dict(n=10, I=5, II=5, III=0, IV=0),
    "A0-B0-C1": dict(n=10, I=9, II=1, III=0, IV=0),
    "A0-B0-C2": dict(n=10, I=8, II=2, III=0, IV=0),
    "A0-B1-C0": None, "A0-B1-C1": None, "A0-B1-C2": None,      # no ejecutables
    "A1-B0-C0": dict(n=10, I=6, II=4, III=0, IV=0),
    "A1-B0-C1": dict(n=10, I=6, II=4, III=0, IV=0),
    "A1-B0-C2": dict(n=10, I=5, II=2, III=3, IV=0),
    "A1-B1-C0": dict(n=10, I=8, II=2, III=0, IV=0),
    "A1-B1-C1": dict(n=10, I=7, II=3, III=0, IV=0),
    "A1-B1-C2": dict(n=10, I=7, II=3, III=0, IV=0),
    "A2-B0-C0": dict(n=10, I=6, II=4, III=0, IV=0),
    "A2-B0-C1": dict(n=10, I=8, II=2, III=0, IV=0),
    "A2-B0-C2": dict(n=10, I=5, II=0, III=5, IV=0),
    "A2-B1-C0": dict(n=10, I=7, II=3, III=0, IV=0),
    "A2-B1-C1": dict(n=10, I=5, II=5, III=0, IV=0),
    "A2-B1-C2": dict(n=10, I=7, II=3, III=0, IV=0),
}
GLOBAL_PUBLICADO = dict(I=99, II=43, III=8, IV=0, intermedio=0, n=150)


# ============================================================================================
# BLOQUE 1 — EL BARRIDO
# ============================================================================================
def correr_una_regla(p, eje_A, eje_B, eje_C):
    """Corre UNA regla con las dos varas de medición a la vez y la clasifica dos veces (con los mismos
    umbrales, sin tocarlos). Devuelve (resultado_regla, filas_crudas, segundos).

    `correr_regla_coarse_doble` es copia exacta de la cadena de `cs090_fase5_motor.correr_regla_coarse`
    (mismos rng derivados, mismo coarse-graining de cs080, mismos NULL_topo) que mide el diámetro de las
    dos maneras en cada escala, tanto en REAL como en los NULL. La dinámica se corre UNA sola vez, así
    que 'viejo' y 'corregido' hablan literalmente del mismo grafo — no de dos realizaciones distintas.
    """
    t0 = time.time()
    filas_viejo, filas_corr, diagnos = DIAM.correr_regla_coarse_doble(
        p, N=N_GRANDE, n_sweeps=N_SWEEPS, escalas_b=ESCALAS_B,
        n_seeds_null_topo=N_SEEDS_NULL_TOPO)
    r_viejo = clasificar_regla(filas_viejo)
    r_corr = clasificar_regla(filas_corr)
    dt = time.time() - t0

    # ---- filas crudas: una por escala, con TODO lo necesario para re-medir sin re-correr ----
    filas_raw = []
    n_escalas_descarrila = 0
    n_escalas_difiere_real = 0
    n_escalas_difiere_null = 0
    for fv, fc, dg in zip(filas_viejo, filas_corr, diagnos):
        descarrila = bool(dg.get("descarrila", False))
        difiere_real = bool(fv["diam_real"] != fc["diam_real"])
        difiere_null = bool(abs(fv["diam_null_topo"] - fc["diam_null_topo"]) > 1e-12)
        n_escalas_descarrila += int(descarrila)
        n_escalas_difiere_real += int(difiere_real)
        n_escalas_difiere_null += int(difiere_null)
        filas_raw.append(dict(
            combo=f"{eje_A}-{eje_B}-{eje_C}", eje_A=eje_A, eje_B=eje_B, eje_C=eje_C,
            rule_id=p["rule_id"], seed=p["seed"],
            K=p["K"], J=p["J"], noise=p["noise"], meandeg=p["meandeg"],
            sim_thr_frac=p["sim_thr_frac"], kcap=p["kcap"],
            escala_b=fv["escala_b"], N=fv["N"],
            diam_real_viejo=fv["diam_real"], diam_real_corr=fc["diam_real"],
            diam_null_viejo=fv["diam_null_topo"], diam_null_corr=fc["diam_null_topo"],
            diam_null_std_viejo=fv["diam_null_topo_std"], diam_null_std_corr=fc["diam_null_topo_std"],
            giant_real=fv["giant_real"], holon_real=fv["holon_real"],
            holon_null_valor=fv["holon_null_valor"],
            n_aristas=fv["n_aristas"], n_triangulos=fv["n_triangulos"],
            tam_comp_medida=dg.get("tam_comp_medida"), tam_gigante=dg.get("tam_gigante"),
            n_componentes=dg.get("n_componentes"), n_aislados=dg.get("n_aislados"),
            descarrila=descarrila, difiere_real=difiere_real, difiere_null=difiere_null,
        ))

    res = dict(
        combo=f"{eje_A}-{eje_B}-{eje_C}", eje_A=eje_A, eje_B=eje_B, eje_C=eje_C,
        rule_id=p["rule_id"], seed=p["seed"],
        K=p["K"], J=p["J"], noise=p["noise"], meandeg=p["meandeg"],
        sim_thr_frac=p["sim_thr_frac"], kcap=p["kcap"],
        clase_vieja=r_viejo["clase"], clase_corregida=r_corr["clase"],
        cambia_clase=bool(r_viejo["clase"] != r_corr["clase"]),
        pendiente_vieja=r_viejo["pendiente_real"], pendiente_corr=r_corr["pendiente_real"],
        z_agg_viejo=r_viejo["z_agg"], z_agg_corr=r_corr["z_agg"],
        z_sost_viejo=r_viejo["z_sostenido"], z_sost_corr=r_corr["z_sostenido"],
        holon_ratio=r_viejo["holon_ratio"],
        n_escalas_descarrila=n_escalas_descarrila,
        n_escalas_difiere_real=n_escalas_difiere_real,
        n_escalas_difiere_null=n_escalas_difiere_null,
        diam_viejo_por_escala="|".join(f"{f['diam_real']:.0f}" for f in filas_viejo),
        diam_corr_por_escala="|".join(f"{f['diam_real']:.0f}" for f in filas_corr),
        gigante_frac_b1=(diagnos[0].get("tam_gigante") or 0) / float(N_GRANDE),
        segundos=round(dt, 2),
    )
    return res, filas_raw, dt


def correr_barrido():
    print("=" * 108)
    print("O2-E — BARRIDO FASE V-A REPRODUCIBLE (18 combos × 10 reglas, seed guardada, DOS mediciones)")
    print(f"       N={N_GRANDE}  n_sweeps={N_SWEEPS}  escalas_b={ESCALAS_B}  nulls={N_SEEDS_NULL_TOPO}")
    print(f"       SEED_RAIZ={SEED_RAIZ}  PASO_COMBO={PASO_COMBO}  (sin hash(), reproducible)")
    print("=" * 108)

    t_inicio = time.time()
    resumen, raw = [], []
    combos_no_ejecutables = {}
    total_admitidas = total_descartadas = 0

    for (eje_A, eje_B, eje_C) in TODOS_COMBOS:
        combo = f"{eje_A}-{eje_B}-{eje_C}"
        sb = seed_base_de_combo(eje_A, eje_B, eje_C)
        admitidas, descartadas = GEN.generar_reglas_clase(
            eje_A, eje_B, eje_C, n_reglas=N_REGLAS_POR_COMBO, seed_base=sb)
        total_admitidas += len(admitidas); total_descartadas += len(descartadas)
        print(f"\n--- {combo}  seed_base={sb}  (admitidas={len(admitidas)} descartadas={len(descartadas)}) ---")

        fallos_seguidos = 0
        for p in admitidas:
            try:
                res, filas_raw, dt = correr_una_regla(p, eje_A, eje_B, eje_C)
            except Exception as e:
                # Mismo comportamiento que el driver original: se documenta, NO se parchea el motor.
                print(f"  {p['rule_id']} (seed={p['seed']}): *** MOTOR NO EJECUTABLE *** "
                      f"{type(e).__name__}: {e}")
                combos_no_ejecutables.setdefault(combo, f"{type(e).__name__}: {e}")
                fallos_seguidos += 1
                if fallos_seguidos >= 2:
                    print(f"  -- {combo}: 2 fallos seguidos -> bug sistemático del motor para esta "
                          f"combinación de ejes; se corta el combo (igual que el barrido original) --")
                    break
                continue
            fallos_seguidos = 0
            resumen.append(res); raw.extend(filas_raw)
            marca = ""
            if res["cambia_clase"]:
                marca = f"   <<< CAMBIA DE CLASE ({res['clase_vieja']} -> {res['clase_corregida']})"
            elif res["n_escalas_difiere_real"] or res["n_escalas_difiere_null"]:
                marca = "   (difiere el diámetro pero NO la clase)"
            print(f"  {p['rule_id']} seed={p['seed']}: clase_vieja={res['clase_vieja']:<24} "
                  f"clase_corr={res['clase_corregida']:<24} pend={res['pendiente_vieja']:+.3f}/"
                  f"{res['pendiente_corr']:+.3f}  z={res['z_agg_viejo']:.2f}/{res['z_agg_corr']:.2f} "
                  f"({dt:.1f}s) [t={time.time()-t_inicio:.0f}s]{marca}")

    # ------------------------------------ CSV ------------------------------------
    if raw:
        with open(OUT_RAW, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(raw[0].keys())); wr.writeheader()
            wr.writerows(raw)
        print(f"\nCSV crudo (una fila por regla×escala, CON seed): {OUT_RAW}  ({len(raw)} filas)")
    if resumen:
        with open(OUT_RESUMEN, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(resumen[0].keys())); wr.writeheader()
            wr.writerows(resumen)
        print(f"CSV resumen (una fila por regla, CON seed): {OUT_RESUMEN}  ({len(resumen)} filas)")

    print(f"\nFiltro P1-P5: admitidas={total_admitidas}  descartadas={total_descartadas}")
    if combos_no_ejecutables:
        print(f"Combos NO ejecutables con el motor congelado: {len(combos_no_ejecutables)}")
        for c, err in combos_no_ejecutables.items():
            print(f"   {c}: {err}")
    print(f"Tiempo total del barrido: {(time.time()-t_inicio)/60:.1f} min")
    return resumen


# ============================================================================================
# BLOQUE 2 — VERIFICACIÓN: dentro de A0, el Eje C no tiene NINGÚN efecto
# ============================================================================================
def verificar_a0_eje_c(n_seeds=5):
    """Corre A0-B0-C0, A0-B0-C1 y A0-B0-C2 con la MISMA semilla y compara los resultados campo por campo.

    Por qué debería dar idénticos (lectura del código, que acá se comprueba en vez de creerse):
    `cs090_fase5_motor.dinamica_B0` empieza con `if kind == "A0": ...; return sustrato` — la rama del
    campo continuo en anillo hace sus n_sweeps de difusión y RETORNA, antes de llegar a cualquier línea
    que consulte `costo_nivel` (kcap de C2, poda por costo de C1/C2, conteo de flips). El parámetro
    `costo_nivel` entra a la función y no se usa nunca en esa rama. Y `medir()` para A0 deriva el grafo
    de medición del campo S, que ya es idéntico. Por lo tanto TODO lo medido debe coincidir bit a bit.

    Implicación para el mapa global: las 3 filas A0-B0-C0/C1/C2 no son 3 celdas independientes del
    diseño factorial; son la MISMA celda. Las 30 reglas A0 del barrido son 30 configuraciones distintas
    sólo porque cada combo usa semillas distintas, no porque los ejes hagan algo distinto.
    """
    print("\n" + "=" * 108)
    print("BLOQUE 2 — ¿el Eje C hace algo dentro de A0? (misma semilla, C0 vs C1 vs C2)")
    print("=" * 108)
    filas_out = []
    todas_iguales = True
    for k in range(n_seeds):
        # fuera del rango del barrido (620000..705000+1844) para no repetir ninguna regla del lote
        seed = 810_000 + 137 * k
        por_c = {}
        for c in ("C0", "C1", "C2"):
            p = GEN.generar_regla("A0", "B0", c, idx=k, seed=seed)
            fv, fc, dg = DIAM.correr_regla_coarse_doble(
                p, N=N_GRANDE, n_sweeps=N_SWEEPS, escalas_b=ESCALAS_B,
                n_seeds_null_topo=N_SEEDS_NULL_TOPO)
            por_c[c] = (fv, fc, clasificar_regla(fv), clasificar_regla(fc))
        # comparación bit a bit de los campos medidos (se ignora `dt`, que es tiempo de reloj)
        campos = ("diam_real", "diam_null_topo", "diam_null_topo_std", "giant_real",
                  "holon_real", "holon_null_valor", "n_aristas", "n_triangulos", "N")
        iguales = {}
        for c in ("C1", "C2"):
            ok = all(por_c["C0"][0][i][campo] == por_c[c][0][i][campo]
                     for i in range(len(ESCALAS_B)) for campo in campos)
            ok = ok and por_c["C0"][2]["clase"] == por_c[c][2]["clase"]
            iguales[c] = ok
            todas_iguales = todas_iguales and ok
        print(f"  seed={seed}: C0 clase={por_c['C0'][2]['clase']:<24} "
              f"pend={por_c['C0'][2]['pendiente_real']:+.3f}   "
              f"¿C1 idéntico a C0? {iguales['C1']}   ¿C2 idéntico a C0? {iguales['C2']}")
        for c in ("C0", "C1", "C2"):
            filas_out.append(dict(
                seed=seed, eje_C=c, clase_vieja=por_c[c][2]["clase"],
                clase_corregida=por_c[c][3]["clase"],
                pendiente_vieja=por_c[c][2]["pendiente_real"],
                pendiente_corr=por_c[c][3]["pendiente_real"],
                z_agg_viejo=por_c[c][2]["z_agg"], z_agg_corr=por_c[c][3]["z_agg"],
                diam_por_escala="|".join(f"{f['diam_real']:.0f}" for f in por_c[c][0]),
                n_aristas_b1=por_c[c][0][0]["n_aristas"],
                giant_b1=por_c[c][0][0]["giant_real"],
                identico_a_C0=(True if c == "C0" else iguales[c]),
            ))
    with open(OUT_A0, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(filas_out[0].keys())); wr.writeheader()
        wr.writerows(filas_out)
    if todas_iguales:
        print(f"\n  -> CONFIRMADO: dentro de A0-B0, C0/C1/C2 producen resultados bit a bit idénticos "
              f"en las {n_seeds}/{n_seeds} semillas probadas. El Eje C no tiene NINGÚN efecto en A0.")
    else:
        print(f"\n  -> NO CONFIRMADO: alguna semilla difiere entre C0/C1/C2 — ver {OUT_A0}")
    print(f"  CSV: {OUT_A0}")
    return todas_iguales


# ============================================================================================
# BLOQUE 3 — ANÁLISIS
# ============================================================================================
CLASES = ("I", "II", "III", "IV", "intermedio (sin clase clara)")


def _leer_resumen():
    with open(OUT_RESUMEN) as fh:
        filas = list(csv.DictReader(fh))
    for f in filas:
        for k in ("pendiente_vieja", "pendiente_corr", "z_agg_viejo", "z_agg_corr",
                  "holon_ratio", "gigante_frac_b1"):
            f[k] = float(f[k])
        for k in ("n_escalas_descarrila", "n_escalas_difiere_real", "n_escalas_difiere_null",
                  "seed", "kcap", "K"):
            f[k] = int(float(f[k]))
        f["cambia_clase"] = (f["cambia_clase"] == "True")
    return filas


def _cuenta(filas, campo):
    c = Counter(f[campo] for f in filas)
    return {k: c.get(k, 0) for k in CLASES}


def analizar(filas=None):
    if filas is None:
        filas = _leer_resumen()
    n = len(filas)
    print("\n" + "=" * 108)
    print(f"ANÁLISIS — {n} reglas corridas")
    print("=" * 108)

    # ---------- 1) impacto DIRECTO del bug de diámetro ----------
    print("\n### 1) IMPACTO DIRECTO DEL BUG DE DIÁMETRO (medición, ya no inferencia) ###")
    n_desc = sum(1 for f in filas if f["n_escalas_descarrila"] > 0)
    n_dif_real = sum(1 for f in filas if f["n_escalas_difiere_real"] > 0)
    n_dif_null = sum(1 for f in filas if f["n_escalas_difiere_null"] > 0)
    n_dif_pend = sum(1 for f in filas if abs(f["pendiente_vieja"] - f["pendiente_corr"]) > 1e-12)
    n_cambia = sum(1 for f in filas if f["cambia_clase"])
    print(f"  reglas con descarrilamiento (comp. medida < 10% de la gigante, alguna escala): {n_desc}/{n}")
    print(f"  reglas donde diám REAL difiere viejo-vs-corregido (alguna escala):              {n_dif_real}/{n}")
    print(f"  reglas donde diám de algún NULL difiere:                                        {n_dif_null}/{n}")
    print(f"  reglas donde la PENDIENTE cambia:                                               {n_dif_pend}/{n}")
    print(f"  reglas que CAMBIAN DE CLASE:                                                    {n_cambia}/{n}")
    if n_cambia:
        print("\n  detalle de las que cambian de clase:")
        for f in filas:
            if f["cambia_clase"]:
                print(f"    {f['combo']:<10} {f['rule_id']:<18} seed={f['seed']:<7} "
                      f"{f['clase_vieja']} -> {f['clase_corregida']}  "
                      f"pend {f['pendiente_vieja']:+.3f} -> {f['pendiente_corr']:+.3f}  "
                      f"diám {f['diam_viejo_por_escala']} -> {f['diam_corr_por_escala']}")
    pend_neg = [f for f in filas if f["pendiente_vieja"] < 0]
    print(f"\n  reglas con pendiente vieja NEGATIVA (la firma clásica del bug): {len(pend_neg)}/{n}")
    diam_b1_min = min(int(f["diam_viejo_por_escala"].split("|")[0]) for f in filas)
    print(f"  diám(b=1) mínimo del lote (vara vieja): {diam_b1_min}   "
          f"[las 15 descarriladas del barrido de 430 tenían <=3; las sanas >=8]")
    gmin = min(f["gigante_frac_b1"] for f in filas)
    print(f"  fracción de componente gigante a b=1, mínima del lote: {gmin:.4f}")

    # ---------- 2) mapa global ----------
    print("\n### 2) MAPA GLOBAL 18×4 ###")
    por_combo = defaultdict(list)
    for f in filas:
        por_combo[f["combo"]].append(f)
    cab = (f"  {'combo':<11} {'n':>3} | {'I':>3} {'II':>3} {'III':>3} {'IV':>3} {'int':>4} (corregida) | "
           f"{'I':>3} {'II':>3} {'III':>3} {'IV':>3} {'int':>4} (vieja) | publicado I/II/III/IV")
    print(cab)
    for (a, b, c) in TODOS_COMBOS:
        combo = f"{a}-{b}-{c}"
        fs = por_combo.get(combo, [])
        pub = MAPA_PUBLICADO[combo]
        pubtxt = "no ejecutable" if pub is None else f"{pub['I']}/{pub['II']}/{pub['III']}/{pub['IV']}"
        if not fs:
            print(f"  {combo:<11} {0:>3} | {'—':>3} {'—':>3} {'—':>3} {'—':>3} {'—':>4}            | "
                  f"{'—':>3} {'—':>3} {'—':>3} {'—':>3} {'—':>4}        | {pubtxt}")
            continue
        cc, cv = _cuenta(fs, "clase_corregida"), _cuenta(fs, "clase_vieja")
        print(f"  {combo:<11} {len(fs):>3} | {cc['I']:>3} {cc['II']:>3} {cc['III']:>3} {cc['IV']:>3} "
              f"{cc['intermedio (sin clase clara)']:>4}            | "
              f"{cv['I']:>3} {cv['II']:>3} {cv['III']:>3} {cv['IV']:>3} "
              f"{cv['intermedio (sin clase clara)']:>4}        | {pubtxt}")

    gc, gv = _cuenta(filas, "clase_corregida"), _cuenta(filas, "clase_vieja")
    def pct(d):
        return {k: f"{v} ({100.0*v/n:.0f}%)" for k, v in d.items()}
    print(f"\n  GLOBAL corregida: {pct(gc)}")
    print(f"  GLOBAL vieja:     {pct(gv)}")
    print(f"  GLOBAL publicado: I=99 (66%), II=43 (29%), III=8 (5%), IV=0, intermedio=0  [n=150]")

    # ---------- 3) las tres conclusiones publicadas ----------
    print("\n### 3) LAS TRES CONCLUSIONES PUBLICADAS, RECALCULADAS ###")

    print('\n  (a) "A0 nunca Clase II o superior" — ¿sigue falsificada?')
    a0 = [f for f in filas if f["eje_A"] == "A0"]
    for etq, campo in (("corregida", "clase_corregida"), ("vieja", "clase_vieja")):
        c = _cuenta(a0, campo)
        n2mas = c["II"] + c["III"] + c["IV"]
        print(f"      medición {etq:<10}: de {len(a0)} reglas A0 -> Clase II+ = {n2mas} "
              f"({100.0*n2mas/max(1,len(a0)):.0f}%)   {c}")
    print(f"      publicado           : de 30 reglas A0 -> Clase II+ = 8 (27%)")
    print(f"      gigante(b=1) de las A0: min={min(f['gigante_frac_b1'] for f in a0):.4f} "
          f"max={max(f['gigante_frac_b1'] for f in a0):.4f}  "
          f"(si es 1.0000 el bug no tiene dónde morder)")
    print(f"      A0 que cambian de clase por la corrección: {sum(1 for f in a0 if f['cambia_clase'])}/{len(a0)}")

    print('\n  (b) "muy fuerte CONTRADICHO" — ¿B0+C2 sigue superando a B1+C1/C2?')
    for etq, campo in (("corregida", "clase_corregida"), ("vieja", "clase_vieja")):
        b0c2 = [f for f in filas if f["eje_B"] == "B0" and f["eje_C"] == "C2"]
        b1c12 = [f for f in filas if f["eje_B"] == "B1" and f["eje_C"] in ("C1", "C2")]
        def n34(fs):
            return sum(1 for f in fs if f[campo] in ("III", "IV"))
        print(f"      medición {etq:<10}: B0+C2 -> Clase III/IV = {n34(b0c2)}/{len(b0c2)}   |   "
              f"B1+C1/C2 -> Clase III/IV = {n34(b1c12)}/{len(b1c12)}")
    print(f"      publicado           : B0+C2 -> 8/30 (A1-B0-C2 3/10, A2-B0-C2 5/10)   |   "
          f"B1+C1/C2 -> 0/40")

    print('\n  (c) "0 reglas en Clase IV en todo V-A" y "B0 nunca Clase IV"')
    for etq, campo in (("corregida", "clase_corregida"), ("vieja", "clase_vieja")):
        n4 = sum(1 for f in filas if f[campo] == "IV")
        n4_b0 = sum(1 for f in filas if f[campo] == "IV" and f["eje_B"] == "B0")
        print(f"      medición {etq:<10}: Clase IV = {n4}/{n} (de ellas, B0: {n4_b0})")
    print(f"      publicado           : Clase IV = 0/150")

    print('\n  (d) criterio "débil": ¿>15% de II+III por combo?')
    for etq, campo in (("corregida", "clase_corregida"), ("vieja", "clase_vieja")):
        ok = 0; total = 0
        for (a, b, c) in TODOS_COMBOS:
            fs = por_combo.get(f"{a}-{b}-{c}", [])
            if not fs:
                continue
            total += 1
            cc = _cuenta(fs, campo)
            if (cc["II"] + cc["III"] + cc["IV"]) / len(fs) > 0.15:
                ok += 1
        print(f"      medición {etq:<10}: {ok}/{total} combos superan 15% en II+III+IV")
    print(f"      publicado           : 14/15 combos")

    # ---------- 4) separar "cambió por el bug" de "cambió por ser otra muestra" ----------
    print("\n### 4) BUG vs MUESTREO — la comparación que importa ###")
    print("  'cambió por el BUG'      = clase_vieja vs clase_corregida, SOBRE LAS MISMAS reglas nuevas")
    print("  'cambió por la MUESTRA'  = este lote (vara vieja) vs los números publicados en agosto")
    print(f"\n  por el BUG:     {n_cambia}/{n} reglas cambian de clase "
          f"({100.0*n_cambia/n:.1f}%). Global vieja {[gv[k] for k in ('I','II','III','IV')]} -> "
          f"corregida {[gc[k] for k in ('I','II','III','IV')]}")
    dif_pub = {k: gv[k] - GLOBAL_PUBLICADO[k] for k in ("I", "II", "III", "IV")}
    print(f"  por la MUESTRA: este lote con la vara VIEJA da "
          f"{[gv[k] for k in ('I','II','III','IV')]} contra el publicado "
          f"{[GLOBAL_PUBLICADO[k] for k in ('I','II','III','IV')]}  -> diferencia {dif_pub}")

    # binomial exacta combo a combo: ¿la diferencia con lo publicado es compatible con muestreo?
    print("\n  ¿es esa diferencia compatible con puro muestreo? (n=10 por combo es chico)")
    try:
        from scipy import stats as _st
        # test global de bondad de ajuste: distribución del lote nuevo (vara vieja) contra las
        # proporciones publicadas, con chi-cuadrado sobre I / II / III+IV
        obs = np.array([gv["I"], gv["II"], gv["III"] + gv["IV"] + gv["intermedio (sin clase clara)"]])
        p_pub = np.array([GLOBAL_PUBLICADO["I"], GLOBAL_PUBLICADO["II"],
                          GLOBAL_PUBLICADO["III"] + GLOBAL_PUBLICADO["IV"]], dtype=float)
        p_pub = p_pub / p_pub.sum()
        esp = p_pub * obs.sum()
        chi2 = float(((obs - esp) ** 2 / esp).sum())
        pval = float(1 - _st.chi2.cdf(chi2, df=2))
        print(f"     chi2 (I / II / III+IV, lote nuevo vara vieja vs proporciones publicadas) = "
              f"{chi2:.2f}, gl=2, p={pval:.4f}")
        print(f"     observado={obs.tolist()}  esperado={[round(x,1) for x in esp]}")
    except Exception as e:
        print(f"     (scipy no disponible: {e})")

    print("\nNo se declara cierre ni veredicto. Números arriba; lectura final de Alexis.")


# ============================================================================================
if __name__ == "__main__":
    args = set(sys.argv[1:])
    if "--analisis" in args:
        analizar()
    elif "--verificar-a0" in args:
        verificar_a0_eje_c()
    else:
        filas = correr_barrido()
        verificar_a0_eje_c()
        analizar()
