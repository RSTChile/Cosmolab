"""
CS090 — FASE V-A: MÉTRICAS NATIVAS DE CAMPO CONTINUO PARA A0 (sin grafo derivado)
====================================================================================================
QUIÉN SOY: script NUEVO (no toca ningún congelado) que compara, sobre las MISMAS instancias de campo
A0, (a) el método VIEJO de clasificación (grafo de medición derivado `_grafo_medicion_A0` +
diám/giant/holonomía + coarse-graining, tal como está en cs090_fase5_motor.py / clasificador.py, sólo
importados) contra (b) métricas NATIVAS que leen el array de fase `S` directamente sobre el anillo fijo
(misma adyacencia left/right que usa la dinámica), sin construir NINGÚN grafo derivado.

POR QUÉ ESTO IMPORTA (contexto, ver FASE5A_completo_resultado_CS.md): en el barrido de 180 reglas,
27% de las reglas A0 (sustrato SIN grafo, campo continuo genuino) igual cayeron en "Clase II —
mundo-pequeño congelado". Eso es sospechoso porque la ÚNICA forma que tiene el pipeline de medir A0 es
derivar un grafo de similitud por MUESTREO DE CANDIDATOS AL AZAR por nodo (`_grafo_medicion_A0`,
motor.py líneas ~307-325): cada sitio se compara contra `n_cand=15` candidatos elegidos AL AZAR (no
sólo sus vecinos físicos) y se conecta si están en fase similar. Ese muestreo de candidatos lejanos es,
estructuralmente, la MISMA receta que un grafo de Watts-Strogatz (unos pocos atajos de largo alcance
sobre una base local) — que es EXACTAMENTE la firma que produce mundo-pequeño. Osea: el método de
medición podría estar fabricando el propio fenómeno que dice medir. Este script pone eso a prueba.

QUÉ SE REUSA (sólo import, cero ediciones a los congelados):
  - cs090_fase5_generador.generar_reglas_clase  → genera reglas A0-B0-C0 admitidas (P1-P5 reales)
  - cs090_fase5_motor.construir_A0 / dinamica_B0 → construye y evoluciona el campo EXACTAMENTE como
    lo hace el pipeline congelado (misma dinámica de difusión local en anillo, mismos parámetros)
  - cs090_fase5_motor.medir / medir_null_topo / medir_null_valor → método VIEJO, sin tocar
  - cs090_fase5_clasificador.clasificar_regla    → clasificación Clase I-IV, sin tocar

QUÉ ES NUEVO (este archivo, nada en los congelados):
  - correlacion_circular / longitud_correlacion  → ξ(r): correlación de fase entre sitios separados
    por distancia r en el anillo NATIVO (misma métrica de distancia que usan los offsets left/right de
    la dinámica -- ninguna coordenada nueva, ningún grafo). Analiza cómo esa longitud de correlación
    escala con N (pendiente log-log), análogo nativo de la pendiente diám-vs-N del método viejo.
  - dominios_locales                             → percolación de dominios de fase SOBRE EL ANILLO
    NATIVO: dos sitios ADYACENTES en el anillo (i, i+1, la misma adyacencia física que usa la
    dinámica) se funden en un dominio si su diferencia de fase < umbral (mismo umbral sim_thr_frac*K
    que usa el método viejo, para comparar manzanas con manzanas). CERO muestreo de candidatos al
    azar ni saltos de largo alcance -- a propósito, es el contraste directo con `_grafo_medicion_A0`.
  - NULL nativo: mismo array de valores S, posiciones BARAJADAS al azar sobre el anillo (destruye el
    orden espacial, conserva la distribución de valores) -- control de que las métricas nativas miden
    estructura ESPACIAL genuina y no sólo "hay algo no-uniforme en los valores".

MÉTRICAS ELEGIDAS Y DESCARTADAS (documentado, no forzado): de las 5 sugeridas en el plan (ξ(r),
box-counting, espectro de difusión, percolación de nivel, escalamiento masa-radio) se implementan 2:
ξ(r) (correlación espacial) y percolación de dominios locales (análogo nativo de "giant component").
Se fusionan porque ambas miden esencialmente la misma pregunta -- "¿hasta qué distancia se sostiene la
coherencia de fase?" -- desde dos ángulos independientes (decaimiento continuo vs. componente conexa
discreta), lo cual ya da comparación cruzada. Box-counting con umbral fijo en 1D casi siempre da
dimensión ≈1 trivialmente (el dominio activo es casi todo el anillo o casi nada, poco informativo).
Espectro del operador de difusión NO se implementa: el operador (1-J)·I + J·(medio vecino izq+der) es
CIRCULANTE y sus autovalores son analíticos -- (1-J)+J·cos(2πk/N) -- dependen sólo de J y N, NUNCA del
estado realizado del campo; no discrimina entre reglas ni entre REAL/NULL (sería measurement-free en el
peor sentido). Masa-radio queda subsumida por dominios_locales (tamaño de dominio vs. radio ES
masa-radio con la métrica de "misma fase").

Guardián no-hornear: ninguna coordenada (x,y,z) en este archivo; toda distancia es índice de anillo
(offset fijo), igual que en el congelado. No se toca ningún script congelado. No declara cierre --
reporta números, la lectura final es de Alexis.
"""
from __future__ import annotations
import sys, time, csv
import numpy as np

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)

import cs090_fase5_generador as GEN     # generar_reglas_clase -- SÓLO IMPORT, no se toca
import cs090_fase5_motor as MOT         # construir_A0/dinamica_B0/medir/medir_null_* -- SÓLO IMPORT
import cs090_fase5_clasificador as CLS  # clasificar_regla -- SÓLO IMPORT


# ============================================================================================
# 1) MÉTRICAS NATIVAS -- leen sustrato["S"] directo sobre el anillo fijo, CERO grafo derivado
# ============================================================================================
def correlacion_circular(S, K, n_r=30):
    """ξ(r): correlación de fase circular <cos(θ_i - θ_{i+r})> entre sitios separados por distancia r
    en el ANILLO NATIVO (índice fijo, misma métrica que usan los offsets left/right de la dinámica).
    r muestreado en `n_r` puntos espaciados logarítmicamente entre 1 y N//2 (no los N//2 enteros
    completos -- mantiene el costo bajo sin perder la escala relevante; la correlación de fase suele
    variar suavemente en r, un muestreo log-espaciado la resuelve bien con 1/30 del costo)."""
    N = len(S)
    r_max = max(1, N // 2)
    rs = np.unique(np.clip(np.round(np.logspace(0, np.log10(r_max), n_r)).astype(int), 1, r_max))
    z = np.exp(1j * 2 * np.pi * S / K)
    C = np.array([np.mean(np.real(z * np.conj(np.roll(z, -r)))) for r in rs])
    C0 = float(np.mean(np.real(z * np.conj(z))))  # =1 por construcción (|z|=1), referencia explícita
    return rs, C, C0


def longitud_correlacion(rs, C, C0, umbral_frac=1.0 / np.e):
    """Primer r (de los muestreados) donde C(r) cae bajo umbral_frac*C0 -- criterio estándar de
    longitud de decaimiento (1/e). Si nunca cae dentro del rango muestreado, la correlación se
    considera SATURADA (coherencia sostenida hasta la escala más grande medida) y se reporta rs[-1]
    (≈N/2) como cota inferior de la longitud real, documentado como tal en el resultado."""
    umbral = umbral_frac * C0
    bajo = np.where(C < umbral)[0]
    if len(bajo) == 0:
        return float(rs[-1]), True   # (longitud, saturada=True)
    return float(rs[bajo[0]]), False


def dominios_locales(S, K, thr):
    """Percolación de dominios de fase sobre el ANILLO NATIVO: dos sitios FÍSICAMENTE ADYACENTES
    (i, i+1 mod N -- la misma adyacencia que usan left/right en la dinámica) quedan en el MISMO
    dominio si su diferencia circular de fase < thr. Ninguna muestra de candidatos al azar, ningún
    salto de largo alcance -- contraste directo con `_grafo_medicion_A0`. Devuelve la lista de tamaños
    de dominio y la fracción del anillo cubierta por el dominio más grande (giant nativo)."""
    N = len(S)
    dif = np.abs(S - np.roll(S, -1)) % K
    dif = np.minimum(dif, K - dif)
    frontera = dif >= thr   # borde entre dominios en el enlace (i, i+1)
    idx_frontera = np.where(frontera)[0]
    if len(idx_frontera) == 0:
        return [N], 1.0   # todo el anillo es un solo dominio (sin fronteras)
    tamanos = np.diff(np.concatenate([idx_frontera, [idx_frontera[0] + N]]))
    return tamanos.tolist(), float(np.max(tamanos) / N)


def metricas_nativas_A0(S, K, thr):
    rs, C, C0 = correlacion_circular(S, K)
    lam, saturada = longitud_correlacion(rs, C, C0)
    tamanos, giant_frac = dominios_locales(S, K, thr)
    return dict(corr_len=lam, corr_len_saturada=saturada,
                giant_nativo_frac=giant_frac, giant_nativo_size=float(np.max(tamanos)),
                n_dominios=len(tamanos))


def _pendiente_loglog(xs, ys):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    mask = np.isfinite(ys) & (ys > 0) & (xs > 0)
    if mask.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(xs[mask]), np.log(ys[mask]), 1)[0])


# ============================================================================================
# 2) COARSE-GRAINING NATIVO -- caja = tramo contiguo de b sitios del anillo (orden intrínseco del
#    dominio, CERO BFS/grafo: a diferencia de cajas_bfs/grafo_grueso de cs080 -- que hacen falta
#    porque un grafo genérico no tiene orden natural -- el anillo YA es 1D y ordenado, agrupar b
#    sitios consecutivos ES la caja de box-counting nativa, sin inventar estructura nueva).
# ============================================================================================
def coarsear_campo_ring(S, K, b):
    """Agrupa el anillo en cajas de b sitios consecutivos; el valor de cada caja es la MEDIA CIRCULAR
    de fase de sus miembros (mismo tipo de promedio que usa `_circ_mean_update`/la propia dinámica de
    A0, vía vector unitario exp(iθ)). Devuelve S_b sobre n_cajas=ceil(N/b) sitios (la última caja
    absorbe el resto si N no es múltiplo de b -- con N=2000 y b en {1,2,4,8,16} siempre es exacto)."""
    N = len(S)
    n_cajas = max(1, -(-N // b))  # ceil(N/b)
    idx_caja = np.minimum(np.arange(N) // b, n_cajas - 1)
    z = np.exp(1j * 2 * np.pi * S / K)
    suma_re = np.bincount(idx_caja, weights=z.real, minlength=n_cajas)
    suma_im = np.bincount(idx_caja, weights=z.imag, minlength=n_cajas)
    cuenta = np.bincount(idx_caja, minlength=n_cajas).astype(float)
    z_b = (suma_re + 1j * suma_im) / np.maximum(cuenta, 1)
    S_b = (np.angle(z_b) / (2 * np.pi) * K) % K
    return S_b, n_cajas


# ============================================================================================
# 3) CORRER UNA REGLA: UN campo A0 a N=2000 (misma semilla/dinámica que `correr_regla_coarse` del
#    motor congelado -- método VIEJO corregido, el que realmente se usó en el barrido de 180 reglas,
#    no el N-independiente ingenuo de `correr_regla`) -- método VIEJO vs. NATIVO sobre la MISMA
#    instancia de campo, coarse-graneados a las MISMAS escalas b=1,2,4,8,16.
# ============================================================================================
def correr_regla_comparada(p, N=2000, n_sweeps=14, escalas_b=(1, 2, 4, 8, 16), n_seeds_null_topo=3):
    # -------- método VIEJO: `correr_regla_coarse` congelado, sin tocar (grafo derivado + BFS coarse-graining) --------
    filas_viejo = MOT.correr_regla_coarse(p, N=N, n_sweeps=n_sweeps, escalas_b=escalas_b,
                                           n_seeds_null_topo=n_seeds_null_topo)

    # -------- reconstrucción INDEPENDIENTE del mismo campo real (misma fórmula de semilla que usa
    # correr_regla_coarse internamente: rng = default_rng(seed*5000+N) -- rng nuevo y separado, no
    # comparte estado con el de arriba, así que el array S es IDÉNTICO bit a bit, sin depender del
    # orden de llamadas de MOT.correr_regla_coarse ni de cuántos sorteos consume medir() después) --------
    rng = np.random.default_rng(p["seed"] * 5000 + N)
    sustrato = MOT.construir_A0(N, rng, p)
    sustrato = MOT.dinamica_B0(sustrato, p, rng, n_sweeps, p["eje_C"])
    S = sustrato["S"].copy()
    thr = p["sim_thr_frac"] * p["K"]   # MISMO umbral que usa _grafo_medicion_A0 (comparación justa)

    # NULL nativo: el MISMO campo real, posiciones barajadas UNA vez sobre el anillo fino (destruye
    # el orden espacial, conserva la distribución de valores), luego coarse-graneado a las mismas
    # escalas -- si el orden espacial no importa, coarse-granear ruido puro no debería producir ni
    # correlación sostenida ni dominios grandes a ninguna escala.
    rng_shuf = np.random.default_rng(p["seed"] * 4000 + N)
    S_shuf = rng_shuf.permutation(S)

    filas_nativo = []
    for b in escalas_b:
        S_b, n_cajas = coarsear_campo_ring(S, p["K"], b)
        S_b_null, _ = coarsear_campo_ring(S_shuf, p["K"], b)
        nat_real = metricas_nativas_A0(S_b, p["K"], thr)
        nat_null = metricas_nativas_A0(S_b_null, p["K"], thr)
        filas_nativo.append(dict(
            rule_id=p["rule_id"], escala_b=b, n_cajas=n_cajas,
            corr_len_real=nat_real["corr_len"], corr_len_sat_real=nat_real["corr_len_saturada"],
            giant_nativo_frac_real=nat_real["giant_nativo_frac"],
            giant_nativo_size_real=nat_real["giant_nativo_size"], n_dominios_real=nat_real["n_dominios"],
            corr_len_null=nat_null["corr_len"], corr_len_sat_null=nat_null["corr_len_saturada"],
            giant_nativo_frac_null=nat_null["giant_nativo_frac"],
            giant_nativo_size_null=nat_null["giant_nativo_size"], n_dominios_null=nat_null["n_dominios"],
        ))
    return filas_viejo, filas_nativo


# ============================================================================================
# 3) LOTE COMPLETO: N_REGLAS reglas A0-B0-C0 admitidas (P1-P5 real), viejo vs. nativo por regla
# ============================================================================================
def main(n_reglas=18, seed_base=20260810, N=2000, escalas_b=(1, 2, 4, 8, 16)):
    t_inicio = time.time()
    admitidas, descartadas = GEN.generar_reglas_clase("A0", "B0", "C0", n_reglas=n_reglas, seed_base=seed_base)
    print(f"[gen] admitidas={len(admitidas)} descartadas={len(descartadas)} ({time.time()-t_inicio:.1f}s)")

    raw_viejo, raw_nativo, resumen = [], [], []
    for k, p in enumerate(admitidas):
        t0 = time.time()
        filas_viejo, filas_nativo = correr_regla_comparada(p, N=N, escalas_b=escalas_b)
        raw_viejo += filas_viejo
        raw_nativo += filas_nativo

        r_viejo = CLS.clasificar_regla(filas_viejo)
        # eje de escala nativo = n_cajas (nº de supernodos), MISMO significado que "N" en el clasificador
        # viejo (que ahí también es n_cajas desde que motor.py pasó a correr_regla_coarse) -- comparable
        cajas_list = [f["n_cajas"] for f in filas_nativo]
        corr_slope = _pendiente_loglog(cajas_list, [f["corr_len_real"] for f in filas_nativo])
        dom_slope = _pendiente_loglog(cajas_list, [f["giant_nativo_size_real"] for f in filas_nativo])
        corr_slope_null = _pendiente_loglog(cajas_list, [f["corr_len_null"] for f in filas_nativo])
        dom_slope_null = _pendiente_loglog(cajas_list, [f["giant_nativo_size_null"] for f in filas_nativo])
        f_last = filas_nativo[0]  # b=1 (resolución más fina, n_cajas=N): lectura de estado más informativa

        fila = dict(
            rule_id=p["rule_id"], K=p["K"], J=p["J"], noise=p["noise"], sim_thr_frac=p["sim_thr_frac"],
            clase_vieja=r_viejo["clase"], pendiente_vieja=round(r_viejo["pendiente_real"], 4),
            z_agg_vieja=round(r_viejo["z_agg"], 3), holon_ratio_vieja=round(r_viejo["holon_ratio"], 3),
            corr_len_b1=f_last["corr_len_real"], corr_len_sat_b1=f_last["corr_len_sat_real"],
            giant_frac_b1=round(f_last["giant_nativo_frac_real"], 4),
            n_dominios_b1=f_last["n_dominios_real"],
            corr_slope_nativo=round(corr_slope, 4), dom_slope_nativo=round(dom_slope, 4),
            corr_slope_null=round(corr_slope_null, 4), dom_slope_null=round(dom_slope_null, 4),
            giant_frac_null_b1=round(f_last["giant_nativo_frac_null"], 4),
        )
        resumen.append(fila)
        print(f"  [{k+1}/{len(admitidas)}] {p['rule_id']}: clase_vieja={r_viejo['clase']:<22} "
              f"pend_vieja={r_viejo['pendiente_real']:.3f} | corr_slope={corr_slope:.3f} "
              f"dom_slope={dom_slope:.3f} (null corr={corr_slope_null:.3f} dom={dom_slope_null:.3f}) "
              f"({time.time()-t0:.2f}s)")

    # -------- CSVs de salida --------
    def _guardar(nombre, filas):
        if not filas:
            return
        campos = list(filas[0].keys())
        with open(f"{_HERE}/{nombre}", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=campos)
            w.writeheader()
            w.writerows(filas)
        print(f"  -> {nombre} ({len(filas)} filas)")

    _guardar("cs090_fase5_a0_nativo_viejo_raw.csv", raw_viejo)
    _guardar("cs090_fase5_a0_nativo_nativo_raw.csv", raw_nativo)
    _guardar("cs090_fase5_a0_nativo_resumen.csv", resumen)

    print(f"\nTOTAL: {len(admitidas)} reglas, {time.time()-t_inicio:.1f}s")
    return resumen, raw_viejo, raw_nativo


if __name__ == "__main__":
    main()
