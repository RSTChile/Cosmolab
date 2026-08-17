#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs074C_limite_modelo.py — ¿Da el modelo relación y proceso, pero NO los números físicos?
=========================================================================================

Quién soy / qué hago (código autodescriptivo):
  Implementa PROTOCOLO_cs074C_limite_del_modelo_PREREGISTRO.md (leer primero). NO corre
  física nueva -- es un análisis de los barridos ya hechos (cs074, cs074-A, cs074-B).

Desviación declarada respecto al protocolo (T3, se reporta explícita, no se esconde):
  Al implementar, se encontró que DOS de los tres números físicos de la tabla §2
  (`ratio_pn_congelado` y masa protón/electrón) son CONSTANTES ESTRUCTURALES del motor
  basal (dependen solo de `tasa_expansion` y de las masas fijas del catálogo
  `cs072_modulos/catalogo.py` -- MU=2.3, MD=4.8, m_e=0.51 MeV, ninguna barrida en cs074/A/B)
  -- no varían en NINGUNA corrida del barrido disponible. El método de §3 (distancia mínima
  contra un control de azar RE-MUESTREADO del barrido) es matemáticamente inaplicable a un
  valor sin varianza: no hay barrido del que resamplear. Se reportan estos dos números
  como "no evaluables por este método" -- no se inventa una comparación, no se los excluye
  en silencio tampoco.

  Además, `masa_trio` (usada para el candidato masa protón/electrón) es la suma de masas
  DESNUDAS de quarks (sin energía de ligadura -- la física real dice que ~99% de la masa
  del protón ES energía de ligadura de la fuerza fuerte, que este motor basal no le suma a
  masa_trio). Comparar esa suma desnuda contra 1836 sería comparar dos cantidades de
  naturaleza distinta incluso si hubiera variación que testear. Se documenta, no se corrige
  a mitad de análisis (eso sería tocar la física validada de cs072_modulos, prohibido).

Solo `frac_masa_ligada` (candidato "fracción de materia") tiene variación real y genuina
en el barrido disponible -- es el único de los tres al que el método de §3 se le puede
aplicar tal como está escrito.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cs072_modulos.freeze_out import freeze_out_neutron  # noqa: E402
from cs072_modulos.catalogo import catalogo  # noqa: E402

OUT = HERE / "resultados_cs074C_limite_modelo"
OUT.mkdir(exist_ok=True)

TASA_EXPANSION_USADA = 0.02  # la misma constante fija en cs074/A/B (nunca barrida)


def cargar_pool_frac_masa_ligada():
    """Reúne frac_masa_ligada de TODO el barrido disponible (protocolo §3.1): cs074
    original (280), cs074-A finitas+infinitas (1920+240), cs074-B rama REAL (1980, NO la
    rama barajada -- esa es el control interno de B, no 'lo que el modelo produce')."""
    pool = []
    procedencia = []

    d0 = json.load(open(HERE / "resultados_cs074_energia_holistica" / "cs074_barrido_completo_result.json"))
    for r in d0["filas"]:
        if r.get("ok"):
            pool.append(r["frac_masa_ligada"]); procedencia.append("cs074")

    dA = json.load(open(HERE / "resultados_cs074A_asimetria_techo" / "cs074A_result_FULL.json"))
    for r in dA["filas"]:
        if r.get("ok"):
            pool.append(r["frac_masa_ligada"]); procedencia.append("cs074A_finita")
    for r in dA["control"].values():
        if r.get("ok"):
            pool.append(r["frac_masa_ligada"]); procedencia.append("cs074A_infinita")

    dB = json.load(open(HERE / "resultados_cs074B_fragmentacion_enfriamiento" / "cs074B_result_FULL.json"))
    for f in dB["filas"]:
        if f["real"].get("ok"):
            pool.append(f["real"]["frac_masa_ligada"]); procedencia.append("cs074B_real")

    return np.array(pool), procedencia


def analizar_fraccion_materia(pool, procedencia, valor_real, tolerancia_pp=0.01, n_boot=2000, seed=42):
    """Protocolo §3, refinado (ver desviación declarada arriba del archivo): d_real =
    distancia mínima observada; contexto de azar = fracción del pool entera dentro de la
    tolerancia (comparado contra la densidad esperable por puro volumen si los valores
    fueran uniformes en su propio rango), más un bootstrap de la MEDIANA de sub-muestras
    para dar una noción de dispersión del "qué tan cerca cae un punto típico"."""
    d = np.abs(pool - valor_real)
    i_min = int(np.argmin(d))
    d_real = float(d[i_min])

    dentro_tol = pool[d <= tolerancia_pp]
    frac_dentro_tol = len(dentro_tol) / len(pool)

    rango = float(pool.max() - pool.min())
    densidad_esperada_uniforme = (2 * tolerancia_pp) / rango if rango > 0 else None

    rng = np.random.default_rng(seed)
    boot_d_min = []
    n_sub = min(50, len(pool))  # tamaño de sub-muestra: comparable a una "corrida típica" (no todo el pool)
    for _ in range(n_boot):
        sub = rng.choice(pool, size=n_sub, replace=True)
        boot_d_min.append(float(np.min(np.abs(sub - valor_real))))
    boot_d_min = np.array(boot_d_min)
    z = float((boot_d_min.mean() - d_real) / boot_d_min.std()) if boot_d_min.std() > 0 else None

    return dict(
        valor_real=valor_real, n_pool=len(pool),
        d_real=d_real, valor_mas_cercano=float(pool[i_min]), procedencia_mas_cercano=procedencia[i_min],
        pool_media=float(pool.mean()), pool_std=float(pool.std()), pool_min=float(pool.min()), pool_max=float(pool.max()),
        n_dentro_tolerancia=int(len(dentro_tol)), frac_dentro_tolerancia=frac_dentro_tol,
        densidad_esperada_por_volumen_uniforme=densidad_esperada_uniforme,
        z_bootstrap_dmin_submuestra=z,
        interpretacion=(
            "z alto (>2) Y frac_dentro_tolerancia muy por encima de la densidad uniforme "
            "esperada -> señal real de acercamiento. Si no, es volumen/ruido del propio "
            "barrido (mismo patrón ya visto en E5.3-1 y en cs074/cs074-A: puntos aislados "
            "'cerca' que no resisten este chequeo)."
        ),
    )


def num_constantes_estructurales():
    """Los dos números NO evaluables por §3 (ver desviación declarada) -- se reporta el
    valor emergente igual, solo que sin z-score contra azar (no hay barrido del que salga)."""
    ratio_pn, T_freeze = freeze_out_neutron(TASA_EXPANSION_USADA)
    color, carga, es_anti, es_quark, masa, dens, temp = catalogo(30, 21, 10, 7, amp_rugosidad=1.5)
    masa_u_quark = float(masa[(carga == 2) & es_quark & (~es_anti)][0])
    masa_d_quark = float(masa[(carga == -1) & es_quark & (~es_anti)][0])
    masa_protonoide = 2 * masa_u_quark + masa_d_quark  # uud, suma desnuda (sin ligadura)
    masa_electron = float(masa[(~es_quark) & (~es_anti)][0])
    ratio_masa = masa_protonoide / masa_electron

    return dict(
        ratio_pn=dict(
            valor_emergente=ratio_pn, valor_real=7.1, T_freeze=T_freeze,
            evaluable_por_metodo_C=False,
            razon="constante estructural -- depende solo de tasa_expansion (fija=0.02 en "
                  "todo cs074/A/B, nunca barrida); no hay distribución de la que resamplear",
        ),
        ratio_masa_proton_electron=dict(
            valor_emergente=ratio_masa, valor_real=1836.15,
            evaluable_por_metodo_C=False,
            razon="constante estructural (masas fijas del catálogo) Y de naturaleza "
                  "distinta al valor real: masa_trio es suma de masas DESNUDAS de quarks, "
                  "sin la energía de ligadura de la fuerza fuerte que domina la masa real "
                  "del protón (~99%) -- el motor basal no se la asigna",
        ),
    )


def relaciones_que_si_sostienen():
    """Protocolo §4: se COMPILAN (no se recalculan) los hallazgos ya obtenidos con su
    propio control, de cs074/A/B y de Enfoque 5."""
    return [
        dict(relacion="Contabilidad de energía cierra exacto bajo gravedad pura",
             fuente="cs074", control="gravedad pura, sin expansión/enfriamiento",
             resultado="1.7% de fuga, límite declarado 5%", sostiene=True),
        dict(relacion="El costo de ligadura tiene efecto causal real (no decorativo)",
             fuente="cs074", control="presupuesto finito vs infinito, 280 puntos",
             resultado="29.3% de celdas difieren, exactamente en la zona de reserva escasa", sostiene=True),
        dict(relacion="Muerte térmica != Nada (retiene el presupuesto de energía)",
             fuente="Enfoque 5, E5.5-4", control="NULL/comparación con E=0 por construcción",
             resultado="240/360 celdas: E se queda en ~1.0 exacto al morir térmicamente", sostiene=True),
        dict(relacion="La expansión rescata/retiene más estructura que sin ella",
             fuente="cs074", control="expansion_on=False vs True",
             resultado="88.4% sin expansión vs 60.7% con expansión (compite contra el colapso)", sostiene=True),
        dict(relacion="El techo no-monótono en épsilon es real, no artefacto de presupuesto",
             fuente="cs074-A", control="energía finita vs infinita, barrido 4x más fino",
             resultado="corr(log eps, frac_ligada) casi idéntica con/sin energía (-0.73 / -0.74)", sostiene=True),
        dict(relacion="Gravedad es indispensable para la ligadura (chequeo de admisibilidad)",
             fuente="cs074", control="gravedad_on=False",
             resultado="frac_masa_ligada cae de 60.7% a 2.0%", sostiene=True),
        dict(relacion="El enfriamiento H2 fragmenta la estructura",
             fuente="cs074-B", control="barajado, barrido 10x de intensidad",
             resultado="0/11 niveles con separación del control (z entre -0.11 y -0.14)", sostiene=False),
    ]


def main():
    log = []

    def p(msg):
        print(msg, file=sys.stderr, flush=True)
        log.append(msg)

    p("[C] cargando pool de frac_masa_ligada de todo el barrido disponible...")
    pool, procedencia = cargar_pool_frac_masa_ligada()
    p(f"[C] pool: {len(pool)} valores (cs074+cs074A+cs074B)")

    p("[C] analizando fraccion de materia vs 4.9% y 31.5%...")
    an_49 = analizar_fraccion_materia(pool, procedencia, 0.049)
    an_315 = analizar_fraccion_materia(pool, procedencia, 0.315)
    p(f"[C] 4.9%: d_real={an_49['d_real']:.4f} z_bootstrap={an_49['z_bootstrap_dmin_submuestra']}")
    p(f"[C] 31.5%: d_real={an_315['d_real']:.4f} z_bootstrap={an_315['z_bootstrap_dmin_submuestra']}")

    p("[C] numeros estructuralmente constantes (no evaluables por el metodo C)...")
    constantes = num_constantes_estructurales()
    p(f"[C] ratio_pn emergente = {constantes['ratio_pn']['valor_emergente']:.3f} (real 7.1)")
    p(f"[C] ratio masa p/e emergente = {constantes['ratio_masa_proton_electron']['valor_emergente']:.3f} (real 1836.15)")

    relaciones = relaciones_que_si_sostienen()
    n_si = sum(1 for r in relaciones if r["sostiene"])
    p(f"[C] relaciones que sostienen con control: {n_si}/{len(relaciones)}")

    resultado = dict(
        pool_size=len(pool),
        columna_numeros_fisicos=dict(
            fraccion_materia_4_9pct=an_49,
            fraccion_materia_31_5pct=an_315,
            ratio_pn=constantes["ratio_pn"],
            ratio_masa_proton_electron=constantes["ratio_masa_proton_electron"],
        ),
        columna_relaciones_procesos=relaciones,
        log=log,
    )
    out_json = OUT / "cs074C_result.json"
    out_json.write_text(json.dumps(resultado, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    p(f"[archivo] {out_json}")


if __name__ == "__main__":
    main()
