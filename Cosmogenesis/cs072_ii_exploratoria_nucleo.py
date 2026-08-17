"""
CS072-II -- EXPLORATORIA de NÚCLEO-II: barrer la expansión continua (p_t) x n_focos, fijar las anclas
P-COHESIÓN / P-BORDE / P-DISOLUCIÓN por PERSISTENCIA-DE-CONECTIVIDAD a través de la filtración (nunca por
el umbral que maximiza β -- G-NO-ELEGIR-PODA). Sigue siendo II-DET (determinista, cero RNG en la dinámica;
el RNG que aparece aquí es SÓLO del lector -- landmarks para δ-Gromov y semillas de los jueces BFS, meta-
herramienta de medición, no parte del motor).

Fuente: ADJUDICACION_CS072_II_puerta_s_completa_CS.md ("LUZ VERDE: ABRIR LA EXPLORATORIA NÚCLEO-II").
Guardianes: (1) barrer p_exp junto con n_focos, incluido n_focos=1 como sub-control SIN gravedad (la
gravedad se anula idénticamente con 1 solo foco -- adjudicado). (2) anclas por persistencia-de-conectividad
(intervalo NO NULO de niveles), nunca por el punto que maximiza β. (3) reportar curvas COMPLETAS de
filtración + β + δ + jueces continuos por régimen, no sólo el punto de onset.

Codea/ejecuta: CC. Diseño/ruling: CS + Codex.
"""
from __future__ import annotations
import sys, time, json
import numpy as np

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)
import cs072_ii_nucleo as II
import cs072_ii_filtracion as F

RNG = np.random.default_rng
DELTA = 1e-4
PASOS = 80
N_PRIMARIO = 200
NS_BETA = [100, 200, 400, 800]
SEED_BASE = 51772

P_EXP_TASAS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.35, 0.5, 1.0]
FOCOS = [1, 2, 5, 20]   # 1 = sub-control SIN gravedad (adjudicado)


def _corre_y_lee(N, n_focos, p_exp, seed, n_checkpoints=12):
    """Corre II-DET completo y aplica el lector de filtración COMPLETO (curva entera, no sólo onset)."""
    T, W = II.estado_inicial(N, n_focos, DELTA)
    for _ in range(PASOS):
        T, W, _ = II.paso_ii_det(T, W, p_exp=p_exp)
    rng_lector = RNG(seed)
    curva, adj = F.curva_filtracion(W, N, n_checkpoints_judges=n_checkpoints, rng_judges=rng_lector)
    jueces = F.jueces_continuos_sin_umbral(W, N)
    tramos = F.persistencia_conectividad(curva, frac_umbral=0.9, min_ancho_frac=0.02)
    return dict(T_final=T, W_final=W, curva=curva, jueces=jueces, tramos=tramos)


def _resumen_regimen(tramos, curva, jueces):
    """Clasifica el régimen leyendo SÓLO persistencia-de-conectividad (nunca el punto que maximiza β):
    - DISOLUCION: no hay ningún tramo con frac_gigante>=0.9 de ancho no nulo (nunca se sostiene conectado).
    - COHESION (hub/orden): el tramo persistente arranca casi en frac_pares=0 (todo pegado casi de entrada,
      poca discriminación posible antes de llegar a componente gigante).
    - BORDE (posible banda): hay un tramo persistente pero arranca en una fraccion intermedia -- hay
      estructura ANTES de alcanzar el gigante que podria ser informativa."""
    if not tramos:
        return "DISOLUCION", None
    inicio_tramo = tramos[0][0]
    if inicio_tramo < 0.05:
        return "COHESION", inicio_tramo
    return "BORDE", inicio_tramo


def main():
    t0 = time.time()
    print("=" * 100, flush=True)
    print("CS072-II -- EXPLORATORIA NUCLEO-II: barrido p_exp x n_focos (N=%d, II-DET)" % N_PRIMARIO, flush=True)
    print("=" * 100, flush=True)

    resultados = []
    for n_focos in FOCOS:
        print(f"\n--- n_focos={n_focos} {'(SUB-CONTROL sin gravedad)' if n_focos == 1 else ''} ---", flush=True)
        for p_exp in P_EXP_TASAS:
            seed = SEED_BASE + n_focos * 1000 + int(round(p_exp * 10000))
            r = _corre_y_lee(N_PRIMARIO, n_focos, p_exp, seed)
            regimen, inicio = _resumen_regimen(r["tramos"], r["curva"], r["jueces"])
            ultimo = r["curva"][-1]
            item = dict(n_focos=n_focos, p_exp=p_exp, regimen=regimen, inicio_tramo=inicio,
                        tramos=r["tramos"], max_h=r["jueces"]["max_h"],
                        k_eff_medio=r["jueces"]["k_eff_medio"], log_dispersion=r["jueces"]["log_dispersion"],
                        frac_gigante_final=ultimo["frac_gigante"], diam_final=ultimo.get("diam"))
            resultados.append(item)
            print(f"  p_exp={p_exp:5.3f}  regimen={regimen:10s}  inicio_tramo={inicio}  "
                  f"tramos={r['tramos']}  max_h={r['jueces']['max_h']:.4f}  "
                  f"k_eff={r['jueces']['k_eff_medio']:.1f}  logdisp={r['jueces']['log_dispersion']:.3f}  "
                  f"(t={(time.time()-t0)/60:.1f}min)", flush=True)

    with open("cs072_ii_exploratoria_barrido.json", "w") as f:
        json.dump(resultados, f, indent=2, default=str)

    print("\n" + "=" * 100, flush=True)
    print("ANCLAS CANDIDATAS (por n_focos, vía persistencia-de-conectividad, N=%d)" % N_PRIMARIO, flush=True)
    print("=" * 100, flush=True)
    anclas = {}
    for n_focos in FOCOS:
        fila = [r for r in resultados if r["n_focos"] == n_focos]
        cohesion = [r["p_exp"] for r in fila if r["regimen"] == "COHESION"]
        borde = [r["p_exp"] for r in fila if r["regimen"] == "BORDE"]
        disolucion = [r["p_exp"] for r in fila if r["regimen"] == "DISOLUCION"]
        anclas[n_focos] = dict(cohesion=cohesion, borde=borde, disolucion=disolucion)
        print(f"n_focos={n_focos}: COHESION(p_exp)={cohesion}  BORDE(p_exp)={borde}  DISOLUCION(p_exp)={disolucion}",
              flush=True)

    print(f"\ntiempo barrido grueso: {(time.time()-t0)/60:.2f} min", flush=True)
    return resultados, anclas


if __name__ == "__main__":
    main()
