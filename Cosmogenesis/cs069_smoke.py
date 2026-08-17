"""
CS069 SMOKE — las 3 anclas obligatorias antes de la tanda blindada
==============================================================================
Ruling de CS (DISENO_CS069_frente_cuantico_CS.md): "SMOKE (antes de lanzar la tanda) — 3 anclas que deben
cumplirse o NO se corre":
1. NULL_CLÁSICO (φ≡0) reproduce el Mundo B de CS068 (diám residual ~6-7.5, pendiente ~0).
2. En un grafo-juguete con tejido métrico CONOCIDO + atajos inyectados, la regla de fase ciega SÍ decohere
   los atajos (verdad de fondo, sin clasificador).
3. NULL_FASE_TOPO y NULL_FASE_AZAR dan π que ESTALLA (control de que los nulls no encienden geometría solos).

Codea/ejecuta: CC. Diseño/ruling: CS.
"""
from __future__ import annotations
import os, sys, time
import numpy as np

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)
import cs069_quantum_graph as Q
import cs068_inflacion_estirar_enfriar as E
import cs067_habitacion_completa as H
import cs068_paso1_sintetico as S1

RNG = np.random.default_rng


# ============================ ANCLA 1 — NULL_CLASICO reproduce Mundo B de CS068 ============================
def ancla1_null_clasico(Ns=(900, 1500, 2500), n_seeds=2):
    print("\n" + "=" * 100, flush=True)
    print("ANCLA 1 — NULL_CLASICO (φ≡0): ¿reproduce el Mundo B de CS068 (diám_q plano, pendiente~0)?",
          flush=True)
    print("Nota: D_q (log-amplitud sobre integral de camino) es una MÉTRICA DISTINTA a la de CS068 (BFS", flush=True)
    print("sobre tejido residual) -- se compara la CONCLUSIÓN cualitativa (pendiente baja), no el valor.",
          flush=True)
    print("=" * 100, flush=True)
    medias = []
    for i, N in enumerate(Ns):
        diams = []
        for s in range(n_seeds):
            seed = 69000 + 97 * i + 13 * s
            adj = E._sustrato(N, seed)
            Dq = Q.brazo_null_clasico(adj, N, RNG(seed + 1))
            diams.append(Q.diam_q_robusto(Dq, N, RNG(seed + 2)))
        m = float(np.mean(diams))
        medias.append(m)
        print(f"  N={N}: diam_q por semilla={[round(d,2) for d in diams]}  media={m:.2f}", flush=True)
    pendiente, _ = Q._pendiente_loglog(Ns, medias)
    print(f"\n  pendiente log-log(diam_q vs N) = {pendiente:.3f}", flush=True)
    ok = pendiente < 0.3
    print(f"  {'PASA' if ok else 'FALLA'}: pendiente {'<' if ok else '>='} 0.3 (mismo umbral cualitativo que CS068)",
          flush=True)
    return ok


def _reticula_2d_8vecinos(side):
    """CORRECCIÓN encontrada al smoke-testear: S1._reticula_2d (4-vecinos, Von Neumann) NO TIENE TRIÁNGULOS
    -- soporte=0 en TODA la retícula, local o atajo por igual. ρ_ij/costo_ij de CS069 vienen de
    H._pesos_correlacion (ingrediente 14 = soporte por vecinos comunes) -- en una retícula sin triángulos
    ese w_ij es CIEGO al local/atajo, ANTES de que la dinámica de fase entre en juego (GIGO: la fase no
    puede decoherer una señal que su propio costo de entrada ya no lleva). Es la MISMA lección de CS068
    (soporte=clustering, y una retícula 2D pura reprueba el CM-null por no tener triángulos). Fix: retícula
    de 8-vecinos (Moore, incluye diagonales) SÍ tiene triángulos locales (dos vecinos ortogonales de un nodo
    más su diagonal común forman un triángulo) -- soporte>0 real en el tejido local, dando a w_ij algo que
    medir, mientras los atajos inyectados (pares lejanos al azar) siguen con soporte~0."""
    N = side * side
    adj_local = [set() for _ in range(N)]

    def idx(r, c):
        return r * side + c

    for r in range(side):
        for c in range(side):
            i = idx(r, c)
            for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                r2, c2 = r + dr, c + dc
                if 0 <= r2 < side and 0 <= c2 < side:
                    j = idx(r2, c2)
                    adj_local[i].add(j); adj_local[j].add(i)
    return adj_local, N


# ============================ ANCLA 2 — decoherencia de atajos en juguete con verdad de fondo ============================
def ancla2_decoherencia_juguete(N_objetivo=400, seed=69100, frac_atajos=0.15):
    print("\n" + "=" * 100, flush=True)
    print("ANCLA 2 — juguete (retícula 2D 8-vecinos + atajos inyectados, verdad de fondo): ¿la fase ciega", flush=True)
    print("decohere los atajos MÁS que el tejido local, sin que la dinámica sepa cuál es cuál?", flush=True)
    print("=" * 100, flush=True)
    side = int(round(N_objetivo ** 0.5))
    adj_local, N = _reticula_2d_8vecinos(side)
    n_local_edges = sum(len(a) for a in adj_local) // 2
    n_atajos = max(1, int(round(frac_atajos * n_local_edges)))
    atajos = S1._inyecta_atajos(adj_local, N, n_atajos, RNG(seed + 1))
    adj = S1._adj_con_atajos(adj_local, atajos)
    locales = [(i, j) for i in range(N) for j in adj_local[i] if i < j]

    rng = RNG(seed + 2)
    edges, rho, _costo = Q._rho_y_costo(adj, N, rng)
    print(f"  w_ij (rho) media: local={np.mean([rho[e] for e in locales]):.4f}  "
          f"atajo={np.mean([rho[e] for e in atajos]):.4f}  (debe discriminar -- si no, es GIGO antes de la fase)",
          flush=True)
    phi_evolved = Q._evoluciona_fase(edges, rho, N, RNG(seed + 3))

    # Chequeo PRIMARIO (el que valida CS): frustración |φ_ij|=|θ_i-θ_j| directa, local vs atajo, + AUC.
    frust_local = np.array([abs(phi_evolved[e]) for e in locales])
    frust_atajo = np.array([abs(phi_evolved[e]) for e in atajos])
    todas = np.concatenate([frust_local, frust_atajo])
    etiquetas = np.concatenate([np.zeros(len(frust_local)), np.ones(len(frust_atajo))])
    orden = np.argsort(todas)
    rangos = np.empty_like(orden, dtype=float); rangos[orden] = np.arange(1, len(todas) + 1)
    auc = (rangos[etiquetas == 1].sum() - len(frust_atajo) * (len(frust_atajo) + 1) / 2) / \
          (len(frust_atajo) * len(frust_local))
    print(f"  frustración |φ| directa: local media={frust_local.mean():.4f}  atajo media={frust_atajo.mean():.4f}",
          flush=True)
    print(f"  AUC (frustración atajo > local) = {auc:.3f}  (0.5=azar, 1.0=separación perfecta)", flush=True)
    ok_frustracion = auc > 0.65

    phi_azar = {e: float(RNG(seed + 4).uniform(0, 2 * np.pi)) for e in edges}
    Dq_completo = Q._K_y_Dq(N, edges, rho, phi_evolved)
    Dq_azar = Q._K_y_Dq(N, edges, rho, phi_azar)

    def _delta(pares, A, B):
        ds = []
        for (i, j) in pares:
            a, b = A[i, j], B[i, j]
            if np.isfinite(a) and np.isfinite(b):
                ds.append(a - b)
        return np.array(ds, float)

    # CORRECCIÓN (encontrada al smoke-testear): comparar contra φ≡0 confunde la MULTIPLICIDAD de caminos
    # (los locales, con más caminos alternos, decoherencian más que cualquier NULL solo por combinatoria --
    # φ≡0 es el máximo teórico de coherencia para CUALQUIER topología, así que Δ contra ese piso mide
    # "cuántos caminos tiene el par", no "qué tan buena es la dinámica"). El baseline correcto, que SÍ
    # controla la multiplicidad de caminos (misma topología, mismo nº de caminos): fase EVOLUCIONADA vs
    # fase AL AZAR -- exactamente el contraste que hace NULL_FASE_AZAR de verdad en la tanda.
    d_atajo = _delta(atajos, Dq_completo, Dq_azar)
    d_local = _delta(locales, Dq_completo, Dq_azar)
    print(f"  n_atajos={len(atajos)} (validos={len(d_atajo)})  n_locales={n_local_edges} (validos={len(d_local)})",
          flush=True)
    print(f"  Δ_Dq (evolucionada-azar) atajos: media={d_atajo.mean():.4f} std={d_atajo.std():.4f}", flush=True)
    print(f"  Δ_Dq (evolucionada-azar) locales: media={d_local.mean():.4f} std={d_local.std():.4f}", flush=True)
    ok_dq = d_atajo.mean() > d_local.mean()
    print(f"  Δ_Dq: {'consistente' if ok_dq else 'INCONSISTENTE'} con frustración -- atajo {'>' if ok_dq else '<='} local",
          flush=True)

    print(f"\n  CRITERIO PRIMARIO (el que valida CS): AUC(frustración)={auc:.3f} {'>' if ok_frustracion else '<='} 0.65",
          flush=True)
    print(f"  {'PASA' if ok_frustracion else 'FALLA'}: la frustración |φ| directa entre extremos "
          f"{'SÍ' if ok_frustracion else 'NO'} separa atajo de local, sin haber etiquetado nada.", flush=True)
    return ok_frustracion


# ============================ ANCLA 3 — los NULLs no encienden geometría solos (π estalla) ============================
def ancla3_nulls_no_encienden(N=900, n_seeds=3):
    print("\n" + "=" * 100, flush=True)
    print("ANCLA 3 — NULL_FASE_TOPO y NULL_FASE_AZAR: ¿π sigue ESTALLANDO (CV alto, no converge)?", flush=True)
    print("=" * 100, flush=True)
    resultados = {}
    for nombre, fn in [("null_fase_topo", Q.brazo_null_fase_topo), ("null_fase_azar", Q.brazo_null_fase_azar)]:
        cvs = []
        for s in range(n_seeds):
            seed = 69200 + 97 * s
            adj = E._sustrato(N, seed)
            Dq = fn(adj, N, RNG(seed + 1))
            _media, cv = Q.cedazo_pi(Dq, N, RNG(seed + 2))
            cvs.append(cv)
        resultados[nombre] = cvs
        print(f"  {nombre}: CV por semilla={[round(c,3) for c in cvs]}  media={np.mean(cvs):.3f}", flush=True)
    ok = all(np.mean(v) > 0.30 for v in resultados.values())  # CV alto = sigue estallando, no converge
    print(f"\n  {'PASA' if ok else 'FALLA'}: CV medio > 0.30 en ambos nulls (no convergen espontáneamente)",
          flush=True)
    return ok


def main():
    t0 = time.time()
    ok1 = ancla1_null_clasico()
    ok2 = ancla2_decoherencia_juguete()
    ok3 = ancla3_nulls_no_encienden()
    print("\n" + "=" * 100, flush=True)
    print(f"RESUMEN: ancla1(NULL_CLASICO~CS068)={'PASA' if ok1 else 'FALLA'}  "
          f"ancla2(decoherencia atajos)={'PASA' if ok2 else 'FALLA'}  "
          f"ancla3(nulls no encienden)={'PASA' if ok3 else 'FALLA'}", flush=True)
    print(f"tiempo total: {(time.time()-t0)/60:.2f} min", flush=True)
    if ok1 and ok2 and ok3:
        print("\nLAS 3 ANCLAS PASAN. Autorizado a correr la tanda blindada completa.", flush=True)
    else:
        print("\nAL MENOS UN ANCLA FALLA. NO correr la tanda -- reportar a CS antes de seguir.", flush=True)


if __name__ == "__main__":
    main()
