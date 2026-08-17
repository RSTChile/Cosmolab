"""
CS087 — LAPLACIANO DE HODGE L_1 SOBRE ARISTAS: ¿el componente armónico explica la Pared de Fase IV?
====================================================================================================
QUIÉN SOY: sigo directo a `FASE4_orden_superior_resultado_CS.md` (cs082) y `FASE4_robustecido_CS.md`
(cs083). Esos dos midieron la HOLONOMÍA de triángulo (suma de 3 aristas de borde, mod K, centrada) en
4 sustratos relacionales y encontraron que sólo el sustrato 4 (2-complejo con feedback cara→arista)
se separaba de NULL con solidez (holonomía ~5x menor). Esa holonomía vive en las ARISTAS/CICLOS, no en
los nodos — el candidato natural para explicarla NO es el laplaciano de grafo de siempre (L_0=D-A, que
sólo ve nodos, ya usado en cs084/cs085/cs086) sino el laplaciano de Hodge de ARISTAS, L_1, construido
con los operadores de borde simpliciales:

    ∂_1 : aristas → nodos   (matriz de incidencia arista-nodo con signo, la de siempre del grafo)
    ∂_2 : caras   → aristas (matriz de incidencia cara-arista con signo, según la orientación del
                              borde del triángulo: para una cara ordenada (i,j,k) con i<j<k, su borde
                              orientado es +arista(j,k) − arista(i,k) + arista(i,j) — es la fórmula
                              estándar de complejos simpliciales orientados, la que garantiza que
                              ∂_1·∂_2 = 0, condición necesaria para que la teoría de Hodge tenga sentido)

    L_1 = ∂_1ᵀ∂_1 + ∂_2∂_2ᵀ          (down-Laplacian + up-Laplacian, ambos actuando en el espacio de
                                       aristas — shape |E|×|E|)

Por el teorema de descomposición de Hodge discreta, el espacio de aristas se parte en 3 subespacios
ORTOGONALES: gradientes (im ∂_1ᵀ, "explicado por un potencial de nodo"), rotores/curl (im ∂_2, "el
borde de alguna cara — cierra localmente porque ALGUNA cara lo explica") y ARMÓNICO (ker L_1 — ciclos
que NO son borde de ninguna cara presente en el complejo; su dimensión es el número de Betti b_1, un
INVARIANTE TOPOLÓGICO puro, independiente de los valores dinámicos que corran sobre las aristas).

LA PREGUNTA CENTRAL DE ESTA TAREA (mandato de Alexis): ¿el espectro de L_1 —en particular su
componente armónico (autovalores ~0)— explica o correlaciona con la holonomía que Fase IV ya midió, y
en particular por qué sólo el sustrato 4 (feedback activo) se separaba de NULL mientras el 3 (medición
pasiva) no?

────────────────────────────────────────────────────────────────────────────────────────────────────
HALLAZGO METODOLÓGICO QUE HAY QUE DECLARAR ANTES DE LEER NÚMEROS (no es un resultado escondido, es
la primera cosa que hay que entender): los sustratos 3 y 4 de `cs082_fase4_4sustratos.py` se construyen
sobre EXACTAMENTE la misma base combinatoria — mismo `construir_base(seed)`, mismas aristas, mismos
triángulos. Sólo cambia la DINÁMICA que corre sobre esa base (3 = sin feedback, 4 = con feedback
cara→arista). Como L_1 depende SÓLO de qué aristas y qué caras existen (∂_1, ∂_2 son matrices de
incidencia 0/±1, no dependen de los valores Z_K que trae cada corrida), **el espectro completo de L_1,
y en particular la dimensión del subespacio armónico (b_1), es IDÉNTICO entre sustrato 3 y sustrato 4
para la misma semilla.** Esto no es un bug — es la primera pieza de evidencia: la topología (el
"tablero") no cambia entre pasivo y activo, sólo cambia dónde CAE el campo (los valores dinámicos)
sobre ese tablero fijo. Por eso este script mide, además del espectro (compartido), la PROYECCIÓN del
campo real de cada sustrato sobre los 3 subespacios de Hodge — eso sí difiere entre 3 y 4, y es donde
hay que buscar la explicación.

Sustratos 1 (grafo diádico) y 2 (hipergrafo): el enunciado de la tarea pide no forzar un L_1 con caras
falsas para ellos. Sustrato 1 SÍ tiene aristas nativas (mismas que 3/4) pero NINGUNA cara activa en su
propia definición (el objeto-relación es sólo la arista) — se reporta un caso DEGENERADO explícito:
L_1_deg = ∂_1ᵀ∂_1 solamente (sin término de caras), dejando claro que no es equivalente al L_1 completo
de 3/4 — sirve sólo para mostrar cuánto CRECE el espacio "armónico" cuando no hay ninguna cara que
explique los ciclos (todo ciclo de 3 aparece como "no explicado"). Sustrato 2 (hipergrafo) NO tiene
aristas como objeto-relación nativo (son proyecciones de hiperaristas, y ni siquiera cubren todas las
aristas base) — se documenta por qué se lo deja fuera de esta batería, sin forzar una construcción
arbitraria.

Código nuevo, autodescriptivo, numpy-only. Importa (no modifica) `cs082_fase4_4sustratos.py` para
reusar el generador de base, las corridas de sustrato 3/4, la holonomía y los controles NULL/SHUFFLED.
No declara cierre — reporta números; el veredicto final es de Alexis.
"""
from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np

_AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _AQUI)
import cs082_fase4_4sustratos as cs082  # noqa: E402  (import tras sys.path, deliberado)

# ============================ CONFIG ============================
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8]     # 8 semillas (pedido: 5-10); 1-5 coinciden con cs082/cs083
UMBRAL_ARMONICO = 1e-6               # autovalor < esto se cuenta como "armónico" (~0)
OUT_CSV = os.path.join(_AQUI, "cs087_hodge_fase4.csv")
OUT_ESPECTROS = os.path.join(_AQUI, "cs087_espectros_L1.npz")
# ==================================================================


def construir_operadores_borde(edges, triangles, n_nodos):
    """∂_1 (n_nodos × n_aristas) y ∂_2 (n_aristas × n_caras), signo estándar de complejo simplicial
    orientado. edges: lista de (i,j) con i<j (orden = orden nativo de cs082.construir_base). triangles:
    lista de (i,j,k) con i<j<k (idem). El orden de columnas de ambas matrices sigue el orden de estas
    listas — así cualquier vector de valores {E[e] for e in edges} queda alineado a las columnas de ∂_1
    y ∂_2 sin reordenar nada.
    Convención (estándar, la que garantiza ∂_1·∂_2 = 0): arista (i,j) con i<j orientada i→j da +1 en el
    nodo j y −1 en el nodo i. Cara (i,j,k) con i<j<k tiene borde orientado +arista(j,k) − arista(i,k) +
    arista(i,j) (la fórmula usual de un 2-símplice ordenado)."""
    idx_edge = {e: c for c, e in enumerate(edges)}
    n_e, n_f = len(edges), len(triangles)

    d1 = np.zeros((n_nodos, n_e))
    for c, (i, j) in enumerate(edges):
        d1[j, c] += 1.0
        d1[i, c] += -1.0

    d2 = np.zeros((n_e, n_f))
    for c, (i, j, k) in enumerate(triangles):
        d2[idx_edge[(j, k)], c] += 1.0
        d2[idx_edge[(i, k)], c] += -1.0
        d2[idx_edge[(i, j)], c] += 1.0

    return d1, d2, idx_edge


def verificar_dd0(d1, d2, tol=1e-8):
    """Chequeo de sanidad matemática: ∂_1·∂_2 debe ser (numéricamente) la matriz cero — es la condición
    'borde-de-un-borde-es-cero' que hace válida toda la teoría de Hodge de abajo. Si esto falla, algo
    está mal en la construcción de signos."""
    resid = d1 @ d2
    return float(np.max(np.abs(resid))) if resid.size else 0.0


def _recentrar_circular(vals, K):
    """Los valores viven en Z_K (círculo), pero ∂_1/∂_2/L_1 son operadores LINEALES sobre números
    reales — no 'saben' de circularidad. Si el arco 0↔K queda en medio de un cúmulo de valores
    parecidos, la resta ve una diferencia gigante (ej. 0.1 y 5.9 en K=6 son angularmente casi iguales
    pero numéricamente están a 5.8 de distancia). Mitigación simple y honesta (no elimina el problema
    del todo, sólo lo reduce): se calcula la media circular de la serie y se rota todo el círculo para
    que esa media caiga en K/2, lejos de la costura 0/K. Se aplica por separado a REAL/NULL/SHUFFLED de
    cada corrida (cada serie tiene su propia 'costura' óptima)."""
    ang = 2 * np.pi * np.asarray(vals) / K
    z = np.mean(np.exp(1j * ang))
    centro = (np.angle(z) / (2 * np.pi) * K) % K
    shift = (K / 2 - centro) % K
    return (np.asarray(vals) + shift) % K


def proyeccion_hodge(vals_vec, eigvecs, eigvals, umbral):
    """Descompone vals_vec (ya centrado, real) en componente armónico (autovectores con autovalor casi
    0) vs el resto (gradiente+rotor juntos, autovalor>0). Devuelve la fracción de energía (norma al
    cuadrado) que cae en el subespacio armónico."""
    norm_total = float(np.dot(vals_vec, vals_vec))
    if norm_total < 1e-12:
        return 0.0, 0.0
    mask_arm = eigvals < umbral
    if not mask_arm.any():
        return 0.0, norm_total
    coef = eigvecs[:, mask_arm].T @ vals_vec
    norm_arm = float(np.dot(coef, coef))
    return norm_arm / norm_total, norm_total


def energia_cuadratica(vals_vec, M):
    """v^T M v — la 'forma cuadrática' de un operador PSD evaluada en el campo real. Para M=L_1_up =
    ∂_2∂_2ᵀ, esto es exactamente Σ_caras (borde_orientado_de_la_cara · campo)^2 — el análogo, en la
    convención de signos ESTÁNDAR, de la holonomía al cuadrado que cs082/cs083 ya miden (con su propia
    convención de signos, ver nota más abajo)."""
    return float(vals_vec @ (M @ vals_vec))


def main():
    t_inicio = time.time()
    print("CS087 — Laplaciano de Hodge L_1 (aristas) sobre los sustratos 3 y 4 de Fase IV")
    print("=" * 100)
    print(f"Semillas: {SEEDS} · umbral armónico |λ|<{UMBRAL_ARMONICO} · K={cs082.K}\n")

    filas = []              # una fila por (seed, sustrato, brazo)
    espectros_guardar = {}  # seed -> autovalores de L_1 (compartido entre sustrato 3 y 4)

    for seed in SEEDS:
        adj, edges, triangles = cs082.construir_base(seed)
        n_e, n_t = len(edges), len(triangles)

        d1, d2, idx_edge = construir_operadores_borde(edges, triangles, cs082.N)
        resid_dd0 = verificar_dd0(d1, d2)

        L1_down = d1.T @ d1
        L1_up = d2 @ d2.T
        L1 = L1_down + L1_up
        eigvals, eigvecs = np.linalg.eigh(L1)
        b1 = int(np.sum(eigvals < UMBRAL_ARMONICO))
        lambda_min_pos = float(eigvals[eigvals >= UMBRAL_ARMONICO][0]) if b1 < n_e else float("nan")
        lambda_max = float(eigvals[-1])
        espectros_guardar[f"seed{seed}_eigvals_L1"] = eigvals

        # --- degenerado sustrato 1: L1_deg = d1^T d1 solamente (sin término de caras) ---
        eigvals_deg = np.linalg.eigvalsh(L1_down)
        b1_deg = int(np.sum(eigvals_deg < UMBRAL_ARMONICO))
        espectros_guardar[f"seed{seed}_eigvals_L1down_deg"] = eigvals_deg

        print(f"seed={seed}  |E|={n_e} |T|={n_t}  max|∂1·∂2|={resid_dd0:.2e} (debe ser ~0)")
        print(f"  L_1 completo (sustrato 3 y 4, TOPOLOGÍA COMPARTIDA):"
              f" b_1(armónico)={b1}  λ_min_pos={lambda_min_pos:.4f}  λ_max={lambda_max:.3f}")
        print(f"  L_1_deg=∂1ᵀ∂1 solo (sustrato 1, sin caras): "
              f"b_1_deg={b1_deg}  (espacio de ciclos COMPLETO, sin filtrar por caras)")

        # --------- sustrato 3 (pasivo) y sustrato 4 (feedback) sobre la MISMA base/L1 ---------
        for nombre_sustrato, fn in [("3_simplicial", cs082.correr_sustrato_3_simplicial),
                                     ("4_2complejo_feedback", cs082.correr_sustrato_4_2complejo)]:
            E_real, dof, n_sweeps, dt = fn(adj, edges, triangles, seed)
            E_null = cs082.null_de(E_real, seed)
            E_shuf = cs082.shuffled_de(E_real, seed)

            for brazo, E_brazo in [("REAL", E_real), ("NULL", E_null), ("SHUFFLED", E_shuf)]:
                vals_raw = np.array([E_brazo[e] for e in edges])
                vals_rc = _recentrar_circular(vals_raw, cs082.K)
                vals_c = vals_rc - vals_rc.mean()   # centrado adicional (media aritmética a cero)

                frac_arm, norm_total = proyeccion_hodge(vals_c, eigvecs, eigvals, UMBRAL_ARMONICO)
                e_up = energia_cuadratica(vals_c, L1_up) / max(n_t, 1)     # "curl energy" por cara
                e_down = energia_cuadratica(vals_c, L1_down) / max(n_e, 1)  # "grad energy" por arista

                # holonomía YA MEDIDA por Fase IV (convención propia de cs082: suma de 3 valores de
                # arista, sin alternar signo, mod K, centrada) — se reusa tal cual, sin tocar el import.
                h_modK = cs082._holonomia_triangulos(E_brazo, triangles)  # array de |holonomía| por cara

                # curl LINEAL con signo estándar (∂_2ᵀ · campo centrado), SIN mod — es la cantidad de
                # la que L_1_up es la forma cuadrática. Se compara con h_modK (abs) triángulo a
                # triángulo: son convenciones de signo DISTINTAS (cs082 suma sin alternar signo; el
                # ∂_2 estándar alterna, condición necesaria para ∂1·∂2=0) — no se espera que coincidan
                # en valor, sólo se reporta si se mueven juntas (correlación).
                curl_lineal = d2.T @ vals_c
                if len(h_modK) == len(curl_lineal) and len(h_modK) > 1 and np.std(h_modK) > 1e-9 \
                        and np.std(np.abs(curl_lineal)) > 1e-9:
                    corr_curl_holon = float(np.corrcoef(np.abs(curl_lineal), h_modK)[0, 1])
                else:
                    corr_curl_holon = float("nan")

                filas.append(dict(
                    seed=seed, sustrato=nombre_sustrato, brazo=brazo,
                    n_edges=n_e, n_tri=n_t, b1_armonico=b1, lambda_min_pos=lambda_min_pos,
                    lambda_max=lambda_max,
                    frac_armonica=frac_arm, curl_energy_up=e_up, grad_energy_down=e_down,
                    holon_modK_media=float(h_modK.mean()),
                    corr_curl_lineal_vs_holon_modK=corr_curl_holon,
                ))

        print()

    # ---------------- guardar CSV crudo ----------------
    campos = list(filas[0].keys())
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)
    np.savez(OUT_ESPECTROS, **espectros_guardar)

    # ---------------- tabla resumen: fracción armónica y curl-energy por sustrato×brazo ----------------
    print("=" * 100)
    print("RESUMEN (promedio ± DE sobre 8 semillas) — fracción armónica y energía de curl (up-Laplacian)")
    print("=" * 100)
    print(f"  {'sustrato':<22} {'brazo':<9} {'frac_armónica':>16} {'curl_energy/cara':>18} "
          f"{'grad_energy/arista':>19} {'holon_modK(cs082)':>19}")
    for sustrato in ["3_simplicial", "4_2complejo_feedback"]:
        for brazo in ["REAL", "NULL", "SHUFFLED"]:
            sub = [f for f in filas if f["sustrato"] == sustrato and f["brazo"] == brazo]
            fa = np.array([f["frac_armonica"] for f in sub])
            cu = np.array([f["curl_energy_up"] for f in sub])
            gd = np.array([f["grad_energy_down"] for f in sub])
            hm = np.array([f["holon_modK_media"] for f in sub])
            print(f"  {sustrato:<22} {brazo:<9} {fa.mean():>8.4f}±{fa.std():<6.4f} "
                  f"{cu.mean():>10.3f}±{cu.std():<6.3f} {gd.mean():>11.3f}±{gd.std():<6.3f} "
                  f"{hm.mean():>11.3f}±{hm.std():<6.3f}")
    print()

    print("Correlación (por semilla) entre |curl lineal (∂_2ᵀ, signo estándar)| y holonomía mod-K "
          "(convención cs082, todas las caras, promedio sobre semillas):")
    for sustrato in ["3_simplicial", "4_2complejo_feedback"]:
        for brazo in ["REAL", "NULL", "SHUFFLED"]:
            sub = [f for f in filas if f["sustrato"] == sustrato and f["brazo"] == brazo]
            corrs = np.array([f["corr_curl_lineal_vs_holon_modK"] for f in sub])
            corrs = corrs[~np.isnan(corrs)]
            if len(corrs):
                print(f"  {sustrato:<22} {brazo:<9} r_prom={corrs.mean():+.3f}  (n_semillas={len(corrs)})")

    print(f"\nb_1 armónico (topológico, COMPARTIDO entre sustrato 3 y 4 en cada semilla):")
    for seed in SEEDS:
        b1_val = [f["b1_armonico"] for f in filas if f["seed"] == seed][0]
        print(f"  seed={seed}: b_1={b1_val}  de |E|={[f['n_edges'] for f in filas if f['seed']==seed][0]}")

    print(f"\nTiempo total: {time.time()-t_inicio:.1f}s. CSV: {OUT_CSV}  Espectros: {OUT_ESPECTROS}")
    print("Fin de la batería. Sin cierre ni veredicto — números para CS087_hodge_fase4_resultado_CS.md.")


if __name__ == "__main__":
    main()
