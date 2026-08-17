"""
CS068 Paso 2b — diagnóstico barato: ¿el tejido residual (blob real menos atajos) es MÉTRICO?
==============================================================================
Ruling de CS (ADJUDICACION_CS068_paso2_diametro_CS.md v2, 16-jul-2026, corrección abierta sobre su propio
ruling previo): el test config-model de soporte (Paso 2, Mundo A por z=122-300) mide CLUSTERING, no
metricidad -- falla en las dos direcciones: un small-world muy clusterizado pasa el test (z≈+332) y NO
tiene "lejos" real (diámetro ∝ log N); una retícula 2D pura, el ideal métrico, REPRUEBA el test (z≈−2.9,
sin triángulos). Soporte por vecinos comunes ≠ metricidad.

El juez correcto: el EXPONENTE de crecimiento del diámetro con N. Verificado por CS en 3 casos de
referencia (N∈{400..2500}): retícula 2D pendiente log-log 0.52 (d≈2, ∝√N, métrica); anillo de cliques
1.01 (d≈1, cadena, métrica en 1D); small-world 0.14 (∝log N, NO métrico). La firma de "lejos real" es
pendiente > 0 apreciable, NO un valor de diámetro a un solo N (un anillo de cliques tiene diámetro GRANDE,
179 a N=900, mayor que la retícula, y sin embargo es métrico -- solo en 1D).

Procedimiento (barato, decisivo, ANTES de re-correr Etapa 1 completa a N grande):
1. Clasificar atajos con clasifica_config_model() (cs068_paso2_mundo_ab.py, ya validado) sobre el blob
   real, en 3 escalas N∈{900,1500,2500}.
2. QUITAR los atajos clasificados -- queda el tejido local residual (posible gigante + fragmentos).
3. Medir el diámetro robusto (H._diam_robusto, gigante) del residual en las 3 escalas y ajustar la
   PENDIENTE log-log por regresión lineal sobre (log N, log diám). También se mide la fracción del gigante
   -- si el residual se fragmenta (gigante <30% de N), eso es Mundo B por otra vía (no hay componente
   métrica grande que revelar), independiente de la pendiente.
4. Regla PRE-INSCRITA (calibrada contra los 3 casos de referencia de CS, no ajustada después de ver el
   número del blob real): pendiente > 0.3 -> Mundo A real (tejido métrico latente, "lejos" real) -> PROCEDE
   a Etapa 1 completa. Pendiente <= 0.3 o fragmentación -> Mundo B -> resultado honesto del arco.

Codea/ejecuta: CC. Diseño/ruling: CS.
"""
from __future__ import annotations
import os, sys, time
from collections import deque
import numpy as np

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)
import cs068_inflacion_estirar_enfriar as E
import cs068_paso2_mundo_ab as P2
import cs067_habitacion_completa as H

RNG = np.random.default_rng

PENDIENTE_UMBRAL = 0.3  # pre-inscrito por CS: calibrado contra retícula2D=0.52, anillo_cliques=1.01, small-world=0.14


def _tamano_gigante(adj, N, rng, n_src=12):
    """Fracción de nodos alcanzable desde el gigante (mismo criterio 0.3*N que H._diam_robusto usa para
    aceptar una fuente) -- detecta si el tejido residual se FRAGMENTA al quitar los atajos clasificados."""
    mejor = 0
    for s in rng.integers(0, N, size=min(n_src, N)):
        s = int(s)
        if not adj[s]:
            continue
        dist = {s: 0}; q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1; q.append(v)
        mejor = max(mejor, len(dist))
    return mejor / N


def _un_N(N, seed, n_realizaciones_clasif=30):
    adj = E._sustrato(N, seed)
    adj_local, atajos, zs = P2.clasifica_config_model(adj, N, RNG(seed + 1),
                                                        n_realizaciones=n_realizaciones_clasif)
    frac_gigante = _tamano_gigante(adj_local, N, RNG(seed + 2))
    diam = H._diam_robusto(adj_local, N, RNG(seed + 3))
    n_local_edges = sum(len(a) for a in adj_local) // 2
    return dict(N=N, seed=seed, n_atajos=len(atajos), n_local_edges=n_local_edges,
                frac_gigante=frac_gigante, diam=diam)


def _pendiente_loglog(Ns, diams):
    x = np.log(np.array(Ns, float))
    y = np.log(np.array(diams, float))
    A = np.vstack([x, np.ones_like(x)]).T
    pendiente, intercepto = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(pendiente), float(intercepto)


def main():
    Ns = [int(x) for x in os.environ.get("CS068_NS", "900,1500,2500").split(",")]
    print("=" * 108, flush=True)
    print("CS068 PASO 2b — diagnóstico barato: ¿el tejido residual (blob real menos atajos) es MÉTRICO?",
          flush=True)
    print("Discriminante: PENDIENTE log-log de diám(N), no el valor de diám a un solo N.", flush=True)
    print(f"escalas N={Ns}  umbral pre-inscrito: pendiente > {PENDIENTE_UMBRAL}", flush=True)
    print("=" * 108, flush=True)
    t0 = time.time()
    resultados = []
    for i, N in enumerate(Ns):
        seed = 68300 + 97 * i
        r = _un_N(N, seed)
        resultados.append(r)
        print(f"\n--- N={N} (seed={seed}) ---", flush=True)
        print(f"  atajos_clasificados={r['n_atajos']}  enlaces_locales_residual={r['n_local_edges']}",
              flush=True)
        print(f"  frac_gigante(residual)={r['frac_gigante']:.3f}  diam_residual={r['diam']:.2f}", flush=True)
        print(f"  tiempo acumulado: {(time.time()-t0)/60:.2f} min", flush=True)

    fragmenta = any(r["frac_gigante"] < 0.3 for r in resultados)
    Ns_ok = [r["N"] for r in resultados]
    diams_ok = [r["diam"] for r in resultados]
    pendiente, intercepto = _pendiente_loglog(Ns_ok, diams_ok)

    print("\n" + "=" * 108, flush=True)
    print(f"diám(N): {[(r['N'], round(r['diam'], 2)) for r in resultados]}", flush=True)
    print(f"frac_gigante(N): {[(r['N'], round(r['frac_gigante'], 3)) for r in resultados]}", flush=True)
    print(f"pendiente log-log = {pendiente:.3f}  (referencia CS: retícula2D=0.52, anillo=1.01, small-world=0.14)",
          flush=True)
    if fragmenta:
        print("\nVEREDICTO: MUNDO B por FRAGMENTACIÓN. El tejido residual pierde el gigante (<30% de N) al",
              flush=True)
        print("quitar los atajos clasificados -- no hay componente métrica grande que el enfriamiento pueda",
              flush=True)
        print("revelar, sin importar la pendiente del fragmento que sobreviva.", flush=True)
    elif pendiente > PENDIENTE_UMBRAL:
        print(f"\nVEREDICTO: MUNDO A real. pendiente={pendiente:.3f} > {PENDIENTE_UMBRAL} -> diám ∝ N^(1/d)",
              flush=True)
        print("con d finito -- hay tejido MÉTRICO latente con 'lejos' real (no solo clustering). PROCEDE a",
              flush=True)
        print("Etapa 1 completa sobre este tejido residual: el enfriamiento tiene algo real que revelar.",
              flush=True)
    else:
        print(f"\nVEREDICTO: MUNDO B. pendiente={pendiente:.3f} <= {PENDIENTE_UMBRAL} -> diám ~ log N, no",
              flush=True)
        print("crece polinómicamente con N. El tejido residual sigue siendo mundo-pequeño -- no hay geometría",
              flush=True)
        print("latente bajo los atajos. El enfriamiento no puede fabricar lo que no está. Resultado honesto:",
              flush=True)
        print("CS066/CS067 nunca encendieron direcciones porque el sustrato jamás tuvo métrica, ni latente.",
              flush=True)
    print(f"\ntiempo total: {(time.time()-t0)/60:.2f} min", flush=True)


# ============================ BLINDAJE de semillas (la corrida de 1 semilla/N saltó 5→5→13: sospechoso) ============================
def main_blindado():
    """La corrida de 1 semilla por N dio diám 5.0→5.0→13.0 (plano y luego un salto) y n_atajos NO monótono
    con N (294→453→218) -- un patrón raro para adjudicar el arco con una sola muestra por escala. Blindaje:
    varias semillas por N, reportar la media y el rango, y ajustar la pendiente sobre la MEDIA -- para no
    dejar que el veredicto dependa de una sola tirada de dados en N=2500."""
    Ns = [int(x) for x in os.environ.get("CS068_NS", "900,1500,2500").split(",")]
    n_seeds = int(os.environ.get("CS068_SEEDS_POR_N", 4))
    print("=" * 108, flush=True)
    print("CS068 PASO 2b — BLINDADO: varias semillas por N (la corrida de 1 semilla saltó 5->5->13,", flush=True)
    print("sospechoso para adjudicar el arco con una muestra)", flush=True)
    print(f"escalas N={Ns}  semillas/N={n_seeds}  umbral pre-inscrito: pendiente > {PENDIENTE_UMBRAL}", flush=True)
    print("=" * 108, flush=True)
    t0 = time.time()
    medias, rangos = [], []
    for i, N in enumerate(Ns):
        diams, gigantes, natajos = [], [], []
        for s in range(n_seeds):
            seed = 68500 + 97 * i + 13 * s
            r = _un_N(N, seed)
            diams.append(r["diam"]); gigantes.append(r["frac_gigante"]); natajos.append(r["n_atajos"])
        m, lo, hi = float(np.mean(diams)), float(np.min(diams)), float(np.max(diams))
        medias.append(m); rangos.append((lo, hi))
        print(f"\n--- N={N} ({n_seeds} semillas) ---", flush=True)
        print(f"  diam por semilla: {diams}", flush=True)
        print(f"  diam media={m:.2f} rango=[{lo:.2f},{hi:.2f}]  frac_gigante media={np.mean(gigantes):.3f}",
              flush=True)
        print(f"  n_atajos por semilla: {natajos}", flush=True)
        print(f"  tiempo acumulado: {(time.time()-t0)/60:.2f} min", flush=True)

    pendiente, intercepto = _pendiente_loglog(Ns, medias)
    # pendiente en los extremos del rango (peor caso / mejor caso) para ver cuán frágil es el veredicto
    pendiente_lo, _ = _pendiente_loglog(Ns, [r[0] for r in rangos])
    pendiente_hi, _ = _pendiente_loglog(Ns, [r[1] for r in rangos])
    print("\n" + "=" * 108, flush=True)
    print(f"diám medio(N): {list(zip(Ns, [round(m,2) for m in medias]))}", flush=True)
    print(f"pendiente sobre la MEDIA = {pendiente:.3f}  (rango de pendiente usando min/max por N: "
          f"[{min(pendiente_lo,pendiente_hi):.3f},{max(pendiente_lo,pendiente_hi):.3f}])", flush=True)
    if pendiente > PENDIENTE_UMBRAL and min(pendiente_lo, pendiente_hi) > PENDIENTE_UMBRAL:
        print(f"\nVEREDICTO (blindado): MUNDO A real. La pendiente > {PENDIENTE_UMBRAL} incluso en el caso", flush=True)
        print("más conservador del rango de semillas -- no depende de una tirada afortunada.", flush=True)
    elif pendiente > PENDIENTE_UMBRAL:
        print(f"\nVEREDICTO (blindado): FRÁGIL. La pendiente sobre la media supera {PENDIENTE_UMBRAL} pero", flush=True)
        print("el rango de semillas cruza el umbral -- no blindado todavía, reportar así a CS sin forzar.", flush=True)
    else:
        print(f"\nVEREDICTO (blindado): MUNDO B. pendiente media <= {PENDIENTE_UMBRAL}, consistente con", flush=True)
        print("mundo-pequeño incluso con varias semillas por escala.", flush=True)
    print(f"\ntiempo total: {(time.time()-t0)/60:.2f} min", flush=True)


if __name__ == "__main__":
    if os.environ.get("CS068_MODO") == "blindado":
        main_blindado()
    else:
        main()
