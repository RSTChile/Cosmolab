"""
CS083b — FASE IV: afinar el 92%/8% con 2 controles NUEVOS (local-roto vs global)
=================================================================================
QUIÉN SOY: sigue directo a `cs083_fase4_robustecer.py` (que a su vez sigue a
`cs082_fase4_4sustratos.py`, ver FASE4_orden_superior_resultado_CS.md y
FASE4_robustecido_CS.md). Ese informe descompuso el aplanamiento de holonomía del sustrato 4
(2-complejo con retroalimentación cara→arista) en:
    ~92% "consenso global"        (el mero VOLUMEN de retroalimentación disperso, sin importar dónde
                                    cae exactamente, ya aplana casi todo)
    ~8%  "cierre topológico LOCAL" (z=-4.14, p≈0.0006 — específico de que la retroalimentación caiga
                                    en el triángulo/cara geométricamente correcto)
usando UN control fino ("4b_rewire_azar" en cs083): la cara empuja 3 aristas elegidas al azar de TODO
el grafo (fijas para la corrida), en vez de sus 3 aristas de borde reales.

El equipo pidió afinar ESE control con dos controles MÁS PRECISOS y mutuamente distintos, para separar
mejor dos cosas que "rewire al azar" mezcla parcialmente:
  (a) ¿la correspondencia GEOMÉTRICA exacta cara↔su-propio-borde importa? (vs. cualquier trío de 3
      aristas, sin importar si están geométricamente relacionadas entre sí)
  (b) ¿importa que la retroalimentación esté CONCENTRADA en tríos de 3 aristas a la vez (aunque sean
      los tríos equivocados), o el efecto sobrevive igual si se DILUYE sobre TODAS las aristas del
      grafo a la vez, sin ningún trío?

Por eso este script construye:

  NULL-LOCAL-ROTO ("control_local_roto"): cada triángulo real T sigue "escuchando" su propio defecto
  (mismo cálculo, misma fuerza J_FACE, mismo número de eventos: uno por triángulo y por sweep — igual
  que el sustrato 4 real) pero el empujón se REDIRIGE a las 3 aristas de OTRO triángulo real del mismo
  grafo (T' ≠ T, escogido por una permutación derangement fija al arrancar la corrida) en vez de a sus
  propias 3 aristas de borde. A diferencia del "rewire al azar" de cs083 (que sortea 3 aristas sueltas
  de CUALQUIER parte del grafo, que típicamente ni comparten nodo entre sí), acá el trío-destino sigue
  siendo un trío GENUINO de 3 aristas mutuamente incidentes (un triángulo real en otra parte del
  grafo) — más parecido en "forma" y en grado de los nodos involucrados al trío original que 3 aristas
  sueltas cualquiera. Rompe la correspondencia geométrica LOCAL exacta, pero preserva "trío coherente
  de 3 aristas de un triángulo genuino" como unidad de empuje.
  HONESTIDAD: esto NO es un bootstrap de grado formalmente controlado (no garantiza que el triángulo
  T' tenga exactamente los mismos 3 grados de nodo que T, nodo a nodo) — es una redirección a OTRO
  trío real del mismo grafo, extraído de la misma población de "tríos de arista que sí forman un
  triángulo genuino", lo cual en espíritu preserva mejor "cantidad/fuerza/grado" que 3 aristas sueltas
  sin relación entre sí, pero no es una garantía exacta.

  NULL-GLOBAL ("control_global"): en vez de que cada cara empuje sólo 3 aristas (las suyas o las de
  otro trío), el mecanismo entero se cambia: en cada sweep se calcula la media circular GLOBAL del
  campo de aristas (mu), y CADA arista del grafo (las N_ARISTAS, no sólo 3 por evento) recibe el MISMO
  tipo de empujón que recibiría una arista real — misma constante de fuerza J_FACE/3 por evento — pero
  dirigido hacia la media circular GLOBAL en vez de hacia el defecto de SU propio triángulo. Así la
  "corrección" ya no se concentra nunca en ningún trío: se diluye por igual sobre TODO el grafo, cada
  sweep.
  HONESTIDAD: esto preserva la MISMA constante de fuerza por evento (J_FACE/3) que usa el sustrato
  real, pero NO intenta igualar el conteo exacto de eventos empuje-por-empuje del sustrato real (eso
  exigiría acoplar paso a paso las dos simulaciones, lo que mezclaría sus trayectorias estocásticas).
  Se reporta explícitamente la escala de cobertura de cada brazo (nº de aristas tocadas por sweep) en
  la tabla de control de equiparación, para que la comparación sea auditable y no se esconda la
  aproximación.

LOS 4 BRAZOS COMPARADOS (mismo N, mismo grafo base por semilla, mismo K/J/J_FACE/ruido/presupuesto de
cómputo que cs082/cs083):
  1) REAL            — sustrato 4 sin cambios (importado íntegro de cs082, sin modificar)
  2) NULL-REWIRE     — el control YA EXISTENTE de cs083 (importado íntegro, sin modificar) — referencia
                        de continuidad con el 92%/8% ya reportado
  3) NULL-LOCAL-ROTO — nuevo (este script): mismo trío-coherente, trío EQUIVOCADO (permutación)
  4) NULL-GLOBAL     — nuevo (este script): sin trío, corrección diluida sobre TODAS las aristas

REGLA DE LA CASA: no toca cs082_fase4_4sustratos.py ni cs083_fase4_robustecer.py — importa sus piezas
ya auditadas (construcción de grafo, dinámica circular, holonomía, sustrato 4 real, control de rewire,
funciones estadísticas). Código nuevo: NULL-LOCAL-ROTO, NULL-GLOBAL, y el ensamblado de la batería de
4 brazos.

No declara cierre ni fracción "definitiva" de local-vs-global — reporta números; la lectura final es
de Alexis.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cs082_fase4_4sustratos import (  # noqa: E402
    N, P_EDGE, K, J, NOISE, J_FACE, COMPUTE_BUDGET,
    construir_base, _linea_adyacencia, _circ_mean_update, _holonomia_triangulos,
    _n_sweeps_para_presupuesto, null_de, shuffled_de,
    correr_sustrato_4_2complejo,
)
from cs083_fase4_robustecer import (  # noqa: E402
    correr_sustrato_4_control_fino,   # el "rewire al azar" ya existente -> aquí "NULL-REWIRE"
    z_score_pareado, test_permutacion_signo_pareado,
)

# ============================ CONFIG NUEVO ============================
N_SEEDS_FULL = 20                       # mismo nº de semillas que cs083 (pedido explícito de la tarea)
SEEDS_FULL   = list(range(1, N_SEEDS_FULL + 1))
SEEDS_PILOT  = [1, 2, 3, 4, 5]           # piloto chico antes de escalar
N_PERM       = 20_000
RNG_MASTER_SEED = 778                    # distinto del 777 de cs083, stream propio de este script
# ========================================================================


def _derangement(n, rng, max_tries=1000):
    """Permutación de {0..n-1} SIN puntos fijos (perm[t] != t para todo t) por rechazo — usada para
    redirigir cada triángulo real a OTRO triángulo real distinto (NULL-LOCAL-ROTO). Con n_tri~190 la
    probabilidad de que una permutación azarosa no tenga puntos fijos es ~1/e, así que el rechazo
    converge en pocos intentos en la práctica; fallback determinista (rotación cíclica, que nunca
    tiene puntos fijos si n>=2) por si acaso."""
    if n <= 1:
        return np.arange(n)
    idx = np.arange(n)
    for _ in range(max_tries):
        perm = rng.permutation(n)
        if not np.any(perm == idx):
            return perm
    return np.roll(idx, 1)


# ============ CONTROL NUEVO 1 — NULL-LOCAL-ROTO (trío coherente, pero el trío EQUIVOCADO) ============
def correr_sustrato_4_null_local_roto(adj, edges, triangles, seed):
    """Copia estructural del sustrato 4 real (mismos sweeps, mismo J_FACE, un evento de empuje por
    triángulo y por sweep = misma 'cantidad de retroalimentación'), con un solo cambio: cada triángulo
    T sigue calculando SU PROPIO defecto de holonomía (a partir de SUS 3 aristas reales, igual que el
    sustrato real: 'la retroalimentación que le corresponde a esa cara'), pero el empujón resultante se
    aplica a las 3 aristas de OTRO triángulo real T'=perm(T) (T'≠T, permutación derangement fija al
    arrancar la corrida) — 3 aristas AJENAS, pero que sí forman un triángulo genuino en otra parte del
    mismo grafo (no 3 aristas sueltas sin relación entre sí, a diferencia del rewire-azar de cs083)."""
    rng = np.random.default_rng(46_000 + seed)          # stream propio, independiente de real/rewire
    n_edges = len(edges)
    idx_edge = {e: i for i, e in enumerate(edges)}
    vecinos = _linea_adyacencia(edges)
    s = rng.uniform(0, K, n_edges)
    dof = n_edges + len(triangles)
    n_sweeps = _n_sweeps_para_presupuesto(n_edges, COMPUTE_BUDGET)   # IDÉNTICO presupuesto que real
    n_tri = len(triangles)
    tri_edges = []
    for (i, j, k) in triangles:
        e1 = (i, j) if i < j else (j, i)
        e2 = (j, k) if j < k else (k, j)
        e3 = (i, k) if i < k else (k, i)
        tri_edges.append((idx_edge[e1], idx_edge[e2], idx_edge[e3]))
    perm = _derangement(n_tri, rng)    # T -> T' != T, fija para toda la corrida
    t0 = time.time()
    for _ in range(n_sweeps):
        s = _circ_mean_update(s, vecinos, J, NOISE, rng)
        if tri_edges:
            correccion = np.zeros(n_edges)
            cuenta = np.zeros(n_edges)
            for t_idx, (a, b, c) in enumerate(tri_edges):
                h = (s[a] + s[b] + s[c]) % K       # defecto de SU PROPIO triángulo (igual que real)
                h = h - K if h > K / 2 else h
                destino = tri_edges[perm[t_idx]]   # pero el empujón va a OTRO triángulo real
                for idx in destino:
                    correccion[idx] += -J_FACE * h / 3.0
                    cuenta[idx] += 1
            mask = cuenta > 0
            s[mask] = (s[mask] + correccion[mask] / cuenta[mask]) % K
    dt = time.time() - t0
    E = {e: s[idx_edge[e]] for e in edges}
    eventos_por_sweep = 3 * n_tri     # mismo conteo de empuje-eventos que el sustrato real
    return E, dof, n_sweeps, dt, eventos_por_sweep


# ============ CONTROL NUEVO 2 — NULL-GLOBAL (sin trío: corrección diluida en TODA arista) ============
def correr_sustrato_4_null_global(adj, edges, triangles, seed):
    """Reemplaza el empuje cara->3-aristas por un empuje 'toda arista hacia la media circular global
    del campo', cada sweep. Usa la MISMA constante de fuerza por evento (J_FACE/3) que cada arista real
    recibe de su cara, pero aplicada a TODAS las N_ARISTAS del grafo (cobertura total) en vez de sólo a
    3 aristas por triángulo (cobertura concentrada en tríos). No hay ningún trío en este mecanismo —
    es la versión más diluida posible del mismo 'volumen de retroalimentación'."""
    rng = np.random.default_rng(47_000 + seed)
    n_edges = len(edges)
    idx_edge = {e: i for i, e in enumerate(edges)}
    vecinos = _linea_adyacencia(edges)
    s = rng.uniform(0, K, n_edges)
    dof = n_edges + len(triangles)
    n_sweeps = _n_sweeps_para_presupuesto(n_edges, COMPUTE_BUDGET)
    n_tri = len(triangles)
    t0 = time.time()
    for _ in range(n_sweeps):
        s = _circ_mean_update(s, vecinos, J, NOISE, rng)
        if n_edges > 0:
            ang = 2 * np.pi * s / K
            z_mean = np.mean(np.exp(1j * ang))
            mu = (np.angle(z_mean) / (2 * np.pi) * K) % K
            h_edge = (s - mu) % K
            h_edge = np.where(h_edge > K / 2, h_edge - K, h_edge)   # defecto de CADA arista vs. la
                                                                      # media GLOBAL (misma convención
                                                                      # de centrado que la holonomía)
            s = (s - (J_FACE / 3.0) * h_edge) % K
    dt = time.time() - t0
    E = {e: s[idx_edge[e]] for e in edges}
    eventos_por_sweep = n_edges     # cobertura total: TODAS las aristas, cada sweep
    return E, dof, n_sweeps, dt, eventos_por_sweep


def resumen(nombre, vals):
    v = np.asarray(vals)
    return f"{nombre:<26} n={len(v):>2}  media={v.mean():.4f}  DE={v.std(ddof=1):.4f}  " \
           f"min={v.min():.4f}  max={v.max():.4f}"


def _reporte_par(nombre, a, b, n_perm, rng_perm, direccion="a<b"):
    z = z_score_pareado(a, b)
    t = test_permutacion_signo_pareado(a, b, n_perm, rng_perm, direccion=direccion)
    print(f"  {nombre:<32} z={z:+7.2f}   diff_obs={t['obs_diff']:+.4f}   "
          f"p_una_cola={t['p_una_cola']:.5f}   p_dos_colas={t['p_dos_colas']:.5f}")
    return dict(z=z, **t)


def correr_bateria(seeds, tag=""):
    print(f"CS083b — FASE IV: NULL-LOCAL-ROTO vs NULL-GLOBAL {tag}")
    print("=" * 100)
    print(f"N={N} nodos · K={K} · J={J} · J_FACE={J_FACE} · ruido={NOISE} · presupuesto={COMPUTE_BUDGET} "
          f"op-relación/sustrato · seeds={seeds}\n")

    filas = []
    t_bateria0 = time.time()
    for seed in seeds:
        adj, edges, triangles = construir_base(seed)
        n_e, n_t = len(edges), len(triangles)

        E_real, dof_real, sw_real, dt_real = correr_sustrato_4_2complejo(adj, edges, triangles, seed)
        E_rw, dof_rw, sw_rw, dt_rw, n_tri_rw = correr_sustrato_4_control_fino(adj, edges, triangles, seed)
        E_lr, dof_lr, sw_lr, dt_lr, ev_lr = correr_sustrato_4_null_local_roto(adj, edges, triangles, seed)
        E_gl, dof_gl, sw_gl, dt_gl, ev_gl = correr_sustrato_4_null_global(adj, edges, triangles, seed)

        h_real = _holonomia_triangulos(E_real, triangles).mean()
        h_rw   = _holonomia_triangulos(E_rw, triangles).mean()
        h_lr   = _holonomia_triangulos(E_lr, triangles).mean()
        h_gl   = _holonomia_triangulos(E_gl, triangles).mean()

        E_null = null_de(E_real, seed)
        h_null = _holonomia_triangulos(E_null, triangles).mean()
        E_shuf = shuffled_de(E_real, seed)
        h_shuf = _holonomia_triangulos(E_shuf, triangles).mean()

        filas.append(dict(
            seed=seed, n_edges=n_e, n_tri=n_t,
            dof_real=dof_real, sw_real=sw_real,
            dof_rw=dof_rw, sw_rw=sw_rw, ev_rw=3 * n_t,
            dof_lr=dof_lr, sw_lr=sw_lr, ev_lr=ev_lr,
            dof_gl=dof_gl, sw_gl=sw_gl, ev_gl=ev_gl,
            h_real=h_real, h_rewire=h_rw, h_local_roto=h_lr, h_global=h_gl,
            h_null=h_null, h_shuf=h_shuf,
        ))
    dt_bateria = time.time() - t_bateria0

    # ---------------- CONTROL DE EQUIPARACIÓN ----------------
    print("── CONTROL DE EQUIPARACIÓN (mismo N/grafo-base/K/J/J_FACE/ruido/presupuesto en los 4 brazos) ──")
    f0 = filas[0]
    print(f"  REAL           : DoF={f0['dof_real']} sweeps={f0['sw_real']} eventos-empuje/sweep={3*f0['n_tri']} (3×n_tri, sobre SUS PROPIAS aristas)")
    print(f"  NULL-REWIRE    : DoF={f0['dof_rw']} sweeps={f0['sw_rw']} eventos-empuje/sweep={f0['ev_rw']} (3×n_tri, sobre 3 aristas sueltas al azar)")
    print(f"  NULL-LOCAL-ROTO: DoF={f0['dof_lr']} sweeps={f0['sw_lr']} eventos-empuje/sweep={f0['ev_lr']} (3×n_tri, sobre el trío de OTRO triángulo real)")
    print(f"  NULL-GLOBAL    : DoF={f0['dof_gl']} sweeps={f0['sw_gl']} eventos-empuje/sweep={f0['ev_gl']} (=n_aristas, TODA arista cada sweep, sin trío)")
    coinciden_sweeps = all(f["sw_real"] == f["sw_rw"] == f["sw_lr"] == f["sw_gl"] for f in filas)
    print(f"  ¿mismos sweeps en los 4 brazos, las {len(seeds)} semillas?  {'sí' if coinciden_sweeps else 'NO -- revisar'}")
    print("  (NULL-GLOBAL toca más aristas/sweep por diseño -- es la variable manipulada: concentrado en")
    print("   tríos [real/rewire/local-roto] vs. diluido en todo el grafo [global]. Ver texto del script.)\n")

    # ---------------- TABLA POR SEMILLA (primeras 5) ----------------
    print("── primeras 5 semillas ──")
    print(f"  {'seed':>4} {'h_REAL':>8} {'h_REWIRE':>9} {'h_LOC_ROTO':>11} {'h_GLOBAL':>9} {'h_NULL':>8} {'h_SHUF':>8}")
    for f in filas[:5]:
        print(f"  {f['seed']:>4} {f['h_real']:>8.3f} {f['h_rewire']:>9.3f} {f['h_local_roto']:>11.3f} "
              f"{f['h_global']:>9.3f} {f['h_null']:>8.3f} {f['h_shuf']:>8.3f}")
    print()

    h_real = np.array([f["h_real"] for f in filas])
    h_rw   = np.array([f["h_rewire"] for f in filas])
    h_lr   = np.array([f["h_local_roto"] for f in filas])
    h_gl   = np.array([f["h_global"] for f in filas])
    h_null = np.array([f["h_null"] for f in filas])
    h_shuf = np.array([f["h_shuf"] for f in filas])

    print(f"── RESUMEN DESCRIPTIVO sobre {len(seeds)} semillas ──")
    print("  " + resumen("h_REAL", h_real))
    print("  " + resumen("h_NULL_REWIRE (cs083)", h_rw))
    print("  " + resumen("h_NULL_LOCAL_ROTO (nuevo)", h_lr))
    print("  " + resumen("h_NULL_GLOBAL (nuevo)", h_gl))
    print("  " + resumen("h_NULL (ruido puro)", h_null))
    print("  " + resumen("h_SHUFFLED", h_shuf))
    print()

    # ---------------- TESTS PAREADOS ----------------
    rng_perm = np.random.default_rng(RNG_MASTER_SEED)
    print(f"── TESTS PAREADOS POR SEMILLA (n={len(seeds)}, unidad=semilla/grafo-base, {N_PERM} permutaciones sign-flip) ──\n")
    tests = {}
    print("  [1] REAL vs NULL-REWIRE (chequeo de reproducibilidad del 92%/8% de cs083)")
    tests["real_vs_rewire"] = _reporte_par("REAL - NULL_REWIRE", h_real, h_rw, N_PERM, rng_perm, "a<b")
    print("\n  [2] REAL vs NULL-LOCAL-ROTO (*** ¿la correspondencia geométrica exacta importa? ***)")
    tests["real_vs_local_roto"] = _reporte_par("REAL - NULL_LOCAL_ROTO", h_real, h_lr, N_PERM, rng_perm, "a<b")
    print("\n  [3] REAL vs NULL-GLOBAL (*** ¿la concentración en tríos importa? ***)")
    tests["real_vs_global"] = _reporte_par("REAL - NULL_GLOBAL", h_real, h_gl, N_PERM, rng_perm, "a<b")
    print("\n  [4] NULL-LOCAL-ROTO vs NULL-GLOBAL (*** el contraste central de esta tarea ***)")
    print("      Si LOCAL-ROTO se acerca más a REAL que GLOBAL -> concentrar en tríos (aunque")
    print("      equivocados) importa más que la correspondencia geométrica exacta.")
    print("      Si GLOBAL se acerca más a REAL que LOCAL-ROTO -> lo que importa es el volumen total,")
    print("      no la concentración en tríos ni la geometría exacta.")
    tests["local_roto_vs_global"] = _reporte_par("NULL_LOCAL_ROTO - NULL_GLOBAL", h_lr, h_gl, N_PERM, rng_perm, "a<b")
    print("\n  [5] NULL-REWIRE vs NULL-LOCAL-ROTO (¿el trío coherente-pero-ajeno se distingue del")
    print("      rewire totalmente suelto?)")
    tests["rewire_vs_local_roto"] = _reporte_par("NULL_REWIRE - NULL_LOCAL_ROTO", h_rw, h_lr, N_PERM, rng_perm, "a<b")
    print("\n  [6] NULL-REWIRE vs NULL-GLOBAL")
    tests["rewire_vs_global"] = _reporte_par("NULL_REWIRE - NULL_GLOBAL", h_rw, h_gl, N_PERM, rng_perm, "a<b")
    print("\n  [7] NULL-LOCAL-ROTO vs NULL (ruido puro) / [8] NULL-GLOBAL vs NULL (ruido puro)")
    tests["local_roto_vs_null"] = _reporte_par("NULL_LOCAL_ROTO - NULL(ruido)", h_lr, h_null, N_PERM, rng_perm, "a<b")
    tests["global_vs_null"] = _reporte_par("NULL_GLOBAL - NULL(ruido)", h_gl, h_null, N_PERM, rng_perm, "a<b")

    # ---------------- DESCOMPOSICIÓN, análoga a cs083 pero con 2 controles nuevos ----------------
    gap_total = h_null.mean() - h_real.mean()
    gap_vs_rewire  = h_rw.mean() - h_real.mean()
    gap_vs_local   = h_lr.mean() - h_real.mean()
    gap_vs_global  = h_gl.mean() - h_real.mean()
    print(f"\n── DESCOMPOSICIÓN comparativa (referencia: gap_total = h_NULL(ruido) - h_REAL = {gap_total:+.4f}) ──")
    print(f"  h_NULL_REWIRE     - h_REAL = {gap_vs_rewire:+.4f}  -> {'frac.=' + format(gap_vs_rewire/gap_total, '+.1%') if abs(gap_total) > 1e-9 else 'n/a'}  (referencia cs083, ya reportado ~8% local)")
    print(f"  h_NULL_LOCAL_ROTO - h_REAL = {gap_vs_local:+.4f}  -> {'frac.=' + format(gap_vs_local/gap_total, '+.1%') if abs(gap_total) > 1e-9 else 'n/a'}")
    print(f"  h_NULL_GLOBAL     - h_REAL = {gap_vs_global:+.4f}  -> {'frac.=' + format(gap_vs_global/gap_total, '+.1%') if abs(gap_total) > 1e-9 else 'n/a'}")
    print("  (estas 'fracciones' son la MISMA lectura que cs083 [cuánto de la caída total NULL-REAL se")
    print("   pierde con cada control] pero ahora con 2 versiones distintas del control -- no se declara")
    print("   cuál es la 'correcta', se reportan las 3 para que Alexis compare.)\n")

    print(f"Tiempo total de la batería ({len(seeds)} semillas x 4 corridas del sustrato 4): {dt_bateria:.1f}s")
    return filas, tests


def _guardar_csv(filas, path):
    import csv
    campos = list(filas[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        for fila in filas:
            w.writerow(fila)
    print(f"CSV crudo guardado en: {path}")


def main():
    modo = sys.argv[1] if len(sys.argv) > 1 else "full"
    if modo == "pilot":
        filas, tests = correr_bateria(SEEDS_PILOT, tag="(PILOTO, 5 semillas)")
        _guardar_csv(filas, str(Path(__file__).resolve().parent / "cs083b_resultados_piloto.csv"))
    else:
        filas, tests = correr_bateria(SEEDS_FULL, tag="(COMPLETO, 20 semillas)")
        _guardar_csv(filas, str(Path(__file__).resolve().parent / "cs083b_resultados.csv"))
    print("\nFin. Ver FASE4_control_local_global_CS.md para la interpretación completa (lectura de Alexis pendiente).")


if __name__ == "__main__":
    main()
