"""
CS083 — FASE IV ROBUSTECIDO: ¿el aplanamiento del sustrato 4 es cierre LOCAL o consenso GLOBAL?
==================================================================================================
QUIÉN SOY: sigue directo a `cs082_fase4_4sustratos.py` (Fase IV, ago-2026, ver
FASE4_orden_superior_resultado_CS.md). Ese informe encontró que, de los 4 sustratos relacionales
probados, sólo el **sustrato 4** (2-complejo con retroalimentación cara→arista: cada triángulo
"empuja" a sus 3 aristas de borde para reducir su propio defecto de holonomía) se separa de NULL con
solidez: holonomía ~5x menor que el azar (h_REAL=0.30 vs h_NULL=1.54, 5 semillas).

Pero el informe original dejó una duda SIN resolver (§3, "Advertencia honesta"): el control SHUFFLED
(mismos valores finales de arista, topología barajada al azar) dio h_SHUF=0.57 — mucho más cerca de
REAL (0.30) que de NULL (1.54). Es decir: **sólo con re-barajar QUÉ arista tiene QUÉ valor** (sin
tocar ni un número), ya se recupera la mayor parte del aplanamiento. Eso sugiere que gran parte del
efecto podría ser "hacia dónde converge la DISTRIBUCIÓN global de valores" (un consenso agregado que
el feedback empuja a TODO el campo por igual) — no que cada triángulo cierre su propio lazo de a tres
por la topología LOCAL específica cara↔sus-3-aristas-de-borde.

ESTE SCRIPT NO RESUELVE la duda mirando SHUFFLED de nuevo (ese control mezcla dos cosas: rompe la
topología del feedback Y rompe la dinámica de alineación diádica de fondo). En cambio construye un
control MÁS FINO, quirúrgico, que aísla exactamente la variable en duda:

  CONTROL FINO ("4b_rewire_azar"): la MISMA maquinaria exacta del sustrato 4 real — mismos sweeps,
  mismo J_FACE, mismo número de triángulos empujando, mismo número TOTAL de empujes cara→arista (la
  misma "presión" global de retroalimentación) — pero la ASIGNACIÓN cara→3-aristas se hace AL AZAR
  sobre el grafo completo (fija para toda la corrida), no usando las 3 aristas de borde reales del
  triángulo. Cada "cara" sigue empujando 3 aristas cualquiera hacia el consenso; lo único que cambia
  es que esas 3 aristas YA NO son necesariamente las del triángulo geométrico correcto.

  Lógica de falsación:
    - Si el control fino APLANA la holonomía de los triángulos VERDADEROS tanto como el sustrato 4
      real → el efecto es consenso global puro: cualquier feedback disperso por el grafo empuja todo
      el campo hacia valores compatibles, y esa compatibilidad "de rebote" también achica la holonomía
      de cualquier triángulo, real o no. No hay nada específicamente LOCAL en juego.
    - Si el control fino se queda mucho más cerca de NULL que el sustrato 4 real → confirma que la
      correspondencia LOCAL correcta cara↔su-propio-borde SÍ importa por encima del mero volumen de
      feedback — hay algo genuinamente topológico/local, no sólo "más actualizaciones en general".

Además sube las semillas de 5 a N_SEEDS (ver abajo) para tener más poder estadístico, y reporta
z-scores + tests de permutación pareados por semilla (mismo estilo que el resto del proyecto esta
semana, p.ej. cs078_kappaV_permutacion.py).

REGLA DE LA CASA: no toca `cs082_fase4_4sustratos.py` — importa sus piezas ya auditadas (construcción
del grafo base, dinámica de alineación circular, medición de holonomía, sustrato 4 real, controles
NULL/SHUFFLED) para no duplicar código ni divergir silenciosamente de lo ya reportado. El único código
nuevo es el control fino de rewire-al-azar y el análisis estadístico ampliado.

No declara cierre de la Pared R7 ni de Fase IV. Reporta números; veredicto es de Alexis.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cs082_fase4_4sustratos import (  # noqa: E402  (import tras sys.path, intencional)
    N, P_EDGE, K, J, NOISE, J_FACE, COMPUTE_BUDGET,
    construir_base, _linea_adyacencia, _circ_mean_update, _holonomia_triangulos,
    _n_sweeps_para_presupuesto, null_de, shuffled_de,
    correr_sustrato_4_2complejo,
)

# ============================ CONFIG NUEVO (sólo lo que agrega este script) ============================
N_SEEDS = 20                 # 5 -> 20: más poder estadístico (pedido explícito de la tarea)
SEEDS   = list(range(1, N_SEEDS + 1))   # seeds 1-5 coinciden EXACTO con cs082 (mismo grafo base,
                                         # mismo sustrato 4 real) -> sirve de chequeo de reproducibilidad
N_PERM  = 20_000             # repeticiones del test de permutación (sign-flip pareado por semilla)
RNG_MASTER_SEED = 777         # semilla del generador de permutaciones (reproducible)
# ==========================================================================================================


# ============================ CONTROL FINO — rewire al azar (código nuevo) ============================
def correr_sustrato_4_control_fino(adj, edges, triangles, seed):
    """Copia estructural EXACTA de correr_sustrato_4_2complejo (mismos sweeps, mismo J_FACE, mismo
    número de tríos cara->arista empujando en cada sweep = misma 'presión' global total de feedback),
    con UN solo cambio deliberado: la lista de 3 aristas que cada 'cara' empuja se sortea AL AZAR
    sobre el grafo completo (fija al arrancar la corrida, no recalculada cada sweep) en vez de ser las
    3 aristas de borde reales del triángulo. Así se preserva la cantidad de retroalimentación pero se
    rompe la correspondencia LOCAL cara<->su-propio-borde -- exactamente la variable que hay que aislar
    para distinguir 'cierre topológico local' de 'consenso global agregado'."""
    rng = np.random.default_rng(45_000 + seed)      # stream propio, independiente del sustrato 4 real
    n_edges = len(edges)
    idx_edge = {e: i for i, e in enumerate(edges)}
    vecinos = _linea_adyacencia(edges)
    s = rng.uniform(0, K, n_edges)
    dof = n_edges + len(triangles)
    n_sweeps = _n_sweeps_para_presupuesto(n_edges, COMPUTE_BUDGET)   # IDÉNTICO presupuesto que el real
    n_tri = len(triangles)
    # asignación FIJA cara->3 aristas al azar (mismo número de tríos = n_tri que en el sustrato real,
    # así el número TOTAL de empujes cara->arista por sweep es idéntico; sólo cambia A CUÁLES aristas)
    tri_edges_azar = [tuple(rng.choice(n_edges, size=3, replace=False)) for _ in range(n_tri)] if n_edges >= 3 else []
    t0 = time.time()
    for _ in range(n_sweeps):
        s = _circ_mean_update(s, vecinos, J, NOISE, rng)
        if tri_edges_azar:
            correccion = np.zeros(n_edges)
            cuenta = np.zeros(n_edges)
            for (a, b, c) in tri_edges_azar:
                h = (s[a] + s[b] + s[c]) % K
                h = h - K if h > K / 2 else h
                for idx in (a, b, c):
                    correccion[idx] += -J_FACE * h / 3.0
                    cuenta[idx] += 1
            mask = cuenta > 0
            s[mask] = (s[mask] + correccion[mask] / cuenta[mask]) % K
    dt = time.time() - t0
    E = {e: s[idx_edge[e]] for e in edges}
    return E, dof, n_sweeps, dt, len(tri_edges_azar)


# ============================ ESTADÍSTICA (código nuevo) ============================
def z_score_pareado(a, b):
    """z-score de la diferencia pareada a-b (por semilla), asumiendo la diferencia ~ normal por CLT
    sobre N_SEEDS observaciones independientes (semillas distintas = grafos base distintos)."""
    d = np.asarray(a) - np.asarray(b)
    se = d.std(ddof=1) / np.sqrt(len(d))
    return float(d.mean() / se) if se > 1e-12 else float("inf") if d.mean() != 0 else 0.0


def test_permutacion_signo_pareado(a, b, n_perm, rng, direccion="a<b"):
    """Test de permutación por 'sign-flip' para muestras PAREADAS (mismo criterio de unidad válida
    que cs078_kappaV_permutacion.py: la semilla/grafo-base es la unidad, no cada triángulo suelto).
    H0: no importa el orden del par (a,b) -> cada diferencia d_i=a_i-b_i pudo tener cualquier signo
    con igual probabilidad. Bajo H0 se generan n_perm distribuciones nulas volteando signos al azar y
    se compara el estadístico observado (media de d) contra esa distribución nula.
    direccion='a<b' -> hipótesis pre-registrada de esta tarea: a (control fino / real) aplana MÁS que
    b (referencia), o sea d=a-b es negativo en promedio."""
    d = np.asarray(a) - np.asarray(b)
    obs = d.mean()
    n = len(d)
    signos = rng.choice([-1, 1], size=(n_perm, n))
    dist_nula = (signos * d[None, :]).mean(axis=1)
    if direccion == "a<b":
        p_una_cola = float(np.mean(dist_nula <= obs))
    else:
        p_una_cola = float(np.mean(dist_nula >= obs))
    p_dos_colas = float(np.mean(np.abs(dist_nula) >= abs(obs)))
    return dict(obs_diff=float(obs), p_una_cola=p_una_cola, p_dos_colas=p_dos_colas)


def resumen(nombre, vals):
    v = np.asarray(vals)
    return f"{nombre:<26} n={len(v):>2}  media={v.mean():.4f}  DE={v.std(ddof=1):.4f}  " \
           f"min={v.min():.4f}  max={v.max():.4f}"


# ============================ BATERÍA ============================
def main():
    print("CS083 — FASE IV robustecido: cierre LOCAL vs consenso GLOBAL en el sustrato 4")
    print("=" * 100)
    print(f"N={N} nodos · K={K} · J={J} · J_FACE={J_FACE} · ruido={NOISE} · presupuesto={COMPUTE_BUDGET} "
          f"op-relación/sustrato · N_SEEDS={N_SEEDS} (antes: 5)")
    print("Reusa de cs082_fase4_4sustratos.py: construir_base, dinámica circular, medición de holonomía,")
    print("sustrato 4 real, NULL/SHUFFLED. Código nuevo: control fino 'rewire al azar' + estadística.\n")

    filas = []
    t_batería0 = time.time()
    for seed in SEEDS:
        adj, edges, triangles = construir_base(seed)
        n_e, n_t = len(edges), len(triangles)

        E_real, dof_real, sw_real, dt_real = correr_sustrato_4_2complejo(adj, edges, triangles, seed)
        E_ctrl, dof_ctrl, sw_ctrl, dt_ctrl, n_tri_ctrl = correr_sustrato_4_control_fino(adj, edges, triangles, seed)

        h_real = _holonomia_triangulos(E_real, triangles).mean()
        h_ctrl = _holonomia_triangulos(E_ctrl, triangles).mean()   # OJO: medido sobre los TRIÁNGULOS
                                                                    # VERDADEROS, no sobre los tríos-azar
        E_null = null_de(E_real, seed)
        h_null = _holonomia_triangulos(E_null, triangles).mean()
        E_shuf = shuffled_de(E_real, seed)
        h_shuf = _holonomia_triangulos(E_shuf, triangles).mean()

        filas.append(dict(
            seed=seed, n_edges=n_e, n_tri=n_t,
            dof_real=dof_real, sw_real=sw_real, dt_real=dt_real,
            dof_ctrl=dof_ctrl, sw_ctrl=sw_ctrl, dt_ctrl=dt_ctrl, n_tri_ctrl=n_tri_ctrl,
            h_real=h_real, h_ctrl=h_ctrl, h_null=h_null, h_shuf=h_shuf,
        ))
    dt_batería = time.time() - t_batería0

    # ---------------- CONTROL DE EQUIPARACIÓN entre REAL y CONTROL FINO ----------------
    print("── CONTROL DE EQUIPARACIÓN real vs control fino (deben coincidir salvo el rewire) ──")
    f0 = filas[0]
    print(f"  sustrato 4 REAL      : DoF={f0['dof_real']} sweeps={f0['sw_real']} op-relación_total={f0['dof_real']*f0['sw_real']}")
    print(f"  sustrato 4 CTRL_FINO : DoF={f0['dof_ctrl']} sweeps={f0['sw_ctrl']} op-relación_total={f0['dof_ctrl']*f0['sw_ctrl']} "
          f"(nº tríos cara->arista/sweep = {f0['n_tri_ctrl']}, igual al nº de triángulos reales)")
    coinciden = all(f["dof_real"] == f["dof_ctrl"] and f["sw_real"] == f["sw_ctrl"] for f in filas)
    print(f"  ¿DoF y sweeps coinciden en las {N_SEEDS} semillas?  {'sí' if coinciden else 'NO -- revisar'}\n")

    # ---------------- TABLA POR SEMILLA (primeras 5, para ver el chequeo de reproducibilidad) ----------------
    print("── primeras 5 semillas (deben reproducir cs082: h_real≈0.30, h_null≈1.54, h_shuf≈0.57) ──")
    print(f"  {'seed':>4} {'h_real':>8} {'h_ctrl_fino':>12} {'h_null':>8} {'h_shuf':>8}")
    for f in filas[:5]:
        print(f"  {f['seed']:>4} {f['h_real']:>8.3f} {f['h_ctrl']:>12.3f} {f['h_null']:>8.3f} {f['h_shuf']:>8.3f}")
    print()

    h_real = np.array([f["h_real"] for f in filas])
    h_ctrl = np.array([f["h_ctrl"] for f in filas])
    h_null = np.array([f["h_null"] for f in filas])
    h_shuf = np.array([f["h_shuf"] for f in filas])

    # ---------------- RESUMEN DESCRIPTIVO (N_SEEDS semillas) ----------------
    print(f"── RESUMEN DESCRIPTIVO sobre {N_SEEDS} semillas ──")
    print("  " + resumen("h_REAL (feedback local)", h_real))
    print("  " + resumen("h_CTRL_FINO (rewire azar)", h_ctrl))
    print("  " + resumen("h_NULL (sin dinámica)", h_null))
    print("  " + resumen("h_SHUFFLED (topología rota)", h_shuf))
    print()

    # ---------------- DESCOMPOSICIÓN DEL EFECTO ----------------
    gap_total = h_null.mean() - h_real.mean()          # NULL - REAL: todo el aplanamiento observado
    gap_local = h_ctrl.mean() - h_real.mean()           # CTRL_FINO - REAL: lo que se PIERDE al romper
                                                         # la correspondencia local (= componente LOCAL)
    gap_global = h_null.mean() - h_ctrl.mean()          # NULL - CTRL_FINO: lo que el CTRL_FINO logra
                                                         # igual, SÓLO por tener feedback disperso
                                                         # (= componente de CONSENSO GLOBAL)
    frac_local = gap_local / gap_total if abs(gap_total) > 1e-9 else float("nan")
    frac_global = gap_global / gap_total if abs(gap_total) > 1e-9 else float("nan")
    print("── DESCOMPOSICIÓN: ¿cuánto del aplanamiento total (NULL-REAL) es LOCAL vs GLOBAL? ──")
    print(f"  gap total   (h_NULL - h_REAL)      = {gap_total:+.4f}")
    print(f"  gap local   (h_CTRL_FINO - h_REAL) = {gap_local:+.4f}   -> fracción LOCAL  = {frac_local:+.1%}")
    print(f"  gap global  (h_NULL - h_CTRL_FINO) = {gap_global:+.4f}   -> fracción GLOBAL = {frac_global:+.1%}")
    print("  (fracción LOCAL = cuánto de la caída de holonomía SE PIERDE cuando la cara empuja aristas")
    print("   al azar en vez de su propio borde real; fracción GLOBAL = cuánto de la caída sobrevive")
    print("   igual con feedback disperso al azar, es decir, es 'puro volumen de retroalimentación'.)\n")

    # ---------------- TESTS ESTADÍSTICOS ----------------
    rng_perm = np.random.default_rng(RNG_MASTER_SEED)
    print(f"── TESTS PAREADOS POR SEMILLA (n={N_SEEDS}, unidad = semilla/grafo-base, {N_PERM} permutaciones sign-flip) ──\n")

    print("  [A] REAL vs NULL  (¿el sustrato 4 real separa de NULL con más semillas?)")
    z_rn = z_score_pareado(h_real, h_null)
    t_rn = test_permutacion_signo_pareado(h_real, h_null, N_PERM, rng_perm, direccion="a<b")
    print(f"      z={z_rn:+.2f}   diff_obs(REAL-NULL)={t_rn['obs_diff']:+.4f}   "
          f"p_una_cola(REAL<NULL)={t_rn['p_una_cola']:.5f}   p_dos_colas={t_rn['p_dos_colas']:.5f}\n")

    print("  [B] REAL vs CONTROL_FINO  (*** el test central de esta tarea ***)")
    print("      H1 pre-registrada: si hay cierre LOCAL genuino, REAL debe aplanar MÁS (h menor) que")
    print("      el control fino, que tiene la MISMA cantidad de feedback pero mal-cableado al azar.")
    z_rc = z_score_pareado(h_real, h_ctrl)
    t_rc = test_permutacion_signo_pareado(h_real, h_ctrl, N_PERM, rng_perm, direccion="a<b")
    print(f"      z={z_rc:+.2f}   diff_obs(REAL-CTRL_FINO)={t_rc['obs_diff']:+.4f}   "
          f"p_una_cola(REAL<CTRL_FINO)={t_rc['p_una_cola']:.5f}   p_dos_colas={t_rc['p_dos_colas']:.5f}\n")

    print("  [C] CONTROL_FINO vs NULL  (¿el control fino igual se distingue de no-tener-feedback?)")
    z_cn = z_score_pareado(h_ctrl, h_null)
    t_cn = test_permutacion_signo_pareado(h_ctrl, h_null, N_PERM, rng_perm, direccion="a<b")
    print(f"      z={z_cn:+.2f}   diff_obs(CTRL_FINO-NULL)={t_cn['obs_diff']:+.4f}   "
          f"p_una_cola(CTRL_FINO<NULL)={t_cn['p_una_cola']:.5f}   p_dos_colas={t_cn['p_dos_colas']:.5f}\n")

    print("  [D] REAL vs SHUFFLED  (repite el chequeo original de cs082 con más poder)")
    z_rs = z_score_pareado(h_real, h_shuf)
    t_rs = test_permutacion_signo_pareado(h_real, h_shuf, N_PERM, rng_perm, direccion="a<b")
    print(f"      z={z_rs:+.2f}   diff_obs(REAL-SHUFFLED)={t_rs['obs_diff']:+.4f}   "
          f"p_una_cola(REAL<SHUFFLED)={t_rs['p_una_cola']:.5f}   p_dos_colas={t_rs['p_dos_colas']:.5f}\n")

    print(f"Tiempo total de la batería ({N_SEEDS} semillas x 2 corridas del sustrato 4): {dt_batería:.1f}s")
    print("Fin. Ver FASE4_robustecido_CS.md para la interpretación completa (lectura de Alexis pendiente).")

    return dict(filas=filas, h_real=h_real, h_ctrl=h_ctrl, h_null=h_null, h_shuf=h_shuf,
                gap_total=gap_total, gap_local=gap_local, gap_global=gap_global,
                frac_local=frac_local, frac_global=frac_global,
                tests=dict(real_vs_null=(z_rn, t_rn), real_vs_ctrl=(z_rc, t_rc),
                           ctrl_vs_null=(z_cn, t_cn), real_vs_shuf=(z_rs, t_rs)))


if __name__ == "__main__":
    main()
