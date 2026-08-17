"""
cs073_ley_escala.py — LEY DE ESCALA del puente (DISENO_CS073_puente_PARA_CC.md v3/instrucción de escala).

No basta con "cruzó Jeans a N grande" (vulnerable a "usaron una simulación discreta"). En cambio: correr
una SERIE de escalas N, medir cómo escalan el discriminante REAL-vs-NULL (z de clusters_ligados) y la
razón masa_cluster/M_J_local con N, ajustar una ley de potencia (log-log), y EXTRAPOLAR esa ley medida
al N físico de una nube de Jeans Pop III real (~10^62-10^63 átomos). Si la ley predice razón>=1 ahí, el
discreto es muestreo grueso de un continuo que enciende la estrella -- el mismo argumento de convergencia
de cualquier simulación N-cuerpos cosmológica. La ley se MIDE de la serie, nunca se ajusta a mano para
cruzar -- si la extrapolación no cruza, es dato honesto (mecanismo real pero subcrítico), no se retoca.

COSTO REAL (encontrado en la práctica, no en abstracto): el motor basal tiene una pared O(N_total^2) en
memoria (`Bq` en estado.py, congelado) -- N=4000 átomos H necesitaría nq~24000, Bq~24GB. Además había
contención real en la máquina (proceso .claude-science corriendo en paralelo) que hizo más lento de lo
esperado incluso un solo pool. Por eso: UNA sola extracción basal cara (pool de N_max átomos reales,
f=10, ya validado seguro esta sesión) y la serie se arma por SUBMUESTREO determinista de ese pool -- cada
punto de la serie usa átomos reales genuinos (masa_trio, densidad #23 reales), no inventados.
"""
import json
import numpy as np

from cs073_cierre_holistico import _dinamica_estructura, _z

N_FISICO_POP_III = 1e62   # orden de magnitud bajo (~10^62-10^63 en el diseño); reportamos ambos extremos


def _mejor_razon_jeans(detalle):
    """max(masa_cluster / M_J_local) entre los clusters de una corrida -- 0.0 si no hay clusters."""
    if not detalle:
        return 0.0
    return max(d["masa"] / d["M_J"] for d in detalle)


def correr_serie(masa_pool, dens_pool, N_series, n_null=8, seed_null_base=5000,
                  seed_submuestreo=42, **kw):
    """Para cada N en N_series: submuestrea N átomos REALES del pool (semilla determinista, distinta por
    N para no repetir el mismo subconjunto), corre REAL + n_null NULL con semilla='causal', mide
    n_clusters_ligados (z) y razón masa/M_J máxima (REAL y NULL)."""
    n_pool = len(masa_pool)
    filas = []
    for N in N_series:
        if N > n_pool:
            filas.append(dict(N=N, ok=False, nota=f"N={N} > pool disponible ({n_pool})"))
            continue
        idx = np.random.default_rng(seed_submuestreo + N).choice(n_pool, size=N, replace=False)
        masa_N, dens_N = masa_pool[idx], dens_pool[idx]

        real = _dinamica_estructura(masa_N, dens_N, 1.5, semilla="causal", seed_dens_null=None, **kw)
        if not real.get("ok"):
            filas.append(dict(N=N, ok=False, nota=real.get("nota")))
            continue

        ligados_null, razon_null = [], []
        for i in range(n_null):
            rn = _dinamica_estructura(masa_N, dens_N, 1.5, semilla="causal",
                                       seed_dens_null=seed_null_base + i * 2, **kw)
            if rn.get("ok"):
                ligados_null.append(rn["n_clusters_ligados"])
                razon_null.append(_mejor_razon_jeans(rn.get("detalle")))

        z_ligados = _z(real["n_clusters_ligados"], ligados_null) if ligados_null else None
        razon_real = _mejor_razon_jeans(real.get("detalle"))

        filas.append(dict(N=N, ok=True,
                           n_clusters_ligados_real=real["n_clusters_ligados"],
                           n_clusters_ligados_null_media=round(float(np.mean(ligados_null)), 3) if ligados_null else None,
                           n_clusters_ligados_null_std=round(float(np.std(ligados_null, ddof=1)), 3) if len(ligados_null) > 1 else 0.0,
                           z_ligados=z_ligados,
                           razon_jeans_real=round(razon_real, 6),
                           razon_jeans_null_media=round(float(np.mean(razon_null)), 6) if razon_null else None))
        print(f"  N={N}: {filas[-1]}", flush=True)
    return filas


def ajustar_ley_potencia(N_vals, y_vals):
    """Regresión lineal en log-log: y = A * N^alpha. Devuelve (alpha, log10(A), R2, error_std_alpha).
    Excluye puntos con y<=0 (log indefinido) -- se reportan aparte, no se fuerzan a un piso arbitrario."""
    N_vals = np.asarray(N_vals, float)
    y_vals = np.asarray(y_vals, float)
    validos = y_vals > 0
    n_validos = int(validos.sum())
    if n_validos < 2:
        return dict(ok=False, nota=f"sólo {n_validos} puntos con y>0 -- no alcanza para ajustar", n_puntos=n_validos)
    x = np.log10(N_vals[validos])
    y = np.log10(y_vals[validos])
    n = len(x)
    xm, ym = x.mean(), y.mean()
    Sxx = np.sum((x - xm) ** 2)
    Sxy = np.sum((x - xm) * (y - ym))
    alpha = Sxy / Sxx
    logA = ym - alpha * xm
    y_pred = alpha * x + logA
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - ym) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    if n > 2:
        s2 = ss_res / (n - 2)
        error_alpha = float(np.sqrt(s2 / Sxx))
    else:
        error_alpha = float("nan")
    return dict(ok=True, n_puntos=n, alpha=round(float(alpha), 4), error_alpha=round(error_alpha, 4),
                log10_A=round(float(logA), 4), R2=round(float(r2), 4))


def extrapolar(ajuste, N_objetivo):
    """y(N_objetivo) = A * N_objetivo^alpha, reportado en LOG10 (no en el número crudo -- a ~60 órdenes
    de magnitud de salto, 10**log_y desborda el float64 y da Infinity, que no informa nada). La banda de
    incertidumbre propaga error_alpha sobre el salto en log10(N): un error minúsculo en alpha, propagado
    ~60 órdenes de magnitud, puede ser enorme. No se esconde -- se reporta el propio tamaño de la banda."""
    if not ajuste.get("ok"):
        return dict(ok=False, nota="sin ajuste válido")
    alpha, logA, err = ajuste["alpha"], ajuste["log10_A"], ajuste["error_alpha"]
    logN = np.log10(N_objetivo)
    log10_y_central = alpha * logN + logA
    if np.isfinite(err):
        delta_log10 = err * abs(logN)   # propagación simple: Δ(log y) = |Δalpha| * |log N|
        log10_y_lo = log10_y_central - delta_log10
        log10_y_hi = log10_y_central + delta_log10
    else:
        delta_log10 = log10_y_lo = log10_y_hi = None
    return dict(ok=True, N_objetivo=N_objetivo,
                log10_y_central=round(float(log10_y_central), 3),
                log10_y_lo=round(float(log10_y_lo), 3) if log10_y_lo is not None else None,
                log10_y_hi=round(float(log10_y_hi), 3) if log10_y_hi is not None else None,
                delta_log10=round(float(delta_log10), 3) if delta_log10 is not None else None,
                cruza_umbral_1=bool(log10_y_lo is not None and log10_y_lo > 0),
                nota_umbral="cruza_umbral_1=True sólo si TODA la banda de incertidumbre (log10_y_lo) queda > 0 (razón>1, Jeans cruzado incluso en el escenario más conservador)")
