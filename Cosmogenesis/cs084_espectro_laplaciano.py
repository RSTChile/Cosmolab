"""
CS084 — ¿SUENA distinto el tejido real? Espectro del laplaciano de grafo (L = D - A)
=====================================================================================
Origen de la pregunta: un afiche que Alexis vio comparaba tres ecuaciones que comparten el MISMO
operador espacial (el laplaciano ∇²) y sólo difieren en el orden de la derivada temporal:
  - Laplace:  ∇²u = 0                (equilibrio estático)
  - Calor:    ∇²u = ∂u/∂t            (difusión, irreversible)
  - Onda:     ∇²u = ∂²u/∂t²          (oscilación, reversible)
El afiche menciona que la ecuación de onda predice las frecuencias resonantes de un tambor -- y esas
frecuencias son, matemáticamente, los VALORES PROPIOS (eigenvalues) del laplaciano del dominio. Éste es
el problema clásico de Kac (1966): "¿se puede oír la forma de un tambor?". La idea de Alexis: si el
DIÁMETRO (Fase 3, `cs080_renormalizacion.py`) no distinguió el tejido real de sus controles NULL, tal
vez el ESPECTRO COMPLETO del laplaciano -- que lleva mucha más información que un solo número -- sí lo
haga. Este script hace exactamente eso, sobre el MISMO tejido y los MISMOS controles que usó Fase 3.

ANTECEDENTE (leído antes de escribir código, no se repite el trabajo):
  - `FASE3_renormalizacion_resultado_CS.md`: la pendiente diam-vs-N_b bajo agrupamiento (coarse-graining)
    NO separó local (real) de local_barajado (NULL 1: mismo grado, sin criterio de localidad) ni de
    er_null (NULL 2, piso: Erdős-Rényi puro) -- 0.376 vs 0.420 vs 0.406, solapadas. El tejido real
    incluso se FRAGMENTA más rápido que el ruido al agrupar (d_s cae de ~4.1 a ~0.8-1.0 en b=16-32).
  - `cs080_renormalizacion.py`: motor reusado TAL CUAL -- `construir_sustrato(N, seed, arm)` construye
    los tres brazos (local / local_barajado / er_null) con el motor `proceso066` de CS066 (k_local=6
    FIJO, el punto más favorable a la localidad según `cs066conf_exponentes.md`) y el mismo generador
    Erdős-Rényi de control. Se importa y se usa sin tocar una línea del archivo original.

QUÉ HACE ESTE SCRIPT (3 diagnósticos espectrales, mismo tejido/controles, ≥5 semillas por brazo):
  1. FORMA de la densidad espectral: histograma de TODOS los eigenvalues de L, real vs barajado vs ER.
     ¿la "silueta sonora" del tejido real se ve distinta más allá de lo que ya vio el diámetro?
  2. DIMENSIÓN ESPECTRAL por núcleo de calor: Tr(e^{-tL}) ~ t^{-d_s/2} -- literalmente resolver la
     ecuación de CALOR del afiche sobre el grafo (el operador es el mismo L). d_s(t) = -2 dlogTr/dlogt.
  3. ESTADÍSTICA DE ESPACIADO DE NIVELES (Poisson vs Wigner-Dyson/GOE): tras "desplegar" (unfold) el
     espectro para que la densidad local sea ~uniforme, ¿los espaciados entre eigenvalues consecutivos
     se comportan como ruido puro (Poisson, exponencial, SIN repulsión de niveles) o como sistemas con
     correlación/estructura genuina (Wigner-Dyson, forma de campana, CON repulsión de niveles cerca de
     cero)? Es el diagnóstico anti-Shannon central: ¿"suena" a estructura o a ruido puro?

MÉTODO -- diagonalización DENSA completa (no truncada): se midió el costo real antes de decidir la N
(no se asumió): a N=8000 (la N EXACTA de Fase 3, para comparar manzana con manzana) la diagonalización
completa de L (scipy.linalg.eigh, denso) tardó ~65s por matriz en esta máquina, y la construcción del
tejido (motor `proceso066` completo) ~5-15s -- 15 matrices (3 brazos × 5 semillas) entran cómodas en el
presupuesto de tiempo. Se usa DENSA (no `eigsh` truncado) porque el diagnóstico 3 (espaciado de niveles)
necesita el espectro COMPLETO en el bulk, no sólo los extremos.

No se toca `cs066_localidad_geometrogenesis.py` ni `cs080_renormalizacion.py` -- sólo import. No se
declara cierre ni veredicto: se reportan números, la lectura final es de Alexis.

Codea/ejecuta: CC (Claude).
"""
from __future__ import annotations
import os, sys, time, csv, math
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.linalg import eigh
from scipy.stats import kstest

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)

import cs080_renormalizacion as C80   # construir_sustrato(N, seed, arm) -- motor proceso066 + 2 controles, SIN tocar
import cs064_smoke as SM              # adj_sparse(adj, N) -- adyacencia dispersa, SIN tocar

N_NODOS = int(os.environ.get("CS084_N", 8000))         # misma N que Fase 3 (cs080), para comparar directo
SEEDS   = [int(x) for x in os.environ.get("CS084_SEEDS", "84100,84200,84300,84400,84500").split(",")]
ARMS    = ("local", "local_barajado", "er_null")
OUT_CSV = os.environ.get("CS084_OUT_CSV", os.path.join(_HERE, "cs084_espectro_laplaciano.csv"))
OUT_NPZ = os.environ.get("CS084_OUT_NPZ", os.path.join(_HERE, "cs084_espectros_crudos.npz"))
FIG_DIR = _HERE

# grilla de t para el núcleo de calor Tr(e^{-tL}) ~ t^{-d_s/2} (log-espaciada, cubre corto y largo alcance)
T_GRID = np.logspace(-2.5, 1.5, 60)


# ============================ CONSTRUCCIÓN: laplaciano denso L=D-A del tejido de un brazo/semilla ============================
def laplaciano_denso(N, seed, arm):
    """Reusa construir_sustrato de cs080 (motor proceso066 + controles, SIN tocar). Devuelve:
    eigenvalues ORDENADOS de L=D-A (denso), nº de componentes conexas, fracción de la componente gigante."""
    adj, V = C80.construir_sustrato(N, seed, arm)
    A = SM.adj_sparse(adj, N)
    n_comp, labels = connected_components(A, directed=False)
    giant = float(np.max(np.bincount(labels))) / N
    deg = np.asarray(A.sum(axis=1)).ravel()
    L = sparse.diags(deg) - A
    Ld = L.toarray()
    w = eigh(Ld, eigvals_only=True)
    w = np.clip(w, 0.0, None)          # el laplaciano es semidefinido positivo; recorta ruido numérico <0
    w.sort()
    return w, int(n_comp), giant


# ============================ DIAGNÓSTICO 2: dimensión espectral por traza del núcleo de calor ============================
def dimension_espectral(eigvals, t_grid=T_GRID):
    """Tr(e^{-tL}) = sum_i exp(-t*lambda_i) -- literalmente la solución de la ecuación de CALOR sobre el
    grafo (el mismo operador L que en Laplace/onda, sólo cambia el orden de la derivada temporal, como
    en el afiche). d_s(t) = -2 * d(log Tr)/d(log t): pendiente local en log-log. Un "d_s(t) plano" en
    algún rango de t es la firma de una dimensión efectiva estable en esa escala (análoga a la d_s por
    crecimiento de bola que ya usó Fase 3, pero calculada por un camino totalmente distinto e
    independiente: difusión en vez de conteo de vecinos)."""
    logt = np.log(t_grid)
    tr = np.array([np.sum(np.exp(-t * eigvals)) for t in t_grid])
    logtr = np.log(np.maximum(tr, 1e-300))
    d_s = -2.0 * np.gradient(logtr, logt)
    return d_s, tr


# ============================ DIAGNÓSTICO 3: espaciado de niveles (unfolding local + Poisson vs GOE) ============================
def unfolding_local(eigvals, n_comp, ventana=51):
    """'Despliega' (unfold) el espectro para poder comparar espaciados a escalas donde la densidad de
    niveles varía. Procedimiento (declarado, sin ajuste oculto):
      1. Se descartan los `n_comp` eigenvalues triviales en 0 (uno por componente conexa -- no son
         'niveles físicos', son el número de piezas separadas del grafo).
      2. Se toma el 80% central del resto (se recorta 10% en cada borde, donde la densidad espectral es
         baja y el 'despliegue' es ruidoso -- práctica estándar en teoría de matrices aleatorias).
      3. Espaciado bruto d_i = lambda_(i+1) - lambda_i. Se normaliza cada d_i por el espaciado medio
         LOCAL (promedio móvil de ventana `ventana`, no un ajuste polinómico global -- más robusto y
         más simple de auditar) -- 'unfolding local', válido cuando la densidad varía suavemente dentro
         de la ventana. El resultado s_i tiene media ~1 por construcción, comparable directo contra las
         distribuciones de referencia Poisson (media 1) y Wigner-Dyson/GOE (media 1).
    Devuelve el array de espaciados normalizados s_i."""
    ev = eigvals[n_comp:]                              # descarta triviales en 0
    lo, hi = int(0.10 * len(ev)), int(0.90 * len(ev))
    ev = ev[lo:hi]
    if len(ev) < ventana * 2:
        return np.array([])
    d = np.diff(ev)
    pad = ventana // 2
    d_pad = np.pad(d, pad, mode="reflect")
    kernel = np.ones(ventana) / ventana
    local_mean = np.convolve(d_pad, kernel, mode="valid")[: len(d)]
    local_mean = np.maximum(local_mean, 1e-300)
    return d / local_mean


def cdf_goe(s):
    """CDF de la distribución de Wigner-Dyson/GOE (surmise de Wigner): pdf(s) = (pi/2) s exp(-pi s^2/4).
    Es una Rayleigh con sigma^2=2/pi; su media es 1 (igual que Poisson-unfolded), pero con REPULSIÓN de
    niveles: pdf(0)=0 (nunca hay dos niveles pegados), a diferencia de Poisson donde pdf(0)=1 (máxima
    probabilidad de espaciado cero -- sin correlación)."""
    return 1.0 - np.exp(-np.pi * np.asarray(s) ** 2 / 4.0)


def estadisticas_espaciado(s):
    """Resume una muestra de espaciados normalizados contra las dos hipótesis de referencia:
    Poisson (ruido puro, sin correlación) vs GOE/Wigner-Dyson (estructura/correlación genuina)."""
    if len(s) < 20:
        return dict(n=len(s), mean_s2=float("nan"), ks_D_poisson=float("nan"), ks_p_poisson=float("nan"),
                    ks_D_goe=float("nan"), ks_p_goe=float("nan"), frac_s_lt_02=float("nan"))
    ks_p = kstest(s, "expon")                # Poisson-unfolded = exponencial de media 1 (scale=1 default)
    ks_g = kstest(s, cdf_goe)
    return dict(
        n=len(s),
        mean_s2=float(np.mean(s ** 2)),       # Poisson teórico=2.0 ; GOE teórico=4/pi≈1.273
        ks_D_poisson=float(ks_p.statistic), ks_p_poisson=float(ks_p.pvalue),
        ks_D_goe=float(ks_g.statistic), ks_p_goe=float(ks_g.pvalue),
        frac_s_lt_02=float(np.mean(s < 0.2)),  # Poisson teórico≈0.181 ; GOE teórico≈0.031 (repulsión)
    )


# ============================ CORRIDA COMPLETA ============================
def main():
    print("=" * 100, flush=True)
    print("CS084 — ESPECTRO DEL LAPLACIANO DE GRAFO (¿se oye la forma del tejido de CS066?)", flush=True)
    print(f"N={N_NODOS}  semillas={SEEDS}  brazos={ARMS}", flush=True)
    print("=" * 100, flush=True)
    t0 = time.time()

    filas = []
    crudos = {}   # para el .npz: clave "arm_seed" -> eigvals

    for arm in ARMS:
        for seed in SEEDS:
            ta = time.time()
            eigvals, n_comp, giant = laplaciano_denso(N_NODOS, seed, arm)
            crudos[f"{arm}_{seed}"] = eigvals

            lam2 = float(eigvals[n_comp]) if n_comp < len(eigvals) else float("nan")   # algebraic connectivity
            lam_max = float(eigvals[-1])

            d_s_curve, tr_curve = dimension_espectral(eigvals)

            s = unfolding_local(eigvals, n_comp)
            stats_esp = estadisticas_espaciado(s)

            fila = dict(seed=seed, arm=arm, N=N_NODOS, n_componentes=n_comp, giant_frac=round(giant, 4),
                        lambda2=round(lam2, 6), lambda_max=round(lam_max, 3),
                        mean_eig=round(float(np.mean(eigvals)), 4), std_eig=round(float(np.std(eigvals)), 4))
            # d_s(t) en 4 puntos representativos del rango (corto/medio/largo alcance)
            for tt in (0.05, 0.2, 1.0, 5.0):
                j = int(np.argmin(np.abs(T_GRID - tt)))
                fila[f"d_s_t{tt}"] = round(float(d_s_curve[j]), 3)
            fila.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats_esp.items()})
            filas.append(fila)

            print(f"  [{arm:<15}] seed={seed}  n_comp={n_comp:<4} giant={giant:.3f}  "
                  f"lambda2={lam2:.5f}  lambda_max={lam_max:.2f}  d_s(t=1.0)={fila['d_s_t1.0']}  "
                  f"<s^2>={stats_esp['mean_s2']:.3f}  KS_Poiss={stats_esp['ks_D_poisson']:.3f}  "
                  f"KS_GOE={stats_esp['ks_D_goe']:.3f}  ({time.time()-ta:.1f}s)", flush=True)

    # -------- guarda CSV resumen + npz con espectros crudos (para graficar / re-analizar sin recomputar) --------
    campos = list(filas[0].keys())
    with open(OUT_CSV, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=campos)
        wr.writeheader()
        for fila in filas:
            wr.writerow(fila)
    np.savez_compressed(OUT_NPZ, t_grid=T_GRID, **crudos)
    print(f"\nCSV -> {OUT_CSV}", flush=True)
    print(f"NPZ (espectros crudos) -> {OUT_NPZ}", flush=True)

    # -------- gráficos --------
    try:
        _graficos(filas, crudos)
    except Exception as e:
        print(f"(aviso: gráficos fallaron, no crítico: {e})", flush=True)

    print(f"\nCOMPLETO en {(time.time()-t0)/60:.1f} min", flush=True)


# ============================ GRÁFICOS (3 figuras, una por diagnóstico) ============================
def _graficos(filas, crudos):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colores = {"local": "#c0392b", "local_barajado": "#2980b9", "er_null": "#7f8c8d"}
    etiquetas = {"local": "real (local)", "local_barajado": "NULL 1: barajado", "er_null": "NULL 2: Erdős-Rényi"}

    # --- Figura 1: forma de la densidad espectral (histograma normalizado, promedio de semillas) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, max(np.max(v) for v in crudos.values()) * 1.02, 80)
    for arm in ARMS:
        todos = np.concatenate([crudos[f"{arm}_{s}"] for s in SEEDS])
        ax.hist(todos, bins=bins, density=True, histtype="step", lw=2, color=colores[arm], label=etiquetas[arm])
    ax.set_xlabel("eigenvalue de L = D - A")
    ax.set_ylabel("densidad espectral (normalizada)")
    ax.set_title(f"CS084 — forma del espectro del laplaciano (N={N_NODOS}, {len(SEEDS)} semillas pooled)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "cs084_forma_espectral.png"), dpi=130)
    plt.close(fig)

    # --- Figura 2: dimensión espectral d_s(t) vs t (media ± banda de semillas) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for arm in ARMS:
        curvas = np.array([dimension_espectral(crudos[f"{arm}_{s}"])[0] for s in SEEDS])
        med, sd = curvas.mean(axis=0), curvas.std(axis=0)
        ax.plot(T_GRID, med, color=colores[arm], lw=2, label=etiquetas[arm])
        ax.fill_between(T_GRID, med - sd, med + sd, color=colores[arm], alpha=0.15)
    ax.set_xscale("log")
    ax.set_xlabel("t (tiempo de difusión del núcleo de calor)")
    ax.set_ylabel("d_s(t) = -2 d(logTr)/d(logt)")
    ax.set_title("CS084 — dimensión espectral vía traza del núcleo de calor")
    ax.axhline(3, color="k", ls=":", lw=1, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "cs084_dimension_espectral.png"), dpi=130)
    plt.close(fig)

    # --- Figura 3: espaciado de niveles (histograma pooled vs Poisson vs GOE) ---
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    s_grid = np.linspace(0, 4, 200)
    poisson_pdf = np.exp(-s_grid)
    goe_pdf = (np.pi / 2) * s_grid * np.exp(-np.pi * s_grid ** 2 / 4)
    for ax, arm in zip(axs, ARMS):
        s_pool = []
        # reconstruye espaciados desde los eigenvalues crudos, usando el n_componentes ya guardado en `filas`
        # (evita recomputar la construcción del tejido / la diagonalización, que ya se hizo una vez arriba)
        for f in filas:
            if f["arm"] != arm:
                continue
            ev = crudos[f"{arm}_{f['seed']}"]
            s_pool.append(unfolding_local(ev, f["n_componentes"]))
        s_pool = np.concatenate(s_pool) if s_pool else np.array([])
        ax.hist(s_pool, bins=40, range=(0, 4), density=True, color=colores[arm], alpha=0.55,
                 label="espaciados (pooled)")
        ax.plot(s_grid, poisson_pdf, "k--", lw=1.5, label="Poisson (ruido puro)")
        ax.plot(s_grid, goe_pdf, "k-", lw=1.5, label="Wigner-Dyson/GOE (estructura)")
        ax.set_title(etiquetas[arm])
        ax.set_xlabel("s (espaciado normalizado)")
    axs[0].set_ylabel("densidad de probabilidad")
    axs[0].legend(fontsize=8)
    fig.suptitle("CS084 — estadística de espaciado de niveles: ¿ruido (Poisson) o estructura (GOE)?")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "cs084_espaciado_niveles.png"), dpi=130)
    plt.close(fig)

    print("gráficos -> cs084_forma_espectral.png, cs084_dimension_espectral.png, cs084_espaciado_niveles.png",
          flush=True)


if __name__ == "__main__":
    main()
