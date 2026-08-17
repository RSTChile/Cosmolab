"""
p02b_gravedad_general.py — PIEZA #2b: GRAVEDAD GENERAL (régimen MÉTRICO, post-Paso A).

Qué hace, en simple: `p02_gravedad.py` (Bgrav) es la gravedad RELACIONAL-CUÁNTICA -- liga átomos por
umbral de proximidad TÉRMICA (un escalar 1D), régimen pre-métrico, sin posición. Dos prototipos de CS
(ver CS073_prototipo_estructura_hallazgo_CS.md) probaron que ese régimen NUNCA fragmenta en estructuras
(ni global ni local-térmico: REAL=NULL exacto, porque un escalar 1D no codifica vecindad 3D).

Este módulo es el régimen NUEVO -- gravedad GENERAL, sobre POSICIONES 3D reales, nunca sobre
temperatura. Contradicción cazada y corregida el 19-jul-2026 (ver RESPUESTA_CC_contradiccion_
posiciones_CS.md y DISENO_CS073_transicion_gravedad_general_PARA_CC.md v2): el Paso A (desplegar la
métrica fosilizada como posiciones) es PRERREQUISITO EXPLÍCITO, no implícito -- es lo que DEFINE la
frontera entre gravedad relacional y gravedad general.

MÓDULO NUEVO, no toca `p02_gravedad.py` ni `Bgrav` ni ningún archivo existente (instrucción: "módulo
nuevo para no tocar lo que ya funciona"). Reimporta piezas ya validadas de proceso_sucesivo.py
(_malla_causal, dimension_acoplada) en vez de reimplementarlas -- reimplementar con una variación
sutil sería el tipo de deriva que puede colar un Shannon sin que nadie lo note.

ESTADO DE ESTE ARCHIVO (19-jul-2026): sólo PASO A implementado (despliegue de posiciones + gate +
verificación de la salvedad A.4 contra un NULL). PASO B (gravedad general ∝ m_i·m_j sobre esas
posiciones + masa de Jeans + colapso) NO está aquí todavía -- sus detalles algorítmicos (regla exacta
de acumulación, radio de interacción, criterio de colapso) no estaban especificados con la precisión
que exige el guardián G-PARAMETROS-ESTRUCTURALES, así que se corre primero el Paso A solo, se reporta
el resultado honesto (incluida la salvedad A.4), y se coordina el Paso B con ese dato en la mesa.
"""
import collections
import numpy as np

from cs072_modulos.nucleo import corre
from cs072_modulos.proceso_sucesivo import _malla_causal, dimension_acoplada


def _bfs_todas(adj, m):
    """Distancias de grafo (nº de saltos) entre TODOS los pares de nodos 0..m-1. O(N^2) BFS -- válido
    a escala de prototipo/smoke (cientos de átomos). A escala grande (10^3-10^4) hace falta
    landmark-MDS (pendiente, fuera de este Paso A: se evalúa el costo antes de escalar)."""
    D = np.full((m, m), np.inf)
    for s in range(m):
        d = {s: 0}
        q = collections.deque([s])
        while q:
            u = q.popleft()
            for v in sorted(adj[u]):
                if v not in d:
                    d[v] = d[u] + 1
                    q.append(v)
        for v, dist in d.items():
            D[s, v] = dist
    return D


def _mds_clasico(Dmat, dims=3):
    """MDS clásico (Torgerson): doble centrado de D^2 + autodescomposición. Sin dependencias nuevas
    (no hay sklearn en el venv del proyecto; scipy no trae MDS). Devuelve también cuántas dimensiones
    hacen falta para explicar el 90% de la varianza -- el mismo tipo de diagnóstico que ya dio "63
    dims" en el prototipo de CS073 cuando el grafo era el hub de Bgrav (dato, no bug)."""
    m = Dmat.shape[0]
    D2 = Dmat ** 2
    J = np.eye(m) - np.ones((m, m)) / m
    B = -0.5 * J @ D2 @ J
    eigval, eigvec = np.linalg.eigh(B)               # ascendente
    orden = np.argsort(eigval)[::-1]
    eigval = eigval[orden]
    eigvec = eigvec[:, orden]
    pos = np.clip(eigval, 0, None)
    total = pos.sum()
    coords_full = eigvec * np.sqrt(pos)[None, :]
    coords = coords_full[:, :dims]
    var_dims = float(pos[:dims].sum() / total) if total > 0 else 0.0
    acumulada = np.cumsum(pos) / total if total > 0 else np.array([])
    dims_90 = int(np.searchsorted(acumulada, 0.90) + 1) if len(acumulada) else None
    return dict(coords=coords, varianza_explicada_dims=round(var_dims, 4), dims_para_90pct=dims_90)


def _ejes_desde_densidad(dens_arr, D, seed0):
    """MISMA receta que dimension_acoplada() (proceso_sucesivo.py, línea ~98-99): D ejes = D barajados
    deterministas distintos del mismo escalar de densidad real. Duplicada aquí a propósito (no se
    factoriza en proceso_sucesivo.py) para no tocar ese archivo -- instrucción explícita del director."""
    n = len(dens_arr)
    return np.column_stack([dens_arr[np.random.default_rng(seed0 + eje).permutation(n)] for eje in range(D)])


def _completar_infinitos(Dmat):
    """Si la malla causal deja componentes desconectadas, BFS da inf. No se descartan esos átomos
    (sería podar a mano); se penaliza con 2x la distancia finita máxima -- 'están, pero lejos' -- y se
    reporta cuántos pares quedaron así, para que la desconexión sea visible, no invisible."""
    finitos = np.isfinite(Dmat)
    n_inf = int((~finitos).sum())
    if n_inf == 0:
        return Dmat, 0
    max_fin = float(Dmat[finitos].max()) if finitos.any() else 1.0
    return np.where(finitos, Dmat, max_fin * 2.0), n_inf


def _construir_embedding(dens_arr, D, k, seed_ejes, dims):
    """Un embedding completo (ejes -> malla causal -> distancias -> MDS) a partir de un vector de
    densidad ya decidido (real o barajado). Factorizado para no duplicar la secuencia entre el REAL y
    cada NULL de la distribución -- una sola fuente de verdad del procedimiento."""
    m = len(dens_arr)
    V = _ejes_desde_densidad(dens_arr, D, seed_ejes)
    adj, m_, _arranque = _malla_causal(V, min(k, m - 1))
    Dmat, n_inf = _completar_infinitos(_bfs_todas(adj, m_))
    mds = _mds_clasico(Dmat, dims=dims)
    mds["pares_desconectados"] = n_inf
    return mds


def _z(real_val, dist):
    """z-score de real_val contra la MEDIA/DESVÍO de una distribución de NULLs (no un solo sorteo).
    Si la distribución de NULLs tiene desvío 0 (todos los nulls idénticos), no hay z definido -- se
    reporta None en vez de dividir por cero o inventar un valor."""
    arr = np.asarray(dist, float)
    mu, sd = float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    if sd == 0.0:
        return None
    return round(float((real_val - mu) / sd), 3)


def desplegar_posiciones(nq, naq, ne, npos, D_distinciones=3, k=4, dims=3,
                          tasa_expansion=0.02, pasos=150, amp_rugosidad=1.5,
                          apagar=frozenset(), n_null=8, seed_null_base=3000, z_umbral=2.0,
                          gate_nq=300, gate_naq=210, gate_ne=100, gate_npos=70):
    """PASO A -- despliegue de la métrica fosilizada como posiciones 3D. Prerrequisito EXPLÍCITO de la
    gravedad general (v2 del diseño). Fuente = malla causal (_malla_causal, la que escapa del
    mundo-pequeño con la expansión de dos fases), NO Bgrav (hub por construcción -- prototipo CS073).

    GATE: dimension_acoplada() debe dar dim_efectiva finito. Si el gas sigue siendo mundo-pequeño, NO
    se despliega -- se aborta y se reporta por qué, nunca se fuerza una geometría 3D que no emergió
    (G-NO-3D-FORZADO). El gate se corre con (gate_nq,gate_naq,gate_ne,gate_npos) -- por DEFECTO la
    línea base ya validada (300,210,100,70, ~20s, dim_efectiva=2.05) -- NO con el (nq,naq,ne,npos)
    grande del despliegue real. Razón encontrada en la práctica, no en abstracto: dimension_acoplada()
    barre internamente escalas=(1,2,3,4) multiplicando su nq -- pasarle un nq ya grande como base
    dispara una corrida a 4x ESE tamaño sólo para el gate, que fue justo lo que colgó el primer sondeo
    de escala (Bq en estado.py es O(N_total^2) en memoria, N_total=nq+naq+ne+npos). El gate mide una
    propiedad del RÉGIMEN (¿esta configuración de D/tasa_expansion escapa del mundo-pequeño en
    principio?), no depende de cuántos átomos reales se van a embeber -- separarlo es honesto, no un
    atajo: mismo criterio ya usado y validado, sólo que barato.

    SALVEDAD A.4, contra una DISTRIBUCIÓN de NULLs (no un solo barajado -- corrección tras el primer
    smoke test a 50 átomos, donde un solo NULL no alcanzaba para distinguir señal de ruido): se corren
    `n_null` sorteos independientes del mismo campo de densidad #23 barajado (misma distribución,
    coherencia destruida), y se compara el REAL contra la media/desvío de esa distribución (z-score).
    z_umbral=2.0 (convención de 2 sigma, no ajustada a que el resultado "salga bien" -- el z crudo se
    reporta siempre, el umbral sólo decide el booleano de conveniencia)."""
    gate = dimension_acoplada(gate_nq, gate_naq, gate_ne, gate_npos, D_distinciones, k=k,
                               tasa_expansion=tasa_expansion, amp_rugosidad=amp_rugosidad, apagar=apagar)
    if gate.get("dim_efectiva") is None:
        return dict(gate_ok=False, gate=gate,
                    nota="mundo-pequeño: dimension_acoplada=None -> no se despliega (correcto, no se fuerza)")

    obs, e = corre(nq, naq, ne, npos, tasa_expansion=tasa_expansion, pasos=pasos,
                    amp_rugosidad=amp_rugosidad, apagar=apagar, devolver_estado=True)
    geo = obs.get("geometria") or {}
    dens_atomos = geo.get("densidades_atomos") or []
    if len(dens_atomos) < 8:
        return dict(gate_ok=False, gate=gate,
                    nota=f"sólo {len(dens_atomos)} átomos reales (<8): sin geometría suficiente que desplegar")

    dens = np.array(dens_atomos, float)
    m = len(dens)

    # --- REAL ---
    real = _construir_embedding(dens, D_distinciones, k, seed_ejes=2000, dims=dims)

    # --- DISTRIBUCIÓN DE NULLs: n_null sorteos independientes del campo #23 barajado ---
    null_var, null_dims90 = [], []
    for i in range(n_null):
        dens_barajada = dens[np.random.default_rng(seed_null_base + i).permutation(m)]
        mds_null = _construir_embedding(dens_barajada, D_distinciones, k, seed_ejes=4000 + i * 100, dims=dims)
        null_var.append(mds_null["varianza_explicada_dims"])
        null_dims90.append(mds_null["dims_para_90pct"])

    z_var = _z(real["varianza_explicada_dims"], null_var)
    z_dims90 = _z(real["dims_para_90pct"], null_dims90)
    distinguible = (z_var is not None and abs(z_var) >= z_umbral) or \
                    (z_dims90 is not None and abs(z_dims90) >= z_umbral)

    return dict(
        gate_ok=True, gate=gate, n_atomos=m, n_null=n_null,
        pares_desconectados=real["pares_desconectados"],
        coords=real["coords"], varianza_explicada_dims=real["varianza_explicada_dims"],
        dims_para_90pct=real["dims_para_90pct"],
        null_varianza_explicada_dims_media=round(float(np.mean(null_var)), 4),
        null_varianza_explicada_dims_std=round(float(np.std(null_var, ddof=1)), 4) if n_null > 1 else 0.0,
        null_dims_para_90pct_media=round(float(np.mean(null_dims90)), 2),
        null_dims_para_90pct_std=round(float(np.std(null_dims90, ddof=1)), 2) if n_null > 1 else 0.0,
        z_varianza_explicada_dims=z_var, z_dims_para_90pct=z_dims90, z_umbral=z_umbral,
        a4_referente_fisico_distinguible_de_null=bool(distinguible),
        nota=("Paso A completo, NULL como distribución (n_null sorteos). Si a4_...=False, el REAL no "
              "se distingue de la distribución de barajados (|z|<umbral) -> dato para re-coordinar, "
              "Paso B NO procede sobre estas coordenadas."),
    )
