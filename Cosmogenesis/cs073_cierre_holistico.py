"""
cs073_cierre_holistico.py — EXPERIMENTO DE CIERRE CS073: átomo -> primera estrella, ejecución HOLÍSTICA.

Orquestador (no toca ningún archivo existente). Sigue INSTRUCCION_CC_cierre_holistico.md v3:
  REGLA 1 -- todos los subsistemas nuevos (CDM, gravedad general, expansión, presión+enfriamiento H2)
             actúan JUNTOS, en el mismo bucle temporal, cada paso. No se corre uno y luego otro.
  REGLA 2 -- se corre el TODO primero; si falla/da NaN, se aíslan módulos (los interruptores de Regla 3)
             recién ENTONCES para depurar -- no se valida cada pieza por separado de antemano.
  REGLA 3 -- cada pieza nueva es su propio módulo con interruptor on/off (p_gravedad_general,
             p_enfriamiento_H2, p_materia_oscura_halo, p_expansion), importados aquí, no reimplementados.

Sobre el motor basal YA VALIDADO (S>0 -> átomos H/He) se despliega el escenario 3D (D=3, dimensión
fosilizada -- adjudicación Q3) y se corre la competencia gravedad-general vs presión vs expansión, con
enfriamiento H2 como el canal que permite fragmentar. Observable pre-registrado: nº de estructuras
ligadas SEPARADAS que superan el umbral de Jeans, REAL vs NULL (campo #23 barajado), con z-score sobre
varias semillas de NULL. Convenciones adimensionales explícitas en cada pieza (G_ADIM=1, softening,
b_FoF=0.2 estándar) -- ninguna ajustada para que el resultado salga en una dirección.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from cs072_modulos.nucleo import corre
from cs072_modulos.piezas.p_expansion import Expansion
from cs072_modulos.piezas.p_gravedad_general import GravedadGeneral, posiciones_escenario
from cs072_modulos.piezas.p_materia_oscura_halo import MateriaOscuraHalo
from cs072_modulos.piezas.p_enfriamiento_H2 import EnfriamientoH2
from cs072_modulos.piezas.p_semilla_causal import malla_causal_atomos, layout_resortes, barajar_aristas

T0 = 3.0
TASA_EXPANSION = 0.02


def _T_reloj(step):
    """MISMO reloj de enfriamiento que nucleo.Estado.enfria -- ninguna ley nueva."""
    return T0 / np.sqrt(1.0 + (TASA_EXPANSION * 50.0) * step)


def _extraer_bariones(nq, naq, ne, npos, pasos_basal, amp_rugosidad):
    """Corre el motor basal YA VALIDADO una vez; devuelve masa y densidad #23 reales de los átomos
    (H+He). No se re-deriva nada del Modelo Estándar -- se reutiliza el resultado."""
    obs, e = corre(nq, naq, ne, npos, tasa_expansion=TASA_EXPANSION, pasos=pasos_basal,
                    amp_rugosidad=amp_rugosidad, devolver_estado=True)
    atomos_H = [n for (n, _) in e.Bem]
    masa = np.array([e.masa_trio.get(a, 1.0) for a in atomos_H], float)
    densidad = np.array([e.densidad[a] for a in atomos_H], float)
    return masa, densidad, obs


def _fof(pos, linking_length, min_miembros=5):
    """Friends-of-friends vía pares dentro de linking_length (cKDTree) + componentes conexas
    (scipy.sparse.csgraph) -- el estándar de campo, no un algoritmo inventado para este experimento."""
    n = len(pos)
    if n < min_miembros:
        return []
    tree = cKDTree(pos)
    pares = list(tree.query_pairs(r=linking_length))
    if not pares:
        return []
    filas = [p[0] for p in pares] + [p[1] for p in pares]
    cols = [p[1] for p in pares] + [p[0] for p in pares]
    adj = csr_matrix((np.ones(len(filas)), (filas, cols)), shape=(n, n))
    n_comp, labels = connected_components(adj, directed=False)
    clusters = []
    for c in range(n_comp):
        miembros = np.where(labels == c)[0]
        if len(miembros) >= min_miembros:
            clusters.append(miembros)
    return clusters


def _dinamica_estructura(masa_bar, dens_bar, amp_rugosidad,
                          n_pasos_estructura=60, dt=0.05, n_subpasos=10, seed_dens_null=None,
                          cdm_on=True, cooling_on=True, expansion_on=True, gravedad_on=True,
                          semilla="uniforme", D_causal=3, k_causal=4, iters_layout=100, seed_layout=12345):
    """UN bucle temporal, TODOS los módulos juntos (Regla 1). Recibe masa_bar/dens_bar YA EXTRAÍDOS (el
    motor basal se corre UNA sola vez, fuera de esta función -- es determinista, no depende de la
    semilla NULL, así que repetirlo por cada NULL sería desperdiciar minutos de corrida basal sin ganar
    nada; sólo la dinámica de estructura cambia con el barajado).

    semilla="uniforme" (legado, Q3/primera corrida holística): posiciones de un escenario 3D uniforme,
    independiente de la densidad. seed_dens_null!=None baraja los VALORES de densidad #23 (el NULL de
    "cantidad marginal", ya corrido y reportado negativo).

    semilla="causal" (el puente, v4 DISENO_CS073_puente_PARA_CC.md): posiciones sembradas por
    layout-de-resortes sobre la malla causal REAL (p_semilla_causal.py) -- pares causalmente cercanos
    (dos fases de expansión, mismo mecanismo que dimension_acoplada) arrancan próximos. La densidad #23
    NUNCA se baraja en este modo (ya se probó negativo aparte); seed_dens_null!=None baraja las ARISTAS
    del grafo (double-edge-swap, preserva grados) -- el NULL de "topología relacional", el que este
    experimento viene a probar.

    n_pasos_estructura = pasos "cosmológicos" -- cada uno evalúa el MISMO reloj T(step) que el resto
    del motor, y aplica UN estiramiento de expansión. n_subpasos = subdivisiones NUMÉRICAS internas de
    gravedad (dt_efectivo = dt/n_subpasos) -- necesarias porque dt=0.05 resultó numéricamente inestable
    en encuentros cercanos (chequeo de cordura: sin expansión, con dt grueso el sistema colapsaba y
    luego salía disparado por error de integración, no por física). Separar ambos evita re-tunear el
    reloj de expansión (ya establecido) sólo para resolver un problema numérico."""
    n_bar = len(masa_bar)
    if n_bar < 8:
        return dict(ok=False, nota=f"sólo {n_bar} átomos reales (<8): sin masa suficiente para el escenario")

    n_cdm = n_bar if cdm_on else 0
    n_total = n_bar + n_cdm
    lado = float(n_total) ** (1.0 / 3.0)

    if semilla == "uniforme":
        if seed_dens_null is not None:
            dens_bar = dens_bar[np.random.default_rng(seed_dens_null).permutation(n_bar)]
        pos_bar, _ = posiciones_escenario(n_bar, lado=lado, seed=12345)
        cdm = MateriaOscuraHalo(n_cdm, amp_rugosidad, lado_escenario=lado, activa=cdm_on,
                                 seed_pos=54321, seed_dens=7000)
        dens_cdm = cdm.barajar_densidad(seed_dens_null + 1) if (seed_dens_null is not None and cdm_on) else cdm.densidad
    elif semilla == "causal":
        adj_bar, _m_bar = malla_causal_atomos(dens_bar, D=D_causal, k=k_causal, seed_ejes=2000)
        if seed_dens_null is not None:
            adj_bar = barajar_aristas(adj_bar, n_bar, seed=seed_dens_null)
        pos_bar = layout_resortes(adj_bar, n_bar, lado=lado, iters=iters_layout, seed=seed_layout)
        cdm = MateriaOscuraHalo(n_cdm, amp_rugosidad, lado_escenario=lado, activa=cdm_on,
                                 seed_pos=54321, seed_dens=7000)
        dens_cdm = cdm.densidad   # densidad SIEMPRE real en este modo -- el NULL es de topología, no de valor
        if n_cdm:
            adj_cdm, _m_cdm = malla_causal_atomos(dens_cdm, D=D_causal, k=k_causal, seed_ejes=6000)
            if seed_dens_null is not None:
                adj_cdm = barajar_aristas(adj_cdm, n_cdm, seed=seed_dens_null + 1)
            cdm.pos = layout_resortes(adj_cdm, n_cdm, lado=lado, iters=iters_layout, seed=seed_layout + 42176)
    else:
        raise ValueError(f"semilla desconocida: {semilla!r} (usar 'uniforme' o 'causal')")

    SOFTENING = 0.3   # una sola convención, compartida por gravedad Y por el estimador de densidad de H2
    grav = GravedadGeneral(activa=gravedad_on, softening=SOFTENING)
    expansion = Expansion(T0=T0, activa=expansion_on)
    h2 = EnfriamientoH2(n_bar, T_inicial=T0, activa_cooling=cooling_on, seed=9000, softening=SOFTENING)

    pos = np.vstack([pos_bar, cdm.pos]) if n_cdm else pos_bar.copy()
    masa_eff = np.concatenate([masa_bar * dens_bar, cdm.masa * dens_cdm]) if n_cdm else masa_bar * dens_bar
    vel = np.zeros_like(pos)
    dt_sub = dt / n_subpasos

    for step in range(n_pasos_estructura):
        T_actual = _T_reloj(step)
        for _sub in range(n_subpasos):
            acc = grav.aceleraciones(pos, masa_eff)
            vel = vel + acc * dt_sub
            if n_bar:
                vel[:n_bar] = vel[:n_bar] + h2.kick_termico(escala=0.02) * np.sqrt(dt_sub)
            pos = pos + vel * dt_sub
            if not np.all(np.isfinite(pos)):
                return dict(ok=False, nota=f"NaN/inf en las posiciones al paso {step} -- falla del TODO (Regla 2: aislar con los interruptores)")
        if n_bar:
            h2.actualizar(pos[:n_bar])
        factor = expansion.paso_de_estiramiento(T_actual)
        if factor != 1.0:
            pos = pos * factor

    a_final = expansion._a_prev if expansion_on else 1.0
    espaciamiento_final = 1.0 * a_final
    linking_length = 0.2 * espaciamiento_final

    pos_bar_final = pos[:n_bar]
    clusters = _fof(pos_bar_final, linking_length, min_miembros=5)

    n_estructuras_jeans = 0
    detalle = []
    for miembros in clusters:
        masa_cluster = float(masa_eff[:n_bar][miembros].sum())
        T_local = float(h2.T[miembros].mean())
        rho_local = float(h2._densidad_local_dinamica(pos_bar_final)[miembros].mean())
        M_J = T_local ** 1.5 / np.sqrt(max(rho_local, 1e-9))
        supera = masa_cluster > M_J
        if supera:
            n_estructuras_jeans += 1
        detalle.append(dict(n_miembros=len(miembros), masa=masa_cluster, T_local=T_local,
                             rho_local=rho_local, M_J=M_J, supera_jeans=bool(supera)))

    return dict(ok=True, n_bariones=n_bar, n_cdm=n_cdm, n_pasos=n_pasos_estructura,
                n_clusters_ligados=len(clusters), n_estructuras_jeans=n_estructuras_jeans,
                a_final=a_final, linking_length=linking_length, detalle=detalle)


def correr_holistico(nq=300, naq=210, ne=100, npos=70, pasos_basal=150, amp_rugosidad=1.5, **kw):
    """Corrida única (REAL) -- conveniencia para smoke test. Extrae bariones UNA vez y corre la
    dinámica. Para el observable de cierre (REAL vs NULL con z-score) usar correr_real_vs_null()."""
    masa_bar, dens_bar, obs_basal = _extraer_bariones(nq, naq, ne, npos, pasos_basal, amp_rugosidad)
    r = _dinamica_estructura(masa_bar, dens_bar, amp_rugosidad, **kw)
    if r.get("ok"):
        r["obs_basal"] = dict(hidrogeno=obs_basal.get("hidrogeno"), helio=obs_basal.get("helio"))
    return r


def correr_control_positivo(nq=1500, naq=1050, ne=500, npos=350, pasos_basal=150, amp_rugosidad=1.5,
                             usar_densidad_como_peso=False, **kw):
    """CONTROL POSITIVO / test de falsación (no es REAL-vs-NULL): dado un número REAL de átomos de
    hidrógeno (del motor basal), dispersos por la expansión REAL (misma pieza p_expansion, mismo reloj
    T(t) que el resto del motor), bajo gravedad general + presión/enfriamiento H2 + andamio CDM --
    ¿EMERGE UNA ESTRELLA (una estructura que supera Jeans)? No busca discriminar coherencia del campo
    #23 (eso ya se midió aparte, negativo); pregunta si el MECANISMO puede formarla en condiciones
    favorables. usar_densidad_como_peso=False (default): masa = masa_trio real, SIN pesar por #23 --
    para no mezclar "el mecanismo no alcanza" con "el campo #23 no tenía coherencia que aprovechar"."""
    masa_bar, dens_bar, obs_basal = _extraer_bariones(nq, naq, ne, npos, pasos_basal, amp_rugosidad)
    peso = dens_bar if usar_densidad_como_peso else np.ones_like(dens_bar)
    r = _dinamica_estructura(masa_bar, peso, amp_rugosidad, seed_dens_null=None, **kw)
    if r.get("ok"):
        r["obs_basal"] = dict(hidrogeno=obs_basal.get("hidrogeno"), helio=obs_basal.get("helio"))
        r["emerge_estrella"] = bool(r["n_estructuras_jeans"] > 0)
    return r


def _z(real_val, dist):
    arr = np.asarray(dist, float)
    mu, sd = float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    if sd == 0.0:
        return None
    return round(float((real_val - mu) / sd), 3)


def correr_real_vs_null(nq=300, naq=210, ne=100, npos=70, pasos_basal=150, amp_rugosidad=1.5,
                         n_null=8, seed_null_base=5000, **kw):
    """EL OBSERVABLE DE CIERRE: motor basal UNA vez (determinista, no depende de la semilla NULL) ->
    dinámica holística REAL una vez + n_null corridas con el campo #23 barajado -> z-score de
    n_estructuras_jeans (y n_clusters_ligados) REAL contra la distribución de NULLs."""
    masa_bar, dens_bar, obs_basal = _extraer_bariones(nq, naq, ne, npos, pasos_basal, amp_rugosidad)
    real = _dinamica_estructura(masa_bar, dens_bar, amp_rugosidad, seed_dens_null=None, **kw)
    if not real.get("ok"):
        return dict(ok=False, nota="la corrida REAL falló -- ver 'real'", real=real)

    nulls = []
    for i in range(n_null):
        r_null = _dinamica_estructura(masa_bar, dens_bar, amp_rugosidad,
                                       seed_dens_null=seed_null_base + i * 2, **kw)
        nulls.append(r_null)

    jeans_null = [r["n_estructuras_jeans"] for r in nulls if r.get("ok")]
    ligados_null = [r["n_clusters_ligados"] for r in nulls if r.get("ok")]

    z_jeans = _z(real["n_estructuras_jeans"], jeans_null) if jeans_null else None
    z_ligados = _z(real["n_clusters_ligados"], ligados_null) if ligados_null else None

    return dict(ok=True, n_null_ok=len(jeans_null),
                real_n_estructuras_jeans=real["n_estructuras_jeans"],
                real_n_clusters_ligados=real["n_clusters_ligados"],
                null_jeans_media=round(float(np.mean(jeans_null)), 3) if jeans_null else None,
                null_jeans_std=round(float(np.std(jeans_null, ddof=1)), 3) if len(jeans_null) > 1 else 0.0,
                null_ligados_media=round(float(np.mean(ligados_null)), 3) if ligados_null else None,
                null_ligados_std=round(float(np.std(ligados_null, ddof=1)), 3) if len(ligados_null) > 1 else 0.0,
                z_n_estructuras_jeans=z_jeans, z_n_clusters_ligados=z_ligados,
                obs_basal=dict(hidrogeno=obs_basal.get("hidrogeno"), helio=obs_basal.get("helio")),
                real_detalle=real.get("detalle"), a_final=real.get("a_final"))


def correr_puente_real_vs_null(nq=300, naq=210, ne=100, npos=70, pasos_basal=150, amp_rugosidad=1.5,
                                n_null=8, seed_null_base=5000, **kw):
    """EL PUENTE (v4, DISENO_CS073_puente_PARA_CC.md): idéntico a correr_real_vs_null(), pero con
    semilla='causal' -- posiciones sembradas por la malla causal REAL (layout de resortes), NULL =
    esa MISMA malla con las aristas barajadas (topología, no densidad). Envoltorio delgado; la lógica
    vive en _dinamica_estructura(semilla='causal') y en p_semilla_causal.py."""
    return correr_real_vs_null(nq=nq, naq=naq, ne=ne, npos=npos, pasos_basal=pasos_basal,
                                amp_rugosidad=amp_rugosidad, n_null=n_null, seed_null_base=seed_null_base,
                                semilla="causal", **kw)
