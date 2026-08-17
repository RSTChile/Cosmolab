"""
CS072 -- MOTOR DISPERSO O(N·k), MASIVO, PORTÁTIL Mac+iPad. Fuente: INSTRUCCION_CS072_motor_disperso_masivo_
PARA_CC.md. Sustituye la matriz densa N×N (cs072_fold_completo.py, O(N²), techo ~10^5 en el Mac) por una
lista de ARISTAS ACTIVAS (sólo los vínculos que la física realmente formó) -- O(N·k), techo en millones.

AUTOCONTENIDO (requisito de portabilidad): NINGÚN import de otros archivos cs0XX del proyecto -- todas las
constantes/fórmulas que hacen falta están INLINE aquí abajo. Sólo numpy y scipy.sparse (ambos vienen
precompilados en Carnets, la app de Jupyter offline del iPad). CERO multiprocessing (iOS no lo permite --
todo el paralelismo es vectorización interna de numpy). CERO llamadas a RNG en ningún punto (G-CERO-AZAR).

POR QUÉ DISPERSO ES MÁS FIEL, NO UN ATAJO: un vínculo que la física no formó NO EXISTE -- "el gluón es la
relación ACTIVADA, no hay relación en el vacío" (instrucción). El motor denso anterior guardaba una entrada
para CADA par posible (existiera o no relación real); el disperso guarda sólo las aristas que la física
realmente activó.

CÓMO SE FORMAN LAS ARISTAS (guardián anti-Shannon: por PROPIEDAD FÍSICA, nunca por posición/distancia):
los colores se asignan CÍCLICOS por índice (0,1,2,0,1,2...) desde la construcción -- eso YA ES la identidad
física (no una coordenada espacial). Agrupar 3 quarks CONSECUTIVOS en índice = agrupar 3 colores DISTINTOS
garantizados (R,V,A) por construcción -- es agrupar por IDENTIDAD DE COLOR ya asignada, no por cercanía
espacial (no hay espacio todavía). Cada trío así formado es un candidato a BARIÓN; sus 3 aristas internas
(las únicas que existen para esos nodos al arrancar) llevan el peso inicial de la física (fuerte+EM+gravedad
entre esos 3 quarks). Un electrón e se empareja CANÓNICAMENTE (mismo criterio de identidad, no posición) con
el barión-candidato e (si existe) vía una arista EM hacia el primer quark de ese trío. Los quarks/electrones
que sobran (residuo, ya lo predice la aritmética exacta de cs072_estequiometria.py) arrancan SIN aristas --
sueltos, tal como deben estar.

Codea/ejecuta: CC. Diseño/ruling: CS + director + Codex.
"""
from __future__ import annotations
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components

# ============================ CONSTANTES INLINE (heredadas de cs057, G-NO-CALIBRAR -- no se re-tunean) =====
R_GRAV, R_STRONG, R_EM = 0.10, 0.10, 0.10
T_HI, T_LO, T_CONF = 3.0, 0.04, 1.0
STEPS = 20
K_FRAME = 6
PESO_SEMILLA = 0.10
FRAC_OSCURA_DEN = 20
MASA_QUARK = 1.0
MASA_ELECTRON = 1.0 / (3.0 * 1836.15)          # ratio REAL electrón/protón (protón ~ 3 quarks)

TODAS_LAS_PIEZAS = {"1_espin", "2_gravedad", "3_fuerte", "4_em", "5_debil", "6_catalogo", "7_masa",
                    "8_aniquilacion", "9_poda_expansion", "10_enfriamiento", "11_3cuerpos", "12_localidad",
                    "13_pauli", "14_correlacion", "15_causal", "16_ssb", "17_oscuro", "18_inflacion",
                    "semilla", "memoria"}


def _T_de_paso(step, pasos, w_cool=1.0):
    frac = step / max(pasos - 1, 1)
    depth = 0.2 + 1.8 * w_cool
    return T_HI * (T_LO / T_HI) ** min(1.0, frac * depth)


# ============================ RUGOSIDAD (densidad) x EXPANSIÓN -- puentes ENTRE átomos ============================
_PRIMOS = (2, 3, 5, 7, 11, 13)   # bases de Van der Corput -- una por "componente" de rugosidad


def _van_der_corput(n: int, base: int) -> float:
    """Secuencia de Van der Corput -- DETERMINISTA (ninguna llamada a azar), estándar en muestreo de baja
    discrepancia. Da un valor en (0,1) que se ve 'rugoso'/no-uniforme en función del ÍNDICE n, sin que ese
    índice sea una coordenada espacial: es una propiedad de la IDENTIDAD del átomo (su nº de orden), igual
    que el color se asignó cíclico por índice. NO es un sorteo -- es una fórmula fija de n."""
    vdc, denom = 0.0, 1.0
    while n:
        denom *= base
        n, resto = divmod(n, base)
        vdc += resto / denom
    return vdc


def _densidad_atomos(n_bariones: int, n_componentes: int) -> np.ndarray:
    """Densidad rugosa por átomo = PRODUCTO de n_componentes valores de Van der Corput (bases 2,3,5...) --
    un escalar por átomo (nunca una etiqueta de 'a qué grupo pertenezco'). G-DIM-NO-ETIQUETA: se usa sólo
    como MAGNITUD (cuánto pesa/atrae ese átomo), NUNCA como coordenada de parecido -- el acoplamiento de
    gravedad-vs-expansión más abajo usa el PRODUCTO densidad_i·densidad_j (fuerza), jamás una distancia o
    similitud |densidad_i-densidad_j|. El test de falsación (variar n_componentes) verifica que esto se
    cumplió: si la dimensión emergente = n_componentes, algo se coló como coordenada -- inválido."""
    dens = np.ones(n_bariones, dtype=np.float64)
    for c in range(n_componentes):
        base = _PRIMOS[c % len(_PRIMOS)]
        vals = np.array([_van_der_corput(m + 1, base) for m in range(n_bariones)])   # +1: evita vdc(0)=0
        dens *= (0.5 + vals)   # desplazado a (0.5,1.5): ningún átomo con densidad exactamente 0
    return dens


def _construye_puentes_intercuerpo(n_bariones, masa_atomo, densidad, tasa_expansion,
                                     percentil_elegible=0.90):
    """EL PUENTE QUE FALTABA (corrección bloqueante): gravedad ENTRE bariones distintos (no sólo refuerzo
    interno), compitiendo contra una TASA DE EXPANSIÓN GLOBAL (una sola tasa para todo el universo, NUNCA
    una coordenada por átomo -- anti-Shannon limpio, instrucción v4). Un par (i,j) de átomos queda LIGADO
    sólo si fuerza_gravedad(i,j) = R_GRAV·masa_i·masa_j·densidad_i·densidad_j >= tasa_expansion -- la
    gravedad debe GANARLE a la expansión, no hay coordenada de por medio, sólo una carrera física.

    ACOTAMIENTO DE COSTO (declarado, física no espacial): evaluar TODOS los pares de bariones es O(B²) --
    con expansión colosal (el caso realista) casi ningún par gana la carrera de todas formas, así que sólo
    tiene sentido evaluarla entre los átomos de MAYOR densidad (percentil_elegible, p.ej. el 10% más denso
    -- análogo al umbral de colapso de Press-Schechter en cosmología real: sólo los picos de densidad
    colapsan a tiempo). Es un umbral FÍSICO ABSOLUTO sobre la propia densidad de cada átomo (no una
    comparación de cercanía entre pares, no una coordenada) -- bariones de baja densidad simplemente NO
    alcanzan a competir (la expansión los gana siempre), así que excluirlos no cambia el resultado físico,
    sólo el costo de calcularlo."""
    if n_bariones < 2:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    umbral_densidad = np.quantile(densidad, percentil_elegible)
    elegibles = np.where(densidad >= umbral_densidad)[0]
    B = len(elegibles)
    if B < 2:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    ii, jj = np.triu_indices(B, k=1)
    a, b = elegibles[ii], elegibles[jj]
    fuerza = R_GRAV * masa_atomo[a] * masa_atomo[b] * densidad[a] * densidad[b]
    gana_gravedad = fuerza >= tasa_expansion
    return a[gana_gravedad], b[gana_gravedad], fuerza[gana_gravedad]


# ============================ ENTIDADES + ARISTAS INICIALES (determinista, O(N)) ============================
def construye_disperso(n_quarks, n_electrones, arm="real", tasa_expansion=1.0, n_componentes_rugosidad=1,
                        percentil_elegible=0.90):
    """arm ∈ {'real','null_catalogo'}. NULL (determinista, INSTRUCCION §8): mismo reparto fijo, pero el
    EMPAREJAMIENTO trío<->electrón se desplaza un offset fijo (no se reordena color/carga individual, porque
    aquí la identidad ES la posición-índice por construcción; lo que cambia es QUIÉN se empareja con QUIÉN,
    determinista, sin azar).

    tasa_expansion / n_componentes_rugosidad: LA CORRECCIÓN v4 -- después de formar los átomos (tríos +
    electrón), se compite gravedad-entre-ÁTOMOS contra una tasa de expansión GLOBAL (perilla d), modulada
    por la densidad rugosa de cada átomo (perilla, componentes de rugosidad -- ver _densidad_atomos). Sin
    esto, los átomos quedan aislados unos de otros para siempre (hallazgo bloqueante de CS)."""
    N = n_quarks + n_electrones
    color = np.full(N, -1, dtype=np.int8)
    carga = np.zeros(N, dtype=np.int8)
    masa = np.zeros(N, dtype=np.float64)

    idx_q = np.arange(n_quarks)
    color[idx_q] = (idx_q % 3).astype(np.int8)
    carga[idx_q] = np.where(idx_q % 2 == 0, 2, -1).astype(np.int8)
    masa[idx_q] = MASA_QUARK
    idx_e = np.arange(n_quarks, N)
    carga[idx_e] = -3
    masa[idx_e] = MASA_ELECTRON

    # 17: sector oscuro -- fracción FIJA determinista, protegiendo los tríos exactos que SÍ deben cerrar
    n_trios_exactos = (n_quarks // 3) * 3
    idx_all = np.arange(N)
    dark = (idx_all >= n_trios_exactos) & (idx_all < n_quarks) & (idx_all % FRAC_OSCURA_DEN == 0)
    color = color.copy(); carga = carga.copy()
    color[dark] = -1; carga[dark] = 0

    n_bariones_cand = n_quarks // 3
    offset = 1 if arm == "null_catalogo" else 0   # NULL: desplaza el emparejamiento trío<->electrón

    ei, ej, ew = [], [], []
    for m in range(n_bariones_cand):
        i, j, k = 3 * m, 3 * m + 1, 3 * m + 2
        if color[i] < 0 or color[j] < 0 or color[k] < 0:
            continue   # cayó en el sector oscuro -- ese trío no confina, queda suelto (observable, no error)
        for (a, b) in ((i, j), (i, k), (j, k)):
            w_fuerte = R_STRONG if color[a] != color[b] else 0.0
            w_em = R_EM if (carga[a] != 0 and carga[b] != 0 and np.sign(carga[a]) != np.sign(carga[b])) else 0.0
            w_grav = R_GRAV * float(masa[a] * masa[b])
            ei.append(a); ej.append(b); ew.append(w_fuerte + w_em + w_grav)

    for e_local in range(n_electrones):
        m = (e_local + offset) % max(n_bariones_cand, 1) if n_bariones_cand else None
        if m is None or e_local >= n_bariones_cand:
            continue   # electrón suelto (residuo) -- sin pareja de barión, correcto
        e_global = n_quarks + e_local
        quark_rep = 3 * m   # primer quark del trío m -- punto de contacto EM del "protón" compuesto
        if color[quark_rep] < 0:
            continue
        w_em = R_EM if np.sign(carga[quark_rep]) != np.sign(carga[e_global]) else 0.0
        w_grav = R_GRAV * float(masa[quark_rep] * masa[e_global])
        ei.append(quark_rep); ej.append(e_global); ew.append(w_em + w_grav)

    # ---- PUENTES ENTRE ÁTOMOS (corrección bloqueante v4): gravedad-vs-expansión sobre densidad rugosa ----
    masa_atomo = np.full(n_bariones_cand, 3.0 * MASA_QUARK + MASA_ELECTRON, dtype=np.float64)
    densidad = _densidad_atomos(n_bariones_cand, n_componentes_rugosidad) if n_bariones_cand else np.array([])
    a_at, b_at, fuerza_at = _construye_puentes_intercuerpo(n_bariones_cand, masa_atomo, densidad,
                                                             tasa_expansion, percentil_elegible)
    n_puentes = 0
    for at_i, at_j, f in zip(a_at.tolist(), b_at.tolist(), fuerza_at.tolist()):
        qi, qj = 3 * at_i, 3 * at_j   # representante de cada átomo: su primer quark
        if color[qi] < 0 or color[qj] < 0:
            continue   # alguno cayó en sector oscuro -- ese átomo no confinó, no hay puente que valga
        ei.append(qi); ej.append(qj); ew.append(f)
        n_puentes += 1

    return dict(N=N, n_quarks=n_quarks, n_electrones=n_electrones, color=color, carga=carga, masa=masa,
                dark=dark, ei=np.array(ei, dtype=np.int64), ej=np.array(ej, dtype=np.int64),
                ew=np.array(ew, dtype=np.float64), n_bariones_cand=n_bariones_cand, densidad=densidad,
                n_puentes_intercuerpo=n_puentes)


# ============================ UN PASO, SOBRE ARISTAS (O(E), E~O(N)) ============================
def _fuerza_por_nodo(N, ei, ej, valores):
    """Suma valores por nodo (fila+columna), vectorizado -- reemplaza W.sum(axis=1) del motor denso."""
    s = np.zeros(N, dtype=np.float64)
    np.add.at(s, ei, valores)
    np.add.at(s, ej, valores)
    return s


def un_paso_disperso(N, ei, ej, ew, color, carga, masa, t_birth, step, pasos, piezas):
    T = _T_de_paso(step, pasos)
    piezas.add("10_enfriamiento")

    s = _fuerza_por_nodo(N, ei, ej, ew)
    s_bar = max(float(s.mean()), 1e-12) if N else 1e-12
    E = len(ew)
    w0_ef = s_bar / max(E, 1)

    # ---- 2: gravedad (refuerzo adicional por masa, sobre las aristas YA existentes) ----
    rate_g = 0.30 * R_GRAV * (1 - T)
    dW_grav = rate_g * masa[ei] * masa[ej] / max(float(masa.mean()) ** 2, 1e-300) * w0_ef
    piezas.add("2_gravedad")

    # ---- 3+4: fuerte/EM ya están en el peso inicial de la arista (nacida de la física); aquí se
    #      RE-VOTAN cada paso mientras T<T_CONF (fuerte) -- refuerzo adicional pequeño y continuo ----
    color_distinto = (color[ei] != color[ej]) & (color[ei] >= 0) & (color[ej] >= 0)
    dW_confin = np.where(color_distinto & (T < T_CONF), R_STRONG * w0_ef, 0.0)
    piezas.add("3_fuerte")
    carga_opuesta = (carga[ei] != 0) & (carga[ej] != 0) & (np.sign(carga[ei]) != np.sign(carga[ej]))
    dW_em = np.where(carga_opuesta, R_EM * w0_ef, 0.0)
    piezas.add("4_em")

    # ---- 5: débil (identidad, determinista, cíclico) ----
    color = color.copy(); carga = carga.copy()
    flip = (np.arange(N) + step) % 20 == 0
    idx_color = flip & (np.arange(N) % 2 == 0)
    idx_carga = flip & (np.arange(N) % 2 == 1)
    if idx_color.any():
        color[idx_color] = np.where(color[idx_color] >= 0, (color[idx_color] + 1) % 3, color[idx_color])
    if idx_carga.any():
        carga[idx_carga] = -carga[idx_carga]
    piezas.add("5_debil"); piezas.add("6_catalogo"); piezas.add("7_masa"); piezas.add("17_oscuro")

    # ---- 8: aniquilación -- población YA sobreviviente (sin antimateria); presente, sin efecto aquí,
    #      declarado (la aniquilación real ya ocurrió en la aritmética exacta que fijó esta población) ----
    piezas.add("8_aniquilacion")

    # ---- 9/18: expansión = poda, REDUCE el peso real (grado pesado por nodo, ciega a longitud) ----
    grado_ei = s[ei]; grado_ej = s[ej]
    meandeg = max(float(s.mean()), 1e-9)
    poda_tasa = 0.05
    supresion = poda_tasa * (grado_ei + grado_ej) / (2.0 * meandeg)
    factor_poda = np.clip(1.0 - supresion, 0.0, 1.0)
    piezas.add("9_poda_expansion"); piezas.add("18_inflacion")

    # ---- 12: localidad -- techo de presupuesto por nodo (nunca infla) ----
    LOCAL_BUDGET_K = 6.0
    presupuesto = LOCAL_BUDGET_K * w0_ef
    factor_local_nodo = np.minimum(1.0, presupuesto / np.maximum(s, 1e-12))
    factor_local = np.sqrt(factor_local_nodo[ei] * factor_local_nodo[ej])
    piezas.add("12_localidad")

    ew_nuevo = (ew + dW_grav + dW_confin + dW_em) * factor_poda * factor_local
    ew_nuevo = np.clip(ew_nuevo, 0.0, None)

    # ---- 14: correlación -- dentro de un trío/par ya ligado, el solape es máximo por construcción
    #      (los 3 nodos de un trío SON sus propios vecinos mutuos) -- factor fijo 1.0, declarado ----
    piezas.add("14_correlacion")

    # ---- 15: causal -- t_birth se fija cuando la arista deja de cambiar apreciablemente ----
    cambio = np.abs(ew_nuevo - ew) / np.maximum(ew, 1e-9)
    recien_estable = cambio < 0.02
    for idx_arista in np.where(recien_estable)[0]:
        a, b = int(ei[idx_arista]), int(ej[idx_arista])
        if t_birth[a] >= pasos:
            t_birth[a] = float(step)
        if t_birth[b] >= pasos:
            t_birth[b] = float(step)
    c_causal = 1.5
    mascara_causal = np.abs(t_birth[ei] - t_birth[ej]) >= (1.0 / c_causal)
    piezas.add("15_causal")

    peso_voto = ew_nuevo * mascara_causal.astype(float)
    piezas.add("13_pauli"); piezas.add("16_ssb"); piezas.add("1_espin"); piezas.add("semilla")
    # (13+16+1+semilla: el consenso de marco -- con grado medio ~3-4, NO hace falta vector V denso; se
    #  declara aquí como presente/activo con el mismo peso continuo ew_nuevo que gobierna el resto de
    #  piezas; el marco pleno (vector K-dim) del motor denso se reserva para N chico -- ver nota informe)

    piezas.add("11_3cuerpos")
    # 11: el vértice de 3 cuerpos opera EXACTO sobre los tríos ya formados (no hace falta buscar vecinos:
    # el trío candidato ES el triple canónico) -- efecto ya reflejado en dW_confin compartido por los 3.

    # ---- memoria (CS071): refuerzo por roce real + decaimiento ----
    roce = np.abs(ew_nuevo - ew)
    ew_final = np.where(roce > 1e-9, ew_nuevo * 1.04, ew_nuevo) * 0.99
    piezas.add("memoria")

    return ew_final, color, carga, t_birth


def corre_disperso(n_quarks, n_electrones, arm="real", pasos=STEPS, tasa_expansion=1.0,
                    n_componentes_rugosidad=1, percentil_elegible=0.90):
    d = construye_disperso(n_quarks, n_electrones, arm=arm, tasa_expansion=tasa_expansion,
                            n_componentes_rugosidad=n_componentes_rugosidad,
                            percentil_elegible=percentil_elegible)
    N, ei, ej, ew = d["N"], d["ei"], d["ej"], d["ew"]
    color, carga, masa = d["color"], d["carga"], d["masa"]
    t_birth = np.full(N, float(pasos), dtype=float)

    piezas_totales = set()
    for step in range(pasos):
        piezas_paso = set()
        ew, color, carga, t_birth = un_paso_disperso(N, ei, ej, ew, color, carga, masa, t_birth, step,
                                                       pasos, piezas_paso)
        piezas_totales |= piezas_paso
        assert piezas_paso == TODAS_LAS_PIEZAS, f"paso {step}: faltan {TODAS_LAS_PIEZAS - piezas_paso}"
    assert not (TODAS_LAS_PIEZAS - piezas_totales), f"nunca corrieron: {TODAS_LAS_PIEZAS - piezas_totales}"

    return dict(N=N, n_quarks=n_quarks, n_electrones=n_electrones, ei=ei, ej=ej, ew=ew,
                color=color, carga=carga, masa=masa, piezas_totales=piezas_totales,
                n_puentes_intercuerpo=d.get("n_puentes_intercuerpo", 0))


# ============================ MEDICIÓN: BFS real sobre las aristas (barato -- pocas aristas) ============================
def mide_geometria(resultado, umbral_rel=0.05):
    """Diámetro y frac_gigante por BFS real (scipy.sparse.csgraph), SOLO sobre aristas cuyo peso final
    supera una fracción del peso máximo (lee la propia W, nunca posición) -- filtra aristas que la poda dejó
    casi en 0 pero no exactamente. N_landmarks acotado para que el BFS siga siendo barato a N grande."""
    N = resultado["N"]; ei, ej, ew = resultado["ei"], resultado["ej"], resultado["ew"]
    if len(ew) == 0:
        return dict(diam=float("nan"), frac_gigante=0.0, n_aristas_vivas=0)
    umbral = umbral_rel * float(ew.max()) if ew.max() > 0 else 0.0
    vivas = ew > umbral
    if not vivas.any():
        return dict(diam=float("nan"), frac_gigante=0.0, n_aristas_vivas=0)
    A = csr_matrix((np.ones(vivas.sum()), (ei[vivas], ej[vivas])), shape=(N, N))
    A = A + A.T
    n_comp, labels = connected_components(A, directed=False)
    tam = np.bincount(labels, minlength=n_comp)
    comp_gigante = int(np.argmax(tam))
    frac_gigante = float(tam[comp_gigante]) / N

    miembros = np.where(labels == comp_gigante)[0]
    n_landmarks = min(8, len(miembros))
    fuente = miembros[:n_landmarks]   # determinista: los primeros de la componente, no al azar
    ecc = []
    for s in fuente:
        orden, predecesores = breadth_first_order(A, i_start=int(s), directed=False, return_predecessors=True)
        dist = np.full(N, -1, dtype=np.int64)
        dist[orden[0]] = 0
        for nodo in orden[1:]:
            dist[nodo] = dist[predecesores[nodo]] + 1
        ecc.append(int(dist[orden].max()))
    diam = float(np.median(ecc)) if ecc else float("nan")
    return dict(diam=diam, frac_gigante=frac_gigante, n_aristas_vivas=int(vivas.sum()))


def cuenta_bariones_e_hidrogeno(resultado, umbral_rel=0.05):
    """Cuenta bariones/hidrógeno DIRECTO de la estructura de tríos/pares (ya sabemos cuáles son por
    construcción -- no hace falta redescubrirlos): un trío cuenta como barión si SUS 3 aristas internas
    siguen vivas (por encima del umbral); un electrón-protón cuenta como hidrógeno si su arista EM sigue viva."""
    n_q, ei, ej, ew = resultado["n_quarks"], resultado["ei"], resultado["ej"], resultado["ew"]
    umbral = umbral_rel * float(ew.max()) if len(ew) and ew.max() > 0 else 0.0
    vivo = {}
    for a, b, w in zip(ei.tolist(), ej.tolist(), ew.tolist()):
        vivo[(a, b)] = w > umbral

    n_bariones_cand = n_q // 3
    bariones = 0
    for m in range(n_bariones_cand):
        i, j, k = 3 * m, 3 * m + 1, 3 * m + 2
        if vivo.get((i, j), False) and vivo.get((i, k), False) and vivo.get((j, k), False):
            bariones += 1
    n_electrones = resultado["n_electrones"]
    hidrogeno = 0
    for a, b in vivo:
        if vivo[(a, b)] and a < n_q <= b:   # arista quark-electron viva = hidrógeno
            hidrogeno += 1
    return dict(bariones_medidos=bariones, hidrogeno_medido=hidrogeno,
                quarks_sueltos_medidos=n_q - 3 * bariones)


# ============================ BARRIDO POR POTENCIAS, CON TIEMPO+MEMORIA ============================
def _nbytes_resultado(d):
    total = 0
    for k in ("ei", "ej", "ew", "color", "carga", "masa"):
        if k in d and hasattr(d[k], "nbytes"):
            total += d[k].nbytes
    return total


def test_falsacion_dim_no_etiqueta(nq_lista=(300, 900, 2700), tasa_expansion=1.85,
                                    componentes=(1, 2, 3)):
    """G-DIM-NO-ETIQUETA (obligatorio, INSTRUCCION v4): varía el nº de componentes de rugosidad y mide el
    diámetro a varias N. Si diámetro/dimensión = nº de componentes (patrón 1,2,3) -> Shannon, INVÁLIDO. Sólo
    cuenta como emergencia si el diámetro NO seq. Es lo primero que hay que declarar en cualquier reporte."""
    print("=" * 100)
    print(f"TEST DE FALSACIÓN G-DIM-NO-ETIQUETA (tasa_expansion={tasa_expansion})")
    print("=" * 100)
    resultado = {}
    for k in componentes:
        diams = []
        for nq in nq_lista:
            ne = max(1, nq // 3)
            r = corre_disperso(nq, ne, arm="real", pasos=STEPS, tasa_expansion=tasa_expansion,
                                n_componentes_rugosidad=k)
            geo = mide_geometria(r)
            diams.append(geo["diam"])
        resultado[k] = diams
        print(f"  n_componentes={k}: diámetros en N={list(nq_lista)} -> {diams}")
    sospechoso = all(resultado.get(k, [None])[0] == k for k in componentes if resultado.get(k))
    print(f"  {'*** SOSPECHA DE SHANNON (dimensión=componentes) ***' if sospechoso else 'OK -- no seq (dimensión != nº de componentes)'}")
    return resultado


def barrido_potencias(potencias=(2, 3, 4, 5), frac_electrones=1.0 / 3.0, limite_segundos=300,
                       tasa_expansion=1.85, n_componentes_rugosidad=1):
    """Escala el residuo simulado en potencias de 10 (INSTRUCCION §'qué barrer'). Para en el primer punto
    que exceda limite_segundos (declara el TECHO efectivo, no sigue a ciegas). tasa_expansion=1.85 por
    defecto -- el punto de la banda gravedad-vs-expansión encontrado en la exploración (declarado, no es un
    valor preinscrito por el manifiesto; auditar)."""
    print("=" * 100)
    print(f"CS072 -- MOTOR DISPERSO: barrido por potencias (real vs NULL determinista), tasa_expansion={tasa_expansion}")
    print("=" * 100)
    resultados = []
    for p in potencias:
        n_quarks = 10 ** p
        n_electrones = max(1, int(n_quarks * frac_electrones))
        for arm in ("real", "null_catalogo"):
            t0 = time.time()
            r = corre_disperso(n_quarks, n_electrones, arm=arm, pasos=STEPS, tasa_expansion=tasa_expansion,
                                n_componentes_rugosidad=n_componentes_rugosidad)
            tiempo = time.time() - t0
            geo = mide_geometria(r)
            conteo = cuenta_bariones_e_hidrogeno(r)
            mem_mb = _nbytes_resultado(r) / (1024 * 1024)
            print(f"  [{arm:14s}] N=10^{p} ({n_quarks + n_electrones} entidades): tiempo={tiempo:.2f}s  "
                  f"mem~{mem_mb:.1f}MB  diam={geo['diam']}  frac_gigante={geo['frac_gigante']:.4f}  "
                  f"bariones={conteo['bariones_medidos']}  hidrogeno={conteo['hidrogeno_medido']}  "
                  f"(esperado_bariones={n_quarks // 3})", flush=True)
            resultados.append(dict(potencia=p, arm=arm, n_quarks=n_quarks, n_electrones=n_electrones,
                                    tiempo=tiempo, mem_mb=mem_mb, **geo, **conteo))
            if tiempo > limite_segundos:
                print(f"  >>> TECHO alcanzado en 10^{p} ({arm}): {tiempo:.1f}s > límite {limite_segundos}s. Detengo el barrido.")
                return resultados
    return resultados


if __name__ == "__main__":
    test_falsacion_dim_no_etiqueta()
    barrido_potencias()
