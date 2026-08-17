"""
CS072-II -- PUERTA S (S0-S7 de S0-S9; S8-S9 diferidas, requieren el módulo de filtración/jueces continuos
que aún no está construido -- declarado, no escondido). BLOQUEANTE: si esto no pasa, no hay exploratoria
NÚCLEO-II ni fold. Fuente: PROPUESTA_CODEX_CS072_II_transicion_sin_sustrato_PARA_CS.md §9, adjudicada en
ADJUDICACION_CS072_II_transicion_sin_sustrato_CS.md ("Especialmente S1, S3, S6, S7").

Codea/ejecuta: CC. Diseño/ruling: CS + Codex.
"""
from __future__ import annotations
import sys
import numpy as np

sys.path.insert(0, "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis")
import cs072_ii_nucleo as II
import cs072_ii_filtracion as F

RNG = np.random.default_rng
TOL = 1e-9   # tolerancia numérica (float64, decenas de pasos de operaciones vectorizadas -- no exacto a 1e-15)


def _linea(ok, nombre, detalle):
    estado = "PASA" if ok else "FALLA"
    print(f"[{estado}] {nombre}: {detalle}", flush=True)
    return ok


def s0_epsilon_cero(N=50, pasos=30):
    """S0: eps=0 (delta=0, sin focos distintos) -> T y W deben permanecer UNIFORMES (no necesariamente
    en su valor inicial -- la expansión SÍ atenúa globalmente incluso sin diferencia -- pero TODOS los
    pares deben seguir siendo iguales entre sí, ninguna entidad emerge)."""
    T, W = II.estado_inicial(N, n_focos=0, delta=0.0)
    for _ in range(pasos):
        T, W, _ = II.paso_ii_det(T, W)
    T_uniforme = np.allclose(T, T[0], atol=TOL)
    iu = np.triu_indices(N, k=1)
    W_uniforme = np.allclose(W[iu], W[iu][0], atol=TOL)
    ok = T_uniforme and W_uniforme
    return _linea(ok, "S0 (eps=0)", f"T uniforme={T_uniforme} (T[0]={T[0]:.6f}), "
                  f"W uniforme={W_uniforme} (W[0,1]={W[0,1]:.6f}, spread={W[iu].std():.2e})")


def s1_permutacion(N=40, n_focos=2, delta=1e-4, pasos=25, seed=7):
    """S1: F(P.T, P.W.P^T) = P.F(T,W).P^T -- relabelar exactamente ANTES de correr debe dar el mismo
    resultado que correr y LUEGO relabelar. Se compara el ESTADO COMPLETO (T y W), no promedios."""
    rng = RNG(seed)
    T0, W0 = II.estado_inicial(N, n_focos, delta)
    P = rng.permutation(N)   # permutación de índices (uso de RNG es DEL TEST, no del motor -- meta-herramienta)

    T, W = T0.copy(), W0.copy()
    for _ in range(pasos):
        T, W, _ = II.paso_ii_det(T, W)
    T_directo_permutado = T[P]
    W_directo_permutado = W[np.ix_(P, P)]

    Tp, Wp = T0[P].copy(), W0[np.ix_(P, P)].copy()
    for _ in range(pasos):
        Tp, Wp, _ = II.paso_ii_det(Tp, Wp)

    dT = np.max(np.abs(T_directo_permutado - Tp))
    dW = np.max(np.abs(W_directo_permutado - Wp))
    ok = dT < TOL and dW < TOL
    return _linea(ok, "S1 (permutacion)", f"max|dT|={dT:.2e}  max|dW|={dW:.2e}  (tol={TOL:.0e})")


def s2_orden_operadores(N=30, n_focos=1, delta=1e-3, pasos=1, seed=3):
    """S2: recomponer el paso calculando los 4 sub-terminos (roce, gravedad, refuerzo, expansion) en
    ORDEN DE CODIGO distinto (pero desde el MISMO T,W de entrada -- ninguno lee la salida de otro) debe
    dar resultado identico. Verifica que no hay cascada/mutacion oculta."""
    T, W = II.estado_inicial(N, n_focos, delta)

    def paso_orden_A(T, W, **kw):
        return II.paso_ii_det(T, W, **kw)

    def paso_orden_B(T, W, tasa_flujo=II.TASA_FLUJO_DEFAULT, grav_rate=II.GRAV_RATE_DEFAULT,
                      refuerzo=II.REFUERZO_DEFAULT, decay=II.DECAY_DEFAULT, p_exp=II.P_EXP_DEFAULT):
        # mismo calculo, pero computando expansion/gravedad ANTES que el roce (orden de codigo invertido)
        Nn = T.shape[0]
        s = W.sum(axis=1); s_safe = np.maximum(s, 1e-12); s_bar = max(float(s.mean()), 1e-12)
        w0_ef = s_bar / max(Nn - 1, 1)
        exp_factor = np.exp(-p_exp * (s[:, None] + s[None, :]) / (2.0 * s_bar))
        cold = np.clip(1.0 - T, 0.0, None)
        dW_grav = grav_rate * np.outer(cold, cold) * w0_ef
        np.fill_diagonal(dW_grav, 0.0)
        D = T[None, :] - T[:, None]; contraste = np.clip(D, 0.0, None)
        raw = tasa_flujo * W * contraste / s_safe[:, None]
        raw_out = raw.sum(axis=1)
        escala = np.where(raw_out > 1e-12, np.minimum(1.0, T / np.maximum(raw_out, 1e-12)), 1.0)
        sent = raw * escala[:, None]
        T_nuevo = np.clip(T - sent.sum(axis=1) + sent.sum(axis=0), 0.0, None)
        roce_pair = sent + sent.T
        reinforce_factor = np.where(roce_pair > 1e-15, 1.0 + refuerzo, 1.0)
        W_nuevo = (W + dW_grav) * reinforce_factor * decay * exp_factor
        np.fill_diagonal(W_nuevo, 0.0)
        return T_nuevo, np.clip(W_nuevo, 0.0, None), roce_pair

    Ta, Wa, _ = paso_orden_A(T.copy(), W.copy())
    Tb, Wb, _ = paso_orden_B(T.copy(), W.copy())
    dT = np.max(np.abs(Ta - Tb)); dW = np.max(np.abs(Wa - Wb))
    ok = dT < TOL and dW < TOL
    return _linea(ok, "S2 (orden operadores)", f"max|dT|={dT:.2e}  max|dW|={dW:.2e}")


def s3_orden_pares(N=40, n_focos=2, delta=1e-4, pasos=1):
    """S3: recorrer/computar por BLOQUES (partir N en dos mitades, computar cada bloque de filas por
    separado y reensamblar) debe dar el mismo resultado que la formula vectorizada completa -- descarta
    dependencia oculta del orden de iteracion sobre pares."""
    T, W = II.estado_inicial(N, n_focos, delta)
    T_full, W_full, _ = II.paso_ii_det(T.copy(), W.copy())

    # recomputo SOLO T por bloques de filas (la formula de T es por-fila, se puede partir exactamente)
    mitad = N // 2
    s = W.sum(axis=1); s_safe = np.maximum(s, 1e-12)
    D = T[None, :] - T[:, None]
    contraste = np.clip(D, 0.0, None)
    bloques_T = []
    for lo, hi in [(0, mitad), (mitad, N)]:
        raw_b = II.TASA_FLUJO_DEFAULT * W[lo:hi] * contraste[lo:hi] / s_safe[lo:hi, None]
        raw_out_b = raw_b.sum(axis=1)
        escala_b = np.where(raw_out_b > 1e-12, np.minimum(1.0, T[lo:hi] / np.maximum(raw_out_b, 1e-12)), 1.0)
        sent_b = raw_b * escala_b[:, None]
        bloques_T.append((lo, hi, sent_b))
    sent_full = np.zeros((N, N))
    for lo, hi, sent_b in bloques_T:
        sent_full[lo:hi] = sent_b
    T_bloques = np.clip(T - sent_full.sum(axis=1) + sent_full.sum(axis=0), 0.0, None)

    dT = np.max(np.abs(T_full - T_bloques))
    ok = dT < TOL
    return _linea(ok, "S3 (orden pares/bloques)", f"max|dT| full-vs-bloques={dT:.2e}")


def s4_gauge_w(N=40, n_focos=2, delta=1e-4, pasos=25):
    """S4: W(0)=1 vs W(0)=1e-3 vs W(0)=1e3 (incluyendo la unidad correspondiente) no debe cambiar la
    TOPOLOGIA RELATIVA -- se compara el patron relativo por fila (W_ij / s_i), invariante a escala."""
    patrones = []
    for w0 in [1.0, 1e-3, 1e3]:
        T, W = II.estado_inicial(N, n_focos, delta, w0=w0)
        for _ in range(pasos):
            T, W, _ = II.paso_ii_det(T, W)
        s = np.maximum(W.sum(axis=1), 1e-300)
        patrones.append(W / s[:, None])   # patron relativo por fila, adimensional
    d01 = np.max(np.abs(patrones[0] - patrones[1]))
    d02 = np.max(np.abs(patrones[0] - patrones[2]))
    ok = d01 < 1e-6 and d02 < 1e-6   # tolerancia mas laxa: hay divisiones por numeros muy chicos/grandes
    return _linea(ok, "S4 (gauge W0)", f"max|patron(w0=1)-patron(w0=1e-3)|={d01:.2e}  "
                  f"max|patron(w0=1)-patron(w0=1e3)|={d02:.2e}")


def s5_resolucion(delta=1e-4, pasos=40, frac_focos=0.05):
    """S5: al crecer N (foco como FRACCION constante, no conteo fijo), las tasas POR NODO (|dT| promedio,
    |dW| promedio por fila) no deben crecer solo por el numero de pares -- deben normalizarse por fortaleza/
    (N-1), verificable comparando el ORDEN DE MAGNITUD entre N chico y N grande."""
    resultados = []
    for N in [100, 400, 1600]:
        n_focos = max(1, int(round(N * frac_focos)))
        T, W = II.estado_inicial(N, n_focos, delta)
        dT_prom = []
        for _ in range(pasos):
            T_nuevo, W_nuevo, _ = II.paso_ii_det(T, W)
            dT_prom.append(np.mean(np.abs(T_nuevo - T)))
            T, W = T_nuevo, W_nuevo
        resultados.append((N, float(np.mean(dT_prom))))
    valores = [v for _, v in resultados]
    ratio = max(valores) / max(min(valores), 1e-300)
    ok = ratio < 10.0   # no deberia crecer en ordenes de magnitud solo por N mayor
    detalle = ", ".join(f"N={N}:|dT|_prom={v:.2e}" for N, v in resultados) + f"  ratio_max/min={ratio:.2f}"
    return _linea(ok, "S5 (resolucion N)", detalle)


def s6_auditoria_rng(N=30, n_focos=1, delta=1e-4, pasos=20):
    """S6: cero llamadas a RNG antes/durante el paso determinista. Se audita el CODIGO EJECUTABLE de la
    funcion (docstring excluido -- __doc__ puede mencionar 'np.random' en prosa sin que sea una llamada)."""
    import inspect
    fuente = inspect.getsource(II.paso_ii_det)
    partes = fuente.split('"""')
    # partes[0]=firma antes del docstring, partes[1]=docstring, partes[2:]=resto del codigo ejecutable
    fuente_sin_docstring = partes[0] + '"""'.join(partes[2:]) if len(partes) >= 3 else fuente
    prohibidos = ["rng.", "np.random", "random.random", ".choice(", ".shuffle(", ".permutation("]
    hallados = [p for p in prohibidos if p in fuente_sin_docstring]
    ok = len(hallados) == 0
    return _linea(ok, "S6 (auditoria RNG)", f"tokens prohibidos en CODIGO (sin docstring): {hallados}")


def s7_no_go(N=100, n_focos=1, delta=1e-4, pasos=80):
    """S7 -- EL TEST CRITICO: con 1 foco, TODOS los nodos 'tibios' (indices n_focos..N-1) deben permanecer
    IDENTICOS entre si (T y fila de W) durante TODA la corrida, a precision numerica -- no O(1) como el
    motor ingenuo que CS encontro (amplificacion de ruido 1e-15 -> O(1) en 40 pasos via W*(1+k*dT)). Si
    aqui aparece dispersion >> ruido de punto flotante acumulado, es un BUG de esta implementacion."""
    T, W = II.estado_inicial(N, n_focos, delta)
    dispersiones_T = []
    dispersiones_W = []
    iu_tibios = np.triu_indices(N - n_focos, k=1)   # SOLO fuera de diagonal -- la diagonal es 0 por
                                                     # construccion (no forma parte de la simetria a probar)
    for _ in range(pasos):
        T, W, _ = II.paso_ii_det(T, W)
        tibios = T[n_focos:]
        dispersiones_T.append(float(np.std(tibios)))
        filas_tibias = W[n_focos:][:, n_focos:][iu_tibios]
        dispersiones_W.append(float(np.std(filas_tibias)))
    disp_T_final = dispersiones_T[-1]
    disp_W_final = dispersiones_W[-1]
    # umbral: ruido de punto flotante acumulado tras ~80 pasos de operaciones float64 -- generoso, 1e-9
    ok = disp_T_final < 1e-9 and disp_W_final < 1e-9
    return _linea(ok, "S7 (no-go, CRITICO)",
                  f"std(T_tibios)[final]={disp_T_final:.2e}  std(W_tibios)[final]={disp_W_final:.2e}  "
                  f"(umbral 1e-9; trayectoria std(T): {[f'{d:.1e}' for d in dispersiones_T[::20]]})")


def _lectura_onset_persistencia(W, N, frac_umbral, rng):
    """Ancla la lectura de diam/d_s en el nivel donde frac_gigante ALCANZA por primera vez frac_umbral --
    no en el nivel que maximiza beta (prohibido por §7.2), un criterio idéntico para toda N."""
    bloques = F._bloques_de_empate(W, N)
    uf = F._UnionFind(N)
    adj = [set() for _ in range(N)]
    total = N * (N - 1) // 2
    incl = 0
    for _, pares in bloques:
        for (i, j) in pares:
            uf.union(i, j); adj[i].add(j); adj[j].add(i)
        incl += len(pares)
        if uf.tam_max() / N >= frac_umbral:
            import cs071_histeresis as S71
            import cs064_smoke as SM
            diam = S71._diam_robusto(adj, N, rng)
            ds = SM.dim_volumen(adj, N, rng=rng)
            return dict(frac_pares=incl / total, diam=diam, d_s=ds, frac_gigante=uf.tam_max() / N)
    return dict(frac_pares=float("nan"), diam=float("nan"), d_s=float("nan"), frac_gigante=uf.tam_max() / N)


def s8_control_positivo(sides=(8, 12, 16, 20), xi=1.5, seed=11):
    """S8: sustrato métrico CONOCIDO (2D, declarado sólo como prueba del instrumento -- Codex §8 brazo 5,
    NO participa de la afirmación de origen). El lector (filtración + onset de persistencia) debe detectar
    beta no-degenerado (consistente con metrica genuina) Y el 2º sello debe dar delta finito/no-degenerado
    en AMBAS transformaciones -- contraste con S9 (W uniforme), donde debe salir degenerado (0/nan)."""
    rng = RNG(seed)
    Ns, diams = [], []
    for side in sides:
        W, N = F.W_control_positivo_2d(side=side, xi=xi)
        r = _lectura_onset_persistencia(W, N, frac_umbral=0.9, rng=rng)
        Ns.append(N); diams.append(r["diam"])
    x = np.log(Ns); y = np.log(diams)
    A = np.vstack([x, np.ones_like(x)]).T
    beta, _ = np.linalg.lstsq(A, y, rcond=None)[0]

    W_mid, N_mid = F.W_control_positivo_2d(side=sides[len(sides) // 2], xi=xi)
    sello = F.segundo_sello(W_mid, N_mid, rng, n_landmarks=40, n_quad=300)
    sello_no_degenerado = (np.isfinite(sello["delta_gromov_log"]) and sello["delta_gromov_log"] > 1e-6 and
                            np.isfinite(sello["delta_gromov_inv"]) and sello["delta_gromov_inv"] > 1e-6)
    ok = (0.25 < beta < 0.75) and sello_no_degenerado
    return _linea(ok, "S8 (control positivo)",
                  f"beta={beta:.3f} (Ns={list(Ns)}, diams={diams})  sello@N={N_mid}: "
                  f"delta_log={sello['delta_gromov_log']:.3f} delta_inv={sello['delta_gromov_inv']:.3f}")


def s9_empate_uniforme(N=200, n_focos=2, delta=1e-4, seed=13):
    """S9 -- EL TEST CLAVE DEL LECTOR: una W perfectamente uniforme NO debe adquirir topologia por el
    procedimiento de filtracion. Si el lector 'inventa' estructura de un empate total, esta roto."""
    rng = RNG(seed)
    T, W = II.estado_inicial(N, n_focos, delta)   # W uniforme (T con eps no importa aqui, W SI importa)
    bloques = F._bloques_de_empate(W, N)
    n_bloques = len(bloques)
    j = F.jueces_continuos_sin_umbral(W, N)
    sello = F.segundo_sello(W, N, rng, n_landmarks=min(40, N - 1), n_quad=200)
    degenerado_sello = ((not np.isfinite(sello["delta_gromov_log"]) or sello["delta_gromov_log"] < 1e-9) and
                         (not np.isfinite(sello["delta_gromov_inv"]) or sello["delta_gromov_inv"] < 1e-9))
    ok = (n_bloques == 1) and (j["log_dispersion"] < 1e-9) and degenerado_sello
    return _linea(ok, "S9 (empate uniforme)",
                  f"n_bloques={n_bloques} (debe ser 1) log_dispersion={j['log_dispersion']:.2e} "
                  f"max_h={j['max_h']:.4f} (=1/N esperado) sello={sello}")


def main():
    print("=" * 100, flush=True)
    print("CS072-II -- PUERTA S (S0-S9 completa)", flush=True)
    print("=" * 100, flush=True)
    resultados = {
        "S0": s0_epsilon_cero(),
        "S1": s1_permutacion(),
        "S2": s2_orden_operadores(),
        "S3": s3_orden_pares(),
        "S4": s4_gauge_w(),
        "S5": s5_resolucion(),
        "S6": s6_auditoria_rng(),
        "S7": s7_no_go(),
        "S8": s8_control_positivo(),
        "S9": s9_empate_uniforme(),
    }
    print("\n" + "=" * 100, flush=True)
    todas = all(resultados.values())
    print(f"RESULTADO PUERTA S (S0-S9): {'TODAS PASAN -- PUERTA S COMPLETA' if todas else 'HAY FALLAS -- NO avanzar a exploratoria/fold'}",
          flush=True)
    print("=" * 100, flush=True)
    return resultados


if __name__ == "__main__":
    main()
