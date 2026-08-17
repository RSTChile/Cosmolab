"""
p_gravedad_general.py — PIEZA: GRAVEDAD GENERAL (régimen métrico, sobre posiciones 3D reales).

Distinta de `p02_gravedad.py` (Bgrav, umbral de proximidad TÉRMICA -- un escalar 1D, régimen
relacional-cuántico, prototipos CS073 probaron que NUNCA fragmenta). Ésta es F=G·m_i·m_j/r² sobre
posiciones 3D reales -- régimen métrico, el que puede fragmentar en estructuras separadas.

SUPERA al Paso A anterior (`p02b_gravedad_general.py`, deprecado -- ver adjudicación Q3 en
INSTRUCCION_CC_cierre_holistico.md v3): NO se derivan posiciones de la malla causal relacional (dio
negativo sólido, z<1 a 750 átomos -- el sustrato relacional no despliega 3D). Las posiciones son el
ESCENARIO (D=3, la dimensión fosilizada YA probada en CS072), no una claim a re-derivar: un volumen 3D
UNIFORME donde se posan las partículas, portando el campo de densidad #23 como perturbación de masa/
densidad LOCAL -- no como coordenada. El 3D no se decreta como resultado (eso sería Shannon); se usa
como escenario porque CS072 ya lo estableció como la dimensión de este universo.

Convenciones adimensionales (declaradas, no ocultas -- ninguna se ajustó para que el resultado "salga
bien", se fijan ANTES de correr):
  G_ADIM = 1.0        -- define la unidad de fuerza del sistema (estándar en N-cuerpos de juguete: fija
                          unidades, no mide ni ajusta el universo real).
  SOFTENING = 0.1×L0   -- necesidad numérica (evita la singularidad r->0 en pares casi-coincidentes),
                          NO un parámetro físico. L0 = espaciamiento medio inicial.
"""
import numpy as np
from scipy.spatial import cKDTree

G_ADIM = 1.0


def posiciones_escenario(n, lado=None, seed=12345):
    """Volumen 3D UNIFORME (el escenario, D=3 ya fosilizado). Semilla FIJA e independiente de cualquier
    valor físico (densidad, masa, índice) -- las posiciones no cargan información, son el contenedor
    neutro; lo que carga información es la densidad #23 que se adjunta DESPUÉS, en cada partícula.
    lado=None -> caja de lado n**(1/3) (espaciamiento medio == 1, unidad de longitud del sistema)."""
    L = float(lado) if lado is not None else float(n) ** (1.0 / 3.0)
    return np.random.default_rng(seed).uniform(0.0, L, size=(n, 3)), L


class GravedadGeneral:
    numero = "grav_gen"
    nombre = "gravedad general (F=G·m·m/r², posiciones 3D)"
    nivel = "estructura"

    def __init__(self, activa=True, G=G_ADIM, softening=None):
        self.activa_flag = activa
        self.G = G
        self.softening = softening   # se fija con L0 al construir el escenario (ver orquestador)

    def activa(self):
        return self.activa_flag

    def aceleraciones(self, pos, masa):
        """a_i = G * sum_j!=i m_j (pos_j - pos_i) / (|r_ij|^2 + eps^2)^(3/2). Vectorizado (numpy), TODAS
        las partículas (bariones + CDM) se atraen entre sí -- la gravedad es universal; lo que distingue
        a la materia oscura es que NO siente presión/enfriamiento (ver p_materia_oscura_halo.py), no que
        gravite distinto."""
        if not self.activa_flag:
            return np.zeros_like(pos)
        eps = self.softening if self.softening is not None else 1e-3
        diff = pos[None, :, :] - pos[:, None, :]                 # (N,N,3): r_j - r_i
        r2 = np.sum(diff ** 2, axis=-1) + eps ** 2                # (N,N)
        np.fill_diagonal(r2, np.inf)                               # sin autointeracción
        inv_r3 = r2 ** (-1.5)
        acc = self.G * np.sum(inv_r3[:, :, None] * diff * masa[None, :, None], axis=1)
        return acc

    def aceleraciones_adaptativas(self, pos, k=6, masa=None):
        """DISENO_CS073_ignicion_PARA_CC.md: resolución ADAPTATIVA -- ninguna constante nueva elegida a
        mano, ambas cantidades son funciones de la densidad local (G-PARAMETROS-ESTRUCTURALES):
          eps_i = distancia al k=6-ésimo vecino más cercano (== rho_local^(-1/3) salvo constante; k=6 es
                  el que ya usa p_enfriamiento_H2.py -- ninguna constante nueva).
          rho_i = k / ((4/3)pi eps_i^3) -- la MISMA cantidad que alimenta el softening por partícula Y
                  el rho_local de H2/Jeans (adjudicación CS: una sola resolución, no dos pisos que se
                  desincronicen -- ver p_enfriamiento_H2.actualizar(rho_externo=...)).
        Softening por PAR: eps_ij = (eps_i+eps_j)/2 -- promedio simple, la simetrización estándar cuando
        cada partícula trae su propio suavizado (no está en el texto del diseño palabra por palabra;
        se documenta aquí explícitamente por transparencia, es la convención más común, no elegida a
        dedo para el resultado).
        Devuelve (aceleraciones, eps_i, rho_i, rho_max) -- rho_max lo consume paso_tiempo_adaptativo()."""
        n = len(pos)
        m = masa if masa is not None else np.ones(n)
        if not self.activa_flag:
            return np.zeros_like(pos), np.full(n, np.inf), np.zeros(n), 0.0
        kk = min(k, n - 1)
        if kk < 1:
            return np.zeros_like(pos), np.full(n, np.inf), np.zeros(n), 0.0
        tree = cKDTree(pos)
        dist, _ = tree.query(pos, k=kk + 1)
        eps_i = dist[:, -1]
        eps_i = np.maximum(eps_i, 1e-12)   # piso puramente numérico (evita división por cero exacta), no físico
        rho_i = kk / ((4.0 / 3.0) * np.pi * eps_i ** 3)
        rho_max = float(rho_i.max()) if n else 0.0

        diff = pos[None, :, :] - pos[:, None, :]                  # (N,N,3)
        eps_par = (eps_i[:, None] + eps_i[None, :]) / 2.0          # (N,N) -- eps_ij simetrizado
        r2 = np.sum(diff ** 2, axis=-1) + eps_par ** 2
        np.fill_diagonal(r2, np.inf)
        inv_r3 = r2 ** (-1.5)
        acc = self.G * np.sum(inv_r3[:, :, None] * diff * m[None, :, None], axis=1)
        return acc, eps_i, rho_i, rho_max


def paso_tiempo_adaptativo(rho, G=G_ADIM, eta=0.1, dt_min=1e-6, dt_max=0.05):
    """Delta_t = eta * t_ff(rho), t_ff=sqrt(3*pi/(32*G*rho)) -- la fórmula ESTÁNDAR de tiempo de caída
    libre (ya citada en INVENTARIO_atomo_a_estrella_CS.md), eta=0.1 = convención de ~10 pasos por tiempo
    dinámico en N-cuerpos (DISENO_CS073_ignicion_PARA_CC.md). Acepta un escalar (rho_max, esquema global,
    v1) O un array (rho_i por partícula, esquema jerárquico, v2) -- misma fórmula, aplicada donde toque.
    dt_min/dt_max son VÁLVULAS DE SEGURIDAD numéricas, no resolución elegida."""
    rho = np.asarray(rho, dtype=float)
    escalar = rho.ndim == 0
    rho = np.atleast_1d(rho)
    t_ff = np.where(rho > 0, np.sqrt(3.0 * np.pi / (32.0 * G * np.maximum(rho, 1e-300))), dt_max / eta)
    dt = np.clip(eta * t_ff, dt_min, dt_max)
    return float(dt[0]) if escalar else dt


def energia_total(pos, vel, masa, eps_i, G=G_ADIM):
    """E = KE + PE, con el MISMO softening por-par (eps_ij=(eps_i+eps_j)/2) que usan las aceleraciones
    -- para que la energía que se mide sea la energía DEL SISTEMA REGULARIZADO que el integrador
    realmente evoluciona (si se midiera con un softening distinto, un ΔE espurio no distinguiría error
    de integración de la diferencia de convención). Diagnóstico de cordura del integrador (DISENO_CS073_
    ignicion_PARA_CC.md v2): sólo tiene sentido como conservación en un tramo de gravedad PURA (sin
    expansión ni kicks térmicos, que inyectan/quitan energía por diseño, no por error numérico)."""
    KE = 0.5 * float(np.sum(masa * np.sum(vel ** 2, axis=-1)))
    diff = pos[None, :, :] - pos[:, None, :]
    r2 = np.sum(diff ** 2, axis=-1)
    eps_par2 = ((eps_i[:, None] + eps_i[None, :]) / 2.0) ** 2
    r_soft = np.sqrt(r2 + eps_par2)
    np.fill_diagonal(r_soft, np.inf)
    PE = -G * float(np.sum(np.triu(masa[:, None] * masa[None, :] / r_soft, k=1)))
    return KE + PE


def paso_bloques_jerarquico(pos, vel, masa, grav, k=6, dt_budget=0.05, eta=0.1, p_max=14):
    """PASO DE TIEMPO INDIVIDUAL/JERÁRQUICO (DISENO_CS073_ignicion_PARA_CC.md v2, la extensión estándar
    de N-cuerpos -- Gadget et al. -- resuelve el problema real que CC encontró en v1: un Delta_t GLOBAL
    atado al punto MÁS denso obligaba a TODO el sistema a integrarse al paso del par más extremo.

    Esquema KDK por bloques, SIMÉTRICO (leapfrog propio, no un kick único por activación -- la primera
    versión de esto violó la simetría medio-kick/medio-kick y dio |dE/E|=0.13, reprobado; corregido
    aquí): TODAS las posiciones se arrastran ('drift') cada subpaso fino común (barato -- sólo usa la
    velocidad ya conocida). La velocidad de cada partícula recibe:
      - un MEDIO-kick de APERTURA (dt_i/2, fuerza en t=0) al principio del bloque,
      - un kick COMPLETO (dt_i) en cada activación intermedia (cierra el intervalo anterior + abre el
        siguiente -- la forma estándar del leapfrog sincronizado),
      - un MEDIO-kick de CIERRE (dt_i/2) en la ÚLTIMA activación del bloque (todas las partículas
        completan su último intervalo exactamente en el último subpaso fino, por construcción: n_fine
        es múltiplo de todos los stride_i).
    dt_i = eta*t_ff(rho_i) propio de cada partícula (misma fórmula, misma rho_i de siempre, por
    partícula en vez de global), cuantizado a potencias de 2 del presupuesto dt_budget (sincronización
    estándar por bloques). El ahorro es real: la fuerza (cara, O(N^2)) sólo se recalcula en los
    subpasos donde ALGUIEN se activa, y el kick sólo se aplica a quien de verdad lo necesita.

    Devuelve (pos, vel, n_fine, n_kicks_totales, ok) -- n_kicks_totales para diagnosticar el ahorro real
    (si == n_fine*n, no hubo ahorro; si << eso, el esquema jerárquico funcionó)."""
    n = len(pos)
    acc0, eps_i, rho_i, _rho_max = grav.aceleraciones_adaptativas(pos, k=k, masa=masa)
    dt_i_deseado = paso_tiempo_adaptativo(rho_i, eta=eta)
    with np.errstate(divide="ignore", over="ignore"):
        p_i = np.ceil(np.log2(np.maximum(dt_budget / np.maximum(dt_i_deseado, 1e-300), 1.0)))
    p_i = np.clip(np.nan_to_num(p_i, nan=0.0, posinf=p_max), 0, p_max).astype(int)
    p_max_usado = int(p_i.max())
    n_fine = 2 ** p_max_usado
    dt_fine = dt_budget / n_fine
    stride = 2 ** (p_max_usado - p_i)      # partícula i completa un intervalo cada stride[i] subpasos finos
    dt_i_real = stride * dt_fine            # su Delta_t individual real (post-cuantización)

    vel = vel + 0.5 * dt_i_real[:, None] * acc0    # medio-kick de APERTURA, fuerza en t=0
    n_kicks_totales = n

    for k_fine in range(n_fine):
        pos = pos + vel * dt_fine
        if not np.all(np.isfinite(pos)):
            return pos, vel, n_fine, n_kicks_totales, False
        completado = ((k_fine + 1) % stride) == 0
        if not completado.any():
            continue
        acc_full, eps_i, rho_i, _ = grav.aceleraciones_adaptativas(pos, k=k, masa=masa)
        n_kicks_totales += int(completado.sum())
        es_cierre_bloque = (k_fine == n_fine - 1)     # última activación de TODOS a la vez (por construcción)
        factor = 0.5 if es_cierre_bloque else 1.0      # medio-kick de CIERRE vs kick completo intermedio
        vel[completado] = vel[completado] + factor * dt_i_real[completado, None] * acc_full[completado]
    return pos, vel, n_fine, n_kicks_totales, True
