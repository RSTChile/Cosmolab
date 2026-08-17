"""
null2_generar_ic.py — Fase II CS073, escalón NULL-2 ("aleatorización de fases", 2do peldaño de la
jerarquía de 6 controles propuesta por el roadmap multi-IA del 5-ago-2026).

Qué pregunta aísla, en simple: NULL-1 (ver NULL1_piloto_distribucion_radial_CS.md) ya mostró que
"tener el mismo perfil radial que REAL" NO alcanza para formar sumideros -- hace falta algo más. Pero
el perfil radial es una estadística muy pobre (1 número por partícula: la distancia al centro). NULL-2
pregunta algo más fino: ¿alcanza con que la nube tenga la MISMA TEXTURA de agrupamiento de a PARES
(la misma función de correlación de dos puntos / el mismo espectro de potencia P(k) -- "¿cuánto se
parece la densidad en un punto a la densidad a distancia r?", promediado sobre toda la nube), o hace
falta la estructura de orden SUPERIOR (3+ puntos -- filamentos, jerarquías, "quién está cerca de quién
que a su vez está cerca de quién") que sólo la malla causal genuina produce?

Método (estándar de la literatura de cosmología observacional, no inventado para este experimento):
ALEATORIZACIÓN DE FASES. Se toma el campo de densidad de REAL, se lo grilla en una malla 3D regular,
se calcula su transformada de Fourier F(k) = |F(k)| * exp(i*fase(k)). El módulo |F(k)| en cada modo
es, por el teorema de Wiener-Khinchin, exactamente lo que determina P(k) (y por lo tanto ξ(r), la
función de correlación de dos puntos) -- así que CONSERVAR |F(k)| modo a modo conserva P(k) EXACTO.
La FASE, en cambio, es donde vive toda la información de orden superior (dónde exactamente caen los
picos entre sí -- la "forma" de los filamentos). Reasignarle a cada modo una fase aleatoria y
transformar de vuelta produce el campo GAUSSIANO más parecido a REAL que existe: mismo P(k)/ξ(r) por
construcción, cero correlación de orden superior más allá de la que ya implica un campo gaussiano.

Truco de implementación (evita re-derivar a mano la simetría hermítica que un campo real exige en su
FFT): en vez de fabricar fases aleatorias "sueltas" modo a modo (lo que rompería la simetría F(-k) =
conj(F(k)) y daría una transformada inversa con parte imaginaria espuria), se genera un campo de RUIDO
BLANCO real e independiente del mismo tamaño de grilla, se le toma SU FFT, y se usa la FASE de ESE
campo de ruido (que por construcción, al venir de un campo real, ya respeta la simetría hermítica
exacta) combinada con el MÓDULO del campo real de REAL. Esto es el mismo truco que usa el método de
"phase randomization" para generar subrogados en análisis de series no lineales (Theiler et al. 1992),
aplicado aquí en 3D a un campo de densidad en vez de en 1D a una serie temporal -- broadly conocido,
no inventado para este experimento.

Conversión campo->partículas: dado que Phantom necesita POSICIONES de partícula (no un campo continuo),
se muestrean N partículas proporcionalmente a la densidad del campo sintético (clip a 0 donde el
campo gaussiano dio densidad negativa -- artefacto esperado de linealizar un campo que en realidad es
positivo-definido, ver limitaciones abajo) usando muestreo por transformada inversa sobre la
distribución de probabilidad discreta de las celdas de la grilla (equivalente exacto, y mucho más
rápido a este N, que un rechazo aceptar/rechazar literal con envolvente uniforme) + un jitter uniforme
dentro de cada celda (para no dejar partículas exactamente sobre nodos de grilla, lo que Phantom
malinterpretaría como coincidencias degeneradas de suavizado).

Limitaciones documentadas (a propósito, no ocultas):
  - Resolución de grilla: con N~500-2000 partículas discretas, una grilla de ~12-20 celdas por lado da
    una ocupación media de sólo ~0.25-0.5 partículas/celda -- el campo está dominado por ruido de Poisson
    de muestreo (shot noise), no por señal física fina. Esto es aceptable para el propósito de NULL-2
    (que sólo necesita reproducir la textura de GRAN escala de dos puntos, no sub-estructura fina), pero
    significa que el campo gaussiano sintético es, en el límite N pequeño, más "ruido con forma global
    correcta" que un campo suave -- se documenta, no se disimula.
  - El campo sintético puede tener densidad negativa en algunas celdas (artefacto de gaussianizar un
    campo que físicamente es positivo-definido) -- se clipea a 0 antes de muestrear partículas. Esto
    introduce un sesgo de conservación de masa pequeño (se pierde la masa que habría caído en celdas
    negativas) que NO afecta la comparación de dos puntos (se verifica aparte, ver
    null2_disenar_verificar.py) pero sí reduce levemente la masa total disponible para muestrear -- no
    relevante aquí porque siempre se pide N partículas fijas, no una masa fija por celda.
  - La verificación de dos puntos se hace con ξ(r) vía conteo de pares (pdist + KS de 2 muestras sobre
    la distribución de distancias par-a-par), NO con P(k) de partícula discreta -- ver
    null2_disenar_verificar.py para la justificación (P(k) de N=500-2000 partículas discretas sobre una
    grilla FFT tiene MÁS ruido de muestreo que ξ(r) por pares directos, que no requiere grillar de nuevo).

No toca: leer_volcado_phantom.py, fase1_traducir_a_phantom.py, null1_generar_ic.py, campo_velocidad_
turbulento.py, ni ninguna carpeta bateria_*/ (todo de sólo lectura/import). No escribe nada bajo
bateria_n2000/, bateria_null1_n2000/, ni bateria_real_extra_n2000/.
"""
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import ks_2samp


# ------------------------------------------------------------------------------------------------
# 1) Partículas -> campo de densidad en grilla regular (NGP: nearest-grid-point vía histogramdd,
#    simple y suficiente dado que N es pequeño -- un esquema CIC más suave no cambiaría la conclusión
#    de qué tan ruidoso es el campo a este N, sólo la estética).
# ------------------------------------------------------------------------------------------------
def gridizar(pos, ngrid, pad=1.05):
    """Grilla las posiciones (N,3) en un cubo de ngrid^3 celdas centrado en el centro de masa, con el
    lado suficiente para cubrir la partícula más lejana * pad. Devuelve (campo, cell_size, origin,
    centro) -- origin es la esquina inferior del cubo en coordenadas absolutas (no relativas a COM)."""
    centro = pos.mean(axis=0)
    half_extent = float(np.abs(pos - centro).max()) * pad
    edges = np.linspace(-half_extent, half_extent, ngrid + 1)
    campo, _ = np.histogramdd(pos - centro, bins=(edges, edges, edges))
    cell_size = edges[1] - edges[0]
    origin = centro - half_extent
    return campo.astype(float), float(cell_size), origin, centro, half_extent


def pk_radial(campo, cell_size, nbins=15):
    """P(k) del campo de sobre-densidad delta=campo/media-1, promediado en cáscaras esféricas de k.
    Sólo para diagnóstico/diseño (Paso 1-2) -- NO es el método de verificación final (ver docstring del
    módulo, se usa ξ(r) por pares en su lugar para la validación de partículas)."""
    ngrid = campo.shape[0]
    media = campo.mean()
    # NOTA: el campo aleatorizado en fase puede terminar con media NEGATIVA (la fase del modo k=0,
    # que es real por construcción, se reasigna a 0 o pi -- ver aleatorizar_fases -- lo que puede
    # voltear el signo de la suma total sin alterar su magnitud). Guardar contra media==0 (grilla
    # vacía), NO contra media<0 -- lo segundo es un resultado válido y frecuente de la aleatorización,
    # no un error (bug real detectado y corregido en null2_disenar_verificar.py: con "media>0" como
    # guardia, delta se ponía en cero para CUALQUIER campo sintético con media negativa, dando P(k)=0
    # espurio en vez del valor real).
    delta = campo / media - 1.0 if media != 0 else campo * 0.0
    Fk = np.fft.fftn(delta)
    pk3d = np.abs(Fk) ** 2
    kfreq = np.fft.fftfreq(ngrid, d=cell_size) * 2 * np.pi
    kx, ky, kz = np.meshgrid(kfreq, kfreq, kfreq, indexing="ij")
    kmag = np.sqrt(kx**2 + ky**2 + kz**2)
    kbins = np.linspace(0, kmag.max(), nbins + 1)
    pk_binned = np.full(nbins, np.nan)
    for i in range(nbins):
        m = (kmag >= kbins[i]) & (kmag < kbins[i + 1])
        if m.sum() > 0:
            pk_binned[i] = pk3d[m].mean()
    kcenters = 0.5 * (kbins[:-1] + kbins[1:])
    return kcenters, pk_binned


# ------------------------------------------------------------------------------------------------
# 2) El generador NULL-2 propiamente dicho: aleatorización de fases.
# ------------------------------------------------------------------------------------------------
def aleatorizar_fases(campo, seed):
    """Devuelve (campo_sintetico, residuo_imaginario_max). El residuo imaginario debe ser ~0 (error de
    punto flotante, no señal) -- se reporta para verificar que la simetría hermítica se respetó."""
    rng = np.random.default_rng(seed)
    ruido = rng.standard_normal(campo.shape)          # campo real independiente -> su FFT ya respeta
    Fk_real = np.fft.fftn(campo)                       # la simetría hermítica exacta que un campo
    Fk_ruido = np.fft.fftn(ruido)                       # real requiere (truco de "phase randomization").
    amp = np.abs(Fk_real)
    fase = np.angle(Fk_ruido)
    Fk_sint = amp * np.exp(1j * fase)
    campo_sint_c = np.fft.ifftn(Fk_sint)
    residuo_imag = float(np.abs(campo_sint_c.imag).max())
    return campo_sint_c.real, residuo_imag


def muestrear_particulas_de_campo(campo, n, cell_size, origin, seed):
    """Puebla n posiciones de partícula proporcionalmente a `campo` (clipeado a >=0). Muestreo por
    transformada inversa sobre la PMF discreta de celdas (equivalente exacto y más eficiente que
    aceptar/rechazar literal a este N) + jitter uniforme dentro de cada celda."""
    rng = np.random.default_rng(seed)
    campo_pos = np.clip(campo, 0.0, None)
    total = campo_pos.sum()
    if total <= 0:
        raise ValueError("campo sintético sin densidad positiva -- revisar aleatorizar_fases/semilla")
    prob = campo_pos.ravel() / total
    idx_flat = rng.choice(prob.size, size=n, p=prob)
    idx = np.array(np.unravel_index(idx_flat, campo.shape)).T  # (n,3) índices enteros de celda
    jitter = rng.uniform(0.0, 1.0, size=(n, 3))
    pos = origin + (idx + jitter) * cell_size
    return pos


def generar_null2(pos_real, n_salida, ngrid, seed, pad=1.05):
    """Orquesta el paso campo NULL-2 completo: grilla pos_real, aleatoriza fases con `seed`, muestrea
    n_salida partículas del campo sintético. Devuelve dict con posiciones + diagnósticos (residuo
    imaginario, fracción de masa de campo negativa clipeada)."""
    campo, cell_size, origin, centro, half_extent = gridizar(pos_real, ngrid, pad=pad)
    campo_sint, residuo_imag = aleatorizar_fases(campo, seed)
    masa_negativa_frac = float(-campo_sint[campo_sint < 0].sum() / campo_sint[campo_sint > 0].sum())
    pos_null2 = muestrear_particulas_de_campo(campo_sint, n_salida, cell_size, origin, seed)
    return dict(pos=pos_null2, cell_size=cell_size, origin=origin, centro=centro,
                half_extent=half_extent, residuo_imag=residuo_imag,
                masa_negativa_frac=masa_negativa_frac, ngrid=ngrid, seed=seed,
                campo_real=campo, campo_sint=campo_sint)


# ------------------------------------------------------------------------------------------------
# 3) Verificación de dos puntos a nivel de PARTÍCULA (el test que realmente importa: ¿el catálogo de
#    partículas final de NULL-2 preserva la estadística de a-pares de REAL?).
# ------------------------------------------------------------------------------------------------
def verificar_dos_puntos_particulas(pos_real, pos_null2, seed_sub=0, max_particulas=None):
    """Compara la distribución de distancias par-a-par (equivalente, sin necesidad de elegir bins de
    r, a comparar ξ(r) completa) de dos catálogos de partículas con un test KS de 2 muestras. Si
    max_particulas se da y alguno de los catálogos lo excede, se sub-muestrea (mismo N en ambos lados)
    para que pdist no explote en memoria/tiempo a N grandes -- no relevante a la escala de este piloto
    (N<=2000, <=2e6 pares, trivial), pero deja el método listo para escalar."""
    rng = np.random.default_rng(seed_sub)

    def _quizas_submuestrear(pos):
        if max_particulas is not None and len(pos) > max_particulas:
            idx = rng.choice(len(pos), size=max_particulas, replace=False)
            return pos[idx]
        return pos

    a = _quizas_submuestrear(pos_real - pos_real.mean(axis=0))
    b = _quizas_submuestrear(pos_null2 - pos_null2.mean(axis=0))
    d_real = pdist(a)
    d_null2 = pdist(b)
    stat, p = ks_2samp(d_real, d_null2)
    return dict(ks_stat=float(stat), ks_p=float(p),
                d_real_mean=float(d_real.mean()), d_real_std=float(d_real.std()),
                d_null2_mean=float(d_null2.mean()), d_null2_std=float(d_null2.std()),
                n_real=len(a), n_null2=len(b))


def escribir_ic_txt(ruta_salida, pos, vel, h, masa_particula, hfact, polyk, comentario):
    """Escribe el mismo formato ASCII (cosmogenesis_ic.txt v2) que fase1_traducir_a_phantom /
    null1_generar_ic -- 1 línea comentario, 1 línea header (n masa hfact polyk), n líneas de datos."""
    n = len(pos)
    with open(ruta_salida, "w") as f:
        f.write(f"# {comentario}\n")
        f.write(f"{n} {masa_particula:.17g} {hfact} {polyk:.17g}\n")
        for i in range(n):
            f.write(f"{float(pos[i,0]):.17g} {float(pos[i,1]):.17g} {float(pos[i,2]):.17g} "
                     f"{float(vel[i,0]):.17g} {float(vel[i,1]):.17g} {float(vel[i,2]):.17g} "
                     f"{float(h[i]):.17g}\n")


if __name__ == "__main__":
    import sys
    print("Uso como módulo -- ver null2_disenar_verificar.py (Paso 1) y null2_piloto_generar.py "
          "(Paso 2, orquestador del piloto).")
    sys.exit(0)
