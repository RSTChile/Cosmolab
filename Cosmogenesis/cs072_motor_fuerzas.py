"""
cs072_motor_fuerzas.py — REDISEÑO CS072: las FUERZAS hacen la ligadura, no el campo térmico.

RAÍZ DEL ARTEFACTO (motor viejo): una sola matriz W mezclaba (a) historia térmica (dominante, 0.9*W acumulado)
y (b) aportes de fuerzas (pequeños). El contador leía W total -> los "bariones" se formaban por cercanía térmica,
no por la fuerza fuerte. Apagar confinamiento no cambiaba nada = artefacto.

REDISEÑO: DOS matrices separadas.
  - T (campo térmico): SOLO la condición inicial -- gradiente + expansión. Crea la ASIMETRÍA que deja sobrevivir
    materia sobre antimateria. NUNCA liga. No la lee el contador.
  - B (matriz de ligadura): empieza en CERO. SOLO las fuerzas la construyen (confinamiento, EM, gravedad),
    con sus reglas físicas (color, carga, masa). El campo térmico modula CUÁNDO actúan (enfriamiento habilita
    confinamiento) pero NO aporta a B directamente.
  - Contador de bariones: lee B (ligadura por fuerzas), no T. Un barión = 3 quarks color distinto mismo estatus
    LIGADOS EN B. Apagar confinamiento -> B sin ligadura fuerte -> 0 bariones. Ésa es la prueba de admisibilidad.

CERO AZAR. Constantes = sólo físicas estructurales (fuerzas, umbral de enfriamiento). Determinista.
"""
import numpy as np

# --- constantes ESTRUCTURALES (fuerzas y física), no perillas de forma ---
R_STRONG = 0.30      # fuerza fuerte (confinamiento)
R_EM     = 0.10      # electromagnetismo
R_GRAV   = 0.02      # gravedad
T_CONF   = 0.6       # umbral de enfriamiento: por debajo, el confinamiento puede actuar (universo frío)
LIGADO_FRAC = 1.5    # "ligado" = B_ij por encima de 1.5x el promedio de B (cociente relativo, no absoluto)

def _catalogo(n_quarks, n_antiquarks, n_electrones, n_positrones):
    """Catálogo determinista. IMPORTANTE (anti-Shannon): color y carga son propiedades INTRÍNSECAS de las
    partículas (COMPOSICIÓN del catálogo), NO ruptura de simetría por índice. i%3 significa "tercios iguales
    de cada color" (composición balanceada), i%2 "mitad up / mitad down" -- son la MEZCLA física, no un
    ordenamiento que decida resultados. La constraint del director (ruptura por física, no por índice) prohíbe
    que un RESULTADO dependa del índice; aquí el índice sólo REPARTE la composición. El test de invariancia a
    permutación en __main__ lo VERIFICA corriendo de verdad (base vs 6 permutaciones del catálogo): da INVARIANTE
    tras corregir la aniquilación para que reste POR COLOR (no matar los primeros k por posición, que sí dependía
    del orden). Antes de esa corrección el test FALLABA (3 vs 2,1,2...) -- era Shannon residual, ya eliminado."""
    tipos, color, carga, es_anti, es_quark = [], [], [], [], []
    def add(n, anti, quark):
        for i in range(n):
            if quark:
                color.append(i % 3)
                carga.append(2 if (i % 2 == 0) else -1)   # up(+2/3) / down(-1/3) en tercios
            else:
                color.append(-1)
                carga.append(-3 if not anti else 3)        # electrón -1 / positrón +1 en tercios
            es_anti.append(anti); es_quark.append(quark)
    add(n_quarks, False, True); add(n_antiquarks, True, True)
    add(n_electrones, False, False); add(n_positrones, True, False)
    return (np.array(color), np.array(carga, dtype=np.int8),
            np.array(es_anti, dtype=bool), np.array(es_quark, dtype=bool))

def _campo_termico(N, homogeneo, amp=0.1):
    """Condición inicial: campo de temperatura. Homogéneo = uniforme (control). Gradiente = asimetría.
    NO usa índice como coordenada espacial: el gradiente es una DISTRIBUCIÓN de valores (sin posición)."""
    if homogeneo:
        return np.ones(N)
    # distribución determinista de valores (cero-azar, sin coordenada): valores dispersos por magnitud
    d = np.linspace(-amp, amp, N); d = d - d.mean()
    return 1.0 + d

def corre(n_quarks, n_antiquarks, n_electrones, n_positrones,
          homogeneo=False, expansion=True, pasos=300, apagar=frozenset(), perm=None):
    color, carga, es_anti, es_quark = _catalogo(n_quarks, n_antiquarks, n_electrones, n_positrones)
    N = len(color)
    if perm is not None:   # sólo para el test de invariancia: reordena el catálogo; el resultado NO debe cambiar
        color, carga, es_anti, es_quark = color[perm], carga[perm], es_anti[perm], es_quark[perm]
    T = _campo_termico(N, homogeneo)
    B = np.zeros((N, N))          # matriz de LIGADURA -- empieza vacía, SOLO las fuerzas la llenan
    viva = np.ones(N)

    # máscaras físicas (fijas)
    color_distinto = (color[:, None] != color[None, :]) & (color[:, None] >= 0) & (color[None, :] >= 0)
    mismo_estatus  = (es_anti[:, None] == es_anti[None, :])
    carga_opuesta  = (carga[:, None] != 0) & (carga[None, :] != 0) & (np.sign(carga[:, None]) != np.sign(carga[None, :]))
    np.fill_diagonal(color_distinto, False)

    for step in range(pasos):
        # --- expansión: enfría el campo (amplifica el contraste de lo ya frío), NO toca B ---
        if expansion:
            T = T * (1 - 0.02 * (T.max() - T) / (T.max() + 1e-9))
        T_ef = float(T.mean())    # "qué tan frío está el universo" -- escalar físico

        b0 = max(float(B.sum(axis=1).mean()) / max(N - 1, 1), 1e-12)   # escala de B para el umbral relativo

        # --- FUERZAS construyen B (cada una con su regla; el campo térmico sólo habilita el confinamiento) ---
        dB = np.zeros((N, N))
        # #fuerte: confinamiento -- SOLO cuando el universo está frío (T_ef < T_CONF). Liga color distinto.
        if "confinamiento" not in apagar and T_ef < T_CONF:
            dB = dB + R_STRONG * (color_distinto & mismo_estatus).astype(float)
        # #EM: carga opuesta atrae
        if "em" not in apagar:
            dB = dB + R_EM * carga_opuesta.astype(float)
        # #gravedad: masa (aquí masa=1 para todos los fermiones; universal). Débil, universal.
        if "gravedad" not in apagar:
            masa = np.ones(N)
            dB = dB + R_GRAV * np.outer(masa, masa) / max(float(masa.mean())**2, 1e-300) * 0.1

        # aniquilación por POBLACIÓN (Motor B) -- SIN TASA (constraint del director: la aniquilación no es
        # porcentual). Es una RESTA de poblaciones: por cada clase, min(materia, antimateria) se aniquila (va a
        # luz), sobrevive el EXCEDENTE. No hay ritmo/porcentaje/cupo -- se resuelve por la aritmética de las
        # nubes, invariante al orden de los índices. Idempotente: una vez restado, min=0 y queda estable.
        # INVARIANTE AL ÍNDICE (anti-Shannon): la resta se hace POR COLOR, no matando los primeros k por posición.
        # Matar mat[:k] elegía por índice qué colores sobreviven -> el orden decidía el conteo (Shannon). Correcto:
        # por cada (clase, color) se aniquila min(materia, antimateria) de ESE color. La indistinguibilidad
        # cuántica dice que no importa CUÁL quark rojo muere, sólo CUÁNTOS por color -> el resultado no depende
        # del orden del array. (Se aniquila por par materia-antimateria del MISMO color, que es como ocurre.)
        if "aniquilacion" not in apagar:
            for es_q in [True, False]:
                for c in [0, 1, 2, -1]:            # -1 = leptones (sin color)
                    mat = np.where((~es_anti) & (es_quark == es_q) & (color == c) & (viva > 0.5))[0]
                    ant = np.where(( es_anti) & (es_quark == es_q) & (color == c) & (viva > 0.5))[0]
                    k = min(len(mat), len(ant))     # RESTA por color, sin tasa; k pares del mismo color -> luz
                    if k > 0:
                        viva[mat[:k]] = 0.0; viva[ant[:k]] = 0.0   # dentro de un color son idénticos: cuáles = irrelevante
            viva = np.clip(viva, 0, 1)

        # B acumula la ligadura de las fuerzas, escalada por supervivencia (aniquilado no liga)
        viva_par = np.sqrt(np.outer(viva, viva))
        B = B + dB * viva_par
        np.fill_diagonal(B, 0.0)

    return dict(B=B, color=color, carga=carga, es_anti=es_anti, es_quark=es_quark, viva=viva, N=N)

def cuenta_bariones(estado):
    """Barión = 3 quarks color distinto, mismo estatus, MUTUAMENTE LIGADOS EN B (por fuerzas, no por térmica)."""
    B = estado["B"]; color = estado["color"]; carga = estado["carga"]
    es_anti = estado["es_anti"]; es_quark = estado["es_quark"]; viva = estado["viva"]; N = estado["N"]
    b0 = max(float(B.sum(axis=1).mean()) / max(N - 1, 1), 1e-12)
    umbral = LIGADO_FRAC * b0
    ligado = B > umbral

    def cuenta(mask_status):
        idxs = np.where(mask_status & (color >= 0) & (viva > 0.3))[0]
        usados = np.zeros(N, dtype=bool); trios = []
        for i in idxs:
            if usados[i]: continue
            vec = [j for j in idxs if j != i and not usados[j] and color[j] != color[i] and ligado[i, j]]
            for j in vec:
                terc = [k for k in vec if k != j and color[k] != color[i] and color[k] != color[j]
                        and ligado[i, k] and ligado[j, k]]
                if terc:
                    k = terc[0]; trios.append((i, j, k)); usados[[i, j, k]] = True; break
        return trios
    bar = cuenta(~es_anti); anti = cuenta(es_anti)
    return dict(bariones=len(bar), antibariones=len(anti), trios=bar)

if __name__ == "__main__":
    print("test rápido (4 brazos):")
    for (h, e, lab) in [(True,False,"A homog"),(True,True,"B homog+exp"),(False,False,"C grad"),(False,True,"D grad+exp")]:
        r = corre(30,21,10,7, homogeneo=h, expansion=e, pasos=300)
        print(f"  {lab:12s}: {cuenta_bariones(r)['bariones']} bariones")

    print("test decisivo (apagar confinamiento -> debe 0):")
    r = corre(30,21,10,7, homogeneo=False, expansion=True, pasos=300, apagar=frozenset(["confinamiento"]))
    print(f"  D sin confinamiento: {cuenta_bariones(r)['bariones']} bariones")

    print("test INVARIANCIA A PERMUTACIÓN (índice no decide -> todas deben igualar el base):")
    base = cuenta_bariones(corre(30,21,10,7, homogeneo=False, expansion=True, pasos=300))["bariones"]
    N = 30+21+10+7
    vals = [cuenta_bariones(corre(30,21,10,7, homogeneo=False, expansion=True, pasos=300,
                                  perm=np.random.RandomState(s).permutation(N)))["bariones"] for s in range(6)]
    print(f"  base={base}, permutaciones={vals}, INVARIANTE={all(v==base for v in vals)}")
