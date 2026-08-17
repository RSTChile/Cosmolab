"""
CS072 -- CADENA ESTEQUIOMÉTRICA HASTA EL HIDRÓGENO. Aritmética EXACTA (enteros grandes de Python, sin límite
de tamaño), NO una simulación partícula-por-partícula. Fuente: MANIFIESTO_FOLD_CS072.md ("LA TESIS DE CS072")
+ INSTRUCCION_CS072_cero_azar_hidrogeno_PARA_CC.md. CERO AZAR: esto es división entera y resto, nada más.

QUÉ CALCULA (las 2 primeras afirmaciones de la tesis):
1) S>0 es NECESARIO: con asimetría=0 sobrevive CERO (universo vacío, sólo luz). Con asimetría>0 sobrevive
   EXACTAMENTE esa asimetría (el resto se aniquiló en pares).
2) La cantidad de diferencia (asimetría) determina TODO lo que sigue -- bariones, residuo, hidrógeno -- de
   forma discreta y exacta (no una probabilidad, una división entera). Es una transición abrupta: por debajo
   de asimetría=1 no hay materia; ya en 1 hay.

NO se instancia ninguna partícula individual -- los "quarks", "electrones" aquí son sólo CANTIDADES (enteros).
La cadena: quarks_supervivientes = asimetría_quarks (los antiquarks se cancelan exactos contra esa misma
cantidad de quarks); bariones = quarks_supervivientes // 3 (3 quarks de 3 colores distintos que cancelan a
neutro -- el resto de dividir por 3, a lo sumo 2, es residuo); electrones_supervivientes = asimetría_electrones
(mismo mecanismo con positrones); hidrógeno = min(bariones, electrones_supervivientes) -- lo limita el que
escasee.

Codea/ejecuta: CC. Diseño/ruling: CS + director.
"""
from __future__ import annotations


def cadena_hasta_hidrogeno(asimetria_quarks: int, asimetria_electrones: int) -> dict:
    """Aritmética exacta. asimetria_quarks/electrones = el desbalance FIJO (p.ej. 1 de cada 1e9 -> aquí se
    pasa directamente el EXCEDENTE, que es lo único que sobrevive a la aniquilación por construcción)."""
    assert asimetria_quarks >= 0 and asimetria_electrones >= 0
    quarks_sobrevivientes = asimetria_quarks
    bariones = quarks_sobrevivientes // 3
    residuo_quarks = quarks_sobrevivientes % 3          # 0, 1 o 2 -- quarks sueltos, ni barión ni mesón estable
    electrones_sobrevivientes = asimetria_electrones
    hidrogeno = min(bariones, electrones_sobrevivientes)
    protones_sueltos = bariones - hidrogeno
    electrones_sueltos = electrones_sobrevivientes - hidrogeno
    return dict(
        asimetria_quarks=asimetria_quarks, asimetria_electrones=asimetria_electrones,
        quarks_sobrevivientes=quarks_sobrevivientes, bariones=bariones, residuo_quarks=residuo_quarks,
        electrones_sobrevivientes=electrones_sobrevivientes, hidrogeno=hidrogeno,
        protones_sueltos=protones_sueltos, electrones_sueltos=electrones_sueltos,
    )


def _fmt(n):
    return f"{n:,}".replace(",", ".")


def barrido_transicion_S(potencias_base=(0, 1, 2, 3, 9, 82)):
    """AFIRMACIÓN 1 de la tesis: S=0 -> CERO sobrevive. S>0 -> sobrevive EXACTO ese S. Barre asimetría en
    potencias de 10 (incluida la real ~10^9 del cociente barión/fotón) hasta 10^82 (nº total de bariones
    estimado del universo observable -- techo real de la tesis, no un límite de cómputo: aquí es aritmética
    exacta, no cuesta más simular 10^82 que 10)."""
    print("=" * 100)
    print("AFIRMACIÓN 1 -- S=0 vs S>0: la transición es ABRUPTA (0,0,0 -> aparece), no una pendiente")
    print("=" * 100)
    for k in [None] + list(potencias_base):
        asim = 0 if k is None else 10 ** k
        r = cadena_hasta_hidrogeno(asim, asim)   # misma asimetría en ambos sectores, para aislar el efecto de S
        etiqueta = "S=0 (simetría exacta)" if k is None else f"S=10^{k}"
        print(f"  {etiqueta:28s}: quarks_sobrev={_fmt(r['quarks_sobrevivientes'])}  "
              f"bariones={_fmt(r['bariones'])}  hidrógeno={_fmt(r['hidrogeno'])}  "
              f"universo_vacío={'SÍ' if r['bariones']==0 and r['hidrogeno']==0 else 'no'}")
    print()


def barrido_potencias_y_proporciones():
    """AFIRMACIÓN 2: barre (a) el TAMAÑO del excedente en potencias de 10 -- rangos declarados por el
    director (1-100, 1e3-1e4, 1e6-1e7) y extendido hasta 1e82 (bariones observables reales, techo de la
    tesis) -- y (b) la PROPORCIÓN entre sectores (quarks vs electrones) para ver qué combinaciones dejan
    hidrógeno y cuáles dejan sólo protones sueltos."""
    print("=" * 100)
    print("AFIRMACIÓN 2 -- barrido de POTENCIAS (tamaño del excedente) y PROPORCIONES (quarks vs electrones)")
    print("=" * 100)

    print("\n-- (a) tamaño total, MISMA proporción quarks=electrones (rangos del director + techo 1e82) --")
    rangos_declarados = [1, 10, 100, 1_000, 10_000, 1_000_000, 10_000_000]
    techo_real = [10 ** 9, 10 ** 20, 10 ** 40, 10 ** 60, 10 ** 82]   # ~cociente barión/fotón real .. bariones observables
    for asim in rangos_declarados + techo_real:
        r = cadena_hasta_hidrogeno(asim, asim)
        print(f"  excedente={_fmt(asim):>25s}: bariones={_fmt(r['bariones']):>25s}  "
              f"residuo_quarks={r['residuo_quarks']}  hidrógeno={_fmt(r['hidrogeno']):>25s}  "
              f"protones_sueltos={_fmt(r['protones_sueltos'])}")

    print("\n-- (b) PROPORCIÓN entre especies: mismo tamaño de excedente de quarks, VARIANDO el de electrones --")
    asim_q_fijo = 3_000
    for asim_e in [0, 1, 500, 1_000, 3_000, 10_000]:
        r = cadena_hasta_hidrogeno(asim_q_fijo, asim_e)
        print(f"  quarks_exc={_fmt(asim_q_fijo)} electrones_exc={_fmt(asim_e):>7s}: bariones={r['bariones']}  "
              f"hidrógeno={r['hidrogeno']}  protones_sueltos={r['protones_sueltos']}  "
              f"electrones_sueltos={r['electrones_sueltos']}  "
              f"{'-- lo limita el ELECTRÓN' if asim_e < r['bariones'] else '-- lo limita el PROTÓN (o empatan)'}")

    print("\n-- ejemplos EXACTOS del manifiesto (verificación literal) --")
    for asim in [101, 3_000]:
        r = cadena_hasta_hidrogeno(asim, asim)
        detalle = "100% persiste" if r["residuo_quarks"] == 0 else f"{r['residuo_quarks']} de {asim} quedan sueltos"
        print(f"  {asim} quarks -> {r['bariones']} bariones, {r['residuo_quarks']} sueltos ({detalle})")
    print()


def barrido_asimetria_real():
    """Referencia cósmica real: cociente barión/fotón observado ~6e-10 (≈1 en 1.6e9). Se ilustra con un
    tamaño de fotones/antipartículas de referencia y la asimetría real -- exacto, sin sorteo."""
    print("=" * 100)
    print("REFERENCIA: la asimetría real del universo observable (cociente barión/fotón ~6e-10)")
    print("=" * 100)
    n_foton_like = 10 ** 9  # orden de magnitud de referencia (1e9 antipartículas por cada exceso, per manifiesto)
    asim = 1
    r = cadena_hasta_hidrogeno(asim, asim)
    print(f"  por cada {_fmt(n_foton_like)} antipartículas, {_fmt(n_foton_like + asim)} partículas "
          f"(excedente exacto = {asim})")
    print(f"  sobrevive: {r['quarks_sobrevivientes']} quark(s) -> {r['bariones']} barión(es), "
          f"{r['electrones_sobrevivientes']} electrón(es) -> hidrógeno={r['hidrogeno']}")
    print()


def excedente_por_ratio(n_poblacion: int, ratio_asimetria: float) -> int:
    """Perilla (c) -- INSTRUCCION §6: la asimetría como RATIO (1 en 10, 1 en 100, ..., 1 en 1e9 = el valor
    cósmico real) en vez de un excedente absoluto ya dado. excedente_nominal = floor(N_población · ratio).
    'Por debajo de que la asimetría × cantidad dé ≥1, sobrevive CERO' (INSTRUCCION §6c, literal) -- es
    truncamiento entero, no una probabilidad."""
    return int(n_poblacion * ratio_asimetria)


def fraccion_congelamiento(tasa_expansion: float, tasa_mezcla: float) -> float:
    """Perilla (d) -- INSTRUCCION §6d: compite EXPANSIÓN (congela la asimetría, irreversible) contra MEZCLA
    TÉRMICA (la re-homogeneiza, la borra). El documento describe la FÍSICA (una carrera: rápida congela,
    lenta borra) pero no fija una fórmula exacta -- ésta es la elección de CC, DECLARADA para que CS la
    audite: fracción_congelada = min(1, tasa_expansión/tasa_mezcla). Si la expansión iguala o supera a la
    mezcla, la asimetría congela COMPLETA (fracción=1, banda de existencia); si es más lenta, se erosiona en
    proporción directa a cuánto más lenta es (fracción<1, hacia 0 = universo vacío)."""
    if tasa_mezcla <= 0:
        return 1.0
    return min(1.0, tasa_expansion / tasa_mezcla)


def cadena_completa_4_perillas(n_poblacion_q, ratio_asimetria_q, n_poblacion_e, ratio_asimetria_e,
                                tasa_expansion, tasa_mezcla) -> dict:
    """Las 4 perillas juntas, en UNA sola cadena exacta: (a) tamaño de población, (b) proporción quark/
    electrón (al pasar poblaciones/ratios distintos por sector), (c) qué tan ínfima es la asimetría (ratio),
    (d) velocidad de expansión vs mezcla térmica (si la asimetría sobrevive o se borra). Sin azar."""
    excedente_nominal_q = excedente_por_ratio(n_poblacion_q, ratio_asimetria_q)
    excedente_nominal_e = excedente_por_ratio(n_poblacion_e, ratio_asimetria_e)
    frac_congela = fraccion_congelamiento(tasa_expansion, tasa_mezcla)
    excedente_real_q = int(excedente_nominal_q * frac_congela)
    excedente_real_e = int(excedente_nominal_e * frac_congela)
    r = cadena_hasta_hidrogeno(excedente_real_q, excedente_real_e)
    r.update(n_poblacion_q=n_poblacion_q, ratio_asimetria_q=ratio_asimetria_q,
              excedente_nominal_q=excedente_nominal_q, excedente_nominal_e=excedente_nominal_e,
              tasa_expansion=tasa_expansion, tasa_mezcla=tasa_mezcla, frac_congelamiento=frac_congela)
    return r


def barrido_ratio_asimetria(n_poblacion=10 ** 9):
    """Perilla (c) sola (expansión ya congelada del todo, frac=1, para aislar el efecto de la perilla c):
    barre 1 en 10, 1 en 100, ... 1 en 1e9 (valor cósmico real) sobre una población fija -- muestra el
    UMBRAL exacto donde ratio*población pasa de <1 (cero sobrevive) a >=1 (aparece materia)."""
    print("=" * 100)
    print(f"PERILLA (c) -- magnitud de la asimetría, población fija = {_fmt(n_poblacion)}")
    print("=" * 100)
    for potencia in range(1, 13):
        ratio = 1.0 / (10 ** potencia)
        r = cadena_completa_4_perillas(n_poblacion, ratio, n_poblacion, ratio,
                                        tasa_expansion=1.0, tasa_mezcla=1.0)  # (d) neutra: congela completo
        umbral = "APARECE MATERIA" if r["hidrogeno"] > 0 else ("bariones sin H" if r["bariones"] > 0 else "CERO (vacío)")
        print(f"  1 en 10^{potencia:<3d} (ratio={ratio:.1e}): excedente_q={_fmt(r['quarks_sobrevivientes'])}  "
              f"bariones={_fmt(r['bariones'])}  hidrógeno={_fmt(r['hidrogeno'])}  -> {umbral}")
    print()


def barrido_velocidad_expansion(n_poblacion=10 ** 9, ratio_asimetria=1e-9):
    """Perilla (d) sola (ratio de asimetría FIJO en el valor cósmico real, aislando el efecto de la
    velocidad de expansión): barre tasa_expansión/tasa_mezcla de muy lenta (<<1, se borra) a muy rápida
    (>>1, congela completo) -- muestra la BANDA donde el universo existe."""
    print("=" * 100)
    print(f"PERILLA (d) -- velocidad de expansión vs mezcla térmica, ratio_asimetría=1e-9 (valor cósmico), "
          f"población={_fmt(n_poblacion)}")
    print("=" * 100)
    razones = [0.0, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0, 2.0, 10.0, 100.0]
    for razon in razones:
        # tasa_mezcla fija=1.0, tasa_expansion = razon -- el COCIENTE es lo único que importa en el modelo
        r = cadena_completa_4_perillas(n_poblacion, ratio_asimetria, n_poblacion, ratio_asimetria,
                                        tasa_expansion=razon, tasa_mezcla=1.0)
        estado = "CONGELA (universo existe)" if r["frac_congelamiento"] >= 1.0 else (
                 "erosión parcial" if r["hidrogeno"] > 0 else "SE BORRA (universo vacío)")
        print(f"  expansión/mezcla={razon:>7.3f}: frac_congelada={r['frac_congelamiento']:.3f}  "
              f"excedente_q={_fmt(r['quarks_sobrevivientes'])}  hidrógeno={_fmt(r['hidrogeno'])}  -> {estado}")
    print()


if __name__ == "__main__":
    barrido_transicion_S()
    barrido_potencias_y_proporciones()
    barrido_asimetria_real()
    barrido_ratio_asimetria()
    barrido_velocidad_expansion()
