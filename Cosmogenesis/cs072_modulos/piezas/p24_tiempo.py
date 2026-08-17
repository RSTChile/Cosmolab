"""
p24_tiempo.py — EMERGENCIA DEL TIEMPO (reloj causal).

Qué hace, en simple: el tiempo NO es el contador de pasos del bucle. Antes del primer átomo el universo es
plasma opaco -- las transiciones van y vienen (simétricas), el tiempo neto es CERO. Con el primer átomo neutro
la transición se vuelve IRREVERSIBLE (el fotón escapa, nace el fondo de microondas) y el tiempo empieza a correr.

El tiempo emergente = conteo de transiciones IRREVERSIBLES = nº de átomos neutros consolidados (H + He).
No hay reloj: el universo mide su propio tiempo contando lo que se vuelve irreversible. Emerge JUNTO con el
espacio (la métrica relacional de Bgrav), ambos en el hito de la recombinación.

Observable: tiempo_emergente. Nivel: global. No es una pieza-fuerza: es un LECTOR del estado (se calcula al final).
"""

# ANTI-CONTRABANDO GEOMÉTRICO: NO se cablean valores de NUESTRO universo (radio de Bohr en metros, 21cm en Hz).
# Esos valores llevan dentro π, α y las masas de NUESTRO universo -> imponerlos sería decretar nuestra geometría
# (el mismo error que usar np.pi en una fórmula). π ya se probó CONTINGENTE en el arco: no es universal. Por eso
# la regla y el reloj del universo simulado deben ser ADIMENSIONALES y MEDIDOS del propio sistema, no numéricos.

def tiempo_emergente(obs):
    """TIEMPO: nace con el primer átomo como CONTEO de transiciones irreversibles (átomos neutros). Es puro conteo,
    adimensional -- no lleva unidad de nuestro universo. El "tictac" es 1 evento irreversible, sin segundos cableados.
    ESPACIO: no se reporta aquí como longitud numérica (sería contrabando). La escala espacial debe MEDIRSE de la
    estructura (relación perímetro/radio de la red de ligadura), no fijarse con el radio de Bohr de nuestro cosmos."""
    H = obs.get("hidrogeno", 0); He = obs.get("helio", 0)
    n_irrev = H + He                 # cada átomo neutro = 1 transición irreversible (fotón desacoplado). Conteo puro.
    return dict(tiempo_emergente=n_irrev,       # adimensional: nº de tics irreversibles (no segundos de nuestro universo)
                tiempo_nacio=(n_irrev > 0),
                nota=("tiempo corre: conteo de transiciones irreversibles (adimensional, sin unidad cableada)"
                      if n_irrev > 0 else "plasma opaco: sin átomo -> tiempo no nace (0 tics)"))
