"""
campo_velocidad_heredado.py — BRAZO 2: velocidad heredada del propio sustrato ("la afirmación fuerte"),
INSTRUCCION CS 20-jul.

Regla explícita del director: nada de movimiento puesto a mano; sale de la estructura que ya teníamos.
Tres candidatos evaluados (INSTRUCCION CS 20-jul), elegido el más limpio -- razones en el reporte a CS,
resumen aquí: FLUJO DE COHERENCIA (gradiente de densidad #23 real a lo largo de los enlaces de la malla
causal). Se descartó "dirección de la malla proyectada a 3D" por redundante (usaría la MISMA información
geométrica que ya construyó `pos`, sin aportar nada nuevo); se descartó "flujo de la expansión sobre la
estructura" por requerir interleavar nuestra propia dinámica con la de Phantom (el problema de
contaminación ya resuelto en la corrección de expansión -- ver fase1_traducir_a_phantom.py).

Definición: para cada átomo i, con vecinos N(i) en el grafo YA CONSTRUIDO (adj -- REAL o barajado, el que
corresponda; NO se reconstruye nada aquí):

    v_i = sum_{j in N(i)} (densidad_j - densidad_i) * (pos_j - pos_i) / |pos_j - pos_i|

Fluye hacia el vecino causal de MAYOR coherencia (densidad #23 real, la misma que ya se usó para construir
la malla) -- no usa masa, no usa G, no usa ninguna ley de fuerza (1/r² o similar): NO es una
reimplementación de gravedad, es un flujo de gradiente sobre el grafo, mecanismo genuinamente distinto.

Único parámetro no derivado de la estructura: la ESCALA global (el patrón direccional/relativo es 100%
heredado, sin tocar). Se normaliza al v_rms REALIZADO por el Brazo 1 en la MISMA corrida (mismo N, mismo
seed_null) -- así ambos brazos arrancan con la MISMA energía cinética total, comparación limpia; ninguno
de los dos elige su propia escala libremente.
"""
import numpy as np


def campo_heredado(pos, adj, dens_bar, v_rms_objetivo):
    n = len(pos)
    vel = np.zeros_like(pos)
    for i in range(n):
        vecinos = np.fromiter(adj.get(i, ()), dtype=int)
        if len(vecinos) == 0:
            continue
        d = pos[vecinos] - pos[i]
        dist = np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        peso = (dens_bar[vecinos] - dens_bar[i])[:, None]
        vel[i] = np.sum(peso * d / dist, axis=0)

    v_rms_actual = np.sqrt(np.mean(np.sum(vel ** 2, axis=1)))
    if v_rms_actual > 0:
        vel *= v_rms_objetivo / v_rms_actual
    return vel, v_rms_actual


def factory(v_rms_objetivo_ref):
    """vel_generador(pos, adj, dens_bar) -> vel. v_rms_objetivo_ref viene de fuera (el v_rms REALIZADO
    del Brazo 1 para el mismo N/seed_null) -- ver docstring del módulo."""
    def _gen(pos, adj, dens_bar):
        vel, _ = campo_heredado(pos, adj, dens_bar, v_rms_objetivo_ref)
        return vel
    return _gen
