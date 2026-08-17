# cs072_modulos/catalogo_plano.py - nuevo archivo

import numpy as np

def catalogo_plano(nq, naq, ne, npos):
    """
    Catálogo SIN DENSIDAD ASIGNADA.
    Solo números cuánticos: color, carga, anti, quark, masa.
    La densidad es 1.0 para todos (plano).
    
    El orden de creación NO importa porque todas las partículas
    son indistinguibles dentro de su especie.
    """
    N = nq + naq + ne + npos
    
    # Números cuánticos
    color = np.zeros(N, dtype=int)
    carga = np.zeros(N, dtype=int)
    es_anti = np.zeros(N, dtype=bool)
    es_quark = np.zeros(N, dtype=bool)
    masa = np.zeros(N, dtype=float)
    
    # Asignar quarks (los primeros nq+naq)
    idx = 0
    for i in range(nq + naq):
        es_quark[i] = True
        color[i] = i % 3  # Esto es inevitable pero no se usa para densidad
        if i < nq:
            # Quarks up (+2) o down (-1) con probabilidad 50/50
            carga[i] = 2 if np.random.random() < 0.5 else -1
            es_anti[i] = False
            masa[i] = 0.002 if carga[i] == 2 else 0.005  # up/down masses
        else:
            # Antiquarks
            carga[i] = -2 if np.random.random() < 0.5 else 1
            es_anti[i] = True
            masa[i] = 0.002 if carga[i] == -2 else 0.005
        idx += 1
    
    # Electrones
    for i in range(nq + naq, nq + naq + ne):
        es_quark[i] = False
        color[i] = -1  # sin color
        carga[i] = -1
        es_anti[i] = False
        masa[i] = 0.0005  # masa del electrón
        idx += 1
    
    # Positrones (si hay)
    for i in range(nq + naq + ne, N):
        es_quark[i] = False
        color[i] = -1
        carga[i] = 1
        es_anti[i] = True
        masa[i] = 0.0005
        idx += 1
    
    # DENSIDAD PLANA: todos 1.0
    densidad = np.ones(N)
    
    # Temperatura inicial: misma para todos
    temp = 3.0 * np.ones(N)
    
    return color, carga, es_anti, es_quark, masa, densidad, temp