#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cs090_fase6_o4a_nbody.py — INTEGRADOR GRAVITACIONAL INDEPENDIENTE DE PHANTOM
============================================================================

QUÉ ES ESTO (en simple)
-----------------------
Phantom es un motor de "fluido con gravedad" (SPH): representa el gas con miles de
bolitas que se empujan por presión, se frenan por viscosidad de choque, y cuando una
región se pone demasiado densa la reemplaza por un "sumidero" (una bolita puntual que
se traga a las vecinas).

Este archivo NO es eso. Es un motor de **gravedad pura**: cada partícula es un punto
con masa, se atraen todas con todas (suma directa, Newton), y no hay presión, ni
viscosidad, ni sumideros. Es deliberadamente OTRA FÍSICA. La idea del checkpoint O4-A
es: si dos motores tan distintos ordenan igual a los mismos pares de condiciones
iniciales, el orden no puede ser un capricho de Phantom.

Analogía: Phantom es una olla de sopa hirviendo simulada al detalle (burbujas,
remolinos, grumos que se fusionan). Esto de acá es la misma olla pero pensada como
canicas que sólo se atraen entre sí. Si en ambas versiones "la olla A hace más grumos
que la olla B", el ordenamiento es de la receta, no del simulador.

CONTENIDO DEL MÓDULO
--------------------
1. `leer_ic_cosmogenesis`  — lee el MISMO archivo de condiciones iniciales que se le
                             dio a Phantom (`cosmogenesis_ic.txt`), sin regenerarlo.
2. `aceleracion_directa`   — gravedad por suma directa O(N²) con suavizado de Plummer,
                             vectorizada con el truco de matriz de Gram + BLAS.
3. `energia_total`         — cinética + potencial CONSISTENTE con el suavizado usado
                             (mismo par fuerza/potencial), para medir deriva de energía.
4. `integrar_leapfrog`     — velocity-Verlet / leapfrog "patada-deriva-patada" (KDK),
                             paso fijo, simpléctico.
5. `observable_fof`        — análogo del observable de Phantom: fracción de masa en
                             las regiones más densas al final (friends-of-friends).
6. `observable_densidad_knn` — variante: fracción de masa por encima de un umbral de
                             densidad local estimada con el k-ésimo vecino.
7. `validar()`             — batería de validación (dos cuerpos con solución conocida,
                             conservación de energía, convergencia en paso de tiempo).

UNIDADES
--------
Las mismas de Phantom en estas corridas: G = 1, masa en g, longitud en cm
(umass = udist = 1 según el encabezado de los volcados). Masa total 18800, 2000
partículas de 9.4 cada una, caja de lado ~97.6 SIN condiciones periódicas
(los volcados no traen xmin/xmax, o sea frontera aislada) — por eso acá también
la gravedad es aislada, sin sumas de Ewald.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

G_GRAV = 1.0  # unidades de código de Phantom en estas corridas


# ---------------------------------------------------------------------------
# 1. LECTURA DE LAS CONDICIONES INICIALES (las MISMAS que recibió Phantom)
# ---------------------------------------------------------------------------
def leer_ic_cosmogenesis(ruta):
    """Lee `cosmogenesis_ic.txt`.

    Formato del archivo (el que consume `setup_cosmogenesis` de Phantom):
      línea 1 : comentario '#' con metadatos
      línea 2 : npart  masa_particula  hfact  polyk
      línea 3+: x y z vx vy vz h   (una por partícula)

    Devuelve (pos[N,3], vel[N,3], masa_particula, cabecera_dict).
    No se modifica ni se re-genera nada: es exactamente el mismo input de Phantom.
    """
    with open(ruta, "r") as f:
        comentario = f.readline()
        cab = f.readline().split()
    npart = int(float(cab[0]))
    m_part = float(cab[1])
    hfact = float(cab[2])
    polyk = float(cab[3])
    datos = np.loadtxt(ruta, skiprows=2)
    assert datos.shape[0] == npart, f"npart declarado {npart} != filas {datos.shape[0]}"
    pos = np.ascontiguousarray(datos[:, 0:3], dtype=np.float64)
    vel = np.ascontiguousarray(datos[:, 3:6], dtype=np.float64)
    return pos, vel, m_part, dict(npart=npart, m_part=m_part, hfact=hfact,
                                  polyk=polyk, comentario=comentario.strip())


# ---------------------------------------------------------------------------
# 2. GRAVEDAD POR SUMA DIRECTA (O(N²)) CON SUAVIZADO DE PLUMMER
# ---------------------------------------------------------------------------
def aceleracion_directa(pos, masas, eps):
    """Aceleración gravitatoria de todos sobre todos, suavizada tipo Plummer.

        a_i = G * sum_{j != i}  m_j * (x_j - x_i) / (|x_j - x_i|² + eps²)^{3/2}

    El suavizado `eps` evita que dos partículas que se acercan mucho generen fuerzas
    infinitas (y pasos de tiempo infinitesimales). Es el análogo "honesto" de lo que en
    Phantom hace la combinación gravedad-suavizada-por-h + creación de sumideros:
    poner un tamaño mínimo por debajo del cual la física deja de resolverse.

    TRUCO DE IMPLEMENTACIÓN (por qué es rápido pese a ser O(N²)):
      sum_j w_ij (x_j - x_i) = (W @ X)_i - (sum_j w_ij) * x_i
    o sea, una vez que tengo la matriz de pesos W (N×N), las tres componentes salen
    de UN producto matriz-matriz (W @ X), que BLAS hace a toda velocidad.
    """
    n = pos.shape[0]
    # distancias al cuadrado por identidad de Gram: |xi-xj|² = |xi|²+|xj|²-2 xi·xj
    cuad = np.einsum("ij,ij->i", pos, pos)
    r2 = cuad[:, None] + cuad[None, :] - 2.0 * (pos @ pos.T)
    np.maximum(r2, 0.0, out=r2)          # limpia negativos por redondeo
    r2 += eps * eps
    inv = 1.0 / np.sqrt(r2)              # 1/sqrt(r²+eps²)
    w = inv * inv * inv                  # (r²+eps²)^{-3/2}
    np.fill_diagonal(w, 0.0)             # sin auto-interacción
    if np.isscalar(masas) or np.ndim(masas) == 0:
        acc = G_GRAV * float(masas) * (w @ pos - w.sum(axis=1)[:, None] * pos)
    else:
        wm = w * masas[None, :]
        acc = G_GRAV * (wm @ pos - wm.sum(axis=1)[:, None] * pos)
    return acc


def energia_total(pos, vel, masas, eps):
    """Energía cinética + potencial, con el potencial CONSISTENTE con la fuerza.

    Para el suavizado de Plummer usado arriba, el par fuerza/potencial exacto es
        U_ij = - G m_i m_j / sqrt(r² + eps²)
    Usar el potencial consistente es lo que permite que la conservación de energía
    sea una prueba real del integrador (y no una prueba de que mezclamos fórmulas).
    """
    n = pos.shape[0]
    m = np.full(n, float(masas)) if np.ndim(masas) == 0 else np.asarray(masas, float)
    ecin = 0.5 * np.sum(m * np.einsum("ij,ij->i", vel, vel))
    cuad = np.einsum("ij,ij->i", pos, pos)
    r2 = cuad[:, None] + cuad[None, :] - 2.0 * (pos @ pos.T)
    np.maximum(r2, 0.0, out=r2)
    r2 += eps * eps
    inv = 1.0 / np.sqrt(r2)
    np.fill_diagonal(inv, 0.0)
    epot = -0.5 * G_GRAV * float(np.einsum("i,ij,j->", m, inv, m))
    return ecin, epot, ecin + epot


# ---------------------------------------------------------------------------
# 3. INTEGRADOR LEAPFROG (velocity-Verlet, "patada-deriva-patada")
# ---------------------------------------------------------------------------
def integrar_leapfrog(pos, vel, masas, eps, t_final, dt, cada=0, verbose=False):
    """Integra hasta `t_final` con paso fijo `dt`, esquema KDK.

        v(t+dt/2) = v(t)      + (dt/2) a(x(t))
        x(t+dt)   = x(t)      + dt v(t+dt/2)
        v(t+dt)   = v(t+dt/2) + (dt/2) a(x(t+dt))

    Paso FIJO a propósito: el leapfrog de paso fijo es simpléctico, y entonces la
    energía no deriva sistemáticamente — sólo oscila. Eso convierte a "¿se conservó la
    energía?" en un termómetro honesto de si el paso alcanza.

    Devuelve (pos_final, vel_final, diagnostico) con la historia de energía.
    """
    pos = pos.copy()
    vel = vel.copy()
    nsteps = int(round(t_final / dt))
    acc = aceleracion_directa(pos, masas, eps)
    e0 = energia_total(pos, vel, masas, eps)
    hist = [(0.0,) + e0]
    for k in range(nsteps):
        vel += 0.5 * dt * acc
        pos += dt * vel
        acc = aceleracion_directa(pos, masas, eps)
        vel += 0.5 * dt * acc
        if cada and ((k + 1) % cada == 0 or k == nsteps - 1):
            t = (k + 1) * dt
            hist.append((t,) + energia_total(pos, vel, masas, eps))
            if verbose:
                print(f"   t={t:.4f}  Etot={hist[-1][3]:.6e}  "
                      f"dE/|E0|={(hist[-1][3]-e0[2])/abs(e0[2]):+.3e}", flush=True)
    ef = energia_total(pos, vel, masas, eps)
    diag = dict(nsteps=nsteps, dt=dt, eps=eps,
                E0=e0[2], Ef=ef[2],
                deriva_rel=(ef[2] - e0[2]) / abs(e0[2]),
                Ekin0=e0[0], Ekinf=ef[0], Epot0=e0[1], Epotf=ef[1],
                historia=hist)
    return pos, vel, diag


# ---------------------------------------------------------------------------
# 4. OBSERVABLE ANÁLOGO: "fracción de masa en las regiones más densas al final"
# ---------------------------------------------------------------------------
def observable_fof(pos, m_part, ell, n_min):
    """Friends-of-friends: agrupa partículas separadas menos de `ell`, y devuelve la
    fracción de masa que quedó en grupos de al menos `n_min` miembros.

    Analogía: si cada partícula "se da la mano" con toda vecina a menos de `ell`,
    los grupos son las cadenas de manos. La fracción de masa en grupos grandes es
    "cuánta materia terminó apelotonada".

    POR QUÉ ES EL ANÁLOGO JUSTO (y por qué NO es la misma cantidad que Phantom):
    Phantom mide masa en sumideros; los sumideros nacen cuando una región supera un
    umbral de densidad y luego ACRETAN vecinas dentro de h_acc. Acá no hay sumideros:
    lo más parecido es medir cuánta masa terminó dentro de aglomeraciones compactas.
    Los números absolutos NO son comparables — la comparación válida es de ORDEN.
    """
    n = pos.shape[0]
    arbol = cKDTree(pos)
    pares = arbol.query_pairs(r=ell, output_type="ndarray")
    if len(pares) == 0:
        return 0.0, np.zeros(n, dtype=int), 0
    g = coo_matrix((np.ones(len(pares)), (pares[:, 0], pares[:, 1])), shape=(n, n))
    ncomp, etiq = connected_components(g, directed=False)
    tam = np.bincount(etiq, minlength=ncomp)
    grandes = np.where(tam >= n_min)[0]
    masa_en_grupos = tam[grandes].sum() * m_part
    frac = masa_en_grupos / (n * m_part)
    return float(frac), etiq, int(len(grandes))


def observable_densidad_knn(pos, m_part, k, rho_umbral):
    """Variante del observable: densidad local por k-ésimo vecino,
        rho_i = k * m / ( (4/3) π r_k³ )
    y fracción de masa con rho_i > rho_umbral. Es otra manera de preguntar
    "¿cuánta masa quedó en lo denso?" sin depender de una longitud de enlace.
    """
    arbol = cKDTree(pos)
    dist, _ = arbol.query(pos, k=k + 1)   # el primero es la propia partícula
    rk = dist[:, k]
    rho = k * m_part / ((4.0 / 3.0) * np.pi * np.maximum(rk, 1e-12) ** 3)
    return float((rho > rho_umbral).mean()), rho


# ---------------------------------------------------------------------------
# 5. VALIDACIÓN DEL INTEGRADOR  (sin esto los resultados no valen nada)
# ---------------------------------------------------------------------------
def _val_dos_cuerpos():
    """PRUEBA 1 — dos cuerpos con solución analítica conocida.

    Dos masas iguales en órbita CIRCULAR alrededor de su centro de masa.
    Con m1=m2=m, separación d, la velocidad orbital de cada una es
        v = sqrt(G m / (2 d))
    y el período es  T = 2π sqrt(d³ / (G * 2m)).
    Se integra un período completo con eps despreciable y se mide cuánto se aleja
    la posición final de la inicial (la órbita debe cerrarse).
    """
    m = 1.0
    d = 1.0
    v = np.sqrt(G_GRAV * m / (2.0 * d))
    T = 2.0 * np.pi * np.sqrt(d ** 3 / (G_GRAV * 2.0 * m))
    pos = np.array([[-d / 2, 0.0, 0.0], [d / 2, 0.0, 0.0]])
    vel = np.array([[0.0, -v, 0.0], [0.0, v, 0.0]])
    masas = np.array([m, m])
    dt = T / 20000.0
    eps = 1e-6
    pf, vf, diag = integrar_leapfrog(pos, vel, masas, eps, T, dt)
    err_pos = np.max(np.abs(pf - pos))
    err_vel = np.max(np.abs(vf - vel))
    return dict(prueba="dos_cuerpos_circular", periodo=T, dt=dt,
                error_max_posicion=err_pos, error_max_velocidad=err_vel,
                deriva_energia_rel=diag["deriva_rel"])


def _val_dos_cuerpos_excentrica():
    """PRUEBA 2 — órbita ELÍPTICA (e=0.5): período analítico por tercera ley de Kepler.

    Se lanza el par en el apoastro de una elipse de semieje a y excentricidad e, y se
    integra un período T = 2π sqrt(a³/(G M_tot)). La órbita debe cerrarse igual.
    Es una prueba más dura que la circular porque la velocidad varía mucho.
    """
    m = 1.0
    Mtot = 2.0 * m
    a = 1.0
    e = 0.5
    r_apo = a * (1 + e)
    v_apo = np.sqrt(G_GRAV * Mtot * (2.0 / r_apo - 1.0 / a))   # vis-viva (relativa)
    T = 2.0 * np.pi * np.sqrt(a ** 3 / (G_GRAV * Mtot))
    pos = np.array([[-r_apo / 2, 0.0, 0.0], [r_apo / 2, 0.0, 0.0]])
    vel = np.array([[0.0, -v_apo / 2, 0.0], [0.0, v_apo / 2, 0.0]])
    masas = np.array([m, m])
    dt = T / 200000.0
    pf, vf, diag = integrar_leapfrog(pos, vel, masas, 1e-8, T, dt)
    return dict(prueba="dos_cuerpos_e0.5", periodo=T, dt=dt,
                error_max_posicion=float(np.max(np.abs(pf - pos))),
                error_max_velocidad=float(np.max(np.abs(vf - vel))),
                deriva_energia_rel=diag["deriva_rel"])


def _val_plummer_equilibrio(n=500, semilla=7):
    """PRUEBA 3 — esfera de Plummer en equilibrio: NO debe colapsar ni explotar.

    Se construye una esfera de Plummer (perfil con solución analítica) con velocidades
    del teorema del virial (2T + U = 0) y se integra un tiempo de cruce. El radio que
    contiene la mitad de la masa debe quedarse casi igual, y el cociente virial cerca
    de -0.5. Es la prueba de que el motor no inventa ni disipa estructura.
    """
    rng = np.random.default_rng(semilla)
    a = 1.0
    Mtot = 1.0
    m = Mtot / n
    # muestreo del perfil de Plummer por inversión de la masa acumulada
    u = rng.random(n)
    r = a / np.sqrt(u ** (-2.0 / 3.0) - 1.0)
    ct = 2 * rng.random(n) - 1
    st = np.sqrt(1 - ct ** 2)
    ph = 2 * np.pi * rng.random(n)
    pos = np.stack([r * st * np.cos(ph), r * st * np.sin(ph), r * ct], axis=1)
    pos -= pos.mean(0)
    eps = 0.05
    # velocidades isotrópicas escaladas para cumplir el virial exactamente
    v = rng.normal(size=(n, 3))
    v -= v.mean(0)
    _, epot, _ = energia_total(pos, v * 0.0, m, eps)
    ecin_obj = -0.5 * epot
    ecin_act = 0.5 * m * np.sum(v ** 2)
    v *= np.sqrt(ecin_obj / ecin_act)
    r_half_0 = np.median(np.linalg.norm(pos - pos.mean(0), axis=1))
    t_cruce = 1.0 / np.sqrt(G_GRAV * Mtot / a ** 3)
    pf, vf, diag = integrar_leapfrog(pos, v, m, eps, t_cruce, t_cruce / 4000.0)
    r_half_f = np.median(np.linalg.norm(pf - pf.mean(0), axis=1))
    ecin, epot, _ = energia_total(pf, vf, m, eps)
    return dict(prueba="plummer_virial", n=n, t_cruce=t_cruce,
                r_mitad_inicial=r_half_0, r_mitad_final=r_half_f,
                cambio_relativo_r_mitad=(r_half_f - r_half_0) / r_half_0,
                cociente_virial_final=ecin / abs(epot),  # esperado ~0.5
                deriva_energia_rel=diag["deriva_rel"])


def _val_momento(pos, vel, masas, eps, t_final, dt):
    """PRUEBA 4 — momento lineal y centro de masa sobre las IC REALES del proyecto.

    En gravedad aislada el momento lineal total se conserva EXACTAMENTE (las fuerzas
    internas son pares acción-reacción). Si mi suma directa tuviera un error de signo o
    de índice, esto se rompería enseguida. También devuelve la deriva de energía sobre
    las mismas condiciones iniciales que se van a usar de verdad.
    """
    m = float(masas)
    p0 = m * vel.sum(0)
    pf, vf, diag = integrar_leapfrog(pos, vel, masas, eps, t_final, dt)
    pfin = m * vf.sum(0)
    return dict(prueba="momento_lineal_IC_real",
                momento_inicial=p0, momento_final=pfin,
                error_rel_momento=float(np.max(np.abs(pfin - p0)) / np.max(np.abs(p0))),
                deriva_energia_rel=diag["deriva_rel"]), pf, vf, diag


if __name__ == "__main__":
    import json
    print("=== VALIDACIÓN DEL INTEGRADOR (cs090_fase6_o4a_nbody.py) ===")
    for f in (_val_dos_cuerpos, _val_dos_cuerpos_excentrica, _val_plummer_equilibrio):
        r = f()
        print(json.dumps({k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                          for k, v in r.items()}, indent=2, ensure_ascii=False))
