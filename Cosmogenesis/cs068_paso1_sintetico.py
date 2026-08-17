"""
CS068 Paso 1 — sustrato SINTÉTICO con verdad de fondo (retícula 2D + atajos INYECTADOS).
==============================================================================
Ruling de CS (ADJUDICACION_CS068_etapa1_tejido_latente_CS.md, 16-jul-2026): antes de preguntar si un
clasificador separa bien tejido/atajo en el blob real de CS067 -- que no tiene retícula base y por eso
cualquier proxy por soporte no tiene sobre qué morder -- primero hay que DES-ARRIESGAR LA MAQUINARIA:
correr el MISMO proceso estirar-enfriar (las mismas funciones de cs068_inflacion_estirar_enfriar.py, SIN
TOCARLAS, importadas tal cual) sobre un sustrato donde tejido-local y atajo son verdad de fondo CONOCIDA
POR CONSTRUCCIÓN, no inferida por soporte de vecinos comunes.

Sustrato: retícula 2D (side x side, 4-vecinos, SIN wraparound -> hay borde real, la distancia geodésica
crece con side, ~2*side ~ 2*sqrt(N)) = tejido local por construcción. Encima, M atajos inyectados: pares de
nodos al azar (típicamente lejos en la retícula dado N grande), conectados directamente -- son el
mundo-pequeño que CS067 hereda, aquí con etiqueta de verdad conocida.

Si el proceso (inflar_dist rompe los atajos MÁS LARGOS primero, midiendo E_nolocal y ganando a
null_corte_azar en el GRADIENTE -- no en el diámetro, que ambos logran) funciona aquí, donde no hay duda de
qué es tejido y qué es atajo, el mecanismo de enfriamiento está validado. Si NO separa ni aquí, el problema
es el proceso de enfriar mismo (no el clasificador de CS067), y hay que arreglarlo ahí antes de tocar el
blob real (Paso 2).

Codea/ejecuta: CC. Diseño/ruling: CS.
"""
from __future__ import annotations
import os, sys, time
import numpy as np

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)
import cs068_inflacion_estirar_enfriar as E
import cs067_habitacion_completa as H

RNG = np.random.default_rng

FRAC_ATAJOS_DEFAULT = 0.15  # atajos inyectados / enlaces locales de la retícula. Fijo, NO se tunea tras ver números.


def _reticula_2d(side):
    """Retícula 2D side x side, 4-vecinos, SIN wraparound (hay borde -> distancia geodésica real, no toroide).
    adj_local = TEJIDO LOCAL por construcción (verdad de fondo, no inferido por soporte)."""
    N = side * side
    adj_local = [set() for _ in range(N)]

    def idx(r, c):
        return r * side + c

    for r in range(side):
        for c in range(side):
            i = idx(r, c)
            if c + 1 < side:
                j = idx(r, c + 1); adj_local[i].add(j); adj_local[j].add(i)
            if r + 1 < side:
                j = idx(r + 1, c); adj_local[i].add(j); adj_local[j].add(i)
    return adj_local, N


def _inyecta_atajos(adj_local, N, n_atajos, rng):
    """n_atajos pares (i,j) al azar, NO ya conectados en la retícula -- verdad de fondo: estos son los
    enlaces no-locales por construcción, no por un proxy de soporte."""
    existentes = set()
    for i in range(N):
        for j in adj_local[i]:
            if i < j:
                existentes.add((i, j))
    atajos = []
    intentos = 0
    tope = max(1, n_atajos) * 50
    while len(atajos) < n_atajos and intentos < tope:
        intentos += 1
        i, j = int(rng.integers(0, N)), int(rng.integers(0, N))
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in existentes:
            continue
        existentes.add((a, b))
        atajos.append((a, b))
    return atajos


def _adj_con_atajos(adj_local, atajos):
    a = [set(x) for x in adj_local]
    for (i, j) in atajos:
        a[i].add(j); a[j].add(i)
    return a


def _centro_geometrico(adj_local, side):
    """CORRECCIÓN metodológica (tras 1er corrida, ver cs068_paso1_run.log): E._centro_y_distancias() elige el
    nodo de MAYOR GRADO -- proxy razonable en el blob real de CS067 (grado ~ hub dinámico), pero en ESTA
    retícula uniforme casi todo nodo interior empata en grado 4, y el desempate de max() cae en un nodo
    arbitrario cerca de una esquina interior, no en el centro geométrico. Eso mata el efecto que el test
    necesita: en un dominio 2D acotado, un nodo cerca del CENTRO tiene distancia geodésica promedio MENOR a
    puntos uniformemente al azar que un nodo cerca de una ESQUINA (la esquina es el peor "punto medio"
    posible en un dominio convexo). Como los atajos se inyectan uniformemente al azar, ese sesgo geométrico
    es la única fuente honesta de gradiente espacial disponible -- y solo aparece si "centro" es el centro
    geométrico real (side//2, side//2), no un empate de grado. No es hornear: es usar la coordenada 2D que
    ya conocemos por construcción, en vez de un proxy de grado diseñado para grafos sin coordenadas."""
    r, c = side // 2, side // 2
    centro = r * side + c
    dist_centro = E._bfs_local(adj_local, centro)
    return centro, dist_centro


def _un_patch_sintetico(N_objetivo, seed, frac_atajos):
    side = int(round(N_objetivo ** 0.5))
    adj_local, N = _reticula_2d(side)
    n_local_edges = sum(len(a) for a in adj_local) // 2
    n_atajos = max(1, int(round(frac_atajos * n_local_edges)))
    atajos = _inyecta_atajos(adj_local, N, n_atajos, RNG(seed + 1))

    diam_ini = H._diam_robusto(_adj_con_atajos(adj_local, atajos), N, RNG(seed + 2))
    ell, cota = E._ell_atajos(adj_local, N, atajos, RNG(seed + 3))
    centro, dist_centro = _centro_geometrico(adj_local, side)

    rD = RNG(seed + 10)
    traj_d, n_rotos, vivos_d = E._correr_inflar_dist(N, atajos, ell, rD)
    rA = RNG(seed + 11)
    traj_a, vivos_a = E._correr_null_corte_azar(N, atajos, rA, n_rotos)
    traj_s, vivos_s = E._correr_sin_enfriar(N, vivos_d)
    rB = RNG(seed + 12)
    E_final_d = traj_d[-1][2]
    E_barajada = E._barajar_energia(E_final_d, rB)

    diam_fin_d = H._diam_robusto(_adj_con_atajos(adj_local, vivos_d), N, RNG(seed + 20))
    diam_fin_a = H._diam_robusto(_adj_con_atajos(adj_local, vivos_a), N, RNG(seed + 21))

    n0 = len(atajos)
    fracciones = [0.75, 0.50, 0.25, 0.05]
    idxs_chk = []
    for frac in fracciones:
        objetivo = frac * n0
        k = next((k for k, (_, nv, _) in enumerate(traj_d) if nv <= objetivo), len(traj_d) - 1)
        idxs_chk.append(k)
    idxs_chk = sorted(set(idxs_chk))
    # Estimador por-cascarón radial (adjudicado por CS): mismo E_nolocal, promediado dentro de cada
    # dist_centro entero -- 4x señal/ruido frente al por-nodo. Es el discriminante desde aquí en adelante.
    corr_d_traj = [(traj_d[k][0], traj_d[k][1], round(E._corr_cascaron(dist_centro, traj_d[k][2], N), 3))
                   for k in idxs_chk]
    corr_a_traj = [(traj_a[k][0], traj_a[k][1], round(E._corr_cascaron(dist_centro, traj_a[k][2], N), 3))
                   for k in idxs_chk]

    return dict(
        N=N, seed=seed, side=side, n_atajos0=n0, n_local_edges=n_local_edges, cota_ell=cota, diam_ini=diam_ini,
        diam_fin_dist=diam_fin_d, diam_fin_azar=diam_fin_a,
        n_atajos_fin_dist=len(vivos_d), n_atajos_fin_azar=len(vivos_a),
        corr_dist_traj=corr_d_traj, corr_azar_traj=corr_a_traj,
        corr_dist_final=round(E._corr_cascaron(dist_centro, E_final_d, N), 3),
        corr_azar_final=round(E._corr_cascaron(dist_centro, traj_a[-1][2], N), 3),
        corr_barajado_final=round(E._corr_cascaron(dist_centro, E_barajada, N), 3),
        corr_sin_enfriar_final=round(E._corr_cascaron(dist_centro, traj_s[-1][2], N), 3),
    )


def main():
    N = int(os.environ.get("CS068_N", 900))
    npatch = int(os.environ.get("CS068_PATCHES", 3))
    frac_atajos = float(os.environ.get("CS068_FRAC_ATAJOS", FRAC_ATAJOS_DEFAULT))
    print("=" * 108, flush=True)
    print("CS068 PASO 1 — SINTÉTICO (verdad de fondo): ¿el PROCESO estirar-enfriar separa tejido/atajo", flush=True)
    print("cuando la clasificación es CONOCIDA por construcción, no inferida por soporte?", flush=True)
    print(f"N~{N} · patches={npatch} · frac_atajos={frac_atajos} · T: {E.T0}->{E.T_FINAL} factor {E.T_FACTOR}",
          flush=True)
    print("=" * 108, flush=True)
    t0 = time.time()
    for p in range(npatch):
        seed = 681000 + 101 * p
        r = _un_patch_sintetico(N, seed, frac_atajos)
        print(f"\n--- patch {p} (seed={seed}) ---", flush=True)
        print(f"  sustrato: N={r['N']} side={r['side']} atajos0={r['n_atajos0']} "
              f"enlaces_locales={r['n_local_edges']} cota_ell={r['cota_ell']} diam_inicial={r['diam_ini']:.2f}",
              flush=True)
        print(f"  diam final: inflar_dist={r['diam_fin_dist']:.2f}  null_corte_azar={r['diam_fin_azar']:.2f}",
              flush=True)
        print(f"  atajos vivos al final: inflar_dist={r['n_atajos_fin_dist']}  "
              f"null_corte_azar={r['n_atajos_fin_azar']}", flush=True)
        print(f"  corr(dist_centro,E_nolocal) TRAYECTORIA inflar_dist: {r['corr_dist_traj']}", flush=True)
        print(f"  corr(dist_centro,E_nolocal) TRAYECTORIA null_corte_azar: {r['corr_azar_traj']}", flush=True)
        print(f"  corr FINAL: inflar_dist={r['corr_dist_final']}  null_corte_azar={r['corr_azar_final']}  "
              f"inflar_barajado={r['corr_barajado_final']}  null_sin_enfriar={r['corr_sin_enfriar_final']}",
              flush=True)
    print(f"\ntiempo total: {(time.time()-t0)/60:.2f} min", flush=True)
    print("\nLECTURA PRE-INSCRITA (Paso 1, verdad de fondo): inflar_dist debe dar corr fuerte NEGATIVA", flush=True)
    print("creciendo en magnitud a lo largo de la trayectoria; null_corte_azar/barajado deben quedar cerca de 0", flush=True)
    print("pese a lograr el MISMO diam/nº de atajos. Si no separa NI AQUÍ -> el proceso de enfriar es el", flush=True)
    print("problema (no el clasificador de CS067); arreglarlo aquí antes de tocar el blob real (Paso 2).", flush=True)


# ============================ AGREGADO con IC95% (blindaje de semillas, estilo CS067) ============================
def _ic95(vals):
    x = np.array(vals, float)
    m = float(x.mean())
    sem = float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
    return m, m - 1.96 * sem, m + 1.96 * sem


def main_agregado():
    """Los patches sueltos (main()) mostraron una tendencia consistente (inflar_dist más negativo que
    null_corte_azar) pero ruidosa patch-a-patch -- no se puede adjudicar 'separa' o 'no separa' mirando
    trayectorias individuales. Blindaje: muchas semillas, media+IC95%, mismo criterio que CS067
    (gamma_sweep_blindaje): SEPARA si IC95%_superior(azar) < IC95%_inferior(dist) (dist más negativo,
    sin solape). Se toma el PRIMER checkpoint de cada trayectoria (más atajos vivos, menos ruido de
    muestra pequeña que los checkpoints tardíos donde quedan 0-5 atajos)."""
    N = int(os.environ.get("CS068_N", 400))
    n_seeds = int(os.environ.get("CS068_SEEDS", 30))
    frac_atajos = float(os.environ.get("CS068_FRAC_ATAJOS", FRAC_ATAJOS_DEFAULT))
    print("=" * 108, flush=True)
    print("CS068 PASO 1 — AGREGADO (blindaje de semillas): media+IC95% del PRIMER checkpoint de la trayectoria",
          flush=True)
    print(f"N~{N} · seeds={n_seeds} · frac_atajos={frac_atajos}", flush=True)
    print("=" * 108, flush=True)
    t0 = time.time()
    corr_d, corr_a = [], []
    for s in range(n_seeds):
        seed = 682000 + 101 * s
        r = _un_patch_sintetico(N, seed, frac_atajos)
        corr_d.append(r["corr_dist_traj"][0][2])
        corr_a.append(r["corr_azar_traj"][0][2])
    md, ld, hd = _ic95(corr_d)
    ma, la, ha = _ic95(corr_a)
    print(f"\ninflar_dist:      media={md:+.4f}  IC95%=[{ld:+.4f},{hd:+.4f}]  (n={n_seeds})", flush=True)
    print(f"null_corte_azar:  media={ma:+.4f}  IC95%=[{la:+.4f},{ha:+.4f}]  (n={n_seeds})", flush=True)
    separa = hd < la or ha < ld
    print(f"\ntiempo total: {(time.time()-t0)/60:.2f} min", flush=True)
    if separa and md < ma:
        print("SEPARA: inflar_dist es significativamente más NEGATIVO que null_corte_azar (IC95% sin solape,", flush=True)
        print("dirección predicha). El mecanismo de enfriamiento SÍ produce un gradiente espacial ordenado", flush=True)
        print("cuando la clasificación tejido/atajo es verdad de fondo. Paso 1 CONFIRMADO -> proceder a Paso 2.",
              flush=True)
    else:
        print("NO SEPARA con blindaje: IC95% se solapan (o la dirección no es la predicha). El mecanismo NO", flush=True)
        print("produce un gradiente ordenado detectable ni con verdad de fondo perfecta. Negativo honesto:", flush=True)
        print("el proceso de enfriar (o el diagnóstico corr(dist_centro,E_nolocal)) es el problema -- no el", flush=True)
        print("clasificador de CS067. Reportar a CS antes de tocar el blob real.", flush=True)


if __name__ == "__main__":
    if os.environ.get("CS068_MODO") == "agregado":
        main_agregado()
    else:
        main()
