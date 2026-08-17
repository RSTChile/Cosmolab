"""
CS080 — Experimento 1 del roadmap "Fase 3": FLUJO DE GEOMETRÍA BAJO RENORMALIZACIÓN
=====================================================================================
Pregunta (línea NUEVA, distinta de la de CS073/Phantom): el arco CS064-CS068 encontró que el
sustrato de grafo puro (sin gravedad, sin Phantom) tiene un tejido con "lejos" real a nivel LOCAL
(CS066: con localidad fuerte en la formación, k_local≈5-6, el tejido queda conexo con dimensión
espectral d_s≈3 y clustering alto) pero GLOBALMENTE sigue siendo mundo-pequeño: el diámetro casi
no crece con N (cs066conf_exponentes.md: pendiente log-log del diámetro ~0.13-0.17 con k_local=6,
muy por debajo de la retícula 2D=0.52; cs068 confirmó lo mismo por otra vía, pendiente≈0.22 con
rango que cruza el umbral 0.3).

La pregunta de HOY: ¿ese resultado "mundo local, blob global" es un artefacto de mirar el tejido
a la escala de un solo nodo? Si agrupamos nodos en "supernodos" (coarse-graining, como el grupo de
renormalización en física de la materia condensada) y medimos las mismas cantidades (dimensión,
diámetro) a escalas cada vez más grandes b=2,4,8,16, ¿aparece a alguna escala una geometría con
"lejos" real que no se veía mirando nodo por nodo? Dos resultados posibles, ninguno es "malo":
  (A) a escala grande el diámetro empieza a crecer como N_b^(1/d) con d finito y estable → había
      condensación geométrica macroscópica oculta bajo el ruido microscópico.
  (B) el diámetro sigue ~log(N_b) en TODAS las escalas de agrupamiento → el sustrato es mundo-
      pequeño "hasta el fondo": no importa cuánto se agrupe, nunca aparece una geometría con
      distancia real.

SUSTRATO: se usa el propio motor de CS066 (cs066_localidad_geometrogenesis.py, import directo, SIN
tocarlo) en su punto más favorable a la localidad según el propio archivo de síntesis del proyecto
(cs066conf_exponentes.md): brazo "local" con k_local FIJO=6 (d_s local ≈3.5-3.9, giant≈0.91,
pendiente-diámetro monótona R²=0.93 -- el mejor candidato disponible a "tejido con localidad real").
Control NULL pre-existente en el propio CS066: brazo "local_barajado" -- MISMO tope de grado
k_local, pero eligiendo qué enlaces persisten AL AZAR en vez de por soporte local (mismo
procedimiento físico completo, sólo cambia el criterio de qué sobrevive). Si el flujo de
renormalización distingue real de barajado, la localidad importa más allá de escala 1.

MÉTODO DE COARSE-GRAINING (nuevo en este archivo, simple y declarado, ninguna calibración oculta):
"cajas" por BFS greedy (variante práctica del box-covcovering de Song-Havlin-Makse): se recorren los
nodos en orden aleatorio: cada nodo sin asignar dispara una caja que crece por BFS hasta juntar
~b nodos (o hasta agotar la componente); todo lo que cae en esa caja se colapsa a UN supernodo. Dos
supernodos quedan unidos en el grafo grueso si existe AL MENOS UN enlace real entre sus miembros. No
hay solapamiento (partición, no cobertura con traslape) -- por simplicidad y para poder usar tal
cual las mismas funciones de juicio que ya existen en el proyecto (dim_volumen, _diam, _giant,
_frame_burgers) sin reinventarlas ni tocarlas.

MÉTRICAS REUSADAS TAL CUAL (ningún juez nuevo, sólo aplicados al grafo grueso en cada escala b):
  - dim_volumen(adj,N,rng)   -- cs064_smoke.py: dimensión por crecimiento de bola |B(v,r)|~r^d.
                                Es la D_H (tipo Hausdorff) que pide el roadmap -- ya estaba hecha.
  - _diam(adj,N)             -- cs055_proceso_acoplado.py: diámetro (doble BFS).
  - _giant(adj,N)            -- cs055_proceso_acoplado.py: tamaño de la componente gigante.
  - _frame_burgers(...)      -- cs059_espin_como_marco.py: holonomía del marco de espín sobre ciclos
                                fundamentales. Se propaga el campo de espín V a cada supernodo
                                (promedio normalizado de sus miembros) y se mide holonomía del grafo
                                grueso con esos espines heredados. NOTA HONESTA: esta holonomía mide
                                consistencia del marco NEMÁTICO (pregunta de "direcciones", Nivel 2
                                del arco CS064-066), no curvatura geométrica del grafo en sí -- se
                                reporta con esa salvedad, no se re-etiqueta como algo que no es.
  π_G: NO SE ENCONTRÓ una función de este nombre ya implementada en cs064-068 (se buscó explícitamente
       antes de escribir código nuevo, `grep -r "pi_G"` sin resultados). No se inventa un juez nuevo
       bajo ese nombre para no fabricar un número sin linaje -- se documenta como AUSENTE, no se
       fuerza. dim_volumen ya cumple el rol de "razón geométrica" que pedía el roadmap.

Control NULL adicional (además de local_barajado): grafo de Erdős-Rényi con MISMO N y MISMO nº de
enlaces que el tejido real a b=1 (vía cg003_diagnostico_gromov.aleatorio, ya usado como generador de
sopa caliente en CS064/066 -- reuso directo), sometido al MISMO coarse-graining. Sirve de piso
absoluto: un grafo sin ninguna estructura de vecindad.

Codea/ejecuta: CC (Claude). Diseño: roadmap consolidado de 5 analistas de IA (5-ago-2026), adaptado
por CC a las funciones ya existentes del proyecto. No se toca ningún script cs064-068 -- sólo import.
No se declara cierre ni veredicto de arco: se reportan números, la lectura es de Alexis.
"""
from __future__ import annotations
import os, sys, time, csv, math, random
import numpy as np
from collections import deque

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)

import cs057_paisaje_completo as C7          # _diam, _giant (reexportados desde cs055 vía cs057)
import cs059_espin_como_marco as C9          # _spins, _frame_burgers (holonomía)
import cg003_diagnostico_gromov as GR        # aleatorio() -- generador de grafo caliente / NULL ER
import cs064_smoke as SM                     # dim_volumen (D_H por crecimiento de bola), adj_sparse
import cs064_sistema_completo as C64         # _cataloga (catálogo de partículas), DMAX_INT
import cs066_localidad_geometrogenesis as C66  # proceso066, gate_localidad -- EL SUSTRATO, sin tocar

RNG = np.random.default_rng
DMAX_INT = C64.DMAX_INT                       # 8, dimensión del campo de espín (igual que CS064/066)

N_NODOS   = int(os.environ.get("CS080_N", 3000))
K_LOCAL   = int(os.environ.get("CS080_KLOC", 6))     # el punto más favorable según cs066conf_exponentes.md
ESCALAS_B = [int(x) for x in os.environ.get("CS080_B", "2,4,8,16").split(",")]
SEEDS     = [int(x) for x in os.environ.get("CS080_SEEDS", "80100,80200,80300").split(",")]
OUT       = os.environ.get("CS080_OUT", os.path.join(_HERE, "cs080_renormalizacion.csv"))


# ============================ COARSE-GRAINING: cajas por BFS greedy (~b nodos c/u) ============================
def cajas_bfs(adj, N, b, rng):
    """Partición del grafo en supernodos de tamaño objetivo ~b, por BFS greedy (box-covering práctico,
    variante de Song-Havlin-Makse). Devuelve: asignación nodo->caja (array de N ints), nº de cajas."""
    orden = list(range(N))
    rng.shuffle(orden)
    asign = -np.ones(N, dtype=np.int64)
    caja_id = 0
    for s in orden:
        if asign[s] != -1:
            continue
        # BFS acotado a b nodos desde s, solo por nodos aún sin asignar
        q = deque([s]); asign[s] = caja_id; tomados = 1
        while q and tomados < b:
            u = q.popleft()
            for v in adj[u]:
                if asign[v] == -1:
                    asign[v] = caja_id; tomados += 1; q.append(v)
                    if tomados >= b:
                        break
        caja_id += 1
    return asign, caja_id


def grafo_grueso(adj, N, asign, n_cajas):
    """Grafo de supernodos: caja u -- caja v si existe >=1 enlace real entre sus miembros."""
    adj_g = [set() for _ in range(n_cajas)]
    for i in range(N):
        ci = asign[i]
        for j in adj[i]:
            if j <= i:
                continue
            cj = asign[j]
            if ci != cj:
                adj_g[ci].add(cj); adj_g[cj].add(ci)
    return adj_g


def propagar_spins(V, asign, n_cajas, K):
    """Espín de cada supernodo = promedio normalizado de los espines de sus miembros (heredado, no
    recalculado desde cero) -- necesario para poder medir holonomía del marco en el grafo grueso con
    la misma función _frame_burgers, sin tocarla."""
    Vg = np.zeros((n_cajas, K))
    cnt = np.zeros(n_cajas)
    for i in range(len(asign)):
        Vg[asign[i]] += V[i]
        cnt[asign[i]] += 1
    Vg /= np.maximum(cnt, 1)[:, None]
    norm = np.linalg.norm(Vg, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1.0
    return Vg / norm


# ============================ MÉTRICAS a una escala b dada ============================
def metricas_escala(adj, N, V, rng):
    if N < 4:
        return dict(N_b=N, diam=float("nan"), giant=float("nan"), d_s=float("nan"),
                    holonomia=float("nan"), n_ciclos=0)
    diam = float(C7._diam(adj, N))
    giant = float(C7._giant(adj, N))          # _giant YA devuelve fracción (best/N), no dividir de nuevo
    ds = SM.dim_volumen(adj, N, rng=rng)
    K = min(4, V.shape[1])
    spins = V[:, :K] / (np.linalg.norm(V[:, :K], axis=1, keepdims=True) + 1e-12)
    try:
        hol, ncic, _ = C9._frame_burgers(adj, N, spins, K, rng, null=False)
    except Exception:
        hol, ncic = float("nan"), 0
    return dict(N_b=N, diam=diam, giant=round(giant, 4), d_s=round(ds, 3) if not math.isnan(ds) else ds,
                holonomia=round(hol, 4) if not math.isnan(hol) else hol, n_ciclos=int(ncic))


# ============================ CONSTRUCCIÓN DEL SUSTRATO (b=1) ============================
def construir_sustrato(N, seed, arm):
    """arm in {'local','local_barajado','er_null'}. 'local'/'local_barajado' = motor CS066 tal cual
    (proceso066), sin modificar el archivo original. 'er_null' = piso absoluto: Erdős-Rényi con
    mismo N y mismo grado medio que el sustrato real, sin ninguna física ni localidad."""
    rng = RNG(seed)
    if arm == "er_null":
        adj0, _ = GR.aleatorio(N, meandeg=6.0, seed=seed)
        adj = [set(a) for a in adj0]
        V = C9._spins(N, DMAX_INT, rng)
        return adj, V
    cat = C64._cataloga(N, rng)
    r2 = RNG(seed * 137 + hash(arm) % 9973 + 5)
    adj, V, D, G = C66.proceso066(N, cat, arm, K_LOCAL, r2)
    return adj, V


# ============================ CORRIDA COMPLETA: 1 semilla × 3 brazos × todas las escalas ============================
def corre_semilla(seed):
    filas = []
    for arm in ("local", "local_barajado", "er_null"):
        t0 = time.time()
        adj, V = construir_sustrato(N_NODOS, seed, arm)
        N = N_NODOS
        rng_m = RNG(seed * 991 + hash(arm) % 7919)
        m1 = metricas_escala(adj, N, V, rng_m)
        m1.update(dict(seed=seed, arm=arm, b=1, n_cajas=N))
        filas.append(m1)
        print(f"  [{arm}] b=1  N_b={m1['N_b']}  diam={m1['diam']:.2f}  d_s={m1['d_s']}  "
              f"giant={m1['giant']:.3f}  hol={m1['holonomia']}  ({time.time()-t0:.1f}s)", flush=True)
        for b in ESCALAS_B:
            tb = time.time()
            rng_b = RNG(seed * 733 + b * 31 + hash(arm) % 4999)
            asign, n_cajas = cajas_bfs(adj, N, b, rng_b)
            adj_g = grafo_grueso(adj, N, asign, n_cajas)
            Vg = propagar_spins(V, asign, n_cajas, V.shape[1])
            rng_m2 = RNG(seed * 1291 + b * 53 + hash(arm) % 3571)
            mb = metricas_escala(adj_g, n_cajas, Vg, rng_m2)
            mb.update(dict(seed=seed, arm=arm, b=b, n_cajas=n_cajas))
            filas.append(mb)
            print(f"  [{arm}] b={b:<3} N_b={mb['N_b']:<5} diam={mb['diam']:.2f}  d_s={mb['d_s']}  "
                  f"giant={mb['giant']:.3f}  hol={mb['holonomia']}  ({time.time()-tb:.1f}s)", flush=True)
    return filas


def _campos():
    return ["seed", "arm", "b", "n_cajas", "N_b", "diam", "giant", "d_s", "holonomia", "n_ciclos"]


def main():
    print("=" * 100, flush=True)
    print("CS080 — Experimento 1: FLUJO DE GEOMETRÍA BAJO RENORMALIZACIÓN (coarse-graining del tejido CS066)",
          flush=True)
    print(f"N={N_NODOS}  k_local={K_LOCAL} (fijo, punto más favorable según cs066conf_exponentes.md)  "
          f"escalas b={ESCALAS_B}  semillas={SEEDS}", flush=True)
    print("brazos: local (real) | local_barajado (NULL: mismo tope de grado, sin criterio de localidad) | "
          "er_null (piso: Erdős-Rényi puro)", flush=True)
    print("=" * 100, flush=True)
    t0 = time.time()
    fout = open(OUT, "w", newline="")
    wr = csv.DictWriter(fout, fieldnames=_campos())
    wr.writeheader()
    for seed in SEEDS:
        print(f"\n--- semilla {seed} ---", flush=True)
        filas = corre_semilla(seed)
        for f in filas:
            wr.writerow({k: f[k] for k in _campos()})
        fout.flush()
        print(f"  (acumulado {(time.time()-t0)/60:.1f} min)", flush=True)
    fout.close()
    print(f"\nCOMPLETO en {(time.time()-t0)/60:.1f} min -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
