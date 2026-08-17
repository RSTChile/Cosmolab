"""
null3_dosis_respuesta.py — Parte B del encargo "robustecer NULL-3": curva DOSIS-RESPUESTA relajando
`tol_relativa` del filtro geométrico de longitud del double-edge-swap (Parte A midió motivos/triángulos
sólo en el punto tol_relativa=0.2 ya cerrado -- esto barre varios niveles de perturbación para ver si el
parecido REAL≈NULL-3 se degrada gradualmente o se rompe de golpe).

Niveles medidos (mismo grafo REAL, mismo seed=501 en todos, para que la única variable sea tol_relativa):
  0.2  -- el original de NULL3_resultado_CS.md (346/49450 aceptados, 0.70%)
  0.4  -- el doble de permisivo
  0.8  -- 4x más permisivo
  sin filtro -- double-edge-swap de Maslov-Sneppen SIN restricción de longitud (`barajar_aristas_
              sin_restriccion`, el mecanismo literal de los NULL1-8 originales de CS073/CS072)

Para cada nivel mide, con el MISMO método que ya usó la jerarquía (nada reinventado, todo importado de
piezas congeladas o de los scripts de investigación ya validados):
  (a) tasa de aceptación del swap
  (b) triángulos / clustering / ciclos de 4 (Parte A, `null3_motivos_directos.py`, reusado tal cual)
  (c) KS del perfil radial contra REAL tras `layout_resortes` + la MISMA dilatación estática que usa
      toda la jerarquía (duplica la FORMA de la cola de `null3_generar_ic.generar_null3` -- layout +
      Expansion -- para poder alimentarle un adj ya construido con cualquier mecanismo de swap, cosa que
      `generar_null3` no permite porque llama a `barajar_aristas_preservando_longitud` internamente; se
      duplica la forma, no la lógica: layout_resortes/Expansion/T0/_T_reloj se IMPORTAN sin modificar).

No toca ningún archivo congelado ni ninguna carpeta de batería/piloto anterior -- sólo lee
`bateria_n2000/ic_real/cosmogenesis_ic.txt` (ya en disco). No corre Phantom (eso es
`null3_dosis_piloto_generar.py`/`_correr.py`, el siguiente escalón de esta misma Parte B).
"""
import time

import numpy as np
from scipy.stats import ks_2samp

from cs073_cierre_holistico import T0, _T_reloj
from cs072_modulos.piezas.p_expansion import Expansion
from cs072_modulos.piezas.p_semilla_causal import layout_resortes
from null3_investigacion_preliminar import (
    reconstruir_grafo_real, barajar_aristas_preservando_longitud, barajar_aristas_sin_restriccion,
)
from null3_motivos_directos import contar_triangulos_y_clustering, contar_ciclos_4

SEED = 501
NIVELES_TOL = [0.2, 0.4, 0.8]   # + "sin filtro" aparte, ver main()
ITERS_LAYOUT = 100
SEED_LAYOUT = 12345
N_PASOS_EXPANSION = 60


def layout_y_expandir(adj_null: dict, n: int, lado: float) -> np.ndarray:
    """Duplica la FORMA de la cola de `null3_generar_ic.generar_null3` (layout de resortes + MISMA
    dilatación isótropa estática que usa toda la jerarquía) para un adj ya construido por cualquier
    mecanismo de swap -- generar_null3 no acepta un adj externo (construye el suyo internamente), así
    que no se puede reusar directo para el nivel 'sin filtro'. layout_resortes/Expansion/T0/_T_reloj se
    IMPORTAN de piezas congeladas, no se reescriben."""
    pos = layout_resortes(adj_null, n, lado=lado, iters=ITERS_LAYOUT, seed=SEED_LAYOUT)
    expansion = Expansion(T0=T0)
    for step in range(N_PASOS_EXPANSION):
        expansion.paso_de_estiramiento(_T_reloj(step))
    a_final = expansion._a_prev
    return pos * a_final


def medir_nivel(nombre, adj_null, n, lado, r_real, aceptados=None, intentos=None):
    t0 = time.time()
    motivos = contar_triangulos_y_clustering(adj_null, n)
    motivos["ciclos_4"] = contar_ciclos_4(adj_null, n)
    t_motivos = time.time() - t0

    t1 = time.time()
    pos_n = layout_y_expandir(adj_null, n, lado)
    r_n = np.linalg.norm(pos_n - pos_n.mean(axis=0), axis=1)
    ks = ks_2samp(r_real, r_n)
    t_layout = time.time() - t1

    tasa = (100.0 * aceptados / intentos) if intentos else 0.0
    print(f"[{nombre}] tasa_aceptacion={tasa:.2f}%  triángulos={motivos['n_triangulos']}  "
          f"clustering_prom={motivos['clustering_promedio']:.5f}  ciclos_4={motivos['ciclos_4']}  "
          f"r_mean={r_n.mean():.3f}  r_std={r_n.std():.3f}  KS_radial={ks.statistic:.4f}  "
          f"p={ks.pvalue:.3e}  (motivos: {t_motivos:.2f}s, layout: {t_layout:.2f}s)", flush=True)

    return dict(nombre=nombre, tasa_aceptacion=tasa, aceptados=aceptados, intentos=intentos,
                n_triangulos=motivos["n_triangulos"], clustering_promedio=motivos["clustering_promedio"],
                clustering_global=motivos["clustering_global"], ciclos_4=motivos["ciclos_4"],
                r_mean=float(r_n.mean()), r_std=float(r_n.std()), ks_stat=float(ks.statistic),
                ks_p=float(ks.pvalue))


def main():
    t0 = time.time()
    print("[1] reconstruyendo grafo REAL + leyendo posiciones REAL ya en disco...", flush=True)
    adj_real, pos_real, n = reconstruir_grafo_real()
    lado = float(n) ** (1.0 / 3.0)
    com_real = pos_real.mean(axis=0)
    r_real = np.linalg.norm(pos_real - com_real, axis=1)
    motivos_real = contar_triangulos_y_clustering(adj_real, n)
    motivos_real["ciclos_4"] = contar_ciclos_4(adj_real, n)
    print(f"    n={n}  r_mean={r_real.mean():.3f} r_std={r_real.std():.3f}  "
          f"triángulos={motivos_real['n_triangulos']}  "
          f"clustering_prom={motivos_real['clustering_promedio']:.5f}  "
          f"ciclos_4={motivos_real['ciclos_4']}  tiempo={time.time()-t0:.2f}s\n", flush=True)

    resultados = [dict(nombre="REAL", tasa_aceptacion=None, aceptados=None, intentos=None,
                        n_triangulos=motivos_real["n_triangulos"],
                        clustering_promedio=motivos_real["clustering_promedio"],
                        clustering_global=motivos_real["clustering_global"],
                        ciclos_4=motivos_real["ciclos_4"], r_mean=float(r_real.mean()),
                        r_std=float(r_real.std()), ks_stat=0.0, ks_p=1.0)]

    print("[2] barriendo tol_relativa (mismo seed=501 en todos los niveles):", flush=True)
    for tol in NIVELES_TOL:
        adj_null, aceptados, intentos = barajar_aristas_preservando_longitud(
            adj_real, n, pos_real, seed=SEED, tol_relativa=tol)
        r = medir_nivel(f"tol_relativa={tol}", adj_null, n, lado, r_real, aceptados, intentos)
        resultados.append(r)

    print("\n[3] nivel sin filtro (Maslov-Sneppen puro, mismo seed=501 -- mecanismo de los NULL1-8 "
          "originales):", flush=True)
    adj_sin = barajar_aristas_sin_restriccion(adj_real, n, seed=SEED)
    r = medir_nivel("sin filtro", adj_sin, n, lado, r_real, aceptados=None, intentos=None)
    resultados.append(r)

    print("\n[4] tabla resumen dosis-respuesta:\n", flush=True)
    header = (f"{'nivel':22s} {'tasa_acept':>11s} {'triángulos':>11s} {'clust_prom':>11s} "
              f"{'ciclos_4':>9s} {'KS_radial':>10s} {'p_KS':>10s}")
    print(header)
    print("-" * len(header))
    for r in resultados:
        tasa_str = f"{r['tasa_aceptacion']:.2f}%" if r["tasa_aceptacion"] is not None else "—"
        print(f"{r['nombre']:22s} {tasa_str:>11s} {r['n_triangulos']:>11d} "
              f"{r['clustering_promedio']:>11.5f} {r['ciclos_4']:>9d} {r['ks_stat']:>10.4f} "
              f"{r['ks_p']:>10.3e}")

    print(f"\n[TOTAL] {time.time()-t0:.2f}s", flush=True)
    return resultados


if __name__ == "__main__":
    main()
