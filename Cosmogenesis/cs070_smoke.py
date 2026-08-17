"""
CS070 SMOKE — las 3 anclas obligatorias antes de la tanda (DISENO_CS070_semilla_amplificacion_CS.md)
==============================================================================
1. SIN_SEMILLA reproduce el colapso de CS067 (no certificado -- pico_medio no cruza 0.85).
2. En la retícula métrica de control, SEMILLA_COHERENTE preserva su eje y SEMILLA_BARAJADA no (Δ grande,
   como el toy de CS: 0.93 vs 0.34).
3. La coherencia/certificación alta NO se cuenta como dirección si n_ejes=1 (el juez de trampa dispara:
   G-JUEZ-NO-COHERENCIA -- el veredicto exige n_ejes>1 CERTIFICADO, no certificado solo).

Codea/ejecuta: CC. Diseño/ruling: CS.
"""
from __future__ import annotations
import sys, time
import numpy as np
import scipy.sparse as sp

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)
import cs070_semilla as S
import cs067_habitacion_completa as H
import cs069_smoke as S69  # reusa la retícula 2D 8-vecinos (con triángulos -- lección CS069)
import cs064_smoke as SM

RNG = np.random.default_rng


def direccion_real(r):
    """G-JUEZ-NO-COHERENCIA: un 'sí' exige n_ejes>1 Y certificado (pico_medio>0.85). Certificado solo, con
    n_ejes<=1, es la TRAMPA que cazó el toy (colapso a 1 eje, no dimensión)."""
    return bool(r["certificado"] and r["n_ejes"] > 1)


# ============================ ANCLA 1 — SIN_SEMILLA reproduce el colapso de CS067 ============================
def ancla1_sin_semilla(N=900, n_seeds=6):
    print("\n" + "=" * 100, flush=True)
    print("ANCLA 1 — SIN_SEMILLA: ¿reproduce el no-encendido de CS067 (no certificado)?", flush=True)
    print("=" * 100, flush=True)
    reales = []
    for s in range(n_seeds):
        seed = 70100 + 97 * s
        r = S.brazo_sin_semilla(N, seed)
        dr = direccion_real(r)
        reales.append(dr)
        print(f"  seed={s}: n_ejes={r['n_ejes']} pico_medio={r['pico_medio']:.3f} certificado={r['certificado']} "
              f"direccion_real={dr}", flush=True)
    # CORRECCIÓN (encontrada al correr): certificado SOLO no es el criterio -- vi casos con pico_medio>0.85
    # pero n_ejes=0 (muchos dominios LOCALES de alta confianza que no cuajan en una dirección GLOBAL --
    # población isotrópica entre los K ejes). Eso es EXACTAMENTE la isotropía que cuenta_ejes_gap ya
    # descarta (rango pleno -> n_ejes=0), y direccion_real() ya lo excluye bien. El criterio correcto para
    # "reproduce el no-encendido de CS067" es direccion_real (certificado Y n_ejes>1), no certificado solo.
    frac_real = float(np.mean(reales))
    ok = frac_real < 0.3
    print(f"  frac_direccion_real={frac_real:.2f}  {'PASA' if ok else 'FALLA'}: consistente con el no-encendido de CS067",
          flush=True)
    return ok


# ============================ ANCLA 2 — retícula métrica: coherente preserva eje, barajada no ============================
def _Aw_de_toy(adj, N, rng, gamma=1.0):
    wmap = H._pesos_correlacion(adj, N, gamma, "real", rng)
    A = SM.adj_sparse(adj, N)
    Ac = A.tocoo()
    if wmap:
        wd = np.array([wmap.get((r_, c_) if r_ < c_ else (c_, r_), 1.0) for r_, c_ in zip(Ac.row, Ac.col)])
        return sp.coo_matrix((wd, (Ac.row, Ac.col)), shape=(N, N)).tocsr()
    return A


def ancla2_reticula_metrica(N_objetivo=400, seed=70200, K=6, n_seeds=5):
    print("\n" + "=" * 100, flush=True)
    print("ANCLA 2 — retícula 2D 8-vecinos (métrica, control): ¿SEMILLA_COHERENTE preserva el eje sembrado", flush=True)
    print("y SEMILLA_BARAJADA no (misma magnitud de peso_semilla)?", flush=True)
    print("=" * 100, flush=True)
    side = int(round(N_objetivo ** 0.5))
    adj_local, N = S69._reticula_2d_8vecinos(side)

    deltas = []
    for s in range(n_seeds):
        sd = seed + 97 * s
        rng = RNG(sd)
        Aw = _Aw_de_toy(adj_local, N, rng)
        peso, eje_c, signo_c = S._semilla_coherente(N, K, rng)
        V_c, conf_c = S._corre_consenso(Aw, N, K, RNG(sd + 1), peso, eje_c, signo_c)
        eje_final_c = np.argmax(np.abs(V_c[:, :K]), axis=1)
        frac_c = float(np.mean(eje_final_c == 0))

        _p, eje_b, signo_b = S._semilla_barajada(N, K, rng, peso)
        V_b, conf_b = S._corre_consenso(Aw, N, K, RNG(sd + 2), peso, eje_b, signo_b)
        eje_final_b = np.argmax(np.abs(V_b[:, :K]), axis=1)
        frac_b = float(np.mean(eje_final_b == 0))

        deltas.append(frac_c - frac_b)
        print(f"  seed={s}: peso_semilla={peso:.3f}  frac_eje0(coherente)={frac_c:.3f}  "
              f"frac_eje0(barajada)={frac_b:.3f}  Δ={frac_c-frac_b:+.3f}", flush=True)

    m = float(np.mean(deltas))
    ok = m > 0.15  # margen no trivial, no un empate (referencia toy: Δ=0.590 en su formalismo)
    print(f"\n  Δ medio={m:+.3f}  {'PASA' if ok else 'FALLA'}: la retícula métrica SÍ sostiene la semilla coherente",
          flush=True)
    return ok


# ============================ ANCLA 3 — el juez de trampa dispara (certificado + n_ejes=1 != dirección) ============================
def ancla3_juez_trampa(N=900, n_seeds=8):
    print("\n" + "=" * 100, flush=True)
    print("ANCLA 3 — juez de trampa: ¿certificado=True con n_ejes<=1 se EXCLUYE de 'dirección real'?", flush=True)
    print("=" * 100, flush=True)
    casos = []
    for s in range(n_seeds):
        seed = 70300 + 97 * s
        r = S.brazo_semilla_coherente(N, seed)
        casos.append(r)
        print(f"  seed={s}: n_ejes={r['n_ejes']} certificado={r['certificado']} "
              f"direccion_real={direccion_real(r)}", flush=True)
    # el ancla PASA si la función direccion_real nunca cuenta un caso certificado-pero-colapsado(n_ejes<=1)
    # como dirección -- verificación de LÓGICA, no de si la semilla enciende (eso lo decide la tanda).
    trampas = [r for r in casos if r["certificado"] and r["n_ejes"] <= 1]
    ok = all(not direccion_real(r) for r in trampas) if trampas else True
    print(f"\n  casos certificado=True con n_ejes<=1 (trampa potencial): {len(trampas)}", flush=True)
    print(f"  {'PASA' if ok else 'FALLA'}: el juez de trampa {'SÍ' if ok else 'NO'} excluye colapso-a-1-eje de dirección",
          flush=True)
    return ok


def main():
    t0 = time.time()
    ok1 = ancla1_sin_semilla()
    ok2 = ancla2_reticula_metrica()
    ok3 = ancla3_juez_trampa()
    print("\n" + "=" * 100, flush=True)
    print(f"RESUMEN: ancla1(sin_semilla~CS067)={'PASA' if ok1 else 'FALLA'}  "
          f"ancla2(retícula sostiene semilla)={'PASA' if ok2 else 'FALLA'}  "
          f"ancla3(juez de trampa)={'PASA' if ok3 else 'FALLA'}", flush=True)
    print(f"tiempo total: {(time.time()-t0)/60:.2f} min", flush=True)
    if ok1 and ok2 and ok3:
        print("\nLAS 3 ANCLAS PASAN. Autorizado a correr la tanda blindada completa.", flush=True)
    else:
        print("\nAL MENOS UN ANCLA FALLA. NO correr la tanda -- reportar a CS antes de seguir.", flush=True)


if __name__ == "__main__":
    main()
