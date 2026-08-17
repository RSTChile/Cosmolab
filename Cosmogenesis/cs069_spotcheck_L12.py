"""
CS069 — spot-check confirmatorio de L (ADJUDICACION_CS069_tanda_cierre_CS.md)
==============================================================================
Ruling de CS: (B) es robusto a la regla de fase (validada, AUC 0.843) y al mecanismo, pero antes de cerrar
falta descartar que L=8 esté truncando algo. Diagnóstico de CS: L=8 acumula 97.9% de la amplitud
(decaimiento geométrico limpio, razón ~0.62/paso, sin resurgencia), y los caminos largos DECOHEREN MÁS (más
términos de fase acumulados), así que si los cortos no codifican dirección, los largos no la rescatan.

Spot-check (no barrido -- honra CS058 sin caer en Shannon): brazo COMPLETO, N=1500, 8 semillas, L=12 (en
vez de L=8). Predicción PRE-INSCRITA de CS: Juez C sigue en 0% certificado. Si aparece cualquier indicio
(gap>0 en >=2 semillas) -> se abre un barrido de L de verdad. Si sigue 0% -> (B) CANÓNICO, CS069 cierra.

Codea/ejecuta: CC. Diseño/ruling: CS.
"""
from __future__ import annotations
import sys, time
import numpy as np

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)
import cs069_quantum_graph as Q
import cs068_inflacion_estirar_enfriar as E

RNG = np.random.default_rng
N = 1500
N_SEEDS = 8
L_SPOT = 12


def main():
    t0 = time.time()
    print("=" * 100, flush=True)
    print(f"CS069 SPOT-CHECK L={L_SPOT} (vs L=8 de la tanda) -- brazo COMPLETO, N={N}, {N_SEEDS} semillas",
          flush=True)
    print("=" * 100, flush=True)
    diamqs, certificados, n_ejess, pico_medios = [], [], [], []
    for s in range(N_SEEDS):
        seed = 69400 + 97 * s
        adj = E._sustrato(N, seed)
        Dq = Q.brazo_completo(adj, N, RNG(seed + 1), L=L_SPOT)
        diamq = Q.diam_q_robusto(Dq, N, RNG(seed + 2))
        juezc = Q.juez_gap_espectral(Dq, N)
        diamqs.append(diamq); certificados.append(juezc["certificado"])
        n_ejess.append(juezc["n_ejes"]); pico_medios.append(juezc["pico_medio"])
        print(f"  s={s}: diam_q={diamq:.2f}  n_ejes={juezc['n_ejes']}  pico_medio={juezc['pico_medio']:.3f}  "
              f"certificado={juezc['certificado']}  (t={(time.time()-t0)/60:.1f}min)", flush=True)

    print(f"\ndiam_q media(L={L_SPOT})={np.mean(diamqs):.2f}  (referencia tanda, L=8, N=1500, completo: "
          f"pendiente 0.132, ver INFORME_CS069_tanda_PARA_CS.md)", flush=True)
    print(f"n_ejes: todos={n_ejess}  frac_certificado(>0.85)={np.mean(certificados):.2f}", flush=True)
    print(f"pico_medio media={np.mean(pico_medios):.3f}", flush=True)

    frac_cert = float(np.mean(certificados))
    n_con_gap = sum(1 for n in n_ejess if n > 0)
    print(f"\ntiempo total: {(time.time()-t0)/60:.2f} min", flush=True)
    if frac_cert == 0.0 and n_con_gap < 2:
        print("\nCONFIRMA la predicción de CS: sigue 0% certificado a L=12. (B) CANÓNICO -- CS069 cierra.",
              flush=True)
    else:
        print("\nAPARECIÓ INDICIO a L=12 (gap>0 en >=2 semillas, o certificado>0) -- NO cierra automático,", flush=True)
        print("reportar a CS: puede hacer falta un barrido de L de verdad.", flush=True)


if __name__ == "__main__":
    main()
