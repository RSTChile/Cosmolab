"""
RE-ANALISIS C-N2.5.5 — re-corre EXACTAMENTE la tanda CS070 (mismas semillas, mismos brazos, mismo motor)
pero GUARDA el espectro completo del tensor de orientacion (ev, K valores) de cada corrida, que _juzga()
descartaba. Con el espectro se puede recomputar n_ejes bajo guardas alternativas sin re-simular.

No toca ningun archivo del proyecto: solo intercepta SM.tensor_orientacion para copiar su salida.
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)
import cs064_smoke as SM
import cs070_semilla as S
from cs070_smoke import direccion_real
import cs067_habitacion_completa as H

_CAPT = {}
_orig_tensor = SM.tensor_orientacion


def _tensor_spy(V):
    ev = _orig_tensor(V)
    _CAPT["ev"] = np.asarray(ev, float).tolist()
    _CAPT["D"] = int(V.shape[1])
    return ev


SM.tensor_orientacion = _tensor_spy
S.SM.tensor_orientacion = _tensor_spy

NS = [900, 1500, 2500]
N_SEEDS = 8
ARMS = ["semilla_coherente", "semilla_barajada", "sin_semilla", "semilla_sustrato_local"]

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cs070_espectros.json")


def main():
    t0 = time.time()
    filas = []
    for arm in ARMS:
        fn = S.BRAZOS[arm]
        for i, N in enumerate(NS):
            for s in range(N_SEEDS):
                seed = 70600 + 1000 * ARMS.index(arm) + 97 * i + 13 * s
                _CAPT.clear()
                r = fn(N, seed)
                r["arm"] = arm; r["N"] = N; r["seed"] = seed
                r["direccion_real"] = direccion_real(r)
                r["ev"] = _CAPT.get("ev")
                r["D"] = _CAPT.get("D")
                r["K_sorteado"] = int(H._sorteo(seed)["K"])
                filas.append(r)
                print(f"  [{arm:22s} N={N:5d} s={s}] K={r['K_sorteado']} D={r['D']} n_ejes={r['n_ejes']} "
                      f"pico={r['pico_medio']:.3f} cert={r['certificado']} "
                      f"(t={(time.time()-t0)/60:.1f}min)", flush=True)
                with open(OUT, "w") as f:
                    json.dump(filas, f, indent=1, default=float)
    print(f"LISTO {(time.time()-t0)/60:.2f} min -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
