#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs075_E5b_prueba_estructura_exergia.py -- El control directo que pide
INSTRUCCION_CS075_v2_EJECUCION_PARA_CC.md paso 2 (bloqueante).

E5 (bateria completa) encontro que X sube en vez de bajar, y CC propuso una lectura
fisica: 2_gravedad y 3_fuerte concentran densidad, compitiendo contra la difusion. El
control que CC invoco (EstadoFisico sola, sin agentes, X SI baja) no prueba esa lectura:
la diferencia entre "sola" y "con los 23" son LOS 23 agentes, no esos dos en particular.

Esta es la prueba directa: correr con los 23 agentes MENOS {2_gravedad, 3_fuerte} (los
otros 21 quedan), y medir si X vuelve a bajar. Si baja, la lectura de CC queda medida, no
supuesta. Si sigue subiendo, la lectura es falsa y hay que parar y reportar (no forzar).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cs075_23_sobre_fisica import construir_23  # noqa: E402

T_TOTAL = 20.0
N = 16
DT = 1e-3
SEED = 7        # mismo seed que E5_direcciones_se_mantienen (cs075_pruebas_23_sobre_fisica.py l.168)
AMP = 0.1       # default de construir_23, igual que E5 (no lo pasa explícito)


def correr(excluir=(), incluir_solo=None, T_total=T_TOTAL):
    proceso, agentes = construir_23(N=N, dt=DT, seed=SEED, amp_asimetria=AMP)
    if incluir_solo is not None:
        agentes = [a for a in agentes if a.nombre in incluir_solo]
    else:
        agentes = [a for a in agentes if a.nombre not in excluir]
    e0 = proceso.estado.estado()
    proceso.correr(agentes, T_total=T_total, registrar_cada=0)
    e1 = proceso.estado.estado()
    X_inicial, X_final = float(e0["X"]), float(e1["X"])
    return dict(X_inicial=X_inicial, X_final=X_final, X_baja=bool(X_final <= X_inicial + 1e-9),
                n_agentes=len(agentes), nombres=[a.nombre for a in agentes])


def main():
    t0 = time.time()
    resultado = {}

    print("=== BASELINE: los 23 completos ===", flush=True)
    resultado["baseline_23"] = correr(excluir=())
    print(resultado["baseline_23"], flush=True)

    print("=== E5b: apagar 2_gravedad y 3_fuerte, dejar los otros 21 ===", flush=True)
    resultado["sin_2_gravedad_3_fuerte"] = correr(excluir=("2_gravedad", "3_fuerte"))
    print(resultado["sin_2_gravedad_3_fuerte"], flush=True)

    print("=== Variante: SOLO 2_gravedad y 3_fuerte encendidos ===", flush=True)
    resultado["solo_2_gravedad_3_fuerte"] = correr(incluir_solo=("2_gravedad", "3_fuerte"))
    print(resultado["solo_2_gravedad_3_fuerte"], flush=True)

    resultado["elapsed_s"] = time.time() - t0
    out = HERE / "cs075_resultado_E5b_prueba_estructura_exergia.json"
    out.write_text(json.dumps(resultado, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[archivo] {out}", flush=True)
    print(f"[elapsed] {resultado['elapsed_s']:.1f}s", flush=True)

    print("\n=== VEREDICTO ===")
    base_sube = not resultado["baseline_23"]["X_baja"]
    e5b_baja = resultado["sin_2_gravedad_3_fuerte"]["X_baja"]
    print(f"baseline (23): X {'baja' if not base_sube else 'SUBE'}")
    print(f"sin 2_gravedad/3_fuerte: X {'baja' if e5b_baja else 'SUBE'}")
    if base_sube and e5b_baja:
        print("-> La lectura de CC queda MEDIDA: los agentes de estructura local generan exergia.")
    elif base_sube and not e5b_baja:
        print("-> La lectura de CC es FALSA. Hay otro mecanismo (o un bug mas). PARAR Y REPORTAR.")
    else:
        print("-> Caso no anticipado por el protocolo. PARAR Y REPORTAR.")


if __name__ == "__main__":
    main()
