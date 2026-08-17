#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E5.2-1 — Verificación cruzada #1 (regla de ejecución #4 / protocolo §9 punto 1)
================================================================================
Compara, paso a paso, la reimplementación vectorizada en lote (E5_2_1_engine.py,
funciones paso_difusion_batch/paso_expansion_batch) contra las funciones ORIGINALES
de cs074_rcruz.py (paso_difusion/paso_expansion), SIN editar ese archivo.

Se restringe a ncols=1 (una sola columna) para poder comparar 1-a-1: ambas
implementaciones deben consumir los MISMOS números aleatorios en el mismo orden
(rng.random(shape) sobre un Generator PCG64 produce la misma secuencia subyacente
sin importar si el shape pedido es (N,) o (N,1) — numpy rellena en el mismo orden y
solo cambia el reshape final), así que deben coincidir a precisión de máquina si la
física es idéntica.

Si NO coinciden, este script para y reporta el error ANTES de correr el barrido
principal (regla: "si ves error, PARA y repórtalo").
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cs074_rcruz import paso_difusion as paso_difusion_base
from cs074_rcruz import paso_expansion as paso_expansion_base
from cs074_rcruz import campo_inicial as campo_inicial_base

from E5_2_1_engine import paso_difusion_batch, paso_expansion_batch

OUT = Path(__file__).resolve().parent


def verificar(N=200, eps=0.3, H=0.02, pasos=500, seed=777):
    rng_base = np.random.default_rng(seed)
    phi_base, _ = campo_inicial_base(N, eps, rng_base)
    activo_base = np.ones(N, dtype=bool)

    rng_batch = np.random.default_rng(seed)
    phi_batch, _ = campo_inicial_base(N, eps, rng_batch)
    activo_batch = np.ones((N, 1), dtype=bool)
    phi_batch = phi_batch.reshape(N, 1).copy()

    max_diff_phi = 0.0
    max_diff_activo = 0
    primer_paso_diff = None

    for t in range(pasos):
        phi_base = paso_difusion_base(phi_base, activo_base)
        activo_base = paso_expansion_base(activo_base, H, rng_base)

        phi_batch = paso_difusion_batch(phi_batch, activo_batch)
        activo_batch = paso_expansion_batch(activo_batch, np.array([[H]]), rng_batch)

        d_phi = float(np.max(np.abs(phi_base - phi_batch[:, 0])))
        d_act = int(np.sum(activo_base != activo_batch[:, 0]))
        max_diff_phi = max(max_diff_phi, d_phi)
        max_diff_activo = max(max_diff_activo, d_act)
        if (d_phi > 0 or d_act > 0) and primer_paso_diff is None:
            primer_paso_diff = t

    resultado = {
        "N": N, "eps": eps, "H": H, "pasos": pasos, "seed": seed,
        "max_diff_phi": max_diff_phi,
        "max_diff_activo_count": max_diff_activo,
        "primer_paso_con_diferencia": primer_paso_diff,
        "PASS": bool(max_diff_phi == 0.0 and max_diff_activo == 0),
    }
    return resultado


def main():
    casos = [
        dict(eps=0.0, H=0.0, pasos=300, seed=1),
        dict(eps=0.3, H=0.0, pasos=300, seed=2),
        dict(eps=0.3, H=0.02, pasos=500, seed=3),
        dict(eps=1.0, H=0.2, pasos=800, seed=4),
        dict(eps=1e-6, H=1.0, pasos=200, seed=5),
    ]
    resultados = [verificar(**c) for c in casos]
    todo_pass = all(r["PASS"] for r in resultados)
    out = {"todo_pass": todo_pass, "casos": resultados}
    (OUT / "E5_2_1_verificacion_motor.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(out, indent=2, ensure_ascii=False))
    if not todo_pass:
        print("\n[PARADA] El motor propio NO reproduce la física original bit a bit. "
              "No se corre el barrido principal hasta corregir esto.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
