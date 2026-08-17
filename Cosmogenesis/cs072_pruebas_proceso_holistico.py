#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs072_pruebas_proceso_holistico.py -- verifica que cs072_proceso_holistico.py (las 23
piezas envueltas como agentes simultaneos) reproduce EXACTO lo que ya da
cs072_motor_23.py (el motor probado, secuencial). No se fuerza nada -- si algo no
coincide, se reporta tal cual, no se ajusta hasta que coincida.

P1: los 4 brazos del propio __main__ de cs072_motor_23.py.
P2: invariancia a permutacion (5 semillas).
P3: admisibilidad pieza por pieza (20 piezas togglables) -- mismo resultado apagado
    cada una, seq vs holistico.
P4: el inventario de 23 (INVENTARIO_23) esta completo, sin repetidos.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cs072_motor_23 import corre, cuenta  # noqa: E402
from cs072_proceso_holistico import corre_holistico, INVENTARIO_23  # noqa: E402

ARGS = (30, 21, 10, 7)
resultados = {}


def log(msg):
    print(msg, flush=True)


def P1_cuatro_brazos():
    log("\n" + "=" * 78)
    log("P1 -- los 4 brazos del __main__ de cs072_motor_23.py, seq vs holistico")
    log("=" * 78)
    filas = []
    todo_ok = True
    for (h, e, lab) in [(True, False, "A homog"), (True, True, "B homog+exp"),
                        (False, False, "C grad"), (False, True, "D grad+exp")]:
        cs = cuenta(corre(*ARGS, homogeneo=h, expansion=e, pasos=300))
        ch = cuenta(corre_holistico(*ARGS, homogeneo=h, expansion=e, pasos=300))
        ok = cs == ch
        todo_ok = todo_ok and ok
        log(f"  {lab:12s}: seq={cs}  hol={ch}  {'IGUAL' if ok else '*** DISTINTO ***'}")
        filas.append(dict(brazo=lab, homogeneo=h, expansion=e, seq=cs, hol=ch, igual=bool(ok)))
    log(f"\n  -> P1: {'PASA' if todo_ok else 'FALLA'}")
    resultados["P1"] = dict(paso=bool(todo_ok), filas=filas)
    return todo_ok


def P2_invariancia_permutacion():
    log("\n" + "=" * 78)
    log("P2 -- invariancia a permutacion (holistico)")
    log("=" * 78)
    base = cuenta(corre_holistico(*ARGS, homogeneo=False, expansion=True, pasos=300))
    N = sum(ARGS)
    vals = [cuenta(corre_holistico(*ARGS, homogeneo=False, expansion=True, pasos=300,
                                    perm=np.random.RandomState(s).permutation(N)))["bariones"]
            for s in range(5)]
    inv = all(v == base["bariones"] for v in vals)
    log(f"  base={base['bariones']} perms={vals} INVARIANTE={inv}")
    log(f"\n  -> P2: {'PASA' if inv else 'FALLA'}")
    resultados["P2"] = dict(paso=bool(inv), base=base["bariones"], perms=vals)
    return inv


def P3_admisibilidad_por_pieza():
    log("\n" + "=" * 78)
    log("P3 -- admisibilidad pieza por pieza, seq vs holistico")
    log("=" * 78)
    piezas = ["1_espin", "2_gravedad", "3_fuerte", "4_em", "5_debil", "7_masa",
              "8_aniquilacion", "9_expansion", "10_enfriamiento", "11_tres_cuerpos",
              "12_localidad", "13_pauli", "14_correlacion", "15_causal", "16_ssb",
              "17_oscuro", "22_qcd", "23_campo", "M1_semilla", "M2_memoria"]
    filas = []
    todo_ok = True
    for p in piezas:
        cs = cuenta(corre(*ARGS, homogeneo=False, expansion=True, pasos=300, apagar=frozenset([p])))
        ch = cuenta(corre_holistico(*ARGS, homogeneo=False, expansion=True, pasos=300, apagar=frozenset([p])))
        clave = ("bariones", "hidrogeno", "quarks_sueltos")
        ok = all(cs[k] == ch[k] for k in clave)
        todo_ok = todo_ok and ok
        log(f"  sin {p:16s}: seq bar={cs['bariones']} H={cs['hidrogeno']}  "
            f"hol bar={ch['bariones']} H={ch['hidrogeno']}  {'SI' if ok else '*** NO ***'}")
        filas.append(dict(pieza=p, seq=cs, hol=ch, igual=bool(ok)))
    log(f"\n  -> P3: {'PASA' if todo_ok else 'FALLA'} ({sum(f['igual'] for f in filas)}/{len(filas)} coinciden)")
    resultados["P3"] = dict(paso=bool(todo_ok), filas=filas)
    return todo_ok


def P4_inventario_23():
    log("\n" + "=" * 78)
    log("P4 -- inventario de 23, sin repetidos")
    log("=" * 78)
    n = len(INVENTARIO_23)
    nombres = [p["nombre"] for p in INVENTARIO_23]
    sin_repetidos = len(nombres) == len(set(nombres))
    ok = (n == 23) and sin_repetidos
    log(f"  {n} entradas, sin repetidos: {sin_repetidos}")
    for p in INVENTARIO_23:
        log(f"    #{p['numero']:<4} {p['nombre']:20s} {p['estado']}")
    log(f"\n  -> P4: {'PASA' if ok else 'FALLA'}")
    resultados["P4"] = dict(paso=bool(ok), n=n, sin_repetidos=sin_repetidos, inventario=INVENTARIO_23)
    return ok


def main():
    t0 = time.time()
    r1 = P1_cuatro_brazos()
    r2 = P2_invariancia_permutacion()
    r3 = P3_admisibilidad_por_pieza()
    r4 = P4_inventario_23()

    log("\n" + "=" * 78)
    log("RESUMEN")
    log("=" * 78)
    log(f"  P1 (4 brazos coinciden)          {'PASA' if r1 else 'FALLA'}")
    log(f"  P2 (invariancia a permutacion)   {'PASA' if r2 else 'FALLA'}")
    log(f"  P3 (admisibilidad por pieza)     {'PASA' if r3 else 'FALLA'}")
    log(f"  P4 (inventario de 23)            {'PASA' if r4 else 'FALLA'}")
    completo = r1 and r2 and r3 and r4
    log(f"\n  bateria completa: {'PASA' if completo else 'FALLA'}")
    elapsed = time.time() - t0
    log(f"  tiempo: {elapsed:.1f}s")

    resultados["resumen"] = dict(P1=bool(r1), P2=bool(r2), P3=bool(r3), P4=bool(r4),
                                 completo=bool(completo), elapsed_s=elapsed)
    out = HERE / "cs072_resultado_pruebas_proceso_holistico.json"
    out.write_text(json.dumps(resultados, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"\n[archivo] {out}")


if __name__ == "__main__":
    main()
