#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs075_diag_quien_sube_X.py -- diagnostico: que agente(s) empujan X hacia arriba.

X ~ mean((rho/mean-1)^2), proporcional a la varianza relativa de rho. A primer orden,
d(var)/dpaso ~ 2*cov(deposito_total, rho-mean). Por agente: cov(deposito_i, rho-mean) > 0
significa que ESE agente, en ESE paso, esta aumentando la varianza (anti-difusivo,
concentra donde ya hay exceso o vacia donde ya hay defecto); < 0 significa que la
esta bajando (homogeneiza). Acumulado sobre toda la corrida, agente por agente, identifica
quien domina la subida medida en E5b -- mismo principio de medicion que caso el bug de
22_qcd (medir, no razonar).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cs075_23_sobre_fisica import construir_23  # noqa: E402

N = 16
DT = 1e-3
SEED = 7
AMP = 0.1
T_TOTAL = 20.0
N_PASOS = int(T_TOTAL / DT)


def main():
    proceso, agentes = construir_23(N=N, dt=DT, seed=SEED, amp_asimetria=AMP)
    cov_acum = {a.nombre: 0.0 for a in agentes}
    cov_abs_acum = {a.nombre: 0.0 for a in agentes}
    n_activo = {a.nombre: 0 for a in agentes}

    for paso in range(N_PASOS):
        hitos = proceso._hitos()
        rho_centrado = proceso.estado.rho - proceso.estado.rho.mean()
        total = np.zeros_like(proceso.estado.rho)
        deps_paso = {}
        for ag in agentes:
            dado = ag.condiciones_dadas(proceso.estado, hitos)
            ag._registrar(proceso.paso_n, dado)
            if dado:
                dep = ag.deposito(proceso.estado, hitos)
                deps_paso[ag.nombre] = dep
                total = total + dep
        proceso.estado.paso(depositos=total)
        proceso.paso_n += 1
        for ag in agentes:
            if ag.nombre in deps_paso:
                cov = float(np.sum(deps_paso[ag.nombre] * rho_centrado))
                cov_acum[ag.nombre] += cov
                cov_abs_acum[ag.nombre] += abs(cov)
                n_activo[ag.nombre] += 1
        for ag in agentes:
            if ag.condiciones_dadas(proceso.estado, hitos):
                ag.consolidar(proceso.estado, hitos)

    ranking = sorted(cov_acum.items(), key=lambda kv: kv[1], reverse=True)
    print(f"{'agente':20s} {'cov_acum (sube X si >0)':>26s} {'pasos activo':>14s}")
    for nombre, cov in ranking:
        print(f"{nombre:20s} {cov:26.6g} {n_activo[nombre]:14d}")

    out = HERE / "cs075_resultado_diag_quien_sube_X.json"
    out.write_text(json.dumps(dict(
        cov_acum=cov_acum, cov_abs_acum=cov_abs_acum, n_activo=n_activo,
        ranking=[dict(nombre=n, cov=c) for n, c in ranking],
        N=N, dt=DT, seed=SEED, amp_asimetria=AMP, T_total=T_TOTAL,
        X_inicial=float(0.0), 
    ), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[archivo] {out}")
    print(f"X final: {proceso.estado.exergia():.6g}")


if __name__ == "__main__":
    main()
