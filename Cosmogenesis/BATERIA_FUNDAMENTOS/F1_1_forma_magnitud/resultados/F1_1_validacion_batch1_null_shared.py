#!/usr/bin/env python3
"""Validación: el motor batched debe reproducir EXACTAMENTE base.corrida()
(mismas fórmulas, ejecutadas vectorizadas). Compara P_real y P_null."""
import importlib.util
import numpy as np
import time

spec = importlib.util.spec_from_file_location('base', '/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs074_rcruz.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def campo_inicial_batch(N, eps_col, rng, S):
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    fondo = np.ones((S, N))
    fases = rng.uniform(0, 2 * np.pi, size=(S, 5))
    pert = np.zeros((S, N))
    for m in range(1, 6):
        pert += np.sin(2 * np.pi * m * x[None, :] + fases[:, m - 1:m]) / m
    pert -= pert.mean(axis=1, keepdims=True)
    std = pert.std(axis=1, keepdims=True)
    pert = np.divide(pert, std, out=np.zeros_like(pert), where=std > 0)
    eps_col = np.asarray(eps_col).reshape(S, 1)
    phi = np.where(eps_col <= 0.0, fondo, fondo + eps_col * pert)
    return phi


def paso_difusion_batch(phi, activo):
    left = np.roll(phi, 1, axis=-1)
    right = np.roll(phi, -1, axis=-1)
    e_left = np.roll(activo, 1, axis=-1)
    e_right = activo
    n_nb = e_left.astype(np.float64) + e_right.astype(np.float64)
    s = e_left * left + e_right * right
    media = np.divide(s, n_nb, out=phi.copy(), where=n_nb > 0)
    nuevo = phi + 0.5 * (media - phi)
    return np.where(n_nb > 0, nuevo, phi)


def paso_expansion_batch(activo, H_col, rng):
    u = rng.random(activo.shape)
    Hthr = np.clip(H_col, 0.0, 1.0)
    cortar = activo & (u < Hthr)
    nuevo = activo & ~cortar
    return nuevo


def persistencia_batch(phi, contraste0_col):
    a = phi
    b = np.roll(phi, 1, axis=-1)
    a_mean = a.mean(axis=-1, keepdims=True)
    b_mean = b.mean(axis=-1, keepdims=True)
    cov = ((a - a_mean) * (b - b_mean)).mean(axis=-1)
    sa = a.std(axis=-1)
    sb = b.std(axis=-1)
    denom = sa * sb
    c = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0)
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    c = np.maximum(c, 0.0)
    v = phi.var(axis=-1) / (contraste0_col.reshape(-1) ** 2)
    contraste0_ok = contraste0_col.reshape(-1) > 0
    phi_std_ok = phi.std(axis=-1) > 1e-12
    P = np.where(contraste0_ok & phi_std_ok, c * v, 0.0)
    return P


def permutar_batch(phi, rng):
    idx = rng.random(phi.shape).argsort(axis=-1)
    return np.take_along_axis(phi, idx, axis=-1)


# --- Validación 1: S=1 debe coincidir EXACTO con base.corrida() ---
N, eps, H, pasos, seed = 100, 1e-3, 0.02, 300, 4242
ref_real = base.corrida(N, eps, H, pasos, seed=seed, null=False)
ref_null = base.corrida(N, eps, H, pasos, seed=seed, null=True)

rng = np.random.default_rng(seed)
phi = campo_inicial_batch(N, [eps], rng, 1)
activo = np.ones((1, N), dtype=bool)
c0 = phi.std(axis=-1).copy()
for _ in range(pasos):
    phi = paso_difusion_batch(phi, activo)
    activo = paso_expansion_batch(activo, np.array([[H]]), rng)
P_real_batch = persistencia_batch(phi, c0)[0]

# null: nueva rng desde la misma seed (misma trayectoria) + permutación al final
rng2 = np.random.default_rng(seed)
phi2 = campo_inicial_batch(N, [eps], rng2, 1)
activo2 = np.ones((1, N), dtype=bool)
c02 = phi2.std(axis=-1).copy()
for _ in range(pasos):
    phi2 = paso_difusion_batch(phi2, activo2)
    activo2 = paso_expansion_batch(activo2, np.array([[H]]), rng2)
phi2 = permutar_batch(phi2, rng2)
P_null_batch = persistencia_batch(phi2, c02)[0]

print("=== Validación S=1 (misma seed, misma trayectoria) ===")
print(f"P_real  base={ref_real['P']:.10f}  batch={P_real_batch:.10f}  diff={abs(ref_real['P']-P_real_batch):.2e}")
print(f"P_null  base={ref_null['P']:.10f}  (batch usa OTRO random.permutation aunque misma seed, no exacto por diseño)")
print(f"P_null_batch={P_null_batch:.10f}")

# --- Validación 2 (más importante): optimización "1 trayectoria -> real+null"
# reproduce el MISMO resultado que llamar corrida(null=False) y corrida(null=True)
# por separado con la misma seed, para el CASO REAL (la trayectoria pre-permutación
# es idéntica). Confirmamos que evolucionar(null=False) y evolucionar(null=True)
# con la MISMA seed producen el MISMO phi final antes de permutar:
rng3 = np.random.default_rng(seed)
phi3, _ = base.campo_inicial(N, eps, rng3)
activo3 = np.ones(N, dtype=bool)
phi3, activo3, c03 = base.evolucionar(phi3, activo3, H, pasos, rng3, null=False)

rng4 = np.random.default_rng(seed)
phi4, _ = base.campo_inicial(N, eps, rng4)
activo4 = np.ones(N, dtype=bool)
# reproducir manualmente el loop de evolucionar SIN permutar, usando la misma rng4
c04 = float(phi4.std())
for _ in range(pasos):
    phi4 = base.paso_difusion(phi4, activo4)
    activo4 = base.paso_expansion(activo4, H, rng4)
# ahora sí permutar con rng4 (equivalente a null=True)
phi4_null = rng4.permutation(phi4)

print("\n=== Validación 2: trayectoria pre-permutación IDÉNTICA (null=False vs manual) ===")
print(f"max|phi3-phi4| (antes de permutar) = {np.max(np.abs(phi3-phi4)):.2e}  (debe ser 0)")
print(f"P_real(phi3) = {base.persistencia(phi3,c03):.10f}")
print(f"P_real(phi4, pre-permutar) = {base.persistencia(phi4,c04):.10f}  (debe == anterior)")
print(f"P_null oficial (corrida null=True) = {ref_null['P']:.10f}")
print(f"P_null(phi4_null, permutado con rng4 tras la MISMA trayectoria) = {base.persistencia(phi4_null,c04):.10f}")
print("(si estos dos últimos no calzan exacto es solo por la aleatoriedad del shuffle -mismo mecanismo, distinta extracción de la rng- ambos son 'permutación válida')")
