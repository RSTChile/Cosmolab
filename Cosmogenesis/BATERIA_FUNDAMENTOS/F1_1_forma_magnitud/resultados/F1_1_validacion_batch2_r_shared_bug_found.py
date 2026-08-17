#!/usr/bin/env python3
"""Validación del motor r-batched (batching sobre el eje r, compartiendo la MISMA
secuencia de aleatoriedad que produce cs074_rcruz.py al llamar corrida() por
separado para cada r con la misma seed -- esto es una IDENTIDAD matemática, no
una aproximación, porque base.corrida() crea un rng FRESCO con la misma seed
para cada r, y el patrón de consumo (fases si eps>0, luego pasos*N draws) NO
depende de H/r. Se valida comparando contra base.corrida() para VARIOS r con
la misma seed/eps."""
import importlib.util
import time
import numpy as np

spec = importlib.util.spec_from_file_location('base', '/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs074_rcruz.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


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


def persistencia_batch(phi, contraste0_arr):
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
    v = phi.var(axis=-1) / (contraste0_arr ** 2)
    ok = (contraste0_arr > 0) & (phi.std(axis=-1) > 1e-12)
    return np.where(ok, c * v, 0.0)


def trayectoria_r_batched(N, eps, seed, pasos, r_list, D):
    rng = np.random.default_rng(seed)
    phi0, _ = base.campo_inicial(N, eps, rng)
    R = len(r_list)
    c0 = float(phi0.std())
    activo = np.ones((R, N), dtype=bool)
    phi = np.tile(phi0, (R, 1))
    if D > 0:
        H_arr = np.array([min(r * D, 1.0) for r in r_list]).reshape(R, 1)
    else:
        H_arr = np.array([0.0 if r == 0 else 1.0 for r in r_list]).reshape(R, 1)
    for _ in range(pasos):
        phi = paso_difusion_batch(phi, activo)
        u = rng.random(N)
        Hthr = np.clip(H_arr, 0.0, 1.0)
        cortar = activo & (u[None, :] < Hthr)
        activo = activo & ~cortar
    c0_arr = np.full(R, c0)
    P_real = persistencia_batch(phi, c0_arr)
    idx = rng.permutation(N)
    phi_null = phi[:, idx]
    P_null = persistencia_batch(phi_null, c0_arr)
    return P_real, P_null


# --- Caso de prueba: N=60 (chico para validar rápido), eps=0.02, seed=777 ---
N = 60
eps = 0.02
seed = 777
D = base.medir_D(N, eps, seed=999)  # D fijo de referencia para el caso de prueba
pasos = 150
r_list = [0.0, 0.3, 0.7, 1.0, 1.3, 2.0, 5.0, 50.0]

t0 = time.time()
P_real_batch, P_null_batch = trayectoria_r_batched(N, eps, seed, pasos, r_list, D)
t_batch = time.time() - t0

print(f"D={D:.6f} pasos={pasos}")
print(f"{'r':>6} {'H':>10} {'P_real_base':>14} {'P_real_batch':>14} {'diff_real':>12} "
      f"{'P_null_base':>14} {'P_null_batch':>14} {'diff_null':>12}")

max_diff_real = 0.0
max_diff_null = 0.0
for i, r in enumerate(r_list):
    H = float(min(r * D, 1.0)) if D > 0 else (0.0 if r == 0 else 1.0)
    ref_real = base.corrida(N, eps, H, pasos, seed=seed, null=False)["P"]
    ref_null = base.corrida(N, eps, H, pasos, seed=seed, null=True)["P"]
    dr = abs(ref_real - P_real_batch[i])
    dn = abs(ref_null - P_null_batch[i])
    max_diff_real = max(max_diff_real, dr)
    max_diff_null = max(max_diff_null, dn)
    print(f"{r:6.2f} {H:10.6f} {ref_real:14.10f} {P_real_batch[i]:14.10f} {dr:12.2e} "
          f"{ref_null:14.10f} {P_null_batch[i]:14.10f} {dn:12.2e}")

print(f"\nmax_diff_real={max_diff_real:.2e}  max_diff_null={max_diff_null:.2e}")
print(f"tiempo batched (los {len(r_list)} r juntos) = {t_batch:.3f}s")
print("VALIDACION:", "PASA (diffs ~0, identidad matemática confirmada)" if max(max_diff_real, max_diff_null) < 1e-9 else "FALLA -- revisar")
