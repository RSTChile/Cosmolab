#!/usr/bin/env python3
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


R_LIST = [
    0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85,
    0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0, 3.0,
    5.0, 7.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0,
]

for N, pasos in [(800, 97520), (1600, 390080)]:
    eps = 1e-3
    D = float(np.mean([base.medir_D(N, eps, s) for s in range(4)]))
    H_mid = [r for r in R_LIST if 0 < min(r * D, 1.0) < 1.0]
    R = len(H_mid)
    print(f"N={N} pasos={pasos} D={D:.3e} R_mid={R}/{len(R_LIST)}")

    rng_m = np.random.default_rng(42)
    phi0_m, _ = base.campo_inicial(N, eps, rng_m)
    activo_m = np.ones((R, N), dtype=bool)
    phi_m = np.tile(phi0_m, (R, 1))
    H_arr = np.array([min(r * D, 1.0) for r in H_mid]).reshape(R, 1)

    # benchmark solo unos pocos pasos y extrapolar (para no esperar horas aquí)
    n_bench = min(pasos, 300)
    t0 = time.time()
    for _ in range(n_bench):
        phi_m = paso_difusion_batch(phi_m, activo_m)
        u = rng_m.random(N)
        cortar = activo_m & (u[None, :] < H_arr)
        activo_m = activo_m & ~cortar
    dt = time.time() - t0
    per_step = dt / n_bench
    est_total_1_eps_seed = per_step * pasos
    n_eps_seed_units = 13 * 12  # eps x semillas (r ya está TODO batcheado en R filas)
    est_total_N = est_total_1_eps_seed * n_eps_seed_units
    print(f"  per_step={per_step*1000:.3f}ms  est 1 trayectoria(pasos completos, {R} r's)={est_total_1_eps_seed:.1f}s "
          f"({est_total_1_eps_seed/60:.2f}min)")
    print(f"  est TOTAL N={N} (13 eps x 12 seeds x {R} r's batched) = {est_total_N:.0f}s "
          f"= {est_total_N/60:.1f}min = {est_total_N/3600:.2f}h")
