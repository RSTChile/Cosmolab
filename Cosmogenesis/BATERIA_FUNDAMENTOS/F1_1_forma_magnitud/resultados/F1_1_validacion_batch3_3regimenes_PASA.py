#!/usr/bin/env python3
"""Validación completa (3 regímenes: H<=0 'zero', H>=1 'full', 0<H<1 'mid')
del motor r-batched contra base.corrida(), incluyendo el caso NULL."""
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


def H_de_r(r, D):
    if D > 0:
        return float(min(r * D, 1.0))
    return 0.0 if r == 0 else 1.0


def trayectoria_regimen(N, eps, seed, pasos, r_list, D):
    r_zero = [r for r in r_list if H_de_r(r, D) <= 0.0]
    r_full = [r for r in r_list if H_de_r(r, D) >= 1.0]
    r_mid = [r for r in r_list if 0.0 < H_de_r(r, D) < 1.0]
    results = {}

    if r_zero:
        rng_z = np.random.default_rng(seed)
        phi0_z, _ = base.campo_inicial(N, eps, rng_z)
        activo_z = np.ones(N, dtype=bool)
        phi_z = phi0_z.copy()
        c0_z = float(phi0_z.std())
        for _ in range(pasos):
            phi_z = base.paso_difusion(phi_z, activo_z)
        P_real_z = base.persistencia(phi_z, c0_z)
        idx_z = rng_z.permutation(N)
        P_null_z = base.persistencia(phi_z[idx_z], c0_z)
        for r in r_zero:
            results[r] = (P_real_z, P_null_z)

    if r_full:
        rng_f = np.random.default_rng(seed)
        phi0_f, _ = base.campo_inicial(N, eps, rng_f)
        activo_init = np.ones(N, dtype=bool)
        phi_f = base.paso_difusion(phi0_f, activo_init)  # 1 solo paso, luego se congela
        c0_f = float(phi0_f.std())
        P_real_f = base.persistencia(phi_f, c0_f)
        idx_f = rng_f.permutation(N)
        P_null_f = base.persistencia(phi_f[idx_f], c0_f)
        for r in r_full:
            results[r] = (P_real_f, P_null_f)

    if r_mid:
        rng_m = np.random.default_rng(seed)
        phi0_m, _ = base.campo_inicial(N, eps, rng_m)
        R = len(r_mid)
        activo_m = np.ones((R, N), dtype=bool)
        phi_m = np.tile(phi0_m, (R, 1))
        c0_m = float(phi0_m.std())
        H_arr = np.array([H_de_r(r, D) for r in r_mid]).reshape(R, 1)
        for _ in range(pasos):
            phi_m = paso_difusion_batch(phi_m, activo_m)
            u = rng_m.random(N)
            cortar = activo_m & (u[None, :] < H_arr)
            activo_m = activo_m & ~cortar
        c0_arr = np.full(R, c0_m)
        P_real_m = persistencia_batch(phi_m, c0_arr)
        idx_m = rng_m.permutation(N)
        P_null_m = persistencia_batch(phi_m[:, idx_m], c0_arr)
        for i, r in enumerate(r_mid):
            results[r] = (float(P_real_m[i]), float(P_null_m[i]))

    return results


def correr_caso(nombre, N, eps, seed, pasos, r_list, D):
    print(f"\n--- caso: {nombre}  (N={N} eps={eps} seed={seed} pasos={pasos} D={D}) ---")
    t0 = time.time()
    res = trayectoria_regimen(N, eps, seed, pasos, r_list, D)
    t_batch = time.time() - t0
    max_dr = max_dn = 0.0
    for r in r_list:
        H = H_de_r(r, D)
        ref_real = base.corrida(N, eps, H, pasos, seed=seed, null=False)["P"]
        ref_null = base.corrida(N, eps, H, pasos, seed=seed, null=True)["P"]
        pr, pn = res[r]
        dr = abs(ref_real - pr)
        dn = abs(ref_null - pn)
        max_dr, max_dn = max(max_dr, dr), max(max_dn, dn)
        regimen = "zero" if H <= 0 else ("full" if H >= 1 else "mid")
        flag = "OK" if max(dr, dn) < 1e-9 else "*** DIFF ***"
        print(f"  r={r:7.2f} H={H:8.4f} [{regimen:4s}] real base={ref_real:.10f} batch={pr:.10f} d={dr:.2e} | "
              f"null base={ref_null:.10f} batch={pn:.10f} d={dn:.2e}  {flag}")
    print(f"  max_diff_real={max_dr:.2e}  max_diff_null={max_dn:.2e}  t_batch={t_batch:.3f}s")
    ok = max(max_dr, max_dn) < 1e-9
    print("  RESULTADO:", "PASA" if ok else "FALLA")
    return ok


ok1 = correr_caso("D moderado (mid+zero, sin full)", N=60, eps=0.02, seed=777, pasos=150,
                   r_list=[0.0, 0.3, 0.7, 1.0, 1.3, 2.0, 5.0, 50.0], D=0.009178)

ok2 = correr_caso("D grande (zero+mid+full mezclados)", N=60, eps=0.02, seed=321, pasos=150,
                   r_list=[0.0, 0.5, 1.0, 5.0, 10.0, 20.0, 60.0, 100.0], D=0.02)

ok3 = correr_caso("eps=0 (D=0 -> zero+full puro)", N=60, eps=0.0, seed=555, pasos=150,
                   r_list=[0.0, 0.001, 0.5, 1.0, 10.0, 100.0], D=0.0)

print("\n=== RESUMEN ===")
print("caso1:", "PASA" if ok1 else "FALLA")
print("caso2:", "PASA" if ok2 else "FALLA")
print("caso3:", "PASA" if ok3 else "FALLA")
print("TODO:", "PASA" if (ok1 and ok2 and ok3) else "FALLA -- no usar el motor batched hasta corregir")
