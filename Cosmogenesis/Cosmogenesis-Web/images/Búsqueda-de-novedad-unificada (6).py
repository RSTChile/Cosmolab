def medir_D(N, eps, seed):
    rng = np.random.default_rng(seed)
    phi,_ = campo_inicial(N, eps, rng)
    activo = np.ones(N, dtype=bool)
    c0 = phi.std()
    for _ in range(3):
        phi = paso_difusion(phi, activo)
    c1 = phi.std()
    return max(0.0, (c0-c1)/c0 / 3)