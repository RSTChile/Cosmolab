def evolucionar(est, H, pasos, rng, null=False):
    tag0 = est["tag"].copy(); vivo0 = est["vivo"].copy()
    net0 = abs(int(tag0[vivo0].sum()))
    n_tag0 = int(vivo0.sum())
    for _ in range(pasos):
        paso_reabsorcion(est, rng)
        paso_expansion(est, H, rng)
    if null:
        _barajar_acople(est, rng) # <- solo una vez, al final
    return {"net0": net0, "n_tag0": n_tag0}