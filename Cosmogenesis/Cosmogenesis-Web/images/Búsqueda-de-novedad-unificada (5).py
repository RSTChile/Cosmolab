# dinámica real idéntica para ambos brazos
for _ in range(pasos):
    phi = paso_difusion(phi, activo)
    activo = paso_expansion(activo, H, rng)
if null:
    phi = rng.permutation(phi) # solo al final, solo una vez