def persistencia(phi, contraste0):
    # correlación a primer vecino: campo suave = 1, ruido blanco = 0
    corr = np.corrcoef(phi, np.roll(phi,1))[0,1]
    return max(0.0, corr) # std ya está en contraste0