def persistencia(phi, activo, contraste0):
    if contraste0 <= 0: return 0.0
    # energía en gradientes que sobrevivieron en regiones aún acopladas
    # si el gradiente está entre dos regiones que la expansión cortó, está congelado y cuenta
    # si está dentro de una región conexa que aún difunde, se borrará y no cuenta
    grad = 0.0
    for i in range(len(phi)):
        if activo[i]: # arista viva entre i e i+1
            grad += (phi[(i+1)%len(phi)] - phi[i])**2
    # normaliza por contraste inicial: cuánto gradiente quedó vs cuánto había
    return float(np.sqrt(grad) / contraste0)