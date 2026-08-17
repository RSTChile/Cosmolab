# Reemplaza temperatura_fisica(frac) por:
def T_fisica(t_seg):
    # Ley radiación, anclada a 1e10 K a 1s, envuelta a 1e20 a 1e-20
    return 1e10 * (t_seg)**(-0.5)  # K

def t_fisico(paso, pasos):
    # paso 0 -> 1e-20, paso=pasos -> 1e-4, log-espaciado
    return 10**(np.log10(1e-20) + (np.log10(1e-4)-np.log10(1e-20))*paso/pasos)

# En barrido, para cada paso calculas T = T_fisica(t_fisico(paso))
# y frac_exp = 1 - activo.mean() sigue siendo lectura del estado, no motor