def persistencia(est, info):
    # P = fracción de masa que quedó en cierres con fase coherente
    # no conteo de vivos. Si un vivo quedó aislado (k=1) sin fase compatible,
    # no cuenta como persistencia estructurada.
    cierres = detectar_cierres(est)
    vivo = est["vivo"]
    # solo cierres con k>=2 y fase suma ~0 (neutralidad emergente de D2)
    masa_estructurada = sum(k*c for k,c in cierres.items() if k>=2)
    return masa_estructurada / max(info["n_tag0"], 1)