"""
fase1_traducir_a_phantom.py — FASE 1: traducir nuestro sustrato a condiciones iniciales de Phantom.

EL ÚNICO LUGAR DONDE PUEDE COLARSE SHANNON (INSTRUCCION_CC_phantom_PARA_CC.md). Por eso esta función
es deliberadamente CORTA y hace UNA sola cosa: llama a las piezas YA VALIDADAS y auditadas del resto
del arco (nunca reimplementa nada), y el ÚNICO parámetro que distingue REAL de NULL es `seed_null`.
Todo lo demás -- masa, N, parámetros de Phantom, semilla del layout de resortes -- es IDÉNTICO.

Qué traduce (literal, según INSTRUCCION_CC_phantom_PARA_CC.md Fase 1):
  - Posiciones 3D: del despliegue dinámico de la malla causal (p_semilla_causal.py) -- REAL = malla
    causal real; NULL = MISMA malla con aristas barajadas (barajar_aristas, preserva grado/peso).
    CORRECCIÓN 20-jul (INSTRUCCION_CC_expansion_en_layout_PARA_CC.md): layout_resortes por sí solo
    entrega la foto MÁS comprimida de todo el arco -- el estado justo antes de que un solo paso de
    expansión actúe -- mientras que _dinamica_estructura/_dinamica_ignicion (el puente, z=6.92) SIEMPRE
    trabajan ya diluidas por el reloj de expansión. Se aplica aquí la MISMA dilatación isótropa
    (p_expansion.Expansion, mismo T0/TASA_EXPANSION/reloj que el resto del arco, mismos 60 "pasos
    cosmológicos" que n_pasos_estructura por defecto), pero como un factor ESTÁTICO único -- SIN
    interleavar nuestra propia gravedad (eso lo hace Phantom desde el IC; aplicarla dos veces
    contaminaría justo lo que Phantom vino a hacer limpio). a_final=√60≈7.75, idéntico en REAL y NULL
    (depende sólo de N/reloj, nunca de seed_null/seed_layout).
  - Masa: la masa REAL de los átomos H (masa_trio, ~9.4, casi uniforme -- NO se pesa por #23; pesarla
    sería inyectar la señal de #23 por la puerta de atrás, cuando lo que este experimento prueba es si
    la POSICIÓN por sí sola -- la coherencia relacional -- ya alcanza. Mismo criterio que el control
    positivo ya corrido).
  - Velocidades: cero por DEFECTO (arranque en reposo -- la misma convención que TODAS las corridas
    N-cuerpo de este arco: el "colapso frío"). CORRECCIÓN 20-jul (INSTRUCCION_CC_velocidad_inicial_
    PARA_CC.md): el chequeo de conservación de Phantom abortaba con L_inicial=0 EXACTO (v=0 en todas
    las partículas por construcción -- ver checkconserved.f90, confirmado artefacto de diagnóstico, no
    falla física). Las nubes reales nunca están en reposo -- se añade el parámetro opcional
    `vel_generador(pos, adj, dens_bar) -> vel` (ver campo_velocidad_turbulento.py y
    campo_velocidad_heredado.py) para los dos brazos de campo de velocidades inicial. Con
    vel_generador=None (default) el comportamiento es IDÉNTICO al de siempre (vel=0) -- no rompe nada
    de lo ya corrido/validado.
  - h inicial: UNIFORME (hfact * espaciado medio global) para TODAS las partículas -- adjudicación CS
    20-jul (INSTRUCCION_CC_hlocal_PARA_CC.md): el k=6-vecino-más-cercano daba un h DIMINUTO justo para
    los pares casi-coincidentes (la cola de la estructura jerárquica, presente por igual en REAL y NULL,
    ver histograma "sin hueco"), y ESO reventaba la iteración de velocidad del leapfrog -- no la física.
    Un arranque uniforme, simple, sin favorecer ninguna partícula, deja que el solver grad-h NATIVO de
    Phantom (Fase 0: 12/12 gravedad, órbita a 2.2e-14) encuentre la densidad/h local real de cada
    partícula por sí solo, en vez de heredar una condición inicial mal armada por nuestro lado.
  - Unidades: G=1 código (Phantom soporta unidades de código G=1, igual que TODO nuestro motor
    adimensional -- NO hay factor de conversión elegido; es la MISMA convención en ambos lados).

PROHIBIDO (y no lo hace): tocar posiciones a mano, sembrar sobredensidades, usar parámetros de Phantom
distintos entre REAL y NULL, pesar la masa por #23.

Salida: un archivo ASCII plano (columnas x y z vx vy vz h), con un header de una línea (npart, masa por
partícula, hfact) -- el formato que lee `setup_cosmogenesis.f90` (Phantom, SETUP=cosmogenesis).
"""
import numpy as np

from cs073_cierre_holistico import _extraer_bariones, T0, TASA_EXPANSION, _T_reloj
from cs072_modulos.piezas.p_expansion import Expansion
from cs072_modulos.piezas.p_enfriamiento_H2 import EnfriamientoH2
from cs072_modulos.piezas.p_semilla_causal import malla_causal_atomos, layout_resortes, barajar_aristas

HFACT = 1.2   # convención SPH estándar (== hfact_default de Phantom, verificado en kernel_cubic.f90)

# polyk = c_s^2 ISOTÉRMICO, DERIVADO del propio motor -- adjudicación CS 20-jul: el piso de enfriamiento
# de EnfriamientoH2 (T_piso = 0.1*T0, ya establecido, no inventado para Fase 2) es la temperatura que la
# física de H2 produce antes del colapso. Se lee del objeto real (no se duplica la fórmula a mano) para
# que nunca pueda desincronizarse si T_piso cambia en su módulo. MISMO valor en REAL y NULL por
# construcción (no depende de seed_null) -- fija el umbral de Jeans simétricamente en ambos brazos.
POLYK = EnfriamientoH2(n=1, T_inicial=T0).T_piso


def traducir_pool(masa_bar, dens_bar, D_causal=3, k_causal=4, iters_layout=100,
                   seed_null=None, seed_layout=12345, n_pasos_expansion=60,
                   vel_generador=None, ruta_salida="cosmogenesis_ic.txt"):
    """Misma traducción, pero recibe masa_bar/dens_bar YA EXTRAÍDOS -- el motor basal es determinista y
    no depende de seed_null/seed_layout, así que re-correrlo por cada uno de los 5+8 archivos de Fase 2
    desperdiciaría ~10 minutos por archivo sin ganar nada (mismo criterio que _dinamica_estructura en
    cs073_cierre_holistico.py). seed_null=None -> REAL. seed_null=<entero> -> NULL (aristas barajadas,
    LA ÚNICA diferencia con REAL). seed_layout varía la realización estocástica del layout de resortes
    (las ">=5 semillas" de la Fase 2), ortogonal a seed_null, misma semántica en ambos brazos.

    n_pasos_expansion=60: mismo default que n_pasos_estructura en _dinamica_estructura -- el a_final que
    resulta (√60≈7.75) es EL MISMO que ya midió el puente (z=6.92), no un valor nuevo elegido para Phantom."""
    n = len(masa_bar)
    if n < 8:
        raise ValueError(f"sólo {n} átomos reales (<8): sin masa suficiente para traducir")

    lado = float(n) ** (1.0 / 3.0)
    adj, _m = malla_causal_atomos(dens_bar, D=D_causal, k=k_causal, seed_ejes=2000)
    if seed_null is not None:
        adj = barajar_aristas(adj, n, seed=seed_null)   # <-- LA ÚNICA DIFERENCIA REAL vs NULL
    pos = layout_resortes(adj, n, lado=lado, iters=iters_layout, seed=seed_layout)

    # dilatación isótropa ESTÁTICA (INSTRUCCION_CC_expansion_en_layout_PARA_CC.md, adjudicación CS
    # 20-jul, "opción 1"): reusa el mecanismo EXACTO de p_expansion.Expansion (mismo T0/TASA_EXPANSION/
    # reloj _T_reloj que _dinamica_estructura), acumulado sobre n_pasos_expansion pasos -- SIN
    # interleavar gravedad propia (Phantom la hace desde el IC; aplicarla aquí también sería aplicarla
    # dos veces). Depende sólo de N (vía lado) y del reloj -- idéntico en REAL y NULL por construcción.
    expansion = Expansion(T0=T0)
    for step in range(n_pasos_expansion):
        expansion.paso_de_estiramiento(_T_reloj(step))
    a_final = expansion._a_prev
    pos = pos * a_final

    # vel_generador recibe el MISMO adj que ya se usó para pos (post-barajado si seed_null!=None) -- así
    # el brazo "heredado" ve exactamente la topología REAL o NULL que corresponde, sin recalcular nada.
    vel = vel_generador(pos, adj, dens_bar) if vel_generador is not None else np.zeros_like(pos)

    # h inicial UNIFORME (adjudicación CS 20-jul, ver docstring del módulo) -- espaciado medio global=1.0
    # por construcción (lado=n**(1/3)); NINGUNA partícula parte de un h sembrado por su vecino más cercano.
    # Phantom resuelve la densidad/h local real desde este arranque parejo (grad-h nativo, Fase 0).
    h_guess = np.full(n, HFACT)

    masa_media = float(masa_bar.mean())   # masa H real, ~9.4, casi uniforme (ver docstring del módulo)

    with open(ruta_salida, "w") as f:
        f.write(f"# cosmogenesis_ic v2 -- npart={n} masa_particula={masa_media:.17g} hfact={HFACT} "
                 f"polyk={POLYK:.17g} (=T_piso de EnfriamientoH2, derivado, no elegido)\n")
        f.write(f"{n} {masa_media:.17g} {HFACT} {POLYK:.17g}\n")
        for i in range(n):
            f.write(f"{float(pos[i,0]):.17g} {float(pos[i,1]):.17g} {float(pos[i,2]):.17g} "
                     f"{float(vel[i,0]):.17g} {float(vel[i,1]):.17g} {float(vel[i,2]):.17g} "
                     f"{float(h_guess[i]):.17g}\n")

    return dict(ruta=ruta_salida, n=n, masa_particula=masa_media, lado=lado, polyk=POLYK,
                seed_null=seed_null, seed_layout=seed_layout, a_final=a_final)


def escribir_ic_phantom(nq, naq, ne, npos, pasos_basal=150, amp_rugosidad=1.5,
                         D_causal=3, k_causal=4, iters_layout=100,
                         seed_null=None, seed_layout=12345, ruta_salida="cosmogenesis_ic.txt"):
    """Conveniencia (re-extrae el pool cada vez -- para usos sueltos/humo, NO para la batería de Fase 2;
    ahí usar traducir_pool con un pool ya extraído una sola vez)."""
    masa_bar, dens_bar, obs_basal = _extraer_bariones(nq, naq, ne, npos, pasos_basal, amp_rugosidad)
    info = traducir_pool(masa_bar, dens_bar, D_causal=D_causal, k_causal=k_causal,
                          iters_layout=iters_layout, seed_null=seed_null, seed_layout=seed_layout,
                          ruta_salida=ruta_salida)
    info["obs_basal"] = dict(hidrogeno=obs_basal.get("hidrogeno"), helio=obs_basal.get("helio"))
    return info


if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "real" else None
    ruta = sys.argv[2] if len(sys.argv) > 2 else "cosmogenesis_ic.txt"
    info = escribir_ic_phantom(nq=300, naq=210, ne=100, npos=70, seed_null=seed, ruta_salida=ruta)
    print(info)
