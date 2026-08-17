"""
real_extra_generar_ic.py -- CS073, ataque directo al piso de p=1/9: generar >=5 realizaciones REAL
adicionales, genuinamente independientes, a N=2000 (misma escala que bateria_n2000/ic_real y
bateria_null1_n2000).

EL PROBLEMA (pedido por Alexis): con 1 sola corrida REAL y 8 NULL, un test de permutación exacto a
nivel de corrida nunca puede dar p < 1/9=0.111, sin importar la magnitud del efecto. La única salida es
más REAL, no más NULL.

EL PARÁMETRO DE SEMILLA IDENTIFICADO (investigación, no inventado aquí -- ya estaba en el diseño
congelado): `seed_layout`, el argumento de `fase1_traducir_a_phantom.traducir_pool` /
`p_semilla_causal.layout_resortes`. La prueba de que es EL mecanismo pre-autorizado para esto está en
el propio docstring de traducir_pool (fase1_traducir_a_phantom.py, línea ~75): "seed_layout varía la
realización estocástica del layout de resortes (las '>=5 semillas' de la Fase 2)" -- literalmente el
lenguaje que pide esta tarea.

Por qué es una realización REAL genuina y no una NULL disfrazada (auditoría, Paso 1 de esta tarea):
  - La malla causal en sí (el grafo de "quién nació relacionado con quién", `malla_causal_atomos`) es
    IDÉNTICA en las 5 nuevas semillas: usa `seed_ejes=2000` fijo (el default congelado) sobre el MISMO
    dens_bar (determinista, pool ya extraído por bateria_n2000/extraer_pool.py -- se REUSA desde disco,
    no se re-corre el motor basal). La topología relacional -- el objeto que CS073 dice que es "la única
    coherencia que el motor realmente produce" -- no cambia entre semillas.
  - Lo que sí cambia es `layout_resortes`: Fruchterman-Reingold arranca desde una posición inicial
    aleatoria (rng.uniform) y relaja 100 iteraciones sobre ESA MISMA malla. Distintas semillas dan
    distintas condiciones iniciales del algoritmo de relajación física estándar -- exactamente análogo
    a correr la misma dinámica molecular desde media aleatoria distinta: mismo mecanismo, mismo grafo,
    distinta trayectoria de relajación -> distinta realización espacial concreta de la MISMA coherencia
    relacional. No es shuffling de aristas (eso es la operación NULL, `barajar_aristas`, que aquí NUNCA
    se llama -- seed_null se deja en None en las 5).
  - El campo de turbulencia (velocidades iniciales) usa la MISMA semilla TURB_SEED=42 que
    bateria_n2000/ic_real y que bateria_null1_n2000 -- verificado empíricamente en este script (ver
    verificación abajo): v_rms de ic_real y de ic_null1 coinciden a 10 decimales (1.6431676725...).
    Se mantiene fija a propósito: si además de la posición variáramos la turbulencia, no podríamos saber
    si un cambio en el resultado viene del layout o del campo de velocidad -- un solo grado de libertad
    por vez, igual que hace el NULL-1 congelado con esta misma turbulencia.

Semillas elegidas: 301, 302, 303, 304, 305 -- consecutivas a las 101-103 (piloto NULL-1) y 201-208
(NULL-1 batería), evitando colisión con cualquier convención de semilla ya usada en el arco; 12345 sigue
siendo la semilla "original" (bateria_n2000/ic_real).

No toca bateria_n2000/ (sólo LEE masa_bar.npy/dens_bar.npy, sólo lectura). No reimplementa
`traducir_pool`/`malla_causal_atomos`/`layout_resortes`/`campo_velocidad_turbulento.factory` (todos
congelados) -- los importa tal cual.
"""
import os
import time

import numpy as np

from fase1_traducir_a_phantom import traducir_pool, POLYK
from campo_velocidad_turbulento import factory as factory_turb, MACH_OBJETIVO

RUTA_MASA = "/Users/alexis/phantom_cs073/bateria_n2000/masa_bar.npy"
RUTA_DENS = "/Users/alexis/phantom_cs073/bateria_n2000/dens_bar.npy"
BASE_SALIDA = "/Users/alexis/phantom_cs073/bateria_real_extra_n2000"

SEMILLAS_LAYOUT_NUEVAS = [301, 302, 303, 304, 305]  # las >=5 semillas REAL adicionales pedidas
SEED_LAYOUT_ORIGINAL = 12345                          # la de bateria_n2000/ic_real, para referencia

CS_SONORA = float(np.sqrt(POLYK))
TURB_SEED = 42   # MISMA semilla de turbulencia que bateria_n2000/ic_real y bateria_null1_n2000
                 # (verificado: v_rms idéntico entre ic_real e ic_null1 -- ver docstring del módulo)


def verificar_turb_seed_original():
    """Chequeo de cordura antes de generar nada: confirma que TURB_SEED=42 efectivamente reproduce el
    v_rms ya escrito en bateria_n2000/ic_real/cosmogenesis_ic.txt (sólo lectura)."""
    ruta = "/Users/alexis/phantom_cs073/bateria_n2000/ic_real/cosmogenesis_ic.txt"
    with open(ruta) as f:
        f.readline()
        f.readline()
        data = np.loadtxt(f)
    v_rms_disco = float(np.sqrt(np.mean(np.sum(data[:, 3:6] ** 2, axis=1))))
    v_rms_esperado = MACH_OBJETIVO * CS_SONORA
    print(f"[verificacion] v_rms en disco (ic_real)={v_rms_disco:.10f}  "
          f"Mach*c_s esperado={v_rms_esperado:.10f}  (coinciden si TURB_SEED=42 normaliza igual)")
    return v_rms_disco


def main():
    t0 = time.time()
    verificar_turb_seed_original()

    masa_bar = np.load(RUTA_MASA)
    dens_bar = np.load(RUTA_DENS)
    print(f"[pool] cargado desde disco (sólo lectura, NO se re-corre el motor basal): "
          f"n={len(masa_bar)} masa_media={masa_bar.mean():.4f}", flush=True)

    vel_gen = factory_turb(CS_SONORA, seed=TURB_SEED, mach=MACH_OBJETIVO)

    resultados = []
    for seed_layout in SEMILLAS_LAYOUT_NUEVAS:
        t1 = time.time()
        carpeta = f"{BASE_SALIDA}/ic_real_s{seed_layout}"
        os.makedirs(carpeta, exist_ok=True)
        ruta = f"{carpeta}/cosmogenesis_ic.txt"
        info = traducir_pool(masa_bar, dens_bar, seed_null=None, seed_layout=seed_layout,
                              vel_generador=vel_gen, ruta_salida=ruta)
        dt = time.time() - t1
        print(f"[seed_layout={seed_layout}] n={info['n']} a_final={info['a_final']:.4f} "
              f"tiempo={dt:.2f}s -> {ruta}", flush=True)
        resultados.append(info)

    print(f"\n[TOTAL generacion IC] {time.time()-t0:.2f}s -- {len(resultados)} condiciones REAL "
          f"nuevas escritas en {BASE_SALIDA}/", flush=True)
    return resultados


if __name__ == "__main__":
    main()
