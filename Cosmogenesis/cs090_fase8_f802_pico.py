#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs090_fase8_f802_pico.py — FASE VIII · F8-02: MANIPULAR A PROPÓSITO EL PICO LOCAL DE DENSIDAD INICIAL
=====================================================================================================

QUÉ PREGUNTA CONTESTA (a nivel módulo)
--------------------------------------
`FASE7_F705_mediacion_nueva_CS.md` dejó un único camino en pie después de condicionar por densidad
sobre 254 corridas: **el pico local de densidad del gas inicial** (`p90/mediana` de la densidad a 8
vecinos), con r parcial +0.64 a +0.90 en los 6 experimentos. Pero ese pico **nunca se manipuló**: se
midió después de los hechos, sobre condiciones iniciales que la topología del grafo había fabricado.
Un correlato medido post-hoc no distingue "el pico causa la masa" de "la topología causa las dos".

Este script fabrica la intervención que falta: **toma condiciones iniciales YA EXISTENTES y les mueve
el pico local, dejando todo lo demás igual** — mismo N, misma masa total (18800), misma caja, mismos
grados, mismas velocidades y, sobre todo, **el mismo grafo de origen** (no se regenera ni una arista).

Analogía: la receta del pan es la misma, el bollo es el mismo bollo, la misma harina y la misma agua.
Lo único que hacemos es meter los dedos en unos pocos grumos y **apretarlos o aflojarlos**, sin sacar
ni agregar harina. Si el pan que sale del horno cambia, cambió por el apretón.

CÓMO SE MUEVE EL PICO — la transformación radial elegida, y por qué ésta
------------------------------------------------------------------------
De las dos vías sugeridas (mapa radial suave / desplazamiento del percentil superior hacia su
centroide) se eligió **la radial suave**, por tres razones concretas:

  1. **Conserva exactamente N y la masa.** No mueve masa de una partícula a otra: mueve posiciones.
     La masa total es N × m_particula y ninguna de las dos cambia. La conservación no es "verificada
     a posteriori", es una identidad del método (igual se verifica, ver §VERIFICACIONES).
  2. **Es local de verdad y no toca el resto.** Cada burbuja tiene radio R; fuera de R **ninguna
     partícula se mueve, ni un dígito**. No es "casi intacto": es intacto, exactamente.
  3. **No deja escalón de densidad en el borde.** El mapa es C¹ en r=R (posición y derivada
     coinciden con la identidad), así que la nube no queda con una cáscara artificial de densidad.

El mapa, para una burbuja de centro c y radio R, con u = r/R y r = |x − c|:

        x' = c + (x − c) · g(u)          con      g(u) = 1 − a · (1 − u²)²

  * `a = 0`  → identidad exacta (nivel de control: el archivo sale **byte a byte** igual al original).
  * `a > 0`  → g < 1 dentro de la burbuja: el núcleo se **comprime** (el pico SUBE).
               Cerca del centro g → 1 − a, o sea el núcleo se encoge un factor (1−a) en cada eje:
               la densidad central se multiplica por ~(1−a)⁻³.
  * `a < 0`  → g > 1: el núcleo se **expande** (el pico BAJA).
  * En r = R: g = 1 y dg/dr = 0 → posición y densidad continuas; y como u·g(u) ≤ 1 para |a| ≤ 0.8,
    **ninguna partícula sale de su burbuja** (la caja no cambia, no hay fugas).
  * Es monótono (dr'/dr = 1 + a(1−u²)(5u²−1) > 0 para |a| ≤ 0.8): no hay inversión de orden radial,
    o sea las partículas no se cruzan entre sí en el radio.

Los **centros** son los máximos locales de densidad: se ordenan las partículas por densidad a 8
vecinos y se toman las más densas de a una, saltando las que caigan a menos de 2R de un centro ya
elegido. Así **las burbujas son disjuntas** (se verifica con un assert: ninguna partícula pertenece
a dos burbujas) y los mapas se componen sin ambigüedad.

Los centros y R se calculan UNA VEZ por condición inicial base, sobre la nube original, y se usan
**idénticos en todos los niveles**. Lo único que cambia entre niveles de una misma IC es el número `a`.
Es una familia de un solo parámetro que pasa por la identidad: el diseño pareado más limpio posible.

CALIBRACIÓN (hecha antes de comprometer la batería, sobre 2 IC)
---------------------------------------------------------------
R = 1.0 × separación media, 30 centros → se mueven ~750 de 2000 partículas (37%), desplazamiento
máximo ~1.5 en una caja de 97.6, y el pico p90/mediana recorre de ~0.85× a ~3× el original, mientras
la **geometría global** (masa en grumos FoF b=0.30, que F7-05 mostró que es la densidad disfrazada:
r = −0.9945 con el grado medio) se mueve menos de **±0.5%**. Es decir: el eje 2 de F7-05 se mueve
mucho y el eje 1 se queda quieto. Esa es exactamente la disección que la tarea pide.

Los cinco niveles (`a` = −0.35, 0.00, +0.20, +0.35, +0.50) se eligieron en la calibración para que el
pico **logrado** recorra aproximadamente el rango del corpus (5.6 a 34 en las 254 corridas) sin salirse
de él. AVISO honesto que la calibración ya mostró y el CSV va a confirmar: **bajar el pico es mucho
más difícil que subirlo** — aflojar un grumo baja el p90 pero también baja la mediana, y el cociente
se mueve poco (y con `a` muy negativo puede hasta subir). Por eso la escalera es asimétrica y por eso
la monotonía se juzga contra el pico **logrado**, nunca contra `a`.

QUÉ SE MANTIENE IDÉNTICO ENTRE NIVELES (lo que hace válido el pareado)
-----------------------------------------------------------------------
  · el grafo de origen        — no se regenera: se copia el `.grafo.gz` de F8-00 y se verifica el sello
  · N = 2000 y la masa por partícula (9.4) → masa total 18800 exacta
  · la caja y el `seed_layout`
  · **las velocidades**, partícula por partícula (el campo turbulento se hereda tal cual: así la
    energía cinética y el momento total son idénticos entre niveles al último bit)
  · el `h` inicial, `hfact`, `polyk`, y la cabecera del archivo, copiada verbatim

SALIDAS
-------
  /Users/alexis/phantom_cs073/bateria_fase8_f802_pico/<rule_id>_s<seed>_f802_L<k>/
        cosmogenesis_ic.txt   meta_regla.json   grafo_f802.grafo.gz   transformacion_f802.json
  cs090_fase8_f802_ic_transformadas.csv     una fila por (IC base × nivel): lo pedido y lo logrado

QUÉ NO HACE
-----------
No corre Phantom (eso es `cs090_fase8_f802_correr.py`). No genera ni altera ningún grafo. No modifica
ningún script, CSV ni carpeta existente. No declara cierre ni veredicto.

USO
---
    ./venv/bin/python cs090_fase8_f802_pico.py            # las 12 IC base × 5 niveles
    ./venv/bin/python cs090_fase8_f802_pico.py --limite 1 # sólo la primera IC (piloto cronometrado)
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

AQUI = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, AQUI)

import cs090_fase8_f800_grafos as G8                       # persistencia/sello de grafos (sólo import)
from cs090_fase6_o4a_observable_comun import fof_masa      # misma vara FoF de O4-A/F7-05 (sólo import)

RAIZ_PHANTOM = "/Users/alexis/phantom_cs073"
BASE_SALIDA = f"{RAIZ_PHANTOM}/bateria_fase8_f802_pico"
RUTA_DATASET = f"{AQUI}/cs090_fase8_f800_dataset_enriquecido.csv"
RUTA_CSV = f"{AQUI}/cs090_fase8_f802_ic_transformadas.csv"

N_IC_BASE = 12
EXP_FUENTE = "F5B_40pares"      # un solo experimento: F7-05 mostró que mezclar diseños trae Simpson
K_VECINOS = 8                   # la misma k que `geoIC_knn8_p90_med` de F7-05
R_EN_SEPARACIONES = 1.0         # radio de burbuja, en unidades de la separación media
N_CENTROS = 30                  # máximos locales de densidad que se aprietan/aflojan
MASA_MIN_FOF = 47.0             # 5 partículas de N=2000 (criterio en masa física de O3-A/F7-05)

# los cinco niveles: L1 (a=0) es el control interno y sale idéntico al original
NIVELES = [("L0", -0.35), ("L1", 0.00), ("L2", +0.20), ("L3", +0.35), ("L4", +0.50)]


# =============================================================================================
# 1) LECTURA / ESCRITURA DE LAS CONDICIONES INICIALES (formato `cosmogenesis_ic v2`)
# =============================================================================================
def leer_ic_completa(ruta):
    """Devuelve (cabecera, linea2, datos) con `datos` = array (N,7): x y z vx vy vz h.

    Se leen las DOS primeras líneas como texto crudo para poder devolverlas verbatim: la cabecera
    lleva metadatos (n_aristas, seed_layout, masa_total_objetivo) que otros scripts de la línea leen,
    y la línea 2 es la que consume `phantomsetup`. No se reescriben: se copian."""
    with open(ruta) as fh:
        cabecera = fh.readline()
        linea2 = fh.readline()
    datos = np.loadtxt(ruta, skiprows=2)
    assert datos.ndim == 2 and datos.shape[1] == 7, f"{ruta}: se esperaban 7 columnas"
    return cabecera, linea2, datos


def escribir_ic(ruta, cabecera, linea2, datos):
    """Escribe con EXACTAMENTE el mismo formato que `generar_ic_masa_fija_desde_grafo` (%.17g), para
    que el nivel de identidad (a=0) reproduzca el archivo original byte a byte."""
    with open(ruta, "w") as f:
        f.write(cabecera)
        f.write(linea2)
        for fila in datos:
            f.write(" ".join(f"{float(v):.17g}" for v in fila) + "\n")


def md5(ruta):
    h = hashlib.md5()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


# =============================================================================================
# 2) LA VARA: densidad local a 8 vecinos, pico y CV — IDÉNTICA a la de F7-05
# =============================================================================================
def densidad_knn(pos, k=K_VECINOS):
    """Densidad local ~ k / r_k³ con r_k la distancia al k-ésimo vecino. Misma definición que
    `cs090_fase7_f705_geometria_ic_todas.descriptores_knn` (se reescribe acá, en vez de importarla,
    porque también se necesita el vector completo para elegir los centros)."""
    arbol = cKDTree(pos)
    dist, _ = arbol.query(pos, k=k + 1)
    return k / (dist[:, k] ** 3 + 1e-300)


def pico_y_cv(pos):
    rho = densidad_knn(pos)
    return float(np.percentile(rho, 90) / np.median(rho)), float(rho.std() / rho.mean())


# =============================================================================================
# 3) LA TRANSFORMACIÓN
# =============================================================================================
def elegir_centros(pos, R, n_centros=N_CENTROS):
    """Máximos locales de densidad, separados entre sí por al menos 2R (burbujas disjuntas).
    Se recorre la lista ordenada por densidad descendente y se descartan los candidatos que caen
    dentro de la burbuja de un centro ya aceptado."""
    rho = densidad_knn(pos)
    sel = []
    for i in np.argsort(-rho):
        if len(sel) >= n_centros:
            break
        if sel and np.min(np.linalg.norm(pos[sel] - pos[i], axis=1)) < 2.0 * R:
            continue
        sel.append(int(i))
    return np.array(sel, dtype=int)


def transformar(pos, a, centros, R):
    """Aplica x' = c + (x−c)·(1 − a(1−u²)²) dentro de cada burbuja. Devuelve (pos_nueva, tocadas).

    Con a = 0 NO se hace la cuenta: se devuelve la posición original tal cual. No es una optimización,
    es una exigencia de exactitud — en coma flotante `c + (x − c)` puede no dar exactamente `x`, y el
    nivel de identidad tiene que salir byte a byte igual al archivo original para servir de control."""
    salida = pos.copy()
    arbol = cKDTree(pos)
    tocadas = np.zeros(len(pos), dtype=bool)
    for c in centros:
        idx = np.array(arbol.query_ball_point(pos[c], R), dtype=int)
        assert not tocadas[idx].any(), "dos burbujas se solapan: los centros no quedaron disjuntos"
        tocadas[idx] = True
        if a == 0.0:
            continue
        d = pos[idx] - pos[c]
        u = np.linalg.norm(d, axis=1) / R
        salida[idx] = pos[c] + d * (1.0 - a * (1.0 - u ** 2) ** 2)[:, None]
    return salida, tocadas


# =============================================================================================
# 4) SELECCIÓN DE LAS 12 CONDICIONES INICIALES BASE
# =============================================================================================
def seleccionar_base(n=N_IC_BASE):
    """12 corridas de `F5B_40pares` (N=2000), espaciadas por cuantiles del pico local ya medido en
    F7-05/F8-00, para cubrir el rango del observable. Un solo experimento a propósito: F7-05 §5.1
    cazó una paradoja de Simpson por mezclar resoluciones/diseños. Se exige que existan en disco
    la condición inicial Y el grafo guardado por F8-00 (con su sello)."""
    D = pd.read_csv(RUTA_DATASET)
    d = D[(D.exp == EXP_FUENTE) & (D.N_nodos == 2000)].copy()

    def ruta_carpeta(r):
        c = str(r["carpeta"])
        return c if c.startswith("/") else os.path.join(RAIZ_PHANTOM, str(r["bateria_raiz"]), c)

    d["ruta_base"] = d.apply(ruta_carpeta, axis=1)
    d["ic"] = d.ruta_base + "/cosmogenesis_ic.txt"
    d = d[d.ic.apply(os.path.exists) & d.f800_archivo.apply(lambda p: isinstance(p, str) and os.path.exists(p))]
    d = d.sort_values("geoIC_knn8_p90_med").reset_index(drop=True)
    idx = np.linspace(0, len(d) - 1, n).round().astype(int)
    return d.iloc[idx].reset_index(drop=True)


# =============================================================================================
# 5) UNA CONDICIÓN INICIAL BASE → SUS CINCO NIVELES
# =============================================================================================
def procesar_una(base):
    cab, linea2, datos = leer_ic_completa(base["ic"])
    pos0 = datos[:, :3].copy()
    N = len(pos0)
    m_part = float(linea2.split()[1])
    lado = float(np.max(pos0.max(0) - pos0.min(0)))
    sep = lado / N ** (1.0 / 3.0)
    R = R_EN_SEPARACIONES * sep
    masas = np.full(N, m_part)

    centros = elegir_centros(pos0, R)
    pico0, cv0 = pico_y_cv(pos0)
    fof0, _ = fof_masa(pos0, masas, 0.30 * sep, MASA_MIN_FOF)

    # el grafo: se copia el que F8-00 dejó sellado, NO se regenera nada
    adj_ref, N_ref, meta_g = G8.cargar_grafo(base["f800_archivo"])       # verifica el sello al leer
    assert N_ref == N, f"{base['rule_id']}: el grafo guardado tiene N={N_ref} y la IC N={N}"
    sello_grafo = meta_g["sha256"]

    filas = []
    for nombre_nivel, a in NIVELES:
        pos, tocadas = transformar(pos0, a, centros, R)   # con a=0 devuelve pos0 sin tocar un dígito
        datos_n = datos.copy()
        datos_n[:, :3] = pos                                     # velocidades y h: intactas

        carpeta = f"{BASE_SALIDA}/{base['rule_id']}_s{int(base['seed'])}_f802_{nombre_nivel}"
        os.makedirs(carpeta, exist_ok=True)
        ruta_ic = f"{carpeta}/cosmogenesis_ic.txt"
        escribir_ic(ruta_ic, cab, linea2, datos_n)

        # -------- VERIFICACIONES sobre lo ESCRITO (se relee del disco, no se confía en memoria) ----
        cab2, l22, datos_leidos = leer_ic_completa(ruta_ic)
        pos_leida = datos_leidos[:, :3]
        assert len(datos_leidos) == N, "cambió el nº de partículas"
        assert np.array_equal(datos_leidos[:, 3:], datos[:, 3:]), "cambiaron velocidades o h"
        assert cab2 == cab and l22 == linea2, "cambió la cabecera"
        ida_vuelta = bool(np.array_equal(pos_leida, pos))
        md5_igual_al_original = (md5(ruta_ic) == md5(base["ic"]))

        pico, cv = pico_y_cv(pos_leida)
        fof, ngr = fof_masa(pos_leida, masas, 0.30 * sep, MASA_MIN_FOF)
        desp = np.linalg.norm(pos_leida - pos0, axis=1)
        caja = pos_leida.max(0) - pos_leida.min(0)

        # copia del grafo (idéntico en los 5 niveles: es el punto del experimento) + sello releído
        shutil.copy(base["f800_archivo"], f"{carpeta}/grafo_f802.grafo.gz")
        _, _, meta_copia = G8.cargar_grafo(f"{carpeta}/grafo_f802.grafo.gz")
        assert meta_copia["sha256"] == sello_grafo, "el grafo copiado no tiene el mismo sello"

        meta = dict(
            tarea="FASE8_F802_pico_local_manipulado", nivel=nombre_nivel, a_pico=a,
            rule_id=str(base["rule_id"]), seed=int(base["seed"]), clase=str(base["clase"]),
            N=N, n_niveles=len(NIVELES), carpeta=carpeta,
            carpeta_ic_original=str(base["ruta_base"]),
            grafo_archivo="grafo_f802.grafo.gz", grafo_sha256=sello_grafo,
            grafo_n_aristas=int(meta_g["E"]),
            n_aristas_grafo_final=int(base["n_aristas"]), grado_medio_grafo_final=float(base["grado_medio"]),
            kcap=(None if pd.isna(base.get("kcap")) else int(base["kcap"])),
            K=(None if pd.isna(base.get("K")) else int(base["K"])),
            masa_particula=m_part, masa_total_ic=m_part * N,
            radio_burbuja=R, n_centros=int(len(centros)),
            pico_p90_med_original=pico0, pico_p90_med=pico,
            frac_masa_historica_original=float(base["frac_masa"]),
        )
        with open(f"{carpeta}/meta_regla.json", "w") as f:
            json.dump(meta, f, indent=2)
        with open(f"{carpeta}/transformacion_f802.json", "w") as f:
            json.dump(dict(mapa="x' = c + (x-c)*(1 - a*(1-u^2)^2), u=|x-c|/R", a=a, R=R,
                           centros=[int(c) for c in centros], k_vecinos=K_VECINOS,
                           sep_media=sep, lado_nube_original=lado), f, indent=2)

        filas.append(dict(
            rule_id=str(base["rule_id"]), seed=int(base["seed"]), clase=str(base["clase"]),
            nivel=nombre_nivel, a_pico=a, carpeta=carpeta,
            N=N, masa_particula=m_part, masa_total=m_part * N,
            n_aristas=int(base["n_aristas"]), grado_medio=float(base["grado_medio"]),
            grafo_sha256=sello_grafo,
            radio_burbuja=R, sep_media=sep, n_centros=int(len(centros)),
            pico_original=pico0, pico_logrado=pico, pico_razon=pico / pico0,
            cv_original=cv0, cv_logrado=cv,
            fof030_original=fof0, fof030_logrado=fof, fof030_razon=fof / fof0, n_grupos_fof=ngr,
            n_particulas_movidas=int(tocadas.sum()), frac_movidas=float(tocadas.mean()),
            desp_max=float(desp.max()), desp_medio_movidas=float(desp[tocadas].mean()) if tocadas.any() else 0.0,
            caja_x=float(caja[0]), caja_y=float(caja[1]), caja_z=float(caja[2]),
            lado_original=lado,
            ic_ida_vuelta_ok=ida_vuelta, ic_md5_igual_al_original=md5_igual_al_original,
            frac_masa_historica=float(base["frac_masa"]),
            kappa_v_historica=float(base["kappa_v"]) if not pd.isna(base["kappa_v"]) else float("nan"),
            ic_original=str(base["ic"]),
        ))
        print(f"    {nombre_nivel} a={a:+.2f}  pico {pico0:6.3f} -> {pico:8.3f} (x{pico/pico0:5.2f})  "
              f"cv {cv0:.2f}->{cv:.2f}  fof030 {fof0:.4f}->{fof:.4f}  "
              f"movidas={int(tocadas.sum())} despmax={desp.max():.2f}  "
              f"masa={m_part*N:.1f}  md5_igual={md5_igual_al_original}", flush=True)
    return filas


def main(limite=None):
    os.makedirs(BASE_SALIDA, exist_ok=True)
    base = seleccionar_base()
    if limite:
        base = base.iloc[:limite]
    print(f"[f802] {len(base)} condiciones iniciales base de {EXP_FUENTE}, "
          f"pico original {base.geoIC_knn8_p90_med.min():.2f} a {base.geoIC_knn8_p90_med.max():.2f}; "
          f"{len(NIVELES)} niveles cada una -> {len(base)*len(NIVELES)} corridas", flush=True)

    t0, todas = time.time(), []
    for k, (_, b) in enumerate(base.iterrows(), 1):
        print(f"[{k}/{len(base)}] {b['rule_id']} s{int(b['seed'])} clase={b['clase']} "
              f"pico_csv={b['geoIC_knn8_p90_med']:.3f} frac_masa_hist={b['frac_masa']:.4f}", flush=True)
        todas.extend(procesar_una(b))

    D = pd.DataFrame(todas)
    if limite:      # el piloto no pisa el CSV de la batería completa
        D.to_csv(RUTA_CSV.replace(".csv", "_piloto.csv"), index=False)
    else:
        D.to_csv(RUTA_CSV, index=False)
    print(f"\n[f802] {len(D)} filas escritas ({time.time()-t0:.0f}s)")
    print(f"  masa total: {D.masa_total.nunique()} valor(es) distinto(s) -> {sorted(D.masa_total.unique())}")
    print(f"  N: {sorted(D.N.unique())}   ida y vuelta OK: {D.ic_ida_vuelta_ok.all()}")
    print(f"  nivel identidad (L1) idéntico al original byte a byte: "
          f"{D[D.nivel=='L1'].ic_md5_igual_al_original.all()}")
    print(f"  pico logrado: x{D.pico_razon.min():.2f} a x{D.pico_razon.max():.2f}   "
          f"FoF global (eje-1): x{D.fof030_razon.min():.4f} a x{D.fof030_razon.max():.4f}")
    return D


if __name__ == "__main__":
    lim = None
    for arg in sys.argv[1:]:
        if arg.startswith("--limite"):
            lim = int(arg.split("=")[1]) if "=" in arg else int(sys.argv[sys.argv.index(arg) + 1])
    main(limite=lim)
