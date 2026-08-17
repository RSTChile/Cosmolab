#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E5.6-5 — Exergía informacional (X_info) y una medida independiente de I
=========================================================================

Pregunta (BATERIA_ENFOQUE5, TEMA 6): ¿la exergía informacional se conecta con la I
(información/estructura) de la ley central S=I·E, y de qué forma? NO se fuerza la forma
S=I·E — solo se mide la relación empírica X_info<->I contra NULL.

Ver protocolo (definiciones exactas, fijadas ANTES de este motor):
  E5_6_5_PROTOCOLO_PREREGISTRO.md

Reutiliza SIN EDITAR la física de cs074_rcruz.py (import directo):
  campo_inicial, paso_difusion, paso_expansion, medir_D, medir_pasos_lavado
No se toca cs074_rcruz.py.

Dos observables, dos métodos matemáticos DISTINTOS (anti-T2), sobre el MISMO estado final φ:

  X_info (exergía informacional, método A — bloques espaciales + histograma de valores):
    1. Se parte el anillo en B=32 bloques contiguos, se promedia cada uno.
    2. Se discretizan las B medias en nbins=8 bins (rango = min/max del propio φ final).
    3. X_info = H_shannon(bins) / log2(nbins)  ∈[0,1].

  I (entropía estructural, método B — entropía de permutación ordinal Bandt-Pompe):
    1. Ventanas deslizantes de m=4 puntos consecutivos (anillo cerrado).
    2. Patrón ordinal (orden relativo, desempate estable por índice) de cada ventana.
    3. I = H_shannon(24 patrones) / log2(24)  ∈[0,1].

NULL = barajar (permutar) el φ final de la MISMA corrida antes de medir (igual convención
que cs074_rcruz.py). Se mide todo: X_info_real, I_real, X_info_null, I_null.

Salida: E5_6_5_resultado.json (crudo, sin adjudicar).
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE
BASE_CODE = HERE.parent.parent / "cs074_rcruz.py"

# --- import sin editar cs074_rcruz.py ---
spec = importlib.util.spec_from_file_location("cs074_rcruz_base", str(BASE_CODE))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)  # type: ignore

campo_inicial = base.campo_inicial
paso_difusion = base.paso_difusion
paso_expansion = base.paso_expansion
medir_D = base.medir_D
medir_pasos_lavado = base.medir_pasos_lavado

# ============================== parámetros pre-registrados ==============================
N = 256
B_BLOQUES = 32          # bloques espaciales contiguos para X_info
NBINS_XINFO = 8         # bins del histograma de medias de bloque
M_PERM = 4              # dimensión de embebido para entropía de permutación (I)
LOG2_NBINS = np.log2(NBINS_XINFO)
LOG2_MFACT = np.log2(math.factorial(M_PERM))  # log2(24)

EPS_LIST = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 3e-1, 1.0]
R_TARGETS = [0.0, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
SEMILLAS = 16
CAL_EPS_REF = 1e-3   # eps de referencia para calibrar pasos (único, no por-eps: evita recalibrar 9x)
CAL_SEMILLAS = 16


def evolucionar_sin_null(phi, activo, H, pasos, rng):
    """Igual que base.evolucionar pero SIN aplicar el barajado interno: aquí controlamos
    manualmente cuándo medir real vs null sobre el MISMO estado final (evita tener que
    correr la física dos veces con semillas distintas)."""
    for _ in range(pasos):
        phi = paso_difusion(phi, activo)
        activo = paso_expansion(activo, H, rng)
    return phi, activo


def x_info_bloques(phi, B=B_BLOQUES, nbins=NBINS_XINFO):
    """Método A: entropía de Shannon normalizada del histograma de medias de bloque
    espaciales contiguos (usa la topología de anillo -> sensible a barajado).

    IMPORTANTE: el rango de binning se fija con min/max del phi RAW completo (no de las
    medias de bloque). Esto es lo que pre-registra el protocolo, y es lo que hace que la
    medida sea sensible al barajado: phi_null es una PERMUTACIÓN de phi_real (mismo
    multiset de valores) -> mismo rango crudo, mismos bordes de bin, en ambos casos. Si en
    cambio se usara el rango de las MEDIAS DE BLOQUE, éste se autoescala a como sea que
    caigan esas medias y borra la señal de contracción hacia la media global que produce
    el barajado (bug detectado y corregido antes de correr el barrido completo)."""
    n = phi.size
    assert n % B == 0, "N debe ser múltiplo de B"
    w = n // B
    bloques = phi.reshape(B, w).mean(axis=1)
    lo, hi = float(phi.min()), float(phi.max())
    if hi - lo < 1e-15:
        # campo perfectamente homogéneo -> un solo estado -> entropía 0
        return 0.0
    edges = np.linspace(lo, hi, nbins + 1)
    counts, _ = np.histogram(bloques, bins=edges)
    p = counts.astype(np.float64)
    p = p[p > 0] / p.sum()
    H_shannon = float(-(p * np.log2(p)).sum())
    return H_shannon / LOG2_NBINS


# Tabla de permutaciones de {0,1,2,3} -> índice único (para contar patrones ordinales)
_PERMS_M4 = {}
from itertools import permutations as _iterperm  # noqa: E402
for _idx, _perm in enumerate(_iterperm(range(M_PERM))):
    _PERMS_M4[_perm] = _idx


def i_entropia_permutacion(phi, m=M_PERM):
    """Método B: entropía de permutación ordinal (Bandt-Pompe) sobre ventanas deslizantes
    de m puntos consecutivos en el anillo. Desempate estable por índice (posición)."""
    n = phi.size
    # matriz de ventanas deslizantes (anillo cerrado): shape (n, m)
    idx = (np.arange(n)[:, None] + np.arange(m)[None, :]) % n
    ventanas = phi[idx]  # (n, m)
    # orden relativo con desempate estable por posición dentro de la ventana:
    # argsort con kind='stable' sobre los valores da el orden; en empates exactos
    # (relevante en eps=0) conserva el orden de índice original (0..m-1) -> determinista.
    ordenes = np.argsort(ventanas, axis=1, kind="stable")  # (n, m)
    counts = np.zeros(len(_PERMS_M4), dtype=np.int64)
    # vectorizamos el conteo de patrones vía codificación en base m
    codigos = np.zeros(n, dtype=np.int64)
    for j in range(m):
        codigos = codigos * m + ordenes[:, j]
    # mapear códigos (que son permutaciones válidas de 0..m-1 codificadas en base m,
    # no todos los códigos posibles de m^m ocurren) a índice de patrón vía diccionario
    codigo_a_patron = {}
    for perm, pidx in _PERMS_M4.items():
        c = 0
        for v in perm:
            c = c * m + v
        codigo_a_patron[c] = pidx
    patrones = np.vectorize(lambda c: codigo_a_patron[int(c)])(codigos)
    vals, freqs = np.unique(patrones, return_counts=True)
    p = freqs.astype(np.float64)
    p = p / p.sum()
    H_shannon = float(-(p * np.log2(p)).sum())
    return H_shannon / LOG2_MFACT


def medir_par(phi):
    """Devuelve (X_info, I) para un estado phi dado."""
    return x_info_bloques(phi), i_entropia_permutacion(phi)


def corrida_par(N, eps, H, pasos, seed):
    """Una corrida física; mide (X_info,I) real y null (null = barajar el MISMO phi final)."""
    rng = np.random.default_rng(seed)
    phi, _ = campo_inicial(N, eps, rng)
    activo = np.ones(N, dtype=bool)
    phi, activo = evolucionar_sin_null(phi, activo, H, pasos, rng)
    x_real, i_real = medir_par(phi)
    phi_null = rng.permutation(phi)
    x_null, i_null = medir_par(phi_null)
    return x_real, i_real, x_null, i_null


# ============================================================================
# VERSIONES BATCHED (misma física y misma matemática EXACTA de arriba, solo con
# un eje extra de semillas para acelerar el barrido 9x13x16=1872 corridas: el
# cuello de botella es el loop de `pasos` (~1e4) en Python puro, así que se
# vectoriza sobre las 16 semillas de cada celda (eps,r) para amortizar el
# overhead de numpy por paso. Verificadas bit-a-bit contra las versiones
# escalares de arriba (test de regresión) antes de usarse en el barrido real.
# ============================================================================

def campo_inicial_batch(N, eps, rng, n_batch):
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    fondo = np.ones((n_batch, N), dtype=float)
    if eps <= 0.0:
        return fondo, x
    fases = rng.uniform(0, 2 * np.pi, size=(n_batch, 5))
    pert = np.zeros((n_batch, N), dtype=float)
    for m in range(1, 6):
        fase = fases[:, m - 1][:, None]
        pert += np.sin(2 * np.pi * m * x[None, :] + fase) / m
    pert -= pert.mean(axis=1, keepdims=True)
    std = pert.std(axis=1, keepdims=True)
    std = np.where(std > 0, std, 1.0)
    pert = pert / std
    return fondo + eps * pert, x


def paso_difusion_batch(phi, activo):
    left = np.roll(phi, 1, axis=1)
    right = np.roll(phi, -1, axis=1)
    e_left = np.roll(activo, 1, axis=1)
    e_right = activo
    n_nb = e_left.astype(np.float64) + e_right.astype(np.float64)
    s = e_left * left + e_right * right
    media = np.divide(s, n_nb, out=phi.copy(), where=n_nb > 0)
    nuevo = phi + 0.5 * (media - phi)
    return np.where(n_nb > 0, nuevo, phi)


def paso_expansion_batch(activo, H, rng):
    if H <= 0.0:
        return activo
    activo = activo.copy()
    if H >= 1.0:
        activo[:, :] = False
        return activo
    u = rng.random(activo.shape)
    cortar = activo & (u < H)
    activo[cortar] = False
    return activo


def evolucionar_batch(phi, activo, H, pasos, rng):
    for _ in range(pasos):
        phi = paso_difusion_batch(phi, activo)
        activo = paso_expansion_batch(activo, H, rng)
    return phi, activo


def permutar_filas(phi, rng):
    n_batch, n = phi.shape
    idx = np.argsort(rng.random((n_batch, n)), axis=1)
    return np.take_along_axis(phi, idx, axis=1)


_CODIGO_A_PATRON = {}
for _perm, _pidx in _PERMS_M4.items():
    _c = 0
    for _v in _perm:
        _c = _c * M_PERM + _v
    _CODIGO_A_PATRON[_c] = _pidx


def x_info_bloques_batch(phi, B=B_BLOQUES, nbins=NBINS_XINFO):
    n_batch, n = phi.shape
    w = n // B
    bloques = phi.reshape(n_batch, B, w).mean(axis=2)
    lo = phi.min(axis=1)
    hi = phi.max(axis=1)
    out = np.zeros(n_batch)
    for i in range(n_batch):
        loi, hii = float(lo[i]), float(hi[i])
        if hii - loi < 1e-15:
            out[i] = 0.0
            continue
        edges = np.linspace(loi, hii, nbins + 1)
        counts, _ = np.histogram(bloques[i], bins=edges)
        p = counts.astype(np.float64)
        p = p[p > 0] / p.sum()
        out[i] = float(-(p * np.log2(p)).sum()) / LOG2_NBINS
    return out


def i_entropia_permutacion_batch(phi, m=M_PERM):
    n_batch, n = phi.shape
    idx = (np.arange(n)[:, None] + np.arange(m)[None, :]) % n
    ventanas = phi[:, idx]  # (n_batch, n, m)
    ordenes = np.argsort(ventanas, axis=2, kind="stable")
    codigos = np.zeros((n_batch, n), dtype=np.int64)
    for j in range(m):
        codigos = codigos * m + ordenes[:, :, j]
    out = np.zeros(n_batch)
    for i in range(n_batch):
        patrones = np.vectorize(lambda c: _CODIGO_A_PATRON[int(c)])(codigos[i])
        vals, freqs = np.unique(patrones, return_counts=True)
        p = freqs.astype(np.float64)
        p = p / p.sum()
        out[i] = float(-(p * np.log2(p)).sum()) / LOG2_MFACT
    return out


def corridas_batch(N, eps, H, pasos, n_batch, base_seed):
    """n_batch corridas independientes (una por semilla) para una celda (eps,r)."""
    rng = np.random.default_rng(base_seed)
    phi, _ = campo_inicial_batch(N, eps, rng, n_batch)
    activo = np.ones((n_batch, N), dtype=bool)
    phi, activo = evolucionar_batch(phi, activo, H, pasos, rng)
    x_real = x_info_bloques_batch(phi)
    i_real = i_entropia_permutacion_batch(phi)
    phi_null = permutar_filas(phi, rng)
    x_null = x_info_bloques_batch(phi_null)
    i_null = i_entropia_permutacion_batch(phi_null)
    return x_real, i_real, x_null, i_null


def main():
    t0 = time.time()
    print(f"[E5.6-5] calibrando pasos (eps_ref={CAL_EPS_REF}, semillas={CAL_SEMILLAS})...", file=sys.stderr, flush=True)
    cal_ref = medir_pasos_lavado(N, CAL_EPS_REF, CAL_SEMILLAS)
    pasos_fijo = cal_ref["pasos"]
    print(
        f"[E5.6-5] calibracion: mediana_lavado={cal_ref['mediana']} pasos={pasos_fijo} "
        f"lavo_todas={cal_ref['lavo_todas']}",
        file=sys.stderr, flush=True,
    )

    filas = []
    meta_por_eps = []
    n_eps, n_r = len(EPS_LIST), len(R_TARGETS)
    for ei, eps in enumerate(EPS_LIST):
        D = float(np.mean([medir_D(N, eps, s) for s in range(SEMILLAS)]))
        meta_por_eps.append({"eps": eps, "D": D})
        for ri, r_tgt in enumerate(R_TARGETS):
            if D > 0:
                H = float(min(r_tgt * D, 1.0))
                r_eff = H / D
            else:
                H = 0.0 if r_tgt == 0 else 1.0
                r_eff = 0.0 if r_tgt == 0 else float("inf")

            # batched sobre las SEMILLAS semillas de esta celda (eps,r) -- misma matemática
            # que corrida_par (verificada bit-a-bit en regresión antes de correr esto),
            # solo vectorizada para que el barrido sobredimensionado completo sea viable
            # en tiempo (~22 min en vez de ~3.3h con el loop escalar puro).
            base_seed = 5000 + ei * 100000 + ri * 1000
            xr_a, ir_a, xn_a, in_a = corridas_batch(N, eps, H, pasos_fijo, SEMILLAS, base_seed)
            filas.append({
                "eps": eps, "r_target": r_tgt, "H": H, "D": D, "r_eff": r_eff,
                "X_info_real_mean": float(xr_a.mean()), "X_info_real_std": float(xr_a.std()),
                "I_real_mean": float(ir_a.mean()), "I_real_std": float(ir_a.std()),
                "X_info_null_mean": float(xn_a.mean()), "X_info_null_std": float(xn_a.std()),
                "I_null_mean": float(in_a.mean()), "I_null_std": float(in_a.std()),
                "X_info_real_all": [round(float(v), 6) for v in xr_a],
                "I_real_all": [round(float(v), 6) for v in ir_a],
                "X_info_null_all": [round(float(v), 6) for v in xn_a],
                "I_null_all": [round(float(v), 6) for v in in_a],
            })
        print(f"[E5.6-5] eps {ei+1}/{n_eps} (eps={eps:g}, D={D:.6g}) listo — {n_r} r_targets x {SEMILLAS} semillas",
              file=sys.stderr, flush=True)

    # ---------- correlaciones agregadas ----------
    def pearson(a, b):
        a = np.asarray(a); b = np.asarray(b)
        if a.std() < 1e-15 or b.std() < 1e-15:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    def spearman(a, b):
        a = np.asarray(a); b = np.asarray(b)
        ra = np.argsort(np.argsort(a))
        rb = np.argsort(np.argsort(b))
        return pearson(ra.astype(float), rb.astype(float))

    # pooled sobre TODAS las corridas individuales (no solo medias por celda)
    X_real_all, I_real_all, X_null_all, I_null_all = [], [], [], []
    for f in filas:
        X_real_all.extend(f["X_info_real_all"])
        I_real_all.extend(f["I_real_all"])
        X_null_all.extend(f["X_info_null_all"])
        I_null_all.extend(f["I_null_all"])

    corr_pooled = {
        "pearson_real": pearson(X_real_all, I_real_all),
        "spearman_real": spearman(X_real_all, I_real_all),
        "pearson_null": pearson(X_null_all, I_null_all),
        "spearman_null": spearman(X_null_all, I_null_all),
        "n_pares_real": len(X_real_all),
        "n_pares_null": len(X_null_all),
    }

    # por-celda (medias por (eps,r) agregando semillas) -> correlación de las curvas medias
    X_cell_real = [f["X_info_real_mean"] for f in filas]
    I_cell_real = [f["I_real_mean"] for f in filas]
    X_cell_null = [f["X_info_null_mean"] for f in filas]
    I_cell_null = [f["I_null_mean"] for f in filas]
    corr_por_celda = {
        "pearson_real": pearson(X_cell_real, I_cell_real),
        "spearman_real": spearman(X_cell_real, I_cell_real),
        "pearson_null": pearson(X_cell_null, I_cell_null),
        "spearman_null": spearman(X_cell_null, I_cell_null),
        "n_celdas": len(filas),
    }

    # bootstrap sobre semillas para dispersión de la correlación pooled-real
    rng_boot = np.random.default_rng(999)
    boot_vals = []
    n_seed_boot = 200
    X_real_arr = np.array(X_real_all); I_real_arr = np.array(I_real_all)
    n_pairs = len(X_real_arr)
    for _ in range(n_seed_boot):
        idx = rng_boot.integers(0, n_pairs, size=n_pairs)
        boot_vals.append(pearson(X_real_arr[idx], I_real_arr[idx]))
    boot_vals = np.array([v for v in boot_vals if np.isfinite(v)])
    bootstrap_pearson_real = {
        "mean": float(boot_vals.mean()) if boot_vals.size else float("nan"),
        "std": float(boot_vals.std()) if boot_vals.size else float("nan"),
        "p2_5": float(np.percentile(boot_vals, 2.5)) if boot_vals.size else float("nan"),
        "p97_5": float(np.percentile(boot_vals, 97.5)) if boot_vals.size else float("nan"),
        "n_boot": int(boot_vals.size),
    }

    result = {
        "experimento": "E5.6-5",
        "descripcion": "Exergia informacional (X_info, metodo bloques+histograma) vs I "
                        "(entropia estructural, metodo permutacion ordinal Bandt-Pompe) -- "
                        "relacion empirica contra NULL, sin forzar S=I*E",
        "N": N, "B_bloques": B_BLOQUES, "nbins_x_info": NBINS_XINFO, "m_permutacion": M_PERM,
        "eps_list": EPS_LIST, "r_targets": R_TARGETS, "semillas": SEMILLAS,
        "pasos_fijo": pasos_fijo, "calibracion_ref": cal_ref,
        "meta_por_eps": meta_por_eps,
        "filas": filas,
        "correlacion_pooled_todas_las_corridas": corr_pooled,
        "correlacion_por_celda_eps_r": corr_por_celda,
        "bootstrap_pearson_pooled_real": bootstrap_pearson_real,
        "elapsed_s": time.time() - t0,
        "pre_registro_archivo": "E5_6_5_PROTOCOLO_PREREGISTRO.md",
        "codigo_base_reutilizado": str(BASE_CODE),
        "timestamp_fin_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    out_json = OUT / "E5_6_5_resultado.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[E5.6-5][archivo] {out_json}", file=sys.stderr)
    print(f"[E5.6-5][corr_pooled] {corr_pooled}", file=sys.stderr)
    print(f"[E5.6-5][corr_por_celda] {corr_por_celda}", file=sys.stderr)
    print(f"[E5.6-5][bootstrap] {bootstrap_pearson_real}", file=sys.stderr)
    print(f"[E5.6-5][elapsed] {result['elapsed_s']:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
