"""
CS067 — BLINDAJE DE SEMILLAS (CS, 15-jul-2026, respuesta al sweep n=4/n=1): el patrón de pico_medio
(media global 0.779, nunca cruza 0.85) y la anti-especificidad (controles puntuando tan alto como completo
en 3/5 regímenes) son reales, pero γ∈{1.5,2.0,2.5} están sub-muestreados — su IC95% SÍ cruza 0.85 con
solo 4 semillas. Lección de CS058 aplicada: no canonizar (B) sobre datos parciales. Blindaje pedido:
~16 semillas/γ para 'completo', ~8/γ para cada control (sin_correlacion, sin_causal), reportando media + IC95%.

Guardas vigentes (sin cambios, ver ADDENDUM_CS067_pico_guarda_rango_CS.md):
  Guarda 1: pico_medio/frac_picados sobre el voto PRE-snap de ssb_potts (conf_i) — no sobre V final (siempre
            one-hot por construcción, CC lo cazó y corrigió; unit-testeado: unánime->1.0, repartido->1/K).
  Guarda 2: n_ejes vía cuenta_ejes_gap restringido a modos POBLADOS (piso=0.02); reproduce 1/3/0/3/5 — PASA.
  Guarda 3: "enciende" exige (i) n_ejes>1, (ii) pico_medio_completo>=0.85 (media, con IC95% que NO cruce el
            piso por debajo), (iii) especificidad: controles con media claramente por debajo de completo.
            gap_val/gap_interno NO entran en el criterio (solo informativos).

Lectura PRE-INSCRITA (sin cambios de las guardas ni del criterio, solo poder estadístico):
  ENCIENDE -> Fase A si algún régimen cumple (i)+(ii)+(iii) con el IC95% de completo por encima de 0.85 Y
  el IC95% de ambos controles claramente por debajo.
  (B) CANÓNICO si para TODO gamma la media de completo sigue <0.85 (o su IC95% no despega el piso con
  claridad) y/o los controles no quedan por debajo de completo con separación real.
No se tunea ningún otro parámetro para forzar el resultado; k_local/c/K/dark siguen SORTEADOS (G-NO-CALIBRAR).
"""
from __future__ import annotations
import os, sys, time
import numpy as np

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)
import cs067_habitacion_completa as H
import cs065_exclusion_pauli as C65

RNG = np.random.default_rng

N_NODOS = int(os.environ.get("CS067_SWEEP_N", 1200))
SEEDS_COMPLETO = int(os.environ.get("CS067_SWEEP_SEEDS_COMPLETO", 16))
SEEDS_CTRL = int(os.environ.get("CS067_SWEEP_SEEDS_CTRL", 8))
GAMMAS = [0.5, 1.0, 1.5, 2.0, 2.5]  # cubre GAMMA_RANGE completo, extremos incluidos

H.STEPS = int(os.environ.get("CS067_SWEEP_STEPS", H.STEPS))

PICO_PISO = 0.85          # Guarda 3(ii): dominio real, no smear
CTRL_PICO_TOPE = 0.85     # control colapsa si pico_medio < esto (o n_ejes<=1)


def _correr(arm, gamma, seed):
    par = H._sorteo(seed)
    par["gamma"] = gamma                      # ÚNICO parámetro fijado por el barrido; el resto sigue sorteado
    cat = C65._cataloga065(N_NODOS, RNG(seed))
    r2 = RNG(seed * 137 + hash(arm) % 9973 + 5)
    adj, V, dark, tb, f, D, conf = H.proceso067(N_NODOS, cat, arm, par, r2)
    m = H.juzga067(adj, V, dark, N_NODOS, arm, par, f, seed * 19 + hash(arm) % 991, conf=conf)
    return m


def _ic95(x):
    x = np.asarray(x, float)
    m = float(x.mean())
    if len(x) < 2:
        return m, m, m
    sem = x.std(ddof=1) / np.sqrt(len(x))
    return m, m - 1.96 * sem, m + 1.96 * sem


def _tanda(arm, gamma, n_seeds, offset):
    return [_correr(arm, gamma, offset + 137 * s + int(gamma * 1000)) for s in range(n_seeds)]


def main():
    print("=" * 108, flush=True)
    print("CS067 — BLINDAJE DE SEMILLAS: barrido de gamma, 'completo' vs sin_correlacion/sin_causal, media+IC95%",
          flush=True)
    print(f"N={N_NODOS} · steps={H.STEPS} · seeds completo={SEEDS_COMPLETO} · seeds/control={SEEDS_CTRL} "
          f"· gammas={GAMMAS}", flush=True)
    print("=" * 108, flush=True)
    t0 = time.time()
    resumen = []
    for gamma in GAMMAS:
        filas_completo = _tanda("completo", gamma, SEEDS_COMPLETO, 67500)
        filas_corr = _tanda("sin_correlacion", gamma, SEEDS_CTRL, 67900)
        filas_caus = _tanda("sin_causal", gamma, SEEDS_CTRL, 67800)

        nej = np.array([m["n_ejes"] for m in filas_completo])
        pico = np.array([m["pico_medio"] for m in filas_completo])
        pico_corr = np.array([m["pico_medio"] for m in filas_corr])
        pico_caus = np.array([m["pico_medio"] for m in filas_caus])
        nej_corr = np.array([m["n_ejes"] for m in filas_corr])
        nej_caus = np.array([m["n_ejes"] for m in filas_caus])

        pm, plo, phi = _ic95(pico)
        cm, clo, chi = _ic95(pico_corr)
        ym, ylo, yhi = _ic95(pico_caus)

        print(f"\n--- gamma={gamma:.2f} ---", flush=True)
        print(f"  completo (n={len(filas_completo)}): n_ejes={list(nej)}", flush=True)
        print(f"    pico_medio: media={pm:.3f} IC95%=[{plo:.3f},{phi:.3f}]  n_ejes_medio={nej.mean():.2f}",
              flush=True)
        print(f"  sin_correlacion (n={len(filas_corr)}): pico_medio media={cm:.3f} IC95%=[{clo:.3f},{chi:.3f}] "
              f"n_ejes_medio={nej_corr.mean():.2f}", flush=True)
        print(f"  sin_causal      (n={len(filas_caus)}): pico_medio media={ym:.3f} IC95%=[{ylo:.3f},{yhi:.3f}] "
              f"n_ejes_medio={nej_caus.mean():.2f}", flush=True)

        # Guarda 3 con poder estadístico: (ii) IC95% inferior de completo por encima del piso;
        # (iii) IC95% superior de AMBOS controles por debajo del IC95% inferior de completo (separación real)
        cumple_ii = plo >= PICO_PISO
        cumple_iii = (chi < plo) and (yhi < plo)
        cumple_i = bool(nej.mean() > 1)
        enciende = cumple_i and cumple_ii and cumple_iii

        resumen.append(dict(gamma=gamma, pico_media=round(pm, 3), pico_ic=(round(plo, 3), round(phi, 3)),
                             nej_medio=round(float(nej.mean()), 2),
                             corr_media=round(cm, 3), corr_ic=(round(clo, 3), round(chi, 3)),
                             caus_media=round(ym, 3), caus_ic=(round(ylo, 3), round(yhi, 3)),
                             cumple_i=cumple_i, cumple_ii=cumple_ii, cumple_iii=cumple_iii, enciende=enciende))

    print("\n" + "=" * 108, flush=True)
    print("RESUMEN por régimen (Guarda 3 con IC95%: (i) n_ejes~>1 (ii) IC95%_inf(completo)>=0.85 "
          "(iii) IC95%_sup(controles) < IC95%_inf(completo)):", flush=True)
    hdr = (f"{'gamma':>6}{'nej~':>6}{'pico~[IC95%]':>24}{'sin_corr~[IC95%]':>26}{'sin_caus~[IC95%]':>26}"
           f"{'ENCIENDE':>10}")
    print(hdr, flush=True)
    algun_encendido = False
    for r in resumen:
        pico_s = f"{r['pico_media']:.3f}[{r['pico_ic'][0]:.3f},{r['pico_ic'][1]:.3f}]"
        corr_s = f"{r['corr_media']:.3f}[{r['corr_ic'][0]:.3f},{r['corr_ic'][1]:.3f}]"
        caus_s = f"{r['caus_media']:.3f}[{r['caus_ic'][0]:.3f},{r['caus_ic'][1]:.3f}]"
        print(f"{r['gamma']:>6.2f}{r['nej_medio']:>6.2f}{pico_s:>24}{corr_s:>26}{caus_s:>26}"
              f"{str(r['enciende']):>10}", flush=True)
        algun_encendido = algun_encendido or r["enciende"]

    print("\nVEREDICTO PRE-INSCRITO (blindado con IC95%, sin tunear tras ver los números):", flush=True)
    if algun_encendido:
        print("  >=1 régimen de gamma cumple (i)+(ii)+(iii) con separación estadística real -> ENCIENDE.",
              flush=True)
        print("  Siguiente paso: reportar a CS ese régimen para adjudicar Fase A.", flush=True)
    else:
        print("  NINGÚN régimen de gamma en [0.5, 2.5] cumple las tres condiciones con poder estadístico.",
              flush=True)
        print("  (B) CANÓNICO: el espacio emergente no es bastante métrico para soportar direcciones, ni con", flush=True)
        print("  16/8 semillas se despega el piso 0.85 ni se separa de los controles. No tunear más. Reportar", flush=True)
        print("  (B) a CS -> reorienta el arco a cerrar el cabo de mundo-pequeño (candidato CS068: inflación).",
              flush=True)
    print(f"\ntiempo total: {(time.time()-t0)/60:.2f} min", flush=True)


if __name__ == "__main__":
    main()
