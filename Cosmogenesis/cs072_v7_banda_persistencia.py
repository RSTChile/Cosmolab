"""
CS072 v7 -- EXPLORATORIA de la BANDA DE PERSISTENCIA.
==============================================================================
Fuente: REFINAMIENTO_CS072_banda_de_persistencia_CS.md + ADJUDICACION_CS072_v6_expansion_poda_CS.md +
ADJUDICACION_CS072_v6_hub_expansion_topologia_CS.md. Diseño/ruling: CS. Codea/ejecuta: CC.

ALCANCE (RULING DE CS en REFINAMIENTO, "RULING DE ALCANCE", respuesta a la pregunta de CC): esta corrida
es la EXPLORATORIA que valida el MOTOR -- que la poda mata el hub sin fragmentar, ciega a longitud, sin
forzar dimensión -- y LOCALIZA las dos fronteras de la banda de persistencia. NO lee veredicto (A/B) de
CS072 (ni tiene número propio): eso sale sólo del fold completo de los 18 elementos + 3 mecanismos nuevos
(la TANDA DE VEREDICTO), que corre DESPUÉS de que CS dé visto bueno a esta exploratoria.

Motor: cs072_v6_nucleo.corre_nucleo_v6 = roce (bootstrap aleatorio) + gravedad cs062 (_grav_peso, SIN
tocar) + flujo-de-enfriamiento (CS, anti-difusivo, piso T>=0) + memoria de enlace (CS071, refuerzo/decay/
poda por debilidad) + PODA POR GRADO (NUEVA aquí, cs072_v6_nucleo._poda_grado): la expansión como operador
de poda topológica -- corta enlaces con probabilidad proporcional al grado de sus extremos relativo al
grado medio del grafo, CIEGA A LA LONGITUD (nunca lee distancia/coordenada, sólo grado -- propiedad del
grafo). Parámetros heredados (DELTA, PASOS de la exploratoria v6 ya validada); lo único NUEVO que se barre
es la tasa de poda (parámetro de realidad, no perilla) y n_focos (ya en el diseño, §6).

DOS ETAPAS (por costo computacional -- declarado, no oculto):
  1) Barrido DENSO de tasa-de-poda x n_focos a N=400 SOLO: grado_max, frac_conectada, delta-Gromov, d_s, CV.
     Estas cuatro son baratas a un solo N y bastan para LOCALIZAR las dos fronteras (dónde deja de ser hub,
     dónde empieza a fragmentar).
  2) beta (pendiente log-log diam vs N, que SÍ requiere escalar N∈{400,900,1600}) sólo en los puntos que
     ANCLAN cada régimen (orden profundo, borde-hub, centro-de-banda-candidato, borde-fragmentación, caos
     profundo), a n_focos=5 ("pocos", representativo). Correr beta en los ~12x3 puntos del barrido denso
     sería carísimo y no aporta más que confirmarlo en los puntos que definen los regímenes.
"""
from __future__ import annotations
import sys, time, json
import numpy as np

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)
import cs072_v6_nucleo as V6
import cs071_histeresis as S71
import cs071_tanda as S71T
import cs064_smoke as SM

RNG = np.random.default_rng
DELTA = 1e-4        # heredado (cs072_v6_exploratoria: invariante en el rango 1e-6..1e-2, se fija el valor medio)
PASOS = 80           # heredado (cs072_v6_exploratoria)
N_PRIMARIO = 400
NS_BETA = [400, 900, 1600]
SEED_BASE = 90172

PODA_TASAS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.35, 0.6, 1.0]
FOCOS = [1, 5, 20]           # una / pocas / muchas (DISEÑO §6)
FOCO_BETA = 5                 # representativo "pocas" para la etapa de beta


def _judges(adj, N, rng):
    diam = S71._diam_robusto(adj, N, rng)
    delta_g = S71._delta_gromov(adj, N, rng)
    frac_gig = S71._frac_gigante(adj, N, rng)
    ds = SM.dim_volumen(adj, N, rng=rng)
    grado_max = max((len(a) for a in adj), default=0)
    return dict(diam=diam, delta_gromov=delta_g, frac_gigante=frac_gig, d_s=ds, grado_max=grado_max)


def _corre_punto(N, n_focos, poda_tasa, seed):
    r = V6.corre_nucleo_v6(N=N, n_focos=n_focos, delta=DELTA, pasos=PASOS, seed=seed, poda_tasa=poda_tasa)
    j = _judges(r["adj"], N, RNG(seed + 1))
    j["cv_final"] = float(r["cvs"][-1])
    return j


def _localiza_fronteras(fila):
    """fila: lista de resultados (mismo n_focos) ordenada por poda_tasa creciente.
    frontera_hub: primer poda donde grado_max cae por debajo de la MITAD de su valor en poda=0
      (deja de ser el régimen 'todo pegado a un pozo' -- umbral RELATIVO al propio barrido, no absoluto).
    frontera_frag: primer poda donde frac_gigante cae por debajo de 0.9 (fragmentación clara, ya no
      'telaraña conectada')."""
    if not fila:
        return None, None
    gmax0 = fila[0]["grado_max"]
    frontera_hub = None
    frontera_frag = None
    for r in fila:
        if frontera_hub is None and gmax0 > 0 and r["grado_max"] < gmax0 / 2.0:
            frontera_hub = r["poda_tasa"]
        if frontera_frag is None and r["frac_gigante"] < 0.9:
            frontera_frag = r["poda_tasa"]
    return frontera_hub, frontera_frag


def main():
    t0 = time.time()
    print("=" * 100, flush=True)
    print("CS072 v7 -- EXPLORATORIA banda de persistencia: barrido tasa-de-poda x n_focos (N=400)", flush=True)
    print("Motor: gravedad + flujo-enfriamiento + memoria + poda-por-grado (SIN los 18 elementos -- eso es la tanda)",
          flush=True)
    print("=" * 100, flush=True)

    resultados = []
    for n_focos in FOCOS:
        print(f"\n--- n_focos={n_focos} ---", flush=True)
        for poda in PODA_TASAS:
            seed = SEED_BASE + n_focos * 1000 + int(round(poda * 10000))
            j = _corre_punto(N_PRIMARIO, n_focos, poda, seed)
            j.update(n_focos=n_focos, poda_tasa=poda, N=N_PRIMARIO)
            resultados.append(j)
            print(f"  poda={poda:5.3f}  grado_max={j['grado_max']:4d}  frac_conectada={j['frac_gigante']:.3f}  "
                  f"delta_gromov={j['delta_gromov']:.2f}  d_s={j['d_s']:.2f}  diam={j['diam']:.2f}  "
                  f"CV={j['cv_final']:.3f}  (t={(time.time()-t0)/60:.1f}min)", flush=True)

    with open("cs072_v7_banda_persistencia_barrido.json", "w") as f:
        json.dump(resultados, f, indent=2)

    print("\n" + "=" * 100, flush=True)
    print("LOCALIZACIÓN DE FRONTERAS (por n_focos, sobre el barrido N=400)", flush=True)
    print("=" * 100, flush=True)
    bandas = {}
    for n_focos in FOCOS:
        fila = sorted([r for r in resultados if r["n_focos"] == n_focos], key=lambda r: r["poda_tasa"])
        fh, ff = _localiza_fronteras(fila)
        bandas[n_focos] = (fh, ff)
        ancho = (ff - fh) if (fh is not None and ff is not None) else None
        print(f"n_focos={n_focos}: frontera_hub(deja_de_ser_hub)~{fh}  frontera_frag(empieza_a_fragmentar)~{ff}  "
              f"ancho_banda~{ancho}", flush=True)

    print("\n" + "=" * 100, flush=True)
    print(f"beta (pendiente log-log diam vs N, N∈{NS_BETA}) en puntos representativos, n_focos={FOCO_BETA}",
          flush=True)
    print("=" * 100, flush=True)
    fh, ff = bandas.get(FOCO_BETA, (None, None))
    candidatos_raw = [0.0]
    if fh is not None:
        candidatos_raw += [fh / 2.0, fh]
    if fh is not None and ff is not None:
        candidatos_raw.append((fh + ff) / 2.0)
    if ff is not None:
        candidatos_raw += [ff, min(1.0, ff * 2.0)]
    else:
        candidatos_raw.append(1.0)
    candidatos = sorted(set(round(x, 4) for x in candidatos_raw))

    resumen_beta = []
    for poda in candidatos:
        diams, fracs, gmax, dgs = [], [], [], []
        for N in NS_BETA:
            seed = SEED_BASE + 5000 + int(round(poda * 10000)) + N
            j = _corre_punto(N, FOCO_BETA, poda, seed)
            diams.append(j["diam"]); fracs.append(j["frac_gigante"])
            gmax.append(j["grado_max"]); dgs.append(j["delta_gromov"])
            print(f"  poda={poda:.4f} N={N}: diam={j['diam']:.2f} grado_max={j['grado_max']} "
                  f"frac_gigante={j['frac_gigante']:.3f} delta_g={j['delta_gromov']:.2f} "
                  f"(t={(time.time()-t0)/60:.1f}min)", flush=True)
        Ns_validos = [N for N, d in zip(NS_BETA, diams) if np.isfinite(d) and d > 0]
        diams_validos = [d for d in diams if np.isfinite(d) and d > 0]
        if len(Ns_validos) >= 2:
            beta, _ = S71T._pendiente_loglog(Ns_validos, diams_validos)
        else:
            beta = float("nan")
        fila = dict(poda_tasa=poda, beta=beta, grado_max_N1600=gmax[-1] if gmax else None,
                    frac_gigante_N1600=fracs[-1] if fracs else None,
                    delta_gromov_N1600=dgs[-1] if dgs else None)
        resumen_beta.append(fila)
        print(f"  >>> poda={poda:.4f}: beta={beta:.3f}  grado_max(N=1600)={fila['grado_max_N1600']}  "
              f"frac_gigante(N=1600)={fila['frac_gigante_N1600']}  delta_g(N=1600)={fila['delta_gromov_N1600']}",
              flush=True)

    with open("cs072_v7_banda_persistencia_beta.json", "w") as f:
        json.dump(dict(bandas={str(k): v for k, v in bandas.items()}, resumen_beta=resumen_beta), f, indent=2)

    print(f"\ntiempo total: {(time.time()-t0)/60:.2f} min", flush=True)


if __name__ == "__main__":
    main()
