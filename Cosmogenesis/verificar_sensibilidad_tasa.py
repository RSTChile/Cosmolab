# INSTRUCCION_CC_sensibilidad_tasa_PARA_CC.md: repetir la comparación REAL-vs-NULL del PUENTE
# (la que dio z=6.92 en verificar_puente_layout_limpio.py) con tasa_expansion en {0.01, 0.02, 0.03}.
# Mismos parametros exactos del puente (N=250 => nq=1500,naq=1050,ne=500,npos=350,pasos_basal=150,
# amp_rugosidad=1.5), mismas 5 semillas REAL x 8 NULL, mismo discriminante (n_clusters_ligados, _z).
# Lo UNICO que cambia entre tandas es cs073_cierre_holistico.TASA_EXPANSION (patcheado antes de cada
# tanda; misma logica del motor, ningun cambio de formula ni de metrica).
import time, json
import numpy as np

import cs073_cierre_holistico as ch

N_SEMILLAS_REAL = 5
N_NULL = 8
SEEDS_REAL = [12345 + i * 1000 for i in range(N_SEMILLAS_REAL)]
SEEDS_NULL = [5000 + i * 2 for i in range(N_NULL)]

resultados = {}
t_total0 = time.time()

for tasa in (0.01, 0.02, 0.03):
    ch.TASA_EXPANSION = tasa  # UNICO parametro que cambia entre tandas (G-SOLO-CAMBIA-TASA)
    t0 = time.time()

    masa_bar, dens_bar, obs_basal = ch._extraer_bariones(1500, 1050, 500, 350, 150, 1.5)
    print(f"[tasa={tasa}] basal: {obs_basal} n_bariones={len(masa_bar)} t={time.time()-t0:.1f}s", flush=True)

    reales = []
    for s in SEEDS_REAL:
        r = ch._dinamica_estructura(masa_bar, dens_bar, 1.5, semilla="causal", seed_dens_null=None,
                                     seed_layout=s, n_pasos_estructura=60)
        reales.append(r)
        print(f"  [tasa={tasa}] REAL seed_layout={s}: n_clusters_ligados={r.get('n_clusters_ligados')}", flush=True)

    nulls = []
    for s in SEEDS_NULL:
        r = ch._dinamica_estructura(masa_bar, dens_bar, 1.5, semilla="causal", seed_dens_null=s,
                                     seed_layout=12345, n_pasos_estructura=60)
        nulls.append(r)
        print(f"  [tasa={tasa}] NULL seed_dens_null={s}: n_clusters_ligados={r.get('n_clusters_ligados')}", flush=True)

    ligados_real = [r["n_clusters_ligados"] for r in reales if r.get("ok")]
    ligados_null = [r["n_clusters_ligados"] for r in nulls if r.get("ok")]

    z_media_real = ch._z(float(np.mean(ligados_real)), ligados_null) if ligados_null and ligados_real else None

    resultados[str(tasa)] = dict(
        tiempo_s=round(time.time() - t0, 1),
        n_bariones=len(masa_bar),
        ligados_real=ligados_real, ligados_real_media=float(np.mean(ligados_real)) if ligados_real else None,
        ligados_real_std=float(np.std(ligados_real, ddof=1)) if len(ligados_real) > 1 else 0.0,
        ligados_null=ligados_null, ligados_null_media=float(np.mean(ligados_null)) if ligados_null else None,
        ligados_null_std=float(np.std(ligados_null, ddof=1)) if len(ligados_null) > 1 else 0.0,
        z_media_real_vs_null=z_media_real,
    )
    print(f"=== [tasa={tasa}] z={z_media_real} ===\n", flush=True)

print("\n=== RESULTADO COMPLETO (sensibilidad tasa_expansion) ===")
print(json.dumps(resultados, indent=2, default=str))
print(f"\ntiempo_total_s={round(time.time()-t_total0,1)}")
