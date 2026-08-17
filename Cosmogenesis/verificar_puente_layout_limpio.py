# PASO 3 (INSTRUCCION_CC_layout_fix_PARA_CC.md): re-verificar el puente con el layout de frontera
# reflectante (sin apilamiento). Mismos parámetros exactos del puente original (N=250, f=5,
# n_pasos_estructura=60), ahora con >=5 semillas REAL (layout) x >=8 NULL (aristas barajadas).
# El z re-verificado REEMPLAZA a z=10.26, sea cual sea -- no se retoca.
import time, json
import numpy as np

from cs073_cierre_holistico import _extraer_bariones, _dinamica_estructura, _z

t0 = time.time()
masa_bar, dens_bar, obs_basal = _extraer_bariones(1500, 1050, 500, 350, 150, 1.5)
print("basal:", obs_basal, " n_bariones=", len(masa_bar), " t=%.1fs" % (time.time() - t0), flush=True)

N_SEMILLAS_REAL = 5
N_NULL = 8
SEEDS_REAL = [12345 + i * 1000 for i in range(N_SEMILLAS_REAL)]
SEEDS_NULL = [5000 + i * 2 for i in range(N_NULL)]

reales = []
for s in SEEDS_REAL:
    r = _dinamica_estructura(masa_bar, dens_bar, 1.5, semilla="causal", seed_dens_null=None,
                              seed_layout=s, n_pasos_estructura=60)
    reales.append(r)
    print(f"  REAL seed_layout={s}: n_clusters_ligados={r.get('n_clusters_ligados')}", flush=True)

nulls = []
for i, s in enumerate(SEEDS_NULL):
    r = _dinamica_estructura(masa_bar, dens_bar, 1.5, semilla="causal", seed_dens_null=s,
                              seed_layout=12345, n_pasos_estructura=60)
    nulls.append(r)
    print(f"  NULL seed_dens_null={s}: n_clusters_ligados={r.get('n_clusters_ligados')}", flush=True)

ligados_real = [r["n_clusters_ligados"] for r in reales if r.get("ok")]
ligados_null = [r["n_clusters_ligados"] for r in nulls if r.get("ok")]

z_media_real = _z(float(np.mean(ligados_real)), ligados_null) if ligados_null else None

print("\n=== RESULTADO (layout limpio, reemplaza a z=10.26) ===")
resumen = dict(
    tiempo_total_s=round(time.time() - t0, 1),
    n_bariones=len(masa_bar),
    ligados_real=ligados_real, ligados_real_media=float(np.mean(ligados_real)),
    ligados_null=ligados_null,
    ligados_null_media=float(np.mean(ligados_null)) if ligados_null else None,
    ligados_null_std=float(np.std(ligados_null, ddof=1)) if len(ligados_null) > 1 else 0.0,
    z_media_real_vs_null=z_media_real,
)
print(json.dumps(resumen, indent=2, default=str))
