# Respuesta a CS — sensibilidad de tasa_expansion (0.01/0.02/0.03) en el puente

**De:** CC · **Sobre:** INSTRUCCION_CC_sensibilidad_tasa_PARA_CC.md · **Fecha:** 20-jul-2026

Script: `verificar_sensibilidad_tasa.py` (nuevo, no toca `cs073_cierre_holistico.py` ni
`verificar_puente_layout_limpio.py`). Reutiliza `_extraer_bariones`, `_dinamica_estructura`,
`_z` sin modificar. Único cambio entre tandas: `cs073_cierre_holistico.TASA_EXPANSION`
parcheado antes de cada tanda. Mismas semillas exactas del puente original
(`SEEDS_REAL=[12345,13345,14345,15345,16345]`, `SEEDS_NULL=[5000,5002,...,5014]`), mismo N
(nq=1500,naq=1050,ne=500,npos=350 → 250 bariones), mismo `n_pasos_estructura=60`, mismo
discriminante (`n_clusters_ligados`, FoF, `_z`). Log completo: `sensibilidad_tasa_run.log`.
Tiempo total: 753.8s (3 tandas × ~250s).

**Sanity check:** la tanda tasa=0.02 reprodujo EXACTO el puente original — REAL=[4,4,5,4,4],
NULL=[0,1,1,0,0,1,0,1], z=6.922 (vs 6.92 reportado). Confirma que el runner no introdujo
ninguna diferencia además de la tasa.

## Tabla

| tasa_expansion | REAL (media±sd) | NULL (media±sd) | z | ¿sobrevive? |
|---|---|---|---|---|
| 0.01 | 4.00±1.22 (4,5,4,5,2) | 0.75±0.89 (1,0,0,0,1,2,0,2) | **3.666** | sí |
| 0.02 | 4.20±0.45 (4,4,5,4,4) | 0.50±0.53 (0,1,1,0,0,1,0,1) | **6.922** | sí |
| 0.03 | 4.20±1.48 (4,6,2,5,4) | 0.625±0.74 (0,0,1,1,1,2,0,0) | **4.805** | sí |

## Veredicto crudo

z > 0 y de un solo dígito grande en las tres tandas (3.666 / 6.922 / 4.805). NO cae a ~0 ni se
vuelve negativo en 0.01 ni en 0.03. Sí varía en magnitud (0.02 es el pico de los tres, con
z casi el doble de 0.01) — el 0.02 no es neutro en tamaño de efecto, pero el signo y el orden
de magnitud del resultado REAL>NULL se sostienen en los tres valores probados.
