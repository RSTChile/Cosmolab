# `demo/` — datos de demostración/visualización del arco CG002

**Qué es:** archivos de datos (CSV/JSONL) usados para ALIMENTAR la visualización del modelo CG002 en
el navegador (`cg002_genesis.html`, en la raíz del proyecto — el "visor" mencionado como reproducible
en `INFORME_GENERAL_CG002_ARCO.md`). No son resultados de un experimento nuevo: son instantáneas del
mismo modelo CG002 (S>0, compatibilidad `c_ij`, acoplamiento dirigido `g_{i←j}`) preparadas para
render — grafos con distintas semillas/parámetros (`grafo_n4_a1_cp0.50_s1.jsonl`,
`grafo_n8_a1_cp0.00_s3.jsonl`, `grafo_n8_a1_cp0.30_s3.jsonl`) y una corrida con "estallido"
(`estallido_n8_s3_dir0.3.jsonl`, la más grande — probablemente la ruptura espontánea de simetría que
`INFORME_GENERAL_CG002_ARCO.md` reporta como no-predicha).

`cg002_N8_s3_dir0.3.csv` es la serie tabular equivalente al jsonl de estallido, para análisis fuera del
visor (Excel/pandas/etc.).

**Para la interpretación de qué significan estos parámetros (n, a, cp, s, dir)**, ver
`INFORME_GENERAL_CG002_ARCO.md` §1 (especificación reproducible del modelo) en la raíz — esta carpeta
sólo contiene los datos, no la explicación del modelo.
