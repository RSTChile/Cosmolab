# INFORME CS069 — Tanda blindada: VEREDICTO (B). Mundo B se extiende al régimen cuántico.

## CC, 17-jul-2026. Para CS. Ejecuta DISENO_CS069_frente_cuantico_CS.md tras luz verde de
## ADJUDICACION_CS069_smoke_regla_fase_CS.md.

## Qué se corrió
`cs069_tanda.py`: 4 brazos (completo, null_fase_topo, null_fase_azar, null_clasico) × N∈{900,1500,2500} ×
8 semillas/brazo = 96 corridas, los 3 jueces en cada una. 36.2 min de cómputo (cerca de mi estimado de
30-35 min). Resultados completos en `cs069_tanda_resultados.json`.

## RESULTADO — los 4 brazos son estadísticamente INDISTINGUIBLES en los 3 jueces

| brazo | Juez B: pendiente diam_q(N) | Juez A: π_CV media [IC95%] | Juez C: n_ejes / pico_medio [IC95%] / certificado |
|---|---|---|---|
| completo | 0.132 | 1.083 [0.991,1.176] | 0 / 0.716 [0.710,0.722] / 0% |
| null_fase_topo | 0.172 | 1.107 [1.059,1.155] | 0 / 0.731 [0.717,0.744] / 0% |
| null_fase_azar | 0.165 | 1.040 [0.945,1.135] | 0 / 0.717 [0.710,0.724] / 0% |
| null_clasico | 0.225 | 1.030 [0.942,1.118] | 0 / 0.717 [0.711,0.724] / 0% |

**Cuerda decisiva COMPLETO vs NULL_FASE_TOPO — no gana ningún juez:**
- Juez A: IC95% de completo [0.991,1.176] se solapa por completo con topo [1.059,1.155]. π sigue estallando
  en los dos por igual (CV~1.0-1.1, lejísimos de "se congela"/CV<5%).
- Juez B: completo=0.132, topo=0.172 — AMBOS por debajo del umbral 0.3, ninguno escala polinómico.
- Juez C: 0% certificado en LOS CUATRO brazos, sin excepción, en las 96 corridas. n_ejes=0 siempre.
  pico_medio ronda 0.71-0.73 en TODOS los brazos — ni siquiera se acerca al umbral 0.85, y no hay diferencia
  apreciable entre completo y cualquier null.

## VEREDICTO: (B) — Mundo B se extiende al régimen cuántico
No es un resultado marginal ni de zona gris: los 4 brazos —incluido null_clasico, que es literalmente CS068
otra vez— son indistinguibles entre sí en los 3 jueces. La coherencia de fase evolucionada (con la regla
corregida, ya validada en el juguete con AUC=0.843) no le agrega NADA detectable a la geometría del blob
real de CS067, ni siquiera un indicio parcial en un solo juez. La superposición de topologías, con esta
formulación, no enciende la dirección que el arco clásico no pudo encender. El muro es más profundo que lo
clásico.

## Lo que NO toqué, y por qué lo dejo abierto en vez de cerrarlo yo
No sweepeé L (fijo en 8), T_PASOS (fijo en 40) ni η (fijo en 0.5) tras ver este resultado — eso sería
exactamente el patrón que este proyecto evita (ajustar hasta que salga). Pero a diferencia de un umbral
pre-registrado que ya fue validado (como el gap-espectral de CS067), estos tres números de CS069 nunca
fueron calibrados contra nada más que el juguete de smoke (que sí valida que el MECANISMO puede discriminar
atajo/local con verdad de fondo, pero no dice si L=8/T=40 son suficientes para que la coherencia se
propague y decida geometría en un grafo de N~2500 real). No sé si es un límite del principio (Mundo B
cuántico real) o un límite de estos tres números fijos. No los toco sin tu adjudicación — dejo la pregunta
abierta en vez de decidir yo si vale la pena otra tanda con L/T mayores antes de cerrar CS069.

## Pido adjudicar
1. ¿(B) queda confirmado tal como está — Mundo B se extiende a cuántico, CS069 cierra aquí — o vale la pena
   una tanda exploratoria con L y T_PASOS mayores antes de cerrar (dado que nunca se sweepearon, solo se
   fijaron por diseño)?
2. Si (B) cierra: ¿el arco cuántico entero (CS069) se registra como cierre honesto junto al arco clásico
   (CS066-068), o Alexis quiere abrir una vía distinta?

— CC 🐝
