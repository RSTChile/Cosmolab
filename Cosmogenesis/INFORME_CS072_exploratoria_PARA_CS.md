# INFORME CS072 — corrida exploratoria: colapso casi total. Reporto ANTES de tocar la tanda (§8).

## CC, 17-jul-2026. Para CS. Ejecuta DISENO_CS072_experimento_unico_CS.md.

## Qué construí
`cs072_experimento_unico.py`: `proceso072` = `H.proceso067` (los 17 ingredientes) con tres mecanismos
plegados dentro del mismo bucle por-paso:
- **CS070 (semilla):** el voto SSB usa `S70._ssb_potts_semilla` en vez de `H.ssb_potts`, con sesgo mínimo
  presente desde t=0.
- **CS071 (memoria):** caminantes ligeros (N/4, 2 saltos) cada paso sobre el adj vivo, refuerzo/decay/
  homeostasis/poda con los parámetros YA validados en CS071 (0.04/0.99/0.15).
- **CS068 (enfriamiento):** reusa el wmap de correlación YA calculado para el voto SSB (ingrediente 14) —
  cada enlace sobrevive el paso con p=exp(-d_ij/T), d_ij=-log(w_ij), T=la temperatura que el motor ya
  calcula ese paso.
- CS069 (fase cuántica): declarado FUERA, como autoriza tu diseño.

## LA CORRIDA EXPLORATORIA (declarada, antes de tocar la tanda de veredicto — §8)
N=300, ambos brazos: el grafo colapsa casi a la nada. `todo`: 6 aristas sobrevivientes (de ~900 iniciales,
meandeg=6). `null_todo_barajado`: 3 aristas. Diámetro trivial (1) en ambos — no es un resultado, es una
degeneración numérica. No lo escondo ni lo suavizo: esto es exactamente el escenario que tu §8 pide
reportar ANTES de la tanda, no ajustar a mitad de camino.

## Causa raíz — medida, no adivinada
El T(step) del motor (el que ya gobierna masa/confinamiento) **se planta en 0.04 desde el paso 8 de 20, y
se queda ahí el 60% restante de la corrida**:

| step | T | exp(-0.2/T) [d_ij chico, enlace BIEN soportado] |
|---|---|---|
| 0 | 3.00 | 0.936 |
| 4 | 0.30 | 0.513 |
| 6 | 0.095 | 0.121 |
| 8 | 0.04 | 0.0067 |
| 10-18 | 0.04 (plano) | 0.0067 (plano) |

Con T=0.04 sostenido 12 pasos seguidos, hasta un enlace con d_ij=0.2 (w_ij≈0.82, bastante bien soportado —
NO un atajo) sobrevive un paso con p=0.0067, y la probabilidad ACUMULADA de sobrevivir los 12 pasos es
0.0067^12 ≈ 10⁻²⁶. Solo el enlace (o los pocos enlaces) con w_ij≈1.000 EXACTO (d_ij≈0) tiene chance real.
Por construcción de `_pesos_correlacion`, w=1.0 exacto solo lo tiene el enlace de MAYOR soporte del grafo
ESE paso — así que casi todo muere, no solo los atajos.

## Por qué no lo arreglé yo solo
El diagnóstico es claro (mismatch de escala entre el T del motor —que sirve para masa/confinamiento, y
llega a un piso bajo por diseño de esos mecanismos— y la fórmula exp(-d/T) de CS068, calibrada en su propio
experimento con un T0=8.0→T_final=0.05 en escala DISTINTA). Pero CORREGIRLO es una decisión de diseño, no
un bug de tipeo, y tu §8 es explícito: "si el motor no arranca con todo plegado, CC lo REPORTA... y se
decide ANTES de la tanda, nunca a mitad de camino". No quiero elegir yo solo cuál de estas rutas es la
correcta:

**Opción 1 — normalizar d_ij por paso (auto-calibrado, sin número nuevo):** d_ij_norm = d_ij / mediana(d_ij
ese paso), y usar p=exp(-d_ij_norm/T). El umbral lo fija la propia distribución de ese paso (mismo espíritu
que "el NULL fija el umbral" de CS068 Paso 2). Sigue siendo sensible al piso T=0.04, pero ahora compara
enlaces ENTRE SÍ cada paso, no contra una escala absoluta que no le corresponde.

**Opción 2 — reloj de enfriamiento INDEPENDIENTE para CS068:** correr el T0=8.0→0.05 (factor 0.6) que CS068
YA validó, como un reloj propio en paralelo al T del motor (que sigue gobernando masa/confinamiento sin
tocar). Más fiel al cronograma que CS068 cerró, pero técnicamente ya no es "la MISMA T que gobierna el
resto" — sería DOS relojes co-actuando, no uno.

**Opción 3 — otra que tú prefieras.**

## Lo que NO hice
No ajusté nada a ciegas ni recorrí valores hasta que "saliera". No toqué CS070 ni CS071 (funcionan como
esperaba dentro del loop — el colapso es 100% atribuible a la regla de enfriamiento). No until corrí la
tanda de veredicto.

Pido tu adjudicación antes de seguir.

— CC 🐝
