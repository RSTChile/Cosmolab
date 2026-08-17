# INFORME CS068 Paso 2 — Mundo A vs Mundo B: VEREDICTO en el blob real de CS067

## CC, 16-jul-2026. Para CS. Ejecuta el ruling de ADJUDICACION_CS068_paso1_resultado_CS.md.

## Qué se corrió
`cs068_paso2_mundo_ab.py`: sobre el blob real de CS067 ('completo', motor SIN TOCAR, mismo `E._sustrato()`
de siempre), comparar el soporte local medio (vecinos comunes por arista) del grafo real contra K=30
realizaciones de reconexión que preserva la secuencia de grados EXACTA (double-edge-swap, implementado a
mano — no hay networkx en el venv). Estadístico: media(soporte) real vs distribución de media(soporte) en
las K reconexiones (z-score, K muestras independientes de la estadística, no aristas agrupadas). Regla
PRE-INSCRITA: Mundo A si z>2.0 en TODAS las semillas; Mundo B si z<=0.5 en todas; zona gris si no.

Escala de diseño (N=1500, la que pide DISENO_CS068), 5 semillas, K=30 reconexiones cada una.

## RESULTADO — inequívoco

| semilla | n_edges | soporte REAL | soporte NULL media±std | z |
|---|---|---|---|---|
| 0 | 6856 | 2.559 | 0.264±0.019 | **+121.98** |
| 1 | 5891 | 2.764 | 0.148±0.016 | **+163.45** |
| 2 | 4910 | 2.825 | 0.054±0.009 | **+300.51** |
| 3 | 5711 | 2.685 | 0.090±0.010 | **+272.90** |
| 4 | 4997 | 2.759 | 0.063±0.015 | **+185.44** |

z medio=208.9, mínimo=121.98 — muy por encima del umbral pre-inscrito (z>2.0) en las 5 semillas, sin
excepción. Percentil real dentro del NULL = 1.000 en todas (el real cae fuera del rango completo de 30
reconexiones).

## VEREDICTO: MUNDO A

El blob real de CS067 tiene MUCHÍSIMO más soporte local que lo que su propia secuencia de grados explica
por reconexión al azar. Hay estructura métrica de vecindario genuina, no solo un artefacto de qué tan
conectado está cada nodo. Cumple limpio el criterio pre-inscrito — no hay zona gris que adjudicar.

## Matiz honesto que no quiero ocultar
La MAGNITUD (z en cientos) es mayor a lo que esperaba, y quiero señalar por qué no me sorprende del todo:
que un grafo real tenga más clustering/soporte local que su null de configuration-model es un efecto MUY
conocido y casi universal en redes reales (mundo pequeño, redes sociales, biológicas, etc. — casi todas
clusterizan más que su CM-null). Eso no invalida el test — es EXACTAMENTE el test que pediste, y lo pasa
limpio — pero significa que "Mundo A" en este sentido específico (soporte > CM-null) es una barra MÁS BAJA
que "hay una retícula 2D-like debajo": confirma que hay ALGO de estructura de vecindario no explicada por
grado, no confirma por sí solo que esa estructura tenga la FORMA métrica (tejido con "lejos" real) que
CS068 necesita para que el enfriamiento revele algo útil. Lo próximo (clasificador config-model + re-correr
Etapa 1 sobre el tejido así extraído) es lo que realmente prueba si esa estructura es métrica o solo
clustering sin geometría.

## Qué sigue, pendiente de tu adjudicación
Ya implementé y smoke-testeé `clasifica_config_model()` en `cs068_paso2_mundo_ab.py` (soporte real vs NULL
por bin de grado — deciles de deg_i+deg_j, z>0 ⟹ tejido; reemplaza la mediana de `_clasifica()`). Smoke a
N=300: 1296/1355 aristas (95.6%) tejido, 59 (4.4%) atajo — orden de magnitud similar al soporte==0 ingenuo
que ya se sabía insuficiente, pero ahora con umbral principiado (fijado por el NULL, no elegido a mano).
NO lo corrí aún sobre el blob real a N=1500 completo ni re-corrí Etapa 1 con esta clasificación — quería
cerrar el veredicto Mundo A/B primero, dado que es el pivote del arco. ¿Procedo a re-correr Etapa 1
(inflar_dist vs null_corte_azar, estimador por-cascarón) sobre el blob real de CS067 usando esta
clasificación, a la escala de diseño (N∈{1500,2500})?

— CC 🐝
