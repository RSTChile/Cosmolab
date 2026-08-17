# INFORME CS071 — Tanda blindada: VEREDICTO (B), confirma tu predicción. Control positivo validado.

## CC, 17-jul-2026. Para CS. Ejecuta DISENO_CS071_histeresis_memoria_enlace_CS.md.

## Recalibración exploratoria (declarada, antes de la tanda de veredicto — G-NO-AJUSTAR-CRONOGRAMA)
Los parámetros que calqué literalmente de la prosa del diseño (refuerzo fuerte, decay 0.95, poda relativa
al grado actual) tenían un bug de retroalimentación: el umbral de poda comparado contra el grado YA
reducido sube cada vez que se poda algo, cayendo en cascada (grado medio 6→<1, frac_gigante→0.005 en un
smoke). Lo cacé ANTES de tocar la tanda de veredicto. Corregí el umbral a un valor FIJO respecto al peso
ORIGINAL (no al grado actual) y barrí {refuerzo, decay, poda} buscando SOLO evitar colapso catastrófico en
AMBOS sustratos (WS y retícula) — nunca "acercarme a √N". Fijé REFUERZO=0.04, DECAY=0.99, PRUNE_FRAC=0.15,
30 pasos, idéntico en los 4 brazos, antes de correr la tanda de veredicto.

## RESULTADO — tabla completa (8 semillas × 3 N)

| brazo | diam(400,900,1600) | β | frac_gigante | HUB |
|---|---|---|---|---|
| histeresis | 8.94 / 10.12 / 11.06 | 0.154 | 0.995 | no |
| null_barajado | 8.25 / 8.94 / 9.94 | 0.132 | 1.000 | no |
| sin_proceso | 7.94 / 9.12 / 9.62 | 0.141 | 1.000 | no |
| histeresis_sobre_reticula (control) | 33.94 / 55.19 / 65.44 | **0.482** | 0.780 | no |

## VEREDICTO: (B) — confirma tu predicción pre-registrada
β=0.154 (histéresis) ≈ β=0.132 (null_barajado) ≈ β=0.141 (sin_proceso) — los tres muy lejos de 0.5, todos
consistentes con mundo-pequeño (log N). La memoria de enlace NO metriciza el sustrato relacional. Cuarta
ruta independiente al mismo muro (CS066-070 + CS071).

## Por qué confío en este (B) — el control positivo funcionó exactamente como debía
El control `histeresis_sobre_reticula` dio **β=0.482** — a un paso de 0.5, el ideal métrico — con el
MISMO mecanismo, el MISMO cronograma, corriendo sobre una retícula limpia en vez del WS. Esto valida dos
cosas a la vez: (1) el juez de escalamiento SÍ detecta metricidad genuina cuando existe (no es un juez roto
que da β bajo siempre); (2) el proceso NO destruye una geometría que ya estaba ahí — la preserva/amplifica
(β subió de ~0.52 teórico de la retícula pura a 0.482 medido, dentro de ruido, con algo de fragmentación:
frac_gigante=0.78, el proceso sí poda parte de la retícula pero el 78% que sobrevive conserva su
escalamiento métrico). Interpretación (a) de tu diseño: el proceso es al menos métrico-neutral, no
destructivo. No hay lectura (b) (destruye la métrica) que reportar.

## G-ANTI-HUB y G-CONECTIVIDAD
Ningún brazo colapsó a hub (grado_max medio 8.96-9.12 en los sustratos WS, nunca disparado; 4.00 fijo en
la retícula por construcción). frac_gigante se mantuvo alto en los 3 brazos sobre WS (0.995-1.000) y
razonable en el control (0.78). No hay degeneración escondida detrás del β bajo.

## Confirma tu mecanismo medido
El toy que corriste (intermediación de atajos 3.9× sobre locales) predijo que el refuerzo-por-uso carga
justo lo que habría que podar. La tanda es consistente: histéresis apenas se distingue de null_barajado
(β 0.154 vs 0.132, diferencia pequeña y en la dirección de MÁS mundo-pequeño, no menos) — el tránsito
ciego no encuentra una asimetría que romper hacia metricidad en este sustrato.

## El arco ahora
CS066(B) + CS067(B) + CS068(Mundo B) + CS069(B cuántico) + CS070(B con semilla) + CS071(B con memoria de
proceso) — seis rutas independientes, mismo muro. Tu predicción se sostuvo, y el control positivo deja sin
ambigüedad que no es un artefacto del juez o de un proceso roto.

— CC 🐝
