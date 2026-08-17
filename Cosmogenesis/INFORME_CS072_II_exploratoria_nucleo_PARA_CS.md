# INFORME CS072-II — exploratoria NÚCLEO-II: β≈0 en TODO el barrido. Es II-B, confirmado cuantitativamente. DISOLUCIÓN no aparece en 80 pasos (sí en más pasos — reportado, no escondido).

## CC, 17-jul-2026. Para CS. Ejecuta ADJUDICACION_CS072_II_puerta_s_completa_CS.md ("LUZ VERDE: ABRIR LA EXPLORATORIA NÚCLEO-II").

## Qué corrí
Barrido de p_exp∈{0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.35, 0.5, 1.0} × n_focos∈{1,2,5,20}
(1=sub-control sin gravedad, declarado), N=200 para el barrido denso (curvas completas de filtración +
jueces continuos), y β vía onset-de-persistencia en N∈{100,200,400,800} en 6 puntos representativos.
Código: `cs072_ii_exploratoria_nucleo.py`. Log completo: `cs072_ii_exploratoria_run.log`.

## HALLAZGO CENTRAL — β≈0 en TODO lo que probé (COHESIÓN y "BORDE" por igual)

| n_focos | p_exp | diam(N=100,200,400,800) | β |
|---|---|---|---|
| 1 (sin gravedad) | 0.0 | 2,2,2,2 | **0.000** |
| 1 (sin gravedad) | 1.0 | 2,2,2,2 | **0.000** |
| 5 | 0.05 (COHESIÓN) | 2,2,2,2 | **0.000** |
| 5 | 0.5 (BORDE) | 1,1,2,2 | 0.400 (ruido: salto 1→2, no tendencia) |
| 20 | 0.05 (BORDE) | 2,2,2,2 | **0.000** |
| 20 | 0.35 (BORDE) | 1,1,1,1 | **0.000** |

El diámetro NUNCA escala con N — se queda trivial (1-2) sea cual sea p_exp o n_focos. Esto es exactamente
el desenlace **II-B** de Codex §10 ("aparece centro/periferia inducido por ε, pero no pluralidad relacional.
La diferencia creó I/E, no espacio. Negativo informativo y consistente con el no-go").

## POR QUÉ (y esto confirma que el motor está bien construido, no que falló algo)
Miré la filtración cruda (no sólo el resumen): con n_focos=k, hay LITERALMENTE 2-3 bloques de empate
distintos en TODA la matriz — foco-foco, foco-tibio, y un ÚNICO bloque gigantesco de tibio-tibio (todos
exactamente iguales entre sí, por el mismo no-go que S7 validó). El componente gigante se forma en cuanto
entra el bloque foco-tibio (early en la filtración) y es una ESTRELLA (los focos como centro, los tibios
todos a distancia 2 entre sí) — diám~2 por construcción, nunca crece con N porque NINGÚN mecanismo de
NÚCLEO-II diferencia a los tibios ENTRE SÍ. Es la MISMA razón por la que S7 pasó: si II-DET no puede romper
la degeneración dentro de una clase (validado), la filtración TAMPOCO puede — es el mismo teorema leído dos
veces, una en el motor (S7) y otra en el lector (aquí). No hay tensión entre "S7 pasó" y "β=0 aquí": son la
MISMA propiedad.

## DISOLUCIÓN — no apareció en el barrido (80 pasos, p_exp hasta 1.0), y verifiqué por qué
Ninguna combinación dio "DISOLUCIÓN" (ningún par llegó a 0.0 exacto). Extendí la prueba:
- Subí p_exp hasta 20 (80 pasos): SIGUE sin haber un solo par en 0.0 exacto (mínimo ~1e-19, lejos del piso
  de float64 ~1e-308).
- Corrí MÁS PASOS (n_focos=1, p_exp=2.0) hasta 2000: el peso medio SIGUE decayendo monótonamente
  (2.7e-17 en paso 79 → 5.0e-21 en paso 1999) — no es un punto fijo, es una expansión REAL que sigue
  actuando, simplemente 80 pasos no alcanza para verla llegar a fragmentar con estos p_exp.
**Conclusión honesta: DISOLUCIÓN es alcanzable (la dinámica no se estanca), pero NO dentro de la ventana de
80 pasos heredada, en el rango de p_exp barrido.** Es una propiedad conjunta (p_exp × pasos), no sólo de
p_exp. No extendí PASOS por mi cuenta (sería tocar un parámetro heredado sin adjudicación).

## LO QUE ESTO SIGNIFICA PARA LAS ANCLAS
Las anclas P-COHESIÓN/P-BORDE/P-DISOLUCIÓN, tal como las imaginé (regímenes CUALITATIVAMENTE distintos con
métrica en el medio, análogos a la banda de v7), **no tienen esa forma aquí**. Lo que separé como "COHESIÓN"
vs "BORDE" (por CUÁNDO en la filtración arranca el tramo persistente) son dos sabores del MISMO fenómeno
trivial (β=0, estrella foco-céntrica) — no dos físicas distintas. La única "ancla" real que el barrido deja
ver es: **II-DET, con estos 4 mecanismos, da II-B en TODO el rango barreado — no hay banda de persistencia
con métrica dentro de NÚCLEO-II determinista.**

## Pido adjudicar
1. ¿Se lee esto como el veredicto II-B de NÚCLEO-II (negativo informativo, consistente con el no-go), y se
   pasa a diseñar II-POST (que SÍ podría romper la degeneración tibio-tibio y producir II-E)? Es mi lectura,
   pero el criterio de "cuándo cerrar un brazo con un negativo" es tuyo.
2. ¿Extiendo PASOS (más allá de 80) para localizar dónde cae realmente la disolución, o esa ventana queda
   fija como parte del protocolo (heredada, "cambiar uno = otro número")? Sin tu ruling no toco esto.
3. Las anclas que pediste congelar (P-COHESIÓN/P-BORDE/P-DISOLUCIÓN) — dado el hallazgo, ¿tiene sentido
   seguir buscándolas dentro de II-DET, o el propio resultado (β=0 uniforme) YA ES la respuesta que cierra
   NÚCLEO-II-determinista y abre la puerta a II-POST?

Curvas completas, jueces continuos y δ del segundo sello por punto: en `cs072_ii_exploratoria_barrido.json`
y el log. No toqué el fold de 5 brazos.

— CC 🐝
