# FASE III — Experimento 2: poda dinámica por costo de enlace

**Fecha:** 8-ago-2026 · **Escribe:** orquestador (el agente que corrió el experimento dejó el cómputo completo
en disco — `cs081_poda_dinamica.py` + `cs081_poda_dinamica.csv`, 126 filas: 3 semillas × 7 variantes × 6
escalas — pero no llegó a escribir este informe). Análisis calculado directamente sobre el CSV crudo, sin
correr nada nuevo. **No se declara cierre ni veredicto — sólo números. La lectura final es de Alexis.**

## Qué se corrió

Sobre el mismo tejido de Fase III Experimento 1 (CS066, brazo `local`, k_local=6, N=8000, 3 semillas), se
probaron 7 variantes:
- `sin_poda` — línea base, sin tocar ningún enlace.
- `costo_P50` / `costo_P70` / `costo_P90` — se le asigna a cada enlace un costo (inconsistencia histórica +
  conflicto de holonomía + bajo soporte local + baja reciprocidad, tal como quedó pre-registrado en el
  protocolo del Experimento 1) y se podan los enlaces por encima del percentil 50/70/90 de ese costo (P50 poda
  más enlaces — los de costo más alto en la mitad superior — que P90, que sólo poda el 10% más caro).
- `azar_P50` / `azar_P70` / `azar_P90` — control: se poda la MISMA CANTIDAD de enlaces que la variante de costo
  correspondiente, pero elegidos al azar en vez de por costo.

Para cada variante y cada escala de renormalización b=1,2,4,8,16,32 (mismo coarse-graining por cajas BFS del
Experimento 1), se midió diámetro, tamaño del componente gigante, dimensión por bola, holonomía.

## Resultado — el criterio de costo SÍ hace algo distinto de podar al azar

Ajustando la pendiente log(diámetro) vs log(N_b) a través de las 6 escalas (mismo diagnóstico que usó el
Experimento 1 para distinguir mundo-pequeño de geometría real), promediado sobre las 3 semillas:

| Variante | Pendiente media | Desviación entre semillas | Enlaces podados (de ~24000) |
|---|---|---|---|
| sin_poda | **0.421** | 0.029 | 0 |
| azar_P90 | 0.420 | 0.021 | ~3000 (12%) |
| costo_P90 | 0.475 | 0.005 | ~3000 (12%) |
| azar_P70 | 0.462 | 0.071 | ~7850 (33%) |
| costo_P70 | 0.571 | 0.052 | ~7850 (33%) |
| azar_P50 | 0.655 | 0.043 | ~14900 (62%) |
| **costo_P50** | **0.786** | **0.030** | ~14900 (62%) |

**Lectura directa de los números:** podar por costo da SIEMPRE una pendiente mayor que podar al azar la misma
cantidad de enlaces, en los 3 niveles probados (P50, P70, P90), de forma consistente entre las 3 semillas
(desviaciones chicas, no es ruido). La brecha entre costo y azar es más grande cuando se poda más (P50: +0.131)
y se achica cuando se poda menos (P90: +0.055). Podar SIN CRITERIO (azar) ya ayuda un poco respecto de no podar
nada (sin_poda=0.421 → azar_P50=0.655) — simplemente quitar enlaces adelgaza algunos atajos por pura
probabilidad — pero podar POR COSTO ayuda sistemáticamente más que azar con la misma cantidad de enlaces
quitados.

## Qué significa esto, en simple

Es como limpiar una autopista con demasiados atajos: sacar carriles al azar ya reduce algo el efecto de "todo
está a un paso de todo" (menos atajos, en general). Pero un criterio que apunta específicamente a los atajos
más "sospechosos" (por su costo — inconsistencia histórica, conflicto de holonomía, poco soporte local, poca
reciprocidad) saca los atajos correctos con más eficiencia que sacar carriles al azar en la misma cantidad. El
criterio de costo SÍ está capturando algo real sobre cuáles enlaces son los "culpables" del mundo-pequeño, no
es indistinguible de una poda ciega.

## Lo que esto NO responde todavía

Ninguna de las pendientes (ni siquiera costo_P50=0.786) se acerca a lo que se esperaría de una geometría 3D
real (pendiente ≈1/3≈0.33 si se lee como 1/d... **ojo, la convención de pendiente usada acá es la misma que el
Experimento 1, pero interpretarla como "1/dimensión" asume que el diámetro sigue una ley de potencia limpia a
través de las 6 escalas — eso no se verificó por separado; si la escala real fuera logarítmica en vez de
potencia, el ajuste lineal en log-log es sólo una aproximación cruda, no una medición de dimensión genuina**).
Lo que sí se puede decir con solidez es la comparación RELATIVA (costo > azar > sin_poda), no un valor absoluto
de "cuán geométrico" es el resultado final.

## Falsación (criterio pre-registrado en el Experimento 1)

El criterio decía, en esencia: ¿el sistema descubre por sí mismo un "lejos" macroscópico vía el costo, más
allá de lo que lograría simplemente quitar enlaces al azar? **Los números dicen que sí hay una diferencia
sistemática y reproducible entre costo y azar** — el criterio de costo no es intercambiable con azar. Si el
criterio de falsación exigía que costo≈azar para caer, esta vez no cae. Pero la magnitud de la diferencia es
modesta comparada con la magnitud total del efecto de simplemente podar (azar ya mueve mucho la aguja) — la
lectura de cuánto de esto es "descubrimiento genuino de geometría" vs "cualquier poda agresiva ayuda un poco
más si además evita simetrías obvias" queda abierta.

## Síntesis de TODA la Fase III (Experimento 1 + Experimento 2)

1. Sin tocar nada, el tejido de CS066 es mundo-pequeño en todas las escalas de agrupamiento probadas
   (Experimento 1, pendiente≈0.42, indistinguible de barajado/Erdős-Rényi).
2. Agrupar (renormalizar) no lo arregla por sí solo — y de hecho la estructura real se fragmenta MÁS rápido
   que el ruido puro al agrandar la escala (hallazgo lateral del Experimento 1).
3. Podar enlaces por un costo relacional (historia, holonomía, soporte local, reciprocidad) SÍ mueve la
   pendiente más que podar al azar la misma cantidad — un efecto real, reproducible, pero modesto en magnitud
   relativa al efecto de podar en sí.

No se declara cierre de la Fase III. Los números completos están en `cs081_poda_dinamica.csv` (126 filas) para
auditoría directa.

## Archivos

- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs081_poda_dinamica.py` — código del experimento, no toca
  nada congelado (`cs064_sistema_completo.py`, `cs066_localidad_geometrogenesis.py`,
  `cs080_renormalizacion.py`).
- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs081_poda_dinamica.csv` — datos crudos completos.
