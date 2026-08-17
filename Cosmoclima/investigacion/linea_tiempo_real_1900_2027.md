# Línea de tiempo real del simulador: lluvia anual ZHCS, 1900-2027

Fecha: 01-ago-2026. A pedido de Alexis: el simulador deja de correr con un "tick"
arbitrario y pasa a un calendario real, 1900 a 2027 (365 días fijos por año, sin años
bisiestos, 46.355 días en total) — y "Lluvia acumulada" deja de ser una perilla manual:
se lee sola del año calendario que esté corriendo. Datos en
`fuentes/lluvia_anual_zhcs_1900_2027.csv`.

## Fuente principal: estación Huintil (CR2)
Dentro de la ZHCS (30.5-31.5°S, el punto caliente de diversidad de *Gyriosomus*),
código CR2 4723002. **67 años reales completos, 1915-2018** — mucho mejor cobertura
real que lo que sugería la ficha de metadatos de Peña Blanca (decía "desde 1901" pero
en los hechos solo tiene 32 años con algún dato, la mayoría después de 1991).

## El hueco real de 1933-1963 (no se rellenó con invención)
Huintil tiene un hueco real de ~30 años sin dato en casi ningún mes. Se revisó si
**Lautaro Embalse** (zona Huasco-Freirina, código 3430006) lo tapaba — solo se
solapa en 5 de esos 31 años (1933-1937). A pedido de Alexis, esos 5 años sí se
rellenaron con el dato real de Lautaro (13,6 / 33,7 / 13,0 / 39,0 / 42,0 mm) — queda
marcado en el CSV como `relleno=si-otra-zona` para no perder la trazabilidad de que
viene de otra estación. El resto (1938-1963 y algunos años sueltos: 1924, 1930, 1965,
1966, 1976, 1979, 2016, 2017) queda como **sin dato real** — no se inventó nada.

## 1900-1914 y 2027
Sin dato real de ninguna fuente disponible — 1900-1914 porque no hay estación con
registro ahí, 2027 porque todavía no ocurre.

## 2019 en adelante: NASA POWER
CR2 termina en 2019. Se extendió con NASA POWER (mismo punto ZHCS, -30.6,-71.2, ya
usado para H5) — 2019 a 2025 completos, 2026 parcial (enero-julio, lo único que ya
ocurrió). Es satelital/reanálisis, no estación en tierra — se marca distinto en la
columna `fuente` para no mezclar el tipo de dato sin avisar.

## Resumen
128 años totales (1900-2027): **79 con dato real (62%), 49 sin dato real (38%)**. El
simulador muestra ese "sin dato" explícitamente (badge visible, mismo patrón que
"SENSOR CIEGO") en vez de fabricar un número para los años vacíos.

## Velocidad de simulación (01-ago-2026)
A 1 día simulado por segundo real (ritmo original), recorrer los 46.720 días
(128 años × 365) tomaría ~13 horas — impracticable. Alexis aclaró que agrupar el
gráfico por semana/mes/año (control de arriba) no acelera la simulación, solo cambia
cómo se ven los puntos ya generados. Se agregó un control separado, "Velocidad de
simulación" (slider 1×-180×, fábrica en 60×): avanza 60 días simulados por segundo
real en vez de 1, así que el rango completo 1900-2027 se recorre en ~13 minutos reales
(46.720 días ÷ 60 = 778,7 s ≈ 12,98 min). Verificado con Playwright: a 60× un "frame"
de simulación avanza exactamente 1 día de calendario; a 1× avanza 0 (submúltiplo de
día), confirmando que el multiplicador funciona como se espera.
