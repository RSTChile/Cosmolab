# Curva de lluvia por estación, bajo demanda por especie (06-ago-2026)

Alexis decidió, tras comparar opciones: **mantener la curva única del reloj**
(`PLUVIOSIDAD_MENSUAL`, Huintil/CR2+NASA POWER — sin tocar) y sumar, aparte,
la lluvia real de la estación más cercana de cada especie, visible solo
cuando esa especie se marca en el listado. No reemplaza nada del instrumento
ET3-Térmico, es una capa de comparación opcional.

## Cómo funciona
`Web/prueba_de_concepto/generar_curvas_estaciones.py` (nuevo, mismo patrón
que `generar_mapa.py`) lee `investigacion/fuentes/precipitacion_mensual_dmc_anuario2025.csv`
y el propio `especiesData` ya embebido en el HTML, calcula el centroide de
cada especie y su estación real más cercana (distancia haversine), e inyecta
dos constantes nuevas en `prueba_de_concepto_ET3-Termico_con_mapa.html`:
- `LLUVIA_ESTACIONES` — serie mensual real de las 17 estaciones que resultan
  "más cercana" de alguna especie (no las 97 completas, para no inflar el
  HTML con series que ninguna curva usa).
- `ESTACION_MAS_CERCANA` — estación + distancia asignada a cada una de las 42
  especies reales.

Al marcar una especie en el listado, `agregarLineaEspecie()` ahora agrega DOS
líneas al gráfico de Pluviosidad (antes solo agregaba una, el pulso de
presencia documentada): esa misma, más la curva de lluvia real de su
estación más cercana, mismo color, eje mm (el mismo que ya usa la Pluviosidad
ZHCS, para que se puedan comparar visualmente). Al desmarcar, se quitan las
dos. Verificado en vivo (Chrome vía servidor local): funciona con *G.
coriaceus* (Puerto Oscuro, 5,4 km) y *G. angustus* (Inca de Oro, 205,7 km).

## Honestidad de la distancia
Cuando la estación más cercana queda a más de 80 km (9 de las 42 especies —
sobre todo el clúster Paposo/Inca de Oro, extremo norte de Atacama, donde
casi no hay estaciones con serie mensual real todavía), la etiqueta de la
curva agrega **"-- referencial, no local"** en vez de mostrarla con la misma
confianza que una estación a 5 km. No se oculta la curva, pero tampoco se
disfraza de dato local cuando no lo es.

## Para actualizar
Correr `generar_curvas_estaciones.py` de nuevo cada vez que se sume una
campaña nueva a `precipitacion_mensual_dmc_anuario2025.csv`, o después de
correr `generar_mapa.py` si cambiaron los puntos de alguna especie (el
centroide, y por lo tanto la estación más cercana, podría cambiar).
