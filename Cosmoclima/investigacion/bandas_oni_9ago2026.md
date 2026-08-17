# Bandas El Niño/La Niña en el gráfico — 09-ago-2026

Alexis, tras ver la lluvia diaria/mensual ya corregida: "se ve muy bien,
con los años del niño muy marcados... sería bueno poner una etiqueta
indicando esos años".

## Fuente y método (no a ojo)

`investigacion/fuentes/oni_historico_completo_1966_2026.csv` — tabla ONI
(Oceanic Niño Index) real, NOAA CPC v5, 1966-2026 completo (2027 no tiene
dato todavía, es futuro, no se inventa). Bajada de
`cpc.ncep.noaa.gov/.../enso/oni/v5/` y cruzada contra el archivo curado que
ya existía en el proyecto (`oni_enso_2026_vs_historico.csv`) para verificar
que es la misma fuente: DJF-1998 2.24 vs 2.2, DJF-2015 0.69 vs 0.7,
DJF-2024 1.92 vs 1.9 — coincide, solo redondeo distinto.

Clasificación con el **criterio oficial de NOAA**, no una lista de años
famosos de memoria: un episodio El Niño/La Niña es una racha de **al menos
5 temporadas trimestrales seguidas** con ONI ≥ +0,5 (Niño) o ≤ −0,5 (Niña).
`generar_bandas_oni.py` implementa exactamente esa regla sobre la serie
completa y arma bandas de fecha (inicio/fin en día-calendario del propio
instrumento) para cada racha que califica — 36 bandas en total (19 Niño,
17 Niña), cubriendo 52,3% del período 1966-2027 (la otra mitad queda
neutral) — es consistente con lo que se sabe de ENSO en general (ocurre
gran parte del tiempo, no es un evento raro).

## Cómo se dibuja

Nuevo plugin propio de Chart.js (`oniBandasPlugin`, en `makeCharts()`),
**sin agregar ninguna librería nueva** — ya hubo un caso esta sesión
(chartjs-plugin-zoom/pan) donde confiar en un plugin de terceros sin
probarlo a fondo salió mal (el pan por arrastre nunca respondió). El
plugin propio pinta rectángulos traslúcidos (rojo=Niño, azul=Niña) detrás
de las curvas en cada `beforeDatasetsDraw` — corre automáticamente en cada
render, incluido después de cualquier zoom/pan, así que las bandas quedan
siempre alineadas sin necesitar wiring aparte con el zoom.

## Verificado

- Corrí el generador: 36 bandas, con los años/rangos que corresponden
  (revisé a mano el rango de 1997-98: banda de abril/mayo-1997 a
  mediados-1998, calza con AMJ-1997=0.8 cruzando el umbral y JJA-1998=-0.8
  ya por debajo).
- En el navegador: `ONI_BANDAS` carga (36), y con zoom a 1996-1998 la
  banda roja cubre EXACTO donde están los picos reales de lluvia y
  floración del mega Niño 97-98, pasando a azul (La Niña) justo cuando la
  curva empieza a bajar — alineación visual perfecta con lo que ya
  sabíamos de esa temporada.
- Sin errores de consola.

## Leyenda

Agregada junto a la ayuda del gráfico: cuadrito rojo = El Niño, cuadrito
azul = La Niña, con la aclaración "ONI real, NOAA, criterio oficial ≥5
temporadas seguidas — no a ojo".
