# Qué sistema de mapas usar para el simulador — investigación (2026-07-31)

## La recomendación en una frase
**Leaflet** como motor del mapa, con capas base gratis (CartoDB o Esri) + capas
satelitales gratis de la NASA (GIBS) + nuestros propios datos (GBIF, zonas, estaciones)
como capas encima. Todo gratis, sin llave de API, sin registro, sin límite de uso real
para un proyecto de este tamaño.

## La analogía simple
Un sistema de mapas web tiene dos partes separadas, como una mesa de dibujo:
1. **El motor** — el programa que sabe cómo dibujar un mapa, moverlo, hacer zoom, y
   prender/apagar capas. Esto es código, no datos geográficos.
2. **Las capas** — las imágenes o datos que se ponen ENCIMA del motor: el mapa base
   (calles, relieve), imágenes satelitales (NDVI, temperatura), y nuestros propios
   puntos/polígonos (dónde hay Gyriosomus, dónde está la ZHCS, dónde llovió).

Cada capa puede venir de una fuente distinta y gratis. No hace falta pagar por el motor
NI por las capas si se eligen bien las fuentes.

## Comparación de motores (los tres candidatos serios, los tres gratis)

| Motor | Qué tan fácil | Fuerte en | Para qué sirve mejor |
|---|---|---|---|
| **Leaflet** | Muy fácil, un solo archivo `<script>`, sin instalar nada | Control de capas con casillas (prender/apagar) integrado de fábrica; ecosistema enorme de plugins | Un simulador de una sola página HTML, que es justo lo que vamos a construir |
| **MapLibre GL JS** | Media | Mapas vectoriales suaves, estilo moderno tipo Google Maps, animaciones fluidas | Proyectos que necesiten verse muy pulidos visualmente o manejar millones de puntos |
| **OpenLayers** | Más difícil, curva de aprendizaje más alta | El más potente para datos geoespaciales "serios" (WMS/WFS, proyecciones raras) | Proyectos GIS profesionales — para nosotros es más poder del que necesitamos |

**Por qué Leaflet para este proyecto**: nuestro simulador es un archivo HTML que hay
que poder abrir y entender rápido, con capas científicas simples (lluvia, floración,
escarabajos, ENSO histórico) que se prenden y apagan — exactamente el caso de uso para
el que Leaflet fue diseñado. Su control de capas (`L.control.layers`) da ya la interfaz
de casillero "por capas" que pediste, sin programarla desde cero.

## Las capas que vamos a poder usar, todas gratis

### Capa base (el mapa de fondo)
- **CartoDB Positron/Voyager** — mapa limpio, gratis, sin llave, uso razonable sin
  límite práctico para un proyecto de investigación.
- **Esri World Imagery** — foto satelital de fondo, gratis, sin llave (alternativa si
  se prefiere ver el desierto real en vez de un mapa dibujado).

### Capas satelitales científicas (NDVI, temperatura, etc.)
- **NASA GIBS** (Global Imagery Browse Services) — el hallazgo más útil de esta
  investigación. Es un servicio de la NASA, **100% gratis, sin llave, sin registro**,
  con más de 1000 capas satelitales (incluida vegetación/NDVI-relacionado y temperatura
  superficial), actualizado casi a diario, con **más de 20 años de historia** — se
  puede pedir la imagen de una fecha específica, lo que calza perfecto con nuestros
  eventos de floración documentados (2011, 2015, 2017...). Se conecta a Leaflet como
  una capa de mosaicos (tiles) más, con la fecha como parámetro en la URL.

### Nuestras propias capas (datos del proyecto)
- Ocurrencias de *Gyriosomus* (ya tenemos el CSV/consultas de GBIF).
- Zona de Alta Simpatría Cladística (polígono simple, 30.5-31.5°S).
- Estaciones de lluvia CR2/Quinta Normal.
- Sitios de anillos de árbol (El Asiento, San Gabriel).

Todo esto se dibuja con **GeoJSON**, un formato de texto simple (no una imagen) que
Leaflet lee de forma nativa — no necesita ningún servicio externo, son archivos
nuestros.

## Costo real: cero
Ningún componente de esta arquitectura tiene límite de pago para el tamaño de este
proyecto (un simulador de investigación con tráfico bajo/personal). Si en algún
momento el simulador se hiciera público y con mucho tráfico, el único punto a vigilar
sería el uso razonable de CartoDB (rara vez un problema real); NASA GIBS y OpenFreeMap
están diseñados explícitamente para no tener ese problema.

## Siguiente paso (no hecho todavía, a la espera de luz verde)
Puedo armar un HTML mínimo de prueba (un solo archivo, sin instalar nada) que muestre:
mapa base + una capa NASA GIBS con fecha + el polígono de la ZHCS + los puntos de
Gyriosomus que ya tenemos — para validar que el enfoque funciona antes de construir el
simulador completo encima. Aviso cuando quieras que lo arme.
