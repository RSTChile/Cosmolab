# Barrido de fuentes · 25-ago-2026

Siete agentes buscando en paralelo, agrupados por organismo fuente y no por
sector, que es como se agrupan los catastros reales.

**911.182 registros · 77 fuentes · 1,18 GB · todas con `PROCEDENCIA.txt`**

Nada de esto está integrado todavía. Está bajado, contado y verificado, listo
para revisar cuáles sirven.

## Cuánto movería la aguja

| | ítems |
|---|---|
| Con activos hoy | 44 |
| Poblables con lo bajado (autodeclarado por cada agente) | +143 |
| Físicos totales | 408 |

De **10,8 %** a en torno al **45 %** de lo catastrable, si todo se confirma al
integrar. Hay solapes entre lotes, así que el número final será menor.

## Lo bajado, por volumen

| fuente | registros |
|---|---|
| RETC · fuentes puntuales con CIIU | 348.274 |
| IDE · Hidrografía V2 (ríos y quebradas) | 122.852 |
| **DGA · Inventario Público de Glaciares** | **78.564** |
| SAG/CIREN · Rol Único Pecuario bovinos | 77.943 |
| CONAF/CIREN · usos de tierra agrícola | 74.981 |
| MMA · humedales | 40.378 |
| SMA/SNIFA · unidades fiscalizables | 33.558 |
| MINDEP · infraestructura deportiva | 20.701 |
| SUBTEL · antenas Ley de Torres | 18.007 |
| …y 68 fuentes más | |

## Lo que se resolvió

**El bloqueo del combustible.** `id_combustible` está nulo y no se llena nunca,
pero el dato vive en las fichas técnicas por unidad del Coordinador. Contrastado
contra la capa `centrales_termoele` de la CNE: **192 de 213 centrales térmicas
(90 %) quedan con combustible** — diésel 153, biomasa 25, gas 18, carbón 10.
Desbloquea los ítems 89, 90 y 91.

**Los glaciares.** 78.564 polígonos, no los ~24.000 de la cifra que circula, con
volumen, equivalente de agua, altura, pendiente y orientación.

**El registro de radios.** El `Token Required` del servidor ArcGIS de Subtel no
bloqueaba nada: el registro está publicado como XLSX abierto en el portal
principal. 2.782 concesiones, 2.778 con coordenada de planta transmisora.

**La desagregación por CIIU**, en el mismo RETC que ya usábamos: 12.789
establecimientos con código de actividad. ⚠️ Sesgo declarado: el universo es
«quien declara emisión al aire», así que da 7 centros de datos y 5 radios — para
esos ítems no sirve.

## Lo que NO se resuelve con más búsqueda

1. **Industrial: el cuello de botella es la Matriz, no el dato.** Hay 1.514
   plantas manufactureras, 1.145 instalaciones fabriles y 1.049 forestales con
   coordenada, y los 44 ítems del sector sólo contemplan Maquinaria, Vehículos,
   Componentes Electrónicos, Equipos Médicos y Armamento. **Un aserradero, una
   cementera o una panadería industrial no tienen dónde entrar.**
2. **Defensa: 20 ítems no publicados oficialmente.** Ningún organismo chileno
   publica armamento, municiones ni instalaciones militares. Sólo entró lo que
   las empresas publican de sí mismas: FAMAE, ASMAR, ENAER, IDIC.
3. **Nuclear: no falta el dato, no existe el activo.** El Coordinador declara el
   tipo «Nuclear» y tiene cero centrales. La CCHEN no publica catastro.
4. **Gobierno: el Estado no se georreferencia.** Cero capas en el IDE Chile y en
   datos.gob.cl. SERVEL georreferencia por ley los locales de votación y no
   publica la capa; el Registro Civil está tras protección anti-bot.
5. **Hoteles (IRMD Alto):** SERNATUR declara 8.721 alojamientos formales y
   ninguno publica coordenada. Es gestión, no descarga.
6. **Componentes internos:** 18 de los 41 ítems de Educación y Comercial son
   aulas, baños, ascensores, proyectores. No tienen ni van a tener catastro.

## Límites de acceso respetados

- `www.ide.cl` bloquea explícitamente a ClaudeBot → no se tocó (se usó `geoportal.cl`)
- `datos.gob.cl` declara `Disallow: /api/` → no se usó la API
- `midas.minsal.cl` (farmacias MINSAL, 2.012 locales) prohíbe agentes de IA →
  **documentado y NO descargado**; lo tiene que bajar una persona
- `seia.sea.gob.cl` declara `Disallow: /` completo → no se tocó
- `siss.gob.cl` prohíbe la ruta de sus tablas → el dato se sacó del Geoportal
- 4 servicios de SUBTEL y la API de la CNE piden token. `bencinaenlinea.cl`
  lleva uno embebido en su configuración pública y **no se usó**: hay que pedir
  uno propio.

## Trampas técnicas anotadas

- **El ArcGIS del MOP ignora `resultOffset` sin dar error**: devuelve el mismo
  lote una y otra vez. Hay que paginar por `objectIds` y por POST.
- **Dos capas de SUBTEL devuelven las coordenadas como texto**, no como número.
  El bajador reventó — que es lo mejor que podía pasar: si hubiera devuelto
  vacío, 4.415 nodos habrían quedado «sin coordenada» en silencio.
- **El rótulo `region` de la CNE viene mal**: una instalación a −24,1° aparece
  como «Arica y Parinacota». Usar la coordenada, nunca el nombre.
- **El catastro cultural publica en UTM 19S**, no en grados.
- 60.489 de los 90.544 registros de Transporte son **líneas**, no puntos.
