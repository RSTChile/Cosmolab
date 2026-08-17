# Pendientes · sesión del 25 de julio de 2026

Anotado a mitad del barrido κ_H (iba en 31 de 60).
Agrupado por **quién lo desbloquea**, no por tema.

---

## 1 · Esperando la corrida

**Informe del EIT-3 Térmico.** Se escribe en dos partes separables:

- *Sin datos nuevos* — el encuadre. El Pastor sale del relato con su nodo (O-N17), entran
  las seis correcciones de junio, la oscilación de 540 pasos se relee como firma del
  controlador y no del sistema, y la correlación residual Δ↔LF de 0,2265 se declara como
  reducción y no como eliminación (n=220, t=3,43, p≈0,0006, IC 95 % [0,097 · 0,348] — no
  contiene el cero).
- *Con la corrida* — la sección de κ_H. Depende del barrido de 60 puntos.

**Molde a seguir:** el Bloque 24 del canon. Capas declaradas, criterios de retractación
numerados, no-correspondencias publicadas como hallazgos.

**Qué mirar cuando llegue:** si la huella se derrumba cerca de luminosity 1,83 —donde el
modelo predice la extinción— y si H cae junto con ella. Si caen juntas, ahí está κ_H. Si la
huella cae y H no, entonces powerLive no es el observable correcto, y eso también es un
resultado.

**Del barrido anterior, para comparar:** 40 puntos, todos viables, banda de conducta entre
13 y 47 veces bajo el propio ruido. Sistema vivo y opaco — el estado que C-N2.8.8b describe
como opacidad analítica.

---

## 2 · Esperando una decisión tuya

| pendiente | qué falta |
|---|---|
| **Canon cerrado / en desarrollo** | El sitio se contradice entre Teoría, Autor y Publicaciones. No es un error: es una posición editorial que tienes que fijar. |
| **Levitrón: informe 5 / simulador 4** | No encontré ningún v5 en el historial; lo último es el v4 del 20 de abril. O el 5 es numeración del informe —como pasó con el óptico— o existe un archivo que nunca subimos. |
| **Chip «Corrección pendiente v1.4»** | En el informe óptico. Solo tú sabes si sigue vigente. |
| **Nombres del CSV en castellano** | 84 % de las columnas está en inglés. Si las cambio, los CSV ya generados dejan de calzar. Opciones: tabla de equivalencias y conversión de los viejos, o exportar las dos versiones. |
| **Force HTTPS** | El API de cPanel responde éxito y no lo aplica. Hay que probarlo desde el panel, en Domains. |
| **Subtítulo del paper** | «Cosmosemiótica de las letras castellanas» queda fijo. Falta decidir si lleva subtítulo del tipo *«Qué le pasa a una distinción cuando se pierde el sonido que la sostenía»*, o va sin él. |

---

## 3 · Míos, listos para ejecutar

**Los (?) del simulador.** Es lo que pediste para poder mostrar la página sin acordarte de nada.

- Un `?` por control que **abre la explicación y la deja abierta** —no globito al pasar el
  mouse, que en proyección no sirve— más un interruptor «explicar todo» como modo demostración.
- Nueve glosas canónicas: κ_H, Λ, Δ_struct, A_sys-env, e_R, LF/mult, y los nodos O-N7,
  O-N4.1, O-N17. El símbolo se mantiene; la glosa se agrega al lado.
- Los tres nombres en inglés que se filtraron a la pantalla: `PowerBase`, `stress`, `envTemp`.
- Tarjeta de encabezado con la pregunta del experimento en tres frases.

**Regla de nombres, que queda fijada:** lo del canon se mantiene y se glosa. Lo del dominio
—PTC, Daisyworld, albedo— se mantiene y se glosa. **Todo lo que yo invente va en castellano
desde el principio**, en pantalla y en CSV.

**Correcciones sueltas del sitio:**

- El medidor sigue rotulado `LF_exp` mientras la métrica de al lado ya dice `mult`. Renombre a medias.
- `informe-anima.html` ofrece «Descargar ANIMA 1.0.4» y el servidor sirve 1.0.7.
- Deslizadores sin nombre accesible: 5 en el óptico, 8 en el dron, 11 en el Levitrón. El
  térmico ya los tiene todos. Necesito leer qué controla cada uno para nombrarlos bien.
- Desborde horizontal en móvil de los otros tres simuladores. El térmico ya quedó resuelto.

**Correcciones al canon** (dos, chicas y verificadas):

- **C-N2.8.11** escribe U_Cos con cinco elementos; C-N2.8.11a lo corrige a seis en el párrafo
  siguiente. El diccionario y la tabla 26 ya dicen seis. Falta alinear el nodo 11.
- El Bloque 24 dice que se incorpora «tras el Bloque 33 (Equivalencia de RC con Energética
  Semiótica)». Ese bloque es el **23**.

---

## 4 · Paper · «Cosmosemiótica de las letras castellanas»

Fuera de la Canónica y fuera del informe del térmico. La razón no es que sea erudito: es
que un caso aislado no distingue regularidad de coincidencia — el mismo estándar que aplicamos
hoy al barrido de un solo punto.

**Corpus, cinco casos:**

| caso | qué pasó | por qué sirve |
|---|---|---|
| **h** (f- latina) | La aspiración se pierde en el s. XV; la métrica de Garcilaso y Fray Luis sigue contándola | Paralelo exacto de la eta. El verso conserva la medida de un sonido que ya nadie pronunciaba |
| **b / v** | Dos fonemas distintos en el medieval, uno solo desde el s. XVI | El iotacismo del castellano: dos grafías, un fonema |
| **yeísmo** (ll / y) | Fusión desde la Baja Edad Media; el norte de España aún distingue *haya* de *halla* | **Tiene control**: la misma lengua con y sin fusión, simultáneamente |
| **sibilantes** | Reajuste de los s. XVI–XVII; desaparecen todas salvo /s/ | **Contraejemplo**: aquí la distinción no migró, se perdió. Sin este caso la tesis sería inrefutable |
| **eta** (Η) | Pierde la aspiración, se vuelve vocal, y el iotacismo la absorbe en la iota | Caso de apertura: hace visible la pregunta |

**Hipótesis falsable:** la distinción migra de sustrato cuando existe un soporte alternativo
ya en uso, y se pierde cuando no lo hay. Se falsa con un caso que tenga soporte disponible y
donde la distinción igual se haya perdido.

**Lo que quedó decidido y por qué:** la lectura del «desalmamiento gráfico» **no resiste** —
Ψ_alma es relacional por O-N3.4b y una letra no tiene un otro. Va publicada como
no-correspondencia, igual que EML-N2. Lo que sí resiste es la exaptación de sustrato
(O-N8.20): la distinción se mudó del oído al ojo. Criterio de retractación: si se mostrara
que ἡμεῖς/ὑμεῖς no sobrevive en la escritura, el nodo cae.

---

## 5 · Heredados del traspaso, siguen abiertos

- **INR tiene tres definiciones** en el corpus vigente. La aritmética exige que sea ruido o
  distancia, nunca no-ruido. Bloquea cualquier medición del CCD.
- **IAS está duplicado** — «Anomia Social» y «Acoplamiento Señal/Ruido». Usar siempre `IAS_X`.
- **Prueba de simetría del índice.** Es la que puede tumbarlo. Dos textos que reescriban la
  misma narrativa en direcciones opuestas, densidad equivalente.
- **Control de densidad contra anclaje.** El trío Austin → Searle → Derrida es la prueba más
  limpia disponible.
- **Codificación ciega.** Quien codifica σs y κs decide el resultado.
- Segundo caso MAPAR-S, anunciado y nunca subido.
- Citas por verificar en fuente primaria: `arXiv:2501.12345` (marcador genérico, no un
  preprint real), Chandra et al. y Odrzywołek, y Mary Poovey en *Feminists Theorize the Political*.

---

## 6 · Publicado hoy, verificado por huella contra el servidor

| archivo | qué cambió |
|---|---|
| `informe-cosmorobot.html` | Video de 22 MB entre la bajada y la portada, 964,8 px, póster propio |
| `teoria.html` | Video Big-Bang. Remuxeado con faststart —flujos idénticos por MD5— y después recodificado: 248 → 92 MB, picos de 9,57 → 4,26 Mbps, PSNR 42–45 dB en los tres momentos más exigentes |
| `index.html` | Pie ampliado: Nido de Cóndores, correo, teléfono, derechos reservados y contador |
| 13 páginas | Metadatos completos —no había ninguno—, rótulo «Introducción a la Teoría», enlace azul de VSTCosmo, desborde a 320 px |
| `sim-eit3-optico.html` | Rótulo unificado en v1.3, con comentario de procedencia de la compilación |
| `sim-eit3-termico.html` | **v7 completo**: doble escala, autocalibración local, barrido, bitácora. Más los tres injertos de julio y el desborde resuelto en los seis anchos |
| `contador.php` | Guarda un entero y nada más. Sin IP, sin agente, sin hora. En cero desde hoy |

**Entregado y sin publicar:** `EIT3_Termico_kappaH_v7.1.html` — corrige el piso de silencio,
que daba exactamente cero en los 40 puntos porque la entropía del ruido se calculaba sobre la
ventana de la conducta y las dos series no se solapan. Probado con tus trazas: el piso pasa de
0,0000 a ~4,34 sobre un máximo de 4,585.

---

## 7 · Datos de acceso, para no volver a buscarlos

- Usuario cPanel: **`geografiasagrada`**, sin `.cl`. Con `.cl` da `invalid_login`.
- Puerto 2083 bloqueado. Entrar por `https://cpanel.geografiasagrada.cl`.
- Document root: `/home/geografiasagrada/cosmosemiotica.cl`. **No** `/public_html`.
- **No se puede borrar nada por la API.** Solo sobrescribir.
- DNS mal cacheado desde el entorno: `--resolve cosmosemiotica.cl:443:162.249.169.18`.
- Cuota tras todo lo de hoy: **2.601 de 3.001 MB**. Quedaban 399 libres.
- La clave viajó por el chat de hoy.

---

## 8 · Parámetros de la corrida en curso, para poder repetirla

```
Barrido:  eje luminosity · rango 0,25 → 1,95 · 60 puntos
          settle 300 · measure 120
Sensor:   Tc PTC = 18 · Exponente PTC = 4,1
Sistema:  potencia base 0,47 · beta 0,94 · sigma 6,8 · ruido 0,0079 · band 1,105 · tOpt 25
Día/noche: APAGADO
```

**Por qué el ruido no se toca:** el término de muerte del Daisyworld es `0,28 + ruido × 10`.
Moverlo corre las fronteras de extinción, que es justo lo que estamos midiendo. Y bajarlo no
mejoraría la razón conducta/ruido, porque el piso y la dinámica llevan el mismo parámetro.

**Fronteras que predice el modelo**, simuladas con sus propios albedos y su término de muerte:
extinción por frío en **0,300**, por calor en **1,831**.

**Al exportar:** los tres archivos —barrido, trazas y bitácora—, y subir «Trazas crudas» de 6
a 20, que es el control que limitaba cuántos puntos guardaban traza completa.
