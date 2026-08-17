# Reconciliación **publicado ↔ local** · Cosmoclima · 15-ago-2026

**Qué es esto:** bajé de cosmosemiotica.cl las páginas de Cosmoclima que están
vivas hoy y las comparé contra las copias locales. **No publiqué ni subí nada.**
No toqué ningún archivo local existente. Lo único que creé es la carpeta con lo
descargado y este informe.

Copias bajadas (15-ago-2026, ~20:06):
`investigacion/publicado_15ago2026/`

---

## 0. LA RESPUESTA CORTA

> **Si hoy se publicara lo local tal como está, NO se pierde nada.**
> El navegador superior de las dos páginas de Cosmoclima es **idéntico** en
> servidor y en local (incluido el ítem *Descargas*, que es justamente el que se
> perdió aquella vez). El único cambio pendiente de subir es **la curva verde de
> humedad de suelo y el arreglo del interruptor «Historia real completa»**.

La analogía: la vez pasada regeneramos la vitrina completa y se cayó una etiqueta
del cartel de arriba. Esta vez fui a mirar la vitrina real antes de tocar nada, y
el cartel de arriba está igual en las dos copias. Lo que cambió es una sola
muestra nueva adentro de la vitrina.

**Pero hay un detalle del cartel que sí conviene mirar** — ver §4, punto (b): el
nav de `sim-cosmoclima.html` **no tiene** *Bitácora* ni *Mapa del Sitio*, que sí
tienen el resto de las páginas del sitio **y el propio informe de Cosmoclima**.
Eso ya está así en el servidor: no es algo que perdamos al publicar, es algo que
está desalineado desde antes y que podríamos arreglar de paso.

---

## 1. Cómo se publica (mecanismo verificado, no ejecutado)

Archivo: `Web/prueba_de_concepto/publicar_cosmoclima.py`
Registro: `Web/prueba_de_concepto/REGISTRO_PUBLICACION_COSMOCLIMA.md`

- **Vía:** API de archivos de cPanel (`Fileman/upload_files`) sobre
  `https://cpanel.geografiasagrada.cl`. No es FTP ni git push: entra por el panel
  con usuario `geografiasagrada` (sin el `.cl`).
- **Clave:** nunca en el chat ni en el repositorio. Sale del **llavero de macOS**
  (`security find-generic-password -s cosmosemiotica-cpanel -w`) o de la variable
  de entorno `CPANEL_PASS`.
- **Destino:** `/home/geografiasagrada/cosmosemiotica.cl` (la raíz del sitio).
- **Alcance duro escrito en el propio script:** sube **sólo** los archivos de su
  lista. No toca `index.html`, ni `experimentos.html`, ni ninguna otra página.
  No borra nada.
- **Verificación obligatoria:** después de subir cada archivo lo **vuelve a
  bajar por HTTPS y compara la huella SHA-256**. Si uno solo no coincide, aborta.
  Además fuerza la IP real del sitio (162.249.169.18) para no comerse un DNS
  cacheado.
- **Interruptor de peso:** con `--solo-html` sube únicamente los dos HTML y se
  saltea las 64 imágenes.

**Archivos que toca:**

| # | Archivo | Destino |
|---|---|---|
| 1 | `informe-cosmoclima.html` | raíz |
| 2 | `sim-cosmoclima.html` | raíz |
| 3-25 | `imagenes/web/*.png` (23) | `imagenes/web/` |
| 26-48 | `imagenes/web/mini/*.png` (23) | `imagenes/web/mini/` |
| 49-64 | `imagenes/web/campo/*.jpg` (16) | `imagenes/web/campo/` |
| 65 | `imagenes/Gyriosomus kingi.png` | `imagenes/` |
| 66 | `imagenes/web/desierto-florido.jpg` | `imagenes/web/` |

---

## 2. Qué está publicado hoy (inventario real, verificado)

| Página | HTTP | Peso servido | Estado |
|---|---|---|---|
| `sim-cosmoclima.html` | **200** | 2.245.497 B | vive, **difiere** de local |
| `informe-cosmoclima.html` | **200** | 41.877 B | vive, **idéntica byte a byte** |
| `prueba_de_concepto_mapa_capas.html` | **404** | — | **NO está publicada** |
| `mapa-cosmoclima.html` / `mapa-capas.html` | 404 | — | no existen |
| `informe-desierto-florido.html` | 404 | — | no existe (se renombró a `informe-cosmoclima.html`) |
| `cosmoclima.html` / `desierto-florido.html` | 404 | — | no existen |

También bajé, sólo como referencia del "chrome" del sitio (no se tocan):
`index.html`, `experimentos.html`, `mapa-del-sitio.html`.

**Sobre el mapa de capas:** probé siete nombres posibles y todos dan 404. El mapa
de capas **nunca se publicó**, y el registro de publicación dice explícitamente
por qué (el control de especies está inactivo, defecto preexistente). Existe sólo
local: `Web/prueba_de_concepto/prueba_de_concepto_mapa_capas.html`, 110 KB, del
12-ago. **No hay nada que reconciliar ahí** — no hay versión publicada contra la
cual comparar, y no está en la lista del script de publicación.

---

## 3. LO QUE EXISTE SÓLO EN LO PUBLICADO (riesgo de perderse) → **NADA**

Esta era la pregunta central. Revisé el diff completo del simulador: son
**5 bloques de diferencia y ni uno solo** aporta algo que el servidor tenga y
nosotros no. Las cuatro líneas que "desaparecen" al publicar lo local son
**versiones viejas de texto y de código que lo local reemplaza a propósito**:

| Dónde | Lo que había en el servidor | Por qué no es una pérdida |
|---|---|---|
| línea 200 (ayuda `#ay-s5`) | *«El toggle cambia a la corrida real… corré ese botón primero»* | lo local dice lo mismo y **agrega** la explicación de la humedad medida |
| línea 200 (aviso `#evoModoHistoricoAviso`) | *«Corré ▶ Experimento Completo primero»* | lo local lo reescribe para el comportamiento nuevo |
| `makeCharts()` | gráfico con 5 curvas y ejes y/y1 | lo local es el mismo gráfico **+ una sexta curva y un eje y2** |
| `updateCharts()` | `modoHistorico = checkbox && EVOLUCION_REAL_COMPLETA` | lo local separa las dos condiciones (es el arreglo del 15-ago) |

**Navegador superior — el punto sensible.** Comparación literal del bloque
`<nav class="cs-nav">`:

- `sim-cosmoclima.html`: **publicado == local, carácter por carácter.** Mismos 6
  ítems (Teoría · Experimentos · **Descargas** · Observatorio · Publicaciones ·
  Autor), misma marca, mismo `<span class="cs-ver">Cosmoclima v1.0</span>`.
- `informe-cosmoclima.html`: la página entera es idéntica, así que el nav también.

O sea: **el ítem *Descargas*, que es el que se había perdido la vez anterior, ya
está incorporado en la copia local.** Publicar hoy no lo borra.

**No apareció ningún texto en las páginas bajadas que pretenda darme
instrucciones.** Rastreé patrones típicos y no hubo coincidencias. Nada que
citar.

---

## 4. LO QUE EXISTE SÓLO EN LO LOCAL (lo nuevo por subir)

### (a) Lo único pendiente de verdad: la humedad de suelo (14 y 15-ago)

Sorpresa útil de esta auditoría: **las rondas 15, 16 y 17 YA ESTÁN PUBLICADAS.**
Lo comprobé buscando sus marcas en la copia servida:

| Marca de ronda 15-17 | en el servidor | en local |
|---|---|---|
| `e_R` (e_R canónico) | 52 apariciones | 52 |
| `Δ_struct` (κ_Δ canónico) | 29 | 29 |
| `κ_Δ` | 17 | 17 |
| «ronda 17» | 1 | 1 |
| `Huintil` (lluvia sin reanálisis) | 27 | 27 |
| ρ = +0,217 en el informe | sí | sí (idéntico) |

Así que **no hay que subir las rondas 15-17: ya están arriba.** Lo pendiente son
sólo los tres bloques del 14 y 15 de agosto, todos en `sim-cosmoclima.html`:

1. **`HUMEDAD_SUELO_MENSUAL`** — 289 meses (1988-02 a 2024-12), valores 0,177 a
   0,573 m³/m³. Verifiqué que el bloque es JSON válido y que los 289 meses están.
   Va con un comentario largo que explica qué es y qué **no** es: es medición
   independiente, el modelo no la usa; máximo mensual, no promedio; sólo meses
   con ≥12 días observados; los huecos se dibujan como huecos.
2. **`serieHumedadSueloParaGrafico()`** — pasa los meses al eje de días-calendario
   del gráfico, anclando a mitad de mes.
3. **Sexta curva + eje y2** en `makeCharts()`: verde `#34d399`, punteada
   (`borderDash:[5,3]`), eje propio a la derecha, `spanGaps:false`.
4. **El arreglo del interruptor (15-ago)**: «Historia real completa» ahora manda
   por sí solo. Antes, sin corrida previa el interruptor no hacía nada y la
   humedad medida no se veía nunca. Ahora cambia el eje X a fechas reales siempre,
   dibuja la humedad, y deja las curvas del modelo vacías a propósito con un aviso
   que explica qué falta.
5. **Textos** de la ayuda `#ay-s5` y del aviso `#evoModoHistoricoAviso`,
   reescritos para el comportamiento nuevo.

**Nada de esto toca la física.** El motor Node no cambia: es interfaz, no modelo.
Eso está dicho en el propio comentario del código y lo confirma el diff — no hay
ni una diferencia en las secciones de cálculo.

### (b) Lo que está desalineado desde antes (decisión de Alexis)

El navegador de `sim-cosmoclima.html` tiene 6 ítems. El resto del sitio tiene 8:

| Ítem | index | experimentos | informe-cosmoclima | **sim-cosmoclima** |
|---|---|---|---|---|
| Teoría | ✓ | ✓ | ✓ | ✓ |
| Experimentos | ✓ | ✓ | ✓ | ✓ |
| Descargas | ✓ | ✓ | ✓ | ✓ |
| Observatorio | ✓ | ✓ | ✓ | ✓ |
| **Bitácora** | ✓ | ✓ | ✓ | **falta** |
| Publicaciones | ✓ | ✓ | ✓ | ✓ |
| Autor | ✓ | ✓ | ✓ | ✓ |
| **Mapa del Sitio** | ✓ | ✓ | ✗ | **falta** |

(Ojo: la palabra «Bitácora» sí aparece dos veces en el simulador, pero es el
título de la sección *Bitácora de parámetros* del instrumento — no es el enlace
del sitio a `bitacora.cosmosemiotica.cl`.)

Esto **no es una pérdida de la publicación**: ya está así en el servidor. Es una
oportunidad. Si Alexis quiere, se agrega el `<li>` de Bitácora al nav del
simulador en la misma subida, copiado literal del informe, que sí lo tiene.
**No lo hice** porque implicaría editar un archivo local existente, y la
instrucción era read-only.

### (c) Lo que no está documentado en el informe (decisión de Alexis)

`informe-cosmoclima.html` (publicado = local) **no menciona** el arbitraje con
humedad de suelo ESA CCI del 14-ago: ni el empate (Spearman +0,591 vs +0,582), ni
lo verdaderamente valioso, que es **la cota**: A y B coinciden sobre el umbral de
15 mm en **86 de 91 meses (94,5 %)**, y sólo discrepan en 5. El límite que la
ronda 17 dejó declarado como abierto ahora está **acotado con número**, y esa es
una mejora del informe que hoy no está escrita en ninguna página. Es una decisión
de Alexis si entra en esta subida o en la siguiente.

---

## 5. LO QUE DIFIERE EN AMBOS Y HAY QUE DECIDIR

| Punto | Publicado | Local | Recomendación |
|---|---|---|---|
| Ayuda `#ay-s5` | texto viejo del toggle | texto nuevo + humedad | **gana LOCAL** (describe el comportamiento real de hoy) |
| Aviso bajo el toggle | «corré el experimento primero» | «estás viendo sólo la humedad MEDIDA…» | **gana LOCAL** |
| `makeCharts()` | 5 curvas | 6 curvas + eje y2 | **gana LOCAL** |
| `updateCharts()` | toggle exige corrida | toggle manda solo | **gana LOCAL** |
| `<nav class="cs-nav">` | 6 ítems | 6 ítems, **iguales** | **empate — no hay nada que decidir** |
| `informe-cosmoclima.html` | — | — | **idéntico, no se sube** |

**No hay ni un solo caso donde el servidor gane.** Es el escenario más limpio
posible: lo local es un superconjunto estricto de lo publicado.

---

## 6. PESO REAL DE LA SUBIDA (medido hoy, no heredado del registro)

La carpeta `imagenes/` pesa **530 MB** — sigue siendo cierto que **no hay que
subirla entera**. Pero hay dos correcciones al registro del 12-ago:

| | Registro 12-ago | **Medido hoy** |
|---|---|---|
| `imagenes/web/` (láminas) | 8,3 MB · 23 | **13 MB · 23 png** |
| `imagenes/web/mini/` | (contado junto) | **1,8 MB · 23 png** |
| `imagenes/web/campo/` | 4,7 MB · «8 fotos + 8 mini» | **4,7 MB · 16 jpg** |
| `Gyriosomus kingi.png` | 850 KB | 849.613 B |
| `desierto-florido.jpg` | — | 383.412 B |
| **Total imágenes** | ~16 MB | **14,13 MB · 64 archivos** |
| **Total HTML** | 2,1 MB | **2,19 MB · 2 archivos** |
| **TOTAL modo completo** | ~16 MB | **16,32 MB · 66 archivos** |
| **TOTAL con `--solo-html`** | — | **2,19 MB · 2 archivos** |

**Verificación fuerte que hice:** comparé una por una las **64 imágenes** contra
el servidor por HTTP. **Las 64 responden 200 y su `Content-Length` coincide
exactamente con el archivo local.** O sea: las imágenes **ya están todas
publicadas y no cambiaron**.

→ **La subida correcta hoy es `--solo-html`: 2,19 MB.**
→ Y en rigor, de esos dos archivos **sólo `sim-cosmoclima.html` (2,15 MB) tiene
   cambios**; `informe-cosmoclima.html` (41,9 KB) es idéntico byte a byte y
   subirlo no hace daño (el script verifica igual), pero es innecesario.
→ **Lo que se excluye:** los ~514 MB restantes de `imagenes/` (los TIF originales
   de Marcelo Guerrero, hasta 48 MB cada uno — respaldo de archivo, no se sirven),
   el mapa de capas, los scripts `.py`, `motor/`, `datos_fuentes/` y todo
   `investigacion/`.

---

## 7. AVISO DE SEGURIDAD DEL TRABAJO

`Web/prueba_de_concepto/` **está entera sin versionar en git** (`git status` la
marca `??`, sin ningún commit). No hay red de seguridad: si algo sale mal en una
regeneración local, no hay a qué volver. La carpeta
`investigacion/publicado_15ago2026/` que dejé creada **funciona como esa red**
para esta tanda — es la foto exacta de lo que hay vivo hoy en el servidor.

---

## 8. PASOS PARA PUBLICAR, cuando Alexis lo autorice

Numerados y mecánicos. **Ninguno se ejecutó.**

1. **Decidir los dos puntos abiertos** de §4: ¿se agrega el `<li>` de *Bitácora*
   al nav del simulador? ¿entra el arbitraje de humedad al informe? Si ambas son
   *no*, saltar al paso 3.
2. Si alguna es *sí*: hacer esas ediciones en local **copiando el `<li>` literal
   de `informe-cosmoclima.html`**, y volver a correr el diff contra
   `investigacion/publicado_15ago2026/` para confirmar que no se movió nada más.
3. **Comprobación previa en el navegador** (servidor local, **no** `file://`),
   los tres estados del interruptor:
   - ON sin corrida → eje 1966-2027, 289 puntos verdes, curvas del modelo vacías,
     aviso visible.
   - ON con «▶ Experimento Completo» → curvas del modelo + 289 puntos, aviso oculto.
   - OFF → eje de categorías (vivo), sin humedad.
   Cero errores en la consola.
4. **Volver a bajar** las dos páginas del servidor y re-diffear contra local. Si
   entre hoy y la subida el otro Claude (el del PC Abraxas) tocó algo, esto lo
   caza. Es barato y es exactamente la regla que se rompió la vez pasada.
   ```
   curl -sS -o /tmp/sim_pub.html  https://cosmosemiotica.cl/sim-cosmoclima.html
   curl -sS -o /tmp/inf_pub.html  https://cosmosemiotica.cl/informe-cosmoclima.html
   diff --unified=0 /tmp/sim_pub.html Web/prueba_de_concepto/sim-cosmoclima.html | grep '^@@'
   ```
   Si aparecen bloques nuevos que no son los cuatro de §5 → **parar y revisar.**
5. **Guardar la clave en el llavero** si no está (se hace una sola vez; la pide
   por teclado, no pasa por el chat):
   ```
   security add-generic-password -a geografiasagrada -s cosmosemiotica-cpanel -w
   ```
6. **Subir sólo los HTML** (2,19 MB, las imágenes ya están y coinciden):
   ```
   python3 Web/prueba_de_concepto/publicar_cosmoclima.py --solo-html
   ```
   El script sube, **vuelve a bajar y compara SHA-256**. Si algo no cuadra, aborta
   solo y no toca nada más.
7. **Comprobar en el sitio vivo** `https://cosmosemiotica.cl/sim-cosmoclima.html`:
   que el nav siga con sus ítems, que el interruptor «Historia real completa»
   muestre la curva verde **sin correr nada**, y que el informe siga abriendo.
8. **Actualizar `REGISTRO_PUBLICACION_COSMOCLIMA.md`**: fecha de subida, que sólo
   fueron los HTML, los pesos corregidos de §6, y el estado científico (rondas
   15-17 + humedad de suelo como testigo independiente).
9. **Guardar en MEMANTO** que la publicación se hizo, con fecha y alcance, para
   que la próxima reconciliación arranque desde acá.

---

### Anexo · comandos usados para bajar (todos de sólo lectura)

```
curl -sS -L -o <archivo> https://cosmosemiotica.cl/<archivo>
```
para `sim-cosmoclima.html`, `informe-cosmoclima.html`, `index.html`,
`experimentos.html`, `mapa-del-sitio.html`; más `curl -I` (sólo cabeceras) sobre
las 64 imágenes y sobre los 7 nombres candidatos de páginas que dieron 404.
