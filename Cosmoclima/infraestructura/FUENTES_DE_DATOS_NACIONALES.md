# Fuentes de datos nacionales: el consolidador multi-amenaza

Instrucción de Alexis (15-ago-2026):

> El proyecto debe tener capacidad de tomar datos públicos de todos estos
> organismos asociados, para **consolidarlos en un único modelo nacional**, que
> es la idea del proyecto.

Esto define la arquitectura: **la Matriz es un consolidador**. Cada organismo es
dueño de su amenaza y publica su propio dato; nadie los cruza con el inventario
de infraestructura. Ese cruce es el producto.

---

## El mapa: una amenaza, un organismo, un adaptador

Las nueve variables de riesgo del glosario oficial tienen cada una su organismo
técnico. El proyecto necesita **un adaptador por fuente**.

| # | Variable de riesgo (glosario) | Organismo | Estado del acceso |
|---|---|---|---|
| 2.1 | Sísmico | **CSN** — Centro Sismológico Nacional | ✅ portal responde · API oficial por verificar |
| 2.2 | Tsunami | **SHOA** — Serv. Hidrográfico y Oceanográfico de la Armada | ⚠️ portal devuelve 403 al acceso programático |
| 2.3 | Volcánico | **SERNAGEOMIN** — RNVV / OVDAS | ✅ portal responde · formato por verificar |
| 2.4 | **Remoción en masa** | **SERNAGEOMIN** — Minuta Técnica | ✅ portal responde · **prioridad máxima** (ver abajo) |
| 2.5 | **Meteorológico** | **DMC** — Dirección Meteorológica de Chile | ⚠️ portal bloquea acceso directo · `archivos.meteochile.gob.cl` **sí responde** |
| 2.6 | Incendio forestal | **CONAF** | ✅ portal responde · formato por verificar |
| 2.7 | Materiales peligrosos | varios | ❓ sin explorar |
| 2.8 | Biológico | MINSAL / ISP | ❓ sin explorar |
| 2.9 | Depósitos y tranques de relaves | **SERNAGEOMIN** | ❓ sin explorar |
| — | Recursos hídricos | **DGA / MOP** | ✅ portal responde · formato por verificar |
| — | Marítimo / borde costero | **DIRECTEMAR** | ❓ sin explorar |
| — | Síntesis y alerta | **SENAPRED** | Es el **destinatario**, no una fuente |

Verificado el 15-ago-2026 con petición HTTP simple. «Portal responde» significa
exactamente eso: que el sitio contesta. **No** significa que haya dato
estructurado ni permiso de uso automatizado — eso hay que verificarlo fuente por
fuente antes de construir cada adaptador.

## Lo único que ya está verificado con dato en mano

**Sismos, en JSON, ahora mismo:**

```
GET https://api.gael.cloud/general/public/sismos
→ [{"Fecha":"2026-08-15 20:45:36","Profundidad":"91","Magnitud":"2.8",
    "RefGeografica":"65 km al E de Quillagua", ...}]
```

⚠️ **Advertencia importante:** `api.gael.cloud` es un **tercero**, no el CSN.
Sirve para prototipar, **no** para un instrumento que alimente decisión pública.
El canal oficial del CSN es su propio archivo diario
(`/sismicidad/catalogo/AAAA/MM/AAAAMMDD.html`, verificado hasta el año 2000).

### CSN — condiciones de uso, y la decisión de Alexis (16-ago-2026)

El CSN autoriza el uso de sus datos para **fines académicos y de divulgación**;
cualquier otro fin requiere aprobación expresa por escrito.

**Alexis autorizó usarlos**, en tanto este es un proyecto académico. Se procede
sobre esa base, citando siempre al CSN como fuente.

Queda anotada una salvedad que conviene tener presente, no para frenar nada sino
para no llevarse una sorpresa más adelante: el destino declarado del instrumento
es servir de insumo a decisiones del COGRID y de SENAPRED. El día que el
instrumento pase de investigación a apoyo operativo, el encuadre académico
podría dejar de cubrirlo. **Pedir la autorización por escrito ahora es barato y
evita tener que rehacer el trabajo después** — y es un trámite que corresponde a
Alexis, no al proyecto.

**Eventos meteorológicos DMC:** `archivos.meteochile.gob.cl/portaldmc/AAA/doc/evento_AA###_AAAA.php`
responde. Es el documento vivo del evento (ej. `AA139_2026`: viento 90-110 km/h
en cordillera del norte, vigente 15→18 de agosto de 2026). Formato HTML, no
estructurado: habría que parsearlo, o buscar el servicio de datos abiertos
oficial de la DMC.

## Prioridad de integración

No todas valen lo mismo para este proyecto. Orden propuesto:

1. **SERNAGEOMIN · remoción en masa** — máxima prioridad. Publica peligro de
   aluviones y derrumbes en **tres niveles (Alta/Moderada/Baja) por zona
   geográfica y con vigencia temporal**: es un `FEN` dinámico ya operativo. Sirve
   de referencia *y* de validación independiente de nuestro `C_clim`.
2. **DMC · avisos y alertas** — es el disparador de la cadena. Además da los
   umbrales cuantitativos por zona y día.
3. **DGA** — caudales y embalses, para `EstHidric` y `D_uso`.
4. **CONAF** — para `RIncFor`, que hoy está a medias.
5. **CSN, SHOA, SERNAGEOMIN volcánico** — otras amenazas. No son clima, pero el
   consolidador nacional las necesita y la MICR ya las contempla vía `FEN`.

## Cómo debe estar construido el consolidador

Tres reglas de diseño, para que esto no se vuelva un enredo:

**1. Un adaptador por fuente, aislado.** Cada organismo cambia su formato cuando
quiere. Un adaptador que se rompe no puede tumbar el modelo: debe degradar a
«sin dato» y decirlo, nunca inventar ni interpolar en silencio.

**2. El dato crudo se guarda tal como llegó.** Con fecha de descarga y URL. Si
después hay que auditar una decisión, tiene que poder reconstruirse exactamente
lo que el modelo vio ese día. Es la misma disciplina de Cosmoclima, y acá pesa
más porque el destinatario es una decisión pública.

**3. Nada se consolida sin declarar su confianza.** El propio MACC lo exige
(«bloqueo: no usar coeficiente si la confianza del dato es baja»). Una fuente
que no responde no es un cero: es un hueco, y el hueco se declara.

## Lo que este proyecto NO es

No es un sistema de monitoreo. **No compite con ninguno de estos organismos** ni
duplica su trabajo: cada uno es dueño de su amenaza y sabe medirla mucho mejor
que nosotros. El proyecto toma lo que ellos ya publican y lo cruza con el
inventario de infraestructura crítica, que es lo que nadie hace.

## Pendientes

- Verificar el **canal oficial del CSN** (no el tercero).
- Ver si la **DMC** tiene servicio de datos abiertos con registro, en vez de
  parsear HTML.
- Ver si la **Minuta Técnica de SERNAGEOMIN** tiene acceso programático **y
  archivo histórico** — sin historia no sirve para validar.
- Resolver el **403 del SHOA**.
- Revisar, para cada fuente, las **condiciones de uso** antes de automatizar
  descargas. Dato público no siempre significa uso automatizado permitido.
