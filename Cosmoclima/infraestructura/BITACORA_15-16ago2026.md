# Bitácora · noche del 15 al 16 de agosto de 2026

Ordenada **por importancia, no cronológicamente**. Lo primero es lo que más te
conviene mirar.

---

## 1 · Lo más importante: el método falló una prueba, y está bien que la fallara

Corrí las pruebas del protocolo —escrito antes de calcular, como quedó acordado—
y el resultado es mixto. Lo reporto tal cual.

| Prueba | Resultado |
|---|---|
| 1 · Ancla Copiapó 2015 | **NO PASA** |
| 2 · Separación territorial | **PASA** (rango 0,112 → 0,625) |
| 5 · Contra fallas eléctricas reales | **Corrida, sin separación convincente** |

### Por qué falló la prueba 1, y qué significa

Copiapó, marzo de 2015, **se encendió perfecto: 0,995**. El aluvión aparece.

El problema es el otro lado: **Punta Arenas ese mismo mes marcó 0,755**, cuando
tenía que quedarse callado. Y peor: un mes tranquilo de Copiapó (agosto 2015)
marcó **0,841**, más alto que Punta Arenas en el mes del aluvión.

La causa es de diseño y la entendí recién al ver esto:

> La variable que construí mide **rareza**, no **peligro**. Y en el desierto todo
> lo que se moja es raro.

En Copiapó llueven ~12 mm al año. Ocho milímetros en agosto son un evento de +3
sigmas —rarísimo— y no le hacen nada a nadie. Ciento cuatro milímetros en 48
horas destruyen la ciudad. Mi número los pone casi al mismo nivel, porque los dos
son «muy raros para ese lugar».

**Lo que falta es la magnitud absoluta.** El peligro real necesita las dos cosas:
que sea raro *y* que sea grande. Y no es una ocurrencia mía — es exactamente lo
que el canon ya dice: `InEvExtre` de MACLIMA combina el evento con un factor de
exposición, y la definición de aluvión (`EAL`) exige «precipitación intensa **más**
condición de suelo». Yo implementé sólo la mitad.

**No ajusté el umbral para que pasara.** El protocolo lo prohíbe expresamente y
además habría sido justo el error que el protocolo existe para evitar. La
corrección propuesta —combinar anomalía relativa con magnitud absoluta— está
diagnosticada pero **no aplicada**: es un cambio de modelo y lo decidís vos.

### La prueba 5, en limpio

Con 1.324 meses-comuna con falla eléctrica registrada contra 3.236 sin falla:

| | n | anomalía media |
|---|---|---|
| con falla | 1.324 | 0,4652 |
| sin falla | 3.236 | 0,4559 |
| **diferencia** | | **+0,0094** |

Contra los brazos nulos: **NULL-1 (fechas barajadas) p = 0,18** — nada.
**NULL-2 (activos barajados) p = 0,052** — al borde, y sin significado sólido.

Dicho derecho: **la anomalía mensual de lluvia en el punto de la subestación no
predice las fallas eléctricas de su comuna.** Puede ser por la misma falla de
diseño de arriba, porque el mes es una ventana demasiado gruesa para un temporal
de dos días, o simplemente porque la mayoría de las 10.150 fallas eléctricas de
diez años son de equipo y no de clima. No lo sé todavía, y no lo voy a decidir
sin datos.

**Lo bueno:** la maquinaria completa funciona de punta a punta, y ya tenemos con
qué equivocarnos y darnos cuenta. Eso es lo que no había ayer.

---

## 2 · Lo que el catastro encontró (seis agentes, todo verificado)

Chile tiene muchísimo más dato del que suponíamos, y varias piezas que
buscábamos existen y están abiertas.

### Las que desbloquean el proyecto

| Hallazgo | Estado |
|---|---|
| **Minuta de SERNAGEOMIN** — no es un PDF, es un servicio ArcGIS que entrega GeoJSON con polígonos y nivel | ✅ bajada |
| **119 zonas morfoclimáticas** — la geografía en que Chile declara la amenaza | ✅ bajada, resuelve las dos geografías |
| **345 comunas con CUT**, del mismo servicio (calzan por construcción) | ✅ bajada, resuelve **H-13** |
| **SENAPRED, 50.457 eventos 2015-2024 por comuna**, con 10.150 fallas eléctricas | ✅ bajada, **es la capa de validación** |
| **380 eventos ReTeRM** de remoción en masa reales, 1996-2026 | ✅ bajado |
| **Inventario vial del MOP**: 14.039 tramos con rol y km, 6.742 puentes, 6.141 emergencias viales | ✅ verificado, sin bajar |
| **SEC**: clientes sin luz **por comuna y por hora**, ≥6 años | ✅ verificado, sin bajar |
| **Coordinador Eléctrico**: 1.269 subestaciones reales (teníamos 39 = 3%) | ✅ verificado — **pero sin coordenadas** |
| **API oficial de la DMC** (`climatologia.meteochile.gob.cl`), 27 servicios | ✅ verificado, responde anónimo |

### ★ Lo urgente que hay que empezar YA

**Ninguna de las capas vivas guarda historia.** La minuta de SERNAGEOMIN se
sobrescribe a sí misma, y los avisos vencidos de la DMC desaparecen del índice.
Cada día que pasa sin guardar una foto es un día que no vamos a poder recuperar
nunca, y sin historia no hay forma de validar nada contra lo que la fuente decía
en su momento.

**Ya guardé la primera foto, la de hoy** (`datos/crudo/sernageomin/2026-08-15/`).
Hay que correr `traer_capas_sernageomin.py` todos los días. Es una línea y vale
más que casi todo lo demás.

### Tres cosas que corrigen supuestos nuestros

1. **Los niveles de peligro son CUATRO, no tres**: Baja / Moderada / Alta / **Muy
   Alta**. Tanto MACLIMA como la Matriz hablan de tres. Calibrar el `FEN` contra
   tres dejaría «Alta» y «Muy Alta» pegadas — justo la distinción que más importa
   para priorizar. Ya lo dejé corregido en el código (`peligro_4`).
2. **La DMC tiene tres escalones, no dos**: Aviso, Alerta y **Alarma**.
3. **El CUT hay que guardarlo siempre como texto de 5 caracteres.** El INE lo
   publica como entero (`2101`) y SENAPRED como texto (`'05303'`). Guardado como
   número, todas las comunas de la I a la IX pierden el cero de adelante y dejan
   de cruzar **en silencio**, sin error ni aviso. Ya está resuelto en el código.

### Dos errores en la sub-matriz de subestaciones

Los encontró el control independiente: la sub-matriz trae región y provincia
escritas a mano, y yo las derivé por coordenada. Coinciden en 34 de 39. De las
cinco restantes, tres son variantes de ortografía («Coihaique»/«Coyhaique»), pero
**dos son errores de verdad**:

| Subestación | Dice la sub-matriz | Dice la coordenada |
|---|---|---|
| Nueva Pozo Almonte | provincia **Iquique** | provincia **Tamarugal** |
| Escondida | provincia **El Loa** | provincia **Antofagasta** |

No corregí nada. Están anotados como **H-16**.

Y una más: **la coordenada de la Subestación Valparaíso no cae dentro de ninguna
comuna** — probablemente está en la bahía. Queda como **H-17**, sin comuna
asignada. No la inventé.

---

## 3 · Lo que quedó funcionando y probado

| Pieza | Estado |
|---|---|
| `esquema.py` — las dos tablas más la de huecos | **14/14 pruebas** |
| `normalizar.py` — la función `f(·)` común | verificada, con H-07 aplicado |
| `territorio.py` — el traductor de las dos geografías | **9/9 pruebas** (huecos e islas incluidos) |
| `adaptadores/era5.py` | 34.320 observaciones |
| `adaptadores/senapred_eventos.py` | 23.909 observaciones |
| `resolver_comunas_subestaciones.py` | 38/39 resueltas |
| `validar.py` | corre las 3 pruebas posibles |
| `PROTOCOLO_VALIDACION.md` | escrito **antes** de calcular |

Dos decisiones de implementación que conviene que sepas:

**El anclaje territorial es «nativo + derivado», no doble.** El plan pedía exigir
zona geográfica *y* comuna en cada dato. Al implementarlo apareció el problema
real: SERNAGEOMIN publica por zona y la CGE por comuna. Exigir las dos al
insertar obligaría a inventar la que falta. Quedó: se exige la que el organismo
publicó, y la otra se completa después, marcada como derivada.

**Bajé las capas generalizadas, no al máximo detalle.** Con todo el detalle, la
capa de zonas pesaba 103 MB y la de comunas no terminaba de bajar en 15 minutos.
Para saber en qué comuna cae una subestación, ese detalle no cambia nada. El
crudo de máximo detalle quedó archivado igual.

---

## 4 · Restricciones de uso que respeté (y que conviene tener presentes)

Verificado fuente por fuente. Esto acota lo que se puede automatizar:

| Fuente | Restricción | Qué hice |
|---|---|---|
| **CSN** (sismos) | Autoriza sólo fines académicos y de divulgación; otro uso exige permiso **escrito** | No automaticé nada |
| **SHOA** | `robots.txt: Disallow: /` para todos | No lo toqué |
| **archivos.meteochile.gob.cl** | `robots.txt: Disallow: /` — y ahí viven los avisos y la capa de zonas de la DMC | No automaticé. Uso la API oficial, que sí permite |
| **CGE** | Sus términos prohíben reproducir el contenido sin autorización escrita | No la usé. Uso la **SEC**, que es organismo público |
| **EFE** | Prohíbe explícitamente a ClaudeBot | No lo tocamos |
| **datos.gob.cl** | Prohíbe `/api/`, con Crawl-Delay 10 | Sólo consultas puntuales |
| **Red Vial MOP** | Declarada Creative Commons **no comercial** | Anotado; conviene confirmarlo con la UGIT |
| **INE** | Se contradice: su sitio dice CC BY-SA, el catálogo nacional dice CC BY-NC | Anotado; conviene pedirlo por escrito |

**Privacidad:** el Excel de SENAPRED trae una columna de texto libre con RUT y
descripciones de personas fallecidas. El adaptador **no la lee nunca** — está en
una lista de columnas prohibidas, no es una omisión sino una regla. Se trabaja
sólo con conteos agregados por comuna.

---

## 5 · Lo que tenés que decidir vos

1. **H-05** — el Listado Blanco de MACC sigue bloqueando la tabla de
   coeficientes. Mi recomendación sigue siendo reescribirlo apuntando a las
   siglas reales del catálogo, no crear variables nuevas.
2. **La corrección del método** que salió de la prueba 1: combinar anomalía
   relativa con magnitud absoluta. Diagnosticado, no aplicado.
3. **`PENDIENTE_K` y `CENTRO`** en `normalizar.py`: definen cuán nervioso es el
   instrumento. Sin calibrar, y no los voy a calibrar sin dato de validación.
4. **Las dos discrepancias de la sub-matriz** (H-16): ¿manda la coordenada o el
   dato escrito?
5. **Cartas que convendría mandar** — no mandé ninguna: al CSN pidiendo permiso
   de uso, a la DMC preguntando por salida CAP, al INE pidiendo condiciones por
   escrito. Todas requieren tu firma, no la mía.

---

## 6 · Lo que NO hice, a propósito

- No modifiqué la Matriz ni el Módulo.
- No publiqué ni subí nada a ningún lado.
- No emití ni simulé alertas.
- No automaticé ninguna descarga con términos que lo prohibieran.
- No recolecté datos de personas ni domicilios.
- No ajusté ningún umbral para que una prueba pasara.
- No fijé ningún veredicto.
