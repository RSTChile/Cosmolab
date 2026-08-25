# ¿Tiene el invierno una firma propia en los cortes de luz?

**Prueba de la hipótesis del director contra el dato horario de la SEC**

Proyecto: Infraestructura Crítica × Clima (RMD 2.0)
Fecha de captura del dato: **16-ago-2026**
Adaptador: `adaptadores/sec_cortes.py` · Dato: `datos/sec_cortes.csv` · Crudo: `datos/crudo/sec/2026-08-16/`

---

## 1. La hipótesis que se pone a prueba

El director planteó:

> «En pleno invierno, por el aumento de demanda eléctrica (calefacción), un
> transformador puede recalentarse e incendiarse (están llenos de aceite) y
> cortar el suministro.»

Es decir, el invierno atacaría a la red **por dos vías a la vez**:

- **desde afuera** — el temporal voltea postes, cae un árbol sobre la línea, el viento hace chocar los conductores;
- **desde adentro** — la calefacción eléctrica infla la demanda, el transformador se calienta más de lo que puede disipar, y falla.

Si eso es cierto, **los cortes de invierno tienen que verse distintos de los de verano**. Este informe busca esa diferencia.

---

## 2. Con qué dato, y qué NO es ese dato

La SEC publica, sin registro ni clave, cuántos clientes están sin suministro eléctrico **en cada comuna, hora por hora**, sumando todas las distribuidoras del país.

Endpoint: `POST https://apps.sec.cl/INTONLINEv1/ClientesAfectados/GetPorFecha`
Cuerpo: `{"anho":2025,"mes":7,"dia":10,"hora":3}`
Respuesta: `[{"NOMBRE_REGION":…,"NOMBRE_COMUNA":…,"CLIENTES_AFECTADOS":…}, …]`

**Lo más importante de esta sección son los límites, porque de ellos sale todo lo que este informe NO puede concluir:**

1. **No es un registro de eventos: es una foto por hora.** La SEC nunca dice «corte tal, empezó 19:12, duró 3 h». Dice «en la comuna X, a las 19:00, había N clientes sin luz». La noción de «evento» que usamos aquí es **una construcción nuestra**, definida en §4.
2. **No dice por qué se cortó.** Un transformador quemado y un camión contra un poste producen exactamente la misma fila. **Esta es la limitación decisiva: la hipótesis del director no se puede confirmar directamente con esta fuente.** Sólo se pueden buscar sus *huellas* — a qué hora empiezan los cortes, de qué tamaño son, cuánto duran — y ver si aparecen o no.
3. **No dice qué activo falló.** Da la consecuencia (clientes), no la causa (transformador, alimentador, línea).
4. **Cuenta clientes, no personas.** Un «cliente» es un empalme: una casa, un hospital y una minera valen 1 cada uno.
5. **La granularidad es la hora en punto.** Un corte de 40 minutos entre dos horas en punto puede no existir para esta fuente.

**Privacidad:** se usa exclusivamente el agregado por comuna. No se consultó, ni se guardó, ningún dato de persona o de número de cliente (los mapas de las distribuidoras sí permiten eso; no se tocaron).

**Condiciones de uso verificadas el 16-ago-2026:** `apps.sec.cl/robots.txt` → 404 (el dominio no declara restricción alguna); `www.sec.cl/robots.txt` → `Allow: /`, sólo bloquea `/sitio-web/wp-admin/`. La SEC es órgano público (Ley 20.285). Ritmo autoimpuesto: **1 petición por segundo, secuencial, un solo hilo**, con reintentos espaciados.

---

## 3. Qué se bajó, y por qué esa muestra y no otra

Para medir **duración** y **hora de inicio** hacen falta horas *consecutivas*: una foto suelta no sirve. Pero barrer el calendario completo serían ~58.000 peticiones sobre un servicio público, y la regla del propio proyecto (ficha A1, punto 10) lo prohíbe.

Solución: **bloques de 3 días completos (72 horas seguidas)**. Dentro de un bloque se ve a un corte nacer, crecer y morir.

| Parámetro | Valor | Por qué |
|---|---|---|
| Estratos | invierno austral (jun-jul-ago) · verano austral (dic-ene-feb) | son las dos estaciones que la hipótesis contrasta |
| Bloques | 2 por estación y por año, **14 + 14 = 28** | que ningún año pese de más |
| Años | 2020-2026 (invierno) · 2020-2026 (verano, nombrado por su enero) | tramo de historia continua verificado |
| Horas | 28 × 72 = **2.016 peticiones** | |
| Selección de días | **al azar, con semilla fija 20260816** | **crítico:** si eligiéramos los días de temporal, el invierno saldría peor por construcción. Se sortean días cualesquiera. |

Extensión de la historia, sondeada el 16-ago-2026 (día 15 de cada trimestre, 12:00): responde desde **2018-01**, hay un **hueco entre 2018-10 y 2019-07** (devuelve vacío), y desde **2019-10** responde de forma continua. La muestra usa sólo el tramo continuo.

---

## 4. CRITERIO — fijado ANTES de mirar ningún resultado

> Esta sección se escribió y se guardó **antes** de correr el análisis. No se modifica después. (Regla 2 del proyecto.)

### 4.1 Qué cuenta como «evento»

> Un **evento** es una racha máxima de horas consecutivas en las que una misma comuna tuvo **≥ 100 clientes** sin luz.

- **Umbral = 100 clientes**, fijado de antemano y con razón física: un transformador de distribución en Chile alimenta del orden de 50-300 clientes, así que 100 es aproximadamente «al menos un transformador». Además resuelve el problema práctico de que las comunas grandes casi nunca bajan a cero — sin umbral, toda la Región Metropolitana sería un único corte eterno.
- **Censura:** si una racha toca la primera o la última hora del bloque, no se le vio el principio o el final. Esos eventos **se descartan por completo** y se informa cuántos fueron. Se prefiere perder casos antes que inventarles duración.
- De cada evento se miden tres cosas: **pico de clientes**, **hora de inicio** (0-23, hora local de Chile) y **duración en horas**.

### 4.2 Las tres preguntas y sus pruebas

| # | Pregunta | Medida | Prueba | Se declara diferencia real si… |
|---|---|---|---|---|
| **P1** | ¿Los cortes de invierno afectan a más clientes por evento? | pico de clientes por evento | Mann-Whitney U (bilateral) | **p < 0,01 Y razón de medianas ≥ 1,5** |
| **P2** | ¿Difiere la hora del día en que ocurren? | hora de inicio del evento | χ² sobre la ventana vespertina | **p < 0,01 Y diferencia de proporción ≥ 3 puntos** |
| **P3** | ¿Difiere la duración? | horas por evento | Mann-Whitney U (bilateral) | **p < 0,01 Y razón de medianas ≥ 1,5** |

Se exige **las dos condiciones a la vez** (significancia *y* tamaño) a propósito: con miles de eventos, cualquier diferencia ridícula sale «significativa». Un p pequeño solo no es un hallazgo.

**Ventana vespertina de demanda: 18:00-21:59**, fijada de antemano. Es la hora en que en Chile se junta la gente en la casa, se enciende la calefacción y la iluminación, y el sistema eléctrico hace su punta diaria de invierno.

### 4.3 Qué resultado apoyaría a cada mecanismo

Esto también se fija antes, porque es lo que hace falsable la hipótesis. Los dos mecanismos predicen cosas **distintas**:

| | **Sobrecarga por demanda** (hipótesis del director) | **Daño externo por temporal** |
|---|---|---|
| **P1 magnitud** | eventos **más bien chicos** — un transformador son cientos de clientes, no cientos de miles | eventos **más grandes** en invierno: cae una línea y se va media comuna |
| **P2 hora** | **exceso vespertino en invierno** (18-21 h), siguiendo la punta de demanda | inicios **repartidos o de madrugada**, siguiendo el paso del frente, sin relación con las 18-21 h |
| **P3 duración** | **cortas o medianas** — se repone el fusible o se reemplaza el equipo | **más largas en invierno**: hay que salir a reparar con temporal encima |

Por lo tanto, y esto es lo que se juzgará:

- **La hipótesis del director queda APOYADA** si P2 muestra exceso vespertino de inicios en invierno (y cumple el umbral), **especialmente** si P1 no muestra que los eventos de invierno sean mucho más grandes.
- **La hipótesis del director queda SIN APOYO** si P2 no muestra exceso vespertino, mientras P1 y/o P3 sí muestran invierno mayor: ese patrón es la firma del daño externo, no de la sobrecarga.
- Si el resultado **no distingue** entre ambos, o si los n son insuficientes, se declara **«sin poder estadístico»** y no se concluye. (Regla 1.)

**Aviso que vale para todo lo que sigue:** aunque P2 salga como predice la hipótesis, **eso no probaría que los transformadores se están incendiando**. Sería consistente con ella, nada más. La fuente no trae causas (§2.2).

---

## 5. Resultados

### 5.0 Qué se consiguió

| | |
|---|---|
| Peticiones hechas | **2.016** (las 2.016 planificadas) |
| Peticiones fallidas | **0** |
| Crudo guardado | 28 MB en `datos/crudo/sec/2026-08-16/` (un JSON por hora + bitácora de peticiones) |
| Días cubiertos | **84** (42 de invierno, 42 de verano) |
| Comunas vistas | **330** |
| Filas del panel comuna×hora | **304.419** (`datos/sec_cortes.csv`) |
| Denominador nacional | 8.156.814 clientes (`GetClientesNacional`, mismo día) |
| Eventos construidos (§4.1) | **7.658 usables** — 4.145 invierno, 3.513 verano |
| Eventos descartados por censura | 746 |

El endpoint respondió mejor de lo que la ficha del proyecto registraba: **~0,15 s** por petición, no 30-100 s. La historia se sondeó y llega hasta **2018-01**, con un hueco entre 2018-10 y 2019-07; desde 2019-10 es continua.

### 5.1 P1 — ¿Afectan a más clientes por evento?

| | invierno | verano |
|---|---|---|
| n eventos | 4.145 | 3.513 |
| **pico mediano** | **248 clientes** | **269 clientes** |
| media | 754 | 899 |
| p90 | 1.656 | 2.218 |
| máximo | 50.714 | 42.667 |

**Razón de medianas invierno/verano = 0,922** · Mann-Whitney p = 0,000388

Criterio (§4.2): p < 0,01 **se cumple**, razón ≥ 1,5 **NO se cumple** (0,922).
→ **NO se declara diferencia.** Y nótese la dirección: los eventos de invierno son, si acaso, **levemente más chicos** que los de verano, no más grandes.

### 5.2 P2 — ¿Difiere la hora del día?

Reparto de las horas de inicio (porcentaje de los eventos de cada estación):

| hora | invierno | verano | | hora | invierno | verano |
|---|---|---|---|---|---|---|
| 00 | 1,6% | 2,0% | | 12 | 5,8% | 5,7% |
| 01 | 2,5% | 2,3% | | 13 | 5,3% | 5,4% |
| 02 | 1,8% | 2,0% | | 14 | 5,3% | 5,3% |
| 03 | 1,5% | 1,9% | | 15 | 5,3% | 5,4% |
| 04 | 1,3% | 1,0% | | 16 | 4,4% | 4,5% |
| 05 | 1,3% | 1,5% | | 17 | 4,5% | 6,1% |
| 06 | 1,1% | 1,6% | | **18** | **4,5%** | **5,8%** |
| 07 | 2,1% | 2,2% | | **19** | **4,1%** | **4,9%** |
| 08 | 3,6% | 3,1% | | **20** | **4,4%** | **3,9%** |
| 09 | 6,3% | 6,2% | | **21** | **3,8%** | **3,2%** |
| **10** | **14,3%** | **11,0%** | | 22 | 3,6% | 4,2% |
| 11 | 9,6% | 8,8% | | 23 | 2,0% | 2,0% |

**Ventana vespertina 18-21 h: invierno 698/4.145 = 16,84% · verano 623/3.513 = 17,73%**
Diferencia = **−0,89 puntos** (invierno **más baja**) · χ² = 1,004, **p = 0,316**

Criterio (§4.2): p < 0,01 **NO se cumple** · diferencia ≥ 3 puntos **NO se cumple** (y va al revés).
→ **NO se declara diferencia. No hay exceso vespertino en invierno.**

El pico del día no está en la tarde en ninguna de las dos estaciones: está a las **10:00 de la mañana** (14,3% invierno / 11,0% verano). Sobre ese pico, ver §5.4.

### 5.3 P3 — ¿Difiere la duración?

| | invierno | verano |
|---|---|---|
| **duración mediana** | **3 h** | **2 h** |
| media | 3,83 h | 3,59 h |
| p75 | 5 h | 5 h |
| p90 | 8 h | 8 h |
| p99 | 20 h | 18 h |
| fracción ≥ 6 h | 22,1% | 21,2% |

**Razón de medianas = 1,500** · Mann-Whitney p = 0,000296

Criterio (§4.2): p < 0,01 **se cumple** · razón ≥ 1,5 **se cumple** (justo, 1,500).
→ **Se declara diferencia: los cortes de invierno duran más.**

**Pero hay que decir inmediatamente lo siguiente, porque si no el número engaña.** El criterio se cumple *por el pelo* y por una razón mecánica: la duración se mide en horas enteras, así que la mediana sólo puede valer 1, 2, 3… y la razón de medianas sólo puede saltar de 1,0 a 1,5 a 2,0. No hay valores intermedios. Por cualquier medida continua el efecto es **pequeño**: la media sube 6,7% (3,83 vs 3,59 h), el p75 y el p90 son **idénticos** (5 h y 8 h). Y el chequeo de sensibilidad (§5.5) muestra que con otro umbral las medianas se igualan. **La diferencia de duración existe pero es del orden del 5-7%, no del 50%.** El veredicto formal no se cambia (regla 2), pero se informa que no es robusto.

### 5.4 El pico de las 10:00 — un contaminante que hay que declarar

El máximo de inicios en ambas estaciones está a las 10:00. **No es meteorología: es calendario laboral.** Inicios a las 10 h por día de muestra, normalizados por cuántos días de cada tipo cayeron en la muestra:

| | lun | mar | mié | jue | vie | sáb | **dom** |
|---|---|---|---|---|---|---|---|
| invierno | 13,0 | 15,7 | 15,0 | 18,3 | 16,6 | 8,1 | **3,0** |
| verano | 6,7 | 11,7 | 13,8 | 8,8 | 10,3 | 9,0 | **4,9** |

De lunes a viernes hay 3 a 6 veces más inicios a las 10 h que el domingo. **El clima no sabe qué día de la semana es.** Estos eventos son, casi con seguridad, **trabajos programados** de las distribuidoras (además duran más: mediana 5 h en invierno contra 2 h del resto del día). La SEC **no distingue corte programado de corte por falla**, y eso mete ruido en todo el conjunto.

Chequeo: **quitando los inicios de las 10:00**, la ventana vespertina queda en **19,65% invierno vs 19,92% verano** — sigue sin haber exceso de invierno. La conclusión de P2 no depende de este contaminante.

### 5.5 Chequeos de robustez *(post-hoc — no forman parte del criterio)*

**Sensibilidad al umbral de 100 clientes:**

| umbral | n inv | n ver | pico med. inv | pico med. ver | dur. med. inv | dur. med. ver | %18-21 inv | %18-21 ver |
|---|---|---|---|---|---|---|---|---|
| 50 | 5.441 | 4.663 | 167 | 180 | 3,0 | 3,0 | 17,6% | 17,9% |
| **100** | 4.145 | 3.513 | 248 | 269 | 3,0 | 2,0 | 16,8% | 17,7% |
| 250 | 2.271 | 1.933 | 553 | 627 | 2,0 | 2,0 | 15,5% | 16,7% |
| 500 | 1.308 | 1.193 | 1.124 | 1.276 | 2,0 | 2,0 | 14,6% | 16,5% |
| 1.000 | 761 | 712 | 2.076 | 2.259 | 2,0 | 2,0 | 16,4% | 17,1% |
| 2.000 | 428 | 415 | 3.569 | 3.608 | 2,0 | 1,0 | 15,7% | 18,3% |

- **P1 es sólido:** con *todos* los umbrales el pico mediano de invierno es **menor** que el de verano.
- **P2 es sólido:** con *todos* los umbrales la ventana vespertina de invierno es **igual o menor** que la de verano.
- **P3 es frágil:** con umbral 50, 250, 500 y 1.000 las medianas de duración son **iguales**. El «1,5×» aparece sólo en el umbral 100. Confirma lo dicho en §5.3.

**Prueba dirigida al mecanismo, a su escala propia.** Si lo que falla es *un transformador*, el corte debería medir del orden de 50-300 clientes. Tomando sólo esos eventos:

> invierno n=3.800 → 19,45% en la ventana 18-21 h
> verano n=3.083 → 19,10% en la ventana 18-21 h
> χ² = 0,107 · **p = 0,743**

Ni siquiera a la escala exacta del mecanismo propuesto aparece el exceso vespertino de invierno. Y dentro de cada estación la ventana 18-21 h apenas sobresale sobre un reparto plano: **1,058× en invierno, 1,076× en verano** (un reparto plano daría 16,67%). El invierno destaca **menos** que el verano.

### 5.6 Lo que sí distingue al invierno *(post-hoc)*

El invierno **sí** es claramente peor, pero no por donde decía la hipótesis:

| | invierno | verano | |
|---|---|---|---|
| Carga total | **340.041** clientes-hora/día | 243.921 clientes-hora/día | **+39%** |
| Nº de eventos | **98,7** por día | 83,6 por día | **+18%** |
| Tamaño por evento | 248 (mediana) | 269 (mediana) | **−8%** |
| Concentración: peor día | **12,0%** de la carga | 5,1% | |
| Concentración: 3 peores días | **25,4%** de la carga | 14,2% | |

Los cinco peores días de invierno de la muestra (encabezados por **21-jun-2024**, con 1,72 millones de clientes-hora, y **09-jul-2022**) concentran la carga. **El invierno pesa más porque tiene días catastróficos, no porque tenga tardes malas.** Una sobrecarga por calefacción produciría un exceso repartido en *muchas* tardes frías; lo que se ve es lo contrario: carga apilada en unos pocos días, que es la forma de un temporal.

---

## 6. Veredicto

### 6.1 Las tres preguntas, según el criterio de §4

| # | Pregunta | Resultado | ¿Cumple el criterio? | Veredicto |
|---|---|---|---|---|
| **P1** | ¿Más clientes por evento en invierno? | mediana 248 vs 269 · razón 0,92 · p=0,0004 | p sí, tamaño **no** | **NO.** No son mayores; si acaso, levemente menores. |
| **P2** | ¿Distinta hora del día? | 16,84% vs 17,73% en 18-21 h · p=0,316 | **no** y **no** | **NO.** No hay exceso vespertino en invierno. |
| **P3** | ¿Distinta duración? | mediana 3 h vs 2 h · razón 1,50 · p=0,0003 | sí y sí (al límite) | **SÍ**, pero el efecto real es de ~6%, y no sobrevive el cambio de umbral. |

### 6.2 Sobre la hipótesis del director

Recordando lo que se fijó en §4.3 antes de mirar nada:

> «La hipótesis del director queda **SIN APOYO** si P2 no muestra exceso vespertino, mientras P1 y/o P3 sí muestran invierno mayor: ese patrón es la firma del daño externo, no de la sobrecarga.»

Es exactamente lo que pasó. Por lo tanto:

> ### La hipótesis de la sobrecarga por calefacción queda SIN APOYO en este dato.
> ### La otra mitad de la hipótesis — que el invierno castiga más a la red — queda CONFIRMADA y es grande: +39% de carga y +18% de eventos.

Dicho en simple: **el director tiene razón en que el invierno es peor, pero el invierno no ataca por dentro, ataca por fuera.** La imagen que dibuja el dato no es la de miles de transformadores calentándose a la hora de la once. Es la de unos pocos días de temporal que voltean la red y dejan a mucha gente sin luz de golpe.

La analogía: no es un motor que se funde por exigirle demasiado todos los días, es un techo que se vuela dos veces al año. Se arreglan de manera distinta, y se priorizan de manera distinta.

### 6.3 Qué NO dice este veredicto — importante

**Esto no prueba que los transformadores no se recalienten ni se incendien.** Con toda seguridad ocurre; es física conocida y la SEC lo sanciona. Lo que se probó es algo más acotado y hay que decirlo con precisión:

> A la resolución que publica la SEC —comuna, hora en punto, sin causa— **la sobrecarga vespertina de invierno no deja una huella distinguible.** Si el mecanismo existe, o es demasiado chico para verse contra el ruido de todo lo demás que corta la luz, o no se concentra en la tarde como suponía la hipótesis.

Un corte de un transformador que sirve a 150 casas es una gota en una serie donde un temporal mueve 50.000 clientes. **La ausencia de huella no es la ausencia del mecanismo.** Y este dato no puede separarlos, porque no trae la causa (§2.2).

### 6.4 Consecuencia para el proyecto

1. **El eje climático de la matriz debe seguir apuntando al daño externo** (viento, lluvia, nieve, remoción en masa) y no incorporar, por ahora, un término de «estrés por demanda» — no hay evidencia que lo sostenga y sí evidencia de que no aparece.
2. **La carga de invierno es episódica, no crónica.** Para la matriz esto es una instrucción de diseño: la variable relevante es la **exposición a días extremos**, no la media estacional. Un promedio mensual borraría justo el 25% de la carga que se juega en tres días.
3. **La SEC no distingue corte programado de corte por falla, y eso es un problema serio** para usarla como juez del modelo (§5.4). Antes de calibrar `C_clim` contra esta serie hay que descontar los trabajos programados, o el modelo se calibrará en parte contra las cuadrillas de mantenimiento.
4. **Vale la pena mirar el pico de las 10:00 por sí mismo.** Es una firma operacional nítida y hasta ahora no descrita en el proyecto.

---

## 7. Limitaciones

**Del dato (no se pueden arreglar con más descargas):**

1. **No hay causa.** Es la limitación que impide confirmar la hipótesis directamente, no sólo refutarla. Todo lo anterior es inferencia sobre huellas.
2. **No hay activo.** No se sabe qué falló. El puente activo→consecuencia sigue sin construir.
3. **No se distingue programado de falla** (§5.4) — se estima que contamina del orden del 12-14% de los eventos, los que empiezan a las 10:00.
4. **Resolución de una hora.** Un corte de 40 minutos puede no existir para esta fuente. Si los transformadores fallan y se reponen rápido, esta serie los pierde.
5. **Clientes, no personas ni criticidad.** Un hospital y una casa valen 1.

**Del diseño de esta prueba:**

6. **42 días por estación, no seis años.** Se muestrearon 28 bloques de 3 días, no el calendario completo (regla del proyecto: no barrer un servicio público). Alcanza de sobra para P1, P2 y P3, que se apoyan en miles de eventos — pero **la cifra de +39% de carga es mucho más incierta**, porque depende de cuántos días de temporal cayeron por azar en la muestra. Con 42 días y una carga tan concentrada (el peor día es el 12% del total), esa cifra puede moverse bastante. **Tómese como orden de magnitud, no como medida.**
7. **«Evento» es una construcción nuestra** (§4.1), no un dato de la SEC. Dos cortes simultáneos e independientes en la misma comuna se cuentan como uno solo.
8. **La razón de medianas es mala medida de tamaño para una variable en horas enteras** (§5.3). El criterio de P3, tal como se fijó, resultó ser demasiado grueso. **Para pruebas futuras conviene fijar el tamaño de efecto sobre la media o sobre un desplazamiento distribucional** — pero eso se aplica de aquí en adelante, no a este informe.
9. **No se controló por región.** Un temporal en el sur y una ola de calor en el norte se mezclan en el total nacional. Un análisis por macrozona podría revelar cosas que el promedio país esconde. Queda pendiente.
10. **No se cruzó con el clima real de cada día.** Este informe compara estaciones del calendario, no condiciones meteorológicas. El cruce con las variables de amenaza (DMC/ERA5, ya disponibles en el proyecto) es el paso natural siguiente y permitiría separar «día de temporal» de «día frío y calmo» — que es justo la separación que dejaría a la hipótesis del director su mejor oportunidad.

**Qué haría falta para zanjar la hipótesis de verdad:**

- El **Anuario SEC / SAIDI-SAIFI recalificado** (ficha A2 del catastro), que **sí clasifica causas** y distingue fuerza mayor de responsabilidad propia.
- Registros de **falla de transformadores de distribución** de las distribuidoras, o vía Ley 20.285 de Transparencia a la SEC.
- Una petición formal por Transparencia del **extracto histórico completo** de interrupciones con causa — que además evitaría tener que muestrear.

---

*Adaptador: `adaptadores/sec_cortes.py` · reproducible con semilla 20260816 · crudo íntegro en `datos/crudo/sec/2026-08-16/`*
