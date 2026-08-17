# INVENTARIO DE LA PARTE I — versión definitiva

**13-ago-2026.** Reemplaza todas las tablas anteriores de esta sesión.

**Criterio, aplicado ANTES de cualquier veredicto:** un nodo admite experimento sólo si su **falsador es
construible dentro de un instrumento**. Si el brazo "donde el nodo no se cumple" resulta ser una guarda de
código, una regla de parada, la ausencia del sustrato o una condición donde no hay nada que observar, el nodo
está en la Columna A y **ningún resultado experimental puede tocarlo** — ni a favor ni en contra.

**Verificación de que el criterio es real, no filosófico:** tres implementaciones independientes intentaron
construir el brazo "sin nada" y las tres devolvieron artefacto — `ε=0` (guarda de división por cero),
`α=0` (regla de parada, se invierte con presupuesto igualado), `NULL-1/NULL-2` (los únicos brazos con cero
sumideros son los únicos sin grafo).

---

# COLUMNA A — Se constata. Ningún experimento aplica.

## Bloque 1 — Persistencia

| Nodo | Qué dice |
|---|---|
| **S > 0** (principio) | La persistencia es condición de posibilidad de toda diferencia |
| **C-N1** | Algo persiste lo suficiente para no anularse |
| **C-N1.1** | Lo que no deja huella no puede diferenciarse |
| **C-N1.2** | La persistencia se confirma al seguir existiendo |
| **C-N1.3** *(la inclusión)* | Lo que persiste ⊆ lo posible |

## Bloque 2 — Acoplamiento

| Nodo | Qué dice |
|---|---|
| **C-N2** *(mitad de fondo)* | Sin acoplamiento no hay persistencia que sostener |
| **C-N2.1** | Hacen falta interior y exterior. *El brazo "interior sin exterior" no se construyó nunca en 370 archivos* |

## Bloque 2.5 — Tiempo

| Nodo | Qué dice |
|---|---|
| **C-N2.5** | El tiempo es el orden inducido por las diferencias |
| **C-N2.5.1** | Diferencia ⇒ hay un antes y un después |
| **C-N2.5.2** | El tiempo es la secuencia de constricciones acumuladas |
| **C-N2.5.3** | Sin diferencia no hay tiempo |
| **C-N2.5.4** | El tiempo es condición de posibilidad del espacio de estados |
| **C-N2.5.5** *(dos partes)* | **(a)** Sin asimetría primordial no hay universo. **(b)** Las violaciones CP son su instancia física observada — física ya establecida fuera de este proyecto, falsador inconstruible |

## Bloque 2.7

| Nodo | Qué dice |
|---|---|
| **C-N2.7.6** | Correspondencia estructural ≠ reducción física. Es una aclaración de método |

## Bloque 2.8 — el marco de los invariantes

| Nodo | Qué dice |
|---|---|
| **C-N2.8** | El cierre impone invariantes universales |
| **C-N2.8.1** | Invariancia ≠ constancia escalar |
| **C-N2.8.2** | Universalidad |
| **C-N2.8.8** | El Teorema: las cinco condiciones simultáneas |
| **C-N2.8.8a** | Fundamentación por reducción al absurdo — es un argumento |
| **C-N2.8.8b** | κ_H es analizabilidad, no viabilidad |
| **C-N2.8.10** | Orden de dependencia κ_P ⇒ κ_Δ ⇒ κ_LF ⇒ κ_O ⇒ κ_V |
| **C-N2.8.10a** | Esa cadena es de **posibilidad**, no operativa |
| **C-N2.8.11 / 11a** | U_Cos y la partición viabilidad/analizabilidad |
| **C-N2.8.12 / 12a** | La fórmula de Λ_Cos y su lectura |
| **C-N2.8.14 / 14a** | Universalidad estructural, y que no implica uniformidad de los valores |

**25 nodos completos + 3 mitades.** Cero experimentos pendientes.

> **Defecto formal, no experimental — C-N2.8.12:** la fórmula da `Λ_Cos → +∞` cuando `|e_R| → 0`, o sea
> salud máxima justo en el límite que C-N2.8.5 declara no viable. Y el problema es de signo, no de límite:
> C-N2.8.8a dice que sin error *"no hay señal correctora"* — el error es requisito, no costo, y está en el
> denominador. Se corrige con lápiz (ver `LAMBDA_COS_correccion_formal_CS.md`). **Hueco destapado:** la cota
> de abajo de `|e_R|` no tiene nombre en el canon.

---

# COLUMNA B — Aquí sí hay experimento

## B1 · Medido, y salió que sí — 16

| Nodo | Experimento | Resultado |
|---|---|---|
| **C-N2.5.10** — cancelación de orientaciones opuestas | CG002-B7 | **★★ El más duro del proyecto.** El ½ que sobrevive: 1080 corridas, banda barrida 6×, **cambio 0,0000**, con regla de decisión que podía declararlo calibración |
| **C-N2.7.10** — la dirección no emerge de la relación pura en mundo-pequeño | CS066-068 · CS069 · CS070/071 | **★★ El mejor sostenido.** Tres rutas con jueces distintos. *Las tres viven en mundo-pequeño — que es exactamente lo que el nodo acota* |
| **C-N2.7.9** — distancia ≠ dirección | CS066 | **★ El más fiel al texto del canon** |
| **C-N2** *(el sostenimiento)* | CG002 producción | Misma asimetría inicial en los 3 brazos; **sólo persiste donde la estructura se mantiene**. 10/10 contra 3/10; λ 21× |
| **C-N2.5.5** *(la amplificación)* | CG002-A1/A2 | **0,02 → 0,51 (25×).** Con firmas idénticas: sobrevive el 100%, **uniforme y sin estructura** |
| **C-N2.7.11** — π distinto entre geometrías | Medición 16-jul | 2,0 · 3,0 · 1,5 — constante dentro de cada geometría |
| **C-N2.7.7** *(distancia local)* | CS066 | Clustering 0,41 contra 0,10 barajado; diámetro 12 contra 3,9 |
| **C-N2.7.12** — la geometría es estado condensado | CS066-069 · Fase III | Aparece distancia sin geometría plena |
| **C-N2.6.2** — las trayectorias siguen el gradiente | CS077 | 7× a 162× contra la ablación correcta, en los 13 puntos |
| **C-N2.6.3** — mínimos locales = atractores | CS057 · cs074-A | 0,47 contra 0,17; y el control sin energía da la curva idéntica |
| **C-N2.6** — curvatura del espacio de estados | CS057 · CS062 | *Caveats: la "gravedad" era el grado del nodo; y el nodo mezcla curvatura de estados con curvatura espacial* |
| **C-N2.2** — si falta uno, colapsa | CS072 | Apagar EM → **0 hidrógeno**; apagar fuerte → **0 helio**. *La identificación EM≈exterior, fuerte≈interior no está justificada en ningún documento* |
| **C-N2.5.7** — orientación emergente | CG002-C1 | ✅ cualificado |
| **C-N2.5.9** — coexistencia de orientaciones | CG002-A5 | ✅ |
| **C-N1.3** *(la fracción que filtra)* | Fase V-A · CS053 · CS057 | **66% de las reglas admitidas se disuelven.** El filtro existe |
| **κ_P** (C-N2.8.3) | CG002, instrumento con mortalidad | **Hay piso**, en el cociente acoplamiento/decaimiento — no en S. Sobrevive el cambio de precisión (desplazamiento 0,000e+00). *En parte derivable* |

## B2 · Medido, con tensión — 1

| Nodo | Resultado |
|---|---|
| **C-N2.5.8** — inercia histórica | Sale, pero **en tensión medida con C-N5.1**: umbral ×2 siempre, pero más de ×8 ya no voltea |

## B3 · Medido, y salió que no — 5

| Nodo | Resultado |
|---|---|
| **C-N2.6.4** *(mitad global)* | **Acotado.** No hay orden global **que se manifieste como crecimiento del diámetro**. De cuatro jueces, sólo uno pasó el control positivo |
| **C-N2.7.7** *(métrica plena)* | Mundo-pequeño hasta el fondo: **13× del lado equivocado**, 0 de 6 celdas en su propio confirmatorio |
| **C-N2.7.8** *(el peldaño de la dimensión)* | El "la dimensión emerge" resultó ser **el input copiado** (REAL 2,77 = NULL 2,80) |
| **C-N2.7.1 – C-N2.7.4** — las cuatro fuerzas | El motor cayó en auditoría. La débil **no actuaba** (0 cambios de sabor en 300 pasos). Lo que sobrevive pertenece a C-N2.2 |
| **κ_Δ** (C-N2.8.4) | **Es la masa**, tres veces: r=+0,997 en Phantom, +1,000000 en CG002 |

## B4 · No se alcanzó en este sustrato — 10

| Nodo | Por qué |
|---|---|
| **C-N2.5.6** — la flecha no es axiomática | **No testeable acá:** el campo es reversible por construcción (error ida y vuelta 5,2·10⁻¹⁶) y el NULL reproducía el **99,65%** del real. *Nota: el resultado nulo era **consistente** con el nodo, no contrario* |
| **C-N2.6.1** — gradientes de estabilidad | Requiere sistema/entorno; Cosmogénesis acopla nodo con nodo |
| **κ_V** (C-N2.8.6) | Igual. Con dato: de 254 biparticiones, el 6,9% da acoplamiento **exactamente 0** y **cero rupturas** |
| **κ_O** (C-N2.8.5) | Requiere una regularidad propia desde la cual distinguir respuesta esperada de realizada |
| **κ_LF** (C-N2.8.7) | Requiere un repertorio propio de respuestas |
| **κ_H** | Requiere variación de conducta, no de estado físico |
| **C-N2.7.5** — ley de regímenes | **Diseñado y no corrido, por riesgo declarado de fabricar el resultado.** Decisión correcta |
| **C-N2.8.9** — violación sostenida ⇒ ruptura | Sin experimento |
| **C-N2.8.9a** — tipología de ruptura | Sin experimento |
| **Λ_crit** (C-N2.8.13) | Bloqueado hasta corregir la fórmula |

## B5 · Parcial

| Nodo | Estado |
|---|---|
| **C-N2.7** — regímenes discretos | Sale sólo para el régimen gravitacional (6 controles NULL, p=0,000333) |
| **C-N2.7.11** *(la mitad "indefinido")* | **El instrumento sólo veía d=2.** Una retícula cúbica perfecta también habría sido declarada indefinida, porque `π(r) ∝ r^(d−2)`. El arreglo: medir el exponente **d** directamente |

---

# Fuera del texto que entregó el director

*El canon que me pasaste se corta en C-N2.8.14a. Estos nodos existen en el registro y tienen veredicto,
pero no puedo juzgarlos contra su texto porque no lo tengo.*

| Nodo | Estado en el registro |
|---|---|
| **C-N3, C-N3.1, C-N3.2** — historia irreversible | Base más delgada de lo que parecía: CS009 y CS014 existen **sólo como dos líneas de tabla**, sin informe ni control. Y el "240 de 240" es **240/360** en la fuente cruda, y medía **retención de energía**, no monotonía de entropía |
| **C-N4, C-N4.1, C-N4.2** — delimitación | **0 de 270** en V6 (máx 1,826 contra umbral 2,0, siendo alcanzable). Y F8-05 mostró que **la frontera elegida invierte el signo** de un resultado sobre las mismas corridas |
| **C-N5, C-N5.1, C-N5.2** — estabilidad en rango | **★ De lo más limpio del proyecto:** cs074-A, 1920 corridas, pre-registro, y el control sin energía da la curva **idéntica** — el techo no es energético |

---

# Balance

**Columna A: 25 nodos completos + 3 mitades.** No se prueban. No hay deuda experimental.

**Columna B: 27 entradas.** De ellas: **16 salieron que sí** · **1 con tensión** · **5 que no** ·
**10 no alcanzadas** *(de las cuales 6 por incompatibilidad de sustrato, 1 por decisión deliberada y 3 sin
tocar)*.

**Lo mejor sostenido, y no es casualidad que sean casi todos negativos o casi-negativos:** el ½ de la
cancelación, la dirección que no emerge, distancia ≠ dirección, el techo que no es energético, y el barajado
que podía sostenerse y no se sostuvo. En este proyecto lo que aguanta es lo que alguien intentó tumbar.

**No se declara ningún cierre.** Esa decisión es del director.

---
---

# SEGUNDA PARTE — LOS DOS BLOQUES EXPERIMENTALES

*Separación pedida por el director. El pivote es **CS072**, 17-jul-2026: el momento en que se agregan las
fuerzas fundamentales. Antes de eso, todo es relacional. Después, se intenta física.*

---

## BLOQUE I — TOPOLÓGICO
### 29-jun → 17-jul · desde S>0, con grafos

**La pregunta:** partiendo sólo de S>0 y relaciones puras, ¿emerge el espacio?

**Volumen:** ~63 experimentos con resultado. **58 grafo · 4 física · 1 híbrido.**
Los únicos 4 de física son **CG001** (campo φ continuo, relajación, exergía/entropía). Todo lo demás es
estructura relacional discreta.

**Advertencia de fondo, verificada en código:** en este bloque *"gravedad"*, *"electromagnetismo"*,
*"fuerza débil"*, *"Higgs"*, *"inflación"* y *"energía oscura"* son **nombres de reglas de recableado de
grafo**, no física integrada. `AUDITORIA_gravedad_sin_masa_CS.md` (5-jul, disparada por el director) verificó
en cuatro scripts que la "gravedad" era `rho=[len(a) for a in adj]` — **el grado del nodo**. Su conclusión
textual: *"eso NO es gravedad: es enlace preferencial"*.

### Lo que el bloque estableció

| | Resultado |
|---|---|
| **El arco de eliminación CS058→CS063** | **Negativo entero y limpio.** Ni el marco (espín), ni la masa, ni el vértice de 3 cuerpos genuino seleccionan dimensión — cada uno colapsando bajo su propio NULL. Con dos falsos positivos cazados por el equipo en el camino. Conclusión asentada: *"la contingencia se gana el derecho a ser la conclusión"* |
| **CS066 — el tejido** | La localidad fuerte **sí** produce tejido con distancias efectivas: clustering 0,41 contra 0,10 barajado, diámetro 12 contra 3,9 |
| **CS066conf — el confirmatorio** | **Pero no es una métrica 3D:** 0 de 6 celdas en el rango esperado. Veredicto textual: *"esponja 3D-local con mundo-pequeño residual, NO 3-manifold métrico"* |
| **CS067 / CS068 / CS069 / CS070 / CS071** | **La dirección no enciende, por cuatro rutas independientes.** El sustrato queda mundo-pequeño hasta el fondo: residual **13× del lado equivocado** |
| **CG002-B7** | **★★ El resultado más duro del proyecto.** El ½ de la cancelación antipodal: 1080 corridas, banda barrida 6×, **cambio 0,0000**, con regla de decisión que podía declararlo calibración |
| **CG002-A1/A2** | La asimetría mínima **0,02 → 0,51 (25×)**. Con firmas idénticas: sobrevive el 100%, uniforme y **sin estructura** |
| **π contingente (16-jul)** | π **distinto entre geometrías** (2,0 · 3,0 · 1,5), constante dentro de cada una |

### Lo que el bloque NO puede establecer

**Nada sobre el mundo físico.** Es su límite constitutivo, no un defecto. Que una fuerza local no seleccione
dimensión **en un grafo** no dice qué pasa en física. Y que la distancia emerja del encadenamiento de
diferencias **en un grafo** tampoco.

---

## BLOQUE II — FÍSICO
### 17-jul → 07-ago · agregando las fuerzas, buscando 3D y una estrella

**La pregunta:** tomando lo que el topológico dejó, y agregando las cuatro fuerzas fundamentales, ¿se obtiene
un espacio 3D donde Phantom pueda encender una estrella?

**Volumen:** ~68 experimentos con resultado, mayoritariamente física e híbridos. **19 retractaciones.**

### Lo que el bloque estableció

| | Resultado |
|---|---|
| **CS072 — el motor de partículas** | **Cayó casi entero.** `bariones = quarks/3` exacto **e invariante a apagar todas las fuerzas**; el 7:1 lo producía un `*20.0` puesto a mano; la dimensión **copiaba su propio input** (REAL 2,77 = NULL 2,80) con el número objetivo escrito en el docstring; la fuerza débil **no actuaba** (0 cambios de sabor en 300 pasos) |
| **Lo único que sobrevivió del motor** | **Dos ablaciones:** apagar electromagnetismo → 0 hidrógeno; apagar fuerza fuerte → 0 helio |
| **CS073 — el puente a Phantom** | **Funciona:** nacen sumideros, z=48,69 contra el NULL original |
| **La jerarquía NULL-0..NULL-5** | Los dos únicos brazos que dan cero (**NULL-1 y NULL-2**) son **los dos únicos que no pasan por el grafo**. Con grafo —aunque destrozado— sale lo mismo: REAL vs NULL-3 **p=0,42** |
| **cs074-A** | **★ De lo más limpio:** 1920 corridas, pre-registro, y el control sin energía da la curva **idéntica** (+0,0 en 20 valores) → el techo no es energético |
| **CF-1 — campo caliente + expansión** | **Resultó ser grafo disfrazado de física:** su "expansión" es **cortar aristas de un anillo**; la fila de ε=0 sale de una guarda de división por cero; y "el tamaño de ε no importa" es cancelación algebraica |
| **CS077** | La gravedad organiza 7× a 162× más que patadas al azar |
| **CS076, CS078, CS079** | Dirección temporal sin señal; κ_V débil; delimitación sin discriminar |

### La respuesta a la pregunta que motivaba el bloque

**No se obtuvo el espacio 3D.** El motor que iba a producirlo se cayó en auditoría, y el sustrato siguió
siendo mundo-pequeño. **La estrella sí se enciende** — Phantom forma sumideros — pero sobre una condición
inicial cuya estructura viene del grafo, no de las fuerzas.

Y el hallazgo que salva a la física de ser mera transportadora: **al umbral real de encendido de Phantom, las
condiciones iniciales no tienen ni una partícula prendida en las 24 corridas.** Ese contraste lo pone la
gravedad entera. *La geometría de partida ordena cuánta masa quedó en montoncitos; sólo la dinámica ordena
cuánta quedó lo bastante apretada como para encender.*

---

## BLOQUE III — LO QUE SE ENREDÓ
### 08-ago → 13-ago · la línea A2-B0-C2

**Volumen:** ~70 experimentos. **Más de 55 no alcanzan ningún nodo del canon.**

Son exploración del instrumento sobre **una sola regla**. Fases V-B, VI, VII y VIII: cinco fases de trabajo,
el bloque más caro en cómputo de todo el proyecto, y **toca cuatro nodos**:

| Nodo | Qué dio |
|---|---|
| C-N1.3 | El filtro existe pero es poco selectivo |
| C-N2.7.12 | Apoya en lectura negativa |
| C-N2.7.7 | **No alcanza** — mide diámetro bajo agrupamiento, no distancia emergente |
| κ_V | **Refuta la operacionalización**: es casi copia de la masa y se invierte a N=4000 |

**Por qué se enredó, en una línea:** la pregunta dejó de ser *"¿emerge el espacio?"* y pasó a ser *"¿el
apiñamiento de triángulos predice la masa acretada cuando fijamos los grados?"*. La segunda es contestable y
la primera no lo era con ese instrumento — así que el trabajo migró hacia donde había respuestas.

**Dato que lo confirma:** en los 29 híbridos, entre brazos lo único que varía es la estructura del grafo. El
layout, la masa, el lado de la caja y la turbulencia (**Mach=3, semilla=42, escrita a mano en los 21
generadores y nunca variada en la historia del proyecto**) son idénticos. **La física transporta; no se la
interroga.**

---

# EXPERIMENTOS IRRELEVANTES PARA LA TEORÍA

*Trabajo real, correctamente ejecutado, que no responde ninguna pregunta del canon.*

| # | Qué | Costo | Por qué es irrelevante |
|---|---|---|---|
| 1 | **Fases V-B, VI, VII y VIII** | El mayor gasto de cómputo del proyecto | 55+ de 70 experimentos no tocan ningún nodo. Exploración del instrumento sobre una regla |
| 2 | **cs074-D** | **48.000 corridas, 61 horas** | **0 de 1647 configuraciones con z>2**, y con **defecto de control declarado dos días antes** de terminar. La recomendación de detenerlo llegó tarde |
| 3 | **El motor CS072 v5, v6, v7, fold, reserva B** | ~19 retractaciones | Dos motores NO ADMISIBLES, una firma suspendida, dos retractaciones formales. Todo comparado, antes del 17-jul 21:36, contra un **NULL isomorfo** — barajar la ficha completa daba afinidad idéntica al real |
| 4 | **F8-04 a N=8000** | 14 corridas | **0 de 14 llegaron a tmax.** Útil como resultado negativo sobre el instrumento; nulo sobre la Teoría |
| 5 | **La batería de clases I-IV** | Fase V-A completa | El umbral 0,7 fabricó bimodalidad: **37% de las reglas cambiarían de clase con sólo re-medir**, y una recta explica R²=0,663 contra 0,182 del escalón |

## Corridas hechas y nunca reportadas

| Qué | Estado |
|---|---|
| **BATERIA_FUNDAMENTOS** — 22 JSON del 24-jul, **9 experimentos con resultado, cero informes** | Y **no son confirmatorios**: F2-6 muestra que barajar el orden temporal no cambia nada (es percolación, no competencia de tasas); F2-1/F2-2 ponen el crítico en r≈0,03 y no en 1, contradiciendo la invariancia en N que sostiene el sello de CF-1; F1-2 **falla** el control donde F1-1 pasa; y **F1-3 muestra que la meseta de ε está acotada por la precisión de la máquina** |
| **Un brazo completo de Phantom con APR**, 3-ago | Corrido y jamás mencionado (masa 1945,8 contra 2124,4) |
| **9 corridas en `bateria_n2000/`** | Fuera de todo informe |
| **El barrido de ε=0 de CF-1** | Corrido, pre-inscrito, y **ausente de su propia adjudicación** |

## Documentos activos que informan mal

| Documento | Problema |
|---|---|
| **`REGISTRO_ACTUALIZACION_CS069-CS073.md`** | Declara **"Parte 1 CERRADA"** y CS072 como *"el resultado positivo mayor de todo el proyecto"*. La auditoría que lo tumba es **de 5 horas después**. Lleva 24 días sin corregir, y **todos sus números están muertos** |
| **`VERIFICACION_NODOS_TEORIA_PARTE_I_2026.md`** | Se escribió **sin leer las adjudicaciones** (0 menciones de la palabra). Difiere de ellas en 5 nodos, en ambas direcciones |
| **`bateria_null5_n2000`** | **No es un control:** sus condiciones iniciales son bit a bit idénticas a REAL |


---
---

# TERCERA PARTE — EL ORIGEN, Y POR QUÉ IMPORTA

*Todo esto salió de un intento de modelar mínimamente S>0 para la web.
Fuente: `Cosmogenesis-Web/`, línea paralela con el equipo Web (Meta ejecutando; Qwen, DeepSeek, Gemini,
Grok y el director analizando).*

## La pregunta fundacional — textual

> *"De una **asimetría ínfima de temperatura** en un **todo infinito** (en el sentido de que es el todo),
> ¿hay **persistencia** si ese todo se expande a una velocidad **mayor a la de la luz**? Ese es el
> experimento."*

**Respuesta del hilo:** sí persiste cuando hay ε≠0 y la expansión gana a la interacción que la borraría
(`H > Γ`). **Con ε=0 no hay qué persistir.**

## El veredicto que ya estaba dado el 21-jul

> **"Topología sí; pre-átomos no."** · **"Puente topología → pre-átomos: NO CRUZADO."**

Doce experimentos con gates declarados de antemano, y fallos limpios: masa 0,206 contra el 0,1 pedido y
**380× lejos de 1/1836**; el estado ligado sale **más pesado**, no más liviano; F no se observa
dinámicamente; contacto pero no confinamiento lineal; **3D da ratios idénticos a 2D**; y la quiralidad no
agrega exclusión — la repulsión ya existía sin ella.

## El rechazo del cierre — decisión del director, 21-jul

Cinco IAs recomendaron por unanimidad cerrar y publicar. El director dijo que no, con este argumento:

> *"La Opción A confunde **límite del instrumento** con **falsación de la Teoría**."*

Y dejó `W-44 — dominio F (físico pleno) con contraste vs T` en **⏳**.

## Dónde encaja el arco del Mac

| | Qué es | Resultado |
|---|---|---|
| **Web** (→21-jul) | La pregunta, respondida en el dominio T | Puente no cruzado; cierre rechazado |
| **Bloque I** (29-jun→17-jul) | El dominio T otra vez, en paralelo y más hondo | **Mismo veredicto.** No fue redundante: agregó el arco de eliminación CS058-063 y el ½ de 1080 corridas |
| **Bloque II** (17-jul→7-ago) | **Éste era el dominio F que W-44 dejó pendiente** | **Falló igual** — y esta vez el motor además estaba roto |
| **Bloque III** (8-13 ago) | Ni T ni F | Exploración del instrumento |

## El hallazgo que más importa

`09_LIBRO_DE_CLAIMS.md` cierra con una **checklist anti-Cosmo de seis preguntas**, escrita el **21 de
julio**. Cotejada contra todo lo que se encontró el 13 de agosto:

| Pregunta del 21-jul | Lo que falló después |
|---|---|
| **1.** ¿Apagar el actor mata el observable? | Sólo las **dos** ablaciones que sobrevivieron (EM→H, fuerte→He) |
| **2.** ¿El contador lee dinámica o forma/catálogo? | κ_P = hora de nacimiento · κ_Δ = la masa **3 veces** · κ_V = la masa · la dimensión copiaba su input |
| **3.** ¿REAL/NULL son el mismo universo menos un factor? | NULL-1/2 sin grafo · el NULL isomorfo del fold · α=0 · ε=0 · el NULL de CS076 al **99,65%** del real |
| **4.** ¿El nombre tiene el ingrediente? | "Gravedad" = **grado del nodo** · "expansión" = **cortar aristas** |
| **5.** ¿El gate es un número de nuestro universo? | El **2.05 escrito en el docstring** · el 7:1 de un `*20.0` |
| **6.** ¿Una redefinición mejoró el score sin dinámica nueva? | Umbrales de clase: **37% de las reglas cambiaban al re-medir** |

**Las seis, sin excepción.** La checklist predijo cada error de las tres semanas siguientes.

**No faltó método: el método estaba escrito y se dejó atrás.** Y con él, la regla que más falta hizo del
libro de claims: *"un claim sólo puede estar en **un** estatus vivo"*.

---

# PENDIENTES — para el workflow acotado

| # | Qué | Dónde va |
|---|---|---|
| 1 | **Dominio F** (W-44) con un motor que pase la checklist anti-Cosmo | Cosmogénesis, si se retoma |
| 2 | **κ_V, κ_O, κ_LF, κ_H** y **C-N2.6.1** — todos requieren sistema/entorno | **Célula Madre / ANIMA**, no acá |
| 3 | **Bloque 6 completo** (replicación) | **Célula Madre** — Cosmogénesis no tiene entidades que se repliquen |
| 4 | **Λ_Cos**: adoptar una de las tres formas corregidas; recién ahí Λ_crit es medible | Lápiz, una hora |
| 5 | **Medir el exponente *d*** de \|S(r)\| ∝ r^(d−1) en vez de π | Sirve a **C-N2.7.8 y C-N2.7.11 juntos** |
| 6 | **Orden global que no crezca con el diámetro** — orientación, jerarquía, flujo neto sobre sustrato compacto | Nombrado hoy, nunca medido |
| 7 | **C-N2.7.5** — ley de regímenes | **Congelado por decisión**, no por deuda |
| 8 | **C-N2.8.9 / 2.8.9a** — tipología de ruptura | Sin experimento |
| 9 | **C-N3**: CS009 y CS014 existen sólo como dos líneas de tabla | Documentar o re-correr |
| 10 | **C-N4**: la frontera elegida invierte el signo de un resultado | Consecuencia metodológica sin resolver |

