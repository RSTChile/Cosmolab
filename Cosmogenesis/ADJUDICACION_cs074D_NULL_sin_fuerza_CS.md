# ADJUDICACIÓN cs074-D — EL BARRIDO FINO NO PUEDE CONTESTAR LA PREGUNTA (defecto de control)

**Adjudica:** Claude Science (asistente) · **Director:** Alexis López Tapia · 27-jul-2026
**Estado:** el experimento **NO se cierra**. Se detiene y se rediseña el control.
Requiere autorización explícita del director (NOTA_PERMANENTE_CS).

---

## 0. Lo primero: hay que decidir sobre un proceso que está corriendo AHORA

El barrido completo (`--full`, 2000 configuraciones) se lanzó el 26-jul a las 20:50.
Último latido en disco (`cs074D_full_stderr.log`): **27-jul 00:08, configuración 100 de
2000, tiempo transcurrido 11.888 s, tiempo estimado restante 225.868 s ≈ 63 horas.**

**Recomendación: detenerlo.** No porque esté mal programado —el código es correcto— sino
porque el número que decide el experimento no puede alcanzar su propio umbral. Las 63
horas producirían 2000 filas de un estadístico que no mide lo que el diseño quería medir.

---

## 1. Qué se verificó en disco (no de palabra)

Se leyó el resultado del smoke (`resultados_cs074D_barrido_fino/cs074D_result_smoke.json`,
20 configuraciones × 12 semillas × 2 brazos = 480 corridas, 2.904 s) y se corrieron
**108 corridas nuevas de verificación** con el motor sin tocar.

### 1.1 El resultado tal como salió

| lectura | valor |
|---|---|
| configuraciones válidas | 16 de 20 |
| configuraciones con z > 2 | **0** |
| z máximo observado | **+0,226** |
| z mínimo observado | −0,563 |
| media de z | −0,066 |
| conectividad | `sin_hits_z2` (no calculable) |

Ninguna de las tres lecturas pre-inscritas del §6 se puede emitir: las tres presuponen
que existe al menos una configuración con z > 2, o que más del 50 % lo supera. Hubo cero.

### 1.2 La prueba que decide: el NULL FALSO

Si el NULL del diseño destruyera estructura real, tendría que separarse claramente de un
NULL **falso** (comparar corridas reales contra otras corridas reales con distinta
semilla, donde por construcción no hay ninguna estructura que destruir). Se corrió esa
comparación en tres configuraciones (24 corridas reales + 12 NULL cada una):

| configuración | z contra el NULL del diseño | z contra el NULL falso |
|---|---|---|
| 2 | +0,226 | +0,079 |
| 5 | −0,208 | +0,118 |
| 19 | −0,563 | −0,064 |
| **magnitud típica \|z\|** | **0,332** | **0,087** |

El NULL del diseño **sí hace algo** (0,332 contra 0,087, casi cuatro veces más que no
hacer nada) — pero está **seis veces por debajo** del umbral 2 que el propio protocolo
inscribió. No es que el NULL sea inerte: es que su efecto es demasiado chico para que el
criterio de decisión lo registre.

### 1.3 Por qué es tan chico — la causa raíz, en el código

El NULL baraja el vector de densidades bariónicas entre partículas
(`cs074_energia_holistica.py`, línea 152). Para que barajar destruya algo, la densidad
tiene que estar acoplada a algo que la gravedad vea. Dos hechos verificados en disco:

1. **La masa bariónica es constante.** En las 16 configuraciones válidas, `masa_bar`
   tiene **un solo valor único** (9,4). La masa que entra a la gravedad es
   `masa_bar × dens_bar`; con la masa plana, la densidad es el único portador.
2. **Las posiciones no dependen de la densidad.** `posiciones_escenario`
   (`p_gravedad_general.py`, línea 29) sortea posiciones uniformes con semilla fija, y su
   propio comentario lo dice: las posiciones *"no cargan información, son el contenedor
   neutro"*. La densidad se adjunta **después**, sin correlación espacial.

Consecuencia: barajar densidades entre posiciones que fueron sorteadas al azar **es
estadísticamente casi la misma configuración**. Lo único que cambia es qué partícula
concreta lleva qué peso — y eso solo importa si los pesos difieren entre sí.

Y eso es exactamente lo que se observa: el efecto del NULL sigue a la **dispersión de las
densidades** (Spearman ρ = 0,64, p = 0,008, n = 16). Donde las densidades son casi todas
iguales (dispersión ~1e-6, siete de las dieciséis configuraciones), barajar no cambia
nada: la diferencia real−NULL es de 1e-7 a 1e-6, ruido de redondeo.

### 1.4 Cuánto le falta al efecto — el número que cierra el diagnóstico

El criterio `z = (media_real − media_null) / desviación_conjunta` es un **tamaño de
efecto**, no un estadístico de significancia: **no crece agregando semillas**. Correr 12,
120 o 12.000 semillas no lo mueve.

- Dispersión entre semillas: **0,057** en fracción de masa ligada.
- Para z > 2 el NULL tendría que destruir **más de 0,114** — unos **11,4 puntos
  porcentuales** de estructura.
- Lo que destruye en promedio: **0,56 puntos porcentuales**. El máximo observado en
  cualquier configuración fue 3,8 puntos.

Falta un factor de **20** en promedio, y un factor de **3** aun en el caso más favorable
de las veinte. Las 2000 configuraciones del barrido completo no cambian esa aritmética.

### 1.5 Un hallazgo lateral, sólido y no buscado: el borde de la expansión

Las 4 configuraciones que fallaron no fallaron al azar. Son **exactamente las 4 tasas de
expansión más bajas del barrido**:

- fallan: 0,0012 · 0,0016 · 0,0021 · 0,0022
- la más baja que funciona: 0,0031

El motivo, reproducido corriendo el motor directamente:
`"sólo 0 átomos reales (<8): sin masa suficiente"`. Por debajo de una tasa de expansión
de aproximadamente **0,003, no se forma ni un solo átomo** — el universo no llega a
tener bariones, con independencia de la asimetría (las fallidas cubren ε desde 1,8e-6
hasta 5,4).

Esto es un **borde real del modelo**, encontrado sin buscarlo, y es del mismo tipo que el
hallazgo del Experimento A (demasiada asimetría destruye estructura): un régimen donde el
proceso simplemente no arranca. No estaba en el diseño y no se cierra aquí, pero queda
anotado: **hay un piso de expansión por debajo del cual no hay materia.**

---

## 2. Veredicto

**El barrido fino, tal como está, no puede contestar la pregunta del director.** No falló
la ejecución (CC implementó el protocolo correctamente y sin tocar el motor, como se le
pidió). Falló el **control**: el NULL por barajado de densidades destruye un efecto
veinte veces menor que el que el umbral pre-inscrito exige detectar.

Dicho en simple: **se preguntó si la estructura vive en una banda estrecha, y el
instrumento no distingue banda de no-banda.** Un resultado de "cero hits" con este NULL
no significa "no hay banda" — significa "no se midió".

**Lo que NO se puede concluir de esto** (y hay que decirlo, porque es la tentación):
no se puede escribir "la estructura no vive en una banda estrecha". Ese sería
exactamente el error que la regla anti-Shannon prohíbe: leer un resultado nulo de un
control que no tiene fuerza para dar un resultado positivo.

**Lo que SÍ queda establecido**, y es un resultado honesto sobre el motor mismo: en este
modelo, **la fracción de masa ligada depende de cuánta masa hay y de dónde está, pero es
prácticamente insensible a cuál partícula lleva qué densidad.** La densidad #23, en el
régimen que barre este experimento, casi no es un canal causal sobre la estructura. Eso
es información real sobre el motor, no sobre el universo.

---

## 3. Qué sigue — tres caminos, para que el director adjudique

Ninguno se ejecuta sin su OK.

**Camino A — arreglar el NULL (recomendado).** El problema es que barajar densidades
entre posiciones azarosas casi no cambia nada. Un NULL con fuerza real debe romper el
acoplamiento que sí importa: **barajar las posiciones**, no las densidades — o mejor,
comparar contra un escenario donde la masa total es idéntica pero repartida de forma
deliberadamente uniforme. Antes de comprometer 63 horas, se corre un **calibrador**: una
configuración donde se sabe que hay estructura y se verifica que el NULL nuevo produce
z > 2. Si el NULL no pasa su propio calibrador, no se usa. Costo: unas 2 horas.

**Camino B — cambiar el observable.** `frac_masa_ligada` puede ser demasiado grueso.
`n_clusters_finales` es bastante más discriminante en el smoke (rango relativo 2,64 contra
1,52 de la masa ligada), y es el observable que más directamente responde a "¿la materia
se juntó en pocos grumos grandes o en muchos chicos?".

**Advertencia — corrijo un error que casi escribo en este mismo documento:** pensé que
esto se podía recalcular sin correr nada, usando el JSON del smoke. **Es falso, y lo
verifiqué antes de afirmarlo.** El JSON guarda `n_clusters_finales_media` y
`frac_masa_en_mayor_cluster_media` **solo del brazo REAL y solo como promedio**, no por
semilla y no del brazo NULL. Sin los valores del NULL no hay z que calcular. El Camino B
exige volver a correr, igual que el A. Lo dejo escrito porque es justamente el tipo de
"verifiqué X" sin X impreso que el Auditor marcó tres veces.

**Camino C — bajar el umbral con justificación previa.** No recomendado: cambiar el
umbral después de ver los datos es exactamente lo que un pre-registro existe para
impedir. Solo sería legítimo si se re-inscribe **antes** de mirar el resultado completo, y
con una razón física, no aritmética.

**En los tres casos, primero: detener el proceso `--full`.** Está gastando cómputo en un
estadístico que no puede cruzar su umbral.

---

## 4. Pendientes que este experimento abrió

- **Piso de expansión ≈ 0,003:** por debajo no se forma ningún átomo. Encontrado sin
  buscarlo. Merece un barrido propio y estrecho alrededor de ese valor.
- **La densidad #23 casi no actúa sobre la estructura.** Se suma al pendiente ya abierto
  de por qué el enfriamiento H₂ tampoco actúa (Experimento B, z ≈ −0,12). Empiezan a ser
  dos canales que el modelo declara y que no mueven el observable. Puede ser una
  propiedad del motor que hay que entender antes de seguir barriendo.
- **El borde inferior de la asimetría sigue sin aparecer.** Este barrido bajó hasta
  ε = 1,8e-6 y ahí el problema fue la expansión, no la asimetría. Sigue abierto.

---

*Verificado en disco: `cs074D_result_smoke.json` (20 configuraciones), 108 corridas de
verificación propias en `verif_null/`, y lectura directa de `cs074_energia_holistica.py`
(líneas 148-152, 64-73) y `p_gravedad_general.py` (líneas 29-35). Ningún número de este
documento fue puesto a mano.*
