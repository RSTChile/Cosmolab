# O-N7.7 — Sistema A/B, cierre a escala grande con masa total FIJA

**Fecha:** 7-ago-2026 · **Método:** código nuevo (`ON77_sistemaA_cierre.py`, `ON77_sistemaB_cierre.py`), no
toca ningún archivo/carpeta congelados — sólo importa. Corridas de Phantom reales, mismos parámetros
físicos de sumideros que toda la jerarquía CS073 (`rho_crit_cgs=1000`, `icreate_sinks=1`, `r_crit=0.6`,
`h_acc=0.3`, `tmax=0.500`, `dtmax=0.001`, copiados de `bateria_n2000/ic_real/cosmog.in`). Presupuesto
~50 min. **No se declara cierre ni veredicto sobre O-N7.7 ni sobre CS073 — sólo se reportan números. La
lectura final es de Alexis.**

---

## 0 · Qué cambió respecto del intento anterior (`ON77_sistemaAB_corregido_CS.md`)

Ese intento usó N=50/100/200/400 (Sistema A) y N=200 fijo (Sistema B) — todo por debajo del piso de
resolución real (documentado entre N=500 y N=1000 en `INFRA_masa_fija_generador_CS.md`) — y dio **cero
sumideros en las 9 corridas**, sin poder probar nada. Este cierre corrige dos cosas a la vez, siguiendo el
encargo:

1. **Escala:** Sistema A ahora barre N **hacia arriba** desde el punto ya sabido limpio, N ∈ {2000, 4000,
   8000}. Sistema B usa N=2000 fijo (el punto confiable) en vez de N=200.
2. **Masa fija:** se reusan las constantes ya validadas de `grafo_random_layout_generar_ic_masa_fija.py`
   (`LADO_FIJO=12.5992`, `MASA_TOTAL_OBJETIVO=18800`) para que la masa TOTAL del sistema sea la MISMA en
   todos los N — así, a partir de N=2000, **subir N ya no significa "más masa", significa MÁS
   RESOLUCIÓN** (partículas más livianas repartiendo el mismo presupuesto de masa). Esto es importante
   para la lectura: la pregunta de O-N7.7(a) pasa a ser "¿más resolución sobre el mismo presupuesto de
   masa sigue encontrando estructura al mismo ritmo, o la ganancia marginal cae?" — no "¿más masa forma
   más estructura?" (eso sería trivial y no es lo que se mide aquí).

**Simplificaciones documentadas por presupuesto de tiempo** (ninguna oculta):
- `dens_bar` a cada N no se re-extrajo físicamente con `_extraer_bariones` (hubiera sido demasiado caro a
  N=8000 dentro de ~50 min) — se re-muestreó (bootstrap con reemplazo, semilla fija) del pool REAL de
  2000 átomos de `bateria_n2000/dens_bar.npy`, el mismo que usa toda la jerarquía CS073. Preserva la
  forma de la distribución real; no es una extracción física independiente por N.
- El layout de resortes (Fruchterman-Reingold, `layout_resortes`, O(N²) por iteración) se corrió con
  `iters=25` en vez del default 100 — medido empíricamente ANTES de correr nada (benchmark propio:
  N=2000/iters=100 → 50s, N=4000/iters=100 → 182s; a `iters=100` y N=8000 el layout solo hubiera tomado
  ~12 minutos). Mismo valor uniforme en Sistema A y B, para que ningún punto reciba más "esfuerzo de
  relajación de semilla" que otro.

---

## 1 · Sistema A — N=2000/4000/8000, regla fija, masa total FIJA

| N | masa en sumideros | n sumideros | masa/N | ganancia marginal (Δmasa/ΔN) | estado |
|---|---|---|---|---|---|
| 2000 | 1240.80 | 10 | 0.6204 | — | completo a tmax=0.5 |
| 4000 | 1941.10 | 29 | 0.4853 | **+0.3502** | completo a tmax=0.5 |
| 8000 | 3188.95 (parcial) | 122 (parcial) | 0.3986 (parcial) | no calculable de forma comparable (ver abajo) | **INTERRUMPIDO en t=0.361/0.500 (72%)** |

**N=16000 no se corrió** — no quedó presupuesto de tiempo tras el problema en N=8000 (ver abajo).

**Qué pasó en N=8000, con precisión:** a diferencia del aborto de N=1000 documentado en
`INFRA_masa_fija_generador_CS.md` (que terminó con un error explícito de conservación de momento
angular), acá Phantom **no abortó con error** — los sub-pasos de tiempo individual (IND TIMESTEPS)
colapsaron progresivamente hasta tomar **~500 segundos de CPU por cada sub-paso individual** alrededor de
t=0.36 (con 122 sumideros ya formados y ~1357 partículas de gas ya acretadas/muertas). A ese ritmo,
llegar a tmax=0.5 hubiera tomado del orden de decenas de horas — inviable dentro del presupuesto. Se
interrumpió la corrida manualmente (no hubo error de Phantom, fue una decisión del orquestador por
presupuesto de tiempo) y se leyó el estado del `.sink` en el último tiempo escrito (t=0.361). **La fila de
N=8000 en la tabla es un estado PARCIAL, no un punto cerrado comparable 1:1 con N=2000/4000** (que sí
llegaron a tmax=0.5) — por eso no se calcula una ganancia marginal formal Δmasa/ΔN entre N=4000 y N=8000:
mezclaría "más estructura por más N" con "más estructura por más tiempo físico transcurrido dentro de la
misma corrida", dos efectos distintos.

**Lectura de lo que SÍ se pudo medir limpio (N=2000→N=4000, ambos completos a tmax=0.5):** la ganancia
marginal fue **positiva y del mismo orden de magnitud que la masa de partida** (+0.35 de masa por
partícula extra, sobre una base de 0.62 masa/N a N=2000) — **no se ve saturación/ganancia decreciente
todavía en este único intervalo limpio**; al contrario, N=4000 casi triplicó el número de sumideros (10→29)
con la MISMA masa total repartida en el doble de partículas. El indicio parcial de N=8000 (122 sumideros
ya en t=0.361, muy por encima de los 29 de N=4000 a tmax completo) apunta en la misma dirección de "más
resolución sigue encontrando más estructura" pero no puede tratarse como un punto de sweep válido por la
razón de arriba.

---

## 2 · Sistema B — N=2000 fijo, mecanismo de reorganización acotada, masa total FIJA, H variable

**Paso previo (grafo, barato) — verificación de memoria genuina, obligatoria antes de gastar Phantom:**

| H | Jaccard vs H=1 | triángulos REAL | clustering REAL | reorganizaciones acumuladas |
|---|---|---|---|---|
| 1 | 1.0000 (ancla) | 2780 | 0.4055 | 0 |
| 2 | 0.7152 | 3709 | 0.3725 | 5 |
| 4 | 0.6339 | 4679 | 0.3493 | 32 |
| 8 | 0.5874 | 5423 | 0.3328 | 127 |
| 16 | 0.5577 | 6203 | 0.3283 | 309 |

El Jaccard se aleja de 1.0 de forma monótona (igual que a N=200 en el intento anterior) — el mecanismo
**sigue teniendo memoria genuina del camino a esta escala también**, no degeneró al escalar N. Verificación
OK → se procedió a Phantom.

**Nivel Phantom — masa en sumideros por H (las 5 corridas completaron a tmax=0.5, ninguna abortó):**

| H | masa en sumideros | n sumideros |
|---|---|---|
| 1 | 1118.60 | 8 |
| 2 | 996.40 | 9 |
| 4 | 836.60 | 8 |
| 8 | 808.40 | 8 |
| 16 | **705.00** | 7 |

**La masa en sumideros DISMINUYE de forma monótona con H** (1118.6 → 996.4 → 836.6 → 808.4 → 705.0), la
dirección **opuesta** a la predicción cuantitativa de O-N7.7(b) ("la estructura debería CRECER, o al
menos no quedar plana, con más historia H"). El número de sumideros se mantiene aproximadamente estable
(7-9) mientras la masa total cae — es decir, con más "historia" (más tandas de reorganización acotada) se
forma un número similar de sumideros pero MÁS PEQUEÑOS en conjunto, no más estructura.

---

## 3 · ¿El criterio de falsación pudo ponerse a prueba de verdad esta vez?

**Sí, en los dos sistemas — a diferencia del intento anterior, que dio cero en absolutamente todo.**

- **Sistema A / O-N7.7(a) (ganancia marginal decreciente esperada):** con el único intervalo limpio
  disponible (N=2000→4000, ambos a tmax completo), la ganancia marginal fue **positiva y grande, sin
  señal de saturación** — el dato disponible **no confirma** la predicción de O-N7.7(a) en ese tramo. El
  tercer punto (N=8000) quedó sin poder cerrarse limpio por el colapso de timestep, así que el sweep
  completo (3 puntos) no llegó a completarse como se planeó — pero el criterio SÍ se puso a prueba en el
  tramo que sí corrió, y ahí el resultado apunta en contra de la saturación, no a favor.
- **Sistema B / O-N7.7(b) (crecimiento con historia esperado):** las 5 corridas completaron limpio a
  tmax=0.5, con memoria genuina verificada en el grafo (Jaccard no degenerado). El resultado es claro y
  **va en la dirección OPUESTA** a la predicción: la masa cae con H, no crece.

---

## 4 · Explicación en simple, con analogía

Es como el ejemplo del horno y el pan de la vez pasada — pero esta vez sí pusimos suficiente masa de pan
en el horno para que cocinara algo, así que el termómetro por fin marcó una temperatura real en vez de
quedarse en cero.

- **Sistema A** es como preguntar "si le doy al panadero el mismo presupuesto total de harina pero se lo
  reparto en panes cada vez más chiquitos (más panes, cada uno más liviano), ¿sigue formando la misma
  cantidad de masa horneada, o empieza a rendir menos por pan adicional?" En el único tramo que se pudo
  medir limpio (de 2000 a 4000 panes), el panadero rindió MÁS, no menos — no se vio "cansancio" todavía.
  El tramo siguiente (8000 panes) se cortó a mitad de horneado porque el horno empezó a tardar
  segundo a segundo real por cada segundo de cocción — no alcanzó el tiempo para terminarlo bien.
- **Sistema B** es como preguntar "si un bibliotecario reorganiza la MISMA biblioteca cada vez más veces
  (más pasadas de poda/reconexión), ¿los libros quedan mejor acomodados con el tiempo (biblioteca más
  sólida) o se van perdiendo libros en el reordenamiento?" El resultado acá es claro: con más pasadas del
  bibliotecario, la biblioteca terminó con MENOS masa acumulada en las secciones bien formadas — como si
  cada reorganización, en vez de consolidar, dispersara un poco más de lo que ya estaba junto.

---

## 5 · Qué falta / limitaciones honestas

- N=16000 (Sistema A) no se corrió — no quedó presupuesto tras el colapso de timestep en N=8000.
- N=8000 (Sistema A) no llegó a tmax=0.5 — el número reportado (122 sumideros, masa 3188.95) es un estado
  a t=0.361 (72% del tiempo total), no comparable 1 a 1 con los otros dos puntos que sí completaron. No
  se investigó la causa raíz del colapso de timestep (posiblemente relacionada con múltiples sumideros
  muy próximos entre sí formándose casi simultáneamente a esta resolución — sólo una hipótesis, no
  verificada).
- El atajo de `dens_bar` por bootstrap (en vez de extracción física independiente por N) es una
  simplificación de presupuesto de tiempo, documentada en los scripts — no reemplaza una extracción real
  si algún experimento futuro necesita la identidad física exacta del pool a cada N.
- `iters_layout=25` (en vez de 100) es una reducción deliberada y documentada, aplicada de forma uniforme
  en A y B — no se midió cuánto cambiaría el resultado con el default de 100 iteraciones a esta escala.
- Sistema B usó el mismo mecanismo de reorganización acotada tal cual (sin re-tunear
  `TAM_MUESTRA_RECONSIDERACION` ni las semillas) — el cambio de escala fue mínimo como anticipó el
  encargo, y el resultado (masa decreciente con H) es limpio y no ambiguo, pero sólo se probó UN diseño de
  mecanismo de historia; no se exploró si otro mecanismo de reorganización daría la dirección opuesta.

**No se declara cierre ni veredicto sobre O-N7.7 ni sobre CS073.** Se reportan los números tal como
salieron, con toda su fuerza en ambas direcciones — la lectura final es de Alexis.

---

## Archivos

- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/ON77_sistemaA_cierre.py`,
  `ON77_sistemaB_cierre.py` — código nuevo, no toca nada congelado.
- `/Users/alexis/phantom_cs073/ON77_sistemaA_cierre/ON77_sistemaA_cierre_resultado.json`,
  `/Users/alexis/phantom_cs073/ON77_sistemaB_cierre/ON77_sistemaB_cierre_resultado.json` — datos crudos.
- `logs/ON77_sistemaA_cierre_run.log`, `logs/ON77_sistemaB_cierre_run.log` — bitácora de ejecución
  completa (incluye el detalle del colapso de timestep en N=8000).
- `/Users/alexis/phantom_cs073/ON77_sistemaA_cierre/ic_N{2000,4000,8000}/`,
  `/Users/alexis/phantom_cs073/ON77_sistemaB_cierre/ic_H{1,2,4,8,16}/` — ICs, `cosmog.in`, dumps binarios,
  `.sink`, `setup.log`, `run.log` de cada corrida.
