# DISEÑO — Barrido fino de todas las variables: ¿la estructura vive en una banda estrecha no azarosa?
### "¿Hay una banda angosta del espacio de configuraciones donde emerge estructura, o emerge en todos lados por igual?"
#### Diseño para CC · leer entero antes de tocar código · NO cambiar el motor, solo barrerlo

**Director:** Alexis López Tapia · **Diseñó:** Claude Science (CS) · **Fecha:** 26-jul-2026
**Motor:** `cs074_energia_holistica.py`, función `correr_holistico_energia` — NO se modifica.
Este experimento solo la LLAMA con muchas configuraciones. Si algo del motor no permite lo
que pide el diseño, PARÁ y reportá a CS — no lo arregles por tu cuenta.

---

## La pregunta, en simple

Ya sabemos dos cosas de los experimentos previos:
- La estructura tiene un **techo** en la asimetría: colapsa en ε≳3,8 (experimento A).
- Pero solo barrimos ε, la reserva de energía y las semillas — **todo lo demás quedó fijo**
  (poblaciones de partículas, tasa de expansión, las fuerzas). Así que no sabemos si la
  estructura vive en una **banda estrecha** del espacio completo, o si aparece en casi
  cualquier configuración.

**Esta es la diferencia entre azar puro y ley:** si movés TODAS las variables juntas y la
estructura solo emerge en una franja angosta y conexa del espacio, eso es una **banda no
azarosa** (una condición real). Si emerge en cualquier lado, o en puntos dispersos sin
patrón, no hay banda — es trivial o es ruido. Esa es la pregunta que este barrido contesta.

---

## Qué se barre (TODAS las variables físicas juntas, no de a una)

Rangos **mucho más amplios que lo que suponemos** que funciona — la regla del director:
barrer siempre más allá del valor esperado, para que el resultado no dependa de dónde
miramos.

| Variable | Símbolo en el motor | Rango a barrer | Por qué |
|---|---|---|---|
| Asimetría inicial | `amp_rugosidad` | 1e-6 a 10 (log) | incluir MUY abajo — el experimento A no encontró borde inferior; hay que buscarlo |
| Tasa de expansión | `tasa_expansion` | 0,001 a 0,2 (log) | nunca se barrió; es la palanca de enfriamiento/dilución |
| Reserva de energía | `E_reserva` | 0,001 a 1000 (log) | ya sabemos que satura, pero entra al barrido conjunto |
| Población de quarks | `nq` | 150 a 600 | cambia la estequiometría y la densidad |
| Razón antiquark/quark | `naq/nq` | 0,5 a 0,95 | la asimetría de aniquilación (cuánto sobrevive) |
| Razón electrón/quark | `ne/nq` | 0,15 a 0,6 | carga y recombinación |
| Pasos de estructura | `n_pasos_estructura` | 60 (fijo) o 60/120 | control de si 60 pasos era corto (duda abierta de B) |

**Las fuerzas** (`cdm_on`, `cooling_on`, `expansion_on`, `gravedad_on`, `energia_on`) se
dejan TODAS en `True` en el barrido principal (queremos el proceso completo). Su apagado ya
se probó como admisibilidad en experimentos previos — no se re-hace aquí.

## Cómo se barre — muestreo, no rejilla densa (para que quepa en cómputo)

Una rejilla densa de 7 variables explota (millones de corridas). Y para detectar "banda
estrecha" el muestreo aleatorio del espacio completo es **mejor** que la rejilla, porque no
privilegia ningún eje. Usar:

- **Muestreo Latin Hypercube (LHS)** del espacio de 6 variables continuas (las de arriba
  salvo pasos), en escala log donde corresponde. **N = 2000 configuraciones.** Si el tiempo
  lo permite, 4000.
- **12 semillas por configuración** (`seed_layout` variando) — para separar señal de ruido
  de semilla en cada punto.
- Sin miedo al cómputo: si son 2000×12 = 24.000 corridas y toma toda la noche, está bien
  (instrucción explícita del director). Reportá tiempo estimado tras un smoke de 20 configs
  antes de lanzar el full.

## Qué se mide (los observables, todos ya en el motor)

Por cada configuración (promediando sus 12 semillas):
1. **`frac_masa_ligada`** — cuánta estructura se formó (el observable central).
2. **`n_clusters_finales`** y **`frac_masa_en_mayor_cluster`** — la forma de la estructura.
3. **`fraccion_materia`** (si el motor la expone) — para ver si el 5% aparece en alguna
   banda y no en otras.

## El NULL (obligatorio — sin esto no hay veredicto)

Para cada configuración, un brazo **NULL** que rompe la estructura conservando las
cantidades: **barajar las densidades bariónicas** (`seed_dens_null` — el motor ya lo
soporta, línea 148-152). Si REAL forma estructura donde NULL no, la estructura es real; si
REAL≈NULL, es artefacto. **z = (frac_real − frac_null)/sd_null** por configuración.

## Cómo se lee el resultado (pre-inscrito — las tres salidas posibles)

1. **BANDA ESTRECHA (el resultado fuerte):** la estructura con z>2 (REAL bien por encima de
   NULL) vive en una región **angosta y conexa** del espacio — p.ej. "solo cuando ε<0,5 Y
   tasa_expansión en tal rango Y naq/nq>0,8". Eso es una **condición no azarosa**: hay una
   receta estrecha para que emerja estructura. → contingencia con estructura, el resultado
   más interesante.
2. **SIN BANDA / TODO EL ESPACIO:** la estructura emerge con z>2 en casi cualquier
   configuración. → la estructura es genérica (trivial), no hay condición fina.
3. **DISPERSO SIN PATRÓN:** los puntos con z>2 están salpicados sin región conexa. → es
   ruido/azar, no hay banda real.

**Cómo se distingue banda de dispersión (método, no a ojo):** sobre las configuraciones con
z>2, medir si forman un cúmulo conexo en el espacio de parámetros (p.ej. proporción de
vecinos-más-cercanos que también son z>2, contra su propio NULL de etiquetas barajadas). Una
banda tiene vecindad coherente; el ruido no. **Reportar esa métrica, no solo el mapa.**

## Reglas (las de siempre)

- **Nada a mano:** ningún valor se fija para que dé estructura; se barren rangos amplios y se
  reporta la curva/mapa completo.
- **El 5% (y cualquier número físico) es test contra la SALIDA, jamás entrada.** Si aparece
  en una banda, se reporta; nunca se centra el barrido en él.
- **No tocar el motor.** Solo llamarlo. Un desacuerdo con el diseño es un dato: PARÁ y
  reportá a CS.
- **Correr completo, no por partes.** Smoke de 20 configs para estimar tiempo; luego el full
  de una vez.
- **Verificar en disco:** guardar el JSON crudo con todas las configuraciones, sus 12
  semillas, REAL y NULL — para que CS pueda recorrer el barrido, no solo leer el resumen.

## Entregables

- `cs074D_barrido_fino_banda.py` (el script que llama al motor)
- `resultados_cs074D_barrido_fino/` con el JSON crudo (todas las configs × semillas × REAL/NULL)
- `RESULTADO_cs074D_barrido_fino_PARA_CS.md` — el mapa, la métrica de conexidad de la banda,
  y cuál de las tres lecturas dio, con la curva/mapa completo a la vista.

---

*Este barrido contesta la pregunta que los anteriores dejaron abierta: ¿la estructura (y los
números) viven en una banda estrecha no azarosa del espacio completo de configuraciones, o
no? Está diseñado para poder dar cualquiera de las tres respuestas — esa es su virtud.*