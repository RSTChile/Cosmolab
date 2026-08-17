# CS075 — Los 23 sobre base física, con puerta de emergencia: resultado

**Fecha:** 2026-07-30 · Arquitectura verificada (5/6 pruebas), smoke de 4 configuraciones
corrido y limpio. Un hallazgo de tiempo requiere tu decisión antes de la corrida "real"
que pide la instrucción. Nada se cierra.

---

## 1. La ambigüedad del §3.3 — resuelta, verificada en disco

La instrucción admitía que su propio desglose sumaba 22, no 23, y pedía verificar contra
el manifiesto antes de implementar. Se verificó:

`MANIFIESTO_FOLD_CS072.md` línea 3 y 22-23: **"21 = 18 ELEMENTOS + 3 MECANISMOS DE
ORIGEN"** — los tres mecanismos (M1, M2, **y M3**) ya están dentro del conteo base de 21,
no dos. `cs072_motor_23.py` (línea 29) declara M3 "fase cuántica... FUERA salvo acople
sin grilla; ausencia declarada" — SÍ cuenta en el inventario, pero su efecto está
arquitectónicamente ausente en cualquier motor sin representación de fase cuántica, como
éste (densidad/temperatura, no amplitud/fase).

**El elemento que faltaba es M3, no #24 tiempo.** #24 no es parte del inventario canónico
(el manifiesto cierra en 18+3+2=23 sin dejar lugar a un 24º elemento numerado; otro
documento del proyecto ya había fijado esa misma distinción). Implementado: M3 en Nivel 0
(junto a M1, ambos mecanismos de origen), depósito cero exacto siempre — no una casilla
de falsación (esas SÍ pueden actuar y su nulo es un resultado físico medido), sino una
ausencia arquitectónica declarada.

## 2. Un bug real encontrado y corregido (mío, no de la base protegida)

Al correr E5 (¿se mantienen las 6 direcciones?), `rho` subía en vez de bajar — 131× en
2000 pasos. Se diagnosticó midiendo, no razonando: el culpable dominante era el depósito
de `22_qcd` (147.682 de un total de ~150.000, 31× el siguiente). Su fórmula usaba una
razón adimensional al cuadrado que, sobre una distribución lognormal (la que ya usa
`EstadoFisico`, asimétrica de cola larga), no suma cero — inyecta densidad neta cada
paso. `2_gravedad` tenía el mismo defecto (sólo sumaba, nunca restaba). Un segundo
problema, más sutil, apareció después de corregir el primero: agentes con salida
adimensional acotada (`tanh`, la razón al cuadrado) no se achican cuando `rho` se diluye
— cerca del piso numérico de la base (`np.maximum(rho, 1e-30)`), eso rompe la
conservación en la práctica aunque la fórmula sea "cero neto" en el papel.

**Se verificó que la base (`EstadoFisico`) está sana:** corrida sin ningún agente, sola,
diluye correctamente de 1,0 a 0,00097 en 2000 pasos, exactamente lo esperado. El bug era
enteramente mío (las fórmulas de depósito que inventé), no de los archivos protegidos —
así que se corrigió directamente, no se paró a reportar (la instrucción reserva "pará y
reportá" para lo que no se puede anclar o lo que requeriría tocar los archivos cerrados).
Corregidos: `22_qcd`, `2_gravedad`, `17_oscuro`, `10_enfriamiento`, `5_debil`,
`6_catalogo`, `M1_semilla` — todos ahora redistribuyen (suma espacial exacta cero) y
escalan con la densidad actual, no con una escala absoluta fija.

## 3. Un hallazgo que NO se corrigió — reportado tal cual

Con el bug de arriba resuelto, `T`, `rho` y `S` se mantienen en la dirección esperada,
pero **la exergía X sube** (0,0099→0,0567) en vez de bajar. No se forzó a bajar tocando
más fórmulas. Lectura física: los agentes de estructura local (`2_gravedad`, `3_fuerte`)
concentran densidad en regiones sobredensas — generan diferencia local activamente,
compitiendo contra la homogeneización de la difusión. Es plausible que sea un efecto
real, no un bug: la base sola (sin agentes) sí bajaba X en sus 6/6 direcciones
verificadas, pero esa prueba no tenía ningún proceso que CREARA estructura local. No se
adjudica — se reporta como la única dirección que no se sostiene, con su explicación
candidata.

## 4. Resultado de las 6 pruebas (E1-E6)

| Prueba | Resultado |
|---|---|
| E1 — inventario completo (23, sin repetidos, nombres justificados) | **PASA** |
| E2 — nadie madruga (LA TESIS) | **PASA** — cero violaciones |
| E3 — cero exacto dormido | **PASA** |
| E4 — cero turnos (orden de consulta no cambia el resultado) | **PASA** — diferencia relativa 0,0 |
| E5 — las 6 direcciones se mantienen | **FALLA** — X sube (ver §3) |
| E6 — orden esperado entre niveles | **PASA** — Nivel 0→1→3 en orden, sin inversiones |

## 5. Un hallazgo de tiempo que SÍ requiere tu decisión — acá paré

`T_bajo_electrodebil` se ancló en `T_inicial` (arranca ahí) y `T_bajo_confinamiento` se
derivó de la razón real 159 GeV/155 MeV ≈ 1026, como pide la instrucción. **Medido, no
asumido: cruzar esa temperatura con `dt=1e-3` y `k_enfriamiento=50` (los defaults) toma
~21 millones de pasos — aproximadamente 20 horas por configuración, no "minutos".**

La instrucción pide 4 configuraciones "smoke... minutos" que crucen el confinamiento —
eso, tal cual, son ~80 horas de cómputo (o ~20 horas corriendo las 4 en paralelo). No lo
corrí sin avisar: es exactamente el tipo de hallazgo que la instrucción pide reportar
antes de comprometer cómputo grande.

## 6. El smoke que SÍ corrí (T_total=20, ~4 minutos, las 4 configuraciones)

Alcanza Nivel 0→1 siempre, y Nivel 3 (sobredensidad) cuando ε≥0,5:

| ε (amp_asimetria) | despiertos | nivel más alto alcanzado | nota |
|---|---|---|---|
| 0,01 | 10/23 | Nivel 1 | sobredensidad nunca aparece en este T_total |
| 0,1 | 10/23 | Nivel 1 | igual que 0,01 |
| 0,5 | 12/23 | Nivel 3 | sobredensidad YA en el paso 0 (condición inicial) |
| 2,0 | 12/23 | Nivel 3 | igual que 0,5; 806 celdas ya "núcleo" en persistencia cruda, pero el hito `hay_nucleos` correctamente sigue en False (falta T_bajo_confinamiento) |

**Nadie madruga, en las 4 configuraciones, sin excepción.** Los 11-13 agentes dormidos
(Niveles 2, 4, 5, 6) tienen su hito faltante identificado con precisión — es exactamente
lo que la instrucción anticipa como resultado válido, no un fallo.

## 7. Dos hitos sin anclaje — declarados, no inventados en silencio

`hay_atomos` y `hay_red` no tienen ninguna fuente en el proyecto: `EstadoFisico` tiene un
solo campo escalar (densidad/temperatura), sin carga ni especies — no hay forma de
definir "neutro" sin agregar una distinción que la base no tiene. Se implementó una
interpretación DECLARADA (persistencia extendida sobre las mismas celdas sobredensas, y
conteo de regiones conexas) usando solo lo que la base ya ofrece, marcada explícitamente
como no verificada en el código y aquí. No se probó en este smoke (nunca se alcanzó
`hay_nucleos`, su precondición).

## 8. Qué decidís vos

1. **¿Corremos el cruce real de confinamiento?** ~20h por configuración, ~80h en serie o
   ~20h en paralelo. Puedo lanzarlo en background (como los experimentos anteriores) si
   lo autorizás.
2. **La dirección de X (§3):** ¿es un hallazgo real (agentes de estructura compiten con
   la homogeneización) o preferís que lo acote más antes de seguir?
3. **`hay_atomos`/`hay_red` sin anclaje (§7):** ¿la interpretación declarada te sirve, o
   hace falta agregar carga/especies a `EstadoFisico` (tocando la base cerrada, lo cual
   requeriría tu autorización explícita)?

**Archivos:** `cs075_23_sobre_fisica.py`, `cs075_pruebas_23_sobre_fisica.py`,
`cs075_smoke_23_sobre_fisica.py`, `cs075_resultado_pruebas_23_sobre_fisica.json`,
`cs075_resultado_23_sobre_fisica.json` — todos en `Cosmogenesis/Campo_Continuo_Estigmergico/`.

---

## ADENDA 2026-07-29 — "¿Por qué 20 horas?" no era la pregunta correcta

Pediste revisar el número de §5 comparándolo contra simulaciones cortas anteriores del
proyecto. La revisión encontró algo más serio que un reloj mal calibrado: **con los
parámetros default, la corrida real nunca hubiera terminado, sin importar cuánto tiempo
se le diera.** Colapsaba en NaN a los ~50.000 pasos (~4 minutos), 0,24% del camino a
confinamiento — no por lentitud, sino por una divergencia numérica genuina en fórmulas
mías. Se encontró, se corrigió, se verificó por medición (no por argumento), y el número
de §5 queda reemplazado por uno honesto: más abajo.

**Qué se hizo, en orden:**

1. Confirmé por código (no por lectura de la instrucción) que `cs074_energia_holistica.py`
   acopla su constante de enfriamiento al PASO crudo, mientras que `cs075_base_fisica.py`
   la acopla a `t=paso·dt` con `dt=1e-3` — de ahí el desajuste de ~1000× que produce los
   ~21 millones de pasos de §5. Esa parte del diagnóstico original era correcta.
2. Antes de aceptar "entonces subamos dt", lo probé: a `dt=1.0` el sistema completo
   revienta en el PASO 0 (la dilución del primer paso es tan violenta que `rho` cae
   exactamente al piso numérico en todas las celdas, `rho.std()=0`, y la división de
   `A5_Debil` por ese cero produce NaN). Un barrido de `dt` (0.001 a 0.1) mostró que
   incluso `dt=0.01` colapsa por *overflow* — sólo más tarde (~3.100 pasos) que `dt=0.1`
   (~1.200 pasos). Subir `dt` no arregla nada: sólo apura el mismo colapso.
3. Eso obligó a preguntar si el colapso también existe al `dt=1e-3` default, sólo que
   más lento. Corrida larga (150.000 pasos, sin filtrar nada): **sí existe** — `rho.max`
   pasa de 4,8e-5 (paso 40.000, sano) a 6,9e+289 (paso 50.000, roto) — el `NaN` llega
   inmediatamente después.
4. Medí, agente por agente, quién dominaba el depósito justo antes del colapso (mismo
   método que cazó el bug original de `22_qcd` en §2). El culpable: **`22_qcd` otra vez**,
   pero un defecto DISTINTO del ya corregido en §2. Su fórmula usaba `(ρ/media − 1)²` —
   el cuadrado de una razón sin acotar. Si una sola celda queda por encima de la media
   (por cualquier motivo, incluida la propia gravedad), este término crece con el
   CUADRADO de su exceso: `dx/dt ~ k·x²`, una ecuación de Riccati que diverge en tiempo
   FINITO, no sólo asintóticamente — exactamente el patrón medido (nada visible por
   48.000 pasos, overflow total en los 2.000 siguientes). El arreglo de escala de §2
   (reescalar por `rho.mean()`) resolvía el problema cerca del piso numérico, pero nunca
   tocaba este segundo defecto, más arriba en la escala.
5. Auditando el archivo completo busqué la misma clase de defecto (retroalimentación
   lineal o cuadrática sin saturar) en los 23 agentes. Encontré dos más, ninguno todavía
   causante del colapso medido pero con el mismo riesgo estructural: `2_gravedad`
   (retroalimentación lineal pura, sin saturar) y `17_oscuro` (copia exacta de la fórmula
   vieja de `2_gravedad`, gateada tras `hay_atomos` — nunca alcanzada en ninguna corrida
   hecha, así que no participó del colapso observado, pero divergiría igual si se
   activara). También `23_campo`, activo sin condición desde el paso 0, con el mismo
   patrón lineal (coeficiente más chico, 0,01, pero corriendo TODA la simulación sin
   apagarse). Los cuatro se corrigieron con el mismo patrón que ya usaba `5_debil`
   (`tanh` saturante + reescalado por `rho.mean()` actual) — no es una técnica nueva, es
   aplicar consistentemente un patrón que el archivo ya usaba en otro lado.
6. **Verificación por medición, no por argumento:** corridas de 40.000-80.000 pasos en
   `amp_asimetria` = 0,01 / 0,5 / 2,0 — las tres estables, sin NaN, muy por encima del
   punto donde antes colapsaba (48.000-50.000 pasos). Re-corrida completa de E1-E6:
   **E1,E2,E3,E4,E6 = PASA** (sin regresión). **E5 sigue en FALLA** — mismo hallazgo de
   §3 (X sube en vez de bajar), pero ahora más marcado (0,0099→0,90 en vez de →0,057):
   consistente con que los agentes de estructura, ya no a punto de romperse, concentran
   densidad de forma sostenida — refuerza la lectura física de §3, no la contradice.
   Re-corrido el smoke de 4 configuraciones: limpio, cero violaciones E2, y un hallazgo
   lateral — con el bug corregido, `amp=0,01` y `amp=0,1` (que antes NUNCA alcanzaban
   sobredensidad en T_total=20, enmascarado por el defecto) ahora SÍ la alcanzan (paso
   ~7.730) — el bug estaba ocultando estructura real, no sólo generando basura numérica.

**El número corregido, honesto, medido (no el de §5):**

Tasa medida post-arreglo: 6,05 ms/paso (amp=0,5, N=16, promedio sobre 80.000 pasos
reales, sin cortarse). `pasos_necesarios` no cambia (~21.045.558, es función de
`dt`/`k_enfriamiento`/la razón 159 GeV / 155 MeV, no de este bug):

**~35,4 horas por configuración** (antes reportaba ~20h — ese número no era una
subestimación optimista, era simplemente un cómputo que jamás iba a completarse). En
paralelo (4 configs a la vez), ~35,4h; en serie, ~141,5h (~5,9 días).

**Lo que cambió en tu decisión pendiente (§8.1):** ya no es "¿autorizás ~20h que de
todos modos iban a fallar?" — ahora es una corrida que, medido, sí puede completarse
técnicamente, a un costo mayor (~35h/config) del que se había reportado. Sigue
esperando tu autorización explícita antes de lanzarse; no se corrió sin avisar.

**Una cosa que quedó fuera, señalada pero no tocada:** `14_correlacion` (Nivel 5,
requiere `hay_atomos`) tiene una fórmula con la misma forma de riesgo (`vec·(ρ−media)`,
sin saturar) que las cuatro corregidas acá. Nunca se activó en ninguna corrida hecha
(`hay_atomos` no se alcanzó), así que no se midió que fallara — y no se lo corrigió a
ciegas, seríamos a "reasoning" y no a medición, contra la cultura del proyecto. Si algún
día se alcanza `hay_atomos` (staged run, dt distinto, etc.), revisar esto primero.

**Archivos tocados en esta adenda:** `cs075_23_sobre_fisica.py` (4 fórmulas de depósito
corregidas: `2_gravedad`, `22_qcd`, `17_oscuro`, `23_campo` — ninguna requirió tocar
`cs075_base_fisica.py` ni `cs075_arquitectura_agentes.py`, los archivos protegidos).
`cs075_resultado_pruebas_23_sobre_fisica.json` y `cs075_resultado_23_sobre_fisica.json`
re-generados con el estado corregido.

---

## ADENDA 2 — 2026-07-30 · Ejecución de `INSTRUCCION_CS075_v2_EJECUCION_PARA_CC.md`

CS entregó una v2 con dos correcciones sobre su propia instrucción v1. Paso 1 cambió el
panorama por completo. Paso 2 (bloqueante) falsó una explicación mía. No se llegó al
paso 3 — el protocolo lo bloquea, y se respeta.

### Paso 1 — el umbral estaba mal anclado, y el arreglo es enorme

Verificado en disco: `cs072_motor_23.py` l.42-43 fija `T_CONF=0.6` y `T_EW=0.9`
(normalizados, `T_ef=T.mean()` arrancando en ~1.0), usados tal cual en l.130/l.147. Mi
anclaje v1 (razón física 159 GeV/155 MeV ≈ 1026) inventaba una escala donde el proyecto
ya tenía la suya — el mismo tipo de error que M3/#24, ahora en los umbrales.

Corregido en `cs075_23_sobre_fisica.py` (`Proceso23SobreFisica.__init__`): `T_EW =
0,9·T_inicial`, `T_CONF = 0,6·T_inicial` (antes: `T_CONF = T_inicial/1026`). Medido, no
estimado — corrida directa hasta que el hito cambia:

| Umbral | Pasos analíticos | Pasos medidos |
|---|---|---|
| `T_bajo_electrodebil` | 4,69 | 5 |
| `T_bajo_confinamiento` | 35,56 | **36** |

**De ~21.045.558 pasos (~35,4 h) a 36 pasos (una fracción de segundo).** No es una mejora
incremental — el "cruce real de confinamiento" que en §5/§8.1 de la adenda anterior
pedía tu autorización para ~35h de cómputo ahora es trivial. La pregunta de §8.1 queda
obsoleta: ya no hace falta autorizar nada grande para cruzar confinamiento.

### Paso 2 (bloqueante) — E5b: mi explicación de §3 queda FALSA, medida directamente

El control que yo había invocado en la adenda anterior (`EstadoFisico` sola, sin
agentes, X baja) no probaba que `2_gravedad`/`3_fuerte` fueran la causa — la diferencia
entre esas dos corridas son los 23 agentes, no esos dos. CS pidió la prueba directa:
apagar sólo esos dos, dejar los otros 21, medir X. Resultado (`amp=0,1`, seed=7,
T_total=20, mismos parámetros que la batería E1-E6 original):

| Corrida | X inicial → final | ¿Baja? |
|---|---|---|
| Baseline, los 23 | 0,0099 → 0,6255 | NO |
| Sin `2_gravedad`/`3_fuerte` (21 agentes) | 0,0099 → 0,3703 | **NO** |
| SÓLO `2_gravedad`+`3_fuerte` (2 agentes) | 0,0099 → 0,0000024 | SÍ (casi a cero) |

**Mi explicación era falsa.** Apagar gravedad y fuerte no revierte la subida de X —
sigue subiendo casi igual (0,37 contra 0,63 del baseline). Y el resultado inverso es más
revelador todavía: encendidos SOLOS, sin los otros 21 compitiendo, gravedad+fuerte
llevan X casi exactamente a cero — el mecanismo que yo acusaba de "generar diferencia
local" en realidad HOMOGENEIZA cuando actúa sin oposición. La subida de X viene de
**alguno de los otros 21 agentes**, no identificado todavía.

Per protocolo (§2 de la instrucción v2, explícito): esto es un desacuerdo, y un
desacuerdo es un dato. **No se sigue al paso 3.** No se corre el smoke con el rango
corregido hasta que esto se resuelva. No se fuerza ni se ajusta nada para que X baje.

**Archivo nuevo:** `cs075_E5b_prueba_estructura_exergia.py` (script de la prueba directa,
reutilizable) y `cs075_resultado_E5b_prueba_estructura_exergia.json` (los tres resultados
completos, con nombres de agentes por corrida).

### Qué decidís vos, ahora

1. **¿Sigo buscando cuál de los otros 21 agentes genera la subida de X**, con el mismo
   método de medición por agente que cazó el bug de `22_qcd`? Puedo hacerlo — no lo hice
   todavía porque el protocolo pide parar y reportar, no seguir por mi cuenta.
2. Con el paso 1 resuelto, el smoke de 4 configuraciones (paso 3) ahora es barato en
   tiempo — pero sigue bloqueado por el paso 2 sin resolver.
3. La pregunta 1 de §8 de la adenda anterior (autorizar ~35h de cómputo) queda sin
   objeto: ya no hace falta.

**Archivos tocados en esta adenda:** `cs075_23_sobre_fisica.py` (umbrales `T_EW`/`T_CONF`
corregidos, docstring actualizado — nada más), `cs075_E5b_prueba_estructura_exergia.py`
(nuevo), `cs075_resultado_E5b_prueba_estructura_exergia.json` (nuevo). No se tocó
`cs075_base_fisica.py` ni `cs075_arquitectura_agentes.py`.

---

## ADENDA 3 — 2026-07-30 · Se encontró y corrigió la causa de la subida de X. E5 PASA.

Alexis pidió seguir por mi cuenta: encontrar todos los errores, corregirlos, prueba corta
y reportar. Esto es lo que se hizo, en orden, todo medido.

### Diagnóstico por agente (mismo método que cazó `22_qcd`)

Script nuevo `cs075_diag_quien_sube_X.py`: mide, por agente y por paso, la covarianza
entre su depósito y `(ρ−media)` — signo positivo acumulado = ese agente aumenta la
varianza/exergía; negativo = la reduce. Corrida completa (T_total=20, `amp=0,1`,
seed=7, igual que el baseline de E5b):

| Agente | cov acumulada | Lectura |
|---|---:|---|
| `22_qcd` | **+46,21** | dominante, ~9× el siguiente |
| `23_campo` | +9,24 | segundo, con refuerzo DECLARADO en su propio docstring |
| `5_debil` | +5,24 | tercero |
| `9_expansion` | −40,42 | mayor homogeneizador (no alcanza a compensar el total) |
| `10_enfriamiento` | −0,90 | homogeneizador menor |
| resto (17 agentes) | ≤10⁻³ en magnitud | irrelevantes para este efecto |

### La causa: dos agentes con la física equivocada, encontrada releyendo sus propios docstrings

`22_qcd` y `5_debil` heredaron, en el arreglo de la ADENDA 1 (para frenar el overflow),
la MISMA forma que `2_gravedad` (redistribución que refuerza donde ya hay exceso, vía
`tanh` recentrada) — una forma correcta para un agente de estructura, pero copiada sin
volver a verificar si correspondía a lo que cada agente decía representar:

- **`22_qcd`**: su propio docstring dice *"un término de auto-energía **proporcional a
  rho**, **ya presente desde el inicio como parte del vacío**"* — una energía de vacío
  es, por definición física, UNIFORME (no compite por estructura existente). La
  redistribución-que-refuerza era, leído contra su propio texto, la física equivocada.
- **`5_debil`**: su comentario decía explícitamente que copiaba la forma de `22_qcd`
  ("mismo problema de escala que #22") — una analogía de conveniencia numérica, no una
  razón física propia. Un cambio de sabor no tiene motivo conocido para concentrar o
  dispersar masa según el signo del exceso local.

**Corrección:** ambos pasan a ser fuentes UNIFORMES (mismo patrón que `M1_Semilla`, no
zero-sum — mecanismos intencionalmente globales, no redistribución), consistente con lo
que sus propios docstrings ya decían. `23_campo`, en cambio, **no se tocó**: su docstring
declara explícitamente la intención de reforzar ("se refuerza levemente cada paso como
recordatorio de que la rugosidad es un proceso") — tocar eso sería sobrescribir un
diseño declarado sin autorización, no corregir un error. Se midió si, dejándolo intacto,
el resultado agregado ya alcanzaba para que X baje — y alcanzó.

### Prueba corta: E1-E6 completa, todo PASA

`cs075_pruebas_23_sobre_fisica.py`, misma escala que siempre (T_total=20, ~224s):

| Prueba | Antes (ADENDA 1/2) | Ahora |
|---|---|---|
| E1-E4, E6 | PASA | PASA (sin cambio) |
| **E5** | **FALLA** (X: 0,0099→0,626, sube) | **PASA** (X: 0,0099→**0,0091**, baja) |

Las otras tres direcciones (T_baja, rho_baja, S_sube) se mantienen. **Batería completa:
PASA por primera vez en este experimento.**

### Verificación de que el rediseño no reabre el riesgo de divergencia

Corridas de 40.000 pasos en `amp_asimetria` = 0,1 / 0,5 / 2,0: las tres estables, sin
NaN, `rho.max`/`rho.mean` en el mismo orden de magnitud saludable de siempre. El cambio
a fuente uniforme es, si acaso, más simple/seguro que la redistribución `tanh` que
reemplaza (no divide por `rho.std()`, no tiene forma de reforzarse a sí mismo).

### Qué decidís vos

El experimento, en su arquitectura (23 agentes, umbrales anclados en el motor,
E1-E6 completa PASA), está en el mejor estado que tuvo hasta ahora. Quedan, sin
resolver todavía y sin que yo los haya tocado por mi cuenta:

1. El smoke de 4 configuraciones (paso 3 de la instrucción v2) — ya no bloqueado (paso 2
   cerró), y ahora barato en tiempo (confinamiento cruza en ~36 pasos). No lo corrí sin
   que me lo pidas explícitamente, para no adelantarme más allá de lo que se me encargó.
2. `14_correlacion` (Nivel 5, `hay_atomos`) sigue con la misma forma de riesgo sin
   auditar (señalado en la ADENDA 1, nunca activado en ninguna corrida hecha).
3. El coeficiente elegido para las nuevas fuentes uniformes de `22_qcd` (0,001) y
   `5_debil` (0,0004) es una elección razonable pero no derivada de ningún archivo del
   proyecto (no hay un "cuánto" anclado para energía de vacío QCD en unidades de este
   modelo) — se declara así, no se disfraza de medido.

**Archivos tocados en esta adenda:** `cs075_23_sobre_fisica.py` (`22_qcd` y `5_debil`
rediseñados a fuente uniforme; docstring de `5_debil` actualizado con los umbrales v2),
`cs075_diag_quien_sube_X.py` (nuevo), `cs075_resultado_diag_quien_sube_X.json` (nuevo),
`cs075_resultado_pruebas_23_sobre_fisica.json` (re-generado, E5 ahora PASA). No se tocó
`cs075_base_fisica.py` ni `cs075_arquitectura_agentes.py`.

---

## ADENDA 4 — 2026-07-30 · Paso 3 (smoke con el rango corregido): corrido, cierra la v2

Con el paso 2 cerrado (ADENDA 3), el paso 3 quedó desbloqueado. Actualizado
`cs075_smoke_23_sobre_fisica.py` a `T_TOTAL=5,0` (5.000 pasos, ~140x margen sobre el
cruce de confinamiento en el paso 36) y corrido, las 4 configuraciones de v2§3:

| `amp_asimetria` | despiertos | nivel más alto | hitos extra |
|---|---|---|---|
| 0,01 | 15/23 | 2 (confinamiento) | — |
| 0,1 | 15/23 | 2 (confinamiento) | — |
| 0,5 | 17/23 | 3 (sobredensidad, desde paso 0) | — |
| **2,0** | **23/23** | **6 (red)** | `hay_nucleos`, `hay_atomos`, `hay_red` — **primera vez en todo el experimento** |

**`amp_asimetria=2,0` activó el inventario completo**: 186 celdas sobredensas → 184
núcleos → 183 átomos → 15 regiones conexas (`hay_red=True`). Los dos hitos que la ADENDA
1 marcó como "declarados, no anclados" (`hay_atomos`, `hay_red`) se pusieron a prueba por
primera vez con datos reales, no sólo en el diseño — incluyendo las dos casillas de
Nivel 6 (`18_poda`, `15_causal`) que nunca habían despertado en ninguna corrida anterior
de este experimento.

**Costo medido** (v2§3 "qué entregar"): 1,31 ms/paso (`amp=0,01`, menos agentes activos)
a 4,28 ms/paso (`amp=2,0`, los 23 activos). **Las 4 configuraciones completas: 46,9
segundos** — el smoke que la instrucción original pedía "en minutos" ahora lo es de
verdad, sin ajustar nada más que el umbral corregido del paso 1.

**Nadie madrugó** en las 4 configuraciones (mismo chequeo que E2/E6, verificado en el
orden de despertar registrado): Nivel 0 en paso 0, Nivel 1 en paso 5, Nivel 2 en paso 36,
Nivel 3 sólo cuando `hay_sobredensidad` ya estaba en la condición inicial, Niveles 4-6
sólo en `amp=2,0`, todos en paso 36 (mismo paso que Nivel 2 — consistente: la corrida
evalúa hitos ANTES de aplicar depósitos ese paso, así que si la persistencia ya alcanzaba
el umbral, varios niveles pueden abrirse en el mismo paso en que se cruza confinamiento).

**Acá paro**, como pide v2§3: no se corrió barrido grande, no se cierra el experimento.
Archivo entregado: `cs075_resultado_23_sobre_fisica_v2.json` (registro completo, tabla de
despertar, dormidos con hito faltante, costo por configuración — igual que pide v2§3).

**Archivos tocados en esta adenda:** `cs075_smoke_23_sobre_fisica.py` (T_TOTAL=5,0, nombre
de salida `_v2.json`, registro de costo por config — nada más),
`cs075_resultado_23_sobre_fisica_v2.json` (nuevo). No se tocó `cs075_base_fisica.py` ni
`cs075_arquitectura_agentes.py`.

---

## ADENDA 5 — 2026-07-30 · El barrido v3: 160 configuraciones, borde nítido, sin techo, sin ruido de semilla

### Un bug propio encontrado y corregido antes de correr nada grande

Al escribir `cs075_barrido_v3.py` para registrar en qué paso se cumple cada hito, llamé a
`proceso._hitos()` una segunda vez por paso (además de la que ya hace `proceso.paso()`
internamente). `_hitos()` tiene efecto secundario: actualiza `contador_sobredenso`, el
contador de persistencia detrás de `hay_nucleos`/`hay_atomos`/`hay_red`. Llamarla dos
veces por paso duplica ese contador. Encontrado ANTES de correr el barrido completo (una
prueba de 2 configuraciones tardó el doble de lo esperado, lo que disparó la revisión).
**Corregido sin tocar `cs075_23_sobre_fisica.py`**: el paso en que cada hito se cumple se
deriva de `paso_despertar` de los agentes (cada `requiere` es de un solo hito, verificado
por grep) en vez de leer `_hitos()` de nuevo.

**Nota sobre el smoke anterior (ADENDA 4):** `cs075_smoke_23_sobre_fisica.py` tenía el
mismo patrón (una llamada extra a `_hitos()` mientras buscaba el paso de confinamiento),
pero sólo durante los primeros 36 pasos (se apaga sola al encontrar el hito). Revisado:
no cambia ninguno de los resultados ya reportados — el umbral de persistencia
(`MIN_PERSISTENCIA=5`, `FACTOR_ATOMOS`×2=10) queda muy por debajo de 36 incluso sin la
duplicación, así que `hay_nucleos`/`hay_atomos` cruzan exactamente en el mismo paso con o
sin el bug. `n_despiertos` y `nivel_maximo` nunca dependieron de esa llamada extra (vienen
de `paso_despertar`, que sólo lee el `_hitos()` interno de `proceso.paso()`). No hace
falta re-correr el smoke.

### El barrido: 160 configuraciones, 599 s con 8 procesos

20 amplitudes (`np.geomspace(0,05, 6,0, 20)` — reproduce sola los 4 tramos que pedía la
instrucción, sin espaciar cada uno por separado) × 8 semillas fijas (12345 + 1..7).
Costo real medido: 29,1 s/config en promedio bajo carga compartida con otros procesos de
este Mac (AudioServer, observation_bus) — más lento que la estimación de CS (11,72
s/config, extrapolada del smoke sin esa carga), pero con 8 procesos igual cerró en 599 s
(~10 min), dentro de lo razonable. Los 160 resultados completos están en
`cs075_resultado_barrido_v3.json`.

### 5.1 — El borde: un escalón, no una rampa

| amp | media n_despiertos | rango | semillas en 23/23 |
|---|---|---|---|
| 0,050 – 0,799 (12 puntos) | 15,0 – 17,0 | sin variación | 0/8 |
| **1,028** | **20,75** | **17 – 23** | **3/8** |
| 1,323 – 6,000 (7 puntos) | 23,0 | sin variación | 8/8 |

**Cruza 0,5 entre amp=1,028 y amp=1,323. Es un escalón**: sólo UN punto de la grilla
(1,028) tiene resultado mixto — todo lo anterior está siempre incompleto, todo lo
posterior siempre completo. Debajo del borde, los agentes dormidos son siempre los
mismos por nivel: `2_gravedad`/`12_localidad` (Nivel 3, faltan de 0,05 a ~0,1) y luego
`4_em`, `14_correlacion`, `M2_memoria`, `17_oscuro`, `18_poda`, `15_causal` (Niveles 4-6,
faltan de ~0,1 a 1,028) — nunca un agente "salteado" fuera de orden.

### 5.2 — La banda: no hay techo, al menos hasta 6,0

Desde amp=1,323 hasta el borde superior de la grilla (6,0) — incluyendo los 4 puntos que
cs074A ubica en "colapso" (>3,8) — **las 8 semillas dan 23/23 sin excepción**. El
inventario completo no vuelve a caer. **Vive en una semirrecta dentro del rango
explorado, no en una banda acotada.** No se puede afirmar que no haya techo más allá de
6,0 — no se barrió más lejos (v3§7 lo prohíbe sin autorización).

### 5.3 — La semilla: el resultado es del modelo, con una sola excepción, exactamente en el borde

**Desviación estándar = 0,000 en 19 de los 20 puntos de la grilla.** Las 8 semillas dan
EXACTAMENTE el mismo `n_despiertos` en todos lados, salvo en amp=1,028 (std=2,332), el
único punto de transición — ahí los 8 valores son 23,23,17,23,21,17,21,21: un reparto
genuino, no ruido disperso. **Esto es la lectura más informativa de las dos que la
instrucción anticipaba**: la dispersión NO es alta en todas partes — es cero en casi toda
la grilla y se concentra exactamente donde debería si el borde fuera real. El 23/23 del
smoke **no fue suerte de la semilla 12345**: 12345 da 23/23 en amp=1,028 (una de las 3
semillas "adelantadas" en ese punto mixto), pero el resultado se sostiene igual de sólido
con las otras 7 semillas en todo el resto de la grilla.

### El dato que había que resolver: ¿coincidencia con la banda de fragmentación de cs074A?

La banda de fragmentación de cs074A (0,9–2,3) contiene TANTO el punto de transición
mixto (1,028) COMO los primeros puntos ya completos (1,323 / 1,702 / 2,190). **Coincide
en el borde de entrada** (la transición cae dentro de esa banda) **pero no en el de
salida**: cs074A marca "colapso" a partir de ~3,8, y ahí el inventario sigue
perfectamente completo (8/8, sin excepción) hasta 6,0. Lectura honesta: puede ser la
MISMA escala física la que dispara ambos fenómenos en su arranque, pero no hay evidencia
de que sean el mismo mecanismo completo — la fragmentación de cs074A tiene una salida
(el colapso, >3,8) que este barrido no reproduce en absoluto. Dos fenómenos que
podrían compartir un origen y ciertamente no comparten un final, sin forzar una
conclusión más fuerte que eso.

### Qué NO contesta este barrido (v3§8, vigente)

No prueba que la estructura sea materia en sentido fuerte — prueba que el inventario
arquitectónico se completa. No hay control NULL (excluido de esta etapa por vos). Malla
fija en 16³: si el borde depende del tamaño de la malla, este barrido no lo vería.

### Qué decidís vos

El barrido está completo y adjudicable, per v3§6. No se corrió nada más fino ni se
extendió la grilla sin autorización, aunque el borde exacto quede entre dos puntos.
Preguntas abiertas para vos/CS:
1. ¿Correr una malla más fina alrededor de amp≈1,0-1,03 para localizar el borde con más
   precisión que "entre estos dos puntos"?
2. ¿Extender la grilla más allá de 6,0 para buscar el techo que no apareció?
3. ¿Vale la pena una corrida con NULL ahora que hay una arquitectura completa medida, o
   se mantiene excluido por ahora?

**Archivos entregados:** `cs075_barrido_v3.py`, `cs075_resultado_barrido_v3.json` (registro
completo de las 160 configuraciones + tabla de 20 filas + los tres análisis). No se tocó
`cs075_base_fisica.py`, `cs075_arquitectura_agentes.py` ni `cs075_23_sobre_fisica.py`.

---

## ADENDA 6 — 2026-07-30 · v4: los dos controles. Uno confirma, el otro deja la tesis sin evidencia

### N1 — sensibilidad de umbrales: el borde es del modelo

`MIN_PERSISTENCIA` ∈ {3,5,10} × `FACTOR_ATOMOS` ∈ {1,2,4}, 9 combinaciones, sólo el tramo
del borde de v3 (6 amplitudes × 8 semillas = 48 configs/combinación, 432 en total).
Parametrizado sin tocar `cs075_23_sobre_fisica.py`: ambas constantes se leen por nombre
global dentro de `_hitos()` en cada llamada (Python hace `LOAD_GLOBAL` tardío) — parchear
`modulo.MIN_PERSISTENCIA` antes de construir el proceso cambia el valor que la corrida ve,
verificado en disco antes de correr nada grande.

**Resultado: 9/9 combinaciones dan el mismo borde exacto**, entre amp=1,028 y amp=1,323 —
sin excepción, sin desplazamiento de un solo escalón de grilla:

| MIN_PERSISTENCIA | FACTOR_ATOMOS | fracción 23/23 en el punto mixto (1,028) |
|---|---|---|
| 3, 5, 10 | 1, 2 | 0,38 (igual en 8 de las 9 combinaciones) |
| 10 | 4 | 0,25 (la única variación, y sigue del mismo lado) |

**Veredicto (pre-inscrito en v4§2): EL BORDE ES DEL MODELO.** Robusto a un rango 3,3× en
`MIN_PERSISTENCIA` y 4× en `FACTOR_ATOMOS` — los umbrales fijan detalles de la fracción
exacta en el punto mixto, no la posición del borde. `cs075_N1_sensibilidad_umbrales.py` +
`cs075_resultado_N1_sensibilidad_umbrales.json`. Costo real: 2.341 s (8 procesos, ~39 min
— más que los ~26 min estimados por CS, misma carga compartida de siempre).

### N2 — puertas permutadas: la cascada es inevitable, la tesis queda sin evidencia (no refutada)

20 permutaciones de la asignación agente↔hito (preservando el histograma exacto 4/5/2/1/3/2),
generadas con semillas declaradas 1..20, ninguna degenerada (ninguna coincide con la
asignación real). Cada una × 8 semillas de física × 3 amplitudes (0,799 bajo el borde,
1,028 en el borde, 1,702 sobre el borde) = 480 configuraciones. El brazo real se reusó del
JSON de v3, no se corrió de nuevo. Override por INSTANCIA (`agente.requiere = (nuevo_hito,)`),
no toca la clase ni el archivo.

**Resultado, antes de aceptarlo:** las 20 permutaciones dieron exactamente el mismo
`n_despiertos` medio y la misma fracción de 23/23 que el brazo real, en las tres
amplitudes, sin ninguna variación — no "parecido", sino idéntico dígito a dígito. Eso es
sospechoso por sí solo, así que se verificó antes de reportarlo: se inspeccionó una
permutación concreta (id=5, `5_debil`→`hay_atomos` en vez de `T_bajo_electrodebil`) y se
confirmó que el *override* SÍ tiene efecto real — en esa corrida, `18_poda` y `15_causal`
(que en la asignación real esperan `hay_red` y despiertan tarde) despertaron en el **paso
0** porque quedaron gateados sobre un hito que ya era cierto desde el inicio a esa
amplitud. **La identidad y el orden de despertar cambian genuinamente con cada
permutación — lo que NO cambia es el conteo total.**

| amplitud | real (n_despiertos, 23/23) | las 20 permutaciones |
|---|---|---|
| 0,799 (bajo el borde) | 17,00 — 0/8 | idéntico en las 20 |
| 1,028 (en el borde) | 20,75 — 3/8 | idéntico en las 20 |
| 1,702 (sobre el borde) | 23,00 — 8/8 | idéntico en las 20 |

**Veredicto (pre-inscrito en v4§3): LA CASCADA ES INEVITABLE.** Con este observable
(`n_despiertos`), reasignar qué agente espera cuál hito no cambia CUÁNTOS terminan
despiertos — sólo cambia CUÁLES. Lectura honesta, tal como la instrucción pide sin
suavizar: **la tesis (que el despertar refleja la cadena causal específica del
inventario) no queda refutada, pero sí queda sin evidencia con este observable.** La
dinámica de qué hitos se cumplen (temperatura bajando, sobredensidad formándose) parece
gobernada por la física de fondo (`EstadoFisico` + los 6 agentes sin precondición, ninguno
tocado por la permutación) de forma robusta a cuál de los 17 agentes secundarios reciba
cada rol — al menos en las 3 amplitudes y el rango de parámetros probado.
`cs075_N2_puertas_permutadas.py` + `cs075_resultado_N2_puertas_permutadas.json`. Costo
real: 1.560 s (8 procesos, ~26 min).

### Qué decidís vos

N1 cierra en positivo (el borde de v3 se sostiene). N2 no refuta la tesis del despertar
ordenado, pero le quita el piso al observable que se venía usando para sostenerla —
"cuántos agentes despiertan" no distingue una cadena causal genuina de un reparto
arbitrario. La instrucción ya lo anticipaba (v4§3): habría que buscar OTRO observable
que sí distinga (candidatos no explorados acá: tiempo de residencia relativo, `X`/`S`
finales bajo cada permutación, o si la estructura ESPACIAL formada difiere aunque el
conteo no). No se avanzó sobre esto sin tu autorización.

**Archivos entregados:** `cs075_N1_sensibilidad_umbrales.py`,
`cs075_resultado_N1_sensibilidad_umbrales.json`, `cs075_N2_puertas_permutadas.py`,
`cs075_resultado_N2_puertas_permutadas.json`. No se tocó `cs075_base_fisica.py`,
`cs075_arquitectura_agentes.py` ni `cs075_23_sobre_fisica.py`.
