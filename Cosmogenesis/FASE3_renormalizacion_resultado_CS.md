# FASE 3 — Flujo de geometría bajo renormalización (arco CS064-CS068, línea DISTINTA de CS073/Phantom)

**Fecha:** 8-ago-2026. **Codea/ejecuta:** CC (Claude). **Diseño:** roadmap consolidado de 5 analistas de
IA (5-ago-2026), adaptado a las funciones ya existentes del proyecto. **Scripts nuevos:**
`cs080_renormalizacion.py` (Experimento 1, corrido completo). Experimento 2 quedó **diseñado en
detalle, NO corrido** (ver §4) — se priorizó tener el Experimento 1 completo y bien hecho, como pidió
Alexis, en vez de forzar los dos a medias.

No se toca ningún script `cs064`-`cs068` (motores ya verificados) — todo es import. No se declara
cierre ni veredicto de arco: acá van los números: **la lectura final es de Alexis.**

---

## 0. Analogía de arranque

Imaginá que tenés un mapa de calles hecho de miles de casas individuales, y sospechás que, aunque
cada esquina parece un enredo sin orden, si te alejás lo suficiente —como cuando volás en avión y las
casas se funden en manzanas, las manzanas en barrios, los barrios en la mancha urbana— tal vez aparezca
una ciudad con avenidas reales que sí van "lejos". La pregunta de este experimento es exactamente esa:
¿alejarse (agrupar nodos en supernodos, una y otra vez) revela una geometría que no se veía casa por
casa? O el enredo sigue siendo un enredo a cualquier altura de vuelo, sin importar cuánto te alejes.

---

## 1. Qué se investigó antes de escribir código (paso de lectura obligatorio)

Se leyeron `cs064_sistema_completo.py`, `cs066_localidad_geometrogenesis.py`, `cs067_gamma_sweep.py`,
`cs068_paso2_mundo_ab.py` y `cs068_paso2b_diametro.py`, más `cs066conf_exponentes.md` (tabla de
exponentes confirmatoria de CS066, no mencionada explícitamente en la tarea pero encontrada durante la
lectura — resultó ser la pieza clave para elegir el sustrato) y la sección "FASE 9" de
`AUDITORIA_COMPLETA_COSMOGENESIS_2026.md`.

**Funciones ya existentes que se reusaron TAL CUAL** (ninguna se reescribió):
- `dim_volumen(adj,N,rng)` (`cs064_smoke.py`) — dimensión por crecimiento de bola `|B(v,r)|~r^d`. Es
  exactamente la "D_H tipo Hausdorff" que pedía el roadmap; **ya estaba hecha**, no hubo que inventarla.
- `_diam(adj,N)` y `_giant(adj,N)` (`cs055_proceso_acoplado.py`, reexportadas vía `cs057`) — diámetro
  (doble BFS) y fracción de componente gigante.
- `_frame_burgers(...)` (`cs059_espin_como_marco.py`) — holonomía del marco de espín sobre ciclos
  fundamentales.
- `proceso066(...)` y `gate_localidad(...)` (`cs066_localidad_geometrogenesis.py`) — EL MOTOR que
  construye el sustrato, usado sin tocar una línea, incluyendo su brazo NULL ya diseñado
  (`local_barajado`).

**π_G: búsqueda explícita, sin resultado.** Se corrió `grep -r "pi_G\|π_G\|PI_G"` sobre todo el
proyecto antes de escribir nada — cero coincidencias. La "razón π" que menciona la auditoría (2.0 en
cuadrada, 2.99 en triangular, 1.5 en hexagonal) se calculó en experimentos anteriores no re-expuestos
como función reusable con ese nombre. **No se fabricó un juez nuevo bajo la etiqueta π_G** para no
inventar un número sin linaje — se documenta como AUSENTE en este experimento. `dim_volumen` cumple el
rol de "razón geométrica" que pedía el roadmap y es la que se usó.

---

## 2. Sustrato elegido: el tejido de CS066, en su punto MÁS favorable a la localidad

De la tabla confirmatoria `cs066conf_exponentes.md` (40 parches/celda, ya corrida por el proyecto antes
de hoy): con `k_local=6` el tejido es el mejor candidato disponible a "localidad real" — dimensión
espectral `d_s` estable entre 3.5 y 3.9 (cerca de 3D), componente gigante sana (91%), pendiente de
diámetro monótona con R²=0.93. Es el mismo punto que el propio archivo de síntesis del proyecto señala
como el régimen sano (`k5-6`).

**Construcción (b=1, antes de cualquier agrupamiento):** motor completo de CS066 (`proceso066`), brazo
`local`, `k_local=6` FIJO, N=8000, 20 pasos de física (gravedad+confinamiento+EM+débil+despliegue+
aniquilación+gate de localidad+co-evolución nemática del marco), 3 semillas independientes.

**Dos controles NULL, ambos corridos con el MISMO procedimiento de coarse-graining:**
1. **`local_barajado`** (brazo ya existente en CS066): mismo tope de grado `k_local`, pero eligiendo
   qué enlaces persisten AL AZAR en vez de por soporte local — aísla si importa el CRITERIO de
   localidad o sólo acotar el grado.
2. **`er_null`** (nuevo, piso absoluto): Erdős-Rényi puro, mismo N y mismo grado medio, sin ninguna
   física ni localidad — vía `cg003_diagnostico_gromov.aleatorio`, ya usado como generador de sopa
   caliente en CS064/066.

---

## 3. Experimento 1 — resultado

### Método de coarse-graining (nuevo, declarado, sin calibración oculta)
"Cajas" por BFS greedy (variante práctica del box-covering de Song-Havlin-Makse): se recorren los
nodos en orden aleatorio; cada nodo sin asignar dispara una caja que crece por BFS hasta juntar ~b
nodos; dos supernodos quedan unidos si existe al menos un enlace real entre sus miembros. Escalas
barridas: **b=2,4,8,16,32** (5 escalas, además de b=1 sin agrupar — más del mínimo de 3-4 pedido).
3 semillas por brazo → 90 mediciones en total.

### Tabla de resultados (media de 3 semillas)

| b | N_b local | diam local | giant local | d_s local | N_b baraj. | diam baraj. | giant baraj. | d_s baraj. | N_b ER | diam ER | giant ER | d_s ER |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1  | 8000 | 9.00 | 0.907 | **4.10** | 8000 | 7.00 | 0.912 | 3.43 | 8000 | 9.33 | 0.997 | 3.72 |
| 2  | 4665 | 7.33 | 0.851 | 4.12 | 4594 | 5.33 | 0.855 | 4.15 | 4469 | 7.00 | 0.996 | 3.96 |
| 4  | 2947 | 6.67 | 0.767 | 4.21 | 2819 | 4.00 | 0.767 | 4.13 | 2632 | 6.00 | 0.992 | 4.14 |
| 8  | 1971 | 5.33 | 0.653 | 3.90 | 1900 | 3.33 | 0.655 | 4.33 | 1668 | 5.00 | 0.988 | 4.43 |
| 16 | 1462 | 5.00 | 0.532 | **0.99** | 1411 | 3.67 | 0.536 | **0.67** | 1204 | 4.00 | 0.983 | 3.92 |
| 32 | 1216 | 4.33 | 0.437 | **0.83** | 1163 | 3.00 | 0.437 | **0.56** | 948  | 4.00 | 0.979 | 3.77 |

Pendiente log-log de `diam(b)` vs `N_b(b)` (el "exponente de dimensión bajo RG", análogo al que usó
CS068 en Paso 2b, umbral pre-inscrito de referencia 0.3), media de 3 semillas:

| brazo | pendiente por semilla | media |
|---|---|---|
| local (real) | 0.289, 0.346, 0.494 | **0.376** |
| local_barajado (NULL 1) | 0.401, 0.393, 0.466 | **0.420** |
| er_null (NULL 2, piso) | 0.435, 0.387, 0.394 | **0.406** |

### Lectura (números, sin declarar cierre)

**Dos hallazgos, ambos negativos para "condensación macroscópica" (Resultado A), y ambos consistentes
entre sí — pero por razones distintas y con una sorpresa metodológica:**

1. **La pendiente diam-vs-N_b NO distingue real de NULL.** Las tres pendientes (0.376, 0.420, 0.406)
   se solapan por completo dentro del rango semilla-a-semilla — el tejido "local" no muestra una
   escalada de diámetro más fuerte que su propio barajado NI que un grafo sin ninguna estructura
   (Erdős-Rényi). **Sorpresa metodológica honesta:** las TRES pendientes superan el umbral 0.3 que
   CS068 usó para declarar "Mundo A" — incluido el piso Erdős-Rényi, que por diseño no tiene ninguna
   geometría. Eso significa que ese umbral, calibrado contra un protocolo distinto (variar N generando
   grafos nuevos de tamaños distintos), **no es el juez correcto bajo ESTE protocolo** (variar N_b
   agrupando un mismo grafo): agrupar en cajas reduce el diámetro mecánicamente incluso en un grafo sin
   estructura, así que una pendiente positiva aparece "gratis" con el coarse-graining mismo. El juez
   útil aquí no es el valor absoluto de la pendiente contra un umbral fijo, sino la COMPARACIÓN directa
   real-vs-NULL — y ahí no hay separación.

2. **El tejido con estructura (local y local_barajado) se FRAGMENTA y su dimensión por bola COLAPSA a
   escalas grandes** — de d_s≈4.1 en b=1 a d_s≈0.8-1.0 en b=16-32 (prácticamente 1D, señal de que lo
   que sobrevive del agrupamiento son cadenas delgadas o fragmentos, no una geometría sana) — mientras
   que la componente gigante cae de 91% a 44%. El Erdős-Rényi, en cambio, se mantiene casi totalmente
   conexo (98-99.7%) y su d_s permanece estable entre 3.7 y 4.4 en TODAS las escalas. Es decir: la
   estructura local de CS066, lejos de "condensar" en algo macroscópico al agruparla, se **desarma**
   más rápido que el ruido puro. Esto es consistente con, y refuerza por una vía nueva e independiente,
   el hallazgo ya registrado del arco (CS066-068): la localidad fuerte crea tejido LOCAL sano pero no
   genera "lejos" que sobreviva a mirar más grueso.

**Conclusión de números (no de veredicto):** el Experimento 1 da **Resultado B reforzado** — no
apareció ninguna escala de agrupamiento (b=2 a 32) en la que el tejido real se separe de sus propios
controles NULL en la dirección de "más geometría". Si acaso, el tejido con localidad se comporta PEOR
que el ruido bajo agrupamiento (colapsa antes). La holonomía (heredada por promedio de espines a cada
supernodo, medida con la misma `_frame_burgers` sin modificar) se mantiene en un rango angosto
(0.9-1.3 rad) en los tres brazos y en todas las escalas, sin tendencia clara — no aporta señal
adicional en ninguna dirección (nota honesta: esta holonomía mide consistencia del marco NEMÁTICO,
pregunta de "direcciones" / Nivel 2 del arco CS064-066, no curvatura geométrica del grafo en sí mismo).

**Caveat metodológico para que quede escrito:** `_diam` y `_giant` (reusadas sin modificar) miden la
componente que contiene el primer nodo no-vacío, no necesariamente la componente gigante exacta cuando
la fracción gigante cae por debajo de ~50% (b≥16) — puede introducir ruido adicional en esas dos
escalas extremas. El patrón de colapso de `d_s`, sin embargo, es consistente y monótono ya desde b=8,
antes de que ese caveat sea crítico.

---

## 4. Experimento 2 — protocolo diseñado en detalle (NO corrido, por presupuesto de tiempo)

**Decisión explícita:** se priorizó, como pidió Alexis, tener el Experimento 1 completo y bien hecho
antes que forzar el Experimento 2 a medias. Acá va el protocolo completo, pre-registrado, listo para
correr en una sesión futura (`cs081_poda_dinamica.py`, aún no escrito).

### Pregunta
¿El sistema mismo, sin que nadie le diga "corta los atajos", puede DESCUBRIR un "lejos" macroscópico si
cada enlace paga un costo por lo que realmente aporta — en vez de que un criterio externo arbitrario
(ya probado y fallido en CS068 Paso 1) decida qué es un "atajo"?

### Costo por enlace, 4 componentes (todas reusan piezas ya existentes del proyecto)
Para cada arista `(i,j)` del tejido de CS066 (mismo motor, mismo `k_local=6`, mismas semillas que el
Experimento 1, para comparar manzana con manzana):

1. **Inconsistencia histórica** — cuántas veces el enlace aparece/desaparece a lo largo de los pasos
   de `proceso066` (requiere instrumentar el motor para registrar el historial arista-por-arista; hoy
   sólo se registra `D`/`G` agregados por paso — instrumentación NUEVA, liviana, sin tocar
   `cs066_localidad_geometrogenesis.py` original, por ejemplo con un wrapper que copia `adj` cada
   pocos pasos y compara).
2. **Conflicto de holonomía** — usando `C9._ciclos_fundamentales` (ya existe) y
   `C9._holonomia_ciclo` (ya existe): para cada ciclo fundamental que pasa por `(i,j)`, cuánto se aleja
   el transporte paralelo de los espines de la identidad; el costo del enlace es el promedio de esa
   holonomía sobre los ciclos que lo usan.
3. **Baja contribución a persistencia** — soporte local (nº de vecinos comunes), YA calculado dentro de
   `gate_localidad` (reuso directo del mismo cálculo, sin reimplementar).
4. **Baja reciprocidad** — un enlace es "recíproco" si AMBOS extremos lo eligen entre sus `k_local`
   más locales dentro de `gate_localidad`; "unilateral" si sólo uno de los dos lo elige (hoy
   `gate_localidad` usa OR — "sobrevive si algún extremo lo conserva" — la reciprocidad NO se está
   registrando, es información que ya se calcula y se tira; instrumentarla es barato).

**Costo total:** z-score de cada componente (calculado sobre TODAS las aristas del tejido en ese
momento) sumados con pesos iguales (1/4 cada uno) — pre-registrado ANTES de correr, no ajustado
después de ver el resultado. Se podan las aristas por encima de un percentil P del costo total
(P a barrer: 50, 70, 90 — no un solo valor fijo, para no repetir el error ya cazado en CS068 Paso 1 de
"un criterio externo arbitrario").

### Control NULL (aísla si importa QUÉ se corta, no CUÁNTO)
Mismo número de aristas podadas, elegidas AL AZAR (no por costo) — mismo P, mismo momento del proceso.
Si el sistema "descubre" un lejos real por podar SEGÚN costo, el resultado debe separarse del control
que poda la MISMA cantidad sin usar el costo.

### Métrica de veredicto
Igual que el Experimento 1: pendiente diam-vs-N_b bajo el mismo coarse-graining de `cs080`, comparando
poda-por-costo vs poda-aleatoria-misma-cantidad vs sin-poda. **Aprendizaje directo del Experimento 1:**
no comparar contra el umbral fijo 0.3 de CS068 (no discrimina bajo coarse-graining, ver §3) — comparar
DIRECTAMENTE las tres pendientes entre sí, con las mismas semillas.

### Falsación pre-registrada
Si poda-por-costo ≈ poda-aleatoria (misma pendiente dentro del rango de semillas) → negativo: el costo
elegido no descubre nada que azar-con-la-misma-densidad no logre igual — resultado honesto, no fracaso.
Si poda-por-costo > poda-aleatoria de forma consistente en las 3+ semillas → hay algo que el costo
captura que el azar no — vale la pena escalar a N grande.

---

## 5. Resumen para Alexis (en simple)

Agarramos el mejor tejido que el proyecto ya tenía armado (CS066, `k_local=6` — el punto donde el
propio archivo de síntesis dice "acá el tejido es lo más sano que se vio") y lo miramos con lentes cada
vez más desenfocados (agrupando 2, 4, 8, 16, 32 casas en una manzana). Ni el tejido real ni sus dos
controles (mismo tejido pero mezclado al azar, y un grafo totalmente sin estructura) mostraron, a
ninguna escala de desenfoque, una ciudad con avenidas reales escondida debajo. Peor: el tejido con
estructura se deshace MÁS RÁPIDO que el ruido puro al agruparlo — pasa de verse "3D sano" a verse casi
"1D, en hilachas" apenas se agrupa 8-16 veces, mientras que el grafo sin ninguna estructura se mantiene
estable en todas las escalas. El Experimento 2 (dejar que el propio sistema decida qué enlaces cortar,
en vez de imponerlo desde afuera) quedó diseñado con todo detalle pero no corrido — es la línea
siguiente natural si se decide seguir en este arco.
