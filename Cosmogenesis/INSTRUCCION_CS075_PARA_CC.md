# INSTRUCCIÓN CS075 PARA CC — Los 23 sobre base física, con puerta de emergencia

**Encarga:** Alexis López Tapia (director) · **Diseña:** Claude Science · **Implementa y ejecuta:** CC
**Fecha:** 29-jul-2026 · **Estado:** listo para implementar. No requiere más diseño.

---

## 0. En una frase

Reescribir los 23 elementos del inventario canónico como 23 agentes autónomos sobre la base
física ya cerrada, donde **cada agente sólo actúa cuando las condiciones que su elemento
necesita para existir ya están en el campo** — y registrar el orden en que despiertan.

---

## 1. Lo que YA está hecho y verificado (no lo rehagas)

Cuatro archivos en `Cosmogenesis/`, corridos y con sus números impresos:

| archivo | qué es | estado |
|---|---|---|
| `cs075_base_fisica.py` | las 6 variables de estado físico | **CERRADO** — 6/6 direcciones verificadas |
| `cs075_arquitectura_agentes.py` | proceso común + interfaz de agente | **CERRADO** — 6/6 pruebas pasan |
| `cs075_pruebas_arquitectura.py` | la batería de arquitectura | **CERRADO** — corre en 18 s |
| `cs075_23_agentes.py` | los 23 con puerta de emergencia | **OBSOLETO** — es lo que hay que reescribir |

**Lo que quedó probado de la arquitectura** (`cs075_resultado_pruebas_arquitectura.json`):
cada agente aporta; permutar el orden de consulta cambia el resultado en 3,6e-16 (ruido de
punto flotante — no hay turnos); ningún agente lee fuera de su radio declarado; reproducible
bit a bit; estable sin NaN; la memoria plástica crece 3 órdenes y correlaciona 0,67 con el campo.

**Lo que quedó cerrado de la base física** (`cs075_resultado_base_fisica.json`): una sola ley
de expansión `a = √(1+k·t)` derivada del motor (`estado.py` l.44 + `p_expansion.py`), con
**una sola constante, k=50**. La velocidad de expansión arranca en 25c, decae como t^(−1/2) y
cruza c en t=12,48 — calculado analíticamente y observado en la corrida. Las seis direcciones:
T baja 191→24, ρ cae 4 órdenes, exergía cae 3, entropía sube 14,0→15,9, gradientes caen 8
órdenes, velocidad de expansión 4,8c→0,61c.

**REGLA DURA:** no toques `cs075_base_fisica.py` ni `cs075_arquitectura_agentes.py`. Se
importan. Si algo que esta instrucción pide no se puede hacer sin modificarlos, **pará y
reportá** — como hiciste correctamente en cs074D.

---

## 2. Por qué `cs075_23_agentes.py` hay que reescribirlo

Dos defectos de fondo, ambos ya diagnosticados:

1. **Está construido sobre un campo abstracto** Φ con reacción Φ(1−Φ²) y atractores en ±1.
   Eso es **topología**, y la rama topológica está CERRADA. El director fue explícito: se
   trabaja con parámetros físicos reales. La base correcta es `EstadoFisico` (densidad,
   temperatura, sus gradientes), no un campo con dos pozos.

2. **Puse masa (#7), catálogo (#6) y débil (#5) en el nivel 0**, sin precondiciones. Eso
   contradice el propio arco: la masa **no existe al inicio**, emerge en la ruptura
   electrodébil y su 99 % en el confinamiento QCD (`LINEA_TIEMPO_MASA_topologia_vs_fisica.md`,
   filas 4-5, donde ese error está registrado como "el error grande"). Lo repetí.

Además: en la primera versión definí una clase `A18_Espacio` tratando el elemento #18 como
"espacio/geometría". **Está mal.** Las tres apariciones de #18 en `cs072_motor_23.py` (l.45,
112, 167) lo definen como **poda/dilución acoplada a #9** (`PODA_FRAC=2.5`), y el manifiesto
(l.32-42, `G-ESPACIO-ES-CONSECUENCIA`) dice que el espacio NO es pieza del inventario. Ya está
corregido a `A18_Poda`, pero verificá que no reaparezca.

---

## 2 bis. La arquitectura de ANIMA que hay que respetar (corrección del director, 29-jul)

**Corrijo un error mío antes de que llegue al código.** En el diseño previo escribí que los dos
archivos que cita el protocolo CS075 "no existen con ese nombre". **Es falso, y en un caso
inexcusable:** `VST_Celula_Madre_001.py` existe con ese nombre exacto en
`VSTCosmo/Célula_Madre/campo/` y yo lo había leído — cité su clase `Hemisferio` (l.520) y le
copié las constantes. `Célula_Madre_Funcional_001.py` también existe; el director lo aportó.
De "no lo encuentro donde busqué" concluí "no existe". No repitas eso: si no encontrás un
archivo, decí dónde buscaste, no que no existe.

**Y lo importante para la implementación: ANIMA ya tiene el patrón estigmérgico, con un
contrato distinto del que yo supuse.**

- `Célula_Madre_Funcional_001.py` (656 líneas) organiza el sistema en **organelos** que
  depositan en un medio común llamado **`Milieu`**, importado de `VST_Genoma.py`.
- El contrato de depósito es `milieu.secretar(clave, valor)` — ver
  `OrganeloSoma.secretar()` (l.487 y siguientes), que vuelca ~25 señales nombradas:
  `omega_A`, `gradiente`, `Omega`, `e_R`, `A_sys_env`, `orientacion`, `delta_struct`,
  `INR`, `demanda_entorno`, entre otras, algunas con `guardar_historial=True`.
- Los consumidores (bloques 5/7/8, homeostasis) **leen del `Milieu`**, no del organelo que
  depositó. Nadie llama a nadie: ése es el desacople estigmérgico, ya implementado y probado.
- El audio entra como **forzamiento de borde** sobre 4 hemisferios Φ — el mismo rol que en
  CS075 tienen la expansión y el enfriamiento.

**Diferencia arquitectónica que hay que decidir, no improvisar:** el `Milieu` de ANIMA es un
diccionario de **señales nombradas escalares**; el campo de CS075 es un **campo espacial** de
densidad sobre malla 3D. No son la misma cosa. Los 23 agentes de CS075 depositan una
contribución *por celda*, no un escalar con nombre.

Las dos son compatibles y conviene usar **ambas**: el campo espacial ρ para los depósitos
físicos (lo que ya está en `cs075_arquitectura_agentes.py`), y un `Milieu` de señales
nombradas para los hitos y observables agregados (`T_bajo_confinamiento`, `hay_atomos`, etc.),
siguiendo el contrato `secretar(clave, valor)` de ANIMA en vez de inventar uno nuevo. **Si al
implementarlo ves que conviene otra cosa, es un desacuerdo: pará y reportá.**

---

## 3. LO QUE HAY QUE IMPLEMENTAR

### 3.1 Archivo nuevo: `cs075_23_sobre_fisica.py`

Los 23 elementos como agentes, cada uno con sus precondiciones **leídas del estado físico**.
Importa `EstadoFisico` de `cs075_base_fisica.py` y la interfaz de `cs075_arquitectura_agentes.py`.

Interfaz de cada agente — cuatro miembros de clase y tres métodos:

- `numero` — número canónico del inventario
- `nombre` — clave VERBATIM de `cs072_motor_23.py` (ej: `3_fuerte`)
- `requiere` — tupla de hitos que necesita (ej: `("T_bajo_confinamiento",)`)
- `es_casilla_falsacion` — booleano
- `condiciones_dadas(estado, hitos) -> bool`
- `deposito(estado, hitos)` — **PURA**: no muta nada, devuelve su contribución al campo de
  densidad. Si las condiciones no están dadas, devuelve **cero exacto**.
- `consolidar(estado, hitos)` — sólo los agentes con memoria propia (M2).

### 3.2 Los hitos: umbrales FÍSICOS, no inventados

Cada hito se lee del estado físico. **Esta es la parte que más cuidado necesita** — los
umbrales son mi traducción de la línea de tiempo a temperaturas del modelo, y si alguno no lo
podés anclar en un archivo del proyecto, **pará y reportá en vez de elegirlo vos.**

| hito | criterio | anclaje |
|---|---|---|
| `expansion_supraluminica` | `estado.es_supraluminico()` | `cs075_base_fisica.py`, ya implementado |
| `T_bajo_electrodebil` | T < T_EW | ruptura electrodébil, 159 GeV — fila 4 de la línea de tiempo |
| `T_bajo_confinamiento` | T < T_CONF | confinamiento QCD, ~155 MeV — fila 5 |
| `hay_sobredensidad` | contraste δ=ρ/⟨ρ⟩ supera umbral en alguna celda | condición para que la gravedad tenga a qué agarrarse |
| `hay_nucleos` | regiones ligadas persistentes por N pasos | #3 fuerte ya actuando |
| `hay_atomos` | regiones neutras persistentes | #4 EM ya actuando |
| `hay_red` | ≥2 entidades persistentes vecinas | precondición de poda (#18) y de tiempo (#24) |

**Escala de temperaturas:** T arranca en 1e3 adimensional. La razón física 159 GeV / 155 MeV
≈ 1026 fija la separación entre `T_bajo_electrodebil` y `T_bajo_confinamiento`; usá esa razón,
no dos números elegidos por separado.

### 3.3 Los 23, con sus precondiciones

**Nivel 0 — el universo primordial (sin precondiciones).** Son las condiciones mismas:
`#23 fluctuación de campo`, `#22 fluctuación QCD`, `#9 expansión`, `#10 enfriamiento`,
`M1 semilla`.

**Nivel 1 — requieren `T_bajo_electrodebil`:** `#5 débil` (nota: es la única que se APAGA al
enfriarse — `cs072_motor_23.py` l.147 usa `T_ef > T_EW`), `#7 masa` (acá emerge, no antes),
`#6 catálogo` (las especies existen cuando hay masa que las distinga), `#16 SSB` (casilla).

**Nivel 2 — requieren `T_bajo_confinamiento`:** `#3 fuerte`, `#8 aniquilación`,
`#1 espín` (casilla), `#11 tres cuerpos` (casilla), `#13 Pauli` (casilla).

**Nivel 3 — requieren `hay_sobredensidad`:** `#2 gravedad`, `#12 localidad`.

**Nivel 4 — requieren `hay_nucleos`:** `#4 EM`.

**Nivel 5 — requieren `hay_atomos`:** `#14 correlación`, `M2 memoria`, `#17 sector oscuro`.

**Nivel 6 — requieren `hay_red`:** `#18 poda`, `#15 causal` (casilla), `#24 tiempo` (lector puro).

Esto suma **22, no 23.** Falta uno y **no lo resolví**: trabajé con las 20 claves explícitas
de `cs072_motor_23.py` más #6, #18 y M3, y ahí hay una ambigüedad real. Antes de implementar,
verificá el inventario contra `MANIFIESTO_FOLD_CS072.md` (l.3 y l.22-31) y establecé cuál es
el que falta con su número y su nombre. **Si el inventario no cierra en 23 con nombres
verificables en un archivo del proyecto, pará y reportá en vez de inventar el que falte.**
Eso ya me pasó a mí en este mismo diseño con #18.

**Las 5 casillas de falsación** (#1, #11, #13, #15, #16) devuelven cero **por diseño**: su
nulo ya está registrado en el arco (`INFORME_CS_motor_23_piezas_construido.md`). No las midas
de nuevo, no las "arregles" para que aporten.

### 3.4 El registro (lo que se adjudica)

Por cada agente: `paso_despertar` (None si nunca), `pasos_dormido`, `pasos_despierto`.
Por cada paso: el estado físico completo + qué hitos se cumplen.
**El paso de despertar es SALIDA del experimento, nunca entrada.** Ningún agente se enciende
en un paso fijado de antemano.

---

## 4. PRUEBAS QUE DEBE PASAR (antes de cualquier corrida larga)

Archivo `cs075_pruebas_23_sobre_fisica.py`. Cada una imprime el número que la decide.

- **E1 — inventario completo:** hay exactamente 23 agentes, sin nombres repetidos, y cada
  `nombre` existe verbatim en `cs072_motor_23.py` (o está justificado por escrito si no).
- **E2 — nadie madruga:** ningún agente tiene `paso_despertar` anterior al primer paso en que
  su hito se cumple. **Cero violaciones.** Esta es la prueba de la tesis del director.
- **E3 — cero exacto dormido:** el depósito de un agente con condiciones no dadas es `0.0`
  exacto, no "pequeño". Verificalo con `np.all(dep == 0.0)`.
- **E4 — cero turnos:** permutar el orden de consulta cambia el resultado menos de 1e-12
  relativo. Ya pasó con 5 agentes; tiene que seguir pasando con 23.
- **E5 — las direcciones se mantienen:** las 6 direcciones de la base física siguen correctas
  con los 23 agentes depositando. Si un agente rompe la flecha termodinámica, es un bug.
- **E6 — orden esperado:** los niveles despiertan en orden creciente. Si el nivel 4 despierta
  antes que el nivel 2, la cadena causal está mal implementada.

**Si E2 o E3 fallan, PARÁ.** Son la tesis del experimento, no un detalle.

---

## 5. LA CORRIDA

### 5.1 Smoke primero — 4 configuraciones, minutos

Malla 16³, dt=1e-3, T_total suficiente para cruzar el confinamiento. Cuatro valores de
`amp_asimetria`: 0,01 / 0,1 / 0,5 / 2,0.

**Qué contesta:** ¿llegan a despertar los niveles altos? ¿Cuántos de los 23 quedan dormidos?

### 5.2 Reporte, y ahí PARÁS

No corras el barrido grande. Entregá:

- `cs075_resultado_23_sobre_fisica.json` con el registro completo
- una tabla del orden de despertar por configuración
- **cuántos agentes quedaron dormidos y por qué hito faltante**

**Es previsible que varios queden dormidos, y eso NO es un fallo.** En mi corrida sobre el
campo topológico, 6 de 23 nunca despertaron porque no llegaron a formarse átomos. Si el
espacio y el tiempo despertaran sin entidades persistentes, la arquitectura estaría mal. Un
agente dormido con su hito faltante identificado **es un resultado**.

---

## 6. LO QUE NO HAY QUE HACER

- **No hay NULL en esta etapa.** El director lo excluyó explícitamente: primero se prueba la
  arquitectura. El NULL viene después, y cuando venga será por **memoria** (congelada /
  barajada), no por pipeline secuencial — ese control mide el integrador, no la física
  (verificado: la divergencia entre simultáneo y por turnos no cae al reducir dt, y el 15,6 %
  de las celdas termina en atractor opuesto por amplificación no lineal).
- **No inventes constantes.** `p_expansion.py` dice textual: *"NO se inventa una ley nueva --
  se deriva del propio reloj de enfriamiento que el motor YA tiene... ninguna constante
  nueva"*. Yo violé esa regla dos veces en este mismo diseño (`H_post`, `fin_inflacion`) y
  las dos veces produjo física equivocada.
- **No enciendas agentes a mano** ni en un paso fijo. La puerta lee el estado.
- **No cierres el experimento.** Requiere autorización explícita del director.

---

## 7. Un aviso sobre mi propio diseño

Cometí cuatro errores en la base física y uno en el inventario, y **ninguno lo encontré
razonando** — todos aparecieron al imprimir números:

1. entropía con histograma de rango variable (el bin se encogía con la dilución)
2. confundí entropía de Shannon con entropía termodinámica (acá van en sentidos opuestos)
3. `H_post` constante, que es de Sitter — expansión acelerada para siempre
4. dos regímenes empalmados donde una sola ley alcanza (lo corrigió el director)
5. clasifiqué #18 como "espacio" cuando es poda/dilución

Los umbrales de hito del §3.2 y el inventario incompleto del §3.3 son las dos partes de este
diseño con menos verificación. Si algo no lo podés anclar en un archivo del proyecto, **es un
desacuerdo, y un desacuerdo es un dato**: pará y reportá.

---

*Nada se cierra sin autorización del director. Verificá en disco, no de palabra: antes de
escribir "verifiqué X", el valor de X tiene que estar impreso en la salida que estás mirando.*
