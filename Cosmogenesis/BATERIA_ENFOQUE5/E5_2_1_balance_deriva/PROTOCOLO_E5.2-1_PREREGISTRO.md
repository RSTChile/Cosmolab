# PROTOCOLO E5.2-1 — Balance de energía paso a paso: deriva del total sobre corridas largas

**Congelado (pre-registro):** 2026-07-24 (America/Santiago, UTC-4)
**Ejecutor:** CC (agente E5.2-1, batería Enfoque 5, corrida en paralelo con 29 agentes más)
**Base de código leída (NO editada):** `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs074_rcruz.py`
**Documento madre:** `BATERIA_ENFOQUE5_energia_exergia_entropia_30exp_PARA_CC.md`, sección "TEMA 2" (intro) + "E5.2-1"

Este documento se escribe y congela ANTES de correr el motor de barrido. Cualquier
desviación respecto de lo aquí escrito se reporta como desviación explícita, no se edita
retroactivamente (T3). **Este experimento es la base de la contabilidad que otros
experimentos del Tema 2-3 (y potencialmente E5.5-1/E5.5-4) van a asumir — la definición de
E_total de aquí abajo está pensada para ser reutilizable tal cual.**

---

## 1. Pregunta

¿El motor de campo+difusión+expansión de `cs074_rcruz.py` fabrica o pierde presupuesto de
energía a lo largo de corridas MUY largas (hasta 1e5 pasos), bajo el axioma declarado E1
(conservación de diseño, NO física real)? Se audita EN CADA PASO, no solo al final.

## 2. Modelo (heredado de cs074_rcruz.py, motor propio bajo mi prefijo — física idéntica, sin editar el original)

Campo escalar φ en un anillo de N sitios:
- Fondo φ=1 + perturbación ε·(suma de 5 armónicos con fase aleatoria, normalizada a
  desviación estándar 1) — `campo_inicial()`.
- **Difusión:** relajación local hacia el promedio de vecinos, SOLO por aristas vivas —
  fórmula idéntica a `paso_difusion()`: `nuevo = φ + 0.5·(media_vecinos_activos − φ)`.
- **Expansión:** cada arista viva se corta con probabilidad de Bernoulli H por paso —
  idéntica a `paso_expansion()`. H≥1 corta todas; H=0 no corta ninguna.
- **D** = fracción de contraste (desviación estándar) borrada en UN paso de difusión pura
  (H=0), MEDIDA del propio campo (no puesta a mano) — igual que `medir_D()`.
- **r** = H/D es la razón expansión/difusión. H se fija como H = min(r_target·D, 1.0); D se
  mide primero, H emerge de esa medida (mismo procedimiento que la base y que E5.1-1).
- **Motor propio:** reimplementación **vectorizada en lote** (batch sobre r×semilla) de las
  MISMAS fórmulas de `paso_difusion`/`paso_expansion` (no una física distinta) — necesaria
  porque este experimento exige medir en CADA paso de corridas de hasta 1e5 pasos × grilla
  completa. Verificada numéricamente idéntica a la función original de la base antes de
  usarse en el barrido (ver §9, verificación cruzada #1).

## 3. Axioma E1 — la definición EXACTA de E_total (contabilidad reutilizable)

**Declaración explícita (no es física real, es una elección de diseño nuestra):**
El campo φ no tiene una "energía física" intrínseca en este modelo. Definimos, como
elección de instrumentación, tres cantidades contabilizables a partir del estado φ(t) y
declaramos que su suma es el presupuesto total E1-conservado:

```
E_campo(t)  = N · mean(φ(t))²        [energía ligada al modo uniforme — "background",
                                        no puede hacer trabajo porque no tiene gradiente]
X(t)        = N · Var(φ(t))          [exergía — energía en la desviación del uniforme,
                                        la parte de la estructura que en principio podría
                                        hacer trabajo]
S_ent(t)    = X(0) − X(t)            [entropía/degradación acumulada — exergía perdida
                                        desde el inicio; bajo E2 la expansión no crea
                                        energía, solo aísla, así que S_ent solo debería
                                        crecer por difusión, nunca por el corte de aristas
                                        en sí mismo]

E_total(t) := E_campo(t) + X(t) + S_ent(t)
```

**Álgebra explícita (transparencia total, sin escamotear nada):** por construcción,
`X(t) + S_ent(t) = X(t) + (X(0) − X(t)) = X(0)` para todo t, así que

```
E_total(t) = E_campo(t) + X(0) = N·mean(φ(t))² + X(0)
```

Esto significa que el TEST REAL, no trivial, al que se reduce la conservación de E_total es:
**¿se mantiene constante N·mean(φ(t))² a lo largo de toda la corrida?** Equivalentemente,
¿se mantiene constante mean(φ(t)) (o Σφ(t))? Esta NO es una pregunta trivial: la fórmula de
difusión pondera cada nodo por su propio grado activo (`n_nb` = 0, 1 o 2 vecinos vivos), así
que cuando la expansión corta aristas de forma asimétrica, la relajación de un nodo de grado
1 (`nuevo = 0.5·φ + 0.5·vecino`) NO es un intercambio simétrico de "flujo" con ese vecino —
por eso Σφ NO está garantizado por construcción a permanecer constante una vez que H>0
empieza a cortar aristas. Verificamos esto de forma independiente ANTES de este
pre-registro con una corrida de sondeo (2000 pasos, N=200, no forma parte del barrido
oficial, solo decide la definición): con H=0 la deriva de mean(φ) es de orden máquina
(~1e-16, exacta); con H>0 la deriva relativa observada alcanza ~1e-3 a 2000 pasos para
r∈[1,10]. Esto es precisamente lo que E5.2-1 debe cuantificar de forma sistemática y
honesta: cuándo y cuánto se rompe la contabilidad.

**Observable pre-registrado:**
```
deriva(t) = |E_total(t) − E_total(0)| / |E_total(0)|
          = |N·mean(φ(t))² − N·mean(φ(0))²| / (N·mean(φ(0))²)     [si mean(φ(0)) ≠ 0]
```//
con **E_total(0) = N·mean(φ(0))² + X(0)** calculado explícitamente en el JSON de salida
(no solo la forma reducida) para que otros experimentos puedan reutilizar la cadena
completa (E_campo, X, S_ent) sin tener que rederivar el álgebra.

**Chequeo secundario de honestidad (no es el PASS/FAIL de este experimento, pero se
reporta):** ¿S_ent(t) ≥ 0 en todo momento? (i.e., ¿X(t) nunca excede X(0)?) — verifica si el
axioma E2 (la expansión no crea, solo redistribuye) se sostiene también a nivel de
varianza, no solo de la media. Si S_ent se vuelve negativo en algún paso, se reporta como
hallazgo aparte, sin ocultarlo ni corregir la definición a posteriori.

## 4. Axioma E2 (declarado, marco interpretativo, no verificado aquí directamente)

La expansión (cortar aristas) no crea energía; solo aísla regiones y así congela
gradientes que la difusión de otro modo borraría. Esto es el marco que motiva por qué
`S_ent(t) = X(0) − X(t)` debería ser monótona no decreciente. Se AUDITA como chequeo
secundario (§3), no se fuerza.

## 5. Barrido (sobredimensionado, regla del director)

| Eje | Rango | Puntos |
|---|---|---|
| pasos | corrida única de 1e5 pasos por celda (r,ε,semilla); se extraen checkpoints en pasos∈{1e2,1e3,1e4,1e5} de la MISMA trayectoria (ver §9, nota de eficiencia: truncar una trayectoria larga con el mismo rng es matemáticamente idéntico a una corrida independiente más corta con la misma semilla, porque el consumo del generador de números aleatorios en cada paso no depende de cuántos pasos falten) | 4 checkpoints, corridas MUY largas |
| ε | {0, 1e-12, 1e-9, 1e-6, 1e-3, 1e-2, 0.1, 1.0} | 8 (incluye 0 estricto y extremos de 12 décadas) |
| r = H/D | {0, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3} | 8 (incluye r=0 sin expansión + 6 décadas) |
| semillas | 0..11 | 12 (mínimo pre-registrado) |
| N | 200 (igual que modo "produccion" de la base) | fijo |

Total celdas (ε,r,semilla) = 8×8×12 = **768 corridas de evolución**, cada una llevada a
1e5 pasos con deriva auditada en CADA paso (no solo en los 4 checkpoints — el máximo de
deriva dentro de cada tramo [0,pasos_checkpoint] es lo que se reporta por checkpoint, y el
paso exacto del primer cruce del umbral se registra si ocurre).

**Umbral PASS (congelado, T6 estricto):** deriva(t) < 1e-6 en TODO paso t de la corrida
completa (1e5 pasos). Si en cualquier paso se cruza el umbral, esa corrida es **FALLO**, se
registra el paso exacto del primer cruce, y NO se promedia con las corridas que sí pasan
para esconderlo (regla del director, T6).

## 6. NULL

No aplica un NULL de permutación aquí (esto no es una prueba de estructura vs. azar, es una
auditoría de contabilidad interna). El "control" natural de este experimento es r=0 (H=0,
sin expansión, difusión pura simétrica en anillo completo): ahí se espera deriva ≈ 0 exacto
(orden máquina), porque la difusión con grado uniforme (n_nb=2 en todos los nodos) SÍ
preserva Σφ exactamente. Si r=0 no da deriva de orden máquina, es un fallo de la
implementación del motor propio (no del axioma), y se para y reporta antes de continuar
con el resto de la grilla.

## 7. Verificación cruzada (regla de ejecución #4)

1. **Motor propio vs. función original de la base:** antes de correr la grilla, se compara
   `paso_difusion_batch`/`paso_expansion_batch` (mi reimplementación vectorizada) contra
   `paso_difusion`/`paso_expansion` de `cs074_rcruz.py` (SIN editarla, solo importada) sobre
   las mismas trayectorias/semillas — deben coincidir a max|Δ|=0 (o precisión de máquina).
   Si no coinciden, se para y se reporta el error antes de usar el motor propio.
2. **Segundo método para el mismo E_total:** además de la fórmula reducida
   `E_total(t) = N·mean(φ(t))² + X(0)`, se calcula también la forma NO reducida
   `E_campo(t) + X(t) + S_ent(t)` sumando los tres términos por separado en cada paso
   (sin simplificar algebraicamente en el código) — deben coincidir entre sí a precisión de
   máquina; si no, hay un bug de instrumentación, se reporta.
3. **Auditoría en disco:** todas las filas crudas (deriva máxima, paso del primer cruce si
   existe, deriva final, S_ent mínimo observado) quedan en JSON para revisión externa por
   quien no escribió este motor.

## 8. Salidas

- `E5_2_1_engine.py` — motor (este archivo, escrito DESPUÉS de este pre-registro).
- `E5_2_1_resultado_crudo.json` — todas las filas del barrido: ε, r, H, D, semilla,
  E_total(0) [con desglose E_campo(0)/X(0)/S_ent(0)], deriva_max por checkpoint
  {1e2,1e3,1e4,1e5}, paso_primer_cruce (o null si nunca cruza), S_ent_min (chequeo de
  honestidad E2), PASS/FAIL por celda y checkpoint.
- `E5_2_1_verificacion_motor.json` — resultado de la verificación cruzada #1 (motor propio
  vs. base) y #2 (forma reducida vs. no reducida de E_total).

## 9. Trampas explícitamente evitadas

- T0: N, ε y r vienen del modelo base / calibración medida (D), no de ajustar a mano.
- T1: el umbral 1e-6 y los rangos de ε/r están fijados aquí ANTES de correr, no ajustados
  después de ver los datos.
- T2: el observable (deriva) es una fórmula fija derivada del álgebra de §3; el veredicto
  (PASS/FAIL por celda) lo da la curva completa de deriva vs. paso, no un solo número final.
- T3: si algo falla, se reporta tal cual — este archivo no se edita retroactivamente.
- T5: se reporta la curva de deriva entera por celda (no solo el máximo), y el paso exacto
  del primer cruce si lo hay.
- T6: **CADA paso se audita**, no solo el paso final — es el punto central de este
  experimento.
- T7: el barrido cruza r y ε en rangos de varias décadas cada uno, con ≥12 semillas
  independientes por celda (perturbación de la condición inicial vía la fase aleatoria de
  los armónicos + el propio proceso estocástico de corte de aristas, que es dinámico paso a
  paso, no solo una semilla cosmética).

No se corre nada del motor de barrido hasta que este archivo esté guardado en disco.
