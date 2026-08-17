# AUDITORÍA EXHAUSTIVA DEL MOTOR CS072 — parámetros puestos a mano
**Fecha:** 20-jul-2026 · **Alcance:** motor modular vivo `cs072_modulos/` (20 archivos, 1496 líneas), el que produjo los resultados adjudicados.
**Método (triple verificación):** (1) cuatro auditores LLM independientes, adversariales, sin verse entre sí, cada uno leyó el motor completo; (2) lectura propia de CS; (3) **ejecución contra el motor real** — la autoridad final. Nada de lo que sigue se afirma por lectura de código o por comentarios (los comentarios ya mintieron una vez); todo lo marcado VERIFICADO se corrió.

## VEREDICTO GLOBAL
El motor NO deriva la materia del Modelo Estándar de la física de sus fuerzas. Los observables "estrella" son en su mayoría **estequiometría del catálogo o parámetros de entrada copiados a la salida**, indistinguibles de su NULL. Tres fuerzas de las cinco no tienen ningún efecto sobre la materia.

## HALLAZGOS VERIFICADOS CONTRA EL MOTOR REAL

### 1. El conteo de bariones NO depende de ninguna fuerza — pura estequiometría [GRAVE, nuevo]
- `bariones = nq/3` EXACTO (30→10, 60→20, 300→100, 900→300). Es el nº de quarks del catálogo dividido en 3.
- **Apagar la fuerza fuerte deja bariones=10** (base=10). Apagar aniquilación, gravedad, fluctuaciones: sin efecto. 
- **El test de admisibilidad "apagar confinamiento → 0 bariones", celebrado como la prueba de que las fuerzas hacen la materia, es FALSO en este motor.** Los bariones son un recuento de población, no un producto del confinamiento.
- La máscara `ligado` que sí sale de la matriz de fuerza fuerte `Bq` (`nucleo.py:34`) **se calcula y nunca se usa** (código muerto). El conteo real (`_detecta_trios`) usa estequiometría de poblaciones, no `Bq`.

### 2. El ratio p:n = 7:1 está puesto a mano [GRAVE, ya documentado, sigue en el código]
- `freeze_out.py:20`: `h = tasa_expansion*20.0`. El 20.0 hace que la tasa por defecto (0.02) caiga en h=0.4 → ratio 7.1. Lo puso CS. Sin justificación en código, git ni Memanto.
- **El motor real da p:n = 1:1 EXACTO** a toda tasa y toda escala (paridad del catálogo). El 7:1 es una fórmula analítica desacoplada, no lee el motor.

### 3. La "dimensión" copia el input D — igual a su NULL [GRAVE, nuevo]
> *Procedencia: re-ejecutado en esta auditoría. El gate documentado (300,210,100,70) con D=3 da dim_efectiva=2.05 EXACTO (el número del docstring), y la dim sigue a D: D=2→1.83, D=3→2.05, D=4→2.82 (input copiado al output). La comparación REAL 2.77 = NULL 2.80 se estableció en trabajo previo de esta sesión.*
- dim de salida sigue a D de entrada: D=1→1.0, D=3→2.77, D=5→3.41.
- REAL (campo del motor, D=3) = 2.77 ; NULL (azar uniforme puro) = 2.80. **Idénticos.** La física no mueve la aguja; es un artefacto geométrico de la malla kNN en D dimensiones.
- `p02b_gravedad_general.py`: el docstring documenta el gate (300,210,100,70) "→ dim_efectiva=2.05" — target escrito.

### 4. El factor 50.0 del enfriamiento — alineado al umbral [SOSPECHOSO→acotado]
- `estado.py:44`: `T = T0/√(1+tasa*50·step)`. Con defaults (T0=3, tasa=0.02, pasos=400): T_final = 3/√400 = 0.15 = T_REC EXACTO. Cuatro constantes coincidiendo en el umbral de recombinación.
- VERIFICADO: los conteos (bar/H/He) NO cambian con pasos=100/400/800 → el resultado es robusto al paso, así que el 50.0 no fija los conteos finales. Queda como constante sin derivar (perilla latente) pero no está fabricando los números. Menos grave que 1-3.

## LO QUE SÍ TIENE DEPENDENCIA FÍSICA REAL (verificado)
- **Apagar EM → H=0** (base H=5). El hidrógeno sí requiere la recombinación EM. Dependencia genuina.
- **Apagar fuerte → He=0** (base He=2). El helio sí requiere la fuerte residual. Dependencia genuina.
- Estas dos son las únicas admisibilidades que el motor real cumple.

## LO QUE NO TIENE EFECTO (código decorativo)
- Apagar **aniquilación**, **gravedad** o **fluctuaciones (#23)**: sin efecto sobre bariones/H/He. La afirmación "apagar aniquilación → 10 en vez de 3 (asimetría bariónica)" también es falsa en este motor.

## RESUMEN POR OBSERVABLE
| Observable | ¿Emerge de la física? | Evidencia (motor real) |
|---|---|---|
| bariones=100 | **NO** — estequiometría nq/3 | invariante a apagar toda fuerza |
| p:n = 7:1 | **NO** — fórmula con 20.0 a mano | motor da 1:1 exacto |
| dimensión ≈ 3 | **NO** — copia D, = su NULL | REAL 2.77 = NULL 2.80 |
| asimetría bariónica | **NO** — sin efecto | apagar aniquilación no cambia nada |
| H (hidrógeno) | **SÍ** — requiere EM | apagar EM → H=0 |
| He (helio) | **SÍ** — requiere fuerte | apagar fuerte → He=0 |

## CONVERGENCIA DE AUDITORES
Los cuatro auditores independientes marcaron, sin verse: el 20.0 (4/4), el 50.0 (2/4), el gate de dimensión con su target "2.05" en el docstring (1/4, solo auditor B), y el conteo de bariones desacoplado de la fuerza (1/4, el más agudo). La ejecución en esta auditoría confirmó los cuatro hallazgos corriendo el motor: 1, 2 y 4 con los apagados de fuerzas y el barrido de pasos; el 3 corriendo el gate documentado (300,210,100,70)→dim 2.05 exacto y la respuesta dim→D. La comparación REAL 2.77 = NULL 2.80 del hallazgo 3 proviene de trabajo previo de esta sesión.

## ALCANCE — qué cae y qué queda
- **Cae:** la afirmación de que el motor deriva la materia del Modelo Estándar (bariones, p:n, asimetría) y la dimensión de forma emergente/parameter-free. Son estequiometría, perillas o inputs copiados.
- **Queda en pie:** las dependencias H↔EM y He↔fuerte (reales); y —fuera de este motor— los nodos NEGATIVOS del arco del espacio (π contingente, la dirección no emerge, la geometría no es a priori), que un parámetro a mano no puede fabricar y que se auditaron contra NULL en su momento.

— Auditoría CS, 20-jul-2026. Verificada contra el motor real, no contra adjudicaciones previas.
