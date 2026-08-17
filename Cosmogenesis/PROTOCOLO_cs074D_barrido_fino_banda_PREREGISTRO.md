# PROTOCOLO cs074-D — ¿La estructura vive en una banda estrecha no azarosa?

**Congelado (pre-registro):** 2026-07-26 · **Ejecutor:** CC · **Director:** Alexis López Tapia
**Diseño base:** `DISENO_barrido_fino_banda_estrecha_PARA_CC.md` (leído entero).
**Motor:** `cs074_energia_holistica.py`, función `correr_holistico_energia` — **NO SE
MODIFICA.** Todo lo que este experimento necesita (NULL vía `seed_dens_null`, observables
`frac_masa_ligada`, `n_clusters_finales`, `frac_masa_en_mayor_cluster`) ya existe en el
motor desde los Experimentos A/B — confirmado leyendo el código antes de escribir este
documento, no se asumió.

---

## 1. Pregunta

¿La estructura emerge en una **banda angosta y conexa** del espacio completo de
configuraciones (una condición real, no azarosa), en **casi todo el espacio** (trivial),
o en **puntos dispersos sin patrón** (ruido)? Los experimentos previos solo barrieron ε,
reserva y semilla — el resto quedó fijo. Este barrido mueve TODO junto.

## 2. Espacio de muestreo (Latin Hypercube, 6 dimensiones continuas)

| Variable | Símbolo | Rango | Escala |
|---|---|---|---|
| Asimetría inicial | `amp_rugosidad` | 1e-6 – 10 | log |
| Tasa de expansión | `tasa_expansion` | 0,001 – 0,2 | log |
| Reserva de energía | `E_reserva` | 0,001 – 1000 | log |
| Población de quarks | `nq` | 150 – 600 | lineal |
| Fracción antiquark | `naq/nq` | 0,5 – 0,95 | lineal |
| Fracción electrón | `ne/nq` | 0,15 – 0,6 | lineal |

`naq = round(nq · naq/nq)`, `ne = round(nq · ne/nq)`. `npos` (positrones) NO está en el
diseño como eje propio — se deriva `npos = round(0,7 · ne)`, la misma razón ne:npos=100:70
que usan cs074/A/B por defecto (no es un valor nuevo inventado, es el default ya
establecido, escalado con `ne` que sí se barre). `n_pasos_estructura=60` fijo (el diseño
lo deja como fijo para el barrido principal; la duda de si 60 pasos alcanza queda anotada
como pregunta abierta, no como eje de este barrido). Las 5 piezas (`cdm_on`, `cooling_on`,
`expansion_on`, `gravedad_on`, `energia_on`) quedan `True` — su apagado ya se probó aparte.

**N = 2000 configuraciones** (4000 si el smoke test de 20 configs proyecta tiempo
razonable) **× 12 semillas** (`seed_layout`) **× 2 brazos (REAL + NULL)**.

## 3. NULL

Por cada configuración × semilla: un brazo gemelo con `seed_dens_null` (barajar las
densidades bariónicas entre partículas, mecanismo YA existente en el motor, verificado en
Experimento B). `z = (frac_real_media − frac_null_media) / sd_conjunta` sobre las 12
semillas, por configuración.

## 4. Observables por configuración (promedio sobre 12 semillas)

`frac_masa_ligada`, `n_clusters_finales`, `frac_masa_en_mayor_cluster` — los tres ya
existen en el motor, no se agrega nada nuevo.

## 5. Método de banda vs dispersión (pre-inscrito, no a ojo)

Sobre las configuraciones con z>2 (en el espacio de 6 dimensiones, cada eje normalizado a
[0,1] -- log-normalizado en los ejes log): para cada punto z>2, sus k=10 vecinos más
cercanos (distancia euclídea en el espacio normalizado) entre TODAS las configuraciones;
`tasa_vecinos_hit` = fracción de esos vecinos que también son z>2. Promediar sobre todos
los puntos z>2 → `tasa_vecinos_hit_observada`.

**Control de azar de la conectividad:** barajar qué configuraciones son "z>2" entre las N
totales (mismo conteo de hits, posiciones re-asignadas al azar), recalcular
`tasa_vecinos_hit`, repetir 1000 veces → `z_conectividad = (obs − media_barajada) /
std_barajada`.

## 6. Lectura pre-inscrita (tres salidas posibles, cualquiera es válida)

1. **BANDA ESTRECHA:** hay configuraciones con z>2 Y `z_conectividad > 2` (los hits se
   agrupan más de lo que el azar explicaría) → condición real, no azarosa. Se describe la
   región (qué combinación de variables la define).
2. **SIN BANDA / GENÉRICO:** más del 50% de las configuraciones dan z>2 → estructura
   trivial, no hay condición fina.
3. **DISPERSO:** hay configuraciones con z>2 pero `z_conectividad ≤ 2` (no más agrupadas
   que el azar) → ruido, no hay banda real.

## 7. Trampas

- **T1:** ningún valor a mano — LHS cubre el espacio, nunca se centra en un resultado.
- **T-target:** 4,9%/31,5% (si se mide `fraccion_materia`) es lectura de salida, nunca
  entrada — si aparece en alguna banda, se reporta, no se busca.
- **No tocar el motor:** si algo que el diseño pide no está disponible tal cual, se para y
  se reporta a CS explícitamente — no se improvisa una solución dentro del motor.
- **Verificación en disco:** JSON crudo con las ~24.000–48.000 corridas completas (todas
  las configuraciones × semillas × REAL/NULL), no solo el resumen.
- **Smoke antes del full:** 20 configuraciones (240 corridas × 2 brazos = 480), para
  proyectar tiempo total antes de comprometer el cómputo completo.

## 8. Qué se entrega a CS, sin adjudicar

`cs074D_barrido_fino_banda.py`, el JSON crudo completo, el mapa de z por configuración, la
métrica de conectividad con su significancia, y cuál de las 3 lecturas del §6 dio. No se
cierra el hallazgo aquí.
