# PROTOCOLO cs074-C — ¿Da el modelo relación y proceso, pero NO los números físicos?

**Congelado (pre-registro):** 2026-07-25 · **Ejecutor:** CC · **Director:** Alexis López Tapia
**Diseño base:** `DISENO_tres_experimentos_holistico_PARA_CC.md` (Experimento C, leído entero).
**Depende de:** cs074 original (280 corridas), cs074-A y cs074-B (una vez terminados) —
este experimento NO corre física nueva, es un ANÁLISIS de los barridos ya hechos.

Este documento se congela ANTES de escribir el script de análisis. Las columnas del cuadro
final (qué números físicos se evalúan, qué relaciones se evalúan) se fijan aquí, ANTES de
mirar si algún número "casi" coincide — eso es lo que evita que este experimento se
convierta en una búsqueda de coincidencias post-hoc (T3).

---

## 1. Pregunta

¿Es cierto que el modelo produce comportamientos/relaciones que persisten bajo control,
pero NUNCA los números concretos del universo real? Convertir la sospecha en una prueba
falsable: para cada número físico que el proyecto intentó reproducir alguna vez, medir si
está más cerca de su valor real de lo que estaría por puro azar del propio barrido.

## 2. Números físicos a evaluar (fijados AHORA, antes de mirar resultados)

| Número | Valor real | De dónde sale en el modelo |
|---|---|---|
| Fracción de materia | 4,9% (bariónica) / 31,5% (total, con oscura) | `frac_masa_ligada` (cs074, cs074-A) |
| Razón protón:neutrón congelado | ~7,1 | `ratio_pn_congelado` (motor basal, `freeze_out.py`, ya reportado en corridas anteriores CS072) |
| Razón masa protón/electrón | ~1836 | `masa_trio` (bariones) / masa de electrón del catálogo — a computar desde `_extraer_bariones`, NUNCA calculado antes en cs074/A/B, se agrega aquí por primera vez con el mismo criterio (test de salida) |

No se agregan números nuevos después de ver los datos (T3) — si aparece la tentación de
sumar un cuarto número "porque casi dio", se declara aparte, explícitamente marcado como
post-hoc, nunca mezclado con esta tabla.

## 3. Método — distancia contra el azar del propio barrido

Para cada número físico:
1. Tomar la distribución completa de valores que el modelo produjo para ese número, sobre
   TODO el barrido disponible (cs074 280 + cs074-A 1920 + cs074-B 3960, según aplique al
   número).
2. Calcular la distancia mínima observada entre el valor real y cualquier punto del
   barrido: `d_real = min_i |valor_modelo_i − valor_real|`.
3. **Control de azar:** re-muestrear la MISMA distribución de valores del modelo, pero
   barajando qué combinación (ε, E_reserva, semilla, ...) le tocó a cada valor —
   equivalentemente, dado que ya es una distribución empírica de valores, el control de
   azar es: elegir un punto AL AZAR de esa misma distribución y medir su distancia al valor
   real, repetido 1000 veces → distribución de "distancia por azar puro".
4. **z de cercanía:** `z = (media(distancia_azar) − d_real) / std(distancia_azar)`. Si
   z>2, el modelo se acerca al valor real más de lo esperable por azar — señal real. Si
   z≤2, la cercanía observada es indistinguible de tropezar con ese valor por casualidad
   dentro del propio barrido.

## 4. En paralelo — lo que el modelo SÍ da (relaciones/procesos, contra su propio NULL)

Se listan (fijado AHORA) las relaciones ya encontradas con su control ya corrido:
- Contabilidad de energía que cierra exacto (cs074, control gravedad pura, 1.7% de fuga).
- Costo de ligadura con efecto causal real (cs074, prueba de admisibilidad, 280 puntos).
- Muerte térmica ≠ Nada: retiene el 100% del presupuesto de energía (E5.5-4, Enfoque 5).
- El techo no-monótono en ε, SI cs074-A confirma que persiste con control (lectura
  "mecánica" o "mixta" del protocolo cs074-A §5) — condicional a ese resultado.
- La fragmentación responde al enfriamiento con separación del NULL, SI cs074-B da
  PASS_cs074B=True — condicional a ese resultado.
- El rescate de exergía por expansión (Enfoque 5, Regla 3 de cs074, y el control
  `expansion_on=False` de cs074 que ya mostró 88,4% vs 60,7% con expansión).

Cada una se reporta con su z-score o su criterio PASS ya obtenido — no se recalculan aquí,
se COMPILAN.

## 5. PASS pre-registrado

Un cuadro de dos columnas:
**Columna "SÍ" (relaciones/procesos con control que muerde, listados en §4, cada uno con
su significancia)** vs **Columna "NO" (números físicos de §2 con z≤2, es decir,
indistinguibles del azar del barrido)**.

Lectura del cuadro completo (no de un ítem aislado):
- Si TODOS los números de §2 caen en z≤2 Y al menos una relación de §4 tiene control real
  → el límite propuesto ("relación sí, números no") queda CONFIRMADO, medido, no asumido.
- Si ALGÚN número de §2 da z>2 de forma robusta (no una sola celda aislada, sino una
  región del barrido, varias semillas) → esa es la excepción a perseguir, se reporta
  explícita, no se descarta.

## 6. Trampas

- **T1/T3:** los números de §2 y las relaciones de §4 se fijan en ESTE documento, antes de
  correr el análisis — no se agregan ni se quitan después de ver el resultado.
- **La cantidad medida ≠ su juez:** la distancia al valor real (§3) se mide sobre datos YA
  producidos por barridos independientes (cs074/A/B), nunca se ajusta el barrido para que
  la distancia salga chica.
- **T-target reforzada:** este es el experimento que estructuralmente MÁS cerca está de
  tentar a "buscar hasta encontrar" — por eso el control de azar (§3.3-4) es obligatorio
  para CUALQUIER número que se reporte como "cerca", no opcional.

## 7. Qué se entrega a CS, sin adjudicar

El cuadro de dos columnas completo, con z-scores para cada número físico y cada relación,
y la lectura de §5. No se cierra el límite del modelo aquí — CS lee, el director decide.
