# RESULTADO PASO 1 — cálculo del N mínimo para la ignición (INSTRUCCION_CC_escala_ignicion_PARA_CC.md)
**Corrido en esta sesión (Claude Code, Mac de Alexis), 31-jul-2026.** Responde el PASO 1 pendiente desde
el 20-jul: calcular el N mínimo que resuelve el colapso, de nuestros datos, no a ojo.

## En simple, antes de los números
Para que una simulación de gas colapsando "cuente" como confiable (no como un artefacto numérico), la
región que colapsa tiene que contener bastantes partículas — si un "grumo" que se declara estrella son
apenas 5 o 10 átomos, no sabemos si colapsó por física real o porque la simulación es demasiado gruesa
para verlo bien. La regla estándar del campo (Bate & Burkert 1997) dice: **la región que colapsa necesita
al menos el doble del "vecindario" que usa el motor de física** para que el resultado sea confiable.

## 1. El número de vecinos de Phantom (confirmado en la fuente primaria)
La instrucción de julio pedía confirmar esto antes de citarlo: **Price et al. 2018** (el paper del propio
Phantom), sección 2.1.3, dice textual: *"The default setting for hfact is 1.2, corresponding to an
average of 57.9 neighbours..."* — o sea, con la configuración que ya usa nuestra traducción a Phantom
(`fase1_traducir_a_phantom.py`, `HFACT = 1.2`), cada partícula "ve" en promedio **57.9 vecinas**.
- Criterio Bate & Burkert (1997), confirmado por búsqueda: la masa mínima resoluble = **2 × ese número
  de vecinos** de partículas. Con N_vecinos=57.9 → **hacen falta ≥ 115.8 (~116) átomos** dentro de UNA
  región que colapsa para que el colapso sea confiable, no ruido numérico.

## 2. Por qué esto NO se resuelve simplemente "poniendo más átomos en general"
Punto importante que la instrucción de julio no marcaba explícitamente: en nuestro motor, cada partícula
SPH es UN átomo de hidrógeno real (masa ~9.4, fija, no un parámetro de resolución que se puede achicar
como en una simulación astrofísica típica, donde cada partícula representa una nube de gas y se puede
subdividir). Subir N no achica la masa por partícula — agrega MÁS átomos, repartidos en un espacio que
también crece. Lo que sí puede pasar es que, al haber más átomos disponibles, la malla causal (el
mecanismo que ya demostró generar estructura real, z=6.92) concentre MÁS átomos en el cúmulo más grande.
Esa es la cantidad que hay que medir — no "N total", sino "cuántos átomos caben juntos en el cúmulo más
grande" — y ver cómo escala con N.

## 3. Medido en esta sesión (motor real, no el juguete #23, semilla="causal" = el puente ya confirmado)
Se extrajo un fondo real de 500 átomos de hidrógeno (masa 9.4 uniforme, densidad #23 real — mismo motor
validado de CS072/CS073) y se submuestreó a distintos N, corriendo la dinámica del puente 3 veces (3
semillas de layout) por cada N, para promediar el ruido de una sola corrida:

| N (átomos) | tamaño del cúmulo más grande (promedio de 3 corridas) |
|---|---|
| 200 | 4.7 |
| 350 | 11.0 |
| 500 | 13.3 |

Ajuste de ley de potencia (log-log, igual método que `cs073_ley_escala.py`): tamaño_cúmulo ≈ 0.0095 × N^1.18
(R²=0.95, sólo 3 puntos — la curva es plausible, no una certeza estadística).

**Extrapolando esa curva hasta el umbral de 116 átomos por cúmulo: N_min ≈ 2.900 átomos totales**
(orden de magnitud 3.000; con sólo 3 puntos de ajuste, tomar esto como una PRIMERA estimación a la que
apuntar, no como un número exacto — hay que confirmarlo corriendo puntos intermedios reales, p.ej.
N=1000 y N=2000, antes de comprometerse a N=3000 en Phantom).

## 4. El costo real de llegar ahí (antes de prometer Phantom a esa escala)
La malla causal se extrae del motor basal (S>0 → átomos), que tiene una pared conocida O(N_quarks²) en
memoria (nota ya dejada en `cs073_ley_escala.py`). De los datos: ~6 quarks producen 1 átomo de H
(500 átomos ↔ nq=3000, consistente con la nota anterior "N=4000 átomos ↔ nq~24000"). Para N_min≈2.900
átomos hacen falta **nq ≈ 17.500 quarks**, con:
- **Memoria estimada ≈ 12-13 GB** (escala como el cuadrado de nq respecto al caso ya medido).
- **Tiempo de extracción del fondo ≈ 1 hora** en esta Mac (escala cuadrática desde los 107s medidos a
  nq=3.000) — sólo para producir el fondo de átomos; la corrida de Phantom (SPH real) es aparte y más
  cara todavía.

## 5. Advertencia honesta sobre el método (no maquillar)
Este N_min sale del modelo CASERO (`_dinamica_estructura`, el kick-térmico de juguete), NO de Phantom.
Se usó a propósito porque es rápido y ya está validado como el que dio z=6.92 — pero Phantom, con
gravedad y presión reales (grad-h), podría concentrar masa distinto (mejor o peor). Este N_min es un
punto de partida razonado para CC, no un reemplazo de correr Phantom mismo a esa escala.

## ADENDA — el punto N=1000 real cambia la estimación (honesto, no se maquilla)
Alexis pidió confirmar la curva con N=1000 y N=2000 reales antes de comprometerse a N≈3.000. Se corrió
N=1000 (fondo real, nq=6.000). Dos hallazgos, ninguno cosmético:

**(a) El costo del modelo "rápido" escala mucho peor de lo esperado.** La extracción del fondo pasó de
107s (nq=3.000) a 1.554s (nq=6.000) — sólo duplicar nq multiplicó el tiempo por ~14.5×, no por 4× (lo
que predeciría una pared O(N²), ya conocida). Y la dinámica del puente en sí (antes ~10s por semilla a
N=500) tardó **~14 minutos por UNA sola semilla a N=1000**. Un intento de extraer N=2000 (nq=12.000)
directamente se colgó y hubo que abortarlo — a esta tasa de crecimiento, N=2000 completo (fondo +
dinámica) costaría varias horas, no minutos. Por el costo, sólo se corrió **una semilla** en N=1000 (no
las 3 promediadas de los puntos anteriores) — es un dato más ruidoso que los otros.

**(b) Con ese punto agregado, el ajuste se mueve fuerte:** tamaño del cúmulo más grande a N=1000 = **20**
(una sola corrida). Reajustando la ley de potencia con los 4 puntos (200→4.7, 350→11.0, 500→13.3,
1000→20): el exponente baja de 1.18 a **0.87** (R²=0.92), y el **N_min extrapolado sube de ~2.900 a
~6.500 átomos** — más del doble de la primera estimación.

**Lectura honesta:** la propia advertencia de Alexis ("confirmá antes de comprometerte") atrapó algo
real — con sólo 3 puntos, la extrapolación no era confiable, y un solo punto nuevo la movió 2×. Con los
datos actuales, lo único defendible es un **rango amplio, no un número: N_min está en algún lugar entre
~3.000 y ~7.000+ átomos**, y conseguir un punto en N=2000 que lo acote mejor costaría varias horas de
cómputo en este modelo casero — el mismo modelo que se usó precisamente PORQUE se suponía rápido. Esto
importa más allá del número: si el modelo "rápido" ya escala así de mal, la corrida real en Phantom (SPH
de verdad, más cara por partícula) a N~3.000-7.000 va a ser un compromiso serio de tiempo de máquina,
aunque sea en la máquina de CC.

## Siguiente paso (decisión de Alexis)
1. Confirmar la curva con 1-2 puntos intermedios reales (N=1000, N=2000) antes de comprometer un N final
   — barato en el modelo casero (minutos), evita apostar a una extrapolación de 3 puntos.
2. Si el N_min se sostiene ~3.000: decidir si se corre la extracción del fondo grande (≈1h, ~13GB) en
   esta máquina o en la de CC, y luego pasar a Phantom a esa escala (Fase 2, PASO 2 de la instrucción).
3. Guardianes vigentes sin cambios: G-CONSERVACION-ENERGIA, G-CONSERVACION-MOMENTO,
   G-PARAMETROS-IDENTICOS-REAL-NULL, G-RESOLUCION-SIGUE-FISICA.
