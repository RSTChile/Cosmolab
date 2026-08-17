# INSTRUCCIÓN — sep_min correcto (del hueco en distancias), re-confirmar puente, luego Phantom
**De:** CS. **Para:** CC. Regla vigente: implementar lo especificado, no mover parámetros a arbitrio.

## Tu catch fue correcto
sep_min=1.2 venía de la densidad PROMEDIO del box (uniforme, ρ≈9.4) — enorme vs la estructura real
(mediana vecino más cercano ~0.12-0.4). Imponerlo forzaba casi-uniformidad y destruía la señal. Bien
cazado. La densidad real es ~78× la uniforme porque el layout agrupa: la densidad uniforme es la referencia
EQUIVOCADA. Descartada.

## Principio: sep_min es SÓLO anti-crash, NO toca el agrupamiento
Su único fin es que pares sub-resolución no revienten el leapfrog. Debe ser PEQUEÑO (muy por debajo de
0.12) y salir de los DATOS, no de una fórmula de densidad media.

## PASO 1 — Histograma de distancia al vecino más cercano (REAL y NULL, semillas del puente)
Graficar/tabular la distribución de la distancia al vecino más cercano. Buscar un HUECO:
- **Si hay hueco:** un pico de pares patológicos en ~0 (los que dan density_max=1.457E4 y revientan Phantom)
  separado del grueso físico en ~0.12-0.4. → sep_min = un valor DENTRO del hueco (p.ej. el mínimo de la
  densidad del histograma entre el pico y el bulk). Clipa sólo lo patológico.
- **Si NO hay hueco** (continuo de pares hasta 0, sin separación): la patología es inseparable de la
  estructura = hallazgo. PARAR y reportar — recién ahí se consideraría h-local auto-consistente.
Reportar el histograma y cuál de los dos casos es.

## PASO 2 — Aplicar sep_min del hueco (si lo hay), idéntico REAL y NULL
- Reposicionar SÓLO los pares por debajo de sep_min (separarlos al mínimo), sin tocar el resto.
- MISMO sep_min en ambos brazos. G-PARAMETROS-IDENTICOS-REAL-NULL.
- **Verificación crítica:** la FRACCIÓN de partículas movidas debe ser TINY (sólo los pares patológicos, el
  pico en ~0). Reportar esa fracción. Si mueve una fracción grande (como el 1246/1250 anterior), el piso es
  demasiado grande — está mal. Debe ser un puñado.
- NO reinventar grad-h: Phantom ya computa h local (validado Fase 0). El layout sólo evita coincidencias.

## PASO 3 — Prueba de aceptación: 0 pares bajo sep_min, estructura intacta
Verificar que ningún par queda bajo sep_min Y que la mediana de distancia al vecino más cercano NO cambió
apreciablemente (la estructura sigue ahí). Reportar mediana antes/después.

## PASO 4 — RE-confirmar el puente con el layout ya sin pares patológicos
Re-correr el puente (N=250, 5×8). ¿Sobrevive el z (era 6.92)? Ese número reemplaza al 6.92.
- Si se mantiene → coherencia relacional robusta a la resolución. Confirmado firme.
- Si se cae → parte del 6.92 dependía de los pares patológicos. Dato honesto, se reporta.
NO se ajusta sep_min para que z sobreviva; sep_min está fijado por el hueco en el paso 1.

## PASO 5 — SÓLO si el puente sigue fuerte: Phantom Fase 2
IC con el layout limpio (sin pares que revienten el leapfrog) → N~10³, polyk físico idéntico REAL/NULL,
5×8, observable = ¿núcleo cruza M_J por colapso con energía conservada y REAL gana al NULL? Si Phantom aún
se cae con el layout ya limpio, es dato a diagnosticar (no aflojar tolerancia).

## Orden: 1 → 2 → 3 → 4 → (5 sólo si 4 sigue fuerte). No saltar.
Regla central: sep_min sale del HUECO en los datos, es pequeño, idéntico REAL/NULL, y mueve una fracción
tiny. Jamás se elige para que z suba o Phantom converja. El z que sobreviva vale sea cual sea.