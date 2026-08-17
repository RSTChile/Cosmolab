# INSTRUCCIÓN — Corregir el layout, re-verificar el puente, y sólo entonces reanudar Fase 2
**De:** CS (diseño + adjudicación). **Para:** CC. Regla de operación vigente: implementar lo especificado,
no modificar a arbitrio; un cambio es un dato a coordinar.

## Contexto
Cazaste (bien) que layout_resortes (p_semilla_causal.py) usa np.clip(pos,0,lado) y apila partículas en las
esquinas por el clip, no por la malla causal (N=250: 246/250 en el borde, 29/250 en esquinas duplicadas).
Es la misma función que produjo z=10.26 → ese resultado queda SUSPENDIDO hasta re-verificar. No se asume ni
que sobrevive ni que muere: se re-corre.

## PASO 1 — Corregir layout_resortes: frontera REFLECTANTE (no clip duro)
- Reemplazar np.clip(pos,0,lado) por reflexión en las paredes: si una coordenada se pasa de [0,lado], se
  refleja de vuelta hacia adentro (pos → 2·borde − pos) y, si hace falta por estabilidad, se invierte la
  componente de velocidad/desplazamiento normal a esa pared. Es el estándar de los layouts por fuerzas
  (Fruchterman-Reingold original), no un invento para este caso.
- NO cambiar nada más del layout: misma repulsión todos-contra-todos, misma atracción por aristas, mismas
  semillas, mismo nº de iteraciones. El ÚNICO cambio es clip → reflexión.

## PASO 2 — Prueba de aceptación del layout (antes de usarlo aguas abajo)
Correr el layout corregido a N=250 y N=1000 y verificar:
- **CERO posiciones duplicadas** (ningún par de partículas en el mismo punto).
- **CERO (o despreciable) apilamiento en bordes/esquinas** (ninguna fracción anómala pegada al borde).
Reportar el conteo. Si aún hay apilamiento, es un dato a diagnosticar — NO se fuerza. G-LAYOUT-SIN-APILAMIENTO.

## PASO 3 — RE-CORRER el experimento del puente con el layout limpio
- Mismos parámetros exactos del puente original (N=250, escala f=5, REAL vs NULL = aristas barajadas,
  ≥5 semillas × ≥8 NULL) — SÓLO cambia el layout (reflectante).
- Medir el MISMO discriminante de antes: clusters ligados REAL vs NULL, z-score.
- **Ese número re-verificado REEMPLAZA a z=10.26, sea cual sea** — mayor, menor, o nulo. Reportarlo tal cual.
- Interpretación pre-inscrita: si el z limpio sigue siendo grande → la coherencia relacional era real, el
  clip no la fabricaba. Si el z limpio se cae a ~0 → el z=10.26 era en buena parte artefacto del apilamiento,
  y el puente NO estaba demostrado. Ambos resultados son honestos; ninguno se retoca.

## PASO 4 — SÓLO si el puente sobrevive limpio: reanudar Fase 2 (Phantom)
- Regenerar los IC de Phantom con el layout corregido (ya no habrá partículas duplicadas → phantomsetup
  las aceptará).
- Continuar Fase 2 como estaba adjudicado: polyk físico (c_s² derivado del piso de enfriamiento H2, MISMO
  en REAL y NULL), N~10³, ≥5×≥8 semillas, observable = ¿núcleo cruza M_J por colapso con energía conservada
  y REAL gana al NULL?
- Si el puente NO sobrevive limpio → PARAR y reportar; no se corre Fase 2 sobre un puente caído.

## Guardianes
G-LAYOUT-SIN-APILAMIENTO (nuevo: 0 duplicados antes de usar el layout). G-DIFERENCIA-INTERNA,
G-TRADUCCION-MECANICA, G-PARAMETROS-IDENTICOS-REAL-NULL, G-CONSERVACION-ENERGIA (heredados).
Regla anti-Shannon central: el layout se corrige por una razón física/numérica (el clip es incorrecto), NO
para que z suba. El z re-verificado vale sea cual sea.

## Orden
Paso 1 → 2 → 3 → (4 sólo si 3 sobrevive). No saltar pasos. Fase 2 en pausa hasta pasar el paso 3 limpio.