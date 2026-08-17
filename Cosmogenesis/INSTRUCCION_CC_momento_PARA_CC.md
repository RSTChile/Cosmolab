# INSTRUCCIÓN — Error de momento angular: diagnosticar el patrón, NO forzar
**De:** CS. **Para:** CC. Regla vigente: no mover parámetros a arbitrio; un cambio es un dato a coordinar.

## Estado
- BIEN: h uniforme resolvió el crash de densidad (1.457E4 → ~7). La corrida avanza (t=0.031, 0.062...).
- NUEVO: se detiene con error de momento angular 45% (err=4.548E-01), fatal por defecto en Phantom.
- **PERFECTO que NO usaras I_WILL_NOT_PUBLISH_CRAP=yes.** Esa variable NO se toca jamás — es el análogo de
  aflojar tolerancia, y hasta su nombre lo dice. Un error de conservación se diagnostica o se acepta como
  límite; nunca se silencia.

## PASO 1 — Diagnosticar el PATRÓN del error (no corregir nada aún)
Correr con iverbose alto y registrar, a lo largo de la corrida:
- |ΔL/L| POR PASO (cómo evoluciona: salto brusco al inicio vs crecimiento gradual).
- QUÉ partículas concentran el error (¿unas pocas, los pares más cercanos? ¿o distribuido en muchas?).
Reportar ambas cosas. El patrón decide la hipótesis:
- **(b) ARTEFACTO DE ARRANQUE:** error salta en los primeros 1-2 pasos, concentrado en pocas partículas
  (pares cercanos). Es el v=0 exacto + colapso repentino generando un cambio espurio al inicio.
- **(a)/(c) LÍMITE REAL:** error crece gradual y distribuido → la dinámica densa no se sigue a esta
  resolución. Sería un (B) honesto (límite físico/numérico de fondo), no algo a "arreglar".

## PASO 2 — Corregir SÓLO si es (b), con medio FÍSICO (no relajar tolerancia)
Si el diagnóstico es (b):
- Opción preferida: arranque con las velocidades que la FÍSICA ya provee (campo de expansión / velocidades
  del propio motor), en vez de v=0 exacto — es más físico, no un truco numérico. IDÉNTICO REAL y NULL.
- Alternativa: paso inicial más corto SÓLO en el arranque (los primeros pasos), volviendo al paso normal
  después — NO relajar la tolerancia global de conservación (eso sería Shannon). IDÉNTICO REAL y NULL.
- Reportar |ΔL/L| tras la corrección: debe caer bajo el umbral de Phantom por física, no por silenciar.

Si el diagnóstico es (a)/(c): PARAR y reportar. Es un límite, se adjudica como tal (posible (B) honesto:
"la ignición no se puede seguir a esta resolución con integrador estándar sin recursos mayores"). NO se
fuerza con la variable ni aflojando nada.

## PASO 3 — Continuar Fase 2 sólo con la corrida conservando momento y energía
Si (b) corregido y |ΔL/L| y |ΔE/E| ambos acotados → seguir Fase 2: N~10³, polyk físico idéntico REAL/NULL,
5×8, observable = ¿núcleo cruza M_J por colapso con AMBAS conservaciones OK, y REAL gana al NULL?

## Guardianes
G-CONSERVACION-MOMENTO (nuevo, hermano de G-CONSERVACION-ENERGIA: |ΔL/L| acotado, NO se silencia).
G-NO-FORZAR-INTEGRADOR (I_WILL_NOT_PUBLISH_CRAP prohibida). G-PARAMETROS-IDENTICOS-REAL-NULL. Cualquier
corrección de arranque es física, idéntica en ambos brazos, jamás para que la corrida "pase".

## Orden: 1 → (2 sólo si es (b)) → (3 sólo si conserva). No saltar. El resultado que salga vale sea cual sea.