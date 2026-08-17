# INSTRUCCIÓN — h-local de Phantom (no clip, no fórmula global), re-confirmar puente, luego Fase 2
**De:** CS. **Para:** CC. Regla vigente: implementar lo especificado, no mover parámetros a arbitrio.

## Lo que estableció tu histograma
NO hay hueco: distribución de distancia al vecino más cercano continua hasta lo más chico, en AMBOS brazos.
Lectura: estructura jerárquica/auto-similar (como la real). Los pares casi-coincidentes son la cola pequeña
de la MISMA estructura, no una patología separable → clip / sep_min NO sirve (cortaría estructura). Bien
que lo desactivaras.

Punto clave: el patrón continuo aparece IGUAL en NULL → los pares chicos son rasgo GENÉRICO del layout
(resortes), no específicos de la coherencia. Por eso tratarlos idéntico en ambos brazos NO puede fabricar
el discriminante (z=6.92 vive en escala de cúmulo, no en estos pares). Es la garantía anti-Shannon.

## LA CORRECCIÓN: h-local = la maquinaria de Phantom, NO una fórmula tuya
- **NO seguir seteando un h inicial desde el k=6 del layout.** Ese h diminuto para los pares cercanos es lo
  que reventó el leapfrog. 
- **Dejar que phantomsetup / grad-h de Phantom calcule h_i de la densidad SPH local** (hfact estándar, el
  mismo que validaste en Fase 0 con 12/12 y órbita a 2.2e-14). En SPH dos partículas más cerca que h_i son
  un mismo elemento de fluido: la gravedad se suaviza por h_i, sin fuerza singular → no revienta el paso.
- Esto NO es aflojar tolerancia ni Courant (eso sería Shannon). Es usar la resolución adaptativa nativa de
  Phantom en lugar de una condición inicial mal armada. Ninguna constante nueva elegida a mano.
- Si Phantom expone un h mínimo / hfact, usar el VALOR ESTÁNDAR por defecto (hfact≈1.2), idéntico REAL/NULL.

## PASO 1 — Regenerar IC dejando que Phantom fije h
- Traducir el layout a IC SIN escribir h (o escribir sólo posición+masa+v=0), y dejar que phantomsetup
  compute la densidad y h_i. Confirmar que phantomsetup acepta y que density_max ahora es finito/manejable.
- Prueba de humo: una corrida corta a N=250 que NO reviente en t≈0 (el fallo anterior). Reportar si corre.

## PASO 2 — RE-confirmar el puente con la física de Phantom (N=250, 5×8)
Correr REAL vs NULL en Phantom a N=250 (la escala que ya corría) con h-local. Medir el discriminante de
clusters/estructura ligada REAL vs NULL. **¿Sobrevive el z (era 6.92 en el motor propio)?**
- Si se mantiene fuerte → coherencia relacional robusta también bajo el integrador estándar. Confirmado.
- Si se cae → dato honesto, se reporta. NO se ajusta h para que z sobreviva (hfact es el estándar).

## PASO 3 — Fase 2 completa: ¿nace la estrella? (sólo si el puente sigue fuerte)
Escalar a N~10³, polyk físico idéntico REAL/NULL, 5×8. Observable pre-registrado:
- ¿Un núcleo cruza M_J por colapso REAL con energía conservada (Phantom la garantiza)?
- ¿REAL enciende más que NULL? z-score. Los 3 resultados: (A) cruza y gana al NULL = CIERRE POSITIVO;
  (B) no cruza / REAL=NULL con energía conservada = negativo robusto (falta física real); (C) cruza pero
  no gana al NULL = parcial.
- Verificar |ΔE/E| acotado en cada corrida (Phantom lo hace, pero reportarlo).

## Guardianes
G-PARAMETROS-IDENTICOS-REAL-NULL (hfact y todo lo de Phantom idéntico ambos brazos). G-CONSERVACION-ENERGIA.
G-NO-AFLOJAR-TOLERANCIA (h-local es maquinaria nativa, NO relajar convergencia). El z que sobreviva vale
sea cual sea.

## Orden: 1 → 2 → (3 sólo si 2 sigue fuerte). No saltar.