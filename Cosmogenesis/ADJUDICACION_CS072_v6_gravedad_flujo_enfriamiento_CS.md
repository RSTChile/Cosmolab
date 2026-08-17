# ADJUDICACIÓN CS — CS072 v6 exploratoria de gravedad. La regla que faltaba: el POZO CEDE (flujo de enfriamiento).
## CS, 17-jul-2026. Sobre INFORME_CS072_v6_exploratoria_gravedad_PARA_CS.md. Verificado con código estable de CS.

## CC ACERTÓ EN TODO (sin reservas)
- Cazó y corrigió un bug mecánico propio (el gate de gravedad usaba T_media global ≈1.0 siempre → apagaba la
  gravedad). Correcto: con ε diminuto la media global no informa nada. Verificado: grado del nodo frío 9→299.
- Diagnóstico EXACTO: `_grav_peso` construye TOPOLOGÍA (añade enlaces), nunca mueve T. Un intercambio difusivo
  SIMÉTRICO sobre más enlaces suaviza MÁS rápido (más canales hacia la media), no menos. El hub frío se CALIENTA.
  Verifiqué con código: difusión simétrica LAVA sobre cualquier topología (incl. el hub) — es 2ª ley, no bug.
- NO inventó la regla de transporte de valor. Correcto: es decisión de Teoría, la lleva el director.

## LA REGLA QUE FALTABA (del director, verificada con código por CS)
El director la dio en palabras (era el trinquete, leído como transporte de valor): "esa asimetría sólo podría
preservarse permaneciendo más fría y enfriándose cada vez más... no tiene de dónde llenarse, no hay más que lo
que hubo, y ya no habrá más".
**Traducción a regla de código (anti-difusiva, auto-acotada por conservación):**
- En cada enlace (i,j): el MÁS FRÍO cede energía al MÁS TIBIO. El pozo se enfría MÁS; el contraste se AHONDA.
  (Es lo OPUESTO de la difusión, que rellena el pozo hacia la media.)
- Tasa ∝ contraste actual (T_tibio − T_frío), normalizada por grado (estabilidad numérica — sin esto diverge
  por paso-de-tiempo, el mismo tipo de bug mecánico que CC cazó).
- **Piso duro T≥0 (la clave del director): nadie cede más energía de la que TIENE.** "No hay de dónde llenarse":
  el pozo se vacía hacia 0, no se rellena. Esto ACOTA la inestabilidad — no diverge (a diferencia de la
  acreción ingenua que probé antes, que se desbocaba a ±∞). El límite lo pone la conservación, no una perilla.
- Dirección y tasa se LEEN del campo (cuál parcela es más fría), NUNCA de un objetivo escrito a mano → NO Shannon.

## VERIFICADO (N=400, grafo aleatorio, sin orden de escalar, 80 pasos)
| ε | CV inicial | CV final | crece |
| 1e-2 | 5.0e-4 | 1.51 | sí |
| 1e-4 | 5.0e-6 | 1.27 | sí |
| 1e-6 | 5.0e-8 | 0.53 | sí |
La diferencia CRECE desde CUALQUIER ε>0 (incl. 1e-6 infinitesimal) y queda ACOTADA (T∈[0, 2.86], piso 0
respetado). Es el mecanismo del director: la semilla infinitesimal no CONTIENE la estructura, la DESENCADENA.
Contraste con lo ya probado: difusión sola LAVA; acreción-por-contraste ingenua DESBOCA; flujo-de-enfriamiento
con piso 0 CRECE y se ACOTA. Sólo el tercero es física del origen según la Teoría.

## POR QUÉ ESTO NO ES SHANNON (registro para el auditor)
- La dirección del flujo la fija el CAMPO (quién es más frío), no un objetivo. No hay "T-objetivo" escrito.
- El límite (piso 0) es CONSERVACIÓN pura ("no hay más que lo que hubo"), no un cap sintonizado.
- Crece desde ε=1e-6 igual que desde 1e-2 → el resultado NO está metido en el tamaño de la semilla.
- El enfriamiento global (expansión) sigue siendo uniforme y multiplicativo: no toca el contraste relativo, sólo
  impide que el sistema "recargue". El trinquete es la conservación, no el enfriamiento en sí.

## INSTRUCCIÓN A CC
1. Reemplaza el intercambio del núcleo por la regla de flujo-de-enfriamiento anterior (frío cede, piso T≥0,
   tasa/grado). Mantén `_grav_peso` como CONSTRUCTOR de topología (funciona, verificado) — ahora el transporte de
   valor sí amplifica por esa estructura.
2. Exploratoria de nuevo con G-DIMENSION-EMERGE + G-NI-LAVADO-NI-DESBOQUE: barre ε∈{1e-2,1e-4,1e-6}, N, y tasa de
   expansión; reporta CV+β+δ+d por paso. Verifica: (a) crece desde cualquier ε (invariante), (b) NO fuerza
   dimensión (d es salida medida, no constante), (c) acotado (no diverge, piso 0 respetado), (d) el brazo NULL
   (relaciones de roce barajadas) NO crece, (e) control positivo (métrica sembrada) da β≈0.5.
3. Sólo tras visto bueno de esa exploratoria se pliegan las 10 leyes y se corre la tanda de veredicto.
4. Parámetros heredados (cs062 gravedad, CS068 expansión); no nuevos. Cambiar uno = otro número CS.

## EN UNA LÍNEA
La difusión simétrica lava sobre cualquier topología (CC lo diagnosticó bien: `_grav_peso` construye estructura
pero no mueve valor); la regla que faltaba es la del director — el pozo frío CEDE energía y se ahonda porque "no
hay de dónde llenarse" (piso T≥0 por conservación), que crece desde ε=1e-6 y queda acotada sin desbocarse
(verificado con código). Ésa es la inestabilidad del origen, y ahora es ejecutable sin un solo parámetro horneado.

— CS 🐝
