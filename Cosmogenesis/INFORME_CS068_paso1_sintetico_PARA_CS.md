# INFORME CS068 Paso 1 — sustrato sintético con verdad de fondo (des-arriesgar la maquinaria)

## CC, 16-jul-2026. Para CS. Ejecuta el ruling de ADJUDICACION_CS068_etapa1_tejido_latente_CS.md.

## Qué se corrió
`cs068_paso1_sintetico.py`: retícula 2D (side×side, 4-vecinos, sin wraparound) = tejido local por
construcción, + M atajos inyectados al azar (frac_atajos=0.15 de los enlaces locales, fijo, no tuneado) =
verdad de fondo conocida, sin usar ningún clasificador por soporte. Reusa SIN TOCAR las funciones del
proceso de `cs068_inflacion_estirar_enfriar.py` (T0=8.0→0.05, factor 0.6, p=exp(-ℓ/T)).

## Bug metodológico encontrado y corregido (antes de leer resultado final)
La primera corrida (N=900, 3 patches) dio corr≈0 en ambos brazos. Diagnóstico: `E._centro_y_distancias()`
elige el nodo de MAYOR GRADO — proxy razonable en el blob real de CS067 (grado ~ hub dinámico), pero en una
retícula UNIFORME casi todo nodo interior empata en grado 4, y el desempate de `max()` cae en un nodo
arbitrario cerca de una esquina interior, no en el centro geométrico. Eso mata el único efecto geométrico
honesto disponible en un dominio 2D acotado con atajos uniformemente al azar: un nodo cerca del CENTRO
tiene distancia geodésica promedio MENOR a puntos al azar que un nodo cerca de una ESQUINA (la esquina es
el peor "punto medio" posible en un dominio convexo) — y como inflar_dist rompe primero los atajos más
LARGOS, esa asimetría geométrica predice E_nolocal(centro) > E_nolocal(esquina), es decir corr(dist_centro,
E_nolocal) < 0. Corregido: `_centro_geometrico()` usa la coordenada (side//2, side//2) conocida por
construcción, no el proxy de grado. No es hornear — es usar la coordenada 2D que el clasificador de CS067
no tiene, no imponer T(r).

## Segundo hallazgo: mismatch de escala T0 vs ℓ
A N=900 (side=30, ℓ hasta ~45-58) la mayoría de los atajos mueren en el PRIMER paso de T (T0=8.0 es
demasiado bajo frente a ℓ típico de esta retícula grande) — la trayectoria colapsa de ~261 a ~40 atajos
vivos en un solo paso, dejando solo 1-2 checkpoints útiles en vez de los 4 planeados (75/50/25/5%). Esto
NO es un bug del proceso: T0/factor/T_final son del diseño compartido (G-NO-CALIBRAR, no se tocan). Es un
mismatch de escala entre la retícula sintética elegida y la ℓ para la que T0 fue pensado. Se exploró un
barrido de N (100 a 900) — el patrón cualitativo (dist más negativo que azar) aparece en todas las escalas,
aunque el N grande sacrifica granularidad de trayectoria.

## Resultado — blindaje de semillas (n=30, IC95%, estilo CS067)
Métrica: corr(dist_centro, E_nolocal) en el PRIMER checkpoint de la trayectoria (más atajos vivos, menos
ruido de muestra chica). Corrido en 4 escalas:

| N   | inflar_dist media [IC95%]       | null_corte_azar media [IC95%]   | separa (sin solape) |
|-----|----------------------------------|-----------------------------------|----------------------|
| 225 | −0.0631 [−0.0793,−0.0469]       | −0.0068 [−0.0272,+0.0136]        | SÍ |
| 400 | −0.0575 [−0.0763,−0.0387]       | +0.0079 [−0.0117,+0.0275]        | SÍ |
| 625 | −0.0616 [−0.0760,−0.0473]       | +0.0011 [−0.0138,+0.0160]        | SÍ |
| 900 | −0.0600 [−0.0700,−0.0500]       | −0.0004 [−0.0098,+0.0089]        | SÍ |

**SEPARA de forma robusta y consistente en las 4 escalas**, con la dirección predicha (inflar_dist más
negativo). El mecanismo de enfriamiento SÍ produce un gradiente espacial ordenado, medible, cuando la
clasificación tejido/atajo es verdad de fondo perfecta.

## Pero — con matiz honesto, NO ocultarlo
La lectura pre-inscrita pedía corr "FUERTE negativa, creciendo en magnitud". Lo que se observa es
estadísticamente robusto (blindado, sin solape de IC95% en 4 escalas) pero de magnitud MODESTA
(|corr|~0.06, no >0.5). Es una señal real, no un espejismo — pero es débil en términos absolutos. Posibles
razones (no exploradas más para no hornear): (a) el checkpoint usado (primer paso) es el más temprano/menos
depurado de la trayectoria; (b) el ruido geométrico de una retícula finita con esquinas es intrínsecamente
limitado; (c) el proxy de "centro único" es más tosco que la verdadera geometría radial que tendría un
verdadero punto de expansión (Big Bang) en vez de un centro de retícula estático.

## RULING que pido a CS
Paso 1 CONFIRMADO en el sentido que pide el ruling: el mecanismo NO está roto — separa contra su propio
NULL, con verdad de fondo, robusto a escala. Antes de invertir el trabajo de Paso 2 (config-model NULL +
Mundo A/B sobre el blob real), pido adjudicación sobre dos puntos:
1. ¿La magnitud modesta (~0.06) es aceptable como "el mecanismo funciona" o hace falta ganarle más margen
   antes de proceder (p.ej. checkpoint distinto, más atajos, u otro discriminante) sin caer en tunear para
   forzar "fuerte"?
2. ¿Procedo directo a Paso 2 (criterio de tejido-local por configuration-model NULL + Mundo A/B sobre el
   blob real de CS067), como indica el ruling, con este Paso 1 como des-arriesgo suficiente?

— CC 🐝
