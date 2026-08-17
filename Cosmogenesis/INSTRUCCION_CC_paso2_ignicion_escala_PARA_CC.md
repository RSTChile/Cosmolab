# INSTRUCCIÓN — PASO 2: correr la ignición en Phantom a la escala del rango (decisión de Alexis, 31-jul)
**De:** CS/CC (sesión Claude Code). **Para:** CC (máquina con Phantom compilado, i7-10700K).
**Continúa:** INSTRUCCION_CC_escala_ignicion_PARA_CC.md (PASO 1 cerrado, ver RESULTADO_PASO1_N_min_ignicion_CS.md).

## PASO 1 — cerrado, con un rango honesto, no un número
Se calculó N_min con el criterio de Bate & Burkert (1997) y el N_neigh real de Phantom (57.9, confirmado
en Price et al. 2018, hfact=1.2). Corriendo el motor real (`_dinamica_estructura`, semilla="causal", el
mismo del puente z=6.92) a N=200/350/500 (promedio de 3 semillas) y N=1000 (1 semilla, el costo de una
segunda semilla ya era ~14 min), el ajuste de ley de potencia es INESTABLE: con 3 puntos daba N_min≈2.900;
agregando el punto N=1000 subió a ≈6.500. **Adjudicación: no hay un N_min preciso defendible — hay un
RANGO, aproximadamente 3.000-7.000 átomos.** Conseguir un punto confiable en N=2000 (3 semillas) para
angostar el rango costaría varias horas en el modelo rápido (la extracción del fondo ya escala peor que
O(N²): 107s→1.554s al duplicar nq). Alexis decidió NO perseguir más precisión ahí — aceptar el rango y
dejar que la corrida REAL en Phantom decida, empezando por el extremo BAJO.

## PASO 2 — correr Phantom en el extremo bajo del rango, escalando si hace falta
1. Generar el IC con `fase1_traducir_a_phantom.py` a **N≈3.000 átomos reales** (extremo bajo del rango;
   quarks≈6× esa cifra por la razón ya validada, nq≈18.000 — confirmar con la extracción real, no asumir).
2. Correr Phantom (misma configuración de Fase 0/1: GRAVITY=yes, PERIODIC=no, h uniforme, polyk=T_piso
   derivado, vel=0 salvo que se use vel_generador). Registrar |ΔE/E| y |ΔL/L| por el mismo método que
   diagnosticó la sub-resolución a N=250 (temporal + espacial + correlación con densidad/h).
3. **Prueba de que el diagnóstico de sub-resolución era correcto:** los errores de energía y momento
   deben CAER respecto a lo visto en N=250 (0.13→0.60 energía; 45% momento angular). Si caen → la
   resolución estaba efectivamente detrás del problema, seguir. Si NO caen → dato nuevo a diagnosticar
   (no se fuerza, no se usa I_WILL_NOT_PUBLISH_CRAP), reportar el patrón (temporal/espacial/correlación)
   igual que se hizo a N=250.
4. Si a N≈3.000 los errores ya caen a un nivel aceptable Y el sistema resuelve el colapso: proceder al
   PASO 3 (observable de cierre) a esa escala. Si los errores siguen altos, subir un escalón dentro del
   rango (p.ej. N≈5.000) y repetir el chequeo — no hace falta ir directo a 7.000 si 5.000 ya resuelve.

## PASO 3 — el observable de cierre (sin cambios respecto a la instrucción original)
A la escala que resuelva (≥5 semillas × ≥8 NULL): ¿un núcleo cruza M_J local y colapsa (densidad crece
varios órdenes) con energía y momento conservados? ¿REAL enciende más/antes que NULL? Los 3 resultados
pre-inscritos siguen intactos:
- (A) cruza por colapso real, conservando, y gana al NULL → CIERRE POSITIVO.
- (B) con resolución suficiente y conservación OK aún no cruza / REAL=NULL → negativo robusto, FÍSICO.
- (C) cruza pero no gana al NULL → parcial.

## Guardianes (sin cambios)
G-CONSERVACION-ENERGIA + G-CONSERVACION-MOMENTO. G-PARAMETROS-IDENTICOS-REAL-NULL.
G-RESOLUCION-SIGUE-FISICA (el N sale de la física/costo, no se elige para que encienda).

## Nota de costo (para presupuestar tiempo de máquina)
El modelo rápido usado para PASO 1 escaló peor de lo esperado (casi cúbico, no cuadrático) tanto en la
extracción del fondo como en la dinámica N-cuerpo. Aunque Phantom es un código optimizado (no el toy de
juguete) y corre en otra máquina, esto es una señal de que N≈3.000-7.000 con gravedad+SPH real puede
tomar bastante más tiempo de lo que tomaron las corridas de Fase 0/1 (N=250-1000). Presupuestar en
consecuencia (ya se toleraron corridas de ~1 día en esa máquina).
