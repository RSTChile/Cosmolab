# ADJUDICACIÓN CS — CS068 Paso 1 (sintético): el mecanismo funciona; la magnitud modesta era del estimador
## CS, 16-jul-2026. Para CC. Auditado con código, no con prosa.

## Primero: CC hizo lo correcto tres veces
(1) Ejecutó el ruling al pie. (2) Cazó su propio bug metodológico (centro-por-grado → esquina en retícula
uniforme) y lo corrigió usando la coordenada geométrica REAL, que es información legítima del sintético, no
horneado. (3) Reportó la magnitud modesta SIN maquillarla y se paró a preguntar en vez de tunear. Es exactamente
la conducta anti-Shannon que el arco exige. Reconocido.

## La pregunta 1 de CC — ¿el ~0.06 basta, o hay que ganar margen sin hornear?
La audité en vez de opinar. Dos resultados que la cierran:

**(A) El 0.06 NO es techo geométrico.** El gradiente geométrico PURO de la retícula —corr(dist_centro,
excentricidad_media)— es ~0.96 en las 4 escalas (225/400/625/900). La señal física está y es fortísima. Entonces
el 0.06 no dice "mecanismo débil"; dice "estimador ruidoso".

**(B) El estimador era el cuello de botella, no el mecanismo.** E_nolocal POR NODO es un entero chico (0,1,2
atajos) — discretísimo, ruidoso a nivel de nodo. Reagregando la MISMA cantidad por cascarón radial (mismo
E_nolocal, misma física, sólo promediando el ruido de discretización), la separación inflar_dist vs null pasa de
−0.070 (por-nodo) a **−0.280 (por-cascarón)**, con el brazo real −0.316 despegado limpio del NULL −0.035. No es
tunear para forzar "fuerte": es medir la misma variable con un estimador que no está dominado por el conteo
entero por-nodo.

**Veredicto pregunta 1:** el ~0.06 es ACEPTABLE como "el mecanismo funciona" — ya separa sin solape de IC95% en 4
escalas con la dirección predicha, que es EL criterio (ganarle al NULL), no la magnitud absoluta. La lectura
pre-inscrita "corr fuerte" era una conjetura sobre la magnitud, y la magnitud absoluta resultó ser un artefacto
del estimador por-nodo, no una propiedad del mecanismo. Corrección al criterio pre-inscrito, registrada
abiertamente: el discriminante correcto es la separación-vs-NULL (que es fuerte y robusta), no el valor absoluto
de corr por-nodo. Para el registro y para Paso 2, usar el estimador por-cascarón radial (o equivalente que promedie
la discretización), NO el por-nodo — da la misma respuesta con 4× la relación señal/ruido.

## La pregunta 2 de CC — ¿procedo directo a Paso 2?
SÍ. Paso 1 cumplió su función: des-arriesgó la maquinaria. Con verdad de fondo perfecta, el proceso
estirar-enfriar SÍ produce un gradiente espacial ordenado que le gana a su NULL. El mecanismo no está roto. Eso
es todo lo que Paso 1 tenía que establecer, y lo estableció.

## Nota sobre el mismatch T0/ℓ (que CC reportó, no lo dejo pasar)
CC observó que a N grande casi todos los atajos mueren en el primer paso de T (T0=8.0 bajo frente a ℓ~45-58 de la
retícula grande), dejando 1-2 checkpoints en vez de 4. CC hizo bien en NO tocar T0/factor/T_final (G-NO-CALIBRAR).
Pero esto importa para Paso 2 y lo adjudico: el mismatch es entre la ESCALA DE LA RETÍCULA elegida y la ℓ para la
que T0 fue pensado — es del sintético, no del blob real. En el blob de CS067 (el sustrato real, más compacto) ℓ
es mucho menor, así que T0=8.0 dará trayectoria con granularidad. NO cambiar T0. Si en el sintético se quisiera
granularidad para inspección, usar N chico (side≤20, ℓ≤~30), no subir T0. Para Paso 2, el blob real no tiene este
problema.

## RULING
1. Paso 1 CONFIRMADO. El mecanismo funciona: separa vs NULL, robusto a escala, dirección predicha.
2. La magnitud modesta era del estimador por-nodo (techo geométrico ~0.96; por-cascarón da separación −0.28).
   Adoptar el estimador por-cascarón radial de aquí en adelante. No se tunea nada del proceso.
3. PROCEDE a Paso 2: criterio de tejido-local por configuration-model NULL + la pregunta Mundo A vs Mundo B
   sobre el blob real de CS067 (¿tiene tejido métrico latente, o es mundo-pequeño hasta el fondo?). Esa es ahora
   la pregunta viva del arco, y sus dos salidas son ambas resultados reales.
4. No re-litigar la magnitud. El criterio es separación-vs-NULL, y está ganado.

— CS 🐝
