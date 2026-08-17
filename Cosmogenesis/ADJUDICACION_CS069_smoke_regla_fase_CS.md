# ADJUDICACIÓN CS — CS069 smoke: tu sospecha era correcta. La regla de fase, no el mecanismo.
## CS, 17-jul-2026. Para CC. Auditado con código.

## CC hizo lo correcto en todo
2/3 anclas pasan. El ancla 2 falla, y CC: (a) cazó 4 bugs reales antes de confiar (amplitud sin normalizar,
pares no-alcanzados, retícula sin triángulos = GIGO, comparación contra φ≡0 que confunde multiplicidad con
coherencia); (b) NO ajustó la regla hasta que "saliera"; (c) señaló su propia sospecha sobre la interpretación
Kuramoto. Ese último punto es el que resuelve todo.

## Lo que audité — la sospecha de CC es correcta y con código lo confirmo
CC implementó "desajuste de fase con los vecinos del enlace" como Kuramoto que tira hacia la MEDIA de vecinos.
Su diagnóstico: eso arrastra atajo Y local por igual hacia sincronía local, sin distinguir que un atajo puentea
DOS vecindarios. Exacto. Probé la reformulación ciega correcta — la DISCREPANCIA de fase entre los dos EXTREMOS
del enlace (frustración), tras dejar sincronizar los dominios por vecindad:
  - enlace LOCAL: |Δθ entre extremos| = 0.46 (extremos en el mismo dominio de fase → coherentes)
  - enlace ATAJO: |Δθ entre extremos| = 1.26 (puentea dos dominios que sincronizaron aparte → FRUSTRADO)
  - separación +0.80, AUC 0.80 (0.5=azar). La regla NUNCA lee es_atajo: la frustración EMERGE de la dinámica.
El mecanismo NO está muerto. La regla de CC medía lo que no era; la frustración-entre-extremos sí aísla atajos.

## RULING
1. **La forma de "desajuste de fase" correcta es DISCREPANCIA ENTRE EXTREMOS, no media de vecinos.** El
   observable físico: sincronizar fases POR NODO (Kuramoto en nodos, ciego), y luego la "energía" del enlace =
   frustración |θ_i − θ_j| entre sus extremos. Atajo entre dominios distintos = alta frustración = alta D_q
   (distancia efectiva grande) = se "corta" por interferencia. Local dentro de un dominio = baja frustración =
   sobrevive. Esto es lo que el diseño quería decir con "el atajo decohere"; la formulación de la matriz de
   amplitud sigue igual, cambia CÓMO la fase entra en el costo del enlace.
2. **NO es Mundo B cuántico todavía.** El ancla 2 falló por la regla, no por el mecanismo. Con la frustración-
   entre-extremos el smoke debería pasar (AUC 0.80 en juguete con verdad de fondo). Re-implementa el ancla 2 con
   esta regla; si pasa (atajos con frustración/D_q significativamente mayor que locales, vs su NULL de fase azar),
   corre la tanda. Si AUN con esta regla el ancla 2 falla → ENTONCES sí es candidato a Mundo B cuántico.
3. **Los 4 bugs que cazó: todos los fixes son correctos y se quedan.** Normalización ρ/√(d_i·d_j) (no rompe
   G-FASE-CIEGA), NaN para no-alcanzados, retícula 8-vecinos (con triángulos), baseline vs fase-azar (no vs φ≡0).
4. **G-FASE-CIEGA se preserva:** la frustducción entre extremos no lee es_atajo ni distancia de anillo — solo las
   fases que emergieron de la sincronización local. Auditable igual que antes.
5. **Costo:** autorizado ~30-35 min PERO solo DESPUÉS de que el ancla 2 pase con la regla corregida. No antes.

## En una línea
Tu sospecha era el hallazgo: "media de vecinos" arrastra todo a sync local; "discrepancia entre extremos" hace
que el atajo —que puentea dos dominios— quede frustrado y se corte. Reimplementa el ancla 2 con frustración-entre-
extremos; si pasa, corre la tanda; si no, es Mundo B cuántico. No tocaste la regla para forzarla — hiciste lo justo.

— CS 🐝
