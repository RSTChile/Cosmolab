# DISEÑO — Ignición de la primera estrella (CS073): colapso real, no extrapolación
**De:** CS (diseño + adjudicación). **Para:** CC (implementación + corrida).

## REGLA DE OPERACIÓN (Alexis, explícita) — CC NO MODIFICA EL EXPERIMENTO A SU ARBITRIO
CC implementa EXACTAMENTE lo que este diseño especifica, y lo corre. NO cambia parámetros, softening,
módulos ni criterios por su cuenta — ni siquiera para "corregir" algo que parezca mal. Si algo parece
un bug o inconsistencia, se REPORTA y se coordina ANTES de tocar; un cambio es un DATO a adjudicar, no
una decisión de CC. (Precedente: el cambio de softening de H2 —aunque corregía un artefacto real— tuvo
la consecuencia no vista de tapar el colapso; por eso todo cambio pasa por adjudicación de CS.)

## Diagnóstico (medido por CS, no opinado): qué detiene la ignición
NO es la física ni el sustrato — es la RESOLUCIÓN NUMÉRICA. El control positivo YA forma colapso
(×375-1500). Barrido de softening (nube favorable, N=800, todo igual salvo soft):
| softening | colapso × |
|---|---|
| 0.30 (el usado) | 375 |
| 0.10 | 1500 |
| 0.03 | 29 (inestable) |
| 0.01 | 9 (inestable) |
El softening FIJO=0.3 pone un PISO a la densidad del núcleo: la gravedad se apaga por debajo de esa
distancia → el colapso se detiene antes de densidad proto-estelar → nunca cruza Jeans de verdad. Bajar el
softening a mano NO sirve: por debajo de 0.03 la integración con paso fijo se vuelve INESTABLE y dispersa
(el colapso BAJA, no sube). Los dos problemas (piso de densidad + inestabilidad) son el mismo: resolución
fija en un sistema cuya densidad crece muchos órdenes.

## La solución (técnica ESTÁNDAR de N-cuerpos, NO un parámetro elegido = NO Shannon)
**Softening adaptativo + paso de tiempo adaptativo:** la resolución se REFINA donde la densidad crece.
- softening local ε_i = k=6-ésimo vecino (≡ ρ^(−1/3)) — se afina solo donde hace falta, no un número
  elegido a mano. En regiones difusas ε grande (estable); en el núcleo denso ε chico (colapsa profundo).
  Convención estándar (Gadget, etc.). **La MISMA ε_i alimenta gravedad Y ρ_local de H2/Jeans** (ver
  adjudicación abajo) — resolución única, no dos pisos que se desincronicen.
- paso de tiempo adaptativo Δt ∝ 1/√(Gρ_max) — se acorta cuando la densidad sube, para que la integración
  NO explote aunque ε sea chico. Resuelve la inestabilidad que vimos a soft=0.01.
Ninguno es un parámetro "elegido para que cruce Jeans": ambos son funciones de la densidad local, se
ajustan solos. G-PARAMETROS-ESTRUCTURALES: la resolución sigue a la física, no se decreta.

## ADJUDICACIÓN de la pregunta de CC (coordinada antes de tocar — CONFIRMADO SÍ)
CC cazó una trampa real que el diseño v1 no veía: si SÓLO la gravedad tiene ε adaptativo pero el estimador
de densidad de H2 (_densidad_local_dinamica, que da ρ_local para Jeans) mantiene el piso fijo 0.3, la
gravedad colapsaría más profundo PERO el criterio de Jeans no lo vería → el experimento fallaría por la
MISMA razón, escondida un paso atrás. Correcto, y es justo el error de acoplamiento desincronizado que
causó el problema original.

**RESUELTO (adjudicación CS): UNA sola longitud de suavizado adaptativa ε_i por partícula, COMPARTIDA por
construcción entre el softening de la gravedad y el ρ_local de H2/Jeans.** Así "lo que la gravedad ve" y
"lo que Jeans mide" son la MISMA resolución, no dos números que puedan desalinearse. Más limpio que tratar
los módulos por separado.

**Constantes confirmadas (estándar, NO ajustadas):**
- ε_i = distancia al k=6-ésimo vecino más cercano (≡ ε∝ρ_local^(−1/3), sin constante extra). k=6 es el que
  H2 YA usa — ninguna constante nueva.
- Δt = 0.1·t_ff(ρ_max), t_ff=√(3π/32Gρ) — fórmula de caída libre YA en el inventario del arco; 0.1 =
  convención de ~10 pasos por tiempo dinámico en N-cuerpos.
Ninguna elegida para que cruce Jeans; ambas son convención. Cumplen G-PARAMETROS-ESTRUCTURALES.

## ADJUDICACIÓN 2 — costo del paso global (CC paró y preguntó, correcto): OPCIÓN 1, con guarda
CC cazó un problema de costo real: Δt GLOBAL atado a ρ_max (16→24→46→162→15387) → basta UN par de CDM sin
presión formando encuentro cercano para que TODO el sistema use paso minúsculo; sub-pasos explotan
(4→35...), N=200 no termina en 300s. Diagnóstico correcto: problema conocido del paso global en N-cuerpos,
no bug ni física fallando.

**DECISIÓN: opción 1 (paso de tiempo INDIVIDUAL/JERÁRQUICO por partícula). Se DESCARTA la opción 2** (tope
de presupuesto + reportar "no llegó = B"). Razón de método: un (B) debe significar "el mecanismo NO
enciende estrella", NO "se acabó el CPU". Aceptar un corte por presupuesto como (B) confundiría un límite
computacional con un resultado físico = conclusión falsa que el pacto prohíbe. El (B) sólo vale si la
física no enciende con RESOLUCIÓN PLENA, no si la máquina se quedó corta.

- Paso individual = cada partícula con SU Δt según SU densidad local (no todas atadas al extremo). Es la
  extensión estándar del MISMO método (Gadget et al.); NINGÚN parámetro nuevo elegido — misma ε_i, mismo
  t_ff, por partícula en vez de global. NO es Shannon.
- **GUARDA CRÍTICA (G-CONSERVACION-ENERGIA):** el paso individual MAL hecho rompe la conservación de
  energía (partículas con pasos distintos que interactúan ganan/pierden E espuria) → invalidaría el
  colapso. CC DEBE verificar que la deriva de energía total |ΔE/E| quede acotada (p.ej. <1e-2) como
  chequeo de cordura ANTES de confiar en cualquier ignición. Si la energía deriva, el colapso no es
  físico y no cuenta. Reportar la deriva junto con el resultado.

## El experimento
Mismo bucle del puente (malla causal como semilla dinámica + expansión + gravedad general + CDM + H2),
pero con gravedad de resolución ADAPTATIVA. Correr hasta que un núcleo alcance densidad de Jeans por
COLAPSO REAL (no por extrapolación), o hasta un tiempo cosmológico máximo.

## Observable de cierre (pre-registrado, REAL vs NULL en el punto de ignición)
- ¿Un núcleo cruza M_J LOCAL por colapso real (masa/M_J ≥ 1 medido, no extrapolado)?
- **Discriminante que lleva el peso (heredado del arco): REAL vs NULL en la MISMA corrida.** NULL = aristas
  de la malla causal barajadas. ¿REAL enciende (cruza Jeans) significativamente más que el NULL? z-score,
  ≥5 semillas × ≥8 NULL. NO basta "cruzó Jeans en absoluto" (ya se retractó dos veces) — tiene que
  GANARLE AL NULL en la ignición.
- Cierre HONESTO sin extrapolar 60 décadas: la estrella nace por colapso medido, y el NULL dice si fue la
  coherencia relacional o azar.

## Tres resultados pre-inscritos
- **(A) CIERRE POSITIVO:** con resolución adaptativa un núcleo cruza Jeans por colapso real Y REAL gana al
  NULL. → la primera estrella EMERGE del mecanismo completo; era resolución, no física. Cierra Cosmogénesis.
- **(B) NEGATIVO:** con resolución adaptativa el núcleo AÚN no cruza Jeans, o REAL=NULL en la ignición. →
  el mecanismo forma estructura ligada pero NO enciende estrella ni con resolución plena; falta un
  ingrediente físico real (no numérico). Cierre robusto.
- **(C) PARCIAL persiste:** cruza en absoluto pero no le gana al NULL → estructura sí, especificidad no.

## Guardianes
G-DIFERENCIA-INTERNA (NULL = aristas barajadas). G-SIN-SIEMBRA. G-SIN-ENERGIA-NUEVA. G-EXPANSION-ISOTROPA.
G-PARAMETROS-ESTRUCTURALES (ε y Δt son funciones de la densidad local, NO valores elegidos).
G-RESOLUCION-SIGUE-FISICA (la resolución adaptativa refina donde la física lo pide; jamás se sintoniza
para que un núcleo cruce Jeans — si cruza, es por colapso; si no, es dato).

## Costo
O(N²) + paso adaptativo (más caro). Escala/tiempo → entorno de CC / segundo plano, no kernel de CS.
Motor CONGELADO salvo el módulo de gravedad adaptativa que este diseño especifica. Nada más se toca.