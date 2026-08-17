# INFORME CS069 — smoke: 2/3 anclas PASAN, el ancla decisiva (2) FALLA — no corro la tanda

## CC, 17-jul-2026. Para CS. Ejecuta DISENO_CS069_frente_cuantico_CS.md.

## Qué construí
`cs069_quantum_graph.py`: matriz de amplitud A_ij=ρ_ij·e^(iφ_ij) sobre el motor de los 17 (ρ_ij=w_ij,
costo=d_ij=-log(w_ij), la MISMA cantidad de H._pesos_correlacion, ingrediente 14 — sin inventar una fuente
nueva); dinámica de fase Kuramoto LOCAL ciega (φ_ij(t+1)=φ_ij(t)+η·mean_vecino(sin(φ_vecino-φ_ij))-κ·costo,
auditable: la función solo recibe topología+costo, nunca es_atajo); K_ij(L)=Σ potencias de la matriz de
amplitud hasta L=8 (fijo); D_q=-log|K|. Los tres jueces (A cedazo de π, B pendiente log-log de diám_q, C
gap espectral vía MDS clásico + picado_por_nodo/cuenta_ejes_gap de CS067 reusados tal cual) y los 4 brazos
(completo, null_fase_topo, null_fase_azar, null_clasico) están implementados y corren sin error.

## Dos bugs de implementación que encontré y corregí ANTES del smoke (no estaban en tu diseño)
1. **Amplitud sin normalizar explotaba.** |K_ij(L)| sin control crecía combinatoriamente (nº de caminos
   ~grado^l le gana a ρ<1) dando D_q NEGATIVO para casi todo par. Fix: ρ_ij_norm=ρ_ij/√(grado_i·grado_j)
   (normalización estándar de caminata cuántica/Laplaciano normalizado — cantidad local, no rompe
   G-FASE-CIEGA).
2. **Pares no alcanzados dentro de L=8** (K_ij(L)=0 exacto) se leían como D_q~690 (un artefacto del piso de
   corte numérico, no una distancia real). Fix: se marcan NaN y se excluyen de los jueces; en el MDS se
   rellenan con un tope robusto (percentil 99×1.5) para no romper el embedding.

## ANCLA 1 — NULL_CLASICO reproduce Mundo B de CS068: **PASA**
diam_q(N) en N∈{900,1500,2500}: medias 24.75→29.09→25.81. Pendiente log-log=**0.041** (<<0.3). Plano,
consistente con el Mundo B de CS068 (nota: D_q es una métrica distinta —log-amplitud sobre integral de
camino, no BFS sobre tejido residual— así que comparo la CONCLUSIÓN cualitativa, no el valor).

## ANCLA 3 — los NULLs no encienden geometría solos: **PASA**
π-CV medio: null_fase_topo=1.210, null_fase_azar=1.091 (ambos >>0.30, siguen estallando, no convergen
espontáneamente).

## ANCLA 2 — decoherencia de atajos en juguete con verdad de fondo: **FALLA**, y encontré 2 problemas en el
camino antes de llegar a un resultado que confío
Primer intento (retícula 2D de 4-vecinos, la misma de CS068 Paso1): **GIGO** — w_ij (soporte por vecinos
comunes, ingrediente 14) salió CASI IGUAL para local y atajo (0.205 vs 0.218) porque una retícula de
4-vecinos NO TIENE TRIÁNGULOS. Es la MISMA lección de tu propio ruling de CS068 (retícula 2D pura reprueba
el CM-null por no tener triángulos) — la aprendí de nuevo por las malas, en un contexto distinto. Arreglé
con retícula de 8-vecinos (Moore, con diagonales) que SÍ tiene triángulos locales: w_ij ahora discrimina
limpio (local=0.628, atajo=0.113).

Con eso arreglado, comparé Δ_Dq=D_q(fase evolucionada)−D_q(φ≡0). Salió AL REVÉS (local Δ=0.307 >
atajo Δ=0.136) — pero antes de reportarlo audité la comparación misma: φ≡0 es el MÁXIMO teórico de
coherencia para cualquier topología (por desigualdad triangular, ningún φ puede superar |K| de φ≡0), así
que Δ contra ese piso mide sobre todo CUÁNTOS CAMINOS tiene el par (multiplicidad), no si la dinámica hizo
algo bueno — y los locales, con muchos más caminos alternos que los atajos, decoherencian más contra
CUALQUIER cosa que no sea perfectamente coherente, por pura combinatoria. Corregí al baseline correcto que
SÍ controla la multiplicidad de caminos (misma topología): fase evolucionada vs fase AL AZAR (el mismo
contraste que NULL_FASE_AZAR real).

**Resultado final, con las dos correcciones aplicadas:** Δ_Dq(evolucionada−azar) atajo=0.053, local=0.191.
Sigue AL REVÉS — los locales se decoherencian MÁS que los atajos relativo a una fase al azar en la MISMA
topología, con el confound de multiplicidad ya controlado. No es un artefacto de medición que yo pueda
seguir cazando sin empezar a ajustar la regla hasta que salga (lo que sería exactamente el error que este
proyecto existe para evitar).

## Lo que NO sé si es un problema del MECANISMO o de MI INTERPRETACIÓN de tu diseño
Tu texto especifica "desajuste de fase con los vecinos del enlace" sin dar la fórmula exacta. Implementé
acoplamiento Kuramoto local: φ_ij tira hacia la fase MEDIA de los enlaces que tocan i o j. Sospecha (sin
verificarla más, para no seguir ajustando): un enlace LOCAL tiene ~14+ enlaces vecinos (grado~8 en la
retícula 8-vecinos) tirando de él — sincroniza RÁPIDO y FUERTE dentro de su vecindad. Un ATAJO también
tiene vecinos (los enlaces locales que tocan sus dos extremos), así que TAMBIÉN es arrastrado hacia
sincronía local, sin que la regla distinga que es un puente entre DOS vecindarios que podrían estar en
fases distintas. Puede que "desajuste con los vecinos del enlace" necesite otra forma (p.ej. pesar el
desajuste por ρ del vecino, o mirar la DISPERSIÓN entre los dos extremos del enlace en vez de la media de
vecinos) para que el ciego SÍ aísle atajos. No lo intenté — quería reportar el ancla tal como cayó, no
iterar la regla hasta que el ancla pasara.

## Lo que pido adjudicar
1. ¿La forma de "desajuste de fase" que implementé (Kuramoto local, media de vecinos del enlace) es la que
   tenías en mente, o hay una forma distinta que el diseño sí especifica y yo interpreté mal?
2. Dado que ancla 2 falla de forma robusta (dos confounds cazados y corregidos, mismo resultado invertido
   las dos veces): ¿reviso la regla de fase, o esto ya cuenta como el desenlace (C)/(B) de tu lectura
   pre-inscrita (el mecanismo tal como está no decohere atajos — sería, si acaso, un Mundo B cuántico
   temprano, aunque el smoke no es la tanda)?
3. Costo de la tanda completa si se autoriza: medí tiempos reales — brazo completo ~40s a N=2500 (fase
   evolucionada + integral de camino), nulls ~30-35s cada uno (sin evolución de fase). Estimo la tanda
   completa (4 brazos × 8 semillas × 3 N) en **~30-35 min** de cómputo. Factible, pero no la corro sin luz
   verde dado que el ancla decisiva no pasó.

— CC 🐝
