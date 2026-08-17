# ADJUDICACIÓN C-N4 — ¿la frontera "amigos-de-amigos" (FoF) es real o arbitraria?

**Fecha:** 5-ago-2026 · **Frente:** Fase I-C, prioridad P0 (roadmap multi-IA) · **Nodo:** C-N4
(DISENO_EXPERIMENTOS_NODOS_ABIERTOS_desde_2.5.5_CS.md, sección 7) · **Script:**
`cn4_delimitacion_fof.py` · **Herramienta que desbloqueó esto:** `leer_volcado_phantom.py`

Este es el primer experimento que corre desde que existe un lector de volcados binarios de
Phantom en Python. No se corrió ninguna simulación nueva — se leyeron volcados ya existentes de
`/Users/alexis/phantom_cs073/bateria_n2000/` (sólo lectura, nada se tocó ahí).

## La pregunta

En CS073, el criterio "amigos-de-amigos" (FoF) traza fronteras entre sumideros/grumos y el
resto del gas usando una longitud de enlace fija (`0.2 × espaciamiento medio`, la convención
cosmológica estándar b=0.2). Nunca se preguntó como hipótesis propia: **¿esa frontera cae donde
el gas de verdad tiene una discontinuidad de densidad, o cualquier longitud de enlace
razonable dibuja "algo parecido a un grupo" igual, incluso sobre un campo sin estructura
genuina?**

## Método

Para 4 corridas de la batería N=2000 (`ic_real`, `ic_null1`, `ic_null2`, `ic_null3`), todas en
el dump `cosmog_00500` (t=0.5, último paso disponible en las cuatro, con 7-8 sumideros ya
nacidos según `RESULTADO_bateria_ignicion_sumideros_N2000_CS.md`):

1. Se extrajeron las posiciones (x,y,z) de las partículas de **gas** (no sumideros) con
   `leer_volcado_phantom.leer_dump()`.
2. Se calculó, para cada partícula, la distancia a su **vecino más cercano** (k=1, vía
   `scipy.spatial.cKDTree`) — elegido porque el propio algoritmo FoF decide "amistad" par a
   par sobre exactamente esta distancia. Se repitió con k=2 y k=3 como chequeo de robustez.
3. Se buscó un "vacío" (gap) en la distribución de esas distancias con un detector automático:
   KDE gaussiana sobre log10(distancia), extremos locales, y una "profundidad de vacío" =
   altura del pico vecino más bajo / altura del valle (1.0 = sin dip; se reporta como "vacío
   notable" si supera 1.5 — umbral de reporte, no de veredicto).
4. **Control NULL #1 (físico):** las corridas `ic_null1..3` — mismo proceso físico, condición
   inicial de la malla causal barajada. Ya se sabía por el resultado previo de la batería que
   el NULL también forma sumideros (con ~1/3 de la masa de REAL), así que no se esperaba un
   NULL sin estructura alguna.
5. **Control NULL #2 (interno, más duro, pedido explícito del protocolo):** para `ic_real` y
   `ic_null1`, se permutaron independientemente los arrays x, y, z entre las partículas
   ("barajado de ejes") — preserva la distribución marginal 1D de cada coordenada pero destruye
   la correlación conjunta (x,y,z) que forma grumos 3D reales.
6. Se comparó la posición del vacío detectado con la longitud de enlace que usa hoy el
   proyecto: `cs073_cierre_holistico.py` (línea ~158) y `cs073_ignicion.py` (línea 109-110)
   usan `linking_length = 0.2 × espaciamiento` (b=0.2, "b_FoF=0.2 estándar" — comentario
   explícito en el docstring de `cs073_cierre_holistico.py`). Se repitió ese cálculo con el
   espaciamiento real de cada dump: `(volumen del bounding box / N_gas)^(1/3)`.

**Nota importante, no escondida:** la batería N2000 de Phantom en sí **no usa FoF** para decidir
qué es un sumidero — usa el criterio propio de Phantom (`rho_crit_cgs=1000, h_acc=0.3,
r_crit=0.6`, un umbral de densidad + radio de acreción). El FoF con b=0.2 vive en el pipeline
paralelo de N-body puro (`cs073_cierre_holistico.py` / `cs074_energia_holistica.py`), que es el
único criterio FoF explícito y citable que el proyecto usa hoy. La comparación de este
experimento es contra **ese** criterio — es la interpretación más directa de "el criterio FoF
que usa el proyecto", dado que la corrida de Phantom no lo usa en absoluto.

## Resultados

Todas las corridas tienen `N_gas` y espaciamiento medio casi idénticos (¬1774-1927 partículas,
espaciamiento 8.02-8.22), así que la longitud de enlace vigente del proyecto es prácticamente
la misma en las cuatro (~1.60-1.64).

| corrida | n_gas | linking_length proyecto (b=0.2) | valle NN1 detectado | b_efectivo (valle/espaciamiento) | profundidad NN1 | profundidad NN2 | profundidad NN3 |
|---|---|---|---|---|---|---|---|
| **ic_real** | 1774 | 1.644 | **2.426** | **0.295** | **2.35** | **4.06** | **5.26** |
| ic_null1 | 1922 | 1.607 | 1.841 | 0.229 | 1.35 | 1.48 | 1.79 |
| ic_null2 | 1927 | 1.605 | 1.903 | 0.237 | 1.51 | 1.66 | 1.79 |
| ic_null3 | 1923 | 1.606 | 1.866 | 0.232 | 1.22 | 1.36 | 1.64 |

Control de barajado de ejes (destruye la correlación 3D, preserva marginales 1D):

| corrida | profundidad NN1 nativa | profundidad NN1 barajada-ejes |
|---|---|---|
| ic_real | 2.35 | **1.18** (sin dip local, "no notable") |
| ic_null1 | 1.35 | **sin mínimo local detectable en absoluto** |

Ver `cn4_histogramas_nn.png` (las 4 corridas) y `cn4_resultados.json` (números crudos completos,
incluida la curva KDE punto por punto).

## Lectura de los números (sin declarar cierre)

- **REAL muestra un vacío mucho más profundo y mucho más consistente entre k=1,2,3** que
  cualquiera de los tres NULL de Phantom: profundidad 2.35→4.06→5.26 (crece monótonamente con
  k) contra 1.2-1.8 en los NULL (se mantiene marginal, cerca o por debajo del umbral de
  "notable"=1.5 en los tres). Visualmente (`cn4_histogramas_nn.png`), el valle de REAL cae casi
  a densidad cero entre dos modos limpios; el "valle" de los NULL es un hombro suave sobre una
  distribución que sigue básicamente unimodal-ancha.
- **El vacío de REAL depende de la estructura 3D genuina, no sólo de la distribución de
  densidades por eje:** al barajar x,y,z independientemente (preservando marginales), la
  profundidad de REAL cae de 2.35 a 1.18 — el vacío desaparece. Esto es evidencia directa a
  favor de que el vacío de REAL no es un artefacto de cómo se reparten las densidades en cada
  eje por separado.
- **NULL (Phantom) no es "sin estructura" — es "con menos estructura"**, coherente con el
  hallazgo previo de la batería (NULL también forma 7-8 sumideros, con ~1/3 de la masa de
  REAL): los tres NULL muestran el mismo hombro suave, en la misma posición aproximada
  (valle ≈1.84-1.90, muy consistente entre semillas), con profundidad que ronda justo el umbral
  de "notable" — ni un campo perfectamente liso ni un vacío tan limpio como el de REAL.
- **Comparación con la longitud de enlace vigente del proyecto:** el criterio actual
  (`b=0.2`, linking_length≈1.60-1.64) cae MUY cerca de donde está el hombro débil de los NULL
  (b_efectivo≈0.23-0.24), pero claramente por DEBAJO de donde cae el vacío genuino de REAL
  (b_efectivo≈0.295, ~48% más lejos que la longitud de enlace vigente). En otras palabras: el
  valor b=0.2 que el proyecto usa hoy está en el orden de magnitud correcto (no es un valor
  descabellado, ni 10× chico ni 10× grande) pero, según esta medición, se queda corto respecto
  de donde el propio gas de REAL muestra su discontinuidad de densidad más clara — tendería a
  agrupar de forma algo más conservadora/ajustada que lo que el sistema mismo sugiere.

## Qué NO dice este experimento (límites honestos)

- Sólo se usaron 3 de las 8 corridas NULL disponibles (protocolo pedía "más de una, no
  necesariamente las 8") y una sola semilla REAL — no hay aquí una estadística de z-score
  formal sobre "profundidad de vacío REAL vs NULL" con n≥8, sólo la comparación directa de
  números crudos en 4 corridas.
- El umbral "profundidad>1.5 = vacío notable" es una convención de reporte de este script, no
  un criterio estadístico validado externamente (no hay, que se sepa, un valor canónico en la
  literatura de FoF para "esto es un vacío real" vs "esto es ruido").
- El detector automático de vacíos requirió un ajuste durante el desarrollo (filtrar mínimos
  locales cuyos picos vecinos son ambos ruido de cola casi-cero) para no confundir un wiggle
  espurio con el valle genuino — documentado en el propio docstring de `detectar_vacio()` en
  `cn4_delimitacion_fof.py`, con el caso concreto (ic_null3) que lo motivó.
- No se corrió el control de barajado de ejes sobre ic_null2/ic_null3 (sólo sobre ic_real e
  ic_null1) — con 2 controles de barajado alcanza para el protocolo mínimo, pero no cubre las
  4 corridas.

## En una línea

REAL muestra un vacío en la distribución de distancias al vecino más cercano notablemente más
profundo, más consistente entre k=1/2/3, y dependiente de la estructura 3D genuina (desaparece
al barajar ejes) que los tres NULL de Phantom examinados, los cuales muestran un hombro débil y
consistente entre sí pero que ronda apenas el umbral de "notable". La longitud de enlace que usa
hoy el proyecto (b=0.2) cae en el orden de magnitud correcto pero más cerca del hombro débil de
los NULL que del vacío genuino de REAL. No se declara aquí si esto "confirma" o "refuta" C-N4 —
queda para adjudicación del director.
