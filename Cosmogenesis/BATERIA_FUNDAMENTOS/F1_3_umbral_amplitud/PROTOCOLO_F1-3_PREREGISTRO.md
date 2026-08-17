# PROTOCOLO F1-3 — Umbral de amplitud: ¿existe un ε mínimo bajo el cual nada persiste?

**Fecha de pre-registro:** 2026-07-24
**Ejecutor:** CC (agente F1-3, batería paralela de 24 experimentos)
**Código base (NO editado):** `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs074_rcruz.py`
**Documento autoritativo:** `BATERIA_FUNDAMENTOS_F1_a_F4_PARA_CC.md`, sección "F1-3"

Este protocolo se escribe y se congela ANTES de correr el barrido de producción
(la "corrida completa" descrita abajo). Trabajo previo de calibración/ingeniería
(medir D, medir tiempos de lavado, confirmar que las funciones importadas de
`cs074_rcruz.py` corren) se hizo para DISEÑAR el barrido — no se usó para ajustar
el criterio de PASS a un resultado ya visto. Ese trabajo de calibración SÍ reveló
un fenómeno relevante para el diseño (ver "Nota de diseño" abajo) y por eso el
protocolo incluye una verificación de precisión numérica que no estaba en el
enunciado original palabra por palabra, pero está permitida por la sección
"Verificación cruzada" del documento madre ("un segundo observable o método
distinto") — aquí se usa un segundo MÉTODO NUMÉRICO (doble precisión) además del
segundo observable. El criterio de PASS de las tres lecturas queda fijo desde
este documento y no se toca después de ver los números (T3).

---

## 1. Pregunta

¿Basta cualquier ε>0 por ínfimo que sea para que la diferencia persista contra el
NULL (lectura "scale-free", S>0 basta), o existe un ε* por debajo del cual la
persistencia desaparece (lectura "umbral")?

## 2. Nota de diseño (por qué se agrega el cross-check de precisión)

Calibración previa (no parte del resultado pre-registrado, solo diseño):
con la física exacta de `cs074_rcruz.py` (difusión vectorizada + corte de aristas
Bernoulli) en float64, se observó que a `pasos≈6095` (N=200, r=100) el observable
P cae a 0 de forma abrupta entre ε=1e-13 y ε=1e-11, mientras que reproduciendo el
MISMO cálculo en `numpy.longdouble` (precisión extendida, epsilon de máquina
~1e-19 vs ~2.2e-16 de float64) el corte desaparece y P sigue alto hasta ε=1e-15.
Esto es evidencia de que un "umbral" visto en float64 puede ser un ARTEFACTO DE
REDONDEO, no un piso físico. Por eso el protocolo pre-registra explícitamente una
tercera lectura ("artefacto numérico") y un cross-check de precisión, ANTES de
correr la producción completa (que se corre después de escribir este documento).

## 3. Método (observable primario)

Igual que F1-1/CS074: campo continuo 1D en anillo, perturbación inicial multi-modo
normalizada a std=1 y escalada por ε; difusión vectorizada solo por aristas vivas;
expansión = corte Bernoulli de aristas con probabilidad H por paso.
Observable primario: **P = corr(φ, roll(φ,1)) × var(φ)/var₀** (forma×magnitud,
idéntico a `cs074_rcruz.persistencia`, importado sin modificar).

**r fijo en régimen de congelamiento:** r = 100 (el valor más alto de
`R_TARGETS` en `cs074_rcruz.py`, firmemente r≫1). H = min(r·D, 1.0), con D medido
por el propio campo (`medir_D`), no impuesto.

**N = 200** (modo "producción" de `cs074_rcruz.py`, campo continuo ya validado).

**pasos:** calibrados UNA VEZ con `medir_pasos_lavado(N=200, eps=1e-3, semillas=16)`
del módulo base (tiempo medido para que P<0.05 a H=0, ×margen 1.15), y mantenidos
FIJOS en todo el barrido de ε (aísla el efecto de amplitud del efecto de duración).

## 4. Barrido (grid completo)

- **ε:** 24 puntos log-espaciados en [1e-15, 1e-1] (≥20 pedidos por el enunciado).
- **semillas:** 16 (seeds 2000..2015), independientes de las semillas de calibración.
- **ruido dinámico (perturbación DINÁMICA, no de semilla — T7):** forzamiento
  estocástico gaussiano aditivo, aplicado a φ DESPUÉS de cada paso de difusión,
  con amplitud σ_ruido barrida en {0.0 (línea base determinista), 1e-16, 1e-14,
  1e-12, 1e-8, 1e-4} — 6 niveles que cruzan deliberadamente por debajo y por
  encima del rango de ε probado, para separar "umbral por amplitud intrínseca"
  de "umbral inducido por ruido ambiental" (dos fenómenos distintos, ambos
  reportables).
- **NULL primario (pre-registrado por el enunciado F1-3):** ε=0 estricto, corrido
  sobre la MISMA grilla de semillas × ruido. Debe dar P≈0.
- **NULL secundario (estilo CS074, para comparación con el resto de la batería):**
  permutación de φ al final, para cada ε>0, mismo grid.
- **Cross-check de precisión (nuevo, ver §2):** subgrid en `numpy.longdouble`,
  mismo grid de ε (24 puntos) × 16 semillas, ruido=0 (línea base determinista),
  real + NULL ε=0. Compara si el eventual punto de quiebre de P(ε) se mueve al
  subir la precisión.
- **Segundo observable independiente (verificación cruzada F1-2, T2):**
  información mutua espacial entre las dos mitades del anillo (φ[0:N/2] vs
  φ[N/2:N], histograma 2D de 8×8 bins por corrida, MI en nats). Calculado sobre
  las MISMAS corridas que P (mismo estado φ final), no es una corrida aparte.
  Bajo NULL (permutación o ε=0) debe colapsar a ~0.

Total de corridas float64: 24 ε × 16 semillas × 6 ruidos × 2 (real+null-perm) =
4608, más el NULL ε=0 estricto sobre semillas×ruido (16×6=96, real solamente ya
que ε=0 no tiene "forma" que permutar de forma no trivial — se computa igual con
la misma función). Total longdouble: 24 ε × 16 semillas × 1 ruido(=0) × 2 = 768,
más NULL ε=0 estricto en longdouble (16).

## 5. Criterio de PASS — TRES LECTURAS pre-registradas (ninguna se descarta)

**(A) SCALE-FREE ("S>0 basta"):** P(ε) en float64 permanece alto (≫ NULL, es
decir separado del NULL por más de la dispersión entre semillas) en TODO el
rango resoluble por la precisión usada, sin quiebre estable — el "piso" visto,
si existe, se mueve o desaparece al subir a longdouble.

**(B) UMBRAL FÍSICO:** existe un ε* con P(ε<ε*)≈P(NULL) y P(ε>ε*)≫P(NULL), Y ese
ε* es ESTABLE (mismo orden de magnitud, no se mueve más de ~1 década) al pasar de
float64 a longdouble. Esto indicaría un piso real de la dinámica, no de la
aritmética.

**(C) ARTEFACTO NUMÉRICO (lectura mixta, la que anticipa la nota de diseño):**
existe un quiebre aparente en float64, PERO se mueve sustancialmente (varias
décadas) hacia ε menores al usar longdouble — el piso observado en precisión
estándar es de redondeo, no físico; la física subyacente es más cercana a (A)
dentro de lo que la precisión puede resolver, pero se reporta como "limitado por
precisión", no como "confirmado scale-free sin condiciones".

Las tres son hallazgo. Se reporta la curva P(ε) completa (float64 y longdouble),
la curva P_mi(ε), la dispersión entre semillas y entre niveles de ruido dinámico,
y cuál lectura sale con los números — sin autoadjudicar cuál "se quería".

## 6. NULL — condiciones de invalidación (T4)

Si el NULL ε=0 estricto NO da P≈0 (p.ej. porque el ruido dinámico por sí solo
genera persistencia espuria cuando σ_ruido es grande), se reporta explícitamente
como un hallazgo separado ("el ruido dinámico sustituye a ε"), y se anota qué
niveles de σ_ruido rompen el NULL — no se oculta ni se filtra del reporte.

## 7. Verificación cruzada — resumen de las tres vías (regla del documento madre)

(a) NULL ε=0 estricto (y NULL de permutación secundario).
(b) Segundo observable: información mutua espacial entre mitades (independiente
de la fórmula corr×var).
(c) Auditoría en disco: código (`F1_3_motor.py`), este protocolo, y JSON crudo
con cada fila (ε, semilla, ruido, precisión, P, P_mi, P_null, H, D, pasos,
frac_exp) — todo bajo prefijo `F1_3_` en esta carpeta, disponible para quien no
escribió el código.

## 8. Qué NO se hace

No se elige a mano ningún ε "bueno" (T1). No se cambia el criterio de PASS tras
ver la curva (T3). No se reporta solo un punto o una semilla (T7): se entrega la
curva P(ε) completa con 24 puntos × 16 semillas × 6 ruidos, y su contraparte en
longdouble. No se edita `cs074_rcruz.py`. No se autoadjudica el veredicto final;
eso corresponde a CS con la curva en mano.
