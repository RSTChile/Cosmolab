# PROTOCOLO E5.6-1 — PRE-REGISTRO
## "Doble medida: exergía termodinámica vs informacional, mismo barrido"

**Fecha/hora de pre-registro:** 2026-07-24 20:42:44 -04
**Ejecutor:** CC (agente E5.6-1, batería Enfoque 5, corrida en paralelo con 29 agentes más)
**Base de motor físico (leída, NO editada):** `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs074_rcruz.py`
**Spec autoritativa:** `BATERIA_ENFOQUE5_energia_exergia_entropia_30exp_PARA_CC.md`, sección E5.6-1 (Tema 6)

Este documento se escribe y se congela **ANTES** de correr el motor (T3). No se edita
después de ver resultados. Cualquier desviación de lo aquí escrito se declara aparte, no
se retoca este archivo.

---

## 1. Pregunta

¿Dos definiciones de exergía, construidas por vías **independientes** (ninguna definida en
términos de la otra — anti-T2), dan la misma respuesta en el mismo barrido de ε (amplitud
de la mancha inicial) × r (razón expansión/difusión)?

---

## 2. El campo físico (heredado sin editar de `cs074_rcruz.py`)

- Anillo 1D de N sitios, campo continuo φ.
- `campo_inicial(N, eps, rng)`: fondo uniforme φ=1 + perturbación εÂ·(combinación de los
  primeros 5 modos de Fourier, fases aleatorias, normalizada a std=1). ε=0 ⇒ φ uniforme
  (equilibrio exacto).
- `paso_difusion`: promedio con vecinos vivos (difusión, solo por aristas activas).
- `paso_expansion(activo, H, rng)`: cada arista viva se corta con probabilidad H
  (Bernoulli) — expansión que aísla.
- D = fracción de contraste borrada en un paso de difusión pura (H=0), **medida**, no
  puesta a mano (`medir_D`).
- r_target es el eje pre-registrado; H = min(r_target·D, 1.0) — H emerge de D medido, r es
  la razón interna real (T1: nada puesto a mano salvo el propio r_target del grid).
- pasos: calibrados por `medir_pasos_lavado(N, eps=1e-3, semillas)` igual que el modo
  `produccion` del motor base (tiempo medido, en pasos, para que a H=0 la persistencia caiga
  bajo P_LAVADO=0.05, con margen 1.15×) — **no elegido a mano**, es el mismo criterio ya
  validado en CS074-rcruz.
- NULL = **barajado espacial** (`rng.permutation(phi)`) aplicado al φ final, exactamente el
  mismo NULL que usa el motor base para E5.1-1. Se reutiliza sin modificar su semántica.

Este experimento **reutiliza el motor sin tocarlo** (import directo de
`campo_inicial`, `paso_difusion`, `paso_expansion`, `medir_D`, `medir_pasos_lavado`,
`persistencia`, `R_TARGETS` desde `cs074_rcruz.py`) y solo añade la segunda medida
(X_info) y la comparación.

---

## 3. Las dos definiciones — EXACTAS, congeladas antes de correr

### 3.1 X_termo — exergía termodinámica ("tipo el X de E5.1-1")

Es la función `persistencia(phi, contraste0)` ya validada en `cs074_rcruz.py`, sin
modificar una coma:

```
c = corr(φ, roll(φ, 1))          # coherencia con el vecino inmediato (correlación espacial local, lag-1)
c = max(0, c)                     # solo coherencia positiva cuenta como "capaz de trabajo"
v = Var(φ) / Var(φ_inicial)       # fracción de la varianza inicial que sobrevive
X_termo = c · v
```

**Lectura física:** v mide cuánta amplitud de desviación-del-equilibrio-uniforme
sobrevive (energía potencial de la desviación, ~exergía clásica ∝ (Δφ)²); c exige que esa
desviación sea *coherente* con el vecino inmediato — ruido descorrelacionado no puede
mover un motor real (un gradiente al azar no impulsa nada de forma sostenida). El
producto es la misma cantidad que ya ancla el observable de E5.1-1 en este código base.
Rango [0,1].

### 3.2 X_info — exergía informacional (estructura espacial vía entropía espectral)

```
Φ = φ − mean(φ)                          # se remueve el nivel DC (la media no es estructura)
F = FFT_real(Φ)                          # transformada de Fourier real
P_k = |F_k|²  para k = 1 … N/2           # potencia por modo, EXCLUYENDO k=0 (DC)
p_k = P_k / Σ P_k                        # se normaliza a distribución de probabilidad
                                          # (Σp_k=1) — esto elimina la ESCALA/amplitud,
                                          # deja solo la FORMA del espectro
H_spec = − Σ p_k · log(p_k)              # entropía de Shannon del espectro normalizado
H_max = log(N/2)                          # entropía de un espectro plano (ruido blanco)
X_info = 1 − H_spec / H_max               # 1 = toda la potencia en un solo modo (máx. estructura)
                                          # 0 = espectro plano (sin estructura, ruido blanco)
```

Si ΣP_k = 0 (φ exactamente constante, ej. ε=0 o difusión total), se define X_info = 0 por
convención (sin desviación no hay espectro que medir; consistente con "sin diferencia no
hay estructura").

**Lectura física:** X_info mide qué tan *concentrada* está la energía espacial del campo
en pocos modos de Fourier — es la medida clásica de "cuán lejos de ruido blanco" está la
distribución espacial, en bits de información (Shannon), sin usar en ningún paso la
amplitud/varianza real ni la autocorrelación de vecino inmediato.

### 3.3 Por qué son independientes (anti-T2)

| | X_termo | X_info |
|---|---|---|
| Dominio de cómputo | espacio real (vecino inmediato, lag-1) | espacio de frecuencias (FFT completa) |
| Usa la AMPLITUD/varianza | sí (factor v) | no — normalizada a Σp_k=1, la escala se cancela |
| Usa la FORMA/distribución espacial | no (solo coherencia local, 1 vecino) | sí (espectro completo, todos los modos) |
| Fórmula de una definida en función de la salida de la otra | no | no |
| Convención en ε=0 o colapso total | 0 | 0 (por separado, no por referencia cruzada) |

Ninguna línea de código de X_info llama a `persistencia()` ni usa su salida; ninguna línea
de X_termo usa FFT ni entropía. Comparten el mismo φ de entrada (inevitable: miden el
mismo campo) pero por caminos matemáticos disjuntos — correlación de vecino-inmediato +
razón de varianzas (real, local, con escala) vs. entropía de Shannon del espectro
normalizado (frecuencial, global, sin escala). Se descartó explícitamente la opción más
obvia de X_info ("entropía de Shannon del histograma de VALORES de φ") porque esa medida
es invariante ante permutación espacial (no le importa el orden, solo el conjunto de
valores) — **no cae bajo el NULL barajado**, violando T4. La versión espectral sí cae
(barajar destruye la concentración de modos bajos y aplana el espectro), lo cual se
verifica en el propio corrido (sección 6).

---

## 4. Barrido (sobredimensionado — regla de oro)

- **ε** (amplitud de la mancha): {0.0, 1e-9, 1e-6, 1e-3, 1e-2, 0.1, 0.5, 1.0} — mismo grid
  que el modo `produccion` del motor base (8 puntos, cubre 9 décadas + los extremos 0 y 1).
- **r** (razón H/D): grid logarítmico de 25 puntos entre 1e-3 y 1e3 (6 décadas, cruza r=1)
  **más** r=0 explícito (control de lavado puro) = 26 puntos. Esto es el rango extremo
  exigido por la regla de oro del director y por el propio E5.1-1 ("mismo barrido de ε y
  r" pide la spec de E5.6-1).
- **Semillas:** 16 por combinación (ε, r), tanto para la corrida real como su NULL pareado
  (misma semilla en real y NULL, igual que el motor base).
- **N** = 200 (igual que el modo `produccion` del motor base).
- **pasos** = fijo, calibrado una vez con `medir_pasos_lavado(N=200, eps=1e-3,
  semillas=16)` (igual criterio que el motor base, no puesto a mano).
- Total de corridas: 8 ε × 26 r × 16 semillas × 2 (real+NULL) = 6656 corridas del campo.

---

## 5. Observable y análisis

- Observable primario: **correlación de Pearson entre X_termo y X_info**, calculada:
  (a) global, agrupando todos los puntos (ε, r, semilla) de las corridas REALES;
  (b) por-ε, como curva vs. r (para no esconder heterogeneidad entre regímenes);
  (c) la misma correlación sobre las corridas NULL (debe degradarse).
- Se reporta también la dispersión entre semillas (std) de X_termo y X_info en cada (ε,r),
  y la magnitud media real vs. NULL de cada medida por separado (para verificar T4: "el
  NULL debe morder" — ambas medidas deben caer, no solo una).

---

## 6. NULL

Barajado espacial (`rng.permutation`) del φ final, aplicado tras la evolución completa —
mismo NULL que ya usa el motor base para el observable de E5.1-1. Ambas medidas
(X_termo y X_info) se calculan también sobre el φ barajado. **Pre-registro de lectura: si
X_termo_NULL no cae (permutación no destruye c, el término de coherencia), o si X_info_NULL
no sube hacia H_max (el espectro no se aplana), el NULL no mordió esa medida y se reporta
como tal — no se oculta.**

---

## 7. Criterio PASS/FALLA (congelado antes de correr)

- **PASS:** corr(X_termo, X_info) > 0.9 en el pool global de corridas REALES, Y ambas caen
  bajo NULL (T4).
- **Discrepancia (corr ≤ 0.9 pero ambas caen bajo NULL):** no es falla del experimento —
  es la medida de robustez que la propia spec pide reportar sin esconder ("la discrepancia
  es medida de robustez, no se esconde"). Se reporta con la curva completa por-ε y se
  señalan los regímenes de r donde diverge.
- **Negativo fuerte:** si una medida no cae bajo NULL (T4 no se cumple) o si ε=0 no da
  X_termo=X_info=0 en ambas — se reporta como hallazgo negativo, no se ajusta el código
  para forzar el PASS (T3: código congelado tras este pre-registro).

---

## 8. Trampas verificadas contra este diseño (T0-T7)

- T0: nada discreto puesto a mano (r_target es el único grid explícito, igual que E5.1-1).
- T1: ningún número calibrado hacia el resultado esperado; D, pasos, y ambas X emergen del
  campo.
- T2: verificado en §3.3 — caminos matemáticos disjuntos, ninguna definida vía la otra.
- T3: este archivo se congela antes de ejecutar el motor.
- T4: NULL barajado se aplica y se verifica que muerda ambas medidas (§6, reportado en
  crudo, no asumido).
- T5: se reporta curva entera (por ε, por r), no un gate binario.
- T6: no aplica un E_total explícito en este motor (el motor de campo φ no lleva
  contabilidad de energía tipo Tema 2/3); se declara este límite de alcance explícitamente
  — E5.6-1 no pide verificar conservación de E_total, pide correlacionar dos observables.
- T7: barrido + 16 semillas + NULL barajado (perturbación estructural, no solo semilla).

---

## 9. Archivos que este experimento va a producir

- `E5_6_1_PROTOCOLO_PREREGISTRO.md` (este archivo, congelado antes de correr).
- `E5_6_1_motor.py` (motor: importa el campo de `cs074_rcruz.py` sin editarlo, añade
  X_info y el análisis de correlación).
- `E5_6_1_resultado.json` (salida cruda: todas las filas ε×r×semilla, medias, std,
  correlaciones globales y por-ε, real y NULL).

No se edita ningún archivo existente. Prefijo `E5_6_1_` en todo archivo nuevo, dentro de
`BATERIA_ENFOQUE5/E5_6_1_doble_medida/`.
