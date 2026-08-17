# E5.5-1 · Barrido fino de ε→0: curvas E, X, S_ent en el límite

**Pre-registro fechado (UTC):** 2026-07-25T00:43:36Z, ANTES de correr el motor.
Regla T3: si algo falla, se reporta — no se edita esto después.

**Base física:** motor de `cs074_rcruz.py` (NO editado — se importa como módulo:
`paso_difusion`, `campo_inicial`, `medir_pasos_lavado`, `P_LAVADO`). Mismo campo
continuo φ en anillo de N nodos, difusión solo por aristas vivas.

**Reutilización de definiciones (regla del director / instrucción de la tarea):**
`E5_2_1_balance_deriva/` está VACÍO en disco al momento de escribir esto (el agente
E5.2-1 aún no ha producido su motor ni definiciones) — no hay nada que reutilizar de
ahí. `E5_2_2_anticorrelacion_X_S/` SÍ tiene protocolo y motor en disco
(`E5_2_2_PROTOCOLO_PREREGISTRO.md`, `E5_2_2_motor.py`, fechado 2026-07-24T20:37:17Z).
Se reutilizan sus definiciones de **X** y **S_ent** VERBATIM (mismas fórmulas, mismo
código, ver §1). La definición de **E** (energía total) no estaba definida por ningún
agente hermano en disco al momento de escribir esto, así que se define aquí de forma
que sea CONSISTENTE con la construcción de S_ent heredada (que ya usa φ² como
"densidad de energía" para normalizar p_i) — no se inventa una tercera convención
desacoplada. Se declara explícitamente como propia.

---

## 1. Definiciones exactas (ANTES de correr)

### X(t) — Exergía · **HEREDADA de E5.2-2, verbatim**

```
X(t) = (1/N) · Σ_i (φ_i(t) − 1)²
```

φ_eq = 1 es el "estado muerto" de referencia (el fondo uniforme con el que arranca
`campo_inicial` antes de sumar ε·pert). Momento cuadrático de la desviación respecto
al equilibrio fijo.

### S_ent(t) — Entropía (Shannon espacial) · **HEREDADA de E5.2-2, verbatim**

```
p_i(t) = φ_i(t)² / Σ_j φ_j(t)²
S_ent(t) = − Σ_i p_i(t) · ln(p_i(t))
```

Campo uniforme (equilibrio) → p_i uniforme → S_ent → ln(N) (MÁXIMO). Campo
estructurado/concentrado → S_ent bajo.

### E(t) — Energía total · **PROPIA de este experimento, declarada**

```
E(t) = Σ_i φ_i(t)²
```

Justificación de consistencia (no es un tercer criterio inventado suelto): es
EXACTAMENTE la constante de normalización que ya usa S_ent heredada de E5.2-2
(p_i = φ_i² / E(t)). Definir E así hace que S_ent = Shannon de {φ_i²/E} sea la
entropía de la distribución de "energía" tal como E la define — cierre interno del
trío (E, X, S_ent), no una unión de piezas inconexas.

**Sobre el axioma E1 (conservación):** el motor base (`cs074_rcruz.py`,
`paso_difusion`) NO impone conservación de Σφ² por construcción — es un operador de
suavizado (difusión) que dispersa varianza, así que Σφ² puede DECAER con el tiempo
en la dinámica cruda (E1 "on" requeriría normalización explícita, que es objeto de
E5.3-4, no de este experimento). Aquí E1 se trata como HIPÓTESIS A VERIFICAR, no como
verdad impuesta (T6): se mide la deriva relativa |E(final)−E(inicial)|/E(inicial) en
cada corrida y se reporta cruda. La predicción pre-registrada es que, en el límite
ε→0, φ→1 exactamente en todo punto y en todo paso (el campo uniforme es punto fijo
exacto de `paso_difusion`: media_i=φ_i ⇒ Δφ_i=0), por lo que la deriva de E debe
→0 idénticamente cuando ε→0 — y crecer (probablemente ∝ε²) al alejarse de ese
límite. Esto es lo que arroja la curva E(ε), no un valor fijado a mano.

### Independencia (anti-T2)

X es suma de cuadrados respecto a una constante externa fija (φ_eq=1). S_ent es
entropía de Shannon de una distribución normalizada de φ². E es la suma cruda de φ²
(sin restar φ_eq, sin normalizar). Las tres se calculan por vías algebraicas
distintas; ninguna es juez de sí misma — el juez del PASS es la comparación directa
de las curvas E(ε), X(ε), S_ent(ε) contra el criterio pre-registrado en §4, no una
sola cantidad derivada de las otras dos.

---

## 2. Barrido (MUY FINO cerca de 0, regla de oro de esta tarea)

- **N = 200** (misma escala que `cs074_rcruz.py modo=produccion` y que E5.2-2).
- **ε** ∈ **24 puntos**: {0} ∪ 23 puntos log-espaciados en [1e-6, 1e-2] (6 décadas de
  resolución fina justo en la región pedida, ε=0 estricto incluido aparte porque
  log(0) no existe). Cumple "≥20 pts, incluye 0 estricto" e intencionalmente NO se
  concentra el barrido lejos de 0 — el rango pedido por la spec ES [0…1e-2], y aquí
  se resuelve fino en TODA esa ventana, no solo en el extremo.
- **r = 0 (H = 0), dinámica de difusión pura, sin expansión** — elección de diseño
  declarada: TEMA 5 caracteriza la "muerte térmica" como el estado de equilibrio
  puro por difusión (el mecanismo que borra estructura); la competencia
  difusión/expansión (r≠0) es objeto de TEMA 1 y de E5.5-2/E5.5-3, no de este
  experimento, cuya spec solo pide barrer ε × semillas. Fijar r=0 aísla la variable
  bajo prueba (ε) sin introducir un segundo eje no pedido.
- **Semillas:** 16 por ε (seeds base 6000..6015 — rango distinto del 5000..5015 de
  E5.2-2 para no colisionar en ningún cache/artefacto compartido).
- **Pasos por corrida:** calibrados (no a mano), mismo criterio que
  `cs074_rcruz.py modo=produccion` y E5.2-2: se mide el lavado
  (`medir_pasos_lavado`, P_LAVADO=0.05) a ε=1e-2 (el extremo superior de ESTE
  barrido, representativo) y se usa `pasos = ceil(mediana_lavado × 1.15)` fijo para
  todo el barrido. Se registra el valor calibrado en el resultado.
- **Trayectoria temporal:** además de medir (E,X,S_ent) en t=0 (inicial) y t=pasos
  (final, aprox. equilibrio), se registra la trayectoria completa
  (E(t),X(t),S_ent(t) en CADA paso) para las 16 semillas de 4 valores de ε de
  referencia (ε=0, ε mínimo positivo del grid, ε mediano del grid, ε=1e-2 máximo)
  — para poder inspeccionar la curva completa en el tiempo, no solo el punto final
  (T5: curva entera, no gate binario).
- **Grid total:** 24 ε × 16 semillas = 384 corridas para (E,X,S_ent) en t=0/t=final;
  + 4 ε × 16 semillas con trayectoria completa por paso.

## 3. NULL

Este experimento es de CARACTERIZACIÓN pura (igual que E5.1-2, E5.2-1, E5.2-4,
E5.2-5, E5.5-2, E5.5-4, E5.5-5 en la spec: "NULL: —"), no de detección de señal
contra ruido — no aplica NULL formal. Lo que se verifica en su lugar (guardia
anti-artefacto): en ε=0 exacto, φ(t)=1 en todo t por construcción algebraica de
`paso_difusion` sobre un campo ya uniforme — esto se comprueba numéricamente (no se
asume) corriendo esa celda igual que las demás, sin atajos de código.

## 4. Juez y criterio de PASS (congelado antes de correr)

Curvas E(ε), X(ε), S_ent(ε) — media y std entre 16 semillas, en t=0 y en t=final,
para los 24 puntos de ε.

- **PASS:** al evaluar la curva en ε→0 (los puntos ε del grid más cercanos a 0,
  incluyendo ε=0 estricto):
  - E(ε) ≈ constante ≈ N (=200) tanto en t=0 como en t=final, con deriva relativa
    |E_final−E_inicial|/E_inicial → 0 monótonamente conforme ε→0 (no un valor
    puntual, la CURVA completa de la deriva vs ε).
  - X(ε) → 0 en t=0 (trivial, por construcción de campo_inicial) Y en t=final (tras
    difusión) — se reporta si la difusión acelera o no esa caída respecto a t=0.
  - S_ent(ε) → ln(N) (máximo) en t=final, y su distancia a ln(N) DECRECE
    monótonamente conforme ε→0.
- **Negativo honesto:** si alguna de las tres curvas no converge a lo esperado, o si
  la convergencia no es monótona, o si aparece una discontinuidad/artefacto
  numérico cerca de ε=0 (p.ej. deriva de E que NO decae con ε), se reporta tal cual,
  sin suavizar ni ajustar el motor.
- Ningún coeficiente se mueve para acercar el resultado al criterio (regla del
  director, T1).

## 5. Qué se entrega crudo a CS

- Tabla completa E(ε), X(ε), S_ent(ε) en t=0 y t=final — media/std entre 16
  semillas, los 24 puntos de ε.
- Deriva relativa de E por ε (curva entera).
- 4 trayectorias completas (E(t),X(t),S_ent(t)) para inspección visual del
  acercamiento al límite en el tiempo.
- Veredicto sin suavizar.

## 6. Archivos

- Motor: `E5_5_1_motor.py` (importa `cs074_rcruz.py` sin editarlo).
- Resultados crudos: `E5_5_1_resultados.json`.
- Este pre-registro: `PROTOCOLO_E5.5-1_PREREGISTRO.md`.

**Firmado (pre-registro, antes de correr):** agente E5.5-1, 2026-07-25T00:43:36Z.
