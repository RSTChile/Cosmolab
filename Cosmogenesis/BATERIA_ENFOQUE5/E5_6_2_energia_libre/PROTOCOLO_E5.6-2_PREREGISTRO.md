# PROTOCOLO E5.6-2 — "Exergía como energía libre: ¿se comporta como una energía libre real?"

**Congelado (pre-registro):** 2026-07-24T20:46 America/Santiago (UTC-4) = 2026-07-25T00:46:01Z
**Ejecutor:** CC (agente E5.6-2, batería Enfoque 5, corrida en paralelo con 29 agentes más)
**Base de código leída (NO editada):** `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs074_rcruz.py`
**Documento madre:** `BATERIA_ENFOQUE5_energia_exergia_entropia_30exp_PARA_CC.md`, sección "E5.6-2"
**Definiciones de X y S_ent alineadas con:** `BATERIA_ENFOQUE5/E5_2_2_anticorrelacion_X_S/E5_2_2_PROTOCOLO_PREREGISTRO.md`
(único preregistro hermano ya en disco con fórmulas exactas de X y S_ent al momento de escribir esto;
se reutilizan literalmente para que la comparación entre experimentos del Enfoque 5 sea consistente).

Este documento se escribe y congela ANTES de tocar el motor. Cualquier desviación respecto de lo
aquí escrito se reporta como desviación explícita, no se edita retroactivamente (T3).

---

## 1. Pregunta

La exergía X medida en esta batería, ¿se comporta como una energía libre termodinámica real,
es decir, obedece X ≈ E − T·S_ent cuando E, T y S_ent se miden CADA UNA por su cuenta, con
fórmulas independientes entre sí? Esto es un test de COHERENCIA de la construcción (¿las tres
piezas encajan como en termodinámica?), no una definición circular: ninguna de las tres se define
en función de las otras dos.

## 2. Modelo (heredado de cs074_rcruz.py, motor propio bajo mi prefijo, NO se edita la base)

Campo escalar φ en un anillo de N=200 sitios (misma física que CS074-rcruz / E5.2-2):
- Fondo φ=1 + perturbación ε·(suma de 5 armónicos con fase aleatoria, normalizada a desviación
  estándar 1) — `campo_inicial()`, sin editar.
- **Difusión:** relajación local hacia el promedio de vecinos, SOLO por aristas vivas, idéntica
  a `paso_difusion()` de la base: `nuevo = φ + 0.5·(media_vecinos − φ)`.
- **Expansión:** cada arista viva se corta con probabilidad de Bernoulli H por paso, idéntica a
  `paso_expansion()`. H≥1 corta todas; H=0 no corta ninguna.
- **D** = fracción de contraste (desviación estándar) borrada en UN paso de difusión pura (H=0),
  MEDIDA del propio campo, igual que `medir_D()` de la base.
- **r = H/D**, razón expansión/difusión. H se fija como H = min(r_target·D, 1.0): D se mide
  primero, H emerge de esa medida (no se impone H a mano).
- **Ruido dinámico (T7):** en CADA paso de evolución se suma al campo ruido gaussiano de amplitud
  NOISE_REL·ε (NOISE_REL = 0.02, misma constante que usó E5.1-1, declarada aquí y NO ajustada
  después de ver resultados). Con ε=0 el ruido es exactamente 0 (preserva el control puro).

## 3. Axiomas declarados (E1/E2, NO física real — igual que el resto de la batería)

- **E1:** el presupuesto Σφ se declara conservado por la difusión (operador lineal de promedio
  local). Se AUDITA, no se fuerza: se reporta la deriva relativa |Σφ_fin−Σφ_ini|/|Σφ_ini| por
  celda. No se renormaliza el campo.
- **E2:** la expansión (cortar aristas) no crea energía; aísla regiones y congela gradientes que
  la difusión, de otro modo, borraría — "enfriamiento adiabático" declarado como marco
  interpretativo de por qué H>0 debería preservar X y, se hipotetiza aquí, bajar T efectiva.

## 4. Definiciones EXACTAS e INDEPENDIENTES de X, E, T, S_ent (el corazón de este experimento)

Las cuatro cantidades se calculan sobre el estado FINAL de cada corrida (φ_final, activo_final)
por fórmulas algebraicamente distintas — ninguna usa la fórmula de otra como insumo.

### X_final — Exergía (desviación cuadrática de un equilibrio FIJO externo)
```
X_final = (1/N) · Σ_i (φ_i − 1)²
```
φ_eq=1 es el "estado muerto" de referencia (el fondo uniforme antes de la perturbación).
Fórmula IDÉNTICA a la de E5.2-2 (consistencia entre agentes del mismo Enfoque).

### E_final — Energía total del campo (segundo momento CRUDO, sin restar referencia)
```
E_final = (1/N) · Σ_i φ_i²
```
Es la "energía" total contenida en el campo (fondo + fluctuación), medida como el momento
cuadrático respecto de CERO (no respecto de 1). Es una cantidad DISTINTA de X: X está centrada
en el equilibrio (φ_eq=1), E está centrada en el origen. Identidad algebraica de verificación
(no es una definición, es una consecuencia que se puede chequear en los datos, ver §7):
`E_final − X_final = 2·mean(φ_final) − 1`. Si E1 se cumpliera exactamente (mean(φ)≡1 siempre),
esa diferencia sería constante ≡1; el experimento mide si el axioma E1 se sostiene con expansión.

### S_ent_final — Entropía de Shannon de la densidad espacial de energía
```
p_i = φ_i² / Σ_j φ_j²          S_ent_final = − Σ_i p_i · ln(p_i)
```
Fórmula IDÉNTICA a la de E5.2-2. Campo uniforme (equilibrio) → p_i uniforme → S_ent → ln(N)
(máximo). Campo concentrado/estructurado → S_ent bajo.

### T_efectiva — Temperatura efectiva MEDIDA de la expansión/dinámica (NO impuesta)
```
T_efectiva = fracción de contraste (std) que UN paso adicional de difusión pura borraría,
             aplicado sobre el estado FINAL (φ_final, activo_final) tal como quedó la
             topología tras los cortes de expansión:

    c0 = std(φ_final)
    φ_prueba = paso_difusion(φ_final, activo_final)     # una sola sonda, no se usa para nada más
    c1 = std(φ_prueba)
    T_efectiva = max(0, (c0 − c1) / c0)
```
Interpretación: mide el acoplamiento térmico REMANENTE del sistema — cuánta mezcla/relajación
todavía es capaz de producir un paso de difusión sobre la topología que dejó la expansión. Cuantas
más aristas cortó la expansión (r/H mayor), menos vecinos activos quedan y menos contraste puede
borrar un paso de sonda ⇒ T_efectiva baja con más expansión (enfriamiento adiabático, consistente
con E2, pero MEDIDO, no supuesto). A H=0 (sin expansión) T_efectiva = D (la topología está intacta,
la sonda mide lo mismo que `medir_D`). Esta es la razón por la que el barrido "T efectiva (de la
expansión)" del documento madre se implementa barriendo r (que fija H) y reportando el T_efectiva
que ESO produce — nunca se elige un valor de T directamente.

**Por qué esto NO es circular (T2):** T_efectiva se calcula con `paso_difusion` (mecánica de
propagación), no con la fórmula de X (cuadrática desde φ_eq=1) ni con la de S_ent (Shannon sobre
p_i=φ²/Σφ²). Es una sonda dinámica adicional, independiente de las otras tres cantidades.

### El observable comparado — F_pred vs X
```
F_pred_final = E_final − T_efectiva · S_ent_final
PASS si X_final ≈ F_pred_final dentro de tolerancia (§6).
```

## 5. Barrido (sobredimensionado, regla del director)

| Eje | Rango | Puntos |
|---|---|---|
| r = H/D (genera T_efectiva, NO se impone T directo) | {0} ∪ logspace(1e-3, 1e3) | 26 (r=0 control + 25 pts log, 6 décadas) |
| ε | {0, 1e-12, 1e-9, 1e-6, 1e-3, 1e-2, 0.1, 0.3, 1.0} | 9 (0 a 1, 12 décadas + control 0) |
| semillas | 0..15 | 16 (≥12 pedido por el doc madre, se sobredimensiona a 16) |
| ruido dinámico | NOISE_REL=0.02·ε, cada paso (T7) | fijo, declarado |
| N | 200 (igual que modo "produccion" de la base / E5.1-1 / E5.2-2) | — |
| pasos | calibrado UNA vez por lavado (P<0.05, ε=1e-2, mediana×1.15), igual método que la base | — |

Total combinaciones (r,ε) = 26×9 = 234. Cada combinación × 16 semillas = **3744 corridas**
(sin pareja NULL — ver §6, este experimento no lleva NULL por diseño del doc madre).

## 6. NULL

El documento madre marca explícitamente **NULL: —** para E5.6-2 (es un experimento de
caracterización/coherencia, no de detección de estructura vs. ruido). No se inventa un NULL no
pedido. Se compensa con las dos verificaciones independientes exigidas por la regla de ejecución
#4 ("su NULL + segundo observable/método + auditoría en disco"):

1. **Verificación en espacio-X:** comparación directa X_final vs F_pred_final = E_final −
   T_efectiva·S_ent_final (arriba).
2. **Segundo método — verificación en espacio-T:** se despeja de la identidad qué T HARÍA que la
   igualdad fuera exacta, `T_implied = (E_final − X_final) / S_ent_final` (cuando S_ent_final>0),
   y se compara T_implied contra T_efectiva MEDIDO (correlación + razón T_implied/T_efectiva).
   T_implied NO se usa para nada más que esta comparación — es un espejo del mismo test, no una
   definición nueva ni un ajuste.
3. **Auditoría en disco:** JSON crudo con las 3744 filas + deriva de E1 por celda, disponible para
   revisión por quien no escribió este motor.

## 7. Umbral de PASS (congelado ANTES de correr — T1, T3)

No existe expectativa previa de a qué distancia numérica caerá X de F_pred (esa es la pregunta).
Se pre-registran umbrales generosos y declarados, NO ajustados después de ver los datos:

- **Coherencia fuerte (PASS):** correlación de Pearson entre {X_final} y {F_pred_final} a través
  de TODA la grilla (r,ε) con al menos 16 semillas agregadas por celda (media) es **> 0.9**, Y el
  error relativo mediano |X_final − F_pred_final| / (|X_final| + |F_pred_final| + 1e−9) sobre la
  grilla es **< 0.20** (20%, tolerancia deliberadamente amplia, "sobredimensionada" en el sentido
  inverso: generosa, no ajustada al target).
- **Coherencia parcial:** correlación > 0.9 pero error relativo mediano ≥ 0.20 (la forma funcional
  coincide, la escala no) — se reporta tal cual, sin forzar un veredicto binario.
- **Negativo honesto:** correlación ≤ 0.9 — se reporta que X NO se comporta como energía libre
  con estas definiciones; es hallazgo, no fracaso (nota final del doc madre).
- Celdas degeneradas (ε=0, o S_ent_final≈0/indefinido) se excluyen del agregado y se reportan
  aparte.
- Se reporta la curva ENTERA (T5): X_final(r,ε) y F_pred_final(r,ε) por cada punto de la grilla,
  no solo el agregado.

## 8. Auditoría T6 (conservación de E1, cada paso agregado por corrida)

Por cada corrida se registra Σφ_inicial y Σφ_final; se reporta la deriva relativa. No se
renormaliza el campo para forzar conservación. Si la deriva es grande a r alto, es exactamente lo
que predice la nota del §4 (E_final−X_final deja de ser ≈1) — se reporta explícitamente, no se
oculta ni se "corrige".

## 9. Qué se entrega crudo a CS

- Tabla completa por (r, ε): X_final, E_final, T_efectiva, S_ent_final, F_pred_final,
  T_implied, correlación y error relativo agregados sobre 16 semillas (media, mediana, std).
- Dispersión completa entre semillas (no solo el promedio) y por la perturbación dinámica.
- Auditoría de deriva E1 por celda.
- Veredicto sin suavizar según §7 — sin adjudicar cierre de arco (nota permanente del director).

## 10. Archivos

- Este pre-registro: `PROTOCOLO_E5.6-2_PREREGISTRO.md` (congelado antes de tocar el motor).
- Motor: `E5_6_2_engine.py` (importa `cs074_rcruz.py` sin editarlo; escrito DESPUÉS de este doc).
- Resultado crudo: `E5_6_2_resultado_crudo.json`.

## 11. Trampas explícitamente evitadas

- T0: N, pasos vienen del modelo base y de calibración medida, no de ajustar-para-que-cruce.
- T1: NOISE_REL=0.02, umbrales de PASS (corr>0.9, error rel<0.20) declarados aquí, antes de correr.
- T2: T_efectiva se mide con una sonda de difusión (mecánica), no con las fórmulas de X ni S_ent;
  X, E, S_ent usan fórmulas algebraicamente distintas entre sí.
- T3: si algo falla o sorprende, se reporta tal cual — este archivo no se edita después de correr.
- T4: no aplica (NULL: — por diseño del doc madre); compensado con doble método (§6).
- T5: curva completa (r,ε) entregada, no gate binario.
- T6: deriva de E1 auditada por celda, reportada aunque rompa la interpretación limpia.
- T7: ruido dinámico NOISE_REL·ε cada paso, además de 16 semillas.

No se corre nada del motor hasta que este archivo esté guardado en disco.

**Firmado (pre-registro, antes de correr):** agente E5.6-2, 2026-07-25T00:46:01Z (UTC).
