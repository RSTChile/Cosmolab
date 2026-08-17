# E5.2-5 — Robustez del balance de energía a la resolución (Δt, N)

**Pre-registro fechado.** Fecha/hora de escritura (UTC): 2026-07-24T20:40Z (antes de
correr el motor — T3: juez congelado antes de correr; NO se edita este archivo tras ver
resultados).

**Pregunta (spec, TEMA 2):** ¿la conservación del presupuesto de energía observada en el
motor de campo continuo (difusión + corte de aristas por expansión, cs074_rcruz.py) es un
**artefacto numérico** (la deriva → 0 al refinar Δt y N) o **algo más profundo** (la
deriva no se achica, o crece, al refinar → problema real de diseño, no de precisión)?

**Estado de E5.2-1 al momento de escribir esto:** `BATERIA_ENFOQUE5/E5_2_1_balance_deriva/`
existe pero está **vacío** (verificado con `find` justo antes de este pre-registro). No hay
definición de E_total en disco que reutilizar. Se define aquí, **con el mismo principio**
declarado en el documento madre (sección 0, axiomas E1/E2), no con un principio distinto.

---

## 1. Modelo base (NO editado)

`Cosmogenesis/cs074_rcruz.py` — campo continuo φ en anillo de N nodos; difusión ponderada
solo por aristas vivas (`paso_difusion`); expansión = corte Bernoulli(H) por arista viva
por paso (`paso_expansion`); D = fracción de contraste borrado en un paso de difusión pura,
**medido, nunca impuesto**; H = min(r·D, 1) — r es el eje de razón expansión/difusión.

El código original **hardcodea** el coeficiente de la actualización de difusión en 0.5:
`nuevo = phi + 0.5*(media - phi)`. No expone un Δt explícito. Para poder barrer Δt (pedido
por el spec) sin tocar `cs074_rcruz.py`, se generaliza ese coeficiente:

```
paso_difusion_dt(phi, activo, dt):
    ... (idéntico a paso_difusion) ...
    nuevo = phi + dt * (media - phi)
```

**Verificación de equivalencia (T0 — nada puesto a mano sin justificar):** a `dt=0.5`,
`paso_difusion_dt` debe reproducir bit-a-bit `paso_difusion` del original. Esto se
comprueba programáticamente al inicio del motor (`assert` con `np.array_equal`), importando
`cs074_rcruz` directamente (solo lectura, sin editarlo). Si la verificación falla, el motor
se detiene y se reporta — no se sigue.

`paso_expansion` se reutiliza **verbatim** (misma función, copiada sin modificar la lógica),
salvo que ahora recibe H medido con el D correspondiente a cada Δt (ver §3).

## 2. Definición de E_total (propia, mismo principio E1/E2)

Se define el presupuesto de energía como el **primer momento del campo**:

```
E_total(t) := Σ_i φ_i(t)
```

**Por qué esta cantidad y no otra:** bajo difusión pura por promediado simétrico entre
vecinos en un grafo **regular** (anillo completo, todas las aristas vivas), la actualización
es una matriz doblemente estocástica → Σφ_i se conserva **analíticamente**, exacto salvo
error de punto flotante, para cualquier Δt. Es la cantidad conservada natural del propio
esquema de difusión — no una construcción tautológica ni tomada de otro experimento (T2: el
observable no es su propio juez). Corresponde al eje declarado por **E1** ("el presupuesto
energético total se conserva", axioma de diseño explícito del documento madre).

**Por qué se rompe (posiblemente) con la expansión:** cuando `paso_expansion` corta aristas,
los grados locales dejan de ser uniformes (nodos con 0, 1 o 2 vecinos activos). La
actualización de difusión deja de ser doblemente estocástica en general (un nodo de grado 1
mueve 0.5·Δt de su valor hacia el vecino, pero ese vecino puede tener grado 2 y solo
devolver 0.25·Δt) → Σφ_i puede migrar. Esto es la interacción declarada por **E2** ("la
expansión redistribuye E latente en exergía") vista desde la contabilidad del primer
momento: si la redistribución fuese perfectamente conservativa, Σφ no debería moverse; si se
mueve, es porque el corte de aristas introduce una asimetría contable no capturada por el
axioma tal como está declarado.

**deriva(t) := |E_total(t) − E_total(0)| / |E_total(0)|** — igual definición de deriva
relativa que usa el spec en E5.2-1 (línea "|E_total(t)−E_total(0)|/E_total(0)").
E_total(0) = N (exacto, porque φ(0) = 1 + ε·pert con pert de media cero).

Se registra **deriva_final** (al término de la corrida) y **deriva_max** (máximo de deriva(t)
en toda la trayectoria, paso a paso — T6: la conservación se verifica cada paso, no solo al
final).

## 3. H, D y el eje de "tiempo físico" para comparar resoluciones

Para que refinar Δt sea una comparación válida ("misma física, más resolución") se fija un
**tiempo físico total T_total** (unidades propias de este experimento, no ligadas a
`reloj_fisico` de cs074 — es un control interno, no un número puesto a mano en la física: es
la vara con la que medimos "resolución", análogo a mantener T fijo al refinar Δt en un
estudio de convergencia numérico estándar). T_total = 5.0. pasos(Δt) = round(T_total/Δt).

D se mide **fresco en cada combinación (Δt, N, ε)**, con `paso_difusion_dt` al Δt en
cuestión (mismo principio que el original: D medido, no impuesto). H = min(r_target·D, 1).
Se muestra que esta elección mantiene la fracción esperada de aristas cortadas en toda la
corrida aproximadamente constante al variar Δt (para Δt pequeño, D(Δt) ≈ D_rate·Δt a primer
orden, luego H·pasos ≈ r·D_rate·Δt·(T_total/Δt) = r·D_rate·T_total, independiente de Δt) —
así "refinar Δt" no cambia silenciosamente cuánta expansión ocurre en total.

Dos condiciones de r_target:
- **Aislado (H=0, r_target=0):** solo difusión, sin cortes. Prueba la robustez numérica pura
  del esquema de difusión (debería dar deriva ≈ épsilon de máquina, ~1e-16, a toda
  resolución — es la conservación analítica del anillo regular).
- **Expansión activa (r_target=1.0):** valor de r donde cs074_rcruz encontró la transición
  interesante (r≈1). Prueba si el corte de aristas introduce deriva genuina, y si esa
  deriva depende de la resolución.

## 4. Barrido (sobredimensionado, regla del director)

- **Δt** ∈ {1e-4, 3.16e-4, 1e-3, 3.16e-3, 1e-2, 3.16e-2, 1e-1} — 7 puntos log-espaciados,
  cubre exactamente el rango pedido [1e-4…1e-1] con las dos décadas y media intermedias.
- **N** ∈ {128, 256, 512, 1024, 2048} — 5 puntos, duplicando, cubre {128…2048} pedido.
- **Semillas** ≥8 → se usan **8** semillas (seed = 2000+s, s=0..7), independientes de la
  medición de D (que usa su propio rng determinista por (Δt,N,ε)).
- **ε** ∈ {0, 1e-2} — dos condiciones: ε=0 es control de validez del arnés (φ uniforme
  exacto, D=0, la deriva debe ser ~1e-16 SIEMPRE, confirma que el harness no introduce
  deriva espuria); ε=1e-2 es la condición con estructura real donde el mecanismo de E2 tiene
  algo que redistribuir.
- **r_target** ∈ {0 (aislado), 1.0 (expansión)} — dos condiciones físicas.

Total: 7×5×8×2×2 = 1120 corridas. Tiempo estimado por benchmark (venv, N=128..2048,
≈170–220 µs/paso): ≈35–40 min de cómputo total. Autorizado por regla "cómputo largo
autorizado, N=2048 puede ser caro".

## 5. Observable, NULL, PASS (T2, T3, T5)

- **Observable primario:** deriva_final(Δt, N) y deriva_max(Δt, N), agregadas por mediana y
  dispersión (percentiles) sobre las 8 semillas, reportadas como **curva completa** (T5 — no
  gate binario), separadas por condición (aislado/expansión) × (ε=0/ε=1e-2).
- **NULL:** — (el propio spec marca NULL: — para E5.2-5; es caracterización de convergencia,
  no un test de significancia contra permutación). El control de validez es ε=0 (ver §4).
- **PASS pre-registrado (umbral fijado ANTES de correr, T3):**
  1. **Converge (artefacto numérico):** al ir de la combinación más gruesa (Δt=1e-1,
     N=128) a la más fina (Δt=1e-4, N=2048), la mediana de deriva_final cae **al menos un
     factor 10**, Y la deriva_final en la combinación más fina es **< 1e-6** (mismo umbral
     absoluto que usa el spec para E5.2-1). Además, la tendencia debe ser monótona (o
     compatible con monótona dentro de la dispersión entre semillas) a lo largo de TODA la
     curva, no solo en los extremos.
  2. **Diverge / hallazgo grave (problema real, no numérico):** la deriva_final en la
     combinación más fina NO es menor que en la más gruesa (se mantiene igual o crece), o no
     baja de 1e-6 pese a refinar tanto Δt como N. Se reporta así de claro, sin suavizar
     (instrucción explícita del director).
  3. **Mixto/localizado:** si aislado converge pero expansión no (o viceversa), se reporta
     exactamente eso — permite **localizar** si la fuente de la deriva es el esquema de
     difusión (numérico) o el corte de aristas (diseño/topológico), que es precisamente la
     pregunta del experimento.

## 6. Lectura pre-inscrita (hipótesis, ANTES de correr — se reporta se cumpla o no)

- Condición **aislada** (H=0): se espera deriva ≈ épsilon de máquina (~1e-15/1e-16) a TODA
  resolución — Σφ es analíticamente invariante bajo difusión simétrica en el anillo regular,
  independiente de Δt y N. Si esto NO se cumple, es un bug del harness, no un hallazgo
  físico, y se reporta como tal.
- Condición **expansión** (r=1): se sospecha que la deriva es **inducida por la topología**
  (el corte de aristas rompe la simetría doblemente-estocástica), no por el tamaño de paso —
  en cuyo caso NO debería achicarse al refinar Δt manteniendo T_total fijo (porque el número
  total esperado de cortes en el tiempo físico total es ~invariante a Δt por diseño, §3), y
  podría o no achicarse con N (a determinar empíricamente: si el efecto promedia por CLT
  sobre más nodos/cortes independientes, podría caer ~1/√N; si es sesgo sistemático, no).
  **Esta es una hipótesis, no el resultado — se reporta lo que salga.**
- ε=0: deriva ≈ 0 exacto en ambas condiciones (control de validez).

## 7. T0–T7 (checklist)

- T0: nada discreto/dimensional puesto a mano — Δt y N son exactamente los parámetros bajo
  prueba (el objeto del experimento), T_total es un control interno documentado, D sigue
  siendo MEDIDO.
- T1: ningún número puesto a mano en la física — H=min(r·D,1) con D medido; r_target∈{0,1}
  son ejes declarados, no ajustados a un blanco.
- T2: el observable (deriva de Σφ) no es su propio juez — juez es el umbral fijado en §5,
  antes de correr.
- T3: este documento se escribe y su timestamp se fija ANTES de ejecutar el motor.
- T4: N/A — no hay NULL de permutación en este experimento (así lo marca el spec); el
  control de validez es ε=0.
- T5: se reporta la curva deriva(Δt,N) completa, no un gate binario.
- T6: deriva_max (a lo largo de TODA la corrida, no solo al final) se registra por cada
  combinación.
- T7: el barrido Δt×N×semillas×condición ES la perturbación — no hay un único punto ni solo
  semillas repetidas al mismo Δt,N.

## 8. Archivos

- Este pre-registro: `E5_2_5_PROTOCOLO_PREREGISTRO.md`
- Motor: `E5_2_5_engine.py` (a escribir después de este documento)
- Resultado crudo: `E5_2_5_resultado.json`
- Reporte: entregado en el mensaje final al coordinador (CS), no se auto-adjudica veredicto
  de cierre de arco.
