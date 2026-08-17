# PROTOCOLO E5.4-1 — Producción de exergía vs enfriamiento medido, expansión de 1 a 1e4

**Pre-registro fechado.** Congelado ANTES de escribir/correr `E5_4_1_motor.py` (T3). El
motor no se edita después de leer resultados. Cualquier cambio posterior a la primera
corrida de producción se documenta como nuevo experimento, no como edición retroactiva.

Agente: E5.4-1 (Enfoque 5, Tema 4). Batería de 30 experimentos, prefijo propio `E5_4_1_`.
No toca `CF2_estiramiento_motor.py` ni ningún archivo fuera de su carpeta.

---

## 1. Pregunta

¿La exergía (X, capacidad de hacer trabajo) aparece porque el campo se enfría al
expandirse? Se mide T(a) del propio estado del campo — nunca impuesta como fórmula de
a — y se define X como una función del campo, distinta de T. Se verifica si X
correlaciona con la caída de T en REAL, y si esa correlación (y la propia producción de
X) desaparece cuando no hay expansión (NULL).

## 2. Herencia del "sello" físico (T1: ningún número nuevo puesto a mano para favorecer el resultado)

Se reutiliza EXACTAMENTE la física de `CF2_estiramiento_motor.py` (leído completo, no
editado): grilla 2D periódica, campo T(x,y)∈[0,1] con salto tipo tanh como condición
inicial, difusión por Laplaciano de vecinos (np.roll), y la ley de dilución D(a) =
D0·ρ(a)/ρ0 con ρ(a)=ρ0/a³ (axioma de diseño **E2**: la expansión redistribuye energía
latente, diluyendo la tasa de difusión — no crea nada).

```
L = 64            H_EXP = 6.0          RHO0 = 1.0
D0 = 0.12         W0 = 1.2 (ancho comóvil del salto inicial)
DT = 0.25         N_SUB = 2            ORIGINAL_STEPS_PER_TG = 399
```

Condición inicial: perfil tanh (idéntico a CF2) + ruido gaussiano inicial 1e-4 (idéntico
a CF2), clip a [0,1].

## 3. Dos modos (REAL vs NULL) — mismo reloj, misma duración

- **REAL**: ρ(a) = ρ0/a³, D(a) = D0·ρ(a)/ρ0 (dilución real con la expansión).
- **NULL_SIN_EXPANSION**: ρ ≡ ρ0, D ≡ D0 fijos TODO el tiempo (no hay expansión física
  real; el eje "a" se conserva solo como etiqueta de checkpoint, para comparar en el
  mismo reloj genético que REAL — igual convención que `NULL_RHO_FIXED` de CF2). Este es
  el NULL pre-registrado: "sin expansión, no debe producir X" (T4: debe poder morder —
  si el modelo estuviera mal construido, NULL podría producir tanta X como REAL).

Ambos modos comparten el mismo reloj genético `t_g` y los mismos checkpoints de a (no se
compara tiempo distinto contra tiempo distinto).

## 4. Barrido (sobredimensionado, regla del director)

- **a**: `np.geomspace(1.0, 1e4, 31)` — 31 puntos, 4 décadas completas (pedido: 1→1e4).
- **ε** (amplitud de ruido dinámico, T7 — perturbación además de semilla, inyectada en
  CADA paso de difusión, no solo en la condición inicial):
  `[0.0, 1e-12, 1e-9, 1e-6, 1e-3, 1e-1, 1.0]` — 7 puntos, 12 décadas + control en 0,
  siguiendo el mismo rango "ε∈[1e-12…1]" que usan los experimentos hermanos de Tema 1 y
  Tema 3 de este mismo documento (consistencia de batería, no elegido para este
  experimento). En ε=1.0 el ruido ya satura la escala del campo ([0,1]) — deliberado:
  el barrido debe ir más allá de donde se espera que sobreviva la señal.
- **Semillas** (≥12 exigidas): `[7, 42, 99, 777, 2025, 3141, 8191, 99991, 12345, 54321,
  271828, 161803]` — las primeras 10 son las semillas estándar del proyecto (idénticas a
  `SEEDS_STANDARD` de CF2_estiramiento_motor.py); se agregan 2 más al mismo estilo
  (dígitos de e y de φ ×1e5) solo para llegar a ≥12, sin tocar ni mirar resultados antes
  de fijarlas.
- Ruido dinámico: en cada paso macro, después de difundir, se suma `N(0, ε²)` elemento a
  elemento sobre T y se clipea a [0,1] (igual convención de clip que la condición
  inicial).

Total de trayectorias: 2 modos × 12 semillas × 7 valores de ε = 168 corridas completas,
cada una con 31 checkpoints internos.

## 5. Observables — MEDIDOS del propio estado del campo, no impuestos (T2)

En cada checkpoint `a_k` de una trayectoria, sobre el campo COMÓVIL T(x,y) tal como está
en la simulación (nunca se reemplaza por una fórmula de a):

- **T_meas(a)** := promedio espacial de `(∂T/∂x)² + (∂T/∂y)²` (diferencias centrales
  periódicas, np.roll) — "energía de gradiente", el término cinético/de agitación de un
  funcional tipo Ginzburg–Landau `F = ∫[(∇T)²/2 + V(T)]`. Juega el papel de temperatura:
  se MIDE del array real en cada checkpoint, nunca se define como función de a.
- **X(a)** := `Var[T(x,y)]` sobre la grilla completa — varianza espacial, medida de
  estructura/desviación del equilibrio uniforme = capacidad de hacer trabajo (exergía).
  Es una estadística DISTINTA de T_meas (varianza vs gradiente al cuadrado) — ninguna se
  define en términos de la otra ni en términos del criterio de PASS (T2).
- **Secundarios (solo diagnóstico, NO entran al criterio de PASS):** versión "física" por
  conversión de unidades bajo expansión — T_phys(a) = T_meas_comov(a)/a² (mismo criterio
  que CF2: ∇_fis = ∇_comov/a ⇒ al cuadrado, /a²); X_phys(a) = X_comov(a)/a⁴ (amplitud
  diluida por volumen, axioma E2 aplicado como conversión de unidad, no como ajuste).
  Se reportan para interpretar magnitudes, pero el veredicto usa las cantidades
  COMÓVILES (evita que la correlación sea mecánica por construcción, es decir, evita que
  "T cae con a" y "X cae con a" solo porque ambas se dividieron por la misma potencia de
  a a mano).
- **E_comov_total(a)** := `sum(T(x,y))` en cada checkpoint — diagnóstico de conservación
  (T6). Con ε=0 el operador de difusión (Laplaciano por np.roll, dominio periódico)
  conserva esta suma exactamente salvo error de punto flotante, para cualquier D — se
  verifica. Con ε>0 el ruido inyectado rompe la conservación exacta; se reporta la deriva
  vs ε como costo explícito del axioma de ruido dinámico, no como violación oculta.

**"Enfriamiento"** := `ΔT_cool(a_k) = T_meas(a_1) − T_meas(a_k)` (caída acumulada desde
el primer checkpoint de la propia trayectoria).

## 6. Correlación (congelada)

Por cada trayectoria (mode, seed, ε): correlación de Pearson `r` entre las series
`{X(a_k)}` y `{T_meas(a_k)}` a lo largo de los 31 checkpoints (misma lógica que E5.2-2:
correlación temporal del observable contra el "juez" de enfriamiento, pero aquí a lo
largo de a en vez de t). Se computa por trayectoria completa, luego se agregan las 84
trayectorias REAL y las 84 NULL (12 semillas × 7 ε cada una).

Si la varianza de X o de T_meas en una trayectoria es < 1e-12 (campo ya totalmente
homogéneo o congelado, sin variación que correlacionar), `r` se marca `NaN` — no se
imputa un valor a mano; se cuenta aparte como "sin señal", NO como r=0 forzado (T1).

## 7. Criterio de PASS (congelado, no se toca tras ver resultados — T3)

- **P1 — REAL produce señal real:** mediana de `|r_REAL|` (sobre las 84 trayectorias
  REAL con r definido) ≥ 0.6, Y ≥70% de las trayectorias REAL con r definido tienen
  `|r| ≥ 0.5`.
- **P2 — NULL no produce X:** mediana sobre las 84 trayectorias NULL de
  `X_NULL(a_final)/X_NULL(a_1)` ≤ 0.05 (colapso ≥95% de la estructura sin expansión), Y
  mediana de `X_NULL(a_final)/X_REAL(a_final)` ≤ 0.10 (REAL retiene ≥10× más estructura
  final que NULL, apareado por semilla y ε).
- **P3 — la correlación es más débil o indefinida en NULL:** mediana de `|r_NULL|`
  (donde esté definida) ≤ 0.3, O ≥50% de las trayectorias NULL tienen r indefinido
  (X colapsada a varianza ~0, sin señal que correlacionar — eso también cuenta como
  "NULL no produce el fenómeno").
- **Veredicto global** = P1 AND P2 AND P3. Si solo se cumple una parte, se reporta como
  parcial/negativo, sin suavizar ni forzar el "PASS" de la batería (regla 6 y 9 de
  EJECUCIÓN: nunca se ajusta un coeficiente hacia el resultado esperado; el motor no se
  toca después de correr).
- Se entregan las curvas ENTERAS X(a) y T(a) por ε y por modo (media±dispersión entre
  semillas) además del número resumen (T5) — el resumen no reemplaza la curva.

## 8. Axiomas declarados

- **E1** (conservación del presupuesto total): NO es el foco de E5.4-1 (eso es Tema 2);
  aquí se verifica solo como diagnóstico (sección 5, E_comov_total) y se reporta la
  deriva, sin gatear el veredicto de este experimento.
- **E2** (la expansión redistribuye energía latente en exergía, no la crea): implementado
  como la tasa de dilución D(a)=D0/a³ que rige la DINÁMICA (no el observable), más las
  versiones "físicas" secundarias de la sección 5, explícitamente marcadas como
  conversión de unidad axiomática y excluidas del criterio de PASS.

## 9. Qué NO se hace

- No se impone T(a) = fórmula cerrada de a (eso sería T2: el observable siendo su
  propio juez). T(a) sale de medir el campo simulado.
- No se ajusta ningún coeficiente para acercar el resultado a 4.9%/31.5% (esos targets
  no aplican a este experimento — es Tema 3).
- No se edita `CF2_estiramiento_motor.py` ni ningún archivo fuera de
  `BATERIA_ENFOQUE5/E5_4_1_produccion_exergia/`.
- No se auto-adjudica el veredicto final de la batería — se entrega crudo a CS.

---
Fecha/hora de congelamiento: ver mtime de este archivo (debe ser anterior al mtime de
`E5_4_1_motor.py`).
