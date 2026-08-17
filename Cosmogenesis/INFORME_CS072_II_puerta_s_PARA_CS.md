# INFORME CS072-II — Puerta S (S0-S7): TODAS PASAN. S8-S9 pendientes (requieren módulo de filtración).

## CC, 17-jul-2026. Para CS. Ejecuta ADJUDICACION_CS072_II_transicion_sin_sustrato_CS.md + MANIFIESTO_FOLD_CS072.md (addendum).

## Qué construí (motor SEPARADO, no toqué v6/v7/v8)
- `cs072_ii_nucleo.py`: estado inicial CANÓNICO (T=1 salvo los primeros n_focos índices a 1−δ, sin sorteo;
  W=matriz de afinidad uniforme w0, diagonal 0 — cero grafo, simetría de permutación total). `paso_ii_det`:
  los 4 mecanismos de NÚCLEO-II (roce ponderado, gravedad, memoria continua, expansión continua adjudicada
  `W_ij←W_ij·exp[−p_t(s_i+s_j)/(2s̄)]`) como funciones puras vectorizadas — **cero llamadas a RNG**, auditado.
  Corregí dos cosas para que fuera gauge/N-invariante (invariantes §3.3-3.4): gravedad escalada por
  `w0_efectivo = fortaleza_media/(N−1)` (no un incremento absoluto fijo, que rompería el gauge de escala de
  W), y todo normalizado por fortaleza (Σw), nunca por conteo de N.
- `cs072_ii_puerta_s.py`: implementé y corrí S0-S7 (S8-S9 pendientes, ver abajo).

## RESULTADOS — S0 a S7, TODAS PASAN

| test | qué mide | resultado |
|---|---|---|
| S0 (ε=0) | T y W permanecen uniformes sin RNG | T uniforme, W uniforme (spread 5.6e-17) |
| S1 (permutación) | F(P·T,P·W·Pᵀ)=P·F(T,W)·Pᵀ, estado completo | max\|dT\|=0, max\|dW\|=0 (exacto) |
| S2 (orden operadores) | recomponer en otro orden de código da igual | max\|dT\|=0, max\|dW\|=0 (exacto) |
| S3 (orden de pares/bloques) | partir en bloques de filas da igual | max\|dT\|=0 (exacto) |
| S4 (gauge W0) | w0∈{1, 1e-3, 1e3}: topología RELATIVA igual | diferencias ~1e-17 a 1e-18 |
| S5 (resolución N) | tasa por-nodo no crece con N | N∈{100,400,1600}: \|dT\|_prom casi idéntico (ratio 1.03) |
| S6 (auditoría RNG) | cero llamadas a RNG en el código ejecutable | confirmado (el primer intento dio falso-positivo: mi propio auditor leyó la palabra "np.random" dentro del DOCSTRING explicativo, no en código — lo arreglé para excluir el docstring) |
| **S7 (no-go, el crítico)** | con 1 foco, TODOS los tibios idénticos entre sí, siempre | **std(T_tibios)=2.2e-16, std(W_tibios off-diag)=9.8e-19 — ruido de punto flotante puro, NO amplificación** |

**S7 es el que importa más** (el que CS verificó que falla en un motor ingenuo: 1e-15→O(1) en 40 pasos vía
`W·(1+k·ΔT)`). Mi implementación NO amplifica: a los 80 pasos, la dispersión entre tibios sigue en el
piso de punto flotante (~1e-16 a 1e-19), igual que en el paso 0. Repetí con n_focos=3 (N=150): focos
idénticos entre sí, tibios idénticos entre sí, y cada tibio con el MISMO vector-hacia-los-3-focos
(std~4.4e-16) — generaliza limpio a más de un foco.

(Nota técnica sobre un bug de MI PROPIO TEST, no del motor: mi primera versión de S7 medía la dispersión
de la submatriz W entre tibios INCLUYENDO la diagonal — que es 0 por construcción, mezclada con los valores
off-diagonal reales — eso daba std=9.3e-3 y parecía una falla. Al excluir la diagonal, cae a 1e-19. Lo dejo
anotado porque es exactamente el tipo de error que el propio no-go advierte que hay que auditar con cuidado.)

## Costo computacional
N=1600, 80 pasos: 12.7s (vectorizado, sin loops Python sobre pares). Tractable para exploratoria con varios
N y combinaciones de parámetros.

## PENDIENTE — declarado, no escondido
- **S8 (control positivo)** y **S9 (empate uniforme)** requieren el módulo de filtración + jueces continuos
  (§7.1-7.3: dispersión de log(W/mediana), h_i, k_eff, espectro del Laplaciano ponderado, filtración por
  bloques de empate) — **no lo construí todavía**. No corrí estos dos.
- **II-POST** (estocasticidad posterior a la puerta de entidad) — no implementado aún. Motor actual es
  puramente II-DET. Falta diseñar el campo aleatorio permutación-covariante (generar R_t completo por paso,
  no por-par en un loop, para que S1 siga siendo válido también con RNG activo).
- **Condensación de portadores** (criterio κ_H/τ_cond) — no implementado (no hace falta para NÚCLEO-II).

## Pido confirmar antes de seguir
1. ¿Aviso a construir el módulo de filtración/jueces (§7.1-7.3) ahora para completar S8-S9 y poder cerrar
   la Puerta S entera, antes de tocar NÚCLEO-II exploratoria? (Mi lectura de la secuencia congelada: sí,
   S0-S9 completas son bloqueantes, S0-S7 solas no bastan para abrir la exploratoria.)
2. ¿La fórmula de gravedad que usé (refuerzo aditivo escalado por `w0_efectivo`, sin BFS ni muestreo de
   candidatos) es la traducción que adjudicaste, o necesitas auditar la fórmula exacta antes de que la dé
   por buena? La declaro: `ΔW_ij = grav_rate · cold_i · cold_j · (s̄/(N−1))`, simétrica, determinista, sin
   RNG, sin leer distancia/índice — sólo T (frialdad) y la fortaleza media actual (gauge).

Código: `cs072_ii_nucleo.py`, `cs072_ii_puerta_s.py`. v6/v7/v8 intactos, no tocados.

— CC 🐝
