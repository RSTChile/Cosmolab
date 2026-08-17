# PROTOCOLO F1-4 — PRE-REGISTRO (fechado, congelado antes de correr el motor)

**Experimento:** F1-4 · "Independencia de la forma inicial: barrido de familias de perturbación"
**Enfoque:** 1 — ¿persiste una diferencia ínfima en un campo continuo caliente?
**Ejecutor:** CC (agente de esta corrida, prefijo `F1_4_`)
**Fecha/hora de congelamiento:** 2026-07-24 05:33 (hora local del sistema, America/Santiago -04)
**Documento fuente:** `BATERIA_FUNDAMENTOS_F1_a_F4_PARA_CC.md`, sección "F1-4"
**Código base (NO editado, solo importado/leído):** `cs074_rcruz.py`

Este documento se escribe y se congela ANTES de ejecutar el motor de producción.
No se edita después de ver resultados (regla T3). Si algo falla, se reporta el
FAIL tal cual.

---

## 1. Hipótesis / pregunta

¿El mapa de persistencia P(ε,r) medido en `cs074_rcruz.py` depende de la forma
arbitraria con que se siembra la mancha inicial (multi-modo Fourier, la única
forma usada hasta ahora), o es invariante a la familia de perturbación elegida?
La dispersión ENTRE familias, a igual (ε,r), es la medida central de robustez de
este experimento — se reporta cruda, no se esconde ni se suaviza (T0/T1: la forma
no se privilegia a mano).

## 2. Observable exacto (congelado, idéntico a F1-1/CS074-rcruz, sin modificar)

`P = corr(φ, roll(φ,1)) · [var(φ) / var(φ₀)]`

Es la función `persistencia()` de `cs074_rcruz.py` (líneas 152-160), **importada
sin modificación** (no se copia a mano) para que el observable sea idéntico
byte a byte al del código base. No es función de la familia, ε, r o N
directamente (T2): se mide sobre φ final cualquiera sea su origen.

## 3. Física reutilizada sin cambios (importada de `cs074_rcruz.py`)

- Difusión por aristas vivas — `paso_difusion()`.
- Expansión = corte Bernoulli por arista viva con probabilidad H por paso —
  `paso_expansion()`.
- Reloj/temperatura de reporte — `reloj_fisico()`, `temperatura_fisica()`.
- Detección de cuantos — `detectar_cuantizacion()`.

Lo único que este experimento generaliza es **la función que siembra el campo
inicial** (`campo_inicial()` en el código base, válida SOLO para la familia
multi-modo). El motor `F1_4_motor.py` define un generador de campo por familia,
parametrizado, y reusa el resto de la física sin tocar `cs074_rcruz.py`.

## 4. Las 6 familias (congeladas, definidas ANTES de correr — ninguna se elige
para favorecer el resultado, T0/T1)

Todas las familias se normalizan igual: perturbación `pert(x)` cruda →
`pert -= pert.mean()` → si `std(pert)>0`: `pert /= std(pert)` → `φ = 1 + ε·pert`.
Esta normalización (media 0, std 1) es una convención de unidades para que ε
signifique "amplitud relativa" de forma comparable entre familias — no es un
ajuste de resultado, es la definición de la variable ε.

| # | Familia (clave) | Construcción | Parámetro libre por semilla (evita elegir "la" forma a mano) |
|---|---|---|---|
| 1 | `multi_modo` | **Baseline = código base sin tocar.** Suma de 5 modos de Fourier m=1..5 con fase aleatoria: Σ sin(2π·m·x + φ_m)/m | fases φ_m ~ U(0,2π), 5 por semilla (igual que `cs074_rcruz.campo_inicial`) |
| 2 | `modo_unico` | Un solo modo de Fourier: sin(2π·m·x + φ) | m ~ entero uniforme en {1,...,8} por semilla (no se fija "el" modo — evita privilegiar una frecuencia) + φ ~ U(0,2π) |
| 3 | `bulto_gaussiano` | Bulto localizado: exp(−d(x,x₀)²/(2σ²)), d = distancia circular en [0,1) | x₀ ~ U(0,1) por semilla (centro no fijo) + σ ~ U(0.02,0.08) por semilla (ancho no fijo) |
| 4 | `ruido_blanco` | Ruido espectral con espectro de amplitud plano: \|A(k)\| = 1 ∀k, fase aleatoria, síntesis por FFT inversa real | fases ~ U(0,2π) por modo k, por semilla |
| 5 | `ruido_rojo` | Ruido espectral con amplitud ∝ 1/k (potencia ∝ 1/k², dominan escalas grandes) | ídem, fases aleatorias por semilla |
| 6 | `ruido_azul` | Ruido espectral con amplitud ∝ k (potencia ∝ k², dominan escalas chicas) | ídem, fases aleatorias por semilla |

Implementación espectral (ruido_blanco/rojo/azul): `k = rfftfreq` índices
1..N/2 (k=0 forzado a amplitud 0, ya que la media se resta de todos modos);
`A(k) = k^α` con α=0 (blanco), α=−1 (rojo), α=+1 (azul); fase U(0,2π) por
modo; señal real = `irfft(A(k)·e^{iφ(k)})`; luego normalización estándar
(media 0, std 1) descrita arriba.

## 5. Barrido (congelado, todos los ejes)

| Eje | Rango | Puntos | Nota |
|---|---|---|---|
| familia | las 6 de la tabla §4 | 6 | ninguna se omite ni se pondera |
| ε | 0 (control) ∪ logspace(1e-4, 1, 8) | 9 | rango donde se espera señal medible; el piso de amplitud (¿existe ε mínimo?) es objeto de F1-3, no de F1-4 — aquí se fija un rango medible común a las 6 familias, no el extremo 1e-12 de F1-1 |
| r = H/D | {0, 0.1, 0.3, 0.5, 1, 2, 5, 10, 30, 100} | 10 | idéntico a `R_TARGETS` de `cs074_rcruz.py` — mismo eje que el resto de la batería, cruza r≈1 |
| N | 200 (fijo) | 1 | robustez a N es objeto de F1-1/F1-6; aquí se aísla el eje "familia" para no mezclar factores — limitación de alcance declarada, no ocultada |
| semillas | 12 | ≥12 pre-registrado | seeds 1000..1011, independiente por combinación (familia, ε, r) — misma semilla numérica reutilizada entre familias para que la única diferencia entre familias, a igual semilla, sea la construcción de la forma, no el azar |

Total combinaciones (familia × ε × r) = 6 × 9 × 10 = 540 puntos de grid, cada
uno con 12 semillas × (REAL + NULL) = 24 corridas → **12.960 corridas totales**.

**Calibración de pasos (por familia, medida, no impuesta — T1):** para cada
familia se mide D (fracción de contraste borrada en un paso de difusión pura,
H=0) y el tiempo de lavado (`pasos_lavado`, P<0.05 a H=0) en ε=1e-2
(representativo, 8 semillas de calibración), con tope `max_steps=50000`
(documentado; si una familia no lava dentro de ese tope se reporta
`lavo_todas=False`, no se extiende el tope a mitad de corrida). El `pasos_fijo`
resultante de esa calibración se usa para TODOS los ε de esa familia — mismo
procedimiento que el modo "produccion" del código base (que calibra en
ε=1e-3 y reusa el valor para todo el barrido de ese N).

**Perturbación dinámica:** F1-4 no pide barrido de ruido dinámico en cada paso
(eso es F1-5); la perturbación de robustez de F1-4 es el barrido de familia
mismo — es la variable que se perturba a propósito.

## 6. NULL (congelado)

Permutación del campo φ al final de la dinámica (`rng.permutation(phi)`),
aplicada **por familia**: cada corrida REAL de una familia tiene su corrida
NULL emparejada (misma semilla, misma φ inicial, misma secuencia de pasos,
solo difiere en la permutación final) — destruye la forma espacial, conserva
el histograma exacto de valores de ESA familia.

## 7. Controles (congelados)

- **Control r=0 (H=0), por familia:** a ε>0, la difusión debe lavar P_real
  hacia ~0 (umbral P_max=0.15, igual que `control_r0_ok()` del código base).
- **Control ε=0, por familia:** sin diferencia sembrada, P_real ≈0 a TODO r
  (umbral P<0.05 en ≥95% de los puntos r).

## 8. Criterio de PASS (congelado, no se cambia tras ver datos — T3)

**Por familia** (aplicado independientemente a cada una de las 6):
1. Control ε=0 no violado (§7).
2. Control r=0 lava (§7).
3. Banda congelada (r≥10, ε>1e-4): z = (P_real_mean − P_null_mean) /
   sd_combinada ≥ 3 en ≥50% de los puntos ε de esa banda (mismo umbral que
   F1-1, para comparabilidad entre experimentos de la batería).

**Veredicto global F1-4 (pre-registrado, tres lecturas):**
- Si las 6 familias cumplen (1)-(3): **persistencia presente en TODAS las
  familias** → la forma inicial NO determina el resultado (PASS de
  invarianza).
- Si una o más familias fallan (1)-(3) mientras otras pasan: **se reporta
  explícitamente cuál(es) y en qué gate fallan** — NO se promedia ni se
  suaviza (D1 vivo, regla explícita del documento madre).
- Si ninguna familia pasa: hallazgo negativo del enfoque completo, se reporta
  igual.

**Dispersión entre familias (métrica central, reportada siempre, no es un
gate pasa/no-pasa binario inventado):** para cada punto (ε,r) del grid común,
se calcula sobre las 6 medias P_real(familia): media, desviación estándar,
rango (máx−mín). Se reporta la dispersión agregada en la banda congelada
(r≥10) y se identifican los puntos de mayor dispersión sin ocultarlos. No se
fija un umbral arbitrario de "dispersión aceptable" — el número crudo es el
resultado que le corresponde interpretar a CS.

**No se ajusta este criterio después de correr.**

## 9. Verificación cruzada (tres vías obligatorias)

(a) **NULL** — descrito arriba, parte del criterio de PASS por familia.
(b) **Segundo método/observable** — aquí el "segundo método" es la
    comparación INTER-familia misma: si `multi_modo` (idéntico al código base
    ya validado por F1-1) da el mismo mapa (ε,r) que las otras 5, eso
    constituye la verificación cruzada específica de F1-4 (el mapa no
    depende del generador de forma). Adicionalmente, `multi_modo` sirve de
    ancla: es el único punto de contacto directo con el resultado ya
    reportado por F1-1/CS074-rcruz.
(c) **Auditoría en disco** — JSON crudo por familia + agregados + este
    protocolo + log de ejecución con timestamps, en
    `BATERIA_FUNDAMENTOS/F1_4_familias_forma/resultados/`.

## 10. Qué puede fallar (T6 — todo gate debe poder fallar)

- Una familia podría no lavar en r=0 (control roto) → invalida esa familia,
  se reporta y no se fuerza lectura.
- El NULL podría no separarse del REAL en ninguna familia → hallazgo negativo
  íntegro del enfoque.
- Las familias podrían dar mapas cualitativamente distintos (p.ej.
  `ruido_rojo` persiste y `ruido_azul` no) → se reporta como hallazgo D1 vivo,
  NO se oculta promediando.
- `pasos_fijo` calibrado en ε=1e-2 podría no ser representativo para ε
  extremos del barrido en alguna familia → se documenta el `pasos_fijo` usado
  por familia en el JSON de salida para que sea auditable.

## 11. Archivos de salida

- `resultados/F1_4_smoke_<timestamp>.json` — corrida pequeña de validación de
  las 6 familias (no es el resultado final).
- `resultados/F1_4_produccion_resultado.json` — barrido completo (12.960
  corridas), crudo, por familia/ε/r.
- `resultados/F1_4_produccion_analisis.json` — agregados: criterio de PASS
  por familia, dispersión inter-familia por punto y en banda congelada.
- `resultados/F1_4_log_ejecucion.txt` — log con timestamps de inicio/fin.

---
*Congelado. No editar después de este punto salvo para anotar FAIL explícito
de algún control (T3: no se cambia el juez tras el resultado).*
