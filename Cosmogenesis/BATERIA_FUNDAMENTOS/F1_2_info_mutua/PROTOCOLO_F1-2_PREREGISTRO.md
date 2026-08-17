# PROTOCOLO F1-2 — PRE-REGISTRO (fechado, congelado antes de correr el motor)

**Experimento:** F1-2 · "Persistencia por información mutua espacial (observable independiente)"
**Enfoque:** 1 — ¿persiste una diferencia ínfima en un campo continuo caliente?
**Ejecutor:** CC (agente de esta corrida, prefijo `F1_2_`)
**Fecha/hora de congelamiento:** 2026-07-24 (hora local del sistema, America/Santiago -04),
antes de ejecutar `F1_2_motor.py produccion`.
**Documento fuente:** `BATERIA_FUNDAMENTOS_F1_a_F4_PARA_CC.md`, sección "F1-2"
**Código base (NO editado, solo importado/leído por ruta):** `cs074_rcruz.py`
**Referencia de grid (mismo barrido, para comparación punto a punto):**
`BATERIA_FUNDAMENTOS/F1_1_forma_magnitud/PROTOCOLO_F1-1_PREREGISTRO.md` y su
`F1_1_motor.py` (grid ya congelado por el agente F1-1 al momento de escribir esto;
F1-2 REUTILIZA ese mismo grid exacto, no lo redefine, para que la comparación
mapa-a-mapa tenga sentido — así lo pide el documento madre).

Este documento se escribe y se congela ANTES de ejecutar el motor de producción.
No se edita después de ver resultados (regla T3 de la batería). Si algo falla, se
reporta el FAIL tal cual.

---

## 1. Hipótesis / pregunta

¿La persistencia de una diferencia ínfima sembrada en un campo continuo caliente
—ya detectada en F1-1 vía autocorrelación de forma— también aparece con un
**estimador completamente distinto y no relacionado**: información mutua espacial
entre las dos mitades del dominio? Si dos observables ortogonales dan el mismo
mapa (ε, r), la persistencia no es un artefacto del observable de F1-1 (T2).

## 2. Observable exacto (congelado) — INDEPENDIENTE de F1-1

**NO se usa `persistencia()` de `cs074_rcruz.py`.** Se define un observable nuevo,
`informacion_mutua_mitades(phi, K)`, implementado en `F1_2_motor.py`:

1. El dominio φ (N puntos, anillo periódico) se parte en dos mitades por índice:
   `A = φ[0:N/2]` (primera mitad) y `B = φ[N/2:N]` (segunda mitad, antipodal en el
   anillo). El punto `A[j]` se empareja con `B[j]` (posición x_j vs. posición
   x_j + 0.5, opuestas en el círculo) — un emparejamiento espacial fijo,
   independiente del contenido, no elegido para dar resultado (T1).
2. Los valores de A y B se discretizan en `K=8` bins de igual frecuencia
   (cuantiles) calculados sobre el conjunto combinado A∪B de ESA corrida (dato-
   dependiente, no un umbral fijo puesto a mano). `K=8` es un metaparámetro
   congelado ANTES de correr, igual para todo el barrido — no se ajusta por punto.
3. Histograma conjunto 2D `H[K,K]` de los pares `(bin(A[j]), bin(B[j]))`,
   `j=0..N/2-1` → probabilidad conjunta `p_ab`, marginales `p_a`, `p_b`.
4. Información mutua en bits: `I(A;B) = Σ p_ab · log2(p_ab / (p_a·p_b))`.
5. **Observable reportado (`P_mi`)** = información mutua normalizada simétrica:
   `NMI = 2·I(A;B) / (H(A)+H(B))`, acotada a `[0,1]` (0 si A o B son degenerados,
   es decir, sin varianza — igual que el guardián de `persistencia()` en el código
   base para contraste0=0).
6. **Diagnóstico secundario** (no forma parte del criterio de PASS, solo se reporta
   para contexto): entropía espacial de bloques `H_bloques_norm` — se parte φ en 8
   bloques espaciales contiguos, se promedia cada bloque, se discretiza el vector de
   8 medias en K=8 cuantiles y se calcula su entropía de Shannon normalizada por
   log2(K). Sirve para verificar cualitativamente que la estructura de bloques
   grandes (no solo el emparejamiento A/B) también refleja la dinámica.

El observable NO es función directa de ε, r o N (no es circular, T2): se mide
sobre el campo φ final de la dinámica, cualesquiera que hayan sido los parámetros.
`P_mi` (NMI) y la `persistencia()` de F1-1 son estimadores matemáticamente no
relacionados (uno mide correlación lineal a primer vecino × varianza; el otro mide
dependencia estadística no lineal, discretizada, entre dos MITADES del dominio) —
cumplen la regla de "dos observables ortogonales, ninguno define al otro".

## 3. Física (idéntica a `cs074_rcruz.py`, importada sin modificar, no copiada a mano)

Se importa el módulo `cs074_rcruz.py` por ruta (`importlib`, archivo ajeno, NO se
edita) y se reutilizan sin cambios:
- `campo_inicial()` — fondo=1 + ε·(perturbación multi-modo Fourier m=1..5, fases
  aleatorias, normalizada a std=1).
- `paso_difusion()` — promedio con vecinos por aristas vivas, vectorizado.
- `paso_expansion()` — corte Bernoulli por arista viva con probabilidad H por paso.
- `evolucionar()` — corre la dinámica pasos veces; si `null=True`, permuta φ al
  final (`rng.permutation(phi)`) — el mismo NULL que usa F1-1, aplicado ANTES de
  que F1-2 calcule su propio observable sobre el φ resultante.
- `medir_D()` — D medido del propio campo (1 paso, H=0).
- `medir_pasos_lavado()` — pasos calibrados por lavado medido (según la
  `persistencia()` de F1-1, que sigue siendo el criterio de calibración física
  compartido — el observable propio de F1-2 se mide DESPUÉS, no interviene en la
  calibración de pasos, para no acoplar el juez con el instrumento, T2).
- H(r) = min(r·D, 1.0) — r es la razón interna H/D medida, no un número puesto a
  mano (T1).

Lo único nuevo de F1-2 es el observable `informacion_mutua_mitades()` (sección 2) y
el bucle de barrido/registro; la física generativa es exactamente la misma que
usa F1-1 y el código base, para que ambos observables midan el MISMO φ final bajo
los MISMOS parámetros.

## 4. Barrido (congelado, mismo grid que F1-1 — reutilizado, no redefinido)

| Eje | Rango | Puntos |
|---|---|---|
| ε | 1e-12 … 1 (log) + ε=0 (control) | `np.logspace(-12,0,12)` ∪ {0} = 13 |
| r = H/D | 0 … 100, fino cerca de r≈1 | 34 puntos: 0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0 |
| N | {200, 400, 800, 1600} | 4 |
| semillas | 12 (seeds 1000..1011, idéntico `SEED_BASE` a F1-1) | ≥12 pre-registrado |

Total combinaciones (ε×r×N) = 13 × 34 × 4 = 1768 puntos de grid, cada uno con 12
semillas × (REAL + NULL) = 24 corridas → 42.432 corridas totales — mismo tamaño de
barrido que F1-1.

**Perturbación dinámica:** igual que F1-1, la robustez de F1-2 viene del barrido
denso de (ε, r, N) con 12 semillas independientes por punto; F1-2 no añade ruido
dinámico (eso es F1-5, fuera de este experimento).

**Nota de presupuesto de cómputo / ingeniería (declarada ANTES de correr
producción real, no es recorte del barrido tras ver resultados — T3):** una
corrida ingenua (una llamada Python por cada una de las 42.432 combinaciones,
usando `corrida()` de `cs074_rcruz.py` tal cual) se midió empíricamente
INTRATABLE (proyección: días para N=1600, por overhead fijo de Python/NumPy
repetido miles de veces sobre arreglos chicos — la física en sí es barata). Por
eso `F1_2_motor.py` corre la dinámica **vectorizada por lotes**: todas las
combinaciones (ε,r,semilla) de un N dado evolucionan como un único arreglo
(M,N) con M=5304 canales en paralelo, bajo la MISMA regla física, calibrada con
`medir_pasos_lavado` al default `max_steps=200000` del código base (SIN
recortar el techo de lavado — ver `F1_2_motor.py`, `barrido_N_batch`). El
kernel batched se valida bit a bit contra el kernel de un solo canal de
`cs074_rcruz.py` antes de cada corrida (`_validar_kernel_batched`, gate T6: si
no coincide, se aborta sin producir resultados). Única desviación declarada:
la dinámica estocástica del lote usa un generador compartido en vez de una
instancia nueva por combinación (cada canal sigue recibiendo muestreo i.i.d.
válido; se renuncia solo a la técnica de "números aleatorios comunes entre
r" que usa F1-1, no a la validez estadística). Los ejes del barrido (ε, r, N,
semillas) quedan completos, sin recortar.

## 5. NULL (congelado)

Permutación del campo φ al final de la dinámica (`rng.permutation(phi)`,
`null=True` en `evolucionar()` del código base) — destruye el emparejamiento
espacial A↔B que usa el observable de F1-2 (misma mecánica de NULL que F1-1,
aplicada sobre el mismo φ, pero jueces distintos la miden). Mismo φ inicial y
misma secuencia de pasos que el REAL, solo difiere en la permutación final.

**Predicción pre-registrada del NULL:** al barajar, el emparejamiento posición-a-
posición entre A y B se destruye → `P_mi` (NMI) debe colapsar hacia el nivel de
sesgo de muestreo finito del estimador (no necesariamente 0 exacto — con K=8 bins
y ~N/2 muestras hay sesgo positivo conocido de los estimadores de información
mutua por conteo; por eso el criterio de PASS compara REAL vs NULL con el MISMO
estimador y el mismo tamaño de muestra, no un umbral absoluto — así el sesgo se
cancela en la comparación).

## 6. Controles (congelados)

- **Control r=0 (H=0):** a ε>0, la difusión debe lavar y dejar `P_mi_real` cerca
  del nivel del NULL (gate de validez del cruce, análogo a `control_r0_ok()` del
  código base, umbral `P_mi_real − P_mi_null < 0.15` en promedio).
- **Control ε=0:** sin diferencia sembrada, `P_mi_real` debe quedar cerca de
  `P_mi_null` en TODO r (no distinguible del azar) — si se separa del NULL en
  ε=0, es fuga del observable, se reporta tal cual.

## 7. Criterio de PASS (congelado, tres lecturas — no se cambia tras ver datos)

1. **NULL cae / REAL gana en la banda congelada:** en r≥10, z-score
   `(P_mi_real_mean − P_mi_null_mean)/sd_combinada ≥ 3` en al menos el 50% de los
   puntos con ε>1e-6, r≥10 (mismo umbral mecánico que F1-1, para comparabilidad).
2. **Control ε=0:** la fracción de puntos r donde `z < 3` (es decir, REAL no se
   distingue del NULL) debe ser ≥95% en ε=0 — si no, violación del control, se
   reporta sin reinterpretar el resto.
3. **Control r=0:** `P_mi_real(r=0, ε>0) − P_mi_null(r=0, ε>0)` promedio < 0.15
   (evidencia de que a r=0 el barajado y la difusión dejan al REAL indistinguible
   del azar).

**Veredicto PASS mecánico de F1-2 (pre-registrado):** persiste si (1) y (3) se
cumplen y (2) no se viola. Esto es el gate MECÁNICO interno de F1-2 (su propia
validez como observable). **El criterio de comparación con F1-1 (T2, la verdadera
verificación cruzada de esta pareja de experimentos) es aparte:**

**PASS de la pareja F1-1/F1-2 (no lo adjudica CC):** el mapa (ε,r) de z-scores (o
de la región z≥3) de F1-2 debe coincidir cualitativamente con el de F1-1 —
misma banda r donde ambos separan del NULL, mismo comportamiento en ε=0 y r=0. La
tolerancia numérica exacta y el veredicto final de coincidencia los fija CS al
cruzar los dos JSON crudos (F1-2 no se auto-adjudica ese veredicto).

## 8. Verificación cruzada (tres vías, T obligatorias)

(a) NULL — descrito arriba, parte del criterio de PASS mecánico interno.
(b) Segundo observable — F1-2 ES el segundo observable de F1-1 (rol asignado por
    el documento madre); la verificación cruzada de F1-2 hacia AFUERA es la
    comparación mapa-a-mapa con F1-1 (sección 7), que CS resuelve con ambos JSON.
(c) Auditoría en disco — código + JSON crudos (por N) + este protocolo + log de
    ejecución con timestamps en `BATERIA_FUNDAMENTOS/F1_2_info_mutua/resultados/`.

## 9. Qué puede fallar (T6 — todo gate debe poder fallar)

- El NULL podría NO caer (P_mi_real ≈ P_mi_null en toda la banda) → hallazgo
  negativo del observable de información mutua (aunque F1-1 sí separe).
- El control ε=0 podría dar separación espuria del NULL → indicaría fuga del
  observable (p.ej. sesgo de binning correlacionado con r), se reporta.
- El mapa (ε,r) de F1-2 podría NO coincidir con el de F1-1 → sería evidencia de
  que uno de los dos observables mide un artefacto, no persistencia real — el
  documento madre pide reportarlo así explícitamente, no forzar coincidencia.
- El binning K=8 con muestras pequeñas (N=200 → 100 puntos por mitad) podría dar
  estimadores ruidosos → se reporta la dispersión entre semillas sin suavizar.

## 10. Archivos de salida

- `resultados/F1_2_smoke_resultado.json` — corrida pequeña de validación del
  motor (no es el resultado final).
- `resultados/F1_2_produccion_N{200,400,800,1600}_resultado.json` — barrido
  completo por N, con valores por semilla (dispersión real, no solo medias).
- `resultados/F1_2_produccion_resumen.json` — agregados + evaluación mecánica del
  criterio de PASS interno (secciones 7-8), por N y global.
- `resultados/F1_2_log_ejecucion.txt` — log con timestamps de inicio/fin de fase.

---
*Congelado. No editar después de este punto salvo para anotar FAIL explícito de
algún control (T3: no se cambia el juez tras el resultado).*
