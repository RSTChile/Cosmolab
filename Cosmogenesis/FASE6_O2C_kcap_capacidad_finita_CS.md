# O2-C — ¿Es `kcap=6` un número especial, o cualquier saturación relacional produce la misma familia de geometrías?

**Fecha:** 11-ago-2026 · Ejecuta: CC (Claude) · Tarea **O2-C** del `FASE6_PLAN_EJECUCION_COMPLETA_CS.md`,
propuesta **F6-06 de GPT-5.6 Sol**. Antecedentes directos: `FASE5_auditoria_C2_resultado_CS.md` (auditoría
que declaró `kcap` topológicamente limpio, PASS), la línea `FASE5_mecanismo_aislado_CS.md` →
`FASE5_control_azar_elastico_CS.md` (lo que produce extensión es la **rigidez del corte + el criterio de
soporte local**), y `FASE6_O2B_genealogias_escaladas_CS.md` (dato llegado *después* de lanzar esta corrida:
`kcap` explica η² = 0,619 de la variación, contra η² = 0,078 de la genealogía).

**Observable principal:** la **pendiente continua corregida** — regresión de log(diámetro de la componente
gigante) contra log(N_cajas) sobre el coarse-graining b = 1, 2, 4, 8, 16, con el diámetro medido por
`cs090_diam_corregido.diam_gigante` (`FASE6_adopcion_diam_corregido_CS.md`). Las clases I-IV se reportan
sólo como **dato secundario**, por consigna del equipo: el escalón de clasificación pierde contra la rampa
continua (R² = 0,663 vs 0,182, `FASE6_reanalisis_azar_continuo_CS.md`).

**No se modificó ningún script existente ni congelado** (sólo se los invocó). **No se corrió Phantom.**
**No se hicieron commits de git.** **No se declara cierre ni veredicto** — se reportan números; la lectura
final es de Alexis.

---

## 0. En simple, con analogía

Imaginá que cada nodo es una persona y `kcap` es **el tope de amigos que puede sostener**. Hasta ahora el
proyecto venía poniendo ese tope en 6 a mano, y aparecía geometría extendida. La sospecha razonable era:
*¿y si el 6 tuviera algo mágico?* Si lo tuviera, la teoría estaría apoyada en un número arbitrario, y eso
sería un problema serio. Si no lo tuviera —si el 5, el 7 o un reparto desigual de topes dieran "la misma
clase de mundo"— entonces la frase que hay que escribir en la teoría no es *"kcap=6 genera geometría"* sino
*"la capacidad finita de relación genera extensión bajo ciertas reglas locales"*.

Las tres pruebas, en simple:

1. **Tope absoluto fijo, pueblos de distinto tamaño.** Mismo tope de 4, 5 ó 6 amigos, en pueblos de 500 a
   4 000 personas. ¿Manda "6 amigos" o manda "6 amigos *por cada tanto* de pueblo"?
2. **Tope que crece con el pueblo.** Que el tope suba con el tamaño (∝ log N, ∝ N^⅓). ¿Alguna de esas
   fórmulas mantiene la misma clase de mundo a todas las escalas?
3. **Topes desiguales, mismo promedio.** Todos con 6 de promedio, pero repartido: unos con 2, otros con
   400. ¿Hace falta que todos tengan el mismo tope, o alcanza con que el tope exista?

**El resumen en cinco frases.** *(a)* Con el tope absoluto fijo, la geometría **casi no se mueve** al
multiplicar el pueblo por 8: manda el número absoluto, no la razón con el tamaño (el peso de log kcap es
81 veces el de log N). *(b)* El 6 **no es especial**: es un punto cualquiera de una **rampa suave** —
pendiente = 2,457 − 1,036·log(kcap), R² = 0,729 — y de hecho es de los kcap que **menos** se despegan de esa
rampa. *(c)* **Ninguna normalización conserva la familia**; al contrario, normalizar al tamaño la **rompe**:
el que la conserva mejor es justamente el tope absoluto constante. *(d)* Los topes desiguales **no
reproducen** la misma geometría: la desplazan mucho, y en dosis-respuesta con lo desigual que sea el reparto
(Spearman +0,745). *(e)* Pero cuando se pregunta *por qué*, la respuesta es en su mayor parte aburrida y
unificadora: **casi todo pasa por cuántas relaciones se llegan a formar de verdad** — las 570 corridas de las
tres pruebas caen sobre **una sola curva** en función del grado medio alcanzado (R² = 0,834), y ahí sí queda
un residuo pequeño pero no nulo que la forma del reparto sí aporta.

---

## 1. Estado de la corrida: qué quedó completo y qué no

La corrida original fue interrumpida por límite de sesión del agente anterior. Al retomar, **las tres partes
estaban completas en disco** con la grilla exactamente balanceada; lo que faltaba era el análisis conjunto
(el PNG en disco databa de las 20:12 y la Parte 3 terminó a las 20:28, o sea que el gráfico previo no
contenía P3) y la verificación de integridad, que nunca se había registrado.

| Parte | CSV | filas | grilla | estado |
|---|---|---:|---|---|
| **P1** — kcap absoluto | `cs090_fase6_o2c_p1_kcap_absoluto.csv` | 180 | kcap {4,5,6} × N {500,1000,2000,4000} × 15 reglas | **completa**, 12/12 celdas con n = 15 |
| **P2** — normalizada | `cs090_fase6_o2c_p2_normalizado.csv` | 120 | {NORM-log, NORM-pot} × N {500,1000,2000,4000} × 15 reglas | **completa**, 8/8 celdas con n = 15 |
| **P3** — heterogénea | `cs090_fase6_o2c_p3_heterogeneo.csv` | 270 | 6 distribuciones × N {1000,2000,4000} × 15 reglas | **completa**, 18/18 celdas con n = 15 |

**Lo que se hizo en esta sesión:** correr `--modo verificar` (nunca registrado antes) y volver a correr
`cs090_fase6_o2c_analizar.py` **sobre las tres partes juntas**, regenerando
`cs090_fase6_o2c_capacidad_finita.png` y dejando el registro en `cs090_fase6_o2c_analisis.log` y
`cs090_fase6_o2c_verificar.log`. No se relanzó ninguna simulación: no hacía falta.

**Nota de interpretación (intérprete):** los scripts requieren `/usr/local/bin/python3.11` (el `python3` por
defecto de esta Mac no tiene `scipy`). Queda anotado para quien retome.

**Lo que la grilla lanzada NO cubre** (limitación de diseño, no de ejecución):

- **P3 no incluye N = 500.** Se corrió a 1000/2000/4000. La comparación heterogéneo-vs-homogéneo se hace
  igual, pareada, sobre los tres tamaños comunes.
- **P2 no corrió `NORM-ramif`** (factor de ramificación constante). Justificado y ahora **verificado
  numéricamente**: en este rango de N produce la secuencia entera **idéntica** a NORM-log (5, 5, 6, 7), así
  que habría sido cómputo duplicado. Tampoco corrió `NORM-grado`, que colapsa sobre el caso absoluto porque
  ⟨grado inicial⟩ = meandeg no depende de N.
- **kcap = 7 y kcap = 8 sólo existen a N = 4000** (entran por la vía de las normalizaciones, no del diseño
  absoluto). Están confundidos con el tamaño; el confound se acota más abajo con el resultado de P1.
- Sólo se exploró el eje **A2-B0-C2** y los kcap **4-8**. No se sabe qué pasa abajo de 4 ni arriba de 8.

---

## 2. Verificaciones de integridad (no se asumen, se corrieron)

```
VERIFICACIÓN 1 — cupo uniforme por vector == C2-hard del motor congelado
  grafos idénticos tras el corte: 12/12  (esperado 12/12)
```

Es decir: la dinámica de cupo-por-nodo que usan P2 y P3, alimentada con un vector **constante**, produce
grafos **bit a bit idénticos** a los del motor congelado con `kcap` escalar. La maquinaria nueva no cambia
nada cuando no debe cambiarlo.

```
VERIFICACIÓN 2 — las fábricas de cupo respetan la media pedida y difieren en dispersión
  UNIF          media=6,000  CV=0,000  min=6  max=6
  HET-rango     media=5,949  CV=0,299  min=3  max=8
  HET-grado     media=6,042  CV=0,401  min=1  max=15
  HET-bimodal   media=6,000  CV=0,667  min=2  max=10
  HET-lognor    media=6,002  CV=0,884  min=1  max=51
  HET-potencia  media=6,005  CV=2,418  min=2  max=434
```

**La media total de cupo es la misma en las seis condiciones** (5,95-6,04). Sólo cambia **cómo se reparte**.
Esto es lo que permite decir después "la diferencia no es que haya más cupo, es la forma del reparto".

---

## 3. PRUEBA 1 — `kcap` absoluto fijo, N variable

### 3.1 La tabla principal

Pendiente continua corregida, media ± error estándar (n = 15 reglas por celda):

| kcap | N = 500 | N = 1000 | N = 2000 | N = 4000 | deriva b(pend ~ log N) | r |
|---|---|---|---|---|---:|---:|
| **4** | 1,112 ± 0,048 | 1,049 ± 0,046 | 1,059 ± 0,034 | 1,080 ± 0,027 | −0,012 | −0,06 |
| **5** | 0,703 ± 0,027 | 0,738 ± 0,026 | 0,776 ± 0,027 | 0,754 ± 0,016 | +0,028 | +0,22 |
| **6** | 0,549 ± 0,020 | 0,544 ± 0,015 | 0,580 ± 0,015 | 0,608 ± 0,014 | +0,031 | +0,36 |

**Leído en simple:** cada fila es prácticamente **plana**. Multiplicar el pueblo por 8 (500 → 4 000) mueve la
pendiente menos de lo que la mueve **cambiar el tope en una sola unidad**.

### 3.2 Los contrastes pareados dicen lo mismo, sin promedios

Entre topes (mismo par regla-N, o sea cada regla es su propio control):

| contraste | n | media dif. | mediana dif. | signos | Wilcoxon |
|---|---:|---:|---:|---|---:|
| kcap 4 vs 5 | 60 | +0,332 | +0,304 | **60 / 0** | p = 1,6·10⁻¹¹ |
| kcap 4 vs 6 | 60 | +0,505 | +0,510 | **60 / 0** | p = 1,6·10⁻¹¹ |
| kcap 5 vs 6 | 60 | +0,172 | +0,148 | **59 / 1** | p = 1,8·10⁻¹¹ |

Entre tamaños (mismo par regla-kcap):

| contraste | n | media dif. | signos (sube/baja) | Wilcoxon |
|---|---:|---:|---|---:|
| N 4000 − 500 | 45 | +0,026 | 27 / 18 | p = 0,10 |
| N 4000 − 1000 | 45 | +0,037 | 29 / 16 | p = 0,01 |
| N 4000 − 2000 | 45 | +0,009 | 30 / 15 | p = 0,28 |

**60 de 60 contra 27 de 45.** Cambiar el tope da un resultado unánime; cambiar el tamaño da algo que apenas
se distingue de una moneda.

### 3.3 El modelo conjunto pone número a la comparación

```
pendiente ≈ 2,684 − 1,254·log(kcap) + 0,016·log(N)      R² = 0,775   n = 180
razón b1/b2 = −80,8
```

Si lo que mandara fuera la **razón capacidad/tamaño** (kcap/N), los dos coeficientes serían comparables y de
signo opuesto — la razón valdría ≈ −1. Vale **−81**. Es decir: **manda el número absoluto de vínculos que
un nodo puede sostener, y el tamaño del sistema es casi irrelevante.**

Analogía: lo que le da forma a la red social no es "cuántos amigos tenés en proporción al pueblo", es
"cuántos amigos podés sostener, punto". Un pueblo de 4 000 con tope 6 se parece a uno de 500 con tope 6, y
no se parece nada a uno de 4 000 con tope 4.

---

## 4. La curva unificada — ¿el 6 tiene algo especial?

Juntando P1 y P2 (300 corridas homogéneas, kcap de 4 a 8):

| | N = 500 | N = 1000 | N = 2000 | N = 4000 | **todos los N** |
|---|---|---|---|---|---|
| kcap = 4 | 1,112 (30) | 1,049 (15) | 1,059 (15) | 1,080 (15) | **1,082** (75) |
| kcap = 5 | 0,703 (30) | 0,738 (45) | 0,776 (15) | 0,754 (15) | **0,736** (105) |
| kcap = 6 | 0,549 (15) | 0,544 (15) | 0,580 (45) | 0,608 (15) | **0,574** (90) |
| kcap = 7 | — | — | — | 0,518 (15) | **0,518** (15) |
| kcap = 8 | — | — | — | 0,456 (15) | **0,456** (15) |

```
Ajuste global:  pendiente = 2,457 − 1,036·log(kcap)     r = −0,854   R² = 0,729   n = 300
```

**Saltos entre topes consecutivos:** 4→5 = −0,347 · 5→6 = −0,162 · 6→7 = −0,055 · 7→8 = −0,062.

**Residuo medio de cada kcap contra esa rampa** (cuánto se despega cada tope de la curva suave):

| kcap | n | residuo medio | ee | t |
|---|---:|---:|---:|---:|
| 4 | 75 | **+0,062** | 0,018 | +3,4 |
| 5 | 105 | −0,054 | 0,010 | −5,6 |
| **6** | 90 | **−0,027** | 0,007 | −4,1 |
| 7 | 15 | +0,077 | 0,021 | +3,8 |
| 8 | 15 | **+0,154** | 0,017 | +9,2 |

**El 6 es el que MENOS se despega de la rampa suave** (|residuo| = 0,027, el más chico de los cinco). Si
hubiera que señalar a un kcap como "raro" respecto de la curva, el candidato sería el 8, no el 6. **No hay
escalón, discontinuidad ni singularidad en 6.**

Lo que sí hay, y conviene decirlo con honestidad porque es un hallazgo aparte: **la rampa se aplana a
partir de 6**. Los primeros escalones (−0,347, −0,162) son más pronunciados que lo que predice el ajuste
logarítmico, y los últimos (−0,055, −0,062) son más suaves. Traducido: **cada unidad adicional de capacidad
compra mucha menos geometría después de 6 que antes de 6.** Eso no hace especial al 6 como número, pero sí
sugiere que la zona 4-6 es donde el mecanismo es sensible y de 6 en adelante entra en rendimientos
decrecientes. Cautela: kcap 7 y 8 tienen n = 15 cada uno y sólo a N = 4000.

### 4.1 Réplica independiente sobre datos ya archivados

El mismo patrón aparece re-midiendo el brazo `C2-hard` de F5-C2-C3 (archivo distinto, corrida distinta,
todo a N = 2000, o sea **sin el confound de tamaño** que tienen kcap 7-8 arriba):

| kcap | n reglas | pendiente media (N = 2000, archivo) |
|---|---:|---:|
| 4 | 3 | 1,038 |
| 5 | 7 | 0,817 |
| 6 | 5 | 0,585 |
| 7 | 5 | 0,487 |

```
Spearman kcap vs pendiente = −0,907  (p = 3,5·10⁻⁸),  n = 20 reglas archivadas
```

Misma rampa monótona, mismo aplanamiento después de 6, en datos que **no** se generaron para esta pregunta.

---

## 5. PRUEBA 2 — ¿Alguna normalización conserva la familia de geometrías?

Las normalizaciones están **ancladas** de modo que las tres coinciden exactamente en N = 2000, kcap = 6. La
pregunta es qué pasa a los costados de ese ancla.

| normalización | N = 500 | N = 1000 | N = 2000 | N = 4000 | deriva | r |
|---|---|---|---|---|---:|---:|
| **ABS (kcap = 6)** | 0,549 [k=6] | 0,544 [k=6] | 0,580 [k=6] | 0,608 [k=6] | **+0,031** | +0,36 |
| NORM-log (∝ log N) | 0,703 [k=5] | 0,738 [k=5] | 0,580 [k=6] | 0,518 [k=7] | −0,103 | −0,64 |
| NORM-pot (∝ N^⅓) | 1,112 [k=4] | 0,738 [k=5] | 0,580 [k=6] | 0,456 [k=8] | −0,306 | −0,88 |

**Dispersión de la pendiente entre tamaños** — cuanto menor, más se conserva la familia:

| | medias por N | sd entre N | rango |
|---|---|---:|---:|
| **ABS (kcap = 6)** | 0,549 · 0,544 · 0,580 · 0,608 | **0,030** | **0,064** |
| NORM-log | 0,703 · 0,738 · 0,580 · 0,518 | 0,103 | 0,219 |
| NORM-pot | 1,112 · 0,738 · 0,580 · 0,456 | 0,284 | 0,655 |

**Respuesta directa a la pregunta del diseño: ninguna normalización conserva la familia — y el que la
conserva es el tope absoluto constante,** por un factor de 3,4× sobre NORM-log y 9,5× sobre NORM-pot.

Esto es el complemento exacto de la Prueba 1, y es más fuerte que un resultado nulo: normalizar la capacidad
al tamaño del sistema **introduce** una deriva de escala que sin normalizar **no existe**. La normalización
no es neutral; es una perturbación. Cuanto más agresiva (N^⅓ vs log N), más rompe.

Analogía: si a un pueblo que crece le vas subiendo el tope de amigos "para compensar", no conservás el tipo
de sociedad — la vas convirtiendo en otra cosa. La sociedad se conserva si el tope se queda quieto.

---

## 6. PRUEBA 3 — Capacidad heterogénea entre nodos, misma media

Seis repartos, todos con media 6 (verificada empíricamente, §2), a N = 1000/2000/4000, 15 reglas.

| condición | CV cupo | kcap medio | kcap máx | N = 1000 | N = 2000 | N = 4000 | **todos** |
|---|---:|---:|---:|---|---|---|---|
| **UNIF** | 0,00 | 6,00 | 6 | 0,544 ± 0,015 | 0,580 ± 0,015 | 0,608 ± 0,014 | **0,578 ± 0,009** |
| HET-rango | 0,33 | 5,99 | 9 | 0,610 ± 0,017 | 0,638 ± 0,011 | 0,667 ± 0,017 | **0,638 ± 0,009** |
| HET-grado | 0,41 | 5,97 | 15,9 | 0,483 ± 0,031 | 0,513 ± 0,025 | 0,532 ± 0,020 | **0,509 ± 0,015** |
| HET-bimodal | 0,67 | 6,00 | 10 | 1,169 ± 0,023 | 1,161 ± 0,031 | 1,158 ± 0,016 | **1,162 ± 0,014** |
| HET-lognor | 0,94 | 6,00 | 73,7 | 0,979 ± 0,037 | 1,044 ± 0,029 | 1,044 ± 0,022 | **1,023 ± 0,018** |
| HET-potencia | 2,94 | 6,00 | 604 | 1,151 ± 0,039 | 1,198 ± 0,033 | 1,281 ± 0,042 | **1,210 ± 0,023** |

Contrastes **pareados** contra el caso homogéneo (misma regla, mismo N):

| condición | n | media dif. | mediana | signos (sube/baja) | Wilcoxon |
|---|---:|---:|---:|---|---:|
| HET-rango | 45 | +0,061 | +0,064 | 34 / 11 | 5,3·10⁻⁶ |
| HET-grado | 45 | **−0,068** | −0,082 | 10 / 35 | 1,8·10⁻⁵ |
| HET-bimodal | 45 | +0,585 | +0,589 | **45 / 0** | 5,7·10⁻¹⁴ |
| HET-lognor | 45 | +0,445 | +0,469 | **45 / 0** | 5,7·10⁻¹⁴ |
| HET-potencia | 45 | +0,632 | +0,611 | **45 / 0** | 5,7·10⁻¹⁴ |

```
Dosis-respuesta:  Spearman(pendiente, CV del cupo) = +0,745  (p = 5,8·10⁻⁴⁹, n = 270)
```

**Lectura directa: la heterogeneidad NO es inocua.** No basta con "que la capacidad sea finita y en promedio
6"; el reparto importa, y mucho — hasta +0,63 de pendiente, o sea del orden del efecto de bajar el tope de
6 a 4. Y hay dosis-respuesta ordenada con lo desigual que sea el reparto. Notar además el **cambio de
signo** de HET-grado: es la única que va para el otro lado.

---

## 7. El control decisivo — ¿la forma del reparto, o sólo la saturación efectiva?

Antes de concluir que "la forma importa", hay que descontar lo obvio: un reparto muy desigual **desperdicia
cupo**. Si a un nodo le tocan 400 de tope pero sólo tiene 8 vecinos disponibles con soporte, ese cupo no se
usa; y sus vecinos, con tope 2, se saturan enseguida. Resultado: la red **termina con menos aristas**, aunque
la media de cupo sea idéntica. Y ya sabemos por P1/P2 que menos vínculos ⇒ más pendiente.

Grado medio efectivamente alcanzado (b = 1), que es la evidencia del desperdicio:

| condición | grado alcanzado | aristas (b=1) | diám. (b=1) | gigante |
|---|---:|---:|---:|---:|
| UNIF | 4,03 | 4 709 | 11,6 | 0,983 |
| HET-rango | 3,63 | 4 255 | 12,3 | 0,975 |
| **HET-grado** | **4,93** | 5 766 | 9,8 | 0,982 |
| HET-bimodal | 2,61 | 3 055 | 18,2 | 0,883 |
| HET-lognor | 2,60 | 3 033 | 16,5 | 0,892 |
| HET-potencia | 2,18 | 2 527 | 22,2 | 0,836 |

Se ajusta entonces una curva de referencia **usando sólo las corridas homogéneas** (P1 + P2, n = 300), en
función del grado realmente alcanzado, y se pregunta si las heterogéneas caen encima:

```
pendiente = 1,812 − 0,888·log(grado medio alcanzado)     r = −0,912   R² = 0,833   (sólo homogéneas)
```

| condición | CV | grado real | pend. observada | pend. predicha | **residuo** | Wilcoxon vs 0 | % de la dif. explicado por el grado |
|---|---:|---:|---:|---:|---:|---:|---:|
| UNIF | 0,00 | 4,03 | 0,578 | 0,576 | **+0,001** | p = 0,79 | (referencia) |
| HET-rango | 0,33 | 3,63 | 0,638 | 0,669 | −0,030 | 0,0061 | 48 % |
| HET-grado | 0,41 | 4,93 | 0,509 | 0,400 | **+0,109** | 6,3·10⁻¹² | (signo invertido) |
| HET-bimodal | 0,67 | 2,61 | 1,162 | 0,963 | **+0,200** | 1,7·10⁻¹³ | 66 % |
| HET-lognor | 0,94 | 2,60 | 1,023 | 0,966 | +0,056 | 0,0039 | 88 % |
| HET-potencia | 2,94 | 2,18 | 1,210 | 1,122 | +0,088 | 3,8·10⁻⁶ | 86 % |

**Dos cosas a la vez, y las dos importan.**

**(1) La mayor parte del efecto de la heterogeneidad es saturación efectiva, no forma.** Para los repartos
de cola larga —lognormal y ley de potencia, los más extremos— el 86-88 % de la diferencia contra el caso
homogéneo se explica simplemente por "terminaron con menos aristas". El caso uniforme cae exactamente sobre
su propia curva (residuo +0,001, p = 0,79), que es el control de sanidad de que el ajuste no está torcido.

**(2) Queda un residuo chico pero real, y en un caso cambia el signo.** Cinco de seis condiciones tienen
residuo estadísticamente distinto de cero. El más grande es **HET-bimodal** (+0,200): con sólo dos clases de
nodos (unos con 2, otros con 10) la red queda **más extendida de lo que su cantidad de aristas explicaría**.
Y **HET-grado** es el contraejemplo instructivo: es la única con **más** aristas que el caso homogéneo (4,93
vs 4,03) y, coherentemente, **menos** pendiente en crudo (−0,068) — pero contra la curva queda **+0,109
arriba**, o sea que el canal "grado" **sobre-explica** su caso. Ahí la forma del reparto está empujando en
dirección contraria al conteo de aristas.

Hay que decir que HET-grado es un caso aparte por construcción: su cupo está **correlacionado con el grado
inicial del nodo en el grafo Erdős-Rényi de partida**, o sea con quién ya tenía vecinos. Las otras cinco
sortean el cupo independientemente de la posición. Que sea la única que se mueve en dirección opuesta es
consistente con que lo que importe no sea la dispersión del cupo *per se*, sino **si el cupo está o no
alineado con la estructura ya existente**. Es una hipótesis que estos datos sugieren pero no testean.

### 7.1 El colapso sobre una sola curva

| conjunto | n | ajuste pendiente ~ log(grado alcanzado) | R² |
|---|---:|---|---:|
| **todas las corridas** | **570** | pend = 1,908 − 0,940·log(grado) | **0,834** |
| sólo homogéneas | 300 | — | 0,833 |
| sólo heterogéneas | 270 | — | 0,854 |
| *comparación:* homogéneas con log(**kcap**) | 300 | — | 0,729 |

**Las 570 corridas de las tres pruebas —tres diseños distintos, cinco valores de tope, seis repartos, cuatro
tamaños— caen sobre una sola curva con R² = 0,834.** Y el grado **alcanzado** predice mejor (0,834) que el
tope **nominal** (0,729).

Es probablemente el número más importante del informe: la variable de control no es `kcap`, es **cuántas
relaciones se llegan a sostener de verdad**. `kcap` importa porque es la manija más directa para moverla.

Como dato de contexto, la fracción de llenado del cupo es notablemente estable en el caso homogéneo:
kcap 4 → 0,592 · 5 → 0,642 · 6 → 0,668 · 7 → 0,671 · 8 → 0,649. Los nodos usan entre el 59 % y el 67 % de su
tope, casi sin importar cuál sea el tope.

### 7.2 Las otras métricas acompañan (la "familia" no es sólo la pendiente)

| | holonomía | gigante | diám. (b=1) |
|---|---:|---:|---:|
| P1 kcap = 4 | 2,396 | 0,860 | 19,6 |
| P1 kcap = 5 | 2,375 | 0,951 | 13,8 |
| P1 kcap = 6 | 2,237 | 0,982 | 11,2 |
| P3 UNIF | 2,255 | 0,983 | 11,6 |
| P3 HET-rango | 2,263 | 0,975 | 12,3 |
| P3 HET-grado | 2,456 | 0,982 | 9,8 |
| P3 HET-bimodal | 2,727 | 0,883 | 18,2 |
| P3 HET-lognor | 2,440 | 0,892 | 16,5 |
| P3 HET-potencia | 2,900 | 0,836 | 22,2 |

Las condiciones de pendiente alta también tienen componente gigante más chica y diámetro más largo. La
"familia de geometrías" se mueve como un bloque, no es un artefacto de un solo estadístico.

---

## 8. Cruce con el dato de O2-B (llegado después de lanzar esta corrida)

`FASE6_O2B_genealogias_escaladas_CS.md`, con 800 corridas y 20 genealogías, reportó % Clase III por kcap:
**4 → 98,4 % · 5 → 71,7 % · 6 → 4,9 % · 7 → 0 %**, con η²(kcap) = 0,619 frente a η²(genealogía) = 0,078.

Esta corrida, con reglas y semillas distintas, reproduce ese perfil casi exactamente:

| kcap | O2-B (% Clase III) | O2-C, P1 (% Clase III+IV) |
|---|---:|---:|
| 4 | 98,4 % | **100,0 %** (n = 60) |
| 5 | 71,7 % | **66,7 %** (n = 60) |
| 6 | 4,9 % | **0,0 %** (n = 60) |

**La réplica es buena, y el aporte de O2-C es reinterpretarla.** Lo que en O2-B parecía "kcap domina la
variación, y algo raro pasa entre 5 y 6", acá se ve como lo que es: **una rampa continua y suave que el
umbral de clasificación corta cerca de kcap ≈ 5,5**. El "dominio de kcap" (η² = 0,619) es real y se confirma;
pero el aparente carácter especial del 6 es **el umbral del clasificador, no la física del modelo**. Es
exactamente el fenómeno que motivó la consigna de usar la rampa continua en vez del escalón: el escalón
convierte una pendiente suave en un acantilado aparente.

---

## 9. Dato secundario — reparto de clases I-IV

No es el endpoint; se reporta por continuidad con informes anteriores.

| conjunto | n | % III+IV | detalle |
|---|---:|---:|---|
| P1 kcap = 4 | 60 | 100,0 % | III: 60 |
| P1 kcap = 5 | 60 | 66,7 % | III: 38, I: 20, IV: 2 |
| P1 kcap = 6 | 60 | 0,0 % | I: 58, II: 2 |
| P2 NORM-log | 60 | 30,0 % | III: 18, I: 40, II: 2 |
| P2 NORM-pot | 60 | 38,3 % | III: 23, I: 30, II: 7 |
| P3 UNIF | 45 | 0,0 % | I: 44, II: 1 |
| P3 HET-rango | 45 | 17,8 % | I: 37, III: 8 |
| P3 HET-grado | 45 | 0,0 % | I: 35, II: 10 |
| P3 HET-bimodal | 45 | 100,0 % | III: 43, IV: 2 |
| P3 HET-lognor | 45 | 100,0 % | III: 43, IV: 2 |
| P3 HET-potencia | 45 | 100,0 % | III: 40, IV: 5 |

Se ve el problema del escalón en vivo: HET-bimodal (pendiente 1,162) y HET-potencia (1,210) dan las dos
100 %, indistinguibles, cuando la rampa continua las separa claramente y además dice que la primera está
0,2 por encima de lo que su densidad de aristas explicaría y la segunda sólo 0,09.

---

## 10. Límites explícitos de lo que se midió

- **No se probó capacidad emergente** (derivada del historial o del costo del nodo). Decisión deliberada,
  documentada en el encabezado del script: ya se probó en F5-C2-C→C5 y el presupuesto elástico **no**
  reproduce el cupo rígido.
- **kcap 7 y 8 sólo a N = 4000**, n = 15 cada uno. El confound con el tamaño se acota con P1 (mover N 8× vale
  ≈ 0,026 de pendiente, contra saltos de 0,05-0,35 por unidad de kcap) y con la réplica de archivo a N = 2000
  fijo (§4.1), pero sigue siendo el punto más débil de la curva unificada.
- **Un solo eje** (A2-B0-C2) y **kcap entre 4 y 8**. No se sabe qué pasa en kcap ≤ 3 ni ≥ 9, ni si la rampa
  sigue aplanándose o se corta.
- **P3 sin N = 500.**
- El aplanamiento de la rampa después de kcap = 6 es una observación de la forma de la curva; **no se testeó
  formalmente contra un modelo alternativo** (por ejemplo, si un ajuste con quiebre gana contra el
  logarítmico simple).
- La hipótesis de §7 sobre HET-grado (que lo que importa es si el cupo está **alineado** con la estructura
  preexistente, y no la dispersión del cupo) **está sugerida por los datos pero no testeada**. Sería un
  experimento chico y directo: sortear cupos con la misma distribución que HET-grado pero **permutados** entre
  nodos, rompiendo la correlación con el grado inicial y conservando exactamente la distribución.

---

## 11. Lectura — la respuesta a la pregunta del diseño

**¿Es `kcap = 6` un número especial?** No, según estas 570 corridas. Es un punto de una rampa suave y
monótona en log(kcap) (R² = 0,729), y es de hecho el valor que **menos** se despega de esa rampa. Lo que en
O2-B parecía un salto entre 5 y 6 (71,7 % → 4,9 % de Clase III) es el **umbral del clasificador** cortando
una pendiente continua, no una transición del modelo. Sí hay una observación adicional, más débil y con menos
datos: la rampa **se aplana de 6 en adelante**, o sea que la zona 4-6 es donde la capacidad compra geometría
y de 6 hacia arriba compra cada vez menos.

**¿Cualquier saturación relacional produce la misma familia de geometrías?** No exactamente la misma —
produce **la misma familia paramétrica, recorrida en distinto punto**. Las 570 corridas de los tres diseños
caen sobre una sola curva en función del **grado medio efectivamente alcanzado** (R² = 0,834), que predice
mejor que el tope nominal (0,729). O sea: es una única ley, y el valor de `kcap`, la normalización elegida y
la forma del reparto son todas maneras distintas de moverse a lo largo de ella. En ese sentido la
reformulación que proponía F6-06 **sí queda sostenida por los números**: lo que hay que escribir no es
"kcap = 6 genera geometría" sino algo del orden de "**la capacidad finita de relación genera extensión, y
cuánta extensión depende de cuántos vínculos se llegan a sostener de hecho**".

**Tres precisiones que la reformulación tiene que llevar puestas, porque los datos las obligan:**

1. **La capacidad relevante es absoluta, no relativa al tamaño.** Es un resultado positivo, no un nulo: el
   tope constante es el que conserva la familia a través de escalas (sd entre N = 0,030) y normalizar al
   tamaño la **rompe** (0,103 con log N, 0,284 con N^⅓). La teoría, si quiere ser invariante de escala,
   necesita una capacidad que **no** escale con el sistema.
2. **La capacidad finita no es una condición binaria.** No basta con decir "hay tope": el valor del tope es
   la variable que más manda, con efecto pareado unánime (60/60) y monótono.
3. **Homogeneidad y heterogeneidad no son equivalentes, aunque la mayor parte de la diferencia sea
   indirecta.** Con la misma media de cupo, repartirlo desigual mueve la pendiente hasta +0,63 en
   dosis-respuesta con el CV (Spearman +0,745). De eso, el 86-88 % pasa por el canal aburrido (menos aristas
   efectivas), pero queda un residuo de forma que no es cero en 5 de 6 condiciones, con máximo en el reparto
   bimodal (+0,200), y un caso —HET-grado— que va en dirección contraria y apunta a que lo que importaría es
   si el cupo está **alineado con la estructura preexistente**.

**Números, no cierre.** Las tres pruebas quedaron completas y la interpretación es de Alexis. El experimento
más barato y más informativo que se desprende de acá es el control de permutación de §10 sobre HET-grado.

---

### Archivos

| archivo | qué es |
|---|---|
| `cs090_fase6_o2c_capacidad_finita.py` | script principal (no modificado; sólo invocado en `--modo verificar`) |
| `cs090_fase6_o2c_analizar.py` | analizador (no modificado; re-corrido sobre las 3 partes) |
| `cs090_fase6_o2c_p1_kcap_absoluto.csv` | 180 filas — Prueba 1 |
| `cs090_fase6_o2c_p2_normalizado.csv` | 120 filas — Prueba 2 |
| `cs090_fase6_o2c_p3_heterogeneo.csv` | 270 filas — Prueba 3 |
| `cs090_fase6_o2c_p1.log` · `_p2.log` · `_p3.log` | logs de las tres corridas originales |
| `cs090_fase6_o2c_verificar.log` | **nuevo** — verificaciones de integridad |
| `cs090_fase6_o2c_analisis.log` | **nuevo** — análisis conjunto de las 3 partes |
| `cs090_fase6_o2c_capacidad_finita.png` | **regenerado** — ahora incluye la Prueba 3 |
