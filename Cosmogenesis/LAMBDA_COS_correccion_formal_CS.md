# Λ_Cos — corrección formal de la fórmula del cierre

**13-ago-2026.** Trabajo de lápiz, sin cómputo. Encargo del director: arreglar la contradicción abierta desde
el 17-jul antes de que alguien vuelva a usar Λ_Cos como juez.

**Esto es una propuesta de corrección, no una decisión.** La elección de forma final es del director.

---

## 1. El defecto, con precisión

El canon define en **C-N2.8.12**:

```
Λ_Cos = (Δ_struct · LF) / |e_R| · A_sys-env
```

Y en **C-N2.8.5** exige **dos cotas** para el error operativo:

```
0 < |e_R| < κ_O
```

**La contradicción:** `lim_{|e_R| → 0⁺} Λ_Cos = +∞`. La fórmula asigna **salud máxima** exactamente al límite
que el propio teorema declara **no viable**. Señalado por Codex el 17-jul
(`AUDITORIA_CODEX_CS072_invariantes_C-N2.8_PARA_CS.md`), con recomendación de no usarla como juez hasta
adjudicarla. No se resolvió.

## 2. El diagnóstico es más profundo que el límite

No es sólo que diverja en un extremo. **El signo de la relación está invertido respecto de lo que dice la
propia teoría.**

`C-N2.8.8a` fundamenta las cinco condiciones por reducción al absurdo, y sobre el error dice:

> *"|e_R| = 0 ⇒ **no hay señal correctora**: el ciclo no puede ajustarse."*

Es decir: **el error no es un costo, es un requisito.** Es la señal con la que el sistema se corrige. Un
sistema sin error no es un sistema sano: es un sistema ciego.

Pero la fórmula pone `|e_R|` en el **denominador**, que es la forma matemática de decir "menos error, más
salud". Eso contradice a C-N2.8.8a en el cuerpo de la fórmula, no sólo en el límite.

**Confirmación empírica independiente:** el proyecto hermano Cosmoclima cazó el mismo defecto por su cuenta
el 12-ago, midiendo — su clasificador premiaba el fallo de cierre porque sólo miraba la cota de arriba, y
`e_R` resultó tener **mediana cero**. Dos proyectos, dos caminos, el mismo defecto.

## 3. Una ambigüedad de escritura que hay que cerrar de paso

Tal como está escrita, `(Δ_struct · LF) / |e_R| · A_sys-env` es ambigua: puede leerse como

- `((Δ_struct · LF) / |e_R|) · A_sys-env`  ← multiplicando por el acoplamiento
- `(Δ_struct · LF) / (|e_R| · A_sys-env)`  ← dividiendo por el acoplamiento

**C-N2.8.12a** aclara la intención: *"A_sys-env refleja κ_V"*, y más acoplamiento es más salud (κ_V es una
cota **inferior**). Así que la lectura correcta es la primera. **Conviene escribir los paréntesis en el canon**,
porque la segunda lectura invierte el papel del acoplamiento y alguien la va a tomar.

## 4. Lo que cualquier reparación tiene que cumplir

Derivado del propio canon, sin agregar supuestos:

| # | Requisito | De dónde sale |
|---|---|---|
| R1 | Λ_Cos → bajo cuando \|e_R\| → 0 | C-N2.8.8a: sin error no hay señal correctora |
| R2 | Λ_Cos → bajo cuando \|e_R\| → κ_O | C-N2.8.8a: el error desborda y destruye el ciclo |
| R3 | Λ_Cos alto en algún punto interior de la banda | C-N2.8.13: el cierre se sostiene si supera Λ_crit |
| R4 | Λ_Cos crece con Δ_struct, LF y A_sys-env | C-N2.8.12a: los tres reflejan invariantes con cota inferior |
| R5 | Λ_Cos **no definida** si S = 0 | C-N2.8.12a, textual: *"si κ_P se viola, el sistema no existe y Λ_Cos no está definida"* |
| R6 | Adimensional o normalizada por dominio | C-N2.8.14a: los κ son **dominio-específicos**; comparar Λ entre dominios sin traducir es error de plano |

La fórmula actual cumple R3, R4 y R5. **Viola R1 y R2**, y R6 queda sin resolver.

## 5. Candidatos

Escribo `u = |e_R| / κ_O ∈ (0, 1)` — la posición del error dentro de su banda, ya normalizada (cumple R6).

### Candidato A — parábola de banda
```
Λ_Cos = Δ_struct · LF · A_sys-env · 4u(1 − u)
```
- Cumple R1–R6. Vale 0 en los dos bordes, máximo 1 en `u = ½`.
- **A favor:** simple, simétrico, sin parámetros nuevos.
- **En contra:** afirma que el óptimo está exactamente en el medio de la banda, y eso **no está en el canon**.
  Es una elección, no una derivación.

### Candidato B — distancia al borde más cercano
```
Λ_Cos = Δ_struct · LF · A_sys-env · 2·min(u, 1 − u)
```
- Mismo comportamiento en los bordes, pero mide *"cuánto me falta para violar la cota más próxima"*.
- **A favor:** es la lectura más literal de "estar dentro de la banda", y se conecta con C-N2.8.9a, que
  distingue **tipos de ruptura** según qué invariante se viola: este candidato dice **cuál** cota está cerca.
- **En contra:** no es derivable en `u = ½`. Irrelevante para diagnóstico, molesto si alguien deriva.

### Candidato C — mantener la forma actual y sólo cortar arriba
```
Λ_Cos = (Δ_struct · LF / |e_R|) · A_sys-env   si |e_R| < κ_O ;   0 si no
```
- **Es la mínima intervención**, y por eso hay que decir con claridad que **no arregla el problema real**:
  sigue premiando `|e_R| → 0` con salud infinita. Sólo tapa R2 y deja R1 intacto.
- Lo incluyo únicamente para que quede constancia de que se consideró y por qué no alcanza.

## 6. Un hueco del canon que esto destapa

**C-N2.8.5 nombra la cota de arriba (κ_O) y no nombra la de abajo.** Dice `0 < |e_R|`, que sólo significa
"distinto de cero" — sin escala.

Pero R1 exige que la salud caiga cuando el error se hace *demasiado chico*, y "demasiado chico" necesita
respecto de qué. Los candidatos A y B lo resuelven de hecho, usando κ_O como única escala y midiendo la
distancia relativa a ella. Es una salida razonable, pero conviene que sea **explícita**: equivale a decir que
la banda de error es simétrica en su propia escala.

Si la teoría quiere una cota inferior con contenido propio —un error mínimo por debajo del cual el sistema
no puede corregirse, análogo a κ_Δ para la diferencia— **hay que nombrarla y darle valor**. Hoy no existe.
Es un nodo faltante, no un error de fórmula.

## 7. La otra inconsistencia, que es fácil

Codex también señaló que **U_Cos aparece con cinco elementos en C-N2.8.11 y con seis en C-N2.8.11a**.

Esta se resuelve sola leyendo el propio canon: **C-N2.8.11a ya trae la resolución**, al partir el conjunto en
`U_Cos_viab = {κ_P, κ_Δ, κ_O, κ_V, κ_LF}` y `U_Cos_anal = {κ_H}`, y aclarar que κ_H está en U_Cos pero fuera
de la cadena de viabilidad.

**Corrección propuesta:** que C-N2.8.11 diga `U_Cos = {κ_P, κ_Δ, κ_O, κ_V, κ_LF, κ_H}` con la partición
explícita. Es redacción, no teoría.

## 8. Recomendación

**Candidato B**, por dos razones: es la lectura más literal de "estar dentro de la banda", y es el único que
además **dice cuál de las dos cotas está en riesgo** — que es justo lo que C-N2.8.12a lamenta que Λ_Cos no
pueda hacer (*"señala que al menos uno de los invariantes está en riesgo, pero no especifica cuál"*).

Con B, el término de error sí especifica cuál. Es una mejora del diagnóstico, no sólo un parche del límite.

**Y una recomendación de proceso:** cualquiera de las tres formas cambia el valor de Λ_Cos en todo lo que se
haya calculado antes. Antes de adoptar una, conviene revisar dónde se usó Λ_Cos como juez —en Cosmogénesis no
se usó nunca, pero en Cosmoclima y en la Célula Madre sí— y re-leer esos resultados con la forma nueva.

**No se declara nada cerrado.** La elección es del director.
