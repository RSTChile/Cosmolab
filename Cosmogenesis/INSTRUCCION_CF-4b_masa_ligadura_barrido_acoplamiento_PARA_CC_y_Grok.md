# CF-4b — "¿Existe un régimen donde la masa-ligadura domina sobre los constituyentes?"
### Barrido de la razón acoplamiento/potencial — instrucción para CC o Grok

**Director:** Alexis López Tapia · **Diseño:** Claude Science (CS) · **Fecha:** 24-jul-2026
**Serie:** CF (Cosmo-Física). **Corrige:** CF-4 (FAIL no concluyente — coeficientes a mano).
**NO toca CF-4:** es un experimento nuevo, pre-registrado aparte.

Siglas: **CDC** = Cromodinámica Cuántica (fuerza nuclear fuerte) · **ME** = Modelo Estándar.

---

## 1. QUÉ QUIERE PROBAR (en simple)

Cuando unos quarks se confinan en un protón, el protón pesa **mucho más que la suma de
sus partes** — porque la masa es la energía que cuesta mantenerlos juntos (la ligadura),
no la materia de los quarks. La pregunta de este experimento:

> **¿Existe algún régimen físico en el que la energía de ligadura de un cierre sea MUCHO
> MAYOR que la masa de sus constituyentes — o es estructuralmente imposible en este campo?**

Cualquiera de las dos respuestas es un resultado válido. Si existe el régimen, es la
primera señal real del mecanismo de masa de la CDC. Si NO existe en todo el barrido, es
un negativo fuerte y honesto (mucho más fuerte que el FAIL de CF-4, que no probaba nada).

---

## 2. POR QUÉ CF-4 NO SIRVIÓ (y qué corrige esto)

CF-4 dio FAIL (ratio tope ~0.2 vs umbral 5.0), **pero el FAIL no era del mecanismo — era
de los coeficientes.** Verificado en disco (CF4_confinamiento.py, líneas 62-64):
`R0=2.0, U=0.5, D_PHI=0.05` estaban **hardcodeados, heredados de v6, nunca barridos.**
Con D_PHI tan chico frente a R0/U, el término de ligadura (D_eff·ΔΦ²) **nunca podía**
dominar al de potencial (R0·Φ²+U·Φ⁴) — el resultado estaba decidido de antemano por tres
números puestos a mano. **Eso violó T1 (número a mano) sin querer.**

CF-4b corrige exactamente eso: **la razón acoplamiento/potencial deja de ser un número
fijo y pasa a ser el EJE del barrido.**

---

## 3. DISEÑO

- **Eje nuevo barrido (lo esencial):** la razón **γ = D_PHI / (R0·U)^escala** — el peso
  relativo del término de ligadura frente al de potencial. **Barrer γ en rango amplio,
  varias décadas** (p.ej. D_PHI de 0.001 a 50 con R0/U fijos como referencia, O mejor:
  barrer los tres y reportar contra γ). NO elegir un valor — recorrer todo el rango.
- **Observable (idéntico a CF-4, NO se cambia — anti-T2):** m₂ = energía de ligadura del
  cierre = trabajo para separarlo, medido del funcional de energía del campo. **Sin usar
  co_member ni linaje** (ese fue el error de v6). ratio_lig = m₂ / (suma de masas de
  constituyentes, m₁).
- **NULL (idéntico):** cierre de misma composición con enlaces barajados. ratio_null =
  m₂(REAL) / m₂(NULL).
- **Barrido secundario:** intensidad de confinamiento (H_TOPO) y tamaño k (medido, no
  impuesto), como en CF-4. Múltiples semillas.

---

## 4. CRITERIO PRE-REGISTRADO (congelar ANTES de correr — anti-T3)

Escribir `PROTOCOLO_CF-4b_PREREGISTRO.md` fechado, con esto, ANTES del motor:

- **PASS del mecanismo:** existe un sub-rango de γ donde `ratio_lig ≥ 5.0` (ligadura
  domina) **Y** `ratio_null ≥ 1.25` (REAL le gana al barajado) de forma **estable**
  (no un punto aislado — una banda con ≥3 puntos contiguos y ≥N semillas).
- **NEGATIVO fuerte:** en TODO el rango de γ, ratio_lig < 5.0 → la ligadura no puede
  dominar en este tipo de campo. **Esto es un hallazgo, se reporta como tal.**
- **Umbrales 5.0 y 1.25 son los de CF-4, heredados — NO se cambian tras ver el dato.**
  Si el resultado queda cerca del umbral, se reporta la curva, no se ajusta el umbral.
- **Se reporta la CURVA ENTERA** ratio_lig(γ), no el punto que pasa (anti-T5).

---

## 5. LAS TRAMPAS QUE ESTE EXPERIMENTO DEBE EVITAR (checklist)

| # | Trampa | Cómo la evita CF-4b |
|---|---|---|
| T1 | número a mano | γ es barrido, no fijado — corrige justo el defecto de CF-4 |
| T2 | observable circular | m₂ = energía (trabajo de separación), NO co_member/linaje |
| T3 | cambiar juez tras FAIL | umbrales 5.0/1.25 congelados en el pre-registro |
| T4 | NULL no muerde | ratio_null contra enlaces barajados; verificar que cae |
| T5 | gate decorativo | la curva ratio_lig(γ) es continua; se reporta entera |
| T7 | un punto/una semilla | banda de ≥3 γ contiguos × múltiples semillas |

**Nota sobre robustez de semilla (lección de CF-2):** si el campo es una PDE casi
determinista, muchas semillas dan casi el mismo número — el "rate 10/10" NO sería
robustez real. Para robustez genuina, **variar γ (que sí perturba la dinámica) es lo que
cuenta**, no la semilla. Reportar la dispersión real entre semillas, no solo el rate.

---

## 6. REGLAS DE EJECUCIÓN (las de la batería CF, §4)

1. Pre-registro fechado ANTES de correr; si falla, se reporta el FAIL — no se edita.
2. Barrido de rango + semillas; nada de un punto.
3. NULL que muerde; verificar que cae.
4. La cantidad medida (m₂) ≠ su juez.
5. Todo gate debe poder fallar.
6. Quien corre no cambia el código a criterio propio; si ve un error, PARA y reporta a CS
   con la línea exacta.
7. Ejecutar completo.
8. Verificación cruzada: quien no lo escribió lo audita en disco (código + JSON), no de
   palabra.
9. Entregar crudo a CS — números y curva completa, sin adjudicar ("domina/no domina" lo
   dice CS con la curva a la vista).

---

## 7. QUÉ ENTREGAR A CS

- `PROTOCOLO_CF-4b_PREREGISTRO.md` (fechado, antes del motor).
- Código `CF-4b_masa_ligadura_barrido.py` (o nombre equivalente descriptivo).
- JSON crudo con la curva ratio_lig(γ) completa, ratio_null(γ), histograma de k, y
  dispersión entre semillas.
- Tiempo de corrida y pico de RAM.
- **NO adjudicar.** CS lee la curva y decide si hay régimen de dominancia o es negativo.

---

**En una frase:** CF-4 falló porque tres números estaban puestos a mano; CF-4b convierte
esos números en un barrido y pregunta honestamente si existe algún régimen donde la masa
sea energía de ligadura dominante — y reporta la curva entera, pase o no pase.
