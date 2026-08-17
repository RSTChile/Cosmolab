# Adjudicación cs074 A/B/C — los tres experimentos de seguimiento del holístico
### Veredicto de CS · verificado en disco (campo correcto) · en lenguaje simple

**Director:** Alexis López Tapia · **Adjudica:** Claude Science (CS) · **Fecha:** 26-jul-2026
**PROPUESTA DE CIERRE — no cierra sin tu autorización explícita** (regla permanente).

---

## Lo que verifiqué yo mismo (no de palabra)

- **A (el techo de asimetría):** abrí el JSON y reconstruí la curva REAL en reserva
  abundante — coincide con CC: meseta ~77% hasta ε≈0,5, caída en ε≈0,9–2,3, colapso a
  ~10-14% en ε≳3,8. El control sin energía (`A['control']`, presupuesto infinito) es un
  dict con clave "{ε}_{semilla}"; agrupado por ε, da la curva **idéntica a la real —
  diferencia +0,0 en los 20 valores de ε** (verificado en código que corrió, tras un primer
  intento fallido en que mi celda no llegó a computar el control). Techo NO energético,
  confirmado con número, no de memoria.
- **C (el 5%):** confirmé en el JSON que la fracción de materia da **z=1,37 (bootstrap,
  4180 puntos) — no significativo.** El modelo se acerca al 4,9% tanto como cualquier punto
  por volumen del barrido, no por mecanismo. Verificado.

---

## A · "¿Por qué demasiada asimetría produce menos estructura?" — RESUELTO, más rico de lo esperado

El techo no-monótono es **real** y se confirmó en un barrido 4× más fino. Pero no es un solo
mecanismo — son tres regímenes:
1. **Meseta (ε≲0,5):** ~77% ligado, plano por casi 3 décadas de ε. Más asimetría NO cambia
   nada aquí. (Ojo: refuta la lectura ingenua "más ε → menos estructura" como ley general.)
2. **Fragmentación (ε≈0,9–2,3):** la masa se reparte en más grumos y más chicos — aquí sí
   pasa lo que sospechábamos.
3. **Colapso (ε≳3,8):** los grumos desaparecen del todo — la condición inicial queda tan
   caótica que la gravedad no organiza nada.
**Causa confirmada: mecánica/dinámica, NO energética** — el control sin presupuesto da la
misma curva. Buen resultado: una ley de comportamiento emergente, medida y explicada.

## B · "¿Dónde actúa el enfriamiento?" — NEGATIVO limpio y robusto (PASS=False)

El enfriamiento H₂ **no mueve ningún observable de estructura** — ni cuánta masa liga
(observable viejo) ni en cuántos pedazos se parte (observable nuevo de B). Curva plana de
apagado a 10× intensidad; en los 11 niveles el real no se separa del control barajado
(z entre −0,11 y −0,14). 1980 corridas + control. **Es un negativo real, no un instrumento
ciego** — los dos observables coinciden en que el canal no actúa. Honesto: CC deja las
posibles razones anotadas (60 pasos puede ser corto; la presión térmica de soporte puede
dominar) como pistas, no como excusa.

## C · "¿Relación y proceso sí, números físicos no?" — CONFIRMADO donde se pudo medir

**La columna NO (números físicos):**
- Fracción de materia (4,9%/31,5%): **z=1,37, NO significativo** — indistinguible del azar
  del barrido. Verificado por mí.
- Razón p:n (7,1) y masa p/e (1836): **no evaluables** — son constantes del motor (no varían
  en el barrido), no salidas emergentes. CC fue honesto marcándolas así en vez de forzar una
  comparación sin sentido. La p:n además usa la física de congelamiento con constantes reales
  como ENTRADA — que dé 7,1 es implementar bien lo conocido, no un hallazgo.

**La columna SÍ (relaciones/procesos) — 6 de 7 sostienen con control real:**
contabilidad de energía cierra (1,7% fuga), costo de ligadura causal (29,3% de celdas
difieren), muerte térmica ≠ Nada (retiene ~100% del presupuesto), expansión rescata
estructura (88,4% sin vs 60,7% con), techo no-monótono real (A), gravedad indispensable
(60,7%→2,0%). La séptima (enfriamiento fragmenta) NO sostiene — es el negativo de B.

---

## VEREDICTO DEL ARCO (propuesta, no cierre)

**El modelo da relación y proceso, no los números físicos del universo — confirmado ahora
por múltiples caminos independientes, y con un hallazgo emergente propio.**

- **El "NO da números" queda confirmado donde se pudo medir** (fracción de materia, z=1,37)
  y honestamente abierto donde el método no aplica (las dos constantes). Nada se ajustó para
  acercarse al blanco — es el anti-20.0 funcionando.
- **El "SÍ da relación y proceso" queda bien respaldado** — 6 de 7 con control significativo.
- **Hallazgo emergente genuino (A):** demasiada asimetría destruye la estructura, en tres
  regímenes (meseta / fragmentación / colapso), por mecanismo dinámico no energético. No lo
  buscamos; salió solo. Es el tipo de contingencia que sí le sirve a la Teoría.
- **Dos negativos limpios (B, y el enfriamiento en C):** el enfriamiento H₂ no actúa sobre
  la estructura en este modelo. Hallazgo, no fracaso.

## Lo que NO cierro / queda honestamente abierto

- Las dos constantes (p:n, masa p/e) no se testearon — quedarían para si algún día se barre
  `tasa_expansion` o se le suma energía de ligadura a la masa bariónica (hoy usa masas
  desnudas de quark, por eso p/e da 18 en vez de 1836 — falta el ~99% que es ligadura).
- El enfriamiento H₂: por qué no actúa (¿pocos pasos? ¿presión térmica dominante?) queda
  como pista, no resuelto.

No sello el arco ni instruyo nada hasta que me digas. La regla de no cerrar sin tu
autorización sigue intacta.
