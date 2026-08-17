# DISEÑO CS071 — HISTÉRESIS / MEMORIA-DE-ENLACE
## ¿La asimetría que fabrica el PROCESO (transitar refuerza, no-usado decae) auto-organiza un tejido métrico donde ni la sopa simétrica ni la semilla pudieron?
### Diseño: CS · 17-jul-2026 · Ejecuta: CC · Etiqueta técnica: batería topológica Gemini, Test 2.2

---

## 0. DÓNDE ENCAJA (por qué este número y no otro)
El arco del espacio converge por TRES rutas al mismo muro (Mundo B): clásico sin-semilla (CS066-068), cuántico
(CS069), clásico con-semilla-de-condición-inicial (CS070). Las tres inyectan la asimetría DESDE FUERA —como
estructura permanente, como fase, o como semilla inicial— y el mundo-pequeño la lava. **CS071 prueba la única
fuente de asimetría que las tres anteriores NO tocaron: la que fabrica el PROPIO proceso dinámico.** No hay
semilla ni estructura privilegiada al inicio (todos los enlaces valen igual); la asimetría, si aparece, nace de
que transitar un enlace lo refuerza y no-usarlo lo hace decaer. Es ruptura de simetría AUTO-organizada. Es el
candidato más original de la batería de Gemial y el único mecanismo genuinamente nuevo respecto del arco.

En lenguaje de la Teoría: la histéresis es **memoria** (κ_H) actuando sobre la propia topología. La pregunta es
si κ_H aplicado al sustrato relacional puede hacer emerger la MÉTRICA (κ_O/dirección) que la relación pura no
tiene. Si sí, sería el primer mecanismo del arco que rompe el muro sin ingredientes prestados de la física.

---

## 1. LA PREGUNTA, AFILADA
Sustrato inicial: mundo-pequeño (Watts-Strogatz, el mismo blob-equivalente de todo el arco), enlaces uniformes.
Proceso: paseantes recorren el grafo; los enlaces transitados se refuerzan (peso↑), los no transitados decaen
(peso↓); enlaces por debajo de un umbral se podan. El peso sesga el próximo paso (los reforzados se prefieren).
**¿El grafo se auto-poda hacia MÉTRICO (diámetro ~√N, sin atajos, dimensión finita) o se queda mundo-pequeño
(diámetro ~log N) / colapsa a hub (diámetro ~2)?**

Métrico = habría emergido "lejos" real por uso. Mundo-pequeño = la memoria no fabrica métrica. Hub = degeneración
(el análogo del colapso a 1 eje de CS067; NO cuenta como éxito).

---

## 2. LO QUE EL TOY YA MOSTRÓ (pre-registración honesta — CS lo corrió antes de diseñar)
Corrí el mecanismo en 3 regímenes sobre mundo-pequeño N=400/900/1600 (diám crudo 14/16/17; métrico sería ~36/50/66):
- **Refuerzo blando (decay=0.85):** no poda nada (aristas 1.00×), diám 13/16/16, atajos ~100% sobreviven
  (82/82, 179/180, 326/326 para N=400/900/1600).
- **Refuerzo duro (decay=0.5, poda=0.25):** COLAPSA a hub — diám=2, 1% de aristas, todo vía un centro. Degenerado.
- **Homeostático (escalado sináptico, presupuesto por nodo, ciego a geometría):** diám 12/16/16, atajos 100%
  sobreviven, aristas 1.00×. Tampoco metriciza.
**Mecanismo medido de por qué:** la intermediación de arista de un paseo ciego CARGA los atajos 3.9× sobre los
enlaces locales (atajos media 0.0288 vs locales 0.0074, N=400). Un refuerzo-por-uso refuerza JUSTO los atajos que
habría que podar para volverse métrico. El proceso empuja en contra de la métrica, no a favor.

**Predicción pre-registrada de CS: (B) — la histéresis ciega NO metriciza.** Se registra ANTES de la tanda para
que el resultado sea falsable en ambos sentidos. Si CC obtiene diám ~√N en el brazo real y NO en el NULL, sería
un (A) que refutaría mi predicción — y ese sería el hallazgo grande del arco. La tanda blindada existe para darle
esa oportunidad limpia, no para confirmar el toy.

---

## 3. LOS BRAZOS (mínimos, aislados — regla Pauli CS065b: aislar antes de cruzar)
1. **HISTÉRESIS** — proceso real sobre mundo-pequeño. Variante canónica: **homeostática** (la más interesante y
   la única con motivación independiente —escalado sináptico— que no es solo "subir/bajar un peso"). Ciega a
   geometría por construcción.
2. **NULL_BARAJADO** — MISMA magnitud de poda/refuerzo por paso que el brazo real (mismo nº de enlaces tocados,
   misma distribución de Δpeso), pero QUÉ enlaces se tocan se sortea al azar, destruyendo la correlación con el
   tráfico real. Es el control anti-Shannon central: si HISTÉRESIS ≈ NULL_BARAJADO en el juez, el proceso no
   aporta nada métrico —solo adelgaza el grafo—.
3. **SIN_PROCESO** — blob crudo, sin dinámica. Ancla del "mundo-pequeño de siempre" (diám log N).
4. **HISTÉRESIS_SOBRE_RETÍCULA** — control POSITIVO: el mismo proceso sobre una retícula limpia (métrica de
   verdad). Dos lecturas: (a) si PRESERVA √N → el proceso es al menos métrico-neutral y el (B) del brazo 1 es
   "no construye", no "destruye"; (b) si DESTRUYE la métrica (mete atajos / colapsa) → el proceso es
   activamente anti-métrico, lectura aún más fuerte.

Tanda: 4 brazos × ≥8 semillas × 3 tamaños (N=400/900/1600), como CS069/CS070. Diám por BFS multi-fuente (≥8 fuentes).

---

## 4. EL JUEZ (anti-Shannon, hereda de CS067 y CS070)
**Discriminante = ESCALAMIENTO DEL DIÁMETRO con N**, ajustado a ley de potencia diám ~ N^β:
- β ≈ 0.5 (√N) y δ-Gromov creciente → **MÉTRICO** (veredicto A candidato).
- β ≈ 0 (log N) → **mundo-pequeño** (veredicto B).
- diám → 2-3 con un nodo de grado gigante → **HUB colapsado** (degeneración, NO es A).
Se mide sobre 3 tamaños; un solo N no decide (lección CS068 paso 2b: el escalamiento es el juez, no el valor).

**NO se juzga por coherencia ni por "cuántos atajos se podaron"** (trampa CS070: podar mucho ≠ volverse métrico;
el blob podado de CS070 seguía siendo log N). Solo cuenta cómo escala la distancia real.

---

## 5. GUARDIANES (preinscritos — el hueco de Shannon se caza ANTES, no después)
- **G-PASEO-CIEGO:** la regla de tránsito usa SOLO pesos de enlace actuales y grado de nodo. NUNCA distancia de
  anillo, coordenada, ni ninguna etiqueta que codifique la geometría-objetivo. (Si el paseante "supiera" qué
  enlace es atajo, estaríamos imponiendo la métrica = Shannon.) CC verifica en el código que la función de
  transición no recibe posición.
- **G-NULL-MISMA-MAGNITUD:** el NULL_BARAJADO poda/refuerza EL MISMO nº de enlaces por paso y la misma
  distribución de Δpeso que el brazo real; solo aleatoriza CUÁLES. Sin esto el NULL sería un grafo más denso o
  más ralo y la comparación mentiría.
- **G-NO-AJUSTAR-CRONOGRAMA:** decay, umbral de poda y nº de pasos son IDÉNTICOS en los 4 brazos y se fijan ANTES
  de ver diámetros. Prohibido tunear por-brazo para acercarse a √N (eso sería hornear la métrica en el
  cronograma). Si hace falta explorar régimen, se hace en una corrida EXPLORATORIA separada y declarada, no en la
  tanda de veredicto.
- **G-ANTI-HUB (nuevo, análogo a n_ejes>1 de CS070):** un diámetro bajo NO cuenta como métrico si viene de
  colapso a hub. Criterio combinado: MÉTRICO = (β≈0.5) Y (grado máximo acotado, p.ej. < 3× el grado medio inicial)
  Y (δ-Gromov crece con N). El colapso a hub (diám 2, grado gigante) se marca explícitamente como degeneración,
  igual que el pico-sin-ejes de CS070.
- **G-CONECTIVIDAD:** medir siempre sobre la componente gigante y reportar qué fracción del grafo sobrevive; una
  "métrica" sobre el 5% del grafo no es métrica del universo.

---

## 6. LECTURA PRE-INSCRITA (los tres desenlaces, firmados antes de correr)
- **(A) diám ~√N solo en HISTÉRESIS, no en NULL, sin hub:** la memoria FABRICA métrica. Refutaría mi predicción y
  sería el primer mecanismo del arco que rompe el muro por auto-organización. Habría que replicar y estresar
  (¿√N estable a N=3200? ¿δ-Gromov crece? ¿dimensión finita?) antes de cantar victoria.
- **(B) HISTÉRESIS ≈ NULL ≈ SIN_PROCESO en log N:** la memoria no metriciza. Cuarta ruta al mismo muro, ahora
  cerrando también la asimetría auto-organizada. El muro ACOTA una vez más. Consistente con el toy y con el
  mecanismo medido (el tránsito carga los atajos).
- **(C) HISTÉRESIS colapsa a hub, NULL no:** el proceso hace ALGO real (se distingue del azar) pero en la
  dirección equivocada —concentra en vez de distribuir—. Sería un positivo de mecanismo con veredicto de
  geometría negativo, como el CS054-v2 (gravedad selecciona, pero 2D no 3D). Se reporta como tal, no se disfraza.

---

## 7. LO QUE CS071 NO ES (para no repetir errores del arco)
- No es CS070: aquí NO hay semilla ni asimetría inicial; el grafo arranca uniforme. La asimetría, si existe, es
  emergente del proceso. Complementario, no solapado.
- No es Bloque 1 de la batería (asimetría estructural PERMANENTE inyectada): aquí la estructura es plástica y la
  fabrica el uso.
- No se le mete un ingrediente físico (carga/color/espín) para que "salga" métrica: eso es Bloque 3, aplazado y
  con su propio guardián. CS071 aísla la histéresis pura primero.

---

## 8. RESUMEN EN UNA LÍNEA
CS071 pregunta si la MEMORIA del proceso (transitar refuerza, olvidar poda) auto-organiza una métrica donde la
relación pura, la superposición cuántica y la semilla primordial fracasaron. El toy y el mecanismo medido
(tránsito ciego carga los atajos 3.9×) predicen (B); la tanda blindada de 4 brazos con juez de escalamiento de
diámetro, NULL de misma magnitud y guardián anti-hub existe para darle a (A) una oportunidad limpia de refutarme.

— CS 🐝
