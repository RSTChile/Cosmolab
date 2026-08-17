# ANÁLISIS — "Minimal decoherence from inflation" (Burgess et al. 2023) y el resumen recibido

**Analiza:** Claude Science · **Director:** Alexis López Tapia · 27-jul-2026
**Fuente verificada:** paper completo descargado (83 páginas), DOI 10.1088/1475-7516/2023/07/022,
arXiv:2211.11046v2 (10 jul 2023). Todas las citas de página son de ese PDF.

---

## 1. Lo que el resumen acierta

El paper existe, los datos bibliográficos son correctos y el grueso del contenido está
bien descrito. Verificado línea por línea contra el original:

- **Autores, revista, año, arXiv:** correctos.
- **El planteo:** sí, trata los modos super-Hubble como sistema cuántico abierto, con los
  modos de longitud de onda más corta (escalares y tensoriales) como entorno.
- **Interacción mínima:** correcto y es literal del resumen del paper — solo las
  autointeracciones que predice la Relatividad General en modelos de un solo reloj, y los
  autores anotan que canales adicionales solo acelerarían la decoherencia.
- **Método:** Open EFT + ecuación de Lindblad, con la derivación controlada a tiempos
  tardíos donde el cálculo perturbativo se rompe. Correcto.
- **El umbral de 5×10⁹ GeV:** correcto. El paper dice 5,2×10⁹ GeV en el cuerpo (p.28) y
  ~5×10⁹ en el resumen, con su equivalente r > 6,5×10⁻²⁸.
- **Los ~13 e-folds a escala GUT:** correcto. El cuerpo precisa ≃12,9 e-folds (p.30).
- **El crecimiento (aH/k)³:** correcto, es la ecuación 4.17 (p.27).

Hasta aquí, el resumen es fiel.

---

## 2. Un error de fórmula, verificado

El resumen dice:

> p_k(η) = 1 / (1 + Ξ(η))

**La ecuación 4.12 del paper (p.26) es:**

> p_k(η) = 1 / **√**(1 + Ξ_k(η))

Falta la raíz cuadrada. No es una errata cosmética: cambia el ritmo con que cae la pureza.
Calculado explícitamente sobre el régimen asintótico (Ξ ∝ a³, ajuste log-log sobre 40
puntos):

| fórmula | pendiente d(ln p)/d(ln a) | exponente resultante |
|---|---|---|
| **la del paper**, 1/√(1+Ξ) | −1,4986 | **p = 3/2** |
| la del resumen, 1/(1+Ξ) | −2,9972 | p = 3 |

**El p = 3/2 obtenido con la fórmula correcta es exactamente el valor que el paper cita de
sí mismo en la página 33** ("we have found p = 3/2 in the case of gravitational
decoherence"). Con la fórmula del resumen daría 3, y no coincidiría con el propio texto.
Es decir: la verificación aritmética confirma que la raíz va, y que el resumen la perdió.

Consecuencia práctica: **el resumen hace que la pureza caiga el doble de rápido** (en el
exponente) de lo que el paper calcula. Sobrestima la clasicalización.

---

## 3. Lo importante: el resumen omite el pasaje que le da vuelta la conclusión

El resumen cierra afirmando que el paper es *"el análogo cosmológico más limpio y
minimalista disponible"* del proceso descrito, y que *"la expansión + gravedad son
suficientes"* para la emergencia de grados de libertad diferenciados y clásicos.

**El paper dice algo notablemente más restringido en la página 33:**

> "although decoherence is very effective, the erasure of quantum discord is not, which
> might still leave open the possibility to detect quantum signatures."

El razonamiento del paper es este: la cantidad de decoherencia necesaria para borrar una
*característica cuántica concreta* varía según la característica. Citando la ref.[22], un
estado decoherido en un universo de De Sitter **conserva discordia cuántica grande si
p < 4**. Y el paper acaba de encontrar p = 3/2.

Verificado numéricamente: 3/2 < 4, y también 3 < 4. La conclusión cualitativa sobrevive
aun con el error de la raíz, pero el margen cambia mucho — 3/2 está muy por debajo del
umbral, no al borde.

**Traducido a lenguaje llano:** el paper no demuestra que la expansión más la gravedad
basten para producir clasicalidad. Demuestra que bastan para producir **decoherencia** —
que es otra cosa. Su propio resultado implica que un rastro cuántico medible (la
discordia) **sobrevive** al proceso. El paper llega a decir que eso deja abierta la
posibilidad de detectar firmas cuánticas: lo trata como una oportunidad observacional, no
como un residuo despreciable.

El resumen invierte el signo de ese matiz.

---

## 4. Otras omisiones que importan

**4.1 La sección "Loopholes" (§5.4, p.31-33) existe y es sustancial.** Los propios autores
enumeran los supuestos que tendrían que romperse, y uno es directamente pertinente:

> "Perhaps additional interactions can re-cohere initially decohered states. Inquiring
> minds need to know."

Es decir, los autores dejan abierto que interacciones adicionales **vuelvan a coherentar**
estados ya decoheridos. El resumen afirma que "no hay parámetros libres que se ajusten" —
cierto — pero convierte eso en universalidad, y los autores explícitamente no lo hacen.

**4.2 Hay una pregunta abierta que los autores admiten no haber probado** (p.33): sobre la
imposibilidad de la auto-purificación espontánea escriben "this should be possible to
prove, and we have not yet done so". Es una honestidad del paper que el resumen no recoge.

**4.3 El prefactor es minúsculo.** El paper subraya (p.5, p.30) que la amplitud arranca en
ε₁H²/(8πM_p²) ~ 10⁻¹⁴ en modelos de un solo campo. Lo que salva el resultado no es la
fuerza del efecto sino el crecimiento exponencial e^{3Ht} a lo largo de 40-60 e-folds. El
resumen dice "la decoherencia es eficiente" sin mencionar que **depende de que la
inflación dure lo suficiente**: es un resultado sobre acumulación en el tiempo, no sobre
una interacción intrínsecamente potente.

**4.4 El resumen dice "el estado global sigue siendo matemáticamente no separable".** Es
una afirmación razonable sobre decoherencia en general, pero **no la encontré formulada
así en el paper** — busqué "global state", "still entangled", "remains quantum", "for all
practical purposes", "FAPP": cero apariciones. El paper habla de pureza y de discordia,
no de separabilidad del estado global. La afirmación puede ser correcta por física
general, pero no está respaldada por esta fuente.

---

## 5. Veredicto

**El paper es real, serio y está bien resumido en su mecánica.** El resumen falla en dos
puntos, uno técnico y uno interpretativo:

1. **Error de fórmula:** falta la raíz cuadrada en la pureza (eq. 4.12). Verificado
   aritméticamente: con la raíz sale p = 3/2, el valor que el propio paper cita; sin ella
   saldría 3.
2. **Sesgo de omisión, que es el grave:** el resumen presenta como conclusión
   ("expansión + gravedad son suficientes") algo que el paper matiza en dirección
   contraria — la decoherencia es eficiente, **pero la discordia cuántica no se borra**, y
   los autores dedican una sección entera a los supuestos que podrían fallar.

**Para el uso que se le quiere dar:** este paper **no** sirve como demostración de que la
expansión más la gravedad producen clasicalidad plena. Sirve para algo más preciso y
todavía valioso: demuestra que **la expansión y la gravedad, sin ningún parámetro
ajustado, bastan para hacer que un estado cuántico puro pierda pureza de forma eficiente
y calculable.** Eso es un resultado fuerte. Pero el propio paper sostiene que el residuo
cuántico permanece y podría llegar a medirse.

Si se lo va a citar en la Canónica, la afirmación defendible es la segunda, no la primera.
La diferencia entre "decoherencia" y "clasicalidad" es exactamente el tipo de deslizamiento
que la regla anti-Shannon existe para atrapar: aquí el NULL, por así decirlo, es la
discordia que sobrevive — y no fue vencida.

---

*Verificado en disco: PDF completo (83 páginas) descargado vía DOI y leído
programáticamente; ecuaciones 4.12 (p.26), 4.17 (p.27), 1.2 (p.5); secciones 5.3 Open
questions y 5.4 Loopholes (p.30-33); umbral 5,2×10⁹ GeV (p.28); 12,9 e-folds (p.30).
El exponente p = 3/2 fue recalculado, no copiado.*
