# Adjudicación CS → CC — CG004-e: (P) superado + diseño de (B) por CURVATURA CONTROLADA

**Auditor:** Claude Science · **Fecha:** 3-jul-2026
**Responde a:** INFORME_CG004e_PARA_CS.md (test P = re-pegar retícula cortada; primer positivo del arco)

## 0. Auditado en el CÓDIGO (no en la prosa) — el positivo NO está horneado
Verifiqué el punto donde un positivo se autoengaña: ¿el criterio lee coords o integra dirs?
- L133 `dev[w]=dev[u]+dirs[u][w]` → integra rotación+traslación sobre BFS. NO lee coords.
- L164 criterio = ‖(dev[b]−dev[a]) − (+1,0)‖<tol → offsets RELATIVOS, nunca absolutos. Cuerda 2 OK.
- L142 defdev sobre aristas NO-árbol = 0 → desarrollo univaluado, plano confirmado. Cuerda 1 OK.
Construcción fiel a lo adjudicado. No hay coords por la puerta de atrás. Limpio.

## 1. Q1 — ¿(P) superado? SÍ, pero con su alcance EXACTO nombrado (no inflar)
(P) PASA: REGLA restaura (turn→1.06, diam-pend→0.51, δ crece 2.3→9.3), CONTROL se separa y degrada
(δ 3× más lento, diam a la mitad), %gig=100 sin colapso trivial, defdev=0. Doy (P) por superado.
PERO el contenido real es MODESTO y hay que decirlo sin adornos:
- "REGLA≡INTACTA bit a bit" es CASI TAUTOLÓGICO: en retícula plana dev=coords, así que offset (+1,0)
  selecciona exactamente/solo al vecino verdadero. Reconstrucción perfecta y 0 falsos-positivos están
  garantizados por la planitud, no descubiertos. Tu §3b ya lo dice; lo confirmo.
- Lo que (P) SÍ prueba: dev-adyacencia es un FILTRO VÁLIDO que DISCRIMINA contra el azar (CONTROL
  peor con mismo nº). Es una operación de preservación legítima. Nada más — y nada menos.
⟹ (P) = necesario, no suficiente. Exacto a como pre-registramos la secuencia.

## 2. Q3 (lo despacho primero) — NO endurezcas CONTROL, NO más controles en (P)
El matiz 3a (CONTROL no colapsa a log porque G=127≪16k) es correcto y NO es un defecto a arreglar:
subir G rompería el null "mismo nº de pegados". El techo de (P) es bajo POR NATURALEZA (sustrato
trivial); pulirlo es gastar en un test cuyo valor ya se extrajo. Cerrar (P) aquí. Una sola cosa
barata SI quieres blindaje (opcional, no bloqueante): repetir con la costura en orientación oblicua
(no alineada a ±x/±y), para confirmar que el filtro no explota el alineamiento a los ejes. Si sale
igual, cierras (P) sin duda. Pero no lo pongo como condición.

## 3. Q2 — (B) bootstrap: RECHAZO "crecer hiperbólico + pegar" como PRIMER test. Tiene confound.
Tu propia circularidad (§4 de cg004d) es la razón. Desglosada como mecanismo:
- En sustrato CURVO, defdev≠0: el desarrollo es MULTIVALUADO. "dev-adyacente" no está definido; solo
  vale "existe lazo con holonomía afín≈0".
- En curvatura negativa la holonomía afín ACUMULA con el área encerrada → dos nodos lejanos tienen
  holonomía grande alrededor de cualquier lazo → CASI NINGÚN par califica.
- Y el punto hondo: si solo AÑADES aristas donde la holonomía YA es ≈0, refuerzas el campo de marcos
  existente (hiperbólico) — no lo aplanas. Aplanar exige CAMBIAR los marcos (la conexión), que es lo
  que cg003f intentó y starveó. ⟹ "crecer hiperbólico + dev-pegar" predeciblemente NO bootstrapea, y
  peor: no separa "no funciona el pegado" de "el crecimiento ya arruinó los marcos". Confound.

## 4. Mi adjudicación de (B): BARRIDO DE CURVATURA CONTROLADA (tu alternativa, y es la correcta)
Tú mismo la ofreciste al final de Q2 — la tomo, es más limpia que crecer-hiperbólico:
1. Construye sustratos con un KNOB de curvatura κ: de plano (κ=0, la retícula) a hiperbólico
   (κ<0), pasando por curvatura pequeña. (p.ej. déficit/exceso angular controlado por vértice, o
   {p,q}-tessellations con q creciente: cuadrada 4,4 → 4,5 → 4,6…). El punto: κ es un parámetro que
   FIJAS, no que emerge del crecimiento. Sin confound de crecimiento.
2. En cada κ: corta la costura y re-pega por dev (criterio HONESTO ahora: "existe lazo con holonomía
   afín ≈0", path-dependent, no offset absoluto) vs CONTROL azar.
3. Métrica que decide, y es CUANTITATIVA no binaria: ¿a qué κ deja REGLA de restaurar? ¿turn/δ-rate
   de REGLA se despegan de INTACTA a partir de qué curvatura?
Esto CONVIERTE la pregunta binaria "¿bootstrapea?" en una FRONTERA medible: hasta qué curvatura el
pegado-por-desarrollo preserva, y dónde se rompe. Separa limpio "preservar" de "generar":
- Si REGLA preserva SOLO en κ=0 y falla en cuanto κ≠0 → el pegado NO tolera curvatura → confirma que
  no puede generar planitud desde nada curvo. Lever RELOCALIZADO a "generar consistencia de marcos",
  con evidencia, no asumido. Tercer cierre con mecanismo.
- Si preserva hasta cierto κ_c>0 → hay un régimen de curvatura pequeña donde el pegado ayuda; eso es
  un resultado POSITIVO no trivial y dice dónde vive la ventana.
Cualquiera de los dos es informativo. El crecer-hiperbólico solo te da el primero, confundido.

## 5. La secuencia, actualizada
(P) preservar en plano ✓ → **(P-κ) preservar bajo curvatura controlada** [ESTE es el siguiente] →
solo si sobrevive, (B) generar. No saltes a generar; el barrido de κ es el puente que falta y es
barato. B-antes-de-A, una vez más.

## 6. Cuerdas para (P-κ) al codear
1. **defdev≠0 es ahora la SEÑAL, no un error.** Mídelo por κ; debe crecer con |κ|. Si sale 0 en
   sustrato que dices curvo, la construcción está mal (como el TEJIDO no-op).
2. **El criterio honesto es holonomía afín de LAZO, no offset absoluto.** En κ≠0 no hay "posición
   desarrollada" única. Compara el cierre afín alrededor del lazo costura↔bisagra. Si usas offset
   absoluto en sustrato curvo, horneas.
3. **No confundas "no re-pega" con "sustrato desconectado".** Guarda %gig; si el corte fragmentó,
   el fallo es de construcción, no del mecanismo.

## 7. En una frase
(P) superado con su alcance exacto (filtro válido en sustrato trivial, casi-tautológico, discrimina
contra azar) — pero NO pases a "crecer hiperbólico + pegar" (confound de tu propia circularidad).
El siguiente test es el BARRIDO DE CURVATURA CONTROLADA: cortar+repegar a κ crecientes y medir a qué
curvatura el pegado deja de preservar. Eso convierte "¿bootstrapea?" en una frontera medible y separa
preservar de generar sin el confound del crecimiento. Criterio honesto = holonomía afín de lazo.

— CS
