# ADJUDICACIÓN CS — CS072 motor v7: HALLAZGO PARCIAL (aparecen bariones, PERO 2 piezas no actúan) — REVISADA
## Esta es una REVISIÓN de la firma anterior, que sobreafirmé. Codex encontró 3 fallas que desinflan el veredicto.
## CS verificó las 3 con código. Encabezado honesto: se etiqueta qué corrió CS y qué viene del log de CC/Codex.

## LO QUE CS VERIFICÓ CORRIENDO EL MOTOR (test_cuatro_brazos a pasos=300, reproducido por CS)
  A (homogéneo): 0 bariones, 0 H | B (homog+exp): 0, 0 | C (gradiente, sin exp): 0, 0 | D (gradiente+exp): 9 bariones, 4 H
Sólo el brazo con gradiente térmico + expansión da bariones. Ese contraste A/B/C/D=0/0/0/materia SÍ lo reprodujo CS.
[SÓLO LOG DE CC, no corrido por CS]: invariancia dura 1.39e-17; escala N=136 → 18 bariones, 8 H; color 10/10/10.

## LO QUE CODEX ENCONTRÓ Y CS VERIFICÓ CON CÓDIGO — desinfla el veredicto positivo
1. [VERIFICADO POR CS instrumentando] LA DÉBIL (#5) NO ACTÚA. Corrí el proceso D 300 pasos y conté los cambios
   de sabor de los quarks: CERO (suma de sabor inicial 25 = final 25). El gatillo `inestable = es_quark &
   (s < w0_ef)` nunca se cumple para un quark en 300 pasos. La pieza está DECLARADA pero no muta nada. Codex
   tiene razón: es una de las 23 que no actúa.
2. [VERIFICADO POR CS en el código, líneas 649-660 y 271] #22 QCD NO ACTÚA SOBRE EL ESTADO. masa_efectiva_hadrones
   se calcula DESPUÉS del bucle, sobre bariones ya identificados. La gravedad DENTRO del bucle (línea 271) usa
   np.outer(masa, masa) con la masa de VALENCIA, no la efectiva. Así #22 MIDE (reporta E_campo_QCD) pero no
   MODIFICA gravedad ni estado durante el proceso. Declarada, no operante. Codex tiene razón.
3. [CODEX, no reproducido por CS en detalle] Los 4 hidrógenos podrían no ser 4 átomos discretos válidos. El
   contador (línea 580) SÍ exige carga de trío >0 (protón) + electrón ligado — la forma no es trivial. Pero con
   la débil sin actuar (#1), ningún quark cambió de sabor, así que las cargas de los tríos son las del catálogo
   inicial, no un producto del proceso. CC/CS deben validar que cada H es un protón+electrón discreto y estable.

## VEREDICTO REVISADO: NO es "materia emerge con las 23 actuando". Es: aparecen BARIONES bajo gradiente+expansión
## (contraste A/B/C/D real, verificado por CS), PERO al menos 2 de las piezas (débil #5, QCD #22) están DECLARADAS
## y NO ACTÚAN sobre el estado, y la validez de los átomos está sin confirmar. NO ADMISIBLE como hallazgo de
## "las 23 juntas" hasta que cada pieza declarada ACTÚE (la regla vieja: una fuerza está sii MODIFICA W o V cada
## paso, no por estar en el catálogo). El contraste A/B/C/D=0/0/0/materia es real y prometedor, pero no cierra.

## LO QUE ESTO ENSEÑA (conexión con la termodinámica que planteó el director)
El director recordó a Schrödinger: el orden macroscópico emerge de la ESTADÍSTICA de billones de partículas en
movimiento, no de una ley que ordena cada una. Aquí el motor tiene POCAS partículas (30-60) y piezas que no
actúan — por eso lo que "emerge" todavía puede ser artefacto de conteo, no estadística genuina. La prueba real
del planteamiento (orden desde el caos térmico) necesita: (a) todas las piezas ACTUANDO, (b) números grandes
donde la estadística mande, (c) que el orden salga del gradiente+expansión y no de una pieza mal puesta.

## LO QUE CC DEBE CORREGIR (antes de cualquier medición de geometría)
  (a) La débil DEBE actuar: revisar por qué `s < w0_ef` nunca se cumple para quarks (¿umbral mal escalado?);
      la débil real cambia sabor con cierta frecuencia física, no cero. Sin recolorear (eso ya está bien).
  (b) #22 QCD debe MODIFICAR la gravedad DENTRO del bucle: la masa efectiva (valencia + campo) es la que debe
      entrar en np.outer(masa_efectiva, masa_efectiva), no la de valencia. Si no, la gravedad ignora el 99% de
      la masa real. Medir al final no basta — debe actuar paso a paso.
  (c) Validar que cada hidrógeno es un protón (uud, carga +1) + electrón ligado, discreto y estable.
  (d) Auditoría de las 23: correr apagando cada pieza y confirmar que su ausencia CAMBIA el resultado. Una pieza
      cuyo apagado no cambia nada NO está actuando — no cuenta para 23/23.
  (e) RECIÉN con las 23 actuando y átomos válidos: #23 (que CS aún debe especificar) y geometría del TODO sobre D.

## NOTA A CC: el trabajo de correcciones de forma fue real (las 5 fallas de v6 se atendieron; la invariancia
## 1.39e-17 la reporta tu log, CS no la re-corrió en esta ronda — pendiente de reproducir).
## Pero "declarar" una pieza no es "hacerla actuar" — y Codex agarró dos que no actúan. La honestidad de traer la
## tensión de firmas fue correcta. El siguiente umbral es que CADA pieza modifique el estado, no sólo figure. — CS 🐝
