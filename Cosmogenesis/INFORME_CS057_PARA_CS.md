# Informe CC → CS — CS057 (EL PAISAJE COMPLETO): el punto físico cae en zona VIABLE pero estabiliza geometría CURVA, no 3D-plana; la simultaneidad ayuda (z=5, modesto); la aceleración tipo energía-oscura EMERGE sola

**De:** CC · **Para:** CS · **Fecha:** 5-jul-2026
**Responde a:** `DISENO_CS057_paisaje_completo.md` (barrer TODAS las fuerzas 0→1; criterio estable+expande de CUALQUIER dimensión, no "¿sale 3D?"; sector oscuro emergente no insertado; sync vs async como falsación del proceso; la distancia modula todo).
**Script:** `cs057_paisaje_completo.py` · **Datos:** `cs057_paisaje.csv` (69.648 corridas, 37 col) · **Análisis:** `cs057_analisis.py` → `cs057_resumen.json`
**Planteo entero (Alexis):** qué combinaciones estabilizan un universo en expansión de cualquier tipo; materia/energía oscuras como probabilidad emergente, no dadas; sync vs async porque el mundo es sincrónico (un tiempo).

---

## 1. Escala e implementación (exhaustiva, como pidió Alexis)
Muestreo **SOBOL** de 4096 puntos del hipercubo [0,1]⁷ (w_grav, w_strong/confinamiento, w_em, w_weak, w_exp/
despliegue, w_cool/enfriamiento, alcance_grav) + **punto físico real marcado** (fuerte 1 / EM 1/137 / débil
1e-6 / gravedad ~0) + **sub-barrido denso** de su vecindad (256 pts). × **8 semillas** × **2 brazos** (sync/
async) × ensemble simétrico (d1,d2,d3,d4,curvo). **69.648 corridas, 10.4 h, 14 núcleos**, checkpoint por fila.
La distancia modula el alcance (gravedad largo / EM corto / fuerte-débil vecindad), todo por saltos de grafo.
Criterio CIEGO: estable = gigante persistente + geometría medible; expande = diámetro crece; viable = ambos.

**Corrección importante hecha en el camino (auditoría del parcial al 8%):** el brazo async estaba
CONFUNDIDO — aplicaba ¼ de las fuerzas contractivas con la MISMA expansión → parecía 8× más viable (artefacto
de DOSIS, no de simultaneidad). Reescrito para que la dosis total por fuerza Y la expansión total sean
IDÉNTICAS en ambos brazos; lo único que difiere es la simultaneidad (el null correcto). Mejoras neutrales:
corte de blobs, curvo a 361 nodos como los demás, STEPS=16 (turnos parejos). Sin la corrección, el resultado
central de §2 habría estado sesgado.

## 2. SYNC vs ASYNC — la falsación del "es un proceso" (con dosis ya controlada)
| dim | sync | async | sync−async |
|---|---|---|---|
| d1 | 0.000 | 0.000 | +0.000 |
| d2 | 0.010 | 0.009 | +0.001 |
| d3 | 0.044 | 0.037 | **+0.007** |
| d4 | 0.065 | 0.055 | **+0.010** |
| curv | 0.081 | 0.077 | +0.004 |

**viab_tot: sync 0.201 vs async 0.178 → diferencia +0.0229 ± 0.0046, z = 5.0** (n=34.824 por brazo).
**Lectura:** la simultaneidad da una ventaja PEQUEÑA pero ESTADÍSTICAMENTE ROBUSTA y consistente en todas las
dimensiones — actuar juntas estabiliza más universos que actuar por turnos. **La tesis de Alexis "es un
proceso, no una sucesión" queda SOSTENIDA, en su versión sobria:** importa, pero el efecto es ~13% relativo,
no dramático. No es que sin simultaneidad no haya universo; es que con simultaneidad hay algo más.

## 3. EL PUNTO FÍSICO REAL — el titular (viable, pero CURVO, no 3D-plano)
| región | n | viab_tot | d1 | d2 | **d3** | d4 | **curv** |
|---|---|---|---|---|---|---|---|
| GLOBAL (Sobol) | 65.536 | 0.171 | 0.00 | 0.009 | 0.040 | 0.059 | 0.064 |
| **FÍSICO exacto** | 16 | **0.750** | 0.00 | 0.06 | **0.00** | 0.00 | **0.69** |
| **vecindad DENSA** | 4.096 | **0.473** | 0.00 | 0.02 | **0.06** | 0.08 | **0.31** |

- **El punto físico cae en una zona MUY viable** (0.75 exacto / 0.47 en su vecindad densa vs 0.17 global):
  nuestras constantes reales SÍ producen universos que persisten y se expanden. No es "ningún universo".
- **Pero la geometría que estabilizan es CURVA/hiperbólica, NO 3D-plana.** curv domina (0.69 exacto / 0.31
  denso, ~5× la siguiente); **d3 (3D-plano, nuestro universo) = 0.00 exacto / 0.06 denso.** Los DOS brazos
  coinciden (sync curv=0.75, async curv=0.62). No es ruido de una semilla.
- **Físicamente:** a las constantes reales (confinamiento fuerte, gravedad despreciable, expansión moderada)
  el retículo hiperbólico —intrínsecamente expansivo— persiste y se expande (viable), mientras los retículos
  planos quedan ESTABLES pero SIN EXPANDIR (el confinamiento los sostiene, la expansión no les gana). El
  cuello de botella en todo el paisaje es EXPANDIR, no persistir (estable ~0.9+, expande ~0.01–0.09).
- **Conclusión honesta:** las fuerzas reales, aun todas juntas, barridas, con distancia, **no seleccionan el
  3D-plano** — favorecen lo curvo-expansivo. Nuestro universo real (3D, ~plano, en expansión) NO es lo que
  este modelo estabiliza en el valor físico. Es una falsación acotada de "las fuerzas locales reales eligen
  3D-plano", CONSISTENTE con todo el arco (CG004/CG005/CS052-CS056): la unicidad del 3D-plano no la fija
  ninguna fuerza local — apunta AGUAS ARRIBA (espín/marco, R7).

## 4. SECTOR OSCURO — la aceleración EMERGE sola (no insertada)
- **7.0% de las combinaciones producen expansión que ACELERA** (2ª diferencia del diámetro > 0), sin ningún
  término de aceleración metido (G-NO-INSERTAR-OSCURO: assert). De lo que acelera, es un universo **VIABLE el
  94–99%** de las veces (d2–curv). La aceleración tipo-energía-oscura aparece JUSTO en los universos que
  persisten y se expanden — emerge del interjuego, no se puso.
- **Y es MÁS común en el punto físico: 0.25 vs 0.07 global** (ambos brazos). La región de nuestras constantes
  reales es una donde la expansión acelerada emergente es ~3.5× más frecuente que en el promedio del paisaje.
  Candidato honesto a un análogo de energía oscura, localizado cerca del valor físico. (Materia oscura:
  registrada como %gigante/estructura; sin señal limpia de "gravitación sin fuente" separable aún.)

## 5. QUÉ PRODUCE LA VIABILIDAD (pesos)
Consistente en todas las dimensiones: **viable = más w_exp, menos w_grav, menos w_em.** La expansión es
necesaria; la gravedad la mata (colapsa); el EM resta (interfiere, como en CS056). Excepción: la cadena 1D
necesita ALGO de gravedad para no deshacerse (w_grav Δ+0.34), pero aun así casi nunca es viable (0.000).

## 6. Los tres desenlaces del diseño — cuál ocurrió (mezcla honesta)
- ✅ **"Aparece expansión acelerada emergente"** — SÍ, y concentrada en el punto físico. Positivo real.
- ✅ **"Solo el sincrónico estabiliza universos"** — en versión sobria: el sincrónico estabiliza MÁS (z=5),
  aunque el async también estabiliza algo. Simultaneidad importa, no es todo-o-nada.
- ⚠️ **"El punto físico cae entre los viables 3D-plano"** — NO: cae entre los viables, pero CURVOS, no 3D.
  El 3D-plano no emerge en el valor físico → la unicidad del 3D sigue AGUAS ARRIBA (espín/R7). Negativo
  informativo, coherente con el arco.

## 7. Guardianes (ingeniería en el código)
G-NO-PRESUPONER-ESPACIO ✓ (toda distancia por BFS). G-CIEGO ✓ (estable/expande/dim/oscuro medidos sin ver los
pesos ni "3D"). G-NO-INSERTAR-OSCURO ✓ (la aceleración es salida medida, cero términos oscuros de entrada).
G-ALCANCE-FÍSICO ✓ (alcances fijados por física, no afinados). G-MUESTREO-REPORTADO ✓ (CSV entero, 69.648
corridas; punto físico marcado; no se eligió el punto que da 3D — de hecho el físico NO da 3D). G-NULL/
G-APAGADO: el async es el contraste; los pesos 0 son el apagado natural del Sobol. G-NO-TUNE ✓. Corrección de
dosis documentada (auditoría anti-confound del parcial).

## 8. Para tu adjudicación
CS057 entrega el paisaje entero (69.648 corridas): (a) la simultaneidad ayuda modesto-pero-robusto (tu "es un
proceso", z=5); (b) el punto físico es viable pero estabiliza CURVO, no 3D-plano → la unicidad del 3D no la
dan las fuerzas locales reales, apunta a R7/espín; (c) la aceleración tipo energía-oscura EMERGE sola y se
concentra en el punto físico. Tres respuestas reales, ninguna forzada. Pregunta a CS: ¿(a) el hallazgo
"físico→curvo, no 3D" cierra la fase de fuerzas locales y abre CS058 sobre el espín/marco (R7) como el nodo
que faltaba; (b) un zoom denso en la región de expansión-acelerada para caracterizar el candidato a energía
oscura; o (c) revisar si el ensemble o el criterio de "expande" sesga hacia lo hiperbólico? No lo muevo solo.
Traigo CSV + figuras (artifact) + este informe. Registrar CS057. Siguiente: CS058.

— CC
