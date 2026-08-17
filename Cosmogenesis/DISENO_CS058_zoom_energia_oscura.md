# DISEÑO CS058 — ZOOM DENSO al candidato de energía oscura: ¿dónde vive la aceleración emergente, sobrevive a la resolución, y es frontera curva? (caracterizar o matar un positivo)

**Rama:** Cosmogénesis · **Nº:** CS058 (dimensión técnica: caracterización local de un hallazgo emergente).
**Diseño:** CS · **Planteo:** Alexis (los DOS experimentos importan) · **Fecha:** 5-jul-2026.
**Estado:** DISEÑO, a codear por CC. **Base:** CS057 (adjudicacion_CS057_CS.md) — la aceleración emergente
(2ª diferencia del diámetro > 0, SIN insertar término) apareció en el 7% global y 2.4× enriquecida cerca del
punto físico. Candidato honesto a energía oscura. CS058 lo caracteriza o lo mata.

---

## 0. LA PREGUNTA, EN UNA LÍNEA
CS057 vio que "algo que acelera solo" es más común cerca de las constantes reales. CS058 pregunta:
¿ese algo es REAL (sobrevive a más resolución y más semillas, tiene una región propia en el espacio de
fuerzas) o es ARTEFACTO (se disuelve cuando lo miramos de cerca)? Y si es real: ¿DÓNDE vive — y vive en la
frontera con lo curvo, lo que lo conectaría directo con R7?

## 1. POR QUÉ ESTE EXPERIMENTO ANTES (o junto a) R7
- Es un POSITIVO a medias. "Vimos algo" es peor que "lo caracterizamos" o "lo matamos". CS058 lo cierra.
- Puede ADELANTAR R7: si la aceleración vive en la frontera curva del espacio de fuerzas, informa
  directamente al marco (R7). Si se disuelve bajo densidad, era artefacto y no arrastramos un fantasma a R7.
- Es BARATO comparado con el paisaje: no re-barre 69k universos, hace un zoom local denso (horas).

## 2. QUÉ SE BARRE (zoom local, no paisaje nuevo)
- **Región de interés:** el entorno del punto físico + la(s) región(es) del hipercubo de CS057 donde
  `acelera=1` fue más frecuente. CC debe primero EXTRAER de cs057_paisaje.csv los centroides de las celdas
  con más aceleración (no elegir a mano — leer del CSV dónde se concentró).
- **Ejes a variar densamente:** los que CS057 marcó como NO despreciables para la aceleración — típicamente
  w_exp (expansión), w_cool (enfriamiento), alcance, y el eje gravedad↔EM. Malla DENSA local (p.ej. 24-32
  puntos por eje en un hipercubo reducido de ~3-4 ejes activos), no Sobol disperso.
- **Resolución subida:** más pasos temporales y N mayor por corrida que en el paisaje, porque la 2ª
  diferencia del diámetro (la señal de aceleración) es sensible a resolución temporal. Si la señal es real,
  se AFILA con más pasos; si es ruido de submuestreo, se DISUELVE.

## 3. LAS TRES PREGUNTAS DE CARACTERIZACIÓN (cada una con su medida)
1. **¿Sobrevive a la resolución?** Correr la MISMA región con 3 niveles de pasos temporales (p.ej. ×1, ×2,
   ×4) y 3 niveles de semillas. PREDICCIÓN pre-registrada (a escribir ANTES): si es real, la fracción
   `acelera` se ESTABILIZA o AFILA al subir pasos; si es artefacto, DECAE hacia 0. Reportar la curva
   acelera-vs-resolución.
2. **¿Tiene región propia?** Mapa de calor local de `acelera` sobre los ejes activos. PREDICCIÓN: si es real,
   hay una REGIÓN CONTIGUA de alta aceleración (no puntos dispersos al azar). Medir: tamaño y compacidad de
   la región (fracción de celdas contiguas vs dispersas).
3. **¿Es frontera curva?** Cruzar `acelera` con `viable_curv` y `viable_d3/d4` celda por celda. PREDICCIÓN
   (la que conecta con R7): la aceleración se concentra en la FRONTERA donde la geometría viable es curva,
   no en el interior 3D-plano. Medir: correlación espacial entre alta-aceleración y dominio-curvo.

## 4. GUARDIANES (anti-Shannon, la regla de Alexis)
1. **G-NO-INSERTAR-OSCURO (heredado de CS057, verificado):** la aceleración SIGUE siendo 2ª diferencia del
   diámetro > 0. NINGÚN término se llama "oscuro", ninguno se ajusta para producirla. Assert en código.
2. **G-REGION-DEL-DATO:** la región a densificar se LEE de cs057_paisaje.csv (donde acelera fue frecuente),
   NO se elige a mano para que salga bonito. El código debe extraer los centroides del CSV.
3. **G-FALSABLE-POR-RESOLUCION:** el brazo de resolución (×1/×2/×4) es la falsación directa. Si la señal no
   sobrevive a más pasos, se DECLARA artefacto y se reporta como tal. El experimento debe poder MATAR el
   candidato.
4. **G-CONTROL-NULL:** brazo NULL (aceleración medida sobre trayectorias barajadas temporalmente). La
   aceleración real debe COLAPSAR bajo barajado; si se sostiene, es artefacto de medición.
5. **G-PREDICCION-CIEGA:** las tres predicciones (§3) se escriben ANTES de correr.

## 5. LOS TRES DESENLACES (pre-escritos)
- **Sobrevive + región contigua + frontera curva → candidato a energía oscura CONFIRMADO y localizado**, y
  —bonus— apunta a R7 (vive donde la geometría es curva, justo el régimen que el marco debería tocar). Se
  caracteriza y se lleva a R7 como dato de entrada.
- **Sobrevive + región contigua PERO interior (no frontera) → candidato real pero desacoplado de R7.** Se
  registra como fenómeno propio (expansión acelerada emergente) para una línea aparte; no bloquea R7.
- **NO sobrevive a la resolución / se sostiene bajo NULL → ARTEFACTO.** Se declara, se documenta el umbral
  donde se disuelve, y se cierra el candidato. R7 arranca limpio, sin fantasma.

## 6. RESUMEN OPERATIVO PARA CC
- Extraer de cs057_paisaje.csv los centroides de alta-`acelera` (G-REGION-DEL-DATO). Definir el hipercubo
  local reducido (3-4 ejes activos) alrededor de ellos + el punto físico.
- Malla DENSA local (24-32 pts/eje) × brazo de resolución (×1/×2/×4 pasos) × semillas × brazo NULL.
- Medir las tres caracterizaciones (§3): curva acelera-vs-resolución, compacidad de la región, correlación
  con dominio-curvo. Predicciones ciegas escritas antes.
- Reusar el motor de CS057 (mismo criterio ciego, mismos guardianes). NO tocar la definición de acelera.
- Entregar CSV local + figuras (heatmap local, curva de resolución) + informe. Traer a CS. Registrar CS058.

— Diseño CS058 por CS. El planteo (caracterizar el candidato de energía oscura antes de/junto a R7) es de
Alexis. La estructura de caracterización, los guardianes y las falsaciones, mías. El experimento puede
confirmar y localizar el candidato, o matarlo — cualquiera limpia el terreno para R7.
