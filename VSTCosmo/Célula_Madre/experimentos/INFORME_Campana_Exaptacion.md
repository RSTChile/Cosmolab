# Informe consolidado — Campaña de Exaptación de la Célula Madre

**Fuente de datos:** `experimentos/resultados/resumen.csv` (120 experimentos, todos `status=ok`, 0 descartados).
**Generación de estímulos:** 117 archivos del proyecto *binaural-real* + 3 demos sintéticos.
**Estilo:** anti-Shannon — se reportan magnitudes, no se sobre-afirma, y se marca explícitamente lo que queda abierto.

---

## 1. Metodología y alcance

- **Diseño:** cada experimento es una simulación de la Célula Madre alimentada por una señal de audio (WAV o demo). Sin hardware en vivo: no hay micrófono, sensores ni actuadores físicos; toda la dinámica es endógena del campo Φ y sus organelos.
- **Duración:** simulación con tope de 20 s por corrida (200 pasos en runs largos, 0,6 s / 6 pasos en runs cortos). Semillas fijas → resultados reproducibles.
- **Métricas de exaptación** (dos ejes que NO miden lo mismo, r(XE_media, exapt_pct) = −0,585):
  - **XE_media / XE_max** = intensidad de la exaptación.
  - **exapt_n / exapt_pct** = cantidad / frecuencia de eventos por paso.
  - **Omega_op_max (Ω)** = intensidad operativa; cumple Ω = 1 + XE_max en régimen no saturado.
- **Controles registrados:** homeostasis (H_ini/H_fin), organización (OI_ini/OI_fin), ajuste (R2_media), complejidad (C_m_max), canal ritual (ritual_pct), invariantes (inv_ok_min).
- **Estructura del dataset:** 2 clases (archivo=117, demo=3); 90 runs largos (200 pasos, 20 s), 24 runs cortos (6 pasos, 0,6 s) que actúan como cuasi-controles de exposición mínima.

### Limitaciones (lo abierto)
- **No hay condición de estímulo-nulo/silencio.** El control más cercano son los 24 runs cortos de 6 pasos. Es la mayor laguna del diseño.
- **Tope de 20 s:** se desconoce la dinámica de largo plazo (¿la homeostasis se estabiliza o sigue cayendo?).
- **Sin hardware vivo:** los resultados son de simulación; no se ha probado transferencia a sensores/actuadores reales.
- **Confound de energía no neutralizado en el diseño** (ver §3): energía y duración covarían entre estímulos, lo que sesga el "ranking por material".

---

## 2. RESPUESTA: ¿qué despierta la exaptación?

La respuesta **depende de qué métrica de exaptación se use**, y ambas cuentan historias distintas:

### Por INTENSIDAD (XE_media) — señal periódica simple y sostenida
Ranking por material (XE_media):

| Material | XE_media |
|---|---|
| pulso (logarítmico) | **0,965** |
| tono (freq_Hz 400–480) | 0,953 |
| tono (puro) | 0,929 |
| mixto (ondas) | 0,835 |
| ruido (blanco) | 0,800 |
| clicks/ritmo | 0,537 |
| nota (musical do/re/mi…) | 0,430 |
| binaural/LR | 0,370 |
| voz+viento (mixto) | 0,353 |
| música compleja (Brandemburgo, Blue Monday) | 0,256 |
| voz/habla | 0,250 |
| BigBang | 0,151 |
| viento | 0,142 |

Máximo individual: `wav__Tono_puro` = 0,974. Mínimo: `demo__clicks` = 0,003.

### Por FRECUENCIA (exapt_pct) — notas musicales sueltas
Otro orden: las **notas musicales aisladas lideran con 70,0%** pese a XE_media bajo (0,43), porque disparan muchos eventos de baja magnitud. Las notas largas/altas llegan a **100% exapt_pct con XE casi nula (~0,06)** — es ruido de eventos, no intensidad. Le siguen mixto(ondas)=47,8%, tono puro=40,5%, ruido=39,2%.

### Síntesis honesta
- **Estímulo rico y enérgico → exaptación intensa.** Pulso, tono y ruido blanco saturan; voz, música compleja, viento y BigBang la deprimen.
- **Notas musicales sueltas → exaptación frecuente pero débil** (techo del 100% en exapt_pct con intensidad mínima).
- **Espacialización (±60°, binaural; n=96):** intensifica (XE_media 0,643 vs 0,347 del mono) pero reduce la frecuencia (exapt_pct 37,4% vs 54,7%). Concentra la exaptación, no la multiplica.

---

## 3. Correlatos: lateralidad vs. riqueza temporal vs. energía (con veredictos adversariales)

Tres hipótesis se sometieron a re-derivación independiente desde el CSV. Veredictos:

### Veredicto A — "La lateralidad NO es el predictor principal" → **MATIZA**
Depende de la métrica:
- **Para intensidad (XE_media):** el predictor dominante es la **ENERGÍA MEDIA (L+R), Pearson +0,945** (Spearman +0,836), por encima de lateralidad (+0,664) y gradiente_std (+0,643). Aquí la afirmación se sostiene: la lateralidad es secundaria.
- **Para cantidad/frecuencia (exapt_n, exapt_pct):** la **lateralidad SÍ es el predictor principal** — exapt_n r=+0,867 (energía solo +0,549); exapt_pct r=−0,947 (mayor magnitud absoluta, signo negativo).
- El driver físico de la lateralidad es la **asimetría espacial del campo** (gradiente_std, r=+0,952 con lateralidad). El **|balance| energético es inútil** (r≈0,0–0,3). `coherencia` es su inverso (campos coherentes exaptan más a menudo pero más débil).

### Veredicto B — "La riqueza temporal predice mejor que energía/lateralidad" → **REFUTA**
- `energia_std` vs XE_media = **−0,518** (¡signo negativo! más variabilidad de volumen → MENOS exaptación-intensidad).
- `gradiente_std` vs XE_media = +0,643 en Pearson pero **se desploma a +0,412 en Spearman** → relación no monótona / artefacto de escala.
- En las **cuatro** métricas, la riqueza temporal NUNCA es el mejor predictor: siempre la supera la energía media (intensidad) o la lateralidad (cantidad).

### Veredicto C — "El ranking por material es un efecto real, no artefacto" → **REFUTA**
- La **energía media explica R²=0,892** de XE_media por sí sola; energía+duración → R²=0,966 (solo 3,4% de varianza residual).
- **Prueba de energía emparejada (smoking gun):** a E≈0,144 fija, 16 notas musicales distintas dan XE entre 0,059 y 0,076 (rango 0,017) — la identidad del material casi no mueve XE. La misma nota "Do" salta de XE=0,10 (6 pasos) a XE=0,96 (200 pasos) solo cambiando energía/duración.
- Los "ganadores" tienen **9× más energía** que los "perdedores" (E media 0,572 vs 0,064) con duración casi igual.
- **Efecto de material puro ≈ 2,6%** de la varianza total tras descontar energía+duración. Existe, pero es marginal.

**Conclusión de §3:** el correlato dominante de la INTENSIDAD es la energía inyectada al campo; el de la CANTIDAD/FRECUENCIA es la lateralidad (asimetría espacial). La riqueza temporal queda descartada como mejor predictor, y buena parte del "ranking por material" es un reflejo de cuánta energía mete cada estímulo, no una propiedad intrínseca del material.

---

## 4. Timing y magnitud

- **CUÁNDO — temprana y casi universal.** Las 120/120 exaptan. t_primera_exapt: mediana **0,10 s**, media 0,54 s. 94/120 disparan en ≤0,2 s; 96/120 en ≤0,5 s; solo 7 son tardíos (≥2,0 s). El suelo temporal de la campaña es 0,10 s (94 experimentos lo comparten).
- **Los más tardíos son señales naturales/musicales de baja excitación** y además exaptan débil: Brandemburgo (t1=7,50 s, XE_max=0,066), Viento (5,60 s), Blue Monday y binaurales (1,90 s).
- **CUÁNTO.** XE_max satura en 1,0 en 62/120. Ω satura en 3,0 solo en 3 casos (Pulso_logarítmico y variantes ±60°). Verificado: **Ω = 1 + XE_max exacto** en los 58 no saturados; cuando XE topa, Ω sigue subiendo (hasta 3,0) captando la intensidad excedente.
- **SOSTENIDA, no en ráfaga (hallazgo central).** **Cero** experimentos de ráfaga (XE_max − XE_fin > 0,3: lista vacía). Los 62 que saturan XE_max=1,0 terminan los 62 con XE_fin=1,0. Patrón típico: **rampa-y-mantén** — arranca temprano, sube y se queda arriba.
- **Top Ω:** Pulso_logarítmico 3,0000 / Tono_puro 2,9345 / Ondas_mixtas 2,8259 / Ruido_blanco 2,5959.

---

## 5. Controles (anti-artefacto)

- **Homeostasis: NO se sostiene.** H baja en 120/120 (ninguna sube ni queda plana). H_ini muy alto y estrecho (0,9895–0,9993) → H_fin 0,2974–0,9955; caída media dH=−0,352. **Escala con la duración** (runs cortos H_fin≈0,9955; runs largos H_fin media 0,5344) → proceso real con dosis, no sesgo fijo.
- **Organización (OI): sube en 120/120.** dOI media +0,120 (max +0,349). Mayores ganancias: Ruido_blanco_pos60deg (0,376→0,726), freq_400_neg60deg_largo (0,410→0,748), Pulso_logaritmico (0,425→0,733).
- **DISOCIACIÓN CLAVE.** H y OI se mueven en sentidos **opuestos en el 100% de los casos** (H baja, OI sube). Si la exaptación fuera artefacto de saturación global, covariarían; la anticorrelación sistemática indica organización real independiente del colapso homeostático.
- **Sin techo de saturación:** R2_media máx 0,703 (media 0,552, 0/120 ≥0,989); C_m_max máx 0,361 (media 0,120, 0/120 cerca de 1,0). Ningún confound de techo.
- **ritual_pct = 0,0 en 120/120** (canal de control silente, como se esperaba).
- **inv_ok_min = 5,0 exacto y constante en 120/120** (integridad estructural intacta; nota: es 5,0 exacto, no el rango 5–6 esperado).
- **Anomalía única a vigilar:** `freq_400_pos60deg_largo` con juego_pct=17,0 mientras el resto está en 0. negacion_pct ≠ 0 en 96/120 (las 24 con negacion=0 son los runs cortos de 6 pasos).

---

## 6. Conclusiones y próximos experimentos

### Conclusiones
1. **La exaptación es real** y no un artefacto obvio: OI sube en 100% de runs, disociado del colapso de homeostasis, sin saturación de R2/C_m, con ritual=0 e invariantes intactas.
2. **Es temprana, intensa y sostenida** (rampa-y-mantén; cero ráfagas; mediana de disparo 0,10 s).
3. **El driver de la INTENSIDAD es la energía inyectada al campo** (R²=0,892); el de la **FRECUENCIA es la lateralidad/asimetría espacial**. La riqueza temporal NO predice mejor.
4. **El "ranking por material" es en su mayoría un confound energético** (~9× más energía en los ganadores); el efecto de material puro es ~2,6% — marginal pero no nulo (señales sintéticas de banda ancha rinden algo por encima de su predicción energética).
5. **El |balance| energético crudo es inútil** como predictor; la lateralidad funcional vive en la asimetría espacial (gradiente_std), no en el desbalance global.

### Próximos experimentos sugeridos
1. **Control de estímulo-nulo / silencio** (la laguna #1): runs largos con entrada cero o ruido sub-umbral, para fijar la línea base de OI y XE sin estímulo.
2. **Diseño de energía emparejada por material:** normalizar todos los estímulos a la misma energía RMS y duración, para aislar el efecto de material puro (~2,6%) de forma limpia.
3. **Simulaciones largas (>20 s):** comprobar si la homeostasis se estabiliza o sigue cayendo monótonamente; caracterizar el destino de largo plazo.
4. **Barrido de lateralidad controlado:** variar gradiente_std a energía fija para confirmar causalmente que la asimetría espacial gobierna la frecuencia (exapt_n/exapt_pct).
5. **Investigar los dos outliers de control:** el juego_pct=17,0 de `freq_400_pos60deg_largo` y la activación de negación, para descartar inestabilidades.
6. **Usar XE_media/XE_max/exapt_n como métrica de éxito de exaptación significativa**, no exapt_pct (que premia ruido de eventos de baja magnitud).
