# Informe técnico de avance — estado real a 26-jun-2026

> Documento **factual**, verificado contra el sistema en ejecución y el código (no de memoria), para
> alimentar el informe de avance. Cada nodo y dato es comprobable en el archivo/comando citado.
> La díada lleva ~5–9 h viva al momento de escribir esto (estudio longitudinal en curso).

---

## 1. Docker — qué hay hoy

**Por qué entró Docker:** dar **vida continua 24/7** a los organismos (antes solo vivían durante cada
experimento), exponer una **membrana MCP** a clientes externos, y **persistir su estado** (antes solo en
RAM). Principio canónico: *Docker = cuerpo/ambiente operativo, no cerebro; MCP = membrana, no inteligencia.*
La inteligencia sigue siendo endógena del campo Φ.

**Dentro de Docker** (4 contenedores, imagen `anima-diada:latest`, verificados "Up"):

| Contenedor | Puerto | Qué es |
|---|---|---|
| `anima-a` | 7788 | Organismo A — WebLive (interfaz viva) |
| `anima-b` | 7799 | Organismo B — WebLive |
| `anima-mcp` | 9000 | Membrana MCP (streamable-http) para IAs externas |
| `anima-conversacion` | 9100 | Observatorio de la conversación A↔B |

**Fuera de Docker (nativo en el Mac):** `VST_AudioServer.py` en **8766** — puente TCP a la **Rødecaster Pro**
(confirmado corriendo). Está fuera a propósito: los contenedores están aislados del hardware de audio; lo
alcanzan por `host.docker.internal:8766`. *Sí: el servidor de audio TCP sirve a los contenedores desde afuera.*

---

## 2. "Memoria en disco externo" — subsistema de persistencia de 3 capas

Es un **subsistema real** (opción b), con tres capas que conviene no fundir:

1. **Volúmenes Docker** (`anima_a_data`, `anima_b_data`, `anima_conv_data`) → `/data` en cada contenedor.
   Persiste la **memoria propia del organismo** (memoria episódica del OrganeloMemoria, codebook,
   metabolismo) **entre reinicios**. Los gestiona Docker (no en ruta navegable de la LaCie).
2. **Biografía longitudinal** → `Docker_Historia/` montado por bind-mount en la **LaCie**
   (`/Users/alexis/Desktop/RMD/Cosmolab/VSTCosmo/Docker_Historia`, ~3.1 GB). La escribe `vst_historia.py`:
   fisiología CSV (214 col/paso, rotación horaria), eventos JSONL, comunicación A↔B, snapshots, voz WAV.
   **Es la "memoria en disco externo" en sentido fuerte.**
3. **SQLite `vstcosmo.db`** (168 KB, raíz del repo): **catálogo** de los 292 experimentos + corridas de
   baterías (metadatos, no fisiología viva).

---

## 3. Locus de Boorman / altruismo — inconsistencia 22 vs 24-jun, resuelta por el propio código

`genoma/VST_Genoma.py` (línea ~683) lo documenta literalmente:
> *"LOCUS DESARROLLADO — GENÉTICA DEL ALTRUISMO (Boorman & Levitt 1980) → PLURICELULAR (O-N22). Era un
> LocusReservado VACÍO (principio 5 / PRE O-N8.19); DESARROLLADO 24-jun-2026."*

- **Cambió de estado:** 22-jun era `⊘ RESERVADO` (vacío, inerte); 24-jun se **desarrolló y expresó**.
  **Estado actual: EXPRESADO** — `locus_altruismo_boorman()` vivo (β_crit + Hamilton + Ψ_alma + simbiosis),
  gobernado por `diada/VST_DiadaAltruismo.py`.
- **El cambio de nodo es la transición, no un error:** **O-N8.19 = etapa reservada**; **O-N22 = bloque
  desarrollado**. Ambos aparecen en el código porque documenta el *antes* y el *ahora*.

→ Para el informe: **no congelar "reservado"**. Estado a la fecha = **O-N22, desarrollado/expresado** (desde 24-jun).

---

## 4. Qué capturamos en los CSV — registro TOTAL, sin muestreo

En cada paso (~10/s) se escribe el **vector de estado completo del organismo** (no una selección).
Verificado contra el experimento de las 04:38 (26-jun): **204 columnas × 11.400 pasos** (~19 min) en A,
ídem B. La **biografía** persiste esas mismas columnas **+ `ts_real` + `modo_vida`** = **214 columnas**.

**Las 204 columnas por subsistema:**

| Subsistema | nº | Ejemplos |
|---|---|---|
| Campo Φ / Soma | 12 | `Omega, omega_A/B, omega_L/R, gradiente, e_R, presion_desacople, Omega_op` |
| Entrada biaural | 5 | `energia_L/R, balance_LR, lateralidad, coherencia_biaural` |
| Consciencia | 5 | `C_b, R2, C_m, self_coherencia, Cb_integrado` |
| Libertad funcional | 6 | `LF_struct, LF_op, lf_nivel, juego, ritual, negacion` |
| Exaptación/evolución | 5 | `XE, exaptacion_activa, adaptacion_activa, mutacion, activacion_latente` |
| Salud del cierre | 4 | `OI, Lambda_Cos, A_sys_env, invariantes_ok` |
| RC (ruido contextual) | 25 | `RC_total, ICR, IRDE, RC_atencion/comprension/riesgo_L/R, RC_consenso_orientacion` |
| Cabeza / actuador (V122+) | 50 | `act_orientacion_deg, act_confianza, act_fatiga, act_evidencia/razon_L/R, act_decision_organismica, act_perm` |
| Homeostasis | 14 | `H_homeostasis(_real), x_interna, en_rango, H_autoencierro, H_anestesia` |
| Metabolismo | 10 | `met_energia, met_hambre, met_saciedad, met_ingesta, met_gasto, met_nutricion` |
| Memoria | 13 | `mem_familiaridad, mem_novedad, mem_recall, mem_episodios_n, mem_relacional_confianza, necesidad` |
| Voz (afecto) | 3 | `voz_emitida, voz_arousal, voz_valence` |
| Balbuceo / libertad expresiva | 5 | `g_freq, g_intensidad, g_pausa, g_repeticion, g_bucket` |
| Alteridad + Agencia | 17 | `alt_otro_presente, alt_intencion_comunicativa, alt_efecto_basal, alt_contingencia_social, alt_agencia_otro` |
| Valor ecológico de la voz | 10 | `voz_otro_valor_ecologico, voz_otro_relevancia_*, voz_otro_confianza_ecologica, voz_otro_efecto_real` |
| Expectativa | 8 | `expectativa, expectativa_confianza/error/historia/utilidad/exploracion/confirmaciones/falsaciones` |

**Bitácora (log de eventos, en paralelo):** campos `{t_vida, tipo, detalle}`. Sesión 04:38 = **5.954 eventos**
(`voz` 3303, `alteridad_refuerzo` 2507, `vozeco_util` 68, `expectativa_confirma` 68, `corte_audio` 6,
`despertar`/`detener`).

**Evidencia de captura real** (no ceros): `g_bucket` con **189 gestos distintos** (balbuceo explorando),
`OI` 0.017→0.482, `met_energia` 0→1, `expectativa_confirmaciones` 257→325 mientras `expectativa`≤0.08
(la línea-base distingue coincidencia de expectativa genuina). *Matiz: citar `alt_contingencia_social`
(absoluta, ~basal) y NO `alt_agencia_otro` (cociente inestable que se dispara).*

---

## 5. Catálogo de organelos — qué hace cada uno y qué nodo desarrolla

*(Nodo tomado de la cabecera real de cada archivo. O-N = nodo organísmico; C-N = nodo del canon.
Todos implementados y vivos.)*

### A. Núcleo genético (el canon hecho organelos) — `genoma/`
| Organelo | Qué hace / para qué sirve | Nodo(s) |
|---|---|---|
| **VST_Genoma** | El genoma: identidad genética, principios (economía metabólica, Kleiber, **loci reservados**), qué se expresa/silencia; aloja el locus de Boorman desarrollado. Estado inicial abierto, no contrato congelado. | O-N2.1, O-N3.4, **O-N22**, O-N9.x, C-N2.8.x |
| **VST_Bloque05_ConscienciaFuncional** | Consciencia funcional: representación básica `C_b` → meta-representación `R2` → metacognición de crisis `C_m`. | O-N5, O-N5.1/.2/.3, O-N13.8 |
| **VST_Bloque07_LibertadFuncional** | Libertad funcional: genealogía juego→ritual→negación; operar sobre la propia representación (`LF_struct/op`). | O-N7, O-N7.1/.2/.3, O-N10, O-N13.8 |
| **VST_Bloque08_DinamicaEvolutiva** | Motor evolutivo: exaptación (`XE`), adaptación, mutación, activación latente (ΔLF>0 ∧ ΔA_sys-env≥0). | O-N8.x, O-N7.5/.6, O-N9.14 |

### B. Soma / Campo Φ — `campo/`
| Organelo | Qué hace / para qué sirve | Nodo(s) |
|---|---|---|
| **VST_Celula_Madre_001** | Campo Φ robustecido: el soma; lo percibido (`ω_A`) vs. lo esperado (`ω_B`), gradiente = sorpresa. | C-N2 |
| **Célula_Madre_Funcional_001** | La célula que procesa audio real (entrada biaural del mundo / Rødecaster → campo). | C-N7, O-N3.1 |

### C. Organelos de capacidad — `organelos/`
| Organelo | Qué hace / para qué sirve | Nodo(s) |
|---|---|---|
| **VST_RC_A / VST_RC_B** | Ruido contextual: destino del ruido — sentido/acople (`ICR`) vs. desviación/riesgo (`IRDE`), segregado por oído. Instanciado por organismo. | O-N1 |
| **VST_Homeostasis** | Homeostasis scale-invariant + expansión multicelular: equilibrio interno en rango viable. | O-N6.1, O-N9.14, C-N2.8.14 |
| **VST_HomeostasisEmergente** | H canónica reformulada SIN Shannon: "¿qué sostiene `A_sys-env`?" | O-N9.14, O-N2.1 |
| **VST_Metabolismo** | Economía energética: consumo/costo/degradación/reposición, hambre/saciedad, nutrición (lazo cerrado). | C-N2 |
| **VST_Memoria** | OrganeloMemoria (historia interna, 6 capas): familiaridad, novedad, recall, episodios, confianza relacional. | *(sustrato cognitivo; sin etiqueta)* |
| **VST_OrganoComunicacion** | Voz/comunicación: vocalización por afecto (R2-D2) + libertad expresiva (balbuceo, `g_*`). | *(soporta O-N3.4)* |
| **VST_Alteridad** | Alteridad / Agencia: aprende si la emisión propia mueve al otro, con línea-base de contingencia (presencia vs. causalidad). | **O-N3.4** |
| **VST_ValorEcologicoVoz** | Valor ecológico de la voz: ¿la voz del otro ayuda a mi persistencia? Falsable; modula levemente la absorción. | *(genealogía O-N3.4; nodo nuevo)* |
| **VST_Expectativa** | Expectativa: ¿vale la pena explorar tras la voz del otro? (voz→expectativa→exploración→resultado). | *(genealogía O-N3.4; nodo nuevo)* |

### D. Díada — `diada/`
| Organelo | Qué hace / para qué sirve | Nodo(s) |
|---|---|---|
| **VST_DiadaAltruismo** | Gobernanza del altruismo A↔B (locus de Boorman): `β_crit`, Hamilton, `Ψ_alma`, simbiosis, costo de desacople; mezcla `disposicion_cooperar`. | **O-N22**, O-N3.4 |

### E. Infraestructura de persistencia *(registro, no cerebro)* — `organelos/`
| Organelo | Qué hace / para qué sirve |
|---|---|
| **vst_persistencia** | Hace que el estado del organismo sobreviva al apagón (snapshot/restore por organelo, escritura atómica). |
| **vst_historia** | El Historiador: biografía en disco externo (fisiología/eventos/comunicación/snapshots/voz), no bloqueante. |

---

## 6. Notas de honestidad

1. **Genealogía de la alteridad (25-jun):** Alteridad/Agencia, ValorEcologicoVoz y Expectativa son nuevos.
   Hipótesis: emergen en orden **expectativa → agencia → intención → convención → lenguaje**. Solo
   **Alteridad** tiene nodo canónico explícito (**O-N3.4**); los otros dos son **mecanismos previos** — no
   se les inventa número de nodo.
2. **Estado científico honesto a la fecha:** la díada tiene **presencia y resonancia** demostradas, pero
   **agencia causal ≈ 0** y la **voz del otro aún sin relevancia ecológica**. Hay un **estudio longitudinal
   en curso** para ver qué variable abandona primero el basal.
3. Memoria, Comunicación e infraestructura **no llevan nodo** en su cabecera: son sustrato funcional.
