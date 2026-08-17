# ANIMA-4 · Experimento social de TOPOLOGÍA — informe de implementación (método)
**Cosmolab / VSTCosmo · para el equipo · 2026-06-29**

> **Advertencia epistémica:** los rótulos de audio ("voz", "música", "ruido") y de voz son **etiquetas
> humanas** de archivo/banco. NO sabemos qué significan para los organismos. Se mide respuesta a
> configuraciones acústicas y sociales, **NO** significados humanos.

Este documento describe **cómo está implementado** el experimento y **cómo leer/segmentar los datos**.
El informe de **resultados** lo escribe la propia batería al cerrar (`informe_social.md` + `resumen_social.md`
dentro del paquete de salida).

---

## 1. Pregunta y diseño

**Pregunta:** ¿la **topología** (quién oye/imita a quién) determina la estructura de convergencia entre los
4 organismos? Cada topología predice una matriz de similitud 4×4 distinta; se comparan contra lo observado.

**Variable manipulada:** la topología, configurada por la **fuente de relación** de cada organismo vía `/start`.

| Topología | Quién oye a quién | Predicción de la matriz |
|---|---|---|
| **PLENA** | todos = "otros organismos" (mezcla de los demás) | todos↔todos similares, sin bloques |
| **CADENA** (abierta D→C→B→A) | A←B, B←C, C←D, D←nadie | vecinos adyacentes similares, decrece con la distancia; A–D mínima |
| **ESTRELLA** (líder D) | A,B,C←D, D←nadie | A,B,C convergen hacia D; bajo entre A/B/C |
| **PAREJAS** | A↔B, C↔D | dos bloques A-B y C-D; cruces bajos |

**El mundo como fuerza divergente:** cada organismo recibe por su **oído de mundo** un audio fijo y distinto
de máximo contraste (A=voz, B=Brandemburgo, C=ruido, D=viento); el **oído de relación** lleva la topología.
Así la topología es la única fuerza que puede acercarlos.

**Desconfundir rol vs. audio:** 2 ciclos de las 4 topologías; en el **ciclo 2** se **rota** la asignación
audio→organismo un lugar. Si el patrón de convergencia sigue a la **posición** en la topología (no al audio
ni al organismo), gobierna la topología.

**Control de falsación:** la CADENA repetida con `ANIMA_CONTROL=shuffled` (recrea contenedores).

---

## 2. Trazabilidad (lo más importante para el análisis)

**Requisito:** cada fila de fisiología debe poder atribuirse sin ambigüedad a su condición, por `ts_real`.

**Vía implementada: OPCIÓN 1 — columnas `exp_*` aditivas por fila** (la preferida; verificado que no rompe):
- Antes de cada bloque, el script hace `POST /exp_tag` a cada organismo con su condición.
- El organismo escribe en **CADA fila** de Docker_Historia 5 columnas:
  `exp_topologia` · `exp_ciclo` · `exp_mundo_audio` · `exp_control` · `exp_fuente_relacion`.
- **No rompe nada:** los CSV antiguos (sin esas columnas) se leen con `union_by_name` (quedan NULL).

**Verificación hecha ANTES de lanzar:**
- `POST /exp_tag` → la fila viva trae las `exp_*` ✅
- Aparecen en el CSV de Docker_Historia (columnas 259–263) ✅
- `union_by_name` lee viejo (NULL) + nuevo (con tag) juntos ✅

**Segundo camino (redundante):** manifiesto `condiciones_<ts>.csv` con una fila por bloque×organismo:
`ts_real_ini, ts_real_fin, organismo, topologia, ciclo, mundo_audio, control, fuente_relacion`.

### Cómo segmentar el dato primario (ejemplo)
```sql
-- DuckDB: imitación media por topología × ciclo, directo desde las columnas exp_*
SELECT exp_topologia, exp_ciclo, organismo_id,
       avg(oao_imitacion_mag) imit, count(*) n
FROM read_csv('Docker_Historia/organismo_ANIMA_*/fisiologia/*.csv',
              union_by_name=true, null_padding=true, ignore_errors=true)
WHERE exp_topologia IS NOT NULL          -- solo las filas de ESTE experimento
GROUP BY 1,2,3 ORDER BY 1,2,3;
```
*(O hacer JOIN por `ts_real` contra el manifiesto si se prefiere esa vía.)*

---

## 3. Arquitectura del experimento (cómo corre)

- **Script único:** `Célula_Madre/experimentos/experimento_anima4_social.py` (código completo, no parches).
- Corre solo, de noche, envuelto en `caffeinate`. Por bloque: etiqueta (`/exp_tag`) → `/start` a los 4 con su
  config → espera ~10 min muestreando → `/control stop` → siguiente.
- **Orden:** 4 topologías ciclo 1 → 4 topologías ciclo 2 (audios rotados) → CADENA-shuffled. ~90 min.
- **No modifica los órganos** salvo el logging aditivo `exp_*` (verificado no-rompe). Observa la arquitectura tal cual.
- **Marcadores de cumplimiento** en el log: ✅ ok · ❌ error · ⊘ omitido · · informativo.

---

## 4. Métricas y salidas

Por organismo × condición se registran (medias/máx desde el muestreo y, sobre todo, en el dato primario
segmentable por `exp_*`): `oao_imitacion_mag`, `oao_echoica_n`, `voz_propias/aprendidas/estables/creadas`,
`alt_intencion_comunicativa`, `alt_agencia_otro`, `voz_otro_valor_ecologico`, `expectativa`, `OI`, `Omega`,
`LF_op`, `ove_experiencias`, `cara_valoracion`, orientación, energía L/R.

**Matrices 4×4 por condición:**
- **Imitación (magnitud):** correlación temporal de `oao_imitacion_mag` entre organismos.
- **Gestos (mejor lag):** similitud de los `g_*` (freq/intensidad/pausa/repetición) probando lags −3..+3
  (imitar deja firma de retardo). Se compara **predicho vs. observado** por topología.

**Léxico:** conteo de propias/aprendidas y solapamiento de repertorios.
**Salvedad obligatoria:** cada organismo numera "palabra propia N" **desde 1**, así que los nombres
**colisionan** entre organismos → la propagación **nominal** ("la palabra de X llegó a Y") **no es
atribuible** en este diseño. Se reporta **solapamiento/conteo agregado**, no rutas nominales.

**Paquete de salida** → `~/Downloads/ANIMA4_TOPO_<ts>/`:
`resultados_sociales.csv` · `matriz_imitacion_por_condicion.csv` · `matriz_gestos_por_condicion.csv` ·
`condiciones_<ts>.csv` (manifiesto) · `difusion_lexica.csv` · `resumen_social.md` · `informe_social.md` ·
`primarios_fisiologia.tar.gz` (segmentables por `exp_*`) · `bitacoras.tar.gz`.

---

## 5. Limitaciones honestas (deben leerse junto con los resultados)

1. **Nodo-fuente no aislado a nivel de gestos.** "D oye a nadie" es limpio por **audio** (silencio), pero su
   imitación de **gestos** cae al roster por entorno (fallback de la arquitectura). Los nodos NO-fuente sí
   siguen la topología del cfg. → en CADENA/ESTRELLA, el nodo D no está 100% aislado para gestos.
2. **`shuffled` desordena el AUDIO, no los gestos.** El canal de imitación son los gestos por HTTP, que bajo
   `shuffled` siguen siendo reales; `shuffled` prueba la contingencia **acústica**. El control que corta los
   **gestos** es `ANIMA_CONTROL=null` (corrida previa: la imitación colapsó). Está documentado en el informe.
3. **El mundo divergente puede aplanar la convergencia.** En la corrida previa (todos↔todos con mundo
   divergente), las correlaciones cayeron a ~0.05 (vs 0.43 con mundo en silencio). Si las matrices salen
   planas, la lectura es: el estímulo divergente fuerte **domina** sobre el acople topológico. Recomendación:
   repetir con mundo **compartido/silencio** para detectar el efecto relacional.
4. **Correlación ≠ causalidad.** Por eso el control y la rotación de audios.

**Un resultado negativo es válido** si la infraestructura funcionó y la topología fue verificable.

---

## 6. Estado y archivos

- **Script:** `Célula_Madre/experimentos/experimento_anima4_social.py`
- **Cambio de organismo (aditivo, verificado):** columnas `exp_*` en la fila + endpoint `POST /exp_tag`
  (en `web/VST_CelulaMadre_WebLive_A/B/C/D.py`).
- **Datos primarios:** `Docker_Historia/organismo_ANIMA_{A,B,C,D}/fisiologia/*.csv` (con `exp_*`).
- **Resultados:** se escriben al cierre de la batería en `~/Downloads/ANIMA4_TOPO_<ts>/`.
