# ANIMA-4 · Correcciones y protocolo v2 (ecología experimental)
**Cosmolab / VSTCosmo · para el equipo · 2026-06-29**

> **Advertencia epistémica:** los rótulos de audio y de voz son etiquetas humanas de archivo/banco.
> NO sabemos qué significan para los organismos. Se mide respuesta a configuraciones acústicas y
> sociales, NO significados humanos.

Tras la revisión de GPT (rigurosa y acertada) corregimos tres cosas y reformulamos el protocolo. Este
documento explica **qué cambió y por qué**, para que la próxima corrida sea interpretable.

---

## 1. Lo que la revisión dejó claro (y aceptamos)

1. **Los "roles" de invención eran INDIVIDUALES, no producidos por la sociedad.** `voz_creadas`/
   `voz_aprendidas` son **contadores monótonos que PERSISTEN entre vidas** (al nacer, `_cargar_creadas()`
   recupera el vocabulario consolidado del disco). Por eso "A inventa 2" reflejaba **patrimonio acumulado**,
   no algo que el experimento produjera: el valor era idéntico en todas las topologías porque casi no se
   movió durante la corrida (A: 2 en las 9 condiciones; D: una sola palabra nueva). La emulación sí ocurría
   en vivo (B 1→3), pero pequeña y con un contador frágil (se resetea al renacer si no consolida).
2. **La trazabilidad de PALABRAS (rutas) no estaba.** Se había implementado trazabilidad de **filas**
   (`exp_*`), no de palabras. La pregunta "¿la palabra de B aparece luego en D?" no tenía vía limpia.
3. **El mundo divergente aplanó la convergencia.** Con cada organismo en un audio fuerte y distinto, el
   canal relacional quedó ahogado (correlaciones ~0.05 vs 0.43 con mundo en silencio). El propio diseño lo
   anticipaba; ahora lo confirmamos.

---

## 2. Correcciones aplicadas (código)

### 2.1 IDs globales de palabra + RUTA léxica
- El `voz_emitida` ya llevaba la letra del organismo en el label (`palabra_A001`, `apr_B002`) — el ID global
  **ya era distinguible**; el informe previo colisionaba porque usaba `voz_titulo` ("palabra propia N"),
  que sí choca entre organismos.
- **Nuevas columnas por fila** (aditivas, en `VST_OrganoComunicacion.voz_actual` + `COLS_VOZ`):
  - `voz_id` — ID global de la palabra emitida (p.ej. `palabra_A001`).
  - `voz_emulada_de` — si la voz es una emulación, **el ID global de la palabra que se copió** (de quién).
- En `quizas_emular` se registra `emulada_de = peer.voz_emitida`. → ahora se puede **reconstruir la ruta**:
  organismo X emite `palabra_A001`; más tarde organismo Y emite una voz con `voz_emulada_de = palabra_A001`
  → ruta **A→Y** para esa palabra. *Verificado:* las columnas existen en la fila y en el esquema CSV.

### 2.2 Reset a cero
- Se **borraron los volúmenes de estado** de los 4 organismos (`anima_*_data`) → **vocabulario, memoria y
  metabolismo en CERO**. Cada palabra que aparezca ahora es **nueva y atribuible a este experimento**.
  (Docker_Historia y el observatorio NO se tocaron; los resultados previos se conservan.)

---

## 3. Protocolo v2 (lo que se corre ahora)

La lógica de GPT: si los roles vienen del individuo, la prueba de que el **acople** hace algo no es "aparecen
roles" sino **"los roles fijos se MODIFICAN bajo acople"**. Eso exige un baseline y un canal relacional no
ahogado. Por eso:

| Fase | Config | Qué mide |
|---|---|---|
| **0 · BASELINE aislado** | `ANIMA_CONTROL=null` (sin percepción del par) · mundo=silencio | tasa **intrínseca** de invención de cada organismo, SIN acople (contra qué comparar) |
| **PLENA** | todos↔todos · mundo=silencio | convergencia general cuando el mundo no compite |
| **CADENA** | A←B←C←D · mundo=silencio | ¿gradiente por distancia en la cadena? |
| **ESTRELLA** | A,B,C←D · mundo=silencio | ¿convergen hacia el líder D? (señal que ya asomó) |
| **PAREJAS** | A↔B, C↔D · mundo=silencio | ¿dos bloques? |
| **Control** | CADENA · `ANIMA_CONTROL=shuffled` | ¿la estructura depende de contingencia? |

- **Mundo COMPARTIDO/silencio** en todas (no divergente) → el canal relacional es la única señal.
- **Trazabilidad completa:** columnas `exp_*` por fila + `voz_id`/`voz_emulada_de` (rutas) + manifiesto.
- ~10 min/condición · ~65 min · `caffeinate`.

**Preguntas que ahora SÍ se pueden contestar:**
1. ¿El acople **mueve** la tasa de invención respecto al baseline aislado?
2. ¿La palabra de X **llega** a Y? (rutas vía `voz_emulada_de`).
3. ¿La topología deja huella cuando el mundo no la tapa?

---

## 4. Notas honestas que persisten

- **`shuffled` desordena el AUDIO, no los gestos** (el canal de imitación es HTTP, sigue real bajo shuffle).
  El control que corta los gestos es `null` (= la fase BASELINE).
- **Nodo-fuente:** "oye a nadie" es limpio por audio, pero su imitación de gestos cae al roster por entorno
  (fallback) — salvo bajo `null`, donde sí queda aislado. Por eso el baseline usa `null`.
- **Medir DELTAS, no absolutos:** como los contadores acumulan, el efecto del experimento es el **crecimiento**
  dentro del bloque y respecto al baseline, no el valor absoluto.

---

## 5. Archivos
- Corrección: `organelos/VST_OrganoComunicacion.py` (voz_id/voz_emulada_de) + `web/VST_CelulaMadre_WebLive_*.py` (`COLS_VOZ`).
- Protocolo: `experimentos/experimento_anima4_social.py` (mundo silencio + fase BASELINE + control).
- Datos: `Docker_Historia/.../fisiologia/*.csv` con `exp_*`, `voz_id`, `voz_emulada_de` (segmentables).
- Salidas de la corrida: `~/Downloads/ANIMA4_TOPO_<ts>/`.
