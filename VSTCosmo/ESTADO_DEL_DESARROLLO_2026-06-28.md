# ANIMA — Estado del desarrollo
**Díada de organismos computacionales A ↔ B · Cosmolab / VSTCosmo**
Fecha de corte: **2026-06-28**. Documento para poner al equipo al día.

> **Advertencia epistémica transversal (vale para TODO lo de abajo):** las etiquetas de los
> audios ("voz", "música", "ruido") y de las vocalizaciones (ternura, dolor…) son **solo etiquetas**.
> No sabemos qué significan esos sonidos para los organismos. Todo lo que medimos describe
> respuesta a configuraciones, no a "categorías de significado".

---

## 1. Resumen ejecutivo

ANIMA es una **díada** de dos organismos computacionales cosmosemióticos (**ANIMA_A** y **ANIMA_B**)
que viven en contenedores Docker, perciben sonido del entorno (audio guardado o el **RØDECaster Pro II**
en vivo), tienen un **cuerpo** (campo Φ + Soma) y un conjunto de **organelos** que les dan metabolismo,
memoria, homeostasis, expresión vocal, reconocimiento del otro (alteridad), aprendizaje por imitación,
y valoración experiencial. Cada uno **oye al otro** y puede **imitar** y **converger** hacia un léxico
compartido — esa es la tesis central del proyecto.

**Estado general:** la díada está **viva y funcional**. En esta sesión se reparó un bug crítico de
imitación, se rediseñó por completo la página de observación, y se construyó la infraestructura para
saber qué fuente de audio escucha cada organismo en cada instante.

---

## 2. Arquitectura del organismo (el "cuerpo" + los organelos)

Cada organismo es una **célula** con un núcleo dinámico (**Campo Φ**: Ω, gradiente) alimentado por el
**Soma** (la membrana que capta el sonido binaural L/R). Sobre ese cuerpo corren los organelos, en este
orden de flujo por paso de simulación (verificado en el código):

```
Mundo (audio) → Soma → Campo Φ (Ω, gradiente)
   → {Metabolismo · Homeostasis · Memoria}        (órganos internos: estado vital)
   → {Alteridad · Expectativa · Valor ecológico · Aprendizaje/OAO}   (órganos relacionales)
   → Expresión (gesto vocal) → Voz/Comunicación → altera el Soma del OTRO  → … (lazo cerrado)
   → OVE (valoración) → Cara/boca   (lectura afectiva, SIN influencia causal)
```

La **voz de cada organismo altera al otro por dos canales**: (1) los **gestos** vía HTTP
(`/comunicacion/estado`) — esto es lo que alimenta la **imitación**; y (2) el **sonido** de su voz por
audio — esto alimenta la "compuerta" de escucha y la orientación. (Ver §9, hallazgos.)

---

## 3. Los organelos

| Organelo (archivo) | Qué hace | Para qué sirve |
|---|---|---|
| **Soma / Campo Φ** (motor en `genoma`/`campo`) | Integra el sonido en el estado interno (Ω orden, gradiente, C_m, XE) | El cuerpo del organismo; todo depende de él |
| **Metabolismo** (`VST_Metabolismo`) | Come la experiencia (nutritiva/tóxica), paga el costo de vivir | Energía: define hambre/saciedad (`met_*`) |
| **Homeostasis** (`VST_Homeostasis` + `…Emergente`) | Vigila variables internas en rango vital | Autorregulación (`H_homeostasis`, `en_rango`, `x_interna`) |
| **Memoria** (`VST_Memoria`) | Guarda episodios; reconoce familiar vs nuevo | Continuidad (`mem_episodios_n`, `mem_familiaridad`, `mem_novedad`, `mem_recall`) |
| **Alteridad** (`VST_Alteridad`) | Detecta si el otro está presente y si su emisión lo mueve (agencia) | Reconocer al otro como sujeto (`alt_intencion_comunicativa`, `alt_agencia_otro`, `alt_contingencia_social`) |
| **Expectativa** (`VST_Expectativa`) | ¿Vale la pena seguir explorando tras la voz del otro? | 1er eslabón de la genealogía del sentido (`expectativa*`) |
| **Valor ecológico de la voz** (`VST_ValorEcologicoVoz`) | ¿La voz del otro ayuda a persistir? | Convierte señal social en valor (`voz_otro_valor_ecologico`) |
| **Aprendizaje / OAO** (`VST_Aprendizaje`) | Memoria ecoica de lo oído → sesgo de imitación de la voz del par | Imitación → convergencia léxica (`oao_imitacion_mag`, `oao_echoica_n`, `oao_oido`) |
| **Expresión** (`VST_Expresion`) | Decide SI vocalizar y con qué gesto (freq/intensidad/pausa/repetición) | Conducta vocal emergente (`expr_vocalizando`, `g_freq/g_intensidad/g_pausa/g_repeticion`) |
| **Órgano de Comunicación** (`VST_OrganoComunicacion`) | Sintetiza la voz (R2D2), publica el estado/gestos al par, emula palabras del otro | El "habla" y el puente entre organismos (`voz_emitida`, `voz_creadas`, `voz_aprendidas`) |
| **Órgano Fonador** (`VST_OrganoFonador`) | Aparato de síntesis vocal | Materializa la vocalización |
| **OVE — Valoración Experiencial** (`VST_OrganoVE`) | Valora la experiencia (favorable/neutra/desfavorable) y la expresa en la **cara** | Lectura afectiva (`cara_valoracion`, `ove_experiencias`). **Read-only: NO influye en la conducta** |
| **Órgano RC** (`VST_RC_A` / `VST_RC_B`) | Reactor cosmosemiótico (acople A/B) | Gobierna el acoplamiento de la díada |

*(`bateria_<organelo>.py` en `experimentos/` valida cada uno por separado — paradigma controlado.)*

---

## 4. Capacidades actuales (qué saben hacer hoy)

- **Vivir** (metabolismo + homeostasis + clausura operacional / "salud del cierre").
- **Oír** el mundo y al par (sonido binaural L/R por el Soma).
- **Vocalizar** conductas vocales emergentes (decidir si hablar; gesto freq/intensidad/pausa/repetición).
- **Reconocer al otro** (alteridad: presencia, contingencia, agencia).
- **Aprender por imitación** (OAO): memoria ecoica de los gestos del par → sesgo que hace converger su voz.
- **Inventar palabras propias** (cuando su banco no cubre su estado) y **emular las del otro** → **léxico compartido**.
- **Valorar la experiencia** (OVE) y **expresarla en la cara** (boca sonríe / recta / invertida) sin que eso altere su conducta.
- **Orientarse** hacia las fuentes de sonido (giro de cabeza).
- **Persistir entre sesiones**: memoria, metabolismo, OVE y codebook se restauran al "renacer".

---

## 5. Infraestructura Docker

| Contenedor | Puerto | Rol |
|---|---|---|
| `anima-a` | 7788 | Organismo A (interfaz web propia) |
| `anima-b` | 7799 | Organismo B (interfaz web propia) |
| `anima-conversacion` | 9100 | **Observatorio** de la díada (página general) |
| `anima-mcp` | 9000 | Membrana MCP |

- Imagen única `anima-diada:latest`. Volúmenes: `anima_a_data`/`anima_b_data` (disco propio: memoria/codebook),
  `Docker_Historia` (biografía longitudinal, CSV por paso, en el disco LaCie).
- `modo_vida`: `comunicacion` (autostart acoplado), `basal` (solo se oyen entre sí), `experimento` (manual `/start`).
- Variables de ruteo: `ANIMA_MUNDO_CANAL`, `ANIMA_OIDO_PAR` (R para A, L para B), `ANIMA_ESCUCHAR_PAR`, `ANIMA_CONTROL` (real|null|shuffled, controles de falsación).

---

## 6. Páginas de visualización

### 6.1 Páginas de organismo — `http://localhost:7788` (A) y `:7799` (B)
- **Cabeza 3D real** (Three.js `drawVSTCabeza3DReal`): esfera porcelana, ojos hundidos, anillos de oído que
  laten con la energía, **boca evaluativa** que sigue la valoración del OVE (sonríe 😊 / recta 😐 / invertida ☹️).
  La cabeza **gira** hacia donde "mira" (orientación).
- **13 cajas** (tablero editable, gridstack): Metabolismo, Memoria, Alteridad/Intención, Libertad expresiva
  (balbuceo), Salud del cierre, Campo Φ/Soma, Homeostasis, Voz/Comunicación, Alteridad/Agencia, Valor ecológico
  de la voz, Expectativa, Expresión vocal, **Aprendizaje (OAO: ecoica + imitación)**.
- Descarga de CSV y bitácora; controles de entrada (mute L/R, fuente por oído), panel de LEDs del RØDECaster.

### 6.2 Observatorio de la díada — `http://localhost:9100` (la "página general")
Rediseñado por completo esta sesión. Tres pestañas:

**🟢 En vivo**
- **Dos cabezas 3D** (las mismas de las páginas de organismo) mirándose, con placas de nombre, anillos de oído
  latiendo y boca evaluativa. Look moderno (fondo azul pizarra, no oscuro/ominoso).
- **Banner de léxico** que se enciende cuando: 🗣️✨ un organismo **inventa** una palabra, o 🗣️↔ **emula** la del otro
  (→ léxico compartido).
- Transcripción de la "conversación" en vivo + histograma compacto (top-6) de pitos usados.
- **Reproductor de audio** (escuchar a A por la izquierda, B por la derecha).
- **Tablero del observatorio** (gridstack editable) con **14 cajas** en formato díada A/B, con los **mismos nombres**
  que las páginas de organismo + una extra:
  Metabolismo, Memoria, Alteridad/Intención, Libertad expresiva, Salud del cierre, Campo Φ/Soma, Homeostasis,
  Voz/Comunicación, Alteridad/Agencia, Valor ecológico, Expectativa, Expresión vocal, Aprendizaje (OAO),
  **Acople (OI A↔B)** (extra propia de la díada) y **🔤 Léxico: palabras propias ↔ compartidas**.

**🕮 Historia** — biografía longitudinal acumulada (por día / por pito), descarga del historial.

**🫀 Circuito vivo** — *anatomía viva de la díada* (lo nuevo más pedagógico):
- Los dos organismos como **células** (membrana translúcida + **núcleo** = Campo Φ pulsante; el **Mundo** queda
  FUERA de la membrana, como perturbación externa).
- Cada organelo es un nodo que **late con su valor real**; las flechas muestran el **flujo del campo** con partículas.
- La **línea Mundo→Soma** ("estímulo") cruza la membrana; los **puentes dorados** = voz de A → Soma de B (y viceversa) = el lazo se cierra.
- Nodo **OVE·Cara** con la sonrisa que cambia en vivo (conexión de "lectura", sin causalidad).
- Botón **✴ Trazar el campo** (enciende los organelos en orden de flujo) y **click en un organelo** → qué hace / para qué sirve.
- Puntos de categoría por organelo (sensorial / núcleo / interno / relacional / salida) + leyenda.

> Nota técnica: el observatorio sirve con cabeceras **anti-caché**, así que cualquier cambio se ve con un refresco normal.
> El poller fusiona `/estado` (curado) + `/ultima_fila` (256 columnas) → `/datos` expone todo para las cajas.

---

## 7. Infraestructura de audio y monitoreo de fuentes

El **RØDECaster Pro II** expone su Main Multitrack de **20 canales** por USB. `VST_AudioServer.py` (nativo en el Mac)
los captura todos y los sirve por TCP al contenedor. Mapa de canales→entrada (verificado):
`1-2 MainMix · 3-4/5-6/7-8 Combo1-3 · 9-10 Bluetooth · 11-12 USB2 · 13-14 USBMain · 15-16 SMARTPads · 17-18 USBChat`.

Tres niveles para saber **qué escucha cada organismo** (construidos esta sesión):

- **Nivel 1 — energía por entrada** (`VST_AudioServer.py --log-canales`): RMS de cada entrada del Rode cada 0.25 s →
  CSV (`canales_rms_<fecha>.csv`) con `ts_real`. Cruza con la fisiología para saber qué fuente sonaba y cuándo. *No guarda audio.*
- **Nivel 3 voz** (`VST_Transcriptor.py`): transcribe en vivo un canal (def. Bluetooth) con **faster-whisper local**,
  por ventanas con compuerta de voz; guarda **solo el texto + timestamps** (JSONL). *No guarda audio.* (Modelo `base` precargado.)
- **Nivel 3 música** (`VST_ReconocedorMusica.py`): identifica el **tema** (título/artista) de un canal (def. SMARTPads)
  vía **Shazam** (shazamio), por ventanas; guarda **solo el resultado + timestamps** (JSONL). *No guarda audio.*
  ⚠️ Corre con su **venv aislado** `venv_musica/bin/python3` (Python 3.11) — NO con el venv principal.

*(El Nivel 2 — grabar el audio crudo por canal — se descartó a propósito en esta etapa.)*

---

## 8. Experimentos

- **Paradigma controlado** (`experimentos/bateria_*.py`): una variable manipulada, tiempos fijos, busca causalidad.
  Hay una batería por capacidad (aprendizaje, alteridad, expectativa, homeostasis, metabolismo, memoria, valor
  ecológico, expresión, libertad expresiva, reconocimiento del otro, etc.). `bateria_factorial.py` cruza tipo de
  audio × relación A-B × ablaciones × ciclos repetidos (test de aprendizaje).
- **Paradigma ecológico** (sesiones en vivo con el Rode): entorno rico e impredecible (película por Bluetooth,
  música por Pads, cortes manuales). Más cercano a condiciones reales; complementa al controlado.
- **Análisis**: `analisis/analizar.py` (DuckDB, para los CSV grandes de `Docker_Historia`).

---

## 9. Hallazgos recientes clave (esta sesión)

1. **Bug crítico de imitación — reparado.** El lazo de imitación leía los gestos del par desde
   `/comunicacion/estado`, pero el organismo publicaba ese estado **antes** de calcular el gesto del paso → el par
   recibía `g_*=None` → memoria ecoica de ceros → **imitación = 0 siempre**. Fix: `estado()` ahora fusiona el gesto
   actual. **Verificado: la imitación dispara.** (Invalidó la conclusión previa "solo el Rode en vivo dispara imitación").

2. **Batería factorial post-fix (2 ciclos):** el **audio guardado SÍ dispara imitación** (14/14 condiciones).
   La imitación **depende de oír al PAR, no del mundo** (hasta el SILENCIO dispara). **Aprende con la repetición**:
   la magnitud sube de Ciclo 1 a Ciclo 2 (media ~0.61 → ~0.81). La VOZ llena más la memoria ecoica que música/ruido,
   pero la magnitud no es proporcional al llenado. (Paquete de datos primarios entregado al equipo.)

3. **Mecanismo de imitación, preciso (sesión viva analizada):**
   - `oao_oido = max(energia_L, energia_R)` — **NO es "¿oye al par?"**, es la energía total oída (dominada por el
     mundo cuando es continuo). La etiqueta del LÉEME está mal y conviene corregirla.
   - La imitación necesita **dos cosas**: (a) los **gestos del par** (contenido, vía HTTP — por eso `imit` correlaciona
     ~0.7 con la ecoica) y (b) **cualquier sonido sobre el umbral** que mantenga abierta la **compuerta** de refresco.
   - Cortar el mundo **solo baja la imitación cuando el par también está callado**; si el par vocaliza durante el corte,
     la imitación se mantiene/sube. → **No hay contradicción con la batería** (silencio sostenido con par hablando vs.
     corte transitorio con par callado son el mismo mecanismo).
   - El ruteo en vivo invierte canales a propósito (en A el par entra por la **derecha**) para que el organismo gire
     hacia el par; pero en los datos la **orientación sigue al mundo** (continuo y fuerte), no al par (intermitente).
     → posible acción: subir ganancia del par o hacer el mundo intermitente.

---

## 10. Pendientes / próximos pasos sugeridos

- Corregir la etiqueta `oao_oido` ("energía total oída", no "¿oye al par?") donde se use, y revisar la frase del
  observatorio sobre que "la voz altera el Soma del otro" (el lazo de **imitación** se cierra por gestos; el audio cierra compuerta/orientación).
- Corrida **controlada** que separe "compuerta de sonido" de "contenido del par" (mundo cortado × par hablando/callado, cruzados).
- Probar el reconocedor de música con música real en vivo (canal 15).
- (Opcional) Script único que lance los tres monitores de audio (energía + voz + música) para una sesión.
- Completar más ciclos de la batería factorial para confirmar si la curva de aprendizaje se estabiliza.

---

## 11. Mapa de archivos (referencia rápida)

```
Célula_Madre/
  web/   VST_CelulaMadre_WebLive_A.py · _B.py     (organismos + páginas 7788/7799, cabeza 3D)
  conversacion/ vst_conversacion.py                (observatorio :9100: En vivo / Historia / Circuito vivo)
  organelos/  VST_*.py                             (metabolismo, memoria, aprendizaje/OAO, OVE, alteridad, …)
  audio/  VST_AudioServer.py                        (captura Rode 20 canales + Nivel 1 --log-canales)
          VST_Transcriptor.py                       (Nivel 3 voz: Whisper local)
          VST_ReconocedorMusica.py                  (Nivel 3 música: Shazam — usar venv_musica)
  experimentos/  bateria_*.py · experimento_*.py    (paradigma controlado + ecológico)
  analisis/  analizar.py                            (DuckDB para CSV grandes)
  docker/  docker-compose.yml · Dockerfile          (díada: anima-a/b/conversacion/mcp)
Docker_Historia/                                    (datos primarios: fisiología por paso, CSV)
venv/            (principal)        venv_musica/     (aislado, solo reconocedor de música, py3.11)
```
