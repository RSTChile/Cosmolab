# Instrumentos de comprensión — *ver y oír* a los organismos

Seis instrumentos **de solo lectura** (no tocan a los organismos ni a los contenedores; leen
`Docker_Historia/`). No imponen hipótesis: hacen visible/audible la estructura que los organismos
**ya formaron**, para entenderlos sin reprocesar miles de pasos a mano. Pensados para correr en
cualquier momento, incluso con la díada viva.

Todos comparten el mismo patrón: leen las biografías fisiológicas (`organismo_ANIMA_*/fisiologia/*.csv`)
y/o las voces guardadas (`organismo_ANIMA_*/voz/*.wav`), submuestrean con `DOWNSAMPLE`/`STEP` para que
horas de vida quepan en un vistazo, y son robustos a cabeceras antiguas (columnas que aún no existían).

| Instrumento | Pregunta que responde | Salida |
|---|---|---|
| `descubrimiento.py` | ¿Qué estructura formó el organismo **sin que la definiéramos**? | regímenes de estado (k-means), secuencias de gesto recurrentes (n-gramas), similitud de repertorio A↔B |
| `mapa_repertorio.py` | ¿Cómo es y cómo **deriva** el espacio de gestos? ¿A y B convergen? | PNG (gestos A vs B, opacidad=tiempo) + ratio separación/dispersión |
| `momentos_interesantes.py` | ¿**Dónde mirar** en una biografía enorme? | top-N instantes que se salen de la rutina (picos, contingencia, gesto nuevo, transición voz↔silencio) con timestamp |
| `inspector_momento.py` | ¿Qué pasaba **exactamente** en un instante? | volcado lado a lado A vs B del estado completo de la díada en ese ts |
| `diada_relacional.py` | ¿Cómo es la **relación** A↔B? (el *nosotros*) | correlación cruzada con desfase (quién lidera) vs control, convergencia del gesto por tramos, turnos |
| `diario_vocal.py` | ¿Cómo **suena** la evolución de la voz? | un WAV concatenado y comprimido de la vida vocal ('tic' agudo = cambia la hora) |

## Uso

```bash
PY=../../venv/bin/python   # o la ruta a venv/bin/python

# 1) estructura no supervisada (regímenes + frases vocales)
DOWNSAMPLE=80 $PY descubrimiento.py

# 2) mapa del repertorio (genera Docker_Historia/MAPA_REPERTORIO.png)
DOWNSAMPLE=60 $PY mapa_repertorio.py

# 3) dónde mirar
ORG=ANIMA_A DOWNSAMPLE=20 TOPN=15 $PY momentos_interesantes.py

# 4) inspeccionar un instante (sin TS = el último; con TS = ese)
$PY inspector_momento.py
TS='1782495730.868' $PY inspector_momento.py

# 5) la relación A↔B
DOWNSAMPLE=10 $PY diada_relacional.py

# 6) escuchar la vida vocal (genera Docker_Historia/DIARIO_VOCAL_ANIMA_A.wav)
ORG=ANIMA_A STEP=12 MAXVOCES=400 $PY diario_vocal.py
afplay ../../Docker_Historia/DIARIO_VOCAL_ANIMA_A.wav
```

## Flujo típico

`momentos_interesantes.py` te dice **dónde** mirar → copias un timestamp → `inspector_momento.py`
te deja **ver** ese instante entero. `descubrimiento.py` y `mapa_repertorio.py` dan la vista de
conjunto (qué modos de ser, qué repertorio); `diada_relacional.py` mira el vínculo; `diario_vocal.py`
es para **oír** lo que los números no transmiten.

## Lecturas honestas (al 26-jun-2026, antes de ANIMA-5 con mundo)

- **Sin secuencias vocales consolidadas**: los n-gramas de gesto aparecen ×1 → todavía exploración,
  ninguna "frase" se ha fijado.
- **Repertorios A/B muy solapados** (ratio ≈ 0.19): A y B ocupan casi el mismo espacio de gesto —
  no diferenciación (¿misma caminata aleatoria, no convergencia por imitación?).
- **La díada sincroniza *cuándo* hablar** (corr. vocalización ≈ 0.87 vs control 0.06) **pero el
  *contenido* del gesto no converge** (la distancia A↔B no baja en el tiempo). Cautela: parte de esa
  sincronía de *timing* puede venir del reloj externo compartido (las fases de ANIMA-5), no solo del
  acoplamiento A↔B; el control de desfase descarta coincidencia, no el arrastre externo común.
