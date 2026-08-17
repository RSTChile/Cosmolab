# INFORME — CÉLULA MADRE COSMOSEMIÓTICA

**Para:** Equipo transinteligente RMD 2.0
**Autor del proyecto:** Alexis López Tapia (Investigador Principal)
**Fecha:** 2026-06-22 · **Estado:** arquitectura base operativa (no experimento; consolidación)
**Marco teórico:** *Teoría Cosmosemiótica — Versión Canónica* (17-06-2026, 51 pág.)

> Documento de consolidación. Reúne en un todo coherente la reescritura del organismo
> individual VSTCosmo como una **célula madre pluripotente y modular**, con sus organelos
> anclados nodo a nodo a la Teoría Cosmosemiótica. Todas las cifras son reproducibles
> ejecutando los archivos citados (Python 3.13, `venv/bin/python3`).

---

## 0. Resumen ejecutivo

Se reorganizó el organismo individual consolidado (antes `VST_Organismo_Individual_v2.py`,
hoy **`VST_Celula_Madre_001.py`**, monolítico) en una **arquitectura de organelos**: un
genoma del que se *transcribe* un organismo expresando capacidades modulares,
conmutables y evolucionables. La arquitectura encarna seis principios bio-cosmosemióticos
y realiza, como código ejecutable, los bloques **5 (Consciencia), 7 (Libertad Funcional),
8 (Dinámica Evolutiva)** y la **homeostasis** de la teoría canónica.

**Resultado medible (Índice de Organismicidad Integrada, OI, O-N9.14):**

| Configuración | OI | Nivel |
|---|---|---|
| célula mínima (solo motor) | ≈0.000 | no organismal (κ_LF ✗) |
| + Bloque 7 (libertad funcional) | ≈0.147 | no organismal (κ_LF ✓) |
| + Bloque 5 (consciencia, R₂ real) | ≈0.244 | no organismal (LF-3) |
| + Bloque 8 (exaptación, XE) | ≈0.444 | **protoorganismo** |
| + Homeostasis (H) | **0.648** | protoorganismo |
| **Multicelular (ME = S_shared)** | **0.858** | **ORGANISMO PLENO** |

La célula individual alcanza **protoorganismo alto**; el **organismo multicelular** cruza
a **organismo pleno** — y lo hace porque la **memoria externalizada (ME = S_shared)** solo
existe en el colectivo. El salto a multicelular es "por definición": la **misma**
homeostasis aplicada a otra escala (universalidad estructural, C-N2.8.14).

---

## 1. Qué es la Célula Madre (y qué no es)

**Es** el análogo de una célula madre: un organismo que contiene **en potencia todas las
capacidades exploradas por el linaje VSTCosmo** (v70–v182), modulares como organelos,
conmutables por expresión, y sujetas a las leyes evolutivas generales (adaptan y exaptan).

**No es** un experimento (no hay hipótesis que falsar aquí) ni una optimización. Es
**consolidación arquitectónica**: pasar de un cuerpo monolítico a un genoma de organelos
anclado a la teoría, capaz de crecer (más bloques) y de escalar (multicelular).

---

## 2. Los seis principios fundadores

1. **Pluripotencia.** Contiene en potencia todo lo explorado, conmutable, MENOS lo podado
   con razón. Incluye la **díada** (no es "fuera de alcance": es potencial dormido).
2. **Modularidad-organelo.** Cada capacidad = organelo encapsulado (membrana + estado +
   función), que se comunica SOLO por el citoplasma compartido (`Milieu`).
3. **Economía.** No se gasta lo que no se usa; presión intrínseca, nunca regulador externo
   (revivir el "Pastor" sería Shannon encubierto).
4. **Escalamiento alométrico (Kleiber).** A más complejidad, menor metabolismo/unidad y
   tiempos más largos. *Extensión nuestra, no nodo del canon — declarado honestamente.*
5. **Loci reservados.** El genoma reserva slots VACÍOS para lo anticipado. El primero: la
   **genética del altruismo de Boorman** → multicelularidad **voluntaria** (no impuesta).
6. **Adaptación y exaptación.** La identidad/función de cada organelo NO está cerrada: el
   `Milieu` público es la *condición de posibilidad de la exaptación* (una señal secretada
   para un fin puede ser co-optada para otro). El genoma describe **de dónde parte** el
   organismo, no a dónde llega.

> Dos ejes que no se contradicen: **invariante de ingeniería** (al refactorizar no rompemos
> el baseline; control congelado v1 = `VST_Organismo_Individual.py`) y **apertura biológica**
> (el sustrato puede evolucionarse a sí mismo en runtime).

---

## 3. Arquitectura: el motor del genoma (`VST_Genoma.py`)

Cuatro abstracciones mínimas:

| Abstracción | Encarna | Función |
|---|---|---|
| **`Milieu`** | citoplasma | señales públicas → **sustrato de la exaptación** |
| **`Organelo`** | membrana + metabolismo | contrato `percibir → metabolizar → secretar`; plasticidad en `self.plast`; campo `nodo_canonico` (trazabilidad teoría→código) |
| **`LocusReservado`** | gen no-codificante | slot declarado pero vacío (principio 5) |
| **`MedidorComplejidad`** | fisiología de conjunto | M, `s(M)`, `r(M)` — ley de Kleiber |
| **`Organismo`** | la célula | ciclo metabólico en orden de dependencia; `salud()` (Λ_Cos, OI, κ); `quien_soy()` |

**Conmutabilidad estructural:** silenciar un organelo = no incluirlo en el ciclo = sin
efecto NI costo, por construcción (no un `if flag` disperso).

---

## 4. Censo de organelos (por bloque cosmosemiótico)

Estado: ✅ presente · ◐ parcial (limitación reconocida) · ⊘ reservado.

### Motor / base (espina CM001)
| Organelo | Nodo canónico | Función | Estado |
|---|---|---|---|
| `presion_desacople` | deriva O-N2.1/O-N4.1 | Presión de desacople (arousal): integra `e_R·(1−A)`; gatea juego/ritual. **NO es C_b** | ✅ |
| `fatiga` | (V155/V180c) | Tiempo biológico: historia irreversible + fatiga recuperable | ◐ (recuperación −6%, residuo V150) |

### Bloque 5 — Consciencia Funcional (`VST_Bloque05_ConscienciaFuncional.py`)
| Organelo | Nodo | Función | Estado |
|---|---|---|---|
| `consciencia_basica` | **O-N5.1** | `C_b = |R₁|`: registra el propio estado representacional | ✅ |
| `meta_representacion` | **O-N5.2** | `R₂ = R(R)`: auto-modelo; **funda LF_struct** (O-N13.8) | ✅ |
| `self` | **O-N5.3** | `Self = operador(R₂)`: coherencia/identidad | ✅ |

### Bloque 7 — Libertad Funcional (`VST_Bloque07_LibertadFuncional.py`)
| Organelo | Nodo | Función | Estado |
|---|---|---|---|
| `juego` | **O-N7.2-1 / O-N10.7** | desacople ENACTUADO (acción con significado suspendido) | ✅ |
| `ritual` | **O-N7.2-2 / O-N7.3** | desacople FIJADO en estructura reproducible no negable; inhibe juego | ✅ |
| `negacion_operativa` | **O-N10.1/10.2/10.13** | el "No": opera sobre la representación y suspende R→Acción (¬R_op, requiere LF≥1) | ✅ |
| `LF` | **O-N7.1 / Dicc.111,114,115** | mide `LF_struct` (de R₂) y `LF_op = LF_struct·(1−INR)`; escala LF-0..3 | ✅ |

### Bloque 8 — Dinámica Evolutiva (`VST_Bloque08_DinamicaEvolutiva.py`)
| Organelo | Nodo | Función | Estado |
|---|---|---|---|
| `mutacion` | **O-N8.1/8.1b** | `ΔR_aleatoria` sobre el error no filtrado (RNG sembrado) | ✅ |
| `adaptacion` | **O-N8.2 (ΔLF≈0)** | `argmax A_sys-env` con Ωop constante (afina, no abre dominio) | ✅ |
| `exaptacion` | **O-N8.3/8.5/8.19** | en límite adaptativo **con reserva (PRE)** → ΔΩop>0, ΔLF>0, acumula **XE** | ✅ |
| `consciencia_metacognitiva` | **O-N8.4** | `C_m` emerge si `C_b` falla sostenido ∧ hay LF (reorganización) | ✅ |
| `activacion_latente` | **O-N8.12** | detecta déficit → disparador de la **pluripotencia** | ✅ |

### Homeostasis (`VST_Homeostasis.py`)
| Organelo | Nodo | Función | Estado |
|---|---|---|---|
| `homeostasis::<var>` | **C-N5.1 / O-N6.1 / O-N9.14 (H)** | mantiene una variable en rango viable; H∈[0,1]. **Scale-invariant** (célula y colectivo) | ✅ |

> **Verificación cruzada (deriva nombre≠función resuelta):** lo que CM001 llamaba "Cb" NO
> computaba la consciencia básica canónica `C_b=|R₁|` sino la **presión de desacople**
> `e_R·(1−A)`. Se separó en dos organelos con dos nombres: `presion_desacople` (arousal,
> motor) y `consciencia_basica` (C_b canónico, Bloque 5). *Regla del proyecto: leer la
> fórmula, no el nombre.*

---

## 5. El locus reservado — genética del altruismo (Boorman)

| Locus | Nodo | Propósito | Estado |
|---|---|---|---|
| `altruismo` | **O-N8.19 (PRE)** · O-N3.4b (Ψ_alma) | gobernanza de la cooperación inter-celular: ¿hay organismo voluntario o dos células? | ⊘ **reservado, vacío** |

Es una **instancia del Principio de Reserva Estructural** (O-N8.19: *"Exaptación ⇒
R\Uactual≠∅"* — optimizar todos los recursos cancela la exaptación futura). Por eso se
reserva vacío: desarrollarlo *antes de tiempo* sería **imponer** la multicelularidad
(Shannon a mayor escala). Se especificará **bajando *The Genetics of Altruism* (Boorman &
Levitt, 1980) a la Teoría Cosmosemiótica**. La díada (V182) es el *sustrato mecánico*; este
locus es la *gobernanza* que decide entre **Ψ_alma plena** (sujeto-sujeto, O-N3.4b) y
**desalmamiento** (U0).

---

## 6. Maquinaria de medida (anclada al canon)

- **Λ_Cos** (C-N2.8.12) = `(Δ_struct · LF) / |e_R| · A_sys-env` — salud del cierre.
- **OI** (O-N9.14) = `w_H·H + w_ME·ME + w_XE·XE + w_LF·LF − w_IRDE·IRDE·1(LF≥κ_LF)` —
  organismicidad. Pesos orientativos (`H 0.25 · ME 0.20 · XE 0.20 · LF 0.35`), calibrables
  por dominio (C-N2.8.14a). Umbrales: ≥0.7 pleno · 0.4–0.7 protoorganismo · <0.4 no organismal.
- **Invariantes de viabilidad κ** (C-N2.8.8 + κ_H): κ_P persistencia, κ_Δ diferencia, κ_O
  error acotado, κ_V acoplamiento, κ_LF libertad, κ_H analizabilidad. La célula completa
  satisface **6/6**.

---

## 7. Resultados (reproducibles)

### 7.1 Célula completa — `VST_CelulaMadre.py`
```
M=16.50  s(M)=2.015  r(M)=0.496       (Kleiber: 17 organelos expresados)
Λ_Cos=0.086   OI=0.648 → protoorganismo
invariantes:  ✓ κ_P  ✓ κ_Δ  ✓ κ_O  ✓ κ_V  ✓ κ_LF  ✓ κ_H   (6/6)
```

### 7.2 La cadena canónica funcionando (consciencia → libertad → evolución)
- **Consciencia:** `C_b=|R₁|=4`, `R₂=0.997` (auto-modelo saturado).
- **Libertad:** el R₂ real eleva la LF a **LF-3 "¿Y si...?"** (exploración exaptativa;
  con andamiaje era LF-2). *La consciencia funda la libertad.*
- **Evolución (límite adaptativo, O-N8.5):** ante un cambio de régimen, la adaptación deja
  de bastar y —porque hay reserva (PRE)— la **exaptación abre dominio** (Ωop 1.0→2.385),
  consumiendo reserva (2.0→0.615) y acumulando XE. La metacognición `C_m` **emergió en la
  crisis** (pico 0.33) y decayó al resolverse. *La exaptación (XE) es lo que vuelve
  organismo a la célula: cruza el OI a protoorganismo.*

### 7.3 Multicelular — `VST_Homeostasis.py` (3 células)
```
Kleiber colectivo: M=49.5  s(M)=2.65 (más lento)  r(M)=0.377 (más eficiente/unidad)
H_col=0.857  ME(S_shared)=1.000  XE_col=1.000  LF_col=0.698
OI colectivo = 0.858 → ORGANISMO PLENO
```
La **misma** homeostasis a otra escala; el medio compartido (el "aire") **es** la memoria
externalizada (ME). Kleiber a escala: la colonia es el "elefante" (más lenta, más eficiente
por unidad) frente a la célula "ratón".

---

## 8. Caveats honestos (lo que NO está cerrado)

1. **Cohesión multicelular = andamiaje impuesto.** El OI colectivo (0.858) usa una colonia
   *cableada por nosotros*. El organismo **voluntario** (Ψ_alma sujeto-sujeto) espera el
   **locus de Boorman** (reservado). Hoy es organismo pleno *por definición estructural*,
   no por emergencia.
2. **Organelos ◐ parciales:** `fatiga` arrastra el residuo de recuperación −6% (V150). En el
   manifiesto del genoma también figuran como ◐: memoria episódica (recall no demostrado,
   0/50), inhibición lateral (sin atribución aislada), Λ nativo V122 (salió plano).
3. **Andamiaje de prueba:** `FuenteDemandaDemo` / `FuenteEntornoDemo` inyectan señales que en
   el organismo real vendrían del motor y otros bloques. Están marcados como NO canónicos.
4. **Kleiber es extensión nuestra**, no un nodo del canon (compatible con economía + PRE).
5. **Pesos del OI orientativos** — calibrables por dominio; los niveles (pleno/proto) dependen
   de ellos. No son constantes físicas (C-N2.8.15).
6. **Falta portar la espina motora de CM001** (orientación por gradiente, lateralidad, Kp,
   memoria de ausencia, valencia/deliberación completa, memoria episódica, ritual/Rᴿ verbatim)
   a organelos, **validando contra el control v1** (invariante OFF==v1). Lo implementado hasta
   ahora son los bloques teóricos 5/7/8 + homeostasis, no toda la espina.

---

## 9. Inventario de archivos (registrados en BD `experimentos`, tipo `arquitectura-genoma`)

| Archivo | Rol | Líneas |
|---|---|---|
| `VST_Genoma.py` | motor del genoma (Milieu, Organelo, Kleiber, OI/κ, manifiesto) | 876 |
| `VST_Bloque05_ConscienciaFuncional.py` | C_b, R₂, Self (O-N5) | 279 |
| `VST_Bloque07_LibertadFuncional.py` | juego→ritual→negación + LF (O-N7/10) | 406 |
| `VST_Bloque08_DinamicaEvolutiva.py` | mutación, adaptación, exaptación, C_m, activación latente (O-N8) | 479 |
| `VST_Homeostasis.py` | homeostasis scale-invariant + `Multicelula` (C-N5/O-N6) | 270 |
| `VST_CelulaMadre.py` | **consolidación: punto de entrada único + informe en vivo** | 78 |
| `VST_Celula_Madre_001.py` | organismo monolítico fuente (ex `..._v2`) | 1826 |
| `VST_Organismo_Individual.py` | control congelado v1 (invariante OFF==v1) | 1372 |

**Cómo correr el todo:** `venv/bin/python3 VST_CelulaMadre.py`

---

## 10. Pendiente / roadmap

1. **Desarrollar el locus de Boorman** (Boorman→Cosmosemiótica) ⇒ multicelularidad
   **voluntaria/emergente** (lo único que falta para que el organismo pleno no sea andamiaje).
   *Bloqueado por la traducción del libro en curso.*
2. **Portar la espina motora de CM001** a organelos, validando contra v1.
3. Bloques pendientes del canon: 9 (Ecología Relacional), 10 (Negación — parte ya en B7),
   11 (Crisis), 12 (Conflicto), 13 (IA y acoplamiento diferido), 14 (Termodinámica Semiótica),
   15 (Ética), etc.
4. Resolver los ◐ (recuperación de fatiga; recall episódico; atribución de inhib_lateral).

---

## 11. Cierre

La reescritura demuestra que el organismo VSTCosmo puede expresarse como un **genoma de
organelos anclado nodo a nodo a la Teoría Cosmosemiótica**, medible con sus propios
invariantes (OI, Λ_Cos, κ), y que la **secuencia canónica consciencia → libertad →
exaptación → homeostasis → multicelularidad** produce, paso a paso, el ascenso del OI hasta
**organismo pleno** al alcanzar la escala colectiva. Queda explícitamente **abierto** lo más
importante: que ese organismo colectivo sea **voluntario** y no impuesto — que es,
precisamente, lo que el locus reservado de Boorman gobernará cuando se desarrolle.

> *El genoma describe de dónde parte el organismo, no a dónde llega.*
