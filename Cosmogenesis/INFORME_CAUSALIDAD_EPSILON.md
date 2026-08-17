# INFORME — Test de causalidad de ε (régimen de densidad máxima)

**Autor:** Claude (sesión CC) · **Para:** Grok · **Fecha:** 2026-06-29
**Ejecutado en:** iPad Pro M1 (Carnets/scipy) — 150 corridas en ~5.4 min/pasada
**Código:** `cg001_ipad_causalidad.py` (física idéntica a `cg001_field.py`, no se tocó la dinámica)
**Logs:** `cg001_causalidad_20260629_124422_n30.json` (γ=8) · `cg001_causalidad_20260629_125649_n30.json` (γ=0)

---

## 0. Por qué este test (contexto)

El **barrido fino** de producción (960 corridas, 30 semillas, RUIDO 0.02→0.001) cerró con
**negativo limpio: NO hay banda.** El signo de la divergencia de concentración nunca llegó
a 0.83 en ningún punto (máx. 0.63 en el extremo). La señal del smoke-test (Δconc +89) era
ruido de pocas semillas. **Conclusión: en la cola lisa, la concentración es artefacto de
ratio ε/RUIDO, no operación de ε.**

Eso movió la pregunta al **régimen correcto** y al **observable correcto**:

- **Régimen:** la singularidad fiel es **densidad máxima** de diferencias (C-N1.3, §43, §7),
  no campo casi-vacío. → **RUIDO = 1.0 fijo.** No se barre hacia liso.
- **Observable causal:** no concentración (sensible a picos) ni localización-en-centro, sino
  **¿el campo de diferencia `|m_B − m_A|` sigue a la posición de ε cuando ε se mueve?**

---

## 1. Diseño (pre-registrado, no post-hoc)

- **Mover la arruga:** condición B con ε en 4 posiciones (centro, esquina, 2 aleatorias).
  A = sin ε. Misma semilla emparejada A/B → A y B difieren **solo** por ε.
- **Observable:** `D = |m_B − m_A|` (campo de memoria final). `pico = argmax(D)`,
  `dist = distancia toroidal(pico, posición de ε)`.
- **Baseline de caos:** `|m_A(s1) − m_A(s2)|` (dos semillas distintas, sin ε) → nivel de azar
  de dónde cae el pico cuando algo que **no** es ε cambia el campo.
- **Control nicho-OFF:** repetir todo con γ=0 (apaga el nicho #131). Si el pico **sigue** en ε
  con γ=0, la localización era **localidad trivial**; si **se borra** sin nicho, el nicho es
  lo que hace **persistir** a ε (causal fuerte).
- **Criterio de lectura fijado de antemano:** comparar la **razón huella(ε)/huella(caos)**
  entre γ=8 y γ=0. Si `razón(γ=0) ≪ razón(γ=8)` → el nicho sostiene a ε selectivamente.
- 30 semillas/posición (#109). Lectura al **t final** (a t bajo el pico está pegado a ε trivialmente).

---

## 2. Resultados

|                 | huella ε `max\|D\|` | caos `max\|D\|` | razón ε/caos | pico→ε (dist) | en arruga (<3) |
|-----------------|:---:|:---:|:---:|:---:|:---:|
| **γ=8 (nicho ON)**  | **0.74** | 117.5 | **0.0063** | **0.00** | 100% (4/4 pos) |
| **γ=0 (nicho OFF)** | **6×10⁻⁷** | 0.0016 | **0.0004** | ~1.0 | 100% (4/4 pos) |

- Baseline de caos: pico a **30.7** del centro (γ=8) / **26.5** (γ=0) → coincide con el azar
  esperado (~28). El baseline es un null correcto.
- Localización: en **ambas** pasadas, el pico de `|m_B−m_A|` cae sobre ε (dist 0–1, 100% de
  semillas, en las 4 posiciones, incluidas las aleatorias). Descarta la rama "caos sin relación
  con ε".

---

## 3. Lectura

### 3.1 El nicho es NECESARIO para que ε persista
Sin nicho (γ=0) la huella de ε es **6×10⁻⁷ — se borró**: el campo relaja a liso y olvida ε
(y olvida todo; el caos también cae a 0.0016). Con nicho (γ=8) la huella de ε es **0.74 —
persiste**. Diferencia ~**10⁶×**. **La persistencia mínima de ε deja estructura localizada
solo porque el nicho (#131, C-N2.6) la sostiene.**

### 3.2 Y es SELECTIVO (criterio pre-registrado)
La razón pasó de **0.0004 (sin nicho) → 0.0063 (con nicho) = 16×**. Si el nicho amplificara
todo por igual, la razón no cambiaría. Cambió: amplifica la huella de ε **16× más** que el
caos. El nicho es retroalimentación positiva (más memoria → menos relajación → más asimetría),
y la **semilla coherente** de ε gana la competencia local que el ruido incoherente no gana.
Por la regla fijada (`razón(γ=0) ≪ razón(γ=8)`), **esto es el ramo causal-fuerte: ε opera.**

### 3.3 Matiz de magnitud (no se barre)
La huella de ε (0.74) es **0.6% de la variabilidad entre semillas** (117.5). ε siembra
estructura persistente **real pero MENOR** — un nicho localizado que sobrevive 400 pasos,
coexistiendo con los nichos mucho mayores que brotan de la rugosidad inicial completa.
**ε produce historia; no la domina.**

---

## 4. Veredicto §115

**¿Basta la persistencia mínima de una diferencia para producir historia?**
En el régimen de densidad máxima, con observable causal y control nulo:

> **Sí.** ε —la asimetría primordial mínima— produce una estructura localizada y persistente,
> y la **causa es el nicho**: sin el mecanismo de memoria ε se disuelve; con él persiste, se
> localiza en ε, y sigue a ε donde se ponga. **Con el matiz:** la estructura es menor, no
> dominante.

Contraste con el barrido fino: ahí la "señal" (concentración en cola lisa) era artefacto de
ratio. Aquí, régimen correcto + observable causal → ε **sí** opera, y lo dice el **cociente**,
no una inferencia.

---

## 5. Nota metodológica (honestidad de proceso)

La prueba mínima (L=24, 80 pasos) había sugerido razones casi iguales con y sin nicho
(~0.01) → apuntaba a **localidad trivial**. La predicción de Claude se inclinó ahí. **El dato
de producción la falsó:** con L=64/400 pasos el nicho tiene tiempo de separar, y la razón se
abre 16×. 80 pasos no bastaban para que el nicho actuara; 400 sí. La inferencia estaba mal;
el cociente la corrigió.

---

## 6. Control de cierre — ¿persistencia o pico coherente? (RESUELTO)

**Código:** `cg001_ipad_persistencia.py` · **Log:** `cg001_persistencia_20260629_143845_n30.json`
**Pregunta:** ¿el nicho favorece a ε por ser **persistente**, o solo por ser un **pico
coherente de una celda**? **Diseño:** ε se pone en φ y se **remueve** tras `t_remove` pasos
(resync φ_B = φ_A; solo queda en la memoria lo que depositó). Se barre t_remove y se compara
la huella final con la persistente (nunca removida). γ=8, 30 semillas, L=64/400.

| t_remove (pasos con ε) | huella / persistente |
|:---:|:---:|
| 1   | 0.035 |
| 5   | 0.054 |
| 20  | 0.064 |
| 100 | 0.085 |
| **persistente (400)** | **1.000** |

**Resultado: PERSISTENCIA.** Con ε presente 1 paso, la huella es 3.5% de la persistente; crece
monótona con cuánto tiempo estuvo ε. Removerla tras **100 de 400** pasos deja solo **8.5%** —
los últimos 300 pasos (con ε presente) aportan el **91.5%** de la estructura. El efecto
**compone**: no es disparar-y-trabar (un seed que se autosostiene), es **amplificación
dependiente de que ε siga presente**. El nicho necesita ε para *seguir creciendo*, no solo
para arrancar.

**Mecanismo completo:** la estructura en ε requiere **ambas** cosas — el nicho (γ>0; γ=0 la
borra, §3.1) **y** que ε **persista** en φ (removerla la colapsa). Quita cualquiera → no hay
estructura.

**Lectura teórica:** es **C-N1 directo** (`S = persistencia mínima`): lo que genera estructura
es la diferencia **sostenida**, no un evento momentáneo (C-N1.1). El nicho no premia cualquier
chispa; premia la **persistencia**. Esto cierra el §115 sin hueco: **ε opera, y opera
precisamente porque persiste.**

---

## 7. Rendimiento (infraestructura)

- iPad Pro M1 vía Carnets+scipy: **~3 s/corrida**, 150 corridas en **~5.4 min/pasada**.
- iMac Intel (referencia): ~14 s/corrida.
- El barrido fino de 960 corridas: ~48 min en el iPad vs ~3.8 h en el iMac.
- Las semillas son independientes → se puede repartir iPad ↔ iMac por rango de semillas.

---

*Documento de resultados. Física no modificada: campo φ, asimetría relacional, nicho
history-dependiente, termodinámica medida. Solo se cambió régimen (RUIDO=1.0) y observable
(causal, mover la arruga + control γ=0), según lo acordado.*
