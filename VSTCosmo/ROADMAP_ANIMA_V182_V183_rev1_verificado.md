# 🧬 ROADMAP ANIMA — Capa Relacional y Emergencia (rev1, verificado contra código)

**Programa:** VSTCosmo · Serie ANIMA-4 · Cuerpo V180
**Actualización:** 19 de junio de 2026 — *revisado tras lectura de los scripts primarios*
**Reemplaza a:** `ROADMAP_ANIMA_V182_V183_2026-06-19.md` (corregido: cadenas A separadas, C-N9 reetiquetado, §3.3 reescrita)
**Estado:** V182C cerrado · próximo nodo = V182D · **ruta fijada (§6): Ruta 1** — D forzado → V183 preliminar → [E→F condicionado]

Marcadores: ✅ logrado · ⊘ degenerado/fuera de test · ⏳ pendiente · ◻ diferido · ⚠ superado/corregido · ❓ no reverificado en esta pasada

---

## 1. Aclaración que evita perderse: hay DOS cadenas "A", no una

| Cadena | Archivos | Cuerpo | Rol |
|---|---|---|---|
| **A canónica** | `V182A2/A3/A4/A5_*.py` | **Importan V180 verbatim** (cuerpo real) | Línea principal de la capa relacional |
| **A exploratoria** | `V182A-v2 … v13.py` | Cuerpo **reescrito inline** (no importa V180) | Acoplamiento/convergencia + variantes de audio. **No es la línea canónica.** No confundir. |

Todo lo que el informe y este roadmap llaman "V182A.3/.4/.5" es la **cadena canónica**. La cadena `A-vN` queda registrada como exploración previa, fuera de la secuencia oficial.

---

## 2. Mapa maestro: experimento ejecutado → nodo → sustrato → estado

### 2.1 Capa individual (V176–V181)

| Versión | Validó | Cuerpo | Estado |
|---|---|---|---|
| V176 | R_op (negación operativa) | V180 | ✅ |
| V180 | Memoria episódica (es el **cuerpo**, no una validación individual aislada) | V180 | ✅* |
| V177–V179, V181 | Generalización, extinción, conflicto, R_af | — | ◻ no ejecutados |

### 2.2 Capa relacional (V182) — cadena canónica, cuerpo V180 real

| Versión | Validó | Sustrato | Estado |
|---|---|---|---|
| **A.2** | Roles por *acierto de orientación* | V180 real | ⚠ **superado**: la métrica se re-equilibra a ~2.4° y no retiene ventaja por banda; corregido en A.3 |
| **A.3** | Roles emergentes = **valencia ganada** por consolidación real; ablación ON/OFF | V180 real | ✅ |
| **A.4** | Transferencia mutua dirigida (alumno aprende, maestro preservado en ON; promedio degrada en OFF) | V180 real | ✅ |
| **A.5** | Acumulación: cultura (ON) vs regresión a la media (OFF); `min(4 bandas)`, spread, retención % | V180 real | ✅ |

### 2.3 Capa relacional (V182) — sub-pista comunicación, **sin cuerpo**

| Versión | Validó | Sustrato | Estado |
|---|---|---|---|
| B (v2→v8) | Iteraciones de comunicación funcional A→B | `OrganismoEstimador` escalar (FFT), **sin V180** | ⚠ iteraciones (softmax inestable, proxies de correlación) |
| **B-v9** | Comunicación A→B con **nulo por-setpoint** (efecto genuino = real − nulo), 20 réplicas pareadas; 0° marcado ⊘ (centroide degenerado) | estimador escalar, sin cuerpo | ✅ (criterio: efecto genuino > 15%) |
| B.1 | Comunicación bidireccional | — | ❓ en registro del equipo; **no reverificado** por mí en esta pasada |

### 2.4 Capa relacional (V182) — convención

| Versión | Validó | Sustrato | Estado |
|---|---|---|---|
| **C** | Sentido compartido = **convención emergente** (Schelling). Tres brazos: aislada / comunic / memoria. Comunicación **necesaria y suficiente**; memoria **no añade** convención a igual coeficiente | Importa V180 **solo como contenedor de `ValenciaLocal`**; dinámica = modelo de valencia tipo Schelling. **El cuerpo no es el locus** (lo declara el propio archivo) | ✅ (modelo reducido) |

> **Nota de rigor sobre C:** su bitácora documenta **tres auto-falsaciones** antes del diseño final (el premio de coordinación hacía converger hasta el brazo aislado → confound; la "ventaja de memoria" era artefacto de magnitud 0.60 vs 0.10). El diseño válido es *alineamiento sin premio, igual coeficiente, 8 pares de semilla*. Es más riguroso de lo que los resúmenes transmitían.

### 2.5 Emergencia

| Versión | Objetivo | Estado |
|---|---|---|
| V183 | Ψ_alma mínima (irreductibilidad relacional) | ⏳ |

---

## 3. Sobre el mecanismo relacional (para leer bien los resultados)

En A.3/A.4/A.5 el cuerpo V180 **gana** la competencia inicial por consolidación real (valencia sube solo si `|error| < zona_muerta`: es campo, no contador). El **intercambio** entre organismos, en cambio, es interpolación aritmética del escalar de valencia, gateada por confianza:

```
val_i ← val_i + ALFA_INC · conf · (val_j − val_i) ,  conf = σ(K_COMP·(val_j − val_i))
```

La asimetría del receptor (el experto no absorbe del novato porque σ→0) es real y es el hallazgo. Lo que **no** se debe afirmar es "los cuerpos se reenseñan orientando": el cuerpo aporta la competencia; el intercambio mueve el número. En B-v9 y C ni siquiera hay cuerpo.

**Memoria vs comunicación (corrige la §3.3 anterior):** son **dos resultados en dos sustratos**, no una disociación dentro de un mismo sistema. En la tarea de acumulación (A.5, cuerpo V180) la memoria relacional es decisiva. En la tarea de convención (C, modelo de valencia reducido) la memoria no añade nada a igual coeficiente. Tareas distintas, sustratos distintos.

---

## 4. Estatuto de nodos C-N / O-N (verificado contra código)

| Nodo | Enunciado | Estatuto | Sustrato del estatuto |
|---|---|---|---|
| C-N8 / C-N8.1 | Recombinación generativa | Operativo | cuerpo V180 (A.5) |
| O-N9.2 | Mutualismo (ambos ganan) | Operativo | cuerpo V180 (A.5) |
| **C-N9 / C-N9.2** | Sentido compartido (S_shared como convención) | **Operativo en modelo reducido** (Schelling), en camino a operativo pleno | C — modelo de valencia, **no V180** |
| O-N3.4 / O-N3.4a | Subj_sem (reconocer al otro como sujeto) | Ilustrativo | objetivo de V182D |
| κ_LF (C-N2.8.14a) | Libertad funcional mínima del receptor | Ilustrativo | A.5 (gate de confianza) |

*Guarda de plano (O-N16.2d): "operativo" lo es dentro del modelo declarado, no como validación general del nodo.*

---

## 5. Mapa de dependencias

```
V176 R_op ✅  (cuerpo V180)
  │
  ├─ individual diferida: V177 ◻ V178 ◻ V179 ◻ V181 ◻
  │
  └─ CAPA RELACIONAL V182
       ├─ cadena canónica (cuerpo V180):
       │     A.2 ⚠superada → A.3 ✅ → A.4 ✅ → A.5 ✅
       ├─ comunicación (estimador escalar, sin cuerpo):
       │     B v2…v8 ⚠ → B-v9 ✅ ;  B.1 ❓
       └─ convención (modelo de valencia reducido):
             C ✅  → C-N9 operativo-en-modelo-reducido
                 │
                 ▼
          RUTA 1 FIJADA (§6):
          V182D ⏳  →  V183 preliminar  →  [E → F  solo si dispara umbral]  →  V183 final
          (D forzado, no negociable)        (umbral declarado por adelantado)
```

---

## 6. Ruta fijada tras V182C — Ruta 1

**Pendientes reales (ninguno absorbido por remapeo — confirmado en código):**
- **V182D** — alteridad / Subj_sem (A.4 mide transferencia de valencia, **no** `Acuracidad_predicción`).
- **V182E** — negociación (A.5 mide acumulación, **no** `Cesión`).
- **V182F** — empatía (C mide convención, **no** `Transfer_rechazo`/`Facilitation`).
- **V183** — irreductibilidad.
- **V177–V181** — capa individual, diferida (no bloquea V183).

**Decisión (Ruta 1):**

```
V182D  →  V183 preliminar  →  [E → F  solo si dispara el umbral]  →  V183 final
```

- **V182D es jugada forzada y va ya.** No es un escalón más: es el *gate interpretativo*. Sin D, la irreductibilidad de V183 (compresión, I(A;B), sincronización) es desinflable como "solo dos variables acopladas" —cualquier par de sistemas dinámicos acoplados exhibe esas métricas—. Con D (A modela a B como sujeto autónomo: predice su elección y ajusta su conducta proactivamente), V183 pasa a medir irreductibilidad de una estructura de **modelado mutuo entre sujetos**. Saltar a V183 sin D repetiría el error de dependencia ontológica que este roadmap ya corrigió una vez (validar empatía antes que comunicación), ahora en la cima de la escalera. **No negociable.**

- **E → F quedan condicionados, NO descartados.** Se ejecutan solo si V183 preliminar no alcanza el umbral de nitidez. Lo que se condiciona es *evidencia conductual adicional*, no el gate interpretativo (ese ya lo da D). F cubre un flanco distinto de D: D hace que V183 **signifique** modelado mutuo (interpretación); F aporta una irreductibilidad **conductual** —B paga un costo por el estado de A, sin beneficio directo— que un escéptico no puede desinflar como desinfla un número de información mutua (evidencia de otra naturaleza). F no es "más de lo mismo" tras D. Dependencia formal: F exige E validada (precondición canónica V182A–E).

**⚠ Criterio de disparo de E→F — se declara ANTES de correr V183, no después:**

> El gatillo "V183 salió ambiguo/deflactable" es elástico: un escéptico desinfla *cualquier* número como "solo acoplamiento", de modo que la rama condicional podría leerse como siempre-verdadera (correr E→F siempre) o siempre-falsa (convencerse de que "salió nítido" para no correrlos). Para evitar la decisión a conveniencia, el umbral se fija cuantitativamente **antes** de ver el resultado:
>
> - Compresión ≥ `[a definir antes de correr V183]`
> - I(A;B) ≥ `[a definir antes de correr V183]` bits
> - Sincronización ≥ `[a definir antes de correr V183]`
>
> Si V183 preliminar **alcanza los tres** → claim mínimo cerrado (D → V183), E→F no se ejecutan.
> Si **falla alguno** → se ejecutan E → F como refuerzo conductual y se re-cierra V183 final.
>
> *Estos cortes deben fijarse en la sesión de diseño de V182D, antes de cualquier corrida de V183.*

---

## 7. Resumen de una línea

Capa individual: solo V176 validado. Capa relacional: **A.3/A.4/A.5 sólidas en cuerpo V180** (roles, transferencia, acumulación); **B-v9 sólida sin cuerpo** (comunicación con nulo); **C sólida en modelo reducido** (convención, C-N9 operativo-en-modelo). Falta D/E/F y V183. El sustrato no es uniforme y el roadmap ahora lo dice en cada fila.
