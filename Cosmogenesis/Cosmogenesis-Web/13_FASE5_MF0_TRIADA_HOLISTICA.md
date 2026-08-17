# 13 — Fase 5 · Option D · MF-0 Tríada holística (canónico)

**Fecha:** 2026-07-21  
**Decisión de director (Alexis):** no es solo plasma; es **tríada basal** acoplada desde t=0.  
**Consenso equipo:** 7/7 con la precisión holística.  
**Precedente:** [11_…](11_RECHAZO_CIERRE_Y_REENCUADRE_DOMINIOS.md), [12_…](12_FASE5_DOMINIO_F_ORDEN_PARTIDA.md) — el orden *secuencial* MF-1→2→3 de 12 queda **subsumido** (no borrado como historia; **no** es el plan de implementación).

---

## 0. Objeción núcleo (1 frase, canónica)

> Topología S>0 prueba persistencia mínima no nula en el dominio matemático; pero **plasma** (estado de materia), **expansión supralumínica** y la **banda de tiempo genético** (pre-tiempo emergente post-átomos) son el proceso físico post-singularidad. Si no entran **juntas desde el comienzo**, se reintroduce el error *todo ← partes* y el modelo no puede ser holístico por etapas.

---

## 1. Tríada basal (variables que no se pueden “añadir después”)

| Pata | Símbolo (modelo) | Rol físico (relato) | Error si se omite al t=0 |
|------|------------------|---------------------|---------------------------|
| **Plasma** | ρ, P, T (estado) | Materia/energía en ventana ~10¹⁵–10¹² K (mapeo); T_c ~ escala de reporte | Solo topología o solo ℰ sin medio |
| **Expansión supralumínica** | a(t), H=ȧ/a | Desacople causal; n∝a⁻³, T∝a⁻¹ (adiabasis del modelo) | H topológico fijo 0.0025–0.005 sin dinámica de escala |
| **Tiempo genético** | t_g ∈ banda pre-átomos | Reloj del proceso **antes** del tiempo emergente t_e post-átomos | Confundir t_g con t_e o con “pasos de malla” sin semántica de etapa |

**Principio holístico:** el todo del proceso en esta etapa es **mayor que la suma** de un experimento de masa + uno de carga + uno de V(r).  
Masa, K_eff, V_eff son **lecturas del mismo sistema acoplado**, no módulos que se encienden en serie para fabricar el todo.

```
        ┌──────────── Plasma (ρ,P,T) ────────────┐
        │              ↕︎ acoplado                 │
 Expansión a(t), H(t) ←──→  Tiempo genético t_g   │
        │         (misma evolución)               │
        └──────── observables m, K, V ────────────┘
```

---

## 2. Corrección al orden de la doc 12

| Plan anterior (12) | Estado tras precisión Alexis |
|--------------------|------------------------------|
| F0 = ℰ,P,T solas → luego MF-1 → MF-2 → MF-3 | **Insuficiente / reduccionista** si MF se implementan como mundos separados |
| Qwen/Perplexity “MF-2 primero” lineal | **Vetado** — orden lineal = todo←partes |
| Plan canónico ahora | **MF-0 = un solo sistema con tríada desde t=0**; m(t), K_eff(t), V(r,t) se miden **en paralelo** sobre ese sistema |

### Qué sí se conserva de 12

- Dominio T como **condición de contorno / semillas**, no como objeto a forzar a 1/1836.  
- Criterio de éxito F: no gates MS; sí desplazamiento, V_eff no solo contacto, orden extensivo, kill-switches.  
- Anti-Shannon y kill-switch holístico (abajo).  
- Documentar dominio T como fase (acta 08–12).

### Qué cambia

- No hay “sprint solo masa” ni “sprint solo Debye” que reemplace la tríada.  
- La **implementación** puede tener hitos de *código* (esqueleto a(t) → EOS → lectura m → lectura K → lectura V), pero el **modelo científico** no se declara válido si alguna pata de la tríada está apagada o es paramétrica muerta.

---

## 3. Option D (única opción de cierre de fase estratégica)

| Paso | Contenido |
|------|-----------|
| **D1** | Cerrar **acta dominio T** (plataforma k3, U(1), 3gen, banda + límites masa O(1), K_c crossover, V contacto, F nula) — docs 08–11, galería |
| **D2** | **Pre-registro MF-0** tríada a(t), T(t), ρ(t), t_g + EOS + observables simultáneos + NULL + kill-switches |
| **D3** | Implementación **un sistema** L~30 (o escala acordada): a,T,ρ acoplados desde t=0 en banda t_g; medir m(t), K_eff(t), V(r,t) **juntos** |
| **D4** | Informe contraste T vs F — sin declarar Teoría cerrada |

**No** Option A (cerrar Teoría).  
**No** B/C de forzar 1/1836 o módulos aislados.

---

## 4. Especificación mínima MF-0 (borrador de contrato)

### 4.1 Evolución de fondo (familia a pre-registrar)

Formas **candidatas** (elegir familia **antes** de ver ratios de masa; barrer parámetros de la familia, no el MS):

| Pieza | Forma ilustrativa del equipo | Notas anti-Shannon |
|-------|------------------------------|--------------------|
| Escala | a(t) = a₀ exp(H_inf t) en tramo inflacionario de t_g, u otra H(t) pre-registrada | H_inf **no** se ajusta para m_k1/m_k3 |
| Temperatura | T(t) = T₀ / a(t) (adiabasis del modelo) | ∇T o ε_T = S>0 como **CI**, no como perilla de éxito |
| Densidad | ρ(t) = ρ₀ / a(t)³ (o ley de la EOS) | Ligada a a(t) |
| EOS | P = w ρ, w=1/3 (radiación) como **default declarado**; barrer w ∈ familia fija | w no se tunear a 1/1836 |
| Reloj | t ∈ t_g (banda genética pre-átomos); **prohibido** identificar t_g ≡ t_e | Documentar bordes de banda como hipótesis de etapa, no “segundos del MS” como verdad |

Los valores 10⁻³⁶–10⁻⁶ s, 10¹⁵–10¹² K, T_c=97 MeV son **anclas de relato / mapeo** del equipo. En el código:

- o bien son **solo reporte** (como en CS074),  
- o bien entran como **escalas de un mapeo fijo pre-registrado** que **ningún juez de éxito** optimiza.

### 4.2 Lecturas acopladas (no módulos sueltos)

Sobre el **mismo** estado (a,T,ρ,P) y las **mismas** semillas/dominios:

| Lectura | Idea (equipo) | Condición de ser lectura F, no renombre T |
|---------|----------------|-------------------------------------------|
| m_fis(t) | ~ ∫ρ_ℰ dV / P(t) [y posible factor a(t)] | Debe responder a apagar ȧ o ā P; no ser monótono de perim solo |
| K_eff(t) o λ_screen(t) | apantallamiento / acoplamiento ∝ estado térmico | Longitud o K **medidos** del correlador; no e,ε₀ pegados sin NULL |
| V_eff(r,t) | costo de separar dominios; σ(t)∝P·f(a) como **hipótesis** | V_eff se obtiene de trabajo/separación; σ no se impone para “salir lineal” |

### 4.3 Topología T = contorno / semilla

| Usar de T | No usar de T |
|-----------|----------------|
| k=3, perim8, U(1) forma, banda como **semillas o filtros de dominio** en ρ,T iniciales | Forzar m→1/1836, reabrir E1–E10 |
| Nulls y z como **controles de forma** | Declarar quarks físicos porque z_k3>3 |

### 4.4 Kill-switch holístico (obligatorio)

El sistema **falla el contrato F** (y se declara) si:

| Condición | Por qué viola holismo |
|-----------|------------------------|
| **a fijo** (sin expansión dinámica) | Solo plasma + topología, sin pata 2 |
| **T paramétrica muerta** (T solo función de paso sin acoplar a a) | Adiabasis de adorno |
| **t_g ≡ t_e** o sin banda genética | Confunde etapas del proceso |
| **m, K, V medidos en mundos con distinta tríada** | Todo←partes otra vez |
| **Éxito = 1/1836 o 938 MeV** | Destino Shannon |

**Kill-switch positivo (debe cumplirse para “F vivo”):**  
variar a(t) o la CI ∇T/ε **cambia** de forma no trivial m(t), K_eff(t), V_eff **a la vez** (acoplamiento real todo↔todo).

### 4.5 NULL de MF-0

| Brazo | Definición |
|-------|------------|
| REAL | Tríada acoplada + CI asimetría (∇T o ε_ℰ) + semillas/contorno T opcionales |
| NULL-A | Misma tríada, **asimetría barajada / ε=0** |
| NULL-B | Misma CI, **a(t)=const** (rompe pata expansión) |
| NULL-C | Misma evolución, **semillas T barajadas** (si se usan) |

Éxito de andamiaje: REAL discrimina de NULL-A/B en al menos un observable de la terna (m, K, V) con la misma corrida.

---

## 5. Implementación: holístico en la ciencia, por capas en el código

El equipo rechaza MF **aislados** como *programas científicos*.  
Eso **no** prohíbe etapas de ingeniería:

| Hito de código | Entrega | Criterio |
|----------------|---------|----------|
| H1 | a(t), T(a), ρ(a), P=wρ, reloj t_g | Tríada corre; logs de estado |
| H2 | CI ∇T/ε + NULL-A/B | Persistencia energética |
| H3 | Dominios (lectura T o coherencia ℰ) | Catálogo de dominios en el plasma en expansión |
| H4 | **Medición simultánea** m_fis, K_eff, V_eff | Un JSON de corrida con las tres curvas |
| H5 | Barrido ciego de {H_inf, T₀, w} en familia pre-registrada | Robustez; sin gate MS |

Si en H4 solo se implementa m_fis y se “deja K y V para después” **sin** dejar los ganchos de estado, se viola el contrato.  
Mínimo holístico en H4: **tres observables, una evolución**.

---

## 6. Criterio de éxito Fase 5 (reafirmado)

| No es éxito | Sí es éxito (indicios de dominio F) |
|-------------|-------------------------------------|
| 938 MeV, 1/1836 como pass | Jerarquía m_fis **estructurada** vs T O(1), con kill-switch de a o P |
| σr impuesto | V_eff(r) **no** plano para r>2, monótono en familia pre-registrada |
| K pegado a Debye de libro | K_eff o ξ_screen colapsa curvas N o deja de ser crossover puro |
| Paper solo topología vendido como QCD | Contraste explícito: dominio T vs misma pregunta en MF-0 |

---

## 7. Claims actualizados

| ID | Claim | Estatus |
|----|-------|---------|
| W-50 | Fase 5 = MF-1 luego MF-2 luego MF-3 como mundos separados | **❌** (reduccionista) |
| W-51 | Fase 5 = MF-0 tríada plasma+expansión+t_g desde t=0 | **✅ canónico** |
| W-52 | Option D: acta T + preregistro MF-0 + sistema único + contraste | **✅** |
| W-53 | t_g ≠ t_e (tiempo genético ≠ tiempo emergente post-átomos) | **✅ instrucción teórica** |
| W-54 | Constantes/mapeos 97 MeV, 10⁻³⁶ s son gates de éxito | **⛔** — solo relato/mapeo fijo o reporte |
| W-55 | Topología T es semilla/contorno en F, no resultado a forzar | **✅** |

---

## 8. Mensaje equipo (copiable)

> Precisión Alexis **canónica**: Fase 5 no es “añadir plasma a la topología”. Es un **proceso holístico por etapas** con tríada basal **Plasma + Expansión supralumínica + Tiempo genético** acoplados desde t=0.  
> Orden lineal MF-1→2→3 o Debye-primero = **todo←partes** — vetado.  
> **Option D:** (1) acta dominio T, (2) pre-registro MF-0, (3) un sistema a(t),T(t),ρ(t),t_g midiendo m, K_eff, V_eff **simultáneos**, (4) contraste T vs F.  
> Kill-switch: a fijo o T muerta o t_g=t_e → no es dominio F.  
> S>0 / ∇T sigue siendo la asimetría basal — ahora **dentro** de la tríada, no al lado.

---

## 9. Próximo entregable

**`14_PREREGISTRO_MF0.md`** — valores de familia (no del MS), lista exacta de campos del estado, tres observables con fórmulas de medición en código, NULL-A/B/C, desenlaces F0-A/B/C, y plantilla de JSON de corrida holística.

Hasta ese pre-registro, **no** se implementa Debye ni σr sueltos.
