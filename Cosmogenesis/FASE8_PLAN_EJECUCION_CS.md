# Fase VIII — plan de ejecución

**12 de agosto de 2026** · Sobre las propuestas de 4 analistas del equipo, reordenadas por
información-por-corrida. Ejecución automática con agentes dedicados; el análisis se hace al final.

## La pregunta que domina Fase VIII

Ya no es "¿hay efecto?" — eso está respondido. Es:

> **¿Qué propiedad relacional exacta convierte la misma cantidad de relaciones y de triángulos en una
> condición física distinta? ¿Y actúa a través del pico local de densidad, o además de él?**

## Reglas vigentes en TODOS los experimentos de esta fase

- **Endpoints continuos. NUNCA "% Clase III"** (4 mediciones muestran que fabrica discontinuidades).
- **NUNCA el coeficiente de clustering como variable explicativa** — F7-03 mostró que **falla en el signo**
  cuando se fija el nº de triángulos. Usar medidas de apiñamiento del soporte.
- **Diámetro → `cs090_diam_corregido.py`**, nunca `_diam` de `cs055`.
- **Unir CSVs por `(rule_id, seed)`**, nunca por `rule_id` solo.
- **No mezclar layouts:** todo brazo comparado debe usar el mismo layout (N² o Barnes-Hut) y el **mismo θ**.
  El sesgo de θ=0.3 (+0.0025 a +0.0071) es mayor que los residuales que perseguimos.
- **Grano del instrumento: 1 partícula = 0.0005** de fracción de masa a N=2000. Todo efecto se reporta
  contra ese grano; si queda por debajo, se dice "falta resolución", no "nulo".
- **Guardar los grafos** de toda corrida nueva.
- Phantom → `./venv/bin/python`; grafos → `python3.9`. Verificación cruzada contra `meta_regla.json`.
- No tocar scripts congelados. No declarar cierre. No commits.

---

## OLA 1 (en ejecución)

| id | experimento | por qué primero |
|---|---|---|
| **F8-00** | **INFRA: guardar y medir los grafos** — regenerar los grafos de las 254 filas del dataset unificado y medir en TODAS las variables de apiñamiento (hoy el clustering está en 24/254). Más: dejar el guardado automático para corridas futuras | Casi gratis y desbloquea todo el análisis posterior |
| **F8-01** | **Desacoplar las 4 medidas de apiñamiento** (triángulos/arista, Gini, solapamiento, aristas-en-triángulo; hoy colineales, dos a ρ=0.981) | Sin esto no sabemos qué perilla girar |
| **F8-04** | **Medir el grano a N=8000** (réplicas con perturbación de redondeo 1e-16) | Decide si la alta resolución es viable ANTES de gastar en ella |
| **F8-05** | **F7-03 en el solver independiente** (`solap` vs `disj`, grados y triángulos fijos) | Valida que el +13.8% no es de Phantom. Barato y decisivo |

## OLA 2 (después)

| id | experimento |
|---|---|
| **F8-02** | Manipular el **pico local de densidad inicial** a propósito (hoy sólo medido post-hoc; r parcial +0.64 a +0.90) |
| **F8-03** | **Mismo pico local, distinta topología** — el control que puede **cerrar** el mecanismo: si Phantom no los distingue, la cadena es `topología → pico local → masa` y está completa |

## OLA 3 (sólo si F8-04 y F8-05 dan verde)

| id | experimento |
|---|---|
| **F8-06** | F7-03 a N=4000 y N=8000, mismo layout y θ en ambos brazos |

## Descartado, con motivo

- **"Helicidad discreta" sobre el layout** (propuesta del 3er analista): `layout_resortes` **no es un flujo
  físico** — sus "velocidades" son pasos de un algoritmo de optimización, no velocidades de un fluido.
  Calcular helicidad ahí sería una analogía disfrazada de medición. Si se explora la vía de fluidos, hay que
  hacerlo sobre el campo de velocidades **real de Phantom**.

## Rama paralela, no iniciada

**Regularidad Cosmogénesis** (R1-R3): ¿el apiñamiento máximo y el pico de densidad quedan acotados cuando
N crece? Requiere ≥4 resoluciones — hoy hay dos, y con dos puntos no se ajusta una ley de escala. Depende de
F8-04.
