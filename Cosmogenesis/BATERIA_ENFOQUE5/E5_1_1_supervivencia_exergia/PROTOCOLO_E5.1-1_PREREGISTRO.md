# PROTOCOLO E5.1-1 — Supervivencia de exergía frente a la razón expansión/difusión, rango extremo

**Congelado (pre-registro):** 2026-07-24 16:35 (America/Santiago, UTC-4)
**Ejecutor:** CC (agente E5.1-1, batería Enfoque 5, corrida en paralelo con 29 agentes más)
**Base de código leída (NO editada):** `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs074_rcruz.py`
**Documento madre:** `BATERIA_ENFOQUE5_energia_exergia_entropia_30exp_PARA_CC.md`, sección "E5.1-1"

Este documento se escribe y congela ANTES de tocar el motor. Cualquier desviación
respecto de lo aquí escrito se reporta como desviación explícita, no se edita
retroactivamente (T3).

---

## 1. Pregunta

¿Sobrevive la capacidad de hacer trabajo (exergía) cuando el sistema se expande, y a
partir de qué razón r = H/D (expansión/difusión) aparece esa supervivencia?

## 2. Modelo (heredado de cs074_rcruz.py, motor propio bajo mi prefijo)

Campo escalar φ en un anillo de N=200 sitios (misma física que CS074-rcruz):
- Fondo φ=1 + perturbación ε·(suma de 5 armónicos con fase aleatoria, normalizada a
  desviación estándar 1).
- **Difusión:** relajación local hacia el promedio de vecinos, SOLO por aristas vivas
  (idéntica fórmula a `paso_difusion` de cs074_rcruz.py: nuevo = φ + 0.5·(media_vecinos−φ)).
- **Expansión:** cada arista viva se corta con probabilidad de Bernoulli H por paso
  (idéntica a `paso_expansion`); H≥1 corta todas; H=0 no corta ninguna.
- **D** = fracción de contraste (desviación estándar) borrada en UN paso de difusión pura
  (H=0), MEDIDA del propio campo (no puesta a mano), igual que `medir_D` en la base.
- **r** = H/D es la razón expansión/difusión, el eje primario pre-registrado del barrido.
  H se fija como H = min(r_target·D, 1.0) — D se mide primero, H emerge de esa medida.
- **Ruido dinámico (T7):** además de la semilla, en CADA paso de evolución se añade al
  campo ruido gaussiano de amplitud NOISE_REL·ε (NOISE_REL = 0.02, constante congelada
  aquí, jamás ajustada a posteriori). Con ε=0 el ruido dinámico es exactamente 0 (preserva
  el control ε=0 puro). Esto NO es un barrido cosmético de semilla: perturba el campo en
  cada paso de la evolución, no solo la condición inicial.

## 3. Axiomas declarados (E1/E2, NO física real)

- **E1 (conservación declarada):** el presupuesto de energía declarada del sistema,
  E_decl = Σφ, se declara conservado por el mecanismo de difusión (promedio local lineal
  = operador que preserva la suma en un grafo regular). Se AUDITA (no se fuerza): se mide
  E_decl al inicio y al final de cada corrida y se reporta la deriva relativa. No se
  renormaliza el campo — si la deriva es grande, es un hallazgo, no se oculta.
- **E2 (redistribución por expansión):** la expansión (cortar aristas) no crea energía;
  solo aísla regiones y con ello congela gradientes que de otro modo la difusión borraría.
  Esto redistribuye la capacidad de trabajo (exergía) sin crear energía nueva. No se
  verifica un mecanismo de enfriamiento aquí (eso es Tema 4); se declara como el marco
  interpretativo de por qué H>0 debería preservar X.

## 4. Observable — Exergía X

**X_final** = fracción de la energía (estructura) capaz de hacer trabajo, medida como
desviación del equilibrio uniforme, IDÉNTICA en fórmula a `persistencia()` de la base:

    c = corr(φ, roll(φ,1))   (autocorrelación a un paso; clip a ≥0)
    v = Var(φ_final) / Var(φ_inicial)     (fracción de varianza retenida)
    X_final = c · v

Justificación de por qué esta fórmula (y no solo la varianza) es el observable correcto:
la varianza sola SOBREVIVE a una permutación (mismo conjunto de valores, otro orden), así
que por sí sola no distinguiría REAL de NULL. El factor de autocorrelación c mide
específicamente la estructura ESPACIAL (el gradiente explotable, lo que realmente permite
extraer trabajo de vecinos correlacionados) y SÍ se destruye por permutación. X_final=0
cuando Var(φ_inicial)=0 (caso ε=0, sin diferencia que evolucionar).

**Juez ≠ observable (T2):** el veredicto se basa en el contraste REAL vs NULL vs
r-dependencia de la curva completa, no en un único número.

## 5. Barrido (sobredimensionado, regla del director)

| Eje | Rango | Puntos |
|---|---|---|
| r = H/D | {0} ∪ logspace(1e-3, 1e3) | 26 (r=0 explícito como control "sin expansión" + 25 puntos log en 6 décadas 1e-3…1e3) |
| ε | {0, 1e-12, 1e-9, 1e-6, 1e-3, 1e-2, 0.1, 0.3, 1.0} | 9 (0 a 1, 12 décadas + control 0) |
| semillas | 0..15 | 16 |
| ruido dinámico | NOISE_REL=0.02·ε, aplicado cada paso | fijo, declarado (no es eje de esta pieza — es E5.1-4) |
| N | 200 (fijo, igual que modo "produccion" de la base) | — |
| pasos | calibrado UNA vez (lavado a P<0.05 en ε=1e-3, H=0, mediana×1.15 de margen), reusado en toda la grilla — igual método que `medir_pasos_lavado`+`pasos_fijo` de la base. Válido porque la difusión es lineal: el tiempo de lavado relativo no depende de la amplitud ε. | — |

Total combinaciones (r,ε) = 26×9 = 234. Cada combinación: 16 semillas × {REAL, NULL} = 32
corridas → **7488 corridas de evolución** + calibración de lavado.

## 6. NULL

Permutar φ al final de la evolución (idéntico a `evolucionar(..., null=True)` de la base:
`phi = rng.permutation(phi)`), calculado con la MISMA semilla y MISMA H/ε que su pareja
REAL, difieren solo en el barajado final. Reportado por cada celda (r,ε): X_real vs X_null
y z-score (Δ/σ combinada).

## 7. PASS / criterios de lectura (congelados antes de correr)

- **ε=0 → X_final=0** a todo r (no hay estructura inicial que sobreviva).
- **r=0 (sin expansión), ε>0 → X_final→0** (la difusión lava todo; control de validez del
  barrido, análogo a `control_r0_ok` de la base, P_max=0.15).
- **r≪1 → X_final bajo** (se reabsorbe antes de que la expansión aísle nada).
- **r≈1 → zona de transición** (si el mecanismo es real, aquí debería verse el cambio de
  régimen).
- **r≫1 → X_final alto, separado del NULL** (aislamiento congela estructura).
- **NULL debe caer cerca de 0 en todo el rango** (T4: el NULL debe morder — si el NULL
  también sube con r, el hallazgo es artefacto del barajado o de la métrica, no de la
  física).
- Si CUALQUIERA de estos falla, se reporta como tal — no se reinterpreta ni se ajusta el
  motor después de ver los datos (T3, regla de ejecución #1).

## 8. Verificación cruzada (regla de ejecución #4)

1. NULL propio (permutación), por celda.
2. Segundo observable/método: `std_ratio` = φ.std()/φ_inicial.std() (varianza retenida
   sola, sin el factor de autocorrelación) reportado en paralelo a X_final — permite ver
   si el hallazgo depende del factor c o ya está en la varianza cruda.
3. Auditoría de conservación E1 (deriva de Σφ inicio→fin) reportada en cada fila para
   revisión externa en disco (JSON crudo).

## 9. Salidas

- `E5_1_1_engine.py` — motor (este archivo, escrito DESPUÉS de este pre-registro).
- `E5_1_1_resultado_crudo.json` — filas completas del barrido (todas las columnas: r,
  eps, H, D, pasos, X_real media/std por semilla, X_null media/std, z, std_ratio_real,
  std_ratio_null, deriva_E_decl, frac_exp, T_fin_K si aplica).
- `E5_1_1_dispersión_semillas.json` (o incluido en el crudo) — todas las X por semilla
  individual, no solo la media, para reportar dispersión real (regla de ejecución #9).

## 10. Trampas explícitamente evitadas

- T0: nada discreto puesto a mano — N y pasos vienen del modelo base y de calibración
  medida, no de ajustar-para-que-cruce.
- T1: NOISE_REL=0.02 y P_LAVADO=0.05/MARGEN=1.15 son constantes de diseño declaradas
  ANTES de correr, no ajustadas para acercar el resultado a nada esperado.
- T2: X_final (observable) es una fórmula fija; el veredicto lo da la curva completa
  contra NULL, no el observable mismo.
- T5: se reporta la curva X_final(r) entera para cada ε, no un gate binario.
- T6: se audita conservación E1 cada corrida (inicio/fin).
- T7: ruido dinámico presente en cada paso, además de 16 semillas.

No se corre nada del motor hasta que este archivo esté guardado en disco.
