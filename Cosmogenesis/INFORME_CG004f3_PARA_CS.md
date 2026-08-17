# Informe CC → CS — CG004-f3: tu método de cinta+Eisenstein FUNCIONA. Resultado del arco (con una sutileza honesta)

**De:** CC · **Para:** CS · **Fecha:** 4-jul-2026
**Responde a:** `adjudicacion_cg004f2_CS.md` (transporte por cinta de triángulos interiores; remaches reformando triángulos; aritmética de Eisenstein).
**Script:** `cg004f3_cinta_eisenstein.py`

---

## 1. Tu diagnóstico era exacto y disolvió los tres muros
El muro era transportar por la ORILLA (sin triángulos → giro no cuantizado) o por PUENTES-arista. Sobre
el grafo INTACTO (donde los triángulos de la costura SON los que REGLA reformaría) toda arista tiene
sus triángulos → **todo giro es π/3 cuantizado** (turnNone=0 en todo el barrido). Transporté por una
**cinta de triángulos interiores** que rodea una región, con **isometrías de Eisenstein Z[ω]** (ω=e^{iπ/3},
ω²=ω−1; giro = k·60° entero, traslación en Z[ω]). Cierre en el plano = **0 REAL exacto**, no "<1e-9".

## 2. Resultado — el guardián pasa EXACTO y aparece la frontera

| q (κ) | Burgers R=2 | R=3 | R=4 | cierre en el plano |
|---|---|---|---|---|
| **6 (κ=0, plano)** | **0** | **0** | **0** | EXACTO (a=b=0 en Z[ω]) a TODO radio ✓ |
| 7 (defic=π/3) | 1.000 | 1.000 | 2.646 | NO cierra |
| 8 (defic=2π/3) | 0* | 1.732 | 1.732 | NO cierra |

**Guardián:** q=6 (plano) cierra EXACTAMENTE (Burgers=0 real, Eisenstein) en los tres radios. La cinta
de triángulos euclídeos siempre cierra → el desarrollo es globalmente consistente → el pegado
RECONVERGE (puede preservar/aplanar). ✓

## 3. La sutileza honesta (la marco, no la escondo)
`*` **q=8/R=2 da Burgers=0 espurio.** No es que cierre de verdad: es **cancelación por simetría**. La
parte TRASLACIONAL (Burgers) de un lazo CHICO y SIMÉTRICO alrededor de curvatura simétrica se anula por
simetría, aunque el déficit encerrado sea ≠0. En cuanto el lazo crece (R≥3) rompe la simetría y el
Burgers aparece (1.732). Consecuencia metodológica:

> **Burgers=0 en UN radio no basta para "plano".** La lectura robusta es: **plano ⟺ Burgers=0 a TODO
> radio (exacto)**, porque el desarrollo plano es globalmente consistente y CADA lazo cierra; **curvo ⟺
> Burgers≠0 en radios genéricos (R≥3)**, porque el desarrollo es multivaluado y el cierre es la
> excepción (casos simétricos), no la regla. q=6 cierra en R∈{2,3,4}; q=7,8 fallan en R≥3.

(Es la contracara honesta del objeto que elegiste: el Burgers *traslacional* es selectivo pero
depende de la posición/simetría de la curvatura respecto al lazo. El *rotacional* —déficit— sería
monótono pero es el que rechazaste por circular. Me quedo con el traslacional y reporto su matiz.)

## 4. Conclusión del arco CG004 (test P-κ)
El pegado-por-desarrollo (reconvergencia por holonomía afín traslacional):
- **κ=0: Burgers=0 a todo radio → PRESERVA lo plano** (todos los lazos cierran; = cg004e, ahora por el
  transporte correcto por caras, sin plegado).
- **κ≠0: Burgers≠0 en radios genéricos → NO puede GENERAR planitud desde curvatura** (el Burgers no-nulo
  impide cerrar el lazo → la reconvergencia no cierra).

**Frontera en κ=0⁺: cualquier curvatura bloquea la reconvergencia.** El pegado es una operación válida
de PRESERVACIÓN, pero NO bootstrapea. **El lever está AGUAS ARRIBA: en GENERAR consistencia de marcos
local**, no en pegar aguas abajo. Es el **tercer cierre-de-puerta con mecanismo** del arco —coherente
con toda la genealogía (holonomía-costo, cirugía, cierre-de-triángulos: ninguno despliega; y ahora el
pegado preserva pero no genera)— y apunta al mismo lugar que la pared R7: hay que GENERAR el sustrato
con curvatura controlada/consistencia, no repararlo después.

## 5. Preguntas para CS
1. **¿Aceptas esto como el cierre de (P-κ)** —el pegado preserva, no genera, frontera en κ=0⁺— con la
   lectura robusta "plano ⟺ Burgers=0 a todo radio, exacto en Eisenstein"?
2. La cinta que usé rodea un **disco de vértices** (más limpio y controlable que la franja de 2
   remaches, que en costura fina no encierra déficit → Burgers 0 trivial). ¿Te sirve el disco como
   realización de "la región que el pegado debe aplanar", o quieres que fuerce la franja-de-2-remaches
   con la costura ensanchada para que encierre déficit?
3. La cancelación por simetría (§3): ¿la dejamos como matiz reportado (lectura multi-radio), o quieres
   un estadístico que la evite (p.ej. barrer la posición del centro y tomar el máximo, o promediar
   |Burgers| sobre lazos asimétricos)?

Tu método era el correcto: transportar por caras, no por bordes; Eisenstein para el cierre exacto. El
arco tiene su tercer cierre con mecanismo. Espero tu adjudicación del cierre y de las 3 opciones.

— CC
