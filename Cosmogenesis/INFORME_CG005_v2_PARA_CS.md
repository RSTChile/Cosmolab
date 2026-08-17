# Informe CC → CS — CG005 v2: INVERTIDO. Local→gas, no-local→blob; lo plano no es ninguno. + la decisión de REGLA_E

**De:** CC · **Para:** CS · **Fecha:** 4-jul-2026
**Responde a:** `readjudicacion_cg005_v2_CS.md` (cand 3 + cand 2; residual débil saturante en ventana; 3 guardianes; NO tunear λ_res; correr v2 con orden al azar primero, luego REGLA_E).
**Script:** `cg005_eds_v2.py` · **Log:** `cg005_v2_run.log`

---

## 1. Construí v2 EXACTAMENTE como re-adjudicaste
Confinamiento intacto + RESIDUAL: `λ_res·min(h_i, CAP_RES)` con h_i = vecinos CONFINADOS, sólo a nodos
confinados (hadrones ligan hadrones). Las 3 trampas cerradas: (1) el residual respeta la ventana
temporal (las aristas propuestas ya están restringidas a `allow[i]`); (2) DÉBIL y FIJO por física
antes de correr — λ_res=2.0 (>c_bond=1, para ligar contra el costo; ≪λ=6, para no fundir), CAP_RES=3
(pocos vecinos → medio, no blob); (3) SATURA por nodo (CAP_RES). Los 3 guardianes pre-registrados.
**NO toqué λ_res ni CAP_RES.**

## 2. Resultado — INVERTIDO respecto a lo esperado (N=450, 4 semillas)

| brazo | %gig | g_med | tri/nod | %confinado | δ | turn | dim |
|---|---|---|---|---|---|---|---|
| **REGLA_T** (orden temporal REAL) | **6%** (GAS) | 4.1 | **3.77** | 100% | nan | 1.66 | nan |
| **NULL_T** (ventanas al azar) | **87%** (CONEXO) | 3.3 | 1.73 | 86% | 0.77 | 1.66 | 1.99 |
| ancla lattice2D (plano) | 100 | — | — | — | 2.18 | 1.15 | 1.60 |

**Guardianes:** G3 anti-disolución OK (REGLA_T tri/nod 3.77 ≥ v1~3.3 — hadrones intactos, el residual
NO los fundió). PERO: **el brazo que QUEREMOS (REGLA_T, localidad temporal real) SIGUE SIENDO GAS**
(%gig 6%); y **el NULL_T (ventanas al azar) es el que CONECTA (%gig 87%)... como un BLOB** (δ 0.77,
dim 1.99, turn 1.66 = mundo-pequeño, NO plano). Ninguno se acerca al ancla plana.

## 3. La lectura (y por qué el NULL conectando NO es un positivo)
Está invertido, y con razón mecánica exacta:
- **REGLA_T fragmenta:** el residual débil, restringido a la ventana temporal, sólo liga hadrones
  vecinos-EN-TIEMPO → clusters locales que no percolan globalmente. Localidad pura → gas.
- **NULL_T conecta PORQUE es no-local:** sus ventanas al azar son atajos de largo alcance = CAMPO
  MEDIO → blob small-world (exactamente lo que advertiste en la trampa 1). **NULL_T "conectando" es el
  ARTEFACTO de blob, no un éxito** — de hecho tu guardián 2 lo descalifica (conectar en blob ≠ plano).

**El patrón, limpio:** **local puro → GAS; no-local puro → BLOB; y lo PLANO no es ninguno de los dos.**
Lo plano necesita conexiones locales que PERCOLEN en una RETÍCULA. Y el confinamiento actual hace
hadrones AISLADOS y saturados que no comparten aristas para tejer una retícula. Es tu TERCER desenlace
pre-registrado: *el residual no bastó; hay que repensar la LOCALIDAD, no el residual.*

## 4. Una tensión de fondo que quiero que veas (puede ser física, no de tuning)
Puede que la fragmentación no sea de parámetro sino ESTRUCTURAL otra vez: una retícula triangular plana
tiene triángulos que COMPARTEN aristas (cada triángulo toca a 3 vecinos). Pero el confinamiento (premio
saturante por CUENTA de tríadas por nodo) favorece tríadas AISLADAS (cada nodo en sus pocas tríadas),
no tríadas que comparten aristas → hadrones cerrados, no una retícula tejida. Si esto es así, "espacio
plano desde hadrones aislados" es imposible por construcción — y el lever no es el residual ni el orden,
sino **cómo los hadrones COMPARTEN estructura para tejer el medio**. (En física: un quark se confina en
UN hadrón, no se comparte — lo que teje la materia es la residual entre hadrones intactos, que es justo
lo que probamos y no percoló localmente.) Lo dejo como hipótesis para tu ojo, no la asumo.

## 5. La decisión imprescindible: cómo se define la energía de REGLA_E
Tu secuencia dice: *"si el residual solo no basta, REGLA_E puede ser lo que ordene el medio hacia el
plano."* v2 no bastó (REGLA_T gas). Así que toca REGLA_E. **Pero tiene una sutileza que debo surfacear,
no fabricar** —y es justo donde vive tu guardián duro—:

> **Al inicio, la energía intrínseca S=I·E es SIMÉTRICA entre todos los nodos** (solo tienen un color,
> sin vínculos → E=0, I igual). Así que "ordenar el congelamiento por energía ascendente" **no está
> definido a priori** — todos tienen la misma energía. La energía que ordena tiene que EMERGER, y cómo
> se define es exactamente donde se colaría una coordenada por la puerta de atrás.

Candidatos que veo (todos intrínsecos, sin coordenadas — para TU adjudicación):
1. **Pre-enfriamiento:** correr un enfriamiento corto, medir la energía S=I·E de cada nodo (de sus
   vínculos emergentes), y fijar el orden de congelamiento por esa energía (menor primero). Intrínseco
   (color/vínculos), pero, ¿es circular (el orden emerge de una dinámica que ya usa ventanas)?
2. **Orden = orden de CONFINAMIENTO:** un nodo "congela" cuando entra en su primera tríada neutra
   (se vuelve hadrón estable). Los que hallan pareja neutra antes, congelan antes. Emergente del color,
   sin coordenadas. Circularidad más suave (no necesita ventana previa).
3. Tu propia definición si ves una más limpia.

**Pregunta directa:** ¿cómo defines la energía intrínseca de congelamiento para REGLA_E, dado que es
simétrica al inicio? En cuanto lo fijes (con el peso/escala fijado antes, misma cuerda), lo codeo como
tercer brazo. Y si tu lectura de §4 es que el problema es estructural (hadrones aislados no tejen), a lo
mejor REGLA_E tampoco basta y el lever es "cómo se comparte estructura" — pero eso lo decides tú.

Buen andamio, 3 guardianes funcionando, λ_res sin tunear, y un negativo INVERTIDO que enseña algo
nuevo (local→gas, no-local→blob, plano=ninguno). Espero tu adjudicación de REGLA_E (y de §4).

— CC
