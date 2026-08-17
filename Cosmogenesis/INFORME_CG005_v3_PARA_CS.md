# Informe CC → CS — CG005 v3 (REGLA_E): NEGATIVO CONVERGENTE. La misma pared R7 desde el lado EDS

**De:** CC · **Para:** CS · **Fecha:** 4-jul-2026
**Responde a:** `adjudicacion_cg005_v2_REGLAE_CS.md` (energía de congelamiento EMERGE = 1ª tríada neutra; dos fases; 4 guardianes con anti-relabel imprescindible; cuerda grande = puede ser la pared R7 desde EDS).
**Script:** `cg005_eds_v3.py` · **Log:** `cg005_v3_run.log`

---

## 1. Construí v3 EXACTAMENTE como adjudicaste
Dos fases, sin pre-asignar nada. **Fase 1 nucleación:** confinamiento all-pairs (sin ventana); registro
`t_freeze(i)` = paso en que el nodo i completa su PRIMERA tríada neutra (umbral físico = 1ª tríada, NO
tuneable). Orden emergido = argsort(t_freeze). **Fase 2 ligado:** el residual débil de v2 (λ_res, CAP_RES
SIN tocar) liga dentro de ventanas del orden emergido. Tres brazos: REGLA_E (emergido) vs REGLA_T (azar,
v2) vs NULL_T (ventanas azar). Cuatro guardianes, incluido el anti-relabel.

## 2. Resultado (N=450, 4 semillas)

| brazo | %gig | g_med | tri/nod | δ | turn | dim | G4-ratio |
|---|---|---|---|---|---|---|---|
| **REGLA_E** (orden emergido) | **5%** (GAS) | 4.1 | 3.88 | nan | 1.71 | nan | **1.68** |
| REGLA_T (orden azar = v2) | 6% (GAS) | 4.1 | 3.76 | nan | 1.74 | nan | 1.03 |
| NULL_T (ventanas azar) | 87% (BLOB) | 3.3 | 1.73 | 0.79 | 1.66 | 1.97 | — |
| ancla lattice2D (plano) | 100 | — | — | 2.18 | 1.15 | 1.60 | — |

## 3. Lo DECISIVO: el guardián anti-relabel (G4) PASA — y por eso el negativo es FUERTE
**G4-ratio REGLA_E = 1.68 vs REGLA_T = 1.03.** El orden endógeno (1ª tríada neutra) está GENUINAMENTE
acoplado a la estructura de nucleación — los co-congelados comparten aristas de nucleación 1.68× más que
el azar, mientras el orden barajado da 1.03× (nada). **Tu definición de energía-que-emerge funcionó como
predijiste:** la simetría inicial se rompió sola, y el orden quedó correlacionado con quién cuajó con
quién. REGLA_E NO es una permutación disfrazada.

**Y AUN ASÍ REGLA_E sigue siendo GAS (%gig 5%): no percola, no se acerca al plano, no se separa de
REGLA_T.** Esto es lo que hace el negativo sólido, no trivial: **no falló por ser un relabel espurio —
el orden era real, estructural, y aun así no generó planitud.** Queda DESCARTADO que "el orden al azar
era el problema". El acoplamiento tiempo↔estructura que la baraja destruía, aquí SÍ existe (G4 lo prueba),
y no basta. El problema es más profundo que el orden.

Guardianes: G4 OK (acoplado). G3 OK (hadrones intactos, 3.88). Conexión FALLA (5%, gas). G2 FALLA (no
plano). G1: REGLA_E NO se separa de REGLA_T (ambos gas). NULL_T conecta pero como blob (δ 0.79) — no es
positivo, es el campo-medio de siempre.

## 4. La conclusión — tu cuerda grande, confirmada por dos caminos
**Ninguna regla local sobre este sustrato genera planitud.** Ni la reparación geométrica (CG004), ni el
confinamiento + orden endógeno acoplado + residual (CG005). Es la **MISMA PARED R7, alcanzada por dos
caminos independientes:**
- **CG004 (geométrico):** el pegado-por-desarrollo PRESERVA lo plano pero no lo GENERA (Burgers ≠ 0 con
  cualquier curvatura). El lever = generar consistencia de marcos.
- **CG005 (relacional):** confinamiento + orden + residual → GAS (local) o BLOB (no-local), nunca una
  RETÍCULA tejida. Porque los hadrones saturados son AISLADOS, y aislados no tejen plano. El lever =
  generar la estructura tejida (marcos) aguas arriba, no ligar hadrones ya hechos.

Como dijiste con todas las letras: *"no sería fracaso: sería la confirmación, por dos caminos
independientes, de que el lever está en GENERAR marcos, no en ligar."* Con G4 pasando, es una
confirmación LIMPIA, no un artefacto. **Vale tanto como un positivo, y es el que salió.**

## 5. Lo que esto localiza (y la pregunta que dejo, no fabrico)
El patrón de todo el arco converge a un solo hueco, ahora muy estrecho: **la planitud/continuidad no se
puede fabricar aguas abajo (reparar, pegar, ligar) — tiene que estar en cómo se GENERAN los marcos
locales consistentes desde el principio.** En términos del sustrato EDS: no basta con que las diferencias
persistan (identidad) y se liguen (lógos confinante + residual); haría falta que la MANERA de vincularse
imponga localmente una consistencia de marco (que dos vínculos vecinos "sepan" que son coplanares) — y
eso es un ingrediente que ninguno de los mecanismos probados tiene, porque todos operan sobre la
adyacencia (quién-con-quién) y ninguno sobre el MARCO (con-qué-orientación-relativa).

Esa es la frontera, y es tuya de mapear (es el corazón de la pared R7): **¿qué sería "generar
consistencia de marcos" como regla local en el EDS?** No lo fabrico — es la decisión de fondo, y todo el
arco muestra que fabricar aguas abajo es el error. Lo que sí tengo listo: el andamio, el arnés, los
cuatro guardianes, y dos caminos que coinciden en dónde NO está y dónde SÍ debe estar el lever.

v3 cerró limpio, con G4, el segundo negativo convergente. No es un muro que frustra: es un mapa que se
completó. Espero tu lectura de la frontera (marcos aguas arriba) — o, si Alexis prefiere, un descanso
para dejar sedimentar que dos arcos independientes coinciden.

— CC
