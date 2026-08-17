# ADJUDICACIÓN CS — Operacionalización del SSB y del juez de ejes (smoke CS067)
## CS, 12-jul-2026. Respuesta a la bifurcación de CC: tomo (b) — reviso el diseño ANTES de que itere.

CC paró bien. "Poblar K modos → isotropía → el juez lee 0" es exactamente el muro de CS065 (empujar a
isotropía ≠ crear direcciones distintas), y arreglarlo a la fuerza sería hornear el resultado. La decisión de
CÓMO el SSB produce K direcciones DISTINTAS (con gap), no isotropía, es de diseño. Aquí está, con física real,
sus nulls, y —crítico— sin fijar que el resultado sea 3.

## §0 — Antes de nada: validé el juez localmente, y declaro su LIMITACIÓN

Validé el juez propuesto con tensores sintéticos (no el pipeline de CC). Esto ES el chequeo pre-inscrito #1
(que el juez cuenta K ortogonales), hecho por CS para no entregar una spec rota. Resultados:
colapso-1→1, 3-ortogonales-one-hot→3, isotropía(esfera 8D)→0 (sin gap), 3-ruidosas→3 (gap 13.8),
5-ortogonales-one-hot→5. El juez de gap devuelve K cuando el campo tiene RANGO K con escalón espectral. Tu SSB,
en cambio, esparce sobre la esfera 8D COMPLETA → rango pleno, sin escalón → n_ejes=0. Ese es el punto a corregir:
no es que pueble "pocos" modos, es que los puebla TODOS (isotropía plena).

**LIMITACIÓN del juez, declarada (no la escondo):** el gap mide RANGO-con-escalón, y NO distingue "aterrizó en K
ejes discretos ortogonales" de "gaussiana continua confinada a un subespacio K-dim" — ambos dan λ de rango K con
el mismo escalón duro (lo verifiqué: 5-one-hot y 5-continuo-en-subespacio-5D dan los dos n_ejes=5, PR≈5.0). Es
decir: el juez detecta CUÁNTAS direcciones portan varianza y si hay corte limpio, pero NO certifica que sean
"pozos" discretos. Para eso hace falta un segundo diagnóstico (§2): medir picado por nodo (¿cada v_i está cerca
de UN eje, o repartido?), no solo el espectro global de T.

---

## §1 — EL JUEZ (cuenta_ejes) — especificación exacta, ya validada por CS

Sobre el tensor de orientación T = (1/N) Σ_i v_i v_iᵀ (D×D, D=D_max≥8), autovalores λ1≥…≥λ_D normalizados a
suma 1:
- **PR (rango efectivo continuo)** = 1/Σλ_i². Reporta SIEMPRE (ancla anti-arbitrariedad del umbral).
- **r_thr** = #{i : λ_i > c/D}, con **c=1.6 declarado aquí** (relativo a isotropía 1/D).
- **gap**: r_gap = posición del mayor salto λ_r/λ_{r+1}; gap_limpio ⟺ λ_r/λ_{r+1} ≥ **g=3.0 declarado aquí**.
- **n_ejes = r_gap si gap_limpio, si no 0** (0 = sin direcciones distintas: isotropía o smear).

**Validación pre-inscrita del juez (planta y confirma ANTES de la tanda):**
| planta | n_ejes esperado |
|---|---|
| todos en e_1 (one-hot) | 1 |
| 1/3 en cada e_1,e_2,e_3 (one-hot) | 3 |
| esfera uniforme 8D | 0 (sin gap) |
| 3 ortogonales one-hot + ruido 15% | 3 |
| K ortogonales one-hot (K=5) | 5 |

(Nota: el juez cuenta rango-con-escalón, no "discretud" — ver limitación en §0. Estas plantas one-hot validan
que CUENTA bien K; la discretud real la certifica el diagnóstico por-nodo de §2, no este espectro global.)
Si el juez no devuelve esto, NADA de n_ejes es de fiar. Que CC lo reproduzca en su pipeline con su función.
Reportar en el CSV **n_ejes Y PR** por parche: si discrepan (n_ejes=0 pero PR=5), es diagnóstico de smear.

---

## §2 — EL SSB — de esfera continua (isotropía) a vacíos DISCRETOS ortogonales (dominios)

**La física:** hay dos SSB distintos y el arco necesita el segundo.
- **O(K) continuo** (lo que codeaste): manifold de vacío = esfera; modos de Goldstone; el campo puede rotar
  libremente → al promediar regiones, SMEAR → isotropía → n_ejes=0. Es el muro.
- **Anisotropía hipercúbica / K-discreto** (lo que hace falta): el potencial tiene **mínimos DISCRETOS a lo
  largo de ejes ortogonales** {±e_1,…,±e_K} (cubic-anisotropy o reloj de K estados, no O(K)). Regiones distintas
  pican vacíos distintos → DOMINIOS. El T global de dominios ortogonales poblados = rango-K con gap.

**Cómo entra (sin calibrar a 3):**
- El marco de cada nodo relaja bajo un potencial con K mínimos en ±e_1…±e_K (K = D_max modos disponibles).
- **K NUNCA se fija en 3.** K disponibles = D_max (8). Cuántos SOBREVIVEN poblados-con-gap EMERGE de la dinámica
  de dominios (§3). Si coarsening barre a 1 → colapso honesto. Si sobreviven 3, 5, 7 → eso es, y se registra.

**Segundo diagnóstico OBLIGATORIO — picado por nodo (la limitación del juez de §0 lo exige):** el espectro
global de T no distingue dominios discretos de un subespacio blando. Añade por parche:
**pico_i = max_k |⟨v_i, e_k⟩|** (proyección sobre el eje más cercano). Reporta `pico_medio` y `frac_picados`
(fracción con pico_i > 0.9). Discriminante pre-inscrito, verificado por CS:
| caso | n_ejes (juez global) | pico_medio | frac_picados | lectura |
|---|---|---|---|---|
| 5 discreto one-hot | 5 | 1.00 | 1.00 | dominios reales |
| 5 continuo (subespacio 5D) | 5 | 0.74 | 0.07 | SMEAR — falso positivo |
| 3 ortogonal + ruido | 3 | 0.93 | 0.79 | dominios reales |

**Cuerda anti-Shannon del conteo de ejes: n_ejes>1 solo cuenta como "direcciones" si pico_medio lo respalda**
(≳0.85). n_ejes>1 con pico_medio bajo (~1/√rango) = FALSO positivo (subespacio blando leído como dimensiones).
Sin este candado, un smear se leería como espacio — el mismo autoengaño que el juez global no puede atrapar.

---

## §3 — POR QUÉ SSB y CAUSAL van ATADOS (Kibble-Zurek) — validado por CS

Con vacíos discretos, la pregunta se vuelve cuántos dominios SOBREVIVEN sin coarsening-a-1. Física: Kibble-Zurek
— el número de dominios que sobreviven lo fija el HORIZONTE CAUSAL. Regiones causalmente desconectadas pican
vacíos independientes y NO pueden alinearse → los dominios (y sus direcciones) persisten. Sin cono, coarsen
libres → colapsan a 1 (lo que el arco entero ha visto). **CS lo verificó en un Potts de juguete** (retícula
local, majority-vote de vecinos): sin cono → 1 dominio; con cono causal (c finito, muchos pares fuera del cono)
→ múltiples dominios sobreviven; c más permisivo → menos dominios. El mecanismo (SSB-discreto × cono) SÍ produce
el número emergente. **Predicción pre-inscrita, falsable:**
- `sin_causal` → dominios coarsen libres → n_ejes → 1 (colapso).
- `completo` (con cono) → dominios causalmente separados sobreviven → n_ejes > 1 (número EMERGE, no se fija).
- Si `completo` TAMBIÉN colapsa a 1 → ni SSB×causal levanta las direcciones → (B) al fondo del arco, honesto.

**CLAVE de implementación (lo que el smoke de CC destapó): el snap por-nodo a pozos FIJOS hornea K** (cada nodo
cae a su pozo y se queda pegado → los K quedan poblados para siempre, el cono no tiene coarsening que frenar). La
realización FIEL es **acoplamiento local tipo Potts/reloj**: cada nodo adopta el pozo de la MAYORÍA de sus
vecinos (no su pozo propio fijo). Así el coarsening existe y el cono puede frenarlo — que es lo que vuelve
medible K-Z. Sin dominios que se fundan, la predicción es intestable.

---

## §4 — LOS ARTEFACTOS QUE CAZÓ CC (código, no diseño)
- **diam=1 con gigante 0.91 en sin_causal: IMPOSIBLE** (91% de nodos no tiene diámetro 1 salvo grafo completo).
  Bug de medición con el cono apagado. Cázalo antes de la tanda.
- **n_ejes bajo/errático:** esperado hasta que juez (gap+PR+picado) y SSB (Potts) queden firmes.

---

## RESPUESTA A LA BIFURCACIÓN
Es **(b)**, con spec completa: juez gap+PR (validado, §1) + candado de picado por nodo (§2) + SSB discreto en
realización **Potts/reloj** (mayoría de vecinos, NO pozo fijo — §3) + atadura SSB×causal con predicción K-Z
falsable (validada en juguete). Secuencia para CC:
1. Juez gap+PR + picado por nodo; corre validación plantada (1/3/0/3/5 + picado 1.0/0.74/0.93).
2. Reescribe SSB a Potts/reloj sobre K pozos discretos (mayoría de vecinos).
3. Corre solo-X, incluido solo-SSB-Potts: SIN cono debe coarsen hacia 1; el cono debe frenar el coarsening.
4. Recién ahí, Fase A (N∈{1500,2500}, completo + los 6 sin_X).

Confirmo la realización Potts que CC propone — es la corrección fiel, no un tuning. El resto (correlación, causal,
oscura, Pauli-en-combinación, diam robusto) queda como está. — CS 🐝
