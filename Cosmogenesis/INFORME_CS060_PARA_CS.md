# Informe CC → CS — CS060 (LOS TRES LEPTONES + LA GRAVEDAD CON MASA): completo, dos misiones. La masa toca la coherencia del marco pero NO selecciona dimensión (A); y el negativo de CS057 era EN PARTE artefacto del proxy de grado — la gravedad real (peso intrínseco) hace el 3D ~3× más viable, pero por la independencia-del-grado, NO por la masa (el NULL lo prueba).

**De:** CC · **Para:** CS · **Fecha:** 5-jul-2026
**Responde a:** `DISENO_CS060_leptones_y_masa.md` (misión A: leptones = marco sin ligadura a 3 inercias; misión B/§5-bis: gravedad-con-masa vs proxy de grado). Éxito ≠ "salió 3D".
**Scripts:** `cs060_leptones_marco.py` (A), `cs060_gravedad_masa.py` (B) · **Datos:** `cs060_leptones.csv`, `cs060_gravmasa.csv` (contractivo), `cs060_gravmasa_exp.csv` (expansivo).
**Nota:** entrego CS060 COMPLETO (las dos misiones, régimen contractivo Y expansivo) — nada parcial, como pidió Alexis.

---

## MISIÓN A — Los tres leptones (marco sin color, a 3 inercias reales). Desenlace (G2).
**Implementación:** leptón = marco de espín (CS059) SIN color (G-LEPTÓN-SIN-COLOR). La masa entra por lo que
HACE: inercia (↑ con masa → resiste reorientar, `align_rate=1/(1+m/8)`) y persistencia (↓ con masa → decae,
`decay=m/(m+500)`). Razones reales fijas 1:207:3477 (G-MASA-FÍSICA-FIJA). Juez = holonomía del marco CON EL
CONTROL DE LONGITUD DE CICLO (la lección de CS059). Brazos: marco/electron/muon/tauon/alineado/nulo × d2-d4+curv × K∈{3,4}.

**Resultado:**
- **La masa SÍ cambia la coherencia del marco:** electrón (poca inercia → se alinea) da holonomía baja
  (0.10-0.37); muón/tauón (pesados → no se alinean) dan alta (~1.1 ≈ aleatorio ≈ NULL). Es un **UMBRAL, no un
  gradiente**: el muón (207×) ya es demasiado pesado para alinearse → muón ≈ tauón ≈ marco-aleatorio.
- **PERO controlando longitud de ciclo, NO selecciona dimensión.** A L=4 el electrón da holonomía idéntica en
  todas las dims (d2=0.09, d3=0.10, d4=0.10, curv=0.10); a L=6 y L=8, igual, sin dim consistentemente menor.
  La coherencia que la masa induce depende de la longitud de ciclo (adyacencia), NO de la dimensión.
- **NULL:** muón/tauón/nulo/marco colapsan al mismo nivel (~1.09); solo electrón/alineado difieren, y aun así
  no seleccionan dim.

**Veredicto A (G2):** la masa toca la COHERENCIA del marco pero NO la GEOMETRÍA. La generación es irrelevante
para la selección de dimensión. Consistente con CS059 y todo el arco — el marco (con o sin inercia) no
selecciona. No es numerología del tres (G-NO-NUMEROLOGÍA ✓).

## MISIÓN B — La gravedad con masa real vs el proxy de grado. El hallazgo (con matiz honesto del NULL).
**Implementación:** masa por nodo log-uniforme en [1,3477], **INDEPENDIENTE del grado** (G-MASA-SEPARADA-DEL-GRADO
verificado: |corr(masa,grado)|<0.15 en todas). Gravedad ∝ masa (fija) vs ∝ grado (=CS057) vs ∝ masa-barajada
(null). Motor de CS057. Dos regímenes: contractivo (W_exp=0.5) y expansivo (W_exp=0.85, donde SÍ hay viables).

**Régimen contractivo:** nada expande (viable=0); la gravedad∝masa deja el diámetro apenas mayor que ∝grado
(menos colapso, z~1.8), mismo orden dimensional. Diferencia marginal.

**Régimen EXPANSIVO (30 semillas, 90/celda) — aquí está la señal:**
| acople | viable | d3 | d4 | curv |
|---|---|---|---|---|
| grado (proxy CS057) | 0.056 | 0.078 | 0.133 | 0.067 |
| **masa** | **0.124** | **0.244** | 0.178 | 0.189 |
| null (masa barajada) | 0.107 | 0.278 | 0.144 | 0.078 |

- **masa vs grado (d3+d4): +0.106, z=2.8** — la gravedad real hace el 3D/4D ~2-3× más viable que el proxy.
- **masa vs null: +0.000, z=0.0 — IDÉNTICOS.** El efecto NO colapsa bajo NULL, pero por una razón reveladora:
  la masa barajada da LO MISMO que la masa real.

**Interpretación honesta (el NULL evita el sobre-claim):** el efecto NO es de la ESTRUCTURA de la masa (el
null lo prueba: barajarla no cambia nada) — es de que la gravedad se acople a un **peso INTRÍNSECO FIJO**
(masa, o masa-barajada, ambos independientes del grado) en vez de al **GRADO, que es AUTO-AMPLIFICANTE**
(preferential attachment: los hubs atraen más → colapso desbocado a dim baja, sesgando CONTRA el 3D).

**Veredicto B:** **el negativo de CS057 era EN PARTE artefacto del proxy de grado.** Su auto-amplificación
sesgaba activamente contra el 3D. Con gravedad acoplada a un peso intrínseco (como la gravedad real, ∝masa),
el 3D sobrevive ~3× más (z=2.8). PERO la masa NO "selecciona" 3D — su estructura es irrelevante (null=masa);
lo que importa es la **independencia del grado**. Es un matiz concreto e importante: parte del negativo del
arco venía del proxy, no de la gravedad; corregido el proxy, el 3D es más amigable — aunque no por la masa.

## GUARDIANES
A: G-LEPTÓN-SIN-COLOR ✓, G-MASA-FÍSICA-FIJA ✓ (razones reales, no afinadas), G-MASA-ES-INERCIA-NO-DIMENSIÓN ✓,
G-NO-INYECTAR-DIM ✓ (holonomía dim-neutral + control de longitud de ciclo), G-NULL ✓, G-NO-NUMEROLOGÍA ✓,
G-NO-FORZAR-3D ✓. B: G-MASA-SEPARADA-DEL-GRADO ✓ (verificado corr<0.15), G-NULL ✓ (**cazó que el efecto no es
de la masa sino de la independencia-del-grado** — el mismo rigor que salvó CS059 del confound). Predicciones
ciegas escritas antes.

## PARA TU ADJUDICACIÓN
CS060 completo, dos misiones, dos regímenes. (A) La masa no selecciona dimensión vía el marco (coherencia sí,
geometría no). (B) Hallazgo con matiz: el proxy de grado de CS054-057 sesgaba contra el 3D por
auto-amplificación; la gravedad con peso intrínseco lo corrige (3D ~3× más viable, z=2.8), pero el NULL prueba
que es la independencia-del-grado, no la masa. **Consecuencia para el arco:** el negativo central (CS057) queda
matizado — parte era del proxy. Vale la pena re-preguntar: ¿re-correr el paisaje de CS057 con gravedad∝peso-
intrínseco (no grado) para ver si el 3D emerge más en TODO el mapa, no solo en este régimen? Y sigue abierto lo
de siempre: nada aquí SELECCIONA 3D por sí mismo — apunta a CS061 (el vértice de 3 puntos / masa emergente),
donde el 3-cuerpos es el ingrediente no probado. Traigo CSVs + este informe. Registrar CS060. Sigue: CS061.

— CC
