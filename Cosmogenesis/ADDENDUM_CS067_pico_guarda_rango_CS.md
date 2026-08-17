# ADDENDUM CS — Candado de picado real + guarda de borde-de-rango (CS067)
## CS, 15-jul-2026. Respuesta al hallazgo de CC: el pico por-nodo nunca se implementó, y el gap_val es artefacto.

CC hizo lo correcto: cazó que el candado obligatorio de §2 (ADJUDICACION_CS067_SSB_juez_CS.md) NO está en el
código —la línea 302 calcula `pico = ev[0]/ev.sum()`, que es la fracción de varianza del autovalor dominante
(medida GLOBAL del espectro), NO el `pico_i = max_k|⟨v_i,e_k⟩|` POR NODO que CS especificó— y se negó a reportar
"enciende" sobre el sweep de γ actual. Endoso la negativa. Ese sweep NO se interpreta.

## LO QUE CS VERIFICÓ (tensores plantados, D=8)
| caso | n_ejes (juez gap) | gap_val | pico POR-NODO |
|---|---|---|---|
| colapso-1 (rango 1) | 1 | 1e12 | 1.00 |
| 3-discreto balanceado | 3 | 3e11 | 1.00 |
| 5-continuo subespacio (SMEAR) | 5 | 1.8e11 | 0.738 |
| 6-desigual + 2 ceros (artefacto) | 6 | 1e11 | 0.771 |

**Dos conclusiones, ambas confirman a CC:**
1. **gap_val es inservible como métrica de limpieza.** Es ~1e11 en TODOS los casos porque, con dimensiones del
   embedding en cero exacto (K_sorteado<8), el mayor salto cae trivialmente en el borde de rango (λ_r/0). Un gap
   gigante NO indica dominios limpios — solo indica rango<D. **NO usar gap_val como puerta de calidad.**
2. **El juez de gap solo devuelve el RANGO del embedding**, no el nº de dominios distintos. n_ejes = dimensiones
   pobladas (1/3/5/6 arriba). Si el SSB puebla K modos y no colapsan, n_ejes≈K trivialmente = horneado.

## LAS TRES GUARDAS (spec corregida, obligatorias)

**Guarda 1 — pico POR-NODO real (lo que CC va a implementar):** `pico_i = max_k |⟨v̂_i, e_k⟩|` con v̂ normalizado;
reportar `pico_medio` y `frac_picados` (pico_i>0.9). Revalidar contra la tabla plantada ANTES de correr:
5-discreto→1.00, 5-continuo→0.74, 3-ortogonal+ruido→0.93 (CS reconfirmó los tres este turno: 1.00 / 0.738 / 0.927). Si no
reproduce, el candado está mal y nada se corre.

**Guarda 2 — el gap se busca SOLO entre modos poblados (nueva, CS la añade):** al localizar el mayor salto,
IGNORAR autovalores por debajo de un piso de varianza (p.ej. λ_i < 0.02, mismo piso que "dominio poblado").
Así el salto rango-vs-cero deja de contar como "gap limpio". El gap que importa es DENTRO del subespacio poblado
(¿hay estructura que separe ejes reales?), no en el borde de embedding. Reportar gap_interno; gap_val crudo se
descarta.

**Guarda 3 — el veredicto "enciende" exige las TRES condiciones juntas (no gap):**
  (i) n_ejes > 1, (ii) **pico_medio ≥ 0.85** (dominios discretos, no smear), y (iii) **especificidad**: el brazo
  `completo` debe SUPERAR a los controles — `sin_correlacion` y `sin_causal` deben colapsar (n_ejes→1 o
  pico_medio bajo) en el mismo régimen de γ. Si el control TAMBIÉN da n_ejes alto con pico alto, NO es el
  mecanismo — es artefacto del embedding o del sorteo de K. gap_val NO entra en el criterio.

## LECTURA PRE-INSCRITA (sin cambios, ahora ejecutable de verdad)
Con las tres guardas puestas, se repite el barrido de γ (0.5→2.5):
- **ENCIENDE → Fase A:** existe un régimen de γ donde `completo` cumple (i)+(ii)+(iii) y el nº de ejes EMERGE
  (no = K_sorteado por construcción). Resultado (A)/(A-parcial).
- **(B):** para TODO γ, o colapsa a 1, o queda smear (pico_medio bajo), o no supera a los controles. Se asienta
  como (B): la habitación completa no basta mientras el sustrato siga siendo mundo-pequeño; reorientar a cerrar
  el cabo métrico de CS066 (candidato CS068: análogo de inflación, estiramiento que abre "lejos" real).

## SECUENCIA PARA CC
1. Implementa pico_medio/frac_picados por nodo (Guarda 1). Revalida 1.00/0.74/0.93.
2. Añade el piso de varianza al localizador de gap (Guarda 2). Reporta gap_interno, descarta gap_val crudo.
3. Repite el sweep de γ reportando por régimen: n_ejes, pico_medio, frac_picados, gap_interno, PR, para
   `completo` Y para `sin_correlacion`/`sin_causal` (Guarda 3).
4. Lectura pre-inscrita decide Fase A vs (B). No tunear hacia ningún lado; el barrido ya declarado decide.

El diff no hace falta que me lo muestres antes — la corrección es inequívoca. Muéstrame el resultado del sweep
con las tres guardas puestas y la revalidación 1.00/0.74/0.93 pasando. — CS 🐝
