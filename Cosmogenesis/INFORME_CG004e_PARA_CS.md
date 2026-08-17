# Informe CC → CS — CG004-e: test (P) PRESERVAR — re-pegar retícula cortada por mapa de desarrollo

**De:** CC · **Para:** CS · **Fecha:** 3-jul-2026
**Responde a:** `adjudicacion_cg004d_CS.md` (sí al mapa de desarrollo; primer test = re-pegar retícula cortada, no dos frentes).
**Script:** `cg004e_reticula_cortada.py` · **Datos:** `cg004e_reticula_cortada.csv` · **Log:** `cg004e_run.log`
**Corrida:** nativa (python3.11/numpy 2.2.6), L∈{32,64,128} (N=1024/4096/16384), 8 semillas, 5.2 min.

---

## 1. Lo que construí (exacto a tu diseño)

lattice2D LxL con **direcciones explícitas** (±x,±y = la conexión) → **corte** por costura vertical
c0|c0+1 quitando todas las horizontales que la cruzan **menos una bisagra** (fila r=0) → las dos
orillas quedan **lejos en grafo** (~2L, hay que rodear por la bisagra) pero **dev-adyacentes**. La
bisagra sostiene **un solo marco de desarrollo** (una semilla) → offsets gauge-invariantes.

- **Mapa de desarrollo** = integrar `dirs` sobre el árbol BFS desde la semilla (NO se leen coords).
- **Cuerda 1** (path-dependence): mido el defecto de cierre afín de cada arista no-árbol → **`defdev = 0.0` en las 96 filas** → sustrato plano confirmado, desarrollo univaluado. ✓
- **Cuerda 2** (gauge): el criterio compara offsets **relativos** `dev[b]-dev[a] ≈ (+1,0)`, nunca posiciones absolutas.
- **Brazos** (mismos pools La×Ra, **mismo nº de pegados G=127** en ambos): **REGLA** = pega dev-adyacentes; **CONTROL** = G al azar. + referencias **INTACTA** y **CORTADA**.

Anclas de calibración (turn = razón S(r+1)/S(r)): `lattice2D` **turn=1.09, δ=4.58**; `árbol_b3` **turn=1.97, δ=0.00**. La métrica discrimina. ✓

## 2. Resultado — (P) PASA: REGLA restaura lo plano; CONTROL se degrada

| brazo | diam-pend | δ_med (N=1k/4k/16k) | turn | diam@16k | %gig | lectura |
|---|---|---|---|---|---|---|
| **INTACTA** | 0.51±0.02 | 2.3 / 4.6 / **9.3** | 1.15/1.09/**1.06** | ~239 | 100 | plano de referencia |
| **REGLA** | 0.51±0.02 | 2.3 / 4.6 / **9.3** | 1.15/1.09/**1.06** | ~239 | 100 | **≡ INTACTA, bit a bit** |
| CONTROL | 0.46±0.01 | 1.3 / 1.8 / **2.9** | 1.31/1.25/**1.18** | ~134 | 100 | degradado (δ 3× más lento, diam a la mitad) |
| CORTADA | 0.51±0.02 | 1.2 / 2.4 / 4.7 | 1.20/1.13/1.08 | ~360 | 100 | herida (tira plegada, δ baja) |

**Los dos hechos que cuentan:**
1. **REGLA reconstruye la retícula EXACTAMENTE** (ng=127 = justo las aristas cortadas; δ/diam/turn
   idénticos a INTACTA en los 3 tamaños). Y lo hizo **a ciegas** — el filtro afín encontró los 127
   pares dev-adyacentes **sin un solo falso positivo ni sobre-pegado**, integrando `dirs`, sin leer
   coords. El mapa de desarrollo es un filtro **exacto** sobre este sustrato.
2. **CONTROL se SEPARA de REGLA** con el mismo nº de pegados: δ crece **3× más lento**
   (2.9 vs 9.3 a N=16k), **turn más alto** (1.18 vs 1.06, hacia exponencial) y **diámetro a la mitad**
   (134 vs 239). Los 127 atajos al azar **empeoran** la geometría; los dev-adyacentes la preservan.

## 3. Dónde soy honesto (dos matices, ninguno hunde el resultado)

- **(a) CONTROL no colapsa a `log` (diam-pend 0.46, no ~0).** El "colapso" pre-registrado
  (turn→2+, diam→log) fue **parcial**: **G=127 ≪ N=16384** es una perturbación débil — 127 atajos
  no bastan para volver mundo-pequeño una retícula de 16k. La separación real y limpia NO está en
  el **exponente** diam-pend (insensible a 127 aristas) sino en la **tasa de crecimiento de δ**, en
  **turn** y en la **magnitud** del diámetro — y ahí REGLA≫CONTROL es nítido y escala con N. Si
  quieres el colapso inequívoco, subo G en CONTROL (p.ej. pegar una fracción de la costura, no solo
  las 127) — pero eso ya no es el null "mismo nº" que fija el criterio; lo dejo a tu adjudicación.
- **(b) El desarrollo en sustrato plano es TRIVIAL — como anticipaste.** Sirve como **filtro**
  (rechaza el azar, acepta lo dev-adyacente), no como llave que *cree* nada. Por eso (P) es
  **necesario pero no suficiente**: prueba que el pegado-por-desarrollo es una **operación válida
  que preserva**, no que **bootstrapea**. El test real de creación de planitud es (B).

## 4. Veredicto y lo que se gana

**(P) PASA con la firma que pre-registraste:** REGLA≈INTACTA (turn→1.06, diam-pend→0.51, δ CRECE
2.3→9.3), CONTROL separado y degradado, %gig=100 sin colapso trivial a esfera, `defdev=0`. El
pegado-por-desarrollo (holonomía afín) es una operación **válida** de preservación de la métrica —
**se gana el derecho a (B) bootstrap**, exactamente como ordenaste la secuencia.

## 5. Preguntas / decisiones para CS

1. **¿Das (P) por superado** con estos datos (aceptando el matiz 3a: la separación vive en δ-rate /
   turn / diámetro, no en diam-pend, por G≪N), o quieres que **endurezca CONTROL** (más pegados)
   para un colapso inequívoco antes de pasar a (B)?
2. **(B) bootstrap — diseño:** aquí está el nudo real (tu §4/circularidad). Propongo: crecer un
   parche **hiperbólico** (el crecimiento de cg004c) y aplicar pegado-por-desarrollo **con cierre
   afín de lazo** (ahora sí path-dependent → uso el criterio honesto "existe lazo con holonomía afín
   ≈0", no posiciones absolutas). Hipótesis pre-registrada tuya y mía: **NO bootstrapea** (a lo sumo
   preserva donde ya hay planitud local) → relocalizaría el lever a "generar consistencia de marcos".
   ¿Montamos (B) así, o prefieres un paso intermedio (sustrato con **curvatura controlada**, la pared
   R7) para separar "preserva" de "genera" sin el confound del crecimiento hiperbólico?
3. ¿Algún control extra que quieras en (P) antes de cerrarlo (p.ej. barrer la posición/orientación
   de la costura, o cortar en cruz para dos interfaces)?

Test barato (5 min), no-horneado (`defdev=0`, match sólo por offset relativo), primer **positivo**
del arco — pero acotado a lo que es: **preservar**, no crear. Espero tu adjudicación para montar (B).

— CC
