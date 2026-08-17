# PROTOCOLO CF-2 — PRE-REGISTRO
## "¿El enfriamiento por expansión suaviza el gradiente?"

**Batería:** BATERÍA CF (Cosmo-Física) — corrige errores metodológicos de sesiones previas.
**Experimento:** CF-2 (independiente de CF-1 y siguientes).
**Motiva:** `HALLAZGO_ABIERTO_etapa7_v6_masa_es_linaje_CS.md` — el patrón de fondo a evitar aquí es
el mismo de fondo en toda la batería CF: (a) un solo punto/semilla no es evidencia (T7), y (b) un
juez que sólo mira si la palabra "PASS" aparece en un texto no es un juez (T7, y el bug confirmado
de lectura de JSON en `motor_1a7/pipeline.py` líneas ~124-125, ver abajo, es la ilustración exacta
de por qué).

**Este documento se escribe y congela ANTES de correr el motor de producción.** El motor de
producción (`CF2_estiramiento_motor.py`) y sus resultados (`results/CF2_estiramiento/`) se generan
DESPUÉS de este archivo — verificar mtime.

---

## 1. Pregunta

Al expandirse el espacio, ¿la temperatura baja sola (sin "afuera") y el gradiente se estira —
enfriar ES expandir? Concretamente: ¿el gradiente térmico medido en espacio FÍSICO se suaviza
(cae en magnitud) de forma monótona a medida que crece el factor de expansión `a`, y lo hace de
forma DISTINGUIBLE de un universo de control donde la densidad no se diluye?

## 2. Sustrato (heredado, no se retoca)

Campo continuo T(x,y) en grilla `L×L` (L=64), sin estructura discreta impuesta (T0). Perfil
inicial: salto abrupto tipo tanh de ancho comóvil `W0=1.2` celdas. Difusión isótropa de vecinos
próximos (laplaciano de 4 vecinos), coeficiente `D`. Reloj de expansión: `a(t_g) = exp(H_EXP · t_g)`
con `H_EXP=6.0` — mismo sello que `TEST_RHO_DISPERSION.py` original (Alexis+Grok, 2026-07-22);
no se cambia para favorecer el resultado (T1). `D0=0.12`, `DT=0.25`, `N_SUB=2` — heredados sin
modificación del test viejo.

## 3. Barrido (fija T7 — la falla del test viejo: SEED único, sin barrido)

- **Factor de expansión `a`:** grilla log-espaciada de 7 puntos cubriendo 3 décadas:
  `a ∈ {1, 3.162, 10, 31.62, 100, 316.2, 1000}` (`np.geomspace(1, 1000, 7)`).
  Para cada `a` del barrido se integra la difusión desde `t_g=0` hasta
  `t_g(a) = ln(a) / H_EXP`, muestreando el estado del campo exactamente en ese instante
  (matemáticamente idéntico a re-simular de cero hasta ese punto, porque la actualización de
  difusión no tiene look-ahead — es una cadena markoviana en el tiempo).
- **Semillas:** las 10 semillas estándar del proyecto —
  `7, 42, 99, 777, 2025, 3141, 8191, 99991, 12345, 54321` (el diseño pide ≥8; se usan las 10).
  La semilla sólo perturba la condición inicial con ruido gaussiano de amplitud `1e-4` (no cambia
  la física determinista del transporte) — es un control de robustez frente a ruido, no una fuente
  de aleatoriedad dominante.

## 4. Brazos (NULL que debe morder — T4)

- **REAL:** `ρ = ρ0 / a³` (dilución real), `D = D0 · (ρ/ρ0) = D0 / a³` (el transporte se apaga
  al expandirse — física real).
- **NULL_RHO_FIXED:** `ρ ≡ ρ0` (sin dilución), `D ≡ D0` (constante) — MISMA trayectoria de `a(t)`,
  MISMO ruido de semilla, MISMA condición inicial; la única diferencia es que la densidad no cae.
  Si REAL ≈ NULL_RHO_FIXED en el observable de abajo, el instrumento no discrimina y así se
  reporta (T4), sin maquillaje.

## 5. Observable (T2 — no puede compartir variables con el juez)

Para cada `(modo, semilla, a)` del barrido:

```
∇_comov(a) = max |∂T/∂x|   (banda central, evita wrap-around periódico)
∇_fis(a)   = ∇_comov(a) / a
```

Esta es la ÚNICA cantidad que entra al criterio de abajo. No usa `co_member_score`,
`n_long_co`, ni ninguna variable de linaje/juez de otros experimentos — es geometría pura del
campo T y el factor de expansión.

## 6. Criterio de PASS (congelado, T3 — no se toca si falla)

Por semilla, se calculan dos estadísticos sobre la curva completa `∇_fis(a)` (7 puntos del
barrido), por separado para REAL y para NULL_RHO_FIXED:

1. **`monotonic(modo, semilla)`**: `True` si `∇_fis(a_{i+1}) ≤ ∇_fis(a_i) · (1 + 1e-9)` para
   TODOS los pares consecutivos del barrido completo (tolerancia numérica mínima, no umbral de
   conveniencia).
2. **`slope(modo, semilla)`**: pendiente de la regresión OLS de `ln(∇_fis)` contra `ln(a)` sobre
   los 7 puntos del barrido (forma de la caída; `slope≈-1` es el estiramiento geométrico puro,
   más negativo que -1 indica erosión adicional por difusión sostenida).

**PASS por semilla** (los dos deben cumplirse):
- (a) `monotonic(REAL, semilla) == True`, **y**
- (b) NULL_RHO_FIXED **no** se comporta "de la misma forma" que REAL, definido como:
  `monotonic(NULL, semilla) == False` **o** `|slope(NULL) − slope(REAL)| ≥ SLOPE_DIFF_MIN`,
  con **`SLOPE_DIFF_MIN = 0.05`** (pre-registrado; justificación: `slope≈-1` es la referencia de
  estiramiento puro, 0.05 es un 5% de esa magnitud de referencia — separación modesta pero no
  arbitraria, fijada ANTES de ver los datos, no ajustada después).

**PASS del experimento CF-2:** `rate = (#semillas con PASS) / (#semillas totales) ≥ 0.55`,
**reportando el barrido completo** (las 7×2 curvas por semilla), no un punto ni una semilla
suelta.

Si `rate < 0.55`: se reporta FAIL con los números crudos — no se cambia el juez, no se sustituye
el observable (T3). Si REAL y NULL nunca difieren en ninguna semilla (`rate` estructuralmente en 0
por el término (b) nunca disparar): eso es un hallazgo T4 (el instrumento no discrimina), se
reporta así explícitamente.

## 7. Qué NO es este experimento

- No mide masa, ni linaje, ni Higgs. Sólo el eslabón expansión→densidad→dispersión del
  gradiente térmico, en espacio físico.
- No se auto-adjudica "persiste/no persiste" la hipótesis cosmológica más amplia — eso lo hace
  CS (Grok/Diotallevi) después de ver los números crudos, no este script.
- No toca `motor_1a7/pipeline.py` ni el JSON del test viejo (`TEST_RHO_DISPERSION_result.json`,
  que queda intacto como registro del defecto que motivó esta batería).

## 8. Bug colateral confirmado (no se arregla aquí — reportado solamente)

`motor_1a7/pipeline.py`, función `stage_3_4_stretch_rho`, líneas ~124-125:

```python
v = d.get("verdict", "")
estado.stretch_ok = "PASS" in str(v) or d.get("flags", {}).get("stretch_pure_ok")
estado.rho_ok = "PASS" in str(v) or d.get("flags", {}).get("rho_effect_ok")
```

`d.get("flags", {})` busca `"flags"` en el nivel raíz del JSON, pero en
`TEST_RHO_DISPERSION_result.json` (y en el `verdict` que emite este motor CF-2 también, por
construcción de payload) los flags viven anidados en `d["verdict"]["flags"]`, no en `d["flags"]`.
Por tanto `d.get("flags", {})` siempre devuelve `{}` y el segundo término de cada `or` siempre es
`None`/falsy — el chequeo se reduce en la práctica a `"PASS" in str(v)`, donde `v = d["verdict"]`
es el dict completo convertido a string; como el campo de texto del veredicto (p.ej.
`"TEST_PASS_stretch_and_rho"`) contiene la subcadena `"PASS"`, el chequeo pasa por coincidencia de
texto, no por lectura real de los flags booleanos. Confirmado por lectura de código; **no se
corrige aquí** — corresponde a CS (Alexis ya indicó que lo arregla él).

---

**Fecha/hora de este pre-registro:** ver mtime del archivo (se congela antes de generar
`CF2_estiramiento_motor.py` y cualquier resultado en `results/CF2_estiramiento/`).
