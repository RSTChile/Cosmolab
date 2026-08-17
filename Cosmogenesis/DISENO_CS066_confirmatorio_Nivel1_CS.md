# DISEÑO CS066-confirmatorio — Clavar el exponente diam~N^(1/d) del tejido local
## CS, 11-jul-2026. Confirmatorio de Nivel 1 de CS066 (NO abre número nuevo — cierra el caveat de CS066).

**Qué cierra:** CS066 salida (B) estableció que el tejido local emerge CON especificidad, apoyado en cuatro
firmas convergentes (d_s se estabiliza ~3, clustering 4× el placebo, diámetro 3× el blob, especificidad vs
barajado). La ÚNICA pata floja fue la ley de potencia `diam ~ N^(1/d)`: no salió limpia (slope ~0.13 ruidoso; el
diámetro de local hasta baja un poco de N=2500 a 3500). Causa diagnosticada: el muestreo ALEATORIO de k_local
dejó los bins de localidad fuerte flacos — solo 9 parches en N=3500·(k5-6). Este confirmatorio ataca exactamente
eso: **malla fija de k_local × N con muchos parches por celda**, para que el exponente se mida en vez de
insinuarse.

**Qué NO toca:** el motor de CS066 (`cs066_localidad_geometrogenesis.py`), la definición de tejido, los juicios
de Nivel 2 (el colapso-a-1 ya está adjudicado). Esto es SOLO medir un exponente que ya sabemos de qué signo es.

---

## LA REGLA ANTI-SHANNON (pre-registro del exponente)

Se escribe AQUÍ, antes de correr, qué contará como cada desenlace — para no leer el slope que nos convenga:

- **(CONFIRMA tejido 3D-espacial):** en el régimen de localidad fuerte, `log(diam)` vs `log(N)` da pendiente
  **1/d con d en [2.5, 3.5]** (es decir slope ∈ [0.29, 0.40]), con R² > 0.9 sobre ≥3 puntos de N, y monótona
  (diam crece con N, no baja). Eso convierte "hay tejido" de 4 firmas convergentes a **ley de potencia medida**.
- **(TEJIDO DÉBIL / mundo-pequeño residual):** slope < 0.15 o no monótona → el "tejido" tiene diámetro que casi
  no crece con N ⇒ sigue habiendo atajos ⇒ la localidad adelgaza pero no da una dimensión métrica limpia. El (B)
  se mantiene pero el Nivel 1 baja de "tejido con d≈3" a "tejido parcial, sin dimensión métrica nítida".
- **(NULO):** el barajado da el mismo slope que local → la ley de potencia no es específica de la localidad. Mata
  la afirmación de tejido (improbable dado clustering 0.43 vs 0.10, pero es la cuerda que hay que poder perder).

**El discriminante decisivo sigue siendo local vs local_barajado**, ahora sobre el exponente, no solo sobre
clustering/diámetro puntuales.

---

## DISEÑO

**Malla (no muestreo aleatorio):**
- `k_local` FIJO en la malla: {3, 4, 5, 6, 8, 10} — cubre de localidad fuerte (3-6) a laxa (8-10). El régimen
  decisivo es k∈{3,4,5,6}: ahí debe vivir el tejido.
- `N` en {1500, 2500, 3500, 5000} — se AÑADE N=5000 respecto a CS066 para dar un cuarto punto al ajuste de
  potencia (3 puntos era parte del problema). Si 5000 es caro, mínimo {1500,2500,3500} pero con muchos más
  parches por celda.
- **≥40 parches por celda (k_local, N)** — vs los 9 que mataron el ajuste en CS066. Esto es lo caro y lo que
  importa: los bins de baja-k tienen que estar llenos.

**Brazos (los mismos de CS066, sin inventar):** `local`, `sin_local` (=CS064, control de blob),
`local_barajado` (placebo de especificidad), `local_marco_congelado` (control: tejido sin ejes). No se agregan
brazos — es un confirmatorio, no un experimento nuevo.

**Medición (por celda k×N, promediando parches):**
1. `diam_fin` medio → para el ajuste `log(diam)` vs `log(N)` por cada k fijo.
2. `d_s` medio → debe seguir estabilizándose ~3 en k fuerte (ya visto, se re-confirma con más n).
3. `clustering`, `gigante` → sanidad (tejido sin gas), ya vistos.
4. **El ajuste:** por cada k de localidad fuerte, regresión lineal de `log(diam)` sobre `log(N)` → slope = 1/d.
   Reportar slope ± error, R², y el mismo ajuste para `local_barajado` (debe dar slope menor / no-tejido).

**Guardianes (heredados de CS066, vigentes):**
- G-TEJIDO-ANTES-QUE-EJES: este confirmatorio es SOLO Nivel 1 (tejido). No se re-abre Nivel 2.
- G-NO-CALIBRAR se RELAJA a propósito aquí: k_local se pone en malla FIJA (no sorteado) — pero eso NO es calibrar
  para un resultado, es barrer el eje para medir el exponente. La malla se declara ANTES (arriba) y cubre ambos
  regímenes; no se elige el k "que da 3".
- G-CONTINUIDAD: `sin_local` debe reproducir el blob de CS064 (diam~3.5 plano, d_s se infla). Si no, hay deriva
  de código y el confirmatorio no vale.
- G-ESPECIFICIDAD: local vs local_barajado sobre el exponente (la cuerda anti-Shannon del arco).

---

## COSTO Y SMOKE

- Malla completa: 6 k × 4 N × 4 brazos × 40 parches ≈ 3840 parches (vs 1040 de CS066). Es ~4× el costo de
  CS066. Si es demasiado: recortar a k∈{3,4,5,6} × N∈{1500,2500,3500,5000} × 4 brazos × 40 = 2560, que ya clava
  el exponente en el régimen que importa (la localidad laxa k8-10 es secundaria).
- **Smoke (obligatorio antes de tanda):** 1 celda (k=5, N=1500, 10 parches) × brazos → verificar que reproduce
  los números de CS066 (local d_s~3, clustering~0.43, diam~11; sin_local blob). Si el smoke reproduce CS066, el
  motor no cambió y la tanda es fiable.

---

## ENTREGABLE

Un CSV `cs066conf_kN.csv` (mismas columnas que CS066 + nada nuevo) y una tabla de exponentes: por cada k fuerte,
slope(local) ± err vs slope(barajado), con R². CS audita el ajuste sobre el CSV (no sobre prosa) y firma si el
exponente cae en [0.29, 0.40] (⇒ tejido con d≈3 medido) o si queda en la zona débil (⇒ el Nivel 1 se reporta como
tejido parcial). En cualquier caso, el (B) global de CS066 (espacio ≠ direcciones) NO depende de este resultado —
esto solo endurece o matiza la fuerza del "hay tejido".

— CS. Confirmatorio, no experimento nuevo. Cierra el único cabo suelto de CS066 Nivel 1. 🐝
