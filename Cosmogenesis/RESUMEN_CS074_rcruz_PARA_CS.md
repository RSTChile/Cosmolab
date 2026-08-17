# CS074-rcruz — régimen de cruce real r≈1

**Fecha:** 2026-07-23  
**Código:** `cs074_rcruz.py` (sucesor; **no** modifica `cs074_persistencia_campo.py`)  
**JSON producción:** `cs074_rcruz_produccion_resultado.json`  
**Smoke:** `cs074_rcruz_chico_resultado.json`  
**Tiempo producción:** ~1035 s  

---

## Por qué se reabrió

El run original (`cs074_produccion_resultado_crudo.json`) **no cruzó r≈1**:

| problema | valor original | efecto |
|----------|----------------|--------|
| D medido (N=800) | ~5e-5 | r = H/D ≥ 0.05/D ≳ **1000** para todo H>0 |
| pasos | 120 | ≪ tiempo de lavado → a H=0, P~0.99 (control roto) |
| expansión `round(H·N)` | con H·N≪1 → 0 cortes | en r pequeño la expansión no actuaba |

La pre-inscripción pedía: r≪1 lava, r≈1 transición, r≫1 congela, H=0 → P≈0.

---

## Diseño r-cruz (pre-registrado)

1. **Misma física de campo:** mancha ε, difusión solo por aristas vivas, NULL = permuta φ al final, P = autocorr×var.
2. **D medido** (un paso, H=0) — no impuesto.
3. **pasos_lavado medido:** tiempo hasta P < 0.05 a H=0 (mediana × 1.15). N=200 → mediana 5300, **pasos=6095**.
4. **Eje r** pre-registrado: `{0, 0.1, 0.3, 0.5, 1, 2, 5, 10, 30, 100}`.  
   **H(r) = min(r·D, 1)** — H sale de D, no se elige la grilla de H a mano.
5. **Expansión Bernoulli** (P_corte = H por arista viva). Corrige el bug `round(H·N)→0` del original en H pequeño. Esperanza de fracción cortada/paso = H.
6. N=200 (lavado tractable; aún campo continuo). semillas=8. ε = [0, 1e-9, …, 1].

---

## Controles (producción)

| control | resultado |
|---------|-----------|
| ε=0 → P=0 a todo r | **PASS** |
| r=0 (H=0), ε>0 → difusión lava | **PASS** — mean P_real = **0.034** (< 0.15) |
| NULL < REAL cuando hay señal | **PASS** (z>0 en r≥0.1) |

---

## Curva producción (media sobre ε>0; idéntica por ε en dinámica lineal)

| r | P_real | P_null | z | std_ratio | frac_exp |
|---|--------|--------|---|-----------|----------|
| **0** | **0.034** | 0.001 | 0.26 | 0.18 | 0.00 |
| 0.1 | 0.621 | 0.008 | **4.90** | 0.79 | 0.40 |
| 0.3 | 0.736 | 0.011 | 5.80 | 0.86 | 0.78 |
| 0.5 | 0.812 | 0.018 | 6.35 | 0.91 | 0.91 |
| **1** | **0.867** | 0.023 | **6.75** | 0.94 | 0.99 |
| 2 | 0.908 | 0.026 | 7.05 | 0.96 | 1.00 |
| 5 | 0.941 | 0.017 | 7.39 | 0.97 | 1.00 |
| 10 | 0.960 | 0.016 | 7.55 | 0.98 | 1.00 |
| 30 | 0.975 | 0.016 | 7.68 | 0.99 | 1.00 |
| 100 | 0.988 | 0.016 | 7.77 | 1.00 | 1.00 |

D medido (ε>0) ≈ **8.41×10⁻⁴** (N=200).

---

## Lectura cruda (para adjudicación CS — no es veredicto final del autor)

1. **El control de lavado se recupera:** sin expansión la diferencia se reabsorbe (P~0.03).
2. **Hay cruce de régimen real** en r: de P≈0.03 (r=0) a P≳0.6 ya en r=0.1, monótono hasta P≈0.99 a r≫1.
3. **REAL ≫ NULL** en cuanto hay expansión efectiva (z~5–8). El NULL (baraja forma) no sostiene persistencia.
4. **El umbral no es un escalón exacto en r=1** (P ya alta en r=0.1). La D de un paso **subestima** el lavado multi-paso efectivo → el r “nominal” 1 no es el punto crítico exacto; la **competencia expansión vs reabsorción** sí se ve.
5. **Independiente de ε** (ε≥1e-9): misma curva → solo importa *haber* diferencia, no su magnitud (en este juguete lineal).
6. **Cuantos:** a r alto el histograma se va a k=1 (regiones aisladas de un punto de malla); a r=0 quedan regiones grandes. Lectura cosmológica = CS.

### Diferencias honestas vs CS074 original

- N=200 (no 800) por tractabilidad del lavado.  
- Expansión Bernoulli (necesaria para H·N≪1).  
- pasos y grilla de r **calibrados al régimen**, no la grilla H fija del original.

---

## Robustez N (pedido CS, 2026-07-23)

Corrida **N=400** mismo protocolo. Umbrales P>0.5 y half-rise en r=0.1 **idénticos** a N=200;  
max|ΔP| N200–N400 = 0.064. Ver `RESUMEN_CS074_rcruz_robustez_N_PARA_CS.md`.

---

## Cómo reproducir

```bash
cd /Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis
./venv/bin/python3 cs074_rcruz.py chico         # N=100 ~50 s
./venv/bin/python3 cs074_rcruz.py produccion    # N=200 ~15–20 min
./venv/bin/python3 cs074_rcruz.py robustez400   # N=400 ~1 h
```

---

## Artefactos

| archivo | rol |
|---------|-----|
| `cs074_rcruz.py` | código |
| `cs074_rcruz_produccion_resultado.json` | curva N=200 |
| `cs074_rcruz_robustez400_resultado.json` | curva N=400 |
| `cs074_rcruz_comparacion_N.json` | tablas N=100/200/400 |
| `cs074_rcruz_produccion_meta.txt` | calibración + tiempo |
| `cs074_rcruz_chico_resultado.json` | smoke |
| `RESUMEN_CS074_rcruz_robustez_N_PARA_CS.md` | informe robustez |
| `cs074_persistencia_campo.py` | **intacto** (run original) |
| `cs074_produccion_resultado_crudo.json` | run original (sin cruce) |
