# CS074-rcruz — Robustez en N (pedido CS)

**Fecha:** 2026-07-23  
**Pregunta CS:** ¿el umbral en r se mueve con el tamaño N, o es propiedad del mecanismo  
(análogo a “R es de T, no de N” en la suite de masa)?  
**Protocolo:** idéntico a producción; solo cambia N → D y `pasos_lavado` se re-miden del campo.

---

## Configuraciones

| set | N | pasos (medidos) | D (ε>0) | semillas | control r=0 | tiempo |
|-----|---|-----------------|---------|----------|-------------|--------|
| chico | 100 | 1553 | 3.35e-3 | 4 | P=0.032 ✅ | 52 s |
| producción | 200 | 6095 | 8.41e-4 | 8 | P=0.034 ✅ | 1035 s |
| **robustez400** | **400** | **24380** | **2.11e-4** | 8 | P=0.034 ✅ | **3617 s** |

D cae ~1/N² (modos de Fourier suaves en malla más fina) — esperado; el eje **r = H/D** reabsorbe eso.

Artefactos:  
`cs074_rcruz_chico_resultado.json`,  
`cs074_rcruz_produccion_resultado.json`,  
`cs074_rcruz_robustez400_resultado.json`,  
`cs074_rcruz_comparacion_N.json`.

---

## Curva P_real(r) — media ε>0

| r | N=100 | N=200 | N=400 |
|---|-------|-------|-------|
| **0** | **0.032** | **0.034** | **0.034** |
| 0.1 | 0.551 | 0.621 | 0.684 |
| 0.3 | 0.675 | 0.736 | 0.798 |
| 0.5 | 0.724 | 0.812 | 0.851 |
| 1 | 0.801 | 0.867 | 0.899 |
| 2 | 0.838 | 0.908 | 0.935 |
| 5 | 0.907 | 0.941 | 0.963 |
| 10 | 0.938 | 0.960 | 0.974 |
| 100 | 0.976 | 0.988 | 0.993 |

**frac_exp(r)** casi idéntica en los tres N (misma competencia de aislamiento).

---

## Umbrales (misma definición en los tres)

| métrica | N=100 | N=200 | N=400 |
|---------|-------|-------|-------|
| r con P>0.2 | 0.1 | 0.1 | 0.1 |
| r con P>0.5 | **0.1** | **0.1** | **0.1** |
| r half-rise (mitad del salto P_r0→P_max) | **0.1** | **0.1** | **0.1** |
| r con P>0.8 | 1.0 | **0.5** | **0.5** |

N200 vs N400: max |ΔP| = **0.064** (en r=0.1), media |ΔP| = **0.028**.  
Umbrales P>0.5 y half-rise: **idénticos**. P>0.8: estable 0.5 en N≥200 (N=100 un poco más “tarde”).

---

## Lectura para CS (cruda, adjudicable)

1. **El control de lavado (r=0) es N-independiente** (P≈0.033–0.034).  
2. **La ubicación del cruce en r no se mueve** al subir N=100→200→400 en las métricas principales (P>0.5 y half-rise = r=0.1 en los tres).  
3. Hay un **efecto de amplitud suave**: a r fijo, P_real sube un poco con N (más persistencia), no un desplazamiento del umbral a r mayor. Eso es finito-N de forma de curva, no “el crítico se va con N”.  
4. **frac_exp(r)** colapsa entre N → la expansión relativa es la misma función de r.  
5. Analogía con la suite de masa: el eje relevante es **r (razón de tasas)**, no N; el crítico no es un artefacto del tamaño de malla en el rango 100–400.

### Caveats honestos

- El grid de r es grueso (siguiente punto tras 0 es 0.1): no resolvemos un r\* continuo; decimos que el **salto efectivo ocurre en el mismo bin de r**.  
- N=100 es smoke (menos semillas); el par serio es **N=200 vs N=400**.  
- No se corrió N=800 (pasos ~1/D ~ 5e4+, coste alto); 100–400 ya cubre factor 4 en N.

---

## Veredicto operativo (propuesta de mesa, no cierra sin CS)

**Robustez N: PASS cualificado** — el umbral en r es **propiedad del mecanismo (razón expansión/reabsorción)**, no del tamaño N en el rango probado.  
Listo para cerrar el arco r-cruz junto con la adjudicación del run N=200, con este addendum de robustez.

---

## Cómo reproducir

```bash
cd /Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis
./venv/bin/python3 cs074_rcruz.py robustez400   # ~1 h
```
