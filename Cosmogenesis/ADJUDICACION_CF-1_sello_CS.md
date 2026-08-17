# ADJUDICACIÓN CF-1 — ¿La diferencia persiste bajo expansión?

**Director:** Alexis López Tapia · **Redacción:** Claude (CC/CS, este instancia) · **Fecha:** 2026-07-23
**Estatuto:** **PROPUESTA DE SELLO — no cierra sin autorización explícita del director** (regla permanente del proyecto).
**Naturaleza:** solo adjudicación, sin código nuevo — re-sello de CS074-rcruz + robustez N, ya corridos.

---

## 1. Pregunta simple

¿Una diferencia mínima en un campo caliente persiste si el todo se expande más rápido de lo que se reabsorbe, en vez de borrarse?

## 2. Qué se corrió (verificado en disco, no de palabra)

| corrida | N | pasos | D (ε>0) | semillas | control r=0 | archivo |
|---|---|---|---|---|---|---|
| chico (smoke) | 100 | 1.553 | 3.35e-3 | 4 | P=0.032 | `cs074_rcruz_chico_resultado.json` |
| producción | 200 | 6.095 | 8.41e-4 | 8 | P=0.034 | `cs074_rcruz_produccion_resultado.json` |
| robustez400 | 400 | 24.380 | 2.11e-4 | 8 | P=0.034 | `cs074_rcruz_robustez400_resultado.json` |

Verifiqué directamente `cs074_rcruz_produccion_resultado.json`: 80 filas, `control_r0_lava: True`, N=200 — coincide con lo narrado en `RESUMEN_CS074_rcruz_robustez_N_PARA_CS.md`.

## 3. Observable y NULL

Forma × magnitud (autocorrelación espacial × varianza normalizada) del gradiente, tras 5 defectos de instrumento corregidos en la sesión previa (piso de varianza, NULL barajado-por-paso, sustrato discreto→continuo, observable de magnitud→forma, forma-sola→forma×magnitud). NULL = mismo campo barajado una sola vez al final (no cada paso).

## 4. Resultado

- **r=0 lava** en los tres N (P≈0.033–0.034), N-independiente.
- **Curva monótona** en r: P sube de ~0.03 a ~0.99 según N (tabla completa en el RESUMEN citado).
- **El cruce vive en el mismo bin de r** (r=0.1) al triplicar N dos veces (100→200→400) en las dos métricas principales (r con P>0.5, r half-rise).

## 5. Reservas honestas (no se suavizan)

1. **Grid de r grueso**: el siguiente punto tras 0 es 0.1 — no se resuelve un r* continuo, solo que el salto efectivo cae en ese bin.
2. **La invariancia a N NO es total**: la métrica r-con-P>0.8 se mueve de bin (1.0 en N=100 → 0.5 en N=200/400). El *encendido* (P>0.5, half-rise) es N-invariante; la cola alta de la curva depende algo de N. (Corrección ya incorporada desde la auditoría cruzada previa, hallazgo `d622550b`.)
3. N=100 es smoke (4 semillas); el par serio es N=200 vs N=400 (8 semillas cada uno).
4. No se corrió N=800 (costo ~horas); el rango probado es 100–400 (factor 4).

## 6. Muro que evita (T4)

El observable forma×magnitud hace morder al NULL de verdad — no es un observable invariante bajo permutación (ese fue el defecto 4/5 de la sesión de instrumento).

## 7. Veredicto propuesto

**CF-1: PASS cualificado.** El mecanismo (expansión congela una diferencia de forma en un campo continuo) está probado y es robusto en su encendido a través de N; el valor crítico fino y la cola alta de la curva quedan como abiertos declarados, no como pendientes de este sello.

**No se declara cierre de arco.** Esta adjudicación queda a la espera de tu autorización explícita, por la regla permanente del proyecto (ningún veredicto de cierre es válido sin el director).

## 8. Artefactos

- `cs074_rcruz_chico_resultado.json`
- `cs074_rcruz_produccion_resultado.json`
- `cs074_rcruz_robustez400_resultado.json`
- `RESUMEN_CS074_rcruz_robustez_N_PARA_CS.md`
- `RESUMEN_CS074_rcruz_PARA_CS.md`
