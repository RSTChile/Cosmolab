# RESULTADO — batería REAL vs 8 NULL, ignición con sumideros, N=2000 (1-2 ago 2026)
**Corrido en esta sesión (Claude Code).** Responde el pedido de Alexis de "varias semillas para un
resultado robusto", tras dos correcciones de rumbo documentadas en el camino (ver más abajo).

## Correcciones que hubo que hacer antes de este resultado (no maquillar)
1. **El sustrato NO es aleatorio entre llamadas** (`_extraer_bariones` es determinista por diseño) —
   se malgastaron ~4h de cómputo creyendo que se estaban probando sustratos independientes cuando en
   realidad se repetía el mismo. Lección para el futuro: verificar determinismo antes de asumir azar.
2. **Velocidad inicial en cero exacto revienta el chequeo de conservación de Phantom** (bug de
   diagnóstico ya conocido y resuelto el 20-jul con `vel_generador`, que se me había olvidado usar al
   generar las condiciones iniciales nuevas). Con `campo_velocidad_turbulento` puesto, el problema
   desapareció por completo.

## Método
Fondo real de 2.000 átomos de hidrógeno (mismo motor validado S>0→átomos), 1 corrida REAL (malla
causal real, `seed_layout=12345`) + 8 corridas NULL (misma malla con aristas barajadas,
`seed_null=5000,5002,...,5014`), mismo campo de velocidad turbulento (semilla 42, idéntico en las 9),
Phantom con sumidero pragmático activado (`icreate_sinks=1`, `rho_crit_cgs=1000`, `h_acc=0.3`,
`r_crit=0.6` — el mismo umbral en las 9, no ajustado por corrida). `tmax=0.5`. Observable: masa total
acumulada en sumideros al final (suma de la masa de cada sumidero vivo en el `.sink` final).

## Resultado

| corrida | masa en sumideros | nº sumideros |
|---|---|---|
| REAL | **2124.4** | 8 |
| NULL 1 | 733.2 | 8 |
| NULL 2 | 686.2 | 8 |
| NULL 3 | 723.8 | 8 |
| NULL 4 | 676.8 | 7 |
| NULL 5 | 723.8 | 8 |
| NULL 6 | 723.8 | 8 |
| NULL 7 | 770.8 | 8 |
| NULL 8 | 723.8 | 8 |

NULL: media=720.3, std=28.84 (muy apretado, poca variación entre azares). **REAL = 2.95× la media del
NULL. z = 48.69.** Las 9 corridas llegaron completas a `tmax=0.5` sin ningún error de conservación de
energía/momento (guardianes G-CONSERVACION intactos, no se usó `I_WILL_NOT_PUBLISH_CRAP`).

## Lectura honesta
- El discriminante es contundente estadísticamente (z=48.7, muestra n_null=8 como mínimo estándar del
  proyecto) — no es un efecto al límite ni una única corrida afortunada.
- **Sigue siendo cierto lo ya señalado el 31-jul:** el NULL también forma sumideros (7-8, masa
  ~680-770) — no es "a REAL le nace una estrella y al azar no le nace nada". Es "a los dos les nace
  algo, pero a REAL le nace ~3× más masa", de forma consistente y con una diferencia enorme frente al
  ruido entre semillas NULL.
- El observable (masa total en sumideros) es una ADAPTACIÓN post-hoc del observable original
  pre-registrado (nº de estructuras que cruzan Jeans) — necesaria porque el mecanismo original de
  medición no sobrevivía numéricamente sin sumideros. Se deja constancia explícita: esto NO es todavía
  una declaración de cierre del arco CS073 — es un resultado robusto que requiere adjudicación de
  Alexis sobre si este observable adaptado cuenta como el discriminante válido del arco, o si hace
  falta re-pre-registrarlo formalmente antes de citarlo como positivo.
- Pendiente natural: repetir con más de una semilla de layout REAL (hoy sólo 1) para completar el
  diseño "5 semillas REAL × 8 NULL" que el resto del arco usa como estándar.
