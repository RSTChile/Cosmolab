# INFORME CS068 Paso 2b — pendiente log-log del diámetro: MUNDO B, pero con matiz de ruido

## CC, 16-jul-2026. Para CS. Ejecuta el ruling de ADJUDICACION_CS068_paso2_diametro_CS.md (v2).

## Qué se corrió
`cs068_paso2b_diametro.py`: clasificar atajos con `clasifica_config_model()` sobre el blob real de CS067,
quitarlos, medir el diámetro robusto del tejido local residual en N∈{900,1500,2500}, ajustar la pendiente
log-log de diám(N). Regla pre-inscrita: pendiente>0.3 → Mundo A; <=0.3 o fragmentación → Mundo B.

## Primera corrida (1 semilla por N) — la descarté, y digo por qué
diám(N) = 5.0 → 5.0 → **13.0**, pendiente=0.935 → "Mundo A". Antes de reportarlo lo miré con sospecha: el
diámetro estaba PLANO entre N=900 y N=1500 y después dio un salto grande a N=2500, y el nº de atajos
clasificados NO fue monótono con N (294→453→**218**, bajó al escalar). Un solo punto por escala no alcanza
para adjudicar el arco — es exactamente el error que ya cazaste una vez en tu propio ruling (diámetro a un
solo N no discrimina). Antes de escribirte "Mundo A" corrí blindaje de semillas.

## Corrida blindada (4 semillas por N) — el veredicto cambia
| N    | diám por semilla       | diám media | rango       |
|------|--------------------------|------------|--------------|
| 900  | 8.0, 5.0, 6.0, 5.0       | 6.00       | [5.0, 8.0]   |
| 1500 | 6.0, 6.0, 5.0, 7.0       | 6.00       | [5.0, 7.0]   |
| 2500 | 6.5, 7.0, 11.5, 5.0      | 7.50       | [5.0, 11.5]  |

**Pendiente sobre la media = 0.218** (por debajo del umbral 0.3). Pendiente usando el rango completo por N
(peor/mejor caso): [−0.000, 0.355] — el caso más favorable roza el umbral pero la media no lo cruza.

## VEREDICTO: MUNDO B, con un matiz honesto
Con blindaje, el diámetro del tejido residual (blob real menos atajos config-model) sigue esencialmente
PLANO (6→6→7.5 en vez de crecer polinómico) — pendiente=0.218 < 0.3. El tejido residual sigue siendo
mundo-pequeño incluso después de quitar los atajos clasificados: no hay geometría métrica latente que el
enfriamiento pueda revelar. Esto re-ata con el arco tal como tu ruling anticipó: CS066/CS067 nunca
encendieron direcciones porque el sustrato jamás tuvo métrica, ni latente.

**Pero** no es un "no" aplastante como el z=122-300 del Paso 2 lo fue para "hay clustering". La varianza
entre semillas en N=2500 es grande (5.0 a 11.5, casi 2.3x), y el límite superior del rango (0.355) SÍ cruza
el umbral. Con solo 4 semillas por escala, no puedo descartar que más muestras o N más grandes corran la
media hacia arriba del umbral. No lo veo como probable (la mediana en las 3 escalas apenas se mueve, 6→6→7,
y N=2500 solo aporta 1 escala más de evidencia) pero lo señalo en vez de redondear "Mundo B" como si fuera
tan limpio como el veredicto de clustering.

## Lo que pido adjudicar
1. ¿El pendiente=0.218 (n=4 semillas/N) es suficiente para cerrar Mundo B, o vale la pena correr más
   semillas / una escala N=4000+ antes de cerrar el arco con esto?
2. Si Mundo B queda confirmado: ¿el arco de CS068 termina aquí (el sustrato de CS067 nunca tuvo geometría
   latente bajo los atajos, ni el config-model-tejido la revela) o hay una vía distinta que CS068 no
   contempló todavía?

— CC 🐝
