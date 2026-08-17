# PROTOCOLO cs074-A — ¿Por qué demasiada asimetría produce menos estructura?

**Congelado (pre-registro):** 2026-07-26 · **Ejecutor:** CC · **Director:** Alexis López Tapia
**Diseño base:** `DISENO_tres_experimentos_holistico_PARA_CC.md` (Experimento A, leído entero).
**Motor reusado (leído, NO editado en su física; solo se le agregaron dos campos aditivos
de fragmentación al final de `correr_holistico_energia()`, que no cambian ningún valor ya
reportado de cs074):** `cs074_energia_holistica.py`, ya verificado (352 corridas, 0 fallas
de conservación, ver `RESULTADO_cs074_energia_holistica_barrido_completo_PARA_CS.md`).

Este documento se congela ANTES de escribir el script del experimento. Cualquier
desviación se reporta explícita, no se edita retroactivamente (T3).

---

## 1. Pregunta

El barrido de cs074 mostró, sin buscarlo, que la fracción de masa ligada en estructura NO
crece con ε (la asimetría fundacional) — al contrario, cae: ε=0,5→74,5%, ε=4,0→8,4%.
¿Por qué? Tres explicaciones posibles, no excluyentes:
1. **Energética:** la asimetría alta hace que la gravedad forme (y cobre) estructura ANTES,
   agotando la reserva temprano, dejando menos presupuesto para lo que sigue.
2. **Mecánica:** la asimetría alta dispersa la materia (la separa en vez de juntarla),
   independiente de cuánta energía haya disponible.
3. **Mixta:** ambas actúan, en proporción distinta según el régimen de ε.

## 2. Barrido (sobredimensionado — mucho más allá de donde vimos el efecto, ε=0,5 a 4,0)

| Eje | Rango | Puntos |
|---|---|---|
| ε (`amp_rugosidad`) | logspace(1e-3, 10) | 20 |
| `E_reserva` | logspace(1e-3, 1e3) × mecánica de referencia (mismo grid que cs074) | 7 |
| semillas | 0..11 | 12 |

Total: 20×7×12 = **1680 corridas finitas**, más 20×12 = **240 corridas de control**
(energía apagada, presupuesto infinito — independiente de `E_reserva`, se corre una vez
por (ε, semilla)). **1920 corridas en total.**

## 3. Tres observables en paralelo (para distinguir las 3 explicaciones)

1. **`frac_masa_ligada`** (el observable viejo, ya existe en el motor) — reproduce el techo.
2. **`frac_reserva_gastada_temprano`** — de la reserva total disponible, qué fracción ya se
   gastó (vía cobros de Regla 4) al llegar a 1/3 del total de pasos de la corrida (lectura
   de la curva por-paso, `guardar_curva=True`, checkpoint `t = n_pasos_estructura//3`).
   Sube hacia 1 si la asimetría alta agota el presupuesto temprano.
3. **`frac_masa_en_mayor_cluster`** y **`n_clusters_finales`** (fragmentación, ya agregados
   al motor) — si la asimetría alta DISPERSA la materia, se espera más clusters, más chicos
   (frac_masa_en_mayor_cluster baja, n_clusters_finales sube), incluso a igualdad de
   reserva.

## 4. Control (mecanismo vs energía)

Mismo barrido de ε × semillas, con `energia_on=False` (presupuesto infinito, el gate de
Regla 4 nunca bloquea). Se compara el techo (`frac_masa_ligada` vs ε) CON energía y SIN
energía:
- Si el techo no-monótono **desaparece** sin el costo de energía → el efecto es
  **energético** (la asimetría alta agota la reserva antes de tiempo, cuando la reserva es
  infinita eso deja de importar).
- Si el techo **persiste** sin energía → el efecto es **mecánico** (la asimetría alta
  dispersa la materia, con o sin presupuesto).
- Si persiste pero atenuado → **mixto**, se reporta la proporción.

## 5. PASS pre-registrado (tres lecturas, cualquiera es un hallazgo real)

Se reportan las curvas completas (media ± dispersión entre semillas) de los 3 observables
vs ε, para ambos brazos (con/sin energía). La lectura (energética / mecánica / mixta) se
determina por:
- Energética si: el techo desaparece sin energía Y `frac_reserva_gastada_temprano` crece
  con ε en el brazo con energía.
- Mecánica si: el techo persiste sin energía Y `frac_masa_en_mayor_cluster` cae (o
  `n_clusters_finales` sube) con ε en AMBOS brazos.
- Mixta si hay señal parcial de ambas.
- Ninguna de las tres anteriores con señal clara → se reporta como no explicado por estos
  tres observables, honesto, sin forzar una lectura.

## 6. Trampas

- **T1:** ningún número a mano — ε, `E_reserva` y semillas barridos, nunca fijados para
  pegarle a un resultado.
- **T-conservación:** se hereda el chequeo duro de cs074 (5% en el control de gravedad
  pura) — si alguna corrida individual lo dispara, se marca y se excluye del promedio, no
  se oculta.
- **T-target:** no aplica aquí (este experimento no compara contra 4,9%/31,5%, es sobre el
  MECANISMO del techo, no sobre el valor final).
- **Perturbación dinámica + semillas:** 12 semillas por celda (cumple el mínimo del
  documento madre).
- **La cantidad medida ≠ su juez:** `frac_masa_ligada` (el techo a explicar) es un
  observable YA CALCULADO por el motor antes de este experimento; `frac_reserva_gastada_
  temprano` y la fragmentación son observables NUEVOS e independientes — ninguno se deriva
  del otro.

## 7. Qué se entrega a CS, sin adjudicar

Curvas completas (los 3 observables vs ε, ambos brazos), la lectura que mejor calza según
§5 (declarada, no forzada), dispersión entre semillas, y el JSON crudo completo.
**No se cierra el hallazgo aquí** — CS lee, el director decide.
