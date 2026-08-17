# ANÁLISIS — cómo integrar los hallazgos CS072 en UniversoCosmosemiotico.html

## 1. Qué tiene la página HOY (inventario real, leído del código)
- **Motor Three.js/WebGL**, 665 líneas, autocontenido. 5 pestañas-modo:
  `genesis` (nube que se auto-organiza), `cosmos` (posiciones fijas, acopla sólo vecinos rc),
  `narrativa` ("del punto al cosmos"), `qcd` (plasma), `kappa` (arco κ_Δ, motor ℤ_K).
- **cfg** = {mode, N(2–100000), d(2–6), theta(θ_CP asimetría), alpha(acoplamiento), seed, rc, L, speed, qcdVer, kappaK, bandMult}.
- **Sliders vivos**: N, d, θ_CP, α, rc, velocidad, config inicial, banda (κ). Los que reconstruyen el universo
  re-inician al SOLTAR (no al arrastrar) — patrón ya resuelto, hay que respetarlo.
- **Física genesis** (stepGenesis): cada par calcula afinidad `gdot`, acumula en matriz `W`, mueve posiciones con
  fuerza cohesión `KC` + repulsión `KB`; S decae por `MU`, muere si S≤`KAPPA_S`; cuenta τ (tiempo) y dStruct.
  Campo-medio O(N) para N grande (stepGenesisBig). YA tiene: S>0, decaimiento, muerte, tiempo emergente τ.
- **HUD/lectores**: nodos vivos, fase, orden local/global, χ (susceptibilidad, pico rc≈3), nodos-cumplidos.

## 2. Lo que hay que meter (hallazgos CS072 de esta sesión) y DÓNDE encaja
El principio rector (del director): **NO parámetros que predeterminen; sólo condiciones estructurales.** Cada control
nuevo debe ser una CONDICIÓN cuya consecuencia se OBSERVA, no un dial que fabrica el resultado. Y todo lo que ya
existe queda incluido (nada se quita).

### (A) Slider de MAGNITUD DE LA ASIMETRÍA — ya existe como θ_CP
El pedido "slider de magnitud de la asimetría" YA está: `theta` (θ_CP, −1..+1). No hay que crear otro; sí RE-ETIQUETARLO
para que se lea como la asimetría primordial S>0, y conectarlo al hallazgo: con θ=0 (uniforme) → no rompe simetría
(no-go); con θ≠0 → emerge estructura. Es la demostración visual del origen de S>0.

### (B) Slider de VELOCIDAD DE EXPANSIÓN — NUEVO, es el corazón del hallazgo de dimensión
No existe hoy. Añadir `tasa_expansion` (0 → alta). Es la palanca de las DOS FASES:
  - expansión baja → la red retiene mundo-pequeño → diámetro satura (dimensión no crece).
  - expansión alta (superlumínica en la fase pre-luz) → corta atajos de largo alcance → el horizonte causal deja
    crecer el diámetro con N → la dimensión EMERGE.
Visualmente: al subir el slider, la nube pasa de "bola apretada" (diámetro chico, todo cerca) a "tejido extendido"
(diámetro grande, dimensión efectiva mayor). Es lo que el director pidió ver: "cómo la expansión hace emerger el espacio".

### (C) CHECKS de las 18+3 fuerzas — NUEVO, panel de apagado
El pedido: "meter las 18 cosas + 3 como checks y ver qué pasa si sacas una o varias". En el motor real esto es el
parámetro `apagar`. En la página: una lista de checkboxes (fuerte, EM, gravedad, débil, aniquilación, fluctuaciones,
materia oscura, …). Al desmarcar una, la simulación RE-INICIA sin esa fuerza y se OBSERVA el colapso:
  - sin EM → no se ligan pares → sin "átomos" → geometría colapsa (diámetro→1). [el guardián, hecho visible]
  - sin materia oscura → sin poda de largo alcance → grafo completo → diámetro=1 (Big Crunch topológico).
  - sin fuerte → estructura de tríos no cierra.
Hipótesis a mostrar (el director las pidió como hipótesis): sacar gravedad cuántica / relatividad general / sector
oscuro → predecir el régimen resultante ANTES de correr, luego correr y comparar.

### (D) DIMENSIÓN EMERGENTE como lector — NUEVO en el HUD
Hoy el HUD muestra orden/χ. Añadir la dimensión efectiva medida en vivo = pendiente log-log del diámetro (el mismo
estimador que validamos, con arranque físico). Que el número suba de ~1 a ~3 al subir N y expansión es LA imagen del
hallazgo. Etiquetar: "dim emergente = 1/pendiente(diámetro vs N)".

### (E) RUGOSIDAD / fondo no uniforme — SÓLO si EMERGE, nunca pintado (corrección del director)
CORRECCIÓN: el fondo rugoso NO puede ser un "modo de vista" que coloree la densidad — eso sería imponer el resultado
(Shannon). El fondo rugoso es POSTERIOR a la expansión inicial y debe SURGIR de la dinámica, o no se muestra.
Regla: el mapa rugoso sólo aparece DESPUÉS de que la expansión + las fuerzas hayan corrido y consolidado diferencias
de distribución que la propia simulación produjo. Si en una corrida dada NO emerge rugosidad (p.ej. θ=0, uniforme),
NO se pinta nada — pantalla homogénea, que es el resultado honesto. La rugosidad es un OBSERVABLE de salida (¿el
estado final tiene estructura de distribución distinta del uniforme?), medido contra su NULL (barajado), no una capa
decorativa. Implementación: medir varianza de densidad local del estado consolidado; mostrar el mapa SÓLO si esa
varianza supera la del barajado. Sin emergencia → sin mapa.

### (F) c CONTINGENTE — nota, no slider
El hallazgo "c es fósil de la transición, contingente por D" NO se mete como dial (sería Shannon: imponer c). Se
muestra como CONSECUENCIA: al variar d (dimensión de firma) + expansión, el "punto de quiebre" donde el diámetro deja
de crecer se corre — y ese punto ES el análogo de c. Se anota en el panel info, no se parametriza.

## 3. Riesgo anti-Shannon (el guardián que el director exige)
- Los sliders nuevos (expansión, checks) deben ser CONDICIONES, no resultados. La dimensión NO se elige con un slider;
  emerge de N + expansión + nº de distinciones. El slider `d` actual (dimensión de firma) es la CONDICIÓN (cuántas
  distinciones), no la dimensión resultante — mantener esa distinción clara en las etiquetas para no confundir.
- El desempate de vecinos y el arranque del estimador deben ser por física (magnitud), no por índice — igual que en el
  motor. Si el layout usa el orden del array para algo visible, es Shannon encubierto.
- NULL visible: un botón "barajar" que permute las identidades debería dejar los OBSERVABLES (diámetro, dimensión)
  intactos. Si cambian, hay Shannon en el render.

## 4. Plan de implementación sugerido (incremental, sin romper lo que anda)
1. Re-etiquetar θ_CP como "magnitud de asimetría S>0" (0 palabras de código nuevo, sólo label + nota info).
2. Añadir slider `tasa_expansion` + su efecto en el layout (las dos fases). Nuevo cfg.expansion.
3. Añadir panel de checks de fuerzas (apagar) → re-init sin esa fuerza. Nuevo cfg.apagar[].
4. Añadir lector "dim emergente" al HUD (pendiente diámetro-vs-N sobre una sub-muestra, barato).
5. Rugosidad EMERGENTE: medir varianza de densidad local del estado consolidado vs barajado; mostrar mapa SÓLO si
   supera el NULL. Sin emergencia → pantalla homogénea (resultado honesto). Nunca pintar la densidad como capa.
6. Nota c-contingente en el info de la pestaña génesis.
NO tocar: el patrón re-init-al-soltar, el motor κ_Δ/QCD (ya cerrados), el campo-medio O(N) para N grande.

## 5. Lo que NO recomiendo
- NO portar el motor Python (`proceso_sucesivo.py`) al navegador tal cual: la página es una VISUALIZACIÓN cualitativa
  en tiempo real, no el motor de veredicto. El veredicto lo da el Python (CC lo corre). La página ILUSTRA, no adjudica.
- NO añadir un slider "dimensión resultante" — sería el Shannon que el arco entero combatió.
