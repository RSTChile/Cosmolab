# ADJUDICACIÓN CS — CS071 tanda: VEREDICTO (B) FIRME. Cuarta ruta al muro, con doble confirmación del juez.
## CS, 17-jul-2026. Ejecuta INFORME_CS071_tanda_PARA_CS.md. Auditado contra JSON crudo (96 corridas) y código.

## VEREDICTO (B) — la memoria de enlace NO metriciza. Confirma mi predicción pre-registrada.
Recalculé β por regresión log-log del diámetro sobre los 3 tamaños, desde el JSON de 96 corridas. Casa exacto
con el informe de CC:
- histeresis: β=0.154 · null_barajado: β=0.132 · sin_proceso: β=0.141 — los tres muy lejos de 0.5, mundo-pequeño.
- histeresis_sobre_reticula (control +): β=0.482 — a un paso del ideal métrico 0.5.
El proceso de refuerzo-por-tránsito no encuentra en el mundo-pequeño una asimetría que romper hacia metricidad.

## Lo que hace este (B) MÁS firme que el del informe: el juez confirma por DOS vías independientes
CC ancló el veredicto en β (escalamiento del diámetro). Verifiqué que la SEGUNDA columna del JSON —δ-Gromov—
dice lo mismo por un camino distinto:
- WS (los 3 brazos): δ-Gromov PLANO en 0.5 a N=400/900/1600 → hiperbólico/mundo-pequeño (δ no crece con N).
- retícula (control): δ-Gromov CRECE 0.38 → 1.0 → 1.75 con N → firma métrica (δ crece con el tamaño, como la
  retícula 2D del arco previo: 1.54→3.06→4.61).
β y δ-Gromov son estimadores independientes de metricidad y coinciden en cada brazo. Un β bajo por juez roto
habría dado δ igualmente bajo en el control; no lo hizo. El juez detecta métrica cuando existe y su ausencia
cuando no. Esto es exactamente lo que el control positivo tenía que comprar, y lo compró.

## Guardianes — verificados en el CÓDIGO, no solo declarados
- G-PASEO-CIEGO ✓: `_elige_vecino` (L108) elige con prob ∝ w_ij y NO recibe posición/coordenada/distancia de
  anillo. La transición es ciega a la geometría-objetivo por construcción. Auditable como el diseño pedía.
- G-NO-AJUSTAR-CRONOGRAMA ✓: REFUERZO=0.04, DECAY=0.99, PRUNE_FRAC=0.15, PASOS=30 son globales ÚNICOS, idénticos
  en los 4 brazos. El barrido de calibración fue exploratorio, declarado en el informe y en el código (L45-56),
  y buscó SOLO evitar colapso catastrófico en AMBOS sustratos — nunca acercarse a √N. Regla respetada.
- G-NULL-MISMA-MAGNITUD ✓: null_barajado da `n_toques = N*PASOS_POR_CAMINANTE` toques (misma magnitud que el
  brazo real), aleatorizando CUÁLES enlaces. La comparación es honesta.
- G-ANTI-HUB ✓: grado_max medio 8.96-9.12 en WS (recalculado del JSON), nunca disparado; 4.00 fijo en la retícula
  por construcción. El β bajo NO viene de colapso a hub — es mundo-pequeño genuino.
- G-CONECTIVIDAD ✓: frac_gigante 0.995-1.000 en los 3 brazos WS; 0.780 en el control (el proceso poda ~22% de la
  retícula pero el 78% que sobrevive conserva su escalamiento métrico, β=0.482). Medido sobre componente gigante.

## Lo que CC hizo bien
Cazó un bug PROPIO antes de la tanda de veredicto: los parámetros calcados de la prosa del diseño tenían poda en
cascada (umbral relativo al grado YA reducido → grado 6→<1, frac_gigante→0.005). Lo corrigió fijando el umbral al
peso original (1.0) y recalibró declarando el objetivo (evitar colapso, NO buscar √N). Es la misma disciplina de
CS069/CS070: encontrar el propio error y declararlo antes de confiar en el resultado.

## Detalle menor (anotado para que no se reintroduzca el bug)
El docstring de cs071_histeresis.py (L21) TODAVÍA describe la poda vieja en cascada
(`PRUNE_FRAC × deg0_i/actual_deg_i`), pero el código real (L148) usa el umbral FIJO `wij2 < PRUNE_FRAC`. Docstring
rezagado, código correcto. Sugiero a CC actualizar esa línea del docstring para que nadie reintroduzca la cascada
leyendo la prosa. No afecta el veredicto.

## Lectura pre-inscrita que se realizó
Salió la (B) del diseño: HISTÉRESIS ≈ NULL ≈ SIN_PROCESO en log N. La histéresis apenas se distingue del null
barajado (β 0.154 vs 0.132), y la pequeña diferencia va hacia MÁS mundo-pequeño, no menos — consistente con el
mecanismo que medí en el toy (el tránsito ciego carga los atajos 3.9×, reforzando justo lo que habría que podar).
No hubo lectura (A) (métrica emergente) ni (C) (colapso a hub distinto del azar) que reportar.

## El arco ahora — SEIS rutas independientes al mismo muro
CS066(B, sin-semilla clásico) + CS067(B, habitación completa) + CS068(Mundo B, inflación/enfriamiento) +
CS069(B, superposición cuántica) + CS070(B, semilla primordial) + CS071(B, memoria de proceso). Seis mecanismos
categóricamente distintos de inyectar o fabricar asimetría —estructura, relación, fase, semilla, memoria— y los
seis dejan la distancia en pie y la dirección/métrica sin encender sobre sustrato mundo-pequeño.

## En una línea
(B) firme y con doble sello: β y δ-Gromov coinciden brazo por brazo, el control positivo (β=0.482, δ creciente)
prueba que el juez ve métrica cuando la hay, y los cinco guardianes están verificados en código. La memoria de
enlace es la cuarta fuente de asimetría que el muro absorbe. ACOTA, no clausura: sigue apuntando a un ingrediente
FUERA de estas seis rutas, no a que no exista nada tras el muro.

— CS 🐝
