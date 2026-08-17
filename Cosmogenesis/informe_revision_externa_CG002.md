# Informe de revisión externa — experimento CG002

**Tipo:** revisión externa (lectura crítica, no ejecución)  
**Fecha:** 30 de junio de 2026  
**Objeto revisado:** arco fundacional + arco κ_Δ, documentos autoritativos CG002  
**Referencias canónicas:** [`INFORME_GENERAL_CG002_ARCO.md`](INFORME_GENERAL_CG002_ARCO.md) · [`INFORME_CG002_VEREDICTOS_TABLA.md`](INFORME_CG002_VEREDICTOS_TABLA.md) · [`NOMENCLATURA_NODOS_CG002.md`](NOMENCLATURA_NODOS_CG002.md) · [`CIERRE_ARCO_CG002_AUTORITATIVO.md`](CIERRE_ARCO_CG002_AUTORITATIVO.md)

---

## Apertura

La idea rectora de Alexis López Tapia, como investigador principal, fue simular las condiciones de emergencia de **S > 0** en una forma visual tridimensional, no como ilustración decorativa, sino como un entorno experimental donde pudiera observarse si una diferencia persistente es capaz de desplegar orden, historia y estructura medible ([`INFORME_GENERAL_CG002_ARCO.md`](INFORME_GENERAL_CG002_ARCO.md), [`INFORME_CG002_GENESIS_VISOR_29jun2026.md`](INFORME_CG002_GENESIS_VISOR_29jun2026.md)). El programa CG002 convierte esa intuición en un sistema computacional reproducible con visor, barridos y criterios de veredicto explícitos, de modo que la visualización tridimensional queda subordinada al experimento y no al revés ([`PROTOCOLO_CG002_ACOPLAMIENTO_ORIGINARIO.md`](PROTOCOLO_CG002_ACOPLAMIENTO_ORIGINARIO.md), [`cg002_experimentos_arco.py`](cg002_experimentos_arco.py)).

## Objeto y diseño

El experimento parte del axioma C-N1, según el cual una diferencia persiste si no se anula, y lo prolonga hacia C-N2 preguntando si esa persistencia solo se sostiene cuando hay acoplamiento entre interior y exterior ([`PROTOCOLO_CG002_ACOPLAMIENTO_ORIGINARIO.md`](PROTOCOLO_CG002_ACOPLAMIENTO_ORIGINARIO.md), [`NOMENCLATURA_NODOS_CG002.md`](NOMENCLATURA_NODOS_CG002.md)). El protocolo impone una sola regla primitiva de acoplamiento, prohíbe introducir resultados por adelantado y exige controles adversariales como acoplamiento apagado, pluralidad mínima y separación entre dirección, tiempo y estructura emergente.

En su versión inicial, el sistema trabajó con firmas discretas en ℤ_K y posteriormente incorporó una firma multicomponente en S^{d−1} para que la tridimensionalidad dejara de ser simple layout y pudiera convertirse en dimensión relacional intrínseca medible ([`PROTOCOLO_CG002_v02_ADDENDUM.md`](PROTOCOLO_CG002_v02_ADDENDUM.md)). El addendum v0.2 muestra que con d=3 el rango relacional emergente llega a 3 y la dimensión efectiva se aproxima a 2.92, mientras que apagar θ_CP anula la asimetría sin colapsar esa dimensionalidad.

## Resultados fundacionales

El arco fundacional reporta varios positivos consistentes. La baryogénesis produce supervivencia aproximada del 50% y amplificación de una asimetría mínima hacia una población orientada, mientras que el caso sin diferencia permanece uniforme y sin estructura ([`INFORME_GENERAL_CG002_ARCO.md`](INFORME_GENERAL_CG002_ARCO.md) §2.1–2.2). La flecha del tiempo también aparece como salida: disminuyen monótonamente tanto la población viva como la entropía de orientación, lo que el programa interpreta como irreversibilidad emergente ligada a la extinción sin reverso (§2.3).

La dimensión emergente rastrea la dimensión del sustrato con valores de correlación cercanos a 0.97, 1.87, 2.69 y 3.44 para firmas en S¹, S², S³ y S⁴ respectivamente, lo que respalda la tesis de que el espacio no se presupone sino que se mide como salida (§2.4). A esto se suma evidencia de coexistencia de dominios bajo acoplamiento local, transición de fase controlada por el alcance y auto-similaridad en la zona crítica, aunque este último punto queda correctamente calificado como preliminar por el propio programa (§2.5–2.7).

## Arco κ_Δ

El cierre más fino del programa aparece en el arco de κ_Δ, donde la pregunta ya no es solo si emerge estructura, sino si existe una huella mínima de diferencia distinguible del ruido finito ([`Informe_arco_kappaDelta_simple.md`](Informe_arco_kappaDelta_simple.md), [`NOMENCLATURA_NODOS_CG002.md`](NOMENCLATURA_NODOS_CG002.md)). La tabla autoritativa resume que el nulo combinatorio escala como 1/√N (exponente −0.5), mientras que el L2 dinámico permanece aproximadamente plano (exponente ≈ 0), con una separación cercana a 12× respecto del nulo a N=2000, lo que respalda la lectura de una estructura no reducible a fluctuación muestral ([`INFORME_CG002_VEREDICTOS_TABLA.md`](INFORME_CG002_VEREDICTOS_TABLA.md) B5–B6; [`grain_null_sweep.csv`](grain_null_sweep.csv), [`cg002_dynamic_l2_sweep.csv`](cg002_dynamic_l2_sweep.csv)).

El resultado decisivo es la estabilización de m_eff/K = ½ y, por equivalencia, del grano 1/√K, sin deriva al barrer la banda de viabilidad en un rango de 6× para K ∈ {6, 8, 12} ([`INFORME_CG002_VEREDICTOS_TABLA.md`](INFORME_CG002_VEREDICTOS_TABLA.md) B7–B8). El cierre autoritativo interpreta esto como derivación estructural de C-N2.5.10 y no como calibración impuesta por C-N5.1, mientras que la capa fina —el exceso de orden sobre el hemisferio— queda situada en torno a 0.015 para d=3, estable frente a η, μ y S_BAND, pero dependiente de la dimensión y por tanto dominio-específica ([`CIERRE_ARCO_CG002_AUTORITATIVO.md`](CIERRE_ARCO_CG002_AUTORITATIVO.md); [`cg002_exceso_barrido.csv`](cg002_exceso_barrido.csv)).

## Evaluación crítica

Como revisor externo, el punto más fuerte del programa es su **disciplina metodológica**. El sistema fue concebido para poder fallar, registra nulls honestos y distingue con claridad entre resultados confirmados, tensiones y no-emergencias, como ocurre con la inercia histórica bajo banda dura, las paredes de dominio nítidas y la no-derivación espontánea de las cuatro fuerzas ([`INFORME_GENERAL_CG002_ARCO.md`](INFORME_GENERAL_CG002_ARCO.md) §2.8–2.11; [`INFORME_CG002_VEREDICTOS_TABLA.md`](INFORME_CG002_VEREDICTOS_TABLA.md) A8, A11–A12). Ese rasgo fortalece la credibilidad interna del conjunto, porque los positivos no aparecen aislados de condiciones de refutación ([`CIERRE_ARCO_CG002_AUTORITATIVO.md`](CIERRE_ARCO_CG002_AUTORITATIVO.md) §Falsabilidad).

La segunda fortaleza es la claridad con que el programa separa estatutos. La nomenclatura canónica establece que un invariante no debe confundirse con una constante escalar, que κ_Δ ≡ 2π/K es una identificación operativa revisable y que la magnitud de ciertos observables depende del dominio en vez de reclamar universalidad numérica inmediata ([`NOMENCLATURA_NODOS_CG002.md`](NOMENCLATURA_NODOS_CG002.md) §§1–5). Esta precaución conceptual evita sobrerreclamar más de lo que los datos muestran.

La principal reserva externa no destruye el resultado, pero sí delimita su alcance. El propio cierre restringe la clausura del arco κ_Δ al régimen documentado con θ_CP=0, banda 6×, K discretos y tamaños finitos hasta N=4000, dejando abiertos otros regímenes y la revisión futura de la identificación de κ_Δ si apareciera otra formulación equivalente ([`CIERRE_ARCO_CG002_AUTORITATIVO.md`](CIERRE_ARCO_CG002_AUTORITATIVO.md) §Alcance). Del mismo modo, el addendum v0.2 reconoce que la flecha dirigida vive en un plano fijo y su peso relativo cae al aumentar d, por lo que una generalización geométrica de θ_CP sería un paso natural de consolidación ([`PROTOCOLO_CG002_v02_ADDENDUM.md`](PROTOCOLO_CG002_v02_ADDENDUM.md) §1).

## Juicio general

El experimento CG002 constituye, en términos de revisión externa, un programa **coherente, reproducible y conceptualmente disciplinado** de cosmosemiótica aplicada. Su contribución principal no es haber derivado una física completa, algo que el propio marco excluye en esta etapa (C-N2.7.6), sino haber mostrado que desde un axioma mínimo de persistencia de la diferencia pueden emerger tiempo, estructura, dimensionalidad, criticidad, invariantes y una huella mínima de diferencia con dos capas distinguibles.

La tesis más defendible del conjunto es esta: CG002 no prueba que el universo físico sea este modelo, pero sí muestra que la cadena teórica que va de **S > 0** a observables discretos puede instanciarse experimentalmente de forma estable, visualizable y falsable en sus propios términos (O-N20.3). Bajo ese criterio, el experimento merece ser considerado un **cierre sólido de fase** para la Cosmogénesis de CG002 y una base legítima para la Parte II del programa (κ_O, κ_LF, κ_H, Λ_Cos).

---

*Revisión alineada con los documentos autoritativos del 30-jun-2026. No supersede `CIERRE_ARCO_CG002_AUTORITATIVO.md`; lo complementa desde una perspectiva externa.*