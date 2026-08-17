# INSTRUCCIÓN — Ignición CS073 vía PHANTOM (SPH de Price & Monaghan), NO módulo casero
**De:** CS (diseño + adjudicación). **Para:** CC. Regla de operación vigente: CC implementa lo
especificado, no modifica a su arbitrio; un cambio es un dato a coordinar.

## CAMBIO DE RUMBO (Alexis, 20-jul): usar Phantom en vez de construir grad-h desde cero
Alexis señaló —correcto— que el formalismo grad-h (Price & Monaghan 2007) que íbamos a construir YA existe,
validado y mantenido por su propio autor: **Phantom** (código SPH público, Daniel Price / Monash). Correr
en Mac Apple Silicon (M-series) es viable hasta ~10⁵ partículas en minutos-horas. Construir grad-h nosotros
sería reinventar la rueda con más superficie de bug. → Se DESCARTA el módulo p_gravedad_gradh casero. Se
usa Phantom como INTEGRADOR gravedad+hidrodinámica.

## LA TRAMPA A EVITAR (crítica — el núcleo anti-Shannon de todo esto)
Phantom es un simulador de FÍSICA ESTÁNDAR. Dale una nube favorable y forma una estrella — pero eso NO
valida Cosmosemiótica, sólo confirma que la gravedad forma estrellas (ya lo sabíamos = el control positivo
que ya corrimos). **La pregunta con peso anti-Shannon NO es "¿se puede formar una estrella?" — es "¿la
coherencia relacional del sustrato (malla causal, dos fases) le gana al NULL?"** Esa pregunta es NUESTRA,
no de Phantom.

## EL ENCUADRE CORRECTO (confirmado por Alexis): Phantom = integrador, condiciones iniciales = NUESTRAS
Phantom NO valida la teoría. Phantom es el integrador HONESTO (el que sí conserva energía, validado por
expertos, estándar del campo) que corremos sobre DOS conjuntos de condiciones iniciales que generamos
NOSOTROS:
- **REAL:** el sustrato con la malla causal desplegada dinámicamente (la coherencia relacional que ya ganó
  al NULL a resolución fija, z=10.26).
- **NULL:** el MISMO sustrato con las aristas de la malla causal barajadas (preservando grado/peso).
Si Phantom, corriendo FÍSICA IDÉNTICA sobre ambos, enciende estrella en REAL y NO en NULL → cierre positivo
con el discriminante intacto Y con un integrador que nadie puede acusar de casero. Doble ganancia: nos
ahorra construir grad-h, y blinda contra "usaron su propio integrador".

## FASE 0 — Instalar y validar Phantom (antes de tocar nuestros datos)
- Instalar: xcode-select --install; brew install gcc make openmpi; clonar Phantom (github.com/danieljprice/
  phantom); compilar el setup de colapso pre-estelar (OpenMP, Apple Silicon nativo). Splash para visualizar.
- **Prueba de humo del propio Phantom:** correr un problema de prueba que trae (p.ej. polytrope/colapso) y
  confirmar que |ΔE/E| queda acotado. Esto NO es física nuestra — sólo confirma que la instalación funciona.
  Guardar la salida. Si Phantom no compila o no conserva energía en su propio test, PARAR y reportar.

## FASE 1 — El punto DELICADO: traducir NUESTRO sustrato a condiciones iniciales de Phantom SIN Shannon
Aquí está el único lugar donde puede colarse Shannon. La traducción debe ser MECÁNICA e IDÉNTICA para REAL
y NULL — la ÚNICA diferencia entre ambos es el barajado de aristas, nada más.
- De cada corrida de nuestro motor (malla causal desplegada dinámicamente) salen: posiciones 3D de las
  partículas + su masa (H, fija ≈9.4) + temperatura/densidad heredadas. Eso se escribe como archivo de
  condiciones iniciales de Phantom (formato de partículas SPH).
- **REAL:** posiciones del despliegue dinámico de la malla causal (la coherencia relacional, z=10.26).
- **NULL:** MISMO procedimiento exacto, pero con las aristas de la malla barajadas (grado/peso preservados)
  ANTES del despliegue → posiciones sin la coherencia relacional. Todo lo demás idéntico (misma masa, misma
  T, mismo N, mismos parámetros de Phantom).
- PROHIBIDO: tocar a mano posiciones/densidades para "ayudar" a que REAL colapse; elegir parámetros de
  Phantom distintos entre REAL y NULL; sembrar sobredensidades. G-DIFERENCIA-INTERNA + G-SIN-SIEMBRA.
- Reportar el script de traducción para que CS lo audite ANTES de correr en escala.

## FASE 2 — Correr Phantom sobre REAL y NULL, física IDÉNTICA
- Mismos parámetros de Phantom (gravedad + hidro + enfriamiento) para ambos brazos. Phantom aporta el
  integrador que SÍ conserva energía (grad-h ya está dentro de Phantom, validado por su autor).
- ≥5 semillas × ≥8 NULL (o el barajado que el presupuesto permita, declarado).
- Escala: empezar chico (N~10³-10⁴, minutos-horas en el Mac) y subir si hace falta.

## Observable de cierre (pre-registrado, SIN CAMBIOS respecto al diseño de ignición)
- ¿Un núcleo cruza M_J local por colapso REAL en Phantom (con su energía conservada, que Phantom garantiza)?
- **REAL vs NULL:** ¿REAL enciende (cruza Jeans / forma sink particle) significativamente más que el NULL?
  z-score. NO basta "cruzó Jeans en absoluto" — tiene que GANARLE AL NULL.
- Tres resultados pre-inscritos intactos: (A) REAL cruza y gana al NULL = CIERRE POSITIVO (estrella emerge
  de la coherencia relacional, sobre integrador estándar del campo); (B) REAL no cruza, o REAL=NULL =
  negativo robusto (falta física real, no numérica — Phantom ya no puede ser el culpable); (C) cruza pero
  no gana al NULL = parcial.

## Lo que Phantom NO hace (recordatorio anti-Shannon)
Phantom NO valida la teoría. Si le das una nube favorable, forma estrella — eso es física estándar, ya
sabido (= control positivo ya corrido). El valor está EXCLUSIVAMENTE en el contraste REAL vs NULL: física
idéntica, sólo cambia si las condiciones iniciales llevan o no la coherencia relacional del sustrato.

## Guardianes
G-DIFERENCIA-INTERNA (NULL = aristas barajadas, único cambio). G-SIN-SIEMBRA (no sobredensidades a mano).
G-SIN-ENERGIA-NUEVA. G-EXPANSION-ISOTROPA. G-PARAMETROS-IDENTICOS-REAL-NULL (mismos parámetros de Phantom
en ambos brazos — nuevo, el riesgo específico de usar un motor externo). G-TRADUCCION-MECANICA (la
conversión sustrato→IC de Phantom es idéntica para REAL y NULL; auditada por CS antes de escalar).

## Costo
Phantom en Mac Apple Silicon: ~10³-10⁴ partículas en minutos-horas; ~10⁵ en horas. En entorno de CC. Sin
prisa: la corrección es correr bien, no rápido.