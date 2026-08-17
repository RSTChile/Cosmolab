# INSTRUCCIÓN — Módulo grad-h (Price & Monaghan) para la ignición CS073
**De:** CS (diseño + adjudicación). **Para:** CC. Decisión de Alexis: construir el formalismo correcto;
vale la pena. Regla de operación vigente: CC implementa lo especificado, no modifica a su arbitrio; un
cambio es un dato a coordinar.

## ARQUITECTURA (Alexis, explícita): MÓDULO APARTE, se integra SÓLO cuando esté probado y funcione
- grad-h se construye como módulo AISLADO (`p_gravedad_gradh.py`), NO se toca el motor validado ni el
  bucle de ignición todavía.
- Se prueba SOLO, contra casos con solución conocida, hasta que conserve energía dentro de tolerancia.
- SÓLO cuando pase esas pruebas se enchufa al bucle. Hasta entonces, el motor y el diseño de ignición
  quedan CONGELADOS como están (con la deriva documentada, sin fingir que se resolvió).

## Qué resuelve (recordatorio del problema)
El softening adaptativo ε_i(posiciones) hace que el "potencial" se mueva bajo el integrador → el leapfrog
no conserva energía (deriva 0.13→0.60). Price & Monaghan (2007, MNRAS) añade los TÉRMINOS DE CORRECCIÓN
por el gradiente del softening (grad-h): el factor Ω_i que corrige la ecuación de movimiento para que la
fuerza SIGA siendo el gradiente exacto de una energía bien definida, aun con ε variable. Con eso el
leapfrog vuelve a conservar energía por construcción.

## Especificación del módulo
1. **h_i (longitud de suavizado) por densidad autoconsistente:** h_i tal que ρ_i = Σ_j m_j W(r_ij, h_i)
   con h_i = η(m_i/ρ_i)^(1/3) — resuelto iterativamente (Newton-Raphson, 2-3 iter), como en SPH estándar.
   η≈1.2 (convención, NO ajustada). Kernel W = cubic spline estándar.
2. **Factor grad-h Ω_i = 1 − (∂h_i/∂ρ_i)·Σ_j m_j ∂W_ij/∂h_i** — el término de corrección. Entra en la
   ecuación de fuerza gravitatoria con softening variable (Price & Monaghan 2007 ec. 22-28, la forma
   energía-conservante del par gravedad+softening).
3. **Integrador:** leapfrog KDK con la fuerza corregida por Ω. Δt individual/jerárquico como ya adjudicado.

## PRUEBAS DE ACEPTACIÓN (obligatorias, ANTES de integrar — casos con solución conocida)
El módulo NO se enchufa hasta pasar las tres:
1. **Colapso de esfera fría (Evrard / cold collapse):** problema estándar con comportamiento conocido;
   |ΔE/E| debe quedar acotado (<1e-2) a través del colapso y rebote. Es EL test canónico de grad-h.
2. **Órbita de 2 cuerpos ligada:** energía y momento angular conservados sobre muchas órbitas (deriva
   secular < tolerancia). Verifica que Ω no introduce fuerza espuria.
3. **Equilibrio de Jeans estático:** una nube en equilibrio hidrostático NO colapsa espuriamente ni se
   dispersa (el criterio de Jeans se respeta en reposo).
Reportar |ΔE/E| de los tres. Si alguno falla, es un DATO (grad-h no basta / hay otro problema), NO se
retoca la tolerancia ni se fuerza el paso.

## SÓLO tras pasar las 3 pruebas — reintegrar al bucle de ignición
Enchufar `p_gravedad_gradh` en lugar del softening fijo, correr el bucle completo (malla causal semilla +
expansión + gravedad grad-h + CDM + H2) hasta ignición, con el observable pre-registrado SIN CAMBIOS:
- ¿Un núcleo cruza M_J local por colapso real, con |ΔE/E| acotado (colapso FÍSICO, no deriva)?
- **REAL vs NULL** (aristas malla barajadas) en el punto de ignición, z-score ≥5 semillas × ≥8 NULL.
- Tres resultados pre-inscritos intactos: (A) cruza y gana al NULL = cierre positivo; (B) no cruza con
  energía conservada = negativo robusto (falta física, no numérica); (C) cruza pero no gana al NULL.

## Guardianes (heredados + nuevo)
G-DIFERENCIA-INTERNA, G-SIN-SIEMBRA, G-SIN-ENERGIA-NUEVA, G-EXPANSION-ISOTROPA, G-PARAMETROS-ESTRUCTURALES
(η=1.2 y kernel spline son convención SPH, no ajustados), G-CONSERVACION-ENERGIA (|ΔE/E|<1e-2, el umbral
que YA fijé — NO se afloja), G-MODULO-PROBADO-ANTES-DE-INTEGRAR (nuevo, Alexis: no se enchufa sin pasar
las 3 pruebas de aceptación).

## Costo
Desarrollo grande + O(N²)+iteración de densidad. Todo en entorno de CC / segundo plano. Sin prisa: la
corrección es construir bien, no rápido.