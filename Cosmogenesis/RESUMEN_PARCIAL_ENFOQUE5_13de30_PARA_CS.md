# ENFOQUE 5 — Resumen parcial (13/30 cerrados)

**Fecha:** 2026-07-25 · **No exhaustivo** — un párrafo por experimento, para revisión mientras cierran los 17 restantes. Ningún veredicto de arco está adjudicado; esto es entrega cruda resumida.

---

## ⚠ Tres cosas que cruzan varios experimentos (léase antes que el resto)

1. **Ruido dinámico no es N-invariante (hallazgo de E5.6-3):** el mecanismo de ruido por paso (heredado de E5.1-1, `NOISE_REL=0.02`, calibrado para N=200) domina la dinámica a N grande — viola conservación (E1) hasta 98% y mata el NULL a N≥2048. **E5.1-1 usa el mismo mecanismo con r hasta 1e3** — sus resultados, cuando lleguen, hay que leerlos con esto en mente.
2. **E5.3-1 se corrigió a sí mismo antes de producción** (excluyó la componente gigante de "estructura ligada"). **E5.3-5 auditó una versión anterior, no corregida**, de esa definición — su hallazgo de "NULL 100% degenerado" no aplica a la versión final de E5.3-1. Reconciliado abajo.
3. **Definiciones de E/X/S_ent divergen entre hermanos del Tema 2/5**: E5.2-1 (la base canónica prevista) terminó su protocolo después de que E5.2-2 y E5.5-1 ya habían arrancado con otra definición razonable. No es error de nadie — es la regla "si no está, define la tuya" funcionando — pero hay que homologar antes de comparar curvas entre ellos.

---

## TEMA 1 — Persistencia de exergía

**E5.1-2 · Vida media (τ vs D, sin expansión) — PASS.** τ≈0.63/D, ley limpia (R²=0.9999), monótona sin excepción. Dispersión entre semillas exactamente cero (explicado: modos de Fourier ortogonales, no artefacto). Caveat: el rango de D pedido por la plantilla (hasta 1e2) es **inalcanzable** con `cs074_rcruz.py` — techo real D≈0.10.

**E5.1-5 · Expansión no monótona (perfiles H(t)) — NO robusto al perfil.** En la zona de transición (r_eff≈0.06–1) el perfil domina sobre el ruido de semilla (hasta 4.3×). Orden físico consistente: ráfaga temprana > frenante > constante ≈ ráfagas múltiples > ráfaga tardía > acelerante — el "cuándo" de la expansión importa, no solo el total. El NULL barajado es ciego por diseño a X_final (std es invariante a permutación); la falsabilidad real la dio el observable secundario P.

---

## TEMA 2 — Conservación del presupuesto

**E5.2-2 · Anticorrelación X↔S_ent — PASS casi perfecto, con frontera real.** r≈−0.9999 a −1.0000, determinista, en 44/44 celdas válidas de la región central. Se rompe por completo en ε=1.0 — causa mecánica identificada (φ cruza cero, S_ent deja de ser monótona respecto a X ahí), no artefacto ni error de calibración.

---

## TEMA 3 — Eficiencia de conversión (el ancla contra 4.9%/31.5%)

**E5.3-1 · Eficiencia 12 décadas — señal real pero débil.** Corregida antes de producción (excluye componente gigante). Diferencia REAL−NULL real pero chica (t≈7.7, +0.4%). 12/169 celdas caen cerca de 4.9% y 2/169 cerca de 31.5% sin ajuste — pero justo en la zona de mayor ruido/menor separación del NULL, así que queda como observación cruda, no confirmación.

**E5.3-5 · Falsación externa — negativo limpio, tras encontrar 2 defectos de instrumento.** Diagnosticó que la definición (temprana) de E5.3-1 tenía NULL degenerado por topología, y que el ruido dinámico de esa misma ficha domina el 85% de la grilla. Controlando por ambos: **ninguna celda con z genuino cae cerca de 4.9% ni 31.5%**. El punto más cercano real y estable (método de E5.3-2, ligadura máxima) da una meseta eficiencia≈0.2725, bien discriminada del NULL (z≈2.4) — pero a 4.25 puntos porcentuales de 31.5%, fuera de tolerancia.

---

## TEMA 4 — Exergía y enfriamiento adiabático

**E5.4-1 · Exergía vs enfriamiento medido — FAIL, bien diagnosticado.** El NULL sin expansión también correlaciona X con T (r≈0.83, a veces más fuerte que REAL). X y T comparten el mismo reloj de relajación difusiva — la correlación temporal no distingue REAL de NULL. La diferencia real (consistente con CF2) está en la magnitud final retenida, no en la correlación — que era justo lo que este protocolo probaba.

**E5.4-4 · Exergía espectral — PASS.** Orden "escalas grandes se congelan primero" confirmado (ρ=1.000, 16/16 semillas), NULL nunca congela, verificado con identidad de Parseval a precisión de máquina. El modo fundamental concentra 84-88% de toda la exergía del sistema.

**E5.4-5 · Control con baño externo — PASS limpio.** 12/12 semillas: T se aplana y converge al baño, X colapsa. Confirmación independiente: con baño se viola la conservación E1 activamente (deriva ~10⁴); sin baño se respeta a precisión de máquina.

---

## TEMA 5 — Muerte térmica vs Nada

**E5.5-1 · Barrido fino ε→0 — PASS en las tres curvas.** Deriva de E1 cae exactamente como ε² (medida, no impuesta). Punto fijo en ε=0 verificado numéricamente (no asumido). Usa definición heredada de E5.2-2 (ver nota cruzada arriba).

**E5.5-3 · Reversibilidad de la muerte térmica — recuperable, no absorbente.** Cualquier amplitud de re-inyección revive X (escala ~amp²), sin punto de no retorno hasta 32× el tiempo de muerte. Pero cada recuperación es transitoria (misma constante de decaimiento τ que la muerte original). Complementa Tema 1: la supervivencia *permanente* de exergía necesita el aislamiento por expansión, no basta con re-inyectar energía bajo difusión pura.

---

## TEMA 6 — Definición y verificación cruzada de la exergía

**E5.6-2 · Energía libre F=E−TS — coherencia parcial (forma sí, escala no).** Correlación de forma X vs (E−T·S_ent) = 0.9996, pero solo 6.7% de las celdas cumple la tolerancia de escala, todas en el extremo ε=1.0. Segundo método independiente confirma el negativo (correlación T_implícito vs T_medido = −0.22, débil y de signo contrario).

**E5.6-3 · Invariancia a N — X/N no estable, pero causa raíz es de diseño, no física.** El ruido dinámico (heredado de E5.1-1) no escala con N: domina a N≥2048, viola E1 hasta 98%, mata el NULL. Zona N≤512 es confiable y muestra decaimiento monótono consistente con efecto de tamaño finito genuino. **Ver nota cruzada #1 arriba — afecta la lectura de E5.1-1.**

**E5.6-4 · Sensibilidad a la referencia de equilibrio — NO invariante.** Solo REF_GLOBAL (la referencia fija clásica, la que usa todo el proyecto desde CS074) reproduce la transición persiste/no-persiste esperada. REF_LOCAL y REF_DINÁMICA fallan, pero por defectos de instrumento identificados (NULL degenerado en LOCAL; la referencia dinámica converge sobre su propia trayectoria) — no por evidencia física en contra.

---

**Pendientes (17):** E5.1-1, E5.1-3, E5.1-4, E5.2-1, E5.2-3, E5.2-4, E5.2-5, E5.3-2, E5.3-3, E5.3-4, E5.4-2, E5.4-3, E5.5-2, E5.5-4, E5.5-5, E5.6-1, E5.6-5.
