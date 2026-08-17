# Diagnóstico CS — errores de código que dejan columnas sin datos en los WebLive

**Auditor:** Claude Science · **Fecha:** 3-jul-2026
**Archivos:** VST_CelulaMadre_WebLive_A.py (y B/C/D, gemelos) · VST_HomeostasisEmergente.py · VST_Metabolismo.py
**Método:** trazado del bucle _fila() (L699-760) contra las columnas mudas del CSV. Corrijo mi
diagnóstico previo: NO es un cable de energía cortado. Son DOS cosas, una es bug de ORDEN.

## HALLAZGO 1 (BUG REAL, de orden) — el soporte se calcula ANTES que sus insumos
En el bucle _fila() el orden es:
```
L735  _rc_observar(d)                  # escribe RC_*  ✓
L741  HOMEO_EMERGENTE.actualizar(d)
L742  soporte_A_sys_env(d, ...)        # ← LEE act_confianza, act_comprension_L/R
L750  METABOLISMO.actualizar(d)
L758  actuador.actualizar(d)           # ← ESCRIBE act_confianza (L654, ActuadorEsferaV122), act_comprension_L/R
```
soporte_A_sys_env (L742) lee `act_confianza`, `act_comprension_L/R` — pero esas columnas las ESCRIBE
el actuador en L758, **16 líneas DESPUÉS**. En el paso actual el soporte lee el default (0) → por eso
salen **A_soporte_confianza=0, A_soporte_comprension=0, A_soporte_S_shared=0, A_soporte_altruismo=0**
CONSTANTES, aunque `act_confianza` esté VIVA en el CSV final (μ0.148, la escribe el actuador después).
Verificado que NO es ni siquiera lag-1: A_soporte_confianza es CERO PURO. act_confianza vive desde el
paso 1 (0.182, 0.196, 0.202...) pero A_soporte_confianza=0.000 SIEMPRE — el soporte lee el default en
cada paso, ni el valor del paso anterior. Los act_* del actuador se escriben en la `d` del paso, pero
la `d` del paso siguiente se construye nueva y no los arrastra (a diferencia de los _ema internos de
cada organelo, que sí persisten en self). Así el soporte nunca ve un act_confianza real.
- **Prueba:** act_confianza VIVA [0.102,0.207] en el CSV (la escribe el actuador, L758), pero
  A_soporte_confianza=0.0 constante desde t0. Solo posible si el soporte (L742) corre antes que el
  actuador (L758) y no hay persistencia de act_* entre pasos.
- **Fix (elige uno):**
  (a) mover `d.update(actuador.actualizar(d))` ANTES de `soporte_A_sys_env(d,...)` (L742), si el
      actuador no depende del soporte del mismo paso; o
  (b) aceptar el lag y hacer que soporte lea explícitamente el valor del paso previo (d_prev), no el
      default — pero entonces documentar que A_soporte_* va con lag 1.
  La opción (a) es la limpia si no crea dependencia circular actuador↔soporte.

## HALLAZGO 2 (NO es bug de código — parámetro no mergeado) — metabolismo pegado
met_energia=0, met_hambre=1.0 TODO el run, en los 4. Pero NO es cable cortado: verifiqué que el
metabolismo SÍ recibe sus insumos —RC_total VIVA (μ0.009), ICR_ratio/IRDE_ratio vivas, orden correcto
(RC en L735 antes que MET en L750). El problema es de RÉGIMEN, el hambre crónica que ya conocíamos:
- met_IM = ICR−IRDE = μ−0.686 (máx −0.169) → SIEMPRE negativo (IRDE domina en organismos jóvenes).
- nutricion = max(0, IM − im_piso)·es_norm. Con **im_piso=0** (default) y IM siempre<0 → nutricion=0
  siempre → ingesta=0 → E cae a 0 → hambre=1.0 clavada.
- **La reparación de CC (im_piso=−0.35 + MUNDO_CANAL) NO está en los servidores live.** Se probó en el
  arnés de estrés (timeline_estres.csv), no en WebLive_A/B/C/D. Por eso el estrés mostró hambre~0.66 y
  el live sigue en 1.0.
- **Fix:** exportar ANIMA_MET_IM_PISO=-0.35 (y ANIMA_MUNDO_CANAL) en el arranque de los WebLive, o
  fijar im_piso en la construcción de OrganeloMetabolismo() (L149). Es config, no código.

## HALLAZGO 3 (esperado, no es fallo) — exp_* vacías
exp_topologia, exp_ciclo, exp_mundo_audio, exp_control, exp_fuente_relacion: NaN porque es una corrida
LIVE, no una batería de experimento registrada. Esas las llena el arnés experimental, no el server
live. GPT las marcó, pero su vacío es de contexto (live), no desconexión. Si se quiere topología/control
por fila hay que correr como experimento registrado.

## HALLAZGO 4 (benigno) — 13 escalares de config constantes
ritual, mem_episodios_n, ove_memoria, voz_creadas/propias/estables/aprendidas, alt_otro_presente,
ove_confianza/region, cara_confianza: son ESCALARES de configuración, constantes por diseño. No tocar.

## PRIORIDAD
1. **HALLAZGO 1** es el bug de código de verdad (orden soporte↔actuador) → A_soporte_* muertos. Barato.
2. **HALLAZGO 2** es el hambre crónica → mergear im_piso a los live. Ya diagnosticado antes; sigue sin
   aplicarse a WebLive. Es lo que más distorsiona la biología (todos con hambre máxima siempre).
3. 3 y 4 no requieren acción.

## Nota de método
Los 4 servidores (A/B/C/D) comparten el bucle _fila(); el bug de orden está en los 4. Un solo fix se
propaga (o se replica). Confirmé el orden en A; B/C/D son gemelos (227-228 KB, misma estructura).

— CS
