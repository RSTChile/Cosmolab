# Auditoria de ponderaciones manuales - Celula Madre E017

Fecha: 2026-06-24

Archivos revisados:

- `/Users/alexis/Desktop/RMD/Cosmolab/VSTCosmo/VST_CelulaMadre_WebLive_A.py`
- `/Volumes/192.168.86.31-1/RSTChile/Cosmolab/VSTCosmo/VST_CelulaMadre_WebLive_B.py`
- `/Users/alexis/Desktop/RMD/Cosmolab/VSTCosmo/VST_RC_A.py`
- `/Volumes/192.168.86.31-1/RSTChile/Cosmolab/VSTCosmo/VST_RC_B.py`
- `/Users/alexis/Desktop/RMD/Cosmolab/VSTCosmo/Célula_Madre_Funcional_001.py`
- `/Users/alexis/Desktop/RMD/Cosmolab/VSTCosmo/VST_Genoma.py`
- `/Users/alexis/Desktop/RMD/Cosmolab/VSTCosmo/VST_Homeostasis.py`
- `/Users/alexis/Desktop/RMD/Cosmolab/VSTCosmo/VST_Bloque05_ConscienciaFuncional.py`
- `/Users/alexis/Desktop/RMD/Cosmolab/VSTCosmo/VST_Bloque07_LibertadFuncional.py`
- `/Users/alexis/Desktop/RMD/Cosmolab/VSTCosmo/VST_Bloque08_DinamicaEvolutiva.py`

## Prioridad alta: afectan sentido/decision

### `VST_RC_A.py` / `VST_RC_B.py`

El organelo RC conserva varias ponderaciones manuales que definen como el ruido contextual se reparte en RC, ICR, IRDE, atencion, comprension y riesgo.

- `rc_rel = 0.62*energia_l + 0.18*sal_l + 0.12*balance + 0.08*novelty_l`
- `rc_ext = 0.64*energia_r + 0.18*sal_r + 0.10*(1-coherencia) + 0.08*novelty_r`
- `rc_total = 0.44*rc_rel + 0.36*rc_ext + 0.20*novelty`
- `delta_salud = 0.22*dOI + 0.18*dH + 0.16*dLambda + 0.12*dA + 0.12*dLF + 0.08*dCm + 0.07*dXE + 0.05*dER`
- Inicializacion sin historia: `salud = 0.25*OI + 0.20*H + 0.18*A + 0.15*LF + 0.12*R2 + 0.10*Cm`
- `at_l/r = 0.50*saliencia + 0.28*RC + 0.12*energia + 0.10*novelty`
- `comp_l/r = at * ICR_ratio * (0.55 + 0.45*peso_canal)`
- `riesgo_l/r = at * IRDE_ratio * (0.55 + 0.45*(1-peso_canal))`
- `confianza_comp = ICR_ratio * (0.35 + 0.65*(comp_l+comp_r))`
- `freno_riesgo = 0.45*IRDE_ratio + 0.55*(riesgo_l+riesgo_r)`

Recomendacion: reemplazar por medias/endogenizacion igual que E017:

- `RC_relacional = energia/saliencia/balance/novelty` normalizados por contribucion relativa, no por pesos fijos.
- `delta_salud = media de signos de mejora` o normalizacion por magnitud observada de cada variable.
- `atencion`, `comprension`, `riesgo` deben emerger por competencia entre bases, no por mezcla fija.

## Prioridad media-alta: actuador/cabeza E017

E017 endogenizo el peso `sentido/riesgo`, pero quedan mezclas manuales.

- Evidencia: `0.70*atencion + 0.30*saliencia`
- Bloqueo IRDE: `0.45*riesgo_total + 0.25*freno_rc + 0.20*conflicto + 0.10*(1-integracion)`
- Confianza base: `0.35*R2 + 0.25*LF + 0.25*H + 0.10*Aenv + 0.05*XE`
- Confianza final: `confianza_base * (0.50 + 0.70*permiso) * (1 - 0.50*bloqueo)`
- Fatiga/freno: `0.35*bloqueo`, `0.9*bloqueo + 0.4*conflicto`

Recomendacion:

- Evidencia: usar `media([atencion, saliencia])` o una competencia endogena entre ambas.
- Bloqueo: usar `base_bloqueo = media([riesgo_total, freno_rc, conflicto_rc, 1-integracion])`.
- Confianza: usar soporte endogeno ya calculado (`soporte_sentido`, `permiso_decisional`, `1-bloqueo`).

## Prioridad media: OI canonico

### `VST_Genoma.py` y `VST_Homeostasis.py`

El OI declara explicitamente pesos orientativos calibrables:

- `w = {'H': 0.25, 'ME': 0.20, 'XE': 0.20, 'LF': 0.35}`
- Penalizacion: `0.15 * IRDE * indicador(LF>=kLF)`
- Niveles: `OI >= 0.7` pleno, `OI >= 0.4` proto.

Recomendacion:

- No cambiar sin decision teorica, porque es parte de la lectura canonica actual.
- Para experimentos anti-Shannon, conviene crear `OI_endogeno` paralelo, calculado como producto/media geometrica de componentes disponibles, y comparar contra `OI` canonico.

## Prioridad media: nucleo funcional audio -> fisiologia

### `Célula_Madre_Funcional_001.py`

Escalas manuales relevantes:

- `e_R = abs(grad) * 30.0`
- `A_sys_env = 1/(1 + 5.0*abs(grad))`
- `orientacion += grad * 10.0 * dt`
- `demanda = 1.0 + energia * 3.0`
- `INR = abs(grad) * 2.0`
- `costo_trabajo = delta_orientacion * 0.5`

Recomendacion:

- Tratar estos numeros como escalas de transduccion, no como teoria final.
- Se pueden reemplazar por normalizaciones respecto a historia local: media/rango movil de `grad`, energia y orientacion.

## Prioridad baja-media: umbrales de organelos

Estos no son ponderaciones de decision fina, pero si son umbrales manuales.

- `VST_Bloque05_ConscienciaFuncional.py`: `self_activo` si `R2 > 0.05`.
- `VST_Bloque07_LibertadFuncional.py`: niveles LF `u1=0.05`, `u2=0.33`, `u3=0.66`.
- `VST_Bloque08_DinamicaEvolutiva.py`: mutacion `th_osc=5.0`, `tasa=0.3`, `escala=0.05`; adaptacion `margen=0.05`; exaptacion `k=0.05`; C_m `th_fallo=5.0`, `tau=10.0`, `umbral_cm=0.3`; activacion latente `umbral=0.1`.
- `VST_Homeostasis.py`: `kp=0.5`, `perturb_escala=0.01`.

Recomendacion:

- Mantener por ahora como constantes operativas, pero documentarlas como calibracion.
- Si pasan a incidir en conclusion teorica, reemplazar por umbrales derivados de distribucion historica del organismo.

## Prioridad baja: visualizacion/UI

Constantes de LED, Three.js, opacidades, escalas de VU y colores no alteran el organismo. No requieren endogenizacion salvo que se usen como datos de experimento.

## Conclusiones

La mayor deuda anti-Shannon esta en `VST_RC_A.py/B.py`, no en E017. E017 ya corrigio la ponderacion sentido/riesgo de la cabeza, pero depende de un RC que todavia calcula ICR/IRDE, atencion, comprension y riesgo con mezclas manuales.

Siguiente paso recomendado:

1. Crear `VST_RC_A_E018_ENDOGENO.py` y `VST_RC_B_E018_ENDOGENO.py`.
2. Reemplazar las mezclas ponderadas por:
   - medias simples,
   - normalizacion por contribucion relativa,
   - competencia `base_sentido/(base_sentido+base_riesgo)`.
3. Mantener columnas antiguas y agregar columnas nuevas para comparacion:
   - `RC_base_relacional`
   - `RC_base_externo`
   - `RC_soporte_conversion`
   - `RC_vulnerabilidad_desviacion`
   - `RC_peso_ICR`
   - `RC_peso_IRDE`

