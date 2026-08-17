# `cs072_modulos/` — el motor CS072 (validado)

**Qué es:** el motor de física determinista que reproduce, en una sola corrida encadenada, el arco
completo desde el plasma de quarks (S>0) hasta la emergencia de átomos H/He, tiempo, espacio y
dimensión. Es el resultado consolidado de la serie de experimentos CS057→CS072 (ver
`REGISTRO_EXPERIMENTOS_CS.md` en la raíz para la traza completa). **Verificado por comprobación
paralela CC↔CS el 18/19-jul-2026, match exacto en todos los valores** (ver
`INSTRUCCION_CS072_CORRER_motor_integrado_PARA_CC.md` y `verificar_cs072_output.txt` en la raíz).

**Regla de la carpeta:** este paquete es código YA VALIDADO. No se edita la física para "arreglar" un
resultado que no gusta — un desacuerdo entre una corrida y lo esperado es un DATO a reportar, no un bug
a tapar (ver `NOTA_PERMANENTE_CS.md`).

## Cómo correrlo
```
PYTHONPATH=. python verificar_cs072.py      # (o venv/bin/python si el `python` del sistema no trae numpy)
```
Desde `proceso_sucesivo.py` se importa `proceso_sucesivo(...)`, la función de entrada de todo el arco.

## Arquitectura (capas, de abajo hacia arriba)

| archivo | qué hace | nivel |
|---|---|---|
| `catalogo.py` | define el catálogo de partículas (quarks/antiquarks/electrones/positrones) y el campo de densidad #23 intrínseco (`densidad_intrinseca`) — heterogeneidad DECLARADA, determinista, NO una coordenada espacial. | catálogo inicial |
| `estado.py` | `Estado`: el contrato compartido que TODAS las piezas leen/escriben. Niveles de ligadura separados (`Bq` quark-quark, `Bnuc` nucleón-nucleón, `Bem` electrón-nucleón, `Bgrav` átomo-átomo) para que ninguna fuerza contamine a otra. | estado global |
| `pieza_base.py` | clase base `Pieza` que heredan todos los módulos de `piezas/` (interruptor on/off, nivel, época). | contrato de piezas |
| `piezas/` | cada fuerza fundamental como módulo aislado, apagable por nombre-clave. Ver `piezas/README.md`. | fuerzas |
| `freeze_out.py` | congelamiento del ratio protón:neutrón (~7.1, emergente, sin tasa cableada). | Modelo Estándar |
| `nucleo.py` | `corre(...)`: EL núcleo que orquesta el cronograma — enfría, llama a las piezas activas en su época, consolida bariones (`_detecta_trios`, por conteo estequiométrico, invariante al orden), cuenta H/He, mide el diámetro de la red de átomos (`_geometria`). | orquestador Modelo Estándar |
| `proceso_sucesivo.py` | capa superior: además del Modelo Estándar, mide la emergencia de DIMENSIÓN (dos medidas: `dimension_acoplada` = la de este universo, fosilizada con el átomo, Nivel 2; `dimension` = ley del régimen de mallado, Nivel 1, **no** es la dimensión de este universo — ver docstrings, re-enmarcado 19-jul), materia oscura necesaria, invariancia a permutación. `proceso_sucesivo(...)` es el punto de entrada único. | orquestador completo |

## Qué está fuera de esta carpeta (a propósito)
- **`p02b_gravedad_general.py`** (en `piezas/`) es un experimento DEPRECADO (Paso A: intentó derivar
  posiciones 3D de la malla causal; dio negativo sólido, z<1 a 750 átomos). Se conserva como registro
  histórico, no como parte del motor validado.
- **El experimento de cierre CS073** (átomo → primera estrella: gravedad general real, expansión, CDM,
  enfriamiento H₂) usa las piezas nuevas `p_*.py` de esta misma carpeta `piezas/`, pero se orquesta
  DESDE FUERA (`cs073_cierre_holistico.py` en la raíz) — no toca nada de lo listado arriba. Ver el README
  de `piezas/` y `ARQUITECTURA_EXPERIMENTO_CS.md` en la raíz.

## Principios de diseño que gobiernan todo el paquete (para no romperlos sin darse cuenta)
1. **Anti-Shannon:** ningún resultado depende del ÍNDICE/orden de las partículas — sólo de magnitudes
   físicas (densidad, masa, carga). Verificado con tests de invariancia a permutación.
2. **Anti-contrabando geométrico** (`p24_tiempo.py`): no se cablean valores numéricos DE NUESTRO
   universo (Bohr en metros, G en SI). Todo es adimensional, medido del propio sistema.
3. **Nada emerge antes de tener con qué:** sin átomos no hay espacio (`_geometria` da `None`), sin
   espacio no hay tiempo (`tiempo_emergente` da 0), sin masa cablear energía nueva está prohibido.
