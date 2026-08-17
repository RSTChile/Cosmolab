# Procedencia de `VST_Genoma.py`

Este archivo es una **copia vendorizada, sin modificar**, de:

```
D:\Cosmolab\VSTCosmo\Célula_Madre\genoma\VST_Genoma.py
```

(originario del proyecto ANIMA, workspace Cosmogenesis/VSTCosmo).

## Por qué está copiado y no importado en vivo

CosmoRobot es un proyecto propio y autocontenido (principio de modularidad:
cada cosa en su lugar, cambiable sin arrastrar el resto). El motor `genoma`
es **puro stdlib** (sin numpy, sin audio, sin Docker) y **genérico**: define
`Milieu`, `Organelo`, `Organismo`, `MedidorComplejidad` (Kleiber) y `KAPPA`
(invariantes de viabilidad) — nada de eso es específico de ANIMA. Es
exactamente la "mente cosmosemiótica" mínima, aplicable a cualquier cuerpo
(un organismo digital de escritorio, o un robot NXT).

## Regla de mantenimiento

**No modificar este archivo aquí.** Si el motor necesita un cambio:
1. Cambiarlo en el original (`Célula_Madre/genoma/VST_Genoma.py`).
2. Volver a copiarlo aquí, actualizando la fecha abajo.

Si CosmoRobot necesita algo que el motor genérico no da, la extensión va en
un **organelo propio** dentro de `organelos/`, nunca parcheando este archivo.

Copiado: 2026-07-09.
