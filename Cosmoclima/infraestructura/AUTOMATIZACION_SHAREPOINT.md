
# Automatización en SharePoint: qué se puede, qué no, y hasta dónde llegamos

**21-ago-2026.** Todo lo que sigue está **probado en el tenant de RMD**, no sacado
de documentación general. Donde hay una cifra oficial va la referencia.

---

## 1 · Los flujos clásicos de SharePoint ya no existen

Los *workflows* de SharePoint 2010 y 2013 —los del diseñador, con sus «etapas» y
«transiciones»— están **retirados**. No es que estén desaconsejados: dejaron de
ejecutarse. Lo que hoy se llama automatizar en SharePoint es otra cosa, y son
cuatro mecanismos distintos con capacidades muy distintas:

| mecanismo | qué hace | dónde vive |
|---|---|---|
| **Columnas calculadas** | recalculan solas, en cada fila | dentro de la lista |
| **Reglas de lista** | «si cambia X, avisa a Y» | dentro de la lista, sin código |
| **Power Automate** | flujos con condiciones, bucles y varios servicios | fuera, en Power Platform |
| **Webhooks de Graph** | avisan a un servidor propio cuando algo cambia | fuera, servidor propio |

---

## 2 · Qué licencia tenemos, medido

Consultado a Graph: **Microsoft 365 Business Standard** más `FLOW_FREE`. Los
planes que importan son **`FLOW_O365_P1`** y **`POWERAPPS_O365_P1`** — la versión
*incluida* de Power Automate, no una de pago.

Eso fija el techo, y está en la documentación oficial de Microsoft:

| | incluido con Office 365 |
|---|---|
| Acciones por día y por usuario | **6.000** (10.000 durante el período de transición) |
| Conectores estándar (SharePoint, Outlook, Teams, Excel, Forms, Aprobaciones) | **sí** |
| **Conectores premium** (acción HTTP, Azure, SQL) | **no** |
| **Conectores personalizados** | **no** |
| Puerta de enlace a datos locales | no |
| Automatización robótica de escritorio | no |
| AI Builder | no |

**Lo más importante de esa tabla no es lo que falta, sino el 6.000.** Una acción
es cada paso de cada ejecución: leer una fila, escribir una fila, evaluar una
condición. Recalcular las 835 filas de la Matriz consume como mínimo 835
acciones. **Alcanza para unos siete recálculos completos al día, y para ninguno
de una sub-matriz de 16.768 filas.**

---

## 3 · Columnas calculadas: lo probado, con una sorpresa incómoda

Creé listas de prueba en el sitio y las borré después. Esto es lo que pasó:

| fórmula | por Graph | resultado |
|---|---|---|
| `=[IB]+[FVT]` | **aceptada** | calculó bien: 0,9 + 0,76 = 1,66 |
| `=[IB]*2` | **aceptada** | bien |
| `=IF([IB]>0.5,1,0)` | **rechazada** | «la fórmula contiene un error de sintaxis o no se admite» |
| `=SI([FEN]="Alta";3;1)` | rechazada | ídem, no es cosa del idioma |
| `=CONCATENATE([FEN]," / ",[IB])` | rechazada | ídem |
| `=([IB]>0.5)*3` | rechazada | ni siquiera una comparación |

**★ Es un límite de la interfaz de programación, no de SharePoint.** La
documentación de Microsoft muestra `IF` y `AND` funcionando en columnas
calculadas creadas **desde la interfaz web**. O sea: las fórmulas con funciones
hay que escribirlas a mano en SharePoint; Graph sólo acepta aritmética.

### ★★ Y la sorpresa: la agregación no da error, da un número equivocado

Probé `=[IB]/MAX([IB])`, que es exactamente la forma de `Pev`, `Peh` y `Pen`.

**Graph la aceptó sin protestar.** Y el resultado fue **1 para todas las filas**:

```
fila a · IB 0,9 → Calc 1
fila b · IB 0,4 → Calc 1
```

Porque `MAX([IB])` no agrega nada: devuelve el valor de **esa misma fila**. Una
columna calculada de SharePoint sólo ve su propia fila, y cuando le pides mirar
la columna entera **no falla: contesta cualquier cosa**.

Es el mismo patrón que este proyecto viene encontrando en todas partes.
Cualquiera habría escrito `=[Pen]/MAX([Pen])`, habría visto columnas llenas de
números plausibles, y habría publicado una priorización donde todo vale 1.

---

## 4 · Hasta dónde podemos llegar, entonces

### Lo que SÍ se puede, y conviene hacer

**Volver reproducibles el `FVTic` y el `PF`.** Son aritmética pura de la misma
fila, así que pueden ser columnas calculadas y **no volver a desincronizarse
nunca**. Eso ataca de raíz el hallazgo de que el `FVT` publicado no es función
de sus entradas: dejaría de poder no serlo.

**El `IRMD` por umbrales**, con una columna calculada escrita desde la interfaz
(necesita `IF`).

**`Pev`, `Peh` y `Pen` — pero sólo si se abandona el divisor variable.** Y aquí
las dos investigaciones convergen de una forma que vale la pena decir en voz
alta:

> El divisor «máximo observado» es lo que causa la no estacionariedad —un ítem
> nuevo reclasifica hasta la mitad de la matriz— **y es exactamente lo mismo que
> impide que estas tres columnas se calculen solas**, porque ninguna columna
> calculada puede mirar la columna entera.
>
> **Fijar el divisor arregla los dos problemas de un solo golpe.** Y si además
> las entradas se llevan todas a 0-1 con pesos que sumen 1, no hay divisor que
> fijar: el índice ya vive en 0-1 por construcción.

Con eso, **la Matriz entera pasaría a recalcularse sola** en cada edición, sin
consumir una sola acción de Power Automate.

**Avisos.** Una regla de lista —sin código, sin consumir cuota— que avise cuando
alguien cambie el `FEN`, el `IB` o el `VTic` de un ítem. Barato y útil: hoy nadie
se entera de que la Matriz cambió.

**Un flujo de coherencia semanal.** Que revise que ninguna sub-matriz apunte a un
ítem que no existe, y avise. Son unas pocas acciones por semana.

### Lo que NO se puede con esta licencia

**Traer datos de fuera.** ERA5, SERNAGEOMIN, la Dirección General de Aguas, el
Coordinador Eléctrico — todo eso exige la acción HTTP, que es conector premium y
**no está incluido**. Power Automate no puede alimentar la Matriz con dato
externo.

**Recalcular en masa.** Cualquier operación que toque fila por fila un inventario
grande está fuera del presupuesto de 6.000 acciones diarias.

**Reaccionar en tiempo real desde fuera.** Los webhooks de Graph funcionan, pero
necesitan un servidor propio escuchando en una dirección pública. Hoy no lo hay.

---

## 5 · El reparto de trabajo que propongo

No es una limitación que haya que lamentar: es un reparto que además es el
correcto.

| | dónde | por qué |
|---|---|---|
| **Cálculo pesado** — clima, cruces geométricos, FEN medido, recálculo de índices | **acá, en Python** | ya está hecho, no tiene límite de acciones, y es auditable |
| **Cálculo por fila** — FVTic, PF, IRMD, y los tres índices con divisor fijo | **columnas calculadas de SharePoint** | se mantienen solas para siempre |
| **Almacenar, mostrar, compartir** | **SharePoint** | es lo que sabe hacer |
| **Avisar** | **reglas de lista** | gratis y sin cuota |
| **Coordinar** | **un flujo semanal chico** | dentro del presupuesto |

**Lo que un flujo NO debe hacer nunca en este proyecto es calcular.** Si el
cálculo vive en dos lugares, se separan — que es exactamente lo que pasó entre
el Excel y SharePoint con la columna `Pen`, que terminó difiriendo en 680 de 835
filas.

---

## 6 · Si algún día hiciera falta más

Sólo dos caminos, y conviene saber lo que cuestan antes de necesitarlos:

**Power Automate Premium**, por usuario y por mes: sube el límite a 40.000
acciones diarias y habilita la acción HTTP y los conectores personalizados. Con
eso sí se podría traer dato externo y recalcular en masa.

**Un servidor propio** que hable con Graph —que es, literalmente, lo que estamos
haciendo esta noche con el script de subida—. No tiene límite de acciones de
Power Platform, sólo la limitación de tasa de SharePoint, que el script ya
respeta. **Es la opción más barata y la que ya está probada.**

---

## 7 · Nota de método

Las listas de prueba que creé para todo esto **quedaron borradas**. No se tocó
`mic`, ni `120`, ni `centrales`, ni ninguna de las 31 sub-matrices. Las pruebas
se hicieron mientras corría la subida y no la afectaron.

Relacionado: `AUDITORIA_MICR_CONTRASTADA.md` (la no estacionariedad) ·
`FUENTE_SHAREPOINT_RMD.md` · `subir_submatrices_sharepoint.py`
