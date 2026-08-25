# Integración con SENAPRED: dónde enchufa la Matriz

Documento de arquitectura. Fija cómo el instrumento entrega datos al sistema
chileno de emergencias, y —más importante— **qué no hace**.

Instrucciones de Alexis del 15-ago-2026, con los ejemplos que entregó.

---

## 1. El límite duro, primero

> **La Matriz NO emite alertas. Produce insumo para que la autoridad decida.**

La decisión de alertar y la emisión son de SENAPRED, en coordinación con la
Delegación Presidencial correspondiente. El instrumento entrega evidencia; nunca
dispara nada por su cuenta, ni automática ni condicionalmente. Cualquier
desarrollo que se acerque a emisión automática hacia el público queda fuera de
alcance por diseño, no por falta de capacidad.

## 2. La cadena real, tal como opera hoy

Reconstruida de la alerta de la Región de Antofagasta vigente desde el
03-ago-2026 (ejemplo entregado por Alexis):

```
FUENTES TÉCNICAS                    SÍNTESIS                    DIFUSIÓN
─────────────────                   ────────                    ────────
DMC · alertamientos             ┌─────────────────┐       ┌──────────────┐
  Alerta AA130, AA131-3         │ Dirección       │       │ SAE          │
  Aviso A388-5, A404-3,         │ Regional        │       │ Plataforma   │
        A406-5, A407-2       ──▶│ SENAPRED        │──────▶│ IADC · CBS   │
                                │   +             │       │ georreferen- │
SERNAGEOMIN · Minuta            │ Delegación      │       │ ciado, a     │
  Técnica remoción en masa   ──▶│ Presidencial    │       │ celulares    │
  (Alta/Moderada/Baja por zona) │ Regional        │       │ de la zona   │
                                │                 │       └──────────────┘
DIRECTEMAR · Aviso de           │ decide y        │
  Condiciones Met. Especiales──▶│ declara         │
                                └─────────────────┘
        ┌───────────────────────────────┘
        │  ◀── ACÁ ENTRA LA MATRIZ, como una fuente técnica más
        └───────────────────────────────
```

El instrumento se sienta al lado de DMC, SERNAGEOMIN y DIRECTEMAR: una entrada
más a la mesa donde se decide. No después, no encima.

## 3. Lo que aporta que ninguna de las otras fuentes aporta

Y esto es, en una frase, el proyecto entero:

> **Chile ya sabe dónde está la amenaza. Ya sabe qué infraestructura importa.
> Nadie cruza las dos cosas.**

- **SERNAGEOMIN** dice: «posibilidad de aluviones **Alta** en Precordillera
  Occidental, Precordillera Alto Loa, Precordillera Salar y Cordillera;
  **Moderada** en Cordillera de la Costa; **Baja** en Pampa y Litoral».
- **DMC** dice: «viento de 90-110 km/h en Cordillera de Atacama, sábado a lunes».
- **La MICR** dice: «las subestaciones eléctricas son infraestructura crítica,
  `PF = 0,75`, `Pen = Muy Alta`».

Nadie dice: **«y estas tres subestaciones están dentro de la zona de peligro
Alto, una de ellas sin acceso alternativo, y su caída deja sin electricidad a
estas comunas»**. Ese cruce es el producto.

### Hallazgo mayor: el FEN dinámico ya existe, para un peligro

La Minuta Técnica de SERNAGEOMIN publica peligro de remoción en masa en
**tres niveles (Alta / Moderada / Baja), por zona geográfica y con vigencia
temporal**. Es decir: **la misma escala del FEN, pero territorial y viva.**

Dos consecuencias, las dos buenas:

1. **La propuesta del proyecto no es especulativa.** El país ya produce
   operativamente un «FEN dinámico» para un peligro. Lo que falta es (a)
   extenderlo a las demás amenazas y (b) —sobre todo— **cruzarlo con el
   inventario de activos**.
2. **Hay contra qué validar.** La minuta de SERNAGEOMIN es una fuente
   independiente para contrastar nuestro `C_clim`. Si nuestro coeficiente marca
   peligro alto donde SERNAGEOMIN marca bajo, el que está mal es el nuestro.

## 4. El problema de las dos geografías

Hay que resolverlo temprano porque rompe implementaciones:

| | Unidades | Quién las usa |
|---|---|---|
| **Geografía de la amenaza** | Litoral · Cordillera de la Costa · Pampa · Precordillera Occidental · Precordillera Alto Loa · Precordillera Salar · Cordillera | DMC y SERNAGEOMIN declaran **acá** |
| **Geografía administrativa** | Comuna · Provincia · Región · Nacional | SENAPRED alerta **acá**; el COGRID se organiza **acá** |

La amenaza se declara por franja geográfica; la alerta se declara por unidad
administrativa. **La matriz tiene que hablar los dos idiomas y traducir entre
ellos.** Nuestras 39 subestaciones tienen coordenadas, así que se pueden ubicar
en ambas — pero hay que conseguir las dos capas.

Un ejemplo de la mezcla, en el propio título del ejemplo: *«Alerta Temprana
Preventiva para la **provincia** de El Loa y las **comunas** de Antofagasta y
Taltal»* — una sola declaración, dos niveles administrativos a la vez. El
instrumento debe poder emitir así, no forzar un nivel único.

## 5. La escala de salida es la oficial, no una propia

El instrumento **no inventa** una escala. Se expresa en la escala del país:

| Alerta | Criterio oficial (glosario ONEMI 2021) | Qué la dispara en nuestros términos |
|---|---|---|
| **Verde** | Vigilancia permanente. Incluye la **Alerta Temprana Preventiva**: reforzamiento del monitoreo ante amenaza probable | Es el producto natural de una matriz predictiva |
| **Amarilla** | La amenaza crece «y se evalúa que **no podrá ser controlada con los recursos locales habituales**» | Ojo: el criterio oficial es **capacidad de respuesta superada**, o sea resiliencia (`FRC` del MCSGS) — no magnitud del daño |
| **Roja** | Requiere movilizar **todos** los recursos | Tramo alto del `ICSGS` |

Que el criterio de Alerta Amarilla sea «recursos locales superados» y no
«daño grande» confirma por qué el MCSGS es necesario: el país ya razona en
términos de capacidad funcional, igual que el módulo de colapso.

## 6. Lo que la salida tiene que traer

Derivado del formato de las fuentes existentes:

- **Georreferenciación** — el SAE envía aviso georreferenciado por celda (CBS);
  si nuestra salida no es georreferenciada hasta comuna, no sirve de insumo.
- **Vigencia temporal explícita** — inicio y fin, como los avisos DMC
  (`11-08` → `12-08`). Un `FEN` dinámico sin fecha de vigencia no es utilizable.
- **Identificador estable y versionado** — la DMC numera `AA130`, `A388-5`; los
  documentos son vivos y se actualizan. Hay que poder citar una versión.
- **Nivel de severidad en escala oficial** — Alta/Moderada/Baja para peligro
  (como SERNAGEOMIN), Verde/Amarilla/Roja para alerta.
- **Doble adscripción territorial** — zona geográfica **y** unidad
  administrativa, por el problema de la sección 4.
- **Trazabilidad del dato** — de qué fuente salió cada número. Es exigencia del
  propio MACC («dato → regla → justificación → variable afectada») y es lo que
  hace auditable un insumo de decisión pública.

## 7. Pendientes que abre esto

1. Conseguir la **capa de zonas geográficas** de DMC/SERNAGEOMIN (litoral,
   pampa, precordilleras, cordillera) — no la tenemos.
2. Conseguir la **capa de límites comunales** oficial (hallazgo H-13).
3. Ver si la **Minuta Técnica de SERNAGEOMIN** es accesible de forma
   programática y con historia, para usarla como validación independiente.
4. Definir el **formato de entrega**: qué archivo, con qué campos, a quién.
   Preguntar a Alexis si SENAPRED tiene un formato de ingesta ya definido —
   sería mucho mejor calzar con el suyo que proponer uno.
