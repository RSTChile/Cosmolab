# Sub-proyecto: poblar las sub-matrices

Instrucción de Alexis (16-ago-2026): generar y poblar las sub-matrices de cada
ítem de la Matriz principal, como sub-proyecto aparte de captura de datos.
Él mismo lo dimensionó: «este proyecto es enorme, más de 800 ítems».

**No son 835.** Medido, el trabajo real es bastante más chico — y una parte ya
está hecha por otros.

---

## 1 · El triage: no todos los ítems admiten una sub-matriz

De los 835 ítems, **659 son activos enumerables** (cosas que se pueden listar
con nombre y coordenada) y **176 no lo son**.

Los 176 no son un olvido de la Matriz: son *propiedades*, no *cosas*.
«Infraestructura Vulnerable a Ransomware», «Personal de Operaciones», «Sistemas
de Monitoreo (SCADA)» describen una condición o una capacidad, no un activo que
esté en algún lugar. **Pedirles una sub-matriz georreferenciada sería un error de
categoría** — no existe la lista de «los ransomware de Chile».

Esos 176 necesitan otro tratamiento: son atributos que califican a los activos
enumerables, no filas propias. Conviene decidirlo antes de empezar, porque si no
alguien va a perder semanas buscando un catastro que no puede existir.

## 2 · El objetivo real son 112 ítems, no 835

Si el propósito es servir al COGRID frente a desastres naturales, el orden lo da
`Pen`. Y de los 659 enumerables, **112 tienen `Pen = Muy Alta`**.

Ese es el frente. Los 547 restantes no se abandonan: se hacen después, o nunca,
según lo que muestren los primeros.

Por sector, los 112 se concentran así: Energía 14 · Represas 11 · Hídrico 11 ·
Nuclear 9 · Comercial 9 · Transporte 9 · Químico 8 · Industria de Defensa 6.

## 3 · Y de esos, varios ya están poblados — por otros

Esto es lo que cambia el tamaño del problema. El catastro de anoche encontró
inventarios públicos que **ya contienen decenas de miles de activos reales**:

| Ítem de la Matriz | Fuente pública | Activos | Estado |
|---|---|---|---|
| Carreteras principales y secundarias | MOP · Red Vial | **14.039 tramos** con rol y km | verificado |
| Puentes de carreteras | MOP · Puentes | **6.742** — con el río que cruza cada uno | verificado |
| Agua potable rural (APR) | MOP · Servicios Sanitarios Rurales | **2.293 sistemas** con comuna y beneficiarios | verificado |
| Depósitos y tranques de relaves | SERNAGEOMIN · CDR 2025 | **839** con volumen y estado | verificado |
| Subestaciones eléctricas | Coordinador Eléctrico | **1.269** (sin coordenadas) | verificado |

Son **~25.000 activos reales** cubriendo cinco ítems, sin capturar nada a mano.
El trabajo ahí no es *levantar* datos: es *ingerir y normalizar* — que es
exactamente lo que ya sabemos hacer.

**La regla del sub-proyecto, entonces:** antes de capturar un solo dato a mano,
buscar quién ya lo tiene. Chile tiene mucho más catastro público del que parece.

## 4 · Una sola forma para todas las sub-matrices

No se construyen 835 planillas distintas. Se construye **un esquema** y cada ítem
lo llena. La sub-matriz de subestaciones que ya tenemos sirve de molde:

**Campos comunes, obligatorios para todos:**

| Campo | Por qué |
|---|---|
| `id_activo` | identificador estable, para poder seguirlo en el tiempo |
| `item_micr` | el número del ítem de la Matriz principal (1-835) |
| `nombre` | como lo llama el operador |
| `lat` / `lon` | sin coordenada no hay cruce con amenaza posible |
| `comuna` · `provincia` · `region` · `cut` | derivados de la coordenada, no escritos a mano (ver H-16) |
| `zona_geografica` | la otra geografía, la de la amenaza |
| `operador` · `contacto` | quién responde cuando falla |
| `fuente` · `fecha_captura` | de dónde salió y cuándo |
| `confianza_ubicacion` | no es lo mismo una coordenada GPS que una deducida de una dirección |

**Campos propios de cada ítem:** los que tengan sentido para ese tipo de activo
(tensión de una subestación, luz de un puente, volumen de un tranque). Van en un
campo flexible, no en columnas fijas, porque si no el esquema se vuelve
inmanejable con 659 tipos.

## 5 · El cuello de botella real: georreferenciar

No es conseguir las listas. Es ubicarlas.

El caso testigo es el Coordinador: **tiene las 1.269 subestaciones y ninguna trae
coordenada**. Sin coordenada, un activo no se puede cruzar con ninguna amenaza —
y el cruce es todo el proyecto.

Vías posibles, en orden de calidad:
1. Que el operador entregue las coordenadas (pedirlas formalmente).
2. Cruzar por nombre contra capas que sí las tengan.
3. Geocodificar desde la dirección — **con `confianza_ubicacion` baja declarada**,
   porque una dirección aproximada puede caer en la comuna equivocada, y eso
   contamina justamente el nivel donde trabaja el COGRID comunal.

Nunca por deducción del nombre. «Chungará», «Collahuasi» y «Maitencillo» son
localidades y faenas, no comunas — ya se comprobó.

## 6 · Cómo yo lo haría

**Fase A · Inventario del inventario** (rápida). Para cada uno de los 112 ítems
prioritarios, una línea: ¿existe catastro público? ¿trae coordenadas? ¿de quién
es? El resultado dice cuánto trabajo real hay, y probablemente muestre que un
tercio ya está resuelto.

**Fase B · Ingerir lo que existe.** Los cinco inventarios de la tabla de arriba,
normalizados al esquema común. Es el 90% del volumen con el 10% del esfuerzo.

**Fase C · Pedir lo que falta.** Cartas a operadores. Trámite tuyo, no del
proyecto — pero el proyecto puede dejar redactado exactamente qué pedir.

**Fase D · Capturar a mano lo que quede**, sólo para ítems de `Pen = Muy Alta`
sin fuente. Y sólo entonces se sabrá si son diez o cien.

## 7 · La advertencia que conviene tener escrita

Este sub-proyecto es el que puede hundir al proyecto entero, y no por difícil
sino por **tentador**. Poblar catastros es trabajo visible, medible y agradable:
se ve la barra avanzar. Es muy fácil pasar seis meses poblando y llegar a fin de
año con 200.000 activos ubicados y **ninguna respuesta** sobre si el método
sirve.

Recordar que anoche el ancla de Copiapó falló. Mientras eso no se resuelva, cada
activo nuevo es un activo más al que el modelo le va a decir algo equivocado.

**Recomendación: Fase A y B sí, en paralelo. Fase D recién después de que el
método pase su validación.** Poblar es barato de postergar; equivocarse en el
método y no enterarse, no.
