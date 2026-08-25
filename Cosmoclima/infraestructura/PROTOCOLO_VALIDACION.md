# Protocolo de validación

**Escrito el 15-ago-2026, ANTES de calcular nada.** Ese orden es el punto: si el
protocolo se escribe después de ver los resultados, se termina acomodando la
prueba al resultado sin querer. Queda fechado y versionado para que se note si
alguien lo cambia.

---

## Qué se está probando, exactamente

**No** se está probando que las subestaciones corran peligro. Se está probando
una afirmación metodológica:

> Cruzar amenaza (SERNAGEOMIN, DMC) con activos ubicados (coordenadas) produce
> una separación entre activos que hoy son idénticos, y esa separación
> corresponde a algo real.

Hoy las 39 subestaciones comparten la misma fila de la matriz —`FEN=Alta`,
`PF=0,75`, `Pen=Muy Alta`— desde Arica hasta Punta Arenas, 35 grados de latitud.
Si el consolidado no las separa, no sirve. Si las separa pero la separación no
corresponde a nada real, sirve menos todavía: sería peor que no tener nada,
porque tendría apariencia de rigor.

## La regla que ordena todo

**Un resultado negativo es un resultado.** Si el brazo REAL no se separa de los
nulos, se reporta así y se para. No se ajustan pesos, ni umbrales, ni se cambia
la ventana temporal buscando que dé. Esa disciplina ya evitó tres falsos
positivos en Cosmoclima: la Fase VI descubrió que un efecto de clase se caía al
descontar densidad, y el «92% de consenso» resultó ser concentración en tríos y
no dispersión global. Las dos veces apareció por controlar, no por calcular
mejor.

---

## Prueba 1 · El ancla de verdad terreno (la más importante)

**Copiapó, 24-25 de marzo de 2015.** Aluvión documentado, con daño real a
infraestructura. El dato ya está bajado y verificado: **39,8 mm y 64,3 mm en dos
días**, sobre una climatología anual que ronda los 12 mm.

| | Debe ocurrir | Si no ocurre |
|---|---|---|
| **Encendido** | El consolidado marca peligro alto en Copiapó, marzo 2015 | El método no sirve. Se para y se investiga por qué |
| **Especificidad** | **No** marca peligro alto en Punta Arenas ese mismo mes | El método marca todo, y marcar todo es no marcar nada |

Es la prueba más barata y la más dura: un evento conocido, con respuesta
conocida, y la exigencia de acertar **y** de no dar falso positivo a 2.900 km.

## Prueba 2 · Separación territorial

Correr el consolidado sobre las 39 subestaciones para un mismo mes y ver la
distribución del `FEN_efectivo`.

- **Se espera** que separe: Chungará en el altiplano (4.500 m), Copiapó en el
  desierto de aluviones, Valdivia en la zona lluviosa y Punta Arenas en la
  subpolar no pueden tener el mismo número.
- **Si todas quedan iguales**, la maquinaria no está haciendo nada y da lo mismo
  que la matriz estática.
- **Si separa**, hay que mirar *cómo* separa: la separación tiene que ser
  explicable por la física del lugar, no por un artefacto del método.

## Prueba 3 · Los brazos nulos

El corazón de la validación. Se compara el brazo real contra dos formas de azar
que rompen, cada una, un vínculo distinto:

| Brazo | Qué se baraja | Qué vínculo rompe | Qué prueba |
|---|---|---|---|
| **REAL** | nada | — | la señal, si existe |
| **NULL-1** | las fechas | amenaza ↔ **cuándo** | que no acertamos por acertarle a la estación del año |
| **NULL-2** | los activos entre zonas | amenaza ↔ **dónde** | que no acertamos por acertarle a «el norte es seco» |

Ambos nulos conservan las distribuciones marginales: misma cantidad de peligros
altos, mismas fechas, mismos activos. Lo único que se destruye es el
emparejamiento. Si el real no se separa de los dos, lo que teníamos era la
estructura de fondo, no información.

**Repeticiones:** 1.000 permutaciones por brazo. Se reporta el percentil del
real dentro de la distribución nula, no un «p < 0,05» pelado.

## Prueba 4 · Contraste contra SERNAGEOMIN

La única fuente independiente que ya emite un juicio del mismo tipo que el
nuestro: peligro de remoción en masa en tres niveles, por zona, con vigencia.

- Donde ambos hablan del mismo lugar y la misma fecha, **tienen que coincidir**
  la mayor parte del tiempo.
- Se reporta la matriz de confusión completa, no sólo el porcentaje de acuerdo.
- **Si discrepamos, el equivocado es el nuestro** hasta demostrar lo contrario.
  Ellos tienen geólogos y terreno; nosotros tenemos aritmética.

⚠️ Cuidado con la circularidad: si el adaptador de SERNAGEOMIN alimenta el
consolidado, no se puede después usar SERNAGEOMIN para validarlo. Hay que correr
esta prueba con el consolidado **sin** esa entrada, o declarar que sólo mide
consistencia interna. Se decide al construir el adaptador y queda anotado.

## Prueba 5 · Contra cortes que ocurrieron de verdad

Con la capa de estado real (CGE, SEC, Transporte Informa), cuando esté
disponible y si sus condiciones de uso lo permiten:

- Para cada corte registrado: ¿el consolidado tenía peligro alto en esa comuna
  esos días?
- Para cada peligro alto declarado: ¿hubo corte?
- Se reportan **las dos direcciones del error por separado**. Un instrumento que
  avisa siempre no se equivoca nunca en la primera pregunta y es inútil en la
  segunda. Confundirlas es la forma más común de sobrevender un modelo.

**Bloqueo conocido:** hoy no está verificado que haya historia de cortes ni que
su uso automatizado esté permitido. Si no la hay, esta prueba **no se hace** y
se declara que el instrumento quedó sin validación contra falla real — que es
una limitación grande y hay que decirla, no esconderla.

---

## Cómo se reporta

Una tabla por prueba, con el número crudo. Sin adjetivos, sin «prometedor», sin
«tendencia a». Y las tres frases que siempre tienen que estar:

1. **Qué se probó** y sobre cuántos casos.
2. **Qué habría refutado** la afirmación, dicho antes de mirar.
3. **Qué quedó sin probar**, y por qué.

## Lo que este protocolo NO puede decidir

- **Si el instrumento sirve para operar.** Prueba que el método distingue; que
  sea suficiente para apoyar una decisión de emergencia es un juicio de Alexis y
  de SENAPRED, no de un estadístico.
- **Los umbrales.** Se calculan y se muestran; se adoptan con autorización
  expresa. En este equipo ningún experimento se cierra sin el director.
- **Nada sobre riesgo real de una subestación concreta.** El piloto corre sobre
  reanálisis ERA5, que en la ronda 17 de Cosmoclima demostró exagerar años secos
  hasta 2,4× en la zona de Illapel. Cualquier cifra de riesgo por activo es
  provisional hasta contrastarla con estaciones DMC/DGA.
