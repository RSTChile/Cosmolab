# Cosmolab

**Laboratorio experimental de Cosmosemiótica**

[🌐 Sitio oficial](https://cosmosemiotica.cl/) ·
[🧪 Explorar los experimentos](https://cosmosemiotica.cl/experimentos.html)

Cosmolab reúne los instrumentos, simulaciones, prototipos, protocolos, datos e
informes con los que la Cosmosemiótica somete sus hipótesis a contraste. No es una
sola aplicación: cada carpeta corresponde a una pregunta experimental distinta y
conserva tanto los hallazgos como los resultados negativos, las limitaciones y las
fronteras todavía abiertas.

Su hilo conductor es `S > 0`: una diferencia mínima no nula. Los experimentos
preguntan qué estructuras temporales, espaciales, orgánicas y semióticas pueden
emerger cuando esas diferencias persisten y se relacionan.

El programa recorre dominios muy diferentes —campos de información, clima,
geometría emergente, organismos digitales y robótica— con una disciplina común:
lo que se afirma debe producir una diferencia medible y distinguirse de sus
controles.

## Mapa del laboratorio

### Linaje de origen

| Línea | Pregunta experimental | Acceso |
|---|---|---|
| VSTCosmo | Cómo un sistema perturbado puede desarrollar organización, historia y sentido operativo para persistir. | [Abrir carpeta](./VSTCosmo/) |

### Instrumentos

| Experimento | Dominio | Qué pone a prueba | Acceso |
|---|---|---|---|
| EIT-3 Óptico | Campo de información | Emergencia y persistencia de asimetría quiral en un campo bidimensional. | [Abrir](./EIT3-Optico/) |
| Dron cosmosemiótico | Vuelo simulado | Control por condiciones de viabilidad en lugar de seguimiento de un objetivo fijo. | [Abrir](./Dron%20Cosmosemi%C3%B3tico/) |
| Levitrón cosmosemiótico | Magnetismo | Frecuencia mínima y límites de un acoplamiento activo capaz de sostener levitación. | [Abrir](./Levitron/) |
| EIT-3 Térmico | Daisyworld | Regulación planetaria emergente y umbrales que separan conducta estructurada de azar. | [Abrir](./EIT3-Termico/) |
| Cosmoclima | Ecología y clima | Si un ecosistema real regula su medio o solamente logra persistir bajo variación climática. | [Abrir](./Cosmoclima/) |
| Procesador de audio EIT-3 Lite | Audio DSP | Exaptación estructural de la voz mediante su acoplamiento con el contexto ambiental. | [Documentación y uso](./docs/PROCESADOR_AUDIO_EIT3.md) |

### Programa central

| Etapa | Pregunta experimental | Acceso |
|---|---|---|
| Cosmogénesis | Si distancia, dimensión y dirección pueden emerger sin introducir previamente un espacio o un tiempo de fondo. | [Abrir](./Cosmogenesis/) |
| Célula Madre | Cuál es la organización mínima necesaria para sostener un organismo no biológico digital. | [Abrir](./VSTCosmo/C%C3%A9lula_Madre/) |
| ÁNIMA | Si organismos mínimos pueden producir comunicación y convenciones compartidas sin recibir un código semántico predefinido. | [Abrir](./VSTCosmo/C%C3%A9lula_Madre/) |
| Cosmorobot | Si los invariantes ensayados digitalmente se sostienen en un cuerpo físico con ruido, latencia y consecuencias reales. | [Abrir](./Cosmorobot/) |

La síntesis pública, el estado vigente y los informes consolidados están en
**[Experimentos · Cosmosemiótica](https://cosmosemiotica.cl/experimentos.html)**.
El repositorio puede contener material de trabajo posterior o anterior a esa
síntesis; la presencia de código no equivale por sí sola a un resultado validado.

## Qué contiene cada experimento

Según su grado de desarrollo, una carpeta puede incluir:

- formulación de la pregunta y criterios de falsación;
- código del instrumento o simulador;
- protocolos, baterías y controles;
- resultados crudos y figuras;
- informes con hallazgos, resultados negativos y límites declarados;
- material histórico necesario para reconstruir el linaje experimental.

## Cómo recorrer el repositorio

1. Comenzar por el [índice público de experimentos](https://cosmosemiotica.cl/experimentos.html).
2. Entrar a la carpeta de la línea que se quiera examinar.
3. Leer primero su `README`, protocolo o informe más reciente.
4. Revisar código, datos y controles antes de interpretar una afirmación como
   reproducida o verificada.

No existe una instalación única para todo Cosmolab. Cada instrumento declara sus
propias dependencias y forma de ejecución. Como punto de partida:

```bash
git clone https://github.com/RSTChile/Cosmolab.git
cd Cosmolab
```

## Alcance de la evidencia

Cosmolab es un laboratorio en desarrollo. En este repositorio conviven:

- simulaciones y exploraciones conceptuales;
- resultados reproducidos por ejecución de código;
- prototipos sometidos a pruebas físicas;
- hipótesis todavía abiertas o no robustas.

Estas categorías no son intercambiables. Los informes de cada experimento deben
indicar qué fue propuesto, implementado, ejecutado, reproducido o validado
físicamente.

## Enlaces principales

- [Cosmosemiótica — sitio oficial](https://cosmosemiotica.cl/)
- [Experimentos — arquitectura, hipótesis, datos, hallazgos y límites](https://cosmosemiotica.cl/experimentos.html)
