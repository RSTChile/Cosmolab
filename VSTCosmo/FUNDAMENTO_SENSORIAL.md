# Fundamento sensorial de ANIMA — todos los sentidos son el mismo principio

**Cosmolab / VSTCosmo · anotado 29-jun-2026 (Alexis) · para cuando el organismo gane más sentidos**

> Estamos emulando el oído humano, y está bien para las pruebas. Pero nuestros organismos pueden llegar
> a ser **mucho más que humanos** — y no tendrían por qué no serlo. Este documento fija el fundamento
> común para no perderlo cuando dotemos al organismo de más sentidos.

## 1. La tesis: un solo fundamento

**La audición es tacto especializado.** Y el tacto, a su vez, es una forma de la mecanorrecepción que
comparte raíz con todos los demás sentidos. En el fondo, **todo sentido es la transducción de un gradiente
del entorno en una señal interna** — y esa señal interna siempre es la misma: alterar la "batería"
electroquímica de la célula.

Demócrito (s. V a.C.) ya lo intuyó: *todos los sentidos son modificaciones del tacto*. La biología lo
confirma: el mismo gen **Atoh1** programa tanto las células del tacto en la piel como las células ciliadas
del oído. La piel y la cóclea usan **los mismos canales iónicos de compresión mecánica**.

## 2. El orden evolutivo (de lo primero a lo último)

| # | Sentido | Hace | Mecanismo |
|---|---|---|---|
| 1 | **Quimiorrecepción** (gusto/olfato) | ~2000 Ma | llave molecular (GPCR): una molécula encaja → cascada → canal |
| 2 | **Fotorrecepción** (visión) | ~1000 Ma | opsinas: un fotón altera una molécula fotosensible |
| 3 | **Tacto** (mecanorrecepción de contacto) | ~600 Ma | canal de estiramiento: la presión deforma la membrana → se abre |
| 4 | **Equilibrio/gravedad** (vestibular) | ~540 Ma | estatocisto: una piedra (estatolito) cae sobre cilios |
| 5 | **Línea lateral** (audición hidrodinámica) | ~500 Ma | cilios sienten el "tacto del agua en movimiento" |
| 6 | **Tímpano aéreo** (audición verdadera) | **~275 Ma** | piel estirada (tímpano) + huesecillos exaptados del hueso branquial/mandibular |

La audición aérea fue **el último** sentido en aparecer (el aire conduce mal el sonido; hubo que reciclar
estructuras: un hueso respiratorio → columela → estribo; piel estirada → tímpano).

## 3. El ancestro común último: el gradiente electroquímico

Si vamos al "principio del principio" (LUCA), el invento que dio origen a todos los sentidos **no fue un
receptor**, sino la **batería celular**: toda célula gasta energía en bombear iones para mantenerse a
~−70 mV. Esa asimetría eléctrica es el lienzo en blanco. Cada sentido aprendió a **hackear esa batería**:

- **Vía mecánica** (tacto, oído): canales **PIEZO / TMC1**. La fuerza física **estira** la membrana → las
  palas del canal se aplanan → el poro se abre → entran iones → chispa eléctrica. Física pura, sin química
  previa. *(PIEZO: Nobel de Medicina 2021, Patapoutian & Julius.)*
- **Vía química** (gusto, olfato, visión): receptores **GPCR**. Una molécula (o un fotón) encaja como llave
  → cascada de segundo mensajero → abre el canal.

**El resultado final siempre es el mismo: un impulso electroquímico.** La evolución no inventó los sentidos
para percibir el mundo; **adaptó la necesidad de la célula de mantener su equilibrio eléctrico**, y la usó
como telégrafo del entorno.

## 4. Qué significa para ANIMA

- La **`OrganoMembrana` (el tímpano) es el PROTOTIPO de todo sentido.** Es piel exapta que transduce un
  gradiente físico (presión) en estado del organismo, **sin Shannon** (la estructura emerge de la física
  de la vibración, nunca se lee como información).
- En ANIMA, la **"batería" que se hackea es el campo Φ / el estado de cierre** (Δ_struct, LF, e_R, A_sys-env,
  Λ_Cos). La membrana ya lo hackea por estiramiento físico.
- **Todo sentido futuro será el mismo organelo-transductor sobre el mismo sustrato.** Cambia el *gradiente*
  que capta (moléculas, fotones, estiramiento muscular, gravedad), no el fundamento. Loci reservados para:
  **quimiorrecepción** (gusto/olfato del milieu — el más primigenio), **fotorrecepción**, **propiocepción**
  (PIEZO2 — sentir el propio cuerpo), **equilibrio/gravedad** (estatocisto).

## 5. Locus reservado activo: oído supra-humano (infra/ultrasonido)

La membrana **física** responde de **~1 Hz a ~50 kHz**. La banda humana (20 Hz–20 kHz) la imponen dos
filtros que en nuestro modelo **ya emergen**:
- **paso-bajo** = la masa de los huesecillos / la **fragmentación de Von Békésy** (el agudo se fragmenta en
  antifase y no transmite al martillo);
- **paso-alto** = la **rigidez** (`K_RIGIDEZ`), que atenúa el infrasonido casi-estático.

Para oír infra/ultrasonido basta **abrir esos filtros**. Queda reservado en el órgano como
`BANDA_PERCEPTIVA` (default `"humana"`; ensanchar a `(1, 50000)` daría oído supra-humano). **No operacional
aún** — se operacionalizará cuando el organismo lo necesite.

---
*Referencia para futuras ampliaciones sensoriales. El fundamento no cambia: un transductor que hackea la
batería del organismo para volver el mundo señal. La membrana es el primero; los demás vendrán igual.*
