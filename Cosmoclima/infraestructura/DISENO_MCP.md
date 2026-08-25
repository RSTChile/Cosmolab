# MCP del modelo: diseño

Pregunta de Alexis (16-ago-2026): *«¿Podemos hacer un MCP asociado al modelo para
automatizar lo más posible todo?»*

**Sí, y creo que es más importante de lo que parece** — no por automatizar, sino
por quién queda habilitado a usar el modelo.

---

## 1 · Qué problema resuelve de verdad

Hoy el modelo se usa escribiendo Python. Eso significa que lo puede usar quien
programa, y nadie más. Vos no programás; un analista de una Dirección Regional de
SENAPRED tampoco tiene por qué.

Un MCP convierte el modelo en algo a lo que se le **pregunta**:

> «¿Qué activos críticos están hoy en zona de peligro alto en la Región de
> Coquimbo?»
> «¿Qué decía la minuta el 24 de marzo de 2015?»
> «¿Cuáles de las subestaciones no tienen acceso vial alternativo?»

Sin abrir un archivo, sin correr un script. **Ese es el salto**: el modelo deja
de ser un experimento que corre alguien y pasa a ser un instrumento que se
consulta. Automatizar es la consecuencia, no el motivo.

## 2 · Qué expondría

Herramientas de **consulta** — el grueso, y lo que da valor inmediato:

| Herramienta | Qué responde |
|---|---|
| `ubicar_activo(lat, lon)` | comuna, provincia, región y zona geográfica — las dos geografías de una vez |
| `peligro_vigente(territorio)` | nivel de peligro de la minuta hoy, por zona o por comuna |
| `activos_en_peligro(nivel, territorio)` | **el cruce que nadie hace**: qué activos críticos están dentro de zona de peligro alto |
| `historia_peligro(zona, desde, hasta)` | qué decía la fuente en una fecha pasada — sobre el archivo que empezamos a construir |
| `fallas_ocurridas(comuna, desde, hasta)` | qué falló de verdad, según SENAPRED |
| `estado_fuentes()` | qué fuentes están frescas, cuáles vencidas, cuáles caídas |
| `ficha_activo(id)` | todo lo que sabemos de un activo y con qué confianza |

Herramientas de **operación** — para mantener el sistema vivo:

| Herramienta | Qué hace |
|---|---|
| `capturar_minuta()` | fuerza una foto ahora, además de las cuatro programadas |
| `ingerir(fuente)` | corre un adaptador y deja el resultado en el consolidado |
| `validar()` | corre el protocolo completo y devuelve los números crudos |

## 3 · Qué NO expondría, y por qué importa

**Nada que emita, dispare o simule una alerta.** Ni siquiera en modo prueba.

La razón no es técnica. Un MCP es una superficie que otros sistemas —y otros
modelos— pueden invocar. Si existe una herramienta llamada `emitir_alerta`,
tarde o temprano algo la va a llamar sin que nadie lo haya decidido. La decisión
de alertar es de SENAPRED con la Delegación Presidencial, y el instrumento
entrega insumo. Esa frontera se defiende mejor **no construyendo el botón** que
poniéndole una advertencia.

Tampoco expondría datos personales: las consultas devuelven conteos agregados por
comuna o por activo, nunca registros individuales.

Y toda respuesta viaja con su **confianza y su fecha**. Una cifra sin eso, en un
canal automatizado, es peor que no tenerla: aparenta autoridad que no tiene.

## 4 · Cómo se construiría

Un servidor MCP en Python, dentro de la misma carpeta, apoyado en lo que ya
existe: `esquema.py` para los datos, `territorio.py` para ubicar, `normalizar.py`
para comparar, los adaptadores para traer. **El MCP no tendría lógica propia** —
sería la ventana, no el motor. Si tuviera lógica propia, habría dos verdades.

Se conecta a tu Claude Code igual que los que ya usás, y desde ahí le preguntás
en castellano.

## 5 · El orden que recomiendo

**No ahora.** Y quiero explicar por qué, porque es la respuesta menos entretenida.

Un MCP es una interfaz. Una interfaz sobre un modelo que todavía no pasa su
validación distribuye respuestas equivocadas más rápido y a más gente — que es
exactamente lo contrario de lo que buscamos. Anoche el ancla de Copiapó falló: el
modelo hoy confunde *raro* con *peligroso*.

El orden que propongo:

1. **Corregir rareza→peligro** y que el ancla pase.
2. **Ingerir la minuta al consolidado**, para que haya qué consultar.
3. **Entonces sí, el MCP** — que en ese momento se construye en poco tiempo,
   porque todas las piezas ya están y sólo hay que asomarlas.

Dicho esto: si preferís tenerlo antes para poder mirar los datos vos mismo sin
depender de que yo corra scripts, se puede hacer una versión **sólo de consulta y
sólo sobre datos crudos** —ubicar, ver la minuta, ver fallas históricas— que no
depende del modelo y por lo tanto no puede propagar su error. Eso sí lo haría ya.

## 6 · Lo que hay que decidir

- ¿MCP ahora en versión sólo-consulta-de-datos-crudos, o después completo?
- ¿Alcance: sólo para vos, o pensado para que lo use un analista de SENAPRED?
  Cambia bastante el diseño — el segundo caso exige control de acceso, registro
  de consultas y un cuidado mucho mayor con la confianza declarada.
