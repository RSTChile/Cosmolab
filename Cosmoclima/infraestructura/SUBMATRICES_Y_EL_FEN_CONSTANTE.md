# Las sub-matrices, y por qué poblarlas no basta

**20-ago-2026.** Escrito a partir de la observación de Alexis: *«lo que no sé si
está hecho son las sub-matrices… sin eso estamos hablando sólo de una submatriz,
subestaciones eléctricas… mientras no tengamos el resto, puentes por ejemplo,
estamos midiendo mal»*.

**La observación es correcta y el diagnóstico se queda corto.** Sí estábamos
midiendo con un solo tipo. Pero al ir a buscar los otros tipos apareció algo
peor: **con los datos que tenemos, agregar los otros tipos no cambiaría ni un
número.**

---

## 1 · Primero, el estado: ¿están hechas las sub-matrices?

**Parcialmente, y más de lo que parecía.** El inventario de anoche dejó
**77.300 activos georreferenciados** en 44 archivos, más 40.409 de respaldo
(mismo activo por otra fuente) y 2.926 sin geometría. Generado con
`adaptadores/inventario_resumen.py`, no escrito a mano.

Cubre, entre otros, los ítems que a ti te preocupaban: puentes (6.742),
carreteras (14.039 tramos), agua potable rural (2.475), embalses (1.370),
subestaciones (1.226), centrales (1.208), salud (5.272), educación (11.829),
puertos (644), aeropuertos (315), relaves (795).

**Lo que NO está hecho es lo que las vuelve utilizables:** ninguno de esos
activos tiene serie climática. `PelPre` sólo existe para las 39 subestaciones del
piloto y para los 91 puntos ReTeRM. Sin serie climática no hay `CClimP`, y sin
`CClimP` no hay `FENef`.

**Dos huecos menores detectados de paso:** los cuatro archivos de
telecomunicaciones de SUBTEL no existen en disco aunque el adaptador está escrito
(`adaptadores/inventario_telecom_subtel.py`); y
`datos/submatrices_quien_lo_tiene.csv` quedó desactualizado — dice «EN CURSO»
para cosas que anoche se bajaron.

---

## 2 · ★★★ El hallazgo: en el frente prioritario, el FEN es una CONSTANTE

`FENef = FEN₄(tipo) × CClimP(lugar, mes)`. El tipo entra por una sola puerta:
`FEN₄`, la fragilidad que la Matriz le asigna a ese tipo de elemento.

Medido sobre las 835 filas:

| | |
|---|---|
| FEN en toda la Matriz | Alta 167 · Media 592 · Baja 76 |
| **FEN en los 112 ítems de `Pen = Muy Alta`** | **Alta 112 · Media 0 · Baja 0** |

**Los 112 ítems del frente prioritario —el objetivo declarado del sub-proyecto de
sub-matrices— tienen todos el mismo FEN.** Sin una sola excepción.

Y se comprueba uno por uno en los catorce tipos que hoy tienen inventario:

| ítem | elemento | FEN |
|---|---|---|
| 120 | Subestaciones Eléctricas | Alta |
| 618 | Puentes de Carreteras | Alta |
| 616 | Carreteras Principales | Alta |
| 265 | Hospitales Generales | Alta |
| 441 · 442 | Escuelas Primarias y Secundarias | Alta |
| 41 · 42 | Tortas y Tranques de Relave | Alta |
| 46 | Embalses | Alta |
| 17 · 16 · 3 | Planta de Agua Potable · Tuberías · Fuentes naturales | Alta |
| 183 | Torres de Telecomunicaciones | Alta |
| 622 | Puertos Marítimos | Alta |

### Qué significa, en simple

Si mañana tuviéramos los 6.742 puentes con su clima bajado y los cruzáramos con
la fórmula, **un puente y una subestación en la misma comuna y el mismo mes
darían exactamente el mismo número de fragilidad efectiva.** Y un hospital, y una
escuela, y un tranque de relaves. La Matriz dice que los cinco son igual de
frágiles ante un evento natural.

No es que falten datos. Es que **la columna que debía distinguirlos no
distingue**.

### Y hay un segundo hallazgo dentro del primero

El cruce completo FEN × Pen no deja lugar a dudas:

| FEN | → Pen |
|---|---|
| Baja (76) | Media (76 de 76) |
| Media (592) | Alta (592 de 592) |
| Alta (167) | Alta (55) o Muy Alta (112) |

**El FEN determina la prioridad final casi por completo.** No es una entrada
entre varias: es *la* entrada. Lo cual explica por qué el recálculo del 16-ago
daba «Alto» en el 100 % de los casos, y por qué mover el clima no lograba
cambiar el orden — el orden ya estaba decidido antes de que el clima entrara.

Es exactamente el mismo patrón que ya habíamos documentado para `FANC`
(*Fragilidad ante Ataques No Convencionales*), que dice «Alta» en 802 de 835
filas, el 96 %. Ahora sabemos que la columna de al lado tiene el mismo problema
dentro del frente que importa.

---

## 3 · ★★ La salida existe, y es la instrucción que ya diste

*«El documento es un ejercicio en abstracto, nosotros tenemos datos reales…
todo lo que ahora hagamos es con dato real, y eso manda.»* (16-ago-2026)

**El FEN por tipo se puede medir en vez de heredarlo.** Ya tenemos el registro
de fallas reales: `datos/mop_emergencias_viales.csv`, **6.141 emergencias**
del Ministerio de Obras Públicas entre 2014 y 2026, cada una con el elemento
afectado, la causa y la gravedad.

Preguntándole al registro qué fracción de las fallas de cada tipo de elemento
tuvo causa meteorológica o de remoción en masa:

| elemento afectado | fallas | de causa meteo | **fragilidad medida** |
|---|---:|---:|---:|
| Enrocado | 23 | 21 | **91,3 %** |
| Carpeta de rodadura (la calzada) | 2.528 | 1.581 | **62,5 %** |
| Elementos de saneamiento | 175 | 96 | 54,9 % |
| Colector | 32 | 17 | 53,1 % |
| Pavimento de camino de acceso | 145 | 59 | 40,7 % |
| Planta de tratamiento | 28 | 11 | 39,3 % |
| **Puente** | **422** | **130** | **30,8 %** |
| Captación | 149 | 40 | 26,8 % |
| Red matriz | 57 | 8 | **14,0 %** |
| Estación de control superficial | 47 | 4 | **8,5 %** |

**La Matriz dice que todos estos son «Alta». El registro de fallas dice que la
calzada es dos veces más frágil que un puente y siete veces más que una red
matriz.** Eso es un FEN medido, con unidades, reproducible y auditable.

### La analogía

La Matriz es como una lista de pacientes donde a los 112 más graves se les
anotó la misma temperatura. No sirve para decidir a quién atender primero —no
porque falten pacientes, sino porque el termómetro anotó lo mismo para todos.
Poblar más sub-matrices es traer más pacientes. Lo que hace falta es volver a
tomar la temperatura, y para eso ya tenemos el termómetro: 6.141 fallas reales.

---

## 4 · Lo que esto reordena

**Poblar las sub-matrices sigue siendo necesario, pero deja de ser lo urgente.**
La advertencia que el propio `SUBPROYECTO_SUBMATRICES.md` §7 dejó escrita se
cumple con precisión incómoda: *«es muy fácil pasar seis meses poblando y llegar
a fin de año con 200.000 activos ubicados y ninguna respuesta sobre si el método
sirve»*. Anoche pasamos de 20.820 a 77.300 activos. Y hoy sabemos que con los
77.300 la fórmula daría el mismo número para todos ellos en un mismo lugar.

El orden que se desprende:

1. **Medir el FEN por tipo** contra el registro de fallas del MOP. Es dato que ya
   está en disco; no hay que bajar nada. Cubre bien el sector Transporte e
   Hídrico, que son dos de los ocho sectores del frente.
2. **Recién entonces** bajar clima para un segundo tipo —los puentes son el
   candidato natural, y además traen el cauce que cruza cada uno— y repetir la
   validación de `FENef` con dos tipos que de verdad se distingan.
3. **Después** el resto de las sub-matrices.

Sobre el costo del paso 2, ya medido: los 6.742 puentes ocupan **1.851 celdas**
de la malla de 0,1° y **555** de la de 0,25°. O sea que no son 6.742 descargas
sino entre 555 y 1.851 — caro pero acotado, y con la mitad de las celdas
compartidas con otros tipos.

---

## 5 · Lo que NO afirmo

- **La fragilidad medida arriba no es todavía un `FEN`.** Es la fracción de
  fallas de causa meteorológica sobre el total de fallas de ese elemento. Mide
  *qué proporción de lo que le pasa a este elemento se lo hace el clima*, que no
  es lo mismo que *cuán probable es que el clima lo rompa*. Para lo segundo hace
  falta el denominador: cuántos elementos de ese tipo existen y estuvieron
  expuestos. Ese denominador ahora lo tenemos (77.300 activos), pero el cruce no
  está hecho.
- **El registro del MOP sólo cubre infraestructura del MOP.** No dice nada de
  subestaciones, hospitales ni escuelas. Para esos hay otros registros
  (`sec_cortes.csv`, 304.419 filas de cortes eléctricos con comuna y hora) y hay
  que tratarlos por separado.
- **2.903 de las 6.141 emergencias dicen «otra o no dice» en la causa.** El
  47 %. Las fragilidades de arriba están calculadas sobre el total, así que
  arrastran ese hueco. Hay que decidir si el denominador correcto es el total o
  sólo las de causa declarada, y declararlo antes de calcular.

Relacionado: `VALIDACION_CCLIMP.md` · `SUBPROYECTO_SUBMATRICES.md` ·
`INVENTARIO_GEORREFERENCIADO.md` · `ESTUDIO_VECTORES_DE_AMENAZA.md`
