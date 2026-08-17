# Auditoría CS — informe de GPT "ANIMA4 organismo por organismo"

**Auditor:** Claude Science · **Fecha:** 3-jul-2026
**Sobre:** ANIMA4_analisis_organismo_por_organismo 2.md (GPT)

## 0. HALLAZGO QUE CAMBIA LA LECTURA: GPT analizó OTRA corrida
No pude verificar las cifras de GPT porque son de una sesión distinta de la que tengo como artefacto:
- GPT reporta duraciones 96.4 / 86.5 / 88.2 / 104.7 s.
- Los CSV guardados aquí (Organismo A/B/C/D, 2026-07-02T03-08) van a ~36 s (359/363/367/362 filas).
- Ninguna cifra coincide (A_sys_env, std, agencia, Δ por cuartos) → no es error de GPT, son datos
  distintos. NO tengo el CSV que GPT procesó, así que sus números específicos quedan SIN verificar.
Si quieres que audite las cifras exactas de GPT, necesito los 4 CSV de ESA corrida (los de ~90-105 s).

## 1. Lo que SÍ puedo evaluar (no depende de tener su CSV): el método y la afirmación central
El informe es descriptivo y honesto en su forma. GPT declara sus propias limitaciones (las columnas
exp_topologia/exp_ciclo/exp_control están vacías → "la topología dinámica no quedó codificada por
fila"). Eso es rigor, bien.

## 2. La cuerda de fondo — la conclusión central está PARCIALMENTE confundida con la ENTRADA
GPT concluye: las diferencias A/B/C/D son "individualidad dinámica emergente, no personalidades
programadas". El problema: su propia tabla "Fuentes por organismo" muestra que **cada organismo oye
COSAS DISTINTAS**. En su corrida:
- A: der = voz de B;  B: izq = otros;  C: izq = Main Mix R;  D: izq = otros...
Cuatro dietas sensoriales distintas. Si cada uno recibe entrada distinta, parte de la divergencia es
**la entrada, no la emergencia**. Eso es exactamente el filtro anti-Shannon: antes de llamar
"individualidad emergente" a una diferencia, hay que descartar que venga de que les dimos comida
distinta. GPT no lo descarta — de hecho su propia tabla lo delata.

Para que "individualidad emergente" se sostenga hace falta el contraste que falta: **misma entrada a
los cuatro** (o al menos pares con entrada idéntica) y ver si AUN ASÍ divergen. Si con oído idéntico
se separan, ES emergente. Si se separan solo cuando la entrada difiere, es eco de la entrada.

## 3. Un regalo del dato que SÍ tengo: la corrida de ~36 s permite ese test
En los CSV que tengo, la configuración de oídos viene en PARES:
- A y C: izq = Main Mix R, der = otros organismos  (config idéntica)
- B y D: izq = otros organismos, der = Main Mix R  (config idéntica, espejo de A/C)
Eso es un cuasi-experimento natural: A vs C reciben la MISMA dieta sensorial. Si A y C divergen entre
sí (y B y D entre sí), esa divergencia NO puede ser la entrada → es el candidato limpio a
individualidad emergente. Ese es el test que propongo correr sobre el dato real, si quieres.

## 4. Veredicto
- Informe de GPT: descriptivo, honesto en sus caveats, ÚTIL como retrato — pero es de otra corrida y
  su conclusión central ("emergente, no programada") NO está demostrada: no separa emergencia de dieta
  sensorial distinta. Es una hipótesis sugerente, no un hallazgo.
- Recomiendo: (a) si importan sus cifras, traer sus 4 CSV; (b) para la pregunta de fondo
  (¿individualidad emergente?), correr el contraste de entrada-igualada — que el dato de ~36 s ya
  permite vía los pares A/C y B/D.

— CS
