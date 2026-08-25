"""
FORMALIZAR LAS VARIABLES DE LA MATRIZ DE INFRAESTRUCTURA CRÍTICA
EN EL EXCEL DEL PROTOCOLO
==================================================================

INSTRUCCIÓN (Alexis, 21-ago-2026)
-----------------------------------
«Hay que formalizar las fórmulas de la Matriz en el excel del protocolo donde
salen todas, en la hoja Variables y Métricas completas, si es que no están.»

QUÉ SE COMPROBÓ ANTES DE ESCRIBIR
-----------------------------------
La hoja tiene 318 variables y métricas, incluido el bloque climático MACLIMA
completo (ítems 304-313). **Ninguna de las columnas propias de la Matriz de
Infraestructura Crítica (MICR) está.** Se verificó sigla por sigla: no existen
FEN, FANC, IB, PF, IRMD, Pev, Peh ni Pen.

★ Sí existe una sigla `VT/FVT`, PERO ES OTRA COSA: «Vulnerabilidad Técnica /
Funcional (Capacidad de Absorción)», del bloque de Presión Demográfica, con
fórmula `1-([DemandaServicios]/[CapacidadServicios])`. Es una colisión de siglas
y se resuelve con la regla del canon —desambiguar con una letra— usando `VTic` y
`FVTic` para las de infraestructura crítica.

DE DÓNDE SALEN LAS FÓRMULAS
-----------------------------
Del Word oficial «Matriz de Infraestructura Crítica - FINAL.docx», leído con
`leer_docx_con_formulas.py` porque son ecuaciones OMML que los extractores
corrientes saltan en silencio.

★ CADA FÓRMULA SE VERIFICÓ CONTRA LAS 835 FILAS REALES de la lista `mic` de
SharePoint antes de escribirla. Lo que se formaliza incluye su tasa de
reproducción medida, para que nadie use una fórmula creyendo que reproduce el
dato cuando no lo hace:

    Pev · Peh · Pen           835 de 835   (100 %)   ✓ reproducen exacto
    PF = IB · FVT             807 de 835   (96,6 %)  ✓ el resto es redondeo
    IRMD por umbrales de PF   606 de 835   (72,6 %)  ⚠ no es función del PF
    FVT = (FEN+FANC+VT·3)/9     3 de 835   (0,4 %)   ✗ asignado, no calculado

★ CORRECCIÓN AL CANON QUE SE INCORPORA: el Word normaliza Pev, Peh y Pen
«dividiendo por el máximo real observado en los datos» y da 1,61 · 1,94 · 1,87.
Esos máximos están desactualizados. Los reales, medidos sobre las 835 filas, son
**1,549 · 1,944 · 1,959**. Con los del Word, Pen reproduce 377 de 835; con los
medidos, 835 de 835. Se escriben los medidos y se anota el cambio.

CÓMO SE ESCRIBE
---------------
Editando el XML del .xlsx directamente, NO con openpyxl. El archivo tiene 46
hojas, comentarios heredados (`comments1.xml` + `vmlDrawing1.vml`) y una tabla
de Excel; un ciclo de lectura y escritura completo con openpyxl puede perder esas
cosas en silencio. Acá se añaden filas al XML de la hoja, se extiende el rango de
`Tabla1` y se deja todo lo demás byte a byte idéntico.

Se respalda el original antes de tocar nada.

USO
---
    ../.venv-esa/bin/python formalizar_micr_en_protocolo.py --revisar
    ../.venv-esa/bin/python formalizar_micr_en_protocolo.py --escribir
"""

import re
import shutil
import sys
import zipfile
from pathlib import Path

# ★ Alexis renombró el archivo el 21-ago-2026 para que el nombre refleje la
# fecha de la versión. El respaldo conserva el nombre viejo a propósito: dice de
# qué versión salió.
PROTOCOLO = Path("/Users/alexis/Desktop/Go en Conflictos/RMD 2.0/"
                 "RMD_2_Protocolo_de_Analisis_21_08_2026.xlsx")
HOJA_XML = "xl/worksheets/sheet4.xml"     # 'Variables y Métricas Completas'
TABLA_XML = "xl/tables/table1.xml"
PRIMERA_FILA_NUEVA = 320                  # la hoja termina en la 319
PRIMER_NUMERO_NUEVO = 319                 # la última fila tiene Número 318

CAT = "MICR"
SUB = "Matriz de Infraestructura Crítica"
FUENTES = ("Word oficial «Matriz de Infraestructura Crítica - FINAL.docx» "
           "(ecuaciones OMML) · lista `mic` de SharePoint RMD, 835 filas "
           "(verificación) · Anexo A.5")

# Las 21 columnas de la hoja, en orden.
COLUMNAS = ["Número", "Sigla", "Nombre", "Tipo", "Categoría",
            "Sub Categorías para Análisis y Gráficos", "Fórmula SharePoint",
            "Descripción Técnica", "Descripción No Técnica", "Tipo de Dato",
            "Dependencias", "Lista Externa", "Fuentes", "Fundamentación Teórica",
            "Metodología y Validez de Fuentes", "Rango de Factores de Ajuste",
            "Instrucciones de Uso y Notas", "Sugerencias de Visualización",
            "Ejemplos Históricos", "Relaciones con Otras Variables",
            "Limitaciones Conocidas"]

FILAS = [
 dict(
  Sigla="FEN", Nombre="Fragilidad ante Eventos Naturales", Tipo="Variable",
  Formula='[FEN_num] = SI([FEN]="Alta";3;SI([FEN]="Media";2;1))',
  Tecnica="Fragilidad estructural del tipo de elemento de infraestructura ante "
          "eventos naturales (sismo, inundación, remoción en masa, erupción). "
          "Entrada ordinal de la MICR, no calculada: se declara por tipo de "
          "elemento. Alimenta FVTic y es el término dominante de Pen (peso 0,5).",
  NoTecnica="Cuán fácil rompe la naturaleza esta cosa. Se anota Alta, Media o "
            "Baja para cada tipo de elemento, no para cada activo.",
  Dato="Escala ordinal de 3 niveles (Alta / Media / Baja)",
  Dependencias="—  (entrada declarada por tipo de elemento)",
  Lista="Lista_MICR",
  Teorica="Vulnerabilidad física de infraestructura en marcos de gestión del "
          "riesgo de desastres (UNDRR). Corresponde al concepto de fragilidad "
          "del glosario de Gestión del Riesgo de Desastres de SENAPRED.",
  Metodologia="Medido sobre las 835 filas de la MICR: Alta 167 · Media 592 · "
              "Baja 76. ★ Ningún elemento declara «Muy Alta»: la escala es de "
              "TRES niveles, aunque SERNAGEOMIN publica cuatro para peligro de "
              "remoción en masa.",
  Rango="Alta=3 · Media=2 · Baja=1 al convertirse a FEN_num. Normalizado 0-1 "
        "por la curva logística del proyecto: Baja 0,119 · Media 0,500 · "
        "Alta 0,881.",
  Instrucciones="Es entrada, no resultado: no se calcula desde otras columnas. "
                "★ Verificado sobre las 835 filas: el FEN PARTICIONA la "
                "prioridad Pen en tres bandas que no se tocan — FEN=Baja da "
                "siempre Pen=Muy Baja (76 de 76); FEN=Media da Baja o Media "
                "(592); FEN=Alta da Alta o Muy Alta (167). Un elemento con "
                "FEN=Media NO puede alcanzar Pen=Alta.",
  Visualizacion="Barras apiladas de FEN por sector. Mapa de calor FEN × Pen.",
  Ejemplos="Ítem 120 Subestaciones Eléctricas: FEN=Alta. Ítem 17 Planta de Agua "
           "Potable: FEN=Alta. Aluvión de Copiapó 2015 y temporal de julio 2026 "
           "como casos de referencia.",
  Relaciones="Entra en FVTic. Término dominante de Pen (0,5). Determina la banda "
             "de Pen. Modulable por CClimP en la propuesta FENef.",
  Limitaciones="★ Es constante dentro del frente prioritario: los 100 ítems con "
               "Pen=Muy Alta tienen los 100 FEN=Alta, así que la columna no "
               "distingue tipos allí donde más importa. Se asignó en abstracto, "
               "sin registro de fallas reales. Medición independiente con las "
               "6.141 emergencias del MOP muestra diferencias grandes entre "
               "elementos que la MICR llama todos «Alta».",
 ),
 dict(
  Sigla="FANC", Nombre="Fragilidad ante Ataques No Convencionales",
  Tipo="Variable",
  Formula='[FANC_num] = SI([FANC]="Alta";3;SI([FANC]="Media";2;1))',
  Tecnica="Fragilidad del tipo de elemento ante sabotaje, ciberataque o acción "
          "deliberada no convencional. Entrada ordinal de la MICR. Alimenta "
          "FVTic, Pev (peso 0,3) y Peh (peso 0,5).",
  NoTecnica="Cuán fácil es que alguien rompa esta cosa a propósito.",
  Dato="Escala ordinal de 3 niveles (Alta / Media / Baja)",
  Dependencias="—  (entrada declarada por tipo de elemento)",
  Lista="Lista_MICR",
  Teorica="Vulnerabilidad ante amenazas deliberadas; marcos de protección de "
          "infraestructura crítica de la Unión Europea y NIST.",
  Metodologia="Medido sobre las 835 filas: Alta 802 · Media 32 · Baja 1.",
  Rango="Alta=3 · Media=2 · Baja=1.",
  Instrucciones="Es entrada, no resultado.",
  Visualizacion="Barras por sector. Contraste FEN vs FANC.",
  Ejemplos="Ítem 129 SCADA: FANC=Alta, VT=0,9. Ítem 133 Reactores Nucleares: "
           "FANC=Alta, IB=0,95.",
  Relaciones="Entra en FVTic, Pev y Peh. Influida por la ponderación CAMO=0,25.",
  Limitaciones="★ Es prácticamente una constante: dice «Alta» en 802 de 835 "
               "filas, el 96 %. Una columna que no varía no ordena nada. El "
               "registro de modos de falla de SENAPRED no permite corregirla: "
               "21 de sus 22 modos no anotan intención.",
 ),
 dict(
  Sigla="IB", Nombre="Importancia Base", Tipo="Variable",
  Formula="[IB]   (entrada declarada, escala 0 a 1)",
  Tecnica="Importancia estratégica del tipo de elemento con independencia de su "
          "vulnerabilidad. Factor multiplicativo de la Ponderación Final y "
          "término dominante de Pev (peso 0,5).",
  NoTecnica="Cuánto importa esta cosa si se pierde, aparte de lo frágil que sea.",
  Dato="Índice continuo 0 a 1",
  Dependencias="—  (entrada declarada por tipo de elemento)",
  Lista="Lista_MICR",
  Teorica="Criticidad de activos en marcos de priorización de riesgo del Banco "
          "Mundial y de la Unión Europea.",
  Metodologia="Declarada por tipo de elemento en la MICR. Reemplazo natural con "
              "dato real: población por comuna (INE) y clientes afectados por "
              "comuna y hora (Superintendencia de Electricidad y Combustibles, "
              "304.419 registros desde 2019).",
  Rango="0 a 1. Valores observados en la MICR entre 0,2 y 0,95.",
  Instrucciones="Es entrada, no resultado. Multiplica al FVTic para dar el PF.",
  Visualizacion="Dispersión IB contra FVTic, con PF como color.",
  Ejemplos="Ítem 133 Reactores Nucleares IB=0,95. Ítem 120 Subestaciones "
           "IB=0,90. Ítem 4 Manantiales Naturales IB=0,70.",
  Relaciones="PF = IB · FVTic. Pev (0,5). Pen (0,3). Ligada a la ponderación "
             "PNT=0,1.",
  Limitaciones="Heredada del ejercicio abstracto: no hay dato real detrás. Es el "
               "próximo reemplazo natural por dato medido.",
 ),
 dict(
  Sigla="VTic", Nombre="Vulnerabilidad Tecnológica (Infraestructura Crítica)",
  Tipo="Variable",
  Formula="[VTic]   (entrada declarada, escala 0 a 1)",
  Tecnica="Grado de dependencia del elemento respecto de sistemas tecnológicos "
          "y de control, y por tanto su exposición a fallas y ataques por esa "
          "vía. Entra en FVTic con peso triple y en Peh con peso 0,3.",
  NoTecnica="Cuánto depende esta cosa de computadores y sistemas de control "
            "para funcionar.",
  Dato="Índice continuo 0 a 1",
  Dependencias="—  (entrada declarada por tipo de elemento)",
  Lista="Lista_MICR",
  Teorica="Dependencia tecnológica y superficie de ataque; ENISA Threat "
          "Landscape; incidentes en sistemas SCADA.",
  Metodologia="Declarada por tipo de elemento. No hay dato real de dependencia "
              "tecnológica por activo en Chile.",
  Rango="0 a 1.",
  Instrucciones="★ DESAMBIGUACIÓN OBLIGATORIA: la sigla `VT/FVT` ya existe en "
                "esta hoja (fila 316, Número 315) y designa otra cosa — "
                "«Vulnerabilidad Técnica / Funcional (Capacidad de Absorción)» "
                "del bloque de Presión Demográfica, con fórmula "
                "1-(DemandaServicios/CapacidadServicios). Por la regla del "
                "canon, la de infraestructura crítica lleva la letra `ic`.",
  Visualizacion="Dispersión VTic contra FANC, con Peh como color.",
  Ejemplos="Ítem 129 SCADA: VTic=0,9. Ítem 120 Subestaciones: VTic=0,8. "
           "Ítem 1 Glaciares: VTic=0,2.",
  Relaciones="Entra en FVTic (peso triple) y en Peh (0,3). No confundir con "
             "VT/FVT (Número 315).",
  Limitaciones="Heredada del ejercicio abstracto, sin dato real por activo.",
 ),
 dict(
  Sigla="FVTic", Nombre="Factor de Vulnerabilidad Total (Infraestructura Crítica)",
  Tipo="Metavariable",
  Formula="([FEN_num]+[FANC_num]+[VTic]*3)/9      ⚠ ver Limitaciones: la columna "
          "publicada NO se reproduce con esta fórmula",
  Tecnica="Índice compuesto de vulnerabilidad del elemento, en escala 0 a 1. El "
          "Word lo escribe como (FEN+FANC+VT)/3 seguido de una normalización "
          "dividiendo por 3, con VT llevada a la escala 1-3; las dos escrituras "
          "equivalen a (FEN_num+FANC_num+VTic·3)/9.",
  NoTecnica="Un solo número que junta lo frágil que es la cosa ante la "
            "naturaleza, ante un ataque, y por depender de tecnología.",
  Dato="Índice continuo 0 a 1",
  Dependencias="FEN_num,FANC_num,VTic",
  Lista="Lista_MICR",
  Teorica="Índices compuestos de vulnerabilidad del Risk Management Framework "
          "del NIST.",
  Metodologia="★ VERIFICADO CONTRA LAS 835 FILAS REALES Y NO CIERRA: la fórmula "
              "reproduce sólo 3 de 835 filas (0,4 %). De 35 combinaciones "
              "distintas de (FEN, FANC, VTic), 22 aparecen con más de un valor "
              "de FVT, afectando a 812 de las 835 filas. Agregando IB al "
              "agrupamiento, 18 de 95 combinaciones siguen siendo ambiguas. "
              "Conclusión medida: la columna NO es función de sus entradas.",
  Rango="0 a 1. Valores observados entre 0,33 y 0,87.",
  Instrucciones="★ USAR CON CUIDADO. La columna publicada de la MICR se asignó "
                "a criterio experto, no se calculó: dos elementos con idénticos "
                "FEN, FANC y VTic tienen FVT distintos. Para reproducir la "
                "matriz publicada hay que LEER la columna, no calcularla. Para "
                "recalcular con dato real, usar esta fórmula y declarar "
                "explícitamente que se está reemplazando la asignación experta.",
  Visualizacion="Histograma de FVTic. Dispersión fórmula contra valor publicado, "
                "que muestra la dispersión de la asignación.",
  Ejemplos="Ítem 133 Reactores Nucleares: FVT=0,87 (el Word calcula 0,933 y "
           "anota que la tabla dice 0,87). Ítems 1, 2 y 3 comparten "
           "(Alta, Alta, 0,2) y traen FVT 0,63; el ítem 4, con las mismas tres "
           "entradas, trae 0,57.",
  Relaciones="Alimenta PF, Pev, Peh y Pen. Influida por CAMO y PNT.",
  Limitaciones="★ No reproducible (0,4 %). Además, correlaciona más con la "
               "IMPORTANCIA (con IB, +0,671) que con la fragilidad natural "
               "(con FEN, +0,337), lo que sugiere que al asignarla pesó cuánto "
               "importa el elemento y no cuán vulnerable es. Hallazgo H-01.",
 ),
 dict(
  Sigla="PF", Nombre="Ponderación Final", Tipo="Métrica",
  Formula="[IB]*[FVTic]",
  Tecnica="Riesgo del elemento como producto de su importancia por su "
          "vulnerabilidad compuesta. Base de la clasificación IRMD.",
  NoTecnica="Cuánto importa multiplicado por cuán frágil es. Un número alto "
            "significa que la cosa es a la vez importante y rompible.",
  Dato="Índice continuo 0 a 1",
  Dependencias="IB,FVTic",
  Lista="Lista_MICR",
  Teorica="Priorización de riesgo del Banco Mundial; evaluación de riesgo de "
          "infraestructura crítica de la Unión Europea.",
  Metodologia="★ VERIFICADO CONTRA LAS 835 FILAS: reproduce 807 de 835 "
              "(96,6 %) con tolerancia de 0,005. Las 28 que no cierran son "
              "redondeo a dos decimales, con error máximo 0,0265 (ítem 45: "
              "0,8265 publicado como 0,80).",
  Rango="0 a 1. Valores observados entre 0,13 y 0,80.",
  Instrucciones="Calcular con el FVTic PUBLICADO, no con el recalculado, si se "
                "quiere reproducir la matriz. Redondear a dos decimales.",
  Visualizacion="Histograma de PF con los cortes del IRMD marcados.",
  Ejemplos="Ítem 133 Reactores Nucleares: PF = 0,95 · 0,87 = 0,8265 ≈ 0,83. "
           "Estructuras de Presas: PF = 0,80.",
  Relaciones="PF = IB · FVTic. Determina el IRMD. Contribuye a PNT (0,1).",
  Limitaciones="Hereda entero el problema del FVTic: si el FVTic se asignó a "
               "criterio, el PF también.",
 ),
 dict(
  Sigla="IRMD", Nombre="Impacto en el RMD", Tipo="Métrica",
  Formula='SI([PF]>0,5;"Alto";SI([PF]>=0,3;"Medio";"Bajo"))      ⚠ ver '
          "Limitaciones: reproduce 606 de 835",
  Tecnica="Clasificación cualitativa del impacto que tendría la pérdida del "
          "elemento sobre la dinámica de conflicto del modelo RMD, a partir de "
          "los umbrales del PF definidos en el Word.",
  NoTecnica="Si esta cosa falla, ¿cuánto se complica todo lo demás? Alto, "
            "Medio o Bajo.",
  Dato="Escala ordinal de 3 niveles (Alto / Medio / Bajo)",
  Dependencias="PF",
  Lista="Lista_MICR",
  Teorica="Conflict and Fragility Framework de la OCDE; Global Conflict Risk "
          "Index del Centro Común de Investigación de la Comisión Europea.",
  Metodologia="★ VERIFICADO CONTRA LAS 835 FILAS Y NO CIERRA: los umbrales "
              "escritos reproducen 606 de 835 (72,6 %). El corte entre Medio y "
              "Bajo sí es limpio (mínimo de «Medio» 0,28 sobre máximo de «Bajo» "
              "0,27). El corte entre Alto y Medio NO existe: 26 filas "
              "etiquetadas «Medio» tienen PF por encima del PF más bajo de las "
              "etiquetadas «Alto» (0,38), y el tramo 0,3-0,5 contiene 225 "
              "«Alto» y 78 «Medio» a la vez.",
  Rango="Alto: PF > 0,5 · Medio: 0,3 ≤ PF ≤ 0,5 · Bajo: PF < 0,3. Reparto real "
        "publicado: Alto 741 · Medio 82 · Bajo 12.",
  Instrucciones="★ Para reproducir la matriz publicada hay que LEER la columna. "
                "Los umbrales sirven para clasificar elementos NUEVOS, "
                "declarando que se usa la regla escrita y no la asignación "
                "histórica.",
  Visualizacion="Histograma de PF coloreado por IRMD, que hace visible el "
                "solapamiento entre Alto y Medio.",
  Ejemplos="Estructuras de Presas: PF=0,80 > 0,5 ⇒ Alto.",
  Relaciones="Se calcula desde PF. Se relaciona con ITMC y con la ponderación "
             "PNT (0,1).",
  Limitaciones="★ No es función del PF en la matriz publicada: hay solapamiento "
               "entre Alto y Medio en 26 filas. Además está saturada — 741 de "
               "835 dicen «Alto», el 89 % — así que casi no ordena.",
 ),
 dict(
  Sigla="Pev", Nombre="Prioridad Estratégica para Conflicto Vertical y Regular",
  Tipo="Métrica",
  Formula="(0,5*[IB]+0,3*[FANC_num]+0,2*[FVTic])/1,549",
  Tecnica="Prioridad de proteger el elemento en un conflicto armado "
          "convencional entre estados, donde pesan la importancia estratégica y "
          "la exposición a ataque directo. Se normaliza dividiendo por el máximo "
          "observado y se clasifica en cinco niveles.",
  NoTecnica="En una guerra clásica entre países, ¿qué hay que proteger primero?",
  Dato="Escala ordinal de 5 niveles, derivada de un índice continuo 0 a 1",
  Dependencias="IB,FANC_num,FVTic",
  Lista="Lista_MICR",
  Teorica="Priorización de objetivos y protección de infraestructura crítica en "
          "conflicto convencional.",
  Metodologia="★ VERIFICADO CONTRA LAS 835 FILAS: reproduce 835 de 835 "
              "(100 %) usando como divisor el máximo observado, 1,549. Con el "
              "máximo que trae el Word (1,61) reproduce 678 de 835 (81,2 %).",
  Rango="Muy Alta: >0,96 · Alta: 0,85 a 0,96 · Media: 0,70 a 0,85 · Baja: 0,50 "
        "a 0,70 · Muy Baja: ≤0,50. ★ Divisor de normalización 1,549 (máximo "
        "observado en las 835 filas), NO 1,61 como dice el Word.",
  Instrucciones="El divisor es el máximo observado en los datos, tal como el "
                "propio Word instruye. Al ampliar o corregir la matriz hay que "
                "RECALCULAR el divisor y declararlo como recálculo.",
  Visualizacion="Barras apiladas de Pev por sector. Comparación Pev / Peh / Pen "
                "del mismo elemento.",
  Ejemplos="Ítem 133 Reactores Nucleares: IB=0,95, FANC=Alta(3), FVT=0,87 ⇒ "
           "0,5·0,95+0,3·3+0,2·0,87 = 1,649.",
  Relaciones="Comparte entradas con Peh y Pen. Ligada a CAMO (0,25) y PNT (0,1).",
  Limitaciones="Hereda el problema del FVTic. Al depender de FANC —que dice "
               "«Alta» en el 96 % de las filas— gran parte de su varianza viene "
               "en realidad de IB.",
 ),
 dict(
  Sigla="Peh",
  Nombre="Prioridad Estratégica para Conflicto Horizontal e Irregular",
  Tipo="Métrica",
  Formula="(0,5*[FANC_num]+0,3*[VTic]+0,2*[FVTic])/1,944",
  Tecnica="Prioridad de proteger el elemento frente a conflicto asimétrico: "
          "sabotaje, ciberataque, acción de actores no estatales. Pesa la "
          "fragilidad ante ataque no convencional y la dependencia tecnológica.",
  NoTecnica="Frente a sabotajes o ataques informáticos, ¿qué hay que proteger "
            "primero?",
  Dato="Escala ordinal de 5 niveles, derivada de un índice continuo 0 a 1",
  Dependencias="FANC_num,VTic,FVTic",
  Lista="Lista_MICR",
  Teorica="ENISA Threat Landscape; Global Terrorism Database; análisis de "
          "incidentes cibernéticos del CSIS.",
  Metodologia="★ VERIFICADO CONTRA LAS 835 FILAS: reproduce 835 de 835 "
              "(100 %) con divisor 1,944 (máximo observado). Con el 1,94 del "
              "Word reproduce 833 de 835.",
  Rango="Muy Alta: >0,96 · Alta: 0,85 a 0,96 · Media: 0,70 a 0,85 · Baja: 0,50 "
        "a 0,70 · Muy Baja: ≤0,50. Divisor 1,944.",
  Instrucciones="Recalcular el divisor si se amplía la matriz, y declararlo.",
  Visualizacion="Barras por sector. Dispersión Peh contra VTic.",
  Ejemplos="Ítem 129 SCADA: FANC=Alta(3), VTic=0,9, FVT=0,8 ⇒ "
           "0,5·3+0,3·0,9+0,2·0,8 = 1,93; normalizado 0,9948 ⇒ Muy Alta.",
  Relaciones="Comparte entradas con Pev. Fuertemente ligada a CAMO (0,25).",
  Limitaciones="Con FANC constante en el 96 % de las filas, casi toda la "
               "variación de Peh la aporta VTic.",
 ),
 dict(
  Sigla="Pen", Nombre="Prioridad Estratégica para Desastres Naturales",
  Tipo="Métrica",
  Formula="(0,5*[FEN_num]+0,3*[IB]+0,2*[FVTic])/1,959",
  Tecnica="Prioridad de proteger el elemento frente a desastres naturales y de "
          "recuperarlo después. Es la columna que ordena el trabajo del proyecto "
          "de Infraestructura Crítica × Clima y el insumo natural para el COGRID "
          "y SENAPRED.",
  NoTecnica="Frente a un terremoto, un aluvión o un temporal, ¿qué hay que "
            "proteger y reponer primero?",
  Dato="Escala ordinal de 5 niveles, derivada de un índice continuo 0 a 1",
  Dependencias="FEN_num,IB,FVTic",
  Lista="Lista_MICR",
  Teorica="Priorización de activos para resiliencia post-desastre; marco de "
          "gestión del riesgo de desastres de la Ley 21.364 y del glosario de "
          "SENAPRED.",
  Metodologia="★ VERIFICADO CONTRA LAS 835 FILAS: reproduce 835 de 835 "
              "(100 %) con divisor 1,959 (máximo observado). ★ Con el 1,87 que "
              "trae el Word reproduce sólo 377 de 835 (45,1 %): el máximo del "
              "Word está desactualizado y hay que usar el medido.",
  Rango="Muy Alta: >0,96 · Alta: 0,85 a 0,96 · Media: 0,70 a 0,85 · Baja: 0,50 "
        "a 0,70 · Muy Baja: ≤0,50. ★ Divisor 1,959, NO 1,87. Reparto real: "
        "Muy Alta 100 · Alta 67 · Media 148 · Baja 444 · Muy Baja 76.",
  Instrucciones="★ El divisor 1,959 es el máximo observado y hay que "
                "RECALCULARLO cada vez que cambie cualquier entrada de la "
                "matriz, declarándolo como recálculo. ⚠ Existe una versión del "
                "Excel de la matriz con un Pen desactualizado que difiere en "
                "680 de las 835 filas; la lista `mic` de SharePoint es la que "
                "reproduce la fórmula y es la que manda.",
  Visualizacion="Mapa de calor FEN × Pen, que hace visible la partición en "
                "bandas. Ranking de los 100 elementos de Pen=Muy Alta.",
  Ejemplos="Ítem 17 Planta de Agua Potable: FEN=Alta(3), IB=0,9, FVT=0,76 ⇒ "
           "0,5·3+0,3·0,9+0,2·0,76 = 1,922; normalizado 0,981 ⇒ Muy Alta.",
  Relaciones="Comparte entradas con Pev. ★ Queda PARTICIONADA por el FEN: "
             "FEN=Baja ⇒ Pen=Muy Baja siempre; FEN=Media ⇒ Baja o Media; "
             "FEN=Alta ⇒ Alta o Muy Alta.",
  Limitaciones="★ El FEN pesa 0,5 sobre una escala 1-3 mientras IB y FVTic "
               "pesan 0,3 y 0,2 sobre escalas 0-1, así que el término del FEN "
               "vale entre 1,5 y 3 veces más que los otros dos juntos: el FEN "
               "no influye en la prioridad, la determina. Y el FEN es constante "
               "(Alta) en los 100 elementos del frente prioritario. Corolario: "
               "mientras el FEN no se mida con dato real, esta columna no puede "
               "distinguir entre los elementos que más importan.",
 ),
]


def escapar(t):
    return (str(t).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def construir_filas():
    """Las filas nuevas, en XML, con cadenas en línea para no tocar sharedStrings."""
    piezas = []
    for i, f in enumerate(FILAS):
        fila = PRIMERA_FILA_NUEVA + i
        numero = PRIMER_NUMERO_NUEVO + i
        valores = [None, f["Sigla"], f["Nombre"], f["Tipo"], CAT, SUB,
                   f["Formula"], f["Tecnica"], f["NoTecnica"], f["Dato"],
                   f["Dependencias"], f["Lista"], FUENTES, f["Teorica"],
                   f["Metodologia"], f["Rango"], f["Instrucciones"],
                   f["Visualizacion"], f["Ejemplos"], f["Relaciones"],
                   f["Limitaciones"]]
        celdas = [f'<c r="A{fila}" s="14"><v>{numero}</v></c>']
        for j, v in enumerate(valores[1:], start=1):
            col = chr(ord("A") + j)
            celdas.append(f'<c r="{col}{fila}" t="inlineStr">'
                          f'<is><t xml:space="preserve">{escapar(v)}</t></is></c>')
        piezas.append(f'<row r="{fila}" spans="1:21" '
                      f'x14ac:dyDescent="0.2">{"".join(celdas)}</row>')
    return "".join(piezas)


def main():
    if not PROTOCOLO.exists():
        print("NO ENCONTRADO:", PROTOCOLO)
        return 1
    ultima = PRIMERA_FILA_NUEVA + len(FILAS) - 1

    print("=" * 78)
    print("FORMALIZAR LAS VARIABLES DE LA MICR EN EL PROTOCOLO")
    print("=" * 78)
    print(f"\n  archivo : {PROTOCOLO.name}")
    print(f"  hoja    : Variables y Métricas Completas ({HOJA_XML})")
    print(f"  añade   : {len(FILAS)} filas · {PRIMERA_FILA_NUEVA}-{ultima} "
          f"· Números {PRIMER_NUMERO_NUEVO}-{PRIMER_NUMERO_NUEVO+len(FILAS)-1}\n")
    for i, f in enumerate(FILAS):
        print(f"      {PRIMER_NUMERO_NUEVO+i}  {f['Sigla']:6s} {f['Tipo']:13s} "
              f"{f['Nombre'][:52]}")

    with zipfile.ZipFile(PROTOCOLO) as z:
        hoja = z.read(HOJA_XML).decode("utf8")
        tabla = z.read(TABLA_XML).decode("utf8")
    if f'<row r="{PRIMERA_FILA_NUEVA}"' in hoja:
        print(f"\n  ✗ La fila {PRIMERA_FILA_NUEVA} YA EXISTE. No se escribe nada "
              f"para no duplicar.")
        return 1

    if "--escribir" not in sys.argv:
        print("\n  (revisión: no se escribió nada · usar --escribir)")
        return 0

    respaldo = PROTOCOLO.with_name(
        "RMD_2_Protocolo_de_Analisis_18_06_2026"
        ".RESPALDO_pre_MICR_2026-08-21.xlsx")
    if not respaldo.exists():
        shutil.copy2(PROTOCOLO, respaldo)
    print(f"\n  respaldo: {respaldo.name}")

    hoja_nueva = hoja.replace("</sheetData>", construir_filas() + "</sheetData>")
    hoja_nueva = re.sub(r'<dimension ref="A1:U\d+"/>',
                        f'<dimension ref="A1:U{ultima}"/>', hoja_nueva)
    tabla_nueva = re.sub(r'ref="A1:U\d+"', f'ref="A1:U{ultima}"', tabla, count=1)

    destino = PROTOCOLO.with_name(PROTOCOLO.name + ".tmp")
    with zipfile.ZipFile(PROTOCOLO) as z_in, \
            zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as z_out:
        for item in z_in.infolist():
            datos = z_in.read(item.filename)
            if item.filename == HOJA_XML:
                datos = hoja_nueva.encode("utf8")
            elif item.filename == TABLA_XML:
                datos = tabla_nueva.encode("utf8")
            z_out.writestr(item, datos)
    destino.replace(PROTOCOLO)

    print(f"  escrito : {len(FILAS)} filas · dimensión y Tabla1 extendidas a "
          f"A1:U{ultima}")
    print("  ★ Todas las demás hojas, los comentarios heredados y el dibujo VML "
          "quedaron byte a byte idénticos.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
