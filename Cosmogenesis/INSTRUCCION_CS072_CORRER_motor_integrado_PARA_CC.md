# INSTRUCCIÓN PARA CC — CORRER el motor integrado CS072 (comprobación paralela)

## REGLA ÚNICA E INNEGOCIABLE
**CORRE el modelo. NO lo cambies, NO lo ajustes, NO lo re-codifiques, NO "mejores" nada.**
Esto es una COMPROBACIÓN PARALELA: CS ya desarrolló y probó el motor de punta a rabo. Tu tarea es ejecutarlo en tu
entorno y confirmar (o refutar) que reproduce los mismos números. Si algo no coincide, NO lo arregles: repórtalo a CS
con el número que te salió. Un desacuerdo es un DATO, no un bug que debas tapar. Si dudas de algo antes de correr,
pregunta a CS; no adivines.

## QUÉ ES ESTE MOTOR
Un solo proceso sucesivo (`proceso_sucesivo.py`) que encadena, en una corrida, TODO el arco CS072:
física del Modelo Estándar (bariones, aniquilación por color/sabor sin tasa, freeze-out 7:1, átomos H/He, tiempo)
+ las dos fases de expansión que hacen EMERGER la dimensión + control de materia oscura. Cero azar (todo determinista).

## ARCHIVOS (ya están en Desktop/RMD/Cosmolab/Cosmogenesis/cs072_modulos/ — NO los edites)
Paquete `cs072_modulos/`: `proceso_sucesivo.py` (motor), `nucleo.py`, `catalogo.py`, `estado.py`, `freeze_out.py`,
`pieza_base.py`, y `piezas/` (p02_gravedad, p03_fuerte, p04_em, p08_aniquilacion, p23_fluctuaciones, p24_tiempo).
Sólo depende de numpy y scipy. Ejecuta con `PYTHONPATH=.` desde la carpeta `Cosmogenesis/`.

## TIEMPOS ESPERADOS (para que NO creas que se colgó)
El bloque 1 tarda ~20s (incluye la medida ACOPLADA, que corre la física varias veces). El bloque 2 (banda) ~130s.
El bloque 3 (guardián con acoplada) ~30s. Total ~3-4 min. Es NORMAL. No lo interrumpas antes de ~6 min.
La medida ACOPLADA se pide con `medir_acoplada=True` (es cara); sin ese flag, el motor corre rápido y sólo da la
dim ENSEMBLE. Úsalo tal como aparece abajo.

## SCRIPT DE VERIFICACIÓN (córrelo TAL CUAL — no lo modifiques)
```python
# guarda esto como verificar_cs072.py en Cosmogenesis/ y corre:  PYTHONPATH=. python verificar_cs072.py
from cs072_modulos.proceso_sucesivo import proceso_sucesivo

print("=== 1) MOTOR COMPLETO D=3 (este universo) ===")
r = proceso_sucesivo(nq=300, naq=210, ne=100, npos=70, D_distinciones=3, pasos=150, medir_acoplada=True)
print("  bariones          =", r["bariones"],          " (esperado 100)")
print("  ratio p:n         =", r["ratio_pn_congelado"]," (esperado 7.1)")
print("  hidrogeno         =", r["hidrogeno"],         " (esperado 50)")
print("  helio             =", r["helio"],             " (esperado 25)")
print("  tiempo            =", r["tiempo"]["tiempo_emergente"], " (esperado 75 = H+He)")
print("  dim ACOPLADA      =", r["dimension_acoplada"]["dim_efectiva"], " (esperado ~2.0-2.4, átomos reales)")
print("  dim ENSEMBLE      =", r["dimension"]["dim_efectiva"],          " (esperado ~2.77, ley del régimen)")
print("  oscura_necesaria  =", r["materia_oscura"]["oscura_necesaria"], " (esperado True)")
print("  invariante        =", r["invariancia"]["invariante"],          " (esperado True)")

print("\n=== 2) BARRIDO DE BANDA: dimensión vs nº de distinciones (otros universos posibles) ===")
print("  D | dim_ensemble | invariante   (esperado: crece con D, invariante en las 5)")
for D in [1,2,3,4,5]:
    r = proceso_sucesivo(nq=300, naq=210, ne=100, npos=70, D_distinciones=D, pasos=100)  # sin acoplada = rápido
    print("  %d |    %-5s     | %s" % (D, r["dimension"]["dim_efectiva"], r["invariancia"]["invariante"]))
# esperado aprox: D1=1.0, D2=2.24, D3=2.77, D4=3.33, D5=3.41 ; invariante=True en todas

print("\n=== 3) GUARDIÁN anti-Shannon: apagar una fuerza destruye su estructura ===")
print("  (nombres-clave EXACTOS: 3_fuerte, 4_em, 8_aniquilacion, 2_gravedad, 23_fluctuaciones)")
r_fuerte = proceso_sucesivo(nq=300,naq=210,ne=100,npos=70,D_distinciones=3,pasos=150,apagar=frozenset(["3_fuerte"]))
print("  apagar 3_fuerte    : H=%s He=%s   (esperado He=0: sin fuerza fuerte no hay fusión)" % (r_fuerte["hidrogeno"], r_fuerte["helio"]))
r_em = proceso_sucesivo(nq=300,naq=210,ne=100,npos=70,D_distinciones=3,pasos=150,apagar=frozenset(["4_em"]),medir_acoplada=True)
print("  apagar 4_em        : H=%s He=%s   (esperado H=0: sin EM no se liga protón+electrón)" % (r_em["hidrogeno"], r_em["helio"]))
print("  apagar 4_em dim_acoplada =", r_em["dimension_acoplada"]["dim_efectiva"], " (esperado None: sin átomos, la geometría COLAPSA)")
```

## QUÉ ESPERAR (valores de referencia de CS — confirma o reporta discrepancia)
1. **Modelo Estándar (D=3):** bariones=100, ratio p:n=7.1, H=50, He=25, tiempo=75.
2. **Dimensión:** ACOPLADA ~2.0-2.4 (de los átomos reales de la corrida; baja resolución, verificado 2.05) y ENSEMBLE ~2.77 (ley del
   régimen sobre N grande, como CDT). Que NO sean idénticas es esperado y correcto (miden cosas distintas).
3. **Banda D=1..5:** la dimensión crece con el nº de distinciones (~1.0/2.24/2.77/3.33/3.41), invariante en las 5.
4. **Guardián:** apagar `3_fuerte` -> He=0; apagar `4_em` -> H=0 Y dim_acoplada=None (sin átomos no hay geometría).

## RESERVAS YA CONOCIDAS (NO son bugs — no las "arregles")
- La dim ACOPLADA subestima (~2.05 vs 3) por pocos átomos reales; es su límite de resolución, esperado.
- En D=4/5 el estimador de Hausdorff subestima (3.33/3.41 en vez de 4/5) por tamaño finito. Conocido.
- Apagar `8_aniquilacion` o `2_gravedad` NO cambia el conteo de bariones/átomos a esta escala: es correcto (la
  aniquilación equilibra materia/antimateria que el conteo no expone por separado; la gravedad actúa en la geometría).
- La dim ENSEMBLE NO reacciona a `apagar` (corre su propio ensemble): es POR DISEÑO (mide la ley, no la corrida).
  La que reacciona a apagar es la ACOPLADA — ésa es la prueba del acoplamiento causal.

## CÓMO REPORTAR
Pega la salida COMPLETA de los tres bloques. Para cada valor: coincide / no coincide (y con qué número). Si algo
falla al importar o correr, pega el traceback tal cual. NO toques el código bajo ninguna circunstancia.

_(No cerrar ningún experimento hasta que Alexis diga que terminó — NOTA_PERMANENTE_CS.md.)_
