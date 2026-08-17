# ESPECIFICACIÓN CS — PIEZA #5 (DÉBIL): CAMBIA SABOR, NUNCA COLOR (resuelve el colapso de color que impide átomos)
## CC reportó honestamente: no aparece ni un átomo a ninguna escala porque los 30 quarks colapsan a UN color.
## CS verificó la causa Y la corrección con código (outputs reales abajo). NO es un ajuste de período — es un
## error conceptual de física que hay que corregir de raíz.

## LA CAUSA (CS verificó corriendo la regla actual)
La pieza #5 (línea 270-274) hace: flip=((color+step)%20==0) y luego color_n=(color_n+1)%3 — es decir, ROTA EL
COLOR de los quarks periódicamente. CS simuló esa regla: al PASO 20 los 30 quarks ya colapsaron todos al mismo
color (bincount 0,30,0), y se quedan así hasta el paso 300. Sin 3 colores distintos (R+V+A) NINGÚN barión cierra
— es matemáticamente imposible. Por eso no hay átomos, a ninguna escala de pasos. CC tiene toda la razón.

## EL ERROR CONCEPTUAL (física del Modelo Estándar)
El comentario del propio código dice que la débil "cambia sabor de quarks (color/carga)" — CONFUNDE dos cosas
que en física son DISTINTAS:
  - La fuerza DÉBIL cambia el SABOR (flavor): up↔down (eso ES el decaimiento beta). Sabor = up/down/strange/etc.
  - El COLOR (rojo/verde/azul) es carga de la fuerza FUERTE. La débil NO lo toca. El color sólo lo intercambian
    los GLUONES (fuerza fuerte), y SIEMPRE conservando el balance global de color.
Hacer que la débil rote el color no existe en la naturaleza. Es lo que vacía los 3 colores en 1.

## LA CORRECCIÓN (CS verificó con código que preserva el color y permite bariones)
La pieza #5 debe cambiar SABOR (up↔down), NUNCA color:
  - Cada quark tiene un atributo SABOR (mínimo: up/down) separado del color.
  - La débil cambia el sabor de algunos quarks (beta decay), disparada por una condición FÍSICA (no por índice
    ni por color — el color no tiene nada que ver con la débil). El color queda INTACTO.
  - El COLOR sólo lo mueve la fuerza FUERTE (#3), y cuando lo mueve, CONSERVA el balance (intercambio de color
    entre dos quarks vía gluón, no rotación unilateral). Nunca colapsa a un color.
CS verificó: con la débil cambiando sabor y dejando color intacto, los 3 colores se preservan (10/10/10 tras 300
pasos) → los bariones PUEDEN cerrar. Con la regla vieja (rota color) → colapso a 1 color al paso 20 → cero átomos.

## OJO — no basta "no tocar el color": hay que AÑADIR el atributo SABOR al catálogo (parte del inventario físico
## de cada quark), porque hasta ahora el motor no distinguía sabor. Sabor y color son dos etiquetas físicas
## independientes de cada quark. La débil actúa sobre una (sabor), la fuerte sobre la otra (color).

## GUARDIÁN: la débil NUNCA modifica el color; si el código toca `color` dentro de la pieza #5, está mal.
## Verificación obligatoria tras el arreglo: correr y confirmar que los 3 colores sobreviven hasta el
## confinamiento (bincount con 3 valores >0 al paso donde T_ef<T_CONF), y que aparecen bariones/hidrógeno.

## VERIFICACIONES DE CS (outputs reales)
- Regla vieja (rota color): colores al paso 20 = [0,30,0] → 1 solo color → 0 bariones posibles.
- Regla correcta (cambia sabor, color intacto): colores tras 300 pasos = [10,10,10] → 3 colores → bariones cierran.
— CS 🐝 (verificado con código)
