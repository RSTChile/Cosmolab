# Cosmolab - actualización segura de GitHub

Estas reglas se aplican cuando el usuario pida actualizar, sincronizar o publicar
Cosmolab en GitHub. La frase breve esperada es:

> Actualiza Cosmolab en GitHub siguiendo la instrucción general del repo.

El objetivo es publicar los desarrollos recientes mediante una rama y un pull
request borrador, con evidencia verificable y sin intervenir experimentos activos.

## Repositorio y destino

- Trabajar desde la raíz de este repositorio, no desde el directorio personal.
- Remoto esperado: `https://github.com/RSTChile/Cosmolab.git`.
- Rama base esperada: `main`.
- Nunca enviar directamente a `main` ni fusionar el pull request sin autorización
  explícita del usuario.
- Si se parte de `main`, usar una rama como
  `agent/actualizar-experimentos-AAAA-MM-DD`.
- Si ya existe un pull request borrador de la misma actualización, continuar en su
  rama solamente cuando corresponda al alcance pedido.

## Exclusiones permanentes

Además de todo lo indicado por `.gitignore`, excluir siempre del cambio:

- `Teoría Cosmosemiótica/` y cualquier variante acentuada del mismo nombre.
- `SDRunoPlugin_Cosmo/`.
- `marcha-motor/`.
- `Fondo de Microondas/`.
- `Falsacion-S/`.
- `Cosmotesla/`.
- `Cosmolab_audio_preserved/`.
- `Cosmoclima/infraestructura/`.
- `results/`.

Algunas carpetas pueden contener archivos ya versionados; por eso no basta con
`.gitignore`. Antes de publicar, comprobar que el diff contra `main` contiene
cero rutas de esta lista.

## Material que nunca debe publicarse automáticamente

- Credenciales, contraseñas, tokens, claves privadas, certificados privados,
  direcciones con autenticación incrustada o valores locales equivalentes.
- Archivos `.env`, configuraciones `settings.local`, memoria operativa, estados de
  sesión, configuraciones Wi-Fi o archivos que identifiquen dispositivos locales.
- Enlaces simbólicos hacia discos, volúmenes, entornos virtuales o rutas de esta
  máquina.
- Entornos virtuales, cachés, dependencias instaladas, builds, temporales y logs
  puramente operativos que no sean evidencia experimental deliberada.
- Datos de terceros cuya licencia limite la redistribución, manuales comerciales y
  copias de artículos o libros que no formen parte de un entregable propio.
- Material interno sensible de RMD 2.0.

Si un archivo útil mezcla código con una credencial local, dejarlo fuera y
explicar el límite. No modificar código de un organismo activo para sanearlo sin
autorización expresa.

## Protección de experimentos y del trabajo local

- No iniciar, detener, reiniciar ni reconstruir Docker, ANIMA, servicios, hardware
  o baterías experimentales para hacer una actualización de GitHub.
- No usar `git reset --hard`, `git checkout --`, limpieza recursiva, `stash` ni
  operaciones que oculten o destruyan trabajo local.
- No borrar archivos locales para excluirlos: dejarlos intactos y fuera del índice.
- No incluir borrados versionados salvo que el usuario los autorice expresamente.
- En un árbol de trabajo mixto, preparar rutas explícitas; no usar `git add -A` de
  manera indiscriminada.

## Flujo obligatorio

1. Leer estas instrucciones y las instrucciones más específicas que existan dentro
   de cada experimento afectado.
2. Inspeccionar rama, remoto, estado, diff, archivos nuevos, tamaños, enlaces
   simbólicos y cambios locales que deben preservarse.
3. Ejecutar `git fetch origin main` y verificar la relación entre `HEAD`, `main` y
   `origin/main`. No cambiar de rama si eso pone en riesgo cambios locales.
4. Definir el alcance por experimento y preparar una rama `agent/...` cuando sea
   necesario.
5. Seleccionar y preparar solo archivos propios y publicables. Mantener fuera las
   exclusiones permanentes, configuraciones locales y material restringido.
6. Verificar antes del commit:
   - cero rutas excluidas en el diff contra `main`;
   - cero borrados no autorizados;
   - cero archivos privados por nombre y cero secretos de alta confianza;
   - cero enlaces simbólicos locales;
   - ningún archivo individual sobre 100 MB; informar los mayores;
   - sintaxis de los Python y JavaScript cambiados;
   - parseo de los JSON cambiados;
   - pruebas pertinentes que puedan ejecutarse sin intervenir organismos ni
     hardware. Informar con precisión cualquier dependencia faltante.
7. Revisar el resumen completo del diff y hacer un commit breve e intencional.
8. Enviar la rama, verificar que el SHA remoto coincida con el local y abrir o
   actualizar un pull request borrador contra `main`.
9. En el pull request documentar:
   - experimentos incluidos;
   - exclusiones aplicadas;
   - controles de privacidad y licencia;
   - verificaciones realmente ejecutadas;
   - advertencias, pruebas no ejecutadas y frontera entre simulación y validación
     física.
10. Confirmar que el pull request es integrable, pero dejarlo en borrador hasta que
    el usuario ordene revisarlo, marcarlo listo o fusionarlo.

## Criterio de evidencia

El cierre debe separar explícitamente:

- informado por el usuario;
- inspeccionado en archivos;
- verificado ejecutando código o comandos;
- pendiente o no verificable en la máquina actual.

No presentar una compilación sintáctica como validación científica ni una
simulación como prueba física. No maquillar resultados parciales o pruebas
bloqueadas.
