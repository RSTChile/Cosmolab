# Descargas — ANIMA

Esta página lista la ubicación pública desde la que se pueden descargar los instaladores de ANIMA y da instrucciones básicas de verificación. Los instaladores completos (Windows, macOS Intel/ARM, Linux x86_64/ARM) están publicados en:

https://cosmosemiotica.cl/descargas/anima.html

Notas rápidas

- Preferible: descargar desde la página anterior y verificar integridad con SHA256 y/o firma GPG (si las proporcionan).
- También es posible crear una GitHub Release que apunte a esa página (sin subir binarios al repositorio). Para eso hay un workflow opcional en este repo que el equipo puede ejecutar desde Actions para publicar una Release con el enlace externo.

Enlaces (ejemplos)

- Página pública de instaladores: https://cosmosemiotica.cl/descargas/anima.html
- Página de Releases del repositorio: https://github.com/RSTChile/Cosmolab/releases
- Última release: https://github.com/RSTChile/Cosmolab/releases/latest

Comprobación de integridad (ejemplos)

- Verificar SHA256 localmente:

  sha256sum anima-v1.2.0-linux-x86_64.tar.gz

  # o para comprobar con un archivo .sha256
  sha256sum -c anima-v1.2.0-linux-x86_64.tar.gz.sha256

- Verificar firma GPG (si se publica la firma):

  gpg --verify anima-v1.2.0-linux-x86_64.tar.gz.asc anima-v1.2.0-linux-x86_64.tar.gz

Crear una Release en GitHub (si se desea)

- Manual con gh CLI (desde tu máquina):

  gh release create v1.2.0 --title "anima v1.2.0" \
    --notes "Instaladores disponibles en https://cosmosemiotica.cl/descargas/anima.html"

- Automática desde Actions: existe la posibilidad de añadir un workflow dispatch que cree una Release (draft) con el enlace externo en el cuerpo; puedo añadirlo en un PR aparte si lo desean.

Advertencias

- No committear instaladores binarios grandes en el árbol del repositorio; usar Releases o un almacenamiento externo (S3, CDN) si son pesados.
- Revisar políticas de licencias y permisos para redistribuir cualquier contenido en los instaladores.

Si quieres, creo también la Pull Request con este archivo y un workflow para crear Releases automáticamente; ya estoy en la rama `agent/add-descargas-link-2026-08-17`.
