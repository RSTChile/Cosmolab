# Instrucción — Desplegar el sitio Cosmosemiótica en hosting cPanel
**Para:** Claude (web/escritorio, con acceso al navegador o a subida de archivos).
**De:** Claude Science (equipo Cosmosemiótica, Alexis López Tapia).

## Contexto en una línea
Hay que publicar un sitio web ESTÁTICO (HTML/CSS/JavaScript, sin backend) en un hosting cPanel ya
contratado. El addon domain `cosmosemiotica.cl` ya está creado, con document root propio y aislado en
`/home/geografiasagrada/cosmosemiotica.cl/`.

## Credenciales (Alexis las pega aquí al iniciar la conversación — NO quedan en ningún archivo)
- Panel: http://geografiasagrada.cl/cpanel
- Usuario: ⟨Alexis lo pega aquí⟩
- Contraseña: ⟨Alexis la pega aquí⟩
- **IMPORTANTE:** esta contraseña viajó por chat; cámbiala en cPanel apenas termines el despliegue.

## Qué es el sitio (naturaleza técnica)
- Páginas HTML autocontenidas: la principal es una simulación 3D del origen del universo
  (`UniversoCosmosemiotico.html`) y hay una segunda de nacimiento estelar (`nace_estrella.html`).
- Usan **three.js cargado por CDN** (r128, vía cdnjs/jsdelivr) — NO hay que instalar nada en el servidor.
- Cargan datos desde archivos `.json` locales (nodos de la teoría, líneas de tiempo, cronología).
- **Sin PHP, sin Node, sin base de datos, sin proceso de servidor.** Es 100% estático: el hosting solo
  tiene que servir archivos. Cualquier cPanel básico lo hace.

## Archivos a subir (Alexis te los adjunta en la conversación)
- `index.html` ← la página principal. (Es `UniversoCosmosemiotico.html` renombrada a `index.html` para
  que cargue sola al entrar al dominio. Si Alexis te pasa el nombre viejo, renómbrala a `index.html`.)
- `nace_estrella.html` y cualquier otra página secundaria.
- Todos los archivos `.json` que acompañan (nodos_teoria.json, timeline_*.json, cronologia_*.json,
  linea_complejidad.json, etc.).
- Cualquier carpeta de recursos (imágenes, audio) si Alexis la incluye.
- **Mantén la MISMA estructura de carpetas** con la que fueron creados: las páginas se referencian entre
  sí y cargan los JSON por RUTA RELATIVA. Si estaban todos en una misma carpeta, súbelos a la misma carpeta.

## Pasos de despliegue (en este orden)
1. **Entrar** a cPanel con las credenciales de arriba.
2. **Activar HTTPS primero.** cPanel → "SSL/TLS Status" → ejecutar AutoSSL (Let's Encrypt) para
   `cosmosemiotica.cl`. CRÍTICO: three.js se carga por CDN vía https; si el sitio va por http el navegador
   bloquea el contenido mixto y la simulación no arranca. No subas nada hasta tener el candado verde.
3. **Ir al File Manager** de cPanel y navegar a `/home/geografiasagrada/cosmosemiotica.cl/`
   (el document root del addon domain — NO `/public_html`, ese es del dominio principal geografiasagrada.cl).
4. **Subir los archivos** ahí (botón Upload, o FTP si prefieres). Respeta subcarpetas.
5. **Verificar que la principal se llame `index.html`** exactamente (en minúsculas). Sin eso, el dominio
   mostrará el listado de archivos o un error 403, no la página.
6. **Probar en el navegador:** abrir `https://cosmosemiotica.cl` y confirmar que:
   - la simulación 3D carga y anima (si la pantalla queda negra, casi siempre es three.js bloqueado por
     http → volver al paso 2, o una ruta de CDN mal escrita);
   - los controles (play, sliders, pestañas) responden;
   - la pestaña/enlace a `nace_estrella.html` funciona;
   - los datos de nodos/timeline se ven (si no, es un `.json` que no se subió o quedó en otra ruta).

## Diagnóstico de los fallos típicos (en orden de probabilidad)
- **Pantalla negra / no carga la simulación** → sitio servido por http en vez de https (three.js CDN
  bloqueado). Solución: activar AutoSSL y forzar https (cPanel → Domains → Force HTTPS Redirect).
- **404 en los .json / faltan los nodos** → un archivo no se subió, o la ruta relativa no coincide.
  Revisar que la estructura de carpetas sea idéntica a la original.
- **Sale el listado de archivos en vez de la web** → falta `index.html` en la raíz del document root.
- **El dominio no resuelve** → DNS: los nameservers de `cosmosemiotica.cl` deben apuntar a este hosting.
  Si el dominio se registró con este mismo proveedor suele estar listo; si se registró en otro, hay que
  apuntar los NS al hosting. Verificable con `dig cosmosemiotica.cl` o esperando propagación.

## Reglas del equipo (respetar)
- **No cambies el contenido de las páginas** — son el resultado de un trabajo científico en curso.
  Tu tarea es PUBLICAR los archivos tal cual, no editarlos. Si algo parece mal en el contenido,
  repórtalo a Alexis, no lo corrijas.
- **Cambiar la contraseña de cPanel** al terminar (viajó por chat).
- Si Alexis dice que aún no es momento de publicar, el sitio puede quedar en un subdominio de prueba
  o con acceso restringido hasta que él dé luz verde.

## Resumen
Sitio estático + cPanel = subir archivos con HTTPS activo. Sin complejidad de servidor. El único punto
que rompe la simulación es servir por http (three.js CDN bloqueado); todo lo demás es subir a la carpeta
correcta con `index.html` en la raíz.