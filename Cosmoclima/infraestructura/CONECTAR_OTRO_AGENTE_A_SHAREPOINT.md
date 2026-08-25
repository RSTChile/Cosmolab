# Conectar otro agente al SharePoint del RMD

Instrucciones para que un agente externo (DeepSeek Harness u otro) opere sobre
`rstchilecom.sharepoint.com/RMD` con permisos de lectura, escritura, creación de
listas y borrado.

Escrito el 23-ago-2026, a partir de lo que ya funciona en este proyecto. Todo lo
que dice «medido» se comprobó contra el inquilino real, no se supone.

---

## 0 · Lo que hay que decidir primero: quién es el agente

Hay dos formas de que un programa entre a SharePoint, y **no son intercambiables**.

| | **Ruta A · Delegada** | **Ruta B · Aplicación propia** |
|---|---|---|
| Quién actúa | Alexis, prestándole su sesión al agente | El agente, con identidad propia |
| Cómo entra | Alexis pega un código en el navegador, una vez | Nadie interviene, nunca |
| Permisos | Los que ya tiene Alexis | Los que el administrador le conceda |
| Vence | El token de refresco dura ~90 días deslizantes | No vence |
| Registro de auditoría | Todo aparece hecho «por Alexis» | Aparece hecho por el agente |
| Se implementa en | 5 minutos, ya está probado aquí | ~30 minutos, requiere ser administrador |

**La analogía:** la ruta A es prestarle tu tarjeta de acceso al edificio. Funciona
al instante, pero el registro dirá que entraste tú, y cuando caduque tu tarjeta
el agente queda afuera. La ruta B es darle su propia tarjeta al agente, con su
propia foto y sus propios permisos.

**Recomendación:** empezar por la **A** para que funcione hoy, y migrar a la **B**
cuando el agente deba correr solo y sin supervisión. Si el agente va a borrar
cosas, la B es la correcta — conviene que el registro diga quién borró qué.

---

## 1 · Datos del sitio (los mismos para las dos rutas)

Copiar tal cual. No son secretos y ya están verificados contra el servicio.

```python
SITIO = ("rstchilecom.sharepoint.com,"
         "d14670c1-197a-4976-ac78-92eb6d9ccff6,"
         "24a0c429-ccba-4544-bd83-e831a7a5a1a8")

GRAPH = "https://graph.microsoft.com/v1.0"

# Listas principales, por si se necesitan directamente
LISTA_MIC = "315db0cf-1a0b-42b5-a140-c5f4bb396d46"   # Matriz de Infraestructura Crítica
```

El formato del identificador de sitio es `hostname,siteCollectionId,webId`. Las
tres partes van juntas y separadas por comas, sin espacios.

Para descubrir cualquier otra lista por su nombre visible:

```
GET /sites/{SITIO}/lists?$select=id,name,displayName&$top=200
```

---

## 2 · Los permisos, y cuál hace falta para qué

Esto es lo que más tiempo ahorra, porque **está medido, no supuesto**.

| Ámbito (scope) | Qué habilita | ¿Hace falta? |
|---|---|---|
| `Sites.Read.All` | Leer listas y elementos | Insuficiente para escribir |
| `Sites.ReadWrite.All` | Crear, editar y **borrar elementos** de listas | Sí |
| `Sites.Manage.All` | **Crear y borrar LISTAS**, crear y modificar COLUMNAS, indexar | Sí |
| `Sites.FullControl.All` | Todo lo anterior + permisos del sitio | Sólo si hay que tocar permisos |
| `offline_access` | Entrega el token de refresco | Sí, imprescindible |

> **★ MEDIDO el 21-ago-2026:** con sólo `Sites.ReadWrite.All`, crear una lista
> devuelve **HTTP 403**. Escribir filas sí basta con `ReadWrite`; **crear listas
> exige `Manage`**. Si el agente va a crear sub-matrices, necesita `Manage.All`.

> **★ Ojo con verificar los ámbitos:** el token devuelto **no incluye
> `offline_access` en su campo `scope`** aunque se haya pedido y concedido. No es
> un error: se comprueba mirando que exista `refresh_token` en la respuesta, no
> la cadena de ámbitos.

> **★ Y `offline_access` no es opcional.** Sin él no hay token de refresco, y la
> sesión muere en una hora. Inservible para cualquier trabajo largo.

### Cadena de ámbitos lista para copiar

```python
AMBITO = ("https://graph.microsoft.com/Sites.Manage.All "
          "https://graph.microsoft.com/Sites.ReadWrite.All "
          "offline_access")
```

Si además debe administrar permisos del sitio, cambiar `Sites.Manage.All` por
`Sites.FullControl.All`.

---

## 3 · RUTA A — Sesión delegada por código de dispositivo

Es la que este proyecto usa hoy y está probada. **No requiere contraseña en
ninguna parte**: Alexis pega un código en el navegador una sola vez.

### 3.1 · El identificador de cliente

```python
CLIENTE   = "14d82eec-204b-4c2f-b7e8-296a70dab67e"   # Microsoft Graph Command Line Tools
AUTORIDAD = "https://login.microsoftonline.com/organizations"
```

Ese identificador es **público y de Microsoft** — es su herramienta oficial de
línea de comandos. No es un secreto y no hay que registrar nada. Ya está
consentido en este inquilino.

### 3.2 · El código completo, funcional

```python
import json, os, time, requests
from pathlib import Path

SITIO     = ("rstchilecom.sharepoint.com,d14670c1-197a-4976-ac78-92eb6d9ccff6,"
             "24a0c429-ccba-4544-bd83-e831a7a5a1a8")
GRAPH     = "https://graph.microsoft.com/v1.0"
CLIENTE   = "14d82eec-204b-4c2f-b7e8-296a70dab67e"
AUTORIDAD = "https://login.microsoftonline.com/organizations"
AMBITO    = ("https://graph.microsoft.com/Sites.Manage.All "
             "https://graph.microsoft.com/Sites.ReadWrite.All offline_access")

# ★ El token va FUERA del proyecto y fuera de cualquier repositorio.
CACHE = Path.home() / ".cache" / "rmd_agente"
TOKEN = CACHE / "token.json"


def guardar(j):
    CACHE.mkdir(parents=True, exist_ok=True)
    j["expira_en"] = time.time() + j.get("expires_in", 3600) - 120   # margen
    TOKEN.write_text(json.dumps(j))
    os.chmod(TOKEN, 0o600)          # sólo el dueño puede leerlo


def login():
    """Se corre UNA vez. Imprime un código que Alexis pega en el navegador."""
    r = requests.post(f"{AUTORIDAD}/oauth2/v2.0/devicecode",
                      data={"client_id": CLIENTE, "scope": AMBITO},
                      timeout=30).json()
    print(r["message"], flush=True)          # ← el código, VISIBLE
    espera = r.get("interval", 5)
    for _ in range(180):
        time.sleep(espera)
        t = requests.post(f"{AUTORIDAD}/oauth2/v2.0/token",
                          data={"client_id": CLIENTE,
                                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                                "device_code": r["device_code"]}, timeout=30).json()
        if "access_token" in t:
            guardar(t)
            print("sesión iniciada · ámbitos:", t.get("scope"))
            return
        if t.get("error") == "authorization_pending":
            continue
        if t.get("error") == "slow_down":
            espera += 5
            continue
        raise SystemExit(f"falló: {t.get('error_description', t)}")
    raise SystemExit("expiró el código")


def token():
    """Token válido. Se renueva solo con el token de refresco."""
    if not TOKEN.exists():
        raise SystemExit("No hay sesión. Corre login() primero.")
    j = json.loads(TOKEN.read_text())
    if time.time() < j.get("expira_en", 0):
        return j["access_token"]
    if "refresh_token" not in j:
        raise SystemExit("Sesión vencida sin refresco. Corre login() otra vez.")
    n = requests.post(f"{AUTORIDAD}/oauth2/v2.0/token",
                      data={"client_id": CLIENTE, "grant_type": "refresh_token",
                            "refresh_token": j["refresh_token"],
                            "scope": AMBITO}, timeout=30).json()
    if "access_token" not in n:
        raise SystemExit("No se pudo renovar. Corre login() otra vez.")
    guardar(n)
    return n["access_token"]


def llamar(metodo, url, cuerpo=None, reintentos=6):
    """UNA llamada a Graph. Si el servidor pide esperar, se espera lo que pide."""
    for intento in range(reintentos):
        r = requests.request(
            metodo, url if url.startswith("http") else GRAPH + url,
            headers={"Authorization": "Bearer " + token(),
                     "Content-Type": "application/json"},
            data=json.dumps(cuerpo) if cuerpo else None, timeout=120)
        if r.status_code in (429, 503, 504):
            time.sleep(int(r.headers.get("Retry-After", 2 ** intento * 5)))
            continue
        return r
    return r
```

### 3.3 · El paso humano, una sola vez

1. El agente corre `login()`.
2. Imprime algo como:
   *«To sign in, use a web browser to open the page https://microsoft.com/devicelogin
   and enter the code XXXXXXXXX to authenticate.»*
3. **Alexis** abre esa página, escribe el código y acepta con
   `alexis.lopez.tapia@rst-chile.com`.
4. Listo. A partir de ahí el agente se renueva solo durante ~90 días.

> **⚠️ Si el agente corre en segundo plano**, el mensaje del código tiene que
> llegar a algún lado visible. Ya pasó aquí: un `print()` sin volcado inmediato
> dejó el archivo de salida vacío y el código nunca apareció. Usar `flush=True`
> o escribir a un archivo que se pueda leer.

---

## 4 · RUTA B — Aplicación propia, sin intervención humana

Para que el agente corra solo indefinidamente y el registro de auditoría diga
quién hizo qué.

### 4.1 · Registrar la aplicación (lo hace Alexis, en el navegador)

1. Entrar a **https://entra.microsoft.com** con la cuenta de administrador.
2. **Aplicaciones → Registros de aplicaciones → Nuevo registro**.
3. Nombre: `Agente RMD` · Cuentas: *sólo este directorio organizativo*.
4. Registrar. Anotar **Id. de aplicación (cliente)** y **Id. de directorio
   (inquilino)**.
5. **Certificados y secretos → Nuevo secreto de cliente**. Copiar el **Valor**
   (no el Id.) — **sólo se muestra una vez**.
6. **Permisos de API → Agregar permiso → Microsoft Graph → Permisos de
   aplicación**. Agregar:
   - `Sites.ReadWrite.All`
   - `Sites.Manage.All` *(o `Sites.FullControl.All` si debe tocar permisos)*
7. **Conceder consentimiento del administrador** ← sin este botón, nada funciona.

### 4.2 · El código

```python
import requests, time

INQUILINO = "«Id. de directorio»"
CLIENTE   = "«Id. de aplicación»"
SECRETO   = os.environ["RMD_SECRETO"]      # ★ NUNCA escrito en el código

def token_app():
    r = requests.post(
        f"https://login.microsoftonline.com/{INQUILINO}/oauth2/v2.0/token",
        data={"client_id": CLIENTE, "client_secret": SECRETO,
              "grant_type": "client_credentials",
              "scope": "https://graph.microsoft.com/.default"}, timeout=30).json()
    return r["access_token"]
```

El resto (`llamar`) es idéntico, cambiando `token()` por `token_app()` con una
caché en memoria del vencimiento.

> **★ Alternativa más segura: `Sites.Selected`.** En vez de dar acceso a TODOS
> los sitios de la organización, se concede `Sites.Selected` y después el
> administrador habilita el agente **sólo en el sitio RMD**:
> ```
> POST /sites/{SITIO}/permissions
> {"roles":["write"],
>  "grantedToIdentities":[{"application":{"id":"«Id. de aplicación»",
>                                          "displayName":"Agente RMD"}}]}
> ```
> Es lo correcto para un agente que sólo debe tocar el RMD. Si toca un sitio que
> no le corresponde, falla — y eso es una virtud, no un estorbo.

---

## 5 · Operaciones, con la sintaxis exacta

Todas asumen la función `llamar()` de arriba.

### Leer una lista completa (con paginación — **obligatoria**)

```python
def leer_lista(lid, campos):
    filas, url = [], f"/sites/{SITIO}/lists/{lid}/items?$expand=fields($select={campos})&$top=500"
    while url:
        d = llamar("GET", url).json()
        filas += [x["fields"] for x in d.get("value", [])]
        url = d.get("@odata.nextLink")     # ★ sin esto sólo se leen las primeras
    return filas
```

### Crear un elemento

```python
llamar("POST", f"/sites/{SITIO}/lists/{lid}/items",
       {"fields": {"Title": "…", "Numero": 1}})
```

### Modificar un elemento

```python
llamar("PATCH", f"/sites/{SITIO}/lists/{lid}/items/{item_id}/fields",
       {"Campo": valor})
```

### Borrar un elemento

```python
llamar("DELETE", f"/sites/{SITIO}/lists/{lid}/items/{item_id}")
```

### Crear una lista con columnas

```python
llamar("POST", f"/sites/{SITIO}/lists", {
    "displayName": "Nombre visible",
    "list": {"template": "genericList"},
    "columns": [
        {"name": "Region",  "displayName": "Región",   "text": {}},
        {"name": "Lat",     "displayName": "Latitud",  "number": {"decimalPlaces": "eight"}},
        # columna de búsqueda hacia otra lista
        {"name": "MICR",    "displayName": "MICR",
         "lookup": {"listId": LISTA_MIC, "columnName": "Title"}},
        # búsqueda de VARIOS valores
        {"name": "SubMatriz", "displayName": "Sub-Matriz",
         "lookup": {"listId": ID_CATALOGO, "columnName": "Title",
                    "allowMultipleValues": True}},
    ]})
```

### Borrar una lista entera

```python
llamar("DELETE", f"/sites/{SITIO}/lists/{lid}")     # ⚠️ irreversible
```

### Agregar una columna a una lista existente

```python
llamar("POST", f"/sites/{SITIO}/lists/{lid}/columns",
       {"name": "Nuevo", "displayName": "Nuevo", "number": {"decimalPlaces": "four"}})
```

### Indexar una columna (necesario sobre 5.000 elementos)

```python
cols = llamar("GET", f"/sites/{SITIO}/lists/{lid}/columns?$select=id,name").json()["value"]
cid  = next(c["id"] for c in cols if c["name"] == "Region")
llamar("PATCH", f"/sites/{SITIO}/lists/{lid}/columns/{cid}", {"indexed": True})
```

### Escribir en lote (hasta 20 por vez)

```python
cuerpo = {"requests": [
    {"id": "1", "method": "PATCH",
     "url": f"/sites/{SITIO}/lists/{lid}/items/{iid}/fields",
     "headers": {"Content-Type": "application/json", "If-Match": "*"},
     "body": {"Campo": valor}},
]}
r = llamar("POST", "/$batch", cuerpo)
```

---

## 6 · Las trampas que ya costaron trabajo en este proyecto

Cada una está medida. Ahorran horas.

### ★★★ `$batch` devuelve 200 aunque las peticiones de adentro fallen

**El 21-ago se perdieron 208 filas en silencio por esto.** El sobre llega bien;
las cartas de adentro pueden venir todas rechazadas.

```python
r = llamar("POST", "/$batch", cuerpo)        # HTTP 200 ✓
for resp in r.json()["responses"]:
    if resp["status"] >= 300:                # ← ESTO hay que mirarlo
        reintentar(resp["id"])
```

**Regla: nunca contar filas escritas por lo que uno cree que envió. Volver a
leer la lista y reconciliar contra el origen.**

### ★★ Las columnas calculadas devuelven `MAX()` mal, sin avisar

`=[IB]/MAX([IB])` **se acepta sin protestar y devuelve 1 en todas las filas**.
Esa función no mira la columna: mira la propia fila. Una normalización que parece
calculada y no lo está.

**Regla: no usar funciones de agregación en columnas calculadas de SharePoint.
Calcular fuera y escribir el resultado.**

### ★★ Los nombres internos de columna no son los visibles

SharePoint codifica los caracteres no ASCII. Hay que usar el nombre **interno**
en las consultas, no el que se ve en pantalla:

| Se ve | Nombre interno |
|---|---|
| N° | `N_x00b0_` |
| Región | `Regi_x00f3_n` |
| Número | `N_x00fa_mero` |
| Descripción | `Descripci_x00f3_n` |
| Teléfono | `Tel_x00e9_fono` |
| Dirección | `Direcci_x00f3_n` |

Para descubrirlos: `GET /sites/{SITIO}/lists/{lid}/columns?$select=name,displayName`

### ★★ El umbral de 5.000 elementos

Una vista que deba recorrer más de 5.000 elementos falla. **En el sitio RMD hay
siete listas por encima**, la mayor con 16.768. La solución es indexar la columna
`Región` y dejar la vista agrupada o filtrada por ella — ninguna región supera
los 5.000 en ninguna de las siete.

⚠️ Crear un índice sobre una lista que **ya** pasó el umbral puede devolver
400 o 503. Reintentar; a veces hay que hacerlo en horario de poca carga.

### ★ Los números vuelven como decimales

Un campo numérico que contiene `1` se lee como `1.0`. Si se compara con `"1"`
como texto, **no cruza nunca** y una comparación entre dos exportaciones de la
misma lista sale «todas nuevas, cero cambios». Normalizar siempre:

```python
def num(v):
    return None if v in (None, "") else int(float(v))
```

### ★ Las columnas de búsqueda se escriben con otro nombre

Para escribir una búsqueda no se usa el nombre de la columna sino
`«Nombre»LookupId`, con el **identificador interno** del elemento destino:

```python
{"MICRLookupId": 101}                                     # un valor
{"SubMatrizLookupId@odata.type": "Collection(Edm.Int32)", # varios valores
 "SubMatrizLookupId": [12, 13, 14]}
```

### ★ Ritmo de escritura sostenible

Medido: **~11 filas por segundo** en lotes de 20 con pausa adaptativa. Si empiezan
los 429, subir la pausa; cuando pasen, **bajarla otra vez** — aquí se quedó
atascada en 8 segundos durante una hora y el ritmo cayó de 11 a 2,1 filas/s.

---

## 7 · Antes de dejarlo borrar cosas

El borrado en SharePoint por Graph **no pregunta**. Tres salvaguardas mínimas:

1. **Exportar antes.** Bajar la lista completa a CSV y guardarla con fecha. En
   este proyecto se hace con `bajar_micr_sharepoint.py` y los exports viejos
   nunca se pisan: son la evidencia contra la cual se mide qué cambió.
2. **Borrado en dos pasos.** Que el agente primero informe qué va a borrar y
   cuántos elementos son, y que ejecute sólo con confirmación explícita. Nunca
   `DELETE` dentro de un bucle sin un tope declarado.
3. **`Sites.Selected` en vez de `.All`** si se va por la ruta B, para que el
   agente no pueda tocar ningún otro sitio de la organización aunque se equivoque.

Los elementos borrados van a la papelera del sitio y se pueden recuperar durante
93 días. **Las listas borradas también**, pero recuperarlas restaura el esquema,
no necesariamente las columnas de búsqueda que apuntaban a ellas desde otras
listas — ésas quedan rotas.

---

## 8 · Prueba de que quedó bien conectado

Tres llamadas, en orden. Si las tres pasan, el agente puede operar. **Están
probadas contra el inquilino real el 23-ago-2026**, con los resultados que se
indican en cada línea.

```python
# 1 · ¿hay sesión y ve el sitio?  (las dos cosas a la vez)
r = llamar("GET", f"/sites/{SITIO}/lists?$select=id,displayName&$top=200")
print(r.status_code, len(r.json()["value"]), "listas")         # espera 200 y ~60

# ⚠️ NO usar `GET /me` como prueba de sesión: devuelve 403 con estos permisos.
# Probado el 23-ago-2026. `/me` exige el ámbito `User.Read`, que NO está en la
# lista de arriba y que el agente no necesita para nada. Un 403 ahí no significa
# que la sesión esté mal — significa que se preguntó algo fuera de alcance.
# Si se quiere igual, agregar `User.Read` a la cadena de ámbitos.

# 2 · ¿puede ESCRIBIR? (crea y borra un elemento de prueba)
lid = next(x["id"] for x in r.json()["value"] if x["displayName"] == "Sub-Matrices")
p = llamar("POST", f"/sites/{SITIO}/lists/{lid}/items",
           {"fields": {"Title": "PRUEBA — borrar"}})
print("crear:", p.status_code)                                 # espera 201
print("borrar:", llamar("DELETE",
      f"/sites/{SITIO}/lists/{lid}/items/{p.json()['id']}").status_code)   # espera 204

# 3 · ¿puede CREAR LISTAS? (la que exige Sites.Manage.All)
q = llamar("POST", f"/sites/{SITIO}/lists",
           {"displayName": "PRUEBA AGENTE — borrar",
            "list": {"template": "genericList"}})
print("crear lista:", q.status_code)                           # 201 = tiene Manage
if q.status_code == 201:
    llamar("DELETE", f"/sites/{SITIO}/lists/{q.json()['id']}")
```

**Si el paso 3 devuelve 403**, el ámbito `Sites.Manage.All` no se concedió. En la
ruta A hay que volver a correr `login()` con la cadena de ámbitos completa; en la
ruta B, falta el consentimiento del administrador.

---

## 9 · Recordatorio pendiente

⚠️ La contraseña que se pegó en el chat el 21-ago-2026 **sigue sin rotar**. No se
usó nunca ni se guardó en ningún archivo del proyecto — todo el acceso va por
código de dispositivo, sin contraseña — pero está en el historial de esa
conversación y conviene cambiarla.

Ninguna de las dos rutas de este documento necesita la contraseña.
