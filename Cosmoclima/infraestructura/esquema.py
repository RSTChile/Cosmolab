"""
El esquema normalizado único: dos tablas donde aterriza TODO.

POR QUÉ EXISTE
--------------
Chile publica peligro de aluvión en «Alta/Moderada/Baja», viento en «km/h»,
sismos en «magnitud», cortes eléctricos en «número de clientes». Nada de eso es
comparable entre sí, y por eso los informes se pueden apilar pero no sumar.
Este esquema es el lugar donde dejan de ser cosas distintas: cada dato entra con
su valor original intacto Y con un valor normalizado 0-1 comparable, y declara
de dónde salió, cuándo vale, dónde vale y cuánto se le puede creer.

LAS DOS TABLAS
--------------
`fuente`      — el registro de quién publica qué, cómo se accede y si su uso
                automatizado está permitido.
`observacion` — el dato. Nunca se guarda un valor sin su vigencia, su confianza
                y su anclaje territorial: sin esas tres cosas un dato no es
                utilizable para decidir nada.

REGLAS QUE LA BASE HACE CUMPLIR (no son sugerencias: la inserción falla)
-----------------------------------------------------------------------
1. Vigencia obligatoria. Un peligro sin fecha de inicio y fin no sirve — así lo
   publican la DMC y SERNAGEOMIN, y así lo necesita quien decide.
2. Confianza obligatoria. El propio MACC exige bloquear el coeficiente cuando la
   confianza del dato es baja. Una fuente caída deja un HUECO DECLARADO, jamás
   un cero.
3. Anclaje territorial nativo obligatorio. Todo dato dice en qué unidad viene
   publicado. La traducción a las otras unidades la hace `territorio.py` después
   y queda marcada como derivada, nunca confundida con el dato original.
4. El valor original NUNCA se pisa. Se guarda tal como llegó, con su unidad. El
   normalizado va aparte, junto con el nombre del método que lo produjo.

NOTA DE DISEÑO — por qué el anclaje es «nativo + derivado» y no «doble»
-----------------------------------------------------------------------
El plan pedía doble adscripción territorial obligatoria (zona geográfica Y
unidad administrativa). Al implementarlo aparece el problema real: SERNAGEOMIN
publica por zona geográfica y la CGE por comuna. Exigir las dos en la inserción
obligaría a inventar la que falta antes de tener la capa territorial cargada —
justo lo que no se debe hacer. Solución: se exige la NATIVA (la que el organismo
publicó) y se deja la derivada en null con una marca `territorio_resuelto=0`
hasta que la capa la complete. El requisito del plan se cumple igual, pero al
consultar y no al insertar, que es donde corresponde.
"""

import sqlite3
from pathlib import Path

AQUI = Path(__file__).parent
BASE = AQUI / "datos" / "consolidado.sqlite"

FAMILIAS = ("AMENAZA", "ACTIVO", "ESTADO", "BASE_TERRITORIAL")
TIPOS_TERRITORIO = ("punto", "zona_geografica", "comuna", "provincia",
                    "region", "pais")

DDL = """
PRAGMA foreign_keys = ON;

-- Quién publica qué, cómo se accede, y si podemos automatizarlo.
CREATE TABLE IF NOT EXISTS fuente (
    id                     TEXT PRIMARY KEY,       -- p.ej. 'sernageomin_remocion'
    organismo              TEXT NOT NULL,
    producto               TEXT NOT NULL,
    url                    TEXT NOT NULL,
    formato                TEXT NOT NULL,          -- json | csv | html | wfs | pdf
    familia                TEXT NOT NULL,
    acceso                 TEXT NOT NULL,          -- anonimo | registro | no_publico
    acceso_verificado      INTEGER NOT NULL,       -- 1 = lo probamos; 0 = supuesto
    condiciones_uso        TEXT,
    permite_automatizacion TEXT NOT NULL,          -- si | no | por_verificar
    granularidad           TEXT NOT NULL,
    historia_desde         TEXT,                   -- ISO o null si no hay archivo
    frecuencia             TEXT,
    confianza_base         REAL NOT NULL,          -- 0-1, techo de sus datos
    notas                  TEXT,
    CHECK (familia IN ('AMENAZA','ACTIVO','ESTADO','BASE_TERRITORIAL')),
    CHECK (acceso IN ('anonimo','registro','no_publico')),
    CHECK (permite_automatizacion IN ('si','no','por_verificar')),
    CHECK (acceso_verificado IN (0,1)),
    CHECK (confianza_base >= 0.0 AND confianza_base <= 1.0)
);

-- El dato. Con todo lo que hace falta para poder auditarlo después.
CREATE TABLE IF NOT EXISTS observacion (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    id_fuente           TEXT NOT NULL REFERENCES fuente(id),
    familia             TEXT NOT NULL,
    variable            TEXT NOT NULL,             -- 'peligro_remocion_masa', 'viento_max', ...

    -- el dato tal como lo publicó el organismo; NO se toca nunca
    valor_original      TEXT NOT NULL,
    unidad_original     TEXT NOT NULL,

    -- el dato hecho comparable; puede ser null si aún no se pudo normalizar
    valor_normalizado   REAL,
    metodo_normalizacion TEXT,

    -- CUÁNDO vale. Obligatorio: un peligro sin vigencia no es accionable.
    vigencia_inicio     TEXT NOT NULL,             -- ISO 8601
    vigencia_fin        TEXT NOT NULL,

    -- DÓNDE vale, como lo publicó el organismo (nativo)
    territorio_tipo     TEXT NOT NULL,
    territorio_id       TEXT NOT NULL,             -- 'Precordillera Alto Loa', '05101', ...
    lat                 REAL,                      -- sólo si territorio_tipo='punto'
    lon                 REAL,

    -- DÓNDE vale, traducido por territorio.py (derivado, se llena después)
    zona_geografica     TEXT,
    comuna              TEXT,
    provincia           TEXT,
    region              TEXT,
    territorio_resuelto INTEGER NOT NULL DEFAULT 0,

    -- CUÁNTO se le puede creer, y de dónde salió exactamente
    confianza           REAL NOT NULL,
    fecha_descarga      TEXT NOT NULL,
    url_exacta          TEXT NOT NULL,
    ruta_crudo          TEXT,                      -- dónde quedó el archivo original
    notas               TEXT,

    CHECK (familia IN ('AMENAZA','ACTIVO','ESTADO','BASE_TERRITORIAL')),
    CHECK (territorio_tipo IN ('punto','zona_geografica','comuna','provincia',
                               'region','pais')),
    CHECK (confianza >= 0.0 AND confianza <= 1.0),
    CHECK (valor_normalizado IS NULL OR
           (valor_normalizado >= 0.0 AND valor_normalizado <= 1.0)),
    CHECK (vigencia_fin >= vigencia_inicio),
    CHECK (territorio_resuelto IN (0,1)),
    -- si dice ser un punto, tiene que traer coordenadas de verdad
    CHECK (territorio_tipo <> 'punto' OR (lat IS NOT NULL AND lon IS NOT NULL)),
    -- si trae coordenadas, que caigan en Chile continental e insular sur
    CHECK (lat IS NULL OR (lat BETWEEN -56.0 AND -17.0)),
    CHECK (lon IS NULL OR (lon BETWEEN -110.0 AND -66.0))
);

CREATE INDEX IF NOT EXISTS ix_obs_fuente   ON observacion(id_fuente);
CREATE INDEX IF NOT EXISTS ix_obs_variable ON observacion(variable);
CREATE INDEX IF NOT EXISTS ix_obs_vigencia ON observacion(vigencia_inicio, vigencia_fin);
CREATE INDEX IF NOT EXISTS ix_obs_comuna   ON observacion(comuna);
CREATE INDEX IF NOT EXISTS ix_obs_zona     ON observacion(zona_geografica);

-- Huecos declarados: cuando una fuente NO entregó dato, se anota acá.
-- Existe para que «no hay dato» sea un hecho registrado y no un silencio que
-- después se confunda con un cero.
CREATE TABLE IF NOT EXISTS hueco (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    id_fuente      TEXT NOT NULL REFERENCES fuente(id),
    momento        TEXT NOT NULL,
    motivo         TEXT NOT NULL,          -- 'sin_respuesta','formato_cambiado',...
    detalle        TEXT,
    url_intentada  TEXT
);
"""


def conectar(ruta=BASE):
    """Abre la base creando las tablas si no existen. Deja las claves foráneas
    activas, que en SQLite vienen apagadas por omisión."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(ruta)
    con.row_factory = sqlite3.Row
    con.executescript(DDL)
    con.commit()
    return con


# ── validación en Python, antes de tocar la base ────────────────────────────
# SQLite atrapa lo estructural (rangos, dominios). Acá atrapamos lo que la base
# no puede ver: campos vacíos que técnicamente son texto válido, y la coherencia
# entre normalizado y método.

CAMPOS_OBLIGATORIOS = ("id_fuente", "familia", "variable", "valor_original",
                       "unidad_original", "vigencia_inicio", "vigencia_fin",
                       "territorio_tipo", "territorio_id", "confianza",
                       "fecha_descarga", "url_exacta")


def validar_observacion(obs):
    """Devuelve una lista de problemas. Lista vacía = la observación es válida.

    Se prefiere devolver TODOS los problemas juntos y no sólo el primero: si un
    adaptador está mal escrito, conviene verlo entero de una vez.
    """
    problemas = []

    for campo in CAMPOS_OBLIGATORIOS:
        valor = obs.get(campo)
        if valor is None or (isinstance(valor, str) and not valor.strip()):
            problemas.append(f"falta «{campo}» (obligatorio)")

    if obs.get("familia") not in FAMILIAS:
        problemas.append(f"familia inválida: {obs.get('familia')!r}")

    if obs.get("territorio_tipo") not in TIPOS_TERRITORIO:
        problemas.append(f"territorio_tipo inválido: {obs.get('territorio_tipo')!r}")

    conf = obs.get("confianza")
    if isinstance(conf, (int, float)) and not 0.0 <= conf <= 1.0:
        problemas.append(f"confianza fuera de 0-1: {conf}")

    # Un valor normalizado sin decir cómo se produjo es un número sin auditoría.
    if obs.get("valor_normalizado") is not None and not obs.get("metodo_normalizacion"):
        problemas.append("hay valor_normalizado pero no se declaró "
                         "metodo_normalizacion")

    if obs.get("territorio_tipo") == "punto":
        if obs.get("lat") is None or obs.get("lon") is None:
            problemas.append("territorio_tipo='punto' sin lat/lon")

    ini, fin = obs.get("vigencia_inicio"), obs.get("vigencia_fin")
    if isinstance(ini, str) and isinstance(fin, str) and ini and fin and fin < ini:
        problemas.append(f"vigencia al revés: {ini} → {fin}")

    return problemas


def insertar_observaciones(con, observaciones, estricto=True):
    """Inserta validando una por una.

    `estricto=True` (por omisión) corta al primer problema: es lo que se quiere
    al desarrollar un adaptador. `estricto=False` inserta las buenas, devuelve
    las malas y sigue: es lo que se quiere en una corrida nocturna, donde una
    fila mala no debe tumbar la cosecha entera.
    """
    buenas, rechazadas = [], []
    for obs in observaciones:
        problemas = validar_observacion(obs)
        if problemas:
            rechazadas.append((obs, problemas))
            if estricto:
                raise ValueError(
                    f"Observación inválida ({obs.get('variable')!r} de "
                    f"{obs.get('id_fuente')!r}): " + "; ".join(problemas))
        else:
            buenas.append(obs)

    if buenas:
        columnas = sorted({k for o in buenas for k in o})
        marcas = ", ".join(f":{c}" for c in columnas)
        sql = (f"INSERT INTO observacion ({', '.join(columnas)}) "
               f"VALUES ({marcas})")
        con.executemany(sql, [{c: o.get(c) for c in columnas} for o in buenas])
        con.commit()
    return len(buenas), rechazadas


def registrar_hueco(con, id_fuente, momento, motivo, detalle=None, url=None):
    """Anota que una fuente no entregó dato. Un hueco declarado vale mucho más
    que un cero silencioso: el modelo puede saltárselo, un cero lo contamina."""
    con.execute(
        "INSERT INTO hueco (id_fuente, momento, motivo, detalle, url_intentada) "
        "VALUES (?,?,?,?,?)", (id_fuente, momento, motivo, detalle, url))
    con.commit()


def registrar_fuente(con, **campos):
    """Alta o actualización de una fuente en el registro."""
    columnas = sorted(campos)
    sql = (f"INSERT OR REPLACE INTO fuente ({', '.join(columnas)}) "
           f"VALUES ({', '.join('?' for _ in columnas)})")
    con.execute(sql, [campos[c] for c in columnas])
    con.commit()


def resumen(con):
    """Foto rápida del estado de la base, para la bitácora."""
    q = lambda s: con.execute(s).fetchone()[0]
    return {
        "fuentes": q("SELECT COUNT(*) FROM fuente"),
        "observaciones": q("SELECT COUNT(*) FROM observacion"),
        "sin_territorio_resuelto": q(
            "SELECT COUNT(*) FROM observacion WHERE territorio_resuelto = 0"),
        "sin_normalizar": q(
            "SELECT COUNT(*) FROM observacion WHERE valor_normalizado IS NULL"),
        "huecos": q("SELECT COUNT(*) FROM hueco"),
    }


if __name__ == "__main__":
    con = conectar()
    print(f"Base lista: {BASE}")
    for k, v in resumen(con).items():
        print(f"  {k}: {v}")
