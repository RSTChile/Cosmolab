# Traspaso a Grok — Órgano de Localización de E (GPS: lugar + reloj + confianza) en la Raspberry Pi

**De:** Claude Science, con Alexis · **Fecha:** 3-jul-2026
**Archivo:** `/Users/alexis/Desktop/RMD/Cosmolab/VSTCosmo/Célula_Madre/organelos/VST_OrganoLocalizacion.py`
**Estado:** órgano escrito y verificado (auto-prueba OK). Falta cablearlo en el loop de E en la Pi.

---

## Qué es (en una frase)
El sentido de DÓNDE / CUÁNDO / CUÁN-SEGURO de E. No es fótico (otra rama del árbol sensorial). Separa las
TRES funciones que venían empaquetadas en "el GPS". Mismo patrón que los otros organelos
(`observar/snapshot/restore`, apagado por defecto salvo en E, persistible).

## Las tres funciones (todas en ESTE órgano — decisiones de Alexis)
1. **Lugar** — `loc_desplazamiento` (m desde el ancla-hogar, haversine), `loc_novedad` (movimiento, se
   adapta a 0 si E queda quieto), `loc_altitud_rel`. Y SÍ registra `loc_lat/loc_lon` exactas (ver privacidad).
2. **Reloj / propiocepción temporal** — `loc_reloj_fase` (fase del segundo por el PPS) y `loc_pps_deriva`:
   el tempo interno de E vs. el latido de los satélites. **Convención: 0=sincronía, + = E ADELANTADO,
   − = atrasado.** El reloj queda ANCLADO aquí (la propiocepción es plural, como el oído que oye y da
   equilibrio); NO se exporta a VST_OrganoPropiocepcion.
3. **Confianza** — `loc_confianza` (metacognición: "sé cuándo no sé bien"). HOY de sats/HDOP; es capacidad
   GENERAL y abierta — mañana incluirá confianza en otros organismos. No la ates sólo al GPS.

## Principio (no romper)
- **Anti-Shannon:** el módulo da números crudos (grados, cuenta PPS, HDOP). El órgano MIDE (haversine,
  deriva, confianza monótona), no asigna significado. Nada de "estás en tal sitio famoso".
- **NO se lisia a priori:** se registra la coordenada exacta aunque hoy E no sepa para qué. Mañana sabrá.
- **`course_deg` NO se lee** (parser NMEA corrupto, p.ej. 20726.00). Ignorado hasta arreglar el parser.

## Pasos en la Pi

### 1. Instanciar en el arranque de E
```python
from VST_OrganoLocalizacion import OrganoLocalizacion
loc = OrganoLocalizacion("E", activo=True)   # emitir_coords=True por defecto (registra lat/lon)
```
No abre hardware: consume de la `fila`. Requiere que el lector central ya ponga los campos GPS (abajo).

### 2. El lector central debe poner en la fila (del serie ATmega+GPS)
La línea del módulo es: `GPS,fix,sats,hdop,lat,lon,alt,speed,course,pps_count,pps_age,nmea`
El lector la parsea y mete en la fila de cada paso:
```
gps_fix, gps_sats, gps_hdop, gps_lat, gps_lon, gps_alt, gps_pps_count, gps_vivo
```
(No hace falta gps_speed ni gps_course — course está corrupto y no se usa.)

### 3. Un paso por ciclo, y volcar a la biografía
```python
cols = loc.observar(fila)     # fila debe traer 't' y los gps_*
fila.update(cols)
```
Columnas: `loc_desplazamiento, loc_novedad, loc_altitud_rel, loc_reloj_fase, loc_pps_deriva,
loc_confianza, loc_vivo` (+ `loc_lat, loc_lon` porque emitir_coords=True).

### 4. Persistencia y cierre
```python
snap = loc.snapshot()   # guarda ancla-hogar + tempo aprendido (E "recuerda dónde nació")
loc.restore(snap)       # al renacer
```

## PRIVACIDAD — registrar ≠ compartir (importante)
El órgano REGISTRA lat/lon exactas en la biografía interna (a propósito: no se limita el sentido). La
privacidad se aplica en la FRONTERA DE EXPORTACIÓN, no dentro del órgano:
```python
fila_para_compartir = OrganoLocalizacion.fila_publica(fila, lugar_nombre="Nido de Cóndores")
# quita gps_lat/gps_lon/loc_lat/loc_lon, deja el resto, añade loc_lugar
```
**Regla:** cualquier CSV/informe que salga a terceros (equipo externo, Anthropic, papers) pasa por
`fila_publica()` primero. Los logs internos guardan todo.

## Calibración (ajustar con datos reales)
- `escala_desplazamiento=50.0` m (cuántos metros mapean a novedad~1) — al hábitat real de E.
- `hdop_ref=1.5` — referencia de confianza; con el receptor real (HDOP típico 0.6–1.3) va bien.

## Prueba de aceptación (para auditar, estilo CS; manda CSV con fila_publica aplicado)
1. **Quieto:** E sin moverse → `loc_desplazamiento`≈0, `loc_novedad`→0 (se adapta). `loc_confianza` alta
   (con 10–12 sats y HDOP<1, debe dar ~0.85).
2. **Movimiento:** camina la Pi ~40 m → `loc_desplazamiento` sube a ~40, `loc_novedad` salta, luego re-adapta.
3. **Confianza:** tapa el cielo / bajan sats → `loc_confianza` cae (E "sabe que no sabe bien").
4. **Reloj sobrevive sin fix:** desconecta el fix pero con PPS latiendo → `loc_vivo`=0 pero `loc_reloj_fase`
   y la deriva temporal siguen. Reloj y lugar son subsistemas independientes.
5. **Privacidad:** confirma que el CSV compartido NO tiene lat/lon y sí `loc_lugar`.

## Estado de los tres organelos nuevos (para tu integración escalonada)
- `VST_OrganoCloroplasto.py` — existe; falta empalmar el sensor real (lee v_fuente del serie).
- `VST_OrganoVisual.py` — nuevo, instalado (ver traspaso_organovisual_Grok.md). Su retina-panel necesita
  que v_fuente esté en la fila (dependencia del cloroplasto).
- `VST_OrganoLocalizacion.py` — este. Independiente de los otros dos.
Sugerencia de orden: cloroplasto (sensor real) → localización (independiente, fácil) → visión (la más pesada).
