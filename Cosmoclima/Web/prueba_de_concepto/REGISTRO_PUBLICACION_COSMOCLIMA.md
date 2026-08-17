# Registro de publicación · Experimento **Cosmoclima**

**Fecha de preparación:** 12-ago-2026
**Dirección científica:** Alexis López Tapia · **Desarrollo:** Claude (Anthropic)
**Estado:** listo en local, verificado. **La subida la autoriza/hace Alexis.**

---

## 1. Qué se publica

| Archivo | Peso | Destino |
|---|---|---|
| `informe-cosmoclima.html` | 40 KB | raíz del sitio |
| `sim-cosmoclima.html` | 2,1 MB | raíz del sitio (el instrumento) |
| `imagenes/web/` (23 láminas + 23 miniaturas) | 8,3 MB | `imagenes/web/` |
| `imagenes/web/campo/` (8 fotos + 8 miniaturas) | 4,7 MB | `imagenes/web/campo/` |
| `imagenes/Gyriosomus kingi.png` | 850 KB | `imagenes/` |

**Total a subir: ~16 MB.**

> ⚠ **NO subir la carpeta `imagenes/` completa: pesa 529 MB.** El grueso son los
> TIF originales de Marcelo Guerrero (hasta 48 MB cada uno), que son el
> respaldo de archivo y no se sirven en la web. Subir solo lo listado arriba.

## 2. Reglas aplicadas (de `\\192.168.86.205\ANIMA-Red\Instruccion_Actualizacion_Versiones`)

1. **Nada de parches regex sobre el HTML en vivo** — todo se generó desde
   scripts versionados (`empalmar_lluvia_calibrada.py`,
   `reemplazar_temperatura_por_era5.py`, `actualizar_fotos_especies.py`,
   `arreglar_selector_ndvi.py`) y la página se sube completa.
2. **Verificar contra lo servido** — tras subir, descargar y comparar. Si no
   coincide, abortar sin tocar el resto.
3. **Registro permanente** — este archivo.
4. **Credenciales solo por entorno** (`CPANEL_USER` / `CPANEL_PASS`), nunca en
   repositorio. Document root: `/home/geografiasagrada/cosmosemiotica.cl`.
5. **No corresponde `publicar_anima.py`**: ese flujo es para releases de
   instaladores, no para páginas del sitio.

## 3. Verificaciones hechas

- [x] JS válido (`node --check`) en todos los bloques inline de ambas páginas.
- [x] Cero errores de consola en el navegador.
- [x] Cero imágenes rotas (62 rutas verificadas en disco + carga real en DOM).
- [x] 11 secciones presentes y numeradas.
- [x] Sin desborde horizontal; tablas con scroll propio; media query a 560 px.
- [x] Enlaces internos: todos resuelven. Los del pie del instrumento pasaron a
      absolutos para que funcione esté en raíz o en subcarpeta.
- [x] Cero apariciones visibles de "prueba de concepto".
- [x] **Paridad de motores**: navegador y Node coinciden en los 4 valores de
      los 62 años (verificado contra el CSV que descargó Alexis: 0 diferencias).

## 4. Estado científico de la corrida publicada

Ronda 15 · lluvia empalmada y calibrada · κ canónicos.

| Criterio | Resultado | Medida |
|---|---|---|
| Trayectorias propias, sin moldes | **cumple** | 61 de 62 únicas, 0 repetidas 3+ |
| Floración documentada > control | **no cumple** | 38,19 vs 38,25 (empate técnico) |
| Megasequía bajo el promedio | **cumple** | 21,2 % vs 40,3 % |
| Correlación con lluvia real | **cumple** | ρ = +0,227 |
| En megasequía domina Cierre | **cumple** | 44,8 % vs 22,9 % |

El criterio que no se cumple está **declarado en la página** con su causa: tres
de los trece años de floración documentada caen dentro de la megasequía, y
varias de esas floraciones fueron en parches locales que un promedio regional
no puede ver. Es un límite del dato de contraste, no del modelo.

## 5. Correcciones de fondo aplicadas en esta tanda

1. **Lluvia sin escalón de fuente** (12-ago). Cambiaba de estación real (CR2) a
   satélite (NASA POWER) en 2019, justo en el tramo de validación. Ahora:
   1966-2018 conserva el dato medido en tierra; 2019+ usa ERA5 corregido por
   sesgo estacional, calibrado contra los 631 meses de solapamiento (sesgo
   0,00 mm/mes, r sube de 0,905 a 0,914).
2. **Temperatura de fuente única** ERA5 1966-2026, 22.139 días sin huecos. Antes
   NASA POWER empezaba en 1981 y los 15 años previos solo podían producir 2 de
   las 4 zonas.
3. **LF dejó de medir la distancia a un control deslizante** de la interfaz
   (era una V: el desierto muerto puntuaba casi el doble que el florido).
4. **Los κ pasaron de medianas a condiciones de posibilidad canónicas**
   (κ_V=0,70 · κ_O=0,20 · κ_LF=0,35, este último del experimento E1 del propio
   EIT-3). Antes la zona medía en qué semana del año caía el tick.
5. **PTC retirado** del cálculo de LF y κ_H: venía de otro instrumento
   ("Pastor del Borde") y era un reflejo fijo, no libertad funcional.

## 6. Pendientes declarados

- Los 8 ejemplares sin determinación taxonómica se publican rotulados
  *"Gyriosomus sp. — en identificación"*, a la espera de Marcelo Guerrero.
- `Lambda` se dispara cuando el error semanal cae a cero (divide por él). Es
  anterior a esta tanda y no afecta la clasificación; conviene revisarlo.
- `prueba_de_concepto_mapa_capas.html` tiene el control de especies inactivo
  (preexistente, verificado contra copia previa). No se publica.
- Los archivos ya llevan el nombre del experimento (`informe-cosmoclima.html` /
  `sim-cosmoclima.html`, convención `informe-`/`sim-` del sitio). La CARPETA
  local sigue llamándose `prueba_de_concepto/`: no afecta a la publicación
  (los archivos van a la raíz del sitio), pero conviene renombrarla por higiene.
