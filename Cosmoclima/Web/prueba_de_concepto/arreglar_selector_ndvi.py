#!/usr/bin/env python3
# Selector NDVI por AÑO + MES sobre la capa MENSUAL (11-ago-2026, a pedido de
# Alexis: "tiene el selector de fecha para supuestamente mostrar las imágenes
# de cobertura vegetal, pero el selector es por año, mes y día... yo lo haría
# por año (y mes) si es posible, porque se vuelve muy complicado saber la
# fecha por día").
#
# Al revisar por qué "no funciona" apareció una causa MÁS DE FONDO que el
# formato del selector, verificada contra el propio servidor de NASA
# (WMTSCapabilities.xml, 1315 capas):
#
#   La capa que estaba puesta, MODIS_Terra_NDVI_8Day, es un compuesto
#   "rolling" que en GIBS SOLO existe entre 2025-02-12 y 2026-08-10 -- 18
#   meses. Pero el selector declaraba min="2000-02-24", prometiendo 26 años.
#   Cualquier fecha anterior a 2025 devuelve HTTP 404 (verificado: 2015-09-06,
#   2017-08-29). Y el propósito declarado del control, en su propia nota, es
#   "cruzar con eventos de floracion documentados" -- que van de 1983 a 2024,
#   casi todos FUERA del rango de esa capa. El control prometía algo que la
#   capa no podía dar.
#
#   La capa MODIS_Terra_L3_NDVI_Monthly cubre 2000-03-01 a 2026-06-01
#   (verificado con imagen real en 2000, 2002, 2011, 2015, 2017, 2024), que
#   alcanza a 10 de los 13 años de floración documentada. Quedan fuera 1983,
#   1991 y 1997 por ser anteriores al satélite MODIS -- eso no lo arregla
#   ningún selector, es el límite del dato real.
#
#   Detalle que costó encontrar: la capa mensual se sirve en
#   GoogleMapsCompatible_Level7, no Level9. Pedirla con Level9 devuelve
#   HTTP 400 "TILEMATRIXSET is invalid for LAYER" -- por eso hay que cambiar
#   la plantilla de URL, no solo el nombre de la capa.
#
# Un producto MENSUAL además calza exactamente con lo que pidió Alexis: el
# día deja de tener sentido porque el dato ya viene agregado por mes.
import re
import os

BASE = os.path.dirname(os.path.abspath(__file__))
HTMLS = ['sim-cosmoclima.html',
         'prueba_de_concepto_mapa_capas.html']

CONTROL_VIEJO = """'<label for="fecha-ndvi">Fecha capa NDVI (NASA GIBS):</label>' +
      '<input type="date" id="fecha-ndvi" value="2026-07-30" min="2000-02-24" max="2026-07-30">' +"""

CONTROL_NUEVO = """'<label for="anio-ndvi">Cobertura vegetal (NDVI, NASA MODIS):</label>' +
      '<div style="display:flex;gap:6px;margin:4px 0;">' +
      '<select id="anio-ndvi" style="flex:1;"></select>' +
      '<select id="mes-ndvi" style="flex:1;"></select>' +
      '</div>' +"""

NOTA_VIEJA = ("'<div class=\"nota\">Fecha exacta de la imagen satelital (util para "
              "cruzar con eventos de floracion documentados).</div>'")
NOTA_NUEVA = ("'<div class=\"nota\">Compuesto MENSUAL de MODIS Terra (2000-2026): el dato "
              "satelital ya viene agregado por mes, por eso no se elige dia. Util para "
              "cruzar con floraciones documentadas -- 1983, 1991 y 1997 quedan fuera "
              "porque son anteriores al satelite MODIS.</div>'")

CAPA_VIEJA = ("'https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Terra_NDVI_8Day/"
              "default/' + fecha + '/GoogleMapsCompatible_Level9/{z}/{y}/{x}.png'")
CAPA_NUEVA = ("'https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Terra_L3_NDVI_Monthly/"
              "default/' + fecha + '/GoogleMapsCompatible_Level7/{z}/{y}/{x}.png'")

ATRIB_VIEJA = "attribution: 'NASA GIBS / MODIS Terra NDVI (8-Day)',"
ATRIB_NUEVA = "attribution: 'NASA GIBS / MODIS Terra NDVI (compuesto mensual)',"

MAXNATIVE_VIEJO = "maxNativeZoom: 9,"
MAXNATIVE_NUEVO = "maxNativeZoom: 7,   // la capa mensual solo se sirve hasta Level7"

INIT_VIEJO = """var capaNDVI = construirCapaNDVI(document.getElementById('fecha-ndvi').value);
capaNDVI.addTo(map);
document.getElementById('fecha-ndvi').addEventListener('change', function (e) {
  map.removeLayer(capaNDVI);
  capaNDVI = construirCapaNDVI(e.target.value);
  capaNDVI.addTo(map);
});"""

INIT_NUEVO = """// Rango real de la capa mensual, verificado contra WMTSCapabilities de NASA:
// 2000-03-01 a 2026-06-01. Se rellenan los dos <select> desde aca para que la
// interfaz no pueda ofrecer una fecha que el servidor no tiene.
var NDVI_ANIO_MIN = 2000, NDVI_ANIO_MAX = 2026;
var NDVI_MESES = ['enero','febrero','marzo','abril','mayo','junio','julio',
                  'agosto','septiembre','octubre','noviembre','diciembre'];
var selAnio = document.getElementById('anio-ndvi');
var selMes = document.getElementById('mes-ndvi');
for (var a = NDVI_ANIO_MAX; a >= NDVI_ANIO_MIN; a--) {
  var o = document.createElement('option'); o.value = a; o.textContent = a; selAnio.appendChild(o);
}
for (var m = 0; m < 12; m++) {
  var om = document.createElement('option'); om.value = m + 1; om.textContent = NDVI_MESES[m]; selMes.appendChild(om);
}
selAnio.value = 2024;  // ultimo anio de floracion documentada con dato satelital
selMes.value = 9;      // septiembre: plena floracion en la ZHCS

function limitarMesesNdvi() {
  // El primer mes disponible es marzo de 2000 y el ultimo, junio de 2026 --
  // se desactivan los meses fuera de rango en vez de dejar elegir un mes que
  // devolveria un tile vacio sin explicacion.
  var a = parseInt(selAnio.value, 10);
  for (var i = 0; i < selMes.options.length; i++) {
    var mes = i + 1;
    var fuera = (a === NDVI_ANIO_MIN && mes < 3) || (a === NDVI_ANIO_MAX && mes > 6);
    selMes.options[i].disabled = fuera;
  }
  if (selMes.options[selMes.selectedIndex] && selMes.options[selMes.selectedIndex].disabled) {
    selMes.value = (a === NDVI_ANIO_MIN) ? 3 : 6;
  }
}
function fechaNdviElegida() {
  return selAnio.value + '-' + String(selMes.value).padStart(2, '0') + '-01';
}
limitarMesesNdvi();
var capaNDVI = construirCapaNDVI(fechaNdviElegida());
capaNDVI.addTo(map);
function refrescarCapaNdvi() {
  limitarMesesNdvi();
  map.removeLayer(capaNDVI);
  capaNDVI = construirCapaNDVI(fechaNdviElegida());
  capaNDVI.addTo(map);
}
selAnio.addEventListener('change', refrescarCapaNdvi);
selMes.addEventListener('change', refrescarCapaNdvi);"""

CAMBIOS = [
    ('control', CONTROL_VIEJO, CONTROL_NUEVO),
    ('nota', NOTA_VIEJA, NOTA_NUEVA),
    ('url capa', CAPA_VIEJA, CAPA_NUEVA),
    ('atribucion', ATRIB_VIEJA, ATRIB_NUEVA),
    ('maxNativeZoom', MAXNATIVE_VIEJO, MAXNATIVE_NUEVO),
    ('init + listeners', INIT_VIEJO, INIT_NUEVO),
]


def main():
    for nombre in HTMLS:
        ruta = os.path.join(BASE, nombre)
        s = open(ruta, encoding='utf-8').read()
        print(f'\n=== {nombre} ===')
        ok = True
        for etiqueta, viejo, nuevo in CAMBIOS:
            if viejo not in s:
                print(f'  [FALLA] no se encontro: {etiqueta}')
                ok = False
                continue
            s = s.replace(viejo, nuevo, 1)
            print(f'  [ok]    {etiqueta}')
        if ok:
            open(ruta, 'w', encoding='utf-8').write(s)
            print('  -> escrito')
        else:
            print('  -> NO se escribio (algun anchor no matcheo)')


if __name__ == '__main__':
    main()
