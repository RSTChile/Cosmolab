#!/usr/bin/env python3
# Reemplaza las fotos de especie por las láminas de Marcelo Guerrero
# (11-ago-2026, a pedido de Alexis: "Reemplaza las especies que tienen
# fotografías en la página, por las que están con fondo blanco... recortar el
# fondo blanco para que queden como png con el fondo transparente").
#
# Las láminas ya recortadas viven en imagenes/web/<especie>.png (fondo
# transparente, generadas con ImageMagick: flood-fill desde el borde + una
# pasada de blanco puro al 3% para los blancos que quedaban ATRAPADOS entre
# las patas y el cuerpo, que el flood-fill no alcanza por no estar conectados
# con el borde).
#
# El mismo bloque `especiesData` está DUPLICADO en los dos HTML (el
# instrumento EIT-3 lleva el mapa embebido), así que se actualizan ambos con
# el mismo script para que no se desincronicen.
#
# Qué NO hace: no toca las especies cuya única imagen disponible es una foto
# de campo (freyi, parvus, resplendens en sus versiones .jpg) -- ahí se
# conserva la foto de iNaturalist que ya estaba. Tampoco asigna las láminas
# sin identificación de especie: quedan en imagenes/web/sin-identificar-*.png
# esperando que Marcelo/Alexis las identifiquen.
import re
import sys
import os

BASE = os.path.dirname(os.path.abspath(__file__))
HTMLS = [
    'sim-cosmoclima.html',
    'prueba_de_concepto_mapa_capas.html',
]

# especie -> archivo original entregado por Marcelo (para dejar el rastro)
LAMINAS = {
    'batesi':          'Gyriosomus batesi.tif',
    'bridgesi':        'Gyriosomus bridgesi.tif',
    'crispaticollis':  'Gyriosomus crispaticollis (Termas de Socos) clonado.tif',
    'foveopunctatus':  'G. foveopunctatua (Cta. El Espino) hembra 1.tif',
    'hoppei':          'G. hopei hembra copia.tif',
    'laevis':          'Gyriosomus laevis 2.tif',
    'luczotii':        'Gyriosomus  luczotiCta Los Porotitos copia.tif',
    'marmoratus':      'Gyriosomus Marmoratus.tif',
    'melcheri':        'Gyriosomus melcheri (Pto. Oscuro) hembra 1.jpg',
    'multigranulosus': 'Gyriosomus multigranulosus (Pte El Teniente) Macho 1.tif',
    'nigrociliatus':   'Gyriosomus nigrociliatus Hembra, vista dorsal.tif',
    'paulseni':        'Gyriosomus paulseni (Canela Baja) macho 1.jpg',
    'resplendens':     'Gyriosomus resplendens.tif',
    'subrugatus':      'Gyriosomus subrugatus Fairmaire, 1876.tif',
}


def bloque_foto(especie, original):
    """Genera el objeto `foto` con fuente=lamina, para que el render distinga
    una lámina de Marcelo de una foto CC de iNaturalist."""
    orig = original.replace('"', '\\"')
    return ('{"url": "imagenes/web/%s.png", "licencia": "Cortesia del autor", '
            '"atribucion": "Marcelo Guerrero", "fuente": "lamina", '
            '"original": "%s"}' % (especie, orig))


def reemplazar_en(texto, especie, nuevo):
    """Reemplaza el "foto": ... del bloque de ESA especie (delimitado por el
    comienzo de su clave y el de la siguiente especie)."""
    pat_ini = re.compile(r'"%s":\s*\{"n":\s*\d+,' % re.escape(especie))
    m = pat_ini.search(texto)
    if not m:
        return texto, 'NO ENCONTRADA'
    # el bloque de esta especie termina donde empieza la siguiente clave "xxx": {"n":
    sig = re.compile(r'"(\w+)":\s*\{"n":\s*\d+,').search(texto, m.end())
    fin = sig.start() if sig else len(texto)
    frag = texto[m.start():fin]
    nfrag, n = re.subn(r'"foto":\s*(null|\{[^{}]*\})', '"foto": ' + nuevo, frag, count=1)
    if n == 0:
        return texto, 'SIN CAMPO foto'
    antes = 'null' if re.search(r'"foto":\s*null', frag) else 'iNaturalist'
    return texto[:m.start()] + nfrag + texto[fin:], antes


def main():
    for nombre in HTMLS:
        ruta = os.path.join(BASE, nombre)
        s = open(ruta, encoding='utf-8').read()
        original = s
        print(f'\n=== {nombre} ===')
        for especie in sorted(LAMINAS):
            s, antes = reemplazar_en(s, especie, bloque_foto(especie, LAMINAS[especie]))
            print(f'  {especie:<18} {antes:>12}  ->  lamina Marcelo Guerrero')
        # kingi: mover la ruta a la carpeta de imagenes
        s = s.replace("'Gyriosomus%20kingi.png'", "'imagenes/Gyriosomus%20kingi.png'")
        s = s.replace('"Gyriosomus%20kingi.png"', '"imagenes/Gyriosomus%20kingi.png"')
        if s != original:
            open(ruta, 'w', encoding='utf-8').write(s)
            print(f'  -> escrito ({len(s)-len(original):+d} caracteres)')
        else:
            print('  -> SIN CAMBIOS')


if __name__ == '__main__':
    main()
