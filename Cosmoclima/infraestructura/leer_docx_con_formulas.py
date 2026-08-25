"""Extrae el texto de un .docx INCLUYENDO las ecuaciones OMML.

python-docx ignora los bloques <m:oMath> (las ecuaciones de Word), que es
justamente donde el Anexo A.5 del RMD guarda las fórmulas. Este lector recorre
el XML en orden y traduce cada ecuación a una línea de texto lineal legible:
fracciones como (a)/(b), subíndices como _{x}, superíndices como ^{x},
raíces como sqrt(...), sumatorias como SUM_{desde}^{hasta}(...).
"""
import sys, re
from xml.etree import ElementTree as ET

W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
M = '{http://schemas.openxmlformats.org/officeDocument/2006/math}'


def omml(nodo):
    """Traduce un nodo de ecuación OMML a texto lineal."""
    t = nodo.tag
    if t == M + 't':                       # texto literal dentro de la ecuación
        return nodo.text or ''
    if t == M + 'r':                       # "run" matemático
        return ''.join(omml(h) for h in nodo)
    if t == M + 'f':                       # fracción
        num = den = ''
        for h in nodo:
            if h.tag == M + 'num':
                num = ''.join(omml(x) for x in h)
            elif h.tag == M + 'den':
                den = ''.join(omml(x) for x in h)
        return f'({num})/({den})'
    if t == M + 'sSub':                    # subíndice
        base = sub = ''
        for h in nodo:
            if h.tag == M + 'e':
                base = ''.join(omml(x) for x in h)
            elif h.tag == M + 'sub':
                sub = ''.join(omml(x) for x in h)
        return f'{base}_{{{sub}}}'
    if t == M + 'sSup':                    # superíndice
        base = sup = ''
        for h in nodo:
            if h.tag == M + 'e':
                base = ''.join(omml(x) for x in h)
            elif h.tag == M + 'sup':
                sup = ''.join(omml(x) for x in h)
        return f'{base}^{{{sup}}}'
    if t == M + 'sSubSup':                 # sub y superíndice juntos
        base = sub = sup = ''
        for h in nodo:
            if h.tag == M + 'e':
                base = ''.join(omml(x) for x in h)
            elif h.tag == M + 'sub':
                sub = ''.join(omml(x) for x in h)
            elif h.tag == M + 'sup':
                sup = ''.join(omml(x) for x in h)
        return f'{base}_{{{sub}}}^{{{sup}}}'
    if t == M + 'rad':                     # raíz
        deg = e = ''
        for h in nodo:
            if h.tag == M + 'deg':
                deg = ''.join(omml(x) for x in h)
            elif h.tag == M + 'e':
                e = ''.join(omml(x) for x in h)
        return f'raiz{deg}({e})' if deg else f'sqrt({e})'
    if t == M + 'd':                       # delimitadores (paréntesis)
        return '(' + ''.join(omml(h) for h in nodo) + ')'
    if t == M + 'nary':                    # sumatoria / producto / integral
        op = sub = sup = e = ''
        for h in nodo:
            if h.tag == M + 'naryPr':
                for p in h:
                    if p.tag == M + 'chr':
                        op = p.get(M + 'val', '')
            elif h.tag == M + 'sub':
                sub = ''.join(omml(x) for x in h)
            elif h.tag == M + 'sup':
                sup = ''.join(omml(x) for x in h)
            elif h.tag == M + 'e':
                e = ''.join(omml(x) for x in h)
        nombre = {'∑': 'SUM', '∏': 'PROD', '∫': 'INT'}.get(op, op or 'SUM')
        return f'{nombre}_{{{sub}}}^{{{sup}}}({e})'
    # cualquier otro contenedor: bajar sin decorar
    return ''.join(omml(h) for h in nodo)


def texto_parrafo(p):
    """Texto de un párrafo, intercalando las ecuaciones donde aparecen."""
    partes = []
    for h in p.iter():
        if h.tag == W + 't':
            partes.append(h.text or '')
        elif h.tag == M + 'oMath':
            partes.append('  ⟦FÓRMULA: ' + omml(h).strip() + '⟧')
    return ''.join(partes)


def main(ruta):
    # un .docx es un zip; el cuerpo del documento vive en word/document.xml
    import zipfile
    with zipfile.ZipFile(ruta) as z:
        raiz = ET.fromstring(z.read('word/document.xml'))
    cuerpo = raiz.find(W + 'body')
    for hijo in cuerpo:
        etq = hijo.tag.split('}')[-1]
        if etq == 'p':
            # un párrafo que ES una ecuación suelta también cae acá
            txt = texto_parrafo(hijo).strip()
            if txt:
                print(txt)
        elif etq == 'tbl':
            print('\n--- TABLA ---')
            for fila in hijo.findall(W + 'tr'):
                celdas = []
                for c in fila.findall(W + 'tc'):
                    celdas.append(' '.join(
                        texto_parrafo(p).strip() for p in c.findall(W + 'p')).strip())
                print(' | '.join(celdas))
            print('--- FIN TABLA ---\n')
        elif etq == 'oMathPara':           # ecuación como bloque propio
            for eq in hijo.findall(M + 'oMath'):
                print('  ⟦FÓRMULA: ' + omml(eq).strip() + '⟧')


if __name__ == '__main__':
    main(sys.argv[1])
