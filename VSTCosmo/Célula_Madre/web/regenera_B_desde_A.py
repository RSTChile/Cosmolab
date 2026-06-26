#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REGENERA B DESDE A — mantiene a los dos gemelos idénticos salvo por su IDENTIDAD.
ANIMA_A y ANIMA_B son el MISMO organismo (misma arquitectura, mismos órganos); difieren sólo en quién
es 'yo' y quién es 'el otro': id, puertos (7788 propio / 7799 par y viceversa), módulo RC y etiquetas.
Por eso B no se edita a mano: se DERIVA de A con un intercambio simétrico de tokens A↔B. Cualquier mejora
que se haga en WebLive_A.py se propaga a B corriendo esto. Idempotente. Verifica que compile antes de escribir.
"""
import os, re, py_compile, sys   # noqa

AQUI = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(AQUI, "VST_CelulaMadre_WebLive_A.py")
DST = os.path.join(AQUI, "VST_CelulaMadre_WebLive_B.py")

# pares simétricos: cada token de A se cambia por el de B y viceversa, todo a la vez (sin pisarse)
PARES = [
    ("ANIMA_A", "ANIMA_B"),
    ("Organismo A", "Organismo B"),
    ("VST_RC_A", "VST_RC_B"),
    ("WebLive_A", "WebLive_B"),
    ('"7788"', '"7799"'),     # PUERTO propio (A) ↔ par (en B pasa a ser su par)
]

def swap(texto):
    # Intercambio SIMULTÁNEO con LÍMITES DE TOKEN: cada token sólo se cambia cuando NO está pegado a más
    # letras/dígitos/_ . Esto evita corromper substrings (p.ej. 'ANIMA_BIND' contiene 'ANIMA_B', y sin
    # límites se convertía en 'ANIMA_AIND'; 'ANIMA_AUTOSTART' contiene 'ANIMA_A' → 'ANIMA_BUTOSTART').
    # Un solo paso de regex (left-to-right) garantiza que A→B y B→A no se pisen.
    mapping = {}
    for a, b in PARES:
        mapping[a] = b; mapping[b] = a
    def tok(t):
        return r"(?<![A-Za-z0-9_])" + re.escape(t) + r"(?![A-Za-z0-9_])"
    pat = re.compile("|".join(tok(t) for t in mapping))
    return pat.sub(lambda m: mapping[m.group(0)], texto)

def main():
    with open(SRC, encoding="utf-8") as fh:
        src = fh.read()
    out = swap(src)
    # cordura: B debe declararse como ANIMA_B en 7799 y apuntar a su par en 7788
    assert 'ORGANISMO_ID = os.environ.get("VST_ORGANISMO_ID", "ANIMA_B")' in out, "identidad B no quedó bien"
    assert 'VST_PUERTO", "7799"' in out, "puerto propio de B no quedó en 7799"
    tmp = DST + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(out)
    py_compile.compile(tmp, doraise=True)     # si no compila, no piso B
    os.replace(tmp, DST)
    print(f"  B regenerado desde A: {DST}  ({out.count(chr(10))+1} líneas) · compila OK")

if __name__ == "__main__":
    main()
