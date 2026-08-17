#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
buscar_fotos_especies.py -- busca UNA foto real, con licencia Creative Commons
verificable, por cada especie de Gyriosomus, via la API publica de iNaturalist
(taxonomia propia de iNat, no depende de que GBIF tenga el backbone alineado a
nivel de especie -- por eso NO se usa la API de GBIF para esto, se probo primero
y solo encontraba 3 de 44).

No inventa nada: si una especie no tiene ninguna observacion de iNaturalist con
foto de licencia CC, esa especie queda simplemente sin foto -- no se le pone la
foto de otra especie parecida ni una foto generica del genero.

Guarda `datos_fuentes/fotos_especies.csv` (especie, url_foto, licencia,
atribucion, pagina_origen, fuente).
"""
import csv
import json
import os
import time
import urllib.parse
import urllib.request

CARPETA = os.path.dirname(os.path.abspath(__file__))
DATOS_FUENTES = os.path.join(CARPETA, 'datos_fuentes')
OUT_CSV = os.path.join(DATOS_FUENTES, 'fotos_especies.csv')

ESPECIES = [
    "amabilis", "angustus", "atacamensis", "barriai", "batesi", "bridgesi",
    "camanchaca", "chango", "confusus", "coriaceus", "crispaticollis", "curtisi",
    "elongatus", "foveopunctatus", "freyi", "gebieni", "granocostatus",
    "granulipennis", "hoppei", "impressus", "kingi", "kulzeri", "laevigatus",
    "laevis", "leechi", "lucens", "luczotii", "maculatus", "marmoratus",
    "melcheri", "modestus", "multigranulosus", "nigrociliatus", "parvus",
    "paulseni", "penai", "penicilliger", "planatus", "planicollis", "pumilus",
    "reedi", "resplendens", "subrugatus", "whitei",
]

LICENCIAS_CC = {
    "cc0": "CC0 1.0 (dominio publico)",
    "cc-by": "CC BY 4.0",
    "cc-by-nc": "CC BY-NC 4.0",
    "cc-by-sa": "CC BY-SA 4.0",
    "cc-by-nc-sa": "CC BY-NC-SA 4.0",
    "cc-by-nd": "CC BY-ND 4.0",
    "cc-by-nc-nd": "CC BY-NC-ND 4.0",
}


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Cosmoclima-Gyriosomus-mapa/1.0 (uso educativo, con atribucion)"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def buscar_foto_inat(epiteto):
    nombre = f"Gyriosomus {epiteto}"
    for quality in ("research", "any"):
        params = {
            "taxon_name": nombre, "photos": "true", "per_page": 10,
            "order_by": "votes", "quality_grade": quality,
        }
        d = get("https://api.inaturalist.org/v1/observations?" + urllib.parse.urlencode(params))
        for obs in d.get("results", []):
            # confirmar que el nombre cientifico de la observacion es EXACTO,
            # no un genero emparentado que iNat devolvio por texto libre
            taxon_nombre = (obs.get("taxon") or {}).get("name", "")
            if taxon_nombre.strip().lower() != nombre.lower():
                continue
            for foto in obs.get("photos", []):
                lic = foto.get("license_code")
                if not lic or lic not in LICENCIAS_CC:
                    continue
                url_original = foto["url"].replace("square.", "medium.").replace("/square", "/medium")
                return {
                    "url_foto": url_original,
                    "licencia": LICENCIAS_CC[lic],
                    "atribucion": (obs.get("user") or {}).get("login", "sin dato"),
                    "pagina_origen": obs.get("uri", ""),
                }
    return None


def main():
    filas = []
    encontradas = 0
    for epiteto in ESPECIES:
        try:
            foto = buscar_foto_inat(epiteto)
        except Exception as e:
            print(f"{epiteto}: ERROR ({e})")
            time.sleep(1)
            continue
        if not foto:
            print(f"{epiteto}: sin foto CC en iNaturalist con nombre exacto")
        else:
            encontradas += 1
            print(f"{epiteto}: FOTO -> {foto['url_foto']} ({foto['licencia']}, c iNaturalist/{foto['atribucion']})")
            filas.append({
                "especie": epiteto,
                "url_foto": foto["url_foto"],
                "licencia": foto["licencia"],
                "atribucion": foto["atribucion"],
                "pagina_origen": foto["pagina_origen"],
                "fuente": "iNaturalist (api.inaturalist.org), observacion con foto CC verificada por nombre exacto de especie",
            })
        time.sleep(1)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["especie", "url_foto", "licencia", "atribucion", "pagina_origen", "fuente"])
        w.writeheader()
        w.writerows(filas)
    print(f"\n{encontradas} de {len(ESPECIES)} especies con foto real encontrada.")
    print(f"Guardado: {OUT_CSV}")


if __name__ == "__main__":
    main()
