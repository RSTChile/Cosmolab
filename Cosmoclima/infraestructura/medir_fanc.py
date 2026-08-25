"""
MEDIR EL FANC CONTRA EL DERECHO, EN CUATRO GRADOS
===================================================

FANC = Fragilidad ante Ataques No Convencionales.

DECISIÓN (Alexis, 23-ago-2026): re-medir con los cuatro grados antes de congelar
ningún corte, «porque el FANC pesa 0,4 en la Peh y arrastra todo».

★ QUÉ MIDE ESTA COLUMNA, DICHO CON PRECISIÓN
----------------------------------------------
No mide «qué tan fácil es romperlo». Mide **qué tan grave sería que alguien lo
atacara deliberadamente** — y esa gravedad ya está graduada, con siglas de
tratado, en el Protocolo I adicional a los Convenios de Ginebra (1977), que
Chile ratificó. No hay que inventar una escala: hay que leer la que existe.

★ LO QUE ESTABA MAL, MEDIDO
-----------------------------
La medición del 22-ago mejoró muchísimo (de 802 «Alta» a 15/587/243) pero tenía
tres defectos que se vieron al auditarla el 23-ago:

  1. **Le faltaba un grado.** Las obras del artículo 56 —nucleares y presas—
     quedaron en «Alta», el mismo cajón que todo lo demás protegido. El
     artículo 56 es protección ESPECIAL: es un grado aparte.
  2. **El Alimentario estaba al revés.** 31 de 44 en «Baja», cuando el artículo
     54 nombra expresamente los alimentos y las cosechas como bienes
     indispensables para la supervivencia de la población civil.
  3. **Se asignó por SECTOR, no por ítem.** Trece de veinte sectores tenían los
     44 ítems con el mismo valor, así que dentro de un sector el FANC no
     distinguía nada. Y produjo incoherencias: «Estructuras de Presas» quedó en
     «Alta» mientras «Represas Hidroeléctricas» quedó en «Media», siendo ambas
     presas.

★ LA REGLA, DECLARADA ANTES DE CORRER
---------------------------------------
    Muy Alta   Art. 56 · obras que contienen fuerzas peligrosas: centrales
               nucleares, PRESAS y diques. Su rasgo común es que el daño no lo
               causa el ataque sino lo que el ataque libera.
    Alta       Art. 54 · bienes indispensables para la supervivencia civil
               (agua potable, alimentos, cosechas) · Art. 12 · unidades
               sanitarias · y las instalaciones de apoyo situadas EN los sitios
               del art. 56, que el propio art. 56.1 protege por extensión.
    Media      Art. 52 · bienes de carácter civil en general.
    Baja       objetivo militar potencial o de doble uso (art. 52.2): industria
               de defensa, química e industrial pesada.

La Ley 21.663 de Ciberseguridad informa el límite Alta/Media en lo digital, pero
**no manda**: la Matriz es nuestra (instrucción de Alexis del 21-ago).

★ CÓMO SE EVITA EL ERROR DE «EM-PRESA-RIALES»
-----------------------------------------------
El 21-ago, marcar por la palabra «presa» metió «Servidores Em**presa**riales» y
«Personal de TI (Presas)» dentro del artículo 56. Aquí se usan **fronteras de
palabra** y una lista de exclusión explícita, y el script **imprime todo lo que
excluyó** para que la exclusión también se pueda auditar.

USO
---
    ../.venv-esa/bin/python medir_fanc.py             # mide y reporta
    ../.venv-esa/bin/python medir_fanc.py --escribir  # deja el CSV
"""

import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

AQUI = Path(__file__).resolve().parent
DATOS = AQUI / "datos"
FUENTE = DATOS / "micr_sharepoint_ultimo.csv"
SALIDA = DATOS / "fanc_medido_4grados.csv"

# ── artículo 56 · obras que contienen fuerzas peligrosas ─────────────────────
# La sustancia peligrosa misma: el material nuclear, o el agua/relave retenido.
ART56 = [r"\bnuclear", r"\bnucleares", r"\breactor", r"\breactores",
         r"\bradiactiv", r"\bradiolog",
         r"\bpresa\b", r"\bpresas\b", r"\brepresa", r"\bdique",
         r"\bembalse", r"\btranque", r"\brelave"]

# ★ Exclusiones. Sin esto, «Servidores Empresariales» entra en el artículo 56.
# Se comprueban ANTES que nada y el script las imprime todas.
NO_ES_LA_OBRA = [r"empresa", r"empresas", r"empresarial",
                 r"\bpersonal\b", r"\bcapacitaci", r"\bseguro",
                 r"\bnormativa", r"\bprotocolo"]

# Apoyo situado EN un sitio del art. 56: protegido por extensión, pero no es la
# obra. Hacen falta DOS condiciones, no una.
#
# ⚠️ Con el paréntesis solo, «Plantas de Generación de Energía (Nuclear)» —el
# caso que el artículo 56 nombra por su nombre— caía en «apoyo». Por eso se
# exige además que el ítem sea una FUNCIÓN de apoyo, no la instalación misma.
SITIO_56 = [r"\(nuclear\)", r"\(presas\)", r"\(represas\)"]
FUNCION_APOYO = [r"red(es)? de comunicaci", r"centros? de gesti",
                 r"veh[ií]culos?", r"centros? de capacitaci",
                 r"sistemas? de monitoreo", r"oficinas?"]

# ── artículo 54 · indispensables para la supervivencia ───────────────────────
ART54 = [r"agua potable", r"potabiliza", r"\bpotable", r"captaci[oó]n de agua",
         r"\bacueducto", r"\bpozo", r"\bpozos", r"\bacu[ií]fero",
         r"\bmanantial", r"desaliniza", r"\briego", r"\bcosecha",
         r"\balimento", r"\balimentos", r"\bagr[ií]cola", r"\bcultivo",
         r"\bganad", r"\bsilo", r"\bfrigor[ií]fic", r"\bmatadero",
         r"\bpanader", r"\babastecimiento de alimentos"]
# Art. 12 · unidades sanitarias
ART12 = [r"\bhospital", r"\bcl[ií]nic", r"\bconsultorio", r"\bcesfam",
         r"\bposta\b", r"\bambulancia", r"\bsanitari", r"\bm[eé]dic",
         r"\bfarmac", r"\bvacuna", r"\bbanco de sangre"]

# ── sectores: el valor por omisión cuando ningún rasgo del ítem decide ───────
SECTOR_BASE = {
    "Nuclear": "Alta",             # el apoyo; la obra sube por palabra clave
    "Represas": "Alta",
    "Hídrico": "Alta",             # agua de consumo, art. 54
    "Alimentario": "Alta",         # ★ art. 54 nombra los alimentos
    "Salud": "Alta",               # ★ art. 12
    "Energía": "Media",
    "Telecomunicaciones": "Media",
    "Comunicaciones": "Media",
    "Servicios de Emergencia": "Alta",
    "Seguridad": "Media",
    "Gobierno": "Media",
    "Transporte": "Media",
    "Financiero": "Media",
    "Educación": "Media",          # ★ art. 52: bien civil, no objetivo
    "Comercial": "Media",
    "Protección Social": "Alta",
    "Tecnologías Informáticas": "Media",
    "Industria de Defensa": "Baja",   # ★ art. 52.2: objetivo militar legítimo
    "Químico": "Baja",
    "Industrial": "Baja",
}
ORDEN = ("Muy Alta", "Alta", "Media", "Baja")


def busca(patrones, texto):
    return next((p for p in patrones if re.search(p, texto)), None)


def clasificar(elemento, sector):
    """Devuelve (grado, artículo, motivo). Regla declarada arriba, sin excepciones."""
    e = elemento.lower()

    excl = busca(NO_ES_LA_OBRA, e)
    if excl:
        base = SECTOR_BASE.get(sector, "Media")
        return base, "—", f"excluido de las palabras clave por «{excl}»"

    # ★ La sustancia manda sobre el paréntesis: primero se pregunta si el ítem
    # ES la obra peligrosa, y sólo si además es una función de apoyo se degrada.
    p = busca(ART56, e)
    if p:
        if busca(SITIO_56, e) and busca(FUNCION_APOYO, e):
            return "Alta", "56.1", "apoyo situado en un sitio de fuerzas peligrosas"
        return "Muy Alta", "56", f"obra con fuerzas peligrosas («{p}»)"

    if busca(SITIO_56, e) and busca(FUNCION_APOYO, e):
        return "Alta", "56.1", "apoyo situado en un sitio de fuerzas peligrosas"

    p = busca(ART12, e)
    if p:
        return "Alta", "12", f"unidad sanitaria («{p}»)"

    p = busca(ART54, e)
    if p:
        return "Alta", "54", f"indispensable para la supervivencia («{p}»)"

    base = SECTOR_BASE.get(sector, "Media")
    art = {"Muy Alta": "56", "Alta": "54", "Media": "52", "Baja": "52.2"}[base]
    return base, art, f"por sector «{sector}»"


def main():
    filas = [x for x in csv.DictReader(FUENTE.open(encoding="utf-8")) if x["n"]]
    print("=" * 78)
    print("FANC MEDIDO EN CUATRO GRADOS · Protocolo I de Ginebra")
    print("=" * 78)
    print(f"\n  fuente : {FUENTE.name} · {len(filas)} filas\n")

    out, excluidos, por_art = [], [], Counter()
    for x in filas:
        g, art, motivo = clasificar(x["elemento"], x["Sector"])
        out.append(dict(n=int(float(x["n"])), elemento=x["elemento"],
                        Sector=x["Sector"], FANC_antes=x["FANC"],
                        FANC=g, articulo=art, motivo=motivo))
        por_art[art] += 1
        if "excluido" in motivo:
            excluidos.append(out[-1])

    # ── el reparto ───────────────────────────────────────────────────────────
    antes = Counter(x["FANC"] for x in filas)
    ahora = Counter(o["FANC"] for o in out)
    print("  grado        antes    ahora")
    for g in ORDEN:
        print(f"  {g:<11} {antes.get(g,0):6d} {ahora.get(g,0):8d}")
    print(f"\n  por artículo: " + " · ".join(f"art.{a}:{n}"
                                            for a, n in sorted(por_art.items())))

    # ── varianza: lo que hace que los pesos sean estimables ──────────────────
    NUM = {"Muy Alta": 4, "Alta": 3, "Media": 2, "Baja": 1}
    for etiqueta, vals in (("antes", [NUM[x["FANC"]] for x in filas]),
                           ("ahora", [NUM[o["FANC"]] for o in out])):
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / len(vals)
        print(f"  varianza {etiqueta}: {var:.4f}")

    # ── ¿distingue DENTRO de cada sector? ────────────────────────────────────
    print("\n" + "=" * 78)
    print("¿DISTINGUE DENTRO DEL SECTOR? · el defecto que se venía a corregir")
    print("=" * 78 + "\n")
    uni_antes = uni_ahora = 0
    for s in sorted({x["Sector"] for x in filas}):
        a = {x["FANC"] for x in filas if x["Sector"] == s}
        b = {o["FANC"] for o in out if o["Sector"] == s}
        uni_antes += len(a) == 1
        uni_ahora += len(b) == 1
        if len(b) > 1:
            c = Counter(o["FANC"] for o in out if o["Sector"] == s)
            print(f"  {s:<24} " + " · ".join(f"{g}:{c[g]}" for g in ORDEN if c[g]))
    print(f"\n  sectores con un solo valor · antes {uni_antes}/20 · "
          f"ahora {uni_ahora}/20")

    # ── auditoría de las exclusiones ─────────────────────────────────────────
    print("\n" + "=" * 78)
    print("EXCLUSIONES · lo que NO entró pese a contener la palabra")
    print("=" * 78 + "\n")
    if not excluidos:
        print("  (ninguna)")
    for o in excluidos:
        print(f"  {o['n']:>4}  {o['elemento'][:48]:<48} → {o['FANC']}")

    # ── las obras del artículo 56, una por una ───────────────────────────────
    print("\n" + "=" * 78)
    print("ARTÍCULO 56 · las obras que contienen fuerzas peligrosas")
    print("=" * 78 + "\n")
    a56 = [o for o in out if o["articulo"] == "56"]
    print(f"  {len(a56)} ítems. Antes eran «Alta» sólo 15 y mezclados con el resto.\n")
    for o in a56[:28]:
        marca = " ←cambia" if o["FANC_antes"] != o["FANC"] else ""
        print(f"  {o['n']:>4}  {o['Sector']:<12.12} {o['elemento'][:42]:<42} "
              f"{o['FANC_antes']:<6}→ {o['FANC']}{marca}")
    if len(a56) > 28:
        print(f"       … y {len(a56)-28} más")

    cambian = sum(1 for o in out if o["FANC"] != o["FANC_antes"])
    print(f"\n  ★ cambian de grado {cambian} de {len(out)} filas "
          f"({100*cambian/len(out):.1f} %)")

    if "--escribir" in sys.argv:
        with SALIDA.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
            w.writeheader()
            w.writerows(out)
        print(f"\n  escrito: {SALIDA.name}")
    else:
        print("\n  (nada escrito · corre con --escribir para dejar el CSV)")
    print("\n  ★ NO se sube a SharePoint sin que el director lo revise.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
