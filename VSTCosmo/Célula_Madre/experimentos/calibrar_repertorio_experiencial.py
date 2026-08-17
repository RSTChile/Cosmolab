#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
calibrar_repertorio_experiencial.py — el estatuto de cada palabra por lo que le HACE al organismo
================================================================================
Calibra el repertorio vocal SIN leer el sentido del espectro (anti-Shannon). Para cada palabra:

    snapshot basal → reproducir la palabra al organismo → ventana de respuesta → snapshot final
    → Δvector fisiológico → régimen inducido (JARDIN_FERTIL/CERRADO/COLAPSO/SELVA_HOSTIL)
    → nombre observacional PROVISIONAL → guardar historia

El régimen y el nombre NO son significado humano literal: son etiquetas observacionales del cuadrante
de respuesta que la palabra produjo en ESTE organismo. El nombre puede cambiar con la historia.

Uso:
    venv/bin/python3 experimentos/calibrar_repertorio_experiencial.py --org A --repeticiones 5 --ventana 8
    flags: --org A | A,B,C,D   --solo-innatas   --solo-propias   --repeticiones N   --ventana K
Genera ~/Downloads/CALIBRACION_LEXICA_<ts>/ con resumen.md + 3 CSV + json.

NOTA honesta (v1): se calibra contra un organismo INSTANCIADO FRESCO por --org. Como A/B/C/D comparten
arquitectura, un organismo fresco responde por su FÍSICA; la divergencia entre organismos (criterio #4)
emerge cuando se calibra contra la HISTORIA viva de cada uno (restaurar estado persistido) — refinamiento
documentado, no incluido aquí. El Calibrador ya es por-organismo y está listo para divergir.
================================================================================
"""
from __future__ import annotations
import os, sys, json, argparse, time
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__)); RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)
[sys.path.insert(0, os.path.join(RAIZ, _d)) for _d in ("genoma", "campo", "organelos", "diada", "web", "audio")
 if os.path.isdir(os.path.join(RAIZ, _d))]
import VST_CelulaMadre_WebLive_A as A
from VST_CalibradorLexicoExperiencial import (CalibradorLexicoExperiencial, snapshot_fisiologico,
                                              establecer_convencion, ConvencionLexica)

SR = A.SR; DT = A.DT


def _silencio(seg=0.6):
    n = int(SR * seg); return (np.zeros(n), np.zeros(n))


def _word_estereo(audio, seg=None):
    """Una palabra (mono) presentada a los dos oídos (sin lateralidad: el sentido no es la posición)."""
    a = np.asarray(audio, dtype=np.float64)
    if seg is not None:
        n = int(SR * seg)
        a = np.tile(a, int(np.ceil(n / max(1, len(a)))))[:n] if len(a) else np.zeros(n)
    return (a.copy(), a.copy())


def _fila_actual(cel):
    try:
        return A._fila(cel, None)
    except Exception:
        return {}


def _calibrar_una(voz, pasos_basal=25, pasos_ventana=8):
    """Una escucha: organismo fresco → settle (basal) → oír la palabra → ventana → final. Δvector."""
    cel = A.cmf.celula_madre_funcional(_silencio(), binaural=True)
    for _ in range(pasos_basal):
        cel.vivir_un_paso(DT)
    basal = _fila_actual(cel)
    soma = cel.organelos.get("soma") if hasattr(cel, "organelos") else None
    try:
        soma_real = getattr(soma, "realimentar", None) or getattr(cel, "realimentar", None)
        if soma_real:
            soma_real(_word_estereo(voz["audio"], seg=max(0.4, pasos_ventana * DT)), True)
    except Exception:
        pass
    final = {}
    for _ in range(pasos_ventana):
        cel.vivir_un_paso(DT)
        final = _fila_actual(cel)
    return basal, final


def _voces_objetivo(args):
    """Banco de voces a calibrar, filtrado por --solo-innatas / --solo-propias."""
    com = A.ORGANO_COMUNICACION
    if com is None:
        return []
    voces = list(getattr(com, "_voces", []))
    def es_propia(v):
        return v.get("afecto_origen") not in ("curado", "provisional") or v.get("propia") or str(v.get("label", "")).startswith(("palabra_", "apr_", "fon_"))
    if args.solo_innatas:
        voces = [v for v in voces if v.get("afecto_origen") == "curado"]
    elif args.solo_propias:
        voces = [v for v in voces if es_propia(v)]
    return voces


def _csv(path, header, filas):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for fila in filas:
            f.write(",".join(str(x) for x in fila) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", default="A", help="A | A,B,C,D")
    ap.add_argument("--solo-innatas", action="store_true")
    ap.add_argument("--solo-propias", action="store_true")
    ap.add_argument("--repeticiones", type=int, default=5)
    ap.add_argument("--ventana", type=int, default=8)
    args = ap.parse_args()
    orgs = [o.strip().upper() for o in args.org.split(",") if o.strip()]

    voces = _voces_objetivo(args)
    if not voces:
        print("Sin voces que calibrar (¿banco vacío o filtro sin coincidencias?)."); return

    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(os.stat(__file__).st_mtime + time.time() % 1))
    out = os.path.join(os.path.expanduser("~/Downloads"), f"CALIBRACION_LEXICA_{ts}")
    os.makedirs(out, exist_ok=True)
    print(f"Calibrando {len(voces)} palabras × {args.repeticiones} rep × {len(orgs)} org → {out}")

    todo = {}          # org -> CalibradorLexicoExperiencial (COHORTE de referencia)
    matriz = {}        # (org, label) -> {regimen: conteo}  (matices)
    for org in orgs:
        cal = CalibradorLexicoExperiencial(org); todo[org] = cal
        for v in voces:
            vid = str(v.get("label"))
            for _ in range(max(1, args.repeticiones)):
                basal, final = _calibrar_una(v, pasos_ventana=max(2, args.ventana))
                obs = cal.observar(vid, basal, final, t=final.get("t", 0.0))
                matriz.setdefault((org, vid), {}).setdefault(obs["regimen"], 0)
                matriz[(org, vid)][obs["regimen"]] += 1
            cal.consolidar(vid, ventana=max(args.repeticiones, 8))   # matiz per-organismo

    # ---- CONVENCIÓN LÉXICA COMPARTIDA (criterio de Alexis): UNA fuente de verdad para el nombre ----
    # La cohorte de referencia = los organismos calibrados. Se vota un estatuto común y se PROPAGA al
    # repertorio compartido. Recalibración colectiva = volver a correr esto (anota historial_nombres).
    ruta_conv = os.path.join(os.environ.get("ANIMA_LEXICO_DIR") or os.path.join(RAIZ, "lexico_comun"),
                             "convencion_lexica.json")
    previa = ConvencionLexica(ruta_conv).lexico
    convencion = establecer_convencion(todo, ventana=max(args.repeticiones, 8), convencion_previa=previa)
    for cal in todo.values():
        cal.aplicar_convencion(convencion)                          # propagación a A/B/C/D
    CL = ConvencionLexica(); CL.actualizar(convencion); CL.guardar(ruta_conv)
    print(f"\nConvención compartida ({len(convencion)} palabras) → {ruta_conv}")
    for vid, est in sorted(convencion.items()):
        print(f"  {vid:18s} → {est['regimen_experiencial']:14s} '{est['nombre_observacional']}' "
              f"({est['confianza']}, acuerdo {est['acuerdo_cohorte']})")

    # ---- salidas ----
    # convención (compartida): UNA fila por palabra, nombre común
    _csv(os.path.join(out, "convencion_lexica.csv"),
         ["voz_id", "regimen", "nombre_comun", "campo_semantico", "confianza", "acuerdo_cohorte", "n_organismos"],
         [(vid, e["regimen_experiencial"], e["nombre_observacional"], e["campo_semantico"], e["confianza"],
           e["acuerdo_cohorte"], e["n_organismos"]) for vid, e in sorted(convencion.items())])
    # matices por organismo: el régimen propio de cada uno (no cambia el nombre común)
    matices = []
    for org in orgs:
        for v in voces:
            vid = str(v.get("label")); m = todo[org].matiz(vid)
            if m:
                matices.append((org, vid, m.get("regimen_propio", ""), m.get("confianza_propia", ""),
                                convencion.get(vid, {}).get("nombre_observacional", ""), v.get("afecto_origen", "")))
    _csv(os.path.join(out, "matices_por_organismo.csv"),
         ["org", "voz_id", "regimen_propio", "confianza_propia", "nombre_comun", "afecto_origen"], matices)
    # matriz palabra×régimen (conteo de observaciones, por organismo)
    _csv(os.path.join(out, "matriz_palabra_regimen.csv"),
         ["org", "voz_id", "JARDIN_FERTIL", "CERRADO", "COLAPSO", "SELVA_HOSTIL"],
         [(org, vid, c.get("JARDIN_FERTIL", 0), c.get("CERRADO", 0), c.get("COLAPSO", 0), c.get("SELVA_HOSTIL", 0))
          for (org, vid), c in sorted(matriz.items())])
    # json completo: convención + biografías por organismo
    with open(os.path.join(out, "calibracion_lexica_experiencial.json"), "w", encoding="utf-8") as f:
        json.dump({"convencion": convencion, "organismos": {org: cal.snapshot() for org, cal in todo.items()}},
                  f, ensure_ascii=False, indent=1, default=float)
    # resumen.md
    with open(os.path.join(out, "resumen.md"), "w", encoding="utf-8") as f:
        f.write("# Calibración léxica experiencial — convención compartida\n\n")
        f.write(f"- Palabras: {len(voces)} · repeticiones: {args.repeticiones} · ventana: {args.ventana} pasos "
                f"· cohorte de referencia: {', '.join(orgs)}\n")
        f.write("- **El sentido no está en la señal.** El régimen/nombre son OBSERVACIONALES y PROVISIONALES; "
                "describen el cuadrante de respuesta que la palabra produjo, no significado humano literal.\n")
        f.write("- **Convención compartida:** el nombre es UNO solo para A/B/C/D (una fuente de verdad), para que "
                "puedan entenderse. Los matices por organismo se registran pero NO cambian el nombre común; "
                "sólo una nueva recalibración colectiva lo hace.\n\n")
        f.write("## Convención (léxico común)\n\n| palabra | régimen | nombre común | campo | confianza | acuerdo |\n|---|---|---|---|---|---|\n")
        for vid, e in sorted(convencion.items()):
            f.write(f"| {vid} | {e['regimen_experiencial']} | {e['nombre_observacional']} | {e['campo_semantico']} | {e['confianza']} | {e['acuerdo_cohorte']} |\n")
        f.write(f"\nLéxico común persistido en: `{ruta_conv}`\n")
        f.write("\n> v1: organismos frescos responden por su física; la divergencia de MATIZ por-organismo "
                "emerge al calibrar contra la historia viva (restaurar estado persistido) — próximo refinamiento. "
                "El NOMBRE, en cambio, es siempre común por diseño.\n")
    print(f"\nListo → {out}/resumen.md")


if __name__ == "__main__":
    main()
