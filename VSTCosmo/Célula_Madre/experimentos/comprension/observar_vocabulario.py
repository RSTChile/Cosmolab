#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OBSERVAR EL VOCABULARIO PROPIO (read-only) — ver la acuñación y la convergencia A↔B en vivo.
No toca a los organismos: lee sus /estado (contadores) y sus léxicos inventados (manifiesto en /data,
vía docker exec). Muestra, por organismo: cuántas palabras ACUÑÓ, cuántas CUAJARON (estables), cuántas
APRENDIÓ del otro (ecos), y las palabras con su afecto. Y la señal que importa para el estudio:
  · TASA DE ACUÑACIÓN (medida de exaptación vivida).
  · CONVERGENCIA A↔B: cuántas palabras de cada uno son ecos del otro + cuánto se solapan sus regiones
    afectivas inventadas (si crece → léxico compartido; si no → dos linajes).
Un disparo por defecto; con WATCH=<seg> hace un seguimiento y lo anota en Docker_Historia/VOCABULARIO_PROPIO.csv.
ENV: A_URL(127.0.0.1:7788), B_URL(127.0.0.1:7799), WATCH(0=un disparo), CONT_A(anima-a), CONT_B(anima-b).
"""
import os, sys, json, time, subprocess, urllib.request

A_URL = os.environ.get("A_URL", "http://127.0.0.1:7788")
B_URL = os.environ.get("B_URL", "http://127.0.0.1:7799")
CONT = {"A": os.environ.get("CONT_A", "anima-a"), "B": os.environ.get("CONT_B", "anima-b")}
WATCH = float(os.environ.get("WATCH", "0"))
RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV = os.path.join(os.path.dirname(RAIZ), "Docker_Historia", "VOCABULARIO_PROPIO.csv")

def _estado(url):
    try:
        with urllib.request.urlopen(url + "/estado", timeout=3) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return {}

def _manifiesto(cont):
    """Lee el léxico inventado persistido del organismo (en su volumen /data)."""
    try:
        out = subprocess.run(["docker", "exec", cont, "cat", "/data/voces_creadas/manifiesto.json"],
                             capture_output=True, text=True, timeout=8)
        return json.loads(out.stdout) if out.stdout.strip() else []
    except Exception:
        return []

def _convergencia(mA, mB):
    """Señal de léxico compartido: ecos (palabras 'aprendida') + solapamiento de regiones afectivas."""
    ecoA = sum(1 for e in mA if e.get("afecto_origen") == "aprendida")
    ecoB = sum(1 for e in mB if e.get("afecto_origen") == "aprendida")
    # solapamiento: fracción de palabras de A con alguna de B a < 0.18 en el plano afecto (y viceversa)
    def cerca(e, otros):
        return any((e["aro"] - o["aro"]) ** 2 + (e["val"] - o["val"]) ** 2 < 0.18 ** 2 for o in otros)
    sol = 0; tot = len(mA) + len(mB)
    if tot:
        sol = (sum(1 for e in mA if cerca(e, mB)) + sum(1 for e in mB if cerca(e, mA))) / tot
    return ecoA, ecoB, round(sol, 3)

def snapshot():
    eA, eB = _estado(A_URL), _estado(B_URL)
    mA, mB = _manifiesto(CONT["A"]), _manifiesto(CONT["B"])
    ecoA, ecoB, sol = _convergencia(mA, mB)
    print(f"\n{'='*78}\n  VOCABULARIO PROPIO — {time.strftime('%H:%M:%S')}\n{'='*78}")
    for nom, e, m in [("A", eA, mA), ("B", eB, mB)]:
        print(f"  {nom}: acuñadas(histór) {e.get('voz_creadas',0)} · propio activo {e.get('voz_propias',0)} "
              f"· estables {e.get('voz_estables',0)} · aprendidas {e.get('voz_aprendidas',0)} "
              f"· persistidas {len(m)}")
        for w in m[-5:]:
            print(f"       · {w.get('titulo','?'):26s} a{w.get('aro',0):+.2f}/v{w.get('val',0):+.2f} "
                  f"[{w.get('afecto_origen','creado')}] usos={w.get('usos','?')}")
    print(f"\n  CONVERGENCIA A↔B: ecos de B←A={ecoB} · ecos de A←B={ecoA} · solapamiento de regiones={sol}")
    print("  (ecos>0 o solapamiento↑ en el tiempo = léxico compartido; 0/0/0 sostenido = dos linajes)")
    return eA, eB, ecoA, ecoB, sol

def main():
    if WATCH <= 0:
        snapshot(); return
    nuevo = not os.path.isfile(CSV)
    with open(CSV, "a", encoding="utf-8") as fh:
        if nuevo:
            fh.write("ts,A_creadas,A_estables,A_aprendidas,B_creadas,B_estables,B_aprendidas,eco_A,eco_B,solapamiento\n")
        try:
            while True:
                eA, eB, ecoA, ecoB, sol = snapshot()
                fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{eA.get('voz_creadas',0)},{eA.get('voz_estables',0)},"
                         f"{eA.get('voz_aprendidas',0)},{eB.get('voz_creadas',0)},{eB.get('voz_estables',0)},"
                         f"{eB.get('voz_aprendidas',0)},{ecoA},{ecoB},{sol}\n"); fh.flush()
                time.sleep(WATCH)
        except KeyboardInterrupt:
            print("\n  (seguimiento detenido)")

if __name__ == "__main__":
    main()
