#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paisajes_experienciales.py — descubre los PAISAJES que emergen del repertorio (no los impone).

Principio (Alexis, inspirado en Jean-Michel Jarre): la unidad de ANIMA no es el sonido sino el
PAISAJE EXPERIENCIAL — el tipo de organismo en que el oír convierte al organismo. Cada palabra se
representa por su Δvector RESIDUAL (la reorganización que induce, ya sin el modo común 'llegó un
sonido'). Aquí AGRUPAMOS esos Δvectores: los paisajes son CLUSTERS emergentes, no los 4 cuadrantes
impuestos. Si dos audios humanamente distintos (Bach, viento, una voz) caen en el mismo cluster, es
porque para el organismo SON el mismo paisaje. Análisis del observador (offline), no del organismo.

Uso:  venv/bin/python3 experimentos/paisajes_experienciales.py [--k 5]
Lee lexico_comun/convencion_lexica.json (delta_residual). Escribe paisajes_experienciales.md/.json al lado.
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__)); RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, os.path.join(RAIZ, "organelos"))
from VST_CalibradorLexicoExperiencial import clasificar, REGIMENES
LEX = os.environ.get("ANIMA_LEXICO_DIR") or os.path.join(RAIZ, "lexico_comun")


def _kmeans(X, k, iters=200, seed=7):
    """k-means de numpy (Lloyd) con init k-means++ determinista. Sin dependencias externas."""
    rng = np.random.RandomState(seed)
    c = [X[rng.randint(len(X))]]
    for _ in range(k - 1):
        d = np.min([np.sum((X - ci) ** 2, axis=1) for ci in c], axis=0)
        p = d / (d.sum() + 1e-12)
        c.append(X[rng.choice(len(X), p=p)])
    C = np.array(c)
    for _ in range(iters):
        asg = np.argmin(((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2), axis=1)
        newC = np.array([X[asg == j].mean(axis=0) if np.any(asg == j) else C[j] for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    inercia = sum(((X[asg == j] - C[j]) ** 2).sum() for j in range(k))
    return asg, C, inercia


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=5, help="nº de paisajes (>=2)")
    args = ap.parse_args()
    conv = json.load(open(os.path.join(LEX, "convencion_lexica.json"), encoding="utf-8")).get("convencion", {})
    words = [w for w in sorted(conv) if conv[w].get("delta_residual")]
    keys = sorted({k for w in words for k in conv[w]["delta_residual"]})
    X = np.array([[conv[w]["delta_residual"].get(k, 0.0) for k in keys] for w in words])
    # estandarizar (cada dimensión pesa igual: la reorganización, no la escala de la variable)
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    Xs = (X - mu) / sd

    k = max(2, args.k)
    asg, C, _ = _kmeans(Xs, k)

    def acople(centro):
        """Score de ACOPLE del paisaje (beneficio sobre el residual): >0 integra, <0 desacopla."""
        g = {keys[t].replace("d_", ""): centro[t] for t in range(len(keys))}
        return (g.get("A_sys_env", 0) + g.get("OI", 0) + g.get("H", 0) + g.get("ICR", 0)
                + g.get("LF_op", 0) - g.get("IRDE", 0) - g.get("RC_total", 0) - g.get("necesidad", 0))

    paisajes = []
    for j in range(k):
        miembros = [words[i] for i in range(len(words)) if asg[i] == j]
        if not miembros:
            continue
        centro = X[asg == j].mean(axis=0)          # centro en unidades reales del Δvector
        ac = acople(centro)
        tendencia = ("ACOPLA / integra" if ac > 0.01 else "DESACOPLA / disrupta" if ac < -0.01 else "neutro / estabiliza")
        orden = np.argsort(centro)
        sube = [(keys[t], round(centro[t], 4)) for t in orden[::-1][:4] if centro[t] > 0]
        baja = [(keys[t], round(centro[t], 4)) for t in orden[:4] if centro[t] < 0]
        humanos = [m for m in miembros if not m.isdigit() and not m.startswith("fon_")]
        paisajes.append({"paisaje": j, "n": len(miembros), "acople": round(float(ac), 4),
                         "tendencia": tendencia, "sube": sube, "baja": baja,
                         "miembros": miembros, "humanos": humanos})
    paisajes.sort(key=lambda p: -p["acople"])      # de más integrador a más disruptor

    print(f"{len(words)} palabras → {k} PAISAJES experienciales emergentes (clusters del Δvector residual)")
    print("(ordenados de ACOPLA→DESACOPLA; el paisaje no se impone, emerge de la reorganización)\n")
    for idx, p in enumerate(paisajes, 1):
        print(f"PAISAJE {idx}  (n={p['n']}, {p['tendencia']}, acople={p['acople']:+.3f})")
        print(f"  reorganización: ↑ {', '.join(k for k,_ in p['sube'])}   ↓ {', '.join(k for k,_ in p['baja'])}")
        print(f"  habitantes: {', '.join(p['miembros'][:16])}{' …' if len(p['miembros'])>16 else ''}")
        if len(p["humanos"]) > 1:
            print(f"  → audios humanamente DISTINTOS, MISMO paisaje: {', '.join(p['humanos'][:12])}")
        print()

    with open(os.path.join(LEX, "paisajes_experienciales.json"), "w", encoding="utf-8") as f:
        json.dump({"k": k, "variables": keys, "paisajes": paisajes}, f, ensure_ascii=False, indent=1, default=float)
    with open(os.path.join(LEX, "paisajes_experienciales.md"), "w", encoding="utf-8") as f:
        f.write(f"# Paisajes experienciales emergentes\n\n{len(words)} palabras agrupadas por la REORGANIZACIÓN "
                f"que inducen (Δvector residual), no por su acústica. {k} paisajes emergentes.\n\n")
        f.write("> La unidad no es el sonido sino el paisaje: *¿qué clase de organismo me vuelvo al oír esto?*\n\n")
        for idx, p in enumerate(paisajes, 1):
            f.write(f"## Paisaje {idx} — {p['tendencia']} (acople {p['acople']:+.3f}) · {p['n']} sonidos\n\n")
            f.write(f"- **Reorganización:** suben {', '.join(k for k,_ in p['sube'])}; bajan {', '.join(k for k,_ in p['baja'])}\n")
            f.write(f"- **Habitantes:** {', '.join(p['miembros'])}\n\n")
    print(f"→ {os.path.join(LEX, 'paisajes_experienciales.md')}")


if __name__ == "__main__":
    main()
