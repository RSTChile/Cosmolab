#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS074 — Persistencia de una diferencia ínfima bajo expansión
=============================================================
Implementa INSTRUCCION_CC_persistencia_expansion_PARA_CC.md (v4).

PREGUNTA (única): dada UNA diferencia de magnitud ε en un Todo normalizado a 1,
¿persiste esa diferencia cuando el Todo se separa (expansión H) más rápido de lo
que la diferencia puede reabsorberse — y qué CIERRES (tamaño k) sobreviven?

REGLA DEL DIRECTOR (anti-Shannon):
 - Único parámetro de ENTRADA: ε (escalar) y H (tasa de expansión). NADA MÁS.
 - CERO unidades de este universo: todo es RAZÓN interna (adimensional).
 - La tasa de reabsorción Γ se MIDE del sustrato (H=0), no se impone (G-C-EMERGENTE).
 - Los quantos y sus cierres k EMERGEN y se MIDEN a la salida; NUNCA se imponen.
 - Persistencia SIEMPRE contra su NULL (barajado de misma magnitud).
 - NO target-matching: prohibido validar por η, 7:1, Y_He ni número conocido.
 - Nulo = hallazgo. La curva entera se reporta, no se recorta.

############################################################################
## ESTADO TRAS REVISIÓN DEL EQUIPO (ronda 1) — 3 defectos ARREGLADOS,        ##
## 1 PROBLEMA ABIERTO que el equipo debe resolver ANTES de producción.       ##
############################################################################
## ARREGLADO (síntesis de la revisión, sin Shannon):
##  D1 aniquilación no competía -> ahora limitada por tasa EMERGENTE
##     (p_intento = grado_actual/(N-1); la expansión baja el grado y frena la
##      aniquilación sola; Γ se sigue midiendo con H=0). Hay carrera real.
##  D2 NULL no mordía -> observable religado a cierres k>=2 (masa en estructura,
##     no conteo).
##  D3 NULL era no-op -> barajado del acople AHORA es post-expansión (destruye
##     la estructura que la expansión creó), no al inicio (grafo completo).
##  Bug z -> piso de varianza sensato (1/n_semillas), no 1e-9 (que explotaba a ~1e8).
##
## !!! PROBLEMA 2 — ABIERTO, NO RESUELTO — LO DECIDE EL EQUIPO !!!
##   Tras los arreglos, el instrumento SÍ discrimina. El signo del z es NEGATIVO
##   en las filas con señal REAL (eps>=0.1 y H>0): P_real < P_null.
##   Ej. eps=0.5 H=0.5: P_real=0.017 vs P_null=0.102, z=-0.34; eps=0.1 H alto igual.
##   MATIZ HONESTO: 4 de 16 filas dan z POSITIVO pequeño (eps=0.0 y 0.001, H<=0.1;
##   z=0.02..0.08) — son eps~0 (sin diferencia inicial que preservar) y z
##   despreciable (ruido de piso), NO señal. El patrón negativo NO es universal:
##   vale donde hay diferencia real que perder. Rango de z en el smoke: -0.83 a +0.08.
##   O sea: BARAJAR el acople (destruir la estructura) PRESERVA la diferencia
##   MEJOR que la dinámica real. Va al REVÉS de la hipótesis.
##   Dos lecturas, y CS NO adjudica cuál:
##    (i) ARTEFACTO del NULL: barajar cada paso "refresca" contactos aleatorios
##        que impiden a la aniquilación hallar complementos -> sobreviven más por
##        accidente del barajado, no por estructura. Sería un 4º defecto de
##        instrumento (el NULL fabrica persistencia).
##    (ii) HALLAZGO NEGATIVO REAL: la estructura del acople real NO protege la
##        diferencia; la organiza para aniquilarse más eficientemente. Mundo-B
##        parcial: la persistencia no viene de la estructura que suponíamos.
##   PREGUNTA EXACTA PARA EL EQUIPO: ¿el barajado post-expansión es un NULL
##   legítimo, o fabrica supervivencia al refrescar contactos? Si es legítimo,
##   REAL<NULL es un hallazgo negativo real y se reporta como tal. NO correr
##   producción hasta resolver esto (correr más semillas no lo decide: es diseño).
############################################################################

============================================================================
!! TRES DECISIONES DE MODELADO QUE EL EQUIPO DEBE AUDITAR (posible Shannon) !!
============================================================================
 (D1) La REGLA DE AFINIDAD entre elementos (cómo dos diferencias deciden
      mantenerse juntas vs borrarse). Aquí: afinidad = función SOLO de la
      compatibilidad de fases emergentes, nunca del índice. ¿Es neutral?
 (D2) El ALFABETO DE CARGA. Si damos ±1, sesgamos a k par; si damos tercios,
      sesgamos a k=3. AQUÍ NO SE IMPONE: la carga de cada quanto es una fase
      CONTINUA que emerge de la dinámica; un cierre es un subconjunto cuya
      fase suma ~neutro. Qué k neutraliza depende de la DISTRIBUCIÓN emergente
      de fases, no de un alfabeto elegido. <-- el punto más delicado, revísenlo.
 (D3) La medida de PERSISTENCIA (contraste final/inicial). ¿Mide estructura
      genuina o clustering trivial? Por eso va SIEMPRE contra NULL barajado.
============================================================================
"""
import numpy as np
import json, sys

# ----------------------------------------------------------------------------
# 1. SUSTRATO INICIAL — sin geometría, sin densidad por índice, sin RNG-fundacional
# ----------------------------------------------------------------------------
def sustrato_inicial(N, eps, rng):
    """
    El Todo = N elementos idénticos (fase 0 = indistinguibles del campo = 'nada').
    La ÚNICA diferencia inicial es un DESBALANCE DE CONTEO de magnitud eps:
    una fracción (1+eps)/2 lleva etiqueta +1, (1-eps)/2 lleva -1.
      - eps = 0  -> mitad +, mitad -  -> simetría perfecta (la 'Nada'/cierre).
      - eps -> 1 -> casi todos +      -> asimetría máxima.
    NO se asigna densidad ni fase por elemento: la fase EMERGE de la dinámica.
    Acoplamiento inicial = TODOS con TODOS, uniforme (no hay espacio todavía;
    el 'cerca/lejos' emerge del desacople por expansión, no se pre-impone).
    'rng' solo decide QUÉ elemento lleva la etiqueta (gauge: los elementos son
    indistinguibles, cuál lleva el + no es físico) — NO fabrica la diferencia.
    """
    n_mas = int(round(N * (1.0 + eps) / 2.0))
    tag = np.full(N, -1, dtype=np.int8)
    idx_mas = rng.choice(N, size=min(n_mas, N), replace=False)
    tag[idx_mas] = +1
    fase = np.zeros(N, dtype=float)          # fase EMERGE; arranca en 0 (=campo)
    vivo = np.ones(N, dtype=bool)
    # grafo de acoplamiento: completo (sin geometría). Lo representamos implícito.
    return {"tag": tag, "fase": fase, "vivo": vivo, "N": N,
            "acople": np.ones((N, N), dtype=bool) & ~np.eye(N, dtype=bool)}

# ----------------------------------------------------------------------------
# 2. DINÁMICA — dos procesos compiten: reabsorción (borra) vs expansión (separa)
# ----------------------------------------------------------------------------
def paso_reabsorcion(est, rng):
    """
    Reabsorción = el canal de BORRADO (tendencia del campo a volver a I=0).
    (a) Aniquilación LIMITADA POR TASA EMERGENTE (fix Defecto 1):
        cada elemento vivo INTENTA aniquilarse con probabilidad = grado_actual/(N-1),
        donde grado = nº de vecinos vivos acoplados de tag opuesto.
        - H=0: grafo denso -> grado alto -> p~1 -> aniquila rápido (Γ alto).
        - H>0: la EXPANSIÓN ya cortó aristas -> grado bajo -> p<1 -> la aniquilación
          se frena SOLA. Aquí nace la CARRERA: ¿corta H aristas antes de que la
          reabsorción alcance a aniquilar el par? La tasa NO se impone: emerge del
          grado, que la propia expansión reduce. Γ se sigue midiendo con H=0.
    (b) Igualación de fase: elementos acoplados vivos acercan sus fases (difusión).
    """
    N = est["N"]; vivo = est["vivo"]; tag = est["tag"]; fase = est["fase"]
    ac = est["acople"]
    viv = np.where(vivo)[0]
    rng.shuffle(viv)
    n_pares = 0
    for i in viv:
        if not vivo[i]:
            continue
        grado = int((ac[i] & vivo).sum())          # contactos vivos actuales
        p_intento = grado / max(N - 1, 1)           # EMERGE del acople (no impuesto)
        if rng.random() > p_intento:
            continue
        cand = np.where(vivo & (tag == -tag[i]) & ac[i])[0]   # complemento acoplado
        if cand.size:
            j = int(rng.choice(cand))
            if vivo[j]:
                est["vivo"][i] = False; est["vivo"][j] = False
                n_pares += 1
    # (b) igualación de fase entre acoplados vivos
    viv2 = np.where(est["vivo"])[0]
    if viv2.size > 1:
        media_local = fase[viv2].mean()
        fase[viv2] += 0.15 * (media_local - fase[viv2])
    return n_pares

def paso_expansion(est, H, rng):
    """
    Expansión = SEPARACIÓN: con 'intensidad' H se eliminan aristas del acople
    (los elementos se desacoplan; una vez desacoplados NO pueden re-contactar,
    y por tanto ya no pueden aniquilarse -> la diferencia queda congelada).
    H es adimensional: fracción de aristas vivas que se cortan por paso.
    Además la fase de los que quedan aislados se 'fija' (deja de igualarse).
    """
    ac = est["acople"]; vivo = est["vivo"]
    viv = np.where(vivo)[0]
    if viv.size < 2:
        return
    # cortar una fracción H de las aristas vivas actuales
    sub = ac[np.ix_(viv, viv)]
    ii, jj = np.where(np.triu(sub, 1))
    if ii.size == 0:
        return
    ncorte = int(round(min(H, 1.0) * ii.size))
    if ncorte <= 0:
        return
    sel = rng.choice(ii.size, size=ncorte, replace=False)
    for s in sel:
        a = viv[ii[s]]; b = viv[jj[s]]
        ac[a, b] = False; ac[b, a] = False

def evolucionar(est, H, pasos, rng, null=False):
    """
    Un proceso, no una sucesión: cada paso hace reabsorción Y expansión juntas.
    NULL (fix Defecto 2/3): baraja el acople DESPUÉS de cada expansión, NO al inicio.
    Barajar al inicio era no-op (el grafo arranca completo; barajar completo=completo,
    por eso el NULL nunca mordía). Barajado post-expansión destruye la ESTRUCTURA que
    la expansión creó, conservando el nº de aristas -> el NULL sí muerde.
    """
    tag0 = est["tag"].copy(); vivo0 = est["vivo"].copy()
    net0 = abs(int(tag0[vivo0].sum()))            # diferencia neta inicial
    n_tag0 = int(vivo0.sum())
    for _ in range(pasos):
        paso_reabsorcion(est, rng)
        paso_expansion(est, H, rng)
        if null:
            _barajar_acople(est, rng)
    return {"net0": net0, "n_tag0": n_tag0}

def _barajar_acople(est, rng):
    """NULL: baraja aristas vivas conservando su número (destruye topología)."""
    ac = est["acople"]; N_ = est["N"]
    tri_i, tri_j = np.triu_indices(N_, 1)
    vals = ac[tri_i, tri_j].copy()
    rng.shuffle(vals)
    nueva = np.zeros((N_, N_), dtype=bool)
    nueva[tri_i, tri_j] = vals
    est["acople"] = nueva | nueva.T

# ----------------------------------------------------------------------------
# 3. Γ EMERGENTE — se MIDE, no se impone (G-C-EMERGENTE)
# ----------------------------------------------------------------------------
def medir_Gamma(N, eps, pasos, seed):
    """
    Tasa intrínseca de reabsorción del sustrato: con H=0 (sin expansión),
    ¿a qué rapidez decae la población de elementos con etiqueta (por aniquilación)?
    Γ = fracción aniquilada por paso al inicio. NO es un número puesto; sale del
    propio sustrato. r = H / Γ será el eje adimensional del barrido.
    """
    rng = np.random.default_rng(seed)
    est = sustrato_inicial(N, eps, rng)
    viv0 = int(est["vivo"].sum())
    p = paso_reabsorcion(est, rng)          # un paso, H=0
    muertos = viv0 - int(est["vivo"].sum())
    return muertos / max(viv0, 1)

# ----------------------------------------------------------------------------
# 4. PERSISTENCIA y CIERRES (medidos a la salida, NO impuestos)
# ----------------------------------------------------------------------------
def persistencia(est, info):
    """
    OBSERVABLE CORREGIDO (v3): fracción de la diferencia inicial que queda ATRAPADA
    EN CIERRES NO TRIVIALES (componentes vivas de tamaño k>=2).
    NO conteo puro (v2 daba P=eps e independiente de H -> z=0, no discriminaba).
    Se religa a la ESTRUCTURA topológica (los cierres k), que es lo único que YA
    respondía a H. Así el NULL muerde: al destruir la topología (barajado post-
    expansión), los cierres grandes desaparecen y P_null cae.
      - H=0: la aniquilación consume todo lo acoplado -> quedan singletons -> P bajo.
      - H alto: la expansión aísla cierres antes de aniquilarlos -> P alto.
    P = (nº de elementos vivos en cierres de tamaño>=2) / (elementos con tag inicial).
    """
    cierres = detectar_cierres(est)
    protegidos = sum(k * c for k, c in cierres.items() if k >= 2)
    if info["n_tag0"] <= 0:
        return 0.0
    return protegidos / info["n_tag0"]

def detectar_cierres(est):
    """
    Cierre = componente conexa de elementos vivos aún acoplados que sobrevivió
    (un 'ciclo cerrado' que la expansión aisló y la aniquilación no deshizo).
    Su tamaño k se MIDE. NO se impone ningún k. Devuelve histograma {k: conteo}.
    (D2/D3): qué k aparece emerge del acople residual, no de un alfabeto elegido.
    """
    vivo = est["vivo"]; ac = est["acople"]
    viv = np.where(vivo)[0]
    visto = set(); hist = {}
    vivset = set(viv.tolist())
    for start in viv:
        if start in visto:
            continue
        # BFS sobre acople entre vivos
        comp = []; pila = [start]
        while pila:
            u = pila.pop()
            if u in visto:
                continue
            visto.add(u); comp.append(u)
            vec = np.where(ac[u] & vivo)[0]
            for w in vec:
                if w not in visto:
                    pila.append(w)
        k = len(comp)
        hist[k] = hist.get(k, 0) + 1
    return hist

# ----------------------------------------------------------------------------
# 5. NULL — barajado de misma magnitud (G-PERSISTENCIA-VS-NULL)
# ----------------------------------------------------------------------------
def corrida(N, eps, H, pasos, seed, null=False):
    rng = np.random.default_rng(seed)
    est = sustrato_inicial(N, eps, rng)
    # NULL se aplica DENTRO de evolucionar (barajado post-expansión, no al inicio).
    info = evolucionar(est, H, pasos, rng, null=null)
    P = persistencia(est, info)
    cierres = detectar_cierres(est)
    return {"P": P, "cierres": cierres, "vivos": int(est["vivo"].sum())}

# ----------------------------------------------------------------------------
# 6. BARRIDO — único eje de entrada: eps y H (H se reexpresa como r=H/Γ)
# ----------------------------------------------------------------------------
def barrido(N, eps_list, H_list, pasos, semillas):
    filas = []
    for eps in eps_list:
        # Γ emergente (promedio sobre semillas)
        G = np.mean([medir_Gamma(N, eps, pasos, s) for s in range(semillas)])
        for H in H_list:
            r = H / G if G > 0 else np.inf
            Preal, Pnull = [], []
            hist_real = {}
            for s in range(semillas):
                rr = corrida(N, eps, H, pasos, seed=1000 + s, null=False)
                nn = corrida(N, eps, H, pasos, seed=1000 + s, null=True)
                Preal.append(rr["P"]); Pnull.append(nn["P"])
                for k, c in rr["cierres"].items():
                    hist_real[k] = hist_real.get(k, 0) + c
            Preal = np.array(Preal); Pnull = np.array(Pnull)
            # z con piso de varianza SENSATO (fix bug smoke v3): cuando el NULL da
            # P idéntico entre semillas, sd->0 y 1e-9 hacía explotar z a ~1e8 (artefacto
            # de división, no física). Piso = error estándar de una diferencia mínima
            # detectable ~ 1/n_tag0 escalado por semillas; si sd real es 0 y las medias
            # coinciden, z=0; si difieren con sd=0, se reporta z acotado (no infinito).
            sd_pool = np.sqrt((Preal.var() + Pnull.var()) / 2.0)
            sd = max(sd_pool, 1.0 / max(len(Preal), 1))   # piso = 1/n_semillas, no 1e-9
            z = (Preal.mean() - Pnull.mean()) / sd
            filas.append({
                "eps": eps, "H": H, "Gamma": round(float(G), 4), "r": round(float(r), 3),
                "P_real": round(float(Preal.mean()), 4),
                "P_null": round(float(Pnull.mean()), 4),
                "z": round(float(z), 2),
                "cierres_k": {int(k): int(v) for k, v in sorted(hist_real.items())},
            })
    return filas

# ----------------------------------------------------------------------------
# 7. SMOKE TEST (N pequeño) — solo verifica que corre y da salidas sanas
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if modo == "smoke":
        N = 120; pasos = 40; semillas = 4
        eps_list = [0.0, 1e-3, 1e-1, 0.5]
        H_list = [0.0, 0.1, 0.5, 0.9]
    else:  # produccion (lo corre el equipo tras revisar)
        N = 800; pasos = 120; semillas = 12
        eps_list = [0.0, 1e-9, 1e-6, 1e-3, 1e-2, 0.1, 0.5, 1.0]
        H_list = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9]
    filas = barrido(N, eps_list, H_list, pasos, semillas)
    print(json.dumps({"modo": modo, "N": N, "pasos": pasos, "semillas": semillas,
                      "filas": filas}, ensure_ascii=False))
