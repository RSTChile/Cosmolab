#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MONITOR DE CANALES (read-only) — niveles por canal EN VIVO, directo del AudioServer.
Sirve para diagnosticar el ruteo de la Rødecaster sin pasar por los organismos ni el navegador: muestra
el RMS de cada canal del stream multitrack y lo refresca, para VER cuáles llevan señal mientras se ajusta
la mesa (faders, paneo, asignación USB, multitrack ON). No toca nada; sólo lee el puerto TCP del servidor.
ENV: HOST(127.0.0.1), PORT(8766), SEG(0.4 ventana por refresco).
"""
import os, socket, struct, json, time
import numpy as np

HOST = os.environ.get("HOST", "127.0.0.1"); PORT = int(os.environ.get("PORT", "8766"))
SEG = float(os.environ.get("SEG", "0.4"))
# nombres legibles de los pares de la Rødecaster Pro II (1-based, verificados por Alexis)
NOMBRES = {1: "Main Mix L", 2: "Main Mix R", 3: "Combo1 L", 4: "Combo1 R", 5: "Combo2 L", 6: "Combo2 R",
           7: "Combo3 L", 8: "Combo3 R", 9: "Bluetooth L", 10: "Bluetooth R", 11: "USB2 L", 12: "USB2 R",
           13: "USB Main L", 14: "USB Main R", 15: "SMART L", 16: "SMART R", 17: "USB Chat L", 18: "USB Chat R"}

def recv(s, n):
    b = b""
    while len(b) < n:
        d = s.recv(n - len(b))
        if not d:
            raise ConnectionError("stream cerrado")
        b += d
    return b

def main():
    s = socket.create_connection((HOST, PORT), timeout=5)
    hdr = json.loads(recv(s, struct.unpack(">I", recv(s, 4))[0]).decode())
    nch = hdr["channels"]; dt = hdr["dtype"]; sr = hdr["sample_rate"]
    print(f"AudioServer: {hdr['device']} · {nch} canales · {sr} Hz · ventana {SEG}s · Ctrl+C para salir\n")
    bloques = max(1, int(SEG * sr / hdr["block_size"]))
    try:
        while True:
            acc = np.zeros(nch); k = 0
            for _ in range(bloques):
                buf = np.frombuffer(recv(s, struct.unpack(">I", recv(s, 4))[0]), dtype=dt).reshape(-1, nch)
                acc += np.sqrt(np.mean(buf.astype(np.float64) ** 2, axis=0)); k += 1
            rms = acc / max(1, k)
            print("\033[2J\033[H", end="")   # limpia pantalla
            print(f"  niveles por canal  ({time.strftime('%H:%M:%S')})")
            con = 0
            for i in range(min(nch, 18)):
                r = rms[i]; on = r > 0.002
                if on:
                    con += 1
                barra = "█" * int(min(1.0, r / 0.25) * 30)
                marca = "●" if on else "○"
                print(f"  {marca} ch{i+1:2d} {NOMBRES.get(i+1,''):12s} {r:.4f} {barra}")
            print(f"\n  {con}/18 canales con señal  ·  (señal = rms > 0.002)")
    except KeyboardInterrupt:
        print("\n  (monitor detenido)")
    finally:
        s.close()

if __name__ == "__main__":
    main()
