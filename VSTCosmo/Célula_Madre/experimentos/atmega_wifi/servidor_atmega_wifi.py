#!/usr/bin/env python3
"""Servidor TCP simple para recibir reportes RD-WiFi del ATmega de E."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import socket
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Recibe paquetes TCP del RD-WiFi/ATmega.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--jsonl",
        default="/tmp/atmega_wifi_rx.jsonl",
        help="Archivo donde se anexan los paquetes recibidos.",
    )
    args = parser.parse_args()

    jsonl = Path(args.jsonl)
    jsonl.parent.mkdir(parents=True, exist_ok=True)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(5)
        print(f"[atmega-wifi] escuchando en {args.host}:{args.port}")
        print(f"[atmega-wifi] log: {jsonl}")

        while True:
            conn, addr = server.accept()
            with conn:
                data = conn.recv(2048)
            if not data:
                continue
            text = data.decode("utf-8", errors="replace").strip()
            event = {
                "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
                "from": addr[0],
                "port": addr[1],
                "raw": text,
            }
            with jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
            print(f"[{event['ts']}] {addr[0]}:{addr[1]} -> {text}", flush=True)


if __name__ == "__main__":
    main()
