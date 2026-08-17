#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Catálogo de fuentes de audio para instalación limpia (perfil minimal/full/custom)."""

from __future__ import annotations

import glob
import json
import os
from typing import Any


def _manifest_path(cm_dir: str) -> str:
    return os.path.join(cm_dir, "audio", "audio_library_minimal.json")


def load_manifest(cm_dir: str) -> dict[str, Any] | None:
    path = _manifest_path(cm_dir)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _matches_exclude(name: str, patterns: list[str]) -> bool:
    low = name.lower()
    for pat in patterns:
        if pat.lower() in low:
            return True
    return False


def listar_archivos(
    audio_dir: str,
    *,
    profile: str = "minimal",
    cm_dir: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Devuelve (entradas para UI, meta del catálogo)."""
    profile = (profile or "minimal").strip().lower()
    manifest = load_manifest(cm_dir or os.path.dirname(os.path.dirname(audio_dir)))
    meta: dict[str, Any] = {
        "profile": profile,
        "audio_dir": audio_dir,
        "manifest": bool(manifest),
        "total_en_disco": 0,
        "total_listados": 0,
    }
    if not os.path.isdir(audio_dir):
        return [], {**meta, "error": "directorio de audio no encontrado"}

    all_wavs = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
    meta["total_en_disco"] = len(all_wavs)

    if profile == "full":
        entries = [
            {
                "tipo": "archivo",
                "nombre": os.path.basename(p),
                "label": "📄 " + os.path.basename(p),
                "categoria": "biblioteca",
                "categoria_label": "Biblioteca completa",
            }
            for p in all_wavs
        ]
        meta["total_listados"] = len(entries)
        return entries, meta

    if profile == "custom" or not manifest:
        max_size = int(manifest.get("max_size_bytes", 10_485_760) if manifest else 10_485_760)
        patterns = list(manifest.get("exclude_patterns", [])) if manifest else [
            "BigBang", "Blue_Monday", "_largo", ".pkf"
        ]
        entries = []
        for p in all_wavs:
            base = os.path.basename(p)
            if _matches_exclude(base, patterns):
                continue
            try:
                if os.path.getsize(p) > max_size:
                    continue
            except OSError:
                continue
            entries.append({
                "tipo": "archivo",
                "nombre": base,
                "label": "📄 " + base,
                "categoria": "biblioteca",
                "categoria_label": "Biblioteca",
            })
        meta["total_listados"] = len(entries)
        return entries, meta

    cats = {c["id"]: c["label"] for c in manifest.get("categories", []) if c.get("id")}
    by_file = {it["file"]: it for it in manifest.get("items", []) if it.get("file")}
    entries = []
    for p in all_wavs:
        base = os.path.basename(p)
        item = by_file.get(base)
        if not item:
            continue
        cat = item.get("category", "biblioteca")
        entries.append({
            "tipo": "archivo",
            "nombre": base,
            "label": "📄 " + item.get("label", base),
            "categoria": cat,
            "categoria_label": cats.get(cat, cat),
            "package": item.get("package", "minimal"),
        })
    meta["total_listados"] = len(entries)
    meta["categories"] = manifest.get("categories", [])
    meta["excluded_note"] = (
        f"{meta['total_en_disco'] - meta['total_listados']} archivos omitidos "
        f"(perfil {profile}; carga manual o perfil full)"
    )
    return entries, meta