#!/usr/bin/env python3
"""Move cosmosemiotic documents from Desktop RMD 2.0 folder to VSTCosmo import area."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

DESKTOP = Path("/Users/alexis/Desktop/Go en Conflictos/RMD 2.0")
DEST_ROOT = Path("/Users/alexis/Desktop/RMD/Cosmolab/VSTCosmo/_import_desktop_28jun2026")

MOVE_DIRS = {
    # NO mover (decisión Alexis 28-jun-2026): Abulafia, Evolución, Genética del Altruismo
    "Resultados experimentos",
    "Recording 41",
}

EXCLUDE_DIRS = {
    "Abulafia",
    "Evolución",
    "Genética del Altruismo",
}

KEEP_DIRS = {
    "RMD",
    "ARCHIVOS RMD",
    "Encuestas",
    "Análisis Realizados",
    "Análisis en Curso",
    "Certificados RMD",
    "Curso RMD - Materiales",
    "DashBoard",
    "Pandemia",
    "Segunda Encuesta",
    "Trabajos Finales",
    "Variables y Métricas",
    "Versiones antiguas",
    "RMD_CORE_IA_PACKAGE",
}

COSMO_PATTERNS = [
    r"cosmosemi[oó]tic",
    r"cosmosemiotica",
    r"vstcosmo",
    r"vstcosmos",
    r"\bv1[0-9]{2}\b",
    r"\bv[0-9]{2,3}[a-z]?\b",
    r"\banima\b",
    r"exaptaci[oó]n",
    r"exaptation",
    r"punctuated",
    r"\beit3\b",
    r"eit3",
    r"levitr[oó]n",
    r"abulafia",
    r"cronolog[ií]a experimental",
    r"informe.*v180",
    r"informe.*v176",
    r"informe.*vst",
    r"informe.*v182",
    r"celula.?madre",
    r"c[eé]lula.?madre",
    r"dron cosmosemi",
    r"robot cosmosemi",
    r"plugin cosmosemi",
    r"f[oó]rmulas cosmosemi",
    r"genealog[ií]a de la evoluci[oó]n",
    r"libertad funcional",
    r"cosmogoni[aá]",
    r"cosmo-corr",
    r"cosmo.?nuev",
    r"tabla_codificacion_cosmosemi",
    r"modelo_cosmosemi",
    r"tesis_cosmosemi",
    r"wdr 2",
    r"nodo.*cn\d",
    r"campocontinuo",
    r"evidenciacomputacional",
    r"cicloevolutivo",
    r"ia salvadora",
    r"evoluci[oó]n puntuada",
    r"evoluci[oó]n.*inteligencia.*cosmosemi",
    r"artificial.?punctuated",
    r"grafico de exaptaciones",
    r"experimentos anima",
    r"informe final v180",
    r"comunicaci[oó]n bidireccional",
    r"respuesta final a google ai",
    r"sesi[oó]n robot cosmosemi",
    r"aportes cosmosemi",
    r"encuesta cosmosemi",
    r"nodos nuevos cosmosemi",
    r"cap[ií]tulo de la exaptaci",
    r"cap[ií]tulo 10-1 libertad",
    r"cap[ií]tulo 13-5-24",
    r"ai as exaptation",
    r"la ia como exaptaci",
    r"nodo_universal",
    r"daisyworld",
    r"addendum.*verific",
    r"ia como exaptaci",
    r"seti",
    r"proyectos cosmosemi",
]

RMD_OVERRIDE_PATTERNS = [
    r"^rmd2?_",
    r"^rmd[_ ]2",
    r"matriz.*rmd",
    r"protocolo.*rmd",
    r"mapar",
    r"homologad",
    r"esquiz",
    r"anaktasis",
    r"variables\.y\.metricas",
    r"variables y m[eé]tricas",
    r"contra.?proceso",
    r"pre-rmd",
    r"certificad",
    r"curso_rmd",
    r"curso rmd",
    r"bolivia.*informe",
    r"paraguay",
    r"yeruti",
    r"epp",
    r"narcotr",
    r"elecciones",
    r"doctrina de seguridad",
    r"poder judicial",
    r"huachipato",
    r"cable chile-china",
    r"piratas de aragua",
    r"go en",
    r"tablero de go",
    r"logo-rmd",
    r"analista certificado rmd",
    r"cosmosemiotica_rmd",
]


def norm(text: str) -> str:
    return unicodedata.normalize("NFC", text).lower()


def matches_any(name: str, patterns: list[str]) -> bool:
    lowered = norm(name)
    return any(re.search(pattern, lowered, re.I) for pattern in patterns)


def classify_file(path: Path) -> str:
    name = path.name
    if name.startswith("~$") or name == ".DS_Store":
        return "SKIP"
    if matches_any(name, RMD_OVERRIDE_PATTERNS):
        return "RMD"
    if matches_any(name, COSMO_PATTERNS):
        return "COSMO"
    return "RMD"


def is_altruismo_dir(name: str) -> bool:
    return "altruismo" in norm(name)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def unique_dest(dest: Path) -> Path:
    if not dest.exists():
        return dest
    stem, suffix = dest.stem, dest.suffix
    counter = 1
    while True:
        candidate = dest.with_name(f"{stem}__desktop{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def collect_moves() -> list[tuple[Path, Path, str]]:
    moves: list[tuple[Path, Path, str]] = []
    seen_sources: set[Path] = set()

    def add_move(src: Path, reason: str) -> None:
        src = src.resolve()
        if src in seen_sources or not src.exists():
            return
        if not src.is_relative_to(DESKTOP.resolve()):
            return
        rel = src.relative_to(DESKTOP)
        if rel.parts and rel.parts[0] in ("RMD", *EXCLUDE_DIRS):
            return
        seen_sources.add(src)
        dest = DEST_ROOT / rel
        moves.append((src, dest, reason))

    for dirname in sorted(MOVE_DIRS):
        src_dir = DESKTOP / dirname
        if not src_dir.exists():
            continue
        for path in src_dir.rglob("*"):
            if path.is_file():
                add_move(path, f"dir:{dirname}")

    for path in DESKTOP.iterdir():
        if path.is_dir():
            if path.name in MOVE_DIRS or path.name in KEEP_DIRS or path.name in EXCLUDE_DIRS or path.name == "RMD":
                continue
            continue
        if path.is_file() and classify_file(path) == "COSMO":
            add_move(path, "root:pattern")

    for keep in KEEP_DIRS:
        keep_dir = DESKTOP / keep
        if not keep_dir.exists() or keep == "RMD":
            continue
        for path in keep_dir.rglob("*"):
            if not path.is_file():
                continue
            if classify_file(path) == "COSMO":
                add_move(path, f"stray:{keep}")

    return moves


def execute_moves(moves: list[tuple[Path, Path, str]], dry_run: bool) -> dict:
    log: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "desktop": str(DESKTOP),
        "destination": str(DEST_ROOT),
        "dry_run": dry_run,
        "moved": [],
        "skipped_identical": [],
        "renamed_collision": [],
        "errors": [],
        "summary": {},
    }

    for src, dest, reason in moves:
        entry = {
            "source": str(src),
            "destination": str(dest),
            "reason": reason,
            "status": "pending",
        }
        try:
            if dry_run:
                entry["status"] = "would_move"
                log["moved"].append(entry)
                continue

            dest.parent.mkdir(parents=True, exist_ok=True)
            final_dest = dest
            if dest.exists():
                try:
                    if file_hash(src) == file_hash(dest):
                        src.unlink()
                        entry["status"] = "removed_duplicate_source"
                        log["skipped_identical"].append(entry)
                        continue
                except OSError:
                    pass
                final_dest = unique_dest(dest)
                entry["renamed_to"] = str(final_dest)
                log["renamed_collision"].append(entry)

            shutil.move(str(src), str(final_dest))
            entry["destination"] = str(final_dest)
            entry["status"] = "moved"
            log["moved"].append(entry)
        except Exception as exc:  # noqa: BLE001 - operational script
            entry["status"] = "error"
            entry["error"] = str(exc)
            log["errors"].append(entry)

    log["summary"] = {
        "total_planned": len(moves),
        "moved": len([e for e in log["moved"] if e["status"] == "moved"]),
        "would_move": len([e for e in log["moved"] if e["status"] == "would_move"]),
        "duplicates_removed": len(log["skipped_identical"]),
        "renamed": len(log["renamed_collision"]),
        "errors": len(log["errors"]),
    }
    return log


def write_inventory(moves: list[tuple[Path, Path, str]], path: Path) -> None:
    lines = ["source\tdestination\treason"]
    for src, dest, reason in moves:
        lines.append(f"{src}\t{dest}\t{reason}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Perform moves (default: dry-run)")
    args = parser.parse_args()

    if not DESKTOP.exists():
        raise SystemExit(f"Desktop folder not found: {DESKTOP}")

    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    moves = collect_moves()
    write_inventory(moves, DEST_ROOT / "DESKTOP_CLASIFICACION.tsv")

    log = execute_moves(moves, dry_run=not args.execute)
    (DEST_ROOT / "move_log.json").write_text(
        json.dumps(log, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(log["summary"], indent=2))
    if not args.execute:
        print(f"\nDry-run complete. {len(moves)} files planned.")
        print(f"Inventory: {DEST_ROOT / 'DESKTOP_CLASIFICACION.tsv'}")
        print("Re-run with --execute to move.")


if __name__ == "__main__":
    main()