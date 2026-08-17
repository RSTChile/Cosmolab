#!/bin/bash
set -euo pipefail

DESKTOP="/Users/alexis/Desktop/Go en Conflictos/RMD 2.0"
DEST="/Users/alexis/Desktop/RMD/Cosmolab/VSTCosmo/_import_desktop_28jun2026"
LOG="$DEST/move_log_fast.json"

mkdir -p "$DEST"

log() { echo "[$(date -Iseconds)] $*"; }

move_dir() {
  local name="$1"
  local src="$DESKTOP/$name"
  local dst="$DEST/$name"
  if [[ ! -e "$src" ]]; then
    log "SKIP missing dir: $name"
    return 0
  fi
  mkdir -p "$(dirname "$dst")"
  if [[ -e "$dst" ]]; then
    log "MERGE dir: $name"
    rsync -a "$src/" "$dst/"
    rm -rf "$src"
  else
    log "MOVE dir: $name"
    mv "$src" "$dst"
  fi
}

# Whole cosmosemiotic directories
move_dir "Abulafia"
move_dir "Resultados experimentos"
move_dir "Recording 41"
move_dir "Evolución"

# Genética del Altruismo (unicode-safe via glob)
for d in "$DESKTOP"/*; do
  base="$(basename "$d")"
  if [[ -d "$d" ]] && [[ "${base,,}" == *altruismo* ]]; then
    move_dir "$base"
  fi
done

# Root-level cosmosemiotic files (pattern list)
patterns=(
  -iname '*cosmosemi*'
  -o -iname '*vstcosmo*'
  -o -iname '*exapt*'
  -o -iname '*punctuated*'
  -o -iname '*eit3*'
  -o -iname '*anima*'
  -o -iname '*v176*'
  -o -iname '*v180*'
  -o -iname '*v182*'
  -o -iname '*v80h*'
  -o -iname '*nodo_universal*'
  -o -iname '*wdr*2*'
  -o -iname '*libertad*funcional*'
  -o -iname '*google*ai*'
)

find "$DESKTOP" -maxdepth 1 -type f \( "${patterns[@]}" \) ! -name '~$*' -print0 | while IFS= read -r -d '' f; do
  base="$(basename "$f")"
  case "${base,,}" in
    rmd2_*|rmd_2_*|*mapar*|*homologad*|*esquiz*|*protocolo*rmd*) continue ;;
  esac
  log "MOVE root: $base"
  mv "$f" "$DEST/$base"
done

# Stray cosmosemiotic files inside RMD-only folders (not RMD repo)
for folder in "Análisis Realizados" "Análisis en Curso" "Curso RMD - Materiales" "Trabajos Finales" "Versiones antiguas"; do
  src_folder="$DESKTOP/$folder"
  [[ -d "$src_folder" ]] || continue
  find "$src_folder" -type f \( "${patterns[@]}" \) ! -name '~$*' -print0 | while IFS= read -r -d '' f; do
    rel="${f#"$DESKTOP/"}"
    dst="$DEST/$rel"
    mkdir -p "$(dirname "$dst")"
    log "MOVE stray: $rel"
    mv "$f" "$dst"
  done
done

# Summary
{
  echo "{"
  echo "  \"timestamp\": \"$(date -Iseconds)\","
  echo "  \"destination\": \"$DEST\","
  echo "  \"imported_files\": $(find "$DEST" -type f ! -name 'DESKTOP_CLASIFICACION.tsv' ! -name 'move_log*.json' | wc -l | tr -d ' '),"
  echo "  \"imported_size_bytes\": $(du -sk "$DEST" | awk '{print $1*1024}')"
  echo "}"
} > "$LOG"

log "DONE -> $LOG"
cat "$LOG"