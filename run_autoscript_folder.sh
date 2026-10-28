#!/usr/bin/env bash
# Generate script (.txt) dari semua video Bahasa Indonesia di sebuah folder.
# Script disimpan di folder yang sama dengan masing-masing videonya.
#
# Pakai:
#   ./run_autoscript_folder.sh <folder_video> [opsi tambahan untuk autoscript.py]
#
# Contoh:
#   ./run_autoscript_folder.sh "~/Desktop/kelas"
#   ./run_autoscript_folder.sh ./videos --srt --timestamps --whisper-model small

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Pakai: $0 <folder_video> [opsi tambahan untuk autoscript.py]" >&2
    exit 1
fi

SOURCE_DIR="${1/#\~/$HOME}"
shift

AUTOSCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$AUTOSCRIPT_DIR/.venv/bin/python" "$AUTOSCRIPT_DIR/autoscript.py" --source "$SOURCE_DIR" "$@"
