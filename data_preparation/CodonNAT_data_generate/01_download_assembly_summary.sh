#!/usr/bin/env bash
#
# Download the NCBI GenBank assembly summary table assembly_summary_genbank.txt
# Usage: ./01_download_assembly_summary.sh [output path]
# By default writes assembly_summary_genbank.txt in the current directory; supports resumable downloads.
set -euo pipefail

URL="https://ftp.ncbi.nlm.nih.gov/genomes/ASSEMBLY_REPORTS/assembly_summary_genbank.txt"
OUT="${1:-assembly_summary_genbank.txt}"
PART="${OUT}.part"

if [[ -f "$OUT" && -s "$OUT" ]]; then
    echo "[skip] Already exists: $OUT ($(du -h "$OUT" | cut -f1))"
    exit 0
fi

mkdir -p "$(dirname "$OUT")"
echo "[info] Target: $URL"
echo "[info] Output: $OUT"

if command -v wget >/dev/null 2>&1; then
    wget -c -t 0 --retry-connrefused --timeout=60 --waitretry=5 \
         -O "$PART" "$URL"
elif command -v curl >/dev/null 2>&1; then
    curl -L -C - --retry 5 --retry-delay 5 --connect-timeout 60 \
         -o "$PART" "$URL"
else
    echo "[error] wget or curl is required" >&2
    exit 1
fi

if [[ ! -s "$PART" ]]; then
    echo "[error] Download result is empty, please check the network" >&2
    rm -f "$PART"
    exit 1
fi

mv "$PART" "$OUT"
echo "[done] $OUT ($(du -h "$OUT" | cut -f1))"
