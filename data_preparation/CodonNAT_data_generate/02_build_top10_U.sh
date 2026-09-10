#!/usr/bin/env bash
#
# Main pipeline: takes a taxon name (e.g. Enterobacteriaceae / Escherichia), then
#   1) Ensures assembly_summary_genbank.txt exists (calls the 01 script to download it if not)
#   2) Ensures the NCBI taxonomy (nodes.dmp / names.dmp) exists (downloads it automatically if not)
#   3) Filters species under the taxon by lineage (by default one representative genome per species)
#   4) Downloads each genome's CDS in parallel -> filter/translate/deduplicate by protein
#   5) Counts codon usage frequencies
#   6) Computes CSI and keeps the top percentage (default top 10%)
#   7) Converts T to U, producing the final top10_U.csv
#
# Usage:
#   ./02_build_top10_U.sh <taxon> [-p percent] [-j JOBS] [-a assembly_summary.txt] [-o output dir]
# Examples:
#   ./02_build_top10_U.sh Enterobacteriaceae
#   ./02_build_top10_U.sh Escherichia -p 10 -j 8      # top 10% (default, same as the original script)
#   ./02_build_top10_U.sh Enterobacteriaceae -a /path/to/assembly_summary_genbank.txt
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="$ROOT/scripts"

TAXON="${1:-}"
if [[ -z "$TAXON" ]]; then
    echo "Usage: $0 <taxon> [-p percent] [-j JOBS] [-a assembly_summary.txt] [-o output dir]" >&2
    echo "Example: $0 Enterobacteriaceae" >&2
    echo "      $0 Escherichia -p 10 -j 8" >&2
    exit 1
fi
shift

PERCENT=10
JOBS=4
ASSEMBLY="$ROOT/data/assembly_summary_genbank.txt"
OUT_DIR="$ROOT/output"

while getopts "p:j:a:o:" opt; do
    case "$opt" in
        p) PERCENT="$OPTARG" ;;
        j) JOBS="$OPTARG" ;;
        a) ASSEMBLY="$OPTARG" ;;
        o) OUT_DIR="$OPTARG" ;;
        *) exit 1 ;;
    esac
done

case "$PERCENT" in
    ''|*[!0-9.]*) echo "[error] -p must be a positive number (percentage)" >&2; exit 1 ;;
esac
case "$JOBS" in
    ''|*[!0-9]*) echo "[error] -j must be a positive integer" >&2; exit 1 ;;
esac

WORK="$ROOT/work/$TAXON"
TAX_DIR="$ROOT/data/taxdump"
mkdir -p "$ROOT/data" "$TAX_DIR" "$WORK" "$OUT_DIR"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# ---------------------------------------------------------------- 1. assembly summary
if [[ ! -f "$ASSEMBLY" || ! -s "$ASSEMBLY" ]]; then
    log "assembly_summary_genbank.txt is missing, downloading..."
    "$ROOT/01_download_assembly_summary.sh" "$ASSEMBLY"
else
    log "Using existing assembly summary: $ASSEMBLY ($(du -h "$ASSEMBLY" | cut -f1))"
fi

# ---------------------------------------------------------------- 2. taxonomy
if [[ ! -f "$TAX_DIR/nodes.dmp" || ! -f "$TAX_DIR/names.dmp" ]]; then
    log "NCBI taxonomy data is missing, downloading taxdump.tar.gz ..."
    TAX_TGZ="$ROOT/data/taxdump.tar.gz"
    TAX_URL="https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz"
    if command -v wget >/dev/null 2>&1; then
        wget -c -t 0 --retry-connrefused --timeout=60 --waitretry=5 \
             -O "$TAX_TGZ" "$TAX_URL"
    else
        curl -L -C - --retry 5 --retry-delay 5 --connect-timeout 60 \
             -o "$TAX_TGZ" "$TAX_URL"
    fi
    tar -xzf "$TAX_TGZ" -C "$TAX_DIR" nodes.dmp names.dmp
    log "Taxonomy ready: $TAX_DIR"
else
    log "Using existing taxonomy data: $TAX_DIR"
fi

# ---------------------------------------------------------------- 3. Filter species
log "Filtering species by taxon [$TAXON] (one representative genome per species)..."
python3 "$SCRIPTS/select_species.py" \
    --assembly-summary "$ASSEMBLY" \
    --taxonomy-dir "$TAX_DIR" \
    --taxon "$TAXON" \
    --out "$WORK/species.txt"

N_ITEMS=$(wc -l < "$WORK/species.txt")
if [[ "$N_ITEMS" -eq 0 ]]; then
    echo "[error] No species matched taxon [$TAXON]" >&2
    exit 1
fi
log "Filtered $N_ITEMS records, species list: $WORK/species.txt"

# ---------------------------------------------------------------- 4. Download CDS + filter/translate/dedup
log "Downloading CDS and filtering/translating/deduplicating ($JOBS in parallel)..."
python3 "$SCRIPTS/fetch_and_filter.py" \
    --list "$WORK/species.txt" \
    --out "$WORK/unique.tsv" \
    --jobs "$JOBS" \
    --cache-dir "$WORK/cds_cache"

N_UNIQUE=$(($(wc -l < "$WORK/unique.tsv") - 1))
if [[ "$N_UNIQUE" -le 0 ]]; then
    echo "[error] No valid sequences were generated, check the download log: $WORK/fetch_failed.txt" >&2
    exit 1
fi
log "$N_UNIQUE unique protein sequences after deduplication"

# ---------------------------------------------------------------- 5. Codon frequencies
log "Counting codon usage frequencies..."
python3 "$SCRIPTS/codon_count.py" \
    --input "$WORK/unique.tsv" \
    --output "$WORK/codon.csv"

# ---------------------------------------------------------------- 6. CSI selection
log "Computing CSI and keeping the top $PERCENT%..."
python3 "$SCRIPTS/calculate_csi.py" \
    --input "$WORK/unique.tsv" \
    --codon "$WORK/codon.csv" \
    --output "$WORK/cds_top${PERCENT}.csv" \
    --percent "$PERCENT"
CSI_CSV="$WORK/cds_top${PERCENT}.csv"
FINAL="top${PERCENT}_U.csv"

# ---------------------------------------------------------------- 7. T -> U
log "Converting T to U and generating the final file..."
python3 "$SCRIPTS/t2u.py" \
    "$CSI_CSV" \
    "$OUT_DIR/$FINAL"

log "Done! Final output: $OUT_DIR/$FINAL"
