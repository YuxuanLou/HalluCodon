#!/bin/bash
# Download a PaxDb v5.0 dataset txt
# Usage:
#   ./get-paxdb-dataset.sh <taxid> list                  # List all datasets for the species
#   ./get-paxdb-dataset.sh <taxid> <dataset name> [output file]   # Download a single dataset
# Dataset names have no taxid prefix, e.g. FLOWER-integrated / WHOLE_ORGANISM-integrated
# Examples:
#   ./get-paxdb-dataset.sh 3702 list
#   ./get-paxdb-dataset.sh 3702 FLOWER-integrated plant-data/Athaliana3702/3702-FLOWER-integrated.txt
set -e

if [ $# -lt 2 ]; then
    echo "Usage:"
    echo "  $0 <taxid> list                  # List all datasets for the species"
    echo "  $0 <taxid> <dataset name> [output file]   # Download a single dataset"
    echo "Example: $0 3702 FLOWER-integrated plant-data/Athaliana3702/3702-FLOWER-integrated.txt"
    exit 1
fi

taxid=$1
name=$2
base="https://pax-db.org/downloads/5.0/datasets/${taxid}"

if [ "$name" = "list" ]; then
    curl -s "$base/" | grep -oE 'href="[^"]+"' | sed 's/href="//;s/"$//' | grep -v '^\.\.$'
    exit 0
fi

# Normalize: tolerate inputs with a taxid prefix or a .txt suffix
name=${name#${taxid}-}
name=${name%.txt}

out=${3:-${taxid}-${name}.txt}
url="${base}/${taxid}-${name}.txt"

echo "Downloading: $url"
curl -fL --retry 3 --retry-delay 5 -o "$out" "$url"

n=$(grep -vc "^#" "$out" || true)
echo "Done: $out ($n data lines)"
