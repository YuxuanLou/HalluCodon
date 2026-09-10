#!/bin/bash
# Download the PaxDb v5.0 protein sequence FASTA
# Usage: ./get-paxdb-proteins.sh <taxid> [output file]
# Example: ./get-paxdb-proteins.sh 3702 plant-data/Athaliana3702/fasta.v11.5.3702.fa
set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <taxid> [output file]"
    echo "Example: $0 3702 plant-data/Athaliana3702/fasta.v11.5.3702.fa"
    exit 1
fi

taxid=$1
out=${2:-fasta.v11.5.${taxid}.fa}
url="https://pax-db.org/downloads/5.0/paxdb-protein-sequences-v5.0/fasta.v11.5.${taxid}.fa"

echo "Downloading: $url"
curl -fL --retry 3 --retry-delay 5 -o "$out" "$url"

n=$(grep -c "^>" "$out")
echo "Done: $out ($n sequences)"
