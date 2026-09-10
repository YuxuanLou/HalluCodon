#!/bin/bash
# 下载 PaxDb v5.0 蛋白序列 FASTA
# 用法: ./get-paxdb-proteins.sh <taxid> [输出文件]
# 示例: ./get-paxdb-proteins.sh 3702 plant-data/Athaliana3702/fasta.v11.5.3702.fa
set -e

if [ $# -lt 1 ]; then
    echo "用法: $0 <taxid> [输出文件]"
    echo "示例: $0 3702 plant-data/Athaliana3702/fasta.v11.5.3702.fa"
    exit 1
fi

taxid=$1
out=${2:-fasta.v11.5.${taxid}.fa}
url="https://pax-db.org/downloads/5.0/paxdb-protein-sequences-v5.0/fasta.v11.5.${taxid}.fa"

echo "下载: $url"
curl -fL --retry 3 --retry-delay 5 -o "$out" "$url"

n=$(grep -c "^>" "$out")
echo "完成: $out ($n 条序列)"
