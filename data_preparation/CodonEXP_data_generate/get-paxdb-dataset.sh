#!/bin/bash
# 下载 PaxDb v5.0 数据集 txt
# 用法:
#   ./get-paxdb-dataset.sh <taxid> list                  # 列出该物种所有数据集
#   ./get-paxdb-dataset.sh <taxid> <数据集名> [输出文件]   # 下载单个数据集
# 数据集名不带 taxid 前缀, 如 FLOWER-integrated / WHOLE_ORGANISM-integrated
# 示例:
#   ./get-paxdb-dataset.sh 3702 list
#   ./get-paxdb-dataset.sh 3702 FLOWER-integrated plant-data/Athaliana3702/3702-FLOWER-integrated.txt
set -e

if [ $# -lt 2 ]; then
    echo "用法:"
    echo "  $0 <taxid> list                  # 列出该物种所有数据集"
    echo "  $0 <taxid> <数据集名> [输出文件]   # 下载单个数据集"
    echo "示例: $0 3702 FLOWER-integrated plant-data/Athaliana3702/3702-FLOWER-integrated.txt"
    exit 1
fi

taxid=$1
name=$2
base="https://pax-db.org/downloads/5.0/datasets/${taxid}"

if [ "$name" = "list" ]; then
    curl -s "$base/" | grep -oE 'href="[^"]+"' | sed 's/href="//;s/"$//' | grep -v '^\.\.$'
    exit 0
fi

# 归一化: 容忍带 taxid 前缀或 .txt 后缀的输入
name=${name#${taxid}-}
name=${name%.txt}

out=${3:-${taxid}-${name}.txt}
url="${base}/${taxid}-${name}.txt"

echo "下载: $url"
curl -fL --retry 3 --retry-delay 5 -o "$out" "$url"

n=$(grep -vc "^#" "$out" || true)
echo "完成: $out ($n 行数据)"
