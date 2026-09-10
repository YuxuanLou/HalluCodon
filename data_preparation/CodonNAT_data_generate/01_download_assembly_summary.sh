#!/usr/bin/env bash
#
# 下载 NCBI GenBank 全部组装汇总表 assembly_summary_genbank.txt
# 用法: ./01_download_assembly_summary.sh [输出路径]
# 默认输出到当前目录下的 assembly_summary_genbank.txt，支持断点续传。
set -euo pipefail

URL="https://ftp.ncbi.nlm.nih.gov/genomes/ASSEMBLY_REPORTS/assembly_summary_genbank.txt"
OUT="${1:-assembly_summary_genbank.txt}"
PART="${OUT}.part"

if [[ -f "$OUT" && -s "$OUT" ]]; then
    echo "[skip] 已存在: $OUT ($(du -h "$OUT" | cut -f1))"
    exit 0
fi

mkdir -p "$(dirname "$OUT")"
echo "[info] 目标: $URL"
echo "[info] 输出: $OUT"

if command -v wget >/dev/null 2>&1; then
    wget -c -t 0 --retry-connrefused --timeout=60 --waitretry=5 \
         -O "$PART" "$URL"
elif command -v curl >/dev/null 2>&1; then
    curl -L -C - --retry 5 --retry-delay 5 --connect-timeout 60 \
         -o "$PART" "$URL"
else
    echo "[error] 需要 wget 或 curl" >&2
    exit 1
fi

if [[ ! -s "$PART" ]]; then
    echo "[error] 下载结果为空，请检查网络" >&2
    rm -f "$PART"
    exit 1
fi

mv "$PART" "$OUT"
echo "[done] $OUT ($(du -h "$OUT" | cut -f1))"
