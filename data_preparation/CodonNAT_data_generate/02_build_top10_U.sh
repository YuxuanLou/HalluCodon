#!/usr/bin/env bash
#
# 主流程: 输入一个分类名（如 Enterobacteriaceae / Escherichia），
#   1) 确保有 assembly_summary_genbank.txt（没有则调用 01 脚本下载）
#   2) 确保有 NCBI taxonomy（nodes.dmp / names.dmp，没有则自动下载）
#   3) 按谱系筛选该分类下的物种（默认每个物种取一个代表基因组）
#   4) 并行下载各基因组 CDS -> 过滤/翻译/按蛋白去重
#   5) 统计密码子使用频率
#   6) 计算 CSI 并取前百分比（默认前 10%）
#   7) T 转 U，最终产出 top10_U.csv
#
# 用法:
#   ./02_build_top10_U.sh <分类名> [-p 百分比] [-j JOBS] [-a assembly_summary.txt] [-o 输出目录]
# 示例:
#   ./02_build_top10_U.sh Enterobacteriaceae
#   ./02_build_top10_U.sh Escherichia -p 10 -j 8      # 前 10%（默认，与原脚本一致）
#   ./02_build_top10_U.sh Enterobacteriaceae -a /path/to/assembly_summary_genbank.txt
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="$ROOT/scripts"

TAXON="${1:-}"
if [[ -z "$TAXON" ]]; then
    echo "用法: $0 <分类名> [-p 百分比] [-j JOBS] [-a assembly_summary.txt] [-o 输出目录]" >&2
    echo "示例: $0 Enterobacteriaceae" >&2
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
    ''|*[!0-9.]*) echo "[error] -p 必须是正数（百分比）" >&2; exit 1 ;;
esac
case "$JOBS" in
    ''|*[!0-9]*) echo "[error] -j 必须是正整数" >&2; exit 1 ;;
esac

WORK="$ROOT/work/$TAXON"
TAX_DIR="$ROOT/data/taxdump"
mkdir -p "$ROOT/data" "$TAX_DIR" "$WORK" "$OUT_DIR"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# ---------------------------------------------------------------- 1. assembly summary
if [[ ! -f "$ASSEMBLY" || ! -s "$ASSEMBLY" ]]; then
    log "缺少 assembly_summary_genbank.txt，开始下载..."
    "$ROOT/01_download_assembly_summary.sh" "$ASSEMBLY"
else
    log "使用已有 assembly summary: $ASSEMBLY ($(du -h "$ASSEMBLY" | cut -f1))"
fi

# ---------------------------------------------------------------- 2. taxonomy
if [[ ! -f "$TAX_DIR/nodes.dmp" || ! -f "$TAX_DIR/names.dmp" ]]; then
    log "缺少 NCBI taxonomy 数据，下载 taxdump.tar.gz ..."
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
    log "taxonomy 就绪: $TAX_DIR"
else
    log "使用已有 taxonomy 数据: $TAX_DIR"
fi

# ---------------------------------------------------------------- 3. 筛选物种
log "按分类 [$TAXON] 筛选物种（每物种取一个代表基因组）..."
python3 "$SCRIPTS/select_species.py" \
    --assembly-summary "$ASSEMBLY" \
    --taxonomy-dir "$TAX_DIR" \
    --taxon "$TAXON" \
    --out "$WORK/species.txt"

N_ITEMS=$(wc -l < "$WORK/species.txt")
if [[ "$N_ITEMS" -eq 0 ]]; then
    echo "[error] 分类 [$TAXON] 没有匹配到任何物种" >&2
    exit 1
fi
log "筛选到 $N_ITEMS 条记录，物种列表: $WORK/species.txt"

# ---------------------------------------------------------------- 4. 下载 CDS + 过滤翻译去重
log "下载 CDS 并过滤/翻译/去重（并发 $JOBS）..."
python3 "$SCRIPTS/fetch_and_filter.py" \
    --list "$WORK/species.txt" \
    --out "$WORK/unique.tsv" \
    --jobs "$JOBS" \
    --cache-dir "$WORK/cds_cache"

N_UNIQUE=$(($(wc -l < "$WORK/unique.tsv") - 1))
if [[ "$N_UNIQUE" -le 0 ]]; then
    echo "[error] 没有生成任何有效序列，请检查下载日志: $WORK/fetch_failed.txt" >&2
    exit 1
fi
log "去重后共有 $N_UNIQUE 条唯一蛋白序列"

# ---------------------------------------------------------------- 5. 密码子频率
log "统计密码子使用频率..."
python3 "$SCRIPTS/codon_count.py" \
    --input "$WORK/unique.tsv" \
    --output "$WORK/codon.csv"

# ---------------------------------------------------------------- 6. CSI 筛选
log "计算 CSI 并取前 $PERCENT%..."
python3 "$SCRIPTS/calculate_csi.py" \
    --input "$WORK/unique.tsv" \
    --codon "$WORK/codon.csv" \
    --output "$WORK/cds_top${PERCENT}.csv" \
    --percent "$PERCENT"
CSI_CSV="$WORK/cds_top${PERCENT}.csv"
FINAL="top${PERCENT}_U.csv"

# ---------------------------------------------------------------- 7. T -> U
log "T 转 U，生成最终文件..."
python3 "$SCRIPTS/t2u.py" \
    "$CSI_CSV" \
    "$OUT_DIR/$FINAL"

log "完成! 最终输出: $OUT_DIR/$FINAL"
