#!/bin/bash

# 用法说明和参数处理
usage() {
    echo "用法: \$0 [-d 分隔符] 输入文件 输出文件"
    echo "示例: "
    echo "  TSV文件处理: \$0 -d $'\t' input.tsv output.tsv"
    echo "  CSV文件处理: \$0 -d ',' input.csv output.csv"
    exit 1
}

# 创建临时目录和清理函数
# 临时目录放在工作目录 ./tmp 下,用完即删
mkdir -p ./tmp
TEMP_DIR=$(mktemp -d ./tmp/cdhit.XXXXXX)
cleanup() {
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

# 默认分隔符为逗号
DELIM=","

# 解析命令行选项
while getopts "d:" opt; do
    case $opt in
        d) DELIM=$OPTARG ;;
        *) usage ;;
    esac
done
shift $((OPTIND-1))

# 检查剩余参数数量
if [ $# -ne 2 ]; then
    usage
fi

input_file=$1
output_file=$2

# 生成唯一的临时文件名
fasta_file="$TEMP_DIR/seq_$$.fasta"
out_fasta_file="$TEMP_DIR/seq-out_$$.fasta"
kept_lines="$TEMP_DIR/kept_lines_$$.txt"
cdhit_log="$TEMP_DIR/cdhit_$$.log"

# 检测列存在的安全方法
header=$(head -1 "$input_file")
if ! echo "$header" | tr "$DELIM" '\n' | grep -qx "protein_sequence"; then
    echo "错误: 输入文件缺少'protein_sequence'列"
    exit 1
fi

# 获取列索引（兼容不同shell）
cds_col=$(echo "$header" | awk -v delim="$DELIM" 'BEGIN {FS=delim} {for(i=1;i<=NF;i++) if($i=="protein_sequence") print i}')
#cds_col=$(echo "$header" | awk -v delim="$DELIM" 'BEGIN {FS=delim} {for(i=1;i<=NF;i++) if($i=="protein_binder") print i}')
if [ -z "$cds_col" ]; then
    echo "错误: 无法定位'protein_sequence'列"
    exit 1
fi

# 生成FASTA文件（处理包含特殊字符的字段）
awk -v delim="$DELIM" -v col="$cds_col" '
BEGIN {FS=delim}
NR>1 {
    gsub(/[[:space:]]+/, "", $col)  # 清除可能存在的空格
    printf ">%d\n%s\n", NR, $col
}' "$input_file" > "$fasta_file"

# CD-HIT聚类处理
cd-hit -i "$fasta_file" -o "$out_fasta_file" -c 0.9 -n 5 -T 16 2>&1 | tee "$cdhit_log"

# 提取保留的行号（优化性能）
grep '^>' "$out_fasta_file" | sed 's/^>//g' > "$kept_lines"

# 生成最终结果文件（优化版本 - 只读取一次文件）
{
    head -1 "$input_file"  # 输出标题行
    awk 'BEGIN {
            # 读取要保留的行号到数组中
            while (getline line_num < "'$kept_lines'") {
                keep[line_num] = 1
            }
            close("'$kept_lines'")
        }
        FNR > 1 && keep[FNR] {print}' "$input_file"
} > "$output_file"


echo "处理完成，结果保存在 $output_file"
echo "保留记录数:  $(wc -l < "$kept_lines")"
# 清理由trap自动处理，无需手动删除临时文件

