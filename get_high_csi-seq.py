import pandas as pd
import numpy as np
import argparse
import math
from collections import defaultdict


def load_codon_table(codon_count_file):
    """从密码子频率文件中构建密码子表"""
    codon_table = {}
    aa_max_count = defaultdict(float)

    # 读取密码子频率文件
    df = pd.read_csv(codon_count_file)

    # 预处理：计算每个氨基酸的最常用密码子频率
    for _, row in df.iterrows():
        aa = row['氨基酸']
        codon = row['密码子']
        count = row['计数']
        if count > aa_max_count[aa]:
            aa_max_count[aa] = count

    # 构建密码子表：{密码子: (氨基酸, 相对适应性)}
    for _, row in df.iterrows():
        aa = row['氨基酸']
        codon = row['密码子']
        count = row['计数']
        if aa_max_count[aa] > 0:
            w_ij = count / aa_max_count[aa]
        else:
            w_ij = 0.0
        codon_table[codon] = (aa, w_ij)

    return codon_table


def calculate_csi(cds_sequence, codon_table):
    """计算单个CDS序列的CSI值"""
    total_ln_w = 0.0
    valid_codons = 0

    for i in range(0, len(cds_sequence), 3):
        codon = cds_sequence[i:i + 3].upper()
        _, w_ij = codon_table[codon]
        if w_ij > 0:
            total_ln_w += math.log(w_ij)
            valid_codons += 1

    if valid_codons > 0:
        csi = math.exp(total_ln_w / valid_codons)
    else:
        csi = 0.0

    return csi


def main():
    parser = argparse.ArgumentParser(
        description='Calculate CSI for CDS sequences and output top 10%.')
    parser.add_argument('--input', required=True,
                        help='Input CSV file with cds_sequence and protein_sequence columns.')
    parser.add_argument('--codon_freq',
                        required=True,
                        help='CSV file with codon frequency data (amino_acid,codon,count,frequency).')
    parser.add_argument('--output', required=True,
                        help='Output CSV file for top 10% CSI sequences.')
    args = parser.parse_args()

    # 加载密码子表
    codon_table = load_codon_table(
        args.codon_freq)

    # 读取输入序列文件
    df = pd.read_csv(args.input)

    # 计算每条序列的CSI
    df['csi_value'] = df['cds_sequence'].apply(
        lambda x: calculate_csi(x, codon_table))

    # 按CSI值降序排序并取前10%
    df_sorted = df.sort_values(by='csi_value',
                               ascending=False)
    top_10_percent = df_sorted.head(
        int(len(df_sorted) * 0.1))

    # 保存结果
    top_10_percent[
        ['cds_sequence', 'protein_sequence',
         'csi_value']].to_csv(args.output,
                              index=False)
    print(
        f"Top 10% CSI sequences saved to {args.output}")


if __name__ == '__main__':
    main()

