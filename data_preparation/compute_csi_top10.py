#!/usr/bin/env python3
"""阶段三：计算CSI（CAI逻辑），取top 10%。
参照频率表: codon_freq/Pinus-codon-count.csv (aa,codon,count,frequency%)，RNA/U格式
输入: filtered_cds.csv (DNA/T)
输出: Pinus_cds_top10_U.csv: cds_sequence,protein_sequence,csi_value (RNA/U格式)
"""
import csv
import math
import sys
from collections import defaultdict

DNA_TO_RNA = str.maketrans('T', 'U')


def load_codon_table(codon_count_file):
    """构建相对适应度表 {密码子(RNA): w}，w=count/max_count(aa)"""
    aa_max_count = defaultdict(float)
    rows = []
    with open(codon_count_file) as f:
        r = csv.DictReader(f)
        for row in r:
            aa = row['aa']
            codon = row['codon']
            count = float(row['count'])
            rows.append((aa, codon, count))
            if count > aa_max_count[aa]:
                aa_max_count[aa] = count
    codon_table = {}
    for aa, codon, count in rows:
        if aa_max_count[aa] > 0:
            codon_table[codon] = count / aa_max_count[aa]
        else:
            codon_table[codon] = 0.0
    return codon_table


def calculate_csi(cds_dna, codon_table):
    """CAI: exp(mean(ln w))，跳过 w=0 的密码子"""
    total_ln_w = 0.0
    valid = 0
    for i in range(0, len(cds_dna), 3):
        dna_codon = cds_dna[i:i + 3].upper()
        rna_codon = dna_codon.translate(DNA_TO_RNA)
        w = codon_table.get(rna_codon, 0.0)
        if w > 0:
            total_ln_w += math.log(w)
            valid += 1
    if valid > 0:
        return math.exp(total_ln_w / valid)
    return 0.0


def main():
    in_csv = sys.argv[1]
    freq_csv = sys.argv[2]
    out_csv = sys.argv[3]

    codon_table = load_codon_table(freq_csv)
    print(f"codon_table size: {len(codon_table)}")

    rows = []
    with open(in_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            cds = row['cds_sequence_dna']
            csi = calculate_csi(cds, codon_table)
            rows.append((csi, cds, row['protein_sequence']))

    rows.sort(key=lambda x: x[0], reverse=True)
    n = len(rows)
    top10 = rows[:int(n * 0.1)]

    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['cds_sequence', 'protein_sequence', 'csi_value'])
        for csi, cds, prot in top10:
            w.writerow([cds.translate(DNA_TO_RNA), prot, f"{csi:.6f}"])

    csi_list = [r[0] for r in rows]
    print(f"total: {n}, top10: {len(top10)}")
    print(f"csi range (all): {min(csi_list):.4f} - {max(csi_list):.4f}")
    print(f"csi top10 min: {top10[-1][0]:.4f}")
    print(f"saved: {out_csv}")


if __name__ == '__main__':
    main()
