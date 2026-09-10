#!/usr/bin/env python3
"""阶段二：统计全基因组密码子频率表（对齐 codon_freq/*.csv 格式）。
输入: filtered_cds.csv (DNA/T)
输出: Pinus_codon_count.csv: aa,codon,count,frequency(%)，RNA/U格式，含终止子*
"""
import csv
import sys
from collections import Counter, defaultdict

DNA_TO_RNA = str.maketrans('T', 'U')

DNA_CODON_TABLE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}


def main():
    in_csv = sys.argv[1]
    out_csv = sys.argv[2]

    counter = Counter()
    with open(in_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            cds = row['cds_sequence_dna'].upper()
            for i in range(0, len(cds), 3):
                counter[cds[i:i + 3]] += 1

    # 统计到氨基酸
    aa_codons = defaultdict(list)
    for dna_codon, count in counter.items():
        aa = DNA_CODON_TABLE.get(dna_codon, 'X')
        rna_codon = dna_codon.translate(DNA_TO_RNA)
        aa_codons[aa].append((rna_codon, count))

    # 写出
    rows = []
    for aa in sorted(aa_codons.keys()):
        entries = aa_codons[aa]
        total = sum(c for _, c in entries)
        for rna_codon, count in sorted(entries, key=lambda x: -x[1]):
            freq = count / total * 100 if total else 0.0
            rows.append([aa, rna_codon, count, round(freq, 2)])

    # 标准氨基酸顺序
    aa_order = 'ACDEFGHIKLMNPQRSTVWY*'
    def sort_key(r):
        return aa_order.index(r[0]) if r[0] in aa_order else 99

    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['aa', 'codon', 'count', 'frequency(%)'])
        for r in sorted(rows, key=sort_key):
            w.writerow(r)

    total_codons = sum(counter.values())
    print(f"total codons counted: {total_codons}")
    print(f"distinct codons: {len(counter)}")
    print(f"saved: {out_csv}")


if __name__ == '__main__':
    main()
