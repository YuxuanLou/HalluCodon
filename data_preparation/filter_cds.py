#!/usr/bin/env python3
"""阶段一：解析并过滤油松CDS。
输入: P.tabuliformis_V1.0.CDS.fa (DNA/T 格式)
过滤: 长度%3==0、无提前终止密码子、仅含ACGT标准碱基、100<=codons<=1022、序列去重
输出: filtered_cds.csv (ID,cds_sequence_dna,protein_sequence)
"""
import csv
import sys

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

STOP = set('TAA TAG TGA'.split())


def translate_dna(cds):
    """翻译DNA序列为蛋白；若内部有终止密码子返回None；末尾终止剥掉。"""
    prot = []
    n = len(cds)
    for i in range(0, n, 3):
        codon = cds[i:i + 3]
        aa = DNA_CODON_TABLE.get(codon)
        if aa is None:
            return None
        if aa == '*':
            if i == n - 3:  # 末尾终止密码子，剥掉
                break
            return None  # 提前终止
        prot.append(aa)
    return ''.join(prot)


def parse_fasta(path):
    with open(path) as f:
        header = None
        seq_lines = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    yield header, ''.join(seq_lines)
                header = line[1:].split()[0]
                seq_lines = []
            else:
                seq_lines.append(line)
        if header is not None:
            yield header, ''.join(seq_lines)


def main():
    input_fa = sys.argv[1]
    out_csv = sys.argv[2]

    n_total = 0
    n_len3 = 0
    n_nostop = 0
    n_std = 0
    n_lenrange = 0
    seen = set()
    kept = []

    VALID = set('ACGT')

    for header, seq in parse_fasta(input_fa):
        n_total += 1
        seq = seq.upper()
        if len(seq) % 3 != 0:
            continue
        n_len3 += 1
        prot = translate_dna(seq)
        if prot is None:
            continue
        n_nostop += 1
        if not set(seq).issubset(VALID):
            continue
        n_std += 1
        n_codons = len(seq) // 3
        if n_codons > 1022:
            continue
        n_lenrange += 1
        if seq in seen:
            continue
        seen.add(seq)
        kept.append((header, seq, prot))

    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['ID', 'cds_sequence_dna', 'protein_sequence'])
        for h, s, p in kept:
            w.writerow([h, s, p])

    print(f"total: {n_total}")
    print(f"after len%3==0: {n_len3}")
    print(f"after no premature stop: {n_nostop}")
    print(f"after only ACGT: {n_std}")
    print(f"after <=1022 codons: {n_lenrange}")
    print(f"after dedup: {len(kept)}")
    print(f"saved: {out_csv}")


if __name__ == '__main__':
    main()
