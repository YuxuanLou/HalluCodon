#!/usr/bin/env python3
"""阶段四 step1：丰度表匹配 CDS。
输入: 蛋白丰度排序.csv (target_id,Intensity，降序), filtered_cds.csv (ID,cds,prot)
逻辑:
  1) 丰度表排除 Intensity=0
  2) CDS ID 去掉 .X 后缀 (Pt1G42640.1 -> Pt1G42640) 与丰度表 ID 匹配
  3) 只保留匹配上的 CDS
  4) 匹配后数据 前1/3 -> label=1, 后1/3 -> label=0, 中间丢弃
输出: Pinus_cds_label_matched.csv: ID,cds_sequence_dna,protein_sequence,label
"""
import csv
import sys


def main():
    ab_file = sys.argv[1]
    cds_file = sys.argv[2]
    out_file = sys.argv[3]

    # 1. 读取丰度表，排除 Intensity=0
    abundance = []  # (target_id, intensity)
    with open(ab_file) as f:
        r = csv.DictReader(f)
        for row in r:
            intensity = float(row['Intensity'])
            if intensity > 0:
                abundance.append((row['target_id'].strip(), intensity))
    # 丰度表已是降序（Intensity从大到小）
    print(f"abundance entries (>0): {len(abundance)}")

    # 建丰度 ID 集合
    ab_ids = {tid for tid, _ in abundance}
    ab_rank = {tid: i for i, (tid, _) in enumerate(abundance)}

    # 2. 读取过滤后 CDS，按去后缀 ID 匹配
    matched = []  # (gene_id, intensity, cds, prot)
    unmatched = 0
    with open(cds_file) as f:
        r = csv.DictReader(f)
        for row in r:
            cds_id = row['ID']
            gene_id = cds_id.split('.')[0]
            if gene_id in ab_ids:
                matched.append((gene_id, ab_rank[gene_id], row['cds_sequence_dna'], row['protein_sequence']))
            else:
                unmatched += 1
    print(f"CDS total: {sum(1 for _ in open(cds_file))-1}")
    print(f"matched: {len(matched)}, unmatched: {unmatched}")

    # 3. 按丰度排名排序（降序丰度）
    matched.sort(key=lambda x: x[1])

    # 4. 前1/3 label=1, 后1/3 label=0, 中间丢弃
    n = len(matched)
    third = n // 3
    high = matched[:third]
    low = matched[n - third:]
    print(f"n={n}, third={third}, high={len(high)}, low={len(low)}")

    with open(out_file, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['ID', 'cds_sequence_dna', 'protein_sequence', 'label'])
        for gene_id, rank, cds, prot in high:
            w.writerow([gene_id, cds, prot, 1])
        for gene_id, rank, cds, prot in low:
            w.writerow([gene_id, cds, prot, 0])
    print(f"saved: {out_file}")


if __name__ == '__main__':
    main()
