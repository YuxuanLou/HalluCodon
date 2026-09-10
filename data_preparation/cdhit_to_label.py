#!/usr/bin/env python3
"""阶段四 step3：从 cd-hit 输出生成最终 CodonEXP 训练CSV。
输入: Pinus_cds_label_matched.csv (ID,cds_sequence_dna,protein_sequence,label)
      matched_prot_cdhit.fasta.clstr (cd-hit聚类结果)
逻辑: 每个簇的代表序列(带*的) 保留其 label；同簇内其他序列丢弃。
输出: Pinus_cds_label.csv: ID,cds_sequence,protein_sequence,label (DNA/T格式)
"""
import csv
import re
import sys


def parse_clstr(clstr_file):
    """解析 clstr 文件，返回每个代表序列的 seqN 编号（带*）。"""
    reps = []
    with open(clstr_file) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('>Cluster'):
                continue
            if line.rstrip().endswith('*'):
                # 形如 0	1021aa, >seq3181|PtXG09320|l... *
                m = re.search(r'>seq(\d+)\|', line)
                if m:
                    reps.append(int(m.group(1)))
    return reps


def main():
    matched_csv = sys.argv[1]
    clstr_file = sys.argv[2]
    out_csv = sys.argv[3]

    reps = parse_clstr(clstr_file)
    print(f"representatives from cd-hit: {len(reps)}")

    # 从 matched 表读回完整记录，按 seqN 索引
    by_seq = {}
    with open(matched_csv) as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            by_seq[i] = row
    print(f"matched rows: {len(by_seq)}")

    # 组装输出
    out_rows = []
    for seqn in reps:
        rec = by_seq[seqn]
        out_rows.append([rec['ID'], rec['cds_sequence_dna'], rec['protein_sequence'], rec['label']])

    # 按 ID 排序保持稳定
    out_rows.sort(key=lambda x: x[0])

    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['ID', 'cds_sequence', 'protein_sequence', 'label'])
        for r in out_rows:
            w.writerow(r)
    print(f"saved: {out_csv}, rows: {len(out_rows)}")

    from collections import Counter
    print("label counts:", dict(Counter(r[3] for r in out_rows)))


if __name__ == '__main__':
    main()
