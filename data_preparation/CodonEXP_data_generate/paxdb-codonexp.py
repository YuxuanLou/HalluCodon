#!/usr/bin/env python3
"""PaxDb 蛋白丰度数据集 → CodonEXP 流程(单数据集)

步骤:
  1. 解析 PaxDb 数据集 txt(string_external_id + abundance)
  2. 按丰度降序: 前 1/3 标记 High(label=1), 后 1/3 标记 Low(label=0), 中间 1/3 丢弃
  3. 从 PaxDb 蛋白序列 FASTA 提取对应蛋白序列(缺序列的丢弃)
  4. 用修复后的 get_cds-batch-blast.py 匹配 CDS(默认 ≥90% 相似度、≥50% 覆盖度)
  5. cd-hit 0.9 按 protein_sequence 去冗余(同 cd-hit-pro.sh),输出最终 CSV

用法:
  python3 paxdb-codonexp.py --cds CDS_FASTA --dataset PAXDB_TXT \
      --proteins PAXDB_PROTEIN_FASTA --out OUT.csv [--threshold 90] [--coverage 50] [--parallel 8]

依赖: pandas / biopython / blastp / makeblastdb / cd-hit
"""
import argparse
import os
import subprocess
import sys

import pandas as pd
from Bio import SeqIO

HERE = os.path.dirname(os.path.abspath(__file__))
GET_CDS = os.path.join(HERE, "get_cds-batch-blast.py")
CDHIT = os.path.join(HERE, "cd-hit-pro.sh")


def parse_dataset(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            try:
                abundance = float(parts[1])
            except ValueError:
                continue
            rows.append((parts[0], abundance))
    return rows


def label_high_low(rows):
    ranked = sorted(rows, key=lambda x: x[1], reverse=True)
    k = len(ranked) // 3
    if k == 0:
        raise ValueError(f"数据量过少({len(ranked)} 条), 无法划分三分位")
    labeled = []
    for pid, abundance in ranked[:k]:
        labeled.append({"id": pid, "abundance": abundance, "exp": "high", "label": 1})
    for pid, abundance in ranked[-k:]:
        labeled.append({"id": pid, "abundance": abundance, "exp": "low", "label": 0})
    return labeled


def load_proteins(fasta):
    seqs = {}
    for rec in SeqIO.parse(fasta, "fasta"):
        seqs[rec.id] = str(rec.seq)
    return seqs


def main():
    ap = argparse.ArgumentParser(
        description="PaxDb 丰度 → 高/低丰度蛋白 → CDS 匹配 → cd-hit 去冗余",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--cds", required=True, help="CDS FASTA 文件")
    ap.add_argument("--dataset", required=True,
                    help="PaxDb 数据集 txt(如 3702-FLOWER-integrated.txt)")
    ap.add_argument("--proteins", required=True,
                    help="PaxDb 蛋白序列 FASTA(如 fasta.v11.5.3702.fa)")
    ap.add_argument("--out", required=True, help="最终输出 CSV")
    ap.add_argument("--threshold", type=float, default=90.0, help="相似度阈值(默认 90)")
    ap.add_argument("--coverage", type=float, default=50.0, help="覆盖度阈值(默认 50)")
    ap.add_argument("--parallel", type=int, default=8, help="blastp 线程数(默认 8)")
    ap.add_argument("--workdir", default=None,
                    help="中间文件目录(默认 tmp/paxdb-codonexp-work/<数据集名>)")
    args = ap.parse_args()

    base = os.path.splitext(os.path.basename(args.dataset))[0]
    work = args.workdir or os.path.join(HERE, "tmp", "paxdb-codonexp-work", base)
    os.makedirs(work, exist_ok=True)

    rows = parse_dataset(args.dataset)
    print(f"[1/5] 数据集 {args.dataset}: {len(rows)} 条蛋白丰度记录")

    labeled = label_high_low(rows)
    n_high = sum(1 for r in labeled if r["exp"] == "high")
    n_low = len(labeled) - n_high
    print(f"[2/5] 标注: High={n_high}, Low={n_low}, 丢弃中间 {len(rows) - len(labeled)} 条")

    seqs = load_proteins(args.proteins)
    print(f"[3/5] 蛋白序列库 {args.proteins}: {len(seqs)} 条")
    qdf = pd.DataFrame(labeled)
    qdf["seq"] = qdf["id"].map(seqs)
    n_missing = int(qdf["seq"].isna().sum())
    qdf = qdf.dropna(subset=["seq"]).reset_index(drop=True)
    print(f"      缺序列丢弃 {n_missing} 条, 剩余查询 {len(qdf)} 条")
    if len(qdf) == 0:
        raise ValueError("没有可用的查询蛋白序列")
    query_csv = os.path.join(work, "query.csv")
    qdf[["id", "abundance", "seq", "exp", "label"]].to_csv(query_csv, index=False)

    matched_csv = os.path.join(work, "matched.csv")
    print(f"[4/5] 匹配 CDS (similarity>={args.threshold}%, coverage>={args.coverage}%) ...")
    subprocess.run(
        [sys.executable, GET_CDS, query_csv, args.cds, matched_csv,
         "--threshold", str(args.threshold),
         "--coverage", str(args.coverage),
         "--parallel", str(args.parallel)],
        check=True)
    n_matched = max(0, sum(1 for _ in open(matched_csv)) - 1)
    print(f"      匹配成功 {n_matched} 条")

    print("[5/5] cd-hit 0.9 去冗余 ...")
    subprocess.run(["bash", CDHIT, "-d", ",", matched_csv, args.out], check=True)
    n_final = max(0, sum(1 for _ in open(args.out)) - 1)
    print(f"完成: {args.out} ({n_final} 条)")
    print(f"中间文件: {query_csv}, {matched_csv}")


if __name__ == "__main__":
    main()
