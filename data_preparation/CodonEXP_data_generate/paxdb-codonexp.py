#!/usr/bin/env python3
"""PaxDb protein abundance dataset → CodonEXP pipeline (single dataset)

Steps:
  1. Parse the PaxDb dataset txt (string_external_id + abundance)
  2. Sort by abundance descending: top 1/3 labeled High (label=1), bottom 1/3 labeled Low (label=0), middle 1/3 discarded
  3. Extract the corresponding protein sequences from the PaxDb protein FASTA (entries without a sequence are dropped)
  4. Match CDS with the fixed get_cds-batch-blast.py (default: similarity ≥90%, coverage ≥50%)
  5. Remove redundancy with cd-hit 0.9 on protein_sequence (same as cd-hit-pro.sh) and write the final CSV

Usage:
  python3 paxdb-codonexp.py --cds CDS_FASTA --dataset PAXDB_TXT \
      --proteins PAXDB_PROTEIN_FASTA --out OUT.csv [--threshold 90] [--coverage 50] [--parallel 8]

Dependencies: pandas / biopython / blastp / makeblastdb / cd-hit
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
        raise ValueError(f"Too few records ({len(ranked)}); cannot split into terciles")
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
        description="PaxDb abundance → high/low abundance proteins → CDS matching → cd-hit de-redundancy",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--cds", required=True, help="CDS FASTA file")
    ap.add_argument("--dataset", required=True,
                    help="PaxDb dataset txt (e.g. 3702-FLOWER-integrated.txt)")
    ap.add_argument("--proteins", required=True,
                    help="PaxDb protein sequence FASTA (e.g. fasta.v11.5.3702.fa)")
    ap.add_argument("--out", required=True, help="Final output CSV")
    ap.add_argument("--threshold", type=float, default=90.0, help="Similarity threshold (default 90)")
    ap.add_argument("--coverage", type=float, default=50.0, help="Coverage threshold (default 50)")
    ap.add_argument("--parallel", type=int, default=8, help="Number of blastp threads (default 8)")
    ap.add_argument("--workdir", default=None,
                    help="Directory for intermediate files (default tmp/paxdb-codonexp-work/<dataset name>)")
    args = ap.parse_args()

    base = os.path.splitext(os.path.basename(args.dataset))[0]
    work = args.workdir or os.path.join(HERE, "tmp", "paxdb-codonexp-work", base)
    os.makedirs(work, exist_ok=True)

    rows = parse_dataset(args.dataset)
    print(f"[1/5] Dataset {args.dataset}: {len(rows)} protein abundance records")

    labeled = label_high_low(rows)
    n_high = sum(1 for r in labeled if r["exp"] == "high")
    n_low = len(labeled) - n_high
    print(f"[2/5] Labeling: High={n_high}, Low={n_low}, discarded middle {len(rows) - len(labeled)} records")

    seqs = load_proteins(args.proteins)
    print(f"[3/5] Protein sequence library {args.proteins}: {len(seqs)} sequences")
    qdf = pd.DataFrame(labeled)
    qdf["seq"] = qdf["id"].map(seqs)
    n_missing = int(qdf["seq"].isna().sum())
    qdf = qdf.dropna(subset=["seq"]).reset_index(drop=True)
    print(f"      Dropped {n_missing} records without sequences, {len(qdf)} queries remaining")
    if len(qdf) == 0:
        raise ValueError("No query protein sequences available")
    query_csv = os.path.join(work, "query.csv")
    qdf[["id", "abundance", "seq", "exp", "label"]].to_csv(query_csv, index=False)

    matched_csv = os.path.join(work, "matched.csv")
    print(f"[4/5] Matching CDS (similarity>={args.threshold}%, coverage>={args.coverage}%) ...")
    subprocess.run(
        [sys.executable, GET_CDS, query_csv, args.cds, matched_csv,
         "--threshold", str(args.threshold),
         "--coverage", str(args.coverage),
         "--parallel", str(args.parallel)],
        check=True)
    n_matched = max(0, sum(1 for _ in open(matched_csv)) - 1)
    print(f"      Matched {n_matched} sequences")

    print("[5/5] cd-hit 0.9 de-redundancy ...")
    subprocess.run(["bash", CDHIT, "-d", ",", matched_csv, args.out], check=True)
    n_final = max(0, sum(1 for _ in open(args.out)) - 1)
    print(f"Done: {args.out} ({n_final} sequences)")
    print(f"Intermediate files: {query_csv}, {matched_csv}")


if __name__ == "__main__":
    main()
