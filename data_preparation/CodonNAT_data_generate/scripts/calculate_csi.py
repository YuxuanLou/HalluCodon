#!/usr/bin/env python3
"""Compute the CSI (codon stability index) of each CDS and select the top N sequences.

CSI = exp( mean( log(w_codon) ) ), where w_codon is the relative adaptiveness of the codon
within its amino acid (the most frequent codon of that amino acid is normalized to 1).

Usage:
  python3 calculate_csi.py --input unique.tsv --codon codon.csv \
      --output top10.csv --top-n 10
  or by percentage: --percent 10
"""

import argparse
import csv
import math
import sys

MAX_CHAR_LIMIT = 131072  # Skip sequences longer than this


def load_weights(codon_csv_path):
    aa_to_codons = {}
    with open(codon_csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            aa = row["aa"]
            codon = row["codon"].replace("U", "T").upper()
            count = float(row["count"])
            aa_to_codons.setdefault(aa, []).append((codon, count))
    w = {}
    for aa, codons in aa_to_codons.items():
        max_count = max(c[1] for c in codons)
        for codon, count in codons:
            w[codon] = count / max_count if max_count > 0 else 0.0
    return w


def csi_of(cds, w):
    s = 0.0
    n = 0
    cds = cds.upper()
    for i in range(0, len(cds) - 2, 3):
        wi = w.get(cds[i : i + 3], 0.0)
        if wi > 0:
            s += math.log(wi)
            n += 1
    return math.exp(s / n) if n else 0.0


def percentile(sorted_vals, p):
    """Linear-interpolation implementation equivalent to numpy.percentile."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = p / 100.0 * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="unique.tsv")
    ap.add_argument("--codon", required=True, help="codon.csv")
    ap.add_argument("--output", required=True, help="Output CSV")
    ap.add_argument("--top-n", type=int, default=None, help="Take the top N sequences (disabled by default)")
    ap.add_argument("--percent", type=float, default=10.0, help="Take the top percentage (default 10%%)")
    args = ap.parse_args()

    maxInt = sys.maxsize
    while True:
        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt / 10)

    print("[calculate_csi] Loading codon weights ...", flush=True)
    w = load_weights(args.codon)

    print("[calculate_csi] Pass 1: computing CSI for all sequences ...", flush=True)
    csi_list = []
    valid = []
    with open(args.input, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            cds = row.get("cds_sequence", "")
            if not cds or len(cds) > MAX_CHAR_LIMIT:
                csi_list.append(-1.0)
                continue
            c = csi_of(cds, w)
            csi_list.append(c)
            valid.append(c)

    if not valid:
        print("[calculate_csi] Error: no valid sequences", file=sys.stderr)
        sys.exit(1)

    if args.top_n is not None:
        valid_sorted = sorted(valid, reverse=True)
        threshold = valid_sorted[min(args.top_n - 1, len(valid_sorted) - 1)]
        print(f"[calculate_csi] Taking top {args.top_n} sequences, CSI threshold = {threshold:.6f}", flush=True)
    else:
        p = 100.0 - args.percent
        threshold = percentile(sorted(valid), p)  # percentile requires ascending order
        print(f"[calculate_csi] Taking top {args.percent}%, CSI threshold = {threshold:.6f}", flush=True)

    print("[calculate_csi] Pass 2: writing selected sequences ...", flush=True)
    kept = 0
    with open(args.input, newline="", encoding="utf-8") as fin, open(
        args.output, "w", newline="", encoding="utf-8"
    ) as fout:
        reader = csv.DictReader(fin, delimiter="\t")
        writer = csv.writer(fout)
        writer.writerow(["cds_sequence", "protein_sequence", "csi_value"])
        for i, row in enumerate(reader):
            if csi_list[i] >= threshold:
                writer.writerow(
                    [
                        row.get("cds_sequence", ""),
                        row.get("protein_sequence", ""),
                        f"{csi_list[i]:.6f}",
                    ]
                )
                kept += 1

    print(f"[calculate_csi] Done: wrote {kept} sequences -> {args.output}")


if __name__ == "__main__":
    main()
